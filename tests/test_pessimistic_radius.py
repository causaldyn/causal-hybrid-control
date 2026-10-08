"""The penalised descent takes the confounding radius' norm by a proximal step, not its gradient."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from scipy.optimize import minimize

from chc.control import SolverResult
from chc.cost import QuadraticCost, total_cost
from chc.dynamics import HybridDynamics, LinearDynamics
from chc.residual import ZeroResidual
from chc.support import SupportModel, pessimistic_solve
from chc.uncertainty import ConfoundingRobustPenalty

_KEY_X, _KEY_U = jax.random.split(jax.random.key(7))
_LOG_X = jax.random.normal(_KEY_X, (500, 2))


def _support(levers: int) -> SupportModel:
    return SupportModel.fit(_LOG_X, 0.4 * jax.random.normal(_KEY_U, (500, levers)))


def _units_problem(levers: int, dtype: jnp.dtype) -> tuple[SolverResult, np.ndarray]:
    """``tests/test_pessimistic_units.py``'s problem, one lever or two, under the radius 0.3 at a
    weight of 0.2, from a zero guess in ``dtype``; and the box's width."""
    a = jnp.array([[-0.5, 1.0], [0.0, -0.3]])
    if levers == 1:
        channel, weights = jnp.array([[0.0], [1.0]]), jnp.array([[0.05]])
        lo, hi = np.array([-5.0]), np.array([5.0])
    else:
        channel, weights = jnp.array([[0.0, 0.5], [1.0, 0.0]]), jnp.diag(jnp.array([0.05, 0.02]))
        lo, hi = np.array([-5.0, -0.3]), np.array([5.0, 0.3])
    model = HybridDynamics(known=LinearDynamics(a, channel), residual=ZeroResidual(2))
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.1])),
        R=weights,
        Qf=jnp.diag(jnp.array([5.0, 1.0])),
        x_target=jnp.zeros(2),
    )
    solve = pessimistic_solve(
        model,
        jnp.array([1.0, 0.0]),
        jnp.zeros((12, levers), dtype),
        0.1,
        cost,
        _support(levers),
        0.0,
        lo,
        hi,
        uncertainty=ConfoundingRobustPenalty(radius=0.3),
        lam_unc=0.2,
    )
    return solve, hi - lo


@pytest.mark.parametrize(("levers", "within"), [(1, 1e-6), (2, 1e-5)])
def test_a_float32_plan_under_the_radius_lands_on_the_float64_plan(
    levers: int, within: float
) -> None:
    """The radius' norm was smoothed over a millionth of the plan's size, and summed in the actions'
    dtype while the cost around it read them in float64. In float32 the descent stopped 1.0e-3 of
    the box from the float64 plan with one lever, after 69 steps against 89, and 1.9e-3 with two,
    after 55 against 102; under the support penalty the two plans are 2.5e-8 apart. With the norm
    taken by a proximal step and read in the cost's precision, one lever takes 45 steps in either
    dtype, to 1.4e-7 of the box, and two take 46 against 50, to 2.7e-6: there the float64 plan
    itself stops 1.8e-6 of the box from where the descent goes with ``tol = 0``, and the float32
    plan 8.3e-7."""
    wide, width = _units_problem(levers, jnp.float64)
    narrow, _ = _units_problem(levers, jnp.float32)
    assert wide.status == narrow.status == "converged"
    assert narrow.iterations <= 2 * wide.iterations
    gap = np.abs(np.asarray(narrow.actions, np.float64) - np.asarray(wide.actions)) / width
    assert float(gap.max()) <= within


def _oscillator(levers: int) -> tuple[LinearDynamics, QuadraticCost, Array]:
    """``test_plan``'s boxed oscillator, with a second lever on the position where asked."""
    channel = jnp.array([[0.0], [1.0]]) if levers == 1 else jnp.array([[0.0, 0.3], [1.0, 0.0]])
    plant = LinearDynamics(jnp.array([[0.0, 1.0], [-2.0, -0.3]]), channel)
    cost = QuadraticCost(
        Q=jnp.eye(2), R=0.05 * jnp.eye(levers), Qf=3.0 * jnp.eye(2), x_target=jnp.zeros(2)
    )
    return plant, cost, jnp.array([1.5, 0.0])


def _prox_residual(us: np.ndarray, gradient: np.ndarray, weight: float, box: float) -> float:
    """``||u - prox(u - g)||``, the prox of ``weight sum_t ||u_t||`` plus the box ``[-box, box]``:
    zero exactly at a stationary point of the unsmoothed problem. Each step's map is the group
    soft-threshold, ``y (1 - weight / ||y||)+``, clipped; this box is a cube about zero, and where
    the clip binds the test finds the map by minimising its objective."""
    landed = []
    for u, g in zip(us, gradient, strict=True):
        y = u - g
        shrunk = y * max(0.0, 1.0 - weight / max(float(np.linalg.norm(y)), 1e-300))
        if np.all(np.abs(shrunk) <= box):
            landed.append(shrunk)
            continue
        found = minimize(
            lambda x, y=y: 0.5 * np.sum((x - y) ** 2) + weight * np.sqrt(np.sum(x**2) + 1e-300),
            np.clip(shrunk, -box, box),
            bounds=[(-box, box)] * len(y),
            method="L-BFGS-B",
            options={"ftol": 1e-15, "gtol": 1e-12},
        )
        landed.append(found.x)
    return float(np.linalg.norm(us - np.asarray(landed)))


@pytest.mark.parametrize(("levers", "lam_unc"), [(1, 3.1), (2, 3.25)])
def test_a_zero_guess_on_the_radius_kink_moves_where_zero_is_not_stationary(
    levers: int, lam_unc: float
) -> None:
    """In the box ±0.2 every action of the zero guess sits on the norm's kink, where the smoothed
    norm's gradient is zero: the descent read the task's gradient alone, no step along it lowered
    the penalised cost, and it took none. Zero is not stationary: some actions' task gradient
    outweighs the radius' weight, so the prox of the norm and the box moves them. The residual it
    reported, 0.69 with one lever and 0.92 with two, was the smoothed gradient's. Now each moves,
    and converges where the unsmoothed problem's residual is zero."""
    plant, cost, x0 = _oscillator(levers)
    guess = jnp.zeros((12, levers))
    weight = lam_unc * 0.3

    def penalised(us: Array) -> float:
        norms = jnp.sqrt(jnp.sum(us**2, axis=-1))
        return float(total_cost(plant, x0, us, 0.1, cost) + weight * jnp.sum(norms))

    def task_gradient(us: Array) -> np.ndarray:
        return np.asarray(jax.grad(lambda u: total_cost(plant, x0, u, 0.1, cost))(us))

    assert _prox_residual(np.zeros((12, levers)), task_gradient(guess), weight, 0.2) > 0.05
    solve = pessimistic_solve(
        plant,
        x0,
        guess,
        0.1,
        cost,
        _support(levers),
        0.0,
        -0.2,
        0.2,
        uncertainty=ConfoundingRobustPenalty(radius=0.3),
        lam_unc=lam_unc,
    )
    assert solve.status == "converged"
    assert solve.iterations > 0
    assert penalised(solve.actions) < penalised(guess)
    residual = _prox_residual(np.asarray(solve.actions), task_gradient(solve.actions), weight, 0.2)
    assert residual <= 1e-6
    assert solve.stationarity <= 1e-6


def test_the_group_shrinkage_is_the_proximal_map_of_the_scaled_norm_and_the_box() -> None:
    """Several levers a step, the norm couples them, and in the descent's variables it is
    ``||v / sigma||``: its prox over a box has no closed form, and is found by bisection on its
    multiplier. On random instances, boxes that hold zero or not, it is the minimiser of the prox's
    objective, and exactly zero where the threshold outweighs the step."""
    from chc.control import _shrink

    rng, zeros = np.random.default_rng(0), 0
    for case in range(40):
        y = rng.normal(size=(1, 3)) * 2.0
        sigma = np.exp(rng.normal(size=(1, 3)))
        lo = -np.abs(rng.normal(size=(1, 3))) - 0.1
        hi = np.abs(rng.normal(size=(1, 3))) + 0.1
        if case % 4 == 0:  # a box that leaves zero out along the first lever
            lo[0, 0], hi[0, 0] = 0.5, 1.5
        threshold = np.abs(rng.normal()) * 2.0
        x = np.asarray(
            _shrink(
                jnp.asarray(y),
                jnp.asarray([threshold]),
                jnp.asarray(sigma),
                jnp.asarray(lo),
                jnp.asarray(hi),
            )
        )[0]

        def objective(
            z: np.ndarray, y: np.ndarray = y[0], sigma: np.ndarray = sigma[0], a: float = threshold
        ) -> float:
            return 0.5 * np.sum((z - y) ** 2) + a * np.linalg.norm(z / sigma)

        best = min(
            (
                minimize(
                    objective,
                    start,
                    bounds=list(zip(lo[0], hi[0], strict=True)),
                    method="Powell",
                    options={"xtol": 1e-12, "ftol": 1e-15},
                )
                for start in (np.clip(y[0], lo[0], hi[0]), np.clip(np.zeros(3), lo[0], hi[0]))
            ),
            key=lambda found: found.fun,
        )
        assert np.all((x >= lo[0]) & (x <= hi[0]))
        assert objective(x) <= best.fun + 1e-12
        if np.all(lo[0] <= 0.0) and np.linalg.norm(sigma[0] * y[0]) <= threshold:
            assert np.all(x == 0.0)
            zeros += 1
        else:
            assert np.any(x != 0.0)
    assert 0 < zeros < 40
