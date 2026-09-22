"""Linear constraints on the action sequence: every iterate is feasible, and the answer is optimal.

The oracle shares nothing with the solver. The plant is linear and the cost quadratic, so the
problem is a strictly convex QP in the actions (RK4 on a linear plant is linear in the actions). It
is solved exactly by a Cholesky change of variables and Lawson & Hanson's least-distance program --
one NNLS call, an active-set method with finite termination -- against the solver's projected
gradient with Dykstra's alternating projections.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from scipy.optimize import nnls

from chc import (
    DampedOscillator,
    LinearConstraint,
    QuadraticCost,
    SupportModel,
    causal_plan,
    pessimistic_solve,
    projected_gradient_control,
    projected_gradient_solve,
)
from chc.control import _constraint_blocks, _dykstra, _no_duals, _polish
from chc.cost import total_cost

DT = 0.1
HORIZON = 30
U_MAX = 5.0


def _least_distance(e: np.ndarray, f: np.ndarray) -> np.ndarray:
    """``argmin |w|`` subject to ``e @ w >= f`` (Lawson & Hanson 1974, ch. 23), by one NNLS."""
    n = e.shape[1]
    stacked = np.vstack([e.T, f[None, :]])
    target = np.zeros(n + 1)
    target[-1] = 1.0
    weights, _ = nnls(stacked, target, maxiter=100 * max(e.shape))
    residual = stacked @ weights - target
    assert abs(residual[-1]) > 1e-12, "the oracle's feasible set is empty"
    return -residual[:n] / residual[-1]


def _rows(lo, hi, matrix, lower, upper) -> tuple[np.ndarray, np.ndarray]:
    """The feasible set as ``G u <= h``, keeping only the sides that are present."""
    n = matrix.shape[1]
    has_upper, has_lower = np.isfinite(upper), np.isfinite(lower)
    g = np.vstack([matrix[has_upper], -matrix[has_lower], np.eye(n), -np.eye(n)])
    h = np.concatenate([upper[has_upper], -lower[has_lower], hi, -lo])
    return g, h


def _exact_projection(y, lo, hi, matrix, lower, upper) -> np.ndarray:
    g, h = _rows(lo, hi, matrix, lower, upper)
    return y + _least_distance(-g, g @ y - h)


def _exact_optimum(hessian, gradient, lo, hi, matrix, lower, upper) -> np.ndarray:
    g, h = _rows(lo, hi, matrix, lower, upper)
    free = -np.linalg.solve(hessian, gradient)
    back = np.linalg.inv(np.linalg.cholesky(hessian)).T  # u = free + back @ w, |w|^2 = 2 J + c
    return free + back @ _least_distance(-g @ back, g @ free - h)


def _random_polytope(rng: np.random.Generator, rows: int, size: int):
    """Random slabs, some one-sided, around a point inside the unit box; and a point to project."""
    matrix = rng.normal(size=(rows, size))
    level = matrix @ rng.uniform(-0.5, 0.5, size)
    lower = np.where(rng.random(rows) < 0.3, -np.inf, level - rng.uniform(0.05, 1.0, rows))
    upper = np.where(rng.random(rows) < 0.3, np.inf, level + rng.uniform(0.05, 1.0, rows))
    return matrix, lower, upper, -np.ones(size), np.ones(size), rng.normal(0.0, 3.0, size)


def _problem():
    dyn = DampedOscillator(omega=1.0, zeta=0.1)
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.0])),
        R=jnp.array([[0.01]]),
        Qf=jnp.diag(jnp.array([10.0, 1.0])),
        x_target=jnp.zeros(2),
    )
    x0 = jnp.array([1.0, 0.0])

    def objective(flat):
        return total_cost(dyn, x0, flat.reshape(HORIZON, 1), DT, cost)

    zero = jnp.zeros(HORIZON)
    hessian = np.asarray(jax.hessian(objective)(zero))
    gradient = np.asarray(jax.grad(objective)(zero))
    return dyn, cost, x0, objective, 0.5 * (hessian + hessian.T), gradient


def _binding_constraints(hessian, gradient):
    """A rate band and a budget, both sized off the box-only optimum so that both bind."""
    lo, hi = np.full(HORIZON, -U_MAX), np.full(HORIZON, U_MAX)
    none = np.zeros((0, HORIZON))
    box_only = _exact_optimum(hessian, gradient, lo, hi, none, np.zeros(0), np.zeros(0))
    rate = LinearConstraint.rate_limit(HORIZON, [0.35 * float(np.max(np.abs(np.diff(box_only))))])
    # The box optimum pushes the position back with negative force, so the budget caps how much
    # of it the whole plan may spend.
    budget = LinearConstraint(np.ones((1, HORIZON)), 0.6 * float(box_only.sum()), np.inf)
    return box_only, (rate, budget), lo, hi


def _stacked(constraints):
    return (
        np.vstack([c.matrix for c in constraints]),
        np.concatenate([c.lower for c in constraints]),
        np.concatenate([c.upper for c in constraints]),
    )


def _violation(u, constraints) -> float:
    matrix, lower, upper = _stacked(constraints)
    level = matrix @ np.asarray(u).ravel()
    return float(np.max(np.maximum(lower - level, level - upper)))


def test_the_constrained_solve_reaches_the_exact_optimum() -> None:
    dyn, cost, x0, objective, hessian, gradient = _problem()
    box_only, constraints, lo, hi = _binding_constraints(hessian, gradient)
    assert _violation(box_only, constraints) > 0.1  # the box optimum breaks both, or this is idle

    exact = _exact_optimum(hessian, gradient, lo, hi, *_stacked(constraints))
    best = float(objective(jnp.asarray(exact)))
    result = projected_gradient_solve(
        dyn, x0, jnp.zeros((HORIZON, 1)), DT, cost, -U_MAX, U_MAX, constraints=constraints
    )

    assert result.status == "converged"
    assert (float(objective(result.actions.ravel())) - best) / best < 1e-6
    assert result.constraint_violation < 1e-9
    assert _violation(result.actions, constraints) == pytest.approx(
        result.constraint_violation, abs=1e-12
    )
    assert float(jnp.max(jnp.abs(result.actions))) <= U_MAX  # the box is exact, not approximate
    assert result.stationarity < 1e-3
    matrix, lower, upper = _stacked(constraints)
    active = np.minimum(matrix @ exact - lower, upper - matrix @ exact) < 1e-8
    assert active.sum() >= 3  # several rows bind at the optimum, not a lucky single one


def test_a_solve_stopped_by_its_budget_still_returns_a_feasible_plan() -> None:
    """The property the design was chosen for: an augmented Lagrangian is feasible only at the end.

    So a solve cut short by its step budget is an executable plan here, and would not be there.
    """
    dyn, cost, x0, _, hessian, gradient = _problem()
    _, constraints, _, _ = _binding_constraints(hessian, gradient)
    result = projected_gradient_solve(
        dyn, x0, jnp.zeros((HORIZON, 1)), DT, cost, -U_MAX, U_MAX, steps=3, constraints=constraints
    )
    assert result.status == "max_iterations"
    assert result.constraint_violation < 1e-9
    assert result.stationarity > 1e-2  # unfinished, and it says so


def test_the_initial_guess_is_projected_rather_than_trusted() -> None:
    dyn, cost, x0, _, hessian, gradient = _problem()
    _, constraints, _, _ = _binding_constraints(hessian, gradient)
    wild = jnp.asarray(np.where(np.arange(HORIZON) % 2, 4.0, -4.0)[:, None])  # breaks the band
    assert _violation(wild, constraints) > 1.0
    us, _ = projected_gradient_control(
        dyn, x0, wild, DT, cost, -U_MAX, U_MAX, steps=1, constraints=constraints
    )
    assert _violation(us, constraints) < 1e-9


def test_no_constraints_and_an_empty_one_are_the_box_solve_exactly() -> None:
    dyn, cost, x0, *_ = _problem()
    guess = jnp.zeros((HORIZON, 1))
    plain = projected_gradient_solve(dyn, x0, guess, DT, cost, -U_MAX, U_MAX, steps=200)
    empty = projected_gradient_solve(
        dyn,
        x0,
        guess,
        DT,
        cost,
        -U_MAX,
        U_MAX,
        steps=200,
        constraints=(LinearConstraint.rate_limit(HORIZON, [np.inf]),),
    )
    assert np.array_equal(np.asarray(plain.actions), np.asarray(empty.actions))
    assert plain.constraint_violation == 0.0


def test_the_pessimistic_solver_holds_the_same_rows() -> None:
    dyn, cost, x0, _, hessian, gradient = _problem()
    _, constraints, _, _ = _binding_constraints(hessian, gradient)
    rng = np.random.default_rng(0)
    support = SupportModel.fit(
        jnp.asarray(rng.normal(size=(200, 2))), jnp.asarray(rng.normal(size=(200, 1)))
    )
    result = pessimistic_solve(
        dyn,
        x0,
        jnp.zeros((HORIZON, 1)),
        DT,
        cost,
        support,
        0.01,
        -U_MAX,
        U_MAX,
        steps=400,
        constraints=constraints,
    )
    assert result.constraint_violation < 1e-9


def test_causal_plan_passes_the_rows_through() -> None:
    dyn, cost, x0, _, hessian, gradient = _problem()
    _, constraints, _, _ = _binding_constraints(hessian, gradient)
    plan = causal_plan(dyn, x0, cost, DT, HORIZON, -U_MAX, U_MAX, constraints=constraints)
    assert _violation(plan.actions, constraints) < 1e-9


def test_single_precision_settles_at_its_own_tolerance() -> None:
    dyn, cost, x0, _, hessian, gradient = _problem()
    _, constraints, _, _ = _binding_constraints(hessian, gradient)
    result = projected_gradient_solve(
        dyn,
        x0,
        jnp.zeros((HORIZON, 1), dtype=jnp.float32),
        DT,
        cost,
        -U_MAX,
        U_MAX,
        steps=300,
        constraints=constraints,
    )
    assert result.actions.dtype == jnp.float32
    assert result.constraint_violation < 1e-4


def test_rate_limit_builds_one_band_per_capped_lever() -> None:
    horizon, caps = 4, [0.5, np.inf, 0.0]
    constraint = LinearConstraint.rate_limit(horizon, caps)
    assert constraint.matrix.shape == (2 * (horizon - 1), horizon * 3)
    schedule = np.array([[0.0, 9.0, 1.0], [0.4, -9.0, 1.0], [0.9, 9.0, 1.0], [0.4, 0.0, 1.0]])
    level = constraint.matrix @ schedule.ravel()
    np.testing.assert_allclose(level, [0.4, 0.5, -0.5, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(constraint.upper, [0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
    assert LinearConstraint.rate_limit(horizon, [np.inf]).matrix.shape == (0, horizon)
    assert LinearConstraint.rate_limit(1, [0.5]).matrix.shape == (0, 1)


@pytest.mark.parametrize(
    ("matrix", "lower", "upper", "match"),
    [
        (np.array([[1.0, 0.0], [0.0, 0.0]]), -1.0, 1.0, "row 1 of the constraint matrix is zero"),
        (np.array([[1.0, 0.0]]), 2.0, 1.0, "row 0 has lower 2.0 above upper 1.0"),
        (np.array([[1.0, 0.0]]), np.nan, 1.0, "NaN"),
        (np.array([[np.inf, 0.0]]), -1.0, 1.0, "non-finite coefficient"),
        (np.array([[1.0, 0.0]]), [0.0, 0.0], 1.0, "one per row"),
    ],
)
def test_a_constraint_that_cannot_mean_anything_is_refused(matrix, lower, upper, match) -> None:
    with pytest.raises(ValueError, match=match):
        LinearConstraint(matrix, lower, upper)


def test_an_empty_feasible_set_is_refused_before_the_solve() -> None:
    dyn, cost, x0, *_ = _problem()
    impossible = LinearConstraint(np.ones((1, HORIZON)), 2 * HORIZON * U_MAX, np.inf)
    with pytest.raises(ValueError, match="no action sequence satisfies"):
        projected_gradient_solve(
            dyn, x0, jnp.zeros((HORIZON, 1)), DT, cost, -U_MAX, U_MAX, constraints=(impossible,)
        )
    too_narrow = LinearConstraint(np.ones((1, 7)), -1.0, 1.0)
    with pytest.raises(ValueError, match="flatten to 30"):
        projected_gradient_solve(
            dyn, x0, jnp.zeros((HORIZON, 1)), DT, cost, -U_MAX, U_MAX, constraints=(too_narrow,)
        )


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    shape=st.sampled_from([(3, 4), (6, 5), (9, 6), (12, 8)]),
)
# Vertices where a constraint Dykstra had not yet released made the polished set over-determined:
# plain Dykstra ran out its sweeps on each, 4.6e-4, 1.9e-3 and 2.7e-3 from the projection.
@example(seed=66, shape=(3, 4))
@example(seed=639, shape=(9, 6))
@example(seed=567, shape=(12, 8))
def test_the_projection_is_the_euclidean_one(seed: int, shape: tuple[int, int]) -> None:
    """The projection against the exact least-distance one, on random polytopes with an interior."""
    rows, size = shape
    matrix, lower, upper, lo, hi, y = _random_polytope(np.random.default_rng(seed), rows, size)
    blocks = _constraint_blocks(
        [LinearConstraint(matrix, lower, upper)], jnp.asarray(lo), jnp.asarray(hi), jnp.float64
    )
    projected, _ = _dykstra(
        jnp.asarray(y),
        jnp.asarray(lo),
        jnp.asarray(hi),
        blocks,
        _no_duals(size, blocks, jnp.float64),
    )
    exact = _exact_projection(y, lo, hi, matrix, lower, upper)
    assert np.max(np.abs(np.asarray(projected) - exact)) < 1e-9


@pytest.mark.parametrize("steps", [0, 2, 3])
def test_the_polish_vouches_only_for_the_projection(steps: int) -> None:
    """Wherever the active-set loop stops, a point it accepts is the projection.

    Random multipliers of admissible sign and a starved step budget stop the loop mid-way on
    purpose -- rows still held by multipliers they no longer bind, coordinates released half-way --
    which is where a check that trusted the method's own bookkeeping would vouch for a wrong point.
    """
    polish = jax.jit(_polish, static_argnames="steps")
    accepted = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        rows, size = ((3, 4), (6, 5), (9, 6))[seed % 3]
        matrix, lower, upper, lo, hi, y = _random_polytope(rng, rows, size)
        blocks = _constraint_blocks(
            [LinearConstraint(matrix, lower, upper)], jnp.asarray(lo), jnp.asarray(hi), jnp.float64
        )
        multipliers = []
        for block_rows, block_lower, block_upper, _ in blocks:
            nu = rng.normal(size=block_rows.shape[0]) * (rng.random(block_rows.shape[0]) < 0.6)
            nu = np.where(np.isinf(block_upper) & (nu > 0), 0.0, nu)  # no bound to push down from
            nu = np.where(np.isinf(block_lower) & (nu < 0), 0.0, nu)
            multipliers.append(jnp.asarray(nu))
        tolerance = 256 * np.finfo(np.float64).eps * (1 + np.max(np.abs(y)))
        kkt, x, _ = polish(
            jnp.asarray(y),
            jnp.asarray(lo),
            jnp.asarray(hi),
            blocks,
            (jnp.zeros(size), tuple(multipliers)),
            jnp.asarray(tolerance),
            steps=steps,
        )
        if kkt:
            accepted += 1
            exact = _exact_projection(y, lo, hi, matrix, lower, upper)
            assert np.max(np.abs(np.asarray(x) - exact)) < 1e-9, seed
    assert accepted > 0


@pytest.mark.parametrize("shift", [0.0, 2.5, -4.0])
def test_a_lever_capped_at_zero_is_held_at_its_clipped_mean(shift: float) -> None:
    """A zero rate cap makes every row an equality; the projection is then ``clip(mean(y))``."""
    rng = np.random.default_rng(7)
    y = shift + rng.normal(0.0, 1.5, HORIZON)
    lo, hi = -np.ones(HORIZON), np.ones(HORIZON)
    blocks = _constraint_blocks(
        [LinearConstraint.rate_limit(HORIZON, [0.0])], jnp.asarray(lo), jnp.asarray(hi), jnp.float64
    )
    projected, _ = _dykstra(
        jnp.asarray(y),
        jnp.asarray(lo),
        jnp.asarray(hi),
        blocks,
        _no_duals(HORIZON, blocks, jnp.float64),
    )
    np.testing.assert_allclose(projected, np.clip(y.mean(), -1.0, 1.0), rtol=0, atol=1e-10)
