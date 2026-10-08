"""Optimal-control gate: projected gradient reduces cost and drives the state toward target."""

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chc import HybridDynamics, QuadraticCost, SupportModel
from chc.adjoint import control_gradient_adjoint
from chc.benchmark import SupportShiftTask, _BumpActuator
from chc.control import (
    box_stationarity,
    lbfgs_box_control,
    nlp_solver_certificate,
    projected_gradient_control,
    projected_gradient_solve,
)
from chc.cost import total_cost
from chc.dynamics import DampedOscillator, DrivenDynamics, LinearDynamics
from chc.games import fixed_point
from chc.integrate import rollout
from chc.residual import (
    ContractiveResidual,
    ControlAffineResidual,
    GraphResidual,
    KANResidual,
    LipschitzResidual,
    MLPResidual,
    PortHamiltonianResidual,
    SpectralResidual,
    ZeroResidual,
)
from chc.support import pessimistic_control, pessimistic_solve
from chc.uncertainty import ConfoundingRobustPenalty

DT = 0.1


def test_projected_gradient_reduces_cost_and_reaches_target() -> None:
    dyn = HybridDynamics(
        known=DampedOscillator(omega=1.0, zeta=0.1), residual=ZeroResidual(out_dim=2)
    )
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.0])),
        R=jnp.array([[0.01]]),
        Qf=jnp.diag(jnp.array([10.0, 1.0])),
        x_target=jnp.zeros(2),
    )
    x0 = jnp.array([1.0, 0.0])
    us0 = jnp.zeros((50, 1))

    us, history = projected_gradient_control(dyn, x0, us0, DT, cost, u_lo=-5.0, u_hi=5.0, steps=150)

    assert history[-1] < 0.7 * history[0]  # meaningful cost reduction
    assert bool((jnp.abs(us) <= 5.0 + 1e-6).all())  # respects box constraints
    xs = rollout(dyn, x0, us, DT)
    assert abs(float(xs[-1, 0])) < abs(float(x0[0]))  # ends closer to target position


def test_lbfgs_matches_the_projected_gradient_objective_and_beats_it() -> None:
    certificate = nlp_solver_certificate()
    assert certificate.ok
    assert certificate.box_respected
    # Comparative rather than absolute: an absolute residual threshold is a claim about the
    # working dtype, not about the solvers, and would fail under float32 alone.
    assert certificate.least_stationarity_ratio > 10.0
    # Both directions: the first-order budget is short where the problem is ill conditioned and
    # adequate where it is not, so the gap is a statement about conditioning, not about the plant.
    assert certificate.worst_relative_gap > 0.05
    assert certificate.best_relative_gap < 0.005


def test_lbfgs_box_control_preserves_the_caller_dtype() -> None:
    # SciPy is a float64 boundary; a float32 caller must not be silently promoted on the way back.
    dyn = HybridDynamics(
        known=DampedOscillator(omega=1.0, zeta=0.1), residual=ZeroResidual(out_dim=2)
    )
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.0])),
        R=jnp.array([[0.01]]),
        Qf=jnp.diag(jnp.array([10.0, 1.0])),
        x_target=jnp.zeros(2),
    )
    us0 = jnp.zeros((20, 1), dtype=jnp.float32)
    us, history = lbfgs_box_control(dyn, jnp.array([1.0, 0.0]), us0, DT, cost, -5.0, 5.0)
    assert us.dtype == us0.dtype
    assert float(history[-1]) < float(history[0])
    assert bool((jnp.abs(us) <= 5.0 + 1e-6).all())


def _one_lever_in_units(units: float, scale: float) -> tuple[HybridDynamics, QuadraticCost]:
    """``test_plan``'s one-lever problem with the lever in units ``units`` times its own and the
    cost ``scale`` times its own: the channel over the units, ``R`` over their square."""
    model = HybridDynamics(
        known=LinearDynamics(
            jnp.array([[-0.5, 1.0], [0.0, -0.3]]), jnp.array([[0.0], [1.0]]) / units
        ),
        residual=ZeroResidual(out_dim=2),
    )
    cost = QuadraticCost(
        Q=scale * jnp.diag(jnp.array([1.0, 0.1])),
        R=scale * jnp.array([[0.05]]) / units**2,
        Qf=scale * jnp.diag(jnp.array([5.0, 1.0])),
        x_target=jnp.zeros(2),
    )
    return model, cost


@pytest.mark.parametrize("guess", [0.0, 0.5])
@pytest.mark.parametrize(
    ("units", "scale"),
    [(units, 1.0) for units in (1e-6, 1e-3, 1e3, 1e6)]
    + [(1.0, scale) for scale in (1e-6, 1e-3, 1e3, 1e6)],
)
def test_lbfgs_reads_one_problem_the_same_in_any_units(
    units: float, scale: float, guess: float
) -> None:
    """L-BFGS-B stops where the projected gradient is under 1e-5, or where the objective falls by
    under 2.2e-9 of the larger of it and 1: in the caller's units, a gradient and a cost. From zero,
    with the lever in units 1e-6 or 1e6 times its own, or the cost 1e-6 times, it took no step, 0.27
    of the box from the plan; with the lever 1e3 times or the cost 1e-3 times it stopped 6.0e-3 of
    the box from it. In the planner's scaled variables each takes 8 iterations to the plan it
    reaches in the problem's own units, to 8.9e-17 of the box, from zero or from 0.5 in the lever's
    own units, and 9.3e-6 and 1.4e-5 of the box from the planner's."""
    x0, us0, width = jnp.array([1.0, 0.0]), jnp.full((12, 1), guess), 10.0
    model, cost = _one_lever_in_units(1.0, 1.0)
    own, _ = lbfgs_box_control(model, x0, us0, DT, cost, -5.0, 5.0)
    planned, _ = projected_gradient_control(model, x0, us0, DT, cost, -5.0, 5.0)
    model, cost = _one_lever_in_units(units, scale)
    us, _ = lbfgs_box_control(model, x0, us0 * units, DT, cost, -5.0 * units, 5.0 * units)
    assert float(jnp.abs(us / units - own).max()) <= 1e-9 * width
    assert float(jnp.abs(own - planned).max()) <= 1e-4 * width


@pytest.mark.parametrize("side", [0.46, 0.47])
def test_an_lbfgs_plan_on_a_side_of_its_box_is_read_back_on_it(side: float) -> None:
    """L-BFGS-B steps in ``v = sigma u`` as the planner does, and its answer is read back as the
    planner reads its own: ``side * sigma / sigma`` rounds one ulp inside the box for 0.46 and one
    ulp past it for 0.47, and the plan lands on the side either way."""
    cost = QuadraticCost(
        Q=jnp.zeros((1, 1)), R=jnp.array([[0.05]]), Qf=jnp.eye(1), x_target=jnp.array([10.0])
    )
    model = LinearDynamics(jnp.zeros((1, 1)), jnp.ones((1, 1)))
    us, _ = lbfgs_box_control(model, jnp.zeros(1), jnp.zeros((3, 1)), DT, cost, -side, side)
    assert np.asarray(us).tolist() == [[side], [side], [side]]


# --- the compiled descent must reproduce the loop it replaced, on every residual backend --------

_ULP_BUDGET = 500.0
_SCALE_ULPS = 8.0  # two routes to one number: at most 2 ULP apart on these backends


def _action_scale(
    dyn: object, x0: jnp.ndarray, us: jnp.ndarray, dt: float, cost: QuadraticCost
) -> jnp.ndarray:
    """Each action's ``sigma``, read off the rollout's whole Jacobian and the cost's matrices.

    The solver runs one backward pass over the steps' Jacobians instead, so the two routes agree to
    rounding, not to the bit.
    """
    level = abs(float(total_cost(dyn, x0, us, dt, cost)))
    paths = jax.jacfwd(lambda actions: rollout(dyn, x0, actions, dt))(us)  # (H + 1, n, H, m)
    curvature = (
        jnp.einsum("tikm,ij,tjkm->km", paths[:-1], cost.Q, paths[:-1])
        + jnp.diagonal(cost.R)
        + jnp.einsum("ikm,ij,jkm->km", paths[-1], cost.Qf, paths[-1])
    )
    return jnp.sqrt(curvature / level)


def _checked_scale(
    dyn: object, x0: jnp.ndarray, us: jnp.ndarray, dt: float, cost: QuadraticCost, bound: float
) -> jnp.ndarray:
    """The solver's own ``sigma`` for a box ``[-bound, bound]``, once it matches the other route."""
    from chc.control import _action_units

    sigma, _, _ = _action_units(
        dyn, x0, us, dt, cost, jnp.full(us.shape, -bound), jnp.full(us.shape, bound)
    )
    assert _ulp_gap(sigma, _action_scale(dyn, x0, us, dt, cost)) <= _SCALE_ULPS
    return sigma


def _naive_projected_gradient(
    dyn: object,
    x0: jnp.ndarray,
    us0: jnp.ndarray,
    dt: float,
    cost: QuadraticCost,
    u_lo: float,
    u_hi: float,
    steps: int,
    sigma: jnp.ndarray,
    lr0: float = 1.0,
    tol: float = 1e-14,
) -> tuple[jnp.ndarray, list[float]]:
    """The plain Python recursion, kept as the oracle the compiled solver is checked against.

    It is deliberately *not* imported from ``chc``: an oracle that shares the implementation under
    test cannot detect the implementation changing. It takes the actions' ``sigma`` as given, and
    :func:`_checked_scale` hands it the solver's once :func:`_action_scale` has matched it, since a
    last bit apart in ``sigma`` grows over the steps to hundreds of ULP of an action near zero.

    A trial counts where it lowers the cost by a third of the fall its gradient predicts, and by
    more than ``tol``; where no halving does, the first one that fell by more than ``tol``.
    """
    us = jnp.clip(us0, u_lo, u_hi)
    current = total_cost(dyn, x0, us, dt, cost)
    level = abs(float(current))
    lo, hi = u_lo * sigma, u_hi * sigma
    history = [float(current)]
    for _ in range(steps):
        grad = control_gradient_adjoint(dyn, x0, us, dt, cost) / sigma
        lr, chosen = lr0 / level, None
        for _ls in range(40):
            trial = jnp.clip(us * sigma - lr * grad, lo, hi)
            # Read back onto the box's sides exactly: ``trial / sigma`` can round off them.
            candidate = jnp.where(
                trial <= lo, u_lo, jnp.where(trial >= hi, u_hi, jnp.clip(trial / sigma, u_lo, u_hi))
            )
            candidate_cost = total_cost(dyn, x0, candidate, dt, cost)
            if candidate_cost < current - tol * level:
                predicted = float(jnp.sum(grad * (us * sigma - trial)))
                if float(current - candidate_cost) >= (1.0 / 3.0) * predicted:
                    chosen = candidate, candidate_cost
                    break
                chosen = (candidate, candidate_cost) if chosen is None else chosen
            lr *= 0.5
        if chosen is None:
            break
        us, current = chosen
        history.append(float(current))
    return us, history


def _ulp_gap(a: jnp.ndarray, b: jnp.ndarray) -> float:
    left, right = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    spacing = np.spacing(np.maximum(np.abs(left), np.abs(right)))
    return float(np.max(np.abs(left - right) / np.maximum(spacing, np.finfo(np.float64).tiny)))


def _backend(name: str, key: jnp.ndarray) -> tuple[object, int]:
    """``(residual, control_dim)`` for each backend; the state is 2-dimensional throughout."""
    if name == "zero":
        return ZeroResidual(out_dim=2), 1
    if name == "mlp":
        return MLPResidual(2, 1, 2, 16, 2, key=key), 1
    if name == "control_affine":
        k_drift, k_channel = jax.random.split(key)
        return (
            ControlAffineResidual(
                drift=0.05 * jax.random.normal(k_drift, (2, 3)),
                channel=0.05 * jax.random.normal(k_channel, (2, 1, 3)),
            ),
            1,
        )
    if name == "kan":
        return KANResidual(2, 1, 2, key=key), 1
    if name == "spectral":  # couples a periodic field to a co-located control, so control_dim = 2
        return SpectralResidual(2, 2, key=key), 2
    if name == "graph":
        return GraphResidual(jnp.array([[0.0, 1.0], [1.0, 0.0]]), 1, 1, key=key), 1
    if name == "port_hamiltonian":
        return PortHamiltonianResidual(2, 1, key=key), 1
    if name == "lipschitz":
        return LipschitzResidual(2, 1, 2, key=key), 1
    if name == "contractive":
        return ContractiveResidual(2, 1, key=key), 1
    raise AssertionError(name)


_BACKENDS = (
    "zero",
    "mlp",
    "control_affine",
    "kan",
    "spectral",
    "graph",
    "port_hamiltonian",
    "lipschitz",
    "contractive",
)


@pytest.mark.parametrize("name", _BACKENDS)
def test_compiled_descent_matches_the_python_recursion(name: str) -> None:
    residual, control_dim = _backend(name, jax.random.key(_BACKENDS.index(name)))
    dyn = HybridDynamics(known=DampedOscillator(omega=2.0, zeta=0.1), residual=residual)
    cost = QuadraticCost(
        Q=jnp.eye(2),
        R=0.05 * jnp.eye(control_dim),
        Qf=5.0 * jnp.eye(2),
        x_target=jnp.array([0.5, 0.0]),
    )
    x0, us0 = jnp.array([1.0, 0.0]), jnp.zeros((20, control_dim))

    sigma = _checked_scale(dyn, x0, us0, DT, cost, 2.0)
    us_ref, history_ref = _naive_projected_gradient(
        dyn, x0, us0, DT, cost, -2.0, 2.0, steps=60, sigma=sigma
    )
    us, history = projected_gradient_control(dyn, x0, us0, DT, cost, -2.0, 2.0, steps=60)

    assert len(history) == len(history_ref)  # the scan stops where the break would have
    assert _ulp_gap(us, us_ref) < _ULP_BUDGET
    assert _ulp_gap(history, jnp.asarray(history_ref)) < _ULP_BUDGET


def test_the_scale_reads_each_step_at_its_own_time() -> None:
    """A driver that moves over the plan pushes the learned residual's states around, so each
    step's Jacobian depends on the step's time as well as its state: the solver's ``sigma`` still
    matches the one read off the rollout's whole Jacobian."""
    plant = DrivenDynamics(
        HybridDynamics(
            known=DampedOscillator(omega=2.0, zeta=0.1),
            residual=MLPResidual(2, 1, 2, 16, 2, key=jax.random.key(5)),
        ),
        gain=jnp.array([[0.0], [1.0]]),
        levels=jnp.linspace(-3.0, 3.0, 21)[:, None],
        dt=DT,
    )
    cost = QuadraticCost(
        Q=jnp.eye(2), R=0.05 * jnp.eye(1), Qf=5.0 * jnp.eye(2), x_target=jnp.array([0.5, 0.0])
    )
    _checked_scale(plant, jnp.array([1.0, 0.0]), jnp.zeros((20, 1)), DT, cost, 2.0)


class _Equilibrium(eqx.Module):
    """``x' = z - x``, with ``z`` the fixed point of ``z = z / 2 + u``: ``x' = 2 u - x``, solved
    by :func:`chc.games.fixed_point`, whose rule is reverse-mode only."""

    def __call__(self, t: float | jnp.ndarray, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        z = fixed_point(lambda push, w: 0.5 * w + push, u, jnp.zeros_like(u), tol=1e-12).x
        return z - x


def test_a_plant_with_a_reverse_rule_only_still_plans() -> None:
    """The adjoint reads each step's Jacobian in reverse mode, and so does the scale. In forward
    mode a step with no forward rule, a ``jax.custom_vjp`` as in an equilibrium solve, raised
    ``TypeError`` where the descent in the caller's units converged in 253 steps. Read in reverse,
    it converges in 36, as the same plant written out does, to 2.4e-12 of the box."""
    cost = QuadraticCost(Q=jnp.eye(1), R=0.1 * jnp.eye(1), Qf=jnp.eye(1), x_target=jnp.ones(1))
    guess = jnp.zeros((8, 1))
    solved, written = (
        projected_gradient_solve(plant, jnp.zeros(1), guess, DT, cost, -2.0, 2.0)
        for plant in (_Equilibrium(), LinearDynamics(-jnp.eye(1), 2.0 * jnp.eye(1)))
    )
    assert solved.status == written.status == "converged"
    assert float(jnp.max(jnp.abs(solved.actions - written.actions))) <= 1e-9


def test_compiled_descent_stops_where_the_line_search_fails() -> None:
    # A heavy control penalty makes the descent converge in a handful of steps, so the history is
    # much shorter than the budget -- the case a fixed-length scan would get wrong if it padded.
    dyn = HybridDynamics(
        known=DampedOscillator(omega=2.0, zeta=0.1), residual=ZeroResidual(out_dim=2)
    )
    cost = QuadraticCost(
        Q=jnp.eye(2), R=5.0 * jnp.eye(1), Qf=5.0 * jnp.eye(2), x_target=jnp.array([0.5, 0.0])
    )
    x0, us0 = jnp.array([1.0, 0.0]), jnp.zeros((20, 1))

    sigma = _checked_scale(dyn, x0, us0, DT, cost, 2.0)
    us_ref, history_ref = _naive_projected_gradient(
        dyn, x0, us0, DT, cost, -2.0, 2.0, steps=400, sigma=sigma
    )
    us, history = projected_gradient_control(dyn, x0, us0, DT, cost, -2.0, 2.0, steps=400)

    assert len(history_ref) < 400  # the oracle really does break early
    assert len(history) == len(history_ref)
    assert _ulp_gap(us, us_ref) < _ULP_BUDGET


def test_the_compiled_solver_amortises_across_calls() -> None:
    # The regression this guards: building the jitted loop inside the caller gives every call a
    # fresh compilation cache, so the second solve costs the same as the first.
    # The odd horizon and step count keep this instance off every other test's compilation key, so
    # the first solve here is genuinely cold rather than warmed by an earlier test in the module.
    dyn = HybridDynamics(
        known=DampedOscillator(omega=1.7, zeta=0.1),
        residual=MLPResidual(2, 1, 2, 16, 2, key=jax.random.key(11)),
    )
    cost = QuadraticCost(
        Q=jnp.eye(2), R=0.05 * jnp.eye(1), Qf=5.0 * jnp.eye(2), x_target=jnp.array([0.5, 0.0])
    )
    x0, us0 = jnp.array([1.0, 0.0]), jnp.zeros((23, 1))

    def solve_seconds() -> float:
        start = time.perf_counter()
        jax.block_until_ready(projected_gradient_control(dyn, x0, us0, DT, cost, -2.0, 2.0, 37))
        return time.perf_counter() - start

    cold = solve_seconds()
    warm = min(solve_seconds(), solve_seconds())
    assert warm < 0.5 * cold


def _two_lever_problem() -> tuple[LinearDynamics, QuadraticCost, jnp.ndarray, jnp.ndarray]:
    """Two levers that both want to push hard, so a per-lever cap is visible in the answer."""
    dyn = LinearDynamics(a_matrix=jnp.zeros((2, 2)), b_matrix=jnp.eye(2))
    cost = QuadraticCost(
        Q=jnp.eye(2),
        R=jnp.diag(jnp.array([1e-3, 1e-3])),
        Qf=10.0 * jnp.eye(2),
        x_target=jnp.zeros(2),
    )
    return dyn, cost, jnp.array([1.0, 1.0]), jnp.zeros((20, 2))


def test_a_per_lever_box_is_the_same_answer_as_the_scalar_it_repeats() -> None:
    # The widened signature must not move any existing caller's numbers: a vector that spells out
    # the scalar has to reproduce it exactly, not merely closely.
    dyn, cost, x0, us0 = _two_lever_problem()
    scalar, _ = projected_gradient_control(dyn, x0, us0, DT, cost, -1.0, 1.0, steps=200)
    vector, _ = projected_gradient_control(
        dyn, x0, us0, DT, cost, jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0]), steps=200
    )
    assert bool((scalar == vector).all())


def test_a_tighter_lever_saturates_where_a_shared_box_would_not() -> None:
    dyn, cost, x0, us0 = _two_lever_problem()
    shared, _ = projected_gradient_control(dyn, x0, us0, DT, cost, -5.0, 5.0, steps=400)
    # The second lever is the constrained actuator; the first keeps the loose bound it had.
    per_lever, _ = projected_gradient_control(
        dyn, x0, us0, DT, cost, jnp.array([-5.0, -0.2]), jnp.array([5.0, 0.2]), steps=400
    )

    assert float(jnp.abs(shared[:, 1]).max()) > 0.2  # the shared box does not bind here
    assert float(jnp.abs(per_lever[:, 1]).max()) <= 0.2 + 1e-6  # the per-lever one does
    assert float(jnp.abs(per_lever[:, 1]).max()) == pytest.approx(0.2, abs=1e-6)  # and it saturates
    assert float(jnp.abs(per_lever[:, 0]).max()) > 0.2  # the free lever is untouched by it
    # Constraining one lever cannot improve the objective: the feasible set only shrank.
    assert float(total_cost(dyn, x0, per_lever, DT, cost)) >= float(
        total_cost(dyn, x0, shared, DT, cost)
    )


def test_a_per_lever_box_does_not_promote_the_caller_dtype() -> None:
    # The box is validated in float64 but must come back in the actions' precision: a float64
    # bound clipped against float32 actions promotes the answer and changes what was asked for.
    dyn, cost, x0, _ = _two_lever_problem()
    us0 = jnp.zeros((20, 2), dtype=jnp.float32)
    lo, hi = jnp.array([-5.0, -0.2]), jnp.array([5.0, 0.2])
    us, _ = projected_gradient_control(dyn, x0, us0, DT, cost, lo, hi, steps=50)
    assert us.dtype == us0.dtype
    lbfgs, _ = lbfgs_box_control(dyn, x0, us0, DT, cost, lo, hi, steps=50)
    assert lbfgs.dtype == us0.dtype


def test_lbfgs_honours_the_same_per_lever_box() -> None:
    # L-BFGS-B keeps one bound pair per coordinate, so the per-lever box has to survive the ravel.
    dyn, cost, x0, us0 = _two_lever_problem()
    lo, hi = jnp.array([-5.0, -0.2]), jnp.array([5.0, 0.2])
    us, _ = lbfgs_box_control(dyn, x0, us0, DT, cost, lo, hi, steps=200)
    assert float(jnp.abs(us[:, 1]).max()) <= 0.2 + 1e-8
    assert float(jnp.abs(us[:, 0]).max()) > 0.2

    reference, _ = projected_gradient_control(dyn, x0, us0, DT, cost, lo, hi, steps=2000)
    assert float(total_cost(dyn, x0, us, DT, cost)) == pytest.approx(
        float(total_cost(dyn, x0, reference, DT, cost)), rel=1e-3
    )


def test_an_ambiguous_or_empty_box_is_rejected_rather_than_broadcast() -> None:
    dyn, cost, x0, us0 = _two_lever_problem()

    # (horizon,) would broadcast along the lever axis and silently constrain the wrong thing.
    with pytest.raises(ValueError, match="per-lever"):
        projected_gradient_control(dyn, x0, us0, DT, cost, jnp.zeros(20) - 1.0, 1.0)

    with pytest.raises(ValueError, match="expected a scalar"):
        projected_gradient_control(dyn, x0, us0, DT, cost, -1.0, jnp.ones((3, 4)))

    # jnp.clip with lo > hi returns hi everywhere without complaint -- a wrong answer, not an error.
    with pytest.raises(ValueError, match="empty action box"):
        projected_gradient_control(dyn, x0, us0, DT, cost, jnp.array([-1.0, 0.5]), 0.1)

    # a nan bound compared false either way, and the clip returned nan for every action
    with pytest.raises(ValueError, match="has a nan bound"):
        projected_gradient_control(dyn, x0, us0, DT, cost, jnp.array([-1.0, jnp.nan]), 1.0)


def test_pessimistic_control_takes_a_per_lever_box_too() -> None:
    dyn, cost, x0, us0 = _two_lever_problem()
    key = jax.random.key(0)
    logged_x = jax.random.normal(key, (256, 2))
    logged_u = 0.5 * jax.random.normal(jax.random.key(1), (256, 2))
    support = SupportModel.fit(logged_x, logged_u)
    us, _ = pessimistic_control(
        dyn,
        x0,
        us0,
        DT,
        cost,
        support,
        0.1,
        jnp.array([-5.0, -0.2]),
        jnp.array([5.0, 0.2]),
        steps=300,
    )
    assert float(jnp.abs(us[:, 1]).max()) <= 0.2 + 1e-6
    assert float(jnp.abs(us[:, 0]).max()) > 0.2


def test_the_solver_reports_why_it_stopped_not_only_where() -> None:
    dyn, cost, x0, us0 = _two_lever_problem()

    # No budget at all: the answer IS the caller's guess, and saying "converged" would be a lie.
    none = projected_gradient_solve(dyn, x0, us0, DT, cost, -5.0, 5.0, steps=0)
    assert none.status == "no_progress"
    assert none.iterations == 0
    assert bool((none.actions == us0).all())

    # Budget exhausted: the descent is wherever it happened to be, not at its stopping rule.
    short = projected_gradient_solve(dyn, x0, us0, DT, cost, -5.0, 5.0, steps=3)
    assert short.status == "max_iterations"
    assert short.iterations == 3

    # This instance needs 3 801 steps under float64. At 3 000 the residual is already down to
    # 6e-6 and the answer looks finished -- and it is not. That gap is the whole reason the status
    # exists: a small residual is not evidence that the solver reached its stopping rule.
    truncated = projected_gradient_solve(dyn, x0, us0, DT, cost, -5.0, 5.0, steps=3000)
    assert truncated.status == "max_iterations"
    assert truncated.stationarity < 1e-3

    # The line search stalls first: this is the method's own stopping rule.
    full = projected_gradient_solve(dyn, x0, us0, DT, cost, -5.0, 5.0, steps=20_000)
    assert full.status == "converged"
    assert 0 < full.iterations < 20_000
    assert full.stationarity < truncated.stationarity

    # The status is a claim about steps; the residual is what makes it checkable, and it has to
    # separate the coarse cases by orders of magnitude or the label is decoration.
    assert full.stationarity < 0.01 * short.stationarity
    assert short.stationarity < none.stationarity


def test_the_rich_solve_and_the_compact_one_are_the_same_solve() -> None:
    # The tuple-returning function is a wrapper, not a second implementation: same numbers.
    dyn, cost, x0, us0 = _two_lever_problem()
    result = projected_gradient_solve(dyn, x0, us0, DT, cost, -1.0, 1.0, steps=400)
    actions, history = projected_gradient_control(dyn, x0, us0, DT, cost, -1.0, 1.0, steps=400)
    assert bool((result.actions == actions).all())
    assert bool((result.cost_history == history).all())
    assert result.iterations == len(history) - 1


def test_pessimistic_stationarity_is_measured_on_what_was_minimised() -> None:
    # The descent minimises task + penalties, so a residual on the task alone would be non-zero
    # exactly where the solver was right to stop -- and would read as a failure.
    dyn, cost, x0, us0 = _two_lever_problem()
    logged_x = jax.random.normal(jax.random.key(0), (256, 2))
    logged_u = 0.5 * jax.random.normal(jax.random.key(1), (256, 2))
    support = SupportModel.fit(logged_x, logged_u)

    result = pessimistic_solve(dyn, x0, us0, DT, cost, support, 1.0, -5.0, 5.0, steps=5000)
    assert result.status == "converged"
    task_only = box_stationarity(dyn, x0, result.actions, DT, cost, -5.0, 5.0)
    assert result.stationarity < 0.1 * task_only


def test_a_plant_whose_effect_collapses_is_planned_where_it_still_answers() -> None:
    """``SupportShiftTask``'s oracle: its plant acts through ``u exp(-(u / 0.8)^2)``, linear at zero
    and spent beyond about 2. The first step from zero is the Newton step of the plant as it acts
    there; halved twice, it sets actions up to 6.2 and still lowers the cost, by 0.4 % of the fall
    the gradient predicts for it. A search that took any fall took it, and the descent settled with
    five actions near -2.2, where the plant no longer answers them, at a cost of 21.104508. A step
    now counts only where it buys a third of that prediction: every action ends at or inside the
    effect's peak, ``0.8 / sqrt(2)``, at 20.645063, the plan the descent in the caller's units
    found."""
    task = SupportShiftTask()
    plant = _BumpActuator(
        a_matrix=jnp.array([[0.0, 1.0], [-1.0, -0.2]]),
        b_matrix=jnp.array([[0.0], [1.0]]),
        u_sat=task.u_sat,
    )
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.0])),
        R=jnp.array([[task.control_weight]]),
        Qf=jnp.diag(jnp.array([10.0, 1.0])),
        x_target=jnp.array([task.x_target, 0.0]),
    )
    x0, us0 = jnp.array([task.x0, 0.0]), jnp.zeros((task.horizon, 1))
    solve = projected_gradient_solve(plant, x0, us0, task.dt, cost, task.u_lo, task.u_hi)
    assert solve.status == "converged"
    assert float(jnp.abs(solve.actions).max()) <= task.u_sat / np.sqrt(2.0)
    assert float(total_cost(plant, x0, solve.actions, task.dt, cost)) == pytest.approx(
        20.645063, rel=0.0, abs=1e-6
    )


def test_a_quadratic_takes_the_newton_step_along_each_action_whole() -> None:
    """Two levers in units a thousand times apart, each driving its own state over one step: the
    cost is a quadratic with a diagonal Hessian, so the Newton step along each action alone is its
    minimiser, and that step lowers the cost by half the fall the gradient predicts for it. The line
    search takes it whole, so one step lands on the closed form."""
    gain, weight, terminal = jnp.array([1.0, 1e-3]), jnp.array([0.05, 2e-8]), jnp.array([5.0, 1.0])
    x0 = jnp.array([1.0, -2.0])
    dyn = LinearDynamics(jnp.zeros((2, 2)), jnp.diag(gain))
    cost = QuadraticCost(
        Q=jnp.zeros((2, 2)), R=jnp.diag(weight), Qf=jnp.diag(terminal), x_target=jnp.zeros(2)
    )
    minimiser = -terminal * DT * gain * x0 / (weight + terminal * DT**2 * gain**2)
    one = projected_gradient_solve(dyn, x0, jnp.zeros((1, 2)), DT, cost, -1e5, 1e5, steps=1)
    assert one.iterations == 1
    np.testing.assert_allclose(one.actions[0], minimiser, rtol=1e-12, atol=0.0)


def test_a_penalised_plan_that_starts_on_the_radius_kink_still_descends() -> None:
    """The confounding radius' norm has a kink at a plan of zeros. From there, on a damped
    oscillator with a learned residual, its lever boxed at ±0.5 and the radius charging a tenth of
    the zero plan's cost on the box's edge, a trial buys 0.034 of the fall the gradient predicts
    for it however short the step, so no halving buys a third. The search then takes its longest
    trial that fell, which moves the lever 0.119, the step a search that took any fall took, and
    the descent converges from there. A search that refused every such trial would stop at the
    zero guess, and one that took the shortest would move the lever 7e-12."""
    dyn = HybridDynamics(
        DampedOscillator(1.0, 0.2), MLPResidual(2, 1, 2, key=jax.random.PRNGKey(3))
    )
    cost = QuadraticCost(
        Q=jnp.eye(2), R=0.05 * jnp.eye(1), Qf=3.0 * jnp.eye(2), x_target=jnp.zeros(2)
    )
    x0, us0 = jnp.array([1.5, 0.0]), jnp.zeros((12, 1))
    start = float(total_cost(dyn, x0, us0, DT, cost))
    support = SupportModel.fit(
        jax.random.normal(jax.random.key(2), (400, 2)),
        0.25 * jax.random.normal(jax.random.key(1), (400, 1)),
    )
    radius = ConfoundingRobustPenalty(radius=0.3)
    weight = 0.1 * start / (0.3 * 12 * 0.5)
    first = pessimistic_solve(
        dyn, x0, us0, DT, cost, support, 0.0, -0.5, 0.5, steps=1, uncertainty=radius, lam_unc=weight
    )
    assert first.iterations == 1
    assert float(jnp.abs(first.actions).max()) > 0.1
    solve = pessimistic_solve(
        dyn, x0, us0, DT, cost, support, 0.0, -0.5, 0.5, uncertainty=radius, lam_unc=weight
    )
    assert solve.status == "converged"
    assert float(total_cost(dyn, x0, solve.actions, DT, cost)) < start - 0.5
