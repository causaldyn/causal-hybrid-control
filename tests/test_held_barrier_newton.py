"""A held barrier's rounds solved by projected Newton (``inner="newton"``).

Each augmented-Lagrangian round minimises the plan's cost plus the condition's penalty over the
box. The diagonal descent steps in the round's diagonal Gauss-Newton metric; projected Newton
(Bertsekas 1982) steps on the dense matrix whose diagonal that metric is. On an epidemic whose
condition binds on most steps the diagonal descent ran out of its budget beyond its first-order
tolerance; the Newton rounds stop within it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import chc.plan as plan_module
from chc import BarrierConstraint, QuadraticCost, SupportModel, causal_plan
from chc.control import LinearConstraint
from chc.dynamics import DampedOscillator, LinearDynamics
from chc.integrate import rollout
from chc.research.epidemic import SIRDynamics
from chc.support import _augmented

SIR = SIRDynamics(beta=0.6, gamma=0.1)
SIR_X0 = jnp.array([0.99, 0.01])
SIR_COST = QuadraticCost(
    Q=jnp.zeros((2, 2)), R=jnp.array([[1.0]]), Qf=jnp.zeros((2, 2)), x_target=jnp.zeros(2)
)
SIR_HORIZON = 60

OSCILLATOR = DampedOscillator(omega=1.0, zeta=0.1)
X0 = jnp.array([1.0, 0.0])
COST = QuadraticCost(
    Q=jnp.diag(jnp.array([1.0, 0.0])),
    R=jnp.array([[0.01]]),
    Qf=jnp.diag(jnp.array([10.0, 1.0])),
    x_target=jnp.zeros(2),
)
DT, HORIZON, U_MAX = 0.1, 20, 5.0


# two forces on the oscillator's velocity, as in the barrier tests: the radius' norm of a
# two-lever action has curvature away from zero
TWO_FORCES = LinearDynamics(
    jnp.array([[0.0, 1.0], [-1.0, -0.2]]), jnp.array([[0.0, 0.0], [1.0, 0.5]])
)
TWO_COST = QuadraticCost(
    Q=COST.Q, R=jnp.diag(jnp.array([0.01, 0.02])), Qf=COST.Qf, x_target=COST.x_target
)
TWO_REACH = np.array([U_MAX, 1.0])


def _capacity(x: Array) -> Array:
    """At most a tenth of the population infected."""
    return 0.1 - x[1]


def _velocity_floor(x: Array) -> Array:
    return x[1] + 0.8


def _sir_plan(inner: plan_module.InnerSolve, alpha: float = 0.5) -> plan_module.CausalPlan:
    return causal_plan(
        SIR,
        SIR_X0,
        SIR_COST,
        1.0,
        SIR_HORIZON,
        0.0,
        0.9,
        barrier=BarrierConstraint(_capacity, alpha=alpha),
        inner=inner,
    )


def _floor_plan(inner: plan_module.InnerSolve, gamma: float = 1.0) -> plan_module.CausalPlan:
    return causal_plan(
        OSCILLATOR,
        X0,
        COST,
        DT,
        HORIZON,
        -U_MAX,
        U_MAX,
        barrier=BarrierConstraint(_velocity_floor, alpha=2.0, gamma=gamma),
        inner=inner,
    )


@pytest.mark.parametrize("alpha", [0.5, 0.1])
def test_newton_rounds_hold_the_epidemic_plan_within_its_first_order_tolerance(
    alpha: float,
) -> None:
    """At ``alpha = 0.5`` the diagonal rounds spent all 10 000 steps and stopped at stationarity
    1.7e-5 against a tolerance of 1.4e-6. The Newton rounds stop ``converged`` and ``within``,
    in fewer than a tenth of the steps, every step certified."""
    plan = _sir_plan("newton", alpha)

    assert plan.solver_status == "converged"
    assert plan.stationarity_status == "within"
    assert plan.violation_status == "within"
    assert plan.solver_iterations < 1_000
    assert plan.safety is not None
    assert plan.safety.certified_steps == SIR_HORIZON


def test_newton_rounds_land_where_the_diagonal_rounds_do_where_those_converge() -> None:
    """The oscillator under a velocity floor, which the diagonal rounds solve: both reach the same
    plan, to the stationarity tolerance's share of the box."""
    for gamma in (1.0, 2.0):
        diagonal, newton = _floor_plan("diagonal", gamma), _floor_plan("newton", gamma)

        assert diagonal.solver_status == newton.solver_status == "converged"
        assert newton.stationarity_status == "within"
        assert newton.safety is not None
        assert newton.safety.certified_steps == HORIZON
        gap = abs(newton.task_cost - diagonal.task_cost) / diagonal.task_cost
        assert gap <= 1e-6
        moved = float(jnp.max(jnp.abs(newton.actions - diagonal.actions)))
        assert moved <= 2 * U_MAX * newton.stationarity_tolerance


def _round(plan: plan_module.CausalPlan) -> tuple[plan_module._HeldBarrier, Array]:
    problem = plan._problem
    assert problem is not None
    assert problem.held is not None
    return problem.held, plan.actions


@pytest.mark.parametrize("levers", [1, 2])
def test_the_round_matrix_is_the_round_s_hessian_on_a_linear_plant(levers: int) -> None:
    """On a plant linear in the state and the action the cost is quadratic in the actions, so its
    Gauss-Newton matrix is its Hessian, and the matrix the Newton rounds factor first, which adds
    the penalty's Hessian, is the round objective's. With two levers at ``gamma = 2`` the
    condition reads the radius' norm of the action, whose curvature the Gauss-Newton matrix drops;
    with one lever at ``gamma = 1`` every held column is affine, and the two matrices are one."""
    if levers == 1:
        model, cost, reach, gamma = OSCILLATOR, COST, np.array([U_MAX]), 1.0
    else:
        model, cost, reach, gamma = TWO_FORCES, TWO_COST, TWO_REACH, 2.0
    plan = causal_plan(
        model,
        X0,
        cost,
        DT,
        HORIZON,
        -reach,
        reach,
        barrier=BarrierConstraint(_velocity_floor, alpha=2.0, gamma=gamma, u_max=tuple(reach)),
        inner="newton",
    )
    held, actions = _round(plan)

    def objective(us: Array) -> Array:
        return _augmented(model, X0, us, DT, cost, None, 0.0, held, 1.0)

    value, gradient, curved, matrix = plan_module._round_terms(model, X0, actions, DT, cost, held)
    hessian = jax.hessian(objective)(actions).reshape(actions.size, actions.size)

    assert plan.solver_status == "converged"
    assert np.asarray(held.multipliers).max() > 0.0  # a column is on, so the penalty has curvature
    assert float(value) == float(objective(actions))
    scale = float(jnp.max(jnp.abs(hessian)))
    eps = float(jnp.finfo(actions.dtype).eps)
    # at the plan the gradient is a cancellation of terms up to the curvature times an action
    rounding = 64 * eps * scale * float(np.max(reach))
    np.testing.assert_allclose(gradient, jax.grad(objective)(actions), rtol=0.0, atol=rounding)
    np.testing.assert_allclose(curved, hessian, rtol=0.0, atol=64 * eps * scale)
    gap = float(jnp.max(jnp.abs(matrix - hessian)))
    if levers == 1:
        assert gap <= 64 * eps * scale
    else:
        assert gap > 1e6 * 64 * eps * scale


@pytest.mark.parametrize("which", ["oscillator", "epidemic"])
def test_the_round_matrix_s_diagonal_is_the_diagonal_metric_s_square(which: str) -> None:
    """The Gauss-Newton matrix the Newton rounds fall back on and the diagonal metric the descent
    steps in are one curvature: the metric is the root of the matrix's diagonal."""
    if which == "oscillator":
        model, x0, cost, dt, box = OSCILLATOR, X0, COST, DT, (-U_MAX, U_MAX)
        plan = _floor_plan("newton")
    else:
        model, x0, cost, dt, box = SIR, SIR_X0, SIR_COST, 1.0, (0.0, 0.9)
        plan = _sir_plan("newton")
    held, actions = _round(plan)
    lo = jnp.full(actions.shape, box[0], actions.dtype)
    hi = jnp.full(actions.shape, box[1], actions.dtype)
    states = rollout(model, x0, actions, dt)[:-1]
    excess = np.asarray(held.shortfall(states, actions), np.float64) + float(held.backoff)

    _, _, _, matrix = plan_module._round_terms(model, x0, actions, dt, cost, held)
    metric = plan_module._round_metric(model, x0, actions, dt, cost, lo, hi, held, excess)

    assert int(np.sum(excess > 0.0) + np.sum(np.asarray(held.multipliers) > 0.0)) > 0
    np.testing.assert_allclose(
        np.diag(np.asarray(matrix)), np.asarray(metric).ravel() ** 2, rtol=1e-12, atol=0.0
    )


def test_a_newton_round_whose_matrix_does_not_factor_is_the_diagonal_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Where the free actions' block is not positive definite the round hands its actions to the
    diagonal descent, with the steps left: with no block that factors, the plan is the diagonal
    rounds' bit for bit."""

    def refuse(*_: object, **__: object) -> None:
        raise np.linalg.LinAlgError("not positive definite")

    diagonal = _floor_plan("diagonal")
    monkeypatch.setattr(plan_module, "cho_factor", refuse)
    handed = _floor_plan("newton")

    assert np.array_equal(np.asarray(handed.actions), np.asarray(diagonal.actions))
    assert handed.solver_iterations == diagonal.solver_iterations
    assert handed.solver_status == diagonal.solver_status


def test_without_a_barrier_the_inner_solve_changes_nothing() -> None:
    free = causal_plan(OSCILLATOR, X0, COST, DT, HORIZON, -U_MAX, U_MAX)
    asked = causal_plan(OSCILLATOR, X0, COST, DT, HORIZON, -U_MAX, U_MAX, inner="newton")

    assert np.array_equal(np.asarray(asked.actions), np.asarray(free.actions))
    assert asked.solver_iterations == free.solver_iterations


def test_the_default_inner_solve_is_the_diagonal_descent() -> None:
    default = causal_plan(
        OSCILLATOR,
        X0,
        COST,
        DT,
        HORIZON,
        -U_MAX,
        U_MAX,
        barrier=BarrierConstraint(_velocity_floor, alpha=2.0),
    )
    diagonal = _floor_plan("diagonal")

    assert np.array_equal(np.asarray(default.actions), np.asarray(diagonal.actions))
    assert default.solver_iterations == diagonal.solver_iterations


def test_newton_rounds_refuse_what_they_cannot_step_on() -> None:
    barrier = BarrierConstraint(_velocity_floor, alpha=2.0)
    common = (OSCILLATOR, X0, COST, DT, HORIZON, -U_MAX, U_MAX)
    with pytest.raises(ValueError, match="box alone"):
        causal_plan(
            *common,
            constraints=(LinearConstraint.rate_limit(HORIZON, [1.0]),),
            barrier=barrier,
            inner="newton",
        )
    rng = np.random.default_rng(0)
    support = SupportModel.fit(
        jnp.asarray(rng.normal(size=(40, 2))), jnp.asarray(rng.normal(size=(40, 1)))
    )
    with pytest.raises(ValueError, match="box alone"):
        causal_plan(*common, support=support, lam_supp=1.0, barrier=barrier, inner="newton")
    with pytest.raises(ValueError, match="past its limit"):
        causal_plan(
            OSCILLATOR,
            X0,
            COST,
            DT,
            plan_module._NEWTON_SIZE + 1,
            -U_MAX,
            U_MAX,
            barrier=barrier,
            inner="newton",
        )
    with pytest.raises(ValueError, match="not a round's solve"):
        causal_plan(*common, barrier=barrier, inner="lbfgs")  # type: ignore[arg-type]
