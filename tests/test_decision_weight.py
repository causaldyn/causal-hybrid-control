"""What an error in the one-step channel costs a plan: the Hessian of its certainty-equivalent
regret, read off the plan's own problem.

The scalar cases have closed forms from ``validation/plan_decision_weight.mac``. Everything else is
checked against the regret itself: the perturbed problem re-solved by SLSQP, and the plan's loss
against it compared with ``vec(E)' W vec(E) / 2`` as ``E`` shrinks.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import minimize

from chc import BarrierConstraint, CausalPlan, LinearConstraint, QuadraticCost, causal_plan
from chc.dynamics import LinearDynamics
from chc.plan import _perturbed_task_cost
from chc.support import SupportModel

DT = 0.1
SCALAR = LinearDynamics(jnp.array([[-0.5]]), jnp.array([[1.2]]))
SCALAR_COST = QuadraticCost(
    Q=jnp.eye(1), R=jnp.array([[0.1]]), Qf=jnp.array([[2.0]]), x_target=jnp.array([0.0])
)
ONE = jnp.array([1.0])


def _rk4(a: float, b: float, dt: float) -> tuple[float, float]:
    """``x1 = phi x0 + gam u``: the RK4 step of ``xdot = a x + b u`` (the .mac file's STEP 1)."""
    h = a * dt
    return 1 + h + h**2 / 2 + h**3 / 6 + h**4 / 24, b * dt * (1 + h / 2 + h**2 / 6 + h**3 / 24)


def _at(plan: CausalPlan, actions: np.ndarray) -> CausalPlan:
    """The same problem, weighed at ``actions``: the closed form's optimum, not the solver's."""
    return replace(plan, actions=jnp.asarray(actions, dtype=plan.actions.dtype))


@pytest.mark.parametrize(
    ("horizon", "u_lo", "optimum", "weight", "free"),
    [
        (1, -10.0, [-1.747878144003991], 9.225094387724484, 1),
        (2, -10.0, [-2.023822333873833, -1.248583272170075], 4.706933453397004, 2),
        (2, [[-0.4], [-10.0]], [-0.4, -1.580797907799666], 6.23700167606525, 1),
    ],
    ids=["one step", "two steps", "the first action on its bound"],
)
def test_a_scalar_plan_weighs_its_channel_as_maxima_derives(
    horizon: int, u_lo: float | list[list[float]], optimum: list[float], weight: float, free: int
) -> None:
    plan = causal_plan(SCALAR, ONE, SCALAR_COST, DT, horizon, jnp.asarray(u_lo), 10.0)
    np.testing.assert_allclose(np.asarray(plan.actions).ravel(), optimum, rtol=1e-3)
    exact = _at(plan, np.array(optimum)[:, None]).decision_weight()
    assert exact.matrix.shape == (1, 1)
    assert exact.matrix[0, 0] == pytest.approx(weight, rel=1e-10)
    assert (exact.free, exact.weakly_active, exact.channel_shape) == (free, 0, (1, 1))
    assert exact.residual < 1e-12
    # at the solver's own plan, as far as the solve went
    solved = plan.decision_weight()
    assert solved.matrix[0, 0] == pytest.approx(weight, rel=1e-3)
    assert solved.residual < 1e-3


@settings(max_examples=25, deadline=None)
@given(
    a=st.floats(-2.0, 1.0),
    b=st.floats(0.2, 3.0),
    x0=st.floats(-2.0, 2.0),
    target=st.floats(-2.0, 2.0),
    r=st.floats(0.01, 1.0),
    qf=st.floats(0.1, 5.0),
)
def test_a_one_step_weight_is_maxima_s_closed_form(
    a: float, b: float, x0: float, target: float, r: float, qf: float
) -> None:
    """``W = qf^2 (r - gam^2 qf)^2 (phi x0 - s)^2 / (r + gam^2 qf)^3`` (the .mac file's STEP 2d),
    at the optimum ``u* = -gam qf (phi x0 - s) / (r + gam^2 qf)`` (STEP 2a)."""
    phi, gam = _rk4(a, b, DT)
    cost = QuadraticCost(
        Q=jnp.eye(1), R=jnp.array([[r]]), Qf=jnp.array([[qf]]), x_target=jnp.array([target])
    )
    model = LinearDynamics(jnp.array([[a]]), jnp.array([[b]]))
    plan = causal_plan(model, jnp.array([x0]), cost, DT, 1, -1e3, 1e3, steps=1)
    optimum = -gam * qf * (phi * x0 - target) / (r + gam**2 * qf)
    weight = qf**2 * (r - gam**2 * qf) ** 2 * (phi * x0 - target) ** 2 / (r + gam**2 * qf) ** 3
    got = _at(plan, np.array([[optimum]])).decision_weight().matrix[0, 0]
    assert got == pytest.approx(weight, rel=1e-8, abs=1e-12)


# ---- the regret itself, on a plant with two states, two levers, a box and a budget ----

PLANT = LinearDynamics(jnp.array([[-0.3, 0.2], [0.1, -0.5]]), jnp.array([[1.0, 0.3], [-0.2, 0.8]]))
PLANT_COST = QuadraticCost(
    Q=jnp.diag(jnp.array([1.0, 0.5])),
    R=jnp.diag(jnp.array([0.2, 0.1])),
    Qf=jnp.diag(jnp.array([2.0, 1.0])),
    x_target=jnp.array([1.5, -1.0]),
)
START = jnp.array([0.2, 0.4])
HORIZON = 5
BOX = 1.5
BUDGET = LinearConstraint(np.tile([1.0, 0.0], HORIZON)[None, :], -np.inf, 4.0)


def _solve(plan: CausalPlan, change: np.ndarray, start: np.ndarray) -> np.ndarray:
    """The perturbed problem, under the plan's box and budget, by SLSQP from ``start``."""
    problem, shape = plan._problem, plan.actions.shape
    assert problem is not None

    def value(u: jax.Array) -> jax.Array:
        return _perturbed_task_cost(problem, u.reshape(shape), jnp.asarray(change))

    gradient = jax.grad(value)
    result = minimize(
        lambda u: float(value(jnp.asarray(u))),
        start.ravel(),
        jac=lambda u: np.asarray(gradient(jnp.asarray(u))),
        bounds=[(-BOX, BOX)] * start.size,
        constraints=[
            {
                "type": "ineq",
                "fun": lambda u: BUDGET.upper - BUDGET.matrix @ u,
                "jac": lambda u: -BUDGET.matrix,
            }
        ],
        method="SLSQP",
        options={"ftol": 1e-15, "maxiter": 2000},
    )
    assert result.success, result.message
    return result.x.reshape(shape)


def _regret(plan: CausalPlan, optimum: np.ndarray, change: np.ndarray) -> float:
    """What the plan at ``optimum`` loses on the perturbed plant to the plan that knew it."""
    problem = plan._problem
    assert problem is not None
    knew = _solve(plan, change, optimum)

    def value(u: np.ndarray) -> float:
        return float(_perturbed_task_cost(problem, jnp.asarray(u), jnp.asarray(change)))

    return value(optimum) - value(knew)


@pytest.fixture(scope="module")
def budgeted() -> tuple[CausalPlan, np.ndarray]:
    plan = causal_plan(
        PLANT, START, PLANT_COST, 0.2, HORIZON, -BOX, BOX, constraints=(BUDGET,), steps=20_000
    )
    optimum = _solve(plan, np.zeros((2, 2)), np.asarray(plan.actions))
    return _at(plan, optimum), optimum


def test_the_budgeted_plan_binds_its_box_and_its_budget(
    budgeted: tuple[CausalPlan, np.ndarray],
) -> None:
    """The case the next test needs: some actions on the box and the budget binding, strictly."""
    plan, optimum = budgeted
    assert np.sum(np.isclose(np.abs(optimum), BOX, atol=1e-9)) >= 1
    assert BUDGET.matrix @ optimum.ravel() == pytest.approx(BUDGET.upper[0], abs=1e-9)
    [row] = plan.shadow_prices().rows
    assert row.status == "exact"
    assert row.price is not None
    assert row.price > 1e-2
    weight = plan.decision_weight()
    assert weight.weakly_active == 0
    assert 0 < weight.free < optimum.size - 1


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_regret_of_a_small_channel_error_is_half_its_weighted_square(
    budgeted: tuple[CausalPlan, np.ndarray], seed: int
) -> None:
    """``R(eps D) / (eps^2 D' W D / 2) -> 1`` with a gap linear in ``eps``, the cubic term's: a
    wrong ``W`` would leave a gap that does not shrink. Measured gaps at ``eps = 1e-3`` are 0.0002,
    0.004 and 0.021 over the three directions, a tenth of each at ``1e-2``."""
    plan, optimum = budgeted
    weight = plan.decision_weight()
    direction = np.random.default_rng(seed).normal(size=(2, 2))
    direction /= np.linalg.norm(direction)
    gaps = []
    for eps in (1e-2, 1e-3):
        predicted = weight.regret(eps * direction)
        assert predicted > 0
        gaps.append(abs(_regret(plan, optimum, eps * direction) / predicted - 1))
    assert gaps[1] < 0.03
    assert 0.05 < gaps[1] / gaps[0] < 0.2


def test_entry_i_j_of_the_channel_is_index_i_m_plus_j(
    budgeted: tuple[CausalPlan, np.ndarray],
) -> None:
    plan, optimum = budgeted
    weight = plan.decision_weight()
    change = np.zeros((2, 2))
    change[1, 0] = 1e-3
    assert weight.matrix[2, 2] / 2 * 1e-6 == pytest.approx(
        weight.regret(change), rel=1e-12, abs=0.0
    )
    assert _regret(plan, optimum, change) == pytest.approx(weight.regret(change), rel=0.02)


def test_a_change_to_the_one_step_channel_is_the_plant_s_b_moved_through_the_rk4_step(
    budgeted: tuple[CausalPlan, np.ndarray],
) -> None:
    """``E`` in the one-step map is ``M^-1 E`` in the plant's ``B``, with ``M = dt (I + h / 2 +
    h^2 / 6 + h^3 / 24)`` and ``h = dt A`` the RK4 step's factor on the action (the .mac file's
    STEP 1, in matrices). ``E`` is not symmetric, so entry ``(i, j)`` must move state ``i`` per
    unit of action ``j``."""
    plan, optimum = budgeted
    problem = plan._problem
    assert problem is not None
    a, b = np.asarray(PLANT.a_matrix), np.asarray(PLANT.b_matrix)
    h = problem.dt * a
    factor = problem.dt * (np.eye(2) + h / 2 + h @ h / 6 + h @ h @ h / 24)
    change = np.array([[0.0, 0.05], [-0.02, 0.0]])
    moved = LinearDynamics(jnp.asarray(a), jnp.asarray(b + np.linalg.solve(factor, change)))
    actions = jnp.asarray(optimum)
    through_b = _perturbed_task_cost(replace(problem, model=moved), actions, jnp.zeros((2, 2)))
    assert float(_perturbed_task_cost(problem, actions, jnp.asarray(change))) == pytest.approx(
        float(through_b), rel=1e-12
    )


def test_the_weight_is_symmetric_positive_semidefinite(
    budgeted: tuple[CausalPlan, np.ndarray],
) -> None:
    matrix = budgeted[0].decision_weight().matrix
    np.testing.assert_array_equal(matrix, matrix.T)
    assert np.linalg.eigvalsh(matrix)[0] > -1e-10 * np.abs(matrix).max()


def _ramp(t: float | jax.Array, x: jax.Array, u: jax.Array) -> jax.Array:
    """A plant whose law moves with the clock."""
    return -0.5 * x + u + jnp.sin(3.0 * t)


@pytest.mark.parametrize("model", [PLANT, _ramp], ids=["linear", "time-varying"])
def test_the_task_cost_it_differentiates_is_the_plan_s_at_no_change(model: object) -> None:
    cost = PLANT_COST if model is PLANT else SCALAR_COST
    start = START if model is PLANT else ONE
    plan = causal_plan(model, start, cost, 0.2, HORIZON, -BOX, BOX)
    assert plan._problem is not None
    shape = (start.shape[0], plan.actions.shape[1])
    at_zero = _perturbed_task_cost(plan._problem, plan.actions, jnp.zeros(shape))
    assert float(at_zero) == pytest.approx(plan.task_cost, rel=1e-12)


# ---- the edges of what it claims ----


def test_a_plan_held_entirely_on_its_box_has_nothing_to_lose() -> None:
    plan = causal_plan(SCALAR, ONE, SCALAR_COST, DT, 2, -0.1, 0.1)
    weight = plan.decision_weight()
    assert (weight.free, weight.weakly_active) == (0, 0)
    np.testing.assert_array_equal(weight.matrix, np.zeros((1, 1)))


def test_a_bound_met_with_no_pull_is_counted_as_weakly_active() -> None:
    """The unconstrained optimum put exactly on the bound: the plan is pinned there with a zero
    multiplier, and ``W`` is the pinned branch, 0, where the free one is 9.225."""
    optimum = -1.747878144003991
    plan = causal_plan(SCALAR, ONE, SCALAR_COST, DT, 1, optimum, 10.0)
    weight = _at(plan, np.array([[optimum]])).decision_weight()
    assert (weight.free, weight.weakly_active) == (0, 1)
    assert weight.matrix[0, 0] == 0.0


def test_a_row_met_with_no_pull_is_counted_as_weakly_active() -> None:
    """The row ``u_0 + u_1 >= c``, with ``c`` the free plan's own sum: it binds with a zero
    multiplier, and ``W`` is the branch that holds it, 0.997, where the free plan's is 4.707 (the
    .mac file's STEP 7c)."""
    optimum = np.array([-2.023822333873833, -1.248583272170075])
    row = LinearConstraint(np.ones((1, 2)), optimum.sum(), np.inf)
    plan = causal_plan(SCALAR, ONE, SCALAR_COST, DT, 2, -10.0, 10.0, constraints=(row,))
    weight = _at(plan, optimum[:, None]).decision_weight()
    assert (weight.free, weight.weakly_active) == (1, 1)  # two actions off the box, less the row
    assert weight.matrix[0, 0] == pytest.approx(0.9970492454922589, rel=1e-10)


def test_an_unfinished_solve_says_how_far_from_stationary_it_is() -> None:
    finished = causal_plan(PLANT, START, PLANT_COST, 0.2, HORIZON, -BOX, BOX)
    stopped = causal_plan(PLANT, START, PLANT_COST, 0.2, HORIZON, -BOX, BOX, steps=2)
    assert finished.decision_weight().residual < 1e-3
    assert stopped.decision_weight().residual > 1e-1


def test_a_lever_that_does_nothing_at_no_price_is_refused() -> None:
    """``B = 0`` and ``R = 0``: the cost is flat in the action, so the plan is no strict minimum,
    and a channel error would move the knowing plan without bound."""
    idle = LinearDynamics(jnp.array([[-0.5]]), jnp.array([[0.0]]))
    free = QuadraticCost(
        Q=jnp.eye(1), R=jnp.zeros((1, 1)), Qf=jnp.eye(1), x_target=jnp.array([0.0])
    )
    plan = causal_plan(idle, ONE, free, DT, 2, -1.0, 1.0)
    with pytest.raises(ValueError, match="not a strict local minimum"):
        plan.decision_weight()


def test_a_barrier_held_pessimistic_or_hand_built_plan_is_not_weighed() -> None:
    barrier = BarrierConstraint(lambda x: 2.0 - x[0])
    held = causal_plan(SCALAR, ONE, SCALAR_COST, DT, 3, -10.0, 10.0, barrier=barrier)
    with pytest.raises(ValueError, match="barrier"):
        held.decision_weight()
    rng = np.random.default_rng(0)
    support = SupportModel.fit(
        jnp.asarray(rng.normal(size=(50, 1))), jnp.asarray(rng.normal(size=(50, 1)))
    )
    pessimistic = causal_plan(
        SCALAR, ONE, SCALAR_COST, DT, 3, -10.0, 10.0, support=support, lam_supp=0.5
    )
    with pytest.raises(ValueError, match="pessimism penalties"):
        pessimistic.decision_weight()
    unweighted = causal_plan(
        SCALAR, ONE, SCALAR_COST, DT, 3, -10.0, 10.0, support=support, lam_supp=0.0
    )
    assert unweighted.decision_weight().matrix.shape == (1, 1)
    by_hand = CausalPlan(
        actions=jnp.zeros((3, 1)),
        trajectory=jnp.zeros((4, 1)),
        task_cost=0.0,
        uncertainty_tube=None,
        certified_horizon=None,
        solver_status="converged",
        solver_iterations=0,
    )
    with pytest.raises(ValueError, match="carries no problem"):
        by_hand.decision_weight()


@pytest.mark.parametrize("change", [np.zeros((1, 2)), np.zeros(1), np.array([[np.nan]])])
def test_a_change_that_is_not_the_channel_s_shape_is_refused(change: np.ndarray) -> None:
    weight = causal_plan(SCALAR, ONE, SCALAR_COST, DT, 1, -10.0, 10.0).decision_weight()
    with pytest.raises(ValueError, match=r"finite \(1, 1\) matrix"):
        weight.regret(change)


@pytest.mark.parametrize(("a", "horizon"), [(60.0, 30), (200.0, 20)])
def test_a_plan_whose_rollout_leaves_the_float_range_is_not_weighed(a: float, horizon: int) -> None:
    """At ``a = 60`` over 30 steps the rollout overflows, and the cost's derivatives with it: LAPACK
    failed to converge on the Hessian, and numpy reads a nan matrix's eigenvalues as finite. At
    ``a = 200`` over 20 the channel's mixed derivative stays finite while the others do not."""
    model = LinearDynamics(jnp.array([[a]]), jnp.array([[1.0]]))
    plan = causal_plan(model, ONE, SCALAR_COST, 1.0, horizon, -1.0, 1.0)
    with pytest.raises(ValueError, match="derivatives at the plan are not finite"):
        plan.decision_weight()
