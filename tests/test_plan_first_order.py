"""A plan says on its face how near it is to a first-order optimum and how far it breaks its rows.

The oracle is the problem's own Hessian. The plant is linear and the cost quadratic, so the cost is
exactly quadratic in the actions, and ``jax.hessian`` reads it exactly. The known optimum then
solves a linear system: unconstrained, under a row that binds, or on a box side found by bounded
least squares. A plan warm-started there takes no step and reads its residual at round-off. A plan
started off it, with no step allowed, reads the residual in closed form. The descent's variables
are ``v = sigma * u`` with ``sigma^2`` the Hessian's diagonal over the cost at zero, so the
gradient there is ``H (u - u*) / (sigma J(0))``.
"""

from __future__ import annotations

import dataclasses
import importlib
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import lsq_linear

from chc.control import LinearConstraint
from chc.cost import QuadraticCost, total_cost
from chc.dynamics import LinearDynamics
from chc.plan import BarrierConstraint, CausalPlan, _diagnostics, causal_plan
from chc.support import SupportModel, _augmented
from chc.uncertainty import ConfoundingRobustPenalty

if hasattr(jax, "enable_x64"):
    enable_x64 = jax.enable_x64
else:  # pragma: no cover - the dependency floor's spelling
    enable_x64 = importlib.import_module("jax.experimental").enable_x64

_A = jnp.array([[-0.5, 1.0], [0.0, -0.3]])
_X0 = jnp.array([1.0, 0.0])
_DT, _HORIZON = 0.1, 12
_STEP = 3  # the action a perturbation moves


def _problem(
    units: tuple[float, ...] = (1.0,), scale: float = 1.0
) -> tuple[LinearDynamics, QuadraticCost]:
    """``test_plan_units``' problem with lever ``j`` in ``units[j]`` times its own units and the
    cost times ``scale``; with two levers the second pushes the first state at half its rate."""
    size = jnp.asarray(units)
    channel = jnp.array([[0.0], [1.0]]) if len(units) == 1 else jnp.array([[0.0, 0.5], [1.0, 0.0]])
    weights = jnp.array([[0.05]]) if len(units) == 1 else jnp.diag(jnp.array([0.05, 0.02]))
    cost = QuadraticCost(
        Q=scale * jnp.diag(jnp.array([1.0, 0.1])),
        R=scale * weights / jnp.outer(size, size),
        Qf=scale * jnp.diag(jnp.array([5.0, 1.0])),
        x_target=jnp.zeros(2),
    )
    return LinearDynamics(_A, channel / size), cost


def _quadratic(model: LinearDynamics, cost: QuadraticCost, levers: int = 1):
    """The cost's Hessian, its gradient at zero and its value there, flattened as the rows are."""

    def task(flat: jax.Array) -> jax.Array:
        return total_cost(model, _X0, flat.reshape(_HORIZON, levers), _DT, cost)

    zero = jnp.zeros(_HORIZON * levers)
    return (
        np.asarray(jax.hessian(task)(zero)),
        np.asarray(jax.grad(task)(zero)),
        float(task(zero)),
    )


def _expected(hessian: np.ndarray, value: float, change: np.ndarray, free: np.ndarray) -> float:
    """The residual of a plan ``change`` off an unconstrained optimum whose bounds, if any, push
    nothing: the scaled gradient's norm over ``max(1, its norm on the free actions)``."""
    scaled = hessian @ change / (np.sqrt(np.diag(hessian) / value) * value)
    return float(np.linalg.norm(scaled)) / max(1.0, float(np.linalg.norm(scaled[free])))


def _plan(model, cost, lo, hi, start: np.ndarray, levers: int = 1, **kwargs) -> CausalPlan:
    return causal_plan(
        model,
        _X0,
        cost,
        _DT,
        _HORIZON,
        lo,
        hi,
        warm_start=jnp.asarray(start.reshape(_HORIZON, levers)),
        **kwargs,
    )


def test_an_lq_plan_at_its_known_optimum_reads_stationary_to_rounding() -> None:
    """At ``u* = -H^-1 g`` inside the box, under an equality row and a floor row that bind, and on
    the box's side, each optimum solved for exactly: the residual is rounding and nothing breaks."""
    model, cost = _problem()
    hessian, gradient, _ = _quadratic(model, cost)
    optimum = -np.linalg.solve(hessian, gradient)
    plan = _plan(model, cost, -5.0, 5.0, optimum)
    assert plan.solver_status == "no_progress"  # the descent takes no step from the optimum
    assert plan.stationarity < 1e-14
    assert plan.constraint_violation == 0.0
    assert (plan.stationarity_status, plan.violation_status) == ("within", "within")

    # a floor on the total the optimum spends below: it binds, its multiplier positive
    floor = 0.5 * float(optimum.sum())
    system = np.block(
        [[hessian, np.ones((_HORIZON, 1))], [np.ones((1, _HORIZON)), np.zeros((1, 1))]]
    )
    solved = np.linalg.solve(system, np.concatenate([-gradient, [floor]]))
    assert solved[-1] < 0.0  # the row's pull, -lambda, against a floor
    for row in (
        LinearConstraint(np.ones((1, _HORIZON)), floor, np.inf),
        LinearConstraint(np.ones((1, _HORIZON)), floor, floor),
    ):
        plan = _plan(model, cost, -5.0, 5.0, solved[:-1], constraints=(row,))
        assert plan.stationarity < 1e-12
        assert plan.constraint_violation <= plan.violation_tolerance

    # a cap below the optimum's largest action, its active set by bounded least squares on the
    # Cholesky factor, then solved exactly on the actions it leaves free
    cap = 0.6 * float(optimum.max())
    factor = np.linalg.cholesky(hessian)
    fit = lsq_linear(
        factor.T, -np.linalg.solve(factor, gradient), bounds=(-5.0, cap), method="bvls"
    )
    capped = np.isclose(fit.x, cap, rtol=0.0, atol=1e-9)
    free = ~capped
    boxed = np.where(capped, cap, 0.0)
    boxed[free] = np.linalg.solve(
        hessian[np.ix_(free, free)], -gradient[free] - hessian[np.ix_(free, capped)] @ boxed[capped]
    )
    assert int(capped.sum()) == 4
    assert np.all((hessian @ boxed + gradient)[capped] < 0.0)  # each side pushes the right way
    plan = _plan(model, cost, -5.0, cap, boxed)
    assert plan.stationarity < 1e-14
    assert plan.constraint_violation == 0.0


@pytest.mark.parametrize("level", [-0.2, 0.2])
def test_a_lever_pinned_by_a_box_of_zero_width_is_pushed_either_way(level: float) -> None:
    """The second lever's box closed at ``level``, as ``prescribe`` holds a lever the log never
    moved, and the first lever at its optimum given that: the pinned actions sit on both sides at
    once, so their push takes either sign, and the plan reads stationary to rounding whichever way
    the cost pulls them."""
    model, cost = _problem((1.0, 1.0))
    hessian, gradient, _ = _quadratic(model, cost, 2)
    first = np.arange(0, 2 * _HORIZON, 2)
    pinned = np.full(2 * _HORIZON, level)
    pinned[first] = 0.0
    optimum = pinned.copy()
    optimum[first] = -np.linalg.solve(
        hessian[np.ix_(first, first)], (gradient + hessian @ pinned)[first]
    )
    pulls = (hessian @ optimum + gradient)[1::2]
    assert np.all(np.sign(pulls) == np.sign(pulls[0]))  # every pinned action pulled one way
    lo, hi = jnp.array([-5.0, level]), jnp.array([5.0, level])
    plan = _plan(model, cost, lo, hi, optimum, 2)
    assert plan.solver_status == "no_progress"
    assert plan.stationarity < 1e-14


@pytest.mark.parametrize("change", [1e-3, 1.0, 10.0])
def test_a_plan_off_its_optimum_reads_how_far_off_in_closed_form(change: float) -> None:
    """One action moved off the optimum, no step allowed: the residual is ``H (u - u*)`` in the
    descent's variables, over ``max(1, its norm)``, which reads it relative past a norm of 1, as
    at a move of 10. Large enough, it reads past its tolerance."""
    model, cost = _problem()
    hessian, gradient, value = _quadratic(model, cost)
    optimum = -np.linalg.solve(hessian, gradient)
    moved = optimum.copy()
    moved[_STEP] += change
    plan = _plan(model, cost, -50.0, 50.0, moved, steps=0)
    expected = _expected(hessian, value, moved - optimum, np.ones(_HORIZON, dtype=bool))
    assert plan.stationarity == pytest.approx(expected, rel=1e-9, abs=0.0)
    assert plan.stationarity_status == "beyond"
    if change > 1.0:  # the scaled gradient's norm past 1, so the reading is relative
        assert plan.stationarity == pytest.approx(1.0, rel=1e-12, abs=0.0)


def test_an_action_held_on_a_side_its_gradient_pulls_it_off_is_not_stationary() -> None:
    """One action held on a floor half a unit off the optimum's level. Above it, the gradient
    pushes the action into the floor, which pushes back: the floor absorbs that action's share and
    the residual is the others'. Below it, the gradient pulls the action up, off the floor, which
    can push only up: the action's share stays in the residual, which reads it with its sign."""
    model, cost = _problem()
    hessian, gradient, value = _quadratic(model, cost)
    optimum = -np.linalg.solve(hessian, gradient)
    free = np.ones(_HORIZON, dtype=bool)
    free[_STEP] = False
    readings = {}
    for side in (0.5, -0.5):
        floor = np.full(_HORIZON, -5.0)
        floor[_STEP] = optimum[_STEP] + side
        assert floor[_STEP] < 0.0  # so the guess, zeros clipped to the box, is still zeros
        held = optimum.copy()
        held[_STEP] = floor[_STEP]
        plan = _plan(model, cost, floor.reshape(_HORIZON, 1), 5.0, held, steps=0)
        pull = (hessian @ (held - optimum))[_STEP]
        assert np.sign(pull) == np.sign(side)  # positive: downhill is down, into the floor
        scaled = hessian @ (held - optimum) / (np.sqrt(np.diag(hessian) / value) * value)
        kept = scaled if side < 0.0 else scaled[free]
        expected = float(np.linalg.norm(kept)) / max(1.0, float(np.linalg.norm(scaled[free])))
        assert plan.stationarity == pytest.approx(expected, rel=1e-9, abs=0.0)
        readings[side] = plan.stationarity
    assert readings[-0.5] > 1.2 * readings[0.5]


def test_a_breach_reads_as_its_distance_in_the_descent_s_variables() -> None:
    """Actions past the cap by ``d`` on one step and past a cap on their total by ``e``: the side's
    breach is ``sigma d``, the row's ``e / ||1 / sigma||``, the larger one read."""
    model, cost = _problem()
    hessian, gradient, value = _quadratic(model, cost)
    optimum = -np.linalg.solve(hessian, gradient)
    sigma = np.sqrt(np.diag(hessian) / value)
    total = float(optimum.sum())
    row = LinearConstraint(np.ones((1, _HORIZON)), -np.inf, total)
    plan = _plan(model, cost, -5.0, 5.0, optimum, constraints=(row,))
    problem = plan._problem
    assert problem is not None
    for side, spent in ((1e-3, 0.0), (0.0, 1e-3), (1e-3, 1e-2)):
        actions = optimum + spent / _HORIZON
        actions[_STEP] = 5.0 + side
        excess = float(actions.sum()) - total
        expected = max(sigma[_STEP] * side, max(excess, 0.0) / float(np.linalg.norm(1.0 / sigma)))
        _, _, violation, tolerance = _diagnostics(problem, jnp.asarray(actions.reshape(-1, 1)))
        assert violation == pytest.approx(expected, rel=1e-9, abs=0.0)
        # the projection's 256 eps (1 + max|v|), four times over
        reach = float(np.max(np.abs(sigma * actions)))
        assert tolerance == pytest.approx(
            1024 * np.finfo(np.float64).eps * (1.0 + reach), rel=1e-9, abs=0.0
        )
        assert violation > tolerance


_UNITS = [
    ((1e-6,), 1.0),
    ((1e-3,), 1.0),
    ((1e3,), 1.0),
    ((1e6,), 1.0),
    ((1.0,), 1e-6),
    ((1.0,), 1e6),
    ((1e-3, 1e3), 1.0),
    ((1e3, 1e-3), 1e-3),
]


@pytest.mark.parametrize(("units", "scale"), _UNITS)
def test_the_readings_are_the_same_in_any_units_of_each_lever_and_of_the_cost(
    units: tuple[float, ...], scale: float
) -> None:
    """One plan off its optimum and one breach, posed in units of each lever and of the cost: a
    cap on the first lever's total in its units, the actions and the box with them. The residual,
    the breach and both tolerances read the same, to rounding."""
    levers = len(units)
    readings = []
    for each, cost_in in (((1.0,) * levers, 1.0), (units, scale)):
        model, cost = _problem(each, cost_in)
        hessian, gradient, _ = _quadratic(model, cost, levers)
        optimum = -np.linalg.solve(hessian, gradient)
        size = np.tile(each, _HORIZON)
        moved = optimum.copy()
        moved[_STEP * levers] += 0.01 * each[0]  # the same move, in the first lever's units
        weights = np.zeros((1, _HORIZON * levers))
        weights[0, ::levers] = 1.0 / each[0]
        cap = float(optimum[::levers].sum()) / each[0]
        row = LinearConstraint(weights, -np.inf, cap + 0.02)
        box = np.array([5.0, 0.3][:levers]) * np.asarray(each)
        plan = _plan(model, cost, -box, box, moved, levers, steps=0, constraints=(row,))
        problem = plan._problem
        assert problem is not None
        breach = moved + 0.01 * size  # every action up, past the cap on the total
        breach[_STEP * levers] = box[0] * 1.001  # and one past its side of the box
        reading = _diagnostics(problem, jnp.asarray(breach.reshape(_HORIZON, levers)))
        readings.append((plan.stationarity, plan.violation_tolerance, *reading))
    reference, other = readings
    stationarity, _, _, _, breach, tolerance = reference
    assert stationarity > 1e-4  # off the optimum by far more than rounding
    assert breach > 1e3 * tolerance
    np.testing.assert_allclose(other, reference, rtol=1e-9, atol=0.0)


def test_a_converged_plan_reads_within_tolerances_set_by_the_descent_s_precision() -> None:
    """The descent stops where no step buys ``max(1e-14, 4 eps)`` of the cost at the guess, which
    leaves a residual near the root of twice that; the tolerance is ten times it. Converged, the
    two-lever plan reads 1.5e-7 in float64 and 5.7e-4 in float32, both within, and the box holds."""
    model, cost = _problem((1.0, 1.0))
    plan = causal_plan(
        model, _X0, cost, _DT, _HORIZON, jnp.array([-5.0, -0.3]), jnp.array([5.0, 0.3])
    )
    assert plan.solver_status == "converged"
    assert plan.stationarity_tolerance == pytest.approx(10.0 * math.sqrt(2e-14), rel=1e-15, abs=0.0)
    assert 1e-8 < plan.stationarity <= plan.stationarity_tolerance
    assert plan.violation_status == "within"
    with enable_x64(False):
        model, cost = _problem((1.0, 1.0))
        start, lo, hi = jnp.array([1.0, 0.0]), jnp.array([-5.0, -0.3]), jnp.array([5.0, 0.3])
        single = causal_plan(model, start, cost, _DT, _HORIZON, lo, hi)
    eps = float(np.finfo(np.float32).eps)
    assert single.stationarity_tolerance == pytest.approx(
        10.0 * math.sqrt(8.0 * eps), rel=1e-12, abs=0.0
    )
    assert single.solver_status == "converged"
    assert plan.stationarity_tolerance < single.stationarity < single.stationarity_tolerance


def test_a_pessimistic_plan_is_read_on_the_objective_it_minimised() -> None:
    """The support and confounding penalties are part of what the descent minimised: read on them
    the plan is stationary, and read on the task cost alone it would not be."""
    model = LinearDynamics(jnp.array([[-0.5]]), jnp.array([[1.0]]))
    cost = QuadraticCost(
        Q=jnp.eye(1), R=jnp.array([[0.05]]), Qf=jnp.eye(1), x_target=jnp.array([2.0])
    )
    logged_x = 1.5 + 0.3 * jax.random.normal(jax.random.key(0), (400, 1))
    logged_u = 0.3 + 0.1 * jax.random.normal(jax.random.key(1), (400, 1))
    support = SupportModel.fit(logged_x, logged_u)
    row = LinearConstraint(np.ones((1, 48)), -np.inf, 10.0)
    plan = causal_plan(
        model,
        jnp.array([1.0]),
        cost,
        0.5,
        48,
        0.0,
        50.0,
        support=support,
        lam_supp=1.0,
        uncertainty=ConfoundingRobustPenalty(radius=0.2),
        lam_unc=1.0,
        constraints=(row,),
    )
    assert plan.solver_status == "converged"
    assert plan.stationarity_status == "within"
    problem = plan._problem
    assert problem is not None
    task_only = dataclasses.replace(problem, support=None, uncertainty=None)
    stationarity, *_ = _diagnostics(task_only, plan.actions)
    assert stationarity > 1e3 * plan.stationarity_tolerance


def test_a_pessimistic_plan_s_reading_is_a_share_of_its_objective_at_the_guess() -> None:
    """Off its optimum, no step allowed, a plan solved with a support penalty reads the
    objective's gradient over ``sigma |J|``, ``|J|`` the objective's at zeros, penalty included,
    and ``sigma^2`` the task cost's curvature over it, as the penalised descent scales it."""
    model = LinearDynamics(jnp.array([[-0.5]]), jnp.array([[1.0]]))
    cost = QuadraticCost(
        Q=jnp.eye(1), R=jnp.array([[0.05]]), Qf=jnp.eye(1), x_target=jnp.array([2.0])
    )
    x0, dt, horizon = jnp.array([1.0]), 0.5, 8
    logged_x = 1.5 + 0.3 * jax.random.normal(jax.random.key(0), (400, 1))
    logged_u = 0.3 + 0.1 * jax.random.normal(jax.random.key(1), (400, 1))
    support = SupportModel.fit(logged_x, logged_u)

    def solve(**kwargs: object) -> CausalPlan:
        return causal_plan(
            model, x0, cost, dt, horizon, -50.0, 50.0, support=support, lam_supp=1.0, **kwargs
        )

    # near the optimum: the support penalty is steep, and further off the reading passes 1
    start = solve().actions + 0.002
    plan = solve(steps=0, warm_start=start)

    def objective(us: jax.Array) -> jax.Array:
        return _augmented(model, x0, us, dt, cost, support, 1.0, None, 0.0)

    zeros = jnp.zeros((horizon, 1))
    value = float(objective(zeros))
    task = float(total_cost(model, x0, zeros, dt, cost))
    assert value > 1.5 * task  # the penalty is a real share of the objective at the guess
    curvature = np.diag(
        np.asarray(
            jax.hessian(lambda u: total_cost(model, x0, u.reshape(horizon, 1), dt, cost))(
                zeros.ravel()
            )
        )
    )
    scaled = np.asarray(jax.grad(objective)(start)).ravel() / np.sqrt(curvature * value)
    assert float(np.linalg.norm(scaled)) < 1.0  # so it is read as it is, not relative
    assert plan.stationarity == pytest.approx(float(np.linalg.norm(scaled)), rel=1e-9, abs=0.0)


def test_what_is_not_read_says_so() -> None:
    """A plan held under a barrier has no stationarity to read, since the barrier's multipliers
    would enter it, but its box and rows are read. A plan built by hand reads neither. One whose
    rollout overflowed reads ``not_finite``, under rows as without. A reading on its tolerance
    is within."""
    model, cost = _problem()
    held = causal_plan(
        model, _X0, cost, _DT, _HORIZON, -5.0, 5.0, barrier=BarrierConstraint(lambda x: x[0] + 0.5)
    )
    assert held.stationarity is None
    assert held.stationarity_status == "not_evaluated"
    assert held.violation_status == "within"
    by_hand = CausalPlan(held.actions, held.trajectory, held.task_cost, None, None, "converged", 1)
    assert (by_hand.stationarity_status, by_hand.violation_status) == ("not_evaluated",) * 2
    edge = dataclasses.replace(by_hand, stationarity=0.5, stationarity_tolerance=0.5)
    assert edge.stationarity_status == "within"
    assert dataclasses.replace(edge, stationarity=0.75).stationarity_status == "beyond"

    growing = LinearDynamics(jnp.array([[50.0]]), jnp.array([[1.0]]))
    price = QuadraticCost(Q=jnp.eye(1), R=jnp.eye(1), Qf=jnp.eye(1), x_target=jnp.zeros(1))
    budget = LinearConstraint(np.ones((1, 40)), -np.inf, 1.0)
    for rows in ((), (budget,)):
        # from a floor of 0 every action starts on a side, which the residual reads by BVLS
        plan = causal_plan(
            growing, jnp.array([1.0]), price, 1.0, 40, 0.0, 1.0, steps=0, constraints=rows
        )
        assert math.isinf(plan.task_cost)
        assert plan.stationarity_status == "not_finite"
        assert plan.violation_status == "within"
