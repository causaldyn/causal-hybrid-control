"""A plan reads one problem the same way in any units of its levers and of its cost."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from chc.control import LinearConstraint, projected_gradient_solve
from chc.cost import QuadraticCost, total_cost
from chc.dynamics import HybridDynamics, LinearDynamics
from chc.plan import CausalPlan, causal_plan
from chc.residual import ZeroResidual

_A = jnp.array([[-0.5, 1.0], [0.0, -0.3]])
_X0 = jnp.array([1.0, 0.0])
_DT, _HORIZON = 0.1, 12


def _plan(
    channel: Array,
    weights: Array,
    units: tuple[float, ...],
    lo: np.ndarray,
    hi: np.ndarray,
    scale: float = 1.0,
    rows: tuple[LinearConstraint, ...] = (),
) -> CausalPlan:
    """``test_plan``'s problem with lever ``j`` in units ``units[j]`` times its own, the cost times
    ``scale``: the channel over the units, ``R`` over their outer product, the box times them."""
    size = jnp.asarray(units)
    model = HybridDynamics(known=LinearDynamics(_A, channel / size), residual=ZeroResidual(2))
    cost = QuadraticCost(
        Q=scale * jnp.diag(jnp.array([1.0, 0.1])),
        R=scale * weights / jnp.outer(size, size),
        Qf=scale * jnp.diag(jnp.array([5.0, 1.0])),
        x_target=jnp.zeros(2),
    )
    return causal_plan(
        model,
        _X0,
        cost,
        _DT,
        _HORIZON,
        lo * np.asarray(units),
        hi * np.asarray(units),
        constraints=rows,
    )


_ONE = (jnp.array([[0.0], [1.0]]), jnp.array([[0.05]]), np.array([-5.0]), np.array([5.0]))
_TWO = (
    jnp.array([[0.0, 0.5], [1.0, 0.0]]),
    jnp.diag(jnp.array([0.05, 0.02])),
    np.array([-5.0, -0.3]),
    np.array([5.0, 0.3]),
)


def _rows(units: tuple[float, float]) -> tuple[LinearConstraint, ...]:
    """``u1 + u2 >= -1`` and ``u1 - u2 <= 0.6`` at every step, and ``|u1|`` moving at most 0.8 a
    step, in the levers' own units. The two rows of a step are orthogonal in sum only, so they stay
    orthogonal in no units but equal ones."""
    pair = np.array([[1.0, 1.0], [1.0, -1.0]])
    matrix = np.kron(np.eye(_HORIZON), pair) / np.tile(units, _HORIZON)
    lower = np.tile([-1.0, -np.inf], _HORIZON)
    upper = np.tile([np.inf, 0.6], _HORIZON)
    rate = LinearConstraint.rate_limit(_HORIZON, [0.8 * units[0], np.inf])
    return LinearConstraint(matrix, lower, upper), rate


def _read_back(plan: CausalPlan, units: tuple[float, ...], width: np.ndarray) -> np.ndarray:
    """The plan in the levers' own units, as a share of each lever's box."""
    return np.asarray(plan.actions) / np.asarray(units) / width


_REFERENCE = {
    "one": _plan(_ONE[0], _ONE[1], (1.0,), _ONE[2], _ONE[3]),
    "two": _plan(_TWO[0], _TWO[1], (1.0, 1.0), _TWO[2], _TWO[3]),
    "rows": _plan(_TWO[0], _TWO[1], (1.0, 1.0), _TWO[2], _TWO[3], rows=_rows((1.0, 1.0))),
}


@pytest.mark.parametrize("units", [1e-6, 1e-3, 1e3, 1e6])
def test_a_plan_reads_the_same_in_any_units_of_its_lever(units: float) -> None:
    """The descent's first step was 0.2 action units per unit of gradient, and a step counted where
    the cost fell by 1e-9 in its own units. With the lever in units 1e3 times its own it ran its
    10 000 steps and stopped at a cost of 4.4227 against 2.9023, its regret bound 8.0; at 1e6 it
    took no step, and at 1e-3 it stopped 8e-5 of the box from the plan in the lever's own units.
    Now every one of them converges in 36 steps to the plan, to 9.7e-17 of the box."""
    reference, width = _REFERENCE["one"], _ONE[3] - _ONE[2]
    plan = _plan(_ONE[0], _ONE[1], (units,), _ONE[2], _ONE[3])
    assert reference.solver_status == plan.solver_status == "converged"
    gap = np.abs(_read_back(plan, (units,), width) - _read_back(reference, (1.0,), width))
    assert float(gap.max()) <= 1e-6


@pytest.mark.parametrize("units", [(1e-3, 1e3), (1e3, 1e-3), (1e-6, 1e6)])
def test_a_plan_reads_the_same_with_each_lever_in_its_own_units(
    units: tuple[float, float],
) -> None:
    """Two levers in units 1e-3 and 1e3 times their own: the first lever's gradient grows a
    thousandfold and the second's shrinks as much, and no one step size serves both. There the
    descent reported ``converged`` after 23 steps with the second lever moved by 3e-11, at a cost of
    2.9023 where the plan in the levers' own units costs 2.5592; in 1e3 and 1e-3 it ran out of
    steps at 3.5740. Now each converges in 35 steps to the plan, to 9.7e-17 of the box."""
    reference, width = _REFERENCE["two"], _TWO[3] - _TWO[2]
    plan = _plan(_TWO[0], _TWO[1], units, _TWO[2], _TWO[3])
    assert reference.solver_status == plan.solver_status == "converged"
    gap = np.abs(_read_back(plan, units, width) - _read_back(reference, (1.0, 1.0), width))
    assert float(gap.max()) <= 1e-6
    lo, hi = _TWO[2] * np.asarray(units), _TWO[3] * np.asarray(units)
    actions = np.asarray(plan.actions)
    assert bool(((actions >= lo) & (actions <= hi)).all())  # exactly, not to a tolerance


@pytest.mark.parametrize("units", [(1e-3, 1e3), (1e3, 1e-3), (1e-6, 1e6)])
def test_a_plan_under_rows_reads_the_same_with_each_lever_in_its_own_units(
    units: tuple[float, float],
) -> None:
    """The rows are carried into the descent's variables with the levers, and projected on there.
    In units 1e-3 and 1e3 the descent reported ``converged`` after 124 steps at a cost of 3.1112,
    where the plan in the levers' own units costs 2.7675, and in 1e3 and 1e-3 it ran out of steps
    at 3.5740. Now each converges in 17 steps to the plan, to 7.6e-17 of the box."""
    reference, width = _REFERENCE["rows"], _TWO[3] - _TWO[2]
    rows = _rows(units)
    plan = _plan(_TWO[0], _TWO[1], units, _TWO[2], _TWO[3], rows=rows)
    assert reference.solver_status == plan.solver_status == "converged"
    gap = np.abs(_read_back(plan, units, width) - _read_back(reference, (1.0, 1.0), width))
    assert float(gap.max()) <= 1e-6
    flat = np.asarray(plan.actions).ravel()
    for row in rows:
        level = row.matrix @ flat
        assert bool(((level >= row.lower - 1e-9) & (level <= row.upper + 1e-9)).all())


@pytest.mark.parametrize(
    ("units", "scale"), [(1.0, 1.0), (1e-3, 1.0), (1e3, 1.0), (1.0, 1e-6), (1.0, 1e6)]
)
def test_a_budget_is_spent_the_same_in_any_units(units: float, scale: float) -> None:
    """``test_plan_oracles``' budget, 35.5 of 100 steps' spend towards a revenue out of reach: the
    plan spends it first, where a step of goodwill earns the most by the end. A trial thousands of
    boxes outside the box was projected onto the budget by Dykstra's sweeps, which creep there by a
    sliver of the box each, and by an active-set polish that moves one constraint a step and gave up
    after 16, so the line search took a point over budget. With the lever in units 1e-3 times its
    own, or the cost 1e6 times, the old descent's first step went that far, and it reported
    ``converged`` having spent 68; with the lever 1e3 times, or the cost 1e-6 times, it ran out of
    its 10 000 steps 0.62 of the box from the plan. Now the polish may take two steps a coordinate,
    and each reading spends the budget as the oracle does, in one step, to 6e-11 of the box."""
    dt, horizon, far, budget = 0.1, 100, 1e3, 35.5
    model = LinearDynamics(jnp.array([[-0.5, 0.0], [1.0, 0.0]]), jnp.array([[1.0 / units], [0.0]]))
    cost = QuadraticCost(
        Q=jnp.zeros((2, 2)),
        R=jnp.zeros((1, 1)),
        Qf=scale * jnp.diag(jnp.array([0.0, 1.0])),
        x_target=jnp.array([0.0, far]),
    )
    row = LinearConstraint(np.ones((1, horizon)), np.array([-np.inf]), np.array([budget * units]))
    plan = causal_plan(model, jnp.zeros(2), cost, dt, horizon, 0.0, units, constraints=(row,))
    assert plan.solver_status == "converged"
    spend = np.zeros(horizon)
    spend[:35], spend[35] = 1.0, budget - 35.0
    assert float(np.max(np.abs(np.asarray(plan.actions)[:, 0] / units - spend))) <= 1e-6


@pytest.mark.parametrize("scale", [1e-9, 1e-6, 1e-3, 1e3, 1e6])
def test_a_plan_reads_the_same_in_any_units_of_its_cost(scale: float) -> None:
    """A step counted where the cost fell by 1e-9, whatever the cost's size: with the cost in units
    1e6 or 1e9 times smaller no step did, and the plan was the zero guess at a cost of 4.4246
    against 2.9023; 1e3 times smaller, the descent ran out of steps at 3.4113. Now each converges
    in 36 steps to the plan, to 1.1e-16 of the box, its cost to 3.1e-16."""
    reference, width = _REFERENCE["one"], _ONE[3] - _ONE[2]
    plan = _plan(_ONE[0], _ONE[1], (1.0,), _ONE[2], _ONE[3], scale=scale)
    assert plan.solver_status == "converged"
    assert plan.task_cost / scale == pytest.approx(reference.task_cost, rel=1e-12, abs=0.0)
    gap = np.abs(_read_back(plan, (1.0,), width) - _read_back(reference, (1.0,), width))
    assert float(gap.max()) <= 1e-6


@pytest.mark.parametrize("units", [(1.0, 1.0), (1e-3, 1e3)])
def test_the_first_step_is_the_newton_step_along_each_action_alone(
    units: tuple[float, float],
) -> None:
    """``lr0 = 1`` is the step to the cost's minimum along each action alone, ``-g / H_ii``, clipped
    to the box: on a plant affine in the state and the action the Gauss-Newton curvature is the
    Hessian's diagonal itself. The full step buys 0.41 of the fall the gradient predicts for it,
    more than the third the line search asks, so the line search takes it, in any units of the
    levers."""
    size = np.asarray(units)
    channel, _, lo, hi = _TWO
    model = LinearDynamics(_A, channel / jnp.asarray(size))
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.1])),
        R=jnp.diag(jnp.array([1.0, 0.5])) / jnp.outer(jnp.asarray(size), jnp.asarray(size)),
        Qf=jnp.diag(jnp.array([5.0, 1.0])),
        x_target=jnp.zeros(2),
    )
    guess = jnp.zeros((_HORIZON, 2))

    def flat(actions: Array) -> Array:
        return total_cost(model, _X0, actions.reshape(_HORIZON, 2), _DT, cost)

    gradient = jax.grad(flat)(guess.ravel())
    curvature = jnp.diagonal(jax.hessian(flat)(guess.ravel()))
    newton = np.clip(np.asarray(-gradient / curvature).reshape(_HORIZON, 2), lo * size, hi * size)
    solve = projected_gradient_solve(model, _X0, guess, _DT, cost, lo * size, hi * size, steps=1)
    assert solve.iterations == 1
    gap = np.abs(np.asarray(solve.actions) - newton) / size / (hi - lo)
    assert float(gap.max()) <= 1e-15


def test_a_guess_outside_the_box_plans_as_the_guess_clipped_to_it() -> None:
    """The actions' scale and the stopping rule's cost are read at the guess clipped to the box,
    which is the descent's first iterate, so a guess far outside the box plans bit for bit as the
    guess clipped to it."""
    channel, weights, lo, hi = _ONE
    model = HybridDynamics(known=LinearDynamics(_A, channel), residual=ZeroResidual(2))
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.1])),
        R=weights,
        Qf=jnp.diag(jnp.array([5.0, 1.0])),
        x_target=jnp.zeros(2),
    )
    outside = jnp.full((_HORIZON, 1), 1e3)
    far, clipped = (
        projected_gradient_solve(model, _X0, guess, _DT, cost, float(lo[0]), float(hi[0]))
        for guess in (outside, jnp.clip(outside, lo[0], hi[0]))
    )
    assert far.status == clipped.status == "converged"
    assert far.iterations == clipped.iterations
    assert np.array_equal(np.asarray(far.actions), np.asarray(clipped.actions))


@pytest.mark.parametrize(("side", "rounds"), [(0.46, -1.0), (0.47, 1.0)])
def test_a_plan_on_a_side_of_its_box_is_read_back_on_it(side: float, rounds: float) -> None:
    """The descent steps in ``v = sigma u`` and reads each trial back as ``v / sigma``. Here every
    action's ``sigma`` is 0.0346 and the plan sits on the side of its box, which ``side * sigma /
    sigma`` rounds one ulp inside the box for 0.46 and one ulp past it for 0.47. An action that
    ``v`` puts on a side is read back as the side itself, so either way the plan lands on it."""
    cost = QuadraticCost(
        Q=jnp.zeros((1, 1)), R=jnp.array([[0.05]]), Qf=jnp.eye(1), x_target=jnp.array([10.0])
    )
    model = LinearDynamics(jnp.zeros((1, 1)), jnp.ones((1, 1)))
    guess, lo, hi = jnp.zeros((3, 1)), jnp.full((3, 1), -side), jnp.full((3, 1), side)
    from chc.control import _action_units

    sigma = float(_action_units(model, jnp.zeros(1), guess, _DT, cost, lo, hi)[0][0, 0])
    assert np.sign(side * sigma / sigma - side) == rounds
    solve = projected_gradient_solve(model, jnp.zeros(1), guess, _DT, cost, -side, side)
    assert solve.status == "converged"
    assert np.asarray(solve.actions).tolist() == [[side], [side], [side]]


class _Gated(eqx.Module):
    """``x1' = x2 u1`` and ``x2' = u2 - x2``: the first lever acts through the state the second
    builds, so where that state is zero the cost does not feel the first lever at all."""

    units: Array

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        u = u / self.units
        return jnp.array([x[1] * u[0], u[1] - x[1]])


@pytest.mark.parametrize("units", [1e-6, 1e-3, 1e3, 1e6])
def test_a_lever_the_cost_does_not_feel_at_the_start_is_measured_by_its_box(units: float) -> None:
    """At the zero guess the cost has no curvature along the first lever and does not charge for
    it, so its curvature gives it no size; its box does. It used to stop with that lever never
    moved, at a cost of 8.61 against 3.50: at 1e3 after its 10 000 steps, and at 1e6 reporting
    ``converged``. Now each converges in 34 steps to the plan in the levers' own units, which it
    matched to the bit when measured."""
    size = np.array([units, 1.0])
    weights = jnp.diag(jnp.array([0.0, 0.05]))
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.1])),
        R=weights / jnp.outer(jnp.asarray(size), jnp.asarray(size)),
        Qf=jnp.diag(jnp.array([5.0, 0.0])),
        x_target=jnp.array([0.0, 0.5]),
    )
    lo, hi = np.array([-2.0, -1.0]), np.array([2.0, 1.0])

    def plan(size: np.ndarray, cost: QuadraticCost) -> CausalPlan:
        return causal_plan(
            _Gated(jnp.asarray(size)), _X0, cost, _DT, _HORIZON, lo * size, hi * size
        )

    reference = plan(np.ones(2), QuadraticCost(cost.Q, weights, cost.Qf, cost.x_target))
    scaled = plan(size, cost)
    assert reference.solver_status == scaled.solver_status == "converged"
    gap = np.abs(
        _read_back(scaled, tuple(size), hi - lo) - _read_back(reference, (1.0, 1.0), hi - lo)
    )
    assert float(gap.max()) <= 1e-6
    assert scaled.task_cost == pytest.approx(reference.task_cost, rel=1e-12, abs=0.0)
