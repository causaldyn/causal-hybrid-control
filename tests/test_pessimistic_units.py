"""The penalised descent reads one problem the same in any units of its levers and of its cost."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import chc.support
from chc.control import LinearConstraint, SolverResult
from chc.cost import QuadraticCost
from chc.dynamics import HybridDynamics, LinearDynamics
from chc.residual import ZeroResidual
from chc.support import SupportModel, pessimistic_control, pessimistic_solve
from chc.uncertainty import ConfoundingRobustPenalty

_A = jnp.array([[-0.5, 1.0], [0.0, -0.3]])
_X0 = jnp.array([1.0, 0.0])
_DT, _HORIZON = 0.1, 12
_KEY_X, _KEY_U = jax.random.split(jax.random.key(7))
_LOG_X = jax.random.normal(_KEY_X, (500, 2))

_ONE = (jnp.array([[0.0], [1.0]]), jnp.array([[0.05]]), np.array([-5.0]), np.array([5.0]))
_TWO = (
    jnp.array([[0.0, 0.5], [1.0, 0.0]]),
    jnp.diag(jnp.array([0.05, 0.02])),
    np.array([-5.0, -0.3]),
    np.array([5.0, 0.3]),
)


def _solve(
    spec: tuple[Array, Array, np.ndarray, np.ndarray],
    units: tuple[float, ...],
    scale: float,
    penalty: str,
    rows: tuple[LinearConstraint, ...] = (),
) -> SolverResult:
    """``test_plan_units``' problem under a penalty, with lever ``j`` in units ``units[j]`` times
    its own -- the channel over the units, ``R`` over their outer product, the box and the log's
    actions times them, the confounding radius over them -- and the cost and the penalty's weight
    ``scale`` times their own. ``support`` fits the support on a log of ``0.4`` of each lever's
    standard deviation; ``radius`` is the confounding radius 0.3, one lever only, since a norm over
    levers in different units is another penalty in each."""
    channel, weights, lo, hi = spec
    size = jnp.asarray(units)
    model = HybridDynamics(known=LinearDynamics(_A, channel / size), residual=ZeroResidual(2))
    cost = QuadraticCost(
        Q=scale * jnp.diag(jnp.array([1.0, 0.1])),
        R=scale * weights / jnp.outer(size, size),
        Qf=scale * jnp.diag(jnp.array([5.0, 1.0])),
        x_target=jnp.zeros(2),
    )
    support = SupportModel.fit(_LOG_X, 0.4 * jax.random.normal(_KEY_U, (500, len(units))) * size)
    if penalty == "support":
        lam_supp, uncertainty, lam_unc = 0.05 * scale, None, 0.0
    else:
        lam_supp, lam_unc = 0.0, 0.2 * scale
        uncertainty = ConfoundingRobustPenalty(radius=0.3 / float(size[0]))
    return pessimistic_solve(
        model,
        _X0,
        jnp.zeros((_HORIZON, len(units))),
        _DT,
        cost,
        support,
        lam_supp,
        lo * np.asarray(units),
        hi * np.asarray(units),
        uncertainty=uncertainty,
        lam_unc=lam_unc,
        constraints=rows,
    )


def _gap(
    solve: SolverResult, reference: SolverResult, units: tuple[float, ...], width: np.ndarray
) -> float:
    """How far apart the two plans are in the levers' own units, as a share of each lever's box."""
    plan = np.asarray(solve.actions) / np.asarray(units)
    return float(np.max(np.abs(plan - np.asarray(reference.actions)) / width))


_REFERENCE = {
    "one support": _solve(_ONE, (1.0,), 1.0, "support"),
    "one radius": _solve(_ONE, (1.0,), 1.0, "radius"),
    "two support": _solve(_TWO, (1.0, 1.0), 1.0, "support"),
}


@pytest.mark.parametrize("units", [1e-6, 1e-3, 1e3, 1e6])
def test_a_penalised_plan_reads_the_same_in_any_units_of_its_lever(units: float) -> None:
    """The descent stepped 0.2 action units per unit of gradient and counted a fall of 1e-9 in the
    cost's own units. With the lever in units 1e3 times its own it ran its 10 000 steps and stopped
    5.8e-2 of the box from where it stopped in the lever's own units, at 1e-6 and 1e-3 times it
    stopped 7.0e-6 and 7.4e-6 of the box from there, and at 1e6 times it took no step. Now each
    converges in 13 steps to the same plan, to 3.9e-17 of the box."""
    reference, width = _REFERENCE["one support"], _ONE[3] - _ONE[2]
    solve = _solve(_ONE, (units,), 1.0, "support")
    assert reference.status == solve.status == "converged"
    assert _gap(solve, reference, (units,), width) <= 1e-6


@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1e3, 1e6])
def test_a_penalised_plan_reads_the_same_in_any_units_of_its_cost(scale: float) -> None:
    """With the cost and the penalty's weight 1e6 times smaller the descent took no step, and 1e3
    times smaller it ran out of steps 9.4e-3 of the box from the plan; larger, it stopped 5.6e-6
    of the box from it. Now each converges in 13 steps, to 5.6e-17 of the box."""
    reference, width = _REFERENCE["one support"], _ONE[3] - _ONE[2]
    solve = _solve(_ONE, (1.0,), scale, "support")
    assert reference.status == solve.status == "converged"
    assert _gap(solve, reference, (1.0,), width) <= 1e-6


@pytest.mark.parametrize("units", [(1e-3, 1e3), (1e3, 1e-3), (1e-6, 1e6)])
def test_a_penalised_plan_reads_the_same_with_each_lever_in_its_own_units(
    units: tuple[float, float],
) -> None:
    """Two levers in units 1e-3 and 1e3 times their own: no one step in the caller's units serves
    both, and the descent reported ``converged`` half the box from the plan, or a tenth with the
    units the other way round. Now each converges in 26 steps, to 2.3e-13 of the box."""
    reference, width = _REFERENCE["two support"], _TWO[3] - _TWO[2]
    solve = _solve(_TWO, units, 1.0, "support")
    assert reference.status == solve.status == "converged"
    assert _gap(solve, reference, units, width) <= 1e-6
    lo, hi = _TWO[2] * np.asarray(units), _TWO[3] * np.asarray(units)
    actions = np.asarray(solve.actions)
    assert bool(((actions >= lo) & (actions <= hi)).all())  # exactly, not to a tolerance


def _rows(units: tuple[float, float]) -> tuple[LinearConstraint, ...]:
    """``u1 + u2 >= -0.3`` and ``u1 - u2 <= 0.1`` at every step, and ``|u1|`` moving at most 0.1
    a step, in the levers' own units: orthogonal in sum only, so orthogonal in no other units. The
    first two hold 17 of the plan's actions where they would not be without them."""
    pair = np.array([[1.0, 1.0], [1.0, -1.0]])
    matrix = np.kron(np.eye(_HORIZON), pair) / np.tile(units, _HORIZON)
    lower = np.tile([-0.3, -np.inf], _HORIZON)
    upper = np.tile([np.inf, 0.1], _HORIZON)
    rate = LinearConstraint.rate_limit(_HORIZON, [0.1 * units[0], np.inf])
    return LinearConstraint(matrix, lower, upper), rate


@pytest.mark.parametrize("units", [(1e-3, 1e3), (1e3, 1e-3)])
def test_a_penalised_plan_under_rows_reads_the_same_with_each_lever_in_its_own_units(
    units: tuple[float, float],
) -> None:
    """The rows are carried into the descent's variables with the levers, in whatever metric the
    descent reads at each step. In units 1e-3 and 1e3 the descent reported ``converged`` 0.33 of
    the box from the plan, and the other way round ran out of its 10 000 steps 0.17 from it. Now
    each converges in 17 steps, to 1.2e-7 of the box, where a step lowers the cost by 1e-14 of
    it."""
    reference = _solve(_TWO, (1.0, 1.0), 1.0, "support", rows=_rows((1.0, 1.0)))
    rows = _rows(units)
    solve = _solve(_TWO, units, 1.0, "support", rows=rows)
    assert reference.status == solve.status == "converged"
    assert _gap(solve, reference, units, _TWO[3] - _TWO[2]) <= 1e-6
    flat = np.asarray(solve.actions).ravel()
    for row in rows:
        level = row.matrix @ flat
        assert bool(((level >= row.lower - 1e-9) & (level <= row.upper + 1e-9)).all())


@pytest.mark.parametrize(
    ("units", "scale"),
    [(1.0, 1.0), (1e-6, 1.0), (1e-3, 1.0), (1e3, 1.0), (1e6, 1.0), (1.0, 1e-6), (1.0, 1e6)],
)
def test_a_plan_under_the_confounding_radius_converges_in_any_units(
    units: float, scale: float
) -> None:
    """The radius' norm is smoothed over a millionth of the plan's size, so its curvature sits
    within that length of zero, where the plan puts most of its actions: no scale read at the guess
    sees it. The descent ran out of its 10 000 steps in the problem's own units; with the lever in
    units 1e-6 to 1e6 times its own, or the cost 1e-6 or 1e6 times, it stopped 1.9e-2 to 0.20 of
    the box from there, and took no step at all with the lever 1e6 times or the cost 1e-6 times.
    Read as the descent goes, the curvature is measured where the plan is, and each converges in 85
    to 97 steps, to 2.6e-7 of the box."""
    reference, width = _REFERENCE["one radius"], _ONE[3] - _ONE[2]
    solve = _solve(_ONE, (units,), scale, "radius")
    assert reference.status == solve.status == "converged"
    assert _gap(solve, reference, (units,), width) <= 1e-6


@pytest.mark.parametrize("pinned", [0, 1])
@pytest.mark.parametrize("scale", [1.0, 1e6])
def test_a_penalised_plan_the_rows_and_the_box_pin_to_one_point_stays_on_it(
    pinned: int, scale: float
) -> None:
    """``3 u1 = u2`` at every step, with one lever's box pinned at zero, leaves the zero plan alone
    feasible, and the projection holds the row only to its tolerance. The penalised descent took a
    trial that moved the plan by that tolerance alone as a step where it lowered the cost by more
    than ``1e-9``: with the cost 1e6 times its own, one step 1.2e-12 off the zero plan, reported
    ``converged``. A move only across the constraints held at both of its ends is not a step."""
    horizon = 3
    model = LinearDynamics(jnp.array([[-0.5]]), jnp.array([[0.8, 0.1]]))
    cost = QuadraticCost(
        Q=scale * jnp.eye(1),
        R=scale * jnp.diag(jnp.array([0.01, 1.0])),
        Qf=scale * jnp.eye(1),
        x_target=jnp.array([1.0]),
    )
    support = SupportModel.fit(
        jax.random.normal(_KEY_X, (200, 1)), jax.random.normal(_KEY_U, (200, 2))
    )
    tie = LinearConstraint(np.kron(np.eye(horizon), [[3.0, -1.0]]) / np.sqrt(10.0), 0.0, 0.0)
    lo, hi = np.full((horizon, 2), -2.0), np.full((horizon, 2), 2.0)
    lo[:, pinned] = hi[:, pinned] = 0.0
    solve = pessimistic_solve(
        model,
        jnp.ones(1),
        jnp.zeros((horizon, 2)),
        _DT,
        cost,
        support,
        0.01 * scale,
        lo,
        hi,
        constraints=(tie,),
    )
    assert solve.status == "no_progress"
    assert np.asarray(solve.actions).tolist() == [[0.0, 0.0]] * horizon


def test_every_penalised_descent_projects_on_classes_its_scaling_keeps_orthogonal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The descent scales each action's column by its own factor, so rows share a class only where
    no action enters both: a rate band and a budget, orthogonal in the caller's units, are projected
    on apart, by both entry points."""
    seen = []
    loop = chc.support._pessimistic_loop

    def recorded(*args):
        seen.append(args[14])  # the constraint blocks
        return loop(*args)

    monkeypatch.setattr(chc.support, "_pessimistic_loop", recorded)
    channel, weights, lo, hi = _ONE
    model = HybridDynamics(known=LinearDynamics(_A, channel), residual=ZeroResidual(2))
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.1])),
        R=weights,
        Qf=jnp.diag(jnp.array([5.0, 1.0])),
        x_target=jnp.zeros(2),
    )
    support = SupportModel.fit(_LOG_X, 0.4 * jax.random.normal(_KEY_U, (500, 1)))
    rate = LinearConstraint.rate_limit(_HORIZON, [0.5])
    budget = LinearConstraint(np.ones((1, _HORIZON)), -np.inf, 2.0)
    guess = jnp.zeros((_HORIZON, 1))
    for solve in (pessimistic_solve, pessimistic_control):
        solve(
            model, _X0, guess, _DT, cost, support, 0.05, lo, hi, steps=3, constraints=(rate, budget)
        )
    assert len(seen) == 2
    for blocks in seen:
        assert sum(rows.shape[0] for rows, *_ in blocks) == _HORIZON  # the band's 11, the budget
        for rows, *_ in blocks:
            support_of = (np.abs(np.asarray(rows)) > 0.0).astype(int)
            shared = support_of @ support_of.T
            assert np.array_equal(shared, np.diag(np.diag(shared)))
