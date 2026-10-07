"""A plan keeps to what the log did along the directions it never moved (ADR 0054).

The logs are a review's: one state ``y``, two levers, and a confounder ``z`` the graph adjusts for,
with ``y`` moving by ``0.1 (-0.5 y + 0.8 u1 + 0.1 u2 + 1.5 z)`` a period. Each policy leaves some
direction of the channel unmoved, and each test reads the plan in the worlds the log cannot tell
apart: they give one path under the plan, or the plan is refused.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from chc.cost import QuadraticCost, total_cost
from chc.decision import (
    DecisionError,
    Lever,
    NotIdentifiedError,
    Prescription,
    Target,
    prescribe,
)
from chc.graph import CausalGraph
from chc.integrate import rollout
from chc.mpc import PeriodBudget
from chc.panel import Panel
from chc.plan import _Ruled

DT = 0.1
EDGES = [("z", "y"), ("u1", "y"), ("u2", "y"), ("z", "u1"), ("z", "u2")]


@functools.cache
def _panel(policy: str, scale: float = 1.0) -> Panel:
    """A hundred units of fifteen periods under ``policy``, the levers logged in units ``scale``
    times their own."""
    rng = np.random.default_rng(303)
    rows: dict[str, list[float]] = {name: [] for name in ("unit", "time", "y", "u1", "u2", "z")}
    for unit in range(100):
        y = rng.normal()
        for period in range(15):
            z, noise = rng.normal(), rng.normal()
            if policy == "together":  # u2 = 2 u1, u1 moved by the confounder and its own draw
                u1 = 0.9 * z - 0.3 * y + 0.5 * noise
                u2 = 2.0 * u1
            elif policy == "thrice":  # u2 = 3 u1, whose weights no float holds in ratio
                u1 = 0.9 * z - 0.3 * y + 0.5 * noise
                u2 = 3.0 * u1
            elif policy == "together_offset":
                u1 = 0.9 * z - 0.3 * y + 0.5 * noise
                u2 = 2.0 * u1 + 0.4
            elif policy == "state":  # u1 set from the state alone
                u1, u2 = -0.3 * y, 0.6 * noise
            elif policy == "state_squared":
                u1, u2 = 0.1 * y * y, 0.6 * noise
            elif policy == "confounder":  # u1 set from a column outside the state
                u1, u2 = 0.7 * z, 0.6 * noise
            elif policy == "combination_of_state":  # u2 - 2 u1 set from the state
                u1 = 0.9 * z + 0.5 * noise
                u2 = 2.0 * u1 - 0.3 * y
            elif policy == "combination_of_confounder":  # u2 - 2 u1 set from outside the state
                u1 = 0.9 * z - 0.3 * y + 0.5 * noise
                u2 = 2.0 * u1 + 0.7 * z
            else:  # "free": every direction moved
                u1, u2 = 0.9 * z + 0.5 * rng.normal(), 0.6 * noise
            for name, value in zip(rows, (unit, period, y, scale * u1, scale * u2, z), strict=True):
                rows[name].append(value)
            y += DT * (-0.5 * y + 0.8 * u1 + 0.1 * u2 + 1.5 * z) + rng.normal(scale=0.01)
    return Panel.from_frame(
        {name: np.asarray(values) for name, values in rows.items()}, unit="unit", time="time"
    )


def _prescribe(policy: str, *, scale: float = 1.0, **kwargs: object) -> Prescription:
    levers = kwargs.pop(
        "levers",
        [
            Lever("u1", lo=-2.0 * scale, hi=2.0 * scale, unit_cost=0.01 / scale**2),
            Lever("u2", lo=-2.0 * scale, hi=2.0 * scale, unit_cost=1.0 / scale**2),
        ],
    )
    target = kwargs.pop("target", Target("y", value=1.0))
    return prescribe(
        _panel(policy, scale),
        levers=levers,  # type: ignore[arg-type]
        target=target,  # type: ignore[arg-type]
        adjustment=CausalGraph.from_edges(EDGES),
        horizon=3,
        dt=DT,
        tolerance=0.5,
        x0=jnp.ones(1),
        **kwargs,  # type: ignore[arg-type]
    )


def _path(schedule: np.ndarray, b1: float, b2: float, a: float = -0.5) -> np.ndarray:
    """The world's own path from ``y = 1`` under the schedule, Euler as the log was made."""
    path = [1.0]
    for u1, u2 in schedule:
        path.append(path[-1] + DT * (a * path[-1] + b1 * u1 + b2 * u2))
    return np.asarray(path)


def test_levers_the_log_moved_together_stay_together() -> None:
    """The log has ``u2 = 2 u1`` in every row. 0.13's plan moved them apart, to ``u1 = 2`` and
    ``u2 = 0.10``, along a direction the channel's moment never saw, and certified 3 steps; now an
    equality row keeps them on the log's line, and two worlds the log cannot tell apart, the
    channels ``[0.8, 0.1]`` and ``[0.2, 0.4]``, run one path under the plan."""
    result = _prescribe("together")
    certificate = result.certificate
    assert (certificate.estimability, certificate.first_loaded_step) == ("held_to_log", None)
    assert (certificate.identification_rank, certificate.unmoved_directions) == (2, 2)
    (relation,) = certificate.relations
    np.testing.assert_allclose(relation.weights, np.array([2.0, -1.0]) / np.sqrt(5.0), atol=1e-12)
    assert relation.level == 0.0
    schedule = np.asarray(result.schedule.magnitudes)
    np.testing.assert_allclose(schedule[:, 1], 2.0 * schedule[:, 0], rtol=0.0, atol=1e-9)
    assert np.max(np.abs(schedule)) > 1e-3  # the plan moves the line, not nothing
    np.testing.assert_allclose(
        _path(schedule, 0.8, 0.1), _path(schedule, 0.2, 0.4), rtol=0.0, atol=1e-12
    )
    assert "- kept at the level the log kept them: 0.8944 `u1` - 0.4472 `u2` = 0" in (
        result.report()
    )
    assert result.to_json()["certificate"]["relations"] == [
        {"weights": list(relation.weights), "level": 0.0}
    ]


def test_a_relation_kept_away_from_zero_is_kept_at_its_level() -> None:
    result = _prescribe("together_offset")
    assert result.certificate.estimability == "held_to_log"
    schedule = np.asarray(result.schedule.magnitudes)
    np.testing.assert_allclose(schedule[:, 1] - 2.0 * schedule[:, 0], 0.4, rtol=0.0, atol=1e-9)


def test_a_relation_the_boxes_cannot_reach_is_held_where_they_reach_and_leaves_the_log() -> None:
    """With ``u1`` in ``[0.5, 2]`` and ``u2`` in ``[-2, 0]``, ``u2 - 2 u1`` reaches no higher than
    -1, short of the log's 0.4: the plan holds it at -1, and from its first step the worlds the log
    cannot tell apart part."""
    levers = [Lever("u1", lo=0.5, hi=2.0, unit_cost=0.01), Lever("u2", lo=-2.0, hi=0.0)]
    result = _prescribe("together_offset", levers=levers)
    certificate = result.certificate
    schedule = np.asarray(result.schedule.magnitudes)
    np.testing.assert_allclose(schedule[:, 1] - 2.0 * schedule[:, 0], -1.0, rtol=0.0, atol=1e-9)
    assert (certificate.estimability, certificate.first_loaded_step) == ("not_estimable", 0)
    assert certificate.trustworthy_steps == 0


def test_a_relation_kept_at_zero_to_rounding_is_kept_at_zero() -> None:
    """``u2 = 3 u1``: the level reads zero to rounding, so ``max_levers`` may hold the unselected
    lever at zero, which keeps the relation."""
    result = _prescribe("thrice", max_levers=1)
    (relation,) = result.certificate.relations
    assert relation.level == 0.0
    assert result.certificate.estimability == "held_to_log"


def test_max_levers_cannot_hold_a_relation_kept_away_from_zero() -> None:
    with pytest.raises(DecisionError, match="max_levers holds an unselected lever at zero"):
        _prescribe("together_offset", max_levers=1)


@pytest.mark.parametrize(
    ("policy", "rule"),
    [("state", lambda y: -0.3 * y), ("state_squared", lambda y: 0.1 * y * y)],
    ids=["linear", "quadratic"],
)
def test_a_lever_set_from_the_state_follows_that_rule(policy, rule) -> None:
    """0.13 held ``u1`` at its mean, 0.0074, where the logger set it from the state: a level the
    log never ran, whose push the log does not fix. Now ``u1`` follows the logger's rule in the
    plan's field and in the schedule, read along the predicted path."""
    result = _prescribe(policy)
    certificate = result.certificate
    assert (certificate.estimability, certificate.first_loaded_step) == ("held_to_log", None)
    assert certificate.rule_levers == certificate.unmoved_levers == ("u1",)
    assert certificate.trustworthy_steps == 3
    schedule = result.schedule
    assert schedule.rules == ("u1",)
    assert result.plan is not None
    path = np.asarray(result.plan.trajectory)[:-1, 0]
    np.testing.assert_allclose(np.asarray(schedule.magnitudes)[:, 0], rule(path), atol=1e-9)
    assert "- set by the log from the state alone, so they follow that rule: `u1`" in (
        result.report()
    )
    assert "from the state as it comes" in result.report()
    assert result.to_json()["certificate"]["rule_levers"] == ["u1"]


def test_worlds_the_logged_rule_leaves_apart_run_one_path() -> None:
    """``A = -0.74 + 0.3 k`` and ``B1 = k`` make the same log for every ``k``: under the logged rule
    ``u1 = -0.3 y`` each world's rate is ``-0.74 y + 0.1 u2``, and a level would make it depend on
    ``k``."""
    schedule = np.asarray(_prescribe("state").schedule.magnitudes)
    paths = []
    for k in (0.0, 0.8, 2.0):
        y = [1.0]
        for _, u2 in schedule:
            y.append(y[-1] + DT * ((-0.74 + 0.3 * k) * y[-1] + k * (-0.3 * y[-1]) + 0.1 * u2))
        paths.append(y)
    np.testing.assert_allclose(paths[1], paths[0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(paths[2], paths[0], rtol=0.0, atol=1e-12)


def test_the_test_reads_the_same_in_any_units() -> None:
    """The levers logged in millionths: the rule is followed, and no fit the log cannot tell apart
    is read as moving the path."""
    result = _prescribe("state", scale=1e6)
    assert result.certificate.estimability == "held_to_log"
    assert result.plan is not None
    path = np.asarray(result.plan.trajectory)[:-1, 0]
    np.testing.assert_allclose(
        np.asarray(result.schedule.magnitudes)[:, 0], -0.3e6 * path, rtol=1e-9, atol=0.0
    )


def test_a_rule_that_leaves_its_box_leaves_the_log_from_that_step(caplog) -> None:
    """``u1``'s box stops at -0.27, which the rule ``-0.3 y`` crosses as ``y`` falls through 0.9:
    inside the second step, before the third starts. A step holds ``u1`` at the rule's level at the
    state it starts from, as the log did, so the second step is one of the log's own; the third
    starts below 0.9 and holds ``u1`` at -0.27, where the log never did, and the trustworthy prefix
    ends. 0.14 read the rule inside the step and ended it a step early."""
    levers = [Lever("u1", lo=-2.0, hi=-0.27, unit_cost=0.01), Lever("u2", lo=-2.0, hi=2.0)]
    with caplog.at_level(logging.WARNING, logger="chc.decision"):
        result = _prescribe("state", levers=levers)
    certificate = result.certificate
    assert result.plan is not None
    path = np.asarray(result.plan.trajectory)[:, 0]
    assert path[1] > 0.9 > path[2]
    held = np.asarray(result.schedule.magnitudes)[:, 0]
    assert held[1] == pytest.approx(-0.3 * path[1], rel=1e-12, abs=0.0)
    assert held[2] == -0.27
    assert (certificate.estimability, certificate.first_loaded_step) == ("not_estimable", 2)
    assert certificate.trustworthy_steps <= 2
    assert "a fit the log cannot tell apart predicts another path from step 2" in result.report()
    (event,) = [r for r in caplog.records if getattr(r, "chc_event", None) == "estimability"]
    assert event.first_loaded_step == 2


def test_a_lever_set_from_outside_the_state_gives_no_schedule() -> None:
    result = _prescribe("confounder")
    certificate = result.certificate
    assert result.plan is None
    assert (certificate.identification, certificate.estimability) == (
        "not_identified",
        "not_estimable",
    )
    assert "the log set `u1` from `z`, outside the plan's state" in certificate.adjustment.reason
    with pytest.raises(NotIdentifiedError, match="leave them out of the levers"):
        _ = result.schedule


def test_a_combination_set_from_the_state_gives_no_schedule() -> None:
    """``u2 - 2 u1 = -0.3 y``: neither lever is unmoved alone, and their combination follows the
    state, which no row of the plan's actions can."""
    result = _prescribe("combination_of_state")
    certificate = result.certificate
    assert result.plan is None
    assert certificate.estimability == "not_estimable"
    assert certificate.unmoved_levers == ()
    assert (
        "the log set 0.8944 `u1` - 0.4472 `u2` from the state, which no row of the plan's "
        "actions holds"
    ) in certificate.adjustment.reason


def test_a_combination_set_from_outside_the_state_gives_no_schedule() -> None:
    """``u2 - 2 u1 = 0.7 z``: the plan reads no ``z``, so no plan can keep what the log did."""
    result = _prescribe("combination_of_confounder")
    certificate = result.certificate
    assert result.plan is None
    assert certificate.estimability == "not_estimable"
    assert certificate.unmoved_levers == ()
    reason = certificate.adjustment.reason
    assert "from columns outside the plan's state, which no plan reads" in reason
    assert "`u1`" in reason
    assert "`u2`" in reason


def test_a_log_that_moved_every_direction_is_estimable() -> None:
    result = _prescribe("free")
    certificate = result.certificate
    assert certificate.estimability == "estimable"
    assert (certificate.identification_rank, certificate.unmoved_directions) == (4, 0)
    assert (certificate.relations, certificate.rule_levers, certificate.first_loaded_step) == (
        (),
        (),
        None,
    )
    assert result.schedule.rules == ()
    assert result.plan is not None
    np.testing.assert_array_equal(result.schedule.magnitudes, result.plan.actions)
    assert "- estimability: **estimable**, the log moved 4 of the channel's 4 directions" in (
        result.report()
    )


def test_a_ruled_lever_takes_no_cap_and_no_budget_and_no_evaluation() -> None:
    capped = [Lever("u1", lo=-2.0, hi=2.0, cap_per_step=0.1), Lever("u2", lo=-2.0, hi=2.0)]
    with pytest.raises(DecisionError, match="no cap on their steps can hold"):
        _prescribe("state", levers=capped)
    budget = PeriodBudget(weights=np.array([1.0, 1.0]), amount=10.0, period=3)
    with pytest.raises(DecisionError, match=r"a budget prices \['u1'\]"):
        _prescribe("state", budgets=[budget])
    with pytest.raises(DecisionError, match="open-loop schedule cannot carry"):
        _prescribe("state").evaluate(_panel("state"))


def _priced(unit_cost: float, level: float | tuple[float, ...] = 1.0) -> Prescription:
    levers = [
        Lever("u1", lo=-2.0, hi=2.0, unit_cost=unit_cost),
        Lever("u2", lo=-2.0, hi=2.0, unit_cost=1.0),
    ]
    return _prescribe("state", levers=levers, target=Target("y", value=level))


@pytest.mark.parametrize("level", [1.0, (1.0, 0.9, 0.8)], ids=["a level", "a path"])
def test_a_ruled_lever_is_priced_at_the_level_it_takes(level: float | tuple[float, ...]) -> None:
    """0.14 priced ``u1`` at its column of the plan, which nothing reads and which sits at its mean
    logged level, 0.028, where the field takes the rule's ``-0.3 y``: at a unit cost of 1000 the
    plan reported 1.255 for a schedule that costs 115.46. The reported cost is now the schedule's,
    summed by hand; a path's start repeats its first level."""
    result = _priced(1000.0, level)
    assert result.plan is not None
    path = np.asarray(result.plan.trajectory)[:, 0]
    schedule = np.asarray(result.schedule.magnitudes)
    goal = np.broadcast_to(level, 3)
    goal = np.concatenate([goal[:1], goal])
    by_hand = 0.5 * np.sum((path - goal) ** 2) + 0.5 * np.sum(schedule**2 @ np.array([1000.0, 1.0]))
    assert result.plan.task_cost == pytest.approx(by_hand, rel=1e-12, abs=0.0)


def test_a_dear_ruled_lever_moves_the_free_one() -> None:
    """Under ``u1 = -0.3 y`` a lower ``y`` sets less of ``u1``: at a unit cost of 1000 the plan
    pushes ``y`` down with ``u2`` and pays 114.15, where the plan made at 0.01 would pay 115.46.
    0.14 made one plan at both prices."""
    cheap, dear = _priced(0.01), _priced(1000.0)
    assert cheap.plan is not None
    assert dear.plan is not None
    problem = dear.plan._problem
    assert problem is not None
    assert (
        np.asarray(dear.schedule.magnitudes)[0, 1] < np.asarray(cheap.schedule.magnitudes)[0, 1] - 1
    )
    kept = total_cost(problem.model, problem.x0, cheap.plan.actions, problem.dt, problem.cost)
    assert dear.plan.task_cost < float(kept) - 1.0


def test_nothing_reads_a_ruled_lever_s_column() -> None:
    """The plan holds ``u1``'s column at its mean logged level only so it has one: moved to 1.7
    with its box, it moves neither the plan's cost nor what an error in the channel costs the plan.
    0.14 priced the column, and weighed an error in ``u1``'s channel at it."""
    plan = _prescribe("state").plan
    assert plan is not None
    problem = plan._problem
    assert problem is not None
    moved = replace(
        plan,
        actions=plan.actions.at[:, 0].set(1.7),
        _problem=replace(
            problem,
            u_lo=jnp.asarray(problem.u_lo).at[..., 0].set(1.7),
            u_hi=jnp.asarray(problem.u_hi).at[..., 0].set(1.7),
        ),
    )
    cost = total_cost(problem.model, problem.x0, moved.actions, problem.dt, problem.cost)
    assert float(cost) == pytest.approx(plan.task_cost, rel=1e-12, abs=0.0)
    np.testing.assert_allclose(
        moved.decision_weight().matrix, plan.decision_weight().matrix, rtol=1e-12, atol=0.0
    )


@pytest.mark.parametrize("policy", ["state", "state_squared"])
def test_a_ruled_lever_is_held_over_each_step_as_the_log_held_it(policy: str) -> None:
    """The log set ``u1`` at each period's start and held it over the period, and the fit reads a
    step so. 0.14 read the rule at every point RK4 reads in a step, so the level moved with the
    state inside the step. The plan's path and its cost are now the schedule's, each level held
    over its step, under the fitted field without the rule."""
    result = _prescribe(policy)
    plan = result.plan
    assert plan is not None
    problem = plan._problem
    assert problem is not None
    assert isinstance(problem.model, _Ruled)
    field, schedule = problem.model.dynamics, jnp.asarray(result.schedule.magnitudes)
    held = rollout(field, problem.x0, schedule, problem.dt)
    np.testing.assert_allclose(np.asarray(plan.trajectory), np.asarray(held), rtol=1e-12, atol=0.0)
    price = QuadraticCost(problem.cost.Q, problem.cost.R, problem.cost.Qf, problem.cost.x_target)
    cost = total_cost(field, problem.x0, schedule, problem.dt, price)
    assert float(cost) == pytest.approx(plan.task_cost, rel=1e-12, abs=0.0)
