"""``max_levers``: which levers to use, by greedy forward selection with the plan as its inner loop.

The load-bearing test is the exhaustive one. On a three-lever plant every subset is planned on its
own, through the public ``prescribe`` with the other levers' boxes pinned to zero, and greedy must
land on the cheapest set at every size -- at the same planned cost to the bit, because each of its
candidates is exactly that pinned plan. Greedy is not exhaustive in general;
``docs/adr/0004-greedy-lever-selection.md`` has an instance where it misses.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import logging

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import chc.decision as decision
from chc.decision import Constraint, DecisionError, Lever, Prescription, Target, prescribe
from chc.graph import CausalGraph
from chc.panel import Panel
from chc.plan import CausalPlan

DT = 0.1
LEVERS = ("incentive", "boost", "calm")
EDGES = [
    ("demand", "incentive"),  # the one confounded lever, as in test_decision.py
    ("demand", "supply"),
    ("incentive", "supply"),
    ("incentive", "wait"),
    ("supply", "wait"),
    ("boost", "supply"),
    ("calm", "wait"),
]


def _logs(n_units: int = 120, n_periods: int = 10, seed: int = 0) -> dict[str, np.ndarray]:
    """A driver pool with three levers: one that lifts supply and drains wait, one that only lifts
    supply, harder, and one that only drains wait. Only the first chases demand."""
    rng = np.random.default_rng(seed)
    columns = ("unit", "time", "supply", "wait", "demand", *LEVERS)
    rows: dict[str, list[float]] = {name: [] for name in columns}
    for unit in range(n_units):
        supply, wait = rng.normal(0.0, 0.2), rng.normal(0.0, 0.2)
        for period in range(n_periods):
            demand = rng.normal(0.0, 1.0)
            incentive = 0.9 * demand + rng.normal(0.0, 0.5)
            boost, calm = rng.normal(0.0, 0.5, 2)
            for name, value in zip(
                columns, (unit, period, supply, wait, demand, incentive, boost, calm), strict=True
            ):
                rows[name].append(value)
            supply_next = supply + DT * (
                -0.6 * supply + 0.3 * wait + 0.8 * incentive + 1.2 * boost + 1.5 * demand
            )
            wait_next = wait + DT * (0.25 * wait - 0.4 * incentive - 0.6 * calm)
            supply = supply_next + rng.normal(0.0, 0.01)
            wait = wait_next + rng.normal(0.0, 0.01)
    return {name: np.asarray(values) for name, values in rows.items()}


PANEL = Panel.from_frame(_logs(), unit="unit", time="time", seed=0)
GRAPH = CausalGraph.from_edges(EDGES)


def _levers(keep: tuple[str, ...] = LEVERS) -> list[Lever]:
    """Every lever in ``[-2, 2]``; a lever not in ``keep`` is pinned to ``[0, 0]``."""
    return [
        Lever(
            name, lo=-2.0 if name in keep else 0.0, hi=2.0 if name in keep else 0.0, unit_cost=0.05
        )
        for name in LEVERS
    ]


def _prescribe(max_levers: int | None, levers: list[Lever] | None = None, **kwargs: object):
    return prescribe(
        PANEL,
        levers=levers if levers is not None else _levers(),
        target=Target("supply", value=1.0),
        adjustment=GRAPH,
        horizon=12,
        dt=DT,
        max_levers=max_levers,
        **kwargs,  # type: ignore[arg-type]
    )


def _without_selection(result: Prescription) -> str:
    payload = result.to_json()
    del payload["selection"]
    return json.dumps(payload, sort_keys=True)


def test_greedy_lands_on_the_cheapest_set_at_every_size() -> None:
    """Exhaustive search over every subset, each planned alone with the rest pinned at zero."""
    greedy = _prescribe(max_levers=len(LEVERS))
    assert greedy.selection is not None
    for size in range(1, len(LEVERS) + 1):
        costs = {
            subset: _prescribe(None, levers=_levers(subset)).plan.task_cost
            for subset in itertools.combinations(LEVERS, size)
        }
        best = min(costs, key=costs.__getitem__)
        steps = greedy.selection.steps[:size]
        assert set(greedy.selection.selected[:size]) == set(best)
        assert steps[-1].task_cost == costs[best]  # the same plan, so the same cost to the bit


def test_selecting_every_lever_is_the_unrestricted_prescription() -> None:
    """The last step plans with every lever, cold, on their own boxes: ``None``'s one solve."""
    unrestricted = _prescribe(None)
    for cap in (len(LEVERS), len(LEVERS) + 2):
        selected = _prescribe(max_levers=cap)
        assert selected.selection is not None
        assert len(selected.selection.steps) == len(LEVERS)
        assert selected.plan is not None
        assert unrestricted.plan is not None
        assert np.array_equal(
            np.asarray(selected.plan.actions), np.asarray(unrestricted.plan.actions)
        )
        assert _without_selection(selected) == _without_selection(unrestricted)


def test_without_max_levers_the_plan_is_the_one_solve_it_always_was(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``None`` does not route through the selection: one cold solve on the levers' own boxes.

    That is what keeps it bit-identical to the code before ``max_levers`` existed, which was checked
    against main by ``scripts/bench_max_levers.py fingerprint``; this pins the shape that makes it
    so, and the count of solves the selection costs instead.
    """
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    real = decision.causal_plan

    def spy(*args: object, **kwargs: object):
        calls.append((args, kwargs))
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(decision, "causal_plan", spy)
    result = _prescribe(None)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert np.array_equal(np.asarray(args[5]), [-2.0, -2.0, -2.0])
    assert np.array_equal(np.asarray(args[6]), [2.0, 2.0, 2.0])
    assert "warm_start" not in kwargs
    assert result.selection is None
    assert result.to_json()["selection"] is None

    calls.clear()
    _prescribe(max_levers=2)
    assert len(calls) == 3 + 2  # every unchosen lever once per step


def test_an_unselected_lever_is_held_at_zero_throughout() -> None:
    result = _prescribe(max_levers=1)
    assert result.selection is not None
    (chosen,) = result.selection.selected
    magnitudes = np.asarray(result.schedule.magnitudes)
    windows = result.schedule.windows()
    for index, name in enumerate(LEVERS):
        if name == chosen:
            assert windows[name] is not None
        else:
            assert np.all(magnitudes[:, index] == 0.0)  # exactly: pinned, not merely small
            assert windows[name] is None


def test_a_tie_goes_to_the_lever_listed_first() -> None:
    """Two levers the caller pinned to zero plan alike wherever they are added: a tie by design."""
    result = _prescribe(max_levers=2, levers=_levers(("boost",)))
    assert result.selection is not None
    assert result.selection.selected == ("boost", "incentive")
    first, second = result.selection.steps
    assert second.task_cost == first.task_cost  # the step bought nothing, and says so


def test_each_step_is_priced_by_the_certificate() -> None:
    """A step's regret bound is priced against every lever's box, so it prices what was left out.

    Priced against the step's own box it would certify only the solve, and would fall below what
    the levers left out demonstrably buy -- the drop to the plan that uses them all.
    """
    result = _prescribe(max_levers=len(LEVERS))
    assert result.selection is not None
    assert result.plan is not None
    steps = result.selection.steps
    final = steps[-1]
    assert final.task_cost == result.plan.task_cost
    assert final.regret_bound == result.certificate.regret_bound
    idle = _prescribe(None, levers=_levers(())).plan
    assert idle is not None
    assert result.selection.idle_cost == idle.task_cost
    before = result.selection.idle_cost
    for step in steps:
        assert step.task_cost <= before + step.regret_bound  # an added lever can stay at zero
        assert step.regret_bound >= step.task_cost - final.task_cost
        before = step.task_cost


def test_held_constraints_rank_a_lever_that_cannot_hold_them_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``wait <= 0.5`` from ``wait = 0.45``, where wait's own drift breaks it at the first step.

    ``boost`` moves wait only through the fit's noise, so its plan cannot hold the bound, and the
    rounds that try make that plan dear enough for cost alone to pass it over. So its plan is made
    free here, its audit untouched. Priced only, it is chosen; held, the audit ranks first, and of
    the levers whose plans clear it the cheaper is chosen.
    """
    real = decision.causal_plan

    def boost_for_free(*args: object, **kwargs: object) -> CausalPlan:
        plan = real(*args, **kwargs)  # type: ignore[arg-type]
        alone = (np.asarray(args[6]) > np.asarray(args[5])).tolist() == [False, True, False]
        return dataclasses.replace(plan, task_cost=0.0) if alone else plan

    monkeypatch.setattr(decision, "causal_plan", boost_for_free)
    kwargs = {"constraints": [Constraint("wait", hi=0.5)], "x0": jnp.array([0.0, 0.45])}
    priced = _prescribe(max_levers=1, **kwargs)
    held = _prescribe(max_levers=1, hold_constraints=True, **kwargs)
    assert priced.selection is not None
    assert held.selection is not None
    assert priced.selection.selected == ("boost",)
    assert priced.certificate.barrier_certified_steps == 0
    assert held.selection.selected == ("incentive",)
    assert held.certificate.barrier_certified_steps == 12


def test_held_constraints_rank_candidates_on_the_audit_the_certificate_reports() -> None:
    """From the midpoint of a two-sided supply bound both margins are the minimum, and a held solve
    enforces only the first there, so ``incentive``, the cheaper route to the target, leaves through
    ``hi`` at the first step. The certificate audits every tied margin and clears no step of that
    plan; ranked on the solve's own audit, greedy kept it anyway."""
    boxes = {"incentive": 2.0, "boost": 2.0, "calm": 0.3}

    def run(keep: tuple[str, ...], max_levers: int | None) -> Prescription:
        levers = [
            Lever(
                name,
                lo=-box if name in keep else 0.0,
                hi=box if name in keep else 0.0,
                unit_cost=0.05,
            )
            for name, box in boxes.items()
        ]
        return prescribe(
            PANEL,
            levers=levers,
            target=Target("wait", value=-1.0),
            constraints=[Constraint("supply", lo=-0.2, hi=0.2)],
            hold_constraints=True,
            adjustment=GRAPH,
            horizon=12,
            dt=DT,
            x0=[0.0, 0.0],
            max_levers=max_levers,
        )

    incentive, calm = run(("incentive",), None), run(("calm",), None)
    assert incentive.plan is not None
    assert calm.plan is not None
    assert incentive.plan.task_cost < calm.plan.task_cost
    assert incentive.certificate.barrier_certified_steps == 0
    assert calm.certificate.barrier_certified_steps == 12

    selected = run(tuple(boxes), max_levers=1)
    assert selected.selection is not None
    assert selected.selection.selected == ("calm",)
    assert selected.certificate.barrier_certified_steps == 12


def test_a_max_levers_that_cannot_be_honoured_is_refused() -> None:
    with pytest.raises(DecisionError, match="selects no lever"):
        _prescribe(max_levers=0)
    floored = [Lever("incentive", lo=0.5, hi=2.0), *_levers()[1:]]
    with pytest.raises(DecisionError, match=r"'incentive' has box \[0.5, 2.0\], which excludes 0"):
        _prescribe(max_levers=1, levers=floored)
    assert _prescribe(None, levers=floored).selection is None  # the box is fine without a cap


def test_each_selection_step_leaves_one_structured_record(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="chc.decision"):
        result = _prescribe(max_levers=2)
    ours = [record for record in caplog.records if record.name == "chc.decision"]
    events = [str(getattr(record, "chc_event", "")) for record in ours]
    assert events == [
        "adjustment",
        "logger_check",
        "fit",
        "selection",
        "selection",
        "plan",
        "estimability",
        "certificate",
    ]
    assert result.selection is not None
    records = [record for record in ours if getattr(record, "chc_event", "") == "selection"]
    for number, (record, step) in enumerate(zip(records, result.selection.steps, strict=True), 1):
        assert getattr(record, "step", None) == number
        assert getattr(record, "lever", None) == step.lever
        assert getattr(record, "task_cost", None) == step.task_cost
        assert getattr(record, "regret_bound", None) == step.regret_bound
        assert getattr(record, "regret_status", None) == step.regret_status == "diagnostic"
        candidates = getattr(record, "candidates", {})
        assert len(candidates) == len(LEVERS) - number + 1
        assert candidates[step.lever]["task_cost"] == min(
            c["task_cost"] for c in candidates.values()
        )
        assert all(c["cleared_steps"] is None for c in candidates.values())  # nothing held
        assert getattr(record, "descent_steps", 0) > 0


def test_the_report_lists_the_levers_kept_in_the_order_greedy_added_them() -> None:
    selected = _prescribe(max_levers=2)
    assert selected.selection is not None
    rows = [line for line in selected.report().splitlines() if line.startswith(("| 1 |", "| 2 |"))]
    assert [row.split("|")[2].strip() for row in rows] == [
        f"`{lever}`" for lever in selected.selection.selected
    ]
    assert "max_levers" not in _prescribe(None).report()


def test_the_selection_travels_in_the_json() -> None:
    result = _prescribe(max_levers=2)
    payload = json.loads(json.dumps(result.to_json()))
    assert payload["schema_version"] == 2  # a number that is not finite reads null (ADR 0055)
    assert result.selection is not None
    assert payload["selection"] == {
        "idle_cost": result.selection.idle_cost,
        "steps": [dataclasses.asdict(step) for step in result.selection.steps],
    }
    assert payload["levers"] == list(LEVERS)  # the schedule keeps a column per lever


# ---- properties: whatever the boxes, prices and target ----


@given(
    los=st.lists(st.floats(-2.0, 0.0), min_size=3, max_size=3),
    his=st.lists(st.floats(0.1, 2.0), min_size=3, max_size=3),
    prices=st.lists(st.floats(0.01, 0.5), min_size=3, max_size=3),
    goal=st.floats(-1.5, 1.5),
    cap=st.integers(min_value=1, max_value=2),
)
@settings(deadline=None, max_examples=8, suppress_health_check=[HealthCheck.too_slow])
def test_the_cap_bounds_the_set_and_a_larger_cap_never_costs_more_than_the_certificate_allows(
    los: list[float], his: list[float], prices: list[float], goal: float, cap: int
) -> None:
    """At most ``cap`` levers, the rest exactly zero; the path is the same whatever the cap, so a
    larger cap only extends it; and the planned cost never rises by more than the regret bound of
    the plan that added a lever -- by exactly nothing, were the solves exact."""
    levers = [
        Lever(name, lo=lo, hi=hi, unit_cost=price)
        for name, lo, hi, price in zip(LEVERS, los, his, prices, strict=True)
    ]

    def select(max_levers: int) -> Prescription:
        return prescribe(
            PANEL,
            levers=levers,
            target=Target("supply", value=goal),
            adjustment=GRAPH,
            horizon=8,
            dt=DT,
            max_levers=max_levers,
        )

    small, large = select(cap), select(cap + 1)
    assert small.selection is not None
    assert large.selection is not None
    assert len(small.selection.selected) == cap
    assert large.selection.steps[:cap] == small.selection.steps
    magnitudes = np.asarray(small.schedule.magnitudes)
    for index, name in enumerate(LEVERS):
        if name not in small.selection.selected:
            assert np.all(magnitudes[:, index] == 0.0)
    assert large.plan is not None
    assert small.plan is not None
    added = large.selection.steps[cap]
    assert large.plan.task_cost <= small.plan.task_cost + added.regret_bound
