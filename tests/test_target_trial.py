"""``Prescription.evaluate`` read as a target trial emulated from a panel (Hernán and Robins,
*Causal Inference: What If*, chapter 22): where each window starts, and what the plan is set
against.

The market has one state, supply, which keeps half of itself a period and takes shocks as large as
its spread, and one lever, an incentive drawn afresh every period. A zone whose supply strays
more than 1.5 from zero is last observed three periods later, the plan's horizon: whether a zone
is still there at a window's end is settled before the window starts, but a zone's windows cut
back from where it left end where it strayed. Every truth is computed here, on the market's own
law, from the starts of the windows each reading takes.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

import chc

DT = 0.1
STEP, CHANNEL, NOISE = 0.5, 0.5, 1.0
PERIODS, HORIZON = 12, 3
STRAY, NOTICE = 1.5, 3


def _market(
    zones: int, seed: int, *, leave: bool = True, periods: int = PERIODS
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """The rows of ``zones`` zones over ``periods`` periods, with each zone's supply and incentive
    by period and the last period it is observed in: the last but for a zone that strayed in time
    to leave before it."""
    rng = np.random.default_rng(seed)
    spread = math.sqrt((NOISE**2 + CHANNEL**2) / (1.0 - STEP**2))
    supply = np.empty((zones, periods))
    supply[:, 0] = rng.normal(0.0, spread, zones)
    incentive = rng.normal(0.0, 1.0, (zones, periods))
    shock = rng.normal(0.0, NOISE, (zones, periods))
    last = np.full(zones, periods - 1)
    for t in range(periods):
        strays = (last == periods - 1) & (np.abs(supply[:, t]) > STRAY)
        if leave and t + NOTICE < periods - 1:
            last[strays] = t + NOTICE
        if t + 1 < periods:
            supply[:, t + 1] = STEP * supply[:, t] + CHANNEL * incentive[:, t] + shock[:, t]
    zone, time = np.nonzero(np.arange(periods) <= last[:, None])
    rows = {
        "zone": zone,
        "time": time,
        "supply": supply[zone, time],
        "incentive": incentive[zone, time],
    }
    return rows, supply, incentive, last


def _starts(last: np.ndarray, time_zero: str) -> list[tuple[np.ndarray, int]]:
    """Each window as the zones that give it and its first period, one entry per period: on the
    calendar, periods 8, 5 and 2 for every zone observed to their window's end; cut back, each
    zone's windows ending at its last period, three before it, and so on."""
    if time_zero == "calendar":
        return [
            (np.flatnonzero(last >= start + HORIZON), start)
            for start in range(PERIODS - 1 - HORIZON, -1, -HORIZON)
        ]
    begins = [last - k * HORIZON for k in range(1, PERIODS // HORIZON + 1)]
    return [
        (np.flatnonzero(begin == start), int(start))
        for begin in begins
        for start in np.unique(begin[begin >= 0])
    ]


def _value(actions: np.ndarray | None, starts: np.ndarray, cost: chc.QuadraticCost) -> float:
    """The expected running cost over the horizon from the law of ``starts``, on the market's own
    law: of the plan's actions, or of the logger's draws when ``actions`` is None."""
    q, r = float(cost.Q[0, 0]), float(cost.R[0, 0])
    targets = np.asarray(cost.targets(HORIZON))[:, 0]
    mean, variance, total = float(starts.mean()), float(starts.var(ddof=1)), 0.0
    for t in range(HORIZON):
        action, draws = (0.0, 1.0) if actions is None else (float(actions[t]), 0.0)
        total += 0.5 * (q * ((mean - targets[t]) ** 2 + variance) + r * (action**2 + draws))
        mean = STEP * mean + CHANNEL * action
        variance = STEP**2 * variance + NOISE**2 + CHANNEL**2 * draws
    return total


@pytest.fixture(scope="module")
def prescription() -> chc.Prescription:
    rows, _, _, _ = _market(400, seed=0, leave=False)
    return chc.prescribe(
        chc.Panel.from_frame(rows, unit="zone", time="time", seed=0),
        levers=[chc.Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.1)],
        target=chc.Target("supply", value=0.5),
        adjustment=chc.CausalGraph.from_edges([("incentive", "supply")]),
        horizon=HORIZON,
        dt=DT,
        tolerance=0.5,
    )


def test_a_zone_that_leaves_on_its_outcome_does_not_set_where_its_windows_start(
    prescription: chc.Prescription,
) -> None:
    """On the calendar the intervals cover the plan's value and its difference from the logger's.
    Cut back from where each zone left, the windows before a zone strayed are those in which it
    had not yet strayed: chosen by their outcomes, they read the plan's cost low, and the interval
    misses it. None of the model's correction is counted (``model_error=0``), so each interval is
    its bootstrap's alone. Over twelve panels of 4000 zones the calendar's covered both truths
    every time, and the cut-back one missed the plan's every time, by 4.6 to 8.1 of its standard
    errors."""
    rows, supply, incentive, last = _market(4000, seed=2000)
    panel = chc.Panel.from_frame(rows, unit="zone", time="time", seed=0)
    plan = prescription.plan
    assert plan is not None
    problem = plan._problem
    assert problem is not None
    actions = np.asarray(plan.actions)[:, 0]
    cost = problem.cost

    calendar = prescription.evaluate(panel, model_error=0.0)
    cut = prescription.evaluate(panel, model_error=0.0, time_zero="unit")

    windows = _starts(last, "calendar")
    starts = np.concatenate([supply[zones, start] for zones, start in windows])
    truth, logged = _value(actions, starts, cost), _value(None, starts, cost)
    target = np.asarray(cost.targets(HORIZON))[:-1, 0]
    q, r = float(cost.Q[0, 0]), float(cost.R[0, 0])
    costs = np.concatenate(
        [
            0.5 * q * (supply[zones, start : start + HORIZON] - target) ** 2
            + 0.5 * r * incentive[zones, start : start + HORIZON] ** 2
            for zones, start in windows
        ]
    )
    comparison, bootstrap = calendar.versus_logger, calendar.bootstrap
    assert comparison is not None
    assert bootstrap is not None
    assert calendar.certificate.samples == sum(zones.size for zones, _ in windows)
    assert cut.certificate.samples == sum(zones.size for zones, _ in _starts(last, "unit"))
    assert bootstrap.units == np.unique(np.concatenate([zones for zones, _ in windows])).size
    assert calendar.interval[0] <= truth <= calendar.interval[1]
    assert comparison.logger_value == pytest.approx(costs.sum(axis=1).mean(), rel=1e-12)
    assert comparison.interval[0] <= truth - logged <= comparison.interval[1]
    cut_starts = np.concatenate([supply[zones, start] for zones, start in _starts(last, "unit")])
    assert cut.interval[1] < _value(actions, cut_starts, cost)


def test_the_draws_of_the_zones_are_the_callers_to_set_and_to_repeat(
    prescription: chc.Prescription,
) -> None:
    rows, _, _, _ = _market(400, seed=1, leave=False)
    panel = chc.Panel.from_frame(rows, unit="zone", time="time", seed=0)

    first = prescription.evaluate(panel, n_resamples=50, seed=3)
    again = prescription.evaluate(panel, n_resamples=50, seed=3)
    other = prescription.evaluate(panel, n_resamples=50, seed=4)

    assert first.bootstrap is not None
    assert (first.bootstrap.units, first.bootstrap.resamples) == (400, 50)
    assert first.interval == again.interval
    assert first.value == other.value
    assert first.interval != other.interval


def test_the_evaluation_refuses_a_time_zero_or_a_count_of_draws_it_cannot_read(
    prescription: chc.Prescription,
) -> None:
    rows, _, _, _ = _market(40, seed=1, leave=False)
    panel = chc.Panel.from_frame(rows, unit="zone", time="time", seed=0)

    with pytest.raises(chc.DecisionError, match="time_zero must be one of"):
        prescription.evaluate(panel, time_zero="first")  # type: ignore[arg-type]
    with pytest.raises(chc.DecisionError, match="needs at least two of them"):
        prescription.evaluate(panel, n_resamples=1)


def test_one_zones_windows_are_read_as_before_and_the_reading_is_deprecated(
    prescription: chc.Prescription, caplog: pytest.LogCaptureFixture
) -> None:
    """One zone's windows leave no zones to resample. Until 1.0 they are read as they were before
    the bootstrap over units, as independent, and every call warns once, and logs it. The readings
    are fe39f1b's ``Prescription.evaluate`` on this panel, before the bootstrap: the smoothing
    chosen under a binding ``min_effective``, and the logger and the smoothing given. Two zones are
    enough to draw."""
    rows, _, _, _ = _market(1, seed=7, leave=False, periods=3601)
    panel = chc.Panel.from_frame(rows, unit="zone", time="time", seed=0)
    logger = chc.AffinePolicy(np.zeros((1, 1)), np.zeros(1), np.eye(1))
    calls = [
        ({"min_effective": 400.0}, 2.36221941740616, (2.078174916606575, 2.646263918205745)),
        (
            {"logger": logger, "smoothing": 0.5},
            2.3818807810249143,
            (2.085154423058448, 2.6786071389913806),
        ),
    ]

    for keywords, value, interval in calls:
        caplog.clear()
        with (
            caplog.at_level(logging.INFO, logger="chc.decision"),
            pytest.warns(
                DeprecationWarning,
                match=r"treats the windows as independent and is too narrow\. From 1\.0 a panel of "
                "one unit raises DecisionError; evaluate on a panel of several units instead",
            ) as caught,
        ):
            reading = prescription.evaluate(panel, model_error=0.5, **keywords)
        records = [r for r in caplog.records if getattr(r, "chc_event", "") == "one_unit"]

        assert [(w.category, w.filename) for w in caught] == [(DeprecationWarning, __file__)]
        assert [(r.levelno, r.getMessage(), r.windows) for r in records] == [
            (logging.WARNING, str(caught[0].message), 1200)
        ]
        assert reading.bootstrap is None
        assert reading.versus_logger is None
        assert reading.logger_check is not None
        assert reading.certificate.samples == 1200
        assert reading.value == pytest.approx(value, rel=1e-9)
        assert reading.interval == pytest.approx(interval, rel=1e-9)

    two, _, _, _ = _market(2, seed=7, leave=False, periods=1801)
    drawn = prescription.evaluate(
        chc.Panel.from_frame(two, unit="zone", time="time", seed=0), n_resamples=2
    )
    assert drawn.bootstrap is not None
    assert drawn.bootstrap.units == 2
