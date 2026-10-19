"""Zones whose incentives move their neighbours' supply: the check names it, and grouping the zones
estimates the effect a plan that moves them all gets.

The panel: zones in clusters of four, each zone's supply pushed by its own incentive (0.8) and by
the mean of its cluster-mates' (``SPILL``), the incentive chasing the zone's demand, which has a
part the cluster shares. A plan that sets every zone's incentive moves supply by 0.8 + ``SPILL``;
fitted zone by zone, the channel reads the 0.8 alone.
"""

from __future__ import annotations

import json
import logging
import math

import numpy as np
import pytest

from chc.decision import (
    DecisionError,
    Lever,
    Target,
    _cluster_panel,
    _exposure,
    _transitions,
    prescribe,
)
from chc.dynamics_id import _interference
from chc.panel import Panel

DT = 0.1
DIRECT = 0.8
SPILL = -0.4
# The log steps by Euler, x' = (1 + a dt) x + dt b u with a = -0.6; prescribe fits the field whose
# RK4 step reproduces it, whose channel is b log(1 + a dt) / (a dt).
RK4 = math.log(1.0 - 0.6 * DT) / (-0.6 * DT)


def _logs(spill: float, *, seed: int = 0, clusters: int = 30, size: int = 4, periods: int = 10):
    rng = np.random.default_rng(seed)
    n = clusters * size
    cluster = np.repeat(np.arange(clusters), size)
    supply = rng.normal(0.0, 0.2, n)
    rows: dict[str, list] = {
        name: [] for name in ("zone", "time", "group", "supply", "incentive", "demand")
    }
    for period in range(periods):
        demand = rng.normal(0.0, 0.7, clusters)[cluster] + rng.normal(0.0, 0.7, n)
        incentive = 0.9 * demand + rng.normal(0.0, 0.5, n)
        mates = (np.bincount(cluster, weights=incentive)[cluster] - incentive) / (size - 1)
        for name, values in (
            ("zone", np.arange(n)),
            ("time", np.full(n, period)),
            ("group", cluster),
            ("supply", supply),
            ("incentive", incentive),
            ("demand", demand),
        ):
            rows[name].extend(values.tolist())
        drift = -0.6 * supply + DIRECT * incentive + spill * mates + 1.5 * demand
        supply = supply + DT * drift + rng.normal(0.0, 0.01, n)
    return {name: np.asarray(values) for name, values in rows.items()}


def _panel(spill: float, **kwargs) -> Panel:
    return Panel.from_frame(_logs(spill, **kwargs), unit="zone", time="time", seed=0)


def _prescribe(panel: Panel, **kwargs):
    return prescribe(
        panel,
        levers=[Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05)],
        target=Target("supply", value=1.0),
        adjustment=["demand"],
        horizon=10,
        dt=DT,
        **kwargs,
    )


def _channel(result) -> float:
    return float(np.asarray(result.model_fit.residual.channel)[0, 0, 0])


@pytest.fixture(scope="module")
def spilled() -> Panel:
    return _panel(SPILL)


def test_a_spillover_between_neighbours_is_found_and_named(
    spilled: Panel, caplog: pytest.LogCaptureFixture
) -> None:
    """Fitted zone by zone the channel reads the incentive's own push, 0.825 on the fit's RK4
    field, half again the 0.41 a plan that moves every zone gets; the zones' residuals move with
    their cluster-mates' incentives, at p far below 5 %, and the report says what to do."""
    with caplog.at_level(logging.WARNING, logger="chc.decision"):
        result = _prescribe(spilled, neighbours="group")
    check = result.interference
    assert check is not None
    assert check.p_value < 1e-6
    assert check.coefficient[0, 0] < 0.0  # the cluster-mates' incentives pull supply away
    assert any(getattr(record, "chc_event", None) == "interference" for record in caplog.records)
    assert "move with their neighbours' levers" in result.report()
    assert "`clusters`" in result.report()
    assert result.to_json()["certificate"]["interference_p"] == check.p_value
    error = result.model_fit.channel_error
    assert error is not None
    assert abs(_channel(result) - DIRECT * RK4) < 4.0 * error
    assert abs(_channel(result) - (DIRECT + SPILL) * RK4) > 10.0 * error


def test_grouping_the_zones_estimates_the_effect_of_moving_them_together(spilled: Panel) -> None:
    """A group's row is the mean of its zones', whose supply moves by the group's mean incentive
    times 0.8 + SPILL: the channel lands within its error of that, where zone by zone it is more
    than ten errors off."""
    result = _prescribe(spilled, clusters="group")
    error = result.model_fit.channel_error
    assert error is not None
    assert abs(_channel(result) - (DIRECT + SPILL) * RK4) < 4.0 * error
    assert result.run is not None
    assert result.run.clusters == "group"
    assert "clusters=`group`" in result.report()
    record = result.to_json()
    assert record["run"]["clusters"] == "group"
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert result.provenance.n_rows == 30 * 10


def test_zones_whose_incentives_stay_their_own_are_not_reported() -> None:
    """Seeds 1 to 4 read p = 0.83, 0.30, 0.44 and 0.48; seed 0 reads 0.038, as about one seed in
    twenty must at 5 %. The check's size is measured on residuals below."""
    result = _prescribe(_panel(0.0, seed=1), neighbours="group")
    assert result.interference is not None
    assert result.interference.p_value > 0.05
    assert "interference: none found" in result.report()


def test_no_neighbours_runs_no_check(spilled: Panel) -> None:
    result = _prescribe(spilled)
    assert result.interference is None
    assert "interference" not in result.report()
    assert result.to_json()["certificate"]["interference_p"] is None


def test_neighbours_given_as_a_mapping_read_as_the_column_that_groups_them(spilled: Panel) -> None:
    _data, labels = _transitions(spilled, states=("supply",), levers=("incentive",), adjust_for=())
    group = dict(zip(spilled["zone"].tolist(), spilled["group"].tolist(), strict=True))
    mapping = {zone: [o for o, g in group.items() if g == mine] for zone, mine in group.items()}
    by_column = _exposure(spilled, ("incentive",), "group", labels)
    by_mapping = _exposure(spilled, ("incentive",), mapping, labels)
    np.testing.assert_array_equal(by_mapping, by_column)
    assert np.all(np.isfinite(by_column))


def test_the_check_holds_its_size_under_a_shock_every_unit_shares() -> None:
    """Residuals and levers that both move with a shock every unit shares in a period, and with
    nothing of each other's: demeaned within periods, the check rejects at 5 % on 0.047 of 1000
    draws (MC SD 0.0069). Not demeaned, the shared shock would read as a spillover every time."""
    rng = np.random.default_rng(0)
    clusters, size, periods = 30, 4, 10
    n = clusters * size
    cluster = np.repeat(np.arange(clusters), size)
    units, steps = np.tile(np.arange(n), periods), np.repeat(np.arange(periods), n)
    rejected = []
    for _ in range(1000):
        shock = np.repeat(rng.normal(0.0, 1.0, periods), n)
        residual = shock + rng.normal(0.0, 1.0, n * periods)
        lever = (0.9 * shock + rng.normal(0.0, 1.0, n * periods)).reshape(periods, n)
        totals = np.stack([np.bincount(cluster, weights=row)[cluster] for row in lever])
        exposure = ((totals - lever) / (size - 1)).reshape(-1)
        rejected.append(_interference(residual[:, None], exposure, units, steps).p_value <= 0.05)
    assert 0.03 <= float(np.mean(rejected)) <= 0.07


def test_a_cluster_row_is_the_mean_of_its_units_rows_that_period() -> None:
    panel = Panel.from_frame(
        {
            "zone": np.array([0, 1, 2, 0, 1, 2]),
            "time": np.array([0, 0, 0, 1, 1, 1]),
            "group": np.array(["a", "a", "b", "a", "a", "b"]),
            "region": np.array(["r", "r", "s", "r", "r", "s"]),
            "supply": np.array([1.0, 3.0, 5.0, 2.0, 6.0, 7.0]),
        },
        unit="zone",
        time="time",
        cluster="region",
    )
    grouped = _cluster_panel(panel, "group", ("supply",))
    assert grouped.unit == "group"
    assert grouped.cluster == "region"
    assert grouped["group"].tolist() == ["a", "a", "b", "b"]
    assert grouped["time"].tolist() == [0, 1, 0, 1]
    assert grouped["supply"].tolist() == [2.0, 4.0, 5.0, 7.0]


def test_a_label_column_is_carried_to_the_groups_and_refused_where_it_splits_one() -> None:
    """A ``neighbours`` column names the groups' neighbours under ``clusters``, so it is carried as
    a label, not averaged; a group whose zones it puts in two groups has no one neighbourhood."""
    data = {
        "zone": np.array([0, 1, 2, 0, 1, 2]),
        "time": np.array([0, 0, 0, 1, 1, 1]),
        "group": np.array([0, 0, 1, 0, 0, 1]),
        "district": np.array([7, 7, 8, 7, 7, 8]),
        "supply": np.array([1.0, 3.0, 5.0, 2.0, 6.0, 7.0]),
    }
    panel = Panel.from_frame(data, unit="zone", time="time")
    grouped = _cluster_panel(panel, "group", ("supply",), ("district",))
    assert grouped["district"].tolist() == [7, 7, 8, 8]
    split = Panel.from_frame(
        data | {"district": np.array([7, 9, 8, 7, 9, 8])}, unit="zone", time="time"
    )
    with pytest.raises(DecisionError, match="in more than one group of 'district'"):
        _cluster_panel(split, "group", ("supply",), ("district",))


def test_a_declared_cluster_that_splits_a_group_is_dropped() -> None:
    panel = Panel.from_frame(
        {
            "zone": np.array([0, 1, 0, 1]),
            "time": np.array([0, 0, 1, 1]),
            "group": np.array([0, 0, 0, 0]),
            "region": np.array([0, 1, 0, 1]),
            "supply": np.array([1.0, 3.0, 2.0, 6.0]),
        },
        unit="zone",
        time="time",
        cluster="region",
    )
    assert _cluster_panel(panel, "group", ("supply",)).cluster is None


def test_a_unit_in_two_groups_is_refused(spilled: Panel) -> None:
    data = _logs(SPILL)
    data["group"][0] = data["group"][0] + 1  # zone 0's first row in the next cluster
    panel = Panel.from_frame(data, unit="zone", time="time")
    for argument in ("clusters", "neighbours"):
        with pytest.raises(DecisionError, match="more than one group of 'group'"):
            _prescribe(panel, **{argument: "group"})


def test_neighbours_that_name_no_unit_of_the_panel_are_refused(spilled: Panel) -> None:
    with pytest.raises(DecisionError, match="not a unit of the panel"):
        _prescribe(spilled, neighbours={0: [1, 2], 1: ["nowhere"]})


@pytest.mark.parametrize("argument", ["clusters", "neighbours"])
def test_a_grouping_column_the_panel_does_not_have_is_refused(
    spilled: Panel, argument: str
) -> None:
    with pytest.raises(KeyError, match="'district' is not in the panel"):
        _prescribe(spilled, **{argument: "district"})
