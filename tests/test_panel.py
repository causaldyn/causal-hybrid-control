"""A long frame becomes a checked panel, and the pivot the estimators need happens exactly once.

The failures are the point. A duplicated ``(unit, time)`` row, a hole in the grid and a NaN all
produce silently wrong numbers when a caller pivots by hand; here each one is an error that names
the column and the entity responsible.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from chc.did import callaway_santanna
from chc.panel import Panel, PanelError

pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")


def _long(n_units: int = 6, n_periods: int = 4, seed: int = 0) -> dict[str, np.ndarray]:
    """A balanced long panel with string unit labels, so ranking is exercised rather than luck."""
    rng = np.random.default_rng(seed)
    units = np.repeat([f"u{i:02d}" for i in range(n_units)], n_periods)
    times = np.tile(np.arange(n_periods), n_units)
    return {
        "region": units,
        "week": times,
        "spend": rng.normal(0.0, 1.0, n_units * n_periods),
        "cluster": np.repeat(np.arange(n_units) % 2, n_periods),
    }


def _panel(**kwargs: object) -> Panel:
    return Panel.from_frame(_long(), unit="region", time="week", **kwargs)  # type: ignore[arg-type]


def test_the_wide_pivot_matches_a_hand_built_one_and_feeds_the_did_estimator() -> None:
    long = _long(n_units=8, n_periods=5)
    panel = Panel.from_frame(long, unit="region", time="week")
    assert (panel.n_units, panel.n_periods) == (8, 5)
    assert panel.is_balanced

    wide = panel.wide("spend")
    expected = long["spend"].reshape(8, 5)  # the rows were emitted unit-major already
    assert np.allclose(wide, expected)
    assert wide.dtype == np.float64

    # the payoff: a long frame reaches a wide-matrix estimator with no pivot at the call site
    group = np.array([2, 2, 3, 3, -1, -1, -1, -1])
    assert isinstance(callaway_santanna(wide, group).overall, float)


def test_row_order_does_not_change_the_pivot() -> None:
    long = _long()
    shuffled = {name: column[::-1].copy() for name, column in long.items()}
    straight = Panel.from_frame(long, unit="region", time="week").wide("spend")
    reversed_rows = Panel.from_frame(shuffled, unit="region", time="week").wide("spend")
    assert np.allclose(straight, reversed_rows)


def test_a_duplicated_unit_period_names_both_rows() -> None:
    long = _long()
    long["week"] = long["week"].copy()
    long["week"][1] = long["week"][0]  # u00 now appears twice at the same week
    with pytest.raises(PanelError, match="appears twice"):
        Panel.from_frame(long, unit="region", time="week")


def test_a_non_finite_value_names_the_column_and_the_entity() -> None:
    long = _long()
    long["spend"] = long["spend"].copy()
    long["spend"][5] = np.nan
    with pytest.raises(PanelError) as caught:
        Panel.from_frame(long, unit="region", time="week")
    message = str(caught.value)
    assert "'spend'" in message
    assert str(long["region"][5]) in message


def test_a_hole_is_reported_at_the_pivot_or_at_construction_on_request() -> None:
    long = {name: np.delete(column, 3) for name, column in _long().items()}
    panel = Panel.from_frame(long, unit="region", time="week")
    assert not panel.is_balanced
    with pytest.raises(PanelError, match="cannot reshape"):
        panel.wide("spend")
    with pytest.raises(PanelError, match="unbalanced"):
        Panel.from_frame(long, unit="region", time="week", require_balanced=True)


def test_a_missing_index_column_lists_the_columns_that_do_exist() -> None:
    with pytest.raises(PanelError, match="is not in the frame"):
        Panel.from_frame(_long(), unit="region", time="quarter")


def test_codes_rank_labels_rather_than_assuming_they_are_indices() -> None:
    long = _long(n_units=3, n_periods=2)
    long["week"] = np.tile(np.array([2020, 2021]), 3)  # periods are years, not 0..T-1
    panel = Panel.from_frame(long, unit="region", time="week")
    units, times = panel.codes()
    assert panel.periods == (2020, 2021)
    assert times.tolist() == [0, 1, 0, 1, 0, 1]
    assert units.tolist() == [0, 0, 1, 1, 2, 2]


def test_pandas_and_polars_give_the_same_panel_as_the_mapping() -> None:
    long = _long()
    from_mapping = Panel.from_frame(long, unit="region", time="week")
    from_pandas = Panel.from_frame(pd.DataFrame(long), unit="region", time="week")
    from_polars = Panel.from_frame(pl.DataFrame(long), unit="region", time="week")
    assert np.allclose(from_mapping.wide("spend"), from_pandas.wide("spend"))
    assert np.allclose(from_mapping.wide("spend"), from_polars.wide("spend"))


def test_provenance_distinguishes_data_that_will_not_reproduce() -> None:
    long = _long()
    base = Panel.from_frame(long, unit="region", time="week").provenance

    same = Panel.from_frame(dict(long), unit="region", time="week").provenance
    assert same.data_sha256 == base.data_sha256  # a re-read of the same bytes is the same dataset

    nudged = dict(long)
    nudged["spend"] = long["spend"].copy()
    nudged["spend"][0] += 1e-12
    assert Panel.from_frame(nudged, unit="region", time="week").provenance.data_sha256 != (
        base.data_sha256
    )

    single = dict(long)
    single["spend"] = long["spend"].astype(np.float32)
    demoted = Panel.from_frame(single, unit="region", time="week").provenance
    assert demoted.data_sha256 != base.data_sha256  # same numbers, different precision, other run

    assert base.columns == ("cluster", "region", "spend", "week")
    assert base.n_rows == 24
    assert base.seed is None  # observed data has no seed, and a fabricated 0 would claim it did
    assert json.loads(json.dumps(base.to_json()))["n_rows"] == 24

    simulated = Panel.from_frame(long, unit="region", time="week", seed=11).provenance
    assert simulated.seed == 11
    assert simulated.data_sha256 == base.data_sha256  # the seed labels the run, not the bytes


def test_the_cluster_column_is_declared_once_and_carried() -> None:
    panel = _panel(cluster="cluster")
    assert panel.cluster == "cluster"
    assert panel["cluster"].shape == (24,)
    with pytest.raises(PanelError, match="cluster column"):
        _panel(cluster="market")
    with pytest.raises(KeyError, match="no column"):
        panel["market"]
