"""A long frame becomes a checked panel, and the pivot the estimators need happens exactly once.

The failures are the point. A duplicated ``(unit, time)`` row, a hole in the grid and a NaN all
produce silently wrong numbers when a caller pivots by hand; here each one is an error that names
the column and the entity responsible.
"""

from __future__ import annotations

import copy
import json
import pickle

import numpy as np
import pytest

from chc.did import callaway_santanna
from chc.panel import Panel, PanelError, _fingerprint

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


def test_one_column_named_as_both_the_unit_and_the_time_is_refused() -> None:
    """Read as both, a column of distinct values made each row a unit with one period of its own,
    and a column that repeats a value failed as a duplicated ``(unit, time)`` pair."""
    long = _long()
    long["row"] = np.arange(len(long["region"]))
    with pytest.raises(PanelError, match="unit and time are both 'row'"):
        Panel.from_frame(long, unit="row", time="row")


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


def test_a_panel_with_no_object_column_hashes_as_it_did() -> None:
    """Only an object column is hashed by its values: a numeric panel's recorded hash still names
    the dataset it named on 0.14.0."""
    fixed = {
        "region": np.array(["u00", "u00", "u01", "u01"]),
        "week": np.array([0, 1, 0, 1], dtype=np.int64),
        "spend": np.array([0.5, 1.25, -2.0, 3.0]),
    }
    panel = Panel.from_frame(fixed, unit="region", time="week")
    assert panel.provenance.data_sha256 == (
        "a3a9449cf23ae31b98de341e3dcf270d92c7f042594971bdb8be31047daaaed2"
    )


def _labelled(region: str = "-A") -> dict[str, np.ndarray]:
    """A unit column of strings made at run time, as a pandas column holds them, so that no two
    calls share the string objects."""
    return {
        "unit": np.array(["".join(["region", region]) for _ in range(3)], dtype=object),
        "time": np.arange(3),
        "y": np.array([0.0, 1.0, 2.0]),
    }


def test_two_equal_object_columns_hash_the_same() -> None:
    """``tobytes`` of an object array is its pointers: two panels of the same strings hashed apart,
    and the hash named no dataset another run could rebuild."""
    first, second = (Panel.from_frame(_labelled(), unit="unit", time="time") for _ in range(2))
    assert first.provenance.data_sha256 == second.provenance.data_sha256
    from_pandas = Panel.from_frame(pd.DataFrame(_labelled()), unit="unit", time="time")
    assert from_pandas.provenance.data_sha256 == first.provenance.data_sha256
    other = Panel.from_frame(_labelled("-B"), unit="unit", time="time")
    assert other.provenance.data_sha256 != first.provenance.data_sha256


@pytest.mark.parametrize(
    ("one", "other"),
    [
        ([1, 2, 3], ["1", "2", "3"]),  # the same text, another type
        (["a", "strb", "c"], ["astr", "b", "c"]),  # the same text once the values run together
    ],
)
def test_object_columns_that_differ_hash_apart(one: list[object], other: list[object]) -> None:
    hashes = []
    for values in (one, other):
        data = _labelled()
        data["code"] = np.array(values, dtype=object)
        hashes.append(Panel.from_frame(data, unit="unit", time="time").provenance.data_sha256)
    assert hashes[0] != hashes[1]


def test_any_text_that_is_no_address_is_hashed() -> None:
    data = _labelled()
    # a lone surrogate is what os.fsdecode makes of a byte that is not UTF-8
    data["code"] = np.array(["<object at 0x10>", "\ud800", ""], dtype=object)
    Panel.from_frame(data, unit="unit", time="time")
    empty = {name: column[:0] for name, column in _labelled().items()}
    assert Panel.from_frame(empty, unit="unit", time="time").provenance.n_rows == 0


def test_the_panel_does_not_change_under_its_hash() -> None:
    """The panel held views of the caller's arrays: a write to one changed the panel and left its
    hash as it was."""
    data = _labelled()
    panel = Panel.from_frame(data, unit="unit", time="time")
    data["y"][1] = 999.0
    assert panel["y"][1] == 1.0
    assert _fingerprint(panel.columns) == panel.provenance.data_sha256
    with pytest.raises(ValueError, match="read-only"):
        panel["y"][1] = 999.0
    with pytest.raises(TypeError):
        panel.columns["y"] = np.zeros(3)  # type: ignore[index]


def test_a_panel_pickles_and_its_copies_stay_read_only() -> None:
    panel = Panel.from_frame(_labelled(), unit="unit", time="time")
    for clone in (pickle.loads(pickle.dumps(panel)), copy.deepcopy(panel)):
        assert _fingerprint(clone.columns) == panel.provenance.data_sha256
        with pytest.raises(ValueError, match="read-only"):
            clone["y"][0] = 1.0


@pytest.mark.parametrize(
    ("column", "values", "message"),
    [
        ("unit", ["region-A", 7, "region-A"], r"'unit' is 7 for unit 7 at time 1: type int, where"),
        (
            "label",
            ["region-A", np.nan, "region-A"],
            r"'label' is nan for unit 'region-A' at time 1",
        ),
        ("label", ["region-A", None, "region-A"], r"'label' is None for unit 'region-A' at time 1"),
        ("code", [1, True, 1], r"'code' is True for unit 'region-A' at time 1: type bool, where"),
    ],
)
def test_an_object_column_of_two_types_is_refused(
    column: str, values: list[object], message: str
) -> None:
    """A missing value is the common case: pandas reads one among strings as nan, polars as None."""
    data = _labelled()
    data[column] = np.array(values, dtype=object)
    with pytest.raises(PanelError, match=message):
        Panel.from_frame(data, unit="unit", time="time")


def test_an_object_column_whose_text_is_its_address_is_refused() -> None:
    data = _labelled()
    data["tag"] = np.array([object() for _ in range(3)], dtype=object)
    with pytest.raises(PanelError, match="whose text is their address in memory"):
        Panel.from_frame(data, unit="unit", time="time")


def test_the_cluster_column_is_declared_once_and_carried() -> None:
    panel = _panel(cluster="cluster")
    assert panel.cluster == "cluster"
    assert panel["cluster"].shape == (24,)
    with pytest.raises(PanelError, match="cluster column"):
        _panel(cluster="market")
    with pytest.raises(KeyError, match="no column"):
        panel["market"]
