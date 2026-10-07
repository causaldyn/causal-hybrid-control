"""A long frame becomes a checked panel, and the pivot the estimators need happens exactly once.

The failures are the point. A duplicated ``(unit, time)`` row, a hole in the grid and a NaN all
produce silently wrong numbers when a caller pivots by hand; here each one is an error that names
the column and the entity responsible.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import pickle
from decimal import Decimal
from fractions import Fraction
from uuid import UUID

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


def _cells(*values: object) -> np.ndarray:
    """An object column of ``values`` as they are: ``np.array`` reads a list, a tuple or an array
    as a row of its items, not as one value."""
    column = np.empty(len(values), dtype=object)
    for row, value in enumerate(values):
        column[row] = value
    return column


class _Tag:
    """A value of the caller's own class: its text names it, and a write to it changes both."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name


def test_a_dict_the_caller_still_holds_is_refused() -> None:
    """The copy of an object column held the caller's objects: a write to a dict the caller still
    held, or to one read back through ``panel[name][row]``, changed the panel under its hash."""
    data = _labelled()
    data["meta"] = _cells({"tag": 1}, {"tag": 2}, {"tag": 3})
    with pytest.raises(
        PanelError,
        match=r"column 'meta' is \{'tag': 1\} for unit 'region-A' at time 0: type dict, which a "
        "panel does not hold",
    ):
        Panel.from_frame(data, unit="unit", time="time")


@pytest.mark.parametrize(
    "values",
    [
        [["a"], ["b"], ["c"]],
        [{"a"}, {"b"}, {"c"}],
        [bytearray(b"a"), bytearray(b"b"), bytearray(b"c")],
        [np.zeros(1), np.zeros(1), np.zeros(1)],
        [_Tag("a"), _Tag("b"), _Tag("c")],
        [("a", 1), ("b", 2), ("c", 3)],
        [pd.Period(f"2024-0{month}", freq="M") for month in (1, 2, 3)],
        [None, None, None],
        [pd.NA, pd.NA, pd.NA],
    ],
    ids=["list", "set", "bytearray", "array", "own-class", "tuple", "period", "none", "pandas-na"],
)
def test_a_value_that_can_change_or_is_missing_is_refused(values: list[object]) -> None:
    """A copy of a value that changes in place still changes through ``panel[name][row]``, so it is
    refused. So is a tuple, which can hold such a value and whose text is its items' ``repr``; a
    type the panel does not list, a pandas ``Period`` among them; and a value missing in every row,
    which a reader of numbers takes as nan, or fails on."""
    data = _labelled()
    data["meta"] = _cells(*values)
    kind = type(values[0]).__name__
    with pytest.raises(
        PanelError,
        match=rf"column 'meta' is .+ for unit 'region-A' at time 0: type {kind}, which a panel "
        "does not hold",
    ):
        Panel.from_frame(data, unit="unit", time="time")


def test_a_unit_that_can_change_is_refused_before_the_index_is_read() -> None:
    """A list as a unit failed the duplicate check with ``TypeError``, a list being unhashable,
    and not with the panel's own error."""
    data = _labelled()
    data["unit"] = _cells(["a"], ["b"], ["c"])
    with pytest.raises(
        PanelError, match=r"column 'unit' is \['a'\] for unit \['a'\] at time 0: type list"
    ):
        Panel.from_frame(data, unit="unit", time="time")


def test_bytes_whose_text_reads_as_an_address_are_held() -> None:
    """The address check reads only the types a panel does not hold: the text of bytes names their
    value, as the text of a str does."""
    data = _labelled()
    data["raw"] = _cells(b"<object at 0x10>", b"", b"a")
    Panel.from_frame(data, unit="unit", time="time")


def _held() -> dict[str, np.ndarray]:
    """A column of each type an object column holds, NumPy's scalars among them."""
    utc = dt.UTC
    return {
        "text": _cells("a", "", "\ud800"),
        "raw": _cells(b"a", b"", b"\xff"),
        "flag": _cells(True, False, True),
        "count": _cells(1, -2, 10**30),
        "share": _cells(0.5, -1.25, 1e-300),
        "phasor": _cells(1 + 2j, -0.5j, 3 + 0j),
        "price": _cells(Decimal("0.10"), Decimal(2), Decimal("-3.5E+2")),
        "ratio": _cells(Fraction(1, 3), Fraction(2), Fraction(-7, 4)),
        "id": _cells(UUID(int=1), UUID(int=2), UUID(int=3)),
        "day": _cells(dt.date(2024, 1, 1), dt.date(2024, 2, 1), dt.date(2024, 3, 1)),
        "clock": _cells(dt.time(9), dt.time(12, 30), dt.time(23, 59, 59, 999999)),
        "stamp": _cells(*(dt.datetime(2024, 1, day, tzinfo=utc) for day in (1, 2, 3))),
        "lag": _cells(dt.timedelta(days=1), dt.timedelta(hours=-3), dt.timedelta(0)),
        "np_count": _cells(np.int64(1), np.int64(-2), np.int64(3)),
        "np_share": _cells(np.float32(0.5), np.float32(-1.25), np.float32(3)),
        "np_phasor": _cells(np.complex64(1j), np.complex64(-2), np.complex64(0)),
        "np_flag": _cells(np.True_, np.False_, np.True_),
        "np_text": _cells(np.str_("a"), np.str_("b"), np.str_("c")),
        "np_day": _cells(*(np.datetime64(f"2024-01-0{day}") for day in (1, 2, 3))),
        "np_lag": _cells(np.timedelta64(1, "D"), np.timedelta64(-2, "h"), np.timedelta64(0, "s")),
    }


def test_the_values_a_panel_holds_are_hashed_as_before() -> None:
    """Text, numbers and times, none of which changes in place: each is held and hashed by its type
    and its text, as ADR 0056 hashes it. The digest is the one 0.14.2 gave this panel."""
    panel = Panel.from_frame({**_labelled(), **_held()}, unit="unit", time="time")
    assert panel.provenance.data_sha256 == (
        "1199bc86f9942db4c79246236706e995cbfda4a3ca9aa05bc2753db455b909d9"
    )


def test_a_pandas_column_of_aware_timestamps_is_held() -> None:
    """pandas hands over a column of time-zone-aware timestamps as objects, each a ``datetime``."""
    frame = pd.DataFrame(_labelled())
    frame["time"] = pd.date_range("2024-01-01", periods=3, tz="UTC")
    panel = Panel.from_frame(frame, unit="unit", time="time")
    assert panel["time"].dtype == object
    assert panel.periods[0] == pd.Timestamp("2024-01-01", tz="UTC")


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
def test_a_non_finite_float_in_an_object_column_is_refused(value: float) -> None:
    """The finite check read float columns only: an object column of floats passed nan and the
    infinities, which a fit reads as float64 among its states."""
    data = _labelled()
    data["y"] = np.array([0.0, value, 2.0], dtype=object)
    with pytest.raises(
        PanelError,
        match=rf"column 'y' is {value} for unit 'region-A' at time np\.int64\(1\) \(1 of 3 rows "
        "are not finite\\)",
    ):
        Panel.from_frame(data, unit="unit", time="time")


@pytest.mark.parametrize(
    ("values", "shown"),
    [
        ([Decimal(1), Decimal("NaN"), Decimal(2)], "NaN"),
        ([Decimal(1), Decimal("sNaN"), Decimal(2)], "sNaN"),
        ([Decimal(1), Decimal("-Infinity"), Decimal(2)], "-Infinity"),
        ([1 + 0j, complex(0.0, np.inf), 2j], "infj"),
        ([np.float32(1), np.float32(np.nan), np.float32(2)], "nan"),
        ([np.complex64(1), np.complex64(complex(np.nan, 0.0)), np.complex64(2)], r"\(nan\+0j\)"),
    ],
    ids=["decimal-nan", "decimal-snan", "decimal-inf", "complex", "float32", "complex64"],
)
def test_a_non_finite_number_of_another_type_is_refused(values: list[object], shown: str) -> None:
    """Each number type with a nan or an infinity is checked as a float is. A Decimal answers by
    its own test, since ``float`` raises on its signalling nan."""
    data = _labelled()
    data["y"] = _cells(*values)
    with pytest.raises(
        PanelError, match=rf"column 'y' is {shown} for unit 'region-A' at time np\.int64\(1\) \("
    ):
        Panel.from_frame(data, unit="unit", time="time")


def test_a_complex_column_is_checked_as_a_float_column_is() -> None:
    """A complex column passed nan for the reason an object column did: the check read floats."""
    data = _labelled()
    data["z"] = np.array([0j, complex(np.nan, 1.0), 2j])
    with pytest.raises(PanelError, match=r"column 'z' is \(nan\+1j\) for unit 'region-A' at time"):
        Panel.from_frame(data, unit="unit", time="time")


def test_a_period_that_is_not_finite_is_refused_before_the_index_is_read() -> None:
    """A Decimal's signalling nan raises ``TypeError`` when it is hashed, as the index check
    hashes each period."""
    data = _labelled()
    data["time"] = _cells(Decimal(0), Decimal("sNaN"), Decimal(2))
    with pytest.raises(
        PanelError, match=r"column 'time' is sNaN for unit 'region-A' at time Decimal\('sNaN'\)"
    ):
        Panel.from_frame(data, unit="unit", time="time")


def test_the_cluster_column_is_declared_once_and_carried() -> None:
    panel = _panel(cluster="cluster")
    assert panel.cluster == "cluster"
    assert panel["cluster"].shape == (24,)
    with pytest.raises(PanelError, match="cluster column"):
        _panel(cluster="market")
    with pytest.raises(KeyError, match="no column"):
        panel["market"]


_WIDE_LONGDOUBLE = np.finfo(np.longdouble).max > np.finfo(np.float64).max
_NOT_READ = ", so the column is not read as numbers"


@pytest.mark.parametrize(
    ("values", "shown", "why"),
    [
        (_cells("0.5", "nan", "2"), r"'0\.5'", "type str, not a real number"),
        (np.array(["0.5", "1", "2"]), r"np\.str_\('0\.5'\)", "dtype <U3, which holds text"),
        (
            np.array(["0.5", "1", "2"], dtype=np.dtypes.StringDType()),
            r"'0\.5'",
            r"dtype StringDType\(\), which holds text",
        ),
        (
            np.array(["2024-01-01", "NaT", "2024-01-03"], dtype="datetime64[D]"),
            r"np\.datetime64\('2024-01-01'\)",
            r"dtype datetime64\[D\], which holds dates or times",
        ),
        (
            np.arange(3).astype("timedelta64[s]"),
            r"np\.timedelta64\(0,'s'\)",
            r"dtype timedelta64\[s\], which holds durations",
        ),
        (
            np.array([0.5, 1.0, 2.0]) + 0j,
            r"np\.complex128\(0\.5\+0j\)",
            "dtype complex128, which holds complex numbers",
        ),
        (
            _cells(*(np.timedelta64(day, "D") for day in (1, 2, 3))),
            r"np\.timedelta64\(1,'D'\)",
            "type timedelta64, not a real number",
        ),
        (
            _cells(*(dt.date(2024, 1, day) for day in (1, 2, 3))),
            r"datetime\.date\(2024, 1, 1\)",
            "type date, not a real number",
        ),
        (_cells(1 + 0j, 2 + 0j, 3 + 0j), r"\(1\+0j\)", "type complex, not a real number"),
        (
            _cells(UUID(int=1), UUID(int=2), UUID(int=3)),
            r"UUID\('0{8}-0{4}-0{4}-0{4}-0{11}1'\)",
            "type UUID, not a real number",
        ),
    ],
    ids=[
        "text",
        "fixed-width-text",
        "numpy-string",
        "datetime64",
        "timedelta64",
        "complex",
        "object-timedelta64",
        "object-date",
        "object-complex",
        "object-uuid",
    ],
)
def test_a_column_that_is_not_numbers_is_held_and_refused_when_read_as_numbers(
    values: np.ndarray, shown: str, why: str
) -> None:
    """NumPy's cast to float64 read the text ``"0.5"`` as 0.5 and ``"nan"`` as nan, a date as its
    count of days and NaT as -9.2e18, a duration as its count of seconds and a complex number as
    its real part, and ``wide`` returned what it read. The panel holds such a column, as it holds
    a label, and refuses it where it is read as numbers, naming the unit and the time of row 0."""
    data = _labelled()
    data["y"] = values
    panel = Panel.from_frame(data, unit="unit", time="time")
    with pytest.raises(
        PanelError,
        match=rf"column 'y' is {shown} for unit 'region-A' at time np\.int64\(0\): {why}.*"
        + _NOT_READ,
    ):
        panel.wide("y")


def test_text_from_a_pandas_or_a_polars_frame_is_refused_when_read_as_numbers() -> None:
    """pandas 3 reads text as its ``str`` dtype and hands it over as objects, polars as fixed-width
    text: a float64 cast read either as numbers."""
    frame = pd.DataFrame(_labelled())
    frame["y"] = frame["y"].astype("str")
    assert str(frame["y"].dtype) == "str"
    polars = pl.DataFrame({"unit": ["region-A"] * 3, "time": [0, 1, 2], "y": ["0.0", "1", "2"]})
    for source, shown in ((frame, r"'0\.0'"), (polars, r"np\.str_\('0\.0'\)")):
        panel = Panel.from_frame(source, unit="unit", time="time")
        with pytest.raises(PanelError, match=rf"column 'y' is {shown} for unit .*region-A.*"):
            panel.wide("y")


@pytest.mark.parametrize(
    ("values", "shown"),
    [
        (_cells(Decimal(1), Decimal("1E+400"), Decimal(2)), r"Decimal\('1E\+400'\)"),
        (_cells(Decimal(0), Decimal("-1E+400"), Decimal(2)), r"Decimal\('-1E\+400'\)"),
        (_cells(1, 10**400, 2), "10{400}"),
        (_cells(Fraction(1), Fraction(10**400, 3), Fraction(2)), r"Fraction\(10{400}, 3\)"),
        pytest.param(
            np.array(["1", "1e400", "2"]).astype(np.longdouble),
            r"np\.longdouble\('1e\+400'\)",
            marks=pytest.mark.skipif(not _WIDE_LONGDOUBLE, reason="longdouble is float64 here"),
        ),
        pytest.param(
            _cells(np.longdouble(1), np.longdouble("1e400"), np.longdouble(2)),
            r"np\.longdouble\('1e\+400'\)",
            marks=pytest.mark.skipif(not _WIDE_LONGDOUBLE, reason="longdouble is float64 here"),
        ),
    ],
    ids=["decimal", "negative-decimal", "int", "fraction", "longdouble", "object-longdouble"],
)
def test_a_finite_number_past_float64s_range_is_held_and_refused_when_read(
    values: np.ndarray, shown: str
) -> None:
    """A float64 cast read a ``Decimal`` of ``1E+400`` and a ``longdouble`` of 1e400 as infinities,
    and raised a bare ``OverflowError`` on an ``int`` or a ``Fraction`` of 400 digits. Each is
    finite in its own type, so the panel holds it, and refuses it where it is read as numbers."""
    data = _labelled()
    data["y"] = values
    panel = Panel.from_frame(data, unit="unit", time="time")
    with pytest.raises(
        PanelError,
        match=rf"column 'y' is {shown} for unit 'region-A' at time np\.int64\(1\): a finite "
        r"number past float64's range, where it is an infinity" + _NOT_READ,
    ):
        panel.wide("y")


def test_a_masked_cell_is_refused_and_an_array_that_masks_none_is_its_data() -> None:
    """The copy of a masked array dropped its mask, and ``wide`` read what lay under it."""
    data = _labelled()
    data["y"] = np.ma.masked_array([0.0, 999.0, 2.0], mask=[False, True, False])
    with pytest.raises(ValueError, match=r"column 'y' is masked at row 1 \(1 of 3 cells\)"):
        Panel.from_frame(data, unit="unit", time="time")
    data["y"] = np.ma.masked_array([0.0, 1.0, 2.0], mask=False)
    held = Panel.from_frame(data, unit="unit", time="time")
    plain = Panel.from_frame(_labelled(), unit="unit", time="time")
    assert held.provenance.data_sha256 == plain.provenance.data_sha256
    assert held.wide("y").tolist() == [[0.0, 1.0, 2.0]]


def test_numbers_are_read_as_a_float64_cast_reads_them() -> None:
    """Booleans, integers and floats of every width, pandas' nullable numbers with no value
    missing, and an object column of real numbers, NumPy's among them: each read bit for bit as a
    float64 cast reads it."""
    held = _held()
    columns = {
        "flag": np.array([True, False, True]),
        "small": np.array([1, -2, 3], dtype=np.int8),
        "big": np.array([0, 1, 2**64 - 1], dtype=np.uint64),
        "half": np.array([0.1, -2.5, 65504.0], dtype=np.float16),
        "single": np.array([0.1, 1e-45, 3.4e38], dtype=np.float32),
        **{
            f"object_{name}": held[name]
            for name in ("flag", "count", "share", "price", "ratio", "np_count", "np_share")
        },
        "object_np_flag": held["np_flag"],
    }
    frame = pd.DataFrame({**_labelled(), **columns})
    frame["nullable_int"] = pd.array([1, 2, 3], dtype="Int64")
    frame["nullable_float"] = pd.array([0.5, 1.5, 2.5], dtype="Float64")
    frame["nullable_bool"] = pd.array([True, False, True], dtype="boolean")
    panel = Panel.from_frame(frame, unit="unit", time="time")
    for name in (*columns, "nullable_int", "nullable_float", "nullable_bool"):
        expected = np.asarray(np.asarray(frame[name]), dtype=np.float64)
        assert panel.wide(name).ravel().tobytes() == expected.tobytes(), name


@pytest.mark.parametrize(
    ("array", "message"),
    [
        (pd.array([1, None, 3], dtype="Int64"), r"column 'y' is nan for unit 'region-A' at time"),
        (pd.array([0.5, None, 2.5], dtype="Float64"), r"column 'y' is nan for unit 'region-A'"),
        (pd.array([True, None, False], dtype="boolean"), r"column 'y' is <NA> .*type NAType"),
    ],
    ids=["Int64", "Float64", "boolean"],
)
def test_a_nullable_column_with_a_missing_value_is_refused(array: object, message: str) -> None:
    """pandas hands over a nullable number's NA as nan, and a nullable boolean's as ``NA`` among
    objects: each is a missing value, which a panel refuses, naming the column, the unit and the
    time, before anything reads it."""
    frame = pd.DataFrame(_labelled())
    frame["y"] = array
    with pytest.raises(PanelError, match=message):
        Panel.from_frame(frame, unit="unit", time="time")


def test_an_empty_column_has_no_value_to_refuse() -> None:
    empty = {name: column[:0] for name, column in _labelled().items()}
    empty["y"] = np.array([], dtype="<U3")
    assert Panel.from_frame(empty, unit="unit", time="time").wide("y").shape == (0, 0)
