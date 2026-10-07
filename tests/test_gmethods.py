"""Sequential g-computation recovers a time-varying effect that naive adjustment misses."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from chc.gmethods import naive_pooled_effect, sequential_g_formula

TH0, TH1, LAM0, LAM1, GAM, DELTA = 1.0, 1.5, 0.5, 0.8, 1.0, 0.7
TRUE_EFFECT = (TH0 + LAM1 * GAM) + TH1  # (1,1)-(0,0): a0 total (direct + via L1) + a1 = 3.3
SPEC = {"treatments": ("a0", "a1"), "confounders": (("l0",), ("l1",)), "outcome": "y"}


def _time_varying_confounded(n: int, seed: int) -> dict[str, np.ndarray]:
    """A0 affects the confounder L1, which drives A1 and Y -- the g-methods failure mode."""
    rng = np.random.default_rng(seed)
    l0 = rng.normal(0.0, 1.0, n)
    a0 = 0.9 * l0 + rng.normal(0.0, 1.0, n)
    l1 = GAM * a0 + DELTA * l0 + rng.normal(0.0, 1.0, n)  # A0 -> L1 (confounder on the causal path)
    a1 = 1.1 * l1 + 0.3 * a0 + rng.normal(0.0, 1.0, n)  # L1 -> A1 (time-varying confounding)
    y = TH0 * a0 + TH1 * a1 + LAM0 * l0 + LAM1 * l1 + rng.normal(0.0, 0.3, n)
    return {"a0": a0, "a1": a1, "l0": l0, "l1": l1, "y": y}


def test_g_formula_recovers_the_time_varying_effect() -> None:
    data = _time_varying_confounded(40_000, seed=0)
    effect = sequential_g_formula(data, regime=(1.0, 1.0), baseline=(0.0, 0.0), **SPEC)
    assert effect == pytest.approx(TRUE_EFFECT, abs=0.1)  # standardising over L1 recovers the truth


def test_naive_adjustment_is_biased_by_the_mediator_confounder() -> None:
    data = _time_varying_confounded(40_000, seed=1)
    naive = naive_pooled_effect(data, **SPEC)
    assert abs(naive - TRUE_EFFECT) > 0.5  # conditioning on L1 underestimates A0's total effect


def test_g_formula_beats_the_naive_pooled_regression() -> None:
    data = _time_varying_confounded(40_000, seed=2)
    g = sequential_g_formula(data, regime=(1.0, 1.0), baseline=(0.0, 0.0), **SPEC)
    naive = naive_pooled_effect(data, **SPEC)
    assert abs(g - TRUE_EFFECT) < abs(naive - TRUE_EFFECT)  # g-formula is the less-biased estimator


def test_mismatched_horizon_raises() -> None:
    data = _time_varying_confounded(200, seed=3)
    with pytest.raises(ValueError, match="horizon"):
        sequential_g_formula(
            data,
            treatments=("a0", "a1"),
            confounders=(("l0",),),
            outcome="y",
            regime=(1.0, 1.0),
            baseline=(0.0, 0.0),
        )


@pytest.mark.parametrize("folds", [0, 1, 201, 2.0, True])
def test_folds_outside_two_to_the_rows_are_refused(folds) -> None:
    """One fold trains on nothing and returned 0; none left every row unwritten and returned what
    the empty array held."""
    data = _time_varying_confounded(200, seed=4)
    with pytest.raises(ValueError, match=r"folds=.*from 2 to the rows \(200\)"):
        sequential_g_formula(data, regime=(1.0, 1.0), baseline=(0.0, 0.0), folds=folds, **SPEC)


@pytest.mark.parametrize("folds", [2, np.int64(3), 200])
def test_folds_from_two_to_the_rows_are_accepted(folds) -> None:
    data = _time_varying_confounded(200, seed=4)
    effect = sequential_g_formula(data, regime=(1.0, 1.0), baseline=(0.0, 0.0), folds=folds, **SPEC)
    assert np.isfinite(effect)


G_SECOND_ROLES = {
    "a treatment among its own step's confounders": (
        {"confounders": (("l0", "a0"), ("l1",))},
        "the treatment 'a0' is set at step 0 and is a confounder measured before step 0",
    ),
    "a later treatment among an earlier step's confounders": (
        {"confounders": (("l0", "a1"), ("l1",))},
        "the treatment 'a1' is set at step 1 and is a confounder measured before step 0",
    ),
    "the outcome among the confounders": (
        {"confounders": (("l0",), ("l1", "y"))},
        "the column 'y' is read as the outcome and as a confounder",
    ),
    "the outcome as a treatment": (
        {"treatments": ("a0", "y")},
        "the column 'y' is read as the outcome and as a treatment",
    ),
    "a treatment twice": (
        {"treatments": ("a0", "a0")},
        "the column 'a0' is read twice as a treatment",
    ),
}


@pytest.mark.parametrize(("names", "match"), G_SECOND_ROLES.values(), ids=G_SECOND_ROLES.keys())
def test_the_g_formula_refuses_a_column_in_two_roles(names: dict, match: str) -> None:
    """A treatment among the confounders of its own step was set in one column of the design and
    kept at its logged value in the other, and the outcome among them explained itself."""
    data = _time_varying_confounded(200, seed=5)
    spec = SPEC | names
    with pytest.raises(ValueError, match=match):
        sequential_g_formula(data, regime=(1.0, 1.0), baseline=(0.0, 0.0), **spec)


def test_an_earlier_treatment_may_confound_a_later_one() -> None:
    """Measured before the later treatment, the earlier one is one of its confounders: it is set
    at its own step, before the later step's design reads it."""
    data = _time_varying_confounded(40_000, seed=0)
    spec = SPEC | {"confounders": (("l0",), ("l1", "a0"))}
    effect = sequential_g_formula(data, regime=(1.0, 1.0), baseline=(0.0, 0.0), **spec)
    assert effect == pytest.approx(TRUE_EFFECT, abs=0.1)


@pytest.mark.parametrize(
    ("names", "match"),
    [
        (
            {"confounders": (("l0",), ("l1", "a0"))},
            r"the treatments \['a0'\] are also confounders",
        ),
        (
            {"confounders": (("l0", "y"), ("l1",))},
            "the column 'y' is read as the outcome and as a confounder",
        ),
    ],
)
def test_the_pooled_regression_refuses_a_treatment_or_the_outcome_among_the_confounders(
    names: dict, match: str
) -> None:
    """Pooled, a treatment that is also a confounder splits its coefficient with its own copy, and
    the sum of the treatments' coefficients misses the copy's share."""
    data = _time_varying_confounded(200, seed=6)
    with pytest.raises(ValueError, match=match):
        naive_pooled_effect(data, **(SPEC | names))


def _estimates(data: object) -> tuple[float, float]:
    """The g-formula's effect and the pooled regression's, on one frame."""
    return (
        sequential_g_formula(data, regime=(1.0, 1.0), baseline=(0.0, 0.0), **SPEC),
        naive_pooled_effect(data, **SPEC),
    )


def _past_float64(values: np.ndarray) -> np.ndarray:
    cells = np.array([Decimal(float(value)) for value in values], dtype=object)
    cells[0] = Decimal("1E+400")
    return cells


def _masked(values: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_array(values, mask=np.arange(values.size) == 3)


@pytest.mark.parametrize(
    ("convert", "message"),
    [
        (lambda v: v.astype(str).astype(object), r"'.+' at row 0: type str, not a real number"),
        (lambda v: v.astype(str), r"np\.str_\('.+'\) at row 0: dtype <U\d+, which holds text"),
        (
            lambda v: np.datetime64("2024-01-01") + np.arange(v.size),
            r"np\.datetime64\('2024-01-01'\) at row 0: dtype datetime64\[D\], which holds dates",
        ),
        (
            lambda v: np.arange(v.size).astype("timedelta64[s]"),
            r"np\.timedelta64\(0,'s'\) at row 0: dtype timedelta64\[s\], which holds",
        ),
        (
            lambda v: v + 1j,
            r"np\.complex128\(.+\+1j\) at row 0: dtype complex128, which holds complex",
        ),
        (
            _past_float64,
            r"Decimal\('1E\+400'\) at row 0: a finite number past float64's range, where it is",
        ),
        (_masked, r"masked at row 3 \(1 of 200 cells\): a masked cell is a missing value"),
    ],
    ids=["text", "fixed-width-text", "datetime64", "timedelta64", "complex", "decimal", "masked"],
)
def test_a_column_that_is_not_numbers_is_refused_naming_it_and_its_row(
    convert, message: str
) -> None:
    """Every column was read with a float64 cast, which read the text "0.5" as 0.5, a date as its
    count of days, a duration as its seconds and a complex number as its real part; a Decimal of
    1E+400 became an infinity and the effect nan; a masked array was read under its mask."""
    data = _time_varying_confounded(200, seed=7)
    data["a1"] = convert(data["a1"])
    with pytest.raises(ValueError, match=rf"column 'a1' is {message}"):
        sequential_g_formula(data, regime=(1.0, 1.0), baseline=(0.0, 0.0), **SPEC)
    with pytest.raises(ValueError, match=rf"column 'a1' is {message}"):
        naive_pooled_effect(data, **SPEC)


def test_a_pandas_text_column_is_refused_and_a_column_not_named_is_not_read() -> None:
    """pandas 3 hands its ``str`` dtype over as text among objects. A label column the call does
    not name, which the float64 cast of every column refused when it was text, is not read."""
    pd = pytest.importorskip("pandas")
    data = _time_varying_confounded(200, seed=8)
    frame = pd.DataFrame(data)
    frame["l1"] = frame["l1"].astype("str")
    with pytest.raises(ValueError, match=r"column 'l1' is '.+' at row 0: type str, not a real"):
        _estimates(frame)
    labelled = dict(data)
    labelled["region"] = np.array([f"r{row % 3}" for row in range(200)])
    labelled["day"] = np.datetime64("2024-01-01") + np.arange(200)
    assert _estimates(labelled) == _estimates(data)


def test_numbers_are_read_as_a_float64_cast_reads_them_and_a_missing_one_reads_nan() -> None:
    """A boolean treatment and pandas' nullable integers read as before, bit for bit; a nullable
    column's NA comes over as nan, as a float column's nan does, and the effect reads nan."""
    pd = pytest.importorskip("pandas")
    data = _time_varying_confounded(200, seed=9)
    data["a0"] = data["a0"] > 0.0
    as_floats = dict(data, a0=data["a0"].astype(np.float64))
    assert _estimates(data) == _estimates(as_floats)
    rounded = dict(data, l0=np.round(data["l0"] * 10.0))
    frame = pd.DataFrame(rounded)
    frame["l0"] = frame["l0"].astype("Int64")
    assert _estimates(frame) == _estimates(rounded)
    frame.loc[5, "l0"] = pd.NA
    assert all(np.isnan(effect) for effect in _estimates(frame))


@pytest.mark.parametrize(
    ("value", "dtype"),
    [
        (Decimal("Infinity"), object),
        (Decimal("NaN"), object),
        (np.inf, object),
        (np.nan, object),
        (np.inf, np.longdouble),
    ],
    ids=["decimal-infinity", "decimal-nan", "infinity", "nan", "longdouble-infinity"],
)
def test_a_nan_or_an_infinity_of_the_columns_own_type_is_read_as_it_is(
    value: object, dtype: type
) -> None:
    """The rule refuses a finite number float64 cannot hold, not the nan or the infinity a column
    holds as such: those read as a float column's do, and the effect reads nan."""
    data = _time_varying_confounded(200, seed=10)
    kind = Decimal if isinstance(value, Decimal) else float
    cells = np.array([kind(float(level)) for level in data["l0"]], dtype=object)
    cells[5] = value
    data["l0"] = cells.astype(dtype)
    assert all(np.isnan(effect) for effect in _estimates(data))
