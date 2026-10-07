"""Sequential g-computation recovers a time-varying effect that naive adjustment misses."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from chc.gmethods import _ridge_fit, _ridge_predict, naive_pooled_effect, sequential_g_formula

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


UNITS = [1e-9, 1e-6, 1e-3, 1e3, 1e6]


@pytest.mark.parametrize("units", UNITS)
def test_the_g_formula_reads_the_same_with_the_treatments_in_any_units(units: float) -> None:
    """The nuisance regressions' ridge was a constant on a Gram of the raw columns: with the
    treatments logged in thousandths of their units the effect of ``(1, 1)`` against ``(0, 0)``
    read 3.194, and in millionths 0.0001, where it reads 3.305. In units ``s`` that regime is
    ``(s, s)``."""
    data = _time_varying_confounded(40_000, seed=0)
    scaled = data | {"a0": data["a0"] * units, "a1": data["a1"] * units}
    one = sequential_g_formula(data, regime=(1.0, 1.0), baseline=(0.0, 0.0), **SPEC)
    other = sequential_g_formula(scaled, regime=(units, units), baseline=(0.0, 0.0), **SPEC)
    assert other == pytest.approx(one, rel=1e-9, abs=0.0)


@pytest.mark.parametrize("units", UNITS)
def test_the_g_formula_reads_the_same_with_the_confounders_in_any_units(units: float) -> None:
    """With the confounders logged in thousandths of their units the effect read 3.351, and in
    millionths 3.427, where it reads 3.305: the ridge took their adjustment away."""
    data = _time_varying_confounded(40_000, seed=0)
    scaled = data | {"l0": data["l0"] * units, "l1": data["l1"] * units}
    one, other = (
        sequential_g_formula(log, regime=(1.0, 1.0), baseline=(0.0, 0.0), **SPEC)
        for log in (data, scaled)
    )
    assert other == pytest.approx(one, rel=1e-9, abs=0.0)


@pytest.mark.parametrize("units", UNITS)
def test_the_pooled_regression_reads_the_same_with_its_columns_in_any_units(units: float) -> None:
    """Its ridge of 1e-6 was a constant on the pooled Gram: with the treatments logged in
    millionths of their units the summed coefficient read 0.094 per original unit where it reads
    2.500, and with the confounders in millionths it read 3.404."""
    data = _time_varying_confounded(40_000, seed=0)
    one = naive_pooled_effect(data, **SPEC)
    treatments = data | {"a0": data["a0"] * units, "a1": data["a1"] * units}
    confounders = data | {"l0": data["l0"] * units, "l1": data["l1"] * units}
    assert naive_pooled_effect(treatments, **SPEC) * units == pytest.approx(one, rel=1e-9, abs=0.0)
    assert naive_pooled_effect(confounders, **SPEC) == pytest.approx(one, rel=1e-9, abs=0.0)


@pytest.mark.parametrize("level", [0.0, 7.3], ids=["zeros", "constant"])
def test_a_confounder_that_never_moves_keeps_a_ridge_term_and_changes_nothing(level: float) -> None:
    """A column that never moves has no variance to scale the ridge by, and a constant one is the
    intercept's column again, so without a term of its own the Gram is singular. It counts as
    constant: its centred values are zeroed and its ridge is scaled by 1, so its coefficient is
    zero and the others are as before. The constant carries a relative jitter of 1e-14, 45
    epsilons, under the 64 at which a column's spread is rounding, and with the ridge on the
    intercept too it moved the effect by 4.4e-7 of itself."""
    data = _time_varying_confounded(2_000, seed=7)
    jitter = 1.0 + 1e-14 * np.random.default_rng(3).standard_normal(2_000)
    padded = data | {"still": level * jitter}
    spec = SPEC | {"confounders": (("l0", "still"), ("l1",))}
    regime = {"regime": (1.0, 1.0), "baseline": (0.0, 0.0)}
    effect = sequential_g_formula(padded, **regime, **spec)
    assert effect == pytest.approx(sequential_g_formula(data, **regime, **SPEC), rel=1e-12, abs=0.0)
    pooled = naive_pooled_effect(padded, **spec)
    assert pooled == pytest.approx(naive_pooled_effect(data, **SPEC), rel=1e-12, abs=0.0)


@pytest.mark.parametrize("column", ["l0", "a0"], ids=["confounder", "treatment"])
def test_the_g_formula_reads_the_same_with_a_column_far_from_zero(column: str) -> None:
    """Each regression has an intercept, so a column's origin carries no information. Shifted by a
    thousand of its spreads and logged in millionths of its units, a confounder moved the effect to
    3.834 and a treatment to 1.639, where it reads 3.305; shifted alone, to 3.368 and 3.077, as the
    ridge was on the intercept too, and a ridge scaled by each column's mean square moves them as
    far. Solved about the columns' means, the shift costs no digits in the solve and three in the
    intercept, so rounding moves the effect by 8e-14, where on the Gram of the raw columns it moved
    it by 2e-9. A treatment's regime and baseline move with it."""
    data = _time_varying_confounded(40_000, seed=0)
    origin = 1e3 * float(np.std(data[column]))
    moved = data | {column: 1e-6 * (data[column] + origin)}
    one = sequential_g_formula(data, regime=(1.0, 1.0), baseline=(0.0, 0.0), **SPEC)
    if column == "a0":
        regime, baseline = (1e-6 * (1.0 + origin), 1.0), (1e-6 * origin, 0.0)
    else:
        regime, baseline = (1.0, 1.0), (0.0, 0.0)
    other = sequential_g_formula(moved, regime=regime, baseline=baseline, **SPEC)
    assert other == pytest.approx(one, rel=1e-11, abs=0.0)


def test_the_pooled_regression_reads_the_same_with_a_confounder_far_from_zero() -> None:
    """With a confounder a thousand of its spreads from zero and in millionths of its units, the
    summed coefficient read 2.545 where it reads 2.500; shifted alone it moved by 1.0e-6, the
    ridge of 1e-6 being on the intercept too, and a ridge scaled by each column's mean square moves
    it as far. Solved about the columns' means, rounding moves it by 4e-15, where on the Gram of
    the raw columns by 2e-11."""
    data = _time_varying_confounded(40_000, seed=0)
    moved = data | {"l0": 1e-6 * (data["l0"] + 1e3 * np.std(data["l0"]))}
    one, other = naive_pooled_effect(data, **SPEC), naive_pooled_effect(moved, **SPEC)
    assert other == pytest.approx(one, rel=1e-11, abs=0.0)


def _plane(n: int = 1_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``y = 1 + 2 x + 0.5 z`` and a tenth of a unit of noise."""
    x, z, noise = np.random.default_rng(0).standard_normal((3, n))
    return x, z, 1.0 + 2.0 * x + 0.5 * z + 0.1 * noise


@pytest.mark.parametrize("offset", [1e-11, 1e-13], ids=["1e-11", "1e-13"])
@pytest.mark.parametrize("ridge", [1e-3, 1e-6], ids=["g-formula", "pooled"])
def test_a_column_constant_but_for_1e_11_or_1e_13_of_its_size_reads_its_slope(
    ridge: float, offset: float
) -> None:
    """``1 + s z`` is ``z`` in units of ``s``, 45,000 or 450 epsilons of its size, so it is data:
    its slope is z's over ``s`` and the fit is the same. The column holds z in steps of 2e-16 / s,
    five significant digits at 1e-11 and three at 1e-13, and its term cancels against the
    intercept from 0.5 / s, so the fit agrees to 1e-15 / s: the fitted values to 1e-5 and 2.6e-4
    of the largest, the slope to 2e-8 and 1.5e-5. In the Gram of the raw columns its spread drowned
    in the rounding of its offset and left the Gram singular; with the ridge a constant its slope
    read zero and the fitted values moved by 1.6. A floor of 1e-12 of its size counted the 1e-13
    column as constant, and its slope read zero."""
    x, z, y = _plane()
    near, plain = np.column_stack([x, 1.0 + offset * z]), np.column_stack([x, z])
    beta, reference = _ridge_fit(near, y, ridge), _ridge_fit(plain, y, ridge)
    assert np.all(np.isfinite(beta))
    expected = _ridge_predict(reference, plain)
    tolerance = 1e-15 / offset
    np.testing.assert_allclose(
        _ridge_predict(beta, near), expected, rtol=0.0, atol=tolerance * np.max(np.abs(expected))
    )
    assert beta[2] * offset == pytest.approx(reference[2], rel=tolerance, abs=0.0)


@pytest.mark.parametrize("ridge", [1e-3, 1e-6], ids=["g-formula", "pooled"])
def test_a_column_constant_to_4e_15_of_its_size_reads_no_slope(ridge: float) -> None:
    """4e-15 of its size is 18 epsilons, under the 64 at which a column's spread is rounding, so
    the column counts as constant: its centred values are zeroed, its coefficient is exactly zero,
    and the fit is that with the column exactly constant. Kept, its centred values took up the
    noise along their 4e-15: with the ridge scaled by the column's mean square its coefficient
    read 1.5e-11 and 1.5e-8, and by its variance, 1.6e-29 of that, 9.5e11; with the ridge a
    constant on the intercept too, it took half the intercept, 0.498."""
    x, z, y = _plane()
    signs = np.where(np.random.default_rng(1).random(x.shape[0]) < 0.5, -1.0, 1.0)
    beta = _ridge_fit(np.column_stack([x, z, 1.0 + 4e-15 * signs]), y, ridge)
    reference = _ridge_fit(np.column_stack([x, z, np.ones_like(x)]), y, ridge)
    assert beta[3] == 0.0
    np.testing.assert_allclose(beta, reference, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("ridge", [1e-3, 1e-6], ids=["g-formula", "pooled"])
def test_a_column_that_never_moves_counts_as_constant_at_a_million_rows(ridge: float) -> None:
    """A constant column deviates from its computed mean by that mean's rounding alone, the same on
    every row. At a million rows the rounding squared to 1.8e-22 of the column's mean square, above
    the floor, so the column was scaled by it and took up the noise with a coefficient of -6.1e-6
    at either ridge. With the deviations' own mean taken off, its variance is zero, it counts as
    constant, and its coefficient is zero, where with its centred values kept, that rounding, it
    read 1e-18 and 1e-15."""
    x, z, y = _plane(1_000_000)
    beta = _ridge_fit(np.column_stack([x, z, np.full_like(x, 0.4)]), y, ridge)
    reference = _ridge_fit(np.column_stack([x, z, np.zeros_like(x)]), y, ridge)
    assert beta[3] == 0.0
    np.testing.assert_allclose(beta[:3], reference[:3], rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("ridge", [1e-3, 1e-6], ids=["g-formula", "pooled"])
def test_a_column_constant_but_for_rounding_counts_as_constant_at_many_rows(ridge: float) -> None:
    """NumPy sums a 2-D array's columns row by row, so the rounding of a column's computed mean
    grows with the rows. About a column constant but for 16 eps of its size, deviations from that
    mean alone spread 8,488 eps at 100,000 rows, above the 64 at which a spread is rounding. With
    the deviations' own mean taken off them, they spread 16 eps, the column is zeroed, and the fit
    is the one with the column all zero, bit for bit."""
    x, z, y = _plane(100_000)
    eps = np.finfo(np.float64).eps
    signs = np.where(np.random.default_rng(1).random(x.shape[0]) < 0.5, -1.0, 1.0)
    beta = _ridge_fit(np.column_stack([x, z, 0.4 * (1.0 + 16 * eps * signs)]), y, ridge)
    reference = _ridge_fit(np.column_stack([x, z, np.zeros_like(x)]), y, ridge)
    assert beta[3] == 0.0
    np.testing.assert_array_equal(beta, reference)
