"""The MSM bound on a mean per covariate cell (``covariates=``) and on Tan's odds scale for a binary
lever (``propensity=``), against ``validation/msm_level_bound.mac``."""

from __future__ import annotations

import itertools
import math
import re

import numpy as np
import pytest
from scipy.optimize import linprog

from chc.uncertainty import (
    _box_worst_case,
    _level_bound,
    confounding_robust_radius,
    msm_worst_case_mean,
    negative_control_gamma,
)

# validation/msm_level_bound.mac L3: X in {0, 1}, Y | x = m_x -+ 1, m = (0, 4), Gamma = 2
ATOMS = np.array([-1.0, 1.0, 3.0, 5.0])
CELL_MEANS = np.array([0.0, 0.0, 4.0, 4.0])
CELL_PROPENSITY = np.array([0.25, 0.25, 0.75, 0.75])

# L2b: the least Gamma reconciling a null of mean 0.2, residual sd 1, read on the pooled sample
POOLED_GAMMA = {0.0: 1.2854785337009367, 0.5: 1.1941215347747158, 0.8: 1.1186713106149841,
                0.95: 1.0576560005500133}  # fmt: skip


def _population(rng, n: int, r2: float, mean: float = 0.2):
    x = rng.standard_normal(n)
    y = mean + x * math.sqrt(r2 / (1.0 - r2)) + rng.standard_normal(n)
    return x, y


def test_the_pooled_and_residualised_bounds_are_maximas_linear_programs() -> None:
    box = np.full(4, 0.5), np.full(4, 2.0)
    pooled, _ = _level_bound(ATOMS, 2.0, np.zeros(4), None)
    residualised, _ = _level_bound(ATOMS, 2.0, CELL_MEANS, None)
    assert pooled == pytest.approx(13.0 / 4.0, rel=1e-15, abs=0.0)
    assert residualised == pytest.approx(5.0 / 2.0, rel=1e-15, abs=0.0)  # the per-x sharp bound
    assert _box_worst_case(ATOMS, *box)[0] == pytest.approx(13.0 / 4.0, rel=1e-15, abs=0.0)


def test_tans_boxes_are_maximas_linear_programs() -> None:
    pooled, _ = _level_bound(ATOMS, 2.0, np.zeros(4), CELL_PROPENSITY)
    residualised, _ = _level_bound(ATOMS, 2.0, CELL_MEANS, CELL_PROPENSITY)
    assert pooled == pytest.approx(41.0 / 16.0, rel=1e-15, abs=0.0)
    assert residualised == pytest.approx(9.0 / 4.0, rel=1e-15, abs=0.0)


@pytest.mark.parametrize("seed", range(5))
def test_the_greedy_fill_is_the_linear_programs_optimum(seed) -> None:
    rng = np.random.default_rng(seed)
    n, gamma = 40, 2.5
    residuals = rng.standard_normal(n)
    e = rng.uniform(0.1, 0.9, n)
    low, high = e + (1.0 - e) / gamma, e + (1.0 - e) * gamma
    found, threshold = _box_worst_case(residuals, low, high)
    best = linprog(
        -residuals / n,
        A_eq=np.ones((1, n)) / n,
        b_eq=[1.0],
        bounds=list(zip(low, high, strict=True)),
        method="highs",
    )
    assert found == pytest.approx(-best.fun, rel=1e-9, abs=1e-12)
    # the dual at the threshold the fill stops at is the same number
    dual = (
        threshold
        + low * (residuals - threshold)
        + (high - low) * np.maximum(residuals - threshold, 0.0)
    )
    assert float(dual.mean()) == pytest.approx(found, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("e", [0.2, 0.5, 0.8])
def test_one_propensity_scales_the_inflation_by_one_less_it(e) -> None:
    y = np.random.default_rng(1).standard_normal(1001)
    box = msm_worst_case_mean(y, 2.0) - y.mean()
    odds = msm_worst_case_mean(y, 2.0, propensity=e) - y.mean()
    assert odds == pytest.approx((1.0 - e) * box, rel=1e-12, abs=0.0)


def test_without_covariates_or_propensity_the_bound_is_the_one_before() -> None:
    y = np.random.default_rng(2).standard_normal(500)
    assert msm_worst_case_mean(y, 2.0, covariates=None, propensity=None) == msm_worst_case_mean(
        y, 2.0
    )
    assert negative_control_gamma(y + 0.1, covariates=None) == negative_control_gamma(y + 0.1)


@pytest.mark.parametrize("r2", [0.5, 0.8, 0.95])
def test_the_residualised_bound_narrows_by_the_covariates_share(r2) -> None:
    # Gaussian location model: the pooled inflation is the residual's over sqrt(1 - R^2)
    x, y = _population(np.random.default_rng(3), 200_000, r2)
    pooled = msm_worst_case_mean(y, 2.0) - y.mean()
    residualised = msm_worst_case_mean(y, 2.0, covariates=x) - y.mean()
    assert residualised == pytest.approx(0.5453996620129763, rel=0.02, abs=0.0)
    assert pooled / residualised == pytest.approx(1.0 / math.sqrt(1.0 - r2), rel=0.02, abs=0.0)


@pytest.mark.parametrize("r2", sorted(POOLED_GAMMA))
def test_a_negative_control_reads_one_gamma_whatever_the_covariates_explain(r2) -> None:
    n = 200_000
    x, y = _population(np.random.default_rng(4), n, r2)
    # the reading moves with the sample mean only, by dGamma/dmu <= 2 here (L2b: the secant from
    # mu = 0.2 to 0.4 on a residual sd of 1 is 1.86, and the pooled sd only flattens it)
    tolerance = 4.5 * 2.0 * float(np.std(y)) / math.sqrt(n)
    assert negative_control_gamma(y) == pytest.approx(POOLED_GAMMA[r2], abs=tolerance, rel=0.0)
    # with the covariates it reads the R^2 = 0 value at every R^2
    assert negative_control_gamma(y, covariates=x) == pytest.approx(
        POOLED_GAMMA[0.0], abs=tolerance, rel=0.0
    )


@pytest.mark.parametrize("seed", [10, 11])
def test_the_residualised_bound_keeps_its_level_on_an_unconfounded_null(seed) -> None:
    # at level 0.95 an unconfounded null reads above 1 at the two-sided t test's rate
    rng, reps, n = np.random.default_rng(seed), 400, 400
    above = 0
    for _ in range(reps):
        x, y = _population(rng, n, 0.8, mean=0.0)
        above += negative_control_gamma(y, level=0.95, covariates=x) > 1.0
    assert abs(above / reps - 0.05) <= 4.5 * math.sqrt(0.05 * 0.95 / reps)


def test_the_radius_widens_by_the_residualised_gap() -> None:
    x, y = _population(np.random.default_rng(5), 5000, 0.8)
    worst = msm_worst_case_mean(y, 3.0, covariates=x, propensity=0.4)
    radius = confounding_robust_radius(0.1, y, 3.0, covariates=x, propensity=0.4)
    assert radius == pytest.approx(0.1 + worst - y.mean(), rel=1e-12, abs=0.0)
    assert radius < confounding_robust_radius(0.1, y, 3.0)


def test_at_gamma_one_the_residualised_bound_is_the_mean() -> None:
    x, y = _population(np.random.default_rng(6), 1000, 0.5)
    assert msm_worst_case_mean(y, 1.0, covariates=x) == pytest.approx(y.mean(), rel=0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"covariates": np.zeros(9)}, "one row per outcome"),
        ({"covariates": np.full(10, np.inf)}, "finite"),
        ({"covariates": np.zeros((10, 8))}, "cross-fitting 9 coefficients"),
        ({"propensity": 1.0}, r"in \(0, 1\)"),
        ({"propensity": np.full(3, 0.5)}, "one number or one per outcome"),
    ],
)
def test_the_bound_refuses_what_it_cannot_read(kwargs, match) -> None:
    y = np.arange(10.0)
    with pytest.raises(ValueError, match=match):
        msm_worst_case_mean(y, 2.0, **kwargs)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_column_of_outcomes_reads_as_the_vector_it_holds(seed) -> None:
    """A sort of an ``(n, 1)`` column moved nothing, so the bound's top tail was the last rows in
    input order: -0.087 where the vector reads 0.529, below the sample's mean of 0.0."""
    rng = np.random.default_rng(seed)
    y = rng.normal(size=300)
    x = rng.normal(size=300)
    for gamma in (1.7, 2.0):  # a fractional tail at 1.7, a whole one at 2 on 300 outcomes
        assert msm_worst_case_mean(y[:, None], gamma) == msm_worst_case_mean(y, gamma)
        assert confounding_robust_radius(0.1, y[:, None], gamma) == confounding_robust_radius(
            0.1, y, gamma
        )
        assert msm_worst_case_mean(y[:, None], gamma, covariates=x) == msm_worst_case_mean(
            y, gamma, covariates=x
        )
    assert negative_control_gamma(y[:, None], level=0.95) == negative_control_gamma(y, level=0.95)
    assert msm_worst_case_mean(y[:, None], 2.0) > y.mean()


@pytest.mark.parametrize(
    "outcomes",
    [np.zeros((1, 10)), np.zeros((5, 2)), np.zeros((10, 1, 1)), np.float64(1.0), np.zeros(0)],
    ids=["row", "matrix", "three-d", "scalar", "empty"],
)
def test_outcomes_of_another_shape_are_refused_naming_it(outcomes) -> None:
    shape = re.escape(str(np.shape(outcomes)))
    for call in (
        lambda: msm_worst_case_mean(outcomes, 2.0),
        lambda: confounding_robust_radius(0.1, outcomes, 2.0),
        lambda: negative_control_gamma(outcomes),
    ):
        with pytest.raises(ValueError, match=rf"shape \(n,\) or \(n, 1\); got shape {shape}"):
            call()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_outcome_is_refused_rather_than_carried_to_the_bound(bad) -> None:
    """A nan outcome returned a nan bound and radius, on the pooled and Tan's paths alike, and an
    infinite one a nan bound, where :func:`negative_control_gamma` refused the same input."""
    y = np.random.default_rng(0).normal(size=300)
    y[5] = bad
    for call in (
        lambda: msm_worst_case_mean(y, 2.0),
        lambda: msm_worst_case_mean(y, 2.0, propensity=0.3),
        lambda: msm_worst_case_mean(y, 2.0, covariates=np.arange(300.0)),
        lambda: confounding_robust_radius(0.1, y, 2.0),
        lambda: negative_control_gamma(y),
    ):
        with pytest.raises(ValueError, match="outcomes must be finite"):
            call()


# the three functions on the panel below, each keyword combination's values in the loop's
# order, as the tree before ``covariate_form`` returned them: 1.0.0 under numpy 2.5.4 and
# scipy 1.18.1, keyed by the covariates and the propensity passed
_BEFORE_COVARIATE_FORM = {
    ("none", "none"): [
        0.14370377737484355,
        0.1,
        0.8351057267114872,
        0.7914019493366437,
        1.0332598490909783,
        0.9895560717161348,
        1.5035997694997947,
        1.4598959921249512,
        1.1134927531673142,
        1.0,
    ],
    ("none", "0.4"): [
        0.14370377737484352,
        0.09999999999999998,
        0.5585449469768299,
        0.5148411696019863,
        0.677437420404525,
        0.6337336430296815,
        0.9596413726498146,
        0.9159375952749711,
        1.1956877821340122,
        1.0,
    ],
    ("none", "e"): [
        0.14370377737484352,
        0.09999999999999998,
        0.48554773456641065,
        0.4418439571915671,
        0.582836967866459,
        0.5391331904916155,
        0.8119456764177474,
        0.7682418990429039,
        1.2420311528944508,
        1.0,
    ],
    ("x", "none"): [
        0.14370377737484355,
        0.1,
        0.7253608515785378,
        0.6816570742036943,
        0.8979960360175914,
        0.8542922586427478,
        1.3037831388155374,
        1.260079361440694,
        1.1364720872717982,
        1.0,
    ],
    ("x", "0.4"): [
        0.14370377737484355,
        0.1,
        0.4926980218970602,
        0.4489942445222166,
        0.5962791325604926,
        0.552575355185649,
        0.83975139423926,
        0.7960476168644165,
        1.2371137073650615,
        1.0,
    ],
    ("x", "e"): [
        0.14370377737484355,
        0.1,
        0.438043842408094,
        0.3943400650332505,
        0.5247246440978672,
        0.4810208667230237,
        0.7277162958373778,
        0.6840125184625343,
        1.2851212448200044,
        1.0,
    ],
}


def test_the_default_form_reads_as_it_did_before_the_quantile_form() -> None:
    """Every keyword combination the three functions took before ``covariate_form``, as the tree
    that did not have it returned them. To 1e-9, not to the bit: a digest of the bits took one
    value a class of machine and dependencies, a fourth on the CI runners, where the numbers
    differ by a few epsilons (under 5e-16 between the lockfile and the dependency floors), and the
    quantile form moves them in their first digits."""
    rng = np.random.default_rng(20261030)
    n = 200
    x = rng.standard_normal((n, 2))
    y = 0.3 + x @ np.array([1.0, -0.5]) + (1.0 + 0.5 * np.abs(x[:, 0])) * rng.standard_normal(n)
    e = rng.uniform(0.2, 0.8, n)
    passed = {"none": None, "x": x}, {"none": None, "0.4": 0.4, "e": e}
    for (c, covariates), (p, propensity) in itertools.product(*(d.items() for d in passed)):
        kwargs = {"covariates": covariates, "propensity": propensity}
        values = []
        for gamma in (1.0, 1.7, 2.0, 3.0):
            values.append(msm_worst_case_mean(y, gamma, **kwargs))
            values.append(confounding_robust_radius(0.1, y, gamma, **kwargs))
        values.extend(negative_control_gamma(y, level=level, **kwargs) for level in (None, 0.95))
        np.testing.assert_allclose(
            values, _BEFORE_COVARIATE_FORM[c, p], rtol=1e-9, atol=0.0, err_msg=f"{c=} {p=}"
        )
