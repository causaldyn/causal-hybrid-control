"""The MSM bound on a mean with a linear conditional quantile taken out of the tails
(``covariate_form="quantile"``), against ``validation/msm_quantile_balancing.mac``."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

import chc.uncertainty
from chc.uncertainty import (
    _box_worst_case,
    _covariate_design,
    _odds_weights,
    _quantile_level_bound,
    confounding_robust_radius,
    msm_worst_case_mean,
    negative_control_gamma,
)

# Q4: the box's inflation per unit sd at Gamma = 2, and the per-x bound on Y = 2X + (1 + 2X) eps
UNIT = 0.5453996620129765
PER_X_UNIFORM = 2.090799324025953


def _se(values) -> float:
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def _two_cells(s1: float, repeats: int = 2):
    """Q3's atoms (0, -1), (0, 1), (1, 4 - s1), (1, 4 + s1), unit ``i`` the atom ``(i // 5) % 4``,
    so each of the 5 interleaved folds holds every atom ``repeats`` times."""
    xs = np.array([0.0, 0.0, 1.0, 1.0])
    ys = np.array([-1.0, 1.0, 4.0 - s1, 4.0 + s1])
    atom = (np.arange(20 * repeats) // 5) % 4
    return xs[atom], ys[atom]


@pytest.mark.parametrize(
    ("s1", "tan", "expected"),
    [
        (1.0, False, (5 / 2, 5 / 2, 13 / 4)),
        (3.0, False, (3.0, 13 / 4, 15 / 4)),
        (1.0, True, (9 / 4, 9 / 4, 41 / 16)),
        (3.0, True, (19 / 8, 39 / 16, 41 / 16)),
    ],
    ids=["box-equal", "box-spread", "tan-equal", "tan-spread"],
)
def test_two_cells_read_maximas_linear_programs(s1, tan, expected) -> None:
    # Q3: quantile, residualised and pooled, cross-fitted; the quantile form is the per-x bound
    x, y = _two_cells(s1)
    e = np.where(x == 1.0, 0.75, 0.25) if tan else None
    quantile = msm_worst_case_mean(y, 2.0, covariates=x, propensity=e, covariate_form="quantile")
    residualised = msm_worst_case_mean(y, 2.0, covariates=x, propensity=e)
    pooled = msm_worst_case_mean(y, 2.0, propensity=e)
    found = (quantile, residualised, pooled)
    assert found == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("gamma", [1.7, 2.0, 3.0])
def test_in_sample_on_an_intercept_it_is_the_sample_bound(seed, gamma) -> None:
    # Q2: an intercept's quantile regression is the dual of the fill, its vertex an order statistic
    rng = np.random.default_rng(seed)
    n = 101
    y = rng.standard_normal(n)
    e = rng.uniform(0.1, 0.9, n)
    ones = np.ones((n, 1))
    box, _ = _quantile_level_bound(y, gamma, ones, None, cross_fitted=False)
    odds, _ = _quantile_level_bound(y, gamma, ones, e, cross_fitted=False)
    assert box == pytest.approx(msm_worst_case_mean(y, gamma), rel=1e-12, abs=0.0)
    fill, _ = _box_worst_case(y, *_odds_weights(gamma, e, n))
    assert odds == pytest.approx(fill, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("seed", range(3))
def test_in_sample_on_cell_dummies_it_is_each_cells_bound_by_its_share(seed) -> None:
    rng = np.random.default_rng(seed)
    cell = rng.choice(3, size=150, p=[0.2, 0.3, 0.5])
    spread = np.array([0.5, 1.0, 3.0])[cell]
    y = np.array([0.0, 2.0, -1.0])[cell] + spread * rng.standard_normal(150)
    design = _covariate_design(y, (cell[:, None] == np.arange(1, 3)).astype(np.float64))
    found, _ = _quantile_level_bound(y, 2.0, design, None, cross_fitted=False)
    per_cell = sum(np.mean(cell == c) * msm_worst_case_mean(y[cell == c], 2.0) for c in range(3))
    assert found == pytest.approx(per_cell, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("seed", [0, 1])
def test_a_fit_on_its_own_rows_would_read_too_narrow(seed) -> None:
    """Ten covariates that carry nothing, 60 outcomes: a quantile fitted on the rows it is read on
    reads below the population's bound, by about 9 of its standard errors, and so would the
    program's own optimum; fitted on the other folds, ``psi``'s mean is the population's bound or
    above in expectation, since ``q`` does not see the row it is read on."""
    rng = np.random.default_rng(seed)
    found = []
    for _ in range(100):
        x, y = rng.standard_normal((60, 10)), rng.standard_normal(60)
        found.append(msm_worst_case_mean(y, 2.0, covariates=x, covariate_form="quantile"))
    assert np.mean(found) >= UNIT - 3.0 * _se(found)


@pytest.mark.parametrize("seed", [3, 4])
def test_a_spread_that_moves_with_x_is_read_per_x(seed) -> None:
    # Q4b: Y = 2X + (1 + 2X) eps, X ~ U(0, 1): per-x 2.0908 where the residualised form reads 2.0998
    rng = np.random.default_rng(seed)
    quantile, residualised = [], []
    for _ in range(40):
        x = rng.uniform(0.0, 1.0, 2000)
        y = 2.0 * x + (1.0 + 2.0 * x) * rng.standard_normal(2000)
        quantile.append(msm_worst_case_mean(y, 2.0, covariates=x, covariate_form="quantile"))
        residualised.append(msm_worst_case_mean(y, 2.0, covariates=x))
    assert np.mean(quantile) == pytest.approx(PER_X_UNIFORM, abs=3.0 * _se(quantile), rel=0.0)
    gap = np.subtract(residualised, quantile)
    assert np.mean(gap) > 3.0 * _se(gap)


@pytest.mark.parametrize("seed", [5, 6])
def test_a_location_model_reads_alike_in_both_forms(seed) -> None:
    rng = np.random.default_rng(seed)
    quantile, residualised = [], []
    for _ in range(40):
        x = rng.uniform(0.0, 1.0, 1000)
        y = 1.0 + 2.0 * x + rng.standard_normal(1000)
        quantile.append(msm_worst_case_mean(y, 2.0, covariates=x, covariate_form="quantile"))
        residualised.append(msm_worst_case_mean(y, 2.0, covariates=x))
    for found in (quantile, residualised):
        assert np.mean(found) == pytest.approx(2.0 + UNIT, abs=3.0 * _se(found), rel=0.0)


@pytest.mark.parametrize("seed", [3, 4])
def test_a_quantile_the_columns_cannot_fit_leaves_the_bound_valid(seed) -> None:
    # sd(x) = 0.5 + 4 (x - 1/2)^2 is no linear function of x: the per-x bound is a floor
    rng = np.random.default_rng(seed)
    found = []
    for _ in range(40):
        x = rng.uniform(0.0, 1.0, 1000)
        y = x + (0.5 + 4.0 * (x - 0.5) ** 2) * rng.standard_normal(1000)
        found.append(msm_worst_case_mean(y, 2.0, covariates=x, covariate_form="quantile"))
    assert np.mean(found) >= 0.5 + UNIT * (0.5 + 4.0 / 12.0) - 3.0 * _se(found)


@pytest.mark.parametrize("seed", [20, 21])
def test_a_negative_control_keeps_its_level_on_a_heteroskedastic_null(seed) -> None:
    # whether the reading exceeds 1 is settled at Gamma = 1: a coarse search past it is enough
    rng, reps = np.random.default_rng(seed), 1000
    above = 0
    for _ in range(reps):
        x = rng.uniform(0.0, 1.0, 200)
        y = (1.0 + 2.0 * x) * rng.standard_normal(200)
        above += (
            negative_control_gamma(
                y, 1e-2, 10.0, level=0.95, covariates=x, covariate_form="quantile"
            )
            > 1.0
        )
    assert above / reps <= 0.05 + 2.0 * math.sqrt(0.05 * 0.95 / reps)


@pytest.mark.parametrize("seed", range(10))
def test_the_bound_grows_with_gamma_on_a_grid(seed) -> None:
    # psi grows with Gamma at a fixed quantile; refitted at each Gamma, it is checked, not proved
    rng = np.random.default_rng(100 + seed)
    x = rng.uniform(0.0, 1.0, 300)
    y = 2.0 * x + (1.0 + 2.0 * x) * rng.standard_normal(300)
    grid = [1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0]
    found = [msm_worst_case_mean(y, g, covariates=x, covariate_form="quantile") for g in grid]
    assert np.all(np.diff(found) >= 0.0)


def test_a_negative_control_reads_the_gamma_its_endpoint_crosses_zero_at() -> None:
    rng = np.random.default_rng(7)
    x = rng.uniform(0.0, 1.0, 400)
    y = 0.3 + (1.0 + 2.0 * x) * rng.standard_normal(400)
    gamma = negative_control_gamma(y, 1e-6, covariates=x, covariate_form="quantile")
    reflected = -np.sign(y.mean()) * y
    design = _covariate_design(reflected, x)
    assert _quantile_level_bound(reflected, gamma, design, None)[0] >= 0.0
    assert _quantile_level_bound(reflected, gamma * (1.0 - 1e-5), design, None)[0] < 0.0


def test_at_gamma_one_it_is_the_mean_and_the_radius_widens_by_the_gap() -> None:
    rng = np.random.default_rng(8)
    x = rng.uniform(0.0, 1.0, 500)
    y = x + rng.standard_normal(500)
    kwargs = {"covariates": x, "covariate_form": "quantile"}
    assert msm_worst_case_mean(y, 1.0, **kwargs) == float(np.mean(y))
    worst = msm_worst_case_mean(y, 3.0, propensity=0.4, **kwargs)
    radius = confounding_robust_radius(0.1, y, 3.0, propensity=0.4, **kwargs)
    assert radius == pytest.approx(0.1 + worst - y.mean(), rel=1e-12, abs=0.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"covariate_form": "quantile"}, "none were given"),
        ({"covariate_form": "quantile", "propensity": 0.3}, "none were given"),
        ({"covariate_form": "median", "covariates": np.arange(10.0)}, "'residualised' or"),
    ],
)
def test_each_function_refuses_a_form_it_cannot_read(kwargs, match) -> None:
    y = np.arange(10.0) - 3.0
    for call in (
        lambda: msm_worst_case_mean(y, 2.0, **kwargs),
        lambda: confounding_robust_radius(0.1, y, 2.0, **kwargs),
        lambda: negative_control_gamma(y, **kwargs),
    ):
        with pytest.raises(ValueError, match=match):
            call()


def test_a_program_highs_leaves_unsolved_is_refused(monkeypatch) -> None:
    def unsolved(*args, **kwargs):
        return SimpleNamespace(status=1, message="Iteration limit reached.", x=None)

    monkeypatch.setattr(chc.uncertainty, "linprog", unsolved)
    y = np.arange(20.0)
    with pytest.raises(RuntimeError, match="Iteration limit reached"):
        msm_worst_case_mean(y, 2.0, covariates=y, covariate_form="quantile")
