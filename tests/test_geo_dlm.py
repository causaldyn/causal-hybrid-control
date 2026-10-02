"""chc.dlm over geos: one geo against chc.dlm's own filter, many against the conjugate regression
on every observation at once, geos that share nothing against a filter each, and one step's
score against SciPy's multivariate t.

The update is West and Harrison's for a vector with one variance scale; validation/geo_dlm.mac
shows it is the sequential one and its density the product of the sequential ones.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import linalg, stats

from chc.dlm import (
    DynamicLinearModel,
    GeoDLM,
    Polynomial,
    Prior,
    Regression,
    Seasonal,
    forward_filter,
    forward_filter_geos,
    stacked_prior,
)

T = 40


def _prior(p: int, *, scale: float = 1.3, dof: float = 6.0, spread: float = 4.0) -> Prior:
    return Prior(np.linspace(-0.5, 0.5, p), spread * np.eye(p), scale, dof)


def _panel(seed: int, geos: int, width: int, missing: float = 0.0):
    """A KPI a geo, its columns, and a share of missing observations."""
    rng = np.random.default_rng(seed)
    x = rng.gamma(2.0, 1.0, (T, geos, width))
    x[rng.random((T, geos, width)) < 0.15] = 0.0  # idle regressors
    y = 2.0 + x.sum(axis=2) + rng.standard_normal((T, geos))
    y[rng.random((T, geos)) < missing] = np.nan
    return y, x


NATIONAL = (Regression(2, 0.97),)
REGIONAL = (Polynomial(2, 0.9), Seasonal(13.0, (1, 2), 0.95), Regression(2, 0.99))


@pytest.mark.parametrize("form", ["additive", "multiplicative"])
@pytest.mark.parametrize(
    ("beta", "dof", "hold"), [(1.0, 6.0, False), (0.95, 6.0, True), (1.0, math.inf, False)]
)
def test_one_geo_is_chc_dlm_s_filter_of_its_national_and_regional_blocks(form, beta, dof, hold):
    regional = (*REGIONAL[:2], Regression(2, 0.99, hold_when_idle=hold))
    y, x = _panel(1, 1, 2, missing=0.1)
    prior = _prior(2 + 2 + 4 + 2, dof=dof)
    geo = forward_filter_geos(GeoDLM(NATIONAL, regional, 1, prior, form, beta), y, x)
    alone = forward_filter(
        DynamicLinearModel(NATIONAL + regional, prior, form, beta),
        y[:, 0],
        np.concatenate([x[:, 0], x[:, 0]], axis=1),  # the national block's columns, then the geo's
    )
    pairs = [
        (geo.prior_mean, alone.prior_mean),
        (geo.prior_covariance, alone.prior_covariance),
        (geo.one_step_mean[:, 0], alone.one_step_mean),
        (geo.one_step_scale[:, 0, 0], alone.one_step_scale),
        (geo.one_step_dof, alone.one_step_dof),
        (geo.mean, alone.mean),
        (geo.covariance, alone.covariance),
        (geo.dof, alone.dof),
        (geo.scale, alone.scale),
    ]
    for ours, theirs in pairs:
        np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=1e-12 * np.abs(theirs).max())
    np.testing.assert_allclose(geo.log_scores, alone.log_scores, rtol=1e-12)
    assert geo.log_likelihood == pytest.approx(alone.log_likelihood, rel=1e-12)


def test_with_every_discount_one_the_filter_is_the_conjugate_regression_on_every_observation():
    """Nothing discounted, the state moves by G alone, ``theta_t = G^(t+1) theta_0``: every
    observation is a row of one regression on ``theta_0``, missing ones dropped. Its Normal-Gamma
    posterior and its marginal, a multivariate t of every observation at once, are the filter's
    last step and log-likelihood."""
    geos = 4
    national = (Regression(2, 1.0),)
    regional = (Polynomial(1, 1.0), Seasonal(12.0, (1,), 1.0), Regression(1, 1.0))
    weights = np.array([0.5, 1.0, 2.0, 1.5])
    prior = stacked_prior(
        Prior(np.array([0.3, -0.1]), np.diag([2.0, 1.0]), 1.7, 5.0),
        Prior(np.array([1.0, 0.2, -0.2, 0.0]), np.diag([3.0, 0.5, 0.5, 0.4]), 1.7, 5.0),
        geos,
    )
    model = GeoDLM(national, regional, geos, prior, relative_variance=weights)
    y, x = _panel(2, geos, 2, missing=0.2)
    fit = forward_filter_geos(model, y, x)

    g = DynamicLinearModel(national + regional * geos, prior).size
    evolution = linalg.block_diag(np.eye(2), *[linalg.block_diag(1.0, _rotation(12.0), 1.0)] * geos)
    assert evolution.shape == (g, g)
    rows, values, variances = [], [], []
    moved = np.eye(g)
    for t in range(T):
        moved = evolution @ moved
        for k in range(geos):
            if not np.isnan(y[t, k]):
                rows.append(fit.design[t, k] @ moved)
                values.append(y[t, k])
                variances.append(weights[k])
    design, kpi, w = np.array(rows), np.array(values), np.diag(variances)
    m0, c0, s0, n0 = prior.mean, prior.covariance, prior.scale, prior.dof

    marginal = stats.multivariate_t(design @ m0, design @ c0 @ design.T + s0 * w, df=n0)
    assert fit.log_likelihood == pytest.approx(marginal.logpdf(kpi), rel=1e-10)

    precision = s0 * np.linalg.inv(c0) + design.T @ np.linalg.inv(w) @ design
    mean = np.linalg.solve(precision, s0 * np.linalg.solve(c0, m0) + design.T @ (kpi / variances))
    error = kpi - design @ m0
    n = n0 + kpi.size
    s = (n0 * s0 + error @ np.linalg.solve(design @ c0 @ design.T / s0 + w, error)) / n
    assert fit.dof[-1] == n
    assert fit.scale[-1] == pytest.approx(s, rel=1e-10)
    np.testing.assert_allclose(fit.mean[-1], moved @ mean, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(
        fit.covariance[-1], s * moved @ np.linalg.inv(precision) @ moved.T, rtol=1e-8, atol=1e-12
    )


def _rotation(period: float) -> np.ndarray:
    w = 2.0 * math.pi / period
    return np.array([[math.cos(w), math.sin(w)], [-math.sin(w), math.cos(w)]])


def test_geos_that_share_nothing_filter_as_one_filter_each_when_the_variance_is_known():
    """No national block and a known variance: nothing couples the geos, so each geo's
    coordinates follow chc.dlm's filter of that geo alone, at its own variance."""
    geos, scale = 3, 1.3
    regional = (Polynomial(1, 0.9), Regression(2, 0.95))
    weights = np.array([0.6, 1.0, 1.8])
    own = Prior(np.array([1.0, 0.5, -0.5]), np.diag([4.0, 1.0, 1.0]), scale, math.inf)
    model = GeoDLM((), regional, geos, stacked_prior(None, own, geos), relative_variance=weights)
    y, x = _panel(3, geos, 2, missing=0.15)
    fit = forward_filter_geos(model, y, x)
    total = 0.0
    for k in range(geos):
        alone = forward_filter(
            DynamicLinearModel(
                regional, Prior(own.mean, own.covariance, scale * weights[k], math.inf)
            ),
            y[:, k],
            x[:, k],
        )
        mine = slice(3 * k, 3 * k + 3)
        np.testing.assert_allclose(fit.mean[:, mine], alone.mean, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(fit.covariance[:, mine, mine], alone.covariance, rtol=1e-12)
        total += alone.log_likelihood
    assert fit.log_likelihood == pytest.approx(total, rel=1e-12)


def test_a_step_s_score_is_the_observed_geos_multivariate_t():
    geos = 3
    model = GeoDLM(
        NATIONAL,
        (Polynomial(1, 0.9),),
        geos,
        stacked_prior(_prior(2), _prior(1), geos),
        relative_variance=[1.0, 2.0, 0.5],
    )
    y, x = _panel(4, geos, 2)
    y[5, 1] = np.nan
    fit = forward_filter_geos(model, y, x)
    for t in (0, 5):
        seen = ~np.isnan(y[t])
        q = fit.one_step_scale[t][np.ix_(seen, seen)]
        expected = stats.multivariate_t(
            fit.one_step_mean[t, seen], q, df=fit.one_step_dof[t]
        ).logpdf(y[t, seen])
        assert fit.log_scores[t] == pytest.approx(expected, rel=1e-12)
    assert fit.dof[5] - fit.dof[4] == 2


def test_a_step_with_no_geo_observed_evolves_and_does_not_update():
    geos = 2
    model = GeoDLM(NATIONAL, (Polynomial(1, 0.9),), geos, stacked_prior(_prior(2), _prior(1), geos))
    y, x = _panel(5, geos, 2)
    y[7] = np.nan
    fit = forward_filter_geos(model, y, x)
    np.testing.assert_array_equal(fit.mean[7], fit.prior_mean[7])
    np.testing.assert_array_equal(fit.covariance[7], fit.prior_covariance[7])
    assert fit.dof[7] == fit.dof[6]
    assert math.isnan(fit.log_scores[7])
    assert np.isnan(fit.errors[7]).all()


@pytest.mark.parametrize(("national", "regional"), [(2, 1), (1, 2)])
def test_a_geo_reads_the_national_blocks_and_its_own_and_no_other_geo_s(national, regional):
    """Each regression reads the geo's columns from the first, whichever is the wider."""
    geos = 3
    model = GeoDLM(
        (Polynomial(1, 0.9), Regression(national, 0.97)),
        (Polynomial(1, 0.9), Regression(regional, 0.99)),
        geos,
        stacked_prior(_prior(1 + national), _prior(1 + regional), geos),
    )
    y, x = _panel(6, geos, 2)
    design = forward_filter_geos(model, y, x).design
    width = 1 + regional
    for k in range(geos):
        expected = np.zeros((T, 1 + national + width * geos))
        expected[:, 0] = 1.0
        expected[:, 1 : 1 + national] = x[:, k, :national]
        own = 1 + national + width * k
        expected[:, own] = 1.0
        expected[:, own + 1 : own + width] = x[:, k, :regional]
        np.testing.assert_array_equal(design[:, k], expected)


def test_a_regressor_held_while_idle_is_live_where_any_geo_that_reads_it_moves():
    """At a step where the first geo's columns are zero and the second's are not, the national
    coefficients and the second geo's deviations are discounted, the first geo's are not."""
    geos, delta = 2, 0.9
    held = Regression(1, delta, hold_when_idle=True)
    model = GeoDLM((held,), (held,), geos, stacked_prior(_prior(1), _prior(1), geos))
    y, x = _panel(7, geos, 1)
    x[:, 0, 0] = 1.0
    x[:, 1, 0] = 1.0
    x[10, 0, 0] = 0.0
    fit = forward_filter_geos(model, y, x)
    grown = np.diag(fit.prior_covariance[10]) / np.diag(fit.covariance[9])
    np.testing.assert_allclose(grown, [1.0 / delta, 1.0, 1.0 / delta], rtol=1e-12)


def test_a_stacked_prior_repeats_the_regional_one_for_each_geo():
    national = _prior(2)
    regional = Prior(np.array([1.0, 2.0]), np.array([[2.0, 0.5], [0.5, 1.0]]), 1.3, 6.0)
    prior = stacked_prior(national, regional, 3)
    np.testing.assert_array_equal(prior.mean, [-0.5, 0.5, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    np.testing.assert_array_equal(prior.covariance[4:6, 4:6], regional.covariance)
    assert not prior.covariance[2:4, 4:6].any()
    assert (prior.scale, prior.dof) == (1.3, 6.0)


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: GeoDLM(NATIONAL, REGIONAL, 0, _prior(2)), "geos must be a positive integer"),
        (
            lambda: GeoDLM(NATIONAL, (), 2, _prior(2), relative_variance=[1.0, 0.0]),
            "relative_variance must be 2 positive finite numbers",
        ),
        (lambda: GeoDLM(NATIONAL, REGIONAL, 2, _prior(2)), "the prior has 2 coordinates"),
        (
            lambda: stacked_prior(_prior(2), _prior(1, scale=2.0), 2),
            "must state one variance",
        ),
        (lambda: stacked_prior(None, None, 2), "needs a national prior, a regional one"),
        (
            lambda: forward_filter_geos(GeoDLM(NATIONAL, (), 2, _prior(2)), np.zeros((T, 3))),
            r"y must be \(T, 2\)",
        ),
        (
            lambda: forward_filter_geos(
                GeoDLM(NATIONAL, (), 2, _prior(2)), np.zeros((T, 2)), np.zeros((T, 2, 3))
            ),
            r"x must have shape \(40, 2, 2\)",
        ),
        (
            lambda: forward_filter_geos(GeoDLM(NATIONAL, (), 2, _prior(2)), np.zeros((T, 2))),
            "need x with 2 columns a geo",
        ),
    ],
)
def test_it_refuses_what_it_cannot_filter(build, match):
    with pytest.raises(ValueError, match=match):
        build()
