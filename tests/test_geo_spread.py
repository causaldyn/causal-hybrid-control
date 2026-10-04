"""chc.dlm.fit_geo_spread: the spread against a dense grid of the marginal likelihood, each
interval's ends where the likelihood ratio crosses its cut, the draws' mixture against quadrature
of the spread's posterior, and the spread of a Gaussian hierarchy recovered."""

from __future__ import annotations

import dataclasses
import logging
import math

import numpy as np
import pytest
from scipy import special

from chc.dlm import (
    GeoDLM,
    Polynomial,
    Prior,
    Regression,
    fit_geo_spread,
    forward_filter_geos,
    stacked_prior,
)

T = 40
VAGUEST = 100.0


def _hierarchy(
    seed: int,
    geos: int,
    spread: float,
    dof: float = math.inf,
    *,
    weeks: int = T,
    deviation: float = VAGUEST,
):
    """Each geo's coefficient the national 1.5 plus a normal deviation of standard deviation
    ``spread``, a level of its own, and unit noise; the prior states the variance at 1, so a
    spread's variance is in the KPI's units. ``deviation`` is the deviations' prior variance."""
    rng = np.random.default_rng(seed)
    x = rng.gamma(2.0, 1.0, (weeks, geos, 1))
    beta = 1.5 + spread * rng.standard_normal(geos)
    level = 3.0 + rng.standard_normal(geos)
    y = level + x[:, :, 0] * beta + rng.standard_normal((weeks, geos))
    national = Prior(np.zeros(1), VAGUEST * np.eye(1), 1.0, dof)
    regional = Prior(np.array([3.0, 0.0]), np.diag([VAGUEST, deviation]), 1.0, dof)
    model = GeoDLM(
        (Regression(1, 1.0),),
        (Polynomial(1, 1.0), Regression(1, 1.0)),
        geos,
        stacked_prior(national, regional, geos),
    )
    return model, y, x, beta


def _at(model, pooled, variance):
    cov = np.array(model.prior.covariance)
    for g in range(model.geos):
        at = 1 + 2 * g + np.array(pooled)
        cov[at, at] = variance
    prior = Prior(model.prior.mean, cov, model.prior.scale, model.prior.dof)
    return dataclasses.replace(model, prior=prior)


def _loglik(model, y, x, pooled, variance) -> float:
    return forward_filter_geos(_at(model, pooled, variance), y, x).log_likelihood


@pytest.mark.parametrize("dof", [math.inf, 4.0])
def test_one_spread_is_the_best_of_a_dense_grid_of_the_marginal_likelihood(dof):
    model, y, x, _ = _hierarchy(1, 6, 0.5, dof)
    spread = fit_geo_spread(model, y, x, [1], 8, 0)
    grid = VAGUEST * np.exp(np.linspace(math.log(1e-12), 0.0, 241))
    values = np.array([_loglik(model, y, x, [1], np.array([v])) for v in grid])
    assert spread.fit.log_likelihood >= values.max() - 1e-9
    step = math.log(grid[1] / grid[0])
    assert abs(math.log(spread.variance[0] / grid[int(np.argmax(values))])) <= step


def test_two_spreads_are_a_local_best_of_the_marginal_likelihood():
    model, y, x, _ = _hierarchy(2, 8, 0.5)
    spread = fit_geo_spread(model, y, x, [0, 1], 8, 0)
    top = spread.fit.log_likelihood
    for i in range(2):
        for sign in (-1.0, 1.0):
            moved = spread.variance.copy()
            moved[i] *= math.exp(sign * 0.05)
            assert _loglik(model, y, x, [0, 1], moved) <= top + 1e-9


def test_each_interval_end_is_where_the_likelihood_ratio_crosses_its_cut():
    model, y, x, _ = _hierarchy(3, 8, 0.5)
    spread = fit_geo_spread(model, y, x, [1], 8, 0, level=0.8)
    cut = float(special.chdtri(1, 0.2)) / 2.0
    top = spread.fit.log_likelihood
    for end in (spread.lower, spread.upper):
        assert top - _loglik(model, y, x, [1], end) == pytest.approx(cut, abs=5e-3)
    assert spread.lower[0] < spread.variance[0] < spread.upper[0]


def test_geos_that_do_not_differ_leave_a_spread_of_zero_in_the_interval():
    model, y, x, _ = _hierarchy(4, 8, 0.0)
    spread = fit_geo_spread(model, y, x, [1], 8, 0)
    assert spread.lower[0] == pytest.approx(VAGUEST * 1e-12)


def test_a_gaussian_hierarchy_s_spread_is_recovered_and_covered():
    model, y, x, beta = _hierarchy(5, 40, 0.5)
    spread = fit_geo_spread(model, y, x, [1], 8, 0)
    assert 0.6 * 0.25 < spread.variance[0] < 1.6 * 0.25
    assert spread.lower[0] < 0.25 < spread.upper[0]
    pooled = spread.fit.mean[-1][0] + spread.fit.mean[-1][2::2]
    alone = forward_filter_geos(model, y, x).mean[-1]
    alone = alone[0] + alone[2::2]
    assert np.mean((pooled - beta) ** 2) < np.mean((alone - beta) ** 2)


def _coefficient(fit):
    """Geo 0's coefficient, the national one plus its deviation, at the last step."""
    m, c = fit.mean[-1], fit.covariance[-1]
    return m[0] + m[2], c[0, 0] + c[2, 2] + 2.0 * c[0, 2]


@pytest.mark.parametrize(
    ("seed", "deviation", "where"),
    [(6, 1.0, "inside"), (11, 1.0, "least"), (6, 0.05, "most")],
)
def test_the_draws_mixture_is_the_spread_s_posterior_by_quadrature(seed, deviation, where):
    """Four geos over twelve weeks, so that one spread is weakly identified, with its best inside
    the range, where it pools the geos almost completely, and at the model's own, below the
    world's. Under a prior flat on the spread's standard deviation its posterior on a grid of its
    log is the likelihood times ``exp(log_variance / 2)``; geo 0's coefficient's mixture over the
    draws is its mixture over the grid within three of the draws' Monte Carlo standard errors."""
    model, y, x, _ = _hierarchy(seed, 4, 0.5, weeks=12, deviation=deviation)
    spread = fit_geo_spread(model, y, x, [1], 200, 7)
    best = spread.variance[0]
    assert {
        "inside": 1e-6 * deviation < best < deviation,
        "least": best < 1e-6 * deviation,
        "most": best == pytest.approx(deviation, rel=1e-12, abs=0.0),
    }[where]

    drawn = np.log(spread.draws)
    assert np.all(
        (drawn >= math.log(1e-12 * deviation) - 1e-9) & (drawn <= math.log(deviation) + 1e-9)
    )

    mean, variance = spread.mixture(_coefficient)
    w = spread.weights
    m, v = np.array(
        [_coefficient(forward_filter_geos(_at(model, [1], d), y, x)) for d in spread.draws]
    ).T
    assert mean == pytest.approx(w @ m, rel=1e-12, abs=0.0)
    assert variance == pytest.approx(w @ (v + (m - mean) ** 2), rel=1e-12, abs=0.0)
    # the self-normalised estimates' standard errors, by the delta method
    mean_error = math.sqrt(np.sum(w**2 * (m - mean) ** 2))
    variance_error = math.sqrt(np.sum(w**2 * (v + (m - mean) ** 2 - variance) ** 2))
    assert 1.0 / np.sum(w**2) > 100

    logs = np.linspace(math.log(deviation * 1e-12), math.log(deviation), 300)
    density, means, variances = [], [], []
    for value in logs:
        fit = forward_filter_geos(_at(model, [1], [math.exp(value)]), y, x)
        density.append(fit.log_likelihood + value / 2.0)
        m_at, v_at = _coefficient(fit)
        means.append(m_at)
        variances.append(v_at)
    q = np.exp(np.array(density) - max(density))
    q /= q.sum()
    expected = float(q @ np.array(means))
    expected_var = float(q @ (np.array(variances) + (np.array(means) - expected) ** 2))
    assert mean == pytest.approx(expected, abs=3.0 * mean_error)
    assert variance == pytest.approx(expected_var, abs=3.0 * variance_error)


def test_at_sets_the_pooled_variance_in_every_geo_and_nothing_else(caplog):
    model, y, x, _ = _hierarchy(7, 3, 0.5)
    with caplog.at_level(logging.INFO, logger="chc.dlm"):
        spread = fit_geo_spread(model, y, x, [1], 4, 0)
    assert [r.chc_event for r in caplog.records if r.name == "chc.dlm"] == ["dlm_geo_spread"]
    cov = np.array(model.prior.covariance)
    cov[[2, 4, 6], [2, 4, 6]] = 0.5
    np.testing.assert_array_equal(spread.at([0.5]).prior.covariance, cov)
    np.testing.assert_array_equal(spread.fit.model.prior.mean, model.prior.mean)
    assert spread.draws.shape == (4, 1)
    assert spread.weights.sum() == pytest.approx(1.0)


def _uneven():
    model, *_ = _hierarchy(8, 2, 0.5)
    cov = np.array(model.prior.covariance)
    cov[4, 4] = 50.0
    return dataclasses.replace(
        model, prior=Prior(model.prior.mean, cov, model.prior.scale, model.prior.dof)
    )


def _correlated():
    model, *_ = _hierarchy(8, 2, 0.5)
    cov = np.array(model.prior.covariance)
    cov[1, 2] = cov[2, 1] = 1.0
    return dataclasses.replace(
        model, prior=Prior(model.prior.mean, cov, model.prior.scale, model.prior.dof)
    )


@pytest.mark.parametrize(
    ("model", "pooled", "draws", "level", "match"),
    [
        (None, [], 4, 0.9, "pooled must name distinct coordinates"),
        (None, [2], 4, 0.9, "pooled must name distinct coordinates"),
        (None, [1, 1], 4, 0.9, "pooled must name distinct coordinates"),
        (None, [True], 4, 0.9, "pooled must name distinct coordinates"),
        (_uneven, [1], 4, 0.9, "a spread is one variance for every geo"),
        (_correlated, [0], 4, 0.9, "a spread is a variance of its own"),
        (None, [1], 0, 0.9, "draws must be a positive integer"),
        (None, [1], 4, 1.0, r"level must be in \(0, 1\)"),
    ],
)
def test_it_refuses_what_it_cannot_fit(model, pooled, draws, level, match):
    base, y, x, _ = _hierarchy(8, 2, 0.5)
    with pytest.raises(ValueError, match=match):
        fit_geo_spread(base if model is None else model(), y, x, pooled, draws, 0, level=level)
