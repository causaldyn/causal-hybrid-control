"""chc.dlm against closed forms, a batch posterior built independently, and its own sampler."""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
from scipy import stats

from chc.dlm import (
    DiscountForm,
    DynamicLinearModel,
    Polynomial,
    Prior,
    Regression,
    Seasonal,
    backward_sample,
    decompose,
    forecast,
    forward_filter,
    monitor_evalues,
    smooth,
)
from chc.gate import DriftAlarm


def _prior(p: int, *, scale: float = 1.0, dof: float = math.inf, spread: float = 10.0) -> Prior:
    return Prior(np.zeros(p), spread * np.eye(p), scale, dof)


def _world(seed: int, horizon: int = 60, width: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """A regression whose coefficients drift, with an intercept column."""
    rng = np.random.default_rng(seed)
    x = np.column_stack([np.ones(horizon), rng.gamma(2.0, 1.0, (horizon, width - 1))])
    beta = np.cumsum(0.05 * rng.standard_normal((horizon, width)), axis=0) + 1.0
    y = np.sum(x * beta, axis=1) + 0.5 * rng.standard_normal(horizon)
    return x, y


def test_filter_equals_the_conjugate_regression_when_nothing_is_discounted():
    x, y = _world(0)
    m0, spread, s0, n0 = np.array([0.5, -0.2]), 4.0, 2.0, 3.0
    model = DynamicLinearModel((Regression(2, 1.0),), Prior(m0, spread * np.eye(2), s0, n0))
    fit = forward_filter(model, y, x)
    # the Normal-Gamma posterior in closed form, with the prior covariance stated at s0
    precision0 = np.eye(2) / (spread / s0)
    precision = precision0 + x.T @ x
    mean = np.linalg.solve(precision, precision0 @ m0 + x.T @ y)
    n = n0 + y.size
    d = n0 * s0 + y @ y + m0 @ precision0 @ m0 - mean @ precision @ mean
    np.testing.assert_allclose(fit.mean[-1], mean, rtol=1e-10)
    np.testing.assert_allclose(fit.dof[-1], n)
    np.testing.assert_allclose(fit.scale[-1], d / n, rtol=1e-10)
    np.testing.assert_allclose(fit.covariance[-1], d / n * np.linalg.inv(precision), rtol=1e-9)


def test_a_single_discount_is_discount_weighted_least_squares():
    x, y = _world(1)
    delta, v = 0.9, 0.25
    model = DynamicLinearModel((Regression(2, delta),), _prior(2, scale=v, spread=3.0))
    fit = forward_filter(model, y, x)
    horizon = y.size
    w = delta ** (horizon - 1 - np.arange(horizon))
    info = delta**horizon * np.eye(2) / 3.0 + (x * w[:, None]).T @ x / v
    mean = np.linalg.solve(info, (x * w[:, None]).T @ y / v)
    np.testing.assert_allclose(fit.mean[-1], mean, rtol=1e-9)
    np.testing.assert_allclose(fit.covariance[-1], np.linalg.inv(info), rtol=1e-9)


def test_a_discounted_level_settles_at_its_steady_state():
    delta, v = 0.8, 2.0
    y = np.random.default_rng(2).standard_normal(400)
    fit = forward_filter(DynamicLinearModel((Polynomial(1, delta),), _prior(1, scale=v)), y)
    c = fit.covariance[-1, 0, 0]
    gain = fit.prior_covariance[-1, 0, 0] / fit.one_step_scale[-1]
    assert c == pytest.approx(v * (1.0 - delta), rel=1e-12)
    assert gain == pytest.approx(1.0 - delta, rel=1e-12)
    w = fit.prior_covariance[-1, 0, 0] - fit.covariance[-2, 0, 0]
    assert w / v == pytest.approx((1.0 - delta) ** 2 / delta, rel=1e-12)


def test_log_likelihood_sums_student_t_densities():
    x, y = _world(3)
    model = DynamicLinearModel(
        (Polynomial(1, 0.95), Regression(1, 0.98)), _prior(2, scale=0.5, dof=4.0), "additive", 0.97
    )
    fit = forward_filter(model, y, x[:, 1])
    expected = stats.t.logpdf(
        y, fit.one_step_dof, loc=fit.one_step_mean, scale=np.sqrt(fit.one_step_scale)
    ).sum()
    assert fit.log_likelihood == pytest.approx(expected, rel=1e-12)


def test_the_variance_degrees_of_freedom_follow_the_variance_discount():
    x, y = _world(4)
    beta, n0 = 0.9, 2.0
    model = DynamicLinearModel((Regression(2, 0.95),), _prior(2, dof=n0), variance_discount=beta)
    fit = forward_filter(model, y, x)
    n = n0
    for t in range(y.size):
        assert fit.one_step_dof[t] == pytest.approx(beta * n)
        n = beta * n + 1.0
        assert fit.dof[t] == pytest.approx(n)


def test_a_missing_observation_evolves_without_updating():
    x, y = _world(5)
    y = y.copy()
    y[[10, 11]] = np.nan
    model = DynamicLinearModel((Regression(2, 0.95),), _prior(2, dof=5.0), variance_discount=0.95)
    fit = forward_filter(model, y, x)
    for t in (10, 11):
        np.testing.assert_array_equal(fit.mean[t], fit.prior_mean[t])
        np.testing.assert_array_equal(fit.covariance[t], fit.prior_covariance[t])
        assert fit.scale[t] == fit.scale[t - 1]
        assert fit.dof[t] == pytest.approx(0.95 * fit.dof[t - 1])
        assert math.isnan(fit.log_scores[t])
    assert np.isfinite(fit.log_likelihood)


def _two_blocks(form: DiscountForm, d1: float, d2: float) -> DynamicLinearModel:
    return DynamicLinearModel(
        (Polynomial(1, d1), Regression(1, d2)),
        Prior(np.zeros(2), np.array([[2.0, 0.7], [0.7, 1.0]]), 1.0, math.inf),
        form,
    )


def test_the_two_discount_forms_evolve_the_covariance_as_stated():
    y = np.array([1.0])
    x = np.array([2.0])
    p = np.array([[2.0, 0.7], [0.7, 1.0]])
    d = np.array([0.9, 0.7])
    additive = forward_filter(_two_blocks("additive", *d), y, x).prior_covariance[0]
    multiplicative = forward_filter(_two_blocks("multiplicative", *d), y, x).prior_covariance[0]
    np.testing.assert_allclose(additive, p + np.diag(np.diag(p) * (1 - d) / d), rtol=1e-14)
    np.testing.assert_allclose(multiplicative, p / np.sqrt(np.outer(d, d)), rtol=1e-14)
    # one discount for every block: the multiplicative form is the single discount, the additive
    # one still leaves the covariance between blocks as it was
    single = forward_filter(_two_blocks("multiplicative", 0.8, 0.8), y, x).prior_covariance[0]
    np.testing.assert_allclose(single, p / 0.8, rtol=1e-14)
    shared = forward_filter(_two_blocks("additive", 0.8, 0.8), y, x).prior_covariance[0]
    assert shared[0, 1] == pytest.approx(0.7)


def test_an_idle_regressor_winds_up_unless_held(caplog):
    delta, idle = 0.9, 30
    y = np.random.default_rng(6).standard_normal(idle)
    x = np.zeros(idle)
    loose = DynamicLinearModel((Regression(1, delta),), _prior(1, spread=2.0))
    with caplog.at_level(logging.WARNING, logger="chc.dlm"):
        fit = forward_filter(loose, y, x)
    assert fit.covariance[-1, 0, 0] == pytest.approx(2.0 * delta**-idle, rel=1e-12)
    assert any(getattr(r, "chc_event", None) == "dlm_windup" for r in caplog.records)
    held = DynamicLinearModel((Regression(1, delta, hold_when_idle=True),), _prior(1, spread=2.0))
    assert forward_filter(held, y, x).covariance[-1, 0, 0] == pytest.approx(2.0, rel=1e-12)


def test_a_seasonal_block_rotates_once_a_period():
    block = Seasonal(12, (1, 2, 6), 1.0)
    g = block._evolution()
    assert block.size == 5
    np.testing.assert_allclose(np.linalg.matrix_power(g, 12), np.eye(5), atol=1e-12)
    t = np.arange(48)
    y = 2.0 * np.cos(2 * np.pi * t / 12 + 0.3) + 0.5 * np.cos(np.pi * t)
    model = DynamicLinearModel((block,), _prior(5, scale=1e-6, spread=100.0))
    fit = forward_filter(model, y)
    np.testing.assert_allclose(fit.one_step_mean[24:], y[24:], atol=1e-4)


def test_a_linear_growth_block_forecasts_a_line():
    y = 3.0 + 0.5 * np.arange(30)
    model = DynamicLinearModel((Polynomial(2, 1.0),), _prior(2, scale=1e-6, spread=100.0))
    ahead = forecast(forward_filter(model, y), 5)
    np.testing.assert_allclose(ahead.mean, 3.0 + 0.5 * np.arange(30, 35), rtol=1e-6)


def test_forecast_variance_under_each_evolution_policy():
    delta, v = 0.95, 1.0
    y = np.random.default_rng(7).standard_normal(50)
    fit = forward_filter(DynamicLinearModel((Polynomial(1, delta),), _prior(1, scale=v)), y)
    c = fit.covariance[-1, 0, 0]
    k = np.arange(1, 53)
    constant = forecast(fit, 52)
    compounding = forecast(fit, 52, evolution="compounding")
    np.testing.assert_allclose(constant.scale, c + k * c * (1 - delta) / delta + v, rtol=1e-12)
    np.testing.assert_allclose(compounding.scale, c * delta**-k + v, rtol=1e-12)
    ratio = (compounding.state_scale[-1, 0, 0]) / (constant.state_scale[-1, 0, 0])
    assert ratio == pytest.approx(3.854, abs=5e-4)


def test_a_constant_forecast_refuses_an_evolution_variance_that_is_not_one():
    # x = 1 at every step: the level and the coefficient are learned only through their sum, so
    # their posterior correlation is strong, and with discounts that differ D P D - P is indefinite
    y = 3.0 + np.random.default_rng(5).standard_normal(30)
    ones = np.ones(10)
    blocks = (Polynomial(1, 0.99), Regression(1, 0.8))
    multiplicative = forward_filter(
        DynamicLinearModel(blocks, _prior(2), "multiplicative"), y, np.ones(30)
    )
    with pytest.raises(ValueError, match="not a variance"):
        forecast(multiplicative, 10, ones)
    compounding = forecast(multiplicative, 10, ones, evolution="compounding")
    assert np.all(np.linalg.eigvalsh(compounding.state_scale) > 0.0)
    additive = forward_filter(DynamicLinearModel(blocks, _prior(2)), y, np.ones(30))
    assert np.all(np.linalg.eigvalsh(forecast(additive, 10, ones).state_scale) > 0.0)


def _joint(fit, w_scale: np.ndarray, r1: np.ndarray, v: float):
    from chc.dlm import _structure

    g = _structure(fit.model).evolution
    horizon, p = fit.mean.shape
    size = horizon * p
    info = np.zeros((size, size))
    vec = np.zeros(size)
    blk = [slice(t * p, (t + 1) * p) for t in range(horizon)]
    r1_inv = np.linalg.inv(r1)
    info[blk[0], blk[0]] += r1_inv
    vec[blk[0]] += r1_inv @ fit.prior_mean[0]
    for t in range(1, horizon):
        wi = np.linalg.inv(w_scale[t])
        info[blk[t], blk[t]] += wi
        info[blk[t - 1], blk[t - 1]] += g.T @ wi @ g
        info[blk[t], blk[t - 1]] -= wi @ g
        info[blk[t - 1], blk[t]] -= g.T @ wi
    for t in range(horizon):
        f = fit.design[t]
        info[blk[t], blk[t]] += np.outer(f, f) / v
        vec[blk[t]] += f * fit.y[t] / v
    cov = np.linalg.inv(info)
    mean = cov @ vec
    return mean.reshape(horizon, p), cov, blk


def _evolution_variances(fit, units: np.ndarray) -> np.ndarray:
    from chc.dlm import _structure

    g = _structure(fit.model).evolution
    w = np.empty_like(fit.prior_covariance)
    for t in range(1, fit.mean.shape[0]):
        w[t] = (fit.prior_covariance[t] - g @ fit.covariance[t - 1] @ g.T) / units[t]
    return w


def _small_fit(dof: float, beta: float = 1.0, seed: int = 8):
    x, y = _world(seed, horizon=25)
    model = DynamicLinearModel(
        (Polynomial(1, 0.9), Regression(1, 0.85)),
        Prior(np.zeros(2), 5.0 * np.eye(2), 0.4, dof),
        variance_discount=beta,
    )
    return forward_filter(model, y, x[:, 1])


def test_smoother_equals_the_batch_posterior_with_a_known_variance():
    fit = _small_fit(math.inf)
    v = fit.model.prior.scale
    w = _evolution_variances(fit, np.ones(fit.y.size))
    mean, cov, blk = _joint(fit, w, fit.prior_covariance[0], v)
    sm = smooth(fit)
    np.testing.assert_allclose(sm.mean, mean, rtol=1e-8, atol=1e-10)
    for t in range(fit.y.size):
        np.testing.assert_allclose(sm.covariance[t], cov[blk[t], blk[t]], rtol=1e-8, atol=1e-12)
    for t in range(fit.y.size - 1):
        np.testing.assert_allclose(
            sm.cross_covariance[t], cov[blk[t], blk[t + 1]], rtol=1e-8, atol=1e-12
        )


def test_smoother_carries_the_final_variance_estimate_to_every_step():
    """With a learned variance the posterior given the precision is Gaussian with covariance
    Sigma* / phi, so every smoothed covariance is E[1 / phi | D_T] = S_T n_T / (n_T - 2) times
    Sigma*. A smoother that keeps each C_t at its own S_t (pydlm's) misses by S_T / S_t, and this
    test fails it."""
    fit = _small_fit(3.0)
    s_prev = np.concatenate([[fit.model.prior.scale], fit.scale[:-1]])
    w = _evolution_variances(fit, s_prev)
    mean, cov, blk = _joint(fit, w, fit.prior_covariance[0] / fit.model.prior.scale, 1.0)
    factor = fit.scale[-1] * fit.dof[-1] / (fit.dof[-1] - 2.0)
    sm = smooth(fit)
    assert sm.exact
    np.testing.assert_allclose(sm.mean, mean, rtol=1e-8, atol=1e-10)
    for t in range(fit.y.size):
        np.testing.assert_allclose(
            sm.covariance[t], factor * cov[blk[t], blk[t]], rtol=1e-8, atol=1e-12
        )

    # the mutant: RTS on C_t and R_(t+1) as the filter states them, rescaled only at the end
    from chc.dlm import _gains

    b = _gains(fit)
    mutant = fit.covariance[-1].copy()
    worst = 0.0
    for t in range(fit.y.size - 2, -1, -1):
        mutant = fit.covariance[t] + b[t] @ (mutant - fit.prior_covariance[t + 1]) @ b[t].T
        truth = fit.scale[-1] * cov[blk[t], blk[t]]
        worst = max(worst, float(np.max(np.abs(mutant - truth) / np.abs(truth).max())))
    assert worst > 0.05


def test_backward_sampling_matches_the_smoother():
    fit = _small_fit(6.0)
    sm = smooth(fit)
    draws = backward_sample(fit, 20_000, seed=9)
    mean = draws.states.mean(axis=0)
    sd = np.sqrt(np.einsum("tii->ti", sm.covariance))
    assert np.max(np.abs(mean - sm.mean) / (sd / math.sqrt(20_000))) < 4.5
    var = draws.states.var(axis=0)
    np.testing.assert_allclose(var, sd**2, rtol=0.06)
    lag = np.mean(
        (draws.states[:, :-1, 0] - mean[:-1, 0]) * (draws.states[:, 1:, 0] - mean[1:, 0]), axis=0
    )
    np.testing.assert_allclose(lag, sm.cross_covariance[:, 0, 0], rtol=0.08, atol=1e-3)


def test_backward_sampling_with_a_moving_variance_matches_the_smoother():
    fit = _small_fit(5.0, beta=0.9)
    sm = smooth(fit)
    assert not sm.exact
    draws = backward_sample(fit, 20_000, seed=10)
    sd = np.sqrt(np.einsum("tii->ti", sm.covariance))
    z = (draws.states.mean(axis=0) - sm.mean) / (sd / math.sqrt(20_000))
    assert np.max(np.abs(z)) < 4.5
    np.testing.assert_allclose(draws.states.var(axis=0), sd**2, rtol=0.06)
    # E[V_t | D_T] by quadrature, and the matched gamma's mean and variance of the precision
    from chc.dlm import _variance_factor

    factor, dof = _variance_factor(fit)
    np.testing.assert_allclose(draws.variance.mean(axis=0), factor, rtol=0.03)
    precision = 1.0 / draws.variance
    mean = precision.mean(axis=0)
    np.testing.assert_allclose(precision.var(axis=0), 2.0 * mean**2 / dof, rtol=0.05)


def _parts_fit():
    """A level with a slope, two regression columns, two harmonics of 12, and a learned variance."""
    rng = np.random.default_rng(12)
    horizon = 48
    t = np.arange(horizon)
    x = rng.gamma(2.0, 1.0, (horizon, 2))
    y = 5.0 + 0.1 * t + x @ np.array([1.0, -0.5]) + np.sin(2.0 * np.pi * t / 12.0)
    y = y + 0.3 * rng.standard_normal(horizon)
    model = DynamicLinearModel(
        (Polynomial(2, 0.95), Regression(2, 0.9), Seasonal(12.0, (1, 2), 0.98)),
        Prior(np.zeros(8), 10.0 * np.eye(8), 0.5, 3.0),
    )
    return forward_filter(model, y, x), x


def test_the_decomposition_adds_up_to_the_mean_draw_by_draw():
    fit, x = _parts_fit()
    draws = backward_sample(fit, 50, seed=3)
    parts = decompose(fit, draws)
    assert parts.shape == (50, fit.y.size, 4)
    whole = np.einsum("dtp,tp->dt", draws.states, fit.design)
    np.testing.assert_allclose(parts.sum(axis=2), whole, rtol=1e-12, atol=1e-12)
    states = draws.states
    np.testing.assert_allclose(parts[..., 0], states[..., 0])
    np.testing.assert_allclose(parts[..., 1], x[:, 0] * states[..., 2])
    np.testing.assert_allclose(parts[..., 2], x[:, 1] * states[..., 3])
    np.testing.assert_allclose(parts[..., 3], states[..., 4] + states[..., 6])


def test_the_decomposition_s_moments_are_the_smoother_s():
    """Each part is linear in the state, so its mean and variance are the smoothed ones'."""
    fit, _ = _parts_fit()
    sm = smooth(fit)
    n = 20_000
    parts = decompose(fit, backward_sample(fit, n, seed=4))
    coordinates = [[0, 1], [2], [3], [4, 5, 6, 7]]
    for i, coords in enumerate(coordinates):
        loading = np.zeros_like(fit.design)
        loading[:, coords] = fit.design[:, coords]
        mean = np.einsum("tp,tp->t", loading, sm.mean)
        var = np.einsum("tp,tpq,tq->t", loading, sm.covariance, loading)
        z = (parts[..., i].mean(axis=0) - mean) / np.sqrt(var / n)
        assert np.max(np.abs(z)) < 4.5
        np.testing.assert_allclose(parts[..., i].var(axis=0), var, rtol=0.06)


def test_a_decomposition_refuses_another_fit_s_draws():
    fit, _ = _parts_fit()
    with pytest.raises(ValueError, match=r"draws of states shaped \(25, 2\)"):
        decompose(fit, backward_sample(_small_fit(6.0), 5, seed=1))


def test_the_variance_factor_is_the_closed_form_at_the_last_step():
    fit = _small_fit(5.0, beta=0.9)
    from chc.dlm import _variance_factor

    factor, dof = _variance_factor(fit)
    n, s = fit.dof[-1], fit.scale[-1]
    assert factor[-1] == pytest.approx(n * s / (n - 2.0), rel=1e-9)
    assert dof[-1] == pytest.approx(n, rel=1e-12)
    # data in other units: the factor scales with the variance, and nothing else moves
    x, y = _world(8, horizon=25)
    model = fit.model
    scaled = DynamicLinearModel(
        model.blocks,
        Prior(model.prior.mean * 1e5, model.prior.covariance * 1e10, 0.4e10, 5.0),
        variance_discount=model.variance_discount,
    )
    big, big_dof = _variance_factor(forward_filter(scaled, 1e5 * y, x[:, 1]))
    np.testing.assert_allclose(big, 1e10 * factor, rtol=1e-8)
    np.testing.assert_allclose(big_dof, dof, rtol=1e-10)


def test_monitor_evalues_are_density_ratios_with_mean_one():
    x, y = _world(11)
    model = DynamicLinearModel((Regression(2, 0.95),), _prior(2, dof=6.0))
    fit = forward_filter(model, y, x)
    e = monitor_evalues(fit, shifts=(2.0,), inflations=(2.5,))
    u = fit.errors / np.sqrt(fit.one_step_scale)
    nu = fit.one_step_dof
    base = stats.t.pdf(u, nu)
    np.testing.assert_allclose(e[:, 0], stats.t.pdf(u - 2.0, nu) / base, rtol=1e-10)
    np.testing.assert_allclose(e[:, 1], stats.t.pdf(u + 2.0, nu) / base, rtol=1e-10)
    np.testing.assert_allclose(e[:, 2], stats.t.pdf(u / 2.5, nu) / 2.5 / base, rtol=1e-10)


def _steady_level(
    seed: int, horizon: int, delta: float
) -> tuple[DynamicLinearModel, np.ndarray, np.ndarray]:
    """A local level with ``W / V = (1 - delta)^2 / delta``, ``V = 1``, from the discount filter's
    steady state, so the filter is the truth's Kalman filter and its standardised errors are iid;
    the model, the level and the series."""
    rng = np.random.default_rng(seed)
    start = math.sqrt(1.0 - delta) * rng.standard_normal()
    level = start + np.cumsum((1.0 - delta) / math.sqrt(delta) * rng.standard_normal(horizon))
    prior = Prior(np.zeros(1), np.array([[1.0 - delta]]), 1.0, math.inf)
    model = DynamicLinearModel((Polynomial(1, delta),), prior)
    return model, level, level + rng.standard_normal(horizon)


def test_the_monitor_s_e_values_average_one_on_the_model_s_own_forecasts():
    """The shift's e-value has variance ``exp(h^2) - 1``, the inflation's
    ``1 / (k^2 sqrt(2 / k^2 - 1)) - 1``, finite for ``k < sqrt 2``. At ``k = 1.25`` a one-step scale
    10% too large moves its mean to 0.975, eight of its standard errors at this length."""
    model, _, y = _steady_level(17, 20_000, 0.9)
    evalues = monitor_evalues(forward_filter(model, y), shifts=(2.0,), inflations=(1.25,))
    k = 1.25
    variance = np.array(
        [math.exp(4.0) - 1.0] * 2 + [1.0 / (k**2 * math.sqrt(2.0 / k**2 - 1.0)) - 1.0]
    )
    error = np.sqrt(variance / y.size)
    assert np.all(np.abs(evalues.mean(axis=0) - 1.0) < 4.0 * error)


def test_an_alarm_on_a_break_is_the_step_to_intervene_at():
    """West and Harrison's feed-back: the step the alarm sounds at is filtered again with its prior
    discounted, and the level catches the break there, not at the filter's gain a step."""
    model, level, y = _steady_level(19, 260, 0.9)
    shift = 4.0 / math.sqrt(0.9)  # four of the steady one-step forecast's deviations
    level[200:] += shift
    y[200:] += shift
    alarm = DriftAlarm(1_000.0)
    rows = monitor_evalues(forward_filter(model, y))
    first = next((step for step, row in enumerate(rows) if alarm.update(row[None, :])), None)
    assert first is not None
    assert 200 <= first <= 205
    after = slice(first, first + 10)
    plain = forward_filter(model, y).mean[after, 0]
    helped = forward_filter(model, y, interventions={first: 0.01}).mean[after, 0]
    assert np.abs(helped - level[after]).mean() < 0.3 * np.abs(plain - level[after]).mean()


def test_an_intervention_lets_the_level_catch_a_break():
    y = np.concatenate([np.zeros(40), 5.0 * np.ones(10)]) + 0.1 * np.random.default_rng(
        13
    ).standard_normal(50)
    model = DynamicLinearModel((Polynomial(1, 0.99),), _prior(1, scale=0.01))
    plain = forward_filter(model, y)
    helped = forward_filter(model, y, interventions={40: 0.01})
    # one step after the break the discounted prior has absorbed most of it; the plain filter
    # has moved by about its one-step gain, 1 - 0.99
    assert abs(helped.mean[41, 0] - 5.0) < 0.25 * abs(plain.mean[41, 0] - 5.0)


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: Polynomial(0, 0.9), "order must be a positive integer"),
        (lambda: Polynomial(1, 0.0), "discount must be in"),
        (lambda: Polynomial(1, 1.5), "discount must be in"),
        (lambda: Seasonal(1.0, (1,), 0.9), "period must be finite and above 1"),
        (lambda: Seasonal(12, (7,), 0.9), "is not an integer in"),
        (lambda: Seasonal(12, (1, 1), 0.9), "harmonics repeat"),
        (lambda: Regression(0, 0.9), "width must be a positive integer"),
        (lambda: Prior(np.zeros(2), np.eye(3), 1.0, 1.0), "covariance must be a finite"),
        (lambda: Prior(np.zeros(1), -np.eye(1), 1.0, 1.0), "not positive definite"),
        (lambda: Prior(np.zeros(2), np.diag([1.0, 0.0]), 1.0, 1.0), "not positive definite"),
        (lambda: Prior(np.zeros(1), np.eye(1), 0.0, 1.0), "scale must be positive"),
        (lambda: Prior(np.zeros(1), np.eye(1), 1.0, 0.0), "dof must be positive"),
        (lambda: DynamicLinearModel((), _prior(1)), "at least one block"),
        (lambda: DynamicLinearModel((Polynomial(1, 0.9),), _prior(2)), "the prior has 2"),
        (
            lambda: DynamicLinearModel((Polynomial(1, 0.9),), _prior(1), variance_discount=0.9),
            "needs a learned variance",
        ),
    ],
)
def test_invalid_specifications_raise(build, match):
    with pytest.raises(ValueError, match=match):
        build()


def test_invalid_data_raise():
    model = DynamicLinearModel((Regression(2, 0.9),), _prior(2))
    with pytest.raises(ValueError, match="x must have shape"):
        forward_filter(model, np.zeros(5), np.zeros((5, 3)))
    with pytest.raises(ValueError, match="x is not finite"):
        forward_filter(model, np.zeros(2), np.array([[1.0, np.nan], [1.0, 1.0]]))
    with pytest.raises(ValueError, match="need x"):
        forward_filter(model, np.zeros(5))
    with pytest.raises(ValueError, match="infinite"):
        forward_filter(model, np.array([np.inf, 0.0]), np.ones((2, 2)))
    with pytest.raises(ValueError, match="outside"):
        forward_filter(model, np.zeros(2), np.ones((2, 2)), interventions={5: 0.5})
