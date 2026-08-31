"""chc.irf: local projections recover the analytic impulse response; dropping z biases it."""

import numpy as np
import pytest

from chc.irf import (
    _irf_from_rows,
    _projection_design,
    delay_estimate,
    innovations,
    irf_control_sequence,
    local_projection_irf,
    peak_lag,
    structured_irf,
)
from chc.toeplitz import levinson_durbin, sample_autocorrelation

# x_{t+1} = a x_t + b u_t + c z_t + noise, with the policy u_t = kappa z_t + eta (confounded by z).
_A, _B, _C, _KAPPA = 0.6, 1.0, 1.5, -1.2


def _confounded_arx(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)  # i.i.d. confounder -> u is serially uncorrelated (clean IRF)
    u = _KAPPA * z + 0.5 * rng.standard_normal(n)  # policy tied to the confounder
    x = np.zeros(n)
    noise = 0.1 * rng.standard_normal(n)
    for t in range(1, n):
        x[t] = _A * x[t - 1] + _B * u[t - 1] + _C * z[t - 1] + noise[t]
    return {"x": x, "u": u, "z": z}


def _analytic_irf(horizon: int) -> np.ndarray:
    # effect of u_t on x_{t+h}: 0 at h=0, then b * a^{h-1} as it propagates through the AR(1) state
    return np.array([0.0] + [_B * _A ** (h - 1) for h in range(1, horizon + 1)])


def test_local_projection_recovers_the_analytic_impulse_response() -> None:
    horizon = 6
    data = _confounded_arx(8000, seed=0)
    irf = np.asarray(local_projection_irf(data, horizon, adjust_for=("x", "z")))
    assert np.max(np.abs(irf - _analytic_irf(horizon))) < 0.1  # adjusted IRF matches the truth


def test_omitting_the_confounder_biases_the_impulse_response() -> None:
    horizon = 6
    data = _confounded_arx(8000, seed=0)
    naive = np.asarray(local_projection_irf(data, horizon, adjust_for=("x",)))  # z omitted
    adjusted = np.asarray(local_projection_irf(data, horizon, adjust_for=("x", "z")))
    assert abs(naive[1] - _B) > 0.3  # the one-step effect is badly confounded without z
    assert abs(adjusted[1] - _B) < 0.1  # ...and recovered with it


def test_structured_irf_agrees_with_local_projections() -> None:
    horizon = 6
    data = _confounded_arx(8000, seed=0)
    local = np.asarray(local_projection_irf(data, horizon, adjust_for=("x", "z")))
    structured = structured_irf(data, horizon, order=4, adjust_for=("x", "z"))
    assert np.max(np.abs(structured - local)) < 0.1  # the two routes to the IRF agree
    assert np.max(np.abs(structured - _analytic_irf(horizon))) < 0.1  # ...and match the truth


def test_innovations_whiten_the_series() -> None:
    x = _confounded_arx(8000, seed=1)["x"]

    def lag1(series: np.ndarray) -> float:
        auto = sample_autocorrelation(series, 1)
        return abs(auto[1] / auto[0])

    assert lag1(innovations(x, order=4)) < 0.3 * lag1(x)  # prewhitening removes the autocorrelation


def test_biased_autocorrelation_keeps_levinson_stable_on_few_samples() -> None:
    rng = np.random.default_rng(3)
    n = 40  # short, strongly autocorrelated series where an unbiased estimate could go non-PD
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + rng.standard_normal()
    _ar, reflection, error = levinson_durbin(sample_autocorrelation(x, 8))
    assert np.all(np.abs(reflection) < 1.0)  # biased (PSD) autocorrelation -> stable Levinson
    assert error > 0.0  # positive innovation power


def _distributed_lag_state(kernel: np.ndarray, u: np.ndarray) -> np.ndarray:
    """State of a delayed plant: x_t = sum_k kernel[k] u_{t-1-k} (the effect starts at h=1)."""
    n = u.shape[0]
    x = np.zeros(n)
    for t in range(n):
        x[t] = sum(kernel[k] * u[t - 1 - k] for k in range(kernel.shape[0]) if t - 1 - k >= 0)
    return x


def _tracking_error(kernel: np.ndarray, control: np.ndarray, target: np.ndarray) -> float:
    padded = np.concatenate([control, np.zeros(kernel.shape[0])])  # let the carryover finish
    achieved = _distributed_lag_state(kernel, padded)[1 : target.shape[0] + 1]  # x_{t+1}
    return float(np.max(np.abs(achieved - target)))


def test_irf_control_tracks_where_one_step_overshoots_a_delayed_plant() -> None:
    kernel = np.array([1.0, 0.6, 0.3, 0.1])  # steady-state gain 2.0, spread over 4 lags
    irf = np.concatenate([[0.0], kernel])
    target = np.ones(30)
    err_irf = _tracking_error(kernel, irf_control_sequence(irf, target), target)
    err_one_step = _tracking_error(kernel, target / irf[1], target)  # inverts only g_1
    assert err_irf < 0.05  # deconvolving the whole response tracks
    assert err_one_step > 0.5  # ...while ignoring carryover overshoots (to sum g_h / g_1)


def test_estimated_irf_control_beats_one_step_end_to_end() -> None:
    kernel = np.array([1.0, 0.6, 0.3, 0.1])
    rng = np.random.default_rng(0)
    u = rng.standard_normal(6000)  # randomised exploration -> the IRF is identified, no confounding
    x = _distributed_lag_state(kernel, u) + 0.02 * rng.standard_normal(6000)
    irf = np.asarray(local_projection_irf({"x": x, "u": u}, horizon=6, adjust_for=()))
    target = np.ones(30)
    err_irf = _tracking_error(kernel, irf_control_sequence(irf, target), target)
    err_one_step = _tracking_error(kernel, target / irf[1], target)
    assert err_irf < 0.2 * err_one_step  # the estimated IRF still tracks far better than one-step


# x_t = phi x_{t-1} + theta[(1-f) u_{t-L} + f u_{t-L-1}] + noise: the effect arrives L steps late.
_PHI, _THETA, _LAG = 0.9, 1.0, 5


def _delayed_arx(
    n: int, seed: int, phi: float = _PHI, sigma: float = 0.3, frac: float = 0.0, effect: float = 1.0
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(n)
    x = np.zeros(n)
    noise = sigma * rng.standard_normal(n)
    for t in range(1, n):
        drive = effect * _THETA * (1.0 - frac) * u[t - _LAG] if t >= _LAG else 0.0
        if t >= _LAG + 1:
            drive += effect * _THETA * frac * u[t - _LAG - 1]
        x[t] = phi * x[t - 1] + drive + noise[t]
    return {"x": x, "u": u}


def test_lag_augmentation_leaves_the_estimand_alone() -> None:
    data = _delayed_arx(4000, seed=0)
    plain = np.asarray(local_projection_irf(data, 10, lags=0))
    augmented = np.asarray(local_projection_irf(data, 10, lags=4))
    assert np.max(np.abs(plain - augmented)) < 0.05  # own-lags buy inference, not a new estimand


def test_the_bootstrap_refits_the_estimator_it_reports() -> None:
    """The block bootstrap must resample the same regression, or its width prices nothing."""
    data = _delayed_arx(1200, seed=3)
    features, responses = _projection_design(data, 10, "u", "x", ("x",), 4)
    rows = np.asarray(features, dtype=np.float64)
    design = np.concatenate([rows, np.ones((rows.shape[0], 1))], axis=1)
    direct = _irf_from_rows(design, np.asarray(responses, dtype=np.float64))
    assert np.max(np.abs(direct - np.asarray(local_projection_irf(data, 10, lags=4)))) < 1e-6


def test_the_delay_estimate_recovers_a_known_lag() -> None:
    est = delay_estimate(_delayed_arx(800, seed=0), horizon=12, dt=0.5)
    assert est.lag == _LAG
    assert est.delay == _LAG * 0.5  # dt carries the estimate into the caller's time unit
    assert est.lo <= _LAG * 0.5 <= est.hi
    assert not est.censored
    assert est.peak_response > 0.0


def test_the_interval_covers_the_true_lag_across_seeds() -> None:
    """A nominal 95% interval that covered 60% of the time would be worse than no interval."""
    intervals = [
        delay_estimate(_delayed_arx(800, seed=s, sigma=2.0), horizon=12, n_resamples=100, seed=s)
        for s in range(20)
    ]
    covered = sum(est.lo <= _LAG <= est.hi for est in intervals)
    assert covered >= 17  # nominal 0.95; measured 1.00 over 40 replications


def test_a_flat_irf_widens_the_interval_instead_of_inventing_a_mode() -> None:
    horizon = 12
    est = delay_estimate(_delayed_arx(800, seed=0, sigma=1.0, effect=0.0), horizon=horizon)
    assert est.hi - est.lo > 0.5 * horizon  # no effect -> the peak wanders over the whole horizon


def test_a_sign_flipping_effect_is_located_on_magnitude_and_reported_signed() -> None:
    data = _delayed_arx(800, seed=1)
    data["x"] = -data["x"]  # same arrival time, opposite sign
    est = delay_estimate(data, horizon=12)
    assert est.lag == _LAG
    assert est.peak_response < 0.0  # located by |beta|, reported with its sign


def test_a_response_still_rising_at_the_edge_is_censored_not_reported_as_the_horizon() -> None:
    est = delay_estimate(_delayed_arx(800, seed=0), horizon=_LAG)  # the peak IS the last horizon
    assert est.censored
    assert est.lag == float(_LAG)


def test_an_effect_that_never_arrives_is_unidentified_rather_than_censored() -> None:
    """censored means "still rising at the edge"; a delay wholly past the horizon leaves noise."""
    horizon = 3  # true lag 5: nothing has arrived, so the argmax lands wherever noise puts it
    est = delay_estimate(_delayed_arx(800, seed=0), horizon=horizon)
    assert not est.censored
    assert est.hi - est.lo >= 0.5 * horizon  # the width, not the flag, is what reports this


def test_the_parabolic_refinement_is_off_because_a_causal_peak_is_one_sided() -> None:
    """Geometric decay after arrival puts the vertex at lag + phi/(2(2-phi)) -- 0.41 at phi=0.9."""
    irf = np.array([0.0] * _LAG + [_PHI**h for h in range(8)])
    assert peak_lag(irf) == (float(_LAG), False)
    refined, _ = peak_lag(irf, refine=True)
    assert abs(refined - (_LAG + _PHI / (2.0 * (2.0 - _PHI)))) < 1e-12


def test_the_refinement_finds_a_peak_that_falls_between_two_samples() -> None:
    """A delay that is not a multiple of dt smears across bins; the argmax can only quantise it."""
    data = _delayed_arx(800, seed=0, phi=0.0, frac=0.5)  # no AR, so bins L and L+1 tie exactly
    refined = delay_estimate(data, horizon=12, refine=True)
    quantised = delay_estimate(data, horizon=12, refine=False)
    assert abs(refined.lag - (_LAG + 0.5)) < 0.1
    assert quantised.lag in (float(_LAG), float(_LAG + 1))


def test_the_refinement_shrinks_a_fractional_lag_it_does_not_split_evenly() -> None:
    """vertex(0, 1-f, f) = f/(4-6f): exact at f=1/2, shrunk elsewhere (validation/...bias.mac)."""
    for frac in (0.5, 1.0 / 3.0, 0.25):
        weights = np.array([0.0, 1.0 - frac, frac, 0.0])
        assert abs(peak_lag(weights, refine=True)[0] - (1.0 + frac / (4.0 - 6.0 * frac))) < 1e-12


def test_the_refinement_reads_a_plateau_from_its_leading_edge() -> None:
    """Three points cannot see a wider plateau -- the documented limit of a local parabola."""
    assert peak_lag(np.array([0.0, 1.0, 1.0, 0.0]), refine=True) == (1.5, False)  # tie -> midpoint
    assert peak_lag(np.array([0.0, 1.0, 1.0, 1.0, 0.0]), refine=True) == (1.5, False)


def test_a_series_too_short_for_an_honest_interval_is_rejected() -> None:
    short = {name: column[:60] for name, column in _delayed_arx(800, seed=0).items()}
    with pytest.raises(ValueError, match="block bootstrap"):
        delay_estimate(short, horizon=12)
    with pytest.raises(ValueError, match="too short for horizon"):
        local_projection_irf(short, horizon=100)
    with pytest.raises(ValueError, match="non-negative"):
        local_projection_irf(_delayed_arx(200, seed=0), horizon=5, lags=-1)
    with pytest.raises(ValueError, match="level must lie"):
        delay_estimate(_delayed_arx(800, seed=0), horizon=12, level=1.0)
    with pytest.raises(ValueError, match="at least 2"):
        delay_estimate(_delayed_arx(800, seed=0), horizon=12, n_resamples=1)
    with pytest.raises(ValueError, match="irf is empty"):
        peak_lag(np.array([]))
