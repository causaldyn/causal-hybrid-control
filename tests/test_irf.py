"""chc.irf: local projections recover the analytic impulse response; dropping z biases it."""

import numpy as np

from chc.irf import innovations, irf_control_sequence, local_projection_irf, structured_irf
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
