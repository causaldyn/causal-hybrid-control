"""chc.irf: local projections recover the analytic impulse response; dropping z biases it."""

import numpy as np

from chc.irf import innovations, local_projection_irf, structured_irf
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
