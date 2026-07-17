"""chc.independence: partial correlation removes a confounded path and stays calibrated on AR."""

import numpy as np

from chc.independence import partial_corr_test


def _ar1(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    return x


def test_partial_corr_test_removes_a_confounded_path() -> None:
    rng = np.random.default_rng(0)
    n = 500
    z = rng.standard_normal(n)
    x = z + 0.3 * rng.standard_normal(n)
    y = z + 0.3 * rng.standard_normal(n)  # x and y are linked ONLY through z
    _, p_marginal = partial_corr_test(x, y)  # marginally dependent (via z)
    _, p_conditional = partial_corr_test(x, y, z)  # independent given z
    assert float(p_marginal) < 0.01
    assert float(p_conditional) > 0.05


def test_partial_corr_test_is_calibrated_under_autocorrelation() -> None:
    phi, n, trials, alpha = 0.8, 250, 200, 0.05
    naive_rejections = mci_rejections = 0
    for trial in range(trials):
        rng = np.random.default_rng(1000 + trial)
        x = _ar1(rng, n, phi)
        y = _ar1(rng, n, phi)  # two INDEPENDENT AR(1) series: there is no true x-y link
        _, p_naive = partial_corr_test(x[1:], y[1:])
        conditioning = np.column_stack([x[:-1], y[:-1]])  # condition on the lagged parents (MCI)
        _, p_mci = partial_corr_test(x[1:], y[1:], conditioning)
        naive_rejections += float(p_naive) < alpha
        mci_rejections += float(p_mci) < alpha
    naive_rate, mci_rate = naive_rejections / trials, mci_rejections / trials
    assert naive_rate > 0.15  # the naive test over-rejects unrelated autocorrelated series
    assert mci_rate < 0.10  # conditioning on the lagged parents restores ~nominal calibration
