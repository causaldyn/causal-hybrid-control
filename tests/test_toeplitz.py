"""chc.toeplitz: FFT matvec matches dense, Levinson matches Yule-Walker, deconvolution round-trips.

Structured-operator primitives for the dynamic-effect route (plans/18).
"""

import numpy as np
from scipy.linalg import toeplitz

from chc.toeplitz import (
    gohberg_semencul_apply,
    gohberg_semencul_covariance,
    gohberg_semencul_generators,
    levinson_durbin,
    sample_autocorrelation,
    solve_toeplitz,
    toeplitz_matvec,
)


def test_toeplitz_matvec_matches_the_dense_product() -> None:
    rng = np.random.default_rng(0)
    n = 32
    first_col = rng.standard_normal(n)
    first_row = np.concatenate([first_col[:1], rng.standard_normal(n - 1)])  # shared corner
    x = rng.standard_normal(n)
    dense = toeplitz(first_col, first_row) @ x
    fast = np.asarray(toeplitz_matvec(first_col, first_row, x))
    assert np.max(np.abs(fast - dense)) < 1e-8


def test_levinson_durbin_matches_the_yule_walker_solve() -> None:
    # autocorrelation of a stable AR(2); Levinson must reproduce the direct Yule-Walker solution
    r = np.array([1.0, 0.6, 0.1, -0.1, -0.05, 0.02, 0.0])
    p = r.shape[0] - 1
    ar, reflection, error = levinson_durbin(r)
    direct = np.linalg.solve(toeplitz(r[:p]), r[1 : p + 1])
    assert np.max(np.abs(ar - direct)) < 1e-9  # AR coefficients agree
    assert np.all(np.abs(reflection) < 1.0)  # a valid PSD autocorrelation -> reflections in (-1, 1)
    assert error > 0.0  # positive innovation power


def test_solve_toeplitz_deconvolves_a_response() -> None:
    rng = np.random.default_rng(1)
    n = 24
    first_col = np.concatenate([[3.0], 0.3 * rng.standard_normal(n - 1)])  # diagonally dominant
    first_row = np.concatenate([first_col[:1], 0.3 * rng.standard_normal(n - 1)])
    u_true = rng.standard_normal(n)
    response = np.asarray(toeplitz_matvec(first_col, first_row, u_true))  # forward convolution
    u_recovered = solve_toeplitz(first_col, first_row, response)  # deconvolve
    assert np.max(np.abs(u_recovered - u_true)) < 1e-8


def test_gohberg_semencul_applies_the_inverse_from_two_generators() -> None:
    rng = np.random.default_rng(2)
    n = 20
    first_col = np.concatenate([[4.0], 0.4 * rng.standard_normal(n - 1)])  # diagonally dominant
    first_row = np.concatenate([first_col[:1], 0.4 * rng.standard_normal(n - 1)])
    dense = toeplitz(first_col, first_row)
    x, y = gohberg_semencul_generators(first_col, first_row)
    assert np.allclose(x, np.linalg.solve(dense, np.eye(n)[0]))  # generator = first inverse column
    # one generator pair, applied fast to many right-hand sides, matches the dense solve each time
    for seed in range(3):
        v = np.random.default_rng(10 + seed).standard_normal(n)
        assert np.max(np.abs(gohberg_semencul_apply(x, y, v) - np.linalg.solve(dense, v))) < 1e-8


def _ar_covariance(ar: list[float], sigma: float, size: int) -> np.ndarray:
    """Ground-truth AR covariance from a long simulation (a valid PD Toeplitz matrix)."""
    rng = np.random.default_rng(0)
    n = 200_000
    x = np.zeros(n)
    for t in range(len(ar), n):
        x[t] = sum(ar[i] * x[t - 1 - i] for i in range(len(ar))) + sigma * rng.standard_normal()
    return toeplitz(sample_autocorrelation(x, size - 1))


def test_gohberg_semencul_covariance_beats_the_sample_covariance_from_few_snapshots() -> None:
    size, order, n_snapshots = 20, 3, 6  # N < size, so the sample covariance is singular
    true_cov = _ar_covariance([0.5, -0.3], sigma=1.0, size=size)
    chol = np.linalg.cholesky(true_cov + 1e-9 * np.eye(size))

    def nmse(estimate: np.ndarray) -> float:
        return float(np.linalg.norm(estimate - true_cov) ** 2 / np.linalg.norm(true_cov) ** 2)

    gs_errors, scm_errors = [], []
    for trial in range(30):
        snapshots = np.random.default_rng(100 + trial).standard_normal((n_snapshots, size)) @ chol.T
        estimate = gohberg_semencul_covariance(snapshots, order)
        gs_errors.append(nmse(estimate))
        scm_errors.append(nmse(snapshots.T @ snapshots / n_snapshots))
        assert np.all(np.linalg.eigvalsh(estimate) > 0)  # PD though the sample covariance isn't
    assert np.mean(gs_errors) < 0.3 * np.mean(scm_errors)  # the structured estimate crushes the SCM
