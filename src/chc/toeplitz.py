"""Structured-operator primitives for LTI dynamics: Toeplitz matvec, Levinson-Durbin, deconvolution.

A Toeplitz operator *is* a linear time-invariant system -- its action is convolution with the
impulse response. These primitives back the structured route to the dynamic causal effect
(``chc.irf``): the fast forward operator, the Yule-Walker AR fit (reflection coefficients and
innovation power), deconvolution (recover the excitation from the response), and the
Gohberg-Semencul fast inverse. See ``plans/18``.

Two ways to apply ``T^{-1}``: ``solve_toeplitz`` is a one-shot dense solve; **Gohberg-Semencul**
(:func:`gohberg_semencul_generators` + :func:`gohberg_semencul_apply`) compresses the whole inverse
into two generator vectors (the first and last columns of ``T^{-1}``), then applies ``T^{-1} v`` in
``O(L log L)`` per vector by FFT. The win is **amortised repeated application** (deconvolve many
signals through one operator) plus the compression itself -- real at any scale, decisive at the
``L ~ 1e3-1e4`` long sequences the idea targets. NumPy float64 (a numerical routine, off jit paths).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import ArrayLike
from scipy.linalg import toeplitz as _dense_toeplitz


def toeplitz_matvec(first_col: Array, first_row: Array, x: Array) -> Array:
    """Toeplitz matrix-vector product ``T x`` in ``O(L log L)`` via circulant embedding + FFT.

    ``T`` has first column ``first_col`` and first row ``first_row`` (shared corner entry). The
    ``2L``-circulant with first column ``[first_col, 0, reverse(first_row[1:])]`` contains ``T`` in
    its leading block, so one FFT convolution recovers ``T x``. Differentiable (JAX).
    """
    first_col, first_row, x = jnp.asarray(first_col), jnp.asarray(first_row), jnp.asarray(x)
    n = first_col.shape[0]
    embedding = jnp.concatenate([first_col, jnp.zeros(1), first_row[1:][::-1]])  # circulant column
    padded = jnp.concatenate([x, jnp.zeros(n)])
    convolved = jnp.fft.ifft(jnp.fft.fft(embedding) * jnp.fft.fft(padded))
    return jnp.real(convolved[:n])


def sample_autocorrelation(x: ArrayLike, max_lag: int) -> np.ndarray:
    """Biased sample autocorrelation ``r[0..max_lag]`` -- always PSD, so Levinson stays safe.

    Divides every lag by ``n`` (not ``n - k``): the biased estimator is guaranteed PSD, so its
    Toeplitz matrix is valid and Levinson-Durbin stays stable even from few samples, where the
    unbiased estimator can be non-PD with reflection coefficients outside ``(-1, 1)``. The robust
    distillation of Gohberg-Semencul covariance estimation (arXiv:2311.14995); see ``plans/18``.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.shape[0]
    return np.array([float(np.dot(x[: n - k], x[k:])) / n for k in range(max_lag + 1)])


def levinson_durbin(autocorrelation: ArrayLike) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve the Yule-Walker system for an AR(p) model from its autocorrelation ``r[0..p]``.

    Returns ``(ar_coeffs, reflection_coeffs, prediction_error)``: the AR coefficients ``a`` with
    ``x_t ~ sum_i a_i x_{t-i}``, the reflection (PARCOR) coefficients, and the prediction-error
    variance -- the **innovation power** left after the AR structure is removed. ``O(p^2)``.
    """
    r = np.asarray(autocorrelation, dtype=np.float64)
    p = r.shape[0] - 1
    a = np.zeros(p)
    reflection = np.zeros(p)
    error = float(r[0])
    for m in range(p):
        acc = r[m + 1] - np.dot(a[:m], r[m:0:-1])  # r[m+1] - sum_{i=1}^{m} a_i r[m+1-i]
        k = acc / error
        a[:m] = a[:m] - k * a[:m][::-1]  # a_i <- a_i - k a_{m-i} (RHS reads the old a)
        a[m] = k
        reflection[m] = k
        error *= 1.0 - k * k
    return a, reflection, error


def solve_toeplitz(first_col: ArrayLike, first_row: ArrayLike, rhs: ArrayLike) -> np.ndarray:
    """Deconvolution: solve ``T u = rhs`` -- recover an LTI system's excitation from its response.

    One-shot dense solve. For many right-hand sides through one operator, use Gohberg-Semencul
    (compute generators once, then apply ``T^{-1}`` in ``O(L log L)`` per vector).
    """
    matrix = _dense_toeplitz(np.asarray(first_col, dtype=np.float64), np.asarray(first_row))
    return np.linalg.solve(matrix, np.asarray(rhs, dtype=np.float64))


def _toeplitz_matvec_np(first_col: np.ndarray, first_row: np.ndarray, v: np.ndarray) -> np.ndarray:
    """NumPy float64 Toeplitz matvec (circulant embedding + FFT) for the numerical inverse."""
    n = first_col.shape[0]
    embedding = np.concatenate([first_col, [0.0], first_row[1:][::-1]])
    padded = np.concatenate([v, np.zeros(n)])
    return np.real(np.fft.ifft(np.fft.fft(embedding) * np.fft.fft(padded)))[:n]


def gohberg_semencul_generators(
    first_col: ArrayLike, first_row: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """The two Gohberg-Semencul generators: the first and last columns of ``T^{-1}``.

    Solve ``T x = e_1`` and ``T y = e_n``; these vectors encode the *whole* inverse (displacement
    rank 2). Pass them to :func:`gohberg_semencul_apply`. Requires ``x[0] != 0`` (generic for a
    nonsingular Toeplitz). Setup is a dense solve; the point is that *application* is then fast.
    """
    first_col = np.asarray(first_col, dtype=np.float64)
    first_row = np.asarray(first_row, dtype=np.float64)
    n = first_col.shape[0]
    matrix = _dense_toeplitz(first_col, first_row)
    unit_first, unit_last = np.zeros(n), np.zeros(n)
    unit_first[0], unit_last[-1] = 1.0, 1.0
    x = np.linalg.solve(matrix, unit_first)  # T^{-1} e_1: first column of the inverse
    y = np.linalg.solve(matrix, unit_last)  # T^{-1} e_n: last column of the inverse
    if abs(x[0]) < 1e-12:
        raise ValueError("Gohberg-Semencul needs (T^-1)[0,0] != 0; this Toeplitz is degenerate")
    return x, y


def gohberg_semencul_covariance(
    snapshots: ArrayLike, order: int, size: int | None = None
) -> np.ndarray:
    """Few-sample Toeplitz covariance estimate via AR / Gohberg-Semencul (arXiv:2311.14995).

    Averages the biased autocovariance over ``snapshots`` (rows), fits an AR(``order``) by Levinson
    (the paper's closed-form projected-least-squares estimator ``a = R_w^{-1} r``), and extends it
    to the maximum-entropy Toeplitz covariance. The estimate is **positive definite and full rank
    even when the sample covariance is singular** (``N < size``). Its inverse is the GS precision
    with generator ``alpha = (1/sigma^2) [1, -a]`` (the AR whitening filter, Eq 40), recoverable via
    :func:`gohberg_semencul_generators`. Distils the paper's PLS estimator.

    It is a **regularised** estimator: low variance (a win at small ``N``) bought with model bias.
    Use it for few samples of a *stationary, roughly low-order-AR* process; on a non-Toeplitz /
    non-stationary covariance it plateaus at a bias floor and the raw sample covariance wins once
    ``N`` is large. Empirically ~30x lower NMSE than the SCM at ``N << size`` on matched AR data,
    fading to a tie as ``N`` grows.
    """
    rows = np.atleast_2d(np.asarray(snapshots, dtype=np.float64))
    size = rows.shape[1] if size is None else size
    autocov = np.mean([sample_autocorrelation(row, order) for row in rows], axis=0)
    ar, _reflection, _error = levinson_durbin(autocov)
    extended = np.zeros(size)
    extended[: min(order + 1, size)] = autocov[: min(order + 1, size)]
    for lag in range(order + 1, size):
        extended[lag] = sum(ar[i - 1] * extended[lag - i] for i in range(1, order + 1))
    return _dense_toeplitz(extended)


def gohberg_semencul_apply(x: ArrayLike, y: ArrayLike, v: ArrayLike) -> np.ndarray:
    """Apply ``T^{-1} v`` in ``O(L log L)`` from the generators ``x, y`` (Gohberg-Semencul).

    ``T^{-1} = (1/x_0) [ L(x) U(J y) - L(S y) U(S J x) ]`` with ``L``/``U`` lower/upper-triangular
    Toeplitz, ``J`` reverse and ``S`` down-shift; each factor is applied by FFT. Amortises over many
    right-hand sides that share the operator.
    """
    x, y, v = np.asarray(x, np.float64), np.asarray(y, np.float64), np.asarray(v, np.float64)

    def lower(col: np.ndarray, w: np.ndarray) -> np.ndarray:  # L(col) @ w
        return _toeplitz_matvec_np(col, np.concatenate([col[:1], np.zeros(col.shape[0] - 1)]), w)

    def upper(row: np.ndarray, w: np.ndarray) -> np.ndarray:  # U(row) @ w
        return _toeplitz_matvec_np(np.concatenate([row[:1], np.zeros(row.shape[0] - 1)]), row, w)

    shifted_y = np.concatenate([[0.0], y[:-1]])
    shifted_rev_x = np.concatenate([[0.0], x[::-1][:-1]])
    term_1 = lower(x, upper(y[::-1], v))
    term_2 = lower(shifted_y, upper(shifted_rev_x, v))
    return (term_1 - term_2) / x[0]
