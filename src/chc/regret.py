"""Certainty-equivalence suboptimality for LQ control -- CHC's regret guarantee (LQ special case).

For a linear-quadratic problem the certainty-equivalent controller (solve the LQR for an *estimated*
model, then apply that gain to the true plant) has a suboptimality gap that is **quadratic** in the
model error: ``J(K_hat) - J* = O(||[dA, dB]||^2)`` in the small-error regime (Dean-Mania-Tu-Recht-
Matni, 2018/2020). This is the analysable special case of CHC's pessimism story -- small model error
costs almost nothing, but the penalty grows with error, which is exactly what the calibrated
uncertainty penalty (:mod:`chc.uncertainty`) is there to price in offline.

A NumPy/scipy analysis tool (like :mod:`chc.did` / :mod:`chc.scm`), independent of the JAX ``x64``
flag. The infinite-horizon discrete LQR is solved via the DARE; a controller's true-plant cost via
the discrete Lyapunov equation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

Matrix = NDArray[np.float64]
Vector = NDArray[np.float64]

_DEFAULT_ERRORS = (0.04, 0.02, 0.01, 0.005, 0.0025)


@dataclass(frozen=True)
class RegretCurve:
    """Empirical certificate: median CE suboptimality vs model error, and its log-log slope."""

    errors: Vector  # median realised model-error magnitude ||[dA, dB]|| at each swept level
    gaps: Vector  # median certainty-equivalence suboptimality gap at each level
    exponent: float  # fitted log-log slope of gap vs error (~2.0 => quadratic suboptimality)


def dlqr(a: Matrix, b: Matrix, q: Matrix, r: Matrix) -> tuple[Matrix, Matrix]:
    """Infinite-horizon discrete LQR: optimal gain ``K`` and cost-to-go ``P`` (via the DARE)."""
    p = solve_discrete_are(a, b, q, r)
    k = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
    return k, p


def closed_loop_cost(a: Matrix, b: Matrix, k: Matrix, q: Matrix, r: Matrix, x0: Vector) -> float:
    """Infinite-horizon LQ cost of applying gain ``k`` to plant ``(a, b)`` from ``x0``.

    ``x0' P x0`` where ``P`` solves the discrete Lyapunov equation for the closed loop; ``+inf`` if
    ``k`` fails to stabilise ``(a, b)`` (a destabilising controller has unbounded cost).
    """
    a_cl = a - b @ k
    if np.max(np.abs(np.linalg.eigvals(a_cl))) >= 1.0:
        return float("inf")
    p = solve_discrete_lyapunov(a_cl.T, q + k.T @ r @ k)
    return float(x0 @ p @ x0)


def certainty_equivalence_gap(
    a: Matrix, b: Matrix, q: Matrix, r: Matrix, a_hat: Matrix, b_hat: Matrix, x0: Vector
) -> float:
    """Suboptimality ``J(K_hat) - J*`` of the certainty-equivalent controller on the true plant.

    ``K_hat`` is the LQR-optimal gain for the estimated model ``(a_hat, b_hat)``; the gap is its
    true-plant cost minus the cost of the true-optimal gain. Zero iff the estimate induces the
    optimal gain; otherwise non-negative (the optimum is optimal).
    """
    k_hat, _ = dlqr(a_hat, b_hat, q, r)
    k_star, _ = dlqr(a, b, q, r)
    return closed_loop_cost(a, b, k_hat, q, r, x0) - closed_loop_cost(a, b, k_star, q, r, x0)


def regret_scaling(
    a: Matrix,
    b: Matrix,
    q: Matrix,
    r: Matrix,
    x0: Vector,
    *,
    errors: Sequence[float] = _DEFAULT_ERRORS,
    n_samples: int = 400,
    seed: int = 0,
) -> RegretCurve:
    """Certificate that CE suboptimality is quadratic in model error (slope ~2 in the small limit).

    At each target magnitude ``eps`` draws ``n_samples`` Gaussian model perturbations ``(dA, dB)``
    scaled by ``eps``, records the :func:`certainty_equivalence_gap`, and fits the log-log slope of
    gap vs realised error over all samples. Perturbations that make the estimate unstabilisable are
    skipped. Theory (Dean et al.) predicts an exponent of 2.
    """
    rng = np.random.default_rng(seed)
    median_errors, median_gaps = [], []
    log_err, log_gap = [], []
    for eps in errors:
        errs, gaps = [], []
        for _ in range(n_samples):
            d_a, d_b = eps * rng.normal(size=a.shape), eps * rng.normal(size=b.shape)
            try:
                gap = certainty_equivalence_gap(a, b, q, r, a + d_a, b + d_b, x0)
            except (np.linalg.LinAlgError, ValueError):
                continue  # estimate not stabilisable: no certainty-equivalent gain exists
            err = float(np.sqrt(np.sum(d_a**2) + np.sum(d_b**2)))
            if np.isfinite(gap) and gap > 0.0:
                errs.append(err)
                gaps.append(gap)
        if errs:
            median_errors.append(float(np.median(errs)))
            median_gaps.append(float(np.median(gaps)))
            log_err.extend(np.log(errs))
            log_gap.extend(np.log(gaps))
    exponent = float(np.polyfit(log_err, log_gap, 1)[0]) if log_err else float("nan")
    return RegretCurve(np.array(median_errors), np.array(median_gaps), exponent)
