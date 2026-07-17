"""Autocorrelation-robust conditional-independence testing (the MCI idea, borrowed from tigramite).

A linear partial-correlation (ParCorr) test of ``x ⊥ y | z``: residualise ``x`` and ``y`` on the
conditioning set ``z``, then Fisher-z test the residual correlation. In serially correlated data a
naive ``corr(x, y)`` is badly miscalibrated -- autocorrelation inflates the effective variance of
the estimator, so unrelated series look linked. Conditioning on the lagged parents (tigramite's
*momentary conditional independence*) whitens the residuals and restores calibration. This is the CI
primitive ``chc.discovery`` screens lagged parents with; see ``plans/17``. The method (partial
correlation) is standard; only the autocorrelation-aware *usage* is borrowed -- no tigramite code.

Computed in NumPy float64, not JAX: a p-value threshold is precision-sensitive near ``alpha``, and a
statistical test must not silently depend on the global ``jax_enable_x64`` flag (float32 would flip
borderline edges in discovery). It is not on any differentiated or jitted path.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

_EPS = 1e-12


def _residualize(target: ArrayLike, conditioning: ArrayLike | None) -> tuple[np.ndarray, int]:
    """Residual of ``target`` after linear regression on ``[1, conditioning]``; returns (resid, k).

    ``k`` is the number of conditioning columns (0 when ``conditioning`` is ``None``) -- the
    degrees-of-freedom correction for the Fisher-z statistic. A ``(n,)`` or ``(n, k)`` conditioning
    set is accepted; a ``(k, n)`` one is transposed to rows-are-samples.
    """
    target = np.asarray(target, dtype=np.float64).ravel()
    if conditioning is None:
        return target - target.mean(), 0
    cond = np.atleast_2d(np.asarray(conditioning, dtype=np.float64))
    if cond.shape[0] != target.shape[0]:
        cond = cond.T
    design = np.column_stack([np.ones(target.shape[0]), cond])
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    return target - design @ coeffs, cond.shape[1]


def partial_corr_test(
    x: ArrayLike, y: ArrayLike, z: ArrayLike | None = None
) -> tuple[float, float]:
    """Test ``x ⊥ y | z`` by partial correlation + Fisher-z; returns ``(partial_corr, p_value)``.

    With ``z=None`` this is the plain marginal-correlation test -- the miscalibrated one under
    autocorrelation. Pass the lagged parents as ``z`` for the calibrated MCI variant. ``z`` may be a
    single covariate ``(n,)`` or several stacked as ``(n, k)``. The p-value is two-sided.
    """
    residual_x, k = _residualize(x, z)
    residual_y, _ = _residualize(y, z)
    n = residual_x.shape[0]
    denom = math.sqrt(float(np.sum(residual_x**2)) * float(np.sum(residual_y**2))) + _EPS
    rho = float(np.sum(residual_x * residual_y)) / denom
    rho = min(max(rho, -1.0 + _EPS), 1.0 - _EPS)
    dof = max(n - k - 3, 1)  # Fisher-z uses sqrt(n - |z| - 3)
    stat = math.atanh(rho) * math.sqrt(dof)
    p_value = math.erfc(abs(stat) / math.sqrt(2.0))  # = 2 * (1 - Phi(|stat|)), two-sided
    return rho, p_value
