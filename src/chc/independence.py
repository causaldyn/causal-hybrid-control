"""Autocorrelation-robust conditional-independence testing (the MCI idea, borrowed from tigramite).

A linear partial-correlation (ParCorr) test of ``x ⊥ y | z``: residualise ``x`` and ``y`` on the
conditioning set ``z``, then Fisher-z test the residual correlation. In serially correlated data a
naive ``corr(x, y)`` is badly miscalibrated -- autocorrelation inflates the effective variance of
the estimator, so unrelated series look linked. Conditioning on the lagged parents (tigramite's
*momentary conditional independence*) whitens the residuals and restores calibration. This is the CI
primitive ``chc.discovery`` screens lagged parents with; see ``plans/17``. The method (partial
correlation) is standard; only the autocorrelation-aware *usage* is borrowed -- no tigramite code.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.scipy.special import erfc

_EPS = 1e-12


def _residualize(target: Array, conditioning: Array | None) -> tuple[Array, int]:
    """Residual of ``target`` after linear regression on ``[1, conditioning]``; returns (resid, k).

    ``k`` is the number of conditioning columns (0 when ``conditioning`` is ``None``) -- the
    degrees-of-freedom correction for the Fisher-z statistic. A ``(n,)`` or ``(n, k)`` conditioning
    set is accepted; a ``(k, n)`` one is transposed to rows-are-samples.
    """
    target = jnp.asarray(target, dtype=float).ravel()
    if conditioning is None:
        return target - target.mean(), 0
    cond = jnp.atleast_2d(jnp.asarray(conditioning, dtype=float))
    if cond.shape[0] != target.shape[0]:
        cond = cond.T
    design = jnp.concatenate([jnp.ones((target.shape[0], 1)), cond], axis=1)
    coeffs, *_ = jnp.linalg.lstsq(design, target, rcond=None)
    return target - design @ coeffs, cond.shape[1]


def partial_corr_test(x: Array, y: Array, z: Array | None = None) -> tuple[Array, Array]:
    """Test ``x ⊥ y | z`` by partial correlation + Fisher-z; returns ``(partial_corr, p_value)``.

    With ``z=None`` this is the plain marginal-correlation test -- the miscalibrated one under
    autocorrelation. Pass the lagged parents as ``z`` for the calibrated MCI variant. ``z`` may be a
    single covariate ``(n,)`` or several stacked as ``(n, k)``. The p-value is two-sided.
    """
    residual_x, k = _residualize(x, z)
    residual_y, _ = _residualize(y, z)
    n = residual_x.shape[0]
    denom = jnp.sqrt(jnp.sum(residual_x**2) * jnp.sum(residual_y**2)) + _EPS
    rho = jnp.clip(jnp.sum(residual_x * residual_y) / denom, -1.0 + _EPS, 1.0 - _EPS)
    dof = jnp.maximum(n - k - 3, 1)  # Fisher-z uses sqrt(n - |z| - 3)
    stat = jnp.arctanh(rho) * jnp.sqrt(dof)
    p_value = erfc(jnp.abs(stat) / jnp.sqrt(2.0))  # = 2 * (1 - Phi(|stat|)), two-sided
    return rho, p_value
