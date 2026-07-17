"""Dynamic causal effects: the impulse-response / carryover kernel for control (plans/18).

CHC's one-step ``estimate_control_effect`` gives the scalar ``d x_next/d u``; a controller planning
over a horizon needs the whole ``d x_{t+h}/d u_t``, ``h = 0..H`` -- how an intervention
propagates over time (the impulse response, or an MMM **adstock carryover kernel**). This is Jorda
**local projections**: one regression per horizon of the ``h``-step-ahead outcome on the treatment
plus an adjustment set, so conditioning on the state/confounders blocks the backdoor path exactly as
the one-step estimate does. The dynamic sibling of :func:`chc.causal.estimate_control_effect`; see
``plans/18``. This is a causal-effect estimator, not a sequence model.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import ArrayLike

from chc.causal import _ols_with_intercept
from chc.toeplitz import levinson_durbin, sample_autocorrelation, solve_toeplitz


def local_projection_irf(
    data: dict[str, Array],
    horizon: int,
    treatment: str = "u",
    outcome: str = "x",
    adjust_for: tuple[str, ...] = ("x",),
) -> Array:
    """Jorda local-projections IRF ``beta_h = d outcome_{t+h}/d treatment_t``, ``h = 0..horizon``.

    ``data`` holds aligned trajectory columns. For each ``h`` the ``h``-step-ahead outcome is
    regressed on ``[treatment_t, *adjust_for_t]`` and an intercept; the treatment coefficient is the
    horizon-``h`` dynamic causal effect. With ``adjust_for`` the backdoor path is blocked (effect
    identified); omit the confounder and it stays confounded. Returns the length ``H + 1`` IRF.
    """
    treatment_series = jnp.asarray(data[treatment])
    outcome_series = jnp.asarray(data[outcome])
    n = treatment_series.shape[0] - horizon
    covariates = [jnp.asarray(data[name])[:n] for name in adjust_for]
    features = jnp.stack([treatment_series[:n], *covariates], axis=1)
    responses = [
        _ols_with_intercept(features, outcome_series[h : h + n])[0]  # treatment coefficient
        for h in range(horizon + 1)
    ]
    return jnp.stack(responses)


def innovations(series: ArrayLike, order: int) -> np.ndarray:
    """Prewhitened residual: fit an AR(``order``) via Levinson, return ``series - AR-prediction``.

    The Box-Jenkins **innovation** -- the unpredictable part left after the autoregressive structure
    is removed. Prewhitening both series before cross-correlating de-biases it under autocorrelation
    -- a classical complement to the MCI test in :mod:`chc.independence`.
    """
    series = np.asarray(series, dtype=np.float64)
    ar, _reflection, _error = levinson_durbin(sample_autocorrelation(series, order))
    n = series.shape[0]
    prediction = sum(ar[i - 1] * series[order - i : n - i] for i in range(1, order + 1))
    return series[order:] - prediction


def structured_irf(
    data: dict[str, Array],
    horizon: int,
    order: int = 4,
    treatment: str = "u",
    outcome: str = "x",
    adjust_for: tuple[str, ...] = ("x",),
) -> np.ndarray:
    """Structured IRF: AR dynamics from Levinson + the one-step impact, propagated (transfer fn).

    Fits the outcome's AR(``order``) via Levinson on the biased (PSD) autocorrelation -- the dynamic
    propagation -- and the one-step treatment impact (confounding-adjusted ``h = 1`` local
    projection), then propagates ``g_0 = 0``, ``g_1 = impact``, ``g_h = sum_i a_i g_{h-i}``. Agrees
    with the local-projections IRF but makes the AR structure explicit (reflection coeffs).
    """
    ar, _reflection, _error = levinson_durbin(sample_autocorrelation(data[outcome], order))
    impact = float(local_projection_irf(data, 1, treatment, outcome, adjust_for)[1])
    response = np.zeros(horizon + 1)
    response[1] = impact
    for h in range(2, horizon + 1):
        response[h] = sum(ar[i - 1] * response[h - i] for i in range(1, min(order, h) + 1))
    return response


def irf_control_sequence(irf: ArrayLike, target: ArrayLike) -> np.ndarray:
    """Feed-forward control to track a target output, by deconvolving the impulse response.

    The output is the causal convolution of the control with the IRF (``x = G u``, ``G`` lower-tri
    Toeplitz of the response), so achieving a target trajectory ``x*`` is the deconvolution
    ``u = G^{-1} x*`` -- solved with the Toeplitz machinery. This makes the *whole* dynamic effect
    actionable: it accounts for carryover, where a one-step controller that inverts only ``g_1``
    over-actuates on a delayed plant (steady-state error ``sum_h g_h / g_1``). See ``plans/18``.
    """
    kernel = np.asarray(irf, dtype=np.float64)[1:]  # drop g_0 = 0; the causal impulse response
    target = np.asarray(target, dtype=np.float64)
    horizon = target.shape[0]
    first_col = np.zeros(horizon)
    first_col[: min(kernel.shape[0], horizon)] = kernel[:horizon]
    first_row = np.zeros(horizon)
    first_row[0] = first_col[0]
    return solve_toeplitz(first_col, first_row, target)
