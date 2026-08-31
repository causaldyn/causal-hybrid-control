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

from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import ArrayLike

from chc.causal import _ols_with_intercept
from chc.toeplitz import levinson_durbin, sample_autocorrelation, solve_toeplitz


def _projection_design(
    data: Mapping[str, ArrayLike],
    horizon: int,
    treatment: str,
    outcome: str,
    adjust_for: tuple[str, ...],
    lags: int,
) -> tuple[Array, Array]:
    """Aligned rows of a lag-augmented local projection: features, and every horizon's response.

    ``responses[i, h]`` is the ``h``-step-ahead outcome for row ``i``. Each row carries its whole
    ``lags + horizon + 1``-step window, so a row can be moved -- or resampled -- without breaking
    the lead-lag alignment it encodes.
    """
    if lags < 0:
        raise ValueError(f"lags must be non-negative; got {lags}")
    treatment_series = jnp.asarray(data[treatment])
    outcome_series = jnp.asarray(data[outcome])
    total = treatment_series.shape[0]
    n = total - horizon - lags
    if n < 2:
        raise ValueError(
            f"series of length {total} is too short for horizon {horizon} with {lags} lags"
        )
    base = lags  # projections start at t = lags so every own-lag exists
    columns = [treatment_series[base : base + n]]
    columns += [jnp.asarray(data[name])[base : base + n] for name in adjust_for]
    for lag in range(1, lags + 1):
        columns.append(outcome_series[base - lag : base - lag + n])
        columns.append(treatment_series[base - lag : base - lag + n])
    features = jnp.stack(columns, axis=1)
    responses = jnp.stack([outcome_series[base + h : base + h + n] for h in range(horizon + 1)], 1)
    return features, responses


def _irf_from_design(features: Array, responses: Array) -> Array:
    """Treatment coefficient at every horizon -- column 0 of each projection."""
    return jnp.stack(
        [_ols_with_intercept(features, responses[:, h])[0] for h in range(responses.shape[1])]
    )


def local_projection_irf(
    data: Mapping[str, ArrayLike],
    horizon: int,
    treatment: str = "u",
    outcome: str = "x",
    adjust_for: tuple[str, ...] = ("x",),
    lags: int = 0,
) -> Array:
    """Jorda local-projections IRF ``beta_h = d outcome_{t+h}/d treatment_t``, ``h = 0..horizon``.

    ``data`` holds aligned trajectory columns. For each ``h`` the ``h``-step-ahead outcome is
    regressed on ``[treatment_t, *adjust_for_t]`` and an intercept; the treatment coefficient is the
    horizon-``h`` dynamic causal effect. With ``adjust_for`` the backdoor path is blocked (effect
    identified); omit the confounder and it stays confounded. Returns the length ``H + 1`` IRF.

    ``lags > 0`` adds ``lags`` own-lags of the outcome and of the treatment to every projection --
    *lag augmentation* (Montiel Olea & Plagborg-Moller 2021). It leaves the estimand alone and buys
    inference: without it, a coefficient's confidence interval loses coverage on persistent data
    and at long horizons, and a plant with a delay is persistent by construction. The default stays
    ``0`` so existing callers get the estimates they had.
    """
    return _irf_from_design(
        *_projection_design(data, horizon, treatment, outcome, adjust_for, lags)
    )


@dataclass(frozen=True)
class DelayEstimate:
    """How long a causal effect takes to arrive, with an interval rather than a bare point.

    ``lag`` is in steps and ``delay`` in the caller's time unit (``lag * dt``); ``lo`` and ``hi``
    bound the *delay*, not the lag. ``peak_response`` keeps its sign, so an effect that arrives
    negative is visible as such even though the peak is located on ``|beta|``.

    ``censored`` marks a peak sitting on an end of the horizon, where the delay is not identified:
    the true peak may lie anywhere beyond the window, so the number is a bound on the delay and must
    not be read as an estimate of it.
    """

    lag: float
    delay: float
    lo: float
    hi: float
    censored: bool
    peak_response: float
    irf: np.ndarray
    n_resamples: int


def peak_lag(irf: ArrayLike, refine: bool = False) -> tuple[float, bool]:
    """Location of the IRF's largest-magnitude response, and whether it is censored.

    Returns ``(lag, censored)``. Magnitude, not signed value, because a dynamic effect that
    overshoots and returns changes sign; the arrival time is where the response is *largest*,
    whichever way it points.

    ``refine`` fits a parabola through the peak and its two neighbours and returns the vertex, so
    the answer is not quantised to the sampling step. **It is off by default because a causal
    impulse response is one-sided** -- zero before the effect arrives, decaying after -- and so is
    maximally asymmetric at exactly the point being located. On a geometrically decaying response
    with ratio ``phi`` the vertex lands at ``lag + phi/(2 * (2 - phi))``: 0.41 of a step late at
    ``phi = 0.9``, a bias that does not shrink with the sample and swamps the bootstrap width.
    Turn it on when the response is genuinely smooth across the grid -- a delay that is not an
    integer multiple of ``dt`` smears across neighbouring bins, and then the integer argmax can only
    quantise. Even there the vertex is exact only for a symmetric peak: a lag split ``(1 - f, f)``
    across two bins puts it at ``f/(4 - 6f)``, so ``f = 1/2`` comes back exactly and ``f = 1/3``
    comes back as ``1/6``. It moves the right way and continuously; it is not unbiased.

    A peak at either end of the horizon is returned unrefined and flagged ``censored``: there is no
    neighbour to fit through, and the response is still rising at the edge, so the true peak may lie
    outside the window. ``censored`` is *not* a test for an identified peak -- an effect that never
    arrives within the horizon leaves noise, whose argmax sits wherever noise puts it. What flags
    that case is the width of the interval :func:`delay_estimate` returns, not this flag.

    The fit is local to three points, so a plateau wider than that is read from its leading edge:
    two tied bins give the midpoint (the wanted answer for a delay halfway between samples), three
    give the midpoint of the first two. Every closed form here is derived in
    ``validation/delay_estimate_bias.mac``.
    """
    response = np.abs(np.asarray(irf, dtype=np.float64))
    if response.size == 0:
        raise ValueError("irf is empty")
    index = int(np.argmax(response))
    if index == 0 or index == response.size - 1:
        return float(index), True
    if not refine:
        return float(index), False
    left, centre, right = response[index - 1], response[index], response[index + 1]
    # Associated this way the curvature is provably negative: argmax returns the *first* maximum,
    # so left < centre strictly, and right <= centre. No degenerate-denominator branch is reachable.
    curvature = (left - centre) + (right - centre)
    return float(index) + 0.5 * (left - right) / curvature, False


def _irf_from_rows(features: np.ndarray, responses: np.ndarray) -> np.ndarray:
    """Treatment coefficient at every horizon, from design rows that already carry an intercept."""
    coefficients, *_ = np.linalg.lstsq(features, responses, rcond=None)
    return coefficients[0]


def delay_estimate(
    data: Mapping[str, ArrayLike],
    horizon: int,
    dt: float = 1.0,
    treatment: str = "u",
    outcome: str = "x",
    adjust_for: tuple[str, ...] = ("x",),
    lags: int = 0,
    refine: bool = False,
    n_resamples: int = 400,
    level: float = 0.95,
    seed: int = 0,
) -> DelayEstimate:
    """Estimate how long a causal effect takes to propagate, with a bootstrap interval.

    This is the delay half of a *delay causal graph*: an edge already carries a lag from
    :class:`chc.discovery.LaggedGraph`, but a lag with no uncertainty is a claim nobody can check.

    The estimand is the **transport lag of the treatment-to-outcome path** -- how long an
    intervention takes to show its largest effect -- read off :func:`local_projection_irf` with
    ``adjust_for`` blocking the backdoor path, so it is the delay of the *causal* effect and not of
    a correlation. That is the ``tau`` of a plant whose delay sits on the actuation path;
    :class:`chc.delay.DelayedDynamics` carries its delay on the state-feedback path instead, so its
    ``tau`` is a different quantity and this estimate does not identify it.

    Set ``lags > 0`` if you intend to read coefficients off ``irf``; it does not change *this*
    interval. Lag augmentation fixes the asymptotic variance of a single local-projection
    coefficient under persistence (Montiel Olea & Plagborg-Moller 2021), and a delayed plant is
    persistent by construction -- but the peak is a location statistic, invariant to a common
    rescaling of the IRF, and the block bootstrap below already handles the serial dependence
    non-parametrically. Measured over 40 replications at ``phi in {0.98, 0.999, 1.0}``, coverage of
    a nominal 95% interval was 0.95-1.00 with and without augmentation and the widths agreed to 3%.
    So the default matches :func:`local_projection_irf`, and the option is offered rather than
    imposed.

    The interval is a **moving-block** percentile bootstrap over the aligned projection rows, refit
    and re-peaked on every resample, with block length ``lags + horizon + 1``. Blocks, because a
    local projection's rows overlap by construction -- rows closer together than the window share
    observations -- so resampling rows independently would destroy the dependence that sets the
    width and report an interval far too narrow. That window length is also exactly the separation
    at which two rows stop sharing anything, which is what makes it the natural block.

    A flat or noise-dominated IRF is not rejected and not smoothed over: the peak wanders across the
    horizon under resampling and the interval comes back wide, which is the honest report. A sharply
    identified lag goes the other way and returns ``lo == hi``: the resampled peak never moves off
    its bin, which is the correct interval for a delay that is an exact multiple of ``dt``.

    ``refine`` is forwarded to :func:`peak_lag`; read its warning before turning it on.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1); got {level}")
    if n_resamples < 2:
        raise ValueError(f"n_resamples must be at least 2; got {n_resamples}")

    features, responses = _projection_design(data, horizon, treatment, outcome, adjust_for, lags)
    irf = np.asarray(_irf_from_design(features, responses))
    lag, censored = peak_lag(irf, refine)

    rows = np.asarray(features, dtype=np.float64)
    design = np.concatenate([rows, np.ones((rows.shape[0], 1))], axis=1)
    targets = np.asarray(responses, dtype=np.float64)
    n = design.shape[0]
    block = lags + horizon + 1
    if n < 4 * block:
        raise ValueError(
            f"{n} usable rows cannot carry a block bootstrap of block length {block}; "
            f"the series needs at least {4 * block + horizon + lags} observations"
        )

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    draws = np.empty(n_resamples)
    for draw in range(n_resamples):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        index = (starts[:, None] + offsets[None, :]).ravel()[:n]
        draws[draw] = peak_lag(_irf_from_rows(design[index], targets[index]), refine)[0]

    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(draws, [alpha, 1.0 - alpha])
    return DelayEstimate(
        lag=lag,
        delay=lag * dt,
        lo=float(lo) * dt,
        hi=float(hi) * dt,
        censored=censored,
        peak_response=float(irf[round(lag)]),
        irf=irf,
        n_resamples=n_resamples,
    )


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
    data: Mapping[str, ArrayLike],
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
