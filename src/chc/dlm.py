"""Discount dynamic linear models: forward filtering with a learned observational variance,
smoothing, backward sampling, forecasting and monitoring. *Experimental.*

The model is West and Harrison's (1997): ``y_t = F_t' theta_t + nu_t`` with
``theta_t = G theta_(t-1) + omega_t``, where the evolution variance is not a parameter but a
discount of the information the last step carried. :class:`DynamicLinearModel` is built from
blocks, each with its own discount ``delta`` in ``(0, 1]``:

* :class:`Polynomial`: a level, or a level with a slope, and higher orders;
* :class:`Seasonal`: Fourier harmonics of a period, which need not be an integer;
* :class:`Regression`: coefficients on columns of ``x``, which follow random walks.

With ``P_t = G C_(t-1) G'`` the evolved posterior covariance, the prior covariance is

* ``form="additive"`` (the default, West and Harrison's component discounting):
  ``R_t = P_t + blockdiag_i(P_t,ii (1 - delta_i) / delta_i)``, which inflates each block's own
  covariance and leaves the covariance between blocks as it was;
* ``form="multiplicative"``: ``R_t = D P_t D``, ``D = diag(delta_i^(-1/2))``, which also inflates
  the covariance between blocks.

With one discount for every block the multiplicative form is the single discount,
``R_t = P_t / delta``; the additive form is not, since it still leaves the covariance between
blocks uninflated. No third form is offered.

The observational variance is learned (Normal-Gamma): ``Prior.dof`` is ``n_0`` and
``Prior.scale`` is ``S_0``, the prior point estimate of the variance; an infinite ``dof`` means
the variance is known and equal to ``scale``. ``variance_discount`` (``beta``) lets it move:
``n_t = beta n_(t-1) + 1``. The one-step forecast is Student ``t`` with ``beta n_(t-1)`` degrees of
freedom, location ``f_t`` and squared scale ``Q_t``, and :attr:`DLMFit.log_likelihood` sums its
log density over the observed steps: that is the criterion for choosing discounts. The update is
the Joseph form, so a covariance stays symmetric and positive semidefinite in floating point; a
step whose ``Q_t`` is not positive and finite raises rather than being zeroed.

Missing observations are ``NaN`` in ``y``. A missing step evolves and does not update:
``m_t = a_t``, ``C_t = R_t``, and the variance's degrees of freedom decay by ``beta``.

What is exact and what is not:

* The filter and the forecasts are exact for the model. So is :func:`backward_sample` (forward
  filtering, backward sampling) when ``beta = 1``; below 1 it draws the variance path by the
  beta-gamma evolution run backwards and each state given the next at that step's variance
  (McAlinn and West 2019, appendix A.2).
* :func:`smooth` returns the exact means, covariances and lag-one covariances of those draws.
  Each covariance is in the units of ``E[V_t | D_T]``: with ``beta = 1`` that is the final
  estimate ``n_T S_T / (n_T - 2)`` at every step, not the ``S_t`` the filter had at ``t`` (a
  smoother that keeps each ``C_t`` in its own units misstates it by ``S_T / S_t``); below 1 it is
  one quadrature a step. Only the marginals' shape is approximate below 1, where they are not
  Student ``t``; :attr:`SmoothedStates.exact` says which.
* :func:`forecast` holds the evolution variance of the first step ahead for every later one by
  default (``evolution="constant"``); ``"compounding"`` applies the discount at every step, which
  inflates the ``k``-step state variance by ``delta^(-k)``. The two differ by a factor 3.9 in the
  state variance of a level 52 steps ahead at ``delta = 0.95``.

What it does not do, and what to watch:

* **Discounting is forgetting.** With ``G = I`` and a single discount the posterior mean is
  weighted least squares with weights ``delta^(T - t)`` (Ameen and Harrison 1984), so the
  effective sample is ``(1 + delta) / (1 - delta)`` observations.
* **A regressor that is zero carries no information, and its coefficient's variance still grows**
  by ``1 / delta`` a step. ``Regression(hold_when_idle=True)`` discounts only the coordinates
  whose regressor is non-zero at the step, a coordinate-wise form of restricted forgetting
  (Kulhavy 1987); a long idle run without it is logged as a warning (``chc_event="dlm_windup"``).
* **A coefficient that moves is not identified from a saturation that bends.** On observational
  marketing data a time-varying coefficient and a nonlinear response can explain the same series
  (Dew et al. 2024, arXiv:2408.07678); this module fits the first and cannot tell them apart.
* **A discount picked by one-step likelihood can be too high for the coefficients' intervals.**
  On the random-walk world of ``scripts/bench_dlm.py``, 500 series, the likelihood picked the
  coefficients' discount at 0.95 or above in 450 and below 0.9 in none, and the 90% intervals of a
  channel's contribution over 13 weeks covered 0.78 and 0.77; fixed at 0.85 or 0.9 they covered
  0.88-0.90. An interval reported from the likelihood's pick alone is too narrow.

:func:`monitor_evalues` turns the one-step errors into e-values for West and Harrison's (1986)
alternatives -- a shift of the location, an inflation of the scale -- which
:class:`chc.gate.DriftAlarm` turns into an alarm with a guaranteed average run length.
``forward_filter(interventions=...)`` applies an extra discount at named steps, their
feed-back intervention, and logs each (``chc_event="dlm_intervention"``).

The implementation is NumPy in float64 and written from the published equations.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate, special

_log = logging.getLogger(__name__)

_Array = NDArray[np.float64]

DiscountForm = Literal["additive", "multiplicative"]
ForecastEvolution = Literal["constant", "compounding"]

# A regressor idle long enough to inflate its coefficient's variance this many times is logged.
_WINDUP_WARN = 10.0


def _discount(value: float, name: str = "discount") -> float:
    delta = float(value)
    if not 0.0 < delta <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value}")
    return delta


@dataclass(frozen=True)
class Polynomial:
    """A polynomial trend: ``order=1`` is a level, ``order=2`` a level and a slope.

    ``F = (1, 0, ..., 0)`` and ``G`` is the upper bidiagonal matrix of ones.

    Raises:
        ValueError: on an ``order`` below 1 or a ``discount`` outside ``(0, 1]``.
    """

    order: int
    discount: float

    def __post_init__(self) -> None:
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 1:
            raise ValueError(f"order must be a positive integer, got {self.order!r}")
        object.__setattr__(self, "discount", _discount(self.discount))

    @property
    def size(self) -> int:
        return self.order

    def _evolution(self) -> _Array:
        return np.eye(self.order) + np.eye(self.order, k=1)

    def _design(self) -> _Array:
        f = np.zeros(self.order)
        f[0] = 1.0
        return f


@dataclass(frozen=True)
class Seasonal:
    """Fourier harmonics ``j`` of a ``period``: a rotation by ``2 pi j / period`` a step for each.

    A harmonic takes two state coordinates, except ``j = period / 2``, which takes one and flips
    sign every step. ``period`` need not be an integer (52.18 weeks to a year).

    Raises:
        ValueError: on a ``period`` not above 1, harmonics that are not distinct integers in
            ``[1, period / 2]``, or a ``discount`` outside ``(0, 1]``.
    """

    period: float
    harmonics: tuple[int, ...]
    discount: float

    def __post_init__(self) -> None:
        period = float(self.period)
        if not (math.isfinite(period) and period > 1.0):
            raise ValueError(f"period must be finite and above 1, got {self.period}")
        harmonics = tuple(self.harmonics)
        if not harmonics:
            raise ValueError("harmonics must name at least one harmonic")
        for j in harmonics:
            if isinstance(j, bool) or not isinstance(j, int) or not 1 <= j <= period / 2.0:
                raise ValueError(
                    f"harmonic {j!r} is not an integer in [1, period / 2 = {period / 2}]"
                )
        if len(set(harmonics)) != len(harmonics):
            raise ValueError(f"harmonics repeat: {harmonics}")
        object.__setattr__(self, "period", period)
        object.__setattr__(self, "harmonics", tuple(sorted(harmonics)))
        object.__setattr__(self, "discount", _discount(self.discount))

    def _nyquist(self, j: int) -> bool:
        return 2 * j == self.period

    @property
    def size(self) -> int:
        return sum(1 if self._nyquist(j) else 2 for j in self.harmonics)

    def _evolution(self) -> _Array:
        g = np.zeros((self.size, self.size))
        i = 0
        for j in self.harmonics:
            if self._nyquist(j):
                g[i, i] = -1.0
                i += 1
                continue
            w = 2.0 * math.pi * j / self.period
            c, s = math.cos(w), math.sin(w)
            g[i : i + 2, i : i + 2] = [[c, s], [-s, c]]
            i += 2
        return g

    def _design(self) -> _Array:
        f = np.zeros(self.size)
        i = 0
        for j in self.harmonics:
            f[i] = 1.0
            i += 1 if self._nyquist(j) else 2
        return f


@dataclass(frozen=True)
class Regression:
    """Coefficients on ``width`` columns of ``x``, each a random walk (``G = I``).

    The columns are taken from ``x`` in block order: the first regression block reads the first
    ``width`` columns, the next the columns after them.

    ``hold_when_idle=True`` discounts, at each step, only the coordinates whose regressor is
    non-zero, so a coefficient's variance does not grow while its column is zero.

    Raises:
        ValueError: on a ``width`` below 1 or a ``discount`` outside ``(0, 1]``.
    """

    width: int
    discount: float
    hold_when_idle: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width < 1:
            raise ValueError(f"width must be a positive integer, got {self.width!r}")
        object.__setattr__(self, "discount", _discount(self.discount))

    @property
    def size(self) -> int:
        return self.width

    def _evolution(self) -> _Array:
        return np.eye(self.width)


Block = Polynomial | Seasonal | Regression


@dataclass(frozen=True)
class Prior:
    """The state and the observational variance before the first observation.

    ``theta_0 ~ T_dof(mean, covariance)`` and ``1 / V ~ Gamma(dof / 2, dof * scale / 2)``: ``scale``
    is the prior point estimate of the variance and ``covariance`` is stated at it. An infinite
    ``dof`` means the variance is known, equal to ``scale``, and the state's prior is normal.

    ``covariance`` must be positive definite. Every block's ``G`` is invertible, so every later
    prior and posterior covariance is positive definite too, which the smoother relies on.

    Raises:
        ValueError: on shapes that do not agree, a non-finite entry, a ``covariance`` that is not
            symmetric positive definite, or a ``scale`` or ``dof`` that is not positive.
    """

    mean: _Array  # (p,)
    covariance: _Array  # (p, p)
    scale: float
    dof: float

    def __post_init__(self) -> None:
        mean = np.array(self.mean, dtype=np.float64)
        if mean.ndim != 1 or mean.shape[0] == 0 or not np.all(np.isfinite(mean)):
            raise ValueError(f"mean must be a finite vector, got shape {mean.shape}")
        p = mean.shape[0]
        cov = np.array(self.covariance, dtype=np.float64)
        if cov.shape != (p, p) or not np.all(np.isfinite(cov)):
            raise ValueError(f"covariance must be a finite ({p}, {p}) matrix, got {cov.shape}")
        scale_of = max(1.0, float(np.abs(cov).max()))
        if not np.allclose(cov, cov.T, rtol=0.0, atol=1e-12 * scale_of):
            raise ValueError("covariance is not symmetric")
        try:
            np.linalg.cholesky(_sym(cov))
        except np.linalg.LinAlgError:
            raise ValueError("covariance is not positive definite") from None
        scale, dof = float(self.scale), float(self.dof)
        if not (math.isfinite(scale) and scale > 0.0):
            raise ValueError(f"scale must be positive and finite, got {self.scale}")
        if not dof > 0.0:
            raise ValueError(
                f"dof must be positive (math.inf for a known variance), got {self.dof}"
            )
        mean.setflags(write=False)
        cov.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", cov)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "dof", dof)


@dataclass(frozen=True)
class DynamicLinearModel:
    """Blocks, their prior, how their discounts combine, and the variance discount.

    The state is the blocks' coordinates in order. See the module docstring for ``form``.

    Raises:
        ValueError: on no blocks, a prior whose size is not the blocks' total, an unknown
            ``form``, a ``variance_discount`` outside ``(0, 1]``, or a ``variance_discount``
            below 1 with a known variance.
    """

    blocks: tuple[Block, ...]
    prior: Prior
    form: DiscountForm = "additive"
    variance_discount: float = 1.0

    def __post_init__(self) -> None:
        blocks = tuple(self.blocks)
        if not blocks:
            raise ValueError("a model needs at least one block")
        for b in blocks:
            if not isinstance(b, Polynomial | Seasonal | Regression):
                raise ValueError(f"not a block: {b!r}")
        size = sum(b.size for b in blocks)
        if self.prior.mean.shape[0] != size:
            raise ValueError(
                f"the prior has {self.prior.mean.shape[0]} coordinates, the blocks {size}"
            )
        if self.form not in ("additive", "multiplicative"):
            raise ValueError(f"form must be 'additive' or 'multiplicative', got {self.form!r}")
        beta = _discount(self.variance_discount, "variance_discount")
        if beta < 1.0 and math.isinf(self.prior.dof):
            raise ValueError(
                "variance_discount below 1 needs a learned variance (a finite prior dof)"
            )
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "variance_discount", beta)

    @property
    def size(self) -> int:
        """The state's dimension."""
        return sum(b.size for b in self.blocks)

    @property
    def regressors(self) -> int:
        """How many columns ``x`` must have."""
        return sum(b.width for b in self.blocks if isinstance(b, Regression))


@dataclass(frozen=True)
class _Structure:
    """What the filter reads off a model, once."""

    evolution: _Array  # G, (p, p)
    design: _Array  # F's fixed part, (p,), zero on the regression coordinates
    regression: NDArray[np.intp]  # the regression coordinates, in x's column order
    slices: tuple[slice, ...]
    rates: _Array  # (1 - delta) / delta per block
    inv_sqrt: _Array  # delta^(-1/2) per coordinate
    hold: _Array  # per coordinate: its discount is skipped while its regressor is zero


def _structure(model: DynamicLinearModel) -> _Structure:
    p = model.size
    g = np.zeros((p, p))
    f = np.zeros(p)
    regression: list[int] = []
    slices: list[slice] = []
    rates: list[float] = []
    inv_sqrt = np.ones(p)
    hold = np.zeros(p, dtype=bool)
    i = 0
    for b in model.blocks:
        sl = slice(i, i + b.size)
        g[sl, sl] = b._evolution()
        if isinstance(b, Regression):
            regression.extend(range(i, i + b.size))
            hold[sl] = b.hold_when_idle
        else:
            f[sl] = b._design()
        slices.append(sl)
        rates.append((1.0 - b.discount) / b.discount)
        inv_sqrt[sl] = b.discount**-0.5
        i += b.size
    return _Structure(
        g,
        f,
        np.array(regression, dtype=np.intp),
        tuple(slices),
        np.array(rates),
        inv_sqrt,
        hold,
    )


def _sym(m: _Array) -> _Array:
    return 0.5 * (m + m.T)


def _evolve(s: _Structure, form: DiscountForm, p: _Array, active: _Array) -> _Array:
    """``R`` from ``P = G C G'``: each block's discount, skipped on idle held coordinates."""
    live = np.where(s.hold, active, True)
    if form == "multiplicative":
        d = np.where(live, s.inv_sqrt, 1.0)
        return d[:, None] * p * d[None, :]
    r = p.copy()
    for sl, rate in zip(s.slices, s.rates, strict=True):
        on = live[sl].astype(np.float64)
        r[sl, sl] += rate * p[sl, sl] * np.outer(on, on)
    return r


def _log_t(e: float, dof: float, q: float) -> float:
    """``log`` of the Student ``t`` density with ``dof`` degrees of freedom and squared scale ``q``
    at ``e``; normal when ``dof`` is infinite."""
    if math.isinf(dof):
        return -0.5 * (math.log(2.0 * math.pi * q) + e * e / q)
    return float(
        special.gammaln((dof + 1.0) / 2.0)
        - special.gammaln(dof / 2.0)
        - 0.5 * math.log(dof * math.pi * q)
        - (dof + 1.0) / 2.0 * math.log1p(e * e / (dof * q))
    )


@dataclass(frozen=True)
class DLMFit:
    """The forward filter's output, one row per step, in West and Harrison's notation.

    At step ``t``: the prior ``theta_t | D_(t-1)`` has mean ``a`` and covariance ``R``; the one-step
    forecast is Student ``t`` with ``one_step_dof`` degrees of freedom, location ``one_step_mean``
    (``f``) and squared scale ``one_step_scale`` (``Q``), so its variance is
    ``Q dof / (dof - 2)``; the posterior has mean ``m`` and covariance ``C``, the variance estimate
    ``S`` with ``n`` degrees of freedom. ``log_scores`` is the one-step forecast's log density at
    the observation, ``NaN`` on a missing step. ``C`` is stated at ``S_t``, ``R`` at ``S_(t-1)``.
    """

    model: DynamicLinearModel
    y: _Array  # (T,)
    design: _Array  # F_t, (T, p)
    prior_mean: _Array  # a_t, (T, p)
    prior_covariance: _Array  # R_t, (T, p, p)
    one_step_mean: _Array  # f_t, (T,)
    one_step_scale: _Array  # Q_t, (T,)
    one_step_dof: _Array  # beta n_(t-1), (T,)
    mean: _Array  # m_t, (T, p)
    covariance: _Array  # C_t, (T, p, p)
    dof: _Array  # n_t, (T,)
    scale: _Array  # S_t, (T,)
    log_scores: _Array  # (T,)

    @property
    def log_likelihood(self) -> float:
        """The sum of the one-step forecasts' log densities over the observed steps."""
        return float(np.nansum(self.log_scores))

    @property
    def errors(self) -> _Array:
        """One-step errors ``y_t - f_t``, ``NaN`` on a missing step."""
        return self.y - self.one_step_mean


def forward_filter(
    model: DynamicLinearModel,
    y: ArrayLike,
    x: ArrayLike | None = None,
    *,
    interventions: Mapping[int, float] | None = None,
) -> DLMFit:
    """Filter ``y`` through ``model``, with ``x`` the regression blocks' columns.

    ``interventions`` maps a step to an extra discount applied to the whole prior covariance at
    that step, ``R_t / delta``: West and Harrison's feed-back intervention after a monitor's
    alarm. Each is logged.

    Raises:
        ValueError: on a ``y`` that is not a vector of finite values and ``NaN``, an ``x`` whose
            shape is not ``(T, model.regressors)`` or that is not finite, or an intervention at a
            step outside ``y`` or with a discount outside ``(0, 1]``.
        FloatingPointError: on a step whose one-step squared scale is not positive and finite.
    """
    ys = np.array(y, dtype=np.float64)
    if ys.ndim != 1 or ys.shape[0] == 0:
        raise ValueError(f"y must be a non-empty vector, got shape {ys.shape}")
    if np.any(np.isinf(ys)):
        raise ValueError("y has an infinite value; a missing observation is NaN")
    horizon = ys.shape[0]
    xs = _regressors(model, x, horizon)
    extra = _interventions(interventions, horizon)
    s = _structure(model)
    p = model.size
    beta = model.variance_discount

    design = np.tile(s.design, (horizon, 1))
    design[:, s.regression] = xs
    active = design != 0.0

    a_all = np.empty((horizon, p))
    r_all = np.empty((horizon, p, p))
    m_all = np.empty((horizon, p))
    c_all = np.empty((horizon, p, p))
    f_all = np.empty(horizon)
    q_all = np.empty(horizon)
    nu_all = np.empty(horizon)
    n_all = np.empty(horizon)
    s_all = np.empty(horizon)
    score = np.full(horizon, np.nan)

    m = model.prior.mean.copy()
    c = model.prior.covariance.copy()
    n, v = model.prior.dof, model.prior.scale
    eye = np.eye(p)
    for t in range(horizon):
        a = s.evolution @ m
        r = _sym(_evolve(s, model.form, s.evolution @ c @ s.evolution.T, active[t]))
        if t in extra:
            r = r / extra[t]
            _log.info(
                "dlm intervention at step %d: prior covariance discounted by %g",
                t,
                extra[t],
                extra={
                    "chc_event": "dlm_intervention",
                    "step": t,
                    "discount": extra[t],
                },
            )
        f_row = design[t]
        rf = r @ f_row
        f = float(f_row @ a)
        q = float(f_row @ rf) + v
        if not (math.isfinite(q) and q > 0.0):
            raise FloatingPointError(f"step {t}: the one-step squared scale is {q}")
        nu = beta * n
        a_all[t], r_all[t], f_all[t], q_all[t], nu_all[t] = a, r, f, q, nu
        if math.isnan(ys[t]):
            m, c, n = a, r, beta * n
        else:
            e = float(ys[t] - f)
            gain = rf / q
            n_new = beta * n + 1.0
            v_new = v if math.isinf(n_new) else v * (beta * n + e * e / q) / n_new
            shrink = eye - np.outer(gain, f_row)
            c = _sym((v_new / v) * (shrink @ r @ shrink.T + np.outer(gain, gain) * v))
            m = a + gain * e
            score[t] = _log_t(e, nu, q)
            n, v = n_new, v_new
        m_all[t], c_all[t], n_all[t], s_all[t] = m, c, n, v

    _warn_windup(model, s, design)
    return DLMFit(
        model,
        ys,
        design,
        a_all,
        r_all,
        f_all,
        q_all,
        nu_all,
        m_all,
        c_all,
        n_all,
        s_all,
        score,
    )


def _regressors(model: DynamicLinearModel, x: ArrayLike | None, horizon: int) -> _Array:
    k = model.regressors
    if x is None:
        if k:
            raise ValueError(f"the model's regression blocks need x with {k} columns")
        return np.zeros((horizon, 0))
    xs = np.array(x, dtype=np.float64)
    if xs.ndim == 1 and k == 1:
        xs = xs[:, None]
    if xs.shape != (horizon, k):
        raise ValueError(f"x must have shape ({horizon}, {k}), got {xs.shape}")
    if not np.all(np.isfinite(xs)):
        raise ValueError("x is not finite")
    return xs


def _interventions(interventions: Mapping[int, float] | None, horizon: int) -> dict[int, float]:
    out: dict[int, float] = {}
    for step, delta in (interventions or {}).items():
        if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < horizon:
            raise ValueError(f"intervention step {step!r} is outside [0, {horizon})")
        out[step] = _discount(delta, f"the intervention discount at step {step}")
    return out


def _warn_windup(model: DynamicLinearModel, s: _Structure, design: _Array) -> None:
    grown: list[str] = []
    i = 0
    for b in model.blocks:
        if isinstance(b, Regression) and not b.hold_when_idle and b.discount < 1.0:
            for j in range(b.size):
                idle = design[:, i + j] == 0.0
                run = longest = 0
                for flag in idle:
                    run = run + 1 if flag else 0
                    longest = max(longest, run)
                growth = b.discount ** (-longest)
                if growth > _WINDUP_WARN:
                    grown.append(f"coordinate {i + j}: {longest} idle steps, x{growth:.3g}")
        i += b.size
    if grown:
        _log.warning(
            "dlm windup: a regressor stayed zero long enough for its coefficient's variance to"
            " grow more than %g times (%s); Regression(hold_when_idle=True) discounts only while"
            " the regressor is non-zero",
            _WINDUP_WARN,
            "; ".join(grown),
            extra={"chc_event": "dlm_windup", "coordinates": grown},
        )


# ------------------------------------------------------------------------------------- smoothing


def _gains(fit: DLMFit) -> _Array:
    """``B_t = C_t G' R_(t+1)^(-1)`` for ``t < T``: scale-free, since ``C_t`` and ``R_(t+1)`` are
    both stated at ``S_t``."""
    g = _structure(fit.model).evolution
    horizon, p = fit.mean.shape
    out = np.empty((max(horizon - 1, 0), p, p))
    for t in range(horizon - 1):
        out[t] = np.linalg.solve(fit.prior_covariance[t + 1], g @ fit.covariance[t]).T
    return out


def _variance_factor(fit: DLMFit) -> tuple[_Array, _Array]:
    """``E[V_t | D_T] = E[1 / phi_t | D_T]`` at every step, and the degrees of freedom of a gamma
    matched to the mean and variance of ``phi_t | D_T``.

    Run backwards, ``phi_t = beta phi_(t+1) + gamma_t`` with an independent
    ``gamma_t ~ G((1 - beta) n_t / 2, n_t S_t / 2)``, the sampler's step. So ``phi_t`` is a sum of
    independent gammas, and ``E[1 / phi_t] = int_0^inf E[exp(-u phi_t)] du``, one quadrature a
    step. With ``beta = 1`` it is ``n_T S_T / (n_T - 2)`` at every step, and the gamma is exact.
    """
    beta = fit.model.variance_discount
    n, s = fit.dof, fit.scale
    horizon = n.shape[0]
    if beta == 1.0:
        return np.full(horizon, s[-1] * n[-1] / (n[-1] - 2.0)), np.full(horizon, n[-1])
    shape = np.append((1.0 - beta) * n[:-1], n[-1]) / 2.0
    rate = n * s / 2.0
    mean, var, factor = np.empty(horizon), np.empty(horizon), np.empty(horizon)
    mean[-1], var[-1] = 1.0 / s[-1], 2.0 / (n[-1] * s[-1] ** 2)
    for t in range(horizon - 2, -1, -1):
        mean[t] = beta * mean[t + 1] + (1.0 - beta) / s[t]
        var[t] = beta**2 * var[t + 1] + 2.0 * (1.0 - beta) / (n[t] * s[t] ** 2)
    for t in range(horizon):
        # in units of the step's own rate, so the integrand's scale does not follow the data's
        a = shape[t:]
        c = beta ** np.arange(horizon - t) * (rate[t] / rate[t:])
        factor[t] = (
            integrate.quad(
                lambda w, a=a, c=c: math.exp(-float(a @ np.log1p(w * c))),
                0.0,
                math.inf,
                epsabs=0.0,
                epsrel=1e-10,
                limit=200,
            )[0]
            * rate[t]
        )
    return factor, 2.0 * mean**2 / var


@dataclass(frozen=True)
class SmoothedStates:
    """``theta_t | D_T`` for every step: means, covariances, and ``Cov(theta_t, theta_(t+1))``.

    A covariance is the variance, not the scale. With a variance discount of 1 (``exact``) the
    marginals are Student ``t`` with ``dof`` degrees of freedom, normal when infinite. Below 1 the
    variance moves, the marginals are scale mixtures of normals that are not Student ``t``, and
    ``dof`` is that of a gamma matched to the smoothed precision's mean and variance, for an
    approximate ``t``; the means and covariances stay exact.
    """

    mean: _Array  # (T, p)
    covariance: _Array  # (T, p, p)
    cross_covariance: _Array  # (T - 1, p, p): Cov(theta_t, theta_(t+1) | D_T)
    dof: _Array  # (T,)
    exact: bool


def smooth(fit: DLMFit) -> SmoothedStates:
    """Retrospective moments of every state given all the data: Rauch-Tung-Striebel, with each
    step's covariance in the units ``E[V_t | D_T]`` of the variance the whole series supports.
    They are the moments of :func:`backward_sample`'s draws.

    Raises:
        ValueError: when the final degrees of freedom ``n_T`` is not above 2, so the covariances
            are infinite.
    """
    horizon, _ = fit.mean.shape
    gains = _gains(fit)
    if math.isinf(fit.model.prior.dof):
        expect_var = fit.scale.copy()  # known variance: E[V | D_T] = S
        dof = np.full(horizon, math.inf)
    else:
        if fit.dof[-1] <= 2.0:
            raise ValueError(
                f"the final degrees of freedom is {float(fit.dof[-1]):.3g}, not above 2, so the"
                " smoothed covariances are infinite; use backward_sample"
            )
        expect_var, dof = _variance_factor(fit)
    mean = np.empty_like(fit.mean)
    cov = np.empty_like(fit.covariance)
    cross = np.empty((max(horizon - 1, 0), *fit.covariance.shape[1:]))
    mean[-1] = fit.mean[-1]
    cov[-1] = expect_var[-1] * fit.covariance[-1] / fit.scale[-1]
    for t in range(horizon - 2, -1, -1):
        b = gains[t]
        mean[t] = fit.mean[t] + b @ (mean[t + 1] - fit.prior_mean[t + 1])
        residual = (fit.covariance[t] - b @ fit.prior_covariance[t + 1] @ b.T) / fit.scale[t]
        cov[t] = _sym(expect_var[t] * residual + b @ cov[t + 1] @ b.T)
        cross[t] = b @ cov[t + 1]
    return SmoothedStates(mean, cov, cross, dof, fit.model.variance_discount == 1.0)


@dataclass(frozen=True)
class PosteriorDraws:
    """Joint draws of the states and the observational variance given all the data."""

    states: _Array  # (draws, T, p)
    variance: _Array  # (draws, T)


def _sqrt_psd(m: _Array) -> _Array:
    w, v = np.linalg.eigh(_sym(m))
    return v * np.sqrt(np.clip(w, 0.0, None))


def backward_sample(fit: DLMFit, draws: int, seed: int) -> PosteriorDraws:
    """Forward filtering, backward sampling: joint draws of ``theta_(1:T)`` and ``V_(1:T)`` given
    ``D_T``, with the volatility path when ``variance_discount < 1`` (McAlinn and West 2019,
    appendix A.2).

    Raises:
        ValueError: on ``draws`` below 1.
    """
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError(f"draws must be a positive integer, got {draws!r}")
    rng = np.random.default_rng(seed)
    horizon, p = fit.mean.shape
    beta = fit.model.variance_discount
    known = math.isinf(fit.model.prior.dof)
    gains = _gains(fit)
    states = np.empty((draws, horizon, p))
    precision = np.empty((draws, horizon))

    def gamma(shape: float, rate: float) -> _Array:
        return rng.gamma(shape, 1.0 / rate, size=draws)

    if known:
        precision[:] = 1.0 / fit.scale[None, :]
    else:
        precision[:, -1] = gamma(fit.dof[-1] / 2.0, fit.dof[-1] * fit.scale[-1] / 2.0)
    root = _sqrt_psd(fit.covariance[-1] / fit.scale[-1])
    z = rng.standard_normal((draws, p))
    states[:, -1] = fit.mean[-1] + (z @ root.T) / np.sqrt(precision[:, -1:])
    for t in range(horizon - 2, -1, -1):
        if not known:
            fresh = (
                gamma((1.0 - beta) * fit.dof[t] / 2.0, fit.dof[t] * fit.scale[t] / 2.0)
                if beta < 1.0
                else 0.0
            )
            precision[:, t] = beta * precision[:, t + 1] + fresh
        b = gains[t]
        residual = (fit.covariance[t] - b @ fit.prior_covariance[t + 1] @ b.T) / fit.scale[t]
        centre = fit.mean[t] + (states[:, t + 1] - fit.prior_mean[t + 1]) @ b.T
        z = rng.standard_normal((draws, p))
        states[:, t] = centre + (z @ _sqrt_psd(residual).T) / np.sqrt(precision[:, t : t + 1])
    return PosteriorDraws(states, 1.0 / precision)


# ----------------------------------------------------------------------------------- forecasting


@dataclass(frozen=True)
class DLMForecast:
    """``k``-step-ahead marginal forecasts from the end of the data, ``k = 1, ..., horizon``.

    ``mean`` and ``scale`` are the Student ``t`` location and squared scale of ``y_(T+k)``, with
    ``dof`` degrees of freedom; ``state_mean`` and ``state_scale`` those of ``theta_(T+k)``.
    """

    mean: _Array  # (h,)
    scale: _Array  # (h,)
    dof: _Array  # (h,)
    state_mean: _Array  # (h, p)
    state_scale: _Array  # (h, p, p)


def forecast(
    fit: DLMFit,
    horizon: int,
    x: ArrayLike | None = None,
    *,
    evolution: ForecastEvolution = "constant",
) -> DLMForecast:
    """Forecast ``horizon`` steps past the data, with ``x`` the regression columns over them.

    ``evolution="constant"`` holds the first step's evolution variance ``W_(T+1)`` for every later
    step; ``"compounding"`` discounts again at every step. The variance's degrees of freedom decay
    by ``variance_discount`` a step and its estimate stays at ``S_T``.

    Under ``form="multiplicative"`` with blocks whose discounts differ, ``W = D P D - P`` is not a
    variance once the blocks are correlated enough, and holding it would drive the state variance
    indefinite; ``"constant"`` refuses such a ``W``.

    Raises:
        ValueError: on a ``horizon`` below 1, an ``x`` whose shape is not
            ``(horizon, model.regressors)``, an unknown ``evolution``, or, with
            ``evolution="constant"``, an evolution variance with a negative eigenvalue.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
    if evolution not in ("constant", "compounding"):
        raise ValueError(f"evolution must be 'constant' or 'compounding', got {evolution!r}")
    model = fit.model
    s = _structure(model)
    xs = _regressors(model, x, horizon)
    design = np.tile(s.design, (horizon, 1))
    design[:, s.regression] = xs
    active = design != 0.0
    g = s.evolution
    beta = model.variance_discount
    v = float(fit.scale[-1])
    n = float(fit.dof[-1])
    a = fit.mean[-1].copy()
    r = fit.covariance[-1].copy()
    p_next = g @ r @ g.T
    means, scales, dofs = np.empty(horizon), np.empty(horizon), np.empty(horizon)
    state_mean = np.empty((horizon, model.size))
    state_scale = np.empty((horizon, model.size, model.size))
    for k in range(horizon):
        a = g @ a
        if evolution == "compounding":
            r = _sym(_evolve(s, model.form, g @ r @ g.T, active[k]))
        else:
            w = _sym(_evolve(s, model.form, p_next, active[k]) - p_next)
            lowest = float(np.linalg.eigvalsh(w)[0])
            if lowest < -1e-12 * float(np.abs(p_next).max()):
                raise ValueError(
                    f"step {k + 1} ahead: the evolution variance W = D P D - P has an eigenvalue"
                    f" {lowest:.3g}; blocks with different discounts are correlated, so W is not"
                    " a variance. Use evolution='compounding' or form='additive'"
                )
            r = _sym(g @ r @ g.T + w)
        n = beta * n
        state_mean[k], state_scale[k] = a, r
        means[k] = design[k] @ a
        scales[k] = design[k] @ r @ design[k] + v
        dofs[k] = n
    return DLMForecast(means, scales, dofs, state_mean, state_scale)


# ------------------------------------------------------------------------------------ monitoring


def monitor_evalues(
    fit: DLMFit,
    *,
    shifts: Sequence[float] = (2.0,),
    inflations: Sequence[float] = (2.0,),
) -> _Array:
    """E-values for "the model's one-step forecasts hold", one row per observed step and one
    column per alternative, for :class:`chc.gate.DriftAlarm`.

    The standardised error ``u_t = e_t / sqrt(Q_t)`` is Student ``t`` under the model. Each
    alternative is a density for it -- the same ``t`` shifted by ``+h`` and by ``-h`` for each
    ``h`` in ``shifts``, and scaled by ``k`` for each ``k`` in ``inflations`` -- and its e-value is
    the density ratio, whose expectation under the model is 1. West and Harrison's (1986)
    cumulative Bayes-factor monitor is e-CUSUM over the same ratios, and its usual threshold
    ``tau = 0.135`` guarantees only that the average run length is at least ``1 / tau``, about
    7.4 steps; :class:`chc.gate.DriftAlarm` takes the average run length it guarantees as its
    threshold.

    Raises:
        ValueError: on no alternative, a shift that is not positive and finite, or an inflation
            that is not above 1 and finite.
    """
    hs = [float(h) for h in shifts]
    ks = [float(k) for k in inflations]
    if not hs and not ks:
        raise ValueError("name at least one shift or inflation")
    if any(not (math.isfinite(h) and h > 0.0) for h in hs):
        raise ValueError(f"shifts must be positive and finite, got {tuple(shifts)}")
    if any(not (math.isfinite(k) and k > 1.0) for k in ks):
        raise ValueError(f"inflations must be finite and above 1, got {tuple(inflations)}")
    observed = ~np.isnan(fit.y)
    u = fit.errors[observed] / np.sqrt(fit.one_step_scale[observed])
    dof = fit.one_step_dof[observed]
    columns: list[_Array] = []
    base = _log_std_t(u, dof)
    for h in hs:
        columns.append(np.exp(_log_std_t(u - h, dof) - base))
        columns.append(np.exp(_log_std_t(u + h, dof) - base))
    columns.extend(np.exp(_log_std_t(u / k, dof) - math.log(k) - base) for k in ks)
    return np.column_stack(columns)


def _log_std_t(u: _Array, dof: _Array) -> _Array:
    """The standard Student ``t`` log density, normal where ``dof`` is infinite, without the
    terms that cancel in a ratio at the same ``dof``."""
    out = np.empty_like(u)
    normal = np.isinf(dof)
    out[normal] = -0.5 * u[normal] ** 2
    nu = dof[~normal]
    out[~normal] = -(nu + 1.0) / 2.0 * np.log1p(u[~normal] ** 2 / nu)
    return out
