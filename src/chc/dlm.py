"""Discount dynamic linear models: forward filtering with a learned observational variance,
smoothing, backward sampling and the decomposition it gives, sampling with coefficients held to a
sign, forecasting and monitoring. *Experimental.*

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
* :func:`constrained_sample` draws the posterior truncated to coefficients' signs by Gibbs
  sampling, which is exact only in the limit of its chains; its R-hat and effective sample size
  say whether a run can be read.
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
  0.88-0.90. An interval reported from the likelihood's pick alone is too narrow; the union over
  :func:`confidence_set`'s fits covered 0.94 there.
* **A monitor has until the filter absorbs a change.** Once the discount has taken a shift into
  the level, the one-step errors are white again. On ``scripts/bench_dlm_monitor.py``'s local
  level at ``delta = 0.9``, 300 series, :class:`chc.gate.DriftAlarm` at an average run length of
  1000 caught a shift of two one-step deviations within 1000 steps in 74% of them, and one of
  three in all but one. West and Harrison's monitor at ``tau = 0.135`` caught each within a
  median of one step, and on unchanged series alarmed within a median of 13; DriftAlarm alarmed
  by step 100 on 2.7% of them, under its bound of 10%. Those series are the model's own, so the
  bound holds there by construction.
* **A known variance set too low reads as a change.** With every deviation 1.25 times what the
  known variance allowed for, the alarm sounded by step 100 on 57% of the series; with the
  variance learned from a prior of five degrees of freedom, on 4.3%.

:func:`monitor_evalues` turns the one-step errors into e-values for West and Harrison's (1986)
alternatives -- a shift of the location, an inflation of the scale -- which
:class:`chc.gate.DriftAlarm` turns into an alarm with a guaranteed average run length.
``forward_filter(interventions=...)`` applies an extra discount at named steps, their
feed-back intervention, and logs each (``chc_event="dlm_intervention"``).

**Over geos.** :class:`GeoDLM` stacks ``national`` blocks every geo reads and ``regional`` blocks
each geo has its own copy of; :func:`forward_filter_geos` filters a KPI a geo, ``y_(g,t) =
F_(g,t)' theta_t + nu_(g,t)`` with ``nu_(g,t) ~ N(0, V w_g)``, ``w_g`` the geo's known variance
relative to the others'. A geo's row of ``F`` reads the national blocks and its own regional ones,
and a national and a regional regression block read the geo's columns of ``x`` from the first, so a
geo's coefficient on a column both read is the national one plus its own deviation. The evolution
and the discounts are this module's for the stacked model; the update is West and Harrison's for a
vector with one variance scale (1997, section 16.4): with ``r_t`` geos observed,
``Q_t = F R_t F' + S_(t-1) W``, ``A_t = R_t F' Q_t^-1``, ``n_t = beta n_(t-1) + r_t`` and
``S_t = S_(t-1) (beta n_(t-1) + e' Q_t^-1 e) / n_t``, and the one-step forecast of the observed
geos is multivariate Student ``t``. That is the update one geo at a time with no evolution between
them, and its density the product of theirs (``validation/geo_dlm.mac``, steps 1 and 2).

* One geo is :func:`forward_filter`'s model of the national blocks and then the regional ones,
  and with every discount 1 the filter is the conjugate regression on the first state over every
  observation at once; the tests hold the first to ``1e-12`` and the second to ``1e-10``.
* The filter is dense: with ``p`` coordinates, the national ones and every geo's, a step costs
  ``O(p^3)`` and the fit holds ``2 T`` covariances of ``p^2`` entries. An observation of one geo
  adds nothing to the posterior precision between two others, and the multiplicative form, a
  diagonal scaling of the covariance, keeps that zero through the evolution while component
  discounting does not (step 3): a filter linear in the number of geos exists for the first form
  and is not built.
* :func:`smooth` and :func:`backward_sample` take a filter over geos and run back over its stacked
  state. They read the filter's moments, the evolution and the variance discount, and the
  variance's backward step reads ``n_t`` and ``S_t``, not how many geos a step observed. Over
  three geos with missing observations the smoother is the joint Gaussian posterior of every state
  to ``1e-8``, the variance known or learned. There is no forecast or decomposition over geos yet.
  One variance scale serves every geo, each geo's share of it fixed.
* :func:`fit_geo_spread` chooses the geos' spread, the regional prior's variance on named
  coordinates, by the marginal likelihood that the filter's log-likelihood is (type-II maximum
  likelihood), with each spread's likelihood-ratio interval. It draws the spread from its
  posterior under a prior flat on each standard deviation by importance sampling, so that
  :meth:`GeoSpread.mixture` carries a quantity's uncertainty about the spread as well as about the
  states. Where one spread is weakly identified the mixture is quadrature's within its Monte Carlo
  error, with the best inside the range, near its floor, and at its ceiling.

The implementation is NumPy in float64 and written from the published equations.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate, linalg, optimize, special

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


def _finite_update(t: int, scale: float, mean: _Array, covariance: _Array) -> None:
    """Refuse an update past the float range. The next step's one-step scale reads the covariance,
    which carries the scale, but never the mean, and after the last step nothing reads either."""
    if not (np.isfinite(mean).all() and np.isfinite(covariance).all()):
        raise FloatingPointError(f"step {t}: the update is not finite, its scale {scale}")


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
        FloatingPointError: on a step whose one-step squared scale is not positive and finite, or
            whose update leaves a scale or a state that is not finite.
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
            _finite_update(t, v_new, m, c)
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


class _Path(Protocol):
    """What smoothing and sampling read of a filter: the model whose evolution and variance
    discount they run back, and the filter's moments. Neither reads how many observations a step
    had, so a filter over geos is read through its stacked model."""

    @property
    def model(self) -> DynamicLinearModel: ...
    @property
    def prior_mean(self) -> _Array: ...
    @property
    def prior_covariance(self) -> _Array: ...
    @property
    def mean(self) -> _Array: ...
    @property
    def covariance(self) -> _Array: ...
    @property
    def dof(self) -> _Array: ...
    @property
    def scale(self) -> _Array: ...


@dataclass(frozen=True)
class _Stacked:
    """A filter over geos as smoothing and sampling read it."""

    model: DynamicLinearModel
    prior_mean: _Array
    prior_covariance: _Array
    mean: _Array
    covariance: _Array
    dof: _Array
    scale: _Array


def _path(fit: DLMFit | GeoDLMFit) -> _Path:
    if isinstance(fit, GeoDLMFit):
        return _Stacked(
            fit.model.stacked,
            fit.prior_mean,
            fit.prior_covariance,
            fit.mean,
            fit.covariance,
            fit.dof,
            fit.scale,
        )
    return fit


def _gains(fit: _Path) -> _Array:
    """``B_t = C_t G' R_(t+1)^(-1)`` for ``t < T``: scale-free, since ``C_t`` and ``R_(t+1)`` are
    both stated at ``S_t``."""
    g = _structure(fit.model).evolution
    horizon, p = fit.mean.shape
    out = np.empty((max(horizon - 1, 0), p, p))
    for t in range(horizon - 1):
        out[t] = np.linalg.solve(fit.prior_covariance[t + 1], g @ fit.covariance[t]).T
    return out


def _variance_factor(fit: _Path) -> tuple[_Array, _Array]:
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


def smooth(fit: DLMFit | GeoDLMFit) -> SmoothedStates:
    """Retrospective moments of every state given all the data: Rauch-Tung-Striebel, with each
    step's covariance in the units ``E[V_t | D_T]`` of the variance the whole series supports.
    They are the moments of :func:`backward_sample`'s draws. A filter over geos is smoothed over
    its stacked state.

    Raises:
        ValueError: when the final degrees of freedom ``n_T`` is not above 2, so the covariances
            are infinite.
    """
    path = _path(fit)
    horizon, _ = path.mean.shape
    gains = _gains(path)
    if math.isinf(path.model.prior.dof):
        expect_var = path.scale.copy()  # known variance: E[V | D_T] = S
        dof = np.full(horizon, math.inf)
    else:
        if path.dof[-1] <= 2.0:
            raise ValueError(
                f"the final degrees of freedom is {float(path.dof[-1]):.3g}, not above 2, so the"
                " smoothed covariances are infinite; use backward_sample"
            )
        expect_var, dof = _variance_factor(path)
    mean = np.empty_like(path.mean)
    cov = np.empty_like(path.covariance)
    cross = np.empty((max(horizon - 1, 0), *path.covariance.shape[1:]))
    mean[-1] = path.mean[-1]
    cov[-1] = expect_var[-1] * path.covariance[-1] / path.scale[-1]
    for t in range(horizon - 2, -1, -1):
        b = gains[t]
        mean[t] = path.mean[t] + b @ (mean[t + 1] - path.prior_mean[t + 1])
        residual = (path.covariance[t] - b @ path.prior_covariance[t + 1] @ b.T) / path.scale[t]
        cov[t] = _sym(expect_var[t] * residual + b @ cov[t + 1] @ b.T)
        cross[t] = b @ cov[t + 1]
    return SmoothedStates(mean, cov, cross, dof, path.model.variance_discount == 1.0)


@dataclass(frozen=True)
class PosteriorDraws:
    """Joint draws of the states and the observational variance given all the data."""

    states: _Array  # (draws, T, p)
    variance: _Array  # (draws, T)


def _sqrt_psd(m: _Array) -> _Array:
    w, v = np.linalg.eigh(_sym(m))
    return v * np.sqrt(np.clip(w, 0.0, None))


def backward_sample(fit: DLMFit | GeoDLMFit, draws: int, seed: int) -> PosteriorDraws:
    """Forward filtering, backward sampling: joint draws of ``theta_(1:T)`` and ``V_(1:T)`` given
    ``D_T``, with the volatility path when ``variance_discount < 1`` (McAlinn and West 2019,
    appendix A.2). A filter over geos is sampled over its stacked state.

    Raises:
        ValueError: on ``draws`` below 1.
    """
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError(f"draws must be a positive integer, got {draws!r}")
    path = _path(fit)
    rng = np.random.default_rng(seed)
    horizon, p = path.mean.shape
    beta = path.model.variance_discount
    known = math.isinf(path.model.prior.dof)
    gains = _gains(path)
    states = np.empty((draws, horizon, p))
    precision = np.empty((draws, horizon))

    def gamma(shape: float, rate: float) -> _Array:
        return rng.gamma(shape, 1.0 / rate, size=draws)

    if known:
        precision[:] = 1.0 / path.scale[None, :]
    else:
        precision[:, -1] = gamma(path.dof[-1] / 2.0, path.dof[-1] * path.scale[-1] / 2.0)
    root = _sqrt_psd(path.covariance[-1] / path.scale[-1])
    z = rng.standard_normal((draws, p))
    states[:, -1] = path.mean[-1] + (z @ root.T) / np.sqrt(precision[:, -1:])
    for t in range(horizon - 2, -1, -1):
        if not known:
            fresh = (
                gamma((1.0 - beta) * path.dof[t] / 2.0, path.dof[t] * path.scale[t] / 2.0)
                if beta < 1.0
                else 0.0
            )
            precision[:, t] = beta * precision[:, t + 1] + fresh
        b = gains[t]
        residual = (path.covariance[t] - b @ path.prior_covariance[t + 1] @ b.T) / path.scale[t]
        centre = path.mean[t] + (states[:, t + 1] - path.prior_mean[t + 1]) @ b.T
        z = rng.standard_normal((draws, p))
        states[:, t] = centre + (z @ _sqrt_psd(residual).T) / np.sqrt(precision[:, t : t + 1])
    return PosteriorDraws(states, 1.0 / precision)


def decompose(fit: DLMFit, draws: PosteriorDraws) -> _Array:
    """The mean ``F_t' theta_t`` split into its parts, draw by draw: ``(draws, T, parts)``.

    A :class:`Polynomial` or :class:`Seasonal` block is one part, its coordinates of ``F_t``
    times its coordinates of the state: the level, or the seasonal effect. A :class:`Regression`
    block is one part per column, the column times its coefficient. Parts follow the state's
    order, and a draw's parts add up to its ``F_t' theta_t`` to rounding.

    Each part's interval is a quantile over the draws. A window's total, a difference between
    two windows, or a return per unit of a column is taken draw by draw first, so its interval
    carries the parts' dependence across steps and on each other.

    Raises:
        ValueError: on draws of another model's or another series' states.
    """
    if draws.states.shape[1:] != fit.mean.shape:
        raise ValueError(
            f"draws of states shaped {draws.states.shape[1:]}, where the fit's are {fit.mean.shape}"
        )
    products = draws.states * fit.design
    parts: list[_Array] = []
    for block, sl in zip(fit.model.blocks, _structure(fit.model).slices, strict=True):
        if isinstance(block, Regression):
            parts.extend(products[:, :, j] for j in range(sl.start, sl.stop))
        else:
            parts.append(products[:, :, sl].sum(axis=2))
    return np.stack(parts, axis=2)


# ------------------------------------------------------------------------------ sign constraints

Sign = Literal["positive", "negative"]


@dataclass(frozen=True)
class ConstrainedDraws:
    """Draws of ``theta_(1:T)`` and ``V`` given ``D_T`` and the signs, from Gibbs chains after
    their warm-up. ``rhat`` is the largest rank-normalised split R-hat, bulk or tail, over every
    state at every step and the variance (Vehtari et al. 2021); ``ess`` the smallest bulk
    effective sample size over the same."""

    states: _Array  # (chains, draws, T, p)
    variance: _Array  # (chains, draws)
    rhat: float
    ess: float

    @property
    def mixed(self) -> bool:
        """Whether the draws can be read: ``rhat`` at most 1.01 and ``ess`` at least 100 a chain
        (Vehtari et al. 2021)."""
        return self.rhat <= 1.01 and self.ess >= 100.0 * self.states.shape[0]

    @property
    def pooled(self) -> PosteriorDraws:
        """The chains pooled into one set of draws, as :func:`decompose` reads them."""
        chains, draws, horizon, p = self.states.shape
        variance = np.repeat(self.variance.reshape(-1, 1), horizon, axis=1)
        return PosteriorDraws(self.states.reshape(chains * draws, horizon, p), variance)


def _rank_normal(x: _Array) -> _Array:
    """Rank-normalised draws, pooled over chains: ``(chains, draws, q)``."""
    chains, draws, q = x.shape
    flat = x.reshape(chains * draws, q)
    ranks = flat.argsort(axis=0).argsort(axis=0) + 1.0
    return special.ndtri((ranks - 0.375) / (chains * draws + 0.25)).reshape(chains, draws, q)


def _split(x: _Array) -> _Array:
    half = x.shape[1] // 2
    return np.concatenate([x[:, :half], x[:, x.shape[1] - half :]], axis=0)


def _rhat_of(z: _Array) -> _Array:
    _, n, _ = z.shape
    means = z.mean(axis=1)
    within = z.var(axis=1, ddof=1).mean(axis=0)
    between = n * means.var(axis=0, ddof=1)
    return np.sqrt(((n - 1.0) / n * within + between / n) / within)


def _ess_of(z: _Array) -> _Array:
    """Bulk effective sample size of split, rank-normalised chains: Geyer's initial monotone
    sequence over the chains' combined autocorrelation."""
    m, n, q = z.shape
    centred = z - z.mean(axis=1, keepdims=True)
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centred, n=size, axis=1)
    acov = np.fft.irfft(spectrum * np.conj(spectrum), n=size, axis=1)[:, :n] / n
    within = z.var(axis=1, ddof=1).mean(axis=0)
    between = n * z.mean(axis=1).var(axis=0, ddof=1)
    var_plus = (n - 1.0) / n * within + between / n
    rho = 1.0 - (within - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    pairs = rho[: 2 * (n // 2)].reshape(n // 2, 2, q).sum(axis=1)
    positive = np.cumprod(pairs > 0.0, axis=0).astype(bool)
    monotone = np.minimum.accumulate(np.where(positive, pairs, 0.0), axis=0)
    tau = -1.0 + 2.0 * monotone.sum(axis=0)
    return m * n / np.maximum(tau, 1.0 / np.log10(m * n))


def _convergence(states: _Array, variance: _Array) -> tuple[float, float]:
    chains, draws, horizon, p = states.shape
    x = np.concatenate([states.reshape(chains, draws, horizon * p), variance[:, :, None]], axis=2)
    x = x[:, :, np.ptp(x.reshape(-1, x.shape[2]), axis=0) > 0.0]
    folded = np.abs(x - np.median(x.reshape(-1, x.shape[2]), axis=0))
    bulk = _split(_rank_normal(x))
    rhat = np.maximum(_rhat_of(bulk), _rhat_of(_split(_rank_normal(folded))))
    return float(rhat.max()), float(_ess_of(bulk).min())


def _transitions(fit: DLMFit) -> tuple[_Array, _Array]:
    """``R*_1``, the first prior's covariance, and ``W*_t`` for ``t = 2, ..., T``, both in units
    of ``V``: ``R_t = G C_(t-1) G' + W_t`` with ``W_t`` the discount's blockwise inflation, zero
    exactly where a block's discount is 1 or a held coordinate is idle.

    Raises:
        ValueError: on a fit whose ``R_t`` is not the discount's, such as one with interventions.
    """
    s = _structure(fit.model)
    horizon, p = fit.mean.shape
    active = fit.design != 0.0
    w = np.zeros((max(horizon - 1, 0), p, p))
    for t in range(1, horizon):
        evolved = s.evolution @ fit.covariance[t - 1] @ s.evolution.T
        w[t - 1] = _evolve(s, "additive", evolved, active[t]) - evolved
        scale_of = float(np.abs(fit.prior_covariance[t]).max())
        if not np.allclose(
            evolved + w[t - 1], fit.prior_covariance[t], rtol=0.0, atol=1e-9 * scale_of
        ):
            raise ValueError(
                f"step {t}: the prior covariance is not the discount's evolution of the last"
                " posterior; constrained sampling does not take a fit with interventions"
            )
        w[t - 1] /= fit.scale[t - 1]
    return fit.prior_covariance[0] / fit.model.prior.scale, w


def _unit_filter(
    g: _Array, design: _Array, observed: _Array, first_cov: _Array, w: _Array
) -> tuple[_Array, _Array, _Array, _Array, _Array]:
    """The filter at ``V = 1``, whose covariances are the posterior's given ``V`` in units of
    ``V`` and do not depend on ``y``: the gains ``(T, p)``, zero where ``y_t`` is missing; the
    priors' and posteriors' covariances ``(T, p, p)``; the smoother's gains ``B_t``
    ``(T - 1, p, p)``; and roots of the backward draw's covariances ``(T, p, p)``."""
    horizon, p = design.shape
    gains = np.zeros((horizon, p))
    covs = np.empty((horizon, p, p))
    priors = np.empty((horizon, p, p))
    r = first_cov
    for t in range(horizon):
        if t:
            r = _sym(g @ covs[t - 1] @ g.T + w[t - 1])
        priors[t] = r
        if observed[t]:
            rf = r @ design[t]
            gains[t] = rf / (design[t] @ rf + 1.0)
            shrink = np.eye(p) - np.outer(gains[t], design[t])
            covs[t] = _sym(shrink @ r @ shrink.T + np.outer(gains[t], gains[t]))
        else:
            covs[t] = r
    back = np.zeros((max(horizon - 1, 0), p, p))
    roots = np.zeros((horizon, p, p))
    roots[-1] = _sqrt_psd(covs[-1])
    for t in range(horizon - 1):
        back[t] = np.linalg.solve(priors[t + 1], g @ covs[t]).T
        roots[t] = _sqrt_psd(covs[t] - back[t] @ priors[t + 1] @ back[t].T)
    return gains, priors, covs, back, roots


def _path_covariance(back: _Array, smoothed: _Array, cs: list[int], times: _Array) -> _Array:
    """``Cov(theta_s[cs], theta_t[cs])`` for ``s`` and ``t`` in the sorted ``times``, shaped
    ``(times, cs, times, cs)``, from the smoothed covariances and
    ``Cov(theta_s, theta_t) = B_s Cov(theta_(s+1), theta_t)`` for ``s < t``."""
    n, k, p = times.size, len(cs), smoothed.shape[1]
    out = np.empty((n, k, n, k))
    column = np.empty((n, p, k))  # Cov(theta_s, theta_t[cs]) for the times t from i on
    i = n
    for s in range(int(times[-1]), -1, -1):
        if i < n:
            column[i:] = back[s] @ column[i:]
        if i and times[i - 1] == s:
            i -= 1
            column[i] = smoothed[s][:, cs]
            block = column[i:, cs, :]
            out[i, :, i:, :] = block.transpose(1, 0, 2)
            out[i:, :, i, :] = block.transpose(0, 2, 1)
    return out


_MAX_BOUNCES = 1_000


def _reflected_orbit(
    mean: _Array, offset: _Array, velocity: _Array, sign: _Array, cov: _Array
) -> tuple[_Array, int]:
    """One exact Hamiltonian trajectory of length ``pi / 2`` for ``N(mean, cov)`` truncated to
    ``sign * x >= 0`` (Pakman and Paninski 2014), a chain to a row: from ``mean + offset`` the
    orbit is ``mean + a cos(t) + b sin(t)``, with ``a = offset`` and ``b = velocity``, until it
    meets a wall, where the velocity reflects in the metric of ``cov``. Returns the final offsets
    and how many walls were met. With no wall in the way the end is ``mean + velocity``, a draw
    independent of the start.

    Raises:
        RuntimeError: on a trajectory that meets more than ``_MAX_BOUNCES`` walls per coordinate.
    """
    a, b = offset.copy(), velocity.copy()
    left = np.full(a.shape[0], 0.5 * math.pi)
    diagonal = np.diag(cov)
    moving = np.arange(a.shape[0])
    bounces = 0
    while moving.size:
        am, bm = a[moving], b[moving]
        radius = np.hypot(am, bm)
        reaches = radius > np.abs(mean)
        cross = np.arccos(np.clip(-mean / np.where(reaches, radius, 1.0), -1.0, 1.0))
        time = np.mod(np.arctan2(bm, am) + sign * cross, 2.0 * math.pi)
        # a coordinate on its wall, or past it by rounding, and heading out leaves now
        time[(sign * (mean + am) <= 0.0) & (sign * bm < 0.0)] = 0.0
        leaving = sign * (bm * np.cos(time) - am * np.sin(time)) < 0.0
        time = np.where(reaches & leaving, time, np.inf)
        wall = time.argmin(axis=1)
        hit = time[np.arange(moving.size), wall]
        ends = hit >= left[moving]
        done = moving[ends]
        rest = left[done][:, None]
        a[done] = a[done] * np.cos(rest) + b[done] * np.sin(rest)
        moving, hit, wall = moving[~ends], hit[~ends, None], wall[~ends]
        if not moving.size:
            break
        bounces += moving.size
        if bounces > _MAX_BOUNCES * mean.size * a.shape[0]:
            raise RuntimeError(
                f"an exact Hamiltonian trajectory met {bounces} walls over {a.shape[0]} chains"
                f" and {mean.size} constrained values, more than {_MAX_BOUNCES} a value"
            )
        position = a[moving] * np.cos(hit) + b[moving] * np.sin(hit)
        speed = b[moving] * np.cos(hit) - a[moving] * np.sin(hit)
        rows = np.arange(moving.size)
        position[rows, wall] = -mean[wall]
        speed -= (2.0 * speed[rows, wall] / diagonal[wall])[:, None] * cov[wall]
        a[moving], b[moving] = position, speed
        left[moving] -= hit[:, 0]
    return a, bounces


def constrained_sample(
    fit: DLMFit,
    signs: Mapping[int, Sign],
    draws: int,
    seed: int,
    *,
    chains: int = 4,
    warmup: int | None = None,
) -> ConstrainedDraws:
    """Draws of the states and the variance given the data with the coefficients of some columns
    of ``x`` held to a sign at every step: the model's posterior truncated to the signs, so the
    point estimate is the truncated posterior's mean, not a clipped one, and a channel's pull
    against its sign moves into the other states. ``signs`` maps a column of ``x`` to
    ``"positive"`` (at least 0) or ``"negative"`` (at most 0).

    The discounts define the evolution variances ``W_t`` through the unconstrained filter, which
    depend on the design and not on ``y``; with them the model is a Gaussian prior over paths,
    and the signs condition it. Gibbs sampling over three blocks, each chain started from an
    unconstrained backward draw moved into the signs (ADR 0042):

    * the constrained coefficients' values with the other states integrated out, a Gaussian given
      ``V`` truncated to the signs, by one exact Hamiltonian trajectory of length ``pi / 2``
      (Pakman and Paninski 2014): the orbit is an ellipse, reflected where it meets a wall, with
      no step size and no rejection. The steps a zero evolution variance ties together (a discount
      of 1, or an idle held regressor) are one value;
    * the other states given those paths, by forward filtering and backward sampling;
    * ``V`` from its inverse gamma given the constrained values alone, the other states
      integrated out as well. Given every state, each evolution innovation would add a degree of
      freedom, which pins ``V`` to the path and slows its chain.

    Truncating each backward-sampling step instead would not give this posterior, since the filter
    that feeds it never saw the signs. The values' covariance is held dense and factored once, so
    memory grows as the square of their number, and a trajectory meets more walls the harder the
    signs bind.

    A constrained coefficient must be a :class:`Regression` block of width one. ``rhat`` and
    ``ess`` say whether the chains mixed; the draws are not to be read when they are not
    :attr:`ConstrainedDraws.mixed`, and the run is then logged as a warning.

    Raises:
        ValueError: on no signs, a column that is not one of ``x``'s or not in a width-one
            regression block, a sign that is neither, ``form="multiplicative"`` (whose evolution
            couples the blocks), a variance discount below 1, a fit with interventions, ``draws``
            below 1, ``chains`` below 2 or a negative ``warmup``.
    """
    model = fit.model
    if not signs:
        raise ValueError("no sign to impose; backward_sample draws the unconstrained posterior")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError(f"draws must be a positive integer, got {draws!r}")
    if isinstance(chains, bool) or not isinstance(chains, int) or chains < 2:
        raise ValueError(f"chains must be an integer of at least 2 for R-hat, got {chains!r}")
    warmup = draws if warmup is None else warmup
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError(f"warmup must be a non-negative integer, got {warmup!r}")
    if model.form != "additive":
        raise ValueError(
            "constrained sampling needs form='additive', whose evolution keeps the blocks apart"
        )
    if model.variance_discount != 1.0:
        raise ValueError("constrained sampling needs a variance discount of 1")
    s = _structure(model)
    block_of = {}
    for block, sl in zip(model.blocks, s.slices, strict=True):
        for j in range(sl.start, sl.stop):
            block_of[j] = block
    constrained: dict[int, float] = {}
    for column, sign in signs.items():
        if isinstance(column, bool) or not isinstance(column, int):
            raise ValueError(f"a sign's key is a column of x, got {column!r}")
        if not 0 <= column < s.regression.size:
            raise ValueError(f"column {column} is not one of x's {s.regression.size}")
        if sign not in ("positive", "negative"):
            raise ValueError(f"a sign is 'positive' or 'negative', got {sign!r}")
        j = int(s.regression[column])
        if block_of[j].size != 1:
            raise ValueError(
                f"column {column}'s coefficient shares a regression block of width"
                f" {block_of[j].size}; give a constrained coefficient a block of its own"
            )
        constrained[j] = 1.0 if sign == "positive" else -1.0

    rng = np.random.default_rng(seed)
    horizon, p = fit.mean.shape
    design, y = fit.design, fit.y
    observed = ~np.isnan(y)
    y0 = np.where(observed, y, 0.0)
    g = s.evolution
    first_cov, w = _transitions(fit)
    first_mean = fit.prior_mean[0]
    learned = not math.isinf(model.prior.dof)
    cs = sorted(constrained)
    us = [j for j in range(p) if j not in constrained]

    # the constrained coefficients with the other states integrated out, in units of V: a
    # Gaussian over each one's distinct values, one per run of steps that a zero evolution
    # variance ties together (a discount of 1, or an idle held regressor)
    full_gain, full_prior, full_cov, full_back, full_root = _unit_filter(
        g, design, observed, first_cov, w
    )
    filtered = np.empty((horizon, p))
    a = first_mean
    for t in range(horizon):
        if t:
            a = g @ filtered[t - 1]
        filtered[t] = a + full_gain[t] * (y0[t] - design[t] @ a)
    smoothed, smoothed_cov = filtered.copy(), full_cov.copy()
    for t in range(horizon - 2, -1, -1):
        smoothed[t] += full_back[t] @ (smoothed[t + 1] - g @ filtered[t])
        smoothed_cov[t] = _sym(
            full_cov[t] + full_back[t] @ (smoothed_cov[t + 1] - full_prior[t + 1]) @ full_back[t].T
        )
    starts = [np.concatenate([[0], 1 + np.flatnonzero(w[:, j, j] > 0.0)]) for j in cs]
    times = np.unique(np.concatenate(starts))
    step = np.concatenate(starts)
    which = np.repeat(np.arange(len(cs)), [run.size for run in starts])
    columns = np.asarray(cs)[which]
    spot = np.searchsorted(times, step)
    value_cov = _path_covariance(full_back, smoothed_cov, cs, times)[
        spot[:, None], which[:, None], spot, which
    ]
    value_mean = smoothed[step, columns]
    value_sign = np.array([constrained[j] for j in columns])
    first_value = np.cumsum([0] + [run.size for run in starts[:-1]])
    fill = np.stack(
        [
            first_value[c] + np.searchsorted(run, np.arange(horizon), side="right") - 1
            for c, run in enumerate(starts)
        ],
        axis=1,
    )

    # the other states given the constrained paths: a DLM with known offsets
    gu, fu = g[np.ix_(us, us)], design[:, us]
    shift = first_cov[np.ix_(us, cs)] @ np.linalg.inv(first_cov[np.ix_(cs, cs)])
    pu = len(us)
    if pu:
        r_first = first_cov[np.ix_(us, us)] - shift @ first_cov[np.ix_(cs, us)]
        gains, _, _, back, roots = _unit_filter(gu, fu, observed, r_first, w[:, us][:, :, us])

    # the variance given the constrained values, the other states integrated out: V given y is
    # IG(n_T / 2, n_T S_T / 2) and the values given V are N(value_mean, V value_cov)
    value_root = np.linalg.cholesky(value_cov)
    shape = (fit.dof[-1] + value_mean.size) / 2.0 if learned else math.inf

    start = backward_sample(fit, chains, seed=int(rng.integers(2**31)))
    theta = start.states.copy()
    theta[:, :, cs] = (value_sign * np.maximum(value_sign * theta[:, step, columns], 0.0))[:, fill]
    variance = start.variance[:, -1] if learned else np.full(chains, model.prior.scale)

    kept_states = np.empty((chains, draws, horizon, p))
    kept_variance = np.empty((chains, draws))
    bounces = 0
    for sweep in range(warmup + draws):
        if learned:
            white = linalg.solve_triangular(
                value_root, (theta[:, step, columns] - value_mean).T, lower=True
            )
            rate = 0.5 * (fit.dof[-1] * fit.scale[-1] + (white**2).sum(axis=0))
            variance = 1.0 / rng.gamma(shape, 1.0 / rate)
        sd = np.sqrt(variance)[:, None]

        # the velocity is a centred draw of the values' Gaussian: a backward draw of every state
        noise = rng.standard_normal((chains, horizon, p))
        path = np.empty((chains, horizon, p))
        path[:, -1] = noise[:, -1] @ full_root[-1].T
        for t in range(horizon - 2, -1, -1):
            path[:, t] = path[:, t + 1] @ full_back[t].T + noise[:, t] @ full_root[t].T
        moved, met = _reflected_orbit(
            value_mean,
            theta[:, step, columns] - value_mean,
            sd * path[:, step, columns],
            value_sign,
            value_cov,
        )
        bounces += met
        values = value_sign * np.maximum(value_sign * (value_mean + moved), 0.0)
        theta[:, :, cs] = values[:, fill]

        if pu:
            offsets = y0 - np.einsum("tp,ctp->ct", design[:, cs], theta[:, :, cs])
            a = first_mean[us] + (theta[:, 0, cs] - first_mean[cs]) @ shift.T
            means = np.empty((chains, horizon, pu))
            ahead = np.empty((chains, horizon, pu))
            for t in range(horizon):
                if t:
                    a = means[:, t - 1] @ gu.T
                ahead[:, t] = a
                if observed[t]:
                    a = a + (offsets[:, t] - a @ fu[t])[:, None] * gains[t]
                means[:, t] = a
            noise = rng.standard_normal((chains, horizon, pu))
            u = means[:, -1] + sd * (noise[:, -1] @ roots[-1].T)
            theta[:, -1, us] = u
            for t in range(horizon - 2, -1, -1):
                u = (
                    means[:, t]
                    + (u - ahead[:, t + 1]) @ back[t].T
                    + sd * (noise[:, t] @ roots[t].T)
                )
                theta[:, t, us] = u

        if sweep >= warmup:
            kept_states[:, sweep - warmup] = theta
            kept_variance[:, sweep - warmup] = variance

    result = ConstrainedDraws(kept_states, kept_variance, *_convergence(kept_states, kept_variance))
    rhat, ess, mixed = result.rhat, result.ess, result.mixed
    walls = bounces / (chains * (warmup + draws))
    _log.log(
        logging.INFO if mixed else logging.WARNING,
        "dlm constrained sample: %d chains of %d draws, R-hat %.4f, bulk ESS %.0f,"
        " %.1f walls a trajectory%s",
        chains,
        draws,
        rhat,
        ess,
        walls,
        "" if mixed else "; the chains have not mixed and the draws are not to be read",
        extra={
            "chc_event": "dlm_constrained_sample",
            "rhat": rhat,
            "ess": ess,
            "chains": chains,
            "draws": draws,
            "walls": walls,
            "mixed": mixed,
        },
    )
    return result


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


# -------------------------------------------------------------------------------- choosing fits


def confidence_set(fits: Sequence[DLMFit], parameters: int, level: float = 0.95) -> tuple[int, ...]:
    """The fits a likelihood-ratio test at ``level`` does not reject, in the order given: those
    whose :attr:`DLMFit.log_likelihood` is within ``chi2_parameters(level) / 2`` of the best.
    ``parameters`` is how many settings the fits vary over, two for a grid over two discounts.

    An interval reported as the union of each member's interval, from its lowest lower end to its
    highest upper end, carries the uncertainty of the discounts as well as the states'
    (projection; Berger and Boos 1994). The best fit's interval alone does not, and a mixture
    weighted by the likelihood carries too little of it: the likelihood is flat where an
    evolution variance is weakly identified, and its best point there is a selection.

    The union over-covers. On ``scripts/bench_dlm.py``'s 500 series a channel's 13-week
    contribution was covered 0.94 at a nominal 0.90, both on a random walk, where the pick
    covered 0.78, and on the discount model's own series, where it covered 0.85. The intervals
    there were 1.2 times as wide as at the true discounts.

    Raises:
        ValueError: on no fits, fits of other observations than the first's, a ``parameters``
            that is not a positive integer, or a ``level`` outside ``(0, 1)``.
    """
    if not fits:
        raise ValueError("confidence_set needs at least one fit")
    if isinstance(parameters, bool) or not isinstance(parameters, int) or parameters < 1:
        raise ValueError(f"parameters must be a positive integer, got {parameters!r}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    for i, fit in enumerate(fits[1:], start=1):
        if not np.array_equal(fit.y, fits[0].y, equal_nan=True):
            raise ValueError(
                f"fit {i} filtered other observations than fit 0; a likelihood ratio compares"
                " fits of the same data"
            )
    loglik = np.array([fit.log_likelihood for fit in fits])
    margin = float(special.chdtri(parameters, 1.0 - level)) / 2.0
    members = tuple(int(i) for i in np.flatnonzero(loglik >= loglik.max() - margin))
    _log.info(
        "dlm confidence set: %d of %d fits within %.3g of the best log-likelihood",
        len(members),
        len(fits),
        margin,
        extra={
            "chc_event": "dlm_confidence_set",
            "members": len(members),
            "fits": len(fits),
            "margin": margin,
        },
    )
    return members


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


# ------------------------------------------------------------------------------------------ geos


@dataclass(frozen=True)
class GeoDLM:
    """A discount DLM over ``geos`` geos: ``national`` blocks every geo reads and ``regional``
    blocks each geo has its own copy of, filtered by :func:`forward_filter_geos`.

    ``prior`` is on the stacked state: the national blocks' coordinates, then each geo's copy of
    the regional blocks', in the geos' order; :func:`stacked_prior` builds one whose geos' regional
    priors are alike and independent. ``relative_variance`` is each geo's observational variance
    relative to the others', 1 for every geo when omitted.

    A national and a regional regression block read each geo's columns of ``x`` from the first,
    so where both read a column, a geo's coefficient on it is the national coefficient plus the
    geo's own deviation. The regional blocks' prior variance is then the spread of the geos about
    the national coefficient, and the regional discount how long that prior is remembered.

    Raises:
        ValueError: on a ``geos`` that is not a positive integer, a ``relative_variance`` that is
            not one positive finite number a geo, and what :class:`DynamicLinearModel` refuses of
            the stacked model: no blocks, a prior of the wrong size, an unknown ``form``, or a
            ``variance_discount`` it does not take.
    """

    national: tuple[Block, ...]
    regional: tuple[Block, ...]
    geos: int
    prior: Prior
    form: DiscountForm = "additive"
    variance_discount: float = 1.0
    relative_variance: ArrayLike | None = None

    def __post_init__(self) -> None:
        if isinstance(self.geos, bool) or not isinstance(self.geos, int) or self.geos < 1:
            raise ValueError(f"geos must be a positive integer, got {self.geos!r}")
        weights = (
            np.ones(self.geos)
            if self.relative_variance is None
            else np.array(self.relative_variance, dtype=np.float64)
        )
        if weights.shape != (self.geos,) or not np.all(np.isfinite(weights) & (weights > 0.0)):
            raise ValueError(
                f"relative_variance must be {self.geos} positive finite numbers, one a geo, got "
                f"{self.relative_variance!r}"
            )
        weights.setflags(write=False)
        object.__setattr__(self, "national", tuple(self.national))
        object.__setattr__(self, "regional", tuple(self.regional))
        object.__setattr__(self, "relative_variance", weights)
        self.stacked  # noqa: B018 -- the stacked model validates the blocks and the prior

    @property
    def stacked(self) -> DynamicLinearModel:
        """The stacked state's model: the national blocks, then the regional ones once a geo."""
        return DynamicLinearModel(
            self.national + self.regional * self.geos,
            self.prior,
            self.form,
            self.variance_discount,
        )

    @property
    def regressors(self) -> int:
        """How many columns each geo's ``x`` must have: the national regression blocks' widths
        summed, or the regional ones', whichever is more."""
        return max(_widths(self.national), _widths(self.regional))


def _widths(blocks: tuple[Block, ...]) -> int:
    return sum(b.width for b in blocks if isinstance(b, Regression))


def stacked_prior(national: Prior | None, regional: Prior | None, geos: int) -> Prior:
    """The stacked state's prior: ``national`` on the national blocks and ``regional`` on each
    geo's copy of the regional blocks, every block independent of the others.

    ``None`` stands for no blocks of that kind.

    Raises:
        ValueError: on two priors whose ``scale`` or ``dof`` differ, both ``None``, or a ``geos``
            that is not a positive integer.
    """
    if isinstance(geos, bool) or not isinstance(geos, int) or geos < 1:
        raise ValueError(f"geos must be a positive integer, got {geos!r}")
    if regional is None and national is None:
        raise ValueError("a stacked prior needs a national prior, a regional one, or both")
    parts = [] if national is None else [national]
    parts += [] if regional is None else [regional] * geos
    first = parts[0]
    if any(p.scale != first.scale or p.dof != first.dof for p in parts):
        raise ValueError(
            "the national and the regional prior must state one variance: their scale and dof "
            f"differ ({national.scale if national else None}, {national.dof if national else None}"
            f" against {regional.scale if regional else None}, "
            f"{regional.dof if regional else None})"
        )
    return Prior(
        np.concatenate([p.mean for p in parts]),
        linalg.block_diag(*[p.covariance for p in parts]),
        first.scale,
        first.dof,
    )


@dataclass(frozen=True)
class GeoDLMFit:
    """:func:`forward_filter_geos`'s output over the stacked state, one row per step, in
    :class:`DLMFit`'s notation.

    ``one_step_mean`` and ``one_step_scale`` are every geo's, observed or not: location ``F a`` and
    squared scale ``F R F' + S_(t-1) W``, ``W`` the relative variances on the diagonal.
    ``log_scores`` is the observed geos' joint one-step log density, ``NaN`` on a step with none
    observed. ``n`` rises by the number of geos observed at a step.
    """

    model: GeoDLM
    y: _Array  # (T, geos)
    design: _Array  # F_t, (T, geos, p)
    prior_mean: _Array  # a_t, (T, p)
    prior_covariance: _Array  # R_t, (T, p, p)
    one_step_mean: _Array  # f_t, (T, geos)
    one_step_scale: _Array  # Q_t, (T, geos, geos)
    one_step_dof: _Array  # beta n_(t-1), (T,)
    mean: _Array  # m_t, (T, p)
    covariance: _Array  # C_t, (T, p, p)
    dof: _Array  # n_t, (T,)
    scale: _Array  # S_t, (T,)
    log_scores: _Array  # (T,)

    @property
    def log_likelihood(self) -> float:
        """The sum of the one-step forecasts' joint log densities over the steps with a geo
        observed."""
        return float(np.nansum(self.log_scores))

    @property
    def errors(self) -> _Array:
        """One-step errors ``y - f``, ``(T, geos)``, ``NaN`` where a geo is missing."""
        return self.y - self.one_step_mean


def forward_filter_geos(model: GeoDLM, y: ArrayLike, x: ArrayLike | None = None) -> GeoDLMFit:
    """Filter ``y``, ``(T, geos)``, through ``model``, with ``x``, ``(T, geos, model.regressors)``,
    each geo's columns for the regression blocks. A missing observation is ``NaN``.

    Raises:
        ValueError: on a ``y`` that is not ``(T, geos)`` of finite values and ``NaN``, or an ``x``
            whose shape is not ``(T, geos, model.regressors)`` or that is not finite.
        FloatingPointError: on a step whose observed geos' one-step squared scale is not positive
            definite and finite, or whose update leaves a scale or a state that is not finite.
    """
    ys = np.array(y, dtype=np.float64)
    if ys.ndim != 2 or ys.shape[0] == 0 or ys.shape[1] != model.geos:
        raise ValueError(f"y must be (T, {model.geos}), a column a geo, got shape {ys.shape}")
    if np.any(np.isinf(ys)):
        raise ValueError("y has an infinite value; a missing observation is NaN")
    horizon, geos = ys.shape
    stacked = model.stacked
    s = _structure(stacked)
    design = _geo_design(model, s, x, horizon)
    active = np.any(design != 0.0, axis=1)
    weights = np.asarray(model.relative_variance, dtype=np.float64)
    p = stacked.size
    beta = stacked.variance_discount

    a_all = np.empty((horizon, p))
    r_all = np.empty((horizon, p, p))
    m_all = np.empty((horizon, p))
    c_all = np.empty((horizon, p, p))
    f_all = np.empty((horizon, geos))
    q_all = np.empty((horizon, geos, geos))
    nu_all = np.empty(horizon)
    n_all = np.empty(horizon)
    s_all = np.empty(horizon)
    score = np.full(horizon, np.nan)

    m = stacked.prior.mean.copy()
    c = stacked.prior.covariance.copy()
    n, v = stacked.prior.dof, stacked.prior.scale
    eye = np.eye(p)
    for t in range(horizon):
        a = s.evolution @ m
        r = _sym(_evolve(s, stacked.form, s.evolution @ c @ s.evolution.T, active[t]))
        rows = design[t]
        f = rows @ a
        q = _sym(rows @ r @ rows.T) + v * np.diag(weights)
        nu = beta * n
        a_all[t], r_all[t], f_all[t], q_all[t], nu_all[t] = a, r, f, q, nu
        seen = ~np.isnan(ys[t])
        k = int(seen.sum())
        if k == 0:
            m, c, n = a, r, beta * n
        else:
            observed = rows[seen]
            q_seen = q[np.ix_(seen, seen)]
            if not np.all(np.isfinite(q_seen)):
                raise FloatingPointError(f"step {t}: the one-step squared scale is not finite")
            try:
                root = linalg.cho_factor(q_seen, lower=True)
            except np.linalg.LinAlgError:
                raise FloatingPointError(
                    f"step {t}: the one-step squared scale is not positive definite"
                ) from None
            e = ys[t, seen] - f[seen]
            gain = linalg.cho_solve(root, observed @ r).T
            quad = float(e @ linalg.cho_solve(root, e))
            n_new = beta * n + k
            v_new = v if math.isinf(n_new) else v * (beta * n + quad) / n_new
            shrink = eye - gain @ observed
            c = _sym((v_new / v) * (shrink @ r @ shrink.T + (gain * (v * weights[seen])) @ gain.T))
            m = a + gain @ e
            _finite_update(t, v_new, m, c)
            log_det = 2.0 * float(np.sum(np.log(np.diag(root[0]))))
            score[t] = _log_multivariate_t(quad, log_det, k, nu)
            n, v = n_new, v_new
        m_all[t], c_all[t], n_all[t], s_all[t] = m, c, n, v

    _warn_windup(stacked, s, active.astype(np.float64))
    return GeoDLMFit(
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


def _geo_design(model: GeoDLM, s: _Structure, x: ArrayLike | None, horizon: int) -> _Array:
    """Each geo's row of ``F`` at each step, ``(T, geos, p)``: the blocks' fixed parts on the
    national coordinates and the geo's own, and its columns of ``x`` on their regressions."""
    geos, k = model.geos, model.regressors
    if x is None:
        if k:
            raise ValueError(f"the model's regression blocks need x with {k} columns a geo")
        xs = np.zeros((horizon, geos, 0))
    else:
        xs = np.array(x, dtype=np.float64)
        if xs.shape != (horizon, geos, k):
            raise ValueError(f"x must have shape ({horizon}, {geos}, {k}), got {xs.shape}")
        if not np.all(np.isfinite(xs)):
            raise ValueError("x is not finite")
    national = sum(b.size for b in model.national)
    regional = sum(b.size for b in model.regional)
    p = s.evolution.shape[0]
    on_national = s.regression[s.regression < national]
    design = np.zeros((horizon, geos, p))
    for g in range(geos):
        own = slice(national + g * regional, national + (g + 1) * regional)
        mine = np.zeros(p)
        mine[:national] = 1.0
        mine[own] = 1.0
        design[:, g, :] = s.design * mine
        design[:, g, on_national] = xs[:, g, : on_national.size]
        on_own = s.regression[(s.regression >= own.start) & (s.regression < own.stop)]
        design[:, g, on_own] = xs[:, g, : on_own.size]
    return design


def _log_multivariate_t(quad: float, log_det: float, k: int, dof: float) -> float:
    """``log`` of the ``k``-variate Student ``t`` density with ``dof`` degrees of freedom and
    squared scale ``Q`` at an error ``e``, given ``quad = e' Q^-1 e`` and ``log_det = log det Q``;
    normal when ``dof`` is infinite."""
    if math.isinf(dof):
        return -0.5 * (k * math.log(2.0 * math.pi) + log_det + quad)
    return float(
        special.gammaln((dof + k) / 2.0)
        - special.gammaln(dof / 2.0)
        - 0.5 * k * math.log(dof * math.pi)
        - 0.5 * log_det
        - (dof + k) / 2.0 * math.log1p(quad / dof)
    )


# ---------------------------------------------------------------------------- the geos' spread

# The least spread searched, a share of the vaguest: below it the geos are pooled completely.
_LEAST_SPREAD = 1e-12
# Central differences, in the log of a spread and in the draws' coordinates: the gradient's step,
# and the curvature's.
_GRADIENT_STEP = 1e-3
_CURVATURE_STEP = 1e-2
# The draws' proposal: a Student t of these degrees of freedom, its scale this many times the
# posterior's curvature's inverse, and no wider than this standard deviation where it is flat.
_PROPOSAL_DOF = 4.0
_PROPOSAL_INFLATION = 1.5
_PROPOSAL_WIDEST = 4.0


@dataclass(frozen=True)
class GeoSpread:
    """How far the geos spread about the national coefficients, chosen by the marginal
    likelihood: :func:`fit_geo_spread`'s output.

    ``variance[i]`` is the prior variance of every geo's regional coordinate ``pooled[i]``, stated
    at the prior's scale as :class:`Prior` states a covariance: at a variance ``V`` the geos spread
    about the national coefficient with variance ``variance[i] V / scale``. ``fit`` is the filter
    at ``variance``, and ``lower`` and ``upper`` each spread's likelihood-ratio interval, the other
    spreads held at their best.

    ``draws`` are spreads drawn from their posterior, ``(draws, pooled)``, under a prior flat on
    each standard deviation over the range searched, with self-normalised importance ``weights``.
    :meth:`mixture` averages a quantity's posterior over them, so that it carries the spread's
    uncertainty as well as the states'.
    """

    fit: GeoDLMFit
    x: _Array | None
    pooled: tuple[int, ...]
    variance: _Array
    lower: _Array
    upper: _Array
    draws: _Array
    weights: _Array

    def at(self, variance: ArrayLike) -> GeoDLM:
        """The model with every geo's pooled coordinates at prior variance ``variance``."""
        return _with_spread(self.fit.model, self.pooled, np.asarray(variance, dtype=np.float64))

    def mixture(
        self, quantity: Callable[[GeoDLMFit], tuple[ArrayLike, ArrayLike]]
    ) -> tuple[_Array, _Array]:
        """A quantity's mean and variance over the spread's posterior.

        ``quantity`` maps a filter to the quantity's posterior mean and variance at that filter's
        spread; the result is their mixture over the draws, ``sum_j w_j m_j`` and
        ``sum_j w_j (v_j + (m_j - mean)^2)``, one filter a draw of positive weight.
        """
        live = np.flatnonzero(self.weights)
        moments = [
            quantity(forward_filter_geos(self.at(self.draws[j]), self.fit.y, self.x)) for j in live
        ]
        means = np.array([np.asarray(m, dtype=np.float64) for m, _ in moments])
        variances = np.array([np.asarray(v, dtype=np.float64) for _, v in moments])
        weights = self.weights[live].reshape((-1,) + (1,) * (means.ndim - 1))
        mean = (weights * means).sum(axis=0)
        return mean, (weights * (variances + (means - mean) ** 2)).sum(axis=0)


def _with_spread(model: GeoDLM, pooled: tuple[int, ...], variance: _Array) -> GeoDLM:
    national = _sizes(model.national)
    regional = _sizes(model.regional)
    covariance = np.array(model.prior.covariance)
    for g in range(model.geos):
        at = national + g * regional + np.array(pooled)
        covariance[at, at] = variance
    prior = Prior(model.prior.mean, covariance, model.prior.scale, model.prior.dof)
    return replace(model, prior=prior)


def _sizes(blocks: tuple[Block, ...]) -> int:
    return sum(b.size for b in blocks)


def _vaguest(model: GeoDLM, pooled: Sequence[int]) -> tuple[tuple[int, ...], _Array]:
    """The pooled coordinates, checked, and their prior variance in the model: the vaguest
    spread."""
    national, regional = _sizes(model.national), _sizes(model.regional)
    coordinates = tuple(pooled)
    if (
        not coordinates
        or any(isinstance(i, bool) or not isinstance(i, int) for i in coordinates)
        or len(set(coordinates)) < len(coordinates)
        or not all(0 <= i < regional for i in coordinates)
    ):
        raise ValueError(
            f"pooled must name distinct coordinates of a geo's regional blocks, 0 to "
            f"{regional - 1}, got {pooled!r}"
        )
    covariance = model.prior.covariance
    k = len(coordinates)
    first = covariance[national + np.array(coordinates), national + np.array(coordinates)]
    for g in range(model.geos):
        at = national + g * regional + np.array(coordinates)
        variance = covariance[at, at]
        if not np.array_equal(variance, first):
            raise ValueError(
                f"geo {g}'s prior variance on the pooled coordinates is {variance.tolist()}, geo"
                f" 0's {first.tolist()}: a spread is one variance for every geo"
            )
        rows = covariance[at].copy()
        rows[np.arange(k), at] = 0.0
        if np.any(rows != 0.0):
            raise ValueError(
                f"geo {g}'s pooled coordinates have prior covariance with other coordinates; a"
                " spread is a variance of its own"
            )
    return coordinates, first.copy()


def fit_geo_spread(
    model: GeoDLM,
    y: ArrayLike,
    x: ArrayLike | None,
    pooled: Sequence[int],
    draws: int,
    seed: int,
    *,
    level: float = 0.9,
) -> GeoSpread:
    """How far the geos spread about the national coefficients, by the marginal likelihood of
    ``y`` (type-II maximum likelihood, empirical Bayes), and draws of the spread from its
    posterior.

    ``pooled`` are coordinates of a geo's regional blocks, counted from 0: each one's prior
    variance, the same in every geo and independent of every other coordinate, is the geos'
    spread on it, and the model's own is the vaguest considered. Each spread's log is searched
    from ``1e-12`` of the model's own up to it, from the model's own, by L-BFGS-B on central
    differences of :attr:`GeoDLMFit.log_likelihood`, which is exact, so the search is the only
    approximation. Each spread's interval at ``level`` is where the log-likelihood, the other
    spreads at their best, is within ``chi2_1(level) / 2`` of the best, or the range's end where
    it stays so.

    The posterior is under a prior flat on each standard deviation over the range (Gelman 2006),
    which unlike a flat prior on the log does not pile its mass at a spread of 0. ``draws``
    spreads are drawn from it with ``seed`` by importance sampling in the range's logistic
    coordinates, ``log_spread = low + (high - low) expit(u)``: a Student ``t`` about the
    posterior's mode in ``u`` with its curvature there. Every draw is in the range, and the mode
    is inside it even where the best is at an end, as when the data would spread the geos wider
    than the model's own prior lets them. A quantity that carries the spread's uncertainty is the
    mixture of its posteriors over the draws, :meth:`GeoSpread.mixture`.

    Raises:
        ValueError: on pooled coordinates that are not distinct coordinates of the regional
            blocks, a pooled coordinate whose prior variance differs between geos or that has
            prior covariance with another, ``draws`` below 1, a ``level`` outside ``(0, 1)``, and
            what :func:`forward_filter_geos` refuses.
    """
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError(f"draws must be a positive integer, got {draws!r}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    coordinates, vaguest = _vaguest(model, pooled)
    ys = np.array(y, dtype=np.float64)
    xs = None if x is None else np.array(x, dtype=np.float64)
    k = len(coordinates)
    high = np.log(vaguest)
    low = high + math.log(_LEAST_SPREAD)
    filters = 0

    def loglik(log_spread: _Array) -> float:
        nonlocal filters
        filters += 1
        spread = _with_spread(model, coordinates, np.exp(log_spread))
        return forward_filter_geos(spread, ys, xs).log_likelihood

    def unit(i: int, step: float) -> _Array:
        out = np.zeros(k)
        out[i] = step
        return out

    def downhill(f: Callable[[_Array], float]) -> Callable[[_Array], tuple[float, _Array]]:
        def value_and_slope(at: _Array) -> tuple[float, _Array]:
            slope = np.array(
                [
                    f(at + unit(i, _GRADIENT_STEP)) - f(at - unit(i, _GRADIENT_STEP))
                    for i in range(k)
                ]
            ) / (2.0 * _GRADIENT_STEP)
            return -f(at), -slope

        return value_and_slope

    search = optimize.minimize(
        downhill(loglik),
        high,
        jac=True,
        method="L-BFGS-B",
        bounds=list(zip(low, high, strict=True)),
    )
    best = np.asarray(search.x, dtype=np.float64)
    fit = forward_filter_geos(_with_spread(model, coordinates, np.exp(best)), ys, xs)
    top = fit.log_likelihood

    cut = float(special.chdtri(1, 1.0 - level)) / 2.0
    lower, upper = best.copy(), best.copy()
    for i in range(k):

        def below(value: float, i: int = i) -> float:
            moved = best.copy()
            moved[i] = value
            return top - loglik(moved) - cut

        lower[i] = (
            low[i] if below(low[i]) <= 0.0 else optimize.brentq(below, low[i], best[i], xtol=1e-3)
        )
        upper[i] = (
            high[i]
            if below(high[i]) <= 0.0
            else optimize.brentq(below, best[i], high[i], xtol=1e-3)
        )

    width = high - low

    def log_spread_at(u: _Array) -> _Array:
        return low + width * special.expit(u)

    def log_posterior(u: _Array) -> float:
        log_spread = log_spread_at(u)
        # the prior's density in the log of a variance, and d log_spread / du
        prior = 0.5 * float(log_spread.sum())
        jacobian = float(np.sum(np.log(width) + special.log_expit(u) + special.log_expit(-u)))
        return loglik(log_spread) + prior + jacobian

    share = np.clip((best - low) / width, 1e-3, 1.0 - 1e-3)
    peak = optimize.minimize(
        downhill(log_posterior), special.logit(share), jac=True, method="L-BFGS-B"
    )
    mode = np.asarray(peak.x, dtype=np.float64)
    height = log_posterior(mode)
    h = _CURVATURE_STEP
    curvature = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            if i == j:
                ends = log_posterior(mode + unit(i, h)) + log_posterior(mode - unit(i, h))
                curvature[i, i] = (ends - 2.0 * height) / h**2
            else:
                corners = (
                    log_posterior(mode + unit(i, h) + unit(j, h))
                    - log_posterior(mode + unit(i, h) - unit(j, h))
                    - log_posterior(mode - unit(i, h) + unit(j, h))
                    + log_posterior(mode - unit(i, h) - unit(j, h))
                )
                curvature[i, j] = curvature[j, i] = corners / (4.0 * h**2)
    eigenvalues, vectors = np.linalg.eigh(_sym(-curvature))
    flattest = _PROPOSAL_WIDEST**-2
    scale = _PROPOSAL_INFLATION * (vectors / np.maximum(eigenvalues, flattest)) @ vectors.T
    root = np.linalg.cholesky(_sym(scale))
    rng = np.random.default_rng(seed)
    z = (
        rng.standard_normal((draws, k))
        / np.sqrt(rng.chisquare(_PROPOSAL_DOF, draws) / _PROPOSAL_DOF)[:, None]
    )
    us = mode + z @ root.T
    proposal = -(_PROPOSAL_DOF + k) / 2.0 * np.log1p(np.sum(z * z, axis=1) / _PROPOSAL_DOF)
    log_weights = np.array([log_posterior(u) for u in us]) - proposal
    log_spreads = np.array([log_spread_at(u) for u in us])
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()
    effective = float(1.0 / np.sum(weights**2))
    _log.info(
        "dlm geo spread: %d coordinates pooled, %d filters, %.3g effective draws of %d",
        k,
        filters,
        effective,
        draws,
        extra={
            "chc_event": "dlm_geo_spread",
            "pooled": k,
            "filters": filters,
            "effective_draws": effective,
            "draws": draws,
            "search": str(search.message),
            "mode_search": str(peak.message),
            "at_least": int(np.sum(best <= low)),
            "at_most": int(np.sum(best >= high)),
        },
    )
    return GeoSpread(
        fit,
        xs,
        coordinates,
        np.exp(best),
        np.exp(lower),
        np.exp(upper),
        np.exp(log_spreads),
        weights,
    )
