"""Which switchback to run, and how to read it, for a named effect of a lever on a plant with
memory.

A switchback randomises a lever over time within a zone, on in some periods and off in others. On a
plant with memory the periods are not independent units: what the lever did last period is still in
the state. So the design has to say which effect it is for. This module plans for a first-order
state per zone,

    x_{t+1} = a x_t + b u_t + eps_t,   u_t in {0, 1} with probability 1/2,   y_t = x_t (+ eta_t),

and for the effect of holding the lever on for ``H`` periods, ``tau_H = b (1 - a^H) / (1 - a)``
(:class:`Horizon`): the channel ``b`` at ``H = 1``, the steady state ``b / (1 - a)`` at
``H = inf``, and any horizon between. ``0 < a < 1``, and nothing here holds outside it.

The results it rests on (the lab's switchback thread, independently verified):

* **The design enters only through one number,** ``A1 = sum_{s >= 1} a^(s-1) rho(s)``, ``rho`` the
  assignment's autocorrelation, when the analysis regresses ``y_{t+1}`` on ``(1, y_t, u_t)``. That
  needs a fair, ergodic design and an intercept.
* **Orthogonality by design.** That regression's plug-in ``b_hat S_H(a_hat)`` has variance at least
  ``(4 sigma^2 / T) S_H(a)^2``, the variance with ``a`` known, with equality iff
  ``A1 = S_H'(a) / S_H(a)``: the design makes the persistence estimate orthogonal to the effect.
  Any other design loses ``q (A1 - A1*)^2 / D(A1)`` of it, ``q = b^2 / (4 sigma^2)``, and the
  optimum does not depend on ``q``. So the channel wants an i.i.d. design, a horizon a Markov
  design that flips with probability ``a / (1 + 2a)`` at ``H = 2``, and the steady state never to
  switch, which a floor on the number of switches stops.
* **Several effects at once** lose ``q (A1 - A1*_H)^2 / D(A1)`` each, with ``D`` common to all, so
  the design that minimises the worst loss sits between the extreme optima.
* **Measurement noise** makes that regression inconsistent for every ``H >= 2``, and for the
  channel at any design but the i.i.d. one. IV with ``(u_t, u_{t-1})`` is consistent at every
  design, and is planned at its own minimax design.
* **Without a trusted state model** only model-free readings are planned: the channel by the
  regression at ``p = 1/2``; ``tau_H`` by the difference in means over blocks of length ``H``
  keeping each block's last period, ``DIM(H, H-1)``, which is unbiased for ``tau_H`` on any linear
  time-invariant plant, or by the local-projection sum at ``p = 1/2``; the steady state by the
  block difference in means with the smallest worst mean squared error, whose bias is quoted and
  not removed, since no model-free design is unbiased for it.

:func:`design_switchback` returns the arms, and for each effect the analysis, its standard error,
its minimum detectable effect and its loss against the effect's own best design, at the least
favourable ``a`` of the prior. Every variance is asymptotic in ``T``. :func:`read_switchback` reads
an effect off the data a design produced, with a standard error from the data, and
:func:`restate_mde` reads it off the first periods of the run, an internal pilot, and restates the
minimum detectable effect the whole run can detect. :func:`randomisation_test` and
:func:`randomisation_interval` infer from the design's own randomisation instead of from ``T``
being large: the test is exact at any number of blocks, and the interval is exact given a range
for ``a``.

What is outside the model, and comes back as a warning rather than a refusal: a second state or a
longer carryover, drift, spillover between zones, ``a`` near 1 on a short run, and fewer than 100
switches. The plug-in's reading tests for a second state, and the block difference's standard error
keeps spillover; the plan's variances assume neither. On the zone market of :mod:`chc.zones`, whose
incentive also acts through a stock and draws half its recruits from the neighbouring zones, the
plug-in read its effects 3% to 14% off, and the block difference spread 1.6-1.7 times as far as
planned; the internal pilot reads that spread off the run. The prior's ``a`` must lie in
``(0, 1)``: for ``a < 0`` the bound is unattainable at even ``H`` and the best design depends on
``q``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize, signal, stats

from chc.frames import _real_numbers

_log = logging.getLogger(__name__)

SwitchbackAnalysis = Literal["state_aware", "iv", "local_projection", "block_dim"]
"""How an effect is read off the switchback:

* ``"state_aware"``: OLS of ``y_{t+1}`` on ``(1, y_t, u_t)``, and ``tau_H = b_hat S_H(a_hat)``;
* ``"iv"``: the same regression with instruments ``(1, u_{t-1}, u_t)``, and the same plug-in;
* ``"local_projection"``: OLS of ``y_{t+1} + ... + y_{t+H}`` on ``(1, u_t, y_t)``, the coefficient
  on ``u_t``;
* ``"block_dim"``: the difference in means of ``y_{t+1}`` over the periods each block keeps.
"""

_GRID = 9  # points of the prior's persistence interval every worst case is taken over
_FEW_SWITCHES = 100
_NEAR_UNIT_ROOT = 0.1  # sd(a_hat) / (1 - a) past which the steady state's delta method is ~8% low
_IID = 5.0  # |lag-one autocorrelation| * sqrt(n) past which a lever is not i.i.d.
_FIRST_ORDER_LEVEL = 0.01  # the p-value under which the data reject a first-order state
_PILOT_BLOCKS = 10  # a block pilot's power at its restated MDE was 0.735 at five blocks a zone


@dataclass(frozen=True)
class Horizon:
    """``tau_H``, the effect of holding the lever on for ``H`` periods. ``H = 1`` is the channel
    ``b`` and ``H = math.inf`` the steady state ``b / (1 - a)``.

    Raises:
        ValueError: when ``periods`` is neither a whole number at least 1 nor ``math.inf``.
    """

    periods: float

    def __post_init__(self) -> None:
        h = float(self.periods)
        if not (h >= 1.0 and (math.isinf(h) or h.is_integer())):
            raise ValueError(f"a horizon is a whole number of periods >= 1, or inf; got {h}")
        object.__setattr__(self, "periods", h)

    @property
    def name(self) -> str:
        if self.periods == 1.0:
            return "channel"
        return "steady_state" if math.isinf(self.periods) else f"tau_{int(self.periods)}"


CHANNEL = Horizon(1)
STEADY_STATE = Horizon(math.inf)


@dataclass(frozen=True)
class PersistencePrior:
    """What a pilot or the history says before the design: ``a`` somewhere in ``[a_lo, a_hi]``, the
    state's shock sd, a guess of the channel ``b``, and the measurement noise's variance as a share
    of the shock's (0 when the state is observed exactly).

    Raises:
        ValueError: unless ``0 < a_lo <= a_hi < 1``, the shock sd is positive, the guess is nonzero,
            the noise ratio is non-negative, and all are finite.
    """

    a_lo: float
    a_hi: float
    shock_sd: float
    channel_guess: float
    noise_ratio: float = 0.0

    def __post_init__(self) -> None:
        values = (self.a_lo, self.a_hi, self.shock_sd, self.channel_guess, self.noise_ratio)
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"the prior must be finite, got {values}")
        if not 0.0 < self.a_lo <= self.a_hi < 1.0:
            raise ValueError(
                f"the persistence interval must satisfy 0 < a_lo <= a_hi < 1, got "
                f"[{self.a_lo}, {self.a_hi}]: for a < 0 the aligned design is unattainable at even "
                "horizons and the best design depends on the signal-to-noise ratio"
            )
        if self.shock_sd <= 0.0 or self.channel_guess == 0.0 or self.noise_ratio < 0.0:
            raise ValueError(
                f"need shock_sd > 0, channel_guess != 0 and noise_ratio >= 0, got {self.shock_sd}, "
                f"{self.channel_guess} and {self.noise_ratio}"
            )

    @property
    def signal_to_noise(self) -> float:
        """``q = b^2 / (4 sigma^2)``: what one period of the lever adds, against the shock."""
        return self.channel_guess**2 / (4.0 * self.shock_sd**2)

    def _grid(self) -> np.ndarray:
        return np.linspace(self.a_lo, self.a_hi, _GRID if self.a_hi > self.a_lo else 1)


@dataclass(frozen=True)
class TargetMDE:
    """Choose the number of zones: the smallest at which every effect's minimum detectable effect is
    at most ``fraction`` of ``|tau_H|``, at the least favourable ``a``.

    Raises:
        ValueError: unless ``0 < fraction`` and it is finite.
    """

    fraction: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.fraction) and self.fraction > 0.0):
            raise ValueError(f"fraction must be positive and finite, got {self.fraction}")


@dataclass(frozen=True)
class MarkovDesign:
    """Each period the lever flips with probability ``flip``; ``1/2`` is i.i.d."""

    flip: float


@dataclass(frozen=True)
class BlockDesign:
    """Blocks of ``length`` periods at one setting, each drawn fair and independently; the analysis
    drops the first ``washout`` periods of each block."""

    length: int
    washout: int


@dataclass(frozen=True)
class SwitchbackArm:
    """A design and the share of each zone's periods it runs for."""

    share: float
    design: MarkovDesign | BlockDesign


@dataclass(frozen=True)
class EstimandReport:
    """One effect's reading: which arm and analysis, and what it resolves at the least favourable
    ``a``.

    ``se`` is the standard error pooled over the zones; ``mde`` is
    ``(z_{1 - alpha/2} + z_power) se + |bias|``; ``bias`` is zero for every consistent analysis and
    the working model's for the steady state read by a block difference in means. ``loss`` is the
    variance over the effect's own best design's in the same mode, minus 1, at its worst ``a``: 0
    when the design is aligned to it.
    """

    estimand: Horizon
    arm: int
    analysis: SwitchbackAnalysis
    se: float
    mde: float
    bias: float
    loss: float


@dataclass(frozen=True)
class SwitchbackPlan:
    """The arms to run in every zone, the reading of each effect, and what the plan cannot see.
    ``alpha`` is the two-sided test's level and ``power`` the power each effect's minimum
    detectable effect is for."""

    arms: tuple[SwitchbackArm, ...]
    reports: tuple[EstimandReport, ...]
    periods: int
    zones: int
    warnings: tuple[str, ...]
    alpha: float
    power: float


# --- closed forms; variances in units of 4 sigma^2 / T per zone -----------------------------------


@overload
def _s(a: float, h: float) -> float: ...
@overload
def _s(a: np.ndarray, h: float) -> np.ndarray: ...
def _s(a, h):
    """``S_H(a) = (1 - a^H) / (1 - a)``, so ``tau_H = b S_H``."""
    return 1.0 / (1.0 - a) if math.isinf(h) else (1.0 - a**h) / (1.0 - a)


@overload
def _ds(a: float, h: float) -> float: ...
@overload
def _ds(a: np.ndarray, h: float) -> np.ndarray: ...
def _ds(a, h):
    if math.isinf(h):
        return 1.0 / (1.0 - a) ** 2
    return (1.0 - a**h) / (1.0 - a) ** 2 - h * a ** (h - 1.0) / (1.0 - a)


def _a1_markov(r: float, a: float) -> float:
    """``A1`` of a Markov design whose lag-one autocorrelation is ``r = 1 - 2 flip``."""
    return r / (1.0 - a * r)


def _information(a1: float, a: float, q: float) -> float:
    """``D(A1)``: a period's information about ``a`` in the regression with ``b`` profiled out, so
    that ``T Var(a_hat) = 1 / D``."""
    return (q * (1.0 + 2.0 * a * a1) + 1.0) / (1.0 - a**2) - q * a1**2


def _v_state_aware(a1: float, a: float, q: float, h: float) -> float:
    s, ds = _s(a, h), _ds(a, h)
    d = _information(a1, a, q)
    return s * s + q * (ds - a1 * s) ** 2 / d


def _v_state_aware_own(a: float, q: float, h: float, r_max: float) -> float:
    """The aligned design's variance, or the switch floor's when the aligned design is past it."""
    return _v_state_aware(min(_ds(a, h) / _s(a, h), _a1_markov(r_max, a)), a, q, h)


def _iv_covariance(r: float, a: float, q: float, kappa: float) -> np.ndarray:
    """``T`` times the covariance of the IV estimates of ``(a, b / (2 sigma))``."""
    beta, a1 = math.sqrt(q), _a1_markov(r, a)
    first_stage = np.array([[beta * a1, 1.0], [beta * (1.0 + a * a1), r]])
    same, lagged = np.array([[1.0, r], [r, 1.0]]), np.array([[r, r * r], [1.0, r]])
    # the error eps_t + eta_{t+1} - a eta_t is MA(1), so the long-run variance keeps its lag
    omega = (1.0 + kappa * (1.0 + a * a)) * same - a * kappa * (lagged + lagged.T)
    inverse = np.linalg.inv(first_stage)
    return inverse @ omega @ inverse.T


def _v_iv(r: float, a: float, q: float, kappa: float, h: float) -> float:
    gradient = np.array([math.sqrt(q) * _ds(a, h), _s(a, h)])
    return float(gradient @ _iv_covariance(r, a, q, kappa) @ gradient)


def _v_naive_channel(a: float, q: float, kappa: float) -> float:
    m0 = (q + 1.0) / (1.0 - a**2)
    return 1.0 + kappa + a**2 * kappa * m0 / (m0 + kappa)


def _v_local_projection(a: float, q: float, kappa: float, h: int) -> float:
    sums = [_s(a, j) for j in range(1, h + 1)]
    m, g, squares = (q + 1.0) / (1.0 - a**2), a * sums[-1], sum(s * s for s in sums)
    cross = sum(a ** (k - 1) * sums[h - k - 1] for k in range(1, h))
    return (
        squares
        + q * (squares - sums[-1] ** 2)
        + h * kappa
        + kappa * g / (m + kappa) * (g * m + 2.0 * q * cross)
    )


def _v_dim(length: int, washout: int, a: float, q: float, kappa: float) -> float:
    kept = length - washout
    within = kept * (1.0 + a) / (1.0 - a) - 2.0 * a * (1.0 - a**kept) / (1.0 - a) ** 2
    carry = a ** (washout + 1) * (1.0 - a**kept) * (1.0 - a**length) / (1.0 - a) ** 2
    return (length / kept**2) * (
        within / (1.0 - a**2) + kappa * kept + q * carry**2 / (1.0 - a ** (2 * length))
    )


def _dim_bias(length: int, washout: int, a: float, h: float) -> float:
    """``(E[DIM] - tau_H) / b``: the block difference in means reads the average of ``tau_{k+1}``
    over the positions it keeps."""
    kept = length - washout
    tail = 0.0 if math.isinf(h) else a**h
    return (tail - a ** (washout + 1) * (1.0 - a**kept) / (kept * (1.0 - a))) / (1.0 - a)


def _minimax(loss, lo: float, hi: float) -> float:
    """The ``r`` in ``[lo, hi]`` with the smallest ``loss``, preferring ``r = 0`` on a tie."""
    grid = np.linspace(lo, hi, 401)
    values = np.array([loss(r) for r in grid])
    i = int(np.argmin(values))
    result = optimize.minimize_scalar(
        loss,
        bounds=(grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]),
        method="bounded",
        options={"xatol": 1e-10},
    )
    r = float(result.x) if result.fun < values[i] else float(grid[i])
    return 0.0 if lo <= 0.0 <= hi and loss(0.0) <= loss(r) + 1e-12 else r


def _dim_choose(grid, q: float, kappa: float, weight: float, longest: int) -> tuple[int, int]:
    """The block difference in means with the smallest worst mean squared error for the steady
    state; ``weight`` puts the squared bias in the variance's units."""
    best, choice = math.inf, (1, 0)
    for length in range(1, longest + 1):
        for washout in range(length):
            mse = max(
                weight * q * _dim_bias(length, washout, a, math.inf) ** 2
                + _v_dim(length, washout, a, q, kappa)
                for a in grid
            )
            if mse < best:
                best, choice = mse, (length, washout)
    return choice


# --- the planner --------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Row:
    estimand: Horizon
    arm: int
    analysis: SwitchbackAnalysis
    variance: Callable[[float], float]  # a -> variance at full share
    bias: Callable[[float], float]  # a -> absolute bias
    own: Callable[[float], float]  # a -> the estimand's own best variance in this mode


def _zero(a: float) -> float:
    return 0.0


def _trusted(estimands, prior: PersistencePrior, periods: int, r_max: float, warnings):
    grid, q, kappa = prior._grid(), prior.signal_to_noise, prior.noise_ratio
    horizons = [e for e in estimands if e.periods > 1.0]
    if kappa == 0.0:
        own = {(e, a): _v_state_aware_own(a, q, e.periods, r_max) for e in estimands for a in grid}
        r = _minimax(
            lambda r: max(
                _v_state_aware(_a1_markov(r, a), a, q, e.periods) / own[e, a]
                for e in estimands
                for a in grid
            ),
            0.0,
            r_max,
        )
        if r >= r_max - 1e-6:
            warnings.append(
                f"never switching is best for the steady state; the floor of {(1 - r_max) / 2:.4f} "
                "flips a period binds, so the floor, not the model, sets the design"
            )
        rows = [
            _Row(
                e,
                0,
                "state_aware",
                lambda a, h=e.periods: _v_state_aware(_a1_markov(r, a), a, q, h),
                _zero,
                lambda a, e=e: own[e, a],
            )
            for e in estimands
        ]
    elif not horizons:
        r = 0.0
        rows = [
            _Row(
                CHANNEL,
                0,
                "state_aware",
                lambda a: _v_naive_channel(a, q, kappa),
                _zero,
                lambda a: _v_naive_channel(a, q, kappa),
            )
        ]
    else:
        own = {}
        for e in estimands:
            for a in grid:
                r_own = _minimax(lambda r, a=a, h=e.periods: _v_iv(r, a, q, kappa, h), -0.5, r_max)
                own[e, a] = _v_iv(r_own, a, q, kappa, e.periods)
                if e.periods == 1.0:
                    own[e, a] = min(own[e, a], _v_naive_channel(a, q, kappa))
        r = _minimax(
            lambda r: max(
                _v_iv(r, a, q, kappa, e.periods) / own[e, a] for e in estimands for a in grid
            ),
            -0.5,
            r_max,
        )
        rows = [
            _Row(
                e,
                0,
                "iv",
                lambda a, h=e.periods: _v_iv(r, a, q, kappa, h),
                _zero,
                lambda a, e=e: own[e, a],
            )
            for e in estimands
        ]
        warnings.append(
            "the state is measured with noise: the regression's plug-in is inconsistent for every "
            "horizon past the channel, so every effect is read by IV with (u_t, u_(t-1))"
        )
    if horizons:
        warnings.append(
            "the plug-in trusts a first-order state: a second state or a longer carryover biases "
            "it, and a drift more persistent than a over-corrects it. read_switchback warns when "
            "the data reject a first-order state; to check the rest, run some zones on the "
            "model-free design (trust_state_model=False) and compare the readings"
        )

        # one zone's, since the O(1/T) bias of a_hat does not shrink with more zones
        def persistence(a: float) -> float:
            if kappa == 0.0:
                return 1.0 / _information(_a1_markov(r, a), a, q)
            return float(_iv_covariance(r, a, q, kappa)[0, 0])

        near, at = max((math.sqrt(persistence(a) / periods) / (1.0 - a), a) for a in grid)
        if near > _NEAR_UNIT_ROOT:
            warnings.append(
                f"one zone's persistence estimate has a standard error {100 * near:.0f}% of 1 - a "
                f"at a = {at:.3g}: its O(1/T) bias does not shrink with more zones, and the "
                "standard errors quoted are optimistic (in variance, 1.07-1.11x at a = 0.6 over "
                "100 periods); for the steady state 1/(1 - a_hat) has no finite moments, so "
                "read its interval, Fieller's, rather than its standard error"
            )
    return [SwitchbackArm(1.0, MarkovDesign(flip=(1.0 - r) / 2.0))], rows


def _model_free(
    estimands, prior: PersistencePrior, periods: int, zones: int, longest: int, warnings
):
    grid, q, kappa = prior._grid(), prior.signal_to_noise, prior.noise_ratio
    warnings.append(
        "no trusted state model: DIM(H, H-1), the local projection and the channel regression at "
        "p = 1/2 are unbiased on any linear time-invariant plant; the variances quoted are the "
        "working first-order model's"
    )
    finite = [e for e in estimands if not math.isinf(e.periods)]
    steady = [e for e in estimands if math.isinf(e.periods)]

    def own_best(e: Horizon, a: float) -> float:
        if e.periods == 1.0:
            return _v_naive_channel(a, q, kappa)
        h = int(e.periods)
        return min(_v_dim(h, h - 1, a, q, kappa), _v_local_projection(a, q, kappa, h))

    def pooled_variance(e: Horizon, a: float) -> float:
        if e.periods == 1.0:
            return _v_naive_channel(a, q, kappa)
        return _v_local_projection(a, q, kappa, int(e.periods))

    pooled_loss = max(
        (pooled_variance(e, a) / own_best(e, a) for e in finite for a in grid), default=0.0
    )
    pooled = bool(finite) and pooled_loss < len(finite)
    designs: list[MarkovDesign | BlockDesign] = []
    rows: list[_Row] = []
    weights: list[float] = []
    if pooled:
        weights.append(pooled_loss)
        designs.append(MarkovDesign(flip=0.5))
        rows.extend(
            _Row(
                e,
                0,
                "state_aware" if e.periods == 1.0 else "local_projection",
                lambda a, e=e: pooled_variance(e, a),
                _zero,
                lambda a, e=e: own_best(e, a),
            )
            for e in finite
        )
    else:
        for e in finite:
            arm = len(designs)
            weights.append(1.0)
            if e.periods == 1.0:
                designs.append(MarkovDesign(flip=0.5))
                rows.append(
                    _Row(
                        e,
                        arm,
                        "state_aware",
                        lambda a: _v_naive_channel(a, q, kappa),
                        _zero,
                        lambda a, e=e: own_best(e, a),
                    )
                )
                continue
            h = int(e.periods)
            projection = all(
                _v_local_projection(a, q, kappa, h) < _v_dim(h, h - 1, a, q, kappa) for a in grid
            )
            designs.append(MarkovDesign(flip=0.5) if projection else BlockDesign(h, h - 1))
            rows.append(
                _Row(
                    e,
                    arm,
                    "local_projection" if projection else "block_dim",
                    (lambda a, h=h: _v_local_projection(a, q, kappa, h))
                    if projection
                    else (lambda a, h=h: _v_dim(h, h - 1, a, q, kappa)),
                    _zero,
                    lambda a, e=e: own_best(e, a),
                )
            )
    if steady:
        weights.append(1.0)
        share = 1.0 / sum(weights)
        length, washout = _dim_choose(grid, q, kappa, zones * share * periods, longest)
        own_length, own_washout = _dim_choose(grid, q, kappa, zones * periods, longest)
        rows.append(
            _Row(
                STEADY_STATE,
                len(designs),
                "block_dim",
                lambda a: _v_dim(length, washout, a, q, kappa),
                lambda a: prior.channel_guess * _dim_bias(length, washout, a, math.inf),
                lambda a: _v_dim(own_length, own_washout, a, q, kappa),
            )
        )
        designs.append(BlockDesign(length, washout))
        warnings.append(
            "the steady state without a trusted model: no model-free design is unbiased for it; "
            "the block difference in means has the smallest worst mean squared error under the "
            "working model, and its bias is quoted, not removed"
        )
    shares = np.array(weights) / sum(weights)
    if len(designs) > 1:
        warnings.append(
            "the effects need different designs, so each zone's periods are split into arms; split "
            "zones instead if the plant drifts"
        )
    return [SwitchbackArm(float(s), d) for s, d in zip(shares, designs, strict=True)], rows


def _assemble(
    estimands, prior, periods, zones, trusted, alpha, power, min_switches
) -> SwitchbackPlan:
    normal = NormalDist()
    z = normal.inv_cdf(1.0 - alpha / 2.0) + normal.inv_cdf(power)
    grid = prior._grid()
    r_max = 1.0 - 2.0 * min_switches / periods
    warnings: list[str] = []
    if trusted:
        arms, rows = _trusted(estimands, prior, periods, r_max, warnings)
    else:
        arms, rows = _model_free(
            estimands, prior, periods, zones, max(1, periods // min_switches), warnings
        )
    reports = []
    for row in rows:
        share = arms[row.arm].share
        variances = [row.variance(a) for a in grid]
        worst = int(np.argmax(variances))
        se = math.sqrt(variances[worst] * 4.0 * prior.shock_sd**2 / (zones * share * periods))
        bias = float(row.bias(grid[worst]))
        reports.append(
            EstimandReport(
                estimand=row.estimand,
                arm=row.arm,
                analysis=row.analysis,
                se=se,
                mde=z * se + abs(bias),
                bias=bias,
                loss=max(row.variance(a) / (share * row.own(a)) for a in grid) - 1.0,
            )
        )
    for j, arm in enumerate(arms):
        d = arm.design
        per_period = d.flip if isinstance(d, MarkovDesign) else 1.0 / d.length
        count = arm.share * periods * per_period
        if count < _FEW_SWITCHES:
            warnings.append(
                f"arm {j}: about {count:.0f} switches or blocks a zone: terms of order 1/blocks "
                "and 1/T are no longer negligible (the block difference's sd ran 10% over the "
                "plan at 40 blocks of 50)"
            )
    if zones > 1:
        warnings.append(
            "several zones: randomise each independently. The plan takes them as independent "
            "replications, so a shock common to the zones, or spillover between them, is outside "
            "its variances: on chc.zones' market, spillover put the block difference's spread at "
            "1.6-1.7x the plan's. The block difference's standard error keeps it; the "
            "regressions' do not"
        )
    return SwitchbackPlan(
        tuple(arms), tuple(reports), periods, zones, tuple(warnings), alpha, power
    )


def design_switchback(
    estimands: tuple[Horizon, ...],
    prior: PersistencePrior,
    periods: int,
    zones: int | TargetMDE = 1,
    *,
    trust_state_model: bool = True,
    alpha: float = 0.05,
    power: float = 0.8,
    min_switches: int = 40,
) -> SwitchbackPlan:
    """The switchback that reads ``estimands`` best over ``periods`` periods in each zone.

    With ``trust_state_model`` the plan is a Markov design at the flip probability that minimises
    the worst loss over the effects and over the prior's persistence interval, read by the
    state-aware plug-in, or by IV when the state is measured with noise and a horizon is asked for.
    Without it, only model-free readings are planned (see the module docstring). ``zones`` is the
    number of zones, or a :class:`TargetMDE` to choose it. ``min_switches`` floors how rarely the
    lever may flip, since never switching is what the steady state asks for.

    Raises:
        ValueError: on no effects or a repeated one, fewer than ``4 * min_switches`` periods, fewer
            than one zone, ``alpha`` or ``power`` outside ``(0, 1)``, or a :class:`TargetMDE`
            smaller than the working-model bias of a steady state read without a model, which no
            number of zones can reach.
    """
    if not estimands or len(set(estimands)) != len(estimands):
        raise ValueError(f"need distinct effects to plan for, got {estimands}")
    if min_switches < 1 or periods < 4 * min_switches:
        raise ValueError(
            f"{periods} periods cannot carry {min_switches} switches: need at least "
            f"{4 * min_switches}"
        )
    if not (0.0 < alpha < 1.0 and 0.0 < power < 1.0):
        raise ValueError(f"alpha and power must lie in (0, 1), got {alpha} and {power}")
    if isinstance(zones, int):
        if zones < 1:
            raise ValueError(f"zones must be at least 1, got {zones}")
        plan = _assemble(
            estimands, prior, periods, zones, trust_state_model, alpha, power, min_switches
        )
    else:
        plan = _zones_for(
            estimands, prior, periods, zones, trust_state_model, alpha, power, min_switches
        )
    _log.info(
        "switchback design: %d arm(s), %d zone(s)",
        len(plan.arms),
        plan.zones,
        extra={
            "chc_event": "switchback_design",
            "arms": [str(arm.design) for arm in plan.arms],
            "zones": plan.zones,
            "warnings": len(plan.warnings),
        },
    )
    return plan


def _zones_for(estimands, prior, periods, target: TargetMDE, trusted, alpha, power, min_switches):
    n = 1
    for _ in range(40):
        plan = _assemble(estimands, prior, periods, n, trusted, alpha, power, min_switches)
        need = 1
        for report in plan.reports:
            smallest = min(
                abs(prior.channel_guess * _s(a, report.estimand.periods)) for a in prior._grid()
            )
            room = target.fraction * smallest - abs(report.bias)
            if room <= 0.0:
                raise ValueError(
                    f"{report.estimand.name}: the working-model bias {report.bias:+.4g} exceeds "
                    f"the target {target.fraction} x |tau| = {target.fraction * smallest:.4g}; no "
                    "number of zones reaches it"
                )
            need = max(need, math.ceil(n * ((report.mde - abs(report.bias)) / room) ** 2))
        if need <= n:
            return plan
        n = need
    raise RuntimeError("the number of zones did not settle in 40 rounds")


# --- reading the data ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SwitchbackReading:
    """An effect read off a switchback's data.

    ``interval`` is the ``1 - alpha`` confidence interval: normal about ``estimate`` with ``se``,
    except for the steady state read through the plant, where it is Fieller's for ``b / (1 - a)``.
    Fieller's is ``(-inf, inf)`` when the data cannot rule out a unit root, which the delta
    method's ``se`` does not show.
    """

    estimand: Horizon
    analysis: SwitchbackAnalysis
    estimate: float
    se: float
    interval: tuple[float, float]
    zones: int
    periods: int


def _within(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=-1, keepdims=True)


def _fit(target, regressors, instruments, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Least squares of ``target`` on ``regressors`` with an intercept per zone, or IV when the
    ``instruments`` differ, and the coefficients' covariance, robust to heteroskedasticity and
    keeping the scores' autocovariances within a zone up to ``lags``. Axes before the regressors'
    are a batch of fits against the one ``target``."""
    y, x, z = _within(target), _within(regressors), _within(instruments)
    bread = np.linalg.inv(np.einsum("...izt,...jzt->...ij", z, x))
    theta = np.einsum("...ij,...j->...i", bread, np.einsum("...izt,...zt->...i", z, y))
    scores = z * (y - np.einsum("...i,...izt->...zt", theta, x))[..., None, :, :]
    meat = np.einsum("...izt,...jzt->...ij", scores, scores)
    for lag in range(1, lags + 1):
        cross = np.einsum("...izt,...jzt->...ij", scores[..., lag:], scores[..., :-lag])
        meat += cross + np.swapaxes(cross, -1, -2)
    n = y.shape[-2] * y.shape[-1]
    scale = n / (n - x.shape[-3] - y.shape[-2])
    return theta, bread @ meat @ np.swapaxes(bread, -1, -2) * scale


def _fieller(a: float, b: float, cov: np.ndarray, z: float) -> tuple[float, float]:
    """The ``tau`` whose ``b - tau (1 - a) = 0`` a level-``z`` test does not reject."""
    d = 1.0 - a
    curvature = d * d - z * z * cov[0, 0]
    if curvature <= 0.0:
        return (-math.inf, math.inf)
    half = b * d + z * z * cov[0, 1]
    spread = math.sqrt(max(half * half - curvature * (b * b - z * z * cov[1, 1]), 0.0))
    return ((half - spread) / curvature, (half + spread) / curvature)


def _lag_one(u: np.ndarray) -> float:
    """The lever's lag-one autocorrelation within the zones."""
    centred = _within(u)
    return float((centred[:, 1:] * centred[:, :-1]).sum() / (centred * centred).sum())


def _first_order(u, y) -> None:
    """Warn when ``y_(t-1)`` and ``u_(t-1)`` add to the plug-in's regression. A first-order state
    observed without noise rules them out, and then the Wald statistic is chi-square on as many
    degrees of freedom as the lags the data tell apart from ``y_t``, ``u_t`` and the zone's
    intercept."""
    if u.shape[0] * (u.shape[1] - 2) <= 4:
        return
    lags = "y_(t-1) and u_(t-1)"
    regressors = np.stack([y[:, 1:-1], u[:, 1:], y[:, :-2], u[:, :-1]])
    within = _within(regressors).reshape(4, -1)
    if np.linalg.matrix_rank(within) < 4:
        # a lever that alternates every period has u_(t-1) = 1 - u_t, which u_t and the intercept
        # absorb. So does a stock the lever drives, which then alternates with it: no test on
        # such data can see one
        if np.linalg.matrix_rank(within[:3]) < 3:
            return
        regressors, lags = regressors[:3], "y_(t-1)"
    theta, cov = _fit(y[:, 2:], regressors, regressors, 0)
    added = theta[2:]
    wald = float(added @ np.linalg.solve(cov[2:, 2:], added))
    p = math.exp(-wald / 2.0) if added.size == 2 else math.erfc(math.sqrt(wald / 2.0))
    if p < _FIRST_ORDER_LEVEL:
        _log.warning(
            "the data reject a first-order state: given y_t and u_t, the outcome still depends on "
            "%s (Wald %.1f against chi-square(%d), p = %.2g). A second state, a longer carryover "
            "or a state measured with noise biases the plug-in (by -3%% to +14%% on chc.zones' "
            "market, whose incentive also acts through a stock); read the effect model-free",
            lags,
            wald,
            added.size,
            p,
            extra={
                "chc_event": "switchback_second_state",
                "wald": wald,
                "degrees_of_freedom": added.size,
                "p": p,
            },
        )


def _plant_fit(u, y, analysis: SwitchbackAnalysis) -> tuple[np.ndarray, np.ndarray]:
    """``(a_hat, b_hat)`` and their covariance. Axes of ``u`` before the zones' are a batch of
    levers read against the one outcome."""
    if analysis == "state_aware":
        regressors = np.stack(np.broadcast_arrays(y[:, :-1], u), axis=-3)
        return _fit(y[:, 1:], regressors, regressors, 0)
    # the error eps_t + eta_(t+1) - a eta_t of a noisy state is MA(1)
    regressors = np.stack(np.broadcast_arrays(y[:, 1:-1], u[..., 1:]), axis=-3)
    return _fit(y[:, 2:], regressors, np.stack([u[..., :-1], u[..., 1:]], axis=-3), 1)


def _plug_in(theta: np.ndarray, cov: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """``b_hat S_H(a_hat)`` and its variance by the delta method."""
    a, b = theta[..., 0], theta[..., 1]
    gradient = np.stack([b * _ds(a, h), _s(a, h)], axis=-1)
    return b * _s(a, h), np.einsum("...i,...ij,...j->...", gradient, cov, gradient)


def _finite_variance(variance: float) -> None:
    """A nan variance passes every sign test, and its standard error would leave
    :func:`randomisation_interval` stepping forever."""
    if not math.isfinite(variance):
        raise ValueError(
            "the reading's variance is not a finite number: the outcome is too large for the "
            "squares its covariance sums; rescale it"
        )


def _plant(u, y, estimand: Horizon, analysis: SwitchbackAnalysis, z: float):
    theta, cov = _plant_fit(u, y, analysis)
    # u_t is independent of everything y_t carries when the lever is i.i.d., so the channel is
    # then read right on any linear time-invariant plant, and there is nothing to test
    if analysis == "state_aware" and (
        estimand.periods > 1.0 or abs(_lag_one(u)) * math.sqrt(u.size) > _IID
    ):
        _first_order(u, y)
    a, b = float(theta[0]), float(theta[1])
    h = estimand.periods
    if h > 1.0:
        # one zone's, as in the design: the O(1/T) bias of a_hat does not shrink with more zones
        near = math.sqrt(cov[0, 0] * u.shape[0]) / (1.0 - a) if a < 1.0 else math.inf
        if near > _NEAR_UNIT_ROOT:
            _log.warning(
                "a_hat = %.4f, and one zone's standard error on it is %.0f%% of 1 - a_hat: its "
                "O(1/T) bias is not in the interval, which covers less than it says (Fieller's "
                "covered 0.930 of 0.95 at a = 0.95 over 200 periods)",
                a,
                100.0 * near,
                extra={"chc_event": "switchback_near_unit_root", "a_hat": a, "ratio": near},
            )
    if math.isinf(h) and a >= 1.0:
        raise ValueError(
            f"the fitted persistence a_hat = {a:.4f} is at or past 1: the plant has no steady "
            "state to read"
        )
    estimate, variance = (float(v) for v in _plug_in(theta, cov, h))
    _finite_variance(variance)
    if variance <= 0.0:
        raise ValueError(
            "the scores' lag-one autocovariance is below minus half their variance, which the "
            "MA(1) error of a first-order state measured with noise cannot produce"
        )
    se = math.sqrt(variance)
    if math.isinf(h):
        return estimate, se, _fieller(a, b, cov, z)
    return estimate, se, (estimate - z * se, estimate + z * se)


def _projection_fit(u, y, h: int) -> tuple[np.ndarray, np.ndarray]:
    """The local projection's coefficient on ``u_t`` and its variance; axes of ``u`` before the
    zones' are a batch of levers."""
    rows = u.shape[-1] - h + 1
    ahead = sum(y[:, j : j + rows] for j in range(1, h + 1))
    regressors = np.stack(np.broadcast_arrays(u[..., :rows], y[:, :rows]), axis=-3)
    # the sums over H periods overlap, so the error is MA(H - 1)
    theta, cov = _fit(ahead, regressors, regressors, h - 1)
    return theta[..., 0], cov[..., 0, 0]


def _projection(u, y, h: int, z: float):
    lag_one = _lag_one(u)
    if abs(lag_one) * math.sqrt(u.size) > _IID:
        raise ValueError(
            f"the local projection needs an i.i.d. lever; this one's lag-one autocorrelation is "
            f"{lag_one:.3f}"
        )
    estimate, variance = (float(v) for v in _projection_fit(u, y, h))
    _finite_variance(variance)
    if variance <= 0.0:
        raise ValueError(
            f"the scores' autocovariances to lag {h - 1} sum to a negative variance, which the "
            f"MA({h - 1}) error of the overlapping sums cannot produce"
        )
    se = math.sqrt(variance)
    return estimate, se, (estimate - z * se, estimate + z * se)


def _block_totals(y, blocks: BlockDesign, count: int) -> np.ndarray:
    """Each of the first ``count`` blocks' total over the periods it keeps, zone by zone."""
    zones, length = y.shape[0], blocks.length
    totals = y[:, 1 : count * length + 1].reshape(zones, count, length)[:, :, blocks.washout :]
    return totals.sum(axis=2)


def _block_reading(sign, totals, kept: int) -> tuple[np.ndarray, np.ndarray]:
    """The block difference and its standard error from each block's setting, ``sign`` = +-1, and
    total, zone by zone. Axes of ``sign`` before the zones' are a batch of schedules; where no zone
    has two usable blocks, both are NaN."""
    # each block is centred on the midpoint of the on and off blocks before it, which its own coin
    # cannot move: the difference in means is then unbiased at any number of blocks, and the
    # scores are martingale differences, so their spread is the standard error. The midpoint and
    # not the plain mean of the blocks before: that one carries the past coins' imbalance, which
    # costs (kept tau / 2)^2 sum 1/b and put the sd at 2.3x on 40 blocks of 50
    on, off = sign > 0.0, sign < 0.0
    seen_on = np.cumsum(on, axis=-1) - on
    seen_off = np.cumsum(off, axis=-1) - off
    usable = (seen_on > 0) & (seen_off > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        midpoint = 0.5 * (
            (np.cumsum(totals * on, axis=-1) - totals * on) / seen_on
            + (np.cumsum(totals * off, axis=-1) - totals * off) / seen_off
        )
        used = usable.sum(axis=-1)
        live = used > 1
        usable &= live[..., None]
        scores = np.where(usable, sign * (totals - midpoint), 0.0)
        # each zone's own difference, the zones weighted alike. Pooling their blocks would weight a
        # zone by how soon its coins showed both settings, a weight that moves from run to run and
        # carries the differences between the zones' effects into the spread
        weight = np.where(live, used, 1)
        mean = np.where(live, scores.sum(axis=-1) / weight, 0.0)
        zones = live.sum(axis=-1)
        estimate = 2.0 * mean.sum(axis=-1) / zones / kept
        # the zones' centred scores are summed block by block before squaring: a zone's neighbours
        # move its outcome with their own coins, which puts their scores for the same block in
        # step. On chc.zones' market, squaring them zone by zone put the standard error at 0.88 of
        # the spread
        step = (np.where(usable, scores - mean[..., None], 0.0) / weight[..., None]).sum(axis=-2)
        blocks = np.where(live, used, 0).sum(axis=-1)
        correction = blocks / (blocks - zones)
        se = 2.0 * np.sqrt((step * step).sum(axis=-1) * correction) / (kept * zones)
    return estimate, se


def _block_difference(u, y, blocks: BlockDesign, z: float):
    zones, periods = u.shape
    length = blocks.length
    count = periods // length
    if count < 3:
        raise ValueError(f"need three whole blocks of {length} a zone, got {count}")
    settings = u[:, : count * length].reshape(zones, count, length)
    if not (settings == settings[:, :, :1]).all():
        raise ValueError(f"the lever changes inside a block of {length} periods")
    estimate, se = (
        float(v)
        for v in _block_reading(
            2.0 * settings[:, :, 0] - 1.0,
            _block_totals(y, blocks, count),
            length - blocks.washout,
        )
    )
    if math.isnan(estimate):
        raise ValueError("need blocks at both settings before the last two in some zone")
    return estimate, se, (estimate - z * se, estimate + z * se)


def read_switchback(
    lever: ArrayLike,
    outcome: ArrayLike,
    estimand: Horizon,
    analysis: SwitchbackAnalysis,
    *,
    blocks: BlockDesign | None = None,
    alpha: float = 0.05,
) -> SwitchbackReading:
    """Read ``estimand`` off a switchback's data by ``analysis``.

    ``lever[z, t]`` is zone ``z``'s lever in period ``t``, 0 or 1, and ``outcome[z, t]`` its
    reading at the start of period ``t``: ``outcome`` has one more period than ``lever``, and
    ``outcome[z, t + 1]`` is the first reading after ``lever[z, t]``. One zone may be passed as 1-D
    arrays. The regressions pool the zones as independent replications, each with an intercept.

    * ``"state_aware"`` and ``"iv"`` fit ``a`` and ``b`` and read ``b S_H(a)``, by the delta method
      on a covariance robust to heteroskedasticity; IV's keeps the lag-one autocovariance of the
      MA(1) error a state measured with noise has. The plug-in also tests the first-order state it
      rests on, by whether ``y_(t-1)`` and ``u_(t-1)`` add to its regression, and warns when the
      data reject it at 1%; not for the channel at an i.i.d. lever, which it reads right on any
      linear time-invariant plant. A lever that alternates every period hides a stock it drives
      from the test, since the stock then alternates with it.
    * ``"local_projection"`` needs an i.i.d. lever, and its covariance keeps the ``H - 1`` lags of
      the overlapping sums.
    * ``"block_dim"`` needs ``blocks``, the design's. Each block's kept periods are centred on the
      midpoint of the on and off blocks before it, so the difference in means is unbiased at any
      number of blocks; the blocks before both settings have been seen only centre. It reads
      ``tau_H`` on ``BlockDesign(H, H - 1)``, and the steady state on any block, with the
      working-model bias :func:`design_switchback` quotes. Each zone's difference is its own and
      the zones are weighted alike, and the standard error sums the zones' scores block by block,
      so it keeps what a spillover or a shock common to the zones puts between them.

    Raises:
        ValueError: when the shapes disagree, the lever is not 0 or 1 or never switches, the
            outcome is not finite, ``alpha`` is outside ``(0, 1)``, ``blocks`` is given without
            ``"block_dim"`` or missing with it, the blocks do not read ``tau_H``, the lever changes
            inside a block, a local projection is asked of a steady state or of a lever that is not
            i.i.d., the steady state is read off a fit with ``a_hat >= 1``, or a variance with lags
            comes out negative, which the working model's errors cannot produce.
    """
    u = np.atleast_2d(np.asarray(_real_numbers(lever, "lever"), dtype=np.float64))
    y = np.atleast_2d(np.asarray(_real_numbers(outcome, "outcome"), dtype=np.float64))
    zones, periods = u.shape
    if y.shape != (zones, periods + 1):
        raise ValueError(
            f"outcome must have one more period than lever: got {y.shape} for lever {u.shape}"
        )
    if not (np.isin(u, (0.0, 1.0)).all() and np.isfinite(y).all()):
        raise ValueError("the lever must be 0 or 1 and the outcome finite")
    if not (u.max(axis=1) > u.min(axis=1)).any():
        raise ValueError("the lever never switches in any zone")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if (analysis == "block_dim") != (blocks is not None):
        raise ValueError(f"blocks are read by block_dim and nothing else; got {analysis}, {blocks}")
    h = estimand.periods
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    if blocks is not None:
        if not (math.isinf(h) or blocks == BlockDesign(int(h), int(h) - 1)):
            raise ValueError(
                f"{estimand.name} is read by blocks of {int(h)} keeping the last, got {blocks}"
            )
        estimate, se, interval = _block_difference(u, y, blocks, z)
    elif analysis == "local_projection":
        if math.isinf(h):
            raise ValueError("a local projection reads a finite horizon, not the steady state")
        estimate, se, interval = _projection(u, y, int(h), z)
    else:
        estimate, se, interval = _plant(u, y, estimand, analysis, z)
    _log.info(
        "switchback reading: %s by %s",
        estimand.name,
        analysis,
        extra={"chc_event": "switchback_reading", "estimate": estimate, "se": se},
    )
    return SwitchbackReading(estimand, analysis, estimate, se, interval, zones, periods)


# --- an internal pilot ----------------------------------------------------------------------------


def _block_pilot(u: np.ndarray, length: int, run_blocks: int) -> tuple[float, float]:
    """The block difference's variance over ``run_blocks`` blocks a zone against its variance over
    the pilot's, and the pilot's degrees of freedom.

    A block is centred on the midpoint of the ``on`` and ``off`` blocks before it, so its score's
    variance is ``1 + (1/on + 1/off) / 4`` times a block's own when the blocks' totals are
    uncorrelated, and the blocks before a zone has shown both settings are lost. The pilot's counts
    are its coins'; the run's later blocks are counted at their expected values, which moves a
    zone's factor by under 0.2% on a balanced pilot and under 1% on one that showed a setting once
    or twice. What spillover puts between two zones' scores for the same block is not inflated by
    the centring, and the ratio carries it as if it were. The standard error sums the zones' scores
    block by block before squaring, so its degrees of freedom are Satterthwaite's over the blocks.
    """
    settings = u[:, : (u.shape[1] // length) * length : length]
    blocks = settings.shape[1]
    if blocks < _PILOT_BLOCKS:
        raise ValueError(
            f"a pilot of {blocks} blocks a zone: the block difference's standard error squares one "
            f"sum a block, and below {_PILOT_BLOCKS} the power at the restated MDE falls short "
            "(0.735 of 0.80 at five blocks on the working model)"
        )
    shown = settings.sum(axis=1)
    alike = np.flatnonzero((shown == 0) | (shown == blocks))
    if alike.size:
        raise ValueError(
            f"zone(s) {alike.tolist()} showed one setting in all {blocks} blocks of the pilot: "
            "their blocks are not centred yet, and when they will be is not known; run a longer "
            "pilot"
        )
    on = np.cumsum(settings, axis=1) - settings
    off = np.arange(blocks) - on
    usable = (on > 0) & (off > 0)
    spread = np.where(
        usable, 1.0 + 0.25 * (1.0 / np.maximum(on, 1) + 1.0 / np.maximum(off, 1)), 0.0
    )
    used = usable.sum(axis=1)
    live = used > 1
    per_block = (spread[live] / used[live, None] ** 2).sum(axis=0)
    later = np.arange(run_blocks - blocks) / 2.0
    ahead = 1.0 + 0.25 * (1.0 / (shown[:, None] + later) + 1.0 / (blocks - shown[:, None] + later))
    run_used = used + run_blocks - blocks
    run_live = run_used > 1
    run = ((spread.sum(axis=1) + ahead.sum(axis=1))[run_live] / run_used[run_live] ** 2).sum()
    pilot = per_block.sum() / live.sum() ** 2
    ratio = run / run_live.sum() ** 2 / pilot
    return float(ratio), float(per_block.sum() ** 2 / (per_block**2).sum())


def restate_mde(
    plan: SwitchbackPlan, estimand: Horizon, lever: ArrayLike, outcome: ArrayLike
) -> EstimandReport:
    """``estimand``'s report, with its standard error and minimum detectable effect re-read from
    the first periods of its arm: an internal pilot.

    The plan's variances are the working model's. On a plant with a second state or spillover the
    model-free readings stay where they were, but those of a horizon or the steady state spread
    further than planned: 1.04 to 1.73 times on :mod:`chc.zones`' market. ``lever`` and
    ``outcome`` are the arm's first periods in every zone, as :func:`read_switchback` takes them.
    They are read by the plan's own analysis, and the standard error is carried to the arm's whole
    length:

    * a regression's by its rows, ``sqrt((n_pilot - L) / (n_run - L))``, with ``L`` the lags its
      covariance keeps, which are also the rows it sheds;
    * the block difference's by its variance over the run's blocks against its variance over the
      pilot's. The blocks before a zone has shown both settings are lost, and the next are centred
      on few blocks before them: rescaled by its periods, a pilot of ten blocks read 1.2 times the
      run's standard error on the working model.

    The restated MDE is ``c se + |bias|``, with ``bias`` the plan's and ``c`` the ``power``
    quantile of the noncentral t with ``nu`` degrees of freedom and noncentrality
    ``z_(1 - alpha/2)``. The run's power at that MDE, averaged over what the pilot could have read,
    is then the plan's when the pilot's variance is chi-square on ``nu``. For a regression
    ``nu = n / (2 L + 1)``, with ``n`` its rows over every zone; for the block difference, whose
    standard error squares one sum over the zones a block, it is Satterthwaite's over the blocks.
    With ``c = z_(1 - alpha/2) + z_power`` the power at the restated MDE was 0.71 of 0.80 at a pilot
    of ten blocks.

    On :mod:`chc.zones`' market, over 4000 runs under each matching, the run's power at the
    restated MDE about the reading's own mean, which leaves the reading's bias out, was 0.81 for
    the channel from a tenth of its arm, 0.78 to 0.79 for ``tau_5`` and 0.80 to 0.81 for ``tau_2``,
    and 0.75, 0.79 and 0.80 for the steady state from ten, fifteen and twenty of its forty blocks.
    The steady state's shortfall at ten is the spillover between zones: it puts a covariance
    between two zones' scores for the same block that the centring does not inflate, and the
    block factor treats it as if it did, so a short pilot reads the standard error low.

    The test at the end is unchanged, and so is its level: the MDE is restated, the run is not
    resized. The report's ``loss`` is the plan's, since the design is.

    Raises:
        ValueError: when the plan does not read ``estimand``, the pilot has other zones than the
            plan or is not shorter than the arm, a block pilot has fewer than ten blocks a zone or
            a zone that showed one setting in all of them, or :func:`read_switchback` refuses the
            pilot.
    """
    report = next((r for r in plan.reports if r.estimand == estimand), None)
    if report is None:
        raise ValueError(
            f"the plan reads {[r.estimand.name for r in plan.reports]}, not {estimand.name}"
        )
    blocks = plan.arms[report.arm].design
    run = round(plan.arms[report.arm].share * plan.periods)
    u = np.atleast_2d(np.asarray(_real_numbers(lever, "lever"), dtype=np.float64))
    zones, periods = u.shape
    if zones != plan.zones or periods >= run:
        raise ValueError(
            f"the pilot is {zones} zone(s) over {periods} periods, and arm {report.arm} of the "
            f"plan is {plan.zones} zone(s) over {run}: a pilot is the arm's first periods"
        )
    if isinstance(blocks, BlockDesign):
        pilot = read_switchback(
            lever, outcome, estimand, report.analysis, blocks=blocks, alpha=plan.alpha
        )
        ratio, dof = _block_pilot(u, blocks.length, run // blocks.length)
    else:
        pilot = read_switchback(lever, outcome, estimand, report.analysis, alpha=plan.alpha)
        # the lags each regression's covariance keeps, which are also the rows it sheds
        if report.analysis == "local_projection":
            lags = int(estimand.periods) - 1
        else:
            lags = 1 if report.analysis == "iv" else 0
        ratio = (periods - lags) / (run - lags)
        dof = zones * (periods - lags) / (2 * lags + 1)
    se = pilot.se * math.sqrt(ratio)
    level = NormalDist().inv_cdf(1.0 - plan.alpha / 2.0)
    mde = float(stats.nct.ppf(plan.power, dof, level)) * se + abs(report.bias)
    _log.info(
        "switchback pilot: %s's MDE restated from %d periods",
        estimand.name,
        periods,
        extra={
            "chc_event": "switchback_pilot",
            "planned_se": report.se,
            "restated_se": se,
            "degrees_of_freedom": dof,
        },
    )
    return EstimandReport(estimand, report.arm, report.analysis, se, mde, report.bias, report.loss)


# --- randomisation inference ----------------------------------------------------------------------

Alternative = Literal["two-sided", "greater", "less"]
"""Which statistics are at least as extreme as the observed one: larger in absolute value, larger,
or smaller."""

_ENUMERATE = 1 << 14  # a block design's schedules are all read up to this many, and drawn past it
_BATCH = 1 << 20  # lever periods over the schedules a batch reads at once
_KEEP = 1 << 23  # lever periods over the schedules an interval keeps instead of drawing again
_TIE = 1e-9  # a statistic within this share of the observed one's size ties with it
_PERSISTENCE_POINTS = 21  # memories on the persistence range the interval is projected over
_WIDEST = 1024.0  # standard errors from the estimate past which an interval's end is infinite
_END = 1e-3  # standard errors an interval's end is found to, and may overshoot by


@dataclass(frozen=True)
class RandomisationTest:
    """The randomisation test of the sharp null that the lever moves no reading in any zone.

    ``statistic`` is the reading's ``estimate / se``, as :func:`read_switchback` reads it, and
    ``p_value`` the chance, over the design's schedules that the reading reads, of a statistic at
    least as extreme against ``alternative``. When ``enumerated`` it is read off all ``schedules``
    of the design, which are equally likely, less those the reading refuses; otherwise off
    ``schedules`` drawn from it, as ``(1 + at least as extreme) / (1 + read)``. Either way it is
    exact given that the data were read: under the null, ``P(p_value <= alpha) <= alpha``, at any
    number of blocks and whatever the plant's memory. A reading with neither an effect nor a noise,
    a constant outcome, has ``statistic`` NaN and ``p_value`` 1.
    """

    estimand: Horizon
    analysis: SwitchbackAnalysis
    alternative: Alternative
    statistic: float
    p_value: float
    schedules: int
    enumerated: bool


def _log_chance(u: np.ndarray, design: MarkovDesign | BlockDesign, coins: int) -> float:
    """The log-probability the design gives the observed lever: over the ``coins`` a zone the
    reading reads for a block design, over every period for a Markov one."""
    zones, periods = u.shape
    if isinstance(design, BlockDesign):
        starts = u[:, :: design.length]
        if not (np.repeat(starts, design.length, axis=1)[:, :periods] == u).all():
            raise ValueError(f"the lever changes inside a block of {design.length} periods")
        return -zones * coins * math.log(2.0)
    if not 0.0 <= design.flip <= 1.0:
        raise ValueError(f"a Markov design flips with a probability in [0, 1], got {design.flip}")
    switches = int(np.count_nonzero(np.diff(u, axis=1)))
    stays = zones * (periods - 1) - switches
    if (switches and design.flip == 0.0) or (stays and design.flip == 1.0):
        raise ValueError(f"MarkovDesign({design.flip}) cannot have drawn this lever")
    chance = -zones * math.log(2.0)
    if switches:
        chance += switches * math.log(design.flip)
    if stays:
        chance += stays * math.log1p(-design.flip)
    return chance


@dataclass(frozen=True)
class _Randomisation:
    """The design's schedules for one observed lever, and the statistic each gives an outcome."""

    u: np.ndarray
    estimand: Horizon
    analysis: SwitchbackAnalysis
    design: MarkovDesign | BlockDesign
    alternative: Alternative
    coins: int
    enumerated: bool
    draws: int
    seed: int

    @classmethod
    def of(
        cls,
        u: np.ndarray,
        estimand: Horizon,
        analysis: SwitchbackAnalysis,
        design: MarkovDesign | BlockDesign,
        alternative: Alternative,
        alpha: float,
        draws: int,
        seed: int,
    ) -> _Randomisation:
        if alternative not in ("two-sided", "greater", "less"):
            raise ValueError(f"alternative is two-sided, greater or less, got {alternative!r}")
        if draws < 1:
            raise ValueError(f"need at least one drawn schedule, got {draws}")
        zones, periods = u.shape
        coins = 0
        if isinstance(design, BlockDesign):
            # the block difference reads whole blocks, and every other reading every period
            whole = analysis == "block_dim"
            coins = periods // design.length if whole else -(-periods // design.length)
        enumerated = isinstance(design, BlockDesign) and (1 << (zones * coins)) <= _ENUMERATE
        sides = 2 if alternative == "two-sided" else 1
        # the complement of a schedule negates every reading and is as likely
        smallest = sides * math.exp(_log_chance(u, design, coins))
        if not enumerated:
            smallest = max(smallest, 1.0 / (1.0 + draws))
        if smallest > alpha:
            raise ValueError(
                f"this schedule's smallest {alternative} p-value is {smallest:.3g}, above "
                f"alpha = {alpha}: no outcome could reject. K fair coins give 2^(1-K) two-sided "
                "and 2^-K one-sided, so at 5% a block design needs six coins over the zones "
                "two-sided and five one-sided"
            )
        return cls(u, estimand, analysis, design, alternative, coins, enumerated, draws, seed)

    @property
    def schedules(self) -> int:
        return 1 << (self.u.shape[0] * self.coins) if self.enumerated else self.draws

    def batches(self):
        """The design's schedules, in batches of levers ``[schedule, zone, period]``."""
        zones, periods = self.u.shape
        size = max(1, _BATCH // (zones * periods))
        design = self.design
        if self.enumerated:
            assert isinstance(design, BlockDesign)
            n = zones * self.coins
            tail = self.u[:, self.coins * design.length :]
            for start in range(0, self.schedules, size):
                index = np.arange(start, min(start + size, self.schedules))
                coins = ((index[:, None] >> np.arange(n)) & 1).reshape(-1, zones, self.coins)
                lever = np.repeat(coins.astype(np.float64), design.length, axis=-1)[..., :periods]
                yield np.concatenate([lever, np.broadcast_to(tail, (len(index), *tail.shape))], -1)
            return
        rng = np.random.default_rng(self.seed)
        for start in range(0, self.draws, size):
            count = min(size, self.draws - start)
            if isinstance(design, MarkovDesign):
                path = np.concatenate(
                    [
                        rng.random((count, zones, 1)) < 0.5,
                        rng.random((count, zones, periods - 1)) < design.flip,
                    ],
                    axis=-1,
                )
                yield (np.cumsum(path, axis=-1) % 2).astype(np.float64)
            else:
                coins = rng.random((count, zones, -(-periods // design.length))) < 0.5
                lever = np.repeat(coins.astype(np.float64), design.length, axis=-1)
                yield lever[..., :periods]

    def statistic(self, levers: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The reading's ``estimate / se`` for each lever against the one outcome ``y``, and NaN
        where :func:`read_switchback` would refuse the schedule: no switch or no usable block to
        read, a steady state past ``a_hat >= 1``, no positive variance, or a local projection's
        lever that is not i.i.d."""
        h, analysis, design = self.estimand.periods, self.analysis, self.design
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            if analysis == "block_dim":
                assert isinstance(design, BlockDesign)
                count = levers.shape[-1] // design.length
                estimate, se = _block_reading(
                    2.0 * levers[..., : count * design.length : design.length] - 1.0,
                    _block_totals(y, design, count),
                    design.length - design.washout,
                )
            else:
                switches = np.abs(np.diff(levers, axis=-1)).sum(axis=-1)
                # a lever that never switches leaves the regression singular, and so does one
                # that alternates every period when it is its own lag's instrument
                idle = switches == 0.0
                if analysis == "iv":
                    idle |= switches == levers.shape[-1] - 1
                degenerate = idle.all(axis=-1)
                if analysis == "local_projection":
                    centred = _within(levers)
                    lag_one = (centred[..., 1:] * centred[..., :-1]).sum(axis=(-2, -1)) / (
                        centred * centred
                    ).sum(axis=(-2, -1))
                    degenerate |= np.abs(lag_one) * math.sqrt(self.u.size) > _IID
                levers = np.where(degenerate[..., None, None], self.u, levers)
                if analysis == "local_projection":
                    estimate, variance = _projection_fit(levers, y, int(h))
                else:
                    theta, cov = _plant_fit(levers, y, analysis)
                    estimate, variance = _plug_in(theta, cov, h)
                    if math.isinf(h):
                        estimate = np.where(theta[..., 0] < 1.0, estimate, np.nan)
                estimate = np.where(degenerate, np.nan, estimate)
                se = np.sqrt(np.where(variance > 0.0, variance, np.nan))
            return estimate / se

    def oriented(self, statistic):
        """The statistic, larger where it is more extreme against the alternative."""
        if self.alternative == "two-sided":
            return np.abs(statistic)
        return statistic if self.alternative == "greater" else -statistic

    def p_value(self, y: np.ndarray, schedules=None) -> tuple[float, float]:
        """The observed statistic and its p-value over the design's schedules the reading reads.

        Under the null the outcome is fixed, and so is which schedules the reading reads, so the
        observed schedule, which it read, is a draw from the design restricted to them: the
        p-value is exact given that the data were read. Counting the other schedules as never
        extreme is exact only over all of them, and given a reading allows up to ``alpha`` over
        the share read: the block difference reads ``1 - 2^(3 - K)`` of one zone's ``K`` coins,
        7/8 at six."""
        observed = float(self.statistic(self.u, y))
        if math.isnan(observed):
            return observed, 1.0
        # a reading without noise has an infinite statistic, which ties only with another
        tie = _TIE * max(abs(observed), 1.0) if math.isfinite(observed) else 0.0
        bar = self.oriented(observed) - tie
        extreme = total = 0
        for levers in self.batches() if schedules is None else schedules:
            statistic = self.statistic(levers, y)
            read = statistic[~np.isnan(statistic)]
            extreme += int(np.count_nonzero(self.oriented(read) >= bar))
            total += read.size
        return observed, extreme / total if self.enumerated else (1 + extreme) / (1 + total)


def _prepare(
    lever, outcome, estimand, analysis, design, alpha
) -> tuple[np.ndarray, np.ndarray, SwitchbackReading]:
    blocks = None
    if analysis == "block_dim":
        if not isinstance(design, BlockDesign):
            raise ValueError(f"the block difference reads a BlockDesign, got {design}")
        blocks = design
    reading = read_switchback(lever, outcome, estimand, analysis, blocks=blocks, alpha=alpha)
    u = np.atleast_2d(np.asarray(lever, dtype=np.float64))
    return u, np.atleast_2d(np.asarray(outcome, dtype=np.float64)), reading


def randomisation_test(
    lever: ArrayLike,
    outcome: ArrayLike,
    estimand: Horizon,
    analysis: SwitchbackAnalysis,
    design: MarkovDesign | BlockDesign,
    *,
    alternative: Alternative = "two-sided",
    alpha: float = 0.05,
    draws: int = 9999,
    seed: int = 0,
) -> RandomisationTest:
    """Test the sharp null that the lever moves no reading, in any zone and at any lag, by
    drawing the design's schedule again: Fisher's randomisation test, with the reading's
    ``estimate / se`` as the statistic.

    Under the null the outcome does not depend on the schedule, so the observed statistic is one
    draw from its distribution over the design's schedules, and the p-value is exact at any number
    of blocks, whatever the plant's memory or the zones' spillover: there is no effect to carry or
    to spill. The schedules the reading refuses, such as a block schedule with fewer than two
    blocks after both settings, are left out of that distribution, since the observed one is a
    schedule the reading read; counting them as never extreme instead would reject up to ``alpha``
    over the share read, 0.107 at 10% on six blocks. The statistic is studentised, which also keeps
    the test's level, in large samples, under the weaker null of an effect that averages zero (Wu
    and Ding 2021).

    ``lever`` and ``outcome`` are as :func:`read_switchback` takes them, and ``design`` is the one
    that drew the lever: every zone's schedule independently, a Markov design's from a fair first
    period and a block design's one fair coin a block. A block design with at most 2^14 schedules
    over the zones is enumerated; otherwise ``draws`` schedules are drawn with ``seed``.

    On the working model with ``a = 0.8`` and no effect (``scripts/bench_switchback_randomisation.py
    size``, 1000 runs each, less those the reading refused), it rejected at 5% 3.6%, 5.2% and 4.6%
    of the block differences of ``tau_5`` over 6, 8 and 12 blocks, where the Wald interval
    excluded 0 in 17.2%, 11.9% and 7.2%; 5.8% and 5.4% of the plug-in's ``tau_3`` over 30 and 60
    periods of ``MarkovDesign(0.3)``, against Wald's 8.7% and 6.1%; and 4.4% of the steady state's
    over 60 periods at ``a = 0.9``, against Fieller's 5.3%. Against pyfixest 0.60.0's ``ritest``,
    on 200 logs of an i.i.d. lever over 80 periods, the channel's statistic agreed with its HC1 t
    to 1e-13. Its p-values, from 999 permutations of the lever where this redraws 4999 schedules,
    differed by 0.008 on average and 0.054 at most, which the permutations' own noise allows.

    Raises:
        ValueError: when :func:`read_switchback` refuses the data, the design cannot have drawn the
            lever or the block difference is asked of a Markov design, or no outcome could reject
            at ``alpha``: ``K`` fair coins give a two-sided p-value of at least ``2^(1 - K)``, so
            at 5% a block design needs six coins over the zones two-sided and five one-sided.
    """
    u, y, _ = _prepare(lever, outcome, estimand, analysis, design, alpha)
    null = _Randomisation.of(u, estimand, analysis, design, alternative, alpha, draws, seed)
    statistic, p = null.p_value(y)
    _log.info(
        "switchback randomisation test: %s by %s, p = %.4g",
        estimand.name,
        analysis,
        p,
        extra={
            "chc_event": "switchback_randomisation",
            "statistic": statistic,
            "p": p,
            "schedules": null.schedules,
            "enumerated": null.enumerated,
        },
    )
    return RandomisationTest(
        estimand, analysis, alternative, statistic, p, null.schedules, null.enumerated
    )


def _kernel(u: np.ndarray, a: float) -> np.ndarray:
    """``sum_{s >= 1} a^(s - 1) u_(t - s)`` for every reading ``t``: what one unit of the channel
    puts into it, the experiment's periods only."""
    carried = signal.lfilter([1.0], [1.0, -a], u, axis=-1)
    return np.concatenate([np.zeros((u.shape[0], 1)), carried], axis=-1)


def randomisation_interval(
    lever: ArrayLike,
    outcome: ArrayLike,
    estimand: Horizon,
    analysis: SwitchbackAnalysis,
    design: MarkovDesign | BlockDesign,
    persistence: tuple[float, float],
    *,
    alpha: float = 0.05,
    draws: int = 999,
    seed: int = 0,
) -> tuple[float, float]:
    """``estimand``'s ``1 - alpha`` confidence interval, by inverting :func:`randomisation_test`
    under the working model's joint null.

    A null that names ``tau_H`` alone is not sharp on a plant with memory: the schedule moves each
    reading through the whole path of the lever, ``b sum_{s >= 1} a^(s - 1) u_(t - s)``, and two
    memories with the same ``tau_H`` impute different readings. The joint null ``(a, b)`` is sharp:
    it imputes the readings with the lever off, ``y - b sum a^(s - 1) u_(t - s)``, which the
    schedule cannot move, and the test of no effect on them is exact
    (``validation/switchback_randomisation.mac``). So the set of every ``tau_H = b S_H(a)`` that
    some ``a`` in ``persistence`` leaves unrejected at ``alpha`` covers ``tau_H`` with probability
    at least ``1 - alpha`` whenever the plant is first-order with its ``a`` in that range: at any
    number of blocks, where the Wald and Fieller intervals are asymptotic. The statistic is read
    off the imputed readings rather than compared with the effect, so a reading that is biased for
    its effect, as the block difference is for the steady state, keeps that. The range is the
    price: it is an assumption, and a long memory read over few blocks is unbounded.

    On the working model (``scripts/bench_switchback_randomisation.py``, 400 runs a case, 1000 for
    the long memory; a run whose every effect the test rejected counts as a miss):

    * the block difference of ``tau_5`` at ``a = 0.8`` over 8, 12 and 20 blocks. Wald covered
      0.886, 0.932 and 0.932. Over ``(0.6, 0.9)`` this covered 0.995, 0.980 and 0.973, unbounded
      in 78%, 10% and none of the runs; at the true memory alone 0.956, 0.935 and 0.948, unbounded
      in 28% of the runs over 8 blocks. At the truth the test rejected 4.0% and 4.9% of 4000 runs
      over 8 and 12 blocks.
    * the plug-in's ``tau_3`` over 40 periods of ``MarkovDesign(0.3)``: Wald 0.925, the range
      0.990, the true memory 0.953, at 1.35 and 1.14 times Wald's median width.
    * the plug-in's steady state at ``a = 0.9`` over 60 and 120 periods: Fieller 0.890 and 0.918,
      unbounded in 23% and 2% of the runs. Over ``(0.8, 0.95)`` this covered 0.993 and 1.000 at
      1.37 and 1.57 times Fieller's median width, and at the true memory 0.938 and 0.940 at half
      of it.
    * ``tau_2`` at ``a = 0.95`` over 12 and 24 blocks of 2: Wald 0.919 and 0.934, the true memory
      0.949 and 0.947, but unbounded in 94% and 84% of the runs.

    A null without the memory, ``persistence = (0, 0)``, is not sharp under carryover. For the
    block difference it still covered 0.950 to 0.966 in every case above, and never unbounded; for
    the plug-in it covered 0.53 of ``tau_3`` and none of the steady states, where it rejected every
    effect in 41% and 48% of the runs.

    Every null is tested on the same ``draws`` schedules, or on all of a block design's when it has
    at most 2^14, at 21 memories spread evenly across ``persistence``; a memory between two of
    them is covered by continuity, not exactly. The ends are found from the first effect accepted
    near the estimate, by steps that double until the test rejects and then bisection to a
    thousandth of a standard error, each end reported on its rejected side. An accepted set with
    gaps may come back with a gap filled, or cut at the first rejection a step lands on. An end
    past 1024 standard errors is infinite.

    Raises:
        ValueError: when :func:`randomisation_test` would refuse, ``persistence`` is not
            ``0 <= lo <= hi < 1``, or the test rejects every effect within eight standard errors
            of the estimate at every memory in the range: the data are at odds with a first-order
            plant there, at least near the estimate.
    """
    lo, hi = (float(v) for v in persistence)
    if not 0.0 <= lo <= hi < 1.0:
        raise ValueError(f"the persistence range must satisfy 0 <= lo <= hi < 1, got {persistence}")
    u, y, reading = _prepare(lever, outcome, estimand, analysis, design, alpha)
    null = _Randomisation.of(u, estimand, analysis, design, "two-sided", alpha, draws, seed)
    kept = list(null.batches()) if null.schedules * u.size <= _KEEP else None
    h = estimand.periods
    memories = np.linspace(lo, hi, _PERSISTENCE_POINTS if hi > lo else 1)
    channels = [(_kernel(u, a) / _s(a, h)) for a in memories]
    order = list(range(len(memories)))

    def accepted(tau: float) -> bool:
        for k, j in enumerate(order):
            if null.p_value(y - tau * channels[j], kept)[1] > alpha:
                order.insert(0, order.pop(k))
                return True
        return False

    # the search's unit; a reading without noise has no standard error to step by
    centre, scale = reading.estimate, reading.se or max(abs(reading.estimate), 1.0)
    candidates = [centre] + [centre + s * k * scale for k in range(1, 9) for s in (1.0, -1.0)]
    start = next((tau for tau in candidates if accepted(tau)), None)
    if start is None:
        raise ValueError(
            f"the randomisation test rejects every {estimand.name} within eight standard errors of "
            f"{centre:.4g} at every memory in [{lo}, {hi}]: the data are at odds with a "
            "first-order plant over that range"
        )
    ends = []
    for direction in (-1.0, 1.0):
        inside, step = start, scale
        while accepted(start + direction * step):
            inside = start + direction * step
            step *= 2.0
            if step > _WIDEST * scale:
                ends.append(direction * math.inf)
                break
        else:
            outside = start + direction * step
            while abs(outside - inside) > _END * scale:
                middle = 0.5 * (inside + outside)
                if accepted(middle):
                    inside = middle
                else:
                    outside = middle
            ends.append(outside)
    interval = (ends[0], ends[1])
    _log.info(
        "switchback randomisation interval: %s by %s, [%.4g, %.4g]",
        estimand.name,
        analysis,
        *interval,
        extra={
            "chc_event": "switchback_randomisation_interval",
            "interval": interval,
            "persistence": (lo, hi),
            "schedules": null.schedules,
            "enumerated": null.enumerated,
        },
    )
    return interval
