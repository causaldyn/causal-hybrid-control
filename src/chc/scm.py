"""Synthetic control and augmented synthetic control for a single treated unit.

The synthetic control method (SCM, Abadie-Diamond-Hainmueller) builds a counterfactual for one
treated unit as a convex combination of donor units matched on the pre-treatment outcome path; the
treatment effect is the treated-minus-synthetic gap in the post period. Its weakness is the simplex
constraint: when the treated unit lies outside the donors' convex hull no weighting balances the
pre-period, so the estimate is biased.

Augmented SCM (ASCM, Ben-Michael-Feller-Rothstein) corrects exactly this: it keeps the SCM donor mix
but adds a ridge outcome-model correction for the residual pre-period imbalance, which is allowed to
extrapolate. When SCM already balances, the correction is ~0 and ASCM reduces to SCM.

A statistical estimator, so NumPy float64 throughout (like :mod:`chc.did`), independent of the JAX
``x64`` flag -- which must not change an estimate.

:func:`synthetic_control_inference` adds Abadie, Diamond and Hainmueller's in-space placebo test and
an interval from Chernozhukov, Wüthrich and Zhu's conformal test. Both are owned here. ``diff-diff``
carries both and is the tests' oracle, but its placebo leaves the treated unit out of every
placebo's donor pool, which gives up the test's exact level, and its weights come from a
Frank-Wolfe solver that needed 200 000 steps to converge; over this module's exact weights each test
is a few lines.

Neither test holds when a design chose the treated unit, as a design that picks its test market
does. :func:`blank_period_inference` gives such a test the reading Abadie and Zhao's design
justifies: the weights fit on the earlier periods before treatment, the latest ones left blank,
and the gaps after treatment ranked against every arrangement of the blank and post periods. Its
``serial="ar1"`` reading keeps that test's level where the errors persist, as a geo panel's do.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import special
from scipy.optimize import nnls

from chc import _units
from chc.frames import _real_numbers

__all__ = [
    "BlankPeriodInference",
    "SyntheticControlInference",
    "SyntheticControlResult",
    "augmented_synthetic_control",
    "blank_period_inference",
    "synthetic_control",
    "synthetic_control_inference",
]

Outcomes = NDArray[np.float64]
Vector = NDArray[np.float64]


@dataclass(frozen=True)
class SyntheticControlResult:
    """A fitted synthetic control: post-period effect path, donor mix, and pre-period fit."""

    att: Vector  # (T1,) treated-minus-synthetic gap for each post-treatment period
    overall: float  # mean post-treatment ATT
    weights: Vector  # (J,) donor weights, aligned with the units minus the treated one
    pre_rmspe: float  # pre-period root-mean-squared prediction error (smaller = better match)


def _scm_weights(donor_pre: Outcomes, treated_pre: Vector, steps: int) -> Vector:
    """Simplex weights minimising ``||treated_pre - donor_pre.T @ w||``, exactly.

    The synthetic control is the point of the donors' hull nearest the treated unit, so with
    ``Z = donor_pre.T - treated_pre`` its weights give the point of the hull of ``Z``'s columns
    nearest the origin. For ``u >= 0`` minimising ``||Z u||^2 + (1'u - 1)^2``, ``u / 1'u`` is that
    point: writing ``u = s w`` with ``w`` on the simplex, the best ``s`` leaves ``a / (1 + a)``
    with ``a = ||Z w||^2``, which grows with ``a``. That is one non-negative least squares, which
    Lawson and Hanson's active set solves in finitely many steps, at most ``steps``. ``Z`` is
    scaled first: the weights do not depend on its scale, and the appended row's weight does.
    """
    z = donor_pre.T - treated_pre[:, None]  # (T0, J)
    scale = max(float(np.sqrt(np.mean(z**2))), np.finfo(np.float64).tiny)
    design = np.vstack([z / scale, np.ones(z.shape[1])])
    target = np.zeros(design.shape[0])
    target[-1] = 1.0
    u, _ = nnls(design, target, maxiter=steps)
    return u / u.sum()


def _split(
    outcomes: Outcomes, treated_unit: int, n_pre: int
) -> tuple[Outcomes, Outcomes, Vector, Vector]:
    outcomes = np.asarray(_real_numbers(outcomes, "outcomes"), dtype=np.float64)
    n_units, n_periods = outcomes.shape
    if not 0 <= treated_unit < n_units:
        msg = f"treated_unit {treated_unit} out of range for {n_units} units"
        raise ValueError(msg)
    if not 1 <= n_pre < n_periods:
        msg = f"n_pre must be in [1, {n_periods - 1}], got {n_pre}"
        raise ValueError(msg)
    donors = np.delete(outcomes, treated_unit, axis=0)
    if donors.shape[0] < 1:
        msg = "need at least one donor unit"
        raise ValueError(msg)
    treated = outcomes[treated_unit]
    return donors[:, :n_pre], donors[:, n_pre:], treated[:n_pre], treated[n_pre:]


def synthetic_control(
    outcomes: Outcomes, treated_unit: int, n_pre: int, *, steps: int = 5000
) -> SyntheticControlResult:
    """Classic simplex synthetic control for one treated unit against the remaining donor units.

    ``outcomes`` is ``(N, T)``; treatment starts at period ``n_pre`` (so periods ``0..n_pre-1`` are
    pre-treatment). Donor weights lie on the probability simplex; ``att[k]`` is the treated-minus-
    synthetic gap in post period ``k``. The weights are the exact optimum, whatever the outcomes'
    units; ``steps`` caps the solver's iterations, and running out raises ``RuntimeError``.
    """
    donor_pre, donor_post, treated_pre, treated_post = _split(outcomes, treated_unit, n_pre)
    w = _scm_weights(donor_pre, treated_pre, steps)
    pre_rmspe = float(np.sqrt(np.mean((treated_pre - donor_pre.T @ w) ** 2)))
    att = treated_post - donor_post.T @ w
    return SyntheticControlResult(att, float(att.mean()), w, pre_rmspe)


def augmented_synthetic_control(
    outcomes: Outcomes, treated_unit: int, n_pre: int, *, ridge: float = 0.1, steps: int = 5000
) -> SyntheticControlResult:
    """Ridge-augmented synthetic control (Ben-Michael-Feller-Rothstein).

    Starts from the SCM donor weights, then de-biases each post period by the residual pre-period
    imbalance passed through a ridge outcome model fit on the donors:
    ``Y_1(0)_t = w' Y_post_t + (treated_pre - w' donor_pre)' theta_t``, with ``theta_t`` the
    slopes of a ridge regression of the donors' ``Y_post_t`` on their pre-period outcomes with a
    free intercept, as Ben-Michael, Feller and Rothstein's has: ``theta_t = (Zc'Zc + ridge * v *
    I)^{-1} Zc' Y_post_t``, with ``Zc`` each pre-period centred on the donors' mean, or zero where
    the donors moved apart by rounding alone (:mod:`chc._units`), and ``v`` the mean of ``Zc**2``,
    the donors' variance before treatment, or 1 where ``Zc`` is all zero. The intercept cancels
    from the estimate, since the weights sum to one. The correction vanishes when SCM already
    balances the pre-period; ``ridge`` controls how far the outcome model may extrapolate, and
    ``ridge = lam / v`` is a ridge of ``lam`` in the outcomes' squared units. As a share of ``v``
    it shrinks the same in any units and from any origin: a ridge in the outcomes' squared units
    outweighed donors logged in small ones, so that at 1e-6 of them the estimate was plain SCM's,
    and without the intercept the estimate moved with the outcomes' level. ``weights`` are the
    (interpretable) SCM donor weights; the augmentation is an additive outcome correction, not
    folded into them.

    The default share is a tenth, a compromise, since no one share suits every panel. On 200 panels
    of 30 donors and 25 periods before treatment, the root mean squared error of the effect at a
    tenth was 0.0526 where the factors are white noise, against 0.0502 at the best share, 1. Where
    they are random walks it was 0.110, against 0.105 at the best share, 0.03, and 0.359 at 1. With
    those panels indexed to their first period it was 0.142, against 0.126 at the best, 0.01: a
    tenth errs by up to 1.133 times the best fixed share. 0.07 would have stayed within 1.1 times it
    on all three kinds, but the default was set before they were measured, and moving it to suit
    them would tune it on them. The earlier default, a ridge of 1 in the outcomes' squared units,
    erred less on these panels, 0.0497, 0.107 and 0.113, by a coincidence of units: their noise is
    0.1 in those units, so it was a fixed multiple of the noise variance. At 1e-3 of the units it
    gave plain SCM's estimate.
    """
    donor_pre, donor_post, treated_pre, treated_post = _split(outcomes, treated_unit, n_pre)
    w = _scm_weights(donor_pre, treated_pre, steps)
    imbalance = treated_pre - donor_pre.T @ w  # (T0,) pre-period residual SCM cannot balance
    pre_rmspe = float(np.sqrt(np.mean(imbalance**2)))
    n_pre_periods = donor_pre.shape[1]
    # Solved in centred coordinates, so that the outcomes' level costs the slopes no digits. The
    # ridge's unit is the donors' variance pooled over the periods, which are one outcome in one
    # unit: a share of each period's own let a period the donors barely moved in take a slope as
    # large as its spread was small.
    centred = _units.centred_np(donor_pre).deviations
    variance = float(np.mean(centred**2)) or 1.0
    gram = centred.T @ centred + ridge * variance * np.eye(n_pre_periods)
    theta = np.linalg.solve(gram, centred.T @ (donor_post - donor_post.mean(axis=0)))  # (T0, T1)
    counterfactual = donor_post.T @ w + theta.T @ imbalance  # SCM + ridge bias correction
    att = treated_post - counterfactual
    return SyntheticControlResult(att, float(att.mean()), w, pre_rmspe)


@dataclass(frozen=True)
class SyntheticControlInference:
    """:func:`synthetic_control`'s estimate, with a placebo test and a conformal interval.
    *Experimental.*

    ``placebo_p_value`` is Abadie, Diamond and Hainmueller's in-space placebo for the null that the
    treatment moved nothing, built as they built it: each unit's synthetic control from all the
    others, and the share of the ``N`` units whose ratio of post- to pre-period prediction error is
    at least the treated unit's. A pre-period error below ``sqrt(eps)`` of the panel's spread before
    treatment counts as that much, so units their donors reproduce, as they can when there are more
    donors than periods before treatment, rank by their error after it. When the treated unit was
    drawn at random from the units, the test holds its level exactly; it moves in steps of
    ``1 / N``.

    ``interval`` holds a constant effect with probability ``1 - alpha``: the hull of the effects
    Chernozhukov, Wüthrich and Zhu's conformal test does not reject, as a search finds them. The
    test takes the effect out of the treated unit's periods after treatment, refits its synthetic
    control over every period, and ranks the residuals after treatment against each cyclic shift of
    the residuals in time. So it needs the residuals exchangeable in time, not the treatment random.
    Its p-values move in steps of ``1 / T``: below ``alpha = 1 / T`` the interval is unbounded, and
    it is ``(nan, nan)`` when the test rejects every effect the search tries, which can miss a set
    of accepted effects narrower than its step. The p-value need not fall away from the estimate,
    which is not refitted after treatment, so the interval can leave the estimate out; and a hull
    can hold effects the test rejects, the estimate among them.

    Measured on factor panels of 20 periods before treatment and 5 after, 1000 a case
    (``scripts/bench_scm_inference.py``). With the treated unit drawn at random, the placebo
    rejected a true null 3.8-4.7% of the time at 5% with 19 donors, and 9.2-9.6% at 10% with 9 or
    19. The interval covered the effect 97.2-98.4% of the time at ``alpha = 0.05`` and 93.1-94.0% at
    0.10, where the test's own levels, on 25 periods, are 4% and 8%. Neither holds when the treated
    unit was chosen as the one its donors fit best, as a design that picks its test market chooses:
    the placebo then rejected 27.8% at 5%, and the interval covered 94.4% and 81.7%.
    """

    estimate: SyntheticControlResult
    placebo_p_value: float
    interval: tuple[float, float]
    alpha: float


def synthetic_control_inference(
    outcomes: Outcomes, treated_unit: int, n_pre: int, *, alpha: float = 0.05
) -> SyntheticControlInference:
    """:func:`synthetic_control` with a placebo test and a conformal interval.

    Experimental: it may change or be withdrawn in any release.

    Takes :func:`synthetic_control`'s arguments, and computes its estimate. The placebo test fits a
    synthetic control to every unit. The interval's search tries 401 effects over ten errors either
    side of the estimate, a twentieth of an error apart, the error being the root mean square of the
    treated unit's residual, over every period, at the estimate. It pushes the outermost effects not
    rejected outwards, and bisects each end to a millionth of that error.

    Raises:
        ValueError: on :func:`synthetic_control`'s refusals, or ``alpha`` outside ``(0, 1)``.
    """
    if not 0.0 < alpha < 1.0:
        msg = f"alpha must lie in (0, 1), got {alpha}"
        raise ValueError(msg)
    estimate = synthetic_control(outcomes, treated_unit, n_pre)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    fits = [synthetic_control(outcomes, unit, n_pre) for unit in range(outcomes.shape[0])]
    # a pre-period error rounding could have left counts as the floor, so the units their donors
    # reproduce rank by their error after treatment; the ratios are compared cross-multiplied, so a
    # panel with no spread before treatment divides by nothing
    floor = math.sqrt(np.finfo(np.float64).eps) * float(np.std(outcomes[:, :n_pre]))
    pre = np.maximum([fit.pre_rmspe for fit in fits], floor)
    post = np.array([np.sqrt(np.mean(fit.att**2)) for fit in fits])
    placebo = float(np.mean(post * pre[treated_unit] >= post[treated_unit] * pre))
    interval = _conformal_interval(outcomes, treated_unit, n_pre, estimate.overall, alpha)
    return SyntheticControlInference(estimate, placebo, interval, alpha)


def _conformal_residual(outcomes: Outcomes, treated_unit: int, n_pre: int, effect: float) -> Vector:
    """The treated unit's residual in every period, the constant ``effect`` taken out after
    treatment and its synthetic control refitted over all the periods."""
    nulled = outcomes.copy()
    nulled[treated_unit, n_pre:] -= effect
    donors = np.delete(nulled, treated_unit, axis=0)
    treated = nulled[treated_unit]
    return treated - donors.T @ _scm_weights(donors, treated, 5000)


def _conformal_p_value(outcomes: Outcomes, treated_unit: int, n_pre: int, effect: float) -> float:
    """Chernozhukov, Wüthrich and Zhu's p-value for the constant ``effect``, over moving blocks: the
    absolute residuals after treatment summed, against the same sum over each cyclic shift of the
    residuals in time."""
    residual = np.abs(_conformal_residual(outcomes, treated_unit, n_pre, effect))
    n_periods = residual.size
    windows = np.lib.stride_tricks.sliding_window_view(np.tile(residual, 2), n_periods - n_pre)
    statistic = windows[n_pre : n_pre + n_periods].sum(axis=1)  # [0] is the periods as observed
    return float(np.mean(statistic >= statistic[0]))


def _conformal_interval(
    outcomes: Outcomes, treated_unit: int, n_pre: int, estimate: float, alpha: float
) -> tuple[float, float]:
    """The hull of the constant effects the conformal test does not reject at ``alpha``."""
    if alpha < 1.0 / outcomes.shape[1]:
        return -math.inf, math.inf  # the observed periods' own shift makes every p-value >= 1 / T
    residual = _conformal_residual(outcomes, treated_unit, n_pre, estimate)
    # where the donors reproduce the unit in every period, any length finds the one effect accepted
    scale = float(np.sqrt(np.mean(residual**2))) or 1.0
    # far enough out, the residuals after treatment carry the effect and every shift ranks below
    # the observed periods, so p = 1 / T <= alpha: the accepted set is bounded
    return _hull(
        lambda effect: _conformal_p_value(outcomes, treated_unit, n_pre, effect) > alpha,
        estimate,
        scale,
    )


def _hull(accepts: Callable[[float], bool], centre: float, scale: float) -> tuple[float, float]:
    """The ends of the bounded set ``accepts`` holds, searched from 401 points over ``centre +- 10
    scale``: the outermost accepted are pushed outwards by doubling steps until one is rejected,
    then each end is bisected to ``1e-6 scale``. ``(nan, nan)`` when no point is accepted."""
    accepted = [
        float(x) for x in centre + scale * np.linspace(-10.0, 10.0, 401) if accepts(float(x))
    ]
    if not accepted:
        return math.nan, math.nan
    ends = []
    for inside, direction in ((accepted[0], -1.0), (accepted[-1], 1.0)):
        step = 0.05 * scale
        while accepts(inside + direction * step):
            inside += direction * step
            step *= 2.0
        outside = inside + direction * step
        middle = 0.5 * (inside + outside)
        while abs(outside - inside) > 1e-6 * scale and middle not in (inside, outside):
            if accepts(middle):
                inside = middle
            else:
                outside = middle
            middle = 0.5 * (inside + outside)
        ends.append(inside)
    return ends[0], ends[1]


# Above this many arrangements, enumerating them costs more than a test of a design should.
_MAX_ARRANGEMENTS = 1_000_000


@dataclass(frozen=True)
class BlankPeriodInference:
    """A designed test's synthetic control fitted on the earlier periods before treatment, with a
    permutation test and split-conformal intervals read from the periods it left blank.
    *Experimental.*

    ``estimate`` is the synthetic control fitted on the fitting periods alone: its ``pre_rmspe`` is
    their error, and its ``att`` and ``overall`` are the gaps after treatment. ``blank_gaps`` are
    the gaps, under the same weights, in the blank periods, which the weights never saw.

    ``p_value`` is Abadie and Zhao's (arXiv 2108.02196 v5, eq. 19) for the null that the treatment
    moved nothing: the share of the ``arrangements`` ways of choosing ``T_post`` of the blank and
    post periods whose mean absolute gap is at least the post periods'. It moves in steps of
    ``1 / arrangements``, ``C(T_blank + T_post, T_post)``, and is never below one step.

    Read with ``serial="ar1"``, ``ar_coefficient`` is an AR(1) coefficient of the gaps, estimated
    from the donors alone, and ``p_value`` ranks the gaps' innovations instead: ``e_t = u_t -
    ar_coefficient * u_{t-1}`` over the blank and post periods, ``u`` the gaps and the first lag
    the last fitting period's. Its statistic is ``|e_first + (1 - ar_coefficient) * (sum of the
    other post innovations)|``, which a constant effect moves most. The ``arrangements`` are each
    choice of ``T_post`` innovations and which of them comes first, ``T_post C(T_blank + T_post,
    T_post)`` of them. ``ar_check_p_value`` is a portmanteau p-value for the AR(1) model, read on
    the donors' gaps filtered by ``ar_coefficient``: small when their innovations still correlate
    at lags 1 and 2, which an AR(2), or factors and errors that persist at different rates, leave.
    It is reported and refuses nothing. Both are ``None`` in the default reading.

    ``intervals[k]`` holds the effect in post period ``k`` with probability at least ``1 - alpha``:
    the gap there, plus or minus the ``ceil((1 - alpha) (T_blank + 1))``-th smallest absolute
    blank gap, which is infinite when that rank exceeds ``T_blank``, as it does for every ``alpha
    < 1 / (T_blank + 1)``. That is split conformal prediction's finite-sample rank, one above Abadie
    and Zhao's eq. 20, which reaches ``1 - alpha`` only as ``T_blank`` grows. Read with
    ``serial="ar1"``, ``intervals[k]`` is the prediction interval of a ``k + 1``-step forecast of
    the gaps from the last blank one, from the blank periods' innovations, with probability
    ``1 - alpha`` when they are normal at the true ``rho`` (see :func:`blank_period_inference`);
    with ``ar_coefficient`` estimated, few donors or a long post period lower it; infinite with
    fewer than two blank periods.

    Measured on factor panels of 19 donors, 15 fitting, 5 blank and 5 post periods, the treated
    unit the one its donors fit best on the fitting periods, 4000 a case
    (``scripts/bench_scm_inference.py blank``). With white-noise factors and errors, the test
    rejected a true null 4.4% of the time at 5%, where :func:`synthetic_control_inference`'s
    placebo rejected 23.5% on 400 such panels. With AR(1) factors and errors of coefficient 0.5 it
    rejected 8.7%, against that placebo's 27.8%: errors that persist break the exchangeability, and
    the rate grew to 21.5% at 0.8. Factors that persist did not, 4.8% with white-noise errors. The
    ``serial="ar1"`` reading rejected 4.9%, 4.8% and 5.3% at coefficients 0, 0.5 and 0.8 (standard
    errors 0.35). The default intervals covered 83.6%, 79.6% and 70.7% at ``alpha = 0.2``; at
    0.05 and 0.10 they were unbounded. The ``serial="ar1"`` intervals, on 2000 panels a case,
    covered 79.6%, 79.7% and 79.6% at 0.2, 89.6% to 89.9% at 0.1 and 94.8% to 94.9% at 0.05.
    """

    estimate: SyntheticControlResult
    blank_gaps: Vector
    p_value: float
    intervals: NDArray[np.float64]  # (T_post, 2): each post period's lower and upper end
    alpha: float
    arrangements: int
    ar_coefficient: float | None = None
    ar_check_p_value: float | None = None


def blank_period_inference(
    outcomes: Outcomes,
    treated_unit: int,
    n_pre: int,
    n_blank: int,
    *,
    alpha: float = 0.05,
    serial: Literal["exchangeable", "ar1"] = "exchangeable",
) -> BlankPeriodInference:
    """A designed geo test read as Abadie and Zhao read it: fit on the earlier periods before
    treatment, hold out the last ``n_blank`` of them, permute over the blank and post periods.

    Experimental: it may change or be withdrawn in any release.

    ``outcomes``, ``treated_unit`` and ``n_pre`` are :func:`synthetic_control`'s. The weights fit
    on periods ``0 .. n_pre - n_blank - 1``; periods ``n_pre - n_blank .. n_pre - 1`` are blank.
    The treated unit may be any unit a design chose by its fit on the fitting periods, or on
    anything else outside the blank and post periods.

    Abadie and Zhao's Theorem 2 brackets the test's size, ``alpha - 1 / arrangements <= P(p <=
    alpha) <= alpha``, when the factors over the blank and post periods are exchangeable and
    independent of the noise, and each unit's noise is i.i.d. over time; then the gaps over those
    periods are exchangeable, whatever the fit. The upper end needs only that exchangeability; the
    lower one needs no ties. So:

    - Noise that persists over time breaks it: the mean gap over adjacent post periods then varies
      more than over scattered ones, and the test rejects too often, 8.7% at 5% with AR(1) errors
      of coefficient 0.5. Factors that persist do not. ``serial="ar1"`` is the remedy, below.
    - The design must not look at the blank periods. A unit chosen for its fit there has blank gaps
      selected small, and the test then rejects too often.
    - The blank periods are the latest before treatment, so that a season or a trend moves them as
      little apart from the post periods as the panel allows. Nothing in the data checks
      exchangeability; a season that turns at treatment breaks it.
    - The blank periods leave the fit: the weights see ``n_pre - n_blank`` periods, not
      ``n_pre``.
    - The p-value moves in steps of ``1 / C(n_blank + T_post, T_post)``: with 5 blank and 5 post
      periods, of 1/252; with 1 blank period, of 1/6, so no test at 5% rejects.

    ``intervals`` hold under the same exchangeability, the effect being additive and the noise
    after treatment distributed as before it, one period at a time, not jointly.

    ``serial="ar1"`` reads gaps that follow an AR(1), ``u_t = c + rho u_{t-1} + e_t`` with ``e``
    i.i.d., as they do when every unit's errors and the factors share one coefficient. Then the
    innovations over the blank and post periods are i.i.d. given the fitting periods, and with the
    true ``rho`` the test is exact again, whatever the design did with the fitting periods
    (Chernozhukov, Wüthrich and Zhu, JASA 2021, §2.4.2 and Lemma 7, filter a synthetic control's
    residuals so before permuting them). ``rho`` is estimated:

    - Not from the treated unit's gaps over the fitting periods. The fit absorbs their persistence:
      on the bench's panels their AR(1) estimate averaged -0.13 where the errors' was 0.5.
    - From the donors: each donor's synthetic control from the other donors, the treated unit left
      out, fitted on the first ``ceil(T_fit / 2)`` periods; its gaps on every later period; and
      Han, Phillips and Sul's panel fully aggregated estimator over them (Econometric Theory 30,
      2014, §2), which removes each donor's level without the within-estimator's bias, kept in
      ``[-1, 1]``. An effect on the treated unit does not reach it. It assumes the donors' gaps
      persist as the treated unit's do.
    - An estimate below ``rho`` leaves the innovations positively correlated, and the test rejects
      too often: with ``rho`` fixed 0.1 low, 5.8% at 5% at 0.5. On the bench's panels the estimate
      averaged 0.49 at 0.5 (s.d. 0.08) and 0.77 at 0.8, and the test rejected 4.8% and 5.3%. With
      10 blank periods, 5 fitting for the donors, it rejected 5.5% on 12000 panels.
    - Its power is for a constant effect: an effect of 1 and 2 on the bench's panels, rejected
      14.9% and 45.2% at 0.5, where the exchangeable reading rejected 19.7% and 44.3% at a size of
      8.7%, and 27.5% and 76.8% where the errors do not persist, against 13.5% and 48.5%.
    - The intervals read the innovations too. A post period's gap is not one innovation, but from
      the last blank gap ``u_0`` it is ``tau_k + rho^k u_0`` plus ``sum_(j<k) rho^j`` times the
      innovations' level plus a weighted sum of ``k`` innovations, whose variance is
      ``sum_(j<k) rho^(2j)`` times theirs. With the blank periods' innovations' mean ``m`` and
      standard deviation ``s``, the interval is that forecast plus or minus Student's t quantile
      at ``T_blank - 1`` degrees of freedom times ``s`` times the forecast error's scale: exact
      for normal innovations at the true ``rho`` (``validation/scm_ar1_intervals.mac``). A
      persistent error leaves the exchangeable intervals at 70.7% where they ask 80%; on the
      bench's panels, 2000 a case, these covered 79.6%, 79.7% and 79.6% at coefficients 0, 0.5
      and 0.8 at ``alpha = 0.2``, 89.6%, 89.9% and 89.6% at 0.1, and 94.8%, 94.9% and 94.8% at
      0.05, where 5 blank periods leave the exchangeable ones unbounded; with 10 blank periods,
      at 0.8, 79.4%, 89.6% and 94.8%. They assume normal innovations, which the test does not.
    - The intervals' level is the true ``rho``'s. They plug the estimate in as if it were known,
      and its downward bias with few donors enters the forecast through ``rho^k`` and the level's
      sum, so the shortfall grows with the post period. On factor panels whose factors and
      errors are AR(1) at 0.95, 15 fitting, 5 blank and 10 post periods, at ``alpha = 0.1``
      (2000 panels a case, 1000 at 19 donors; one standard error 0.007 and 0.010):

      | donors | mean estimate | coverage at k = 1 .. 10 | at the true ``rho`` |
      |---|---|---|---|
      | 2 | 0.857 | 0.904 falling to 0.854-0.858 | 0.904-0.912 |
      | 4 | 0.893 | 0.892 falling to 0.864-0.870 | 0.886-0.898 |
      | 19 | 0.917 | 0.889 falling to 0.878 | 0.883-0.899 |

      At ``alpha = 0.05`` on 4 donors they covered 0.946 falling to 0.920. Two donors' gaps are one
      series and its negative, each donor fitted by the other, so the estimate reads one series
      however the donors are resampled.
    - ``ar_check_p_value`` checks the AR(1) model on the donors' gaps the coefficient came from.
      Each donor's innovations less their mean are multiplied at lags 1 and 2 and summed over the
      donors period by period, since the donors' common factors correlate them; each lag's total,
      less its mean under the model, over its spread across the periods, squared and summed,
      against a chi-square with 2 degrees of freedom. On the bench's panels it rejected at 5%
      on 2.5%, 3.6% and 4.9% at 0, 0.5 and 0.8; on 600 panels whose errors are an AR(2) with
      coefficients 0.5 and 0.3, on 62%, and on 8% to 12% of those whose factors and errors
      persist at 0.8 and 0 or 0 and 0.8. A rejection says the filter left the innovations
      correlated, so the test's size and the intervals' level may be off; it refuses nothing.

    Raises:
        ValueError: on :func:`synthetic_control`'s refusals; when ``n_blank`` is not in ``[1,
            n_pre - 1]``; when ``alpha`` is outside ``(0, 1)``; when ``serial`` is neither
            reading; when there are more than a million arrangements to enumerate; or, for
            ``"ar1"``, with fewer than two donors or when the donors' gaps cannot give an AR(1)
            coefficient (fewer than four periods after their fit, or none that move).
    """
    if not 0.0 < alpha < 1.0:
        msg = f"alpha must lie in (0, 1), got {alpha}"
        raise ValueError(msg)
    if serial not in ("exchangeable", "ar1"):
        msg = f"serial must be 'exchangeable' or 'ar1', got {serial!r}"
        raise ValueError(msg)
    donor_pre, donor_post, treated_pre, treated_post = _split(outcomes, treated_unit, n_pre)
    if not 1 <= n_blank < n_pre:
        msg = f"n_blank must be in [1, {n_pre - 1}], got {n_blank}"
        raise ValueError(msg)
    n_fit = n_pre - n_blank
    weights = _scm_weights(donor_pre[:, :n_fit], treated_pre[:n_fit], 5000)
    fit_rmspe = float(
        np.sqrt(np.mean((treated_pre[:n_fit] - donor_pre[:, :n_fit].T @ weights) ** 2))
    )
    blank_gaps = treated_pre[n_fit:] - donor_pre[:, n_fit:].T @ weights
    att = treated_post - donor_post.T @ weights
    estimate = SyntheticControlResult(att, float(att.mean()), weights, fit_rmspe)

    n_post = att.size
    n_held = n_blank + n_post
    arrangements = math.comb(n_held, n_post) * (n_post if serial == "ar1" else 1)
    if arrangements > _MAX_ARRANGEMENTS:
        msg = (
            f"{arrangements} arrangements of {n_blank} blank and {n_post} post periods exceed "
            f"{_MAX_ARRANGEMENTS}; leave fewer periods blank"
        )
        raise ValueError(msg)
    chosen = np.fromiter(
        itertools.chain.from_iterable(itertools.combinations(range(n_held), n_post)),
        dtype=np.intp,
        count=math.comb(n_held, n_post) * n_post,
    ).reshape(-1, n_post)
    # the last combination in lexicographic order is the post periods
    ar_coefficient = ar_check_p_value = None
    if serial == "ar1":
        donor_gaps = _donor_gaps(np.concatenate([donor_pre, donor_post], axis=1), n_fit)
        ar_coefficient = _ar1_coefficient(donor_gaps)
        ar_check_p_value = _ar1_check(donor_gaps, ar_coefficient)
        last_fit_gap = treated_pre[n_fit - 1] - donor_pre[:, n_fit - 1] @ weights
        path = np.concatenate([[last_fit_gap], blank_gaps, att])
        innovations = path[1:] - ar_coefficient * path[:-1]
        # a constant effect d adds d to the first post period's innovation and (1 - rho) d to each
        # later one: the statistic weighs them so, each arrangement once for each of its periods
        # that could come first
        totals = innovations[chosen].sum(axis=1)
        stats = np.abs(
            (1.0 - ar_coefficient) * totals[:, None] + ar_coefficient * innovations[chosen]
        )
        observed = stats[-1, 0]
        # a statistic is a sum of n_post + 1 terms each at most this large, so two that are equal
        # in exact arithmetic differ by under (n_post + 2) eps of it
        largest = (abs(1.0 - ar_coefficient) * n_post + abs(ar_coefficient)) * float(
            np.max(np.abs(innovations))
        )
        tolerance = (n_post + 2) * np.finfo(np.float64).eps * largest
    else:
        stats = np.abs(np.concatenate([blank_gaps, att]))[chosen].sum(axis=1)
        observed = stats[-1]
        # two sums of n_post non-negative gaps that are equal in exact arithmetic differ by under
        # n_post * eps of the larger, so a tie counts as one, as the p-value's definition has it
        tolerance = n_post * np.finfo(np.float64).eps * float(stats.max())
    p_value = float(np.mean(stats >= observed - tolerance))

    if ar_coefficient is not None:
        intervals = _ar1_intervals(
            innovations[:n_blank], float(blank_gaps[-1]), att, ar_coefficient, alpha
        )
    else:
        # rounded first, so that (1 - alpha) (n_blank + 1) one ulp above a whole number takes its
        # rank
        rank = math.ceil(round((1.0 - alpha) * (n_blank + 1), 9))
        radius = float(np.sort(np.abs(blank_gaps))[rank - 1]) if rank <= n_blank else math.inf
        intervals = np.column_stack([att - radius, att + radius])
    return BlankPeriodInference(
        estimate,
        blank_gaps,
        p_value,
        intervals,
        alpha,
        arrangements,
        ar_coefficient,
        ar_check_p_value,
    )


def _donor_gaps(donors: Outcomes, n_fit: int) -> Outcomes:
    """The donors' synthetic controls' gaps outside their fit, one row per donor: each donor
    against the others, fitted on the first half of the fitting periods."""
    n_donors, n_periods = donors.shape
    if n_donors < 2:
        msg = "serial='ar1' needs at least two donors: each donor's gaps come from the others"
        raise ValueError(msg)
    n_seen = (n_fit + 1) // 2
    gaps = np.empty((n_donors, n_periods - n_seen))
    for j in range(n_donors):
        others = np.delete(donors, j, axis=0)
        w = _scm_weights(others[:, :n_seen], donors[j, :n_seen], 5000)
        gaps[j] = donors[j, n_seen:] - others[:, n_seen:].T @ w
    return gaps


def _ar1_coefficient(gaps: Outcomes) -> float:
    """The AR(1) coefficient of the donors' gaps by Han, Phillips and Sul's panel fully aggregated
    estimator, kept in ``[-1, 1]``."""
    # y_t - y_s = rho (y_{t-1} - y_{s+1}) + an error uncorrelated with the regressor, for t >= s + 3
    pairs = [(s, t) for s, t in itertools.combinations(range(gaps.shape[1]), 2) if t - s >= 3]
    start, end = np.array(pairs, dtype=np.intp).reshape(-1, 2).T
    regressor = gaps[:, end - 1] - gaps[:, start + 1]
    denominator = float(np.sum(regressor**2))
    if denominator == 0.0:
        msg = (
            f"serial='ar1' cannot read an AR(1) coefficient from the donors' gaps over the last "
            f"{gaps.shape[1]} periods: it needs four that move"
        )
        raise ValueError(msg)
    estimate = float(np.sum(regressor * (gaps[:, end] - gaps[:, start]))) / denominator
    return min(max(estimate, -1.0), 1.0)


# the lags the AR(1) model's residual check reads
_CHECK_LAGS = 2


def _ar1_check(gaps: Outcomes, rho: float) -> float:
    """A portmanteau p-value for the AR(1) model, over the donors' gaps filtered by ``rho``.

    Each donor's innovations ``e_t = y_t - rho y_{t-1}`` are taken less their own mean, which
    removes the donor's level. Under the model the innovation vectors across donors are i.i.d.
    over time, whatever their covariance across donors, which the donors' common factors make
    far from diagonal. So the check sums the products ``e_(j,t) e_(j,t-h)`` over the donors in
    each period, and reads each lag's total against its spread over the periods, not over the
    donors. Demeaning a series of ``n`` i.i.d. terms leaves its lag-``h`` products a mean of
    ``-(n - h) / n`` times the variance, and its squares ``n - 1`` times it, so each total is read
    less ``-(n - h) / n`` times the squares' total over ``n - 1``
    (``validation/scm_ar1_intervals.mac``). The squares of the lags' ratios, summed over lags 1
    and 2, are read against a chi-square with 2 degrees of freedom, whose tail is
    ``exp(-q / 2)``; ``nan`` where a lag has fewer than two periods or no spread."""
    innovations = gaps[:, 1:] - rho * gaps[:, :-1]
    innovations = innovations - innovations.mean(axis=1, keepdims=True)
    n = innovations.shape[1]
    scale = float(np.sum(innovations**2)) / (n - 1)
    q = 0.0
    for h in range(1, _CHECK_LAGS + 1):
        products = np.sum(innovations[:, h:] * innovations[:, :-h], axis=0)
        if products.size < 2:
            return math.nan
        spread = float(np.sum((products - products.mean()) ** 2)) * products.size
        spread /= products.size - 1
        if spread == 0.0:
            return math.nan
        q += (float(products.sum()) + (n - h) / n * scale) ** 2 / spread
    return math.exp(-q / 2.0)


def _ar1_intervals(
    innovations: Vector, last_blank_gap: float, att: Vector, rho: float, alpha: float
) -> NDArray[np.float64]:
    """Each post period's interval under gaps that follow an AR(1) with Gaussian innovations.

    From the last blank gap ``u_0``, ``u_k - rho^k u_0 = tau_k + sum_(j=1..k) rho^(k-j) e~_j``,
    each ``e~_j = u_j - rho u_(j-1)`` an innovation with the gaps' level in it, and the blank
    periods' ``T_B`` innovations are draws of the same law. With ``m`` and ``s`` their mean and
    standard deviation, ``a_k = sum_(j<k) rho^j`` and ``v_k = sum_(j<k) rho^(2j)``,
    ``(u_k - rho^k u_0 - m a_k - tau_k) / (s sqrt(v_k + a_k^2 / T_B))`` is Student's t with
    ``T_B - 1`` degrees of freedom when the innovations are normal: the prediction interval of a
    ``k``-step forecast of an AR(1) with a known coefficient and an unknown level. Infinite with
    fewer than two blank periods."""
    n_blank = innovations.size
    if n_blank < 2:
        return np.full((att.size, 2), [-math.inf, math.inf])
    lags = np.arange(att.size)
    level = np.cumsum(rho**lags)
    spread = np.cumsum(rho ** (2 * lags))
    centre = att - rho ** (lags + 1) * last_blank_gap - float(np.mean(innovations)) * level
    quantile = float(special.stdtrit(n_blank - 1, 1.0 - alpha / 2.0))
    radius = quantile * float(np.std(innovations, ddof=1)) * np.sqrt(spread + level**2 / n_blank)
    return np.column_stack([centre - radius, centre + radius])
