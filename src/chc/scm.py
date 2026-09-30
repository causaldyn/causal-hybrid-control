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
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls

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
    outcomes = np.asarray(outcomes, dtype=np.float64)
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
    outcomes: Outcomes, treated_unit: int, n_pre: int, *, ridge: float = 1.0, steps: int = 5000
) -> SyntheticControlResult:
    """Ridge-augmented synthetic control (Ben-Michael-Feller-Rothstein).

    Starts from the SCM donor weights, then de-biases each post period by the residual pre-period
    imbalance passed through a ridge outcome model fit on the donors:
    ``Y_1(0)_t = w' Y_post_t + (treated_pre - w' donor_pre)' theta_t`` with
    ``theta_t = (Z'Z + ridge*I)^{-1} Z' Y_post_t`` and ``Z`` the donor pre-period matrix. The
    correction vanishes when SCM already balances the pre-period; ``ridge`` controls how far the
    outcome model may extrapolate. ``weights`` are the (interpretable) SCM donor weights; the
    augmentation is an additive outcome correction, not folded into them.
    """
    donor_pre, donor_post, treated_pre, treated_post = _split(outcomes, treated_unit, n_pre)
    w = _scm_weights(donor_pre, treated_pre, steps)
    imbalance = treated_pre - donor_pre.T @ w  # (T0,) pre-period residual SCM cannot balance
    pre_rmspe = float(np.sqrt(np.mean(imbalance**2)))
    n_pre_periods = donor_pre.shape[1]
    gram = donor_pre.T @ donor_pre + ridge * np.eye(n_pre_periods)  # (T0, T0)
    theta = np.linalg.solve(gram, donor_pre.T @ donor_post)  # (T0,T1): one ridge model per period
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
