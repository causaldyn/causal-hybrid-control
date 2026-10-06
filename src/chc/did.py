"""Callaway-Sant'Anna staggered-adoption difference-in-differences.

Group-time average treatment effects ``ATT(g, t)`` -- the effect at period ``t`` on the cohort
first treated at period ``g`` -- from clean 2x2 DiD comparisons against never- or not-yet-treated
units. Under staggered adoption with dynamic/heterogeneous effects, two-way fixed effects is biased:
already-treated units enter as controls with negative weights (Goodman-Bacon decomposition), so a
single TWFE coefficient is a contaminated weighted average. These cohort-specific comparisons never
use an already-treated unit as a control, so they are not.

A statistical estimator, so NumPy float64 throughout (like :mod:`chc.independence`) -- independent
of the JAX ``x64`` flag, which must not change an estimate.

:func:`callaway_santanna_inference` adds the estimates' uncertainty through ``diff-diff``, the
``did`` extra, whose Callaway-Sant'Anna reproduces these estimates.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

Outcomes = NDArray[np.float64]
Groups = NDArray[np.int64]

_DID_HINT = (
    "callaway_santanna_inference needs the 'did' extra: pip install 'causal-hybrid-control[did]'. "
    "Its diff-diff supports Python below 3.15, where the extra installs nothing."
)
# diff-diff warns of the cells a design leaves empty: the base period's, which is zero by
# construction, and those with no clean control. callaway_santanna leaves both out without a word.
_EMPTY_CELLS = (
    r".*had non-finite or zero bootstrap SE.*",
    r".*cell\(s\) could not be estimated.*",
)
# agreement between the two implementations, relative to the largest effect
_AGREE = 1e-9


@dataclass(frozen=True)
class GroupTimeATT:
    """Callaway-Sant'Anna output: cohort ``ATT(g,t)``, the event-study curve, and overall ATT."""

    att: dict[tuple[int, int], float]
    event_study: dict[int, float]
    overall: float
    groups: tuple[int, ...]
    n_periods: int

    def pretrend(self) -> dict[tuple[int, int], float]:
        """Pre-treatment ``ATT(g,t)`` for ``t < g`` -- a placebo; ~0 supports parallel trends."""
        return {k: v for k, v in self.att.items() if k[1] < k[0]}


def callaway_santanna(
    outcomes: Outcomes,
    group: Groups,
    *,
    control: Literal["notyet", "never"] = "notyet",
    never_treated: int = -1,
) -> GroupTimeATT:
    """Group-time ``ATT(g,t)`` for a balanced panel via never-/not-yet-treated 2x2 DiD.

    ``outcomes`` is ``(N, T)`` with ``outcomes[i, t]`` unit ``i``'s outcome at period ``t``;
    ``group`` is ``(N,)`` with each unit's first-treated period (0-indexed) or ``never_treated``.
    ``control="notyet"`` compares against units not yet treated by ``t`` (never-treated included --
    more controls, de Chaisemartin-d'Haultfoeuille-robust); ``"never"`` uses only never-treated.

    Universal base period ``g - 1``:
    ``ATT(g,t) = E[Y_t - Y_{g-1} | G=g] - E[Y_t - Y_{g-1} | control]``. ``event_study`` aggregates
    by relative time ``e = t - g``; ``overall`` is the size-weighted mean of post-treatment effects.
    """
    outcomes, group = _panel(outcomes, group)
    n_periods = int(outcomes.shape[1])
    treated_groups = tuple(sorted({int(g) for g in group.tolist() if g != never_treated}))
    sizes = {g: int(np.sum(group == g)) for g in treated_groups}

    att: dict[tuple[int, int], float] = {}
    for g in treated_groups:
        base = g - 1
        if base < 0:
            continue  # a first-period adopter has no clean pre-treatment period
        treated = group == g
        for t in range(n_periods):
            if t == base:
                att[(g, t)] = 0.0  # normalisation point
                continue
            if control == "never":
                ctrl = group == never_treated
            else:  # not-yet-treated: untreated at base and t (never-treated always qualify)
                ctrl = (group != g) & ((group == never_treated) | (group > max(t, base)))
            if not treated.any() or not ctrl.any():
                continue
            d_treat = outcomes[treated, t] - outcomes[treated, base]
            d_ctrl = outcomes[ctrl, t] - outcomes[ctrl, base]
            att[(g, t)] = float(d_treat.mean() - d_ctrl.mean())

    ev_num: dict[int, float] = {}
    ev_den: dict[int, float] = {}
    post_num = post_den = 0.0
    for (g, t), value in att.items():
        e = t - g
        weight = sizes[g]
        ev_num[e] = ev_num.get(e, 0.0) + weight * value
        ev_den[e] = ev_den.get(e, 0.0) + weight
        if t >= g:
            post_num += weight * value
            post_den += weight
    event_study = {e: ev_num[e] / ev_den[e] for e in sorted(ev_num)}
    overall = post_num / post_den if post_den else float("nan")
    return GroupTimeATT(att, event_study, overall, treated_groups, n_periods)


def _panel(outcomes: Outcomes, group: Groups) -> tuple[Outcomes, Groups]:
    outcomes = np.asarray(outcomes, dtype=np.float64)
    group = np.asarray(group, dtype=np.int64)
    if outcomes.ndim != 2 or group.shape != (outcomes.shape[0],):
        msg = "outcomes must be (N, T) and group (N,) with matching N"
        raise ValueError(msg)
    if not np.isfinite(outcomes).all():
        # a cell's mean would carry a nan, where diff-diff drops the unit-period and reads the rest
        msg = "outcomes must be finite: the estimator reads a balanced panel"
        raise ValueError(msg)
    return outcomes, group


@dataclass(frozen=True)
class EventStudyInference:
    """:func:`callaway_santanna`'s estimates, with their uncertainty. *Experimental.*

    ``estimate`` is :func:`callaway_santanna`'s own. The rest is diff-diff's multiplier bootstrap,
    which draws whole units, so it holds however a unit's errors are correlated over time. ``se``
    and ``band`` are by relative time ``e``, without the base period ``e = -1``, where the event
    study is zero by construction. ``band`` holds every ``e`` at once with probability
    ``1 - alpha``: the bootstrap's sup-t band, where a pointwise interval is
    ``event_study[e] +- z se[e]``. ``overall_interval`` is the overall ATT's.
    """

    estimate: GroupTimeATT
    se: dict[int, float]
    band: dict[int, tuple[float, float]]
    overall_se: float
    overall_interval: tuple[float, float]
    alpha: float
    _analytic: Any = field(repr=False, compare=False)

    def robust_interval(self, m: float) -> tuple[float, float]:
        """The average effect after treatment, robust to trends that part smoothly.

        Rambachan and Roth's smoothness restriction: the treated units' trend may differ from the
        comparison's by a line whose slope changes by at most ``m`` a period, before treatment and
        after. ``m = 0`` allows exactly the linear difference the periods before treatment can
        show. The target is the event study's effects after treatment averaged with equal weights,
        which is not ``estimate.overall``'s weighting. The interval is its optimal fixed-length
        interval at level ``1 - alpha``, from diff-diff's HonestDiD on the event study's analytic
        covariance; it agrees with R's HonestDiD to ``1e-3``. At ``m = 0``, with trends parallel or
        apart by a line, it covered 0.95 at 300 units, 0.92 at 100 and 0.78-0.79 at 30.

        Raises:
            ValueError: for a negative ``m``.
        """
        if not m >= 0.0:
            msg = f"m must be at least 0, got {m}"
            raise ValueError(msg)
        import diff_diff

        robust = diff_diff.HonestDiD(method="smoothness", M=m, alpha=self.alpha).fit(self._analytic)
        return float(robust.ci_lb), float(robust.ci_ub)


def callaway_santanna_inference(
    outcomes: Outcomes,
    group: Groups,
    *,
    control: Literal["notyet", "never"] = "notyet",
    never_treated: int = -1,
    alpha: float = 0.05,
    draws: int = 999,
    seed: int = 0,
) -> EventStudyInference:
    """:func:`callaway_santanna` with standard errors, intervals and a uniform band.

    Experimental: it may change or be withdrawn in any release.

    Takes :func:`callaway_santanna`'s arguments, and computes its estimates. diff-diff's
    Callaway-Sant'Anna, at the same comparison and base period ``g - 1``, reproduces them; they
    are checked to agree before its bootstrap is read, so every interval is centred on this
    library's estimate. ``draws`` bootstrap draws are taken with ``seed``.

    The levels hold with enough units, and not below. On panels of ten periods, three cohorts and
    never-treated units, with errors correlated within a unit, at 5%: the overall ATT's interval
    rejected a true null 5.7-6.6% of the time from 100 units, and 6.9-7.8% at 30; the band
    6.4-7.2% at 300 units, 7.9-8.7% at 100, and 21-24% at 30.

    Needs the ``did`` extra (``pip install 'causal-hybrid-control[did]'``).

    Raises:
        ValueError: on :func:`callaway_santanna`'s refusals; a first-treated period outside the
            panel, where a unit first treated after it is never treated within it; a panel with
            no cell after treatment; ``alpha`` outside ``(0, 1)``; fewer than two draws.
        RuntimeError: when diff-diff's estimates do not reproduce these, which would put the
            intervals around another estimate.
    """
    if not 0.0 < alpha < 1.0:
        msg = f"alpha must lie in (0, 1), got {alpha}"
        raise ValueError(msg)
    if draws < 2:
        msg = f"draws must be at least 2, got {draws}"
        raise ValueError(msg)
    outcomes, group = _panel(outcomes, group)
    n_periods = int(outcomes.shape[1])
    outside = sorted(
        {int(g) for g in group.tolist() if g != never_treated and not 0 <= g < n_periods}
    )
    if outside:
        msg = f"group holds first-treated periods outside the panel's 0..{n_periods - 1}: {outside}"
        raise ValueError(msg)
    estimate = callaway_santanna(outcomes, group, control=control, never_treated=never_treated)
    if math.isnan(estimate.overall):
        msg = "no cohort has a cell after its treatment with a clean control: nothing to infer"
        raise ValueError(msg)
    try:
        import diff_diff
        import pandas as pd
    except ImportError as exc:
        raise ImportError(_DID_HINT) from exc

    # a unit treated from the first period has no base period: no cell of its own, and no control
    kept = np.flatnonzero(group != 0)
    frame = pd.DataFrame(
        {
            "unit": np.repeat(kept, n_periods),
            # diff-diff reads a first-treated period of 0 as never treated, so periods count from 1
            "time": np.tile(np.arange(1, n_periods + 1), kept.size),
            "outcome": outcomes[kept].ravel(),
            "first_treat": np.repeat(
                np.where(group[kept] == never_treated, 0, group[kept] + 1), n_periods
            ),
        }
    )
    comparison = {"notyet": "not_yet_treated", "never": "never_treated"}[control]
    fit, study = _fit(diff_diff, frame, comparison, alpha, draws, seed)
    # a bootstrapped event study carries no covariance, and HonestDiD then reads its errors as
    # independent; the analytic fit's covariance is the one it needs
    analytic_fit, analytic = _fit(diff_diff, frame, comparison, alpha, 0, seed)
    times = [int(e) for e in study.event_time]
    for fitted, container in ((fit, study), (analytic_fit, analytic)):
        cells = {
            (int(g) - 1, int(t) - 1): float(cell["effect"])
            for (g, t), cell in fitted.group_time_effects.items()
            if math.isfinite(cell["effect"])
        }
        studied = {
            int(e): float(a) for e, a in zip(container.event_time, container.att, strict=True)
        }
        _agree(estimate, cells, studied, fitted.overall_att)
    if analytic.vcov is None or np.shape(analytic.vcov) != (len(times) - 1,) * 2:
        msg = "diff-diff's analytic event study carries no covariance of its effects"
        raise RuntimeError(msg)
    se, band = {}, {}
    for e, is_base, s, lo, hi in zip(
        times, study.is_reference, study.se, study.cband_lower, study.cband_upper, strict=True
    ):
        if not is_base:
            se[e] = float(s)
            band[e] = (float(lo), float(hi))
    lo, hi = fit.overall_conf_int
    return EventStudyInference(
        estimate, se, band, float(fit.overall_se), (float(lo), float(hi)), alpha, analytic
    )


def _fit(
    diff_diff: Any, frame: Any, comparison: str, alpha: float, draws: int, seed: int
) -> tuple[Any, Any]:
    """diff-diff's Callaway-Sant'Anna at base period ``g - 1``, and its event study."""
    with warnings.catch_warnings():
        for message in _EMPTY_CELLS:
            warnings.filterwarnings("ignore", message=message)
        fit = diff_diff.CallawaySantAnna(
            control_group=comparison,
            base_period="universal",
            alpha=alpha,
            n_bootstrap=draws,
            seed=seed,
        ).fit(frame, outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        return fit, fit.aggregate("event_study")


def _agree(
    estimate: GroupTimeATT,
    cells: dict[tuple[int, int], float],
    event_study: dict[int, float],
    overall: float,
) -> None:
    """Raise unless diff-diff's cells, event study and overall ATT are ``estimate``'s own."""
    scale = max([1.0, *(abs(v) for v in estimate.att.values())])
    tolerance = _AGREE * scale
    if set(cells) != set(estimate.att):
        msg = (
            "diff-diff estimated other group-time cells than callaway_santanna: "
            f"only diff-diff {sorted(set(cells) - set(estimate.att))}, "
            f"only callaway_santanna {sorted(set(estimate.att) - set(cells))}"
        )
        raise RuntimeError(msg)
    # each test holds only on a number within the tolerance: a nan, which every comparison reads
    # as false, does not pass
    apart = [k for k in estimate.att if not abs(cells[k] - estimate.att[k]) <= tolerance]
    if apart:
        worst = max(apart, key=lambda k: np.nan_to_num(abs(cells[k] - estimate.att[k]), nan=np.inf))
        msg = (
            f"diff-diff's ATT{worst} is {cells[worst]!r}, callaway_santanna's "
            f"{estimate.att[worst]!r}"
        )
        raise RuntimeError(msg)
    if set(event_study) != set(estimate.event_study) or not all(
        abs(event_study[e] - estimate.event_study[e]) <= tolerance for e in event_study
    ):
        msg = "diff-diff's event study is not callaway_santanna's"
        raise RuntimeError(msg)
    if not abs(overall - estimate.overall) <= tolerance:
        msg = f"diff-diff's overall ATT is {overall!r}, callaway_santanna's {estimate.overall!r}"
        raise RuntimeError(msg)


def twoway_fixed_effects_att(
    outcomes: Outcomes, group: Groups, *, never_treated: int = -1
) -> float:
    """The single two-way fixed-effects treatment coefficient -- the biased baseline CS beats.

    Regresses the twice-demeaned outcome on the twice-demeaned treatment indicator
    ``D[i,t] = 1{t >= group[i]}``. Under staggered timing with dynamic effects this is a
    negative-weighted average of the ``ATT(g,t)`` (Goodman-Bacon), not the average effect.
    """
    outcomes = np.asarray(outcomes, dtype=np.float64)
    group = np.asarray(group, dtype=np.int64)
    treated = np.zeros_like(outcomes)
    for i, g in enumerate(group.tolist()):
        if g != never_treated:
            treated[i, g:] = 1.0

    def demean(m: Outcomes) -> Outcomes:
        return m - m.mean(axis=1, keepdims=True) - m.mean(axis=0, keepdims=True) + m.mean()

    y_d, d_d = demean(outcomes), demean(treated)
    denom = float(np.sum(d_d * d_d))
    if denom == 0.0:
        msg = "no treatment variation after two-way demeaning"
        raise ValueError(msg)
    return float(np.sum(y_d * d_d) / denom)


def de_chaisemartin(outcomes: Outcomes, group: Groups, *, never_treated: int = -1) -> float:
    """de Chaisemartin-d'Haultfoeuille DID_M -- the average instantaneous (first-exposure) effect.

    A switcher-count-weighted average, over consecutive-period 2x2 DiDs, of the outcome change of
    units first treated at ``t`` minus that of units still untreated at ``t`` (untreated stayers).
    Like :func:`callaway_santanna` it is heterogeneity-robust where TWFE is not, but its estimand is
    the effect at the moment of switching (relative time ``e = 0``), not the size-weighted average
    over post periods -- a growing effect yields the first-period impact, not the overall ATT.
    """
    outcomes = np.asarray(outcomes, dtype=np.float64)
    group = np.asarray(group, dtype=np.int64)
    n_periods = int(outcomes.shape[1])
    numerator = denominator = 0.0
    for t in range(1, n_periods):
        switchers = group == t  # untreated at t-1, treated at t (a 0 -> 1 switch)
        stayers = (group == never_treated) | (group > t)  # untreated at both t-1 and t
        if switchers.any() and stayers.any():
            change_switch = float((outcomes[switchers, t] - outcomes[switchers, t - 1]).mean())
            change_stay = float((outcomes[stayers, t] - outcomes[stayers, t - 1]).mean())
            weight = int(switchers.sum())
            numerator += weight * (change_switch - change_stay)
            denominator += weight
    if denominator == 0.0:
        msg = "no treatment switches found in the panel"
        raise ValueError(msg)
    return numerator / denominator
