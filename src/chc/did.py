"""Callaway-Sant'Anna staggered-adoption difference-in-differences.

Group-time average treatment effects ``ATT(g, t)`` -- the effect at period ``t`` on the cohort
first treated at period ``g`` -- from clean 2x2 DiD comparisons against never- or not-yet-treated
units. Under staggered adoption with dynamic/heterogeneous effects, two-way fixed effects is biased:
already-treated units enter as controls with negative weights (Goodman-Bacon decomposition), so a
single TWFE coefficient is a contaminated weighted average. These cohort-specific comparisons never
use an already-treated unit as a control, so they are not.

A statistical estimator, so NumPy float64 throughout (like :mod:`chc.independence`) -- independent
of the JAX ``x64`` flag, which must not change an estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Panel = NDArray[np.float64]
Groups = NDArray[np.int64]


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
    outcomes: Panel,
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
    outcomes = np.asarray(outcomes, dtype=np.float64)
    group = np.asarray(group, dtype=np.int64)
    if outcomes.ndim != 2 or group.shape != (outcomes.shape[0],):
        msg = "outcomes must be (N, T) and group (N,) with matching N"
        raise ValueError(msg)
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


def twoway_fixed_effects_att(
    outcomes: Panel, group: Groups, *, never_treated: int = -1
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

    def demean(m: Panel) -> Panel:
        return m - m.mean(axis=1, keepdims=True) - m.mean(axis=0, keepdims=True) + m.mean()

    y_d, d_d = demean(outcomes), demean(treated)
    denom = float(np.sum(d_d * d_d))
    if denom == 0.0:
        msg = "no treatment variation after two-way demeaning"
        raise ValueError(msg)
    return float(np.sum(y_d * d_d) / denom)


def de_chaisemartin(outcomes: Panel, group: Groups, *, never_treated: int = -1) -> float:
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
