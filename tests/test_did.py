"""Callaway-Sant'Anna DiD recovers group-time ATT under staggered effects; TWFE is biased."""

from __future__ import annotations

import numpy as np

from chc.did import callaway_santanna, de_chaisemartin, twoway_fixed_effects_att

DELTA = 0.5  # per-period dynamic effect: ATT(g,t) = DELTA * (t - g + 1) for t >= g


def _staggered_panel(seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Balanced panel: unit + parallel time FE, cohorts at g=3 and g=5, never-treated controls."""
    rng = np.random.default_rng(seed)
    n_per, n_periods = 300, 8
    group = np.array([3] * n_per + [5] * n_per + [-1] * n_per)
    n = group.shape[0]
    unit_fe = rng.normal(0.0, 1.0, (n, 1))
    time_fe = (0.3 * np.arange(n_periods))[None, :]  # common (parallel) trend
    outcomes = unit_fe + time_fe + rng.normal(0.0, 0.3, (n, n_periods))
    for i, g in enumerate(group):
        if g >= 0:
            for t in range(g, n_periods):
                outcomes[i, t] += DELTA * (t - g + 1)  # effect grows with exposure
    true_avg_post = float(
        np.mean([DELTA * (t - g + 1) for i, g in enumerate(group) if g >= 0 for t in range(g, 8)])
    )
    return outcomes, group, true_avg_post


def test_callaway_santanna_recovers_group_time_att() -> None:
    outcomes, group, _ = _staggered_panel(seed=0)
    result = callaway_santanna(outcomes, group)
    for (g, t), estimate in result.att.items():
        true = DELTA * (t - g + 1) if t >= g else 0.0
        assert abs(estimate - true) < 0.1  # each cohort-time effect recovered (placebos too)


def test_event_study_matches_the_dynamic_effect() -> None:
    outcomes, group, _ = _staggered_panel(seed=1)
    event_study = callaway_santanna(outcomes, group).event_study
    for e, estimate in event_study.items():
        if e >= 0:
            assert abs(estimate - DELTA * (e + 1)) < 0.1  # dynamic ATT(e) tracks DELTA*(e+1)
        else:
            assert abs(estimate) < 0.1  # pre-treatment placebo


def test_pretrend_placebo_is_near_zero() -> None:
    outcomes, group, _ = _staggered_panel(seed=2)
    pretrend = callaway_santanna(outcomes, group).pretrend()
    assert pretrend  # there are pre-treatment (g,t) cells
    assert max(abs(v) for v in pretrend.values()) < 0.1  # parallel trends hold in the placebo


def test_never_and_notyet_controls_agree() -> None:
    outcomes, group, _ = _staggered_panel(seed=3)
    never = callaway_santanna(outcomes, group, control="never").overall
    notyet = callaway_santanna(outcomes, group, control="notyet").overall
    assert abs(never - notyet) < 0.05  # both clean-control schemes give the same overall ATT


def test_callaway_santanna_beats_biased_twoway_fixed_effects() -> None:
    outcomes, group, true_avg_post = _staggered_panel(seed=4)
    cs_overall = callaway_santanna(outcomes, group).overall
    twfe = twoway_fixed_effects_att(outcomes, group)
    assert abs(cs_overall - true_avg_post) < 0.05  # CS recovers the true average post effect
    assert abs(twfe - true_avg_post) > 0.15  # TWFE contaminated by forbidden controls
    assert abs(cs_overall - true_avg_post) < abs(twfe - true_avg_post)  # CS strictly less biased


def test_de_chaisemartin_recovers_the_instantaneous_effect() -> None:
    outcomes, group, _ = _staggered_panel(seed=5)
    instantaneous = de_chaisemartin(outcomes, group)
    assert abs(instantaneous - DELTA) < 0.1  # DID_M targets the e=0 first-exposure effect (DELTA)
