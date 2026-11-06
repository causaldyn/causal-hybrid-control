"""Callaway-Sant'Anna DiD recovers group-time ATT under staggered effects; TWFE is biased."""

from __future__ import annotations

import numpy as np
import pytest

from chc.did import (
    callaway_santanna,
    callaway_santanna_inference,
    de_chaisemartin,
    twoway_fixed_effects_att,
)

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


@pytest.mark.parametrize("estimator", [twoway_fixed_effects_att, de_chaisemartin])
def test_an_outcome_that_is_no_number_is_refused(estimator) -> None:
    """Neither estimator read the panel through its check, so one nan outcome made the effect
    nan."""
    outcomes, group, _ = _staggered_panel(0)
    outcomes[0, 4] = np.nan
    with pytest.raises(ValueError, match="outcomes must be finite"):
        estimator(outcomes, group)


def _overall(outcomes, group, **kwargs) -> float:
    return callaway_santanna(outcomes, group, **kwargs).overall


@pytest.mark.parametrize("estimator", [twoway_fixed_effects_att, de_chaisemartin, _overall])
@pytest.mark.parametrize(("label", "row"), [(-2, 600), (8, 601), (11, 899)])
def test_a_group_outside_the_panel_is_refused_naming_its_row(estimator, label, row) -> None:
    """A label of -2 marked a unit's last two periods treated in TWFE (0.98 -> 0.80 on a panel
    of effect 1) and dropped it from CS's controls; a first-treated period past the panel crashed
    CS with an IndexError, where TWFE and DID_M read it as never treated."""
    outcomes, group, _ = _staggered_panel(0)
    group[row] = label
    with pytest.raises(
        ValueError, match=rf"group is {label} at row {row}, outside the panel's 0\.\.7"
    ):
        estimator(outcomes, group)


def test_another_never_treated_label_is_read_as_never_treated() -> None:
    outcomes, group, _ = _staggered_panel(0)
    relabelled = np.where(group == -1, 99, group)
    assert _overall(outcomes, relabelled, never_treated=99) == _overall(outcomes, group)
    assert twoway_fixed_effects_att(outcomes, relabelled, never_treated=99) == (
        twoway_fixed_effects_att(outcomes, group)
    )


def _inference(outcomes, group, **kwargs):
    return callaway_santanna_inference(outcomes, group, draws=99, **kwargs)


@pytest.mark.parametrize("estimator", [callaway_santanna, _inference])
@pytest.mark.parametrize("control", ["nevr", "never_treated", "not_yet_treated"])
def test_a_control_neither_estimator_knows_is_refused(estimator, control) -> None:
    """Any string but ``"never"`` was read as ``"notyet"`` by the estimate, and the inference
    raised a bare ``KeyError`` on it."""
    outcomes, group, _ = _staggered_panel(0)
    with pytest.raises(ValueError, match=rf"control is '{control}': 'notyet' compares"):
        estimator(outcomes, group, control=control)


@pytest.mark.parametrize(
    "estimator", [twoway_fixed_effects_att, de_chaisemartin, _overall, _inference]
)
@pytest.mark.parametrize("as_objects", [False, True], ids=["bool array", "bools as objects"])
def test_a_group_of_booleans_is_refused(estimator, as_objects) -> None:
    """A flag of ever treated was read as cohorts 0 and 1: CS returned nan, TWFE 0.24 on a panel of
    effect 1, and the others refused it for another reason."""
    outcomes, group, _ = _staggered_panel(0)
    flag = group >= 0
    with pytest.raises(ValueError, match=r"says? whether a unit is treated and not when"):
        estimator(outcomes, flag.astype(object) if as_objects else flag)


@pytest.mark.parametrize(
    "estimator", [twoway_fixed_effects_att, de_chaisemartin, _overall, _inference]
)
@pytest.mark.parametrize("never_treated", [1, 5, 7])
def test_a_never_treated_label_inside_the_panel_is_refused(estimator, never_treated) -> None:
    """A cohort first treated at the label's period read as never treated: on a panel with
    cohorts at 2 and 4 of five periods, ``never_treated=4`` returned the cohort at 2 alone."""
    outcomes, group, _ = _staggered_panel(0)
    with pytest.raises(ValueError, match=rf"never_treated is {never_treated}, a period"):
        estimator(outcomes, group, never_treated=never_treated)


def test_never_treated_units_may_be_labelled_zero() -> None:
    """R's ``did`` and diff-diff's coding, where no unit is treated from the first period."""
    outcomes, group, _ = _staggered_panel(0)
    zero = np.where(group == -1, 0, group)
    assert callaway_santanna(outcomes, zero, never_treated=0) == callaway_santanna(outcomes, group)
    for estimator in (twoway_fixed_effects_att, de_chaisemartin):
        assert estimator(outcomes, zero, never_treated=0) == estimator(outcomes, group)
