"""callaway_santanna_inference: diff-diff's bootstrap around callaway_santanna's own estimates."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from typing import Literal

import numpy as np
import pytest

import chc.did
from chc.did import GroupTimeATT, callaway_santanna, callaway_santanna_inference

N_PERIODS = 8
Z = 1.959963984540054
# the did extra installs nothing on Python 3.15, and diff-diff ships no free-threaded wheels
needs_diff_diff = pytest.mark.skipif(
    importlib.util.find_spec("diff_diff") is None, reason="needs the did extra"
)


def _effect(g: int, t: int) -> float:
    return 0.0 if t < g else (1.0 + 0.25 * (t - g)) * (1.5 if g == 3 else 1.0)


def _panel(
    seed: int,
    cohorts: tuple[int, ...] = (3, 5, -1),
    per: int = 40,
    rho: float = 0.5,
    *,
    frozen: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Unit and time effects, AR(1) errors within a unit, and a dynamic, cohort-specific effect.

    ``frozen`` draws from ``RandomState``, whose stream numpy keeps fixed across releases, for a
    panel whose numbers were computed elsewhere."""
    rng = np.random.RandomState(seed) if frozen else np.random.default_rng(seed)
    group = np.repeat(np.array(cohorts), per)
    noise = np.empty((group.size, N_PERIODS))
    noise[:, 0] = rng.normal(size=group.size) / np.sqrt(1.0 - rho**2)
    for t in range(1, N_PERIODS):
        noise[:, t] = rho * noise[:, t - 1] + rng.normal(size=group.size)
    level = rng.normal(size=(group.size, 1)) + 0.1 * np.maximum(group, 0)[:, None]
    outcomes = level + 0.3 * np.arange(N_PERIODS)[None, :] + noise
    for i, g in enumerate(group):
        outcomes[i] += [_effect(int(g), t) if g >= 0 else 0.0 for t in range(N_PERIODS)]
    return outcomes, group


def _overall(cohorts: tuple[int, ...]) -> float:
    """The overall ATT at equal cohort sizes: every cell after treatment, weighted alike."""
    cells = [(g, t) for g in cohorts if g > 0 for t in range(g, N_PERIODS)]
    return sum(_effect(g, t) for g, t in cells) / len(cells)


@needs_diff_diff
@pytest.mark.parametrize("control", ["notyet", "never"])
def test_the_inference_is_centred_on_callaway_santannas_own_estimates(
    control: Literal["notyet", "never"],
) -> None:
    # a cohort treated from the first period has no cell and is no control, in either library
    outcomes, group = _panel(0, cohorts=(0, 3, 5, -1))
    inference = callaway_santanna_inference(outcomes, group, control=control, draws=199)
    assert inference.estimate == callaway_santanna(outcomes, group, control=control)
    assert set(inference.se) == set(inference.estimate.event_study) - {-1}
    assert set(inference.band) == set(inference.se)


@needs_diff_diff
def test_units_treated_from_the_first_period_change_nothing() -> None:
    """They have no base period, so no cell and no place among the controls; R's did drops them."""
    outcomes, group = _panel(10)
    always, _ = _panel(11, cohorts=(0,), per=15)
    with_them = callaway_santanna_inference(
        np.vstack([outcomes, always]), np.concatenate([group, np.zeros(15, np.int64)]), draws=199
    )
    without = callaway_santanna_inference(outcomes, group, draws=199)
    assert (with_them.se, with_them.band, with_them.overall_interval) == (
        without.se,
        without.band,
        without.overall_interval,
    )


@needs_diff_diff
def test_a_panel_without_never_treated_units_leaves_its_late_cells_out_in_both() -> None:
    outcomes, group = _panel(1, cohorts=(3, 5, 7))
    inference = callaway_santanna_inference(outcomes, group, draws=199)
    assert (3, 7) not in inference.estimate.att
    assert all(np.isfinite(s) for s in inference.se.values())


@needs_diff_diff
def test_the_uniform_band_is_wider_than_every_pointwise_interval() -> None:
    inference = callaway_santanna_inference(*_panel(2), draws=499)
    for e, (lo, hi) in inference.band.items():
        centre = inference.estimate.event_study[e]
        assert lo < centre - Z * inference.se[e]
        assert hi > centre + Z * inference.se[e]


@needs_diff_diff
def test_a_seed_reproduces_the_bootstrap_and_another_moves_it() -> None:
    outcomes, group = _panel(3)
    first = callaway_santanna_inference(outcomes, group, draws=199, seed=7)
    again = callaway_santanna_inference(outcomes, group, draws=199, seed=7)
    other = callaway_santanna_inference(outcomes, group, draws=199, seed=8)
    assert (first.se, first.band, first.overall_interval) == (
        again.se,
        again.band,
        again.overall_interval,
    )
    assert first.se != other.se


@needs_diff_diff
def test_the_standard_error_is_the_estimates_spread_over_panels() -> None:
    """(estimate - truth) / se is standard normal over panels: its square averages one.

    Sixty panels put that mean within 0.6-1.6 with probability above 0.99; a standard error half
    or twice the right one puts it near 4 or 0.25."""
    truth = _overall((3, 5, -1))
    squares = []
    for seed in range(60):
        inference = callaway_santanna_inference(*_panel(100 + seed), draws=199, seed=seed)
        squares.append(((inference.estimate.overall - truth) / inference.overall_se) ** 2)
    assert 0.6 < float(np.mean(squares)) < 1.6


@needs_diff_diff
def test_diff_diffs_estimates_must_be_callaway_santannas(monkeypatch: pytest.MonkeyPatch) -> None:
    def shifted(
        outcomes: np.ndarray,
        group: np.ndarray,
        *,
        control: Literal["notyet", "never"] = "notyet",
        never_treated: int = -1,
    ) -> GroupTimeATT:
        estimate = callaway_santanna(outcomes, group, control=control, never_treated=never_treated)
        cell = next(iter(estimate.att))
        return replace(estimate, att={**estimate.att, cell: estimate.att[cell] + 1e-6})

    monkeypatch.setattr(chc.did, "callaway_santanna", shifted)
    with pytest.raises(RuntimeError, match="diff-diff's ATT"):
        callaway_santanna_inference(*_panel(4), draws=199)


def test_it_refuses_a_level_or_a_bootstrap_it_cannot_read() -> None:
    outcomes, group = _panel(5)
    for alpha in (0.0, 1.0):
        with pytest.raises(ValueError, match="alpha must lie in"):
            callaway_santanna_inference(outcomes, group, alpha=alpha)
    with pytest.raises(ValueError, match="draws must be at least 2"):
        callaway_santanna_inference(outcomes, group, draws=1)


def test_it_refuses_a_first_treated_period_outside_the_panel() -> None:
    outcomes, group = _panel(6)
    with pytest.raises(ValueError, match=r"outside the panel's 0\.\.7: \[9\]"):
        callaway_santanna_inference(outcomes, np.where(group == 5, 9, group))


def test_it_refuses_a_panel_with_no_cell_after_treatment() -> None:
    outcomes, group = _panel(7, cohorts=(0, -1))
    with pytest.raises(ValueError, match="nothing to infer"):
        callaway_santanna_inference(outcomes, group)


def test_without_the_extra_it_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "diff_diff", None)
    with pytest.raises(ImportError, match=r"causal-hybrid-control\[did\]"):
        callaway_santanna_inference(*_panel(8))


@needs_diff_diff
def test_the_robust_interval_widens_as_the_trends_may_part_further() -> None:
    inference = callaway_santanna_inference(*_panel(12, per=100), draws=199)
    narrow, wider, widest = (inference.robust_interval(m) for m in (0.0, 0.05, 0.2))
    assert widest[0] < wider[0] < narrow[0] < narrow[1] < wider[1] < widest[1]


@needs_diff_diff
def test_a_linear_trend_the_event_study_misses_is_inside_the_robust_interval() -> None:
    """The treated drift 0.15 a period away from the comparison: the event study's average
    after treatment is biased by 0.15 (e + 1) averaged over e, and ``m = 0`` extrapolates the
    line the periods before treatment show."""
    outcomes, group = _panel(13, per=150)
    outcomes = outcomes + 0.15 * np.arange(N_PERIODS) * (group >= 0)[:, None]
    inference = callaway_santanna_inference(outcomes, group, control="never", draws=199)
    post = [e for e in inference.estimate.event_study if e >= 0]
    truth = float(
        np.mean([np.mean([_effect(g, g + e) for g in (3, 5) if g + e < N_PERIODS]) for e in post])
    )
    naive = float(np.mean([inference.estimate.event_study[e] for e in post]))
    lo, hi = inference.robust_interval(0.0)
    assert naive - truth > 0.4
    assert lo <= truth <= hi


@needs_diff_diff
def test_the_robust_interval_is_rs_honestdid_on_the_same_panel() -> None:
    """R's did 2.5.1 and HonestDiD 0.2.8 on this panel: the smoothness interval for the event
    study's average effect after treatment, from did's event study and the covariance of its
    influence function. The two optimisers agree to 6e-4 here; read off the bootstrapped event
    study's standard errors alone, the interval at ``m = 0`` is at least 0.13 off."""
    inference = callaway_santanna_inference(*_panel(20260930, per=60, frozen=True), draws=199)
    for m, expected in ((0.0, (1.763756, 2.778221)), (0.1, (0.918917, 3.499549))):
        np.testing.assert_allclose(inference.robust_interval(m), expected, atol=5e-3)


@needs_diff_diff
def test_the_robust_interval_refuses_a_negative_bound() -> None:
    inference = callaway_santanna_inference(*_panel(14), draws=199)
    with pytest.raises(ValueError, match="m must be at least 0"):
        inference.robust_interval(-0.1)
