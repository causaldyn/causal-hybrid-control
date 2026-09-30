"""synthetic_control_inference: the placebo test against the count that makes it exact, and the
conformal test against diff-diff's."""

from __future__ import annotations

import math

import numpy as np
import pytest

from chc.scm import _conformal_p_value, _hull, synthetic_control, synthetic_control_inference

N_PRE = 24
N_POST = 6


def _panel(seed: int, units: int = 20, effect: float = 0.0) -> np.ndarray:
    """Three AR(1) factors with loadings in [0, 1], unit levels, AR(1) errors; unit 0 is treated,
    with a constant effect after ``N_PRE``."""
    rng = np.random.default_rng(seed)
    periods = N_PRE + N_POST
    factors = np.empty((periods, 3))
    errors = np.empty((units, periods))
    factors[0] = rng.normal(size=3) / np.sqrt(0.75)
    errors[:, 0] = rng.normal(size=units) / np.sqrt(0.75)
    for t in range(1, periods):
        factors[t] = 0.5 * factors[t - 1] + rng.normal(size=3)
        errors[:, t] = 0.5 * errors[:, t - 1] + rng.normal(size=units)
    loadings = rng.uniform(0.0, 1.0, (units, 3))
    outcomes = loadings @ factors.T + 0.5 * errors + rng.normal(size=(units, 1))
    outcomes[0, N_PRE:] += effect
    return outcomes


def _diff_diff(outcomes: np.ndarray):
    """diff-diff's synthetic control for unit 0 on the same problem: the pre-period outcomes, alike
    weighted and unscaled, with its Frank-Wolfe run to convergence."""
    diff_diff = pytest.importorskip("diff_diff")
    pd = pytest.importorskip("pandas")
    n_units, n_periods = outcomes.shape
    treated = np.zeros((n_units, n_periods), dtype=np.int64)
    treated[0, N_PRE:] = 1
    frame = pd.DataFrame(
        {
            "unit": np.repeat(np.arange(n_units), n_periods),
            "time": np.tile(np.arange(n_periods), n_units),
            "outcome": outcomes.ravel(),
            "treated": treated.ravel(),
        }
    )
    model = diff_diff.SyntheticControl(
        v_method="custom",
        custom_v=np.ones(N_PRE),
        standardize="none",
        inner_max_iter=200_000,
        seed=0,
    )
    return model.fit(
        frame,
        outcome="outcome",
        treatment="treated",
        unit="unit",
        time="time",
        pre_period_outcomes="all",
    )


def test_the_inference_is_about_synthetic_controls_own_estimate() -> None:
    outcomes = _panel(0, effect=1.0)
    inference = synthetic_control_inference(outcomes, 0, N_PRE)
    estimate = synthetic_control(outcomes, 0, N_PRE)
    np.testing.assert_array_equal(inference.estimate.weights, estimate.weights)
    assert inference.estimate.overall == estimate.overall


def test_whichever_unit_is_treated_the_placebo_p_values_are_the_ranks() -> None:
    """Each unit in turn treated, the N p-values are 1/N, 2/N, ..., 1: so exactly k of them reach
    k/N, which makes the test exact when the treated unit is drawn at random. Leaving the treated
    unit out of every placebo's donor pool, as diff-diff does, breaks this on this panel."""
    outcomes = _panel(0, units=10)
    p = [synthetic_control_inference(outcomes, unit, N_PRE).placebo_p_value for unit in range(10)]
    np.testing.assert_allclose(np.sort(p), np.arange(1, 11) / 10, rtol=0, atol=1e-12)


def test_a_unit_its_donors_reproduce_in_every_period_ranks_as_showing_nothing() -> None:
    outcomes = _panel(1, effect=10.0)
    outcomes[2] = outcomes[1]
    assert synthetic_control_inference(outcomes, 0, N_PRE).placebo_p_value == 1 / 20


def test_units_their_donors_fit_exactly_rank_by_their_error_after_treatment() -> None:
    """Three periods before treatment and nineteen donors: nine units, the treated one among them,
    sit inside the others' hull and fit to rounding. Their ratios all exceed the rest, and among
    them the error after treatment decides, not the rounding."""
    outcomes = _panel(0, effect=3.0)[:, N_PRE - 3 :]
    fits = [synthetic_control(outcomes, unit, 3) for unit in range(20)]
    exact = [unit for unit, fit in enumerate(fits) if fit.pre_rmspe < 1e-12]
    assert len(exact) == 9
    assert 0 in exact
    post = {unit: float(np.sqrt(np.mean(fits[unit].att ** 2))) for unit in exact}
    expected = sum(post[unit] >= post[0] for unit in exact) / 20
    assert synthetic_control_inference(outcomes, 0, 3).placebo_p_value == expected


def test_as_the_effect_grows_the_placebo_falls_to_its_floor_and_the_interval_moves_with_it() -> (
    None
):
    """The failure this catches is an in-time placebo's: read before treatment, its p-value cannot
    move with the effect. The interval moves by the effect exactly, since the test of ``c`` on the
    shifted panel is the test of ``c - effect`` on the first."""
    base = synthetic_control_inference(_panel(3), 0, N_PRE)
    moved = synthetic_control_inference(_panel(3, effect=10.0), 0, N_PRE)
    assert moved.placebo_p_value == 1 / 20 < base.placebo_p_value
    np.testing.assert_allclose(np.subtract(moved.interval, 10.0), base.interval, atol=1e-6)
    assert moved.interval[0] > 0.0


def test_the_conformal_p_value_is_diff_diffs() -> None:
    outcomes = _panel(2, effect=2.0)
    fit = _diff_diff(outcomes)
    estimate = synthetic_control(outcomes, 0, N_PRE)
    for effect in estimate.overall + estimate.pre_rmspe * np.linspace(-6.0, 6.0, 25):
        theirs = fit.conformal_test(float(effect))
        assert theirs["proxy_converged"]
        assert _conformal_p_value(outcomes, 0, N_PRE, float(effect)) == theirs["p_value"]


def test_the_interval_ends_where_diff_diffs_test_starts_to_reject() -> None:
    outcomes = _panel(2, effect=2.0)
    fit = _diff_diff(outcomes)
    lo, hi = synthetic_control_inference(outcomes, 0, N_PRE).interval
    step = 0.02 * synthetic_control(outcomes, 0, N_PRE).pre_rmspe
    for inside, outside in ((lo + step, lo - step), (hi - step, hi + step)):
        assert fit.conformal_test(inside)["p_value"] > 0.05
        assert fit.conformal_test(outside)["p_value"] <= 0.05


def test_the_search_follows_an_accepted_set_past_its_grid() -> None:
    lo, hi = _hull(lambda x: -250.0 < x < 130.0, 0.0, 1.0)
    assert -250.0 < lo <= -250.0 + 1e-6
    assert 130.0 - 1e-6 <= hi < 130.0


def test_below_the_tests_resolution_the_interval_is_unbounded() -> None:
    """Thirty periods give p-values in steps of 1/30, so at 0.02 nothing is rejected."""
    inference = synthetic_control_inference(_panel(4), 0, N_PRE, alpha=0.02)
    assert inference.interval == (-math.inf, math.inf)


def test_an_effect_no_constant_describes_is_rejected_or_bracketed_by_a_hull() -> None:
    """An effect of +10 and -10 in turn: the test accepts effects near +10 and near -10, at
    p = 2/30, and rejects the rest, the estimate among them. At 0.05 the hull spans both; at 0.1
    nothing is accepted."""
    outcomes = _panel(5)
    outcomes[0, N_PRE:] += 10.0 * (-1.0) ** np.arange(N_POST)
    estimate = synthetic_control(outcomes, 0, N_PRE).overall
    assert _conformal_p_value(outcomes, 0, N_PRE, estimate) <= 0.05
    lo, hi = synthetic_control_inference(outcomes, 0, N_PRE).interval
    assert lo < -9.0 < 9.0 < hi
    lo, hi = synthetic_control_inference(outcomes, 0, N_PRE, alpha=0.1).interval
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_it_refuses_a_level_outside_zero_and_one() -> None:
    for alpha in (0.0, 1.0):
        with pytest.raises(ValueError, match="alpha must lie in"):
            synthetic_control_inference(_panel(6), 0, N_PRE, alpha=alpha)
