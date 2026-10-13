"""blank_period_inference: the p-value and the intervals against an exact enumeration of the
arrangements its theorem averages over, and the size of a designed test by Monte Carlo."""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from chc.scm import blank_period_inference, synthetic_control


def _panel(seed: int, units: int = 12, periods: int = 25, effect: float = 0.0) -> np.ndarray:
    """White-noise factors with loadings in [0, 1], unit levels and white-noise errors, so the
    periods are exchangeable; unit 0 has a constant effect over the last five periods."""
    rng = np.random.default_rng(seed)
    loadings = rng.uniform(0.0, 1.0, (units, 3))
    outcomes = (
        rng.normal(size=(units, 1))
        + loadings @ rng.normal(size=(periods, 3)).T
        + rng.normal(size=(units, periods))
    )
    outcomes[0, -5:] += effect
    return outcomes


def _arrangements(outcomes: np.ndarray, n_fit: int, n_post: int):
    """Each panel with the blank and post periods' columns rearranged so that another choice of
    ``n_post`` of them sits after treatment, the fitting periods untouched."""
    held = np.arange(n_fit, outcomes.shape[1])
    for post in itertools.combinations(range(held.size), n_post):
        blank = [i for i in range(held.size) if i not in post]
        yield outcomes[:, np.concatenate([np.arange(n_fit), held[blank], held[list(post)]])]


@pytest.mark.parametrize(("n_blank", "n_post"), [(3, 2), (5, 5)])
def test_over_every_arrangement_the_p_values_are_the_steps_of_one_over_their_count(
    n_blank: int, n_post: int
) -> None:
    """Exchangeable periods make each arrangement of the blank and post columns equally likely, so
    the share of arrangements with ``p <= alpha`` is the test's size given the columns. Without ties
    the p-values are 1/|Pi|, 2/|Pi|, ..., 1: exactly ``floor(alpha |Pi|)`` of them reach ``alpha``,
    which is the bracket ``alpha - 1/|Pi| <= P(p <= alpha) <= alpha`` at every ``alpha``."""
    n_fit = 15
    outcomes = _panel(0, periods=n_fit + n_blank + n_post)
    p = np.sort(
        [
            blank_period_inference(panel, 0, n_fit + n_blank, n_blank).p_value
            for panel in _arrangements(outcomes, n_fit, n_post)
        ]
    )
    count = math.comb(n_blank + n_post, n_post)
    np.testing.assert_allclose(p, np.arange(1, count + 1) / count, rtol=0, atol=1e-12)
    for alpha in np.arange(1, 100) / 100:
        share = np.mean(p <= alpha + 1e-12)
        assert alpha - 1 / count <= share <= alpha


def test_a_tie_that_rounding_splits_counts_as_a_tie() -> None:
    """One donor at zero, so the gaps are the treated unit's outcomes. The post periods' 0.3 + 0.5
    is 0.8 exactly, as are 0.7 + 0.1, but in floating point the first sum is one ulp above the
    second: the p-value is 8/10, not 7/10."""
    treated = [1.0, 2.0, 0.7, 0.9, 0.1, 0.3, 0.5]
    outcomes = np.array([treated, np.zeros(7)])
    found = blank_period_inference(outcomes, 0, 5, 3)
    assert found.arrangements == 10
    assert found.p_value == 0.8


def test_with_every_gap_alike_no_arrangement_rejects() -> None:
    outcomes = np.array([[1.0, 2.0, 3.0, 0.1, 0.1, 0.1, 0.1, 0.1], np.zeros(8)])
    for panel in _arrangements(outcomes, 3, 2):
        assert blank_period_inference(panel, 0, 6, 3).p_value == 1.0


@pytest.mark.parametrize(("alpha", "n_blank", "rank"), [(0.2, 4, 4), (0.5, 3, 2), (0.7, 9, 3)])
def test_each_periods_interval_covers_the_share_of_arrangements_its_rank_gives(
    alpha: float, n_blank: int, rank: int
) -> None:
    """The radius is the ``ceil((1 - alpha)(n_blank + 1))``-th smallest absolute blank gap. Over the
    ``n_blank + 1`` places a post period's column can take among the blank ones, the interval holds
    its effect, zero here, in exactly ``rank`` of them: coverage ``rank / (n_blank + 1) >= 1 -
    alpha``. At 0.7 and 9 blank, ``0.3 * 10`` is ``3.0000000000000004`` in floating point, and its
    rank is 3, not 4."""
    n_fit = 15
    outcomes = _panel(1, periods=n_fit + n_blank + 1)
    found = blank_period_inference(outcomes, 0, n_fit + n_blank, n_blank, alpha=alpha)
    radius = np.sort(np.abs(found.blank_gaps))[rank - 1]
    gap = found.estimate.att[0]
    np.testing.assert_array_equal(found.intervals[0], [gap - radius, gap + radius])
    covered = 0
    for panel in _arrangements(outcomes, n_fit, 1):
        arranged = blank_period_inference(panel, 0, n_fit + n_blank, n_blank, alpha=alpha)
        lo, hi = arranged.intervals[0]
        covered += lo <= 0.0 <= hi
    assert covered == rank
    assert Fraction(rank, n_blank + 1) >= 1 - Fraction(str(alpha))


def test_below_one_over_the_blank_periods_plus_one_the_intervals_are_unbounded() -> None:
    outcomes = _panel(2)
    bounded = blank_period_inference(outcomes, 0, 20, 5, alpha=1 / 6).intervals
    assert np.all(np.isfinite(bounded))
    unbounded = blank_period_inference(outcomes, 0, 20, 5, alpha=0.16).intervals
    assert np.all(unbounded[:, 0] == -math.inf)
    assert np.all(unbounded[:, 1] == math.inf)


def test_the_weights_fit_on_the_fitting_periods_and_never_see_the_blank_ones() -> None:
    outcomes = _panel(3, effect=1.0)
    found = blank_period_inference(outcomes, 0, 20, 5)
    fitted = synthetic_control(np.delete(outcomes, np.arange(15, 20), axis=1), 0, 15)
    np.testing.assert_array_equal(found.estimate.weights, fitted.weights)
    np.testing.assert_allclose(found.estimate.att, fitted.att, rtol=0, atol=1e-12)
    assert found.estimate.overall == pytest.approx(fitted.overall, rel=0, abs=1e-12)
    assert found.estimate.pre_rmspe == pytest.approx(fitted.pre_rmspe, rel=0, abs=1e-12)
    donors = outcomes[1:, 15:20]
    np.testing.assert_allclose(
        found.blank_gaps, outcomes[0, 15:20] - donors.T @ fitted.weights, rtol=0, atol=1e-12
    )
    moved = outcomes.copy()
    moved[:, 15:20] += np.random.default_rng(0).normal(size=(12, 5))
    np.testing.assert_array_equal(
        blank_period_inference(moved, 0, 20, 5).estimate.weights, found.estimate.weights
    )


@pytest.mark.parametrize("effect", [10.0, -10.0])
def test_an_effect_of_either_sign_takes_the_p_value_to_its_floor(effect: float) -> None:
    base = blank_period_inference(_panel(4), 0, 20, 5, alpha=0.2)
    moved = blank_period_inference(_panel(4, effect=effect), 0, 20, 5, alpha=0.2)
    assert moved.p_value == 1 / 252 < base.p_value
    np.testing.assert_allclose(moved.intervals - effect, base.intervals, rtol=0, atol=1e-12)


def test_a_design_that_chose_its_unit_on_the_fitting_periods_holds_its_size() -> None:
    """400 panels of exchangeable periods, 19 donors, 15 fitting, 5 blank and 5 post periods, no
    effect, the treated unit the one its donors fit best on the fitting periods. The theorem gives
    a size in [5% - 1/252, 5%]; the bound allows three standard errors of 400 draws above it."""
    rejected = 0
    for seed in range(400):
        outcomes = _panel(100 + seed, units=20)
        fits = [synthetic_control(outcomes, unit, 15).pre_rmspe for unit in range(20)]
        treated = int(np.argmin(fits))
        rejected += blank_period_inference(outcomes, treated, 20, 5).p_value <= 0.05
    assert rejected / 400 <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / 400)


def test_it_refuses_what_it_cannot_read() -> None:
    outcomes = _panel(5)
    for n_blank in (0, 20):
        with pytest.raises(ValueError, match="n_blank must be in"):
            blank_period_inference(outcomes, 0, 20, n_blank)
    for alpha in (0.0, 1.0):
        with pytest.raises(ValueError, match="alpha must lie in"):
            blank_period_inference(outcomes, 0, 20, 5, alpha=alpha)
    with pytest.raises(ValueError, match="n_pre must be in"):
        blank_period_inference(outcomes, 0, 25, 5)
    # 21 blank and 7 post periods: C(28, 7) = 1 184 040 arrangements, just past the cap of a million
    long = _panel(5, periods=40)
    with pytest.raises(ValueError, match="1184040 arrangements"):
        blank_period_inference(long, 0, 33, 21)
