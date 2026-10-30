"""blank_period_inference: the p-value and the intervals against an exact enumeration of the
arrangements its theorem averages over, and the size of a designed test by Monte Carlo."""

from __future__ import annotations

import hashlib
import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
from scipy import special, stats

from chc.scm import _ar1_intervals, blank_period_inference, synthetic_control


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


def _ar_panel(seed: int, rho: float, units: int = 20, periods: int = 25) -> np.ndarray:
    """Unit levels, three factors and the errors all AR(1) with coefficient ``rho``, loadings in
    [0, 1]: each synthetic control's gaps are then AR(1) with that coefficient too."""
    rng = np.random.default_rng(seed)
    factors = np.empty((periods, 3))
    factors[0] = rng.normal(size=3) / math.sqrt(1.0 - rho**2)
    errors = np.empty((units, periods))
    errors[:, 0] = rng.normal(size=units) / math.sqrt(1.0 - rho**2)
    for t in range(1, periods):
        factors[t] = rho * factors[t - 1] + rng.normal(size=3)
        errors[:, t] = rho * errors[:, t - 1] + rng.normal(size=units)
    return rng.normal(size=(units, 1)) + rng.uniform(0.0, 1.0, (units, 3)) @ factors.T + errors


def test_ar1_over_every_arrangement_of_the_innovations_the_p_values_are_the_steps() -> None:
    """Given the coefficient, the gaps' innovations over the blank and post periods are what the
    test permutes. Rebuilding the treated unit's gaps from each arrangement of them, the donors
    untouched, gives every one of the ``n_post * C(n_blank + n_post, n_post)`` p-values once: an
    arrangement is the set of innovations after treatment and which of them comes first."""
    n_fit, n_blank, n_post = 15, 3, 2
    outcomes = _ar_panel(6, 0.5, units=12, periods=n_fit + n_blank + n_post)
    found = blank_period_inference(outcomes, 0, n_fit + n_blank, n_blank, serial="ar1")
    rho = found.ar_coefficient
    assert rho is not None
    count = n_post * math.comb(n_blank + n_post, n_post)
    assert found.arrangements == count == 20
    synthetic = outcomes[1:].T @ found.estimate.weights
    path = outcomes[0] - synthetic
    innovations = path[n_fit:] - rho * path[n_fit - 1 : -1]
    p = []
    for post in itertools.combinations(range(n_blank + n_post), n_post):
        blank = [i for i in range(n_blank + n_post) if i not in post]
        for first in post:
            order = [*blank, first, *(i for i in post if i != first)]
            arranged = outcomes.copy()
            gap = path[n_fit - 1]
            for k, i in enumerate(order):
                gap = rho * gap + innovations[i]
                arranged[0, n_fit + k] = synthetic[n_fit + k] + gap
            again = blank_period_inference(arranged, 0, n_fit + n_blank, n_blank, serial="ar1")
            assert again.ar_coefficient == rho
            p.append(again.p_value)
    np.testing.assert_allclose(np.sort(p), np.arange(1, count + 1) / count, rtol=0, atol=1e-12)


def test_ar1_reads_its_coefficient_from_the_donors_fitted_on_half_the_fitting_periods() -> None:
    """Han, Phillips and Sul's panel fully aggregated estimator, written out: each donor's synthetic
    control from the other donors, fitted on the first 8 of 15 fitting periods, and every pair of
    its later gaps three or more periods apart. The treated unit takes no part."""
    outcomes = _ar_panel(7, 0.8, units=8)
    found = blank_period_inference(outcomes, 0, 20, 5, serial="ar1")
    donors = outcomes[1:]
    numerator = denominator = 0.0
    for j in range(donors.shape[0]):
        gaps = synthetic_control(donors, j, 8).att
        for s in range(gaps.size):
            for t in range(s + 3, gaps.size):
                regressor = gaps[t - 1] - gaps[s + 1]
                numerator += regressor * (gaps[t] - gaps[s])
                denominator += regressor**2
    assert found.ar_coefficient == pytest.approx(numerator / denominator, rel=0, abs=1e-12)
    moved = outcomes.copy()
    moved[0, 15:] += np.random.default_rng(1).normal(size=10)
    assert blank_period_inference(moved, 0, 20, 5, serial="ar1").ar_coefficient == (
        found.ar_coefficient
    )


@pytest.mark.parametrize(("ratio", "end"), [(1.5, 1.0), (-1.5, -1.0)])
def test_ar1_keeps_its_coefficient_in_the_closed_unit_interval(ratio: float, end: float) -> None:
    """Donors ``j ratio**t``: the outer two lie outside the others' hull, and their gaps grow as
    ``ratio**t``, past either end of an AR(1) coefficient's range."""
    periods = np.arange(12)
    outcomes = np.array([[0.5, 2.5, 1.0, 3.0, -1.0, 0.0, 2.0, 1.5, 0.2, 0.7, 1.1, 0.4]])
    outcomes = np.vstack([outcomes, np.outer(np.arange(1.0, 6.0), ratio**periods)])
    assert blank_period_inference(outcomes, 0, 8, 2, serial="ar1").ar_coefficient == end


def test_ar1_a_tie_that_rounding_splits_counts_as_a_tie() -> None:
    """Donors ``j 1.5**t`` for ``j = 0 .. 5`` set the coefficient to 1, so the innovations are the
    gaps' differences, and the treated unit, below every donor, takes the zero donor alone: its
    gaps are its outcomes. Each difference is 0.3 exactly, but in floating point ``1 - 0.7`` and
    ``0.4 - 0.1`` are one ulp above ``0.7 - 0.4``: the p-value is 3/3, not 2/3."""
    donors = np.outer(np.arange(6.0), 1.5 ** np.arange(9))
    treated = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.7, -0.4, -0.1]
    found = blank_period_inference(np.vstack([treated, donors]), 0, 8, 2, serial="ar1")
    assert found.ar_coefficient == 1.0
    assert found.arrangements == 3
    assert found.p_value == 1.0


def test_ar1_changes_the_p_value_and_the_intervals_only() -> None:
    outcomes = _ar_panel(8, 0.5)
    default = blank_period_inference(outcomes, 3, 20, 5, alpha=0.2)
    assert default.ar_coefficient is None
    assert default.ar_check_p_value is None
    named = blank_period_inference(outcomes, 3, 20, 5, alpha=0.2, serial="exchangeable")
    assert named.p_value == default.p_value
    assert named.ar_coefficient is None
    np.testing.assert_array_equal(named.intervals, default.intervals)
    ar1 = blank_period_inference(outcomes, 3, 20, 5, alpha=0.2, serial="ar1")
    assert not np.array_equal(ar1.intervals, default.intervals)
    for found in (named, ar1):
        np.testing.assert_array_equal(found.blank_gaps, default.blank_gaps)
        np.testing.assert_array_equal(found.estimate.att, default.estimate.att)
        np.testing.assert_array_equal(found.estimate.weights, default.estimate.weights)
        assert found.estimate.pre_rmspe == default.estimate.pre_rmspe
    assert named.arrangements == default.arrangements == 252
    assert ar1.arrangements == 1260


@pytest.mark.parametrize("effect", [10.0, -10.0])
def test_ar1_an_effect_of_either_sign_takes_the_p_value_to_its_floor(effect: float) -> None:
    outcomes = _ar_panel(9, 0.5)
    base = blank_period_inference(outcomes, 0, 20, 5, serial="ar1")
    outcomes[0, 20:] += effect
    moved = blank_period_inference(outcomes, 0, 20, 5, serial="ar1")
    assert moved.p_value == 1 / 1260 < base.p_value


def test_ar1_holds_the_size_a_persistent_error_breaks() -> None:
    """400 panels whose factors and errors are AR(1) with coefficient 0.8, 19 donors, 15 fitting,
    5 blank and 5 post periods, no effect, the treated unit the one its donors fit best on the
    fitting periods. The exchangeable reading rejects far above 5%; the AR(1) reading stays within
    three standard errors of 400 draws of it."""
    exchangeable = ar1 = 0
    for seed in range(400):
        outcomes = _ar_panel(1000 + seed, 0.8)
        fits = [synthetic_control(outcomes, unit, 15).pre_rmspe for unit in range(20)]
        treated = int(np.argmin(fits))
        exchangeable += blank_period_inference(outcomes, treated, 20, 5).p_value <= 0.05
        ar1 += blank_period_inference(outcomes, treated, 20, 5, serial="ar1").p_value <= 0.05
    bound = 0.05 + 3 * math.sqrt(0.05 * 0.95 / 400)
    assert ar1 / 400 <= bound < exchangeable / 400


def test_ar1_refuses_what_it_cannot_read() -> None:
    outcomes = _ar_panel(10, 0.5, units=6)
    with pytest.raises(ValueError, match="serial must be"):
        blank_period_inference(outcomes, 0, 20, 5, serial="ar2")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="at least two donors"):
        blank_period_inference(outcomes[:2], 0, 20, 5, serial="ar1")
    # 2 fitting, 1 blank and 1 post period: the donors' gaps after the first period span three
    with pytest.raises(ValueError, match="cannot read an AR"):
        blank_period_inference(outcomes[:, :4], 0, 3, 1, serial="ar1")
    # 28 blank and 5 post periods: 5 C(33, 5) = 1 186 680 arrangements, where the exchangeable
    # reading's C(33, 5) = 237 336 stay under the cap
    long = _ar_panel(10, 0.5, units=4, periods=36)
    with pytest.raises(ValueError, match="1186680 arrangements"):
        blank_period_inference(long, 0, 31, 28, serial="ar1")


def _digest(values: list) -> str:
    """Rounded to ten digits, so that a last bit a BLAS or a CPU moves does not move it."""
    text = repr([f"{v:.10g}" if isinstance(v, float) else v for v in values])
    return hashlib.sha256(text.encode()).hexdigest()


def test_the_exchangeable_reading_and_the_ar1_test_are_what_they_were() -> None:
    """Six panels with AR(0.5) errors and effects 0 to 2.5, at ``alpha`` 0.05 and 0.2: the
    exchangeable reading's every output, and the ``"ar1"`` reading's p-value, coefficient and
    blank gaps, digested as the previous release produced them."""
    exchangeable: list = []
    ar1: list = []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        units, periods = 12, 25
        errors = np.empty((units, periods))
        errors[:, 0] = rng.normal(size=units)
        for t in range(1, periods):
            errors[:, t] = 0.5 * errors[:, t - 1] + rng.normal(size=units)
        outcomes = (
            rng.normal(size=(units, 1))
            + rng.uniform(0, 1, (units, 3)) @ rng.normal(size=(periods, 3)).T
            + errors
        )
        outcomes[0, -5:] += 0.5 * seed
        for alpha in (0.05, 0.2):
            found = blank_period_inference(outcomes, 0, 20, 5, alpha=alpha)
            exchangeable += [found.p_value, found.arrangements, found.estimate.overall]
            exchangeable += found.intervals.ravel().tolist() + found.blank_gaps.tolist()
            serial = blank_period_inference(outcomes, 0, 20, 5, alpha=alpha, serial="ar1")
            ar1 += [serial.p_value, serial.ar_coefficient, serial.arrangements]
            ar1 += serial.blank_gaps.tolist()
    assert _digest(exchangeable) == (
        "63c57cacb67276ad8a5dcf2c94a52c5fbf945e98dc810027c0f7b7816d0bfb77"
    )
    assert _digest(ar1) == "84a324364ebc9cef97bb7002edd3a5954847b7edc9c1a7c9b524340d0ba9340a"


def test_ar1_intervals_are_the_t_prediction_intervals_maxima_computes() -> None:
    """The last fitting gap 0.3, blank gaps -0.5, 1.1 and 0.4, post gaps 0.9 and -0.2, at
    coefficient 0.6 and ``alpha = 0.2``: the endpoints ``validation/scm_ar1_intervals.mac``
    computes from the closed form of the t quantile at 2 degrees of freedom."""
    path = np.array([0.3, -0.5, 1.1, 0.4, 0.9, -0.2])
    found = _ar1_intervals(path[1:4] - 0.6 * path[:3], 0.4, path[4:], 0.6, 0.2)
    expected = [[-1.8881260380920215, 2.9014593714253545], [-3.674807897148537, 2.4961412304818706]]
    np.testing.assert_allclose(found, expected, rtol=0, atol=1e-12)
    one = _ar1_intervals(np.array([0.5]), 0.4, path[4:], 0.6, 0.2)
    assert np.isinf(one).all()


def test_ar1_intervals_read_the_gaps_innovations_written_out() -> None:
    """Through the public reading: the gaps' innovations over the blank periods, the first against
    the last fitting period's gap, and each post period's k-step forecast from the last blank
    gap, ``rho^k u_0`` plus the innovations' mean times ``1 + rho + .. + rho^(k-1)``."""
    outcomes = _ar_panel(8, 0.5)
    found = blank_period_inference(outcomes, 3, 20, 5, alpha=0.1, serial="ar1")
    rho = found.ar_coefficient
    assert rho is not None
    gaps = outcomes[3] - np.delete(outcomes, 3, axis=0).T @ found.estimate.weights
    innovations = gaps[15:20] - rho * gaps[14:19]
    mean, spread = innovations.mean(), innovations.std(ddof=1)
    quantile = stats.t.ppf(0.95, 4)
    for k in range(1, 6):
        level = sum(rho**j for j in range(k))
        variance = sum(rho ** (2 * j) for j in range(k)) + level**2 / 5
        centre = gaps[19 + k] - rho**k * gaps[19] - mean * level
        radius = quantile * spread * math.sqrt(variance)
        np.testing.assert_allclose(
            found.intervals[k - 1], [centre - radius, centre + radius], rtol=1e-12, atol=1e-12
        )


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.8])
def test_ar1_intervals_cover_at_their_level_where_the_errors_persist(rho: float) -> None:
    """1000 designed panels a coefficient, factors and errors AR(1), 19 donors, 15 fitting, 5 blank
    and 5 post periods, a constant effect of 1. The ``"ar1"`` intervals cover each post period's
    effect within three binomial standard errors of 1000 panels of ``1 - alpha``, at 0.1, where
    5 blank periods leave the exchangeable reading's intervals unbounded, and at 0.2. Each panel's
    share of its 5 periods has a variance at most ``p (1 - p)``, so the bound is conservative.
    The AR(1) check holds its size, within three standard errors of 5%."""
    panels = 1000
    covers = {0.1: 0.0, 0.2: 0.0}
    checks = 0
    for seed in range(panels):
        outcomes = _ar_panel(3000 + seed, rho)
        fits = [synthetic_control(outcomes, unit, 15).pre_rmspe for unit in range(20)]
        treated = int(np.argmin(fits))
        outcomes[treated, 20:] += 1.0
        for alpha in covers:
            found = blank_period_inference(outcomes, treated, 20, 5, alpha=alpha, serial="ar1")
            lo, hi = found.intervals.T
            covers[alpha] += float(np.mean((lo <= 1.0) & (hi >= 1.0)))
        assert found.ar_check_p_value is not None
        checks += found.ar_check_p_value <= 0.05
    for alpha, total in covers.items():
        level = 1.0 - alpha
        assert abs(total / panels - level) <= 3 * math.sqrt(level * alpha / panels)
    assert checks / panels <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / panels)


def _ar2_panel(seed: int, phi: tuple[float, float], units: int = 20, periods: int = 25):
    """``_ar_panel``'s, factors AR(1) with coefficient ``phi[0]`` and errors AR(2) with
    ``phi``, both from 50 periods of burn-in."""
    rng = np.random.default_rng(seed)
    factors = np.zeros((periods + 50, 3))
    errors = np.zeros((units, periods + 50))
    for t in range(2, periods + 50):
        factors[t] = phi[0] * factors[t - 1] + rng.normal(size=3)
        errors[:, t] = phi[0] * errors[:, t - 1] + phi[1] * errors[:, t - 2]
        errors[:, t] += rng.normal(size=units)
    loadings = rng.uniform(0.0, 1.0, (units, 3))
    return rng.normal(size=(units, 1)) + loadings @ factors[50:].T + errors[:, 50:]


def test_ar1_check_reads_the_donors_innovations_written_out_and_sees_an_ar2() -> None:
    """The check, written out on one panel: each donor's gaps from the others fitted on the first
    8 periods, filtered by the coefficient, less their own mean; per period, the products at lags
    1 and 2 summed over the donors; each lag's total less ``-(n - h) / n`` times the squares'
    total over ``n - 1``, over the spread of its per-period sums. And on 100 panels whose errors
    are AR(2) with coefficients 0.5 and 0.3, it rejects at 5% on more than 40% of 300 panels,
    where under an AR(1) it rejects on at most 5% plus three standard errors."""
    outcomes = _ar_panel(11, 0.5)
    found = blank_period_inference(outcomes, 0, 20, 5, serial="ar1")
    rho = found.ar_coefficient
    assert rho is not None
    donors = outcomes[1:]
    gaps = np.array([synthetic_control(donors, j, 8).att for j in range(donors.shape[0])])
    e = gaps[:, 1:] - rho * gaps[:, :-1]
    e -= e.mean(axis=1, keepdims=True)
    n = e.shape[1]
    q = 0.0
    for h in (1, 2):
        sums = [float(np.sum(e[:, t] * e[:, t - h])) for t in range(h, n)]
        null = -(n - h) / n * float(np.sum(e**2)) / (n - 1)
        q += (sum(sums) - null) ** 2 / (len(sums) * float(np.var(sums, ddof=1)))
    assert found.ar_check_p_value == pytest.approx(stats.chi2.sf(q, 2), rel=1e-9, abs=0.0)
    rejects = 0
    for seed in range(300):
        panel = _ar2_panel(4000 + seed, (0.5, 0.3))
        fits = [synthetic_control(panel, unit, 15).pre_rmspe for unit in range(20)]
        found = blank_period_inference(panel, int(np.argmin(fits)), 20, 5, serial="ar1")
        assert found.ar_check_p_value is not None
        rejects += found.ar_check_p_value <= 0.05
    assert rejects / 300 > 0.4


def _ar1_outputs(found) -> list:
    return [
        found.p_value,
        found.arrangements,
        found.ar_coefficient,
        found.ar_check_p_value,
        found.estimate.overall,
        *found.intervals.ravel().tolist(),
        *found.blank_gaps.tolist(),
    ]


def test_ar1_plug_in_reading_is_what_it_was() -> None:
    """Four panels with AR(0.8) factors and errors, 3, 5 and 19 donors, at ``alpha`` 0.05 and
    0.2: every output of the ``"ar1"`` reading, intervals and check included, digested as the
    previous release produced them, before ``ar_error`` existed; the default is ``"plug-in"`` and
    carries no variance."""
    values: list = []
    for seed, units in ((0, 4), (1, 6), (2, 20), (3, 20)):
        outcomes = _ar_panel(5000 + seed, 0.8, units=units, periods=30)
        for alpha in (0.05, 0.2):
            found = blank_period_inference(outcomes, 0, 20, 5, alpha=alpha, serial="ar1")
            assert found.ar_coefficient_variance is None
            named = blank_period_inference(
                outcomes, 0, 20, 5, alpha=alpha, serial="ar1", ar_error="plug-in"
            )
            assert _ar1_outputs(named) == _ar1_outputs(found)
            values += _ar1_outputs(found)
    assert _digest(values) == "47b4b0a8e5fff52100e47847f48944c89ecda0cfd559ac13635bdce2b5efb283"


def test_ar_error_refuses_what_it_cannot_read() -> None:
    outcomes = _ar_panel(12, 0.5, units=6)
    with pytest.raises(ValueError, match="ar_error must be"):
        blank_period_inference(outcomes, 0, 20, 5, serial="ar1", ar_error="bootstrap")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="needs serial='ar1'"):
        blank_period_inference(outcomes, 0, 20, 5, ar_error="jackknife")
    blank_period_inference(outcomes, 0, 20, 5, ar_error="plug-in")
    with pytest.raises(ValueError, match="at least three donors, got 2: two donors' gaps"):
        blank_period_inference(outcomes[:3], 0, 20, 5, serial="ar1", ar_error="jackknife")
    found = blank_period_inference(outcomes[:4], 0, 20, 5, serial="ar1", ar_error="jackknife")
    assert found.ar_coefficient_variance is not None


def _coefficient(donors: np.ndarray, n_seen: int) -> float:
    """Han, Phillips and Sul's estimator over each donor's synthetic control from the others,
    written out as in the test of the coefficient above."""
    numerator = denominator = 0.0
    for j in range(donors.shape[0]):
        gaps = synthetic_control(donors, j, n_seen).att
        for s in range(gaps.size):
            for t in range(s + 3, gaps.size):
                regressor = gaps[t - 1] - gaps[s + 1]
                numerator += regressor * (gaps[t] - gaps[s])
                denominator += regressor**2
    return min(max(numerator / denominator, -1.0), 1.0)


def test_ar_error_jackknife_leaves_out_each_donor_and_never_the_treated_unit() -> None:
    """Eight donors: each left out in turn, the other seven's synthetic controls rebuilt from each
    other on the first 8 of 15 fitting periods, the coefficient read again; the variance is
    ``(n - 1) / n`` times the left-out estimates' squared deviations. Moving the treated unit's
    every period leaves it as it was."""
    outcomes = _ar_panel(13, 0.8, units=9)
    found = blank_period_inference(outcomes, 0, 20, 5, serial="ar1", ar_error="jackknife")
    donors = outcomes[1:]
    n = donors.shape[0]
    left_out = [_coefficient(np.delete(donors, j, axis=0), 8) for j in range(n)]
    mean = sum(left_out) / n
    variance = (n - 1) / n * sum((r - mean) ** 2 for r in left_out)
    assert variance > 0.0
    assert found.ar_coefficient_variance == pytest.approx(variance, rel=0, abs=1e-12)
    moved = outcomes.copy()
    moved[0] += np.random.default_rng(2).normal(size=moved.shape[1])
    again = blank_period_inference(moved, 0, 20, 5, serial="ar1", ar_error="jackknife")
    assert again.ar_coefficient_variance == found.ar_coefficient_variance


def test_ar_error_widening_is_the_delta_method_maxima_computes() -> None:
    """``scm_ar1_intervals``' sample path at a coefficient variance of 1/50: the endpoints
    ``validation/scm_ar1_rho_error.mac`` computes in exact arithmetic. At a variance of 0 the
    intervals are the known coefficient's to the bit, whatever the lagged gaps' mean."""
    path = np.array([0.3, -0.5, 1.1, 0.4, 0.9, -0.2])
    innovations = path[1:4] - 0.6 * path[:3]
    lagged = float(np.mean(path[:3]))
    found = _ar1_intervals(
        innovations, 0.4, path[4:], 0.6, 0.2, lagged_mean=lagged, rho_variance=1 / 50
    )
    expected = [[-1.888274503775009, 2.901607837108343], [-3.675078816224816, 2.496412149558149]]
    np.testing.assert_allclose(found, expected, rtol=0, atol=1e-12)
    rng = np.random.default_rng(3)
    for _ in range(200):
        innovations, att = rng.normal(size=5), rng.normal(size=10)
        last, rho = float(rng.normal()), float(rng.uniform(-1.0, 1.0))
        known = _ar1_intervals(innovations, last, att, rho, 0.1)
        zero = _ar1_intervals(
            innovations, last, att, rho, 0.1, lagged_mean=float(rng.normal()), rho_variance=0.0
        )
        np.testing.assert_array_equal(zero, known)
        # the radius as the previous release computed it, operation for operation: the digest
        # above rounds to ten digits, this holds the last bit
        lags = np.arange(10)
        level, spread = np.cumsum(rho**lags), np.cumsum(rho ** (2 * lags))
        quantile = float(special.stdtrit(4, 0.95))
        radius = quantile * float(np.std(innovations, ddof=1)) * np.sqrt(spread + level**2 / 5)
        centre = att - rho ** (lags + 1) * last - float(np.mean(innovations)) * level
        np.testing.assert_array_equal(known, np.column_stack([centre - radius, centre + radius]))


def test_ar_error_jackknife_widens_about_the_same_centres_by_the_delta_method() -> None:
    """Through the public reading: the jackknife's intervals share the plug-in's centres, and each
    radius, squared, is the plug-in's plus the centre's derivative in the coefficient, squared,
    times the jackknife variance and the t quantile squared, the derivative written out."""
    outcomes = _ar_panel(14, 0.8, units=6)
    plug_in = blank_period_inference(outcomes, 2, 20, 5, alpha=0.1, serial="ar1")
    found = blank_period_inference(
        outcomes, 2, 20, 5, alpha=0.1, serial="ar1", ar_error="jackknife"
    )
    rho, variance = found.ar_coefficient, found.ar_coefficient_variance
    assert rho == plug_in.ar_coefficient
    assert rho is not None
    assert variance is not None
    assert variance > 0.0
    assert found.p_value == plug_in.p_value
    np.testing.assert_allclose(
        found.intervals.sum(axis=1), plug_in.intervals.sum(axis=1), rtol=0, atol=1e-12
    )
    gaps = outcomes[2] - np.delete(outcomes, 2, axis=0).T @ found.estimate.weights
    innovations = gaps[15:20] - rho * gaps[14:19]
    quantile = stats.t.ppf(0.95, 4)
    for k in range(1, 6):
        level = sum(rho**j for j in range(k))
        level_slope = sum(j * rho ** (j - 1) for j in range(1, k))
        slope = -k * rho ** (k - 1) * gaps[19] + gaps[14:19].mean() * level
        slope -= innovations.mean() * level_slope
        old = (plug_in.intervals[k - 1, 1] - plug_in.intervals[k - 1, 0]) / 2
        new = (found.intervals[k - 1, 1] - found.intervals[k - 1, 0]) / 2
        assert new**2 == pytest.approx(
            old**2 + quantile**2 * slope**2 * variance, rel=1e-10, abs=0.0
        )


def test_ar_error_jackknife_lifts_four_donors_last_period_coverage() -> None:
    """300 panels whose factors and errors are AR(1) at 0.95, 4 donors, 15 fitting, 5 blank and
    10 post periods, the unit chosen on the fitting periods, ``alpha = 0.1``. The plug-in
    coefficient, biased low, leaves the last post period's coverage below 90%; the jackknife's
    is within two binomial standard errors of 300 panels of 90% or above it."""
    panels, alpha = 300, 0.1
    plug_in = jackknife = 0
    for seed in range(panels):
        outcomes = _ar_panel(6000 + seed, 0.95, units=5, periods=30)
        fits = [synthetic_control(outcomes, unit, 15).pre_rmspe for unit in range(5)]
        treated = int(np.argmin(fits))
        for reading in ("plug-in", "jackknife"):
            found = blank_period_inference(
                outcomes, treated, 20, 5, alpha=alpha, serial="ar1", ar_error=reading
            )
            lo, hi = found.intervals[-1]
            covered = lo <= 0.0 <= hi
            if reading == "plug-in":
                plug_in += covered
            else:
                jackknife += covered
    se = math.sqrt(alpha * (1 - alpha) / panels)
    assert jackknife / panels >= 1 - alpha - 2 * se
    assert jackknife > plug_in
