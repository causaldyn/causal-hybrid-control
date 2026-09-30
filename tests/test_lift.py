"""chc.lift: a channel fitted to geo tests' gaps, the intervals its profile gives, and an
observational channel checked against them.

The oracle for the intervals is written here without the module: the kernel and the curve in
NumPy, the coefficient in closed form, the scale by a bounded search, so the profile over the
retention and its crossings are computed by a second route. The check's oracle is the same NumPy
channel, and the fit's least squares as the fit recorded it.
"""

import dataclasses
import math

import equinox as eqx
import numpy as np
import pytest
from scipy import optimize, stats

from chc.lift import GeoArm, LiftTest, ObservationalCheck, check_observational, fit_lift
from chc.response import Channel, GeometricAdstock, MichaelisMenten, Tanh

LENGTH = 6
TRUE = {"kernel.retention": 0.2, "curve.scale": 250.0, "coefficient": 1100.0}
STARTS = (20, 55, 100, 140)
LEVELS = (0.7, 1.7, 0.85, 0.75)  # the tests' spend around the mean: Heusch 2026a's four levels


def _channel(retention: float, scale: float, coefficient: float) -> Channel:
    return Channel(
        GeometricAdstock(retention, length=LENGTH, normalized=True), Tanh(scale), coefficient
    )


def _world(
    seed: int,
    *,
    noise: float = 55.0,
    truth: Channel | None = None,
    levels: tuple[float, ...] = LEVELS,
    share: float = 1.0,
) -> tuple[LiftTest, ...]:
    """Go-dark tests on a random spend path, read over 4 weeks before, 4 dark and 8 after."""
    truth = truth or _channel(*TRUE.values())
    rng = np.random.default_rng(seed)
    spend = np.exp(np.log(75.0) + 0.35 * rng.standard_normal(156))
    for start, level in zip(STARTS, levels, strict=True):
        spend[start - 10 : start + 12] *= level
    tests = []
    for start in STARTS:
        dark = spend.copy()
        dark[start : start + 4] = 0.0
        window = slice(start - 4, start + 12)
        noisy = rng.normal(0.0, noise / 2.0, 16)
        tests.append(
            LiftTest(
                treated=GeoArm(
                    share * dark[: window.stop],
                    share * (2000.0 + np.asarray(truth(dark))[window] - noisy),
                    share,
                ),
                control=GeoArm(
                    share * spend[: window.stop],
                    share * (2000.0 + np.asarray(truth(spend))[window] + noisy),
                    share,
                ),
            )
        )
    return tuple(tests)


START = _channel(0.5, 100.0, 1.0)


@pytest.fixture(scope="module")
def fit():
    return fit_lift(_world(0), START)


def test_the_fit_recovers_the_channel_from_near_noiseless_gaps() -> None:
    fit = fit_lift(_world(1, noise=1e-6), START)
    np.testing.assert_allclose(fit.estimate, list(TRUE.values()), rtol=1e-6)
    assert fit.parameters == tuple(TRUE)


def _oracle_cost(tests: tuple[LiftTest, ...], retention: float, scale: float) -> float:
    """The least cost over the coefficient, by NumPy and in closed form."""
    weights = retention ** np.arange(LENGTH)
    weights = weights / weights.sum()
    shapes, gaps = [], []
    for test in tests:
        curves = [
            np.tanh(np.convolve(arm.spend / arm.share, weights)[: arm.spend.size] / scale)
            for arm in (test.treated, test.control)
        ]
        shapes.append((curves[0] - curves[1])[test.treated.history :])
        gaps.append(test.difference)
    shape, gap = np.concatenate(shapes), np.concatenate(gaps)
    coefficient = (gap @ shape) / (shape @ shape)
    return float(np.sum((gap - coefficient * shape) ** 2))


def _oracle_profile(tests: tuple[LiftTest, ...], retention: float) -> float:
    grid = np.exp(np.linspace(np.log(5.0), np.log(1e7), 400))
    costs = [_oracle_cost(tests, retention, scale) for scale in grid]
    best = int(np.argmin(costs))
    lo, hi = grid[max(best - 1, 0)], grid[min(best + 1, grid.size - 1)]
    found = optimize.minimize_scalar(
        lambda log_scale: _oracle_cost(tests, retention, math.exp(log_scale)),
        bounds=(math.log(lo), math.log(hi)),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return min(float(found.fun), min(costs))


def test_the_retention_interval_is_where_an_independent_profile_crosses_the_cutoff(fit) -> None:
    tests = _world(0)
    least = optimize.minimize_scalar(
        lambda r: _oracle_profile(tests, r), bounds=(0.0, 0.99), method="bounded"
    )
    dof = 16 * len(tests) - 3
    cutoff = least.fun / dof * stats.f.ppf(0.95, 1, dof)
    ends = [
        optimize.brentq(lambda r: _oracle_profile(tests, r) - least.fun - cutoff, a, b, xtol=1e-9)
        for a, b in ((0.0, least.x), (least.x, 0.99))
    ]
    assert fit.dof == dof
    assert fit.noise_sd == pytest.approx(math.sqrt(least.fun / dof), rel=1e-6)
    np.testing.assert_allclose(fit.interval("kernel.retention"), ends, rtol=1e-4)
    assert ends[0] < TRUE["kernel.retention"] < ends[1]


def test_a_channel_that_loses_sales_is_fitted_with_its_sign() -> None:
    fit = fit_lift(_world(2, truth=_channel(0.2, 250.0, -600.0)), START)
    low, high = fit.interval("coefficient")
    assert high < 0.0
    assert low <= -600.0 <= high


def test_a_channel_that_does_nothing_leaves_its_shape_open() -> None:
    """With no effect the kernel and the curve are not in the gap: the retention's interval is its
    whole box and the coefficient's holds zero."""
    fit = fit_lift(_world(3, truth=_channel(0.2, 250.0, 0.0)), START)
    assert fit.interval("kernel.retention") == (0.0, 1.0)
    low, high = fit.interval("coefficient")
    assert low < 0.0 < high


def test_a_curve_the_tests_never_bent_has_no_upper_scale() -> None:
    """Tests at a tenth of the spend read the line the curve starts on: its scale and coefficient
    trade along a ridge to infinity, while the carryover, read at a tenth of the noise, is
    pinned."""
    fit = fit_lift(_world(4, noise=5.5, levels=(0.1, 0.1, 0.1, 0.1)), START)
    assert fit.interval("curve.scale")[1] == math.inf
    assert fit.interval("coefficient")[1] == math.inf
    low, high = fit.interval("kernel.retention")
    assert 0.0 < low < high < 1.0  # whether it covers is a rate, which the bench measures


def test_noise_ten_times_the_gap_s_still_settles_on_a_least_cost() -> None:
    """At that noise the least cost lies far along a ridge the solver must follow to its end: the
    fit settles there, and costs no more than the true channel's shape with its best coefficient."""
    tests = _world(5, noise=550.0)
    fit = fit_lift(tests, START)
    retention, scale, _ = fit.estimate
    cost = fit.noise_sd**2 * fit.dof
    assert cost == pytest.approx(_oracle_cost(tests, retention, scale), rel=1e-9)
    assert cost <= _oracle_cost(tests, TRUE["kernel.retention"], TRUE["curve.scale"])


def test_a_solve_that_cannot_converge_within_its_budget_raises(monkeypatch) -> None:
    monkeypatch.setattr("chc.lift._EVALUATIONS", 2)
    with pytest.raises(RuntimeError, match="spent 10 evaluations without converging"):
        fit_lift(_world(0), START)


def test_a_group_s_share_scales_it_to_the_market() -> None:
    whole = fit_lift(_world(5), START)
    part = fit_lift(_world(5, share=0.4), START)
    np.testing.assert_allclose(part.estimate, whole.estimate, rtol=1e-7)
    np.testing.assert_allclose(part.lower, whole.lower, rtol=1e-6)


def test_the_template_s_coefficient_does_not_move_the_fit(fit) -> None:
    again = fit_lift(_world(0), _channel(0.5, 100.0, -5e4))
    np.testing.assert_allclose(again.estimate, fit.estimate, rtol=1e-7)


def test_the_tested_adstock_is_what_the_readouts_covered(fit) -> None:
    covered = np.concatenate(
        [
            np.asarray(fit.channel.kernel(arm.spend))[arm.history :]
            for test in _world(0)
            for arm in (test.treated, test.control)
        ]
    )
    assert fit.tested_adstock == (pytest.approx(covered.min()), pytest.approx(covered.max()))
    assert fit.tested_adstock[0] < 1.0  # the dark weeks read the curve near zero


def test_a_readout_without_the_spend_before_it_is_refused() -> None:
    test = _world(0)[0]
    short = LiftTest(
        treated=GeoArm(test.treated.spend[-18:], test.treated.outcome),
        control=GeoArm(test.control.spend[-18:], test.control.outcome),
    )
    with pytest.raises(ValueError, match="test 1 carries 2 periods of spend before its readout"):
        fit_lift((test, short), START)


@pytest.mark.parametrize(
    ("spend", "outcome", "share", "message"),
    [
        ([1.0, -1.0], [1.0], 1.0, "negative"),
        ([1.0], [1.0, 2.0], 1.0, "covers the readout"),
        ([1.0, 2.0], [1.0], 0.0, "share"),
        ([1.0, 2.0], [1.0], 1.5, "share"),
        ([1.0, math.nan], [1.0], 1.0, "non-finite"),
        ([], [1.0], 1.0, "non-empty"),
    ],
)
def test_an_arm_that_is_not_a_reading_is_refused(spend, outcome, share, message) -> None:
    with pytest.raises(ValueError, match=message):
        GeoArm(spend, outcome, share)


def test_groups_over_different_periods_are_refused() -> None:
    with pytest.raises(ValueError, match="same periods"):
        LiftTest(treated=GeoArm([1.0, 2.0, 3.0], [1.0]), control=GeoArm([1.0, 2.0], [1.0]))


def test_what_cannot_be_fitted_is_refused() -> None:
    tests = _world(0)
    with pytest.raises(ValueError, match="at least one test"):
        fit_lift((), START)
    with pytest.raises(ValueError, match="level"):
        fit_lift(tests, START, level=1.0)
    with pytest.raises(TypeError, match=r"chc\.response\.Channel"):
        fit_lift(tests, Tanh(100.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"kernel\.retention is 0\.0; it is fitted as its log"):
        fit_lift(tests, _channel(0.0, 100.0, 1.0))
    one = tests[0]
    few = LiftTest(
        treated=GeoArm(one.treated.spend[:-13], one.treated.outcome[:3]),
        control=GeoArm(one.control.spend[:-13], one.control.outcome[:3]),
    )
    with pytest.raises(ValueError, match="cannot fit 3 parameters"):
        fit_lift((few,), START)


def _oracle_gaps(
    tests: tuple[LiftTest, ...], retention: float, scale: float, coefficient: float
) -> np.ndarray:
    """The gaps a channel predicts, by NumPy."""
    weights = retention ** np.arange(LENGTH)
    weights = weights / weights.sum()

    def curve(arm: GeoArm) -> np.ndarray:
        return np.tanh(np.convolve(arm.spend / arm.share, weights)[: arm.spend.size] / scale)

    return np.concatenate(
        [coefficient * (curve(t.treated) - curve(t.control))[t.treated.history :] for t in tests]
    )


def _times(channel: Channel, factor: float) -> Channel:
    return eqx.tree_at(lambda c: c.coefficient, channel, channel.coefficient * factor)


def test_the_check_reads_a_channel_s_gaps_against_the_fit_s_least_squares(fit) -> None:
    tests = _world(0)
    observed = (0.35, 180.0, 900.0)
    check = check_observational(fit, _channel(*observed))
    gap = np.concatenate([test.difference for test in tests])
    predicted = _oracle_gaps(tests, *observed)
    variance = fit.noise_sd**2
    statistic = (np.sum((gap - predicted) ** 2) - variance * fit.dof) / 3 / variance
    factor = (predicted @ gap) / (predicted @ predicted)
    half = stats.t.ppf(0.975, fit.dof) * math.sqrt(variance / (predicted @ predicted))
    assert check.dof == (3, fit.dof)
    assert check.statistic == pytest.approx(statistic, rel=1e-8)
    assert check.p_value == pytest.approx(stats.f.sf(statistic, 3, fit.dof), rel=1e-6)
    assert check.factor == pytest.approx(factor, rel=1e-10)
    np.testing.assert_allclose(check.factor_interval, (factor - half, factor + half), rtol=1e-10)


def test_the_world_s_own_channel_passes_the_check_of_its_tests(fit) -> None:
    """On the fixture's tests the truth's p-value is 0.19: not rejected at the fit's level, its
    factor's interval holds 1, and no confounding is needed to reconcile it."""
    check = check_observational(fit, _channel(*TRUE.values()))
    assert 0.05 < check.p_value < 0.95
    assert not check.rejected
    assert check.factor_interval[0] < 1.0 < check.factor_interval[1]
    assert check.least_gamma(1.0) == 1.0


@pytest.mark.parametrize("k", [0.5, 2.0])
def test_the_fit_reads_a_factor_of_one_and_k_times_it_reads_one_over_k(fit, k) -> None:
    """The fit's coefficient is the least-squares multiple of its shape, so the fit's own channel
    reads 1 with nothing to reject, and k times it reads 1/k with an excess of (k-1)^2 |g|^2."""
    own = check_observational(fit, fit.channel)
    assert own.factor == pytest.approx(1.0, rel=1e-9)
    assert own.statistic == pytest.approx(0.0, abs=1e-9)
    assert not own.rejected
    scaled = check_observational(fit, _times(fit.channel, k))
    predicted = _oracle_gaps(_world(0), *fit.estimate)
    excess = (k - 1.0) ** 2 * (predicted @ predicted)
    assert scaled.factor == pytest.approx(1.0 / k, rel=1e-9)
    assert scaled.statistic == pytest.approx(excess / 3 / fit.noise_sd**2, rel=1e-7)
    assert scaled.rejected


def test_a_channel_of_the_right_size_and_the_wrong_carryover_is_rejected(fit) -> None:
    """Scaled to the size the tests read, a channel whose carryover lasts four times as long reads
    a factor of 1, and the F still says the gaps are not its."""
    slow = _channel(0.8, TRUE["curve.scale"], TRUE["coefficient"])
    sized = _times(slow, check_observational(fit, slow).factor)
    check = check_observational(fit, sized)
    assert check.factor == pytest.approx(1.0, rel=1e-9)
    assert check.least_gamma(1.0) == 1.0
    assert check.rejected
    assert check.p_value < 1e-6


@pytest.mark.parametrize(
    "interval", [(0.3, 0.6), (1.2, 1.5), (0.8, 1.1), (-0.4, 0.2), (2.5, 3.0), (-1.0, -0.2)]
)
def test_the_least_gamma_is_the_first_on_a_grid_whose_set_reaches_the_interval(interval) -> None:
    check = ObservationalCheck(0.0, 1.0, (3, 60), sum(interval) / 2.0, interval, 0.95)
    gammas = np.linspace(1.0, 60.0, 590_001)
    for gap in (0.5, 1.0, 2.0):
        radius = (gammas - 1.0) / (gammas + 1.0) * gap
        reaches = (1.0 - radius <= interval[1]) & (interval[0] <= 1.0 + radius)
        least = check.least_gamma(gap)
        if reaches.any():
            first = gammas[np.argmax(reaches)]
            assert first - 1e-4 <= least <= first * (1.0 + 1e-12)
        else:
            assert least == math.inf


def test_a_channel_that_predicts_no_gap_has_no_factor(fit) -> None:
    check = check_observational(fit, _times(fit.channel, 0.0))
    assert math.isnan(check.factor)
    assert math.isnan(check.least_gamma(1.0))
    assert check.rejected  # the tests read a lift the channel says is not there
    with pytest.raises(ValueError, match="cvar_gap"):
        check_observational(fit, fit.channel).least_gamma(0.0)


def test_the_check_reads_a_group_at_the_market_s_scale() -> None:
    truth = _channel(*TRUE.values())
    whole = check_observational(fit_lift(_world(5), START), truth)
    part = check_observational(fit_lift(_world(5, share=0.4), START), truth)
    assert part.statistic == pytest.approx(whole.statistic, rel=1e-6)
    assert part.factor == pytest.approx(whole.factor, rel=1e-9)


def test_what_the_check_cannot_read_is_refused(fit) -> None:
    kernel = GeometricAdstock(0.2, length=LENGTH, normalized=True)
    others = [
        Channel(kernel, MichaelisMenten(250.0), 1100.0),
        Channel(GeometricAdstock(0.2, length=LENGTH + 2, normalized=True), Tanh(250.0), 1100.0),
        Channel(GeometricAdstock(0.2, length=LENGTH, normalized=False), Tanh(250.0), 1100.0),
    ]
    for other in others:
        with pytest.raises(ValueError, match="not of the fit's families"):
            check_observational(fit, other)
    with pytest.raises(TypeError, match=r"reads a chc\.response\.Channel"):
        check_observational(fit, Tanh(250.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not finite"):
        check_observational(fit, _times(fit.channel, math.nan))


def test_a_channel_cheaper_than_the_fit_says_the_fit_is_not_the_least(fit) -> None:
    off = dataclasses.replace(fit, channel=_times(fit.channel, 1.5))
    with pytest.raises(ValueError, match="fits the tests better than the fit"):
        check_observational(off, fit.channel)
    exact = []
    for test in _world(0):
        treated, control = test.treated.spend, test.control.spend
        gap = np.asarray(fit.channel(treated) - fit.channel(control))[test.treated.history :]
        exact.append(
            LiftTest(treated=GeoArm(treated, gap), control=GeoArm(control, np.zeros(gap.size)))
        )
    with pytest.raises(ValueError, match="no noise"):
        check_observational(dataclasses.replace(fit, tests=tuple(exact)), fit.channel)
