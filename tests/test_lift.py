"""chc.lift: a channel fitted to geo tests' gaps, and the intervals its profile gives.

The oracle for the intervals is written here without the module: the kernel and the curve in
NumPy, the coefficient in closed form, the scale by a bounded search, so the profile over the
retention and its crossings are computed by a second route.
"""

import math

import numpy as np
import pytest
from scipy import optimize, stats

from chc.lift import GeoArm, LiftTest, fit_lift
from chc.response import Channel, GeometricAdstock, Tanh

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
