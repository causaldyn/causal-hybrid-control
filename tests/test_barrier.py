"""Safety under a partially identified control effect (Result 40, ``chc.barrier``)."""

import numpy as np
import pytest

from chc.barrier import (
    admissible_action_interval,
    barrier_confounding_certificate,
    barrier_gamma_star,
    control_channel,
    identification_radius_threshold,
    robust_barrier_margin,
    robust_safe_action,
    robust_safety_filter,
    safety_filter_benchmark,
)
from chc.uncertainty import confounding_robust_inflation


def _brute_force_margin(drift: float, channel: float, radius: float, u_max: float) -> float:
    """``max_u min_{g'} (drift + g'*u)`` by enumeration -- the closed form is not used as oracle."""
    us = np.linspace(-u_max, u_max, 4001)
    return float(
        np.max(np.minimum(drift + (channel - radius) * us, drift + (channel + radius) * us))
    )


@pytest.mark.parametrize("radius", [0.0, 0.3, 0.9, 1.0, 1.7])
def test_robust_margin_matches_a_brute_force_search_over_both_players(radius: float) -> None:
    drift, channel, u_max = -0.4, 1.0, 2.5
    closed = robust_barrier_margin(drift, channel, radius, u_max)
    assert abs(closed - _brute_force_margin(drift, channel, radius, u_max)) < 1e-9


def test_optimal_action_is_exactly_zero_once_the_channel_sign_is_unidentified() -> None:
    """Not conservatism: with the sign unidentified every action's worst case is at least as bad."""
    grad_h, b_hat = np.array([1.0, 0.0]), np.array([[0.8], [0.0]])
    assert control_channel(grad_h, b_hat) == pytest.approx(0.8)

    acting = robust_safe_action(grad_h, b_hat, radius=0.5, u_max=2.0)
    assert float(np.linalg.norm(acting)) == pytest.approx(2.0)

    for radius in (0.8, 1.2):
        assert float(np.linalg.norm(robust_safe_action(grad_h, b_hat, radius, 2.0))) == 0.0


def test_the_feasibility_threshold_is_sharp_in_the_identification_radius() -> None:
    drift, channel, u_max, alpha_h = -0.9, 0.6, 2.0, 0.5
    d_star = identification_radius_threshold(drift, channel, u_max, alpha_h)
    assert d_star == pytest.approx(0.4)  # channel - deficit/u_max, deficit = 0.4

    assert robust_barrier_margin(drift, channel, d_star - 1e-6, u_max) >= -alpha_h
    assert robust_barrier_margin(drift, channel, d_star + 1e-6, u_max) < -alpha_h


def test_no_deficit_means_no_threshold_rather_than_a_large_one() -> None:
    """With the drift already satisfying the barrier, no identification radius can break safety."""
    assert identification_radius_threshold(-0.1, 0.6, 2.0, 0.5) == float("inf")
    for radius in (0.0, 5.0, 500.0):
        assert robust_barrier_margin(-0.1, 0.6, radius, 2.0) >= -0.5


def test_gamma_star_inverts_the_msm_radius_exactly() -> None:
    """Round-trip through §32: the radius at ``Gamma*`` is the threshold radius, to machine zero."""
    cvar_gap, grad_norm, threshold = 1.0, 1.0, 0.4
    gamma_star = barrier_gamma_star(threshold, cvar_gap, grad_norm)
    assert gamma_star == pytest.approx(7.0 / 3.0)
    radius_at_star = confounding_robust_inflation(cvar_gap, 0.0, gamma_star) * grad_norm
    assert radius_at_star == pytest.approx(threshold, abs=1e-12)


def test_gamma_star_reports_the_degenerate_ends_instead_of_a_misleading_number() -> None:
    assert barrier_gamma_star(0.0, 1.0, 1.0) == 1.0  # no slack: any confounding breaks it
    assert barrier_gamma_star(2.0, 1.0, 1.0) == float("inf")  # beyond what the model can produce
    with pytest.raises(ValueError, match="must be positive"):
        barrier_gamma_star(0.4, 0.0, 1.0)


def test_a_nominally_infeasible_barrier_is_not_reported_as_certified_at_gamma_one() -> None:
    """``d* < 0`` means no radius works -- including the zero radius that ``Gamma = 1`` produces."""
    drift, channel, u_max, alpha_h = -2.0, 0.6, 2.0, 0.5
    deficit = -alpha_h - drift
    assert deficit > u_max * channel  # full authority on a perfect channel still falls short
    assert robust_barrier_margin(drift, channel, 0.0, u_max) < -alpha_h

    assert np.isnan(identification_radius_threshold(drift, channel, u_max, alpha_h))
    assert np.isnan(barrier_gamma_star(-0.15, 1.0, 1.0))


def test_the_margin_loss_saturates_once_the_radius_swallows_the_channel() -> None:
    """First order in the radius while authority lasts, then flat -- not affine everywhere."""
    drift, channel, u_max = -1.0, 0.6, 2.0
    nominal = robust_barrier_margin(drift, channel, 0.0, u_max)
    for radius in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
        loss = nominal - robust_barrier_margin(drift, channel, radius, u_max)
        assert loss == pytest.approx(u_max * min(radius, channel))
    assert nominal - robust_barrier_margin(drift, channel, 50.0, u_max) == pytest.approx(
        u_max * channel
    )


def test_zero_is_optimal_exactly_when_the_identified_interval_contains_it() -> None:
    """The general asymmetric zero-action rule; ``d >= |g|`` is its symmetric-ball special case."""
    u_max = 2.0

    def best_asymmetric(drift: float, g: float, d_lo: float, d_hi: float) -> float:
        """Brute-force maximiser of the worst case over ``g_true`` in ``[g - d_lo, g + d_hi]``."""
        grid = np.linspace(-u_max, u_max, 20001)
        worst = drift + np.where(grid >= 0.0, (g - d_lo) * grid, (g + d_hi) * grid)
        return float(grid[int(np.argmax(worst))])

    assert best_asymmetric(-1.0, 0.6, 0.2, 0.9) == pytest.approx(u_max)  # [0.4, 1.5]: act fully
    assert best_asymmetric(-1.0, 0.6, 0.9, 0.2) == pytest.approx(
        0.0, abs=2e-4
    )  # [-0.3, 0.8]: stand
    assert best_asymmetric(-1.0, -0.6, 0.2, 0.3) == pytest.approx(-u_max)  # [-0.8, -0.3]: act down
    assert best_asymmetric(-1.0, -0.6, 0.2, 0.9) == pytest.approx(0.0, abs=2e-4)  # [-0.8, 0.3]


def test_safety_is_first_order_in_the_effect_error_while_performance_is_second() -> None:
    """The dichotomy: the envelope theorem protects an objective, not a binding constraint."""
    curve = barrier_confounding_certificate()
    assert curve.safety_slope == pytest.approx(1.0, abs=1e-6)
    assert curve.regret_slope == pytest.approx(2.0, abs=1e-6)
    assert curve.ok


def test_the_certificate_sweep_crosses_the_threshold_and_the_zero_action_rule() -> None:
    """A sweep that certified everything would be evidence of nothing."""
    curve = barrier_confounding_certificate()
    assert curve.gamma_star == pytest.approx(7.0 / 3.0)
    assert curve.certified[0]  # certified at Gamma = 1 ...
    assert not curve.certified[-1]  # ... and refused at the top of the grid: both regimes appear
    assert curve.last_certified_gamma == 2.0
    assert curve.actions[0] > 0.0
    assert curve.actions[-1] == 0.0  # the zero-action rule fires once the radius eats the channel


def test_the_filter_returns_the_admissible_action_closest_to_the_nominal() -> None:
    """Least-restrictive, in the exact sense: clipped into the certified interval, not rescaled."""
    lo, hi = admissible_action_interval(channel=-1.0, radius=0.1, u_max=6.0, drift=0.6, alpha_h=0.0)
    assert lo == pytest.approx(-6.0)
    assert hi == pytest.approx(0.6 / 1.1)  # the u >= 0 branch has slope channel - radius = -1.1

    assert robust_safety_filter(0.2, -1.0, 0.1, 6.0, 0.6, 0.0) == pytest.approx(0.2)  # already safe
    assert robust_safety_filter(5.0, -1.0, 0.1, 6.0, 0.6, 0.0) == pytest.approx(hi)  # clipped to it
    assert robust_safety_filter(-3.0, -1.0, 0.1, 6.0, 0.6, 0.0) == pytest.approx(-3.0)  # braking


def test_a_satisfied_barrier_does_not_make_every_action_admissible() -> None:
    """Regression: ``deficit <= 0`` means ``u = 0`` is admissible, not that the interval is the box.

    Getting this wrong let the task controller spend the drift's slack and cross the boundary while
    the filter reported no restriction -- which is precisely the failure the filter exists to stop.
    """
    lo, hi = admissible_action_interval(channel=-1.0, radius=0.0, u_max=6.0, drift=0.6, alpha_h=0.0)
    assert hi == pytest.approx(0.6)  # NOT u_max
    assert lo == pytest.approx(-6.0)  # braking is unrestricted, as it should be

    margin = 0.6 + (-1.0) * hi - 0.0 * abs(hi)
    assert margin == pytest.approx(0.0, abs=1e-12)  # exactly on the boundary, so the clip is tight


def test_an_empty_admissible_set_is_reported_rather_than_silently_clipped() -> None:
    unreachable = admissible_action_interval(
        channel=0.2, radius=0.0, u_max=1.0, drift=-5.0, alpha_h=0.0
    )
    assert np.isnan(unreachable[0])  # deficit 5.0 needs u = 25, far past the actuation limit
    assert robust_safety_filter(0.5, 0.2, 0.0, 1.0, -5.0, 0.0) == pytest.approx(1.0)  # best effort

    unidentified = admissible_action_interval(
        channel=0.2, radius=0.5, u_max=6.0, drift=-1.0, alpha_h=0.0
    )
    assert np.isnan(unidentified[0])
    assert robust_safety_filter(3.0, 0.2, 0.5, 6.0, -1.0, 0.0) == 0.0  # the zero-action rule


def test_the_safety_filter_holds_the_limit_where_a_regret_sized_budget_does_not() -> None:
    """The operational half of §40, measured in closed loop rather than argued from slopes."""
    bench = safety_filter_benchmark()
    by_name = dict(zip(bench.controllers, bench.violation_rate, strict=True))
    assert by_name["oracle"] == 0.0
    assert by_name["safety_calibrated"] == 0.0
    assert by_name["regret_calibrated"] > 0.5  # the same radius, spent on the objective, does not
    assert by_name["greedy"] > 0.5
    assert bench.ok


def test_the_safety_rows_pay_for_the_guarantee_in_tracking_cost() -> None:
    """No free lunch, and the honest floor: the reference is unreachable inside the safe set."""
    bench = safety_filter_benchmark()
    cost = dict(zip(bench.controllers, bench.tracking_cost, strict=True))
    floor = (2.0 - 3.0) ** 2  # (x_limit - x_ref)**2, the best a safe controller can do
    assert (
        cost["safety_calibrated"] > cost["oracle"] > floor
    )  # the radius costs, and so does safety
    assert cost["greedy"] < floor  # only reachable by leaving the safe set
    assert 1.0 < bench.gamma_star < float("inf")  # a real ceiling, not a degenerate one


def test_the_multivariate_channel_is_attained_not_merely_bounded() -> None:
    """Cauchy-Schwarz is tight, so the vector formula is the same statement, not a relaxation."""
    rng = np.random.default_rng(0)
    grad_h, b_hat = rng.standard_normal(4), rng.standard_normal((4, 3))
    radius, u_max, drift = 0.2, 1.5, -0.3

    channel = control_channel(grad_h, b_hat)
    best = robust_safe_action(grad_h, b_hat, radius, u_max)
    attained = drift + float((b_hat.T @ grad_h) @ best) - radius * float(np.linalg.norm(best))
    assert attained == pytest.approx(
        robust_barrier_margin(drift, channel, radius, u_max), abs=1e-12
    )
    assert float(np.linalg.norm(best)) == pytest.approx(u_max)  # and it spends the full authority


def test_a_negative_actuation_limit_is_rejected_rather_than_flipping_the_bound() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        robust_barrier_margin(0.0, 1.0, 0.1, -1.0)
    with pytest.raises(ValueError, match="must be positive"):
        identification_radius_threshold(-0.9, 0.6, 0.0, 0.5)
