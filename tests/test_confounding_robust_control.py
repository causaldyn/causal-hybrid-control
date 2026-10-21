"""chc.regret: the minimax confounding-robust controller under asymmetric loss.

Under a symmetric loss CE is already minimax (pessimism = centre). Under an ASYMMETRIC loss the
pessimism radius shifts the gain and strictly beats CE (Rocq ``confounding_robust_control.v``).
"""

import numpy as np
import pytest

from chc.research.regret import (
    _confounded_effect_estimate,
    asymmetric_control_improvement,
    certainty_equivalence_control,
    confounding_robust_control,
    confounding_robust_control_benchmark,
    confounding_robust_control_certificate,
    confounding_robust_tracking_benchmark,
    confounding_robust_tracking_loop,
    worst_case_asymmetric_loss,
)


def _numeric_improvement(bhat: float, d: float, tau: float, a: float, b: float) -> float:
    u_ce = certainty_equivalence_control(bhat, tau)
    u_rob = confounding_robust_control(bhat, d, tau, a, b)
    w_ce = worst_case_asymmetric_loss(u_ce, bhat, d, tau, a, b)
    w_rob = worst_case_asymmetric_loss(u_rob, bhat, d, tau, a, b)
    return w_ce - w_rob


def test_certificate_confirms_shift_and_strict_improvement() -> None:
    cert = confounding_robust_control_certificate()
    assert cert.ok
    assert cert.u_robust == pytest.approx(cert.numeric_argmin, abs=1e-3)  # closed form == minimax
    assert cert.u_robust < cert.u_certainty_equivalence  # overshoot costlier -> gain shifted down
    assert cert.worst_case_loss_robust < cert.worst_case_loss_ce  # strictly better worst case
    assert cert.symmetric_equals_ce


def test_symmetric_loss_recovers_certainty_equivalence() -> None:
    # alpha=beta: the minimax control is the CE centre (the symmetric "pessimism = centre")
    u_ce = certainty_equivalence_control(1.4, 1.0)
    u_rob = confounding_robust_control(1.4, 0.3, 1.0, 2.0, 2.0)
    assert u_rob == pytest.approx(u_ce)


def test_gain_shifts_further_with_more_confounding_and_more_asymmetry() -> None:
    u_ce = certainty_equivalence_control(1.3, 1.0)
    small = confounding_robust_control(1.3, 0.1, 1.0, 3.0, 1.0)
    wide = confounding_robust_control(1.3, 0.4, 1.0, 3.0, 1.0)
    skewed = confounding_robust_control(1.3, 0.1, 1.0, 9.0, 1.0)
    assert small < u_ce  # any asymmetry + confounding shifts the gain down
    assert wide < small  # more confounding -> more conservative
    assert skewed < small  # more asymmetry -> more conservative


def test_undershoot_costlier_shifts_the_gain_up() -> None:
    # beta > alpha (undershoot worse): the robust controller becomes MORE aggressive than CE
    u_ce = certainty_equivalence_control(1.3, 1.0)
    u_rob = confounding_robust_control(1.3, 0.25, 1.0, 1.0, 3.0)
    assert u_rob > u_ce  # push harder to avoid the costlier undershoot


def test_worst_case_loss_is_the_max_of_the_weighted_tails() -> None:
    # a control that overshoots badly: the alpha-weighted overshoot tail dominates
    w = worst_case_asymmetric_loss(1.0, 1.3, 0.25, 1.0, 4.0, 1.0)
    over = 4.0 * ((1.3 + 0.25) * 1.0 - 1.0)  # alpha * overshoot at b_hat+D
    assert w == pytest.approx(over)


def test_piecewise_improvement_matches_numeric_in_both_regimes() -> None:
    # reviewer-8: the analytic gap is piecewise in max(alpha,beta); it must match the numeric
    # W_CE - W_rob in BOTH the overshoot-dominant (alpha>beta) and undershoot-dominant (beta>alpha,
    # marketplace regime) branches -- the old single formula only covered alpha>=beta
    over = asymmetric_control_improvement(1.3, 0.25, 1.0, 4.0, 1.0)  # alpha>beta
    under = asymmetric_control_improvement(1.3, 0.25, 1.0, 1.0, 4.0)  # beta>alpha (churn 4x)
    assert over == pytest.approx(_numeric_improvement(1.3, 0.25, 1.0, 4.0, 1.0), abs=1e-12)
    assert under == pytest.approx(_numeric_improvement(1.3, 0.25, 1.0, 1.0, 4.0), abs=1e-12)
    assert over > 0.0  # overshoot-dominant branch strictly beats CE
    assert under > 0.0  # undershoot-dominant branch (marketplace regime) strictly beats CE too
    assert asymmetric_control_improvement(1.3, 0.25, 1.0, 2.0, 2.0) == pytest.approx(0.0)  # a=b


# --- Grounding on a synthetic marketplace task (full estimate -> control pipeline) ---


def test_the_benchmarks_naive_estimate_is_the_least_squares_slope() -> None:
    """Up to 0.12 a covariance over ``n - 1`` was divided by a variance over ``n``, which made the
    slope ``n / (n - 1)`` too large: 1.0025 at the benchmarks' 400 periods."""
    n = 400
    estimate = _confounded_effect_estimate(2.0, 0.8, 1.0, 0.6, n, np.random.default_rng(0))
    rng = np.random.default_rng(0)  # the same draws, in the order the benchmark takes them
    z = rng.standard_normal(n)
    u = z + 0.6 * rng.standard_normal(n)
    y = 2.0 * u + 0.8 * z + 0.6 * rng.standard_normal(n)
    assert estimate == pytest.approx(np.polyfit(u, y, 1)[0], rel=1e-12, abs=0.0)


def test_benchmark_robust_bounds_the_downside_and_wins_under_confounding() -> None:
    curve = confounding_robust_control_benchmark()
    assert curve.ok
    assert curve.robust_worst_case < curve.ce_worst_case  # pessimism bounds the worst-case cost
    assert curve.savings_at_target_pct > 0.0  # robust wins where confounding is real


def test_benchmark_ce_cost_degrades_with_confounding() -> None:
    curve = confounding_robust_control_benchmark()
    ce = curve.ce_costs
    # CE trusts the biased estimate: its realised cost climbs as the confounding grows
    assert ce[0] < ce[-1]
    assert all(ce[i] <= ce[i + 1] + 1e-9 for i in range(len(ce) - 1))


def test_benchmark_pays_an_honest_premium_when_unconfounded() -> None:
    curve = confounding_robust_control_benchmark()
    zero_idx = list(curve.confounding_levels).index(0.0)
    # no confounding: the conservative robust controller over-serves, costing more than CE (honest)
    assert curve.robust_costs[zero_idx] >= curve.ce_costs[zero_idx]
    assert curve.unconfounded_premium_pct > 0.0


# --- DYNAMIC grounding -- a real closed-loop controller on a confounded plant ---


def test_tracking_loop_halfwidth_zero_is_the_ce_baseline() -> None:
    # the minimax formula collapses to tau/b_hat at halfwidth=0, so a zero-radius loop IS CE
    _, us, _ = confounding_robust_tracking_loop(0.6, 2.0, 2.5, 0.0, 1.0, 1.0, 4.0, n_steps=5)
    # first step: tau_0 = x_target - a*x0 = 1 - 0 = 1, u_0 = tau_0 / b_hat = 1/2.5
    assert us[0] == pytest.approx(certainty_equivalence_control(2.5, 1.0))


def test_tracking_loop_robust_pushes_the_gain_up_under_undershoot_cost() -> None:
    # biased b_hat (2.5 > b_true) makes CE under-actuate; with churn 4x the robust loop pushes u up
    _, us_ce, cost_ce = confounding_robust_tracking_loop(
        0.6, 2.0, 2.5, 0.0, 1.0, 1.0, 4.0, n_steps=30
    )
    _, us_rob, cost_rob = confounding_robust_tracking_loop(
        0.6, 2.0, 2.5, 1.0, 1.0, 1.0, 4.0, n_steps=30
    )
    assert us_rob[0] > us_ce[0]  # radius shifts the per-step gain up to hedge costlier undershoot
    assert cost_rob < cost_ce  # ...cutting the accumulated closed-loop asymmetric cost


def test_dynamic_benchmark_robust_bounds_the_closed_loop_downside() -> None:
    curve = confounding_robust_tracking_benchmark()
    assert curve.ok
    assert curve.robust_worst_case < curve.ce_worst_case  # pessimism bounds the worst-case cost
    assert curve.savings_at_target_pct > 0.0  # robust wins in closed loop where confounding is real


def test_dynamic_benchmark_ce_cost_climbs_and_premium_is_bounded() -> None:
    curve = confounding_robust_tracking_benchmark()
    ce = curve.ce_costs
    assert ce[0] < ce[-1]  # CE trusts the biased estimate: closed-loop cost climbs with confounding
    assert all(ce[i] <= ce[i + 1] + 1e-9 for i in range(len(ce) - 1))
    zero_idx = list(curve.confounding_levels).index(0.0)
    assert curve.robust_costs[zero_idx] >= curve.ce_costs[zero_idx]  # honest premium at zero conf.


@pytest.mark.parametrize(
    ("overshoot", "undershoot", "halfwidth"),
    [(3.0, 1.0, 0.25), (1.0, 3.0, 0.25), (1.0, 4.0, 0.4), (2.0, 2.0, 0.25), (3.0, 1.0, 0.0)],
)
def test_the_certificate_holds_on_both_sides_of_the_sign_dichotomy(
    overshoot: float, undershoot: float, halfwidth: float
) -> None:
    """Overshoot costlier shifts the gain below CE, undershoot costlier above it, and a symmetric
    loss or an interval with no width leaves it at CE with no gain: up to 0.18.0 ``ok`` asked for a
    shift down and a strict gain whatever the penalties."""
    cert = confounding_robust_control_certificate(
        halfwidth=halfwidth, overshoot_penalty=overshoot, undershoot_penalty=undershoot
    )
    assert cert.ok
    if overshoot > undershoot and halfwidth > 0.0:
        assert cert.u_robust < cert.u_certainty_equivalence
    elif undershoot > overshoot:
        assert cert.u_robust > cert.u_certainty_equivalence
    else:
        assert cert.u_robust == pytest.approx(cert.u_certainty_equivalence, rel=1e-12, abs=0.0)


def _grid_minimax(bhat: float, d: float, tau: float, a: float, b: float) -> tuple[float, float]:
    """The worst-case loss's minimum over a grid of controls of either sign, by brute force."""
    controls = np.linspace(-4.0, 4.0, 800_001)
    ends = np.stack([(bhat - d) * controls, (bhat + d) * controls])
    worst = np.maximum(
        a * np.maximum(ends.max(axis=0) - tau, 0.0), b * np.maximum(tau - ends.min(axis=0), 0.0)
    )
    k = int(np.argmin(worst))
    return float(controls[k]), float(worst[k])


@pytest.mark.parametrize(
    ("overshoot", "undershoot", "halfwidth"), [(1.0, 2.0, 1.5), (1.0, 3.0, 1.5), (3.0, 1.0, 1.01)]
)
def test_an_interval_past_the_estimate_is_refused(
    overshoot: float, undershoot: float, halfwidth: float
) -> None:
    """With ``D > b_hat`` the interval holds effects of both signs and the minimax control is 0:
    at ``b_hat = 1``, ``D = 1.5``, ``beta = 3`` the closed form returned 4, worst-case loss 9
    against the minimax 3, and the improvement over CE read -1 at ``beta = 2`` where it is 1."""
    control, loss = _grid_minimax(1.0, 1.5, 1.0, 1.0, 3.0)
    assert control == pytest.approx(0.0, abs=1e-5)
    assert loss == pytest.approx(3.0, rel=1e-12, abs=0.0)
    for function in (confounding_robust_control, asymmetric_control_improvement):
        with pytest.raises(ValueError, match="one sign"):
            function(1.0, halfwidth, 1.0, overshoot, undershoot)
    with pytest.raises(ValueError, match="one sign"):
        confounding_robust_control(1.0, -0.1, 1.0, overshoot, undershoot)


@pytest.mark.parametrize(("overshoot", "undershoot"), [(3.0, 1.0), (1.0, 4.0)])
def test_a_negative_target_is_the_mirror_minimax(overshoot: float, undershoot: float) -> None:
    """A negative target needs a negative control, whose outcome is largest at ``b_hat - D``:
    the penalties trade places. The closed form for a positive target returned -0.702 at
    ``(3, 1)``, worst-case loss 0.789 where the minimax is 0.319, and the worst-case loss of any
    negative control read 0."""
    bhat, d, tau = 1.3, 0.25, -1.0
    control, loss = _grid_minimax(bhat, d, tau, overshoot, undershoot)
    robust = confounding_robust_control(bhat, d, tau, overshoot, undershoot)
    assert robust == pytest.approx(control, abs=2e-5)
    worst = worst_case_asymmetric_loss(robust, bhat, d, tau, overshoot, undershoot)
    assert worst == pytest.approx(loss, rel=1e-4, abs=0.0)
    assert asymmetric_control_improvement(bhat, d, tau, overshoot, undershoot) == pytest.approx(
        _numeric_improvement(bhat, d, tau, overshoot, undershoot), rel=1e-12, abs=0.0
    )
