"""chc.regret: §35 minimax confounding-robust controller under asymmetric loss.

Under a symmetric loss CE is already minimax (pessimism = centre, §33). Under an ASYMMETRIC loss the
pessimism radius shifts the gain and strictly beats CE (Rocq ``confounding_robust_control.v``).
"""

import pytest

from chc.regret import (
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
    # alpha=beta: the minimax control is the CE centre (recovers §33's "pessimism = centre")
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
    # Result 37's regime) branches -- the old single formula only covered alpha>=beta
    over = asymmetric_control_improvement(1.3, 0.25, 1.0, 4.0, 1.0)  # alpha>beta
    under = asymmetric_control_improvement(1.3, 0.25, 1.0, 1.0, 4.0)  # beta>alpha (churn 4x)
    assert over == pytest.approx(_numeric_improvement(1.3, 0.25, 1.0, 4.0, 1.0), abs=1e-12)
    assert under == pytest.approx(_numeric_improvement(1.3, 0.25, 1.0, 1.0, 4.0), abs=1e-12)
    assert over > 0.0  # overshoot-dominant branch strictly beats CE
    assert under > 0.0  # undershoot-dominant branch (Result 37 regime) strictly beats CE too
    assert asymmetric_control_improvement(1.3, 0.25, 1.0, 2.0, 2.0) == pytest.approx(0.0)  # a=b


# --- Result 37: grounding on a synthetic marketplace task (full estimate -> control pipeline) ---


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


# --- Result 38: DYNAMIC grounding -- a real closed-loop controller on a confounded plant ---


def test_tracking_loop_halfwidth_zero_is_the_ce_baseline() -> None:
    # the §35 formula collapses to tau/b_hat at halfwidth=0, so a zero-radius loop IS CE
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
