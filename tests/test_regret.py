"""LQ regret guarantee: certainty-equivalence suboptimality is quadratic in the model error."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from chc.dynamics import DampedOscillator
from chc.lqr import linearize_discrete, linearized_regret_certificate
from chc.network_causal import cycle_shells
from chc.regret import (
    _matrix_gamma,
    _permutation_cycles,
    _sandwich_plan,
    adaptive_exploration_certificate,
    bandit_causal_certificate,
    capped_exploration_policy,
    causal_vs_predictive_certificate,
    ce_explicit_constant_certificate,
    certainty_equivalence_gap,
    closed_loop_cost,
    cluster_fold_leakage_certificate,
    clustered_lower_bound_certificate,
    composition_transfer_certificate,
    confounded_turnpike_certificate,
    conjugate_time_certificate,
    constrained_ce_regret_certificate,
    delayed_network_certificate,
    dlqr,
    doubly_robust_control_certificate,
    dynamic_causal_regret_certificate,
    end_to_end_c2_certificate,
    ensemble_control_certificate,
    exact_matrix_ratio_moment,
    exact_ratio_moment,
    exposure_map_certificate,
    finite_horizon_pl_certificate,
    highprob_regret_certificate,
    hinf_robust_regret_certificate,
    information_lower_bound_certificate,
    interference_convexity_certificate,
    interference_orthogonal_certificate,
    interference_regret_certificate,
    matrix_ratio_certificate,
    minimax_exploration_certificate,
    multichannel_control_certificate,
    multivariate_interference_certificate,
    multivariate_transfer_certificate,
    nonlinear_regret_certificate,
    optimal_exploration_certificate,
    optimal_fold_partition,
    orthogonal_control_certificate,
    partial_id_control_certificate,
    pessimism_variance_certificate,
    regret_scaling,
    strong_convexity_regret_bound,
    transportability_regret_certificate,
    van_trees_certificate,
)

A = np.array([[1.0, 0.1], [0.0, 0.95]])
B = np.array([[0.5], [1.0]])
Q = np.eye(2)
R = np.array([[0.5]])
X0 = np.array([1.0, 0.5])


def test_dlqr_solves_the_dare_and_stabilises() -> None:
    k, p = dlqr(A, B, Q, R)
    residual = A.T @ p @ A - p - A.T @ p @ B @ np.linalg.solve(R + B.T @ p @ B, B.T @ p @ A) + Q
    assert np.allclose(residual, 0.0, atol=1e-8)  # P satisfies the discrete algebraic Riccati eqn
    assert np.max(np.abs(np.linalg.eigvals(A - B @ k))) < 1.0  # the optimal loop is stable


def test_closed_loop_cost_matches_the_riccati_value_and_diverges_when_unstable() -> None:
    k, p = dlqr(A, B, Q, R)
    assert np.isclose(closed_loop_cost(A, B, k, Q, R, X0), float(X0 @ p @ X0))  # cost = x0' P x0
    no_control = np.zeros((1, 2))
    assert closed_loop_cost(A, B, no_control, Q, R, X0) == float("inf")  # A is marginally unstable


def test_gap_is_zero_at_the_true_model_and_positive_when_perturbed() -> None:
    assert abs(certainty_equivalence_gap(A, B, Q, R, A, B, X0)) < 1e-8  # exact model -> no regret
    a_hat = A + np.array([[0.0, 0.0], [0.02, -0.03]])
    b_hat = B + np.array([[0.01], [-0.02]])
    assert certainty_equivalence_gap(A, B, Q, R, a_hat, b_hat, X0) > 0.0  # misspecification costs


def test_regret_scales_quadratically_with_model_error() -> None:
    curve = regret_scaling(A, B, Q, R, X0, n_samples=300, seed=0)
    assert 1.7 < curve.exponent < 2.3  # Dean et al.: quadratic suboptimality (theory exponent 2)
    assert (np.diff(curve.gaps) < 0).all()  # gap shrinks monotonically as the error level drops


def test_interference_regret_is_quadratic_in_the_total_error() -> None:
    curve = interference_regret_certificate(A, B, Q, R, X0, interference_ratio=1.0, n_samples=300)
    assert 1.7 < curve.exponent < 2.3  # regret ~ (eid + eint)^2 (proofs/interference_regret.v)
    assert (np.diff(curve.gaps) < 0).all()  # gap shrinks as the total error drops


def test_sign_identification_threshold_and_partial_id_robust_control() -> None:
    # partial-ID control (proofs/partial_id_control.v): the action direction is robust iff Delta<|b|
    # (the sign-identification threshold); worst-case regret grows; minimax beats CE
    curve = partial_id_control_certificate()
    assert curve.sign_id_threshold == 1.0  # = |b_hat|
    below = curve.half_widths < curve.sign_id_threshold
    assert curve.sign_identified[below].all()  # action direction robust below the threshold
    assert not curve.sign_identified[~below].any()  # no longer identified at/above the threshold
    assert (curve.robust_worst_regret <= curve.ce_worst_regret + 1e-9).all()  # minimax beats CE
    assert (np.diff(curve.ce_worst_regret) > 0).all()  # worst-case regret grows with the interval


def test_doubly_robust_control_vanishes_if_either_model_correct() -> None:
    # DR version of result 0 (proofs/doubly_robust.v): the AIPW control effect's bias is the product
    # of the outcome and propensity errors, so regret -> 0 if EITHER model is correct (double
    # robustness), unlike outcome-regression (needs the outcome) or IPW (needs the propensity)
    curve = doubly_robust_control_certificate(n_seeds=16)
    assert curve.dr_outcome_ok < 1e-4  # outcome model correct -> AIPW consistent
    assert curve.dr_propensity_ok < 1e-4  # propensity model correct -> AIPW consistent
    assert curve.outcome_reg_fails > 0.05  # outcome-regression fails when its model is wrong
    assert curve.ipw_fails > 0.005  # IPW fails when its model is wrong
    assert curve.dr_propensity_ok < 0.01 * curve.outcome_reg_fails  # AIPW robust here
    assert curve.dr_slope > 2.7  # product-quartic (super-quadratic): beyond the single-robust rate


def test_online_causal_control_has_log_regret() -> None:
    # bandit / adaptive (proofs/bandit_causal.v): learning the effect online, de-confounded control
    # has O(log T) cumulative regret (doubling cum(T)/cum(T/2) -> 1); confounded has Theta(T) (-> 2)
    curve = bandit_causal_certificate()
    assert curve.deconfounded_doubling < 1.4  # sublinear (log T)
    assert curve.confounded_doubling > 1.7  # linear (Theta T)
    assert curve.confounded_regret[-1] > 10 * curve.deconfounded_regret[-1]  # confounded explodes
    assert (np.diff(curve.deconfounded_regret) >= -1e-9).all()  # cumulative regret nondecreasing


def test_control_regret_has_an_information_lower_bound() -> None:
    # Cramer-Rao lower bound (proofs/information_lower_bound.v): no unbiased causal controller beats
    # E[regret] >= C*sigma^2/(n*V_id). The efficient controller HITS the floor (rate 1/n matches the
    # online upper bound above -> O(log T) optimal); confounding steals V_id, raising the floor.
    curve = information_lower_bound_certificate(n_seeds=200)
    assert (curve.experimental_regret >= 0.7 * curve.cramer_rao_floor).all()  # above the floor
    ratio = curve.experimental_regret / curve.cramer_rao_floor
    assert (np.abs(ratio - 1.0) < 0.35).all()  # efficient estimator is tight against the CR bound
    assert (curve.confounded_floor > curve.cramer_rao_floor).all()  # confounding raises the floor
    assert curve.floor_ratio > 1.0  # less identifying information => strictly higher floor
    # Result 57: the unbiased CR floor is beatable AT a point -- a Hodges estimator drives the
    # regret there to exactly zero -- which is why it was never a minimax statement. The van Trees
    # floor, which assumes nothing about bias, is not beatable, and it separates the superefficient
    # estimator from the efficient one instead of being vacuous for both.
    assert curve.hodges_pointwise_ratio < 1e-6
    assert curve.hodges_bayes_ratio > 10.0
    assert 1.0 < curve.plugin_bayes_ratio < 0.1 * curve.hodges_bayes_ratio
    assert curve.van_trees_action_floor > 0.0
    assert -1.25 < curve.rate_slope < -0.75  # ~ 1/n rate: matches the online upper bound (optimal)


def test_van_trees_inequality_is_tight_for_the_gaussian_conjugate() -> None:
    # Contribution 3, formal (proofs/van_trees.v): the van-Trees (Bayesian CR) inequality that
    # Result 20 assumed. Bayes risk >= 1/(I_prior + n*I_data); the posterior-mean estimator hits it
    # (tight for Gaussian). Confounding shrinks the identifying info, raising the floor.
    curve = van_trees_certificate()
    assert (curve.empirical_mse >= 0.9 * curve.van_trees_bound).all()  # van Trees is a lower bound
    assert np.allclose(
        curve.empirical_mse, curve.van_trees_bound, rtol=0.15
    )  # tight for the Gaussian
    assert (curve.confounded_bound > curve.van_trees_bound).all()  # confounding raises the floor
    assert np.isclose(curve.tight_ratio, 1.0, atol=0.15)  # Bayes MSE hits the bound
    assert (np.diff(curve.empirical_mse) < 0).all()  # MSE falls with more observations


def test_adaptive_exploration_achieves_the_van_trees_sqrt_t_rate() -> None:
    # Contribution 3 (proofs/adaptive_exploration.v): control with online effect learning.
    # A tapering schedule v_t ~ 1/sqrt(t) reaches cumulative regret Theta(sqrt(T)) -- matching the
    # van-Trees sqrt(T) lower bound -- while greedy (no explore) is Theta(T) and the static v* of
    # Result 11 over-explores (~T). The optimal schedule decreases; not a static rule.
    curve = adaptive_exploration_certificate()
    assert 0.4 < curve.adaptive_slope < 0.65  # tapering schedule -> sqrt(T) cumulative regret
    assert curve.greedy_slope > 0.9  # never exploring -> linear (Theta T)
    assert curve.adaptive_regret[-1] < curve.greedy_regret[-1]  # adaptive beats greedy at large T
    assert curve.adaptive_regret[-1] < curve.static_regret[-1]  # ... and the over-exploring static
    assert (np.diff(curve.schedule) < 0).all()  # the optimal schedule TAPERS (decreasing)
    assert (curve.adaptive_regret >= curve.lower_bound - 1e-9).all()  # van-Trees lower bound
    assert 1.0 <= curve.adaptive_over_bound < 2.5  # matches the lower bound up to a constant
    # REGRESSION: the van-Trees numerator is K = C*sigma^2, so the bound is linear in sigma. The
    # certificate once computed C and used it as K, silently pinning sigma^2 = 1; this catches that.
    assert np.allclose(
        adaptive_exploration_certificate(sigma=1.0).lower_bound,
        2.0 * adaptive_exploration_certificate(sigma=0.5).lower_bound,
    )


def test_cluster_fold_leakage_prices_the_a8_violation() -> None:
    # Contribution 2, Result 43 (proofs/cluster_fold_leakage.v): assumption A8 asks for folds of
    # WHOLE clusters; the certificates used to split by row parity, leaving every cluster in both
    # folds. The finding is NOT a bias -- it is a change of estimator, visible only in the second
    # moment. Each block below can fail on its own.
    curve = cluster_fold_leakage_certificate()
    # (1) NO BIAS, either scheme, either nuisance class. Falsifies Result 43(a) if it fails.
    assert np.abs(curve.lin_bias_row).max() < 0.02
    assert np.abs(curve.lin_bias_cluster).max() < 0.02
    assert np.abs(curve.aware_bias_row).max() < 0.02
    assert np.abs(curve.aware_bias_cluster).max() < 0.02
    # (2) For the library's OWN linear nuisance the fold scheme washes out as G grows -- which is
    # what licenses the C2 numbers published before the fold was corrected.
    assert curve.lin_sd_ratio[-1] > 0.95
    assert curve.lin_sd_ratio[-1] > curve.lin_sd_ratio[0]
    # (3) THE A8 DISCRIMINATOR. With a cluster-aware nuisance, row folds estimate each cluster's
    # own effect from its own training rows and subtract it, annihilating the residual ICC;
    # whole-cluster folds cannot see the held-out cluster's effect, so they preserve it. This is
    # the assertion that makes this a test OF A8 rather than one that happens to pass under it.
    assert curve.resid_icc_row.max() < 0.02
    assert curve.resid_icc_cluster[-1] > 0.8 * curve.icc_grid[-1]
    assert np.all(np.diff(curve.resid_icc_cluster) > 0)  # tracks rho_ICC
    # (4) The fold-geometry constant, from the projector algebra: c(m,2) = m*(m+14)/(m+2)^2.
    assert abs(curve.c_fold_measured / curve.c_fold_predicted - 1.0) < 0.15
    # (5) THE LAW psi/c = 1 - rho_ICC is exact; a finite sample adds only the O(1/G) coupling
    # through the shared global nuisance, so the deviation must SHRINK with G. Antitone in rho.
    assert curve.law_maxdev < curve.law_maxdev_half_g  # the O(1/G) term, vanishing
    assert curve.law_maxdev < 0.15
    assert np.all(
        np.diff(curve.psi_normalised) < 0
    )  # antitone in rho_ICC (Rocq: leak_ratio_antitone)
    # (6) A genuine cluster-MEASURABLE exposure -- what a partial-interference spillover actually
    # is -- is reproduced exactly from the held-out row's own cluster, so row folds annihilate the
    # channel: it loses its variation entirely (Result 43(c)).
    assert curve.exposure_resid_var_row < 0.02 * curve.exposure_resid_var_cluster
    # (7) ... and the consequence is a bias of EXACTLY the annihilated coefficient, -b_s. This is
    # the one regime where the leak is not merely a variance question; cluster folds fix it.
    assert abs(curve.annihilated_bias_row - curve.annihilated_bias_predicted) < 0.05
    assert abs(curve.annihilated_bias_cluster) < 0.05
    assert abs(curve.annihilated_bias_row) > 20.0 * abs(curve.annihilated_bias_cluster)


def test_ce_explicit_constants_certify_a_ball_and_a_constant() -> None:
    # Contribution 1, Result 44 (proofs/ce_explicit_constants.v): every earlier C1 statement took
    # "regret <= cc*||dB||^2" as a HYPOTHESIS and cited Mania-Tu-Recht's LOCAL bound. Here the ball
    # and the constant are both computed. Each block below can fail on its own.
    curve = ce_explicit_constant_certificate()
    # (1) THE EXACT GAIN-GAP IDENTITY. Checked on gains FAR from optimal -- if it only held near the
    # optimum it would be the local bound again, not the identity that replaces it.
    assert curve.identity_residual < 1e-10
    # (2) The plant constants are well-posed: eta is a genuine decay rate, theta a genuine margin.
    assert 0.0 < curve.eta <= 1.0
    assert 0.0 < curve.theta < 1.0
    assert curve.rho > 0.0
    assert curve.c_const > 0.0
    # (3) THE CITED HYPOTHESIS, MEASURED. Uniform validity of the gain map's Lipschitz constant on
    # the ball (Konstantinov et al.; Sun) is the one citation left on the control side, so the sup
    # over the certified ball is measured against the derivative rather than assumed.
    assert curve.l_k_ball <= 1.2 * curve.l_k_local
    # (4) INSIDE the ball: the Lyapunov certificate holds, nothing destabilises, the bound binds.
    inside = curve.radius_multiples <= 1.0
    assert (curve.worst_lyapunov[inside] <= curve.lyapunov_ceiling).all()
    assert (curve.unstable_fraction[inside] == 0.0).all()
    assert (curve.worst_gap[inside] <= curve.bound[inside]).all()
    # (5) IT MUST BE ABLE TO FAIL. Far outside the ball the certainty-equivalent controller really
    # does destabilise the true plant, and the Lyapunov certificate really is violated -- otherwise
    # the ball would be doing no work and the certificate would prove nothing.
    assert curve.unstable_fraction[-1] > 0.0
    assert curve.worst_lyapunov[-1] > curve.lyapunov_ceiling
    # (6) The guarantee is conservative, but not vacuously so.
    assert 1.0 < curve.conservatism < 100.0


def test_regret_scaling_reports_the_unbounded_event_it_conditions_on() -> None:
    # Defect fixed with Result 44: regret_scaling used to `continue` past both unstabilisable
    # draws and draws whose CE gain destabilises the TRUE plant. There regret is +inf and E[R] does
    # not exist, so the reported exponent is CONDITIONAL -- the share is now returned, not dropped.
    a = np.array([[1.0, 0.1], [0.0, 0.95]])
    b = np.array([[0.5], [1.0]])
    q, r, x0 = np.eye(2), np.array([[0.5]]), np.array([1.0, 0.5])
    small = regret_scaling(a, b, q, r, x0, n_samples=200, seed=0)
    assert small.infinite_fraction.shape == small.errors.shape
    assert (small.infinite_fraction == 0.0).all()  # inside the certified regime: no unbounded draw
    # A perturbation far outside the certified ball DOES produce unbounded draws, and they are now
    # visible instead of silently deleted.
    big = regret_scaling(a, b, q, r, x0, errors=(4.0, 2.0), n_samples=200, seed=1)
    assert big.infinite_fraction.max() > 0.0


def test_conjugate_time_bounds_the_horizon_every_regret_constant_lives_on() -> None:
    # Result 45 (proofs/conjugate_time.v): Result 14's turnpike says a long horizon is BENIGN, but
    # only for a positive-definite stage cost. Under an indefinite one the backward Riccati solution
    # escapes in finite time, and every constant in this module is built on an object that stops
    # existing there. Three distinct pole orders are the claim, so they are checked separately.
    curve = conjugate_time_certificate()
    # (1) The conjugate point exists and is finite: an indefinite cost is what creates it.
    assert curve.mu > 0.0
    assert 0.0 < curve.t_conj < np.inf
    # (2) THE VALUE HAS A SIMPLE POLE whose residue is -r/b^2 -- free of a, q and the terminal
    # weight. Both the exponent and the constant are predictions, so both are checked.
    assert abs(curve.value_slope - 1.0) < 0.05
    assert abs(curve.value_residue[-1] - (-1.0)) < 1e-3
    # (3) THE SENSITIVITY POLE IS DOUBLE, not simple: differentiating the tangent squares the
    # secant. This is the order that actually prices the regret bound, and it is strictly worse
    # than the value's -- an inspection of the cost-to-go alone would understate the blow-up.
    assert abs(curve.sensitivity_slope - 2.0) < 0.05
    assert curve.sensitivity_slope > curve.value_slope + 0.5
    assert curve.sensitivity_residue[-1] > 0.1
    # (4) Result 44 has C ~ L_K^2, so the realised regret coefficient must carry FOURTH order.
    assert abs(curve.regret_slope - 4.0) < 0.1
    assert (np.diff(curve.regret_coefficient) > 0.0).all()
    # (5) THE COUNTER-INTUITIVE DIRECTION, and the reason this is an obstruction rather than a
    # curiosity: more control authority raises mu, so it moves the conjugate point EARLIER. A
    # stronger actuator buys a shorter certifiable horizon, not a longer one.
    assert curve.authority_shrinks_horizon < 1.0
    # (6) IT MUST BE ABLE TO NOT HAPPEN. With the sign of q flipped the same algebra runs with tanh
    # in place of tan: the solution saturates, so the sensitivity is finite and stops growing.
    # Without this arm the blow-up above could be an artefact of the sweep rather than of the cost.
    assert np.isfinite(curve.definite_sensitivity)
    assert abs(curve.definite_sensitivity_growth - 1.0) < 1e-3


def test_minimax_exploration_floor_is_valid_sharp_and_causal() -> None:
    # Contribution 3, the open item closed (proofs/minimax_exploration.v): the sequential bound is
    # now an inf over ALL policies, not over schedules, and its constant is explicit --
    # c_causal = 2*A*|du*/db|*sigma/sqrt(eta). Three separate claims, checked separately.
    curve = minimax_exploration_certificate()
    # (1) VALID: it is a lower bound, so no policy in the family may fall below it.
    assert curve.min_policy_ratio >= 1.0
    assert (curve.burst_regret >= curve.floor - 1e-9).all()
    assert (curve.greedy_regret >= curve.floor).all()
    assert (curve.constant_regret >= curve.floor).all()
    # (2) SHARP: the front-loaded design attains it, so the constant is not merely an order.
    assert 1.0 <= curve.burst_over_floor < 1.001
    # (3) TAPERING COSTS A CONSTANT: the 1/sqrt(t) schedule, at its own optimal scale, sits at
    # sqrt(2) times the floor -- rate-optimal but not constant-optimal (Rocq taper_gap_is_sqrt_two).
    assert abs(curve.taper_over_floor - np.sqrt(2.0)) < 0.01
    # and the causal content: the constant scales as eta^{-1/2} in the identification efficiency.
    assert abs(curve.eta_slope + 0.5) < 1e-6
    # greedy is linear in T, so it loses to the sqrt(T) designs by a growing margin
    assert curve.greedy_regret[-1] / curve.burst_regret[-1] > 100.0


def test_multichannel_control_needs_every_channel_orthogonalised() -> None:
    # Contribution 2 (proofs/multichannel_control.v): on a clustered network the total effect
    # B=b_d+b_s has two interference channels. Cross-fit Robinson DML: orthogonalising only the
    # direct caps regret at O(delta^2) (spillover bottleneck); orthogonalising both -> ~delta^4.
    # The effective sample size is the number of clusters G (estimate concentrates at 1/sqrt(G)).
    curve = multichannel_control_certificate()
    assert 1.6 < curve.half_slope < 2.5  # half-orth: spillover plug-in bottleneck -> O(delta^2)
    assert curve.full_slope > 3.0  # full-orth: both channels debiased -> ~O(delta^4)
    assert curve.full_slope > curve.half_slope + 1.0  # orthogonalising every channel lifts order
    assert -0.75 < curve.cluster_se_slope < -0.35  # estimate concentrates at ~1/sqrt(G) (cluster n)


def test_multivariate_interference_needs_every_channel_orthogonalised() -> None:
    # Contribution 2, multivariate (composes proofs/multichannel_control.v with multivariate LQ):
    # the total effect is an input MATRIX B = B_d + B_s. On a 2-state/1-input LQ, orthogonalising
    # only direct (spillover plug-in) caps the LQ regret at O(delta^2); both -> O(delta^4).
    curve = multivariate_interference_certificate()
    assert np.isclose(
        curve.half_slope, 2.0, atol=0.2
    )  # spillover plug-in bottleneck -> LQ ~ delta^2
    assert np.isclose(curve.full_slope, 4.0, atol=0.3)  # both channels debiased -> LQ ~ delta^4
    assert (
        curve.full_slope > curve.half_slope + 1.5
    )  # order-bottleneck in the multivariate LQ setting


def test_end_to_end_c2_two_regimes() -> None:
    # Contribution 2, END-TO-END (proofs/c2_end_to_end.v): multichannel causal estimation ->
    # bottleneck rate -> dynamic control regret on a clustered LQ network, R = O_p[G^-1 +
    # (sum delta^p)^2]. Two regimes: real cross-fit DML G-sweep (sampling-dominated, ~1/G) and a
    # deterministic delta-sweep (half ~ delta^2, full ~ delta^4).
    curve = end_to_end_c2_certificate()
    assert -1.3 < curve.g_slope < -0.6  # sampling floor: regret ~ 1/G with real cross-fit DML
    assert curve.regret_vs_g[-1] < curve.regret_vs_g[0]  # regret falls as clusters grow
    assert curve.floor_g > 0.0  # the 1/G sampling floor is irreducible (cluster_floor_irreducible)
    assert np.isclose(curve.half_slope, 2.0, atol=0.2)  # half-orth: spillover bottleneck -> delta^2
    assert np.isclose(curve.full_slope, 4.0, atol=0.3)  # full-orth -> delta^4
    assert curve.full_slope > curve.half_slope + 1.5  # every channel debiased lifts the order


def test_clustered_lower_bound_makes_the_sampling_floor_irreducible() -> None:
    # Contribution 2, the LOWER bound (proofs/clustered_van_trees.v): the G^-1 sampling regret is
    # IRREDUCIBLE, not just an upper bound. Clustered van Trees (effective info I0 + G*Ic) + the
    # lower-Lipschitz regret map give G*E[R] >= kappa0/(I0+Ic) > 0 for all G, up to kappa0/Ic.
    # Empirically G*regret is a flat, positive plateau -> regret ~ 1/G on BOTH sides (tight).
    curve = clustered_lower_bound_certificate()
    assert (
        curve.floor_positive > 0.0
    )  # uniform positive lower bound (regret_floor_uniform_positive)
    assert curve.c0_estimate > 0.0  # the kappa0/Ic plateau constant
    assert abs(curve.plateau_slope) < 0.3  # G*regret flat => regret ~ 1/G exactly, not o(1/G)
    ratio = float(np.max(curve.g_times_regret) / np.min(curve.g_times_regret))
    assert ratio < 2.0  # G*regret stays within a bounded band (a genuine plateau, not decay)


def test_exposure_map_has_three_channel_bottleneck() -> None:
    # Contribution 2, exposure-map generalization (proofs/exposure_map_c2.v): the marketplace plant
    # x_{t+1}=A x_t + (B_d + B_s W)u_t + eps has three effect channels (direct, spillover-coeff, and
    # exposure map W). R = O_p[G^-1 + (r_d+r_s+r_W)^2]. Orthogonalising all three -> delta^4; else
    # the exposure map W plug-in (r_W ~ delta) makes it the bottleneck -> ~delta^2.
    curve = exposure_map_certificate()
    assert np.isclose(
        curve.full_slope, 4.0, atol=0.3
    )  # all three channels orthogonalised -> delta^4
    assert np.isclose(
        curve.wbottleneck_slope, 2.0, atol=0.2
    )  # W left plug-in -> delta^2 bottleneck
    assert (
        curve.full_slope > curve.wbottleneck_slope + 1.5
    )  # r_W must be orthogonalised (Hays-Raghavan)


def test_transfer_theorem_holds_in_multivariate_lq() -> None:
    # Contribution 1, multivariate (shares proofs/composition_transfer.v, error-agnostic): on a
    # 2-state/1-input LQ plant the exact Dean et al. certainty-equivalence gap is quadratic in the
    # effect-matrix error, so an order-p estimator (||dB||~delta^p) gives LQ regret ~ delta^(2p) --
    # the same order-doubling as the scalar transfer, now for matrices.
    curve = multivariate_transfer_certificate()
    assert np.isclose(curve.slope_order1, 2.0, atol=0.2)  # ||dB|| ~ delta -> LQ regret ~ delta^2
    assert np.isclose(curve.slope_order2, 4.0, atol=0.3)  # ||dB|| ~ delta^2 -> LQ regret ~ delta^4
    assert (
        curve.slope_order2 > curve.slope_order1 + 1.5
    )  # order doubling in the multivariate setting


def test_control_map_doubles_the_estimator_order() -> None:
    # general orthogonal-to-control transfer (proofs/composition_transfer.v): an order-p effect
    # estimator (error delta^p) yields order-2p control regret -- the control map doubles the order,
    # for every p. Generalises Result 0 (p=1 -> 2 plug-in; p=2 -> 4 DML). Uses the exact regret map.
    curve = composition_transfer_certificate()
    assert np.allclose(curve.slopes, curve.expected_slopes, atol=0.35)  # regret slope ~ 2p
    assert (np.diff(curve.slopes) > 1.5).all()  # each higher order is ~2 steeper (order doubling)


def test_one_control_over_a_heterogeneous_population_pays_a_floor() -> None:
    # ensemble/heterogeneity floor (proofs/ensemble_control.v): one control over a heterogeneous
    # effect population pays an irreducible regret = curvature-weighted Var(u*), quadratic in the
    # heterogeneity and zero for a homogeneous population; weighted mean beats the naive u*(mean).
    curve = ensemble_control_certificate()
    assert 1.8 < curve.floor_slope < 2.2  # floor ~ heterogeneity^2
    assert curve.homogeneous_floor < 0.01 * curve.ensemble_floor[-1]  # vanishes when homogeneous
    assert (curve.ensemble_floor <= curve.naive_mean_regret + 1e-12).all()  # weighted mean optimal
    assert curve.naive_excess_max > 0.0  # curvature-weighting beats the naive mean
    assert (np.diff(curve.ensemble_floor) > 0).all()  # more heterogeneity => higher floor


def test_transportability_regret_is_quadratic_in_wasserstein_distance() -> None:
    # transportability (proofs/transportability_regret.v): a source-optimal controller deployed on a
    # target has ZERO regret if the effect is recoverable (transportable), else a residual quadratic
    # in W1(P,P'); a W-DRO radius covers it. Connects to chc.uncertainty.WassersteinPenalty.
    curve = transportability_regret_certificate()
    assert (curve.transportable_regret == 0).all()  # transportability => zero regret, any distance
    assert np.isclose(curve.nontransport_slope, 2.0)  # CE-quadratic regret ~ W1^2
    assert 1.7 < curve.exact_slope < 2.4  # the simulated cost-gap is also quadratic in W1
    assert (curve.nontransport_regret <= curve.wdro_bound + 1e-12).all()  # W-DRO covers d<=eps
    assert (np.diff(curve.nontransport_regret) > 0).all()  # regret grows with the domain shift


def test_regret_band_holds_with_high_probability() -> None:
    # high-prob regret bound (proofs/highprob_regret.v): the sub-Gaussian upgrade of Result 10 (in
    # expectation). With prob >= 1-delta, regret <= 2*log(2/delta) * the CR floor; confounding
    # (smaller V_id) widens the band. Concentration checked by Monte-Carlo coverage.
    curve = highprob_regret_certificate()
    assert (curve.empirical_coverage >= 1 - curve.deltas).all()  # coverage >= 1-delta
    assert np.allclose(curve.band_over_floor, 2 * np.log(2 / curve.deltas))  # band = 2*log(2/delta)
    assert (np.diff(curve.band_over_floor) > 0).all()  # smaller delta (more confidence) => wider
    assert (curve.confounded_bands > curve.highprob_bands).all()  # confounding widens the band
    assert curve.cr_floor > 0.0  # the Result 10 in-expectation floor exists


def test_confounded_controller_settles_at_the_wrong_turnpike() -> None:
    # confounded turnpike gap (proofs/confounded_turnpike.v): a confounded controller (biased effect
    # b_obs = b + beta) converges to the WRONG turnpike x_conf = b*xref/(b+beta) and pays the offset
    # every step. Undiscounted cumulative regret T*c is unbounded (sharpens Result 1d); discounted
    # sum stays below c/(1-g) -- finite via the discount.
    curve = confounded_turnpike_certificate()
    assert np.isclose(curve.turnpike_offset_simulated, curve.turnpike_offset_formula)  # gap formula
    assert curve.per_step_regret > 0.0  # the confounded controller pays an offset forever
    assert np.isclose(curve.undiscounted_slope, curve.per_step_regret, rtol=1e-6)  # linear: slope c
    assert (np.diff(curve.undiscounted_regret) > 0).all()  # undiscounted regret never settles
    assert (curve.discounted_regret <= curve.discounted_bound + 1e-9).all()  # discounted bounded
    assert np.isclose(curve.discounted_regret[-1], curve.discounted_bound, rtol=1e-3)  # saturated
    assert curve.undiscounted_regret[-1] > 10 * curve.discounted_regret[-1]  # undiscounted explodes


def test_constrained_ce_regret_is_piecewise_quadratic() -> None:
    # constrained CE regret (proofs/constrained_ce_regret.v): a budget cap u <= umax clips control.
    # True effect at the activation threshold: quadratic regret on the inactive side, ZERO on active
    # (control freezes at umax -> curvature collapses = the kink). Clipping is non-expansive
    # (constrained <= unconstrained); a naive controller pays the active-set gap.
    curve = constrained_ce_regret_certificate()
    assert 1.8 < curve.inactive_slope < 2.4  # quadratic within the inactive active-set cell
    assert curve.active_regret_max < 1e-9  # frozen on the active side: curvature collapses to 0
    assert curve.max_constrained_ratio <= 1.0 + 1e-9  # clipping is non-expansive (Rocq)
    assert (np.diff(curve.regret_inactive) > 0).all()  # regret grows with the effect-estimate error
    assert curve.pessimism_budget > 0.0  # a budget is needed to cover the active-set transition
    assert 0.0 < curve.threshold < 1.0  # a well-defined activation threshold b_t exists


def test_the_frozen_active_side_holds_only_inside_the_bounded_active_interval() -> None:
    # u*(b) = xt*b/(b^2+rr) is unimodal, so u*(b) = umax has TWO roots and the cap binds on the
    # bounded interval between them, with b_lo*b_hi = rr (validation/constrained_ce_regret.mac,
    # STEPs 4a-4g). The certificate probes the LOWER root, and "active-side regret is exactly 0"
    # is a consequence of its sweep staying inside that interval -- not a property of the sweep.
    xt, rr, umax = 1.0, 1.0, 0.45
    disc = np.sqrt(xt * xt - 4.0 * rr * umax * umax)
    b_lo, b_hi = (xt - disc) / (2.0 * umax), (xt + disc) / (2.0 * umax)
    assert np.isclose(b_lo * b_hi, rr)  # Vieta: the roots are reciprocal about sqrt(rr)
    assert b_lo < np.sqrt(rr) < b_hi  # du*/db vanishes INSIDE, so both thresholds are real kinks

    inside = constrained_ce_regret_certificate(xt=xt, rr=rr, umax=umax, delta_hi=0.15)
    assert np.isclose(inside.threshold, b_lo)  # the certificate probes the lower root
    assert inside.threshold + 0.15 < b_hi  # its widest probe stays active
    assert inside.active_regret_max < 1e-9  # ... hence frozen

    # Push the sweep past b_hi and the control un-freezes: the zero above is load-bearing, not luck.
    outside = constrained_ce_regret_certificate(xt=xt, rr=rr, umax=umax, delta_hi=1.2)
    assert outside.threshold + 1.2 > b_hi
    assert outside.active_regret_max > 1e-6


def test_hinf_robustness_is_pessimism_with_gamma_as_the_knob() -> None:
    # H-inf robust control (proofs/hinf_robust_regret.v): confounding as an adversary on the gain,
    # budget gamma^2. Robustness INFLATES cost above nominal (= pessimism), antitone in gamma (gamma
    # -> inf recovers CE); at the regret-optimal gamma* the robust gain reproduces the variance-
    # optimal pessimistic control (Result 2) -- two roads to the same cautious control.
    curve = hinf_robust_regret_certificate()
    assert (np.diff(curve.inflation_at_uce) < 0).all()  # antitone in the budget gamma (Rocq C)
    assert (curve.inflation_at_uce >= curve.nominal_at_uce - 1e-9).all()  # inflates (Rocq B)
    assert np.isclose(curve.inflation_at_uce[-1], curve.nominal_at_uce, rtol=0.03)  # -> nominal
    assert np.isclose(curve.robust_gain[-1], curve.u_ce, rtol=0.02)  # robust -> CE as gamma -> inf
    assert curve.u_pess_star < curve.u_ce  # pessimism shrinks the gain below certainty equivalence
    i = int(np.argmin(curve.expected_regret))
    assert 0 < i < curve.gamma_grid.size - 1  # interior regret-optimal robustness level gamma*
    span = abs(curve.u_ce - curve.u_pess_star)
    assert abs(curve.gain_at_gamma_star - curve.u_pess_star) < 0.15 * span  # ~ variance-optimal
    assert curve.expected_regret_min < 0.1 * curve.ce_expected_regret  # robust tuned beats naive CE


def test_optimal_exploration_balances_control_and_identification() -> None:
    # explore-exploit (proofs/optimal_exploration.v): excess = A*v + B/v, minimum at v* = sqrt(B/A)
    # -- interior (pure exploitation never optimal); confounding lifts both v* and the floor. Dual
    # of Result 10 (B is the CR floor exploration buys down).
    curve = optimal_exploration_certificate(n_seeds=400)
    i = int(np.argmin(curve.total_cost))
    assert 0 < i < curve.exploration_grid.size - 1  # interior optimum, not a boundary
    nearest = int(np.argmin(np.abs(np.log(curve.exploration_grid) - np.log(curve.vstar_theory))))
    assert i == nearest  # empirical argmin lands on the grid point closest to v* = sqrt(B/A)
    assert np.isclose(curve.total_cost.min(), curve.floor_theory, rtol=0.1)  # min ~ 2*sqrt(A*B)
    assert curve.vstar_confounded_theory > curve.vstar_theory  # confounding demands more exploring
    assert curve.vstar_confounded_empirical >= curve.vstar_empirical  # ... empirically too
    assert curve.total_cost_confounded.min() > curve.total_cost.min()  # ... at a higher floor
    assert (np.diff(curve.explore_cost) > 0).all()  # explore cost A*v rises with v
    assert (np.diff(curve.estimation_cost) < 0).all()  # estimation floor B/v falls with v


def test_finite_horizon_pl_bound_holds_over_the_trajectory() -> None:
    # finite-horizon PL bound (proofs/nonlinear_regret.v pl_mode_bound): over the full horizon the
    # self-certifying bound ‖∇J‖²/(2·λ_min(H)) upper-bounds the true regret and is tight along the
    # min-curvature eigenvector; the strong-convexity constant does not degrade with the horizon
    curve = finite_horizon_pl_certificate()
    assert (curve.bound_slack >= -1e-7).all()  # PL bound valid (upper-bounds regret) at every T
    assert np.allclose(curve.worst_mode_ratio, 1.0, atol=1e-6)  # tight in the min-curvature mode
    assert (curve.mu_min > 0.5).all()  # lambda_min bounded away from 0 (horizon-robust)


def test_interference_tightens_the_self_certifying_bound() -> None:
    # strong convexity under interference (proofs/interference_convexity.v): cannibalisation raises
    # the effective convexity mu_eff = mu + kappa, and the PL bound is antitone in it, so the
    # interference-aware bound is exact while the blind one over-states (more with kappa)
    curve = interference_convexity_certificate()
    assert np.allclose(curve.mu_eff, 1.0 + curve.cannibalisation)  # adds convexity
    assert np.allclose(curve.aware_bound, curve.true_regret)  # PL bound exact for the quadratic
    assert (curve.blind_bound + 1e-9 >= curve.true_regret).all()  # blind valid (upper bound)
    assert (curve.aware_bound <= curve.blind_bound + 1e-9).all()  # aware tighter than blind (Rocq)
    ratio = curve.blind_bound / curve.aware_bound
    assert ratio[-1] > ratio[0] + 0.5  # blind over-states more as interference grows


def test_interference_forces_double_debiasing() -> None:
    # interference x orthogonality (proofs/interference_orthogonal.v): under spillover you must
    # orthogonalise BOTH channels -- debiasing only the direct effect leaves the spillover error at
    # O(eps), so regret stays O(eps^2); only full double-debiasing reaches O(eps^4)
    curve = interference_orthogonal_certificate()
    assert 1.7 < curve.plugin_exponent < 2.7  # plug-in both -> O(eps^2)
    assert 1.7 < curve.half_orthogonal_exponent < 2.7  # orth direct only: spillover dominates
    assert curve.full_orthogonal_exponent > 3.3  # orth both channels -> O(eps^4)
    assert curve.full_orthogonal_exponent > curve.half_orthogonal_exponent + 1.0  # the gap


def test_confounded_control_regret_grows_with_the_horizon() -> None:
    # dynamic confounding theorem (proofs/dynamic_causal_mpc.v): a confounded effect estimate leaves
    # a steady-state offset paid every step, so cumulative regret grows linearly in T (slope =
    # per-step floor q*offset^2); the causal controller's cumulative cost stays bounded
    curve = dynamic_causal_regret_certificate()
    floor = 1.0 * (1.0 * 0.5 / 1.5) ** 2  # q * (x_ref*beta/b_obs)^2
    assert curve.growth_slope > 0.0
    assert abs(curve.growth_slope - floor) < 0.1 * floor  # slope equals the per-step floor
    assert curve.predictive_regret[-1] > 5 * curve.predictive_regret[0]  # grows (unbounded in T)
    assert abs(curve.causal_cost[-1] - curve.causal_cost[-2]) < 0.01  # causal cost bounded


def test_strong_convexity_regret_bound_holds_beyond_linearisation() -> None:
    # proofs/nonlinear_regret.v: for a mu-strongly-convex cost the self-certifying grad^2/(2 mu)
    # upper-bounds the true regret globally, while a fixed-Hessian estimate under-states it far out
    curve = nonlinear_regret_certificate()
    assert (curve.pl_bound + 1e-9 >= curve.true_regret).all()  # valid GLOBAL upper bound everywhere
    assert curve.linearized_estimate[-1] < curve.true_regret[-1]  # linearised under-states (unsafe)
    assert strong_convexity_regret_bound(0.0, mu=1.0) == 0.0  # zero gradient certifies zero regret
    tight = curve.pl_bound[1] / curve.true_regret[1]  # tight near the optimum (first nonzero point)
    assert tight < 1.3


def test_optimal_pessimism_equals_the_effect_variance() -> None:
    # pessimism in the optimality condition (proofs/pessimistic_optimality.v): the expected-regret-
    # minimising effective-effort rho* tracks the effect-estimate variance s^2, and the pessimistic
    # control beats the certainty-equivalent one under uncertainty
    curve = pessimism_variance_certificate()
    assert np.corrcoef(curve.variances, curve.optimal_rho)[0, 1] > 0.95  # rho* ~ s^2
    for var, rho in zip(curve.variances, curve.optimal_rho, strict=True):
        assert abs(rho - var) < 0.5 * var + 0.03  # optimal pessimism = variance (+ grid step)
    assert (curve.pessimistic_regret < curve.ce_regret).all()  # pessimism beats greedy


def test_predictive_control_is_asymptotically_wrong_under_confounding() -> None:
    # notebook-01 hardened into a theorem (proofs/causal_mpc.v): the observational (predictive)
    # controller inherits a systematic omitted-variable bias, so its regret plateaus at a positive
    # floor that does not vanish with n; the interventional (causal) controller converges to oracle
    curve = causal_vs_predictive_certificate(n_seeds=6)
    floor = curve.predictive_floor
    assert floor > 0.0  # confounding creates a positive control-regret floor
    assert curve.predictive_regret[-1] > 0.7 * floor  # predictive plateaus at it (never vanishes)
    assert curve.causal_regret[-1] < 0.1 * floor  # causal regret vanishes toward the oracle
    assert curve.predictive_regret[-1] > 10 * curve.causal_regret[-1] + 1e-3  # ID closes it


def test_orthogonal_control_is_doubly_debiased() -> None:
    # novel result (proofs/orthogonal_control.v): a controller built on the Neyman-orthogonal DML
    # effect has regret ~ eps^4 in the nuisance error, vs ~ eps^2 for a single-residualisation
    # plug-in -- the DML orthogonality compounds with the certainty-equivalence quadratic regret map
    curve = orthogonal_control_certificate(n=200_000, n_seeds=4)
    assert 1.8 < curve.single_exponent < 2.7  # plug-in control regret ~ eps^2
    assert curve.orthogonal_exponent > 3.3  # orthogonal DML control regret ~ eps^4
    assert curve.orthogonal_exponent > curve.single_exponent + 1.0  # the double-debiasing gap
    assert curve.orthogonal_regret[0] < 0.05 * curve.single_regret[0]  # far lower regret at top eps


def test_interference_strictly_increases_the_regret() -> None:
    # the empirical mirror of the Rocq lemma interference_strictly_worse: adding the exposure-map
    # error channel (eint > 0) enlarges the certificate over the interference-blind (eint = 0) case
    blind = interference_regret_certificate(A, B, Q, R, X0, interference_ratio=0.0, n_samples=300)
    aware = interference_regret_certificate(A, B, Q, R, X0, interference_ratio=1.0, n_samples=300)
    assert aware.gaps[0] > blind.gaps[0]  # interference is not free


def test_linearized_certificate_zero_at_truth_and_positive_under_model_error() -> None:
    dyn = DampedOscillator(omega=1.0, zeta=0.15)
    x_star, u_star = jnp.zeros(2), jnp.zeros(1)
    exact = linearized_regret_certificate(dyn, dyn, x_star, u_star, Q, R, X0, 0.1)
    wrong = DampedOscillator(omega=1.3, zeta=0.05)
    certificate = linearized_regret_certificate(dyn, wrong, x_star, u_star, Q, R, X0, 0.1)
    assert abs(exact) < 1e-8  # a correct model implies no certainty-equivalence regret
    assert certificate > 0.0  # a wrong model implies a positive local suboptimality certificate


def test_linearized_certificate_matches_the_lq_gap_on_the_linearisation() -> None:
    dyn, dyn_hat = DampedOscillator(omega=1.0, zeta=0.15), DampedOscillator(omega=1.2, zeta=0.1)
    x_star, u_star = jnp.zeros(2), jnp.zeros(1)
    a, b = linearize_discrete(dyn, x_star, u_star, 0.1)
    a_hat, b_hat = linearize_discrete(dyn_hat, x_star, u_star, 0.1)
    direct = certainty_equivalence_gap(
        np.asarray(a), np.asarray(b), Q, R, np.asarray(a_hat), np.asarray(b_hat), X0
    )
    cert = linearized_regret_certificate(dyn, dyn_hat, x_star, u_star, Q, R, X0, 0.1)
    assert np.isclose(cert, direct)  # the helper just linearises, then calls the LQ gap


def test_capped_exploration_front_loads_and_does_not_become_a_taper() -> None:
    # Result 56, validation/capped_exploration.mac, proofs/capped_exploration.v.

    # 1. against a projected-gradient solve of the full convex program: the front-loaded block is
    #    the GLOBAL optimum, not a heuristic. The objective is convex, so the reference is exact.
    horizon, cap = 400, 0.05
    b, rr, xt, sigma, i0, eta = 1.0, 0.5, 1.0, 0.7, 1.0, 0.6
    curvature = b * b + rr
    sensitivity = -xt * (rr - b * b) / (rr + b * b) ** 2
    numerator = curvature * sensitivity * sensitivity
    info_rate = eta / sigma**2

    def cumulative(schedule: np.ndarray) -> float:
        info = i0 + info_rate * np.concatenate([[0.0], np.cumsum(schedule)[:-1]])
        return float(np.sum(curvature * schedule + numerator / info))

    def gradient(schedule: np.ndarray) -> np.ndarray:
        info = i0 + info_rate * np.concatenate([[0.0], np.cumsum(schedule)[:-1]])
        weights = numerator / info**2
        tail = np.concatenate([np.cumsum(weights[::-1])[::-1][1:], [0.0]])
        return curvature - info_rate * tail

    guess = np.full(horizon, 1e-3)
    step, best = 1.0, cumulative(guess)
    for _ in range(4000):
        trial = np.clip(guess - step * gradient(guess), 0.0, cap)
        value = cumulative(trial)
        if value < best:
            guess, best, step = trial, value, step * 1.05
        else:
            step *= 0.5
            if step < 1e-14:
                break
    policy = capped_exploration_policy(horizon=horizon, cap=cap)
    assert policy.cost == pytest.approx(best, rel=1e-3)
    assert (guess > 0.999 * cap).sum() == pytest.approx(policy.block_rounds, abs=2)

    # 2. the conjecture the log recorded -- that a cap makes the taper the right shape -- is false
    assert policy.taper_cost > 1.2 * policy.cost

    # 3. the cap's price is ADDITIVE and logarithmic, so the ratio to the uncapped floor falls
    ratios = [
        capped_exploration_policy(horizon=t, cap=0.03).cost
        / capped_exploration_policy(horizon=t, cap=0.03).uncapped_floor
        for t in (10**3, 10**4, 10**5)
    ]
    assert ratios[0] > ratios[1] > ratios[2] > 1.0
    # ... while the block length GROWS like sqrt(T): a tighter deadline is not a gentler one
    lengths = [capped_exploration_policy(horizon=t, cap=0.03).block_rounds for t in (10**3, 10**5)]
    assert lengths[1] / lengths[0] == pytest.approx(10.0, rel=0.25)

    with pytest.raises(ValueError, match="at least one round"):
        capped_exploration_policy(horizon=0, cap=0.1)
    with pytest.raises(ValueError, match="cap must be positive"):
        capped_exploration_policy(horizon=10, cap=0.0)


def test_exact_matrix_ratio_moment_matches_the_haar_law_and_prices_the_two_channel_gap() -> None:
    # Result 54, validation/matrix_ratio_moment.mac, proofs/matrix_ratio_moment.v.

    # 1. Wishart: A = Sigma = I and Om = I_2n makes the object E[(X'X)^-1] = I/(n - 3)
    wishart = exact_matrix_ratio_moment(np.eye(6), np.eye(6), np.eye(12), nodes=32)
    assert np.allclose(wishart, np.eye(2) / 3.0, atol=1e-5)

    # 2. a CORRELATED numerator still has a closed form: X = UR with Haar U independent of R
    #    turns the sandwich into R^-1 U' Sigma U R^-T, and E[U' Sigma U] = (tr Sigma / n) I, so
    #    the answer is (tr Sigma / n) / (n - 3) times the identity -- an anchor the quadrature
    #    can only hit by getting the tilt and the three-form moment both right
    n = 7
    lags = np.arange(n)[:, None] - np.arange(n)[None, :]
    sigma = 0.5 ** np.abs(lags)
    haar = exact_matrix_ratio_moment(sigma, np.eye(n), np.eye(2 * n), nodes=32)
    assert np.allclose(haar, np.eye(2) * float(np.trace(sigma)) / n / (n - 3), atol=1e-5)

    # 3. existence is the inverse-Wishart threshold n >= q + 2 and is ENFORCED -- below it the
    #    integral diverges while a plug-in sandwich still quotes a number
    with pytest.raises(ValueError, match="inverse-Wishart"):
        exact_matrix_ratio_moment(np.eye(3), np.eye(3), np.eye(6), nodes=16)
    # the channel count is READ OFF regressor_cov's shape, so a qn-by-qn block is accepted for
    # any q; what is rejected is q outside the matrix route's own range
    with pytest.raises(ValueError, match="qn-by-qn"):
        exact_matrix_ratio_moment(np.eye(4), np.eye(4), np.eye(6), nodes=16)
    with pytest.raises(ValueError, match="q = 1 channels"):
        exact_matrix_ratio_moment(np.eye(4), np.eye(4), np.eye(4), nodes=16)
    with pytest.raises(ValueError, match="q = 4 channels"):
        exact_matrix_ratio_moment(np.eye(4), np.eye(4), np.eye(16), nodes=4)

    # 4. the application: a SINGULAR Om, because the spillover column is a deterministic map of
    #    the own column. The delayed-network panel on C_4, two shells, phi = 0.5.
    m, p_t, phi = 4, 2, 0.5
    gammas = np.array([1.0, 0.6])
    shells = [np.asarray(s) for s in cycle_shells(m, 1)]
    t_gap = np.arange(p_t)[:, None] - np.arange(p_t)[None, :]
    sig = np.zeros((m * p_t, m * p_t))
    for d in range(2):
        for e in range(2):
            block = phi ** np.abs(t_gap - (d - e))
            sig += gammas[d] * gammas[e] * np.kron(shells[d] @ shells[e], block)
    spillover = np.kron(shells[1] / 2.0, np.eye(p_t))
    om = np.block([[sig, sig @ spillover.T], [spillover @ sig, spillover @ sig @ spillover.T]])
    assert np.linalg.matrix_rank(om) == m * p_t  # rank n out of 2n: nothing may invert Om

    exact = exact_matrix_ratio_moment(sig, np.eye(m * p_t), om, nodes=36)
    coarse = exact_matrix_ratio_moment(sig, np.eye(m * p_t), om, nodes=24)
    assert np.allclose(exact, exact.T)
    assert np.allclose(exact, coarse, rtol=0.01)  # converged, unlike the divergent regime

    moments = np.empty((2, 2, 2))
    for which, op in enumerate((np.eye(m * p_t), sig)):
        for i in range(2):
            for j in range(2):
                selector = np.zeros((2, 2))
                selector[i, j] = 1.0
                moments[which, i, j] = np.trace(np.kron(selector.T, op) @ om)
    inverse = np.linalg.inv(moments[0])
    plug = inverse @ moments[1] @ inverse
    # the matrix Jensen gap has a sign: the plug-in sandwich UNDERSTATES both channel variances
    assert np.all(np.diag(exact) > np.diag(plug))


def test_general_q_ratio_moment_lifts_the_sandwich_off_two_channels() -> None:
    # Result 63, validation/general_q_ratio_moment.mac, proofs/general_q_ratio_moment.v.

    # 1. the S_m weights ARE Isserlis: sum over S_m of 2^(m - cycles) is the number of perfect
    #    matchings of 2m points, (2m-1)!!. This is the identity that lets the hand-written
    #    three-form expansion be deleted -- if it failed, every q would be wrong at once.
    for m, expected in ((1, 1), (2, 3), (3, 15), (4, 105), (5, 945)):
        total = sum(2 ** (m - len(cyc)) for cyc in _permutation_cycles(m))
        assert total == expected

    # 2. the plan's shape is the algebra: adj(M) has degree q - 1, so each entry is a sum of
    #    products of m = 2q - 1 forms, and S_m contributes m! permutations to each product
    for q, products in ((2, 12), (3, 216)):
        plan = _sandwich_plan(q)
        m = 2 * q - 1
        assert len(plan["idx"]) == products * math.factorial(m)
        assert plan["idx"].shape[1] == m
        assert len(plan["forms"]) == 2 * q * (q + 1) // 2  # q(q+1)/2 forms of M and of N

    # 3. Gamma_q(2) is finite iff 2 > (q-1)/2, so the ROUTE stops at q = 4 whatever the cost
    assert _matrix_gamma(2, 2.0) == pytest.approx(np.pi / 2)
    assert _matrix_gamma(3, 2.0) == pytest.approx(np.pi**2 / 2)
    assert _matrix_gamma(4, 2.0) == pytest.approx(np.pi**4 / 2)
    assert not np.isfinite(_matrix_gamma(5, 2.0))

    # 4. q = 3 against the Wishart anchor E[M^-1] = I/(n - q - 1). Two things are EXACT at any
    #    node count and one is not: the sandwich is symmetric, and under exchangeability its
    #    off-diagonal vanishes structurally -- while the diagonal carries the whole quadrature
    #    error, which on a 6-dimensional cone is large at a coarse grid and is not hidden here.
    n = 5
    got = exact_matrix_ratio_moment(np.eye(n), np.eye(n), np.eye(3 * n), nodes=3)
    assert got.shape == (3, 3)
    assert np.allclose(got, got.T)
    assert float(np.max(np.abs(got - np.diag(np.diag(got))))) < 1e-12

    # 5. and the error bar needs no reference value: the exact answer is isotropic, so half the
    #    observed spread LOWER-BOUNDS the largest entry error (half_spread_bounds_the_max_error).
    #    Necessary, not sufficient -- a grid can be isotropic and uniformly wrong.
    target = 1.0 / (n - 3 - 1)
    error = float(np.max(np.abs(np.diag(got) - target)))
    assert float(np.ptp(np.diag(got))) / 2 <= error + 1e-12


def test_matrix_ratio_certificate_reports_what_the_grid_is_worth() -> None:
    # Result 63 (d): at q = 3 the value alone hides a percent-scale quadrature error, and the
    # isotropy bar of (e) only exists on exchangeable problems. The refinement residual works on
    # any problem and is CONSERVATIVE -- measured 1.25x to 5.26x the true error, never below it.

    # 1. q = 2 is converged, so the certificate says so and the value is unchanged
    n = 6
    fine = exact_matrix_ratio_moment(np.eye(n), np.eye(n), np.eye(2 * n), nodes=32)
    cert = matrix_ratio_certificate(np.eye(n), np.eye(n), np.eye(2 * n), nodes=32)
    assert np.allclose(cert.value, fine)
    assert cert.nodes == 32
    assert cert.coarse_nodes == 31
    assert cert.ok
    assert cert.relative_residual < 1e-6

    # 2. q = 3 on a coarse grid is NOT converged, and the residual is what says so -- while the
    #    returned array on its own looks perfectly ordinary
    n = 5
    coarse = matrix_ratio_certificate(np.eye(n), np.eye(n), np.eye(3 * n), nodes=5)
    assert not coarse.ok
    assert coarse.relative_residual > 0.1

    # 3. and it never UNDER-states the true error, which is the property that makes it usable:
    #    the exact answer here is I/(n - q - 1), so the true error is computable
    truth = np.eye(3) / (n - 3 - 1)
    true_error = float(np.max(np.abs(coarse.value - truth)))
    assert coarse.residual >= true_error

    # 4. the contract: a tolerance must be positive, and a residual needs a coarser grid
    with pytest.raises(ValueError, match="tolerance must be positive"):
        matrix_ratio_certificate(np.eye(n), np.eye(n), np.eye(3 * n), tolerance=0.0)
    with pytest.raises(ValueError, match="at least 4"):
        matrix_ratio_certificate(np.eye(n), np.eye(n), np.eye(3 * n), nodes=3)


def test_exact_ratio_moment_closed_forms_tail_condition_and_crossover_immunity() -> None:
    # Result 51 (l)/(m), validation/omega_jensen_gap.mac, proofs/omega_jensen_gap.v.

    # 1. chi-square closed forms: B = C = I, Om = I gives E[1/chi2_n] = 1/(n - 2)
    assert abs(exact_ratio_moment(np.eye(3), np.eye(3), np.eye(3)) - 1.0) < 1e-9
    assert abs(exact_ratio_moment(np.eye(5), np.eye(5), np.eye(5)) - 1.0 / 3.0) < 1e-9

    # 2. numerator mass on the denominator's null space splits off an independent chi-square:
    #    E[X/Y^2] = E[1/chi2_5] + E[chi2_1] E[1/chi2_5^2] = 1/3 + 1/((5-2)(5-4)) = 2/3
    c_null = np.diag([1.0] * 5 + [0.0])
    assert abs(exact_ratio_moment(np.eye(6), c_null, np.eye(6)) - 2.0 / 3.0) < 1e-9

    # 3. existence is a tail exponent and is ENFORCED: k = 2 diverges, and null-space mass
    #    raises the bar from 3 to 5 -- the plug-in number exists in both cases, the moment does not
    with pytest.raises(ValueError, match="diverges"):
        exact_ratio_moment(np.eye(2), np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match="null space"):
        exact_ratio_moment(np.eye(5), np.diag([1.0] * 4 + [0.0]), np.eye(5))

    # 4. Result 51 (m): at Om = I the plug-in crossover is EXACT. Build the delayed-network
    #    Sigma on C_6, close A_u from its closed form, bisect the plug-in ratio to its root in
    #    phi, and the exact ratio must equal 1 there -- no Jensen shift.
    m, p_t, k_folds, lag, d_max = 6, 5, 2, 1, 2
    gammas = np.array([1.0, 0.7, 0.4])
    shells = [np.asarray(s) for s in cycle_shells(m, d_max)]
    t_gap = np.arange(p_t)[:, None] - np.arange(p_t)[None, :]

    def sigma_of(phi: float) -> np.ndarray:
        return sum(
            gammas[d]
            * gammas[e]
            * np.kron(shells[d] @ shells[e], phi ** np.abs(t_gap - lag * (d - e)))
            for d in range(d_max + 1)
            for e in range(d_max + 1)
        )

    r = k_folds / (k_folds - 1)
    operators = {}
    for name, fold in (("parity", np.arange(m) % 2), ("block", (np.arange(m) >= m // 2))):
        within = (fold[:, None] == fold[None, :]) * (k_folds / m)
        a_u = np.eye(m) - r**2 * np.full((m, m), 1.0 / m) + (r**2 - 1) * within
        operators[name] = np.kron(a_u, np.eye(p_t))

    def plug_ratio(phi: float) -> float:
        sig = sigma_of(phi)
        v = {n: np.trace(a @ sig @ a) / np.trace(a) ** 2 for n, a in operators.items()}
        return v["parity"] / v["block"]

    lo, hi = 0.1, 0.8
    assert (plug_ratio(lo) - 1.0) * (plug_ratio(hi) - 1.0) < 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if (plug_ratio(lo) - 1.0) * (plug_ratio(mid) - 1.0) <= 0.0:
            hi = mid
        else:
            lo = mid
    phi_star = 0.5 * (lo + hi)
    assert abs(phi_star**lag - 0.3954669988481472) < 1e-3  # the (h) closed form, same gammas

    sig = sigma_of(phi_star)
    omega = np.eye(m * p_t)
    exact = {n: exact_ratio_moment(a @ sig @ a, a, omega) for n, a in operators.items()}
    assert abs(exact["parity"] / exact["block"] - 1.0) < 1e-7


def test_delayed_network_law_and_the_limit_of_its_design_rule() -> None:
    # Result 51 (proofs/delayed_network_exposure.v): once a delay PROPAGATES THROUGH THE NETWORK,
    # Sigma stops being separable and Psi becomes a polynomial in phi^delta. Everything below is
    # measured against trajectories drawn from the generative definition, not from the formula.
    curve = delayed_network_certificate(n_draws=20_000)

    # 1. the polynomial predicts the simulated ratio, on both fold partitions
    for law, sim in (
        (curve.psi_law_parity, curve.psi_sim_parity),
        (curve.psi_law_block, curve.psi_sim_block),
    ):
        assert np.all(np.abs(sim - law) < 5.0 * curve.sim_stderr)

    # 2. the damage is a DESIGN question, not a sample-size one: alternating folds put every
    #    neighbour across the boundary and the delay moves Psi 4x further than for aligned folds
    assert curve.swing_parity > 4.0 * curve.swing_block
    assert curve.swing_parity > 3.0  # the misaligned arm loses a factor of several on Psi

    # 3. the theta* law holds in ITS OWN regime and is overturned outside it. With only
    #    nearest-neighbour spillover the sign is decided by which side of theta* the partition sits;
    #    the shell-1<->shell-2 term is larger than the D=1 term on the aligned arm and flips it
    #    back.
    assert curve.theta_parity < curve.theta_star < curve.theta_block
    assert curve.u1_nearest_parity < 0.0 < curve.u1_nearest_block
    assert abs(curve.u1_cross_block) > abs(curve.u1_nearest_block)
    assert curve.u_block[1] < 0.0  # so the aligned arm's u_1 is negative after all
    assert curve.theta_star > 0.5  # strictly above the 1/K a random balanced partition achieves

    # 4. delta = 0 must lose phi entirely -- STEP 7 of the gate file, reached from this side
    assert abs(curve.separable_spread) < 1e-12

    # 5. the exchangeable limit is Result 43's fold-geometry constant, unchanged
    assert abs(curve.c_fold_measured / curve.c_fold_predicted - 1.0) < 1e-12


def test_optimal_fold_partition_recovers_the_stripe_law_and_certifies() -> None:
    # Result 52, validation/fold_spectrum_law.mac, proofs/fold_spectrum_law.v.
    import itertools

    gammas = (1.0, 0.7, 0.4)

    def brute(m: int, shells: list[np.ndarray], phi: float) -> float:
        x = phi
        q = np.zeros((m, m))
        for d in range(3):
            for e in range(3):
                q += gammas[d] * gammas[e] * x ** abs(d - e) * (shells[d] @ shells[e])
        best = np.inf
        for cmb in itertools.combinations(range(1, m), m // 2 - 1):
            fold = np.ones(m, dtype=int)
            fold[0] = 0
            fold[list(cmb)] = 0
            same = fold[:, None] == fold[None, :]
            best = min(best, float(q[same].sum()))
        return best

    # 1. C_8 exhaustive: stripes below g1/g0 = 0.7, parity inside (0.7, 0.875), stripes above.
    shells8 = [np.asarray(sh, dtype=float) for sh in cycle_shells(8, 2)]
    for phi, expect_edges in ((0.5, 4), (0.8, 0), (0.95, 4)):
        design = optimal_fold_partition(shells8, gammas, phi, lag=1)
        assert design.exhaustive
        assert abs(design.objective - brute(8, shells8, phi)) < 1e-9
        assert design.lower_bound <= design.objective + 1e-9
        edges = sum(design.fold[i] == design.fold[(i + 1) % 8] for i in range(8))
        assert edges == expect_edges  # 0 = alternating, 4 = width-2 stripes

    # 2. local search still finds the C_12 global optimum. banded_exact=False is required as
    # well as the lowered limit: a cycle is circulant, so otherwise the banded dynamic program
    # takes the case and this stops testing the search at all.
    shells12 = [np.asarray(sh, dtype=float) for sh in cycle_shells(12, 2)]
    for phi in (0.3, 0.8):
        design = optimal_fold_partition(
            shells12, gammas, phi, lag=1, exhaustive_limit=10, banded_exact=False
        )
        assert not design.exhaustive
        assert design.route == "spectral-swap"
        assert abs(design.objective - brute(12, shells12, phi)) < 1e-9

    # 3. K = 3 on C_6: matches brute force over all balanced 3-colourings.
    shells6 = [np.asarray(sh, dtype=float) for sh in cycle_shells(6, 2)]
    x = 0.4
    q6 = np.zeros((6, 6))
    for d in range(3):
        for e in range(3):
            q6 += gammas[d] * gammas[e] * x ** abs(d - e) * (shells6[d] @ shells6[e])
    best3 = np.inf
    for assign in itertools.product(range(3), repeat=6):
        fold = np.asarray(assign)
        if np.bincount(fold, minlength=3).tolist() != [2, 2, 2]:
            continue
        same = fold[:, None] == fold[None, :]
        best3 = min(best3, float(q6[same].sum()))
    design3 = optimal_fold_partition(shells6, gammas, 0.4, lag=1, k_folds=3)
    assert abs(design3.objective - best3) < 1e-9
    assert design3.lower_bound <= design3.objective + 1e-9

    # 4. equal fold sizes are a precondition of the law, not a silent approximation.
    with pytest.raises(ValueError, match="divisible"):
        optimal_fold_partition(shells6, gammas, 0.4, lag=1, k_folds=4)
