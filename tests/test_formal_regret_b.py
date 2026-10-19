"""Property tests tying the regret certificates' cited Rocq lemmas and Maxima derivations to code.

Each test checks a cited statement on the float code by an independent route -- a brute-force
optimum, a closed-loop rollout, a quadrature, a Monte Carlo average with its standard error, or a
closed-form moment -- rather than by re-evaluating the formula the code itself uses.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import expit, ndtr

from chc import regret as R

FORMAL_LINKS = {
    "chc.regret.delayed_network_certificate": {
        "cites": (
            "delayed_exposure_gate.mac",
            "delayed_network_exposure.mac",
            "delayed_network_exposure.v",
        ),
        "tests": (
            "test_delayed_network_law_matches_the_generative_covariance",
            "test_delayed_network_simulation_matches_the_law",
            "test_delayed_network_u1_is_affine_in_theta_with_root_theta_star",
            "test_delayed_network_crossover_is_the_closed_form_root",
            "test_delayed_network_separable_limit_loses_phi",
            "test_delayed_network_c_fold_matches_the_k2_closed_form",
        ),
    },
    "chc.regret.doubly_robust_control_certificate": {
        "cites": ("doubly_robust.mac", "doubly_robust.v"),
        "tests": (
            "test_doubly_robust_bias_is_the_product_law",
            "test_doubly_robust_regret_vanishes_when_either_nuisance_is_correct",
        ),
    },
    "chc.regret.dynamic_causal_regret_certificate": {
        "cites": ("dynamic_causal_mpc.mac", "dynamic_causal_mpc.v"),
        "tests": ("test_dynamic_causal_regret_grows_at_the_offset_floor",),
    },
    "chc.regret.end_to_end_c2_certificate": {
        "cites": ("c2_end_to_end.mac", "c2_end_to_end.v"),
        "tests": (
            "test_c2_regret_is_the_true_plant_ce_gap",
            "test_c2_regret_obeys_the_three_channel_bound",
            "test_c2_sampling_regret_falls_like_one_over_g",
        ),
    },
    "chc.regret.ensemble_control_certificate": {
        "cites": ("ensemble_control.mac", "ensemble_control.v"),
        "tests": (
            "test_ensemble_floor_is_the_best_single_control_regret",
            "test_ensemble_two_context_floor_is_the_closed_form",
        ),
    },
    "chc.regret.exact_matrix_ratio_moment": {
        "cites": (
            "general_q_ratio_moment.mac",
            "general_q_ratio_moment.v",
            "matrix_ratio_moment.v",
        ),
        "tests": (
            "test_matrix_ratio_moment_meets_the_inverse_wishart_mean",
            "test_matrix_ratio_moment_meets_the_matrix_beta_law",
            "test_matrix_ratio_moment_matches_monte_carlo",
            "test_matrix_ratio_moment_refuses_below_the_existence_threshold",
            "test_matrix_gamma_at_two_is_finite_iff_q_at_most_four",
        ),
    },
    "chc.regret.exact_ratio_moment": {
        "cites": ("omega_jensen_gap.mac", "omega_jensen_gap.v"),
        "tests": (
            "test_ratio_moment_matches_inverse_chi_square_means",
            "test_ratio_moment_matches_monte_carlo",
            "test_ratio_moment_exceeds_the_plug_in_when_numerator_is_denominator",
            "test_ratio_moment_second_order_gap_is_two_traces",
            "test_ratio_moment_refuses_divergent_moments",
            "test_ratio_moment_is_scale_equivariant",
        ),
    },
    "chc.regret.exposure_map_certificate": {
        "cites": ("exposure_map_c2.mac", "exposure_map_c2.v"),
        "tests": (
            "test_exposure_map_regret_is_the_true_plant_ce_gap",
            "test_exposure_map_regret_obeys_the_four_term_bound",
        ),
    },
    "chc.regret.finite_horizon_pl_certificate": {
        "cites": ("nonlinear_regret.v",),
        "tests": ("test_finite_horizon_pl_bound_holds_and_is_tight_on_the_soft_mode",),
    },
    "chc.regret.highprob_regret_certificate": {
        "cites": ("highprob_regret.mac", "highprob_regret.v"),
        "tests": (
            "test_highprob_band_is_a_log_multiple_of_the_floor",
            "test_highprob_band_covers_the_exact_regret_where_c_holds",
            "test_highprob_band_undercovers_outside_the_local_regime",
        ),
    },
    "chc.regret.hinf_robust_regret_certificate": {
        "cites": ("hinf_robust_regret.mac", "hinf_robust_regret.v"),
        "tests": (
            "test_hinf_robust_value_is_the_adversarys_maximum",
            "test_hinf_robust_gain_minimises_the_robust_value",
            "test_hinf_calibrated_gain_matches_the_documented_values",
            "test_hinf_robust_gain_contracts_below_ce",
            "test_hinf_ce_regret_exceeds_the_calibrated_minimum",
        ),
    },
    "chc.regret.information_lower_bound_certificate": {
        "cites": (
            "action_van_trees.mac",
            "information_lower_bound.mac",
            "information_lower_bound.v",
        ),
        "tests": (
            "test_information_floor_constant_is_the_exact_action_curvature",
            "test_information_unbiased_arm_sits_on_the_exact_ols_risk",
            "test_information_van_trees_floor_lies_below_both_bayes_risks",
            "test_information_certificate_survives_the_knife_edge",
        ),
    },
    "chc.regret.interference_convexity_certificate": {
        "cites": ("interference_convexity.mac", "interference_convexity.v"),
        "tests": ("test_interference_convexity_aware_bound_is_exact_and_tighter",),
    },
    "chc.regret.interference_orthogonal_certificate": {
        "cites": ("interference_orthogonal.mac", "interference_regret.v", "orthogonal_control.v"),
        "tests": ("test_interference_orthogonal_slopes_match_the_population_bias_orders",),
    },
    "chc.regret.interference_regret_certificate": {
        "cites": ("interference_regret.v",),
        "tests": ("test_interference_regret_is_bounded_by_the_curvature_times_total_error",),
    },
    "chc.regret.lq_regret_sensitivity": {
        "cites": ("confounding_lq_regret.mac",),
        "tests": ("test_lq_regret_sensitivity_is_the_second_order_regret_coefficient",),
    },
    "chc.regret.minimax_action": {
        "cites": ("minimax_lq.mac", "minimax_lq.v"),
        "tests": (
            "test_minimax_action_cost_matches_brute_force",
            "test_minimax_action_regret_matches_brute_force",
            "test_minimax_action_beats_ce_on_the_rocq_instance",
            "test_minimax_action_branch_is_decided_by_the_kink_test",
            "test_minimax_action_kink_rule_on_a_negative_interval",
            "test_minimax_action_survives_a_tiny_midpoint",
        ),
    },
    "chc.regret.minimax_exploration_certificate": {
        "cites": ("minimax_exploration.mac", "minimax_exploration.v"),
        "tests": (
            "test_minimax_exploration_floor_is_the_optimal_budget_value",
            "test_minimax_exploration_no_schedule_beats_the_floor",
            "test_minimax_exploration_burst_attains_and_taper_pays_sqrt_two",
        ),
    },
    "chc.regret._sandwich_plan": {
        "cites": ("general_q_ratio_moment.mac",),
        "tests": (
            "test_sandwich_plan_reproduces_isserlis_at_q2",
            "test_sandwich_plan_matches_monte_carlo_at_q3",
            "test_sandwich_plan_weights_count_the_pairings",
        ),
    },
    "chc.regret.minimax_lq_policy": {
        "cites": ("minimax_lq.mac",),
        "tests": (
            "test_minimax_lq_value_is_the_constant_effect_worst_case",
            "test_minimax_lq_value_is_attained_at_b_lo_on_a_negative_interval",
            "test_minimax_lq_gains_beat_perturbed_gains",
        ),
    },
    "chc.regret.minimax_regret_move": {
        "cites": ("minimax_regret_move.mac",),
        "tests": (
            "test_minimax_regret_move_worst_regret_matches_quadrature",
            "test_minimax_regret_move_documented_regrets",
            "test_minimax_regret_move_no_linear_rule_does_better",
            "test_minimax_regret_move_full_move_gamma_is_the_share_boundary",
        ),
    },
}

FAST = settings(max_examples=25, deadline=None)
SLOW = settings(max_examples=8, deadline=None)


# ---------------------------------------------------------------- shared brute-force LQ machinery


def _riccati_gain(a, b, q, r):
    p = q.copy()
    for _ in range(20_000):
        k = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
        nxt = q + a.T @ p @ (a - b @ k)
        if np.max(np.abs(nxt - p)) <= 1e-14 * np.max(np.abs(nxt)):
            return np.linalg.solve(r + b.T @ nxt @ b, b.T @ nxt @ a)
        p = nxt
    raise AssertionError("value iteration did not converge")


def _rollout_cost(a, b, k, q, r, x0, steps=4000):
    x, cost = x0.astype(float), 0.0
    for _ in range(steps):
        u = -k @ x
        cost += float(x @ q @ x + u @ r @ u)
        x = a @ x + b @ u
    return cost


def _ce_gap_by_rollout(a, b, q, r, a_hat, b_hat, x0):
    k_hat, k_star = _riccati_gain(a_hat, b_hat, q, r), _riccati_gain(a, b, q, r)
    return _rollout_cost(a, b, k_hat, q, r, x0) - _rollout_cost(a, b, k_star, q, r, x0)


def _hessian(f, dim, h=1e-3):
    eye = np.eye(dim)
    out = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            out[i, j] = (
                f(h * eye[i] + h * eye[j])
                - f(h * eye[i] - h * eye[j])
                - f(-h * eye[i] + h * eye[j])
                + f(-h * eye[i] - h * eye[j])
            ) / (4 * h * h)
    return 0.5 * (out + out.T)


C2_PLANT = (
    np.array([[1.0, 0.1], [0.0, 0.95]]),
    np.array([[0.5], [1.0]]),
    np.eye(2),
    np.array([[0.5]]),
    np.array([1.0, 0.5]),
)
EXPOSURE_PLANT = (
    np.array([[1.0, 0.1, 0.0], [0.0, 0.95, 0.1], [0.0, 0.0, 0.9]]),
    np.array([[0.5], [1.0], [0.7]]),
    np.eye(3),
    np.array([[0.5]]),
    np.array([1.0, 0.5, 0.3]),
)


def _input_curvature(plant):
    a, b, q, r, x0 = plant
    n = b.shape[0]

    def gap(e):
        return R.certainty_equivalence_gap(a, b, q, r, a, b + e.reshape(n, 1), x0)

    return float(np.linalg.eigvalsh(_hessian(gap, n))[-1]) / 2.0


# ------------------------------------------------------------------- delayed network (Result 51)


def _cycle_distance(m):
    i = np.arange(m)
    gap = np.abs(i[:, None] - i[None, :])
    return np.minimum(gap, m - gap)


def _cross_fit_sandwich(fold):
    same = (fold[:, None] == fold[None, :]).astype(float)
    opposite = (1.0 - same) / (1.0 - same).sum(axis=1, keepdims=True)
    resid = np.eye(fold.size) - opposite
    return resid.T @ resid


def _generative_psi(m, p, lag, gammas, phi, fold):
    """Psi = N^2 tr(A Sig A) / (tr(A)^2 tr(Sig)), Sig built from x = sum_d g_d S_d z(t - lag d)."""
    dist = _cycle_distance(m)
    dmax = len(gammas) - 1
    span = p + lag * dmax
    base = span - p
    load = np.zeros((m * p, m * span))
    for i in range(m):
        for t in range(p):
            for j in range(m):
                d = int(dist[i, j])
                if d <= dmax:
                    load[i * p + t, j * span + base + t - lag * d] += gammas[d]
    times = np.arange(span)
    toeplitz = float(phi) ** np.abs(times[:, None] - times[None, :])
    sigma = load @ np.kron(np.eye(m), toeplitz) @ load.T
    big = np.kron(_cross_fit_sandwich(fold), np.eye(p))
    n = m * p
    return n**2 * np.trace(big @ sigma @ big) / (np.trace(big) ** 2 * np.trace(sigma))


def _folds(m):
    return {"parity": np.arange(m) % 2, "block": (np.arange(m) >= m // 2).astype(int)}


@SLOW
@given(
    m=st.sampled_from([4, 6, 8]),
    lag=st.integers(0, 2),
    gammas=st.lists(st.floats(0.1, 1.0), min_size=2, max_size=3),
    phis=st.lists(st.floats(0.0, 0.95), min_size=2, max_size=3),
)
def test_delayed_network_law_matches_the_generative_covariance(m, lag, gammas, phis):
    """delayed_network_exposure.mac: Psi is the polynomial in phi^delta of the generative Sigma."""
    curve = R.delayed_network_certificate(
        cluster_size=m, n_times=3, lag=lag, gammas=gammas, phi_grid=phis, n_draws=50
    )
    for name, law in (("parity", curve.psi_law_parity), ("block", curve.psi_law_block)):
        brute = [_generative_psi(m, 3, lag, gammas, phi, _folds(m)[name]) for phi in phis]
        assert law == pytest.approx(brute, rel=1e-10, abs=1e-12)  # Psi is O(1)


def test_delayed_network_simulation_matches_the_law():
    """delayed_network_exposure.mac: trajectories from the generative model reproduce the law."""
    curve = R.delayed_network_certificate(n_draws=6000)
    for law, sim in (
        (curve.psi_law_parity, curve.psi_sim_parity),
        (curve.psi_law_block, curve.psi_sim_block),
    ):
        assert np.all(np.abs(sim - law) <= 5.0 * curve.sim_stderr)


@pytest.mark.parametrize("m", [6, 8, 10])
def test_delayed_network_u1_is_affine_in_theta_with_root_theta_star(m):
    """delayed_network_exposure.v (u1_root_at_theta_star): nearest-shell u_1 is 0 at theta*."""
    adjacency = (_cycle_distance(m) == 1).astype(float)
    thetas, u1 = [], []
    for members in itertools.combinations(range(m), m // 2):
        fold = np.zeros(m, dtype=int)
        fold[list(members)] = 1
        same = (fold[:, None] == fold[None, :]).astype(float)
        au = _cross_fit_sandwich(fold)
        thetas.append(float((same * adjacency).sum() / adjacency.sum()))
        u1.append(2.0 * float(np.trace(au @ adjacency @ au)))
    slope, intercept = np.polyfit(thetas, u1, 1)
    assert np.max(np.abs(np.polyval([slope, intercept], thetas) - u1)) <= 1e-9 * max(map(abs, u1))
    curve = R.delayed_network_certificate(cluster_size=m, n_draws=50)
    assert -intercept / slope == pytest.approx(curve.theta_star, rel=1e-9, abs=0.0)
    assert curve.theta_star == pytest.approx(8.0 / 15.0, rel=1e-15, abs=0.0)
    assert curve.u1_nearest_parity < 0.0 < curve.u1_nearest_block
    assert curve.theta_parity < curve.theta_star < curve.theta_block


def test_delayed_network_crossover_is_the_closed_form_root():
    """delayed_network_exposure.v (x_star_is_the_crossover): the swap is at (49 - sqrt 1101)/40."""
    curve = R.delayed_network_certificate(n_draws=50)
    x_star = (49.0 - math.sqrt(1101.0)) / 40.0
    assert curve.crossover == pytest.approx(x_star, rel=1e-12, abs=0.0)
    parity = _generative_psi(6, 5, 1, (1.0, 0.7, 0.4), x_star, _folds(6)["parity"])
    block = _generative_psi(6, 5, 1, (1.0, 0.7, 0.4), x_star, _folds(6)["block"])
    assert parity == pytest.approx(block, rel=1e-12, abs=0.0)
    assert curve.trace_gap <= 1e-12


def test_delayed_network_separable_limit_loses_phi():
    """delayed_network_exposure.v (separable_limit_kills_phi): at delta = 0 Psi does not see phi."""
    curve = R.delayed_network_certificate(lag=0, phi_grid=(0.0, 0.5, 0.9), n_draws=6000)
    assert np.ptp(curve.psi_law_parity) <= 1e-12 * curve.psi_law_parity[0]
    assert np.ptp(curve.psi_law_block) <= 1e-12 * curve.psi_law_block[0]
    assert np.all(np.abs(curve.psi_sim_parity - curve.psi_law_parity) <= 5.0 * curve.sim_stderr)
    assert np.all(np.abs(curve.psi_sim_block - curve.psi_law_block) <= 5.0 * curve.sim_stderr)


@pytest.mark.parametrize("m", [4, 6, 8, 12])
def test_delayed_network_c_fold_matches_the_k2_closed_form(m):
    """delayed_network_exposure.v (c_fold_at_K2): m tr(Au^2)/tr(Au)^2 = m(m+14)/(m+2)^2."""
    curve = R.delayed_network_certificate(cluster_size=m, n_draws=50)
    au = _cross_fit_sandwich(_folds(m)["parity"])
    measured = m * np.trace(au @ au) / np.trace(au) ** 2
    assert curve.c_fold_measured == pytest.approx(measured, rel=1e-13, abs=0.0)
    assert measured == pytest.approx(m * (m + 14) / (m + 2) ** 2, rel=1e-13, abs=0.0)


# --------------------------------------------------------------------------- doubly robust AIPW


def _aipw_difference_sd(delta, n_rows, rng):
    x = rng.standard_normal(n_rows)
    e = expit(0.8 * x)
    d = (rng.random(n_rows) < e).astype(float)
    mu0, mu1 = 0.5 * x, 0.5 * x + 1.0
    y = np.where(d > 0.5, mu1, mu0) + 0.3 * rng.standard_normal(n_rows)

    def psi(dmu, de):
        e_hat = e + de * (0.5 - e)
        return (mu1 + dmu - mu0) + d * (y - mu1 - dmu) / e_hat - (1 - d) * (y - mu0) / (1 - e_hat)

    return float(np.std(psi(delta, delta) - psi(0.0, 0.0), ddof=1))


def _aipw_exact_bias(dmu, de):
    def integrand(x):
        e = expit(0.8 * x)
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi) * (0.5 - e) / (e + de * (0.5 - e))

    return dmu * de * quad(integrand, -40, 40, epsabs=0.0, epsrel=1e-12, limit=200)[0]


def test_doubly_robust_bias_is_the_product_law():
    """doubly_robust.mac: the AIPW bias is dmu*de*E[(1/2 - e)/e_hat], so regret is its square."""
    n, n_seeds, errors = 100_000, 8, (0.4, 0.2, 0.1)
    curve = R.doubly_robust_control_certificate(n=n, n_seeds=n_seeds, errors=errors)
    rng = np.random.default_rng(11)
    for delta, regret in zip(errors, curve.dr_regret_both, strict=True):
        sd = _aipw_difference_sd(delta, 200_000, rng) / math.sqrt(n * n_seeds)
        assert abs(math.sqrt(regret) - abs(_aipw_exact_bias(delta, delta))) <= 5.0 * sd
        assert regret <= delta**2


def test_doubly_robust_regret_vanishes_when_either_nuisance_is_correct():
    """doubly_robust.v (dr_zero_if_*): either correct nuisance zeroes the AIPW bias."""
    n, n_seeds = 100_000, 8
    curve = R.doubly_robust_control_certificate(n=n, n_seeds=n_seeds, errors=(0.4, 0.2))
    sd = _aipw_difference_sd(0.4, 200_000, np.random.default_rng(12)) / math.sqrt(n * n_seeds)
    assert curve.dr_outcome_ok <= (5.0 * sd) ** 2
    assert curve.dr_propensity_ok <= (5.0 * sd) ** 2
    assert curve.outcome_reg_fails == pytest.approx(0.4**2, rel=1e-12, abs=0.0)
    assert curve.ipw_fails >= 100.0 * (5.0 * sd) ** 2


# ----------------------------------------------------------------------- dynamic causal tracking


@FAST
@given(
    a=st.floats(0.0, 0.9),
    b=st.floats(0.3, 3.0),
    beta_share=st.floats(-0.6, 2.0).filter(lambda v: abs(v) > 0.05),
    x_ref=st.floats(0.2, 3.0),
    x0=st.floats(-2.0, 2.0),
    q=st.floats(0.1, 5.0),
)
def test_dynamic_causal_regret_grows_at_the_offset_floor(a, b, beta_share, x_ref, x0, q):
    """dynamic_causal_mpc.v: confounded tracking pays q*(x_ref*beta/b_obs)^2 per step."""
    beta = beta_share * b
    horizons = (10, 40, 160, 320, 640)
    curve = R.dynamic_causal_regret_certificate(
        a=a, b=b, beta=beta, x_ref=x_ref, x0=x0, q=q, horizons=horizons
    )

    def closed_form(b_hat, horizon):
        x_ss = b * x_ref / b_hat
        t = np.arange(horizon)
        return float(np.sum(q * (x_ss - x_ref + a**t * (x0 - x_ss)) ** 2))

    b_obs = b + beta
    expected = [closed_form(b_obs, t) - closed_form(b, t) for t in horizons]
    scale = q * (abs(x0) + 2 * x_ref * (1 + abs(b / b_obs))) ** 2 * horizons[-1]
    assert curve.predictive_regret == pytest.approx(expected, rel=0.0, abs=1e-12 * scale)
    floor = q * (x_ref * beta / b_obs) ** 2
    assert floor > 0.0
    # a <= 0.9 leaves a^320 < 3e-15 of transient past the fourth horizon
    increment = (curve.predictive_regret[-1] - curve.predictive_regret[-2]) / 320.0
    assert increment == pytest.approx(floor, rel=1e-9, abs=1e-12 * scale / 320.0)
    assert curve.causal_cost[-1] - curve.causal_cost[-2] <= 1e-12 * scale


# --------------------------------------------------------------------- C2 end-to-end and exposure


def test_c2_regret_is_the_true_plant_ce_gap():
    """c2_end_to_end.mac: the regret is the CE controller's true-plant cost gap (MTR quantity)."""
    a, b, q, r, x0 = C2_PLANT
    curve = R.end_to_end_c2_certificate(n_seeds=2, g_grid=(10, 20))
    e_d, e_s = np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]])
    for d, half, full in zip(curve.deltas, curve.half_regret, curve.full_regret, strict=True):
        assert half == pytest.approx(
            _ce_gap_by_rollout(a, b, q, r, a, b + d**2 * e_d + d * e_s, x0), rel=1e-7, abs=0.0
        )
        assert full == pytest.approx(
            _ce_gap_by_rollout(a, b, q, r, a, b + d**2 * (e_d + e_s), x0), rel=1e-6, abs=0.0
        )


@SLOW
@given(deltas=st.lists(st.floats(0.003, 0.2), min_size=3, max_size=5, unique=True))
def test_c2_regret_obeys_the_three_channel_bound(deltas):
    """c2_end_to_end.v (three_channel_bound): R <= cc (e_d + e_s)^2 <= 3 cc (e_d^2 + e_s^2)."""
    deltas = sorted(deltas)
    curve = R.end_to_end_c2_certificate(n_seeds=2, g_grid=(10, 20), deltas=deltas)
    cc = _input_curvature(C2_PLANT)
    d = np.asarray(deltas)
    for regret, e_d, e_s in ((curve.half_regret, d**2, d), (curve.full_regret, d**2, d**2)):
        assert np.all(regret >= 0.0)
        assert np.all(regret <= cc * (e_d + e_s) ** 2)
        assert np.all(cc * (e_d + e_s) ** 2 <= 3.0 * cc * (e_d**2 + e_s**2) * (1 + 1e-15))


def test_c2_sampling_regret_falls_like_one_over_g():
    """c2_end_to_end.v (end_to_end_full): with delta ~ 0 the regret is the 1/G sampling floor."""
    curve = R.end_to_end_c2_certificate(g_grid=(10, 20, 40, 80, 160, 320), n_seeds=60)
    scaled = curve.regret_vs_g * curve.g_grid
    assert np.max(scaled) <= 2.5 * np.min(scaled)
    assert -1.4 <= curve.g_slope <= -0.6


def test_exposure_map_regret_is_the_true_plant_ce_gap():
    """exposure_map_c2.mac: the three-channel regret is the CE controller's true-plant cost gap."""
    a, b, q, r, x0 = EXPOSURE_PLANT
    curve = R.exposure_map_certificate(n_delta=4)
    e_d, e_s, e_w = np.eye(3)[:, [0]], np.eye(3)[:, [1]], np.eye(3)[:, [2]]
    for d, full, wbn in zip(curve.deltas, curve.full_regret, curve.wbottleneck_regret, strict=True):
        assert full == pytest.approx(
            _ce_gap_by_rollout(a, b, q, r, a, b + d**2 * (e_d + e_s + e_w), x0), rel=1e-6, abs=0.0
        )
        assert wbn == pytest.approx(
            _ce_gap_by_rollout(a, b, q, r, a, b + d**2 * (e_d + e_s) + d * e_w, x0),
            rel=1e-7,
            abs=0.0,
        )


@SLOW
@given(lo=st.floats(0.002, 0.05), hi=st.floats(0.06, 0.2), n=st.integers(3, 8))
def test_exposure_map_regret_obeys_the_four_term_bound(lo, hi, n):
    """exposure_map_c2.v (exposure_full_orth, sum4_sq_bound): R <= cc (sum r)^2 <= 4 cc sum r^2."""
    curve = R.exposure_map_certificate(delta_lo=lo, delta_hi=hi, n_delta=n)
    cc = _input_curvature(EXPOSURE_PLANT)
    d = curve.deltas
    full_sum, wbn_sum = 3 * d**2, 2 * d**2 + d
    assert np.all(curve.full_regret <= cc * full_sum**2)
    assert np.all(curve.full_regret <= 4 * cc * 3 * d**4 * (1 + 1e-15))
    assert np.all(curve.wbottleneck_regret <= cc * wbn_sum**2)
    assert np.all(curve.wbottleneck_regret <= 4 * cc * (2 * d**4 + d**2) * (1 + 1e-15))
    assert curve.full_slope == pytest.approx(4.0, abs=0.15)
    assert curve.wbottleneck_slope == pytest.approx(2.0, abs=0.15)


# ------------------------------------------------------------------------------ ensemble control


def _lq_cost(b, u, xt, rr):
    return (b * u - xt) ** 2 + rr * u**2


@FAST
@given(
    b0=st.floats(0.3, 2.0),
    rr=st.floats(0.1, 2.0),
    xt=st.floats(0.2, 3.0),
    n_ctx=st.integers(2, 25),
    hi=st.floats(0.05, 0.25),
)
def test_ensemble_floor_is_the_best_single_control_regret(b0, rr, xt, n_ctx, hi):
    """ensemble_control.v (ensemble_optimum): one control's least mean regret is the floor."""
    curve = R.ensemble_control_certificate(
        b0=b0, rr=rr, xt=xt, n_ctx=n_ctx, spread_lo=0.02, spread_hi=hi, n_spread=3
    )
    for s, floor, naive in zip(
        curve.spreads, curve.ensemble_floor, curve.naive_mean_regret, strict=True
    ):
        bs = np.linspace(b0 - s, b0 + s, n_ctx)
        best = np.array(
            [minimize_scalar(lambda u, bb=bb: _lq_cost(bb, u, xt, rr)).fun for bb in bs]
        )

        def mean_regret(u, bs=bs, best=best):
            return float(np.mean(_lq_cost(bs, u, xt, rr) - best))

        brute = minimize_scalar(mean_regret, bounds=(-10 * xt, 10 * xt), method="bounded")
        brute = min(brute.fun, *(mean_regret(u) for u in np.linspace(-xt, xt, 401)))
        tol = 1e-9 * xt**2
        assert floor <= brute + tol
        assert brute <= floor + tol
        assert naive >= floor - tol


@FAST
@given(b0=st.floats(0.3, 2.0), rr=st.floats(0.1, 2.0), xt=st.floats(0.2, 3.0))
def test_ensemble_two_context_floor_is_the_closed_form(b0, rr, xt):
    """ensemble_control.v (heterogeneity_floor): R(u_ens) = w1 w2 (u1 - u2)^2/(w1 + w2)."""
    curve = R.ensemble_control_certificate(
        b0=b0, rr=rr, xt=xt, n_ctx=2, spread_lo=1e-4, spread_hi=0.25, n_spread=4
    )
    for s, floor in zip(curve.spreads, curve.ensemble_floor, strict=True):
        b1, b2 = b0 - s, b0 + s
        w1, w2 = (b1**2 + rr) / 2, (b2**2 + rr) / 2
        u1, u2 = b1 * xt / (b1**2 + rr), b2 * xt / (b2**2 + rr)
        assert floor == pytest.approx(w1 * w2 * (u1 - u2) ** 2 / (w1 + w2), rel=1e-9, abs=0.0)
    assert curve.homogeneous_floor <= 1e-7 * xt**2


# ----------------------------------------------------------------------- exact ratio moments


def _random_pd(rng, n, cond=5.0):
    q_mat, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return q_mat @ np.diag(np.geomspace(1.0, cond, n)) @ q_mat.T


@FAST
@given(k=st.integers(3, 12), s=st.floats(0.3, 3.0))
def test_ratio_moment_matches_inverse_chi_square_means(k, s):
    """omega_jensen_gap.mac: E[1/R'R] = 1/(s^2 (k-2)), E[R_1^2/(R'R)^2] = 1/(s^2 k (k-2))."""
    om = s * s * np.eye(k)
    assert R.exact_ratio_moment(np.eye(k), np.eye(k), om) == pytest.approx(
        1.0 / (s * s * (k - 2)), rel=1e-7, abs=0.0
    )
    first = np.zeros((k, k))
    first[0, 0] = 1.0
    assert R.exact_ratio_moment(first, np.eye(k), om) == pytest.approx(
        1.0 / (s * s * k * (k - 2)), rel=1e-7, abs=0.0
    )


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_ratio_moment_matches_monte_carlo(seed):
    """omega_jensen_gap.mac: the resolvent quadrature is E[X/Y^2] for R ~ N(0, Om)."""
    rng = np.random.default_rng(seed)
    n = 8
    om, c = _random_pd(rng, n), _random_pd(rng, n, cond=3.0)
    half = rng.standard_normal((n, n))
    b = half @ half.T
    draws = rng.multivariate_normal(np.zeros(n), om, size=400_000)
    ratio = (
        np.einsum("ni,ij,nj->n", draws, b, draws) / np.einsum("ni,ij,nj->n", draws, c, draws) ** 2
    )
    sd = float(np.std(ratio, ddof=1)) / math.sqrt(ratio.size)
    assert abs(R.exact_ratio_moment(b, c, om) - float(ratio.mean())) <= 5.0 * sd


@FAST
@given(seed=st.integers(0, 10_000), n=st.integers(3, 9))
def test_ratio_moment_exceeds_the_plug_in_when_numerator_is_denominator(seed, n):
    """omega_jensen_gap.v (sigma_eq_identity_gap_positive): E[1/Y] >= 1/E[Y] at B = C."""
    rng = np.random.default_rng(seed)
    om, c = _random_pd(rng, n), _random_pd(rng, n, cond=4.0)
    assert R.exact_ratio_moment(c, c, om) > 1.0 / np.trace(c @ om)


def test_ratio_moment_second_order_gap_is_two_traces():
    """omega_jensen_gap.v (second_order_gap_is_two_traces): the relative gap's leading term."""
    rng = np.random.default_rng(5)
    n = 6
    om, c = _random_pd(rng, n, cond=2.0), _random_pd(rng, n, cond=2.0)
    half = rng.standard_normal((n, n))
    b = half @ half.T
    residuals = []
    for copies in (2, 4, 8):
        eye = np.eye(copies)
        bb, cb, ob = np.kron(eye, b), np.kron(eye, c), np.kron(eye, om)
        t_bo, t_co = np.trace(bb @ ob), np.trace(cb @ ob)
        predicted = -4 * np.trace(bb @ ob @ cb @ ob) / (t_bo * t_co) + 6 * np.trace(
            cb @ ob @ cb @ ob
        ) / (t_co**2)
        gap = R.exact_ratio_moment(bb, cb, ob) / (t_bo / t_co**2) - 1.0
        residuals.append(abs(gap - predicted) / abs(predicted))
    assert residuals[0] < 0.3
    assert residuals[1] < 0.7 * residuals[0]
    assert residuals[2] < 0.7 * residuals[1]


def test_ratio_moment_refuses_divergent_moments():
    """omega_jensen_gap.mac: E[X/Y^2] needs k >= 3 live eigenvalues, k >= 5 with null-space mass."""
    rank_two = np.diag([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="diverges"):
        R.exact_ratio_moment(rank_two, rank_two, np.eye(6))
    rank_four = np.diag([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="diverges"):
        R.exact_ratio_moment(np.eye(6), rank_four, np.eye(6))
    assert np.isfinite(R.exact_ratio_moment(rank_four, rank_four, np.eye(6)))


@pytest.mark.xfail(
    strict=True,
    reason="B = C = I_7, Om = 1e4 I_7: returns -3.3e-10 against 1/(1e4 * 5) = 2e-5; quad on "
    "[0, inf) misses the peak at t ~ 1/(2 s^2) and its default epsabs 1.5e-8 accepts that",
)
def test_ratio_moment_is_scale_equivariant():
    """omega_jensen_gap.mac: E[X/Y^2] at Om = s^2 I is 1/(s^2 (k-2)) at every scale s."""
    value = R.exact_ratio_moment(np.eye(7), np.eye(7), 1e4 * np.eye(7))
    assert value == pytest.approx(1.0 / (1e4 * 5), rel=1e-7, abs=0.0)


# ---------------------------------------------------------------- the matrix (q-channel) moment


@pytest.mark.parametrize("rho", [0.0, 0.5])
def test_matrix_ratio_moment_meets_the_inverse_wishart_mean(rho):
    """general_q_ratio_moment.mac: E[(X'X)^-1] = R^-1/(n - q - 1) for X'X ~ Wishart_2(n, R)."""
    n = 8
    corr = np.array([[1.0, rho], [rho, 1.0]])
    out = R.exact_matrix_ratio_moment(np.eye(n), np.eye(n), np.kron(corr, np.eye(n)), nodes=20)
    exact = np.linalg.inv(corr) / (n - 3)
    assert np.max(np.abs(out - exact)) <= 1e-5 * np.max(np.abs(exact))


def test_matrix_ratio_moment_meets_the_matrix_beta_law():
    """general_q_ratio_moment.mac: E[M^-1 X'PX M^-1] = (r/n) I/(n-q-1), P a rank-r projector."""
    n, rank = 8, 3
    basis, _ = np.linalg.qr(np.random.default_rng(4).standard_normal((n, rank)))
    out = R.exact_matrix_ratio_moment(basis @ basis.T, np.eye(n), np.eye(2 * n), nodes=20)
    exact = (rank / n) * np.eye(2) / (n - 3)
    assert np.max(np.abs(out - exact)) <= 1e-5 * np.max(np.abs(exact))


def test_matrix_ratio_moment_matches_monte_carlo():
    """general_q_ratio_moment.mac: the cone quadrature is E[(X'CX)^-1 X'BX (X'CX)^-1]."""
    rng = np.random.default_rng(8)
    n = 9
    om, c = _random_pd(rng, 2 * n, cond=3.0), _random_pd(rng, n, cond=3.0)
    half = rng.standard_normal((n, n))
    b = half @ half.T / n
    out = R.exact_matrix_ratio_moment(b, c, om, nodes=20)
    z = rng.multivariate_normal(np.zeros(2 * n), om, size=200_000)
    x = np.stack([z[:, :n], z[:, n:]], axis=2)
    m_inv = np.linalg.inv(np.einsum("nia,ij,njb->nab", x, c, x))
    sample = m_inv @ np.einsum("nia,ij,njb->nab", x, b, x) @ m_inv
    sd = np.std(sample, axis=0, ddof=1) / math.sqrt(sample.shape[0])
    assert np.all(np.abs(out - sample.mean(axis=0)) <= 5.0 * sd)


def test_matrix_ratio_moment_refuses_below_the_existence_threshold():
    """general_q_ratio_moment.v: the inverse-Wishart moment needs n >= q + 2 per channel."""
    with pytest.raises(ValueError, match="diverges"):
        R.exact_matrix_ratio_moment(np.eye(3), np.eye(3), np.eye(6))


def test_matrix_gamma_at_two_is_finite_iff_q_at_most_four():
    """general_q_ratio_moment.v (ingham_siegel_valid_iff_q_le_four): Gamma_q(2) < inf iff q <= 4."""
    for q in (1, 2, 3, 4):
        expected = math.pi ** (q * (q - 1) / 4) * math.prod(math.gamma(2 - j / 2) for j in range(q))
        assert R._matrix_gamma(q, 2.0) == pytest.approx(expected, rel=1e-13, abs=0.0)
    assert R._matrix_gamma(5, 2.0) == math.inf


# ------------------------------------------------------------------- the symmetric-group plan


def _evaluate_plan(q, b, c, om):
    plan = R._sandwich_plan(q)
    n = c.shape[0]
    kernels = []
    for side, u, v in plan["forms"]:
        sel = np.zeros((q, q))
        sel[u, v] += 0.5
        sel[v, u] += 0.5
        kernels.append(np.kron(sel, c if side == 0 else b) @ om)
    traces = []
    for word in plan["words"]:
        prod = np.eye(q * n)
        for f in word:
            prod = prod @ kernels[f]
        traces.append(np.trace(prod))
    traces.append(1.0)
    traces = np.asarray(traces)
    terms = plan["weight"] * np.prod(traces[plan["idx"]], axis=1)
    out = np.zeros((q, q))
    for (i, j), value in zip(plan["entries"], np.bincount(plan["seg"], weights=terms), strict=True):
        out[i, j] = out[j, i] = value
    return out


def _isserlis3(a, b, c):
    ta, tb, tc = np.trace(a), np.trace(b), np.trace(c)
    return (
        ta * tb * tc
        + 2 * (ta * np.trace(b @ c) + tb * np.trace(a @ c) + tc * np.trace(a @ b))
        + 8 * np.trace(a @ b @ c)
    )


@FAST
@given(seed=st.integers(0, 10_000), n=st.integers(1, 3))
def test_sandwich_plan_reproduces_isserlis_at_q2(seed, n):
    """general_q_ratio_moment.mac STEP 1: at q = 2 the S_3 sum is Isserlis' 15 pairings."""
    rng = np.random.default_rng(seed)
    om = _random_pd(rng, 2 * n)
    hb, hc = rng.standard_normal((n, n)), rng.standard_normal((n, n))
    b, c = hb + hb.T, hc + hc.T

    def form(mat, u, v):
        sel = np.zeros((2, 2))
        sel[u, v] += 0.5
        sel[v, u] += 0.5
        return np.kron(sel, mat) @ om

    adj = {(0, 0): (1, (1, 1)), (0, 1): (-1, (0, 1)), (1, 0): (-1, (1, 0)), (1, 1): (1, (0, 0))}
    exact = np.zeros((2, 2))
    for i, j, k, ell in itertools.product(range(2), repeat=4):
        s1, f1 = adj[(i, k)]
        s2, f2 = adj[(ell, j)]
        exact[i, j] += s1 * s2 * _isserlis3(form(c, *f1), form(b, k, ell), form(c, *f2))
    out = _evaluate_plan(2, b, c, om)
    scale = (np.linalg.norm(c) * np.linalg.norm(om)) ** 2 * np.linalg.norm(b) * np.linalg.norm(om)
    assert out == pytest.approx(exact, rel=1e-10, abs=1e-12 * scale)


def test_sandwich_plan_matches_monte_carlo_at_q3():
    """general_q_ratio_moment.mac STEP 1: at q = 3 the S_5 sum is E[adj(M) N adj(M)]."""
    rng = np.random.default_rng(21)
    n, q = 3, 3  # n < q makes M singular, and then adj(M) N adj(M) = 0 identically
    om = _random_pd(rng, q * n, cond=2.0)
    b, c = _random_pd(rng, n, cond=2.0), _random_pd(rng, n, cond=2.0)
    out = _evaluate_plan(q, b, c, om)
    z = rng.multivariate_normal(np.zeros(q * n), om, size=400_000)
    x = z.reshape(-1, q, n).transpose(0, 2, 1)
    m = np.einsum("nia,ij,njb->nab", x, c, x)
    adj = np.empty_like(m)
    for i, j in itertools.product(range(q), repeat=2):
        minor = np.delete(np.delete(m, j, axis=1), i, axis=2)
        adj[:, i, j] = (-1) ** (i + j) * np.linalg.det(minor)
    sample = adj @ np.einsum("nia,ij,njb->nab", x, b, x) @ adj
    sd = np.std(sample, axis=0, ddof=1) / math.sqrt(sample.shape[0])
    assert np.all(np.abs(out - sample.mean(axis=0)) <= 5.0 * sd)


@pytest.mark.parametrize("q", [2, 3])
def test_sandwich_plan_weights_count_the_pairings(q):
    """general_q_ratio_moment.mac STEP 1: sum over S_m of 2^(m - c) is (2m - 1)!! per product."""
    plan = R._sandwich_plan(q)
    m = 2 * q - 1
    pairings = math.prod(range(1, 2 * m, 2))
    per_product = math.factorial(m)
    assert plan["idx"].shape == (plan["weight"].size, m)
    assert plan["weight"].size % per_product == 0
    groups = plan["weight"].reshape(-1, per_product)
    assert np.all(np.abs(np.abs(groups.sum(axis=1)) - pairings) == 0.0)


# ------------------------------------------------------------------------------ finite-horizon PL


@FAST
@given(
    a=st.floats(0.2, 1.1),
    b=st.floats(0.2, 2.0),
    q=st.floats(0.1, 3.0),
    r=st.floats(0.05, 2.0),
    qf=st.floats(0.0, 10.0),
    x0=st.floats(-2.0, 2.0).filter(lambda v: abs(v) > 0.05),
    seed=st.integers(0, 100),
)
def test_finite_horizon_pl_bound_holds_and_is_tight_on_the_soft_mode(a, b, q, r, qf, x0, seed):
    """nonlinear_regret.v (pl_mode_bound): grad^2/(2 mu) >= regret, equality on the softest mode."""
    horizons = (2, 4, 7)
    curve = R.finite_horizon_pl_certificate(
        a=a, b=b, q=q, r=r, qf=qf, x0=x0, horizons=horizons, n_samples=40, seed=seed
    )
    for t, mu in zip(horizons, curve.mu_min, strict=True):

        def cost(u, t=t):
            x, total = x0, 0.0
            for k in range(t):
                x = a * x + b * u[k]
                total += (qf if k == t - 1 else q) * x * x + r * u[k] ** 2
            return total

        hess = _hessian(cost, t, h=0.5)
        assert mu == pytest.approx(np.linalg.eigvalsh(hess)[0], rel=1e-7, abs=0.0)
    assert np.all(curve.bound_slack >= 0.0)
    assert curve.worst_mode_ratio == pytest.approx(np.ones(len(horizons)), rel=1e-8, abs=0.0)


# --------------------------------------------------------------------------- high-probability band


@FAST
@given(
    b=st.floats(0.3, 2.0),
    rr=st.floats(0.1, 2.0),
    xt=st.floats(0.3, 2.0),
    sigma=st.floats(0.1, 2.0),
    n=st.integers(20, 2000),
    v_exp=st.floats(0.5, 4.0),
    shrink=st.floats(0.1, 1.0),
)
def test_highprob_band_is_a_log_multiple_of_the_floor(b, rr, xt, sigma, n, v_exp, shrink):
    """highprob_regret.v (band_is_log_multiple_of_floor, confounding_widens_band)."""
    if abs(rr - b * b) < 0.05:
        return  # the knife edge C = 0 makes band/floor 0/0
    deltas = (0.5, 0.1, 0.01)
    curve = R.highprob_regret_certificate(
        b=b, rr=rr, xt=xt, sigma=sigma, n=n, v_exp=v_exp, v_conf=shrink * v_exp,
        deltas=deltas, n_trials=1,
    )  # fmt: skip
    h = 1e-4 * b

    def regret(b_hat):
        u_hat = b_hat * xt / (b_hat**2 + rr)
        best = minimize_scalar(lambda u: _lq_cost(b, u, xt, rr)).fun
        return _lq_cost(b, u_hat, xt, rr) - best

    curvature = (regret(b + h) + regret(b - h)) / (2 * h * h)
    assert curve.cr_floor == pytest.approx(
        curvature * sigma**2 / (n * v_exp), rel=1e-4, abs=1e-6 * xt**2 * sigma**2 / (n * v_exp)
    )
    logs = 2.0 * np.log(2.0 / np.asarray(deltas))
    assert curve.band_over_floor == pytest.approx(logs, rel=1e-13, abs=0.0)
    assert np.all(curve.confounded_bands >= curve.highprob_bands)


def test_highprob_band_covers_the_exact_regret_where_c_holds():
    """highprob_regret.mac: P(regret > band) <= delta, read on the exact regret at the defaults."""
    n_trials = 3000
    curve = R.highprob_regret_certificate(n_trials=n_trials)
    sd = np.sqrt(curve.deltas * (1 - curve.deltas) / n_trials)
    assert np.all(curve.empirical_coverage >= 1.0 - curve.deltas - 4.0 * sd)


def test_highprob_band_undercovers_outside_the_local_regime():
    """highprob_regret.v covers the quadratic proxy only: off the local regime it undercovers."""
    curve = R.highprob_regret_certificate(b=0.6, sigma=1.0, n=30, deltas=(0.01,), n_trials=2000)
    sd = math.sqrt(0.01 * 0.99 / 2000)
    assert curve.empirical_coverage[0] == pytest.approx(0.916, abs=0.0005)
    assert curve.empirical_coverage[0] < 0.99 - 4.0 * sd


# ------------------------------------------------------------------------------------- H-infinity


@FAST
@given(
    b=st.floats(0.3, 2.0),
    rr=st.floats(0.1, 2.0),
    xt=st.floats(0.3, 2.0),
    s=st.floats(0.1, 1.5),
)
def test_hinf_robust_value_is_the_adversarys_maximum(b, rr, xt, s):
    """hinf_robust_regret.v (robust_value_is_worst_case, inflates, antitone_in_budget)."""
    u_ce = b * xt / (b * b + rr)
    curve = R.hinf_robust_regret_certificate(
        b=b, rr=rr, xt=xt, s=s, gamma_lo=1.05 * u_ce, gamma_hi=8.0 * max(u_ce, 1.0), n_gamma=6,
        n_u=50,
    )  # fmt: skip
    for g, value in zip(curve.gamma_grid, curve.inflation_at_uce, strict=True):

        def game(w, g=g):
            return -(((b + w) * u_ce - xt) ** 2 + rr * u_ce**2 - g * g * w * w)

        best = -minimize_scalar(game).fun
        assert value == pytest.approx(best, rel=1e-9, abs=1e-12 * xt**2)
        assert value >= curve.nominal_at_uce
    assert np.all(np.diff(curve.inflation_at_uce) <= 0.0)


@FAST
@given(
    b=st.floats(0.3, 2.0),
    rr=st.floats(0.1, 2.0),
    xt=st.floats(0.3, 2.0),
    s=st.floats(0.1, 1.5),
)
def test_hinf_robust_gain_minimises_the_robust_value(b, rr, xt, s):
    """hinf_robust_regret.mac (D): u_rob(gamma) = argmin of gamma^2 e^2/(gamma^2 - u^2) + rr u^2."""
    n_u = 4000
    curve = R.hinf_robust_regret_certificate(b=b, rr=rr, xt=xt, s=s, n_gamma=6, n_u=n_u)
    for g, gain in zip(curve.gamma_grid, curve.robust_gain, strict=True):

        def value(u, g=g):
            return g * g * (b * u - xt) ** 2 / (g * g - u * u) + rr * u * u

        best = minimize_scalar(
            value, bounds=(0.0, 0.999 * g), method="bounded", options={"xatol": 1e-12}
        )
        assert abs(gain - best.x) <= 0.999 * g / (n_u - 1)
        if rr >= b * b:
            assert gain <= curve.u_ce + 0.999 * g / (n_u - 1)
    assert np.all(curve.expected_regret >= -1e-15 * xt**2)


def test_hinf_calibrated_gain_matches_the_documented_values():
    """hinf_robust_regret.mac (E): at the defaults gamma* gives 0.5415 against u_pess = 0.538."""
    curve = R.hinf_robust_regret_certificate()
    assert curve.gain_at_gamma_star == pytest.approx(0.5415, abs=5e-5)
    assert curve.u_pess_star == pytest.approx(1.0 / 1.86, rel=1e-15, abs=0.0)
    assert curve.robust_gain[-1] == pytest.approx(curve.u_ce, abs=0.01)


@pytest.mark.xfail(
    strict=True,
    reason="b=1, rr=0.5, xt=1, gamma=2: u_rob = 0.67969 > u_ce = 0.66667; robustness raises the "
    "gain whenever b^2 > rr and gamma is large",
)
def test_hinf_robust_gain_contracts_below_ce():
    """hinf_robust_regret.mac (D), (E): the docstring's pessimistic contraction u_rob <= u_ce."""
    curve = R.hinf_robust_regret_certificate(gamma_lo=2.0, gamma_hi=2.0, n_gamma=1, n_u=200_001)
    assert curve.robust_gain[0] <= curve.u_ce


@pytest.mark.xfail(
    strict=True,
    reason="b=1.5, rr=0.5, xt=1, s=0.3: CE regret 8.486e-4 < calibrated minimum 8.620e-4; no "
    "gamma on the default grid takes the gain below u_ce",
)
def test_hinf_ce_regret_exceeds_the_calibrated_minimum():
    """hinf_robust_regret.mac (E): the CE control's regret exceeds the calibrated robust one."""
    curve = R.hinf_robust_regret_certificate(b=1.5, rr=0.5, xt=1.0, s=0.3)
    assert curve.ce_expected_regret > curve.expected_regret_min


# ---------------------------------------------------------- information lower bound / van Trees


def _optimal_action(effect, xt, rr):
    design = np.array([[effect], [math.sqrt(rr)]])
    return float(np.linalg.lstsq(design, np.array([-xt, 0.0]), rcond=None)[0][0])


@FAST
@given(
    b=st.floats(0.2, 2.0),
    rr=st.floats(0.1, 2.0),
    xt=st.floats(0.3, 2.0),
    sigma=st.floats(0.1, 2.0),
    alpha=st.floats(0.0, 2.0),
)
def test_information_floor_constant_is_the_exact_action_curvature(b, rr, xt, sigma, alpha):
    """action_van_trees.mac STEP 2: the floor's C is (b^2 + rr) psi'(b)^2; confounding raises it."""
    if abs(rr - b * b) < 1e-3:
        return  # the knife edge crashes; see the xfail below
    curve = R.information_lower_bound_certificate(
        b=b, rr=rr, xt=xt, sigma=sigma, alpha=alpha, sample_sizes=(50, 100), n_seeds=2
    )
    h = 1e-5 * max(b, 1.0)
    slope = (_optimal_action(b + h, xt, rr) - _optimal_action(b - h, xt, rr)) / (2 * h)
    v_exp = alpha**2 + 1.0
    expected = (b * b + rr) * slope**2 * sigma**2 / (curve.sample_sizes * v_exp)
    scale = xt**2 * sigma**2 / (curve.sample_sizes * v_exp * (b * b + rr))
    assert curve.cramer_rao_floor == pytest.approx(expected, rel=1e-6, abs=1e-8 * scale.max())
    assert curve.confounded_floor / curve.cramer_rao_floor == pytest.approx(
        np.full(2, v_exp), rel=1e-13, abs=0.0
    )
    assert curve.floor_ratio >= 1.0


@pytest.mark.xfail(
    strict=True,
    raises=ZeroDivisionError,
    reason="b = rr = 1: the floor C sigma^2/(n V) is 0 at the knife edge the docstring names, "
    "and hodges_pointwise / cr_floor[-1] raises ZeroDivisionError",
)
def test_information_certificate_survives_the_knife_edge():
    """action_van_trees.mac STEP 5: at rr = b^2 the floor vanishes and the certificate returns."""
    curve = R.information_lower_bound_certificate(b=1.0, rr=1.0, sample_sizes=(50, 100), n_seeds=2)
    assert np.all(curve.cramer_rao_floor == 0.0)


def test_information_unbiased_arm_sits_on_the_exact_ols_risk():
    """information_lower_bound.v: the unbiased OLS risk sigma^2/(V (n-2)) is at least the floor."""
    n_seeds, sigma, alpha = 400, 0.5, 1.0
    curve = R.information_lower_bound_certificate(n_seeds=n_seeds, sample_sizes=(50, 200, 1000))
    v = alpha**2 + 1.0
    n = curve.sample_sizes
    coeff = curve.cramer_rao_floor * n * v / sigma**2
    mean_sq = sigma**2 / (v * (n - 2))
    fourth = 3 * sigma**4 / (v**2 * (n - 2) * (n - 4))
    sd = coeff * np.sqrt((fourth - mean_sq**2) / n_seeds)
    assert np.all(coeff * mean_sq >= curve.cramer_rao_floor)
    assert np.all(np.abs(curve.experimental_regret - coeff * mean_sq) <= 4.0 * sd)


def test_information_van_trees_floor_lies_below_both_bayes_risks():
    """action_van_trees.mac STEP 3: (E psi')^2/(n I1 + I(lambda)) bounds every estimator's risk."""
    n_seeds, b, rr, xt, sigma = 300, 1.0, 0.5, 1.0, 0.5
    curve = R.information_lower_bound_certificate(n_seeds=n_seeds)

    def sensitivity(effect):
        return -xt * (rr - effect**2) / (rr + effect**2) ** 2

    def weighted(z):
        return sensitivity(b + 0.25 * z) * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

    mean = quad(weighted, -12, 12, epsabs=0.0, epsrel=1e-12)[0]
    second = quad(lambda z: weighted(z) * sensitivity(b + 0.25 * z), -12, 12, epsrel=1e-12)[0]
    draws = 40 * n_seeds
    mean_sd = math.sqrt(second - mean**2) / math.sqrt(draws)
    info = 4000 * 2.0 / sigma**2 + 16.0
    floor = mean**2 / info
    assert curve.van_trees_action_floor == pytest.approx(floor, rel=10.0 * mean_sd / abs(mean))
    assert curve.plugin_bayes_ratio > 1.0
    assert curve.hodges_bayes_ratio > curve.plugin_bayes_ratio
    assert curve.hodges_pointwise_ratio == 0.0


# ------------------------------------------------------------------- interference certificates


@FAST
@given(
    mu=st.floats(0.1, 5.0),
    a=st.floats(-3.0, 3.0),
    u_eval=st.floats(-3.0, 3.0),
    kappas=st.lists(st.floats(0.0, 10.0), min_size=1, max_size=5),
)
def test_interference_convexity_aware_bound_is_exact_and_tighter(mu, a, u_eval, kappas):
    """interference_convexity.v (interference_tightens_pl_bound): mu + kappa tightens the bound."""
    curve = R.interference_convexity_certificate(mu=mu, a=a, u_eval=u_eval, cannibalisations=kappas)
    for k, aware, blind, true in zip(
        kappas, curve.aware_bound, curve.blind_bound, curve.true_regret, strict=True
    ):
        mu_eff = mu + k

        def objective(u, mu_eff=mu_eff):
            return 0.5 * mu_eff * u * u - a * u

        best = minimize_scalar(objective).fun
        scale = mu_eff * (abs(u_eval) + abs(a) / mu_eff) ** 2
        assert true == pytest.approx(
            objective(u_eval) - best, rel=1e-9, abs=1e-12 * scale + 1e-20 * mu_eff
        )
        assert aware == pytest.approx(true, rel=1e-12, abs=1e-15 * scale + 1e-300)
        assert aware <= blind * (1 + 1e-15)


def _population_effect(b, eps, orthogonal, alpha=1.0, gamma=1.5):
    if orthogonal:
        return (b + eps * eps) / (1 + eps * eps)
    return b - gamma * eps / (1 - alpha * eps)


def test_interference_orthogonal_slopes_match_the_population_bias_orders():
    """orthogonal_control.v, interference_regret.v: plug-in and half O(eps^2), full O(eps^4)."""
    b_d, b_i, rr, target = 2.0, 1.5, 0.5, 1.0
    errors = np.array([0.2, 0.1, 0.05, 0.025, 0.0125])
    curve = R.interference_orthogonal_certificate(noise=0.0, n=150_000, n_seeds=3)

    def regret(b_hat):
        b_total = b_d + b_i

        def cost(u):
            return (b_total * u - target) ** 2 + rr * u * u

        return cost(target * b_hat / (b_hat**2 + rr)) - minimize_scalar(cost).fun

    expected = {}
    for name, (od, oi) in {"plugin": (0, 0), "half": (1, 0), "full": (1, 1)}.items():
        regs = [
            regret(_population_effect(b_d, e, od) + _population_effect(b_i, e, oi)) for e in errors
        ]
        expected[name] = np.polyfit(np.log(errors), np.log(regs), 1)[0]
    assert curve.plugin_exponent == pytest.approx(expected["plugin"], abs=0.15)
    assert curve.half_orthogonal_exponent == pytest.approx(expected["half"], abs=0.15)
    assert curve.full_orthogonal_exponent == pytest.approx(expected["full"], abs=0.25)
    assert curve.full_orthogonal_exponent > curve.half_orthogonal_exponent + 1.0


def test_interference_regret_is_bounded_by_the_curvature_times_total_error():
    """interference_regret.v (interference_regret_bound): gap <= kappa/2 (eid + eint)^2, slope 2."""
    a, b, q, r, x0 = C2_PLANT

    def gap(v):
        return R.certainty_equivalence_gap(
            a, b, q, r, a + v[:4].reshape(2, 2), b + v[4:].reshape(2, 1), x0
        )

    kappa = float(np.linalg.eigvalsh(_hessian(gap, 6))[-1])
    for ratio in (1.0, 0.3):
        curve = R.interference_regret_certificate(
            a, b, q, r, x0, interference_ratio=ratio, n_samples=120
        )
        assert np.all(curve.gaps <= kappa / 2 * curve.errors**2)
        assert curve.exponent == pytest.approx(2.0, abs=0.2)


@FAST
@given(
    b=st.floats(-3.0, 3.0).filter(lambda v: abs(v) > 0.05),
    effort=st.floats(0.05, 4.0),
    target=st.floats(-3.0, 3.0).filter(lambda v: abs(v) > 0.05),
)
def test_lq_regret_sensitivity_is_the_second_order_regret_coefficient(b, effort, target):
    """confounding_lq_regret.mac (2): regret(b + delta) = L_reg delta^2 + O(delta^3)."""

    def regret(b_hat):
        u_hat = _optimal_action(-b_hat, target, effort)
        u_best = _optimal_action(-b, target, effort)
        return _lq_cost(b, u_hat, target, effort) - _lq_cost(b, u_best, target, effort)

    h = 1e-3 * max(abs(b), math.sqrt(effort))
    second = (regret(b + h) + regret(b - h)) / (2 * h * h)
    scale = target**2 / (b * b + effort)
    value = R.lq_regret_sensitivity(b, effort, target)
    assert value == pytest.approx(second, rel=1e-4, abs=1e-5 * scale)
    assert value >= 0.0
    assert R.lq_regret_sensitivity(b, b * b, target) == 0.0


# ------------------------------------------------------------------------------------ minimax LQ


def _not_tiny(value):
    # a midpoint below ~1e-154 overflows minimax_action's equalising candidate; see the xfail below
    return value == 0.0 or abs(value) > 1e-6


def _worst_over_effects(u, target, b_lo, b_hi, effort, curvature):
    bs = np.linspace(b_lo, b_hi, 201)
    return float(np.max(curvature * (bs * u - target) ** 2 + effort * u * u))


@FAST
@given(
    target=st.floats(-3.0, 3.0),
    b_lo=st.floats(-3.0, 3.0).filter(_not_tiny),
    width=st.floats(0.0, 3.0).filter(_not_tiny),
    effort=st.floats(0.05, 4.0),
    curvature=st.floats(0.0, 4.0),
)
def test_minimax_action_cost_matches_brute_force(target, b_lo, width, effort, curvature):
    """minimax_lq.v (minimax_lower_branch): the candidate enumeration attains min_u max_b cost."""
    b_hi = b_lo + width
    out = R.minimax_action(target, b_lo, b_hi, effort, curvature)
    bound = abs(target) * math.sqrt(curvature / effort) + 1.0
    brute = minimize_scalar(
        lambda u: _worst_over_effects(u, target, b_lo, b_hi, effort, curvature),
        bounds=(-bound, bound), method="bounded", options={"xatol": 1e-12},
    )  # fmt: skip
    scale = curvature * target**2 + 1e-300
    assert out.worst_case == pytest.approx(
        _worst_over_effects(out.action, target, b_lo, b_hi, effort, curvature),
        rel=1e-12, abs=1e-15 * scale,
    )  # fmt: skip
    assert out.worst_case <= brute.fun + 1e-12 * scale


def _refined_max(f, grid):
    """Max of a 1-D function: a dense grid, then a bounded search around every local peak."""
    values = f(grid)
    best = float(values.max())
    peaks = np.flatnonzero(
        (values >= np.roll(values, 1)) & (values >= np.roll(values, -1))
        | (np.arange(grid.size) == 0)
        | (np.arange(grid.size) == grid.size - 1)
    )
    for k in peaks:
        lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, grid.size - 1)]
        if hi > lo:
            found = minimize_scalar(
                lambda v: -float(f(np.array([v]))[0]),
                bounds=(lo, hi), method="bounded", options={"xatol": 1e-14},
            )  # fmt: skip
            best = max(best, -float(found.fun))
    return best


def _worst_regret(u, target, b_lo, b_hi, effort, curvature):
    def regret(b):
        d = curvature * b * b + effort
        return d * (u - curvature * b * target / d) ** 2

    return _refined_max(regret, np.linspace(b_lo, b_hi, 801))


@SLOW
@given(
    target=st.floats(0.2, 3.0),
    b_lo=st.floats(-2.0, 3.0),
    width=st.floats(0.05, 6.0),
    effort=st.floats(0.1, 3.0),
    curvature=st.floats(0.2, 3.0),
)
def test_minimax_action_regret_matches_brute_force(target, b_lo, width, effort, curvature):
    """minimax_lq.mac: the regret criterion's action attains min_u max_b D(b)(u - u*(b))^2."""
    b_hi = b_lo + width
    out = R.minimax_action(target, b_lo, b_hi, effort, curvature, criterion="regret")
    bound = abs(target) * math.sqrt(curvature / effort) + 1.0
    brute = minimize_scalar(
        lambda u: _worst_regret(u, target, b_lo, b_hi, effort, curvature),
        bounds=(-bound, bound), method="bounded", options={"xatol": 1e-12},
    )  # fmt: skip
    at_action = _worst_regret(out.action, target, b_lo, b_hi, effort, curvature)
    assert out.worst_case == pytest.approx(at_action, rel=1e-8, abs=0.0)
    assert out.worst_case <= brute.fun * (1 + 1e-8)


def test_minimax_action_beats_ce_on_the_rocq_instance():
    """minimax_lq.v (ce_not_minimax): on b in [1/2, 3/2], 2/5 costs 4/5 against CE's 13/16."""
    out = R.minimax_action(1.0, 0.5, 1.5, 1.0, 1.0)
    assert out.action == pytest.approx(0.4, rel=1e-15, abs=0.0)
    assert out.worst_case == pytest.approx(0.8, rel=1e-15, abs=0.0)
    assert out.binding == "lower"
    assert _worst_over_effects(0.5, 1.0, 0.5, 1.5, 1.0, 1.0) == pytest.approx(13 / 16, rel=1e-15)


@FAST
@given(
    target=st.floats(0.1, 3.0),
    b_lo=st.floats(0.05, 3.0),
    width=st.floats(0.01, 3.0),
    effort=st.floats(0.05, 4.0),
    curvature=st.floats(0.05, 4.0),
)
def test_minimax_action_branch_is_decided_by_the_kink_test(target, b_lo, width, effort, curvature):
    """minimax_lq.mac (4)-(7): lower-endpoint CE iff curvature b_lo D <= effort, else equaliser."""
    b_hi = b_lo + width
    half = width / 2
    margin = curvature * b_lo * half - effort
    if abs(margin) <= 1e-9 * effort:
        return
    out = R.minimax_action(target, b_lo, b_hi, effort, curvature)
    if margin < 0:
        assert out.action == pytest.approx(
            curvature * b_lo * target / (curvature * b_lo**2 + effort), rel=1e-12, abs=0.0
        )
    else:
        assert out.action == pytest.approx(target / (b_lo + half), rel=1e-12, abs=0.0)
    b_hat = b_lo + half
    ce = curvature * b_hat * target / (curvature * b_hat**2 + effort)
    if abs(curvature * b_lo * b_hat - effort) > 1e-9 * effort:
        assert (out.action < ce) == (curvature * b_lo * b_hat < effort)


@pytest.mark.xfail(
    strict=True,
    reason="target=1, b in [-2, -1], r=curvature=1: the kink test passes (-1 <= 1) but the action "
    "is -0.5 = CE(b_hi), not CE(b_lo) = -0.4; the rule holds for b_lo >= 0",
)
def test_minimax_action_kink_rule_on_a_negative_interval():
    """minimax_lq.v (minimax_lower_branch): the docstring's kink rule, on an interval below zero."""
    out = R.minimax_action(1.0, -2.0, -1.0, 1.0, 1.0)
    assert out.action == pytest.approx(-2.0 / 5.0, rel=1e-12, abs=0.0)


def test_minimax_action_survives_a_tiny_midpoint():
    """minimax_lq.v (stage): every candidate's worst stage cost is a real number or +inf. The
    equalising candidate u = -1.3e189 overflowed u**2 and raised OverflowError."""
    out = R.minimax_action(-1.0, 0.0, 1.5624508741048306e-189, 1.0, 0.0)
    assert out.action == 0.0
    regret = R.minimax_action(-1.0, 0.0, 1.5624508741048306e-189, 1.0, 0.0, criterion="regret")
    assert np.isfinite(regret.action)


def _constant_effect_cost(gains, state_gain, effect, state_cost, effort, terminal):
    p = terminal
    for gain in gains[::-1]:
        s = -gain
        p = state_cost + effort * s * s + p * (state_gain + effect * s) ** 2
    return p


def _per_step_worst(gains, state_gain, b_lo, b_hi, state_cost, effort, terminal):
    bs = np.linspace(b_lo, b_hi, 101)
    p = terminal
    for gain in gains[::-1]:
        s = -gain
        p = state_cost + effort * s * s + p * float(np.max((state_gain + bs * s) ** 2))
    return p


@FAST
@given(
    state_gain=st.floats(-2.0, 2.0),
    b_lo=st.floats(-2.0, 2.0).filter(_not_tiny),
    width=st.floats(0.0, 2.0).filter(_not_tiny),
    state_cost=st.floats(0.1, 2.0),
    effort=st.floats(0.1, 2.0),
    terminal=st.floats(0.0, 3.0),
    horizon=st.integers(1, 6),
)
def test_minimax_lq_value_is_the_constant_effect_worst_case(
    state_gain, b_lo, width, state_cost, effort, terminal, horizon
):
    """minimax_lq.mac (8), (9): the per-step robust value is attained by a constant effect."""
    b_hi = b_lo + width
    pol = R.minimax_lq_policy(state_gain, b_lo, b_hi, state_cost, effort, terminal, horizon)
    per_step = _per_step_worst(pol.gains, state_gain, b_lo, b_hi, state_cost, effort, terminal)
    constant = [
        _constant_effect_cost(pol.gains, state_gain, b, state_cost, effort, terminal)
        for b in np.linspace(b_lo, b_hi, 101)
    ]
    assert pol.value == pytest.approx(per_step, rel=1e-12, abs=0.0)
    assert pol.value == pytest.approx(max(constant), rel=1e-10, abs=0.0)
    if b_hi > 0.0:
        assert pol.value == pytest.approx(constant[0], rel=1e-10, abs=0.0)
    assert pol.value <= pol.ce_value * (1 + 1e-12)
    if horizon == 1:
        brute = minimize_scalar(
            lambda s: state_cost
            + effort * s * s
            + terminal * float(np.max((state_gain + np.linspace(b_lo, b_hi, 101) * s) ** 2)),
            bounds=(-10.0, 10.0), method="bounded", options={"xatol": 1e-12},
        )  # fmt: skip
        assert pol.value <= brute.fun * (1 + 1e-10)


@pytest.mark.xfail(
    strict=True,
    reason="a=1, b in [-2, -1], q=r=1, p_T=0, T=2: value 1.5 but the constant effect b_lo costs "
    "1.25; on a negative interval b_hi, the endpoint nearest 0, attains it",
)
def test_minimax_lq_value_is_attained_at_b_lo_on_a_negative_interval():
    """minimax_lq.mac (9): the docstring's constant effect b_lo, on an interval below zero."""
    pol = R.minimax_lq_policy(1.0, -2.0, -1.0, 1.0, 1.0, 0.0, 2)
    constant = _constant_effect_cost(pol.gains, 1.0, -2.0, 1.0, 1.0, 0.0)
    assert pol.value == pytest.approx(constant, rel=1e-10, abs=0.0)


@FAST
@given(
    state_gain=st.floats(-2.0, 2.0),
    b_lo=st.floats(-2.0, 2.0).filter(_not_tiny),
    width=st.floats(0.0, 2.0).filter(_not_tiny),
    horizon=st.integers(1, 5),
    seed=st.integers(0, 10_000),
)
def test_minimax_lq_gains_beat_perturbed_gains(state_gain, b_lo, width, horizon, seed):
    """minimax_lq.mac (8): dynamic programming over the per-step adversary is minimax."""
    pol = R.minimax_lq_policy(state_gain, b_lo, b_lo + width, 1.0, 0.5, 2.0, horizon)
    rng = np.random.default_rng(seed)
    for _ in range(10):
        other = pol.gains + 0.3 * rng.standard_normal(horizon)
        worst = _per_step_worst(other, state_gain, b_lo, b_lo + width, 1.0, 0.5, 2.0)
        assert pol.value <= worst * (1 + 1e-12)


# --------------------------------------------------------------------------- minimax regret move


def _move_worst_regret(rho, k, sigma):
    """Worst regret of clip((x + rho)/(2 rho), 0, 1), rho = 0 the sign rule, x ~ N(beta, s^2)."""

    def phi(z):
        return np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

    def expected_share(beta):
        if rho == 0.0:
            return ndtr(beta / sigma)
        lo, hi = (-rho - beta) / sigma, (rho - beta) / sigma
        ramp = (beta + rho) * (ndtr(hi) - ndtr(lo)) + sigma * (phi(lo) - phi(hi))
        return ndtr(-hi) + ramp / (2 * rho)

    def regret(beta):
        d = expected_share(beta)
        return np.maximum(np.maximum(beta + k, 0.0) * (1 - d), np.maximum(k - beta, 0.0) * d)

    span = k + 10 * sigma
    return _refined_max(regret, np.linspace(-span, span, 1201))


@SLOW
@given(
    sigma=st.floats(0.2, 2.0),
    gamma=st.floats(1.0, 6.0),
    gap=st.floats(0.1, 6.0),
    estimate=st.floats(-2.0, 2.0),
)
def test_minimax_regret_move_worst_regret_matches_quadrature(sigma, gamma, gap, estimate):
    """minimax_regret_move.mac: the rule's worst regret is k/2 (linear) or max (b + k) Phi(-b/s)."""
    out = R.minimax_regret_move(estimate, sigma, gamma, cvar_gap=gap)
    k = out.halfwidth
    brute = _move_worst_regret(out.threshold, k, sigma)
    assert out.worst_regret == pytest.approx(brute, rel=1e-6, abs=1e-9 * sigma)
    sign = _move_worst_regret(0.0, k, sigma)
    assert out.sign_regret == pytest.approx(sign, rel=1e-6, abs=1e-9 * sigma)


def test_minimax_regret_move_documented_regrets():
    """minimax_regret_move.mac: sign rule 1.0509 vs 1 at k = 2, 3.371 vs 2.5 at k = 5, sigma 1."""
    two = R.minimax_regret_move(0.3, 1.0, 3.0, cvar_gap=4.0)
    five = R.minimax_regret_move(0.3, 1.0, 3.0, cvar_gap=10.0)
    assert two.halfwidth == pytest.approx(2.0, rel=1e-15, abs=0.0)
    assert (two.rule, five.rule) == ("linear", "linear")
    assert two.sign_regret == pytest.approx(1.0509, abs=5e-5)
    assert two.worst_regret == pytest.approx(1.0, rel=1e-15, abs=0.0)
    assert five.sign_regret == pytest.approx(3.371, abs=5e-4)
    assert five.worst_regret == pytest.approx(2.5, rel=1e-15, abs=0.0)


@SLOW
@given(sigma=st.floats(0.3, 2.0), k_over_sigma=st.floats(0.2, 4.0))
def test_minimax_regret_move_no_linear_rule_does_better(sigma, k_over_sigma):
    """minimax_regret_move.mac: no rule of the linear family beats the returned worst regret."""
    k = k_over_sigma * sigma
    gamma = (1 + k) / (1 - k) if k < 1 else 3.0
    gap = 1.0 if k < 1 else 2.0 * k
    out = R.minimax_regret_move(0.0, sigma, gamma, cvar_gap=gap)
    assert out.halfwidth == pytest.approx(k, rel=1e-12, abs=0.0)
    for rho in np.geomspace(0.05 * sigma, 3.0 * k + sigma, 5):
        rival = _move_worst_regret(float(rho), k, sigma)
        assert out.worst_regret <= rival * (1 + 1e-6)


@FAST
@given(
    estimate=st.floats(0.05, 3.0),
    sigma=st.floats(0.05, 2.0),
    gap=st.floats(0.5, 20.0),
)
def test_minimax_regret_move_full_move_gamma_is_the_share_boundary(estimate, sigma, gap):
    """minimax_regret_move.mac: the move is whole up to full_move_gamma and partial past it."""
    out = R.minimax_regret_move(estimate, sigma, 1.0, cvar_gap=gap)
    g = out.full_move_gamma
    if math.isinf(g):
        assert R.minimax_regret_move(estimate, sigma, 1e6, cvar_gap=gap).share == 1.0
        return
    assert R.minimax_regret_move(estimate, sigma, g * (1 - 1e-6), cvar_gap=gap).share == 1.0
    assert R.minimax_regret_move(estimate, sigma, g * (1 + 1e-4), cvar_gap=gap).share < 1.0


# ---------------------------------------------------------------------- minimax exploration floor


@FAST
@given(
    b=st.floats(0.3, 2.0),
    rr=st.floats(0.1, 2.0).filter(lambda v: v > 0.0),
    sigma=st.floats(0.2, 2.0),
    eta=st.floats(0.05, 1.0),
    horizon=st.sampled_from([10**4, 10**5, 10**6]),
)
def test_minimax_exploration_floor_is_the_optimal_budget_value(b, rr, sigma, eta, horizon):
    """minimax_exploration.v (minimax_sequence_floor): the floor is min_S A S + T K/I(S)."""
    if abs(rr - b * b) < 0.05:
        return
    i0 = 1.0
    curve = R.minimax_exploration_certificate(
        b=b, rr=rr, sigma=sigma, i0=i0, eta=eta, horizons=(horizon,)
    )
    a_curv = b * b + rr
    slope = -(rr - b * b) / (rr + b * b) ** 2  # psi'(b) at xt = 1
    k_id = a_curv * slope**2

    def total(budget):
        return a_curv * budget + horizon * k_id / (i0 + eta * budget / sigma**2)

    s_star = sigma**2 / eta * (abs(slope) * math.sqrt(horizon * eta) / sigma - i0)
    if s_star <= 0.0:
        return
    brute = minimize_scalar(
        total, bounds=(0.0, 4 * s_star), method="bounded", options={"xatol": 1e-9}
    )
    assert curve.floor[0] == pytest.approx(brute.fun, rel=1e-8, abs=0.0)
    assert curve.burst_regret[0] >= curve.floor[0]


@FAST
@given(
    shape=st.sampled_from(["decay", "block", "random"]),
    scale=st.floats(1e-3, 50.0),
    rate=st.floats(0.0, 2.0),
    seed=st.integers(0, 10_000),
)
def test_minimax_exploration_no_schedule_beats_the_floor(shape, scale, rate, seed):
    """minimax_exploration.v (amgm_floor): every exploration schedule pays at least the floor."""
    b, rr, xt, sigma, i0, eta, horizon = 1.0, 0.5, 1.0, 0.7, 1.0, 0.6, 2000
    curve = R.minimax_exploration_certificate(horizons=(horizon,))
    a_curv = b * b + rr
    slope = (_optimal_action(b + 1e-6, xt, rr) - _optimal_action(b - 1e-6, xt, rr)) / 2e-6
    k_id = a_curv * slope**2
    t = np.arange(1, horizon + 1, dtype=float)
    rng = np.random.default_rng(seed)
    if shape == "decay":
        sched = scale * t**-rate
    elif shape == "block":
        sched = np.where(t <= 1 + rate * 50, scale, 0.0)
    else:
        sched = scale * rng.exponential(size=horizon) * (rng.random(horizon) < 0.1)
    info = i0 + eta / sigma**2 * np.concatenate([[0.0], np.cumsum(sched)[:-1]])
    assert float(np.sum(a_curv * sched + k_id / info)) >= curve.floor[0]


def test_minimax_exploration_burst_attains_and_taper_pays_sqrt_two():
    """minimax_exploration.v (taper_gap_is_sqrt_two): burst/floor -> 1, taper/floor -> sqrt 2."""
    curve = R.minimax_exploration_certificate()
    a_curv = 1.0 + 0.5
    k_id = a_curv * ((0.5 - 1.0) / a_curv**2) ** 2
    assert np.all(curve.burst_regret - curve.floor <= k_id / 1.0)
    assert np.all(curve.burst_regret >= curve.floor)
    assert curve.min_policy_ratio >= 1.0
    assert curve.burst_over_floor == pytest.approx(1.0, abs=1e-4)
    assert curve.taper_over_floor == pytest.approx(math.sqrt(2.0), abs=5e-3)
    assert curve.eta_slope == pytest.approx(-0.5, rel=1e-12, abs=0.0)
