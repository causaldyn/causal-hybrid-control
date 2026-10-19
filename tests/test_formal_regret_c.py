"""Ties the cited proofs and derivations of twenty ``chc.regret`` objects to their float code.

Each test checks a cited statement -- a Rocq lemma or a Maxima closed form -- on the code that
implements it, by a route other than the code's own: a Riccati iteration in place of the DARE, a
least-squares action in place of the closed-form one, an enumeration in place of a bound, a
quadrature in place of a Monte-Carlo mean, or a population limit derived by hand.
"""

from __future__ import annotations

import itertools
from functools import cache

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import minimize, minimize_scalar

from chc.network_causal import cycle_shells
from chc.regret import (
    _dml_two_channel,
    _psi_ratio_closed_form,
    _shifted_kyfan_bound,
    fold_exactness_certificate,
    multichannel_control_certificate,
    multivariate_action_floor,
    multivariate_interference_certificate,
    multivariate_transfer_certificate,
    multivariate_van_trees_certificate,
    nonlinear_regret_certificate,
    optimal_exploration_certificate,
    optimal_fold_partition,
    orthogonal_control_certificate,
    panel_estimator_certificate,
    partial_id_control_certificate,
    pessimism_variance_certificate,
    regret_scaling,
    space_time_fold_certificate,
    strong_convexity_regret_bound,
    transportability_regret_certificate,
    van_trees_certificate,
)

FORMAL_LINKS = {
    "chc.regret.regret_scaling": {
        "cites": ("ce_explicit_constants.v",),
        "tests": (
            "test_regret_scaling_gaps_are_the_gain_gap_identity",
            "test_regret_scaling_exponent_is_two_in_the_small_error_limit",
            "test_regret_scaling_reports_unbounded_draws_that_vanish_with_the_error",
        ),
    },
    "chc.regret.orthogonal_control_certificate": {
        "cites": ("orthogonal_control.mac", "orthogonal_control.v"),
        "tests": (
            "test_orthogonal_control_regrets_meet_their_population_limits",
            "test_orthogonal_control_orthogonal_regret_never_exceeds_the_plugin_one",
        ),
    },
    "chc.regret.pessimism_variance_certificate": {
        "cites": ("pessimistic_optimality.mac", "pessimistic_optimality.v"),
        "tests": (
            "test_pessimism_optimal_rho_is_the_effect_variance",
            "test_pessimism_regret_gain_is_the_completed_square",
        ),
    },
    "chc.regret.multivariate_action_floor": {
        "cites": (
            "multivariate_van_trees.mac",
            "multivariate_van_trees.v",
            "multivariate_van_trees_lift.v",
        ),
        "tests": (
            "test_multivariate_action_floor_is_the_trace_against_an_independent_jacobian",
            "test_multivariate_action_floor_regret_is_an_exact_quadratic_form",
            "test_multivariate_action_floor_is_antitone_in_the_information",
            "test_multivariate_action_floor_ignores_a_blind_direction",
            "test_multivariate_action_floor_is_local_as_its_docstring_says",
            "test_multivariate_action_floor_refuses_an_indefinite_curvature",
        ),
    },
    "chc.regret.multivariate_van_trees_certificate": {
        "cites": ("multivariate_van_trees.mac", "multivariate_van_trees.v"),
        "tests": (
            "test_multivariate_van_trees_bayes_floor_is_below_the_bayes_risk",
            "test_multivariate_van_trees_confounding_price_is_a_convex_combination",
        ),
    },
    "chc.regret.multivariate_transfer_certificate": {
        "cites": ("composition_transfer.v",),
        "tests": ("test_multivariate_transfer_regret_is_order_two_p_with_the_lq_constant",),
    },
    "chc.regret.multichannel_control_certificate": {
        "cites": ("multichannel_control.mac", "multichannel_control.v"),
        "tests": (
            "test_multichannel_regrets_meet_the_population_bias",
            "test_multichannel_full_orthogonal_regret_never_exceeds_the_half_one",
        ),
    },
    "chc.regret.multivariate_interference_certificate": {
        "cites": ("multichannel_control.v",),
        "tests": ("test_multivariate_interference_orders_follow_the_least_debiased_channel",),
    },
    "chc.regret.transportability_regret_certificate": {
        "cites": ("transportability_regret.mac", "transportability_regret.v"),
        "tests": (
            "test_transportability_exact_regret_is_the_curvature_identity",
            "test_transportability_exact_regret_follows_the_maxima_expansion",
            "test_transportability_wdro_bound_covers_the_quadratic_residual",
            "test_transportability_wdro_bound_covers_the_exact_regret",
        ),
    },
    "chc.regret.van_trees_certificate": {
        "cites": ("van_trees.mac", "van_trees.v"),
        "tests": (
            "test_van_trees_posterior_mean_attains_the_floor",
            "test_van_trees_confounding_raises_the_floor",
        ),
    },
    "chc.regret.partial_id_control_certificate": {
        "cites": ("partial_id_control.mac", "partial_id_control.v"),
        "tests": (
            "test_partial_id_sign_is_identified_iff_the_width_is_below_the_estimate",
            "test_partial_id_ce_worst_regret_is_the_maxima_constant_times_width_squared",
            "test_partial_id_minimax_action_does_no_worse_than_certainty_equivalence",
        ),
    },
    "chc.regret.optimal_exploration_certificate": {
        "cites": ("optimal_exploration.mac", "optimal_exploration.v"),
        "tests": (
            "test_optimal_exploration_optimum_minimises_the_excess",
            "test_optimal_exploration_estimation_cost_is_the_inverse_chi_square_mean",
            "test_optimal_exploration_confounding_raises_the_optimum_and_the_floor",
        ),
    },
    "chc.regret.nonlinear_regret_certificate": {
        "cites": ("nonlinear_regret.mac", "nonlinear_regret.v"),
        "tests": ("test_nonlinear_regret_bound_gap_is_the_maxima_factor",),
    },
    "chc.regret.strong_convexity_regret_bound": {
        "cites": ("nonlinear_regret.v",),
        "tests": (
            "test_strong_convexity_bound_holds_on_random_scalar_costs",
            "test_strong_convexity_bound_holds_on_random_vector_costs",
            "test_strong_convexity_bound_is_tight_on_a_quadratic",
        ),
    },
    "chc.regret.optimal_fold_partition": {
        "cites": ("fold_spectrum_law.mac", "fold_spectrum_law.v"),
        "tests": (
            "test_fold_partition_on_c8_follows_the_two_threshold_law",
            "test_fold_partition_with_one_shell_follows_the_single_threshold",
            "test_fold_partition_never_picks_contiguous_blocks",
        ),
    },
    "chc.regret._shifted_kyfan_bound": {
        "cites": ("fold_shifted_bound.mac",),
        "tests": (
            "test_shifted_kyfan_bound_lies_between_ky_fan_and_the_optimum",
            "test_shifted_kyfan_bound_dominates_every_valid_shift",
            "test_shifted_kyfan_bound_equals_ky_fan_on_a_circulant",
            "test_shifted_kyfan_bound_moves_by_c_m_under_a_diagonal_shift",
        ),
    },
    "chc.regret.fold_exactness_certificate": {
        "cites": ("fold_cut_duality.v",),
        "tests": (
            "test_fold_exactness_structural_gates_hold",
            "test_fold_exactness_dynamic_program_is_the_max_cut_optimum",
        ),
    },
    "chc.regret.space_time_fold_certificate": {
        "cites": ("space_time_folds.mac", "space_time_folds.v"),
        "tests": (
            "test_space_time_panel_weight_is_the_toeplitz_ratio",
            "test_space_time_kronecker_ranks_follow_the_commutation_law",
            "test_space_time_one_axis_optimum_is_the_enumerated_one",
        ),
    },
    "chc.regret._psi_ratio_closed_form": {
        "cites": ("panel_estimator_gate.mac",),
        "tests": (
            "test_psi_ratio_closed_form_is_the_explicit_sandwich_ratio",
            "test_psi_ratio_washes_out_monotonically_in_the_cluster_count",
        ),
    },
    "chc.regret.panel_estimator_certificate": {
        "cites": ("panel_estimator_gate.mac",),
        "tests": ("test_panel_estimator_prediction_is_the_explicit_sandwich_ratio",),
    },
}

SETTINGS = settings(max_examples=25, deadline=None)


# ---------------------------------------------------------------- independent LQ machinery


def _riccati(a, b, q, r):
    """Gain and cost-to-go by value iteration -- no DARE solver involved."""
    p = q.copy()
    for _ in range(200_000):
        gain = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
        nxt = q + a.T @ p @ (a - b @ gain)
        done = np.max(np.abs(nxt - p)) <= 1e-15 * np.max(np.abs(nxt))
        p = nxt
        if done:
            break
    return np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a), p


def _lyapunov_sum(a_cl, weight):
    """``sum_t (A^t)' W A^t`` by doubling -- no Lyapunov solver involved."""
    total, power = weight.copy(), a_cl.copy()
    for _ in range(64):
        total = total + power.T @ total @ power
        power = power @ power
    return total


TRANSFER_PLANT = (
    np.array([[1.0, 0.1], [0.0, 0.95]]),
    np.array([[0.5], [1.0]]),
    np.eye(2),
    np.array([[0.5]]),
    np.array([1.0, 0.5]),
)


def _second_order_regret(direction):
    """Leading coefficient of ``J(K(B + t D)) - J*`` in ``t^2``, from the gain-gap identity."""
    a, b, q, r, x0 = TRANSFER_PLANT
    gain, p = _riccati(a, b, q, r)
    step = 1e-6
    slope = (
        _riccati(a, b + step * direction, q, r)[0] - _riccati(a, b - step * direction, q, r)[0]
    ) / (2.0 * step)
    curvature = r + b.T @ p @ b
    return float(x0 @ _lyapunov_sum(a - b @ gain, slope.T @ curvature @ slope) @ x0)


def _scalar_constant(b, rr, xt):
    return xt**2 * (rr - b**2) ** 2 / (rr + b**2) ** 3


def _scalar_regret(b_hat, b, rr, xt):
    """``(b^2+rr)(u*(b_hat) - u*(b))^2``: the exact one-step regret of the action for ``b_hat``."""
    return (b**2 + rr) * (xt * b_hat / (b_hat**2 + rr) - xt * b / (b**2 + rr)) ** 2


# ---------------------------------------------------------------- regret_scaling


def test_regret_scaling_gaps_are_the_gain_gap_identity() -> None:
    """ce_explicit_constants.v gain_gap_identity: J(K') - J* sums (K'-K)'R_K(K'-K) on the loop."""
    a = np.array([[0.9, 0.2], [0.0, 0.7]])
    b = np.array([[0.5], [1.0]])
    q, r, x0 = np.eye(2), np.array([[1.0]]), np.array([1.0, -0.5])
    errors, n_samples, seed = (0.02, 0.005), 30, 3
    curve = regret_scaling(a, b, q, r, x0, errors=errors, n_samples=n_samples, seed=seed)
    gain, p = _riccati(a, b, q, r)
    curvature = r + b.T @ p @ b
    rng = np.random.default_rng(seed)
    for level, eps in enumerate(errors):
        errs, gaps = [], []
        for _ in range(n_samples):
            d_a, d_b = eps * rng.normal(size=a.shape), eps * rng.normal(size=b.shape)
            moved = _riccati(a + d_a, b + d_b, q, r)[0] - gain
            gaps.append(
                float(x0 @ _lyapunov_sum(a - b @ (gain + moved), moved.T @ curvature @ moved) @ x0)
            )
            errs.append(float(np.sqrt(np.sum(d_a**2) + np.sum(d_b**2))))
        assert curve.infinite_fraction[level] == 0.0
        assert curve.gaps[level] == pytest.approx(np.median(gaps), rel=1e-7, abs=0.0)
        assert curve.errors[level] == pytest.approx(np.median(errs), rel=1e-14, abs=0.0)


def test_regret_scaling_exponent_is_two_in_the_small_error_limit() -> None:
    """ce_explicit_constants.v: the suboptimality is a quadratic form in the gain error, slope 2."""
    a = np.array([[0.9, 0.2], [0.0, 0.7]])
    b = np.array([[0.5], [1.0]])
    curve = regret_scaling(
        a, b, np.eye(2), np.array([[1.0]]), np.array([1.0, -0.5]),
        errors=(1e-2, 1e-3, 1e-4), n_samples=40, seed=5,
    )  # fmt: skip
    assert np.all(curve.infinite_fraction == 0.0)
    assert curve.exponent == pytest.approx(2.0, abs=0.25)


def test_regret_scaling_reports_unbounded_draws_that_vanish_with_the_error() -> None:
    """ce_explicit_constants.v perturbed_loop_stable: inside a radius every draw stabilises."""
    a, b, q, r, x0 = TRANSFER_PLANT
    curve = regret_scaling(a, b, q, r, x0, errors=(0.5, 0.01), n_samples=60, seed=0)
    assert curve.infinite_fraction[0] > 0.0
    assert curve.infinite_fraction[1] == 0.0


# ---------------------------------------------------------------- orthogonal_control_certificate


def test_orthogonal_control_regrets_meet_their_population_limits() -> None:
    """orthogonal_control.mac Parts 1-3: plug-in effect error O(eps), orthogonal O(eps^2)."""
    b, alpha, gamma, target, effort, noise, n = 2.0, 1.0, 1.5, 1.0, 0.5, 0.05, 400_000
    errors = (0.2, 0.1)
    curve = orthogonal_control_certificate(n=n, n_seeds=1, errors=errors)

    def regret(b_hat):
        return _scalar_regret(b_hat, b, effort, target)

    for j, eps in enumerate(errors):
        # u_res = nu - eps z; the plug-in limit is <u_res, y>/<u_res, u>, the orthogonal one
        # <u_res, y - m_y>/<u_res, u_res>; u_res and the residual w are uncorrelated Gaussians,
        # so Var(u_res w) = Var(u_res) Var(w) and the estimator SD follows in closed form.
        single = b - eps * gamma / (1.0 - eps * alpha)
        w_single = ((b - single) * alpha + gamma) ** 2 + (b - single) ** 2 + noise**2
        sd_single = np.sqrt((1.0 + eps**2) * w_single / n) / (1.0 - eps * alpha)
        orth = (b + eps**2) / (1.0 + eps**2)
        w_orth = (b - orth) ** 2 + eps**2 * (orth - 1.0) ** 2 + noise**2
        sd_orth = np.sqrt(w_orth / ((1.0 + eps**2) * n))
        for measured, limit, sd in (
            (curve.single_regret[j], single, sd_single),
            (curve.orthogonal_regret[j], orth, sd_orth),
        ):
            slope = abs(regret(limit + 1e-7) - regret(limit - 1e-7)) / 2e-7
            assert abs(measured - regret(limit)) <= 5.0 * slope * sd


def test_orthogonal_control_orthogonal_regret_never_exceeds_the_plugin_one() -> None:
    """orthogonal_control.v orthogonal_dominates_plugin: for eps in [0,1] quartic <= quadratic."""
    curve = orthogonal_control_certificate(n=100_000, n_seeds=1, errors=(0.8, 0.4, 0.2, 0.1))
    assert np.all(curve.orthogonal_regret <= curve.single_regret)


# ---------------------------------------------------------------- pessimism_variance_certificate


def test_pessimism_optimal_rho_is_the_effect_variance() -> None:
    """pessimistic_optimality.v: grad_num = 2 b0^2 xt^2 (rho - s2) vanishes exactly at rho = s2."""
    b0, rr, target, n, seed = 1.0, 1.0, 1.0, 50_000, 4
    variances, rho_max, rho_points = (0.05, 0.1, 0.2, 0.4), 1.2, 121
    curve = pessimism_variance_certificate(
        b0=b0, rr=rr, target=target, variances=variances, n=n,
        rho_max=rho_max, rho_points=rho_points, seed=seed,
    )  # fmt: skip
    spacing = rho_max / (rho_points - 1)
    rng = np.random.default_rng(seed)
    for i, s2 in enumerate(variances):
        b = b0 + np.sqrt(s2) * rng.standard_normal(n)
        m1, m2 = float(b.mean()), float((b**2).mean())
        # the sample-mean regret is quadratic in u with minimiser target*m1/(m2+rr)
        rho_hat = b0 * (m2 + rr) / m1 - b0**2 - rr
        assert abs(curve.optimal_rho[i] - rho_hat) <= spacing + 1e-12
        grad = np.array([-b0 * (s2 + b0**2 + rr) / b0**2, b0 / b0])
        cov = np.array([[s2, 2 * b0 * s2], [2 * b0 * s2, 4 * b0**2 * s2 + 2 * s2**2]])
        assert abs(rho_hat - s2) <= 5.0 * np.sqrt(grad @ cov @ grad / n)


def test_pessimism_regret_gain_is_the_completed_square() -> None:
    """pessimistic_optimality.mac (2): E[regret] is quadratic in u, so CE minus pessimistic regret
    is a difference of two squares about the minimiser."""
    b0, rr, target, n, seed = 1.0, 1.0, 1.0, 20_000, 9
    variances = (0.05, 0.2, 0.6)
    curve = pessimism_variance_certificate(
        b0=b0, rr=rr, target=target, variances=variances, n=n, rho_points=7, seed=seed
    )
    rng = np.random.default_rng(seed)
    for i, s2 in enumerate(variances):
        b = b0 + np.sqrt(s2) * rng.standard_normal(n)
        m1, m2 = float(b.mean()), float((b**2).mean())
        best = target * m1 / (m2 + rr)
        u_ce, u_pess = target * b0 / (b0**2 + rr), target * b0 / (b0**2 + rr + s2)
        gain = (m2 + rr) * ((u_ce - best) ** 2 - (u_pess - best) ** 2)
        assert curve.ce_regret[i] - curve.pessimistic_regret[i] == pytest.approx(
            gain, rel=1e-9, abs=0.0
        )
        assert curve.pessimistic_regret[i] < curve.ce_regret[i]


# ---------------------------------------------------------------- multivariate_action_floor


def _plant(seed, states, actions):
    rng = np.random.default_rng(seed)
    effect = rng.normal(size=(states, actions))
    lq = rng.normal(size=(states, states))
    lr = rng.normal(size=(actions, actions))
    state_weight = lq @ lq.T + 0.2 * np.eye(states)
    action_weight = lr @ lr.T + 0.2 * np.eye(actions)
    target = rng.normal(size=states)
    d = states * actions
    w = rng.normal(size=(d, d))
    information = w @ w.T + 0.5 * np.eye(d)
    return effect, state_weight, action_weight, target, information


def _least_squares_action(effect, state_weight, action_weight, target):
    """argmin of ``(x + Bu)'Q(x + Bu) + u'Ru`` as a stacked least-squares problem."""
    lq = np.linalg.cholesky(state_weight)
    lr = np.linalg.cholesky(action_weight)
    design = np.vstack([lq.T @ effect, lr.T])
    rhs = -np.concatenate([lq.T @ target, np.zeros(effect.shape[1])])
    return np.linalg.lstsq(design, rhs, rcond=None)[0]


def _difference_jacobian(effect, state_weight, action_weight, target, step=1e-5):
    columns = []
    for index in range(effect.size):
        bump = np.zeros_like(effect)
        bump.flat[index] = step
        columns.append(
            (
                _least_squares_action(effect + bump, state_weight, action_weight, target)
                - _least_squares_action(effect - bump, state_weight, action_weight, target)
            )
            / (2.0 * step)
        )
    return np.stack(columns, axis=1)


@SETTINGS
@given(st.integers(0, 2**32 - 1), st.integers(1, 3), st.integers(1, 3))
def test_multivariate_action_floor_is_the_trace_against_an_independent_jacobian(
    seed: int, states: int, actions: int
) -> None:
    """multivariate_van_trees_lift.v the_floor_decomposes_over_eigendirections: the floor is
    tr(M Psi' G^-1 Psi'') = sum_i (Psi' v_i)' M (Psi' v_i) / lambda_i."""
    effect, state_weight, action_weight, target, information = _plant(seed, states, actions)
    out = multivariate_action_floor(
        effect=effect, state_weight=state_weight, action_weight=action_weight,
        target=target, information=information,
    )  # fmt: skip
    jac = _difference_jacobian(effect, state_weight, action_weight, target)
    curvature = effect.T @ state_weight @ effect + action_weight
    floor = float(np.trace(curvature @ jac @ np.linalg.solve(information, jac.T)))
    assert out.floor == pytest.approx(floor, rel=1e-6, abs=0.0)
    lam, vec = np.linalg.eigh(information)
    mapped = jac @ vec
    shares = np.einsum("ij,ik,kj->j", mapped, curvature, mapped) / lam
    assert out.direction_floors == pytest.approx(shares, rel=1e-5, abs=1e-12 * floor)
    assert out.direction_floors.sum() == pytest.approx(out.floor, rel=1e-12, abs=0.0)


@SETTINGS
@given(st.integers(0, 2**32 - 1), st.integers(1, 3), st.integers(1, 3))
def test_multivariate_action_floor_regret_is_an_exact_quadratic_form(
    seed: int, states: int, actions: int
) -> None:
    """multivariate_van_trees_lift.v regret_is_an_exact_quadratic_form_in_the_action:
    J(u) - J(u*) = (u - u*)' M (u - u*) at any u, with u* and M from the floor."""
    effect, state_weight, action_weight, target, information = _plant(seed, states, actions)
    out = multivariate_action_floor(
        effect=effect, state_weight=state_weight, action_weight=action_weight,
        target=target, information=information,
    )  # fmt: skip

    def cost(u):
        moved = target + effect @ u
        return float(moved @ state_weight @ moved + u @ action_weight @ u)

    assert out.optimal_action == pytest.approx(
        _least_squares_action(effect, state_weight, action_weight, target), rel=1e-9, abs=1e-12
    )
    u = out.optimal_action + np.random.default_rng(seed).normal(size=actions)
    gap = u - out.optimal_action
    assert cost(u) - cost(out.optimal_action) == pytest.approx(
        float(gap @ out.curvature @ gap), rel=1e-9, abs=1e-12 * cost(u)
    )


@SETTINGS
@given(st.integers(0, 2**32 - 1), st.integers(1, 3), st.integers(1, 3))
def test_multivariate_action_floor_is_antitone_in_the_information(
    seed: int, states: int, actions: int
) -> None:
    """multivariate_van_trees.v more_information_lowers_the_floor, in the PSD order."""
    effect, state_weight, action_weight, target, information = _plant(seed, states, actions)
    extra = np.random.default_rng(seed + 1).normal(size=(information.shape[0], 2))
    kwargs = {
        "effect": effect,
        "state_weight": state_weight,
        "action_weight": action_weight,
        "target": target,
    }
    less = multivariate_action_floor(information=information, **kwargs).floor
    more = multivariate_action_floor(information=information + extra @ extra.T, **kwargs).floor
    assert more <= less * (1.0 + 1e-12)


@SETTINGS
@given(st.integers(0, 2**32 - 1), st.floats(1.5, 50.0))
def test_multivariate_action_floor_ignores_a_blind_direction(seed: int, loss: float) -> None:
    """multivariate_van_trees.v a_blind_direction_contributes_nothing: cutting the information
    along a kernel direction of Psi' leaves the floor; cutting it along Psi's row space does not."""
    effect, state_weight, action_weight, target, _ = _plant(seed, 2, 1)
    jac = _difference_jacobian(effect, state_weight, action_weight, target)
    blind = np.linalg.svd(jac)[2][-1]
    seen = jac[0] / np.linalg.norm(jac[0])
    kwargs = {
        "effect": effect,
        "state_weight": state_weight,
        "action_weight": action_weight,
        "target": target,
    }
    base = multivariate_action_floor(information=np.eye(2), **kwargs).floor
    cut_blind = np.eye(2) - (1.0 - 1.0 / loss) * np.outer(blind, blind)
    cut_seen = np.eye(2) - (1.0 - 1.0 / loss) * np.outer(seen, seen)
    assert multivariate_action_floor(information=cut_blind, **kwargs).floor == pytest.approx(
        base, rel=1e-8, abs=0.0
    )
    assert multivariate_action_floor(information=cut_seen, **kwargs).floor == pytest.approx(
        loss * base, rel=1e-8, abs=0.0
    )


def test_multivariate_action_floor_is_local_as_its_docstring_says() -> None:
    """multivariate_van_trees.v: the floor is read at one plant; the docstring's scalar plant at
    b = 0 has floor 1/3 while the posterior Bayes action regrets about 0.075."""
    rr, xt, prior_sd, n_obs = 0.5, 1.0, 1.0, 5
    out = multivariate_action_floor(
        effect=np.array([[0.0]]), state_weight=np.eye(1), action_weight=np.array([[rr]]),
        target=np.array([xt]), information=np.array([[n_obs + 1.0 / prior_sd**2]]),
    )  # fmt: skip
    assert out.floor == pytest.approx(1.0 / 3.0, rel=1e-12, abs=0.0)
    post_var = 1.0 / (n_obs + 1.0 / prior_sd**2)
    nodes, weights = hermegauss(80)
    weights = weights / weights.sum()
    risk = 0.0
    for mean, w_mean in zip(np.sqrt(prior_sd**2 - post_var) * nodes, weights, strict=True):
        b = mean + np.sqrt(post_var) * nodes
        curv = b**2 + rr
        action = -b * xt / curv
        risk += w_mean * (
            weights @ (curv * action**2) - (weights @ (curv * action)) ** 2 / (weights @ curv)
        )
    assert risk == pytest.approx(0.075, abs=5e-4)
    assert risk < out.floor


def test_multivariate_action_floor_refuses_an_indefinite_curvature() -> None:
    """multivariate_van_trees.v (B): the trace bound needs a PSD curvature M. M = B'QB + R = -1
    returned a floor of -9.0."""
    with pytest.raises(ValueError, match="definite"):
        multivariate_action_floor(
            effect=np.array([[1.0]]), state_weight=np.eye(1), action_weight=np.array([[-2.0]]),
            target=np.array([1.0]), information=np.eye(1),
        )  # fmt: skip


# ------------------------------------------------------- multivariate_van_trees_certificate


def test_multivariate_van_trees_bayes_floor_is_below_the_bayes_risk() -> None:
    """multivariate_van_trees.v (B): every estimator's Bayes action error is at least
    tr(E[Psi'] G^-1 E[Psi']); checked with the exact Bayes estimator by quadrature."""
    rr, gains, target, sigma, width, n_probe, draws = (
        0.5,
        (1.0, 0.4),
        (1.0, 0.8),
        0.5,
        0.25,
        20,
        40_000,
    )
    curve = multivariate_van_trees_certificate(
        rr=rr, gains=gains, target=target, sigma=sigma, prior_width=width,
        n_probe=n_probe, draws=draws,
    )  # fmt: skip
    info = n_probe / sigma**2 + 1.0 / width**2
    post_var = 1.0 / info
    nodes, weights = hermegauss(80)
    weights = weights / weights.sum()
    floor, floor_se, risk = 0.0, 0.0, 0.0
    for centre, x in zip(gains, target, strict=True):
        prior = centre + width * nodes
        jac = -x * (rr - prior**2) / (rr + prior**2) ** 2
        mean_jac = float(weights @ jac)
        floor += mean_jac**2 / info
        floor_se += 2.0 * abs(mean_jac) * np.sqrt(weights @ (jac - mean_jac) ** 2 / draws) / info
        for post_mean, w in zip(
            centre + np.sqrt(width**2 - post_var) * nodes, weights, strict=True
        ):
            b = post_mean + np.sqrt(post_var) * nodes
            action = -b * x / (b**2 + rr)
            risk += w * float(weights @ (action - weights @ action) ** 2)
    assert curve.bayes_floor == pytest.approx(floor, abs=5.0 * floor_se)
    assert risk >= floor
    assert curve.plugin_ratio > 1.0


@SETTINGS
@given(
    st.floats(1.0, 50.0),
    st.floats(0.1, 3.0).filter(lambda rr: abs(rr - 1.0) > 1e-3),
    st.integers(0, 2**16),
)
def test_multivariate_van_trees_confounding_price_is_a_convex_combination(
    loss: float, rr: float, seed: int
) -> None:
    """multivariate_van_trees.v the_confounding_factor_is_a_convex_combination: one direction cut by
    k raises the floor by 1 + (k-1) a_w / sum(a), between 1 and k, and exactly 1 on the kernel."""
    target = (1.0, 0.8)
    curve = multivariate_van_trees_certificate(
        rr=rr, information_loss=loss, seed=seed, draws=50, n_probe=100, target=target
    )
    dense = np.random.default_rng(seed).normal(size=(2, 2))
    weight = rr * np.eye(2)
    jac = _difference_jacobian(dense, np.eye(2), weight, np.asarray(target))
    load = np.linalg.eigvalsh(jac.T @ (dense.T @ dense + weight) @ jac)
    assert curve.aligned_ratio == pytest.approx(
        1.0 + (loss - 1.0) * load[-1] / load.sum(), rel=1e-7, abs=0.0
    )
    assert 1.0 - 1e-12 <= curve.aligned_ratio <= loss * (1.0 + 1e-12)
    assert curve.orthogonal_ratio == pytest.approx(1.0, rel=1e-10, abs=0.0)
    assert curve.knife_edge_weight == pytest.approx(0.0, abs=1e-15)
    assert curve.live_channel_weight > 0.0


# ---------------------------------------------------------------- transfer and interference


def test_multivariate_transfer_regret_is_order_two_p_with_the_lq_constant() -> None:
    """composition_transfer.v regret_order_2p: an order-p effect-matrix error gives an order-2p
    regret, with the constant read off the gain-gap identity."""
    # below delta ~ 5e-3 the delta^4 regret (~1e-10) nears the rounding of the two costs (~1e-15)
    # the library subtracts, so the window stops there
    curve = multivariate_transfer_certificate(delta_lo=0.005, delta_hi=0.05, n_delta=5)
    direction = np.array([[1.0], [-0.5]]) / np.linalg.norm([1.0, -0.5])
    constant = _second_order_regret(direction)
    first = curve.regret_order1 / (constant * curve.deltas**2)
    second = curve.regret_order2 / (constant * curve.deltas**4)
    assert first == pytest.approx(np.ones(5), abs=1e-2)
    assert first[0] == pytest.approx(1.0, abs=2e-3)
    assert second == pytest.approx(np.ones(5), abs=1e-3)
    assert curve.slope_order1 == pytest.approx(2.0, abs=0.01)
    assert curve.slope_order2 == pytest.approx(4.0, abs=1e-3)


def test_multivariate_interference_orders_follow_the_least_debiased_channel() -> None:
    """multichannel_control.v half_orthogonal_order2 / full_orthogonal_order4: the spillover
    plug-in caps the LQ regret at delta^2; debiasing both channels reaches delta^4."""
    curve = multivariate_interference_certificate(delta_lo=0.002, delta_hi=0.02, n_delta=5)
    spill = _second_order_regret(np.array([[0.0], [1.0]]))
    both = _second_order_regret(np.array([[1.0], [1.0]]))
    assert curve.half_orth_regret / (spill * curve.deltas**2) == pytest.approx(np.ones(5), abs=1e-2)
    assert curve.full_orth_regret / (both * curve.deltas**4) == pytest.approx(np.ones(5), abs=2e-3)
    assert np.all(curve.full_orth_regret <= curve.half_orth_regret)


# ---------------------------------------------------------------- multichannel_control_certificate


def _multichannel_errors(delta, seeds, cluster_size=10, clusters=40):
    errors = {False: [], True: []}
    for s in range(seeds):
        rng = np.random.default_rng(6151 * s + int(1e5 * delta))
        cid = np.repeat(np.arange(clusters), cluster_size)
        n = cid.size
        z = rng.standard_normal(n)
        a = (0.5 * rng.standard_normal(clusters))[cid]
        u = z + 0.7 * rng.standard_normal(n)
        g = 0.8 * z + 0.7 * rng.standard_normal(n)
        y = u + 0.6 * g + z + a + 0.5 * rng.standard_normal(n)
        for orth in (False, True):
            errors[orth].append(_dml_two_channel(z, u, g, y, delta, orth, np.mod(cid, 2)) - 1.6)
    return {k: np.asarray(v) for k, v in errors.items()}


def test_multichannel_regrets_meet_the_population_bias() -> None:
    """multichannel_control.mac: R = C (e_d + e_s)^2 with e_s = O(delta) for the spillover plug-in
    and e_d + e_s = O(delta^2) when both channels are orthogonalised."""
    curve = multichannel_control_certificate()
    b_d, b_s, alpha_u, alpha_g, gamma, seeds = 1.0, 0.6, 1.0, 0.8, 1.0, 60
    total = b_d + b_s
    constant = _scalar_constant(total, 0.5, 1.0)
    for i in (4, 5):
        delta = float(curve.deltas[i])
        # population limits of the cross-fit Robinson fit with nuisance slopes off by delta
        e_d = -(delta**2) * (total - 1.0) / (0.49 + 2.0 * delta**2)
        e_s = (-b_s * delta * alpha_g - gamma * delta + e_d * delta * alpha_u - b_s * delta**2) / (
            0.49 + delta**2
        )
        draws = _multichannel_errors(delta, seeds)
        for orth, limit, measured in (
            (False, e_d + e_s, curve.half_orth_regret[i]),
            (True, 2.0 * e_d, curve.full_orth_regret[i]),
        ):
            assert measured == pytest.approx(constant * draws[orth].mean() ** 2, rel=1e-12, abs=0.0)
            se = draws[orth].std(ddof=1) / np.sqrt(seeds)
            assert abs(np.sqrt(measured / constant) - abs(limit)) <= 4.0 * se


def test_multichannel_full_orthogonal_regret_never_exceeds_the_half_one() -> None:
    """multichannel_control.v: orthogonalising every channel reaches the order the bottleneck caps;
    the cluster count, not the row count, sets the 1/sqrt(G) concentration."""
    curve = multichannel_control_certificate()
    assert np.all(curve.full_orth_regret < curve.half_orth_regret)
    assert curve.half_slope == pytest.approx(2.0, abs=0.3)
    assert curve.cluster_se_slope == pytest.approx(-0.5, abs=0.2)


# ---------------------------------------------------------------- transportability


def test_transportability_exact_regret_is_the_curvature_identity() -> None:
    """transportability_regret.mac: the deployment cost gap is (b_tgt^2+rr)(u_src - u_tgt)^2."""
    b_src, rr, xt, lip = 1.0, 0.5, 1.0, 1.0
    curve = transportability_regret_certificate(b_src=b_src, rr=rr, xt=xt, lip=lip)
    exact = _scalar_regret(b_src, b_src + lip * curve.w1_distances, rr, xt)
    assert curve.exact_regret == pytest.approx(exact, rel=1e-8, abs=0.0)
    assert np.all(curve.transportable_regret == 0.0)


def test_transportability_exact_regret_follows_the_maxima_expansion() -> None:
    """orthogonal_control.mac Part 2's Taylor series, read at the target effect: the exact regret is
    C(b_tgt) d^2 + c3(b_tgt) d^3 + O(d^4), so the W1 law is quadratic only to leading order."""
    rr, xt = 0.5, 1.0
    curve = transportability_regret_certificate(rr=rr, xt=xt, d_lo=1e-3, d_hi=1e-2, n_d=4)
    b = 1.0 + curve.w1_distances
    cubic = (6 * b * rr**2 - 8 * b**3 * rr + 2 * b**5) * xt**2 / (rr + b**2) ** 4
    expansion = _scalar_constant(b, rr, xt) * curve.w1_distances**2 + cubic * curve.w1_distances**3
    assert curve.exact_regret == pytest.approx(expansion, rel=1e-3, abs=0.0)
    assert curve.nontransport_regret == pytest.approx(
        _scalar_constant(1.0, rr, xt) * curve.w1_distances**2, rel=1e-14, abs=0.0
    )


def test_transportability_wdro_bound_covers_the_quadratic_residual() -> None:
    """transportability_regret.v wdro_radius_covers: C Lip^2 eps^2 >= C Lip^2 d^2 for d <= eps."""
    curve = transportability_regret_certificate(lip=1.7, wdro_radius=0.15, d_hi=0.15)
    assert np.all(curve.nontransport_regret <= curve.wdro_bound * (1.0 + 1e-12))


@pytest.mark.xfail(
    strict=True,
    reason="at d = eps = 0.15 the exact deployment regret 2.318e-3 exceeds the bound 1.667e-3",
)
def test_transportability_wdro_bound_covers_the_exact_regret() -> None:
    """transportability_regret.v covers the quadratic proxy only, not the exact cost gap."""
    curve = transportability_regret_certificate(wdro_radius=0.15, d_hi=0.15)
    assert np.all(curve.exact_regret <= curve.wdro_bound)


# ---------------------------------------------------------------- van_trees_certificate


def test_van_trees_posterior_mean_attains_the_floor() -> None:
    """van_trees.mac: the posterior mean's Bayes risk is exactly 1/(I_prior + n I_data)."""
    n_trials, sizes = 2000, (5, 20, 100)
    curve = van_trees_certificate(tau=2.0, sigma=1.5, sample_sizes=sizes, n_trials=n_trials)
    floor = 1.0 / (1.0 / 2.0**2 + np.asarray(sizes) / 1.5**2)
    assert curve.van_trees_bound == pytest.approx(floor, rel=1e-14, abs=0.0)
    # the error is N(0, floor) exactly, so its square has SD sqrt(2) * floor
    assert np.all(np.abs(curve.empirical_mse / floor - 1.0) <= 5.0 * np.sqrt(2.0 / n_trials))


@SETTINGS
@given(st.floats(0.01, 1.0), st.floats(0.01, 1.0), st.integers(1, 500))
def test_van_trees_confounding_raises_the_floor(v1: float, v2: float, n_obs: int) -> None:
    """van_trees.v confounding_raises_van_trees_floor: the floor is antitone in the information."""
    low, high = sorted((v1, v2))
    weak = van_trees_certificate(v_id=low, sample_sizes=(n_obs,), n_trials=1)
    strong = van_trees_certificate(v_id=high, sample_sizes=(n_obs,), n_trials=1)
    assert weak.confounded_bound[0] >= strong.confounded_bound[0] >= weak.van_trees_bound[0]


# ---------------------------------------------------------------- partial_id_control_certificate


@SETTINGS
@given(
    st.floats(0.01, 3.0) | st.floats(-3.0, -0.01),
    st.lists(st.floats(0.0, 4.0), min_size=1, max_size=4),
)
def test_partial_id_sign_is_identified_iff_the_width_is_below_the_estimate(
    b_hat: float, widths: list[float]
) -> None:
    """partial_id_control.v sign_robust_below_threshold / decision_reversible_above_threshold."""
    curve = partial_id_control_certificate(
        b_hat=b_hat, half_widths=tuple(widths), u_points=5, b_points=11
    )
    assert curve.sign_id_threshold == abs(b_hat)
    assert curve.sign_identified.tolist() == [w < abs(b_hat) for w in widths]


def test_partial_id_ce_worst_regret_is_the_maxima_constant_times_width_squared() -> None:
    """partial_id_control.mac (1): worst-case CE regret = C Delta^2 to leading order."""
    widths = (1e-3, 2e-3, 4e-3)
    for b_hat, rr in ((1.0, 0.5), (-0.6, 1.3), (2.0, 0.2)):
        curve = partial_id_control_certificate(b_hat=b_hat, rr=rr, half_widths=widths)
        ratio = curve.ce_worst_regret / (_scalar_constant(b_hat, rr, 1.0) * np.asarray(widths) ** 2)
        assert ratio == pytest.approx(np.ones(3), abs=2e-2)


def test_partial_id_minimax_action_does_no_worse_than_certainty_equivalence() -> None:
    """partial_id_control.mac (1): the minimax action reduces the CE action's worst case, and is the
    convex minimax over the same effect grid."""
    b_hat, rr, widths = 1.0, 0.5, (0.2, 0.5, 1.0, 1.5)
    curve = partial_id_control_certificate(b_hat=b_hat, rr=rr, half_widths=widths, u_points=4001)
    for i, delta in enumerate(widths):
        b = np.linspace(b_hat - delta, b_hat + delta, 101)

        def worst(u, b=b):
            return float(np.max((b**2 + rr) * (u - b / (b**2 + rr)) ** 2))

        best = minimize_scalar(
            worst, bounds=(-2.0, 2.0), method="bounded", options={"xatol": 1e-10}
        )
        # the library's action grid is uniform; within half a step of the minimiser the worst
        # case rises at most by its steepest slope times that half step
        step = 2.0 * (abs(b_hat / (b_hat**2 + rr)) + 0.5) / 4000
        steepest = float(np.max(2.0 * (b**2 + rr) * np.abs(best.x - b / (b**2 + rr))))
        assert curve.robust_worst_regret[i] < curve.ce_worst_regret[i]
        assert best.fun - 1e-9 <= curve.robust_worst_regret[i] <= best.fun + steepest * step


# ---------------------------------------------------------------- optimal_exploration_certificate


@SETTINGS
@given(st.floats(0.05, 2.0), st.floats(0.05, 2.0), st.integers(5, 200), st.floats(1e-3, 10.0))
def test_optimal_exploration_optimum_minimises_the_excess(
    b: float, rr: float, n: int, v: float
) -> None:
    if abs(rr - b**2) < 0.05:
        return
    """optimal_exploration.v exploration_optimum: A v + B/v >= A v* + B/v* with A v*^2 = B."""
    curve = optimal_exploration_certificate(b=b, rr=rr, n=n, n_seeds=1, n_grid=2)
    a_curv = b**2 + rr
    b_floor = _scalar_constant(b, rr, 1.0) / n
    vstar = curve.vstar_theory
    assert a_curv * vstar**2 == pytest.approx(b_floor, rel=1e-12, abs=0.0)
    assert curve.floor_theory == pytest.approx(a_curv * vstar + b_floor / vstar, rel=1e-12, abs=0.0)
    assert a_curv * v + b_floor / v >= curve.floor_theory * (1.0 - 1e-12)
    found = minimize_scalar(
        lambda w: a_curv * np.exp(w) + b_floor * np.exp(-w), bracket=(-20.0, 5.0), tol=1e-12
    )
    assert vstar == pytest.approx(np.exp(found.x), rel=1e-5, abs=0.0)


def test_optimal_exploration_estimation_cost_is_the_inverse_chi_square_mean() -> None:
    """optimal_exploration.mac: the estimation arm is C Var(b_hat) = C sigma^2/(v (n-2)) exactly --
    the docstring's B/v = C sigma^2/(n v) is its large-n form."""
    b, rr, sigma, n, seeds = 0.3, 0.2, 1.0, 30, 300
    curve = optimal_exploration_certificate(b=b, rr=rr, sigma=sigma, n=n, n_seeds=seeds)
    c = _scalar_constant(b, rr, 1.0)
    v = curve.exploration_grid
    mean = c * sigma**2 / (v * (n - 2))
    sd = c * sigma**2 / v * np.sqrt(3.0 / ((n - 2) * (n - 4)) - 1.0 / (n - 2) ** 2)
    assert np.all(np.abs(curve.estimation_cost - mean) <= 5.0 * sd / np.sqrt(seeds))
    assert curve.explore_cost == pytest.approx((b**2 + rr) * v, rel=1e-14, abs=0.0)


def test_optimal_exploration_confounding_raises_the_optimum_and_the_floor() -> None:
    """optimal_exploration.v confounding_raises_exploration / confounding_raises_floor."""
    kappa = 0.4
    curve = optimal_exploration_certificate(kappa_conf=kappa, n_seeds=1)
    assert curve.vstar_confounded_theory / curve.vstar_theory == pytest.approx(
        1.0 / np.sqrt(kappa), rel=1e-12, abs=0.0
    )
    assert curve.vstar_confounded_theory > curve.vstar_theory


# ---------------------------------------------------------------- nonlinear and strong convexity


@SETTINGS
@given(st.floats(0.1, 5.0), st.floats(0.0, 5.0), st.floats(0.1, 3.0))
def test_nonlinear_regret_bound_gap_is_the_maxima_factor(
    mu: float, kappa: float, u_max: float
) -> None:
    """nonlinear_regret.mac (ex): bound - regret = kappa u^4 (8 kappa u^2 + 3 mu)/mu >= 0 and the
    linearised estimate falls short by kappa u^4."""
    curve = nonlinear_regret_certificate(mu=mu, kappa=kappa, u_max=u_max, points=9)
    u = curve.control
    floor = minimize_scalar(lambda w: mu / 2 * w**2 + kappa * w**4, bracket=(-1.0, 1.0), tol=1e-12)
    regret = mu / 2 * u**2 + kappa * u**4 - floor.fun
    scale = mu * u_max**2 + kappa * u_max**4 + (mu * u_max + 4 * kappa * u_max**3) ** 2 / mu
    assert curve.true_regret == pytest.approx(regret, abs=1e-12 * scale)
    assert curve.pl_bound - curve.true_regret == pytest.approx(
        kappa * u**4 * (8 * kappa * u**2 + 3 * mu) / mu, abs=1e-12 * scale
    )
    assert curve.true_regret - curve.linearized_estimate == pytest.approx(
        kappa * u**4, abs=1e-12 * scale
    )
    assert np.all(curve.pl_bound >= curve.true_regret * (1.0 - 1e-15))


@SETTINGS
@given(
    st.floats(0.1, 4.0),
    st.floats(-2.0, 2.0),
    st.floats(0.0, 3.0),
    st.floats(-3.0, 3.0),
    st.floats(-4.0, 4.0),
)
def test_strong_convexity_bound_holds_on_random_scalar_costs(
    mu: float, centre: float, weight: float, slope: float, u: float
) -> None:
    """nonlinear_regret.v strong_convexity_regret_bound: 2 mu (J(u) - J*) <= J'(u)^2."""

    def cost(w):
        return mu / 2 * (w - centre) ** 2 + weight * np.logaddexp(0.0, slope * w)

    def grad(w):
        return mu * (w - centre) + weight * slope / (1.0 + np.exp(-slope * w))

    best = minimize_scalar(cost, bracket=(-10.0, 10.0), tol=1e-14)
    bound = strong_convexity_regret_bound(grad(u) ** 2, mu)
    assert cost(u) - best.fun <= bound * (1.0 + 1e-9) + 1e-12


@SETTINGS
@given(st.floats(0.1, 4.0), st.integers(0, 2**32 - 1))
def test_strong_convexity_bound_holds_on_random_vector_costs(mu: float, seed: int) -> None:
    """nonlinear_regret.v: the vector case the docstring states, with ||grad J||^2."""
    rng = np.random.default_rng(seed)
    mix, centre, u = rng.normal(size=(4, 3)), rng.normal(size=3), 2.0 * rng.normal(size=3)

    def cost(w):
        return mu / 2 * float((w - centre) @ (w - centre)) + float(np.logaddexp.reduce(mix @ w))

    def grad(w):
        soft = np.exp(mix @ w - np.logaddexp.reduce(mix @ w))
        return mu * (w - centre) + mix.T @ soft

    best = minimize(cost, centre, jac=grad, method="BFGS", options={"gtol": 1e-12})
    bound = strong_convexity_regret_bound(float(grad(u) @ grad(u)), mu)
    assert cost(u) - best.fun <= bound * (1.0 + 1e-9) + 1e-10


@SETTINGS
@given(st.floats(0.01, 10.0), st.floats(-10.0, 10.0).filter(lambda d: d == 0.0 or abs(d) > 1e-100))
def test_strong_convexity_bound_is_tight_on_a_quadratic(mu: float, d: float) -> None:
    """nonlinear_regret.v quadratic_bound_is_tight: 2 mu (mu/2 d^2) = (mu d)^2. A subnormal
    (mu d)^2 carries fewer than 53 bits, so the identity is read on normal numbers only."""
    assert strong_convexity_regret_bound((mu * d) ** 2, mu) == pytest.approx(
        mu / 2 * d**2, rel=1e-14, abs=0.0
    )


# ---------------------------------------------------------------- folds


def _weight_matrix(m, gammas, x):
    """``Q(x) = sum_{d,e} g_d g_e x^|d-e| S_d S_e`` on C_m, from circular distances."""
    i = np.arange(m)
    dist = np.minimum(np.abs(i[:, None] - i[None, :]), m - np.abs(i[:, None] - i[None, :]))
    shells = [(dist == d).astype(float) for d in range(len(gammas))]
    return sum(
        gammas[d] * gammas[e] * x ** abs(d - e) * (shells[d] @ shells[e])
        for d in range(len(gammas))
        for e in range(len(gammas))
    )


def _bisections(m):
    for chosen in itertools.combinations(range(1, m), m // 2 - 1):
        fold = np.ones(m, dtype=int)
        fold[[0, *chosen]] = 0
        yield fold


def _partitions(m, k):
    """Every balanced k-partition of m units, each once (labels canonical by first appearance)."""
    size = m // k

    def place(rest):
        if not rest:
            yield []
            return
        head, tail = rest[0], rest[1:]
        for others in itertools.combinations(tail, size - 1):
            group = (head, *others)
            remaining = [v for v in tail if v not in others]
            for more in place(remaining):
                yield [group, *more]

    for groups in place(list(range(m))):
        fold = np.empty(m, dtype=int)
        for label, group in enumerate(groups):
            fold[list(group)] = label
        yield fold


def _same_fold_mass(q, fold):
    return float(q[fold[:, None] == fold[None, :]].sum())


@SETTINGS
@given(
    st.floats(0.01, 0.99).filter(lambda x: min(abs(x - 0.7), abs(x - 0.875)) > 0.01),
)
def test_fold_partition_on_c8_follows_the_two_threshold_law(x: float) -> None:
    """fold_spectrum_law.v stripe_parity_difference_factors / parity_window_dichotomy: on C_8 at
    (1, 0.7, 0.4) parity wins between x = g1/g0 and g1/(2 g2) and width-2 stripes outside."""
    g0, g1, g2 = gammas = (1.0, 0.7, 0.4)
    m = 8
    design = optimal_fold_partition(cycle_shells(m, 2), gammas, x)
    degree = (1.0, 2.0, 2.0)
    total = m * sum(
        gammas[d] * gammas[e] * x ** abs(d - e) * degree[d] * degree[e]
        for d in range(3)
        for e in range(3)
    )
    lam_par = g0**2 + 4 * g1**2 + 4 * g2**2 - 4 * g0 * g1 * x - 8 * g1 * g2 * x + 4 * g0 * g2 * x**2
    lam_run = g0**2 + 4 * g2**2 - 4 * g0 * g2 * x**2
    assert lam_par - lam_run == pytest.approx(
        4 * (g0 * x - g1) * (2 * g2 * x - g1), rel=1e-12, abs=1e-14
    )
    assert design.objective == pytest.approx(
        (total + m * min(lam_par, lam_run)) / 2.0, rel=1e-12, abs=0.0
    )
    parity_wins = g1 / g0 < x < g1 / (2 * g2)
    sign = 1 - 2 * design.fold
    assert abs(int(sign @ np.roll(sign, 1))) == (m if parity_wins else 0)
    assert design.objective == pytest.approx(
        _same_fold_mass(_weight_matrix(m, gammas, x), design.fold), rel=1e-12, abs=0.0
    )


@SETTINGS
@given(
    st.floats(0.1, 0.95),
    st.floats(0.01, 0.99),
    st.sampled_from((8, 12)),
)
def test_fold_partition_with_one_shell_follows_the_single_threshold(
    ratio: float, x: float, m: int
) -> None:
    """fold_spectrum_law.v d1_single_threshold: with one shell, stripes below x = g1/g0 and
    alternation above, over all balanced labellings."""
    if abs(x - ratio) < 0.01:
        return
    gammas = (1.0, ratio)
    design = optimal_fold_partition(cycle_shells(m, 1), gammas, x)
    total = m * (1.0 + 4.0 * ratio * x + 4.0 * ratio**2)
    lam_par, lam_run = 1.0 + 4.0 * ratio**2 - 4.0 * ratio * x, 1.0
    assert design.objective == pytest.approx(
        (total + m * min(lam_par, lam_run)) / 2.0, rel=1e-12, abs=0.0
    )


@settings(max_examples=12, deadline=None)
@given(st.floats(0.01, 0.99), st.sampled_from((8, 10, 12, 14)), st.sampled_from((1, 2)))
def test_fold_partition_never_picks_contiguous_blocks(x: float, m: int, dmax: int) -> None:
    """fold_spectrum_law.mac: contiguous blocks are never optimal on the cycles certified."""
    gammas = (1.0, 0.7, 0.4)[: dmax + 1]
    design = optimal_fold_partition(cycle_shells(m, dmax), gammas, x)
    q = _weight_matrix(m, gammas, x)
    best = min(_same_fold_mass(q, fold) for fold in _bisections(m))
    block = np.repeat([0, 1], m // 2)
    assert design.objective == pytest.approx(best, rel=1e-12, abs=0.0)
    assert _same_fold_mass(q, block) > best * (1.0 + 1e-9)


def _ky_fan(q, k, shift):
    """``sum(u) + (m/K)(1'(Q-D)1/m + sum of the K-1 least eigenvalues on the mean-free space)``."""
    m = q.shape[0]
    centred = np.linalg.qr(np.eye(m) - 1.0 / m)[0][:, : m - 1]
    moved = q - np.diag(shift)
    low = np.linalg.eigvalsh(centred.T @ moved @ centred)[: k - 1]
    return float(shift.sum()) + (m / k) * (float(moved.sum()) / m + float(low.sum()))


@settings(max_examples=8, deadline=None)
@given(st.integers(0, 2**32 - 1), st.sampled_from(((8, 2), (6, 3), (9, 3), (6, 2))))
def test_shifted_kyfan_bound_lies_between_ky_fan_and_the_optimum(seed: int, shape) -> None:
    """fold_shifted_bound.mac STEP 1/3: plain <= shifted <= least same-fold mass, at K = 2 and 3."""
    m, k = shape
    raw = np.random.default_rng(seed).normal(size=(m, m))
    q = raw + raw.T
    shifted, plain = _shifted_kyfan_bound(q, k)
    best = min(_same_fold_mass(q, fold) for fold in _partitions(m, k))
    scale = float(np.abs(q).sum())
    assert plain == pytest.approx(_ky_fan(q, k, np.zeros(m)), rel=1e-10, abs=1e-12 * scale)
    assert plain <= shifted + 1e-12 * scale
    assert shifted <= best + 1e-9 * scale


@settings(max_examples=8, deadline=None)
@given(st.integers(0, 2**32 - 1))
def test_shifted_kyfan_bound_dominates_every_valid_shift(seed: int) -> None:
    """fold_shifted_bound.mac STEP 1: LB(u) is valid at every u, and the library maximises it."""
    rng = np.random.default_rng(seed)
    m, k = 8, 2
    q = _weight_matrix(m, (1.0, 0.7, 0.4), 0.3) + 0.3 * np.diag(rng.normal(size=m))
    q[0, 3] = q[3, 0] = q[0, 3] + 0.5
    shifted, _ = _shifted_kyfan_bound(q, k)
    best = min(_same_fold_mass(q, fold) for fold in _bisections(m))
    for _ in range(20):
        bound = _ky_fan(q, k, rng.normal(size=m))
        assert bound <= best + 1e-9 * abs(best)
        assert bound <= shifted + 2e-6 * abs(shifted)


@SETTINGS
@given(st.sampled_from((8, 12, 16)), st.floats(0.01, 0.99), st.sampled_from((2, 4)))
def test_shifted_kyfan_bound_equals_ky_fan_on_a_circulant(m: int, x: float, k: int) -> None:
    """fold_shifted_bound.mac STEP 2: on a vertex-transitive Q the best shift is uniform."""
    shifted, plain = _shifted_kyfan_bound(_weight_matrix(m, (1.0, 0.7, 0.4), x), k)
    assert shifted == pytest.approx(plain, rel=1e-9, abs=0.0)


@settings(max_examples=6, deadline=None)
@given(st.integers(0, 2**32 - 1), st.floats(-3.0, 3.0))
def test_shifted_kyfan_bound_moves_by_c_m_under_a_diagonal_shift(seed: int, c: float) -> None:
    """fold_shifted_bound.mac STEP 2: Q + cI moves every partition's mass by c m, and the bounds."""
    q = _weight_matrix(8, (1.0, 0.7, 0.4), 0.3)
    q = q + 0.3 * np.diag(np.random.default_rng(seed).normal(size=8))
    shifted, plain = _shifted_kyfan_bound(q, 2)
    moved_shifted, moved_plain = _shifted_kyfan_bound(q + c * np.eye(8), 2)
    scale = float(np.abs(q).sum()) + 8 * abs(c)
    assert moved_plain - plain == pytest.approx(8 * c, abs=1e-12 * scale)
    assert moved_shifted - shifted == pytest.approx(8 * c, abs=2e-6 * scale)


def test_fold_exactness_structural_gates_hold() -> None:
    """fold_cut_duality.v certified_gap_dominates_the_true_shortfall, min_internal_iff_max_cut and
    the two shift lemmas, on the certificate's own sweep."""
    curve = fold_exactness_certificate(sizes=(24,), decays=(0.01, 0.3), small=(10, 12), restarts=2)
    assert curve.ok
    assert np.all(curve.exact <= curve.heuristic * (1.0 + 1e-12))
    assert np.all(curve.kyfan <= curve.exact * (1.0 + 1e-12))
    assert np.all(curve.certified_gap >= curve.heuristic_shortfall - 1e-12)
    assert np.all(curve.bound_looseness > 0.0)


@SETTINGS
@given(
    st.sampled_from((10, 12)),
    st.floats(0.01, 0.99),
    st.tuples(st.floats(0.1, 1.0), st.floats(0.0, 1.0)),
)
def test_fold_exactness_dynamic_program_is_the_max_cut_optimum(m: int, x: float, tail) -> None:
    """fold_cut_duality.v min_internal_iff_max_cut: the banded DP the certificate prices returns the
    least same-fold mass, which is 1'Q1 - 2 max cut over every balanced bisection."""
    gammas = (1.0, *tail)
    dp = optimal_fold_partition(cycle_shells(m, 2), gammas, x, exhaustive_limit=0)
    assert dp.route == "banded-dp"
    q = _weight_matrix(m, gammas, x)
    cuts = [float(q[fold[:, None] != fold[None, :]].sum()) / 2.0 for fold in _bisections(m)]
    assert dp.objective == pytest.approx(float(q.sum()) - 2.0 * max(cuts), rel=1e-12, abs=0.0)


# ---------------------------------------------------------------- space-time folds and Psi


def _panel_sigma(m, gammas, phi, lag, n_times):
    """Space-time covariance from the generative law, in kron(space, time) order."""
    i = np.arange(m)
    dist = np.minimum(np.abs(i[:, None] - i[None, :]), m - np.abs(i[:, None] - i[None, :]))
    shells = [(dist == d).astype(float) for d in range(len(gammas))]
    sigma = np.zeros((m * n_times, m * n_times))
    for d, e in itertools.product(range(len(gammas)), repeat=2):
        spatial = gammas[d] * gammas[e] * (shells[d] @ shells[e])
        for t, s in itertools.product(range(n_times), repeat=2):
            sigma[t::n_times, s::n_times] += spatial * phi ** abs((t - s) - lag * (d - e))
    return sigma


@cache
def _space_time():
    return space_time_fold_certificate(n_draws=20_000, restarts=4)


def test_space_time_panel_weight_is_the_toeplitz_ratio() -> None:
    """space_time_folds.v panel_weight_exceeds_the_slice_weight: w1/w0 at p = 3 beats phi."""
    curve = _space_time()
    ph = 0.6
    w0 = 2 * ph**2 + 4 * ph + 3
    w1 = ph**3 + 2 * ph**2 + 4 * ph + 2
    assert curve.panel_weight == pytest.approx(w1 / w0, rel=1e-14, abs=0.0)
    assert curve.panel_weight - curve.slice_weight == pytest.approx(
        (ph + 2) * (1 - ph) * (1 + ph) / w0, rel=1e-12, abs=0.0
    )
    sd = np.sqrt(2.0) * curve.covariance_scale / np.sqrt(20_000)
    assert curve.covariance_error <= 5.0 * sd


def test_space_time_kronecker_ranks_follow_the_commutation_law() -> None:
    """space_time_folds.v kronecker_rank_bound / zero_delay_collapses_the_shifts: d+1 on a cycle,
    2d on a path, never 2d+1, and 1 at lag 0."""
    curve = _space_time()
    assert list(curve.commuting_rank) == [d + 1 for d in curve.dmax_grid]
    assert list(curve.generic_rank) == [2 * d for d in curve.dmax_grid]
    assert curve.separable_rank == 1
    assert curve.saturated_rank[0] < curve.commuting_rank[-1]
    assert curve.saturated_rank[1] < curve.generic_rank[-1]


def test_space_time_one_axis_optimum_is_the_enumerated_one() -> None:
    """space_time_folds.mac (6): a time-constant fold is scored on the panel covariance; its optimum
    is the enumeration over the units' bisections, and the slice design is never better."""
    curve = _space_time()
    m, n_times = 8, 4
    for index, lag in enumerate(curve.lags):
        sigma = _panel_sigma(m, (1.0, 0.7, 0.4), 0.6, lag, n_times)
        best = min(_same_fold_mass(sigma, np.repeat(fold, n_times)) for fold in _bisections(m))
        assert curve.space_only[index] == pytest.approx(best, rel=1e-12, abs=0.0)
        assert curve.slice_design[index] >= best * (1.0 - 1e-12)


def _explicit_psi(sigma, fold, g, k):
    """``N^2 tr(A^2 Sigma) / (tr(A)^2 tr(Sigma))`` with every operator built as a matrix."""
    big = np.kron(np.eye(g), sigma)
    labels = np.tile(fold, g)
    n = big.shape[0]
    r = k / (k - 1)
    same = (labels[:, None] == labels[None, :]).astype(float) * (k / n)
    a = np.eye(n) - r**2 * np.full((n, n), 1.0 / n) + (r**2 - 1) * same
    return n**2 * float(np.trace(a @ a @ big)) / (float(np.trace(a)) ** 2 * float(np.trace(big)))


@SETTINGS
@given(
    st.integers(0, 2**32 - 1), st.sampled_from(((6, 2), (8, 2), (6, 3), (9, 3))), st.integers(1, 4)
)
def test_psi_ratio_closed_form_is_the_explicit_sandwich_ratio(seed: int, shape, g: int) -> None:
    """panel_estimator_gate.mac (4): the ratio of two Psi's is the Mobius closed form."""
    cells, k = shape
    rng = np.random.default_rng(seed)
    root = rng.normal(size=(cells, cells))
    sigma = root @ root.T + 0.1 * np.eye(cells)
    first = rng.permutation(np.arange(cells) % k)
    second = rng.permutation(np.arange(cells) % k)
    expected = _explicit_psi(sigma, first, g, k) / _explicit_psi(sigma, second, g, k)
    assert _psi_ratio_closed_form(sigma, first, second, g, k) == pytest.approx(
        expected, rel=1e-10, abs=0.0
    )


@SETTINGS
@given(st.integers(0, 2**32 - 1), st.integers(1, 30))
def test_psi_ratio_washes_out_monotonically_in_the_cluster_count(seed: int, g: int) -> None:
    """panel_estimator_gate.mac (5): the better design's ratio sits below 1 and rises toward it."""
    rng = np.random.default_rng(seed)
    root = rng.normal(size=(8, 8))
    sigma = root @ root.T + 0.1 * np.eye(8)
    first, second = rng.permutation(np.arange(8) % 2), rng.permutation(np.arange(8) % 2)
    gap = _same_fold_mass(sigma, first) - _same_fold_mass(sigma, second)
    if abs(gap) < 1e-9 * float(np.abs(sigma).sum()):
        return
    if gap > 0:
        first, second = second, first
    now = _psi_ratio_closed_form(sigma, first, second, g, 2)
    later = _psi_ratio_closed_form(sigma, first, second, g + 1, 2)
    assert now < later < 1.0


def test_panel_estimator_prediction_is_the_explicit_sandwich_ratio() -> None:
    """panel_estimator_gate.mac (2)-(5): the predicted parity/block ratio is the explicit Psi ratio,
    below 1 and rising toward 1 with the number of independent clusters."""
    counts, m, p = (2, 6), 12, 12
    gate = panel_estimator_certificate(cluster_counts=counts, draws=2, bootstrap=2)
    sigma = _panel_sigma(m, (1.0, 0.7, 0.4), 0.9, 1, p)
    unit = np.repeat(np.arange(m), p)
    parity, block = unit % 2, unit * 2 // m
    expected = [
        _explicit_psi(sigma, parity, g, 2) / _explicit_psi(sigma, block, g, 2) for g in counts
    ]
    assert gate.predicted == pytest.approx(expected, rel=1e-10, abs=0.0)
    assert gate.closed_form_error <= 1e-12
    assert gate.predicted_washout
    assert np.all(gate.predicted < 1.0)
