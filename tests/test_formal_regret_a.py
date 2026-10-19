"""chc.regret (first half): each certificate that cites a Rocq proof or a Maxima derivation, tested
against the statement it cites on the float code, by a second route: exact rational arithmetic, a
brute-force or SciPy optimiser, an integrated Riccati ODE, an explicit projector matrix, or a
seeded simulation whose Monte Carlo spread is computed in the test.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.integrate import solve_ivp
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
from scipy.optimize import minimize, minimize_scalar

from chc.regret import (
    _self_consistent_mass,
    adaptive_exploration_certificate,
    asymmetric_control_improvement,
    bandit_causal_certificate,
    capped_exploration_policy,
    causal_vs_predictive_certificate,
    ce_explicit_constant_certificate,
    cluster_fold_leakage_certificate,
    clustered_lower_bound_certificate,
    composition_transfer_certificate,
    confounded_turnpike_certificate,
    confounding_robust_control,
    confounding_robust_lq_regret,
    confounding_robust_lq_regret_certificate,
    confounding_robust_lq_regret_matrix,
    conjugate_time_certificate,
    constrained_ce_regret_certificate,
    lq_regret_sensitivity,
)

FORMAL_LINKS = {
    "chc.regret.CappedExplorationPolicy": {
        "cites": ("capped_exploration_schedule.mac",),
        "tests": (
            "test_capped_block_length_beats_every_other_block",
            "test_capped_stopping_mass_is_free_of_the_cap_schedule",
            "test_capped_block_is_the_global_optimum_of_the_convex_program",
            "test_capped_schedule_is_the_box_optimum",
            "test_capped_block_search_finds_the_best_block_at_a_long_horizon",
        ),
    },
    "chc.regret._self_consistent_mass": {
        "cites": ("capped_exploration_schedule.mac",),
        "tests": (
            "test_self_consistent_mass_is_the_remaining_horizon_balance",
            "test_self_consistent_mass_at_a_constant_cap_is_the_quadratic_root",
            "test_self_consistent_mass_finds_the_balance_at_a_tight_cap",
            "test_self_consistent_mass_of_a_dead_prefix_is_the_shifted_problems",
        ),
    },
    "chc.regret.adaptive_exploration_certificate": {
        "cites": ("adaptive_exploration.mac", "adaptive_exploration.v", "van_trees.v"),
        "tests": (
            "test_adaptive_exploration_every_schedule_meets_the_sequence_floor",
            "test_adaptive_exploration_floor_numerator_is_the_decision_sensitivity",
            "test_adaptive_exploration_lower_efficiency_raises_the_leading_term",
            "test_adaptive_exploration_taper_is_sqrt_t_and_greedy_is_linear",
        ),
    },
    "chc.regret.asymmetric_control_improvement": {
        "cites": ("confounding_robust_control.v",),
        "tests": (
            "test_asymmetric_improvement_is_the_brute_force_worst_case_gap",
            "test_asymmetric_improvement_is_positive_iff_the_loss_is_asymmetric",
        ),
    },
    "chc.regret.bandit_causal_certificate": {
        "cites": ("bandit_causal.mac", "bandit_causal.v"),
        "tests": (
            "test_bandit_confounded_per_round_regret_is_the_bias_floor",
            "test_bandit_deconfounded_regret_is_sublinear_and_below_the_floor",
        ),
    },
    "chc.regret.capped_exploration_policy": {
        "cites": (
            "capped_exploration.v",
            "capped_exploration_o1.gp",
            "capped_exploration_schedule.mac",
            "capped_exploration_schedule.v",
        ),
        "tests": (
            "test_capped_policy_cost_is_the_materialised_schedule_cost",
            "test_capped_policy_never_beats_the_uncapped_floor",
            "test_capped_policy_dead_prefix_costs_exactly_m_k_over_i0",
            "test_capped_policy_binding_budget_is_spent_in_full_as_early_as_possible",
            "test_capped_policy_block_tends_to_the_floor_and_the_taper_does_not",
            "test_capped_policy_cap_free_mass_overstates_by_at_most_the_cap_term",
        ),
    },
    "chc.regret.causal_vs_predictive_certificate": {
        "cites": ("causal_mpc.mac", "causal_mpc.v"),
        "tests": (
            "test_predictive_floor_is_the_completed_square_cost_gap",
            "test_predictive_floor_is_the_cost_gap_at_the_simulated_population_bias",
            "test_predictive_regret_plateaus_and_causal_regret_vanishes",
        ),
    },
    "chc.regret.ce_explicit_constant_certificate": {
        "cites": ("c2_end_to_end.v", "ce_explicit_constants.mac", "ce_explicit_constants.v"),
        "tests": (
            "test_ce_constants_are_the_ones_the_riccati_solution_defines",
            "test_ce_controller_inside_the_ball_contracts_and_obeys_the_constant",
        ),
    },
    "chc.regret.cluster_fold_leakage_certificate": {
        "cites": ("cluster_fold_leakage.mac", "cluster_fold_leakage.v"),
        "tests": (
            "test_cluster_fold_constant_is_the_explicit_projector_trace_ratio",
            "test_cluster_fold_row_folds_annihilate_a_cluster_measurable_exposure",
            "test_cluster_fold_row_folds_do_not_bias_the_estimate",
        ),
    },
    "chc.regret.clustered_lower_bound_certificate": {
        "cites": ("clustered_van_trees.v",),
        "tests": (
            "test_clustered_regret_map_is_lower_lipschitz",
            "test_clustered_g_times_regret_stays_above_a_positive_floor",
        ),
    },
    "chc.regret.composition_transfer_certificate": {
        "cites": (
            "composition_transfer.mac",
            "composition_transfer.v",
            "order_transfer_window.mac",
        ),
        "tests": (
            "test_composition_transfer_regret_is_the_exact_cost_gap",
            "test_composition_transfer_regret_obeys_the_order_2p_bound",
            "test_composition_transfer_slope_excess_is_the_window_prediction",
        ),
    },
    "chc.regret.confounded_turnpike_certificate": {
        "cites": ("confounded_turnpike.mac", "confounded_turnpike.v"),
        "tests": (
            "test_confounded_turnpike_offset_is_the_settled_tracking_error",
            "test_confounded_turnpike_discounted_regret_is_a_bounded_geometric_sum",
        ),
    },
    "chc.regret.confounding_robust_control": {
        "cites": ("confounding_robust_control.mac", "confounding_robust_control.v"),
        "tests": (
            "test_confounding_robust_control_minimises_the_brute_force_worst_case",
            "test_confounding_robust_control_shift_has_the_sign_of_the_asymmetry",
        ),
    },
    "chc.regret.confounding_robust_lq_regret": {
        "cites": ("confounding_lq_regret.v",),
        "tests": (
            "test_confounding_lq_regret_is_the_leading_term_of_the_exact_regret",
            "test_confounding_lq_regret_floor_is_quadratic_monotone_and_below_linear",
            "test_confounding_lq_regret_third_order_excess_is_as_documented",
        ),
    },
    "chc.regret.confounding_robust_lq_regret_matrix": {
        "cites": ("composition_transfer.v",),
        "tests": (
            "test_matrix_lq_regret_order_doubles_and_is_bracketed_by_the_curvature",
            "test_matrix_lq_regret_reduces_to_the_scalar_one",
        ),
    },
    "chc.regret.conjugate_time_certificate": {
        "cites": ("conjugate_time.mac", "conjugate_time.v"),
        "tests": (
            "test_conjugate_time_is_the_escape_time_of_the_integrated_riccati",
            "test_conjugate_time_value_matches_the_integrated_riccati",
            "test_conjugate_time_pole_orders_are_one_two_four",
            "test_conjugate_time_more_authority_brings_the_escape_closer",
            "test_conjugate_time_definite_arm_is_the_integrated_riccati",
            "test_conjugate_time_refuses_a_cost_with_no_conjugate_point",
        ),
    },
    "chc.regret.constrained_ce_regret_certificate": {
        "cites": ("constrained_ce_regret.mac", "constrained_ce_regret.v"),
        "tests": (
            "test_constrained_regret_is_the_exact_cost_gap_at_the_threshold",
            "test_constrained_regret_never_exceeds_unconstrained",
            "test_constrained_threshold_is_the_lower_root",
            "test_constrained_pessimism_budget_is_the_naive_controllers_regret",
        ),
    },
}


def _u_star(b, xt, rr):
    return b * xt / (b * b + rr)


def _exact_cost_gap(u, u_opt, b, xt, rr) -> float:
    """``J(u; b) - J(u_opt; b)`` for ``J = (b u - xt)^2 + rr u^2``, in exact rationals.

    ``u_opt=None`` is the exact minimiser ``b xt/(b^2 + rr)``, not its float rounding.
    """
    u, b, xt, rr = (Fraction(v) for v in (u, b, xt, rr))
    u_opt = b * xt / (b * b + rr) if u_opt is None else Fraction(u_opt)
    return float((b * u - xt) ** 2 + rr * u * u - (b * u_opt - xt) ** 2 - rr * u_opt * u_opt)


def _exploration_constants(b=1.0, rr=0.5, xt=1.0, sigma=0.7, i0=1.0, eta=0.6):
    a = b * b + rr
    du = _u_star(complex(b, 1e-30), xt, rr).imag / 1e-30  # complex step: exact to rounding
    return a, a * du * du, eta / sigma**2, i0


def _central_l(b, rr, xt, h=1e-5) -> float:
    """Regret / error^2 at b, averaging +-h so the third-order term cancels: error O(h^2)."""
    gaps = (_exact_cost_gap(_u_star(b + s, xt, rr), None, b, xt, rr) for s in (h, -h))
    return sum(gaps) / (2.0 * h * h)


def _schedule_cost(v, a, k, c, i0) -> float:
    prefix = np.concatenate([[0.0], np.cumsum(v)[:-1]])
    return float(a * np.sum(v) + np.sum(k / (i0 + c * prefix)))


# ---------------------------------------------------------------- capped exploration


def _policy_schedule(pol, caps, budget=None) -> np.ndarray:
    """The schedule the policy reports: the greedy fill up to its last round, which is partial."""
    fill = caps if budget is None else np.diff(np.r_[0.0, np.minimum(np.cumsum(caps), budget)])
    v = np.zeros(caps.size)
    v[: pol.block_rounds] = fill[: pol.block_rounds]
    if pol.block_rounds:
        assert 0.0 < pol.last_round <= fill[pol.block_rounds - 1]
        v[pol.block_rounds - 1] = pol.last_round
    return v


@pytest.mark.parametrize(
    ("horizon", "cap"), [(50, 0.3), (400, 0.05), (3000, 0.01), (10, 3.0), (30, 1.0)]
)
def test_capped_block_length_beats_every_other_block(horizon: int, cap: float) -> None:
    """capped_exploration_schedule.mac: the policy is no worse than the best saturate-then-stop
    block, ends within one round of it, and costs what its schedule costs. At T=10, cap=3 the best
    block is no exploration at all (0.741), which a search from one round (4.717) missed."""
    a, k, c, i0 = _exploration_constants()
    pol = capped_exploration_policy(horizon=horizon, cap=cap)
    costs = [
        _schedule_cost(np.r_[np.full(n, cap), np.zeros(horizon - n)], a, k, c, i0)
        for n in range(horizon + 1)
    ]
    best = int(np.argmin(costs))
    assert pol.block_rounds in (best, best + 1)
    assert pol.cost <= min(costs) * (1.0 + 1e-12)
    caps = np.full(horizon, cap)
    assert pol.cost == pytest.approx(
        _schedule_cost(_policy_schedule(pol, caps), a, k, c, i0), rel=1e-12, abs=0.0
    )


@settings(max_examples=30, deadline=None)
@given(
    horizon=st.integers(20, 3000),
    seed=st.integers(0, 2**32 - 1),
    lo=st.floats(0.001, 0.05),
    hi=st.floats(0.06, 0.3),
)
def test_capped_stopping_mass_is_free_of_the_cap_schedule(
    horizon: int, seed: int, lo: float, hi: float
) -> None:
    """capped_exploration_schedule.mac STEP 2b: the stopping mass solves a cap-free balance."""
    caps = np.random.default_rng(seed).uniform(lo, hi, horizon)
    pol = capped_exploration_policy(horizon=horizon, cap=caps)
    _, balanced, _, _ = _balance_crossing(caps, horizon)
    assert abs(pol.exploration_mass - balanced) <= float(caps.max())


@pytest.mark.parametrize(("horizon", "cap"), [(10_000, 0.01), (1_000_000, 0.002), (100_000, 0.03)])
def test_capped_block_search_finds_the_best_block_at_a_long_horizon(
    horizon: int, cap: float
) -> None:
    """capped_exploration.v: the block cost is convex in its length, so the best whole block is
    the minimum over every length, here by materialised harmonic sums. A bracket with a 64-round
    sweep picked 90932 rounds against 95108 at T = 1e6, cap = 0.002, 0.60 dearer."""
    a, k, c, i0 = _exploration_constants()
    pol = capped_exploration_policy(horizon=horizon, cap=cap)
    rounds = np.arange(horizon + 1, dtype=np.float64)
    info = i0 + c * cap * rounds
    inside = np.r_[0.0, np.cumsum(k / info[:-1])]
    costs = a * cap * rounds + inside + k * (horizon - rounds) / info
    best = int(np.argmin(costs))
    assert pol.block_rounds in (best, best + 1)
    assert pol.cost <= float(costs[best]) * (1.0 + 1e-12)


@pytest.mark.parametrize(("horizon", "cap"), [(50, 0.3), (30, 1.0), (400, 0.05), (10, 3.0)])
def test_capped_block_is_the_global_optimum_of_the_convex_program(horizon: int, cap: float) -> None:
    """capped_exploration_schedule.mac: a saturated prefix and one partial round is the global
    optimum over the box. T=50, cap=0.3: the whole block [0.3, 0.3] cost 3.0779209 against the
    optimum [0.3, 0.2747] at 3.0772412."""
    a, k, c, i0 = _exploration_constants()
    pol = capped_exploration_policy(horizon=horizon, cap=cap)
    best = min(
        minimize_scalar(
            lambda x, n=n: _schedule_cost(
                np.r_[np.full(n, cap), [x], np.zeros(horizon - n - 1)], a, k, c, i0
            ),
            bounds=(0.0, cap),
            method="bounded",
            options={"xatol": 1e-12},
        ).fun
        for n in range(horizon)
    )
    assert pol.cost <= best + 1e-12
    assert pol.cost == pytest.approx(best, rel=1e-10, abs=0.0)


@settings(max_examples=12, deadline=None)
@given(
    seed=st.integers(0, 2**32 - 1),
    idle=st.floats(0.0, 0.6),
    budget=st.one_of(st.none(), st.floats(0.05, 2.0)),
)
def test_capped_schedule_is_the_box_optimum(seed: int, idle: float, budget: float | None) -> None:
    """capped_exploration.v earlier_is_strictly_cheaper: with caps that vary and rounds with no
    authority, the policy's schedule is the box optimum, by L-BFGS-B over every round."""
    a, k, c, i0 = _exploration_constants()
    rng = np.random.default_rng(seed)
    horizon = 40
    caps = rng.uniform(0.0, 0.4, horizon) * (rng.uniform(size=horizon) >= idle)
    caps[0] = max(caps[0], 1e-3)  # one round with authority, which the policy requires
    pol = capped_exploration_policy(horizon=horizon, cap=caps, budget=budget)
    schedule = _policy_schedule(pol, caps, budget)
    assert pol.cost == pytest.approx(_schedule_cost(schedule, a, k, c, i0), rel=1e-12, abs=0.0)
    total = np.inf if budget is None else budget
    # the budget is a linear constraint; a penalty-free route is to minimise over the fill's mass
    masses = np.linspace(0.0, min(float(caps.sum()), total), 2001)
    fills = [np.diff(np.r_[0.0, np.minimum(np.cumsum(caps), m)]) for m in masses]
    grid = min(_schedule_cost(v, a, k, c, i0) for v in fills)
    assert pol.cost <= grid + 1e-12
    if budget is None:
        free = minimize(
            lambda v: _schedule_cost(v, a, k, c, i0),
            caps / 2.0,
            method="L-BFGS-B",
            bounds=list(zip(np.zeros(horizon), caps, strict=True)),
            options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10_000},
        )
        assert pol.cost <= free.fun + 1e-10
        assert pol.cost == pytest.approx(free.fun, rel=1e-7, abs=0.0)


def _balance_crossing(caps, horizon):
    """Smallest round n whose balance mass S(n) the greedy fill reaches by round n, by a full scan.

    ``S(n) = sqrt(K (T - n)/(A c)) - I0/c`` falls with n and the round reaching it rises with S, so
    ``reach(S(n)) - n`` is strictly decreasing: this n is the fixed point when one exists.
    """
    a, k, c, i0 = _exploration_constants()
    prefix = np.r_[np.cumsum(caps), np.inf]
    for n in range(1, horizon + 1):
        mass = max(math.sqrt(k * (horizon - n) / (a * c)) - i0 / c, 0.0)
        reached = int(np.flatnonzero(prefix >= mass)[0]) + 1
        if reached <= n:
            previous = max(math.sqrt(k * (horizon - n + 1) / (a * c)) - i0 / c, 0.0)
            return n, mass, previous, reached == n
    raise AssertionError("the balance never crosses")


def _scm(caps, horizon):
    a, k, c, i0 = _exploration_constants()
    return _self_consistent_mass(
        caps, horizon=horizon, curvature=a, numerator=k, info_rate=c, prior_info=i0
    )


@settings(max_examples=40, deadline=None)
@given(horizon=st.integers(5, 2500), seed=st.integers(0, 2**32 - 1), lo=st.floats(0.05, 0.3))
def test_self_consistent_mass_is_the_remaining_horizon_balance(
    horizon: int, seed: int, lo: float
) -> None:
    """capped_exploration_schedule.mac STEP 2: A (I0 + c S)^2 = K c (T - n), n reaching S."""
    caps = np.random.default_rng(seed).uniform(lo, 0.3, horizon)
    mass = _scm(caps, horizon)
    _, balanced, previous, exact = _balance_crossing(caps, horizon)
    if exact:
        assert mass == pytest.approx(balanced, rel=1e-12, abs=1e-15)
    else:
        # no round balances exactly: the balance straddles the boundary between n - 1 and n
        assert any(mass == pytest.approx(s, rel=1e-12, abs=1e-15) for s in (balanced, previous))


def test_self_consistent_mass_finds_the_balance_at_a_tight_cap() -> None:
    """capped_exploration_schedule.mac STEP 2: the stopping mass is the balance's fixed point.
    T=41, cap=0.01: n=17 balances with S=0.16715; iterating the map two-cycled and read 0.0."""
    caps = np.full(41, 0.01)
    n, balanced, _, exact = _balance_crossing(caps, 41)
    assert (n, exact) == (17, True)
    assert _scm(caps, 41) == pytest.approx(balanced, rel=1e-12, abs=0.0)


def test_self_consistent_mass_of_a_dead_prefix_is_the_shifted_problems() -> None:
    """capped_exploration_schedule.v an_idle_prefix_is_a_constant_plus_the_shifted_problem.
    Iterating the map from n=0 ran 0 -> 363 -> 1 -> 363 and read 0.0, not the balance 0.4848."""
    caps = np.r_[np.zeros(332), np.full(47, 0.1)]
    n, balanced, _, exact = _balance_crossing(caps, 379)
    assert (n, exact) == (337, True)
    assert balanced == pytest.approx(_scm(np.full(47, 0.1), 47), rel=1e-12, abs=0.0)
    assert _scm(caps, 379) == pytest.approx(balanced, rel=1e-12, abs=0.0)


@settings(max_examples=25, deadline=None)
@given(horizon=st.integers(2000, 200_000), cap=st.floats(0.005, 0.5))
def test_self_consistent_mass_at_a_constant_cap_is_the_quadratic_root(
    horizon: int, cap: float
) -> None:
    """capped_exploration_schedule.mac STEP 7: A w^2 + (K/cap) w = K c T + K I0/cap."""
    a, k, c, i0 = _exploration_constants()
    mass = _self_consistent_mass(
        np.full(horizon, cap), horizon=horizon, curvature=a, numerator=k, info_rate=c, prior_info=i0
    )
    w = (-(k / cap) + math.sqrt((k / cap) ** 2 + 4.0 * a * (k * c * horizon + k * i0 / cap))) / (
        2.0 * a
    )
    # the fixed point sits on an integer round, so it may miss the continuous root by one round's
    # worth of horizon: dS/dn = -K/(2 A w)
    assert abs(mass - (w - i0) / c) <= k / (2.0 * a * w) + 1e-9


@settings(max_examples=30, deadline=None)
@given(
    horizon=st.integers(10, 4000),
    cap=st.floats(0.002, 1.0),
    b=st.floats(0.2, 3.0),
    rr=st.floats(0.05, 3.0),
    sigma=st.floats(0.2, 2.0),
    eta=st.floats(0.05, 1.0),
)
def test_capped_policy_cost_is_the_materialised_schedule_cost(
    horizon: int, cap: float, b: float, rr: float, sigma: float, eta: float
) -> None:
    """capped_exploration.v: the digamma block cost is F(v) of the saturate-then-stop schedule."""
    a, k, c, i0 = _exploration_constants(b=b, rr=rr, sigma=sigma, eta=eta)
    pol = capped_exploration_policy(horizon=horizon, cap=cap, b=b, rr=rr, sigma=sigma, eta=eta)
    v = _policy_schedule(pol, np.full(horizon, cap))
    assert pol.cost == pytest.approx(_schedule_cost(v, a, k, c, i0), rel=1e-9, abs=0.0)


@settings(max_examples=40, deadline=None)
@given(
    horizon=st.integers(1, 100_000),
    cap=st.floats(0.001, 2.0),
    b=st.floats(0.2, 3.0),
    rr=st.floats(0.05, 3.0),
    sigma=st.floats(0.2, 2.0),
    eta=st.floats(0.05, 1.0),
    i0=st.floats(0.1, 5.0),
)
def test_capped_policy_never_beats_the_uncapped_floor(
    horizon: int, cap: float, b: float, rr: float, sigma: float, eta: float, i0: float
) -> None:
    """capped_exploration.v block_cost_floor: (A/c) m + K T/m >= 2 sqrt(A K T/c), less A I0/c."""
    pol = capped_exploration_policy(
        horizon=horizon, cap=cap, b=b, rr=rr, sigma=sigma, eta=eta, i0=i0
    )
    assert pol.cost >= pol.uncapped_floor * (1.0 - 1e-12) - 1e-12


@pytest.mark.parametrize("dead", [1, 7, 120])
def test_capped_policy_dead_prefix_costs_exactly_m_k_over_i0(dead: int) -> None:
    """capped_exploration_schedule.v: an idle prefix of m rounds adds m K/I0, then shifts."""
    _, k, _, i0 = _exploration_constants()
    horizon, cap = 900, 0.04
    shifted = capped_exploration_policy(horizon=horizon - dead, cap=cap)
    padded = capped_exploration_policy(
        horizon=horizon, cap=np.r_[np.zeros(dead), np.full(horizon - dead, cap)]
    )
    assert padded.cost == pytest.approx(shifted.cost + dead * k / i0, rel=1e-12, abs=0.0)
    assert padded.block_rounds == shifted.block_rounds + dead


@settings(max_examples=30, deadline=None)
@given(horizon=st.integers(50, 3000), cap=st.floats(0.01, 0.3), share=st.floats(0.05, 0.9))
def test_capped_policy_binding_budget_is_spent_in_full_as_early_as_possible(
    horizon: int, cap: float, share: float
) -> None:
    """capped_exploration_schedule.v below_the_root_more_mass_still_pays: a budget binds."""
    free = capped_exploration_policy(horizon=horizon, cap=cap)
    budget = share * free.exploration_mass
    pol = capped_exploration_policy(horizon=horizon, cap=cap, budget=budget)
    assert pol.budget_binds
    assert pol.exploration_mass == pytest.approx(budget, rel=1e-12, abs=0.0)
    assert pol.block_rounds == math.ceil(budget / cap - 1e-9)


def test_capped_policy_block_tends_to_the_floor_and_the_taper_does_not() -> None:
    """capped_exploration.v capped_leading_term_is_the_uncapped_floor: block/floor -> 1."""
    ratios, tapers = [], []
    for horizon in (10**4, 10**5, 10**6, 10**7):
        pol = capped_exploration_policy(horizon=horizon, cap=0.03)
        ratios.append(pol.cost / pol.uncapped_floor)
        tapers.append(pol.taper_cost / pol.uncapped_floor)
    assert all(r1 > r2 > 1.0 for r1, r2 in itertools.pairwise(ratios))
    assert ratios[-1] < 1.01
    assert min(tapers) > 1.3
    at_1e5 = capped_exploration_policy(horizon=10**5, cap=0.03)
    assert at_1e5.cost == pytest.approx(196.11, abs=0.005)
    assert at_1e5.taper_cost == pytest.approx(266.81, abs=0.005)


@pytest.mark.parametrize(
    ("horizon", "documented"), [(10**4, 1.834), (10**5, 1.959), (10**6, 1.998), (10**7, 2.011)]
)
def test_capped_policy_cap_free_mass_overstates_by_at_most_the_cap_term(
    horizon: int, documented: float
) -> None:
    """capped_exploration_schedule.v (H): 0 <= S56 - S* <= K/(2 A c cap), the gap rising to it."""
    a, k, c, i0 = _exploration_constants()
    cap = 0.01
    pol = capped_exploration_policy(horizon=horizon, cap=cap)
    s56 = math.sqrt(k * horizon / (a * c)) - i0 / c
    gap = s56 - pol.predicted_mass
    ceiling = k / (2.0 * a * c * cap)
    assert 0.0 <= gap <= ceiling
    assert gap == pytest.approx(documented, abs=2e-3)
    tail = (
        (k + 4.0 * a * i0 * cap)
        * math.sqrt(k / (a * c))
        / (8.0 * a * c * cap**2 * math.sqrt(horizon))
    )
    assert ceiling - gap == pytest.approx(tail, rel=0.05, abs=0.0)


# ---------------------------------------------------------------- adaptive exploration


@settings(max_examples=40, deadline=None)
@given(
    b=st.floats(0.2, 3.0),
    rr=st.floats(0.05, 3.0),
    sigma=st.floats(0.1, 2.0),
    m0=st.floats(0.1, 5.0),
    eta=st.floats(0.05, 1.0),
    static_v=st.floats(0.0, 1.0),
)
def test_adaptive_exploration_every_schedule_meets_the_sequence_floor(
    b: float, rr: float, sigma: float, m0: float, eta: float, static_v: float
) -> None:
    """adaptive_exploration.v reduced_objective_lower_bound: R >= 2 sqrt(A K T/eta) - A m0/eta."""
    cert = adaptive_exploration_certificate(
        b=b, rr=rr, sigma=sigma, m0=m0, eta=eta, static_v=static_v, horizons=(10, 100, 1000)
    )
    a = b * b + rr
    _, k, _, _ = _exploration_constants(b=b, rr=rr, sigma=sigma, eta=eta)
    floor = 2.0 * np.sqrt(a * k * sigma**2 * cert.horizons / eta) - a * m0 / eta
    assert cert.lower_bound == pytest.approx(floor, rel=1e-12, abs=1e-12 * a * m0 / eta)
    for regret in (cert.adaptive_regret, cert.greedy_regret, cert.static_regret):
        assert np.all(regret >= floor - 1e-12 * np.abs(floor) - 1e-12)


@settings(max_examples=30, deadline=None)
@given(
    b=st.floats(0.2, 3.0),
    rr=st.floats(0.05, 3.0),
    sigma=st.floats(0.1, 2.0),
    eta=st.floats(0.05, 1.0),
)
def test_adaptive_exploration_floor_numerator_is_the_decision_sensitivity(
    b: float, rr: float, sigma: float, eta: float
) -> None:
    """adaptive_exploration.mac: K = A (du*/db)^2 sigma^2; the leading term is 2 sqrt(A K T/eta)."""
    cert = adaptive_exploration_certificate(b=b, rr=rr, sigma=sigma, eta=eta, horizons=(10, 1000))
    a, k, _, _ = _exploration_constants(b=b, rr=rr, sigma=1.0, eta=1.0)
    m0 = 1.0  # greedy never explores, so every round pays the prior floor K/m0
    expected = 2.0 * np.sqrt(a * k * sigma**2 * cert.horizons / eta)
    leading = cert.lower_bound + a * m0 / eta
    assert leading == pytest.approx(expected, rel=1e-12, abs=1e-12 * a * m0 / eta)
    greedy = cert.horizons * k * sigma**2 / m0
    assert cert.greedy_regret == pytest.approx(greedy, rel=1e-12, abs=0.0)


@settings(max_examples=30, deadline=None)
@given(eta1=st.floats(0.02, 1.0), eta2=st.floats(0.02, 1.0))
def test_adaptive_exploration_lower_efficiency_raises_the_leading_term(
    eta1: float, eta2: float
) -> None:
    """adaptive_exploration.v lower_efficiency_raises_sequence_floor: antitone in eta."""
    lo, hi = sorted((eta1, eta2))
    a = _exploration_constants()[0]  # the leading term drops the floor's -A m0/eta, m0 = 1
    b_lo = adaptive_exploration_certificate(eta=lo, horizons=(100, 1000)).lower_bound + a / lo
    b_hi = adaptive_exploration_certificate(eta=hi, horizons=(100, 1000)).lower_bound + a / hi
    assert np.all(b_hi <= b_lo)
    assert b_lo / b_hi == pytest.approx(np.full(2, math.sqrt(hi / lo)), rel=1e-12, abs=0.0)


def test_adaptive_exploration_taper_is_sqrt_t_and_greedy_is_linear() -> None:
    """adaptive_exploration.mac: the 1/sqrt(t) taper attains Theta(sqrt T); greedy is Theta(T)."""
    cert = adaptive_exploration_certificate(horizons=(10**4, 10**5, 10**6))
    assert cert.greedy_slope == pytest.approx(1.0, abs=1e-12)
    assert cert.adaptive_slope == pytest.approx(0.5, abs=0.03)
    ratio = cert.adaptive_regret / cert.lower_bound
    assert np.all((ratio >= 1.0) & (ratio < 1.5))
    assert float(np.max(ratio) / np.min(ratio)) < 1.1


# ---------------------------------------------------------------- asymmetric robust control


def _worst_case_loss(u, bhat, d, target, a, b) -> float:
    effects = np.linspace(bhat - d, bhat + d, 2001)
    y = effects * u
    return float(np.max(a * np.maximum(y - target, 0.0) + b * np.maximum(target - y, 0.0)))


def _minimax(bhat, d, target, a, b) -> tuple[float, float]:
    """Golden-section search; the worst-case loss is convex in u between the endpoint actions."""
    lo, hi = target / (bhat + d), target / (bhat - d)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(120):
        x1, x2 = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
        if _worst_case_loss(x1, bhat, d, target, a, b) < _worst_case_loss(
            x2, bhat, d, target, a, b
        ):
            hi = x2
        else:
            lo = x1
    u = 0.5 * (lo + hi)
    return u, _worst_case_loss(u, bhat, d, target, a, b)


_asym = {
    "bhat": st.floats(0.3, 3.0),
    "share": st.floats(0.02, 0.95),
    "target": st.floats(0.1, 5.0),
    "a": st.floats(0.1, 10.0),
    "b": st.floats(0.1, 10.0),
}


@settings(max_examples=40, deadline=None)
@given(**_asym)
def test_confounding_robust_control_minimises_the_brute_force_worst_case(
    bhat: float, share: float, target: float, a: float, b: float
) -> None:
    """confounding_robust_control.mac (1): u_rob balances the two worst tails: the minimax u."""
    d = share * bhat
    u = confounding_robust_control(bhat, d, target, a, b)
    u_num, w_num = _minimax(bhat, d, target, a, b)
    assert u == pytest.approx(u_num, rel=1e-6, abs=0.0)
    assert _worst_case_loss(u, bhat, d, target, a, b) <= w_num * (1.0 + 1e-9) + 1e-15


@settings(max_examples=40, deadline=None)
@given(**_asym)
def test_confounding_robust_control_shift_has_the_sign_of_the_asymmetry(
    bhat: float, share: float, target: float, a: float, b: float
) -> None:
    """confounding_robust_control.v robust_gain_conservative / robust_gain_aggressive."""
    d = share * bhat
    u = confounding_robust_control(bhat, d, target, a, b)
    u_ce = target / bhat
    assert (u - u_ce) * (a - b) <= 1e-12 * u_ce * abs(a - b)
    assert confounding_robust_control(bhat, d, target, a, a) == pytest.approx(
        u_ce, rel=1e-14, abs=0.0
    )


@settings(max_examples=40, deadline=None)
@given(**_asym)
def test_asymmetric_improvement_is_the_brute_force_worst_case_gap(
    bhat: float, share: float, target: float, a: float, b: float
) -> None:
    """confounding_robust_control.v rob_gap / rob_gap_under: W_CE - W_rob on each branch."""
    d = share * bhat
    w_ce = _worst_case_loss(target / bhat, bhat, d, target, a, b)
    _, w_rob = _minimax(bhat, d, target, a, b)
    gap = asymmetric_control_improvement(bhat, d, target, a, b)
    assert gap == pytest.approx(w_ce - w_rob, rel=1e-6, abs=1e-9 * w_ce)


@settings(max_examples=40, deadline=None)
@given(**_asym)
def test_asymmetric_improvement_is_positive_iff_the_loss_is_asymmetric(
    bhat: float, share: float, target: float, a: float, b: float
) -> None:
    """confounding_robust_control.v rob_gap_pos / rob_gap_symmetric_zero: > 0 iff a != b."""
    d = share * bhat
    assert asymmetric_control_improvement(bhat, d, target, a, a) == 0.0
    gap = asymmetric_control_improvement(bhat, d, target, a, b)
    assert gap > 0.0 if a != b else gap == 0.0


# ---------------------------------------------------------------- bandit / causal vs predictive


def test_bandit_confounded_per_round_regret_is_the_bias_floor() -> None:
    """bandit_causal.v per_round_conf: the confounded per-round regret tends to C beta^2."""
    alpha, gamma, b, rr, xt = 1.0, 1.0, 1.0, 0.5, 1.0
    cert = bandit_causal_certificate(n_rounds=2000, n_seeds=6)
    half, full = cert.rounds[-2], cert.rounds[-1]
    per_round = (cert.confounded_regret[-1] - cert.confounded_regret[-2]) / (full - half)
    coeff = _central_l(b, rr, xt)
    beta = gamma * alpha / (alpha**2 + 1.0)
    # sampling error of the confounded slope adds ~Var(u-resid)/(t Var u) ~ 1/t: below 1% here
    assert per_round == pytest.approx(coeff * beta**2, rel=0.03, abs=0.0)
    assert cert.confounded_doubling == pytest.approx(2.0, abs=0.15)


def test_bandit_deconfounded_regret_is_sublinear_and_below_the_floor() -> None:
    """bandit_causal.v deconfounded_beats_confounded: C sigma^2/t < C beta^2 for large t."""
    cert = bandit_causal_certificate(n_rounds=2000, n_seeds=6)
    half, full = cert.rounds[-2], cert.rounds[-1]
    d_inc = cert.deconfounded_regret[-1] - cert.deconfounded_regret[-2]
    c_inc = cert.confounded_regret[-1] - cert.confounded_regret[-2]
    assert full == 2 * half
    assert d_inc < 0.05 * c_inc
    assert cert.deconfounded_doubling < 1.3 < 1.85 < cert.confounded_doubling
    ratio = cert.confounded_regret / cert.deconfounded_regret
    assert ratio[-1] > 1.3 * ratio[-2]


@settings(max_examples=30, deadline=None)
@given(
    b=st.floats(0.2, 3.0),
    alpha=st.floats(-2.0, 2.0),
    gamma=st.floats(-2.0, 2.0),
    target=st.floats(0.1, 3.0),
    effort=st.floats(0.05, 3.0),
)
def test_predictive_floor_is_the_completed_square_cost_gap(
    b: float, alpha: float, gamma: float, target: float, effort: float
) -> None:
    """causal_mpc.v cost_completing_square / predictive_control_positive_regret."""
    cert = causal_vs_predictive_certificate(
        b_true=b,
        alpha=alpha,
        gamma=gamma,
        target=target,
        effort=effort,
        sample_sizes=(20,),
        n_seeds=1,
    )
    beta = gamma * alpha / (alpha**2 + 1.0)
    u_hat, u_opt = _u_star(b + beta, target, effort), _u_star(b, target, effort)
    exact = _exact_cost_gap(u_hat, None, b, target, effort)
    square = (b * b + effort) * (u_hat - u_opt) ** 2
    assert cert.predictive_floor == pytest.approx(exact, rel=1e-6, abs=1e-15)
    assert square == pytest.approx(exact, rel=1e-6, abs=1e-15)
    if u_hat != u_opt and abs(u_hat - u_opt) > 1e-6:
        assert cert.predictive_floor > 0.0


def test_predictive_floor_is_the_cost_gap_at_the_simulated_population_bias() -> None:
    """causal_mpc.mac Part 1: the OVB is a population constant; the floor is the gap it induces."""
    rng = np.random.default_rng(20261019)
    n = 2_000_000
    z = rng.standard_normal(n)
    u = z + rng.standard_normal(n)
    y = u + z + 0.3 * rng.standard_normal(n)
    b_obs = float(np.dot(u, y) / np.dot(u, u))
    # Var of the OLS slope here is ~ Var(z - beta u)/(n Var u) = 0.59/(2 n)
    se = math.sqrt(0.59 / (2.0 * n))
    assert abs(b_obs - 1.5) < 6.0 * se
    cert = causal_vs_predictive_certificate(sample_sizes=(20,), n_seeds=1)

    def floor(effect: float) -> float:
        return _exact_cost_gap(_u_star(effect, 1.0, 0.5), None, 1.0, 1.0, 0.5)

    slope = abs(floor(b_obs + se) - floor(b_obs - se)) / (2.0 * se)
    assert cert.predictive_floor == pytest.approx(floor(b_obs), abs=6.0 * se * slope)


def test_predictive_regret_plateaus_and_causal_regret_vanishes() -> None:
    """causal_mpc.mac Part 3: the predictive regret stays at the floor; the causal one -> 0."""
    cert = causal_vs_predictive_certificate(sample_sizes=(1000, 16000, 64000), n_seeds=6)
    assert cert.predictive_regret[-1] == pytest.approx(cert.predictive_floor, rel=0.05, abs=0.0)
    assert np.all(cert.predictive_regret > 0.8 * cert.predictive_floor)
    assert cert.causal_regret[-1] < 1e-3 * cert.predictive_floor
    assert cert.causal_regret[-1] < cert.causal_regret[0]


# ---------------------------------------------------------------- explicit CE constants

_PLANTS = [
    (np.array([[1.0, 0.1], [0.0, 0.95]]), np.array([[0.5], [1.0]]), np.array([[0.5]])),
    (np.array([[0.9, 0.2], [0.0, 1.05]]), np.array([[0.0], [1.0]]), np.array([[2.0]])),
]


def _riccati(a, b, q, r):
    p = solve_discrete_are(a, b, q, r)
    k = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
    return k, p


@pytest.mark.parametrize("plant", range(len(_PLANTS)))
def test_ce_constants_are_the_ones_the_riccati_solution_defines(plant: int) -> None:
    """ce_explicit_constants.mac: eta, theta, kappa_P, ||R_K||, beta_B, the gain-gap identity."""
    a, b, r = _PLANTS[plant]
    q = np.eye(2)
    cert = ce_explicit_constant_certificate(
        a_mat=a, b_mat=b, q_mat=q, r_mat=r, multiples=(0.5,), n_dirs=4
    )
    k, p = _riccati(a, b, q, r)
    ev = np.linalg.eigvalsh(p)
    eta = float(np.min(np.linalg.eigvalsh(q + k.T @ r @ k))) / ev[-1]
    assert cert.eta == pytest.approx(eta, rel=1e-9, abs=0.0)
    assert cert.theta == pytest.approx(1.0 - math.sqrt(1.0 - eta), rel=1e-9, abs=0.0)
    assert cert.kappa_p == pytest.approx(ev[-1] / ev[0], rel=1e-9, abs=0.0)
    assert cert.rk_norm == pytest.approx(np.linalg.norm(r + b.T @ p @ b, 2), rel=1e-9, abs=0.0)
    w, v = np.linalg.eigh(p)
    p_sqrt = v @ np.diag(np.sqrt(w)) @ v.T
    assert cert.beta_b == pytest.approx(
        np.linalg.norm(p_sqrt @ b, 2) / math.sqrt(ev[0]), rel=1e-9, abs=0.0
    )
    assert cert.identity_residual < 1e-9 * float(np.max(np.abs(p)))
    assert cert.rho == pytest.approx(
        cert.theta / (2.0 * cert.beta_b * max(cert.l_k_local, cert.l_k_ball)), rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("plant", range(len(_PLANTS)))
def test_ce_controller_inside_the_ball_contracts_and_obeys_the_constant(plant: int) -> None:
    """ce_explicit_constants.v perturbed_loop_contraction: in rho, J(Khat) - J* <= C ||dB||^2."""
    a, b, r = _PLANTS[plant]
    q, x0 = np.eye(2), np.array([1.0, 0.5])
    cert = ce_explicit_constant_certificate(
        a_mat=a, b_mat=b, q_mat=q, r_mat=r, multiples=(0.5,), n_dirs=4
    )
    k_star, p = _riccati(a, b, q, r)
    w, v = np.linalg.eigh(p)
    p_half, p_half_inv = v @ np.diag(np.sqrt(w)) @ v.T, v @ np.diag(1.0 / np.sqrt(w)) @ v.T

    def cost(k):
        a_cl = a - b @ k
        assert np.max(np.abs(np.linalg.eigvals(a_cl))) < 1.0
        return float(x0 @ solve_discrete_lyapunov(a_cl.T, q + k.T @ r @ k) @ x0)

    j_star = cost(k_star)
    rng = np.random.default_rng(7 + plant)
    for _ in range(60):
        db = rng.normal(size=b.shape)
        db *= cert.rho * rng.uniform(0.05, 1.0) / np.linalg.norm(db)
        k_hat, _ = _riccati(a, b + db, q, r)
        a_cl = a - b @ k_hat
        contraction = np.linalg.norm(p_half @ a_cl @ p_half_inv, 2) ** 2
        assert contraction <= cert.lyapunov_ceiling
        assert cost(k_hat) - j_star <= cert.c_const * float(np.sum(db**2))


# ---------------------------------------------------------------- cluster fold leakage


def _row_fold_constant(m: int) -> float:
    s = np.zeros((m, m))
    fold = np.arange(m) % 2
    for i in range(m):
        others = np.flatnonzero(fold != fold[i])
        s[i, others] = 1.0 / others.size
    mm = (np.eye(m) - s).T @ (np.eye(m) - s)
    return m * float(np.trace(mm @ mm)) / float(np.trace(mm)) ** 2


@pytest.mark.parametrize("m", [2, 4, 8, 10, 40])
def test_cluster_fold_constant_is_the_explicit_projector_trace_ratio(m: int) -> None:
    """cluster_fold_leakage.v c_fold_k2: c(m,2) = m tr((M'M)^2)/tr(M'M)^2 = m(m+14)/(m+2)^2."""
    cert = cluster_fold_leakage_certificate(
        n_clusters=6, cluster_size=m, icc_grid=(0.0,), lin_cluster_grid=(4,), n_seeds=3
    )
    assert cert.c_fold_predicted == pytest.approx(_row_fold_constant(m), rel=1e-12, abs=0.0)


def test_cluster_fold_row_folds_annihilate_a_cluster_measurable_exposure() -> None:
    """cluster_fold_leakage.mac STEP 9 (h -> 1): Dtil = 0 under row folds, not cluster folds."""
    cert = cluster_fold_leakage_certificate(
        n_clusters=30, cluster_size=8, icc_grid=(0.0,), lin_cluster_grid=(4,), n_seeds=20
    )
    assert cert.exposure_resid_var_row < 1e-20
    assert cert.exposure_resid_var_cluster > 0.1
    assert cert.annihilated_bias_predicted == -0.6
    assert np.all(cert.resid_icc_row == 0.0)


def test_cluster_fold_row_folds_do_not_bias_the_estimate() -> None:
    """cluster_fold_leakage.v moment_unbiased_under_orthogonality: no bias for any folds."""
    kwargs = {"n_clusters": 20, "cluster_size": 6, "icc_grid": (0.6,), "lin_cluster_grid": (20,)}
    # seed s draws the same data whatever n_seeds is, so consecutive seed-averages difference out
    # each seed's own error (seeds 0 and 1 together, n_seeds = 1 cannot form a std)
    certs = [cluster_fold_leakage_certificate(n_seeds=n, **kwargs) for n in range(2, 33)]
    for field, centre in (
        (lambda cc: float(cc.aware_bias_row[0]), 0.0),
        (lambda cc: float(cc.lin_bias_row[0]), 0.0),
        (lambda cc: cc.annihilated_bias_row, -0.6),
    ):
        totals = np.array([n * field(cc) for n, cc in zip(range(2, 33), certs, strict=True)])
        per_seed = np.diff(totals)
        se = float(np.std(per_seed, ddof=1)) / math.sqrt(per_seed.size)
        assert abs(float(np.mean(per_seed)) - centre) < 4.0 * se


# ---------------------------------------------------------------- clustered lower bound


def test_clustered_regret_map_is_lower_lipschitz() -> None:
    """clustered_van_trees.v LowerLipschitzRegret: R >= kappa0 (Bhat - B)^2 on a direction."""
    a, b, r = _PLANTS[0]
    q, x0 = np.eye(2), np.array([1.0, 0.5])
    unit = np.array([[1.0], [-0.5]]) / math.sqrt(1.25)
    k_star, _ = _riccati(a, b, q, r)

    def regret(err):
        k_hat, _ = _riccati(a, b + err * unit, q, r)
        out = []
        for k in (k_hat, k_star):
            a_cl = a - b @ k
            out.append(float(x0 @ solve_discrete_lyapunov(a_cl.T, q + k.T @ r @ k) @ x0))
        return out[0] - out[1]

    errs = np.r_[-np.geomspace(1e-1, 1e-3, 7), np.geomspace(1e-3, 1e-1, 7)]
    ratios = np.array([regret(e) / e**2 for e in errs])
    assert np.all(ratios > 0.0)
    assert float(np.min(ratios)) > 0.5 * float(np.max(ratios))


def test_clustered_g_times_regret_stays_above_a_positive_floor() -> None:
    """clustered_van_trees.v regret_floor_uniform_positive: G E[R] >= kappa0 G/(I0 + G Ic) > 0."""
    cert = clustered_lower_bound_certificate(g_grid=(20, 80, 320), n_seeds=24)
    assert cert.floor_positive > 0.0
    assert cert.floor_positive == pytest.approx(
        float(np.min(cert.g_times_regret)), rel=1e-15, abs=0.0
    )
    # a faster-than-1/G regret would drive G*regret to zero: its log-slope would be negative
    assert -0.25 < cert.plateau_slope < 0.5


# ---------------------------------------------------------------- order transfer


@settings(max_examples=30, deadline=None)
@given(b=st.floats(0.2, 3.0), rr=st.floats(0.05, 3.0), xt=st.floats(0.2, 3.0))
def test_composition_transfer_regret_is_the_exact_cost_gap(b: float, rr: float, xt: float) -> None:
    """composition_transfer.mac: the regret map is the exact CE cost gap, not a linearisation."""
    cert = composition_transfer_certificate(b=b, rr=rr, xt=xt, n_delta=5)
    for i, p in enumerate((1, 2, 3)):
        for d, reg in zip(cert.deltas, cert.regrets[i], strict=True):
            u_hat, u_b = _u_star(b + d**p, xt, rr), _u_star(b, xt, rr)
            exact = _exact_cost_gap(u_hat, None, b, xt, rr)
            # the docstring's own floor: u*(b + e) - u*(b) cancels, relative error ~4 eps|u*|/|du|
            rel = 8.0 * np.finfo(float).eps * abs(u_b) / abs(u_hat - u_b) + 1e-12
            assert reg == pytest.approx(exact, rel=rel, abs=1e-300)


@settings(max_examples=30, deadline=None)
@given(b=st.floats(0.2, 3.0), rr=st.floats(0.05, 3.0), xt=st.floats(0.2, 3.0))
def test_composition_transfer_regret_obeys_the_order_2p_bound(
    b: float, rr: float, xt: float
) -> None:
    """composition_transfer.v regret_order_2p: |du| <= L d^p gives k du^2 <= k L^2 d^(2p)."""
    cert = composition_transfer_certificate(b=b, rr=rr, xt=xt, n_delta=5)
    for i, p in enumerate((1, 2, 3)):
        for d, reg in zip(cert.deltas, cert.regrets[i], strict=True):
            grid = np.linspace(b, b + d**p, 401)
            slope = np.abs(xt * (rr - grid**2) / (grid**2 + rr) ** 2)
            lip = float(np.max(slope)) * (1.0 + 1e-3) + 1e-12
            assert reg <= (b * b + rr) * lip**2 * d ** (2 * p) * (1.0 + 1e-9)


def test_composition_transfer_slope_excess_is_the_window_prediction() -> None:
    """order_transfer_window.mac: slope = 2p + (lam cov(t,e) + 2 c2 cov(t,e^2))/var t + O(e^3)."""
    b, rr = 1.0, 0.5
    cert = composition_transfer_certificate()
    lam = (2 * b**3 - 6 * b * rr) / (rr**2 - b**4)
    c2 = (b**6 - 8 * b**4 * rr + 5 * b**2 * rr**2 - 2 * rr**3) / (2 * (rr**2 - b**4) ** 2)
    t = np.log(cert.deltas)

    def cov(y):
        return float(np.mean((t - t.mean()) * (y - y.mean())))

    for i, p in enumerate((1, 2, 3)):
        e = cert.deltas**p
        predicted = (lam * cov(e) + 2.0 * c2 * cov(e**2)) / float(np.var(t))
        excess = cert.slopes[i] - 2 * p
        assert excess > 0.0
        # what is left is one more power of e = delta^p: a fifth of the excess at p = 1, far less
        # at higher orders
        assert abs(excess - predicted) < 0.2 * excess / 20.0 ** (p - 1)
    assert cert.slopes == pytest.approx([2.048478, 4.012425, 6.002273], abs=1e-6)


# ---------------------------------------------------------------- confounded turnpike


@settings(max_examples=40, deadline=None)
@given(
    b=st.floats(0.1, 3.0),
    beta=st.floats(-0.09, 3.0),
    xref=st.floats(-3.0, 3.0),
    q=st.floats(0.01, 5.0),
    g=st.floats(0.0, 0.99),
)
def test_confounded_turnpike_offset_is_the_settled_tracking_error(
    b: float, beta: float, xref: float, q: float, g: float
) -> None:
    """confounded_turnpike.v turnpike_gap_formula: xref - b xref/(b + beta) = xref beta/(b+beta)."""
    cert = confounded_turnpike_certificate(b=b, beta=beta, xref=xref, q=q, g=g, horizon=20)
    # the plant relaxes to its static response, x_{t+1} = x_t/2 + b u/2, under the confounded
    # controller's constant action u = xref/(b + beta)
    u_conf = xref / (b + beta)
    settled = 0.0
    for _ in range(200):
        settled = 0.5 * settled + 0.5 * b * u_conf
    # xref - settled cancels: its error is the rounding of the settled state, not of the offset
    offset, rounding = xref - settled, 64 * np.finfo(float).eps * max(1.0, abs(settled))
    assert cert.turnpike_offset_simulated == pytest.approx(offset, rel=1e-12, abs=rounding)
    assert cert.turnpike_offset_formula == pytest.approx(
        cert.turnpike_offset_simulated, rel=1e-12, abs=rounding
    )
    assert cert.per_step_regret == pytest.approx(
        q * offset**2, rel=1e-12, abs=2 * q * (abs(offset) + rounding) * rounding
    )


@settings(max_examples=40, deadline=None)
@given(
    b=st.floats(0.1, 3.0),
    beta=st.floats(0.0, 3.0),
    q=st.floats(0.01, 5.0),
    g=st.floats(0.0, 0.99),
    horizon=st.integers(1, 400),
)
def test_confounded_turnpike_discounted_regret_is_a_bounded_geometric_sum(
    b: float, beta: float, q: float, g: float, horizon: int
) -> None:
    """confounded_turnpike.v discounted_regret_bounded: c (1 - g^T)/(1 - g) <= c/(1 - g)."""
    cert = confounded_turnpike_certificate(b=b, beta=beta, q=q, g=g, horizon=horizon)
    c = cert.per_step_regret
    t = cert.horizons
    closed = c * (1.0 - g**t) / (1.0 - g) if g > 0 else np.full(t.size, c)
    assert cert.discounted_regret == pytest.approx(closed, rel=1e-10, abs=1e-300)
    assert np.all(cert.discounted_regret <= cert.discounted_bound * (1.0 + 1e-12))
    assert cert.undiscounted_regret == pytest.approx(t * c, rel=1e-12, abs=1e-300)


# ---------------------------------------------------------------- confounding-robust LQ regret


def _regret_at(effect, effort, target, err):
    return _exact_cost_gap(
        _u_star(effect + err, target, effort),
        None,
        effect,
        target,
        effort,
    )


@settings(max_examples=40, deadline=None)
@given(
    effect=st.floats(0.2, 3.0),
    effort=st.floats(0.05, 3.0),
    target=st.floats(0.2, 3.0),
    eps=st.floats(0.0, 1.0),
    share=st.floats(0.0, 1.0),
)
def test_confounding_lq_regret_is_the_leading_term_of_the_exact_regret(
    effect: float, effort: float, target: float, eps: float, share: float
) -> None:
    """confounding_lq_regret.v cr_regret: L (eps + Delta)^2 leads the exact regret."""
    if abs(effort - effect**2) < 0.05:
        return  # L_reg vanishes at the knife edge and the leading term is not quadratic there
    l_fd = _central_l(effect, effort, target)
    assert lq_regret_sensitivity(effect, effort, target) == pytest.approx(l_fd, rel=1e-6, abs=0.0)
    scale = 1e-3 * min(math.sqrt(effort), effect)
    e1, e2 = scale * eps, scale * eps * share
    bound = confounding_robust_lq_regret(l_fd, e1, e2)
    exact = _regret_at(effect, effort, target, e1 + e2)
    assert bound == pytest.approx(exact, rel=0.05, abs=1e-30)


@settings(max_examples=40, deadline=None)
@given(
    l_reg=st.floats(0.0, 10.0),
    eps=st.floats(0.0, 1.0),
    d1=st.floats(0.0, 1.0),
    d2=st.floats(0.0, 1.0),
)
def test_confounding_lq_regret_floor_is_quadratic_monotone_and_below_linear(
    l_reg: float, eps: float, d1: float, d2: float
) -> None:
    """confounding_lq_regret.v: the floor is quadratic, monotone in the width, below linear."""
    lo, hi = sorted((d1, d2))
    assert confounding_robust_lq_regret(l_reg, eps, lo) <= confounding_robust_lq_regret(
        l_reg, eps, hi
    )
    assert confounding_robust_lq_regret(l_reg, 0.0, 2.0 * lo) == pytest.approx(
        4.0 * confounding_robust_lq_regret(l_reg, 0.0, lo), rel=1e-14, abs=1e-300
    )
    assert confounding_robust_lq_regret(l_reg, 0.0, lo) <= l_reg * lo
    assert confounding_robust_lq_regret(l_reg, eps, 0.0) == pytest.approx(
        l_reg * eps**2, rel=1e-14, abs=0.0
    )


def test_confounding_lq_regret_third_order_excess_is_as_documented() -> None:
    """confounding_lq_regret.v: a leading term, exceeded 0.9% at Gamma=1 and 5-8% at Gamma 1.5-5."""
    effect, effort, target = 1.3, 0.4, 1.0
    cert = confounding_robust_lq_regret_certificate()
    l_reg = lq_regret_sensitivity(effect, effort, target)
    excess = []
    for bound in cert.regret_bounds:
        width = math.sqrt(bound / l_reg)
        worst = max(_regret_at(effect, effort, target, s * width) for s in (-1.0, 1.0))
        excess.append(worst / bound - 1.0)
    assert excess[0] == pytest.approx(0.009, abs=0.001)
    assert all(0.045 <= x <= 0.085 for x in excess[1:])


@settings(max_examples=40, deadline=None)
@given(
    n=st.integers(1, 4),
    m=st.integers(1, 3),
    seed=st.integers(0, 2**32 - 1),
    delta=st.floats(1e-3, 1.0),
    p=st.integers(1, 3),
)
def test_matrix_lq_regret_order_doubles_and_is_bracketed_by_the_curvature(
    n: int, m: int, seed: int, delta: float, p: int
) -> None:
    """composition_transfer.v regret_order_2p, lifted: tr((d^p E)'H(d^p E)) = d^(2p) tr(E'HE)."""
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(n, n))
    h = g @ g.T
    stat, conf = rng.normal(size=(n, m)), rng.normal(size=(n, m))
    base = confounding_robust_lq_regret_matrix(h, stat, conf)
    scaled = confounding_robust_lq_regret_matrix(h, delta**p * stat, delta**p * conf)
    assert scaled == pytest.approx(delta ** (2 * p) * base, rel=1e-12, abs=1e-300)
    e = stat + conf
    ev = np.linalg.eigvalsh(h)
    fro = float(np.sum(e**2))
    tol = 1e-12 * ev[-1] * fro
    assert ev[0] * fro - tol <= base <= ev[-1] * fro + tol
    explicit = sum(float(e[:, j] @ h @ e[:, j]) for j in range(m))
    assert base == pytest.approx(explicit, rel=1e-10, abs=tol)


@settings(max_examples=40, deadline=None)
@given(l_reg=st.floats(0.0, 10.0), eps=st.floats(-1.0, 1.0), d=st.floats(-1.0, 1.0))
def test_matrix_lq_regret_reduces_to_the_scalar_one(l_reg: float, eps: float, d: float) -> None:
    """composition_transfer.v: the 1x1 matrix regret is the scalar L (eps + Delta)^2."""
    got = confounding_robust_lq_regret_matrix(
        np.array([[l_reg]]), np.array([[eps]]), np.array([[d]])
    )
    assert got == pytest.approx(confounding_robust_lq_regret(l_reg, eps, d), rel=1e-14, abs=1e-300)


# ---------------------------------------------------------------- conjugate time


def _riccati_reverse(a, b, q, r, s_t, horizon, stop=None):
    c = b * b / r

    def rhs(_, p):
        return [2.0 * a * p[0] + q - c * p[0] ** 2]

    events = None
    if stop is not None:

        def hit(_, p):
            return p[0] + stop

        hit.terminal = True
        events = hit
    return solve_ivp(
        rhs, (0.0, horizon), [s_t], rtol=1e-12, atol=1e-12, method="LSODA", events=events
    )


_conj = {
    "a": st.floats(-2.0, 2.0),
    "b": st.floats(0.3, 2.0),
    "r": st.floats(0.3, 2.0),
    "s_t": st.floats(-1.0, 1.0),
    "excess": st.floats(0.1, 3.0),
}


@settings(max_examples=12, deadline=None)
@given(**_conj)
def test_conjugate_time_is_the_escape_time_of_the_integrated_riccati(
    a: float, b: float, r: float, s_t: float, excess: float
) -> None:
    """conjugate_time.v t_conj_finite: the reverse Riccati escapes to -inf at (pi/2 + phi0)/mu."""
    c = b * b / r
    q = -(a * a) / c - excess
    cert = conjugate_time_certificate(a=a, b=b, q=q, r=r, s_t=s_t, margins=(1e-2, 1e-3))
    stop = 1e8
    sol = _riccati_reverse(a, b, q, r, s_t, 10.0 * cert.t_conj + 10.0, stop=stop)
    escape = float(sol.t_events[0][0])
    # p ~ -1/(c (t_conj - tau)) near the pole, so hitting -stop happens 1/(c stop) early
    assert escape == pytest.approx(cert.t_conj - 1.0 / (c * stop), rel=1e-6, abs=1e-8)


def test_conjugate_time_value_matches_the_integrated_riccati() -> None:
    """conjugate_time.mac STEP 2d: p(0) at horizon T is the closed-form tangent, before the pole."""
    cert = conjugate_time_certificate(margins=(0.3, 0.1, 0.03))
    for eps, val in zip(cert.margins, cert.value, strict=True):
        sol = _riccati_reverse(0.3, 1.0, -4.0, 1.0, 0.0, cert.t_conj - eps)
        assert val == pytest.approx(float(sol.y[0, -1]), rel=1e-7, abs=0.0)


def test_conjugate_time_pole_orders_are_one_two_four() -> None:
    """conjugate_time.v value_pole_residue: eps p(0) -> -r/b^2; eps^2 dk/db -> STEP 5e."""
    cert = conjugate_time_certificate()
    r, b = 1.0, 1.0
    # p(0) = -1/(c eps) + a/c + O(eps): the residue error is linear in eps
    for eps, res in zip(cert.margins, cert.value_residue, strict=True):
        assert abs(res + r / b**2) <= 0.6 * eps
    # coef2 = -0.6580462394032149 from conjugate_time.mac STEP 5e; the finite difference uses
    # delta_b = 1e-6, so the margins 1e-3 and 1e-4 are the ones it resolves
    assert cert.sensitivity_residue[2] == pytest.approx(0.6580462394032149, rel=2e-3, abs=0.0)
    assert cert.sensitivity_residue[3] == pytest.approx(0.6580462394032149, rel=2e-4, abs=0.0)
    assert cert.value_slope == pytest.approx(1.0, abs=0.01)
    assert cert.sensitivity_slope == pytest.approx(2.0, abs=0.02)
    assert cert.regret_slope == pytest.approx(4.0, abs=0.05)


@settings(max_examples=40, deadline=None)
@given(**_conj)
def test_conjugate_time_more_authority_brings_the_escape_closer(
    a: float, b: float, r: float, s_t: float, excess: float
) -> None:
    """conjugate_time.v t_conj_antitone_in_speed / speed_increases_with_authority."""
    c = b * b / r
    q = -(a * a) / c - excess
    cert = conjugate_time_certificate(a=a, b=b, q=q, r=r, s_t=s_t, margins=(1e-2, 1e-3))
    doubled = conjugate_time_certificate(a=a, b=2.0 * b, q=q, r=r, s_t=s_t, margins=(1e-2, 1e-3))
    assert cert.authority_shrinks_horizon < 1.0
    assert cert.authority_shrinks_horizon == pytest.approx(
        doubled.t_conj / cert.t_conj, rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("s_t", [0.0, 1.0, 3.0, 8.0])
def test_conjugate_time_definite_arm_is_the_integrated_riccati(s_t: float) -> None:
    """conjugate_time.mac STEP 7: under a positive-definite cost the Riccati solution stays
    bounded, on the tanh branch below the upper equilibrium (a + nu)/c = 2.32 and the coth branch
    above it. With arctanh alone s_t = 3 read nan."""
    a, b, q, r, delta = 0.3, 1.0, 4.0, 1.0, 1e-4
    cert = conjugate_time_certificate(a=a, b=b, q=-q, r=r, s_t=s_t, delta_b=delta)
    horizon = 10.0 * cert.t_conj
    ends = [_riccati_reverse(a, bb, q, r, s_t, horizon).y[0, -1] for bb in (b + delta, b - delta)]
    integrated = abs(ends[0] - ends[1]) * b / (r * 2.0 * delta)
    # LSODA at rtol 1e-12 over a central difference of step 1e-4: ~1e-8 relative
    assert cert.definite_sensitivity == pytest.approx(integrated, rel=1e-6, abs=1e-9)


def test_conjugate_time_refuses_a_cost_with_no_conjugate_point() -> None:
    """conjugate_time.v mu_sq_pos_iff: a conjugate point exists iff a^2 + (b^2/r) q < 0."""
    with pytest.raises(ValueError, match="no conjugate point"):
        conjugate_time_certificate(q=-0.05)


# ---------------------------------------------------------------- constrained CE regret


def _constrained_gap(u, b, xt, rr, umax) -> float:
    best = min(_u_star(b, xt, rr), umax)
    return _exact_cost_gap(u, best, b, xt, rr)


@settings(max_examples=40, deadline=None)
@given(xt=st.floats(0.3, 3.0), rr=st.floats(0.05, 3.0), frac=st.floats(0.2, 0.95))
def test_constrained_regret_is_the_exact_cost_gap_at_the_threshold(
    xt: float, rr: float, frac: float
) -> None:
    """constrained_ce_regret.v ce_regret / regret_zero_when_both_active, as true cost gaps."""
    umax = frac * xt / (2.0 * math.sqrt(rr))
    b_t0 = (xt - math.sqrt(xt * xt - 4.0 * rr * umax * umax)) / (2.0 * umax)
    hi = 0.9 * min(b_t0, rr / b_t0 - b_t0)
    cert = constrained_ce_regret_certificate(
        xt=xt, rr=rr, umax=umax, delta_lo=hi / 30, delta_hi=hi, n_delta=6
    )
    b_t = cert.threshold
    for d, r_in, r_act in zip(cert.deltas, cert.regret_inactive, cert.regret_active, strict=True):
        u_in = min(_u_star(b_t - d, xt, rr), umax)
        u_act = min(_u_star(b_t + d, xt, rr), umax)
        assert r_in == pytest.approx(_constrained_gap(u_in, b_t, xt, rr, umax), rel=1e-6, abs=1e-14)
        assert r_act == pytest.approx(
            _constrained_gap(u_act, b_t, xt, rr, umax), rel=1e-6, abs=1e-14
        )
        # "exactly 0" up to the rounding of the threshold itself: u*(b_t) may land below umax
        slack = umax - min(_u_star(b_t, xt, rr), umax)
        assert r_act <= (b_t * b_t + rr) * slack**2 * (1.0 + 1e-9)


@settings(max_examples=40, deadline=None)
@given(xt=st.floats(0.3, 3.0), rr=st.floats(0.05, 3.0), frac=st.floats(0.2, 0.95))
def test_constrained_regret_never_exceeds_unconstrained(xt: float, rr: float, frac: float) -> None:
    """constrained_ce_regret.v constrained_regret_le_unconstrained (clipping is non-expansive)."""
    umax = frac * xt / (2.0 * math.sqrt(rr))
    cert = constrained_ce_regret_certificate(
        xt=xt, rr=rr, umax=umax, delta_lo=1e-3, delta_hi=0.05, n_delta=6
    )
    assert np.all(cert.regret_inactive <= cert.regret_unconstrained * (1.0 + 1e-12))
    assert cert.max_constrained_ratio <= 1.0 + 1e-12


@settings(max_examples=40, deadline=None)
@given(xt=st.floats(0.3, 3.0), rr=st.floats(0.05, 3.0), frac=st.floats(0.2, 0.95))
def test_constrained_threshold_is_the_lower_root(xt: float, rr: float, frac: float) -> None:
    """constrained_ce_regret.mac STEP 4: u*(b_t) = umax at the lower root, b^- b^+ = rr."""
    umax = frac * xt / (2.0 * math.sqrt(rr))
    cert = constrained_ce_regret_certificate(xt=xt, rr=rr, umax=umax, n_delta=4)
    b_t = cert.threshold
    assert _u_star(b_t, xt, rr) == pytest.approx(umax, rel=1e-9, abs=0.0)
    assert b_t <= math.sqrt(rr) * (1.0 + 1e-12)
    upper = rr / b_t
    assert _u_star(upper, xt, rr) == pytest.approx(umax, rel=1e-9, abs=0.0)


def test_constrained_pessimism_budget_is_the_naive_controllers_regret() -> None:
    """constrained_ce_regret.mac (iii): a naive controller pays J(u*(bh); b) - J(umax; b), which
    is (b^2+rr)[(u*(bh) - u*(b))^2 - (umax - u*(b))^2]; it was 0.005784 at the defaults, 45% low."""
    xt, rr, umax = 1.0, 1.0, 0.45
    cert = constrained_ce_regret_certificate()
    b0 = cert.threshold + 0.1
    paid = max(
        _constrained_gap(_u_star(cert.threshold - d, xt, rr), b0, xt, rr, umax) for d in cert.deltas
    )
    assert cert.pessimism_budget == pytest.approx(paid, rel=1e-6, abs=0.0)
