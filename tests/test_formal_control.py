"""The control line's cited proofs and derivations, checked on the float code that implements them.

Each test reads one statement of a Rocq lemma or a Maxima derivation that a docstring in
``chc.uncertainty``, ``chc.plan``, ``chc.barrier``, ``chc.research.sensitivity`` or
``chc.allocation`` cites, and checks it on the library's output by another route: rollouts simulated
in NumPy against the error tubes, a linear program against the MSM closed form, brute-force grids
against the barrier margins, re-solved plans against the decision weights and the planned cost's
bias, a Monte Carlo of the plant against the noise tube.
"""

from __future__ import annotations

import math
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy import integrate
from scipy.optimize import linprog, minimize
from scipy.stats import chi2

from chc import QuadraticCost, causal_plan
from chc.allocation import decision_weight as allocation_decision_weight
from chc.barrier import (
    admissible_action_interval,
    barrier_gamma_star,
    control_channel,
    identification_radius_threshold,
    robust_barrier_margin,
    robust_safe_action,
    robust_safety_filter,
)
from chc.control import LinearConstraint
from chc.dynamics import LinearDynamics
from chc.plan import plan_regret_bound
from chc.response import Channel, Exponential, GeometricAdstock
from chc.uncertainty import (
    _linear_tube,
    _noise_covariances,
    _noise_tube,
    _tube,
    confounding_robust_closed_loop_bound,
    confounding_robust_inflation,
    confounding_robust_radius,
    constraint_margin,
    contractive_rollout_bound,
    gaussian_mixing,
    linear_rollout_bound,
    lipschitz_rollout_bound,
    time_varying_rollout_bound,
)

FORMAL_LINKS = {
    "chc.allocation.decision_weight": {
        "cites": ("allocation_decision_weight.mac",),
        "tests": (
            "test_allocation_weight_is_the_hessian_of_the_realised_loss",
            "test_allocation_weight_ignores_an_error_that_leaves_the_plan",
        ),
    },
    "chc.barrier": {
        "cites": ("barrier_feasibility.mac", "barrier_feasibility.v"),
        "tests": (
            "test_barrier_margin_is_the_best_guaranteed_derivative",
            "test_barrier_safe_action_attains_the_margin",
            "test_barrier_margin_loss_saturates_at_the_channel",
            "test_barrier_certified_iff_below_the_threshold",
            "test_barrier_gamma_star_inverts_the_msm_radius",
            "test_barrier_admissible_interval_is_the_admissible_set",
            "test_barrier_filter_action_is_guaranteed",
        ),
    },
    "chc.plan.CausalPlan.decision_weight": {
        "cites": ("plan_decision_weight.mac",),
        "tests": (
            "test_plan_decision_weight_is_the_hessian_of_the_replanned_regret",
            "test_plan_decision_weight_reads_only_the_free_actions",
        ),
    },
    "chc.plan.CausalPlan.planned_cost_bias": {
        "cites": ("planned_cost_bias.mac",),
        "tests": ("test_planned_cost_bias_is_the_expansion_of_the_replanned_gap",),
    },
    "chc.plan.PlanRegretBound": {
        "cites": ("constrained_plan_regret.mac",),
        "tests": (
            "test_plan_regret_bound_orders_its_three_numbers",
            "test_plan_regret_bound_vanishes_at_the_box_optimum",
        ),
    },
    "chc.plan.causal_plan": {
        "cites": ("predictive_tube.mac",),
        "tests": (
            "test_noise_tube_on_a_state_matrix_covers_at_its_level",
            "test_noise_tube_on_a_norm_bound_covers_at_least_its_level",
        ),
    },
    "chc.plan.plan_regret_bound": {
        "cites": (
            "constrained_plan_regret.mac",
            "constrained_plan_regret.v",
            "regret_bound_rows.mac",
        ),
        "tests": (
            "test_plan_regret_bound_holds_against_the_box_optimum",
            "test_plan_regret_bound_over_rows_holds_against_their_optimum",
            "test_plan_regret_bound_orders_its_three_numbers",
            "test_plan_regret_bound_vanishes_at_the_box_optimum",
        ),
    },
    "chc.research.sensitivity": {
        "cites": ("barrier_feasibility_lift.v",),
        "tests": (
            "test_sensitivity_margin_is_the_worst_case_over_the_operator_ball",
            "test_sensitivity_margin_is_the_worst_case_over_the_frobenius_ball",
            "test_sensitivity_filter_keeps_the_true_plant_safe",
        ),
    },
    "chc.uncertainty._step_parts": {
        "cites": ("euler_tube.mac",),
        "tests": (
            "test_euler_linear_tube_holds_on_euler_rollouts_with_field_errors",
            "test_euler_noise_tube_is_the_double_sum_on_its_own_step_matrices",
        ),
    },
    "chc.uncertainty._noise_tube": {
        "cites": ("predictive_tube.mac", "var_noise_tube.mac"),
        "tests": (
            "test_noise_tube_on_a_state_matrix_covers_at_its_level",
            "test_noise_tube_on_a_norm_bound_covers_at_least_its_level",
            "test_noise_tube_under_a_var_noise_is_the_double_sum",
            "test_noise_tube_under_a_var_noise_on_a_norm_bound_bounds_the_largest_variance",
        ),
    },
    "chc.uncertainty.confounding_robust_closed_loop_bound": {
        "cites": ("lipschitz_rollout.v",),
        "tests": (
            "test_closed_loop_confounding_bound_is_monotone_in_gamma",
            "test_closed_loop_confounding_bound_holds_on_a_confounded_loop",
            "test_closed_loop_confounding_bound_at_a_tiny_rate",
        ),
    },
    "chc.uncertainty.confounding_robust_inflation": {
        "cites": ("confounding_robust_cvar.mac", "confounding_robust_cvar.v"),
        "tests": (
            "test_msm_inflation_is_the_sharp_linear_program",
            "test_msm_inflation_is_nonnegative_tight_and_monotone",
        ),
    },
    "chc.uncertainty.confounding_robust_radius": {
        "cites": ("confounding_robust_cvar.v",),
        "tests": ("test_robust_radius_is_the_linear_program_and_grows_with_gamma",),
    },
    "chc.uncertainty.constraint_margin": {
        "cites": ("lipschitz_rollout.v",),
        "tests": ("test_constraint_margin_certifies_every_state_in_the_tube",),
    },
    "chc.uncertainty.contractive_rollout_bound": {
        "cites": ("contractive_euler.mac", "contractive_euler.v", "lipschitz_rollout.v"),
        "tests": (
            "test_contractive_bound_is_attained_on_a_rotating_contraction",
            "test_contractive_bound_is_infinite_past_the_step_condition",
            "test_contractive_radius_tends_to_the_continuous_one",
        ),
    },
    "chc.uncertainty.gaussian_mixing": {
        "cites": ("tube_levels.mac",),
        "tests": (
            "test_gaussian_mixing_bounds_the_exact_total_variation",
            "test_gaussian_mixing_is_pinsker_on_the_mean_kl",
            "test_gaussian_mixing_reads_windows_by_stride_and_span",
        ),
    },
    "chc.uncertainty.linear_rollout_bound": {
        "cites": ("rk4_rollout.mac",),
        "tests": (
            "test_linear_tube_holds_on_rk4_rollouts_with_stage_errors",
            "test_linear_tube_on_a_scalar_rate_is_the_lipschitz_tube",
        ),
    },
    "chc.uncertainty.lipschitz_rollout_bound": {
        "cites": ("lipschitz_rollout.mac", "lipschitz_rollout.v"),
        "tests": (
            "test_lipschitz_bound_holds_on_euler_rollouts",
            "test_lipschitz_bound_is_attained_on_a_linear_field",
            "test_lipschitz_bound_tends_to_the_linear_envelope",
            "test_lipschitz_bound_covers_the_gap_at_a_tiny_rate",
        ),
    },
    "chc.uncertainty.time_varying_rollout_bound": {
        "cites": ("lipschitz_rollout.v", "rk4_rollout.mac"),
        "tests": (
            "test_rk4_tube_holds_on_rk4_rollouts",
            "test_rk4_tube_is_the_gap_on_a_linear_field",
            "test_euler_tube_undercuts_rk4_rollouts",
        ),
    },
}

DT = 0.1


def _rk4_scalar(f, t: float, x: float, dt: float) -> float:
    k1 = f(t, x, 0)
    k2 = f(t + dt / 2, x + dt / 2 * k1, 1)
    k3 = f(t + dt / 2, x + dt / 2 * k2, 2)
    k4 = f(t + dt, x + dt * k3, 3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def _rk4_maps(a: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """RK4's step of ``x' = A x + B u`` with ``u`` held: ``x1 = R(dt A) x0 + dt phi(dt A) B u``."""
    z = dt * a
    eye = np.eye(a.shape[0])
    z2 = z @ z
    z3 = z2 @ z
    return eye + z + z2 / 2 + z3 / 6 + z3 @ z / 24, dt * (eye + z / 2 + z2 / 6 + z3 / 24)


# ---- error tubes: lipschitz_rollout.v, contractive_euler.v, rk4_rollout.mac ----


@settings(max_examples=40, deadline=None)
@given(
    lipschitz=st.floats(0.01, 3.0),
    eps=st.floats(1e-3, 1.0),
    dt=st.floats(0.01, 0.5),
    horizon=st.integers(1, 40),
    seed=st.integers(0, 2**31 - 1),
)
def test_lipschitz_bound_holds_on_euler_rollouts(lipschitz, eps, dt, horizon, seed) -> None:
    """lipschitz_rollout.v rollout_error_bound: Euler rollouts of an L-Lipschitz field, one off by
    at most eps a step, stay within eps ((1 + L dt)^k - 1) / L at every step k."""
    phase = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, horizon)
    x = y = 0.3
    for k in range(horizon):
        x, y = (
            x + dt * lipschitz * math.sin(x + 0.1),
            y + dt * (lipschitz * math.sin(y + 0.1) + eps * math.cos(phase[k] + 3.0 * y)),
        )
        bound = lipschitz_rollout_bound(lipschitz, eps, dt, k + 1)
        assert abs(x - y) <= bound * (1 + 1e-12)


@settings(max_examples=40, deadline=None)
@given(
    lipschitz=st.floats(1e-3, 3.0),
    eps=st.floats(1e-3, 1.0),
    dt=st.floats(0.01, 0.5),
    horizon=st.integers(1, 60),
)
def test_lipschitz_bound_is_attained_on_a_linear_field(lipschitz, eps, dt, horizon) -> None:
    """lipschitz_rollout.mac (2): on x' = L x against x' = L x + eps the Euler gap is the closed
    form exactly."""
    gap = 0.0
    for _ in range(horizon):
        gap = gap + dt * (lipschitz * gap + eps)
    assert lipschitz_rollout_bound(lipschitz, eps, dt, horizon) == pytest.approx(
        gap, rel=1e-9, abs=0.0
    )


@pytest.mark.parametrize("horizon", [1, 7, 300])
def test_lipschitz_bound_tends_to_the_linear_envelope(horizon) -> None:
    """lipschitz_rollout.mac (4): as L -> 0 the bound tends to eps dt H, and is that at L = 0."""
    eps, dt = 0.2, 0.05
    assert lipschitz_rollout_bound(0.0, eps, dt, horizon) == pytest.approx(
        eps * dt * horizon, rel=1e-15, abs=0.0
    )
    near = lipschitz_rollout_bound(1e-3, eps, dt, horizon)
    assert near == pytest.approx(eps * dt * horizon, rel=1e-3 * horizon * dt, abs=0.0)
    # at H = 1 the two are equal in exact arithmetic, so only to rounding
    assert near >= eps * dt * horizon * (1 - 4 * np.finfo(float).eps)


def test_lipschitz_bound_covers_the_gap_at_a_tiny_rate() -> None:
    """lipschitz_rollout.v rollout_error_bound at L = 1e-14: the bound is at least the Euler gap
    of x' = L x against x' = L x + eps, which is eps dt H to 1e-15. (1 + L dt)^H - 1 read 0.0
    there, and 6e-9 short of eps dt at L = 1e-7, H = 1."""
    gap = 0.0
    for _ in range(10):
        gap = (1.0 + 1e-14 * 0.01) * gap + 0.01 * 0.2
    assert lipschitz_rollout_bound(1e-14, 0.2, 0.01, 10) >= gap * (1 - 1e-12)
    assert lipschitz_rollout_bound(1e-7, 0.2, 0.05, 1) == pytest.approx(0.01, rel=1e-15, abs=0.0)


@settings(max_examples=40, deadline=None)
@given(
    rate=st.floats(0.1, 2.0),
    spin=st.floats(0.0, 3.0),
    share=st.floats(0.05, 0.95),
    eps=st.floats(1e-3, 1.0),
    horizon=st.integers(1, 200),
)
def test_contractive_bound_is_attained_on_a_rotating_contraction(
    rate, spin, share, eps, horizon
) -> None:
    """contractive_euler.mac (1)-(2), lipschitz_rollout.v gronwall_bounded: on the field
    x' = [[-c, w], [-w, -c]] x, log-norm -c and Lipschitz sqrt(c^2 + w^2), an Euler rollout pushed
    by eps along its own gap reaches eps dt (1 - q^H) / (1 - q), below eps dt / (1 - q)."""
    lipschitz = math.hypot(rate, spin)
    dt = share * 2.0 * rate / lipschitz**2
    step = np.eye(2) + dt * np.array([[-rate, spin], [-spin, -rate]])
    gap = np.zeros(2)
    for _ in range(horizon):
        moved = step @ gap
        size = float(np.linalg.norm(moved))
        gap = moved + dt * eps * (moved / size if size > 0 else np.array([1.0, 0.0]))
    bound = contractive_rollout_bound(rate, lipschitz, eps, dt, horizon)
    assert bound == pytest.approx(float(np.linalg.norm(gap)), rel=1e-9, abs=0.0)
    q = math.sqrt(1.0 - 2.0 * rate * dt + lipschitz**2 * dt**2)
    assert bound <= eps * dt / (1.0 - q) * (1 + 1e-12)


@settings(max_examples=40, deadline=None)
@given(rate=st.floats(0.1, 2.0), spin=st.floats(0.0, 3.0), share=st.floats(1.01, 3.0))
def test_contractive_bound_is_infinite_past_the_step_condition(rate, spin, share) -> None:
    """contractive_euler.v euler_factor_lt_one: the factor is below 1 only for dt < 2 c / L^2;
    past it the bound reads inf, and so does a field with no contraction."""
    lipschitz = math.hypot(rate, spin)
    dt = share * 2.0 * rate / lipschitz**2
    assert contractive_rollout_bound(rate, lipschitz, 0.1, dt, 5) == math.inf
    assert contractive_rollout_bound(-rate, lipschitz, 0.1, 0.01, 5) == math.inf


def test_contractive_radius_tends_to_the_continuous_one() -> None:
    """contractive_euler.mac (4): as dt -> 0 the uniform radius eps dt / (1 - q) reaches eps / c."""
    rate, lipschitz, eps = 0.7, 2.0, 0.3
    for dt in (1e-3, 1e-4, 1e-5):
        far = contractive_rollout_bound(rate, lipschitz, eps, dt, int(60.0 / (rate * dt)))
        assert far == pytest.approx(eps / rate, rel=4.0 * lipschitz**2 * dt / rate, abs=0.0)


@settings(max_examples=40, deadline=None)
@given(
    rates=st.lists(st.floats(0.0, 3.0), min_size=1, max_size=15),
    seed=st.integers(0, 2**31 - 1),
    dt=st.floats(0.01, 0.5),
)
def test_rk4_tube_holds_on_rk4_rollouts(rates, seed, dt) -> None:
    """lipschitz_rollout.v rk4_rollout_error_bound: RK4 rollouts of an L_k-Lipschitz field and of
    the field off by at most eps_k at every stage stay within the RK4 tube."""
    rng = np.random.default_rng(seed)
    budgets = rng.uniform(0.0, 1.0, len(rates))
    phases = rng.uniform(0.0, 2.0 * np.pi, (len(rates), 4))
    tube = np.asarray(time_varying_rollout_bound(list(rates), list(budgets), dt))
    x = y = 0.4
    for k, (lk, ek) in enumerate(zip(rates, budgets, strict=True)):
        x = _rk4_scalar(lambda t, s, i, lk=lk: lk * math.sin(s), 0.0, x, dt)
        y = _rk4_scalar(
            lambda t, s, i, lk=lk, ek=ek, k=k: lk * math.sin(s) + ek * math.cos(phases[k, i] + s),
            0.0,
            y,
            dt,
        )
        assert abs(x - y) <= tube[k + 1] * (1 + 1e-12)


@settings(max_examples=40, deadline=None)
@given(
    lipschitz=st.floats(0.0, 3.0),
    eps=st.floats(1e-3, 1.0),
    dt=st.floats(0.01, 0.5),
    horizon=st.integers(1, 30),
)
def test_rk4_tube_is_the_gap_on_a_linear_field(lipschitz, eps, dt, horizon) -> None:
    """rk4_rollout.mac (1), (4): on x' = L x against x' = L x + eps the RK4 tube is the two RK4
    rollouts' gap exactly."""
    tube = np.asarray(time_varying_rollout_bound([lipschitz] * horizon, [eps] * horizon, dt))
    x, y = 0.0, 0.0
    gaps = [0.0]
    for _ in range(horizon):
        x = _rk4_scalar(lambda t, s, i: lipschitz * s, 0.0, x, dt)
        y = _rk4_scalar(lambda t, s, i: lipschitz * s + eps, 0.0, y, dt)
        gaps.append(y - x)
    np.testing.assert_allclose(tube, gaps, rtol=1e-9, atol=0.0)


def test_euler_tube_undercuts_rk4_rollouts() -> None:
    """rk4_rollout.mac (4): x' = x against x' = x + 0.1 at dt = 1, from one start; the RK4 gaps
    41/240, 3649/5760, 260801/138240 meet the RK4 tube and pass the Euler tube 0.1, 0.3, 0.7."""
    x = y = 1.0
    gaps = []
    for _ in range(3):
        x = _rk4_scalar(lambda t, s, i: s, 0.0, x, 1.0)
        y = _rk4_scalar(lambda t, s, i: s + 0.1, 0.0, y, 1.0)
        gaps.append(y - x)
    np.testing.assert_allclose(gaps, [41 / 240, 3649 / 5760, 260801 / 138240], 1e-14)
    rk4 = np.asarray(time_varying_rollout_bound([1.0] * 3, [0.1] * 3, 1.0))[1:]
    euler = np.asarray(time_varying_rollout_bound([1.0] * 3, [0.1] * 3, 1.0, integrator="euler"))
    np.testing.assert_allclose(rk4, gaps, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(euler[1:], [0.1, 0.3, 0.7], rtol=1e-12, atol=0.0)
    assert np.all(euler[1:] < np.asarray(gaps))


@settings(max_examples=40, deadline=None)
@given(
    entries=st.lists(st.floats(-2.0, 2.0), min_size=4, max_size=4),
    horizon=st.integers(1, 10),
    dt=st.floats(0.01, 0.3),
    seed=st.integers(0, 2**31 - 1),
)
def test_linear_tube_holds_on_rk4_rollouts_with_stage_errors(entries, horizon, dt, seed) -> None:
    """rk4_rollout.mac (2): RK4 rollouts of x' = A x + c and of the field off by at most eps_k at
    each of its four stages stay within e_k = sum_j eps_j g_(k-1-j)."""
    a = np.array(entries).reshape(2, 2)
    rng = np.random.default_rng(seed)
    budgets = rng.uniform(0.0, 1.0, horizon)
    tube = np.asarray(linear_rollout_bound(a, list(budgets), dt))
    drive = np.array([0.3, -0.2])
    for _ in range(8):
        x = y = np.array([0.5, -1.0])
        for k in range(horizon):
            errors = rng.standard_normal((4, 2))
            errors *= budgets[k] / np.linalg.norm(errors, axis=1, keepdims=True)
            stages_x, stages_y = [], []
            for i, (half, err) in enumerate(zip((0.0, 0.5, 0.5, 1.0), errors, strict=True)):
                bx = x + dt * half * (stages_x[-1] if i else 0.0)
                by = y + dt * half * (stages_y[-1] if i else 0.0)
                stages_x.append(a @ bx + drive)
                stages_y.append(a @ by + drive + err)
            weights = (1.0, 2.0, 2.0, 1.0)
            x = x + dt / 6 * sum(w * s for w, s in zip(weights, stages_x, strict=True))
            y = y + dt / 6 * sum(w * s for w, s in zip(weights, stages_y, strict=True))
            assert np.linalg.norm(x - y) <= tube[k + 1] * (1 + 1e-12) + 1e-15


@settings(max_examples=40, deadline=None)
@given(
    lipschitz=st.floats(0.0, 3.0),
    budgets=st.lists(st.floats(0.0, 1.0), min_size=1, max_size=12),
    dt=st.floats(0.01, 0.5),
)
def test_linear_tube_on_a_scalar_rate_is_the_lipschitz_tube(lipschitz, budgets, dt) -> None:
    """rk4_rollout.mac (3): every stage weight is nonnegative at z >= 0, so on A = L >= 0 the
    matrix tube is the norm-Lipschitz RK4 tube; a contracting A gives a narrower one."""
    linear = np.asarray(linear_rollout_bound([[lipschitz]], budgets, dt))
    lipschitz_tube = np.asarray(time_varying_rollout_bound([lipschitz] * len(budgets), budgets, dt))
    np.testing.assert_allclose(linear, lipschitz_tube, rtol=1e-12, atol=0.0)
    contracting = np.asarray(linear_rollout_bound(-2.0 * np.eye(2), budgets, dt))
    widened = np.asarray(time_varying_rollout_bound([2.0] * len(budgets), budgets, dt))
    assert np.all(contracting[1:] <= widened[1:])


@settings(max_examples=40, deadline=None)
@given(
    lipschitz_g=st.floats(0.0, 5.0),
    radii=st.lists(st.floats(0.0, 1.0), min_size=1, max_size=6),
    seed=st.integers(0, 2**31 - 1),
)
def test_constraint_margin_certifies_every_state_in_the_tube(lipschitz_g, radii, seed) -> None:
    """lipschitz_rollout.v constraint_tightening: where g(x_hat) + L_g e <= 0, an L_g-Lipschitz g
    is <= 0 at every x within e of x_hat; on a linear g the margin is attained."""
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(2)
    direction /= np.linalg.norm(direction)
    offset = rng.uniform(-3.0, 1.0)

    def g(x: np.ndarray) -> np.ndarray:
        return offset + lipschitz_g * np.sin(x @ direction)

    nominal = rng.uniform(-2.0, 2.0, (len(radii), 2))
    margins = np.asarray(
        constraint_margin(jnp.asarray(g(nominal)), lipschitz_g, jnp.asarray(radii))
    )
    for state, radius, margin in zip(nominal, radii, margins, strict=True):
        if margin > 0.0:
            continue
        moves = rng.standard_normal((64, 2))
        moves *= radius * rng.uniform(0, 1, (64, 1)) ** 0.5 / np.linalg.norm(moves, axis=1)[:, None]
        assert np.all(g(state + moves) <= 1e-12)
    linear = offset + lipschitz_g * nominal @ direction
    worst = offset + lipschitz_g * (nominal + np.outer(radii, direction)) @ direction
    np.testing.assert_allclose(
        np.asarray(constraint_margin(jnp.asarray(linear), lipschitz_g, jnp.asarray(radii))),
        worst,
        rtol=1e-12,
        atol=1e-12,
    )


# ---- the tube's levels: tube_levels.mac ----


def _exact_scalar_beta(rho: float, k: int) -> float:
    """``E_x TV(N(rho^k x, 1 - rho^(2k)), N(0, 1))``, ``x ~ N(0, 1)``, by Gauss-Hermite in ``x``
    and Simpson's rule in ``y``."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(120)
    y = np.linspace(-14.0, 14.0, 28001)
    var = 1.0 - rho ** (2 * k)
    far = np.exp(-(y**2) / 2) / math.sqrt(2 * math.pi)
    total = 0.0
    for node, weight in zip(nodes, weights, strict=True):
        near = np.exp(-((y - rho**k * node) ** 2) / (2 * var)) / math.sqrt(2 * math.pi * var)
        total += weight * integrate.simpson(np.abs(near - far), x=y) / 2
    return total / math.sqrt(2 * math.pi)


@pytest.mark.parametrize(("rho", "k"), [(0.8, 1), (0.8, 3), (-0.6, 2), (0.95, 5), (0.3, 1)])
def test_gaussian_mixing_bounds_the_exact_total_variation(rho, k) -> None:
    """tube_levels.mac (3): on one state the bound sqrt(-log(1 - rho^(2k))) / 2 is above the exact
    mean total variation from the k-step law to the stationary one."""
    beta = gaussian_mixing([[rho]], [[1.0 - rho**2]])
    assert _exact_scalar_beta(rho, k) <= beta(k)
    assert beta(k) == pytest.approx(
        math.sqrt(-math.log1p(-(rho ** (2 * k)))) / 2, rel=1e-12, abs=0.0
    )


@settings(max_examples=40, deadline=None)
@given(
    size=st.integers(1, 3),
    seed=st.integers(0, 2**31 - 1),
    radius=st.floats(0.3, 0.95),
    k=st.integers(1, 6),
)
def test_gaussian_mixing_is_pinsker_on_the_mean_kl(size, seed, radius, k) -> None:
    """tube_levels.mac (2): beta(k) is min(1, sqrt(E KL / 2)), E KL the mean over the stationary
    law of KL(N(F^k x, S - F^k S F^k') || N(0, S)) written out term by term."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((size, size))
    f *= radius / max(abs(np.linalg.eigvals(f)))
    root = rng.standard_normal((size, size))
    q = root @ root.T + 0.1 * np.eye(size)
    stationary = np.linalg.solve(np.eye(size * size) - np.kron(f, f), q.ravel()).reshape(size, size)
    stationary = (stationary + stationary.T) / 2
    fk = np.linalg.matrix_power(f, k)
    left = stationary - fk @ stationary @ fk.T
    inverse = np.linalg.inv(stationary)
    sign, logdet = np.linalg.slogdet(inverse @ left)
    assume(sign > 0)
    mean_kl = (
        np.trace(inverse @ left) + np.trace(inverse @ fk @ stationary @ fk.T) - size - logdet
    ) / 2
    assert gaussian_mixing(f, q)(k) == pytest.approx(
        min(1.0, math.sqrt(mean_kl / 2)), rel=1e-6, abs=0.0
    )


@pytest.mark.parametrize(("stride", "span"), [(1, 0), (3, 2), (2, 5)])
def test_gaussian_mixing_reads_windows_by_stride_and_span(stride, span) -> None:
    """tube_levels.mac (2): windows j apart, stride steps between starts and span steps long, are
    beta(j stride - span) apart, and 1 where they meet."""
    f, q = [[0.7, 0.2], [0.0, 0.5]], [[1.0, 0.3], [0.3, 0.5]]
    step = gaussian_mixing(f, q)
    windows = gaussian_mixing(f, q, stride=stride, span=span)
    for j in range(1, 8):
        lag = j * stride - span
        assert windows(j) == (step(lag) if lag > 0 else 1.0)


# ---- the noise tube: predictive_tube.mac ----

PLANT_A = np.array([[-1.0, 0.5], [0.0, -2.0]])
PLANT_B = np.array([[1.0], [0.5]])
NOISE = np.array([[4.0, 1.0], [1.0, 2.0]]) / 100.0
HORIZON = 6


def _plan_with_noise(lipschitz, persistence: float):
    cost = QuadraticCost(Q=jnp.eye(2), R=jnp.eye(1), Qf=jnp.eye(2), x_target=jnp.array([0.5, 0.0]))
    return causal_plan(
        LinearDynamics(jnp.asarray(PLANT_A), jnp.asarray(PLANT_B)),
        jnp.array([1.0, -0.5]),
        cost,
        DT,
        HORIZON,
        -2.0,
        2.0,
        lipschitz=lipschitz,
        noise=NOISE,
        noise_persistence=persistence,
        steps=20,
    )


def _plant_coverage(plan, persistence: float, sims: int = 20000, seed: int = 0) -> np.ndarray:
    """The share of plants, each off the model's RK4 step by AR(1) Gaussian ``w_k``, that stay
    within the plan's tube at each step."""
    step, gain = _rk4_maps(PLANT_A, DT)
    rng = np.random.default_rng(seed)
    root = np.linalg.cholesky(NOISE)
    actions = np.asarray(plan.actions)
    trajectory = np.asarray(plan.trajectory)
    tube = np.asarray(plan.uncertainty_tube)
    x = np.broadcast_to(trajectory[0], (sims, 2)).copy()
    w = rng.standard_normal((sims, 2)) @ root.T
    inside = []
    for k in range(HORIZON):
        x = x @ step.T + (gain @ PLANT_B @ actions[k]) + w
        inside.append(np.mean(np.linalg.norm(x - trajectory[k + 1], axis=1) <= tube[k + 1]))
        fresh = rng.standard_normal((sims, 2)) @ root.T
        w = persistence * w + math.sqrt(1.0 - persistence**2) * fresh
    return np.array(inside)


@pytest.mark.parametrize("persistence", [0.0, 0.5, -0.5])
def test_noise_tube_on_a_state_matrix_covers_at_its_level(persistence) -> None:
    """predictive_tube.mac (1): on a state matrix the gap's covariance is exact, so a plant off the
    plan's step by AR(1) Gaussian noise is inside the tube with probability 0.95 at every step."""
    plan = _plan_with_noise(PLANT_A, persistence)
    coverage = _plant_coverage(plan, persistence)
    sd = math.sqrt(0.95 * 0.05 / 20000)
    np.testing.assert_allclose(coverage, 0.95, rtol=0.0, atol=4.5 * sd)


@pytest.mark.parametrize("persistence", [0.0, 0.5, -0.5])
def test_noise_tube_on_a_norm_bound_covers_at_least_its_level(persistence) -> None:
    """predictive_tube.mac (2), (5): on a norm bound L >= ||A|| the scalar recursion bounds the
    gap's largest variance, so the plant is inside the tube with probability at least 0.95."""
    plan = _plan_with_noise(float(np.linalg.norm(PLANT_A, 2)), persistence)
    coverage = _plant_coverage(plan, persistence)
    sd = math.sqrt(0.95 * 0.05 / 20000)
    assert np.all(coverage >= 0.95 - 4.5 * sd)
    assert coverage[0] == pytest.approx(0.95, rel=0.0, abs=4.5 * sd)


VAR_PHI = np.array([[0.7, 0.4], [-0.1, 0.25]])


def _var_covariance(phi: np.ndarray, innovation: np.ndarray) -> np.ndarray:
    """The stationary ``S`` of ``w_{k+1} = Phi w_k + eta``: ``S = Phi S Phi' + Cov(eta)``."""
    n = phi.shape[0]
    return np.linalg.solve(np.eye(n * n) - np.kron(phi, phi), innovation.ravel()).reshape(n, n)


def _var_double_sum(steps: list[np.ndarray], phi: np.ndarray, s: np.ndarray, k: int) -> np.ndarray:
    """``sum_{i,j<k} A_{k,i} Cov(w_i, w_j) A_{k,j}'``, ``A_{k,i} = M_{k-1} ... M_{i+1}``."""

    def carry(i: int) -> np.ndarray:
        out = np.eye(s.shape[0])
        for step in steps[i + 1 : k]:
            out = step @ out
        return out

    def cov(i: int, j: int) -> np.ndarray:
        if i >= j:
            return np.linalg.matrix_power(phi, i - j) @ s
        return s @ np.linalg.matrix_power(phi.T, j - i)

    return sum(carry(i) @ cov(i, j) @ carry(j).T for i in range(k) for j in range(k))


def test_noise_tube_under_a_var_noise_is_the_double_sum() -> None:
    """var_noise_tube.mac (1b), (2): under a non-normal VAR(1) noise and state matrices that move
    each step, the recursion's covariance is the double sum over the noise's lagged covariances, and
    at Phi = rho I it is the AR(1) recursion's to the bit's neighbourhood."""
    rng = np.random.default_rng(7)
    rates = rng.uniform(-1.0, 1.0, (5, 2, 2)) - np.eye(2)
    s = _var_covariance(VAR_PHI, NOISE)
    covariances = _noise_covariances(rates, s, VAR_PHI, DT, 5)
    steps = [_rk4_maps(rate, DT)[0] for rate in rates]
    for k in range(1, 6):
        exact = _var_double_sum(steps, VAR_PHI, s, k)
        np.testing.assert_allclose(covariances[k], exact, rtol=1e-12, atol=1e-15)
    scalar = _noise_covariances(rates, s, 0.6, DT, 5)
    matrix = _noise_covariances(rates, s, 0.6 * np.eye(2), DT, 5)
    for a, b in zip(scalar, matrix, strict=True):
        np.testing.assert_allclose(b, a, rtol=1e-13, atol=1e-17)


def test_noise_tube_under_a_var_noise_on_a_norm_bound_bounds_the_largest_variance() -> None:
    """var_noise_tube.mac (4): on a norm bound L >= ||A|| the scalar recursion with ||Phi|| bounds
    the largest eigenvalue of the exact covariance, on a non-normal A and a non-normal Phi."""
    s = _var_covariance(VAR_PHI, NOISE)
    horizon = 8
    bound = float(np.linalg.norm(PLANT_A, 2))
    exact = _noise_covariances(PLANT_A, s, VAR_PHI, DT, horizon)
    radii = _noise_tube(np.asarray(bound), s, VAR_PHI, DT, horizon)
    level = float(chi2.ppf(0.95, 2))
    for k in range(2, horizon + 1):
        assert radii[k] ** 2 / level >= np.linalg.eigvalsh(exact[k])[-1] * (1 - 1e-12)
    assert radii[1] == pytest.approx(_noise_tube(PLANT_A, s, VAR_PHI, DT, 1)[1], rel=1e-12, abs=0.0)


@settings(max_examples=40, deadline=None)
@given(
    entries=st.lists(st.floats(-2.0, 2.0), min_size=4, max_size=4),
    horizon=st.integers(1, 10),
    dt=st.floats(0.01, 1.0),
    seed=st.integers(0, 2**31 - 1),
)
def test_euler_linear_tube_holds_on_euler_rollouts_with_field_errors(
    entries, horizon, dt, seed
) -> None:
    """euler_tube.mac (1), (2): Euler rollouts of x' = A x + c and of the field off by at most
    eps_k over step k stay within e_k = sum_j dt ||(I + dt A)^(k-1-j)|| eps_j; on A = L >= 0 that
    is the Euler Gronwall tube, which the gap meets with every error along +1."""
    a = np.array(entries).reshape(2, 2)
    rng = np.random.default_rng(seed)
    budgets = rng.uniform(0.0, 1.0, horizon)
    tube = _linear_tube(a, list(budgets), dt, integrator="euler")
    drive = np.array([0.3, -0.2])
    for _ in range(8):
        x = y = np.array([0.5, -1.0])
        for k in range(horizon):
            error = rng.standard_normal(2)
            error *= budgets[k] / np.linalg.norm(error)
            x, y = x + dt * (a @ x + drive), y + dt * (a @ y + drive + error)
            assert np.linalg.norm(x - y) <= tube[k + 1] * (1 + 1e-12) + 1e-15
    rate = abs(entries[0])
    scalar = _linear_tube([[rate]], list(budgets), dt, integrator="euler")
    gronwall = _tube([rate] * horizon, list(budgets), dt, "euler")
    np.testing.assert_allclose(scalar, gronwall, rtol=1e-12, atol=0.0)
    x = y = 0.0
    for k in range(horizon):
        x, y = x + dt * rate * x, y + dt * (rate * y + budgets[k])
        assert y - x == pytest.approx(gronwall[k + 1], rel=1e-12, abs=1e-15)


def test_euler_noise_tube_is_the_double_sum_on_its_own_step_matrices() -> None:
    """euler_tube.mac (5), var_noise_tube.mac (1b): under the Euler map the noise's covariance is
    the double sum over the noise's lagged covariances carried by M_k = I + dt A_k; on a norm bound
    the scalar recursion's growth is 1 + L dt, which on one state is the matrix tube's own."""
    rng = np.random.default_rng(11)
    rates = rng.uniform(-1.0, 1.0, (5, 2, 2)) - np.eye(2)
    s = _var_covariance(VAR_PHI, NOISE)
    dt = 0.9
    covariances = _noise_covariances(rates, s, VAR_PHI, dt, 5, integrator="euler")
    steps = [np.eye(2) + dt * rate for rate in rates]
    for k in range(1, 6):
        exact = _var_double_sum(steps, VAR_PHI, s, k)
        np.testing.assert_allclose(covariances[k], exact, rtol=1e-12, atol=1e-15)
    rk4 = _noise_covariances(rates, s, VAR_PHI, dt, 5)
    assert not np.allclose(rk4[2], covariances[2], rtol=1e-3, atol=0.0)
    one = np.array([[0.04]])
    for rate in (0.0, 0.7):
        bound = _noise_tube(np.asarray(rate), one, 0.5, dt, 6, integrator="euler")
        matrix = _noise_tube(np.array([[rate]]), one, 0.5, dt, 6, integrator="euler")
        np.testing.assert_allclose(bound, matrix, rtol=1e-12, atol=0.0)
    assert _noise_tube(np.asarray(0.7), one, 0.5, dt, 6)[3] > bound[3]


# ---- the MSM radius: confounding_robust_cvar.mac and .v ----


def _sharp_worst_case(outcomes: np.ndarray, gamma: float) -> float:
    """``max (1/n) sum w_i y_i`` over ``w_i in [1/Gamma, Gamma]`` with ``mean(w) = 1``, by HiGHS."""
    n = outcomes.size
    found = linprog(
        -outcomes / n,
        A_eq=np.ones((1, n)),
        b_eq=[float(n)],
        bounds=[(1.0 / gamma, gamma)] * n,
        method="highs",
    )
    assert found.status == 0
    return -float(found.fun)


def _tails(outcomes: np.ndarray, gamma: float) -> tuple[float, float]:
    """The top ``tau = 1/(Gamma + 1)`` share's mean, a fractional point at its share, and the
    rest's."""
    y = np.sort(outcomes)[::-1]
    mass = y.size / (gamma + 1.0)
    weights = np.clip(mass - np.arange(y.size), 0.0, 1.0)
    top = float(weights @ y) / mass
    return top, float((1.0 - weights) @ y) / (y.size - mass)


@settings(max_examples=40, deadline=None)
@given(n=st.integers(2, 30), gamma=st.floats(1.0, 10.0), seed=st.integers(0, 2**31 - 1))
def test_msm_inflation_is_the_sharp_linear_program(n, gamma, seed) -> None:
    """confounding_robust_cvar.mac (2)-(3): the mean plus (Gamma - 1)/(Gamma + 1) times the tails'
    gap is the largest mean any weights in [1/Gamma, Gamma] with mean 1 give."""
    y = np.random.default_rng(seed).standard_normal(n) * 3.0
    upper, lower = _tails(y, gamma)
    closed = float(np.mean(y)) + confounding_robust_inflation(upper, lower, gamma)
    assert closed == pytest.approx(_sharp_worst_case(y, gamma), rel=0.0, abs=1e-9)


@settings(max_examples=40, deadline=None)
@given(
    gap=st.floats(0.0, 100.0),
    low=st.floats(1.0, 50.0),
    high=st.floats(1.0, 50.0),
)
def test_msm_inflation_is_nonnegative_tight_and_monotone(gap, low, high) -> None:
    """confounding_robust_cvar.v msm_inflation_nonneg, _tight_at_one, _monotone: the inflation is
    >= 0 on a nonnegative gap, 0 at Gamma = 1, and nondecreasing in Gamma."""
    low, high = min(low, high), max(low, high)
    assert confounding_robust_inflation(gap, 0.0, 1.0) == 0.0
    assert confounding_robust_inflation(gap, 0.0, low) >= 0.0
    assert confounding_robust_inflation(gap, 0.0, low) <= confounding_robust_inflation(
        gap, 0.0, high
    )
    assert confounding_robust_inflation(gap, 0.0, high) <= gap


@settings(max_examples=40, deadline=None)
@given(
    n=st.integers(2, 25),
    low=st.floats(1.0, 8.0),
    high=st.floats(1.0, 8.0),
    nominal=st.floats(0.0, 5.0),
    seed=st.integers(0, 2**31 - 1),
)
def test_robust_radius_is_the_linear_program_and_grows_with_gamma(
    n, low, high, nominal, seed
) -> None:
    """confounding_robust_cvar.v robust_radius_ge_nominal, robust_radius_monotone: the radius is the
    nominal one plus the sharp worst case's gap over the mean, never below the nominal one and
    nondecreasing in Gamma with the tails moving."""
    low, high = min(low, high), max(low, high)
    y = np.random.default_rng(seed).exponential(2.0, n)
    mean = float(np.mean(y))
    at_low = confounding_robust_radius(nominal, y, low)
    at_high = confounding_robust_radius(nominal, y, high)
    assert at_low == pytest.approx(nominal + _sharp_worst_case(y, low) - mean, rel=0.0, abs=1e-9)
    assert at_low >= nominal
    assert at_high >= at_low - 1e-12


@settings(max_examples=40, deadline=None)
@given(
    low=st.floats(1.0, 20.0),
    high=st.floats(1.0, 20.0),
    horizon=st.integers(0, 40),
    base=st.floats(0.0, 0.5),
)
def test_closed_loop_confounding_bound_is_monotone_in_gamma(low, high, horizon, base) -> None:
    """lipschitz_rollout.v closed_loop_confounding_monotone: the closed-loop tube is nondecreasing
    in Gamma at every horizon, and at Gamma = 1 it is the tube of the base error alone."""
    low, high = min(low, high), max(low, high)
    args = (1.0, 0.5, 1.5, base, 0.8, 0.4)
    assert confounding_robust_closed_loop_bound(
        *args, low, 0.05, horizon
    ) <= confounding_robust_closed_loop_bound(*args, high, 0.05, horizon)
    assert confounding_robust_closed_loop_bound(*args, 1.0, 0.05, horizon) == pytest.approx(
        lipschitz_rollout_bound(1.0 + 0.5 * 1.5, base, 0.05, horizon), rel=1e-15, abs=0.0
    )


@settings(max_examples=40, deadline=None)
@given(
    drift=st.floats(-1.0, 1.0),
    effect=st.floats(-2.0, 2.0),
    gain=st.floats(-2.0, 2.0),
    gamma=st.floats(1.0, 6.0),
    sign=st.sampled_from([-1.0, 1.0]),
    horizon=st.integers(1, 40),
)
def test_closed_loop_confounding_bound_holds_on_a_confounded_loop(
    drift, effect, gain, gamma, sign, horizon
) -> None:
    """lipschitz_rollout.v rollout_error_bound at rate L_x + L_u L_pi: an Euler loop u = k x on a
    plant whose effect is off by Delta(Gamma) stays within the tube of eps = |u|_max Delta."""
    rate = abs(drift) + abs(effect) * abs(gain)
    assume(rate == 0.0 or rate >= 1e-3)  # below, test_closed_loop_confounding_bound_at_a_tiny_rate
    dt, gap = 0.05, 0.6
    delta = confounding_robust_inflation(gap, 0.0, gamma)
    model = plant = 1.0
    gaps, actions = [], []
    for _ in range(horizon):
        actions.append(gain * plant)
        model = model + dt * ((drift + effect * gain) * model)
        plant = plant + dt * ((drift + effect * gain) * plant + sign * delta * gain * plant)
        gaps.append(abs(plant - model))
    magnitude = max(abs(u) for u in actions)
    for k, gap_k in enumerate(gaps, start=1):
        bound = confounding_robust_closed_loop_bound(
            abs(drift), abs(effect), abs(gain), 0.0, magnitude, gap, gamma, dt, k
        )
        assert gap_k <= bound * (1 + 1e-9) + 1e-15


def test_closed_loop_confounding_bound_at_a_tiny_rate() -> None:
    """lipschitz_rollout.v rollout_error_bound at L_x + L_u L_pi = 1e-45: one Euler step of
    x' = 1e-45 u, u = x, against the plant whose effect is off by Delta(2) = 0.2 is 0.01 apart."""
    delta = confounding_robust_inflation(0.6, 0.0, 2.0)
    gap = abs((1.0 + 0.05 * (1e-45 - delta)) - (1.0 + 0.05 * 1e-45))
    bound = confounding_robust_closed_loop_bound(0.0, 1e-45, 1.0, 0.0, 1.0, 0.6, 2.0, 0.05, 1)
    assert gap <= bound * (1 + 1e-9)


# ---- safety under partial identification: barrier_feasibility.mac/.v and the vector lift ----


@settings(max_examples=40, deadline=None)
@given(
    drift=st.floats(-3.0, 3.0),
    channel=st.floats(-3.0, 3.0),
    radius=st.floats(0.0, 4.0),
    u_max=st.floats(0.01, 5.0),
)
def test_barrier_margin_is_the_best_guaranteed_derivative(drift, channel, radius, u_max) -> None:
    """barrier_feasibility.v margin_upper_bound, margin_attained, zero_action_optimal_when_
    unidentified: max over |u| <= U of min over |delta| <= d of a + (g + delta) u is
    a + (|g| - d)_+ U."""
    grid = np.concatenate([np.linspace(-u_max, u_max, 2001), [0.0]])
    guaranteed = np.minimum(drift + (channel + radius) * grid, drift + (channel - radius) * grid)
    assert robust_barrier_margin(drift, abs(channel), radius, u_max) == pytest.approx(
        float(np.max(guaranteed)), rel=0.0, abs=1e-12
    )


@settings(max_examples=40, deadline=None)
@given(
    states=st.integers(1, 4),
    levers=st.integers(1, 4),
    share=st.floats(0.0, 2.0),
    u_max=st.floats(0.1, 5.0),
    seed=st.integers(0, 2**31 - 1),
)
def test_barrier_safe_action_attains_the_margin(states, levers, share, u_max, seed) -> None:
    """barrier_feasibility_lift.v margin_upper_bound_mv, margin_attained_mv, zero_action_optimal_mv:
    no action in the ball beats a + (||w|| - d)_+ U, robust_safe_action attains it, and it is 0
    once d >= ||w||."""
    rng = np.random.default_rng(seed)
    grad_h, b_hat = rng.standard_normal(states), rng.standard_normal((states, levers))
    w = b_hat.T @ grad_h
    channel = control_channel(grad_h, b_hat)
    assert channel == pytest.approx(float(np.linalg.norm(w)), rel=1e-14, abs=0.0)
    radius, drift = share * channel, -0.3
    best = robust_barrier_margin(drift, channel, radius, u_max)
    action = np.ravel(robust_safe_action(grad_h, b_hat, radius, u_max))
    assert np.linalg.norm(action) <= u_max * (1 + 1e-14)
    attained = drift + w @ action - radius * np.linalg.norm(action)
    assert attained == pytest.approx(best, rel=0.0, abs=1e-12 * (1 + abs(best)))
    tries = rng.standard_normal((256, levers))
    tries *= u_max * rng.uniform(0, 1, (256, 1)) / np.linalg.norm(tries, axis=1, keepdims=True)
    assert np.all(drift + tries @ w - radius * np.linalg.norm(tries, axis=1) <= best + 1e-12)
    if radius >= channel:
        assert not np.any(action)


@settings(max_examples=40, deadline=None)
@given(
    channel=st.floats(0.0, 3.0),
    radius=st.floats(0.0, 6.0),
    u_max=st.floats(0.01, 5.0),
)
def test_barrier_margin_loss_saturates_at_the_channel(channel, radius, u_max) -> None:
    """barrier_feasibility.v margin_loss_saturates: a radius d costs the margin U min(d, |g|)."""
    lost = robust_barrier_margin(0.0, channel, 0.0, u_max) - robust_barrier_margin(
        0.0, channel, radius, u_max
    )
    assert lost == pytest.approx(u_max * min(radius, channel), rel=1e-12, abs=1e-15)


@settings(max_examples=40, deadline=None)
@given(
    drift=st.floats(-3.0, 3.0),
    channel=st.floats(0.0, 3.0),
    radius=st.floats(0.0, 4.0),
    u_max=st.floats(0.05, 5.0),
    alpha_h=st.floats(-2.0, 2.0),
)
def test_barrier_certified_iff_below_the_threshold(drift, channel, radius, u_max, alpha_h) -> None:
    """barrier_feasibility.v certified_below_threshold, violated_above_threshold, no_certificate_
    when_deficit_exceeds_authority: the margin clears -alpha h exactly when d <= d*."""
    threshold = identification_radius_threshold(drift, channel, u_max, alpha_h)
    margin = robust_barrier_margin(drift, channel, radius, u_max)
    deficit = -alpha_h - drift
    room = 1e-9 * (1 + abs(drift) + abs(alpha_h) + channel * u_max)
    assume(abs(margin + alpha_h) > room)
    certified = margin >= -alpha_h
    if deficit <= 0.0:
        assert threshold == math.inf
        assert certified
    elif deficit > u_max * channel:
        assert math.isnan(threshold)
        assert not certified
    else:
        assert 0.0 <= threshold <= channel
        assert certified == (radius <= threshold)


@settings(max_examples=40, deadline=None)
@given(gap=st.floats(0.01, 10.0), share=st.floats(1e-3, 0.999), grad_norm=st.floats(0.01, 10.0))
def test_barrier_gamma_star_inverts_the_msm_radius(gap, share, grad_norm) -> None:
    """barrier_feasibility.v msm_radius_at_gamma_star, gamma_star_above_one: Delta(Gamma*) times
    ||grad h|| is the threshold radius, and Gamma* > 1."""
    threshold = share * gap * grad_norm
    gamma_star = barrier_gamma_star(threshold, gap, grad_norm)
    assert gamma_star > 1.0
    assert confounding_robust_inflation(gap, 0.0, gamma_star) * grad_norm == pytest.approx(
        threshold, rel=1e-9, abs=0.0
    )


@settings(max_examples=40, deadline=None)
@given(
    channel=st.floats(-3.0, 3.0),
    radius=st.floats(0.0, 4.0),
    u_max=st.floats(0.1, 5.0),
    drift=st.floats(-3.0, 3.0),
    alpha_h=st.floats(-2.0, 2.0),
)
def test_barrier_admissible_interval_is_the_admissible_set(
    channel, radius, u_max, drift, alpha_h
) -> None:
    """barrier_feasibility.mac: the actions in |u| <= U whose guaranteed derivative
    a + g u - d |u| clears -alpha h form the interval returned, or none where it is (nan, nan)."""
    lo, hi = admissible_action_interval(channel, radius, u_max, drift, alpha_h)
    grid = np.linspace(-u_max, u_max, 4001)
    clears = drift + channel * grid - radius * np.abs(grid) >= -alpha_h
    slack = 2e-9 * (1 + abs(channel) + radius) * u_max
    sharp = np.abs(drift + channel * grid - radius * np.abs(grid) + alpha_h) > slack
    if math.isnan(lo):
        assert not np.any(clears & sharp)
    else:
        inside = (grid >= lo) & (grid <= hi)
        assert np.array_equal(clears[sharp], inside[sharp])


@settings(max_examples=40, deadline=None)
@given(
    nominal=st.floats(-10.0, 10.0),
    channel=st.floats(-3.0, 3.0),
    radius=st.floats(0.0, 4.0),
    u_max=st.floats(0.1, 5.0),
    drift=st.floats(-3.0, 3.0),
    alpha_h=st.floats(-2.0, 2.0),
)
def test_barrier_filter_action_is_guaranteed(
    nominal, channel, radius, u_max, drift, alpha_h
) -> None:
    """barrier_feasibility.v robust_margin: where the admissible set is not empty, the filtered
    action clears -alpha h for every channel in [g - d, g + d], and stays within |u| <= U."""
    lo, _ = admissible_action_interval(channel, radius, u_max, drift, alpha_h)
    action = robust_safety_filter(nominal, channel, radius, u_max, drift, alpha_h)
    assert abs(action) <= u_max
    if not math.isnan(lo):
        for true_channel in (channel - radius, channel + radius):
            assert drift + true_channel * action >= -alpha_h - 1e-12 * (1 + abs(drift))


def _worst_case_check(seed: int, ball: str) -> None:
    rng = np.random.default_rng(seed)
    states, levers = 3, 2
    grad_h, b_hat = rng.standard_normal(states), rng.standard_normal((states, levers))
    delta, u_max, drift = 0.4 * rng.uniform(0.1, 2.0), 1.5, -0.2
    radius = delta * float(np.linalg.norm(grad_h))
    best = robust_barrier_margin(drift, control_channel(grad_h, b_hat), radius, u_max)
    action = np.ravel(robust_safe_action(grad_h, b_hat, radius, u_max))
    for _ in range(200):
        error = rng.standard_normal((states, levers))
        size = np.linalg.norm(error, 2 if ball == "operator" else "fro")
        error *= delta * rng.uniform(0, 1) / size
        assert drift + grad_h @ (b_hat + error) @ action >= best - 1e-12
    if np.any(action):
        worst = (
            -delta * np.outer(grad_h, action) / (np.linalg.norm(grad_h) * np.linalg.norm(action))
        )
        assert np.linalg.norm(worst, 2 if ball == "operator" else "fro") == pytest.approx(
            delta, rel=1e-12, abs=0.0
        )
        assert drift + grad_h @ (b_hat + worst) @ action == pytest.approx(
            best, rel=0.0, abs=1e-12 * (1 + abs(best))
        )


@pytest.mark.parametrize("seed", range(6))
def test_sensitivity_margin_is_the_worst_case_over_the_operator_ball(seed) -> None:
    """barrier_feasibility_lift.v worst_case_over_operator_ball: at the robust action, no channel
    error of operator norm <= Delta takes the barrier derivative below the margin with
    d = Delta ||grad h||, and a rank-one one reaches it."""
    _worst_case_check(seed, "operator")


@pytest.mark.parametrize("seed", range(6))
def test_sensitivity_margin_is_the_worst_case_over_the_frobenius_ball(seed) -> None:
    """barrier_feasibility_lift.v worst_case_over_frobenius_ball: the same over the Frobenius ball
    of radius Delta."""
    _worst_case_check(seed, "frobenius")


@settings(max_examples=40, deadline=None)
@given(
    b_hat=st.floats(0.5, 2.0),
    share=st.floats(0.0, 0.6),
    where=st.floats(-1.0, 1.0),
    decay=st.floats(0.0, 0.5),
    alpha_dt=st.floats(0.05, 1.0),
    seed=st.integers(0, 2**31 - 1),
)
def test_sensitivity_filter_keeps_the_true_plant_safe(
    b_hat, share, where, decay, alpha_dt, seed
) -> None:
    """barrier_feasibility_lift.v robust_filter_keeps_the_true_plant_safe, discrete_forward_
    invariance: an Euler plant whose true channel is anywhere within d of the estimate, filtered at
    every step, keeps h = x_limit - x >= (1 - alpha dt)^k h_0 >= 0."""
    dt, limit = 0.1, 2.0
    alpha = alpha_dt / dt
    radius = share * b_hat
    u_max = 1.1 * decay * limit / (b_hat - radius) + 0.1
    rng = np.random.default_rng(seed)
    x = 1.0
    h0 = limit - x
    for k in range(60):
        true_b = b_hat + where * radius if k % 2 else b_hat + rng.uniform(-1, 1) * radius
        h = limit - x
        lo, _ = admissible_action_interval(-b_hat, radius, u_max, -decay * x, alpha * h)
        assert not math.isnan(lo)
        u = robust_safety_filter(u_max, -b_hat, radius, u_max, -decay * x, alpha * h)
        x = x + dt * (decay * x + true_b * u)
        assert limit - x >= (1 - alpha_dt) ** (k + 1) * h0 - 1e-12


# ---- plans: plan_decision_weight.mac, planned_cost_bias.mac, constrained_plan_regret.mac ----


def _extrapolated(difference, step: float = 1e-3) -> float:
    """Richardson's limit of a central difference whose error is even in the step:
    ``(4 D(h / 2) - D(h)) / 3`` cancels its ``h^2`` term."""
    return (4.0 * difference(step / 2) - difference(step)) / 3.0


def _linear_problem(seed: int, levers: int = 2, horizon: int = 3):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1.0, 1.0, (2, 2)) - 0.5 * np.eye(2)
    b = rng.uniform(-1.0, 1.0, (2, levers)) + np.eye(2, levers)
    x0 = rng.uniform(-1.0, 1.0, 2)
    q, r, qf = (
        np.diag(rng.uniform(0.5, 2.0, 2)),
        np.diag(rng.uniform(0.05, 0.5, levers)),
        2 * np.eye(2),
    )
    return a, b, x0, q, r, qf, horizon


def _quadratic(problem, change: np.ndarray):
    """``J(U) = U' H U / 2 + c' U + k`` on the RK4 one-step map moved by ``change @ u``, built from
    the stacked rollout."""
    a, b, x0, q, r, qf, horizon = problem
    levers = b.shape[1]
    step, gain = _rk4_maps(a, DT)
    channel = gain @ b + change
    free = [np.linalg.matrix_power(step, k) @ x0 for k in range(horizon + 1)]
    reach = np.zeros((horizon + 1, 2, horizon * levers))
    for k in range(1, horizon + 1):
        for j in range(k):
            reach[k, :, j * levers : (j + 1) * levers] = (
                np.linalg.matrix_power(step, k - 1 - j) @ channel
            )
    weights = [q] * horizon + [qf]
    hessian = np.kron(np.eye(horizon), r)
    linear = np.zeros(horizon * levers)
    constant = 0.0
    for k in range(horizon + 1):
        hessian += reach[k].T @ weights[k] @ reach[k]
        linear += reach[k].T @ weights[k] @ free[k]
        constant += free[k] @ weights[k] @ free[k] / 2
    return hessian, linear, constant


def _value(problem, change, actions):
    hessian, linear, constant = _quadratic(problem, change)
    return actions @ hessian @ actions / 2 + linear @ actions + constant


def _optimum(problem, change):
    hessian, linear, _ = _quadratic(problem, change)
    return -np.linalg.solve(hessian, linear)


def _library_plan(problem, actions: np.ndarray, lo=-1e3, hi=1e3):
    a, b, x0, q, r, qf, horizon = problem
    cost = QuadraticCost(
        Q=jnp.asarray(q), R=jnp.asarray(r), Qf=jnp.asarray(qf), x_target=jnp.zeros(2)
    )
    model = LinearDynamics(jnp.asarray(a), jnp.asarray(b))
    plan = causal_plan(model, jnp.asarray(x0), cost, DT, horizon, lo, hi, steps=1)
    shaped = jnp.asarray(actions.reshape(horizon, b.shape[1]))
    return replace(plan, actions=shaped), model, cost


def _replanned_regret(problem, change, base, held=None):
    """``J(U*(0); E) - min_U J(U; E)``, exactly: ``(U*(0) - U*(E))' H_E (U*(0) - U*(E)) / 2``; with
    ``held`` the actions kept where they are, the minimum over the others."""
    hessian, linear, _ = _quadratic(problem, change)
    moving = np.ones(base.size, dtype=bool)
    if held is not None:
        moving[held] = False
    best = base.copy()
    sub = hessian[np.ix_(moving, moving)]
    best[moving] = -np.linalg.solve(
        sub, linear[moving] + hessian[np.ix_(moving, ~moving)] @ base[~moving]
    )
    return (base - best) @ hessian @ (base - best) / 2


@pytest.mark.parametrize("seed", range(2))
def test_plan_decision_weight_is_the_hessian_of_the_replanned_regret(seed) -> None:
    """plan_decision_weight.mac STEP 2-3: the regret against the plan that knew the channel's error
    E is vec(E)' W vec(E) / 2 to second order: the symmetric second difference of the re-planned
    regret along any direction reads W there."""
    problem = _linear_problem(seed)
    base = _optimum(problem, np.zeros((2, 2)))
    plan, _, _ = _library_plan(problem, base)
    weight = plan.decision_weight().matrix
    rng = np.random.default_rng(100 + seed)
    for direction in [*np.eye(4), *rng.standard_normal((3, 4))]:
        second = _extrapolated(
            lambda h, direction=direction: (
                (
                    _replanned_regret(problem, h * direction.reshape(2, 2), base)
                    + _replanned_regret(problem, -h * direction.reshape(2, 2), base)
                )
                / h**2
            )
        )
        assert direction @ weight @ direction == pytest.approx(second, rel=1e-6, abs=1e-12)


def test_plan_decision_weight_reads_only_the_free_actions() -> None:
    """plan_decision_weight.mac STEP 4: with an action held at a bound its gradient pushes against,
    the regret re-plans the other actions only, and W is that regret's Hessian."""
    problem = _linear_problem(7, levers=1)
    free = _optimum(problem, np.zeros((2, 1)))
    limit = 0.5 * abs(free[0])
    held = np.sign(free[0]) * limit
    hessian, linear, _ = _quadratic(problem, np.zeros((2, 1)))
    base = free.copy()
    base[0] = held
    base[1:] = -np.linalg.solve(hessian[1:, 1:], linear[1:] + hessian[1:, 0] * held)
    pull = (hessian @ base + linear)[0]
    assert pull * np.sign(held) < 0  # the bound binds with a strict multiplier
    box = np.array([[limit], [1e3], [1e3]])
    plan, _, _ = _library_plan(problem, base, jnp.asarray(-box), jnp.asarray(box))
    found = plan.decision_weight()
    assert found.free == base.size - 1
    assert found.weakly_active == 0
    for direction in np.eye(2):
        second = _extrapolated(
            lambda h, direction=direction: (
                (
                    _replanned_regret(problem, h * direction.reshape(2, 1), base, held=[0])
                    + _replanned_regret(problem, -h * direction.reshape(2, 1), base, held=[0])
                )
                / h**2
            )
        )
        assert direction @ found.matrix @ direction == pytest.approx(second, rel=1e-6, abs=1e-12)


@pytest.mark.parametrize("seed", range(2))
def test_planned_cost_bias_is_the_expansion_of_the_replanned_gap(seed) -> None:
    """planned_cost_bias.mac STEP 1, 4: a plan made on a channel off by d reads its cost short of
    what it pays by gap(d), whose expansion is -J_e . d + d' (2 W - J_ee) d / 2: the bias at
    S = v v' is half the second difference along v, and at E[e] = m the first difference along m."""
    problem = _linear_problem(seed)
    base = _optimum(problem, np.zeros((2, 2)))
    plan, _, _ = _library_plan(problem, base)

    def gap(d: np.ndarray) -> float:
        moved = _optimum(problem, d)
        return _value(problem, np.zeros((2, 2)), moved) - _value(problem, d, moved)

    rng = np.random.default_rng(200 + seed)
    for v in rng.standard_normal((2, 4)):
        d = v.reshape(2, 2)
        second = _extrapolated(lambda h, d=d: (gap(h * d) + gap(-h * d)) / (2 * h**2))
        found = plan.planned_cost_bias(np.outer(v, v), error_mean=d)
        assert found.optimism + found.curvature == pytest.approx(second, rel=1e-6, abs=1e-10)
        assert found.optimism >= 0.0
        first = _extrapolated(lambda h, d=d: (gap(h * d) - gap(-h * d)) / (2 * h))
        assert found.first_order == pytest.approx(first, rel=1e-6, abs=1e-10)


def _box_optimum(problem, lo: float, hi: float) -> tuple[np.ndarray, float]:
    hessian, linear, constant = _quadratic(problem, np.zeros((2, problem[1].shape[1])))
    found = minimize(
        lambda u: u @ hessian @ u / 2 + linear @ u,
        np.zeros(linear.size),
        jac=lambda u: hessian @ u + linear,
        bounds=[(lo, hi)] * linear.size,
        method="L-BFGS-B",
        options={"ftol": 1e-15, "gtol": 1e-13, "maxiter": 10_000},
    )
    return found.x, float(found.fun) + constant


@pytest.mark.parametrize("seed", range(3))
def test_plan_regret_bound_holds_against_the_box_optimum(seed) -> None:
    """constrained_plan_regret.v strong_convexity_turns_a_gain_bound_into_a_regret_bound,
    a_bound_at_the_smaller_modulus_still_holds: at any plan in the box, J(U) - min over the box of
    J is at most the bound, on the measured modulus and on the smaller lambda_min(R)."""
    problem = _linear_problem(seed)
    lo, hi = -0.3, 0.3
    _, best = _box_optimum(problem, lo, hi)
    plan, model, cost = _library_plan(problem, np.zeros(6), lo, hi)
    rng = np.random.default_rng(300 + seed)
    floor = float(np.min(np.diag(problem[4])))
    for actions in rng.uniform(lo, hi, (4, 6)):
        at = replace(plan, actions=jnp.asarray(actions.reshape(3, 2)))
        gap = _value(problem, np.zeros((2, 2)), actions) - best
        measured = plan_regret_bound(at, model, jnp.asarray(problem[2]), cost, DT, lo, hi)
        assert measured.status == "certified"
        assert gap <= measured.bound + 1e-10
        supplied = plan_regret_bound(
            at, model, jnp.asarray(problem[2]), cost, DT, lo, hi, modulus=floor
        )
        assert gap <= supplied.bound + 1e-10


@pytest.mark.parametrize("seed", range(3))
def test_plan_regret_bound_orders_its_three_numbers(seed) -> None:
    """constrained_plan_regret.v the_box_only_tightens_the_bound,
    the_frank_wolfe_gap_bounds_the_gain, a_gain_bound_over_the_box_is_nonnegative,
    the_box_separates_into_per_lever_terms:
    0 <= bound <= min(|g|^2 / (2 mu), Frank-Wolfe gap), and the per-lever terms sum to it."""
    problem = _linear_problem(seed)
    plan, model, cost = _library_plan(problem, np.zeros(6), -0.3, 0.3)
    rng = np.random.default_rng(400 + seed)
    for actions in rng.uniform(-0.3, 0.3, (3, 6)):
        at = replace(plan, actions=jnp.asarray(actions.reshape(3, 2)))
        found = plan_regret_bound(at, model, jnp.asarray(problem[2]), cost, DT, -0.3, 0.3)
        assert found.bound >= 0.0
        assert found.bound <= found.unconstrained_bound * (1 + 1e-12)
        assert found.bound <= found.frank_wolfe_gap * (1 + 1e-12)
        assert found.unconstrained_bound == pytest.approx(
            found.gradient_norm**2 / (2 * found.modulus), rel=1e-12, abs=0.0
        )
        assert sum(found.per_lever) == pytest.approx(found.bound, rel=1e-12, abs=1e-15)
        assert all(term >= 0.0 for term in found.per_lever)


def _row_optimum(problem, lo: float, hi: float, row: np.ndarray, budget: float) -> float:
    hessian, linear, constant = _quadratic(problem, np.zeros((2, problem[1].shape[1])))
    found = minimize(
        lambda u: u @ hessian @ u / 2 + linear @ u,
        np.zeros(linear.size),
        jac=lambda u: hessian @ u + linear,
        bounds=[(lo, hi)] * linear.size,
        constraints=[{"type": "ineq", "fun": lambda u: budget - row @ u, "jac": lambda u: -row}],
        method="SLSQP",
        options={"ftol": 1e-15, "maxiter": 1000},
    )
    assert found.success
    return float(found.fun) + constant


@pytest.mark.parametrize("seed", range(3))
def test_plan_regret_bound_over_rows_holds_against_their_optimum(seed) -> None:
    """regret_bound_rows.mac STEP 1 (weak duality): with a budget row beside the box, J(U) less the
    minimum over the box and the row is at most the bound read off the dual, which is never above
    the box's own bound at the same plan."""
    problem = _linear_problem(seed)
    lo, hi = -0.3, 0.3
    _, unconstrained = _box_optimum(problem, lo, hi)
    row, budget = np.ones(6), 0.4
    best = _row_optimum(problem, lo, hi, row, budget)
    assert best >= unconstrained - 1e-12
    plan, model, cost = _library_plan(problem, np.zeros(6), lo, hi)
    rows = (LinearConstraint(row, -np.inf, budget),)
    rng = np.random.default_rng(500 + seed)
    for actions in rng.uniform(lo, hi, (12, 6)):
        if row @ actions > budget:
            continue
        at = replace(plan, actions=jnp.asarray(actions.reshape(3, 2)))
        x0 = jnp.asarray(problem[2])
        found = plan_regret_bound(at, model, x0, cost, DT, lo, hi, constraints=rows)
        box = plan_regret_bound(at, model, x0, cost, DT, lo, hi)
        gap = _value(problem, np.zeros((2, 2)), actions) - best
        assert found.status == "certified"
        assert gap <= found.bound + 1e-10
        assert found.bound <= box.bound * (1 + 1e-9) + 1e-12


@pytest.mark.parametrize("seed", range(3))
def test_plan_regret_bound_vanishes_at_the_box_optimum(seed) -> None:
    """constrained_plan_regret.v the_constrained_gap_vanishes_at_a_clipped_optimum,
    result_six_s_bound_charges_regret_at_that_optimum: at the box's optimum the bound is 0 where
    |g|^2 / (2 mu) is not."""
    problem = _linear_problem(seed)
    best, _ = _box_optimum(problem, -0.3, 0.3)
    plan, model, cost = _library_plan(problem, best, -0.3, 0.3)
    found = plan_regret_bound(plan, model, jnp.asarray(problem[2]), cost, DT, -0.3, 0.3)
    pinned = np.isclose(np.abs(best), 0.3, rtol=0.0, atol=1e-12)
    assert pinned.any()
    assert found.bound <= 1e-9 * found.unconstrained_bound
    assert found.unconstrained_bound > 1e-6


# ---- allocation: allocation_decision_weight.mac ----

ONE_PERIOD = GeometricAdstock(0.0, length=1, normalized=False)


def _exponential_split(coefficients, scales, budget):
    """The best split of ``budget`` over ``b_c (1 - exp(-u_c / K_c))``, every channel inside its
    box: the slopes ``b_c / K_c exp(-u_c / K_c)`` meet at one price ``lam``."""
    log_price = (np.sum(scales * np.log(coefficients / scales)) - budget) / np.sum(scales)
    return scales * (np.log(coefficients / scales) - log_price), math.exp(log_price)


def _realised_loss(coefficients, scales, budget, change):
    best, price = _exponential_split(coefficients, scales, budget)
    planned, _ = _exponential_split(coefficients + change[0::2], scales + change[1::2], budget)
    move = planned - best
    # b_c exp(-u_c / K_c) = price K_c at the best split, and the moves sum to 0
    return price * float(np.sum(scales * (np.expm1(-move / scales) + move / scales)))


def _allocation_case(seed: int):
    rng = np.random.default_rng(seed)
    coefficients = rng.uniform(400.0, 1200.0, 3)
    scales = rng.uniform(30.0, 120.0, 3)
    budget = 150.0
    split, _ = _exponential_split(coefficients, scales, budget)
    assert np.all((split > 1.0) & (split < budget - 1.0))
    channels = tuple(
        Channel(ONE_PERIOD, Exponential(float(k)), float(b))
        for b, k in zip(coefficients, scales, strict=True)
    )
    found = allocation_decision_weight(channels, budget, 1, lower=[0.0] * 3, upper=[budget] * 3)
    order = [f"{c}.{name}" for c in range(3) for name in ("coefficient", "curve.scale")]
    index = [found.parameters.index(name) for name in order]
    return coefficients, scales, budget, found.matrix[np.ix_(index, index)], found


@pytest.mark.parametrize("seed", [1, 2])
def test_allocation_weight_is_the_hessian_of_the_realised_loss(seed) -> None:
    """allocation_decision_weight.mac STEP 1: three exponential channels, one period; the split made
    on parameters off by d and scored on the true ones loses d' W d / 2 to second order."""
    coefficients, scales, budget, weight, found = _allocation_case(seed)
    assert found.pinned == ()
    scale = np.ravel(np.column_stack([coefficients, scales]))
    rng = np.random.default_rng(500 + seed)
    for direction in rng.standard_normal((4, 6)):
        d = direction * scale
        second = _extrapolated(
            lambda h, d=d: (
                (
                    _realised_loss(coefficients, scales, budget, h * d)
                    + _realised_loss(coefficients, scales, budget, -h * d)
                )
                / h**2
            )
        )
        assert d @ weight @ d == pytest.approx(second, rel=1e-6, abs=0.0)


def test_allocation_weight_ignores_an_error_that_leaves_the_plan() -> None:
    """allocation_decision_weight.mac: W = D' diag(h) D weighs an error by the plan's move, so
    scaling every coefficient by one factor, which moves no split, costs nothing."""
    coefficients, _, _, weight, _ = _allocation_case(3)
    d = np.ravel(np.column_stack([coefficients, np.zeros(3)]))
    assert abs(d @ weight @ d) <= 1e-10 * float(np.abs(weight).max()) * float(d @ d)
