"""Property tests that tie the field and PDE modules -- Galerkin convection-diffusion, the LQ
mean-field game and its dual-weighted estimators, the congestion games, the structured residual
backbones and the circulant operators -- to the Rocq lemmas and Maxima derivations their docstrings
cite, each checked on the float code by a route other than the formula the code evaluates."""

from __future__ import annotations

import itertools
import math
from dataclasses import replace
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st
from scipy.integrate import solve_ivp
from scipy.linalg import circulant, expm
from scipy.optimize import brentq, minimize

from chc.games import congestion_contraction_modulus
from chc.research.deep_galerkin import (
    _ANTI_MONOTONE_GAME,
    CongestedMeanFieldGame,
    LQMeanFieldGame,
    MeanFieldDGM,
    adjoint_weighted_error,
    dual_weighted_error_estimate,
    finite_population_gap_certificate,
    residual_blindness_sweep,
)
from chc.research.galerkin import (
    convection_diffusion_1d,
    convection_diffusion_exact,
    optimal_upwind,
)
from chc.research.games import equilibrium_monotonicity_certificate
from chc.research.transport import advection_diffusion_field
from chc.residual import (
    ContractiveResidual,
    PortHamiltonianResidual,
    SpectralResidual,
    _ConvexEnergy,
    _critical_points,
    _unforced_flow,
    damping_injection_certificate,
)
from chc.toeplitz import circulant_operator_norm, circulant_symbol

FORMAL_LINKS = {
    "chc.research.deep_galerkin": {
        "cites": ("lq_mean_field.mac", "lq_mean_field.v"),
        "tests": (
            "test_lq_closed_form_solves_the_two_point_problem",
            "test_obstruction_horizon_is_the_first_zero_of_the_denominator",
            "test_no_obstruction_below_the_gain_threshold",
        ),
    },
    "chc.research.deep_galerkin.CongestedMeanFieldGame": {
        "cites": ("nonlinear_dwr.mac",),
        "tests": (
            "test_congested_solve_closes_the_terminal_row",
            "test_congested_game_at_zero_congestion_is_the_lq_game",
        ),
    },
    "chc.research.deep_galerkin.LQMeanFieldGame.obstruction_horizon": {
        "cites": ("lq_mean_field.v",),
        "tests": (
            "test_obstruction_horizon_is_the_first_zero_of_the_denominator",
            "test_no_obstruction_below_the_gain_threshold",
        ),
    },
    "chc.research.deep_galerkin.MeanFieldSolution.finite_population_rms": {
        "cites": ("finite_population_gap.mac",),
        "tests": (
            "test_finite_population_rms_is_the_closed_loop_variance_over_n",
            "test_finite_population_rms_prices_a_simulated_population",
        ),
    },
    "chc.research.deep_galerkin.adjoint_weighted_error": {
        "cites": ("nonlinear_dwr.mac",),
        "tests": (
            "test_adjoint_weighted_error_is_exact_for_an_affine_field",
            "test_adjoint_weighted_error_remainder_is_second_order",
        ),
    },
    "chc.research.deep_galerkin.dual_weighted_error_estimate": {
        "cites": ("mean_field_dwr.mac", "mean_field_dwr.v"),
        "tests": (
            "test_dual_weighted_estimate_is_the_exact_error_of_an_affine_path",
            "test_dual_weighted_estimate_off_the_riccati_root_is_the_model_rate_s_error",
        ),
    },
    "chc.research.deep_galerkin.finite_population_gap_certificate": {
        "cites": ("finite_population_gap.mac",),
        "tests": ("test_finite_population_gap_constants_match_their_odes",),
    },
    "chc.research.deep_galerkin.residual_blindness_sweep": {
        "cites": ("mean_field_dwr.v",),
        "tests": ("test_blindness_sweep_error_outgrows_a_bounded_residual",),
    },
    "chc.research.galerkin": {
        "cites": ("convection_diffusion.mac", "convection_diffusion.v"),
        "tests": (
            "test_petrov_galerkin_nodes_are_the_discrete_mode",
            "test_bubnov_galerkin_alternates_exactly_above_unit_peclet",
            "test_full_upwind_amplification_is_one_plus_two_peclet",
            "test_optimal_upwind_is_nodally_exact",
        ),
    },
    "chc.research.galerkin.convection_diffusion_certificate": {
        "cites": ("convection_diffusion.mac", "convection_diffusion.v"),
        "tests": (
            "test_petrov_galerkin_nodes_are_the_discrete_mode",
            "test_bubnov_galerkin_alternates_exactly_above_unit_peclet",
            "test_full_upwind_amplification_is_one_plus_two_peclet",
            "test_optimal_upwind_is_nodally_exact",
            "test_galerkin.py::test_convection_diffusion_certificate_locates_the_threshold_and_prices_the_cure",
        ),
    },
    "chc.research.galerkin.convection_diffusion_exact": {
        "cites": ("convection_diffusion.mac",),
        "tests": (
            "test_convection_diffusion_exact_solves_the_boundary_value_problem",
            "test_convection_diffusion_exact_stays_bounded_for_a_reversed_flow",
            "test_convection_diffusion_exact_keeps_its_digits_when_diffusion_dominates",
        ),
    },
    "chc.research.galerkin.optimal_upwind": {
        "cites": ("convection_diffusion.mac",),
        "tests": (
            "test_optimal_upwind_hits_the_tanh_target_inside_the_stable_range",
            "test_optimal_upwind_matches_high_precision_values",
            "test_optimal_upwind_keeps_double_precision_above_the_series_crossover",
            "test_optimal_upwind_is_nodally_exact",
        ),
    },
    "chc.games.congestion_contraction_modulus": {
        "cites": ("congestion_contraction.mac", "congestion_contraction.v"),
        "tests": (
            "test_congestion_jacobian_norm_stays_under_the_certified_bound",
            "test_congestion_bound_is_attained_at_the_two_point_softmax",
            "test_congestion_map_contracts_pairs_at_the_certified_rate",
        ),
    },
    "chc.games.exact_congestion_allocation": {
        "cites": ("exact_congestion_allocation.mac",),
        "tests": (
            "test_games.py::test_no_split_of_the_budget_completes_more_than_the_exact_allocation",
            "test_games.py::test_the_exact_allocation_carries_its_certificate",
        ),
    },
    "chc.marketplace._polish": {
        "cites": ("relative_offset_stop.mac",),
        "tests": (
            "test_marketplace.py::test_the_fit_converges_where_no_gauss_newton_step_is_left",
        ),
        "via": "fit_equilibrium",
    },
    "chc.research.games.equilibrium_monotonicity_certificate": {
        "cites": ("equilibrium_monotonicity.mac", "equilibrium_monotonicity.v"),
        "tests": (
            "test_congestion_residual_map_is_one_strongly_monotone_on_pairs",
            "test_congestion_tangent_modulus_meets_its_bound",
            "test_equilibrium_monotonicity_certificate_moves_no_more_than_the_push",
        ),
    },
    "chc.residual.ContractiveResidual": {
        "cites": ("contractive_euler.v",),
        "tests": (
            "test_contractive_residual_log_norm_and_lipschitz_hold",
            "test_contractive_residual_euler_step_contracts_by_the_factor",
        ),
    },
    "chc.residual.PortHamiltonianResidual": {
        "cites": ("convex_port_hamiltonian.mac",),
        "tests": (
            "test_port_hamiltonian_energy_rate_is_supply_minus_dissipation",
            "test_convex_port_hamiltonian_flow_stays_in_the_invariant_ball",
            "test_mlp_port_hamiltonian_energy_is_bounded_with_no_radius",
        ),
    },
    "chc.residual.SpectralResidual": {
        "cites": ("spectral_circulant.mac", "spectral_circulant.v"),
        "tests": (
            "test_spectral_residual_norm_is_attained_on_a_fourier_mode",
            "test_spectral_residual_is_translation_equivariant",
            "test_spectral_residual_gains_multiply_under_composition",
        ),
    },
    "chc.residual._ConvexEnergy": {
        "cites": ("convex_port_hamiltonian.mac",),
        "tests": ("test_convex_energy_is_strongly_convex_above_its_floor",),
    },
    "chc.residual._critical_points": {
        "cites": ("convex_port_hamiltonian.v",),
        "tests": ("test_convex_energy_has_one_critical_point",),
    },
    "chc.residual.convex_energy_certificate": {
        "cites": ("convex_port_hamiltonian.mac", "convex_port_hamiltonian.v"),
        "tests": (
            "test_port_hamiltonian_energy_rate_is_supply_minus_dissipation",
            "test_convex_energy_is_strongly_convex_above_its_floor",
            "test_convex_port_hamiltonian_flow_stays_in_the_invariant_ball",
            "test_mlp_port_hamiltonian_energy_is_bounded_with_no_radius",
            "test_convex_energy_has_one_critical_point",
            "test_residual_backbones.py::test_a_convex_energy_turns_the_passivity_inequality_into_an_invariant_ball",
        ),
    },
    "chc.residual.damping_injection_certificate": {
        "cites": ("port_hamiltonian_lyapunov.v",),
        "tests": (
            "test_damping_injection_rate_adds_the_port_dissipation",
            "test_damping_injection_certificate_rollout_loses_energy",
        ),
    },
    "chc.toeplitz.circulant_operator_norm": {
        "cites": ("spectral_circulant.v",),
        "tests": ("test_circulant_operator_norm_is_the_dense_spectral_norm",),
    },
    "chc.toeplitz.circulant_symbol": {
        "cites": ("spectral_circulant.mac",),
        "tests": ("test_circulant_symbol_holds_the_dense_eigenvalues",),
    },
    "chc.research.transport.advection_diffusion_symbol": {
        "cites": ("spectral_circulant.mac",),
        "via": "advection_diffusion_field",
        "tests": (
            "test_advection_diffusion_field_differentiates_trigonometric_fields",
            "test_advection_diffusion_field_has_no_first_derivative_at_nyquist",
        ),
    },
}

_EPS = float(np.finfo(np.float64).eps)

# ---------------------------------------------------------------------------------------------
# chc.research.galerkin: the cell-Peclet threshold, full upwind and the optimal Petrov-Galerkin
# parameter.
# ---------------------------------------------------------------------------------------------


def _effective_peclet(peclet: float, alpha: float) -> float:
    return peclet / (1.0 + alpha * abs(peclet))


def _mode(amplification: float, n: int) -> np.ndarray:
    """``u_i = A + B r^i`` with ``u_0 = 0`` and ``u_n = 1``."""
    powers = amplification ** np.arange(n + 1, dtype=np.float64)
    return (powers - 1.0) / (powers[-1] - 1.0)


def _solve_at_peclet(peclet: float, n: int, alpha: float, flow: float) -> np.ndarray:
    h = 1.0 / n
    eps = abs(flow) * h / (2.0 * peclet)
    return np.asarray(_scheme(eps, flow, n, alpha=alpha)[1])


# One compile per mesh size: eager, every call re-traces the Thomas sweep's scans.
_scheme = jax.jit(convection_diffusion_1d, static_argnums=2)
_meshes = st.sampled_from((5, 8, 13, 24))


def _layer_solution(x: np.ndarray, ratio: float) -> np.ndarray:
    """``(exp(r x) - 1)/(exp(r) - 1)`` in whichever form cannot overflow for this sign of ``r``."""
    if ratio < 600.0:
        return np.expm1(ratio * x) / np.expm1(ratio)
    return (np.exp(ratio * (x - 1.0)) - np.exp(-ratio)) / -np.expm1(-ratio)


@settings(max_examples=15, deadline=None)
@given(
    peclet=st.floats(0.05, 4.0),
    alpha=st.floats(0.0, 1.0),
    n=st.sampled_from((5, 8, 13)),
    flow=st.sampled_from((1.0, -1.0, 2.5)),
)
def test_petrov_galerkin_nodes_are_the_discrete_mode(
    peclet: float, alpha: float, n: int, flow: float
) -> None:
    """convection_diffusion.v (amp_is_a_root): the nodes are A + B r^i, r = (1+Pe_t)/(1-Pe_t)."""
    signed = math.copysign(_effective_peclet(peclet, alpha), flow)
    assume(abs(abs(signed) - 1.0) > 0.15)
    amplification = (1.0 + signed) / (1.0 - signed)
    nodes = _solve_at_peclet(peclet, n, alpha, flow)
    np.testing.assert_allclose(nodes, _mode(amplification, n), rtol=0.0, atol=1e-11)


@settings(max_examples=15, deadline=None)
@given(
    peclet=st.floats(0.05, 6.0),
    n=_meshes,
    flow=st.sampled_from((1.0, -1.0)),
)
def test_bubnov_galerkin_alternates_exactly_above_unit_peclet(
    peclet: float, n: int, flow: float
) -> None:
    """convection_diffusion.v (oscillation_above_threshold, monotone_below_threshold)."""
    assume(abs(peclet - 1.0) > 0.02)
    steps = np.diff(_solve_at_peclet(peclet, n, 0.0, flow))
    # a difference below the rounding of u ~ 1 carries no sign, so only resolved pairs are read
    resolved = (np.abs(steps[:-1]) > 1e-12) & (np.abs(steps[1:]) > 1e-12)
    products = (steps[:-1] * steps[1:])[resolved]
    assert products.size > 0
    if peclet > 1.0:
        assert np.all(products < 0.0)
    else:
        assert np.all(products >= 0.0)
    upwind_steps = np.diff(_solve_at_peclet(peclet, n, 1.0, flow))
    assert np.all(upwind_steps[:-1] * upwind_steps[1:] >= 0.0)


@settings(max_examples=15, deadline=None)
@given(peclet=st.floats(0.05, 20.0), n=_meshes)
def test_full_upwind_amplification_is_one_plus_two_peclet(peclet: float, n: int) -> None:
    """convection_diffusion.v (upwind_amplification, upwind_first_order): r = 1 + 2Pe < e^(2Pe)."""
    steps = np.diff(_solve_at_peclet(peclet, n, 1.0, 1.0))
    keep = np.abs(steps[:-1]) > 1e-250
    ratios = steps[1:][keep] / steps[:-1][keep]
    np.testing.assert_allclose(ratios, 1.0 + 2.0 * peclet, rtol=1e-9, atol=0.0)
    assert 1.0 + 2.0 * peclet < math.exp(2.0 * peclet)


@settings(max_examples=15, deadline=None)
@given(
    peclet=st.floats(1e-3, 40.0),
    n=_meshes,
    flow=st.sampled_from((1.0, -1.0, 3.0)),
)
def test_optimal_upwind_is_nodally_exact(peclet: float, n: int, flow: float) -> None:
    """convection_diffusion.mac STEP 4/7: alpha = coth(Pe) - 1/Pe reproduces the exact nodes."""
    h = 1.0 / n
    eps = abs(flow) * h / (2.0 * peclet)
    nodes, u = _scheme(eps, flow, n, alpha=optimal_upwind(flow * h / (2.0 * eps)))
    exact = _layer_solution(np.asarray(nodes), flow / eps)
    np.testing.assert_allclose(np.asarray(u), exact, rtol=0.0, atol=64 * _EPS)


@settings(max_examples=40, deadline=None)
@given(peclet=st.floats(1e-6, 15.0))
def test_optimal_upwind_hits_the_tanh_target_inside_the_stable_range(peclet: float) -> None:
    """convection_diffusion.v (optimal_alpha_hits_target, _is_stable, _below_full_upwind)."""
    alpha = optimal_upwind(peclet)
    assert optimal_upwind(-peclet) == alpha
    if peclet < 1e-3:
        series = peclet / 3.0 - peclet**3 / 45.0 + 2.0 * peclet**5 / 945.0
        assert alpha == pytest.approx(series, rel=1e-14, abs=0.0)
    else:
        assert _effective_peclet(peclet, alpha) == pytest.approx(
            math.tanh(peclet), rel=1e-12, abs=0.0
        )
    assert 1.0 - 1.0 / peclet < alpha < 1.0


@pytest.mark.parametrize(
    ("peclet", "reference"),
    [
        (1e-4, 3.3333333311111111e-05),
        (0.5, 0.16395341373865285),
        (2.5, 0.61356730981260846),
        (30.0, 0.96666666666666667),
    ],
)
def test_optimal_upwind_matches_high_precision_values(peclet: float, reference: float) -> None:
    """convection_diffusion.mac STEP 4f: coth(Pe) - 1/Pe, against 50-digit PARI/GP values."""
    assert optimal_upwind(peclet) == pytest.approx(reference, rel=4 * _EPS, abs=0.0)


def test_optimal_upwind_keeps_double_precision_above_the_series_crossover() -> None:
    """convection_diffusion.mac STEP 4f, just above the crossover, against PARI/GP values."""
    for peclet, reference in (
        (1e-3, 3.3333331111111323e-04),
        (1.0797751623277097e-3, 3.5992502613311941e-04),
    ):
        assert optimal_upwind(peclet) == pytest.approx(reference, rel=1e-13, abs=0.0)


@settings(max_examples=15, deadline=None)
@given(ratio=st.floats(0.5, 2e4), flow=st.floats(0.1, 5.0))
def test_convection_diffusion_exact_solves_the_boundary_value_problem(
    ratio: float, flow: float
) -> None:
    """convection_diffusion.mac STEP 1: -eps u'' + s u' = 0, u(0) = 0, u(1) = 1, no overflow."""
    eps = flow / ratio

    def u(x: jax.Array) -> jax.Array:
        return convection_diffusion_exact(x, eps, flow)

    xs = jnp.linspace(0.0, 1.0, 41)
    slope = jax.vmap(jax.grad(u))(xs)
    curvature = jax.vmap(jax.grad(jax.grad(u)))(xs)
    residual = np.asarray(-eps * curvature + flow * slope)
    scale = np.asarray(flow * jnp.abs(slope))
    assert np.all(np.abs(residual) <= 1e-12 * scale + 1e-300)
    values = np.asarray(u(xs))
    assert values[0] == pytest.approx(0.0, abs=4 * _EPS)
    assert values[-1] == pytest.approx(1.0, rel=0.0, abs=4 * _EPS)
    assert np.all(np.isfinite(values))
    assert np.all(np.diff(values) >= 0.0)
    if ratio < 600.0:
        naive = np.expm1(ratio * np.asarray(xs)) / np.expm1(ratio)
        np.testing.assert_allclose(values, naive, rtol=0.0, atol=8 * _EPS)


def test_convection_diffusion_exact_stays_bounded_for_a_reversed_flow() -> None:
    """convection_diffusion.mac STEP 1, reversed flow: u = expm1(rx)/expm1(r), bounded at r < 0."""
    xs = np.linspace(0.0, 1.0, 25)
    values = np.asarray(convection_diffusion_exact(jnp.asarray(xs), 1.0 / 720.0, -1.0))
    np.testing.assert_allclose(values, _layer_solution(xs, -720.0), rtol=0.0, atol=8 * _EPS)


def test_convection_diffusion_exact_keeps_its_digits_when_diffusion_dominates() -> None:
    """convection_diffusion.mac STEP 1d at s/eps = 1e-8, against 50-digit PARI/GP values."""
    xs = jnp.asarray([0.25, 0.5, 0.9])
    values = np.asarray(convection_diffusion_exact(xs, 1e8, 1.0))
    reference = [2.4999999906250000e-01, 4.9999999875000000e-01, 8.9999999955000000e-01]
    np.testing.assert_allclose(values, reference, rtol=1e-13, atol=0.0)
    np.testing.assert_array_equal(np.asarray(convection_diffusion_exact(xs, 1.0, 0.0)), xs)


# ---------------------------------------------------------------------------------------------
# chc.research.deep_galerkin: the LQ mean-field closed form and its obstruction.
# ---------------------------------------------------------------------------------------------


def _game(**fields: float) -> LQMeanFieldGame:
    base = {
        "a": -0.5,
        "b": 1.0,
        "q": 1.0,
        "r": 1.0,
        "coupling": 0.5,
        "terminal_coupling": 0.5,
        "sigma": 0.7,
        "horizon": 1.0,
        "mean_initial": 1.0,
        "variance_initial": 0.25,
    }
    return LQMeanFieldGame(**(base | fields))


def _reduced_matrix(game: LQMeanFieldGame) -> tuple[np.ndarray, float]:
    """``M`` of the (m, S) system, built from the plant data and the Riccati quadratic alone."""
    roots = np.roots([game.b**2 / game.r, -2.0 * game.a, -game.q])
    p_bar = float(roots.real.max())
    rate = game.a - game.b**2 * p_bar / game.r
    return np.array([[rate, -(game.b**2) / game.r], [game.q * game.coupling, -rate]]), p_bar


def _expm_denominator(game: LQMeanFieldGame, horizon: float) -> float:
    matrix, p_bar = _reduced_matrix(game)
    e = expm(matrix * horizon)
    return float(e[1, 1] + p_bar * game.terminal_coupling * e[0, 1])


_plant = {
    "a": st.floats(-2.0, 2.0),
    "b": st.floats(0.3, 2.0),
    "q": st.floats(0.2, 3.0),
    "r": st.floats(0.2, 3.0),
}


@settings(max_examples=15, deadline=None)
@given(
    **_plant,
    coupling=st.floats(-1.0, 6.0),
    terminal_coupling=st.floats(-1.0, 3.0),
    horizon=st.floats(0.1, 2.0),
    mean_initial=st.floats(-2.0, 2.0).filter(lambda m: abs(m) > 0.1),
)
def test_lq_closed_form_solves_the_two_point_problem(**fields: float) -> None:
    """lq_mean_field.mac STEPs 3-5: y' = M y from (m0, S0) meets S(T) = -P c_T m(T)."""
    game = _game(**fields)
    assume(abs(_expm_denominator(game, game.horizon)) > 0.05)
    matrix, p_bar = _reduced_matrix(game)
    assert game.riccati_root == pytest.approx(p_bar, rel=1e-12, abs=0.0)
    solution = game.solve(n_time=101)
    start = np.array([solution.mean[0], solution.value_s[0]])
    path = solve_ivp(
        lambda _, y: matrix @ y,
        (0.0, game.horizon),
        start,
        t_eval=solution.times,
        rtol=1e-12,
        atol=1e-14,
    )
    scale = np.abs(path.y).max()
    np.testing.assert_allclose(path.y[0], solution.mean, rtol=0.0, atol=1e-9 * scale)
    np.testing.assert_allclose(path.y[1], solution.value_s, rtol=0.0, atol=1e-9 * scale)
    terminal = path.y[1, -1] + p_bar * game.terminal_coupling * path.y[0, -1]
    assert abs(terminal) <= 1e-9 * scale
    variance = solve_ivp(
        lambda _, v: 2.0 * matrix[0, 0] * v + game.sigma**2,
        (0.0, game.horizon),
        [game.variance_initial],
        t_eval=solution.times,
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(solution.variance, variance.y[0], rtol=1e-9, atol=0.0)


def _game_with_gain(fields: dict[str, float], gap: float) -> LQMeanFieldGame | None:
    """A game on the real branch whose obstruction gain is ``k = lam + gap``, or None."""
    game = _game(**fields)
    lam_sq = game.lambda_squared
    if lam_sq <= 0.01:
        return None
    target = math.sqrt(lam_sq) + gap
    # k = A + P c_T b^2/r is affine in c_T, so the terminal coupling that gives k is explicit.
    slope = game.riccati_root * game.b**2 / game.r
    return replace(game, terminal_coupling=(target - game.closed_loop_rate) / slope)


@settings(max_examples=30, deadline=None)
@given(
    **_plant,
    above=st.floats(0.05, 4.0),
    gap=st.floats(0.05, 3.0),
    terminal_coupling=st.floats(-1.0, 3.0),
    oscillatory=st.booleans(),
)
def test_obstruction_horizon_is_the_first_zero_of_the_denominator(
    above: float, gap: float, terminal_coupling: float, oscillatory: bool, **plant: float
) -> None:
    """lq_mean_field.v (den_real_vanishes_at_obstruction, den_osc_vanishes_..., _in_range)."""
    threshold = 1.0 + plant["r"] * plant["a"] ** 2 / (plant["q"] * plant["b"] ** 2)
    if oscillatory:
        game = _game(**plant, coupling=threshold + above, terminal_coupling=terminal_coupling)
    else:
        built = _game_with_gain(plant | {"coupling": threshold - above}, gap)
        assume(built is not None)
        game = cast("LQMeanFieldGame", built)
    horizon = game.obstruction_horizon()
    assert 0.0 < horizon < math.inf
    if oscillatory:
        assert horizon < math.pi / math.sqrt(-game.lambda_squared)
    matrix, p_bar = _reduced_matrix(game)
    e = expm(matrix * horizon)
    size = np.abs(e).max() * (1.0 + abs(p_bar * game.terminal_coupling))
    assert abs(_expm_denominator(game, horizon)) <= 1e-9 * size
    before = [_expm_denominator(game, t) for t in np.linspace(0.0, horizon, 60)[:-1]]
    assert min(before) > 0.0
    root = brentq(lambda t: _expm_denominator(game, t), 0.5 * horizon, 1.5 * horizon, xtol=1e-14)
    assert root == pytest.approx(horizon, rel=1e-9, abs=0.0)


@settings(max_examples=25, deadline=None)
@given(**_plant, below=st.floats(0.05, 4.0), gap=st.floats(0.01, 5.0))
def test_no_obstruction_below_the_gain_threshold(below: float, gap: float, **plant: float) -> None:
    """lq_mean_field.v (den_real_positive_below): k < lam keeps den > 0 at every horizon."""
    threshold = 1.0 + plant["r"] * plant["a"] ** 2 / (plant["q"] * plant["b"] ** 2)
    built = _game_with_gain(plant | {"coupling": threshold - below}, -gap)
    assume(built is not None)
    game = cast("LQMeanFieldGame", built)
    assert game.obstruction_horizon() == math.inf
    lam = math.sqrt(game.lambda_squared)
    horizons = np.linspace(0.0, 30.0 / lam, 200)
    assert min(_expm_denominator(game, t) for t in horizons) > 0.0


# ---------------------------------------------------------------------------------------------
# chc.research.deep_galerkin: the finite-population gap.
# ---------------------------------------------------------------------------------------------


def _variance_ode(game: LQMeanFieldGame, times: np.ndarray) -> np.ndarray:
    rate = _reduced_matrix(game)[0][0, 0]
    path = solve_ivp(
        lambda _, v: 2.0 * rate * v + game.sigma**2,
        (0.0, float(times[-1])),
        [game.variance_initial],
        t_eval=times,
        rtol=1e-12,
        atol=1e-14,
    )
    return path.y[0]


@settings(max_examples=20, deadline=None)
@given(**_plant, sigma=st.floats(0.05, 2.0), n_agents=st.integers(1, 10_000))
def test_finite_population_rms_is_the_closed_loop_variance_over_n(
    sigma: float, n_agents: int, **plant: float
) -> None:
    """finite_population_gap.mac STEPs 1-2: rms(N, t) = sqrt(v(t)/N), v' = 2 A v + sigma^2."""
    solution = _game(**plant, sigma=sigma).solve(n_time=51)
    expected = np.sqrt(_variance_ode(solution.game, solution.times) / n_agents)
    np.testing.assert_allclose(
        solution.finite_population_rms(n_agents), expected, rtol=1e-9, atol=0.0
    )


def test_finite_population_rms_prices_a_simulated_population() -> None:
    """finite_population_gap.mac STEP 2b: E[(m_N - m)^2] = v/N, against Euler-Maruyama agents."""
    solution = _game().solve()
    replicates, n_agents = 400, 32
    times, paths = solution.simulate_population(n_agents, replicates=replicates, seed=3, n_step=200)
    for index in (100, 200):
        gap = paths[:, index] - np.interp(times[index], solution.times, solution.mean)
        measured = math.sqrt(float(np.mean(gap**2)))
        predicted = float(
            np.interp(times[index], solution.times, solution.finite_population_rms(n_agents))
        )
        # the RMS of R squared Gaussians has relative SD sqrt(2/R)/2; Euler adds |A| dt ~ 0.6 %
        sd = math.sqrt(2.0 / replicates) / 2.0
        assert abs(measured / predicted - 1.0) < 4.0 * sd + 0.01


def test_finite_population_gap_constants_match_their_odes() -> None:
    """finite_population_gap.mac STEPs 2-3, 5: the gap, its exponent and the coupled bias."""
    replicates = 100
    gap = finite_population_gap_certificate(replicates=replicates, n_step=100)
    game = _game()
    matrix, _ = _reduced_matrix(game)
    start = game.solve().value_s[0]
    terminal_variance = _variance_ode(game, np.array([0.0, game.horizon]))[-1]
    expected = [math.sqrt(terminal_variance / n) for n in gap.sizes]
    np.testing.assert_allclose(gap.closed_form_rms, expected, rtol=1e-9, atol=0.0)

    sd = math.sqrt(2.0 / (replicates - 1)) / 2.0
    ratios = np.asarray(gap.measured_rms) / np.asarray(gap.closed_form_rms)
    assert np.all(np.abs(ratios - 1.0) < 4.0 * sd + 0.02)
    assert abs(gap.measured_exponent + 0.5) < 4.0 * gap.exponent_standard_error
    assert gap.common_shock_exponent > -0.25

    gain = 6.0  # the certificate's kappa = c/N arm: N (m_N - m) -> c int e^{A(T-s)} m(s) ds
    bias = solve_ivp(
        lambda _, y: np.array([*(matrix @ y[:2]), matrix[0, 0] * y[2] + gain * y[0]]),
        (0.0, game.horizon),
        [game.mean_initial, start, 0.0],
        rtol=1e-12,
        atol=1e-14,
    ).y[2, -1]
    assert gap.predicted_bias_constant == pytest.approx(bias, rel=1e-4, abs=0.0)


# ---------------------------------------------------------------------------------------------
# chc.research.deep_galerkin: the dual-weighted estimators.
# ---------------------------------------------------------------------------------------------


class _QuadraticPath:
    """Model-shaped: ``V = P x^2/2 + S_hat(t) x`` and a polynomial mean with ``m(0) = m0``."""

    def __init__(self, curvature: float, mean: tuple[float, ...], slope: tuple[float, ...]):
        self.curvature, self.mean_coef, self.slope_coef = curvature, mean, slope

    def mean(self, t: jax.Array) -> jax.Array:
        return sum(c * t**k for k, c in enumerate(self.mean_coef))

    def value_at(self, t: jax.Array, x: jax.Array, mean_terminal: jax.Array) -> jax.Array:
        slope = sum(c * t**k for k, c in enumerate(self.slope_coef))
        return 0.5 * self.curvature * x**2 + slope * x


@settings(max_examples=6, deadline=None)
@given(
    coupling=st.floats(-0.5, 2.5),
    terminal_coupling=st.floats(-0.5, 2.0),
    horizon=st.floats(0.3, 1.2),
    offset=st.floats(0.1, 2.0),
    sign=st.sampled_from((1.0, -1.0)),
    mean_shape=st.tuples(st.floats(-2.0, 2.0), st.floats(-2.0, 2.0)),
    slope_shape=st.tuples(st.floats(-2.0, 2.0), st.floats(-2.0, 2.0)),
)
def test_dual_weighted_estimate_is_the_exact_error_of_an_affine_path(
    coupling: float,
    terminal_coupling: float,
    horizon: float,
    offset: float,
    sign: float,
    mean_shape: tuple[float, float],
    slope_shape: tuple[float, float],
) -> None:
    """mean_field_dwr.v (error_is_defect_over_den): S_hat(0) - S(0) = d / den(T), exactly. The
    float code integrates the defect on a grid, with an error second order in its step that grows
    with the path's curvature (1.3e-6 on 401 points for an offset of 0.125 and a slope ``t^2``);
    Richardson's extrapolation over 401 and 1601 points removes it, and leaves the identity."""
    game = _game(coupling=coupling, terminal_coupling=terminal_coupling, horizon=horizon)
    assume(abs(game.fixed_point_denominator()) > 0.05)
    truth = float(game.solve().value_s[0])
    path = _QuadraticPath(
        game.riccati_root,
        (game.mean_initial, *mean_shape),
        (truth + sign * offset, *slope_shape),
    )
    coarse, fine = (
        dual_weighted_error_estimate(game, cast("MeanFieldDGM", path), n_time=n)
        for n in (401, 1601)
    )
    assert (16.0 * fine - coarse) / 15.0 == pytest.approx(offset, rel=1e-7, abs=0.0)


def _model_rate_slope(game, curvature: float) -> float:
    """``S~(0)`` of the two-point problem at the model's own rate ``a - b^2 V_xx/r``: ``y' = M~ y``,
    ``m(0) = m0``, ``P c m(T) + S(T) = 0``, by the matrix exponential."""
    rate = game.a - game.b**2 * curvature / game.r
    reduced = np.array([[rate, -(game.b**2) / game.r], [game.q * game.coupling, -rate]])
    phi = expm(reduced * game.horizon)
    functional = np.array([game.riccati_root * game.terminal_coupling, 1.0])
    return -game.mean_initial * (functional @ phi[:, 0]) / (functional @ phi[:, 1])


@pytest.mark.parametrize("curvature_error", [0.05, -0.1, 0.3])
def test_dual_weighted_estimate_off_the_riccati_root_is_the_model_rate_s_error(
    curvature_error: float,
) -> None:
    """mean_field_dwr.v (error_is_defect_over_den) read at the model's own rate: with V_xx = P + e
    the estimate is |S_hat(0) - S~(0)|, S~ the problem solved at a - b^2 (P + e)/r, not the error.
    On the anti-monotone game at T = 0.7 with e = 0.05 and S_hat off by 0.13 it reads 2.509."""
    game = replace(_ANTI_MONOTONE_GAME, horizon=0.7)
    truth = float(game.solve().value_s[0])
    curvature = game.riccati_root + curvature_error
    path = _QuadraticPath(curvature, (game.mean_initial,), (truth + 0.13,))
    estimate = dual_weighted_error_estimate(game, cast("MeanFieldDGM", path))
    # RK4 transition and trapezoid quadrature at 401 nodes: 1.7e-5 relative at e = 0.05
    assert estimate == pytest.approx(
        abs(truth + 0.13 - _model_rate_slope(game, curvature)), rel=1e-4, abs=0.0
    )
    if curvature_error == 0.05:
        assert estimate == pytest.approx(2.5094, rel=1e-4, abs=0.0)


def _affine_problem(
    matrix: np.ndarray, row: np.ndarray, start: float, horizon: float, n_time: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """The exact solution of ``y' = M y``, ``y(0)[0] = start``, ``row . y(T) = 0``, on a grid."""
    phi = expm(matrix * horizon)
    slope = -start * (row @ phi[:, 0]) / (row @ phi[:, 1])
    times = np.linspace(0.0, horizon, n_time)
    path = np.stack([expm(matrix * t) @ np.array([start, slope]) for t in times])
    return times, path, float(slope)


def _bump(times: np.ndarray, horizon: float) -> tuple[np.ndarray, np.ndarray]:
    s = times / horizon
    shape = np.stack([s * np.cos(2.0 * s), 0.7 - s + s**2], axis=1)
    rate = np.stack([np.cos(2.0 * s) - 2.0 * s * np.sin(2.0 * s), -1.0 + 2.0 * s], axis=1)
    return shape, rate / horizon


@settings(max_examples=8, deadline=None)
@given(
    entries=st.tuples(*[st.floats(-1.5, 1.5)] * 4),
    row=st.tuples(st.floats(-2.0, 2.0), st.floats(0.5, 2.0)),
    size=st.floats(1e-3, 3.0),
)
@example(entries=(1.5, 1.0, 0.0, 0.0), row=(-0.234375, 0.75), size=0.00390625)
@example(entries=(1.5, -0.5625, 1.5, 1.0), row=(1.0, 1.0), size=1.0)
def test_adjoint_weighted_error_is_exact_for_an_affine_field(
    entries: tuple[float, float, float, float], row: tuple[float, float], size: float
) -> None:
    """nonlinear_dwr.mac STEPs 2-4e: on an affine field the formula has no remainder at any eta."""
    matrix, terminal_row, horizon = np.array(entries).reshape(2, 2), np.array(row), 1.0
    phi = expm(matrix * horizon)
    assume(abs(terminal_row @ phi[:, 1]) > 0.1)
    times, path, _ = _affine_problem(matrix, terminal_row, 1.0, horizon, 401)
    shape, shape_rate = _bump(times, horizon)
    operator = jnp.asarray(matrix)

    def estimate(eta: float) -> float:
        return adjoint_weighted_error(
            lambda y: operator @ y,
            times,
            path + eta * shape,
            path @ matrix.T + eta * shape_rate,
            terminal_row,
        )

    # no remainder: the estimate is linear in eta to rounding, quadrature error included. The
    # defect y' - M y rounds at about 1e-15 of the path's scale per node, which is a relative
    # 1e-13/eta of the estimate at the small eta: 2.6e-8 at eta = 3.9e-6
    small = 1e-3 * size
    assert estimate(size) / size == pytest.approx(
        estimate(small) / small, rel=1e-9 + 1e-13 / small, abs=0.0
    )
    # and the 401-point trapezoid's error is O(h^2) with a constant that grows with the field's
    # entries (1.3e-4 relative at entries (1.5, -0.5625, 1.5, 1.0)): Richardson extrapolation
    # against a 1601-point grid removes the h^2 term and leaves O(h^4)
    fine_times, fine_path, _ = _affine_problem(matrix, terminal_row, 1.0, horizon, 1601)
    fine_shape, fine_rate = _bump(fine_times, horizon)
    fine = adjoint_weighted_error(
        lambda y: operator @ y,
        fine_times,
        fine_path + size * fine_shape,
        fine_path @ matrix.T + size * fine_rate,
        terminal_row,
    )
    assert (16.0 * fine - estimate(size)) / 15.0 == pytest.approx(
        size * shape[0, 1], rel=1e-7, abs=0.0
    )


def test_adjoint_weighted_error_remainder_is_second_order() -> None:
    """nonlinear_dwr.mac STEP 4f: a cubic field leaves an O(eta^2) remainder, not zero."""
    rate, transfer, weight, cubic, horizon = -1.1, 1.0, 0.5, 0.8, 1.0
    row = np.array([0.6, 1.0])

    def field(y: jax.Array) -> jax.Array:
        return jnp.stack(
            [rate * y[0] - transfer * y[1], weight * y[0] + cubic * y[0] ** 3 - rate * y[1]]
        )

    def reference(_: float, y: np.ndarray) -> np.ndarray:
        return np.array(
            [rate * y[0] - transfer * y[1], weight * y[0] + cubic * y[0] ** 3 - rate * y[1]]
        )

    def flow(slope: float, t_eval: np.ndarray | None = None) -> np.ndarray:
        return solve_ivp(
            reference,
            (0.0, horizon),
            [1.0, slope],
            t_eval=t_eval,
            rtol=1e-12,
            atol=1e-14,
        ).y

    slope = brentq(lambda s: float(row @ flow(s)[:, -1]), -5.0, 5.0, xtol=1e-14)
    times = np.linspace(0.0, horizon, 401)
    path = flow(slope, times).T
    path_rate = np.asarray(jax.vmap(field)(jnp.asarray(path)))
    shape, shape_rate = _bump(times, horizon)
    remainders = []
    for size in (0.08, 0.04, 0.02):
        estimate = adjoint_weighted_error(
            field, times, path + size * shape, path_rate + size * shape_rate, row
        )
        remainders.append(abs(estimate - size * shape[0, 1]))
    assert remainders[0] > 1e-3 * 0.08 * abs(shape[0, 1])
    for coarse, fine in itertools.pairwise(remainders):
        assert 3.5 < coarse / fine < 4.5


def _congested_rhs(game: CongestedMeanFieldGame, y: np.ndarray) -> np.ndarray:
    base = game.base
    rate = -math.sqrt(base.a**2 + base.q * base.b**2 / base.r)
    target = base.coupling * y[0] + game.congestion * y[0] ** 3
    return np.array([rate * y[0] - base.b**2 / base.r * y[1], base.q * target - rate * y[1]])


@pytest.mark.parametrize("congestion", [0.3, 0.6, -0.2])
def test_congested_solve_closes_the_terminal_row(congestion: float) -> None:
    """nonlinear_dwr.mac STEP 1: (m, S) solve m' = A m - (b^2/r) S, S' = q(c m + g m^3) - A S."""
    game = CongestedMeanFieldGame(base=_game(), congestion=congestion)
    times, path = game.solve(n_time=201)
    replay = solve_ivp(
        lambda _, y: _congested_rhs(game, y),
        (0.0, game.base.horizon),
        path[0],
        t_eval=times,
        rtol=1e-12,
        atol=1e-14,
    ).y.T
    np.testing.assert_allclose(path, replay, rtol=0.0, atol=1e-8)
    assert path[0, 0] == game.base.mean_initial
    p_bar = _reduced_matrix(game.base)[1]
    assert abs(replay[-1, 1] + p_bar * game.base.terminal_coupling * replay[-1, 0]) < 1e-7


def test_congested_game_at_zero_congestion_is_the_lq_game() -> None:
    """nonlinear_dwr.mac STEP 4e: at gamma = 0 the congested reduction is the LQ one."""
    base = _game(coupling=1.5, terminal_coupling=0.8, horizon=0.9)
    _, path = CongestedMeanFieldGame(base=base, congestion=0.0).solve(n_time=301)
    closed = base.solve(n_time=301)
    np.testing.assert_allclose(path[:, 0], closed.mean, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(path[:, 1], closed.value_s, rtol=0.0, atol=1e-9)


def test_blindness_sweep_error_outgrows_a_bounded_residual() -> None:
    """mean_field_dwr.v (bounded_approximator_is_blind, error_exceeds_amplitude_gap)."""
    # steps=0 keeps the network at its initialisation: one bounded approximator at both horizons
    sweep = residual_blindness_sweep((0.5, 0.79), width=8, steps=0)
    game = _game(coupling=3.0, terminal_coupling=3.0)
    for horizon, den, exact, fitted, error in zip(
        sweep.horizons,
        sweep.denominators,
        sweep.exact_slopes,
        sweep.fitted_slopes,
        sweep.errors,
        strict=True,
    ):
        assert den == pytest.approx(_expm_denominator(game, horizon), rel=1e-9, abs=0.0)
        assert error >= abs(exact) - abs(fitted) - 1e-12
    # den(T) shrinks by ~20x towards T* = 0.8036; the exact slope grows with 1/den while the
    # bounded model's slope does not, so the error outgrows any fixed multiple of the residual.
    assert sweep.denominators[1] < 0.1 * sweep.denominators[0]
    error_growth = sweep.errors[1] / sweep.errors[0]
    residual_growth = sweep.residuals[1] / sweep.residuals[0]
    assert error_growth > 10.0
    assert error_growth > 10.0 * residual_growth
    assert abs(sweep.fitted_slopes[1]) < 0.1 * abs(sweep.exact_slopes[1])


# ---------------------------------------------------------------------------------------------
# chc.games: the logit congestion map.
# ---------------------------------------------------------------------------------------------


def _congestion_map(
    attract: np.ndarray, beta: float, congestion: float, damping: float, mass: float
) -> object:
    weights = jnp.asarray(attract)

    def step(x: jax.Array) -> jax.Array:
        value = beta * (weights - congestion * x / mass)
        return (1.0 - damping) * x + damping * mass * jax.nn.softmax(value)

    return step


_game_box = {
    "beta": st.floats(0.1, 5.0),
    "congestion": st.floats(0.0, 4.0),
    "damping": st.floats(0.01, 1.0),
    "seed": st.integers(0, 2**16),
    "n": st.sampled_from((3, 6)),
}


@settings(max_examples=10, deadline=None)
@given(**_game_box)
def test_congestion_jacobian_norm_stays_under_the_certified_bound(
    beta: float, congestion: float, damping: float, seed: int, n: int
) -> None:
    """congestion_contraction.v (operator_norm_bound): ||T'(x)||_2 <= 1 - mu at every x."""
    rng = np.random.default_rng(seed)
    mass = float(n)
    step = _congestion_map(rng.normal(size=n), beta, congestion, damping, mass)
    points = rng.uniform(-3.0 * mass, 3.0 * mass, size=(16, n))
    jacobians = np.asarray(jax.vmap(jax.jacobian(step))(jnp.asarray(points)))
    norms = np.linalg.norm(jacobians, ord=2, axis=(1, 2))
    bound = 1.0 - congestion_contraction_modulus(beta, congestion, damping)
    assert np.all(norms <= bound + 1e-12)


@settings(max_examples=10, deadline=None)
@given(**_game_box)
def test_congestion_bound_is_attained_at_the_two_point_softmax(
    beta: float, congestion: float, damping: float, seed: int, n: int
) -> None:
    """congestion_contraction.v (popoviciu_half, boundary_is_sharp): s = (1/2, 1/2, 0..) binds."""
    del seed
    kappa = beta * congestion
    probs = np.zeros(n)
    probs[:2] = 0.5
    jacobian = (1.0 - damping) * np.eye(n) - damping * kappa * (
        np.diag(probs) - np.outer(probs, probs)
    )
    modulus = congestion_contraction_modulus(beta, congestion, damping)
    assert np.linalg.norm(jacobian, 2) == pytest.approx(1.0 - modulus, rel=1e-12, abs=1e-15)
    assert (modulus > 0.0) == (damping < 4.0 / (2.0 + kappa))
    assert (congestion_contraction_modulus(beta, congestion) > 0.0) == (kappa < 6.0)


@settings(max_examples=10, deadline=None)
@given(**_game_box)
def test_congestion_map_contracts_pairs_at_the_certified_rate(
    beta: float, congestion: float, damping: float, seed: int, n: int
) -> None:
    """congestion_contraction.v (modulus_positive): ||T(x) - T(y)|| <= (1 - mu) ||x - y||."""
    rng = np.random.default_rng(seed)
    mass = float(n)
    step = _congestion_map(rng.normal(size=n), beta, congestion, damping, mass)
    left, right = rng.uniform(0.0, mass, size=(2, 32, n))
    moved = np.asarray(jax.vmap(step)(jnp.asarray(left)) - jax.vmap(step)(jnp.asarray(right)))
    ratios = np.linalg.norm(moved, axis=1) / np.linalg.norm(left - right, axis=1)
    rate = 1.0 - congestion_contraction_modulus(beta, congestion, damping)
    assert np.all(ratios <= rate + 1e-12)


def _residual_map(attract: np.ndarray, beta: float, congestion: float, mass: float) -> object:
    weights = jnp.asarray(attract)

    def residual(x: jax.Array) -> jax.Array:
        return x - mass * jax.nn.softmax(beta * (weights - congestion * x / mass))

    return residual


@settings(max_examples=10, deadline=None)
@given(**_game_box)
def test_congestion_residual_map_is_one_strongly_monotone_on_pairs(
    beta: float, congestion: float, damping: float, seed: int, n: int
) -> None:
    """equilibrium_monotonicity.v (ambient_modulus_is_one, _attained): <dF, dx> >= |dx|^2."""
    del damping
    rng = np.random.default_rng(seed)
    mass = float(n)
    residual = _residual_map(rng.normal(size=n), beta, congestion, mass)
    left, right = rng.uniform(0.0, mass, size=(2, 64, n))
    gaps = left - right
    deltas = np.asarray(
        jax.vmap(residual)(jnp.asarray(left)) - jax.vmap(residual)(jnp.asarray(right))
    )
    moduli = np.sum(deltas * gaps, axis=1) / np.sum(gaps * gaps, axis=1)
    assert np.all(moduli >= 1.0 - 1e-12)
    shift = rng.normal()
    along = np.asarray(residual(jnp.asarray(left[0] + shift)) - residual(jnp.asarray(left[0])))
    np.testing.assert_allclose(along, shift, rtol=0.0, atol=1e-12)


@settings(max_examples=10, deadline=None)
@given(**_game_box)
def test_congestion_tangent_modulus_meets_its_bound(
    beta: float, congestion: float, damping: float, seed: int, n: int
) -> None:
    """equilibrium_monotonicity.v (tangent_bound): on 1-perp, F' >= 1 + kappa n s_min^2."""
    del damping
    rng = np.random.default_rng(seed)
    mass, kappa = float(n), beta * congestion
    attract = rng.normal(size=n)
    residual = _residual_map(attract, beta, congestion, mass)
    points = rng.uniform(0.0, mass, size=(16, n))
    jacobians = np.asarray(jax.vmap(jax.jacobian(residual))(jnp.asarray(points)))
    basis = np.linalg.qr(np.eye(n) - 1.0 / n)[0][:, : n - 1]
    projected = np.einsum(
        "ji,njk,kl->nil", basis, 0.5 * (jacobians + jacobians.transpose(0, 2, 1)), basis
    )
    logits = beta * (attract - congestion * points / mass)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    bounds = 1.0 + kappa * n * probs.min(axis=1) ** 2
    assert np.all(np.linalg.eigvalsh(projected)[:, 0] >= bounds - 1e-10)


def test_equilibrium_monotonicity_certificate_moves_no_more_than_the_push() -> None:
    """equilibrium_monotonicity.v (inverse_lipschitz_from_monotone): |x*(S+g) - x*(S)| <= |g|."""
    cert = equilibrium_monotonicity_certificate(samples=64, perturbations=6, seed=3)
    assert cert.ambient_modulus == pytest.approx(1.0, rel=0.0, abs=1e-9)
    assert cert.tangent_modulus >= cert.tangent_bound - 1e-9
    assert cert.local_tangent_modulus >= cert.local_tangent_bound - 1e-9
    assert cert.finite_monotonicity >= 1.0 - 1e-9
    assert cert.displacement_ratio <= 1.0 + 1e-6
    assert cert.ok


# ---------------------------------------------------------------------------------------------
# chc.residual: the structured backbones.
# ---------------------------------------------------------------------------------------------


@settings(max_examples=5, deadline=None)
@given(
    seed=st.integers(0, 2**16),
    state_dim=st.sampled_from((2, 3)),
    control_dim=st.sampled_from((0, 1)),
    rate=st.floats(0.1, 3.0),
    drift=st.floats(-3.0, 3.0),
)
def test_contractive_residual_log_norm_and_lipschitz_hold(
    seed: int, state_dim: int, control_dim: int, rate: float, drift: float
) -> None:
    """contractive_euler.v hypotheses: mu_2(J_r) <= -c < 0 and ||J_r||_2 <= L (|mu| <= L)."""
    key, k_drift, k_x, k_u = jax.random.split(jax.random.key(seed), 4)
    model = ContractiveResidual(state_dim, control_dim, width=8, rate=rate, key=key)
    margins = drift + jax.random.normal(k_drift, (state_dim,))
    model = eqx.tree_at(lambda m: m.log_drift, model, margins)
    xs = 3.0 * jax.random.normal(k_x, (12, state_dim))
    us = jax.random.normal(k_u, (12, control_dim))
    jacobians = np.asarray(jax.vmap(jax.jacobian(lambda x, u: model(0.0, x, u)))(xs, us))
    log_norms = np.linalg.eigvalsh(0.5 * (jacobians + jacobians.transpose(0, 2, 1)))[:, -1]
    contraction, lipschitz = float(model.contraction_rate()), float(model.lipschitz_constant())
    assert contraction > 0.0
    assert np.all(log_norms <= -contraction + 1e-12)
    assert np.all(np.linalg.norm(jacobians, ord=2, axis=(1, 2)) <= lipschitz + 1e-12)


@settings(max_examples=6, deadline=None)
@given(
    seed=st.integers(0, 2**16),
    state_dim=st.sampled_from((2, 3)),
    rate=st.floats(0.1, 3.0),
    drift=st.floats(-2.0, 3.0),
    fraction=st.floats(0.05, 0.95),
)
def test_contractive_residual_euler_step_contracts_by_the_factor(
    seed: int, state_dim: int, rate: float, drift: float, fraction: float
) -> None:
    """contractive_euler.v (euler_factor_nonneg, euler_factor_lt_one) on the Euler step itself."""
    key, k_x, k_y, k_u = jax.random.split(jax.random.key(seed), 4)
    model = ContractiveResidual(state_dim, 1, width=8, rate=rate, key=key)
    model = eqx.tree_at(lambda m: m.log_drift, model, jnp.full(state_dim, drift))
    contraction, lipschitz = float(model.contraction_rate()), float(model.lipschitz_constant())
    dt = fraction * 2.0 * contraction / lipschitz**2
    factor_sq = 1.0 - 2.0 * contraction * dt + lipschitz**2 * dt**2
    assert 0.0 <= factor_sq < 1.0
    xs, ys = 2.0 * jax.random.normal(k_x, (32, state_dim)), jax.random.normal(k_y, (32, state_dim))
    us = jax.random.normal(k_u, (32, 1))

    def euler(x: jax.Array, u: jax.Array) -> jax.Array:
        return x + dt * model(0.0, x, u)

    moved = np.asarray(jax.vmap(euler)(xs, us) - jax.vmap(euler)(ys, us))
    ratios = np.linalg.norm(moved, axis=1) / np.linalg.norm(np.asarray(xs - ys), axis=1)
    assert np.all(ratios <= math.sqrt(factor_sq) + 1e-12)


def _difference_error(velocity: jax.Array) -> float:
    """Central difference at h = 1e-5: h^2 |H'''| |v|^3 / 6 truncation plus eps |H| / h rounding."""
    return 1e-9 * (1.0 + float(jnp.linalg.norm(velocity))) ** 3


def _port_model(seed: int, kind: str, convexity: float = 0.25) -> PortHamiltonianResidual:
    model = PortHamiltonianResidual(
        3, 2, width=8, energy=kind, convexity=convexity, key=jax.random.key(seed)
    )
    # A random R = L L' has eigenvalues near 0.01; a larger one makes the dissipation term visible.
    return eqx.tree_at(lambda m: m.l_raw, model, 5.0 * model.l_raw)


@settings(max_examples=5, deadline=None)
@given(seed=st.integers(0, 2**16), kind=st.sampled_from(("mlp", "icnn")))
def test_port_hamiltonian_energy_rate_is_supply_minus_dissipation(seed: int, kind: str) -> None:
    """convex_port_hamiltonian.mac STEP 1: dH/dt along the field = y'u - dH' R dH (differences)."""
    model = _port_model(seed, kind)
    rng = np.random.default_rng(seed)
    _, dissipation = model.structure_matrices()
    step = 1e-5
    for _ in range(6):
        x, u = jnp.asarray(2.0 * rng.normal(size=3)), jnp.asarray(rng.normal(size=2))
        velocity = model(0.0, x, u)
        measured = (model.energy(x + step * velocity) - model.energy(x - step * velocity)) / (
            2.0 * step
        )
        gradient = model.energy_gradient(x)
        port = model.input_map(x).reshape(3, 2)
        supply, loss = (port.T @ gradient) @ u, gradient @ dissipation @ gradient
        assert float(measured) == pytest.approx(
            float(supply - loss), rel=0.0, abs=_difference_error(velocity)
        )
        assert float(loss) >= 0.0


@settings(max_examples=4, deadline=None)
@given(seed=st.integers(0, 2**16), convexity=st.floats(0.05, 2.0))
def test_convex_port_hamiltonian_flow_stays_in_the_invariant_ball(
    seed: int, convexity: float
) -> None:
    """convex_port_hamiltonian.v (a_quadratic_floor_bounds_the_sublevel_set): |x| <= (2H0/eps)^.5"""
    model = _port_model(seed, "icnn", convexity)
    starts = 3.0 * jax.random.normal(jax.random.key(seed + 1), (6, 3))
    paths = jax.vmap(lambda s: _unforced_flow(model, s, 0.02, 400))(starts)
    radii = np.asarray(jax.vmap(lambda s: model.invariant_radius(model.energy(s)))(starts))
    energies = np.asarray(jax.vmap(jax.vmap(model.energy))(paths))
    assert np.all(energies <= np.asarray(jax.vmap(model.energy)(starts))[:, None] + 1e-9)
    excursions = np.max(np.linalg.norm(np.asarray(paths), axis=2), axis=1)
    assert np.all(excursions <= radii + 1e-9)
    assert np.all(np.linalg.norm(np.asarray(starts), axis=1) <= radii)


def test_mlp_port_hamiltonian_energy_is_bounded_with_no_radius() -> None:
    """convex_port_hamiltonian.v (a_bounded_energy_admits_no_finite_radius) for the tanh energy."""
    model = PortHamiltonianResidual(3, 1, width=8, key=jax.random.key(4))
    last = model.energy.layers[-1]
    ceiling = float(jnp.sum(jnp.abs(last.weight)) + jnp.sum(jnp.abs(last.bias)))
    far = 1e6 * jax.random.normal(jax.random.key(5), (256, 3))
    values = np.asarray(jax.vmap(model.energy)(far))
    assert np.all(np.abs(values) <= ceiling)
    assert float(model.invariant_radius(ceiling)) == math.inf


@settings(max_examples=6, deadline=None)
@given(
    seed=st.integers(0, 2**16),
    convexity=st.floats(0.01, 3.0),
    width=st.integers(2, 12),
    depth=st.integers(1, 3),
)
def test_convex_energy_is_strongly_convex_above_its_floor(
    seed: int, convexity: float, width: int, depth: int
) -> None:
    """convex_port_hamiltonian.mac STEPs 4, 6: H((x+y)/2) <= mean - eps|x-y|^2/8, H >= eps|x|^2/2"""
    energy = _ConvexEnergy(3, width, depth, convexity, key=jax.random.key(seed))
    energy = eqx.tree_at(lambda e: e.biases, energy, [b - 1.0 for b in energy.biases])
    rng = np.random.default_rng(seed)
    left, right = (jnp.asarray(rng.normal(scale=s, size=(64, 3))) for s in (1.0, 10.0))
    h = jax.vmap(energy)
    midpoint = np.asarray(h(0.5 * (left + right)))
    chord = np.asarray(0.5 * (h(left) + h(right)))
    gap = 0.125 * convexity * np.sum(np.asarray(left - right) ** 2, axis=1)
    assert np.all(midpoint <= chord - gap + 1e-9 * np.abs(chord))
    floor = 0.5 * convexity * np.sum(np.asarray(right) ** 2, axis=1)
    assert np.all(np.asarray(h(right)) >= floor)


def test_convex_energy_has_one_critical_point() -> None:
    """convex_port_hamiltonian.v (strong_convexity_gives_a_unique_critical_point)."""
    model = PortHamiltonianResidual(
        2, 1, width=8, energy="icnn", convexity=0.25, key=jax.random.key(7)
    )
    axis = jnp.linspace(-3.0, 3.0, 3)
    starts = jnp.stack(jnp.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    rested = np.asarray(_critical_points(model, starts, 0.2, 3000))
    gradients = np.asarray(jax.vmap(jax.grad(model.energy))(jnp.asarray(rested)))
    assert np.max(np.linalg.norm(gradients, axis=1)) < 1e-10
    found = minimize(
        lambda x: float(model.energy(jnp.asarray(x))),
        np.array([2.5, -2.5]),
        jac=lambda x: np.asarray(model.energy_gradient(jnp.asarray(x))),
        method="BFGS",
        options={"gtol": 1e-12},
    )
    np.testing.assert_allclose(rested, np.broadcast_to(found.x, rested.shape), rtol=0.0, atol=1e-7)


@settings(max_examples=6, deadline=None)
@given(seed=st.integers(0, 2**16), kappa=st.floats(0.0, 5.0))
def test_damping_injection_rate_adds_the_port_dissipation(seed: int, kappa: float) -> None:
    """port_hamiltonian_lyapunov.v (skew_energy_neutral, damping_decrease) by differences."""
    model = PortHamiltonianResidual(3, 1, width=8, key=jax.random.key(seed))
    _, dissipation = model.structure_matrices()
    rng, step = np.random.default_rng(seed), 1e-5
    for _ in range(5):
        x = jnp.asarray(rng.normal(size=3))
        gradient = model.energy_gradient(x)
        output = model.input_map(x).reshape(3, 1).T @ gradient
        velocity = model(0.0, x, -kappa * output)
        measured = (model.energy(x + step * velocity) - model.energy(x - step * velocity)) / (
            2.0 * step
        )
        loss = gradient @ dissipation @ gradient
        expected = -loss - kappa * float(output @ output)
        assert float(measured) == pytest.approx(expected, rel=0.0, abs=_difference_error(velocity))
        assert expected <= -float(loss) <= 0.0


def test_damping_injection_certificate_rollout_loses_energy() -> None:
    """port_hamiltonian_lyapunov.v (damping_decrease, euler_energy_decrease) on the rollout."""
    cert = damping_injection_certificate(seed=2, horizon=10, kappa=2.0)
    assert cert.max_energy_rate <= 0.0
    assert cert.damping_dissipation >= 0.0
    assert cert.energy_dissipated >= 0.0
    assert cert.ok


# ---------------------------------------------------------------------------------------------
# chc.residual.SpectralResidual, chc.toeplitz and chc.research.transport: circulant operators.
# ---------------------------------------------------------------------------------------------

_kernels = st.integers(2, 16).flatmap(
    lambda n: st.lists(st.floats(-3.0, 3.0), min_size=n, max_size=n).map(np.asarray)
)


@settings(max_examples=40, deadline=None)
@given(kernel=_kernels)
def test_circulant_symbol_holds_the_dense_eigenvalues(kernel: np.ndarray) -> None:
    """spectral_circulant.mac STEP 1: C v_k = lambda_k v_k, v_k = exp(2 pi i j k / n)."""
    n = kernel.size
    dense = circulant(kernel)
    symbol = np.asarray(circulant_symbol(jnp.asarray(kernel)))
    assert symbol.shape == (n // 2 + 1,)
    j = np.arange(n)
    scale = np.abs(kernel).sum() + 1.0
    for k, value in enumerate(symbol):
        mode = np.exp(2j * np.pi * j * k / n)
        np.testing.assert_allclose(dense @ mode, value * mode, rtol=0.0, atol=1e-13 * scale)


@settings(max_examples=40, deadline=None)
@given(kernel=_kernels)
def test_circulant_operator_norm_is_the_dense_spectral_norm(kernel: np.ndarray) -> None:
    """spectral_circulant.v (two_mode_bounded, two_mode_norm_attained): ||C||_2 = max |lambda_k|."""
    dense_norm = np.linalg.norm(circulant(kernel), 2)
    norm = float(circulant_operator_norm(jnp.asarray(kernel)))
    assert norm == pytest.approx(dense_norm, rel=1e-12, abs=1e-14)


def _spectral(seed: int, n: int) -> SpectralResidual:
    return SpectralResidual(n, n, key=jax.random.key(seed), scale=1.0)


def _dense(model: SpectralResidual) -> np.ndarray:
    n = model.grid
    zero = jnp.zeros(n)
    return np.asarray(jax.vmap(lambda e: model(0.0, e, zero))(jnp.eye(n))).T


@settings(max_examples=8, deadline=None)
@given(seed=st.integers(0, 2**16), n=st.sampled_from((5, 8)))
def test_spectral_residual_norm_is_attained_on_a_fourier_mode(seed: int, n: int) -> None:
    """spectral_circulant.v (mode_gain_is_exact, two_mode_norm_attained): a Fourier witness."""
    model = _spectral(seed, n)
    norm = float(model.operator_norm())
    assert norm == pytest.approx(np.linalg.norm(_dense(model), 2), rel=1e-12, abs=0.0)
    k = int(np.argmax(np.abs(np.asarray(model.symbol()))))
    witness = jnp.cos(2.0 * jnp.pi * k * jnp.arange(n) / n + 0.3)
    image = model(0.0, witness, jnp.zeros(n))
    gain = float(jnp.linalg.norm(image) / jnp.linalg.norm(witness))
    assert gain == pytest.approx(norm, rel=1e-12, abs=0.0)


@settings(max_examples=8, deadline=None)
@given(seed=st.integers(0, 2**16), n=st.sampled_from((5, 8)), shift=st.integers(-20, 20))
def test_spectral_residual_is_translation_equivariant(seed: int, n: int, shift: int) -> None:
    """spectral_circulant.mac STEP 1: a circulant commutes with every cyclic shift."""
    model = _spectral(seed, n)
    x, u = jax.random.normal(jax.random.key(seed + 1), (2, n))
    rolled = model(0.0, jnp.roll(x, shift), jnp.roll(u, shift))
    np.testing.assert_allclose(
        np.asarray(rolled), np.roll(np.asarray(model(0.0, x, u)), shift), rtol=0.0, atol=1e-13
    )


@settings(max_examples=8, deadline=None)
@given(seed=st.integers(0, 2**16), n=st.sampled_from((5, 8)))
def test_spectral_residual_gains_multiply_under_composition(seed: int, n: int) -> None:
    """spectral_circulant.v (gain_multiplies, modes_commute_*): |lambda mu| per mode, exactly."""
    first, second = _spectral(seed, n), _spectral(seed + 1, n)
    left, right = _dense(first), _dense(second)
    np.testing.assert_allclose(left @ right, right @ left, rtol=0.0, atol=1e-12)
    per_mode = np.abs(np.asarray(first.symbol()) * np.asarray(second.symbol()))
    assert np.linalg.norm(right @ left, 2) == pytest.approx(per_mode.max(), rel=1e-12, abs=1e-14)
    assert per_mode.max() <= float(first.operator_norm() * second.operator_norm()) * (1 + 1e-12)


@settings(max_examples=10, deadline=None)
@given(
    n=st.sampled_from((8, 13)),
    length=st.floats(0.5, 10.0),
    speed=st.floats(-3.0, 3.0),
    viscosity=st.floats(0.0, 0.5),
    amplitudes=st.lists(st.floats(-1.0, 1.0), min_size=3, max_size=3),
    phases=st.lists(st.floats(0.0, 6.3), min_size=3, max_size=3),
)
def test_advection_diffusion_field_differentiates_trigonometric_fields(
    n: int,
    length: float,
    speed: float,
    viscosity: float,
    amplitudes: list[float],
    phases: list[float],
) -> None:
    """spectral_circulant.mac STEP 5: the symbol -i c k - nu k^2 is -c d_x + nu d_xx, exactly."""
    x = np.arange(n) * length / n
    modes = (1, 2, (n - 1) // 2)
    u = np.zeros(n)
    expected = np.zeros(n)
    for m, a, phi in zip(modes, amplitudes, phases, strict=True):
        k = 2.0 * np.pi * m / length
        u += a * np.cos(k * x + phi)
        expected += a * (speed * k * np.sin(k * x + phi) - viscosity * k**2 * np.cos(k * x + phi))
    field = np.asarray(
        advection_diffusion_field(jnp.asarray(u), length, speed=speed, viscosity=viscosity)
    )
    scale = 1.0 + np.abs(expected).max()
    np.testing.assert_allclose(field, expected, rtol=0.0, atol=1e-12 * scale)


@settings(max_examples=10, deadline=None)
@given(
    half=st.sampled_from((4, 8)),
    length=st.floats(0.5, 10.0),
    speed=st.floats(-3.0, 3.0),
    viscosity=st.floats(0.0, 0.5),
)
def test_advection_diffusion_field_has_no_first_derivative_at_nyquist(
    half: int, length: float, speed: float, viscosity: float
) -> None:
    """spectral_circulant.mac STEP 7: the sampled d/dx of (-1)^j is 0, d^2/dx^2 is -k_N^2 (-1)^j."""
    n = 2 * half
    nyquist = (-1.0) ** np.arange(n)
    k = np.pi * n / length
    field = np.asarray(
        advection_diffusion_field(jnp.asarray(nyquist), length, speed=speed, viscosity=viscosity)
    )
    np.testing.assert_allclose(
        field, -viscosity * k**2 * nyquist, rtol=0.0, atol=1e-12 * (1 + k**2)
    )
