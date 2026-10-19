"""The estimation layer's cited derivations and proofs, checked on the float code that states them.

Each test reads a statement of a Maxima derivation or a Rocq lemma that a docstring in
``chc.delay``, ``chc.dlm``, ``chc.dynamics_id``, ``chc.experiment``, ``chc.gate``, ``chc.irf``,
``chc.pathway``, ``chc.response``, ``chc.switchback`` or ``chc.zones`` cites, and checks it on the
code by a route other than the formula the code evaluates: a Lambert W branch, a brute-force
maximiser, a simulated plant, a sequential update, an exact sigma-point expectation or a seeded
Monte Carlo average.
"""

from __future__ import annotations

import dataclasses
import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from scipy import optimize, stats
from scipy.special import lambertw

from chc.delay import STABILISING_RATIO_FLOOR, delay_design_loss, delay_margin
from chc.dlm import GeoDLM, Polynomial, Prior, forward_filter_geos, stacked_prior
from chc.dynamics_id import (
    ChannelRegion,
    closed_loop_gain_attribution,
    fit_causal_residual,
    omitted_confounder_bound,
)
from chc.experiment import ZoneDecision
from chc.gate import (
    BlockBatch,
    ChannelMove,
    DecisionLog,
    DeploymentGate,
    GateConfig,
    ZonePlan,
    _per_block,
    channel_drift_evalues,
    channel_move,
)
from chc.irf import peak_lag, structured_irf
from chc.pathway import _ancestor_onsets, causal_pathway
from chc.plan import DecisionWeight
from chc.response import (
    ChapmanRichards,
    Exponential,
    Gompertz,
    Hill,
    Logistic,
    MichaelisMenten,
    Weibull,
)
from chc.switchback import BlockDesign, Horizon, _kernel, randomisation_interval
from chc.zones import ZoneMarketSystem

FORMAL_LINKS = {
    "chc.delay.delay_design_loss": {
        "cites": ("delay_ball.mac",),
        "tests": (
            "test_delay_design_loss_is_one_plus_the_rightmost_characteristic_root",
            "test_delay_design_loss_runs_from_zero_at_the_truth_to_one_at_the_floor",
            "test_delay_design_loss_opens_as_a_square_root_above_and_linearly_below",
            "test_delay_design_loss_series_is_short_by_its_next_term",
            "test_delay_design_loss_is_finite_just_above_the_stabilising_floor",
        ),
    },
    "chc.delay.delay_margin": {
        "cites": ("delay_margin.mac",),
        "tests": (
            "test_delay_margin_is_where_the_rightmost_root_crosses_the_imaginary_axis",
            "test_delay_margin_of_a_pure_integrator_is_pi_over_twice_the_gain",
        ),
    },
    "chc.dlm": {
        "cites": ("geo_dlm.mac",),
        "tests": (
            "test_geo_filter_joint_update_is_the_sequential_one",
            "test_geo_filter_multiplicative_discount_keeps_geos_apart_in_precision",
        ),
    },
    "chc.dynamics_id.ChannelRegion.stability_level": {
        "cites": ("tube_levels.mac",),
        "tests": ("test_stability_level_is_the_largest_level_whose_operator_radius_is_rho",),
    },
    "chc.dynamics_id.closed_loop_gain_attribution": {
        "cites": ("closed_loop_attribution.v",),
        "tests": (
            "test_tracked_log_identifies_the_interaction_as_minus_gain_times_curvature",
            "test_tracked_log_leaves_the_drift_free_along_the_rocq_family",
            "test_exploration_off_the_manifold_recovers_the_drift",
        ),
    },
    "chc.dynamics_id.omitted_confounder_bound": {
        "cites": ("omitted_confounder_bound.mac",),
        "tests": (
            "test_omitted_confounder_bound_is_attained_with_one_latent_left_out",
            "test_omitted_confounder_robustness_value_moves_the_bound_onto_the_null",
        ),
    },
    "chc.experiment": {
        "cites": ("experiment_bayes_action.mac",),
        "tests": (
            "test_expected_cost_under_a_gaussian_channel_adds_the_hadamard_term",
            "test_bayes_action_is_the_certainty_equivalent_one_with_a_raised_action_weight",
            "test_one_zone_example_regrets_match_the_module_docstring",
        ),
    },
    "chc.gate": {
        "cites": ("gate_block_contrast.mac",),
        "tests": (
            "test_block_read_meets_the_geometric_memory_bound_and_attains_it",
            "test_block_read_under_carryover_of_order_one",
            "test_gate_in_blocks_reads_the_sustained_contrast_not_the_myopic_one",
        ),
    },
    "chc.gate.ChannelMove.price": {
        "cites": ("dither_channel_move.mac", "dither_channel_move.v"),
        "tests": (
            "test_move_price_keep_is_unbiased_and_replan_is_the_estimates_mean_regret",
            "test_move_price_keep_error_is_the_keep_prices_spread",
        ),
    },
    "chc.gate.channel_drift_evalues": {
        "cites": ("dither_drift_evalue.mac",),
        "tests": (
            "test_drift_evalues_average_one_over_one_minus_theta_k",
            "test_clipped_drift_evalues_average_the_clipped_closed_form",
        ),
    },
    "chc.gate.channel_move": {
        "cites": ("dither_channel_move.mac",),
        "tests": (
            "test_channel_move_reads_the_move_without_bias_and_a_calibrated_covariance",
            "test_channel_move_effective_size_is_the_geometric_closed_form",
        ),
    },
    "chc.irf.peak_lag": {
        "cites": ("delay_estimate_bias.mac",),
        "tests": (
            "test_refined_peak_of_a_geometric_response_is_late_by_the_closed_form",
            "test_refined_peak_of_a_split_lag_is_f_over_4_minus_6f",
            "test_peak_at_an_end_is_censored_and_a_plateau_reads_its_leading_pair",
        ),
    },
    "chc.pathway": {
        "cites": ("causal_pathway.mac", "causal_pathway.v"),
        "tests": (
            "test_ancestor_onsets_are_the_shortest_lagged_walks",
            "test_pathway_lever_response_is_the_product_along_its_chain",
            "test_structured_irf_tail_past_the_horizon_is_geometric",
        ),
    },
    "chc.response": {
        "cites": ("response_curves.mac",),
        "tests": (
            "test_tangency_maximises_the_return_per_unit_spend",
            "test_inflection_is_where_the_curvature_changes_sign",
            "test_a_change_of_currency_moves_only_the_scale",
        ),
    },
    "chc.response.Saturation.nonconvexity": {
        "cites": ("envelope_nonconvexity.mac",),
        "tests": (
            "test_nonconvexity_is_the_largest_rise_of_the_chord_over_the_curve",
            "test_hill_nonconvexity_is_the_reduced_root",
            "test_hill_nonconvexity_anchors_and_threshold",
        ),
    },
    "chc.switchback.randomisation_interval": {
        "cites": ("switchback_randomisation.mac",),
        "tests": (
            "test_lever_path_enters_the_readings_through_the_memory_kernel",
            "test_randomisation_interval_misses_on_at_most_alpha_of_the_schedules",
        ),
    },
    "chc.zones.ZoneMarketSystem.zone_decision": {
        "cites": ("zone_market.mac",),
        "tests": (
            "test_zone_decision_is_the_settled_trips_under_linear_matching",
            "test_zone_decision_is_the_first_order_response_under_harmonic_matching",
        ),
    },
}

SEEDS = st.integers(0, 2**32 - 1)


# --------------------------------------------------------------------------------- chc.delay


def _rightmost_root(kappa: float) -> float:
    """Real part of the rightmost root of ``s e^s = -kappa``, over the Lambert W branches."""
    principal = lambertw(-kappa, 0)
    if np.isnan(principal):  # -kappa rounded past the branch point -1/e: the double root s = -1
        return -1.0
    return max(float(np.real(lambertw(-kappa, k))) for k in range(-6, 7))


@settings(max_examples=40, deadline=None)
@given(ratio=st.floats(STABILISING_RATIO_FLOOR * 1.001, 60.0))
def test_delay_design_loss_is_one_plus_the_rightmost_characteristic_root(ratio: float) -> None:
    """delay_ball.mac: the loss is 1 + Re s, s the rightmost root of s e^s + 1/(e r) = 0."""
    kappa = 1.0 / (math.e * ratio)
    assert delay_design_loss(ratio) == pytest.approx(1.0 + _rightmost_root(kappa), abs=1e-9)


def test_delay_design_loss_runs_from_zero_at_the_truth_to_one_at_the_floor() -> None:
    """delay_ball.mac: the root is double at r = 1; the boundary kappa = pi/2 is r = 2/(pi e)."""
    assert delay_design_loss(1.0) == 0.0
    assert delay_design_loss(STABILISING_RATIO_FLOOR * (1.0 + 1e-9)) == pytest.approx(1.0, abs=1e-6)
    assert math.isinf(delay_design_loss(STABILISING_RATIO_FLOOR * (1.0 - 1e-9)))
    ratios = np.geomspace(STABILISING_RATIO_FLOOR * 1.01, 50.0, 400)
    losses = np.array([delay_design_loss(r) for r in ratios])
    below = ratios < 1.0
    assert np.all(np.diff(losses[below]) < 0.0)
    assert np.all(np.diff(losses[~below]) > 0.0)


def test_delay_design_loss_opens_as_a_square_root_above_and_linearly_below() -> None:
    """delay_ball.mac sections 3, 6: at |eps| = 0.05 over costs 0.287, under 0.033."""
    over = delay_design_loss(1.0 / (1.0 - 0.05))
    under = delay_design_loss(1.0 / (1.0 + 0.05))
    assert over == pytest.approx(0.287, abs=5e-4)
    assert under == pytest.approx(0.033, abs=5e-4)
    assert over / under == pytest.approx(8.8, abs=0.05)
    assert under == pytest.approx(2.0 * 0.05 / 3.0, abs=1e-3)
    # the square-root branch: the exact loss over sqrt(2 eps) tends to 1
    small = 1e-6
    assert delay_design_loss(1.0 / (1.0 - small)) / math.sqrt(2.0 * small) == pytest.approx(
        1.0, abs=1e-3
    )


def test_delay_design_loss_series_is_short_by_its_next_term() -> None:
    """delay_ball.mac section 5b: at |eps| = 0.05 the two-term series misses by 4.2e-3 over and
    7.7e-4 under; over, the gap is the eps^(3/2) term 11/(9 2^(3/2)) to order eps^2, under it is
    of order eps^2. The docstring said 1e-3 on both sides."""
    third = 11.0 / (9.0 * 2.0**1.5)
    for eps in (0.05, 0.01, 0.002):
        over = delay_design_loss(1.0 / (1.0 - eps)) - (math.sqrt(2.0 * eps) - 2.0 * eps / 3.0)
        under = delay_design_loss(1.0 / (1.0 + eps)) - 2.0 * eps / 3.0
        # the w^4 term, -43/540 (2 eps)^2, bounds what the eps^(3/2) term leaves
        assert abs(over - third * eps**1.5) <= 4.0 * 43.0 / 540.0 * eps**2
        assert abs(under) <= 0.35 * eps**2
    assert delay_design_loss(1.0 / 0.95) - (math.sqrt(0.1) - 0.1 / 3.0) == pytest.approx(
        4.154e-3, rel=1e-3, abs=0.0
    )
    assert delay_design_loss(1.0 / 1.05) - 0.1 / 3.0 == pytest.approx(-7.712e-4, rel=1e-3, abs=0.0)


def test_delay_design_loss_is_finite_just_above_the_stabilising_floor() -> None:
    """delay_ball.mac section 2: every ratio above 2/(pi e) stabilises, with a loss in [0, 1].
    brentq's bracket stopped at pi/2 - 1e-12, short of the root, and raised ValueError."""
    for above in (1e-15, 1e-13, 1e-12, 2e-12):
        loss = delay_design_loss(STABILISING_RATIO_FLOOR * (1.0 + above))
        assert loss == pytest.approx(1.0, rel=0.0, abs=1e-9)


def _rightmost_delay_root(pole: float, gain: float, tau: float) -> float:
    """Re of the rightmost root of lambda - pole + gain exp(-lambda tau) = 0, by Lambert W."""
    argument = -gain * tau * math.exp(-pole * tau)
    return max(pole + float(np.real(lambertw(argument, k))) / tau for k in range(-6, 7))


@settings(max_examples=40, deadline=None)
@given(pole=st.floats(-3.0, 3.0), excess=st.floats(0.05, 5.0))
def test_delay_margin_is_where_the_rightmost_root_crosses_the_imaginary_axis(
    pole: float, excess: float
) -> None:
    """delay_margin.mac section 1: the root crosses the imaginary axis at tau_c, stable below it."""
    gain = abs(pole) + excess
    critical = delay_margin(pole, gain)
    assert _rightmost_delay_root(pole, gain, 0.98 * critical) < 0.0
    assert _rightmost_delay_root(pole, gain, 1.02 * critical) > 0.0
    assert _rightmost_delay_root(pole, gain, critical) == pytest.approx(0.0, abs=1e-8 * gain)


def test_delay_margin_of_a_pure_integrator_is_pi_over_twice_the_gain() -> None:
    """delay_margin.mac sections 2, 4: K tau_c = pi/2 at a = 0, and tau_c -> 1/a as K -> a+."""
    for gain in (0.3, 1.0, 7.0):
        assert delay_margin(0.0, gain) == pytest.approx(math.pi / (2.0 * gain), rel=1e-14, abs=0.0)
    assert delay_margin(0.5, 0.5 * (1.0 + 1e-8)) == pytest.approx(2.0, rel=1e-4, abs=0.0)


# ----------------------------------------------------------------------------------- chc.dlm


def _observe(a, r, s, n, f, y, w):
    """One geo's update with no evolution, as geo_dlm.mac's ``observe`` writes it."""
    q = float(f @ r @ f) + s * w
    e = y - float(f @ a)
    gain = r @ f / q
    n1 = n + 1.0
    s1 = s * (n + e * e / q) / n1
    return a + gain * e, (s1 / s) * (r - np.outer(gain, gain) * q), s1, n1, e, q


@settings(max_examples=40, deadline=None)
@given(seed=SEEDS)
def test_geo_filter_joint_update_is_the_sequential_one(seed: int) -> None:
    """geo_dlm.mac steps 1-2: two geos' joint update and t density are the sequential ones."""
    rng = np.random.default_rng(seed)
    root = rng.normal(size=(3, 3))
    covariance = root @ root.T + 0.1 * np.eye(3)
    mean, scale, dof = rng.normal(size=3), float(rng.uniform(0.2, 5.0)), float(rng.uniform(3, 40))
    weights = rng.uniform(0.2, 5.0, 2)
    y = 3.0 * rng.normal(size=(1, 2))
    model = GeoDLM(
        national=(Polynomial(1, 1.0),),
        regional=(Polynomial(1, 1.0),),
        geos=2,
        prior=Prior(mean, covariance, scale, dof),
        relative_variance=weights,
    )
    fit = forward_filter_geos(model, y)
    rows = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    a1, r1, s1, n1, e1, q1 = _observe(mean, covariance, scale, dof, rows[0], y[0, 0], weights[0])
    a2, r2, s2, n2, e2, q2 = _observe(a1, r1, s1, n1, rows[1], y[0, 1], weights[1])
    np.testing.assert_allclose(fit.mean[0], a2, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(fit.covariance[0], r2, rtol=1e-8, atol=1e-12 * np.abs(r2).max())
    assert fit.scale[0] == pytest.approx(s2, rel=1e-10, abs=0.0)
    assert fit.dof[0] == n2
    sequential = stats.t.logpdf(e1, dof, scale=math.sqrt(q1)) + stats.t.logpdf(
        e2, dof + 1.0, scale=math.sqrt(q2)
    )
    assert fit.log_likelihood == pytest.approx(float(sequential), rel=1e-10, abs=1e-12)


def _cross_precision(prior_covariance: np.ndarray) -> float:
    """The largest correlation-scaled precision between two geos' regional coordinates."""
    precision = np.linalg.inv(prior_covariance)
    diagonal = np.sqrt(np.diag(precision))
    scaled = precision / np.outer(diagonal, diagonal)
    return max(abs(scaled[i, j]) for i, j in itertools.combinations(range(1, 4), 2))


@settings(max_examples=20, deadline=None)
@given(seed=SEEDS)
def test_geo_filter_multiplicative_discount_keeps_geos_apart_in_precision(seed: int) -> None:
    """geo_dlm.mac step 3: D C D keeps the arrowhead precision's zero; component discounting not."""
    rng = np.random.default_rng(seed)
    y = rng.normal(size=(6, 3))
    prior = stacked_prior(Prior([0.0], [[1.0]], 1.0, 10.0), Prior([0.0], [[0.5]], 1.0, 10.0), 3)
    kept = {}
    for form in ("multiplicative", "additive"):
        model = GeoDLM(
            national=(Polynomial(1, 0.9),),
            regional=(Polynomial(1, 0.7),),
            geos=3,
            prior=prior,
            form=form,
        )
        fit = forward_filter_geos(model, y)
        kept[form] = [_cross_precision(fit.prior_covariance[t]) for t in range(1, 6)]
    assert max(kept["multiplicative"]) < 1e-10
    assert min(kept["additive"]) > 1e-4


# ---------------------------------------------------------------------------- chc.dynamics_id


def _region(rng: np.random.Generator) -> ChannelRegion:
    """A region over two states and two levers with a degree-1 channel: 3 features a lever."""
    n, m, k = 2, 2, 3
    root = rng.normal(size=(m * k, m * k))
    return ChannelRegion(
        level=0.9,
        channel=rng.normal(size=(n, m, k)),
        gram=root @ root.T + rng.uniform(0.5, 5.0) * np.eye(m * k),
        radius=np.zeros(n),
        noise=rng.uniform(0.05, 2.0, n),
        prior=rng.uniform(0.0, 0.5, n),
        log_det_ratio=float(rng.uniform(0.0, 12.0)),
        channel_degree=1,
    )


def _radii_at(region: ChannelRegion, level: float) -> np.ndarray:
    """Abbasi-Yadkori et al.'s radius at ``level``, ``delta / n`` a state."""
    share = (1.0 - level) / region.noise.size
    return (
        region.noise * math.sqrt(2.0 * math.log(1.0 / share) + region.log_det_ratio) + region.prior
    )


@settings(max_examples=25, deadline=None)
@given(seed=SEEDS, reach=st.floats(0.01, 50.0))
@example(seed=42259, reach=23.75)
def test_stability_level_is_the_largest_level_whose_operator_radius_is_rho(
    seed: int, reach: float
) -> None:
    """tube_levels.mac (5): the level solves sum_s (R_s A + b_s)^2 = (rho/g)^2 for delta."""
    rng = np.random.default_rng(seed)
    region = _region(rng)
    state = rng.normal(size=2)
    unit = dataclasses.replace(region, radius=np.array([1.0, 0.0])).operator_radius(state)
    rho = reach * unit
    level = region.stability_level(rho, state)
    assert 0.0 <= level <= 1.0
    if level == 0.0:
        floor = dataclasses.replace(region, radius=_radii_at(region, 0.0))
        assert floor.operator_radius(state) >= rho * (1.0 - 1e-9)
        return
    if level == 1.0:  # the miss probability is below a double's resolution of 1
        nearly = dataclasses.replace(region, radius=_radii_at(region, 1.0 - 1e-14))
        assert nearly.operator_radius(state) <= rho * (1.0 + 1e-9)
        return
    at = dataclasses.replace(region, radius=_radii_at(region, level))
    # 1 - level carries the miss probability with an absolute error of a double's epsilon
    resolution = 1e-9 + 1e-15 / (1.0 - level)
    assert at.operator_radius(state) == pytest.approx(rho, rel=resolution, abs=0.0)
    # within an ulp of 1 the midpoint rounds to 1, a level with no radius
    if 0.5 * (1.0 + level) < 1.0:
        higher = dataclasses.replace(region, radius=_radii_at(region, 0.5 * (1.0 + level)))
        assert higher.operator_radius(state) > rho


def _tracked(gain: float, setpoint: float, samples: int, rng: np.random.Generator, noise: float):
    actions = 20.0 + 2.0 * rng.standard_normal(samples)
    states = setpoint - actions / gain + noise * rng.standard_normal(samples)
    return states, actions


@settings(max_examples=30, deadline=None)
@given(
    seed=SEEDS,
    gain=st.floats(0.2, 10.0),
    plant=st.tuples(*(st.floats(-1.0, 1.0) for _ in range(4))),
    curvature=st.floats(-0.5, 0.5),
)
def test_tracked_log_identifies_the_interaction_as_minus_gain_times_curvature(
    seed: int, gain: float, plant: tuple[float, float, float, float], curvature: float
) -> None:
    """closed_loop_attribution.v interaction_identified, interaction_from_gain: b1 = -Kp C."""
    d, a, b0, b1 = plant
    x, u = _tracked(gain, 17.0, 400, np.random.default_rng(seed), 0.0)
    rates = d + a * x + (b0 + b1 * x) * u + curvature * u**2
    fit = closed_loop_gain_attribution(x, u, rates)
    size = max(1.0, abs(b1), gain * abs(curvature))
    assert fit.fitted_interaction == pytest.approx(b1 - gain * curvature, abs=1e-6 * size)
    assert fit.predicted_interaction == pytest.approx(fit.fitted_interaction, abs=1e-6 * size)
    assert fit.implied_gain == pytest.approx(gain, rel=1e-9, abs=0.0)
    assert fit.manifold_r2 == pytest.approx(1.0, abs=1e-12)


@settings(max_examples=30, deadline=None)
@given(seed=SEEDS, gain=st.floats(0.2, 10.0), shift=st.floats(0.1, 2.0))
def test_tracked_log_leaves_the_drift_free_along_the_rocq_family(
    seed: int, gain: float, shift: float
) -> None:
    """closed_loop_attribution.v drift_not_identified: (d + (a - a') c, a', b0 + (a - a') m)."""
    d, a, b0, b1 = 1.0, -0.35, 0.4, -0.3
    setpoint, slope = 17.0, -1.0 / gain
    x, u = _tracked(gain, setpoint, 400, np.random.default_rng(seed), 0.0)
    other = a - shift
    first = d + a * x + (b0 + b1 * x) * u
    second = (d + shift * setpoint) + other * x + (b0 + shift * slope + b1 * x) * u
    np.testing.assert_allclose(second, first, rtol=1e-10, atol=1e-10)
    one, two = closed_loop_gain_attribution(x, u, first), closed_loop_gain_attribution(x, u, second)
    assert one.fitted_drift == pytest.approx(two.fitted_drift, abs=1e-6)
    assert one.fitted_interaction == pytest.approx(b1, abs=1e-6)
    assert two.fitted_interaction == pytest.approx(b1, abs=1e-6)
    assert one.exploration_budget == pytest.approx(0.0, abs=1e-12)


def test_exploration_off_the_manifold_recovers_the_drift() -> None:
    """closed_loop_attribution.v: off the manifold the design regains rank and d, a, b0 return."""
    d, a, b0, b1 = 1.0, -0.35, 0.4, -0.3
    x, u = _tracked(2.6, 17.0, 2000, np.random.default_rng(3), 1.0)
    fit = closed_loop_gain_attribution(x, u, d + a * x + (b0 + b1 * x) * u)
    assert fit.fitted_drift == pytest.approx(a, abs=1e-8)
    assert fit.fitted_interaction == pytest.approx(b1, abs=1e-8)
    assert fit.exploration_budget > 0.01


_DT, _NOISE, _ETA = 0.05, 0.01, 0.5


@pytest.fixture(scope="module")
def confounded_fit():
    """One lever, an observed confounder adjusted for and a latent one left out; channel 1."""
    rng = np.random.default_rng(0)
    n = 4000
    x, observed, latent = rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n)
    u = 0.8 * observed + latent + _ETA * rng.standard_normal(n)
    after = x + _DT * (-0.5 * x + u + 0.7 * observed + latent) + _NOISE * rng.standard_normal(n)
    data = {
        "x": jnp.asarray(x[:, None]),
        "u": jnp.asarray(u[:, None]),
        "x_next": jnp.asarray(after[:, None]),
        "observed": jnp.asarray(observed[:, None]),
    }
    return fit_causal_residual(
        lambda t, s, a: jnp.zeros_like(s),
        data,
        _DT,
        adjust_for=("observed",),
        degree=1,
        channel_degree=0,
        integrator="euler",
        influence=True,
    )


def test_omitted_confounder_bound_is_attained_with_one_latent_left_out(confounded_fit) -> None:
    """omitted_confounder_bound.mac STEP 1: at the true shares the bound's side is the channel."""
    lever_variance, lever_noise = 1.0 + _ETA**2, _ETA**2  # V = a^2 + sv with a = 1
    rate_noise = (_NOISE / _DT) ** 2
    bias = 1.0 / lever_variance  # a b / V with b = 1
    sigma2 = rate_noise + lever_noise / lever_variance
    cf_d, cf_y = 1.0 / lever_variance, (sigma2 - rate_noise) / sigma2
    bound = omitted_confounder_bound(confounded_fit, np.ones((1, 1, 1)), cf_y=cf_y, cf_d=cf_d)
    side = 4.0 * confounded_fit.channel_error
    assert bound.estimate - 1.0 == pytest.approx(bias, abs=side)
    assert bound.strength * bound.bias_scale == pytest.approx(bias, abs=side)
    assert bound.lower == pytest.approx(1.0, abs=side)


@pytest.mark.parametrize("null", [0.0, 1.0, 1.5])
def test_omitted_confounder_robustness_value_moves_the_bound_onto_the_null(
    confounded_fit, null: float
) -> None:
    """omitted_confounder_bound.mac STEP 5: cf_y = cf_d = rv puts the facing bound at the null."""
    weights = np.ones((1, 1, 1))
    first = omitted_confounder_bound(confounded_fit, weights, cf_y=0.1, cf_d=0.1, null=null)
    share = first.robustness_value
    at = omitted_confounder_bound(confounded_fit, weights, cf_y=share, cf_d=share, null=null)
    assert at.strength == pytest.approx(share / math.sqrt(1.0 - share), rel=1e-12, abs=0.0)
    assert at.lower == pytest.approx(null, abs=1e-9)
    share_ci = first.robustness_value_ci
    assert 0.0 < share_ci < share
    at_ci = omitted_confounder_bound(
        confounded_fit, weights, cf_y=share_ci, cf_d=share_ci, null=null
    )
    assert at_ci.ci_lower == pytest.approx(null, abs=1e-8)


# --------------------------------------------------------------------------- chc.experiment


def _decision(rng: np.random.Generator, zones: int, lo: float, hi: float) -> ZoneDecision:
    root = rng.normal(size=(zones, zones))
    action_root = rng.normal(size=(zones, zones))
    return ZoneDecision(
        coupling=np.eye(zones) + 0.3 * rng.normal(size=(zones, zones)),
        baseline=rng.normal(size=zones),
        target=rng.normal(size=zones),
        state_weight=root @ root.T,
        action_weight=action_root @ action_root.T + 0.2 * np.eye(zones),
        lo=np.full(zones, lo),
        hi=np.full(zones, hi),
    )


def _expected_cost(decision: ZoneDecision, u: np.ndarray, mean: np.ndarray, root: np.ndarray):
    """E J(u; b) for b ~ N(mean, root root'), exact for J quadratic in b: symmetric sigma points."""
    k = mean.size
    points = [mean + s * math.sqrt(k) * root[:, i] for i in range(k) for s in (1.0, -1.0)]
    return float(np.mean([decision.cost(u, b) for b in points]))


def _channel_law(rng: np.random.Generator, zones: int) -> tuple[np.ndarray, np.ndarray]:
    return 1.0 + 0.5 * rng.normal(size=zones), 0.4 * rng.normal(size=(zones, zones))


@settings(max_examples=40, deadline=None)
@given(seed=SEEDS, zones=st.integers(1, 4))
def test_expected_cost_under_a_gaussian_channel_adds_the_hadamard_term(
    seed: int, zones: int
) -> None:
    """experiment_bayes_action.mac STEP 1: E J(u; b) = J(u; m) + u'((C'QC) o S) u / 2."""
    rng = np.random.default_rng(seed)
    decision = _decision(rng, zones, -np.inf, np.inf)
    mean, root = _channel_law(rng, zones)
    u = rng.normal(size=zones)
    covariance = root @ root.T
    weight = decision.coupling.T @ decision.state_weight @ decision.coupling
    closed = decision.cost(u, mean) + 0.5 * float(u @ (weight * covariance) @ u)
    assert _expected_cost(decision, u, mean, root) == pytest.approx(closed, rel=1e-10, abs=1e-12)


@settings(max_examples=30, deadline=None)
@given(seed=SEEDS, zones=st.integers(1, 3), bounded=st.booleans())
def test_bayes_action_is_the_certainty_equivalent_one_with_a_raised_action_weight(
    seed: int, zones: int, bounded: bool
) -> None:
    """experiment_bayes_action.mac STEP 1: argmin E J is the CE action with R + (C'QC) o S."""
    rng = np.random.default_rng(seed)
    lo, hi = (-0.5, 0.7) if bounded else (-np.inf, np.inf)
    decision = _decision(rng, zones, lo, hi)
    mean, root = _channel_law(rng, zones)
    weight = decision.coupling.T @ decision.state_weight @ decision.coupling
    raised = dataclasses.replace(
        decision, action_weight=decision.action_weight + weight * (root @ root.T)
    )
    bayes = raised.solve(mean)
    found = optimize.minimize(
        lambda v: _expected_cost(decision, v, mean, root),
        np.zeros(zones),
        method="L-BFGS-B",
        bounds=[(lo, hi)] * zones if bounded else None,
        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10_000},
    )
    best = _expected_cost(decision, bayes, mean, root)
    assert best <= found.fun + 1e-10 * max(1.0, abs(found.fun))
    np.testing.assert_allclose(bayes, found.x, atol=1e-5)


def test_one_zone_example_regrets_match_the_module_docstring() -> None:
    """chc.experiment: Q = 1, R = 0.05, target 1, b ~ N(1, 0.6^2): regret 0.12 and Bayes 0.079.
    The docstring said 0.078; the Bayes action's regret is 0.07858."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(200)
    weights = weights / weights.sum()
    one = np.ones((1, 1))
    decision = ZoneDecision(one, [0.0], [1.0], one, 0.05 * one, [-np.inf], [np.inf])
    bayes = dataclasses.replace(decision, action_weight=(0.05 + 0.36) * one)

    def regret(action: np.ndarray) -> float:
        return float(
            sum(
                w * decision.regret(action, [1.0 + 0.6 * z])
                for z, w in zip(nodes, weights, strict=True)
            )
        )

    assert regret(decision.solve([1.0])) == pytest.approx(0.12, abs=0.005)
    assert regret(bayes.solve([1.0])) == pytest.approx(0.079, abs=5e-4)
    assert regret(bayes.solve([1.0])) < 0.100


# --------------------------------------------------------------------------------- chc.gate


@settings(max_examples=40, deadline=None)
@given(
    seed=SEEDS,
    memory=st.floats(0.05, 0.95),
    burn=st.integers(0, 6),
    read=st.integers(1, 8),
    scale=st.floats(0.0, 0.25),
    values=st.tuples(st.floats(0.3, 0.7), st.floats(0.3, 0.7)),
)
def test_block_read_meets_the_geometric_memory_bound_and_attains_it(
    seed: int,
    memory: float,
    burn: int,
    read: int,
    scale: float,
    values: tuple[float, float],
) -> None:
    """gate_block_contrast.mac STEP 1: the read mean is within 2 C a^b (1 - a^r)/(r (1 - a))."""
    new, base = values
    envelope = scale * memory ** np.arange(burn + read)
    bound = 2.0 * scale * memory**burn * (1.0 - memory**read) / (read * (1.0 - memory))
    rng = np.random.default_rng(seed)
    worst = np.array([new + envelope, base - envelope])
    jitter = rng.uniform(-1.0, 1.0, (2, burn + read)) * envelope
    for rewards in (worst, np.array([new, base])[:, None] + jitter):
        blocks = _per_block(BlockBatch(rewards, np.array([True, False]), np.full(2, 0.5)), burn)
        np.testing.assert_array_equal(blocks.candidate, [1.0, 0.0])
        contrast = blocks.reward[0] - blocks.reward[1]
        assert abs(contrast - (new - base)) <= bound + 1e-12
    worst_blocks = _per_block(BlockBatch(worst, np.array([True, False]), np.full(2, 0.5)), burn)
    contrast = worst_blocks.reward[0] - worst_blocks.reward[1]
    assert contrast - (new - base) == pytest.approx(bound, abs=1e-12)


def _carryover_rewards(ran: np.ndarray, before: bool, gain: float, cost: float) -> np.ndarray:
    """Rewards 1/2 + gain u_t - cost u_(t-1) of blocks of two periods, ``ran`` a block's policy."""
    actions = np.repeat(ran.astype(np.float64), 2)
    last = np.concatenate([[float(before)], actions[:-1]])
    return (0.5 + gain * actions - cost * last).reshape(-1, 2)


def test_block_read_under_carryover_of_order_one() -> None:
    """gate_block_contrast.mac STEP 3: one burned period reads gain - cost, none gain - cost/2."""
    gain, cost = 0.3, 0.4
    contrasts = {0: [], 1: []}
    for before in (False, True):
        rewards = np.vstack(
            [_carryover_rewards(np.array([ran]), before, gain, cost) for ran in (True, False)]
        )
        batch = BlockBatch(rewards, np.array([True, False]), np.full(2, 0.5))
        for burn_in in (0, 1):
            read = _per_block(batch, burn_in).reward
            contrasts[burn_in].append(read[0] - read[1])
    assert contrasts[1] == pytest.approx([gain - cost] * 2, abs=1e-15)
    # unburned, the first read period carries the block before, the candidate's half the time
    assert np.mean(contrasts[0]) == pytest.approx(gain - cost / 2.0, abs=1e-15)


@pytest.mark.parametrize(("burn_in", "verdict"), [(1, "hold"), (0, "deploy")])
def test_gate_in_blocks_reads_the_sustained_contrast_not_the_myopic_one(
    burn_in: int, verdict: str
) -> None:
    """gate_block_contrast.mac STEP 3: burned, the gate reads -0.1 and holds; unburned, +0.1."""
    gain, cost, rho = 0.3, 0.4, 0.5
    config = GateConfig(
        delta=0.0,
        delta_harm=0.0,
        min_effect=0.05,
        horizon=10**6,
        rho=rho,
        block_length=2,
        burn_in=burn_in,
    )
    gate = DeploymentGate({"zone": ZonePlan(chi2=1.0, tv=0.5)}, config)
    rng = np.random.default_rng(7)
    before, seen = False, "experiment"
    for _ in range(500):
        ran = rng.random(20) < rho
        rewards = _carryover_rewards(ran, before, gain, cost)
        before = bool(ran[-1])
        logged = np.where(ran, rho, 1.0 - rho)
        seen = gate.update({"zone": BlockBatch(rewards, ran, logged)})["zone"]
        if seen != "experiment":
            break
    assert seen == verdict


def _move_case(rng: np.random.Generator, states: int, actions: int):
    q = states * actions
    w_root = rng.normal(size=(q, q))
    s_root = 0.3 * rng.normal(size=(q, q))
    weight = DecisionWeight(w_root @ w_root.T, (states, actions), q, 0, 0.0)
    return weight, s_root @ s_root.T, rng.normal(size=q)


@settings(max_examples=8, deadline=None)
@given(seed=SEEDS, states=st.integers(1, 2), actions=st.integers(1, 2))
def test_move_price_keep_is_unbiased_and_replan_is_the_estimates_mean_regret(
    seed: int, states: int, actions: int
) -> None:
    """dither_channel_move.v keep_price_is_unbiased, replan_price; .mac STEPs 4-5."""
    rng = np.random.default_rng(seed)
    weight, covariance, move = _move_case(rng, states, actions)
    w = weight.matrix
    draws = rng.multivariate_normal(move, covariance, size=3000)
    prices = [
        ChannelMove(dh.reshape(states, actions), covariance, 1.0).price(weight) for dh in draws
    ]
    keep = np.array([p.keep for p in prices])
    assert abs(keep.mean() - move @ w @ move / 2.0) <= 5.0 * keep.std() / math.sqrt(keep.size)
    errors = draws - move
    regret = 0.5 * np.einsum("ni,ij,nj->n", errors, w, errors)
    assert {p.replan for p in prices} == {prices[0].replan}
    assert abs(regret.mean() - prices[0].replan) <= 5.0 * regret.std() / math.sqrt(regret.size)


def test_move_price_keep_error_is_the_keep_prices_spread() -> None:
    """dither_channel_move.mac STEP 4: Var(dh' W dh) = 4 d'WSWd + 2 tr((WS)^2), unbiased read."""
    rng = np.random.default_rng(11)
    weight, covariance, move = _move_case(rng, 2, 2)
    move = 3.0 * move  # far from 0, so that the estimate of d'WSWd is rarely floored
    w = weight.matrix
    draws = rng.multivariate_normal(move, covariance, size=20_000)
    prices = [ChannelMove(dh.reshape(2, 2), covariance, 1.0).price(weight) for dh in draws]
    keep = np.array([p.keep for p in prices])
    ws = w @ covariance
    spread = math.sqrt(4.0 * move @ ws @ w @ move + 2.0 * np.trace(ws @ ws)) / 2.0
    centred = keep - keep.mean()
    variance_se = math.sqrt((np.mean(centred**4) - keep.var() ** 2) / keep.size)
    assert abs(keep.var() - spread**2) <= 5.0 * variance_se
    squares = np.array([p.keep_error**2 for p in prices])
    assert abs(squares.mean() - spread**2) <= 5.0 * squares.std() / math.sqrt(squares.size)


def test_drift_evalues_average_one_over_one_minus_theta_k() -> None:
    """dither_drift_evalue.mac STEPs 2-4: E[e | c] = 1/|1 - theta k| whatever c holds."""
    rng = np.random.default_rng(5)
    n, sigma = 40_000, 0.5
    xi = rng.standard_normal(n)
    nominal = 0.3 * np.sin(np.arange(n) / 5.0)
    action = nominal + sigma * xi
    moved, radius, scale = np.array([0.15, 0.02]), np.array([0.05, 0.05]), np.array([0.5, 1.0])
    model_error = np.column_stack([0.2 * np.cos(np.arange(n) / 9.0), -0.1 * nominal])
    residual = model_error + np.outer(action, moved) + 0.1 * rng.standard_normal((n, 2))
    log = DecisionLog(action, np.ones(n), np.zeros(n, dtype=bool), sigma * xi)
    e = channel_drift_evalues(
        log, residual, dither_scale=sigma, radius=radius[:, None], residual_scale=scale
    ).reshape(n, 2, 1, 2, 8)
    for i, (side, j) in itertools.product(range(2), itertools.product(range(2), range(8))):
        sign = 1.0 if side == 0 else -1.0
        theta = sign * 2.0**-j
        k = (moved[i] - sign * radius[i]) * sigma / scale[i]
        column = e[:, i, 0, side, j]
        expected = 1.0 / abs(1.0 - theta * k)
        assert abs(column.mean() - expected) <= 5.0 * column.std() / math.sqrt(n)
        if theta * k <= 0.0:
            assert expected <= 1.0


def test_clipped_drift_evalues_average_the_clipped_closed_form() -> None:
    """dither_drift_evalue.mac STEP 8: 1 - (Phi(u2) - Phi(u1)) (1 - 1/s), 1 on the radius's edge."""
    rng = np.random.default_rng(6)
    n, sigma, nominal, lo, hi = 60_000, 1.0, 0.2, -0.5, 1.0
    xi = rng.standard_normal(n)
    action = np.clip(nominal + sigma * xi, lo, hi)
    moved, radius, scale = np.array([0.1, -0.2]), np.array([0.1, 0.1]), np.array([1.0, 0.8])
    model_error = np.array([0.4, -0.3])
    residual = model_error + np.outer(action, moved)
    log = DecisionLog(action, np.ones(n), action != nominal + sigma * xi, sigma * xi)
    e = channel_drift_evalues(
        log, residual, dither_scale=sigma, radius=radius[:, None], residual_scale=scale
    ).reshape(n, 2, 1, 2, 8)
    m1, m2 = (nominal - lo) / sigma, (hi - nominal) / sigma
    for i, (side, j) in itertools.product(range(2), itertools.product(range(2), range(8))):
        sign = 1.0 if side == 0 else -1.0
        theta = sign * 2.0**-j
        k = (moved[i] - sign * radius[i]) * sigma / scale[i]
        c = (model_error[i] + (moved[i] - sign * radius[i]) * nominal) / scale[i]
        s = 1.0 - theta * k
        u1, u2 = -s * m1 - theta * c, s * m2 - theta * c
        expected = 1.0 - (stats.norm.cdf(u2) - stats.norm.cdf(u1)) * (1.0 - 1.0 / s)
        column = e[:, i, 0, side, j]
        assert abs(column.mean() - expected) <= 5.0 * column.std() / math.sqrt(n) + 1e-12
        if i == 0 and side == 0:
            assert expected == pytest.approx(1.0, abs=1e-15)


def test_channel_move_reads_the_move_without_bias_and_a_calibrated_covariance() -> None:
    """dither_channel_move.mac STEPs 1, 2, 6: E[z] = d whatever c; the covariance is unbiased."""
    rng = np.random.default_rng(8)
    sigma, n, runs = np.array([0.5, 1.0]), 150, 600
    move = np.array([[0.3, -0.2], [0.05, 0.4]])
    estimates, covariances = [], []
    for _ in range(runs):
        xi = rng.standard_normal((n, 2))
        nominal = np.column_stack([np.sin(np.arange(n) / 4.0), np.ones(n)])
        action = nominal + sigma * xi
        residual = (
            np.column_stack([0.5 * np.cos(np.arange(n) / 6.0), 0.2 * nominal[:, 0]])
            + action @ move.T
            + 0.3 * rng.standard_normal((n, 2))
        )
        log = DecisionLog(action, np.ones(n), np.zeros(n, dtype=bool), sigma * xi)
        read = channel_move(log, residual, dither_scale=sigma)
        estimates.append(read.estimate.ravel())
        covariances.append(read.covariance)
    estimates = np.array(estimates)
    spread = estimates.std(axis=0)
    assert np.all(np.abs(estimates.mean(axis=0) - move.ravel()) <= 5.0 * spread / math.sqrt(runs))
    reported = np.mean([np.diag(c) for c in covariances], axis=0)
    # the variance of a sample variance over runs: 2 sigma^4 / (runs - 1) at the Gaussian
    np.testing.assert_allclose(reported, spread**2, rtol=5.0 * math.sqrt(2.0 / (runs - 1)))


@pytest.mark.parametrize(("forgetting", "n"), [(0.99, 1000), (0.9, 40), (1.0, 25)])
def test_channel_move_effective_size_is_the_geometric_closed_form(
    forgetting: float, n: int
) -> None:
    """dither_channel_move.mac STEP 3: (sum w)^2 / sum w^2 -> (1 + lam)/(1 - lam)."""
    rng = np.random.default_rng(9)
    xi = rng.standard_normal(n)
    log = DecisionLog(xi, np.ones(n), np.zeros(n, dtype=bool), xi)
    read = channel_move(log, rng.standard_normal(n), dither_scale=1.0, forgetting=forgetting)
    if forgetting == 1.0:
        assert read.effective_size == pytest.approx(n, rel=1e-14, abs=0.0)
        return
    lam = forgetting
    closed = ((1.0 - lam**n) / (1.0 - lam)) ** 2 / ((1.0 - lam ** (2 * n)) / (1.0 - lam**2))
    assert read.effective_size == pytest.approx(closed, rel=1e-12, abs=0.0)
    limit = (1.0 + lam) / (1.0 - lam)
    assert closed < limit


# ---------------------------------------------------------------------------------- chc.irf


@settings(max_examples=40, deadline=None)
@given(
    phi=st.floats(0.01, 0.99),
    lag=st.integers(1, 12),
    tail=st.integers(2, 30),
    sign=st.sampled_from([1.0, -1.0]),
)
def test_refined_peak_of_a_geometric_response_is_late_by_the_closed_form(
    phi: float, lag: int, tail: int, sign: float
) -> None:
    """delay_estimate_bias.mac: the vertex of a one-sided geometric decay is phi/(2 (2 - phi))."""
    irf = np.concatenate([np.zeros(lag), sign * phi ** np.arange(tail)])
    assert peak_lag(irf) == (float(lag), False)
    refined, censored = peak_lag(irf, refine=True)
    assert not censored
    assert refined == pytest.approx(lag + phi / (2.0 * (2.0 - phi)), rel=1e-12, abs=0.0)
    coefficients = np.polyfit([-1.0, 0.0, 1.0], [0.0, 1.0, phi], 2)
    assert refined - lag == pytest.approx(
        -coefficients[1] / (2.0 * coefficients[0]), rel=1e-9, abs=1e-12
    )


@settings(max_examples=40, deadline=None)
@given(share=st.floats(0.01, 0.5), lag=st.integers(1, 12))
def test_refined_peak_of_a_split_lag_is_f_over_4_minus_6f(share: float, lag: int) -> None:
    """delay_estimate_bias.mac: a lag split (1 - f, f) across two bins reads f/(4 - 6 f)."""
    irf = np.zeros(lag + 6)
    irf[lag], irf[lag + 1] = 1.0 - share, share
    refined, censored = peak_lag(irf, refine=True)
    assert not censored
    assert refined == pytest.approx(lag + share / (4.0 - 6.0 * share), rel=1e-12, abs=0.0)


def test_peak_at_an_end_is_censored_and_a_plateau_reads_its_leading_pair() -> None:
    """delay_estimate_bias.mac: two tied bins give 1/2; an edge peak is returned unrefined."""
    assert peak_lag([3.0, 1.0, 0.5], refine=True) == (0.0, True)
    assert peak_lag([0.1, 0.5, -2.0], refine=True) == (2.0, True)
    assert peak_lag([0.0, 0.0, 1.0, 1.0, 0.0], refine=True) == (2.5, False)
    assert peak_lag([0.0, 1.0, 1.0, 1.0, 0.0], refine=True) == (1.5, False)


# ------------------------------------------------------------------------------ chc.pathway

_EDGES = st.lists(
    st.tuples(
        st.integers(0, 4),
        st.integers(0, 4),
        st.integers(1, 4),
        st.sampled_from(["state", "control"]),
    ),
    max_size=14,
)


@settings(max_examples=40, deadline=None)
@given(edges=_EDGES, target=st.integers(0, 4))
def test_ancestor_onsets_are_the_shortest_lagged_walks(
    edges: list[tuple[int, int, int, str]], target: int
) -> None:
    """causal_pathway.mac L1: a source reaches the target first at its shortest lagged walk."""
    best: dict[tuple[str, int], int] = {}
    changed = True
    while changed:  # Bellman-Ford: every lag is positive, so it settles
        changed = False
        for into, source, lag, kind in edges:
            onward = 0 if into == target else best.get(("state", into))
            if onward is not None and onward + lag < best.get((kind, source), math.inf):
                best[(kind, source)] = onward + lag
                changed = True
    assert _ancestor_onsets(edges, target) == best


def _impulse(steps: int, coefficients: dict[str, float]) -> np.ndarray:
    """The noiseless response of x to a unit lever at t = 0 on u -> m -> x."""
    m, x = np.zeros(steps + 1), np.zeros(steps + 1)
    u = np.zeros(steps + 1)
    u[0] = 1.0
    for t in range(1, steps + 1):
        m[t] = coefficients["mm"] * m[t - 1] + coefficients["um"] * u[t - 1]
        x[t] = coefficients["xx"] * x[t - 1] + coefficients["mx"] * m[t - 1]
    return x


def test_pathway_lever_response_is_the_product_along_its_chain() -> None:
    """causal_pathway.v irf_g2_path_sum, weakest_link_product: g_2 of u -> m -> x is 0.8 * -0.9."""
    coefficients = {"mm": 0.5, "um": 0.8, "xx": 0.6, "mx": -0.9}
    rng = np.random.default_rng(2)
    n, horizon = 3000, 6
    u, decoy = rng.standard_normal(n), rng.standard_normal(n)
    m, x = np.zeros(n), np.zeros(n)
    noise = 0.1 * rng.standard_normal((n, 2))
    for t in range(1, n):
        m[t] = coefficients["mm"] * m[t - 1] + coefficients["um"] * u[t - 1] + noise[t, 0]
        x[t] = coefficients["xx"] * x[t - 1] + coefficients["mx"] * m[t - 1] + noise[t, 1]
    path = causal_pathway(np.column_stack([x, m, decoy]), 0, u[:, None], horizon, max_lag=2)
    edges = {(edge.kind, edge.source): edge for edge in path.edges}
    lever = edges[("control", 0)]
    truth = _impulse(horizon, coefficients)
    assert truth[2] == pytest.approx(0.8 * -0.9, rel=1e-15, abs=0.0)
    # the lever is white with unit variance, so each coefficient's error is about the residual
    # spread over sqrt(n): under 0.04 here
    np.testing.assert_allclose(lever.irf, truth, atol=0.12)
    assert (lever.lag, lever.sign, lever.actionable) == (2, -1, True)
    assert edges[("state", 1)].lag == 1
    assert ("state", 2) not in edges or edges[("state", 2)].contribution < 0.1 * lever.contribution


def test_structured_irf_tail_past_the_horizon_is_geometric() -> None:
    """causal_pathway.v geo_sum_closed, geometric_tail_decreasing: the tail is |g_H| r/(1 - r)."""
    rng = np.random.default_rng(4)
    n, rho, impact, horizon = 4000, 0.7, 0.5, 8
    u = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + impact * u[t - 1] + 0.2 * rng.standard_normal()
    irf = structured_irf({"x": jnp.asarray(x), "u": jnp.asarray(u)}, 400, order=1)
    np.testing.assert_allclose(irf[1 : horizon + 1], impact * rho ** np.arange(horizon), atol=0.03)
    ratio = irf[horizon + 1] / irf[horizon]
    tail = float(np.sum(np.abs(irf[horizon + 1 :])))
    bound = abs(irf[horizon]) * ratio / (1.0 - ratio)
    assert tail <= bound * (1.0 + 1e-12)
    assert tail == pytest.approx(bound, rel=1e-9, abs=0.0)
    terms = np.abs(irf[horizon + 1 : 60])
    np.testing.assert_allclose(terms[1:] / terms[:-1], ratio, rtol=1e-9)


# ----------------------------------------------------------------------------- chc.response


def _s_curve(family: int, shape: float, scale: float):
    """An S-shaped curve of five families, ``shape`` in [0, 1] mapped into its S-shaped range."""
    if family == 0:
        return Hill(scale, 1.2 + 7.0 * shape)
    if family == 1:
        return Weibull(scale, 1.2 + 5.0 * shape)
    if family == 2:
        return Logistic(scale, 1.5 + 9.0 * shape)
    if family == 3:
        return Gompertz(scale, 1.5 + 18.0 * shape)
    return ChapmanRichards(scale, 1.5 + 5.0 * shape)


def _standard(curve, z: np.ndarray | float) -> np.ndarray:
    return np.asarray(curve.standard(jnp.asarray(z, dtype=jnp.float64)), dtype=np.float64)


def _argmax(f, grid: np.ndarray) -> tuple[float, float]:
    """The largest of ``f`` on a grid, refined by bounded Brent between the grid's neighbours."""
    values = f(grid)
    i = int(np.argmax(values))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]
    found = optimize.minimize_scalar(
        lambda z: -float(f(np.array([z]))[0]),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-14 * hi},
    )
    return float(found.x), -float(found.fun)


_CURVES = st.tuples(st.integers(0, 4), st.floats(0.0, 1.0), st.floats(0.05, 50.0))


@settings(max_examples=15, deadline=None)
@given(curve=_CURVES)
def test_tangency_maximises_the_return_per_unit_spend(curve: tuple[int, float, float]) -> None:
    """response_curves.mac STEP 3: g(t) = t g'(t) past the inflection, where g/z is largest."""
    family, shape, scale = curve
    s = _s_curve(family, shape, scale)
    touch = s.tangency()
    spend, best = _argmax(
        lambda x: np.asarray(s(jnp.asarray(x)), dtype=np.float64) / x,
        scale * np.geomspace(1e-3, 1e2, 4001),
    )
    assert touch == pytest.approx(spend, rel=1e-5, abs=0.0)
    assert float(s(touch)) / touch == pytest.approx(best, rel=1e-12, abs=0.0)
    assert touch > s.inflection()
    for concave in (MichaelisMenten(scale), Exponential(scale)):
        assert concave.tangency() == 0.0
        assert concave.inflection() == 0.0


@settings(max_examples=15, deadline=None)
@given(curve=_CURVES)
def test_inflection_is_where_the_curvature_changes_sign(curve: tuple[int, float, float]) -> None:
    """response_curves.mac STEP 2: g'' = 0 at the stated inflection, convex below, concave above."""
    family, shape, scale = curve
    s = _s_curve(family, shape, scale)
    bend = s.inflection() / scale
    curvature = jax.grad(jax.grad(lambda z: s.standard(z)))
    assert float(curvature(jnp.float64(bend * 0.99))) > 0.0
    assert float(curvature(jnp.float64(bend * 1.01))) < 0.0
    slope = float(jax.grad(lambda z: s.standard(z))(jnp.float64(bend)))
    assert abs(float(curvature(jnp.float64(bend)))) <= 1e-7 * max(slope / bend, 1.0)


@settings(max_examples=8, deadline=None)
@given(curve=_CURVES, factor=st.floats(0.01, 100.0))
def test_a_change_of_currency_moves_only_the_scale(
    curve: tuple[int, float, float], factor: float
) -> None:
    """chc.response: h(spend) = g(spend / K), so K moves every reading in spend with it."""
    family, shape, scale = curve
    one, other = _s_curve(family, shape, scale), _s_curve(family, shape, scale * factor)
    spend = scale * np.array([0.1, 0.7, 1.3, 4.0])
    np.testing.assert_allclose(
        np.asarray(other(jnp.asarray(spend * factor))),
        np.asarray(one(jnp.asarray(spend))),
        rtol=1e-12,
        atol=1e-15,
    )
    assert other.tangency() == pytest.approx(factor * one.tangency(), rel=1e-9, abs=0.0)
    assert other.inflection() == pytest.approx(factor * one.inflection(), rel=1e-12, abs=0.0)
    assert other.nonconvexity() == pytest.approx(one.nonconvexity(), rel=1e-9, abs=1e-15)


@settings(max_examples=12, deadline=None)
@given(curve=_CURVES)
def test_nonconvexity_is_the_largest_rise_of_the_chord_over_the_curve(
    curve: tuple[int, float, float],
) -> None:
    """envelope_nonconvexity.mac: rho = max over 0 < z < t of g(t) z / t - g(z)."""
    family, shape, scale = curve
    s = _s_curve(family, shape, scale)
    touch = s.tangency() / scale
    chord = float(_standard(s, touch)) / touch
    _, rise = _argmax(lambda z: chord * z - _standard(s, z), np.linspace(0.0, touch, 4001))
    assert s.nonconvexity() == pytest.approx(rise, rel=1e-7, abs=1e-14)


@settings(max_examples=15, deadline=None)
@given(slope=st.floats(1.05, 30.0))
def test_hill_nonconvexity_is_the_reduced_root(slope: float) -> None:
    """envelope_nonconvexity.mac STEP 1: rho = (A-1)(n-A)/A^2, (A-1)/(n-1) = (A/n)^(2n/(n-1))."""
    n = slope

    def gap(a: float) -> float:
        return (a - 1.0) / (n - 1.0) - (a / n) ** (2.0 * n / (n - 1.0))

    grid = np.linspace(1.0, n, 20001)[1:-1]
    top = grid[int(np.argmax((grid - 1.0) / (n - 1.0) - (grid / n) ** (2.0 * n / (n - 1.0))))]
    a = optimize.brentq(gap, 1.0 + 1e-15, top, xtol=1e-15, rtol=4 * np.finfo(float).eps)
    assert Hill(1.0, n).nonconvexity() == pytest.approx(
        (a - 1.0) * (n - a) / a**2, rel=1e-7, abs=0.0
    )


def test_hill_nonconvexity_anchors_and_threshold() -> None:
    """envelope_nonconvexity.mac STEPs 2, 4, 6: 0.0674, 2/sqrt(3) - 1, 0.2920; a(1-a) eps^2."""
    assert Hill(1.0, 2.0).nonconvexity() == pytest.approx(0.0674, abs=5e-5)
    assert Hill(1.0, 3.0).nonconvexity() == pytest.approx(2.0 / math.sqrt(3.0) - 1.0, rel=1e-10)
    assert Hill(1.0, 5.0).nonconvexity() == pytest.approx(0.2920, abs=5e-5)
    assert Hill(1.0, 400.0).nonconvexity() > 0.95
    a = -float(np.real(lambertw(-2.0 / math.e**2))) / 2.0
    assert a == pytest.approx(0.2031879, abs=1e-7)
    for eps in (1e-2, 3e-3):
        leading = a * (1.0 - a) * eps**2 * (1.0 - (1.0 + a) * eps)
        assert Hill(1.0, 1.0 + eps).nonconvexity() == pytest.approx(leading, rel=3.0 * eps**2)
    assert Hill(1.0, 1.0).nonconvexity() == 0.0


# --------------------------------------------------------------------------- chc.switchback


def _states(lever: np.ndarray, memory: float, channel: float, start: float, shocks: np.ndarray):
    x = np.empty(lever.size + 1)
    x[0] = start
    for t in range(lever.size):
        x[t + 1] = memory * x[t] + channel * lever[t] + shocks[t]
    return x


@settings(max_examples=40, deadline=None)
@given(seed=SEEDS, memory=st.floats(0.0, 0.99), channel=st.floats(-3.0, 3.0))
def test_lever_path_enters_the_readings_through_the_memory_kernel(
    seed: int, memory: float, channel: float
) -> None:
    """switchback_randomisation.mac STEP 1: x(w) - x(0) = b sum_s a^(s-1) w_(t-s), any shocks."""
    rng = np.random.default_rng(seed)
    lever = (rng.random(24) < 0.5).astype(np.float64)
    shocks, start = rng.standard_normal(24), float(rng.normal())
    moved = _states(lever, memory, channel, start, shocks)
    still = _states(np.zeros(24), memory, channel, start, shocks)
    kernel = _kernel(lever[None, :], memory)[0]
    np.testing.assert_allclose(moved - still, channel * kernel, rtol=1e-12, atol=1e-12)
    # the first H terms of the kernel of a lever held on are tau_H / b = S_H(a)
    held = _kernel(np.ones((1, 5)), memory)[0]
    assert held[3] == pytest.approx(1.0 + memory + memory**2, rel=1e-14, abs=0.0)


def test_randomisation_interval_misses_on_at_most_alpha_of_the_schedules() -> None:
    """switchback_randomisation.mac STEP 1: the joint null (a, b) is sharp, so the interval at the
    plant's memory misses tau_H on at most alpha of the design's schedules, enumerated."""
    memory, channel, alpha, blocks, length = 0.7, 1.0, 0.25, 6, 3
    design, estimand = BlockDesign(length, length - 1), Horizon(length)
    tau = channel * (1.0 - memory**length) / (1.0 - memory)
    rng = np.random.default_rng(12)
    shocks, start = rng.standard_normal(blocks * length), float(rng.normal())
    read = missed = 0
    for coins in itertools.product((0.0, 1.0), repeat=blocks):
        lever = np.repeat(np.array(coins), length)
        y = _states(lever, memory, channel, start, shocks)
        try:
            lo, hi = randomisation_interval(
                lever[None, :],
                y[None, :],
                estimand,
                "block_dim",
                design,
                (memory, memory),
                alpha=alpha,
            )
        except ValueError:
            continue
        read += 1
        missed += not lo <= tau <= hi
    assert read >= 2**blocks // 2
    assert missed <= alpha * read


# -------------------------------------------------------------------------------- chc.zones


def _settled(system: ZoneMarketSystem, incentive: np.ndarray) -> np.ndarray:
    """The plant's own steady state under a held incentive, by Newton on its vector field."""
    plant = system.plant()
    lever = jnp.concatenate([jnp.asarray(incentive), jnp.zeros(system.zones)])
    field = jax.jit(lambda x: plant(0.0, x, lever))
    jacobian = jax.jit(jax.jacfwd(lambda x: plant(0.0, x, lever)))
    x = system.do_nothing
    for _ in range(30):
        x = x - jnp.linalg.solve(jacobian(x), field(x))
    assert float(jnp.max(jnp.abs(field(x)))) < 1e-12
    return np.asarray(plant.trips(x), dtype=np.float64)


def _system(seed: int, matching: str) -> ZoneMarketSystem:
    rng = np.random.default_rng(seed)
    zones = 4
    return ZoneMarketSystem(
        supply=tuple(rng.uniform(2.0, 9.0, zones)),
        queue=tuple(rng.uniform(1.0, 5.0, zones)),
        recruit=tuple(rng.uniform(0.0, 2.0, zones)),
        carry=tuple(rng.uniform(0.0, 0.7, zones)),
        elasticity=tuple(rng.uniform(0.0, 0.9, zones)),
        gamma=float(rng.uniform(0.05, 0.5)),
        alpha=float(rng.uniform(0.1, 0.6)),
        theta=float(rng.uniform(0.1, 0.6)),
        spill=float(rng.uniform(0.0, 1.0)),
        matching=matching,
    )


def _predicted(system: ZoneMarketSystem, incentive: np.ndarray) -> np.ndarray:
    decision = system.zone_decision(
        target=np.zeros(system.zones),
        state_weight=np.eye(system.zones),
        action_weight=np.eye(system.zones),
    )
    return decision.baseline + (decision.coupling * system.incentive_channel) @ incentive


@settings(max_examples=8, deadline=None)
@given(seed=SEEDS, incentive=st.lists(st.floats(0.0, 1.0), min_size=4, max_size=4))
def test_zone_decision_is_the_settled_trips_under_linear_matching(
    seed: int, incentive: list[float]
) -> None:
    """zone_market.mac step 3: dm/df = alpha mu_s / D, exact under linear matching."""
    system = _system(seed, "linear")
    lever = np.asarray(incentive)
    np.testing.assert_allclose(_predicted(system, lever), _settled(system, lever), rtol=1e-10)


@settings(max_examples=5, deadline=None)
@given(seed=SEEDS, zone=st.integers(0, 3))
def test_zone_decision_is_the_first_order_response_under_harmonic_matching(
    seed: int, zone: int
) -> None:
    """zone_market.mac steps 1, 3: the linear law is the harmonic one's tangent at do-nothing."""
    system = _system(seed, "harmonic")
    step = 1e-6
    lever = np.zeros(system.zones)
    lever[zone] = step
    moved = (_settled(system, lever) - _settled(system, np.zeros(system.zones))) / step
    expected = (_predicted(system, lever) - _predicted(system, np.zeros(system.zones))) / step
    np.testing.assert_allclose(moved, expected, rtol=1e-4, atol=1e-8)
