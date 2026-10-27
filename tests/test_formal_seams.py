"""The seams' cited derivations, checked on the float code that implements them.

Each test reads one statement of a Maxima derivation that a docstring in ``chc.decision``,
``chc.evaluation``, ``chc.plan`` or ``chc.switchback`` cites, and checks it on the library's output
by another route: quadrature of the studentised gap against the tube's scale, a linear program and
exact enumeration of discrete confounded laws against the channel's sensitivity radius, bisection
against the largest certified level, a sequential quadratic program and vertex enumeration against
the regret bound's dual, and Monte Carlo against the closed forms of the "mis" and "dr" limits and
of the local projection's bias.
"""

from __future__ import annotations

import itertools
import logging
import math

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import integrate
from scipy.optimize import brentq, linprog, minimize
from scipy.signal import lfilter
from scipy.stats import f as fisher
from scipy.stats import norm, t

from chc import QuadraticCost
from chc.decision import _field_scale
from chc.dynamics import LinearDynamics, _EulerStep
from chc.dynamics_id import CausalDynamicsFit
from chc.evaluation import AffinePolicy, EvaluationMethod, LinearGaussianPlant, evaluate_plan
from chc.integrate import rollout
from chc.plan import (
    CausalPlan,
    ChannelSensitivity,
    _barrier_room,
    _deviation_tube,
    _room_curvature,
    _row_bound,
    _sensitivity_gamma_star,
    certify_safety,
)
from chc.residual import ControlAffineResidual
from chc.switchback import Horizon, read_switchback

FORMAL_LINKS = {
    "chc.decision._field_scale": {
        "cites": ("tube_level.mac",),
        "tests": (
            "test_field_scale_squared_is_the_f_quantile",
            "test_field_scale_holds_one_state_at_its_level",
            "test_field_scale_holds_two_states_at_its_level",
        ),
    },
    "chc.decision._field_errors": {
        "cites": ("rk4_inherited.mac", "tube_projected_action.mac", "rk4_stage_term.mac"),
        "tests": (
            "test_field_terms_tube.py::test_the_field_s_error_is_the_delta_method_through_the_one_step_map",
            "test_field_terms_tube.py::test_two_levers_read_the_box_s_vertex_farthest_from_the_projected_action",
            "test_rk4_stage_tube.py::test_the_stages_read_is_never_narrower_than_the_plan_s_form_under_rk4",
            "test_rk4_stage_tube.py::"
            "test_a_stage_s_error_is_largest_at_a_vertex_where_the_channel_does_not_read_the_state",
        ),
    },
    "chc.dynamics_id._field_terms": {
        "cites": ("push_mean_variance.mac",),
        "via": "fit_causal_residual",
        "tests": (
            "test_field_terms_tube.py::test_the_field_s_error_is_the_delta_method_through_the_one_step_map",
        ),
    },
    "chc.evaluation": {
        "cites": ("dr_logged_weights.mac", "mis_model_error.mac"),
        "tests": ("test_mis_reads_the_model_s_cost_and_dr_removes_its_first_order_error",),
    },
    "chc.plan.ChannelSensitivity": {
        "cites": ("channel_bias_bound.mac",),
        "tests": (
            "test_channel_factor_is_the_sharp_share",
            "test_no_law_in_the_box_correlates_past_the_sharp_share",
            "test_channel_factor_inverts_at_every_level",
            "test_channel_radius_bounds_the_bias_of_a_discrete_confounded_law",
            "test_channel_radius_meets_the_bias_of_the_two_valued_lever",
        ),
    },
    "chc.plan._barrier_room": {
        "cites": ("barrier_room.mac",),
        "tests": (
            "test_a_disk_s_room_is_the_least_h_its_plant_reaches",
            "test_the_room_of_a_quadratic_barrier_is_at_most_its_least_h",
        ),
    },
    "chc.plan._deviation_tube": {
        "cites": ("barrier_deviation_tube.mac",),
        "tests": (
            "test_deviation_tube_is_the_gap_to_the_plant_at_the_radius_s_edge",
            "test_deviation_tube_bounds_the_gap_on_two_states_and_two_levers",
        ),
    },
    "chc.plan.certify_safety": {
        "cites": ("barrier_centred.mac", "barrier_deviation_tube.mac"),
        "tests": (
            "test_a_step_whose_plant_can_leave_the_set_is_not_certified",
            "test_deviation_tube_is_the_gap_to_the_plant_at_the_radius_s_edge",
            "test_centred_gamma_star_on_the_exact_two_lever_example",
        ),
    },
    "chc.plan._row_bound": {
        "cites": ("regret_bound_rows.mac",),
        "tests": (
            "test_row_bound_on_the_exact_two_lever_example",
            "test_row_bound_meets_the_concave_program",
            "test_row_bound_linear_program_is_the_best_vertex",
            "test_row_bound_refuses_an_empty_set",
        ),
    },
    "chc.plan._sensitivity_gamma_star": {
        "cites": ("channel_bias_bound.mac",),
        "tests": (
            "test_gamma_star_is_where_the_radius_meets_the_threshold",
            "test_gamma_star_reads_the_edge_cases",
        ),
    },
    "chc.switchback.read_switchback": {
        "cites": ("local_projection_bias.mac",),
        "tests": (
            "test_local_projection_bias_on_one_zone",
            "test_local_projection_bias_on_four_zones_depends_on_a",
        ),
    },
}

Z975 = float(norm.ppf(0.975))


# ---- the tube's scale: tube_level.mac ----


def _clustered_fit(clusters: int) -> CausalDynamicsFit:
    """A fit of one state and one lever whose channel's error is read over ``clusters`` clusters
    of two transitions each, so ``nu = clusters - 1``."""
    return CausalDynamicsFit(
        residual=ControlAffineResidual(drift=jnp.zeros((1, 2)), channel=jnp.zeros((1, 1, 2))),
        identified=True,
        method="orthogonal",
        folds=2,
        channel_error=0.1,
        drift_error=0.1,
        action_residual_variance=1.0,
        nuisance_r2_state=0.0,
        nuisance_r2_action=0.0,
        moment_norm=0.0,
        defect=jnp.zeros((2 * clusters, 1)),
        clusters=np.repeat(np.arange(clusters), 2),
        _lever_errors=np.array([[0.1]]),
    )


def _one_state_cover(radius: float, field: float, nu: int) -> float:
    """``P(|n + field T| <= radius)``, ``n`` standard normal, ``T`` Student's at ``nu``."""

    def inside(value: float) -> float:
        return float(
            (norm.cdf(radius - field * value) - norm.cdf(-radius - field * value))
            * t.pdf(value, nu)
        )

    return integrate.quad(inside, -np.inf, np.inf, epsabs=1e-13, epsrel=1e-12, limit=400)[0]


@settings(max_examples=20, deadline=None)
@given(clusters=st.integers(4, 53))
def test_field_scale_squared_is_the_f_quantile(clusters) -> None:
    """tube_level.mac (1): on one state kappa = t_nu(0.975) / z, so kappa^2 z^2 is the 0.95
    quantile of F(1, nu), read here off Fisher's distribution, not Student's."""
    kappa = _field_scale(_clustered_fit(clusters))
    assert (kappa * Z975) ** 2 == pytest.approx(
        float(fisher.ppf(0.95, 1, clusters - 1)), rel=1e-10, abs=0.0
    )


@settings(max_examples=12, deadline=None)
@given(clusters=st.integers(4, 53), field=st.sampled_from([0.25, 0.5, 1.0, 2.0, 4.0, 16.0]))
def test_field_scale_holds_one_state_at_its_level(clusters, field) -> None:
    """tube_level.mac (2): the radius sqrt(z^2 + (kappa z f)^2) holds n + f T, n standard normal
    and T Student's at nu from 3 to 52, with probability 0.9495 to 0.9525, by quadrature over T."""
    nu = clusters - 1
    kappa = _field_scale(_clustered_fit(clusters))
    radius = Z975 * math.sqrt(1.0 + (kappa * field) ** 2)
    assert 0.9495 <= _one_state_cover(radius, field, nu) <= 0.9525


def _disc_mass(radius2: float, wide: float, narrow: float) -> float:
    """``P(wide X^2 + narrow Y^2 <= radius2)`` for independent standard normals, over the chord."""
    edge = math.sqrt(radius2 / narrow)

    def chord(y: float) -> float:
        room = (radius2 - narrow * y * y) / wide
        return float(norm.pdf(y) * (2.0 * norm.cdf(math.sqrt(max(room, 0.0))) - 1.0))

    return integrate.quad(chord, -edge, edge, epsabs=1e-13, epsrel=1e-11, limit=200)[0]


def _two_state_cover(radius2: float, field: float, nu: int, narrow: float, widest: bool) -> float:
    """``P(|g + f T e| <= radius)``, ``g ~ N(0, diag(1, narrow))``, ``e`` the first axis where
    ``widest`` and the second otherwise: Gauss-Legendre over T's quantile and over the disc's
    chord, ``y = radius sin(theta)``, whose integrand is smooth at the chord's ends."""
    sd = (1.0, math.sqrt(narrow)) if widest else (math.sqrt(narrow), 1.0)
    nodes, weights = np.polynomial.legendre.leggauss(400)
    shift = field * t.ppf(0.5 * (nodes + 1.0), nu)[:, None]  # (T, 1)
    theta = 0.5 * math.pi * nodes[None, :]  # (1, chord)
    radius = math.sqrt(radius2)
    y, half = radius * np.sin(theta), radius * np.cos(theta)
    inside = norm.cdf((half - shift) / sd[0]) - norm.cdf((-half - shift) / sd[0])
    chord = norm.pdf(y / sd[1]) / sd[1] * inside * half * 0.5 * math.pi
    return float(0.5 * weights @ (chord @ weights))


@pytest.mark.parametrize("nu", [3, 52])
@pytest.mark.parametrize("narrow", [0.2, 1.0])
@pytest.mark.parametrize("field", [0.5, 2.0])
def test_field_scale_holds_two_states_at_its_level(nu, narrow, field) -> None:
    """tube_level.mac (3): two states, noise diag(1, l2). The radius^2 is the 0.95 quantile of
    (1 + (kappa f)^2) chi2_1 + l2 chi2_1, found here by root-finding a chord quadrature, and holds
    the gap at 0.9495 to 0.9525 when the field's error lies along the widest direction; along the
    least, at no less than 0.95."""
    kappa = _field_scale(_clustered_fit(nu + 1))
    wide = 1.0 + (kappa * field) ** 2
    radius2 = brentq(
        lambda r2: _disc_mass(r2, wide, narrow) - 0.95, 1e-6, 50.0 * wide, xtol=1e-13, rtol=1e-13
    )
    assert 0.9495 <= _two_state_cover(radius2, field, nu, narrow, widest=True) <= 0.9525
    if field == 2.0 and narrow == 0.2:
        least = _two_state_cover(
            brentq(lambda r2: _disc_mass(r2, 1.0 + kappa**2, narrow) - 0.95, 1e-6, 100.0),
            1.0,
            nu,
            narrow,
            widest=False,
        )
        assert least >= 0.95


# ---- "mis" and "dr" under a model error: mis_model_error.mac ----

_K, _O, _S, _Q, _R = 0.3, 0.05, 0.04, 2.0, 0.4


def _scalar_law(a: float, b: float, c: float, v: float) -> tuple[float, float]:
    """The stationary mean and variance of ``x`` under the plan ``u = K x + O + e``."""
    f = a + b * _K
    return (b * _O + c) / (1.0 - f), (b * b * _S + v) / (1.0 - f * f)


def _scalar_cost(a: float, b: float = -0.6, c: float = 0.0, v: float = 0.05) -> float:
    """The plan's average of ``(Q x^2 + R u^2) / 2`` on the plant ``(a, b, c, v)``."""
    m, s = _scalar_law(a, b, c, v)
    return 0.5 * (_Q * (s + m * m) + _R * (_K * _K * s + _S + (_K * m + _O) ** 2))


def test_mis_reads_the_model_s_cost_and_dr_removes_its_first_order_error() -> None:
    """mis_model_error.mac STEP 1-3 and dr_logged_weights.mac STEP 2 on the scalar loop a = 0.95,
    b = -0.6, the plan u = 0.3 x + 0.05 + N(0, 0.04), one log of 400 000 steps of the true plant,
    the model's a off by 0.05 and 0.1 on the same log. "mis" moves by the model's own cost less
    the true one, -0.0304 and -0.0507, its whole first-order error. "dr" weighs toward the plan's
    law under the plant fitted to the log, so its remainder is the fit's error times the model's,
    and on this log it does not move: over 8 logs its moves were 0 within SDs of 2.5e-4 and
    4.1e-4 (1.0e-4 and 1.7e-4 for "mis"); the tolerances are 4 SDs."""
    a = 0.95
    rng = np.random.default_rng(20261019)
    steps, logger_gain, logger_sd, noise = 400_000, 0.1, 0.5, 0.05
    e = logger_sd * rng.normal(size=steps)
    w = math.sqrt(noise) * rng.normal(size=steps)
    phi = a - 0.6 * logger_gain
    x0 = rng.normal(0.0, math.sqrt((0.36 * logger_sd**2 + noise) / (1.0 - phi * phi)))
    x = np.concatenate([[x0], lfilter([1.0], [1.0, -phi], -0.6 * e + w, zi=[phi * x0])[0]])
    logs = {"x": x[:, None], "u": (logger_gain * x[:-1] + e)[:, None]}
    plan = AffinePolicy(np.array([[_K]]), np.array([_O]), np.array([[_S]]))
    cost = QuadraticCost(
        Q=jnp.array([[_Q]]), R=jnp.array([[_R]]), Qf=jnp.array([[_Q]]), x_target=jnp.zeros(1)
    )

    def read(shift: float, method: EvaluationMethod) -> float:
        model = LinearGaussianPlant(
            np.array([[a - shift]]), np.array([[-0.6]]), np.zeros(1), np.array([[noise]])
        )
        return evaluate_plan(logs, plan, method, plant=model, cost=cost).value

    truth = _scalar_cost(a)
    mis0, dr0 = read(0.0, "mis"), read(0.0, "dr")
    assert mis0 == pytest.approx(truth, abs=4 * 5.3e-4)
    assert dr0 == pytest.approx(truth, abs=4 * 9.9e-4)
    moves = {}
    for shift, sd_mis, sd_dr in ((0.05, 1.0e-4, 2.5e-4), (0.1, 1.7e-4, 4.1e-4)):
        mis, dr = read(shift, "mis") - mis0, read(shift, "dr") - dr0
        assert mis == pytest.approx(_scalar_cost(a - shift) - truth, abs=4 * sd_mis)
        assert dr == pytest.approx(0.0, abs=4 * sd_dr)
        moves[shift] = mis
    # halving the error halves "mis"'s move, to first order
    assert moves[0.1] / moves[0.05] < 1.8


# ---- the channel's sensitivity radius: channel_bias_bound.mac ----


def _kappa(gamma: float) -> float:
    """kappa(Gamma) from rho = 1 - 1/Gamma as kappa^2 = rho^2 / (1 - rho^2), the denominator read
    as (1 - rho)(1 + rho) with 1 - rho = 1/Gamma, which does not cancel at a large Gamma."""
    rho = 1.0 - 1.0 / gamma
    return rho / math.sqrt((1.0 + rho) / gamma)


def _maximal_correlation(p: np.ndarray, q: np.ndarray, ratio: np.ndarray) -> float:
    """The maximal correlation of the discrete law ``p_i q_j ratio_ij``: the second singular value
    of ``diag(sqrt p) ratio diag(sqrt q)``, whose first is 1 (Renyi 1959)."""
    scaled = np.sqrt(p)[:, None] * ratio * np.sqrt(q)[None, :]
    return float(np.linalg.svd(scaled, compute_uv=False)[1])


@settings(max_examples=30, deadline=None)
@given(gamma=st.floats(1.0, 50.0))
def test_channel_factor_is_the_sharp_share(gamma) -> None:
    """channel_bias_bound.mac STEP 3a and 4a-4b: the lever of two values that takes the
    confounder's sign with probability 1 - 1/(2 Gamma) has density ratios 1/Gamma and
    2 - 1/Gamma, inside [1/Gamma, Gamma], and maximal correlation 1 - 1/Gamma with it, the share's
    root; the factor is kappa^2 = rho^2 / (1 - rho^2) at rho = 1 - 1/Gamma."""
    agree = 1.0 - 1.0 / (2.0 * gamma)
    ratio = 2.0 * np.array([[agree, 1.0 - agree], [1.0 - agree, agree]])
    assert ratio.min() == pytest.approx(1.0 / gamma, rel=1e-14, abs=0.0)
    assert ratio.max() <= gamma * (1.0 + 1e-15)
    half = np.full(2, 0.5)
    assert _maximal_correlation(half, half, ratio) == pytest.approx(
        1.0 - 1.0 / gamma, rel=1e-12, abs=1e-15
    )
    rho2 = (1.0 - 1.0 / gamma) ** 2
    assert ChannelSensitivity.factor(gamma) == pytest.approx(
        math.sqrt(rho2 / (1.0 - rho2)), rel=1e-13, abs=0.0
    )


def _vertex(p: np.ndarray, q: np.ndarray, gamma: float, objective: np.ndarray) -> np.ndarray | None:
    """The density ratio of largest ``sum objective_ij ratio_ij`` among the laws with marginals
    ``p`` and ``q`` whose ratio is at least ``1/gamma``, a vertex of that polytope."""
    rows, cols = p.size, q.size
    equal = [np.kron(np.eye(rows)[i], q) for i in range(rows)]
    equal += [np.kron(p, np.eye(cols)[j]) for j in range(cols)]
    program = linprog(
        -objective.ravel(),
        A_eq=np.array(equal),
        b_eq=np.ones(rows + cols),
        bounds=[(1.0 / gamma, None)] * (rows * cols),
        method="highs",
    )
    return program.x.reshape(rows, cols) if program.status == 0 else None


@settings(max_examples=40, deadline=None)
@given(
    seed=st.integers(0, 2**31 - 1),
    rows=st.integers(2, 6),
    cols=st.integers(2, 6),
    gamma=st.floats(1.01, 20.0),
)
def test_no_law_in_the_box_correlates_past_the_sharp_share(seed, rows, cols, gamma) -> None:
    """channel_bias_bound.mac STEP 1: over discrete laws whose density ratio is at least 1/Gamma,
    the factor's lower side alone, the maximal correlation never passes 1 - 1/Gamma. The laws read
    are the polytope's vertices a random objective reaches, then the ones an ascent on the
    correlation's linearisation at its top singular pair reaches, twenty steps of it, where the
    largest correlations lie. Up to 0.21 the share's root was (Gamma - 1)/sqrt(Gamma), which no
    law reaches at any Gamma > 1."""
    rng = np.random.default_rng(seed)
    p, q = rng.dirichlet(np.ones(rows)), rng.dirichlet(np.ones(cols))
    ratio = _vertex(p, q, gamma, rng.normal(size=(rows, cols)))
    assert ratio is not None  # the product law, ratio 1, is in the polytope
    sharp = 1.0 - 1.0 / gamma
    for _ in range(20):
        assert _maximal_correlation(p, q, ratio) <= sharp + 1e-9
        scaled = np.sqrt(p)[:, None] * ratio * np.sqrt(q)[None, :]
        left, _, right = np.linalg.svd(scaled)
        f, g = left[:, 1] / np.sqrt(p), right[1] / np.sqrt(q)
        moved = _vertex(p, q, gamma, (p * f)[:, None] * (q * g)[None, :])
        if moved is None:
            break
        ratio = moved
    assert _maximal_correlation(p, q, ratio) <= sharp + 1e-9


@settings(max_examples=40, deadline=None)
@given(gamma=st.floats(1.0, 1e300))
def test_channel_factor_inverts_at_every_level(gamma) -> None:
    """channel_bias_bound.mac STEP 3b-3d: the factor is finite at every finite level, which up to
    0.21 ended at (3 + sqrt 5)/2, and level_at inverts it there; G(0) = 1, the factor grows as
    sqrt(Gamma / 2), and inf maps to inf both ways."""
    kappa = ChannelSensitivity.factor(gamma)
    assert math.isfinite(kappa)
    assert ChannelSensitivity.level_at(kappa) == pytest.approx(gamma, rel=1e-12, abs=0.0)
    if gamma > 1e8:
        assert kappa == pytest.approx(math.sqrt(gamma / 2.0), rel=1e-7, abs=0.0)
    assert ChannelSensitivity.level_at(0.0) == 1.0
    assert ChannelSensitivity.factor(math.inf) == math.inf
    assert ChannelSensitivity.level_at(math.inf) == math.inf


@settings(max_examples=40, deadline=None)
@given(
    seed=st.integers(0, 2**31 - 1),
    levels=st.integers(2, 5),
    hidden=st.integers(2, 4),
    mix=st.floats(0.05, 1.0),
)
def test_channel_radius_bounds_the_bias_of_a_discrete_confounded_law(
    seed, levels, hidden, mix
) -> None:
    """channel_bias_bound.mac STEP 1-2 and the summary: on an exact discrete law -- a state x on
    three points, a hidden Z independent of it, the lever u | x, Z whose density ratio to u | x
    is at least 1/Gamma, and the rate's noise g(x, Z) + eta -- the least-squares channel on
    the action residuals a = u - E[u | x], with features [1, x], is off by at most the radius at
    every state, read with sampling 0, the design's second moment and the residual's. The radius
    is kappa(Gamma) sqrt(E r^2) times the largest |[1, x] c| over the ellipsoid c'Vc <= 1, read
    off a grid of its boundary."""
    rng = np.random.default_rng(seed)
    states = np.array([-1.0, 0.3, 1.5])
    p_state = rng.dirichlet(np.ones(3))
    p_hidden = rng.dirichlet(np.ones(hidden))
    actions = rng.uniform(-2.0, 2.0, (3, levels))
    base = rng.dirichlet(np.ones(levels), 3)
    tilt = rng.dirichlet(np.ones(levels), (3, hidden))
    given_hidden = (1.0 - mix) * base[:, None, :] + mix * tilt  # (x, Z, u)
    given_state = np.einsum("z,xzu->xu", p_hidden, given_hidden)
    ratio = given_hidden / given_state[:, None, :]
    gamma = float((1.0 / ratio).max())  # the factor's lower side, all the bound reads
    push = rng.normal(0.0, 1.0, (3, hidden))
    push -= np.einsum("z,xz->x", p_hidden, push)[:, None]  # what the drift does not absorb
    eta2 = float(rng.uniform(0.0, 1.0))

    mass = p_state[:, None, None] * p_hidden[None, :, None] * given_hidden  # (x, Z, u)
    residual = actions - np.einsum("xu,xu->x", given_state, actions)[:, None]
    features = np.stack([np.ones(3), states], 1)  # (x, 2)
    gram = np.einsum(
        "xu,xu,xf,xh->fh", p_state[:, None] * given_state, residual**2, features, features
    )
    cross = np.einsum("xzu,xu,xf,xz->f", mass, residual, features, push)
    beta = np.linalg.solve(gram, cross)
    noise2 = float(np.einsum("xzu,xz->", mass, push**2)) + eta2
    left = noise2 - float(beta @ gram @ beta)
    assert left > 0.0
    sensitivity = ChannelSensitivity(
        gram=gram, channel_degree=1, sampling=0.0, confounding=math.sqrt(left), level=0.95
    )
    # nu(x), the largest |[1, x] c| over the ellipsoid c'Vc <= 1, read off its boundary
    angle = np.linspace(0.0, 2.0 * np.pi, 20_001)
    values, vectors = np.linalg.eigh(gram)
    boundary = vectors @ (np.stack([np.cos(angle), np.sin(angle)]) / np.sqrt(values)[:, None])
    for x in (*states, -3.0, 0.0, 4.0):
        radius = sensitivity.radius([x], gamma)
        width = float(np.max(np.abs(np.array([1.0, x]) @ boundary)))
        assert radius == pytest.approx(width * _kappa(gamma) * math.sqrt(left), rel=1e-6, abs=0.0)
        assert abs(float(np.array([1.0, x]) @ beta)) <= radius * (1 + 1e-9) + 1e-14


@settings(max_examples=30, deadline=None)
@given(gamma=st.floats(1.0, 50.0), push=st.floats(0.1, 5.0))
def test_channel_radius_meets_the_bias_of_the_two_valued_lever(gamma, push) -> None:
    """channel_bias_bound.mac STEP 4c: Z = +-1, the lever u = +-1 taking Z's sign with probability
    1 - 1/(2 Gamma), and the rate's noise push Z alone. On the exact law, enumerated, the
    least-squares channel's bias, push (1 - 1/Gamma), is the radius at Gamma read with sampling
    0, to rounding: the bound cannot be narrowed at any level."""
    agree = 1.0 - 1.0 / (2.0 * gamma)
    lever = np.array([1.0, 1.0, -1.0, -1.0])
    hidden = np.array([1.0, -1.0, 1.0, -1.0])
    mass = 0.5 * np.where(lever == hidden, agree, 1.0 - agree)  # P(Z) P(u | Z)
    gram = float(np.sum(mass * lever**2))
    beta = float(np.sum(mass * lever * push * hidden)) / gram
    left = float(np.sum(mass * (push * hidden) ** 2)) - beta * gram * beta
    sensitivity = ChannelSensitivity(
        gram=np.array([[gram]]),
        channel_degree=0,
        sampling=0.0,
        confounding=math.sqrt(max(left, 0.0)),
        level=0.95,
    )
    assert beta == pytest.approx(push * (1.0 - 1.0 / gamma), rel=1e-12, abs=1e-15)
    assert sensitivity.radius([0.0], gamma) == pytest.approx(beta, rel=1e-9, abs=1e-12)


# ---- the largest certified level: channel_bias_bound.mac STEP 3c ----


def _sensitivity(sampling: float, confounding: float) -> ChannelSensitivity:
    return ChannelSensitivity(
        gram=np.eye(1), channel_degree=0, sampling=sampling, confounding=confounding, level=0.95
    )


@settings(max_examples=60, deadline=None)
@given(
    unit=st.floats(0.01, 10.0),
    sampling=st.floats(0.0, 2.0),
    confounding=st.floats(0.01, 5.0),
    room=st.floats(1e-6, 1e3),
)
def test_gamma_star_is_where_the_radius_meets_the_threshold(
    unit, sampling, confounding, room
) -> None:
    """channel_bias_bound.mac STEP 3c: the step's largest Gamma is the root, found by bisection
    on kappa(Gamma) = sqrt(rho^2 / (1 - rho^2)), of unit (sampling + kappa(Gamma) confounding)
    = threshold, finite at every finite threshold; the radius grows in Gamma, so every smaller
    level holds."""
    threshold = unit * (sampling + room)
    star = _sensitivity_gamma_star(threshold, unit, _sensitivity(sampling, confounding))
    assert 1.0 < star < math.inf
    upper = 2.0
    while unit * (sampling + _kappa(upper) * confounding) <= threshold:
        upper *= 2.0
    root = brentq(
        lambda g: unit * (sampling + _kappa(g) * confounding) - threshold,
        1.0,
        upper,
        xtol=1e-15,
        rtol=4 * np.finfo(float).eps,
    )
    assert star == pytest.approx(root, rel=1e-8, abs=0.0)
    below = 1.0 + 0.999 * (star - 1.0)
    assert unit * (sampling + _kappa(below) * confounding) <= threshold * (1 + 1e-12)


def test_gamma_star_reads_the_edge_cases() -> None:
    """channel_bias_bound.mac STEP 3c's ends: nan where the sampling part alone passes the
    threshold, 1 where it takes all of it, inf where nothing scales with Gamma."""
    model = _sensitivity(0.5, 2.0)
    assert math.isnan(_sensitivity_gamma_star(0.9, 2.0, model))
    assert _sensitivity_gamma_star(1.0, 2.0, model) == 1.0
    assert _sensitivity_gamma_star(math.inf, 2.0, model) == math.inf
    assert _sensitivity_gamma_star(5.0, 2.0, _sensitivity(0.5, 0.0)) == math.inf
    assert _sensitivity_gamma_star(0.0, 0.0, model) == math.inf
    assert math.isnan(_sensitivity_gamma_star(-1.0, 0.0, model))


# ---- the barrier's deviation tube: barrier_deviation_tube.mac ----


def _hand_plan(model: object, actions: np.ndarray, dt: float, states: int = 1) -> CausalPlan:
    """A plan of ``actions`` from 0, its trajectory the model's own rollout."""
    us = jnp.asarray(actions)
    return CausalPlan(
        actions=us,
        trajectory=rollout(model, jnp.zeros(states), us, dt),  # type: ignore[arg-type]
        task_cost=0.0,
        uncertainty_tube=None,
        certified_horizon=None,
        solver_status="converged",
        solver_iterations=0,
    )


def _ceiling(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 - x[0]


@pytest.mark.parametrize("euler", [False, True])
@pytest.mark.parametrize(("drift", "dt"), [(-1.0, 0.1), (0.0, 1.0), (0.3, 0.1), (0.3, 1.0)])
def test_deviation_tube_is_the_gap_to_the_plant_at_the_radius_s_edge(
    drift: float, dt: float, euler: bool
) -> None:
    """barrier_deviation_tube.mac (1), (2), (4): on one state, one lever and actions of one sign,
    the RK4 or Euler rollout of the plant at ``b_hat + Delta`` less the model's is the tube at
    every step, to rounding; on the stable, the integrating and the unstable drift."""
    b_hat, delta = 0.5021, 0.1004
    actions = np.linspace(0.3, 2.0, 12)[:, None]

    def stepped(channel: float) -> object:
        field = LinearDynamics(jnp.array([[drift]]), jnp.array([[channel]]))
        return _EulerStep(field) if euler else field

    model = stepped(b_hat)
    audit = certify_safety(
        _hand_plan(model, actions, dt),
        model,  # type: ignore[arg-type]
        _ceiling,
        dt,
        u_max=5.0,
        sensitivity=_sensitivity(delta, 0.0),
    )
    nominal = np.asarray(rollout(model, jnp.zeros(1), jnp.asarray(actions), dt))[:, 0]  # type: ignore[arg-type]
    edge = np.asarray(rollout(stepped(b_hat + delta), jnp.zeros(1), jnp.asarray(actions), dt))  # type: ignore[arg-type]
    gap = np.abs(edge[:, 0] - nominal)
    deviation = np.asarray(audit.deviation)
    assert deviation[0] == 0.0
    np.testing.assert_allclose(deviation, gap, rtol=1e-12, atol=1e-14)
    direct = _deviation_tube(
        model,  # type: ignore[arg-type]
        jnp.asarray(nominal[:-1, None]),
        jnp.asarray(actions),
        dt,
        delta,
    )
    np.testing.assert_allclose(np.asarray(direct), deviation, rtol=1e-12, atol=1e-14)


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), dt=st.sampled_from([0.1, 0.5, 1.0]))
def test_deviation_tube_bounds_the_gap_on_two_states_and_two_levers(seed: int, dt: float) -> None:
    """barrier_deviation_tube.mac (1): a 2-state, 2-lever field affine in the state, a channel
    error ``E`` with ``||E D||_2`` at the radius, the levers' largest actions 1 and 4, actions of
    either sign: the plant's RK4 rollout stays within the tube of the model's at every step."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 0.6, (2, 2))
    b = rng.normal(0.0, 1.0, (2, 2))
    reach = np.array([1.0, 4.0])
    shape = reach / reach.max()
    error = rng.normal(size=(2, 2))
    delta = 0.3
    error = error / np.linalg.norm(error * shape[None, :], 2) * delta
    actions = rng.uniform(-1.0, 1.0, (8, 2)) * reach
    gram = np.eye(2)
    sensitivity = ChannelSensitivity(
        gram=gram, channel_degree=0, sampling=delta, confounding=0.0, level=0.95
    )
    model = LinearDynamics(jnp.asarray(a), jnp.asarray(b))
    plant = LinearDynamics(jnp.asarray(a), jnp.asarray(b + error))
    audit = certify_safety(
        _hand_plan(model, actions, dt, states=2),
        model,
        lambda x: 100.0 - x[0],
        dt,
        u_max=reach,
        sensitivity=sensitivity,
    )
    xs = np.asarray(rollout(model, jnp.zeros(2), jnp.asarray(actions), dt))
    ys = np.asarray(rollout(plant, jnp.zeros(2), jnp.asarray(actions), dt))
    gap = np.linalg.norm(ys - xs, axis=1)
    deviation = np.asarray(audit.deviation)
    assert np.all(gap <= deviation * (1.0 + 1e-12) + 1e-14), (gap, deviation)


def test_a_step_whose_plant_can_leave_the_set_is_not_certified() -> None:
    """barrier_deviation_tube.mac (3): an integrator at ``dt = 1``, ``h = 1 - x``, ``b_hat = 1`` and
    an exact radius of 1/2, each action ``h(x_k) / 1.5`` meeting the worst case condition with
    equality. The model's path keeps ``h`` at 1/3, 1/9, 1/27, and the condition held at every
    step, so all three were certified; the plant at ``b = 3/2`` reads 0, -1/3, -4/9. The tube's
    room ``h(x_k) - e_k`` is that, so one step is certified, the one whose next state the plant
    reaches with ``h = 0``."""
    model = LinearDynamics(jnp.zeros((1, 1)), jnp.ones((1, 1)))
    x, actions = 0.0, []
    for _ in range(3):
        actions.append((1.0 - x) / 1.5)
        x += actions[-1]
    plan = _hand_plan(model, np.asarray(actions)[:, None], 1.0)
    audit = certify_safety(
        plan, model, _ceiling, 1.0, u_max=5.0, sensitivity=_sensitivity(0.5, 0.0)
    )
    assert np.all(np.asarray(audit.guaranteed_derivative - audit.required) >= -1e-15)
    room = 1.0 - np.asarray(plan.trajectory)[:, 0] - np.asarray(audit.deviation)
    np.testing.assert_allclose(room, [1.0, 0.0, -1.0 / 3.0, -4.0 / 9.0], rtol=0.0, atol=1e-15)
    assert audit.certified_steps == 1


# ---- the barrier's room: barrier_room.mac ----


def test_a_disk_s_room_is_the_least_h_its_plant_reaches() -> None:
    """barrier_room.mac STEP 5: an integrator at ``dt = 1``, ``h = 1 - x^2``, ``b_hat = 1``, an
    exact radius of 8/13 and one action 13/20, so the deviation at the next state is 2/5. The room
    read to first order, ``h - ||grad h|| e = 23/400``, certified the step; the plant at
    ``b = 21/13`` reaches ``h = -41/400``, which the room less ``kappa e^2 / 2``, ``kappa = 2``,
    reads exactly, so the step is not certified."""
    model = LinearDynamics(jnp.zeros((1, 1)), jnp.ones((1, 1)))
    plan = _hand_plan(model, np.array([[13.0 / 20.0]]), 1.0)

    def disk(x: jnp.ndarray) -> jnp.ndarray:
        return 1.0 - x[0] ** 2

    audit = certify_safety(
        plan, model, disk, 1.0, u_max=5.0, sensitivity=_sensitivity(8.0 / 13.0, 0.0)
    )
    assert audit.room_model == "quadratic"
    assert float(audit.deviation[1]) == pytest.approx(0.4, rel=1e-14, abs=0.0)
    assert bool(audit.guaranteed_derivative[0] >= audit.required[0])
    plant = LinearDynamics(jnp.zeros((1, 1)), jnp.full((1, 1), 21.0 / 13.0))
    reached = float(rollout(plant, jnp.zeros(1), plan.actions, 1.0)[1, 0])  # type: ignore[arg-type]
    room = _barrier_room(disk, plan.trajectory, audit.deviation, 2.0)
    assert float(disk(jnp.array([reached]))) == pytest.approx(-41.0 / 400.0, rel=1e-13, abs=0.0)
    assert float(room[1]) == pytest.approx(-41.0 / 400.0, rel=1e-13, abs=0.0)
    assert audit.certified_steps == 0
    assert (
        certify_safety(
            plan, model, disk, 1.0, u_max=5.0, sensitivity=_sensitivity(0.5, 0.0)
        ).certified_steps
        == 1
    )  # e = 13/40: the plant reaches h = 1 - (39/40)^2 > 0


def _trust_region_least(gradient: np.ndarray, hessian: np.ndarray, radius: float) -> float:
    """The least of ``g'd + d'Hd / 2`` over ``||d|| <= radius``: the trust region's subproblem, by
    the eigenbasis and a root of its secular equation (More and Sorensen 1983); the hard case,
    ``g`` orthogonal to the least eigenvector, has measure zero under the draws."""
    values, vectors = np.linalg.eigh(hessian)
    along = vectors.T @ gradient

    def step(mu: float) -> np.ndarray:
        return -along / (values + mu)

    if values[0] > 0.0 and np.linalg.norm(step(0.0)) <= radius:
        d = step(0.0)
    else:
        low = max(0.0, -values[0]) + 1e-14 * (1.0 + abs(values[0]))
        high = low + 1.0
        while np.linalg.norm(step(high)) > radius:
            high *= 2.0
        mu = brentq(lambda m: np.linalg.norm(step(m)) - radius, low, high, xtol=1e-15)
        d = step(mu)
    return float(along @ d + 0.5 * d @ (values * d))


@settings(max_examples=60, deadline=None)
@given(
    seed=st.integers(0, 2**31 - 1),
    states=st.integers(1, 3),
    ball=st.booleans(),
    radius=st.floats(1e-3, 2.0),
)
def test_the_room_of_a_quadratic_barrier_is_at_most_its_least_h(seed, states, ball, radius) -> None:
    """barrier_room.mac STEP 1-3: on ``h(y) = h0 + g'(y - x) + (y - x)'H(y - x)/2``, ``H`` drawn
    symmetric, the room at ``x`` with deviation ``e`` is read as quadratic, ``kappa`` the largest
    eigenvalue of ``-H`` clipped at 0, and is at most the least ``h`` over the ball, the trust
    region's subproblem; on ``H = -c I``, a ball's, it is that least, to rounding."""
    rng = np.random.default_rng(seed)
    centre = rng.normal(size=states)
    gradient = rng.normal(size=states)
    if ball:
        hessian = -abs(float(rng.normal())) * np.eye(states) - 0.1 * np.eye(states)
    else:
        square = rng.normal(size=(states, states))
        hessian = 0.5 * (square + square.T)
    h0 = float(rng.normal())
    h_matrix, g_vector, x_vector = (jnp.asarray(v) for v in (hessian, gradient, centre))

    def quadratic(y: jnp.ndarray) -> jnp.ndarray:
        d = y - x_vector
        return h0 + g_vector @ d + 0.5 * d @ h_matrix @ d

    path = jnp.asarray(centre[None, :] + rng.normal(size=(3, states)))
    path = path.at[0].set(x_vector)
    kappa, model = _room_curvature(quadratic, path, None)
    assert model == "quadratic"
    assert kappa == pytest.approx(
        max(float(np.linalg.eigvalsh(-hessian)[-1]), 0.0), rel=1e-12, abs=1e-15
    )
    room = float(_barrier_room(quadratic, path[:1], jnp.array([radius]), kappa)[0])
    least = h0 + _trust_region_least(gradient, hessian, radius)
    scale = abs(h0) + np.linalg.norm(gradient) * radius + np.abs(hessian).max() * radius**2
    assert room <= least + 1e-12 * scale
    if ball:
        assert room == pytest.approx(least, rel=0.0, abs=1e-12 * scale)


def test_the_room_reads_what_the_barrier_is() -> None:
    """An affine ``h`` reads ``affine`` and no curvature; a quadratic one its Hessian's; one whose
    Hessian moves along the path, ``1 - x^4``, ``first_order`` and the room of an affine ``h``,
    unless a ``curvature`` bounds it; several barriers read as the weakest; a curvature negative
    or not finite is refused."""
    model = LinearDynamics(jnp.zeros((1, 1)), jnp.ones((1, 1)))
    plan = _hand_plan(model, np.array([[0.2], [0.2], [0.2]]), 1.0)
    path = plan.trajectory
    assert _room_curvature(_ceiling, path, None) == (0.0, "affine")
    assert _room_curvature(lambda x: 1.0 - 3.0 * x[0] ** 2, path, None) == (6.0, "quadratic")
    assert _room_curvature(lambda x: 1.0 + x[0] ** 2, path, None) == (0.0, "quadratic")

    def quartic(x: jnp.ndarray) -> jnp.ndarray:
        return 1.0 - x[0] ** 4

    assert _room_curvature(quartic, path, None) == (0.0, "first_order")
    assert _room_curvature(quartic, path, 7.5) == (7.5, "bounded")
    radius = _sensitivity(0.5, 0.0)
    loose = certify_safety(plan, model, quartic, 1.0, u_max=5.0, sensitivity=radius)
    bounded = certify_safety(
        plan, model, quartic, 1.0, u_max=5.0, sensitivity=radius, curvature=12.0 * 0.9**2
    )
    assert (loose.room_model, bounded.room_model) == ("first_order", "bounded")
    first = _barrier_room(quartic, path, loose.deviation)
    np.testing.assert_array_equal(
        np.asarray(_barrier_room(quartic, path, loose.deviation, 0.0)), np.asarray(first)
    )
    several = certify_safety(plan, model, (_ceiling, quartic), 1.0, u_max=5.0, sensitivity=radius)
    assert several.room_model == "first_order"
    assert certify_safety(plan, model, (_ceiling,), 1.0, u_max=5.0).room_model == "affine"
    for wrong in (-1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="bounds no bend"):
            certify_safety(plan, model, quartic, 1.0, u_max=5.0, curvature=wrong)


# ---- the barrier read from the log's mean action: barrier_centred.mac ----


def _centred_example(centre: tuple[float, float] | None, shift: float = 0.0) -> list[float]:
    """barrier_centred.mac STEP 6's one state and two levers, ``u1``'s zero moved by ``shift``,
    the action and the centre with it: the step's ``gamma*`` under the relative radius and under
    the absolute one."""
    b = jnp.array([[0.3, 8.0 / 15.0]])

    class Shifted(LinearDynamics):
        def __call__(self, t, x, u):  # type: ignore[no-untyped-def]
            return self.a_matrix @ x + self.b_matrix @ (u - jnp.array([shift, 0.0]))

    model = Shifted(jnp.array([[-0.5]]), b)
    actions = jnp.array([[1.0 + shift, 0.5]])
    plan = CausalPlan(
        actions=actions,
        trajectory=rollout(model, jnp.array([0.25]), actions, 0.1),
        task_cost=0.0,
        uncertainty_tube=None,
        certified_horizon=None,
        solver_status="converged",
        solver_iterations=0,
    )
    moved = None if centre is None else (centre[0] + shift, centre[1])
    reach = (2.0, 1.5) if centre is not None else (3.0, 1.0)
    sensitivity = ChannelSensitivity(
        gram=np.eye(2) / 4.0, channel_degree=0, sampling=0.1, confounding=0.2, level=0.95
    )

    def ceiling(x: jnp.ndarray) -> jnp.ndarray:
        return 2.0 * (0.3 - x[0])

    return [
        certify_safety(
            plan, model, ceiling, 0.1, gamma=1.5, u_max=reach, sensitivity=one, centre=moved
        ).step_gamma_star[0]
        for one in (None, sensitivity)
    ]


@pytest.mark.parametrize("shift", [0.0, 100.0, -7.5])
def test_centred_gamma_star_on_the_exact_two_lever_example(shift: float) -> None:
    """barrier_centred.mac STEP 6: x' = -x/2 + (3/10) u1 + (8/15) u2, h = 2 (3/10 - x) at x = 1/4,
    the log's mean action (1, 1/2) and each lever's largest distance from it (2, 3/2). The
    threshold is d* = 73/120, so gamma* is 193/47 under the relative radius of gap 1 and
    1 + k^2 + k sqrt(1 + k^2) at k = 25/96 under the absolute one, sampling 1/10, confounding 1/5
    and nu = 2: the same wherever u1's zero is. Read from the zero, the box [-1, 3] x [-1, 1]
    giving (3, 1), the drift at zero alone keeps the step, and gamma* is inf."""
    k = 25.0 / 96.0
    relative, absolute = _centred_example((1.0, 0.5), shift)
    assert relative == pytest.approx(193.0 / 47.0, rel=1e-12, abs=0.0)
    assert absolute == pytest.approx(1.0 + k * k + k * math.sqrt(1.0 + k * k), rel=1e-12, abs=0.0)
    assert ChannelSensitivity.factor(absolute) == pytest.approx(k, rel=1e-12, abs=0.0)
    assert _centred_example(None) == [math.inf, math.inf]


# ---- the regret bound over a box and rows: regret_bound_rows.mac ----


def test_row_bound_on_the_exact_two_lever_example() -> None:
    """regret_bound_rows.mac STEP 3: g = (-2, -1), c = (1, 3), the box [-1, 1]^2 and the row
    d1 + d2 <= 1/4. The primal's maximum, at lam* = 25/16 and d* = (7/16, -3/16), is 69/128,
    and the dual meets it; the linear program's is 5/4, at the vertex (1, -3/4). The row binds,
    so the gains at d* sum to the bound."""
    gains, total, linear = _row_bound(
        np.array([-2.0, -1.0]),
        np.array([1.0, 3.0]),
        -np.ones(2),
        np.ones(2),
        np.array([[1.0, 1.0]]),
        np.array([0.25]),
    )
    assert total == pytest.approx(69.0 / 128.0, rel=1e-9, abs=0.0)
    assert linear == pytest.approx(1.25, rel=1e-12, abs=0.0)
    d = np.array([7.0, -3.0]) / 16.0
    np.testing.assert_allclose(
        gains, [2.0 * d[0] - 0.5 * d[0] ** 2, d[1] - 1.5 * d[1] ** 2], rtol=1e-6, atol=1e-9
    )


def _feasible_problem(seed: int, levers: int, rows: int):
    rng = np.random.default_rng(seed)
    below = -rng.uniform(0.1, 2.0, levers)
    above = rng.uniform(0.1, 2.0, levers)
    matrix = rng.normal(size=(rows, levers))
    right = rng.uniform(0.0, 1.0, rows)  # the plan itself, d = 0, meets the rows
    return rng, rng.normal(0.0, 2.0, levers), below, above, matrix, right


def _primal(gradient, curvature, below, above, matrix, right, starts) -> float:
    """The largest -g d - sum c d^2 / 2 over the box and the rows, by SLSQP from several starts."""
    best = 0.0  # d = 0 is feasible
    for start in starts:
        found = minimize(
            lambda d: float(gradient @ d + 0.5 * curvature @ d**2),
            start,
            jac=lambda d: gradient + curvature * d,
            method="SLSQP",
            bounds=list(zip(below, above, strict=True)),
            constraints=[
                {"type": "ineq", "fun": lambda d: right - matrix @ d, "jac": lambda d: -matrix}
            ],
            options={"ftol": 1e-14, "maxiter": 500},
        )
        d = np.clip(found.x, below, above)
        if np.all(matrix @ d <= right + 1e-12):
            best = max(best, -float(found.fun))
    return best


@settings(max_examples=40, deadline=None)
@given(
    seed=st.integers(0, 2**31 - 1),
    levers=st.integers(2, 4),
    rows=st.integers(1, 2),
    flat=st.booleans(),
)
def test_row_bound_meets_the_concave_program(seed, levers, rows, flat) -> None:
    """regret_bound_rows.mac STEP 1 (weak duality) and STEP 3c (strong duality): the dual bound is
    never below the box-and-rows maximum, found by SLSQP from five starts, and with every
    curvature positive it meets it. With some curvature zero only the first is asserted."""
    rng, gradient, below, above, matrix, right = _feasible_problem(seed, levers, rows)
    curvature = rng.uniform(0.2, 3.0, levers)
    if flat:
        curvature[rng.integers(levers)] = 0.0
    starts = [np.zeros(levers), *(rng.uniform(below, above) * 0.1 for _ in range(4))]
    best = _primal(gradient, curvature, below, above, matrix, right, starts)
    _, total, _ = _row_bound(gradient, curvature, below, above, matrix, right)
    assert total >= best - 1e-9 * max(1.0, abs(best))
    if not flat:
        assert total == pytest.approx(best, rel=1e-6, abs=1e-8)


def _best_vertex(gradient, below, above, matrix, right) -> float:
    """The linear program's optimum over the box and the rows, by enumerating its vertices."""
    levers = gradient.size
    walls = np.vstack([np.eye(levers), -np.eye(levers), matrix])
    sides = np.concatenate([above, -below, right])
    best = -math.inf
    for chosen in itertools.combinations(range(walls.shape[0]), levers):
        rows = list(chosen)
        if abs(np.linalg.det(walls[rows])) < 1e-12:
            continue
        d = np.linalg.solve(walls[rows], sides[rows])
        if np.all(walls @ d <= sides + 1e-9):
            best = max(best, -float(gradient @ d))
    return best


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), levers=st.integers(2, 4), rows=st.integers(1, 2))
def test_row_bound_linear_program_is_the_best_vertex(seed, levers, rows) -> None:
    """regret_bound_rows.mac STEP 3f: with c = 0 the bound is the linear program's, and its dual
    at the multipliers the program ends on is the best vertex of the box and the rows, found by
    enumerating every vertex; with every curvature 0 the two bounds are one."""
    _, gradient, below, above, matrix, right = _feasible_problem(seed, levers, rows)
    vertex = _best_vertex(gradient, below, above, matrix, right)
    _, total, linear = _row_bound(gradient, np.zeros(levers), below, above, matrix, right)
    assert linear == pytest.approx(vertex, rel=1e-8, abs=1e-9)
    assert total == pytest.approx(vertex, rel=1e-8, abs=1e-9)
    _, _, curved = _row_bound(gradient, np.ones(levers), below, above, matrix, right)
    assert curved == pytest.approx(vertex, rel=1e-8, abs=1e-9)


def test_row_bound_refuses_an_empty_set() -> None:
    """No move in the box [1/2, 1] meets d <= 0: there is no primal, and no bound."""
    with pytest.raises(ValueError, match="no plan in the box"):
        _row_bound(
            np.array([1.0]),
            np.array([1.0]),
            np.array([0.5]),
            np.array([1.0]),
            np.array([[1.0]]),
            np.array([0.0]),
        )


# ---- the local projection's bias: local_projection_bias.mac ----


def _switchback_runs(
    runs: int, zones: int, periods: int, a: float, b: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """``runs`` switchbacks of ``y' = a y + b u + eps``, a fair coin each period and zone, each
    started after 60 periods of the same plant."""
    rng = np.random.default_rng(seed)
    y = np.zeros((runs, zones))
    for _ in range(60):
        y = a * y + b * rng.integers(0, 2, (runs, zones)) + rng.normal(size=(runs, zones))
    lever = rng.integers(0, 2, (runs, zones, periods)).astype(np.float64)
    shocks = rng.normal(size=(runs, zones, periods))
    outcome = np.empty((runs, zones, periods + 1))
    outcome[:, :, 0] = y
    for period in range(periods):
        outcome[:, :, period + 1] = (
            a * outcome[:, :, period] + b * lever[:, :, period] + shocks[:, :, period]
        )
    return lever, outcome


def _projection_mean(zones: int, periods: int, runs: int, seed: int) -> tuple[float, float]:
    """The local projection's mean reading of tau_3 over ``runs`` switchbacks, and its standard
    error."""
    lever, outcome = _switchback_runs(runs, zones, periods, 0.8, 2.0, seed)
    logging.disable(logging.WARNING)  # every run is short: each warns it is
    try:
        readings = np.array(
            [
                read_switchback(lever[i], outcome[i], Horizon(3), "local_projection").estimate
                for i in range(runs)
            ]
        )
    finally:
        logging.disable(logging.NOTSET)
    return float(readings.mean()), float(readings.std(ddof=1) / math.sqrt(runs))


def _partial(m: int) -> float:
    return sum(0.8**i for i in range(m))


def test_local_projection_bias_on_one_zone() -> None:
    """local_projection_bias.mac STEP 3: on one zone of y' = 0.8 y + 2 u + eps over 120 periods,
    R = 118 sums, the local projection reads tau_3 = 4.88 low by (H - 1) tau_3 / R = 0.0827 to
    first order. Over 4000 runs the readings' SD is 0.71, the mean's 0.011: the bias is 7 of
    those from zero, and the mean is within 4 of the first-order value."""
    tau = 2.0 * _partial(3)
    mean, se = _projection_mean(1, 120, 4000, 11)
    first_order = -2.0 * tau / 118
    assert se < 0.02
    assert mean - tau == pytest.approx(first_order, abs=4 * se)
    assert mean - tau < -4 * se


def test_local_projection_bias_on_four_zones_depends_on_a() -> None:
    """local_projection_bias.mac STEP 3 on four zones over 200 periods, R = 198: the bias is
    -b (sum_k S_(H-k) + sum_k (S_H - S_k) / Z) / R = -0.0335 at a = 0.8, b = 2, not the
    one-zone -(H - 1) tau_3 / R = -0.0493. Over 8000 runs the mean's SD is 0.003: the reading
    is within 4 of the pooled value and more than 2 above the one-zone one."""
    tau = 2.0 * _partial(3)
    later = sum(_partial(3 - k) for k in (1, 2))
    spread = sum(_partial(3) - _partial(k) for k in (1, 2))
    pooled = -2.0 * (later + spread / 4) / 198
    assert pooled == pytest.approx(-0.033535, abs=1e-6)
    mean, se = _projection_mean(4, 200, 8000, 12)
    assert se < 0.004
    assert mean - tau == pytest.approx(pooled, abs=4 * se)
    assert mean - tau > -2.0 * tau / 198 + 2 * se
