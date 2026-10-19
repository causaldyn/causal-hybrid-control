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
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy import integrate
from scipy.optimize import brentq, linprog, minimize
from scipy.signal import lfilter
from scipy.stats import f as fisher
from scipy.stats import norm, t

from chc import QuadraticCost
from chc.decision import _field_scale
from chc.dynamics_id import CausalDynamicsFit
from chc.evaluation import AffinePolicy, EvaluationMethod, LinearGaussianPlant, evaluate_plan
from chc.plan import ChannelSensitivity, _row_bound, _sensitivity_gamma_star
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
    "chc.evaluation": {
        "cites": ("dr_logged_weights.mac", "mis_model_error.mac"),
        "tests": ("test_mis_reads_the_model_s_cost_and_dr_removes_its_first_order_error",),
    },
    "chc.plan.ChannelSensitivity": {
        "cites": ("channel_bias_bound.mac",),
        "tests": (
            "test_channel_factor_is_the_bhatia_davis_share",
            "test_channel_factor_inverts_and_ends_at_the_limit",
            "test_channel_radius_bounds_the_bias_of_a_discrete_confounded_law",
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
GAMMA_LIMIT = (3.0 + math.sqrt(5.0)) / 2.0


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
    """kappa(Gamma) from rho^2 = (Gamma - 1)^2 / Gamma as kappa^2 = rho^2 / (1 - rho^2)."""
    rho2 = (gamma - 1.0) ** 2 / gamma
    return math.sqrt(rho2 / (1.0 - rho2))


@settings(max_examples=30, deadline=None)
@given(gamma=st.floats(1.0, 2.6))
def test_channel_factor_is_the_bhatia_davis_share(gamma) -> None:
    """channel_bias_bound.mac STEP 1 and 3a: the largest variance of w - 1 over laws of a weight
    w in [1/Gamma, Gamma] with mean 1, a linear program over a grid holding both ends, is
    rho^2 = (Gamma - 1)^2 / Gamma; the factor is kappa^2 = rho^2 / (1 - rho^2)."""
    grid = np.unique(np.concatenate([np.linspace(1.0 / gamma, gamma, 401), [1.0]]))
    program = linprog(
        -((grid - 1.0) ** 2),
        A_eq=np.vstack([np.ones_like(grid), grid]),
        b_eq=[1.0, 1.0],
        bounds=[(0.0, None)] * grid.size,
        method="highs",
    )
    assert program.status == 0
    rho2 = -float(program.fun)
    assert rho2 == pytest.approx((gamma - 1.0) ** 2 / gamma, rel=1e-9, abs=1e-15)
    if rho2 < 1.0:
        assert ChannelSensitivity.factor(gamma) == pytest.approx(
            math.sqrt(rho2 / (1.0 - rho2)), rel=1e-7, abs=1e-12
        )


@settings(max_examples=40, deadline=None)
@given(gamma=st.floats(1.0, GAMMA_LIMIT, exclude_max=True))
def test_channel_factor_inverts_and_ends_at_the_limit(gamma) -> None:
    """channel_bias_bound.mac STEP 3b-3d: level_at inverts factor below (3 + sqrt 5)/2, where the
    factor is infinite and from which on it stays so; G(0) = 1 and G(inf) is the limit."""
    kappa = ChannelSensitivity.factor(gamma)
    assume(math.isfinite(kappa) and kappa < 1e6)
    assert ChannelSensitivity.level_at(kappa) == pytest.approx(gamma, rel=1e-9, abs=0.0)
    assert ChannelSensitivity.factor(GAMMA_LIMIT) == math.inf
    assert ChannelSensitivity.factor(GAMMA_LIMIT + gamma) == math.inf
    assert ChannelSensitivity.level_at(0.0) == 1.0
    assert ChannelSensitivity.level_at(math.inf) == GAMMA_LIMIT
    assert ChannelSensitivity.level_at(1e12) == pytest.approx(GAMMA_LIMIT, rel=1e-12, abs=0.0)


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
    lies in [1/Gamma, Gamma], and the rate's noise g(x, Z) + eta -- the least-squares channel on
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
    gamma = float(max(ratio.max(), (1.0 / ratio).max()))
    assume(gamma < 0.99 * GAMMA_LIMIT)
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
    = threshold, below (3 + sqrt 5)/2; the radius grows in Gamma, so every smaller level holds."""
    threshold = unit * (sampling + room)
    star = _sensitivity_gamma_star(threshold, unit, _sensitivity(sampling, confounding))
    assert 1.0 < star < GAMMA_LIMIT
    upper = GAMMA_LIMIT * (1.0 - 1e-12)
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
