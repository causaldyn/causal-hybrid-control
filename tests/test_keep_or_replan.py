"""Keep a plan whose one-step channel moved, or re-plan on the move's estimate.

The null law and its closed forms are ``validation/keep_or_replan.mac``'s. Here they are checked on
the library's Imhof inversion with noncentralities, on Gaussian estimates whose move sits where the
test's null is least favourable, and on logs of the budgeted plan of
``tests/test_decision_weight.py`` run with a dither on a plant whose one-step map is the model's
plus ``E u``. The re-solved plan is the minimiser of the cost the decision weight perturbs, at the
estimate, and on the plant it beats the plan it replaces.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, stats
from scipy.optimize import brentq, minimize

from chc import DecisionLog, LinearConstraint, QuadraticCost, causal_plan, channel_move
from chc.cost import total_cost
from chc.dynamics import LinearDynamics, _EulerStep
from chc.gate import ChannelMove, _null_law, _replan_test
from chc.integrate import rk4_step
from chc.plan import CausalPlan, DecisionWeight, _Moved, _MovedRate, _perturbed_task_cost
from chc.uncertainty import _chi_square_mixture_survival

MODEL = LinearDynamics(jnp.array([[-0.3, 0.2], [0.1, -0.5]]), jnp.array([[1.0, 0.3], [-0.2, 0.8]]))
COST = QuadraticCost(
    Q=jnp.diag(jnp.array([1.0, 0.5])),
    R=jnp.diag(jnp.array([0.2, 0.1])),
    Qf=jnp.diag(jnp.array([2.0, 1.0])),
    x_target=jnp.array([1.5, -1.0]),
)
START = jnp.array([0.2, 0.4])
DT, HORIZON, BOX = 0.2, 5, 1.5
BUDGET = LinearConstraint(np.tile([1.0, 0.0], HORIZON)[None, :], -np.inf, 4.0)
SIGMA = np.array([0.3, 0.2])  # the dither's scale per lever
# the plant's one-step channel less the model's: about a third of each entry of the channel
LARGE_MOVE = np.array([[0.09, -0.045], [0.06, 0.045]])
WEIGHT = np.array(
    [[2.0, 0.3, 0.0, 0.1], [0.3, 1.0, -0.2, 0.0], [0.0, -0.2, 1.5, 0.4], [0.1, 0.0, 0.4, 0.8]]
)


def _weight(matrix: np.ndarray, shape: tuple[int, int]) -> DecisionWeight:
    return DecisionWeight(matrix, shape, free=matrix.shape[0], weakly_active=0, residual=0.0)


def _p_value(weight: np.ndarray, move: ChannelMove) -> float:
    d = move.estimate.ravel()
    return _chi_square_mixture_survival(float(d @ weight @ d), *_null_law(weight, move.covariance))


# ---- the null law ----


@pytest.mark.parametrize(
    ("weight", "shift", "q"), [(2.0, 1.0, 3.0), (1.0, 9.0, 30.0), (0.5, 0.3, 0.01)]
)
def test_one_noncentral_term_is_scipys_noncentral_chi_square(
    weight: float, shift: float, q: float
) -> None:
    survival = _chi_square_mixture_survival(q, np.array([weight]), np.array([shift]))
    assert survival == pytest.approx(stats.ncx2.sf(q / weight, 1, shift), abs=1e-10)


def test_two_noncentral_terms_read_the_conditional_integral_and_maximas_tails() -> None:
    """Eigenvalues 1 and 0.3 with ``tr(W S) = 1.3`` split between them, at the critical value the
    test reads at 0.05: ``P(Q > c) = int phi(y) P(0.3 (z + m2)^2 > c - (y + m1)^2) dy``, and the
    tails ``validation/keep_or_replan.mac`` STEP 4 integrates. Unshifted, the inversion is the
    central one, to the bit."""
    weights = np.array([1.0, 0.3])
    critical = 8.096474267762279
    maxima = {0.0: 0.050000000010478117, 0.25: 0.04279754766625622, 0.5: 0.03500909739609143}
    maxima[1.0] = 0.01724789748780306
    for split, tail in maxima.items():
        shifts = np.array([(1.0 - split) * 1.3, split * 1.3 / 0.3])
        m1, m2 = np.sqrt(shifts)

        def given(y: float, m1: float = m1, m2: float = m2) -> float:
            rest = critical - (y + m1) ** 2
            if rest <= 0.0:
                return stats.norm.pdf(y)
            r = math.sqrt(rest / 0.3)
            return stats.norm.pdf(y) * (stats.norm.cdf(m2 - r) + stats.norm.cdf(-m2 - r))

        edge = math.sqrt(critical)
        conditional = sum(
            integrate.quad(given, low, high, epsabs=1e-13, epsrel=1e-12, limit=400)[0]
            for low, high in (
                (-12.0 - m1, -m1 - edge),
                (-m1 - edge, -m1 + edge),
                (-m1 + edge, 12.0),
            )
        )
        survival = _chi_square_mixture_survival(critical, weights, shifts)
        assert survival == pytest.approx(conditional, abs=1e-9)
        assert survival == pytest.approx(tail, abs=1e-9)
    assert _chi_square_mixture_survival(2.0, weights) == _chi_square_mixture_survival(
        2.0, weights, np.zeros(2)
    )


def test_one_directions_margin_and_p_value_are_maximas_closed_forms() -> None:
    """On one direction the statistic is ``W s (z + mu)^2`` and the least favourable move is
    ``mu = 1``: the critical ``t^2`` solves ``Phi(1 - t) + Phi(-1 - t) = level``
    (``validation/keep_or_replan.mac`` STEP 3), and the margin is ``W s (t^2 - 2) / 2`` in the
    plan's cost, whatever the units of ``W`` and ``s``."""
    for w, s in ((1.0, 1.0), (3.0, 0.2)):
        weight = _weight(np.array([[w]]), (1, 1))
        move = ChannelMove(np.array([[1.5 * math.sqrt(s)]]), np.array([[s]]), 10.0)
        price = move.price(weight)
        for level, t_squared in ((0.05, 7.0020862623397095), (0.01, 11.066480529979527)):
            margin, p_value = _replan_test(weight, move, price, level)
            assert margin == pytest.approx(w * s * (t_squared - 2.0) / 2.0, rel=1e-9, abs=0.0)
            assert p_value == pytest.approx(0.31474720405176303, abs=1e-10)


def test_all_of_the_noncentrality_on_the_largest_eigenvalue_is_least_favourable() -> None:
    """At the critical value the p-value is read with, no other split of ``tr(W S)`` between the
    eigenvalues reads a heavier tail, at the levels the test takes."""
    for spectrum in ([1.0, 0.3, 0.05], [1.0, 0.9, 0.9], [1.0, 0.02]):
        weights = np.array(spectrum)
        total = float(weights.sum())
        least = np.zeros_like(weights)
        least[0] = total
        for level in (0.01, 0.2):
            critical = brentq(
                lambda q, w=weights, s=least, lv=level: _chi_square_mixture_survival(q, w, s) - lv,
                0.0,
                50.0 * total,
                xtol=1e-12,
            )
            for share in ([0.5, 0.5, 0.0][: weights.size], [0.0] * (weights.size - 1) + [1.0]):
                shifts = np.array(share) * total / weights
                tail = _chi_square_mixture_survival(critical, weights, shifts)
                assert tail <= level + 1e-7


def test_the_test_holds_its_level_at_the_least_favourable_move() -> None:
    """Over 40 000 Gaussian estimates ``dh ~ N(d, S)``: with ``d`` the null's least favourable
    move, ``d' W d = tr(W S)`` all along ``W S``'s largest eigenvalue, the test re-plans on 0.05 of
    them; with the same ``d' W d`` spread over every direction, and with no move, on fewer."""
    covariance = 0.01 * np.array(
        [[1.0, 0.2, 0.0, 0.1], [0.2, 0.5, 0.1, 0.0], [0.0, 0.1, 0.8, -0.2], [0.1, 0.0, -0.2, 0.6]]
    )
    weight = _weight(WEIGHT, (2, 2))
    move = ChannelMove(np.zeros((2, 2)), covariance, 100.0)
    price = move.price(weight)
    margin, _ = _replan_test(weight, move, price, 0.05)
    values, vectors = np.linalg.eigh(covariance)
    root = (vectors * np.sqrt(values)) @ vectors.T
    spectrum, axes = np.linalg.eigh(root @ WEIGHT @ root)
    total = float(spectrum.sum())
    least = root @ axes[:, -1] * math.sqrt(total / spectrum[-1])
    spread = root @ axes @ np.sqrt(np.full(4, total / 4.0) / spectrum)
    assert least @ WEIGHT @ least == pytest.approx(total, rel=1e-12, abs=0.0)
    assert spread @ WEIGHT @ spread == pytest.approx(total, rel=1e-12, abs=0.0)
    rng = np.random.default_rng(41)
    draws = rng.standard_normal((40_000, 4)) @ root
    rates = {}
    for name, d in (("least", least), ("spread", spread), ("none", np.zeros(4))):
        dh = d + draws
        difference = (np.einsum("pi,ij,pj->p", dh, WEIGHT, dh) - 2.0 * total) / 2.0
        rates[name] = float(np.mean(difference > margin))
    # 3.5 standard errors of a rate of 0.05 over 40 000 draws
    assert abs(rates["least"] - 0.05) < 3.5 * math.sqrt(0.05 * 0.95 / 40_000)
    assert rates["spread"] < rates["least"]
    assert rates["none"] < 0.01


# ---- the budgeted plan, its logs and its re-solve ----


@pytest.fixture(scope="module")
def budgeted() -> tuple[CausalPlan, np.ndarray, np.ndarray, np.ndarray]:
    """The plan, its decision weight, and the RK4 step's two matrices the plant is built on."""
    plan = causal_plan(
        MODEL, START, COST, DT, HORIZON, -BOX, BOX, constraints=(BUDGET,), steps=20_000
    )
    phi, gamma = (
        np.asarray(m, dtype=np.float64)
        for m in jax.jacfwd(lambda x, u: rk4_step(MODEL, 0.0, x, u, DT), argnums=(0, 1))(
            START, jnp.zeros(2)
        )
    )
    return plan, np.asarray(plan.decision_weight().matrix), phi, gamma


def _read(
    plan: CausalPlan,
    phi: np.ndarray,
    gamma: np.ndarray,
    runs: int,
    move: np.ndarray,
    rng: np.random.Generator,
) -> ChannelMove:
    """``runs`` runs of the plan with a dither, on a plant whose one-step map is the model's plus
    ``move u``, a drift the model does not have and noise; the move read off them."""
    drawn = SIGMA * rng.standard_normal((runs, HORIZON, 2))
    action = np.asarray(plan.actions, dtype=np.float64) + drawn
    x = np.broadcast_to(np.asarray(START, dtype=np.float64), (runs, 2))
    residual = np.empty((runs, HORIZON, 2))
    for t in range(HORIZON):
        model_step = x @ phi.T + action[:, t] @ gamma.T
        after = (
            model_step
            + action[:, t] @ move.T
            + 0.05 * np.tanh(x)
            + 0.05 * rng.standard_normal((runs, 2))
        )
        residual[:, t] = after - model_step
        x = after
    size = runs * HORIZON
    log = DecisionLog(
        action.reshape(-1, 2), np.ones(size), np.zeros(size, dtype=bool), drawn.reshape(-1, 2)
    )
    return channel_move(log, residual.reshape(-1, 2), dither_scale=SIGMA)


def test_an_unmoved_channel_is_kept_at_the_nominal_rate(
    budgeted: tuple[CausalPlan, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Over 100 logs of 400 decisions of a channel that did not move, the test at 0.05 re-planned
    on 12 at most, the count a rate of 0.05 exceeds with probability below 0.001; and the method
    keeps the plan on one of them."""
    plan, weight, phi, gamma = budgeted
    rng = np.random.default_rng(42)
    moves = [_read(plan, phi, gamma, 80, np.zeros((2, 2)), rng) for _ in range(100)]
    replanned = sum(_p_value(weight, move) <= 0.05 for move in moves)
    assert replanned <= 12
    assert moves[0].keep_or_replan(plan, steps=20_000).decision == "keep"


def test_a_large_move_is_replanned_with_high_power(
    budgeted: tuple[CausalPlan, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """A move of about a third of the channel's entries, read over 1600 decisions: the test at
    0.05 re-plans on 38 of 40 logs at least. It re-planned on all 40, and over 400 decisions on 18:
    the move's own part in each product's variance grows with it."""
    plan, weight, phi, gamma = budgeted
    rng = np.random.default_rng(43)
    moves = [_read(plan, phi, gamma, 320, LARGE_MOVE, rng) for _ in range(40)]
    assert sum(_p_value(weight, move) <= 0.05 for move in moves) >= 38


def test_the_replanned_plan_is_the_moved_costs_minimiser_and_beats_the_kept_one_on_the_plant(
    budgeted: tuple[CausalPlan, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Read over 1600 decisions, the large move is re-planned on. The re-solved plan is SLSQP's
    minimiser of the cost the decision weight perturbs, at the estimate, under the plan's box and
    budget; and on the plant it costs less than the plan it replaces, within a tenth of that gain of
    the plan that knew the move."""
    plan, _, phi, gamma = budgeted
    move = _read(plan, phi, gamma, 320, LARGE_MOVE, np.random.default_rng(44))
    chosen = move.keep_or_replan(plan, steps=20_000)
    assert chosen.decision == "replan"
    assert chosen.p_value <= 0.05
    assert chosen.difference > chosen.margin
    assert chosen.difference == pytest.approx(
        chosen.price.keep - chosen.price.replan, rel=1e-12, abs=0.0
    )
    replanned = chosen.replanned
    assert replanned.solver_status == "converged"
    assert replanned.certificate_status == "not_evaluated"
    problem = plan._problem
    assert problem is not None
    value = jax.jit(lambda u, e: _perturbed_task_cost(problem, u.reshape(HORIZON, 2), e))
    gradient = jax.jit(jax.grad(value))

    def solve(change: np.ndarray) -> np.ndarray:
        e = jnp.asarray(change)
        result = minimize(
            lambda u: float(value(jnp.asarray(u), e)),
            np.asarray(plan.actions, dtype=np.float64).ravel(),
            jac=lambda u: np.asarray(gradient(jnp.asarray(u), e)),
            bounds=[(-BOX, BOX)] * (2 * HORIZON),
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda u: BUDGET.upper - BUDGET.matrix @ u,
                    "jac": lambda u: -BUDGET.matrix,
                }
            ],
            method="SLSQP",
            options={"ftol": 1e-15, "maxiter": 2000},
        )
        assert result.success, result.message
        return result.x.reshape(HORIZON, 2)

    assert np.asarray(replanned.actions) == pytest.approx(solve(move.estimate), abs=1e-5)
    assert replanned.task_cost == pytest.approx(
        float(value(replanned.actions, jnp.asarray(move.estimate))), rel=1e-9, abs=0.0
    )
    truth = jnp.asarray(LARGE_MOVE)
    kept = float(value(plan.actions, truth))
    resolved = float(value(replanned.actions, truth))
    best = float(value(jnp.asarray(solve(LARGE_MOVE)), truth))
    assert resolved < kept
    assert resolved - best < 0.1 * (kept - best)


def test_the_moved_model_steps_as_the_decision_weight_perturbs_it() -> None:
    """A step of the moved model is the model's step plus ``E u``, under RK4 and under Euler's map,
    and a rollout of it costs what the decision weight's perturbed cost reads."""
    change = jnp.array([[0.05, -0.02], [0.01, 0.03]])
    x, u = jnp.array([0.3, -0.7]), jnp.array([0.4, 1.1])
    for model in (MODEL, _EulerStep(MODEL)):
        moved = _Moved(_MovedRate(model, change, DT))
        step = rk4_step(moved, 0.0, x, u, DT)
        assert np.asarray(step) == pytest.approx(
            np.asarray(rk4_step(model, 0.0, x, u, DT) + change @ u), rel=1e-13, abs=1e-15
        )
    plan = causal_plan(MODEL, START, COST, DT, HORIZON, -BOX, BOX)
    problem = plan._problem
    assert problem is not None
    moved = _Moved(_MovedRate(MODEL, change, DT))
    assert float(total_cost(moved, START, plan.actions, DT, COST)) == pytest.approx(
        float(_perturbed_task_cost(problem, plan.actions, change)), rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("level", [0.0, 0.25, 1.0, math.nan])
def test_a_level_outside_the_one_the_test_holds_is_refused(level: float) -> None:
    move = ChannelMove(np.zeros((2, 2)), np.eye(4), 10.0)
    plan = CausalPlan(
        jnp.zeros((HORIZON, 2)), jnp.zeros((HORIZON + 1, 2)), 0.0, None, None, "converged", 0
    )
    with pytest.raises(ValueError, match=r"false re-plan rate in \(0, 0.2\]"):
        move.keep_or_replan(plan, level=level)


def test_a_plan_with_no_problem_or_a_move_of_another_shape_is_refused(
    budgeted: tuple[CausalPlan, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    plan = budgeted[0]
    move = ChannelMove(np.zeros((2, 2)), np.eye(4), 10.0)
    with pytest.raises(ValueError, match="carries no problem to weigh"):
        move.keep_or_replan(dataclasses.replace(plan, _problem=None))
    other = ChannelMove(np.zeros((1, 2)), np.eye(2), 10.0)
    with pytest.raises(ValueError, match=r"the move is of shape \(1, 2\)"):
        other.keep_or_replan(plan)


def test_with_no_noise_along_the_weight_the_test_replans_where_the_move_is_positive() -> None:
    """Where ``W S`` is nothing, ``dh' W dh`` is ``d' W d`` exactly: the test re-plans wherever it
    is positive, with a margin of 0."""
    weight = _weight(np.diag([1.0, 0.0]), (1, 2))
    still = ChannelMove(np.array([[0.0, 0.3]]), np.diag([0.0, 1.0]), 10.0)
    assert _replan_test(weight, still, still.price(weight), 0.05) == (0.0, 1.0)
    moved = ChannelMove(np.array([[0.2, 0.3]]), np.diag([0.0, 1.0]), 10.0)
    assert _replan_test(weight, moved, moved.price(weight), 0.05) == (0.0, 0.0)
    assert moved.price(weight).replan == 0.0
