"""An absolute radius on the fitted channel under a marginal sensitivity model.

The radius the barrier audit read was a share of the fitted channel, so it stayed within the
estimate's own size of the estimate at every level: on a log whose levers follow a hidden
confounder, an estimate of 0.2 for an effect of 1 certified plans the effect itself takes out of
the set. :class:`chc.plan.ChannelSensitivity` bounds the least-squares channel's bias under the
levers' density moving by a factor within ``[1/Gamma, Gamma]`` given the confounder, beside its
sampling error (``validation/channel_bias_bound.mac``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from scipy.optimize import minimize

from chc.cost import QuadraticCost, total_cost
from chc.dynamics import HybridDynamics, LinearDynamics
from chc.dynamics_id import CausalDynamicsFit, fit_causal_residual
from chc.integrate import rollout
from chc.plan import (
    _ROUND_STEPS,
    BarrierConstraint,
    CausalPlan,
    ChannelSensitivity,
    InnerSolve,
    OuterRule,
    _action_shape,
    _ChannelWidth,
    _diagnostics,
    _HeldBarrier,
    _reach,
    causal_plan,
    certify_safety,
)

_DT, _ROWS, _NOISE, _TRUE_CHANNEL = 0.1, 2000, 0.3, 1.0
_KNOWN = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
_COST = QuadraticCost(Q=jnp.eye(1), R=1e-4 * jnp.eye(1), Qf=jnp.eye(1), x_target=jnp.array([2.0]))


def _ceiling(x: Array) -> Array:
    return 1.0 - x[0]


def _confounded_fit(
    seed: int, gamma: float, push: float, drift: float = -1.0, degree: int = 0
) -> CausalDynamicsFit:
    """``xdot = drift x + u + push Z``, ``Z = +-1`` hidden. The logged lever's sign follows ``Z``
    with probability ``(1 + t)/2``, ``t = 1 - 1/gamma``, so its density given ``Z`` is its own
    times ``1 +- t``, within ``[1/gamma, gamma]``. A column of noise is the asserted adjustment."""
    rng = np.random.default_rng(seed)
    tilt = 1.0 - 1.0 / gamma
    z = rng.choice([-1.0, 1.0], _ROWS)
    u = np.where(rng.random(_ROWS) < (1.0 + tilt) / 2.0, z, -z) * np.abs(rng.normal(size=_ROWS))
    x = rng.normal(size=_ROWS)
    rate = drift * x + _TRUE_CHANNEL * u + push * z
    x_next = x + _DT * rate + math.sqrt(_DT) * _NOISE * rng.normal(size=_ROWS)
    data = {
        "x": jnp.asarray(x[:, None]),
        "u": jnp.asarray(u[:, None]),
        "x_next": jnp.asarray(x_next[:, None]),
        "w": jnp.asarray(rng.normal(size=(_ROWS, 1))),
    }
    return fit_causal_residual(_KNOWN, data, _DT, adjust_for=("w",), channel_degree=degree)


def _fitted_channel(model: HybridDynamics, x: Array) -> float:
    return float(jax.jacobian(model, argnums=2)(0.0, x, jnp.zeros(1))[0, 0])


def test_the_factor_and_its_inverse() -> None:
    """``kappa(Gamma) = (Gamma-1)/sqrt(2 Gamma - 1)``: 0 at 1, ``1/sqrt 3`` at 2, 4/3 at 5, finite
    at every finite level, and ``level_at`` its inverse, ``1 + k^2 + k sqrt(1 + k^2)``, whose
    ``Gamma`` at ``kappa = 1`` is ``2 + sqrt 2`` (``channel_bias_bound.mac`` STEP 3). Up to 0.21
    the factor was ``(Gamma-1)/sqrt(3 Gamma - Gamma^2 - 1)``: 1 at 2, and infinite from
    ``(3 + sqrt 5)/2`` on."""
    assert ChannelSensitivity.factor(1.0) == 0.0
    assert ChannelSensitivity.factor(2.0) == pytest.approx(1.0 / math.sqrt(3.0), rel=1e-15, abs=0.0)
    assert ChannelSensitivity.factor(5.0) == pytest.approx(4.0 / 3.0, rel=1e-15, abs=0.0)
    assert ChannelSensitivity.level_at(1.0) == pytest.approx(
        2.0 + math.sqrt(2.0), rel=1e-15, abs=0.0
    )
    old = (3.0 + math.sqrt(5.0)) / 2.0
    for gamma in (1.0, 1.01, 1.5, 2.0, 2.5, old, 3.0, 10.0, 1e6, 1e300):
        kappa = ChannelSensitivity.factor(gamma)
        assert math.isfinite(kappa)
        back = ChannelSensitivity.level_at(kappa)
        assert back == pytest.approx(gamma, rel=1e-12, abs=0.0)
    assert ChannelSensitivity.factor(math.inf) == math.inf
    assert ChannelSensitivity.level_at(math.inf) == math.inf
    with pytest.raises(ValueError, match="sensitivity level"):
        ChannelSensitivity.factor(0.5)


def test_the_truth_lies_inside_the_radius_at_the_level_the_log_was_confounded_at() -> None:
    """Over 20 logs confounded at ``Gamma = 1.5`` the channel's estimate is 0.21 for an effect of
    1, and the radius at 1.5 holds the effect on every one; at 0.95 the Monte Carlo SD over 20
    logs is 0.049, so 18 of 20 is two SDs below. At ``Gamma = 1`` the sampling error alone holds
    it on none: the bias is the confounder's."""
    inside, sampling_only = 0, 0
    for seed in range(20):
        fit = _confounded_fit(seed, 1.5, -3.0)
        sensitivity = ChannelSensitivity.from_fit(fit)
        model = HybridDynamics(known=_KNOWN, residual=fit.residual)
        error = abs(_fitted_channel(model, jnp.zeros(1)) - _TRUE_CHANNEL)
        inside += error <= sensitivity.radius(jnp.zeros(1), 1.5)
        sampling_only += error <= sensitivity.radius(jnp.zeros(1), 1.0)
    assert inside >= 18
    assert sampling_only == 0


def test_a_share_of_the_estimate_certified_what_the_effect_took_out_of_the_set() -> None:
    """The plan believes a channel of 0.15 and pushes at full force for a state of 2 past a ceiling
    of 1. Read as a share of that estimate, the audit at ``Gamma = 1.5`` certified all 30 steps,
    and the true plant left the set at the third. The absolute radius holds the effect and
    certifies none, and a plan held under it keeps the true plant inside at every step."""
    fit = _confounded_fit(1000, 1.5, -3.0)
    model = HybridDynamics(known=_KNOWN, residual=fit.residual)
    sensitivity = ChannelSensitivity.from_fit(fit)
    truth = LinearDynamics(jnp.array([[-1.0]]), jnp.array([[_TRUE_CHANNEL]]))
    x0 = jnp.zeros(1)
    plan = causal_plan(model, x0, _COST, _DT, 30, -5.0, 5.0)
    true_h = 1.0 - np.asarray(rollout(truth, x0, plan.actions, _DT)[:, 0])
    assert _fitted_channel(model, x0) < 0.3
    assert float(np.min(true_h)) < 0.0

    share = certify_safety(plan, model, _ceiling, _DT, gamma=1.5, u_max=5.0)
    assert share.certified_steps == 30  # what the share of the estimate read
    assert share.radius_model == "relative"  # and says it read a margin, not a bound
    absolute = certify_safety(
        plan, model, _ceiling, _DT, gamma=1.5, u_max=5.0, sensitivity=sensitivity
    )
    assert absolute.certified_steps == 0
    assert absolute.radius_model == "absolute"
    assert absolute.radius >= abs(_fitted_channel(model, x0) - _TRUE_CHANNEL)
    assert absolute.gamma_star == math.inf  # standing still is safe for every channel

    held = causal_plan(
        model,
        x0,
        _COST,
        _DT,
        30,
        -5.0,
        5.0,
        barrier=BarrierConstraint(_ceiling, gamma=1.5, sensitivity=sensitivity),
    )
    assert held.safety is not None
    assert held.safety.certified_steps == 30
    assert held.safety.radius_model == "absolute"
    held_h = 1.0 - np.asarray(rollout(truth, x0, held.actions, _DT)[:, 0])
    assert float(np.min(held_h)) > 0.0


def test_gamma_star_is_the_level_whose_radius_the_step_just_absorbs() -> None:
    """With an unstable drift the plan's steps need the channel, and ``gamma_star`` is finite: the
    audit at a level just below it certifies the step's best action and just above it not, read
    through the closed form of the threshold (:mod:`chc.barrier`) at the step's own radius."""
    fit = _confounded_fit(7, 1.2, -1.0, drift=0.5)
    model = HybridDynamics(known=_KNOWN, residual=fit.residual)
    sensitivity = ChannelSensitivity.from_fit(fit)
    x0 = jnp.array([0.5])
    plan = causal_plan(model, x0, _COST, _DT, 10, -5.0, 5.0)
    audit = certify_safety(plan, model, _ceiling, _DT, u_max=5.0, sensitivity=sensitivity)
    finite = [g for g in audit.step_gamma_star if math.isfinite(g)]
    assert finite
    assert all(g > 1.0 for g in finite)
    step = audit.step_gamma_star.index(min(finite))
    state = plan.trajectory[step]
    channel = abs(_fitted_channel(model, state))
    drift = float(model(0.0, state, jnp.zeros(1))[0])
    deficit = -1.0 * float(_ceiling(state)) + drift  # -alpha h - grad h . f, grad h = -1
    for level, holds in (
        (0.999 * audit.step_gamma_star[step], True),
        (1.001 * audit.step_gamma_star[step], False),
    ):
        radius = sensitivity.radius(state, level)
        assert (5.0 * (channel - radius) >= deficit) == holds


def test_the_radius_is_read_at_each_step_s_state() -> None:
    """A channel that reads the state is known best where the log moved the levers: the radius
    widens as the plan's states leave the log's, and the audit's radius is the widest step's."""
    fit = _confounded_fit(3, 1.2, -1.0, degree=1)
    model = HybridDynamics(known=_KNOWN, residual=fit.residual)
    sensitivity = ChannelSensitivity.from_fit(fit)
    near, far = sensitivity.radius(jnp.zeros(1), 1.2), sensitivity.radius(jnp.array([3.0]), 1.2)
    assert far > 2.0 * near
    plan = causal_plan(model, jnp.zeros(1), _COST, _DT, 20, -5.0, 5.0)
    audit = certify_safety(
        plan, model, _ceiling, _DT, gamma=1.2, u_max=5.0, sensitivity=sensitivity
    )
    widest = max(sensitivity.radius(x, 1.2) for x in plan.trajectory[:-1])
    assert audit.radius == pytest.approx(widest, rel=1e-9, abs=0.0)  # ||grad h|| = 1
    assert widest > near


def test_every_finite_level_has_a_radius_and_an_infinite_one_is_refused() -> None:
    """At no finite level are the levers a function of the confounder: a share ``1/Gamma`` of their
    law does not read it. So a level past ``(3 + sqrt 5)/2``, which up to 0.21 was refused as
    unbounded, reads a finite radius, wider at each level; only ``inf`` is refused."""
    fit = _confounded_fit(0, 1.5, -3.0)
    sensitivity = ChannelSensitivity.from_fit(fit)
    model = HybridDynamics(known=_KNOWN, residual=fit.residual)
    plan = causal_plan(model, jnp.zeros(1), _COST, _DT, 5, -5.0, 5.0)
    widths = [sensitivity.radius(jnp.zeros(1), gamma) for gamma in (2.5, 2.7, 10.0, 1e6)]
    assert all(math.isfinite(width) for width in widths)
    assert widths == sorted(widths)
    BarrierConstraint(_ceiling, gamma=2.7, sensitivity=sensitivity)
    audit = certify_safety(plan, model, _ceiling, _DT, gamma=2.7, sensitivity=sensitivity)
    assert audit.radius == pytest.approx(widths[1], rel=1e-9, abs=0.0)  # ||grad h|| = 1
    with pytest.raises(ValueError, match="no level"):
        certify_safety(plan, model, _ceiling, _DT, gamma=math.inf, sensitivity=sensitivity)
    with pytest.raises(ValueError, match="finite"):
        BarrierConstraint(_ceiling, gamma=math.inf, sensitivity=sensitivity)


def test_a_fit_without_a_least_squares_channel_has_no_sensitivity() -> None:
    """An ``rk4`` fit's fixed point carries the drift's confounding into the channel, which no
    bound on the levers limits."""
    rng = np.random.default_rng(0)
    x, u = rng.normal(size=(400, 1)), rng.normal(size=(400, 1))
    data = {
        "x": jnp.asarray(x),
        "u": jnp.asarray(u),
        "x_next": jnp.asarray(x + _DT * (u - x)),
        "w": jnp.asarray(rng.normal(size=(400, 1))),
    }
    fit = fit_causal_residual(_KNOWN, data, _DT, adjust_for=("w",), integrator="rk4")
    with pytest.raises(ValueError, match="rk4"):
        ChannelSensitivity.from_fit(fit)


def _held_on_an_unstable_drift(
    seed: int, steps: int, outer: OuterRule, inner: InnerSolve = "diagonal"
) -> CausalPlan:
    """The plan held to ``x <= 1`` under the fit's own sensitivity at ``Gamma = 1.5``, on a log of
    an unstable drift, 0.5, whose lever follows the hidden confounder at that level."""
    fit = _confounded_fit(seed, 1.5, -3.0, drift=0.5)
    model = HybridDynamics(known=_KNOWN, residual=fit.residual)
    barrier = BarrierConstraint(
        _ceiling, gamma=1.5, u_max=(5.0,), sensitivity=ChannelSensitivity.from_fit(fit)
    )
    return causal_plan(
        model,
        jnp.zeros(1),
        _COST,
        _DT,
        30,
        -5.0,
        5.0,
        steps=steps,
        barrier=barrier,
        outer=outer,
        inner=inner,
    )


def test_safeguarded_rounds_grow_the_penalty_where_every_round_runs_out_of_its_share(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """On this log every halved round runs out of its share, and grows no penalty: it stayed at
    its start over all 15 rounds while the shortfall stopped falling. The safeguarded rounds grow
    it after every round whose measure fails to halve, its share run out or not, here first after
    the second round, which ran out of its share. It then reaches its ceiling, and a round cut
    there ends the rounds ``max_iterations``: read ``infeasible``, as after a solved round, it
    called a condition SLSQP meets at every step infeasible."""
    with caplog.at_level(logging.DEBUG, logger="chc.plan"):
        plan = _held_on_an_unstable_drift(2, 10_000, "safeguarded")
    rounds = [
        (float(record.penalty), int(record.descent_steps))  # type: ignore[attr-defined]
        for record in caplog.records
        if getattr(record, "chc_event", None) == "barrier_round"
    ]
    grown = [k for k in range(len(rounds) - 1) if rounds[k + 1][0] > rounds[k][0]]
    assert grown
    assert rounds[grown[0]][1] == _ROUND_STEPS
    assert plan.solver_status == "max_iterations"
    assert plan.safety is not None
    assert plan.safety.certified_steps > 0


def _safeguarded_rounds(
    seed: int, steps: int, caplog: pytest.LogCaptureFixture
) -> list[tuple[int, float, float, int]]:
    """Each safeguarded round's number, penalty, measure and descent steps, as the plan logs."""
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="chc.plan"):
        _held_on_an_unstable_drift(seed, steps, "safeguarded")
    return [
        (r.round, r.penalty, r.measure, r.descent_steps)  # type: ignore[attr-defined]
        for r in caplog.records
        if getattr(r, "chc_event", None) == "barrier_round"
    ]


@pytest.mark.parametrize("seed", [1, 2])
def test_a_larger_budget_runs_the_same_safeguarded_rounds_further(
    seed: int, caplog: pytest.LogCaptureFixture
) -> None:
    """A safeguarded round's share is the same whatever the budget, so ten times the budget runs
    the same rounds further: every round of the 10 000-step plan but its last, which the budget
    cut, is a round of the 100 000-step plan, and the larger budget stops no sooner. Here it cost
    65.95086 against 65.95203 on log 1, both certified on 30 steps of 30, and log 2's rounds
    stopped at the penalty's ceiling at either budget; the halved shares move with the budget,
    and on log 2 the plan at 100 000 steps was certified on 27 steps of 30, at 10 000 on all 30.
    Those costs are of plans the rounds did not converge on, so only the rounds are compared."""
    small = _safeguarded_rounds(seed, 10_000, caplog)
    large = _safeguarded_rounds(seed, 100_000, caplog)
    shared = len(small) - 1
    assert shared > 0
    assert len(large) >= len(small)
    for one, other in zip(small[:shared], large[:shared], strict=True):
        assert one[0] == other[0]
        assert one[3] == other[3]
        assert one[1:3] == pytest.approx(other[1:3], rel=1e-9, abs=0.0)


def _slsqp_on_the_held_columns(seed: int) -> float:
    """SLSQP's cost, from zeros, on the columns the audit certifies a step by on the log of
    :func:`_held_on_an_unstable_drift`: each step's condition and the room at its next state, every
    next state held, no back-off, the rounds' factors at one."""
    fit = _confounded_fit(seed, 1.5, -3.0, drift=0.5)
    model = HybridDynamics(known=_KNOWN, residual=fit.residual)
    shape, inverse, _ = _action_shape(_reach((5.0,), 1))
    assert shape is not None
    assert inverse is not None
    held = _HeldBarrier(
        model=model,
        barrier=_ceiling,
        dt=_DT,
        alpha=1.0,
        delta=jnp.asarray(0.0),
        inverse=jnp.asarray(inverse),
        width=_ChannelWidth.build(ChannelSensitivity.from_fit(fit), 1, shape, 1.5, jnp.float64),
        multipliers=jnp.zeros((30, 2)),
        weight=jnp.asarray(1.0),
        backoff=jnp.asarray(0.0),
        curvature=jnp.zeros(1),
    )

    def task(u: Array) -> Array:
        return total_cost(model, jnp.zeros(1), u.reshape(30, 1), _DT, _COST)

    def margins(u: Array) -> Array:
        us = u.reshape(30, 1)
        return -held.shortfall(rollout(model, jnp.zeros(1), us, _DT)[:-1], us).ravel()

    value, slope = jax.jit(task), jax.jit(jax.grad(task))
    margin, margin_slope = jax.jit(margins), jax.jit(jax.jacobian(margins))
    found = minimize(
        lambda u: float(value(jnp.asarray(u))),
        np.zeros(30),
        jac=lambda u: np.asarray(slope(jnp.asarray(u))),
        method="SLSQP",
        bounds=[(-5.0, 5.0)] * 30,
        constraints=[
            {
                "type": "ineq",
                "fun": lambda u: np.asarray(margin(jnp.asarray(u))),
                "jac": lambda u: np.asarray(margin_slope(jnp.asarray(u))),
            }
        ],
        options={"ftol": 1e-14, "maxiter": 2000},
    )
    assert float(np.min(np.asarray(margin(jnp.asarray(found.x))))) > -1e-8
    return float(found.fun)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_orthant_rounds_rest_the_unused_actions_on_the_kink_at_slsqp_s_cost(seed: int) -> None:
    """On these logs SLSQP's plan spends at most three actions and rests the rest on zero, the
    kink of the tube's ``||D^+ u||``. The diagonal and Newton rounds never reach it, each action
    crossing and recrossing zero: they end ``max_iterations``, the diagonal ones 0.17 %, 1.5 % and
    0.32 % above SLSQP's cost, Newton's 0.69 %, 0.21 % and 0.52 %, every step certified. The
    orthant rounds stop an action there as on a side of the box, and converge 6e-8, 8e-8 and 4e-8
    above SLSQP's cost, which holds the columns with no back-off, every step certified. SLSQP
    stops on its iteration limit or a failed line search, which moves its cost by 1e-10 between
    starts; the fifth of a percent asked is far above both."""
    plan = _held_on_an_unstable_drift(seed, 10_000, "halved", "orthant")

    assert plan.solver_status == "converged"
    assert plan.stationarity_status == "within"
    assert plan.safety is not None
    assert plan.safety.certified_steps == 30
    assert int(np.sum(np.asarray(plan.actions) == 0.0)) >= 20
    assert plan.task_cost <= 1.002 * _slsqp_on_the_held_columns(seed)


@pytest.mark.parametrize("seed", [1, 2])
def test_a_larger_budget_leaves_the_orthant_plan_no_costlier(seed: int) -> None:
    """Under the halved rounds the plan at 100 000 steps was certified on 27 steps of 30 on log 2,
    where 10 000 steps certified all 30. The orthant rounds converge within either budget, so the
    larger one changes neither the plan's cost nor its certified steps."""
    small = _held_on_an_unstable_drift(seed, 10_000, "halved", "orthant")
    large = _held_on_an_unstable_drift(seed, 100_000, "halved", "orthant")

    assert small.solver_status == large.solver_status == "converged"
    assert large.task_cost <= small.task_cost
    assert large.safety is not None
    assert small.safety is not None
    assert large.safety.certified_steps >= small.safety.certified_steps


def test_the_orthant_plan_s_first_order_check_reads_the_kink_by_its_subdifferential() -> None:
    """The plan's first-order check reads the gradient of the held condition's norm, which is
    masked to zero on its kink, one element of its subdifferential: at the orthant rounds' plan,
    whose unused actions rest on zero, that reads about 0.7, some 1e5 times its tolerance, so the
    rounds never ended ``converged``. Read by the whole subdifferential, with the rounds' own
    multipliers, it is within the tolerance."""
    plan = _held_on_an_unstable_drift(1, 10_000, "halved", "orthant")
    problem = plan._problem
    assert problem is not None
    assert problem.kinked

    masked, tolerance, _, _ = _diagnostics(replace(problem, kinked=False), plan.actions)

    assert masked is not None
    assert masked > 1e3 * tolerance
    assert plan.stationarity is not None
    assert plan.stationarity <= tolerance
