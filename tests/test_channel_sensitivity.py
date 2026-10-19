"""An absolute radius on the fitted channel under a marginal sensitivity model.

The radius the barrier audit read was a share of the fitted channel, so it stayed within the
estimate's own size of the estimate at every level: on a log whose levers follow a hidden
confounder, an estimate of 0.2 for an effect of 1 certified plans the effect itself takes out of
the set. :class:`chc.plan.ChannelSensitivity` bounds the least-squares channel's bias under the
levers' density moving by a factor within ``[1/Gamma, Gamma]`` given the confounder, beside its
sampling error (``validation/channel_bias_bound.mac``).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from chc.cost import QuadraticCost
from chc.dynamics import HybridDynamics, LinearDynamics
from chc.dynamics_id import CausalDynamicsFit, fit_causal_residual
from chc.integrate import rollout
from chc.plan import BarrierConstraint, ChannelSensitivity, causal_plan, certify_safety

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
    """``kappa(Gamma) = (Gamma-1)/sqrt(3 Gamma - Gamma^2 - 1)``: 0 at 1, 1 at 2, unbounded at
    ``(3 + sqrt 5)/2``, and ``level_at`` its inverse (``channel_bias_bound.mac`` STEP 3)."""
    limit = (3.0 + math.sqrt(5.0)) / 2.0
    assert ChannelSensitivity.factor(1.0) == 0.0
    assert ChannelSensitivity.factor(2.0) == pytest.approx(1.0, rel=1e-15, abs=0.0)
    assert ChannelSensitivity.factor(limit) == math.inf
    assert ChannelSensitivity.factor(3.0) == math.inf
    for gamma in (1.0, 1.01, 1.5, 2.0, 2.5, 2.6):
        back = ChannelSensitivity.level_at(ChannelSensitivity.factor(gamma))
        assert back == pytest.approx(gamma, rel=1e-12, abs=0.0)
    assert ChannelSensitivity.level_at(math.inf) == limit
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
    absolute = certify_safety(
        plan, model, _ceiling, _DT, gamma=1.5, u_max=5.0, sensitivity=sensitivity
    )
    assert absolute.certified_steps == 0
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
    assert all(1.0 < g < (3.0 + math.sqrt(5.0)) / 2.0 for g in finite)
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


def test_a_level_past_the_limit_bounds_nothing_and_is_refused() -> None:
    fit = _confounded_fit(0, 1.5, -3.0)
    sensitivity = ChannelSensitivity.from_fit(fit)
    model = HybridDynamics(known=_KNOWN, residual=fit.residual)
    plan = causal_plan(model, jnp.zeros(1), _COST, _DT, 5, -5.0, 5.0)
    with pytest.raises(ValueError, match="at or past"):
        BarrierConstraint(_ceiling, gamma=2.7, sensitivity=sensitivity)
    with pytest.raises(ValueError, match="at or past"):
        certify_safety(plan, model, _ceiling, _DT, gamma=2.7, sensitivity=sensitivity)


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
