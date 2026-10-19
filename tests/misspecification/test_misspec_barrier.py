"""The barrier's certified steps and its confounding ceiling on plants the fit gets wrong.

A plan held at its barrier at level ``gamma`` is certified on a step where the barrier condition
holds for every channel in the sensitivity set at that level, and where the model's path is in the
safe set at that step's state and the next. The contract read here: where the plant's channel is
inside the set the plan was held and audited at, the plant's own mean path, with demand, the
disturbance and the state noise at zero, stays in the safe set over every certified step; and
``gamma_star = inf``, which says the barrier holds at every level, holds for a plant whose
confounding reaches any level, or the certificate says its radius is a margin relative to the
estimate (``radius_model``), not a bound on the channel's error.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from _misspec_world import DT, World, ask
from jax import Array

from chc import Constraint
from chc.cost import QuadraticCost
from chc.dynamics import HybridDynamics, LinearDynamics
from chc.dynamics_id import CausalDynamicsFit, fit_causal_residual
from chc.integrate import rollout
from chc.plan import BarrierConstraint, ChannelSensitivity, causal_plan, certify_safety

pytestmark = pytest.mark.misspecification

CEILING, WAIT = 0.8, 0.5
TRUE_CHANNEL = np.array([0.8, -0.4])  # supply's and wait's, per unit of the incentive
LOGGED = 1.5  # the level of confounding the one-state logs are drawn at
UNKNOWN = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))


def _unit_ceiling(x: Array) -> Array:
    return 1.0 - x[0]


def _held(world: World, seed: int, gamma: float) -> tuple[np.ndarray, int, np.ndarray]:
    """The plan held below ``supply <= 0.8`` and ``wait <= 0.5`` at ``gamma``: the plant's mean path
    under it, its certified steps, and the fitted channel at the start."""
    result = ask(
        world,
        seed,
        constraints=(Constraint("supply", hi=CEILING), Constraint("wait", hi=WAIT)),
        hold_constraints=True,
        gamma=gamma,
    )
    assert result.plan is not None
    certified = result.certificate.barrier_certified_steps
    assert certified is not None
    planned = np.asarray(result.plan.trajectory, dtype=float)
    path = world.mean_path(np.asarray(result.plan.actions, dtype=float)[:, 0], planned[0])
    channel = np.asarray(result.model_fit.residual.channel, dtype=float)[:, 0, 0]
    return path, certified, channel


def _safe(path: np.ndarray, certified: int) -> bool:
    """Whether the path is in the safe set at every state a certified step reads."""
    states = path[: certified + 1]
    return bool(np.all(states[:, 0] <= CEILING) and np.all(states[:, 1] <= WAIT))


def test_certified_steps_keep_the_plant_safe_where_the_fit_is_right() -> None:
    """The control case at ``gamma = 1``: on four logs every step is certified, and the plant's
    path keeps 0.17 to 0.24 below the ceiling."""
    for seed in range(4):
        path, certified, _ = _held(World(), seed, 1.0)
        assert certified > 0
        assert _safe(path, certified), (seed, path[:, 0].max())


def test_certified_steps_keep_the_plant_safe_where_its_channel_is_in_the_set() -> None:
    """A confounder the panel does not record moves the channel the log reads from 0.8 to 0.42.
    At ``gamma = 20`` the set holds every channel within 0.905 of the fitted one's size, about
    0.52, and the plant's is 0.38 from it; the plan held there keeps the plant 0.21 or more below
    the ceiling over every certified step. (At ``gamma = 1``, whose set the plant is outside, the
    same plan crosses it at step 7 on every log.)"""
    gamma = 20.0
    for seed in range(3):
        path, certified, channel = _held(World(hidden=0.4), seed, gamma)
        radius = (gamma - 1.0) / (gamma + 1.0) * np.linalg.norm(channel)
        assert np.linalg.norm(TRUE_CHANNEL - channel) < radius
        assert certified > 0
        assert _safe(path, certified), (seed, path[:, 0].max())


def _confounded_fit(seed: int) -> CausalDynamicsFit:
    """``xdot = -x + u - 3 Z``, ``Z = +-1`` unlogged, 2000 transitions: the logged lever's sign
    follows ``Z`` with probability ``(1 + t)/2``, ``t = 1 - 1/LOGGED``, so its density given ``Z``
    moves by a factor within ``[1/LOGGED, LOGGED]``. A column of noise is the asserted adjustment
    set, and the channel the Euler fit reads is about 0.2 for the plant's 1."""
    rng = np.random.default_rng(seed)
    rows, tilt = 2000, 1.0 - 1.0 / LOGGED
    z = rng.choice([-1.0, 1.0], rows)
    u = np.where(rng.random(rows) < (1.0 + tilt) / 2.0, z, -z) * np.abs(rng.normal(size=rows))
    x = rng.normal(size=rows)
    x_next = x + DT * (-x + u - 3.0 * z) + math.sqrt(DT) * 0.3 * rng.normal(size=rows)
    data = {
        "x": jnp.asarray(x[:, None]),
        "u": jnp.asarray(u[:, None]),
        "x_next": jnp.asarray(x_next[:, None]),
        "w": jnp.asarray(rng.normal(size=(rows, 1))),
    }
    return fit_causal_residual(UNKNOWN, data, DT, adjust_for=("w",), channel_degree=0)


def test_a_ceiling_held_under_the_channel_s_sensitivity_model_keeps_the_plant_safe() -> None:
    """Ten logs whose lever follows a hidden confounder at level 1.5, each fitted by Euler with an
    adjustment set that misses it: the channel reads 0.07 to 0.27 for the plant's 1. Under the
    fit's own sensitivity model (:class:`~chc.plan.ChannelSensitivity`) at ``gamma = 2`` the radius,
    3.3 to 3.5, holds the plant's channel on every log; the plan held to ``x <= 1`` under it,
    chasing 2, is certified on all 30 steps of each, and the plant's own path keeps 0.70 or more
    below the ceiling. (The plan not held, audited by a share of its estimate at the same level,
    was certified on all 30 steps of two logs, along which the plant crosses the ceiling.)"""
    gamma, horizon, box = 2.0, 30, 5.0
    cost = QuadraticCost(
        Q=jnp.eye(1), R=1e-4 * jnp.eye(1), Qf=jnp.eye(1), x_target=jnp.array([2.0])
    )
    plant = LinearDynamics(jnp.array([[-1.0]]), jnp.array([[1.0]]))
    start = jnp.zeros(1)
    for seed in range(10):
        fit = _confounded_fit(seed)
        model = HybridDynamics(known=UNKNOWN, residual=fit.residual)
        sensitivity = ChannelSensitivity.from_fit(fit)
        fitted = float(np.asarray(fit.residual.channel).ravel()[0])
        assert abs(fitted - 1.0) <= sensitivity.radius(start, gamma), (seed, fitted)
        plan = causal_plan(
            model,
            start,
            cost,
            DT,
            horizon,
            -box,
            box,
            barrier=BarrierConstraint(_unit_ceiling, gamma=gamma, sensitivity=sensitivity),
        )
        audit = certify_safety(
            plan, model, _unit_ceiling, DT, gamma=gamma, u_max=box, sensitivity=sensitivity
        )
        assert audit.radius_model == "absolute"
        assert audit.certified_steps > 0
        barrier = 1.0 - np.asarray(rollout(plant, start, plan.actions, DT))[:, 0]
        assert np.all(barrier[: audit.certified_steps + 1] >= 0.0), (seed, barrier.min())


def test_a_ceiling_at_every_level_of_a_share_of_the_estimate_says_its_radius_is_relative() -> None:
    """Without a sensitivity model the radius is a share of the fitted channel, so at every level
    it stays within the fit's own size of it: a fitted channel of 0.004 where the plant's is 1
    reads ``gamma* = inf`` and 60 of 60 steps certified at ``gamma = 1e6``, and the plant crosses
    the ceiling at state 3 (least barrier value -3.99). The certificate says its radius is
    relative, a margin on the estimate and not a bound on the channel's error."""
    horizon, dt, box = 60, 0.1, 5.0
    fitted = LinearDynamics(jnp.array([[-1.0]]), jnp.array([[0.004]]))
    plant = LinearDynamics(jnp.array([[-1.0]]), jnp.array([[1.0]]))
    cost = QuadraticCost(
        Q=jnp.eye(1), R=1e-8 * jnp.eye(1), Qf=jnp.eye(1), x_target=jnp.array([2.0])
    )
    start = jnp.array([0.0])
    plan = causal_plan(fitted, start, cost, dt, horizon, -box, box)
    audit = certify_safety(plan, fitted, lambda x: 1.0 - x[0], dt, gamma=1e6, u_max=box)
    barrier = 1.0 - np.asarray(rollout(plant, start, plan.actions, dt))[:, 0]
    holds = audit.gamma_star != np.inf or bool(np.all(barrier[: audit.certified_steps + 1] >= 0.0))
    assert holds or audit.radius_model == "relative", barrier.min()
