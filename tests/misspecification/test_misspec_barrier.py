"""The barrier's certified steps and its confounding ceiling on plants the fit gets wrong.

A plan held at its barrier at level ``gamma`` is certified on a step where the barrier condition
holds for every channel in the sensitivity set at that level, and where the model's path, widened
by the deviation a plant whose channel is in the set can reach, is in the safe set at that step's
state and the next. The contract read here: where the plant's channel is inside the set the plan
was held and audited at, the plant's own mean path, with demand, the disturbance and the state
noise at zero, stays in the safe set over every certified step, on a stable drift, an integrator
and an unstable drift, under an absolute radius and a relative one; and
``gamma_star = inf``, which says the barrier holds at every level, holds for a plant whose
confounding reaches any level, or the certificate says its radius is a margin relative to the
estimate (``radius_model``), not a bound on the channel's error.
"""

from __future__ import annotations

import math

import equinox as eqx
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
from chc.plan import (
    BarrierConstraint,
    ChannelSensitivity,
    SafetyCertificate,
    causal_plan,
    certify_safety,
)
from chc.residual import ControlAffineResidual

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


def _confounded_fit(seed: int, drift: float = -1.0, dt: float = DT) -> CausalDynamicsFit:
    """``xdot = drift x + u - 3 Z``, ``Z = +-1`` unlogged, 2000 transitions of length ``dt``: the
    logged lever's sign follows ``Z`` with probability ``(1 + t)/2``, ``t = 1 - 1/LOGGED``, so its
    density given ``Z`` moves by a factor within ``[1/LOGGED, LOGGED]``. A column of noise is the
    asserted adjustment set, and the channel the Euler fit reads is about 0.2 for the plant's 1."""
    rng = np.random.default_rng(seed)
    rows, tilt = 2000, 1.0 - 1.0 / LOGGED
    z = rng.choice([-1.0, 1.0], rows)
    u = np.where(rng.random(rows) < (1.0 + tilt) / 2.0, z, -z) * np.abs(rng.normal(size=rows))
    x = rng.normal(size=rows)
    x_next = x + dt * (drift * x + u - 3.0 * z) + math.sqrt(dt) * 0.3 * rng.normal(size=rows)
    data = {
        "x": jnp.asarray(x[:, None]),
        "u": jnp.asarray(u[:, None]),
        "x_next": jnp.asarray(x_next[:, None]),
        "w": jnp.asarray(rng.normal(size=(rows, 1))),
    }
    return fit_causal_residual(UNKNOWN, data, dt, adjust_for=("w",), channel_degree=0)


# Each drift's own step and horizon: the stable drift and the unstable one at dt = 0.1, the
# integrator at dt = 1, where a plan certified by the condition alone took the plant at the
# radius's edge to h = -0.5.
DRIFTS = {"stable": (-1.0, 0.1, 30), "integrator": (0.0, 1.0, 12), "unstable": (0.3, 0.1, 30)}


def _read_as(
    radius: str, sensitivity: ChannelSensitivity, gamma: float, size: float
) -> tuple[float, float, ChannelSensitivity | None]:
    """``(gamma, cvar_gap, sensitivity)`` of a radius as wide at the start as ``sensitivity``'s at
    ``gamma``: that sensitivity itself (``absolute``), or the share of a channel of ``size`` at
    level 2, whose factor is 1/3 (``relative``)."""
    if radius == "absolute":
        return gamma, 1.0, sensitivity
    return 2.0, 3.0 * sensitivity.radius(jnp.zeros(1), gamma) / size, None


def _held_and_audited(
    model: LinearDynamics | HybridDynamics,
    dt: float,
    horizon: int,
    read: tuple[float, float, ChannelSensitivity | None],
    target: float,
    weight: float,
) -> tuple[jnp.ndarray, SafetyCertificate]:
    """The plan held to ``x <= 1`` from 0 chasing ``target`` under the radius ``read``
    (:func:`_read_as`), each action priced at ``weight``, and its audit by the same radius."""
    gamma, share, sensitivity = read
    start = jnp.zeros(1)
    cost = QuadraticCost(
        Q=jnp.eye(1), R=weight * jnp.eye(1), Qf=jnp.eye(1), x_target=jnp.array([target])
    )
    barrier = BarrierConstraint(_unit_ceiling, gamma=gamma, cvar_gap=share, sensitivity=sensitivity)
    plan = causal_plan(model, start, cost, dt, horizon, -5.0, 5.0, barrier=barrier)
    audit = certify_safety(
        plan,
        model,
        _unit_ceiling,
        dt,
        gamma=gamma,
        cvar_gap=share,
        u_max=5.0,
        sensitivity=sensitivity,
    )
    return plan.actions, audit


@pytest.mark.parametrize("radius", ["absolute", "relative"])
@pytest.mark.parametrize("drift", list(DRIFTS))
def test_certified_steps_keep_every_plant_in_the_radius_safe(drift: str, radius: str) -> None:
    """A known drift and a channel of 1, off by at most 0.3 in the plant (0.5 at dt = 1): the plan
    held below ``x <= 1`` under that radius, absolute or relative, keeps every plant whose channel
    is within it, stepped as the model is, in the set over every certified step. The condition
    read at the model's states alone certified every step while the plant at the radius's edge
    left the set: by 0.17 on the unstable drift, by 0.5 on the integrator, under either radius."""
    a, dt, horizon = DRIFTS[drift]
    delta = 0.5 if drift == "integrator" else 0.3
    model = LinearDynamics(jnp.array([[a]]), jnp.array([[1.0]]))
    exact = ChannelSensitivity(
        gram=np.eye(1), channel_degree=0, sampling=delta, confounding=0.0, level=0.95
    )
    actions, audit = _held_and_audited(
        model, dt, horizon, _read_as(radius, exact, 1.0, 1.0), 3.0, 1e-2
    )
    assert audit.radius_model == radius
    assert audit.certified_steps > 0
    worst = math.inf
    for channel in np.linspace(1.0 - delta, 1.0 + delta, 21):
        plant = LinearDynamics(jnp.array([[a]]), jnp.array([[channel]]))
        path = np.asarray(rollout(plant, jnp.zeros(1), actions, dt))
        worst = min(worst, float(np.min(1.0 - path[: audit.certified_steps + 1, 0])))
    assert worst >= 0.0, (drift, radius, audit.certified_steps, worst)


def _with_channel(model: HybridDynamics, channel: float) -> HybridDynamics:
    """``model`` with its constant channel set to ``channel``: its own drift, another channel."""
    residual = model.residual
    assert isinstance(residual, ControlAffineResidual)
    moved = eqx.tree_at(
        lambda term: term.channel, residual, jnp.full_like(residual.channel, channel)
    )
    return HybridDynamics(known=model.known, residual=moved)


@pytest.mark.parametrize("radius", ["absolute", "relative"])
@pytest.mark.parametrize("drift", list(DRIFTS))
def test_a_ceiling_held_under_the_channel_s_sensitivity_model_keeps_the_plant_safe(
    drift: str, radius: str
) -> None:
    """Four logs per drift whose lever follows a hidden confounder at level 1.5, each fitted by
    Euler with an adjustment set that misses it: the channel reads about 0.2 for the plant's 1.
    The fit's own sensitivity model (:class:`~chc.plan.ChannelSensitivity`) at ``gamma = 1.5``
    holds the plant's channel on every log, and the plan is held to ``x <= 1`` under that radius,
    read as the sensitivity's or as the share of the fitted channel that is as wide at the start.
    The plant with the fitted drift and the true channel, which is what the radius is about,
    stays in the set over every certified step. Certified by the condition alone, the first log's
    plan took it out to ``h = -2.04`` on the unstable drift and to ``h = -10.5`` on the
    integrator, under either radius."""
    a, dt, horizon = DRIFTS[drift]
    gamma = 1.5
    for seed in range(4):
        fit = _confounded_fit(seed, a, dt)
        model = HybridDynamics(known=UNKNOWN, residual=fit.residual)
        sensitivity = ChannelSensitivity.from_fit(fit)
        fitted = float(np.asarray(fit.residual.channel).ravel()[0])
        width = sensitivity.radius(jnp.zeros(1), gamma)
        assert abs(fitted - 1.0) <= width, (seed, fitted, width)
        read = _read_as(radius, sensitivity, gamma, abs(fitted))
        actions, audit = _held_and_audited(model, dt, horizon, read, 2.0, 1e-4)
        assert audit.radius_model == radius
        assert audit.certified_steps > 0
        path = np.asarray(rollout(_with_channel(model, 1.0), jnp.zeros(1), actions, dt))
        barrier = 1.0 - path[: audit.certified_steps + 1, 0]
        assert np.all(barrier >= 0.0), (seed, barrier.min())


def test_a_relative_radius_holds_the_plants_in_its_share_and_says_so() -> None:
    """Without a sensitivity model the radius is a share of the fitted channel, so at every level
    it stays within the fit's own size of it: a fitted channel of 0.004 where the plant's is 1
    reads ``gamma* = inf`` at ``gamma = 1e6``, and the plant, outside that radius, crosses the
    ceiling at state 3 (least barrier value -3.99). The certificate says its radius is relative,
    and what it certifies holds for the plants inside it. An absolute radius of 1 on the same
    plan holds the plant's channel, and certifies no step the plant leaves the set at."""
    horizon, dt, box = 60, 0.1, 5.0
    fitted = LinearDynamics(jnp.array([[-1.0]]), jnp.array([[0.004]]))
    plant = LinearDynamics(jnp.array([[-1.0]]), jnp.array([[1.0]]))
    cost = QuadraticCost(
        Q=jnp.eye(1), R=1e-8 * jnp.eye(1), Qf=jnp.eye(1), x_target=jnp.array([2.0])
    )
    start = jnp.array([0.0])
    plan = causal_plan(fitted, start, cost, dt, horizon, -box, box)
    left = 1.0 - np.asarray(rollout(plant, start, plan.actions, dt))[:, 0]
    assert left.min() < -3.9

    gamma = 1e6
    audit = certify_safety(plan, fitted, _unit_ceiling, dt, gamma=gamma, u_max=box)
    assert audit.radius_model == "relative"
    assert audit.gamma_star == math.inf
    share = (gamma - 1.0) / (gamma + 1.0) * 0.004
    assert abs(1.0 - 0.004) > share  # the plant is outside the radius
    for channel in np.linspace(0.004 - share, 0.004 + share, 11):
        inside = LinearDynamics(jnp.array([[-1.0]]), jnp.array([[channel]]))
        path = np.asarray(rollout(inside, start, plan.actions, dt))[: audit.certified_steps + 1]
        assert np.all(1.0 - path[:, 0] >= 0.0), channel

    absolute = certify_safety(
        plan,
        fitted,
        _unit_ceiling,
        dt,
        u_max=box,
        sensitivity=ChannelSensitivity(
            gram=np.eye(1), channel_degree=0, sampling=1.0, confounding=0.0, level=0.95
        ),
    )
    assert absolute.radius_model == "absolute"
    assert np.all(left[: absolute.certified_steps + 1] >= 0.0), absolute.certified_steps
