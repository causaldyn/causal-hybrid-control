"""The barrier's certified steps and its confounding ceiling on plants the fit gets wrong.

A plan held at its barrier at level ``gamma`` is certified on a step where the barrier condition
holds for every channel in the sensitivity set at that level, and where the model's path is in the
safe set at that step's state and the next. The contract read here: where the plant's channel is
inside the set the plan was held and audited at, the plant's own mean path, with demand, the
disturbance and the state noise at zero, stays in the safe set over every certified step; and
``gamma_star = inf``, which says the barrier holds at every level, holds for a plant whose
confounding reaches any level.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from _misspec_world import World, ask

from chc import Constraint
from chc.cost import QuadraticCost
from chc.dynamics import LinearDynamics
from chc.integrate import rollout
from chc.plan import causal_plan, certify_safety

pytestmark = pytest.mark.misspecification

CEILING, WAIT = 0.8, 0.5
TRUE_CHANNEL = np.array([0.8, -0.4])  # supply's and wait's, per unit of the incentive


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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the sensitivity set is a share of the fitted channel, so at every level it stays within "
        "the fit's own size of it: a fitted channel of 0.004 where the plant's is 1 reads "
        "gamma* = inf and 60 of 60 steps certified at gamma = 1e6, and the plant crosses the "
        "ceiling at state 3 (least barrier value -3.99)"
    ),
)
def test_a_ceiling_at_every_level_holds_for_a_channel_confounded_far_from_its_estimate() -> None:
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
    if audit.gamma_star == np.inf:
        assert np.all(barrier[: audit.certified_steps + 1] >= 0.0), barrier.min()
