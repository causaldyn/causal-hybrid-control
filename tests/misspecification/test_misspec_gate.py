"""The deployment gate's DEPLOY where rewards persist and where an action pays some periods late.

A DEPLOY claims that the candidate beat the baseline by more than ``delta``, and across paths the
gate keeps wrong ones to ``alpha = 0.05``. The contract read here: on a zone whose candidate does
not beat the baseline, the gate deploys on at most ``alpha`` of the paths, within three Monte
Carlo standard deviations, or it says it cannot read the case.

Every zone logs as its mode asks: the baseline takes action 1 with probability 0.5, the candidate
with 0.8, an experiment mixes the two, and the reward is Bernoulli.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from chc.gate import BlockBatch, DeploymentGate, GateConfig, ZoneBatch, ZonePlan

pytestmark = pytest.mark.misspecification

BASELINE, CANDIDATE, ALPHA = 0.5, 0.8, 0.05
PLAN = ZonePlan(
    chi2=CANDIDATE**2 / BASELINE + (1 - CANDIDATE) ** 2 / (1 - BASELINE) - 1.0,
    tv=CANDIDATE - BASELINE,
)
DEAD, GAIN, COST = 3, 0.4, 0.6  # an action pays GAIN now and costs COST three periods on


def _slack(paths: int) -> float:
    return ALPHA + 3.0 * math.sqrt(ALPHA * (1.0 - ALPHA) / paths)


def _per_decision(paths: int, decisions: int, reward, batch: int = 50) -> float:
    """The share of paths on which the gate, read after every ``batch`` decisions, deploys.
    ``reward(rng, action, state)`` returns the reward and the next state."""
    config = GateConfig(delta=0.0, delta_harm=0.02, min_effect=0.02, horizon=decisions)
    deployed = 0
    for path in range(paths):
        rng = np.random.default_rng(path)
        gate = DeploymentGate({"z": PLAN}, config)
        state = None
        for _ in range(decisions // batch):
            mode = gate.mode("z")
            if mode == "retired":
                break
            p = {
                "shadow": BASELINE,
                "deployed": CANDIDATE,
                "experiment": (1 - config.rho) * BASELINE + config.rho * CANDIDATE,
            }[mode]
            action = rng.random(batch) < p
            rewards = np.empty(batch)
            for t in range(batch):
                rewards[t], state = reward(rng, action[t], state)
            verdict = gate.update(
                {
                    "z": ZoneBatch(
                        rewards,
                        np.where(action, CANDIDATE, 1 - CANDIDATE),
                        np.where(action, BASELINE, 1 - BASELINE),
                        np.where(action, p, 1 - p),
                    )
                }
            )["z"]
            if verdict == "deploy":
                deployed += 1
                break
    return deployed / paths


def _persistent_null(rng: np.random.Generator, action: bool, level: float | None):
    """No contrast; the reward's mean wanders as ``0.5 + 0.2 tanh(s)``, ``s`` AR(0.95)."""
    s = rng.normal() if level is None else level
    s = 0.95 * s + math.sqrt(1.0 - 0.95**2) * rng.normal()
    return float(rng.random() < 0.5 + 0.2 * math.tanh(s)), s


def _late(rng: np.random.Generator, action: bool, past: tuple[bool, ...] | None):
    """The action pays ``GAIN`` now and costs ``COST`` ``DEAD`` periods on: myopically the
    candidate is ``0.3 GAIN = +0.12`` better, sustained ``0.3 (GAIN - COST) = -0.06`` worse."""
    past = (False,) * DEAD if past is None else past
    mean = 0.6 + GAIN * action - COST * past[0]
    return float(rng.random() < mean), (*past[1:], action)


def _blocks(paths: int, blocks: int, cost: float = COST, per_read: int = 5) -> float:
    """The share of paths on which the gate in blocks of 10 periods, the first ``DEAD`` burned,
    deploys the candidate of :func:`_late` with a late ``cost``, read every ``per_read`` blocks."""
    config = GateConfig(
        delta=0.0,
        delta_harm=0.02,
        min_effect=0.02,
        horizon=10**6,
        block_length=10,
        burn_in=DEAD,
    )
    rng = np.random.default_rng(20261019)
    gates = [DeploymentGate({"z": PLAN}, config) for _ in range(paths)]
    past = np.zeros((paths, DEAD), dtype=bool)
    deployed = np.zeros(paths, dtype=bool)
    for _ in range(blocks // per_read):
        modes = np.array([gate.mode("z") for gate in gates])[:, None]
        experiment = modes == "experiment"
        ran = np.where(experiment, rng.random((paths, per_read)) < config.rho, modes == "deployed")
        logged = np.where(experiment, np.where(ran, config.rho, 1.0 - config.rho), 1.0)
        reward = np.empty((paths, per_read, 10))
        for b in range(per_read):
            p = np.where(ran[:, b], CANDIDATE, BASELINE)
            for t in range(10):
                action = rng.random(paths) < p
                mean = 0.6 + GAIN * action - cost * past[:, 0]
                reward[:, b, t] = rng.random(paths) < mean
                past = np.concatenate([past[:, 1:], action[:, None]], axis=1)
        for i, gate in enumerate(gates):
            if modes[i, 0] != "retired":
                verdict = gate.update({"z": BlockBatch(reward[i], ran[i], logged[i])})["z"]
                deployed[i] |= verdict == "deploy"
    return float(np.mean(deployed))


def test_rewards_that_persist_deploy_no_candidate_that_is_no_better() -> None:
    """A reward whose mean wanders with an AR(0.95) common to both actions: no deploy on 100
    paths of 6000 decisions."""
    assert _per_decision(100, 6000, _persistent_null) <= _slack(100)


def test_in_blocks_a_late_cost_is_read_and_not_deployed() -> None:
    """Blocks whose burned periods cover the dead time, so the read periods see only their own
    block's policy: over 1500 blocks the candidate is deployed on none of 200 paths, where the
    same candidate without its late cost, +0.12 better sustained, is deployed on 66."""
    assert _blocks(200, 1500) <= _slack(200)
    assert _blocks(200, 1500, cost=0.0) >= 0.2


@pytest.mark.xfail(
    strict=True,
    reason=(
        "without blocks the gate reads each decision's myopic contrast, and a cost three periods "
        "on is not in it: the candidate, -0.06 worse sustained, is deployed on 40 of 40 paths. "
        "The module's docstring names the premise; nothing at run time looks for it, though the "
        "logged randomised actions would let a lagged-action term in the reward be tested"
    ),
)
def test_a_late_cost_is_not_deployed_or_is_said_to_be_unread() -> None:
    assert _per_decision(40, 6000, _late) <= _slack(40)
