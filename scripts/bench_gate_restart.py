"""The measurements behind the 2026-11-25 amendment of
docs/adr/0146-a-deployed-zone-in-blocks-may-keep-a-randomised-baseline-and-a-designed-test-s-intervals-read-its-innovations.md:
a deployed zone's rollback read since the DEPLOY (``GateConfig(rollback_restart=0)``, today's
default) against the rollback started again on a grid (``rollback_restart=K``), with the drift
alarm's epoch weights or with weights alike over a horizon. Counts, rates and medians only, no
wall time. Every setting of a cell reads the same stream.

    blocks     The ADR's cells, in blocks, each zone deployed by twenty blocks the candidate wins
               outright and then run with the baseline's blocks at a share ``q`` of 0.1, 0.2 and
               0.3, plain and centred, --paths paths a cell:
               turn40   the lab's plant, blocks of 50 with 40 burned, read every 5 over 400 blocks:
                        the candidate acts ``N(0.4, 1)`` and from 40 blocks after the DEPLOY
                        ``N(1, 1)``, 0.157 worse than the baseline; rolled back within the 360
                        blocks after the turn.
               o1worse  a plant of order 1, the reward ``clip(0.5 + 0.1 u_t - 0.2 u_(t-1) +
                        0.2 eta, 0, 1)``, the candidate acting 1 and the baseline 0, so 0.1 worse
                        sustained from the DEPLOY; blocks of 2 with 1 burned, read every 10 over
                        4000 blocks.
               o1turn   the same plant with a gain of 0.3 for the first 100 blocks after the
                        DEPLOY, 0.1 better sustained, and 0.1 after, 0.1 worse; over 2000 blocks,
                        rolled back within the 1900 after the turn.
               o1null   gain and cost both -0.2, exactly as good sustained: every rollback is
                        wrong; over 2000 blocks.
               Per cell and setting: the rollbacks before the turn, and the rate and the median
               blocks after it within the window, or, on o1null, the wrong rollbacks.
    decisions  Per decision, the module docstring's contrast, read every 50 decisions: the baseline
               takes action 1 with probability 0.5 and the candidate 0.8, the reward Bernoulli at
               0.6 for action 1 and 0.4 for action 0 (+0.06 for the candidate) until the turn,
               then the reverse (-0.06), or 0.5 for both (flat: a ROLLBACK is wrong). A turn at
               3000 of 6000 decisions, one at 10 000 of 16 000, and flat from 3000 of 23 000,
               --decision-paths paths, each path its own stream. Per setting: the paths deployed
               before the turn, rolled back before it, and rolled back after it, with the median
               decisions from the turn.

The settings: ``rollback_restart`` 0, the epoch weights at ``K`` 10, 50 and 250 and alike at
``K = 250`` over 10 000 blocks in blocks; 0 and the epoch weights at 50, 250 and 1000 per decision.
Cells run over --workers processes.

Run: JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run python scripts/bench_gate_restart.py > out.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import jax
import numpy as np

from chc.gate import BlockBatch, DeploymentGate, GateConfig, ZoneBatch, ZonePlan

Period = Callable[[np.random.Generator, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
Start = Callable[[np.random.Generator, int], np.ndarray]

# read by no rule in blocks
PLAN = ZonePlan(chi2=math.exp(0.4**2) - 1.0, tv=0.1585)
SHARES = (0.1, 0.2, 0.3)
IN_BLOCKS = {
    "since_deploy": {"rollback_restart": 0},
    "epochs_K10": {"rollback_restart": 10},
    "epochs_K50": {"rollback_restart": 50},
    "epochs_K250": {"rollback_restart": 250},
    "alike_K250_H10000": {"rollback_restart": 250, "rollback_horizon": 10_000},
}
PER_DECISION = {"since_deploy": 0, "epochs_K50": 50, "epochs_K250": 250, "epochs_K1000": 1000}
# (mode, turn, decisions)
CONTRASTS = (("turn", 3000, 6000), ("turn", 10_000, 16_000), ("flat", 3000, 23_000))


def _turning(after: int) -> Period:
    """The lab's plant whose candidate acts ``N(0.4, 1)`` for its first ``after`` periods after the
    DEPLOY and ``N(1, 1)`` from then on."""
    periods = [0]

    def period(rng: np.random.Generator, x: np.ndarray, candidate: np.ndarray) -> tuple:
        action = 0.4 if periods[0] < after else 1.0
        periods[0] += 1
        u = np.where(candidate, action, 0.0) + rng.normal(size=x.shape)
        mean = 0.3 * np.exp(-((u - 1.0) ** 2) / 2.0) + 0.7 * np.exp(-(x**2) / 2.0)
        reward = (rng.random(x.shape) < mean).astype(float)
        return reward, 0.9 * x + 0.1 * u + 0.2 * rng.normal(size=x.shape)

    return period


def _order_one(gain: float, cost: float, *, until: int = 0, then: float = 0.0) -> Period:
    """Carryover of order 1 with Gaussian noise: the candidate acts 1 and the baseline 0, and the
    reward is ``clip(0.5 + gain u_t - cost u_(t-1) + 0.2 eta_t, 0, 1)``, the gain ``then`` from the
    ``until``-th period after the DEPLOY on, where ``until`` is positive."""
    periods = [0]

    def period(rng: np.random.Generator, last: np.ndarray, candidate: np.ndarray) -> tuple:
        now = gain if until == 0 or periods[0] < until else then
        periods[0] += 1
        u = candidate.astype(float)
        noise = 0.2 * rng.standard_normal(u.shape)
        return np.clip(0.5 + now * u - cost * last + noise, 0.0, 1.0), u

    return period


def _deployed(
    period: Period,
    config: GateConfig,
    *,
    paths: int,
    blocks: int,
    per_read: int,
    seed: int,
    start: Start,
) -> np.ndarray:
    """The blocks after its DEPLOY at which each path's gate rolled its zone back, -1 where it did
    not. Twenty blocks the candidate wins outright deploy the zone; then each block runs the
    baseline with probability ``rollback_baseline_share``. One stream serves every path, so it is
    the same whatever the gates decide."""
    assert config.block_length is not None
    share = config.rollback_baseline_share or 0.0
    rng = np.random.default_rng(seed)
    gates = [DeploymentGate({"z": PLAN}, config) for _ in range(paths)]
    ran = np.arange(20) % 2 == 0
    length = config.block_length
    win = BlockBatch(np.repeat(ran[:, None], length, axis=1).astype(float), ran, np.full(20, 0.5))
    for gate in gates:
        assert gate.update({"z": win}) == {"z": "deploy"}
    state = start(rng, paths)
    rolled = np.full(paths, -1)
    for read in range(blocks // per_read):
        candidate = rng.random((paths, per_read)) >= share
        logged = np.where(candidate, 1.0 - share, share)
        reward = np.empty((paths, per_read, length))
        for b in range(per_read):
            for t in range(length):
                reward[:, b, t], state = period(rng, state, candidate[:, b])
        for i, gate in enumerate(gates):
            if gate.mode("z") != "deployed":
                continue
            if gate.update({"z": BlockBatch(reward[i], candidate[i], logged[i])}) == {
                "z": "rollback"
            }:
                rolled[i] = (read + 1) * per_read
    return rolled


def _lab_start(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.normal(0.4, 0.51, n)


def _acted(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.ones(n)


def _cell(cell: str, centred: bool, share: float, setting: str, paths: int) -> dict[str, Any]:
    logging.disable(logging.CRITICAL)
    one = {"block_length": 2, "burn_in": 1}
    lab = {"block_length": 50, "burn_in": 40}
    # (period, shape, blocks, read every, seed, start, the turn's block, the window after it)
    runs: dict[str, tuple[Period, dict[str, int], int, int, int, Start, int | None, int]] = {
        "turn40": (_turning(40 * 50), lab, 400, 5, 20261028, _lab_start, 40, 360),
        "o1worse": (_order_one(0.1, 0.2), one, 4000, 10, 20261029, _acted, 0, 4000),
        "o1turn": (
            _order_one(0.3, 0.2, until=200, then=0.1),
            one,
            2000,
            10,
            20261030,
            _acted,
            100,
            1900,
        ),
        "o1null": (_order_one(-0.2, -0.2), one, 2000, 10, 20261028, _acted, None, 2000),
    }
    period, shape, blocks, per_read, seed, start, turn, window = runs[cell]
    options: dict[str, Any] = {
        "delta": 0.0,
        "delta_harm": 0.02,
        "min_effect": 0.02,
        "horizon": 10**6,
        "rollback_baseline_share": share,
        "centred": centred,
    }
    config = GateConfig(**options, **shape, **IN_BLOCKS[setting])
    rolled = _deployed(
        period, config, paths=paths, blocks=blocks, per_read=per_read, seed=seed, start=start
    )
    out: dict[str, Any] = {"cell": cell, "centred": centred, "share": share, "setting": setting}
    if turn is None:
        out["wrong"] = int((rolled >= 0).sum())
        return out
    after = rolled[rolled > turn] - turn
    within = after[after <= window]
    out["before_the_turn"] = int(((rolled >= 0) & (rolled <= turn)).sum())
    out["within"] = int(within.size)
    out["rate"] = within.size / paths
    out["median"] = float(np.median(within)) if within.size else None
    return out


def _contrast(mode: str, turn: int, decisions: int, setting: str, paths: int) -> dict[str, Any]:
    logging.disable(logging.CRITICAL)
    pb, pc, batch = 0.5, 0.8, 50
    plan = ZonePlan(chi2=pc**2 / pb + (1 - pc) ** 2 / (1 - pb) - 1.0, tv=pc - pb)
    config = GateConfig(
        delta=0.0,
        delta_harm=0.0,
        min_effect=0.03,
        horizon=decisions,
        rollback_restart=PER_DECISION[setting],
    )
    deployed = early = 0
    delays = []
    for path in range(paths):
        rng = np.random.default_rng(path)
        gate = DeploymentGate({"z": plan}, config)
        before = False
        for k in range(decisions // batch):
            state = gate.mode("z")
            if state == "retired":
                break
            p = {
                "shadow": pb,
                "deployed": pc,
                "experiment": (1 - config.rho) * pb + config.rho * pc,
            }[state]
            a = rng.random(batch) < p
            t = k * batch + np.arange(batch)
            good = np.where(a, 0.6, 0.4)
            later = 1.0 - good if mode == "turn" else np.full(batch, 0.5)
            r = (rng.random(batch) < np.where(t < turn, good, later)).astype(float)
            logged = ZoneBatch(
                r, np.where(a, pc, 1 - pc), np.where(a, pb, 1 - pb), np.where(a, p, 1 - p)
            )
            verdict = gate.update({"z": logged})["z"]
            read = (k + 1) * batch
            before |= verdict == "deploy" and read <= turn
            if verdict == "rollback":
                if read <= turn:
                    early += 1
                elif before:
                    delays.append(read - turn)
        deployed += before
    return {
        "contrast": f"{mode} at {turn} of {decisions}",
        "setting": setting,
        "deployed_before_the_turn": deployed,
        "rolled_back_before_the_turn": early,
        "rolled_back_after": len(delays),
        "median_after": float(np.median(delays)) if delays else None,
    }


def blocks(paths: int, workers: int) -> list[dict[str, Any]]:
    tasks = list(
        itertools.product(
            ("turn40", "o1worse", "o1turn", "o1null"), (True, False), SHARES, IN_BLOCKS
        )
    )
    context = multiprocessing.get_context("spawn")  # JAX does not survive a fork
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        return list(pool.map(_cell, *zip(*tasks, strict=True), [paths] * len(tasks)))


def decisions(paths: int, workers: int) -> list[dict[str, Any]]:
    tasks = [(*contrast, setting) for contrast in CONTRASTS for setting in PER_DECISION]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        return list(pool.map(_contrast, *zip(*tasks, strict=True), [paths] * len(tasks)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=400)
    parser.add_argument("--decision-paths", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sections", nargs="+", default=["blocks", "decisions"])
    args = parser.parse_args()
    out: dict[str, Any] = {"x64": bool(jax.config.jax_enable_x64)}
    if "blocks" in args.sections:
        out["blocks"] = blocks(args.paths, args.workers)
    if "decisions" in args.sections:
        out["decisions"] = decisions(args.decision_paths, args.workers)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
