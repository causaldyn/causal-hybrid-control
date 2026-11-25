"""The measurements behind docs/adr/0119-the-barrier-certificate-carries-the-plant-s-deviation.md's
amendment of 2026-11-25: the barrier's room with and without the bound on the drift's error
(``certify_safety(drift_error=)``, ``prescribe(barrier_drift_error=)``), on many logs. Counts,
steps and ratios only, no wall time.

Each log is the 20-log misspecification test's (``tests/misspecification/test_misspec_barrier.py``):
``xdot = 0.5 x + u - 3 Z``, ``Z = +-1`` unlogged, 2000 transitions of length 0.1, the lever's sign
following ``Z`` at a confounding of 1.5, fitted by Euler with a column of noise as the asserted
adjustment set. The plan steers ``x`` to 2 over 30 steps in the box ``[-5, 5]``, held below the
ceiling ``x <= 1`` under the fit's own sensitivity at ``gamma = 1.5``, once with the bound
``t sqrt(n) drift_error`` in the room (``t`` Student's at 0.975 on the rows less one) and once
without it. The true plant, ``xdot = 0.5 x + u``, then runs the plan's actions.

    stay   --logs logs, seeds 0 on: per room, the logs on which the true plant stayed in the set
           over the certified steps, their Clopper-Pearson 95 % interval, the mean certified steps
           and the solver's statuses; the first 20 logs' count, the test's; the logs the room
           without the bound lost, with the true plant's least ``h``; and ADR 0115's floors at 0.9
           and 0.95 on 20 logs and on --logs (the largest count a rate at that level falls below
           with probability 1/1000 at most).
    reach  The plan held with the bound on the logs of --reach-seeds: the drift's own standard error
           at each state of the plan's predicted path against the bound's root mean over the log's
           rows, ``drift_error``, which the room reads.

Logs run over --workers processes.

Run: JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run python scripts/bench_drift_room.py > out.json
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.stats import beta
from scipy.stats import t as student_t

from chc.cost import QuadraticCost
from chc.decision import _features
from chc.dynamics import HybridDynamics, LinearDynamics
from chc.dynamics_id import CausalDynamicsFit, fit_causal_residual
from chc.integrate import rollout
from chc.plan import BarrierConstraint, CausalPlan, ChannelSensitivity, causal_plan

DRIFT, DT, HORIZON, BOX, GAMMA, LOGGED = 0.5, 0.1, 30, 5.0, 1.5, 1.5
UNKNOWN = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
TRUTH = LinearDynamics(jnp.array([[DRIFT]]), jnp.array([[1.0]]))
START = jnp.zeros(1)
COST = QuadraticCost(Q=jnp.eye(1), R=1e-4 * jnp.eye(1), Qf=jnp.eye(1), x_target=jnp.array([2.0]))
FLOOR_LEVEL = Fraction(1, 1000)


def _ceiling(x: Array) -> Array:
    return 1.0 - x[0]


def _fit(seed: int) -> CausalDynamicsFit:
    rng = np.random.default_rng(seed)
    rows, tilt = 2000, 1.0 - 1.0 / LOGGED
    z = rng.choice([-1.0, 1.0], rows)
    u = np.where(rng.random(rows) < (1.0 + tilt) / 2.0, z, -z) * np.abs(rng.normal(size=rows))
    x = rng.normal(size=rows)
    x_next = x + DT * (DRIFT * x + u - 3.0 * z) + math.sqrt(DT) * 0.3 * rng.normal(size=rows)
    data = {
        "x": jnp.asarray(x[:, None]),
        "u": jnp.asarray(u[:, None]),
        "x_next": jnp.asarray(x_next[:, None]),
        "w": jnp.asarray(rng.normal(size=(rows, 1))),
    }
    return fit_causal_residual(UNKNOWN, data, DT, adjust_for=("w",), channel_degree=0)


def _bound(fit: CausalDynamicsFit) -> float:
    assert fit.drift_error is not None
    rows = int(np.asarray(fit.defect).shape[0])
    return float(student_t.ppf(0.975, rows - 1)) * fit.drift_error


def _held(fit: CausalDynamicsFit, drift_error: float | None) -> tuple[HybridDynamics, CausalPlan]:
    model = HybridDynamics(known=UNKNOWN, residual=fit.residual)
    barrier = BarrierConstraint(
        _ceiling,
        gamma=GAMMA,
        u_max=(BOX,),
        sensitivity=ChannelSensitivity.from_fit(fit),
        drift_error=drift_error,
    )
    return model, causal_plan(model, START, COST, DT, HORIZON, -BOX, BOX, barrier=barrier)


def _log(seed: int) -> dict[str, Any]:
    """One log: per room, the certified steps, whether the true plant stayed in the set over them,
    its least ``h`` there, and the solver's status."""
    fit = _fit(seed)
    bound = _bound(fit)
    row: dict[str, Any] = {"seed": seed, "bound": bound}
    for room, drift_error in (("with_bound", bound), ("without", None)):
        _, plan = _held(fit, drift_error)
        assert plan.safety is not None
        certified = int(plan.safety.certified_steps)
        path = np.asarray(rollout(TRUTH, START, plan.actions, DT))
        least = float(np.min(1.0 - path[: certified + 1, 0]))
        row[room] = {
            "certified": certified,
            "inside": least >= 0.0,
            "least_h": least,
            "status": str(plan.solver_status),
        }
    return row


def _clopper_pearson(count: int, logs: int) -> tuple[float, float]:
    low = 0.0 if count == 0 else float(beta.ppf(0.025, count, logs - count + 1))
    high = 1.0 if count == logs else float(beta.ppf(0.975, count + 1, logs - count))
    return low, high


def _floor(level: str, logs: int) -> int:
    """ADR 0115's floor: the largest ``c`` with ``P(count < c) <= 1/1000`` at the rate ``level``."""
    rate, tail, floor = Fraction(level), Fraction(0), 0
    while floor < logs:
        mass = math.comb(logs, floor) * rate**floor * (1 - rate) ** (logs - floor)
        if tail + mass > FLOOR_LEVEL:
            break
        tail += mass
        floor += 1
    return floor


def stay(logs: int, workers: int) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")  # JAX does not survive a fork
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        rows = list(pool.map(_log, range(logs)))
    out: dict[str, Any] = {"logs": logs, "seeds": [0, logs - 1]}
    for room in ("with_bound", "without"):
        reads = [row[room] for row in rows]
        inside = sum(read["inside"] for read in reads)
        out[room] = {
            "inside": inside,
            "rate": inside / logs,
            "clopper_pearson_95": _clopper_pearson(inside, logs),
            "mean_certified": float(np.mean([read["certified"] for read in reads])),
            "inside_first_20": sum(read["inside"] for read in reads[:20]),
            "status": dict(Counter(read["status"] for read in reads)),
        }
    out["left_without_the_bound"] = [
        {"seed": row["seed"], "least_h": row["without"]["least_h"]}
        for row in rows
        if not row["without"]["inside"]
    ]
    out["bound_range"] = [min(row["bound"] for row in rows), max(row["bound"] for row in rows)]
    out["floors"] = {
        f"{level} on {count}": _floor(level, count)
        for level in ("0.9", "0.95")
        for count in (20, logs)
    }
    return out


def reach(seeds: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for seed in seeds:
        fit = _fit(seed)
        model, plan = _held(fit, _bound(fit))
        assert fit.drift_error is not None
        assert fit._drift_blocks is not None
        states = np.asarray(rollout(model, START, plan.actions, DT))[:-1]
        blocks = np.asarray(fit._drift_blocks)
        features = np.asarray(_features(states, fit.residual.degree))
        drivers = np.zeros((states.shape[0], blocks.shape[1] - features.shape[1]))
        terms = np.concatenate([features, drivers], axis=1)
        # the drift's standard error at each state, read as drift_error reads it at the log's rows:
        # the root mean over the blocks of the value's variance
        error = np.sqrt(np.einsum("ha,gab,hb->h", terms, blocks, terms) / blocks.shape[0])
        ratio = error / fit.drift_error
        out[f"seed {seed}"] = {
            "states": [float(states.min()), float(states.max())],
            "error_over_bound": [float(ratio.min()), float(ratio.max())],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=int, default=200)
    parser.add_argument("--reach-seeds", type=int, nargs="+", default=[0, 18])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sections", nargs="+", default=["stay", "reach"])
    args = parser.parse_args()
    out: dict[str, Any] = {"x64": bool(jax.config.jax_enable_x64)}
    if "stay" in args.sections:
        out["stay"] = stay(args.logs, args.workers)
    if "reach" in args.sections:
        out["reach"] = reach(args.reach_seeds)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
