"""The measurements behind chc.switchback's randomisation test and interval, on the working model
``x_{t+1} = a x_t + b u_t + eps_t``, ``y_t = x_t``, where the Wald and Fieller intervals are
asymptotic: few blocks, carryover and short runs. Rates and widths only, no wall time.

    size      the null of no effect (b = 0): how often the randomisation test and the Wald or
              Fieller interval reject at 5%, for the block difference over 6, 8 and 12 blocks and
              the plug-in over 30 and 60 periods.
    coverage  b = 1: how often each interval covers, and its median width and share unbounded. The
              randomisation interval is projected over a persistence range holding the truth, over
              the true memory alone, and over no memory (a = 0), the null that names the effect
              and nothing else.
    memory    the same over strong carryover, a = 0.95: where a null without the plant's memory
              should fail, if anywhere.

Run: uv run python scripts/bench_switchback_randomisation.py {size,coverage,memory} [--replicates N]
     [--workers W]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from chc.switchback import (
    STEADY_STATE,
    BlockDesign,
    Horizon,
    MarkovDesign,
    randomisation_interval,
    randomisation_test,
    read_switchback,
)

SEED = 20260930
BURN = 200

# (name, analysis, estimand, design, count of blocks or periods, a, persistence range)
SIZE = [
    ("blocks=6", "block_dim", Horizon(5), BlockDesign(5, 4), 6, 0.8, None),
    ("blocks=8", "block_dim", Horizon(5), BlockDesign(5, 4), 8, 0.8, None),
    ("blocks=12", "block_dim", Horizon(5), BlockDesign(5, 4), 12, 0.8, None),
    ("plug-in tau_3, T=30", "state_aware", Horizon(3), MarkovDesign(0.3), 30, 0.8, None),
    ("plug-in tau_3, T=60", "state_aware", Horizon(3), MarkovDesign(0.3), 60, 0.8, None),
    ("plug-in steady state, T=60", "state_aware", STEADY_STATE, MarkovDesign(0.3), 60, 0.9, None),
]
COVERAGE = [
    ("blocks=8", "block_dim", Horizon(5), BlockDesign(5, 4), 8, 0.8, (0.6, 0.9)),
    ("blocks=12", "block_dim", Horizon(5), BlockDesign(5, 4), 12, 0.8, (0.6, 0.9)),
    ("blocks=20", "block_dim", Horizon(5), BlockDesign(5, 4), 20, 0.8, (0.6, 0.9)),
    ("plug-in tau_3, T=40", "state_aware", Horizon(3), MarkovDesign(0.3), 40, 0.8, (0.6, 0.9)),
    (
        "plug-in steady state, T=60",
        "state_aware",
        STEADY_STATE,
        MarkovDesign(0.3),
        60,
        0.9,
        (0.8, 0.95),
    ),
    (
        "plug-in steady state, T=120",
        "state_aware",
        STEADY_STATE,
        MarkovDesign(0.3),
        120,
        0.9,
        (0.8, 0.95),
    ),
]
MEMORY = [
    ("tau_2, blocks=12", "block_dim", Horizon(2), BlockDesign(2, 1), 12, 0.95, (0.9, 0.97)),
    ("tau_2, blocks=24", "block_dim", Horizon(2), BlockDesign(2, 1), 24, 0.95, (0.9, 0.97)),
]


def _schedule(rng, design, count: int) -> np.ndarray:
    if isinstance(design, BlockDesign):
        coins = rng.integers(0, 2, (1, count))
        return np.repeat(coins, design.length, axis=1).astype(np.float64)
    path = np.concatenate([rng.random((1, 1)) < 0.5, rng.random((1, count - 1)) < design.flip], 1)
    return (np.cumsum(path, axis=1) % 2).astype(np.float64)


def _first_order(u, a: float, b: float, rng) -> np.ndarray:
    n = u.shape[1]
    lever = np.concatenate([np.zeros((1, BURN)), u], axis=1)
    eps = rng.standard_normal((1, n + BURN))
    x = np.zeros((1, n + BURN + 1))
    for t in range(n + BURN):
        x[:, t + 1] = a * x[:, t] + b * lever[:, t] + eps[:, t]
    return x[:, BURN:]


def _tau(estimand: Horizon, a: float, b: float) -> float:
    h = estimand.periods
    return b / (1.0 - a) if math.isinf(h) else b * (1.0 - a**h) / (1.0 - a)


def _replicate(job) -> dict | None:
    section, case, index = job
    _, analysis, estimand, design, count, a, persistence = case
    logging.disable(logging.WARNING)
    rng = np.random.default_rng([SEED, index])
    b = 0.0 if section == "size" else 1.0
    u = _schedule(rng, design, count)
    y = _first_order(u, a, b, rng)
    blocks = design if analysis == "block_dim" else None
    try:
        reading = read_switchback(u, y, estimand, analysis, blocks=blocks)
    except ValueError:
        return None  # a schedule the reading refuses: no switch, or no usable block
    row: dict = {"asymptotic": reading.interval}
    if section == "size":
        row["p"] = randomisation_test(
            u, y, estimand, analysis, design, draws=999, seed=index
        ).p_value
        return row
    ranges = {"range": persistence, "known": (a, a), "memoryless": (0.0, 0.0)}
    for label, memory in ranges.items():
        try:
            row[label] = randomisation_interval(
                u, y, estimand, analysis, design, memory, draws=999, seed=index
            )
        except ValueError:
            row[label] = None  # the data reject every effect at every memory in the range
    return row


def _rate(hits: int, n: int) -> dict:
    p = hits / n
    return {"rate": round(p, 4), "se": round(math.sqrt(p * (1.0 - p) / n), 4)}


def _summarise(section: str, case, rows: list[dict]) -> dict:
    name, analysis, estimand, _, _, a, persistence = case
    ran = [r for r in rows if r is not None]
    out: dict = {"case": name, "analysis": analysis, "estimand": estimand.name, "a": a}
    out["replicates"], out["refused_by_reading"] = len(ran), len(rows) - len(ran)
    if section == "size":
        out["randomisation"] = _rate(sum(r["p"] <= 0.05 for r in ran), len(ran))
        out["asymptotic"] = _rate(
            sum(not lo <= 0.0 <= hi for lo, hi in (r["asymptotic"] for r in ran)), len(ran)
        )
        return out
    tau = _tau(estimand, a, 1.0)
    out["tau"], out["persistence"] = tau, persistence
    for label in ("asymptotic", "range", "known", "memoryless"):
        intervals = [r[label] for r in ran if r[label] is not None]
        widths = np.array([hi - lo for lo, hi in intervals])
        # a run whose data the test rejects at every effect has an empty set, which misses
        out[label] = {
            "coverage": _rate(sum(lo <= tau <= hi for lo, hi in intervals), len(ran)),
            "median_width": float(np.median(widths)) if len(widths) else None,
            "unbounded": round(float(np.mean(~np.isfinite(widths))), 4) if len(widths) else None,
            "rejects_every_effect": len(ran) - len(intervals),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("section", choices=["size", "coverage", "memory"])
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cases = {"size": SIZE, "coverage": COVERAGE, "memory": MEMORY}[args.section]
    section = "size" if args.section == "size" else "coverage"
    context = multiprocessing.get_context("spawn")
    results = []
    with ProcessPoolExecutor(args.workers, mp_context=context) as pool:
        for case in cases:
            jobs = [(section, case, i) for i in range(args.replicates)]
            rows = list(pool.map(_replicate, jobs, chunksize=8))
            results.append(_summarise(section, case, rows))
            print(json.dumps(results[-1]), flush=True)
    print(json.dumps({"section": args.section, "replicates": args.replicates, "results": results}))


if __name__ == "__main__":
    main()
