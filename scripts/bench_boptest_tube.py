"""The scale of the BOPTEST case study's error tube, as the library on the path reads it.

The case study's live runs read the adjusted certificate's tube one step ahead at 0.29 K, under
the library at `7796c69`, and at 0.31 K in the rerun, under `f3f9eae`. Their logs were not kept, so
they cannot be refitted. This script makes one-unit logs shaped like theirs and reads, on each, the
tube one step ahead of the call the case made. Run it once under each library, its `src` first on
`PYTHONPATH`, then compare two runs log by log with `ratio`. Ratios and counts only, no wall time.

    logs    960 half-hour rows of one unit, as the case's: the zone in Celsius, the heat pump's
            modulation in [0, 1], the outdoor air in Celsius, the sun in kW/m2 and the lower
            comfort bound, 21 C on weekdays before 7 and from 20 and all weekend, 15 C otherwise.
            The modulation is the case's logging policy: 0.15 + 0.10 (bound + 1 - zone)
            + 0.05 (5 - outdoor) plus a draw N(0, 0.12^2) held for 4 steps, clipped to the box.
            The outdoor air is 4 C, a daily swing of 2.5 K and an AR(1) of spread 3 K whose
            correlation falls by e over 36 hours; the sun a half-sine from 8:30 to 16:30 that
            peaks at 0.25 times a day's clearness, uniform on [0.15, 1]. The zone loses
            0.03 (zone - outdoor) K/h and gains 2 K/h per kW/m2 of sun and 0.1 K/h when occupied.
            `one`: the heat pump adds (1.2 - 0.025 zone) K/h per unit of modulation. `slab`: it
            heats a floor, the floor the zone, as the case's floor heating does. The rate noise
            has spread 0.18 K/h and is AR(rho) from one half-hour to the next. Six substeps a
            half-hour. Cells: --cell kind,rho, or the defaults.
    first   The first run's call: the adjusted arm's graph, the box at no cost, horizon 16,
            dt 0.5 h, tolerance 0.5, from the log's last zone toward 21 C.
    rerun   The rerun's call on its held forecast: the same, the band's upper edge 24 C held,
            the outdoor air and the sun as drivers in the drift, 21 C at every step.
    ratio   Two runs' tubes one step ahead, log by log, the first over the second, per cell.

Run: JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run python scripts/bench_boptest_tube.py first > new.json
     PYTHONPATH=<a checkout of 7796c69>/src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
         python scripts/bench_boptest_tube.py first > old.json
     python scripts/bench_boptest_tube.py ratio new.json old.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROWS, DT, SUBSTEPS = 960, 0.5, 6
CELLS = ("one,0.0", "one,0.5", "one,0.9", "one,0.99", "slab,0.0", "slab,0.9")
EDGES = (
    ("outdoor", "modulation"),
    ("outdoor", "zone"),
    ("solar", "zone"),
    ("bound", "modulation"),
    ("occupancy", "bound"),
    ("occupancy", "zone"),
    ("modulation", "zone"),
)


def weather(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(ROWS + 1)
    hour = (t % 48) / 2.0
    day = t // 48 + int(rng.integers(0, 7))
    synoptic = np.empty(ROWS + 1)
    phi = np.exp(-DT / 36.0)
    synoptic[0] = rng.normal(0.0, 3.0)
    for i in range(1, ROWS + 1):
        synoptic[i] = phi * synoptic[i - 1] + rng.normal(0.0, 3.0 * np.sqrt(1 - phi**2))
    outdoor = 4.0 + 2.5 * np.sin(2 * np.pi * (hour - 9.0) / 24.0) + synoptic
    clearness = rng.uniform(0.15, 1.0, size=day.max() + 1)[day]
    solar = 0.25 * clearness * np.clip(np.sin(np.pi * (hour - 8.5) / 8.0), 0.0, None)
    solar[(hour < 8.5) | (hour > 16.5)] = 0.0
    occupied = ((day % 7) >= 5) | (hour < 7.0) | (hour >= 20.0)
    return outdoor, solar, np.where(occupied, 21.0, 15.0)


def make_log(rng: np.random.Generator, kind: str, rho: float) -> dict[str, np.ndarray]:
    outdoor, solar, bound = weather(rng)
    occupied = (bound > 18.0).astype(float)
    zone, floor, noise, level = 19.0, 22.0, rng.normal(0.0, 0.18), 0.0
    rows: dict[str, list[float]] = {name: [] for name in ("zone", "modulation")}
    for i in range(ROWS):
        if i % 4 == 0:
            level = rng.normal(0.0, 0.12)
        raw = 0.15 + 0.10 * (bound[i] + 1.0 - zone) + 0.05 * (5.0 - outdoor[i]) + level
        u = float(np.clip(raw, 0.0, 1.0))
        rows["zone"].append(zone)
        rows["modulation"].append(u)
        noise = rho * noise + np.sqrt(1.0 - rho**2) * rng.normal(0.0, 0.18)
        for _ in range(SUBSTEPS):
            gains = -0.03 * (zone - outdoor[i]) + 2.0 * solar[i] + 0.1 * occupied[i]
            if kind == "one":
                zone += DT / SUBSTEPS * (gains + (1.2 - 0.025 * zone) * u + noise)
            else:
                to_floor = 2.4 * (2.0 - 0.05 * floor) * u - 0.6 * (floor - zone)
                zone, floor = (
                    zone + DT / SUBSTEPS * (gains + 0.25 * (floor - zone) + noise),
                    floor + DT / SUBSTEPS * to_floor,
                )
    return {
        "zone": np.asarray(rows["zone"]),
        "modulation": np.asarray(rows["modulation"]),
        "outdoor": outdoor[:ROWS],
        "solar": solar[:ROWS],
        "bound": bound[:ROWS],
    }


def read(log: dict[str, np.ndarray], call: str, seed: int) -> dict[str, Any]:
    import jax.numpy as jnp

    from chc import CausalGraph, Lever, Panel, Target, prescribe

    panel = Panel.from_frame(
        {"log": np.zeros(ROWS, dtype=np.int64), "step": np.arange(ROWS), **log},
        unit="log",
        time="step",
        seed=seed,
    )
    common: dict[str, Any] = {
        "levers": [Lever("modulation", 0.0, 1.0, unit_cost=0.0)],
        "horizon": 16,
        "adjustment": CausalGraph.from_edges(EDGES, latent=("occupancy",)),
        "dt": DT,
        "tolerance": 0.5,
        "x0": jnp.asarray([float(log["zone"][-1])]),
    }
    if call == "first":
        decided = prescribe(panel, target=Target("zone", value=21.0), **common)
    else:
        from chc import Constraint, Driver

        held = np.ones(17)
        decided = prescribe(
            panel,
            target=Target("zone", value=21.0 * held[1:]),
            constraints=[Constraint("zone", hi=24.0)],
            hold_constraints=True,
            drivers=[
                Driver("outdoor", float(log["outdoor"][-1]) * held),
                Driver("solar", float(log["solar"][-1]) * held),
            ],
            **common,
        )
    tube = decided.plan.uncertainty_tube
    return {
        "tube": None if tube is None else float(tube[1]),
        "channel_error": decided.model_fit.channel_error,
        "cosines": getattr(decided.model_fit, "error_cosines", None),
        "trusted": int(decided.certificate.trustworthy_steps),
    }


def run(call: str, cells: list[str], logs: int, seed: int) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for index, cell in enumerate(cells):
        kind, rho = cell.split(",")
        children = np.random.SeedSequence([seed, index]).spawn(logs)
        out[cell] = [
            read(make_log(np.random.default_rng(child), kind, float(rho)), call, k)
            for k, child in enumerate(children)
        ]
    return out


def ratio(top: Path, bottom: Path) -> dict[str, dict[str, float]]:
    over, under = json.loads(top.read_text()), json.loads(bottom.read_text())
    summary: dict[str, dict[str, float]] = {}
    pooled: list[np.ndarray] = []
    for cell in over:
        ratios = np.array(
            [a["tube"] / b["tube"] for a, b in zip(over[cell], under[cell], strict=True)]
        )
        pooled.append(ratios)
        summary[cell] = _quantiles(ratios)
    summary["pooled"] = _quantiles(np.concatenate(pooled))
    return summary


def _quantiles(values: np.ndarray) -> dict[str, float]:
    low, median, high = np.percentile(values, [5, 50, 95])
    return {
        "mean": float(values.mean()),
        "p5": float(low),
        "median": float(median),
        "p95": float(high),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("first", "rerun", "ratio"))
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--cell", action="append")
    parser.add_argument("--logs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    if args.mode == "ratio":
        print(json.dumps(ratio(*args.files), indent=1))
    else:
        print(json.dumps(run(args.mode, args.cell or list(CELLS), args.logs, args.seed)))


if __name__ == "__main__":
    main()
