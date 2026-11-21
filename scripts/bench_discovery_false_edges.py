"""The false-edge rates chc.discovery.discover_lagged_parents' docstring quotes. Rates only, no
wall time.

Three states and one control over three lags, so 36 candidate edges, of which 5 are parents:

    x0' = 0.6 x0 + 0.5 u + e0,    x1' = 0.5 x1 + 0.4 x0 + e1,    x2' = 0.7 x2 + e2,

with ``u`` and each ``e`` independent standard normals at every step, from a start at zero. Each
graph is discovered at ``alpha=0.01`` from one series, and is scored for whether it holds at least
one of the 31 edges that are not parents, and for the share of those 31 it holds.

Run: uv run python scripts/bench_discovery_false_edges.py > out.json
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from chc.discovery import discover_lagged_parents

# (steps, seed): the series' length and the seed its graphs were first measured with
CASES = ((300, 57), (1000, 58))


def series(rng: np.random.Generator, steps: int) -> tuple[np.ndarray, np.ndarray]:
    u = rng.normal(size=(steps, 1))
    x = np.zeros((steps, 3))
    for t in range(1, steps):
        x[t, 0] = 0.6 * x[t - 1, 0] + 0.5 * u[t - 1, 0] + rng.normal()
        x[t, 1] = 0.5 * x[t - 1, 1] + 0.4 * x[t - 1, 0] + rng.normal()
        x[t, 2] = 0.7 * x[t - 1, 2] + rng.normal()
    return x, u


def rates(steps: int, graphs: int, seed: int) -> dict[str, float | int]:
    rng = np.random.default_rng([seed, 32])
    state = np.zeros((3, 3, 3), dtype=bool)
    state[0, 0, 0] = state[1, 1, 0] = state[1, 0, 0] = state[2, 2, 0] = True
    control = np.zeros((3, 1, 3), dtype=bool)
    control[0, 0, 0] = True
    candidates = int(np.sum(~state) + np.sum(~control))
    held = []
    for _ in range(graphs):
        graph = discover_lagged_parents(*series(rng, steps), max_lag=3, alpha=0.01)
        held.append(
            int(
                np.sum(np.asarray(graph.state_parents) & ~state)
                + np.sum(np.asarray(graph.control_parents) & ~control)
            )
        )
    false = np.asarray(held)
    share = false / candidates
    any_false = float(np.mean(false > 0))
    return {
        "steps": steps,
        "graphs": graphs,
        "seed": seed,
        "candidates_not_parents": candidates,
        "any_false_edge": any_false,
        "any_false_edge_se": math.sqrt(any_false * (1.0 - any_false) / graphs),
        "share_entered": float(share.mean()),
        "share_entered_se": float(share.std(ddof=1) / math.sqrt(graphs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graphs", type=int, default=500)
    graphs = parser.parse_args().graphs
    print(json.dumps([rates(steps, graphs, seed) for steps, seed in CASES], indent=2))


if __name__ == "__main__":
    main()
