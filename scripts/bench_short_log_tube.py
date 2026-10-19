"""The coverage of ``prescribe``'s tube on short one-unit logs (ADR 0104).

For each of ``LOGS`` logs of ``N`` transitions of a plant, seeded 0 on, it runs ``prescribe`` and
then the plant itself, 2,000 times, from the plan's start under the plan's actions, the confounder
and the noise drawn afresh (seed ``10^6 + seed``). It prints, at each step of the plan, the share
of runs inside the tube averaged over the logs and its Monte Carlo standard error over them, and
the share of logs whose certificate trusts a step past the last at which their own plant stays
within the tolerance at 95 %.

Plants: ``ar1`` and ``bilinear`` are ``scripts/bench_tube_levels.py``'s, the bilinear one fitted
under ``rk4``, a model class that does not hold its Euler map; ``ar1two`` is one state and two
levers, ``x' = -0.5 x + u1 + 0.6 u2 + 0.8 z`` under ``u1 = -0.5 x + 0.8 z + 0.5 e1``,
``u2 = 0.3 x - 0.4 z + 0.5 e2``. A log whose fit raises (the ``rk4`` fixed point on a short
bilinear log) is skipped and counted.

usage: python scripts/bench_short_log_tube.py {ar1,ar1two,bilinear} N LOGS
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_tube_levels import AR1, Bilinear  # noqa: E402

from chc import CausalGraph, Lever, Panel, Target, prescribe  # noqa: E402

RUNS = 2000


class AR1Two(AR1):
    """One state, two levers."""

    name = "ar1two"

    def step(self, x, u, z, rng):
        x = x[:, 0]
        rate = -0.5 * x + u[:, 0] + 0.6 * u[:, 1] + 0.8 * z
        return (x + self.dt * rate + 0.05 * rng.normal(size=x.size))[:, None]

    def policy(self, x, z, rng):
        u1 = -0.5 * x[:, 0] + 0.8 * z + 0.5 * rng.normal(size=z.size)
        u2 = 0.3 * x[:, 0] - 0.4 * z + 0.5 * rng.normal(size=z.size)
        return np.stack([u1, u2], 1)

    def episodes(self, units, steps, rng):
        x = self.start(units, rng)
        xs, us, zs = [x], [], []
        for _ in range(steps):
            z = rng.normal(size=units)
            u = self.policy(x, z, rng)
            x = self.step(x, u, z, rng)
            xs.append(x)
            us.append(u)
            zs.append(z)
        return np.stack(xs, 1), np.stack(us, 1), np.stack(zs, 1)

    def ask(self, panel):
        return prescribe(
            panel,
            adjustment=CausalGraph.from_edges(
                [("z", "u1"), ("z", "u2"), ("z", "x"), ("u1", "x"), ("u2", "x")]
            ),
            levers=[Lever("u1", lo=-2.0, hi=2.0), Lever("u2", lo=-2.0, hi=2.0)],
            target=Target("x", value=1.0),
            horizon=self.horizon,
            dt=self.dt,
            tolerance=0.4,
        )


def _frame(plant, xs, us, zs) -> dict[str, np.ndarray]:
    """One unit's transitions, the last state dropped, as the logs keep."""
    steps = us.shape[1]
    levers = ("u",) if us.shape[2] == 1 else ("u1", "u2")
    frame = {"unit": np.zeros(steps, dtype=int), "time": np.arange(steps), "z": zs[0]}
    for index, name in enumerate(levers):
        frame[name] = us[0, :, index]
    for index, name in enumerate(plant.states):
        frame[name] = xs[0, :steps, index]
    return frame


def _runs(plant, x0: np.ndarray, actions: np.ndarray, rng) -> np.ndarray:
    """``(RUNS, H, n)``: the plant run on from ``x0`` under ``actions``."""
    x = np.repeat(x0[None], RUNS, 0)
    path = []
    for action in actions:
        held = np.repeat(np.atleast_1d(action)[None], RUNS, 0)
        x = plant.step(x, held if held.shape[1] > 1 else held[:, 0], rng.normal(size=RUNS), rng)
        path.append(x)
    return np.stack(path, 1)


def main() -> None:
    name, transitions, logs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    plants = {"ar1": AR1, "ar1two": AR1Two, "bilinear": Bilinear}
    plant = plants[name]()
    inside, past, skipped = [], [], 0
    for seed in range(logs):
        xs, us, zs = plant.episodes(1, transitions, np.random.default_rng(seed))
        panel = Panel.from_frame(_frame(plant, xs, us, zs), unit="unit", time="time", seed=0)
        try:
            result = plant.ask(panel)
        except ValueError:
            skipped += 1
            continue
        plan = result.plan
        assert plan is not None
        assert plan.uncertainty_tube is not None
        tube = np.asarray(plan.uncertainty_tube)[1:]
        planned = np.asarray(plan.trajectory)[1:]
        runs = _runs(
            plant,
            np.asarray(result.start.value, dtype=np.float64),
            np.asarray(plan.actions),
            np.random.default_rng(10**6 + seed),
        )
        distance = np.linalg.norm(runs - planned[None], axis=2)
        inside.append(np.mean(distance <= tube[None], axis=0))
        held = np.cumprod(np.quantile(distance, 0.95, axis=0) <= 0.4)
        past.append((result.certificate.certified_horizon or 0) > int(np.sum(held)))
    share = np.asarray(inside)
    print(f"{name}, {transitions} transitions: {len(share)} logs, {skipped} whose fit raised")
    print("inside, by step:", np.round(share.mean(axis=0), 3).tolist())
    error = share.std(axis=0, ddof=1) / math.sqrt(len(share))
    print("its standard error:", np.round(error, 3).tolist())
    print(f"trusted past their plant: {np.mean(past):.3f}")


if __name__ == "__main__":
    main()
