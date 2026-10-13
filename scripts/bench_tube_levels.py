"""The tube's levels, measured on three plants: a joint band over the horizon, and a region for
the channel (ADR 0099). Nothing here moves a number of the tube.

For each log of a plant it runs ``prescribe`` on a training log and reads the plan's tube
``r_1..r_H``, then:

- **joint**: draws ``calibration`` fresh windows of ``H`` steps under the logging policy, each
  from its own unit, and ``tests`` more; calibrates ``HorizonConformal`` on the first against the
  tube's profile, and reads on the second the share of windows held at every step by the band, by
  the tube itself (0.95 a step), and by per-step conformal radii at ``delta / H`` (Bonferroni);
- **region**: fits the channel under Euler on the same log, adjusted for the confounder and, as a
  failure case, for a column of noise in its place, and reads whether ``channel_region`` at 0.9
  holds the plant's channel, and its rate error at a level beside the tube's one-standard-error
  rate error;
- **series** (``ar1`` only): windows cut from one long series, calibrated with the series' beta
  mixing, its ``eta`` and the share of later windows held.

usage: python scripts/bench_tube_levels.py {ar1,bilinear,quickstart} LOGS OUT.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from chc import CausalGraph, Constraint, Lever, Panel, Target, prescribe  # noqa: E402
from chc.decision import _model_error  # noqa: E402
from chc.dynamics import HybridDynamics, LinearDynamics  # noqa: E402
from chc.dynamics_id import channel_region, fit_causal_residual  # noqa: E402
from chc.integrate import rk4_step, rollout  # noqa: E402
from chc.uncertainty import (  # noqa: E402
    HorizonConformal,
    _conformal_rank,
    gaussian_mixing,
    horizon_errors,
)

DELTAS = (0.05, 0.1)
CALIBRATION, TESTS, REGION_LEVEL = 399, 4000, 0.9


class Plant:
    """A plant, its logging policy, and what ``prescribe`` is asked on it."""

    name: str
    states: tuple[str, ...]
    dt: float
    horizon: int
    channel: np.ndarray  # (n, 1, n + 1): the true channel's coefficients on [1, x]

    def step(self, x: np.ndarray, u: np.ndarray, z: np.ndarray, rng) -> np.ndarray:
        raise NotImplementedError

    def policy(self, x: np.ndarray, z: np.ndarray, rng) -> np.ndarray:
        raise NotImplementedError

    def start(self, units: int, rng) -> np.ndarray:
        raise NotImplementedError

    def episodes(self, units: int, steps: int, rng) -> tuple[np.ndarray, ...]:
        """``units`` independent units of ``steps`` transitions: states, actions, confounders."""
        x = self.start(units, rng)
        xs, us, zs = [x], [], []
        for _ in range(steps):
            z = rng.normal(size=units)
            u = self.policy(x, z, rng)
            x = self.step(x, u, z, rng)
            xs.append(x)
            us.append(u)
            zs.append(z)
        return np.stack(xs, 1), np.stack(us, 1)[:, :, None], np.stack(zs, 1)

    def log(self, seed: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def ask(self, panel: Panel):
        raise NotImplementedError


class AR1(Plant):
    name, states, dt, horizon = "ar1", ("x",), 0.2, 8
    channel = np.array([[[1.0, 0.0]]])

    def step(self, x, u, z, rng):
        x = x[:, 0]
        return (x + self.dt * (-0.5 * x + u + 0.8 * z) + 0.05 * rng.normal(size=x.size))[:, None]

    def policy(self, x, z, rng):
        return -0.5 * x[:, 0] + 0.8 * z + 0.5 * rng.normal(size=z.size)

    def start(self, units, rng):
        x = np.zeros((units, 1))
        for _ in range(60):  # to the stationary law under the policy
            z = rng.normal(size=units)
            x = self.step(x, self.policy(x, z, rng), z, rng)
        return x

    def log(self, seed):
        xs, us, zs = self.episodes(1, 1500, np.random.default_rng(seed))
        return _frame(xs, us, zs, self.states)

    def ask(self, panel):
        return prescribe(
            panel,
            adjustment=CausalGraph.from_edges([("z", "u"), ("z", "x"), ("u", "x")]),
            levers=[Lever("u", lo=-2.0, hi=2.0)],
            target=Target("x", value=1.0),
            horizon=self.horizon,
            dt=self.dt,
            tolerance=0.4,
        )


class Bilinear(Plant):
    """``x1' = -0.4 x1 + 0.2 x2 + (0.8 + 0.3 x1) u + 0.2 z``,
    ``x2' = -0.3 x2 + (0.5 - 0.2 x2) u + 0.1 z``, under ``u = 0.5 z - 0.3 x1 + 0.4 e``."""

    name, states, dt, horizon = "bilinear", ("x1", "x2"), 0.5, 12
    channel = np.array([[[0.8, 0.3, 0.0]], [[0.5, 0.0, -0.2]]])

    def step(self, x, u, z, rng):
        x1, x2 = x[:, 0], x[:, 1]
        rate1 = -0.4 * x1 + 0.2 * x2 + (0.8 + 0.3 * x1) * u + 0.2 * z
        rate2 = -0.3 * x2 + (0.5 - 0.2 * x2) * u + 0.1 * z
        noise = 0.1 * rng.normal(size=x.shape)
        return x + self.dt * np.stack([rate1, rate2], 1) + noise

    def policy(self, x, z, rng):
        return 0.5 * z - 0.3 * x[:, 0] + 0.4 * rng.normal(size=z.size)

    def start(self, units, rng):
        x = np.zeros((units, 2))
        for _ in range(60):
            z = rng.normal(size=units)
            x = self.step(x, self.policy(x, z, rng), z, rng)
        return x

    def log(self, seed):
        xs, us, zs = self.episodes(1, 960, np.random.default_rng(seed))
        return _frame(xs, us, zs, self.states)

    def ask(self, panel):
        edges = [("z", "u"), ("z", "x1"), ("z", "x2"), ("u", "x1"), ("u", "x2"), ("x2", "x1")]
        return prescribe(
            panel,
            adjustment=CausalGraph.from_edges(edges),
            levers=[Lever("u", lo=-2.0, hi=2.0)],
            target=Target("x1", value=1.0),
            constraints=[Constraint("x2", hi=10.0)],  # a loose bound that makes x2 a state
            horizon=self.horizon,
            dt=self.dt,
            tolerance=0.4,
        )


class Quickstart(Plant):
    """The quickstart's two-state plant: supply and wait under an incentive set from demand."""

    name, states, dt, horizon = "quickstart", ("supply", "wait"), 0.1, 15
    channel = np.array([[[0.8, 0.0, 0.0]], [[-0.4, 0.0, 0.0]]])

    def step(self, x, u, z, rng):
        s, w = x[:, 0], x[:, 1]
        noise = rng.normal(0.0, 0.01, size=x.shape)
        rate = np.stack([-0.6 * s + 0.3 * w + 0.8 * u + 1.5 * z, 0.25 * w - 0.4 * u], 1)
        return x + self.dt * rate + noise

    def policy(self, x, z, rng):
        return 0.9 * z + rng.normal(0.0, 0.5, size=z.size)

    def start(self, units, rng):
        return rng.normal(0.0, 0.2, size=(units, 2))

    def log(self, seed):
        xs, us, zs = self.episodes(200, 12, np.random.default_rng(seed))
        return _frame(xs, us, zs, self.states, confounder="demand", lever="incentive")

    def ask(self, panel):
        edges = [
            ("demand", "incentive"),
            ("demand", "supply"),
            ("incentive", "supply"),
            ("incentive", "wait"),
            ("wait", "supply"),
        ]
        return prescribe(
            panel,
            adjustment=CausalGraph.from_edges(edges),
            levers=[Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05)],
            target=Target("supply", value=1.0),
            constraints=[Constraint("wait", hi=0.5)],
            horizon=self.horizon,
            dt=self.dt,
            tolerance=0.5,
        )


def _frame(xs, us, zs, states, *, confounder="z", lever="u") -> dict[str, np.ndarray]:
    """A panel's columns: each unit's transitions, the last state dropped, as the logs keep."""
    units, steps = us.shape[0], us.shape[1]
    frame = {
        "unit": np.repeat(np.arange(units), steps),
        "time": np.tile(np.arange(steps), units),
        lever: us[:, :, 0].ravel(),
        confounder: zs.ravel(),
    }
    for index, name in enumerate(states):
        frame[name] = xs[:, :steps, index].ravel()
    return frame


def _transitions(xs, us, zs, rng) -> dict[str, jnp.ndarray]:
    """The Euler fit's columns: every unit's transitions, with a column of noise as a decoy."""
    x = xs[:, :-1].reshape(-1, xs.shape[2])
    return {
        "x": jnp.asarray(x),
        "u": jnp.asarray(us.reshape(-1, 1)),
        "x_next": jnp.asarray(xs[:, 1:].reshape(-1, xs.shape[2])),
        "z": jnp.asarray(zs.reshape(-1, 1)),
        "decoy": jnp.asarray(rng.normal(size=(x.shape[0], 1))),
    }


def _open_loop_errors(model, xs: np.ndarray, us: np.ndarray, dt: float) -> np.ndarray:
    """``(units, H, n)``: each unit's one window, batched; checked against ``horizon_errors``."""

    def one(x0, window):
        def body(x, u):
            nxt = rk4_step(model, 0.0, x, u, dt)
            return nxt, nxt

        return jax.lax.scan(body, x0, window)[1]

    predicted = jax.jit(jax.vmap(one))(jnp.asarray(xs[:, 0]), jnp.asarray(us))
    return xs[:, 1:] - np.asarray(predicted)


def _bonferroni(calibration: np.ndarray, delta: float) -> np.ndarray | None:
    """Per-step conformal radii at ``delta / H``, or None where too few windows certify it."""
    norms = np.linalg.norm(calibration, axis=2)
    n, horizon = norms.shape
    rank = _conformal_rank(n, delta / horizon)
    if rank > n:
        return None
    return np.sort(norms, axis=0)[rank - 1]


def run(plant: Plant, logs: int, out: str) -> None:
    rows = []
    for seed in range(logs):
        began = time.time()
        rng = np.random.default_rng(10_000 + seed)
        data = plant.log(seed)
        panel = Panel.from_frame(data, unit="unit", time="time", seed=0)
        result = plant.ask(panel)
        plan, fit = result.plan, result.model_fit
        assert plan is not None
        assert result.start is not None
        assert tuple(result.start.states) == plant.states
        model = HybridDynamics(
            known=LinearDynamics(
                jnp.zeros((len(plant.states),) * 2), jnp.zeros((len(plant.states), 1))
            ),
            residual=fit.residual,
        )
        replayed = rollout(model, jnp.asarray(result.start.value), plan.actions, plant.dt)
        assert np.allclose(np.asarray(replayed), np.asarray(plan.trajectory), atol=1e-9)
        tube = np.asarray(plan.uncertainty_tube, dtype=np.float64)
        profile = tube[1 : plant.horizon + 1]
        row: dict[str, object] = {"seed": seed, "tube": profile.tolist()}

        xs, us, _ = plant.episodes(CALIBRATION + TESTS, plant.horizon, rng)
        errors = _open_loop_errors(model, xs, us, plant.dt)
        for unit in range(3):
            np.testing.assert_allclose(
                errors[unit : unit + 1],
                horizon_errors(model, xs[unit], us[unit], plant.dt, plant.horizon),
                rtol=1e-9,
                atol=1e-12,
            )
        calibration, test = errors[:CALIBRATION], errors[CALIBRATION:]
        norms = np.linalg.norm(test, axis=2)
        row["tube_joint"] = float(np.mean(np.all(norms <= profile, axis=1)))
        row["tube_step"] = np.mean(norms <= profile, axis=0).tolist()
        for delta in DELTAS:
            band = HorizonConformal.calibrate(calibration, profile.tolist(), delta)
            row[f"joint_{delta}"] = float(np.mean(band.covers(test)))
            row[f"scale_{delta}"] = band.scale
            radii = _bonferroni(calibration, delta)
            row[f"bonferroni_{delta}"] = (
                None if radii is None else float(np.mean(np.all(norms <= radii, axis=1)))
            )
            row[f"bonferroni_end_{delta}"] = None if radii is None else float(radii[-1])
            row[f"band_end_{delta}"] = band.band[-1]

        xs, us, zs = plant.episodes(*_log_shape(plant), np.random.default_rng(seed))
        columns = _transitions(xs, us, zs, rng)
        reach = np.array([max(abs(-2.0 - float(np.mean(us))), abs(2.0 - float(np.mean(us))))])
        for adjust in ("z", "decoy"):
            euler = fit_causal_residual(
                LinearDynamics(
                    jnp.zeros((len(plant.states),) * 2), jnp.zeros((len(plant.states), 1))
                ),
                columns,
                plant.dt,
                adjust_for=(adjust,),
                integrator="euler",
            )
            region = channel_region(euler, REGION_LEVEL)
            row[f"region_{adjust}"] = region.contains(plant.channel)
            if adjust == "z":
                centre = np.mean(np.asarray(columns["x"]), axis=0)
                row["rate_error_level"] = region.rate_error(centre[None], reach)
                row["rate_error_level_max"] = region.rate_error(np.asarray(columns["x"]), reach)
                row["rate_error_se"] = _model_error(euler, reach)
                row["operator_radius"] = region.operator_radius(centre)
                row["channel_error"] = euler.channel_error
        if isinstance(plant, AR1):
            row |= _series(plant, model, seed)
        row["seconds"] = time.time() - began
        rows.append(row)
        print(json.dumps({key: _short(value) for key, value in row.items() if key != "tube"}))
        sys.stdout.flush()
    summary = _summary(rows)
    with Path(out).open("w") as handle:
        json.dump({"plant": plant.name, "rows": rows, "summary": summary}, handle, indent=1)
    print(json.dumps(summary, indent=1))


def _log_shape(plant: Plant) -> tuple[int, int]:
    """The training log's units and transitions, as ``log`` draws them."""
    return {"ar1": (1, 1500), "bilinear": (1, 960), "quickstart": (200, 12)}[plant.name]


def _series(plant: AR1, model, seed: int) -> dict[str, object]:
    """Windows cut from one stationary series, ``stride = H``: calibrated on the first 20,000
    with the series' beta mixing, tested on the next 20,000. The window's errors are read off the
    states ``x_t..x_{t+H}`` and the actions and confounders between, all set by the chain
    ``x_{t+1} = (1 - dt) x_t + innovation``, so the windows' lag ``j`` is the chain's
    ``j H - H``."""
    rng = np.random.default_rng(90_000 + seed)
    windows = 20_000
    xs, us, _ = plant.episodes(1, 2 * windows * plant.horizon, rng)
    errors = horizon_errors(model, xs[0], us[0], plant.dt, plant.horizon)
    profile = np.ones(plant.horizon)
    innovation = plant.dt**2 * (1.6**2 + 0.5**2) + 0.05**2
    beta = gaussian_mixing(
        [[1.0 - plant.dt]], [[innovation]], stride=plant.horizon, span=plant.horizon
    )
    found: dict[str, object] = {}
    for delta in (0.1, 0.2):
        band = HorizonConformal.calibrate(errors[:windows], profile, delta, mixing=beta)
        found[f"series_eta_{delta}"] = band.eta
        found[f"series_level_{delta}"] = band.level
        found[f"series_joint_{delta}"] = float(np.mean(band.covers(errors[windows:])))
    return found


def _short(value: object) -> object:
    if isinstance(value, float):
        return round(value, 5)
    if isinstance(value, list):
        return [_short(item) for item in value][:4]
    return value


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    """The mean of each number over the logs, with its standard error across them."""
    summary: dict[str, object] = {"logs": len(rows)}
    for key in rows[0]:
        values = [row[key] for row in rows]
        if key in ("seed", "tube", "seconds") or any(value is None for value in values):
            continue
        array = np.asarray(values, dtype=np.float64)
        mean = np.mean(array, axis=0)
        se = np.std(array, axis=0, ddof=1) / math.sqrt(len(rows)) if len(rows) > 1 else 0.0
        summary[key] = {"mean": np.round(mean, 5).tolist(), "se": np.round(se, 5).tolist()}
    return summary


if __name__ == "__main__":
    plants = {"ar1": AR1, "bilinear": Bilinear, "quickstart": Quickstart}
    run(plants[sys.argv[1]](), int(sys.argv[2]), sys.argv[3])
