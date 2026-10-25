"""A two-state plant whose logs can break, one at a time, a premise the decision's model makes.

The plant is the quickstart's: supply ``s`` and wait ``w`` under an incentive ``u`` that the log
sets from demand ``z``, a confounder the panel records,

    s' = -0.6 s + 0.3 w + 0.8 g(u) + 1.5 z + d,    w' = 0.25 w - 0.4 g(u),

stepped by Euler at ``dt = 0.1`` with a small state noise. Each field of :class:`World` breaks one
premise: demand that persists from period to period, an unlogged disturbance ``d`` that does, an
action that reaches the plant some periods late, a response ``g`` that saturates where the fitted
channel is linear, a drift outside any polynomial, and a confounder the panel does not record. At
its defaults every premise holds.
"""

from __future__ import annotations

import functools
import logging
import math
from dataclasses import dataclass

import numpy as np

from chc import CausalGraph, Constraint, Lever, Panel, Target, prescribe
from chc.decision import Prescription

DT, HORIZON = 0.1, 15
STATES = ("supply", "wait")
EDGES = [
    ("demand", "incentive"),
    ("demand", "supply"),
    ("incentive", "supply"),
    ("incentive", "wait"),
    ("wait", "supply"),
]


@dataclass(frozen=True)
class World:
    """One plant and its logging rule; the defaults are the well-specified case."""

    demand_persistence: float = 0.0  # AR(1) coefficient of z within a unit
    disturbance: float = 0.0  # stationary sd of the unlogged disturbance d on supply's rate
    disturbance_persistence: float = 0.0  # its AR(1) coefficient
    dead_time: int = 0  # periods before an action reaches the plant
    saturation: float | None = None  # g(u) = saturation * tanh(u / saturation); None: g(u) = u
    sine_drift: bool = False  # supply's own drift -0.3 sin(2 s) in place of -0.6 s
    logged_gain: float = 0.9  # the log's incentive: logged_gain * z + logged_spread * noise
    logged_spread: float = 0.5
    # an unlogged confounder q ~ N(0, 1) the rule adds 0.5 q for and supply's rate -hidden q, which
    # moves the channel the log reads by -hidden
    hidden: float = 0.0

    def response(self, u: np.ndarray) -> np.ndarray:
        if self.saturation is None:
            return u
        return self.saturation * np.tanh(u / self.saturation)

    def rule(self, z: np.ndarray, q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return self.logged_gain * z + 0.5 * q + self.logged_spread * rng.normal(size=z.size)

    def step(
        self,
        x: np.ndarray,
        applied: np.ndarray,
        z: np.ndarray,
        d: np.ndarray,
        rng: np.random.Generator | None,
        q: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        """One period; ``rng`` None steps the mean path, with no state noise."""
        s, w = x[:, 0], x[:, 1]
        own = -0.3 * np.sin(2.0 * s) if self.sine_drift else -0.6 * s
        g = self.response(applied)
        rate = np.stack(
            [own + 0.3 * w + 0.8 * g + 1.5 * z + d - self.hidden * q, 0.25 * w - 0.4 * g], 1
        )
        noise = 0.0 if rng is None else rng.normal(0.0, 0.01, size=x.shape)
        return x + DT * rate + noise

    def _shocks(self, units: int, steps: int, rng) -> tuple[np.ndarray, np.ndarray]:
        """Demand and the disturbance, ``(units, steps)`` each, stationary from the first."""
        z = np.empty((units, steps))
        d = np.empty((units, steps))
        a, b = self.demand_persistence, self.disturbance_persistence
        z[:, 0] = rng.normal(size=units)
        d[:, 0] = self.disturbance * rng.normal(size=units)
        for t in range(1, steps):
            z[:, t] = a * z[:, t - 1] + math.sqrt(1.0 - a * a) * rng.normal(size=units)
            d[:, t] = b * d[:, t - 1] + self.disturbance * math.sqrt(1.0 - b * b) * rng.normal(
                size=units
            )
        return z, d

    def log(self, seed: int, units: int = 200, steps: int = 12) -> dict[str, np.ndarray]:
        """A panel's columns: ``units`` units of ``steps`` periods under the logging rule."""
        rng = np.random.default_rng(seed)
        z, d = self._shocks(units, steps + self.dead_time, rng)
        x = rng.normal(0.0, 0.2, size=(units, 2))
        pending = [np.zeros(units)] * self.dead_time
        xs, us = [], []
        for t in range(steps + self.dead_time):
            q = rng.normal(size=units) if self.hidden else np.zeros(units)
            u = self.rule(z[:, t], q, rng)
            pending.append(u)
            applied = pending.pop(0)
            if t >= self.dead_time:
                xs.append(x)
                us.append(u)
            x = self.step(x, applied, z[:, t], d[:, t], rng, q)
        zs = z[:, self.dead_time :]
        frame = {
            "unit": np.repeat(np.arange(units), steps),
            "time": np.tile(np.arange(steps), units),
            "incentive": np.stack(us, 1).ravel(),
            "demand": zs.ravel(),
        }
        states = np.stack(xs, 1)
        for index, name in enumerate(STATES):
            frame[name] = states[:, :, index].ravel()
        return frame

    def deploy(self, actions: np.ndarray, x0: np.ndarray, episodes: int, seed: int) -> np.ndarray:
        """``(episodes, H + 1, 2)``: the plant's paths under the plan's actions from ``x0``, each
        with its own demand and disturbance, the actions before the plan's first held at zero."""
        rng = np.random.default_rng(seed)
        horizon = actions.shape[0]
        z, d = self._shocks(episodes, horizon, rng)
        x = np.tile(np.asarray(x0, dtype=float), (episodes, 1))
        pending = [np.zeros(episodes)] * self.dead_time
        paths = [x]
        for k in range(horizon):
            pending.append(np.full(episodes, actions[k]))
            q = rng.normal(size=episodes) if self.hidden else 0.0
            x = self.step(x, pending.pop(0), z[:, k], d[:, k], rng, q)
            paths.append(x)
        return np.stack(paths, 1)

    def mean_path(self, actions: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """``(H + 1, 2)``: the plant's path under the plan's actions from ``x0`` with demand, the
        disturbance and the state noise at their mean, zero."""
        x = np.asarray(x0, dtype=float)[None]
        zero = np.zeros(1)
        pending = [zero] * self.dead_time
        path = [x[0]]
        for action in actions:
            pending.append(np.full(1, action))
            x = self.step(x, pending.pop(0), zero, zero, None)
            path.append(x[0])
        return np.stack(path)


def ask(
    world: World,
    seed: int,
    *,
    box: float = 2.0,
    constraints: tuple[Constraint, ...] = (Constraint("wait", hi=0.5),),
    integrator: str = "rk4",
    **options: object,
) -> Prescription:
    """``prescribe`` on one log of ``world``: steer supply to 1 from rest, by default with wait at
    most 0.5, a bound the plan never comes near. Fitted and planned by ``"rk4"`` unless asked
    otherwise: the suite's floors were measured under it, before ``prescribe``'s default became
    ``"euler"``."""
    panel = Panel.from_frame(world.log(seed), unit="unit", time="time", seed=0)
    return prescribe(
        panel,
        adjustment=CausalGraph.from_edges(EDGES),
        levers=[Lever("incentive", lo=-box, hi=box, unit_cost=0.05)],
        target=Target("supply", value=1.0),
        constraints=constraints,
        horizon=HORIZON,
        dt=DT,
        tolerance=0.5,
        x0=np.zeros(2),
        integrator=integrator,  # type: ignore[arg-type]
        **options,  # type: ignore[arg-type]
    )


class _TubeWarnings(logging.Handler):
    """Counts the ``WARNING`` records that say the tube's premise fails (``chc_event="tube"``)."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.count += getattr(record, "chc_event", None) == "tube"


@functools.cache
def tube_coverage(
    world: World, seeds: range, episodes: int
) -> tuple[np.ndarray, tuple[int, ...], tuple[bool, ...]]:
    """Per step, the share of the plant's paths the plan's tube holds, pooled over one log per
    seed, steps 1 to H; each log's trustworthy steps; and whether a ``tube`` warning was logged on
    each. Cached: two tests read one run, and the warnings are counted here for that reason."""
    held, trusted, warned = [], [], []
    logger = logging.getLogger("chc.decision")
    for seed in seeds:
        handler = _TubeWarnings()
        logger.addHandler(handler)
        try:
            result = ask(world, seed)
        finally:
            logger.removeHandler(handler)
        warned.append(handler.count > 0)
        assert result.plan is not None
        assert result.plan.uncertainty_tube is not None
        tube = np.asarray(result.plan.uncertainty_tube, dtype=float)[1:]
        planned = np.asarray(result.plan.trajectory, dtype=float)
        paths = world.deploy(
            np.asarray(result.plan.actions, dtype=float)[:, 0], planned[0], episodes, 1000 + seed
        )
        gaps = np.linalg.norm(paths[:, 1:] - planned[None, 1:], axis=2)
        held.append(np.mean(gaps <= tube, axis=0))
        trusted.append(result.certificate.trustworthy_steps)
    return np.mean(held, axis=0), tuple(trusted), tuple(warned)
