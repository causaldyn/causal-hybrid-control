"""The numbers behind docs/adr/0004-greedy-lever-selection.md. Costs and counts only, no wall time.

    instances    ``prescribe(max_levers=k)`` against exhaustive search at every set size, on three
                 plants: the driver pool of tests/test_lever_selection.py, the marketing-mix case
                 study with spend allowed down to zero, and a plant built so that greedy misses.
                 Each with the group-L1 path's order of entry, and the solves greedy spent.
    random       the same comparison on random two-state plants with four levers.
    fingerprint  digests of four prescriptions made without ``max_levers``. Run once here and once
                 with ``PYTHONPATH=<a checkout of main>/src``: equal digests mean equal bytes.

Exhaustive search plans each subset through the public ``prescribe``, the other levers' boxes
pinned to ``[0, 0]``; it shares nothing with the selection but the solve both call, so a cost it
finds is comparable to the bit. The group-L1 path is this script's own and ships nowhere: an
accelerated proximal gradient on ``J(U) + lambda * sum_j ||U[:, j]||``, the box kept exact by a
prox solved per column, on the planning objective rebuilt from the prescription -- rebuilt, and
asserted to reproduce the prescription's plan to the bit, so a rebuild that drifts from the library
stops the run instead of reporting on a different problem.

Run: uv run python scripts/bench_max_levers.py {instances,random,fingerprint} > out.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import math
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

from chc import (  # noqa: E402
    CausalGraph,
    Constraint,
    HybridDynamics,
    Lever,
    LinearConstraint,
    LinearDynamics,
    MarketingMixSystem,
    Panel,
    Prescription,
    QuadraticCost,
    Target,
    causal_plan,
    prescribe,
    total_cost,
)
from chc.mmm import SALES, adstock_dynamics  # noqa: E402

# ── instances ─────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Instance:
    """One decision: a panel and everything ``prescribe`` is called with but the levers' boxes."""

    label: str
    panel: Panel
    adjustment: CausalGraph
    levers: tuple[Lever, ...]
    target: Target
    constraints: tuple[Constraint, ...]
    horizon: int
    dt: float
    known: LinearDynamics | None = None
    x0: tuple[float, ...] | None = None

    def prescribe(
        self, keep: tuple[int, ...] | None, max_levers: int | None = None
    ) -> Prescription:
        """All levers, or only ``keep`` with the others pinned to ``[0, 0]``."""
        levers = [
            lever
            if keep is None or index in keep
            else Lever(lever.name, 0.0, 0.0, lever.unit_cost, lever.cap_per_step)
            for index, lever in enumerate(self.levers)
        ]
        return prescribe(
            self.panel,
            levers=levers,
            target=self.target,
            constraints=self.constraints,
            adjustment=self.adjustment,
            horizon=self.horizon,
            dt=self.dt,
            known=self.known,
            x0=None if self.x0 is None else jnp.asarray(self.x0),
            max_levers=max_levers,
        )


def driver_pool() -> Instance:
    """tests/test_lever_selection.py's plant, at its size: one confounded lever and two clean."""
    dt, rng = 0.1, np.random.default_rng(0)
    names = ("unit", "time", "supply", "wait", "demand", "incentive", "boost", "calm")
    rows: dict[str, list[float]] = {name: [] for name in names}
    for unit in range(120):
        supply, wait = rng.normal(0.0, 0.2), rng.normal(0.0, 0.2)
        for period in range(10):
            demand = rng.normal(0.0, 1.0)
            incentive = 0.9 * demand + rng.normal(0.0, 0.5)
            boost, calm = rng.normal(0.0, 0.5, 2)
            for name, value in zip(
                names, (unit, period, supply, wait, demand, incentive, boost, calm), strict=True
            ):
                rows[name].append(value)
            supply_next = supply + dt * (
                -0.6 * supply + 0.3 * wait + 0.8 * incentive + 1.2 * boost + 1.5 * demand
            )
            wait_next = wait + dt * (0.25 * wait - 0.4 * incentive - 0.6 * calm)
            supply = supply_next + rng.normal(0.0, 0.01)
            wait = wait_next + rng.normal(0.0, 0.01)
    edges = [
        ("demand", "incentive"),
        ("demand", "supply"),
        ("incentive", "supply"),
        ("incentive", "wait"),
        ("supply", "wait"),
        ("boost", "supply"),
        ("calm", "wait"),
    ]
    return Instance(
        label="driver pool (tests/test_lever_selection.py), H=12",
        panel=Panel.from_frame(
            {k: np.asarray(v) for k, v in rows.items()}, unit="unit", time="time", seed=0
        ),
        adjustment=CausalGraph.from_edges(edges),
        levers=tuple(Lever(name, -2.0, 2.0, 0.05) for name in ("incentive", "boost", "calm")),
        target=Target("supply", 1.0),
        constraints=(),
        horizon=12,
        dt=dt,
    )


def marketing_mix() -> Instance:
    """chc.mmm's case study as ``run_marketing_mix`` plans it, but spend may fall to zero.

    Its own levers keep a spend floor above zero, which ``max_levers`` refuses: held there, a
    lever is still on.
    """
    system = MarketingMixSystem()
    logs = system.sample(seed=0)
    return Instance(
        label="marketing mix (chc.mmm, spend floor 0), H=12",
        panel=Panel.from_frame(logs, unit="region", time="week", seed=0),
        adjustment=system.graph(),
        levers=tuple(Lever(name, 0.0, system.spend_ceiling, 0.08) for name in system.spend_columns),
        target=Target(SALES, 8.0),
        constraints=tuple(Constraint(name, lo=0.0) for name in system.adstock_columns),
        horizon=12,
        dt=1.0,
        known=adstock_dynamics(jnp.array(system.theta)),
    )


def _linear_logs(
    a_matrix: np.ndarray, b_matrix: np.ndarray, seed: int, dt: float
) -> dict[str, np.ndarray]:
    """60 units x 10 periods of ``x' = A x + B u``, every lever randomised, so nothing confounds."""
    rng = np.random.default_rng(seed)
    n, m = b_matrix.shape
    names = ("unit", "time", *(f"x{i}" for i in range(n)), *(f"u{j}" for j in range(m)))
    rows: dict[str, list[float]] = {name: [] for name in names}
    for unit in range(60):
        x = rng.normal(0.0, 0.3, n)
        for period in range(10):
            u = rng.normal(0.0, 0.7, m)
            for name, value in zip(names, (unit, period, *x, *u), strict=True):
                rows[name].append(value)
            x = x + dt * (a_matrix @ x + b_matrix @ u) + rng.normal(0.0, 0.01, n)
    return {name: np.asarray(values) for name, values in rows.items()}


def _linear_graph(n: int, m: int) -> CausalGraph:
    edges = [(f"u{j}", f"x{i}") for j in range(m) for i in range(n)]
    return CausalGraph.from_edges(edges + [(f"x{i}", "x0") for i in range(1, n)])


def saturation() -> Instance:
    """Three levers with one channel: two cheap ones that each saturate short of the target, and a
    wider, dearer one that alone gets closest. Planned from the target itself, so the cost is the
    upkeep: the best single lever is the wide one, the best pair the two cheap ones."""
    dt = 0.25
    names = ("narrow_a", "narrow_b", "wide")
    logs = _linear_logs(np.array([[-1.0]]), np.array([[1.0, 1.0, 1.0]]), seed=0, dt=dt)
    logs = {
        {"u0": names[0], "u1": names[1], "u2": names[2]}.get(key, key): value
        for key, value in logs.items()
    }
    return Instance(
        label="saturation (built for greedy to miss), H=20",
        panel=Panel.from_frame(logs, unit="unit", time="time", seed=0),
        adjustment=CausalGraph.from_edges([(name, "x0") for name in names]),
        levers=(
            Lever(names[0], 0.0, 0.6, 0.01),
            Lever(names[1], 0.0, 0.6, 0.01),
            Lever(names[2], 0.0, 0.9, 0.1),
        ),
        target=Target("x0", 1.0),
        constraints=(),
        horizon=20,
        dt=dt,
        x0=(1.0,),
    )


def random_instance(seed: int, m: int = 4) -> Instance:
    """Two states, the second in the plan as a loose constraint, so a lever can reach the target
    directly, through the second state, or both; boxes two-sided or spend-like ``[0, hi]``."""
    rng = np.random.default_rng(10_000 + seed)
    a_matrix = np.array(
        [
            [-rng.uniform(0.3, 1.2), rng.normal(0.0, 0.5)],
            [rng.normal(0.0, 0.5), -rng.uniform(0.3, 1.2)],
        ]
    )
    b_matrix = rng.normal(0.0, 1.0, (2, m))
    levers = []
    for j in range(m):
        width = rng.uniform(0.3, 2.0)
        lo = -rng.uniform(0.3, 2.0) if rng.random() < 0.5 else 0.0
        levers.append(Lever(f"u{j}", lo, width, float(10.0 ** rng.uniform(-2.0, -0.5))))
    goal = float(rng.uniform(0.5, 2.0) * rng.choice([-1.0, 1.0]))
    dt = 0.25
    return Instance(
        label=f"random {seed}",
        panel=Panel.from_frame(
            _linear_logs(a_matrix, b_matrix, seed, dt), unit="unit", time="time", seed=seed
        ),
        adjustment=_linear_graph(2, m),
        levers=tuple(levers),
        target=Target("x0", goal),
        constraints=(Constraint("x1", lo=-100.0, hi=100.0),),
        horizon=12,
        dt=dt,
    )


# ── the planning objective, rebuilt; and the group-L1 path on it ──────────────────────────────


def rebuilt_objective(instance: Instance, result: Prescription) -> Callable[[jax.Array], jax.Array]:
    """``prescribe``'s planning objective, rebuilt from its documented wiring and checked by
    re-solving: target weighted, constrained states free, each lever priced by its unit cost."""
    assert result.plan is not None
    n, m = int(result.plan.trajectory.shape[1]), len(instance.levers)
    base = instance.known or LinearDynamics(jnp.zeros((n, n)), jnp.zeros((n, m)))
    model = HybridDynamics(known=base, residual=result.model_fit.residual)
    start = result.plan.trajectory[0]
    weights = jnp.array([instance.target.weight] + [0.0] * (n - 1))
    cost = QuadraticCost(
        Q=jnp.diag(weights),
        R=jnp.diag(jnp.array([lever.unit_cost for lever in instance.levers])),
        Qf=jnp.diag(weights),
        x_target=jnp.array([instance.target.value] + [0.0] * (n - 1)),
    )
    again = causal_plan(
        model,
        start,
        cost,
        instance.dt,
        instance.horizon,
        jnp.array([lever.lo for lever in instance.levers]),
        jnp.array([lever.hi for lever in instance.levers]),
        constraints=(LinearConstraint.rate_limit(instance.horizon, [math.inf] * m),),
    )
    assert np.array_equal(np.asarray(again.actions), np.asarray(result.plan.actions)), (
        "the rebuilt objective drifted from prescribe's"
    )

    def objective(actions: jax.Array) -> jax.Array:
        return total_cost(model, start, actions, instance.dt, cost)

    return objective


def _tangent(columns: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Each column projected onto the tangent cone of its box at zero."""
    inside = np.where(lo[None, :] < 0.0, columns, np.maximum(columns, 0.0))
    return np.where(hi[None, :] > 0.0, inside, np.minimum(inside, 0.0))


def group_prox(columns: np.ndarray, lo: np.ndarray, hi: np.ndarray, tau: float) -> np.ndarray:
    """Per column, ``argmin 0.5 ||v - y||^2 + tau ||v||`` over its box, which contains zero.

    Zero exactly when the tangent part of ``y`` is within ``tau``. Otherwise the minimiser is
    ``clip(c y)`` with ``c = r / (r + tau)`` and ``r = ||v||``, one root in ``c`` in ``(0, 1)``,
    found by bisection. :func:`prox_check` holds it to a numerical minimiser before any path runs.
    """
    if tau == 0.0:
        return np.clip(columns, lo, hi)
    zero = np.linalg.norm(_tangent(columns, lo, hi), axis=0) <= tau
    left, right = np.zeros(columns.shape[1]), np.ones(columns.shape[1])
    for _ in range(64):
        c = 0.5 * (left + right)
        above = np.linalg.norm(np.clip(columns * c, lo, hi), axis=0) * (1.0 - c) > c * tau
        left, right = np.where(above, c, left), np.where(above, right, c)
    out = np.clip(columns * (0.5 * (left + right)), lo, hi)
    out[:, zero] = 0.0
    return out


def prox_check(cases: int = 300) -> float:
    """The worst objective excess of :func:`group_prox` over L-BFGS-B from two starts and zero, on
    random columns, boxes of all three shapes that contain zero, and thresholds."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(cases):
        length, width = int(rng.integers(1, 6)), int(rng.integers(1, 4))
        columns = rng.normal(0.0, 2.0, (length, width))
        shapes = [(-rng.uniform(0.2, 2.0), rng.uniform(0.2, 2.0)), (0.0, rng.uniform(0.2, 2.0))]
        shapes.append((-rng.uniform(0.2, 2.0), 0.0))
        boxes = [shapes[int(rng.integers(0, 3))] for _ in range(width)]
        lo, hi = np.array([b[0] for b in boxes]), np.array([b[1] for b in boxes])
        tau = float(rng.uniform(0.0, 3.0))
        proxed = group_prox(columns, lo, hi, tau)
        for j in range(width):
            y = columns[:, j]

            def value(v: np.ndarray, y: np.ndarray = y, tau: float = tau) -> float:
                return float(0.5 * np.sum((v - y) ** 2) + tau * np.sqrt(np.sum(v * v)))

            starts = (np.clip(y, lo[j], hi[j]), np.clip(rng.normal(0.0, 1.0, length), lo[j], hi[j]))
            found = [
                minimize(
                    value,
                    start,
                    bounds=[(lo[j], hi[j])] * length,
                    method="L-BFGS-B",
                    options={"ftol": 1e-15, "gtol": 1e-12},
                ).fun
                for start in starts
            ]
            worst = max(worst, value(proxed[:, j]) - min(*found, value(np.zeros(length))))
    return worst


def group_l1_path(
    objective: Callable[[jax.Array], jax.Array],
    shape: tuple[int, int],
    lo: np.ndarray,
    hi: np.ndarray,
    points: int = 40,
    floor: float = 1e-4,
    finest: float = 1e-3,
) -> tuple[list[float], list[tuple[float, tuple[int, ...]]]]:
    """Supports of the group-L1 solution from ``lambda_max`` down to ``floor * lambda_max``.

    ``lambda_max`` is the largest tangent norm of ``-grad J`` at the idle plan: the level at which
    the first lever enters. A geometric grid of ``points``, each solve warm-started from the last,
    then refined wherever two neighbours differ by more than one lever, by geometric midpoints,
    until they do not or are within a factor ``1 + finest`` -- so two levers entering together
    means together to that resolution. Returns the per-lever entry norms, and each distinct support
    with the ``lambda / lambda_max`` it first appears at.
    """
    value = jax.jit(objective)
    value_and_grad = jax.jit(jax.value_and_grad(objective))
    entry = np.linalg.norm(
        _tangent(-np.asarray(jax.grad(objective)(jnp.zeros(shape))), lo, hi), axis=0
    )
    top = float(entry.max())

    def solve(lam: float, start: np.ndarray) -> np.ndarray:
        """FISTA with backtracking and gradient restarts, stopped when an iterate stops moving."""
        point, previous, momentum, step = start.copy(), start.copy(), 1.0, 1.0
        for _ in range(5_000):
            f, g = value_and_grad(jnp.asarray(point))
            f, g = float(f), np.asarray(g)
            while True:
                candidate = group_prox(point - step * g, lo, hi, step * lam)
                move = candidate - point
                bound = f + float(np.sum(g * move)) + float(np.sum(move * move)) / (2.0 * step)
                if float(value(jnp.asarray(candidate))) <= bound:
                    break
                step *= 0.5
            if float(np.sum(g * (candidate - previous))) > 0.0:
                momentum = 1.0
            accelerated = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum**2))
            point = candidate + ((momentum - 1.0) / accelerated) * (candidate - previous)
            settled = float(np.max(np.abs(candidate - previous))) <= 1e-10
            previous, momentum, step = candidate, accelerated, 1.5 * step
            if settled:
                break
        return previous

    def support(actions: np.ndarray) -> tuple[int, ...]:
        return tuple(j for j in range(shape[1]) if np.any(actions[:, j] != 0.0))

    solved: list[tuple[float, np.ndarray]] = []
    current = np.zeros(shape)
    for ratio in np.logspace(0.0, math.log10(floor), points):
        current = solve(float(ratio) * top, current)
        solved.append((float(ratio), current))
    index = 0
    while index < len(solved) - 1:
        (upper, sparse), (lower, dense) = solved[index], solved[index + 1]
        jump = set(support(sparse)) ^ set(support(dense))
        if len(jump) > 1 and upper / lower > 1.0 + finest:
            middle = math.sqrt(upper * lower)
            solved.insert(index + 1, (middle, solve(middle * top, sparse)))
        else:
            index += 1
    supports: list[tuple[float, tuple[int, ...]]] = []
    for ratio, actions in solved:
        if not supports or supports[-1][1] != support(actions):
            supports.append((ratio, support(actions)))
    return [float(v) for v in entry], supports


# ── the comparison ────────────────────────────────────────────────────────────────────────────


class _Steps(logging.Handler):
    """Collects the ``selection`` records, for the solves and descent steps each step spent."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "chc_event", None) == "selection":
            self.records.append(record)


def _greedy_within(
    costs: dict[tuple[int, ...], float], allowed: tuple[int, ...], size: int
) -> tuple[int, ...]:
    """Greedy over the exhaustive table with only ``allowed`` as candidates: screen, then greedy."""
    chosen: tuple[int, ...] = ()
    while len(chosen) < min(size, len(allowed)):
        chosen = min(
            (tuple(sorted((*chosen, j))) for j in allowed if j not in chosen),
            key=costs.__getitem__,
        )
    return chosen


def compare(instance: Instance) -> dict:
    """Greedy, the group-L1 path, a group-L1 screen before greedy, greedy followed by single swaps,
    and backward elimination, each against exhaustive search at every size short of all the
    levers."""
    m = len(instance.levers)
    names = [lever.name for lever in instance.levers]
    costs = {
        subset: float(instance.prescribe(subset).plan.task_cost)  # type: ignore[union-attr]
        for size in range(1, m + 1)
        for subset in itertools.combinations(range(m), size)
    }
    collector = _Steps()
    log = logging.getLogger("chc.decision")
    log.addHandler(collector)
    log.setLevel(logging.INFO)
    try:
        greedy = instance.prescribe(None, max_levers=m)
    finally:
        log.removeHandler(collector)
        log.setLevel(logging.NOTSET)
    unrestricted = instance.prescribe(None)
    assert greedy.selection is not None
    assert greedy.plan is not None
    assert unrestricted.plan is not None
    order = [names.index(lever) for lever in greedy.selection.selected]
    lo = np.array([lever.lo for lever in instance.levers])
    hi = np.array([lever.hi for lever in instance.levers])
    entry, supports = group_l1_path(
        rebuilt_objective(instance, unrestricted), (instance.horizon, m), lo, hi
    )

    def named(subset: tuple[int, ...] | None) -> list[str] | None:
        return None if subset is None else [names[j] for j in subset]

    sizes = []
    for size in range(1, m):
        best = min((s for s in costs if len(s) == size), key=costs.__getitem__)
        mine = tuple(sorted(order[:size]))
        step = greedy.selection.steps[size - 1]
        assert step.task_cost == costs[mine], "a greedy step is not the pinned plan of its set"
        path_set = next((s for _, s in supports if len(s) == size), None)
        pool = next((s for _, s in supports if len(s) > size), tuple(range(m)))
        screened = _greedy_within(costs, pool, size)
        swapped = _single_swaps(costs, mine, m)
        eliminated = _backward(costs, m, size)
        sizes.append(
            {
                "size": size,
                "exhaustive": named(best),
                "exhaustive_cost": costs[best],
                "greedy": named(mine),
                "greedy_cost": costs[mine],
                "step_regret_bound": step.regret_bound,
                "path_set": named(path_set),
                "path_set_cost": None if path_set is None else costs[path_set],
                "screen": named(pool),
                "screen_then_greedy": named(screened),
                "screen_then_greedy_cost": costs[screened],
                "greedy_then_swaps": named(swapped),
                "greedy_then_swaps_cost": costs[swapped],
                "backward": named(eliminated),
                "backward_cost": costs[eliminated],
            }
        )
    return {
        "instance": instance.label,
        "levers": names,
        "idle_cost": greedy.selection.idle_cost,
        "all_levers_cost": costs[tuple(range(m))],
        "sizes": sizes,
        "solves": sum(len(getattr(r, "candidates", {})) for r in collector.records),
        "descent_steps_greedy": sum(int(getattr(r, "descent_steps", 0)) for r in collector.records),
        "descent_steps_unrestricted": unrestricted.plan.solver_iterations,
        "last_step_is_the_unrestricted_plan": bool(
            np.array_equal(np.asarray(greedy.plan.actions), np.asarray(unrestricted.plan.actions))
        ),
        "group_l1_entry_gradient_norms": dict(zip(names, entry, strict=True)),
        "group_l1_supports": [
            {"lambda_over_max": ratio, "levers": named(support)} for ratio, support in supports
        ],
    }


def _single_swaps(
    costs: dict[tuple[int, ...], float], start: tuple[int, ...], m: int
) -> tuple[int, ...]:
    """Best-improvement single swaps on the exhaustive table from greedy's set, until none helps."""
    current = start
    while True:
        swaps = [
            tuple(sorted((*(k for k in current if k != out), into)))
            for out in current
            for into in range(m)
            if into not in current
        ]
        cheapest = min(swaps, key=costs.__getitem__, default=current)
        if costs[cheapest] >= costs[current]:
            return current
        current = cheapest


def _backward(costs: dict[tuple[int, ...], float], m: int, size: int) -> tuple[int, ...]:
    """Backward elimination on the exhaustive table: from every lever, drop the one whose removal
    leaves the cheapest set, until ``size`` remain."""
    current = tuple(range(m))
    while len(current) > size:
        current = min(
            (tuple(k for k in current if k != out) for out in current), key=costs.__getitem__
        )
    return current


METHODS = ("greedy", "path_set", "screen_then_greedy", "greedy_then_swaps", "backward")
"""``greedy`` is what ships. ``path_set`` is the group-L1 path's first support of the size, planned
alone; ``screen_then_greedy`` is greedy among the levers of the path's first support larger than
the size, the screen-then-select design weighed in ``docs/adr/0004-greedy-lever-selection.md``;
``greedy_then_swaps`` trades one lever at a time from the greedy set while that lowers the cost;
``backward`` removes levers one at a time from the full set. The last four are read off the
exhaustive table, so they cost no solve here."""


def tally(rows: list[dict]) -> dict:
    """Per method, the cells whose set costs what the exhaustive best does; and how each method
    moves greedy's answer -- the misses it repairs and the hits it breaks -- in all and per size,
    since greedy's first step is exhaustive by construction. Then whether each greedy step's regret
    bound covers its gap to the best set of its size, which it must if it is valid."""
    cells = [cell for row in rows for cell in row["sizes"]]
    ratios = sorted(row["descent_steps_greedy"] / row["descent_steps_unrestricted"] for row in rows)

    def hit(cell: dict, method: str) -> bool:
        cost = cell[f"{method}_cost"]
        return cost is not None and cost <= cell["exhaustive_cost"]

    def counts(chosen: list[dict]) -> dict:
        return {
            "cells": len(chosen),
            "best_set_found": {
                method: sum(hit(cell, method) for cell in chosen) for method in METHODS
            },
            "greedy_misses_repaired": {
                method: sum(hit(cell, method) and not hit(cell, "greedy") for cell in chosen)
                for method in METHODS[1:]
            },
            "greedy_hits_broken": {
                method: sum(hit(cell, "greedy") and not hit(cell, method) for cell in chosen)
                for method in METHODS[1:]
            },
        }

    return {
        **counts(cells),
        "by_size": {
            size: counts([cell for cell in cells if cell["size"] == size])
            for size in sorted({cell["size"] for cell in cells})
        },
        "path_skips_the_size": sum(cell["path_set"] is None for cell in cells),
        "worst_greedy_relative_excess": max(
            (cell["greedy_cost"] - cell["exhaustive_cost"]) / abs(cell["exhaustive_cost"])
            for cell in cells
        ),
        "step_bound_covers_the_gap": sum(
            cell["step_regret_bound"] >= cell["greedy_cost"] - cell["exhaustive_cost"]
            for cell in cells
        ),
        "step_bound_infinite": sum(math.isinf(cell["step_regret_bound"]) for cell in cells),
        "solves_per_instance": sorted({row["solves"] for row in rows}),
        "descent_steps_greedy_over_unrestricted": {
            "min": round(ratios[0], 2),
            "median": round(statistics.median(ratios), 2),
            "max": round(ratios[-1], 2),
        },
        "last_step_is_the_unrestricted_plan": all(
            row["last_step_is_the_unrestricted_plan"] for row in rows
        ),
    }


def instances() -> dict:
    worst = prox_check()
    assert worst <= 1e-12, f"the group prox is {worst} above a numerical minimiser"
    rows = []
    for build in (driver_pool, marketing_mix, saturation):
        rows.append(compare(build()))
        print(json.dumps(rows[-1]), file=sys.stderr, flush=True)
    return {"prox_worst_excess": worst, "tally": tally(rows), "instances": rows}


def random_sweep(count: int = 30) -> dict:
    worst = prox_check()
    assert worst <= 1e-12, f"the group prox is {worst} above a numerical minimiser"
    rows = []
    for seed in range(count):
        rows.append(compare(random_instance(seed)))
        print(json.dumps(rows[-1]), file=sys.stderr, flush=True)
    # Where no lever lowers the cost every set ties at the idle plan, a hit for every method.
    idle = [row["instance"] for row in rows if row["all_levers_cost"] >= row["idle_cost"]]
    informative = [row for row in rows if row["instance"] not in idle]
    misses = [
        {"instance": row["instance"], **cell}
        for row in informative
        for cell in row["sizes"]
        if cell["greedy_cost"] > cell["exhaustive_cost"]
    ]
    return {
        "prox_worst_excess": worst,
        "idle_plan_optimal": idle,
        "tally": tally(informative),
        "greedy_misses": misses,
    }


def fingerprint() -> dict[str, str]:
    """Digests of prescriptions made without ``max_levers``: their JSON, ``selection`` aside, and
    the bytes of their actions and trajectory. Calls nothing ``main`` lacks, so it runs on both."""
    pool, mix = driver_pool(), marketing_mix()
    system = MarketingMixSystem()
    calls = {
        "driver pool, three levers": {"levers": list(pool.levers)},
        "driver pool, rate-limited lever, priced bound": {
            "levers": [Lever("incentive", -2.0, 2.0, 0.05, cap_per_step=0.1)],
            "constraints": [Constraint("wait", hi=0.5)],
        },
        "driver pool, held bound": {
            "levers": list(pool.levers),
            "constraints": [Constraint("wait", hi=0.5)],
            "hold_constraints": True,
            "x0": jnp.array([0.0, 0.45]),
        },
    }
    out = {}
    for label, kwargs in calls.items():
        result = prescribe(
            pool.panel,
            target=pool.target,
            adjustment=pool.adjustment,
            horizon=pool.horizon,
            dt=pool.dt,
            **kwargs,
        )
        out[label] = _digest(result)
    case_study = prescribe(
        mix.panel,
        levers=system.levers(),
        target=mix.target,
        constraints=list(mix.constraints),
        adjustment=mix.adjustment,
        known=mix.known,
        horizon=mix.horizon,
        dt=mix.dt,
        tolerance=1.0,
    )
    out["marketing mix, as run_marketing_mix plans it"] = _digest(case_study)
    return out


def _digest(result: Prescription) -> str:
    assert result.plan is not None
    payload = result.to_json()
    payload.pop("selection", None)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    digest.update(np.asarray(result.plan.actions).tobytes())
    digest.update(np.asarray(result.plan.trajectory).tobytes())
    return digest.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("part", choices=("instances", "random", "fingerprint"))
    part = parser.parse_args().part
    result = {"instances": instances, "random": random_sweep, "fingerprint": fingerprint}[part]()
    json.dump(result, sys.stdout, indent=1)
    print()
