"""The measurements behind docs/adr/0012-a-switchback-for-a-named-effect.md on a plant the working
model does not describe: the zone market of ``chc.zones``. Bias, spread, power and coverage only, no
wall time.

Each zone's incentive is switched between 0 and ``LEVEL`` by the zone's own coins, its price held at
the list price, and the outcome read is its idle supply at the start of each period. The market
departs from the working model three ways: the incentive also acts through a stock, so the state is
not first order; half of what a zone recruits is drawn from its two neighbours; and the zones'
channels differ. Under harmonic matching it is not linear either.

The prior is fitted to a long i.i.d. pilot, 200 000 periods in each zone, so that it is the working
model's best fit and not its noise: two pilots of 20 000 read the channel 3% apart.
The truth is the effect of holding one zone's incentive on for ``H`` periods, found by branching a
run of the design: from the run's own state, ``H`` periods with that lever forced on and forced off,
everything else as the run drew it, averaged over the zones as the pooled readings average them.
Under linear matching it is the closed form from ``linear_gaussian``, which the script checks.

    zones   five plans, the simulations of tests/test_switchback.py that need no measurement
            noise, planned from the pilot and run on the market. For each reading: its bias, its
            spread against the plan's standard error, its data standard error against its spread,
            its power at the planned MDE with each, its coverage, and how often the plug-in's
            first-order check warned.
    pilots  the three model-free plans, each reading's MDE restated by ``restate_mde`` from the
            first tenth, quarter and half of its arm, or ten, fifteen and twenty blocks a zone,
            and the run's power at it: against the truth, and against the reading's own mean,
            which leaves out the reading's bias and keeps the pilot's error.

Run: uv run python scripts/bench_switchback.py {zones,pilots} [--matching harmonic] [--spill S]
     [--replicates N]
"""

from __future__ import annotations

import argparse
import json
import logging
import math

import jax
import jax.numpy as jnp
import numpy as np

from chc.integrate import rk4_step
from chc.switchback import (
    CHANNEL,
    STEADY_STATE,
    BlockDesign,
    Horizon,
    MarkovDesign,
    PersistencePrior,
    design_switchback,
    read_switchback,
    restate_mde,
)
from chc.zones import ZoneMarketSystem

jax.config.update("jax_enable_x64", True)

LEVEL = 0.3  # the incentive when a zone's coin says on; the logged operator's own level
PERIODS, BURN, PILOT, CHUNK = 2000, 400, 200_000, 250
HELD = 120  # periods a lever is held for the steady state: the slowest mode is 0.806 a period
PILOT_SHARES, PILOT_BLOCKS = (0.1, 0.25, 0.5), (10, 15, 20)
Z = 1.959964
SEED = 20260928

PLANS = {
    "tau_5": ((Horizon(5),), True),
    "channel and steady state": ((CHANNEL, STEADY_STATE), True),
    "channel and tau_5 without a model": ((CHANNEL, Horizon(5)), False),
    "steady state without a model": ((STEADY_STATE,), False),
    "tau_2 without a model": ((Horizon(2),), False),
}


def _stepper(system: ZoneMarketSystem):
    k = system.zones
    jitter = jnp.concatenate([jnp.full(2 * k, system.noise), jnp.zeros(k)])

    def step(x, inputs):
        coin, shock, eps = inputs
        lever = jnp.concatenate([LEVEL * coin, jnp.zeros(k)])
        x_next = rk4_step(system.plant(shock), 0.0, x, lever, 1.0) + jitter * eps
        return x_next, x_next

    def run(x0, coins, shocks, eps):
        _, states = jax.lax.scan(step, x0, (coins, shocks, eps))
        return jnp.concatenate([x0[None], states])

    return jax.jit(jax.vmap(run))


def _coins(rng, runs: int, n: int, k: int, design) -> np.ndarray:
    """``(runs, n, k)`` settings, each zone's drawn independently."""
    if isinstance(design, BlockDesign):
        blocks = rng.integers(0, 2, (runs, n // design.length + 1, k)).astype(float)
        return np.repeat(blocks, design.length, axis=1)[:, :n]
    v = np.empty((runs, n, k))
    v[:, 0] = rng.integers(0, 2, (runs, k))
    flips = rng.random((runs, n, k)) < design.flip
    for t in range(1, n):
        v[:, t] = np.where(flips[:, t], 1.0 - v[:, t - 1], v[:, t - 1])
    return v


def _burn(design) -> int:
    length = design.length if isinstance(design, BlockDesign) else 1
    return length * math.ceil(BURN / length)


def _draws(system, rng, runs: int, n: int, design):
    k = system.zones
    coins = _coins(rng, runs, n, k, design)
    return (
        coins,
        rng.normal(0.0, system.shock_sd, (runs, n, k)),
        rng.standard_normal((runs, n, 3 * k)),
    )


def _switchback(system, run, rng, runs: int, periods: int, design):
    """Levers ``(runs, zones, periods)`` and supply ``(runs, zones, periods + 1)``, burnt in."""
    burn, k = _burn(design), system.zones
    levers, supply = [], []
    for start in range(0, runs, CHUNK):
        coins, shocks, eps = _draws(system, rng, min(CHUNK, runs - start), periods + burn, design)
        x0 = jnp.tile(system.do_nothing, (coins.shape[0], 1))
        states = np.asarray(run(x0, jnp.asarray(coins), jnp.asarray(shocks), jnp.asarray(eps)))
        levers.append(coins[:, burn:].transpose(0, 2, 1))
        supply.append(states[:, burn:, :k].transpose(0, 2, 1))
    return np.concatenate(levers), np.concatenate(supply)


def _pilot(system, run, rng) -> PersistencePrior:
    """The working model's least-squares fit to one long i.i.d. run, an intercept per zone."""
    u, y = _switchback(system, run, rng, 1, PILOT, MarkovDesign(0.5))
    u, y = u[0], y[0]
    target, lagged = y[:, 1:], y[:, :-1]
    x = np.stack([lagged - lagged.mean(1, keepdims=True), u - u.mean(1, keepdims=True)], axis=-1)
    x = x.reshape(-1, 2)
    t = (target - target.mean(1, keepdims=True)).reshape(-1)
    bread = np.linalg.inv(x.T @ x)
    (a, b), resid = bread @ x.T @ t, t - x @ (bread @ x.T @ t)
    se_a = math.sqrt((bread @ (x.T * resid**2) @ x @ bread)[0, 0])
    sigma = math.sqrt(resid @ resid / (resid.size - 2 - u.shape[0]))
    return PersistencePrior(a - 2.0 * se_a, a + 2.0 * se_a, sigma, b)


def _closed_form(system, h: float) -> np.ndarray:
    """Each zone's own ``tau_H`` under the plant's one-period Jacobians at the do-nothing point."""
    lg = system.linear_gaussian(1.0)
    k = system.zones
    if math.isinf(h):
        return LEVEL * np.diag(np.linalg.solve(np.eye(3 * k) - lg.a, lg.b[:, :k])[:k])
    total, v = np.zeros(k), lg.b[:, :k].copy()
    for _ in range(int(h)):
        total += np.diag(v[:k])
        v = lg.a @ v
    return LEVEL * total


def _branched(system, rng, design, held: int, runs: int = 200, starts: int = 40):
    """Each zone's effect of holding its incentive on for ``held`` periods, and its Monte Carlo
    standard error, by branching runs of ``design`` at ``starts`` points ``max(held, block)`` apart
    (block starts for a block design)."""
    k = system.zones
    run = _stepper(system)
    spacing = design.length if isinstance(design, BlockDesign) else 1
    spacing *= math.ceil(held / spacing)
    burn = _burn(design)
    coins, shocks, eps = _draws(system, rng, runs, burn + starts * spacing + held, design)
    x0 = jnp.tile(system.do_nothing, (runs, 1))
    states = np.asarray(run(x0, jnp.asarray(coins), jnp.asarray(shocks), jnp.asarray(eps)))
    at = [burn + m * spacing for m in range(starts)]
    x = jnp.asarray(np.concatenate([states[:, s] for s in at]))
    window = [np.concatenate([d[:, s : s + held] for s in at]) for d in (coins, shocks, eps)]
    effect, error = np.empty(k), np.empty(k)
    for i in range(k):
        on, off = window[0].copy(), window[0].copy()
        on[:, :, i], off[:, :, i] = 1.0, 0.0
        ends = [
            np.asarray(run(x, jnp.asarray(c), jnp.asarray(window[1]), jnp.asarray(window[2])))
            for c in (on, off)
        ]
        paired = ends[0][:, -1, i] - ends[1][:, -1, i]
        effect[i], error[i] = paired.mean(), paired.std(ddof=1) / math.sqrt(paired.size)
    return effect, error


class _Events(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.events.append(getattr(record, "chc_event", ""))


def _zones(replicates: int, matching: str, spill: float) -> dict:
    system = ZoneMarketSystem(matching=matching, spill=spill)  # type: ignore[arg-type]
    run = _stepper(system)
    prior = _pilot(system, run, np.random.default_rng([SEED, 0]))
    handler = _Events()
    logger = logging.getLogger("chc.switchback")
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    out: dict = {"prior": prior.__dict__, "level": LEVEL, "matching": matching, "spill": spill}
    for index, (name, (estimands, trusted)) in enumerate(PLANS.items()):
        plan = design_switchback(estimands, prior, PERIODS, system.zones, trust_state_model=trusted)
        rng = np.random.default_rng([SEED, 1, index])
        readings = {}
        for j, arm in enumerate(plan.arms):
            u, y = _switchback(system, run, rng, replicates, round(arm.share * PERIODS), arm.design)
            blocks = arm.design if isinstance(arm.design, BlockDesign) else None
            for report in (r for r in plan.reports if r.arm == j):
                h = report.estimand.periods
                held = HELD if math.isinf(h) else int(h)
                zone_tau, zone_error = _branched(
                    system, np.random.default_rng([SEED, 2, index, j]), arm.design, held
                )
                linearised = _closed_form(system, h)
                if matching == "linear":
                    assert np.allclose(zone_tau, linearised, rtol=1e-8), (zone_tau, linearised)
                tau = float(zone_tau.mean())
                handler.events.clear()
                reads = [
                    read_switchback(u[r], y[r], report.estimand, report.analysis, blocks=blocks)
                    for r in range(replicates)
                ]
                warned = handler.events.count("switchback_second_state") / replicates
                x = np.array([r.estimate for r in reads])
                se = np.array([r.se for r in reads])
                spread = float(x.std(ddof=1))
                null = tau - report.mde
                entry = {
                    "analysis": report.analysis,
                    "design": str(arm.design),
                    "tau": tau,
                    "tau_monte_carlo_se": float(np.sqrt(np.sum(zone_error**2)) / zone_tau.size),
                    "tau_linearised": float(linearised.mean()),
                    "bias_over_tau": float(x.mean() / tau - 1.0),
                    "planned_bias_over_tau": report.bias / tau,
                    "spread_over_planned_se": spread / report.se,
                    "data_se_over_spread": float(se.mean()) / spread,
                    "power_with_data_se": float(np.mean(np.abs(x - null) / se > Z)),
                    "power_with_planned_se": float(np.mean(np.abs(x - null) / report.se > Z)),
                    "coverage": float(
                        np.mean([r.interval[0] <= tau <= r.interval[1] for r in reads])
                    ),
                    "first_order_warned": warned,
                }
                if blocks is not None:
                    alone = np.array(
                        [
                            [
                                read_switchback(
                                    u[r][i], y[r][i], report.estimand, "block_dim", blocks=blocks
                                ).estimate
                                for i in range(system.zones)
                            ]
                            for r in range(replicates)
                        ]
                    )
                    corr = np.corrcoef(alone.T)
                    entry["neighbour_correlation"] = float(
                        np.mean([corr[i, (i + 1) % system.zones] for i in range(system.zones)])
                    )
                readings[report.estimand.name] = entry
        out[name] = {"warnings": list(plan.warnings), **readings}
    return out


def _pilots(replicates: int, matching: str, spill: float) -> dict:
    system = ZoneMarketSystem(matching=matching, spill=spill)  # type: ignore[arg-type]
    run = _stepper(system)
    prior = _pilot(system, run, np.random.default_rng([SEED, 0]))
    out: dict = {"prior": prior.__dict__, "level": LEVEL, "matching": matching, "spill": spill}
    for index, (name, (estimands, trusted)) in enumerate(PLANS.items()):
        if trusted:
            continue
        plan = design_switchback(estimands, prior, PERIODS, system.zones, trust_state_model=False)
        rng = np.random.default_rng([SEED, 9, index])
        readings = {}
        for j, arm in enumerate(plan.arms):
            periods = round(arm.share * PERIODS)
            u, y = _switchback(system, run, rng, replicates, periods, arm.design)
            blocks = arm.design if isinstance(arm.design, BlockDesign) else None
            if blocks is None:
                cuts = {f"a share of {s}": int(s * periods) for s in PILOT_SHARES}
            else:
                cuts = {f"{m} blocks": m * blocks.length for m in PILOT_BLOCKS}
            for report in (r for r in plan.reports if r.arm == j):
                h = report.estimand.periods
                held = HELD if math.isinf(h) else int(h)
                zone_tau, _ = _branched(
                    system, np.random.default_rng([SEED, 2, index, j]), arm.design, held
                )
                tau = float(zone_tau.mean())
                reads = [
                    read_switchback(u[r], y[r], report.estimand, report.analysis, blocks=blocks)
                    for r in range(replicates)
                ]
                x = np.array([r.estimate for r in reads])
                se = np.array([r.se for r in reads])
                spread = float(x.std(ddof=1))
                entry: dict = {
                    "analysis": report.analysis,
                    "design": str(arm.design),
                    "bias_over_spread": float((x.mean() - tau) / spread),
                    "planned_se_over_spread": report.se / spread,
                    "power_at_the_planned_mde": float(
                        np.mean(np.abs(x - (tau - report.mde)) / se > Z)
                    ),
                }
                for label, cut in cuts.items():
                    kept, restated = [], []
                    for r in range(replicates):
                        try:
                            pilot = restate_mde(
                                plan, report.estimand, u[r][:, :cut], y[r][:, : cut + 1]
                            )
                        except ValueError:  # a zone whose pilot showed one setting only
                            continue
                        kept.append(r)
                        restated.append((pilot.se, pilot.mde))
                    rows = np.array(kept)
                    se_mde = np.array(restated)
                    # the reading's own mean, and the plan's bias its MDE allows for
                    centre = x.mean() + abs(report.bias)
                    entry[label] = {
                        "periods": cut,
                        "refused": replicates - rows.size,
                        "restated_se_over_spread": float(
                            np.sqrt(np.mean(se_mde[:, 0] ** 2)) / spread
                        ),
                        "restated_se_cv": float(se_mde[:, 0].std() / se_mde[:, 0].mean()),
                        "power": float(
                            np.mean(np.abs(x[rows] - (tau - se_mde[:, 1])) / se[rows] > Z)
                        ),
                        "power_about_the_mean": float(
                            np.mean(np.abs(x[rows] - (centre - se_mde[:, 1])) / se[rows] > Z)
                        ),
                    }
                readings[report.estimand.name] = entry
        out[name] = readings
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=["zones", "pilots"])
    parser.add_argument("--matching", choices=["linear", "harmonic"], default="linear")
    parser.add_argument("--spill", type=float, default=ZoneMarketSystem.spill)
    parser.add_argument("--replicates", type=int, default=8000)
    args = parser.parse_args()
    case = _zones if args.case == "zones" else _pilots
    print(json.dumps(case(args.replicates, args.matching, args.spill), indent=2))


if __name__ == "__main__":
    main()
