"""The measurements behind the logger check (chc.independence.gcm_test, wired into
chc.decision.prescribe and Prescription.evaluate): whether it holds its level where the logger
read the state alone, at the sample sizes CHC's cases have, how much dependence it catches, and
whether what it flags is what costs the evaluation its coverage. Rates only, no wall time.

Three worlds. In each the logger reads the state and draws noise of its own; under the null that
is all it reads, and a ``reads`` of gamma adds gamma times a column the check tests.

    panel    units x periods: an AR(1) state the lever moves (0.8, unit levels), a season every
             unit sees; the columns are the season seen, the state and the lever one period back.
    mm       geos x weeks: sales with carryover 0.85, a 52-week season, spend reacting to sales;
             the columns are a promotion calendar that follows the season, and sales and spend
             one week back.
    track_o  one long run of a lightly damped oscillator (omega 3, dt 0.05) under a logger that
             adds damping 0.15 and a dither; the columns are an AR(0.9) wind the logger may read,
             and the angle, the velocity and the command one step back. The angle one step back
             is the current angle less dt times the velocity, so the check leaves it out.

The logger's noise is Gaussian ("homo"), grows with the state ("hetero"), or has half its
variance drawn once per period for every unit ("common", a national budget shock). In the panel
world two loggers read the state through more than the check's quadratic, with Gaussian noise: one
clips its lever to [-1.5, 1.5] ("clip", a box), and one switches it at zero, -tanh(2 x) in place of
-0.6 x ("switch", a threshold rule). Both read the state alone, so the null holds.

    size          the shipped check under the null, 1000 replicates a case.
    alternatives  under the null, 500 replicates a case: the plain GCM (rows, a Gaussian
                  multiplier), clusters by unit, and cross-fitting by unit or by time, beside the
                  shipped check. Why the shipped one is what it is.
    power         the shipped check where the logger also reads the season, the calendar or the
                  wind, 400 replicates a case, with the medians of the partial correlation it
                  found between the lever and that column and of what it reports detectable there.
    detectable    whether a pair at the partial correlation the check reports detectable is
                  caught eight times in ten: independent rows in 10 to 1000 clusters, one or three
                  pairs, the dependence in the first. The target is the mean of what 200 null
                  draws report, and 400 draws at it are tested; beside the shipped reading of the
                  statistic, one that takes it for a normal.
    protects      Prescription.evaluate on the market of tests/test_lifecycle_loop.py, where
                  demand pushes supply by 0.15, and on one where it pushes 0.6: one plan each,
                  fitted where the incentive was drawn afresh, evaluated on later panels whose
                  logger also chased demand, or kept part of its last incentive. How often the
                  interval covers the plan's true cost, at the default model_error = 1 and at 0,
                  beside how often the check rejects, 400 panels a case.

Run: uv run python scripts/bench_logger_check.py {size,alternatives,power,detectable,protects}
     [--replicates N] [--workers W]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("JAX_ENABLE_X64", "1")  # the tests' precision, which the plan is made in

import numpy as np

from chc import independence
from chc.independence import gcm_test

SEED = 20260930
DRAWS = 999

# (world, units, periods, noise, reads)
CASES = {
    "size": [
        ("track_o", 1, 4000, "homo", 0.0),
        ("track_o", 1, 4000, "hetero", 0.0),
        ("mm", 1, 104, "homo", 0.0),
        ("mm", 1, 156, "homo", 0.0),
        ("mm", 1, 156, "hetero", 0.0),
        ("mm", 1, 260, "homo", 0.0),
        ("mm", 5, 156, "homo", 0.0),
        ("mm", 5, 156, "hetero", 0.0),
        ("mm", 5, 156, "common", 0.0),
        ("panel", 30, 20, "homo", 0.0),
        ("panel", 30, 20, "common", 0.0),
        ("panel", 100, 20, "homo", 0.0),
        ("panel", 100, 20, "common", 0.0),
        ("panel", 300, 10, "homo", 0.0),
        ("panel", 300, 10, "common", 0.0),
        ("panel", 4000, 12, "homo", 0.0),
        ("panel", 4000, 12, "common", 0.0),
        ("panel", 400, 12, "clip", 0.0),
        ("panel", 4000, 12, "clip", 0.0),
        ("panel", 400, 12, "switch", 0.0),
        ("panel", 4000, 12, "switch", 0.0),
    ],
    "alternatives": [
        ("mm", 1, 156, "homo", 0.0),
        ("mm", 5, 156, "homo", 0.0),
        ("mm", 5, 156, "common", 0.0),
        ("panel", 30, 20, "homo", 0.0),
        ("panel", 100, 20, "common", 0.0),
        ("track_o", 1, 4000, "homo", 0.0),
    ],
    "power": [
        ("mm", 1, 156, "homo", 0.2),
        ("mm", 1, 156, "homo", 0.3),
        ("mm", 1, 156, "homo", 0.5),
        ("mm", 5, 156, "homo", 0.1),
        ("mm", 5, 156, "homo", 0.2),
        ("panel", 100, 20, "homo", 0.05),
        ("panel", 100, 20, "homo", 0.1),
        ("track_o", 1, 4000, "homo", 0.005),
        ("track_o", 1, 4000, "homo", 0.01),
    ],
}
ALTERNATIVES = ("shipped", "rows", "units", "crossfit_units", "crossfit_time")


def _noise(rng: np.random.Generator, units: int, noise: str, scale) -> np.ndarray:
    if noise == "common":
        return scale * math.sqrt(0.5) * (rng.normal() + rng.normal(size=units))
    return scale * rng.normal(size=units)


def _panel(rng, units, periods, noise, reads):
    burn = 20
    total = periods + burn
    season = np.sin(2 * np.pi * np.arange(total) / 12.0) + 0.5 * rng.normal(size=total)
    level = rng.normal(size=units)
    x, u = np.zeros((units, total)), np.zeros((units, total))
    seen = season[None, :] + rng.normal(size=(units, total))
    for t in range(total):
        if t:
            x[:, t] = 0.8 * x[:, t - 1] + 0.5 * u[:, t - 1] + 0.3 * level + rng.normal(size=units)
        scale = np.sqrt((0.5 + x[:, t] ** 2) / 1.5) if noise == "hetero" else 1.0
        mean = -np.tanh(2.0 * x[:, t]) if noise == "switch" else -0.6 * x[:, t]
        u[:, t] = mean + reads * seen[:, t] + _noise(rng, units, noise, scale)
        if noise == "clip":
            u[:, t] = np.clip(u[:, t], -1.5, 1.5)
    now, before = slice(burn, total), slice(burn - 1, total - 1)
    columns = np.stack([seen[:, now], x[:, before], u[:, before]], axis=2)
    return x[:, now, None], u[:, now], columns


def _mm(rng, units, periods, noise, reads):
    burn = 60
    total = periods + burn
    season = np.sin(2 * np.pi * np.arange(total) / 52.0) + 0.3 * rng.normal(size=total)
    level = rng.normal(size=units)
    sales, spend = np.zeros((units, total)), np.zeros((units, total))
    calendar = season[None, :] + 0.5 * rng.normal(size=(units, total))
    for t in range(total):
        if t:
            sales[:, t] = (
                0.85 * sales[:, t - 1]
                + 0.4 * spend[:, t - 1]
                + 0.5 * season[t]
                + 0.3 * level
                + 0.5 * rng.normal(size=units)
            )
        scale = np.sqrt((0.5 + sales[:, t] ** 2 / 4) / 1.5) if noise == "hetero" else 1.0
        spend[:, t] = -0.3 * sales[:, t] + reads * calendar[:, t] + _noise(rng, units, noise, scale)
    now, before = slice(burn, total), slice(burn - 1, total - 1)
    columns = np.stack([calendar[:, now], sales[:, before], spend[:, before]], axis=2)
    return sales[:, now, None], spend[:, now], columns


def _track_o(rng, units, periods, noise, reads):
    dt, omega, damping = 0.05, 3.0, 2 * 0.15 * 3.0
    burn = 400
    total = periods + burn + 1
    angle, velocity = np.zeros((units, total)), np.zeros((units, total))
    command, wind = np.zeros((units, total)), np.zeros((units, total))
    for t in range(total):
        if t:
            wind[:, t] = 0.9 * wind[:, t - 1] + math.sqrt(1 - 0.81) * rng.normal(size=units)
            pull = -(omega**2) * angle[:, t - 1] + command[:, t - 1] + 0.15 * rng.normal(size=units)
            velocity[:, t] = velocity[:, t - 1] + dt * pull
            angle[:, t] = angle[:, t - 1] + dt * velocity[:, t]
        scale = np.sqrt((0.5 + (velocity[:, t] / 0.5) ** 2) / 1.5) if noise == "hetero" else 1.0
        command[:, t] = (
            -damping * velocity[:, t] + reads * wind[:, t] + 0.25 * _noise(rng, units, noise, scale)
        )
    now, before = slice(burn + 1, total), slice(burn, total - 1)
    states = np.stack([angle[:, now], velocity[:, now]], axis=2)
    columns = np.stack([wind[:, now], angle[:, before], velocity[:, before], command[:, before]], 2)
    return states, command[:, now], columns


WORLDS = {"panel": _panel, "mm": _mm, "track_o": _track_o}


def _draw(case, index):
    world, units, periods, noise, reads = case
    rng = np.random.default_rng(
        [SEED, *map(ord, world), units, periods, *map(ord, noise), round(1000 * reads), index]
    )
    states, lever, columns = WORLDS[world](rng, units, periods, noise, reads)
    rows = units * periods
    period = np.tile(np.arange(periods), units)
    unit = np.repeat(np.arange(units), periods)
    return (
        lever.reshape(rows),
        columns.reshape(rows, -1),
        states.reshape(rows, -1),
        period,
        unit,
    )


# ------------------------------------------------------------------------------ the alternatives


def _monomials(z: np.ndarray) -> np.ndarray:
    z = (z - z.mean(axis=0)) / z.std(axis=0)
    columns = [np.ones(len(z))]
    for i in range(z.shape[1]):
        columns.append(z[:, i])
        columns.extend(z[:, i] * z[:, j] for j in range(i, z.shape[1]))
    return np.column_stack(columns)


def _residual(target, design, fold):
    residual = np.empty_like(target)
    for k in np.unique(fold):
        test = fold == k
        train = ~test if len(np.unique(fold)) > 1 else test
        coefficients, *_ = np.linalg.lstsq(design[train], target[train], rcond=None)
        residual[test] = target[test] - design[test] @ coefficients
    return residual


def _gaussian_multiplier(products, clusters, rng) -> float:
    """The GCM as usually calibrated: centred cluster sums, a Gaussian multiplier over them."""
    _, labels = np.unique(clusters, return_inverse=True)
    sums = np.zeros((labels.max() + 1, products.shape[1]))
    np.add.at(sums, labels, products)
    centred = sums - sums.mean(axis=0)
    scale = np.sqrt(np.sum(centred**2, axis=0))
    keep = scale > 0
    if not keep.any():  # one cluster: nothing to centre against
        return math.nan
    statistic = np.max(np.abs(sums.sum(axis=0)[keep]) / scale[keep])
    draws = rng.normal(size=(DRAWS, len(sums)))
    flipped = np.max(np.abs(draws @ centred[:, keep]) / scale[keep], axis=1)
    return float((1 + np.count_nonzero(flipped >= statistic)) / (DRAWS + 1))


def _alternative(name, lever, columns, states, period, unit, rng) -> float:
    design = _monomials(states)
    if name == "crossfit_units":
        order = rng.permutation(unit.max() + 1)
        fold = (np.argsort(order) % 2)[unit] if unit.max() > 0 else np.zeros_like(unit)
    elif name == "crossfit_time":
        fold = (period * 2) // (period.max() + 1)
    else:
        fold = np.zeros_like(unit)
    x = _residual(lever[:, None], design, fold)
    y = _residual(columns, design, fold)
    live = y.var(axis=0) > 1e-8 * columns.var(axis=0)
    products = x * y[:, live]
    clusters = unit if name == "units" else np.arange(len(lever))
    return _gaussian_multiplier(products, clusters, rng)


# --------------------------------------------------------------------------------------- sections


def _replicate(job) -> dict:
    section, case, index = job
    lever, columns, states, period, unit = _draw(case, index)
    test = gcm_test(lever, columns, states, clusters=period, draws=DRAWS, seed=index)
    # the column a logger that reads more reads is the first in every world
    row = {
        "p": test.p_value,
        "found": float(test.partial_correlation[0, 0]),
        "detectable": float(test.detectable[0, 0]),
    }
    if section == "alternatives":
        rng = np.random.default_rng([SEED, index, 1])
        for name in ALTERNATIVES[1:]:
            row[name] = _alternative(name, lever, columns, states, period, unit, rng)
    return row


def _rate(hits: int, n: int) -> dict:
    p = hits / n if n else math.nan
    return {"rate": round(p, 4), "se": round(math.sqrt(p * (1.0 - p) / n), 4) if n else None}


def _summarise(section, case, rows) -> dict:
    world, units, periods, noise, reads = case
    n = len(rows)
    out: dict = {
        "world": world,
        "units": units,
        "periods": periods,
        "noise": noise,
        "reads": reads,
        "replicates": n,
    }
    names = ALTERNATIVES if section == "alternatives" else ("shipped",)
    for name in names:
        key = "p" if name == "shipped" else name
        defined = [r[key] for r in rows if not math.isnan(r[key])]
        for level in (0.05, 0.10):
            hits = sum(p <= level for p in defined)
            out[f"{name}_rejects_{level:.2f}"] = _rate(hits, len(defined)) if defined else None
    if section == "power":
        for key in ("found", "detectable"):
            out[f"median_{key}"] = round(float(np.median([r[key] for r in rows])), 4)
    return out


# ----------------------------------------------------------------------------------- detectable

# (clusters, rows, pairs)
DETECTABLE = [
    (10, 1000, 1),
    (20, 1000, 1),
    (20, 1000, 3),
    (40, 1000, 3),
    (156, 1560, 3),
    (1000, 1000, 1),
]
_SHIPPED_REACH = independence._reach


def _normal_reach(critical: float, clusters: int) -> float:
    """The statistic read as a normal: the pair's sum ``critical + 0.84`` spreads from zero."""
    return critical + independence._POWER_QUANTILE


def _detectable_case(case, replicates: int) -> dict:
    clusters, rows, pairs = case
    labels = np.arange(rows) // (rows // clusters)
    out: dict = {"clusters": clusters, "rows": rows, "pairs": pairs}
    for name, reach in (("shipped", _SHIPPED_REACH), ("normal", _normal_reach)):
        independence._reach = reach
        nulls = (
            gcm_test(rng.normal(size=rows), rng.normal(size=(rows, pairs)), clusters=labels)
            for rng in (np.random.default_rng([SEED, clusters, pairs, 0, i]) for i in range(200))
        )
        target = float(np.mean([null.detectable[0, 0] for null in nulls]))
        caught = 0
        for index in range(replicates):
            rng = np.random.default_rng([SEED, clusters, pairs, 1, index])
            x, y = rng.normal(size=rows), rng.normal(size=(rows, pairs))
            y[:, 0] = target * x + math.sqrt(1.0 - target**2) * y[:, 0]
            test = gcm_test(x, y, clusters=labels, draws=DRAWS, seed=index)
            caught += test.p_value <= 0.05
        out[name] = {"detectable": round(target, 4), "caught": _rate(caught, replicates)}
    independence._reach = _SHIPPED_REACH
    return out


# ------------------------------------------------------------------------------------- protects

DT = 0.1
STEP = np.array([[0.94, 0.03], [0.0, 1.025]])
CHANNEL = np.array([0.08, -0.04])
SHOCK = 0.01
EDGES = [("demand", "supply"), ("incentive", "supply"), ("incentive", "wait"), ("supply", "wait")]
# (logger, strength, demand's push on supply): the test's market pushes 0.15, a market where demand
# matters four times as much pushes 0.6
PROTECTS = [
    ("chase", 0.0, 0.15),
    ("chase", 0.25, 0.15),
    ("chase", 0.5, 0.15),
    ("chase", 1.0, 0.15),
    ("chase", 0.0, 0.6),
    ("chase", 0.25, 0.6),
    ("chase", 0.5, 0.6),
    ("chase", 1.0, 0.6),
    ("sticky", 0.5, 0.15),
    ("sticky", 0.8, 0.15),
]


def _market(
    units: int, seed, *, push: float, chase: float = 0.0, sticky: float = 0.0
) -> dict[str, np.ndarray]:
    """tests/test_lifecycle_loop.py's market with demand's push on supply ``push``: twelve periods
    a unit, the incentive sd 1.5, plus ``chase`` times the period's demand, or keeping ``sticky``
    of the last incentive at the same spread."""
    demand_push = np.array([push, 0.0])
    rng = np.random.default_rng(seed)
    names = ("unit", "time", "supply", "wait", "incentive", "demand")
    rows: dict[str, list[float]] = {name: [] for name in names}
    for unit in range(units):
        x = rng.normal(0.0, 0.2, 2)
        incentive = rng.normal(0.0, 1.5)
        for period in range(12):
            demand = rng.normal()
            fresh = rng.normal(0.0, 1.5)
            incentive = sticky * incentive + math.sqrt(1 - sticky**2) * fresh + chase * demand
            for name, value in zip(names, (unit, period, *x, incentive, demand), strict=True):
                rows[name].append(value)
            x = STEP @ x + CHANNEL * incentive + demand_push * demand + rng.normal(0.0, SHOCK, 2)
    return {name: np.asarray(values) for name, values in rows.items()}


def _value(actions, starts, q, r, targets, push: float) -> float:
    demand_push = np.array([push, 0.0])
    noise = np.outer(demand_push, demand_push) + SHOCK**2 * np.eye(2)
    mean, covariance, total = starts.mean(axis=0), np.cov(starts, rowvar=False), 0.0
    for t, u in enumerate(actions):
        gap = mean - targets[t]
        total += 0.5 * float(np.trace(q @ covariance) + gap @ q @ gap + u @ r @ u)
        mean = STEP @ mean + CHANNEL * u[0]
        covariance = STEP @ covariance @ STEP.T + noise
    return total


_PRESCRIPTIONS: dict = {}


def _prescription(push: float):
    """One plan per market, fitted where the incentive was drawn afresh, and kept by the worker."""
    if push not in _PRESCRIPTIONS:
        import chc

        fitted_on = _market(400, [SEED, 0], push=push)
        panel = chc.Panel.from_frame(fitted_on, unit="unit", time="time", seed=0)
        _PRESCRIPTIONS[push] = chc.prescribe(
            panel,
            levers=[chc.Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05)],
            target=chc.Target("supply", value=0.3),
            constraints=[chc.Constraint("wait", hi=0.5)],
            adjustment=chc.CausalGraph.from_edges(EDGES),
            horizon=3,
            dt=DT,
            tolerance=0.5,
        )
    return _PRESCRIPTIONS[push]


def _protect(job) -> dict:
    import chc

    logging.getLogger("chc").setLevel(logging.ERROR)
    (kind, strength, push), index = job
    prescription = _prescription(push)
    later = _market(400, [SEED, 1, index], push=push, **{kind: strength})
    panel = chc.Panel.from_frame(later, unit="unit", time="time", seed=0)
    try:
        wide = prescription.evaluate(panel)
        trusted = prescription.evaluate(panel, model_error=0.0)
    except chc.InfeasibleEvaluation:
        return {"refused": True}
    problem = prescription.plan._problem
    columns = np.stack([later["supply"], later["wait"]], axis=1).reshape(400, 12, 2)
    starts = np.concatenate([columns[:, s] for s in (8, 5, 2)])
    truth = _value(
        np.asarray(prescription.plan.actions),
        starts,
        np.asarray(problem.cost.Q),
        np.asarray(problem.cost.R),
        np.asarray(problem.cost.targets(len(prescription.plan.actions))),
        push,
    )
    row: dict = {"refused": False, "rejects": bool(wide.logger_check.test.p_value <= 0.05)}
    for name, evaluation in (("default", wide), ("trusted", trusted)):
        lo, hi = evaluation.interval
        row[name] = {
            "covers": bool(lo <= truth <= hi),
            "error": (evaluation.value - truth) / ((hi - lo) / 3.92),
        }
    return row


def _summarise_protects(case, rows) -> dict:
    kept = [r for r in rows if not r["refused"]]
    n = len(kept)
    passed = [r for r in kept if not r["rejects"]]
    out: dict = {
        "logger": case[0],
        "strength": case[1],
        "demand_push": case[2],
        "replicates": len(rows),
        "refused": len(rows) - n,
        "check_rejects": _rate(sum(r["rejects"] for r in kept), n),
    }
    for name in ("default", "trusted"):
        out[name] = {
            "covers": _rate(sum(r[name]["covers"] for r in kept), n),
            "covers_where_it_passes": _rate(sum(r[name]["covers"] for r in passed), len(passed)),
            "mean_error_in_se": round(float(np.mean([r[name]["error"] for r in kept])), 3),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("section", choices=[*sorted(CASES), "detectable", "protects"])
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    default = {"size": 1000, "alternatives": 500, "power": 400, "detectable": 400, "protects": 400}
    replicates = args.replicates or default[args.section]
    if args.section == "detectable":
        results = []
        for case in DETECTABLE:
            results.append(_detectable_case(case, replicates))
            print(json.dumps(results[-1]), flush=True)
        print(json.dumps({"section": "detectable", "replicates": replicates, "results": results}))
        return
    context = multiprocessing.get_context("spawn")
    results = []
    with ProcessPoolExecutor(args.workers, mp_context=context) as pool:
        if args.section == "protects":
            for case in PROTECTS:
                jobs = [(case, i) for i in range(replicates)]
                rows = list(pool.map(_protect, jobs, chunksize=8))
                results.append(_summarise_protects(case, rows))
                print(json.dumps(results[-1]), flush=True)
        else:
            for case in CASES[args.section]:
                jobs = [(args.section, case, i) for i in range(replicates)]
                rows = list(pool.map(_replicate, jobs, chunksize=8))
                results.append(_summarise(args.section, case, rows))
                print(json.dumps(results[-1]), flush=True)
    print(json.dumps({"section": args.section, "replicates": replicates, "results": results}))


if __name__ == "__main__":
    main()
