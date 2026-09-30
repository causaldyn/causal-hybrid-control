"""The measurements behind chc.did's callaway_santanna_inference: whether diff-diff's multiplier
bootstrap holds its level on staggered panels with unit effects and errors correlated within a
unit. Rates and widths only, no wall time.

The panels have ten periods, cohorts first treated at periods 4, 6 and 8 and never-treated units,
a quarter each in the population; a unit effect that moves with the cohort, a common trend, and
AR(1) errors within a unit. Comparisons are with the not-yet-treated.

    size      no effect: how often the overall ATT's interval excludes 0 at 5%, how often the
              uniform band excludes 0 somewhere, and each event time's pointwise rejection.
    coverage  a dynamic effect that differs by cohort: how often the overall ATT's interval, the
              band and each pointwise interval cover the truth, and their median widths.
    robust    the same effect, with parallel trends failing by a line: the treated drift delta a
              period from the never-treated (delta = 0 and 0.05), who are the comparison; AR(1)
              0.5. How often robust_interval(0) covers the event study's average effect after
              treatment, which the plain average misses by delta times the mean of e + 1, and its
              median width.

Run: uv run python scripts/bench_did_inference.py {size,coverage,robust} [--replicates N]
     [--workers W]
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from chc.did import callaway_santanna_inference

SEED = 20260930
N_PERIODS = 10
COHORTS = (4, 6, 8)
Z = 1.959963984540054

# (units, AR(1) coefficient of the errors)
CASES = [(30, 0.0), (30, 0.6), (100, 0.0), (100, 0.6), (300, 0.0), (300, 0.6)]
# (units, the treated's drift a period)
ROBUST = [(30, 0.0), (30, 0.05), (100, 0.0), (100, 0.05), (300, 0.0), (300, 0.05)]


def _effect(g: int, t: int) -> float:
    return 0.0 if t < g else (1.0 + 0.25 * (t - g)) * (1.5 if g == 4 else 1.0)


def _truth() -> tuple[float, dict[int, float]]:
    """The overall ATT and the event study at the population's equal cohort shares."""
    cells = [(g, t) for g in COHORTS for t in range(g, N_PERIODS)]
    overall = sum(_effect(g, t) for g, t in cells) / len(cells)
    event = {}
    for e in range(-max(COHORTS), N_PERIODS - min(COHORTS)):
        members = [g for g in COHORTS if 0 <= g + e < N_PERIODS]
        if members and e != -1:
            event[e] = float(np.mean([_effect(g, g + e) for g in members]))
    return overall, event


def _panel(rng, units: int, rho: float, effect: bool) -> tuple[np.ndarray, np.ndarray]:
    group = rng.choice([*COHORTS, -1], size=units)
    noise = np.empty((units, N_PERIODS))
    noise[:, 0] = rng.normal(size=units) / math.sqrt(1.0 - rho**2)
    for t in range(1, N_PERIODS):
        noise[:, t] = rho * noise[:, t - 1] + rng.normal(size=units)
    level = 0.125 * np.maximum(group, 0) + rng.normal(size=units)
    outcomes = level[:, None] + 0.3 * np.arange(N_PERIODS)[None, :] + noise
    if effect:
        for i, g in enumerate(group):
            if g >= 0:
                outcomes[i] += [_effect(int(g), t) for t in range(N_PERIODS)]
    return outcomes, group


def _replicate(job) -> dict | None:
    section, case, index = job
    if section == "robust":
        return _robust(case, index)
    units, rho = case
    effect = section == "coverage"
    rng = np.random.default_rng([SEED, units, round(100 * rho), int(effect), index])
    outcomes, group = _panel(rng, units, rho, effect)
    if len(set(group.tolist())) < len(COHORTS) + 1:
        return None  # a cohort, or the never-treated, drew no unit: another estimand
    inference = callaway_santanna_inference(outcomes, group, draws=999, seed=index)
    overall, event = _truth() if effect else (0.0, dict.fromkeys(_truth()[1], 0.0))
    lo, hi = inference.overall_interval
    study = inference.estimate.event_study
    return {
        "overall": bool(lo <= overall <= hi),
        "overall_width": hi - lo,
        "band": all(inference.band[e][0] <= event[e] <= inference.band[e][1] for e in event),
        "pointwise": {e: bool(abs(study[e] - event[e]) <= Z * inference.se[e]) for e in event},
        "band_width": float(np.median([hi - lo for lo, hi in inference.band.values()])),
    }


def _robust(case, index: int) -> dict | None:
    units, delta = case
    rng = np.random.default_rng([SEED, units, round(1000 * delta), 2, index])
    outcomes, group = _panel(rng, units, 0.5, True)
    if len(set(group.tolist())) < len(COHORTS) + 1:
        return None
    outcomes += delta * np.arange(N_PERIODS)[None, :] * (group >= 0)[:, None]
    inference = callaway_santanna_inference(outcomes, group, control="never", draws=199, seed=index)
    post = {e: v for e, v in _truth()[1].items() if e >= 0}
    truth = float(np.mean(list(post.values())))
    lo, hi = inference.robust_interval(0.0)
    plain = float(np.mean([inference.estimate.event_study[e] for e in post]))
    return {"covers": bool(lo <= truth <= hi), "width": hi - lo, "plain_bias": plain - truth}


def _rate(hits: int, n: int) -> dict:
    p = hits / n
    return {"rate": round(p, 4), "se": round(math.sqrt(p * (1.0 - p) / n), 4)}


def _summarise(section: str, case, drawn: list[dict | None]) -> dict:
    rows = [r for r in drawn if r is not None]
    n = len(rows)
    if section == "robust":
        return {
            "units": case[0],
            "delta": case[1],
            "replicates": n,
            "missing_a_group": len(drawn) - n,
            "covers": _rate(sum(r["covers"] for r in rows), n),
            "median_width": round(float(np.median([r["width"] for r in rows])), 4),
            "plain_bias": round(float(np.mean([r["plain_bias"] for r in rows])), 4),
        }
    units, rho = case
    out: dict = {"units": units, "rho": rho, "replicates": n, "missing_a_group": len(drawn) - n}
    pointwise = {
        e: float(np.mean([r["pointwise"][e] for r in rows])) for e in sorted(rows[0]["pointwise"])
    }
    if section == "size":
        out["overall_rejects"] = _rate(sum(not r["overall"] for r in rows), n)
        out["band_rejects"] = _rate(sum(not r["band"] for r in rows), n)
        out["pointwise_rejects"] = [
            round(1.0 - max(pointwise.values()), 4),
            round(1.0 - min(pointwise.values()), 4),
        ]
        return out
    out["overall_covers"] = _rate(sum(r["overall"] for r in rows), n)
    out["band_covers"] = _rate(sum(r["band"] for r in rows), n)
    out["pointwise_covers"] = [round(min(pointwise.values()), 4), round(max(pointwise.values()), 4)]
    out["overall_median_width"] = round(float(np.median([r["overall_width"] for r in rows])), 4)
    out["band_median_width"] = round(float(np.median([r["band_width"] for r in rows])), 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("section", choices=["size", "coverage", "robust"])
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    context = multiprocessing.get_context("spawn")
    results = []
    with ProcessPoolExecutor(args.workers, mp_context=context) as pool:
        for case in ROBUST if args.section == "robust" else CASES:
            jobs = [(args.section, case, i) for i in range(args.replicates)]
            rows = list(pool.map(_replicate, jobs, chunksize=8))
            results.append(_summarise(args.section, case, rows))
            print(json.dumps(results[-1]), flush=True)
    print(json.dumps({"section": args.section, "replicates": args.replicates, "results": results}))


if __name__ == "__main__":
    main()
