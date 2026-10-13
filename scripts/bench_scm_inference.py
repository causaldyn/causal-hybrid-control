"""The measurements behind chc.scm's synthetic_control_inference: whether its placebo test and its
conformal interval hold their levels on factor panels, when the treated unit is drawn at random and
when a design chose it; and behind blank_period_inference, the reading a designed test gets. Rates
and widths only, no wall time.

The panels: a unit level, three AR(1) factors (coefficient 0.5) with loadings uniform on [0, 1],
and AR(1) errors with unit innovations; 20 periods before treatment and 5 after, and a constant
effect after treatment. The treated unit is drawn at random, or, in the designed case, chosen as
the unit its donors fit best before treatment, as a design that picks its test market does.

Beside the shipped placebo, each replicate computes the construction diff-diff ships, which leaves
the treated unit out of every placebo's donor pool, from the same exact weights.

    size      no effect, random assignment: 9 or 19 donors, errors AR(1) 0 or 0.5.
    power     an effect of 1 and of 2, random assignment; 19 donors, AR(1) 0.5.
    designed  no effect, the treated unit chosen; 19 donors, AR(1) 0.5.
    blank     blank_period_inference, 19 donors: the designed case with its unit chosen on the
              first 15 periods and the last 5 before treatment blank; the same chosen on all 20,
              which looks at the blank periods; drawn at random; an effect of 1 and of 2; factors
              and errors white noise, where the theorem's exchangeability holds; and 10 blank.

Each of the first three reports the placebo's rejections at 5% and 10%, both constructions and the
replicates where only one rejects; and, at alpha = 0.05 (the default) and 0.10, how often the
interval covers the effect, excludes 0, leaves out the estimate, is empty or is unbounded, and its
median width.

The blank section reports the test's rejections at 5% and 10%, and at alpha = 0.05, 0.10 and 0.20
how often each period's interval covers its effect, how many are unbounded, and their median width.

Run: uv run python scripts/bench_scm_inference.py {size,power,designed,blank} [--replicates N]
     [--workers W]
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from chc.scm import blank_period_inference, synthetic_control, synthetic_control_inference

SEED = 20260930
N_PRE = 20
N_POST = 5
ALPHAS = (0.05, 0.10)

# (donors, AR(1) coefficient of the errors, effect)
CASES = {
    "size": [(9, 0.5, 0.0), (19, 0.0, 0.0), (19, 0.5, 0.0)],
    "power": [(19, 0.5, 1.0), (19, 0.5, 2.0)],
    "designed": [(19, 0.5, 0.0)],
}
# (donors, AR(1) coefficient of the errors and the factors, effect, how the treated unit is chosen,
# blank periods): "fit" on the fitting periods, "all" on every period before treatment
BLANK_CASES = [
    (19, 0.5, 0.0, "fit", 5),
    (19, 0.5, 0.0, "all", 5),
    (19, 0.5, 0.0, "random", 5),
    (19, 0.5, 1.0, "fit", 5),
    (19, 0.5, 2.0, "fit", 5),
    (19, 0.0, 0.0, "fit", 5),
    (19, 0.5, 0.0, "fit", 10),
]
BLANK_ALPHAS = (0.05, 0.10, 0.20)


def _panel(rng: np.random.Generator, units: int, rho: float, factor_rho: float = 0.5) -> np.ndarray:
    periods = N_PRE + N_POST
    factors = np.empty((periods, 3))
    factors[0] = rng.normal(size=3) / math.sqrt(1.0 - factor_rho**2)
    for t in range(1, periods):
        factors[t] = factor_rho * factors[t - 1] + rng.normal(size=3)
    loadings = rng.uniform(0.0, 1.0, (units, 3))
    errors = np.empty((units, periods))
    errors[:, 0] = rng.normal(size=units) / math.sqrt(1.0 - rho**2)
    for t in range(1, periods):
        errors[:, t] = rho * errors[:, t - 1] + rng.normal(size=units)
    return rng.normal(size=(units, 1)) + loadings @ factors.T + errors


def _ratio(outcomes: np.ndarray, unit: int) -> float:
    fit = synthetic_control(outcomes, unit, N_PRE)
    return float(np.sqrt(np.mean(fit.att**2))) / fit.pre_rmspe


def _left_out_p_value(outcomes: np.ndarray, treated: int) -> float:
    """diff-diff's construction: each donor's synthetic control from the other donors only."""
    donors = np.delete(outcomes, treated, axis=0)
    own = _ratio(outcomes, treated)
    placebos = [_ratio(donors, j) for j in range(donors.shape[0])]
    return (1 + sum(r >= own for r in placebos)) / outcomes.shape[0]


def _replicate(job) -> dict:
    section, (donors, rho, effect), index = job
    designed = section == "designed"
    rng = np.random.default_rng(
        [SEED, donors, round(100 * rho), round(100 * effect), designed, index]
    )
    outcomes = _panel(rng, donors + 1, rho)
    if designed:
        fits = [synthetic_control(outcomes, unit, N_PRE).pre_rmspe for unit in range(donors + 1)]
        treated = int(np.argmin(fits))
    else:
        treated = int(rng.integers(donors + 1))
    outcomes[treated, N_PRE:] += effect
    inference = synthetic_control_inference(outcomes, treated, N_PRE)
    row = {"p": inference.placebo_p_value, "p_left_out": _left_out_p_value(outcomes, treated)}
    for alpha in ALPHAS:
        found = synthetic_control_inference(outcomes, treated, N_PRE, alpha=alpha)
        lo, hi = found.interval
        row[alpha] = {
            "covers": bool(lo <= effect <= hi),
            "excludes_zero": bool(not lo <= 0.0 <= hi),
            "leaves_out_estimate": bool(not lo <= found.estimate.overall <= hi),
            "empty": math.isnan(lo),
            "unbounded": math.isinf(lo) or math.isinf(hi),
            "width": hi - lo,
        }
    return row


def _blank_replicate(job) -> dict:
    (donors, rho, effect, chosen, n_blank), index = job
    rng = np.random.default_rng(
        [
            SEED,
            donors,
            round(100 * rho),
            round(100 * effect),
            n_blank,
            index,
            1 + ("fit", "all", "random").index(chosen),
        ]
    )
    outcomes = _panel(rng, donors + 1, rho, factor_rho=rho)
    if chosen == "random":
        treated = int(rng.integers(donors + 1))
    else:
        n_seen = N_PRE - n_blank if chosen == "fit" else N_PRE
        fits = [synthetic_control(outcomes, unit, n_seen).pre_rmspe for unit in range(donors + 1)]
        treated = int(np.argmin(fits))
    outcomes[treated, N_PRE:] += effect
    row: dict = {}
    for alpha in BLANK_ALPHAS:
        found = blank_period_inference(outcomes, treated, N_PRE, n_blank, alpha=alpha)
        row["p"] = found.p_value
        lo, hi = found.intervals.T
        row[alpha] = {
            "covers": int(np.sum((lo <= effect) & (effect <= hi))),
            "unbounded": int(np.sum(np.isinf(hi))),
            "widths": [float(w) for w in hi - lo if math.isfinite(w)],
        }
    return row


def _blank_summary(case, rows: list[dict]) -> dict:
    donors, rho, effect, chosen, n_blank = case
    n = len(rows)
    out: dict = {
        "donors": donors,
        "rho": rho,
        "effect": effect,
        "chosen": chosen,
        "n_blank": n_blank,
        "replicates": n,
        "arrangements": math.comb(n_blank + N_POST, N_POST),
    }
    for level in (0.05, 0.10):
        out[f"rejects_{level:.2f}"] = _rate(sum(r["p"] <= level for r in rows), n)
    for alpha in BLANK_ALPHAS:
        widths = [w for r in rows for w in r[alpha]["widths"]]
        out[f"interval_{alpha:.2f}"] = {
            "covers": _rate(sum(r[alpha]["covers"] for r in rows), n * N_POST),
            "unbounded": sum(r[alpha]["unbounded"] for r in rows),
            "median_width": round(float(np.median(widths)), 4) if widths else None,
        }
    return out


def _rate(hits: int, n: int) -> dict:
    p = hits / n
    return {"rate": round(p, 4), "se": round(math.sqrt(p * (1.0 - p) / n), 4)}


def _summarise(case, rows: list[dict]) -> dict:
    donors, rho, effect = case
    n = len(rows)
    out: dict = {"donors": donors, "rho": rho, "effect": effect, "replicates": n}
    for level in (0.05, 0.10):
        shipped = [r["p"] <= level + 1e-12 for r in rows]
        left_out = [r["p_left_out"] <= level + 1e-12 for r in rows]
        out[f"placebo_rejects_{level:.2f}"] = _rate(sum(shipped), n)
        out[f"left_out_rejects_{level:.2f}"] = _rate(sum(left_out), n)
        out[f"only_left_out_rejects_{level:.2f}"] = sum(
            b and not a for a, b in zip(shipped, left_out, strict=True)
        )
        out[f"only_placebo_rejects_{level:.2f}"] = sum(
            a and not b for a, b in zip(shipped, left_out, strict=True)
        )
    for alpha in ALPHAS:
        found = [r[alpha] for r in rows]
        key = f"interval_{alpha:.2f}"
        finite = [f["width"] for f in found if math.isfinite(f["width"])]
        out[key] = {
            "covers": _rate(sum(f["covers"] for f in found), n),
            "excludes_zero": _rate(sum(f["excludes_zero"] for f in found), n),
            "leaves_out_estimate": _rate(sum(f["leaves_out_estimate"] for f in found), n),
            "empty": sum(f["empty"] for f in found),
            "unbounded": sum(f["unbounded"] for f in found),
            "median_width": round(float(np.median(finite)), 4) if finite else None,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("section", choices=[*sorted(CASES), "blank"])
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    context = multiprocessing.get_context("spawn")
    results = []
    with ProcessPoolExecutor(args.workers, mp_context=context) as pool:
        if args.section == "blank":
            for case in BLANK_CASES:
                jobs = [(case, i) for i in range(args.replicates)]
                rows = list(pool.map(_blank_replicate, jobs, chunksize=8))
                results.append(_blank_summary(case, rows))
                print(json.dumps(results[-1]), flush=True)
        for case in CASES.get(args.section, []):
            jobs = [(args.section, case, i) for i in range(args.replicates)]
            rows = list(pool.map(_replicate, jobs, chunksize=8))
            results.append(_summarise(case, rows))
            print(json.dumps(results[-1]), flush=True)
    print(json.dumps({"section": args.section, "replicates": args.replicates, "results": results}))


if __name__ == "__main__":
    main()
