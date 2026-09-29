"""The measurements behind docs/adr/0022-a-discount-dynamic-linear-model.md: how often the
discount DLM's one-step forecasts and smoothed contributions cover the truth, and how the answer
depends on how the discounts were chosen. Counts and coverage only, no wall time.

Two worlds, each over ``--replicates`` series:

    own      104 steps of a level and one media coefficient, each ``y_t`` drawn from the model's
             own one-step forecast given the steps before it (the filter re-run on the prefix).
             Coverage is nominal by construction, so a miss is a bug in what the filter
             reports, not a property of the method.
    walk     156 weekly steps: a level and two media coefficients that follow Gaussian random
             walks with a fixed step variance, which no discount reproduces exactly, and noise of
             variance 0.25. Media are gamma draws, zero in a quarter of the weeks.

Reported, at the 90% level: for ``own``, one-step coverage after a 20-step burn-in, at three more
levels, and a Kolmogorov-Smirnov test of the probability integral transforms; for ``walk``,
one-step coverage and the coverage of each channel's contribution over the last 13 weeks, from
1000 backward draws, with the discounts

    picked       the pair of a 6 x 6 grid with the highest one-step log-likelihood;
    averaged     every pair of the grid, weighted by its likelihood (a flat prior on the grid),
                 and the same with the first 20 steps left out of the likelihood;
    fixed        the coefficients' discount fixed, the level's picked;

and how far the picked pair's log-likelihood is above the same level's with the coefficients'
discount at 0.9.

A coverage's Monte Carlo standard error is ``sqrt(0.9 * 0.1 / count)``.

Run: uv run python scripts/bench_dlm.py [--replicates 500] [--seed S] [--own-only] > out.json

The own world draws first, so ``--own-only`` reproduces a full run's own world at the same seed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math

import numpy as np
from scipy import stats

from chc.dlm import (
    DLMFit,
    DynamicLinearModel,
    Polynomial,
    Prior,
    Regression,
    backward_sample,
    forward_filter,
)

SEED = 20260929
LEVEL = 0.90
BURN = 20
GRID = (0.8, 0.85, 0.9, 0.95, 0.98, 1.0)
FIXED = (0.85, 0.9, 0.95, 0.98)
WINDOW = 13
DRAWS = 1000


def _pit(fit: DLMFit, start: int) -> np.ndarray:
    u = fit.errors[start:] / np.sqrt(fit.one_step_scale[start:])
    return stats.t.cdf(u, fit.one_step_dof[start:])


def _covered(pit: np.ndarray, level: float = LEVEL) -> np.ndarray:
    return np.abs(pit - 0.5) <= level / 2.0


def own(replicates: int, rng: np.random.Generator) -> dict[str, object]:
    horizon = 104
    model = DynamicLinearModel(
        (Polynomial(1, 0.95), Regression(1, 0.9)),
        Prior(np.array([10.0, 1.0]), np.diag([4.0, 0.25]), 0.5, 5.0),
        variance_discount=0.98,
    )
    pits = []
    for _ in range(replicates):
        x = rng.gamma(2.0, 1.0, horizon) * (rng.random(horizon) > 0.25)
        y = np.full(horizon, np.nan)
        for t in range(horizon):
            step = forward_filter(model, y[: t + 1], x[: t + 1])
            y[t] = step.one_step_mean[t] + math.sqrt(step.one_step_scale[t]) * rng.standard_t(
                step.one_step_dof[t]
            )
        pits.append(_pit(forward_filter(model, y, x), BURN))
    pit = np.concatenate(pits)
    return {
        "one_step_coverage": float(_covered(pit).mean()),
        "coverage_at": {str(lv): float(_covered(pit, lv).mean()) for lv in (0.5, 0.8, 0.95)},
        "pit_ks_pvalue": float(stats.kstest(pit, "uniform").pvalue),
        "steps": int(pit.size),
    }


def _contributions(fit: DLMFit, x: np.ndarray, draws: int, seed: int) -> np.ndarray:
    window = slice(x.shape[0] - WINDOW, x.shape[0])
    states = backward_sample(fit, draws, seed=seed).states
    return np.einsum("tj,dtj->dj", x[window], states[:, window, 1:])


def _hit(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    lo, hi = np.quantile(samples, [0.5 - LEVEL / 2.0, 0.5 + LEVEL / 2.0], axis=0)
    return (lo <= truth) & (truth <= hi)


def _fit(y: np.ndarray, x: np.ndarray, prior: Prior, d_level: float, d_coef: float) -> DLMFit:
    blocks = (Polynomial(1, d_level), Regression(2, d_coef))
    return forward_filter(DynamicLinearModel(blocks, prior), y, x)


def walk(replicates: int, rng: np.random.Generator) -> dict[str, object]:
    horizon = 156
    one_hits = one_count = 0
    picked_hits = np.zeros(2)
    averaged_hits = np.zeros(2)
    burned_hits = np.zeros(2)
    gaps: list[float] = []
    fixed_hits = {d: np.zeros(2) for d in FIXED}
    widths = np.zeros(2)
    chosen: list[tuple[float, float]] = []
    pairs = list(itertools.product(GRID, GRID))
    for _ in range(replicates):
        level = 10.0 + np.cumsum(0.05 * rng.standard_normal(horizon))
        coef = 1.0 + np.cumsum(0.02 * rng.standard_normal((horizon, 2)), axis=0)
        x = rng.gamma(2.0, 1.0, (horizon, 2)) * (rng.random((horizon, 2)) > 0.25)
        y = level + np.sum(x * coef, axis=1) + 0.5 * rng.standard_normal(horizon)
        truth = np.sum(x[-WINDOW:] * coef[-WINDOW:], axis=0)
        prior = Prior(np.zeros(3), np.diag([100.0, 10.0, 10.0]), 1.0, 1.0)
        fits = {pair: _fit(y, x, prior, *pair) for pair in pairs}
        loglik = np.array([fits[pair].log_likelihood for pair in pairs])
        best = pairs[int(np.argmax(loglik))]
        chosen.append(best)
        covered = _covered(_pit(fits[best], BURN))
        one_hits += int(covered.sum())
        one_count += covered.size

        sampled = _contributions(fits[best], x, DRAWS, int(rng.integers(2**31)))
        picked_hits += _hit(sampled, truth)
        lo, hi = np.quantile(sampled, [0.5 - LEVEL / 2.0, 0.5 + LEVEL / 2.0], axis=0)
        widths += (hi - lo) / np.abs(truth)

        gaps.append(fits[best].log_likelihood - fits[(best[0], 0.9)].log_likelihood)
        burned = np.array([np.nansum(fits[pair].log_scores[BURN:]) for pair in pairs])
        for scores, hits in ((loglik, averaged_hits), (burned, burned_hits)):
            weights = np.exp(scores - scores.max())
            counts = rng.multinomial(DRAWS, weights / weights.sum())
            mixed = [
                _contributions(fits[pair], x, int(c), int(rng.integers(2**31)))
                for pair, c in zip(pairs, counts, strict=True)
                if c
            ]
            hits += _hit(np.concatenate(mixed), truth)

        for d in FIXED:
            held = fits[(best[0], d)]
            fixed_hits[d] += _hit(_contributions(held, x, DRAWS, int(rng.integers(2**31))), truth)
    picked = np.array(chosen)
    return {
        "one_step_coverage": one_hits / one_count,
        "steps": one_count,
        "contribution_coverage": {
            "picked": (picked_hits / replicates).tolist(),
            "averaged": (averaged_hits / replicates).tolist(),
            "averaged_after_burn_in": (burned_hits / replicates).tolist(),
            **{f"fixed_{d}": (h / replicates).tolist() for d, h in fixed_hits.items()},
        },
        "picked_relative_width": (widths / replicates).tolist(),
        "picked_discount_level_mean": float(picked[:, 0].mean()),
        "picked_loglik_above_coefficient_0.9": {
            "median": float(np.median(gaps)),
            "share_below_2": float(np.mean(np.array(gaps) < 2.0)),
        },
        "picked_discount_coefficient_counts": {
            str(d): int(np.sum(picked[:, 1] == d)) for d in GRID
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--own-only", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    out: dict[str, object] = {
        "replicates": args.replicates,
        "seed": args.seed,
        "level": LEVEL,
        "own": own(args.replicates, rng),
    }
    if not args.own_only:
        out["walk"] = walk(args.replicates, rng)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
