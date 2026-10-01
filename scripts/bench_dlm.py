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

MM9, the discounts' uncertainty carried by projection. Pre-registered 2026-10-02, before any scored
run of it, on a pilot of 100 fresh series of each world drawn outside this script's streams:

    projected    the union of the 90% intervals of every grid pair in
                 ``chc.dlm.confidence_set(fits, 2)``, the likelihood ratio at 0.95;

scored on the random-walk world's series, the same as the rows above (its draws come from a
stream of their own, so the rows above do not move), and on a third world:

    discount     156 steps of the discount model itself at 0.9 and 0.9, with an explicit state
                 path: ``W_t`` is the blockwise ``(1 - delta) / delta`` times ``C_(t-1)`` of the
                 covariance recursion at the known variance 0.25, started where 104 steps of
                 other media settle it. Media and noise as in the random walk, from a stream of
                 its own.

Gate: each channel's contribution over the last 13 weeks covered in at least 0.88 of the series,
in both worlds. Below 0.85 on either is a failure of the method, not a near miss. Predicted from
the pilot: 0.91 and 0.93 on the random walk, 0.93 and 0.99 on the discount world, intervals about
1.2 times as wide as at the true discounts. What the pilot found on the way: at the true law and
evolution variance the random walk's intervals cover 0.89 and 0.86, at the likelihood's pick of
that same law 0.78 and 0.77, so the loss is the selection of a weakly identified evolution
variance and not the discount law; on the discount world the pick covers 0.87 and 0.95. The other
candidates covered the random walk at 0.79 and 0.80 (the likelihood-weighted mixture over the
grid), 0.84 and 0.84 (the pick's interval at the level whose coverage over 40 series simulated
from the fitted model is 0.90), 0.73 and 0.71 (the discounts chosen by the 13-step-ahead score)
and 0.79 and 0.81 (the set's draws pooled in place of the union).

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
    confidence_set,
    forward_filter,
)

SEED = 20260929
LEVEL = 0.90
BURN = 20
GRID = (0.8, 0.85, 0.9, 0.95, 0.98, 1.0)
FIXED = (0.85, 0.9, 0.95, 0.98)
WINDOW = 13
DRAWS = 1000
NOISE = 0.25  # the random walk's and the discount world's observational variance


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


def _bounds(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.quantile(samples, [0.5 - LEVEL / 2.0, 0.5 + LEVEL / 2.0], axis=0)
    return lo, hi


def _projected(
    fits: list[DLMFit], x: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, int]:
    """The union of the 90% intervals over the grid pairs the likelihood ratio keeps."""
    members = confidence_set(fits, 2)
    los, his = zip(
        *(_bounds(_contributions(fits[i], x, DRAWS, int(rng.integers(2**31)))) for i in members),
        strict=True,
    )
    return np.min(los, axis=0), np.max(his, axis=0), len(members)


def walk(replicates: int, rng: np.random.Generator, seed: int) -> dict[str, object]:
    horizon = 156
    one_hits = one_count = 0
    projected_hits = np.zeros(2)
    projected_widths: list[np.ndarray] = []
    set_sizes: list[int] = []
    picked_hits = np.zeros(2)
    averaged_hits = np.zeros(2)
    burned_hits = np.zeros(2)
    gaps: list[float] = []
    fixed_hits = {d: np.zeros(2) for d in FIXED}
    widths = np.zeros(2)
    chosen: list[tuple[float, float]] = []
    pairs = list(itertools.product(GRID, GRID))
    for replicate in range(replicates):
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

        aside = np.random.default_rng([seed, 2, replicate])
        lo_set, hi_set, size = _projected([fits[pair] for pair in pairs], x, aside)
        projected_hits += (lo_set <= truth) & (truth <= hi_set)
        projected_widths.append((hi_set - lo_set) / (hi - lo))
        set_sizes.append(size)
    picked = np.array(chosen)
    return {
        "one_step_coverage": one_hits / one_count,
        "steps": one_count,
        "contribution_coverage": {
            "picked": (picked_hits / replicates).tolist(),
            "averaged": (averaged_hits / replicates).tolist(),
            "averaged_after_burn_in": (burned_hits / replicates).tolist(),
            **{f"fixed_{d}": (h / replicates).tolist() for d, h in fixed_hits.items()},
            "projected": (projected_hits / replicates).tolist(),
        },
        "projected_width_over_picked_median": np.median(projected_widths, axis=0).tolist(),
        "projected_set_size_median": float(np.median(set_sizes)),
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


def _discount_path(
    x: np.ndarray, rates: tuple[float, float], c: np.ndarray, rng: np.random.Generator | None
) -> tuple[np.ndarray, np.ndarray]:
    """A level and two coefficients moving as the discount model says, along ``x``: the
    covariance recursion at the known variance gives each step's ``W_t``. Without ``rng`` only
    the covariance is run, to where it settles."""
    theta = np.array([10.0, 1.0, 1.0])
    path = np.empty((x.shape[0], 3))
    for t in range(x.shape[0]):
        w = np.zeros((3, 3))
        w[0, 0] = rates[0] * c[0, 0]
        w[1:, 1:] = rates[1] * c[1:, 1:]
        r = c + w
        f = np.array([1.0, *x[t]])
        q = f @ r @ f + NOISE
        gain = r @ f / q
        c = r - np.outer(gain, gain) * q
        if rng is not None:
            theta = theta + np.linalg.cholesky(w) @ rng.standard_normal(3)
        path[t] = theta
    return path, c


def discount(replicates: int, seed: int) -> dict[str, object]:
    horizon, true_pair = 156, (0.9, 0.9)
    rates = ((1.0 - true_pair[0]) / true_pair[0], (1.0 - true_pair[1]) / true_pair[1])
    rng = np.random.default_rng([seed, 1])
    pairs = list(itertools.product(GRID, GRID))
    picked_hits, projected_hits, reference_hits = np.zeros(2), np.zeros(2), np.zeros(2)
    widths: list[np.ndarray] = []
    chosen: list[float] = []
    for _ in range(replicates):
        settle_x = rng.gamma(2.0, 1.0, (104, 2)) * (rng.random((104, 2)) > 0.25)
        _, start = _discount_path(settle_x, rates, np.diag([100.0, 10.0, 10.0]), None)
        x = rng.gamma(2.0, 1.0, (horizon, 2)) * (rng.random((horizon, 2)) > 0.25)
        path, _ = _discount_path(x, rates, start, rng)
        y = path[:, 0] + np.sum(x * path[:, 1:], axis=1)
        y = y + math.sqrt(NOISE) * rng.standard_normal(horizon)
        truth = np.sum(x[-WINDOW:] * path[-WINDOW:, 1:], axis=0)
        prior = Prior(np.zeros(3), np.diag([100.0, 10.0, 10.0]), 1.0, 1.0)
        fits = {pair: _fit(y, x, prior, *pair) for pair in pairs}
        loglik = np.array([fits[pair].log_likelihood for pair in pairs])
        best = pairs[int(np.argmax(loglik))]
        chosen.append(best[1])
        picked_hits += _hit(_contributions(fits[best], x, DRAWS, int(rng.integers(2**31))), truth)
        reference = _contributions(fits[true_pair], x, DRAWS, int(rng.integers(2**31)))
        reference_hits += _hit(reference, truth)
        lo, hi, _ = _projected([fits[pair] for pair in pairs], x, rng)
        projected_hits += (lo <= truth) & (truth <= hi)
        ref_lo, ref_hi = _bounds(reference)
        widths.append((hi - lo) / (ref_hi - ref_lo))
    return {
        "contribution_coverage": {
            "picked": (picked_hits / replicates).tolist(),
            "projected": (projected_hits / replicates).tolist(),
            "at_the_true_discounts": (reference_hits / replicates).tolist(),
        },
        "projected_width_over_true_median": np.median(widths, axis=0).tolist(),
        "picked_discount_coefficient_counts": {str(d): chosen.count(d) for d in GRID},
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
        out["walk"] = walk(args.replicates, rng, args.seed)
        out["discount"] = discount(args.replicates, args.seed)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
