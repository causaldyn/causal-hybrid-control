"""The measurements behind docs/adr/0047-the-geos-spread-by-the-marginal-likelihood.md: how well
the discount DLM over geos recovers each geo's media effect, and the spread of the geos, on worlds
chc.geo_world draws. Counts and coverage only, no wall time.

The worlds: ``chc.geo_world.GeoMediaMix`` of four channels over 12 geos and 104 weeks, its
defaults otherwise (populations log-normal about 1e5 with log-SD 1, no spillover, no drift):

    search  Michaelis-Menten at 2 a head, effect 3, spread 0.4, retention 0.3, spend 1.5
    social  exponential at 1, effect 1.5, spread 0.5, retention 0.5, spend 0.8
    video   tanh at 3, effect 2, spread 0.3, retention 0.6, spend 1.2
    tv      tanh at 4, effect 2, spread 0.2, retention 0.7, spend 2, bought nationally

at a noise a head of 30 and of 100, ``--replicates`` worlds each from seed ``--seed`` on. Each
world in two hierarchies:

    lognormal   as drawn: a geo's effect log-normal about the channel's median, log-SD its spread
    gaussian    the control: the effects redrawn normal with the log-normal's mean and variance,
                from a stream of their own, and the KPI rebuilt from them, so that the model's
                hierarchy is the world's

The model knows the media's shape: a geo's regressor on a channel is the channel's lift there
with the geo's effect divided out, so its coefficient is the effect. National coefficients, and
each geo's level, annual season and deviations from the national coefficients, every block
static; each geo's variance in proportion to its population, the prior's guess of it each geo's
least-squares residual variance over its share; vague priors on the KPI's own scale, the
deviations' prior variance the vaguest spread. A geo's effect is read off the last step.

The guess matters. A prior's covariance is stated at its guess of the variance, so a guess of the
KPI's whole variance, many times the noise's at a low noise, tightens the coefficients' prior by as
much once the variance is learned. With that guess a first pilot's vaguest spread fell below the
hierarchy's own variance on some worlds.

The effects, five ways:

    pooled      at chc.dlm.fit_geo_spread's spread of each channel's deviations
    integrated  mixed over the spread's 64 draws (GeoSpread.mixture)
    alone       no pooling: the deviations at the vaguest spread
    together    complete pooling: the deviations at 1e-12 of it
    true        at the world's own spread, the variance of its hierarchy: the reference

Reported, at the 90% level: each way's coverage of the effects, by channel, and the root mean
square error over every geo and channel; each channel's spread interval's coverage of the variance
of its hierarchy, ``mean^2 (exp(tau^2) - 1)`` for a log-normal of log-SD ``tau``, and the median
ratio of the estimate to it; the draws' effective number.

Pre-registered 2026-10-02, before the scored run: 100 worlds an arm from seed 20261002, each of
the four arms (two hierarchies at two noises) its own process. The pilot's worlds were seeds 0 to
19. The gate is met when, in every arm:

    G1  pooling recovers the effects: the pooled root mean square error is below each geo's alone
        and below complete pooling's, each paired difference's 95% interval below 0;
    G2  the integrated intervals cover the effects in at least 0.87 of the geos and channels;
    G3  the hierarchy's variance is recovered: each spread interval covers it in at least 0.80 of
        the worlds, averaged over the channels, and each channel's median ratio of the estimate to
        it is within 0.5 and 2.

The thresholds sit below the nominal 0.90 by the pilot's shortfall and a margin: at 100 worlds the
standard error of an arm's coverage is 0.005-0.007 for the effects and 0.015-0.022 for the
spreads. A spread is a variance of 12 geos: were the effects observed, a 90% interval's ends would
be a factor of 4.3 apart, chi2_11's 95% point over its 5%.

Predicted from the pilot, 20 worlds an arm, and not gated:

    the integrated coverage 0.88-0.89 in every arm; the pooled one, which takes the spread as
        known, 0.83 at a noise of 100 and 0.87-0.88 at 30;
    the pooled error 34-36% below alone at a noise of 100 and 7-11% at 30, and within 4% of the
        error at the world's own spread;
    the spread's coverage 0.82-0.89; the median ratio near 0.86, the median of chi2_11 / 12, in
        the gaussian arms, and lower for the most skewed channels in the lognormal ones, whose
        hierarchy has an excess kurtosis of 3.3 and 5.9 for search and social;
    a median of 40-44 effective draws of 64; the pilot's least was 16.

Run: uv run python scripts/bench_geo_dlm.py [--replicates 100] [--seed S] [--noise 30]
     [--hierarchy lognormal] > out.json
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import jax
import numpy as np
from scipy import linalg, stats

from chc.dlm import (
    GeoDLM,
    GeoDLMFit,
    Polynomial,
    Prior,
    Regression,
    Seasonal,
    fit_geo_spread,
    forward_filter_geos,
)
from chc.geo_world import GeoMediaMix, GeoMediaWorld, MediaChannel
from chc.response import Exponential, MichaelisMenten, Tanh

jax.config.update("jax_enable_x64", True)

SEED = 20261002
LEVEL = 0.90
DRAWS = 64
NOISES = (30.0, 100.0)
HIERARCHIES = ("lognormal", "gaussian")
CHANNELS = (
    MediaChannel("search", MichaelisMenten(2.0), 3.0, spread=0.4, retention=0.3, spend=1.5),
    MediaChannel("social", Exponential(1.0), 1.5, spread=0.5, retention=0.5, spend=0.8),
    MediaChannel("video", Tanh(3.0), 2.0, spread=0.3, retention=0.6, spend=1.2),
    MediaChannel("tv", Tanh(4.0), 2.0, spread=0.2, retention=0.7, spend=2.0, national=True),
)
GEOS, WEEKS = 12, 104
WAYS = ("pooled", "integrated", "alone", "together", "true")
_C = len(CHANNELS)
_REGIONAL = 3 + _C  # a level, a season's two coordinates, the deviations
_TAU = np.array([c.spread for c in CHANNELS])
_MEDIAN = np.array([c.effect for c in CHANNELS])
MEAN = _MEDIAN * np.exp(_TAU**2 / 2.0)
VARIANCE = MEAN**2 * (np.exp(_TAU**2) - 1.0)
_OWN = np.array([[_C + g * _REGIONAL + 3 + c for c in range(_C)] for g in range(GEOS)])


def _variance(y: np.ndarray, x: np.ndarray, relative: np.ndarray) -> float:
    """The prior's guess of the variance at the mean population: each geo's least-squares
    residual variance on a level, the season and the regressors, over its share."""
    angle = 2.0 * np.pi * np.arange(WEEKS) / 52.0
    shares = []
    for g in range(GEOS):
        design = np.column_stack([np.ones(WEEKS), np.cos(angle), np.sin(angle), x[:, g]])
        coefficients = np.linalg.lstsq(design, y[:, g], rcond=None)[0]
        residual = y[:, g] - design @ coefficients
        shares.append(residual @ residual / (WEEKS - design.shape[1]) / relative[g])
    return float(np.mean(shares))


def _model(world: GeoMediaWorld, effect: np.ndarray) -> tuple[GeoDLM, np.ndarray, np.ndarray]:
    """The model, the KPI and the regressors, ``(weeks, geos)`` and ``(weeks, geos, channels)``."""
    x = np.transpose(world.media / world.effect[:, None, :], (1, 0, 2))
    y = (world.base + world.spilled + world.noise).T + np.einsum("tgc,gc->tg", x, effect)
    relative = world.population / world.population.mean()
    scale = float(np.var(y - y.mean(axis=0)))
    vaguest = 100.0 * scale / np.mean(x**2, axis=(0, 1))
    blocks = [np.diag(vaguest)]
    means = [np.zeros(_C)]
    for g in range(GEOS):
        level = float(y[:, g].mean())
        blocks.append(np.diag([100.0 * y[:, g].var() + level**2, 100.0 * scale, 100.0 * scale]))
        blocks.append(np.diag(vaguest))
        means += [np.array([level, 0.0, 0.0]), np.zeros(_C)]
    model = GeoDLM(
        (Regression(_C, 1.0),),
        (Polynomial(1, 1.0), Seasonal(52.0, (1,), 1.0), Regression(_C, 1.0)),
        GEOS,
        Prior(np.concatenate(means), linalg.block_diag(*blocks), _variance(y, x, relative), 1.0),
        relative_variance=relative,
    )
    return model, y, x


def _effects(fit: GeoDLMFit) -> tuple[np.ndarray, np.ndarray]:
    """Each geo's effect on each channel, ``(geos, channels)``: the national coefficient plus the
    geo's deviation at the last step, and its variance."""
    m, c = fit.mean[-1], fit.covariance[-1]
    national = np.arange(_C)[None, :]
    return (
        m[national] + m[_OWN],
        c[national, national] + c[_OWN, _OWN] + 2.0 * c[national, _OWN],
    )


def replicate(noise: float, hierarchy: str, seed: int) -> dict[str, Any]:
    world = GeoMediaMix(CHANNELS, geos=GEOS, weeks=WEEKS, noise=noise).draw(seed)
    effect = world.effect
    if hierarchy == "gaussian":
        z = np.random.default_rng((seed, 1)).standard_normal((GEOS, _C))
        effect = MEAN + np.sqrt(VARIANCE) * z
    model, y, x = _model(world, effect)
    spread = fit_geo_spread(model, y, x, range(3, _REGIONAL), DRAWS, seed, level=LEVEL)
    vaguest = np.diag(model.prior.covariance)[_OWN[0]]
    # a spread's prior variance is stated at the prior's scale, the hierarchy's at V's
    at_truth = VARIANCE * model.prior.scale / (noise**2 * float(world.population.mean()))
    ways = {
        "pooled": _effects(spread.fit),
        "integrated": spread.mixture(_effects),
        "alone": _effects(forward_filter_geos(model, y, x)),
        "together": _effects(forward_filter_geos(spread.at(1e-12 * vaguest), y, x)),
        "true": _effects(forward_filter_geos(spread.at(at_truth), y, x)),
    }
    quantile = float(stats.t.ppf(0.5 + LEVEL / 2.0, spread.fit.dof[-1]))
    units = spread.fit.scale[-1] / model.prior.scale
    return {
        "seed": seed,
        "covered": {
            way: (np.abs(m - effect) <= quantile * np.sqrt(v)).mean(axis=0).tolist()
            for way, (m, v) in ways.items()
        },
        "rmse": {way: float(np.sqrt(np.mean((m - effect) ** 2))) for way, (m, _) in ways.items()},
        "spread_covered": (
            (spread.lower * units <= VARIANCE) & (spread.upper * units >= VARIANCE)
        ).tolist(),
        "spread_ratio": (spread.variance * units / VARIANCE).tolist(),
        "effective_draws": float(1.0 / np.sum(spread.weights**2)),
    }


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(values: list[Any]) -> list[float]:
        return np.round(np.mean(values, axis=0), 4).tolist()

    covered = {way: mean([r["covered"][way] for r in rows]) for way in WAYS}
    spread_covered = mean([r["spread_covered"] for r in rows])
    paired = {
        way: np.array([r["rmse"]["pooled"] - r["rmse"][way] for r in rows])
        for way in ("alone", "together")
    }
    return {
        "covered": covered,
        "covered_over_channels": {way: float(np.mean(c)) for way, c in covered.items()},
        "rmse": {way: float(np.mean([r["rmse"][way] for r in rows])) for way in WAYS},
        "pooled_minus": {
            way: {
                "mean": float(d.mean()),
                "half_width_95": float(1.96 * d.std(ddof=1) / math.sqrt(d.size)),
            }
            for way, d in paired.items()
        },
        "spread_covered": spread_covered,
        "spread_covered_over_channels": float(np.mean(spread_covered)),
        "spread_ratio_median": np.round(
            np.median([r["spread_ratio"] for r in rows], axis=0), 4
        ).tolist(),
        "effective_draws_median": float(np.median([r["effective_draws"] for r in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--noise", type=float, choices=NOISES, action="append")
    parser.add_argument("--hierarchy", choices=HIERARCHIES, action="append")
    args = parser.parse_args()
    out: dict[str, object] = {
        "replicates": args.replicates,
        "seed": args.seed,
        "level": LEVEL,
        "draws": DRAWS,
        "x64": bool(jax.config.jax_enable_x64),
    }
    for noise in args.noise or NOISES:
        for hierarchy in args.hierarchy or HIERARCHIES:
            rows = [replicate(noise, hierarchy, args.seed + i) for i in range(args.replicates)]
            out[f"{hierarchy}_{noise:g}"] = {"summary": summary(rows), "rows": rows}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
