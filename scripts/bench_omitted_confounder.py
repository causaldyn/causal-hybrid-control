"""The measurements behind MM7's omitted-confounder bound, chc.dynamics_id.omitted_confounder_bound:
whether its confidence bounds cover the channel at the shares the confounder it leaves out has.
Rates only, no wall time.

Each world is one state, two levers, an observed confounder the fit adjusts for and one latent it
leaves out, all linear and Gaussian, with the loadings drawn afresh: the channel and the latent's
pull on each lever and on the rate from standard normals, the rate's own noise from 0.2 to 1. Half
the worlds are logged at ``dt = 0.05`` and read by Euler, half at a step from 0.05 to 1.4 and read
by ``rk4``, the drift at -0.5. The functional is the reallocation, lever b's effect less lever a's.
With one latent the bound is attained (``validation/omitted_confounder_bound.mac``, STEP 2), so at
the latent's shares the bound on the side the bias points to sits on the truth, and its confidence
bound covers it as often as its sampling error is right: 95% one-sided, where a bound that is only
valid would cover every time.

    truth   the shares the latent has, from the world's population: how often the two confidence
            bounds cover the truth together, the one the bias points to alone, and its point bound
    half    half of each share, an analyst who understates the latent
    none    shares of zero: the fit's own one-sided intervals, which the latent biases

Run: uv run python scripts/bench_omitted_confounder.py [--worlds 500] [--n 2000] > out.json

Measured 2026-09-30, 500 worlds of 2000 transitions at the default seed, whose shares ran from
0.006 to 0.73 for ``cf_y`` and from 0.022 to 0.81 for ``cf_d`` between their 10th and 90th
percentiles, with medians 0.18 and 0.33:

* truth: the two confidence bounds covered the truth together in 0.940 of the worlds
  (Clopper-Pearson 0.915-0.959), and every miss was on the side the bias points to; under Euler
  0.932 (0.893-0.960), under ``rk4`` 0.948 (0.913-0.972). That side's point bound covered it in
  0.472 (0.428-0.517), about the half a bound that sits on the truth should.
* half: 0.216 (0.181-0.255).
* none: 0.138 (0.109-0.171).
"""

from __future__ import annotations

import argparse
import json
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("JAX_ENABLE_X64", "1")  # the tests' precision

import jax.numpy as jnp
import numpy as np
from scipy import stats

from chc.dynamics_id import fit_causal_residual, omitted_confounder_bound

SEED = 20260930
DRIFT, ETA = -0.5, 0.5  # the drift, and each lever's own noise
TO_ACTION, TO_RATE = 0.8, 0.7  # the observed confounder's loadings
MOVE = np.array([-1.0, 1.0])  # lever a's spend moved to lever b
LEVEL = 0.95


def _known(t: float, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros_like(x)


def _t4(z: float) -> float:
    return 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0


def _s4(z: float) -> float:
    return 1.0 + z / 2.0 + z**2 / 6.0 + z**3 / 24.0


def _world(rng: np.random.Generator, n: int, scheme: str) -> dict:
    """One log and its population: the truth, the latent's bias on the move and its two shares.
    The rate's noise rides on the held input, so it is in the rate's units under both readings."""
    channel, to_action = rng.standard_normal(2), rng.standard_normal(2)
    to_rate, noise = float(rng.standard_normal()), float(rng.uniform(0.2, 1.0))
    dt = 0.05 if scheme == "euler" else float(rng.uniform(0.05, 1.4))
    x, observed, latent = rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n)
    u = TO_ACTION * observed[:, None] + latent[:, None] * to_action
    u = u + ETA * rng.standard_normal((n, 2))
    push = u @ channel + TO_RATE * observed + to_rate * latent + noise * rng.standard_normal(n)
    z = DRIFT * dt
    after = x + dt * (DRIFT * x + push) if scheme == "euler" else _t4(z) * x + dt * _s4(z) * push

    short = np.outer(to_action, to_action) + ETA**2 * np.eye(2)
    solved = np.linalg.solve(short, to_action)
    nu2 = float(MOVE @ np.linalg.solve(short, MOVE))
    explained = to_rate**2 * (1.0 - float(to_action @ solved))
    return {
        "data": {
            "x": jnp.asarray(x)[:, None],
            "u": jnp.asarray(u),
            "x_next": jnp.asarray(after)[:, None],
            "observed": jnp.asarray(observed)[:, None],
        },
        "dt": dt,
        "truth": float(MOVE @ channel),
        "bias": float(MOVE @ solved) * to_rate,
        "cf_y": explained / (explained + noise**2),
        "cf_d": 1.0 - nu2 * ETA**2 / float(MOVE @ MOVE),
    }


def _read(world: dict, scheme: str) -> dict:
    fit = fit_causal_residual(
        _known,
        world["data"],
        world["dt"],
        adjust_for=("observed",),
        degree=1,
        channel_degree=0,
        integrator=scheme,
        influence=True,
    )
    truth, rises = world["truth"], world["bias"] > 0.0  # the estimate above the truth
    out = {"scheme": scheme, "cf_y": world["cf_y"], "cf_d": world["cf_d"]}
    for case, scale in (("truth", 1.0), ("half", 0.5), ("none", 0.0)):
        bound = omitted_confounder_bound(
            fit,
            MOVE.reshape(1, 2, 1),
            cf_y=scale * world["cf_y"],
            cf_d=scale * world["cf_d"],
            level=LEVEL,
        )
        facing = bound.ci_lower <= truth if rises else bound.ci_upper >= truth
        out[case] = {
            "both": bound.ci_lower <= truth <= bound.ci_upper,
            "facing": facing,
            "point": bound.lower <= truth if rises else bound.upper >= truth,
        }
    return out


def _rate(hits: list[bool]) -> dict:
    count, total = int(sum(hits)), len(hits)
    low = stats.beta.ppf(0.025, count, total - count + 1) if count else 0.0
    high = stats.beta.ppf(0.975, count + 1, total - count) if count < total else 1.0
    return {"rate": count / total, "clopper_pearson": [float(low), float(high)], "of": total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worlds", type=int, default=500)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    rngs = np.random.default_rng(args.seed).spawn(args.worlds)
    readings = [
        _read(_world(rng, args.n, scheme), scheme)
        for rng, scheme in zip(rngs, ["euler", "rk4"] * math.ceil(args.worlds / 2), strict=False)
    ]
    result: dict = {"worlds": args.worlds, "n": args.n, "seed": args.seed, "level": LEVEL}
    for group in ("all", "euler", "rk4"):
        chosen = [r for r in readings if group in ("all", r["scheme"])]
        result[group] = {
            case: {kind: _rate([r[case][kind] for r in chosen]) for kind in ("both", "facing")}
            for case in ("truth", "half", "none")
        }
        result[group]["truth"]["point"] = _rate([r["truth"]["point"] for r in chosen])
    result["shares"] = {
        name: [float(q) for q in np.quantile([r[name] for r in readings], [0.1, 0.5, 0.9])]
        for name in ("cf_y", "cf_d")
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
