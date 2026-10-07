"""The measurements behind :func:`chc.dynamics_id.persistence_check` and the Markov premise of
:func:`chc.dynamics_id.fit_causal_residual`. Rates and ratios only, no wall time.

    bias    Panels of --units units, one state and one lever, whose channel is 0.8:
            ``x' = 0.95 x + 0.1 (0.8 u + 1.5 z) + e``, the lever ``u = a + 0.9 z``, ``z`` drawn
            afresh and adjusted for, ``a`` AR(0.7) of spread 0.5 and the noise ``e`` AR(rho) of
            spread 0.05 within each unit. The state starts at N(0, 1) (``cold``), or 300 periods
            before the log does (``warm``). The log keeps the transitions from its second period
            on, so that each has its unit's one before it. Each of --logs panels is fitted four
            ways: adjusting for ``z`` (``base``); for ``z`` and the state a period earlier
            (``xlag``); for ``z`` and the state, the lever and ``z`` a period earlier (``lags``);
            and adjusting for ``z`` with the lever a period earlier as the instrument (``iv``).
            Each way's mean error in the channel and its Monte Carlo standard error; and the
            persistence check of the ``base`` fit: its mean correlation and how often it rejects
            at 5 %. Cells: --cell rho,periods,start, or the defaults.
    size    The persistence check alone, on ``base`` fits at --logs-size panels a cell, where the
            noise does not persist (rho 0) and where it does a little (rho 0.3), at 5, 10, 40 and
            200 units over 40 periods: how often it rejects at 5 %.

Run: JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run python scripts/bench_persistent_noise.py > out.json
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from chc.dynamics_id import fit_causal_residual, persistence_check

DT, CHANNEL, DRIFT, CONFOUNDING, MIX = 0.1, 0.8, 0.95, 1.5, 0.9
LEVER_SPREAD, LEVER_PERSISTENCE, NOISE_SPREAD, BURN = 0.5, 0.7, 0.05, 300
WAYS: dict[str, dict[str, Any]] = {
    "base": {"adjust_for": ("z",)},
    "xlag": {"adjust_for": ("z", "x_lag")},
    "lags": {"adjust_for": ("z", "x_lag", "u_lag", "z_lag")},
    "iv": {"adjust_for": ("z",), "instrument": "u_lag"},
}
BIAS_CELLS = [(rho, periods, "cold") for rho in (0.0, 0.3, 0.7, 0.9) for periods in (20, 40, 100)]
BIAS_CELLS += [(0.7, 40, "warm")]


def _known(t: float | jax.Array, x: jax.Array, u: jax.Array) -> jax.Array:
    return jnp.zeros_like(x)


def _ar(rng: np.random.Generator, shape: tuple[int, int], spread: float, rho: float) -> np.ndarray:
    """Stationary AR(``rho``) paths of spread ``spread``, one column a unit."""
    out = np.empty(shape)
    out[0] = rng.normal(0.0, spread, shape[1])
    for t in range(1, shape[0]):
        out[t] = rho * out[t - 1] + rng.normal(0.0, spread * math.sqrt(1.0 - rho**2), shape[1])
    return out


def _panel(
    rng: np.random.Generator, rho: float, units: int, periods: int, start: str
) -> tuple[dict[str, jax.Array], np.ndarray, np.ndarray]:
    """The log's columns, and each transition's unit and period."""
    burn = BURN if start == "warm" else 0
    steps = periods + burn
    z = rng.normal(size=(steps, units))
    u = _ar(rng, (steps, units), LEVER_SPREAD, LEVER_PERSISTENCE) + MIX * z
    e = _ar(rng, (steps, units), NOISE_SPREAD, rho)
    x = np.empty((steps + 1, units))
    x[0] = rng.normal(size=units)
    for t in range(steps):
        x[t + 1] = DRIFT * x[t] + DT * (CHANNEL * u[t] + CONFOUNDING * z[t]) + e[t]
    x, u, z = x[burn:], u[burn:], z[burn:]

    def column(values: np.ndarray) -> jax.Array:
        return jnp.asarray(values.reshape(-1, 1))

    now, before = slice(1, periods), slice(0, periods - 1)
    data = {
        "x": column(x[:-1][now]),
        "u": column(u[now]),
        "z": column(z[now]),
        "x_next": column(x[1:][now]),
        "x_lag": column(x[:-1][before]),
        "u_lag": column(u[before]),
        "z_lag": column(z[before]),
    }
    labels = np.broadcast_to(np.arange(units), (periods - 1, units)).reshape(-1)
    steps_of = np.broadcast_to(np.arange(1, periods)[:, None], (periods - 1, units)).reshape(-1)
    return data, labels, steps_of


def bias(rho: float, periods: int, start: str, units: int, logs: int, seed: int) -> dict:
    rng = np.random.default_rng([seed, round(rho * 10), periods, int(start == "warm")])
    reads: dict[str, list[float]] = {way: [] for way in WAYS}
    correlations, rejected = [], 0
    for _ in range(logs):
        data, labels, steps = _panel(rng, rho, units, periods, start)
        for way, options in WAYS.items():
            fit = fit_causal_residual(_known, data, DT, channel_degree=0, **options)
            reads[way].append(float(np.ravel(fit.residual.channel)[0]))
            if way == "base":
                check = persistence_check(fit, labels, steps)
                correlations.append(check.correlation)
                rejected += check.p_value <= 0.05
    out: dict[str, Any] = {"rho": rho, "periods": periods, "start": start, "units": units}
    out["logs"] = logs
    for way, values in reads.items():
        out[f"bias_{way}"] = float(np.mean(values) - CHANNEL)
        out[f"mc_se_{way}"] = float(np.std(values, ddof=1) / math.sqrt(logs))
    out["correlation"] = float(np.mean(correlations))
    out["rejected"] = rejected / logs
    return out


def size(rho: float, units: int, logs: int, seed: int) -> dict:
    rng = np.random.default_rng([seed, round(rho * 10), units, 7])
    correlations, rejected = [], 0
    for _ in range(logs):
        data, labels, steps = _panel(rng, rho, units, 40, "cold")
        fit = fit_causal_residual(_known, data, DT, channel_degree=0, **WAYS["base"])
        check = persistence_check(fit, labels, steps)
        correlations.append(check.correlation)
        rejected += check.p_value <= 0.05
    return {
        "rho": rho,
        "units": units,
        "logs": logs,
        "correlation": float(np.mean(correlations)),
        "rejected": rejected / logs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", type=int, default=50)
    parser.add_argument("--logs-size", type=int, default=400)
    parser.add_argument("--units", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sections", nargs="+", choices=("bias", "size"), default=["bias", "size"])
    parser.add_argument("--cell", action="append", help="a bias cell, rho,periods,start")
    parser.add_argument("--size-cell", action="append", help="a size cell, rho,units")
    args = parser.parse_args()
    out: dict[str, Any] = {}
    if "bias" in args.sections:
        cells = [
            (float(rho), int(periods), start)
            for rho, periods, start in (cell.split(",") for cell in args.cell or [])
        ] or BIAS_CELLS
        out["bias"] = [
            bias(rho, periods, start, args.units, args.logs, args.seed)
            for rho, periods, start in cells
        ]
    if "size" in args.sections:
        cells = [
            (float(rho), int(units))
            for rho, units in (cell.split(",") for cell in args.size_cell or [])
        ] or [(rho, units) for rho in (0.0, 0.3) for units in (5, 10, 40, 200)]
        out["size"] = [size(rho, units, args.logs_size, args.seed) for rho, units in cells]
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
