"""The measurements behind ``channel_degree`` in :func:`chc.dynamics_id.fit_causal_residual`.
Channel errors only, no wall time.

    zones    :class:`chc.zones.ZoneMarketSystem` at four zones under both matching laws, over
             ``--logs`` logs each, fitted by the Euler reading with the shocks adjusted for, with a
             constant channel (``channel_degree=0``) and with one affine in the state (the default
             at ``degree=1``). Each is read at the do-nothing point, at the log's mean state and at
             ``x = 0``, against one RK4 period's response at the do-nothing point, over the rows the
             fit learns (supply and queue): the relative root mean square error, its mean and its
             largest value over the logs.

Run: JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run python scripts/bench_channel_degree.py > out.json
"""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from chc.dynamics_id import fit_causal_residual
from chc.zones import ZoneMarketSystem

DEGREES = {"constant": 0, "affine": None}


def _transitions(system: ZoneMarketSystem, logs: dict[str, np.ndarray]) -> dict[str, jax.Array]:
    """Consecutive periods of a day as ``(x, u, x_next)``, plus each zone's shock."""
    day = logs["day"]
    current = np.flatnonzero(np.r_[day[1:] == day[:-1], False])
    x = np.stack([logs[name] for name in system.state_columns], 1)
    u = np.stack([logs[name] for name in system.lever_columns], 1)
    data = {"x": jnp.asarray(x[current]), "u": jnp.asarray(u[current])}
    data["x_next"] = jnp.asarray(x[current + 1])
    for name in system.shock_columns:
        data[name] = jnp.asarray(logs[name][current][:, None])
    return data


def zones(logs: int) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    out: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for matching in ("linear", "harmonic"):
        system = ZoneMarketSystem(matching=matching)
        learned = slice(0, 2 * system.zones)
        truth = np.asarray(system.linear_gaussian(1.0).b)[learned]
        errors: dict[tuple[str, str], list[float]] = {}
        for seed in range(logs):
            data = _transitions(system, system.sample(seed=seed))
            points = {
                "do_nothing": system.do_nothing,
                "log_mean": jnp.mean(data["x"], axis=0),
                "zero": jnp.zeros(3 * system.zones),
            }
            for name, channel_degree in DEGREES.items():
                fit = fit_causal_residual(
                    system.stock_dynamics(),
                    data,
                    1.0,
                    adjust_for=system.shock_columns,
                    integrator="euler",
                    seed=0,
                    channel_degree=channel_degree,
                )
                for point, state in points.items():
                    read = np.asarray(fit.residual.control_channel(state))[learned]
                    relative = np.sqrt(np.mean((read - truth) ** 2) / np.mean(truth**2))
                    errors.setdefault((name, point), []).append(float(relative))
        out[matching] = {
            name: {
                point: {
                    "mean": float(np.mean(errors[name, point])),
                    "max": float(np.max(errors[name, point])),
                }
                for point in ("do_nothing", "log_mean", "zero")
            }
            for name in DEGREES
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=int, default=8)
    args = parser.parse_args()
    out = {"logs": args.logs, "x64": bool(jax.config.jax_enable_x64), "zones": zones(args.logs)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
