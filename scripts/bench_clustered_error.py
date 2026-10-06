"""The measurements behind ``clusters`` in :func:`chc.dynamics_id.fit_causal_residual`. Rates and
ratios only, no wall time.

    repeated  A log of 100 transitions, each repeated 16 times: the channel and its error, row by
              row and summed within each transition's copies, against the log's own.
    size      Panels of --units units followed for --periods transitions, one state and one lever,
              whose channel is 0: ``x' = -0.5 x + 0 u + 0.3 w + e``, stepped by Euler at --dt, the
              lever ``u = 0.5 w + a`` and ``w`` adjusted for. In ``ar`` the noise ``e``, the
              lever's own part ``a`` and ``w`` are AR(0.7) within each unit; in ``unit`` half the
              variance of ``e`` and of ``a`` is the unit's, constant over its periods, and ``w`` is
              AR(0.7); in ``iid`` all three are drawn afresh at each period. Each of --logs panels
              is fitted twice, row by row and summed within units, and the Wald test of the true
              channel, ``|b| > 1.96`` standard errors, is counted: its size, with the median ratio
              of the two errors.

Run: JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run python scripts/bench_clustered_error.py > out.json
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from chc.dynamics_id import fit_causal_residual

ADJUSTED: dict[str, Any] = {"adjust_for": ("w",), "channel_degree": 0}


def _known(t: float | jax.Array, x: jax.Array, u: jax.Array) -> jax.Array:
    return jnp.zeros_like(x)


def _ar(rng: np.random.Generator, units: int, periods: int, rho: float) -> np.ndarray:
    """Stationary AR(``rho``) paths of unit variance, one row a unit."""
    out = np.empty((units, periods))
    out[:, 0] = rng.normal(size=units)
    for t in range(1, periods):
        out[:, t] = rho * out[:, t - 1] + math.sqrt(1.0 - rho**2) * rng.normal(size=units)
    return out


def _shared(rng: np.random.Generator, units: int, periods: int) -> np.ndarray:
    """Unit variance, half of it the unit's own, constant over its periods."""
    return math.sqrt(0.5) * (rng.normal(size=(units, 1)) + rng.normal(size=(units, periods)))


def _panel(
    rng: np.random.Generator, design: str, units: int, periods: int, dt: float
) -> dict[str, jax.Array]:
    if design == "ar":
        w, a, e = (_ar(rng, units, periods, 0.7) for _ in range(3))
    elif design == "unit":
        w = _ar(rng, units, periods, 0.7)
        a, e = _shared(rng, units, periods), _shared(rng, units, periods)
    else:
        w, a, e = (rng.normal(size=(units, periods)) for _ in range(3))
    u = 0.5 * w + a
    x = np.empty((units, periods + 1))
    x[:, 0] = rng.normal(size=units)
    for t in range(periods):
        x[:, t + 1] = x[:, t] + dt * (-0.5 * x[:, t] + 0.3 * w[:, t] + e[:, t])
    return {
        "x": jnp.asarray(x[:, :-1].reshape(-1, 1)),
        "u": jnp.asarray(u.reshape(-1, 1)),
        "w": jnp.asarray(w.reshape(-1, 1)),
        "x_next": jnp.asarray(x[:, 1:].reshape(-1, 1)),
    }


def repeated() -> dict[str, float]:
    rng = np.random.default_rng(23)
    x, z, a, e = rng.normal(size=(4, 100))
    u = z + a
    data = {
        "x": jnp.asarray(x[:, None]),
        "u": jnp.asarray(u[:, None]),
        "z": jnp.asarray(z[:, None]),
        "x_next": jnp.asarray((x + 0.1 * (-0.5 * x + 0.8 * u + 1.5 * z + 0.2 * e))[:, None]),
    }
    options: dict[str, Any] = {
        "adjust_for": ("z",),
        "channel_degree": 0,
        "nuisance_degree": 1,
        "folds": 1,
    }
    original = fit_causal_residual(_known, data, 0.1, **options)
    copies = {name: jnp.repeat(column, 16, axis=0) for name, column in data.items()}
    rows = fit_causal_residual(_known, copies, 0.1, **options)
    clustered = fit_causal_residual(
        _known, copies, 0.1, **options, clusters=np.repeat(np.arange(100), 16)
    )
    assert original.channel_error is not None
    assert rows.channel_error is not None
    assert clustered.channel_error is not None
    return {
        "channel_moved": abs(
            float(clustered.residual.channel[0, 0, 0] - original.residual.channel[0, 0, 0])
        ),
        "original_error": original.channel_error,
        "rows_error_ratio": rows.channel_error / original.channel_error,
        "clustered_error_ratio": clustered.channel_error / original.channel_error,
    }


def size(design: str, units: int, periods: int, dt: float, logs: int, seed: int) -> dict:
    rng = np.random.default_rng([seed, units, periods])
    labels = np.repeat(np.arange(units), periods)
    rejected = {"rows": 0, "units": 0}
    ratios = []
    for _ in range(logs):
        data = _panel(rng, design, units, periods, dt)
        rows = fit_causal_residual(_known, data, dt, **ADJUSTED)
        clustered = fit_causal_residual(_known, data, dt, **ADJUSTED, clusters=labels)
        assert rows.channel_error is not None
        assert clustered.channel_error is not None
        channel = float(rows.residual.channel[0, 0, 0])
        rejected["rows"] += abs(channel) > 1.96 * rows.channel_error
        rejected["units"] += abs(channel) > 1.96 * clustered.channel_error
        ratios.append(clustered.channel_error / rows.channel_error)
    return {
        "design": design,
        "units": units,
        "periods": periods,
        "dt": dt,
        "logs": logs,
        "size_rows": rejected["rows"] / logs,
        "size_units": rejected["units"] / logs,
        "error_ratio_median": float(np.median(ratios)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", type=int, default=400)
    parser.add_argument("--periods", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cells = [(design, units, 0.1) for design in ("ar", "unit") for units in (5, 10, 20, 40, 80)]
    cells += [("ar", 40, 1.0), ("unit", 40, 1.0), ("iid", 40, 0.1), ("iid", 40, 1.0)]
    out = {
        "repeated": repeated(),
        "size": [
            size(design, units, args.periods, dt, args.logs, args.seed)
            for design, units, dt in cells
        ],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
