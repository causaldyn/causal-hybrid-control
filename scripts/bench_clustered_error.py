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
              channel, ``|b| > 1.96`` standard errors, is counted: its size, by unit at
              ``t(G - 1)``'s quantile too, with the median ratio of the two errors.
    two_way   Panels of a unit count and a period count each (--cell, or all twelve), one state
              and one lever, whose channel is 0: ``x_next = x + 0.1 (-0.5 x + 0 u + 1.5 z) + e``,
              the lever ``u = c + a + 0.9 z`` and ``z`` adjusted for, ``z`` drawn afresh. The
              lever's own part ``a`` and the noise's unit part are AR(0.7) within each unit. In
              ``shared`` every unit shares a part of the lever ``c``, AR(0.7) over the periods, and
              a shock in the noise each period; in ``none`` neither. Each of --logs panels is fitted
              row by row, summed within units, within periods, and two ways (``(N, 2)`` clusters,
              the largest of three reads), and each test's size is counted at 1.96, the two-way one
              at ``t(G - 1)`` too, ``G`` the smaller count. Read off the one-way errors beside them:
              the two ways' sums alone, at the smaller count's CR1 factor and at each sum's own,
              and the units' and the periods' sums added; each error's root-mean-square over the
              channel's spread; how often each of the three reads was the largest; and how many
              panels the two ways' sums read nothing in.

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
from scipy import stats

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
    rejected = {"rows": 0, "units": 0, "units_t": 0}
    quantile = float(stats.t.ppf(0.975, units - 1))
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
        rejected["units_t"] += abs(channel) > quantile * clustered.channel_error
        ratios.append(clustered.channel_error / rows.channel_error)
    return {
        "design": design,
        "units": units,
        "periods": periods,
        "dt": dt,
        "logs": logs,
        "size_rows": rejected["rows"] / logs,
        "size_units": rejected["units"] / logs,
        "size_units_t": rejected["units_t"] / logs,
        "error_ratio_median": float(np.median(ratios)),
    }


TWO_WAY_CELLS = [
    (design, units, periods)
    for design in ("shared", "none")
    for units, periods in ((10, 40), (20, 40), (40, 40), (40, 5), (40, 10), (40, 20))
]


def _two_way_panel(
    rng: np.random.Generator, design: str, units: int, periods: int
) -> tuple[dict[str, jax.Array], np.ndarray]:
    z = rng.normal(size=(units, periods))
    shared = design == "shared"
    common = _ar(rng, 1, periods, 0.7) if shared else np.zeros((1, periods))
    u = common + 0.5 * _ar(rng, units, periods, 0.7) + 0.9 * z
    shock = 0.05 * rng.normal(size=(1, periods)) if shared else np.zeros((1, periods))
    e = shock + 0.05 * _ar(rng, units, periods, 0.7)
    x = np.empty((units, periods + 1))
    x[:, 0] = rng.normal(size=units)
    for t in range(periods):
        x[:, t + 1] = x[:, t] + 0.1 * (-0.5 * x[:, t] + 1.5 * z[:, t]) + e[:, t]
    data = {
        "x": jnp.asarray(x[:, :-1].reshape(-1, 1)),
        "u": jnp.asarray(u.reshape(-1, 1)),
        "z": jnp.asarray(z.reshape(-1, 1)),
        "x_next": jnp.asarray(x[:, 1:].reshape(-1, 1)),
    }
    labels = np.column_stack(
        [np.repeat(np.arange(units), periods), np.tile(np.arange(periods), units)]
    )
    return data, labels


def two_way(design: str, units: int, periods: int, logs: int, seed: int) -> dict:
    rng = np.random.default_rng([seed, units, periods, int(design == "shared")])
    errors: dict[str, list[float]] = {"rows": [], "unit": [], "period": [], "two_way": []}
    channels = []
    for _ in range(logs):
        data, labels = _two_way_panel(rng, design, units, periods)
        groupings = (None, labels[:, 0], labels[:, 1], labels)
        for name, clusters in zip(errors, groupings, strict=True):
            fit = fit_causal_residual(
                _known, data, 0.1, adjust_for=("z",), channel_degree=0, clusters=clusters
            )
            assert fit.channel_error is not None
            errors[name].append(fit.channel_error)
        channels.append(float(fit.residual.channel[0, 0, 0]))
    channel = np.abs(np.array(channels))
    read = {name: np.array(values) for name, values in errors.items()}
    # one coefficient, so (N - 1) / (N - k) is 1 and each cell is one transition: the cells' sums
    # are the rows' own, which the row-by-row error reads at N / (N - 1)
    rows = units * periods
    smaller = min(units, periods)
    sums = (
        read["unit"] ** 2 * (units - 1) / units
        + read["period"] ** 2 * (periods - 1) / periods
        - read["rows"] ** 2 * (rows - 1) / rows
    )
    read["two_way_sums"] = np.sqrt(np.maximum(smaller / (smaller - 1) * sums, 0.0))
    read["two_way_own"] = np.sqrt(
        np.maximum(read["unit"] ** 2 + read["period"] ** 2 - read["rows"] ** 2, 0.0)
    )
    read["unit_plus_period"] = np.sqrt(read["unit"] ** 2 + read["period"] ** 2)
    three = np.stack([read["two_way_sums"], read["unit"], read["period"]])
    np.testing.assert_allclose(read["two_way"], three.max(axis=0), rtol=1e-9, atol=0.0)
    largest = np.argmax(three, axis=0)
    spread = float(np.std(channels, ddof=1))
    quantile = float(stats.t.ppf(0.975, min(units, periods) - 1))
    return {
        "design": design,
        "units": units,
        "periods": periods,
        "logs": logs,
        **{f"size_{name}": float(np.mean(channel > 1.96 * error)) for name, error in read.items()},
        "size_two_way_t": float(np.mean(channel > quantile * read["two_way"])),
        **{
            f"error_over_spread_{name}": float(np.sqrt(np.mean(error**2))) / spread
            for name, error in read.items()
        },
        **{
            f"largest_{name}": float(np.mean(largest == k))
            for k, name in enumerate(("two_way_sums", "unit", "period"))
        },
        "two_way_sums_read_nothing": int(np.sum(read["two_way_sums"] == 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", type=int, default=400)
    parser.add_argument("--periods", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=("repeated", "size", "two_way"),
        default=["repeated", "size"],
    )
    parser.add_argument(
        "--cell", action="append", help="a two_way cell, design,units,periods; all where none"
    )
    args = parser.parse_args()
    cells = [(design, units, 0.1) for design in ("ar", "unit") for units in (5, 10, 20, 40, 80)]
    cells += [("ar", 40, 1.0), ("unit", 40, 1.0), ("iid", 40, 0.1), ("iid", 40, 1.0)]
    out: dict[str, Any] = {}
    if "repeated" in args.sections:
        out["repeated"] = repeated()
    if "size" in args.sections:
        out["size"] = [
            size(design, units, args.periods, dt, args.logs, args.seed)
            for design, units, dt in cells
        ]
    if "two_way" in args.sections:
        chosen = [
            (design, int(units), int(periods))
            for design, units, periods in (cell.split(",") for cell in args.cell or [])
        ] or TWO_WAY_CELLS
        out["two_way"] = [
            two_way(design, units, periods, args.logs, args.seed)
            for design, units, periods in chosen
        ]
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
