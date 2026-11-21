"""Where the dither check's power goes: :class:`chc.decision.DitherCheck` against the one-degree
closed form of ``validation/dither_hausman.mac``. Rates only, no wall time.

The world is ADR 0139's: the plant ``x' = a x + b u + c w + sg e``, its logger ``u = d w + s eta``,
``w`` the column the graph leaves out and ``eta`` the logged draw, both drawn afresh, at
``a = 0.9``, ``b = 0.5`` and ``d = s = sg = 1``, over 100 units and 6 periods, 500 transitions. Log
``r`` is ``numpy.random.default_rng(r)``: the units' first states, then each period
``standard_normal((3, 100))`` for ``(w, eta, e)``. The graph says nothing confounds the lever, and
``prescribe`` plans a 5-step tracking plan with ``dither="draw"``. At ``c = 0`` the graph is
right.

Each log's contrast is caught where :func:`chc.misspecification._contrast` reads it: the two fits'
difference ``d`` over the channel's two parameters and the drift's two, the regret's curvature
``W`` and the three reads of the difference's covariance that the check takes the largest of (two
ways, the units alone, the 5 periods alone). Every log is then tested six ways at 5 %:

    library               the check's own p-value: ``d' W d`` against the chi-square mixture of
                          ``W S``, ``S`` the read with the largest ``tr(W S)``
    one_degree_closed     the channel's level alone, ``d_0``, studentised by the closed form's
                          standard error at the fixed alternative (``dither_hausman.mac`` STEP 3b):
                          what the closed form's power is a power of
    one_degree_spread     ``d_0`` studentised by its spread over the cell's logs
    one_degree_read       ``d_0`` studentised by the largest read's own entry for it
    mixture_spread        ``d' W d`` against the mixture of ``W`` and the covariance of ``d`` over
                          the cell's logs: the check with a covariance it does not have to read
    mixture_mean_read     the same with the largest read's mean over the cell's logs: read high
                          as the check reads it, but without a log's own noise

``one_degree_closed`` against the closed form checks the contrast's law, ``mixture_spread``
against ``one_degree_spread`` prices the three parameters that carry no bias, and ``library``
against ``mixture_spread`` prices reading the covariance off one log.

Run: JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run python scripts/bench_dither_power.py > out.json
``--records logs.npz`` keeps every log's contrast, and ``--replay logs.npz`` reads them back instead
of fitting.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import jax
import numpy as np
from scipy import stats

A, B, D, S, SG = 0.9, 0.5, 1.0, 1.0, 1.0
UNITS, PERIODS, LEVEL = 100, 6, 0.05
TRANSITIONS = UNITS * (PERIODS - 1)
READS = ("two_ways", "units", "periods")


def _panel(c: float, seed: int) -> Any:
    from chc import Panel

    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, UNITS)
    names = ("unit", "time", "x", "incentive", "draw")
    rows: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for t in range(PERIODS):
        w, eta, e = rng.standard_normal((3, UNITS))
        u = D * w + S * eta
        for name, value in zip(
            names, (np.arange(UNITS), np.full(UNITS, t), x, u, eta), strict=True
        ):
            rows[name].append(value)
        x = A * x + B * u + c * w + SG * e
    frame = {name: np.concatenate(value) for name, value in rows.items()}
    return Panel.from_frame(frame, unit="unit", time="time", seed=0)


def _one_log(c: float, seed: int) -> dict[str, Any]:
    """One log's contrast, caught at :func:`chc.misspecification._contrast`."""
    import chc.decision as decision
    from chc import CausalGraph, Lever, Target, prescribe
    from chc._robust_cov import _clustered_squares
    from chc.misspecification import _parameter_weight, _parameters

    logging.getLogger("chc").setLevel(logging.ERROR)
    caught: dict[str, Any] = {}
    contrast = decision._contrast

    def catch(plan: Any, reference: Any, alternative: Any, tolerance: Any) -> Any:
        caught.update(plan=plan, reference=reference, alternative=alternative)
        return contrast(plan, reference, alternative, tolerance)

    decision._contrast = catch
    try:
        check = prescribe(
            _panel(c, seed),
            levers=[Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05)],
            target=Target("x", value=1.0),
            adjustment=CausalGraph.from_edges([("incentive", "x")]),
            horizon=5,
            dt=1.0,
            dither="draw",
        ).dither_check
    finally:
        decision._contrast = contrast
    if check is None or check.contrast is None:
        raise RuntimeError(f"log {seed} at c = {c} read no contrast")
    reference, alternative = caught["reference"], caught["alternative"]
    difference = np.asarray(alternative.influence) - np.asarray(reference.influence)
    reads = _clustered_squares(difference, reference.clusters, reference.periods)
    return {
        "d": _parameters(alternative.residual) - _parameters(reference.residual),
        "weight": _parameter_weight(caught["plan"], reference.residual, None),
        "reads": np.stack(reads),
        "size": np.asarray(reference._parameter_size, dtype=np.float64),
        "p": check.contrast.p_value,
    }


def _weights(weight: np.ndarray, covariance: np.ndarray, size: np.ndarray) -> np.ndarray:
    """The eigenvalues of ``W S``, largest first, scaled as the check scales them."""
    from chc.misspecification import _mixture_weights

    scaled = _mixture_weights(weight / np.outer(size, size), covariance * np.outer(size, size))
    return np.sort(scaled)[::-1]


def _mixture_p(
    d: np.ndarray, weight: np.ndarray, covariance: np.ndarray, size: np.ndarray
) -> float:
    """The check's p-value of ``d' W d`` against the mixture of ``W`` and ``covariance``."""
    from chc.uncertainty import _chi_square_mixture_survival

    return _chi_square_mixture_survival(float(d @ weight @ d), _weights(weight, covariance, size))


def _closed_form_se(c: float) -> float:
    """The standard error of the channel's contrast over the log at the fixed alternative:
    ``beta0^2 / lambda1`` a transition, ``validation/dither_hausman.mac`` STEP 3b."""
    per_transition = SG**2 * D**2 / (S**2 * (D**2 + S**2))  # STEP 2b, the c = 0 limit
    if c != 0.0:
        beta0 = c * D / (D**2 + S**2)
        lam1 = c**2 * S**2 / (SG**2 * (D**2 + S**2) + c**2 * (D**2 + 4.0 * S**2))
        per_transition = beta0**2 / lam1
    return math.sqrt(per_transition / TRANSITIONS)


def _rate(p: np.ndarray) -> dict[str, float]:
    rate = float(np.mean(p <= LEVEL))
    return {"rejected": rate, "mc_se": math.sqrt(rate * (1.0 - rate) / p.size)}


def _cell(c: float, logs: list[dict[str, Any]]) -> dict[str, Any]:
    d = np.stack([log["d"] for log in logs])
    weights = np.stack([log["weight"] for log in logs])
    reads = np.stack([log["reads"] for log in logs])
    sizes = np.stack([log["size"] for log in logs])
    chosen = np.array(
        [
            int(np.argmax([np.trace(w @ r) for r in rs]))
            for w, rs in zip(weights, reads, strict=True)
        ]
    )
    largest = reads[np.arange(len(logs)), chosen]
    spread = np.cov(d, rowvar=False)
    mean_read = largest.mean(axis=0)

    def two_sided(z: np.ndarray) -> np.ndarray:
        return 2.0 * stats.norm.sf(z)

    p = {
        "library": np.array([log["p"] for log in logs]),
        "one_degree_closed": two_sided(np.abs(d[:, 0]) / _closed_form_se(c)),
        "one_degree_spread": two_sided(np.abs(d[:, 0]) / math.sqrt(spread[0, 0])),
        "one_degree_read": two_sided(np.abs(d[:, 0]) / np.sqrt(largest[:, 0, 0])),
        "mixture_spread": np.array(
            [_mixture_p(di, w, spread, s) for di, w, s in zip(d, weights, sizes, strict=True)]
        ),
        "mixture_mean_read": np.array(
            [_mixture_p(di, w, mean_read, s) for di, w, s in zip(d, weights, sizes, strict=True)]
        ),
    }
    replayed = np.array(
        [_mixture_p(di, w, r, s) for di, w, r, s in zip(d, weights, largest, sizes, strict=True)]
    )
    trace_spread = np.einsum("nij,ji->n", weights, spread)
    mixtures = [_weights(w, spread, s) for w, s in zip(weights, sizes, strict=True)]
    return {
        "c": c,
        "logs": len(logs),
        "power": {name: _rate(values) for name, values in p.items()},
        "library_replayed_max_gap": float(np.max(np.abs(replayed - p["library"]))),
        "contrast": {
            "level_mean": float(d[:, 0].mean()),
            "level_mean_closed": -c * D / (D**2 + S**2),
            "level_se_spread": math.sqrt(spread[0, 0]),
            "level_se_closed": _closed_form_se(c),
        },
        "read": {
            "chosen_share": {name: float(np.mean(chosen == k)) for k, name in enumerate(READS)},
            "level_variance_over_spread_mean": float(largest[:, 0, 0].mean() / spread[0, 0]),
            "level_variance_over_spread_quartiles": np.quantile(
                largest[:, 0, 0] / spread[0, 0], [0.25, 0.5, 0.75]
            ).tolist(),
            "trace_over_spread_mean": float(
                np.mean(np.einsum("nij,nji->n", weights, largest) / trace_spread)
            ),
        },
        # each log's mixture weights over its largest, averaged: how many directions carry noise
        "mixture_spread_weights_over_largest": np.mean(
            [ratio / ratio[0] for ratio in mixtures], axis=0
        ).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confounders", type=float, nargs="+", default=[0.0, 0.15, 0.27])
    parser.add_argument("--logs", type=int, default=400)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--records", default=None)
    parser.add_argument("--replay", default=None)
    args = parser.parse_args()
    fields = ("d", "weight", "reads", "size", "p")
    cells: dict[float, list[dict[str, Any]]] = {}
    if args.replay is not None:
        with np.load(args.replay) as saved:
            for c in args.confounders:
                columns = [saved[f"{c}/{field}"] for field in fields]
                cells[c] = [
                    dict(zip(fields, row, strict=True)) for row in zip(*columns, strict=True)
                ]
    else:
        context = multiprocessing.get_context("spawn")  # JAX does not survive a fork
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            for c in args.confounders:
                cells[c] = list(pool.map(_one_log, [c] * args.logs, range(args.logs)))
        if args.records is not None:
            np.savez(
                args.records,
                **{
                    f"{c}/{field}": np.stack([np.asarray(log[field]) for log in logs])
                    for c, logs in cells.items()
                    for field in fields
                },
            )
    out = {
        "x64": bool(jax.config.jax_enable_x64),
        "cells": [_cell(c, logs) for c, logs in cells.items()],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
