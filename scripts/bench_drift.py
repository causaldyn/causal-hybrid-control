"""The measurements behind docs/adr/0018-a-channel-drift-monitor.md. Run lengths and alarm rates
only, no wall time.

The lab's plant, which the monitor's model gets wrong everywhere but in the channel's
neighbourhood: ``x' = x - 0.2 x + b u + eps``, with ``eps`` Laplace of standard deviation 0.5, and
``u = -0.5 x + 0.3 xi``, with ``xi ~ N(0, 1)`` logged as the dither. The model is
``x' = x - 0.1 x + 0.05 + 1.0 u``: its drift is off in slope and intercept, and its channel 0.07
from the plant's ``b = 1.07``, inside the radius 0.1. The residual scale is the noise's standard
deviation, 0.5. Every run is watched with the radius and, for contrast, without it.

    null     b = 1.07 throughout: the alarm's run length against the average run length ``A`` it
             was set for, censored at 80 000 decisions, so its mean is a lower bound.
    change   b = 1.4 from the start: the delay to the first alarm, over 3000 decisions, against an
             oracle that knows the entry, the side and the best constant bet, alarming at the same
             ``A``.
    drift    b = 1.07, and the plant's drift off by a further +0.15: the run length as in null,
             since the guarantee is meant to hold whatever the drift model's error.
    box      null and change with every action clipped to [-0.5, 0.5] and every decision
             dithered, so that about a quarter of them clip; the log carries the draw. For
             contrast, the same alarm with the clipped decisions' e-values set to 0, which is also
             valid.

Run: uv run python scripts/bench_drift.py [--paths 300] > out.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math

import numpy as np

from chc.gate import DecisionLog, DriftAlarm, channel_drift_evalues

MODEL_CHANNEL, RADIUS, DITHER, NOISE = 1.0, 0.1, 0.3, 0.5
PRE, POST, DRIFT_ERROR = 1.07, 1.4, 0.15
CAP, HORIZON, CHUNK = 80_000, 3000, 1000
BOX = 0.5
SEED = 20260928


def _step(
    x: np.ndarray, b: float, shift: float, rng: np.random.Generator, box: float = math.inf
) -> tuple[np.ndarray, ...]:
    """One decision on every path: the action as applied, its dither as drawn, the residual
    against the model, the next state, and whether the box clipped the action."""
    dither = DITHER * rng.standard_normal(x.shape)
    wanted = -0.5 * x + dither
    u = np.clip(wanted, -box, box)
    noise = rng.laplace(scale=NOISE / math.sqrt(2.0), size=x.shape)
    after = x + (-0.2 * x + shift) + b * u + noise
    residual = after - (x + (-0.1 * x + 0.05) + MODEL_CHANNEL * u)
    return u, dither, residual, after, u != wanted


def _first_alarm(alarm: DriftAlarm, evalues: np.ndarray) -> int | None:
    """The row at which ``alarm`` sounds on ``evalues``, if it does, found by replaying the rows
    one at a time from the statistic's state before them."""
    before = copy.deepcopy(alarm)
    if not alarm.update(evalues):
        return None
    for t in range(evalues.shape[0]):
        if before.update(evalues[t : t + 1]):
            return t
    raise AssertionError("the alarm sounded on the block and on none of its rows")


def _oracle_bet(rng: np.random.Generator) -> float:
    """The best constant bet on the growth side of the entry after the change, in raw units:
    ``k / (E c^2 + k^2)``, with ``E c^2`` read off a long run of the changed plant."""
    k = (POST - MODEL_CHANNEL - RADIUS) * DITHER
    x, squares = rng.normal(0.0, 0.7, 20_000), []
    for step in range(300):
        u, dither, residual, x, _ = _step(x, POST, 0.0, rng)
        c = residual - RADIUS * u - k * dither / DITHER
        if step >= 50:
            squares.append(float(np.mean(c * c)))
    return k / (float(np.mean(squares)) + k * k)


def _run(
    paths: int,
    steps: int,
    b: float,
    shift: float,
    arl: float,
    oracle: float | None,
    seed: int,
    box: float = math.inf,
) -> dict[str, np.ndarray]:
    """First-alarm times per path, ``inf`` where none came within ``steps``."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.7, paths)
    watchers = {"radius": RADIUS, "no radius": 0.0}
    if math.isfinite(box):
        watchers = {"radius": RADIUS, "clipped zeroed": RADIUS}
    alarms = {name: [DriftAlarm(arl) for _ in range(paths)] for name in watchers}
    first = {name: np.full(paths, np.inf) for name in watchers}
    if oracle is not None:
        alarms["oracle"] = [DriftAlarm(arl) for _ in range(paths)]
        first["oracle"] = np.full(paths, np.inf)
    for start in range(0, steps, CHUNK):
        rows = min(CHUNK, steps - start)
        block = [_step(x, b, shift, rng, box) for _ in range(rows)]
        x = block[-1][3]
        u, dither, residual, saturated = (
            np.stack([step[i] for step in block]) for i in (0, 1, 2, 4)
        )
        for p in range(paths):
            log = DecisionLog(
                action=u[:, p],
                propensity=np.ones(rows),
                saturated=saturated[:, p],
                dither=dither[:, p],
            )
            for name, radius in watchers.items():
                if np.isfinite(first[name][p]):
                    continue
                evalues = channel_drift_evalues(
                    log, residual[:, p], dither_scale=DITHER, radius=radius, residual_scale=NOISE
                )
                if name == "clipped zeroed":
                    evalues = np.where(saturated[:, p, None], 0.0, evalues)
                t = _first_alarm(alarms[name][p], evalues)
                if t is not None:
                    first[name][p] = start + t + 1
            if oracle is not None and not np.isfinite(first["oracle"][p]):
                edge = residual[:, p] - RADIUS * u[:, p]
                xi = dither[:, p] / DITHER
                evalues = np.exp(oracle * edge * xi - (oracle * edge) ** 2 / 2.0)
                t = _first_alarm(alarms["oracle"][p], evalues)
                if t is not None:
                    first["oracle"][p] = start + t + 1
        if all(np.isfinite(times).all() for times in first.values()):
            break
    return first


def _clipped_share(b: float, rng: np.random.Generator) -> float:
    """The share of decisions the box clips on the plant with channel ``b``, once it has settled."""
    x, shares = rng.normal(0.0, 0.7, 20_000), []
    for step in range(300):
        *_, x, saturated = _step(x, b, 0.0, rng, BOX)
        if step >= 50:
            shares.append(float(saturated.mean()))
    return float(np.mean(shares))


def _null(times: np.ndarray, arl: float) -> dict[str, float]:
    alarmed = np.isfinite(times)
    censored = np.where(alarmed, times, CAP)
    return {
        "mean_run_length_at_least": float(np.mean(censored)),
        "over_target_at_least": float(np.mean(censored) / arl),
        "censored": float(1.0 - alarmed.mean()),
        "alarm_by_3000": float(np.mean(times <= HORIZON)),
    }


def _delay(times: np.ndarray) -> dict[str, float]:
    alarmed = np.isfinite(times)
    return {
        "alarm_by_3000": float(alarmed.mean()),
        "mean_delay": float(np.mean(times[alarmed])) if alarmed.any() else math.nan,
        "median_delay": float(np.median(times[alarmed])) if alarmed.any() else math.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=300)
    args = parser.parse_args()
    logging.getLogger("chc.gate").setLevel(logging.ERROR)
    oracle = _oracle_bet(np.random.default_rng(SEED))
    out: dict[str, object] = {
        "oracle_bet": oracle,
        "paths": args.paths,
        "box_clipped": {
            f"b={b}": _clipped_share(b, np.random.default_rng(SEED)) for b in (PRE, POST)
        },
    }
    for arl in (1e3, 1e4):
        seed = SEED + int(arl)
        null = _run(args.paths, CAP, PRE, 0.0, arl, None, seed)
        change = _run(args.paths, HORIZON, POST, 0.0, arl, oracle, seed + 1)
        drift = _run(args.paths, CAP, PRE, DRIFT_ERROR, arl, None, seed + 2)
        box_null = _run(args.paths, CAP, PRE, 0.0, arl, None, seed + 3, BOX)
        box_change = _run(args.paths, HORIZON, POST, 0.0, arl, None, seed + 4, BOX)
        out[f"A={arl:g}"] = {
            "null": {name: _null(t, arl) for name, t in null.items()},
            "change": {name: _delay(t) for name, t in change.items()},
            "drift": {name: _null(t, arl) for name, t in drift.items()},
            "box null": {name: _null(t, arl) for name, t in box_null.items()},
            "box change": {name: _delay(t) for name, t in box_change.items()},
        }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
