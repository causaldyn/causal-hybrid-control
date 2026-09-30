"""The measurements behind MM2: the discount DLM's monitor, :func:`chc.dlm.monitor_evalues` read by
:class:`chc.gate.DriftAlarm`, against West and Harrison's cumulative monitor at its usual threshold.
Counts and run lengths only, no wall time.

The world is a local level whose evolution variance is ``W = V (1 - delta)^2 / delta``, ``V = 1``,
started from the filter's own steady state, so the discount model with ``delta`` is the Kalman
filter of the truth and its standardised one-step errors are independent standard normals: the
null holds by construction (R19), and the only question is whether the pipeline keeps its bound.
The monitor reads the default alternatives, a shift of 2 either way and an inflation of 2, one
detector each. West and Harrison's monitor is read on all three, alarming when any does, and on the
shifts and the inflation apart.

    null      ``--paths`` series of ``CAP`` steps. DriftAlarm at ``ARL``, read after every step: the
              share of series that alarm by ``HORIZON`` steps, which ``HORIZON / ARL`` bounds,
              with its Clopper-Pearson interval, and the run length. Beside it West and Harrison's
              monitor on the same e-values: each detector runs ``M_t = max(1, M_(t-1)) e_t``,
              which is ``1 / L_t`` for their ``L_t = H_t min(1, L_(t-1))``, and alarms when one
              reaches ``1 / TAU``.
    break     ``CHANGE + AFTER`` steps with, from step ``CHANGE``, the level moved by ``s`` of the
              steady one-step forecast's standard deviations, or the noise's standard deviation
              multiplied by ``k``. Both monitors start at ``CHANGE``, and the delay counts the
              change's own step as 1, censored at ``AFTER``.
    variance  ``AFTER`` steps of the null world multiplied by ``r``, filtered with the variance
              known and equal to 1, so every one-step scale is off by ``r^2``, and with it learned
              from a prior of 5 degrees of freedom at 1. The share that alarm by ``HORIZON`` and
              by ``AFTER`` steps.

Run: uv run python scripts/bench_dlm_monitor.py [--paths 300] [--seed S] > out.json

Measured 2026-09-30, 300 paths at the default seed:

* null: DriftAlarm alarmed by step 100 on 0.027 of the series (Clopper-Pearson 0.012-0.052,
  against the bound 0.1), by step 1000 on 0.26, with a median run length of 2368.5. West and
  Harrison's monitor alarmed by step 100 on 0.997, with a median run length of 13 (mean 17.9);
  on the inflation alone, 0.76 and 51.
* break, DriftAlarm then West and Harrison's on all three, the median delay: a level moved by 2
  deviations, 6 (caught within 1000 steps in 0.74; the rest not by then) and 1; by 3, 3 and 1; by
  4, 2 and 1; the noise's deviation times 1.5, 28 and 4; times 2, 10 and 2; times 3, 4 and 2.
* variance, alarmed by step 100 then by step 1000: known, at ``r`` = 0.5 and 0.8 none (0.003 by
  1000 at 0.8), at 1.25 0.57 and 1.0, at 2 1.0; learned, 0.003 and 0.19, 0 and 0.22, 0.043 and
  0.30, 0.073 and 0.32.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from scipy import stats

from chc.dlm import DLMFit, DynamicLinearModel, Polynomial, Prior, forward_filter, monitor_evalues
from chc.gate import DriftAlarm

SEED = 20260930
DELTA = 0.9
ARL = 1_000.0
HORIZON = 100  # on the null, an alarm by HORIZON has probability at most HORIZON / ARL
CAP = 10_000
TAU = 0.135  # West and Harrison's threshold
CHANGE, AFTER = 200, 1_000
SHIFTS = (2.0, 3.0, 4.0)
INFLATIONS = (1.5, 2.0, 3.0)
RATIOS = (0.5, 0.8, 1.25, 2.0)
# monitor_evalues' columns at its defaults: the shift up, the shift down, the inflation
ALTERNATIVES = {"all": [0, 1, 2], "shifts": [0, 1], "inflation": [2]}


def _series(rng: np.random.Generator, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """A local level from the steady state, and its noise, before any change."""
    start = math.sqrt(1.0 - DELTA) * rng.standard_normal()
    level = start + np.cumsum((1.0 - DELTA) / math.sqrt(DELTA) * rng.standard_normal(steps))
    return level, rng.standard_normal(steps)


def _filter(y: np.ndarray, dof: float = math.inf) -> DLMFit:
    prior = Prior(np.zeros(1), np.array([[1.0 - DELTA]]), 1.0, dof)
    return forward_filter(DynamicLinearModel((Polynomial(1, DELTA),), prior), y)


def _drift_alarm(evalues: np.ndarray) -> int | None:
    """The step at which a fresh DriftAlarm at ARL first sounds, counting from 1."""
    alarm = DriftAlarm(ARL)
    for step, row in enumerate(evalues, start=1):
        if alarm.update(row[None, :]):
            return step
    return None


def _west_harrison(evalues: np.ndarray) -> int | None:
    running = np.ones(evalues.shape[1])
    for step, row in enumerate(evalues, start=1):
        running = np.maximum(1.0, running) * row
        if running.max() >= 1.0 / TAU:
            return step
    return None


def _clopper_pearson(hits: int, n: int) -> tuple[float, float]:
    low = float(stats.beta.ppf(0.025, hits, n - hits + 1)) if hits > 0 else 0.0
    high = float(stats.beta.ppf(0.975, hits + 1, n - hits)) if hits < n else 1.0
    return low, high


def _run_lengths(lengths: list[int | None], cap: int) -> dict[str, object]:
    censored = sum(length is None for length in lengths)
    capped = np.array([cap if length is None else length for length in lengths], dtype=float)
    return {
        "median": float(np.median(capped)),
        "quartiles": [float(q) for q in np.quantile(capped, [0.25, 0.75])],
        "mean_capped": float(capped.mean()),  # a lower bound on the mean when any is censored
        "censored_at": cap,
        "censored": censored,
    }


def _share(lengths: list[int | None], by: int) -> tuple[float, tuple[float, float]]:
    """The share of runs that alarmed by step ``by``, and its Clopper-Pearson interval."""
    hits = sum(length is not None and length <= by for length in lengths)
    return hits / len(lengths), _clopper_pearson(hits, len(lengths))


def _shown(lengths: list[int | None], by: int) -> dict[str, object]:
    share, interval = _share(lengths, by)
    return {"share": share, "interval": list(interval)}


def null(paths: int, rng: np.random.Generator) -> dict[str, object]:
    drift: list[int | None] = []
    textbook: dict[str, list[int | None]] = {name: [] for name in ALTERNATIVES}
    for _ in range(paths):
        level, noise = _series(rng, CAP)
        evalues = monitor_evalues(_filter(level + noise))
        drift.append(_drift_alarm(evalues))
        for name, columns in ALTERNATIVES.items():
            textbook[name].append(_west_harrison(evalues[:, columns]))
    _, (low, _) = _share(drift, HORIZON)
    return {
        "drift_alarm": {
            "alarmed_by_horizon": _shown(drift, HORIZON),
            "bound": HORIZON / ARL,
            "gate_met": low <= HORIZON / ARL,
            "alarmed_by_arl": _shown(drift, int(ARL)),
            "run_length": _run_lengths(drift, CAP),
        },
        "west_harrison": {
            name: {
                "alarmed_by_horizon": _shown(lengths, HORIZON),
                "run_length": _run_lengths(lengths, CAP),
            }
            for name, lengths in textbook.items()
        },
    }


def _after_change(
    paths: int, rng: np.random.Generator, shift: float, scale: float
) -> dict[str, object]:
    drift: list[int | None] = []
    textbook: dict[str, list[int | None]] = {name: [] for name in ALTERNATIVES}
    for _ in range(paths):
        level, noise = _series(rng, CHANGE + AFTER)
        level[CHANGE:] += shift / math.sqrt(DELTA)  # the steady one-step forecast's sd
        noise[CHANGE:] *= scale
        evalues = monitor_evalues(_filter(level + noise))[CHANGE:]
        drift.append(_drift_alarm(evalues))
        for name, columns in ALTERNATIVES.items():
            textbook[name].append(_west_harrison(evalues[:, columns]))
    return {
        "drift_alarm": {"caught": _shown(drift, AFTER), "delay": _run_lengths(drift, AFTER)},
        "west_harrison": {
            name: {"caught": _shown(lengths, AFTER), "delay": _run_lengths(lengths, AFTER)}
            for name, lengths in textbook.items()
        },
    }


def change(paths: int, rng: np.random.Generator) -> dict[str, object]:
    out: dict[str, object] = {}
    for shift in SHIFTS:
        out[f"level_{shift:g}"] = _after_change(paths, rng, shift, 1.0)
    for scale in INFLATIONS:
        out[f"noise_{scale:g}"] = _after_change(paths, rng, 0.0, scale)
    return out


def variance(paths: int, rng: np.random.Generator) -> dict[str, object]:
    out: dict[str, object] = {}
    for ratio in RATIOS:
        known, learned = [], []
        for _ in range(paths):
            level, noise = _series(rng, AFTER)
            y = ratio * (level + noise)
            known.append(_drift_alarm(monitor_evalues(_filter(y))))
            learned.append(_drift_alarm(monitor_evalues(_filter(y, dof=5.0))))
        out[f"ratio_{ratio:g}"] = {
            "known": {"by_horizon": _shown(known, HORIZON), "by_after": _shown(known, AFTER)},
            "learned": {
                "by_horizon": _shown(learned, HORIZON),
                "by_after": _shown(learned, AFTER),
            },
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", type=int, default=300)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    null_rng, change_rng, variance_rng = np.random.default_rng(args.seed).spawn(3)
    result = {
        "paths": args.paths,
        "seed": args.seed,
        "delta": DELTA,
        "arl": ARL,
        "horizon": HORIZON,
        "tau": TAU,
        "null": null(args.paths, null_rng),
        "change": change(args.paths, change_rng),
        "variance": variance(args.paths, variance_rng),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
