"""Classical closed-loop step-response quality metrics.

Cost and oracle-regret (:mod:`chc.benchmark`) say *how well* a controller did on the task objective;
these say *how* it got there -- the transient-response vocabulary a control engineer reads first:
overshoot, settling time, rise time, steady-state error. They apply to any tracking/regulation
signal (one output component vs its target), independent of the controller that produced it.

Post-hoc trajectory analysis, so NumPy float64 throughout (like :mod:`chc.did`), not part of the
differentiable control core.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Signal = NDArray[np.float64]


def _prepare(signal: Signal, initial: float | None) -> tuple[Signal, float]:
    s = np.asarray(signal, dtype=np.float64)
    x0 = float(s[0]) if initial is None else float(initial)
    return s, x0


def overshoot(signal: Signal, target: float, *, initial: float | None = None) -> float:
    """Peak excursion past ``target`` as a fraction of the step ``target - initial`` (0 if none).

    ``0.05`` means the response overshot the target by 5% of the step. ``initial`` defaults to the
    first sample.
    """
    s, x0 = _prepare(signal, initial)
    step = target - x0
    if step == 0.0:
        return 0.0
    peak = float(np.max(s)) if step > 0 else float(np.min(s))
    return max((peak - target) / step, 0.0)


def settling_time(
    signal: Signal, target: float, dt: float, *, initial: float | None = None, tol: float = 0.02
) -> float:
    """Time after which ``signal`` stays within ``tol`` of the step and never leaves again.

    ``tol`` is a fraction of the step magnitude (2% by convention). Returns ``0.0`` if never outside
    the band and ``inf`` if still outside at the end of the record (did not settle).
    """
    s, x0 = _prepare(signal, initial)
    band = tol * abs(target - x0)
    outside = np.abs(s - target) > band
    if not outside.any():
        return 0.0
    last_outside = int(np.nonzero(outside)[0][-1])
    if last_outside == s.size - 1:
        return float("inf")
    return float((last_outside + 1) * dt)


def rise_time(
    signal: Signal,
    target: float,
    dt: float,
    *,
    initial: float | None = None,
    low: float = 0.1,
    high: float = 0.9,
) -> float:
    """Time to cross from ``low`` to ``high`` fraction of the step (10%-90% by convention)."""
    s, x0 = _prepare(signal, initial)
    step = target - x0
    low_level, high_level = x0 + low * step, x0 + high * step
    reached = (s >= low_level, s >= high_level) if step > 0 else (s <= low_level, s <= high_level)
    return float((int(np.argmax(reached[1])) - int(np.argmax(reached[0]))) * dt)


def steady_state_error(signal: Signal, target: float, *, window: int = 1) -> float:
    """Absolute error at rest: ``|mean(signal[-window:]) - target|`` (final value by default)."""
    s = np.asarray(signal, dtype=np.float64)
    return float(abs(float(np.mean(s[-window:])) - target))
