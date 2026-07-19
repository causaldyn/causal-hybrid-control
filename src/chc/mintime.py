"""Minimum-time (time-optimal) control via Pontryagin's minimum principle -- the double integrator.

For a control-affine plant with a bounded input, PMP makes the time-optimal control *bang-bang*: the
Hamiltonian is linear in ``u``, so the optimum sits on a bound, ``u = -u_max * sign(switching fn)``,
switching when it changes sign. For the double integrator ``x'' = u``, ``|u| <= u_max`` driven to
rest at the origin, that switching function is ``sigma = x + v|v|/(2 u_max)`` and the optimal
trajectory needs at most one switch (accelerate, then brake on the switching parabola).
The textbook complement to the quadratic-cost controllers in :mod:`chc.control` / :mod:`chc.lqr`:
minimise *time* to target, not a quadratic cost.

Scoped to the double integrator (general PMP is a costate two-point boundary-value problem); NumPy
float64, and the sign-based law is non-differentiable, so it lives outside the JAX control core.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BangBangResult:
    """A time-optimal rollout: the trajectory, the applied control, and its switch count."""

    states: NDArray[np.float64]  # (N+1, 2) rows [x, v]
    controls: NDArray[np.float64]  # (N,) applied bang-bang control, each +/- u_max
    time: float  # elapsed time to the origin (steps * dt)
    switches: int  # control sign changes (1 for a generic time-optimal run, 0 already on the curve)


def switching_function(x: float, v: float, u_max: float) -> float:
    """PMP switching function ``sigma = x + v|v|/(2 u_max)``; its sign picks the control."""
    return x + v * abs(v) / (2.0 * u_max)


def bang_bang_control(x: float, v: float, u_max: float = 1.0) -> float:
    """Time-optimal feedback for the double integrator: ``-u_max sign(sigma)``, the <=1-switch law.

    Off the switching curve ``-u_max sign(sigma)``; on it ``-u_max sign(v)`` (brake to the origin);
    at the origin ``0``.
    """
    sigma = switching_function(x, v, u_max)
    if sigma > 0.0:
        return -u_max
    if sigma < 0.0:
        return u_max
    if v != 0.0:
        return -math.copysign(u_max, v)
    return 0.0


def double_integrator_min_time(x0: float, v0: float, u_max: float = 1.0) -> float:
    """Closed-form minimum time to drive ``x''=u``, ``|u|<=u_max``, from ``(x0, v0)`` to the origin.

    From rest (``v0 = 0``) this is the familiar ``2 sqrt(|x0| / u_max)``.
    """
    sigma = switching_function(x0, v0, u_max)
    if sigma > 0.0:
        return (v0 + 2.0 * math.sqrt(v0**2 / 2.0 + u_max * x0)) / u_max
    if sigma < 0.0:
        return (-v0 + 2.0 * math.sqrt(v0**2 / 2.0 - u_max * x0)) / u_max
    return abs(v0) / u_max


def _switch_time(x0: float, v0: float, u_max: float, sigma: float) -> tuple[float, float]:
    """First-phase control and the single switch time for the time-optimal double-integrator run."""
    if sigma > 0.0:
        return -u_max, (v0 + math.sqrt(v0**2 / 2.0 + u_max * x0)) / u_max
    if sigma < 0.0:
        return u_max, (-v0 + math.sqrt(v0**2 / 2.0 - u_max * x0)) / u_max
    return (-math.copysign(u_max, v0) if v0 != 0.0 else 0.0), math.inf  # already on the curve


def bang_bang_rollout(
    x0: float, v0: float, u_max: float = 1.0, *, dt: float = 0.01
) -> BangBangResult:
    """Open-loop time-optimal trajectory: bang ``u1`` until the analytic switch time, then ``-u1``.

    Uses the closed-form switch time rather than a discrete ``sign(sigma)`` feedback, which would
    chatter across the switching curve; this gives the clean at-most-one-switch trajectory PMP
    predicts. Per-step integration is exact for the piecewise-constant control.
    """
    sigma = switching_function(x0, v0, u_max)
    u1, switch = _switch_time(x0, v0, u_max, sigma)
    total = double_integrator_min_time(x0, v0, u_max)
    x, v = float(x0), float(v0)
    states: list[tuple[float, float]] = [(x, v)]
    controls: list[float] = []
    steps = max(1, round(total / dt))
    for i in range(steps):
        u = u1 if i * dt < switch else -u1
        x += v * dt + 0.5 * u * dt * dt
        v += u * dt
        controls.append(u)
        states.append((x, v))
    switches = sum(1 for i in range(1, len(controls)) if controls[i] * controls[i - 1] < 0)
    return BangBangResult(np.array(states), np.array(controls), steps * dt, switches)
