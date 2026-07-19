"""Time-optimal bang-bang for the double integrator: PMP closed form + <=1-switch structure."""

from __future__ import annotations

import math

import numpy as np
import pytest

from chc.mintime import bang_bang_control, bang_bang_rollout, double_integrator_min_time

STATES = [(-1.0, 0.0), (1.0, 0.0), (2.0, -1.0), (-0.5, 1.5), (0.0, 2.0), (3.0, 0.0)]


def _feedback_time_to_origin(x0: float, v0: float, u_max: float, dt: float = 1e-4) -> float:
    """Independent check: integrate the sign(sigma) feedback to the origin ball, return the time."""
    x, v, t = x0, v0, 0.0
    while t < 30.0 and math.hypot(x, v) > 2e-3:
        u = bang_bang_control(x, v, u_max)
        x += v * dt + 0.5 * u * dt * dt
        v += u * dt
        t += dt
    return t


@pytest.mark.parametrize(("x0", "v0"), STATES)
def test_closed_form_min_time_matches_a_feedback_simulation(x0: float, v0: float) -> None:
    analytic = double_integrator_min_time(x0, v0, 1.0)
    assert _feedback_time_to_origin(x0, v0, 1.0) == pytest.approx(analytic, abs=0.01)


def test_rest_to_rest_time_is_two_root_distance() -> None:
    for x0 in (0.5, 1.0, 4.0):
        assert double_integrator_min_time(x0, 0.0, 1.0) == pytest.approx(2.0 * math.sqrt(x0))


@pytest.mark.parametrize(("x0", "v0"), STATES)
def test_open_loop_rollout_reaches_the_origin(x0: float, v0: float) -> None:
    result = bang_bang_rollout(x0, v0, 1.0, dt=1e-3)
    assert np.linalg.norm(result.states[-1]) < 0.05  # the analytic switch time drives it home


def test_control_is_bang_bang_and_switches_at_most_once() -> None:
    for x0, v0 in STATES:
        result = bang_bang_rollout(x0, v0, 1.0, dt=1e-3)
        assert np.all(np.abs(np.abs(result.controls) - 1.0) < 1e-9)  # every control is +/- u_max
        assert result.switches <= 1  # PMP: at most one switch


def test_generic_state_switches_exactly_once() -> None:
    assert bang_bang_rollout(3.0, 0.0, 1.0, dt=1e-3).switches == 1  # accelerate, then brake


def test_stronger_actuator_reaches_the_target_faster() -> None:
    assert double_integrator_min_time(4.0, 0.0, 4.0) < double_integrator_min_time(4.0, 0.0, 1.0)
