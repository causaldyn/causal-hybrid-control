"""Classical step-response metrics match the analytic second-order response."""

from __future__ import annotations

import numpy as np
import pytest

from chc.metrics import overshoot, rise_time, settling_time, steady_state_error

DT = 0.001


def _second_order_step(zeta: float, wn: float = 5.0, horizon: float = 12.0) -> np.ndarray:
    """Unit step response (0 -> 1) of a second-order system with damping ``zeta``."""
    t = np.arange(0.0, horizon, DT)
    if zeta < 1.0:
        wd = wn * np.sqrt(1 - zeta**2)
        envelope = np.exp(-zeta * wn * t) / np.sqrt(1 - zeta**2)
        return 1.0 - envelope * np.sin(wd * t + np.arccos(zeta))
    return 1.0 - np.exp(-wn * t) * (1.0 + wn * t)  # critically damped: no overshoot


def test_overshoot_matches_the_analytic_formula() -> None:
    zeta = 0.2
    analytic = np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2))  # Mp = exp(-pi z / sqrt(1 - z^2))
    assert overshoot(_second_order_step(zeta), 1.0) == pytest.approx(analytic, abs=0.01)


def test_critically_damped_response_has_no_overshoot() -> None:
    assert overshoot(_second_order_step(1.0), 1.0) == 0.0  # no excursion past the target


def test_settling_time_near_the_four_over_zeta_wn_rule() -> None:
    zeta, wn = 0.2, 5.0
    ts = settling_time(_second_order_step(zeta, wn), 1.0, DT, tol=0.02)
    assert ts == pytest.approx(4.0 / (zeta * wn), rel=0.15)  # 2% settling ~ 4 / (zeta * wn)


def test_rise_time_is_positive_and_precedes_settling() -> None:
    y = _second_order_step(0.2)
    assert 0.0 < rise_time(y, 1.0, DT) < settling_time(y, 1.0, DT)  # rises before it settles


def test_settling_time_is_infinite_when_the_response_never_enters_the_band() -> None:
    ramp = np.linspace(0.0, 0.5, 5000)  # rises only halfway; target 1.0 is never approached
    assert settling_time(ramp, 1.0, DT, initial=0.0) == float("inf")


def test_steady_state_error_reads_the_final_offset() -> None:
    settled = np.concatenate([np.linspace(0.0, 0.9, 500), np.full(500, 0.9)])  # rests at 0.9
    assert steady_state_error(settled, 1.0, window=100) == pytest.approx(0.1, abs=1e-6)
