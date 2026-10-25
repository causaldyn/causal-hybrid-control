"""What the logger check cannot see, and what it does, read on the check itself.

``validation/confounded_log_equivalence.mac`` shows that a confounder the log does not hold, drawn
afresh each period and Gaussian, leaves the logged path exactly the law of a loop without it, whose
channel is ``b + c d/(d^2 + s^2)``; and that a confounder which persists makes the lever depend on
its own lag given the state, even where it does not move the plant. Each test simulates a plant
``x' = a x + b u + c w + sg e`` and its logger ``u = k x + d w + s eta`` and reads one of these on
the :class:`chc.evaluation.LoggerCheck` that ``chc.decision`` computes: the projection that makes
the two worlds one, the check's rejection rate against its level at two panel sizes, and the
persistence it sees.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from chc.decision import _check_logger, _Columns
from chc.evaluation import LoggerCheck
from chc.panel import Panel

FORMAL_LINKS = {
    "chc.evaluation.LoggerCheck": {
        "cites": ("confounded_log_equivalence.mac",),
        "tests": (
            "test_a_fresh_gaussian_confounder_leaves_the_noise_of_a_loop_without_one",
            "test_the_logger_check_rejects_a_fresh_confounder_at_its_level",
            "test_the_logger_check_sees_a_persistent_confounder_that_moves_nothing",
        ),
    },
}

A, B, K, S, SG = 0.9, 0.5, -0.3, 1.0, 1.0  # the plant, the logger's gain and the two noises
BURN = 50  # steps run before the log starts, so the log starts near the stationary law
ALPHA = 0.05
COLUMNS = _Columns(states=("x",), logged_on=())


def _draws(
    units: int, periods: int, rho: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The confounder ``w`` (unit variance, persistence ``rho``), the logger's ``eta`` and the
    plant's ``e``, each ``(BURN + periods, units)``."""
    total = BURN + periods
    nu = rng.standard_normal((total, units))
    w = np.empty((total, units))
    w[0] = nu[0]
    for t in range(1, total):
        w[t] = rho * w[t - 1] + math.sqrt(1.0 - rho**2) * nu[t]
    return w, rng.standard_normal((total, units)), rng.standard_normal((total, units))


def _path(
    k: float, logger_noise: np.ndarray, plant_noise: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The states ``x_t`` and the levers ``u_t = k x_t + logger_noise_t``, ``(BURN + periods,
    units)``, of ``x' = A x + B u + plant_noise`` from zero."""
    total, units = logger_noise.shape
    x = np.zeros((total + 1, units))
    u = np.empty((total, units))
    for t in range(total):
        u[t] = k * x[t] + logger_noise[t]
        x[t + 1] = A * x[t] + B * u[t] + plant_noise[t]
    return x, u


def _check(x: np.ndarray, u: np.ndarray) -> LoggerCheck:
    """The logger check on the logged periods, the levers read against the state alone."""
    states, levers = x[BURN:-1], u[BURN:]
    periods, units = levers.shape
    frame = {
        "unit": np.tile(np.arange(units), periods),
        "time": np.repeat(np.arange(periods), units),
        "x": states.ravel(),
        "u": levers.ravel(),
    }
    panel = Panel.from_frame(frame, unit="unit", time="time", seed=0)
    check = _check_logger(panel, levers=("u",), columns=COLUMNS)
    assert isinstance(check, LoggerCheck)
    return check


def _channel(x: np.ndarray, u: np.ndarray) -> float:
    """The channel an adjustment on the state reads: least squares of ``x'`` on ``(1, x, u)``
    over the logged steps."""
    now, ahead, levers = x[BURN:-1].ravel(), x[BURN + 1 :].ravel(), u[BURN:].ravel()
    design = np.column_stack([np.ones_like(now), now, levers])
    return float(np.linalg.lstsq(design, ahead, rcond=None)[0][2])


def test_a_fresh_gaussian_confounder_leaves_the_noise_of_a_loop_without_one() -> None:
    """STEP 2: the plant's noise less its projection on the logger's, ``c w + sg e - beta0 v``,
    is uncorrelated with ``v = d w + s eta`` and with the state, and has the variance
    ``sg^2 + c^2 s^2/(d^2 + s^2)``; the loop ``x' = (a - beta0 k) x + (b + beta0) u`` driven by it
    is the logged path itself."""
    c, d = 1.0, 1.0
    beta = c * d / (d**2 + S**2)
    w, eta, e = _draws(400, 12, 0.0, np.random.default_rng([2026, 10, 25]))
    v = d * w + S * eta
    x, u = _path(K, v, c * w + SG * e)
    residual = c * w + SG * e - beta * v
    n = residual.size
    # Sample correlations of independent draws scatter by 1/sqrt(n): 4 of those is the band.
    assert abs(np.corrcoef(residual.ravel(), v.ravel())[0, 1]) < 4.0 / math.sqrt(n)
    assert abs(np.corrcoef(residual.ravel(), x[:-1].ravel())[0, 1]) < 4.0 / math.sqrt(n)
    variance = SG**2 + c**2 * S**2 / (d**2 + S**2)
    # a Gaussian's sample variance scatters by sqrt(2/n) of itself
    assert residual.var() == pytest.approx(variance, rel=4.0 * math.sqrt(2.0 / n), abs=0.0)
    twin = np.zeros_like(x)
    for t in range(u.shape[0]):
        twin[t + 1] = (A - beta * K) * twin[t] + (B + beta) * u[t] + residual[t]
    assert np.max(np.abs(twin - x)) <= 1e-12 * np.max(np.abs(x))


@pytest.mark.parametrize(("units", "panels"), [(50, 400), (400, 200)])
def test_the_logger_check_rejects_a_fresh_confounder_at_its_level(units: int, panels: int) -> None:
    """STEP 3: the confounder moves the channel an adjustment on the state reads from 0.5 to 1.0,
    and the check rejects at its level, at 50 units as at 400: its power against the confounder is
    its size. Over five seeds of 400 panels the rate read 0.040 to 0.065."""
    c, d = 1.0, 1.0
    rejected, channels = 0, []
    for panel in range(panels):
        w, eta, e = _draws(units, 12, 0.0, np.random.default_rng([2026, units, panel]))
        x, u = _path(K, d * w + S * eta, c * w + SG * e)
        rejected += _check(x, u).test.p_value <= ALPHA
        channels.append(_channel(x, u))
    # A binomial count at the level scatters by sqrt(alpha (1 - alpha)/panels): 4 of those.
    assert rejected / panels <= ALPHA + 4.0 * math.sqrt(ALPHA * (1.0 - ALPHA) / panels)
    spread = np.std(channels, ddof=1) / math.sqrt(panels)
    assert np.mean(channels) == pytest.approx(B + c * d / (d**2 + S**2), abs=4.0 * spread)


def test_the_logger_check_sees_a_persistent_confounder_that_moves_nothing() -> None:
    """STEP 4b: at ``c = 0`` the confounder does not move the plant, and the channel read on the
    state is ``b``; persistent at 0.8, it makes the lever depend on its own lag given the state,
    and the check rejects nearly every panel. It reads the logger's memory, not confounding."""
    rejected, channels = 0, []
    panels = 50
    for panel in range(panels):
        w, eta, e = _draws(50, 12, 0.8, np.random.default_rng([2026, 8, panel]))
        x, u = _path(0.0, w + S * eta, SG * e)
        rejected += _check(x, u).test.p_value <= ALPHA
        channels.append(_channel(x, u))
    assert rejected / panels >= 0.9
    spread = np.std(channels, ddof=1) / math.sqrt(panels)
    assert np.mean(channels) == pytest.approx(B, abs=4.0 * spread)
