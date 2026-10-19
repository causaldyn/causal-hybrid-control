"""The lift fit's check and interval where each test's noise persists over its periods.

:func:`~chc.lift.check_observational` rejects a channel at 5 %, and
:meth:`~chc.lift.LiftFit.interval` states a 95 % interval. The contract read here, over 40 worlds
of four go-dark tests whose noise is AR(0.8) over each test's 16 periods: the true channel is
rejected on at most 5 % of worlds, and the coefficient's interval covers on at least 95 %, each
within three Monte Carlo standard deviations (``sqrt(0.05 * 0.95 / 40) = 0.034``), or the fit says
its reading fails: :attr:`~chc.lift.LiftFit.serial_p_value` below 0.05, where it logs a WARNING
that names ``serial="ar1"``. The worlds are ``tests/test_lift.py``'s.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest

from chc.lift import GeoArm, LiftTest, check_observational, fit_lift
from chc.response import Channel, GeometricAdstock, Tanh

pytestmark = pytest.mark.misspecification

TRUE = {"kernel.retention": 0.2, "curve.scale": 250.0, "coefficient": 1100.0}
STARTS = (20, 55, 100, 140)
LEVELS = (0.7, 1.7, 0.85, 0.75)
WORLDS = 40
SLACK = 3.0 * math.sqrt(0.05 * 0.95 / WORLDS)


def _channel(retention: float, scale: float, coefficient: float) -> Channel:
    return Channel(GeometricAdstock(retention, length=6, normalized=True), Tanh(scale), coefficient)


START = _channel(0.5, 100.0, 1.0)


def _world(seed: int, persistence: float, noise: float = 55.0) -> tuple[LiftTest, ...]:
    """Go-dark tests on a random spend path, read over 4 weeks before, 4 dark and 8 after; each
    test's noise an AR(1) of coefficient ``persistence``, of the same marginal sd."""
    truth = _channel(*TRUE.values())
    rng = np.random.default_rng(seed)
    spend = np.exp(np.log(75.0) + 0.35 * rng.standard_normal(156))
    for start, level in zip(STARTS, LEVELS, strict=True):
        spend[start - 10 : start + 12] *= level
    tests = []
    for start in STARTS:
        dark = spend.copy()
        dark[start : start + 4] = 0.0
        window = slice(start - 4, start + 12)
        noisy = rng.normal(0.0, noise / 2.0, 16)
        for t in range(1, 16):
            noisy[t] = persistence * noisy[t - 1] + math.sqrt(1.0 - persistence**2) * noisy[t]
        tests.append(
            LiftTest(
                treated=GeoArm(
                    dark[: window.stop], 2000.0 + np.asarray(truth(dark))[window] - noisy
                ),
                control=GeoArm(
                    spend[: window.stop], 2000.0 + np.asarray(truth(spend))[window] + noisy
                ),
            )
        )
    return tuple(tests)


@functools.cache
def _read(serial: str) -> tuple[float, float]:
    """Over the worlds: the share whose check rejects the true channel, and the share whose
    coefficient interval covers it, a world whose fit says its reading fails counted with
    neither the first nor the second's misses."""
    truth = _channel(*TRUE.values())
    rejected = covered = 0
    for seed in range(WORLDS):
        fit = fit_lift(_world(seed, 0.8), START, serial=serial)  # type: ignore[arg-type]
        said = fit.serial_p_value is not None and fit.serial_p_value < 0.05
        rejected += check_observational(fit, truth).rejected and not said
        low, high = fit.interval("coefficient")
        covered += low <= TRUE["coefficient"] <= high or said
    return rejected / WORLDS, covered / WORLDS


def test_an_ar1_reading_keeps_the_level_where_the_noise_persists() -> None:
    """``serial="ar1"``: the check rejects on none of the 40, and the interval covers on all."""
    rejected, covered = _read("ar1")
    assert rejected <= 0.05 + SLACK
    assert covered >= 0.95 - SLACK


def test_the_default_keeps_the_level_where_the_noise_persists_or_says_so() -> None:
    """``serial="independent"``: at AR(0.8) the check rejects the true channel on 14 of 40 worlds
    where it states 5 %, and the coefficient's interval covers on 31; the fit's serial check
    rejects independence on all 40, and says so."""
    rejected, covered = _read("independent")
    assert rejected <= 0.05 + SLACK
    assert covered >= 0.95 - SLACK
