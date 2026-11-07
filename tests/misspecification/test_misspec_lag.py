"""The lagged-lever check on a plant that reads its lever late.

``lag_check`` regresses a fit's moment residual on its levers some periods back, the fit's own
regressors held: under the fit's premise those slopes are zero. On :mod:`_misspec_world`'s plant
with a dead time it rejects and names the lag; without one it rejects at its size. The seeds are
read from the run's block (:mod:`_misspec_counts`).
"""

from __future__ import annotations

import functools
import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from _misspec_counts import at_most, seeds
from _misspec_world import DT, World

from chc.dynamics import LinearDynamics
from chc.dynamics_id import fit_causal_residual, lag_check

pytestmark = pytest.mark.misspecification

KNOWN = LinearDynamics(jnp.zeros((2, 2)), jnp.zeros((2, 1)))
UNITS, STEPS = 200, 12


@functools.cache
def _read(dead_time: int, seed: int) -> tuple[float, int | None]:
    """The check's p-value and named lag on one log of ``UNITS`` units over ``STEPS`` periods."""
    frame = World(dead_time=dead_time).log(seed, UNITS, STEPS)
    states = np.stack([frame["supply"], frame["wait"]], 1).reshape(UNITS, STEPS, 2)
    levers = frame["incentive"].reshape(UNITS, STEPS)
    demand = frame["demand"].reshape(UNITS, STEPS)
    data = {
        "x": jnp.asarray(states[:, :-1].reshape(-1, 2)),
        "x_next": jnp.asarray(states[:, 1:].reshape(-1, 2)),
        "u": jnp.asarray(levers[:, :-1].reshape(-1, 1)),
        "z": jnp.asarray(demand[:, :-1].reshape(-1, 1)),
    }
    units = np.repeat(np.arange(UNITS), STEPS - 1)
    periods = np.tile(np.arange(STEPS - 1), UNITS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_causal_residual(
            KNOWN, data, DT, adjust_for=("z",), integrator="euler", clusters=units
        )
    check = lag_check(fit, data, units, periods, adjust_for=("z",))
    assert check.transitions == UNITS * (STEPS - 1 - 3)
    assert check.groups == UNITS
    return check.p_value, check.lag


@pytest.mark.parametrize("seed", seeds(0, 6))
def test_a_lever_read_two_periods_late_is_named(seed: int) -> None:
    p_value, lag = _read(2, seed)
    assert p_value <= 1e-6
    assert lag == 2


@pytest.mark.parametrize("seed", seeds(0, 2))
def test_a_lever_read_one_period_late_is_named(seed: int) -> None:
    p_value, lag = _read(1, seed)
    assert p_value <= 1e-6
    assert lag == 1


def test_without_a_dead_time_the_check_rejects_at_its_size() -> None:
    """200 control logs, rejected at 0.05 on at most 24 and at 0.2 on at most 70, the ceilings of
    ``Binomial(200, 0.06)`` and ``Binomial(200, 0.25)`` at ``alpha = 1/1000``: over 1000 logs the
    check rejected 0.056 and 0.214 of them, a little above its levels. Block 0: 12 and 45. A
    p-value 0.33 times its own rejects ``p1`` = 0.161 and 0.547, where the ceilings fail a
    block with probability 0.93 and above 0.999999; the second fails with probability 0.8 down to a
    p-value 0.52 times its own, its minimal detectable defect."""
    p_values = np.array([_read(0, seed)[0] for seed in seeds(0)])
    rejected = {level: int(np.sum(p_values <= level)) for level in (0.05, 0.2)}
    assert rejected[0.05] <= at_most("0.06"), rejected
    assert rejected[0.2] <= at_most("0.25"), rejected


def test_lag_check_refuses_labels_that_do_not_pair_and_a_lag_that_is_not_one() -> None:
    frame = World().log(0, 20, 6)
    states = np.stack([frame["supply"], frame["wait"]], 1).reshape(20, 6, 2)
    data = {
        "x": jnp.asarray(states[:, :-1].reshape(-1, 2)),
        "x_next": jnp.asarray(states[:, 1:].reshape(-1, 2)),
        "u": jnp.asarray(frame["incentive"].reshape(20, 6)[:, :-1].reshape(-1, 1)),
        "z": jnp.asarray(frame["demand"].reshape(20, 6)[:, :-1].reshape(-1, 1)),
    }
    units, periods = np.repeat(np.arange(20), 5), np.tile(np.arange(5), 20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_causal_residual(KNOWN, data, DT, adjust_for=("z",), integrator="euler")
    with pytest.raises(ValueError, match="positive whole number"):
        lag_check(fit, data, units, periods, lags=0, adjust_for=("z",))
    with pytest.raises(ValueError, match="one each"):
        lag_check(fit, data, units[:-1], periods, adjust_for=("z",))
