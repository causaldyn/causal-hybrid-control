"""The lagged-lever check on a plant that reads its lever late.

``lag_check`` regresses a fit's moment residual on its levers some periods back, the fit's own
regressors held: under the fit's premise those slopes are zero. On :mod:`_misspec_world`'s plant
with a dead time it rejects and names the lag; without one it rejects at its size.
"""

from __future__ import annotations

import functools
import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from _misspec_world import DT, World

from chc.dynamics import LinearDynamics
from chc.dynamics_id import fit_causal_residual, lag_check

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


@pytest.mark.parametrize("seed", range(6))
def test_a_lever_read_two_periods_late_is_named(seed: int) -> None:
    p_value, lag = _read(2, seed)
    assert p_value <= 1e-6
    assert lag == 2


@pytest.mark.parametrize("seed", range(2))
def test_a_lever_read_one_period_late_is_named(seed: int) -> None:
    p_value, lag = _read(1, seed)
    assert p_value <= 1e-6
    assert lag == 1


def test_without_a_dead_time_the_check_rejects_at_its_size() -> None:
    """Six control logs: at 0.05 the check rejected on 1, seed 1 (p = 0.039), where 2 or more
    would have a chance of 0.033 at the check's size; on 60 it rejected on 5."""
    rejected = [_read(0, seed)[0] <= 0.05 for seed in range(6)]
    assert sum(rejected) <= 1


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
