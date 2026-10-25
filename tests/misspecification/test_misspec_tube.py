"""The tube and its trustworthy steps on plants the decision's model gets wrong.

``prescribe`` states that the plant stays inside the plan's tube with probability 0.95 at each
step, where its noise is Gaussian and AR(1), and that the trustworthy steps are where the tube is
inside the tolerance. Each test fits one log per seed of a plant from :mod:`_misspec_world`, runs
the plant 2000 times under the plan's actions from the plan's start, each run with its own
demand and disturbance, and reads, per step, the share of runs the tube holds, pooled over the
logs. The contract is that share at 0.95 or more, less three standard errors of a mean over the
logs, or a warning that says the premise fails. The logs, not the runs, size the floor: one log's
share varies from log to log by more than its 2000 runs' binomial error (0.0049), and by how much
depends on the plant, so each test states its plant's standard deviation, measured over 64 logs of
2000 runs. A test marked ``xfail`` is a case that does neither; its reason gives the numbers
measured.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from _misspec_world import HORIZON, World, ask, tube_coverage

from chc import decision

pytestmark = pytest.mark.misspecification

LOGS, RUNS = range(4), 2000


def _floor(per_log_sd: float, logs: range) -> float:
    """The level less three standard errors of the mean share over ``logs``."""
    return 0.95 - 3.0 * per_log_sd / np.sqrt(len(logs))


def _holds_or_says_so(held: np.ndarray, warned: tuple[bool, ...], per_log_sd: float) -> bool:
    """The tube holds at every step, or a ``tube`` warning was logged on 90 % of the logs."""
    return bool(held.min() >= _floor(per_log_sd, LOGS) or np.mean(warned) >= 0.9)


def test_the_tube_holds_its_level_at_every_step_where_its_premises_hold() -> None:
    """The control case, over 16 logs: 0.949 at its weakest step. One log's share has a standard
    deviation of 0.0078, so the floor is 0.944."""
    logs = range(16)
    held, trusted, _ = tube_coverage(World(), logs, RUNS)
    assert held.shape == (HORIZON,)
    assert held.min() >= _floor(0.0078, logs), np.round(held, 3)
    assert min(trusted) >= 1


def test_the_tube_holds_where_the_action_reaches_the_plant_late() -> None:
    """Two periods of dead time: the channel the log reads is near zero (0.02 for 0.8), so the plan
    barely acts. The tube holds the plant on average over the logs, 0.950 at its weakest step over
    64, but not on each log: one log's share has a standard deviation of 0.077 there, sixteen
    times the runs' binomial error, so 16 logs read 0.937 and the floor is 0.892."""
    logs = range(16)
    held, _, _ = tube_coverage(World(dead_time=2), logs, RUNS)
    assert held.min() >= _floor(0.077, logs), np.round(held, 3)


def test_a_response_that_saturates_beyond_the_logged_actions_is_warned_of(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The log moves the incentive within about +-0.9, the box allows +-2, and the plant's
    response is ``0.5 tanh(u / 0.5)``. The tube falls to 0.57 at step 9 (one log's share has a
    standard deviation of 0.066); the plan sets the lever outside its logged range, and a
    ``lever_range`` warning says so on every log, each log read on its own."""
    world = World(saturation=0.5, logged_gain=0.2, logged_spread=0.1)
    held, _, _ = tube_coverage(world, LOGS, RUNS)
    ranged = []
    for seed in LOGS:
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="chc.decision"):
            ask(world, seed)
        ranged.append(any(getattr(r, "chc_event", None) == "lever_range" for r in caplog.records))
    assert held.min() >= _floor(0.066, LOGS) or all(ranged), (np.round(held, 3), ranged)


def test_the_tube_holds_its_level_under_a_persistent_confounder() -> None:
    """Demand that persists, AR(0.8) within a unit, recorded and adjusted for, read over 16 logs.
    Without its read's error carried the tube fell to 0.9444 by step 13; with it, over 256 further
    logs, it holds 0.9484 at its weakest step, 0.0008 its standard error over the logs, and 0.0009
    of the 0.0016 short is the VAR(1)'s reading of an error that is AR(0.8) plus white (ADR 0105).
    Here the floor counts the logs, not the runs: one log's share has a standard deviation of
    0.0166 at step 13 over those 256 logs, where 2000 runs alone give 0.0049, so a floor sized by
    the runs turned on a last digit: these 16 logs read 0.94628 against its 0.94634 up to 0.21.0,
    and 0.9483 since the push's mean on the log is read across the unit's periods (ADR 0133)."""
    logs = range(16)
    held, _, warned = tube_coverage(World(demand_persistence=0.8), logs, RUNS)
    assert held.min() >= _floor(0.0166, logs) or np.mean(warned) >= 0.9, (np.round(held, 4), warned)


def test_the_noise_read_s_error_is_what_holds_the_tube_under_a_persistent_confounder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The floor above cannot tell the tube from one that drops its noise read's error: that tube
    held 0.9465 at its weakest step, inside the floor's three errors of 0.0166 over 16 logs. So the
    same 16 logs and the same runs are read again with the read's error left out, and the two
    tubes are compared run for run: the read's error widens the tube at every step, by 0.0007 to
    0.0024 of coverage here, and lifts the weakest step from 0.9465 to 0.9483."""
    world, logs = World(demand_persistence=0.8), range(16)
    held, _, _ = tube_coverage(world, logs, RUNS)
    monkeypatch.setattr(decision, "_noise_read", lambda *args, **kwargs: None)
    without, _, _ = tube_coverage.__wrapped__(world, logs, RUNS)
    assert np.all(held - without > 0.0), np.round(held - without, 4)
    assert held.min() - without.min() >= 0.001, (held.min(), without.min())


def test_persistent_noise_is_held_or_warned_of() -> None:
    """An unlogged disturbance on supply, sd 2, AR(0.9), beside white demand: the fit absorbs part
    of it into the drift, and the tube falls to 0.902 at step 5 (0.918 at step 1). The channel's
    residual persists within units, and a ``tube`` warning says the tube's level is not assured
    on each of the four logs. One log's share has a standard deviation of 0.018."""
    held, _, warned = tube_coverage(World(disturbance=2.0, disturbance_persistence=0.9), LOGS, RUNS)
    assert _holds_or_says_so(held, warned, 0.018), (np.round(held, 3), warned)


def test_the_trustworthy_steps_hold_under_a_drift_outside_the_polynomial() -> None:
    """Supply's own drift is ``-0.3 sin(2 s)``, which the fit reads by a polynomial of degree 1.
    Over the trustworthy steps, 2 or 3 per log, the tube holds: 0.950 at its weakest. Over the
    first two steps one log's share has a standard deviation of 0.0065, so the floor is 0.940."""
    held, trusted, _ = tube_coverage(World(sine_drift=True), LOGS, RUNS)
    assert held[: min(trusted)].min() >= _floor(0.0065, LOGS), np.round(held, 3)


def test_the_whole_tube_holds_or_warns_under_a_drift_outside_the_polynomial() -> None:
    """Past the trustworthy steps the plant leaves the tube more often than it says, 0.888 at
    step 15, and a ``tube`` warning says the tube's level is not assured on each of the four
    logs. One log's share has a standard deviation of 0.019."""
    held, _, warned = tube_coverage(World(sine_drift=True), LOGS, RUNS)
    assert _holds_or_says_so(held, warned, 0.019), (np.round(held, 3), warned)
