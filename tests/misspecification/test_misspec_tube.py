"""The tube and its trustworthy steps on plants the decision's model gets wrong.

``prescribe`` states that the plant stays inside the plan's tube with probability 0.95 at each
step, where its noise is Gaussian and AR(1), and that the trustworthy steps are where the tube is
inside the tolerance. Each test fits one log per seed of a plant from :mod:`_misspec_world`, runs
the plant 2000 times under the plan's actions from the plan's start, each run with its own
demand and disturbance, and reads, per step, the share of runs the tube holds, pooled over the
logs. The contract is that share at 0.95 or more, less three Monte Carlo standard deviations
(``sqrt(0.95 * 0.05 / 8000) = 0.0024``), or a warning that says the premise fails. A test marked
``xfail`` is a case that does neither; its reason gives the numbers measured.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from _misspec_world import HORIZON, World, tube_coverage

pytestmark = pytest.mark.misspecification

LOGS, RUNS = range(4), 2000
FLOOR = 0.95 - 3.0 * np.sqrt(0.95 * 0.05 / (len(LOGS) * RUNS))


def _holds_or_says_so(held: np.ndarray, warned: tuple[bool, ...]) -> bool:
    """The tube holds at every step, or a ``tube`` warning was logged on 90 % of the logs."""
    return bool(held.min() >= FLOOR or np.mean(warned) >= 0.9)


def test_the_tube_holds_its_level_at_every_step_where_its_premises_hold() -> None:
    """The control case: measured 0.944 to 0.952 over the 15 steps."""
    held, trusted, _ = tube_coverage(World(), LOGS, RUNS)
    assert held.shape == (HORIZON,)
    assert held.min() >= FLOOR, np.round(held, 3)
    assert min(trusted) >= 1


def test_the_tube_holds_where_the_action_reaches_the_plant_late() -> None:
    """Two periods of dead time: the channel the log reads is near zero (0.02 for 0.8), so the plan
    barely acts, and the tube still holds the plant, 0.978 at its weakest step."""
    held, _, _ = tube_coverage(World(dead_time=2), LOGS, RUNS)
    assert held.min() >= FLOOR, np.round(held, 3)


def test_a_response_that_saturates_beyond_the_logged_actions_is_warned_of(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The log moves the incentive within about +-0.9, the box allows +-2, and the plant's
    response is ``0.5 tanh(u / 0.5)``. The tube falls to 0.57 at step 9; the plan sets the lever
    outside its logged range, and a ``lever_range`` warning says so on every log."""
    world = World(saturation=0.5, logged_gain=0.2, logged_spread=0.1)
    with caplog.at_level(logging.WARNING, logger="chc.decision"):
        held, _, _ = tube_coverage(world, LOGS, RUNS)
    ranges = [r for r in caplog.records if getattr(r, "chc_event", None) == "lever_range"]
    assert held.min() >= FLOOR or len(ranges) >= len(LOGS), np.round(held, 3)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "demand that persists, AR(0.8) within a unit, recorded and adjusted for: the tube carries "
        "the noise as a VAR(1) and holds 0.950 over steps 1-3, but 0.942 at step 5, below the "
        "floor of 0.9427, and 0.944 to 0.948 past it, where the control case reads 0.944 at its "
        "weakest (16 logs: 0.944 to 0.950). A tube warning on 1 of 4 logs; trustworthy step 1"
    ),
)
def test_the_tube_holds_its_level_under_a_persistent_confounder() -> None:
    held, _, warned = tube_coverage(World(demand_persistence=0.8), LOGS, RUNS)
    assert _holds_or_says_so(held, warned), (np.round(held, 4), warned)


def test_persistent_noise_is_held_or_warned_of() -> None:
    """An unlogged disturbance on supply, sd 2, AR(0.9), beside white demand: the fit absorbs part
    of it into the drift, and the tube falls to 0.899 at step 5 (0.918 at step 1). The channel's
    residual persists within units, and a ``tube`` warning says the tube's level is not assured
    on each of the four logs."""
    held, _, warned = tube_coverage(World(disturbance=2.0, disturbance_persistence=0.9), LOGS, RUNS)
    assert _holds_or_says_so(held, warned), (np.round(held, 3), warned)


def test_the_trustworthy_steps_hold_under_a_drift_outside_the_polynomial() -> None:
    """Supply's own drift is ``-0.3 sin(2 s)``, which the fit reads by a polynomial of degree 1.
    Over the trustworthy steps, 2 or 3 per log, the tube holds: 0.955 at its weakest."""
    held, trusted, _ = tube_coverage(World(sine_drift=True), LOGS, RUNS)
    assert held[: min(trusted)].min() >= FLOOR, np.round(held, 3)


def test_the_whole_tube_holds_or_warns_under_a_drift_outside_the_polynomial() -> None:
    """Past the trustworthy steps the plant leaves the tube more often than it says, 0.876 at
    step 15, and a ``tube`` warning says the tube's level is not assured on each of the four
    logs."""
    held, _, warned = tube_coverage(World(sine_drift=True), LOGS, RUNS)
    assert _holds_or_says_so(held, warned), (np.round(held, 3), warned)
