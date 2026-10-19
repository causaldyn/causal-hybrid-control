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


def test_the_tube_holds_its_level_at_every_step_where_its_premises_hold() -> None:
    """The control case: measured 0.950 to 0.958 over the 15 steps."""
    held, trusted = tube_coverage(World(), LOGS, RUNS)
    assert held.shape == (HORIZON,)
    assert held.min() >= FLOOR, np.round(held, 3)
    assert min(trusted) >= 1


def test_the_tube_holds_where_the_action_reaches_the_plant_late() -> None:
    """Two periods of dead time: the channel the log reads is near zero (0.02 for 0.8), so the plan
    barely acts, and the tube still holds the plant, 0.978 at its weakest step."""
    held, _ = tube_coverage(World(dead_time=2), LOGS, RUNS)
    assert held.min() >= FLOOR, np.round(held, 3)


def test_a_response_that_saturates_beyond_the_logged_actions_is_warned_of(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The log moves the incentive within about +-0.9, the box allows +-2, and the plant's
    response is ``0.5 tanh(u / 0.5)``. The tube falls to 0.57 at step 9; the plan sets the lever
    outside its logged range, and a ``lever_range`` warning says so on every log."""
    world = World(saturation=0.5, logged_gain=0.2, logged_spread=0.1)
    with caplog.at_level(logging.WARNING, logger="chc.decision"):
        held, _ = tube_coverage(world, LOGS, RUNS)
    ranges = [r for r in caplog.records if getattr(r, "chc_event", None) == "lever_range"]
    assert held.min() >= FLOOR or len(ranges) >= len(LOGS), np.round(held, 3)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "demand that persists, AR(0.8) within a unit, recorded and adjusted for: the tube's noise "
        "is a Gaussian AR(1) with one lag-1 coefficient (0.38 read), and the plant's error over "
        "steps 2-15 runs past it. Coverage 0.952 at step 1, 0.868 at step 4, 0.765 at step 15; "
        "trustworthy steps 1-2, where it is 0.952 and 0.918. No warning or field says so"
    ),
)
def test_the_tube_holds_its_level_under_a_persistent_confounder() -> None:
    held, _ = tube_coverage(World(demand_persistence=0.8), LOGS, RUNS)
    assert held.min() >= FLOOR, np.round(held, 3)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "an unlogged disturbance on supply, sd 2, AR(0.9), beside white demand: the plan's defect "
        "is an AR(1) plus white noise, whose lag-1 correlation (0.16 read) understates its "
        "persistence at every longer lag. Coverage 0.932 at step 1 (the only trustworthy step), "
        "0.890 at step 5. The certificate's noise_persistence_p reads 2e-31, but that check is the "
        "channel's, and the tube carries no warning"
    ),
)
def test_the_tube_holds_its_level_under_persistent_noise() -> None:
    held, _ = tube_coverage(World(disturbance=2.0, disturbance_persistence=0.9), LOGS, RUNS)
    assert held.min() >= FLOOR, np.round(held, 3)


def test_the_trustworthy_steps_hold_under_a_drift_outside_the_polynomial() -> None:
    """Supply's own drift is ``-0.3 sin(2 s)``, which the fit reads by a polynomial of degree 1.
    Over the trustworthy steps, 2 or 3 per log, the tube holds: 0.955 at its weakest."""
    held, trusted = tube_coverage(World(sine_drift=True), LOGS, RUNS)
    assert held[: min(trusted)].min() >= FLOOR, np.round(held, 3)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "supply's drift -0.3 sin(2 s), fitted by degree 1: the tube's rate reads 'global' and its "
        "mode 'exact', and past the trustworthy steps the plant leaves it more often than it "
        "says: 0.939 at step 8, 0.895 at step 15. No warning or field says so"
    ),
)
def test_the_whole_tube_holds_its_level_under_a_drift_outside_the_polynomial() -> None:
    held, _ = tube_coverage(World(sine_drift=True), LOGS, RUNS)
    assert held.min() >= FLOOR, np.round(held, 3)
