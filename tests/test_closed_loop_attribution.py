"""chc.dynamics_id: what a setpoint-tracked log identifies, and what it manufactures (§41 (e)).

Result 41 left open why the interaction coefficient comes out large and negative on a tracked zone.
A proportional loop puts the log on an affine manifold, and on that manifold only the interaction
reaches the action's quadratic term -- so it is the identified coefficient while the pole is not,
and a misspecified curvature is reported as `-gain * curvature`, growing with the controller.
"""

import numpy as np
import pytest

from chc.dynamics_id import (
    closed_loop_attribution_certificate,
    closed_loop_gain_attribution,
)


@pytest.fixture(scope="module")
def certificate():
    return closed_loop_attribution_certificate()


def _tracked_log(gain: float, curvature: float, samples: int = 3000, seed: int = 1):
    rng = np.random.default_rng(seed)
    actions = 20.0 + 2.0 * rng.standard_normal(samples)
    states = 17.0 - actions / gain
    rates = 1.0 + 0.4 * actions + curvature * actions**2
    return states, actions, rates


def test_the_log_reveals_the_controller_gain_not_the_plant() -> None:
    states, actions, rates = _tracked_log(gain=2.6, curvature=0.1454)
    fit = closed_loop_gain_attribution(states, actions, rates)
    assert fit.implied_gain == pytest.approx(2.6, rel=1e-9)
    assert fit.manifold_r2 == pytest.approx(1.0, abs=1e-12)
    assert fit.exploration_budget < 1e-20  # exact tracking: nothing off the manifold


def test_the_interaction_is_the_identified_coefficient_on_a_tracked_log() -> None:
    states, actions, rates = _tracked_log(gain=2.6, curvature=0.1454)
    fit = closed_loop_gain_attribution(states, actions, rates)
    # b1 = C/m to machine precision even though the design is singular -- the pole is what is lost.
    assert fit.fitted_interaction == pytest.approx(fit.predicted_interaction, rel=1e-9)
    assert fit.design_condition > 1e10


@pytest.mark.parametrize("gain", [0.5, 1.0, 2.6, 8.0])
def test_the_manufactured_interaction_grows_linearly_with_the_loop_gain(gain: float) -> None:
    states, actions, rates = _tracked_log(gain=gain, curvature=0.1454)
    fit = closed_loop_gain_attribution(states, actions, rates)
    assert fit.fitted_interaction == pytest.approx(-gain * 0.1454, rel=1e-6)


def test_a_real_interaction_survives_exact_tracking_while_the_pole_does_not(certificate) -> None:
    assert certificate.ok
    assert all(
        b == pytest.approx(certificate.true_interaction, abs=1e-6) for b in certificate.recovered
    )
    assert max(certificate.drift_error) > 1e-2


def test_the_standing_guess_of_minus_one_over_the_gain_is_refuted(certificate) -> None:
    # -1/gain is the MANIFOLD slope. It shrinks with the gain while the interaction grows, so the
    # two cannot be the same quantity -- they even move in opposite directions.
    growing = certificate.spurious
    guess = certificate.refuted_guess
    assert abs(growing[-1]) > abs(growing[0])
    assert abs(guess[-1]) < abs(guess[0])


def test_exploration_off_the_manifold_buys_the_drift_back(certificate) -> None:
    assert certificate.drift_error_by_exploration[0] > 1e-2
    assert certificate.drift_error_by_exploration[-1] < 1e-6
    assert certificate.condition_by_exploration[-1] < certificate.condition_by_exploration[0]
