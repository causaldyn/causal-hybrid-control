"""The pendulum case study: one log, three causal claims, every schedule run on the true plant.

The script is the deliverable, so the test drives the file itself rather than a copy of it. The
thresholds were set on seeds 0-3 at this panel size, in float64 as the suite runs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import pytest

from chc.decision import NotIdentifiedError

if TYPE_CHECKING:
    from scripts.pendulum_demo import PendulumCase


def _load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "pendulum_demo.py"
    spec = importlib.util.spec_from_file_location("pendulum_demo", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(module)
    return module


demo = _load_script()


@pytest.fixture(scope="module")
def case() -> PendulumCase:
    """One run shared by every test: three `prescribe` calls, two of them solving a held barrier."""
    return demo.run_pendulum(episodes=10, steps=60)


def test_the_three_readings_are_three_identification_statuses(case: PendulumCase) -> None:
    statuses = [reading.prescription.certificate.identification for reading in case.readings]
    assert statuses == ["identified", "asserted", "not_identified"]
    assert case.reading("adjusted").prescription.certificate.adjustment.covariates == ("wind",)
    assert case.reading("asserted").prescription.model_fit.identified is False


def test_adjusting_recovers_the_actuator_and_asserting_flips_its_sign(case: PendulumCase) -> None:
    """The asserted channel is not merely noisy: it is the omitted-variable bias of the wind."""
    assert case.reading("adjusted").channel == pytest.approx(demo.GAIN, abs=0.05)
    asserted = case.reading("asserted").channel
    assert asserted < -0.5  # the truth is +3: the log says torque pushes backwards
    assert asserted == pytest.approx(case.omitted_wind_channel, abs=0.2)


def test_a_latent_wind_leaves_no_schedule_to_run(case: PendulumCase) -> None:
    latent = case.reading("latent")
    assert latent.prescription.plan is None
    assert latent.audit is None
    assert latent.prescription.certificate.solver_status is None
    assert "wind" in latent.prescription.certificate.adjustment.reason
    with pytest.raises(NotIdentifiedError):
        _ = latent.prescription.schedule


def test_only_the_adjusted_schedule_is_trusted_and_only_for_its_tube(case: PendulumCase) -> None:
    """The tube, not the barrier, sets the adjusted prefix; the asserted fit has no channel error,
    so it has no tube, and its barrier clearing any number of steps must not read as trust."""
    adjusted = case.reading("adjusted").prescription.certificate
    assert adjusted.certificate_status == "partial"
    assert adjusted.certified_horizon is not None
    assert adjusted.barrier_certified_steps == demo.HORIZON
    assert adjusted.trustworthy_steps == adjusted.certified_horizon > 0

    asserted = case.reading("asserted").prescription.certificate
    assert asserted.certified_horizon is None
    assert asserted.barrier_certified_steps is not None
    assert asserted.trustworthy_steps == 0
    assert case.reading("latent").prescription.certificate.trustworthy_steps == 0


def test_on_the_true_pendulum_the_adjusted_schedule_holds_and_the_asserted_one_goes_the_wrong_way(
    case: PendulumCase,
) -> None:
    adjusted, asserted = case.reading("adjusted").audit, case.reading("asserted").audit
    assert adjusted is not None
    assert asserted is not None
    assert abs(adjusted.final_error) < 0.05  # held within three degrees of the target
    assert adjusted.peak_speed < demo.SPEED_LIMIT

    assert case.idle.final_error == pytest.approx(demo.LIFT)  # never acting leaves it hanging
    assert asserted.final_error > 1.5 * case.idle.final_error  # past hanging, on the far side
    assert asserted.rms_error > 2.0 * adjusted.rms_error


def test_the_log_stays_inside_pendulum_v1s_clips() -> None:
    """Beyond either clip the logged torque stops being the applied one, and the fitted class
    stops containing the plant; the design stays well clear of both, as Track J's does."""
    logs = demo.sample_log(episodes=20, steps=100)
    assert float(np.max(np.abs(logs["torque"] + logs["wind"]))) < demo.MAX_TORQUE / 2
    assert float(np.max(np.abs(logs["omega"]))) < 8.0  # Pendulum-v1's max_speed
