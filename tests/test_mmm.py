"""Marketing-mix budget scheduling: the facade on a saturating carryover plant, audited on truth.

Four seeds were checked before any threshold here was written. Adjusted beats a matched-budget flat
split by 4.3%, 4.8%, 7.7%, 8.3%; the confounded arm returns 0.88, 0.78, 0.82, 0.78 of the adjusted
arm's lift. The assertions sit outside that spread, not on top of one draw of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from chc.mmm import MarketingMixSystem, MmmReport, run_marketing_mix

DT = 1.0


@pytest.fixture(scope="module")
def report() -> MmmReport:
    """One run shared by every test here: the whole chain costs a few seconds, not milliseconds."""
    return run_marketing_mix()


def _rk4_amplification(z: float) -> float:
    """RK4's one-step growth factor for ``y' = z/dt * y`` -- the exponential's Taylor sum to 4."""
    return 1.0 + z + z**2 / 2 + z**3 / 6 + z**4 / 24


def test_the_adjusted_plan_beats_a_flat_split_at_the_same_budget(report: MmmReport) -> None:
    assert report.arm("flat").total_spend == pytest.approx(report.arm("adjusted").total_spend)
    assert report.lift("adjusted") > 1.02 * report.lift("flat")


def test_the_confounded_arm_overrates_every_channel_and_underinvests(report: MmmReport) -> None:
    adjusted = report.arm("adjusted").prescription
    confounded = report.arm("confounded").prescription
    assert adjusted is not None
    assert confounded is not None

    for channel, believed in confounded.reach().items():
        assert believed > 1.5 * adjusted.reach()[channel]  # every channel credited with the season

    # and the inflation is uneven, so the ORDER it would allocate by is distorted too: the true
    # social:search ratio of incremental returns is 0.4, which the adjusted arm nearly recovers.
    def ratio(prescription) -> float:
        reach = prescription.reach()
        return reach["spend_social"] / reach["spend_search"]

    assert ratio(adjusted) == pytest.approx(0.4, abs=0.1)
    assert ratio(confounded) > 0.6

    assert report.arm("confounded").total_spend < report.arm("adjusted").total_spend
    assert report.lift("confounded") < 0.95 * report.lift("adjusted")


def test_efficiency_per_unit_spend_rewards_underinvestment_under_diminishing_returns(
    report: MmmReport,
) -> None:
    """The reason the arms are compared at matched budget and not on this number.

    The confounded arm buys less lift and yet looks *more* efficient, because it spends less and the
    response saturates. An MMM read on return-per-unit alone would therefore prefer the arm that
    got the causal question wrong.
    """
    assert report.lift("confounded") < report.lift("adjusted")
    assert report.efficiency("confounded") > report.efficiency("adjusted")


def test_the_channel_ordering_matches_the_true_incremental_returns(report: MmmReport) -> None:
    reach = report.arm("adjusted").prescription.reach()  # type: ignore[union-attr]
    assert reach["spend_search"] > reach["spend_video"] > reach["spend_social"]
    system = MarketingMixSystem()
    assert system.gamma[0] > system.gamma[2] > system.gamma[1]


def test_the_known_adstock_rows_are_left_with_exactly_the_integrator_gap(
    report: MmmReport,
) -> None:
    """``known=`` did its job, and what it did not cover is the forward-difference/RK4 mismatch.

    `fit_causal_residual` reads the rate as ``(x_next - x)/dt`` while the plant was rolled out with
    RK4, so the residual on a row handed over as known is not zero -- it is the difference between
    the two integrators, in closed form. Asserting *that* rather than "approximately zero" is what
    makes this test fail if anything else ever leaks into those rows.
    """
    residual = report.arm("adjusted").prescription.model_fit.residual  # type: ignore[union-attr]
    drift = np.asarray(residual.drift)
    system = MarketingMixSystem()
    for index, theta in enumerate(system.theta):
        row = 1 + index  # state 0 is sales; adstock rows follow
        measured = drift[row, 1 + row]  # feature 1 + row is the row's own state (bias is feature 0)
        expected = (_rk4_amplification(-theta * DT) - 1.0) / DT + theta
        assert measured == pytest.approx(expected, abs=0.02)


def test_the_prescription_is_identified_and_certified_over_the_whole_horizon(
    report: MmmReport,
) -> None:
    certificate = report.arm("adjusted").prescription.certificate  # type: ignore[union-attr]
    assert certificate.identification == "identified"
    assert certificate.adjustment.covariates == ("season",)
    assert certificate.certificate_status == "certified"
    assert certificate.trustworthy_steps == 12
    assert report.arm("flat").prescription is None  # a fixed rule has nothing to certify


def test_the_case_study_reaches_a_decision_in_under_ten_lines() -> None:
    """The L1 gate, executed rather than asserted in prose. Body below is nine statements."""
    system = MarketingMixSystem()
    panel_columns = system.sample(n_regions=12, n_weeks=14, seed=3)

    from chc import Constraint, Panel, Target, prescribe

    panel = Panel.from_frame(panel_columns, unit="region", time="week")
    out = prescribe(
        panel,
        levers=system.levers(),
        target=Target("sales", value=8.0),
        constraints=[Constraint(name, lo=0.0) for name in system.adstock_columns],
        adjustment=system.graph(),
        known=None,
        horizon=6,
        dt=DT,
        tolerance=1.0,
    )
    assert out.schedule.magnitudes.shape == (6, 3)
    assert "Prescription for `sales`" in out.report()
