"""Marketing-mix budget scheduling: the facade on a saturating carryover plant, audited on truth.

Four seeds were checked before any threshold here was written. Adjusted beats a matched-budget flat
split by 4.5%, 3.2%, 7.7%, 7.9%; the confounded arm returns 0.87, 0.78, 0.81, 0.78 of the adjusted
arm's lift. The assertions sit outside that spread, not on top of one draw of it.

Those first four moved when `prescribe` switched to `integrator="rk4"` (from 4.3 / 4.8 / 7.7 / 8.3),
which is the right size of move: the fitted field changed materially and the *conclusion* did not,
because every arm here is audited on the true plant rather than on the planner's own forecast. They
moved again in 0.7.0 (from 4.4 / 4.3 / 7.4 / 8.0), when the rk4 fit stopped halting short of its
fixed point, after one pass on the sales row, and was solved to it. In 0.13.0 the cross-fitting
folds were redrawn from a stream of their own, and the four read 4.2%, 5.5%, 6.4%, 8.1%, the
confounded arm 0.87, 0.76, 0.81, 0.70.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chc.mmm import SALES, MarketingMixSystem, MmmReport, _start_state, run_marketing_mix
from chc.panel import Panel, PanelError

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


def test_the_myopic_arm_spends_the_same_budget_on_this_week_alone(report: MmmReport) -> None:
    """The carryover-blind baseline: matched budget, constant in time, ranked by the fitted
    immediate return. Its point is to separate what identification buys from what the objective's
    HORIZON buys -- it reads the adjusted arm's own fit, so the two differ only in the second.

    Pinned here as construction, not as an outcome: which of the two earns more is a property of
    the plant (``causaldyn_bench.allocation``, Track M, measures both regimes over seeds), and
    asserting an order at one seed would be asserting a sign the measurement says is not there.
    """
    myopic, adjusted = report.arm("myopic"), report.arm("adjusted")
    assert myopic.total_spend == pytest.approx(adjusted.total_spend)
    assert myopic.prescription is None  # a fixed rule, not a plan
    assert bool(jnp.all(myopic.spend == myopic.spend[0]))  # constant in time: the whole myopia

    fit = adjusted.prescription
    assert fit is not None
    reach = fit.reach()
    best = max(reach, key=lambda lever: reach[lever])
    worst = min(reach, key=lambda lever: reach[lever])
    columns, system = list(reach), MarketingMixSystem()
    # the best immediate channel is filled to its ceiling and the worst is left at the floor
    assert float(myopic.spend[0, columns.index(best)]) == pytest.approx(system.spend_ceiling)
    assert float(myopic.spend[0, columns.index(worst)]) == pytest.approx(system.spend_floor)
    assert report.lift("myopic") > report.lift("flat")  # concentration beats an equal split here


def test_the_confounded_arm_overrates_every_channel_and_underinvests(report: MmmReport) -> None:
    adjusted = report.arm("adjusted").prescription
    confounded = report.arm("confounded").prescription
    assert adjusted is not None
    assert confounded is not None

    for channel, believed in confounded.reach().items():
        assert believed > 1.5 * adjusted.reach()[channel]  # every channel credited with the season

    # and the inflation is uneven, so the ORDER it would allocate by is distorted too: the true
    # social:search ratio of incremental returns is 0.4, and the confounded arm reads 0.67-0.81
    # over four seeds, 0.18-0.64 above the adjusted arm on each. The adjusted arm's own ratio
    # is not pinned to 0.4, and never was: its sales row carries a saturating carryover and a
    # seasonal push the model class leaves out, so where the channels land moves with how far the
    # rk4 fit gets, and with the state the channel is read at -- 0.43 on seed 0 read as Euler,
    # 0.33 after one pass of the old iteration, 0.20 at the fixed point, all at the zero state;
    # 0.12 at the plan's start, where `reach` reads it since 0.13.0 -- and across seeds it spans
    # 0.12-0.50. Up to 0.6.0 this line asserted 0.4 +- 0.1 on seed 0, where one pass happened to
    # land; up to 0.12 the margin was 0.2, which four seeds read at the zero state cleared by
    # 0.28-0.67 and which the plan's start leaves at 0.18 on seed 3.
    def ratio(prescription) -> float:
        reach = prescription.reach()
        return reach["spend_social"] / reach["spend_search"]

    assert ratio(confounded) > 0.6
    assert ratio(confounded) > ratio(adjusted) + 0.1

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


def test_the_known_adstock_rows_lose_the_integrator_gap_under_the_planner_s_own_integrator(
    report: MmmReport,
) -> None:
    """``known=`` did its job, and the integrator gap that used to be left behind is mostly closed.

    Both halves are asserted, because "small" on its own would also pass if the row were being
    fitted right by accident: the same log read with ``integrator="euler"`` must still leave the
    *closed-form* difference between the two integrators, which is what every release up to 0.4.0
    silently shipped.

    Not zero, and the residue is the log's noise, not the model class: on the same rows with the
    noise taken out, the rk4 fixed point sits at zero on every known row to 1e-6, and over 40 noise
    draws each row's own-state coefficient scatters by 0.053 / 0.041 / 0.014. The thresholds are
    set on the measurement (0.004 / 0.22 / 0.08 of each row's gap, aggregating to 0.06) with room,
    not on the hope -- and the per-row ones are single draws of that noise.
    """
    drift = np.asarray(report.arm("adjusted").prescription.model_fit.residual.drift)  # type: ignore[union-attr]
    system = MarketingMixSystem()
    euler = run_marketing_mix(integrator="euler")
    euler_drift = np.asarray(euler.arm("adjusted").prescription.model_fit.residual.drift)  # type: ignore[union-attr]

    gaps, corrected, uncorrected = [], [], []
    for index, theta in enumerate(system.theta):
        row = 1 + index  # state 0 is sales; adstock rows follow
        column = 1 + row  # feature 1 + row is the row's own state (bias is feature 0)
        gaps.append((_rk4_amplification(-theta * DT) - 1.0) / DT + theta)
        corrected.append(abs(drift[row, column]))
        uncorrected.append(abs(euler_drift[row, column]))

    for gap, left in zip(gaps, uncorrected, strict=True):
        assert left == pytest.approx(gap, abs=0.02)  # euler leaves exactly the amplification
    for gap, left in zip(gaps, corrected, strict=True):
        assert left < 0.5 * gap  # rk4 leaves less than half of it on every row
    assert sum(corrected) < 0.25 * sum(uncorrected)  # measured 0.06
    assert corrected[0] < 0.1 * gaps[0]  # and on the fastest row, where the gap is largest, 0.004


def test_the_prescription_is_identified_and_its_tube_certifies_a_prefix(
    report: MmmReport,
) -> None:
    """The fitted field is affine in the state, so the tube's rate is global: the slope's norm
    over the levers' box, 1.25 a week. At `dt = 1` RK4 carries the tube by about 3.5 a step, and
    it leaves the tolerance 1.0 at the third step (0.11, 0.49, 1.82). The field contracts, its
    slope's eigenvalues -0.25 to -0.73 at the box's centre, which a norm cannot see; the twelve
    steps this certified before came from a negative log-norm in Euler's recursion, no bound."""
    certificate = report.arm("adjusted").prescription.certificate  # type: ignore[union-attr]
    assert certificate.identification == "identified"
    assert certificate.adjustment.covariates == ("season",)
    assert certificate.certificate_status == "partial"
    assert certificate.tube_rate == "global"
    assert certificate.trustworthy_steps == 2
    assert report.arm("flat").prescription is None  # a fixed rule has nothing to certify


def test_the_confounded_arm_is_trusted_for_no_step(report: MmmReport) -> None:
    """Its fit carries no channel error, so no tube bounds its trajectory; the barrier alone
    clearing all twelve steps must not read as twelve trustworthy ones."""
    certificate = report.arm("confounded").prescription.certificate  # type: ignore[union-attr]
    assert certificate.certified_horizon is None
    assert certificate.barrier_certified_steps == 12
    assert certificate.trustworthy_steps == 0


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


def test_the_start_state_reads_a_state_as_numbers_only_where_it_holds_them() -> None:
    """The pooled start read each state with a float64 cast, which read text as the numbers it
    spells."""
    logs = MarketingMixSystem().sample(n_regions=2, n_weeks=3, seed=0)
    logs[SALES] = logs[SALES].astype(str)
    panel = Panel.from_frame(logs, unit="region", time="week")
    with pytest.raises(
        PanelError,
        match=r"column 'sales' is np\.str_\('.+'\) for unit np\.int64\(0\) at time np\.int64\(0\): "
        r"dtype <U\d+, which holds text",
    ):
        _start_state(panel, (SALES,))
