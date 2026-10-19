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

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chc.decision import Constraint, Target, _transitions, prescribe
from chc.dynamics_id import fit_causal_residual
from chc.panel import Panel, PanelError
from chc.research.mmm import (
    SALES,
    MarketingMixSystem,
    MmmReport,
    _myopic_spend,
    _start_state,
    adstock_dynamics,
    run_marketing_mix,
)

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
    # is not pinned to 0.4, and never was: where the channels land moves with how far the rk4 fit
    # gets, and with the state the channel is read at -- 0.43 on seed 0 read as Euler, 0.33 after
    # one pass of the old iteration, 0.20 at the fixed point, all at the zero state; 0.12 at the
    # plan's start, where `reach` reads it since 0.13.0 -- and over seeds 0-11 it spans 0.12-0.82.
    # The spread is not the saturating carryover's: with beta = 0, no carryover at all, seeds 0-5
    # still span 0.14-0.83. What is left to move it is the season's push, which the model class
    # leaves out of the sales row. Up to 0.6.0 this line asserted 0.4 +- 0.1 on seed 0, where one
    # pass happened to land; up to 0.12 the margin was 0.2, which four seeds read at the zero
    # state cleared by 0.28-0.67 and which the plan's start leaves at 0.18 on seed 3.
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


@pytest.fixture(scope="module")
def turned() -> tuple[MmmReport, ...]:
    """The two seeds of twelve on which the adjusted fit's channel order turns."""
    return tuple(run_marketing_mix(seed=seed) for seed in (4, 8))


def test_the_channel_ordering_is_read_across_seeds_not_off_one(
    report: MmmReport, turned: tuple[MmmReport, ...]
) -> None:
    """Over seeds 0-11 the adjusted fit orders the three channels as their true incremental
    returns on ten; on seeds 4 and 8 video's reach passes search's, 0.78 against 0.58 and 0.83
    against 0.74. Up to 0.18.0 this asserted the full order on seed 0 alone. What held on all
    twelve, the true best channel above the true worst and the adjusted arm's lift above the
    confounded arm's, is asserted on seed 0 and on the two seeds where the order turns."""
    system = MarketingMixSystem()
    assert system.gamma[0] > system.gamma[2] > system.gamma[1]
    for result in (report, *turned):
        reach = result.arm("adjusted").prescription.reach()  # type: ignore[union-attr]
        assert reach["spend_search"] > reach["spend_social"]
        assert result.lift("adjusted") > result.lift("confounded")


def test_the_rk4_fit_holds_the_season_s_push_and_reads_the_incremental_returns() -> None:
    """A plant the fitted class contains, the Hill response far from saturation, its noise taken
    out: the season is held in the sales row's drift over each week, so RK4 carries no push of its
    into the channel, and the incremental returns come back to their truth. Left out of the drift,
    the season's push read (0.478, 0.142, 0.412) on this seed for (0.5, 0.2, 0.35), with an error
    of 6e-6; over three seeds they scattered by 0.10, 0.03 and 0.11."""
    system = MarketingMixSystem(noise=1e-6, half=(1e4, 1e4, 1e4), beta=(0.9e4, 0.6e4, 0.4e4))
    logs = system.sample(seed=0)
    result = prescribe(
        Panel.from_frame(logs, unit="region", time="week", seed=0),
        levers=system.levers(),
        target=Target(SALES, value=8.0),
        constraints=[Constraint(name, lo=0.0) for name in system.adstock_columns],
        adjustment=system.graph(),
        known=adstock_dynamics(jnp.array(system.theta)),
        horizon=2,
        dt=DT,
    )
    states = np.stack([logs[SALES], *[logs[name] for name in system.adstock_columns]], axis=1)
    channel = np.asarray(jax.vmap(result.model_fit.residual.control_channel)(jnp.asarray(states)))
    np.testing.assert_allclose(channel[:, 0, :].mean(axis=0), system.gamma, rtol=0.0, atol=2e-3)


def test_the_myopic_arm_refuses_a_channel_whose_own_return_the_log_does_not_identify(
    report: MmmReport,
) -> None:
    prescription = report.arm("adjusted").prescription
    assert prescription is not None
    hidden = replace(prescription, _alone=(False,) + (True,) * (len(prescription.levers) - 1))
    with pytest.raises(ValueError, match=r"does not identify for \['spend_"):
        _myopic_spend(hidden, MarketingMixSystem(), 12, 1.0)


@pytest.mark.parametrize("budget", [7.0, 0.05])
def test_the_myopic_arm_refuses_a_budget_its_levers_cannot_spend(
    report: MmmReport, budget: float
) -> None:
    """It spent 6.0 and 0.15 instead, silently, and the arms' matched budget with it."""
    prescription = report.arm("adjusted").prescription
    assert prescription is not None
    with pytest.raises(ValueError, match=r"is outside \[0\.15\d*, 6\.0\], what the channels"):
        _myopic_spend(prescription, MarketingMixSystem(), 12, budget)


def test_the_known_adstock_rows_lose_the_integrator_gap_under_the_planner_s_own_integrator(
    report: MmmReport,
) -> None:
    """``known=`` did its job, and the integrator gap that used to be left behind is mostly closed.

    Both halves are asserted, because "small" on its own would also pass if the row were being
    fitted right by accident: the same log read with ``integrator="euler"`` must still leave the
    *closed-form* difference between the two integrators, which is what every release up to 0.4.0
    silently shipped. The Euler read is the fit's own, on the transitions and with the settings
    ``prescribe`` hands it.

    Not zero, and the residue is the log's noise, not the model class: on the same rows with the
    noise taken out, the rk4 fixed point sits at zero on every known row to 1e-6, and over 40 noise
    draws each row's own-state coefficient scatters by 0.053 / 0.041 / 0.014; over five logs, with
    the season held in the drift or not, by a root mean square of 0.033 / 0.028 / 0.016. The
    thresholds are set on the measurement with room, not on the hope -- and the per-row ones are
    single draws of that noise: on the fastest row 0.024, 0.12 of its gap, where it read 0.019
    with the season's push left in the drift's residual.
    """
    adjusted = report.arm("adjusted").prescription
    assert adjusted is not None
    drift = np.asarray(adjusted.model_fit.residual.drift)
    system = MarketingMixSystem()
    logs = system.sample(n_regions=40, n_weeks=26, dt=DT, seed=0)
    covariates = adjusted.certificate.adjustment.covariates
    data, _ = _transitions(
        Panel.from_frame(logs, unit="region", time="week", seed=0),
        states=(SALES, *system.adstock_columns),
        levers=system.spend_columns,
        adjust_for=covariates,
    )
    euler = fit_causal_residual(
        adstock_dynamics(jnp.array(system.theta)),
        data,
        DT,
        adjust_for=covariates,
        nuisance_degree=2,
        integrator="euler",
    )
    euler_drift = np.asarray(euler.residual.drift)

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
    assert corrected[0] < 0.25 * gaps[0]  # and on the fastest row, where the gap is largest


def test_the_prescription_is_identified_and_its_tube_carries_the_season(
    report: MmmReport,
) -> None:
    """The fitted field is affine in the state, so the tube's rate is global: the slope's norm
    over the levers' box, 1.25 a week. The field contracts, its slope's eigenvalues -0.25 to -0.73
    at the box's centre, which a norm cannot see; the twelve steps this certified before came from
    a negative log-norm in Euler's recursion, no bound. The season, adjusted for and not forecast,
    moves sales by about 1.24 a week that the fitted field leaves unexplained, and the tube holds
    that noise (ADR 0080): it passes the tolerance of 1.0 in the first week, so no week is
    certified. On the channel's error alone the tube read 0.11, 0.49, 1.82 and certified two
    weeks, the error of a week with no season, which no week is."""
    certificate = report.arm("adjusted").prescription.certificate  # type: ignore[union-attr]
    assert certificate.identification == "identified"
    assert certificate.adjustment.covariates == ("season",)
    assert certificate.certificate_status == "uncertified"
    assert certificate.tube_rate == "global"
    assert certificate.tube_noise is not None
    assert certificate.tube_noise > 1.0
    assert certificate.trustworthy_steps == 0
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
