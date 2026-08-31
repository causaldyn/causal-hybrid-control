"""Benchmark gate: causal control is near-oracle on pricing; predictive is catastrophic."""

import numpy as np
import pytest

from chc.benchmark import (
    CausalDynamicsTask,
    ConfoundingRobustTask,
    DelayOscillationTask,
    InventoryTask,
    PricingTask,
    SupportShiftTask,
    leaderboard,
)
from chc.delay import STABILISING_RATIO_FLOOR, delay_margin, robust_delay_design
from chc.estimators import DoubleML
from chc.irf import delay_estimate


def test_pricing_benchmark_ranks_causal_above_predictive() -> None:
    results = {r.controller: r for r in PricingTask().run()}

    assert results["oracle"].regret == 0.0  # oracle vs itself
    assert results["causal-CHC"].regret < 1.0  # near-oracle
    assert results["predictive"].regret > 100.0  # catastrophic

    assert results["causal-CHC"].constraint_violations == 0.0  # stays in the safe set
    assert results["predictive"].constraint_violations > 0.0  # diverges out of it
    assert results["predictive"].ood_rate > results["causal-CHC"].ood_rate  # slams the actuator OOD


def test_leaderboard_is_sorted_by_regret() -> None:
    lines = leaderboard(PricingTask().run()).splitlines()
    assert lines[0].startswith("controller")
    assert lines[-1].split()[0] == "predictive"  # worst regret is last (unambiguous)
    assert lines[1].split()[0] in {"oracle", "causal-CHC"}  # a near-oracle controller ranks first


def test_pricing_benchmark_runs_over_a_pluggable_estimator() -> None:
    """The control loop consumes a swappable causal backend, not a hardwired estimator."""
    results = {r.controller: r for r in PricingTask().run(estimator=DoubleML())}
    assert results["causal-CHC"].regret < 1.0  # the DoubleML backend also lands near-oracle
    assert results["predictive"].regret > 100.0  # naive baseline is unchanged


def test_no_confounding_predictive_is_fine() -> None:
    """Honesty: with no confounding (kappa=0), predictive is near-oracle too."""
    results = {r.controller: r for r in PricingTask(kappa=0.0).run()}
    assert results["predictive"].regret < 1.0
    assert results["predictive"].constraint_violations == 0.0


def test_inventory_benchmark_ranks_causal_above_predictive() -> None:
    results = {r.controller: r for r in InventoryTask().run()}
    assert results["oracle"].regret == 0.0
    assert results["causal-CHC"].regret < 0.2  # near oracle
    assert results["predictive"].regret > 0.5  # confounded demand estimate costs more
    assert results["predictive"].constraint_violations > 0.5  # frequent stockouts (under-orders)
    assert results["causal-CHC"].constraint_violations < 0.4  # near the optimal newsvendor fractile


def test_support_shift_pessimism_beats_greedy() -> None:
    results = {r.controller: r for r in SupportShiftTask().run()}
    assert results["oracle"].regret == 0.0
    assert results["pessimistic"].regret < 0.7 * results["greedy"].regret  # pessimism helps
    assert results["greedy"].ood_rate > 0.3  # greedy exploits the model off-support
    assert results["pessimistic"].ood_rate < 0.1  # pessimism stays in-support


def test_confounding_robust_sensitivity_radius_beats_greedy() -> None:
    """Under HIDDEN confounding no estimator can help; the sensitivity radius still can."""
    results = {r.controller: r for r in ConfoundingRobustTask().run()}
    assert results["oracle"].regret == 0.0
    assert results["robust"].regret < 0.7 * results["greedy"].regret  # the radius pays
    assert results["robust"].regret > 0.0  # ...but is NOT the oracle (task is not degenerate)
    # the attenuated gain makes greedy over-command: it blows past the target and leaves the log
    assert results["greedy"].constraint_violations > results["robust"].constraint_violations
    assert results["greedy"].ood_rate > results["robust"].ood_rate


def test_confounding_robust_greedy_degrades_with_hidden_confounding() -> None:
    """The task is genuinely about confounding: greedy regret grows with the latent driver."""
    regrets = [
        {r.controller: r for r in ConfoundingRobustTask(confounding=c).run()}["greedy"].regret
        for c in (0.0, 0.5, 1.0)
    ]
    assert regrets[0] < regrets[1] < regrets[2]
    assert regrets[0] < 1e-2  # no confounding -> the calibrated gain is right -> greedy is oracle


def test_confounding_robust_point_identification_recovers_greedy() -> None:
    """Gamma=1 is point identification: the radius is 0, so the penalty channel is inert."""
    results = {r.controller: r for r in ConfoundingRobustTask(gamma=1.0).run()}
    assert results["robust"].cost == pytest.approx(results["greedy"].cost, rel=1e-3)


def test_confounding_robust_pays_a_premium_when_unconfounded() -> None:
    """Honesty: with no confounding (kappa=0) the pessimism is pure cost."""
    results = {r.controller: r for r in ConfoundingRobustTask(kappa=0.0).run()}
    assert results["greedy"].regret < 1e-2  # unbiased log -> certainty-equivalence is near-oracle
    assert results["robust"].regret > results["greedy"].regret  # robustness is not free


def test_confounding_robust_one_sided_hedge_hurts_when_the_gain_is_inflated() -> None:
    """Honesty: the penalty only SHRINKS actions, so it is the wrong hedge for an inflated gain."""
    results = {r.controller: r for r in ConfoundingRobustTask(kappa=0.5).run()}
    assert results["greedy"].regret > 0.0  # the inflated estimate still costs something
    assert results["robust"].regret > results["greedy"].regret  # ...and shrinking makes it worse


def test_causal_dynamics_identification_beats_prediction_error_fitting() -> None:
    """Three tiers, not two: adjusted, instrumented, and not identified at all.

    The IV row is deliberately *not* held to the adjusted row's bar. Its shifter explains ~18% of
    the action's variance, so it pays a real variance premium — an order of magnitude more channel
    error, and ~10x the regret. Asserting it near-oracle would mean tuning the sample size until a
    weak instrument looked free.
    """
    results = {r.controller: r for r in CausalDynamicsTask().run()}
    assert results["oracle"].regret == 0.0
    assert results["causal-id"].regret < 0.05  # measured 0.014
    assert results["causal-iv"].regret < 0.3  # measured 0.132 — identified, but noisier
    assert results["mse-id"].regret > 1.0  # measured 6.41 against a 6.10 oracle cost
    assert results["causal-iv"].regret < 0.1 * results["mse-id"].regret


def test_causal_dynamics_failure_is_invisible_to_the_safety_columns() -> None:
    """The honest trap: a mis-scaled channel under-actuates, so viol and ood stay clean.

    Worth a test of its own -- it is the case where a reader would otherwise conclude from a green
    constraint column that the plan was fine.
    """
    results = {r.controller: r for r in CausalDynamicsTask().run()}
    assert results["mse-id"].constraint_violations == 0.0
    assert results["mse-id"].ood_rate == 0.0
    assert results["mse-id"].regret > results["causal-id"].regret


def test_a_delay_blind_controller_walks_into_the_hopf() -> None:
    """Same cost, same grid search, only the assumed delay differs -- and one arm diverges."""
    results = {r.controller: r for r in DelayOscillationTask().run(0)}
    blind, aware, oracle = results["delay-blind"], results["delay-aware"], results["oracle"]
    assert blind.regret > 1e3 * max(aware.regret, 1e-3)  # six orders, not a tuning difference
    assert aware.regret < 0.05  # estimating the delay recovers essentially the oracle
    assert oracle.regret == 0.0


def test_the_delay_failure_is_loud_on_the_safety_columns() -> None:
    """The opposite of CausalDynamicsTask's trap, where the same columns stay silent."""
    results = {r.controller: r for r in DelayOscillationTask().run(0)}
    assert results["delay-blind"].constraint_violations > 0.5
    assert results["delay-blind"].ood_rate > 0.5
    for identified in ("delay-aware", "oracle"):
        assert results[identified].constraint_violations == 0.0
        assert results[identified].ood_rate == 0.0


def test_the_blind_arm_fails_on_its_own_terms_not_by_construction() -> None:
    """Its gain is the memoryless optimum sqrt(q/r); it is handed no penalty, only no delay."""
    task = DelayOscillationTask()
    memoryless = task._best_gain(0.0)
    analytic = np.sqrt(task.state_weight / task.control_weight)
    # explicit Euler decays at -ln(1 - dt K)/dt > K, which shifts the optimum a little down
    assert 0.94 * analytic < memoryless < analytic
    assert memoryless * task.channel * task.tau > np.pi / 2  # ...and lands past the boundary


def test_the_measured_oscillation_onset_matches_the_closed_form() -> None:
    """The gate: where the loop actually turns, against ``delay_margin`` at pole 0."""
    task = DelayOscillationTask()
    grid = np.geomspace(0.8, 3.0, 200)
    states, _ = task._sweep(grid, task.tau)
    steps = states.shape[1]
    early = np.max(np.abs(states[:, : steps // 4]), axis=1)
    late = np.max(np.abs(states[:, 3 * steps // 4 :]), axis=1)
    onset = float(grid[late >= early].min())
    boundary_gain = np.pi / (2.0 * task.channel * task.tau)
    # the same number the other way round: at that gain, delay_margin returns exactly this tau
    assert delay_margin(0.0, task.channel * boundary_gain) == pytest.approx(task.tau, rel=1e-12)
    # explicit Euler with an exact integer lag sits ~1/(2m) BELOW the continuous boundary,
    # m = tau/dt = 100, so the measured onset must be just under it -- and it must be close.
    assert 0.985 * boundary_gain < onset < boundary_gain


def test_the_estimated_delay_is_biased_low_which_is_the_destabilising_side() -> None:
    """The refinement's shrinkage has a sign, so more seeds do not average it away."""
    task = DelayOscillationTask()
    designs = []
    for seed in range(6):
        coarse, observed = task._log(seed)
        estimate = delay_estimate(
            {"x": np.diff(observed), "u": coarse[:-1]},
            horizon=12,
            dt=task.dt * task.observe_every,
            adjust_for=(),
            refine=True,
            seed=seed,
        )
        designs.append(robust_delay_design(max(estimate.lo, task.dt), estimate.hi).tau_design)
    ratios = np.array(designs) / task.tau
    assert np.all(ratios < 1.0)  # every seed, not most -- the bias is deterministic in sign
    assert np.ptp(ratios) < 0.05  # ...and small, so the run-to-run spread is not what carries it
    # harmless only because chc.delay's ball tolerates a 76.6% shortfall on this side, against a
    # realised worst of 7.7% -- a factor of 10. The same 7.7% bites on a plant with a tighter ball.
    assert 1.0 - ratios.min() < 0.15 * (1.0 - STABILISING_RATIO_FLOOR)
