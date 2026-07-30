"""Benchmark gate: causal control is near-oracle on pricing; predictive is catastrophic."""

import pytest

from chc.benchmark import (
    CausalDynamicsTask,
    ConfoundingRobustTask,
    InventoryTask,
    PricingTask,
    SupportShiftTask,
    leaderboard,
)
from chc.estimators import DoubleML


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
