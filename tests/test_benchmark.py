"""Benchmark gate: causal control is near-oracle on pricing; predictive is catastrophic."""

from chc.benchmark import InventoryTask, PricingTask, SupportShiftTask, leaderboard


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
