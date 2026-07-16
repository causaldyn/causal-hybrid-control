"""Benchmark gate: causal control is near-oracle on pricing; predictive is catastrophic."""

from chc.benchmark import PricingTask, leaderboard


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
