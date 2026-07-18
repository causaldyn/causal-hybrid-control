"""Multi-seed leaderboard: honest bootstrap CIs on regret; the causal controller wins repeatably."""

from __future__ import annotations

import pytest

from chc.benchmark import InventoryTask, leaderboard_multiseed, run_multiseed

SEEDS = range(5)


@pytest.fixture(scope="module")
def results() -> list:
    return run_multiseed(InventoryTask(), SEEDS)


def test_ci_brackets_the_mean_for_every_controller(results: list) -> None:
    for r in results:
        assert r.regret_lo <= r.regret_mean <= r.regret_hi  # the bootstrap CI contains its own mean
        assert r.n_seeds == 5
        assert r.regret_std >= 0.0


def test_causal_controller_beats_predictive_with_separated_intervals(results: list) -> None:
    by_name = {r.controller: r for r in results}
    causal, predictive = by_name["causal-CHC"], by_name["predictive"]
    assert causal.regret_hi < predictive.regret_lo  # the win survives across seeds, not luck


def test_leaderboard_multiseed_sorts_by_mean_regret_and_shows_cis(results: list) -> None:
    lines = leaderboard_multiseed(results).splitlines()
    assert "95% CI" in lines[0]  # header advertises the interval
    assert lines[-1].startswith("predictive")  # the worst mean regret sorts last
    assert "[" in lines[-1]  # each row prints a CI
    assert "]" in lines[-1]


def test_single_seed_gives_a_degenerate_interval() -> None:
    by_name = {r.controller: r for r in run_multiseed(InventoryTask(), seeds=[0])}
    predictive = by_name["predictive"]
    assert predictive.n_seeds == 1
    assert predictive.regret_lo == predictive.regret_hi == predictive.regret_mean  # one point only
