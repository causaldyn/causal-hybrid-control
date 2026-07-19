"""Causal-methods leaderboard: every frontier estimator beats its naive baseline on a known DGP."""

from __future__ import annotations

import pytest

from chc.causal_bench import CausalBenchRow, causal_bench_report, run_causal_bench


@pytest.fixture(scope="module")
def rows() -> list[CausalBenchRow]:
    return run_causal_bench(seed=0)


def test_every_method_beats_its_naive_baseline(rows: list[CausalBenchRow]) -> None:
    for r in rows:
        assert r.method_bias < r.baseline_bias  # the modern estimator is less biased


def test_every_method_is_close_to_the_truth(rows: list[CausalBenchRow]) -> None:
    for r in rows:
        assert r.method_bias < 0.15  # each recovers its known effect


def test_report_lists_all_methods_with_a_bias_column(rows: list[CausalBenchRow]) -> None:
    text = causal_bench_report(rows)
    assert "|bias|" in text.splitlines()[0]  # header advertises the bias comparison
    for name in ("Callaway-Sant'Anna", "R-learner", "g-formula", "augmented SCM"):
        assert name in text
