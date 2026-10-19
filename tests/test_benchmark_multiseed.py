"""Multi-seed leaderboard: honest bootstrap CIs on regret; the causal controller wins repeatably."""

from __future__ import annotations

import math

import numpy as np
import pytest

from chc.research.benchmark import (
    ConfoundingRobustTask,
    InventoryTask,
    MultiSeedResult,
    TaskResult,
    leaderboard_multiseed,
    run_multiseed,
)

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


def test_a_five_digit_regret_keeps_its_interval_apart() -> None:
    """The pricing task's predictive regret over 12 seeds filled the interval's column and ran into
    it: ``13734.15[13732.55, 13735.31]``."""
    row = MultiSeedResult("predictive", 13734.15, 13732.55, 13735.31, 1.2, 1.0, 12)
    line = leaderboard_multiseed([row]).splitlines()[1]
    assert line.split()[1:4] == ["13734.15", "[13732.55,", "13735.31]"]


def test_confounding_robust_win_separates_across_seeds() -> None:
    """The sensitivity radius beats certainty-equivalence repeatably, not on one lucky log."""
    by_name = {r.controller: r for r in run_multiseed(ConfoundingRobustTask(), SEEDS)}
    assert by_name["robust"].regret_hi < by_name["greedy"].regret_lo


def test_a_single_seed_gives_no_interval() -> None:
    """One regret shows no spread, so a 95% interval for the mean is unknown. It read
    ``[mean, mean]``, a width of zero labelled 95%, which held the mean with probability 0."""
    (result,) = run_multiseed(_Regrets({0: 1.5}), [0])
    assert result.n_seeds == 1
    assert result.regret_mean == 1.5
    assert (result.regret_lo, result.regret_hi) == (-math.inf, math.inf)
    row = leaderboard_multiseed([result]).splitlines()[1]
    assert row.split()[1:4] == ["1.50", "[-inf,", "inf]"]


class _Regrets:
    """A task whose one controller's regret on each seed is read from a table, or drawn from a
    normal law on its own stream when no table is given."""

    def __init__(self, table: dict[int, float] | None = None, stream: int = 0) -> None:
        self.table, self.stream = table, stream

    def run(self, seed_data: int = 0) -> list[TaskResult]:
        if self.table is not None:
            regret = self.table[seed_data]
        else:
            regret = float(np.random.default_rng([self.stream, seed_data]).standard_normal())
        return [TaskResult("controller", regret, regret, 0.0, 0.0)]


@pytest.mark.parametrize(
    ("regrets", "lo", "hi"),
    [
        # mean +- t(0.975, n - 1) sd / sqrt(n), the quantiles in closed form, evaluated in Maxima
        # at 30 digits: tan(0.475 pi) at one degree of freedom, 0.95 / sqrt(0.04875) at two
        ((0.0, 1.0), 0.5 - 12.706204736174705 * 0.5, 0.5 + 12.706204736174705 * 0.5),
        ((1.0, 2.0, 3.0), 2.0 - 4.302652729749464 / 3**0.5, 2.0 + 4.302652729749464 / 3**0.5),
    ],
)
def test_the_interval_is_students_over_the_seeds(regrets, lo, hi) -> None:
    """To SciPy's quantile: 1.13, the oldest release supported, reads ``t(0.975, 1)`` 2e-11 off the
    closed form."""
    table = dict(enumerate(regrets))
    (result,) = run_multiseed(_Regrets(table), list(table))
    assert result.regret_lo == pytest.approx(lo, rel=1e-10, abs=0.0)
    assert result.regret_hi == pytest.approx(hi, rel=1e-10, abs=0.0)


@pytest.mark.parametrize("n_seeds", [3, 5, 12])
def test_the_interval_covers_the_mean_regret_at_its_level(n_seeds: int) -> None:
    """On normal regrets the interval holds the mean 95% of the time at any number of seeds. The
    percentile bootstrap it replaced covered 0.75 at three seeds, 0.83 at five and 0.91 at twelve:
    resampling so few values cannot reach past them."""
    replicates = 2000
    covered = 0
    for stream in range(replicates):
        (result,) = run_multiseed(_Regrets(stream=stream), range(n_seeds))
        covered += result.regret_lo <= 0.0 <= result.regret_hi
    # four standard errors of a 0.95 coverage over 2000 replicates
    assert covered / replicates == pytest.approx(0.95, abs=4 * (0.95 * 0.05 / replicates) ** 0.5)
