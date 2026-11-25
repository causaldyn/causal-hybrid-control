"""The robust fitted-Q evaluation under memoryless confounding, read against Maxima's exact plant.

``validation/stepwise_confounding.mac`` STEP 5 takes a two-state, two-action plant whose logger is
confounded by a binary ``U`` drawn afresh each step, at MSM level 2, and computes in rationals the
plan's true value and its population bounds. A log whose empirical law in each cell is that
population law, each atom repeated by its mass in twelfths, reads those numbers; a log sampled
from the plant reads them up to its sampling error; a log whose ``U`` persists is refused.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from chc.dynamics_id import PersistenceCheck
from chc.research.stepwise_confounding import (
    PersistentConfounding,
    stepwise_breakdown,
    stepwise_robust_value,
)
from chc.uncertainty import msm_worst_case_mean

PLAN = np.array([[1, 0], [1, 1], [0, 1]])  # step by state
TRUTH = 65 / 16  # validation/stepwise_confounding.mac STEP 5b
MAXIMA_BOUNDS = {  # STEP 5c, (lower, upper) by Gamma
    1.0: (41 / 9, 41 / 9),
    1.2: (23651 / 5400, 30581 / 6480),
    2.0: (31 / 8, 81 / 16),
    3.0: (287 / 81, 847 / 162),
}
MAXIMA_LIMIT = 49 / 18  # STEP 5c, Gamma -> inf
MAXIMA_CROSSING = 1.663311211180917  # STEP 5e, the lower value at the true value


def _logger_share(a: int, u: int) -> Fraction:
    """``P(u | s, a)`` on the plant: ``P(u | s) = 1/2``, ``e(a | s) = 1/2``."""
    one = Fraction(2, 3) if u == 1 else Fraction(1, 3)
    return one if a == 1 else 1 - one


def _population_log() -> dict[str, np.ndarray]:
    """Twelve transitions a cell, each ``(u, s')`` atom of the plant repeated by its mass in
    twelfths; each transition its own unit, in one period, so the premise's check has no pair."""
    rows = []
    for s in (0, 1):
        for a in (0, 1):
            for u in (0, 1):
                up = Fraction(1, 4) + Fraction(a, 4) + Fraction(u, 4)
                for following, p in ((0, 1 - up), (1, up)):
                    copies = 12 * _logger_share(a, u) * p
                    assert copies.denominator == 1
                    rows += [(s, a, s + a / 2 + u, following)] * int(copies)
    state, action, reward, following = (np.array(c) for c in zip(*rows, strict=True))
    n = state.size
    return {
        "state": state,
        "action": action,
        "reward": reward.astype(np.float64),
        "next_state": following,
        "unit": np.arange(n),
        "period": np.zeros(n, dtype=int),
    }


def _sampled_log(units: int, periods: int, persistence: float, seed: int) -> dict[str, np.ndarray]:
    """The plant logged over ``units`` from a fair initial state, ``U`` kept from the period before
    with probability ``persistence`` and drawn afresh otherwise."""
    rng = np.random.default_rng(seed)
    state, u = rng.integers(0, 2, units), rng.integers(0, 2, units)
    columns: list[list[np.ndarray]] = [[] for _ in range(6)]
    for t in range(periods):
        if t > 0:
            u = np.where(rng.random(units) < persistence, u, rng.integers(0, 2, units))
        action = (rng.random(units) < np.where(u == 1, 2 / 3, 1 / 3)).astype(int)
        following = (rng.random(units) < 0.25 + action / 4 + u / 4).astype(int)
        values = (state, action, state + action / 2 + u, following, np.arange(units))
        for column, value in zip(columns, (*values, np.full(units, t)), strict=True):
            column.append(value)
        state = following
    keys = ("state", "action", "reward", "next_state", "unit", "period")
    return {key: np.concatenate(column) for key, column in zip(keys, columns, strict=True)}


@pytest.fixture(scope="module")
def sampled() -> dict[str, np.ndarray]:
    return _sampled_log(2000, 10, 0.0, seed=1)


def _plain_fqe(logs: dict[str, np.ndarray], plan: np.ndarray, start: int) -> float:
    """Fitted-Q evaluation with each cell's mean, from ``start``."""
    following = np.zeros(plan.shape[1])
    for t in range(plan.shape[0] - 1, -1, -1):
        current = np.full(plan.shape[1], np.nan)
        for s in range(plan.shape[1]):
            rows = (logs["state"] == s) & (logs["action"] == plan[t, s])
            if rows.any():
                current[s] = np.mean(logs["reward"][rows] + following[logs["next_state"][rows]])
        following = current
    return float(following[start])


@pytest.mark.parametrize("gamma", sorted(MAXIMA_BOUNDS))
def test_the_population_log_reads_maximas_exact_bounds(gamma: float) -> None:
    value = stepwise_robust_value(_population_log(), PLAN, gamma, initial=0)
    lower, upper = MAXIMA_BOUNDS[gamma]
    assert value.lower == pytest.approx(lower, rel=1e-13, abs=0.0)
    assert value.upper == pytest.approx(upper, rel=1e-13, abs=0.0)
    assert value.nominal == pytest.approx(41 / 9, rel=1e-13, abs=0.0)
    assert value.least_count == 12
    assert math.isnan(value.persistence.p_value)


def test_at_the_logger_s_own_level_the_truth_is_inside_and_below_it_may_not_be() -> None:
    inside = stepwise_robust_value(_population_log(), PLAN, 2.0, initial=0)
    assert inside.lower < TRUTH < inside.upper
    narrow = stepwise_robust_value(_population_log(), PLAN, 1.2, initial=0)
    assert narrow.lower > TRUTH  # a level below the confounding's own excludes the truth
    assert narrow.nominal - TRUTH == pytest.approx(71 / 144, rel=1e-12, abs=0.0)


def test_a_sampled_log_holds_the_truth_at_the_logger_s_level_and_its_nominal_misses_it(
    sampled: dict[str, np.ndarray],
) -> None:
    value = stepwise_robust_value(sampled, PLAN, 2.0, initial=0)
    assert value.persistence.p_value > 0.05
    assert value.lower < TRUTH < value.upper
    lower, upper = MAXIMA_BOUNDS[2.0]
    assert abs(value.lower - lower) < 0.03
    assert abs(value.upper - upper) < 0.03
    # the plain evaluation reads the confounded log as it is, 0.49 above the truth
    assert value.nominal - TRUTH > 0.4
    assert stepwise_robust_value(sampled, PLAN, 1.2, initial=0).lower > TRUTH


def test_at_gamma_one_every_value_is_the_plain_fitted_evaluation_exactly(
    sampled: dict[str, np.ndarray],
) -> None:
    for start in (0, 1):
        value = stepwise_robust_value(sampled, PLAN, 1.0, initial=start)
        plain = _plain_fqe(sampled, PLAN, start)
        assert value.lower == value.nominal == value.upper == plain
    sequence = np.array([1, 0, 1])
    value = stepwise_robust_value(sampled, sequence, 1.0, initial=1)
    assert value.nominal == _plain_fqe(sampled, np.repeat(sequence[:, None], 2, axis=1), 1)


def test_the_bounds_widen_monotonically_in_gamma(sampled: dict[str, np.ndarray]) -> None:
    levels = np.geomspace(1.0, 50.0, 40)
    values = [stepwise_robust_value(sampled, PLAN, g, initial=0) for g in levels]
    lower = np.array([v.lower for v in values])
    upper = np.array([v.upper for v in values])
    assert np.all(np.diff(lower) <= 0.0)
    assert np.all(np.diff(upper) >= 0.0)
    assert lower[-1] < lower[0] - 1.0
    assert upper[-1] > upper[0] + 0.5
    limit = stepwise_breakdown(sampled, PLAN, threshold=TRUTH, initial=0).limit
    assert limit <= lower[-1] < limit + 0.1


def test_one_step_is_msm_worst_case_mean_at_the_cell_s_propensity() -> None:
    rng = np.random.default_rng(7)
    n = 301
    logs = {
        "state": rng.integers(0, 3, n),
        "action": rng.integers(0, 2, n),
        "reward": rng.standard_t(3, n),
        "next_state": rng.integers(0, 3, n),
        "unit": np.arange(n),
        "period": np.zeros(n, dtype=int),
    }
    for s in range(3):
        for a in range(2):
            rows = (logs["state"] == s) & (logs["action"] == a)
            y, e = logs["reward"][rows], rows.sum() / (logs["state"] == s).sum()
            for gamma in (1.37, 4.0):
                odds = stepwise_robust_value(logs, np.array([a]), gamma, initial=s)
                upper = msm_worst_case_mean(y, gamma, propensity=e)
                lower = -msm_worst_case_mean(-y, gamma, propensity=e)
                assert odds.upper == pytest.approx(upper, rel=1e-12, abs=1e-14)
                assert odds.lower == pytest.approx(lower, rel=1e-12, abs=1e-14)
                box = stepwise_robust_value(
                    logs, np.array([a]), gamma, initial=s, weight_box="density"
                )
                assert box.upper == pytest.approx(
                    msm_worst_case_mean(y, gamma), rel=1e-12, abs=1e-14
                )
                assert box.lower <= odds.lower
                assert box.upper >= odds.upper


def test_the_default_initial_law_is_the_units_first_states(
    sampled: dict[str, np.ndarray],
) -> None:
    first = sampled["state"][sampled["period"] == 0]
    share = np.bincount(first, minlength=2) / first.size
    default = stepwise_robust_value(sampled, PLAN, 2.0)
    each = [stepwise_robust_value(sampled, PLAN, 2.0, initial=s) for s in (0, 1)]
    assert default.lower == pytest.approx(share @ [v.lower for v in each], rel=1e-12, abs=0.0)
    law = stepwise_robust_value(sampled, PLAN, 2.0, initial=share)
    assert law.upper == pytest.approx(default.upper, rel=1e-14, abs=0.0)


def test_the_breakdown_is_maximas_crossing_and_the_limit_maximas() -> None:
    breakdown = stepwise_breakdown(_population_log(), PLAN, threshold=TRUTH, initial=0)
    assert breakdown.gamma_star == pytest.approx(MAXIMA_CROSSING, rel=2e-6, abs=0.0)
    assert breakdown.gamma_held < breakdown.gamma_star <= breakdown.gamma_held * (1 + 1e-6)
    assert breakdown.limit == pytest.approx(MAXIMA_LIMIT, rel=1e-13, abs=0.0)
    assert breakdown.nominal == pytest.approx(41 / 9, rel=1e-13, abs=0.0)


def test_the_bisected_breakdown_agrees_with_a_grid(sampled: dict[str, np.ndarray]) -> None:
    baseline = np.array([0, 1, 1])  # an action sequence; the plan is a table
    breakdown = stepwise_breakdown(sampled, PLAN, baseline=baseline, initial=0)
    comparator = stepwise_robust_value(sampled, baseline, 1.0, initial=0).nominal
    assert breakdown.comparator == comparator
    assert breakdown.nominal > comparator > breakdown.limit
    grid = np.geomspace(1.0, 20.0, 300)
    lower = np.array([stepwise_robust_value(sampled, PLAN, g, initial=0).lower for g in grid])
    k = int(np.argmax(lower < comparator))
    assert k > 0
    assert lower[k] < comparator <= lower[k - 1]
    assert grid[k - 1] <= breakdown.gamma_held < breakdown.gamma_star <= grid[k]
    at = stepwise_robust_value(sampled, PLAN, breakdown.gamma_star, initial=0)
    held = stepwise_robust_value(sampled, PLAN, breakdown.gamma_held, initial=0)
    assert at.lower < comparator <= held.lower


def test_the_breakdown_reads_one_below_the_nominal_and_inf_above_the_limit(
    sampled: dict[str, np.ndarray],
) -> None:
    above = stepwise_breakdown(sampled, PLAN, threshold=10.0, initial=0)
    assert above.gamma_star == 1.0
    assert math.isnan(above.gamma_held)
    below = stepwise_breakdown(sampled, PLAN, threshold=0.0, initial=0)
    assert below.gamma_star == math.inf
    assert below.limit >= 0.0


def test_a_persisting_confounder_is_refused_and_a_memoryless_one_is_not(
    sampled: dict[str, np.ndarray],
) -> None:
    persisting = _sampled_log(2000, 10, 0.9, seed=1)
    with pytest.raises(PersistentConfounding, match="persist within units") as refused:
        stepwise_robust_value(persisting, PLAN, 2.0, initial=0)
    assert isinstance(refused.value.check, PersistenceCheck)
    assert refused.value.check.p_value < 1e-10
    assert refused.value.check.correlation > 0.1
    with pytest.raises(PersistentConfounding):
        stepwise_breakdown(persisting, PLAN, threshold=TRUTH, initial=0)
    assert stepwise_robust_value(sampled, PLAN, 2.0, initial=0).persistence.p_value > 0.05


def test_what_the_log_cannot_read_is_refused(sampled: dict[str, np.ndarray]) -> None:
    for gamma in (0.5, math.nan, math.inf):
        with pytest.raises(ValueError, match="Gamma must be >= 1"):
            stepwise_robust_value(sampled, PLAN, gamma)
    one_action = {**sampled, "action": np.where(sampled["state"] == 1, 0, sampled["action"])}
    with pytest.raises(ValueError, match="never took that action"):
        stepwise_robust_value(one_action, PLAN, 2.0, initial=0)
    always = np.array([[1, 0], [1, 0], [1, 0]])
    with pytest.raises(ValueError, match="propensity in"):
        stepwise_robust_value(one_action, always, 2.0, initial=0)
    density = stepwise_robust_value(one_action, always, 2.0, initial=0, weight_box="density")
    assert density.lower < density.nominal < density.upper
    weekly = {**sampled, "period": 7 * sampled["period"]}
    with pytest.raises(ValueError, match="one period apart"):
        stepwise_robust_value(weekly, PLAN, 2.0)
    with pytest.raises(KeyError, match="unit"):
        stepwise_robust_value({k: v for k, v in sampled.items() if k != "unit"}, PLAN, 2.0)
    with pytest.raises(ValueError, match="exactly one comparator"):
        stepwise_breakdown(sampled, PLAN)
    with pytest.raises(ValueError, match="exactly one comparator"):
        stepwise_breakdown(sampled, PLAN, threshold=1.0, baseline=np.zeros(3, dtype=int))
    with pytest.raises(ValueError, match="horizon"):
        stepwise_breakdown(sampled, PLAN, baseline=np.zeros(2, dtype=int))
