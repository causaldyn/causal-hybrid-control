"""The switchback's randomisation test and interval: exact by enumeration, exact by drawing, the
reading they are built on, and what they refuse.

The plant is the working model, ``x_{t+1} = a x_t + b u_t + eps_t`` with ``y_t = x_t``, simulated
here independently of the module.
"""

from __future__ import annotations

import contextlib
import itertools
import math

import numpy as np
import pytest

from chc.switchback import (
    CHANNEL,
    STEADY_STATE,
    BlockDesign,
    Horizon,
    MarkovDesign,
    _Randomisation,
    randomisation_interval,
    randomisation_test,
    read_switchback,
)

A, B = 0.8, 1.0


def _first_order(u: np.ndarray, a: float, b: float, rng, burn: int = 200) -> np.ndarray:
    """Readings ``y[:, 0..T]`` of the working model driven by ``u``, the lever off before it."""
    zones, n = u.shape
    lever = np.concatenate([np.zeros((zones, burn)), u], axis=1)
    eps = rng.standard_normal((zones, n + burn))
    x = np.zeros((zones, n + burn + 1))
    for t in range(n + burn):
        x[:, t + 1] = a * x[:, t] + b * lever[:, t] + eps[:, t]
    return x[:, burn:]


def _blocks(rng, zones: int, count: int, length: int) -> np.ndarray:
    return np.repeat(rng.integers(0, 2, (zones, count)), length, axis=1).astype(np.float64)


def _markov(rng, zones: int, periods: int, flip: float) -> np.ndarray:
    path = np.concatenate(
        [rng.random((zones, 1)) < 0.5, rng.random((zones, periods - 1)) < flip], axis=1
    )
    return (np.cumsum(path, axis=1) % 2).astype(np.float64)


def _reading_ratio(u, y, estimand, analysis, design) -> float:
    blocks = design if analysis == "block_dim" else None
    reading = read_switchback(u, y, estimand, analysis, blocks=blocks)
    return reading.estimate / reading.se


# --- exactness ------------------------------------------------------------------------------------


def test_the_enumerated_p_value_is_the_share_of_the_read_schedules_at_least_as_extreme():
    """Every schedule of eight blocks, read by read_switchback one at a time. It refuses the 8 of
    256 with fewer than two blocks after both settings. The test's p-value is the share of the
    other 248 whose statistic is at least as extreme, and over them it is at most alpha with chance
    at most alpha: exact given that the data were read."""
    rng = np.random.default_rng(1)
    design, estimand = BlockDesign(3, 2), Horizon(3)
    y = _first_order(np.zeros((1, 24)), A, 0.0, rng)
    statistics = {}
    for coins in itertools.product((0.0, 1.0), repeat=8):
        u = np.repeat(np.array([coins]), 3, axis=1)
        with contextlib.suppress(ValueError):
            statistics[coins] = _reading_ratio(u, y, estimand, "block_dim", design)
    assert len(statistics) == 248
    size = np.abs(np.array(list(statistics.values())))
    rejected = dict.fromkeys((0.05, 0.1, 0.25), 0)
    for coins, statistic in statistics.items():
        u = np.repeat(np.array([coins]), 3, axis=1)
        test = randomisation_test(u, y, estimand, "block_dim", design)
        assert test.enumerated
        assert test.schedules == 256
        assert test.p_value == pytest.approx(np.mean(size >= abs(statistic) - 1e-9), abs=1e-12)
        for alpha in rejected:
            rejected[alpha] += test.p_value <= alpha
    for alpha, count in rejected.items():
        assert count <= alpha * 248


def test_a_drawn_p_value_is_uniform_when_the_lever_does_nothing():
    """A Markov design's p-value from 99 drawn schedules takes each of k / 100 with chance 1 / 100
    under the null, whatever the outcome it is read against."""
    rng = np.random.default_rng(2)
    y = _first_order(np.zeros((2, 40)), A, 0.0, rng)
    design = MarkovDesign(0.3)
    p = []
    for seed in range(400):
        u = _markov(np.random.default_rng(1000 + seed), 2, 40, design.flip)
        p.append(
            randomisation_test(u, y, CHANNEL, "state_aware", design, draws=99, seed=seed).p_value
        )
    p = np.array(p)
    assert set(np.round(p * 100).astype(int)) <= set(range(1, 101))
    # uniform on {0.01, ..., 1}: mean 0.505, sd 0.289 / sqrt(400)
    assert abs(p.mean() - 0.505) < 3.0 * 0.289 / 20.0
    assert np.mean(p <= 0.05) < 0.05 + 3.0 * math.sqrt(0.05 * 0.95 / 400)


# --- the statistic --------------------------------------------------------------------------------

CASES = [
    ("block_dim", Horizon(3), BlockDesign(3, 2), "blocks"),
    ("block_dim", STEADY_STATE, BlockDesign(4, 1), "blocks"),
    ("state_aware", CHANNEL, MarkovDesign(0.5), "markov"),
    ("state_aware", Horizon(3), MarkovDesign(0.3), "markov"),
    ("state_aware", STEADY_STATE, MarkovDesign(0.3), "markov"),
    ("iv", Horizon(2), MarkovDesign(0.3), "markov"),
    ("local_projection", Horizon(3), MarkovDesign(0.5), "markov"),
]


def _case_data(kind: str, design, seed: int, zones: int = 2):
    rng = np.random.default_rng(seed)
    if kind == "blocks":
        u = _blocks(rng, zones, 12, design.length)
    else:
        u = _markov(rng, zones, 120, design.flip)
    return u, _first_order(u, A, B, rng)


@pytest.mark.parametrize(("analysis", "estimand", "design", "kind"), CASES)
def test_the_statistic_is_the_readings_estimate_over_its_standard_error(
    analysis, estimand, design, kind
):
    u, y = _case_data(kind, design, 3)
    test = randomisation_test(u, y, estimand, analysis, design, draws=19)
    assert test.statistic == pytest.approx(_reading_ratio(u, y, estimand, analysis, design))


@pytest.mark.parametrize(("analysis", "estimand", "design", "kind"), CASES)
def test_a_batch_of_schedules_is_read_as_read_switchback_reads_each(
    analysis, estimand, design, kind
):
    """Drawn schedules, one on for the first half and off for the second, which a local
    projection refuses as not i.i.d., and one never on, which every reading refuses: the batch's
    statistic is NaN exactly where read_switchback refuses."""
    u, y = _case_data(kind, design, 4)
    null = _Randomisation.of(u, estimand, analysis, design, "two-sided", 0.05, 99, 5)
    half = np.repeat([[1.0, 0.0]], u.shape[1] // 2, axis=1).repeat(u.shape[0], axis=0)
    levers = np.concatenate([next(null.batches())[:16], [half, np.zeros_like(u)]])
    batched = null.statistic(levers, y)
    refused = 0
    for lever, statistic in zip(levers, batched, strict=True):
        try:
            expected = _reading_ratio(lever, y, estimand, analysis, design)
        except ValueError:
            refused += 1
            assert np.isnan(statistic)
        else:
            assert statistic == pytest.approx(expected, rel=1e-9, abs=1e-12)
    assert refused == (2 if analysis == "local_projection" else 1)


@pytest.mark.parametrize(("analysis", "estimand", "design", "kind"), CASES)
def test_the_complement_negates_the_statistic(analysis, estimand, design, kind):
    """Swapping on and off negates every reading and keeps its standard error: the two-sided
    p-values of a schedule and its complement are one, which sets the resolution."""
    u, y = _case_data(kind, design, 6)
    one = randomisation_test(u, y, estimand, analysis, design, draws=99)
    other = randomisation_test(1.0 - u, y, estimand, analysis, design, draws=99)
    assert other.statistic == pytest.approx(-one.statistic, rel=1e-9)
    assert other.p_value == pytest.approx(one.p_value)


def test_a_one_sided_p_value_counts_one_tail():
    u, y = _case_data("blocks", BlockDesign(3, 2), 7, zones=1)
    two = randomisation_test(u, y, Horizon(3), "block_dim", BlockDesign(3, 2))
    greater = randomisation_test(
        u, y, Horizon(3), "block_dim", BlockDesign(3, 2), alternative="greater"
    )
    less = randomisation_test(u, y, Horizon(3), "block_dim", BlockDesign(3, 2), alternative="less")
    assert greater.statistic > 0.0
    assert greater.p_value <= two.p_value <= 2.0 * greater.p_value + 1e-12
    assert less.p_value >= 0.5


# --- what it refuses ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("count", "alternative", "refused"),
    [(4, "greater", True), (5, "greater", False), (5, "two-sided", True), (6, "two-sided", False)],
)
def test_a_design_that_cannot_reject_is_refused(count, alternative, refused):
    """``K`` fair coins give 2^(1 - K) two-sided and 2^-K one-sided: at 5% six and five."""
    rng = np.random.default_rng(8)
    coins = np.array([[1.0, 0.0, 1.0, 0.0, 1.0, 1.0][:count]])
    u = np.repeat(coins, 3, axis=1)
    y = _first_order(u, A, B, rng)
    call = lambda: randomisation_test(  # noqa: E731
        u, y, Horizon(3), "block_dim", BlockDesign(3, 2), alternative=alternative
    )
    if refused:
        with pytest.raises(ValueError, match="no outcome could reject"):
            call()
    else:
        assert call().schedules == 2**count


def test_the_zones_coins_add_up():
    """Four blocks are four coins, too few two-sided; in each of two zones they are eight."""
    rng = np.random.default_rng(9)
    coins = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]])
    u = np.repeat(coins, 3, axis=1)
    y = _first_order(u, A, B, rng)
    design = BlockDesign(3, 2)
    with pytest.raises(ValueError, match="no outcome could reject"):
        randomisation_test(u[:1], y[:1], Horizon(3), "block_dim", design)
    assert randomisation_test(u, y, Horizon(3), "block_dim", design).schedules == 256


def test_too_few_draws_are_refused():
    u, y = _case_data("markov", MarkovDesign(0.5), 10)
    with pytest.raises(ValueError, match="no outcome could reject"):
        randomisation_test(u, y, CHANNEL, "state_aware", MarkovDesign(0.5), draws=10)


@pytest.mark.parametrize(
    ("lever", "design", "message"),
    [
        (np.array([[1.0, 0.0, 1.0, 0.0, 0.0, 1.0] * 4]), MarkovDesign(1.0), "cannot have drawn"),
        (np.array([[1.0, 1.0, 0.0, 0.0] * 6]), MarkovDesign(0.0), "cannot have drawn"),
        (
            np.array([[1.0] * 9 + [0.0] * 9 + [1.0] * 5 + [0.0]]),
            BlockDesign(3, 2),
            "inside a block",
        ),
    ],
)
def test_a_lever_the_design_cannot_have_drawn_is_refused(lever, design, message):
    rng = np.random.default_rng(11)
    y = _first_order(lever, A, B, rng)
    analysis = "block_dim" if isinstance(design, BlockDesign) else "state_aware"
    with pytest.raises(ValueError, match=message):
        randomisation_test(lever, y, Horizon(3), analysis, design)


def test_the_block_difference_of_a_markov_design_is_refused():
    u, y = _case_data("markov", MarkovDesign(0.5), 12)
    with pytest.raises(ValueError, match="reads a BlockDesign"):
        randomisation_test(u, y, Horizon(3), "block_dim", MarkovDesign(0.5))


@pytest.mark.parametrize("persistence", [(-0.1, 0.5), (0.5, 1.0), (0.9, 0.8)])
def test_a_persistence_range_outside_the_unit_interval_is_refused(persistence):
    u, y = _case_data("blocks", BlockDesign(3, 2), 13)
    with pytest.raises(ValueError, match="persistence range"):
        randomisation_interval(u, y, Horizon(3), "block_dim", BlockDesign(3, 2), persistence)


# --- the interval ---------------------------------------------------------------------------------


@pytest.mark.parametrize("b", [0.0, 3.0])
def test_the_interval_excludes_zero_exactly_when_the_test_rejects(b):
    """At ``tau = 0`` the joint null imputes the observed readings at every memory, so the
    interval's test there is the test of no effect."""
    rng = np.random.default_rng(14)
    design = BlockDesign(3, 2)
    u = _blocks(rng, 1, 12, 3)
    y = _first_order(u, A, b, rng)
    test = randomisation_test(u, y, Horizon(3), "block_dim", design)
    lo, hi = randomisation_interval(u, y, Horizon(3), "block_dim", design, (0.5, 0.9))
    assert (test.p_value <= 0.05) == (not lo <= 0.0 <= hi)


def test_the_interval_covers_on_the_working_model():
    """Ten blocks and carryover, where the Wald interval is asymptotic: the interval projected
    over a persistence range holding the truth covers at least at its level."""
    rng = np.random.default_rng(15)
    design, estimand = BlockDesign(5, 4), Horizon(5)
    tau = B * (1.0 - A**5) / (1.0 - A)
    covered = runs = 0
    for _ in range(40):
        u = _blocks(rng, 1, 10, 5)
        y = _first_order(u, A, B, rng)
        try:
            lo, hi = randomisation_interval(u, y, estimand, "block_dim", design, (0.7, 0.9))
        except ValueError:
            continue
        runs += 1
        covered += lo <= tau <= hi
    assert runs >= 30
    assert covered / runs >= 0.95 - 3.0 * math.sqrt(0.05 * 0.95 / runs)


def test_a_null_that_names_the_effect_without_the_memory_misses_a_horizon():
    """Under carryover, ``tau_3 = tau_0`` with no memory imputes the readings wrongly, and its
    interval misses ``tau_3``; the joint null at the plant's memory covers it."""
    rng = np.random.default_rng(0)
    u = _markov(rng, 1, 120, 0.3)
    y = _first_order(u, A, B, rng)
    tau = B * (1.0 + A + A * A)
    design = MarkovDesign(0.3)
    lo, hi = randomisation_interval(u, y, Horizon(3), "state_aware", design, (A, A))
    assert lo <= tau <= hi
    lo, hi = randomisation_interval(u, y, Horizon(3), "state_aware", design, (0.0, 0.0))
    assert not lo <= tau <= hi
