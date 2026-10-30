"""chc.allocation.cvar_allocate: a split for the most mean gain over a reference in the worst share
of the readings.

Checked against closed forms, linear readings whose gains are lines in the split; on channels with
carryover under thirty readings against every split of a fine grid, the gains there computed from
the channels run over the history, the plan and the tail as one series; against `allocate` where
every reading is one; and for what holds at any readings and level: the split never does worse in
the worst share than the reference, and its level's mean gain is the gains' own. On S-shaped curves
the boxes' search is checked against every split of a grid, with and without carryover. Readings
that carry weights are checked against the readings repeated, against closed forms derived in exact
arithmetic, and at the level 1 against the weighted mean's best split on a grid; the readings
weighed alike plan the bits they did before weights were taken.
"""

import hashlib
import importlib
import logging
from fractions import Fraction
from itertools import pairwise

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from chc.allocation import _cut, _cvar, _cvar_weights, _onto, allocate, cvar_allocate
from chc.response import (
    Channel,
    Exponential,
    GeometricAdstock,
    Hill,
    Logistic,
    MichaelisMenten,
    Power,
    Saturation,
    Tanh,
)

# Public as jax.enable_x64 from jax 0.8.0; the floor, 0.4.30, has only jax.experimental.enable_x64,
# which jax 0.11 no longer has.
if hasattr(jax, "enable_x64"):
    enable_x64 = jax.enable_x64
else:
    enable_x64 = importlib.import_module("jax.experimental").enable_x64

ONE = GeometricAdstock(0.0, length=1, normalized=False)  # no carryover
PERIODS = 13
KERNELS = (
    GeometricAdstock(0.3, length=6, normalized=True),
    GeometricAdstock(0.6, length=8, normalized=True),
    GeometricAdstock(0.1, length=4, normalized=True),
)
HISTORY = np.random.default_rng(5).gamma(4.0, 20.0, size=(30, 3))
LOWER = np.array([30.0, 20.0, 10.0])
UPPER = np.array([200.0, 150.0, 120.0])
BUDGET = PERIODS * 180.0
CURRENT = np.array([90.0, 60.0, 30.0])  # a period, spending the budget


def _readings(
    count: int, seed: int, spread: float = 0.4, families=(MichaelisMenten, Exponential, Tanh)
) -> list[tuple[Channel, ...]]:
    """``count`` readings of three concave channels, as a posterior's draws scatter them: each
    scale and coefficient off its centre by a log-normal factor of ``spread``, in a family drawn
    from ``families``."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        scale = np.array([150.0, 90.0, 60.0]) * rng.lognormal(0.0, spread, 3)
        coefficient = np.array([1000.0, 700.0, 450.0]) * rng.lognormal(0.0, spread, 3)
        family = families[int(rng.integers(len(families)))]
        out.append(
            tuple(
                Channel(kernel, family(float(s)), float(c))
                for kernel, s, c in zip(KERNELS, scale, coefficient, strict=True)
            )
        )
    return out


def _returns(channel, column, rates) -> np.ndarray:
    """The channel's return over the plan and its tail at each of ``rates``, each from the history,
    the plan and the tail run as one series."""
    history = jnp.asarray(HISTORY[:, column])
    tail = jnp.zeros(channel.kernel.length - 1)

    def one(rate):
        series = jnp.concatenate([history, jnp.full(PERIODS, rate), tail])
        return jnp.sum(channel(series)[HISTORY.shape[0] :])

    return np.asarray(jax.vmap(one)(jnp.asarray(rates, dtype=float)))


def _worth(channels, spend) -> float:
    return sum(
        float(_returns(channel, column, [rate])[0])
        for column, (channel, rate) in enumerate(zip(channels, spend, strict=True))
    )


def _tail_mean(gains: np.ndarray, level: float, axis: int = 0) -> np.ndarray:
    """The mean of the worst ``level`` share of ``gains`` along ``axis``, the last in part."""
    ordered = np.sort(gains, axis=axis)
    share = level * gains.shape[axis]
    whole = int(np.floor(share))
    taken = np.take(ordered, range(whole), axis=axis).sum(axis=axis)
    if whole < gains.shape[axis]:
        taken = taken + (share - whole) * np.take(ordered, whole, axis=axis)
    return taken / share


def _linear(first: float, second: float, unit: float = 1.0) -> tuple[Channel, Channel]:
    return (
        Channel(ONE, Power(100.0, 1.0), 100.0 * first * unit),
        Channel(ONE, Power(100.0, 1.0), 100.0 * second * unit),
    )


@pytest.mark.parametrize("unit", [1.0, 1e18])
@pytest.mark.parametrize(
    ("level", "first", "cvar"), [(0.5, 50.0, 0.0), (0.8, 100.0, 6.25), (1.0, 100.0, 25.0)]
)
def test_two_linear_readings_move_the_split_only_where_the_worst_share_gains(
    unit, level, first, cvar
):
    """Reading A returns 3 and 1 a unit on the two channels, B 1 and 2; from the even split of 100
    their gains are `2 d` and `-d`, `d` the first channel's move. The worst half is B's alone, so
    the split stays; the worst 0.8 is B's and 0.6 of A's, `0.2 d / 1.6`, and the mean is `d / 2`,
    so both move all the way, to 6.25 and 25. Counted in a unit 10^18 times smaller the split is the
    same."""
    plan = cvar_allocate(
        [_linear(3.0, 1.0, unit), _linear(1.0, 2.0, unit)],
        100.0,
        1,
        level=level,
        against=[50.0, 50.0],
        lower=np.zeros(2),
        upper=np.full(2, 100.0),
    )
    np.testing.assert_allclose(plan.spend, [first, 100.0 - first], rtol=1e-9, atol=1e-9)
    d = first - 50.0
    np.testing.assert_allclose(plan.gain, [2.0 * d * unit, -d * unit], rtol=1e-9, atol=1e-9 * unit)
    assert plan.cvar == pytest.approx(cvar * unit, rel=1e-9, abs=1e-9 * unit)
    assert plan.cvar <= plan.bound <= plan.cvar + 1e-9 * 200.0 * unit


@pytest.mark.parametrize("level", [0.1, 0.3, 1.0])
def test_no_split_on_a_fine_grid_has_a_larger_mean_gain_in_the_worst_share(level):
    """Thirty readings of one family a tenth apart, as a posterior the data have narrowed: at 0.3
    the split moves part of the way the mean's best split goes, and at 0.1 the worst share agrees
    on no move, so the split is the reference, which no split on the grid beats there."""
    readings = _readings(30, seed=11, spread=0.1, families=(MichaelisMenten,))
    plan = cvar_allocate(
        readings,
        BUDGET,
        PERIODS,
        level=level,
        against=CURRENT,
        lower=LOWER,
        upper=UPPER,
        history=HISTORY,
    )
    most = _checked_on_a_grid(plan, readings, level)
    assert (plan.stopped, plan.boxes) == ("closed", 1)
    mean = cvar_allocate(
        readings,
        BUDGET,
        PERIODS,
        level=1.0,
        against=CURRENT,
        lower=LOWER,
        upper=UPPER,
        history=HISTORY,
    )
    moved = np.linalg.norm(plan.spend - CURRENT) / np.linalg.norm(mean.spend - CURRENT)
    if level == 0.1:
        np.testing.assert_array_equal(plan.spend, CURRENT)
        assert plan.cvar == 0.0
        assert most < 0.0
    else:
        assert plan.cvar - most < 0.01 * plan.cvar  # the grid comes close, so the case bites
    if level == 0.3:
        assert 0.1 < moved < 0.9
        # the mean's best split, read in the worst share, gains less than this one
        assert float(_tail_mean(mean.gain, level)) < plan.cvar - 0.01 * plan.cvar


def _checked_on_a_grid(plan, readings, level: float) -> float:
    """Checks ``plan`` on the three channels with carryover against every split of a grid of 241
    rates a channel at the budget: it spends the budget in the box, its gains and their worst
    share's mean are the readings' own, and no split of the grid has a larger mean there. Returns
    the grid's largest mean."""
    rate = BUDGET / PERIODS
    assert plan.spend.sum() == pytest.approx(rate, rel=1e-12)
    assert np.all(plan.spend >= LOWER)
    assert np.all(plan.spend <= UPPER)
    base = np.array([_worth(r, CURRENT) for r in readings])
    gain = np.array([_worth(r, plan.spend) for r in readings]) - base
    np.testing.assert_allclose(plan.gain, gain, rtol=1e-9, atol=1e-9 * base.max())
    assert plan.cvar == pytest.approx(float(_tail_mean(gain, level)), rel=1e-9, abs=1e-9)
    assert plan.cvar <= plan.bound <= plan.cvar + 1e-9 * base.max()
    first = np.linspace(LOWER[0], UPPER[0], 241)
    second = np.linspace(LOWER[1], UPPER[1], 241)
    third = rate - first[:, None] - second[None, :]
    inside = (third >= LOWER[2]) & (third <= UPPER[2])
    gains = np.stack(
        [
            _returns(a, 0, first)[:, None]
            + _returns(b, 1, second)[None, :]
            + _returns(c, 2, np.clip(third, LOWER[2], UPPER[2]).ravel()).reshape(third.shape)
            - base[k]
            for k, (a, b, c) in enumerate(readings)
        ]
    )
    most = float(_tail_mean(gains, level)[inside].max())
    assert plan.cvar >= most - 1e-9 * base.max()
    return most


def test_one_reading_however_often_is_planned_as_allocate_plans_it():
    """Every reading alike, the gain is one function at every level: the reading's worth on the
    split less the reference's, at its best where `allocate` spends."""
    reading = _readings(1, seed=3)[0]
    alone = allocate(reading, BUDGET, PERIODS, lower=LOWER, upper=UPPER, history=HISTORY)
    reference = _worth(reading, CURRENT)
    for level in (0.05, 1.0):
        plan = cvar_allocate(
            [reading] * 7,
            BUDGET,
            PERIODS,
            level=level,
            against=CURRENT,
            lower=LOWER,
            upper=UPPER,
            history=HISTORY,
        )
        assert plan.cvar == pytest.approx(alone.worth - reference, rel=1e-9)
        np.testing.assert_allclose(plan.spend, alone.spend, rtol=1e-4)


def test_an_s_curve_s_gain_is_read_on_the_curve():
    """On a Hill of slope 3 the gain is the curve's, not that of the envelope over the box, which
    stands above it below the tangency: each reading's return on the split less its return on the
    reference, both on its curves."""
    readings = [
        (Channel(ONE, Hill(100.0, 3.0), 1000.0), Channel(ONE, MichaelisMenten(100.0), 300.0)),
        (Channel(ONE, MichaelisMenten(60.0), 700.0), Channel(ONE, MichaelisMenten(100.0), 300.0)),
    ]
    plan = cvar_allocate(
        readings,
        90.0,
        1,
        level=1.0,
        against=[45.0, 45.0],
        lower=np.zeros(2),
        upper=np.full(2, 90.0),
    )

    def returns(hill: float, other: float) -> np.ndarray:
        searched = 300.0 * other / (100.0 + other)
        return np.array(
            [
                1000.0 * hill**3 / (100.0**3 + hill**3) + searched,
                700.0 * hill / (60.0 + hill) + searched,
            ]
        )

    np.testing.assert_allclose(plan.gain, returns(*plan.spend) - returns(45.0, 45.0), rtol=1e-9)
    assert 0.0 <= plan.cvar <= plan.bound


@pytest.mark.parametrize("level", [0.1, 1.0])
def test_a_split_the_envelopes_favour_does_not_lose_to_the_reference_on_the_curves(level):
    """Two Hill curves of slope 3 at scales 1 and 1.01, a budget of 1.6 and the reference all on
    the first, ten readings alike: on the envelopes over the whole box the planes propose about
    ``[1.27, 0.33]``, which returns 0.705 on the curves against the reference's
    ``1.6^3 / (1.6^3 + 1)`` = 0.804, a gain of -0.099. The reference is the best split, a corner of
    the budget's line with no rise between, so the boxes close on it: its ``cvar`` 0, and no split
    in the box above it."""
    reading = (Channel(ONE, Hill(1.0, 3.0), 1.0), Channel(ONE, Hill(1.01, 3.0), 1.0))
    plan = cvar_allocate(
        [reading] * 10,
        1.6,
        1,
        level=level,
        against=[1.6, 0.0],
        lower=np.zeros(2),
        upper=np.full(2, 1.6),
    )
    np.testing.assert_array_equal(plan.spend, [1.6, 0.0])
    np.testing.assert_array_equal(plan.gain, np.zeros(10))
    assert plan.cvar == 0.0
    assert (plan.stopped, plan.boxes > 1) == ("closed", True)
    assert plan.bound <= 1e-9 * 1.6**3 / (1.6**3 + 1.0)


@pytest.mark.parametrize("budget", [1e-160, 7.59e-310])
def test_a_budget_however_small_is_planned(budget):
    """A split was moved onto the budget's line by scipy's root finder, to four ulps of the rate:
    below about ``1e-157`` its interpolating steps underflowed to zero, and a budget of ``1e-160``
    raised ``Failed to converge``; below about ``1e-293`` the tolerance rounded to zero, and a
    budget of ``7.59e-310`` raised ``xtol too small``."""
    reading = (Channel(ONE, Hill(1.0, 3.0), 1.0), Channel(ONE, Hill(1.01, 3.0), 1.0))
    plan = cvar_allocate(
        [reading] * 3,
        budget,
        1,
        level=0.5,
        against=[budget, 0.0],
        lower=np.zeros(2),
        upper=np.array([1.0, 4.0]),
    )
    assert np.all((plan.spend >= 0.0) & (plan.spend <= [1.0, 4.0]))
    assert abs(plan.spend.sum() - budget) <= 4 * np.spacing(budget)
    assert plan.cvar >= 0.0


def _hill_readings(count: int, seed: int) -> list[tuple[Channel, ...]]:
    """``count`` readings of two Hill channels that start convex, without carryover, each draw with
    a scale, slope and coefficient of its own."""
    rng = np.random.default_rng(seed)
    return [
        tuple(
            Channel(ONE, Hill(float(s), float(n)), float(c))
            for s, n, c in zip(
                rng.uniform(0.5, 1.5, 2),
                rng.uniform(1.5, 4.0, 2),
                rng.uniform(0.5, 1.5, 2),
                strict=True,
            )
        )
        for _ in range(count)
    ]


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(0, 2**32 - 1),
    count=st.integers(1, 8),
    level=st.floats(0.05, 1.0),
    share=st.floats(0.0, 1.0),
)
def test_on_s_curves_no_split_on_a_grid_beats_the_split_or_passes_the_bound(
    seed, count, level, share
):
    """On Hill curves that start convex, whatever the readings, the level and the reference: the
    gains are the curves', the split's mean gain in the worst share is at least the reference's 0
    and is its gains' own, and no split on a grid of 2001 points of the budget has more than the
    bound, or, the search closed, than the split."""
    readings = _hill_readings(count, seed)
    budget = 2.0
    reference = np.array([budget * share, budget * (1.0 - share)])
    plan = cvar_allocate(
        readings,
        budget,
        1,
        level=level,
        against=reference,
        lower=np.zeros(2),
        upper=np.full(2, budget),
    )

    def returns(reading: tuple[Channel, ...], first: np.ndarray) -> np.ndarray:
        one, two = reading
        return np.asarray(
            one.coefficient * one.curve(jnp.asarray(first))
            + two.coefficient * two.curve(jnp.asarray(budget - first))
        )

    base = np.array([float(returns(r, np.array([reference[0]]))[0]) for r in readings])
    gain = np.array([float(returns(r, np.array([plan.spend[0]]))[0]) for r in readings]) - base
    scale = float(np.max(np.abs(base))) or 1.0
    np.testing.assert_allclose(plan.gain, gain, rtol=1e-9, atol=1e-12 * scale)
    assert plan.cvar >= 0.0
    assert plan.cvar == pytest.approx(float(_tail_mean(gain, level)), rel=1e-9, abs=1e-12)
    first = np.linspace(0.0, budget, 2001)
    grid = np.stack([returns(r, first) for r in readings]) - base[:, None]
    most = float(_tail_mean(grid, level).max())
    assert plan.bound >= most - 1e-9 * scale
    assert plan.stopped == "closed"
    assert plan.cvar >= most - 1e-9 * scale


def _s_shaped_readings(count: int, seed: int) -> list[tuple[Channel, ...]]:
    """``count`` readings of three Hill channels that start convex, each draw with slopes of its
    own, so each relaxes to envelopes with tangencies of their own."""
    rng = np.random.default_rng(seed)
    return [
        tuple(
            Channel(kernel, Hill(float(s), float(n)), float(c))
            for kernel, s, n, c in zip(
                KERNELS,
                np.array([150.0, 90.0, 60.0]) * rng.lognormal(0.0, 0.3, 3),
                rng.uniform(1.5, 4.0, 3),
                np.array([1000.0, 700.0, 450.0]) * rng.lognormal(0.0, 0.3, 3),
                strict=True,
            )
        )
        for _ in range(count)
    ]


@pytest.mark.parametrize(("level", "moves"), [(0.2, False), (1.0, True)])
def test_on_s_curves_with_carryover_no_split_on_a_grid_beats_the_split(level, moves):
    """Eight readings of three Hill channels that start convex, each with carryover: the boxes
    close, and no split of the grid has a larger mean gain in the worst share. At 1 the split moves
    off the reference; at 0.2 the worst share agrees on no move, and the boxes close on the
    reference."""
    readings = _s_shaped_readings(8, seed=7)
    plan = cvar_allocate(
        readings,
        BUDGET,
        PERIODS,
        level=level,
        against=CURRENT,
        lower=LOWER,
        upper=UPPER,
        history=HISTORY,
    )
    assert plan.stopped == "closed"
    _checked_on_a_grid(plan, readings, level)
    assert (plan.cvar > 0.0) == moves


def test_readings_of_mixed_families_are_read_each_on_its_own_curves():
    """Each channel's readings mix families and slopes, Hill's of slope 1 concave from zero among
    them: each reading's gain is its own curves', and no split of a grid of the budget has a larger
    mean gain in the worst share."""
    readings = [
        (Channel(ONE, Hill(1.0, 3.0), 1.0), Channel(ONE, MichaelisMenten(0.8), 0.9)),
        (Channel(ONE, MichaelisMenten(1.2), 1.1), Channel(ONE, Hill(0.9, 2.5), 1.0)),
        (Channel(ONE, Hill(1.1, 1.0), 0.8), Channel(ONE, Hill(1.0, 4.0), 1.2)),
        (Channel(ONE, Hill(0.7, 2.0), 1.0), Channel(ONE, Tanh(1.0), 0.7)),
    ]
    budget = 2.0
    first = np.linspace(0.0, budget, 2001)
    reference = np.array([0.5, 1.5])

    def returns(reading: tuple[Channel, ...], spend: np.ndarray) -> np.ndarray:
        one, two = reading
        return np.asarray(
            one.coefficient * one.curve(jnp.asarray(spend))
            + two.coefficient * two.curve(jnp.asarray(budget - spend))
        )

    base = np.array([float(returns(r, reference[:1])[0]) for r in readings])
    grid = np.stack([returns(r, first) for r in readings]) - base[:, None]
    for level in (0.25, 0.5, 1.0):
        plan = cvar_allocate(
            readings,
            budget,
            1,
            level=level,
            against=reference,
            lower=np.zeros(2),
            upper=np.full(2, budget),
        )
        gain = np.array([float(returns(r, plan.spend[:1])[0]) for r in readings]) - base
        np.testing.assert_allclose(plan.gain, gain, rtol=1e-9, atol=1e-12 * base.max())
        assert plan.stopped == "closed"
        assert plan.cvar >= float(_tail_mean(grid, level).max()) - 1e-9 * base.max()


@pytest.mark.parametrize(
    ("limit", "s_shaped"), [("max_boxes", True), ("rounds", False)], ids=["boxes", "rounds"]
)
def test_a_search_its_cap_stops_says_so_and_logs_its_gap(monkeypatch, caplog, limit, s_shaped):
    """Stopped by its cap, on S-shaped curves after one box or on concave ones after one round of
    planes, the search says so, with the gap open, and logs one warning that carries it. At the
    level 1 the readings' mean gains from a move, so one round's planes leave the gap open."""
    if s_shaped:
        cap = {"max_boxes": 1}
        reading = (Channel(ONE, Hill(1.0, 3.0), 1.0), Channel(ONE, Hill(1.01, 3.0), 1.0))
        readings, budget, periods, history = [reading] * 10, 1.6, 1, None
        against, lower, upper = [1.6, 0.0], np.zeros(2), np.full(2, 1.6)
    else:
        readings, budget, periods, history = _readings(5, seed=3), BUDGET, PERIODS, HISTORY
        against, lower, upper = CURRENT, LOWER, UPPER
        cap = {}
        monkeypatch.setattr("chc.allocation._ROUNDS", 1)
    with caplog.at_level(logging.WARNING, logger="chc.allocation"):
        plan = cvar_allocate(
            readings,
            budget,
            periods,
            level=1.0,
            against=against,
            lower=lower,
            upper=upper,
            history=history,
            **cap,
        )
    assert (plan.stopped, plan.boxes, plan.limit) == ("cap", 1, limit)
    assert plan.bound > plan.cvar + 1e-6
    assert plan.gap == plan.bound - plan.cvar > plan.tolerance
    assert (plan.unsolved, plan.readings) == ((), len(readings))
    [record] = [r for r in caplog.records if r.name == "chc.allocation"]
    assert record.levelno == logging.WARNING
    assert (record.chc_event, record.planner) == ("allocation_cap", "cvar_allocate")
    assert (record.boxes, record.cvar, record.bound) == (plan.boxes, plan.cvar, plan.bound)
    assert record.limit == limit


def _untied_readings() -> tuple[list[tuple[Channel, ...]], dict]:
    """Ten readings of the untied example, the reference all on the first channel, and the box."""
    reading = (Channel(ONE, Hill(1.0, 3.0), 1.0), Channel(ONE, Hill(1.01, 3.0), 1.0))
    box = {"level": 0.1, "against": [1.6, 0.0], "lower": np.zeros(2), "upper": np.full(2, 1.6)}
    return [reading] * 10, box


@pytest.mark.parametrize(
    ("max_boxes", "stopped", "boxes"),
    [(1, "cap", 1), (2, "cap", 1), (3, "cap", 3), (4, "cap", 3), (7, "closed", 7)],
)
def test_no_box_is_searched_past_the_cap(max_boxes, stopped, boxes):
    """A cut searches two boxes, so the search stops where the next cut would pass the cap, with
    the split of a search capped there. Ten readings of the untied example close in seven."""
    readings, box = _untied_readings()
    plan = cvar_allocate(readings, 1.6, 1, max_boxes=max_boxes, **box)
    assert (plan.stopped, plan.boxes) == (stopped, boxes)
    assert plan.limit == ("max_boxes" if stopped == "cap" else None)
    same = cvar_allocate(readings, 1.6, 1, max_boxes=boxes, **box)
    assert (plan.cvar, plan.bound, plan.boxes) == (same.cvar, same.bound, same.boxes)
    np.testing.assert_array_equal(plan.spend, same.spend)


@pytest.mark.parametrize(
    ("rtol", "atol", "boxes"), [(1e-9, 0.0, 7), (0.05, 0.0, 3), (0.0, 0.04, 3), (0.0, 0.0, 7)]
)
def test_the_search_closes_at_the_larger_of_atol_and_rtol_of_the_reference_s_return(
    rtol, atol, boxes
):
    """Under each reading the reference returns ``1.6^3 / (1 + 1.6^3)``, 0.804; the first box
    leaves a gap of 0.042, three 0.037, five 0.014 and seven none. Each tolerance stops the search
    at the first count whose gap it covers. Asked for none, it closes to 64 epsilons of the
    reference's return, and says the floor raised its tolerance."""
    readings, box = _untied_readings()
    plan = cvar_allocate(readings, 1.6, 1, rtol=rtol, atol=atol, **box)
    scale = 1.6**3 / (1.0 + 1.6**3)
    asked, floor = max(atol, rtol * scale), 64 * np.finfo(float).eps * scale
    assert plan.tolerance == pytest.approx(max(asked, floor), rel=1e-14, abs=0.0)
    assert plan.floored == (floor > asked)
    assert (plan.stopped, plan.boxes, plan.limit) == ("closed", boxes, None)
    assert plan.gap == plan.bound - plan.cvar <= plan.tolerance
    assert plan.relative_gap == pytest.approx(plan.gap / scale, rel=1e-14, abs=0.0)
    assert (plan.unsolved, plan.readings) == ((), 10)


def test_a_float32_search_closes_where_rounding_leaves_its_gap():
    """One reading of two channels with carryover, read in float32: the default share 1e-9 of the
    reference's return is a hundredth of float32's epsilon, and the gap stops falling near two
    epsilons of it, where a search without a floor ran to its cap of 500 boxes. At the floor, 64
    epsilons, it closes, and the split says the floor raised its tolerance. Asked for the share 1,
    a search's tolerance is the reference's return itself."""
    history = np.array([[80.1, 73.2], [11.0, 133.1], [97.8, 50.7], [29.1, 81.8], [20.7, 48.7]])
    rate = 74.1655
    box = {
        "level": 0.3,
        "against": np.array([47.6885, rate - 47.6885]),
        "lower": np.array([29.786, 23.04]),
        "upper": np.array([81.087, 155.605]),
        "history": history,
    }
    with enable_x64(False):
        reading = (
            Channel(GeometricAdstock(0.49, length=5, normalized=True), Hill(149.25, 1.74), 774.1),
            Channel(
                GeometricAdstock(0.05, length=5, normalized=True), Logistic(117.51, 7.9), 530.6
            ),
        )
        plan = cvar_allocate([reading], 10 * rate, 10, **box)
        scale = cvar_allocate([reading], 10 * rate, 10, rtol=1.0, **box).tolerance
    eps = float(np.finfo(np.float32).eps)
    assert (plan.tolerance, plan.floored) == (64 * eps * scale, True)
    assert (plan.stopped, plan.limit) == ("closed", None)
    assert plan.gap <= plan.tolerance
    assert plan.boxes < 50


@pytest.mark.parametrize(
    ("setting", "match"),
    [
        ({"rtol": -0.5}, r"^rtol=-0.5 is not a finite tolerance of at least 0$"),
        ({"atol": float("nan")}, r"^atol=nan is not a finite tolerance"),
        ({"max_boxes": 0}, r"^max_boxes=0 is not a whole number of boxes, at least 1$"),
        ({"max_boxes": 1.0}, r"^max_boxes=1.0 is not a whole number"),
    ],
)
def test_it_refuses_a_tolerance_or_a_cap_no_search_meets(setting, match):
    with pytest.raises(ValueError, match=match):
        cvar_allocate(
            _readings(2, seed=1),
            BUDGET,
            PERIODS,
            level=0.5,
            against=CURRENT,
            lower=LOWER,
            upper=UPPER,
            history=HISTORY,
            **setting,
        )


class _Unturned(Saturation):
    """Hill's curve of slope 3 that reports its inflection at ``1e-300`` scales: from zero spend the
    chord's tangency is past every double the bisection reads."""

    scale: jax.Array = eqx.field(converter=lambda value: jnp.asarray(value, dtype=float))

    def standard(self, z: jax.Array) -> jax.Array:
        return z**3 / (1.0 + z**3)

    def _standard_inflection(self) -> float:
        return 1e-300


@pytest.mark.parametrize(
    "plan",
    [
        lambda reading, box: allocate(reading, 2.0, 1, **box),
        lambda reading, box: cvar_allocate(
            [reading] * 3, 2.0, 1, level=0.5, against=[1.0, 1.0], **box
        ),
    ],
    ids=["allocate", "cvar_allocate"],
)
def test_a_curve_whose_tangency_is_not_found_is_refused_by_both_planners(plan):
    """Where the chord from a box's floor touches the curve nowhere the bisection reads, neither
    planner bounds the box: both refuse the curve by name, as :func:`allocate` always has."""
    reading = (Channel(ONE, _Unturned(1.0), 1.0), Channel(ONE, MichaelisMenten(1.0), 1.0))
    box = {"lower": np.zeros(2), "upper": np.full(2, 2.0)}
    with pytest.raises(RuntimeError, match=r"^_Unturned\(.*no tangency within 2\^64 times its"):
        plan(reading, box)


def _projected(
    split: np.ndarray, lower: np.ndarray, upper: np.ndarray, rate: float
) -> list[Fraction]:
    """The box's split nearest ``split`` whose rates sum to ``rate``, in rationals: between two
    consecutive shifts that take a rate to its floor or its cap, the rates that move are known, and
    the shift that meets the budget spreads what the others leave over them."""
    split_, lower_, upper_ = ([Fraction(x) for x in v] for v in (split, lower, upper))
    budget = Fraction(rate)
    if sum(lower_) >= budget:
        return lower_
    if sum(upper_) <= budget:
        return upper_
    bounds = list(zip(split_, lower_, upper_, strict=True))
    kinks = sorted({a - s for s, a, _ in bounds} | {b - s for s, _, b in bounds})
    for left, right in pairwise(kinks):
        moving = [a - s <= left and right <= b - s for s, a, b in bounds]
        if not any(moving):
            continue
        pairs = list(zip(bounds, moving, strict=True))
        held = sum(a if a - s >= right else b for (s, a, b), m in pairs if not m)
        free = sum(s for (s, _, _), m in pairs if m)
        shift = (budget - held - free) / sum(moving)
        if left <= shift <= right:
            return [min(max(s + shift, a), b) for s, a, b in bounds]
    raise AssertionError("no piece meets the budget")


@settings(max_examples=300, deadline=None)
@given(data=st.data(), size=st.integers(1, 6), exponent=st.integers(-320, 10))
def test_a_split_is_moved_onto_the_budget_where_exact_arithmetic_puts_it(data, size, exponent):
    """At every scale, subnormal budgets among them, the split moved onto the budget is the exact
    projection to ``size + 3`` ulps of the largest number it is given, a rounding a rate summed and
    three in the shift; the most over 20000 draws was 2. scipy's root finder, which found it
    before, raised on budgets from about 1e-157 to 1e-293, its steps there underflowing to 0."""

    def draws(least: float, most: float) -> np.ndarray:
        return np.array(data.draw(st.lists(st.floats(least, most), min_size=size, max_size=size)))

    scale = 10.0**exponent
    lower = draws(0.0, 10.0) * scale
    upper = lower + draws(0.0, 10.0) * scale
    split = lower + draws(-0.5, 1.5) * (upper - lower)
    rate = float(lower.sum() + data.draw(st.floats(0.0, 1.0)) * (upper.sum() - lower.sum()))
    moved = _onto(split, lower, upper, rate)
    exact = _projected(split, lower, upper, rate)
    largest = float(np.max(np.abs(np.concatenate([lower, upper, split, [rate]]))))
    error = max(abs(Fraction(x) - y) for x, y in zip(moved, exact, strict=True))
    assert error <= (size + 3) * Fraction(np.spacing(largest))


@settings(max_examples=300, deadline=None)
@given(data=st.data(), size=st.integers(2, 5))
def test_a_box_is_cut_into_halves_that_each_hold_a_split_of_the_budget(data, size):
    """Whatever the box, its best split on the budget and the envelopes' excess there: the channel
    cut is the one of the largest excess, the halves cover every rate a split of the budget gives
    it, each half holds such a split, and the cut is at the split's rate where that is inside."""

    def draws(least: float, most: float) -> np.ndarray:
        return np.array(data.draw(st.lists(st.floats(least, most), min_size=size, max_size=size)))

    low = draws(0.0, 10.0)
    high = low + draws(0.0, 10.0)
    rate = float(low.sum() + data.draw(st.floats(0.0, 1.0)) * (high.sum() - low.sum()))
    at = _onto(low + draws(0.0, 1.0) * (high - low), low, high, rate)
    excess = np.array(
        data.draw(st.lists(st.just(0.0) | st.floats(0.0, 1.0), min_size=size, max_size=size))
    )
    channel, halves = _cut(low, high, at, excess, rate, np.ones(size))
    floors = np.maximum(low, rate - (high.sum() - high))
    caps = np.minimum(high, rate - (low.sum() - low))
    # where no envelope stands above its curve, the widest interval a split of the budget allows
    assert channel == int(np.argmax(excess if excess.max() > 0.0 else caps - floors))
    others_low, others_high = low.sum() - low[channel], high.sum() - high[channel]
    floor = max(low[channel], rate - others_high)
    cap = min(high[channel], rate - others_low)
    assert halves[0][0] == min(floor, cap)
    assert halves[-1][1] == max(floor, cap)
    slack = 1e-12 * max(1.0, float(high.sum()))
    for (a, b), after in zip(halves, [*halves[1:], None], strict=True):
        assert low[channel] - slack <= a <= b <= high[channel] + slack
        assert others_low + a <= rate + slack  # the half's least spend is within the budget
        assert others_high + b >= rate - slack  # and its most reaches it
        if after is not None:
            assert after[0] == b
    if len(halves) == 2 and floor < at[channel] < cap:
        assert halves[0][1] == at[channel]


@settings(max_examples=50, deadline=None)
@given(
    gains=st.lists(st.floats(-1e3, 1e3), min_size=1, max_size=20),
    level=st.floats(0.01, 1.0),
)
@example(gains=[5e-324], level=0.5)
@example(gains=[49 * 5e-324], level=0.01)
def test_the_worst_shares_weights_read_its_mean(gains, level):
    """The weights the boxes' branching reads are the risk envelope's at the gains: none above
    ``1 / (level * n)``, summing to 1, and their mean of the gains is the worst share's.

    Below the least normal double a product's rounding error is bounded by half a subnormal's
    spacing, not by a share of the product. The weights' mean takes one such error a product;
    ``_cvar`` takes one on its partial product and divides it by the share: ``0.5 * 5e-324``
    rounds to 0, so ``_cvar`` reads ``[5e-324]`` at 0.5 as 0, and ``[49 * 5e-324]`` at 0.01 too."""
    gain = np.array(gains)
    weights = _cvar_weights(gain, level)
    assert np.all(weights >= 0.0)
    assert np.all(weights <= (1.0 + 1e-12) / (level * gain.size))
    assert weights.sum() == pytest.approx(1.0, rel=1e-12, abs=0.0)
    underflow = (gain.size + 1 + 1 / (level * gain.size)) * np.finfo(float).smallest_subnormal
    assert weights @ gain == pytest.approx(
        _cvar(gain, level), rel=1e-9, abs=1e-12 * float(np.abs(gain).max()) + underflow
    )


def test_another_posteriors_s_shaped_draws_compile_nothing_new(caplog):
    """The programs compiled for one posterior's draws serve the next posterior's. Each compiled
    program is held in memory, so a program per draw and channel grows a long run without bound."""

    def plan(readings):
        return cvar_allocate(
            readings,
            BUDGET,
            PERIODS,
            level=0.5,
            against=CURRENT,
            lower=LOWER,
            upper=UPPER,
            history=HISTORY,
        )

    plan(_s_shaped_readings(6, seed=1))
    with jax.log_compiles(True), caplog.at_level(logging.WARNING):
        second = plan(_s_shaped_readings(6, seed=2))
    compiled = [r.getMessage() for r in caplog.records if "XLA compilation" in r.getMessage()]
    assert compiled == []
    assert second.compile_seconds == 0.0
    assert second.search_seconds > 0.0


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(0, 2**32 - 1),
    count=st.integers(1, 12),
    level=st.floats(0.01, 1.0),
    weights=st.lists(st.floats(0.05, 1.0), min_size=3, max_size=3),
)
def test_the_split_never_loses_to_the_reference_in_its_worst_share(seed, count, level, weights):
    """Whatever the readings, the level and the reference in the box at the budget: the reference
    gains 0 under every reading, so the split's mean gain in the worst share is at least 0; it is
    its own gains' mean there; and the bound is above it by no more than the gap, closed in one box
    of concave curves."""
    rate = BUDGET / PERIODS
    share = np.array(weights) / np.sum(weights)
    reference = LOWER + (rate - LOWER.sum()) * share  # each above its floor, under its cap
    readings = _readings(count, seed=seed)
    plan = cvar_allocate(
        readings,
        BUDGET,
        PERIODS,
        level=level,
        against=reference,
        lower=LOWER,
        upper=UPPER,
        history=HISTORY,
    )
    scale = max(abs(_worth(r, reference)) for r in readings)
    assert plan.cvar >= 0.0
    assert plan.cvar == pytest.approx(float(_tail_mean(plan.gain, level)), rel=1e-9, abs=1e-12)
    assert plan.cvar <= plan.bound <= plan.cvar + 1e-9 * scale
    assert (plan.stopped, plan.boxes) == ("closed", 1)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"readings": []}, "no readings"),
        ({"readings": "mixed"}, "each reads them all"),
        ({"level": 0.0}, r"level=0.0 is not a share of the readings in \(0, 1\]"),
        ({"level": 1.5}, r"level=1.5 is not a share"),
        ({"against": CURRENT[:2]}, "against has shape"),
        ({"against": np.array([5.0, 85.0, 90.0])}, "outside the box"),
        ({"against": CURRENT * 1.01}, "the gain is read against a split of the same budget"),
        ({"weights": [1.0]}, r"weights has shape \(1,\); it needs one a reading, 2"),
        ({"weights": [1.0, 1.0, 1.0]}, r"weights has shape \(3,\)"),
        ({"weights": [1.0, -0.5]}, "has a negative weight"),
        ({"weights": [1.0, np.nan]}, "are not all finite"),
        ({"weights": [np.inf, 1.0]}, "are not all finite"),
        ({"weights": [0.0, 0.0]}, "weights are all 0"),
    ],
)
def test_it_refuses_what_it_cannot_plan(change, match):
    readings = _readings(2, seed=1)
    arguments = {"readings": readings, "level": 0.5, "against": CURRENT, "weights": None}
    arguments.update(change)
    if arguments["readings"] == "mixed":
        arguments["readings"] = [readings[0], readings[1][:2]]
    with pytest.raises(ValueError, match=match):
        cvar_allocate(
            arguments["readings"],
            BUDGET,
            PERIODS,
            level=arguments["level"],
            against=arguments["against"],
            lower=LOWER,
            upper=UPPER,
            history=HISTORY,
            weights=arguments["weights"],
        )


@pytest.mark.parametrize(("level", "at"), [(1.0, 2), (1.0, 30), (0.2, 10)])
def test_a_program_its_iteration_limit_stops_leaves_its_box_the_last_bound(stall, level, at):
    """A program HiGHS stalls on ends its box's rounds, and the box keeps the bound of the programs
    before it, which the planes added since could only have tightened. The search goes on, and its
    bound still holds the best split's mean gain in the worst share from above."""
    readings = _s_shaped_readings(8, seed=7)
    box = {"level": level, "against": CURRENT, "lower": LOWER, "upper": UPPER, "history": HISTORY}
    exact = cvar_allocate(readings, BUDGET, PERIODS, **box)
    limits = stall(at)
    plan = cvar_allocate(readings, BUDGET, PERIODS, **box)
    assert len(limits) > at
    assert min(limits) > 0
    assert plan.bound >= exact.cvar - 1e-6
    assert plan.cvar <= exact.bound + 1e-6
    assert (exact.unsolved, plan.unsolved) == ((), (limits[at - 1],))


@pytest.mark.parametrize("status", [1, 4])
def test_a_program_stalled_on_concave_curves_ends_the_search_with_its_gap(stall, caplog, status):
    """On concave curves the one box's planes are the search, so a program HiGHS stalls on ends
    it, with the gap the programs before it left; the plan says so as a search its cap stops,
    names the program as its limit, and records its iterations: all it was allowed at status 1,
    the iteration limit, and half as many at 4, as the stall fixture reports them."""
    limits = stall(2, status)
    with caplog.at_level(logging.WARNING, logger="chc.allocation"):
        plan = cvar_allocate(
            _readings(5, seed=3),
            BUDGET,
            PERIODS,
            level=1.0,
            against=CURRENT,
            lower=LOWER,
            upper=UPPER,
            history=HISTORY,
        )
    assert len(limits) == 2
    assert (plan.stopped, plan.boxes, plan.limit) == ("cap", 1, "unsolved")
    assert plan.bound > plan.cvar + 1e-6
    assert plan.unsolved == ((limits[1] if status == 1 else limits[1] // 2),)
    [record] = [r for r in caplog.records if r.chc_event == "allocation_cap"]
    assert (record.cvar, record.bound, record.limit) == (plan.cvar, plan.bound, "unsolved")


@pytest.mark.parametrize(
    ("builder", "level"),
    [("_readings", 1.0), ("_s_shaped_readings", 1.0), ("_readings", 0.001)],
)
def test_its_programs_keep_their_costs_out_of_highs_scaling(monkeypatch, builder, level):
    """HiGHS folds a program's costs into its scaling when the least nonzero one is under 0.1. An
    excess cost of one over the worst share's readings stretched its factors to 2^15 at 400
    readings: the solution then missed 1e-10 once unscaled, and HiGHS solved the unscaled program
    again with its costs unperturbed, which at the level 1 cycled past eleven million iterations. On
    thirty readings every cost the planes pass is at least 0.1 where it is not zero, at the level
    1 and at 0.001, where the worst share is 0.03 of a reading."""
    import chc.allocation

    real, least = chc.allocation.linprog, []

    def linprog(c, *args, **kwargs):
        least.append(float(np.min(np.abs(c[c != 0.0]))))
        return real(c, *args, **kwargs)

    monkeypatch.setattr(chc.allocation, "linprog", linprog)
    plan = cvar_allocate(
        globals()[builder](30, seed=7),
        BUDGET,
        PERIODS,
        level=level,
        against=CURRENT,
        lower=LOWER,
        upper=UPPER,
        history=HISTORY,
    )
    assert plan.stopped == "closed"
    assert least
    assert min(least) >= 0.1


def _digest(plan) -> str:
    """A digest of what a plan decides and reports, its seconds left out."""
    digest = hashlib.sha256()
    for array in (plan.spend, plan.gain, np.array([plan.cvar, plan.bound])):
        digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
    digest.update(f"{plan.stopped}:{plan.boxes}".encode())
    return digest.hexdigest()


def _pinned_problems() -> dict[str, tuple[list[tuple[Channel, ...]], float, int, dict]]:
    """Three fixed problems: concave channels without carryover, Hill channels that start convex,
    and concave channels with carryover over a history; each split moves off its reference."""
    concave = [tuple(Channel(ONE, c.curve, c.coefficient) for c in r) for r in _readings(6, 21)]
    s_shaped = {"lower": np.zeros(2), "upper": np.full(2, 2.0), "against": np.array([1.2, 0.8])}
    box = {"against": CURRENT, "lower": LOWER, "upper": UPPER}
    return {
        "concave": (concave, 180.0, 1, {"level": 0.7, **box}),
        "s_shaped": (_hill_readings(5, seed=3), 2.0, 1, {"level": 0.4, **s_shaped}),
        "carryover": (
            _readings(8, seed=13, spread=0.1),
            BUDGET,
            PERIODS,
            {"level": 0.4, "history": HISTORY, **box},
        ),
    }


@pytest.mark.parametrize(
    ("problem", "digest"),
    [
        ("concave", "451555d8df150f41550b653d464396759a643185105884aeb2ca2614239db4bd"),
        ("s_shaped", "e04ff945e849c3b249652b12aff92da758b60e456eec806f77640ccfda09419c"),
        ("carryover", "2eab7f497be3c8152f6ef3915ccaa328a81e8479626132699a5bc8ae7030e7c3"),
    ],
)
def test_the_readings_weighed_alike_plan_as_they_did(problem, digest):
    """The split, gains, ``cvar``, ``bound``, how the search stopped and its boxes, on three fixed
    problems, are the bits 1.0.0 returned, the same under both resolutions of the lock's
    dependencies."""
    readings, budget, periods, box = _pinned_problems()[problem]
    assert _digest(cvar_allocate(readings, budget, periods, **box)) == digest


def _weighted_tail(gains: np.ndarray, weights: np.ndarray, level: float) -> float:
    """The weighted mean of the worst ``level`` share of ``gains``, read as Rockafellar and Uryasev
    (2000) write it, ``max_eta eta - sum_i w_i (eta - g_i)_+ / level``, the weights scaled to sum
    to 1: the function is concave and piecewise linear in ``eta``, with its kinks at the gains, so
    its most is at one of them."""
    weights = np.asarray(weights, dtype=float) / np.sum(weights)
    below = np.maximum(gains[:, None] - gains[None, :], 0.0) @ weights
    return float(np.max(gains - below / level))


def _exact_weighted_tail(gains, weights, level) -> Fraction:
    """:func:`_weighted_tail` in exact arithmetic, on ``Fraction`` gains, weights and level."""
    total = sum(weights)
    return max(
        eta - sum(w * max(eta - g, 0) for g, w in zip(gains, weights, strict=True)) / total / level
        for eta in gains
    )


@pytest.mark.parametrize(
    ("weights", "level", "first", "cvar"),
    [
        ((1, 3), 0.5, 50.0, 0.0),
        ((1, 3), 0.8, 0.0, 3.125),
        ((1, 3), 1.0, 0.0, 12.5),
        ((3, 1), 0.5, 100.0, 25.0),
        ((3, 1), 0.8, 100.0, 53.125),
        ((3, 1), 1.0, 100.0, 62.5),
    ],
)
def test_two_weighted_linear_readings_move_the_split_where_their_weighted_share_gains(
    weights, level, first, cvar
):
    """Reading A returns 3 and 1 a unit on the two channels, B 1 and 2; from the even split of 100
    their gains are `2 d` and `-d`, `d` the first channel's move, and the weighted worst share's
    mean is piecewise linear in `d` with its one kink at 0, so its most is at an end or at 0.
    Weighed 1 and 3, B fills the worst half alone wherever `d > 0` and A's quarter takes the lead
    where `d < 0`: the split stays at 0.5, and moves all onto the second channel at 0.8, to 3.125,
    where the readings alike move it onto the first, and at 1, to the mean's 12.5. Weighed 3 and 1
    it moves onto the first at every level, to 25, 53.125 and 62.5. Each closed form is checked in
    exact arithmetic here and in validation/cvar_weighted_closed_forms.mac."""
    weighed = [Fraction(w) for w in weights]
    exact = {
        Fraction(d): _exact_weighted_tail(
            [2 * Fraction(d), -Fraction(d)], weighed, Fraction(str(level))
        )
        for d in (-50, 0, 50)
    }
    best = max(exact, key=exact.__getitem__)
    assert (best + 50, exact[best]) == (Fraction(first), Fraction(cvar))
    assert sorted(exact.values())[-2] < exact[best]  # the best is unique
    plan = cvar_allocate(
        [_linear(3.0, 1.0), _linear(1.0, 2.0)],
        100.0,
        1,
        level=level,
        against=[50.0, 50.0],
        lower=np.zeros(2),
        upper=np.full(2, 100.0),
        weights=weights,
    )
    np.testing.assert_allclose(plan.spend, [first, 100.0 - first], rtol=1e-9, atol=1e-9)
    d = first - 50.0
    np.testing.assert_allclose(plan.gain, [2.0 * d, -d], rtol=1e-9, atol=1e-9)
    assert plan.cvar == pytest.approx(cvar, rel=1e-9, abs=1e-9)
    assert plan.cvar <= plan.bound <= plan.cvar + 1e-9 * 200.0
    assert (plan.stopped, plan.readings) == ("closed", 2)


def _repeated_problem(kind: str) -> tuple[list[tuple[Channel, ...]], float, int, dict]:
    """Three readings and their box: concave channels with carryover, or Hill channels that
    start convex."""
    if kind == "concave":
        box = {"against": CURRENT, "lower": LOWER, "upper": UPPER, "history": HISTORY}
        return _readings(3, seed=4), BUDGET, PERIODS, box
    box = {"against": np.array([1.8, 0.2]), "lower": np.zeros(2), "upper": np.full(2, 2.0)}
    return _hill_readings(3, seed=0), 2.0, 1, box


@pytest.mark.parametrize("level", [0.45, 1.0])
@pytest.mark.parametrize("kind", ["concave", "s_shaped"])
def test_integer_weights_plan_as_the_readings_repeated(kind, level):
    """Weights 2, 1 and 3 are the readings repeated twice, once and three times: the same ``cvar``
    and ``bound`` to the search's tolerance, a split that gains as much in the repeated readings'
    worst share, and the same boxes, each cut on the channel the worst share weighs alike: on the
    Hill readings at 0.45, 23, where a cut that weighed the readings alike would search 25. Near its
    best the share's mean is flat in the split to first order, so two splits within the tolerance
    of the best may part by the tolerance's square root of the budget."""
    readings, budget, periods, box = _repeated_problem(kind)
    repeated = cvar_allocate(
        [readings[0], readings[0], readings[1], *[readings[2]] * 3],
        budget,
        periods,
        level=level,
        **box,
    )
    weighed = cvar_allocate(readings, budget, periods, level=level, weights=[2, 1, 3], **box)
    tolerance = repeated.tolerance
    assert (repeated.stopped, weighed.stopped) == ("closed", "closed")
    assert weighed.tolerance == tolerance
    assert weighed.cvar == pytest.approx(repeated.cvar, rel=0.0, abs=tolerance)
    assert weighed.bound == pytest.approx(repeated.bound, rel=0.0, abs=tolerance)
    as_repeated = float(_tail_mean(np.repeat(weighed.gain, [2, 1, 3]), level))
    assert as_repeated == pytest.approx(weighed.cvar, rel=1e-12, abs=1e-15 * tolerance)
    assert as_repeated >= repeated.cvar - tolerance
    np.testing.assert_allclose(weighed.spend, repeated.spend, rtol=0.0, atol=1e-4 * budget)
    assert weighed.boxes == repeated.boxes
    assert weighed.readings == 3


@pytest.mark.parametrize("kind", ["concave", "s_shaped"])
def test_weights_scaled_alike_plan_alike_and_a_reading_of_no_weight_is_left_out(kind):
    """Weights scaled by any positive factor are the same weights. A reading of weight 0 is no
    reading of the share: the plan is the plan without it, to the bit against the same positive
    weights and to the tolerance against the readings weighed alike, and its gain is read on the
    split. A weight of 1e-200 is as good as none."""
    readings, budget, periods, box = _repeated_problem(kind)
    level = 0.45
    plain = cvar_allocate(readings, budget, periods, level=level, weights=[2, 1, 3], **box)
    for factor in (1e-7 * np.pi, 3.0, 2.0**600):
        scaled = cvar_allocate(
            readings, budget, periods, level=level, weights=[2 * factor, factor, 3 * factor], **box
        )
        assert scaled.cvar == pytest.approx(plain.cvar, rel=0.0, abs=plain.tolerance)
        assert scaled.bound == pytest.approx(plain.bound, rel=0.0, abs=plain.tolerance)
        np.testing.assert_allclose(scaled.spend, plain.spend, rtol=0.0, atol=1e-4 * budget)

    kept = [readings[0], readings[2]]
    without = cvar_allocate(kept, budget, periods, level=level, weights=[2, 3], **box)
    left_out = cvar_allocate(readings, budget, periods, level=level, weights=[2, 0, 3], **box)
    np.testing.assert_array_equal(left_out.spend, without.spend)
    np.testing.assert_array_equal(left_out.gain[[0, 2]], without.gain)
    assert (left_out.cvar, left_out.bound, left_out.boxes) == (
        without.cvar,
        without.bound,
        without.boxes,
    )
    assert left_out.readings == 3
    if kind == "concave":
        own = _worth(readings[1], left_out.spend) - _worth(readings[1], CURRENT)
        scale = _worth(readings[1], CURRENT)
    else:
        one, two = readings[1]

        def returns(split):
            return float(
                one.coefficient * one.curve(jnp.asarray(split[0]))
                + two.coefficient * two.curve(jnp.asarray(split[1]))
            )

        own = returns(left_out.spend) - returns(box["against"])
        scale = returns(box["against"])
    assert left_out.gain[1] == pytest.approx(own, rel=1e-9, abs=1e-12 * scale)

    alike = cvar_allocate(readings, budget, periods, level=level, weights=[1, 0, 1], **box)
    unweighed = cvar_allocate(kept, budget, periods, level=level, **box)
    assert alike.cvar == pytest.approx(unweighed.cvar, rel=0.0, abs=unweighed.tolerance)
    assert alike.bound == pytest.approx(unweighed.bound, rel=0.0, abs=unweighed.tolerance)
    faint = cvar_allocate(readings, budget, periods, level=level, weights=[2, 1e-200, 3], **box)
    assert faint.cvar == pytest.approx(without.cvar, rel=0.0, abs=without.tolerance)
    assert faint.bound == pytest.approx(without.bound, rel=0.0, abs=without.tolerance)


def _two_channel_readings(kind: str) -> list[tuple[Channel, ...]]:
    """Four readings of two channels without carryover, concave or S-shaped."""
    if kind == "s_shaped":
        return _hill_readings(4, seed=17)
    rng = np.random.default_rng(17)
    families = (MichaelisMenten, Exponential, Tanh)
    return [
        tuple(
            Channel(ONE, families[int(rng.integers(3))](float(s)), float(c))
            for s, c in zip(rng.uniform(0.5, 1.5, 2), rng.uniform(0.5, 1.5, 2), strict=True)
        )
        for _ in range(4)
    ]


@pytest.mark.parametrize("kind", ["concave", "s_shaped"])
def test_at_level_one_the_split_has_the_most_weighted_mean_gain(kind):
    """At the level 1 the worst share is every reading, each by its weight: no split on a grid of
    2001 points of the budget has a larger weighted mean gain than the split, nor one above the
    bound, and the split's ``cvar`` is its gains' weighted mean."""
    readings = _two_channel_readings(kind)
    weights = np.array([0.1, 0.5, 0.15, 0.25])
    budget = 2.0
    reference = np.array([0.7, 1.3])
    plan = cvar_allocate(
        readings,
        budget,
        1,
        level=1.0,
        against=reference,
        lower=np.zeros(2),
        upper=np.full(2, budget),
        weights=weights * 8.0,
    )

    def returns(reading: tuple[Channel, ...], first: np.ndarray) -> np.ndarray:
        one, two = reading
        return np.asarray(
            one.coefficient * one.curve(jnp.asarray(first))
            + two.coefficient * two.curve(jnp.asarray(budget - first))
        )

    base = np.array([float(returns(r, reference[:1])[0]) for r in readings])
    first = np.linspace(0.0, budget, 2001)
    grid = weights @ (np.stack([returns(r, first) for r in readings]) - base[:, None])
    gain = np.array([float(returns(r, plan.spend[:1])[0]) for r in readings]) - base
    np.testing.assert_allclose(plan.gain, gain, rtol=1e-9, atol=1e-12 * base.max())
    assert plan.cvar == pytest.approx(float(weights @ gain), rel=1e-9, abs=1e-12 * base.max())
    assert plan.stopped == "closed"
    assert plan.cvar >= float(grid.max()) - plan.tolerance
    assert plan.bound >= float(grid.max())
    assert float(grid.max()) > 0.01 * base.max()  # the best is a move, so the grid bites
    # the readings weighed alike have another best
    alike = float((np.stack([returns(r, first) for r in readings]) - base[:, None]).mean(0).max())
    assert float(np.mean(gain)) < alike - 1e-6 * base.max()


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(0, 2**32 - 1),
    count=st.integers(1, 8),
    level=st.floats(0.01, 1.0),
    weights=st.lists(st.floats(0.0, 1.0), min_size=8, max_size=8),
)
@example(seed=4, count=3, level=0.5, weights=[1.0, 0.25, 9.244250458183258e-206, *[0.0] * 5])
@example(seed=0, count=3, level=1.0, weights=[0.0, 1.0, 9.244250458183258e-206, *[0.0] * 5])
def test_a_weighted_split_never_loses_to_the_reference_in_its_weighted_worst_share(
    seed, count, level, weights
):
    """Whatever the readings, their weights and the level: the reference gains 0 under every
    reading, so the split's weighted mean gain in the worst share is at least 0; it is its own
    gains' weighted worst share, read in numpy as Rockafellar and Uryasev write it; and the bound
    is above it by no more than the tolerance, in one box of concave curves. The examples are
    weights 1e205 apart, which HiGHS read unbounded, or stalled on, where the least weight's
    excess cost 1 and eta's the level over that weight."""
    weighed = np.array(weights[:count])
    if not np.any(weighed > 0.0):
        weighed[0] = 1.0
    readings = _readings(count, seed=seed)
    plan = cvar_allocate(
        readings,
        BUDGET,
        PERIODS,
        level=level,
        against=CURRENT,
        lower=LOWER,
        upper=UPPER,
        history=HISTORY,
        weights=weighed,
    )
    scale = max(abs(_worth(r, CURRENT)) for r in readings)
    assert plan.cvar >= 0.0
    assert plan.cvar == pytest.approx(
        _weighted_tail(plan.gain, weighed, level), rel=1e-9, abs=1e-12 * scale
    )
    assert plan.cvar <= plan.bound <= plan.cvar + plan.tolerance
    assert (plan.stopped, plan.boxes, plan.readings) == ("closed", 1, count)


@settings(max_examples=100, deadline=None)
@given(
    data=st.data(),
    size=st.integers(1, 20),
    level=st.floats(0.01, 1.0),
)
def test_the_weighted_worst_shares_weights_read_its_mean(data, size, level):
    """With the readings' own weights the worst share's: none above a reading's weight over the
    level, summing to 1, their mean of the gains the weighted worst share's, which is Rockafellar
    and Uryasev's. A reading of weight 0 takes none of it."""
    gain = np.array(data.draw(st.lists(st.floats(-1e3, 1e3), min_size=size, max_size=size)))
    raw = np.array(data.draw(st.lists(st.floats(0.0, 1.0), min_size=size, max_size=size)))
    if not np.any(raw > 0.0):
        raw[0] = 1.0
    weights = raw / raw.sum()
    taken = _cvar_weights(gain, level, weights)
    assert np.all(taken >= 0.0)
    assert np.all(taken <= weights / level * (1.0 + 1e-12))
    assert np.all(taken[weights == 0.0] == 0.0)
    assert taken.sum() == pytest.approx(1.0, rel=1e-12, abs=0.0)
    scale = float(np.abs(gain).max()) or 1.0
    assert taken @ gain == pytest.approx(_cvar(gain, level, weights), rel=1e-9, abs=1e-12 * scale)
    assert _cvar(gain, level, weights) == pytest.approx(
        _weighted_tail(gain, weights, level), rel=1e-9, abs=1e-12 * scale
    )
