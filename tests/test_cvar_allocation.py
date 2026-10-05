"""chc.allocation.cvar_allocate: a split for the most mean gain over a reference in the worst share
of the readings.

Checked against closed forms, linear readings whose gains are lines in the split; on channels with
carryover under thirty readings against every split of a fine grid, the gains there computed from
the channels run over the history, the plan and the tail as one series; against `allocate` where
every reading is one; and for what holds at any readings and level: the split never does worse in
the worst share than the reference, and its level's mean gain is the gains' own. On S-shaped curves
the boxes' search is checked against every split of a grid, with and without carryover.
"""

import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from chc.allocation import _cut, _cvar, _cvar_weights, _onto, allocate, cvar_allocate
from chc.response import (
    Channel,
    Exponential,
    GeometricAdstock,
    Hill,
    MichaelisMenten,
    Power,
    Saturation,
    Tanh,
)

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


def test_a_budget_below_the_least_normal_float_is_planned():
    """A split was moved onto the budget's line to four ulps of the rate, which a rate below about
    ``1e-293`` rounds to a tolerance of zero, and scipy's root finder refused it: a budget of
    ``7.59e-310`` raised ``xtol too small``."""
    reading = (Channel(ONE, Hill(1.0, 3.0), 1.0), Channel(ONE, Hill(1.01, 3.0), 1.0))
    budget = 7.59e-310
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
    assert abs(plan.spend.sum() - budget) <= 2.0 * np.finfo(float).tiny
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
    ("cap", "s_shaped"), [("_NODES", True), ("_ROUNDS", False)], ids=["boxes", "rounds"]
)
def test_a_search_its_cap_stops_says_so_and_logs_its_gap(monkeypatch, caplog, cap, s_shaped):
    """Stopped by its cap, on S-shaped curves after one box or on concave ones after one round of
    planes, the search says so, with the gap open, and logs one warning that carries it. At the
    level 1 the readings' mean gains from a move, so one round's planes leave the gap open."""
    monkeypatch.setattr(f"chc.allocation.{cap}", 1)
    if s_shaped:
        reading = (Channel(ONE, Hill(1.0, 3.0), 1.0), Channel(ONE, Hill(1.01, 3.0), 1.0))
        readings, budget, periods, history = [reading] * 10, 1.6, 1, None
        against, lower, upper = [1.6, 0.0], np.zeros(2), np.full(2, 1.6)
    else:
        readings, budget, periods, history = _readings(5, seed=3), BUDGET, PERIODS, HISTORY
        against, lower, upper = CURRENT, LOWER, UPPER
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
        )
    assert (plan.stopped, plan.boxes) == ("cap", 1)
    assert plan.bound > plan.cvar + 1e-6
    [record] = [r for r in caplog.records if r.name == "chc.allocation"]
    assert record.levelno == logging.WARNING
    assert (record.chc_event, record.planner) == ("allocation_cap", "cvar_allocate")
    assert (record.boxes, record.cvar, record.bound) == (plan.boxes, plan.cvar, plan.bound)


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
def test_the_worst_shares_weights_read_its_mean(gains, level):
    """The weights the boxes' branching reads are the risk envelope's at the gains: none above
    ``1 / (level * n)``, summing to 1, and their mean of the gains is the worst share's."""
    gain = np.array(gains)
    weights = _cvar_weights(gain, level)
    assert np.all(weights >= 0.0)
    assert np.all(weights <= (1.0 + 1e-12) / (level * gain.size))
    assert weights.sum() == pytest.approx(1.0, rel=1e-12, abs=0.0)
    assert weights @ gain == pytest.approx(
        _cvar(gain, level), rel=1e-9, abs=1e-12 * float(np.abs(gain).max())
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
        plan(_s_shaped_readings(6, seed=2))
    compiled = [r.getMessage() for r in caplog.records if "XLA compilation" in r.getMessage()]
    assert compiled == []


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
    ],
)
def test_it_refuses_what_it_cannot_plan(change, match):
    readings = _readings(2, seed=1)
    arguments = {"readings": readings, "level": 0.5, "against": CURRENT}
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
        )
