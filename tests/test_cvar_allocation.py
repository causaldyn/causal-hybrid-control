"""chc.allocation.cvar_allocate: a split for the most mean gain over a reference in the worst share
of the readings.

Checked against closed forms, linear readings whose gains are lines in the split; on channels with
carryover under thirty readings against every split of a fine grid, the gains there computed from
the channels run over the history, the plan and the tail as one series; against `allocate` where
every reading is one; and for what holds at any readings and level: the split never does worse in
the worst share than the reference, and its level's mean gain is the gains' own.
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from chc.allocation import allocate, cvar_allocate
from chc.response import Channel, Exponential, GeometricAdstock, Hill, MichaelisMenten, Power, Tanh

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
    the first, ten readings alike: the planes propose about ``[1.27, 0.33]``, which returns 0.705 on
    the curves against the reference's ``1.6^3 / (1.6^3 + 1)`` = 0.804, a gain of -0.099. The
    reference is returned, its ``cvar`` 0, and the envelopes' bound stands above it."""
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
    assert plan.bound > 0.0


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
def test_on_s_curves_no_split_loses_to_the_reference_or_passes_the_bound(seed, count, level, share):
    """On Hill curves that start convex, whatever the readings, the level and the reference: the
    gains are the curves', the split's mean gain in the worst share is at least the reference's 0
    and is its gains' own, and no split on a grid of 2001 points of the budget has more than the
    bound."""
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
    assert plan.bound >= float(_tail_mean(grid, level).max()) - 1e-9 * scale


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
    its own gains' mean there; and the bound is above it by no more than the gap."""
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
