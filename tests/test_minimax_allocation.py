"""chc.allocation.minimax_allocate: a split for several readings of the channels.

Checked against a closed form, two linear readings mirrored, whose worst regret is least at the
even split; and on channels with carryover against every split of a fine grid, the regrets there
computed from the channels run over the history, the plan and the tail as one series.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chc.allocation import allocate, minimax_allocate
from chc.response import (
    Channel,
    Exponential,
    GeometricAdstock,
    Hill,
    MichaelisMenten,
    Power,
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


def _reading(scales, coefficients, family=MichaelisMenten):
    return tuple(
        Channel(kernel, family(scale), coefficient)
        for kernel, scale, coefficient in zip(KERNELS, scales, coefficients, strict=True)
    )


READINGS = (
    _reading((120.0, 80.0, 60.0), (1100.0, 700.0, 500.0)),
    _reading((300.0, 60.0, 90.0), (1500.0, 600.0, 650.0)),
    (
        Channel(KERNELS[0], Tanh(250.0), 900.0),
        Channel(KERNELS[1], Tanh(150.0), 800.0),
        Channel(KERNELS[2], Tanh(90.0), 450.0),
    ),
    _reading((200.0, 200.0, 40.0), (900.0, 1000.0, 300.0), family=Exponential),
)


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


def test_two_mirrored_linear_readings_are_hedged_by_the_even_split():
    """Reading A returns 2 and 1 a unit on the two channels, B 1 and 2: each is best all on its
    better channel, worth 200 of a budget of 100, so a split's regrets are 100 - s1 and s1, and the
    least worst regret is 50, at the even split."""
    readings = [
        (Channel(ONE, Power(100.0, 1.0), 200.0), Channel(ONE, Power(100.0, 1.0), 100.0)),
        (Channel(ONE, Power(100.0, 1.0), 100.0), Channel(ONE, Power(100.0, 1.0), 200.0)),
    ]
    plan = minimax_allocate(
        readings,
        100.0,
        1,
        lower=np.zeros(2),
        upper=np.full(2, 100.0),
        history=np.zeros((0, 2)),
    )
    np.testing.assert_allclose(plan.spend, [50.0, 50.0], rtol=1e-9)
    np.testing.assert_allclose(plan.best, [200.0, 200.0], rtol=1e-12)
    np.testing.assert_allclose(plan.regret, [50.0, 50.0], rtol=1e-9)
    assert plan.worst - plan.bound <= 1e-9 * 200.0


def test_no_split_on_a_fine_grid_has_a_smaller_worst_regret():
    plan = minimax_allocate(READINGS, BUDGET, PERIODS, lower=LOWER, upper=UPPER, history=HISTORY)
    rate = BUDGET / PERIODS
    assert plan.spend.sum() == pytest.approx(rate, rel=1e-12)
    assert np.all(plan.spend >= LOWER)
    assert np.all(plan.spend <= UPPER)
    best = np.array(
        [
            _worth(r, allocate(r, BUDGET, PERIODS, lower=LOWER, upper=UPPER, history=HISTORY).spend)
            for r in READINGS
        ]
    )
    np.testing.assert_allclose(plan.best, best, rtol=1e-12)
    regret = best - np.array([_worth(r, plan.spend) for r in READINGS])
    np.testing.assert_allclose(plan.regret, regret, rtol=1e-9, atol=1e-9 * best.max())
    assert plan.worst == pytest.approx(regret.max(), rel=1e-9)
    assert plan.bound <= plan.worst
    assert plan.worst - plan.bound <= 1e-9 * best.max()
    first = np.linspace(LOWER[0], UPPER[0], 241)
    second = np.linspace(LOWER[1], UPPER[1], 241)
    third = rate - first[:, None] - second[None, :]
    inside = (third >= LOWER[2]) & (third <= UPPER[2])
    worst = np.full(third.shape, -np.inf)
    for k, (a, b, c) in enumerate(READINGS):
        returns = (
            _returns(a, 0, first)[:, None]
            + _returns(b, 1, second)[None, :]
            + _returns(c, 2, np.clip(third, LOWER[2], UPPER[2]).ravel()).reshape(third.shape)
        )
        worst = np.maximum(worst, best[k] - returns)
    least = float(worst[inside].min())
    assert plan.worst <= least * (1 + 1e-9)
    assert least - plan.worst < 0.01 * plan.worst  # the grid comes close, so the case bites
    # the readings disagree: trusting any one of them leaves a worse worst regret than the hedge
    for reading in READINGS:
        alone = allocate(reading, BUDGET, PERIODS, lower=LOWER, upper=UPPER, history=HISTORY)
        assert (
            max(best[k] - _worth(r, alone.spend) for k, r in enumerate(READINGS)) > 1.1 * plan.worst
        )


def test_a_single_reading_is_planned_as_allocate_plans_it():
    reading = READINGS[0]
    plan = minimax_allocate([reading], BUDGET, PERIODS, lower=LOWER, upper=UPPER, history=HISTORY)
    alone = allocate(reading, BUDGET, PERIODS, lower=LOWER, upper=UPPER, history=HISTORY)
    np.testing.assert_allclose(plan.spend, alone.spend, rtol=1e-12)
    assert plan.worst <= 1e-9 * alone.worth


def test_an_s_curve_s_regret_is_its_envelope_s():
    """On a Hill of slope 3 the reading's best is allocate's bound, on the envelope, and the
    regret is measured there too: below the tangency, 100 * 2^(1/3), the envelope is the chord
    from the origin to where the curve is 2/3, which the curve lies under."""
    readings = [
        (Channel(ONE, Hill(100.0, 3.0), 1000.0), Channel(ONE, MichaelisMenten(100.0), 300.0)),
        (Channel(ONE, MichaelisMenten(60.0), 700.0), Channel(ONE, MichaelisMenten(100.0), 300.0)),
    ]
    plan = minimax_allocate(
        readings,
        90.0,
        1,
        lower=np.zeros(2),
        upper=np.full(2, 90.0),
        history=np.zeros((0, 2)),
    )
    alone = allocate(
        readings[0], 90.0, 1, lower=np.zeros(2), upper=np.full(2, 90.0), history=np.zeros((0, 2))
    )
    assert plan.best[0] == pytest.approx(alone.bound, rel=1e-12)
    assert np.all(plan.regret >= -1e-9 * plan.best.max())
    hill, other = plan.spend
    tangency = 100.0 * 2.0 ** (1 / 3)  # where h = 2/3, the chord's end
    assert 0.0 < hill < tangency
    chord = hill * (2.0 / 3.0) / tangency
    on_envelope = 1000.0 * chord + 300.0 * other / (100.0 + other)
    assert plan.regret[0] == pytest.approx(plan.best[0] - on_envelope, rel=1e-9)


@pytest.mark.parametrize(
    ("readings", "match"),
    [
        ([], "no readings"),
        ([READINGS[0], READINGS[1][:2]], "each reads them all"),
    ],
)
def test_it_refuses_readings_it_cannot_hedge_between(readings, match):
    with pytest.raises(ValueError, match=match):
        minimax_allocate(readings, BUDGET, PERIODS, lower=LOWER, upper=UPPER, history=HISTORY)
