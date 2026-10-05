"""chc.allocation.minimax_allocate: a split for several readings of the channels.

Checked against a closed form, two linear readings mirrored, whose worst regret is least at the
even split; and on channels with carryover against every split of a fine grid, the regrets there
computed from the channels run over the history, the plan and the tail as one series.
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chc.allocation import _on_envelopes, allocate, minimax_allocate
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


@pytest.mark.parametrize("unit", [1.0, 1e18])
def test_two_mirrored_linear_readings_are_hedged_by_the_even_split(unit):
    """Reading A returns 2 and 1 a unit on the two channels, B 1 and 2: each is best all on its
    better channel, worth 200 of a budget of 100, so a split's regrets are 100 - s1 and s1, and the
    least worst regret is 50, at the even split. Counted in a unit 10^18 times smaller the split is
    the same: a fit run to the edge of its family reads returns that large."""
    readings = [
        (
            Channel(ONE, Power(100.0, 1.0), 200.0 * unit),
            Channel(ONE, Power(100.0, 1.0), 100.0 * unit),
        ),
        (
            Channel(ONE, Power(100.0, 1.0), 100.0 * unit),
            Channel(ONE, Power(100.0, 1.0), 200.0 * unit),
        ),
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
    np.testing.assert_allclose(plan.best, [200.0 * unit, 200.0 * unit], rtol=1e-12)
    np.testing.assert_allclose(plan.regret, [50.0 * unit, 50.0 * unit], rtol=1e-9)
    assert plan.worst - plan.bound <= 1e-9 * 200.0 * unit


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
    """On a Hill of slope 3 the reading's best is the bound of the plan on its envelope from zero
    spend, not allocate's search, and the regret is measured there too: below the tangency,
    100 * 2^(1/3), the envelope is the chord from the origin to where the curve is 2/3, which the
    curve lies under."""
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
    alone = _on_envelopes(readings[0], 90.0, 1, np.zeros(2), np.full(2, 90.0), np.zeros((0, 2)))
    assert plan.best[0] == pytest.approx(alone.bound, rel=1e-12)
    assert np.all(plan.regret >= -1e-9 * plan.best.max())
    hill, other = plan.spend
    tangency = 100.0 * 2.0 ** (1 / 3)  # where h = 2/3, the chord's end
    assert 0.0 < hill < tangency
    chord = hill * (2.0 / 3.0) / tangency
    on_envelope = 1000.0 * chord + 300.0 * other / (100.0 + other)
    assert plan.regret[0] == pytest.approx(plan.best[0] - on_envelope, rel=1e-9)


@pytest.mark.parametrize("weight", [300.0, 30000.0], ids=["excess", "own"])
def test_an_s_curve_s_regret_is_off_the_curves_by_at_most_the_gap_it_logs(caplog, weight):
    """A reading's best return on the curves, :func:`allocate`'s search, lies between its plan on
    the envelopes from zero spend read on the curves and that plan's bound; the split's return on
    the curves is below the envelopes' by their excess there. The warning's gap is the larger of the
    two, and bounds how far each regret on the curves is from the envelopes'. At a weight of 300 on
    the second reading's second channel the split's excess is the larger; at 30000 that reading
    holds the split near zero on the Hill, where its envelope stands barely above it, and the first
    reading's own plan's gap is."""
    readings = [
        (Channel(ONE, Hill(100.0, 3.0), 1000.0), Channel(ONE, MichaelisMenten(100.0), 300.0)),
        (Channel(ONE, MichaelisMenten(60.0), 700.0), Channel(ONE, MichaelisMenten(100.0), weight)),
    ]
    box = {"lower": np.zeros(2), "upper": np.full(2, 90.0), "history": np.zeros((0, 2))}
    with caplog.at_level(logging.WARNING, logger="chc.allocation"):
        plan = minimax_allocate(readings, 90.0, 1, **box)
    [record] = [r for r in caplog.records if r.name == "chc.allocation"]
    assert (record.chc_event, record.planner) == ("allocation_unsearched", "minimax_allocate")
    hill, other = plan.spend
    on_curves = (
        1000.0 * hill**3 / (100.0**3 + hill**3) + 300.0 * other / (100.0 + other),
        700.0 * hill / (60.0 + hill) + weight * other / (100.0 + other),
    )
    off = [
        allocate(reading, 90.0, 1, **box).worth - value - regret
        for reading, value, regret in zip(readings, on_curves, plan.regret, strict=True)
    ]
    # the Hill's envelope from zero spend is the chord to its tangency, z^3 = 2, past the cap
    chord = 1000.0 * (2.0 / 3.0) * 2.0 ** (-1.0 / 3.0) / 100.0
    own = chord * 90.0 - 1000.0 * 0.9**3 / (1.0 + 0.9**3)  # its plan on them: all 90 on the Hill
    excess = chord * hill - 1000.0 * (hill / 100.0) ** 3 / (1.0 + (hill / 100.0) ** 3)
    assert record.gap == pytest.approx(max(own, excess), rel=1e-9, abs=0.0)
    assert (own > excess) == (weight > 300.0)
    # the regret on the curves is the envelopes' less what the reading's best loses to its
    # envelopes' and plus the split's excess, so the two pull apart and each bounds a side
    assert -own * (1 + 1e-9) <= off[0] <= excess * (1 + 1e-9)
    assert off[1] == pytest.approx(0.0, abs=1e-9 * plan.best.max())


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
