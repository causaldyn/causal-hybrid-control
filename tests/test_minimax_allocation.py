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


def _hill_and_saturating(weight: float) -> list[tuple[Channel, Channel]]:
    """The first reading a Hill of slope 3 and a saturating curve, the second two saturating curves,
    its second weighed ``weight``."""
    return [
        (Channel(ONE, Hill(100.0, 3.0), 1000.0), Channel(ONE, MichaelisMenten(100.0), 300.0)),
        (Channel(ONE, MichaelisMenten(60.0), 700.0), Channel(ONE, MichaelisMenten(100.0), weight)),
    ]


def _on_the_curves(weight: float, hill, other) -> np.ndarray:
    return np.array(
        [
            1000.0 * hill**3 / (100.0**3 + hill**3) + 300.0 * other / (100.0 + other),
            700.0 * hill / (60.0 + hill) + weight * other / (100.0 + other),
        ]
    )


S_BOX = {"lower": np.zeros(2), "upper": np.full(2, 90.0), "history": np.zeros((0, 2))}


@pytest.mark.parametrize(
    ("weight", "worst"),
    [(300.0, 6.647329059634728), (30000.0, 279.5257374204743)],
    ids=["split", "corner"],
)
def test_an_s_curve_s_regret_is_read_on_the_curves_against_allocate_s_best(caplog, weight, worst):
    """Each reading's best is :func:`allocate`'s plan's return, all 90 on the Hill for the first,
    421.63, and each regret is read on the curves. On the envelopes from zero spend the first best
    was their plan's bound, 476.22, and the regrets were read on them too: at a weight of 300 on
    the second reading's second channel the split put 87.6 on the Hill, a worst regret on the
    curves of 12.42; at 30000 it put 4.25 there, where the Hill's chord returned 5.29 a unit and
    the curve returns 0.08 in all, a worst regret of 315.22. Against every split of a grid of steps
    of 0.001."""
    readings = _hill_and_saturating(weight)
    with caplog.at_level(logging.WARNING, logger="chc.allocation"):
        plan = minimax_allocate(readings, 90.0, 1, **S_BOX)
    assert not [r for r in caplog.records if r.name == "chc.allocation"]
    best = np.array([allocate(reading, 90.0, 1, **S_BOX).worth for reading in readings])
    np.testing.assert_array_equal(plan.best, best)
    assert best[0] == pytest.approx(1000.0 * 0.9**3 / (1.0 + 0.9**3), rel=1e-12, abs=0.0)
    np.testing.assert_allclose(
        plan.regret,
        best - _on_the_curves(weight, *plan.spend),
        rtol=1e-12,
        atol=1e-12 * best.max(),
    )
    assert plan.worst == plan.regret.max()
    assert plan.worst == pytest.approx(worst, rel=1e-9, abs=0.0)
    assert not np.signbit(plan.regret).any()  # a regret of nothing reads 0.0
    assert (plan.stopped, plan.limit) == ("closed", None)
    assert 0.0 <= plan.worst - plan.bound <= plan.tolerance
    hill = np.linspace(0.0, 90.0, 90001)
    least = float((best[:, None] - _on_the_curves(weight, hill, 90.0 - hill)).max(axis=0).min())
    assert plan.worst <= least + plan.tolerance
    assert least - plan.worst < 1e-3 * plan.worst


def test_mirrored_s_curve_readings_are_hedged_all_on_one_channel():
    """Two Hill curves of slope 3 at scales 1 and 1.01, read once each way round: each reading's
    best is all of 1.6 on its channel at scale 1, ``512/637``. All on either channel leaves one
    reading nothing to regret and the other ``512/637 - h(1.6/1.01)``, 0.00475; the even split, the
    plan on the envelopes from zero spend, leaves both 0.133
    (validation/envelope_on_interval.mac, STEP 4)."""
    untied = (Channel(ONE, Hill(1.0, 3.0), 1.0), Channel(ONE, Hill(1.01, 3.0), 1.0))
    plan = minimax_allocate(
        [untied, untied[::-1]], 1.6, 1, lower=np.zeros(2), upper=np.full(2, 1.6)
    )
    np.testing.assert_allclose(np.sort(plan.spend), [0.0, 1.6], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(plan.best, [512 / 637, 512 / 637], rtol=1e-12, atol=0.0)
    z = 1.6 / 1.01
    assert plan.worst == pytest.approx(512 / 637 - z**3 / (1 + z**3), rel=1e-9, abs=0.0)
    assert (plan.stopped, plan.limit) == ("closed", None)
    s = np.linspace(0.0, 1.6, 16001)
    h = lambda x, scale: (x / scale) ** 3 / (1.0 + (x / scale) ** 3)  # noqa: E731
    worst = np.maximum(
        512 / 637 - h(s, 1.0) - h(1.6 - s, 1.01), 512 / 637 - h(s, 1.01) - h(1.6 - s, 1.0)
    )
    assert plan.worst <= float(worst.min()) + plan.tolerance


def test_a_minimax_search_its_cap_stops_says_so_and_logs_its_gap(caplog):
    """The mirrored readings in one box: each reading's own search is the plan on the envelopes
    from zero spend, 1.27/0.33 for 0.7048 under a bound of 0.8448, and logs that gap as
    :func:`allocate` does. The best is that plan's return, not its bound; against such bests a
    split can gain up to the least of those gaps, so no worst regret is bounded above its
    negation until a box is cut."""
    untied = (Channel(ONE, Hill(1.0, 3.0), 1.0), Channel(ONE, Hill(1.01, 3.0), 1.0))
    readings = [untied, untied[::-1]]
    box = {"lower": np.zeros(2), "upper": np.full(2, 1.6)}
    own = [allocate(reading, 1.6, 1, **box, max_boxes=1) for reading in readings]
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="chc.allocation"):
        plan = minimax_allocate(readings, 1.6, 1, **box, max_boxes=1)
    assert (plan.stopped, plan.boxes, plan.limit) == ("cap", 1, "max_boxes")
    np.testing.assert_array_equal(plan.best, [p.worth for p in own])
    gaps = [p.bound - p.worth for p in own]
    assert min(gaps) > 0.1
    assert plan.bound >= -min(gaps) * (1 + 1e-12)
    records = [r for r in caplog.records if r.name == "chc.allocation"]
    assert [r.planner for r in records] == ["allocate", "allocate", "minimax_allocate"]
    record = records[-1]
    assert record.chc_event == "allocation_cap"
    assert (record.boxes, record.limit) == (1, "max_boxes")
    assert (record.worst, record.bound) == (plan.worst, plan.bound)
    assert plan.gap == plan.worst - plan.bound
    assert plan.gap > 0.1


@pytest.mark.parametrize(
    ("setting", "match"),
    [({"rtol": -1.0}, "rtol"), ({"atol": np.inf}, "atol"), ({"max_boxes": 0}, "max_boxes")],
)
def test_it_refuses_search_settings_it_cannot_search_to(setting, match):
    with pytest.raises(ValueError, match=match):
        minimax_allocate(
            READINGS, BUDGET, PERIODS, lower=LOWER, upper=UPPER, history=HISTORY, **setting
        )


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


@pytest.mark.parametrize(("at", "status"), [(1, 1), (5, 1), (5, 4)])
def test_a_program_left_unsolved_ends_the_rounds_on_the_last_floor(stall, at, status):
    """HiGHS can pivot without end on planes cut at nearly the same split, so each program runs
    under an iteration limit. One stopped there, or that HiGHS ends unsolved, ends the rounds: the
    floor of the programs before it still bounds the least worst regret from below, as the split's
    worst regret bounds it from above."""
    box = {"lower": LOWER, "upper": UPPER, "history": HISTORY}
    exact = minimax_allocate(READINGS, BUDGET, PERIODS, **box)
    limits = stall(at, status)
    plan = minimax_allocate(READINGS, BUDGET, PERIODS, **box)
    assert len(limits) == at
    assert min(limits) > 0
    assert plan.bound <= exact.worst + 1e-6
    assert plan.worst >= exact.bound - 1e-6
