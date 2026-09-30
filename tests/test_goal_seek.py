"""chc.allocation.budget_for: a goal in place of a budget.

The closed forms are validation/goal_seek.mac's: one channel with no carryover, where each goal's
budget solves an equation in the curve, and a floor that makes the average return cross its target
twice. On channels with carryover and a history, each goal is checked on the plan's gain as the
channels themselves return it, run over the history, the plan and the tail as one series.
"""

import numpy as np
import pytest
from scipy.special import lambertw

from chc.allocation import (
    MarginalReturnTarget,
    ReturnOnSpendTarget,
    ReturnTarget,
    allocate,
    budget_for,
)
from chc.response import (
    Channel,
    DelayedAdstock,
    Exponential,
    GeometricAdstock,
    Hill,
    MichaelisMenten,
    Power,
    Tanh,
    WeibullAdstock,
)

ONE = GeometricAdstock(0.0, length=1, normalized=False)  # no carryover
NONE_BEFORE = np.zeros((0, 1))

PERIODS = 13
CHANNELS = (
    Channel(GeometricAdstock(0.3, length=6, normalized=True), Tanh(250.0), 1100.0),
    Channel(DelayedAdstock(0.6, 1.5, length=8, normalized=True), MichaelisMenten(120.0), 900.0),
    Channel(WeibullAdstock(1.2, 3.0, length=10, normalized=True), Exponential(300.0), 1400.0),
)
HISTORY = np.random.default_rng(3).gamma(4.0, 20.0, size=(30, 3))
LOWER = np.array([40.0, 25.0, 15.0])
UPPER = np.array([160.0, 100.0, 60.0])


def _one(curve, coefficient, goal):
    channel = (Channel(ONE, curve, coefficient),)
    return budget_for(channel, goal, 1, lower=[0.0], upper=[2000.0], history=NONE_BEFORE)


def _seek(goal, channels=CHANNELS, lower=LOWER, upper=UPPER, history=HISTORY):
    return budget_for(channels, goal, PERIODS, lower=lower, upper=upper, history=history)


def _gain(spend, channels=CHANNELS, history=HISTORY) -> float:
    """What the plan adds, from each channel run over history, plan and tail as one series."""

    def worth(rates) -> float:
        total = 0.0
        for column, (channel, rate) in enumerate(zip(channels, rates, strict=True)):
            tail = np.zeros(channel.kernel.length - 1)
            series = np.concatenate([history[:, column], np.full(PERIODS, rate), tail])
            total += float(np.sum(np.asarray(channel(series))[history.shape[0] :]))
        return total

    return worth(spend) - worth(np.zeros(len(channels)))


@pytest.mark.parametrize(
    ("goal", "budget"),
    [
        (ReturnTarget(450.0), 450.0 * 50.0 / (900.0 - 450.0)),
        (MarginalReturnTarget(1.0), np.sqrt(900.0 * 50.0 / 1.0) - 50.0),
        (ReturnOnSpendTarget(4.0), 900.0 / 4.0 - 50.0),
    ],
)
def test_each_goal_s_budget_on_a_michaelis_menten_channel_is_its_closed_form(goal, budget):
    plan = _one(MichaelisMenten(50.0), 900.0, goal)
    assert plan.budget == pytest.approx(budget, rel=1e-12)
    assert plan.gain == pytest.approx(900.0 * budget / (50.0 + budget), rel=1e-12)


def test_each_goal_s_budget_on_an_exponential_channel_is_its_closed_form():
    beta, scale = 1400.0, 300.0
    c = beta / (3.0 * scale)
    closed = {
        ReturnTarget(700.0): -scale * np.log(1.0 - 700.0 / beta),
        MarginalReturnTarget(2.0): scale * np.log(beta / (scale * 2.0)),
        ReturnOnSpendTarget(3.0): scale * (c + lambertw(-c * np.exp(-c)).real),
    }
    for goal, budget in closed.items():
        plan = _one(Exponential(scale), beta, goal)
        assert plan.budget == pytest.approx(budget, rel=1e-11), goal


def test_a_target_return_on_spend_is_the_most_budget_that_meets_it_not_the_first():
    """validation/goal_seek.mac STEP 3: a linear channel returning 1/2 a unit is held at its floor
    of 200, so the average return rises from 1/2 as the concave channel takes spend, crosses 2 at
    325 - 25 sqrt(13), peaks, and falls back through 2 at 325 + 25 sqrt(13)."""
    channels = (
        Channel(ONE, MichaelisMenten(50.0), 900.0),
        Channel(ONE, Power(100.0, 1.0), 50.0),
    )
    plan = budget_for(
        channels,
        ReturnOnSpendTarget(2.0),
        1,
        lower=[0.0, 200.0],
        upper=[400.0, 400.0],
        history=np.zeros((0, 2)),
    )
    assert plan.budget == pytest.approx(325.0 + 25.0 * np.sqrt(13.0), rel=1e-12)
    assert plan.spend[1] == 200.0
    assert plan.gain / plan.budget == pytest.approx(2.0, rel=1e-12)


def test_a_return_target_inside_a_linear_channel_s_jump_mixes_its_ends():
    """At a price of 2 the Michaelis-Menten channel runs to sqrt(900 * 50 / 2) - 50 = 100, worth
    600, and the linear one, returning 2 a unit, jumps from 0 to its cap; a gain of 700 takes 50
    of it, so the budget is 150, which neither end of the jump spends."""
    channels = (
        Channel(ONE, MichaelisMenten(50.0), 900.0),
        Channel(ONE, Power(100.0, 1.0), 200.0),
    )
    plan = budget_for(
        channels,
        ReturnTarget(700.0),
        1,
        lower=np.zeros(2),
        upper=np.full(2, 150.0),
        history=np.zeros((0, 2)),
    )
    np.testing.assert_allclose(plan.spend, [100.0, 50.0], rtol=1e-12)
    assert plan.gain == pytest.approx(700.0, rel=1e-13)


def test_a_return_target_is_met_by_the_least_budget():
    goal = ReturnTarget(10000.0)
    plan = _seek(goal)
    assert _gain(plan.spend) == pytest.approx(10000.0, rel=1e-10)
    assert plan.gain == pytest.approx(10000.0, rel=1e-10)
    less = allocate(
        CHANNELS, plan.budget * (1 - 1e-6), PERIODS, lower=LOWER, upper=UPPER, history=HISTORY
    )
    assert less.gain < 10000.0


def test_a_marginal_target_is_the_price_every_channel_inside_its_box_meets():
    plan = _seek(MarginalReturnTarget(3.5))
    assert plan.price == pytest.approx(3.5, rel=1e-9)
    inside = 0
    for column in range(len(CHANNELS)):
        if LOWER[column] < plan.spend[column] < UPPER[column]:
            inside += 1
            step = 1e-4 * plan.spend[column]
            up, down = plan.spend.copy(), plan.spend.copy()
            up[column] += step
            down[column] -= step
            slope = (_gain(up) - _gain(down)) / (2 * step) / PERIODS
            assert slope == pytest.approx(3.5, rel=1e-6)
    assert inside >= 2  # the case exercises the interior, not only the corners


def test_a_target_return_on_spend_is_met_and_a_larger_budget_misses_it():
    plan = _seek(ReturnOnSpendTarget(4.4))
    assert _gain(plan.spend) / plan.budget == pytest.approx(4.4, rel=1e-10)
    more = allocate(
        CHANNELS, plan.budget * (1 + 1e-6), PERIODS, lower=LOWER, upper=UPPER, history=HISTORY
    )
    assert more.gain / more.budget < 4.4


@pytest.mark.parametrize(
    ("goal", "end"),
    [
        (ReturnTarget(0.0), LOWER),  # the floors already gain it
        (MarginalReturnTarget(1e6), LOWER),  # no channel returns that much a unit at its floor
        (MarginalReturnTarget(0.0), UPPER),
        (MarginalReturnTarget(-1.0), UPPER),
        (ReturnOnSpendTarget(1e-3), UPPER),  # even the caps return more a unit on average
    ],
)
def test_a_goal_the_box_binds_is_met_at_the_end_of_the_box(goal, end):
    np.testing.assert_array_equal(_seek(goal).spend, end)


def test_an_s_curve_s_return_target_is_met_on_the_true_curve():
    channels = (Channel(ONE, Hill(100.0, 3.0), 1000.0), Channel(ONE, MichaelisMenten(100.0), 300.0))
    plan = budget_for(
        channels,
        ReturnTarget(900.0),
        1,
        lower=np.zeros(2),
        upper=np.full(2, 400.0),
        history=np.zeros((0, 2)),
    )
    s = plan.spend
    revenue = 1000.0 * s[0] ** 3 / (100.0**3 + s[0] ** 3) + 300.0 * s[1] / (100.0 + s[1])
    assert revenue == pytest.approx(900.0, rel=1e-10)
    assert plan.gain == pytest.approx(900.0, rel=1e-10)


@pytest.mark.parametrize(
    "goal",
    [ReturnTarget(10000.0), MarginalReturnTarget(3.5), ReturnOnSpendTarget(4.4)],
    ids=["return", "marginal", "on-spend"],
)
def test_a_change_of_currency_moves_the_budget_and_leaves_the_gain(goal):
    """The channels' coefficients are in the outcome's units, so a return a currency unit moves
    with the currency and a return does not."""
    k = 1000.0

    def rescaled(channel: Channel) -> Channel:
        return Channel(
            channel.kernel, type(channel.curve)(k * channel.curve.scale), channel.coefficient
        )

    moved_goal = goal if isinstance(goal, ReturnTarget) else type(goal)(per_unit=goal.per_unit / k)
    plan = _seek(goal)
    moved = _seek(
        moved_goal,
        channels=tuple(rescaled(c) for c in CHANNELS),
        lower=k * LOWER,
        upper=k * UPPER,
        history=k * HISTORY,
    )
    assert moved.budget == pytest.approx(k * plan.budget, rel=1e-9)
    assert moved.gain == pytest.approx(plan.gain, rel=1e-9)


@pytest.mark.parametrize(
    ("goal", "error", "match"),
    [
        (ReturnTarget(1e7), ValueError, "no plan in the box gains"),
        (ReturnOnSpendTarget(50.0), ValueError, "no budget in the box gains"),
        (ReturnTarget(float("nan")), ValueError, "needs a finite one"),
        (MarginalReturnTarget(float("inf")), ValueError, "needs a finite one"),
        (10000.0, TypeError, "not a Goal"),
    ],
)
def test_it_refuses_a_goal_it_cannot_meet(goal, error, match):
    with pytest.raises(error, match=match):
        _seek(goal)
