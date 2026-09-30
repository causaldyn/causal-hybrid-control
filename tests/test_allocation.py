"""chc.allocation: a budget spread over channels, checked against a second route to each number.

A plan's worth is recomputed here from the channels themselves, on the history, the plan and the
tail as one spend series; the best plan against SciPy's general solver and a grid; the price
against the worth's own slope.
"""

import numpy as np
import pytest
from scipy.optimize import minimize

from chc.allocation import allocate
from chc.response import (
    Channel,
    DelayedAdstock,
    Exponential,
    GeometricAdstock,
    Hill,
    MichaelisMenten,
    Power,
    Ricker,
    Tanh,
    WeibullAdstock,
)

PERIODS = 13
CHANNELS = (
    Channel(GeometricAdstock(0.3, length=6, normalized=True), Tanh(250.0), 1100.0),
    Channel(DelayedAdstock(0.6, 1.5, length=8, normalized=True), MichaelisMenten(120.0), 900.0),
    Channel(WeibullAdstock(1.2, 3.0, length=10, normalized=True), Exponential(300.0), 1400.0),
)
HISTORY = np.random.default_rng(3).gamma(4.0, 20.0, size=(30, 3))
LOWER = np.array([40.0, 25.0, 15.0])
UPPER = np.array([160.0, 100.0, 60.0])
BUDGET = PERIODS * 150.0


def _worth(channels, spend, history=HISTORY, periods=PERIODS) -> float:
    """What the plan returns, from each channel run over history, plan and tail as one series."""
    total = 0.0
    for column, (channel, rate) in enumerate(zip(channels, spend, strict=True)):
        tail = np.zeros(channel.kernel.length - 1)
        series = np.concatenate([history[:, column], np.full(periods, rate), tail])
        total += float(np.sum(np.asarray(channel(series))[history.shape[0] :]))
    return total


def _plan(channels=CHANNELS, budget=BUDGET, lower=LOWER, upper=UPPER, history=HISTORY):
    return allocate(channels, budget, PERIODS, lower=lower, upper=upper, history=history)


def test_a_plan_spends_the_budget_in_the_box_and_its_worth_is_the_channels_return():
    plan = _plan()
    assert PERIODS * plan.spend.sum() == pytest.approx(BUDGET, rel=1e-13)
    assert np.all(plan.spend >= LOWER)
    assert np.all(plan.spend <= UPPER)
    assert plan.worth == pytest.approx(_worth(CHANNELS, plan.spend), rel=1e-12)
    assert plan.bound == plan.worth


def test_the_price_is_the_slope_of_every_channel_inside_its_box():
    plan = _plan()
    inside = 0
    for column in range(len(CHANNELS)):
        step = 1e-4 * plan.spend[column]
        up, down = plan.spend.copy(), plan.spend.copy()
        up[column] += step
        down[column] -= step
        slope = (_worth(CHANNELS, up) - _worth(CHANNELS, down)) / (2 * step) / PERIODS
        if LOWER[column] < plan.spend[column] < UPPER[column]:
            inside += 1
            assert slope == pytest.approx(plan.price, rel=1e-6)
        elif plan.spend[column] == LOWER[column]:
            assert slope <= plan.price * (1 + 1e-6)
        else:
            assert slope >= plan.price * (1 - 1e-6)
    assert inside == 3  # the case exercises the interior, not only the corners


def test_no_plan_in_the_box_returns_more():
    plan = _plan()

    def loss(spend):
        return -_worth(CHANNELS, spend) / BUDGET

    general = minimize(
        loss,
        np.full(3, BUDGET / PERIODS / 3),
        method="SLSQP",
        bounds=list(zip(LOWER, UPPER, strict=True)),
        constraints={"type": "eq", "fun": lambda s: PERIODS * s.sum() / BUDGET - 1.0},
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    assert general.success, general.message
    assert -general.fun * BUDGET <= plan.worth * (1 + 1e-10)
    assert -general.fun * BUDGET == pytest.approx(plan.worth, rel=1e-8)
    rng = np.random.default_rng(0)
    for _ in range(200):
        spend = rng.uniform(LOWER, UPPER)
        spend *= BUDGET / PERIODS / spend.sum()
        if np.all(spend >= LOWER) and np.all(spend <= UPPER):
            assert _worth(CHANNELS, spend) <= plan.worth * (1 + 1e-12)


def test_a_linear_channel_takes_exactly_what_the_concave_one_leaves():
    """A linear rate jumps from floor to cap at one price; the plan mixes the two sides so the
    budget is spent to rounding, not to the bisection's tolerance."""
    channels = (
        Channel(GeometricAdstock(0.0, length=1, normalized=False), Power(100.0, 1.0), 200.0),
        Channel(GeometricAdstock(0.0, length=1, normalized=False), MichaelisMenten(50.0), 900.0),
    )
    history = np.zeros((0, 2))
    plan = allocate(channels, 150.0, 1, lower=np.zeros(2), upper=np.full(2, 150.0), history=history)
    assert plan.spend.sum() == pytest.approx(150.0, rel=1e-14)
    # the concave channel runs until its slope, 900 * 50 / (50 + s)^2, falls to the linear one's, 2
    assert plan.spend[1] == pytest.approx(np.sqrt(900.0 * 50.0 / 2.0) - 50.0, rel=1e-12)
    assert plan.price == pytest.approx(2.0, rel=1e-9)


def test_the_history_s_carryover_holds_back_the_channel_it_has_saturated():
    twin = Channel(GeometricAdstock(0.7, length=8, normalized=False), Tanh(100.0), 500.0)
    history = np.zeros((20, 2))
    history[-5:, 0] = 120.0
    plan = allocate(
        (twin, twin), 2 * 4 * 50.0, 4, lower=np.zeros(2), upper=np.full(2, 100.0), history=history
    )
    assert plan.spend[0] < plan.spend[1]
    even = allocate((twin, twin), 2 * 4 * 50.0, 4, lower=np.zeros(2), upper=np.full(2, 100.0))
    np.testing.assert_allclose(even.spend, [50.0, 50.0], rtol=1e-9)


def test_an_s_curve_is_planned_on_its_envelope_and_the_gap_bounds_the_shortfall():
    """The S-curve counterexample of chc.response's tests: from zero spend a descent stops at the
    greedy 225; the plan on the envelope reaches the best split, 1047.87, with no gap, since the
    best split spends past the tangency. With a budget short of the tangency the envelope's plan
    runs on the chord, and the best split lies between its worth and its bound."""
    one = GeometricAdstock(0.0, length=1, normalized=False)
    channels = (Channel(one, Hill(100.0, 3.0), 1000.0), Channel(one, MichaelisMenten(100.0), 300.0))

    def best(budget: float) -> float:
        grid = np.linspace(0.0, budget, 200001)
        revenue = 1000.0 * grid**3 / (100.0**3 + grid**3) + 300.0 * (budget - grid) / (
            100.0 + budget - grid
        )
        return float(revenue.max())

    for budget in (300.0, 90.0):
        plan = allocate(
            channels,
            budget,
            1,
            lower=np.zeros(2),
            upper=np.full(2, budget),
            history=np.zeros((0, 2)),
        )
        assert plan.spend.sum() == pytest.approx(budget, rel=1e-13)
        optimum = best(budget)
        assert plan.worth <= optimum * (1 + 1e-9)
        assert plan.bound >= optimum * (1 - 1e-9)
        if budget == 300.0:
            assert plan.worth == pytest.approx(1047.867, abs=1e-3)
            assert plan.bound - plan.worth <= 1e-9 * plan.worth
        else:
            assert plan.bound - plan.worth > 1e-3


def test_a_change_of_currency_moves_the_spend_and_the_price_and_leaves_the_worth():
    k = 1000.0

    def rescaled(channel: Channel) -> Channel:
        return Channel(
            channel.kernel, type(channel.curve)(k * channel.curve.scale), channel.coefficient
        )

    plan = _plan()
    moved = allocate(
        tuple(rescaled(c) for c in CHANNELS),
        k * BUDGET,
        PERIODS,
        lower=k * LOWER,
        upper=k * UPPER,
        history=k * HISTORY,
    )
    np.testing.assert_allclose(moved.spend, k * plan.spend, rtol=1e-9)
    assert moved.worth == pytest.approx(plan.worth, rel=1e-12)
    assert moved.price == pytest.approx(plan.price / k, rel=1e-9)


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        ({"budget": PERIODS * 400.0}, ValueError, "outside what the box spends"),
        ({"lower": LOWER[:2]}, ValueError, "one finite rate a channel"),
        ({"upper": LOWER - 1.0}, ValueError, "not 0 <= lower <= upper"),
        ({"history": HISTORY[:, :2]}, ValueError, "one column a channel"),
        ({"history": -HISTORY}, ValueError, "negative or non-finite"),
        (
            {"channels": (*CHANNELS[:2], Channel(CHANNELS[2].kernel, Ricker(100.0), 50.0))},
            ValueError,
            "not increasing and concave",
        ),
        (
            {"channels": (*CHANNELS[:2], Channel(CHANNELS[2].kernel, Tanh(100.0), -5.0))},
            ValueError,
            "turns its curve convex",
        ),
        ({"channels": (*CHANNELS[:2], Tanh(100.0))}, TypeError, "not a Channel"),
    ],
)
def test_it_refuses_what_it_cannot_plan(change, error, match):
    with pytest.raises(error, match=match):
        _plan(**change)
