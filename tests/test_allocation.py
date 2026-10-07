"""chc.allocation: a budget spread over channels, checked against a second route to each number.

A plan's worth is recomputed here from the channels themselves, on the history, the plan and the
tail as one spend series; the best plan against SciPy's general solver and a grid; the price
against the worth's own slope.
"""

import importlib
import logging
import threading

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import minimize, minimize_scalar

import chc.allocation
from chc.allocation import (
    MarginalReturnTarget,
    ReturnTarget,
    _bounded,
    _on_envelopes,
    _slope,
    _value,
    _Worth,
    _worths,
    allocate,
    budget_for,
)
from chc.response import (
    Channel,
    DelayedAdstock,
    Exponential,
    GeometricAdstock,
    Hill,
    Logistic,
    MichaelisMenten,
    Power,
    Ricker,
    Tanh,
    Weibull,
    WeibullAdstock,
)

# Public as jax.enable_x64 from jax 0.8.0; the floor, 0.4.30, has only jax.experimental.enable_x64,
# which jax 0.11 no longer has.
if hasattr(jax, "enable_x64"):
    enable_x64 = jax.enable_x64
else:
    enable_x64 = importlib.import_module("jax.experimental").enable_x64

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


def test_a_channel_without_carryover_on_a_curve_steep_at_zero_spend_is_planned():
    """With no carryover the periods after the plan see no spend, and the planner reads the curve
    there at zero spend through a zero weight; below slope 1 a Hill's slope there is infinite, and
    the product was nan."""
    channels = (
        Channel(GeometricAdstock(0.5, length=10, normalized=False), Exponential(80.0), 300.0),
        Channel(GeometricAdstock(0.0, length=10, normalized=False), Hill(51.0, 0.5), 234.6),
    )
    history = np.zeros((0, 2))
    budget = 1400.0
    plan = allocate(
        channels, budget, PERIODS, lower=np.zeros(2), upper=np.full(2, 200.0), history=history
    )
    assert PERIODS * plan.spend.sum() == pytest.approx(budget, rel=1e-13)
    assert plan.worth == pytest.approx(_worth(channels, plan.spend, history), rel=1e-12)
    rate = budget / PERIODS
    best = minimize_scalar(
        lambda first: -_worth(channels, [first, rate - first], history),
        bounds=(0.0, rate),
        method="bounded",
        options={"xatol": 1e-10},
    )
    assert plan.worth >= -best.fun * (1 - 1e-12)
    assert plan.spend[0] == pytest.approx(best.x, rel=1e-5)


def test_an_s_curve_is_planned_to_the_best_split_and_the_bound_closes_on_it():
    """The S-curve counterexample of chc.response's tests: from zero spend a descent stops at the
    greedy 225, and the best split, 1047.87, spends past the tangency. With a budget short of the
    tangency the plan on the envelope from zero spend is the best split, all of it on the S-curve,
    but its chord bounds the plan 54.6 above it; the envelope over the box closes the bound."""
    one = GeometricAdstock(0.0, length=1, normalized=False)
    channels = (Channel(one, Hill(100.0, 3.0), 1000.0), Channel(one, MichaelisMenten(100.0), 300.0))

    def best(budget: float) -> float:
        grid = np.linspace(0.0, budget, 200001)
        revenue = 1000.0 * grid**3 / (100.0**3 + grid**3) + 300.0 * (budget - grid) / (
            100.0 + budget - grid
        )
        return float(revenue.max())

    for budget in (300.0, 90.0):
        box = {"lower": np.zeros(2), "upper": np.full(2, budget), "history": np.zeros((0, 2))}
        plan = allocate(channels, budget, 1, **box)
        assert plan.spend.sum() == pytest.approx(budget, rel=1e-13, abs=0.0)
        optimum = best(budget)
        assert plan.worth == pytest.approx(optimum, rel=1e-9, abs=0.0)
        assert plan.bound >= optimum * (1 - 1e-12)
        assert plan.bound - plan.worth <= 1e-9 * plan.bound
        if budget == 300.0:
            assert plan.worth == pytest.approx(1047.867, abs=1e-3)
        else:
            envelope = _on_envelopes(channels, budget, 1, *box.values())
            assert envelope.bound - envelope.worth > 50.0


def test_the_untied_example_is_planned_all_on_the_first_channel():
    """Two Hill curves of slope 3 at scales 1 and 1.01 and a budget of 1.6: on the envelopes the
    plan is 1.27/0.33 for 0.705; all of it on the first returns 512/637, and no split does better
    (validation/envelope_on_interval.mac, STEP 4)."""
    one = GeometricAdstock(0.0, length=1, normalized=False)
    channels = (Channel(one, Hill(1.0, 3.0), 1.0), Channel(one, Hill(1.01, 3.0), 1.0))
    plan = allocate(channels, 1.6, 1, lower=np.zeros(2), upper=np.full(2, 1.6))
    np.testing.assert_array_equal(plan.spend, [1.6, 0.0])
    assert plan.worth == pytest.approx(512 / 637, rel=1e-15, abs=0.0)
    assert plan.bound - plan.worth <= 1e-9 * plan.bound
    envelope = _on_envelopes(channels, 1.6, 1, np.zeros(2), np.full(2, 1.6), np.zeros((0, 2)))
    assert envelope.worth == pytest.approx(0.705, abs=5e-4)


def test_a_cap_short_of_the_tangency_is_bounded_on_its_own_interval():
    """Capped at 1, short of its tangency 2^(1/3), a Hill curve of slope 3 is bounded by the chord
    to its cap, not by the one from the origin to the tangency, which stands above the cap's value:
    the envelopes from zero spend bound the plan by 0.815, and the best split returns 0.660."""
    one = GeometricAdstock(0.0, length=1, normalized=False)
    channels = (Channel(one, Hill(1.0, 3.0), 1.0), Channel(one, Hill(1.0, 3.0), 0.9))
    box = (np.zeros(2), np.ones(2), np.zeros((0, 2)))
    plan = allocate(channels, 1.6, 1, lower=box[0], upper=box[1], history=box[2])
    np.testing.assert_allclose(plan.spend, [1.0, 0.6], rtol=1e-12)
    assert plan.worth == pytest.approx(0.5 + 0.9 * 0.216 / 1.216, rel=1e-14, abs=0.0)
    assert plan.bound - plan.worth <= 1e-9 * plan.bound
    assert _on_envelopes(channels, 1.6, 1, *box).bound > plan.bound + 0.15


def _brute(channels, budget, periods, lower, upper, history, points=4001) -> float:
    """The best split of two channels, on a grid of the first's rate and polished between its
    neighbours, each split's worth from the channels run over history, plan and tail."""
    rate = budget / periods

    def worth(first):
        total = 0.0
        for column, (channel, spend) in enumerate(
            zip(channels, (first, rate - first), strict=True)
        ):
            tail = jnp.zeros(channel.kernel.length - 1)
            series = jnp.concatenate([history[:, column], jnp.full(periods, spend), tail])
            total += jnp.sum(channel(series)[history.shape[0] :])
        return total

    low, high = max(lower[0], rate - upper[1]), min(upper[0], rate - lower[1])
    grid = np.linspace(low, high, points)
    values = np.asarray(jax.vmap(worth)(jnp.asarray(grid)))
    at = int(np.argmax(values))
    polished = minimize_scalar(
        lambda first: -float(worth(first)),
        bounds=(grid[max(at - 1, 0)], grid[min(at + 1, points - 1)]),
        method="bounded",
        options={"xatol": 1e-12},
    )
    return max(float(values[at]), -float(polished.fun))


@pytest.mark.parametrize(
    "channels",
    [
        (
            Channel(GeometricAdstock(0.6, length=6, normalized=True), Hill(1.0, 3.0), 1.0),
            Channel(GeometricAdstock(0.3, length=4, normalized=False), Logistic(1.2, 6.0), 0.7),
        ),
        (
            Channel(GeometricAdstock(0.5, length=5, normalized=True), Weibull(0.8, 3.0), 0.8),
            Channel(GeometricAdstock(0.0, length=1, normalized=False), MichaelisMenten(1.0), 0.6),
        ),
    ],
    ids=["hill-logistic", "weibull-concave"],
)
def test_s_curves_with_carryover_and_history_are_planned_to_the_best_split(channels):
    """A longer kernel runs each period at its own adstock, so each period's envelope is over its
    own interval; the plan is still the best split, against a grid on the channels themselves.
    Where a chord the cap cut short read the curve's slope at the cap, the Weibull case returned
    11.344346 and called it the bound, where 11.345382 is there."""
    history = np.random.default_rng(7).gamma(2.0, 0.3, size=(8, 2))
    lower, upper, periods = np.array([0.0, 0.2]), np.full(2, 2.5), 13
    budget = periods * 1.4
    plan = allocate(channels, budget, periods, lower=lower, upper=upper, history=history)
    best = _brute(channels, budget, periods, lower, upper, history)
    assert plan.worth == pytest.approx(_worth(channels, plan.spend, history), rel=1e-12)
    assert plan.worth >= best * (1 - 1e-9)
    assert plan.bound >= best * (1 - 1e-12)
    assert plan.bound - plan.worth <= 1e-9 * plan.bound


def test_a_chord_the_cap_cuts_short_keeps_its_own_slope_at_the_cap():
    """With carryover, each period's chord in this box is cut short by the cap. Read on the curve at
    the cap, the envelope's slope jumped to the curve's own there, steeper than the chord's, so the
    bisection held the channel at its cap at prices where the chords say less: the plan returned
    1.9622452274 and called it the bound, 2.2e-7 below the best split. A case found among 40 random
    ones, the only one of them it moved."""
    channels = (
        Channel(
            GeometricAdstock(0.3606111610980326, length=4, normalized=True),
            Hill(1.0, 3.573927321363234),
            0.9953053039431363,
        ),
        Channel(
            GeometricAdstock(0.0, length=1, normalized=False),
            MichaelisMenten(0.6458541751309267),
            0.8153814087978317,
        ),
    )
    spent = [1.4299767185967542, 0.5263948002104396, 0.8562372506316906, 0.5626505532446576]
    spent += [0.36167322263551166, 0.1545685893070136, 1.370787027095805, 0.4383762020156633]
    history = np.column_stack([spent, np.zeros(8)])
    lower, upper, periods = np.zeros(2), np.array([1.2119759132759984, 3.0]), 3
    budget = 3.639370652709662
    plan = allocate(channels, budget, periods, lower=lower, upper=upper, history=history)
    best = _brute(channels, budget, periods, lower, upper, history)
    assert plan.worth >= best * (1 - 1e-9)
    assert plan.bound >= best * (1 - 1e-12)
    assert plan.bound - plan.worth <= 1e-9 * plan.bound


def test_a_box_s_envelope_bounds_the_worth_meets_it_at_both_ends_and_is_concave():
    channel = Channel(GeometricAdstock(0.6, length=6, normalized=True), Hill(1.0, 4.0), 2.0)
    history = np.random.default_rng(1).gamma(2.0, 0.2, size=(5, 1))
    (worth,) = _worths((channel,), history, 9)
    for low, high in ((0.0, 3.0), (0.4, 0.9), (1.3, 1.31), (2.0, 3.0)):
        bounded = _bounded(worth, low, high)
        rates = np.linspace(low, high, 401)
        above = np.array([_value(bounded, r) for r in rates])
        on = np.array([_value(worth, r) for r in rates])
        assert np.all(above >= on - 1e-12)
        assert above[0] == pytest.approx(on[0], rel=1e-14, abs=0.0)
        assert above[-1] == pytest.approx(on[-1], rel=1e-14, abs=0.0)
        assert np.all(np.diff(above, 2) <= 1e-12)


def test_a_box_whose_adstock_spans_less_than_a_normal_double_reads_the_curve():
    """Both ends of this box leave a normal double of adstock, 2.5e-308 and 2.6e-308, but their
    difference is subnormal, and XLA flushes it to zero on the CPU while the ends still compare
    unequal: the chord's slope read 0 / 0, and a search handed its linear program a nan. A tail
    period that keeps almost nothing of a unit of spend's adstock leaves such a box."""
    channel = Channel(GeometricAdstock(0.0, length=1, normalized=False), Hill(1.0, 2.0), 1.0)
    worth = _Worth(channel, jnp.zeros(1), jnp.array([1e-307]))
    bounded = _bounded(worth, 0.25, 0.26)
    for rate in (0.25, 0.255, 0.26):
        assert _value(bounded, rate) == _value(worth, rate)
        assert np.isfinite(_slope(bounded, rate))


def _equal_s_curves(slope: float, copies: int) -> tuple[Channel, ...]:
    one = GeometricAdstock(0.0, length=1, normalized=False)
    return tuple(Channel(one, Hill(1.0, slope), 1.0) for _ in range(copies))


@pytest.mark.parametrize(
    ("slope", "copies", "budget"), [(3.0, 2, 1.6), (2.0, 2, 1.2), (3.0, 3, 2.4), (5.0, 4, 3.0)]
)
def test_equal_s_curves_leave_at_most_one_channel_inside_its_chord(slope, copies, budget):
    """Equal curves jump at one price on their envelopes, where every split of the jump is best
    there. Moved together, two equal Hill curves of slope 3 at 1.6 split 0.8/0.8 for 0.677, both
    inside their chords; filled one at a time, at most one channel is, so the gap is at most one
    curve's nonconvexity (Shapley and Folkman's lemma for one constraint). The goals and the splits
    for several readings plan on these envelopes, and the search starts from them."""
    channels = _equal_s_curves(slope, copies)
    plan = _on_envelopes(
        channels, budget, 1, np.zeros(copies), np.full(copies, budget), np.zeros((0, copies))
    )
    touch = channels[0].curve.tangency()
    inside = (plan.spend > 1e-12) & (plan.spend < touch * (1 - 1e-9))
    assert inside.sum() <= 1
    assert plan.spend.sum() == pytest.approx(budget, rel=1e-13, abs=0.0)
    assert plan.bound - plan.worth <= channels[0].curve.nonconvexity() * (1 + 1e-9)


@pytest.mark.parametrize(
    ("slope", "copies", "budget"), [(3.0, 2, 1.6), (2.0, 2, 1.2), (3.0, 3, 2.4), (5.0, 4, 3.0)]
)
def test_equal_s_curves_are_planned_to_the_best_split(slope, copies, budget):
    """Three Hill curves of slope 3 at 2.4 split 1.2/1.2/0 for 1.267, two of them inside their
    chords but past the bend; the envelopes' vertex, 1.26/1.14/0, returns 1.264. Against every
    split on a grid of steps of budget / 120 over the simplex."""
    channels = _equal_s_curves(slope, copies)
    plan = allocate(channels, budget, 1, lower=np.zeros(copies), upper=np.full(copies, budget))
    steps = 120
    shares = np.stack(
        np.meshgrid(*[np.arange(steps + 1)] * (copies - 1), indexing="ij"), axis=-1
    ).reshape(-1, copies - 1)
    shares = shares[shares.sum(axis=1) <= steps]
    split = np.concatenate([shares, steps - shares.sum(axis=1, keepdims=True)], axis=1)
    z = split * budget / steps
    best = float((z**slope / (1.0 + z**slope)).sum(axis=1).max())
    assert plan.worth >= best * (1 - 1e-12)
    assert plan.bound - plan.worth <= 1e-9 * plan.bound
    envelope = _on_envelopes(
        channels, budget, 1, np.zeros(copies), np.full(copies, budget), np.zeros((0, copies))
    )
    assert plan.worth >= envelope.worth * (1 - 1e-12)


def test_a_goal_on_equal_s_curves_is_met_with_at_most_one_channel_inside_its_chord():
    channels = _equal_s_curves(3.0, 2)
    plan = budget_for(channels, ReturnTarget(0.7), 1, lower=np.zeros(2), upper=np.full(2, 5.0))
    touch = channels[0].curve.tangency()
    inside = (plan.spend > 1e-12) & (plan.spend < touch * (1 - 1e-9))
    assert inside.sum() <= 1
    assert plan.gain == pytest.approx(0.7, rel=1e-9, abs=0.0)
    # the first channel at its tangency, 2^(1/3), for 2/3, then the second where z^3/(1 + z^3) is
    # the 1/30 left, at 29^(-1/3); moved together the two needed 2 x 0.8135
    assert plan.budget == pytest.approx(2 ** (1 / 3) + 29 ** (-1 / 3), rel=1e-12, abs=0.0)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == "chc.allocation"]


def _untied() -> tuple[Channel, ...]:
    one = GeometricAdstock(0.0, length=1, normalized=False)
    return Channel(one, Hill(1.0, 3.0), 1.0), Channel(one, Hill(1.01, 3.0), 1.0)


def test_a_search_that_closes_says_so_and_counts_its_boxes(caplog):
    """Concave curves are planned in one box, a goal on them too; the untied example needs more."""
    with caplog.at_level(logging.WARNING, logger="chc.allocation"):
        concave = _plan()
        goal = budget_for(
            CHANNELS,
            MarginalReturnTarget(1.0),
            PERIODS,
            lower=LOWER,
            upper=UPPER,
            history=HISTORY,
        )
        searched = allocate(_untied(), 1.6, 1, lower=np.zeros(2), upper=np.full(2, 1.6))
    assert (concave.stopped, concave.boxes) == ("closed", 1)
    assert (goal.stopped, goal.boxes) == ("closed", 1)
    assert searched.stopped == "closed"
    assert searched.boxes > 1
    assert concave.limit is goal.limit is searched.limit is None
    assert searched.gap == searched.bound - searched.worth <= searched.tolerance
    assert not _warnings(caplog)


def test_a_search_the_cap_stops_says_so_and_logs_its_gap(caplog):
    with caplog.at_level(logging.WARNING, logger="chc.allocation"):
        plan = allocate(_untied(), 1.6, 1, lower=np.zeros(2), upper=np.full(2, 1.6), max_boxes=1)
    assert (plan.stopped, plan.boxes, plan.limit) == ("cap", 1, "max_boxes")
    assert plan.bound - plan.worth > 1e-9 * plan.bound
    assert plan.gap == plan.bound - plan.worth > plan.tolerance
    [record] = _warnings(caplog)
    assert (record.chc_event, record.planner) == ("allocation_cap", "allocate")
    assert (record.boxes, record.worth, record.bound) == (1, plan.worth, plan.bound)
    assert record.limit == "max_boxes"


@pytest.mark.parametrize(
    ("max_boxes", "stopped", "boxes"),
    [(1, "cap", 1), (2, "cap", 1), (3, "cap", 3), (4, "cap", 3), (5, "closed", 5)],
)
def test_no_box_is_planned_past_the_cap(max_boxes, stopped, boxes):
    """A cut plans two boxes, so the search stops where the next cut would pass the cap: at one box
    under a cap of 2, at three under 4, with the plan of a search capped there. The untied example
    closes in five."""
    box = {"lower": np.zeros(2), "upper": np.full(2, 1.6)}
    plan = allocate(_untied(), 1.6, 1, max_boxes=max_boxes, **box)
    assert (plan.stopped, plan.boxes) == (stopped, boxes)
    assert plan.limit == ("max_boxes" if stopped == "cap" else None)
    same = allocate(_untied(), 1.6, 1, max_boxes=boxes, **box)
    assert (plan.worth, plan.bound, plan.boxes) == (same.worth, same.bound, same.boxes)
    np.testing.assert_array_equal(plan.spend, same.spend)


@pytest.mark.parametrize(
    ("rtol", "atol", "boxes"),
    [
        (1e-9, 0.0, 5),
        (0.05, 0.0, 3),
        (0.0, 0.04, 3),
        (0.01, 0.03, 5),
        (0.06, 0.01, 3),
        (0.2, 0.0, 1),
    ],
)
def test_the_search_closes_at_the_larger_of_atol_and_rtol_of_the_first_box_s_bound(
    rtol, atol, boxes
):
    """The untied example's first box bounds it at 0.845 with a gap of 0.140; three boxes leave
    0.036 and five none. Each tolerance stops the search at the first count whose gap it covers:
    0.03 of atol over 0.0084 of rtol does not cover 0.036, which their sum would."""
    box = {"lower": np.zeros(2), "upper": np.full(2, 1.6)}
    first = allocate(_untied(), 1.6, 1, max_boxes=1, **box)
    plan = allocate(_untied(), 1.6, 1, rtol=rtol, atol=atol, **box)
    assert plan.tolerance == max(atol, rtol * first.bound)
    assert not plan.floored
    assert (plan.stopped, plan.boxes, plan.limit) == ("closed", boxes, None)
    assert plan.gap == plan.bound - plan.worth <= plan.tolerance
    assert plan.relative_gap == plan.gap / first.bound


def test_a_tolerance_finer_than_rounding_is_raised_to_it():
    """Asked to close the gap to nothing, the search closes it to 64 epsilons of the first box's
    bound, and says the floor raised its tolerance."""
    box = {"lower": np.zeros(2), "upper": np.full(2, 1.6)}
    first = allocate(_untied(), 1.6, 1, max_boxes=1, **box)
    plan = allocate(_untied(), 1.6, 1, rtol=0.0, atol=0.0, **box)
    assert plan.tolerance == 64 * np.finfo(float).eps * first.bound
    assert plan.floored
    assert (plan.stopped, plan.boxes) == ("closed", 5)


def test_a_float32_search_closes_at_its_own_rounding_and_says_so():
    """Read in float32, the default share 1e-9 of the bound is a hundredth of an epsilon; the floor
    raises it to 64 epsilons of float32, the search closes there, and the plan says the floor
    raised it. On concave curves the one box is judged at the same floor."""
    box = {"lower": np.zeros(2), "upper": np.full(2, 1.6)}
    eps = float(np.finfo(np.float32).eps)
    with enable_x64(False):
        first = allocate(_untied(), 1.6, 1, max_boxes=1, **box)
        plan = allocate(_untied(), 1.6, 1, **box)
        one = GeometricAdstock(0.0, length=1, normalized=False)
        concave = allocate(
            (Channel(one, MichaelisMenten(1.0), 1.0), Channel(one, Tanh(2.0), 1.5)), 1.6, 1, **box
        )
    assert plan.tolerance == 64 * eps * first.bound
    assert plan.floored
    assert (plan.stopped, plan.limit) == ("closed", None)
    assert plan.gap <= plan.tolerance
    assert concave.tolerance == 64 * eps * abs(concave.bound)
    assert (concave.floored, concave.stopped, concave.boxes) == (True, "closed", 1)


@pytest.mark.parametrize(
    ("setting", "match"),
    [
        ({"rtol": -1e-9}, r"^rtol=-1e-09 is not a finite tolerance of at least 0$"),
        ({"rtol": float("nan")}, r"^rtol=nan is not a finite tolerance"),
        ({"atol": float("inf")}, r"^atol=inf is not a finite tolerance"),
        ({"atol": -1.0}, r"^atol=-1.0 is not a finite tolerance"),
        ({"max_boxes": 0}, r"^max_boxes=0 is not a whole number of boxes, at least 1$"),
        ({"max_boxes": -2}, r"^max_boxes=-2 is not a whole number"),
        ({"max_boxes": 2.5}, r"^max_boxes=2.5 is not a whole number"),
    ],
)
def test_it_refuses_a_tolerance_or_a_cap_no_search_meets(setting, match):
    with pytest.raises(ValueError, match=match):
        allocate(_untied(), 1.6, 1, lower=np.zeros(2), upper=np.full(2, 1.6), **setting)


def test_a_cap_may_be_any_whole_number():
    plan = allocate(
        _untied(), 1.6, 1, lower=np.zeros(2), upper=np.full(2, 1.6), max_boxes=np.int64(3)
    )
    assert (plan.stopped, plan.boxes) == ("cap", 3)


def test_the_stopwatch_counts_each_compile_of_its_own_thread_once():
    """JAX reports a compile's phases as they end, each by its duration: a phase inside another
    counts once, within it; another thread's compile and other events count nothing; and nothing
    is heard once it stops."""
    record = jax.monitoring.record_event_duration_secs
    with chc.allocation._Stopwatch() as clock:
        record("/jax/core/compile/jaxpr_to_mlir_module_duration", 0.25)
        record("/jax/core/compile/jaxpr_trace_duration", 1.0)  # began before the one above
        record("/jax/core/compile/backend_compile_duration", 0.5)
        record("/jax/core/dispatch/other_duration", 7.0)
        other = threading.Thread(
            target=record, args=("/jax/core/compile/backend_compile_duration", 7.0)
        )
        other.start()
        other.join()
        compiling, searching = clock.read()
    record("/jax/core/compile/backend_compile_duration", 9.0)
    assert compiling == pytest.approx(1.0, abs=0.05)
    assert searching == 0.0  # what the clock ran is less than the compiles it was told of
    assert clock.read()[0] == compiling


def test_a_plan_records_the_seconds_jax_spent_compiling_it():
    """Curves whose kernel is of a length no other test reads are compiled for on the first plan;
    the same plan again compiles nothing."""
    kernel = GeometricAdstock(0.2, length=37, normalized=True)
    channels = (Channel(kernel, Hill(1.0, 3.0), 1.0), Channel(kernel, Hill(1.01, 3.0), 1.0))
    box = {"lower": np.zeros(2), "upper": np.full(2, 1.6)}
    first = allocate(channels, 1.6, 1, **box)
    again = allocate(channels, 1.6, 1, **box)
    assert first.compile_seconds > 0.0
    assert again.compile_seconds == 0.0
    assert first.search_seconds > 0.0
    assert again.search_seconds > 0.0


def test_a_goal_planned_on_the_envelopes_says_it_was_unsearched_and_logs_its_gap(caplog):
    """The second channel is left inside its chord, at 29^(-1/3), where the envelope from zero
    spend stands above the curve."""
    with caplog.at_level(logging.WARNING, logger="chc.allocation"):
        plan = budget_for(
            _equal_s_curves(3.0, 2), ReturnTarget(0.7), 1, lower=np.zeros(2), upper=np.full(2, 5.0)
        )
    assert (plan.stopped, plan.boxes, plan.limit) == ("unsearched", 1, None)
    [record] = _warnings(caplog)
    assert (record.chc_event, record.planner) == ("allocation_unsearched", "budget_for")
    assert record.gap == plan.bound - plan.worth == plan.gap
    assert record.gap > 0.1
    assert (plan.tolerance, plan.floored) == (1e-9 * plan.bound, False)


_S_CURVE = st.tuples(
    st.sampled_from([Hill, Weibull]),
    st.floats(1.2, 8.0),  # the shape
    st.floats(0.5, 2.0),  # the scale
    st.floats(0.5, 2.0),  # the coefficient
)


@settings(max_examples=12, deadline=None)
@given(
    curves=st.lists(_S_CURVE, min_size=1, max_size=3),
    copies=st.integers(1, 2),
    periods=st.integers(1, 3),
    share=st.floats(0.05, 1.5),
)
def test_the_gap_on_s_curves_is_at_most_one_channels_nonconvexity(curves, copies, periods, share):
    """Before any plan: with kernels of length one, floors at zero and caps past the tangencies, at
    most one channel is left inside its chord on the envelopes, so their plan's gap is at most its
    excess there, and the search, which starts from it, closes the gap. Each curve comes ``copies``
    times, so ties are drawn as often as not."""
    one = GeometricAdstock(0.0, length=1, normalized=False)
    channels = tuple(
        Channel(one, family(scale, shape), coefficient)
        for family, shape, scale, coefficient in curves
        for _ in range(copies)
    )
    touches = np.array([c.curve.tangency() for c in channels])
    budget = periods * share * float(touches.sum())
    lower = np.zeros(len(channels))
    upper = np.full(len(channels), 2.0 * max(budget / periods, float(touches.max())))
    envelope = _on_envelopes(channels, budget, periods, lower, upper, np.zeros((0, len(channels))))
    inside = (envelope.spend > 1e-12 * touches) & (envelope.spend < touches * (1 - 1e-9))
    assert inside.sum() <= 1
    most = max(periods * float(c.coefficient) * c.curve.nonconvexity() for c in channels)
    assert envelope.bound - envelope.worth <= most * (1 + 1e-9) + 1e-12
    plan = allocate(channels, budget, periods, lower=lower, upper=upper)
    assert plan.worth >= envelope.worth - 1e-9 * envelope.bound
    assert plan.bound <= envelope.bound * (1 + 1e-12)
    assert plan.bound - plan.worth <= 1e-9 * envelope.bound


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
