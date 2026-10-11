"""chc.allocation.budget_for_geos: a goal in place of a budget, over every geo's channels.

One geo is :func:`budget_for`'s budget and plan for each goal. On three geos of three channels with
carryover, a cap on a geo and a floor on a channel, each goal is met on the plan's gain as the
cells themselves return it, run over the history, the plan and the tail as one series, and the
budget is the least or the most that meets it; a goal the totals bind is met at the end of the
budgets they allow, worked out by hand.
"""

import logging

import numpy as np
import pytest

import chc.allocation
from chc.allocation import (
    MarginalReturnTarget,
    ReturnOnSpendTarget,
    ReturnTarget,
    Totals,
    allocate,
    allocate_geos,
    budget_for,
    budget_for_geos,
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

PERIODS = 13
KERNELS = (
    GeometricAdstock(0.3, length=6, normalized=True),
    DelayedAdstock(0.6, 1.5, length=8, normalized=True),
    WeibullAdstock(1.2, 3.0, length=10, normalized=True),
)
CURVES = (Tanh(250.0), MichaelisMenten(120.0), Exponential(300.0))
SIZES = np.array([[1100.0, 900.0, 1400.0], [500.0, 700.0, 300.0], [1600.0, 400.0, 1200.0]])
CELLS = tuple(
    tuple(Channel(k, c, size) for k, c, size in zip(KERNELS, CURVES, row, strict=True))
    for row in SIZES
)
HISTORY = np.random.default_rng(3).gamma(4.0, 20.0, size=(30, 3, 3))
LOWER = np.full((3, 3), 10.0)
UPPER = np.array([[160.0, 100.0, 60.0], [120.0, 140.0, 80.0], [200.0, 90.0, 110.0]])
CAP = 1700.0  # geo 0's spend over the plan, below what its boxes reach, 13 * 320
FLOOR = 900.0  # channel 2's, above its floors' 13 * 30
CAPS = Totals(least=np.zeros(3), most=np.array([CAP, np.inf, np.inf]))
FLOORS = Totals(least=np.array([0.0, 0.0, FLOOR]), most=np.full(3, np.inf))
LEAST = PERIODS * LOWER.sum() + FLOOR - PERIODS * LOWER[:, 2].sum()  # channel 2 raised to its floor
MOST = CAP + PERIODS * UPPER[1:].sum()  # geo 0 held at its cap, the others at theirs
ONE = GeometricAdstock(0.0, length=1, normalized=False)
SATURATING = MichaelisMenten(10.0)


def _seek(goal, cells=CELLS, scale=1.0):
    return budget_for_geos(
        cells,
        goal,
        PERIODS,
        lower=scale * LOWER,
        upper=scale * UPPER,
        geo_totals=Totals(least=scale * CAPS.least, most=scale * CAPS.most),
        channel_totals=Totals(least=scale * FLOORS.least, most=scale * FLOORS.most),
        history=scale * HISTORY,
    )


def _planned(budget):
    return allocate_geos(
        CELLS,
        budget,
        PERIODS,
        lower=LOWER,
        upper=UPPER,
        geo_totals=CAPS,
        channel_totals=FLOORS,
        history=HISTORY,
    )


def _gain(spend) -> float:
    """What the plan adds, each cell run over history, plan and tail as one spend series."""

    def worth(rates) -> float:
        total = 0.0
        for g, row in enumerate(CELLS):
            for c, cell in enumerate(row):
                tail = np.zeros(cell.kernel.length - 1)
                series = np.concatenate([HISTORY[:, g, c], np.full(PERIODS, rates[g, c]), tail])
                total += float(np.sum(np.asarray(cell(series))[HISTORY.shape[0] :]))
        return total

    return worth(spend) - worth(np.zeros((3, 3)))


def _within_totals(plan) -> bool:
    return bool(
        PERIODS * plan.spend[0].sum() <= CAP * (1 + 1e-12)
        and PERIODS * plan.spend[:, 2].sum() >= FLOOR * (1 - 1e-12)
    )


@pytest.mark.parametrize(
    "goal",
    [ReturnTarget(10000.0), MarginalReturnTarget(3.5), ReturnOnSpendTarget(4.4)],
    ids=["return", "marginal", "on-spend"],
)
def test_one_geo_meets_each_goal_at_budget_for_s_budget(goal):
    channels = CELLS[0]
    lower, upper = np.array([40.0, 25.0, 15.0]), UPPER[0]
    history = HISTORY[:, 0]
    alone = budget_for(channels, goal, PERIODS, lower=lower, upper=upper, history=history)
    plan = budget_for_geos(
        (channels,), goal, PERIODS, lower=lower[None], upper=upper[None], history=history[:, None]
    )
    assert plan.budget == pytest.approx(alone.budget, rel=1e-12)
    np.testing.assert_allclose(plan.spend[0], alone.spend, rtol=1e-12)
    assert plan.gain == pytest.approx(alone.gain, rel=1e-12)


def test_a_return_target_is_met_by_the_least_budget_the_totals_allow():
    plan = _seek(ReturnTarget(25000.0))
    assert _gain(plan.spend) == pytest.approx(25000.0, rel=1e-10)
    assert plan.gain == pytest.approx(25000.0, rel=1e-12)
    assert _within_totals(plan)
    assert plan.geo_prices[0] > 0.0  # the cap binds, so the goal is met past where it would
    assert _planned(plan.budget * (1 - 1e-9)).gain < 25000.0


def test_a_marginal_target_is_the_plan_at_its_budget_whose_price_it_is():
    plan = _seek(MarginalReturnTarget(3.5))
    assert plan.price == 3.5
    assert _within_totals(plan)
    at = _planned(plan.budget)
    assert at.price == pytest.approx(3.5, rel=1e-12)
    np.testing.assert_allclose(plan.spend, at.spend, rtol=1e-12)
    np.testing.assert_allclose(plan.geo_prices, at.geo_prices, rtol=1e-12, atol=1e-14)
    assert plan.gain == pytest.approx(at.gain, rel=1e-13)


def test_a_target_return_on_spend_is_the_most_budget_that_meets_it():
    plan = _seek(ReturnOnSpendTarget(4.6))
    assert _gain(plan.spend) / plan.budget == pytest.approx(4.6, rel=1e-10)
    assert plan.gain / plan.budget == pytest.approx(4.6, rel=1e-12)
    more = _planned(plan.budget * (1 + 1e-9))
    assert more.gain / more.budget < 4.6


@pytest.mark.parametrize(
    ("goal", "budget"),
    [
        (ReturnTarget(0.0), LEAST),  # the floors and the channel's floor already gain it
        (MarginalReturnTarget(1e6), LEAST),  # no cell returns that much a unit at its floor
        (MarginalReturnTarget(0.77), MOST),  # the price falls only to 0.7734 below the most
        (MarginalReturnTarget(0.0), MOST),
        (MarginalReturnTarget(-1.0), MOST),
        (ReturnOnSpendTarget(1e-3), MOST),  # even the caps return more a unit on average
    ],
)
def test_a_goal_the_totals_bind_is_met_at_the_end_of_the_budgets_they_allow(goal, budget):
    plan = _seek(goal)
    assert plan.budget == pytest.approx(budget, rel=1e-12)
    assert _within_totals(plan)


@pytest.mark.parametrize(
    ("goal", "most"),
    [
        (ReturnTarget(25000.0), 10),
        (ReturnOnSpendTarget(4.6), 10),
        (MarginalReturnTarget(3.5), 0),
        (MarginalReturnTarget(0.77), 0),
    ],
)
def test_a_goal_takes_a_few_plans_and_a_marginal_target_none_at_a_budget(monkeypatch, goal, most):
    """The gain's slope in the budget is the budget's price, so Newton's method needs a few plans
    where bisection on the budget would need forty; a marginal target needs no budget, the plan
    charging each unit it spends the target's return."""
    plans = []
    plan = chc.allocation._plan

    def counted(layout, periods, spend):
        if not isinstance(spend, MarginalReturnTarget):
            plans.append(spend)
        return plan(layout, periods, spend)

    monkeypatch.setattr(chc.allocation, "_plan", counted)
    _seek(goal)
    assert len(plans) <= most
    assert (len(plans) > 0) == (most > 0)


def test_an_s_curve_s_goal_is_searched_and_logs_nothing_once_it_closes(caplog):
    """A gain of 300 is met with the Hill channel inside its chord, at ``100 (3/7)^(1/3)``, 75.4.
    On the envelope from zero spend that plan's bound stood 99 above it, and the gap was logged;
    searched, the bound closes on it."""
    cells = ((Channel(ONE, Hill(100.0, 3.0), 1000.0), Channel(ONE, MichaelisMenten(100.0), 300.0)),)
    box = {"lower": np.zeros((1, 2)), "upper": np.full((1, 2), 400.0)}
    with caplog.at_level(logging.WARNING, logger="chc.allocation"):
        plan = budget_for_geos(cells, ReturnTarget(300.0), 1, **box)
    assert not [r for r in caplog.records if r.name == "chc.allocation"]
    np.testing.assert_allclose(plan.spend, [[100.0 * (3.0 / 7.0) ** (1.0 / 3.0), 0.0]], rtol=1e-12)
    assert plan.gain == pytest.approx(300.0, rel=1e-12, abs=0.0)
    assert (plan.stopped, plan.limit) == ("closed", None)
    assert plan.gap <= plan.tolerance


def test_an_s_curve_s_goal_logs_the_gap_its_cap_leaves_once_however_many_budgets_it_tries(caplog):
    """In one box a search is the envelope's plan, 99 under its bound at 75.4; Brent's method tries
    many budgets on the way, and only the plan it returns is logged."""
    cells = ((Channel(ONE, Hill(100.0, 3.0), 1000.0), Channel(ONE, MichaelisMenten(100.0), 300.0)),)
    box = {"lower": np.zeros((1, 2)), "upper": np.full((1, 2), 400.0)}
    with caplog.at_level(logging.WARNING, logger="chc.allocation"):
        plan = budget_for_geos(cells, ReturnTarget(300.0), 1, **box, max_boxes=1)
    [record] = [r for r in caplog.records if r.name == "chc.allocation"]
    assert (record.chc_event, record.planner) == ("allocation_cap", "budget_for_geos")
    assert (plan.stopped, plan.boxes, plan.limit) == ("cap", 1, "max_boxes")
    assert (record.worth, record.bound) == (plan.worth, plan.bound)
    assert plan.gap == plan.bound - plan.worth == pytest.approx(99.0, abs=1.0)


S_CELLS = (
    (Channel(ONE, Hill(1.0, 3.0), 1.0), Channel(ONE, Hill(1.01, 3.0), 1.0)),
    (Channel(ONE, Hill(0.8, 4.0), 0.7), Channel(ONE, Hill(1.2, 2.5), 1.2)),
)
S_BOX = {"lower": np.zeros((2, 2)), "upper": np.full((2, 2), 2.0)}


def _cells_alone(budget: float):
    """:func:`allocate`'s plan of the grid's four cells, no geo apart."""
    flat = S_CELLS[0] + S_CELLS[1]
    return allocate(flat, budget, 1, lower=np.zeros(4), upper=np.full(4, 2.0))


@pytest.mark.parametrize(
    ("amount", "envelopes"), [(1.2, 2.317499), (2.4, 4.778682)], ids=["1.2", "2.4"]
)
def test_s_curves_over_two_geos_meet_a_return_target_at_the_least_budget_allocate_meets_it(
    amount, envelopes
):
    """Brent's method on the budget over the grid's searched plans: :func:`allocate`'s plan of its
    four cells gains the target there, and a hair short of it does not. Along the plans on the
    envelopes from zero spend the budgets were ``envelopes``, where that plan gains 1.2104 and
    2.4612."""
    plan = budget_for_geos(S_CELLS, ReturnTarget(amount), 1, **S_BOX)
    assert plan.gain == pytest.approx(amount, rel=1e-12, abs=0.0)
    assert (plan.stopped, plan.limit) == ("closed", None)
    assert _cells_alone(plan.budget).gain == pytest.approx(amount, rel=1e-9, abs=0.0)
    assert _cells_alone(plan.budget * (1 - 1e-6)).gain < amount
    assert envelopes - plan.budget > 0.01


def test_s_curves_over_two_geos_take_a_budget_that_rises_with_the_return_target():
    """Each budget's plan is :func:`allocate_geos`' there and gains its target."""
    amounts = (0.6, 1.2, 2.4)
    plans = [budget_for_geos(S_CELLS, ReturnTarget(a), 1, **S_BOX) for a in amounts]
    assert plans[0].budget < plans[1].budget < plans[2].budget
    for amount, plan in zip(amounts, plans, strict=True):
        at = allocate_geos(S_CELLS, plan.budget, 1, **S_BOX)
        np.testing.assert_array_equal(at.spend, plan.spend)
        assert at.gain == pytest.approx(amount, rel=1e-12, abs=0.0)


def test_one_geo_s_marginal_target_on_an_s_curve_with_a_floor_is_budget_for_s():
    """A floor of 1 on the Hill, inside its chord from zero spend, whose slope is 0.53: on the
    envelope the Hill was held at its floor for 0.55 a unit, where its own slope is 0.75. Searched,
    the Hill takes 1.2353, where its slope meets 0.55, as :func:`budget_for` plans it."""
    pair = (Channel(ONE, Hill(1.0, 3.0), 1.0), Channel(ONE, MichaelisMenten(1.0), 0.5))
    lower, upper = np.array([1.0, 0.0]), np.full(2, 3.0)
    goal = MarginalReturnTarget(0.55)
    alone = budget_for(pair, goal, 1, lower=lower, upper=upper)
    plan = budget_for_geos((pair,), goal, 1, lower=lower[None], upper=upper[None])
    assert plan.budget == pytest.approx(alone.budget, rel=1e-12, abs=0.0)
    np.testing.assert_allclose(plan.spend[0], alone.spend, rtol=1e-12, atol=0.0)
    assert plan.budget > 1.2


def test_with_every_geo_s_total_fixed_the_budget_is_theirs():
    budgets = np.array([1500.0, 1600.0, 2100.0])
    fixed = Totals(least=budgets, most=budgets)

    def seek(goal):
        return budget_for_geos(
            CELLS, goal, PERIODS, lower=LOWER, upper=UPPER, geo_totals=fixed, history=HISTORY
        )

    plan = seek(ReturnTarget(100.0))
    assert plan.budget == pytest.approx(budgets.sum(), rel=1e-12)
    assert seek(MarginalReturnTarget(1.0)).budget == pytest.approx(budgets.sum(), rel=1e-12)
    with pytest.raises(ValueError, match="no plan the constraints allow gains"):
        seek(ReturnTarget(2 * plan.gain))


def _falling(goal, curve=SATURATING):
    """Geo 0's channel 0 returns a hundred times what any other cell does, and a cap on geo 0 and
    one on channel 0 leave a budget past 10 only to the other cells, each unit of it taking one
    from the first: past 10 the best plan's gain falls."""
    good = Channel(ONE, curve, 1000.0)
    poor = Channel(ONE, MichaelisMenten(10.0), 10.0)
    caps = Totals(least=np.zeros(2), most=np.array([10.0, np.inf]))
    return budget_for_geos(
        ((good, poor), (poor, poor)),
        goal,
        1,
        lower=np.zeros((2, 2)),
        upper=np.array([[10.0, 10.0], [10.0, 0.0]]),
        geo_totals=caps,
        channel_totals=caps,
    )


@pytest.mark.parametrize(
    ("curve", "budget", "peak", "rel"),
    [
        (SATURATING, 20.0 / 3.0, 500, 1e-12),  # 1000 B / (10 + B) = 400
        # S-shaped, its envelope a chord below 5, where the plan is the cutting planes' within the
        # gap: 1000 B^2 / (25 + B^2) = 400
        (Hill(5.0, 2.0), np.sqrt(10000.0 / 600.0), 800, 1e-8),
    ],
    ids=["concave", "s-curve"],
)
def test_where_totals_make_the_gain_fall_a_return_target_is_met_short_of_its_peak(
    curve, budget, peak, rel
):
    """Up to a budget of 10 the first cell takes it all; at 20 the plan gains 10. A target the
    peak meets is met, one above it refused at the peak."""
    plan = _falling(ReturnTarget(400.0), curve)
    assert plan.budget == pytest.approx(budget, rel=rel)
    np.testing.assert_allclose(plan.spend, [[budget, 0.0], [0.0, 0.0]], rtol=rel, atol=1e-12)
    with pytest.raises(ValueError, match=f"the most a plan gains is {peak}, at a budget of 10"):
        _falling(ReturnTarget(peak + 100.0), curve)


def test_past_the_peak_the_plans_meet_a_negative_marginal_return_and_a_return_on_spend():
    """Past 10 the first cell spends ``20 - B`` and the two others ``B - 10`` each, so the price
    is ``-10000 / (30 - B)^2 + 200 / B^2``, ``-29.475`` at 12."""
    plan = _falling(MarginalReturnTarget(-10000.0 / 18.0**2 + 200.0 / 12.0**2))
    assert plan.budget == pytest.approx(12.0, rel=1e-12)
    np.testing.assert_allclose(plan.spend, [[8.0, 2.0], [2.0, 0.0]], rtol=1e-12)
    plan = _falling(ReturnOnSpendTarget(30.0))
    budget = plan.budget
    assert budget > 10.0
    np.testing.assert_allclose(
        plan.spend, [[20.0 - budget, budget - 10.0], [budget - 10.0, 0.0]], rtol=1e-12
    )
    a, b, c, _ = plan.spend.ravel()
    revenue = 1000.0 * a / (10.0 + a) + 10.0 * b / (10.0 + b) + 10.0 * c / (10.0 + c)
    assert revenue / budget == pytest.approx(30.0, rel=1e-12)


@pytest.mark.parametrize(
    "goal",
    [ReturnTarget(900.0), MarginalReturnTarget(2.0), ReturnOnSpendTarget(2.5)],
    ids=["return", "marginal", "on-spend"],
)
def test_an_s_curve_s_goal_is_met_on_the_true_curve_as_budget_for_meets_it(goal):
    """The plans are searched, their gain not concave in the budget: a return target and a return
    on spend are met where it crosses them."""
    cells = ((Channel(ONE, Hill(100.0, 3.0), 1000.0), Channel(ONE, MichaelisMenten(100.0), 300.0)),)
    box = {"lower": np.zeros((1, 2)), "upper": np.full((1, 2), 400.0)}
    plan = budget_for_geos(cells, goal, 1, **box)
    s = plan.spend[0]
    revenue = 1000.0 * s[0] ** 3 / (100.0**3 + s[0] ** 3) + 300.0 * s[1] / (100.0 + s[1])
    if isinstance(goal, ReturnTarget):
        assert revenue == pytest.approx(900.0, rel=1e-10)
    if isinstance(goal, ReturnOnSpendTarget):
        assert revenue / plan.budget == pytest.approx(2.5, rel=1e-10)
    alone = budget_for(cells[0], goal, 1, lower=np.zeros(2), upper=np.full(2, 400.0))
    assert plan.budget == pytest.approx(alone.budget, rel=1e-9)
    np.testing.assert_allclose(plan.spend[0], alone.spend, rtol=1e-9)


def test_an_s_curve_s_return_on_spend_the_most_budget_meets_is_met_there():
    """Brent's method needs the gain to cross the target between the budgets either side, and none
    is crossed where the most budget already returns 1.53 a unit."""
    cells = ((Channel(ONE, Hill(100.0, 3.0), 1000.0), Channel(ONE, MichaelisMenten(100.0), 300.0)),)
    goal = ReturnOnSpendTarget(1.5)
    plan = budget_for_geos(cells, goal, 1, lower=np.zeros((1, 2)), upper=np.full((1, 2), 400.0))
    assert plan.budget == 800.0
    assert plan.gain / plan.budget > 1.5
    assert budget_for(cells[0], goal, 1, lower=np.zeros(2), upper=np.full(2, 400.0)).budget == 800.0


def test_a_marginal_target_on_a_straight_stretch_bounds_the_best_plan():
    """A linear cell inside its box leaves the cutting planes' plan, which charges each unit it
    spends the target's return: its bound adds the charge back, so no plan spending as much returns
    more. A unit returns 2 on the linear cell, so under the cap the concave one's slope meets 2."""
    cells = ((Channel(ONE, Power(100.0, 1.0), 200.0), Channel(ONE, MichaelisMenten(50.0), 900.0)),)
    plan = budget_for_geos(
        cells,
        MarginalReturnTarget(1.0),
        1,
        lower=np.zeros((1, 2)),
        upper=np.full((1, 2), 150.0),
        geo_totals=Totals(least=np.zeros(1), most=np.array([150.0])),
    )
    best = 200.0 * 50.0 / 100.0 + 900.0 * 100.0 / 150.0
    assert plan.budget == pytest.approx(150.0, rel=1e-12)
    assert plan.price == 1.0
    assert plan.worth <= best * (1 + 1e-12)
    assert 0.0 <= plan.bound - plan.worth <= 1e-9 * plan.bound
    assert plan.bound >= best * (1 - 1e-12)


@pytest.mark.parametrize(
    "goal",
    [ReturnTarget(25000.0), MarginalReturnTarget(3.5), ReturnOnSpendTarget(4.6)],
    ids=["return", "marginal", "on-spend"],
)
def test_a_change_of_currency_moves_the_budget_and_leaves_the_gain(goal):
    k = 1000.0

    def rescaled(cell: Channel) -> Channel:
        return Channel(cell.kernel, type(cell.curve)(k * cell.curve.scale), cell.coefficient)

    moved_goal = goal if isinstance(goal, ReturnTarget) else type(goal)(per_unit=goal.per_unit / k)
    plan = _seek(goal)
    moved = _seek(
        moved_goal, cells=tuple(tuple(rescaled(c) for c in row) for row in CELLS), scale=k
    )
    assert moved.budget == pytest.approx(k * plan.budget, rel=1e-9)
    assert moved.gain == pytest.approx(plan.gain, rel=1e-9)


@pytest.mark.parametrize(
    ("goal", "change", "error", "match"),
    [
        (ReturnTarget(1e7), {}, ValueError, "no plan the constraints allow gains"),
        (ReturnOnSpendTarget(50.0), {}, ValueError, "no budget the constraints allow gains"),
        (ReturnTarget(float("nan")), {}, ValueError, "needs a finite one"),
        (MarginalReturnTarget(float("inf")), {}, ValueError, "needs a finite one"),
        (10000.0, {}, TypeError, "not a Goal"),
        (
            ReturnTarget(100.0),
            {
                "geo_totals": Totals(least=np.zeros(3), most=np.full(3, PERIODS * 120.0)),
                "channel_totals": Totals(
                    least=np.array([PERIODS * 360.0, 0.0, 0.0]), most=np.full(3, np.inf)
                ),
            },
            ValueError,
            "no plan meets the boxes and the totals together",
        ),
        (ReturnTarget(100.0), {"lower": LOWER[:2]}, ValueError, r"it needs \(3, 3\), one a cell"),
    ],
)
def test_it_refuses_a_goal_it_cannot_meet(goal, change, error, match):
    arguments = {
        "lower": LOWER,
        "upper": UPPER,
        "geo_totals": CAPS,
        "channel_totals": FLOORS,
        "history": HISTORY,
    } | change
    with pytest.raises(error, match=match):
        budget_for_geos(CELLS, goal, PERIODS, **arguments)
