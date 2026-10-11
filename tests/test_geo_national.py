"""chc.allocation's national channels: a channel bought once for every geo, its spend a period one
rate and each geo's cell spending its share of it, checked against a second route to each number.

On two geos of two channels the plan stands against a dense lattice of the national spend and a
geo's own cell, on concave and on S-shaped curves, under a cap on a geo that reads the national
spend by its share. Over one geo a national channel is :func:`allocate`'s channel. Shares at the
ratios the grid's own plan holds leave that plan, and any other shares return less. Every price
meets the slope of the cell or the national channel it prices; a goal is met over the national
spend; and what the planners cannot read is refused.
"""

import math

import numpy as np
import pytest

from chc.allocation import (
    MarginalReturnTarget,
    National,
    ReturnOnSpendTarget,
    ReturnTarget,
    Totals,
    allocate,
    allocate_geos,
    budget_for_geos,
)
from chc.response import (
    Channel,
    DelayedAdstock,
    Exponential,
    GeometricAdstock,
    Hill,
    MichaelisMenten,
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
BUDGET = PERIODS * 400.0
ONE = GeometricAdstock(0.0, length=1, normalized=False)
SHARES = np.array([0.2, 0.5, 0.3])  # channel 1's, bought nationally


def _worth(cells, spend, history=HISTORY, periods=PERIODS) -> float:
    """What the plan returns, each cell run over history, plan and tail as one spend series."""
    total = 0.0
    for g, row in enumerate(cells):
        for c, cell in enumerate(row):
            tail = np.zeros(cell.kernel.length - 1)
            series = np.concatenate([history[:, g, c], np.full(periods, spend[g, c]), tail])
            total += float(np.sum(np.asarray(cell(series))[history.shape[0] :]))
    return total


def _boxes(column: int) -> tuple[np.ndarray, np.ndarray]:
    """The grid's boxes with ``column`` a national channel's: nan, no box of its own."""
    lower, upper = LOWER.copy(), UPPER.copy()
    lower[:, column] = upper[:, column] = np.nan
    return lower, upper


def _national(**totals) -> object:
    lower, upper = _boxes(1)
    return allocate_geos(
        CELLS,
        BUDGET,
        PERIODS,
        lower=lower,
        upper=upper,
        national={1: National(SHARES, 30.0, 300.0)},
        history=HISTORY,
        **totals,
    )


CONCAVE = (
    (Channel(ONE, MichaelisMenten(80.0), 1000.0), Channel(ONE, Exponential(150.0), 800.0)),
    (Channel(ONE, Tanh(120.0), 600.0), Channel(ONE, MichaelisMenten(60.0), 400.0)),
)
S_SHAPED = (
    (Channel(ONE, Hill(1.0, 3.0), 1.0), Channel(ONE, Hill(1.01, 3.0), 1.0)),
    (Channel(ONE, Hill(0.8, 4.0), 0.7), Channel(ONE, Hill(1.2, 2.5), 1.2)),
)


@pytest.mark.parametrize(
    ("cells", "budget", "shares", "top", "cap", "binds"),
    [
        (CONCAVE, 500.0, np.array([0.35, 0.65]), 500.0, 200.0, True),
        (S_SHAPED, 3.2, np.array([0.4, 0.6]), 2.0, 0.8, True),
        (S_SHAPED, 3.2, np.array([0.4, 0.6]), 2.0, 1.2, False),
    ],
    ids=["concave", "s-shaped", "s-shaped-free"],
)
def test_on_two_geos_no_lattice_plan_of_the_national_spend_returns_more(
    cells, budget, shares, top, cap, binds
):
    """Channel 1 bought nationally, channel 0 by each geo, and geo 0's total capped: its own cell
    and its share of the national spend. Every plan whose national spend and geo 0's own cell are
    whole steps of a 400th of the budget, geo 1's own cell taking the rest, within the boxes and
    the cap: none returns more than the plan, within its gap, and the best is within the lattice's
    reach of it. Unbound, the S-shaped grid puts all of the budget on the national channel; capped
    at 0.8 the cap binds, and at 1.2 the best plan leaves geo 0 at 0.90, short of it."""
    steps = 400
    plan = allocate_geos(
        cells,
        budget,
        1,
        lower=np.array([[0.0, np.nan], [0.0, np.nan]]),
        upper=np.array([[top, np.nan], [top, np.nan]]),
        national={1: National(shares, 0.0, budget)},
        geo_totals=Totals(least=np.zeros(2), most=np.array([cap, np.inf])),
    )
    national = plan.spend[:, 1].sum()
    np.testing.assert_allclose(plan.spend[:, 1], shares * national, rtol=1e-15, atol=0.0)
    assert plan.spend.sum() == pytest.approx(budget, rel=1e-12, abs=0.0)
    assert plan.spend[0].sum() <= cap * (1 + 1e-12)
    assert np.all(plan.spend[:, 0] >= 0.0)
    assert np.all(plan.spend[:, 0] <= top)
    assert plan.worth == pytest.approx(
        _worth(cells, plan.spend, np.zeros((0, 2, 2)), 1), rel=1e-12, abs=0.0
    )
    assert (plan.stopped, plan.limit) == ("closed", None)
    unit = budget / steps
    lattice = np.arange(steps + 1) * unit
    own, tied = np.meshgrid(np.arange(steps + 1), np.arange(steps + 1), indexing="ij")
    rest = steps - own - tied
    keep = (rest >= 0) & (lattice[own] <= top) & (rest * unit <= top)
    keep &= lattice[own] + shares[0] * lattice[tied] <= cap
    tables = [
        np.asarray(cells[0][0](lattice)),
        np.asarray(cells[0][1](shares[0] * lattice)),
        np.asarray(cells[1][0](lattice)),
        np.asarray(cells[1][1](shares[1] * lattice)),
    ]
    revenue = (
        tables[0][own[keep]] + tables[1][tied[keep]] + tables[2][rest[keep]] + tables[3][tied[keep]]
    )
    best = float(revenue.max())
    assert best <= plan.worth + plan.gap + 1e-12 * plan.worth
    assert plan.worth - best <= 1e-4 * plan.worth
    if binds:
        assert plan.spend[0].sum() == pytest.approx(cap, rel=1e-9, abs=0.0)
        assert plan.geo_prices[0] > 0.0
    else:
        assert plan.spend[0].sum() < 0.9 * cap


@pytest.mark.parametrize("column", [0, 1, 2])
def test_a_national_channel_over_one_geo_is_allocate_s_channel(column):
    """With one geo a national channel's share is 1, and its spend is the cell's: the plan, its
    worth, its price and its idle return are :func:`allocate`'s with the national box as the
    channel's."""
    lower, upper = LOWER[:1].copy(), UPPER[:1].copy()
    lower[0, column] = upper[0, column] = np.nan
    plan = allocate_geos(
        CELLS[:1],
        BUDGET / 3,
        PERIODS,
        lower=lower,
        upper=upper,
        national={column: National([1.0], LOWER[0, column], UPPER[0, column])},
        history=HISTORY[:, :1],
    )
    alone = allocate(
        CELLS[0], BUDGET / 3, PERIODS, lower=LOWER[0], upper=UPPER[0], history=HISTORY[:, 0]
    )
    np.testing.assert_allclose(plan.spend[0], alone.spend, rtol=1e-12)
    assert plan.worth == pytest.approx(alone.worth, rel=1e-13, abs=0.0)
    assert plan.price == pytest.approx(alone.price, rel=1e-12, abs=0.0)
    assert plan.bound == plan.worth
    assert plan.idle == pytest.approx(alone.idle, rel=1e-13, abs=0.0)


def test_a_national_s_curve_over_one_geo_is_searched_to_allocate_s_plan():
    """Two Hill curves of slope 3 at scales 1 and 1.01 and a budget of 1.6, the first bought
    nationally: the search plans all of it on the first, for 512/637, as :func:`allocate` does."""
    pair = S_SHAPED[0]
    plan = allocate_geos(
        (pair,),
        1.6,
        1,
        lower=np.array([[np.nan, 0.0]]),
        upper=np.array([[np.nan, 1.6]]),
        national={0: National([1.0], 0.0, 1.6)},
    )
    alone = allocate(pair, 1.6, 1, lower=np.zeros(2), upper=np.full(2, 1.6))
    np.testing.assert_allclose(plan.spend[0], alone.spend, rtol=0.0, atol=1e-12)
    assert plan.worth == pytest.approx(512.0 / 637.0, rel=1e-12, abs=0.0)
    assert alone.worth - plan.worth <= plan.gap + 1e-15
    assert (plan.stopped, plan.limit) == ("closed", None)
    assert plan.boxes > 1


def test_shares_at_the_grid_s_own_ratios_leave_its_plan_and_any_others_return_less():
    """Geos 0 and 1 at a budget of 250 a period, planned cell by cell, hold channel 1 inside both
    boxes, at 48.98 and 27.13. Bought nationally at those ratios, within a national box that keeps
    each cell in its own box, the plan is the cell-by-cell one: it meets the ratios, and every
    national plan is one of the grid's. At even shares the national plan returns less."""
    budget, lower, upper, history = PERIODS * 250.0, LOWER[:2], UPPER[:2], HISTORY[:, :2]
    free = allocate_geos(CELLS[:2], budget, PERIODS, lower=lower, upper=upper, history=history)
    assert np.all((lower[:, 1] < free.spend[:, 1]) & (free.spend[:, 1] < upper[:, 1]))
    shares = free.spend[:, 1] / free.spend[:, 1].sum()
    tied_lower, tied_upper = lower.copy(), upper.copy()
    tied_lower[:, 1] = tied_upper[:, 1] = np.nan

    def planned(shares):
        box = National(shares, np.max(lower[:, 1] / shares), np.min(upper[:, 1] / shares))
        return allocate_geos(
            CELLS[:2],
            budget,
            PERIODS,
            lower=tied_lower,
            upper=tied_upper,
            national={1: box},
            history=history,
        )

    plan = planned(shares)
    np.testing.assert_allclose(plan.spend, free.spend, rtol=1e-11)
    assert plan.worth == pytest.approx(free.worth, rel=1e-13, abs=0.0)
    assert plan.price == pytest.approx(free.price, rel=1e-11, abs=0.0)
    assert plan.bound == plan.worth
    even = planned(np.full(2, 0.5))
    assert even.bound == even.worth
    assert even.worth < free.worth


def test_every_cell_and_the_national_channel_inside_their_boxes_return_their_prices():
    """A geo's own cell returns, a currency unit, the budget's price plus its geo's and its
    channel's. The national channel, its spend in every geo at once, returns the budget's price
    plus its channel's and each geo's weighted by the geo's share. A cap on geo 0 and a floor on
    channel 2 bind, the cap reading channel 1 by geo 0's share."""
    free = _national()
    geo = PERIODS * free.spend.sum(axis=1)
    channel = PERIODS * free.spend.sum(axis=0)
    caps = Totals(least=np.zeros(3), most=np.array([0.85 * geo[0], np.inf, np.inf]))
    floors = Totals(least=np.array([0.0, 0.0, 1.15 * channel[2]]), most=np.full(3, np.inf))
    plan = _national(geo_totals=caps, channel_totals=floors)
    assert PERIODS * plan.spend.sum() == pytest.approx(BUDGET, rel=1e-13, abs=0.0)
    assert PERIODS * plan.spend[0].sum() == pytest.approx(caps.most[0], rel=1e-12, abs=0.0)
    assert PERIODS * plan.spend[:, 2].sum() == pytest.approx(floors.least[2], rel=1e-12, abs=0.0)
    assert plan.worth == pytest.approx(_worth(CELLS, plan.spend), rel=1e-12, abs=0.0)
    assert plan.bound == plan.worth
    assert (plan.stopped, plan.boxes, plan.limit) == ("closed", 1, None)
    assert plan.geo_prices[0] > 0.0
    assert plan.channel_prices[2] < 0.0
    national = plan.spend[:, 1].sum()
    assert 30.0 < national < 300.0
    step = 1e-4 * national
    up, down = plan.spend.copy(), plan.spend.copy()
    up[:, 1] += SHARES * step
    down[:, 1] -= SHARES * step
    slope = (_worth(CELLS, up) - _worth(CELLS, down)) / (2 * step) / PERIODS
    prices = plan.price + SHARES @ plan.geo_prices + plan.channel_prices[1]
    assert slope == pytest.approx(prices, rel=1e-6, abs=0.0)
    inside = 0
    for g in range(3):
        for c in (0, 2):
            step = 1e-4 * plan.spend[g, c]
            up, down = plan.spend.copy(), plan.spend.copy()
            up[g, c] += step
            down[g, c] -= step
            slope = (_worth(CELLS, up) - _worth(CELLS, down)) / (2 * step) / PERIODS
            prices = plan.price + plan.geo_prices[g] + plan.channel_prices[c]
            if LOWER[g, c] < plan.spend[g, c] < UPPER[g, c]:
                inside += 1
                assert slope == pytest.approx(prices, rel=1e-6, abs=0.0)
            elif plan.spend[g, c] == LOWER[g, c]:
                assert slope <= prices * (1 + 1e-6)
            else:
                assert slope >= prices * (1 - 1e-6)
    assert inside >= 4


def test_every_geo_s_total_fixed_passes_the_budget_s_price_to_the_geos():
    """Each geo's total fixed at what the free plan spends there fixes the budget with them, since
    the national channel's shares sum to 1: the plan is the free one, the budget's price 0 and
    each geo's the free plan's budget price."""
    free = _national()
    spent = PERIODS * free.spend.sum(axis=1)
    plan = _national(geo_totals=Totals(least=spent, most=spent))
    np.testing.assert_allclose(plan.spend, free.spend, rtol=1e-10)
    assert plan.price == 0.0
    np.testing.assert_allclose(plan.geo_prices, free.price, rtol=1e-8)


def test_a_cap_a_little_above_what_a_geo_spends_does_not_bind():
    """Geo 1 spends its own cells and half the national spend. A cap 5% above that does not bind:
    the plan is the free one, its bound its worth and the cap's price 0. Read with the national
    spend whole, geo 1 would break the cap."""
    free = _national()
    most = np.array([np.inf, 1.05 * PERIODS * free.spend[1].sum(), np.inf])
    assert most[1] < PERIODS * (free.spend[1, [0, 2]].sum() + free.spend[:, 1].sum())
    plan = _national(geo_totals=Totals(least=np.zeros(3), most=most))
    np.testing.assert_allclose(plan.spend, free.spend, rtol=1e-12)
    np.testing.assert_array_equal(plan.geo_prices, 0.0)
    assert plan.bound == plan.worth


@pytest.mark.parametrize("short", [1e-9, 1e-6, 1e-3])
def test_a_budget_a_hair_below_the_most_a_cap_allows_is_planned_exactly(short):
    """A cap on geo 0 of 1700, and a budget a hair below the most it allows: the national channel
    at its cap of 300, geo 0's own cells at the 1700 / 13 - 60 its share leaves them, every other
    cell at its cap but the one the last hair takes off it. The plan is exact, and geo 0 spends
    its cap."""
    cap = 1700.0
    most = cap + PERIODS * (UPPER[1:, [0, 2]].sum() + 0.8 * 300.0)
    plan = allocate_geos(
        CELLS,
        most - short,
        PERIODS,
        **_arguments(),
        geo_totals=Totals(least=np.zeros(3), most=np.array([cap, np.inf, np.inf])),
    )
    assert plan.bound == plan.worth
    assert (plan.stopped, plan.boxes) == ("closed", 1)
    assert PERIODS * plan.spend[0].sum() == pytest.approx(cap, rel=1e-13, abs=0.0)
    assert plan.geo_prices[0] > 0.0
    assert plan.spend[:, 1].sum() == pytest.approx(300.0, rel=1e-12, abs=0.0)


def test_a_national_channel_s_box_bounds_its_national_spend():
    """A national box below what the free plan spends on the channel holds it at its cap, and
    the cells read their shares of the cap."""
    free = _national()
    cap = 0.8 * free.spend[:, 1].sum()
    lower, upper = _boxes(1)
    plan = allocate_geos(
        CELLS,
        BUDGET,
        PERIODS,
        lower=lower,
        upper=upper,
        national={1: National(SHARES, 30.0, cap)},
        history=HISTORY,
    )
    np.testing.assert_allclose(plan.spend[:, 1], SHARES * cap, rtol=1e-15, atol=0.0)
    assert plan.worth < free.worth


def test_a_channel_total_on_a_national_channel_bounds_its_national_spend():
    """A channel's total on a national channel is its national spend over the plan: held at 0.8 of
    the free plan's, the plan is the one its national box holds there, and the total is priced."""
    free = _national()
    cap = 0.8 * free.spend[:, 1].sum()
    boxed = allocate_geos(
        CELLS, BUDGET, PERIODS, **_arguments(national={1: National(SHARES, 30.0, cap)})
    )
    most = np.array([np.inf, PERIODS * cap, np.inf])
    plan = allocate_geos(
        CELLS, BUDGET, PERIODS, **_arguments(), channel_totals=Totals(least=np.zeros(3), most=most)
    )
    np.testing.assert_allclose(plan.spend, boxed.spend, rtol=1e-11)
    assert plan.worth == pytest.approx(boxed.worth, rel=1e-13, abs=0.0)
    assert plan.channel_prices[1] > 0.0


def test_the_budgets_a_goal_runs_over_read_a_geo_s_total_by_its_share():
    """A floor of 60 a period on geo 0's total and a cap of 200, worked out by hand. The least
    budget holds the national channel at its floor of 30, of which geo 0 spends 6, and raises geo
    0's own cells to the other 54: a unit of geo 0's own cells costs a unit of budget, and one of
    its share costs five. The most budget holds the national channel at its cap of 300, of which
    geo 0 spends 60, and leaves geo 0's own cells the other 140. A return of nothing is met at the
    least budget, and a return on spend of a thousandth at the most."""
    totals = Totals(
        least=np.array([PERIODS * 60.0, 0.0, 0.0]), most=np.array([PERIODS * 200.0, np.inf, np.inf])
    )
    own = [0, 2]
    least = PERIODS * (60.0 + LOWER[1:, own].sum() + 0.8 * 30.0)
    most = PERIODS * (200.0 + UPPER[1:, own].sum() + 0.8 * 300.0)
    start = budget_for_geos(CELLS, ReturnTarget(0.0), PERIODS, **_arguments(), geo_totals=totals)
    assert start.budget == pytest.approx(least, rel=1e-12, abs=0.0)
    assert PERIODS * start.spend[0].sum() == pytest.approx(PERIODS * 60.0, rel=1e-12, abs=0.0)
    end = budget_for_geos(
        CELLS, ReturnOnSpendTarget(1e-3), PERIODS, **_arguments(), geo_totals=totals
    )
    assert end.budget == pytest.approx(most, rel=1e-12, abs=0.0)
    np.testing.assert_allclose(end.spend[:, 1], SHARES * 300.0, rtol=1e-12)


def test_a_goal_is_met_over_the_national_spend():
    """A return target is met at the budget whose plan gains it, the plan :func:`allocate_geos`
    makes there; at a marginal target the national channel returns the target a currency unit."""
    lower, upper = _boxes(1)
    grid = {
        "lower": lower,
        "upper": upper,
        "national": {1: National(SHARES, 30.0, 300.0)},
        "history": HISTORY,
    }
    met = budget_for_geos(CELLS, ReturnTarget(25000.0), PERIODS, **grid)
    assert met.gain == pytest.approx(25000.0, rel=1e-10, abs=0.0)
    same = allocate_geos(CELLS, met.budget, PERIODS, **grid)
    np.testing.assert_allclose(met.spend, same.spend, rtol=1e-10)
    marginal = budget_for_geos(CELLS, MarginalReturnTarget(3.5), PERIODS, **grid)
    assert marginal.price == 3.5
    national = marginal.spend[:, 1].sum()
    assert 30.0 < national < 300.0
    step = 1e-4 * national
    up, down = marginal.spend.copy(), marginal.spend.copy()
    up[:, 1] += SHARES * step
    down[:, 1] -= SHARES * step
    slope = (_worth(CELLS, up) - _worth(CELLS, down)) / (2 * step) / PERIODS
    assert slope == pytest.approx(3.5, rel=1e-6, abs=0.0)
    assert marginal.budget == pytest.approx(PERIODS * marginal.spend.sum(), rel=1e-13, abs=0.0)


def test_no_national_channel_is_the_grid_planned_cell_by_cell():
    plain = allocate_geos(CELLS, BUDGET, PERIODS, lower=LOWER, upper=UPPER, history=HISTORY)
    for none in (None, {}):
        same = allocate_geos(
            CELLS, BUDGET, PERIODS, lower=LOWER, upper=UPPER, history=HISTORY, national=none
        )
        np.testing.assert_array_equal(same.spend, plain.spend)
        assert (same.worth, same.bound, same.price) == (plain.worth, plain.bound, plain.price)


def test_shares_that_sum_to_1_only_to_rounding_are_read_as_given():
    """Shares a population divides into sum to 1 only to rounding, and ten of 0.1 to 1 less an
    ulp; each is held as given."""
    population = np.random.default_rng(0).lognormal(11.0, 1.0, size=7)
    for shares in (population / population.sum(), np.full(10, 0.1)):
        held = National(shares, 0.0, 1.0)
        np.testing.assert_array_equal(held.shares, shares)
        assert abs(math.fsum(held.shares) - 1.0) <= shares.size * np.finfo(float).eps


def test_a_national_channel_holds_a_copy_the_caller_cannot_change():
    shares = np.array([0.25, 0.75])
    held = National(shares, 0.0, 1.0)
    shares[0] = 0.5
    np.testing.assert_array_equal(held.shares, [0.25, 0.75])
    with pytest.raises(ValueError, match="read-only"):
        held.shares[0] = 0.5


@pytest.mark.parametrize(
    ("shares", "lower", "upper", "match"),
    [
        ([0.3, 0.6], 0.0, 1.0, "sum to 0.8999999999999999, not 1: divide them by their sum"),
        ([0.5, 0.5, 1e-3], 0.0, 1.0, "sum to 1.001, not 1"),
        ([1.2, -0.2], 0.0, 1.0, "positive"),
        ([0.5, 0.0, 0.5], 0.0, 1.0, "positive"),
        ([0.5, np.nan], 0.0, 1.0, "positive"),
        ([], 0.0, 1.0, "one share a geo"),
        ([[0.5, 0.5]], 0.0, 1.0, "one share a geo"),
        ([0.5, 0.5], 2.0, 1.0, "lower <= upper"),
        ([0.5, 0.5], -1.0, 1.0, "lower <= upper"),
        ([0.5, 0.5], 0.0, np.inf, "lower <= upper"),
    ],
)
def test_a_national_channel_refuses_shares_and_a_box_it_cannot_read(shares, lower, upper, match):
    with pytest.raises(ValueError, match=match):
        National(shares, lower, upper)


def _arguments(**change) -> dict:
    lower, upper = _boxes(1)
    return {
        "lower": lower,
        "upper": upper,
        "national": {1: National(SHARES, 30.0, 300.0)},
        "history": HISTORY,
    } | change


def _open_per_geo_cell() -> np.ndarray:
    lower = _boxes(1)[0]
    lower[2, 0] = np.nan
    return lower


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        (
            {"lower": LOWER},
            ValueError,
            "channel 1 is national, and lower and upper give its cells a box of their own",
        ),
        (
            {"upper": np.where(np.isnan(_boxes(1)[1]), 50.0, UPPER)},
            ValueError,
            "channel 1 is national, and lower and upper give its cells a box of their own",
        ),
        (
            {"national": {1: National(SHARES, 30.0, 300.0), 2: National(SHARES, 0.0, 9.0)}},
            ValueError,
            "channel 2 is national, and lower and upper give its cells a box of their own",
        ),
        ({"national": {1: National([0.5, 0.5], 30.0, 300.0)}}, ValueError, "2 shares; 3 geos"),
        ({"national": {3: National(SHARES, 30.0, 300.0)}}, ValueError, "no channel 3"),
        ({"national": {-1: National(SHARES, 30.0, 300.0)}}, ValueError, "no channel -1"),
        ({"national": {True: National(SHARES, 30.0, 300.0)}}, TypeError, "not a channel's column"),
        ({"national": {1: SHARES}}, TypeError, "not a National"),
        ({"national": [National(SHARES, 30.0, 300.0)]}, TypeError, "not a mapping"),
        ({"lower": _open_per_geo_cell()}, ValueError, "it needs one finite rate a channel"),
        (
            {
                "geo_totals": Totals(
                    least=np.array([PERIODS * 300.0, 0.0, 0.0]), most=np.full(3, np.inf)
                )
            },
            ValueError,
            r"geo 0's total .* is outside what its cells' boxes spend over 13 periods, "
            r"\[338.0, 3640.0\]",
        ),
    ],
)
def test_it_refuses_national_channels_it_cannot_plan(change, error, match):
    """A channel both national and given a box of its own in a geo, wholly or in one bound, is
    refused by both planners, as are a national channel the grid has no column for, shares that
    are not one a geo, and a geo's own cell left with no box. Geo 0 spends at most 280 a period,
    60 of it its share of the national channel's cap, not the 300 of the whole."""
    with pytest.raises(error, match=match):
        allocate_geos(CELLS, BUDGET, PERIODS, **_arguments(**change))
    with pytest.raises(error, match=match):
        budget_for_geos(CELLS, ReturnTarget(1.0), PERIODS, **_arguments(**change))


def test_a_budget_below_the_national_floor_is_refused():
    """The national channel's floor of 30 a period counts whole in what the boxes spend."""
    with pytest.raises(ValueError, match=r"outside what the box spends over 13 periods, \[1170"):
        allocate_geos(CELLS, PERIODS * 89.0, PERIODS, **_arguments())
