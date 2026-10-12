"""chc.allocation.allocate_geos: one budget over every geo's channels, checked against a second
route to each number.

One geo is :func:`allocate`'s plan, and fixed geo budgets are each geo's own; the best plan stands
against an exhaustive lattice on two geos of two channels and SciPy's general solver on three of
three; each price against the cells' own slopes; a total a hair from binding is the free plan or
the plan that holds it fixed; a plan made in two steps, the geos first, never returns more.
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

import chc.allocation
from chc.allocation import (
    MarginalReturnTarget,
    Totals,
    _Group,
    _on_envelopes,
    _priced,
    _worths,
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
    Power,
    Ricker,
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


def _worth(cells, spend, history=HISTORY, periods=PERIODS) -> float:
    """What the plan returns, each cell run over history, plan and tail as one spend series."""
    total = 0.0
    for g, row in enumerate(cells):
        for c, cell in enumerate(row):
            tail = np.zeros(cell.kernel.length - 1)
            series = np.concatenate([history[:, g, c], np.full(periods, spend[g, c]), tail])
            total += float(np.sum(np.asarray(cell(series))[history.shape[0] :]))
    return total


def _free():
    return allocate_geos(CELLS, BUDGET, PERIODS, lower=LOWER, upper=UPPER, history=HISTORY)


def _totals():
    """A cap on geo 0 and a floor on channel 2 that the free plan breaks, so both bind."""
    free = _free()
    geo = PERIODS * free.spend.sum(axis=1)
    channel = PERIODS * free.spend.sum(axis=0)
    caps = Totals(least=np.zeros(3), most=np.array([0.85 * geo[0], np.inf, np.inf]))
    floors = Totals(least=np.array([0.0, 0.0, 1.15 * channel[2]]), most=np.full(3, np.inf))
    return caps, floors


def _tied():
    caps, floors = _totals()
    return allocate_geos(
        CELLS,
        BUDGET,
        PERIODS,
        lower=LOWER,
        upper=UPPER,
        geo_totals=caps,
        channel_totals=floors,
        history=HISTORY,
    )


def test_one_geo_is_the_allocation_of_its_channels():
    plan = allocate_geos(
        CELLS[:1], BUDGET / 3, PERIODS, lower=LOWER[:1], upper=UPPER[:1], history=HISTORY[:, :1]
    )
    alone = allocate(
        CELLS[0], BUDGET / 3, PERIODS, lower=LOWER[0], upper=UPPER[0], history=HISTORY[:, 0]
    )
    np.testing.assert_allclose(plan.spend[0], alone.spend, rtol=1e-12)
    assert plan.worth == pytest.approx(alone.worth, rel=1e-13)
    assert plan.price == pytest.approx(alone.price, rel=1e-12)
    assert plan.bound == plan.worth
    assert plan.idle == pytest.approx(alone.idle, rel=1e-13)


def test_a_plan_meets_the_budget_the_boxes_and_the_totals_and_its_worth_is_the_cells_return():
    caps, floors = _totals()
    plan = _tied()
    assert PERIODS * plan.spend.sum() == pytest.approx(BUDGET, rel=1e-13)
    assert np.all(plan.spend >= LOWER)
    assert np.all(plan.spend <= UPPER)
    assert PERIODS * plan.spend[0].sum() == pytest.approx(caps.most[0], rel=1e-12)
    assert PERIODS * plan.spend[:, 2].sum() == pytest.approx(floors.least[2], rel=1e-12)
    assert plan.worth == pytest.approx(_worth(CELLS, plan.spend), rel=1e-12)
    assert plan.bound == plan.worth
    assert plan.worth < _free().worth


def test_every_cell_inside_its_box_returns_its_three_prices():
    """A cell's slope a currency unit is the budget's price plus its geo's and its channel's; the
    cap that binds is worth more room, the floor that binds less spend."""
    plan = _tied()
    assert plan.geo_prices[0] > 0.0
    assert plan.channel_prices[2] < 0.0
    np.testing.assert_array_equal(plan.geo_prices[1:], 0.0)
    np.testing.assert_array_equal(plan.channel_prices[:2], 0.0)
    inside = 0
    for g in range(3):
        for c in range(3):
            step = 1e-4 * plan.spend[g, c]
            up, down = plan.spend.copy(), plan.spend.copy()
            up[g, c] += step
            down[g, c] -= step
            slope = (_worth(CELLS, up) - _worth(CELLS, down)) / (2 * step) / PERIODS
            prices = plan.price + plan.geo_prices[g] + plan.channel_prices[c]
            if LOWER[g, c] < plan.spend[g, c] < UPPER[g, c]:
                inside += 1
                assert slope == pytest.approx(prices, rel=1e-6)
            elif plan.spend[g, c] == LOWER[g, c]:
                assert slope <= prices * (1 + 1e-6)
            else:
                assert slope >= prices * (1 - 1e-6)
    assert inside >= 6  # the case exercises the interior, not only the corners


def test_no_plan_meeting_the_constraints_returns_more():
    caps, floors = _totals()
    plan = _tied()

    def loss(flat):
        return -_worth(CELLS, flat.reshape(3, 3)) / BUDGET

    def rows(flat):
        spend = PERIODS * flat.reshape(3, 3)
        return np.array([caps.most[0] - spend[0].sum(), spend[:, 2].sum() - floors.least[2]])

    general = minimize(
        loss,
        np.clip(np.full(9, BUDGET / PERIODS / 9), LOWER.ravel(), UPPER.ravel()),
        method="SLSQP",
        bounds=list(zip(LOWER.ravel(), UPPER.ravel(), strict=True)),
        constraints=[
            {"type": "eq", "fun": lambda s: PERIODS * s.sum() / BUDGET - 1.0},
            {"type": "ineq", "fun": lambda s: rows(s) / BUDGET},
        ],
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    assert general.success, general.message
    assert -general.fun * BUDGET <= plan.worth * (1 + 1e-10)
    assert -general.fun * BUDGET == pytest.approx(plan.worth, rel=1e-8)


def test_on_two_geos_of_two_channels_no_lattice_plan_returns_more():
    """Every plan whose four rates are whole steps of a 400th of the budget, the totals checked
    point by point: none beats the plan, and the best is within the lattice's reach of it."""
    cells = (
        (Channel(ONE, MichaelisMenten(80.0), 1000.0), Channel(ONE, Exponential(150.0), 800.0)),
        (Channel(ONE, Tanh(120.0), 600.0), Channel(ONE, MichaelisMenten(60.0), 400.0)),
    )
    budget, steps = 500.0, 400
    caps = Totals(least=np.zeros(2), most=np.array([240.0, np.inf]))
    floors = Totals(least=np.array([0.0, 250.0]), most=np.full(2, np.inf))
    plan = allocate_geos(
        cells,
        budget,
        1,
        lower=np.zeros((2, 2)),
        upper=np.full((2, 2), budget),
        geo_totals=caps,
        channel_totals=floors,
    )
    unit = budget / steps
    table = [[np.asarray(cell(np.arange(steps + 1) * unit)) for cell in row] for row in cells]
    b, c = np.meshgrid(np.arange(steps + 1), np.arange(steps + 1), indexing="ij")
    best = -np.inf
    # a plane of the lattice at a time, its first rate fixed: the whole is 64 million plans
    for a in range(steps + 1):
        d = steps - a - b - c
        keep = (
            (d >= 0)
            & (a * unit + b * unit <= caps.most[0])
            & (b * unit + d * unit >= floors.least[1])
        )
        if keep.any():
            revenue = (
                table[0][0][a] + table[0][1][b[keep]] + table[1][0][c[keep]] + table[1][1][d[keep]]
            )
            best = max(best, float(revenue.max()))
    assert best <= plan.worth * (1 + 1e-12)
    assert plan.worth - best <= 1e-4 * plan.worth
    assert plan.spend[0].sum() == pytest.approx(240.0, rel=1e-12)
    assert plan.spend[:, 1].sum() == pytest.approx(250.0, rel=1e-12)


def test_a_plan_made_geo_by_geo_never_returns_more():
    """The budget split over the geos first, then each geo's share planned on its own: in
    proportion to each geo's return a currency unit at an even spread, evenly, and in proportion to
    its boxes. Each is one plan of the grid; the first, which reads an average where the margin
    decides, leaves more than a percent of the joint plan's gain behind."""
    plan = _free()
    even = np.clip(np.full((3, 3), BUDGET / PERIODS / 9), LOWER, UPPER)
    average = np.array(
        [
            _worth(CELLS[g : g + 1], even[g : g + 1], HISTORY[:, g : g + 1])
            - _worth(CELLS[g : g + 1], np.zeros((1, 3)), HISTORY[:, g : g + 1])
            for g in range(3)
        ]
    ) / (PERIODS * even.sum(axis=1))
    splits = {
        "average": average / average.sum(),
        "even": np.full(3, 1.0 / 3.0),
        "boxes": UPPER.sum(axis=1) / UPPER.sum(),
    }
    for name, share in splits.items():
        budgets = np.clip(BUDGET * share, PERIODS * LOWER.sum(axis=1), PERIODS * UPPER.sum(axis=1))
        budgets *= BUDGET / budgets.sum()
        steps = [
            allocate(
                CELLS[g],
                budgets[g],
                PERIODS,
                lower=LOWER[g],
                upper=UPPER[g],
                history=HISTORY[:, g],
            )
            for g in range(3)
        ]
        worth = sum(step.worth for step in steps)
        assert worth <= plan.worth * (1 + 1e-12), name
        if name == "average":
            assert plan.worth - worth > 0.01 * plan.gain


def test_fixed_geo_budgets_plan_each_geo_on_its_own():
    """With every geo's total fixed the budget is fixed with them: its price is 0 and each geo's
    is the price its own plan would carry."""
    budgets = np.array([0.3, 0.3, 0.4]) * BUDGET
    plan = allocate_geos(
        CELLS,
        BUDGET,
        PERIODS,
        lower=LOWER,
        upper=UPPER,
        geo_totals=Totals(least=budgets, most=budgets),
        history=HISTORY,
    )
    assert plan.price == 0.0
    for g in range(3):
        alone = allocate(
            CELLS[g], budgets[g], PERIODS, lower=LOWER[g], upper=UPPER[g], history=HISTORY[:, g]
        )
        np.testing.assert_allclose(plan.spend[g], alone.spend, rtol=1e-11)
        assert plan.geo_prices[g] == pytest.approx(alone.price, rel=1e-10)


@pytest.mark.parametrize(
    ("kind", "bound", "delta"),
    [
        ("geo", "most", -1e-6),
        ("geo", "most", 1e-6),
        ("channel", "least", -1e-6),
        ("channel", "least", 1e-6),
    ],
)
def test_a_total_a_hair_from_binding_is_planned_exactly(kind, bound, delta):
    """A cap on geo 0 or a floor on channel 0 a millionth either side of what the free plan spends
    there. The cutting planes' duals name the totals that bind only to within their gap, and so may
    name such a total wrongly; the plan is still exact. A total that does not bind leaves the free
    plan, its price 0; one that binds is met, its price on its side, the plan the one that holds the
    total fixed."""
    free = _free()
    axis = 1 if kind == "geo" else 0
    level = PERIODS * free.spend.sum(axis=axis)[0] * (1 + delta)

    def planned(floor, cap):
        least, most = np.zeros(3), np.full(3, np.inf)
        least[0], most[0] = floor, cap
        totals = {f"{kind}_totals": Totals(least=least, most=most)}
        return allocate_geos(
            CELLS, BUDGET, PERIODS, lower=LOWER, upper=UPPER, history=HISTORY, **totals
        )

    plan = planned(level, np.inf) if bound == "least" else planned(0.0, level)
    prices = plan.geo_prices if kind == "geo" else plan.channel_prices
    assert plan.bound == plan.worth
    if (bound == "most") == (delta < 0):
        held = planned(level, level)
        np.testing.assert_allclose(plan.spend, held.spend, rtol=1e-12)
        assert PERIODS * plan.spend.sum(axis=axis)[0] == pytest.approx(level, rel=1e-13)
        assert plan.price == pytest.approx(held.price, rel=1e-12)
        same = held.geo_prices if kind == "geo" else held.channel_prices
        assert prices[0] == pytest.approx(
            same[0], abs=1e-13 * plan.price
        )  # to the budget's rounding
        assert np.sign(prices[0]) == (1.0 if bound == "most" else -1.0)
    else:
        np.testing.assert_allclose(plan.spend, free.spend, rtol=1e-12)
        assert plan.price == pytest.approx(free.price, rel=1e-12)
        np.testing.assert_array_equal(prices, 0.0)


@pytest.mark.parametrize("short", [1e-9, 1e-6, 1e-3])
def test_a_budget_a_hair_below_the_most_a_cap_allows_is_planned_exactly(short):
    """Geo 0 held at its cap and every other cell at its own but one, which the last hair of budget
    takes off its cap and which alone sets the budget's price. A Newton step on the prices does not
    move that price, every cell it would move held at its cap; the plan is still exact, and every
    cell's slope, through its whole series, meets its prices."""
    cap = 1700.0
    caps = Totals(least=np.zeros(3), most=np.array([cap, np.inf, np.inf]))
    most = cap + PERIODS * UPPER[1:].sum()
    plan = allocate_geos(
        CELLS, most - short, PERIODS, lower=LOWER, upper=UPPER, geo_totals=caps, history=HISTORY
    )
    assert plan.bound == plan.worth
    assert PERIODS * plan.spend[0].sum() == pytest.approx(cap, rel=1e-13)
    assert plan.geo_prices[0] > 0.0
    for g in range(3):
        for c in range(3):
            cell, rate = CELLS[g][c], float(plan.spend[g, c])
            tail = jnp.zeros(cell.kernel.length - 1)

            def worth(r, g=g, c=c, cell=cell, tail=tail):
                series = jnp.concatenate([HISTORY[:, g, c], jnp.full(PERIODS, r), tail])
                return jnp.sum(cell(series)[HISTORY.shape[0] :])

            slope = float(jax.grad(worth)(rate)) / PERIODS
            prices = plan.price + plan.geo_prices[g]
            if LOWER[g, c] < rate < UPPER[g, c]:
                assert slope == pytest.approx(prices, rel=1e-10), (g, c)
            elif rate == LOWER[g, c]:
                assert slope <= prices * (1 + 1e-10), (g, c)
            else:
                assert slope >= prices * (1 - 1e-10), (g, c)


def test_an_s_curve_is_searched_as_allocate_searches_it():
    """The S-curve counterexample: a descent from zero stops at the greedy 225; at a budget of 300
    the plan on the envelopes from zero spend is already the best split, 1047.87, with no gap. At
    90, short of the Hill's tangency, that plan put all 90 on the Hill, for 421.6, but its bound was
    the chord's, 476.2; the search plans the same and closes the gap, as :func:`allocate` does."""
    cells = ((Channel(ONE, Hill(100.0, 3.0), 1000.0), Channel(ONE, MichaelisMenten(100.0), 300.0)),)
    for budget in (300.0, 90.0):
        box = {"lower": np.zeros((1, 2)), "upper": np.full((1, 2), budget)}
        plan = allocate_geos(cells, budget, 1, **box)
        alone = allocate(cells[0], budget, 1, lower=np.zeros(2), upper=np.full(2, budget))
        envelope = _on_envelopes(
            cells[0], budget, 1, np.zeros(2), np.full(2, budget), np.zeros((0, 2))
        )
        np.testing.assert_allclose(plan.spend[0], alone.spend, rtol=1e-12, atol=1e-12)
        assert plan.worth == pytest.approx(alone.worth, rel=1e-12)
        assert plan.spend.sum() == pytest.approx(budget, rel=1e-12)
        assert (plan.stopped, plan.limit) == ("closed", None)
        assert 0.0 <= plan.gap <= plan.tolerance
        assert plan.worth >= envelope.worth * (1 - 1e-12)
    np.testing.assert_array_equal(plan.spend, [[90.0, 0.0]])
    assert plan.worth == pytest.approx(1000.0 * 0.9**3 / (1.0 + 0.9**3), rel=1e-12)
    assert envelope.bound - plan.bound > 54.0
    assert allocate_geos(
        cells, 300.0, 1, lower=np.zeros((1, 2)), upper=np.full((1, 2), 300.0)
    ).worth == (pytest.approx(1047.867, abs=1e-3))


def test_an_s_curve_searched_to_its_tolerance_logs_nothing(caplog):
    """At 90 the plan on the envelopes from zero spend left a gap of 54.6, which a warning logged;
    the search closes it and has nothing to log."""
    cells = ((Channel(ONE, Hill(100.0, 3.0), 1000.0), Channel(ONE, MichaelisMenten(100.0), 300.0)),)
    with caplog.at_level(logging.WARNING, logger="chc.allocation"):
        short = allocate_geos(cells, 90.0, 1, lower=np.zeros((1, 2)), upper=np.full((1, 2), 90.0))
    assert not [r for r in caplog.records if r.name == "chc.allocation"]
    assert short.stopped == "closed"
    assert short.bound - short.worth <= short.tolerance


def _untied() -> tuple[Channel, Channel]:
    """Two Hill curves of slope 3 at scales 1 and 1.01, no carryover."""
    return Channel(ONE, Hill(1.0, 3.0), 1.0), Channel(ONE, Hill(1.01, 3.0), 1.0)


S_GRID = (_untied(), (Channel(ONE, Hill(0.8, 4.0), 0.7), Channel(ONE, Hill(1.2, 2.5), 1.2)))
S_BOX = {"lower": np.zeros((2, 2)), "upper": np.full((2, 2), 2.0)}


def _s_worth(spend: np.ndarray) -> float:
    """What a plan of ``S_GRID`` returns, read on each cell's curve."""
    return sum(
        float(cell(np.array([spend[g, c]]))[0])
        for g, row in enumerate(S_GRID)
        for c, cell in enumerate(row)
    )


def test_one_geo_of_s_curves_is_searched_to_allocate_s_plan():
    """At a budget of 1.6 the cutting planes on the envelopes from zero spend planned 1.27/0.33,
    for 0.705 against a bound of 0.845; the search plans all of it on the first curve, for
    512/637, 0.804, as :func:`allocate` does."""
    pair = _untied()
    plan = allocate_geos((pair,), 1.6, 1, lower=np.zeros((1, 2)), upper=np.full((1, 2), 1.6))
    alone = allocate(pair, 1.6, 1, lower=np.zeros(2), upper=np.full(2, 1.6))
    np.testing.assert_allclose(plan.spend[0], alone.spend, rtol=0.0, atol=1e-12)
    assert plan.worth == pytest.approx(512.0 / 637.0, rel=1e-12, abs=0.0)
    assert plan.worth == pytest.approx(alone.worth, rel=1e-12, abs=0.0)
    assert (plan.stopped, plan.limit) == ("closed", None)
    assert 0.0 <= plan.gap <= plan.tolerance
    assert plan.boxes > 1


@pytest.mark.parametrize(
    ("budget", "envelopes"), [(1.6, 0.704806), (3.2, 1.552281)], ids=["short", "long"]
)
def test_s_curves_over_two_geos_are_searched_to_allocate_s_plan_of_their_cells(budget, envelopes):
    """With no totals the grid's cells are :func:`allocate`'s channels, and its search the oracle:
    each search's worth is within its gap of the best, which both bounds hold. On the envelopes
    from zero spend the planes planned ``envelopes``, short of the best by 0.10 and 0.081."""
    plan = allocate_geos(S_GRID, budget, 1, **S_BOX)
    cells = [cell for row in S_GRID for cell in row]
    flat = allocate(cells, budget, 1, lower=np.zeros(4), upper=np.full(4, 2.0))
    assert plan.spend.sum() == pytest.approx(budget, rel=1e-12)
    assert np.all((plan.spend >= 0.0) & (plan.spend <= 2.0))
    assert plan.worth == pytest.approx(_s_worth(plan.spend), rel=1e-12)
    assert (plan.stopped, plan.limit) == ("closed", None)
    assert 0.0 <= plan.gap <= plan.tolerance
    assert flat.worth - plan.worth <= plan.gap + 1e-15  # the two meet the budget to an ulp
    assert plan.worth - flat.worth <= flat.gap + 1e-15
    assert plan.worth - envelopes > 0.08


SLOW = GeometricAdstock(0.5, length=4, normalized=True)
CARRY_GRID = (
    (
        Channel(SLOW, Hill(1.0, 3.0), 1.0),
        Channel(GeometricAdstock(0.3, length=3, normalized=True), Hill(1.01, 3.0), 1.0),
    ),
    (Channel(ONE, Hill(0.8, 4.0), 0.7), Channel(SLOW, Hill(1.2, 2.5), 1.2)),
)
CARRY_HISTORY = np.random.default_rng(5).gamma(2.0, 0.2, size=(6, 2, 2))


@pytest.mark.parametrize(
    ("rate", "envelopes"), [(1.6, 1.832848), (3.2, 4.056499)], ids=["short", "long"]
)
def test_s_curves_with_carryover_over_two_geos_are_searched_to_allocate_s_plan_of_their_cells(
    rate, envelopes
):
    """Three periods, each cell run over its history at its own adstock: :func:`allocate`'s search
    of the four cells is the oracle, 2.2896 and 4.6326, and the plan's worth is read on the cells
    themselves. On the envelopes from zero spend the planes planned ``envelopes``, short of it by
    0.46 and 0.58."""
    periods = 3
    plan = allocate_geos(CARRY_GRID, periods * rate, periods, **S_BOX, history=CARRY_HISTORY)
    flat = allocate(
        CARRY_GRID[0] + CARRY_GRID[1],
        periods * rate,
        periods,
        lower=np.zeros(4),
        upper=np.full(4, 2.0),
        history=CARRY_HISTORY.reshape(6, 4),
    )
    assert plan.spend.sum() == pytest.approx(rate, rel=1e-12)
    assert plan.worth == pytest.approx(
        _worth(CARRY_GRID, plan.spend, CARRY_HISTORY, periods), rel=1e-12
    )
    assert plan.worth - envelopes > 0.4
    assert (plan.stopped, plan.limit) == ("closed", None)
    assert flat.worth - plan.worth <= plan.gap + 1e-15
    assert plan.worth - flat.worth <= flat.gap + 1e-15


def test_a_box_s_bound_is_its_envelopes_where_newton_s_method_closes_its_plan_on_them():
    """With carryover a cell's periods run at their own adstocks, so its envelope over a box bends
    where the plan's periods are past their chords and the tail's are on them: Newton's method then
    makes the box's plan exact on the envelopes, which stand above the curves there. The box's
    bound is the envelopes' at that plan, 0.23 above :func:`allocate`'s best; read on the curves
    it would be the plan's own worth, 0.033 short of the best, and the search would stop in its
    first box. On the envelopes from zero spend the plan returned 4.2580, 0.046 short."""
    cells = (
        Channel(GeometricAdstock(0.5, length=2, normalized=True), Hill(0.67, 4.2), 1.3),
        Channel(GeometricAdstock(0.3, length=5, normalized=True), Hill(0.65, 3.2), 0.65),
    )
    history = np.random.default_rng(1).gamma(2.0, 0.2, size=(5, 2))
    budget, periods, upper = 6.45, 2, np.full(2, 3.0)
    alone = allocate(cells, budget, periods, lower=np.zeros(2), upper=upper, history=history)
    box = {"lower": np.zeros((1, 2)), "upper": upper[None], "history": history[:, None, :]}
    plan = allocate_geos((cells,), budget, periods, **box)
    assert plan.worth == pytest.approx(alone.worth, rel=1e-12, abs=0.0)
    assert (plan.stopped, plan.limit) == ("closed", None)
    assert alone.worth - plan.worth <= plan.gap + 1e-15
    assert plan.worth - alone.worth <= alone.gap + 1e-15
    one = allocate_geos((cells,), budget, periods, **box, max_boxes=1)
    assert (one.stopped, one.boxes, one.limit) == ("cap", 1, "max_boxes")
    assert alone.worth - one.worth > 0.03
    assert one.bound - alone.worth > 0.2


def test_a_search_of_the_grid_reads_a_cell_s_slope_hundreds_of_times_not_millions(monkeypatch):
    """An envelope's chord holds its cell at one end of it at every price but the chord's slope,
    where its rate jumps across the chord. In a box Newton's method on the totals' prices stops at
    its first step that does not shrink their miss: the search of the grid at 1.6 reads a cell's
    slope 687 times. With a line search along each such step it circled the jump, and read it
    2,093,555 times for the same plan."""
    reads = 0
    slope = chc.allocation._slope

    def counted(worth, rate):
        nonlocal reads
        reads += 1
        return slope(worth, rate)

    monkeypatch.setattr(chc.allocation, "_slope", counted)
    plan = allocate_geos(S_GRID, 1.6, 1, **S_BOX)
    assert (plan.stopped, plan.limit) == ("closed", None)
    assert reads < 5_000


@pytest.mark.parametrize(
    ("kind", "fixed"),
    [("geo", np.array([1.6, 1.6])), ("channel", np.array([2.0, 1.2]))],
    ids=["geos", "channels"],
)
def test_s_curves_under_fixed_totals_are_searched_to_each_group_s_own_plan(kind, fixed):
    """Every geo's total fixed, the grid is each geo's :func:`allocate`; every channel's, each
    channel's over the geos. The search meets that oracle within each search's gap."""
    groups = list(S_GRID) if kind == "geo" else [(S_GRID[0][c], S_GRID[1][c]) for c in range(2)]
    totals = {f"{kind}_totals": Totals(least=fixed, most=fixed)}
    plan = allocate_geos(S_GRID, float(fixed.sum()), 1, **S_BOX, **totals)
    own = [
        allocate(group, total, 1, lower=np.zeros(2), upper=np.full(2, 2.0))
        for group, total in zip(groups, fixed, strict=True)
    ]
    spent = plan.spend.sum(axis=1 if kind == "geo" else 0)
    np.testing.assert_allclose(spent, fixed, rtol=1e-12)
    assert (plan.stopped, plan.limit) == ("closed", None)
    assert sum(p.worth for p in own) - plan.worth <= plan.gap
    assert plan.worth - sum(p.worth for p in own) <= sum(p.gap for p in own) + 1e-15


def test_s_curves_under_totals_on_two_geos_no_lattice_plan_returns_more():
    """A cap of 1.5 on channel 0's total and a floor of 1.0 on geo 1's, at a budget of 3.2: every
    plan whose four rates are whole steps of a 400th of the budget, within the boxes and the
    totals, returns no more than the plan, within its gap, and the best is within the lattice's
    reach of it."""
    budget, steps = 3.2, 400
    caps = Totals(least=np.zeros(2), most=np.array([1.5, np.inf]))
    floors = Totals(least=np.array([0.0, 1.0]), most=np.full(2, np.inf))
    plan = allocate_geos(S_GRID, budget, 1, **S_BOX, channel_totals=caps, geo_totals=floors)
    assert plan.spend.sum() == pytest.approx(budget, rel=1e-12)
    assert plan.spend[:, 0].sum() <= 1.5 * (1 + 1e-12)
    assert plan.spend[1].sum() >= 1.0 * (1 - 1e-12)
    assert plan.stopped == "closed"
    unit = budget / steps
    top = round(2.0 / unit)
    table = [[np.asarray(cell(np.arange(top + 1) * unit)) for cell in row] for row in S_GRID]
    b, c = np.meshgrid(np.arange(top + 1), np.arange(top + 1), indexing="ij")
    best = -np.inf
    for a in range(top + 1):
        d = steps - a - b - c
        keep = (d >= 0) & (d <= top) & ((a + c) * unit <= 1.5) & ((c + d) * unit >= 1.0)
        if keep.any():
            revenue = (
                table[0][0][a] + table[0][1][b[keep]] + table[1][0][c[keep]] + table[1][1][d[keep]]
            )
            best = max(best, float(revenue.max()))
    assert best <= plan.worth + plan.gap
    assert plan.worth - best <= 1e-4 * plan.worth


CUT_ROW = (
    Channel(KERNELS[0], Tanh(250.0), 1100.0),
    Channel(KERNELS[1], Hill(110.0, 2.5), 900.0),
    Channel(KERNELS[2], Hill(200.0, 3.0), 1400.0),
)
CUT_HISTORY = np.random.default_rng(3).gamma(4.0, 20.0, size=(30, 3))
CUT_LOWER = np.array([40.0, 25.0, 15.0])
CUT_UPPER = np.array([160.0, 100.0, 60.0])
CUT_BUDGET = PERIODS * 150.0
# the row twice, the second geo's history 0.7 of the first's
CUT_GRID = (CUT_ROW, CUT_ROW)
CUT_GRID_HISTORY = np.stack([CUT_HISTORY, 0.7 * CUT_HISTORY], axis=1)
CUT_BOX = {"lower": np.stack([CUT_LOWER] * 2), "upper": np.stack([CUT_UPPER] * 2)}


def _in(cell: Channel, currency: float, outcome: float) -> Channel:
    """``cell`` with spend in ``currency`` times its units and return in ``outcome`` times."""
    curve = cell.curve
    scaled = (
        Hill(currency * curve.scale, curve.slope)
        if isinstance(curve, Hill)
        else type(curve)(currency * curve.scale)  # type: ignore[call-arg]
    )
    return Channel(cell.kernel, scaled, outcome * cell.coefficient)


def _cut_plan(currency=1.0, outcome=1.0, **totals):
    """The two geos of ``CUT_ROW`` planned at twice its budget, in ``currency`` and ``outcome``
    times the units, under ``totals`` given in the units."""
    grid = tuple(tuple(_in(cell, currency, outcome) for cell in row) for row in CUT_GRID)
    scaled = {
        name: Totals(least=currency * bound.least, most=currency * bound.most)
        for name, bound in totals.items()
    }
    return allocate_geos(
        grid,
        currency * 2 * CUT_BUDGET,
        PERIODS,
        lower=currency * CUT_BOX["lower"],
        upper=currency * CUT_BOX["upper"],
        history=currency * CUT_GRID_HISTORY,
        **scaled,
    )


def _meets_its_prices(plan, grid, history, lower, upper, held=(), rel=1e-6) -> int:
    """Each cell's slope a currency unit, by central differences of its return over history, plan
    and tail, against its price, the budget's plus its geo's and its channel's: equal inside its
    box, no more at its floor, no less at its cap. A total named in ``held`` is held, and every
    other total's price is nothing. Returns how many cells run inside their boxes."""
    held_geos = {k for kind, k in held if kind == "geo"}
    held_channels = {k for kind, k in held if kind == "channel"}
    for g, price in enumerate(plan.geo_prices):
        assert (price != 0.0) == (g in held_geos), ("geo", g, price)
    for c, price in enumerate(plan.channel_prices):
        assert (price != 0.0) == (c in held_channels), ("channel", c, price)
    inside = 0
    # a rate the search planned at an end of its box can stand a rounding off it
    at_floor = np.isclose(plan.spend, lower, rtol=1e-12, atol=0.0)
    at_cap = np.isclose(plan.spend, upper, rtol=1e-12, atol=0.0)
    for g, row in enumerate(grid):
        for c in range(len(row)):
            step = 1e-4 * plan.spend[g, c]
            up, down = plan.spend.copy(), plan.spend.copy()
            up[g, c] += step
            down[g, c] -= step
            rise = _worth(grid, up, history) - _worth(grid, down, history)
            slope = rise / (2 * step) / PERIODS
            price = plan.price + plan.geo_prices[g] + plan.channel_prices[c]
            if at_floor[g, c]:
                assert slope <= price * (1 + rel), (g, c)
            elif at_cap[g, c]:
                assert slope >= price * (1 - rel), (g, c)
            else:
                inside += 1
                assert slope == pytest.approx(price, rel=rel, abs=0.0), (g, c)
    return inside


@pytest.mark.parametrize(
    ("currency", "outcome"), [(1.0, 1.0), (1e-3, 1.0), (1e3, 1.0), (1.0, 1e-3), (1.0, 1e3)]
)
def test_one_geo_cut_at_its_own_plan_is_priced_on_the_curves_as_allocate_prices_it(
    currency, outcome
):
    """The search cuts the middle channel at its plan, 95, so both halves hold the plan; each
    half's envelopes price it otherwise, and the plan was priced 4.88 in the units and 4.29 with
    spend in thousandths. On the curves one more unit returns 5.61 on the middle channel, in any
    units, as :func:`allocate` prices it."""
    row = tuple(_in(cell, currency, outcome) for cell in CUT_ROW)
    box = {"lower": currency * CUT_LOWER, "upper": currency * CUT_UPPER}
    plan = allocate_geos(
        (row,),
        currency * CUT_BUDGET,
        PERIODS,
        lower=box["lower"][None],
        upper=box["upper"][None],
        history=currency * CUT_HISTORY[:, None, :],
    )
    alone = allocate(row, currency * CUT_BUDGET, PERIODS, **box, history=currency * CUT_HISTORY)
    np.testing.assert_allclose(plan.spend[0] / currency, [40.0, 95.0, 15.0], rtol=1e-12)
    assert plan.boxes == 3
    assert plan.price == pytest.approx(alone.price, rel=1e-12, abs=0.0)
    assert plan.price * currency / outcome == pytest.approx(5.6122, rel=1e-4, abs=0.0)
    inside = _meets_its_prices(
        plan,
        (row,),
        currency * CUT_HISTORY[:, None, :],
        box["lower"][None],
        box["upper"][None],
    )
    assert inside == 1


GEO_CAP = Totals(least=np.zeros(2), most=np.array([0.9 * CUT_BUDGET, np.inf]))
CHANNEL_CAP = Totals(least=np.zeros(3), most=np.array([np.inf, PERIODS * 120.0, np.inf]))
GEO_FLOOR = Totals(least=np.array([0.0, 1.1 * CUT_BUDGET]), most=np.full(2, np.inf))


@pytest.mark.parametrize(
    ("totals", "held", "price"),
    [
        ({"geo_totals": GEO_CAP}, ("geo", 0), 1.604),
        ({"channel_totals": CHANNEL_CAP}, ("channel", 1), 1.5705),
        ({"geo_totals": GEO_FLOOR}, ("geo", 1), -1.604),
    ],
    ids=["geo", "channel", "floor"],
)
@pytest.mark.parametrize(("currency", "outcome"), [(1.0, 1.0), (1e-3, 1e3), (1e3, 1e-3)])
def test_a_total_that_binds_on_s_curves_is_priced_on_the_curves_at_the_plan(
    totals, held, price, currency, outcome
):
    """A cap on geo 0's total, 0.9 of one geo's budget, or on channel 1's, 120 a period, binds,
    or a floor on geo 1's, 1.1 of it, and the search finds the plan in a box cut at the plan's own
    rate. That box's envelopes priced geo 0's room 0.054, channel 1's 0.84 and geo 1's -0.054, short
    of what a cell inside its box under them returns over the budget's price: 1.604, 1.5705 and
    -1.604 on the curves. Every cell meets its prices on the curves, in any units."""
    plan = _cut_plan(currency, outcome, **totals)
    kind, index = held
    room = plan.geo_prices if kind == "geo" else plan.channel_prices
    assert room[index] * currency / outcome == pytest.approx(price, rel=1e-3, abs=0.0)
    grid = tuple(tuple(_in(cell, currency, outcome) for cell in row) for row in CUT_GRID)
    inside = _meets_its_prices(
        plan,
        grid,
        currency * CUT_GRID_HISTORY,
        currency * CUT_BOX["lower"],
        currency * CUT_BOX["upper"],
        held=(held,),
    )
    assert inside >= 2
    if (currency, outcome) != (1.0, 1.0):
        base = _cut_plan(**totals)
        np.testing.assert_allclose(plan.spend, currency * base.spend, rtol=1e-12, atol=0.0)
        ratio = outcome / currency
        assert plan.price == pytest.approx(ratio * base.price, rel=1e-12, abs=0.0)
        np.testing.assert_allclose(plan.geo_prices, ratio * base.geo_prices, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(
            plan.channel_prices, ratio * base.channel_prices, rtol=1e-12, atol=0.0
        )


def test_a_total_a_searched_plan_does_not_hold_is_priced_nothing():
    """A cap on geo 0's total at one geo's budget leaves it 11 short of the cap. The searched
    rates inside their boxes return a currency unit within 4e-6 of each other, not the same, and
    pricing their miss alongside the cap's priced the cap 1.9e-5; it holds nothing, and its price
    is nothing."""
    cap = Totals(least=np.zeros(2), most=np.array([CUT_BUDGET, np.inf]))
    plan = _cut_plan(geo_totals=cap)
    assert cap.most[0] - PERIODS * plan.spend[0].sum() > 10.0
    np.testing.assert_array_equal(plan.geo_prices, 0.0)
    assert _meets_its_prices(plan, CUT_GRID, CUT_GRID_HISTORY, **CUT_BOX, rel=1e-5) == 2


def test_s_curves_under_fixed_geo_totals_carry_the_price_in_the_geos():
    """Every geo's total fixed fixes the budget: its price is nothing, and each geo's is the price
    :func:`allocate` reads on its own plan, one more unit on its curves."""
    fixed = np.array([1.6, 1.6])
    plan = allocate_geos(S_GRID, 3.2, 1, **S_BOX, geo_totals=Totals(least=fixed, most=fixed))
    assert plan.price == 0.0
    for g, row in enumerate(S_GRID):
        alone = allocate(row, fixed[g], 1, lower=np.zeros(2), upper=np.full(2, 2.0))
        np.testing.assert_allclose(plan.spend[g], alone.spend, rtol=0.0, atol=1e-12)
        assert plan.geo_prices[g] == pytest.approx(alone.price, rel=1e-12, abs=0.0)


def test_s_curves_under_geo_caps_that_spend_the_budget_carry_its_price_in_the_geos():
    """Caps on the geos' totals that sum to the budget fix it as fixed totals do, though each
    total may spend less: the budget's price is nothing, and each geo's the price
    :func:`allocate` reads on its own plan."""
    caps = np.array([1.6, 1.6])
    plan = allocate_geos(S_GRID, 3.2, 1, **S_BOX, geo_totals=Totals(least=np.zeros(2), most=caps))
    assert plan.price == 0.0
    for g, row in enumerate(S_GRID):
        alone = allocate(row, caps[g], 1, lower=np.zeros(2), upper=np.full(2, 2.0))
        np.testing.assert_allclose(plan.spend[g], alone.spend, rtol=0.0, atol=1e-12)
        assert plan.geo_prices[g] == pytest.approx(alone.price, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("outcome", [1.0, 1e-20, 1e20])
def test_prices_no_plan_meets_miss_least_where_the_plan_stands_nearest_the_excusing_end(outcome):
    """Off a best plan no price meets every slope. Two rates under a total held at its most,
    nothing charged a unit: 8 of a box of 10 on the steeper curve, 1 on the shallower. A price at
    the steeper slope misses the shallower's, and that rate stands 1 from the floor that would
    excuse the miss; a price at the shallower misses the steeper's, whose rate stands 2 from the
    cap that would. Weighted by those distances the miss is least at the steeper slope,
    ``100 / 81``, in any units of return."""
    cells = (
        Channel(ONE, MichaelisMenten(1.0), 100.0 * outcome),
        Channel(ONE, MichaelisMenten(1.0), outcome),
    )
    worths = _worths(cells, np.zeros((0, 2)), 1)
    held = _Group(np.arange(2), 0.0, 9.0, np.ones(2))
    [price] = _priced(
        worths, np.array([8.0, 1.0]), np.zeros(2), np.full(2, 10.0), [held], 1, 0.0, False
    )
    assert price == pytest.approx(outcome * 100.0 / 81.0, rel=1e-12, abs=0.0)


def test_the_budget_s_price_nearest_nothing_stands_while_the_totals_prices_are_made_least():
    """A plan that meets the conditions at a range of prices: geo 0's two rates at their cap and
    their floor leave the budget's price anywhere from 0.5 to 1, and geo 1's rate inside its box
    under a cap on the geo's total returns 3, the budget's price plus the cap's. The budget's price
    is the end nearest nothing, 0.5, and the cap's the rest, 2.5, though a budget's price of 1
    would leave the cap's less."""
    cells = tuple(Channel(ONE, MichaelisMenten(1.0), size) for size in (121.0, 0.5, 12.0, 1.0))
    worths = _worths(cells, np.zeros((0, 4)), 1)
    rates = np.array([10.0, 0.0, 1.0, 0.0])
    budget = _Group(np.arange(4), 11.0, 11.0, np.ones(4))
    cap = _Group(np.array([2, 3]), 0.0, 1.0, np.ones(2))
    prices = _priced(worths, rates, np.zeros(4), np.full(4, 10.0), [budget, cap], 1, 0.0, True)
    np.testing.assert_allclose(prices, [0.5, 2.5], rtol=1e-12, atol=0.0)


def test_under_a_budget_a_rate_a_rounding_below_its_cap_has_no_room_to_price():
    """No rate with room below its cap returns more than the budget's price, but HiGHS can leave a
    rate it holds at its cap a rounding below it. Of two rates, one two ulps short of its cap of 1,
    where its curve returns 1 a unit, and one inside its box returning 0.25, the price is 0.25:
    read as room, the two ulps priced the budget 1."""
    cells = (Channel(ONE, MichaelisMenten(1.0), 4.0), Channel(ONE, MichaelisMenten(1.0), 4.0))
    worths = _worths(cells, np.zeros((0, 2)), 1)
    rates = np.array([1.0 - 2.0**-52, 3.0])
    budget = _Group(np.arange(2), float(rates.sum()), float(rates.sum()), np.ones(2))
    [price] = _priced(worths, rates, np.zeros(2), np.array([1.0, 10.0]), [budget], 1, 0.0, True)
    assert price == pytest.approx(0.25, rel=1e-12, abs=0.0)


def test_a_rate_its_box_pins_sets_no_scale_for_the_prices():
    """A rate its box pins holds no row, however steep its curve stands there: a power of 0.2 reads
    6.7e11 a unit at nothing. As the program's scale it left the slope of the rate inside its box,
    0.25, under HiGHS's tolerance, and the budget's price came out 0."""
    cells = (Channel(ONE, Power(1.0, 0.2), 1.0), Channel(ONE, MichaelisMenten(1.0), 4.0))
    worths = _worths(cells, np.zeros((0, 2)), 1)
    budget = _Group(np.arange(2), 3.0, 3.0, np.ones(2))
    rates, upper = np.array([0.0, 3.0]), np.array([0.0, 10.0])
    [price] = _priced(worths, rates, np.zeros(2), upper, [budget], 1, 0.0, True)
    assert price == pytest.approx(0.25, rel=1e-12, abs=0.0)


def test_a_cell_its_box_pins_at_nothing_leaves_the_price_to_the_cells_it_frees():
    """Geo 1's first channel does not run there: its box pins it at nothing, on a power of 0.2 with
    no carryover, where its curve stands at 2.9e12 a unit. The budget's price is what the cells
    inside their boxes return, 5.635; read with that slope as the prices' scale, it was 0."""
    row = (Channel(KERNELS[0], Power(250.0, 0.2), 1100.0), *CUT_ROW[1:])
    grid = (CUT_ROW, row)
    history = CUT_GRID_HISTORY.copy()
    history[:, 1, 0] = 0.0
    lower, upper = CUT_BOX["lower"].copy(), CUT_BOX["upper"].copy()
    lower[1, 0] = upper[1, 0] = 0.0
    budget = 2 * CUT_BUDGET - PERIODS * CUT_LOWER[0]
    plan = allocate_geos(grid, budget, PERIODS, lower=lower, upper=upper, history=history)
    at_end = np.isclose(plan.spend, lower, rtol=1e-12, atol=0.0) | np.isclose(
        plan.spend, upper, rtol=1e-12, atol=0.0
    )
    assert (~at_end).sum() == 2
    for g, c in zip(*np.nonzero(~at_end), strict=True):
        step = 1e-4 * plan.spend[g, c]
        up, down = plan.spend.copy(), plan.spend.copy()
        up[g, c] += step
        down[g, c] -= step
        slope = (_worth(grid, up, history) - _worth(grid, down, history)) / (2 * step) / PERIODS
        assert plan.price == pytest.approx(slope, rel=1e-5, abs=0.0), (g, c)


def test_a_price_program_its_iteration_limit_stops_is_refused(stall):
    """The prices' programs run under the planes' limit of iterations: weights down to roundings
    put their costs in HiGHS's scaling, where at their tolerance it can pivot on a degenerate
    program without end. A program the limit stops leaves no price to read."""
    limits = stall(1)
    cells = (Channel(ONE, MichaelisMenten(1.0), 4.0), Channel(ONE, MichaelisMenten(1.0), 4.0))
    worths = _worths(cells, np.zeros((0, 2)), 1)
    budget = _Group(np.arange(2), 4.0, 4.0, np.ones(2))
    with pytest.raises(RuntimeError, match="prices' linear program failed"):
        _priced(
            worths, np.array([1.0, 3.0]), np.zeros(2), np.array([1.0, 10.0]), [budget], 1, 0.0, True
        )
    assert len(limits) == 1
    assert limits[0] > 0


@pytest.mark.parametrize("seed", range(6))
def test_under_the_budget_alone_any_plan_is_priced_as_allocate_prices_it(seed):
    """With the budget the only total, a plan off its best misses the conditions whatever its price,
    and its price is still the one :func:`allocate` reads: the steepest slope of a rate below its
    cap, however far the other rates stand from their floors and caps."""
    rng = np.random.default_rng(seed)
    cells = tuple(Channel(ONE, MichaelisMenten(1.0), size) for size in rng.uniform(1.0, 100.0, 5))
    worths = _worths(cells, np.zeros((0, 5)), 1)
    upper = np.full(5, 10.0)
    rates = rng.uniform(0.0, 10.0, 5)
    rates[:2] = 0.0, 10.0
    budget = _Group(np.arange(5), float(rates.sum()), float(rates.sum()), np.ones(5))
    [price] = _priced(worths, rates, np.zeros(5), upper, [budget], 1, 0.0, True)
    alone = chc.allocation._marginal(worths, rates, upper, 1)
    assert price == pytest.approx(alone, rel=1e-12, abs=0.0)


def test_a_cell_a_rounding_below_its_cap_leaves_the_budget_s_price_to_the_cell_with_room():
    """Caps drawn at random on the S-curve grid: every cell but geo 0's second ends at its cap, and
    the planes' program leaves geo 0's first two ulps below it, where its curve returns 0.387 a
    unit. Read as room, those ulps priced the budget 0.387, 13 times what one more unit returns on
    the only cell with room: 0.0297 on its curve."""
    upper = np.array(
        [[1.4460624847872807, 0.5732431358810791], [1.297579627393073, 1.8858181113512407]]
    )
    plan = allocate_geos(S_GRID, 4.730640195168108, 1, lower=np.zeros((2, 2)), upper=upper)
    held = np.ones((2, 2), dtype=bool)
    held[0, 1] = False
    np.testing.assert_allclose(plan.spend[held], upper[held], rtol=1e-15, atol=0.0)
    rate, cell = plan.spend[0, 1], S_GRID[0][1]
    step = 1e-6 * rate
    rise = float(cell(np.array([rate + step]))[0]) - float(cell(np.array([rate - step]))[0])
    assert plan.price == pytest.approx(rise / (2 * step), rel=1e-6, abs=0.0)


@pytest.mark.parametrize("kind", ["geo", "channel"])
def test_a_total_far_from_binding_leaves_the_one_that_binds_its_price(kind):
    """A cap the plan stands 1e12 short of holds nothing, and the cap that binds beside it is
    priced as it is with no cap there. Read beside that 1e12, every rate's distance from the ends
    of its box was a rounding, and the binding cap on geo 0's total was priced 0.054 where it
    returns 1.60, the one on channel 1's 0 where it returns 1.57."""
    if kind == "geo":
        alone = _cut_plan(geo_totals=GEO_CAP)
        plan = _cut_plan(
            geo_totals=Totals(least=np.zeros(2), most=np.array([GEO_CAP.most[0], 1e12]))
        )
    else:
        alone = _cut_plan(channel_totals=CHANNEL_CAP)
        plan = _cut_plan(
            channel_totals=Totals(least=np.zeros(3), most=np.array([1e12, *CHANNEL_CAP.most[1:]]))
        )
    np.testing.assert_allclose(plan.spend, alone.spend, rtol=1e-12, atol=0.0)
    assert plan.price == pytest.approx(alone.price, rel=1e-12, abs=0.0)
    np.testing.assert_allclose(plan.geo_prices, alone.geo_prices, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(plan.channel_prices, alone.channel_prices, rtol=1e-12, atol=0.0)


def test_a_marginal_target_on_s_curves_prices_a_total_that_binds_on_the_curves():
    """With the budget free and each unit charged 4, the cap on geo 0 binds, and the search finds
    the plan in a box cut at its rate. The budget's price is the charge, and geo 0's the rest of
    what its middle channel, inside its box, returns: 1.84 on the curves, where that box's
    envelopes priced it 0.90."""
    plan = budget_for_geos(
        CUT_GRID,
        MarginalReturnTarget(4.0),
        PERIODS,
        **CUT_BOX,
        geo_totals=GEO_CAP,
        history=CUT_GRID_HISTORY,
    )
    assert plan.price == 4.0
    assert plan.geo_prices[0] == pytest.approx(1.838, rel=1e-3, abs=0.0)
    assert PERIODS * plan.spend[0].sum() == pytest.approx(GEO_CAP.most[0], rel=1e-12)
    assert _meets_its_prices(plan, CUT_GRID, CUT_GRID_HISTORY, **CUT_BOX, held=[("geo", 0)]) >= 2


def test_a_search_of_the_grid_its_cap_stops_says_so_and_logs_its_gap(caplog):
    """One box leaves the untied pair's gap open, 0.14, and logs it; five close it."""
    pair = (_untied(),)
    box = {"lower": np.zeros((1, 2)), "upper": np.full((1, 2), 1.6)}
    with caplog.at_level(logging.WARNING, logger="chc.allocation"):
        plan = allocate_geos(pair, 1.6, 1, **box, max_boxes=1)
    assert (plan.stopped, plan.boxes, plan.limit) == ("cap", 1, "max_boxes")
    assert plan.gap == plan.bound - plan.worth
    assert plan.gap > 0.1
    [record] = [r for r in caplog.records if r.name == "chc.allocation"]
    assert (record.chc_event, record.planner, record.limit) == (
        "allocation_cap",
        "allocate_geos",
        "max_boxes",
    )
    assert (record.boxes, record.worth, record.bound) == (1, plan.worth, plan.bound)
    closed = allocate_geos(pair, 1.6, 1, **box, max_boxes=5)
    assert (closed.stopped, closed.boxes) == ("closed", 5)


def test_concave_cells_are_planned_in_one_box():
    plan = _tied()
    assert (plan.stopped, plan.boxes, plan.limit) == ("closed", 1, None)
    assert plan.gap == 0.0
    assert plan.tolerance == pytest.approx(1e-9 * plan.bound, rel=1e-12)
    assert not plan.floored


@pytest.mark.parametrize(
    ("setting", "match"),
    [({"rtol": -1.0}, "rtol"), ({"atol": np.inf}, "atol"), ({"max_boxes": 0}, "max_boxes")],
)
def test_it_refuses_search_settings_it_cannot_search_to(setting, match):
    with pytest.raises(ValueError, match=match):
        _plan(**setting)


def test_a_straight_stretch_leaves_the_cutting_planes_plan_within_its_gap():
    """A linear cell's rate does not follow its price, so no Newton step makes the plan exact: the
    cutting planes' plan comes back, its worth within its gap of the best, 1e-9 of the bound."""
    cells = ((Channel(ONE, Power(100.0, 1.0), 200.0), Channel(ONE, MichaelisMenten(50.0), 900.0)),)
    plan = allocate_geos(cells, 150.0, 1, lower=np.zeros((1, 2)), upper=np.full((1, 2), 150.0))
    best = 200.0 * 50.0 / 100.0 + 900.0 * 100.0 / 150.0  # the concave cell's slope meets 2 at 100
    assert plan.spend.sum() == pytest.approx(150.0, rel=1e-12)
    assert plan.worth <= best * (1 + 1e-12)
    assert 0.0 <= plan.bound - plan.worth <= 1e-9 * plan.bound
    assert plan.bound >= best * (1 - 1e-12)
    np.testing.assert_allclose(plan.spend[0], [50.0, 100.0], rtol=1e-3)


def test_a_cell_its_box_holds_at_zero_is_planned_where_its_slope_there_is_infinite():
    """A square root's slope at zero has no tangent to bound its worth, so its cap's value bounds
    it; held at zero, it spends nothing and the other cell takes the budget."""
    cells = ((Channel(ONE, Power(100.0, 0.5), 300.0), Channel(ONE, MichaelisMenten(50.0), 900.0)),)
    plan = allocate_geos(cells, 80.0, 1, lower=np.zeros((1, 2)), upper=np.array([[0.0, 100.0]]))
    assert plan.spend[0, 0] == 0.0
    assert plan.spend[0, 1] == pytest.approx(80.0, rel=1e-13)
    assert plan.worth == pytest.approx(900.0 * 80.0 / 130.0, rel=1e-13)
    assert plan.bound == plan.worth


def test_a_change_of_currency_moves_the_spend_and_the_prices_and_leaves_the_worth():
    k = 1000.0
    caps, floors = _totals()

    def rescaled(cell: Channel) -> Channel:
        return Channel(cell.kernel, type(cell.curve)(k * cell.curve.scale), cell.coefficient)

    plan = _tied()
    moved = allocate_geos(
        tuple(tuple(rescaled(cell) for cell in row) for row in CELLS),
        k * BUDGET,
        PERIODS,
        lower=k * LOWER,
        upper=k * UPPER,
        geo_totals=Totals(least=k * caps.least, most=k * caps.most),
        channel_totals=Totals(least=k * floors.least, most=k * floors.most),
        history=k * HISTORY,
    )
    np.testing.assert_allclose(moved.spend, k * plan.spend, rtol=1e-9)
    assert moved.worth == pytest.approx(plan.worth, rel=1e-12)
    assert moved.price == pytest.approx(plan.price / k, rel=1e-9)
    np.testing.assert_allclose(moved.geo_prices, plan.geo_prices / k, rtol=1e-9)
    np.testing.assert_allclose(moved.channel_prices, plan.channel_prices / k, rtol=1e-9)


def _plan(**change):
    arguments = {
        "cells": CELLS,
        "budget": BUDGET,
        "periods": PERIODS,
        "lower": LOWER,
        "upper": UPPER,
        "history": HISTORY,
    } | change
    return allocate_geos(
        arguments.pop("cells"), arguments.pop("budget"), arguments.pop("periods"), **arguments
    )


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        ({"cells": ()}, ValueError, "needs a geo with a channel"),
        ({"cells": (CELLS[0], CELLS[1][:2], CELLS[2])}, ValueError, "each needs every channel"),
        ({"lower": LOWER[:2]}, ValueError, r"it needs \(3, 3\), one a cell"),
        ({"history": HISTORY[:, :2]}, ValueError, r"it needs \(T, 3, 3\)"),
        ({"budget": PERIODS * 2000.0}, ValueError, "outside what the box spends"),
        (
            {
                "cells": (
                    CELLS[0],
                    (*CELLS[1][:2], Channel(KERNELS[2], Ricker(100.0), 50.0)),
                    CELLS[2],
                )
            },
            ValueError,
            "geo 1's channel 2's curve is a Ricker",
        ),
        ({"geo_totals": (np.zeros(3), np.full(3, np.inf))}, TypeError, "not Totals"),
        (
            {"geo_totals": Totals(least=np.zeros(2), most=np.full(2, np.inf))},
            ValueError,
            "bound 2 groups; there are 3 geos",
        ),
        (
            {"channel_totals": Totals(least=np.array([0.0, 0.0, 1e6]), most=np.full(3, np.inf))},
            ValueError,
            "channel 2's total",
        ),
        (
            {
                "geo_totals": Totals(least=np.zeros(3), most=np.full(3, PERIODS * 120.0)),
                "channel_totals": Totals(
                    least=np.array([PERIODS * 360.0, 0.0, 0.0]), most=np.full(3, np.inf)
                ),
            },
            ValueError,
            "no plan meets the budget, the boxes and the totals together",
        ),
    ],
)
def test_it_refuses_what_it_cannot_plan(change, error, match):
    with pytest.raises(error, match=match):
        _plan(**change)


def test_totals_refuse_bounds_that_are_not_ordered():
    with pytest.raises(ValueError, match="not 0 <= least <= most"):
        Totals(least=np.array([5.0]), most=np.array([4.0]))
    with pytest.raises(ValueError, match="one bound a group"):
        Totals(least=np.zeros(2), most=np.zeros(3))


def test_totals_hold_a_copy_the_caller_cannot_change():
    """Totals kept the caller's arrays, so a nan written into them later passed every check."""
    least, most = np.zeros(2), np.array([5.0, np.inf])
    totals = Totals(least=least, most=most)
    least[0], most[0] = np.nan, -1.0
    np.testing.assert_array_equal(totals.least, [0.0, 0.0])
    np.testing.assert_array_equal(totals.most, [5.0, np.inf])
    for bound in (totals.least, totals.most):
        with pytest.raises(ValueError, match="read-only"):
            bound[0] = np.nan


def _tied_by(caps, floors):
    return allocate_geos(
        CELLS,
        BUDGET,
        PERIODS,
        lower=LOWER,
        upper=UPPER,
        geo_totals=caps,
        channel_totals=floors,
        history=HISTORY,
    )


@pytest.mark.parametrize("at", [2, 7])
def test_a_program_its_iteration_limit_stops_ends_the_planes_on_the_last_one_solved(stall, at):
    """A program HiGHS stalls on ends the planes: the plan is the best they found, its bound and
    prices the last solved program's, and Newton's method on the totals that bind makes it exact
    from those prices."""
    caps, floors = _totals()
    exact = _tied_by(caps, floors)
    limits = stall(at)
    plan = _tied_by(caps, floors)
    assert len(limits) == at
    assert min(limits) > 0
    np.testing.assert_allclose(plan.spend, exact.spend, rtol=1e-9)
    assert plan.bound >= exact.worth - 1e-6


def test_a_first_program_its_iteration_limit_stops_is_refused(stall):
    """With no program solved there is neither a bound nor a price to keep."""
    caps, floors = _totals()
    stall(1)
    with pytest.raises(RuntimeError, match="left it unsolved"):
        _tied_by(caps, floors)
