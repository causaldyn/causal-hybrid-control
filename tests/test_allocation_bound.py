"""chc.allocation's cutting-plane bound: read from HiGHS's duals on each program as written, worked
exactly and rounded down, never HiGHS's objective.

Checked against the same bound worked in rational arithmetic from the same duals, on random programs
and duals good and bad, each product split into two doubles that sum to it; against the least of
small programs found exactly at their vertices; on two programs whose HiGHS objective passes their
least, by a slope HiGHS leaves out and by a rounding, and on two plans at a budget whose every slope
HiGHS leaves out; on the boxes the planners' free and one-sided columns are read in, each planner's
around HiGHS's solutions; on the duals a cvar program's readings are evened to; and on a plan with
nothing to gain, whose bound of nothing closes its search.
"""

import math
from fractions import Fraction
from itertools import combinations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import sparse
from scipy.optimize import linprog

import chc.allocation as allocation
from chc.allocation import allocate_geos, cvar_allocate, minimax_allocate
from chc.response import Channel, GeometricAdstock, Hill, MichaelisMenten, Power

ONE = GeometricAdstock(0.0, length=1, normalized=False)  # no carryover
# dyadic numbers from 2**-130 to 2**33 and nothing, every product of which a double pair holds;
# Hypothesis draws few integers with all 53 bits, so a share of them is drawn as a float's
NUMBERS = st.one_of(
    st.sampled_from([0.0, 1.0, -1.0, 0.5, 3.0]),
    st.builds(math.ldexp, st.integers(-(2**53), 2**53), st.integers(-130, -20)),
    st.builds(
        math.ldexp,
        st.floats(0.5, 1.0, exclude_max=True) | st.floats(-1.0, -0.5, exclude_min=True),
        st.integers(-76, 33),
    ),
)


def _exact(objective, matrix, sides, box, duals) -> Fraction | float:
    """The bound :func:`_least` reads, in rational arithmetic: each row's dual times the side it
    presses on, a dual not finite or pressing on an infinite side read as nothing, and each column's
    reduced cost times the end of its box it presses on, ``-inf`` where that end is infinite."""
    least, most = sides
    low, high = box
    dense = sparse.csr_array(matrix).toarray()
    y = [
        Fraction(0)
        if not math.isfinite(v)
        or (v > 0.0 and least[i] == -math.inf)
        or (v < 0.0 and most[i] == math.inf)
        else Fraction(v)
        for i, v in enumerate(map(float, duals))
    ]
    total = sum(
        (v * Fraction(least[i] if v > 0 else most[i]) for i, v in enumerate(y) if v), Fraction(0)
    )
    for j, cost in enumerate(objective):
        reduced = Fraction(cost) - sum(
            (Fraction(dense[i, j]) * v for i, v in enumerate(y)), Fraction(0)
        )
        if reduced:
            end = float(low[j] if reduced > 0 else high[j])
            if not math.isfinite(end):
                return -math.inf
            total += reduced * Fraction(end)
    return total


def _below(value: Fraction | float) -> float:
    """The largest double at or below ``value``."""
    if value == -math.inf:
        return -math.inf
    nearest = float(value)
    return math.nextafter(nearest, -math.inf) if Fraction(nearest) > value else nearest


def _vertex_least(objective, rows, limits, low, high) -> Fraction | None:
    """The least of ``objective @ x`` over ``rows @ x <= limits`` in the box, two columns, found
    exactly at the vertices; None where no point meets them."""
    lines = [
        (Fraction(a), Fraction(b), Fraction(c)) for (a, b), c in zip(rows, limits, strict=True)
    ]
    lines += [
        (Fraction(-1), Fraction(0), -Fraction(low[0])),
        (Fraction(1), Fraction(0), Fraction(high[0])),
    ]
    lines += [
        (Fraction(0), Fraction(-1), -Fraction(low[1])),
        (Fraction(0), Fraction(1), Fraction(high[1])),
    ]
    values = []
    for (a, b, c), (d, e, f) in combinations(lines, 2):
        determinant = a * e - b * d
        if determinant:
            x, y = (c * e - b * f) / determinant, (a * f - c * d) / determinant
            if all(p * x + q * y <= r for p, q, r in lines):
                values.append(Fraction(objective[0]) * x + Fraction(objective[1]) * y)
    return min(values, default=None)


def test_a_slope_highs_leaves_out_is_read_into_the_bound():
    """HiGHS leaves out of the program it solves every entry of at most 1e-9, its
    ``small_matrix_value``: here the slope 2^-33 of x2 in x1 + 2^-33 x2 >= 1, so it reports 1, the
    least of x1 without x2. With x2 up to 2^20 the least is 1 - 2^-13 = 8191/8192, and the bound
    read from HiGHS's duals on the program as written is that exactly."""
    objective, rows, limits = (
        np.array([1.0, 0.0]),
        np.array([[-1.0, -(2.0**-33)]]),
        np.array([-1.0]),
    )
    bounds = [(0.0, 2.0), (0.0, 2.0**20)]
    highs = linprog(objective, A_ub=rows, b_ub=limits, bounds=bounds, method="highs")
    assert highs.fun == 1.0  # HiGHS's objective stands above the least
    program = allocation._planes(objective, rows, limits, bounds, None, None)
    assert program.status == 0
    assert Fraction(program.fun) == Fraction(8191, 8192)


@pytest.mark.parametrize("divisor", [5.0, 10.0, 13.0, 25.0])
def test_a_rounding_highs_leaves_above_the_least_is_not_in_the_bound(divisor):
    """The least of x over divisor * x >= 1 is 1 / divisor, which no double is; HiGHS's x is the
    nearest one, above it for these divisors. The bound is at or below the least, by no more than
    rounding."""
    objective, rows, limits = np.array([1.0]), np.array([[-divisor]]), np.array([-1.0])
    least = Fraction(1) / Fraction(divisor)
    highs = linprog(objective, A_ub=rows, b_ub=limits, bounds=[(0.0, 1.0)], method="highs")
    assert Fraction(highs.fun) > least
    program = allocation._planes(objective, rows, limits, [(0.0, 1.0)], None, None)
    assert Fraction(program.fun) <= least
    assert least - Fraction(program.fun) <= 4 * Fraction(math.ulp(float(least)))


def _linear(*coefficients: float) -> tuple[Channel, ...]:
    """A reading of linear channels, each returning its coefficient over 100 a unit."""
    return tuple(Channel(ONE, Power(100.0, 1.0), c) for c in coefficients)


def test_a_split_whose_slopes_highs_leaves_out_keeps_its_bound_under_the_least_worst_regret():
    """At a budget of 1e10 each slope over the largest best return is 1e-10, which HiGHS leaves
    out. Readings returning 2 and 1 a unit, and 1 and 2, are each best all on the better channel,
    at 2e10, and the least worst regret is a quarter of that, 5e9, at the even split. HiGHS's
    objective on the program without its slopes read 1e10, the worst regret of the split all on
    one channel, and closed there; the bound read from its duals holds."""
    budget = 1e10
    plan = minimax_allocate(
        [_linear(200.0, 100.0), _linear(100.0, 200.0)],
        budget,
        1,
        lower=np.zeros(2),
        upper=np.full(2, budget),
        history=np.zeros((0, 2)),
    )
    assert plan.bound <= 5e9 <= plan.worst


def test_a_worst_share_split_whose_slopes_highs_leaves_out_keeps_its_bound_over_the_best():
    """At the level 1, readings returning 2 and 1 a unit, and 1 and 1.5, against all on the second
    channel: the mean gain is a quarter of the spend on the first, 2.5e9 all on it at a budget of
    1e10, where HiGHS leaves out every slope. Its objective read 0, the reference's, and closed
    there; the bound read from its duals holds, and the search closes only where the split is
    within its tolerance of it."""
    budget = 1e10
    plan = cvar_allocate(
        [_linear(200.0, 100.0), _linear(100.0, 150.0)],
        budget,
        1,
        level=1.0,
        against=np.array([0.0, budget]),
        lower=np.zeros(2),
        upper=np.full(2, budget),
        history=np.zeros((0, 2)),
    )
    assert plan.bound >= 2.5e9
    assert plan.stopped == "cap" or plan.cvar >= 2.5e9 * (1.0 - 1e-9)


@settings(max_examples=300, deadline=None)
@given(data=st.data(), rows=st.integers(1, 5), columns=st.integers(1, 5))
def test_the_bound_is_its_duals_exact_bound_rounded_down(data, rows, columns):
    """Whatever the duals, a basis's or none, signed either way, infinite or not a number, and
    whatever the sides and ends, infinite among them: where every product's error is a double the
    bound is the exact bound of those duals, rounded down to the double at or below it."""

    def draws(elements, size):
        return np.array(data.draw(st.lists(elements, min_size=size, max_size=size)), dtype=float)

    matrix = draws(NUMBERS, rows * columns).reshape(rows, columns)
    objective = draws(NUMBERS, columns)
    least = draws(NUMBERS | st.just(-math.inf), rows)
    most = draws(NUMBERS | st.just(math.inf), rows)
    low = draws(NUMBERS | st.just(-math.inf), columns)
    high = draws(NUMBERS | st.just(math.inf), columns)
    duals = draws(NUMBERS | st.sampled_from([math.nan, math.inf, -math.inf]), rows)
    bound = allocation._least(objective, matrix, (least, most), (low, high), duals)
    assert bound == _below(_exact(objective, matrix, (least, most), (low, high), duals))


@settings(max_examples=300, deadline=None)
@given(data=st.data(), rows=st.integers(1, 4), columns=st.integers(1, 4))
def test_where_a_product_underflows_or_passes_2_to_300_the_bound_stays_below(data, rows, columns):
    """Numbers from the least subnormal to past ``2**300``: the bound is still at or below the
    exact bound of its duals, ``-inf`` where a number passes ``2**300``."""
    wide = st.just(0.0) | st.builds(
        math.ldexp, st.integers(-(2**53), 2**53), st.integers(-1126, 300)
    )

    def draws(elements, size):
        return np.array(data.draw(st.lists(elements, min_size=size, max_size=size)), dtype=float)

    matrix = draws(wide, rows * columns).reshape(rows, columns)
    objective = draws(wide, columns)
    sides = (draws(wide | st.just(-math.inf), rows), draws(wide | st.just(math.inf), rows))
    box = (draws(wide | st.just(-math.inf), columns), draws(wide | st.just(math.inf), columns))
    duals = draws(wide, rows)
    bound = allocation._least(objective, matrix, sides, box, duals)
    exact = _exact(objective, matrix, sides, box, duals)
    assert bound == -math.inf or Fraction(bound) <= exact


def test_a_product_that_underflows_is_allowed_for():
    """2^-600 times 2^-600 underflows to nothing, but leaves a reduced cost of 2^-1200 on a column
    reaching -1, so the exact bound is -2^-1200, below the least subnormal, and the bound is under
    it. A factor past 2^300 gives no bound. Nothing times 2^-1000 is nothing exactly, so a bound of
    nothing stays nothing however small the dual."""
    objective, matrix = np.zeros(1), np.array([[2.0**-600]])
    sides, box = (np.array([-math.inf]), np.zeros(1)), (np.array([-1.0]), np.array([1.0]))
    bound = allocation._least(objective, matrix, sides, box, np.array([-(2.0**-600)]))
    assert _exact(objective, matrix, sides, box, [-(2.0**-600)]) == -(Fraction(2) ** -1200)
    assert -2 * math.ulp(0.0) <= bound < 0.0
    assert (
        allocation._least(objective, np.array([[2.0**301]]), sides, box, np.array([-1.0]))
        == -math.inf
    )
    # the dual's side is nothing, and its plane's entry leaves the column no reduced cost
    tiny = np.array([-(2.0**-1000)])
    assert (
        allocation._least(-(2.0**-900) * np.ones(1), np.array([[2.0**100]]), sides, box, tiny) == 0
    )
    assert _exact(-(2.0**-900) * np.ones(1), np.array([[2.0**100]]), sides, box, tiny) == 0


@settings(max_examples=300, deadline=None)
@given(
    a=st.floats(-(2.0**300), 2.0**300, allow_nan=False),
    b=st.floats(-(2.0**300), 2.0**300, allow_nan=False),
)
def test_a_product_is_two_doubles_that_sum_to_it_or_its_rounding_and_a_miss(a, b):
    """Factors under ``2**300``, subnormal ones among them: where the split misses nothing, the
    product's rounding and rest sum to it exactly; elsewhere the rest is nothing and the rounding
    within the miss of the product."""
    upper, lower, miss = (
        float(part[0]) for part in allocation._product(np.array([a]), np.array([b]))
    )
    exact = Fraction(a) * Fraction(b)
    if miss == 0.0:
        assert Fraction(upper) + Fraction(lower) == exact
    else:
        assert lower == 0.0
        assert abs(exact - Fraction(upper)) <= Fraction(miss)


@settings(max_examples=200, deadline=None)
@given(
    data=st.data(),
    rows=st.integers(1, 4),
    noise=st.lists(st.floats(-1.0, 1.0), min_size=4, max_size=4),
)
def test_no_bound_passes_the_least_of_a_small_program(data, rows, noise):
    """Two columns in a box under up to four planes, the least found exactly at the vertices: the
    bound from HiGHS's duals is at or below it and within 1e-9 of it, and the bound from those duals
    bent, scaled or turned is at or below the exact bound of the duals it was given, which is at or
    below the least. No entry is as small as HiGHS's ``small_matrix_value``, 1e-9, which it would
    leave out, its duals then those of another program."""
    small = (
        st.integers(-32, 32).map(lambda k: k / 8.0) | st.floats(1e-3, 4.0) | st.floats(-4.0, -1e-3)
    )

    def draws(elements, size):
        return np.array(data.draw(st.lists(elements, min_size=size, max_size=size)), dtype=float)

    matrix = draws(small, 2 * rows).reshape(rows, 2)
    limits = draws(small, rows)
    objective = draws(small, 2)
    low = draws(st.floats(-4.0, 0.0), 2)
    high = low + draws(st.floats(0.0, 4.0), 2)
    least = _vertex_least(objective, matrix, limits, low, high)
    program = allocation._planes(
        objective, matrix, limits, list(zip(low, high, strict=True)), None, None
    )
    if least is None or program.status != 0:
        return
    assert Fraction(program.fun) <= least
    assert float(least - Fraction(program.fun)) <= 1e-9 * (1.0 + abs(float(least)))
    sides, box = (np.full(rows, -math.inf), limits), (low, high)
    marginals = program.ineqlin.marginals
    for duals in (
        marginals * (1.0 + 1e-3 * np.resize(noise, rows)),
        marginals * 1e6,
        -marginals,
        np.resize(noise, rows),
    ):
        exact = _exact(objective, matrix, sides, box, duals)
        assert Fraction(allocation._least(objective, matrix, sides, box, duals)) <= exact <= least


@settings(max_examples=200, deadline=None)
@given(data=st.data(), rows=st.integers(1, 6), columns=st.integers(1, 4))
def test_a_box_s_span_holds_every_plane_s_exact_reach(data, rows, columns):
    """Each plane's limit less its product with a point of the box, over every plane and every
    point: the least and the most :func:`_spans` reads are at and past the exact ones, by no more
    than rounding of the planes' magnitudes."""

    def draws(elements, size):
        return np.array(data.draw(st.lists(elements, min_size=size, max_size=size)), dtype=float)

    matrix = draws(NUMBERS, rows * (columns + 1)).reshape(rows, columns + 1)
    limits = draws(NUMBERS, rows)
    low = draws(NUMBERS, columns)
    high = np.maximum(low, draws(NUMBERS, columns))
    least, most = allocation._spans(matrix, limits, low, high)
    planes = list(zip(matrix, limits, strict=True))

    def level(row, limit, corner) -> Fraction:
        """The plane's limit less its product with the box's ``corner``, exactly; the plane's
        last entry, its level's, is not in the box."""
        return Fraction(limit) - sum(
            corner(Fraction(a) * Fraction(start), Fraction(a) * Fraction(stop))
            for a, start, stop in zip(row, low, high, strict=False)
        )

    exact_least = min(level(row, limit, max) for row, limit in planes)
    exact_most = max(level(row, limit, min) for row, limit in planes)
    assert Fraction(least) <= exact_least
    assert Fraction(most) >= exact_most
    size = max(
        abs(limit)
        + sum(
            max(abs(a * start), abs(a * stop))
            for a, start, stop in zip(row, low, high, strict=False)
        )
        for row, limit in planes
    )
    reach = (columns + 4) * 2 * math.ulp(max(size, 1e-300))
    assert float(exact_least - Fraction(least)) <= reach
    assert float(Fraction(most) - exact_most) <= reach


def test_a_span_allows_for_a_product_rounded_toward_its_plane():
    """0.1 times 5 rounds to 0.5, 2^-55 below the exact product: the plane 0.1 x <= 0.5 reads
    nothing at x = 5, where it reaches -2^-55 exactly, and -0.1 x <= -0.5 reads nothing there, where
    it reaches 2^-55. The least and the most read are past both."""
    off = Fraction(0.1) * 5 - Fraction(0.5)
    assert off == Fraction(1, 2**55)
    low, high = np.zeros(1), np.array([5.0])
    least, _ = allocation._spans(np.array([[0.1, 1.0]]), np.array([0.5]), low, high)
    _, most = allocation._spans(np.array([[-0.1, 1.0]]), np.array([-0.5]), low, high)
    assert Fraction(least) <= -off
    assert Fraction(most) >= off


def _cvar_program(gains, slopes, readings, budget):
    """A cvar program at the level 1 on planes ``gains + slopes @ s``, each of a reading in
    ``readings``, over two rates in [0, 1] spending ``budget``: the columns are the rates, eta and
    an excess a reading, eta free and each excess held above nothing, as :func:`cvar_allocate`
    writes them. Returns the program's parts and its least, found exactly over the rates'
    breakpoints."""
    count = max(readings) + 1
    objective = np.concatenate([np.zeros(2), [-float(count)], np.ones(count)])
    rows = np.zeros((len(gains), 3 + count))
    rows[:, :2], rows[:, 2] = -slopes, 1.0
    rows[np.arange(len(gains)), 3 + np.asarray(readings)] = -1.0
    equal = np.array([[1.0, 1.0, *np.zeros(1 + count)]])

    def worst(
        first: Fraction,
    ) -> Fraction:  # at the level 1 the least is less every reading's least plane
        second = Fraction(budget) - first
        lows = {}
        for g, (a, b), r in zip(gains, slopes, readings, strict=True):
            level = Fraction(g) + Fraction(a) * first + Fraction(b) * second
            lows[r] = min(lows.get(r, level), level)
        return -sum(lows.values())

    start, stop = max(Fraction(0), Fraction(budget) - 1), min(Fraction(1), Fraction(budget))
    points = {start, stop}
    for (g, (a, b)), (h, (c, d)) in combinations(zip(gains, slopes, strict=True), 2):
        if (a - b) != (c - d):  # where two planes cross along the budget
            first = (Fraction(h) - Fraction(g) + (Fraction(d) - Fraction(b)) * Fraction(budget)) / (
                Fraction(a) - Fraction(b) - Fraction(c) + Fraction(d)
            )
            if start <= first <= stop:
                points.add(first)
    return objective, rows, np.asarray(gains, dtype=float), equal, min(map(worst, points))


PLANES = (
    [0.0, 0.3, -0.2, 0.1, 0.4, -0.1],
    np.array([[1.0, 0.2], [0.1, 0.9], [0.8, 0.5], [0.3, 1.1], [0.2, 0.2], [1.2, 0.0]]),
    [0, 0, 1, 1, 2, 2],
)


def test_a_free_eta_and_excesses_held_above_nothing_are_read_in_their_implied_box():
    """With the planes' duals off the costs, eta's and an excess's reduced costs are off nothing,
    and in the program's bounds, eta free and each excess unbounded above, the bound is ``-inf``.
    Read within the box the planes imply, eta between the least and the most any plane reaches and
    each excess below their difference, it is finite and at or below the least."""
    gains, slopes, readings = PLANES
    objective, rows, limits, equal, least = _cvar_program(gains, slopes, readings, 1.0)
    bounds = [(0.0, 1.0), (0.0, 1.0), (None, None), *((0.0, None),) * 3]
    low, high = allocation._spans(rows, limits, np.zeros(2), np.ones(2))
    box = [(0.0, 1.0), (0.0, 1.0), (low, high), *((0.0, math.nextafter(high - low, math.inf)),) * 3]
    program = allocation._planes(objective, rows, limits, bounds, equal, [1.0], box)
    assert program.status == 0
    assert Fraction(program.fun) <= least
    assert float(least - Fraction(program.fun)) <= 1e-12
    matrix = np.vstack([rows, equal])
    sides = (
        np.concatenate([np.full(len(gains), -math.inf), [1.0]]),
        np.concatenate([limits, [1.0]]),
    )
    duals = np.concatenate([program.ineqlin.marginals * 1.001, program.eqlin.marginals])
    unbounded = (np.array([0.0, 0.0, -math.inf, *[0.0] * 3]), np.array([1.0, 1.0, *[math.inf] * 4]))
    assert allocation._least(objective, matrix, sides, unbounded, duals) == -math.inf
    implied = tuple(np.array(ends, dtype=float) for ends in zip(*box, strict=True))
    bound = allocation._least(objective, matrix, sides, implied, duals)
    assert -math.inf < bound
    assert Fraction(bound) <= least


MIXED = (
    (Channel(ONE, MichaelisMenten(10.0), 1000.0), Channel(ONE, MichaelisMenten(20.0), 300.0)),
    (Channel(ONE, MichaelisMenten(30.0), 500.0), Channel(ONE, Hill(5.0, 2.0), 700.0)),
    (Channel(ONE, MichaelisMenten(5.0), 200.0), Channel(ONE, MichaelisMenten(40.0), 900.0)),
)
PLANNERS = {
    "minimax": lambda: minimax_allocate(
        MIXED[:2], 60.0, 1, lower=np.zeros(2), upper=np.full(2, 60.0), history=np.zeros((0, 2))
    ),
    "cvar": lambda: cvar_allocate(
        MIXED,
        60.0,
        1,
        level=0.7,
        against=np.array([0.0, 60.0]),
        lower=np.zeros(2),
        upper=np.full(2, 60.0),
        history=np.zeros((0, 2)),
    ),
    "geos": lambda: allocate_geos(
        [MIXED[0], MIXED[2]], 20.0, 1, lower=np.zeros((2, 2)), upper=np.full((2, 2), 10.0)
    ),
}


@pytest.mark.parametrize("planner", PLANNERS)
def test_each_planner_reads_its_bound_in_a_box_that_holds_its_programs_solution(
    monkeypatch, planner
):
    """Each planner reads its programs' bounds in the box its planes imply, its levels bounded where
    the program leaves them free or open above. HiGHS's solution of each program is a least point
    and lies in that box, to HiGHS's tolerance, and the bound read there is finite."""
    real, solved = allocation._planes, []

    def recorded(objective, rows, limits, bounds, equal, totals, box=None, evened=None):
        program = real(objective, rows, limits, bounds, equal, totals, box, evened)
        if program.status == 0:
            solved.append((program, bounds if box is None else box))
        return program

    monkeypatch.setattr(allocation, "_planes", recorded)
    PLANNERS[planner]()
    assert solved
    for program, box in solved:
        low, high = (np.array(ends, dtype=float) for ends in zip(*box, strict=True))
        assert np.all(program.x >= low - 1e-9 * (1.0 + np.abs(low)))
        assert np.all(program.x <= high + 1e-9 * (1.0 + np.abs(high)))
        assert math.isfinite(program.fun)


@pytest.mark.parametrize("off", [1e-9, 1e-6, -1e-6])
def test_duals_evened_reading_by_reading_lose_the_masses_bias(monkeypatch, off):
    """Each reading's dual mass moved by its own share, as HiGHS's moved by up to 3e-7 on 400
    readings, lowers the bound by about that times the reach of eta's box; evened, so that no
    reading's mass passes its excess's cost and they come to eta's, the bound is back within
    rounding of the least, and :func:`_planes` keeps the higher of the two."""
    gains, slopes, readings = PLANES
    objective, rows, limits, equal, least = _cvar_program(gains, slopes, readings, 1.0)
    bounds = [(0.0, 1.0), (0.0, 1.0), (None, None), *((0.0, None),) * 3]
    low, high = allocation._spans(rows, limits, np.zeros(2), np.ones(2))
    box = [(0.0, 1.0), (0.0, 1.0), (low, high), *((0.0, math.nextafter(high - low, math.inf)),) * 3]
    program = allocation._planes(objective, rows, limits, bounds, equal, [1.0], box)
    spread = 1.0 + off * np.array([1.0, 1.0, -1.0, -1.0, 2.0, 2.0])
    moved = program.ineqlin.marginals * spread
    evened = allocation._evened(moved, np.array(readings), 3, 1.0, 3.0)
    mass = np.bincount(readings, weights=-evened, minlength=3)
    assert np.all(mass <= 1.0 + 1e-15)
    assert mass.sum() == pytest.approx(3.0, rel=1e-15, abs=0.0)
    matrix = np.vstack([rows, equal])
    sides = (
        np.concatenate([np.full(len(gains), -math.inf), [1.0]]),
        np.concatenate([limits, [1.0]]),
    )
    implied = tuple(np.array(ends, dtype=float) for ends in zip(*box, strict=True))
    raw = allocation._least(
        objective, matrix, sides, implied, np.concatenate([moved, program.eqlin.marginals])
    )
    even = allocation._least(
        objective, matrix, sides, implied, np.concatenate([evened, program.eqlin.marginals])
    )
    assert Fraction(raw) <= Fraction(even) <= least
    assert float(least - Fraction(raw)) > 0.1 * abs(off)
    assert float(least - Fraction(even)) <= 1e-12
    kept = allocation._planes(objective, rows, limits, bounds, equal, [1.0], box, lambda y: 0.0 * y)
    assert kept.fun == program.fun
    real = allocation.linprog

    def moving(*args, **kwargs):
        solved = real(*args, **kwargs)
        solved.ineqlin.marginals = solved.ineqlin.marginals * spread
        return solved

    monkeypatch.setattr(allocation, "linprog", moving)
    evening = allocation._planes(
        objective,
        rows,
        limits,
        bounds,
        equal,
        [1.0],
        box,
        lambda y: allocation._evened(y, np.array(readings), 3, 1.0, 3.0),
    )
    assert evening.fun == even


@pytest.mark.parametrize(
    ("duals", "total", "masses"),
    [
        ([-2.0, -0.5, -0.5], 2.0, [1.0, 0.5, 0.5]),
        ([-0.5, 0.0, -0.25], 1.5, [0.8, 0.0, 0.7]),
        ([-0.5, -0.5, -0.5], 1.2, [0.4, 0.4, 0.4]),
    ],
)
def test_evened_masses_keep_under_their_cap_and_come_to_the_total(duals, total, masses):
    """A plane a reading, the cap 1: a mass past the cap is clamped to it; masses short of the total
    are raised in proportion to their room below the cap, a reading with no mass given none; masses
    past the total are scaled down to it."""
    evened = allocation._evened(np.array(duals), np.arange(3), 3, 1.0, total)
    assert -evened == pytest.approx(masses, rel=1e-15, abs=0.0)


@settings(max_examples=300, deadline=None)
@given(
    value=st.floats(allow_nan=False, allow_infinity=False),
    factor=st.floats(1e-300, 1e300),
    divisor=st.just(1.0) | st.floats(1e-300, 1e300),
)
def test_a_bound_scaled_into_the_callers_units_stays_a_bound(value, factor, divisor):
    """Scaled from a program's units, a floor rounds down and a ceiling up, past the exact value;
    nothing stays nothing."""
    exact = Fraction(value) * Fraction(factor) / Fraction(divisor)
    down = allocation._outward(value, factor, divisor, -math.inf)
    up = allocation._outward(value, factor, divisor, math.inf)
    if value == 0.0:
        assert down == up == 0.0
    if math.isfinite(down):
        assert Fraction(down) <= exact
    if math.isfinite(up):
        assert Fraction(up) >= exact


def test_a_plan_with_nothing_to_gain_closes_on_its_first_program(monkeypatch):
    """A budget of nothing: every plan's worth is nothing, and so is the bound read from the first
    program's duals, exactly, and that closes the search, whose tolerance is a share of the bound.
    A bound a rounding below nothing would not close it before 500 rounds."""
    real, programs = allocation.linprog, []

    def counted(*args, **kwargs):
        programs.append(None)
        return real(*args, **kwargs)

    monkeypatch.setattr(allocation, "linprog", counted)
    cells = [
        [Channel(ONE, MichaelisMenten(10.0), 1000.0), Channel(ONE, MichaelisMenten(20.0), 300.0)],
        [Channel(ONE, MichaelisMenten(5.0), 10.0), Channel(ONE, Hill(5.0, 2.0), 700.0)],
    ]
    plan = allocate_geos(cells, 0.0, 1, lower=np.zeros((2, 2)), upper=np.full((2, 2), 10.0))
    assert plan.worth == plan.bound == 0.0
    assert len(programs) == 1
