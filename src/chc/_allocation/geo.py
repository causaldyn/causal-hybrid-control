"""The layout of geos and channels under their totals, and its search: boxes of the
cells' rates, the prices that meet the totals, and the cutting planes over them."""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.optimize import OptimizeResult, brentq, linprog

from chc._allocation.curves import (
    _EPS,
    _boxed,
    _curvature,
    _National,
    _rate,
    _slope,
    _value,
    _value_and_slope,
    _Variable,
    _Worth,
    _worths,
)
from chc._allocation.planes import _UNSOLVED, _outward, _planes, _spans, _unit
from chc._allocation.search import _ROUNDS, _Ending, _rounding, _Settings
from chc.response import Channel

if TYPE_CHECKING:
    from chc.allocation import SearchLimit, Totals


@dataclass(frozen=True)
class _Group:
    """One total: the planned rates it sums, by index, each weighted by the share of it the total
    counts, and its bounds over the plan."""

    members: np.ndarray
    least: float
    most: float
    weights: np.ndarray


@dataclass(frozen=True)
class _Layout:
    """A grid of cells in one row, geo by geo, with each cell's history, as :func:`allocate_geos`
    accepts them; the rates its plans choose, every geo's own cell in the grid's order and then
    each national channel in its column's, with each rate's box and the totals' groups over them;
    and the settings its plans are searched to."""

    geos: int
    width: int
    cells: tuple[Channel, ...]
    history: np.ndarray
    owner: np.ndarray  # each cell's rate
    share: np.ndarray  # the share of its rate each cell spends
    national: int  # how many rates, the last, are national channels'
    lower: np.ndarray
    upper: np.ndarray
    geo_totals: Totals | None
    channel_totals: Totals | None
    groups: tuple[_Group, ...]  # the geos' totals, then the channels'
    settings: _Settings


def _variables(layout: _Layout, periods: int) -> tuple[_Variable, ...]:
    """The worth of each rate ``layout`` plans: a geo's own cell's, or a national channel's, its
    cells' each reading its share of the national spend."""
    worths = _worths(layout.cells, layout.history, periods)
    if not layout.national:
        return worths
    local = layout.lower.size - layout.national
    own = [worth for worth, rate in zip(worths, layout.owner, strict=True) if rate < local]
    shared = [
        _National(
            tuple(
                _Worth(worth.channel, worth.carry, float(share) * worth.reach)
                for worth, rate, share in zip(worths, layout.owner, layout.share, strict=True)
                if rate == k
            )
        )
        for k in range(local, layout.lower.size)
    ]
    return (*own, *shared)


@dataclass(frozen=True)
class _Planned:
    """One box of :func:`_geo_search`: its plan's rates and the prices it was found at; a bound on
    the worth less the charge of every plan in the box; the plan's worth less the charge on the
    curves, its worth and its charge; the bound in worth of every plan in the box spending as much;
    each cell's envelope over its curve at the plan; and what stopped the box's planes short of
    closing."""

    rates: np.ndarray
    prices: np.ndarray
    top: float
    value: float
    worth: float
    charged: float
    bound: float
    excess: np.ndarray
    limit: SearchLimit | None


def _geo_box(
    worths: tuple[_Variable, ...],
    bounded: tuple[_Variable, ...],
    low: np.ndarray,
    high: np.ndarray,
    groups: Sequence[_Group],
    periods: int,
    charge: float,
    close: Callable[[float], float],
    settle: float,
    searching: bool,
) -> _Planned:
    """The box ``[low, high]`` planned on ``bounded``, each cell's envelope over it: the cutting
    planes' plan (:func:`_outer`), made exact by Newton's method on the totals that bind where every
    free cell is strictly concave (:func:`_polish`), and read on the curves ``worths``. In a search
    Newton's method stops at its first step that does not shrink the totals' miss (``searching``),
    and the planes' plan stands."""
    rates, ceiling, prices, limit = _outer(
        bounded, low, high, groups, periods, charge, close, settle
    )
    exact = _polish(bounded, low, high, groups, periods, charge, prices, searching)
    if exact is None:
        # the program bounds the worth less the charge, so the worth of any plan spending as much
        charged = charge * periods * float(rates.sum())
        tops = [_value(w, r) for w, r in zip(bounded, rates, strict=True)]
        bound = max(ceiling + charged, sum(tops))
    else:
        rates, prices = exact
        charged = charge * periods * float(rates.sum())
        tops = [_value(w, r) for w, r in zip(bounded, rates, strict=True)]
        bound = sum(tops)
    values = [_value(w, r) for w, r in zip(worths, rates, strict=True)]
    worth = sum(values)
    excess = np.array(tops) - np.array(values)
    return _Planned(
        rates, prices, bound - charged, worth - charged, worth, charged, bound, excess, limit
    )


def _geo_search(
    worths: tuple[_Variable, ...],
    lower: np.ndarray,
    upper: np.ndarray,
    groups: Sequence[_Group],
    periods: int,
    charge: float,
    settings: _Settings,
    budget: bool,
) -> tuple[_Planned, float, _Ending]:
    """The best plan of the cells under ``groups``' totals, each currency unit it spends charged
    ``charge``; a bound on the worth of every plan spending as much; and how the search ended.
    Where ``budget``, the first group is the budget, fixed.

    Where every curve is concave one box is planned (:func:`_geo_box`), its planes closing to the
    tolerance ``settings`` sets on their bound. Where a curve is S-shaped the rates are searched in
    boxes, as :func:`allocate` searches them (Udell and Boyd 2016): a box's bound is the planes'
    on each cell's envelope over the box (:func:`_bounded`), concave in the rate and above the
    worth, so it bounds every plan in the box, and its plan, read on the curves, is a plan. Boxes
    are weighed by their worth less the charge. The box of the largest bound is cut first, on the
    cell inside its interval whose envelope stands furthest above its curve at the box's plan, at
    its rate there, so that both halves hold the plan and meet the totals; where no envelope stands
    above its curve, on the cell inside its interval whose interval is widest against its range.
    A box whose plan has no cell inside its interval is not cut, and its bound stands. The
    tolerance is the larger of ``settings``' ``atol`` and ``rtol`` of the first box's bound, but not
    below its rounding; a box is cut only where the count of boxes, with its two halves, stays
    within ``settings.max_boxes``, and each box's planes close to the tolerance or stop once the
    box's bound is within it of the best plan's. A searched plan's prices are read on the curves at
    the plan (:func:`_priced`), not on the envelopes of the box it was found in: a cut at the plan's
    own rate puts it in both halves, and each half's envelopes price it otherwise.
    """
    eps = _rounding(worths)
    bounded = tuple(_boxed(w, a, b) for w, a, b in zip(worths, lower, upper, strict=True))
    searching = any(b is not w for b, w in zip(bounded, worths, strict=True))
    first = _geo_box(
        worths,
        bounded,
        lower,
        upper,
        groups,
        periods,
        charge,
        lambda level: settings.tolerance(level, eps)[0],
        -math.inf,
        searching,
    )
    scale = abs(first.bound) or 1.0
    tolerance, floored = settings.tolerance(scale, eps)
    if not searching:
        closed = first.bound - first.worth <= tolerance
        ending = _Ending(
            boxes=1,
            stopped="closed" if closed else "cap",
            limit=None if closed else first.limit or "rounds",
            tolerance=tolerance,
            floored=floored,
            scale=scale,
        )
        return first, first.bound, ending
    best = first
    heap = [(-first.top, 0, lower, upper, bounded, first)]
    settled, stuck, boxes = -np.inf, -np.inf, 1
    stalled: SearchLimit | None = None
    widths = np.maximum(upper - lower, _EPS)
    while heap and -heap[0][0] - best.value > tolerance:
        _, _, low, high, boxed, node = heap[0]
        inside = (low < node.rates) & (node.rates < high)
        if not inside.any():
            # no cut holds the plan in both halves, and its envelopes meet the curves there: the
            # gap is its planes', which only more rounds close
            heapq.heappop(heap)
            stuck, stalled = max(stuck, node.top), node.limit or "rounds"
            continue
        excess = np.where(inside, node.excess, -np.inf)
        reach = np.where(inside, (high - low) / widths, -np.inf)
        cut_at = int(np.argmax(excess)) if np.max(excess) > 0.0 else int(np.argmax(reach))
        if boxes + 2 > settings.max_boxes:
            break
        heapq.heappop(heap)
        cut = float(node.rates[cut_at])
        for floor, cap in ((float(low[cut_at]), cut), (cut, float(high[cut_at]))):
            child_low, child_high = low.copy(), high.copy()
            child_low[cut_at], child_high[cut_at] = floor, cap
            child = list(boxed)
            child[cut_at] = _boxed(worths[cut_at], floor, cap)
            planned = _geo_box(
                worths,
                tuple(child),
                child_low,
                child_high,
                groups,
                periods,
                charge,
                lambda level: tolerance,
                best.value,
                True,
            )
            boxes += 1
            if planned.value > best.value:
                best = planned
            if planned.top - best.value > tolerance:
                heapq.heappush(
                    heap, (-planned.top, boxes, child_low, child_high, tuple(child), planned)
                )
            else:
                settled = max(settled, planned.top)
    left = -heap[0][0] if heap else -np.inf
    top = max(best.value, settled, stuck, left)
    # a box was settled within the tolerance of a best plan that has only risen since, so a gap
    # left open is a box the cap left uncut, or one whose planes stopped short
    closed = top - best.value <= tolerance
    limit: SearchLimit | None = (
        None if closed else "max_boxes" if left - best.value > tolerance else stalled
    )
    ending = _Ending(
        boxes=boxes,
        stopped="closed" if closed else "cap",
        limit=limit,
        tolerance=tolerance,
        floored=floored,
        scale=scale,
    )
    prices = _priced(worths, best.rates, lower, upper, groups, periods, charge, budget)
    return replace(best, prices=prices), top + best.charged, ending


def _priced(
    worths: Sequence[_Variable],
    rates: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    groups: Sequence[_Group],
    periods: int,
    charge: float,
    budget: bool,
) -> np.ndarray:
    """Each group's price for the plan ``rates``, read on the curves ``worths`` at the plan.

    A rate's price is ``charge`` plus its groups' prices, each weighted by the rate's share in the
    group. At a best plan a rate inside its box returns its price a currency unit, one at its floor
    no more and one at its cap no less; a group's price is not negative where its total is held at
    its most, not positive at its least, and nothing where the total is not held (Karush, Kuhn and
    Tucker). The prices are those that miss these conditions least, each miss weighted by how far
    the plan stands from the end of the box or the total that would excuse it: the complementary
    slackness gap, nothing at a plan that meets them. The totals' misses are made least first, so
    a total the plan does not hold is priced nothing though the rates, searched only to a
    tolerance, meet their prices only to it; the rates' misses after. Where ``budget`` the first
    group is the budget, and the misses of rates that return more than their price are made least
    before those that return less, so that no rate with room below its cap returns more than its
    price: as :func:`_marginal` reads :func:`allocate`'s price, one more currency unit returns no
    more than the price wherever it goes. A rate a rounding below its cap, where HiGHS can leave a
    rate it holds at its cap, has no room to weigh: a weight under about ``1e-9`` of the farthest a
    rate stands from an end of its box is taken as nothing (:func:`_lexicographic`). Where several
    prices leave the same gaps, the budget's is the one nearest nothing, and then the totals' are
    the least in the sum of their sizes: one geo with no totals is priced as :func:`allocate`
    prices its channels, and where every geo's total is fixed, which fixes the budget, the budget's
    price is nothing and theirs carry it.

    The program is written in a power of two of the steepest slope of a rate its box leaves free
    and of the farthest a rate stands from an end of its box, so the prices move with the units of
    currency and of return. A cell its box pins, as at nothing where its channel does not run,
    holds no row: a power of 0.2 from nothing stands at 2.9e12 a unit there, and as the scale it
    left every other slope under HiGHS's tolerance and the budget's price 0. A total's slack
    weighs at most that far: the totals' misses can always be nothing, so a slack only frees a
    price or holds it, and read on its own scale a total 1e12 short of binding made every rate's
    weight a rounding. Uncapped on the rates' scale, a slack past 1e20 of it is a cost HiGHS reads
    as infinite, and a cap 1e30 short of binding was priced 4.23 and the budget 0.
    """
    size, count = rates.size, len(groups)
    slopes = np.array([_slope(w, r) for w, r in zip(worths, rates, strict=True)]) / periods
    below, above = rates < upper, rates > lower
    # a rate its box pins holds no row, however steep its curve stands there
    unit = _unit(float(np.max(np.abs(slopes[below | above]), initial=abs(charge))))
    member = np.zeros((count, size))
    for k, group in enumerate(groups):
        member[k, group.members] = group.weights
    spent = periods * (member @ rates)
    fixed = np.array([group.least == group.most for group in groups], dtype=bool)
    most = np.array([group.most for group in groups], dtype=float)
    least = np.array([group.least for group in groups], dtype=float)
    raised = fixed | np.isfinite(most)  # a price that can be positive
    lowered = fixed | (least > 0.0)  # and one that can be negative
    # how far the plan stands from what would excuse each miss, in currency over the plan, read in
    # a power of two of the farthest a rate stands from an end of its box
    distances = periods * np.concatenate([rates - lower, upper - rates])
    scale = _unit(float(np.max(distances, initial=0.0)))
    slacks = np.concatenate(
        [
            np.where(fixed | ~np.isfinite(most), 0.0, np.maximum(most - spent, 0.0)),
            np.where(fixed, 0.0, np.maximum(spent - least, 0.0)),
        ]
    )
    weights = np.concatenate([np.minimum(slacks / scale, 1.0), distances / scale])
    # columns: each group's positive part, its negative part, each rate's price above its slope
    # past its floor, and below its slope short of its cap
    rows, limits = [], []
    eye = np.eye(size)
    for i in np.flatnonzero(below):
        rows.append(np.concatenate([-member[:, i], member[:, i], np.zeros(size), -eye[i]]))
        limits.append((charge - slopes[i]) / unit)
    for i in np.flatnonzero(above):
        rows.append(np.concatenate([member[:, i], -member[:, i], -eye[i], np.zeros(size)]))
        limits.append((slopes[i] - charge) / unit)
    open_ = (0.0, None)
    bounds = [
        *(open_ if free else (0.0, 0.0) for free in raised),
        *(open_ if free else (0.0, 0.0) for free in lowered),
        *(open_ if free else (0.0, 0.0) for free in above),
        *(open_ if free else (0.0, 0.0) for free in below),
    ]
    totals = np.concatenate([weights[: 2 * count], np.zeros(2 * size)])
    sizes = np.concatenate([np.ones(2 * count), np.zeros(2 * size)])
    objectives = [totals, weights - totals]
    if budget:
        # a rate's slope above its price first, weighted by its room below its cap
        short = np.concatenate([np.zeros(2 * count + size), weights[2 * count + size :]])
        objectives[1:] = [short, weights - totals - short]
        first = np.zeros(count)
        first[0] = 1.0
        objectives.append(np.concatenate([first, first, np.zeros(2 * size)]))
        sizes[[0, count]] = 0.0
    objectives.append(sizes)
    matrix = np.array(rows, dtype=float).reshape(len(rows), 2 * (count + size))
    x = _lexicographic(objectives, matrix, np.array(limits, dtype=float), bounds)
    return (x[:count] - x[count : 2 * count]) * unit


def _lexicographic(
    objectives: Sequence[np.ndarray],
    rows: np.ndarray,
    limits: np.ndarray,
    bounds: Sequence[tuple[float, float | None]],
) -> np.ndarray:
    """A point of ``rows @ x <= limits`` within ``bounds`` that is least in each objective in turn
    among the points least in those before it.

    Each program's optimal face bounds the next: every least point meets complementary slackness
    with any one optimal dual (Goldman and Tucker 1956), so it holds a column whose reduced cost is
    not nothing at the end that cost presses on, and a row whose dual is not nothing at its limit.
    The objectives' weights are below 2, and a reduced cost or a dual within ``1e-9`` is taken as
    nothing: a column it would hold is left free, so a later objective may move an earlier one by
    about that much a unit the column moves.

    Each program runs under the planes' limit of ten iterations a row and a column
    (:func:`_planes`), and one it stops raises. Weights down to roundings put the costs in HiGHS's
    scaling, and at this tolerance HiGHS can then pivot on a degenerate program without end. Of
    2,668 programs, on plans of the tests and on random caps of two S-curve geos, the most any
    took was 0.14 iterations a row and a column.
    """
    bounds = list(bounds)
    held = np.zeros(len(limits), dtype=bool)
    x = np.zeros(len(bounds))
    for cost in objectives:
        loose, tight = rows[~held], rows[held]
        program = linprog(
            cost,
            A_ub=loose if len(loose) else None,
            b_ub=limits[~held] if len(loose) else None,
            A_eq=tight if len(tight) else None,
            b_eq=limits[held] if len(tight) else None,
            bounds=bounds,
            method="highs",
            options={
                "primal_feasibility_tolerance": 1e-10,
                "dual_feasibility_tolerance": 1e-10,
                "maxiter": 10 * (len(limits) + len(bounds)),
            },
        )
        if program.status != 0:
            raise RuntimeError(f"the prices' linear program failed: {program.message}")
        x = program.x
        floor = 1e-9
        for column, (low, high) in enumerate(bounds):
            if program.lower.marginals[column] > floor:
                bounds[column] = (low, low)
            elif high is not None and program.upper.marginals[column] < -floor:
                bounds[column] = (high, high)
        if len(loose):
            held[np.flatnonzero(~held)[program.ineqlin.marginals < -floor]] = True
    return x


def _reach(layout: _Layout, periods: int) -> tuple[float, float]:
    """The least and the most a plan within the boxes and the totals spends over its periods.

    Raises:
        ValueError: where no plan meets the boxes and the totals together.
    """
    least, most = periods * float(layout.lower.sum()), periods * float(layout.upper.sum())
    size = layout.lower.size
    unit = float(np.max(layout.upper)) or 1.0
    per = periods * unit
    rows: list[np.ndarray] = []
    limits: list[float] = []
    for group in layout.groups:
        member = np.zeros(size)
        member[group.members] = group.weights
        if np.isfinite(group.most):
            rows.append(member)
            limits.append(group.most / per)
        if group.least > 0.0:
            rows.append(-member)
            limits.append(-group.least / per)
    if not rows:
        return least, most
    ends = []
    for sense in (1.0, -1.0):
        program = linprog(
            np.full(size, sense),
            A_ub=np.array(rows),
            b_ub=np.array(limits),
            bounds=list(zip(layout.lower / unit, layout.upper / unit, strict=True)),
            method="highs",
        )
        if program.status == 2:
            raise ValueError("no plan meets the boxes and the totals together")
        if program.status != 0:
            raise RuntimeError(f"the budgets' linear program failed: {program.message}")
        ends.append(per * float(program.x.sum()))
    # a vertex's rates are the boxes' and the totals' own numbers, to rounding
    return float(np.clip(ends[0], least, most)), float(np.clip(ends[1], least, most))


def _outer(  # noqa: C901, PLR0912, PLR0915
    envelopes: Sequence[_Variable],
    lower: np.ndarray,
    upper: np.ndarray,
    groups: Sequence[_Group],
    periods: int,
    charge: float,
    close: Callable[[float], float],
    settle: float = -math.inf,
) -> tuple[np.ndarray, float, np.ndarray, SearchLimit | None]:
    """The best plan on the envelopes by cutting planes (Kelley 1960), each currency unit it
    spends charged ``charge`` of its worth.

    Each cell's worth is bounded above by its value at its cap and by its tangents; the linear
    program maximises the bounds less the charge over the plans that meet the constraints, and each
    plan it proposes adds a tangent where a cell's bound is loose. The bound each program's duals
    give bounds every plan from above; the plans it proposes approach it from below. They stop once
    the bound is within ``close`` of the best plan's worth less the charge, ``close`` read on the
    bound's level in worth, or of ``settle``, a worth less the charge some other plan reaches.
    Returns the best plan's rates, the least of those bounds, each group's price, the last
    program's duals in return a currency unit, and what stopped the planes short of closing,
    ``rounds`` or ``unsolved``, or None.
    """
    size = len(envelopes)
    # rates are read in units of the largest cap, and worths in units of every cap's together:
    # HiGHS works to absolute tolerances, and a channel's worth in currency can pass 1e9
    unit = float(np.max(upper)) or 1.0
    caps = [_value(w, hi) for w, hi in zip(envelopes, upper, strict=True)]
    scale = float(np.sum(np.abs(caps))) or 1.0
    rows: list[int] = []
    columns: list[int] = []
    entries: list[float] = []
    limits: list[float] = []

    def bound(cell: int, slope: float, height: float) -> None:
        """The cell's height at most ``height + slope * rate``, in currency and worth."""
        rows.extend((len(limits), len(limits)))
        columns.extend((size + cell, cell))
        entries.extend((1.0, -slope * unit / scale))
        limits.append(height / scale)

    def tangent(cell: int, at: float) -> tuple[float, float]:
        value, slope = _value_and_slope(envelopes[cell], jnp.asarray(at, dtype=float))
        return float(value), float(slope)

    for cell in range(size):
        bound(cell, 0.0, caps[cell])  # an increasing worth is at most its cap's
        for at in {float(lower[cell]), 0.5 * float(lower[cell] + upper[cell]), float(upper[cell])}:
            value, slope = tangent(cell, at)
            if np.isfinite(slope):
                bound(cell, slope, value - slope * at)

    per = periods * unit
    sides: list[tuple[int, float]] = []  # each inequality row's group, +1 at its most, -1 its least
    side_rows: list[np.ndarray] = []
    side_limits: list[float] = []
    fixed: list[int] = []
    fixed_rows: list[np.ndarray] = []
    fixed_limits: list[float] = []
    for index, group in enumerate(groups):
        member = np.zeros(size)
        member[group.members] = group.weights
        if group.least == group.most:
            fixed.append(index)
            fixed_rows.append(member)
            fixed_limits.append(group.least / per)
            continue
        if np.isfinite(group.most):
            sides.append((index, 1.0))
            side_rows.append(member)
            side_limits.append(group.most / per)
        if group.least > 0.0:
            sides.append((index, -1.0))
            side_rows.append(-member)
            side_limits.append(-group.least / per)
    totals = sparse.csr_array(
        np.hstack([np.array(side_rows).reshape(len(sides), size), np.zeros((len(sides), size))])
    )
    equal = sparse.csr_array(
        np.hstack([np.array(fixed_rows).reshape(len(fixed), size), np.zeros((len(fixed), size))])
    )
    objective = np.concatenate([np.full(size, charge * per / scale), -np.ones(size)])
    box = [*zip(lower / unit, upper / unit, strict=True), *[(None, None)] * size]

    best, plan, ceiling = -np.inf, lower.copy(), np.inf
    priced: tuple[OptimizeResult, int] | None = None
    limit: SearchLimit | None = "rounds"
    for _ in range(_ROUNDS):
        solved = len(limits)
        cuts = sparse.coo_array((entries, (rows, columns)), shape=(solved, 2 * size))
        least, most = _spans(cuts, np.array(limits), lower / unit, upper / unit)
        program = _planes(
            objective,
            sparse.vstack([cuts, totals]),
            np.array([*limits, *side_limits]),
            box,
            equal if fixed else None,
            np.array(fixed_limits) if fixed else None,
            # a best height is its cell's least bound at the plan's rate
            [*box[:size], *((least, most),) * size],
        )
        if program.status == 2:
            raise ValueError("no plan meets the budget, the boxes and the totals together")
        if program.status in _UNSOLVED and priced is not None:
            program, solved = priced  # the last program's bound and prices stand
            limit = "unsolved"
            break
        if program.status != 0:
            raise RuntimeError(f"the plan's linear program failed: {program.message}")
        priced = (program, solved)
        ceiling = min(ceiling, _outward(-float(program.fun), scale, 1.0, math.inf))
        rates = np.clip(program.x[:size] * unit, lower, upper)
        heights = program.x[size:] * scale
        pairs = [tangent(cell, float(r)) for cell, r in enumerate(rates)]
        values = np.array([value for value, _ in pairs])
        charged = charge * periods * float(rates.sum())
        if values.sum() - charged > best:
            best, plan = float(values.sum()) - charged, rates
        level = abs(ceiling + charged)  # a share of the worth: less the charge it can be nothing
        gap = close(level)
        if ceiling - max(best, settle) <= gap:
            limit = None
            break
        for cell in np.flatnonzero(heights - values > gap / size):
            value, slope = pairs[cell]
            if np.isfinite(slope):
                bound(int(cell), slope, value - slope * float(rates[cell]))
    duals = -scale / per
    prices = np.zeros(len(groups))
    for (index, sign), marginal in zip(sides, program.ineqlin.marginals[solved:], strict=True):
        prices[index] += sign * float(marginal) * duals
    for index, marginal in zip(fixed, program.eqlin.marginals, strict=True):
        prices[index] = float(marginal) * duals
    return plan, ceiling, prices, limit


def _polish(
    envelopes: Sequence[_Variable],
    lower: np.ndarray,
    upper: np.ndarray,
    groups: Sequence[_Group],
    periods: int,
    charge: float,
    prices: np.ndarray,
    searching: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """The plan made exact on the totals that bind, a fixed budget among them, or None.

    A cell's price is the charge on spend plus those of the binding totals it is in, and its rate
    is exact at it: where its slope meets ``periods`` times the price, or the end of its box the
    slope presses on. Newton's method on the binding totals' prices closes those totals, starting
    from the totals and prices the cutting planes found. A binding total whose
    price comes out on the wrong side is released, and a free total the plan breaks is bound where
    it breaks, until neither happens: the plan then meets the conditions for the best on concave
    worths. None where a free cell is not strictly concave, Newton's method does not close, or the
    binding totals keep changing; and, ``searching``, at Newton's first step that does not shrink
    the miss (:func:`_close`).
    """
    size = len(envelopes)
    # each binding total's side: 1 held at its most, -1 at its least, 0 fixed
    sides = {
        k: 0.0 if group.least == group.most else float(np.sign(prices[k]))
        for k, group in enumerate(groups)
        if group.least == group.most or prices[k] != 0.0
    }
    full = prices.copy()
    tolerance = 16 * _EPS * size * float(np.max(upper))
    margin = periods * tolerance
    for _ in range(2 * len(groups) + 1):  # every round but the last changes the binding totals
        binding = list(sides)
        targets = np.array(
            [(groups[k].most if sides[k] >= 0.0 else groups[k].least) / periods for k in binding]
        )
        membership = np.zeros((len(binding), size))
        for row, k in enumerate(binding):
            membership[row, groups[k].members] = groups[k].weights
        closed = _close(
            envelopes,
            lower,
            upper,
            membership,
            targets,
            periods,
            charge,
            full[binding],
            tolerance,
            searching,
        )
        if closed is None:
            return None
        unknown, exact = closed
        full = np.zeros(len(groups))
        full[binding] = unknown
        floor = -1e-9 * float(np.max(np.abs(unknown), initial=charge))
        # held at its most and better spending less, or at its least and better spending more
        wrong = [k for k in binding if sides[k] * full[k] < floor]
        spent = np.array(
            [periods * (exact[group.members] * group.weights).sum() for group in groups]
        )
        broken = {
            k: 1.0 if spent[k] > group.most else -1.0
            for k, group in enumerate(groups)
            if k not in sides and not group.least - margin <= spent[k] <= group.most + margin
        }
        if not wrong and not broken:
            return exact, full
        for k in wrong:
            del sides[k]
        sides |= broken
    return None


def _close(  # noqa: C901
    envelopes: Sequence[_Variable],
    lower: np.ndarray,
    upper: np.ndarray,
    membership: np.ndarray,
    targets: np.ndarray,
    periods: int,
    charge: float,
    unknown: np.ndarray,
    tolerance: float,
    searching: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """The prices, from ``unknown``, at which the cells' rates sum through ``membership`` to
    ``targets``, each cell's price ``charge`` plus its rows'; and those rates. None where a free
    cell is not strictly concave or the prices do not close, and, ``searching``, at the first
    Newton step that does not shrink the residual.

    Each step is Newton's on the prices, the free cells' curvatures its Jacobian. The residual is,
    up to sign and scale, the gradient of the dual, convex in the prices, so along any direction
    its projection on the direction falls as the step grows, and the dual is least where that
    projection reaches nothing. A Newton step that shrinks the residual is taken whole; one that
    does not is cut or stretched to that least. A cell held at an end of its box adds nothing to
    the Jacobian, so where the free cells leave a price open the direction is taken with a little
    damping, as Levenberg and Marquardt damp it, and the search along it moves that price as far
    as it takes a held cell off its end.
    """

    def respond(prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        at = periods * (charge + membership.T @ prices)
        rates = np.array(
            [
                _rate(w, lo, hi, float(t))
                for w, lo, hi, t in zip(envelopes, lower, upper, at, strict=True)
            ]
        )
        return rates, membership @ rates - targets

    rates, residual = respond(unknown)
    for _ in range(100):
        if np.all(np.abs(residual) <= tolerance):
            return unknown, rates
        free = (lower < rates) & (rates < upper)
        bends = np.array(
            [
                float(_curvature(w, jnp.asarray(float(r), dtype=float))) if inside else -1.0
                for w, r, inside in zip(envelopes, rates, free, strict=True)
            ]
        )
        if not np.all(bends < 0.0):
            return None  # a straight stretch: its rate does not follow its price
        jacobian = (membership * np.where(free, periods / bends, 0.0)) @ membership.T
        direction, _, rank, _ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        if rank < len(jacobian):
            scale = float(np.max(np.abs(jacobian))) or 1.0
            damped = jacobian - 1e-9 * scale * np.eye(len(jacobian))
            direction = np.linalg.solve(damped, -residual)
        trial_rates, trial_residual = respond(unknown + direction)
        if np.linalg.norm(trial_residual) < np.linalg.norm(residual):
            unknown, rates, residual = unknown + direction, trial_rates, trial_residual
            continue
        if searching:
            # an envelope's chord holds its cell at one end of it at every price but its slope, and
            # at that price the rate jumps across it: a search along the step would circle the jump
            return None
        if direction @ residual <= 0.0:
            return None  # rounding has turned the step uphill

        def along(
            step: float, start: np.ndarray = unknown, direction: np.ndarray = direction
        ) -> float:
            return float(direction @ respond(start + step * direction)[1])

        far = 1.0
        while along(far) > 0.0:
            far *= 2.0
            if far > 2.0**60:
                return None  # the dual falls without end along it
        step = brentq(along, 0.0, far, xtol=1e-12 * far, rtol=4 * _EPS)
        unknown = unknown + step * direction
        rates, residual = respond(unknown)
    return None
