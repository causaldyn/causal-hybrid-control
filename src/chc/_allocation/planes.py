"""The cutting planes: the linear programs over the planes cut so far, read to a
bound that holds through HiGHS's rounding, and the projection onto the budget's box."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import OptimizeResult, linprog

from chc._allocation.curves import _EPS

_TINY = float(np.finfo(float).smallest_subnormal)


_HUGE = 2.0**300


# linprog's statuses for a program HiGHS leaves unsolved, stopped by _planes' iteration limit or
# ended with its optimality conditions unmet (HiGHS's "Unknown"): it bounds nothing, and the
# planners keep the last program's bound
_UNSOLVED = (1, 4)


def _planes(
    objective: np.ndarray,
    rows: np.ndarray | sparse.sparray,
    limits: np.ndarray,
    bounds: Sequence[tuple[float | None, float | None]],
    equal: np.ndarray | sparse.sparray | None,
    totals: np.ndarray | Sequence[float] | None,
    box: Sequence[tuple[float | None, float | None]] | None = None,
    evened: Callable[[np.ndarray], np.ndarray] | None = None,
) -> OptimizeResult:
    """HiGHS's solution of a cutting-plane program, ``rows @ x <= limits`` and ``equal @ x =
    totals`` within ``bounds``, at the tolerance ``1e-10`` the planes' bound is read to, and under
    an iteration limit: status 1 where the limit stopped it, 4 where HiGHS ended it unsolved.

    A solved program's ``fun`` is not HiGHS's objective but the bound :func:`_least` reads from
    HiGHS's duals, below every point of the program as written, within ``box``: a box that holds
    a least point of the program, ``bounds`` where none is given. Where ``evened`` is given it is
    also read from the duals ``evened`` makes of the planes', and the higher bound kept. HiGHS's
    objective is its basis's value worked in floating point: on the ill-conditioned bases of a
    search over 400 readings it stood off that value by up to 1e-7 of the program's units, above
    it in 47 of 87 programs, and in 10 above the least of the program, by up to 9e-8. HiGHS also
    leaves out of the program it solves every entry of at most 1e-9, its ``small_matrix_value``,
    which moves the program by up to that times its column's range.

    HiGHS can pivot without end at that tolerance: where the solution of its scaled program misses
    the tolerance once unscaled, it solves the unscaled program again with its costs unperturbed,
    and on a degenerate program that solve can cycle. :func:`cvar_allocate`'s programs at the
    level 1 did, one past eleven million iterations, in HiGHS 1.12.0, which SciPy 1.18 ships, and
    1.15.1 alike, until their costs were kept out of HiGHS's scaling. The limit is ten iterations a
    row and a column, where the most any program took, over the tests and on 400 readings, was 0.8,
    so one stopped there has stalled; its caller keeps the bound its last program gave, which the
    planes added since could only have tightened."""
    count = len(limits) + (0 if totals is None else len(totals))
    program = linprog(
        objective,
        A_ub=rows,
        b_ub=limits,
        A_eq=equal,
        b_eq=totals,
        bounds=bounds,
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-10,
            "dual_feasibility_tolerance": 1e-10,
            "maxiter": 10 * (count + objective.size),
        },
    )
    if program.status != 0:
        return program  # SciPy keeps no duals for a program HiGHS did not solve
    matrix, fixed = sparse.csr_array(rows), np.zeros(0)
    if equal is not None and totals is not None:
        matrix = sparse.vstack([matrix, sparse.csr_array(equal)], format="csr")
        fixed = np.asarray(totals, dtype=float)
    columns = bounds if box is None else box
    sides = (
        np.concatenate([np.full(len(limits), -np.inf), fixed]),
        np.concatenate([limits, fixed]),
    )
    ends = (
        np.array([-np.inf if low is None else low for low, _ in columns], dtype=float),
        np.array([np.inf if high is None else high for _, high in columns], dtype=float),
    )
    planes = program.ineqlin.marginals
    program.fun = max(
        _least(objective, matrix, sides, ends, np.concatenate([duals, program.eqlin.marginals]))
        for duals in ([planes] if evened is None else [planes, evened(planes)])
    )
    return program


def _outward(value: float, factor: float, divisor: float, toward: float) -> float:
    """``value * factor / divisor`` for positive ``factor`` and ``divisor``, each rounding moved a
    place past the exact value toward ``toward``, so a bound read in a program's units stays one in
    the caller's; nothing stays nothing, which a search whose tolerance is a share of its bound
    needs to close there. A power of two scales exactly unless the result leaves the normal
    doubles, which scaling it back tells, and an exact scaling is not moved."""
    if value == 0.0:
        return 0.0
    scaled = value * factor
    if not (math.frexp(factor)[0] == 0.5 and scaled / factor == value):
        scaled = math.nextafter(scaled, toward)
    quotient = scaled / divisor
    if not (math.frexp(divisor)[0] == 0.5 and quotient * divisor == scaled):
        quotient = math.nextafter(quotient, toward)
    return quotient


def _unit(amount: float) -> float:
    """The power of two at or below ``amount``, or 1 for nothing: a program written in it holds the
    caller's numbers each scaled exactly, unless one leaves the normal doubles."""
    return math.ldexp(1.0, math.frexp(amount)[1] - 1) if amount > 0.0 else 1.0


def _least(
    objective: np.ndarray,
    matrix: np.ndarray | sparse.sparray,
    sides: tuple[np.ndarray, np.ndarray],
    box: tuple[np.ndarray, np.ndarray],
    duals: np.ndarray,
) -> float:
    """A bound ``objective @ x`` never falls below where ``sides[0] <= matrix @ x <= sides[1]``
    and ``box[0] <= x <= box[1]``, read from any row ``duals`` (Neumaier and Shcherbina 2004;
    Jansson 2004).

    For any ``y``, ``objective @ x = y @ (matrix @ x) + d @ x`` with ``d = objective - matrix.T @
    y``. Over the program each row's term is at least its dual times the side its sign presses on,
    and each column's at least its reduced cost times the end of its box that cost presses on, so
    the bound holds whatever the duals are, a dual whose side is infinite read as nothing. The
    bound is worked exactly and rounded down: each product is split into two doubles that sum to
    it (Dekker 1971), and each reduced cost's sign, and the bound, are read off :func:`math.fsum`
    of the parts, which rounds their exact sum correctly. So a bound of nothing comes out nothing,
    as a search whose tolerance is a share of its bound needs to close there. Where a product's
    factors are too small for its split to be exact its rounding is allowed for instead. The bound
    is ``-inf`` where a reduced cost presses on an infinite end, or where a number it multiplies
    passes ``2**300``, past which the split could overflow.
    """
    least, most = sides
    low, high = box
    y = np.where(np.isfinite(duals), duals, 0.0)
    y = np.where(((y > 0.0) & np.isfinite(least)) | ((y < 0.0) & np.isfinite(most)), y, 0.0)
    used = np.flatnonzero(y)
    weights = y[used]
    side = np.where(weights > 0.0, least[used], most[used])
    rows = sparse.csc_array(sparse.csr_array(matrix)[used])
    size = objective.size
    if (
        max(np.abs(part).max(initial=0.0) for part in (objective, rows.data, weights, side))
        >= _HUGE
    ):
        return -np.inf
    # each column's reduced cost in parts: its cost, and less the split product of each of its
    # entries with that row's dual
    column = np.repeat(np.arange(size), np.diff(rows.indptr))
    upper, lower, miss = _product(rows.data, weights[rows.indices])
    parts = np.concatenate([objective, -upper, -lower])
    owner = np.concatenate([np.arange(size), column, column])
    values = parts[np.argsort(owner, kind="stable")].tolist()
    stops = np.cumsum(np.bincount(owner, minlength=size)).tolist()
    reduced = np.array(
        [math.fsum(values[a:b]) for a, b in zip([0, *stops[:-1]], stops, strict=True)]
    )
    end = np.where(reduced > 0.0, low, high)
    pressed = reduced != 0.0
    # where a split was not exact the reduced cost may stand off its parts' sum by the misses,
    # which moves its column's term by at most their sum times the farthest end of its box
    off = np.bincount(column, weights=miss, minlength=size)
    fragile = off > 0.0
    reach = np.maximum(np.abs(low[fragile]), np.abs(high[fragile]))
    if np.any(pressed & ~(np.abs(end) < _HUGE)) or not np.all(np.isfinite(reach)):
        return -np.inf
    # the misses' sum, rounded at most once an entry, raised past its exact value
    slack = np.nextafter(off[fragile] * (1.0 + 2.0**-30) * reach, np.inf)
    taken = pressed[owner]
    upper_term, lower_term, miss_term = _product(parts[taken], end[owner[taken]])
    upper_row, lower_row, miss_row = _product(weights, side)
    terms = np.concatenate(
        [upper_term, lower_term, upper_row, lower_row, -miss_term, -miss_row, -slack]
    ).tolist()
    total = math.fsum(terms)
    # the sum rounded to nearest, and where that is above the terms' exact sum the double below
    return math.nextafter(total, -math.inf) if math.fsum([*terms, -total]) < 0.0 else total


def _product(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each ``a * b`` as its rounding and the rest, two doubles whose sum is the exact product, and
    what that pair can miss the product by.

    The split is Dekker's (1971): each factor is cut into halves whose products are exact, and the
    rest is worked from them. :func:`_least` keeps the factors under ``2**600`` and the products
    under ``2**900``, so nothing overflows, and the rest is exact where a factor is nothing or
    their exponents come to -900 or more together, for it then has no bit below the least normal
    double. Elsewhere the rest is taken as nothing and the miss covers the rounding: a share
    ``2**-52`` of the product, and the least subnormal for an underflow."""
    upper = a * b
    halves = []
    for factor in (a, b):
        cut = 134217729.0 * factor  # 2**27 + 1 leaves each half 26 bits
        top = cut - (cut - factor)
        halves.append((top, factor - top))
    (a_top, a_rest), (b_top, b_rest) = halves
    lower = a_rest * b_rest - (((upper - a_top * b_top) - a_rest * b_top) - a_top * b_rest)
    exact = (a == 0.0) | (b == 0.0) | (np.frexp(a)[1] + np.frexp(b)[1] >= -900)
    return upper, np.where(exact, lower, 0.0), np.where(exact, 0.0, np.abs(upper) * _EPS + _TINY)


def _spans(
    rows: np.ndarray | sparse.sparray, limits: np.ndarray, low: np.ndarray, high: np.ndarray
) -> tuple[float, float]:
    """The least and the most any of ``rows`` reaches over the box ``low <= x <= high`` of their
    first ``low.size`` columns, each read as its limit less its product with ``x`` there, rounded
    outward.

    A cutting-plane program's other columns are its levels, free or held above nothing, which the
    planes bound: at some least point each lies between these two, or below the most by no more
    than the most less the least, and :func:`_least` reads a program within such a box."""
    part = sparse.csr_array(rows)[:, : low.size]
    columns = part.indices
    at_low, at_high = part.data * low[columns], part.data * high[columns]

    def summed(values: np.ndarray) -> np.ndarray:
        return sparse.csr_array((values, columns, part.indptr), shape=part.shape).sum(axis=1)

    top = summed(np.maximum(at_low, at_high))
    bottom = summed(np.minimum(at_low, at_high))
    # as _least's reduced costs: a unit in the last place of the magnitudes for each entry, and two
    # more for the limit's subtraction and the magnitudes' own rounding
    count = np.diff(part.indptr) + 2
    size = np.abs(limits) + summed(np.maximum(np.abs(at_low), np.abs(at_high)))
    error = count * _EPS * size + count * _TINY
    least = np.nextafter(np.min(limits - top - error), -np.inf)
    most = np.nextafter(np.max(limits - bottom + error), np.inf)
    return float(least), float(most)


def _onto(split: np.ndarray, lower: np.ndarray, upper: np.ndarray, rate: float) -> np.ndarray:
    """The nearest split to ``split`` in the box whose rates sum to ``rate``: every rate shifted
    by one amount and clipped to its box, the amount where the sum is ``rate``."""

    # the sum is linear in the amount between the amounts that take a rate to its floor or its cap
    shifts = np.unique(np.concatenate([lower - split, upper - split]))
    sums = np.clip(split + shifts[:, None], lower, upper).sum(axis=1)
    if sums[0] >= rate:
        return lower.copy()
    if sums[-1] <= rate:
        return upper.copy()
    # the amount is read off the first piece whose end reaches the rate, as a share of the piece:
    # a root finder's step there multiplies two of its lengths, and below about 1e-157 that product
    # underflows to zero, which kept brentq stepping by its tolerance past its hundred iterations
    end = int(np.searchsorted(sums, rate))
    share = (rate - sums[end - 1]) / (sums[end] - sums[end - 1])
    shift = shifts[end - 1] + share * (shifts[end] - shifts[end - 1])
    return np.clip(split + shift, lower, upper)
