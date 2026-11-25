"""The search's settings and the branch and bound over boxes of the channels' rates
on their concave envelopes, bisected on the budget's price."""

from __future__ import annotations

import heapq
import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chc._allocation.curves import _EPS, _bounded, _rates, _slope, _value, _Worth

if TYPE_CHECKING:
    from chc.allocation import SearchLimit, SearchStatus


# the cutting planes stop after this many rounds, the gap then reported as it stands; a plan on
# the envelopes from zero spend made with no settings judges its gap closed at this share of its
# bound
_GAP, _ROUNDS = 1e-9, 500


# a search's tolerance is never below this many epsilons of the dtype its curves are read in, of
# the scale its rtol is a share of: below it the gap is rounding, which no box closes
_ROUNDING = 64


@dataclass(frozen=True)
class _Settings:
    """A search's settings: it closes the gap to the larger of ``atol`` and ``rtol`` of a scale,
    but to no less than ``rounding`` epsilons of the dtype it reads, of the scale, and plans at
    most ``max_boxes`` boxes.

    Raises:
        ValueError: an ``rtol`` or ``atol`` that is negative or not finite, or a ``max_boxes``
            that is not a whole number of at least 1.
    """

    rtol: float
    atol: float
    max_boxes: int
    rounding: float = _ROUNDING

    def __post_init__(self) -> None:
        for name in ("rtol", "atol"):
            value = getattr(self, name)
            if not (np.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name}={value!r} is not a finite tolerance of at least 0")
            object.__setattr__(self, name, float(value))
        if not (isinstance(self.max_boxes, numbers.Integral) and self.max_boxes >= 1):
            raise ValueError(
                f"max_boxes={self.max_boxes!r} is not a whole number of boxes, at least 1"
            )
        object.__setattr__(self, "max_boxes", int(self.max_boxes))

    def tolerance(self, scale: float, eps: float) -> tuple[float, bool]:
        """The gap a search closes to on ``scale``, read in a dtype whose epsilon is ``eps``, and
        whether the floor raised it above the caller's."""
        asked = max(self.atol, self.rtol * scale)
        floor = self.rounding * eps * scale
        return max(asked, floor), floor > asked


def _rounding(worths: object) -> float:
    """The epsilon of the dtype ``worths``, a pytree of them, are read in."""
    leaves = jax.tree.leaves(eqx.filter(worths, eqx.is_inexact_array))
    return float(jnp.finfo(jnp.result_type(*leaves)).eps)


def _split(
    envelopes: Sequence[_Worth], lower: np.ndarray, upper: np.ndarray, periods: int, target: float
) -> tuple[np.ndarray, float]:
    """The rates in the box that spend ``target`` a period where every concave worth's slope meets
    ``periods`` times one price or presses on an end, and that price."""

    def rates(price: float) -> np.ndarray:
        return _rates(envelopes, lower, upper, periods, price)

    # Every rate falls as the price rises. At no price any rate is as good as its cap, since no
    # slope is negative; at the steepest slope the floors allow, every channel sits at its floor.
    # So the caps overspend and the floors underspend, and the bisection keeps it that way.
    cheap, many = 0.0, upper
    dear = max(0.0, *(_slope(w, low) for w, low in zip(envelopes, lower, strict=True)))
    dear, few = dear / periods, lower
    tolerance = 4 * _EPS * dear
    while dear - cheap > tolerance:
        middle = 0.5 * (cheap + dear)
        at = rates(middle)
        if at.sum() > target:
            many, cheap = at, middle
        else:
            few, dear = at, middle
    # both ends are best at a price within rounding of the other's, and so is any mix of them: the
    # one that spends the budget exactly is the plan, and it matters where a linear rate jumps
    return np.clip(_fill(few, many, target - float(few.sum())), lower, upper), 0.5 * (cheap + dear)


@dataclass(frozen=True)
class _Ending:
    """How a search ended: the boxes it planned, why it stopped and at what limit, the gap it
    closes to and whether the floor raised that, the scale its relative gap reads, and the
    iterations of each program HiGHS left unsolved."""

    boxes: int
    stopped: SearchStatus
    limit: SearchLimit | None
    tolerance: float
    floored: bool
    scale: float
    unsolved: tuple[int, ...] = ()


def _marginal(
    worths: Sequence[_Worth], spend: np.ndarray, upper: np.ndarray, periods: int
) -> float:
    """The return one more currency unit of budget buys on the curves at ``spend``: the steepest
    slope a period of a channel below its cap, where the unit returns the most, and 0 where every
    channel is at its cap. At a best plan every channel inside its box has that slope, one at its
    floor no steeper and one at its cap no shallower; where no channel is inside, the budget's
    price is an interval, and this is its lower end, the return of one more unit, not of one
    less."""
    slopes = (
        _slope(w, rate) for w, rate, cap in zip(worths, spend, upper, strict=True) if rate < cap
    )
    return max(slopes, default=0.0) / periods


def _branch_and_bound(  # noqa: C901, PLR0912, PLR0915
    worths: tuple[_Worth, ...],
    lower: np.ndarray,
    upper: np.ndarray,
    periods: int,
    target: float,
    settings: _Settings,
    charge: float | None = None,
    groups: tuple[tuple[int, ...], ...] = (),
    span: tuple[float, float] | None = None,
) -> tuple[np.ndarray, float, float, float, _Ending]:
    """The best split of ``target`` a period on the curves, its worth, a bound on every split's, its
    price, and how the search ended: Udell and Boyd's (2016) branch-and-bound for sums of S-shaped
    functions.

    A node is a box of the rates. Its bound is the split on each channel's envelope over the box
    (:func:`_bounded`), which no split in the box beats on the curves, and that split, read on the
    curves, is a plan. The node of the largest bound is split first, on the channel whose envelope
    stands furthest above its curve at the split, at its rate there, or halfway along its interval
    where the rate is an end of it; the envelopes of both halves meet the curve at the cut. Nodes
    whose bound is within the tolerance ``settings`` sets on the first node's bound of the best
    plan's worth are settled, and the bound returned is the largest left or settled. A node is
    split only where the count of nodes, with the halves that hold a split of the budget, stays
    within ``settings.max_boxes``.

    With a ``charge`` the budget is free and every currency unit spent is charged ``charge`` of
    return: a node's split is each channel's rate at that price on its envelope over the box, the
    worth and the bounds are read less the charge, and ``target`` is not read. The scale of the
    tolerance is still the first node's bound in worth. A ``span`` holds that free budget, what
    the rates spend over the periods, between its two ends: a box's split is then the rates at the
    charge where they spend inside it, and otherwise the split of the end they pass, which is the
    best in the box under the charge, since the envelopes' worth less the charge is concave in what
    they spend.

    The price is ``charge``, or else the best plan's own (:func:`_marginal`), read on the curves
    in the caller's box: a box's envelopes and its cut ends price the plan differently in each box
    that holds it, and a cut at the plan's own rate puts it in two.

    Each of ``groups`` holds rates whose worths and boxes are the same, so that any order of their
    values is as good as another: the search reads them in falling order. A cut of one at a rate
    holds the group's rates after it at most that rate in its lower half, and those before it at
    least that rate in its upper half; without that every order of a split would be bounded apart.
    """

    def solve(low: np.ndarray, high: np.ndarray, bounded: tuple[_Worth, ...]):
        if charge is None:
            spend = _split(bounded, low, high, periods, target)[0]
            spent = 0.0
        else:
            spend = _rates(bounded, low, high, periods, charge)
            if span is not None and not span[0] <= periods * float(spend.sum()) <= span[1]:
                end = span[0] if periods * float(spend.sum()) < span[0] else span[1]
                spend = _split(bounded, low, high, periods, end / periods)[0]
            spent = charge * periods * float(spend.sum())
        tops = np.array([_value(b, rate) for b, rate in zip(bounded, spend, strict=True)])
        values = np.array([_value(w, rate) for w, rate in zip(worths, spend, strict=True)])
        return spend, float(tops.sum()) - spent, float(values.sum()) - spent, tops - values

    bounded = tuple(_bounded(w, a, b) for w, a, b in zip(worths, lower, upper, strict=True))
    spend, bound, worth, gaps = solve(lower, upper, bounded)
    best = (worth, spend)
    charged = 0.0 if charge is None else charge * periods * float(spend.sum())
    scale = abs(bound + charged) or 1.0
    tolerance, floored = settings.tolerance(scale, _rounding(worths))
    heap = [(-bound, 0, lower, upper, bounded, spend, gaps)]
    settled, nodes = -np.inf, 1
    while heap and -heap[0][0] - best[0] > tolerance:
        _, _, low, high, bounded, spend, gaps = heap[0]
        cut_at = int(np.argmax(gaps))
        a, b = low[cut_at], high[cut_at]
        cut = spend[cut_at] if a < spend[cut_at] < b else a + 0.5 * (b - a)
        halves = []
        for floor, cap in ((a, cut), (cut, b)):
            child_low, child_high = low.copy(), high.copy()
            child_low[cut_at], child_high[cut_at] = floor, cap
            for group in groups:
                if cut_at in group:
                    place = group.index(cut_at)
                    before, after = list(group[:place]), list(group[place + 1 :])
                    child_low[before] = np.maximum(child_low[before], floor)
                    child_high[after] = np.minimum(child_high[after], cap)
            # a split of the budget fits, or the budget is free, within its span
            if charge is None:
                fits = float(child_low.sum()) <= target <= float(child_high.sum())
            else:
                fits = span is None or (
                    periods * float(child_low.sum()) <= span[1]
                    and periods * float(child_high.sum()) >= span[0]
                )
            if fits and np.all(child_low <= child_high):
                halves.append((floor, cap, child_low, child_high))
        if nodes + len(halves) > settings.max_boxes:
            break
        heapq.heappop(heap)
        for _, _, child_low, child_high in halves:
            child = list(bounded)
            moved = (child_low != low) | (child_high != high)
            for index in np.flatnonzero(moved):
                child[index] = _bounded(worths[index], child_low[index], child_high[index])
            rates, top, value, excess = solve(child_low, child_high, tuple(child))
            nodes += 1
            if value > best[0]:
                best = (value, rates)
            if top - best[0] > tolerance:
                heapq.heappush(
                    heap, (-top, nodes, child_low, child_high, tuple(child), rates, excess)
                )
            else:
                settled = max(settled, top)
    worth, spend = best
    price = _marginal(worths, spend, upper, periods) if charge is None else charge
    left = -heap[0][0] if heap else -np.inf
    bound = max(worth, settled, left)
    # a box was settled within the tolerance of a best plan that has only risen since, so a gap
    # left open is the open boxes', which only the cap left uncut
    closed = bound - worth <= tolerance
    ending = _Ending(
        boxes=nodes,
        stopped="closed" if closed else "cap",
        limit=None if closed else "max_boxes",
        tolerance=tolerance,
        floored=floored,
        scale=scale,
    )
    return spend, worth, bound, price, ending


def _fill(few: np.ndarray, many: np.ndarray, amount: float) -> np.ndarray:
    """``few`` raised toward ``many`` by ``amount`` in all, one rate after another.

    The rates either side of a price are both best there, and so is any mix of them. Where several
    jump at one price, on envelope chords of one slope, a mix that moved them together would leave
    each inside its chord, where the curve is below the envelope; filled in turn, at most one is.
    """
    steps = many - few
    before = np.concatenate([[0.0], np.cumsum(steps)[:-1]])
    return few + np.clip(amount - before, 0.0, steps)
