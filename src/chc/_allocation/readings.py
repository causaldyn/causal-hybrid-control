"""The search over several readings of the channels: the stacks of their curves,
the cutting planes over the readings' regrets, and the worst share's weights."""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from scipy import sparse

from chc._allocation.curves import _EPS, _Bounded, _chord, _Worth
from chc._allocation.planes import _UNSOLVED, _onto, _outward, _planes, _spans, _unit
from chc._allocation.search import _ROUNDS, _Ending, _rounding, _Settings
from chc.response import Saturation, _bisected_touches

if TYPE_CHECKING:
    from chc.allocation import SearchLimit


# in a box of cvar_allocate's search the cutting planes stop after this many rounds, or once their
# bound is within half the box's gap to the best split of the box's best on its envelopes: a box is
# cut sooner than its planes are refined
_BOX_ROUNDS = 30


def _reading_weights(weights: ArrayLike, count: int) -> np.ndarray:
    """``weights``, one a reading of ``count``, finite, none below 0 and not all 0, scaled to sum
    to 1: first by their largest, so that no sum of them overflows."""
    weighed = np.asarray(weights, dtype=float)
    if weighed.shape != (count,):
        raise ValueError(f"weights has shape {weighed.shape}; it needs one a reading, {count}")
    if not np.all(np.isfinite(weighed)):
        raise ValueError(f"weights {weighed} are not all finite")
    if np.any(weighed < 0.0):
        raise ValueError(f"weights {weighed} has a negative weight; a reading weighs 0 or more")
    if not np.any(weighed > 0.0):
        raise ValueError("weights are all 0; at least one reading must weigh something")
    weighed = weighed / weighed.max()
    return weighed / weighed.sum()


@dataclass(frozen=True)
class _Stack:
    """One channel's worths under every reading, in groups whose worths share a structure, each
    group stacked on a leading axis so that one compiled program reads or bounds all of it."""

    members: tuple[np.ndarray, ...]  # the readings in each group
    worths: tuple[_Worth, ...]  # each group's worths, stacked
    bends: tuple[np.ndarray | None, ...]  # each group's inflections; None where none is S-shaped


def _stacks(readings: Sequence[tuple[_Worth, ...]]) -> tuple[_Stack, ...]:
    """Each channel's worths under ``readings``, grouped by their pytree structure and shapes."""
    stacks = []
    for column in range(len(readings[0])):
        groups: dict[object, list[int]] = {}
        for index, worths in enumerate(readings):
            leaves, tree = jax.tree.flatten(worths[column])
            groups.setdefault((tree, tuple(jnp.shape(leaf) for leaf in leaves)), []).append(index)
        members, stacked, bends = [], [], []
        for indices in groups.values():
            group = [readings[index][column] for index in indices]
            members.append(np.asarray(indices))
            stacked.append(jax.tree.map(lambda *leaves: jnp.stack(leaves), *group))
            bent = np.array([_inflection(worth) for worth in group])
            bends.append(bent if np.any(bent > 0.0) else None)
        stacks.append(_Stack(tuple(members), tuple(stacked), tuple(bends)))
    return tuple(stacks)


def _inflection(worth: _Worth) -> float:
    curve = worth.channel.curve
    return curve._standard_inflection() if isinstance(curve, Saturation) else 0.0


@eqx.filter_jit
def _bounded_stack(worths: _Worth, bends: Array, low: Array, high: Array) -> tuple[_Bounded, Array]:
    """:func:`_bounded` of every stacked worth over one box of its rate, and whether each found
    its touches. A worth whose curve is concave from zero touches where it starts, so its chord is
    empty and its envelope its curve."""

    def one(worth: _Worth, inflection: Array) -> tuple[_Bounded, Array]:
        # a group is bounded only where it bends, and only a saturation's curve bends
        curve = cast(Saturation, worth.channel.curve)
        floor = (worth.carry + low * worth.reach) / curve.scale
        cap = (worth.carry + high * worth.reach) / curve.scale
        touch = _bisected_touches(curve, floor, inflection, type(curve)._support)
        return _chord(worth, floor, cap, touch), jnp.all(jnp.isfinite(touch))

    return jax.vmap(one)(worths, bends)


def _enveloped(stack: _Stack, low: float, high: float) -> _Stack:
    """``stack``'s worths bounded over ``[low, high]`` of the channel's rate (:func:`_bounded`)."""
    worths = []
    for group, bends in zip(stack.worths, stack.bends, strict=True):
        if bends is None:
            worths.append(group)
            continue
        bounded, found = _bounded_stack(
            group,
            jnp.asarray(bends),
            jnp.asarray(low, dtype=float),
            jnp.asarray(high, dtype=float),
        )
        missing = np.flatnonzero(~np.asarray(found))
        if missing.size:
            first = int(missing[0])
            curve = jax.tree.map(lambda leaf, first=first: leaf[first], group).channel.curve
            raise RuntimeError(f"{curve!r}: no tangency within 2^64 times its inflection")
        worths.append(bounded)
    return _Stack(stack.members, tuple(worths), stack.bends)


@eqx.filter_jit
def _read_stack(worths: _Worth, rate: Array) -> tuple[Array, Array]:
    return jax.vmap(lambda worth: jax.value_and_grad(worth)(rate))(worths)


def _read(stacks: Sequence[_Stack], split: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """``(readings, channels)`` values of every reading's worths at ``split``, and their slopes."""
    values = np.empty((count, len(stacks)))
    slopes = np.empty((count, len(stacks)))
    for column, (stack, rate) in enumerate(zip(stacks, split, strict=True)):
        at = jnp.asarray(rate, dtype=float)
        for members, worths in zip(stack.members, stack.worths, strict=True):
            value, slope = _read_stack(worths, at)
            values[members, column] = np.asarray(value)
            slopes[members, column] = np.asarray(slope)
    return values, slopes


def _cvar_search(  # noqa: C901, PLR0915
    curves: tuple[_Stack, ...],
    starts: Sequence[np.ndarray],
    base: np.ndarray,
    top: float,
    rate: float,
    level: float,
    lower: np.ndarray,
    upper: np.ndarray,
    settings: _Settings,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, _Ending]:
    """:func:`cvar_allocate`'s split, its gains, their worst share's mean, a bound on every
    split's, and how the search ended.

    A gain is a reading's return on the curves less its ``base``: the reference's return for
    :func:`cvar_allocate`, each reading's best for :func:`minimax_allocate`. The first box tries
    ``starts`` first, and the first of them is the split returned where none gains more; ``top``
    bounds every split's worst share's mean before any plane does. The tolerance is set by
    ``settings`` on the largest of ``base``, and a box is cut only where the count of boxes, with
    its halves, stays within ``settings.max_boxes``. The readings are weighed alike, or each by its
    own of ``weights``, positive and summing to 1."""
    count, size = sum(members.size for members in curves[0].members), starts[0].size
    # where every curve is concave each envelope is its curve: one box, each split read once
    alike = all(bends is None for stack in curves for bends in stack.bends)
    scale = float(np.max(np.abs(base))) or 1.0
    tolerance, floored = settings.tolerance(scale, _rounding(tuple(s.worths for s in curves)))
    unsolved: list[int] = []
    # the variables are the rates, eta and one excess u_r a reading; each plane reads
    # u_r >= eta - (gain_r + slope_r @ (s - tried)). The rates are read in a power of two of the
    # budget a period, each held at most the budget, which the budget's row implies, and the gains
    # in one of the largest base: HiGHS leaves out every entry of at most 1e-9, where a slope a
    # currency unit over that return falls from a budget a period of 1e9, and refuses one past
    # 1e15, which a slope in currency passes where a reading's fit has run to the edge of its
    # family, a coefficient of 1e35 on a curve barely bent. A power of two scales each number
    # exactly, so the bound and the split come back exactly. The program's value is the worst
    # share's mean times -max(share, 1), so its least cost is 1: HiGHS folds the costs into its
    # scaling when the least is under 0.1, and at 1 / share that stretched the scaling's factors to
    # 2^15; the solution then missed 1e-10 once unscaled, and HiGHS's second, unscaled solve, its
    # costs unperturbed, cycled at the level 1, where every excess's reduced cost is zero. Weighed,
    # a reading's excess costs its weight over the level times the same factor, so a reading of
    # weight 1 / count costs what it costs alike. Raising the least weight's cost to 1 instead
    # raised eta's to the level over that weight: past a spread of 1e7 in the weights HiGHS then
    # stalled, or read the program unbounded
    share = level * count
    weight = max(share, 1.0)
    costs = weight / share if weights is None else weight * weights / level
    lines = np.repeat(np.arange(count), size + 2)
    columns = np.concatenate(
        [np.tile(np.arange(size + 1), (count, 1)), size + 1 + np.arange(count)[:, None]], axis=1
    ).ravel()
    objective = np.concatenate([np.zeros(size), [-weight], np.broadcast_to(costs, count)])
    unit, measure = _unit(rate), _unit(scale)
    best: tuple[float, np.ndarray, np.ndarray] = (-np.inf, starts[0], np.zeros(count))

    def tried(
        envelopes: tuple[_Stack, ...], split: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Each reading's gain on ``split`` read on the envelopes, its slopes, the envelopes'
        excess over the curves by channel, and the gain on the curves, which the best split
        returned is chosen on."""
        nonlocal best
        values, slopes = _read(envelopes, split, count)
        exact = values if alike else _read(curves, split, count)[0]
        gain = exact.sum(axis=1) - base
        value = _cvar(gain, level, weights)
        if value > best[0]:
            best = (value, split, gain)
        return values.sum(axis=1) - base, slopes, values - exact, gain

    def box(
        low: np.ndarray,
        high: np.ndarray,
        envelopes: tuple[_Stack, ...],
        rows: list[sparse.csr_array],
        limits: list[np.ndarray],
        starts: Sequence[np.ndarray],
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """The box's bound from the planes on its envelopes, its best split on them, and their
        excess over the curves there by channel, each reading weighed as the worst share of the
        curves' gains weighs it, the planes cut first at ``starts``. Leaves in ``rows`` and
        ``limits`` the planes worth keeping."""
        ceiling, relaxed, at, excess = top, -np.inf, starts[0], np.zeros(size)
        splits, slack, kept = list(starts), None, 0
        floors, caps = low / unit, np.minimum(high, rate) / unit
        for _ in range(_ROUNDS if alike else _BOX_ROUNDS):
            for split in splits:
                gain, slopes, over, actual = tried(envelopes, split)
                value = _cvar(gain, level, weights)
                if value > relaxed:
                    relaxed, at, excess = value, split, _cvar_weights(actual, level, weights) @ over
                plane = np.concatenate(
                    [-slopes * (unit / measure), np.ones((count, 1)), -np.ones((count, 1))], 1
                )
                rows.append(
                    sparse.csr_array(
                        (plane.ravel(), (lines, columns)), shape=(count, size + 1 + count)
                    )
                )
                limits.append((gain - slopes @ split) / measure)
            if ceiling - best[0] <= tolerance:
                break
            # on concave curves the planes close; past them a box is cut once its planes come
            # within half its gap of its best on the envelopes
            if ceiling - relaxed <= tolerance or (
                not alike
                and np.isfinite(ceiling)
                and ceiling - relaxed <= 0.5 * (ceiling - best[0])
            ):
                break
            matrix, ends = sparse.vstack(rows, format="csr"), np.concatenate(limits)
            least, most = _spans(matrix, ends, floors, caps)
            # each plane's reading, read off the excess column, its last
            reading = np.maximum.reduceat(matrix.indices, matrix.indptr[:-1]) - size - 1
            program = _planes(
                objective,
                matrix,
                ends,
                [*zip(floors, caps, strict=True), (None, None), *((0.0, None),) * count],
                np.concatenate([np.ones(size), np.zeros(1 + count)])[None, :],
                [rate / unit],
                # a best eta is one of the readings' least planes at the split, and each excess
                # its distance below eta
                [
                    *zip(floors, caps, strict=True),
                    (least, most),
                    *((0.0, float(np.nextafter(most - least, np.inf))),) * count,
                ],
                partial(_evened, reading=reading, count=count, cap=costs, total=weight),
            )
            if program.status in _UNSOLVED:
                unsolved.append(int(program.nit))
                break  # the box keeps its last program's bound
            if program.status != 0:
                raise RuntimeError(f"the cutting planes' linear program failed: {program.message}")
            ceiling = min(ceiling, _outward(-float(program.fun), measure, weight, math.inf))
            slack, kept = program.ineqlin.residual, matrix.shape[0]
            splits = [_onto(program.x[:size] * unit, low, high, rate)]
        if slack is not None and not alike:
            # the halves' bounds need only the planes the last program held tight, and those
            # tried since; a plane dropped that a half needs is cut again at its split
            matrix = sparse.vstack(rows, format="csr")
            keep = np.concatenate([slack <= 1e-8, np.ones(matrix.shape[0] - kept, dtype=bool)])
            rows[:] = [matrix[keep]]
            limits[:] = [np.concatenate(limits)[keep]]
        return ceiling, at, excess

    envelopes = tuple(
        _enveloped(stack, low, high) for stack, low, high in zip(curves, lower, upper, strict=True)
    )
    rows: list[sparse.csr_array] = []
    limits: list[np.ndarray] = []
    ceiling, at, excess = box(lower, upper, envelopes, rows, limits, starts)
    heap = [(-ceiling, 0, lower, upper, envelopes, rows, limits, at, excess)]
    settled, boxes, widths = -np.inf, 1, np.maximum(upper - lower, _EPS)
    while not alike and heap and -heap[0][0] - best[0] > tolerance:
        _, _, low, high, envelopes, rows, limits, at, excess = heap[0]
        cut_at, halves = _cut(low, high, at, excess, rate, widths)
        if boxes + len(halves) > settings.max_boxes:
            break
        heapq.heappop(heap)
        for floor, cap in halves:
            child_low, child_high = low.copy(), high.copy()
            child_low[cut_at], child_high[cut_at] = floor, cap
            child = list(envelopes)
            child[cut_at] = _enveloped(curves[cut_at], floor, cap)
            inside = bool(np.all((child_low <= at) & (at <= child_high)))
            start = at if inside else _onto(at, child_low, child_high, rate)
            child_rows, child_limits = list(rows), list(limits)
            peak, child_at, child_excess = box(
                child_low, child_high, tuple(child), child_rows, child_limits, [start]
            )
            boxes += 1
            if peak - best[0] > tolerance:
                heapq.heappush(
                    heap,
                    (
                        -peak,
                        boxes,
                        child_low,
                        child_high,
                        tuple(child),
                        child_rows,
                        child_limits,
                        child_at,
                        child_excess,
                    ),
                )
            else:
                settled = max(settled, peak)
    value, spend, gains = best
    left = -heap[0][0] if heap else -np.inf
    bound = max(value, settled, left)
    closed = bound - value <= tolerance
    # past concave curves a gap left open is the open boxes', which only the cap left uncut; on
    # them it is the one box's, whose planes ran out of rounds or stopped at a program unsolved
    limit: SearchLimit | None = (
        None if closed else "max_boxes" if not alike else "unsolved" if unsolved else "rounds"
    )
    ending = _Ending(
        boxes=boxes,
        stopped="closed" if closed else "cap",
        limit=limit,
        tolerance=tolerance,
        floored=floored,
        scale=scale,
        unsolved=tuple(unsolved),
    )
    return np.asarray(spend), gains, float(value), float(bound), ending


def _cut(
    low: np.ndarray,
    high: np.ndarray,
    at: np.ndarray,
    excess: np.ndarray,
    rate: float,
    widths: np.ndarray,
) -> tuple[int, tuple[tuple[float, float], ...]]:
    """The channel a box of :func:`_cvar_search` is cut on, and its halves' intervals of that
    channel's rate.

    The channel is the one whose envelope stands furthest above its curve at ``at``, the box's best
    split on its envelopes, or where none does the one whose interval is widest against
    ``widths``. Each interval is first narrowed to the rates a split of the budget in the box can
    give its channel, so each half holds such a split: the cut is at ``at``'s rate, which a split of
    the budget gives, or halfway along the narrowed interval where that rate is an end of it. A
    channel the budget leaves one rate is not cut but narrowed to it."""
    floor = np.maximum(low, rate - (high.sum() - high))
    cap = np.minimum(high, rate - (low.sum() - low))
    channel = (
        int(np.argmax(excess)) if np.max(excess) > 0.0 else int(np.argmax((cap - floor) / widths))
    )
    a, b = float(floor[channel]), float(cap[channel])
    if not a < b:
        return channel, ((min(a, b), max(a, b)),)
    cut = float(at[channel]) if a < at[channel] < b else a + 0.5 * (b - a)
    return channel, ((a, cut), (cut, b))


def _evened(
    duals: np.ndarray, reading: np.ndarray, count: int, cap: float | np.ndarray, total: float
) -> np.ndarray:
    """The duals of :func:`_cvar_search`'s planes rescaled reading by reading, ``reading`` each
    plane's, so that no reading's mass passes ``cap``, its excess's cost, one for every reading or
    one a reading, and the masses come to ``total``, eta's, as far as the readings with a mass can
    take them.

    A reading's mass past its excess's cost leaves that excess's reduced cost below nothing, and
    masses that miss eta's cost leave eta's off nothing; either costs the bound read from them the
    reduced cost times its column's far end. On a search over 400 readings HiGHS's masses stood off
    by up to 3e-7, which cost the bound 1.4e-9 of the reference's largest return, past the
    search's tolerance of 1e-9; from the masses rescaled, 2e-11."""
    mass = np.bincount(reading, weights=np.maximum(-duals, 0.0), minlength=count)
    target = np.minimum(mass, cap)
    room = np.where(target > 0.0, cap - target, 0.0)
    short = total - target.sum()
    if short < 0.0:
        target = target * (total / target.sum())
    elif room.sum() > 0.0:
        target = target + room * min(1.0, short / room.sum())
    factor = np.divide(target, mass, out=np.zeros(count), where=mass > 0.0)
    return np.where(duals < 0.0, duals * factor[reading], 0.0)


def _cvar_weights(gain: np.ndarray, level: float, weights: np.ndarray | None = None) -> np.ndarray:
    """The weight :func:`_cvar` puts on each of ``gain``: ``1 / (level * n)`` on each of the worst
    whole share, the rest of the share on the next, none on the others. Where the readings carry
    ``weights``, summing to 1, each of the worst takes its own weight over ``level`` until the share
    is full, the reading at its boundary the rest of it."""
    if weights is not None:
        order, taken = _taken(gain, level, weights)
        out = np.zeros(gain.size)
        out[order] = taken / level
        return out
    order = np.argsort(gain, kind="stable")
    share = level * gain.size
    whole = int(np.floor(share))
    weights = np.zeros(gain.size)
    weights[order[:whole]] = 1.0 / share
    if whole < gain.size:
        weights[order[whole]] = (share - whole) / share
    return weights


def _taken(gain: np.ndarray, level: float, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``gain``'s order, worst first, and the mass of the ``level`` share each takes in that order:
    its whole weight up to the share's boundary, a weighted quantile, the reading there the rest."""
    order = np.argsort(gain, kind="stable")
    mass = weights[order]
    before = np.concatenate([[0.0], np.cumsum(mass)[:-1]])
    return order, np.clip(level - before, 0.0, mass)


def _cvar(gain: np.ndarray, level: float, weights: np.ndarray | None = None) -> float:
    """The mean of the worst ``level`` share of ``gain``, each weighed alike, the last in part, or
    each by its own of ``weights``, summing to 1, where the readings carry them."""
    if weights is not None:
        order, taken = _taken(gain, level, weights)
        return float(taken @ gain[order]) / level
    ordered = np.sort(gain)
    share = level * ordered.size
    whole = int(np.floor(share))
    total = float(ordered[:whole].sum())
    if whole < ordered.size:
        total += (share - whole) * float(ordered[whole])
    return total / share
