"""A schedule over periods where a kernel is longer than one period: local ascents
from the envelopes' plan and from random bursts, polished by active-set Newton steps."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.optimize import minimize

from chc._allocation.curves import _worths
from chc._allocation.planes import _onto
from chc._allocation.search import _branch_and_bound, _rounding, _Settings
from chc.response import Channel, relax

if TYPE_CHECKING:
    from chc.allocation import Allocation, SearchLimit, _Stopwatch


class _Schedule(eqx.Module):
    """The channels' return on a schedule, a ``(periods, channels)`` spend: the history's adstock
    runs into it, and its spend runs on for each kernel's length after it with nothing spent."""

    channels: tuple[Channel, ...]
    history: Array  # (T, channels) spend before the schedule

    def __call__(self, grid: Array) -> Array:
        before = self.history.shape[0]
        total = jnp.zeros((), dtype=grid.dtype)
        for column, channel in enumerate(self.channels):
            after = jnp.zeros(channel.kernel.length - 1, dtype=grid.dtype)
            series = jnp.concatenate([self.history[:, column], grid[:, column], after])
            adstock = channel.kernel(series)[before:]
            total = total + channel.coefficient * jnp.sum(channel.curve(adstock))
        return total


@eqx.filter_jit
def _schedule_value_and_grad(schedule: _Schedule, grid: Array) -> tuple[Array, Array]:
    return jax.value_and_grad(schedule)(grid)


@eqx.filter_jit
def _schedule_curvature(schedule: _Schedule, grid: Array) -> Array:
    return jax.hessian(schedule)(grid)


def _read_schedule(schedule: _Schedule, grid: np.ndarray) -> tuple[float, np.ndarray]:
    value, slope = _schedule_value_and_grad(schedule, jnp.asarray(grid, dtype=float))
    return float(value), np.asarray(slope, dtype=float)


def _ascend(
    schedule: _Schedule, start: np.ndarray, low: np.ndarray, high: np.ndarray, budget: float
) -> np.ndarray:
    """A local best schedule from ``start``: SLSQP on the worth in the box, the schedule spending
    the budget, projected onto both after it stops."""
    shape = start.shape
    scale = abs(_read_schedule(schedule, start)[0]) or 1.0

    def loss(flat: np.ndarray) -> tuple[float, np.ndarray]:
        value, slope = _read_schedule(schedule, flat.reshape(shape))
        return -value / scale, -slope.ravel() / scale

    found = minimize(
        loss,
        start.ravel(),
        jac=True,
        method="SLSQP",
        bounds=list(zip(low, high, strict=True)),
        constraints=[{"type": "eq", "fun": lambda x: x.sum() - budget, "jac": np.ones_like}],
        options={"maxiter": 500, "ftol": 1e-15},
    )
    return _newton_face(
        schedule, _onto(np.asarray(found.x, dtype=float), low, high, budget), low, high
    ).reshape(shape)


def _newton_face(
    schedule: _Schedule, flat: np.ndarray, low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    """``flat`` moved by Newton's method on the faces of the box, the budget held: SLSQP stops with
    the slopes of the cells inside their boxes equal only to its tolerance, and on a cell inside its
    box that belongs on a bound, which leaves a Frank--Wolfe gap of about that share. Each step is
    to the stationary point of the worth's quadratic on the face, taken as far as the box allows; a
    cell the step carries to a bound stays there, and a cell on a bound whose slope beats the face's
    price leaves it. A step is halved until it gains, so the worth never falls."""
    cells = flat.size
    grid_shape = (cells // schedule.history.shape[1], schedule.history.shape[1])
    x = flat.copy()
    value, slope = _read_schedule(schedule, x.reshape(grid_shape))
    span = np.maximum(high - low, 1.0)
    for _ in range(4 * cells + 20):
        g = slope.ravel()
        floor, cap = x <= low + 1e-14 * span, x >= high - 1e-14 * span
        free = ~(floor | cap)
        price = (
            float(np.mean(g[free]))
            if free.any()
            else 0.5
            * (float(np.max(g[floor], initial=-np.inf)) + float(np.min(g[cap], initial=np.inf)))
        )
        moving = free | (floor & (g > price)) | (cap & (g < price))
        curvature = np.asarray(
            _schedule_curvature(schedule, jnp.asarray(x.reshape(grid_shape), dtype=float)),
            dtype=float,
        ).reshape(cells, cells)
        step = np.zeros(cells)
        while (size := int(moving.sum())) >= 2:
            system = np.block(
                [
                    [curvature[np.ix_(moving, moving)], np.ones((size, 1))],
                    [np.ones((1, size)), np.zeros((1, 1))],
                ]
            )
            rhs = np.concatenate([-g[moving], [0.0]])
            step = np.zeros(cells)
            step[moving] = np.linalg.lstsq(system, rhs, rcond=None)[0][:size]
            # a cell on a bound that the step would carry out of its box stays on the bound
            outward = (floor & (step < 0.0)) | (cap & (step > 0.0))
            if not outward.any():
                break
            moving &= ~outward
        if size < 2:
            break
        if not g @ step > 0.0:
            # not an ascent off a concave face: the slope's own direction on the budget's plane
            step[moving] = g[moving] - float(np.mean(g[moving]))
            if not g @ step > 0.0:
                break
        with np.errstate(divide="ignore", invalid="ignore"):
            room = np.where(
                step > 0.0, (high - x) / step, np.where(step < 0.0, (low - x) / step, np.inf)
            )
        reach = min(1.0, max(float(np.min(room)), 0.0))
        for _ in range(60):
            moved = np.clip(x + reach * step, low, high)
            reached, rise = _read_schedule(schedule, moved.reshape(grid_shape))
            if reached > value:
                break
            reach /= 2.0
        else:
            break
        x, value, slope = moved, reached, rise
    return x


def _frank_wolfe(
    schedule: _Schedule, grid: np.ndarray, low: np.ndarray, high: np.ndarray, budget: float
) -> float:
    """No schedule in the box that spends the budget returns more on ``schedule``, concave, than
    its worth at ``grid`` plus the most its slope there gains toward one: the schedule that gains
    the most fills the cells of the steepest slopes from their floors to their caps."""
    value, slope = _read_schedule(schedule, grid)
    slope = slope.ravel()
    best, left = low.copy(), budget - float(low.sum())
    for cell in np.argsort(-slope, kind="stable"):
        if left <= 0.0:
            break
        step = min(high[cell] - low[cell], left)
        best[cell] += step
        left -= step
    return value + max(float(slope @ (best - grid.ravel())), 0.0)


def _scheduled(
    given: tuple[Channel, ...],
    flat: Allocation,
    periods: int,
    lower: np.ndarray,
    upper: np.ndarray,
    spent: np.ndarray,
    settings: _Settings,
    clock: _Stopwatch,
) -> Allocation:
    """:func:`allocate`'s schedule: searched in boxes over the periods and channels where every
    kernel is one period long, otherwise the best of local ascents bounded on the envelopes;
    ``flat`` is the best rate a period, its seconds read on ``clock``."""
    size, budget = len(given), flat.budget
    low, high = np.tile(lower, periods), np.tile(upper, periods)
    curves = _Schedule(given, jnp.asarray(spent, dtype=float))
    limit: SearchLimit | None
    if all(channel.kernel.length == 1 for channel in given):
        # each period's spend returns in its own period: a cell of every period and channel
        cells = _worths(given, np.zeros((0, size)), 1) * periods
        groups = tuple(tuple(range(column, size * periods, size)) for column in range(size))
        spend, searched, bound, price, ending = _branch_and_bound(
            cells, low, high, 1, budget, settings, groups=groups
        )
        grid = spend.reshape(periods, size)
        worth = _read_schedule(curves, grid)[0]
        bound += worth - searched
        stopped, boxes, limit = ending.stopped, ending.boxes, ending.limit
        tolerance, floored, scale = ending.tolerance, ending.floored, ending.scale
    else:
        enveloped = relax(given)
        relaxed = _Schedule(enveloped, curves.history)
        start = np.tile(flat.spend, (periods, 1))
        envelope = _ascend(relaxed, start, low, high, budget)
        grid, worth = start, flat.worth
        if enveloped is given:
            # concave: the envelopes are the curves, and their ascent is the curves' best
            ascents = [envelope]
        else:
            # bursts: schedules that spend the budget in a few cells, seeded so the call repeats
            draws = np.random.default_rng(_BURST_SEED).dirichlet(
                np.full(low.size, 0.3), size=_BURSTS
            )
            bursts = [_onto(budget * d, low, high, budget).reshape(periods, size) for d in draws]
            ascents = [
                _ascend(curves, begin, low, high, budget) for begin in (start, envelope, *bursts)
            ]
        for ascended in ascents:
            value = _read_schedule(curves, ascended)[0]
            if value > worth:
                grid, worth = ascended, value
        bound = max(_frank_wolfe(relaxed, envelope, low, high, budget), worth)
        slope = _read_schedule(curves, grid)[1].ravel()
        price = float(np.max(slope[grid.ravel() < high], initial=0.0))
        scale = abs(bound) or 1.0
        tolerance, floored = settings.tolerance(scale, _rounding(given))
        closed = bound - worth <= tolerance
        stopped = "closed" if closed else "cap"
        boxes, limit = flat.boxes, None if closed else "local"
    compiling, searching = clock.read()
    return replace(
        flat,
        spend=grid.mean(axis=0),
        worth=worth,
        bound=bound,
        price=price,
        stopped=stopped,
        boxes=boxes,
        gap=bound - worth,
        relative_gap=(bound - worth) / scale,
        tolerance=tolerance,
        floored=floored,
        limit=limit,
        compile_seconds=compiling,
        search_seconds=searching,
        schedule=grid,
    )


_BURSTS, _BURST_SEED = 8, 0
"""How many random schedules :func:`allocate`'s local schedule also ascends from, drawn from a
Dirichlet of concentration 0.3 over the cells, which spends most of the budget in a few, and the
seed they are drawn from."""
