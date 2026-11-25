"""A channel's worth as a function of its rate, its slope and curvature, and the
concave envelopes over a box of rates that the searches bound a plan with."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.optimize import brentq

from chc.response import Channel, Saturation, _touches

_EPS = float(np.finfo(float).eps)


class _Worth(eqx.Module):
    """One channel's return on the plan, as a function of its spend a period."""

    channel: Channel
    carry: Array  # the adstock the history leaves, over the plan and the tail
    reach: Array  # the adstock one unit a period over the plan leaves

    def __call__(self, rate: Array) -> Array:
        return self.channel.coefficient * jnp.sum(
            self.channel.curve(self.carry + rate * self.reach)
        )


class _Bounded(_Worth):
    """A channel's worth bounded from above over a box of its rate: each period's curve replaced by
    its concave envelope over the adstock the box allows there (:func:`_bounded`)."""

    floor: Array  # each period's adstock at the box's floor, in the curve's scales
    corner: Array  # where each period's chord gives way to the curve, in the curve's scales
    base: Array  # the curve at the floor
    rise: Array  # the chord's slope, in the curve's scales

    def __call__(self, rate: Array) -> Array:
        curve = self.channel.curve
        z = (self.carry + rate * self.reach) / curve.scale
        chord = self.base + (z - self.floor) * self.rise
        return self.channel.coefficient * jnp.sum(
            jnp.where(z <= self.corner, chord, curve.standard(z))
        )


class _National(eqx.Module):
    """A national channel's return on the plan, as a function of its national spend a period: each
    geo's cell reads its share of it, its ``reach`` scaled by the share, and the cells add."""

    parts: tuple[_Worth, ...]

    def __call__(self, rate: Array) -> Array:
        total = self.parts[0](rate)
        for part in self.parts[1:]:
            total = total + part(rate)
        return total


_Variable = _Worth | _National
"""What the geo planners choose a rate for: a geo's own cell, or a national channel over its geos'
cells."""


def _boxed(variable: _Variable, low: float, high: float) -> _Variable:
    """``variable`` bounded from above over ``[low, high]`` of its rate: a cell as
    :func:`_bounded` bounds it, a national channel by its cells' bounds over the box, or itself
    where none of its curves is S-shaped.

    Each cell's bound is concave in the national rate, above its worth and equal to it at both ends
    of the box, so their sum is too: no tighter than the sum's own envelope, but as good for a
    search, which cuts the box until the bound meets the worth."""
    if isinstance(variable, _Worth):
        return _bounded(variable, low, high)
    parts = tuple(_bounded(part, low, high) for part in variable.parts)
    if all(new is old for new, old in zip(parts, variable.parts, strict=True)):
        return variable
    return _National(parts)


def _bounded(worth: _Worth, low: float, high: float) -> _Worth:
    """``worth`` bounded from above over ``[low, high]`` of its rate, or ``worth`` itself where its
    curve is concave from zero.

    A rate in the box leaves each period's adstock between its values at the two ends, and on that
    interval the curve's concave envelope is the chord from the floor to where it touches the curve
    (:func:`chc.response._touches`), or to the cap where that comes first, and the curve beyond.
    Each period's envelope is concave in the rate, so their sum is, and it meets the worth at both
    ends of the box, so a box split at a rate is bounded tightly there on both sides.
    """
    curve = worth.channel.curve
    if not (isinstance(curve, Saturation) and curve._standard_inflection() > 0.0):
        return worth
    floor = (worth.carry + low * worth.reach) / curve.scale
    cap = (worth.carry + high * worth.reach) / curve.scale
    return _chord(worth, floor, cap, _touches(curve, floor))


def _chord(worth: _Worth, floor: Array, cap: Array, touch: Array) -> _Bounded:
    """``worth`` with each period's curve replaced by the chord from ``floor`` to ``touch``, or to
    ``cap`` where that comes first, and the curve beyond: :func:`_bounded`'s envelope once the
    touches are found."""
    curve = worth.channel.curve
    end = jnp.minimum(touch, cap)
    # XLA flushes a subnormal difference to zero on the CPU, where the ends still compare unequal
    width = end - floor
    straight = width > 0.0
    base = curve.standard(floor)
    rise = jnp.where(straight, (curve.standard(end) - base) / jnp.where(straight, width, 1.0), 0.0)
    # A chord that ends at the touch meets the curve there with the curve's slope, so either side
    # may read a rate that lands on it. One cut short by the cap is the whole box's envelope: read
    # on the curve at the cap, the slope would jump to the curve's own, steeper than the chord's,
    # and a rate that lands an ulp either side of the cap would read either.
    corner = jnp.where(straight, jnp.where(touch < cap, touch, jnp.inf), -jnp.inf)
    return _Bounded(worth.channel, worth.carry, worth.reach, floor, corner, base, rise)


@eqx.filter_jit
def _value_and_slope(worth: _Variable, rate: Array) -> tuple[Array, Array]:
    return jax.value_and_grad(worth)(rate)


@eqx.filter_jit
def _curvature(worth: _Variable, rate: Array) -> Array:
    return jax.grad(jax.grad(worth))(rate)


def _worths(channels: Sequence[Channel], history: np.ndarray, periods: int) -> tuple[_Worth, ...]:
    before = history.shape[0]
    worths = []
    for column, channel in enumerate(channels):
        tail = channel.kernel.length - 1
        idle = np.concatenate([history[:, column], np.zeros(periods + tail)])
        unit = np.concatenate([np.zeros(before), np.ones(periods), np.zeros(tail)])
        carry = channel.kernel(jnp.asarray(idle))[before:]
        reach = channel.kernel(jnp.asarray(unit))[before:]
        worths.append(_Worth(channel, carry, reach))
    return tuple(worths)


def _value(worth: _Variable, rate: float) -> float:
    return float(_value_and_slope(worth, jnp.asarray(rate, dtype=float))[0])


def _slope(worth: _Variable, rate: float) -> float:
    return float(_value_and_slope(worth, jnp.asarray(rate, dtype=float))[1])


def _rate(worth: _Variable, low: float, high: float, target: float) -> float:
    """Where the worth's slope meets ``target``, or the end of the box it presses on."""
    if _slope(worth, low) <= target:
        return low
    if _slope(worth, high) >= target:
        return high
    return brentq(
        lambda rate: _slope(worth, rate) - target, low, high, xtol=4 * _EPS * high, rtol=4 * _EPS
    )


def _rates(
    envelopes: Sequence[_Worth], lower: np.ndarray, upper: np.ndarray, periods: int, price: float
) -> np.ndarray:
    """Each channel's rate at ``price``: every rate falls as the price rises."""
    return np.array(
        [
            _rate(worth, low, high, periods * price)
            for worth, low, high in zip(envelopes, lower, upper, strict=True)
        ]
    )
