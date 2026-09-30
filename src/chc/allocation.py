"""A media budget spread over channels: one spend a period for each channel, within its box, for
the most return, carryover counted in and out.

:func:`allocate` plans ``periods`` periods of spend on :class:`chc.response.Channel`\\ s. Each
channel runs at one rate a period, between its ``lower`` and ``upper`` rates, and the rates spend
the budget. A plan's worth is what its spend returns on the channels: the adstock the ``history``
leaves runs into the plan's periods, and the plan's spend runs on for each kernel's length after
them with nothing spent, so a channel whose carryover is long is not undervalued for the periods
the plan ends before its return does.

**Exact where every curve is concave.** A channel's worth is its coefficient times the curve summed
over periods whose adstock is affine in the rate, so it is concave in the rate when the curve is
concave and the coefficient not negative, and the channels add. At a price ``mu`` per currency unit
each channel then runs where the slope of its worth meets ``periods * mu``, or at an end of its
box, and ``mu`` is where the rates spend the budget. The price moves every rate the same way, so a
bisection on it is exact; where a channel is linear its rate jumps at one price, and the plan
spends the last of the budget on the rates either side of it, both of which are best there.

**An S-shaped curve is planned on its envelope.** Where a curve starts convex, the rates are the
plan on its concave envelope (:class:`chc.response.Envelope`), whose return bounds any plan's on
the curve: :attr:`Allocation.bound` is that best, and ``bound - worth`` bounds how far the plan is
from the best on the true curves. On concave curves the two are one number.

**The price** is the budget's shadow price on the planned channels: the return one more currency
unit of budget buys, spread the way the plan spends it. It is where the channels running inside
their boxes meet; a channel at its cap returns more a unit and one at its floor less.

HONEST SCOPE:

* The channels are read as given: a fitted channel's error goes straight into the plan. Where the
  channels come from an experiment (:func:`chc.lift.fit_lift`), a plan that runs a channel past
  the adstock the tests covered reads the family's shape, not the experiment.
* One rate a period for each channel, not a schedule: with a response that does not change with
  the period, only the history's carryover at the start and the tail at the end make periods
  differ.
* The tail is counted with nothing spent after the plan. Spend that continues after it would take
  the tail further up the curve, so a long kernel's tail is valued at its most.
* :class:`chc.response.Ricker` is not monotone, and a negative coefficient turns a concave curve
  convex; the bisection proves nothing for either, and both are refused.
* :func:`chc.mmm.prescribe` plans a budget over a continuous plant whose adstock is a state; this
  plans discrete channels, which that plant does not describe.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from scipy.optimize import brentq

from chc.response import Channel, Logarithmic, Power, Saturation, relax

__all__ = ["Allocation", "allocate"]

_EPS = float(np.finfo(float).eps)


@dataclass(frozen=True)
class Allocation:
    """A plan's spend a period for each channel, what it returns, and what the budget is worth.

    Attributes:
        spend: ``(channels,)`` spend a period, in the budget's currency.
        worth: the channels' return on the plan over its periods and each kernel's length after.
        bound: the most any plan in the box at the budget returns on the channels' concave
            envelopes; ``bound - worth`` bounds the plan's shortfall from the best on the channels,
            and is ``0`` when every curve is concave.
        price: the return on one more currency unit of budget, on the envelopes.
    """

    spend: np.ndarray
    worth: float
    bound: float
    price: float


class _Worth(eqx.Module):
    """One channel's return on the plan, as a function of its spend a period."""

    channel: Channel
    carry: Array  # the adstock the history leaves, over the plan and the tail
    reach: Array  # the adstock one unit a period over the plan leaves

    def __call__(self, rate: Array) -> Array:
        return self.channel.coefficient * jnp.sum(
            self.channel.curve(self.carry + rate * self.reach)
        )


@eqx.filter_jit
def _value_and_slope(worth: _Worth, rate: Array) -> tuple[Array, Array]:
    return jax.value_and_grad(worth)(rate)


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


def _value(worth: _Worth, rate: float) -> float:
    return float(_value_and_slope(worth, jnp.asarray(rate, dtype=float))[0])


def _slope(worth: _Worth, rate: float) -> float:
    return float(_value_and_slope(worth, jnp.asarray(rate, dtype=float))[1])


def _rate(worth: _Worth, low: float, high: float, target: float) -> float:
    """Where the worth's slope meets ``target``, or the end of the box it presses on."""
    if _slope(worth, low) <= target:
        return low
    if _slope(worth, high) >= target:
        return high
    return brentq(
        lambda rate: _slope(worth, rate) - target, low, high, xtol=4 * _EPS * high, rtol=4 * _EPS
    )


def _check(
    channels: Sequence[Channel],
    budget: float,
    periods: int,
    lower: np.ndarray,
    upper: np.ndarray,
    history: np.ndarray,
) -> None:
    if not channels:
        raise ValueError("no channels to allocate over")
    for index, channel in enumerate(channels):
        if not isinstance(channel, Channel):
            raise TypeError(f"channel {index} is a {type(channel).__name__}, not a Channel")
        if not isinstance(channel.curve, Saturation | Logarithmic | Power):
            raise ValueError(
                f"channel {index}'s curve is a {type(channel.curve).__name__}, which is not "
                "increasing and concave or S-shaped; the allocation cannot certify a plan on it"
            )
        if not float(channel.coefficient) >= 0.0:
            raise ValueError(
                f"channel {index}'s coefficient is {float(channel.coefficient)}; a negative one "
                "turns its curve convex"
            )
    if not (isinstance(periods, int) and periods >= 1):
        raise ValueError(f"periods={periods!r} is not a whole number of periods, at least 1")
    size = len(channels)
    for name, rates in (("lower", lower), ("upper", upper)):
        if rates.shape != (size,) or not np.all(np.isfinite(rates)):
            raise ValueError(f"{name} has shape {rates.shape}; it needs one finite rate a channel")
    if np.any(lower < 0.0) or np.any(upper < lower):
        raise ValueError(f"the box [{lower}, {upper}] is not 0 <= lower <= upper")
    if history.ndim != 2 or history.shape[1] != size:
        raise ValueError(f"history has shape {history.shape}; it needs one column a channel")
    if not (np.all(np.isfinite(history)) and np.all(history >= 0.0)):
        raise ValueError("history holds a negative or non-finite spend")
    least, most = periods * float(lower.sum()), periods * float(upper.sum())
    if not (np.isfinite(budget) and least <= budget <= most):
        raise ValueError(
            f"a budget of {budget} is outside what the box spends over {periods} periods, "
            f"[{least}, {most}]"
        )


def allocate(
    channels: Sequence[Channel],
    budget: float,
    periods: int,
    *,
    lower: ArrayLike,
    upper: ArrayLike,
    history: ArrayLike | None = None,
) -> Allocation:
    """Spend ``budget`` over ``periods`` at one rate a period for each channel, for the most return.

    Args:
        channels: the channels, each read as given.
        budget: what the plan spends over its periods, every channel together.
        periods: how many periods the plan runs.
        lower, upper: ``(channels,)`` each channel's least and most spend a period. There is no
            default: a box is where the channels were seen, and a plan outside it reads the curves
            where nothing measured them.
        history: ``(T, channels)`` spend in the periods before the plan, whose adstock runs into
            it; nothing spent before the plan when omitted.

    Raises:
        TypeError: a channel is not a :class:`chc.response.Channel`.
        ValueError: a curve the bisection cannot certify on (see the module's scope), a negative
            coefficient, a box or history of the wrong shape, a box with ``lower > upper`` or a
            negative end, or a budget the box cannot spend over the periods.
    """
    lower_rates = np.asarray(lower, dtype=float)
    upper_rates = np.asarray(upper, dtype=float)
    spent = np.zeros((0, len(channels))) if history is None else np.asarray(history, dtype=float)
    _check(channels, float(budget), periods, lower_rates, upper_rates, spent)
    given = tuple(channels)
    relaxed = relax(given)
    envelopes = _worths(relaxed, spent, periods)
    target = float(budget) / periods

    def rates(price: float) -> np.ndarray:
        return np.array(
            [
                _rate(worth, low, high, periods * price)
                for worth, low, high in zip(envelopes, lower_rates, upper_rates, strict=True)
            ]
        )

    # Every rate falls as the price rises. At no price any rate is as good as its cap, since no
    # slope is negative; at the steepest slope the floors allow, every channel sits at its floor.
    # So the caps overspend and the floors underspend, and the bisection keeps it that way.
    cheap, many = 0.0, upper_rates
    dear = max(0.0, *(_slope(w, low) for w, low in zip(envelopes, lower_rates, strict=True)))
    dear, few = dear / periods, lower_rates
    tolerance = 4 * _EPS * dear
    while dear - cheap > tolerance:
        middle = 0.5 * (cheap + dear)
        at = rates(middle)
        if at.sum() > target:
            many, cheap = at, middle
        else:
            few, dear = at, middle
    # both ends are best at a price within rounding of the other's, and so is any mix of them: the
    # mix that spends the budget exactly is the plan, and it matters where a linear rate jumps
    surplus = float(many.sum() - few.sum())
    share = 0.0 if surplus <= 0.0 else (target - float(few.sum())) / surplus
    spend = np.clip(few + share * (many - few), lower_rates, upper_rates)
    bound = sum(_value(w, rate) for w, rate in zip(envelopes, spend, strict=True))
    worths = envelopes if relaxed is given else _worths(given, spent, periods)
    worth = sum(_value(w, rate) for w, rate in zip(worths, spend, strict=True))
    return Allocation(spend=spend, worth=worth, bound=bound, price=0.5 * (cheap + dear))
