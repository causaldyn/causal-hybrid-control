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

**A goal in place of a budget.** :func:`budget_for` finds the budget that meets a goal, and plans
it: the least budget that gains a return (:class:`ReturnTarget`), the budget at which one more unit
returns a given amount (:class:`MarginalReturnTarget`), or the most budget whose plan returns a
given amount a unit on average, a target return on ad spend (:class:`ReturnOnSpendTarget`). A
plan's gain is what its spend adds, ``worth - idle``, where ``idle`` is what the channels return
with nothing spent in the plan: the history's carryover alone. As the price falls every rate rises,
so the plans for every budget lie on one path, and each goal is a point on it.

**A split for several readings of the channels.** Tests that never bent a curve fit several
families alike, and the families part where the plan goes. :func:`minimax_allocate` takes each
reading of every channel and chooses the split whose worst regret over them is least: the regret
under a reading is its best return at the budget over the split's. A reading's regret is convex in
the split where its curves are concave, so the worst is too, and cutting planes (Kelley 1960) close
on it from below while the splits they propose close on it from above; the two ends are the
certificate.

**What a wrong channel costs the plan.** A plan made on channels whose parameters are off by ``d``
loses ``d' W d / 2`` of the worth the true channels' plan returns, to second order:
:func:`decision_weight` is that ``W``, the decision weight of an experiment's value of information
(:mod:`chc.experiment`) with this allocation as the decision. The rates inside their boxes meet at
one price, so an error moves them along the budget, each by its slope's response to the error, less
the share of the whole that keeps the budget spent; ``W`` weighs those moves by each worth's
curvature. An experiment that leaves the parameters with covariance ``S`` leaves an expected regret
of ``tr(W S) / 2`` (:meth:`AllocationWeight.expected_regret`).

**Geos and channels together.** :func:`allocate_geos` plans one budget over a grid of cells, every
geo's channels, each cell a channel of its own with its curve, carryover and history. Besides each
cell's box, :class:`Totals` bound what each geo spends over the plan, and what each channel spends
across the geos. At the best plan a cell inside its box returns, a currency unit, the budget's price
plus its geo's and its channel's: a total's price is what one more currency unit of room in it
returns, positive where it binds at its most, negative at its least, nothing where it does not bind.
A plan made in two steps, the budget split over the geos first and each geo's share over its
channels after, is one plan of the grid, so it never returns more. The best plan is found by cutting
planes on the cells' worths (Kelley 1960) under a linear program, whose value bounds every plan from
above while the plans it proposes approach from below; where every cell inside its box is strictly
concave, Newton's method on the binding totals' prices then makes the plan exact. The program's
duals name the totals that bind only to within its gap, so a total a hair from binding may be named
wrongly: one whose price comes out on the wrong side is released and one the plan breaks is bound,
until the plan meets the conditions that make it the best. :func:`budget_for_geos` finds the budget
that meets a goal over the grid: on concave curves the best plan's gain is concave in the budget
and its slope is the budget's price, so Newton's method on the budget meets a return target or a
return on spend in a few plans, and a marginal target is one plan, the budget left free and each
unit of it charged the target's return.

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
* On an S-shaped curve a goal is met on the path of the envelope's plans, the ones :func:`allocate`
  makes, and read on the true curves. The gain still rises along it, so a return target is the
  least budget of those plans, but a plan off the path may meet it for less; and a target return on
  ad spend is a budget where the average crosses the target, not proved the most.
* A split for several readings is robust to the readings it is given and to no other: it hedges
  between the families the tests could not tell apart, not against one none of them is. On an
  S-shaped curve the regret is the envelope's.
* The decision weight is local: second order in the error, at the plan on the channels as given,
  and a channel held at an end of its box carries no weight. One that would leave its end under a
  small error costs more than ``W`` says, as :mod:`chc.experiment`'s pinned levers do. It is read
  on concave curves only.
* :func:`chc.mmm.prescribe` plans a budget over a continuous plant whose adstock is a state; this
  plans discrete channels, which that plant does not describe.
* Over geos and channels, a plan with a cell inside its box on a straight stretch of its curve (an
  envelope's chord, a linear curve) is the cutting planes', within a share ``1e-9`` of the bound or
  as near as 500 rounds come; its rates are as near the best as that gap allows, not exact to
  rounding, and its prices are the last program's duals, which certify the bound but need not be
  the best plan's where the program is degenerate and several prices nearly certify it.
  Where every geo's spend and every channel's are fixed, the split of the price between the geos'
  totals and the channels' is the one the solver reaches; the sum each cell meets is the same
  however it is split. In single precision, JAX's default, Newton's method does not close and the
  cutting planes' plan comes back: its worth and bound are good to single precision, its rates
  only as near as the gap allows.
* A goal over geos and channels on an S-shaped curve is met where the true gain crosses it along
  the envelopes' plans, not proved the least or the most budget: under totals the plans need not
  all rise with the budget, so the true gain need not either.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.flatten_util import ravel_pytree
from jax.typing import ArrayLike
from scipy import sparse
from scipy.optimize import brentq, linprog

from chc.response import Channel, Logarithmic, Power, Saturation, relax

__all__ = [
    "Allocation",
    "AllocationWeight",
    "GeoAllocation",
    "Goal",
    "MarginalReturnTarget",
    "MinimaxAllocation",
    "ReturnOnSpendTarget",
    "ReturnTarget",
    "Totals",
    "allocate",
    "allocate_geos",
    "budget_for",
    "budget_for_geos",
    "decision_weight",
    "minimax_allocate",
]

_EPS = float(np.finfo(float).eps)
# the cutting planes stop when the worst regret is this share of the largest best return above
# their bound, or after this many rounds, the gap then reported as it stands
_GAP, _ROUNDS = 1e-9, 500


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
        budget: what the plan spends over its periods.
        idle: what the channels return over the same periods with nothing spent in the plan, the
            history's carryover alone.
    """

    spend: np.ndarray
    worth: float
    bound: float
    price: float
    budget: float
    idle: float

    @property
    def gain(self) -> float:
        """What the plan's spend adds to the channels' return: ``worth - idle``."""
        return self.worth - self.idle


@dataclass(frozen=True)
class ReturnTarget:
    """The least budget whose plan gains ``amount``, in the channels' units."""

    amount: float


@dataclass(frozen=True)
class MarginalReturnTarget:
    """The budget at which one more currency unit returns ``per_unit``: below it a unit returns
    more, past it less. On revenue, ``per_unit = 1`` is where spend stops paying for itself."""

    per_unit: float


@dataclass(frozen=True)
class ReturnOnSpendTarget:
    """The most budget whose plan gains ``per_unit`` for each currency unit it spends: a target
    return on ad spend, read on the gain."""

    per_unit: float


Goal = ReturnTarget | MarginalReturnTarget | ReturnOnSpendTarget


@dataclass(frozen=True)
class MinimaxAllocation:
    """A split chosen for the least worst regret over several readings of the channels.

    Attributes:
        spend: ``(channels,)`` spend a period, in the budget's currency.
        best: ``(readings,)`` each reading's best return at the budget, on its envelopes: the
            ``bound`` :func:`allocate` gives it.
        regret: ``(readings,)`` each reading's best return over the split's, were it true.
        worst: the largest regret, the split's.
        bound: no split in the box at the budget has a worst regret below this, so
            ``worst - bound`` bounds how far the split is from the least worst regret.
    """

    spend: np.ndarray
    best: np.ndarray
    regret: np.ndarray
    worst: float
    bound: float


@dataclass(frozen=True)
class AllocationWeight:
    """How much a plan loses when the channels it is made on are wrong. *Experimental.*

    Attributes:
        parameters: ``(p,)`` each parameter's place, the channel's position and the parameter's
            name in it, ``"1.curve.scale"``.
        matrix: ``(p, p)`` the regret's Hessian ``W`` in the parameters: a plan made on parameters
            off by ``d`` returns ``d' W d / 2`` less than the plan on the channels as given, to
            second order.
        allocation: the plan on the channels as given.
        pinned: the channels held at an end of their box, whose parameters carry no weight.
    """

    parameters: tuple[str, ...]
    matrix: np.ndarray
    allocation: Allocation
    pinned: tuple[int, ...]

    def expected_regret(self, covariance: ArrayLike) -> float:
        """``tr(W S) / 2`` at ``covariance = S``: what a plan made on estimates with that
        covariance loses on average, to second order.

        Raises:
            ValueError: when ``covariance`` is not ``(p, p)``.
        """
        spread = np.asarray(covariance, dtype=float)
        size = len(self.parameters)
        if spread.shape != (size, size):
            raise ValueError(f"covariance has shape {spread.shape}; the weight is {size} by {size}")
        return 0.5 * float(np.sum(self.matrix * spread))


@dataclass(frozen=True)
class Totals:
    """Bounds on what each of several groups of cells spends over the plan, every period
    together: ``least[k] <= periods * (group k's rates summed) <= most[k]``.

    :func:`allocate_geos` takes one for the geos, a geo's channels a group, and one for the
    channels, a channel across the geos a group. A fixed spend is ``least == most``; ``most`` may
    be infinite, and a ``least`` of zero binds nothing a box does not.

    Raises:
        ValueError: on bounds that are not one ``least`` and one ``most`` a group, with
            ``0 <= least <= most`` and ``least`` finite.
    """

    least: np.ndarray
    most: np.ndarray

    def __post_init__(self) -> None:
        least = np.asarray(self.least, dtype=float)
        most = np.asarray(self.most, dtype=float)
        if least.ndim != 1 or most.shape != least.shape:
            raise ValueError(
                f"least has shape {least.shape} and most {most.shape}; they need one bound a group"
            )
        if not (
            np.all(np.isfinite(least))
            and np.all(least >= 0.0)
            and not np.any(np.isnan(most))
            and np.all(most >= least)
        ):
            raise ValueError(f"the totals [{least}, {most}] are not 0 <= least <= most")
        object.__setattr__(self, "least", least)
        object.__setattr__(self, "most", most)


@dataclass(frozen=True)
class GeoAllocation:
    """A plan's spend a period for each geo's channels, what it returns, and what each constraint
    is worth.

    Attributes:
        spend: ``(geos, channels)`` spend a period, in the budget's currency.
        worth: the cells' return on the plan over its periods and each kernel's length after.
        bound: the most any plan meeting the constraints returns on the cells' concave envelopes;
            ``bound - worth`` bounds the plan's shortfall from the best on the cells, and is ``0``
            when every curve is concave and the plan is exact.
        price: the return on one more currency unit of budget, on the envelopes; ``0`` where
            every geo's total is fixed, or every channel's, which fixes the budget and carries its
            price.
        geo_prices: ``(geos,)`` the return on one more currency unit of room in each geo's total:
            positive where the total binds at its most, negative at its least, ``0`` where it does
            not bind or no geo totals were given.
        channel_prices: ``(channels,)`` the same for each channel's total across the geos.
        budget: what the plan spends over its periods.
        idle: what the cells return over the same periods with nothing spent in the plan, the
            history's carryover alone.
    """

    spend: np.ndarray
    worth: float
    bound: float
    price: float
    geo_prices: np.ndarray
    channel_prices: np.ndarray
    budget: float
    idle: float

    @property
    def gain(self) -> float:
        """What the plan's spend adds to the cells' return: ``worth - idle``."""
        return self.worth - self.idle


@dataclass(frozen=True)
class _Group:
    """One total: the cells it sums, in the grid's flat order, and its bounds over the plan."""

    members: np.ndarray
    least: float
    most: float


@dataclass(frozen=True)
class _Layout:
    """A grid of cells in one row, geo by geo, with each cell's box and history and the totals'
    groups, as :func:`allocate_geos` accepts them."""

    geos: int
    width: int
    cells: tuple[Channel, ...]
    lower: np.ndarray
    upper: np.ndarray
    history: np.ndarray
    geo_totals: Totals | None
    channel_totals: Totals | None
    groups: tuple[_Group, ...]  # the geos' totals, then the channels'


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


@eqx.filter_jit
def _curvature(worth: _Worth, rate: Array) -> Array:
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


def _check(
    channels: Sequence[Channel],
    periods: int,
    lower: np.ndarray,
    upper: np.ndarray,
    history: np.ndarray,
    names: Sequence[str] | None = None,
) -> None:
    if not channels:
        raise ValueError("no channels to allocate over")
    for index, channel in enumerate(channels):
        name = f"channel {index}" if names is None else names[index]
        if not isinstance(channel, Channel):
            raise TypeError(f"{name} is a {type(channel).__name__}, not a Channel")
        if not isinstance(channel.curve, Saturation | Logarithmic | Power):
            raise ValueError(
                f"{name}'s curve is a {type(channel.curve).__name__}, which is not "
                "increasing and concave or S-shaped; the allocation cannot certify a plan on it"
            )
        if not float(channel.coefficient) >= 0.0:
            raise ValueError(
                f"{name}'s coefficient is {float(channel.coefficient)}; a negative one "
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


def _inputs(
    channels: Sequence[Channel],
    periods: int,
    lower: ArrayLike,
    upper: ArrayLike,
    history: ArrayLike | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower_rates = np.asarray(lower, dtype=float)
    upper_rates = np.asarray(upper, dtype=float)
    spent = np.zeros((0, len(channels))) if history is None else np.asarray(history, dtype=float)
    _check(channels, periods, lower_rates, upper_rates, spent)
    return lower_rates, upper_rates, spent


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
    lower_rates, upper_rates, spent = _inputs(channels, periods, lower, upper, history)
    budget = float(budget)
    least, most = periods * float(lower_rates.sum()), periods * float(upper_rates.sum())
    if not (np.isfinite(budget) and least <= budget <= most):
        raise ValueError(
            f"a budget of {budget} is outside what the box spends over {periods} periods, "
            f"[{least}, {most}]"
        )
    given = tuple(channels)
    relaxed = relax(given)
    envelopes = _worths(relaxed, spent, periods)
    target = budget / periods

    def rates(price: float) -> np.ndarray:
        return _rates(envelopes, lower_rates, upper_rates, periods, price)

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
    return Allocation(
        spend=spend,
        worth=worth,
        bound=bound,
        price=0.5 * (cheap + dear),
        budget=budget,
        idle=sum(_value(w, 0.0) for w in worths),
    )


def _cross(
    rates: Callable[[float], np.ndarray],
    excess: Callable[[np.ndarray], float],
    few: np.ndarray,
    dear: float,
    many: np.ndarray,
) -> np.ndarray:
    """The least rates on the path where ``excess``, which does not fall as the rates rise,
    reaches nothing: ``few`` are the rates at price ``dear``, short of it, and ``many`` at price 0,
    not. Bisected on the price as :func:`allocate` bisects it; where a linear rate jumps, the plans
    either side are mixed to meet it, as :func:`allocate` mixes them to spend its budget."""
    cheap, tolerance = 0.0, 4 * _EPS * dear
    while dear - cheap > tolerance:
        middle = 0.5 * (cheap + dear)
        at = rates(middle)
        if excess(at) >= 0.0:
            many, cheap = at, middle
        else:
            few, dear = at, middle

    def mixed(share: float) -> float:
        return excess(few + share * (many - few))

    if mixed(1.0) <= 0.0:  # met at the cheap end, to rounding
        return many
    return few + brentq(mixed, 0.0, 1.0, xtol=4 * _EPS, rtol=4 * _EPS) * (many - few)


def budget_for(
    channels: Sequence[Channel],
    goal: Goal,
    periods: int,
    *,
    lower: ArrayLike,
    upper: ArrayLike,
    history: ArrayLike | None = None,
) -> Allocation:
    """The budget that meets ``goal``, and the plan :func:`allocate` makes with it.

    A plan's gain is :attr:`Allocation.gain`, what its spend adds to the channels' return over its
    periods and each kernel's length after. As the budget grows every rate rises along one path of
    plans, and each goal is a point on it:

    * :class:`ReturnTarget`: the least budget whose plan gains ``amount``. The gain rises along the
      path, so the budget is exact.
    * :class:`MarginalReturnTarget`: the rates at which each channel's slope meets ``per_unit``, or
      the end of its box it presses on. Where a channel's return is linear with that slope, its
      least rate.
    * :class:`ReturnOnSpendTarget`: the most budget whose gain is ``per_unit`` times the budget or
      more. The gain less ``per_unit`` a unit is concave in the budget and peaks at the marginal
      target's budget, so past that budget it falls, and the most budget is where it falls through
      nothing. The average can rise before the peak, where a floor holds spend on a channel that
      returns little a unit, so it can meet the target at a smaller budget too; that one is not
      returned.

    Args:
        channels, periods, lower, upper, history: as :func:`allocate` takes them.
        goal: what the budget must meet.

    Raises:
        TypeError: ``goal`` is none of the three, or a channel is not a
            :class:`chc.response.Channel`.
        ValueError: what :func:`allocate` refuses of the channels and the box; a goal whose value is
            not finite; a gain beyond what the box returns at its caps; a return on spend that no
            budget in the box reaches, which the error says by how much it falls short at the peak.
    """
    if not isinstance(goal, ReturnTarget | MarginalReturnTarget | ReturnOnSpendTarget):
        raise TypeError(f"goal is a {type(goal).__name__}, not a Goal")
    value = goal.amount if isinstance(goal, ReturnTarget) else goal.per_unit
    if not np.isfinite(value):
        raise ValueError(f"the goal's value is {value}; it needs a finite one")
    lower_rates, upper_rates, spent = _inputs(channels, periods, lower, upper, history)
    given = tuple(channels)
    relaxed = relax(given)
    envelopes = _worths(relaxed, spent, periods)
    worths = envelopes if relaxed is given else _worths(given, spent, periods)
    idle = sum(_value(w, 0.0) for w in worths)

    def rates(price: float) -> np.ndarray:
        return _rates(envelopes, lower_rates, upper_rates, periods, price)

    def gain(spend: np.ndarray) -> float:
        return sum(_value(w, rate) for w, rate in zip(worths, spend, strict=True)) - idle

    def cost(spend: np.ndarray) -> float:
        return periods * float(spend.sum())

    match goal:
        case MarginalReturnTarget(per_unit=per_unit):
            spend = rates(max(per_unit, 0.0))
        case ReturnTarget(amount=amount):
            most = gain(upper_rates)
            if amount > most:
                raise ValueError(
                    f"no plan in the box gains {amount}: at every cap the channels gain {most:.6g}"
                )
            # at the steepest slope the floors allow every channel sits at its floor, as allocate
            dear = max(
                0.0, *(_slope(w, low) for w, low in zip(envelopes, lower_rates, strict=True))
            )
            spend = (
                lower_rates
                if gain(lower_rates) >= amount
                else _cross(
                    rates, lambda at: gain(at) - amount, lower_rates, dear / periods, upper_rates
                )
            )
        case ReturnOnSpendTarget(per_unit=per_unit):

            def surplus(at: np.ndarray) -> float:
                return gain(at) - per_unit * cost(at)

            if surplus(upper_rates) >= 0.0:
                spend = upper_rates
            else:
                peak = rates(per_unit)  # per_unit > 0 here, since the gain is never negative
                if surplus(peak) < 0.0:
                    raise ValueError(
                        f"no budget in the box gains {per_unit} a currency unit it spends: the "
                        f"gain less {per_unit} a unit peaks at {surplus(peak):.6g}, at a budget "
                        f"of {cost(peak):.6g}"
                    )
                spend = _cross(rates, lambda at: -surplus(at), peak, per_unit, upper_rates)
    budget = cost(np.clip(spend, lower_rates, upper_rates))
    return allocate(given, budget, periods, lower=lower_rates, upper=upper_rates, history=spent)


def minimax_allocate(
    readings: Sequence[Sequence[Channel]],
    budget: float,
    periods: int,
    *,
    lower: ArrayLike,
    upper: ArrayLike,
    history: ArrayLike | None = None,
) -> MinimaxAllocation:
    """Spend ``budget`` for the least worst regret over ``readings`` of the channels.

    Each reading is a whole set of channels, one a column, as :func:`allocate` takes them: the same
    channels read another way, by another curve family, say. The regret of a split under a reading
    is the reading's best return at the budget over the split's, both on its envelopes, and the
    split returned has the least worst regret, to a share ``1e-9`` of the largest best return or as
    near as 500 rounds of cutting planes come. A reading's regret is convex in the split, so its
    tangent at any split lies under it: the planes are every reading's tangents at every split
    tried so far, the linear program over them is a bound from below, and each split it proposes is
    tried next. :attr:`MinimaxAllocation.bound` is the last program's value and ``worst`` the best
    split's, so the gap between them is what the split may still be from the least worst regret.

    Args:
        readings: the readings, each with one channel a column; at least one.
        budget, periods, lower, upper, history: as :func:`allocate` takes them.

    Raises:
        TypeError, ValueError: what :func:`allocate` refuses of any reading, the box, the history
            or the budget; no readings; readings of different numbers of channels.
    """
    readings = tuple(tuple(reading) for reading in readings)
    if not readings:
        raise ValueError("no readings to hedge between")
    size = len(readings[0])
    if any(len(reading) != size for reading in readings):
        raise ValueError(
            f"the readings have {sorted({len(r) for r in readings})} channels; each reads them all"
        )
    lower_rates, upper_rates, spent = _inputs(readings[0], periods, lower, upper, history)
    plans = [
        allocate(reading, budget, periods, lower=lower_rates, upper=upper_rates, history=spent)
        for reading in readings
    ]
    best = np.array([plan.bound for plan in plans])
    envelopes = [_worths(relax(reading), spent, periods) for reading in readings]
    rate = float(budget) / periods

    def returns(split: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Every reading's return on ``split``, and its slope in each channel's rate."""
        values = np.empty(len(readings))
        slopes = np.empty((len(readings), size))
        for index, worths in enumerate(envelopes):
            pairs = [
                _value_and_slope(w, jnp.asarray(r, dtype=float))
                for w, r in zip(worths, split, strict=True)
            ]
            values[index] = sum(float(value) for value, _ in pairs)
            slopes[index] = [float(slope) for _, slope in pairs]
        return values, slopes

    # t >= best - value - slope @ (s - split) for every reading, over s in the box spending the
    # budget; the variables are the rates and t, and t is the worst regret the planes allow. Both
    # sides are read in units of the largest best return: HiGHS refuses a program with an entry past
    # 1e15, and a reading's slope in currency reaches that when its fit has run to the edge of its
    # family, a coefficient of 1e35 on a curve barely bent
    rows: list[np.ndarray] = []
    limits: list[float] = []
    worst, chosen, regret = np.inf, plans[0].spend, best
    splits = [plan.spend for plan in plans]
    floor = 0.0
    scale = float(np.max(np.abs(best))) or 1.0
    for _ in range(_ROUNDS):
        for tried in splits:
            values, slopes = returns(tried)
            losses = best - values
            if losses.max() < worst:
                worst, chosen, regret = float(losses.max()), tried, losses
            rows.extend(np.concatenate([-slopes / scale, -np.ones((len(readings), 1))], axis=1))
            limits.extend((values - slopes @ tried - best) / scale)
        if worst - floor <= _GAP * scale:
            break
        program = linprog(
            np.concatenate([np.zeros(size), [1.0]]),
            A_ub=np.array(rows),
            b_ub=np.array(limits),
            A_eq=np.concatenate([np.ones(size), [0.0]])[None, :],
            b_eq=[rate],
            bounds=[*zip(lower_rates, upper_rates, strict=True), (0.0, None)],
            method="highs",
            options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10},
        )
        if program.status != 0:
            raise RuntimeError(f"the cutting planes' linear program failed: {program.message}")
        floor = max(floor, float(program.fun) * scale)
        splits = [_onto(program.x[:size], lower_rates, upper_rates, rate)]
    return MinimaxAllocation(
        spend=np.asarray(chosen), best=best, regret=regret, worst=worst, bound=min(floor, worst)
    )


def decision_weight(
    channels: Sequence[Channel],
    budget: float,
    periods: int,
    *,
    lower: ArrayLike,
    upper: ArrayLike,
    history: ArrayLike | None = None,
) -> AllocationWeight:
    """How much :func:`allocate`'s plan loses when the channels it is made on are wrong.
    *Experimental.*

    The arguments are :func:`allocate`'s, and so is the plan. Each channel's parameters are its
    inexact leaves, the kernel's, the curve's and the coefficient. Inside their boxes the rates meet
    at the budget's price, ``V_c'(u_c) = periods * price``; an error ``d_c`` in channel ``c`` moves
    its rate by ``g_c d_c / h_c``, ``g_c`` the slope's gradient in its parameters and ``h_c`` the
    worth's curvature ``-V_c''``, and the price moves every rate back by the share ``(1/h_c) /
    sum(1/h)`` of the total, so the budget stays spent. The first-order loss is nil, since the
    slopes are equal and the moves sum to nothing, and the second is ``du' diag(h) du / 2``, which
    is ``d' W d / 2``. ``validation/allocation_decision_weight.mac`` holds it to the Hessian of the
    realised loss for two exponential and two Michaelis-Menten channels.

    Raises:
        TypeError, ValueError: as :func:`allocate`; and ValueError on an S-shaped curve, whose plan
            is its envelope's, or a channel inside its box whose worth is not strictly concave at
            its rate.
    """
    allocation = allocate(channels, budget, periods, lower=lower, upper=upper, history=history)
    lower_rates, upper_rates, spent = _inputs(channels, periods, lower, upper, history)
    before = spent.shape[0]
    names: list[str] = []
    blocks: list[tuple[int, int]] = []
    curvature, response = [], []
    free = []
    for column, channel in enumerate(channels):
        if isinstance(channel.curve, Saturation) and channel.curve.inflection() > 0.0:
            raise ValueError(
                f"channel {column}'s curve is S-shaped; its plan is its envelope's, where the "
                "decision weight is not read"
            )
        parameters, static = eqx.partition(channel, eqx.is_inexact_array)
        flat, unravel = ravel_pytree(parameters)
        start = sum(size for _, size in blocks)
        blocks.append((start, flat.size))
        names += [f"{column}.{name}" for name in _names(parameters)]
        rate = float(allocation.spend[column])
        if not lower_rates[column] < rate < upper_rates[column]:
            continue
        tail = channel.kernel.length - 1
        idle = jnp.asarray(np.concatenate([spent[:, column], np.zeros(periods + tail)]))
        unit = jnp.asarray(np.concatenate([np.zeros(before), np.ones(periods), np.zeros(tail)]))

        def worth(at: Array, theta: Array, static=static, unravel=unravel, idle=idle, unit=unit):
            read = eqx.combine(unravel(theta), static)
            carry = read.kernel(idle)[before:]
            reach = read.kernel(unit)[before:]
            return read.coefficient * jnp.sum(read.curve(carry + at * reach))

        slope = jax.grad(worth)
        at = jnp.asarray(rate, dtype=flat.dtype)
        bend = -float(jax.grad(slope)(at, flat))
        if not bend > 0.0:
            raise ValueError(
                f"channel {column} runs inside its box where its worth is not strictly concave "
                f"(curvature {bend}); its rate does not answer an error smoothly"
            )
        free.append(column)
        curvature.append(bend)
        response.append(np.asarray(jax.grad(slope, argnums=1)(at, flat), dtype=float) / bend)
    size = sum(width for _, width in blocks)
    moves = np.zeros((len(free), size))  # each free rate's move per unit error in each parameter
    for row, (column, gain) in enumerate(zip(free, response, strict=True)):
        start, width = blocks[column]
        moves[row, start : start + width] = gain
    if free:
        share = (1.0 / np.array(curvature)) / np.sum(1.0 / np.array(curvature))
        moves -= np.outer(share, moves.sum(axis=0))
    weight = moves.T @ (np.array(curvature)[:, None] * moves) if free else np.zeros((size, size))
    return AllocationWeight(
        parameters=tuple(names),
        matrix=0.5 * (weight + weight.T),
        allocation=allocation,
        pinned=tuple(c for c in range(len(channels)) if c not in free),
    )


def allocate_geos(
    cells: Sequence[Sequence[Channel]],
    budget: float,
    periods: int,
    *,
    lower: ArrayLike,
    upper: ArrayLike,
    geo_totals: Totals | None = None,
    channel_totals: Totals | None = None,
    history: ArrayLike | None = None,
) -> GeoAllocation:
    """Spend ``budget`` over ``periods`` on every geo's channels at once, for the most return.

    Each cell, one geo's channel, runs at one rate a period as :func:`allocate`'s channels do, its
    worth counted the same way, carryover in from its history and out after the plan. The budget is
    spent exactly; each geo's spend over the plan, its channels together, stays within
    ``geo_totals``, and each channel's, its geos together, within ``channel_totals``.

    The plan is found by cutting planes on the cells' envelopes, each worth bounded above by its
    tangents under a linear program, until the program's value is within a share ``1e-9`` of the
    best plan it has proposed, or after 500 rounds. Where every cell inside its box is then strictly
    concave, Newton's method on the prices of the totals that bind closes the budget and those
    totals with each cell's rate exact at its prices; a step that does not bring them closer is
    replaced by the least of the dual, convex in the prices, along it. A total whose price comes out
    on the wrong side is released and one the plan breaks is bound, until each binding total's price
    has its side's sign and the others hold: the conditions for the best plan on concave worths.
    Otherwise the cutting planes' best plan is returned, with the program's value as its bound.

    Args:
        cells: one row a geo, each with the same channels in the same order.
        budget, periods: as :func:`allocate` takes them.
        lower, upper: ``(geos, channels)`` each cell's least and most spend a period.
        geo_totals: what each geo spends over the plan, its channels together; no bound when
            omitted.
        channel_totals: what each channel spends over the plan, its geos together; no bound when
            omitted.
        history: ``(T, geos, channels)`` spend in the periods before the plan, whose adstock runs
            into it; nothing spent before the plan when omitted.

    Raises:
        TypeError: a cell is not a :class:`chc.response.Channel`, or a total is not a
            :class:`Totals`.
        ValueError: what :func:`allocate` refuses of a cell, its box or its history; rows of
            different lengths; totals of the wrong length, or one outside what its cells' boxes
            spend; and a budget, boxes and totals no plan meets together.
    """
    layout = _layout(cells, periods, lower, upper, geo_totals, channel_totals, history)
    budget = float(budget)
    least, most = periods * float(layout.lower.sum()), periods * float(layout.upper.sum())
    if not (np.isfinite(budget) and least <= budget <= most):
        raise ValueError(
            f"a budget of {budget} is outside what the box spends over {periods} periods, "
            f"[{least}, {most}]"
        )
    return _plan(layout, periods, budget)


def budget_for_geos(
    cells: Sequence[Sequence[Channel]],
    goal: Goal,
    periods: int,
    *,
    lower: ArrayLike,
    upper: ArrayLike,
    geo_totals: Totals | None = None,
    channel_totals: Totals | None = None,
    history: ArrayLike | None = None,
) -> GeoAllocation:
    """The budget that meets ``goal`` over every geo's channels, and the plan
    :func:`allocate_geos` makes with it.

    The budgets run from the least to the most a plan within the boxes and the totals spends. On
    concave curves the best plan's gain is concave in the budget, and its slope there is the
    budget's price; each step below is one plan. Under totals the gain can fall past some budget:
    a cap on a geo and one on a channel can leave the last of the budget only to cells that return
    less than the spend they take from a cell in both.

    * :class:`ReturnTarget`: the least budget whose plan gains ``amount``, by Newton's method on
      the budget from the least, whose steps a concave gain keeps short of the target. Refused
      above the most any plan gains, the plan with nothing charged for its spend.
    * :class:`MarginalReturnTarget`: one plan, its budget left free and each currency unit it
      spends charged ``per_unit`` of return: every cell inside its box then returns ``per_unit`` a
      unit plus its totals' prices, and the budget is what the plan spends, an end of the budgets
      where the price stays on one side of ``per_unit`` across them.
    * :class:`ReturnOnSpendTarget`: the most budget whose gain is ``per_unit`` times the budget or
      more. The gain less ``per_unit`` a unit is concave in the budget and peaks where the price is
      ``per_unit``, so past that budget it falls, and the most budget is where it falls through
      nothing: Newton's method from the most budget.

    On an S-shaped curve the plans are the envelopes', and the gain is read on the true curves,
    which need not be concave in the budget: a return target and a return on spend are met where
    the true gain crosses them, by Brent's method between budgets either side, and neither is
    proved the least or the most.

    Args:
        cells, periods, lower, upper, geo_totals, channel_totals, history: as
            :func:`allocate_geos` takes them.
        goal: what the budget must meet.

    Raises:
        TypeError: ``goal`` is none of the three, or what :func:`allocate_geos` refuses as a type.
        ValueError: what :func:`allocate_geos` refuses of the cells, the boxes and the totals; a
            goal whose value is not finite; a gain beyond the most any plan returns; a return on
            spend no budget reaches, which the error says by how much it falls short at the peak.
    """
    if not isinstance(goal, ReturnTarget | MarginalReturnTarget | ReturnOnSpendTarget):
        raise TypeError(f"goal is a {type(goal).__name__}, not a Goal")
    value = goal.amount if isinstance(goal, ReturnTarget) else goal.per_unit
    if not np.isfinite(value):
        raise ValueError(f"the goal's value is {value}; it needs a finite one")
    grid = tuple(tuple(row) for row in cells)
    layout = _layout(grid, periods, lower, upper, geo_totals, channel_totals, history)
    least, most = _reach(layout, periods)
    concave = relax(layout.cells) is layout.cells
    plans: dict[float, GeoAllocation] = {}

    def plan(budget: float) -> GeoAllocation:
        budget = float(np.clip(budget, least, most))
        if budget not in plans:
            plans[budget] = allocate_geos(
                grid,
                budget,
                periods,
                lower=lower,
                upper=upper,
                geo_totals=geo_totals,
                channel_totals=channel_totals,
                history=history,
            )
        return plans[budget]

    match goal:
        case MarginalReturnTarget():
            return _plan(layout, periods, goal)
        case ReturnTarget(amount=amount):

            def short(p: GeoAllocation) -> float:
                return p.gain - amount

            start = plan(least)
            if short(start) >= 0.0:
                return start
            top = plan(_plan(layout, periods, MarginalReturnTarget(0.0)).budget)
            if short(top) < 0.0:
                raise ValueError(
                    f"no plan the constraints allow gains {amount}: the most a plan gains is "
                    f"{top.gain:.6g}, at a budget of {top.budget:.6g}"
                )
            found = _newton(plan, short, lambda p: p.price, start) if concave else None
            if found is not None:
                return found
            return plan(_crossing(plan, short, least, top.budget))
        case ReturnOnSpendTarget(per_unit=per_unit):

            def surplus(p: GeoAllocation) -> float:
                return p.gain - per_unit * p.budget

            end = plan(most)
            if surplus(end) >= 0.0:
                return end
            found = _newton(plan, surplus, lambda p: p.price - per_unit, end) if concave else None
            if found is not None:
                return found
            peak = plan(_plan(layout, periods, MarginalReturnTarget(per_unit)).budget)
            if surplus(peak) < 0.0:
                raise ValueError(
                    f"no budget the constraints allow gains {per_unit} a currency unit it spends: "
                    f"the gain less {per_unit} a unit peaks at {surplus(peak):.6g}, at a budget "
                    f"of {peak.budget:.6g}"
                )
            return plan(_crossing(plan, lambda p: -surplus(p), peak.budget, most))


def _layout(
    cells: Sequence[Sequence[Channel]],
    periods: int,
    lower: ArrayLike,
    upper: ArrayLike,
    geo_totals: Totals | None,
    channel_totals: Totals | None,
    history: ArrayLike | None,
) -> _Layout:
    """The grid laid out for planning, after what :func:`allocate_geos` refuses of it."""
    grid = tuple(tuple(row) for row in cells)
    if not grid or not grid[0]:
        raise ValueError("no cells to allocate over: it needs a geo with a channel")
    geos, width = len(grid), len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError(
            f"the geos have {sorted({len(row) for row in grid})} channels; each needs every channel"
        )
    flat = tuple(channel for row in grid for channel in row)
    low, high = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    for name, rates in (("lower", low), ("upper", high)):
        if rates.shape != (geos, width):
            raise ValueError(
                f"{name} has shape {rates.shape}; it needs {(geos, width)}, one a cell"
            )
    spent = np.zeros((0, geos, width)) if history is None else np.asarray(history, dtype=float)
    if spent.ndim != 3 or spent.shape[1:] != (geos, width):
        raise ValueError(f"history has shape {spent.shape}; it needs (T, {geos}, {width})")
    names = [f"geo {g}'s channel {c}" for g in range(geos) for c in range(width)]
    low, high, spent = low.ravel(), high.ravel(), spent.reshape(spent.shape[0], geos * width)
    _check(flat, periods, low, high, spent, names)
    cell = np.arange(geos * width).reshape(geos, width)
    groups: list[_Group] = []
    for kind, totals, members in (
        ("geo", geo_totals, list(cell)),
        ("channel", channel_totals, list(cell.T)),
    ):
        if totals is None:
            continue
        if not isinstance(totals, Totals):
            raise TypeError(f"{kind}_totals is a {type(totals).__name__}, not Totals")
        if totals.least.shape != (len(members),):
            raise ValueError(
                f"{kind}_totals bound {totals.least.size} groups; there are {len(members)} {kind}s"
            )
        for index, (group, floor, cap) in enumerate(
            zip(members, totals.least, totals.most, strict=True)
        ):
            reach = periods * float(low[group].sum()), periods * float(high[group].sum())
            if floor > reach[1] or cap < reach[0]:
                raise ValueError(
                    f"{kind} {index}'s total [{floor}, {cap}] is outside what its cells' boxes "
                    f"spend over {periods} periods, [{reach[0]}, {reach[1]}]"
                )
            groups.append(_Group(group, float(floor), float(cap)))
    return _Layout(geos, width, flat, low, high, spent, geo_totals, channel_totals, tuple(groups))


def _plan(layout: _Layout, periods: int, spend: float | MarginalReturnTarget) -> GeoAllocation:
    """The best plan of the grid that spends a budget, or that spends to where one more currency
    unit returns a marginal target's ``per_unit``: the budget left free, each unit of it charged
    that return.

    A budget is one more total, every cell's, fixed at the budget.
    """
    if isinstance(spend, MarginalReturnTarget):
        groups, charge = layout.groups, spend.per_unit
    else:
        everything = _Group(np.arange(layout.lower.size), spend, spend)
        groups, charge = (everything, *layout.groups), 0.0
    low, high = layout.lower, layout.upper
    given = layout.cells
    relaxed = relax(given)
    envelopes = _worths(relaxed, layout.history, periods)
    worths = envelopes if relaxed is given else _worths(given, layout.history, periods)
    rates, ceiling, prices = _outer(envelopes, low, high, groups, periods, charge)
    exact = _polish(envelopes, low, high, groups, periods, charge, prices)
    if exact is None:
        # the program bounds the worth less the charge, so the worth of any plan spending as much
        charged = charge * periods * float(rates.sum())
        bound = max(
            ceiling + charged, sum(_value(w, r) for w, r in zip(envelopes, rates, strict=True))
        )
    else:
        rates, prices = exact
        bound = sum(_value(w, r) for w, r in zip(envelopes, rates, strict=True))
    if isinstance(spend, MarginalReturnTarget):
        price, budget = charge, periods * float(rates.sum())
    else:
        price, prices, budget = float(prices[0]), prices[1:], spend
    worth = sum(_value(w, r) for w, r in zip(worths, rates, strict=True))
    geos, width = layout.geos, layout.width
    geo_totals, channel_totals = layout.geo_totals, layout.channel_totals
    split = iter(prices)
    geo_prices = (
        np.zeros(geos) if geo_totals is None else np.array([next(split) for _ in range(geos)])
    )
    channel_prices = np.zeros(width) if channel_totals is None else np.array(list(split))
    # every geo's total fixed, or every channel's, fixes the budget, whose price is then theirs: a
    # cell is in one geo and one channel, so the sum it meets does not move
    if geo_totals is not None and np.array_equal(geo_totals.least, geo_totals.most):
        geo_prices, price = geo_prices + price, 0.0
    elif channel_totals is not None and np.array_equal(channel_totals.least, channel_totals.most):
        channel_prices, price = channel_prices + price, 0.0
    return GeoAllocation(
        spend=rates.reshape(geos, width),
        worth=worth,
        bound=bound,
        price=price,
        geo_prices=geo_prices,
        channel_prices=channel_prices,
        budget=budget,
        idle=sum(_value(w, 0.0) for w in worths),
    )


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
        member[group.members] = 1.0
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


def _newton(
    plan: Callable[[float], GeoAllocation],
    excess: Callable[[GeoAllocation], float],
    slope: Callable[[GeoAllocation], float],
    start: GeoAllocation,
) -> GeoAllocation | None:
    """The plan whose ``excess``, concave in the budget, reaches nothing, by Newton's method from
    ``start``, where it falls short; ``slope`` is its derivative in the budget.

    A concave excess lies under its tangents, so from short of nothing every step stays short, and
    the steps go one way and shrink; a step that reaches nothing, which only rounding makes, is the
    last. None where the slope stops pointing toward nothing, so the excess peaks short of it.
    """
    current, heading = start, np.sign(slope(start))
    for _ in range(100):
        rise = slope(current)
        if rise == 0.0 or np.sign(rise) != heading:
            return None
        step = -excess(current) / rise
        current = plan(current.budget + step)
        if excess(current) >= 0.0 or abs(step) <= 4 * _EPS * current.budget:
            return current
    return None


def _crossing(
    plan: Callable[[float], GeoAllocation],
    excess: Callable[[GeoAllocation], float],
    short: float,
    past: float,
) -> float:
    """A budget where ``excess`` crosses nothing between ``short``, where it is below, and
    ``past``, where it is not, by Brent's method on the budget."""
    return brentq(
        lambda budget: excess(plan(budget)), short, past, xtol=4 * _EPS * past, rtol=4 * _EPS
    )


def _outer(
    envelopes: Sequence[_Worth],
    lower: np.ndarray,
    upper: np.ndarray,
    groups: Sequence[_Group],
    periods: int,
    charge: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """The best plan on the envelopes by cutting planes (Kelley 1960), each currency unit it
    spends charged ``charge`` of its worth.

    Each cell's worth is bounded above by its value at its cap and by its tangents; the linear
    program maximises the bounds less the charge over the plans that meet the constraints, and each
    plan it proposes adds a tangent where a cell's bound is loose. The program's value bounds every
    plan from above; the plans it proposes approach it from below. Returns the best plan's rates,
    the last value, and each group's price, the last program's duals in return a currency unit.
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
        member[group.members] = 1.0
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
    for _ in range(_ROUNDS):
        solved = len(limits)
        program = linprog(
            objective,
            A_ub=sparse.vstack(
                [sparse.coo_array((entries, (rows, columns)), shape=(solved, 2 * size)), totals]
            ),
            b_ub=np.array([*limits, *side_limits]),
            A_eq=equal if fixed else None,
            b_eq=np.array(fixed_limits) if fixed else None,
            bounds=box,
            method="highs",
            options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10},
        )
        if program.status == 2:
            raise ValueError("no plan meets the budget, the boxes and the totals together")
        if program.status != 0:
            raise RuntimeError(f"the plan's linear program failed: {program.message}")
        ceiling = min(ceiling, -float(program.fun) * scale)
        rates = np.clip(program.x[:size] * unit, lower, upper)
        heights = program.x[size:] * scale
        pairs = [tangent(cell, float(r)) for cell, r in enumerate(rates)]
        values = np.array([value for value, _ in pairs])
        charged = charge * periods * float(rates.sum())
        if values.sum() - charged > best:
            best, plan = float(values.sum()) - charged, rates
        level = abs(ceiling + charged)  # a share of the worth: less the charge it can be nothing
        if ceiling - best <= _GAP * level:
            break
        for cell in np.flatnonzero(heights - values > _GAP * level / size):
            value, slope = pairs[cell]
            if np.isfinite(slope):
                bound(int(cell), slope, value - slope * float(rates[cell]))
    duals = -scale / per
    prices = np.zeros(len(groups))
    for (index, sign), marginal in zip(sides, program.ineqlin.marginals[solved:], strict=True):
        prices[index] += sign * float(marginal) * duals
    for index, marginal in zip(fixed, program.eqlin.marginals, strict=True):
        prices[index] = float(marginal) * duals
    return plan, ceiling, prices


def _polish(
    envelopes: Sequence[_Worth],
    lower: np.ndarray,
    upper: np.ndarray,
    groups: Sequence[_Group],
    periods: int,
    charge: float,
    prices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """The plan made exact on the totals that bind, a fixed budget among them, or None.

    A cell's price is the charge on spend plus those of the binding totals it is in, and its rate
    is exact at it: where its slope meets ``periods`` times the price, or the end of its box the
    slope presses on. Newton's method on the binding totals' prices closes those totals, starting
    from the totals and prices the cutting planes found. A binding total whose
    price comes out on the wrong side is released, and a free total the plan breaks is bound where
    it breaks, until neither happens: the plan then meets the conditions for the best on concave
    worths. None where a free cell is not strictly concave, Newton's method does not close, or the
    binding totals keep changing.
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
            membership[row, groups[k].members] = 1.0
        closed = _close(
            envelopes, lower, upper, membership, targets, periods, charge, full[binding], tolerance
        )
        if closed is None:
            return None
        unknown, exact = closed
        full = np.zeros(len(groups))
        full[binding] = unknown
        floor = -1e-9 * float(np.max(np.abs(unknown), initial=charge))
        # held at its most and better spending less, or at its least and better spending more
        wrong = [k for k in binding if sides[k] * full[k] < floor]
        spent = np.array([periods * exact[group.members].sum() for group in groups])
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


def _close(
    envelopes: Sequence[_Worth],
    lower: np.ndarray,
    upper: np.ndarray,
    membership: np.ndarray,
    targets: np.ndarray,
    periods: int,
    charge: float,
    unknown: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """The prices, from ``unknown``, at which the cells' rates sum through ``membership`` to
    ``targets``, each cell's price ``charge`` plus its rows'; and those rates. None where a free
    cell is not strictly concave or the prices do not close.

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


def _names(parameters: Channel) -> list[str]:
    """Each flattened leaf's place, ``"curve.scale"``, with an index where a leaf is not a
    scalar."""
    names = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(parameters)[0]:
        name = ".".join(str(getattr(key, "name", key)) for key in path)
        count = int(np.size(leaf))
        names += [name] if count == 1 else [f"{name}[{k}]" for k in range(count)]
    return names


def _onto(split: np.ndarray, lower: np.ndarray, upper: np.ndarray, rate: float) -> np.ndarray:
    """The nearest split to ``split`` in the box whose rates sum to ``rate``: every rate shifted
    by one amount and clipped to its box, the amount where the sum is ``rate``."""

    def excess(shift: float) -> float:
        return float(np.clip(split + shift, lower, upper).sum()) - rate

    low, high = float(np.min(lower - split)), float(np.max(upper - split))
    if excess(low) >= 0.0:
        return lower.copy()
    if excess(high) <= 0.0:
        return upper.copy()
    return np.clip(split + brentq(excess, low, high, xtol=4 * _EPS * rate), lower, upper)
