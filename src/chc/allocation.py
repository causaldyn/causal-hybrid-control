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
from scipy.optimize import brentq, linprog

from chc.response import Channel, Logarithmic, Power, Saturation, relax

__all__ = [
    "Allocation",
    "AllocationWeight",
    "Goal",
    "MarginalReturnTarget",
    "MinimaxAllocation",
    "ReturnOnSpendTarget",
    "ReturnTarget",
    "allocate",
    "budget_for",
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
    # budget; the variables are the rates and t, and t is the worst regret the planes allow
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
            rows.extend(np.concatenate([-slopes, -np.ones((len(readings), 1))], axis=1))
            limits.extend(values - slopes @ tried - best)
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
        floor = max(floor, float(program.fun))
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
