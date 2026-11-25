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

**An S-shaped curve is planned by branch and bound.** Where a curve starts convex the worth is not
concave, and the bisection's plan on each curve's concave envelope (:class:`chc.response.Envelope`)
returns less on the curves than the best: two Hill curves of slope 3 and scales 1 and 1.01 at a
budget of 1.6 are planned 1.27/0.33 for 0.705 there, where 1.6 on the first returns 512/637, 0.804.
So the rates are searched in boxes (Udell and Boyd 2016). A box of a channel's rate is an interval
of each period's adstock, and the curve's envelope over that interval is the chord from its floor
to where it touches the curve, or to its cap where that comes first, and the curve beyond: concave
in the rate, above the worth, and equal to it at both ends. The bisection on those envelopes bounds
every plan in the box, and its plan, read on the curves, is a plan. The box of the largest bound
is cut first, on the channel whose envelope stands furthest above its curve at the plan, at its
rate, where both halves' envelopes meet the curve. :attr:`Allocation.bound` is the largest bound
left when every bound is within the search's tolerance of the best plan's worth, or once the next
cut would plan past ``max_boxes`` boxes, 500 by default; on concave curves it is the worth. The
tolerance is the caller's: the larger of ``atol`` and ``rtol`` of the first box's bound, a share
``1e-9`` by default, but never below 64 epsilons of the dtype the curves are read in, of that
bound, where the gap is rounding that no cut closes. :attr:`Allocation.stopped` says which,
:attr:`Allocation.limit` what stopped the search, and :attr:`Allocation.boxes` how many boxes
were planned; a search the cap stops logs a warning (``chc_event="allocation_cap"``) with its
gap. Rates that jump at one price, on chords of one slope, are filled one at a time, so at most
one channel is left inside its chord, where the curve is below the envelope. With kernels of
length one, floors at zero and caps past the tangencies, that bounds the gap of the first box
before any is cut: the largest
``periods * coefficient * nonconvexity`` (:meth:`chc.response.Saturation.nonconvexity`), Shapley
and Folkman's lemma for one constraint (Aubin and Ekeland 1976; Udell and Boyd 2016). It is all but
reached: four equal Hill curves of slope 3 at three scales leave a gap of 0.1544 against 0.1547.

**The price** is the budget's shadow price on the planned channels: the return one more currency
unit of budget buys, spread the way the plan spends it. It is where the channels running inside
their boxes meet; a channel at its cap returns more a unit and one at its floor less. On S-shaped
curves it is read on the curves at the plan, in the caller's box: the slope of the channel below
its cap where one more unit returns the most, the same in whichever box the search found the plan.

**A goal in place of a budget.** :func:`budget_for` finds the budget that meets a goal, and plans
it: the least budget that gains a return (:class:`ReturnTarget`), the budget at which one more unit
returns a given amount (:class:`MarginalReturnTarget`), or the most budget whose plan returns a
given amount a unit on average, a target return on ad spend (:class:`ReturnOnSpendTarget`). A
plan's gain is what its spend adds, ``worth - idle``, where ``idle`` is what the channels return
with nothing spent in the plan: the history's carryover alone. As the price falls every rate rises,
so the plans for every budget lie on one path, and each goal is a point on it. On S-shaped curves
each plan is :func:`allocate`'s search at its budget; those plans lie on no one path, and the
budget is found over them.

**A split for several readings of the channels.** Tests that never bent a curve fit several
families alike, and the families part where the plan goes. :func:`minimax_allocate` takes each
reading of every channel and chooses the split whose worst regret over them is least: the regret
under a reading is its best return at the budget over the split's, both on its curves, the best
its :func:`allocate` plan's. A reading's regret is convex in the split where its curves are
concave, so the worst is too, and cutting planes (Kelley 1960) close on it from below while the
splits they propose close on it from above; the two ends are the certificate. The worst regret is
the least of the readings' gains over their bests, negated, so it is the worst share below at one
reading, each gain read against the reading's best, and on S-shaped curves it is searched in boxes
as that share is.

**A split that gains in the worst share of a posterior's draws.** A fit's draws are readings too,
many and alike, and over them the worst case is one draw. :func:`cvar_allocate` takes the readings
and a reference split, the plan in place, and chooses the split whose mean gain over the reference
in the worst ``level`` share of the readings is the most: the gain's conditional value at risk
(Rockafellar and Uryasev 2000). Every gain is read on the curves, and the reference gains nothing
under any reading, so the split never does worse in that share than keeping it; at ``level = 1``
it is the split for the mean return. The share's mean gain is concave in the split where the curves
are, and the same cutting planes close on it from above. On S-shaped curves the rates are
searched in boxes, as :func:`allocate` searches them: in each box the planes are the tangents of
each curve's envelope over the box, which stand above the curve, so they bound the share's mean
gain in the box from above, and each split they propose, read on the curves, is a candidate, the
reference among them. Every reading's curves of a channel are read and bounded together, by one
compiled program for each structure of curve among them.

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
until the plan meets the conditions that make it the best. On S-shaped curves the rates are
searched in boxes, as :func:`allocate` searches them, the totals rows of each box's program: the
planes on each cell's envelope over the box bound every plan in it, and their plan, read on the
curves, is a plan. :func:`budget_for_geos` finds the budget that meets a goal over the grid: on
concave curves the best plan's gain is concave in the budget and its slope is the budget's price,
so Newton's method on the budget meets a return target or a return on spend in a few plans, and a
marginal target is one plan, the budget left free and each unit of it charged the target's return.

**A channel bought nationally.** A channel bought once for every geo, as national media allotted
by population is, has one spend a period, of which each geo's cell spends a fixed share
(:class:`National`). Both geo planners then choose that national spend, within its own box, in
place of its cells' rates. Its worth is its cells' summed, each reading its share, so it is
concave where their curves are; it counts in the budget and in its channel's total whole, and in
each geo's total by the geo's share. Inside its box it returns, a currency unit, the budget's price
plus its channel's and each geo's weighted by the geo's share. On S-shaped cells its worth over a
box of the national spend is bounded by its cells' envelopes over the box, summed: concave, above
the worth and equal to it at both ends of the box, so the box search cuts it as it cuts a cell.

HONEST SCOPE:

* The channels are read as given: a fitted channel's error goes straight into the plan. Where the
  channels come from an experiment (:func:`chc.lift.fit_lift`), a plan that runs a channel past
  the adstock the tests covered reads the family's shape, not the experiment.
* One rate a period for each channel unless :func:`allocate` is asked for a schedule: with a
  response that does not change with the period, only the history's carryover at the start and the
  tail at the end make periods differ. On S-shaped curves "the best plan" everywhere else here is
  the best of these flat plans, and a schedule that spends in bursts can return more than any of
  them: one Hill curve of slope 3 and scale 1, with no carryover, returns 0.105 on 1.2 spread over
  four periods and 0.633 on 1.2 spent in one, six times more. ``allocate(schedule=True)`` finds
  that schedule; where a kernel is longer than one period its schedule is the best of local
  ascents, bounded but not proved the best.
* The tail is counted with nothing spent after the plan. Spend that continues after it would take
  the tail further up the curve, so a long kernel's tail is valued at its most.
* :class:`chc.response.Ricker` is not monotone, and a negative coefficient turns a concave curve
  convex; the bisection proves nothing for either, and both are refused.
* On an S-shaped curve the best plan's gain rises with the budget but need not be concave in it,
  so a goal is met over searched plans by Brent's method on the budget: a return target where the
  gain crosses it, the least budget to the search's tolerance, and a target return on ad spend
  where the gain less the target a unit last falls through nothing past its peak, the most budget
  to the search's tolerance: each stretch of budgets is searched in boxes for a plan that meets the
  target, whose bound rules the stretch out where none does.
* On S-shaped curves every plan and split here is the best only to the search's tolerance, a share
  ``1e-9`` of its scale by default and never finer than 64 epsilons of the dtype, or as near as
  ``max_boxes`` boxes come, 500 by default; a goal's, at each budget it tries. Sums of S-shaped
  curves under a budget are NP-hard to plan (Udell and Boyd 2016), so nothing short of that cap
  bounds the count of boxes, and many channels near their thresholds at once can reach it; the gap
  is then reported as it stands. Each box of :func:`allocate`'s costs one bisection of the price,
  as a plan on concave curves does; each of :func:`cvar_allocate`'s and
  :func:`minimax_allocate`'s up to 30 rounds of planes, every one a read of each reading on its
  envelopes and its curves and a linear program over the planes kept, one row a reading for every
  split tried; each of :func:`allocate_geos`' as many rounds of planes as close it, up to 500,
  every one a linear program over the planes on its cells and the totals. The bound on the first
  box's gap holds only as stated above: a longer kernel runs each period at its own adstock, and a
  floor or a cap inside a chord holds its channel there, each with an excess of its own.
* A split for several readings is robust to the readings it is given and to no other: it hedges
  between the families the tests could not tell apart, not against one none of them is.
* A split for the worst share weighs every reading alike, as draws of a posterior are, and its
  share is of the readings, a probability only as far as they are a posterior's draws. Where they
  disagree on which way the budget should move, no move may gain on average in the worst share, and
  the split is then the reference.
* The decision weight is local: second order in the error, at the plan on the channels as given,
  and a channel held at an end of its box carries no weight. One that would leave its end under a
  small error costs more than ``W`` says, as :mod:`chc.experiment`'s pinned levers do. It is read
  on concave curves only.
* :func:`chc.research.mmm.prescribe` plans a budget over a continuous plant whose adstock is a
  state; this plans discrete channels, which that plant does not describe.
* Over geos and channels, a plan with a cell inside its box on a straight stretch of its curve (a
  linear curve, or an envelope's chord in a box of a search) is the cutting planes', within the
  search's tolerance of the bound or as near as 500 rounds come; its rates are as near the best as
  that gap allows, not exact to rounding, and its prices are the last program's duals, which
  certify the bound but need not be the best plan's where the program is degenerate and several
  prices nearly certify it. A searched plan is priced on the curves at the plan instead, by the
  prices that miss the conditions for the best plan least, the totals' misses first: its rates
  meet their prices only to the search's tolerance. In a search Newton's method on the prices
  stops at its first step that does not shrink the totals' miss: a chord holds its cell at an end
  of it at every price but its slope, and at that price the rate jumps across it, which a line
  search would circle. Where every geo's spend and every channel's are fixed, the split of the
  price between the geos' totals and the channels' is the one the solver reaches, or on S-shaped
  curves the one least in the sum of the prices' sizes; the sum each cell meets is the same
  however it is split. In single precision, JAX's default, Newton's method does not close and the
  cutting planes' plan comes back: its worth and bound are good to single precision, its rates
  only as near as the gap allows.
* A goal over geos and channels on an S-shaped curve is met where the searched plans' gain crosses
  it: a return target not proved the least budget, since under totals the best plan's gain need not
  rise with the budget; a return on spend the most, as :func:`budget_for` proves it.
* A national channel's shares are the caller's and held fixed: the plan chooses how much to buy,
  not where it lands. Its bound over a box of its spend, the sum of its cells' envelopes, can stand
  above the sum's own envelope, so on S-shaped cells it may take more boxes than one cell's.
"""

from __future__ import annotations

import importlib
import logging
import math
import numbers
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Literal, TypeVar

import equinox as eqx
import jax
import jax.monitoring
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.flatten_util import ravel_pytree
from jax.typing import ArrayLike
from scipy.optimize import brentq

from chc._allocation.curves import _EPS, _rates, _slope, _value, _Worth, _worths
from chc._allocation.geo import _geo_search, _Group, _Layout, _reach, _variables
from chc._allocation.readings import _cvar_search, _read, _reading_weights, _stacks
from chc._allocation.schedule import _scheduled
from chc._allocation.search import _GAP, _branch_and_bound, _fill, _rounding, _Settings, _split
from chc.frames import _real_numbers
from chc.response import (
    Channel,
    Logarithmic,
    Power,
    Saturation,
    relax,
)

__all__ = [
    "Allocation",
    "AllocationWeight",
    "CvarAllocation",
    "GeoAllocation",
    "Goal",
    "MarginalReturnTarget",
    "MinimaxAllocation",
    "National",
    "ReturnOnSpendTarget",
    "ReturnTarget",
    "SearchLimit",
    "SearchStatus",
    "Totals",
    "allocate",
    "allocate_geos",
    "budget_for",
    "budget_for_geos",
    "cvar_allocate",
    "decision_weight",
    "minimax_allocate",
]

# the phases of a compile JAX reports a duration for: tracing, lowering and the backend's compile
_COMPILING = frozenset(
    {
        "/jax/core/compile/jaxpr_trace_duration",
        "/jax/core/compile/jaxpr_to_mlir_module_duration",
        "/jax/core/compile/backend_compile_duration",
    }
)
# jax.monitoring publishes a way to remove a listener from jax 0.8.1; up to 0.8.0 its own module
# holds one
if hasattr(jax.monitoring, "unregister_event_duration_listener"):
    _unlisten = jax.monitoring.unregister_event_duration_listener
else:
    _unlisten = vars(importlib.import_module("jax._src.monitoring"))[
        "_unregister_event_duration_listener_by_callback"
    ]

SearchStatus = Literal["closed", "cap"]
"""Why a plan's search for the best stopped: ``closed``, its bound within the search's tolerance of
its worth; ``cap``, with the gap open at a limit (:data:`SearchLimit`)."""

SearchLimit = Literal["max_boxes", "rounds", "unsolved", "local"]
"""What stopped a search with its gap open: ``max_boxes``, the caller's cap, where the next box's
split would plan past it; ``rounds``, the 500 rounds of a box's cutting planes where no cut of the
box is left to close it: the one box of concave curves of :func:`cvar_allocate`,
:func:`minimax_allocate` and :func:`allocate_geos`, or a box of :func:`allocate_geos`' whose plan
holds every cell at an end of its interval; ``unsolved``, a linear program HiGHS left unsolved
there, at its iteration limit or with its optimality conditions unmet, which ends that box's
planes; ``local``, :func:`allocate`'s schedule over periods where a kernel is longer than one
period, which is the best of local ascents and whose bound is the curves' concave envelopes'."""

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Allocation:
    """A plan's spend a period for each channel, what it returns, and what the budget is worth.

    Attributes:
        spend: ``(channels,)`` spend a period, in the budget's currency.
        worth: the channels' return on the plan over its periods and each kernel's length after.
        bound: no plan in the box at the budget returns more on the channels; ``bound - worth``
            bounds the plan's shortfall from the best, and is ``0`` when every curve is concave. On
            S-shaped curves it is the largest bound :func:`allocate`'s search left.
        price: the return on one more currency unit of budget: on concave curves the budget's
            shadow price, on S-shaped curves read on the curves at the plan, where the unit goes to
            the channel below its cap that returns the most a unit.
        budget: what the plan spends over its periods.
        idle: what the channels return over the same periods with nothing spent in the plan, the
            history's carryover alone.
        stopped: whether ``bound - worth`` closed, or the search stopped at its cap
            (:data:`SearchStatus`).
        boxes: how many boxes of the rates were planned: 1 for a plan on concave curves.
        gap: ``bound - worth``.
        relative_gap: ``gap`` as a share of the first box's bound, the scale ``rtol`` is a share
            of, or of 1 where that bound is 0.
        tolerance: the gap the search closes at: the larger of ``atol`` and ``rtol`` of the first
            box's bound, but not below 64 epsilons of the dtype the curves are read in, of that
            bound.
        floored: whether that floor, the dtype's rounding, raised the tolerance above the
            caller's.
        limit: what stopped a search with its gap open (:data:`SearchLimit`); ``None`` where the
            gap closed.
        compile_seconds: the seconds JAX spent compiling during the call: tracing, lowering and
            the backend's compile, none where every program was compiled before.
        search_seconds: the rest of the call's seconds. Both are recorded, not promised.
        schedule: ``(periods, channels)`` spend in each period, where :func:`allocate` searched a
            schedule (``schedule=True``); ``spend`` is then each channel's mean over the periods.
            ``None`` for one rate a period.
    """

    spend: np.ndarray
    worth: float
    bound: float
    price: float
    budget: float
    idle: float
    stopped: SearchStatus
    boxes: int
    gap: float
    relative_gap: float
    tolerance: float
    floored: bool
    limit: SearchLimit | None
    compile_seconds: float
    search_seconds: float
    schedule: np.ndarray | None = None

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
        best: ``(readings,)`` each reading's best return at the budget on its curves: the
            ``worth`` of :func:`allocate`'s plan for it, exact on concave curves and the best to
            that search's tolerance on S-shaped ones where that search closed. A reading's search
            that stopped at its cap may leave its best short, and every regret read against it
            low; ``stopped`` is then ``cap``.
        regret: ``(readings,)`` each reading's best return over the split's, both on its curves,
            were it true.
        worst: the largest regret, the split's.
        bound: no split in the box at the budget has a worst regret below this, so
            ``worst - bound`` bounds how far the split is from the least worst regret. On S-shaped
            curves it is the least bound the search left.
        stopped: whether ``worst - bound`` closed to the search's tolerance with every reading's
            own search closed, or a search, the split's or a reading's own, stopped at its cap
            (:data:`SearchStatus`).
        boxes: how many boxes of the rates were searched: 1 where every curve is concave.
        gap: ``worst - bound``.
        relative_gap: ``gap`` as a share of the largest of ``best``, the scale ``rtol`` is a share
            of, or of 1 where every best return is 0.
        tolerance: the gap the search closes at: the larger of ``atol`` and ``rtol`` of the
            largest best return, but not below 64 epsilons of the dtype the curves are read in, of
            that return.
        floored: whether that floor, the dtype's rounding, raised the tolerance above the
            caller's.
        limit: what stopped the search with its gap open, or else the first reading's own
            search that stopped at its cap (:data:`SearchLimit`); ``None`` where every search
            closed.
        unsolved: the iterations of each linear program HiGHS left unsolved, in the order they
            were solved; each left its box the bound of the programs before it.
        compile_seconds: the seconds JAX spent compiling during the call, each reading's plan
            included: tracing, lowering and the backend's compile, none where every program was
            compiled before.
        search_seconds: the rest of the call's seconds. Both are recorded, not promised.
    """

    spend: np.ndarray
    best: np.ndarray
    regret: np.ndarray
    worst: float
    bound: float
    stopped: SearchStatus
    boxes: int
    gap: float
    relative_gap: float
    tolerance: float
    floored: bool
    limit: SearchLimit | None
    unsolved: tuple[int, ...]
    compile_seconds: float
    search_seconds: float


@dataclass(frozen=True)
class CvarAllocation:
    """A split chosen for the most mean gain over a reference in the worst share of the readings.

    Attributes:
        spend: ``(channels,)`` spend a period, in the budget's currency.
        gain: ``(readings,)`` each reading's return on the split less its return on the reference,
            both on its curves.
        cvar: the mean of the worst ``level`` share of ``gain``, the split's, each reading weighed
            by its ``weights`` where :func:`cvar_allocate` was given them; never below 0, which is
            the reference's own.
        bound: no split in the box at the budget has a ``cvar`` above this, so ``bound - cvar``
            bounds how far the split is from the best. On S-shaped curves it is the largest bound
            the search left.
        stopped: whether ``bound - cvar`` closed to the search's tolerance, or the search stopped
            at its cap (:data:`SearchStatus`).
        boxes: how many boxes of the rates were searched: 1 where every curve is concave.
        gap: ``bound - cvar``.
        relative_gap: ``gap`` as a share of the reference's largest return, among the readings of
            positive weight where weights were given, the scale ``rtol`` is a share of, or of 1
            where every such return on the reference is 0.
        tolerance: the gap the search closes at: the larger of ``atol`` and ``rtol`` of the
            reference's largest return, but not below 64 epsilons of the dtype the curves are read
            in, of that return.
        floored: whether that floor, the dtype's rounding, raised the tolerance above the
            caller's.
        limit: what stopped the search with its gap open (:data:`SearchLimit`); ``None`` where
            the gap closed.
        unsolved: the iterations of each linear program HiGHS left unsolved, in the order they
            were solved; each left its box the bound of the programs before it.
        readings: how many readings were passed, one a ``gain``; those of weight 0 among them
            are left out of the share the split was chosen over.
        compile_seconds: the seconds JAX spent compiling during the call: tracing, lowering and
            the backend's compile, none where every program was compiled before.
        search_seconds: the rest of the call's seconds. Both are recorded, not promised.
    """

    spend: np.ndarray
    gain: np.ndarray
    cvar: float
    bound: float
    stopped: SearchStatus
    boxes: int
    gap: float
    relative_gap: float
    tolerance: float
    floored: bool
    limit: SearchLimit | None
    unsolved: tuple[int, ...]
    readings: int
    compile_seconds: float
    search_seconds: float


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
        margin: ``(len(pinned),)`` how far each pinned channel's return on a currency unit at its
            end stands from the budget's price, on the side that holds it there: below the price
            at its floor, above it at its cap. The weight reads the loss of a plan made on errors
            that leave every pinned channel at its end. An error ``d`` uses up
            ``margin_gradient[k] @ d`` of channel ``k``'s margin, and past it the plan made on the
            errors spends on a channel the true plan holds at its end, a loss the weight does not
            read (:meth:`margin_sds`). ``inf`` for a channel whose box is a point, and where the
            budget fills every box or none; ``0`` where no channel runs inside its box otherwise,
            since the price is then an interval.
        margin_gradient: ``(len(pinned), p)`` how much of each margin a unit error in each
            parameter uses up: the pinned channel's slope's response to it, less the price's.
    """

    parameters: tuple[str, ...]
    matrix: np.ndarray
    allocation: Allocation
    pinned: tuple[int, ...]
    margin: np.ndarray
    margin_gradient: np.ndarray

    def expected_regret(self, covariance: ArrayLike) -> float:
        """``tr(W S) / 2`` at ``covariance = S``: what a plan made on estimates with that
        covariance loses on average, to second order.

        Raises:
            ValueError: when ``covariance`` is not a ``(p, p)`` covariance: finite, symmetric and
                positive semidefinite.
        """
        return 0.5 * float(np.sum(self.matrix * self._covariance(covariance)))

    def margin_sds(self, covariance: ArrayLike) -> np.ndarray:
        """Each pinned channel's ``margin`` in standard deviations of what errors of covariance
        ``S`` use up of it, ``margin / sqrt(g' S g)`` with ``g`` its ``margin_gradient``: the
        radius of :meth:`expected_regret`'s second order. ``inf`` where the errors use none of it.

        Raises:
            ValueError: when ``covariance`` is not a ``(p, p)`` covariance: finite, symmetric and
                positive semidefinite.
        """
        spread = self._covariance(covariance)
        squared = np.einsum("kp,pq,kq->k", self.margin_gradient, spread, self.margin_gradient)
        used = np.sqrt(np.maximum(squared, 0.0))  # a semidefinite S may round g' S g below zero
        reach = np.where(self.margin > 0.0, np.inf, 0.0)
        np.divide(self.margin, used, out=reach, where=used > 0.0)
        return reach

    def _covariance(self, covariance: ArrayLike) -> np.ndarray:
        """``covariance`` as a ``(p, p)`` covariance, refused where it is not one: a ``nan`` or a
        negative variance read as ``inf`` standard deviations, or as a regret of either sign. A
        relative ``1e-10`` of its largest entry is allowed for the rounding of an estimate."""
        spread = np.asarray(covariance, dtype=float)
        size = len(self.parameters)
        if spread.shape != (size, size):
            raise ValueError(f"covariance has shape {spread.shape}; the weight is {size} by {size}")
        if not np.all(np.isfinite(spread)):
            raise ValueError("covariance is not finite")
        scale = 1e-10 * float(np.max(np.abs(spread), initial=0.0))
        if not np.all(np.abs(spread - spread.T) <= scale):
            raise ValueError("covariance is not symmetric")
        least = float(np.linalg.eigvalsh(spread)[0]) if size else 0.0
        if least < -scale:
            raise ValueError(
                f"covariance is not positive semidefinite: its least eigenvalue is {least:.3g}"
            )
        return spread


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
        least = np.array(self.least, dtype=float)
        most = np.array(self.most, dtype=float)
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
        least.setflags(write=False)
        most.setflags(write=False)
        object.__setattr__(self, "least", least)
        object.__setattr__(self, "most", most)


@dataclass(frozen=True)
class National:
    """A channel bought once for every geo: one spend a period, of which each geo's cell spends its
    share, as national media allotted by population is.

    :func:`allocate_geos` and :func:`budget_for_geos` take one for each national channel, keyed by
    the channel's column in the grid. The plan chooses the national spend a period, between
    ``lower`` and ``upper``; geo ``g``'s cell of the channel spends ``shares[g]`` of it and reads
    that on its own curve, carryover and history. The cells have no box of their own.

    Attributes:
        shares: ``(geos,)`` each geo's share of the national spend, held as given.
        lower, upper: the least and the most national spend a period.

    Raises:
        ValueError: shares that are not one a geo, each positive and finite, summing to 1 within
            as many epsilons as there are shares: rounding, as shares a population divides into
            leave; bounds that are not finite with ``0 <= lower <= upper``.
    """

    shares: np.ndarray
    lower: float
    upper: float

    def __post_init__(self) -> None:
        shares = np.array(self.shares, dtype=float)
        if shares.ndim != 1 or shares.size == 0:
            raise ValueError(f"shares has shape {shares.shape}; it needs one share a geo")
        if not np.all(np.isfinite(shares) & (shares > 0.0)):
            raise ValueError(f"the shares {shares} are not each positive and finite")
        total = math.fsum(shares.tolist())
        if abs(total - 1.0) > shares.size * _EPS:
            raise ValueError(f"the shares sum to {total!r}, not 1: divide them by their sum")
        lower, upper = float(self.lower), float(self.upper)
        if not (math.isfinite(upper) and 0.0 <= lower <= upper):
            raise ValueError(
                f"the national box [{lower}, {upper}] is not 0 <= lower <= upper, finite"
            )
        shares.setflags(write=False)
        object.__setattr__(self, "shares", shares)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True)
class GeoAllocation:
    """A plan's spend a period for each geo's channels, what it returns, and what each constraint
    is worth.

    Attributes:
        spend: ``(geos, channels)`` spend a period, in the budget's currency. A national
            channel's column holds each geo's share of its national spend, which the column
            sums to, to rounding.
        worth: the cells' return on the plan over its periods and each kernel's length after.
        bound: no plan meeting the constraints and spending as much returns more on the cells;
            ``bound - worth`` bounds the plan's shortfall from the best, and is ``0`` when every
            curve is concave and the plan is exact. On S-shaped curves it is the largest bound
            :func:`allocate_geos`' search left.
        price: the return on one more currency unit of budget; ``0`` where every geo's total is
            fixed, or every channel's, which fixes the budget and carries its price. On S-shaped
            curves it and the totals' prices are read on the curves at the plan: the prices that
            meet the conditions for the best plan there, or miss them least, the budget's nearest
            nothing and then the totals' least; one geo with no totals is priced as
            :func:`allocate` prices its channels.
        geo_prices: ``(geos,)`` the return on one more currency unit of room in each geo's total:
            positive where the total binds at its most, negative at its least, ``0`` where it does
            not bind or no geo totals were given.
        channel_prices: ``(channels,)`` the same for each channel's total across the geos, a
            national channel's its national spend over the plan.
        budget: what the plan spends over its periods.
        idle: what the cells return over the same periods with nothing spent in the plan, the
            history's carryover alone.
        stopped: whether ``bound - worth`` closed to the search's tolerance, or the search stopped
            at a limit (:data:`SearchStatus`).
        boxes: how many boxes of the rates were planned: 1 where every curve is concave.
        gap: ``bound - worth``.
        relative_gap: ``gap`` as a share of the first box's bound, the scale ``rtol`` is a share
            of, or of 1 where that bound is 0.
        tolerance: the gap the search closes at: the larger of ``atol`` and ``rtol`` of the first
            box's bound, but not below 64 epsilons of the dtype the curves are read in, of that
            bound.
        floored: whether that floor, the dtype's rounding, raised the tolerance above the
            caller's.
        limit: what stopped the search with its gap open (:data:`SearchLimit`); ``None`` where
            the gap closed.
        compile_seconds: the seconds JAX spent compiling during the call: tracing, lowering and
            the backend's compile, none where every program was compiled before.
        search_seconds: the rest of the call's seconds. Both are recorded, not promised.
    """

    spend: np.ndarray
    worth: float
    bound: float
    price: float
    geo_prices: np.ndarray
    channel_prices: np.ndarray
    budget: float
    idle: float
    stopped: SearchStatus
    boxes: int
    gap: float
    relative_gap: float
    tolerance: float
    floored: bool
    limit: SearchLimit | None
    compile_seconds: float
    search_seconds: float

    @property
    def gain(self) -> float:
        """What the plan's spend adds to the cells' return: ``worth - idle``."""
        return self.worth - self.idle


_Plan = TypeVar("_Plan", Allocation, GeoAllocation)


def _check(  # noqa: C901
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
    spent = (
        np.zeros((0, len(channels)))
        if history is None
        else np.asarray(_real_numbers(history, "history"), dtype=float)
    )
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
    rtol: float = 1e-9,
    atol: float = 0.0,
    max_boxes: int = 500,
    schedule: bool = False,
) -> Allocation:
    """Spend ``budget`` over ``periods`` at one rate a period for each channel, for the most return,
    or with ``schedule=True`` at a spend of its own in each period.

    Exact where every curve is concave; on S-shaped curves the best plan of one rate a period to
    the tolerance ``rtol`` and ``atol`` set, by branch and bound (see the module). There a schedule
    that spends in bursts can return more: one Hill curve of slope 3 and scale 1, 1.2 over four
    periods, returns 0.105 at one rate a period and 0.633 spent in one.

    ``schedule=True`` searches each period's spend, each in the channel's box, the periods together
    spending the budget (:attr:`Allocation.schedule`). Where every kernel is one period long, each
    period's spend returns in its own period alone, and the schedule is searched in boxes over the
    periods and channels as the rates are: the best schedule to the same tolerance. Where a kernel
    is longer, the periods' adstocks share their spend and no such search applies: the schedule is
    the best of SLSQP ascents on the curves, each polished by Newton's method on the face of the box
    it stops on, from the best rate a period, from the best schedule on the curves' concave
    envelopes and from eight seeded random schedules that spend most of the budget in a few cells;
    never less than the best rate a period, and not proved the best. Its bound is the envelopes'
    schedule's worth plus its Frank--Wolfe gap, which no schedule's worth exceeds; where that bound
    is not within the tolerance of the worth, ``stopped`` is ``cap`` and ``limit`` ``local``, and
    the call logs a warning (``chc_event="allocation_local"``). That bound can be loose by much
    more than the schedule is short. On 23 plans of one or two Hill or Michaelis--Menten channels
    over four to six periods, through geometric kernels of two to four periods, each schedule
    matched the best of a brute-force search over a grid of schedules, polished by ascents, to
    rounding; on 19 of them ``stopped`` read ``cap``, and the bound stood 17% above the worth at
    the median and 72% at most. Read such a ``cap`` as the envelopes' slack, not as a schedule
    known to be short. On concave curves the envelopes are the curves, one ascent from the best
    rate a period is the best schedule, and the gap closes. The price is read on the curves at the
    schedule, as for one rate a period: the most one more unit returns in a period and channel
    below its cap.

    Args:
        channels: the channels, each read as given.
        budget: what the plan spends over its periods, every channel together.
        periods: how many periods the plan runs.
        lower, upper: ``(channels,)`` each channel's least and most spend a period. There is no
            default: a box is where the channels were seen, and a plan outside it reads the curves
            where nothing measured them.
        history: ``(T, channels)`` spend in the periods before the plan, whose adstock runs into
            it; nothing spent before the plan when omitted.
        rtol: the share of the first box's bound the search closes the gap to.
        atol: the gap the search closes to, in the channels' units. The search stops at the
            larger of the two, but not below 64 epsilons of the dtype the curves are read in, of
            the first box's bound: there the gap is rounding (:attr:`Allocation.floored`).
        max_boxes: the most boxes the search plans. A box is cut only where its halves fit
            within the count.
        schedule: search a spend for each period rather than one rate a period.

    Raises:
        TypeError: a channel is not a :class:`chc.response.Channel`.
        ValueError: a curve the bisection cannot certify on (see the module's scope), a negative
            coefficient, a box or history of the wrong shape, a box with ``lower > upper`` or a
            negative end, a budget the box cannot spend over the periods, an ``rtol`` or ``atol``
            that is negative or not finite, or a ``max_boxes`` that is not a whole number of at
            least 1.
    """
    settings = _Settings(rtol, atol, max_boxes)
    lower_rates, upper_rates, spent = _inputs(channels, periods, lower, upper, history)
    budget = _spendable(budget, periods, lower_rates, upper_rates)
    given = tuple(channels)
    with _Stopwatch() as clock:
        if relax(given) is given:
            plan = _on_envelopes(
                given, budget, periods, lower_rates, upper_rates, spent, settings, clock
            )
        else:
            worths = _worths(given, spent, periods)
            plan = _searched(worths, budget, periods, lower_rates, upper_rates, settings, clock)
        if schedule:
            plan = _scheduled(
                given, plan, periods, lower_rates, upper_rates, spent, settings, clock
            )
    if plan.limit == "local":
        _log.warning(
            "allocate's schedule is the best of local ascents: it returns %.6g on the curves, and "
            "no schedule more than %.6g",
            plan.worth,
            plan.bound,
            extra={"chc_event": "allocation_local", "worth": plan.worth, "bound": plan.bound},
        )
    else:
        _capped("allocate", plan)
    return plan


def _searched(
    worths: tuple[_Worth, ...],
    budget: float,
    periods: int,
    lower: np.ndarray,
    upper: np.ndarray,
    settings: _Settings,
    clock: _Stopwatch,
) -> Allocation:
    """:func:`allocate`'s plan on S-shaped curves, searched by :func:`_branch_and_bound`, its
    seconds read on ``clock``."""
    spend, worth, bound, price, ending = _branch_and_bound(
        worths, lower, upper, periods, budget / periods, settings
    )
    idle = sum(_value(w, 0.0) for w in worths)
    compiling, searching = clock.read()
    return Allocation(
        spend=spend,
        worth=worth,
        bound=bound,
        price=price,
        budget=budget,
        idle=idle,
        stopped=ending.stopped,
        boxes=ending.boxes,
        gap=bound - worth,
        relative_gap=(bound - worth) / ending.scale,
        tolerance=ending.tolerance,
        floored=ending.floored,
        limit=ending.limit,
        compile_seconds=compiling,
        search_seconds=searching,
    )


def _capped(planner: str, plan: Allocation | GeoAllocation) -> None:
    """Log once that ``planner``'s search stopped at a limit with its gap open."""
    if plan.stopped == "cap":
        _log.warning(
            "%s stopped at its cap with the gap open, after %d boxes: the plan returns %.6g on the "
            "curves, and no plan in the box more than %.6g",
            planner,
            plan.boxes,
            plan.worth,
            plan.bound,
            extra={
                "chc_event": "allocation_cap",
                "planner": planner,
                "boxes": plan.boxes,
                "limit": plan.limit,
                "worth": plan.worth,
                "bound": plan.bound,
            },
        )


# a plan on the envelopes from zero spend made with no settings, as the tests make one, is judged
# as a search of one box, at the share _GAP of its bound, unfloored
_UNSEARCHED = _Settings(_GAP, 0.0, 1, 0.0)


class _Stopwatch:
    """The seconds since it started, and how many of them JAX spent compiling in its thread.

    JAX reports a compile's phases as they end, each by its duration, so each is read as the span
    that ends as it is reported; a phase inside another, as jax before 0.10 reports a jit traced
    inside another's trace, counts once, within the other.
    """

    def __init__(self) -> None:
        self._thread = threading.get_ident()
        self._spans: list[tuple[float, float]] = []
        self._started = 0.0
        self._listener = self._heard

    def _heard(self, event: str, duration_secs: float, **kwargs: str | int) -> None:
        if event in _COMPILING and threading.get_ident() == self._thread:
            end = time.perf_counter()
            self._spans.append((end - duration_secs, end))

    def __enter__(self) -> _Stopwatch:
        jax.monitoring.register_event_duration_secs_listener(self._listener)
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exception: object) -> None:
        _unlisten(self._listener)

    def read(self) -> tuple[float, float]:
        """The seconds spent compiling so far, and the others since it started."""
        compiling, reach = 0.0, -np.inf
        for start, end in sorted(self._spans):
            if end > reach:
                compiling += end - max(start, reach)
                reach = end
        return compiling, max(time.perf_counter() - self._started - compiling, 0.0)


def _spendable(budget: float, periods: int, lower: np.ndarray, upper: np.ndarray) -> float:
    budget = float(budget)
    least, most = periods * float(lower.sum()), periods * float(upper.sum())
    if not (np.isfinite(budget) and least <= budget <= most):
        raise ValueError(
            f"a budget of {budget} is outside what the box spends over {periods} periods, "
            f"[{least}, {most}]"
        )
    return budget


def _on_envelopes(
    given: tuple[Channel, ...],
    budget: float,
    periods: int,
    lower: np.ndarray,
    upper: np.ndarray,
    spent: np.ndarray,
    settings: _Settings = _UNSEARCHED,
    clock: _Stopwatch | None = None,
) -> Allocation:
    """The plan on the curves' envelopes from zero spend (:func:`chc.response.relax`), with what it
    returns on the curves: :func:`allocate`'s own where every curve is concave, its bound its
    worth. On S-shaped curves, which no planner hands it, its bound is the envelopes', and a gap it
    leaves reads as a search's stopped at a cap of one box. Its gap is judged by ``settings`` on its
    bound, and its seconds are ``clock``'s, or its own where none runs."""
    with nullcontext(clock) if clock is not None else _Stopwatch() as running:
        relaxed = relax(given)
        envelopes = _worths(relaxed, spent, periods)
        spend, price = _split(envelopes, lower, upper, periods, budget / periods)
        bound = sum(_value(w, rate) for w, rate in zip(envelopes, spend, strict=True))
        worths = envelopes if relaxed is given else _worths(given, spent, periods)
        worth = sum(_value(w, rate) for w, rate in zip(worths, spend, strict=True))
        scale = abs(bound) or 1.0
        tolerance, floored = settings.tolerance(scale, _rounding(worths))
        idle = sum(_value(w, 0.0) for w in worths)
        compiling, searching = running.read()
    return Allocation(
        spend=spend,
        worth=worth,
        bound=bound,
        price=price,
        budget=budget,
        idle=idle,
        stopped="closed" if bound - worth <= tolerance else "cap",
        boxes=1,
        gap=bound - worth,
        relative_gap=(bound - worth) / scale,
        tolerance=tolerance,
        floored=floored,
        limit=None if bound - worth <= tolerance else "max_boxes",
        compile_seconds=compiling,
        search_seconds=searching,
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
    either side are mixed to meet it, as :func:`allocate` mixes them to spend its budget, one rate
    at a time (:func:`_fill`)."""
    cheap, tolerance = 0.0, 4 * _EPS * dear
    while dear - cheap > tolerance:
        middle = 0.5 * (cheap + dear)
        at = rates(middle)
        if excess(at) >= 0.0:
            many, cheap = at, middle
        else:
            few, dear = at, middle
    surplus = float((many - few).sum())

    def filled(amount: float) -> float:
        return excess(_fill(few, many, amount))

    if filled(surplus) <= 0.0:  # met at the cheap end, to rounding
        return many
    return _fill(few, many, brentq(filled, 0.0, surplus, xtol=4 * _EPS * surplus, rtol=4 * _EPS))


def budget_for(
    channels: Sequence[Channel],
    goal: Goal,
    periods: int,
    *,
    lower: ArrayLike,
    upper: ArrayLike,
    history: ArrayLike | None = None,
    rtol: float = 1e-9,
    atol: float = 0.0,
    max_boxes: int = 500,
) -> Allocation:
    """The budget that meets ``goal``, and the plan :func:`allocate` makes with it.

    A plan's gain is :attr:`Allocation.gain`, what its spend adds to the channels' return over its
    periods and each kernel's length after. Where every curve is concave, as the budget grows every
    rate rises along one path of plans, and each goal is a point on it:

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

    Where a curve is S-shaped the plans are :func:`allocate`'s search at each budget, and no one
    path holds them; the best plan's gain still rises with the budget, but it need not be concave
    in it. A return target is met by Brent's method on the budget, between the floors and the caps,
    where the searched plan's gain crosses it: the least budget, since that gain rises. A marginal
    target is the budget of the best plan with the budget free and each currency unit charged
    ``per_unit``, searched in boxes as :func:`allocate` searches a budget: no budget gains more less
    its charge, so below it a unit returns more on average and past it less. A return on spend is
    met where the searched gain less ``per_unit`` a unit falls through nothing past that peak,
    refused where the peak falls short. Past the peak the gain need not fall all the way: it can
    cross the target, rise back over it on a curve that turns up later, and cross again, and
    Brent's method returns any of the crossings. So the stretch from the peak to the caps is
    halved: its upper half is searched in boxes for the plan that gains the most less ``per_unit``
    a unit with its budget held in the half, as the marginal target is searched with it free. Where
    that search's bound falls short of the target no budget in the half meets it, and the stretch
    is the lower half; otherwise the stretch starts at that plan's budget. The budget returned is
    the most that meets the target, to the search's tolerance.

    Args:
        channels, periods, lower, upper, history: as :func:`allocate` takes them.
        goal: what the budget must meet.
        rtol, atol, max_boxes: each plan's search's, as :func:`allocate` takes them.

    Raises:
        TypeError: ``goal`` is none of the three, or a channel is not a
            :class:`chc.response.Channel`.
        ValueError: what :func:`allocate` refuses of the channels, the box and the search's
            settings; a goal whose value is not finite; a gain beyond what the box returns at its
            caps; a return on spend that no budget in the box reaches, which the error says by how
            much it falls short at the peak.
    """
    settings = _Settings(rtol, atol, max_boxes)
    with _Stopwatch() as clock:
        plan = _budget_for(channels, goal, periods, lower, upper, history, clock, settings)
    _capped("budget_for", plan)
    return plan


def _budget_for(  # noqa: C901
    channels: Sequence[Channel],
    goal: Goal,
    periods: int,
    lower: ArrayLike,
    upper: ArrayLike,
    history: ArrayLike | None,
    clock: _Stopwatch,
    settings: _Settings,
) -> Allocation:
    """:func:`budget_for`'s plan, its seconds read on ``clock``."""
    if not isinstance(goal, ReturnTarget | MarginalReturnTarget | ReturnOnSpendTarget):
        raise TypeError(f"goal is a {type(goal).__name__}, not a Goal")
    value = goal.amount if isinstance(goal, ReturnTarget) else goal.per_unit
    if not np.isfinite(value):
        raise ValueError(f"the goal's value is {value}; it needs a finite one")
    lower_rates, upper_rates, spent = _inputs(channels, periods, lower, upper, history)
    given = tuple(channels)
    worths = _worths(given, spent, periods)
    if relax(given) is not given:
        return _sought(worths, goal, periods, lower_rates, upper_rates, settings, clock)
    idle = sum(_value(w, 0.0) for w in worths)

    def rates(price: float) -> np.ndarray:
        return _rates(worths, lower_rates, upper_rates, periods, price)

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
            dear = max(0.0, *(_slope(w, low) for w, low in zip(worths, lower_rates, strict=True)))
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
    return _on_envelopes(given, budget, periods, lower_rates, upper_rates, spent, settings, clock)


def _sought(  # noqa: C901
    worths: tuple[_Worth, ...],
    goal: Goal,
    periods: int,
    lower: np.ndarray,
    upper: np.ndarray,
    settings: _Settings,
    clock: _Stopwatch,
) -> Allocation:
    """:func:`budget_for`'s plan where a curve is S-shaped: :func:`allocate`'s search at the budget
    that meets ``goal``, the budget found over searched plans, its seconds the whole call's on
    ``clock``."""
    least, most = periods * float(lower.sum()), periods * float(upper.sum())
    plans: dict[float, Allocation] = {}

    def plan(budget: float) -> Allocation:
        budget = float(np.clip(budget, least, most))
        if budget not in plans:
            plans[budget] = _searched(worths, budget, periods, lower, upper, settings, clock)
        return plans[budget]

    def peak(per_unit: float) -> Allocation:
        """The plan at the budget whose best plan gains the most less ``per_unit`` a unit."""
        spend = _branch_and_bound(worths, lower, upper, periods, 0.0, settings, per_unit)[0]
        return plan(periods * float(spend.sum()))

    match goal:
        case MarginalReturnTarget(per_unit=per_unit):
            found = peak(per_unit)
        case ReturnTarget(amount=amount):
            top = plan(most)
            if amount > top.gain:
                raise ValueError(
                    f"no plan in the box gains {amount}: at every cap the channels gain "
                    f"{top.gain:.6g}"
                )
            found = plan(least)
            if found.gain < amount:
                found = plan(_crossing(plan, lambda p: p.gain - amount, least, most))
        case ReturnOnSpendTarget(per_unit=per_unit):

            def surplus(p: Allocation) -> float:
                return p.gain - per_unit * p.budget

            found = plan(most)
            if surplus(found) < 0.0:
                # per_unit > 0 here, since the gain is never negative
                best = peak(per_unit)
                if surplus(best) < 0.0:
                    raise ValueError(
                        f"no budget in the box gains {per_unit} a currency unit it spends: the "
                        f"gain less {per_unit} a unit peaks at {surplus(best):.6g}, at a budget "
                        f"of {best.budget:.6g}"
                    )
                idle = found.idle

                def within(a: float, b: float) -> tuple[float, float]:
                    spend, _, ceiling, _, _ = _branch_and_bound(
                        worths, lower, upper, periods, 0.0, settings, per_unit, span=(a, b)
                    )
                    return ceiling - idle, periods * float(spend.sum())

                found = plan(
                    _last_crossing(
                        within,
                        lambda budget: -surplus(plan(budget)),
                        best.budget,
                        most,
                        found.tolerance / per_unit,
                    )
                )
    compiling, searching = clock.read()
    return replace(found, compile_seconds=compiling, search_seconds=searching)


def minimax_allocate(
    readings: Sequence[Sequence[Channel]],
    budget: float,
    periods: int,
    *,
    lower: ArrayLike,
    upper: ArrayLike,
    history: ArrayLike | None = None,
    rtol: float = 1e-9,
    atol: float = 0.0,
    max_boxes: int = 500,
) -> MinimaxAllocation:
    """Spend ``budget`` for the least worst regret over ``readings`` of the channels.

    Each reading is a whole set of channels, one a column, as :func:`allocate` takes them: the same
    channels read another way, by another curve family, say. The regret of a split under a reading
    is the reading's best return at the budget over the split's, both on its curves, the best being
    the return of :func:`allocate`'s plan for the reading. The split returned has the least worst
    regret, to the tolerance ``rtol`` and ``atol`` set, or as near as 500 rounds of cutting planes
    come on concave curves and ``max_boxes`` boxes of the rates past them;
    :attr:`MinimaxAllocation.stopped` says which, and :attr:`MinimaxAllocation.limit` what stopped
    it. A program HiGHS leaves unsolved, one it stalls on stopped by an iteration limit, leaves its
    box the bound of the programs before it.

    The worst regret is the least of the readings' gains over their bests, negated: the worst share
    :func:`cvar_allocate` maximises, at one reading of them, each gain read against the reading's
    best in place of a reference's return. It is searched as :func:`cvar_allocate` searches that
    share, from every reading's own plan. A reading's regret is convex in the split where its curves
    are concave, so the worst is too: the cutting planes are every reading's tangents at every split
    tried, the linear program over them bounds the worst regret from below, and each split it
    proposes is tried next. Where a curve is S-shaped the rates are searched in boxes, each box's
    planes cut on each curve's envelope over the box, which stands above the curve, so they bound
    every split in the box. :attr:`MinimaxAllocation.bound` is read from HiGHS's duals on each
    program as written and rounded down, not HiGHS's objective, which can stand above the program's
    least; ``worst`` is the best split's, so the gap between them is what the split may still be
    from the least worst regret. A reading's own plan returns its best, so no worst regret is below
    nothing where each best is exact; on S-shaped curves, below the least of the readings' own
    searches' gaps, negated. A reading's own search that stops at its cap logs a warning as
    :func:`allocate` does, and leaves ``stopped`` at ``cap`` however the split's search ends: its
    best may be short of the reading's, and every regret read against it low. Two readings of two
    channels at a budget of 1.6, one concave and one two Hill curves of slope 3 at scales 1.01 and
    1, under a cap of two boxes: the second reading's search stops at 0.705 of its best, 0.804,
    and the split returned regrets nothing against 0.705 and 0.090 against 0.804, where the least
    worst regret is 0.015.

    Args:
        readings: the readings, each with one channel a column; at least one.
        budget, periods, lower, upper, history: as :func:`allocate` takes them.
        rtol: the share of the largest best return the search closes the gap to, and each
            reading's own plan's search the share of its first box's bound.
        atol: the gap the search closes to, in the channels' units, and each reading's own plan's
            search. The search stops at the larger of the two, but not below 64 epsilons of the
            dtype the curves are read in, of the largest best return: there the gap is rounding
            (:attr:`MinimaxAllocation.floored`).
        max_boxes: the most boxes the search plans, and each reading's own plan's search. A box
            is cut only where its halves fit within the count.

    Raises:
        TypeError, ValueError: what :func:`allocate` refuses of any reading, the box, the history,
            the budget or the search's settings; no readings; readings of different numbers of
            channels.
    """
    settings = _Settings(rtol, atol, max_boxes)
    readings = tuple(tuple(reading) for reading in readings)
    if not readings:
        raise ValueError("no readings to hedge between")
    size = len(readings[0])
    if any(len(reading) != size for reading in readings):
        raise ValueError(
            f"the readings have {sorted({len(r) for r in readings})} channels; each reads them all"
        )
    lower_rates, upper_rates, spent = _inputs(readings[0], periods, lower, upper, history)
    budget = _spendable(budget, periods, lower_rates, upper_rates)
    for reading in readings[1:]:
        _check(reading, periods, lower_rates, upper_rates, spent)
    with _Stopwatch() as clock:
        plans = [
            allocate(
                reading,
                budget,
                periods,
                lower=lower_rates,
                upper=upper_rates,
                history=spent,
                rtol=rtol,
                atol=atol,
                max_boxes=max_boxes,
            )
            for reading in readings
        ]
        best = np.array([plan.worth for plan in plans])
        spend, gains, _, top, ending = _cvar_search(
            _stacks([_worths(reading, spent, periods) for reading in readings]),
            [plan.spend for plan in plans],
            best,
            min(plan.gap for plan in plans),
            budget / periods,
            1.0 / len(readings),
            lower_rates,
            upper_rates,
            settings,
        )
        compiling, searching = clock.read()
    # nothing less a gain, not the gain negated: a regret of nothing reads 0.0, not -0.0
    regret = 0.0 - gains
    worst, least = float(regret.max()), 0.0 - top
    gap_open = ending.stopped == "cap"
    short = next((plan for plan in plans if plan.stopped == "cap"), None)
    if short is not None and not gap_open:
        # a best its own search left short reads every regret against it low, however closely
        # the split's search closed on the least worst regret over such bests
        ending = replace(ending, stopped="cap", limit=short.limit)
    if ending.stopped == "cap":
        _log.warning(
            "minimax_allocate stopped at a cap after %d boxes, %s: the split's worst regret is "
            "%.6g, and no split in the box has less than %.6g",
            ending.boxes,
            "with the gap open"
            if gap_open
            else "a reading's own search short of its best, so each regret may read low",
            worst,
            least,
            extra={
                "chc_event": "allocation_cap",
                "planner": "minimax_allocate",
                "boxes": ending.boxes,
                "limit": ending.limit,
                "worst": worst,
                "bound": least,
            },
        )
    return MinimaxAllocation(
        spend=spend,
        best=best,
        regret=regret,
        worst=worst,
        bound=least,
        stopped=ending.stopped,
        boxes=ending.boxes,
        gap=worst - least,
        relative_gap=(worst - least) / ending.scale,
        tolerance=ending.tolerance,
        floored=ending.floored,
        limit=ending.limit,
        unsolved=ending.unsolved,
        compile_seconds=compiling,
        search_seconds=searching,
    )


def cvar_allocate(
    readings: Sequence[Sequence[Channel]],
    budget: float,
    periods: int,
    *,
    level: float,
    against: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    history: ArrayLike | None = None,
    rtol: float = 1e-9,
    atol: float = 0.0,
    max_boxes: int = 500,
    weights: ArrayLike | None = None,
) -> CvarAllocation:
    """Spend ``budget`` for the most mean gain over ``against`` in the worst ``level`` share of
    ``readings`` of the channels: the gain's conditional value at risk.

    Each reading is a whole set of channels, one a column, as :func:`minimax_allocate` takes them: a
    posterior's draws, say, weighed alike unless ``weights`` weigh them. A split's gain under a
    reading is its return less the reference's, both on the reading's curves, and the split returned
    has the most mean gain over the worst ``level`` share of the readings, to the tolerance ``rtol``
    and ``atol`` set, or as near as 500 rounds of cutting planes come on concave curves and
    ``max_boxes`` boxes of the rates past them; :attr:`CvarAllocation.stopped` says which, and
    :attr:`CvarAllocation.limit` what stopped it. A program HiGHS leaves unsolved, one it stalls on
    stopped by an iteration limit, leaves its box the bound of the programs before it. The reference
    spends the budget in the box and gains nothing under any reading, so the split returned never
    does worse in that share than keeping it. At ``level = 1`` the split has the most mean gain, the
    posterior's expected return; as the level falls it moves only as far as the worst readings agree
    it gains.

    The mean of the worst share is ``max_eta eta - E[(eta - gain)_+] / level`` (Rockafellar and
    Uryasev 2000), concave in the split where every curve is concave. The cutting planes (Kelley
    1960) are every reading's tangent at every split tried, starting from the reference, and the
    linear program over them bounds the most from above; its solution, moved onto the budget, is
    the next split tried.

    Where a curve starts convex the rates are searched in boxes, as :func:`allocate` searches them
    (Udell and Boyd 2016). In a box the planes are cut on each curve's envelope over the box, which
    stands above the curve and is concave in the rate, so the program bounds every split in the box,
    and each split it proposes, read on the curves, is a candidate. The box of the largest bound is
    cut first, on the channel whose envelope stands furthest above its curve at the box's best
    split on the envelopes, each reading weighed as the worst share weighs it there, since the
    share's mean gain on the envelopes exceeds the curves' by no more than that weighed excess
    (the risk envelope's dual). A box's planes stop after 30 rounds, or once their bound is within
    half the box's gap to the best split of its best on the envelopes, and the box is cut instead;
    a box's planes bound its halves too, so they carry over, those the last program left slack
    dropped. Two Hill curves of slope 3 at scales 1 and 1.01, a budget of 1.6 and the reference all
    on the first: on the envelopes the planes propose ``[1.27, 0.33]``, which returns 0.705 against
    the reference's 0.804, and the search closes on the reference, its ``cvar`` 0.

    A posterior that stacks several fits (Yao, Vehtari, Simpson and Gelman 2018) weighs its draws
    unequally, and ``weights`` carry that: the worst share's mean is then
    ``max_eta eta - sum_i w_i (eta - gain_i)_+ / level``, the weights scaled to sum to 1, its
    boundary a weighted quantile, the reading there taking the rest of the share. The planes, the
    boxes' cuts, ``cvar`` and ``bound`` all read the weights. Integer weights plan as the readings
    repeated that many times, to the search's tolerance, without a reading a copy; a reading of
    weight 0 is left out of the share, its gain still read on the split. Two linear readings, A
    returning 3 and 1 a unit on two channels and B 1 and 2, the reference the even split of 100, at
    the level 0.8: weighed alike the split moves all onto the first channel, ``cvar`` 6.25;
    weighed 1 and 3 it moves all onto the second, ``cvar`` 3.125.

    Args:
        readings: the readings, each with one channel a column; at least one.
        budget, periods, lower, upper, history: as :func:`allocate` takes them.
        level: the share of the readings, the worst, whose mean gain the split maximises, in
            ``(0, 1]``. There is no default: it is how much of the readings the plan may not lose
            on.
        against: ``(channels,)`` the reference's spend a period, in the box and spending the
            budget: the plan the split must beat, the current one at this budget, say.
        rtol: the share of the reference's largest return the search closes the gap to.
        atol: the gap the search closes to, in the channels' units. The search stops at the
            larger of the two, but not below 64 epsilons of the dtype the curves are read in, of
            the reference's largest return: there the gap is rounding
            (:attr:`CvarAllocation.floored`).
        max_boxes: the most boxes the search plans. A box is cut only where its halves fit
            within the count.
        weights: ``(readings,)`` each reading's weight in the share, finite and at least 0, not
            all 0, scaled to sum to 1; ``None``, the default, weighs every reading alike, as
            before weights were taken. With weights the tolerance's scale is the reference's
            largest return among the readings of positive weight.

    Raises:
        TypeError, ValueError: what :func:`allocate` refuses of any reading, the box, the history
            or the search's settings; no readings; readings of different numbers of channels; a
            level outside ``(0, 1]``; a reference of the wrong shape, outside the box or not
            spending the budget; weights of the wrong shape, not finite, negative or all 0.
    """
    settings = _Settings(rtol, atol, max_boxes)
    readings = tuple(tuple(reading) for reading in readings)
    if not readings:
        raise ValueError("no readings to plan over")
    size = len(readings[0])
    if any(len(reading) != size for reading in readings):
        raise ValueError(
            f"the readings have {sorted({len(r) for r in readings})} channels; each reads them all"
        )
    if not 0.0 < level <= 1.0:
        raise ValueError(f"level={level!r} is not a share of the readings in (0, 1]")
    weighed = None if weights is None else _reading_weights(weights, len(readings))
    lower_rates, upper_rates, spent = _inputs(readings[0], periods, lower, upper, history)
    for reading in readings[1:]:
        _check(reading, periods, lower_rates, upper_rates, spent)
    rate = float(budget) / periods
    reference = np.asarray(against, dtype=float)
    if reference.shape != (size,) or not np.all(np.isfinite(reference)):
        raise ValueError(f"against has shape {reference.shape}; it needs one finite rate a channel")
    if np.any(reference < lower_rates) or np.any(reference > upper_rates):
        raise ValueError(
            f"the reference {reference} is outside the box [{lower_rates}, {upper_rates}]"
        )
    if not abs(float(reference.sum()) - rate) <= 1e-9 * abs(rate):
        raise ValueError(
            f"the reference spends {float(reference.sum())} a period and the budget {rate}; the "
            "gain is read against a split of the same budget"
        )
    # a reading of no weight is no reading of the share's: the split is searched without it, and
    # its gain read on the split after
    kept = np.ones(len(readings), dtype=bool) if weighed is None else weighed > 0.0
    with _Stopwatch() as clock:
        worths = [_worths(reading, spent, periods) for reading in readings]
        curves = _stacks([w for w, keep in zip(worths, kept, strict=True) if keep])
        spend, gains, cvar, bound, ending = _cvar_search(
            curves,
            [reference],
            _read(curves, reference, int(kept.sum()))[0].sum(axis=1),
            math.inf,
            rate,
            level,
            lower_rates,
            upper_rates,
            settings,
            None if weighed is None else weighed[kept],
        )
        if not kept.all():
            left = _stacks([w for w, keep in zip(worths, kept, strict=True) if not keep])
            out = int((~kept).sum())
            on_split, on_reference = _read(left, spend, out)[0], _read(left, reference, out)[0]
            every = np.empty(len(readings))
            every[kept], every[~kept] = gains, on_split.sum(axis=1) - on_reference.sum(axis=1)
            gains = every
        compiling, searching = clock.read()
    if ending.stopped == "cap":
        _log.warning(
            "cvar_allocate stopped at its cap with the gap open, after %d boxes: the split's mean "
            "gain in the worst share is %.6g, and no split in the box has more than %.6g",
            ending.boxes,
            cvar,
            bound,
            extra={
                "chc_event": "allocation_cap",
                "planner": "cvar_allocate",
                "boxes": ending.boxes,
                "limit": ending.limit,
                "cvar": cvar,
                "bound": bound,
            },
        )
    return CvarAllocation(
        spend=spend,
        gain=gains,
        cvar=cvar,
        bound=bound,
        stopped=ending.stopped,
        boxes=ending.boxes,
        gap=bound - cvar,
        relative_gap=(bound - cvar) / ending.scale,
        tolerance=ending.tolerance,
        floored=ending.floored,
        limit=ending.limit,
        unsolved=ending.unsolved,
        readings=len(readings),
        compile_seconds=compiling,
        search_seconds=searching,
    )


def decision_weight(  # noqa: C901, PLR0915
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

    It is local. A channel at an end of its box carries no weight while the errors leave it there;
    past its :attr:`AllocationWeight.margin` the plan made on the errors moves it off, a loss the
    weight does not read, and :meth:`AllocationWeight.margin_sds` reads each margin in standard
    deviations of the errors. On three instances of two or three concave channels, one held at a
    floor of 0, under independent errors of 0.1--16% of each parameter: where the margin stood 2.7
    standard deviations or more, 400 plans' losses summed to their own ``d' W d / 2`` to 1.1%, and
    over 1600 plans the mean loss was 0.92--1.05 of ``tr(W S) / 2``, a miss the errors' sampling
    explains. At 1.2--1.4 standard deviations 8--13% of the plans moved the channel off its end, and
    the losses summed to 1.15--1.20 times their ``d' W d / 2``; at 0.3, 1.41 times. On the three
    exponential channels of ``validation/allocation_decision_weight.mac``'s STEP 3 with the third's
    coefficient raised to 320, 400 plans each (seed 11), the ratio read 1.002 +- 0.003 at a margin
    of 3 standard deviations, 1.09 +- 0.05 at 2, 1.27 +- 0.08 at 1.3 and 1.84 +- 0.13 at 0.3, where
    37% of the plans moved the channel off its floor.

    Raises:
        TypeError, ValueError: as :func:`allocate`; and ValueError on an S-shaped curve, where a
            small error can move the plan's spend from one channel to another at once, which no
            second-order weight reads, or a channel inside its box whose worth is not strictly
            concave at its rate.
    """
    lower_rates, upper_rates, spent = _inputs(channels, periods, lower, upper, history)
    for column, channel in enumerate(channels):
        if isinstance(channel.curve, Saturation) and channel.curve.inflection() > 0.0:
            raise ValueError(
                f"channel {column}'s curve is S-shaped, where a small error can move the plan's "
                "spend between channels at once and the decision weight is not read"
            )
    allocation = allocate(channels, budget, periods, lower=lower, upper=upper, history=history)
    before = spent.shape[0]
    names: list[str] = []
    blocks: list[tuple[int, int]] = []
    curvature, response, slopes = [], [], []
    free = []
    held: dict[int, tuple[float, float, np.ndarray]] = {}  # side, slope and its gradient at the end
    for column, channel in enumerate(channels):
        parameters, static = eqx.partition(channel, eqx.is_inexact_array)
        flat, unravel = ravel_pytree(parameters)
        start = sum(size for _, size in blocks)
        blocks.append((start, flat.size))
        names += [f"{column}.{name}" for name in _names(parameters)]
        rate = float(allocation.spend[column])
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
        if not lower_rates[column] < rate < upper_rates[column]:
            if lower_rates[column] < upper_rates[column]:
                held[column] = (
                    1.0 if rate <= lower_rates[column] else -1.0,
                    float(slope(at, flat)),
                    np.asarray(jax.grad(slope, argnums=1)(at, flat), dtype=float),
                )
            continue
        bend = -float(jax.grad(slope)(at, flat))
        if not bend > 0.0:
            raise ValueError(
                f"channel {column} runs inside its box where its worth is not strictly concave "
                f"(curvature {bend}); its rate does not answer an error smoothly"
            )
        free.append(column)
        curvature.append(bend)
        slopes.append(float(slope(at, flat)))
        response.append(np.asarray(jax.grad(slope, argnums=1)(at, flat), dtype=float) / bend)
    size = sum(width for _, width in blocks)
    moves = np.zeros((len(free), size))  # each free rate's move per unit error in each parameter
    for row, (column, gain) in enumerate(zip(free, response, strict=True)):
        start, width = blocks[column]
        moves[row, start : start + width] = gain
    # the price's move per unit error: the free rates' moves, each its slope's response over its
    # curvature, sum to nothing once the price moves by their sum over the sum of 1/curvature
    price_move = moves.sum(axis=0) / np.sum(1.0 / np.array(curvature)) if free else None
    if free:
        share = (1.0 / np.array(curvature)) / np.sum(1.0 / np.array(curvature))
        moves -= np.outer(share, moves.sum(axis=0))
    weight = moves.T @ (np.array(curvature)[:, None] * moves) if free else np.zeros((size, size))
    pinned = tuple(c for c in range(len(channels)) if c not in free)
    margin = np.full(len(pinned), np.inf)
    margin_gradient = np.zeros((len(pinned), size))
    # with no rate inside its box the budget fills every box or none, or else the price is an
    # interval between the channels held at their floors and those at their caps
    one_side = len({side for side, _, _ in held.values()}) <= 1
    for row, column in enumerate(pinned):
        if column not in held:
            continue  # a box of one point holds its channel whatever the errors
        side, slope_at_end, gradient = held[column]
        if price_move is None:
            margin[row] = np.inf if one_side else 0.0
            continue
        start, width = blocks[column]
        own = np.zeros(size)
        own[start : start + width] = gradient
        margin[row] = side * (float(np.mean(slopes)) - slope_at_end) / periods
        margin_gradient[row] = side * (own - price_move) / periods
    return AllocationWeight(
        parameters=tuple(names),
        matrix=0.5 * (weight + weight.T),
        allocation=allocation,
        pinned=pinned,
        margin=margin,
        margin_gradient=margin_gradient,
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
    national: Mapping[int, National] | None = None,
    history: ArrayLike | None = None,
    rtol: float = 1e-9,
    atol: float = 0.0,
    max_boxes: int = 500,
) -> GeoAllocation:
    """Spend ``budget`` over ``periods`` on every geo's channels at once, for the most return.

    Each cell, one geo's channel, runs at one rate a period as :func:`allocate`'s channels do, its
    worth counted the same way, carryover in from its history and out after the plan. The budget is
    spent exactly; each geo's spend over the plan, its channels together, stays within
    ``geo_totals``, and each channel's, its geos together, within ``channel_totals``.

    A channel bought once for every geo (:class:`National`) is one rate, its national spend a
    period, within its own box: each geo's cell spends its share of it. The plan chooses that rate
    as it chooses a cell's. Its worth is its cells' summed, each reading its share, and it counts in
    the budget and its channel's total whole and in each geo's total by the geo's share. Inside its
    box it returns, a currency unit, the budget's price plus its channel's and each geo's weighted
    by the geo's share.

    The plan is found by cutting planes on the cells' worths, each bounded above by its tangents
    under a linear program, until the program's bound is within the tolerance ``rtol`` and ``atol``
    set of the best plan it has proposed, or after 500 rounds, or at a program HiGHS leaves
    unsolved, one it stalls on stopped by an iteration limit; a first program left so is refused.
    Where every cell inside its box is then strictly concave, Newton's method on the prices of the
    totals that bind closes the budget and those totals with each cell's rate exact at its prices;
    a step that does not bring them closer is replaced by the least of the dual, convex in the
    prices, along it. A total whose price comes out on the wrong side is released and one the plan
    breaks is bound, until each binding total's price has its side's sign and the others hold: the
    conditions for the best plan on concave worths.
    Otherwise the cutting planes' best plan is returned, with the least bound their programs gave.
    Each program's bound is read from HiGHS's duals and rounded up, not HiGHS's objective, which
    can fall below the program's most.

    Where a curve is S-shaped the rates are searched in boxes, as :func:`allocate` searches them
    (Udell and Boyd 2016). In a box each cell's worth is bounded by its envelope over the box,
    concave in the rate, above the curve and equal to it at both ends, and the planes on those
    envelopes bound every plan in the box, under the same totals; their plan, read on the curves,
    is a plan. The box of the largest bound is cut first, on the cell inside its interval whose
    envelope stands furthest above its curve at the box's plan, at its rate there, so that both
    halves hold that plan. :attr:`GeoAllocation.bound` is the largest bound left when every bound is
    within the tolerance of the best plan's worth, or once the next cut would plan past
    ``max_boxes`` boxes; :attr:`GeoAllocation.stopped` says which, and a search the cap stops logs a
    warning (``chc_event="allocation_cap"``) with its gap. A national channel's worth over a box of
    its spend is bounded by its cells' envelopes over the box, summed, and it is cut as a cell is.

    Args:
        cells: one row a geo, each with the same channels in the same order.
        budget, periods: as :func:`allocate` takes them.
        lower, upper: ``(geos, channels)`` each cell's least and most spend a period; ``nan`` in a
            national channel's column, whose cells have no box of their own.
        geo_totals: what each geo spends over the plan, its channels together; no bound when
            omitted.
        channel_totals: what each channel spends over the plan, its geos together; no bound when
            omitted.
        national: the channels bought once for every geo, each by its column in the grid; none
            when omitted.
        history: ``(T, geos, channels)`` spend in the periods before the plan, whose adstock runs
            into it; nothing spent before the plan when omitted. A national channel's cells keep
            their own: each geo's share of the past need not be its share of the plan.
        rtol, atol, max_boxes: as :func:`allocate` takes them; the scale ``rtol`` is a share of is
            the first box's bound.

    Raises:
        TypeError: a cell is not a :class:`chc.response.Channel`, a total is not a
            :class:`Totals`, ``national`` is not a mapping, or one of its keys not a column or its
            values not a :class:`National`.
        ValueError: what :func:`allocate` refuses of a cell, its box, its history or the search's
            settings; rows of different lengths; totals of the wrong length, or one outside what
            its cells' boxes spend; a national channel the grid has no column for, or with shares
            not one a geo; a channel both national and given a box of its own in ``lower`` or
            ``upper``; and a budget, boxes and totals no plan meets together.
    """
    settings = _Settings(rtol, atol, max_boxes)
    layout = _layout(
        cells, periods, lower, upper, geo_totals, channel_totals, history, settings, national
    )
    budget = float(budget)
    least, most = periods * float(layout.lower.sum()), periods * float(layout.upper.sum())
    if not (np.isfinite(budget) and least <= budget <= most):
        raise ValueError(
            f"a budget of {budget} is outside what the box spends over {periods} periods, "
            f"[{least}, {most}]"
        )
    found = _plan(layout, periods, budget)
    _capped("allocate_geos", found)
    return found


def budget_for_geos(
    cells: Sequence[Sequence[Channel]],
    goal: Goal,
    periods: int,
    *,
    lower: ArrayLike,
    upper: ArrayLike,
    geo_totals: Totals | None = None,
    channel_totals: Totals | None = None,
    national: Mapping[int, National] | None = None,
    history: ArrayLike | None = None,
    rtol: float = 1e-9,
    atol: float = 0.0,
    max_boxes: int = 500,
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
      unit plus its totals' prices, a national channel ``per_unit`` plus its channel's and each
      geo's weighted by its share, and the budget is what the plan spends, an end of the budgets
      where the price stays on one side of ``per_unit`` across them.
    * :class:`ReturnOnSpendTarget`: the most budget whose gain is ``per_unit`` times the budget or
      more. The gain less ``per_unit`` a unit is concave in the budget and peaks where the price is
      ``per_unit``, so past that budget it falls, and the most budget is where it falls through
      nothing: Newton's method from the most budget.

    On an S-shaped curve each plan is :func:`allocate_geos`' search at its budget, and the best
    plan's gain need not be concave in the budget: a return target and a return on spend are met
    where that gain crosses them, by Brent's method between budgets either side, and neither is
    proved the least. A return on spend is met at the last crossing past the peak, proved the most
    budget as :func:`budget_for` proves it: each half of the stretch is searched with the budget
    held in it as one more total. The marginal target's plan is searched with its budget free.

    Args:
        cells, periods, lower, upper, geo_totals, channel_totals, national, history: as
            :func:`allocate_geos` takes them.
        goal: what the budget must meet.
        rtol, atol, max_boxes: each plan's search's, as :func:`allocate_geos` takes them.

    Raises:
        TypeError: ``goal`` is none of the three, or what :func:`allocate_geos` refuses as a type.
        ValueError: what :func:`allocate_geos` refuses of the cells, the boxes, the totals and the
            search's settings; a goal whose value is not finite; a gain beyond the most any plan
            returns; a return on spend no budget reaches, which the error says by how much it falls
            short at the peak.
    """
    settings = _Settings(rtol, atol, max_boxes)
    if not isinstance(goal, ReturnTarget | MarginalReturnTarget | ReturnOnSpendTarget):
        raise TypeError(f"goal is a {type(goal).__name__}, not a Goal")
    value = goal.amount if isinstance(goal, ReturnTarget) else goal.per_unit
    if not np.isfinite(value):
        raise ValueError(f"the goal's value is {value}; it needs a finite one")
    with _Stopwatch() as clock:
        layout = _layout(
            cells, periods, lower, upper, geo_totals, channel_totals, history, settings, national
        )
        found = _meet(layout, periods, goal)
        compiling, searching = clock.read()
    found = replace(found, compile_seconds=compiling, search_seconds=searching)
    _capped("budget_for_geos", found)
    return found


def _meet(layout: _Layout, periods: int, goal: Goal) -> GeoAllocation:  # noqa: C901
    """:func:`budget_for_geos`'s plan for ``goal`` over the grid ``layout`` lays out."""
    least, most = _reach(layout, periods)
    concave = relax(layout.cells) is layout.cells
    plans: dict[float, GeoAllocation] = {}

    def plan(budget: float) -> GeoAllocation:
        budget = float(np.clip(budget, least, most))
        if budget not in plans:
            plans[budget] = _plan(layout, periods, budget)
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
            worths = _variables(layout, periods)
            size = layout.lower.size

            def within(a: float, b: float) -> tuple[float, float]:
                spent = _Group(np.arange(size), a, b, np.ones(size))
                found, bound, _ = _geo_search(
                    worths,
                    layout.lower,
                    layout.upper,
                    (spent, *layout.groups),
                    periods,
                    per_unit,
                    layout.settings,
                    False,
                )
                ceiling = bound - found.charged - end.idle
                return ceiling, periods * float(found.rates.sum())

            return plan(
                _last_crossing(
                    within,
                    lambda budget: -surplus(plan(budget)),
                    peak.budget,
                    most,
                    end.tolerance / per_unit,
                )
            )


def _layout(  # noqa: C901, PLR0912
    cells: Sequence[Sequence[Channel]],
    periods: int,
    lower: ArrayLike,
    upper: ArrayLike,
    geo_totals: Totals | None,
    channel_totals: Totals | None,
    history: ArrayLike | None,
    settings: _Settings,
    national: Mapping[int, National] | None,
) -> _Layout:
    """The grid laid out for planning, after what :func:`allocate_geos` refuses of it, its plans
    searched to ``settings``: a rate for every geo's own cell, and one for each ``national``
    channel, whose cells spend their shares of it."""
    grid = tuple(tuple(row) for row in cells)
    if not grid or not grid[0]:
        raise ValueError("no cells to allocate over: it needs a geo with a channel")
    geos, width = len(grid), len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError(
            f"the geos have {sorted({len(row) for row in grid})} channels; each needs every channel"
        )
    flat = tuple(channel for row in grid for channel in row)
    low, high = np.array(lower, dtype=float), np.array(upper, dtype=float)
    for name, rates in (("lower", low), ("upper", high)):
        if rates.shape != (geos, width):
            raise ValueError(
                f"{name} has shape {rates.shape}; it needs {(geos, width)}, one a cell"
            )
    spent = (
        np.zeros((0, geos, width))
        if history is None
        else np.asarray(_real_numbers(history, "history"), dtype=float)
    )
    if spent.ndim != 3 or spent.shape[1:] != (geos, width):
        raise ValueError(f"history has shape {spent.shape}; it needs (T, {geos}, {width})")
    bought = _bought(national, geos, width)
    for column, channel in bought.items():
        if not (np.all(np.isnan(low[:, column])) and np.all(np.isnan(high[:, column]))):
            raise ValueError(
                f"channel {column} is national, and lower and upper give its cells a box of their "
                "own: they spend their shares of its national spend, within its own box; give "
                "nan there"
            )
        # each cell's share of the national box, for _check to read; the plan reads the box
        low[:, column] = channel.shares * channel.lower
        high[:, column] = channel.shares * channel.upper
    names = [f"geo {g}'s channel {c}" for g in range(geos) for c in range(width)]
    low, high, spent = low.ravel(), high.ravel(), spent.reshape(spent.shape[0], geos * width)
    _check(flat, periods, low, high, spent, names)
    cell = np.arange(geos * width).reshape(geos, width)
    order = sorted(bought)
    own = cell[:, [c for c in range(width) if c not in bought]].ravel()
    owner, share = np.empty(geos * width, dtype=int), np.ones(geos * width)
    owner[own] = np.arange(own.size)
    for k, column in enumerate(order):
        owner[cell[:, column]] = own.size + k
        share[cell[:, column]] = bought[column].shares
    least = np.concatenate([low[own], [bought[c].lower for c in order]])
    most = np.concatenate([high[own], [bought[c].upper for c in order]])
    channels = [
        (owner[column[:1]], np.ones(1)) if c in bought else (owner[column], share[column])
        for c, column in enumerate(cell.T)
    ]
    groups: list[_Group] = []
    for kind, totals, members in (
        ("geo", geo_totals, [(owner[row], share[row]) for row in cell]),
        ("channel", channel_totals, channels),
    ):
        if totals is None:
            continue
        if not isinstance(totals, Totals):
            raise TypeError(f"{kind}_totals is a {type(totals).__name__}, not Totals")
        if totals.least.shape != (len(members),):
            raise ValueError(
                f"{kind}_totals bound {totals.least.size} groups; there are {len(members)} {kind}s"
            )
        for index, ((group, weights), floor, cap) in enumerate(
            zip(members, totals.least, totals.most, strict=True)
        ):
            reach = (
                periods * float((least[group] * weights).sum()),
                periods * float((most[group] * weights).sum()),
            )
            if floor > reach[1] or cap < reach[0]:
                raise ValueError(
                    f"{kind} {index}'s total [{floor}, {cap}] is outside what its cells' boxes "
                    f"spend over {periods} periods, [{reach[0]}, {reach[1]}]"
                )
            groups.append(_Group(group, float(floor), float(cap), weights))
    return _Layout(
        geos,
        width,
        flat,
        spent,
        owner,
        share,
        len(order),
        least,
        most,
        geo_totals,
        channel_totals,
        tuple(groups),
        settings,
    )


def _bought(national: Mapping[int, National] | None, geos: int, width: int) -> dict[int, National]:
    """The national channels by column, after what :func:`allocate_geos` refuses of them."""
    if national is None:
        return {}
    if not isinstance(national, Mapping):
        raise TypeError(
            f"national is a {type(national).__name__}, not a mapping of channels' columns to "
            "National"
        )
    bought: dict[int, National] = {}
    for column, channel in national.items():
        if isinstance(column, bool) or not isinstance(column, numbers.Integral):
            raise TypeError(f"national names {column!r}, not a channel's column")
        if not 0 <= column < width:
            raise ValueError(
                f"national names channel {column}, and the grid has no channel {column}: its "
                f"columns are 0 to {width - 1}"
            )
        if not isinstance(channel, National):
            raise TypeError(
                f"national channel {column} is a {type(channel).__name__}, not a National"
            )
        if channel.shares.size != geos:
            raise ValueError(
                f"national channel {column} has {channel.shares.size} shares; {geos} geos need "
                "one each"
            )
        bought[int(column)] = channel
    return bought


def _plan(layout: _Layout, periods: int, spend: float | MarginalReturnTarget) -> GeoAllocation:
    """The best plan of the grid that spends a budget, or that spends to where one more currency
    unit returns a marginal target's ``per_unit``: the budget left free, each unit of it charged
    that return. Searched in boxes where a curve is S-shaped (:func:`_geo_search`), to the
    tolerance and cap ``layout.settings`` sets.

    A budget is one more total, every rate's, fixed at the budget: a national channel's shares
    sum to 1, so it counts whole.
    """
    size = layout.lower.size
    if isinstance(spend, MarginalReturnTarget):
        groups, charge = layout.groups, spend.per_unit
    else:
        everything = _Group(np.arange(size), spend, spend, np.ones(size))
        groups, charge = (everything, *layout.groups), 0.0
    with _Stopwatch() as clock:
        worths = _variables(layout, periods)
        found, bound, ending = _geo_search(
            worths,
            layout.lower,
            layout.upper,
            groups,
            periods,
            charge,
            layout.settings,
            not isinstance(spend, MarginalReturnTarget),
        )
        idle = sum(_value(w, 0.0) for w in worths)
        compiling, searching = clock.read()
    rates, prices, worth = found.rates, found.prices, found.worth
    if isinstance(spend, MarginalReturnTarget):
        price, budget = charge, periods * float(rates.sum())
    else:
        price, prices, budget = float(prices[0]), prices[1:], spend
    geos, width = layout.geos, layout.width
    geo_totals, channel_totals = layout.geo_totals, layout.channel_totals
    split = iter(prices)
    geo_prices = (
        np.zeros(geos) if geo_totals is None else np.array([next(split) for _ in range(geos)])
    )
    channel_prices = np.zeros(width) if channel_totals is None else np.array(list(split))
    # every geo's total fixed, or every channel's, fixes the budget, whose price is then theirs: a
    # cell is in one geo and one channel, and a national channel in every geo by shares that sum
    # to 1, so the sum each meets does not move
    if geo_totals is not None and np.array_equal(geo_totals.least, geo_totals.most):
        geo_prices, price = geo_prices + price, 0.0
    elif channel_totals is not None and np.array_equal(channel_totals.least, channel_totals.most):
        channel_prices, price = channel_prices + price, 0.0
    return GeoAllocation(
        spend=(rates[layout.owner] * layout.share).reshape(geos, width),
        worth=worth,
        bound=bound,
        price=price,
        geo_prices=geo_prices,
        channel_prices=channel_prices,
        budget=budget,
        idle=idle,
        stopped=ending.stopped,
        boxes=ending.boxes,
        gap=bound - worth,
        relative_gap=(bound - worth) / ending.scale,
        tolerance=ending.tolerance,
        floored=ending.floored,
        limit=ending.limit,
        compile_seconds=compiling,
        search_seconds=searching,
    )


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
    plan: Callable[[float], _Plan],
    excess: Callable[[_Plan], float],
    short: float,
    past: float,
) -> float:
    """A budget where ``excess`` crosses nothing between ``short``, where it is below, and
    ``past``, where it is not, by Brent's method on the budget."""
    return brentq(
        lambda budget: excess(plan(budget)), short, past, xtol=4 * _EPS * past, rtol=4 * _EPS
    )


def _last_crossing(
    within: Callable[[float, float], tuple[float, float]],
    excess: Callable[[float], float],
    short: float,
    past: float,
    resolution: float,
) -> float:
    """The most budget between ``short``, whose best plan meets a goal, and ``past``, whose does
    not, at which the best plan meets it, to ``resolution``.

    ``within(a, b)`` searches every plan whose budget lies in ``[a, b]`` for the one that meets the
    goal by the most, and returns a bound on how much any of them meets it by, below nothing where
    none does, and that plan's budget. Every budget past ``high`` misses the goal and ``low``'s
    best plan meets it; each round reads the upper half of the two, which either misses, lowering
    ``high`` to halfway, or holds a plan that meets, or meets to the search's tolerance, raising
    ``low`` to its budget, past halfway. No grid is read, so a crossing and its return inside any
    stretch are found: on S-shaped curves the gain less a target a unit can cross nothing three
    times past its peak. Once the two are within ``resolution``, Brent's method on ``excess``, the
    best plan's at a budget, closes the crossing between them to rounding."""
    low, high = short, past
    xtol = 4 * _EPS * past
    while high - low > max(xtol, resolution):
        middle = 0.5 * (low + high)
        ceiling, budget = within(middle, high)
        if ceiling < 0.0:
            high = middle
        else:
            low = max(budget, middle)
    if high > low:
        if excess(high) <= 0.0:
            return high
        if excess(low) <= 0.0:
            return brentq(excess, low, high, xtol=xtol, rtol=4 * _EPS)
    return low


def _names(parameters: Channel) -> list[str]:
    """Each flattened leaf's place, ``"curve.scale"``, with an index where a leaf is not a
    scalar."""
    names = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(parameters)[0]:
        name = ".".join(str(getattr(key, "name", key)) for key in path)
        count = int(np.size(leaf))
        names += [name] if count == 1 else [f"{name}[{k}]" for k in range(count)]
    return names
