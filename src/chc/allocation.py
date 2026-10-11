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
curves it is read on the envelopes of the box the plan was found in.

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
* On an S-shaped curve the best plan's gain rises with the budget but need not be concave in it,
  so a goal is met over searched plans by Brent's method on the budget: a return target where the
  gain crosses it, the least budget to the search's tolerance, and a target return on ad spend
  where the gain less the target a unit falls through nothing past its peak, not proved the most.
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
* :func:`chc.mmm.prescribe` plans a budget over a continuous plant whose adstock is a state; this
  plans discrete channels, which that plant does not describe.
* Over geos and channels, a plan with a cell inside its box on a straight stretch of its curve (a
  linear curve, or an envelope's chord in a box of a search) is the cutting planes', within the
  search's tolerance of the bound or as near as 500 rounds come; its rates are as near the best as
  that gap allows, not exact to rounding, and its prices are the last program's duals, which
  certify the bound but need not be the best plan's where the program is degenerate and several
  prices nearly certify it. In a search Newton's method on the prices stops at its first step that
  does not shrink the totals' miss: a chord holds its cell at an end of it at every price but its
  slope, and at that price the rate jumps across it, which a line search would circle.
  Where every geo's spend and every channel's are fixed, the split of the price between the geos'
  totals and the channels' is the one the solver reaches; the sum each cell meets is the same
  however it is split. In single precision, JAX's default, Newton's method does not close and the
  cutting planes' plan comes back: its worth and bound are good to single precision, its rates
  only as near as the gap allows.
* A goal over geos and channels on an S-shaped curve is met where the searched plans' gain crosses
  it, not proved the least or the most budget: under totals the best plan's gain need not rise with
  the budget.
"""

from __future__ import annotations

import heapq
import importlib
import logging
import math
import numbers
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from functools import partial
from typing import Literal, TypeVar, cast

import equinox as eqx
import jax
import jax.monitoring
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.flatten_util import ravel_pytree
from jax.typing import ArrayLike
from scipy import sparse
from scipy.optimize import OptimizeResult, brentq, linprog

from chc.response import (
    Channel,
    Logarithmic,
    Power,
    Saturation,
    _bisected_touches,
    _touches,
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

_EPS = float(np.finfo(float).eps)
_TINY = float(np.finfo(float).smallest_subnormal)
_HUGE = 2.0**300
# the cutting planes stop after this many rounds, the gap then reported as it stands; a plan on
# the envelopes from zero spend made with no settings judges its gap closed at this share of its
# bound
_GAP, _ROUNDS = 1e-9, 500
# a search's tolerance is never below this many epsilons of the dtype its curves are read in, of
# the scale its rtol is a share of: below it the gap is rounding, which no box closes
_ROUNDING = 64
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
# in a box of cvar_allocate's search the cutting planes stop after this many rounds, or once their
# bound is within half the box's gap to the best split of the box's best on its envelopes: a box is
# cut sooner than its planes are refined
_BOX_ROUNDS = 30
# linprog's statuses for a program HiGHS leaves unsolved, stopped by _planes' iteration limit or
# ended with its optimality conditions unmet (HiGHS's "Unknown"): it bounds nothing, and the
# planners keep the last program's bound
_UNSOLVED = (1, 4)

SearchStatus = Literal["closed", "cap"]
"""Why a plan's search for the best stopped: ``closed``, its bound within the search's tolerance of
its worth; ``cap``, with the gap open at a limit (:data:`SearchLimit`)."""

SearchLimit = Literal["max_boxes", "rounds", "unsolved"]
"""What stopped a search with its gap open: ``max_boxes``, the caller's cap, where the next box's
split would plan past it; ``rounds``, the 500 rounds of a box's cutting planes where no cut of the
box is left to close it: the one box of concave curves of :func:`cvar_allocate`,
:func:`minimax_allocate` and :func:`allocate_geos`, or a box of :func:`allocate_geos`' whose plan
holds every cell at an end of its interval; ``unsolved``, a linear program HiGHS left unsolved
there, at its iteration limit or with its optimality conditions unmet, which ends that box's
planes."""

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
        price: the return on one more currency unit of budget, on the envelopes the plan was found
            on.
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
            that search's tolerance on S-shaped ones.
        regret: ``(readings,)`` each reading's best return over the split's, both on its curves,
            were it true.
        worst: the largest regret, the split's.
        bound: no split in the box at the budget has a worst regret below this, so
            ``worst - bound`` bounds how far the split is from the least worst regret. On S-shaped
            curves it is the least bound the search left.
        stopped: whether ``worst - bound`` closed to the search's tolerance, or the search stopped
            at its cap (:data:`SearchStatus`).
        boxes: how many boxes of the rates were searched: 1 where every curve is concave.
        gap: ``worst - bound``.
        relative_gap: ``gap`` as a share of the largest of ``best``, the scale ``rtol`` is a share
            of, or of 1 where every best return is 0.
        tolerance: the gap the search closes at: the larger of ``atol`` and ``rtol`` of the
            largest best return, but not below 64 epsilons of the dtype the curves are read in, of
            that return.
        floored: whether that floor, the dtype's rounding, raised the tolerance above the
            caller's.
        limit: what stopped the search with its gap open (:data:`SearchLimit`); ``None`` where
            the gap closed.
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
        cvar: the mean of the worst ``level`` share of ``gain``, the split's; never below 0, which
            is the reference's own.
        bound: no split in the box at the budget has a ``cvar`` above this, so ``bound - cvar``
            bounds how far the split is from the best. On S-shaped curves it is the largest bound
            the search left.
        stopped: whether ``bound - cvar`` closed to the search's tolerance, or the search stopped
            at its cap (:data:`SearchStatus`).
        boxes: how many boxes of the rates were searched: 1 where every curve is concave.
        gap: ``bound - cvar``.
        relative_gap: ``gap`` as a share of the reference's largest return, the scale ``rtol`` is
            a share of, or of 1 where every return on the reference is 0.
        tolerance: the gap the search closes at: the larger of ``atol`` and ``rtol`` of the
            reference's largest return, but not below 64 epsilons of the dtype the curves are read
            in, of that return.
        floored: whether that floor, the dtype's rounding, raised the tolerance above the
            caller's.
        limit: what stopped the search with its gap open (:data:`SearchLimit`); ``None`` where
            the gap closed.
        unsolved: the iterations of each linear program HiGHS left unsolved, in the order they
            were solved; each left its box the bound of the programs before it.
        readings: how many readings the split was chosen over.
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
class GeoAllocation:
    """A plan's spend a period for each geo's channels, what it returns, and what each constraint
    is worth.

    Attributes:
        spend: ``(geos, channels)`` spend a period, in the budget's currency.
        worth: the cells' return on the plan over its periods and each kernel's length after.
        bound: no plan meeting the constraints and spending as much returns more on the cells;
            ``bound - worth`` bounds the plan's shortfall from the best, and is ``0`` when every
            curve is concave and the plan is exact. On S-shaped curves it is the largest bound
            :func:`allocate_geos`' search left.
        price: the return on one more currency unit of budget, on the envelopes of the box the plan
            was found in; ``0`` where every geo's total is fixed, or every channel's, which fixes
            the budget and carries its price.
        geo_prices: ``(geos,)`` the return on one more currency unit of room in each geo's total:
            positive where the total binds at its most, negative at its least, ``0`` where it does
            not bind or no geo totals were given.
        channel_prices: ``(channels,)`` the same for each channel's total across the geos.
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


@dataclass(frozen=True)
class _Group:
    """One total: the cells it sums, in the grid's flat order, and its bounds over the plan."""

    members: np.ndarray
    least: float
    most: float


@dataclass(frozen=True)
class _Layout:
    """A grid of cells in one row, geo by geo, with each cell's box and history and the totals'
    groups, as :func:`allocate_geos` accepts them, and the settings its plans are searched to."""

    geos: int
    width: int
    cells: tuple[Channel, ...]
    lower: np.ndarray
    upper: np.ndarray
    history: np.ndarray
    geo_totals: Totals | None
    channel_totals: Totals | None
    groups: tuple[_Group, ...]  # the geos' totals, then the channels'
    settings: _Settings


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
    rtol: float = 1e-9,
    atol: float = 0.0,
    max_boxes: int = 500,
) -> Allocation:
    """Spend ``budget`` over ``periods`` at one rate a period for each channel, for the most return.

    Exact where every curve is concave; on S-shaped curves the best to the tolerance ``rtol`` and
    ``atol`` set, by branch and bound (see the module).

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
            return _on_envelopes(
                given, budget, periods, lower_rates, upper_rates, spent, settings, clock
            )
        worths = _worths(given, spent, periods)
        plan = _searched(worths, budget, periods, lower_rates, upper_rates, settings, clock)
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


# a plan on the envelopes from zero spend made with no settings, as the tests make one, is judged
# as a search of one box, at the share _GAP of its bound, unfloored
_UNSEARCHED = _Settings(_GAP, 0.0, 1, 0.0)


def _rounding(worths: object) -> float:
    """The epsilon of the dtype ``worths``, a pytree of them, are read in."""
    leaves = jax.tree.leaves(eqx.filter(worths, eqx.is_inexact_array))
    return float(jnp.finfo(jnp.result_type(*leaves)).eps)


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


def _branch_and_bound(
    worths: tuple[_Worth, ...],
    lower: np.ndarray,
    upper: np.ndarray,
    periods: int,
    target: float,
    settings: _Settings,
    charge: float | None = None,
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
    tolerance is still the first node's bound in worth.
    """

    def solve(low: np.ndarray, high: np.ndarray, bounded: tuple[_Worth, ...]):
        if charge is None:
            spend, price = _split(bounded, low, high, periods, target)
            spent = 0.0
        else:
            spend, price = _rates(bounded, low, high, periods, charge), charge
            spent = charge * periods * float(spend.sum())
        tops = np.array([_value(b, rate) for b, rate in zip(bounded, spend, strict=True)])
        values = np.array([_value(w, rate) for w, rate in zip(worths, spend, strict=True)])
        return spend, price, float(tops.sum()) - spent, float(values.sum()) - spent, tops - values

    bounded = tuple(_bounded(w, a, b) for w, a, b in zip(worths, lower, upper, strict=True))
    spend, price, bound, worth, gaps = solve(lower, upper, bounded)
    best = (worth, spend, price)
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
            # a split of the budget fits, or the budget is free
            if charge is not None or float(child_low.sum()) <= target <= float(child_high.sum()):
                halves.append((floor, cap, child_low, child_high))
        if nodes + len(halves) > settings.max_boxes:
            break
        heapq.heappop(heap)
        for floor, cap, child_low, child_high in halves:
            child = list(bounded)
            child[cut_at] = _bounded(worths[cut_at], floor, cap)
            rates, at, top, value, excess = solve(child_low, child_high, tuple(child))
            nodes += 1
            if value > best[0]:
                best = (value, rates, at)
            if top - best[0] > tolerance:
                heapq.heappush(
                    heap, (-top, nodes, child_low, child_high, tuple(child), rates, excess)
                )
            else:
                settled = max(settled, top)
    worth, spend, price = best
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


def _fill(few: np.ndarray, many: np.ndarray, amount: float) -> np.ndarray:
    """``few`` raised toward ``many`` by ``amount`` in all, one rate after another.

    The rates either side of a price are both best there, and so is any mix of them. Where several
    jump at one price, on envelope chords of one slope, a mix that moved them together would leave
    each inside its chord, where the curve is below the envelope; filled in turn, at most one is.
    """
    steps = many - few
    before = np.concatenate([[0.0], np.cumsum(steps)[:-1]])
    return few + np.clip(amount - before, 0.0, steps)


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
    met where the searched gain less ``per_unit`` a unit falls through nothing past that peak, by
    Brent's method; refused where the peak falls short, but past the peak the gain need not fall
    all the way, and the budget is not proved the most.

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


def _budget_for(
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


def _sought(
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
                found = plan(_crossing(plan, lambda p: -surplus(p), best.budget, most))
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
    :func:`allocate` does.

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
    box = {"lower": lower_rates, "upper": upper_rates, "history": spent}
    with _Stopwatch() as clock:
        plans = [
            allocate(reading, budget, periods, **box, rtol=rtol, atol=atol, max_boxes=max_boxes)
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
    if ending.stopped == "cap":
        _log.warning(
            "minimax_allocate stopped at its cap with the gap open, after %d boxes: the split's "
            "worst regret is %.6g, and no split in the box has less than %.6g",
            ending.boxes,
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
) -> CvarAllocation:
    """Spend ``budget`` for the most mean gain over ``against`` in the worst ``level`` share of
    ``readings`` of the channels: the gain's conditional value at risk.

    Each reading is a whole set of channels, one a column, as :func:`minimax_allocate` takes them: a
    posterior's draws, say, weighed alike. A split's gain under a reading is its return less the
    reference's, both on the reading's curves, and the split returned has the most mean gain over
    the worst ``level`` share of the readings, to the tolerance ``rtol`` and ``atol`` set, or as
    near as 500 rounds of cutting planes come on concave curves and ``max_boxes`` boxes of the
    rates past them; :attr:`CvarAllocation.stopped` says which, and :attr:`CvarAllocation.limit`
    what stopped it. A program HiGHS leaves unsolved, one it stalls on stopped by an iteration
    limit, leaves its box the bound of the programs before it. The reference spends the budget in
    the box and gains nothing under any reading, so the split returned never does worse in that
    share than keeping it. At ``level = 1`` the split has the most mean gain, the posterior's
    expected return; as the level falls it moves only as far as the worst readings agree it gains.

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

    Raises:
        TypeError, ValueError: what :func:`allocate` refuses of any reading, the box, the history
            or the search's settings; no readings; readings of different numbers of channels; a
            level outside ``(0, 1]``; a reference of the wrong shape, outside the box or not
            spending the budget.
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
    with _Stopwatch() as clock:
        curves = _stacks([_worths(reading, spent, periods) for reading in readings])
        spend, gains, cvar, bound, ending = _cvar_search(
            curves,
            [reference],
            _read(curves, reference, len(readings))[0].sum(axis=1),
            math.inf,
            rate,
            level,
            lower_rates,
            upper_rates,
            settings,
        )
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


def _cvar_search(
    curves: tuple[_Stack, ...],
    starts: Sequence[np.ndarray],
    base: np.ndarray,
    top: float,
    rate: float,
    level: float,
    lower: np.ndarray,
    upper: np.ndarray,
    settings: _Settings,
) -> tuple[np.ndarray, np.ndarray, float, float, _Ending]:
    """:func:`cvar_allocate`'s split, its gains, their worst share's mean, a bound on every
    split's, and how the search ended.

    A gain is a reading's return on the curves less its ``base``: the reference's return for
    :func:`cvar_allocate`, each reading's best for :func:`minimax_allocate`. The first box tries
    ``starts`` first, and the first of them is the split returned where none gains more; ``top``
    bounds every split's worst share's mean before any plane does. The tolerance is set by
    ``settings`` on the largest of ``base``, and a box is cut only where the count of boxes, with
    its halves, stays within ``settings.max_boxes``."""
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
    # costs unperturbed, cycled at the level 1, where every excess's reduced cost is zero
    share = level * count
    weight = max(share, 1.0)
    lines = np.repeat(np.arange(count), size + 2)
    columns = np.concatenate(
        [np.tile(np.arange(size + 1), (count, 1)), size + 1 + np.arange(count)[:, None]], axis=1
    ).ravel()
    objective = np.concatenate([np.zeros(size), [-weight], np.full(count, weight / share)])
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
        value = _cvar(gain, level)
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
                value = _cvar(gain, level)
                if value > relaxed:
                    relaxed, at, excess = value, split, _cvar_weights(actual, level) @ over
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
                partial(_evened, reading=reading, count=count, cap=weight / share, total=weight),
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
    duals: np.ndarray, reading: np.ndarray, count: int, cap: float, total: float
) -> np.ndarray:
    """The duals of :func:`_cvar_search`'s planes rescaled reading by reading, ``reading`` each
    plane's, so that no reading's mass passes ``cap``, its excess's cost, and the masses come to
    ``total``, eta's, as far as the readings with a mass can take them.

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


def _cvar_weights(gain: np.ndarray, level: float) -> np.ndarray:
    """The weight :func:`_cvar` puts on each of ``gain``: ``1 / (level * n)`` on each of the worst
    whole share, the rest of the share on the next, none on the others."""
    order = np.argsort(gain, kind="stable")
    share = level * gain.size
    whole = int(np.floor(share))
    weights = np.zeros(gain.size)
    weights[order[:whole]] = 1.0 / share
    if whole < gain.size:
        weights[order[whole]] = (share - whole) / share
    return weights


def _cvar(gain: np.ndarray, level: float) -> float:
    """The mean of the worst ``level`` share of ``gain``, each weighed alike, the last in part."""
    ordered = np.sort(gain)
    share = level * ordered.size
    whole = int(np.floor(share))
    total = float(ordered[:whole].sum())
    if whole < ordered.size:
        total += (share - whole) * float(ordered[whole])
    return total / share


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
    curvature, response = [], []
    free = []
    for column, channel in enumerate(channels):
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
    rtol: float = 1e-9,
    atol: float = 0.0,
    max_boxes: int = 500,
) -> GeoAllocation:
    """Spend ``budget`` over ``periods`` on every geo's channels at once, for the most return.

    Each cell, one geo's channel, runs at one rate a period as :func:`allocate`'s channels do, its
    worth counted the same way, carryover in from its history and out after the plan. The budget is
    spent exactly; each geo's spend over the plan, its channels together, stays within
    ``geo_totals``, and each channel's, its geos together, within ``channel_totals``.

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
    warning (``chc_event="allocation_cap"``) with its gap.

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
        rtol, atol, max_boxes: as :func:`allocate` takes them; the scale ``rtol`` is a share of is
            the first box's bound.

    Raises:
        TypeError: a cell is not a :class:`chc.response.Channel`, or a total is not a
            :class:`Totals`.
        ValueError: what :func:`allocate` refuses of a cell, its box, its history or the search's
            settings; rows of different lengths; totals of the wrong length, or one outside what
            its cells' boxes spend; and a budget, boxes and totals no plan meets together.
    """
    settings = _Settings(rtol, atol, max_boxes)
    layout = _layout(cells, periods, lower, upper, geo_totals, channel_totals, history, settings)
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
      unit plus its totals' prices, and the budget is what the plan spends, an end of the budgets
      where the price stays on one side of ``per_unit`` across them.
    * :class:`ReturnOnSpendTarget`: the most budget whose gain is ``per_unit`` times the budget or
      more. The gain less ``per_unit`` a unit is concave in the budget and peaks where the price is
      ``per_unit``, so past that budget it falls, and the most budget is where it falls through
      nothing: Newton's method from the most budget.

    On an S-shaped curve each plan is :func:`allocate_geos`' search at its budget, and the best
    plan's gain need not be concave in the budget: a return target and a return on spend are met
    where that gain crosses them, by Brent's method between budgets either side, and neither is
    proved the least or the most. The marginal target's plan is searched with its budget free.

    Args:
        cells, periods, lower, upper, geo_totals, channel_totals, history: as
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
            cells, periods, lower, upper, geo_totals, channel_totals, history, settings
        )
        found = _meet(layout, periods, goal)
        compiling, searching = clock.read()
    found = replace(found, compile_seconds=compiling, search_seconds=searching)
    _capped("budget_for_geos", found)
    return found


def _meet(layout: _Layout, periods: int, goal: Goal) -> GeoAllocation:
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
            return plan(_crossing(plan, lambda p: -surplus(p), peak.budget, most))


def _layout(
    cells: Sequence[Sequence[Channel]],
    periods: int,
    lower: ArrayLike,
    upper: ArrayLike,
    geo_totals: Totals | None,
    channel_totals: Totals | None,
    history: ArrayLike | None,
    settings: _Settings,
) -> _Layout:
    """The grid laid out for planning, after what :func:`allocate_geos` refuses of it, its plans
    searched to ``settings``."""
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
    return _Layout(
        geos, width, flat, low, high, spent, geo_totals, channel_totals, tuple(groups), settings
    )


def _plan(layout: _Layout, periods: int, spend: float | MarginalReturnTarget) -> GeoAllocation:
    """The best plan of the grid that spends a budget, or that spends to where one more currency
    unit returns a marginal target's ``per_unit``: the budget left free, each unit of it charged
    that return. Searched in boxes where a curve is S-shaped (:func:`_geo_search`), to the
    tolerance and cap ``layout.settings`` sets.

    A budget is one more total, every cell's, fixed at the budget.
    """
    if isinstance(spend, MarginalReturnTarget):
        groups, charge = layout.groups, spend.per_unit
    else:
        everything = _Group(np.arange(layout.lower.size), spend, spend)
        groups, charge = (everything, *layout.groups), 0.0
    with _Stopwatch() as clock:
        worths = _worths(layout.cells, layout.history, periods)
        found, bound, ending = _geo_search(
            worths, layout.lower, layout.upper, groups, periods, charge, layout.settings
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
    worths: tuple[_Worth, ...],
    bounded: tuple[_Worth, ...],
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
    worths: tuple[_Worth, ...],
    lower: np.ndarray,
    upper: np.ndarray,
    groups: Sequence[_Group],
    periods: int,
    charge: float,
    settings: _Settings,
) -> tuple[_Planned, float, _Ending]:
    """The best plan of the cells under ``groups``' totals, each currency unit it spends charged
    ``charge``; a bound on the worth of every plan spending as much; and how the search ended.

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
    box's bound is within it of the best plan's.
    """
    eps = _rounding(worths)
    bounded = tuple(_bounded(w, a, b) for w, a, b in zip(worths, lower, upper, strict=True))
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
            child[cut_at] = _bounded(worths[cut_at], floor, cap)
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
    return best, top + best.charged, ending


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


def _outer(
    envelopes: Sequence[_Worth],
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
    envelopes: Sequence[_Worth],
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
            membership[row, groups[k].members] = 1.0
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


def _names(parameters: Channel) -> list[str]:
    """Each flattened leaf's place, ``"curve.scale"``, with an index where a leaf is not a
    scalar."""
    names = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(parameters)[0]:
        name = ".".join(str(getattr(key, "name", key)) for key in path)
        count = int(np.size(leaf))
        names += [name] if count == 1 else [f"{name}[{k}]" for k in range(count)]
    return names


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
