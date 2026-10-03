"""One call from a panel of logs to a certified intervention schedule.

Every layer this needs already exists --- :mod:`chc.graph` for which covariates to adjust for,
:mod:`chc.dynamics_id` for a control channel that is interventional rather than observational,
:mod:`chc.plan` for the constrained solve and its error tube, and :func:`chc.plan.certify_safety`
for what the plan survives under confounding. What did not exist was a way to run them in that
order without re-deriving the wiring, so the demo in every README was five imports and forty lines.
:func:`prescribe` is that wiring and nothing more: no new estimator, no new solver, no new
guarantee.

The module is ``chc.decision`` rather than ``chc.prescribe`` for one reason: the headline export is
a *function* named ``prescribe``, and a module of the same name inside the same package is shadowed
by it --- after ``import chc.prescribe``, ``chc.prescribe`` is the function and
``chc.prescribe.Lever`` raises ``AttributeError``. It was the only such collision in the library.

**The causal assumption is a required argument.** ``adjustment=`` takes either a
:class:`~chc.graph.CausalGraph`, from which the adjustment set is *derived* and can come back
``not_identified``, or an explicit sequence of column names, which *asserts* it. There is no
default, because the default would be "adjust for nothing", which is a causal claim this library
exists to stop people making by accident.

**Two questions, never conflated**, following :class:`chc.plan.CausalPlan`. Whether the effect is
identified is about the *data and the graph*; whether the plan is certified is about the *model
error and the barrier*. :class:`DecisionCertificate` reports both, and an unidentified effect
produces no schedule at all --- :attr:`Prescription.schedule` raises rather than hand back actions
computed from a channel nothing in the log pins down.
"""

from __future__ import annotations

import logging
import math
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, get_args

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import ArrayLike, NDArray

from chc.control import LinearConstraint, SolverStatus
from chc.cost import QuadraticCost, total_cost
from chc.dynamics import DrivenDynamics, Dynamics, HybridDynamics, LinearDynamics
from chc.dynamics_id import CausalDynamicsFit, Integrator, fit_causal_residual
from chc.evaluation import (
    AffinePolicy,
    AffineSchedule,
    LinearGaussianPlant,
    LoggerCheck,
    PlanEvaluation,
    _evaluate_by_unit,
    evaluate_plan,
)
from chc.graph import AdjustmentSet, CausalGraph
from chc.independence import gcm_test
from chc.integrate import rk4_step
from chc.lqr import linearize_continuous, linearize_discrete
from chc.mpc import PeriodBudget, _period_rows
from chc.panel import Panel, Provenance
from chc.plan import (
    BarrierConstraint,
    CausalPlan,
    CertificateStatus,
    RowPrice,
    SafetyCertificate,
    causal_plan,
    certify_safety,
    plan_regret_bound,
)

SCHEMA_VERSION = 1
"""``to_json``'s schema version. Bumped when a field changes meaning, not when one is added."""


class DecisionError(ValueError):
    """A decision could not be set up: a lever, a target, a constraint or an adjustment is wrong.

    Subclasses ``ValueError`` so it stays catchable the way it always was, and exists so a caller
    wrapping :func:`prescribe` can tell *its own* mis-specification apart from a ``ValueError``
    thrown out of jax, numpy or the solver. Every message names the offending column or bound.

    Columns that are simply absent still raise ``KeyError``, matching :class:`~chc.panel.Panel`
    lookup: asking for a name that is not there is a lookup failure, not a bad decision.
    """


class NotIdentifiedError(DecisionError):
    """The effect is not identified by adjustment, so there is no schedule to hand back.

    Its own type because it is the one failure this library exists to produce. A caller may want to
    fall back to :mod:`chc.sensitivity`'s partial-identification path on exactly this and on
    nothing else; catching it by message would be catching it by accident.
    """


_log = logging.getLogger(__name__)
"""Decision-point log for :func:`prescribe`, on the stdlib and nothing else.

Every record carries a ``chc_event`` key in its ``extra`` payload naming the point it was emitted
at --- ``precision``, ``adjustment``, ``logger_check``, ``fit``, ``abort``, ``selection`` (one per
step under ``max_levers``), ``driver_range``, ``plan``, ``certificate``, and ``one_unit`` from
:meth:`Prescription.evaluate` --- so a JSON formatter downstream can route on one field rather than
parse a sentence. The library installs no handler and sets no level: that is the application's
call, and a library that reaches for ``basicConfig`` takes it away.

The records that are not ``INFO`` are the ones worth waking someone for: identifying in single
precision, a graph that says the effect is not identified at all, a driver's forecast outside the
range the panel logged, levers that read a column besides the state and their recorded parents,
and an evaluation that reads one unit's windows as independent.
"""

IdentificationStatus = Literal["identified", "asserted", "not_identified"]
"""How the set was arrived at. ``asserted`` means a caller named it and nothing here checked it."""

TimeZero = Literal["calendar", "unit"]
"""Where :meth:`Prescription.evaluate`'s windows start: on one calendar for every unit, or cut back
from each unit's own latest period."""


@dataclass(frozen=True)
class Lever:
    """One actionable column and the box it may move in.

    ``unit_cost`` is the quadratic price of using it, and defaults to free: with a box constraint a
    free lever is still bounded, so the default is safe rather than merely convenient.

    ``cap_per_step`` bounds how far the lever may move between consecutive planned steps. ``None``
    lets it jump anywhere in its box from one step to the next, which is a schedule a media plan, a
    rota or a price list often cannot execute. The move from the level already in force to the
    first planned step is not capped, since the planner is not told what that level is.
    """

    name: str
    lo: float
    hi: float
    unit_cost: float = 0.0
    cap_per_step: float | None = None

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise DecisionError(f"lever {self.name!r} has lo={self.lo} above hi={self.hi}")
        if self.cap_per_step is not None and not self.cap_per_step >= 0.0:
            raise DecisionError(
                f"lever {self.name!r} has cap_per_step={self.cap_per_step}; a rate limit is a "
                "non-negative distance"
            )


@dataclass(frozen=True)
class Target:
    """The column to steer and where to steer it.

    ``value`` is one level for the whole plan, or a schedule with one level per step:
    ``value[k]`` is the level for the state after ``k + 1`` actions, so a schedule has ``horizon``
    entries. A schedule is how a set point that moves inside the horizon reaches the plan --- a
    comfort band that tightens at eight is steered for before eight, not re-planned for after it.
    """

    name: str
    value: ArrayLike
    weight: float = 1.0


@dataclass(frozen=True)
class Constraint:
    """A state that must stay inside a range --- what the safety certificate is priced against.

    ``state`` may be the target column: the steered state is then bounded as well as steered, which
    is how a comfort band around a set point is stated. A target value outside its own bound is
    allowed and means "as close as the bound permits"; holding the bound is what keeps the plan
    from crossing it on the way.
    """

    state: str
    lo: float | None = None
    hi: float | None = None

    def __post_init__(self) -> None:
        if self.lo is None and self.hi is None:
            raise DecisionError(f"constraint on {self.state!r} bounds nothing")
        if self.lo is not None and self.hi is not None and self.lo > self.hi:
            raise DecisionError(f"constraint on {self.state!r} has lo={self.lo} above hi={self.hi}")


@dataclass(frozen=True)
class Driver:
    """An exogenous column that pushes the state, and its forecast over the plan.

    ``forecast[k]`` is the level at ``t = k * dt``, so a forecast has ``horizon + 1`` entries: one
    at the start of every step and one at the end of the last. Between two entries the driver moves
    linearly, which is how the fit reads it off the log (:class:`chc.dynamics.DrivenDynamics`).
    Nothing the plan does moves a driver --- weather, a published demand forecast; a column the
    levers can move is a state, not a driver.
    """

    name: str
    forecast: ArrayLike


@dataclass(frozen=True)
class InterventionSchedule:
    """What to set each lever to, step by step."""

    levers: tuple[str, ...]
    magnitudes: Array  # (horizon, m)

    def windows(self, *, tol: float = 1e-6) -> dict[str, tuple[int, int] | None]:
        """First and last step at which each lever is active, or ``None`` if it never is.

        Derived rather than stored: a start/stop pair that can disagree with the magnitudes it
        summarises is a worse footgun than computing it twice.
        """
        magnitudes = np.asarray(self.magnitudes)
        out: dict[str, tuple[int, int] | None] = {}
        for index, name in enumerate(self.levers):
            active = np.flatnonzero(np.abs(magnitudes[:, index]) > tol)
            out[name] = (int(active[0]), int(active[-1])) if active.size else None
        return out


@dataclass(frozen=True)
class DecisionCertificate:
    """What is known about the decision, on the two axes that must not be merged.

    Identification is about the data and the graph; certification is about model error and the
    barrier. A plan can be fully certified against a channel nothing identifies --- that is a
    trustworthy tube around a meaningless action --- so both are reported and neither is derived
    from the other.
    """

    identification: IdentificationStatus
    # The adjustment set and how it was justified. Note this is a DIFFERENT question from
    # ``Prescription.model_fit.identified``, which is the estimator's own mechanical flag: "was I
    # handed covariates or an instrument?". With an unconfounded lever a graph can prove the effect
    # identified while that flag is False, because there was correctly nothing to adjust for.
    adjustment: AdjustmentSet
    identification_radius: float | None  # channel standard error; None when not identified
    overlap: float  # residualised action variance: 0 means the log has no variation to learn from
    certificate_status: CertificateStatus
    certified_horizon: int | None  # steps the Gronwall tube keeps inside ``tolerance``
    barrier_certified_steps: int | None  # leading prefix clearing the barrier; None if no bound
    # The weakest step's confounding ceiling; None if unconstrained. At a step where two bounds tie
    # (the midpoint of a two-sided bound) it is the weaker tied margin's ceiling, an upper bound on
    # the joint one: each margin's is reached by its own best action, and no one action need reach
    # both.
    gamma_star: float | None
    # None when the effect is not identified: no solve was attempted, which is a third answer and
    # not one of the solver's three. Same convention as the certified-horizon fields above.
    solver_status: SolverStatus | None
    solver_iterations: int
    # How far the plan can be from the best one the SAME BOX allows (Result 69), certified from its
    # own gradient with no optimum needed. ``inf`` means the objective was not convex over the box,
    # so nothing certifies it; ``None`` means no plan was solved at all. This is a gap in the
    # PLANNING objective -- how far the planning model is from the plant is the tube's question,
    # and the two must not be added.
    regret_bound: float | None
    # The marginal sensitivity model's level the barrier's prefix was audited at, the caller's
    # ``gamma``: ``barrier_certified_steps`` counts the steps the plan clears there, and
    # ``gamma_star`` is where no action would. None where no bound was audited.
    gamma: float | None = None

    @property
    def trustworthy_steps(self) -> int:
        """The prefix that survives *both* axes --- the number an operator can act on.

        Zero whenever the effect is not identified, whatever the tube says, and never longer than
        the shorter of the two certified prefixes. The two ``None`` values mean different things.
        ``certified_horizon is None`` means the tube was not evaluated, so nothing bounds the
        trajectory's error and it contributes zero rather than infinity --- a barrier cleared by an
        unbounded trajectory proves nothing. ``barrier_certified_steps is None`` means no state was
        bounded, so there is no safety prefix to respect and the tube alone decides.
        """
        if self.identification == "not_identified" or self.certified_horizon is None:
            return 0
        if self.barrier_certified_steps is None:
            return self.certified_horizon
        return min(self.certified_horizon, self.barrier_certified_steps)


@dataclass(frozen=True)
class SelectionStep:
    """One step of the greedy lever selection: the lever it added, and what the plan cost with it.

    ``regret_bound`` is :func:`chc.plan.plan_regret_bound` on that step's plan, priced against
    every lever's box, the levers not yet selected included. So it bounds how far below
    ``task_cost`` any plan the boxes allow can go: what the levers left out at this step could
    still buy, plus what the solve left on the table. ``inf`` when the objective was not convex
    over the boxes, as in :attr:`DecisionCertificate.regret_bound`.
    """

    lever: str
    task_cost: float  # the planned cost with the levers selected so far, this one included
    regret_bound: float


@dataclass(frozen=True)
class LeverSelection:
    """Which levers :func:`prescribe` kept under ``max_levers``, in the order it added them.

    What a step bought is the drop in planned cost from the step before it; ``idle_cost`` is the
    planned cost with every lever held at zero, which is what the first step is measured against.
    """

    idle_cost: float
    steps: tuple[SelectionStep, ...]

    @property
    def selected(self) -> tuple[str, ...]:
        return tuple(step.lever for step in self.steps)


@dataclass(frozen=True)
class _Columns:
    """What evaluating a plan needs to know of the panel it was made from."""

    states: tuple[str, ...]  # in the model's order: the target, then every other constrained one
    # What the levers were logged on besides one another: their parents in the graph, or the
    # covariates an asserted adjustment names.
    logged_on: tuple[str, ...]
    # Columns the record says were fixed before the levers moved and not logged on: the graph's
    # observed columns that no lever causes, and the drivers. What the logger check tests besides
    # the past.
    unlogged: tuple[str, ...] = ()
    # The levers' parents the logger check cannot condition on: latent in the graph, or not a
    # numeric column of the panel. The check needs every parent, so it is not run without them.
    unreadable: tuple[str, ...] = ()

    @property
    def given(self) -> tuple[str, ...]:
        """What the levers may have read: the states, then their parents outside the states."""
        return tuple(dict.fromkeys((*self.states, *self.logged_on)))


@dataclass(frozen=True)
class Prescription:
    """The decision, the evidence for it, and what it took to get there."""

    levers: tuple[Lever, ...]
    target: str
    plan: CausalPlan | None  # None iff the effect is not identified, so no plan was made
    certificate: DecisionCertificate
    model_fit: CausalDynamicsFit
    provenance: Provenance
    # None unless ``max_levers`` was given and a plan was made. The schedule still has a column per
    # lever: an unselected one is zero throughout.
    selection: LeverSelection | None = None
    drivers: tuple[Driver, ...] = ()  # the forecasts the plan was made against
    # Whether the panel's levers read anything but the states and their parents in the record. It
    # is reported, not acted on; None when a lever's parent is not logged, or the panel has too few
    # rows for it. *Experimental.*
    logger_check: LoggerCheck | None = None
    budgets: tuple[PeriodBudget, ...] = ()  # what the plan was held to spend
    _columns: _Columns | None = field(default=None, repr=False, compare=False)
    # How many of the plan's constraint rows each budget holds: the last ones, in order.
    _budget_rows: tuple[int, ...] = field(default=(), repr=False, compare=False)

    @property
    def lever_names(self) -> tuple[str, ...]:
        return tuple(lever.name for lever in self.levers)

    @property
    def schedule(self) -> InterventionSchedule:
        """The actions to take.

        Raises:
            ValueError: if the effect is not identified. Returning a schedule here is the exact
                failure this library exists to prevent: the numbers would look like every other
                schedule and mean nothing. Read
                :attr:`DecisionCertificate.adjustment` for why, and
                :mod:`chc.sensitivity` for what to do about it.
        """
        if self.plan is None:
            raise NotIdentifiedError(
                "the effect is not identified, so no schedule was computed: "
                f"{self.certificate.adjustment.reason}"
            )
        return InterventionSchedule(levers=self.lever_names, magnitudes=self.plan.actions)

    def budget_prices(self, tolerance: float | None = None) -> tuple[tuple[RowPrice, ...], ...]:
        """What each budget is worth to the plan: one price per period it holds in the plan.

        A period's price is how much the planned cost falls per unit more it may spend, in the
        cost's units per unit of spend; zero where the period has room. Read as
        :meth:`chc.plan.CausalPlan.shadow_prices` reads every row, whose ``tolerance`` this is.

        Raises:
            NotIdentifiedError: if the effect is not identified, so no plan was made.
            ValueError: if the plan was held under the constraints' barrier
                (``hold_constraints``), whose rows' prices are not built.
        """
        if self.plan is None:
            raise NotIdentifiedError(
                "the effect is not identified, so no plan was made and no budget was priced: "
                f"{self.certificate.adjustment.reason}"
            )
        rows = self.plan.shadow_prices(tolerance).rows
        first = len(rows) - sum(self._budget_rows)
        prices = []
        for count in self._budget_rows:
            prices.append(rows[first : first + count])
            first += count
        return tuple(prices)

    def evaluate(
        self,
        panel: Panel,
        *,
        logger: AffinePolicy | None = None,
        smoothing: float | None = None,
        model_error: float = 1.0,
        min_effective: float = 100.0,
        time_zero: TimeZero = "calendar",
        n_resamples: int = 200,
        seed: int = 0,
    ) -> PlanEvaluation:
        """The schedule's expected cost over its horizon, estimated from ``panel`` before it is
        deployed, beside the cost of the policy that logged the panel:
        :func:`chc.evaluation.evaluate_plan` by ``"pdis"``, whose keywords the first four are, with
        its intervals read from a bootstrap over the units.

        What it is handed: the episodes are windows of ``H + 1`` consecutive periods of a unit,
        over the plan's states and the levers; the schedule is the plan's actions, open loop; the
        cost is the plan's own, less its terminal term; the plant is the plan's model over one
        step, linearised at the episodes' mean state and action, with the covariance of its
        one-step residuals on them as the noise. ``panel`` may be the one the plan was fitted on,
        and the plan was chosen on it, so a value read off it can be optimistic; a later one is
        not.

        A window starts at its time zero, where a target trial starts each unit's follow-up: when
        the unit is eligible and is assigned a strategy (Hernán and Robins, *Causal Inference:
        What If*, section 22.4). Under ``"calendar"``, the default, the time zeros are the panel's
        periods ``H``, ``2H``, ... before its last, the same for every unit, and a unit gives
        every window it was observed throughout. Under ``"unit"`` each unit's windows are cut back
        from its own latest period, a gap ending a run: where a unit's record ends then sets where
        each of its windows starts, so when a unit leaves the panel because of what happened to
        it, its windows are chosen by their own outcomes, and the value read off them is biased.

        One unit's windows follow each other, dependent through its state, so the intervals come
        from ``n_resamples`` draws of the units with replacement, from ``seed``, every window of a
        drawn unit kept. Each draw runs the evaluation again: the plant is linearised again, the
        logger fitted again unless it is given, the smoothing chosen again unless it is given, and
        the certificate and the values computed again. An interval is its estimate plus or minus
        1.96 standard deviations of its draws, widened by the model's correction as
        :func:`~chc.evaluation.evaluate_plan` widens its own; a draw the certificate refuses is
        left out and counted (:attr:`PlanEvaluation.bootstrap`). With few units the draws are
        few distinct panels and the intervals too narrow. Windows of one unit leave nothing to
        resample, so they are read as before, as independent, under an interval too narrow for
        them, with ``bootstrap`` and ``versus_logger`` ``None``: deprecated, the call warns with a
        :class:`DeprecationWarning`, and from 1.0 it raises :class:`DecisionError`.

        :attr:`PlanEvaluation.versus_logger` holds the plan's value less the logged policy's, on
        the same windows, its interval read off the difference within each draw. The policy in
        place can cost less than the plan; the difference is what switching to the plan would
        change.

        Scope: what :mod:`chc.evaluation` states, and two things more. The logs' actions must
        depend on the state and a randomisation of their own alone, so a plan whose levers were
        logged on a column outside its state is refused: the weights need the logger's propensity
        given that column, and no policy of the state is that. The graph says what the levers were
        logged on, their parents; an asserted adjustment is taken to name it. A covariate adjusted
        for only because it moves the target is no reason to refuse. And a window is read only
        when its unit was observed to its end. A unit that leaves the panel because of what
        happened to it after a window's time zero takes that window with it, and the windows left
        are not those an unselected panel would hold; weighting them by the chance of staying,
        which would correct that, is not built. Leaving on what happened before a time zero
        moves only the law the windows start from, which the value is read at.

        What the graph cannot say, the panel is asked: whether the levers read a column besides the
        state, or the state's past (:attr:`PlanEvaluation.logger_check`, as
        :attr:`Prescription.logger_check` on the panel the plan was fitted on). A rejection is
        logged as a warning and changes nothing else. *Experimental.*

        Raises:
            NotIdentifiedError: if the effect is not identified, so there is no plan.
            DecisionError: on a ``time_zero`` other than ``"calendar"`` and ``"unit"``; fewer than
                two resamples; a plan made against driver forecasts, whose plant changes with the
                step; a plan whose levers were logged on a column outside its state, or whose
                record does not say; or a panel with fewer than two windows.
            InfeasibleEvaluation: when the evaluation's certificate refuses, on the panel or on
                every draw but one.
        """
        if time_zero not in get_args(TimeZero):
            raise DecisionError(f"time_zero must be one of {get_args(TimeZero)}, got {time_zero!r}")
        if n_resamples < 2:
            raise DecisionError(
                f"n_resamples={n_resamples}: an interval read from draws of the units needs at "
                "least two of them"
            )
        plan = self.plan
        if plan is None:
            raise NotIdentifiedError(
                "the effect is not identified, so there is no plan to evaluate: "
                f"{self.certificate.adjustment.reason}"
            )
        if self.drivers:
            raise DecisionError(
                "the plan was made against driver forecasts, so its plant changes with the step, "
                "and the evaluation's plant is one step for every step"
            )
        columns = self._columns
        if columns is None:
            raise DecisionError(
                "this prescription does not record its states or what its levers were logged on, "
                "and importance weights are right only if that is the plan's state"
            )
        outside = [name for name in columns.logged_on if name not in columns.states]
        if outside:
            raise DecisionError(
                f"the levers were logged on {outside}, outside the plan's state "
                f"{list(columns.states)}: importance weights need the logger's propensity given "
                "them, which no policy of the state is"
            )
        problem = plan._problem
        if problem is None:
            raise DecisionError("the plan carries no problem, so there is no model to evaluate on")
        actions = np.asarray(plan.actions, dtype=np.float64)
        episodes = _episodes(
            panel,
            states=columns.states,
            levers=self.lever_names,
            horizon=len(actions),
            time_zero=time_zero,
        )
        logger_check = _check_logger(panel, levers=self.lever_names, columns=columns)
        model, dt = problem.model, problem.dt
        schedule = AffineSchedule.open_loop(actions, len(columns.states))
        if np.unique(episodes.units).size < 2:
            # A break on the stable tier waits for 1.0 behind a DeprecationWarning: until then one
            # unit's windows are read as they were before the bootstrap over units.
            message = (
                "the windows are one unit's, so there are no units to resample: the interval "
                "treats the windows as independent and is too narrow. From 1.0 a panel of one "
                "unit raises DecisionError; evaluate on a panel of several units instead"
            )
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            _log.warning(
                message, extra={"chc_event": "one_unit", "windows": int(episodes.x.shape[0])}
            )
            evaluation = evaluate_plan(
                {"x": episodes.x, "u": episodes.u},
                schedule,
                "pdis",
                plant=_linearised(model, episodes.x, episodes.u, dt),
                cost=problem.cost,
                logger=logger,
                smoothing=smoothing,
                model_error=model_error,
                min_effective=min_effective,
            )
        else:
            evaluation = _evaluate_by_unit(
                episodes.x,
                episodes.u,
                episodes.units,
                schedule,
                plant=lambda x, u: _linearised(model, x, u, dt),
                cost=problem.cost,
                logger=logger,
                smoothing=smoothing,
                model_error=model_error,
                min_effective=min_effective,
                resamples=n_resamples,
                seed=seed,
            )
        return replace(evaluation, logger_check=logger_check)

    def reach(self) -> dict[str, float]:
        """Per lever, how far it can move the target's rate across its own box.

        ``channel[target, lever, 0] * (hi - lo)``: the constant term of the fitted control channel
        on the target's row, times the width of the lever's box. A large coefficient on a lever
        that may barely move is not a large lever, which is why the range is in the number and not
        only in the footnote.

        Read from the *same* fit that produced the plan, deliberately. Ranking by a second
        estimator --- local projections, say --- invites an ordering that contradicts the schedule
        printed beside it, and two disagreeing orderings on one page is worse than one.
        """
        channel = np.asarray(self.model_fit.residual.channel)  # (n_states, n_levers, n_features)
        constant = channel[0, :, 0]
        return {
            lever.name: float(constant[index] * (lever.hi - lever.lo))
            for index, lever in enumerate(self.levers)
        }

    def explain(self) -> str:
        """The levers ranked by :meth:`reach`, widest first, as text."""
        ranked = sorted(self.reach().items(), key=lambda item: -abs(item[1]))
        lines = [f"levers ranked by reach on {self.target!r} (channel x box width):"]
        lines += [f"  {name:<20} {value:+.4g}" for name, value in ranked]
        return "\n".join(lines)

    def report(self) -> str:
        """A Markdown summary: the decision, both certificate axes, and the provenance.

        Text only, from string templates. No plotting dependency, so the output is deterministic
        and can be asserted on in a test rather than eyeballed.
        """
        certificate = self.certificate
        lines = [
            f"# Prescription for `{self.target}`",
            "",
            "## Decision",
        ]
        if self.plan is None:
            lines += ["", "**No schedule.** " + certificate.adjustment.reason, ""]
        else:
            windows = self.schedule.windows()
            lines += ["", "| lever | active steps | first | last |", "|---|---|---|---|"]
            magnitudes = np.asarray(self.plan.actions)
            for index, name in enumerate(self.lever_names):
                window = windows[name]
                span = "never" if window is None else f"{window[0]}-{window[1]}"
                first, last = magnitudes[0, index], magnitudes[-1, index]
                lines.append(f"| `{name}` | {span} | {first:+.4g} | {last:+.4g} |")
            lines += ["", f"Planned task cost: {self.plan.task_cost:.6g}.", ""]
            # a plan held under the constraints' barrier carries no row prices
            priced = self.budget_prices() if self.budgets and self.plan.safety is None else None
            for index, budget in enumerate(self.budgets):
                worth = (
                    "not priced, as the plan was held under the constraints' barrier"
                    if priced is None
                    else "a unit more in each period would lower the planned cost by "
                    + ", ".join(_show_price(price) for price in priced[index])
                )
                lines += [f"Budget of {budget.amount:.6g} per {budget.period} steps: {worth}.", ""]
            if self.selection is not None:
                lines += [
                    "Levers kept under `max_levers`, in the order greedy added them; with none, "
                    f"the planned cost is {self.selection.idle_cost:.6g}.",
                    "",
                    "| step | lever | planned cost | regret bound |",
                    "|---|---|---|---|",
                ]
                for index, step in enumerate(self.selection.steps, start=1):
                    bound = _show(step.regret_bound)
                    lines.append(f"| {index} | `{step.lever}` | {step.task_cost:.6g} | {bound} |")
                lines.append("")
        lines += [
            "## Certificate",
            "",
            f"- identification: **{certificate.identification}** ({certificate.adjustment.reason})",
            f"- adjusted for: {list(certificate.adjustment.covariates) or 'nothing'}",
            *self._driver_lines(),
            f"- channel standard error: {_show(certificate.identification_radius)}",
            f"- overlap (residualised action variance): {certificate.overlap:.4g}",
            self._logger_line(),
            f"- error tube: **{certificate.certificate_status}**, "
            f"certified horizon {_show(certificate.certified_horizon)}",
            f"- barrier: certified steps {_show(certificate.barrier_certified_steps)}"
            + ("" if certificate.gamma is None else f" at gamma {certificate.gamma:.4g}")
            + f", gamma* {_show(certificate.gamma_star)} (marginal sensitivity model)",
            f"- solver: {certificate.solver_status} after "
            f"{certificate.solver_iterations} accepted steps",
            "",
            f"**Trustworthy prefix: {certificate.trustworthy_steps} steps.**",
            "",
            "## Provenance",
            "",
            f"- data sha256: `{self.provenance.data_sha256[:16]}...`",
            f"- chc {self.provenance.chc_version}, {self.provenance.n_rows} rows, "
            f"x64={self.provenance.x64}, seed={self.provenance.seed}",
        ]
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """The schedule, both certificate axes and the provenance, as plain JSON-safe values."""
        certificate = self.certificate
        return {
            "schema_version": SCHEMA_VERSION,
            "target": self.target,
            "levers": list(self.lever_names),
            "schedule": None if self.plan is None else np.asarray(self.plan.actions).tolist(),
            "certificate": {
                "identification": certificate.identification,
                "adjusted_for": list(certificate.adjustment.covariates),
                "reason": certificate.adjustment.reason,
                "identification_radius": certificate.identification_radius,
                "overlap": certificate.overlap,
                "certificate_status": certificate.certificate_status,
                "certified_horizon": certificate.certified_horizon,
                "barrier_certified_steps": certificate.barrier_certified_steps,
                "gamma": certificate.gamma,
                "gamma_star": certificate.gamma_star,
                "solver_status": certificate.solver_status,
                "solver_iterations": certificate.solver_iterations,
                "trustworthy_steps": certificate.trustworthy_steps,
                "regret_bound": certificate.regret_bound,
            },
            "selection": None
            if self.selection is None
            else {
                "idle_cost": self.selection.idle_cost,
                "steps": [
                    {
                        "lever": step.lever,
                        "task_cost": step.task_cost,
                        "regret_bound": step.regret_bound,
                    }
                    for step in self.selection.steps
                ],
            },
            "drivers": [
                {"name": driver.name, "forecast": np.asarray(driver.forecast, dtype=float).tolist()}
                for driver in self.drivers
            ],
            "budgets": [
                {
                    "weights": budget.weights.tolist(),
                    "amount": budget.amount,
                    "period": budget.period,
                    "start": budget.start,
                }
                for budget in self.budgets
            ],
            "provenance": self.provenance.to_json(),
        }

    def _logger_line(self) -> str:
        check = self.logger_check
        if check is None:
            unreadable = () if self._columns is None else self._columns.unreadable
            if unreadable:
                names = list(unreadable)
                return f"- logger check: not run, the levers' parents {names} are not logged"
            return "- logger check: not run, too few rows"
        test = check.test
        if math.isnan(test.p_value):
            return "- logger check: nothing to test, the state determines every column"
        head = f"p = {test.p_value:.3g} over {test.clusters} periods"
        if test.p_value <= _LOGGER_CHECK_ALPHA:
            correlation = test.partial_correlation
            lever, column = np.unravel_index(np.nanargmax(np.abs(correlation)), correlation.shape)
            return (
                f"- logger check: **the levers read more than the record says** ({head}); "
                f"strongest: `{check.levers[lever]}` on `{check.columns[column]}`, partial "
                f"correlation {correlation[lever, column]:+.2f}"
            )
        seen = test.detectable[np.isfinite(test.detectable)]
        missed = f"a partial correlation up to {seen.max():.2g}" if seen.size else "any dependence"
        return f"- logger check: passed ({head}); a pass can miss {missed}"

    def _driver_lines(self) -> list[str]:
        gain = self.model_fit.driver_gain
        if not self.drivers or gain is None:
            return []
        pushes = ", ".join(
            f"`{driver.name}` {float(np.asarray(gain)[0, index]):+.4g}"
            for index, driver in enumerate(self.drivers)
        )
        return [f"- drivers in the drift, with their push on `{self.target}`'s rate: {pushes}"]


def prescribe(
    panel: Panel,
    *,
    levers: Sequence[Lever],
    target: Target,
    horizon: int,
    adjustment: CausalGraph | Sequence[str],
    constraints: Sequence[Constraint] = (),
    hold_constraints: bool = False,
    budgets: Sequence[PeriodBudget] = (),
    max_levers: int | None = None,
    known: Dynamics | None = None,
    dt: float = 1.0,
    gamma: float = 1.0,
    tolerance: float | None = None,
    x0: Array | None = None,
    folds: int = 2,
    seed: int = 0,
    integrator: Integrator = "rk4",
    drivers: Sequence[Driver] = (),
) -> Prescription:
    """Fit the causal control channel from ``panel`` and plan a certified schedule on it.

    Args:
        panel: long-format logs. Consecutive periods within a unit become the ``(x, u, x_next)``
            transitions the channel is fitted on; gaps are dropped rather than interpolated, so an
            unbalanced panel is fine and a silently invented row is not.
        levers, target, constraints: the decision, in the domain's own names. States are the target
            column followed by each other constrained column, in that order. A constraint may name
            the target column itself: it then bounds the steered state, which gets the barrier and
            ``gamma_star`` like any other constrained column, and adds no state. A lever's
            ``cap_per_step`` is held by every iterate of the solve; the regret bound stays priced
            against the box alone, so with a rate limit it is conservative rather than tight.
        hold_constraints: hold ``constraints`` inside the solve, as the barrier condition
            :func:`chc.plan.certify_safety` audits at ``gamma``
            (:class:`chc.plan.BarrierConstraint`), rather than only price the plan against them.
            Off by default, so no existing schedule moves; the certificate reads the same audit
            either way, and it is still the verdict -- a solve stopped by its budget can come back
            short of the condition, and where two bounds tie the solve holds only the first. The
            regret bound then includes what holding cost, since it is still priced against the
            box alone.
        budgets: what the plan may spend (:class:`chc.mpc.PeriodBudget`): at most ``amount`` in
            each ``period`` steps from the plan's first, a step spending ``weights @ u``, with each
            lever's spend per unit in the levers' order. A period the horizon cuts short gets its
            share of ``amount``, so a budget for the whole horizon has ``period=horizon``; the rows
            are :class:`chc.mpc.RecedingHorizon`'s at its first step, with nothing spent, and every
            iterate of the solve holds them. :meth:`Prescription.budget_prices` says what each
            period's budget is worth to the plan; the regret bound stays priced against the box
            alone, as with a rate limit.
        max_levers: plan with at most this many levers, chosen by greedy forward selection with
            :func:`chc.plan.causal_plan` as its inner loop. Starting from no lever, each step plans
            once per lever not yet chosen, with that lever added, and keeps the cheapest plan ---
            under ``hold_constraints``, the one the audit clears over the longest prefix first.
            Greedy is not exhaustive and can miss the best set;
            ``docs/adr/0004-greedy-lever-selection.md`` shows where. **An unselected lever is held
            at zero** at every step: the level at which the fitted control-affine channel credits
            it with no effect and its ``unit_cost`` charges nothing, and the level
            :meth:`InterventionSchedule.windows` reads as inactive. So every lever's box must
            contain zero. The steps, each with its planned cost and regret bound, are
            :attr:`Prescription.selection`; the certificate's regret bound stays priced against
            every lever's box, so it includes what leaving levers out cost. ``None`` plans with
            every lever, as before; a value at or above the number of levers selects them all and
            returns that same plan.
        adjustment: a :class:`~chc.graph.CausalGraph` to *derive* the adjustment set from, or a
            sequence of column names to *assert* it. Required, and deliberately so --- omitting it
            would default to adjusting for nothing, which is a causal claim, not an absence of one.
        known: physics kept fixed and not estimated. ``None`` means nothing is known and the whole
            vector field is the fitted residual.
        dt: the time step one panel period represents.
        gamma: the marginal sensitivity model's level the barrier is priced at (§40), passed to
            :func:`chc.plan.certify_safety`. It prices the plan, and changes it only under
            ``hold_constraints``.
        tolerance: the trajectory error above which the plan stops being certified. **Omitting it
            switches the tube off** rather than setting it to infinity: this library cannot know
            how much error a caller accepts, and a certificate with an infinite tolerance passes
            over the whole horizon while proving nothing.
        x0: the state to plan from. Defaults to the mean over units of each unit's last observed
            state, which is the pooled "where we are now" and is recorded as such.
        integrator: the one-step map the fit is made consistent with, passed to
            :func:`~chc.dynamics_id.fit_causal_residual`. Defaults to ``"rk4"`` here and to
            ``"euler"`` there, deliberately: the low-level fit has no idea what will consume it and
            keeps its contract, while this function *knows* the field goes straight to
            :func:`chc.plan.causal_plan`, which rolls out with RK4. Fitting one integrator and
            planning with another is a bias the caller never asked for --- on the plant in
            :mod:`chc.mmm` it costs 28% of the control channel. Choose ``"euler"`` when the panel is
            genuinely discrete-time (a weekly budget is not a sample of an ODE) and the one-step map
            *is* the model.
        drivers: exogenous columns the plan can forecast but not move, each with its forecast over
            the horizon (:class:`Driver`). They are fitted into the drift jointly with the states
            (:func:`~chc.dynamics_id.fit_causal_residual`), and the plan is made against the
            forecast, so it acts ahead of a push it can see coming. The forecast moves the drift
            and never the channel, so ``gamma_star`` is priced with it in place. Two things are not
            priced: the forecast's own error, and the fitted gain's standard error, which the tube's
            model error leaves out as it leaves out the drift's. A forecast outside the range the
            panel logged is extrapolated by the fitted gain, and logged as a warning.

    Returns:
        A :class:`Prescription`. Read :attr:`DecisionCertificate.identification` before
        :attr:`Prescription.schedule`, which raises when the effect is not identified.

    Raises:
        DecisionError: the decision is mis-specified --- no lever, a column constrained twice,
            a target schedule whose length is not ``horizon``, constraints to hold with none
            given, ``max_levers`` below one or with a lever whose box excludes zero, a driver that
            is also a lever or a state or is named twice, a forecast that is not ``horizon + 1``
            finite levels, a budget that does not weigh one spend per lever, starts its periods
            anywhere but the plan's first step, or allows less than the levers' boxes spend at
            the least, or a panel with no consecutive pair of periods to fit a transition on.
        KeyError: a lever, target, constraint, driver or asserted covariate names a column the
            panel does not have. The message lists the panel's columns.

    Each decision point emits one ``logging`` record on ``chc.decision``, keyed by ``chc_event``
    (see :data:`_log`). Nothing is configured here; a caller that wants them calls
    ``logging.basicConfig`` itself. An unidentified effect, a single-precision panel, a forecast
    outside the logged range and levers that read more than the record says come through at
    ``WARNING``.

    Before fitting, the panel is asked whether the levers read anything but the states and their
    recorded parents: :attr:`Prescription.logger_check` (*experimental*). It reports; it changes
    nothing else.
    """
    if not levers:
        raise DecisionError(
            "prescribe needs at least one lever; there is nothing to decide otherwise"
        )
    if hold_constraints and not constraints:
        raise DecisionError("hold_constraints was set, but no constraint was given to hold")
    if np.shape(target.value) not in ((), (horizon,)):
        raise DecisionError(
            f"target {target.name!r} has a schedule of shape {np.shape(target.value)}; a schedule "
            f"has one level per step, {horizon} for this horizon"
        )
    if max_levers is not None:
        if max_levers < 1:
            raise DecisionError(
                f"max_levers={max_levers} selects no lever; leave it unset to plan with every lever"
            )
        for lever in levers:
            if not lever.lo <= 0.0 <= lever.hi:
                raise DecisionError(
                    f"lever {lever.name!r} has box [{lever.lo}, {lever.hi}], which excludes 0, the "
                    "level an unselected lever is held at; under max_levers every lever must be "
                    "able to stay off, so express it as a move from its current level"
                )
    for budget in budgets:
        if budget.weights.shape[0] != len(levers):
            raise DecisionError(
                f"a budget weighs {budget.weights.shape[0]} levers and the decision has "
                f"{len(levers)}; its weights are each lever's spend per unit, in the levers' order"
            )
        if budget.start != 0.0:
            raise DecisionError(
                f"a budget's periods start at t = {budget.start:g}, and the plan starts at 0 with "
                "nothing spent; a period already under way has spent what the plan cannot see"
            )
        least = sum(
            min(weight * lever.lo, weight * lever.hi)
            for weight, lever in zip(budget.weights, levers, strict=True)
            if weight != 0.0  # a free lever's unbounded side must not read as 0 * inf
        )
        if budget.amount < budget.period * least:
            raise DecisionError(
                f"a budget of {budget.amount:g} per {budget.period} steps allows less than the "
                f"{budget.period * least:g} the levers' boxes spend in that many at the least"
            )
    constrained = tuple(constraint.state for constraint in constraints)
    twice = sorted({name for name in constrained if constrained.count(name) > 1})
    if twice:
        raise DecisionError(
            f"columns constrained more than once: {twice}; "
            "give each one Constraint with both bounds"
        )
    # A bound on the target column bounds the target's own coordinate. A second coordinate for the
    # same column would hand the fit two identical rows, one of them with nothing to steer it.
    states = (target.name, *(name for name in constrained if name != target.name))
    lever_names = tuple(lever.name for lever in levers)
    driver_names = _check_drivers(drivers, horizon=horizon, taken=(*states, *lever_names))
    for name in (*states, *lever_names, *driver_names):
        if name not in panel.columns:
            raise KeyError(
                f"column {name!r} is not in the panel; columns are {sorted(panel.names)}"
            )

    if not panel.provenance.x64:
        _log.warning(
            "identifying in single precision; export JAX_ENABLE_X64=1 if the fit is load-bearing",
            extra={"chc_event": "precision", "x64": False, "rows": panel.provenance.n_rows},
        )

    resolved = _resolve_adjustment(adjustment, panel=panel, target=target, levers=lever_names)
    if isinstance(adjustment, CausalGraph):
        parents = {p for lever in lever_names for p in adjustment.parents(lever)}
        logged_on = tuple(sorted(parents - {*lever_names}))
        uncaused = set(adjustment.observed) - adjustment.descendants(lever_names)
        latent = set(logged_on) - set(adjustment.observed)
    else:
        logged_on, uncaused, latent = resolved.covariates, set(), set()
    columns = _Columns(
        states=states,
        logged_on=logged_on,
        unlogged=tuple(sorted((uncaused | {*driver_names}) - {*states, *logged_on})),
        unreadable=tuple(
            name for name in logged_on if name in latent or not _readable(panel, name)
        ),
    )
    _log.info(
        "adjustment resolved",
        extra={
            "chc_event": "adjustment",
            "source": "graph" if isinstance(adjustment, CausalGraph) else "asserted",
            "status": resolved.status,
            "covariates": list(resolved.covariates),
        },
    )
    logger_check = _check_logger(panel, levers=lever_names, columns=columns)
    data = _transitions(
        panel,
        states=states,
        levers=lever_names,
        adjust_for=resolved.covariates,
        drivers=driver_names,
    )
    n_states, n_levers = len(states), len(lever_names)

    base = known or LinearDynamics(jnp.zeros((n_states, n_states)), jnp.zeros((n_states, n_levers)))
    started = time.perf_counter()
    fit = fit_causal_residual(
        base,
        data,
        dt,
        adjust_for=resolved.covariates,
        folds=folds,
        seed=seed,
        integrator=integrator,
        drivers=driver_names,
    )
    _log.info(
        "control channel fitted",
        extra={
            "chc_event": "fit",
            "method": fit.method,
            "identified": fit.identified,
            "channel_error": fit.channel_error,
            "integrator": fit.integrator,
            "integrator_defect": fit.integrator_defect,
            "overlap": fit.action_residual_variance,
            "drivers": list(driver_names),
            "driver_gain": None
            if fit.driver_gain is None
            else np.asarray(fit.driver_gain).tolist(),
            "transitions": int(jnp.asarray(data["x"]).shape[0]),
            "seconds": time.perf_counter() - started,
        },
    )
    identification: IdentificationStatus = (
        "not_identified"
        if resolved.status == "not_identified"
        else ("identified" if isinstance(adjustment, CausalGraph) else "asserted")
    )

    if identification == "not_identified":
        _log.warning(
            "no schedule: no observed set identifies the effect",
            extra={"chc_event": "abort", "reason": resolved.reason},
        )
        return Prescription(
            levers=tuple(levers),
            target=target.name,
            plan=None,
            certificate=DecisionCertificate(
                identification=identification,
                adjustment=resolved,
                identification_radius=None,
                overlap=fit.action_residual_variance,
                certificate_status="not_evaluated",
                certified_horizon=None,
                barrier_certified_steps=None,
                gamma_star=None,
                solver_status=None,
                solver_iterations=0,
                regret_bound=None,
            ),
            model_fit=fit,
            provenance=panel.provenance,
            drivers=tuple(drivers),
            logger_check=logger_check,
            budgets=tuple(budgets),
            _columns=columns,
        )

    if not fit.identified:
        resolved = AdjustmentSet(
            resolved.covariates,
            resolved.status,
            resolved.reason
            + "; the estimator was handed no covariate and no instrument, so its channel is the "
            "observational fit, which is the interventional one only if the lever is unconfounded",
        )

    model: Dynamics = HybridDynamics(known=base, residual=fit.residual)
    if drivers and fit.driver_gain is not None:
        _warn_outside_logged_range(panel, drivers)
        forecast = jnp.stack([jnp.asarray(driver.forecast, dtype=float) for driver in drivers], 1)
        model = DrivenDynamics(model, fit.driver_gain, forecast, dt)
    start = jnp.asarray(data["x0"]) if x0 is None else jnp.asarray(x0)
    u_lo = jnp.array([lever.lo for lever in levers])
    u_hi = jnp.array([lever.hi for lever in levers])
    u_max = float(jnp.max(jnp.maximum(jnp.abs(u_lo), jnp.abs(u_hi))))
    caps = [math.inf if lever.cap_per_step is None else lever.cap_per_step for lever in levers]
    rate = LinearConstraint.rate_limit(horizon, caps)
    # a prescription is the first step of a loop at t = 0 with nothing spent, and reads a budget so
    spend_rows = tuple(
        _period_rows(
            budget, 0.0, 0.0, dt=dt, horizon=horizon, u_lo=u_lo, u_hi=u_hi, levers=n_levers
        )
        for budget in budgets
    )

    margins = _margins(states, constraints)
    held = BarrierConstraint(_barrier(margins), gamma=gamma) if hold_constraints else None

    started = time.perf_counter()
    planning_cost = _cost(states, levers, target, horizon)
    lipschitz = _log_norm(model, start, n_levers)
    model_error = 0.0 if tolerance is None else _model_error(fit, u_max)

    def solve(lo: Array, hi: Array) -> CausalPlan:
        solved = causal_plan(
            model,
            start,
            planning_cost,
            dt,
            horizon,
            lo,
            hi,
            lipschitz=lipschitz,
            model_error=model_error,
            tolerance=float("inf") if tolerance is None else tolerance,
            constraints=(rate, *spend_rows),
            barrier=held,
        )
        if held is None:
            return solved
        # The solve's own audit reads one margin at a tie. Greedy ranks candidates on, and the
        # certificate reports, the audit of every tied margin, so the two cannot disagree.
        audit = _certify(solved, model, margins, dt, gamma=gamma, u_max=u_max)
        return replace(solved, safety=audit)

    def price(solved: CausalPlan) -> float:
        return plan_regret_bound(
            solved, model, start, planning_cost, dt, u_lo, u_hi, probes=4
        ).bound

    selection: LeverSelection | None = None
    if max_levers is None:
        plan = solve(u_lo, u_hi)
    else:
        idle = total_cost(model, start, jnp.zeros((horizon, n_levers)), dt, planning_cost)
        plan, steps = _select_levers(levers, max_levers, solve, price)
        selection = LeverSelection(idle_cost=float(idle), steps=steps)

    _log.info(
        "plan solved",
        extra={
            "chc_event": "plan",
            "solver_status": plan.solver_status,
            "solver_iterations": plan.solver_iterations,
            "certificate_status": plan.certificate_status,
            "certified_horizon": plan.certified_horizon,
            "task_cost": plan.task_cost,
            "rate_limited_levers": [
                lever.name for lever in levers if lever.cap_per_step is not None
            ],
            "constraints_held": held is not None,
            "budget_bounds": [rows.upper.tolist() for rows in spend_rows],
            "seconds": time.perf_counter() - started,
        },
    )

    if held is not None:
        safety = plan.safety  # solve() already audited every tied margin
    elif margins:
        safety = _certify(plan, model, margins, dt, gamma=gamma, u_max=u_max)
    else:
        safety = None
    certificate = DecisionCertificate(
        identification=identification,
        adjustment=resolved,
        identification_radius=fit.channel_error,
        overlap=fit.action_residual_variance,
        certificate_status=plan.certificate_status,
        certified_horizon=plan.certified_horizon,
        barrier_certified_steps=None if safety is None else safety.certified_steps,
        gamma_star=None if safety is None else safety.gamma_star,
        solver_status=plan.solver_status,
        solver_iterations=plan.solver_iterations,
        regret_bound=price(plan),
        gamma=None if safety is None else float(gamma),
    )
    _log.info(
        "decision certified",
        extra={
            "chc_event": "certificate",
            "identification": certificate.identification,
            "gamma": certificate.gamma,
            "gamma_star": certificate.gamma_star,
            "barrier_certified_steps": certificate.barrier_certified_steps,
            "trustworthy_steps": certificate.trustworthy_steps,
            "regret_bound": certificate.regret_bound,
        },
    )
    return Prescription(
        levers=tuple(levers),
        target=target.name,
        plan=plan,
        certificate=certificate,
        model_fit=fit,
        provenance=panel.provenance,
        selection=selection,
        drivers=tuple(drivers),
        logger_check=logger_check,
        budgets=tuple(budgets),
        _columns=columns,
        _budget_rows=tuple(rows.matrix.shape[0] for rows in spend_rows),
    )


# ---- wiring ---------------------------------------------------------------------------------


def _select_levers(
    levers: Sequence[Lever],
    max_levers: int,
    solve: Callable[[Array, Array], CausalPlan],
    price: Callable[[CausalPlan], float],
) -> tuple[CausalPlan, tuple[SelectionStep, ...]]:
    """Greedy forward selection: add the lever whose plan ranks first, ``max_levers`` times at most.

    Each candidate is a cold solve on its own box, the levers not in it pinned to ``[0, 0]``, so its
    plan is the one :func:`prescribe` would make for that set alone, whatever the path to it; once
    every lever is in, the plan is the one ``max_levers=None`` makes. Ties go to the lever listed
    first.
    """
    chosen: list[int] = []
    plans: list[CausalPlan] = []
    steps: list[SelectionStep] = []
    for _ in range(min(max_levers, len(levers))):
        started = time.perf_counter()
        candidates: dict[int, CausalPlan] = {}
        for index in range(len(levers)):
            if index in chosen:
                continue
            keep = {*chosen, index}
            lo = jnp.array([lever.lo if i in keep else 0.0 for i, lever in enumerate(levers)])
            hi = jnp.array([lever.hi if i in keep else 0.0 for i, lever in enumerate(levers)])
            candidates[index] = solve(lo, hi)
        best = min(candidates, key=lambda index: _rank(candidates[index]))
        chosen.append(best)
        plans.append(candidates[best])
        steps.append(SelectionStep(levers[best].name, plans[-1].task_cost, price(plans[-1])))
        _log.info(
            "lever selected",
            extra={
                "chc_event": "selection",
                "step": len(steps),
                "lever": steps[-1].lever,
                "task_cost": steps[-1].task_cost,
                "regret_bound": steps[-1].regret_bound,
                "candidates": {
                    levers[index].name: {
                        "task_cost": plan.task_cost,
                        "cleared_steps": None
                        if plan.safety is None
                        else plan.safety.certified_steps,
                    }
                    for index, plan in candidates.items()
                },
                "descent_steps": sum(plan.solver_iterations for plan in candidates.values()),
                "seconds": time.perf_counter() - started,
            },
        )
    return plans[-1], tuple(steps)


def _rank(plan: CausalPlan) -> tuple[int, float]:
    """Held constraints first --- the longest prefix the audit clears --- then the planned cost.

    ``plan.safety`` is there only when the constraints were held, so otherwise this is the cost.
    """
    cleared = 0 if plan.safety is None else plan.safety.certified_steps
    return -cleared, plan.task_cost


def _resolve_adjustment(
    adjustment: CausalGraph | Sequence[str],
    *,
    panel: Panel,
    target: Target,
    levers: tuple[str, ...],
) -> AdjustmentSet:
    """Derive the set from a graph, or take the caller's word and record that that is what it is."""
    if isinstance(adjustment, CausalGraph):
        adjustment.require_columns(panel.names)
        return adjustment.adjustment_set(treatment=levers, outcome=target.name)
    named = tuple(str(name) for name in adjustment)
    missing = sorted(set(named) - set(panel.names))
    if missing:
        raise KeyError(f"asserted adjustment set names columns not in the panel: {missing}")
    return AdjustmentSet(
        named,
        "identified",
        "asserted by the caller, not derived from a graph; nothing here checked it",
    )


def _transitions(
    panel: Panel,
    *,
    states: tuple[str, ...],
    levers: tuple[str, ...],
    adjust_for: tuple[str, ...],
    drivers: tuple[str, ...] = (),
) -> dict[str, Array]:
    """``(x, u, x_next)`` over consecutive periods within a unit, plus the adjustment columns, and
    each driver at both ends of the transition (``name`` and ``f"{name}_next"``).

    Gaps are dropped, not interpolated: a unit missing period ``t`` contributes the transitions on
    either side of the hole and nothing across it. That is why an unbalanced panel is allowed here
    while :meth:`chc.panel.Panel.wide` refuses one --- a transition needs two adjacent rows, not a
    rectangle.
    """
    unit_codes, time_codes = panel.codes()
    row_of = {
        (int(unit), int(period)): row
        for row, (unit, period) in enumerate(zip(unit_codes, time_codes, strict=True))
    }
    current = np.array(
        [row for (unit, period), row in sorted(row_of.items()) if (unit, period + 1) in row_of],
        dtype=np.int64,
    )
    if current.size == 0:
        raise DecisionError(
            "no unit has two consecutive periods, so there is not a single transition to fit on"
        )
    following = np.array(
        [row_of[(int(unit_codes[row]), int(time_codes[row]) + 1)] for row in current],
        dtype=np.int64,
    )

    def stack(names: tuple[str, ...], rows: np.ndarray) -> Array:
        if not names:
            return jnp.zeros((rows.size, 0))
        return jnp.stack(
            [jnp.asarray(np.asarray(panel[name], dtype=float)[rows]) for name in names], axis=1
        )

    data: dict[str, Array] = {
        "x": stack(states, current),
        "u": stack(levers, current),
        "x_next": stack(states, following),
    }
    for name in adjust_for:
        data[name] = stack((name,), current)
    for name in drivers:
        data[name] = stack((name,), current)
        data[f"{name}_next"] = stack((name,), following)

    units = sorted({unit for unit, _ in row_of})
    final_rows = np.array(
        [row_of[(unit, max(p for u, p in row_of if u == unit))] for unit in units], dtype=np.int64
    )
    data["x0"] = jnp.mean(stack(states, final_rows), axis=0)
    return data


# The class :func:`chc.dynamics_id.fit_causal_residual` regresses the levers on by default.
_LOGGER_CHECK_DEGREE = 2
_LOGGER_CHECK_ALPHA = 0.05  # the level at which a check is logged as a warning


def _readable(panel: Panel, name: str) -> bool:
    return name in panel.columns and np.issubdtype(panel[name].dtype, np.number)


def _check_logger(
    panel: Panel, *, levers: tuple[str, ...], columns: _Columns
) -> LoggerCheck | None:
    """Whether the levers read anything but ``columns.given``, over every period of a unit that
    follows another of the same unit, by :func:`chc.independence.gcm_test` with the periods as
    clusters.

    Tested against the numeric columns of ``columns.unlogged`` the panel holds, at the period, and
    against ``given`` and the levers one period back: the past a Markov logger ignores. ``None``,
    logged, when a lever's parent cannot be read or the rows are fewer than twice the regression's
    terms. A rejection is logged as a warning.
    """
    if columns.unreadable:
        _log.info(
            "logger not checked: the levers' parents are latent or not in the panel",
            extra={"chc_event": "logger_check", "unreadable": list(columns.unreadable)},
        )
        return None
    given = columns.given
    unit_codes, time_codes = panel.codes()
    row_of = {
        (int(unit), int(period)): row
        for row, (unit, period) in enumerate(zip(unit_codes, time_codes, strict=True))
    }
    pairs = [
        (row, row_of[(unit, period - 1)])
        for (unit, period), row in sorted(row_of.items())
        if (unit, period - 1) in row_of
    ]
    terms = math.comb(len(given) + _LOGGER_CHECK_DEGREE, _LOGGER_CHECK_DEGREE)
    if len(pairs) < 2 * terms:
        _log.info(
            "logger not checked: too few rows",
            extra={"chc_event": "logger_check", "rows": len(pairs), "terms": terms},
        )
        return None
    now = np.array([row for row, _ in pairs], dtype=np.int64)
    before = np.array([previous for _, previous in pairs], dtype=np.int64)

    def read(names: Sequence[str], rows: NDArray[np.int64]) -> NDArray[np.float64]:
        return np.column_stack([np.asarray(panel[name], dtype=np.float64)[rows] for name in names])

    present = tuple(name for name in columns.unlogged if _readable(panel, name))
    past = (*given, *levers)
    against = (
        read(past, before) if not present else np.hstack([read(present, now), read(past, before)])
    )
    check = LoggerCheck(
        levers=levers,
        given=given,
        columns=(*present, *(CausalGraph.lagged_name(name, 1) for name in past)),
        test=gcm_test(
            read(levers, now),
            against,
            read(given, now),
            clusters=time_codes[now],
            degree=_LOGGER_CHECK_DEGREE,
        ),
    )
    rejected = check.test.p_value <= _LOGGER_CHECK_ALPHA
    _log.log(
        logging.WARNING if rejected else logging.INFO,
        "the levers read a column besides the state and their recorded parents"
        if rejected
        else "logger checked",
        extra={
            "chc_event": "logger_check",
            "p_value": check.test.p_value,
            "given": list(check.given),
            "columns": list(check.columns),
            "rows": len(pairs),
            "clusters": check.test.clusters,
        },
    )
    return check


@dataclass(frozen=True)
class _Episodes:
    """The windows :meth:`Prescription.evaluate` reads: ``x (E, H + 1, n)``, ``u (E, H, m)``, and
    ``units (E,)``, each window's unit as :meth:`chc.panel.Panel.codes` codes it."""

    x: NDArray[np.float64]
    u: NDArray[np.float64]
    units: NDArray[np.int64]


def _episodes(
    panel: Panel,
    *,
    states: tuple[str, ...],
    levers: tuple[str, ...],
    horizon: int,
    time_zero: TimeZero,
) -> _Episodes:
    """Every window of ``H + 1`` consecutive periods of a unit that starts at a time zero, a unit's
    latest first.

    ``"calendar"``: the time zeros are the panel's periods ``H``, ``2H``, ... before its last, the
    same for every unit, and a unit gives each window it was observed throughout. ``"unit"``: each
    unit's windows are cut back from its own latest period, so that a window starts where the one
    before it ends; a gap ends a run, as in :func:`_transitions`, and the oldest periods of a run
    short of a window are left out.

    Raises:
        DecisionError: on fewer than two windows.
    """
    unit_codes, time_codes = panel.codes()
    order = np.lexsort((time_codes, unit_codes))
    if time_zero == "calendar":
        # Sorted by unit and period, a unit's periods rise strictly: H + 1 rows in a row of one
        # unit whose periods differ by H are H + 1 consecutive periods.
        units, times = unit_codes[order], time_codes[order]
        last = panel.n_periods - 1
        first = np.flatnonzero(((last - times) % horizon == 0) & (times + horizon <= last))
        first = first[first + horizon < order.size]
        end = first + horizon
        first = first[(units[end] == units[first]) & (times[end] - times[first] == horizon)]
        first = first[np.lexsort((-times[first], units[first]))]
        rows = order[first[:, None] + np.arange(horizon + 1)]
    else:
        breaks = np.flatnonzero(
            (np.diff(unit_codes[order]) != 0) | (np.diff(time_codes[order]) != 1)
        )
        windows = [
            order[run[end - horizon : end + 1]]
            for run in np.split(np.arange(order.size), breaks + 1)
            for end in range(run.size - 1, horizon - 1, -horizon)
        ]
        rows = np.stack(windows) if windows else np.empty((0, horizon + 1), dtype=np.int64)
    if rows.shape[0] < 2:
        raise DecisionError(
            f"an evaluation over episodes needs two windows of {horizon + 1} consecutive periods "
            f"in a unit, and the panel has {rows.shape[0]}"
        )

    def stack(names: tuple[str, ...]) -> NDArray[np.float64]:
        return np.stack([np.asarray(panel[name], dtype=np.float64) for name in names], axis=1)

    return _Episodes(stack(states)[rows], stack(levers)[rows[:, :-1]], unit_codes[rows[:, 0]])


@eqx.filter_jit
def _one_step(model: Dynamics, x: Array, u: Array, dt: float) -> tuple[Array, Array, Array]:
    """``model``'s RK4 step over ``dt`` from ``(x, u)`` and its Jacobians there, compiled: an
    evaluation linearises once for every draw of its bootstrap."""
    a, b = linearize_discrete(model, x, u, dt)
    return a, b, rk4_step(model, 0.0, x, u, dt)


def _linearised(
    model: Dynamics, x: NDArray[np.float64], u: NDArray[np.float64], dt: float
) -> LinearGaussianPlant:
    """``model``'s step over ``dt`` linearised at the mean state and action of the episodes
    ``x (E, H + 1, n)`` and ``u (E, H, m)``, with the covariance of its one-step residuals on them
    as the noise."""
    n, m = x.shape[-1], u.shape[-1]
    xs, xn, us = x[:, :-1].reshape(-1, n), x[:, 1:].reshape(-1, n), u.reshape(-1, m)
    x_bar, u_bar = xs.mean(axis=0), us.mean(axis=0)
    a, b, step = (
        np.asarray(part, dtype=np.float64)
        for part in _one_step(model, jnp.asarray(x_bar), jnp.asarray(u_bar), dt)
    )
    offset = step - a @ x_bar - b @ u_bar
    residual = xn - xs @ a.T - us @ b.T - offset
    return LinearGaussianPlant(a, b, offset, np.atleast_2d(np.cov(residual, rowvar=False)))


def _check_drivers(
    drivers: Sequence[Driver], *, horizon: int, taken: tuple[str, ...]
) -> tuple[str, ...]:
    """The drivers' names, once each checked against the horizon and the decision's own columns."""
    names = tuple(driver.name for driver in drivers)
    twice = sorted({name for name in names if names.count(name) > 1})
    if twice:
        raise DecisionError(f"drivers named more than once: {twice}")
    clash = sorted(set(names) & set(taken))
    if clash:
        raise DecisionError(
            f"columns {clash} are named as drivers and as a lever or a state; a driver is "
            "exogenous, and a column the plan can move is not"
        )
    for driver in drivers:
        levels = np.asarray(driver.forecast, dtype=float)
        if levels.shape != (horizon + 1,):
            raise DecisionError(
                f"driver {driver.name!r} has a forecast of shape {levels.shape}; it needs a level "
                f"at the start of every step and one at the end of the last, {horizon + 1} for "
                "this horizon"
            )
        if not np.all(np.isfinite(levels)):
            raise DecisionError(f"driver {driver.name!r} has a forecast that is not finite")
    return names


def _warn_outside_logged_range(panel: Panel, drivers: Sequence[Driver]) -> None:
    for driver in drivers:
        logged = np.asarray(panel[driver.name], dtype=float)
        low, high = float(logged.min()), float(logged.max())
        levels = np.asarray(driver.forecast, dtype=float)
        if levels.min() < low or levels.max() > high:
            _log.warning(
                "the forecast for %r leaves the range the panel logged, [%.4g, %.4g]; the fitted "
                "gain is extrapolated there",
                driver.name,
                low,
                high,
                extra={
                    "chc_event": "driver_range",
                    "driver": driver.name,
                    "logged": [low, high],
                    "forecast": [float(levels.min()), float(levels.max())],
                },
            )


def _cost(
    states: tuple[str, ...], levers: Sequence[Lever], target: Target, horizon: int
) -> QuadraticCost:
    """Weight the target's own coordinate and price each lever; constrained states are free.

    A constrained state gets weight zero rather than a small one: its bound is enforced by the
    barrier and priced by the certificate, and adding a quadratic pull towards zero would be a
    second, unstated objective.

    A schedule becomes one target per state. The start's row repeats the first level; no action
    reaches the start, so that row moves the reported cost and nothing the plan does.
    """
    weights = jnp.array([target.weight] + [0.0] * (len(states) - 1))
    if np.ndim(target.value) == 0:
        goal = jnp.array([target.value] + [0.0] * (len(states) - 1))
    else:
        levels = jnp.asarray(target.value)
        path = jnp.concatenate([levels[:1], levels])
        goal = jnp.zeros((horizon + 1, len(states))).at[:, 0].set(path)
    q = jnp.diag(weights)
    return QuadraticCost(
        Q=q, R=jnp.diag(jnp.array([lever.unit_cost for lever in levers])), Qf=q, x_target=goal
    )


def _margins(states: tuple[str, ...], constraints: Sequence[Constraint]) -> tuple[_Margin, ...]:
    """An affine margin per finite bound, ``x - lo`` then ``hi - x``, in constraint order."""
    index = {name: position for position, name in enumerate(states)}
    margins: list[_Margin] = []
    for constraint in constraints:
        position = index[constraint.state]
        if constraint.lo is not None:
            margins.append(_margin(position, constraint.lo, 1.0))
        if constraint.hi is not None:
            margins.append(_margin(position, constraint.hi, -1.0))
    return tuple(margins)


class _Margin(eqx.Module):
    """``sign (x[position] - bound)``: one bound's margin, safe where it is ``>= 0``.

    A module and not a closure over the bound: the held solve's compiled programs take the barrier
    as an argument, and a closure would be a new static one at every call, so a replanning loop
    would compile its programs again at every call and keep them all. As an array, a bound that
    moves between calls is a value, not a program.
    """

    position: int = eqx.field(static=True)
    sign: float = eqx.field(static=True)
    bound: Array

    def __call__(self, x: Array) -> Array:
        return self.sign * (x[self.position] - self.bound)


def _margin(position: int, bound: float, sign: float) -> _Margin:
    return _Margin(position, sign, jnp.asarray(bound, dtype=float))


class _Barrier(eqx.Module):
    """``h(x) = min_j m_j(x)``, safe where ``h >= 0``: the one barrier a held solve takes.

    Where margins tie the minimum has no gradient, and the solve reads one anyway: the first tied
    margin's, selected explicitly, because ``jnp.min`` averages tied gradients and the two margins
    of a two-sided bound tie at its midpoint with opposite ones --- a zero gradient, which held
    nothing there. The audit does not read this gradient; :func:`_certify` checks every tied margin.
    """

    margins: tuple[_Margin, ...]

    def __call__(self, x: Array) -> Array:
        stacked = jnp.stack([margin(x) for margin in self.margins])
        return stacked[jnp.argmin(stacked)]


def _barrier(margins: Sequence[_Margin]) -> _Barrier:
    return _Barrier(tuple(margins))


def _certify(
    plan: CausalPlan,
    model: Dynamics,
    margins: Sequence[Callable[[Array], Array]],
    dt: float,
    *,
    gamma: float,
    u_max: float,
) -> SafetyCertificate:
    """:func:`chc.plan.certify_safety` for ``h = min_j m_j``, run on each margin as its own barrier.

    Along a trajectory the right derivative of a minimum is the smallest of its tied terms', so the
    condition on ``h`` holds at a step iff it holds for every margin at the minimum there, and the
    step is certified only if each of them is. At a tie ``h`` has no gradient, and one margin's
    would pass a state leaving through another: at the midpoint of a two-sided bound, a state moving
    fast towards ``hi`` clears ``lo``'s condition. Where one margin is the minimum --- the ordinary
    case --- this is the audit of ``h`` itself.

    A tied step's ``gamma_star`` is the weakest of its margins', an upper bound rather than the
    ceiling itself: each margin's is reached by its own best action, and no one action need reach
    the smallest. The two agree when every tied margin holds with no action at all, where both are
    infinite.
    """
    audits = [
        certify_safety(plan, model, margin, dt, gamma=gamma, u_max=u_max) for margin in margins
    ]
    values = jnp.stack([audit.barrier_values for audit in audits])
    tied = values == jnp.min(values, axis=0)  # no margin at all where a value is nan
    passed = jnp.stack([audit.planned_certified for audit in audits])
    certified = jnp.any(tied, axis=0) & jnp.all(passed | ~tied, axis=0)
    guaranteed = jnp.stack([audit.guaranteed_derivative for audit in audits])
    ceilings, rows = np.asarray([audit.step_gamma_star for audit in audits]), np.asarray(tied)
    step_gamma_star = tuple(
        float(np.min(column[on])) if on.any() else math.nan
        for column, on in zip(ceilings.T, rows.T, strict=True)
    )
    return SafetyCertificate(
        barrier_values=jnp.min(values, axis=0),
        guaranteed_derivative=jnp.nanmin(jnp.where(tied, guaranteed, jnp.nan), axis=0),
        required=jnp.max(jnp.stack([audit.required for audit in audits]), axis=0),  # -alpha * h
        planned_certified=certified,
        certified_steps=(
            int(jnp.argmin(certified)) if not bool(jnp.all(certified)) else int(certified.shape[0])
        ),
        gamma_star=(
            math.nan if any(math.isnan(g) for g in step_gamma_star) else min(step_gamma_star)
        ),
        step_gamma_star=step_gamma_star,
        radius=max(audit.radius for audit in audits),
    )


def _log_norm(model: Dynamics, x0: Array, n_levers: int) -> float:
    """The Gronwall rate: the logarithmic norm of the drift Jacobian at ``(x0, 0)``.

    ``lambda_max((A + A^T)/2)``, which is the tightest one-sided bound on ``||e||``'s growth and can
    be negative --- a contractive plant then *shrinks* the tube (§30) rather than being clamped to
    zero growth, which a plain spectral-norm bound would do.
    """
    a_matrix, _ = linearize_continuous(model, x0, jnp.zeros(n_levers))
    symmetric = (a_matrix + a_matrix.T) / 2
    return float(jnp.max(jnp.linalg.eigvalsh(symmetric)))


def _model_error(fit: CausalDynamicsFit, u_max: float) -> float:
    """Turn the channel's standard error into the per-step rate error the Gronwall tube wants.

    ``channel_error`` bounds ``||B_hat - B||`` per entry; the rate error it induces is that times
    the largest action the box allows, which is what this returns. It is a *scale*, not coverage:
    ``channel_error`` is the robust sandwich's root-mean diagonal, calibrated on average but
    scattering from one log to the next, so one fit's tube is as wide as that fit's reading. The
    docstring of :class:`chc.dynamics_id.CausalDynamicsFit` says by how much.
    """
    return 0.0 if fit.channel_error is None else float(fit.channel_error) * max(u_max, 1.0)


def _show(value: object) -> str:
    if value is None:
        return "not evaluated"
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def _show_price(price: RowPrice) -> str:
    if price.price is None:
        return f"[{price.interval[0]:.4g}, {price.interval[1]:.4g}]"
    return f"{price.price:.4g}"
