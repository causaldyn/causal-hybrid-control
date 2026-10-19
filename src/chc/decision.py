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

import datetime
import logging
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from importlib import metadata
from typing import Any, Literal, get_args

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2
from scipy.stats import t as student_t

from chc import _units
from chc.causal import _polynomial_features
from chc.control import LinearConstraint, SolverStatus
from chc.cost import QuadraticCost, total_cost
from chc.dynamics import (
    DampedOscillator,
    DrivenDynamics,
    Dynamics,
    HybridDynamics,
    LinearDynamics,
    _EulerStep,
    _SetAtStep,
)
from chc.dynamics_id import (
    CausalDynamicsFit,
    Integrator,
    InterferenceCheck,
    _absorbed,
    _channel_design,
    _cosines,
    _degrees_of_freedom,
    _logged_relations,
    _LoggedRelations,
    _nuisance_inputs,
    _persistence,
    _unmoved_actions,
    fit_causal_residual,
    interference_check,
    persistence_check,
)
from chc.evaluation import (
    AffinePolicy,
    AffineSchedule,
    LinearGaussianPlant,
    LoggerCheck,
    PlanEvaluation,
    _evaluate_across_time,
    _evaluate_by_unit,
)
from chc.frames import _not_numbers, _real_numbers
from chc.graph import AdjustmentSet, CausalGraph
from chc.independence import gcm_test
from chc.integrate import rk4_step
from chc.lqr import linearize_continuous, linearize_discrete
from chc.mpc import PeriodBudget, _period_rows
from chc.panel import Panel, PanelError, Provenance, _grid_steps, _ordinals, _x64_enabled
from chc.plan import (
    BarrierConstraint,
    CausalPlan,
    CertificateStatus,
    DiagnosticStatus,
    PlanRegretBound,
    RegretStatus,
    RowPrice,
    _audit,
    _carried_along,
    _diagnostic_status,
    _Rule,
    _Ruled,
    _RuledCost,
    causal_plan,
    plan_regret_bound,
)
from chc.residual import ControlAffineResidual, ZeroResidual, control_affine_features
from chc.uncertainty import _TUBE_COVERAGE, _NoiseRead, _read_covariances

SCHEMA_VERSION = 2
"""``to_json``'s schema version. Bumped when a field changes meaning, not when one is added: 2
writes a number that is not finite as null (ADR 0055)."""

POLICY_SCHEMA_VERSION = 1
""":meth:`PrescribedPolicy.to_json`'s schema version, which :meth:`PrescribedPolicy.from_json`
reads, and no other."""


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
    fall back to :mod:`chc.research.sensitivity`'s partial-identification path on exactly this and
    on nothing else; catching it by message would be catching it by accident.
    """


_log = logging.getLogger(__name__)
"""Decision-point log for :func:`prescribe`, on the stdlib and nothing else.

Every record carries a ``chc_event`` key in its ``extra`` payload naming the point it was emitted
at --- ``precision``, ``adjustment``, ``logger_check``, ``fit``, ``abort``, ``unmoved``, ``rule``,
``relation``, ``tube``, ``selection`` (one per step under ``max_levers``), ``driver_range``,
``plan``, ``lever_range``, ``estimability``, ``certificate``, and ``one_unit`` from
:meth:`Prescription.evaluate` --- so a JSON formatter downstream can route on one field rather than
parse a sentence. The library installs no handler and sets no level: that is the application's
call, and a library that reaches for ``basicConfig`` takes it away.

The records that are not ``INFO`` are the ones worth waking someone for: identifying in single
precision, a graph that says the effect is not identified at all, a driver's forecast outside the
range the panel logged, levers that read more than the state and their recorded parents or read
those through more than a quadratic, levers held to what the log did with them, a plan some fit
the log cannot tell apart predicts differently, a tube whose rate or budget is not a finite
number, and an evaluation that reads one unit's windows as independent.
"""

IdentificationStatus = Literal["identified", "asserted", "not_identified"]
"""How the set was arrived at. ``asserted`` means a caller named it and nothing here checked it."""

TubeRate = Literal["global", "local"]
"""Where the error tube's rate bounds the field's slope in the state: at every state, the levers
anywhere in their box, or at the plan's start alone."""

TubeMode = Literal["exact", "bound"]
"""What each radius of the error tube is. Each holds, with probability 0.95, the plant's own noise
(ADR 0080) and the fitted field's error, the channel's and the drift's, beside it: a standard
error whose spread the log estimates, read at the state each of the plan's steps starts from and
at its studentised quantile as an error independent of the noise (ADR 0104). ``exact``: on a field
affine in the state, both carried through RK4's step matrix of the field's slope at each planned
action, so the radius is the exact 0.95 quantile of the two at that step under the fitted field.
``bound``: both carried by a bound on the norm of the slope, so the radius holds at least 0.95,
exactly at the first step."""

TimeZero = Literal["calendar", "unit"]
"""Where :meth:`Prescription.evaluate`'s windows start: on one calendar for every unit, or cut back
from each unit's own latest period."""

Estimability = Literal["estimable", "held_to_log", "not_estimable"]
"""Whether the log determines the plan's predicted path (ADR 0054). ``estimable``: the log moved
every direction of the channel. ``held_to_log``: it left some unmoved, and the plan keeps to what
the log did along them, so every fit the log cannot tell from the fitted one predicts the same
path. ``not_estimable``: some such fit predicts another path from
:attr:`DecisionCertificate.first_loaded_step` on, or no plan could keep to the log and none was
made."""

GammaStarStatus = Literal["finite", "every_level", "no_level"]
"""What :attr:`DecisionCertificate.gamma_star` reads. ``finite``: the level past which some step's
barrier no admissible action keeps. ``every_level``: ``inf``, every step's barrier is kept at every
level, the radius then the whole gap's share of the fitted channel: against every effect within
that much of the estimate, not against any effect at all. ``no_level``: nan, some step's barrier no
admissible action keeps, not even at ``gamma = 1``."""


@dataclass(frozen=True)
class LeverRelation:
    """A combination of the levers the log kept at one level, ``sum_j weights[j] u_j = level``,
    which the plan keeps at every step. ``weights`` has one entry per lever and unit length."""

    weights: tuple[float, ...]
    level: float


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
        # a nan bound compared false and reached the solve, which clipped every action to nan
        for side, bound in (("lo", self.lo), ("hi", self.hi)):
            if math.isnan(bound):
                raise DecisionError(
                    f"lever {self.name!r} has {side}=nan; a bound is a number, inf on a free side"
                )
        if self.lo > self.hi:
            raise DecisionError(f"lever {self.name!r} has lo={self.lo} above hi={self.hi}")
        if not math.isfinite(self.unit_cost):
            raise DecisionError(
                f"lever {self.name!r} has unit_cost={self.unit_cost}; a price is a finite number"
            )
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

    def __post_init__(self) -> None:
        value = _real_numbers(self.value, f"the value of target {self.name!r}", DecisionError)
        object.__setattr__(self, "value", value)
        # a nan level or weight made the task cost nan, and the solve stopped where it started
        values = np.asarray(value, dtype=np.float64).reshape(-1)
        bad = np.flatnonzero(~np.isfinite(values))
        if bad.size:
            raise DecisionError(
                f"target {self.name!r} has value {values[bad[0]]} at entry {bad[0]}; a level to "
                "steer to is a finite number"
            )
        if not math.isfinite(self.weight):
            raise DecisionError(
                f"target {self.name!r} has weight={self.weight}; a weight is a finite number"
            )


@dataclass(frozen=True)
class Constraint:
    """A state that must stay inside a range --- what the safety certificate is priced against.

    Each finite bound is a barrier of its own, ``x - lo`` or ``hi - x``, and a step is certified
    where the condition of every bound holds, the far one's as well as the near one's: the state
    approaches neither faster than the class-K gain times its distance to it.

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
        # a lone nan bound was never compared, and held a barrier no action can keep
        for side, bound in (("lo", self.lo), ("hi", self.hi)):
            if bound is not None and math.isnan(bound):
                raise DecisionError(
                    f"constraint on {self.state!r} has {side}=nan; a bound is a number"
                )
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

    def __post_init__(self) -> None:
        forecast = _real_numbers(
            self.forecast, f"the forecast of driver {self.name!r}", DecisionError
        )
        object.__setattr__(self, "forecast", forecast)


@dataclass(frozen=True)
class InterventionSchedule:
    """What to set each lever to, step by step."""

    levers: tuple[str, ...]
    magnitudes: Array  # (horizon, m)
    # The levers whose column is the rule of the state the log set them by, read along the
    # predicted path: set them from the state as it comes (:meth:`Prescription.policy`), not to
    # these numbers.
    rules: tuple[str, ...] = ()

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
class PrescribedPolicy:
    """The decision as a rule to run, step by step, in the state each step reaches.

    A lever the plan moves takes the schedule's level at the step. A lever the log set from the
    state alone (:attr:`DecisionCertificate.rule_levers`) takes the log's rule at the state it is
    given, as the plan's field and its price read it (ADR 0054), so its column of
    :attr:`schedule` is nan. A lever the plan holds to the log, or one the selection left out,
    keeps the level the schedule holds it at. Along the path the plan predicts, the actions are
    :attr:`Prescription.schedule`'s. *Experimental.*
    """

    levers: tuple[str, ...]
    states: tuple[str, ...]  # the order :meth:`actions_at` reads a state's values in
    dt: float
    schedule: NDArray[np.float64]  # (horizon, levers); a ruled lever's column is nan
    _rule: _Rule | None = field(default=None, repr=False, compare=False)

    @property
    def horizon(self) -> int:
        """How many steps the policy runs."""
        return int(self.schedule.shape[0])

    @property
    def ruled(self) -> tuple[str, ...]:
        """The levers set from the state on the log's rule, in the order of :attr:`levers`."""
        rule = self._rule
        return () if rule is None else tuple(self.levers[index] for index in rule.levers)

    def actions_at(self, step: int, state: ArrayLike | Mapping[str, float]) -> NDArray[np.float64]:
        """Each lever's level at ``step`` in ``state``, in the order of :attr:`levers`.

        ``state`` is the states' values in the order of :attr:`states`, or a mapping that names
        each of them; a name it holds besides is not read.

        Raises:
            ValueError: on a step that is not an integer from 0 to ``horizon - 1``, or a state
                that does not give each of :attr:`states` a finite value.
        """
        if isinstance(step, bool) or not isinstance(step, int | np.integer):
            raise ValueError(f"a step is an integer, not {step!r}")
        if not 0 <= step < self.horizon:
            raise ValueError(f"the policy runs steps 0 to {self.horizon - 1}, not {step}")
        return _act(self._rule, self._read(state), self.schedule[step])

    def _read(self, state: ArrayLike | Mapping[str, float]) -> NDArray[np.float64]:
        if isinstance(state, Mapping):
            missing = [name for name in self.states if name not in state]
            if missing:
                raise ValueError(f"the state does not name {missing}")
            state = [state[name] for name in self.states]
        x = np.asarray(_real_numbers(state, "state"), dtype=np.float64)
        if x.shape != (len(self.states),):
            raise ValueError(
                f"a state holds one value for each of {list(self.states)}, not shape {x.shape}"
            )
        if not np.all(np.isfinite(x)):
            raise ValueError(f"a state's values are finite numbers, not {x.tolist()}")
        return x

    def to_json(self) -> dict[str, Any]:
        """The policy as plain JSON-safe values, which :meth:`from_json` reads back to the bit.

        No number is infinite or nan (ADR 0055). A ruled lever's column of ``schedule`` is None,
        as its rule sets it; ``rule`` names those levers and holds the rule, an infinite bound of
        its as None: ``lo``'s is -inf, ``hi``'s inf. ``precision`` names the floats the rule
        reads the state in, None where there is no rule.
        """
        rule = self._rule

        def floats(values: ArrayLike) -> Any:
            return np.asarray(values, dtype=np.float64).tolist()

        return _strict(
            {
                "schema_version": POLICY_SCHEMA_VERSION,
                "levers": list(self.levers),
                "states": list(self.states),
                "dt": float(self.dt),
                "precision": None if rule is None else str(rule.centre.dtype),
                "schedule": floats(self.schedule),
                "rule": None
                if rule is None
                else {
                    "levers": list(self.ruled),
                    "degree": rule.degree,
                    "centre": floats(rule.centre),
                    "shift": floats(rule.shift),
                    "factor": floats(rule.factor),
                    "coefficients": floats(rule.coefficients),
                    "lo": floats(rule.lo),
                    "hi": floats(rule.hi),
                },
            }
        )

    @classmethod
    def from_json(cls, record: Mapping[str, Any]) -> PrescribedPolicy:
        """The policy :meth:`to_json` wrote.

        Raises:
            ValueError: on a record of another schema version, or one that does not run: names
                that are not distinct, a schedule that is not a finite number for each lever at
                each step, a ruled lever's None excepted, a rule that does not fit the levers and
                the states, or a rule read in a precision JAX does not now run in.
        """
        if record.get("schema_version") != POLICY_SCHEMA_VERSION:
            raise ValueError(
                f"this version reads schema_version {POLICY_SCHEMA_VERSION}, not "
                f"{record.get('schema_version')!r}"
            )
        levers, states = _names(record, "levers"), _names(record, "states")
        dt = record.get("dt")
        if isinstance(dt, bool) or not isinstance(dt, int | float) or not 0.0 < dt < math.inf:
            raise ValueError(f"dt is a finite step above 0, not {dt!r}")
        written = record.get("rule")
        rule = None if written is None else _read_rule(written, levers, states, record)
        ruled = set() if rule is None else set(rule.levers)
        rows = record.get("schedule")
        if (
            not isinstance(rows, list)
            or not rows
            or not all(isinstance(row, list) and len(row) == len(levers) for row in rows)
        ):
            raise ValueError(f"the schedule is a row of {len(levers)} levels for each step")
        for row in rows:
            for index, value in enumerate(row):
                if (value is None) != (index in ruled) or not (value is None or _number(value)):
                    raise ValueError(
                        f"a ruled lever's level is None, as its rule sets it, and every other "
                        f"lever's a finite number; {levers[index]!r} reads {value!r}"
                    )
        schedule = np.array(
            [[np.nan if value is None else value for value in row] for row in rows],
            dtype=np.float64,
        )
        return cls(levers=levers, states=states, dt=float(dt), schedule=schedule, _rule=rule)


def _names(record: Mapping[str, Any], key: str) -> tuple[str, ...]:
    names = record.get(key)
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError(f"{key} is a list of distinct names, not {names!r}")
    return tuple(names)


def _number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _read_rule(
    written: object, levers: tuple[str, ...], states: tuple[str, ...], record: Mapping[str, Any]
) -> _Rule:
    """The rule :meth:`PrescribedPolicy.to_json` wrote, in the floats it was read in."""
    if not isinstance(written, Mapping):
        raise ValueError(f"the rule is a mapping, not {written!r}")
    names = _names(written, "levers")
    unknown = [name for name in names if name not in levers]
    if unknown:
        raise ValueError(f"the rule sets {unknown}, which are not among the levers {list(levers)}")
    degree = written.get("degree")
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
        raise ValueError(f"the rule's degree is a whole number, not {degree!r}")
    precision = record.get("precision")
    if precision not in ("float32", "float64"):
        raise ValueError(f"a rule's precision is float32 or float64, not {precision!r}")
    if (precision == "float64") != _x64_enabled():
        raise ValueError(
            f"the rule was read in {precision} and JAX now runs in the other precision: set "
            "jax_enable_x64 as it was set where the rule was written, so it runs as written"
        )
    dtype = jnp.float64 if precision == "float64" else jnp.float32
    features = _polynomial_features(jnp.zeros((1, len(states))), degree).shape[1]
    shapes = {
        "centre": (len(states),),
        "shift": (len(states),),
        "factor": (len(states),),
        "coefficients": (features, len(names)),
        "lo": (len(names),),
        "hi": (len(names),),
    }
    arrays = {}
    for key, shape in shapes.items():
        values = written.get(key)
        if key in ("lo", "hi") and isinstance(values, list):
            free = -math.inf if key == "lo" else math.inf
            values = [free if value is None else value for value in values]
        array = np.asarray(values, dtype=np.float64) if _numeric(values) else None
        if array is None or array.shape != shape or np.isnan(array).any():
            raise ValueError(f"the rule's {key} is {shape} numbers, not {values!r}")
        arrays[key] = array
    if not all(np.isfinite(arrays[key]).all() for key in ("centre", "shift", "coefficients")):
        raise ValueError("the rule's centre, shift and coefficients are finite numbers")
    # a state the log held at one level reads 0 in the rule, by a factor of 0
    if not (np.isfinite(arrays["factor"]).all() and (arrays["factor"] >= 0.0).all()):
        raise ValueError(
            f"the rule's factor is finite and at least 0, not {arrays['factor'].tolist()}"
        )
    if not (arrays["lo"] <= arrays["hi"]).all():
        raise ValueError("the rule's lo lies at or below its hi")
    return _Rule(
        **{key: jnp.asarray(array, dtype=dtype) for key, array in arrays.items()},
        levers=tuple(levers.index(name) for name in names),
        degree=degree,
    )


def _numeric(values: object) -> bool:
    """Whether ``values`` is a nested list of numbers that are not booleans."""
    if isinstance(values, list):
        return all(_numeric(value) for value in values)
    return isinstance(values, int | float) and not isinstance(values, bool)


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
    # handed covariates, an instrument whose moment has rank, or the word that the levers are
    # unconfounded?". A graph that proves the effect identified with nothing to adjust for passes
    # that word on; an asserted empty set does not, and leaves the flag False.
    adjustment: AdjustmentSet
    identification_radius: float | None  # channel standard error; None when not identified
    overlap: float  # residualised action variance: 0 means the log has no variation to learn from
    certificate_status: CertificateStatus
    # steps the tube keeps inside ``tolerance``: the radius that holds, with probability 0.95 at
    # each step, the plan's one-step error on the log (``tube_noise``) and the field's error, the
    # channel's at each lever's largest distance from the log's mean action and the drift's, read
    # at each planned state, both carried by the plan's steps, the field's read at its studentised
    # quantile (:data:`TubeMode`)
    certified_horizon: int | None
    barrier_certified_steps: int | None  # leading prefix clearing the barrier; None if no bound
    # The weakest step's confounding ceiling; None if unconstrained. With several bounds a step's
    # is the weakest bound's ceiling, an upper bound on the joint one: each bound's is reached by
    # its own best action, and no one action need reach them all.
    gamma_star: float | None
    # None when the effect is not identified: no solve was attempted, which is a third answer and
    # not one of the solver's three. Same convention as the certified-horizon fields above.
    solver_status: SolverStatus | None
    solver_iterations: int
    # How far the plan can be from the best one the SAME BOX allows on the FITTED model (Result 69),
    # read from its own gradient with no optimum needed; ``regret_status`` says whether that is a
    # certificate. ``inf`` means the objective was not convex where its curvature was read, so
    # nothing bounds it; ``None`` means no plan was solved at all. This is a gap in the PLANNING
    # objective -- how far the planning model is from the plant is the tube's question, and the two
    # must not be added.
    regret_bound: float | None
    # The marginal sensitivity model's level the barrier's prefix was audited at, the caller's
    # ``gamma``: ``barrier_certified_steps`` counts the steps the plan clears there, and
    # ``gamma_star`` is where no action would. None where no bound was audited.
    gamma: float | None = None
    # What ``regret_bound`` is (:data:`chc.plan.RegretStatus`): ``certified`` only where the fitted
    # model is linear, so the objective is quadratic and its one Hessian is the box's; elsewhere the
    # curvature is sampled at the plan and a few points, and the bound is ``diagnostic``. None where
    # no plan was solved.
    regret_status: RegretStatus | None = None
    # The levers whose whole channel the log never moved (:attr:`CausalDynamicsFit.unmoved`): no
    # transition shows what moving them does, so the plan keeps each to what the log did, at its
    # logged level or on its logged rule of the state (``rule_levers``), and with every lever here
    # there is no plan.
    unmoved_levers: tuple[str, ...] = ()
    # Where the tube's rate holds (:data:`TubeRate`): ``global`` on a field affine in the state,
    # whose slope in it is the same at every state and is bounded over the levers' whole box, so
    # the tube is a bound; ``local`` where the slope is read at the start alone, so away from it
    # nothing proves the tube. None where no tube was evaluated.
    tube_rate: TubeRate | None = None
    # The plant's own noise the tube carries (ADR 0080): the plan's one-step error on the log, the
    # next state less the plan's step of the fitted field from the logged state and levers, which
    # the channel's error does not cover. ``tube_noise`` is the root of its covariance's largest
    # eigenvalue, in the states' units, the error's standard deviation on one state;
    # ``tube_persistence`` its lag-1 correlation within units, pooled over the states, each in its
    # own units; and ``tube_autoregression`` the ``(n, n)`` coefficient ``Phi`` of the VAR(1) that
    # carries it from step to step, ``Cov(w_{k+1}, w_k) = Phi S``, each direction of the noise
    # persisting at its own rate (ADR 0105). None where no tube was evaluated.
    tube_noise: float | None = None
    tube_persistence: float | None = None
    tube_autoregression: tuple[tuple[float, ...], ...] | None = None
    # What each radius of ``uncertainty_tube`` is (:data:`TubeMode`, ADR 0096): ``exact`` where the
    # tube is carried by the field's step matrices along the plan, ``bound`` where by a norm bound
    # on its slope. None where no tube was evaluated.
    tube_mode: TubeMode | None = None
    # Under ``exact``, each state's own 0.95 radius of the noise's part of the gap at each step,
    # ``sqrt(chi2_1(0.95) P_k[i, i])``, ``P_k`` that part's covariance at step ``k``, steps 0 to the
    # horizon: what each state's error alone reaches, in that state's units. A report, not the
    # tube: ``tolerance`` reads the states as one Euclidean ball, and the field's error scale is
    # not split by state. None under ``bound``, whose covariance is a bound on the largest
    # variance alone, and where no tube was evaluated.
    tube_state_noise: tuple[tuple[float, ...], ...] | None = None
    # The column whose groups ``identification_radius`` sums the scores within before squaring them
    # (CR1): the panel's cluster where it declares one, else its unit, whose transitions share any
    # persistent noise; ``error_clusters`` counts the groups. ``error_periods`` counts the periods
    # the transitions start in where the error is read two ways (ADR 0061), so that a shock every
    # unit shares in a period is read: the largest of the groups' and the periods' sums together
    # and of each alone. None where it is not, as where they all start in one. ``error_cosines``
    # counts the cosines of the periods the error is read with where the transitions name one group
    # alone, which leaves no second group to sum within (EWC, ADR 0075): on one unit, whose
    # ``error_clustered_by`` is None, and on several units in one declared cluster, whose
    # ``error_clustered_by`` names it with ``error_clusters`` 1 and whose transitions are summed
    # within each period first (Driscoll-Kraay). A test read off the error is then sized against
    # ``t(error_cosines)``. None where it is not, and so is ``error_clustered_by`` where the one
    # group's transitions all start in one period: the error then takes each transition as
    # independent.
    error_clustered_by: str | None = None
    error_clusters: int | None = None
    error_periods: int | None = None
    error_cosines: int | None = None
    # The lag-1 autocorrelation of the channel moment's residual within units, and its two-sided
    # p-value (:func:`chc.dynamics_id.persistence_check`): where the noise persists and a lever does
    # too, the channel is biased by an amount its error does not cover (ADR 0064). None where the
    # channel is not identified or no unit has two consecutive transitions; the p-value is None
    # where the pairs leave no spread to read, as one pair does. Where the error is read across the
    # periods (``error_cosines``), the p-value reads the pairs' products the same way, summed
    # within each period first (ADR 0075); where it is read within groups and periods
    # (``error_clustered_by``, ``error_periods``), it reads them within those, the largest of the
    # same three reads, against ``t(G - 1)``.
    noise_persistence: float | None = None
    noise_persistence_p: float | None = None
    # Whether the log determines the plan's predicted path (:data:`Estimability`, ADR 0054); None
    # where the effect is not identified and nothing was asked of the log's actions.
    estimability: Estimability | None = None
    # Per state, how many directions of the channel's coefficients the log's actions moved, and how
    # many they did not (:attr:`CausalDynamicsFit.unmoved`).
    identification_rank: int | None = None
    unmoved_directions: int | None = None
    # The first step at which a fit the log cannot tell from the fitted one predicts another path,
    # counted from 0, so also the number of steps before it, which every such fit predicts alike;
    # None where none does. ``trustworthy_steps`` ends there.
    first_loaded_step: int | None = None
    # The combinations of the levers the log kept at one level, which the plan keeps by an equality
    # row at every step, and the levers the log set from the state alone, which follow that rule
    # along the plan.
    relations: tuple[LeverRelation, ...] = ()
    rule_levers: tuple[str, ...] = ()
    # The plan's first-order check (:attr:`chc.plan.CausalPlan.stationarity_status`,
    # :attr:`chc.plan.CausalPlan.violation_status`): its KKT residual and its largest breach of its
    # box and rows, each beside its tolerance and read in the descent's own variables, so the same
    # in any units of the levers, the states or the target; under the constraints' barrier, with
    # its binding steps among the constraints. None where no plan was solved.
    stationarity: float | None = None
    stationarity_tolerance: float | None = None
    constraint_violation: float | None = None
    violation_tolerance: float | None = None

    @property
    def stationarity_status(self) -> DiagnosticStatus | None:
        """How ``stationarity`` reads against its tolerance (:data:`chc.plan.DiagnosticStatus`);
        None where no plan was solved."""
        if self.solver_status is None:
            return None
        return _diagnostic_status(self.stationarity, self.stationarity_tolerance)

    @property
    def violation_status(self) -> DiagnosticStatus | None:
        """How ``constraint_violation`` reads against its tolerance; None where no plan was
        solved."""
        if self.solver_status is None:
            return None
        return _diagnostic_status(self.constraint_violation, self.violation_tolerance)

    @property
    def trustworthy_steps(self) -> int:
        """The prefix that survives *both* axes --- the number an operator can act on.

        Zero whenever the effect is not identified, whatever the tube says, and never longer than
        the shorter of the two certified prefixes, nor past the first step a fit the log cannot
        tell apart predicts differently. The two ``None`` values mean different things.
        ``certified_horizon is None`` means the tube was not evaluated, so nothing bounds the
        trajectory's error and it contributes zero rather than infinity --- a barrier cleared by an
        unbounded trajectory proves nothing. ``barrier_certified_steps is None`` means no state was
        bounded, so there is no safety prefix to respect and the tube alone decides.

        Where the channel's residual persists within units (``noise_persistence_p`` at most 0.05)
        the noise the tube reads may not be the plant's, and a ``tube`` warning says so; the
        prefix still counts the tube's steps (ADR 0105).
        """
        if self.identification == "not_identified" or self.certified_horizon is None:
            return 0
        steps = self.certified_horizon
        if self.barrier_certified_steps is not None:
            steps = min(steps, self.barrier_certified_steps)
        return steps if self.first_loaded_step is None else min(steps, self.first_loaded_step)

    @property
    def gamma_star_status(self) -> GammaStarStatus | None:
        """What ``gamma_star`` reads (:data:`GammaStarStatus`); None where no bound was audited."""
        if self.gamma_star is None:
            return None
        if math.isnan(self.gamma_star):
            return "no_level"
        return "every_level" if math.isinf(self.gamma_star) else "finite"


@dataclass(frozen=True)
class SelectionStep:
    """One step of the greedy lever selection: the lever it added, and what the plan cost with it.

    ``regret_bound`` is :func:`chc.plan.plan_regret_bound` on that step's plan, priced against
    every lever's box, the levers not yet selected included. So it bounds how far below
    ``task_cost`` any plan the boxes allow can go: what the levers left out at this step could
    still buy, plus what the solve left on the table. ``inf`` when the objective was not convex
    over the boxes, as in :attr:`DecisionCertificate.regret_bound`; ``regret_status`` says which.
    """

    lever: str
    task_cost: float  # the planned cost with the levers selected so far, this one included
    regret_bound: float
    regret_status: RegretStatus | None = None  # what ``regret_bound`` is; None where not given


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
class StartState:
    """The state the plan starts from, or would have, and where it came from.

    ``source`` is ``"given"`` for a caller's ``x0`` and ``"panel"`` for the panel's own: the mean
    over its units of each unit's last logged state. A mean over units that sit far apart is a
    start no unit is at, so the panel's start keeps how many units it averages, the periods their
    last states were logged at, and each state's spread over them.
    """

    source: Literal["given", "panel"]
    states: tuple[str, ...]  # the plan's states, in its order: the target, then the constrained
    value: tuple[float, ...]
    units: int | None = None  # how many units the mean is over; None when given
    periods: tuple[Any, ...] = ()  # the distinct periods the units' last states were logged at
    # Each state's standard deviation over the units, as a population: 0 with one unit. None when
    # given.
    spread: tuple[float, ...] | None = None


@dataclass(frozen=True)
class RunProvenance:
    """What the run used, beside the panel's :class:`~chc.panel.Provenance` (ADR 0063).

    The panel's record is made when the panel is built, and a run need not match it: a panel built
    under JAX's ``x64`` and planned on with it off fits on float32 transitions while its record
    reads True. So each field here is read where the run used it: ``x64`` as :func:`prescribe` ran;
    the device and the fit's dtype off the transitions the fit ran on; the solve's dtype, why it
    stopped and after how many steps off the plan the planner returned; each setting as the fit and
    the planner were given it; and each version from the installed distribution's metadata.
    """

    x64: bool  # jax_enable_x64 when prescribe ran
    # The device the fit's transitions sat on, where jax ran the fit: its platform, as
    # jax.default_backend() names one ("cpu", "gpu", "tpu"), and its kind. Inside a
    # jax.default_device block that is the block's device, which the default backend need not hold.
    backend: str
    device_kind: str
    fit_dtype: str  # of the transitions the fit ran on
    # The plan the planner returned, the one Prescription.plan holds: the dtype of its actions, why
    # its descent stopped (chc.control.SolverStatus) and its accepted steps, as the certificate
    # reads them too. None where no plan was solved.
    solve_dtype: str | None
    solver_status: SolverStatus | None
    solver_iterations: int | None
    seed: int  # the cross-fitting folds' seed
    integrator: Integrator  # the one-step map the fit was made consistent with
    # (distribution, version): the library and those that compute its numbers, from their metadata;
    # "unknown" for one installed without any
    versions: tuple[tuple[str, str], ...]
    folds: int
    degree: int  # the drift's feature degree
    channel_degree: int  # the channel's feature degree
    nuisance_degree: int  # the cross-fitted nuisances' polynomial degree
    ridge: float  # the fit's ridge
    steps: int  # the most steps each of the planner's descents takes
    # The tube's radius past which a step is not certified; None where no tube was asked for. JSON
    # writes an infinite one as null as well, and the certificate's status says whether a tube was
    # evaluated.
    tolerance: float | None
    hold_constraints: bool  # whether the solve held the constraints, or only the audit priced them
    max_levers: int | None  # the greedy selection's cap; None where every lever was planned with
    clusters: str | None = None  # the column the units were grouped by; None where not grouped
    nuisance_knots: int | str | None = None  # the nuisances' spline knots; None, polynomial alone

    def to_json(self) -> dict[str, Any]:
        """A plain dict, ready for ``json.dumps(..., allow_nan=False)``: the versions an object
        keyed by distribution, and an infinite tolerance null."""
        return _strict(
            {
                "x64": self.x64,
                "backend": self.backend,
                "device_kind": self.device_kind,
                "fit_dtype": self.fit_dtype,
                "solve_dtype": self.solve_dtype,
                "solver_status": self.solver_status,
                "solver_iterations": self.solver_iterations,
                "seed": self.seed,
                "integrator": self.integrator,
                "versions": dict(self.versions),
                "folds": self.folds,
                "degree": self.degree,
                "channel_degree": self.channel_degree,
                "nuisance_degree": self.nuisance_degree,
                "ridge": self.ridge,
                "steps": self.steps,
                "tolerance": self.tolerance,
                "hold_constraints": self.hold_constraints,
                "max_levers": self.max_levers,
                "clusters": self.clusters,
                "nuisance_knots": self.nuisance_knots,
            }
        )


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
    # Whether the units' transitions move with their neighbours' levers, where ``neighbours`` was
    # given and the fit kept a moment residual; reported, not acted on. *Experimental.*
    interference: InterferenceCheck | None = None
    budgets: tuple[PeriodBudget, ...] = ()  # what the plan was held to spend
    start: StartState | None = None  # where the plan starts, or would have; None if not recorded
    run: RunProvenance | None = None  # what the run used; None if not recorded
    _columns: _Columns | None = field(default=None, repr=False, compare=False)
    # The state the plan starts from, or would have: where :meth:`reach` reads the channel.
    _start: Array | None = field(default=None, repr=False, compare=False)
    # Per lever, whether the log identifies its own effect where :meth:`reach` reads it; None on a
    # prescription :func:`prescribe` did not build.
    _alone: tuple[bool, ...] | None = field(default=None, repr=False, compare=False)
    # How many of the plan's constraint rows each budget holds: the last ones, in order.
    _budget_rows: tuple[int, ...] = field(default=(), repr=False, compare=False)
    # The plan's actions with each lever that follows its logged rule read off that rule along the
    # predicted path; None where no lever does, and the schedule is the plan's actions.
    _magnitudes: Array | None = field(default=None, repr=False, compare=False)

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
                :mod:`chc.research.sensitivity` for what to do about it.
        """
        if self.plan is None:
            raise NotIdentifiedError(
                "the effect is not identified, so no schedule was computed: "
                f"{self.certificate.adjustment.reason}"
            )
        return InterventionSchedule(
            levers=self.lever_names,
            magnitudes=self.plan.actions if self._magnitudes is None else self._magnitudes,
            rules=self.certificate.rule_levers,
        )

    def policy(self) -> PrescribedPolicy:
        """The decision as a rule to run step by step in the states it reaches: the schedule for
        the levers the plan moves, the log's rule of the state for those the log set from it.

        Raises:
            NotIdentifiedError: if the effect is not identified, as :attr:`schedule` does.
            ValueError: on a prescription :func:`prescribe` did not build, which does not record
                its states or the rule its levers keep to.
        """
        plan = self.plan
        if plan is None:
            raise NotIdentifiedError(
                "the effect is not identified, so no policy was computed: "
                f"{self.certificate.adjustment.reason}"
            )
        problem = plan._problem
        if self._columns is None or problem is None:
            raise ValueError(
                "this prescription does not record its states and its levers' rule: build it "
                "with prescribe"
            )
        rule = problem.model.rule if isinstance(problem.model, _Ruled) else None
        schedule = np.array(plan.actions, dtype=np.float64)
        if rule is not None:
            schedule[:, list(rule.levers)] = np.nan  # nothing reads them: the rule sets them
        return PrescribedPolicy(
            levers=self.lever_names,
            states=self._columns.states,
            dt=float(problem.dt),
            schedule=schedule,
            _rule=rule,
        )

    def budget_prices(self, tolerance: float | None = None) -> tuple[tuple[RowPrice, ...], ...]:
        """What each budget is worth to the plan: one price per period it holds in the plan.

        A period's price is how much the planned cost falls per unit more it may spend, in the
        cost's units per unit of spend; zero where the period has room. Read as
        :meth:`chc.plan.CausalPlan.shadow_prices` reads every row, whose ``tolerance`` this is.
        A period the horizon cuts short may spend its share of the budget's ``amount``, so a unit
        more of ``amount`` is worth its price times that share, which :meth:`report` prints.

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
        max_refused: float = 0.05,
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
        few distinct panels and the intervals too narrow. The estimate of a quadratic cost under
        importance weights is skewed to the right, and the spread its draws read rises and falls
        with it, so the misses fall where it reads low: on panels of 60 units of five windows,
        no draw refused, the plan's interval covered 0.925 on a two-state market (800 panels) and
        0.902 on one state (757 panels), the difference's 0.945 and 0.916. On 30 units of ten
        windows they covered 0.914 and 0.880 (500 panels), the difference's 0.916 and 0.928, and
        the draws' spread read 0.83 to 0.87 of the estimate's. A BCa interval over units, a log or
        square-root transform of the cost, Hall's skewness correction and the logged policy's
        fitted influence each left one of these four short of 0.95.

        Near the certificate's edge a draw whose units start wider or weigh heavier than the
        panel's is refused, so the draws read are the narrow side of the spread: the intervals
        are then conditional on certification, and :attr:`~chc.evaluation.UnitBootstrap.reading`
        says so, ``"conditional"``. Where more than ``max_refused`` of the draws are refused it
        is ``"inconclusive"``, logged as a warning: the intervals are too narrow to read.
        :attr:`~chc.evaluation.UnitBootstrap.accepted` and ``accepted_share`` count the draws
        read.

        Windows of one unit leave nothing to resample, and they follow each other through its
        state and its noise. Each window's influence on the estimate, the delta method's, is read
        across the windows' time zeros by the equal-weighted cosine estimate, ``nu`` cosines of
        ``E`` windows, ``nu`` the most whole ones at or below ``0.4 E^(2/3)``, and each interval is
        Student's ``t(nu)`` on it (Lazarus, Lewis, Stock and Watson 2018), widened by the model's
        correction; :attr:`PlanEvaluation.degrees_of_freedom` is ``nu`` and ``bootstrap`` is
        ``None``. A window's time zero takes its place on the grid of ``H`` periods, so a gap leaves
        its places empty. The logger and the smoothing are held at the estimate's. Measured on one
        simulated unit (a loop keeping 0.5 or 0.8 of its state a step, its noise AR(0), AR(0.5) or
        AR(0.9), horizons 4 and 10, 25 to 1600 windows, 1000 logs a cell), the 95 % intervals
        covered 0.87 to 0.99, where the windows taken as independent covered 0.75 to 0.99; where
        the loop keeps 0.8 and the noise is AR(0.9), 0.82 to 0.95 against 0.64 to 0.85. They were
        1.04 to 2.3 times as wide, 1.66 at 25 windows of fresh noise (ADR 0083). The
        difference's interval, :attr:`PlanEvaluation.versus_logger`, covered less, for the same
        skew: 0.88 at 100 windows of four steps of fresh noise where the plan's covered 0.97, and
        0.74 to 0.79 at ten steps where the loop keeps 0.8 and the noise is AR(0.9).

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
        :attr:`Prescription.logger_check` on the panel the plan was fitted on). On enough rows it
        also flags levers that read the state through more than a quadratic, such as a logger that
        switches at a threshold; the logger fitted here is affine, so that premise fails too. A
        rejection is logged as a warning and changes nothing else. *Experimental.*

        Raises:
            NotIdentifiedError: if the effect is not identified, so there is no plan.
            DecisionError: on a ``time_zero`` other than ``"calendar"`` and ``"unit"``; fewer than
                two resamples; a ``max_refused`` outside ``[0, 1)``; a plan made against driver
                forecasts, whose plant changes with the step; a plan with a lever that follows its
                logged rule of the state, which no open-loop schedule carries; a plan whose levers
                were logged on a column outside its state, or whose record does not say; or a
                panel with fewer than two windows.
            InfeasibleEvaluation: when the evaluation's certificate refuses, on the panel or on
                every draw but one.
            PanelError: when a state's or a lever's column of ``panel`` does not hold real
                numbers, as :func:`prescribe` refuses it.
        """
        if time_zero not in get_args(TimeZero):
            raise DecisionError(f"time_zero must be one of {get_args(TimeZero)}, got {time_zero!r}")
        if n_resamples < 2:
            raise DecisionError(
                f"n_resamples={n_resamples}: an interval read from draws of the units needs at "
                "least two of them"
            )
        if not 0.0 <= max_refused < 1.0:
            raise DecisionError(
                f"max_refused={max_refused}: the share of refused draws above which the intervals "
                "are inconclusive lies in [0, 1)"
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
        if self.certificate.rule_levers:
            raise DecisionError(
                f"{list(self.certificate.rule_levers)} follow their logged rule of the state, a "
                "policy the evaluation's open-loop schedule cannot carry"
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
            # Windows H periods apart, or further across a gap, each hold a place of their own on
            # the grid of H periods.
            places = (episodes.starts - episodes.starts.min()) // len(actions)
            cosines = _cosines(places)
            _log.info(
                "one unit's windows read across their time zeros by cosines",
                extra={
                    "chc_event": "one_unit",
                    "windows": int(episodes.x.shape[0]),
                    "cosines": int(cosines.shape[0]),
                },
            )
            evaluation = _evaluate_across_time(
                episodes.x,
                episodes.u,
                cosines,
                schedule,
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
                max_refused=max_refused,
            )
        return replace(evaluation, logger_check=logger_check)

    def reach(self) -> dict[str, float | None]:
        """Per lever, how far it can move the target's rate across its own box, where the plan
        starts; None where the log does not identify the lever's own effect.

        The fitted control channel on the target's row, read at the state the plan starts from, or
        would have where the effect is not identified, times the width of the lever's box. A
        channel affine in the state has no one value: up to 0.12 this read its constant term, its
        value at ``x = 0``, which a log of rooms at 20 °C never comes near, and a fit whose channel
        was negative everywhere in the log read positive there. A large coefficient on a lever that
        may barely move is not a large lever, which is why the range is in the number and not only
        in the footnote.

        A lever the log set from the state, or moved only together with others, reads None. Its
        entry of the channel there leans on a direction the log never moved
        (:attr:`chc.dynamics_id.CausalDynamicsFit.unmoved`), where the fit holds the channel by a
        convention, not an estimate: on a log whose second lever was always twice the first, up to
        0.14 this read 2.05 and 1.02, a split of their joint effect that any other split fits as
        well.

        Read from the *same* fit that produced the plan, deliberately. Ranking by a second
        estimator --- local projections, say --- invites an ordering that contradicts the schedule
        printed beside it, and two disagreeing orderings on one page is worse than one.

        Raises:
            ValueError: on a prescription that records no start, one built other than by
                :func:`prescribe` without a plan; or on one whose fit left a direction unmoved
                and that does not record which levers the log identifies alone, one built other
                than by :func:`prescribe`.
        """
        start = self._start
        if start is None and self.plan is not None:
            start = self.plan.trajectory[0]
        if start is None:
            raise ValueError(
                "this prescription records no state to read the channel at: build it with prescribe"
            )
        alone = self._alone
        if alone is None:
            unmoved = self.model_fit.unmoved
            if unmoved is not None and unmoved.shape[1] > 0:
                raise ValueError(
                    "the fit left directions of the channel unmoved, and this prescription does "
                    "not record which levers the log identifies alone: build it with prescribe"
                )
            alone = (True,) * len(self.levers)
        channel = np.asarray(self.model_fit.residual.control_channel(start))  # (states, levers)
        return {
            lever.name: float(channel[0, index] * (lever.hi - lever.lo)) if alone[index] else None
            for index, lever in enumerate(self.levers)
        }

    def explain(self) -> str:
        """The levers ranked by :meth:`reach`, widest first, as text, and those whose own effect
        the log does not identify, by name."""
        reach = self.reach()
        ranked = sorted(
            ((name, value) for name, value in reach.items() if value is not None),
            key=lambda item: -abs(item[1]),
        )
        lines = [f"levers ranked by reach on {self.target!r} (channel x box width):"]
        lines += [f"  {name:<20} {value:+.4g}" for name, value in ranked]
        joint = [name for name, value in reach.items() if value is None]
        if joint:
            lines.append(
                "not ranked, the log set them from the state or moved them only together with "
                f"other levers, so it does not identify their own effect: {', '.join(joint)}"
            )
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
        if self.start is not None:
            lines += ["", _show_start(self.start)]
        if self.plan is None:
            lines += ["", "**No schedule.** " + certificate.adjustment.reason, ""]
        else:
            windows = self.schedule.windows()
            lines += ["", "| lever | active steps | first | last |", "|---|---|---|---|"]
            magnitudes = np.asarray(self.schedule.magnitudes)
            for index, name in enumerate(self.lever_names):
                window = windows[name]
                span = "never" if window is None else f"{window[0]}-{window[1]}"
                first, last = magnitudes[0, index], magnitudes[-1, index]
                lines.append(f"| `{name}` | {span} | {first:+.4g} | {last:+.4g} |")
            if certificate.rule_levers:
                lines += [
                    "",
                    "Set "
                    + ", ".join(f"`{name}`" for name in certificate.rule_levers)
                    + " from the state as it comes, by the rule the log set them by: their columns "
                    "are that rule read along the predicted path.",
                ]
            lines += ["", f"Planned task cost: {self.plan.task_cost:.6g}.", ""]
            # a plan held under the constraints' barrier carries no row prices
            priced = self.budget_prices() if self.budgets and self.plan.safety is None else None
            horizon = int(self.plan.actions.shape[0])
            for index, budget in enumerate(self.budgets):
                # a period the horizon cuts short holds its share of the amount, so a unit more of
                # the amount raises its row by that share, and lowers the cost by its price times it
                shares = [
                    min(budget.period, horizon - first) / budget.period
                    for first in range(0, horizon, budget.period)
                ]
                worth = (
                    "not priced, as the plan was held under the constraints' barrier"
                    if priced is None
                    else "a unit more in each period would lower the planned cost by "
                    + ", ".join(
                        _show_price(price, share)
                        for price, share in zip(priced[index], shares, strict=True)
                    )
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
            *self._estimability_lines(),
            *self._driver_lines(),
            f"- channel standard error: {_show(certificate.identification_radius)}"
            + self._error_grouping(),
            f"- overlap (residualised action variance): {certificate.overlap:.4g}",
            self._logger_line(),
            self._persistence_line(),
            *self._interference_lines(),
            f"- error tube: **{certificate.certificate_status}**, "
            + ("" if certificate.tube_mode is None else f"{certificate.tube_mode} radii, ")
            + f"certified horizon {_show(certificate.certified_horizon)}"
            + ("" if certificate.tube_rate is None else f", {certificate.tube_rate} rate"),
            *self._noise_lines(),
            f"- barrier: certified steps {_show(certificate.barrier_certified_steps)}"
            + ("" if certificate.gamma is None else f" at gamma {certificate.gamma:.4g}")
            + f", gamma* {_show(certificate.gamma_star)} (a relative margin on a share of the "
            "fitted channel, not a level of confounding)",
            "- solver: not run"
            if certificate.solver_status is None
            else f"- solver: {certificate.solver_status} after "
            f"{certificate.solver_iterations} accepted steps",
            *self._first_order_lines(),
            f"- regret bound on the fitted model: {_show(certificate.regret_bound)}"
            + ("" if certificate.regret_status is None else f", {certificate.regret_status}"),
            "",
            f"**Trustworthy prefix: {certificate.trustworthy_steps} steps.**",
            "",
            "## Provenance",
            "",
            f"- data sha256: `{self.provenance.data_sha256[:16]}...`",
            f"- panel: chc {self.provenance.chc_version}, {self.provenance.n_rows} rows, "
            f"x64={self.provenance.x64}, seed={self.provenance.seed}",
            *self._run_lines(),
        ]
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """The schedule, both certificate axes and the provenance, as plain JSON-safe values.

        No number is infinite or nan: one that is not finite is None (ADR 0055). Where None has
        more than one reading, a status beside it says which: ``regret_status`` ``refused`` for an
        infinite ``regret_bound``, a selection step's own ``regret_status`` for its bound,
        ``gamma_star_status`` for ``gamma_star`` (:data:`GammaStarStatus`), and
        ``stationarity_status`` and ``violation_status`` for ``stationarity`` and
        ``constraint_violation`` (:data:`chc.plan.DiagnosticStatus`).
        """
        certificate = self.certificate
        record = {
            "schema_version": SCHEMA_VERSION,
            "target": self.target,
            "levers": list(self.lever_names),
            "schedule": None
            if self.plan is None
            else np.asarray(self.schedule.magnitudes).tolist(),
            "start": None
            if self.start is None
            else {
                "source": self.start.source,
                "states": list(self.start.states),
                "value": list(self.start.value),
                "units": self.start.units,
                "periods": [_json_label(period) for period in self.start.periods],
                "spread": None if self.start.spread is None else list(self.start.spread),
            },
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
                "gamma_star_status": certificate.gamma_star_status,
                "solver_status": certificate.solver_status,
                "solver_iterations": certificate.solver_iterations,
                "stationarity": certificate.stationarity,
                "stationarity_tolerance": certificate.stationarity_tolerance,
                "stationarity_status": certificate.stationarity_status,
                "constraint_violation": certificate.constraint_violation,
                "violation_tolerance": certificate.violation_tolerance,
                "violation_status": certificate.violation_status,
                "trustworthy_steps": certificate.trustworthy_steps,
                "regret_bound": certificate.regret_bound,
                "regret_status": certificate.regret_status,
                "unmoved_levers": list(certificate.unmoved_levers),
                "tube_rate": certificate.tube_rate,
                "tube_noise": certificate.tube_noise,
                "tube_persistence": certificate.tube_persistence,
                "tube_autoregression": None
                if certificate.tube_autoregression is None
                else [list(row) for row in certificate.tube_autoregression],
                "tube_mode": certificate.tube_mode,
                "tube_state_noise": None
                if certificate.tube_state_noise is None
                else [list(radii) for radii in certificate.tube_state_noise],
                "error_clustered_by": certificate.error_clustered_by,
                "error_clusters": certificate.error_clusters,
                "error_periods": certificate.error_periods,
                "error_cosines": certificate.error_cosines,
                "noise_persistence": certificate.noise_persistence,
                "noise_persistence_p": certificate.noise_persistence_p,
                "interference_p": None
                if self.interference is None
                else _finite(self.interference.p_value),
                "estimability": certificate.estimability,
                "identification_rank": certificate.identification_rank,
                "unmoved_directions": certificate.unmoved_directions,
                "first_loaded_step": certificate.first_loaded_step,
                "relations": [
                    {"weights": list(relation.weights), "level": relation.level}
                    for relation in certificate.relations
                ],
                "rule_levers": list(certificate.rule_levers),
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
                        "regret_status": step.regret_status,
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
            "run": None if self.run is None else self.run.to_json(),
        }
        return _strict(record)

    def _first_order_lines(self) -> list[str]:
        certificate = self.certificate
        if certificate.solver_status is None:
            return []
        return [
            f"- first-order check: stationarity {_show(certificate.stationarity)} (tolerance "
            f"{_show(certificate.stationarity_tolerance)}, {certificate.stationarity_status}), "
            f"constraint violation {_show(certificate.constraint_violation)} (tolerance "
            f"{_show(certificate.violation_tolerance)}, {certificate.violation_status})"
        ]

    def _run_lines(self) -> list[str]:
        run = self.run
        if run is None:
            return []
        solve = "no solve" if run.solve_dtype is None else f"solve in {run.solve_dtype}"
        tolerance = "none" if run.tolerance is None else f"{run.tolerance:g}"
        cap = "none" if run.max_levers is None else str(run.max_levers)
        return [
            f"- run: on {run.backend} ({run.device_kind}), x64={run.x64}, fit in {run.fit_dtype}, "
            f"{solve}, seed={run.seed}, integrator={run.integrator}",
            f"- settings: folds={run.folds}, degree={run.degree}, "
            f"channel_degree={run.channel_degree}, nuisance_degree={run.nuisance_degree}, "
            f"ridge={run.ridge:g}, steps={run.steps}, tolerance={tolerance}, "
            f"hold_constraints={run.hold_constraints}, max_levers={cap}"
            + ("" if run.clusters is None else f", clusters=`{run.clusters}`"),
            "- versions: " + ", ".join(f"{name} {version}" for name, version in run.versions),
        ]

    def _error_grouping(self) -> str:
        certificate = self.certificate
        if certificate.identification_radius is None:
            return ""
        if certificate.error_cosines is not None and certificate.error_clustered_by is None:
            return (
                f", read across the one unit's periods by {certificate.error_cosines} cosines of "
                "them (EWC)"
            )
        if certificate.error_cosines is not None:
            return (
                f", summed within each period over the units of the one group of "
                f"`{certificate.error_clustered_by}`, then read across the periods by "
                f"{certificate.error_cosines} cosines of them (Driscoll-Kraay, EWC)"
            )
        if certificate.error_clustered_by is None:
            return ", each transition taken as independent"
        grouped = (
            f", summed within {certificate.error_clusters} groups of "
            f"`{certificate.error_clustered_by}`"
        )
        if certificate.error_periods is None:
            return f"{grouped} (CR1)"
        return (
            f"{grouped}, within each of {certificate.error_periods} periods, and within both, "
            "whichever reads largest (two-way CR1)"
        )

    def _persistence_line(self) -> str:
        certificate = self.certificate
        correlation, p_value = certificate.noise_persistence, certificate.noise_persistence_p
        if correlation is None:
            if certificate.identification == "not_identified":
                return "- noise persistence: not read, the channel is not identified"
            if self.model_fit.moment_residual is None:
                return (
                    "- noise persistence: not read, the fit is the observational one, which keeps "
                    "no moment residual"
                )
            return "- noise persistence: not read, no unit has two consecutive transitions"
        head = f"lag-1 autocorrelation {correlation:+.2f} within units"
        if p_value is None:
            return f"- noise persistence: {head}, not tested: its pairs leave no spread to read"
        if p_value > _PERSISTENCE_ALPHA:
            return f"- noise persistence: {head} (p = {p_value:.3g})"
        return (
            f"- noise persistence: **the noise persists within units** ({head}, "
            f"p = {p_value:.3g}): where a lever persists too, the channel is biased by an amount "
            "its error does not cover, and the noise the tube reads may not be the plant's; "
            "adjust for the states, the levers and the adjustment set a period earlier"
        )

    def _interference_lines(self) -> list[str]:
        check = self.interference
        if check is None:
            return []
        if math.isnan(check.p_value):
            return [
                "- interference: not tested, the neighbours' levers leave fewer than two groups or "
                "never move within a period"
            ]
        head = f"p = {check.p_value:.3g} over {check.groups} groups"
        if check.p_value > _INTERFERENCE_ALPHA:
            return [f"- interference: none found ({head})"]
        state, lever = np.unravel_index(np.nanargmax(np.abs(check.t)), check.t.shape)
        name = "a state" if self._columns is None else f"`{self._columns.states[state]}`"
        return [
            f"- interference: **the units' transitions move with their neighbours' levers** "
            f"({head}); strongest: `{self.lever_names[lever]}` on {name}, "
            f"{check.coefficient[state, lever]:+.4g} per unit of the neighbours' mean: the channel "
            "is the effect of moving one unit with its neighbours held; group the units with "
            "`clusters` to estimate the effect of moving them together"
        ]

    def _noise_lines(self) -> list[str]:
        certificate = self.certificate
        if certificate.tube_noise is None or certificate.tube_persistence is None:
            return []
        return [
            f"- the tube's noise: the plan's one-step error on the log, standard deviation "
            f"{certificate.tube_noise:.4g}, lag-1 autocorrelation "
            f"{certificate.tube_persistence:+.2f} pooled over the states, each state's carried at "
            "its own rate; with the fitted field's error, the channel's and the drift's, the tube "
            "holds the plant at each step with probability 0.95",
            *self._state_noise_lines(),
        ]

    def _state_noise_lines(self) -> list[str]:
        certificate = self.certificate
        radii, start = certificate.tube_state_noise, self.start
        if radii is None or start is None:
            return []
        step = max(certificate.certified_horizon or 0, 1)
        if step >= len(radii):
            return []
        each = ", ".join(
            f"`{name}` {radius:.4g}" for name, radius in zip(start.states, radii[step], strict=True)
        )
        return [f"- each state's own 0.95 radius of the noise at step {step}: {each}"]

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
            return (
                "- logger check: nothing to test, the state determines every column or the log "
                "never moved it"
            )
        if test.clusters < 2:
            return "- logger check: nothing to test, every row it can test falls in one period"
        head = f"p = {test.p_value:.3g} over {test.clusters} periods"
        if test.p_value <= _LOGGER_CHECK_ALPHA:
            correlation = test.partial_correlation
            lever, column = np.unravel_index(np.nanargmax(np.abs(correlation)), correlation.shape)
            return (
                f"- logger check: **the levers read more than the record says, or read it "
                f"through more than a quadratic** ({head}); "
                f"strongest: `{check.levers[lever]}` on `{check.columns[column]}`, partial "
                f"correlation {correlation[lever, column]:+.2f}"
            )
        seen = test.detectable[np.isfinite(test.detectable)]
        missed = f"a partial correlation up to {seen.max():.2g}" if seen.size else "any dependence"
        return f"- logger check: passed ({head}); a pass can miss {missed}"

    def _estimability_lines(self) -> list[str]:
        certificate = self.certificate
        held = [name for name in certificate.unmoved_levers if name not in certificate.rule_levers]
        lines = []
        if held:
            names = ", ".join(f"`{name}`" for name in held)
            lines.append(f"- never moved by the log, so held at their logged level: {names}")
        if certificate.rule_levers:
            names = ", ".join(f"`{name}`" for name in certificate.rule_levers)
            lines.append(
                f"- set by the log from the state alone, so they follow that rule: {names}"
            )
        if certificate.relations:
            kept = "; ".join(
                _show_relation(relation, self.lever_names) for relation in certificate.relations
            )
            lines.append(f"- kept at the level the log kept them: {kept}")
        if certificate.estimability is not None:
            line = f"- estimability: **{certificate.estimability}**"
            if certificate.identification_rank is not None:
                total = certificate.identification_rank + (certificate.unmoved_directions or 0)
                line += (
                    f", the log moved {certificate.identification_rank} of the channel's {total} "
                    "directions a state"
                )
            if certificate.first_loaded_step is not None:
                agree = certificate.first_loaded_step
                steps = "step" if agree == 1 else "steps"
                line += (
                    f"; a fit the log cannot tell apart predicts the plan's first {agree} {steps} "
                    f"alike and another path from step {agree + 1} on"
                )
            lines.append(line)
        return lines

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
    x0: ArrayLike | None = None,
    folds: int = 2,
    seed: int = 0,
    integrator: Integrator = "rk4",
    drivers: Sequence[Driver] = (),
    neighbours: Mapping[Any, Iterable[Any]] | str | None = None,
    clusters: str | None = None,
    nuisance_knots: int | Literal["auto"] | None = None,
) -> Prescription:
    """Fit the causal control channel from ``panel`` and plan a certified schedule on it.

    Args:
        panel: long-format logs. Consecutive periods within a unit become the ``(x, u, x_next)``
            transitions the channel is fitted on; gaps are dropped rather than interpolated, so an
            unbalanced panel is fine and a silently invented row is not. The channel's standard
            error, and with it the tube's budget, sums the scores within each group of the panel's
            declared cluster, or of its unit where it declares none, before squaring them
            (``clusters`` in :func:`~chc.dynamics_id.fit_causal_residual`): the transitions of
            one unit share whatever persistent noise the model leaves out. It sums them within
            each period as well, and the error is the largest of the groups' and the periods' sums
            together and of each alone (ADR 0061): a shock every unit shares in a period, met by
            levers the units move together, makes the transitions of one period move together
            across units, and summed by unit alone the error read 0.29 to 0.58 of the estimate's
            spread. A panel whose transitions all fall in one group, one unit's or one declared
            cluster's, has no second group: it sums its transitions' scores within each period
            and reads the sums' dependence across the periods by cosines of them (``periods`` in
            :func:`~chc.dynamics_id.fit_causal_residual`, ADR 0075), and where they all start in
            one period as well, takes them as independent. One whose transitions all start in one
            period sums within groups alone. The certificate says which it was
            (:attr:`DecisionCertificate.error_clustered_by`, ``error_periods``,
            ``error_cosines``).
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
            short of the condition. Each bound's condition is held on its own, as the audit reads
            it. The regret bound then includes what holding cost, since it is still priced against
            the box alone.
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
            contain zero. A lever the log never moved is no candidate, and stays on what the log
            did (below). The steps, each with its planned cost and regret bound, are
            :attr:`Prescription.selection`; the certificate's regret bound stays priced against
            every lever's box, so it includes what leaving levers out cost. ``None`` plans with
            every lever, as before; a value at or above the number of levers selects them all and
            returns that same plan.
        adjustment: a :class:`~chc.graph.CausalGraph` to *derive* the adjustment set from, or a
            sequence of column names to *assert* it. Required, and deliberately so --- omitting it
            would default to adjusting for nothing, which is a causal claim, not an absence of one.
            A graph's set is derived for the levers' effect on every state, the target and each
            constrained column together, since one set reads the channel into all of them: where
            no set does, the effect is ``not_identified``, and the reason names the states no set
            identifies it on alone.
            An empty set a graph derives, for a randomised lever or one set by the states alone,
            is fitted as identified, with its errors and its tube; an asserted empty set is
            fitted as the observational regression, with neither.
        known: physics kept fixed and not estimated. ``None`` means nothing is known and the whole
            vector field is the fitted residual. The fit reads it at each transition's start, its
            period's steps from the panel's first period times ``dt``; the plan reads it from
            ``t = 0``, so a ``known`` that reads the time is planned on the clock of the panel's
            first period, not of the start.
        dt: the time step one panel period represents.
        gamma: the marginal sensitivity model's level the barrier is priced at (§40), passed to
            :func:`chc.plan.certify_safety` with its default gap, the fitted channel's own size,
            and each lever's largest action as ``u_max``, so ``gamma_star`` and the barrier's
            certified steps read the same in any units of the state or of each lever (ADR 0085,
            ADR 0091), and the levers' box, which may be asymmetric. It prices the plan, and changes
            it only under ``hold_constraints``. The radius is that factor times a share of the
            fitted channel, so at every level the identified set stays within the channel's own
            size of the estimate: ``gamma_star`` is ``inf`` where every step holds over that whole
            set, which says nothing of a channel the fit has wrong by more than its size. The
            channel's standard error does not enter it. Read so, ``gamma`` and ``gamma_star`` are
            a relative margin, not a level of confounding: the absolute radius of
            :class:`chc.plan.ChannelSensitivity` needs a least-squares Euler channel, and this fit
            is ``rk4`` by default.
        tolerance: the trajectory error above which the plan stops being certified. **Omitting it
            switches the tube off** rather than setting it to infinity: this library cannot know
            how much error a caller accepts, and a certificate with an infinite tolerance passes
            over the whole horizon while proving nothing. The tube carries two errors over the
            plan's steps, by the ``integrator``'s map, at the rate
            :attr:`DecisionCertificate.tube_rate` names: the channel's standard error, each lever's
            share of it times the largest distance that lever's box
            allows from the log's mean action, where the fit's drift absorbed the channel, as
            the root mean square of the Euclidean norm over the states of the error it moves the
            field by, which reads the same in any units of each lever (ADR 0087) and wherever its
            zero is (ADR 0091), with the drift's own in quadrature, both read at the state each
            step of the plan starts from, a standard error and not a bound; and the plant's own
            noise, the plan's one-step error on the log, read off the fit's defect with its lag-1
            correlation within units (ADR 0080, :attr:`DecisionCertificate.tube_noise`). The
            radius holds the two together at each step with probability 0.95 where the noise is
            Gaussian and AR(1) and the field's error is its standard error times Student's ``t``
            at the degrees of freedom its spread is read with (ADR 0104). Up to 0.16.0
            the tube carried the channel's error alone: on 600 one-unit logs shaped like the
            BOPTEST case's it read 0.051 K one step ahead, where the plant's error one step ahead
            reached 0.16 to 0.22 K at 95 %; it held the plant at the first step on 40 % of draws
            and trusted 6 to 13 steps on nine logs in ten. It now holds the plant at the first step
            on 97 % to 99 % of draws and trusts 1 or 2 steps, where the plant's error stays within
            the tolerance of 0.5 K at 95 % for 2 or 3. A tube whose rate, budget or noise comes out
            other than a finite number is not evaluated, and a warning says so; as is one on an
            asserted empty adjustment set, whose observational fit has no channel error, and one
            over a noisy log in which no unit has two consecutive transitions, which leaves the
            noise's persistence unread. The tube
            is one norm over the states, and ``tolerance`` one number in their units: states in
            units unlike each other mix in it, as the levers' once did. To give each state its own
            scale, log it in units of the error you accept on it, its target, its bounds and the
            prices with it, which the plan reads the same (ADR 0059), and pass ``tolerance=1``
            (ADR 0091).
        x0: the state to plan from, one finite value per state: the target's, then each
            constrained column's. Defaults to the mean over units of each unit's last observed
            state, the pooled "where we are now". :attr:`Prescription.start` records which, and
            for the panel's, over how many units, at which periods and with what spread.
        integrator: the one-step map the field is fitted to and planned by, ``"rk4"`` or
            ``"euler"``, passed to :func:`~chc.dynamics_id.fit_causal_residual`. It is ``"euler"``
            there by default, deliberately: the low-level fit has no idea what will consume it and
            keeps its contract, while here the field goes straight to
            :func:`chc.plan.causal_plan`, and the plan, its tube, its barrier's audit, its regret
            bound and every rollout the prescription carries step it by the map it was fitted to:
            RK4's under ``"rk4"``, ``x + dt f(x, u)`` under ``"euler"`` (ADR 0118). Fitting one map
            and planning by another is a bias the caller never asked for --- on the plant in
            :mod:`chc.research.mmm` an Euler fit planned by RK4 costs 28% of the control channel.
            ``"rk4"`` reads a log sampled from a plant in continuous time: its fit is the field
            whose RK4 step reproduces the log's one-step map. Choose ``"euler"`` when the panel is
            genuinely discrete-time (a weekly budget is not a sample of an ODE) and the one-step
            map *is* the model. Where the log keeps less than 0.27 of a mode from one step to the
            next, no field stepped by RK4 reproduces it, RK4's growth factor being never smaller,
            and the ``"rk4"`` fit raises.
        drivers: exogenous columns the plan can forecast but not move, each with its forecast over
            the horizon (:class:`Driver`). They are fitted into the drift jointly with the states
            (:func:`~chc.dynamics_id.fit_causal_residual`), and the plan is made against the
            forecast, so it acts ahead of a push it can see coming. The forecast moves the drift
            and never the channel, so ``gamma_star`` is priced with it in place. Two things are not
            priced: the forecast's own error, and the fitted gain's standard error, which the tube's
            model error leaves out as it leaves out the drift's. A forecast outside the range the
            panel logged is extrapolated by the fitted gain, and logged as a warning.
        neighbours: the units whose levers may move a unit's states beside its own: a mapping from
            a unit to its neighbours, or the name of a column grouping the units, each unit's
            neighbours the others in its group. The fit takes each unit's transitions to depend on
            its own levers alone; where they do not, as with drivers or customers moving between
            zones, its channel is the effect of moving one unit with its neighbours held, and a
            plan that moves them all gets another. Given, the fit's residual is regressed on the
            neighbours' mean levers within periods (:func:`~chc.dynamics_id.interference_check`,
            *experimental*): :attr:`Prescription.interference` holds the result, and a rejection
            at 5 % is logged as a warning and named in the report. It reports; it changes nothing
            else. Under ``clusters`` it names the clusters, not the units.
        clusters: the name of a column grouping the units, each unit in one group throughout:
            fit and plan on the groups instead of the units, each group's row in a period the mean
            of its units' rows logged then (ADR 0113). A spillover between the units of one group
            is then inside its rows, so the channel is the effect of moving the group's units
            together, the effect a plan that moves them all gets. A spillover between groups is
            not: check it with ``neighbours``. The fit has a row per group and period, so its
            errors read the groups' spread, not the units'. The panel's declared ``cluster`` is
            kept where it holds one value over each group's rows, and dropped otherwise.
            :attr:`Prescription.provenance` is the grouped panel's.
        nuisance_knots: experimental; the cross-fitted nuisances' spline, passed to
            :func:`~chc.dynamics_id.fit_causal_residual`: ``"auto"`` adds a natural cubic spline
            in each covariate at ``ceil(N^(1/3))`` of its quantiles beside the polynomial, an
            integer that many knots, ``None``, the default, none. With ``None`` the fit warns
            where the spline would move the channel by more than 3 of its errors; ``"auto"`` is
            then the fit that warning names. :attr:`RunProvenance.nuisance_knots` records it.

    Returns:
        A :class:`Prescription`. Read :attr:`DecisionCertificate.identification` before
        :attr:`Prescription.schedule`, which raises when the effect is not identified.
        :attr:`Prescription.run` records what the run used (:class:`RunProvenance`).

    Raises:
        DecisionError: the decision is mis-specified --- no lever, a lever named twice or also
            the target or a constraint, a column constrained twice, a target schedule whose
            length is not ``horizon``, constraints to hold with none given, ``max_levers`` below
            one or with a lever whose box excludes zero, a driver that
            is also a lever or a state or is named twice, a forecast that is not ``horizon + 1``
            finite levels, a budget that does not weigh one spend per lever, starts its periods
            anywhere but the plan's first step, or allows less than the levers' boxes spend at
            the least, a ``dt`` that is not a finite positive step, an ``integrator`` that is
            neither ``"rk4"`` nor ``"euler"``, a ``tolerance`` that is
            negative or nan, a ``gamma`` below 1 or not finite, which is not a sensitivity level,
            a panel with no consecutive pair of periods to fit a transition on, a ``cap_per_step``
            on a lever that follows its logged rule or a budget that prices one,
            ``max_levers`` where the log kept a combination of the levers away from zero, an
            ``x0`` that is not one finite value per state, an asserted adjustment set that names
            a lever, or a covariate or a driver whose column
            would be read in a second role: one named ``u`` or ``x_next``, or ``x`` other than as
            the one state, under which the transitions hold the levers, the states a period on
            and the states, or one named for a driver's level a period on, ``f"{driver}_next"``;
            ``neighbours`` that name a unit the panel does not have, or a unit, under ``clusters``
            a cluster, that ``clusters`` or a ``neighbours`` column puts in two groups.
        KeyError: a lever, target, constraint, driver, asserted covariate, ``clusters`` or a
            ``neighbours`` column names a column the panel does not have. The message lists the
            panel's columns. Also a graph that has no node for a lever, the target or a
            constrained column.
        PanelError: a column the fit reads as numbers --- a lever's, a state's, a covariate's or
            a driver's --- holds text, dates, times, durations, complex numbers or any other value
            that is not a real number, or a number finite in its own type that is an infinity in
            float64, as :meth:`chc.panel.Panel.wide` refuses it. The message names the column, the
            value, its unit and its time.

    A direction of the channel the log never moves apart from what the states and the covariates
    predict (:attr:`~chc.dynamics_id.CausalDynamicsFit.unmoved`) is not identified on this log, and
    the plan keeps to what the log did along it (ADR 0054). A lever whose whole channel is so is
    held at its mean logged level, clipped to its box, where the log kept it at one level, and
    follows the log's least-squares rule of the state inside the plan's field where the log set it
    from the state alone; :attr:`DecisionCertificate.unmoved_levers` and
    :attr:`~DecisionCertificate.rule_levers` name them, and the schedule's column carries the rule
    read along the predicted path. Among the other levers, a combination the log kept at one level
    is held there by an equality row at every step (:attr:`~DecisionCertificate.relations`). A lever
    or a combination the log set from anything else gives no plan, and neither does a log that
    moves no lever: the identification is then ``not_identified``, its reason naming the levers and
    the columns they were set from. The plan is checked exactly: the fits the log cannot tell apart
    differ along directions in which the field is linear, and
    :attr:`~DecisionCertificate.first_loaded_step` is the first step at one of whose stages any
    of them moves the field, where the trustworthy prefix ends.
    :attr:`~DecisionCertificate.estimability` says which case holds.

    Each decision point emits one ``logging`` record on ``chc.decision``, keyed by ``chc_event``
    (see :data:`_log`). Nothing is configured here; a caller that wants them calls
    ``logging.basicConfig`` itself. An unidentified effect, a lever the log never moved, one on its
    logged rule, a combination kept at its logged level, a plan the log does not determine, a run
    in single precision, a forecast outside the logged range, a planned action outside its lever's
    logged range and levers that read more than the record says come through at ``WARNING``.

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
    if not 0.0 < dt < math.inf:
        raise DecisionError(f"dt={dt} is not a step, which is finite and positive")
    if integrator not in ("rk4", "euler"):
        raise DecisionError(
            f"integrator={integrator!r} is neither 'rk4', for a log sampled from a plant in "
            "continuous time, nor 'euler', for a discrete-time panel whose one-step map is the "
            "model"
        )
    if tolerance is not None and not tolerance >= 0.0:
        raise DecisionError(
            f"tolerance={tolerance} is not a radius, which is never negative or nan; leave it "
            "None to evaluate no tube"
        )
    if not 1.0 <= gamma < math.inf:
        raise DecisionError(
            f"gamma={gamma} is not a sensitivity level, which is finite and at least 1; a plan "
            "that keeps its barrier at every level reads gamma* = inf"
        )
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
    twice = sorted({name for name in lever_names if lever_names.count(name) > 1})
    if twice:
        raise DecisionError(f"levers named more than once: {twice}; give each column one Lever")
    moved = sorted(set(lever_names) & set(states))
    if moved:
        raise DecisionError(
            f"columns {moved} are named as levers and as the target or a constraint; a lever is "
            "the action, which its box bounds, and the target and the constraints are states the "
            "action moves"
        )
    driver_names = _check_drivers(drivers, horizon=horizon, taken=(*states, *lever_names))
    # read as floats here: an integer start failed inside the fit's linearisation
    given = None if x0 is None else np.asarray(_real_numbers(x0, "x0", DecisionError), dtype=float)
    if given is not None:
        if given.shape != (len(states),):
            raise DecisionError(
                f"x0 has shape {given.shape}; the plan starts from one value per state, "
                f"{len(states)} for {list(states)}"
            )
        if not np.all(np.isfinite(given)):
            raise DecisionError(f"x0 is {given.tolist()}, and a start must be finite")
    for name in (*states, *lever_names, *driver_names, *_named(clusters), *_named(neighbours)):
        if name not in panel.columns:
            raise KeyError(
                f"column {name!r} is not in the panel; columns are {sorted(panel.names)}"
            )
    if clusters is not None:
        numbers = (*states, *lever_names, *driver_names)
        panel = _cluster_panel(panel, clusters, numbers, _named(neighbours))

    # the run's precision; the panel's flag was read when the panel was built, and need not match
    x64 = _x64_enabled()
    if not x64:
        _log.warning(
            "identifying in single precision; export JAX_ENABLE_X64=1 if the fit is load-bearing",
            extra={"chc_event": "precision", "x64": False, "rows": panel.provenance.n_rows},
        )

    resolved = _resolve_adjustment(adjustment, panel=panel, states=states, levers=lever_names)
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
    data, labels = _transitions(
        panel,
        states=states,
        levers=lever_names,
        adjust_for=resolved.covariates,
        drivers=driver_names,
    )
    groups, periods, units = (int(np.unique(column).size) for column in labels.T)
    # one group, one unit's or one declared cluster's, has no second to read a spread between: its
    # transitions are summed within each period and read across the periods instead (ADR 0075)
    serial = groups == 1 and periods > 1
    clustered_by = (panel.cluster or panel.unit) if groups > 1 or (serial and units > 1) else None
    two_way = groups > 1 and periods > 1
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
        clusters=None if groups == 1 else labels[:, :2] if two_way else labels[:, 0],
        periods=labels[:, 1] if serial else None,
        nuisance_degree=_NUISANCE_DEGREE,
        nuisance_knots=nuisance_knots,
        ridge=_RIDGE,
        times=labels[:, 1] * dt,
        # a graph that finds the empty set enough proves the levers unconfounded; an asserted
        # empty set stays the observational fit it always was
        unconfounded=isinstance(adjustment, CausalGraph) and resolved.status == "identified",
    )
    _log.info(
        "control channel fitted",
        extra={
            "chc_event": "fit",
            "method": fit.method,
            "identified": fit.identified,
            "channel_error": fit.channel_error,
            "clustered_by": clustered_by,
            "clusters": groups if clustered_by is not None else None,
            "cosines": fit.error_cosines,
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
    (device,) = data["x"].devices()
    run = RunProvenance(
        x64=x64,
        backend=device.platform,
        device_kind=device.device_kind,
        fit_dtype=str(data["x"].dtype),
        solve_dtype=None,
        solver_status=None,
        solver_iterations=None,
        seed=int(seed),
        integrator=fit.integrator,
        versions=_versions(),
        folds=int(fit.folds),
        degree=fit.residual.degree,
        channel_degree=fit.residual.channel_degree,
        nuisance_degree=_NUISANCE_DEGREE,
        ridge=_RIDGE,
        steps=_PLAN_STEPS,
        tolerance=None if tolerance is None else float(tolerance),
        hold_constraints=bool(hold_constraints),
        max_levers=None if max_levers is None else int(max_levers),
        clusters=clusters,
        nuisance_knots=nuisance_knots,
    )
    persistence = (
        None
        if fit.moment_residual is None
        else persistence_check(fit, units=labels[:, 2], periods=labels[:, 1])
    )
    interference = (
        None
        if neighbours is None or fit.moment_residual is None
        else interference_check(
            fit, _exposure(panel, lever_names, neighbours, labels), labels[:, 0], labels[:, 1]
        )
    )
    if interference is not None and interference.p_value <= _INTERFERENCE_ALPHA:
        _log.warning(
            "the units' transitions move with their neighbours' levers",
            extra={
                "chc_event": "interference",
                "p_value": interference.p_value,
                "coefficient": interference.coefficient.tolist(),
                "groups": interference.groups,
            },
        )
    identification: IdentificationStatus = (
        "not_identified"
        if resolved.status == "not_identified"
        else ("identified" if isinstance(adjustment, CausalGraph) else "asserted")
    )
    unmoved = _unmoved_actions(fit)
    unmoved_levers = tuple(lever_names[index] for index in unmoved)
    per_state = int(fit.unmoved.shape[1]) // n_states if fit.unmoved is not None else None
    rank = None if per_state is None else int(fit.residual.channel[0].size) - per_state
    covariates, _, _, driver_read = _nuisance_inputs(
        data, resolved.covariates, driver_names, integrator
    )
    covariate_names = (
        *states,
        *resolved.covariates,
        *(name for name in driver_names if name not in resolved.covariates),
        *(f"{name} (next period)" for name in driver_names),
    )
    kept = _keep_to_log(fit, data, unmoved, covariates, covariate_names, lever_names, n_states)
    abort = "no schedule: no observed set identifies the effect"
    estimability: Estimability | None = None
    if identification != "not_identified" and len(unmoved) == n_levers:
        identification, abort = "not_identified", "no schedule: the log never moves a lever"
        estimability = "not_estimable"
        resolved = AdjustmentSet(
            resolved.covariates,
            "not_identified",
            f"{resolved.reason}; but the log never moves {list(unmoved_levers)} apart from what "
            "the states and the covariates predict, so no transition shows what moving them does",
        )
    elif identification != "not_identified" and kept.refusal is not None:
        identification, abort = "not_identified", "no schedule: no plan keeps to what the log did"
        estimability = "not_estimable"
        resolved = AdjustmentSet(
            resolved.covariates, "not_identified", f"{resolved.reason}; but {kept.refusal}"
        )

    if given is None:
        start, start_state = _panel_start(panel, states)
    else:
        start = jnp.asarray(given)
        start_state = StartState("given", states, tuple(given.tolist()))
    alone = _alone(fit, data, start)
    if identification == "not_identified":
        _log.warning(abort, extra={"chc_event": "abort", "reason": resolved.reason})
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
                unmoved_levers=unmoved_levers,
                estimability=estimability,
                identification_rank=rank,
                unmoved_directions=per_state,
            ),
            model_fit=fit,
            provenance=panel.provenance,
            drivers=tuple(drivers),
            logger_check=logger_check,
            interference=interference,
            budgets=tuple(budgets),
            start=start_state,
            run=run,
            _columns=columns,
            _start=start,
            _alone=alone,
        )

    if not fit.identified:
        resolved = AdjustmentSet(
            resolved.covariates,
            resolved.status,
            resolved.reason
            + "; the estimator was handed no covariate and no instrument, so its channel is the "
            "observational fit, which is the interventional one only if the lever is unconfounded",
        )

    rule_levers = tuple(lever_names[index] for index in kept.rules)
    capped = [
        name
        for index, name in zip(kept.rules, rule_levers, strict=True)
        if levers[index].cap_per_step is not None
    ]
    if capped:
        raise DecisionError(
            f"{capped} follow the rule of the state the log set them by, which moves them as the "
            "state moves, so no cap on their steps can hold"
        )
    priced = [
        name
        for index, name in zip(kept.rules, rule_levers, strict=True)
        if any(budget.weights[index] != 0.0 for budget in budgets)
    ]
    if priced:
        raise DecisionError(
            f"a budget prices {priced}, which follow the rule of the state the log set them by: "
            "their spend is set by the path, and a budget's rows hold the plan's own actions"
        )
    if max_levers is not None and any(relation.level != 0.0 for relation in kept.relations):
        raise DecisionError(
            "the log kept "
            + "; ".join(_show_relation(relation, lever_names) for relation in kept.relations)
            + ", and max_levers holds an unselected lever at zero, which leaves that level"
        )

    model: Dynamics = HybridDynamics(known=base, residual=fit.residual)
    driven: DrivenDynamics | None = None
    if drivers and fit.driver_gain is not None:
        _warn_outside_logged_range(panel, drivers)
        forecast = jnp.stack([jnp.asarray(driver.forecast, dtype=float) for driver in drivers], 1)
        model = driven = DrivenDynamics(model, fit.driver_gain, forecast, dt)
    if integrator == "euler":
        # every step of the plan, its audits and its rollouts is the map the field was fitted to
        model = _EulerStep(model)
    rule: _Rule | None = None
    if kept.logged is not None and kept.rules:
        rule = _Rule(
            centre=kept.logged.centre,
            shift=kept.logged.shift,
            factor=kept.logged.factor,
            coefficients=kept.logged.rule[:, jnp.array(kept.rules)],
            lo=jnp.array([levers[index].lo for index in kept.rules]),
            hi=jnp.array([levers[index].hi for index in kept.rules]),
            levers=kept.rules,
            degree=_NUISANCE_DEGREE,
        )
        model = _Ruled(model, rule)
        _log.warning(
            "levers the log set from the state alone follow that rule",
            extra={"chc_event": "rule", "levers": list(rule_levers)},
        )
    # A lever the log never moved is held at its mean logged level: the one level the fit has seen
    # its push at, which the drift has absorbed, where the log kept it there. One the log set from
    # the state follows that rule in the field and in the price, and its column, which neither
    # reads, is held there too.
    held_at = jnp.array(unmoved, dtype=int)
    levels = jnp.array(
        [jnp.clip(jnp.mean(data["u"][:, i]), levers[i].lo, levers[i].hi) for i in unmoved]
    )
    if unmoved and len(kept.rules) < len(unmoved):
        _log.warning(
            "levers the log never moved are held at their logged level",
            extra={
                "chc_event": "unmoved",
                "levers": [name for name in unmoved_levers if name not in rule_levers],
                "levels": [
                    float(level)
                    for index, level in zip(unmoved, np.asarray(levels), strict=True)
                    if index not in kept.rules
                ],
            },
        )
    if kept.relations:
        _log.warning(
            "combinations of the levers the log kept at one level are kept there",
            extra={
                "chc_event": "relation",
                "relations": [_show_relation(relation, lever_names) for relation in kept.relations],
            },
        )

    def pin(lo: Array, hi: Array) -> tuple[Array, Array]:
        return lo.at[held_at].set(levels), hi.at[held_at].set(levels)

    u_lo, u_hi = pin(
        jnp.array([lever.lo for lever in levers]), jnp.array([lever.hi for lever in levers])
    )
    # each lever's largest action, in its units: the barrier's audit reads each lever in units of
    # its own (ADR 0091)
    reach = np.asarray(jnp.maximum(jnp.abs(u_lo), jnp.abs(u_hi)), dtype=np.float64)
    caps = [math.inf if lever.cap_per_step is None else lever.cap_per_step for lever in levers]
    rate = LinearConstraint.rate_limit(horizon, caps)
    relation_rows = tuple(
        _relation_rows(relation, horizon, u_lo, u_hi) for relation in kept.relations
    )
    # a prescription is the first step of a loop at t = 0 with nothing spent, and reads a budget so
    spend_rows = tuple(
        _period_rows(
            budget, 0.0, 0.0, dt=dt, horizon=horizon, u_lo=u_lo, u_hi=u_hi, levers=n_levers
        )
        for budget in budgets
    )

    margins = _margins(states, constraints)
    held = (
        BarrierConstraint(
            margins,
            gamma=gamma,
            u_max=tuple(reach.tolist()),
            u_box=(tuple(np.asarray(u_lo).tolist()), tuple(np.asarray(u_hi).tolist())),
        )
        if hold_constraints
        else None
    )

    started = time.perf_counter()
    planning_cost = _cost(states, levers, target, horizon, rule)
    lipschitz, tube_rate = _rate(model, start, u_lo, u_hi)
    # On a field affine in the state RK4 steps the gap by the matrix of the slope at each planned
    # action, exactly; the norm bound over the box stays where the slope moves with the state
    # (ADR 0096).
    slopes = _slopes(model, start) if tube_rate == "global" else None
    # The drift absorbed the channel at the log's mean action, so the channel's error moves the
    # field by the action's distance from it, not from the lever's zero (ADR 0091): each lever's
    # largest distance from it the box allows, in its units. A lever held at its mean logged
    # level is held at that action, read the way ``levels`` reads it, and adds no error.
    centre = np.array([float(jnp.mean(data["u"][:, i])) for i in range(n_levers)])
    distance = np.maximum(
        np.abs(np.asarray(u_lo, dtype=np.float64) - centre),
        np.abs(np.asarray(u_hi, dtype=np.float64) - centre),
    )
    model_error = 0.0 if tolerance is None else _field_error(fit, distance)
    # the drivers as the fit's drift reads them over each of the plan's steps: their forecast's
    # mean over the step, as the ``rk4`` fit reads a transition's, or its level at the step's
    # start, all the Euler map reads (``_nuisance_inputs``)
    forecasts = (
        np.zeros((horizon + 1, 0))
        if driven is None
        else np.stack([np.asarray(driver.forecast, dtype=np.float64) for driver in drivers], 1)
    )
    driver_reads = (
        forecasts[:-1] if integrator == "euler" else 0.5 * (forecasts[:-1] + forecasts[1:])
    )
    # the field's error is a standard error whose spread the log estimates: the tube reads it at
    # its studentised quantile, beside the noise and independent of it
    field_scale = None if tolerance is None else _field_scale(fit)
    # with no channel error nothing bounds the field the plan rolls out, and the noise alone would
    # switch on a tube that trusts it
    if tolerance is not None and fit.channel_error is None:
        _log.warning(
            "the error tube is not evaluated: the fit is the observational one, with no channel "
            "error; derive the adjustment set from a graph, which passes on an empty set it proves "
            "enough",
            extra={"chc_event": "tube", "method": fit.method},
        )
    step_noise, step_carry, step_persistence = (
        _step_noise(fit, dt, labels)
        if tolerance is not None and fit.channel_error is not None
        else (np.zeros((n_states, n_states)), np.zeros((n_states, n_states)), 0.0)
    )
    if tolerance is not None and not (
        math.isfinite(lipschitz)
        and math.isfinite(model_error)
        and bool(np.all(np.isfinite(step_noise)))
        and math.isfinite(step_persistence)
        and bool(np.all(np.isfinite(step_carry)))
    ):
        # a nan radius once read as within every tolerance; no tube is the answer that holds
        _log.warning(
            "the error tube is not evaluated: its rate, its budget or its noise is not a finite "
            "number, or no unit has two consecutive transitions to read the noise's persistence",
            extra={
                "chc_event": "tube",
                "rate": lipschitz,
                "model_error": model_error,
                "noise": step_noise.tolist(),
                "noise_persistence": step_persistence,
            },
        )
        lipschitz, model_error = 0.0, 0.0
        step_noise, step_carry, step_persistence = (
            np.zeros_like(step_noise),
            np.zeros_like(step_noise),
            0.0,
        )
    step_read = _noise_read(fit, dt, labels) if step_noise.any() else None

    if (
        tolerance is not None
        and persistence is not None
        and math.isfinite(persistence.p_value)
        and persistence.p_value <= _PERSISTENCE_ALPHA
    ):
        _log.warning(
            "the error tube's noise may not be the plant's: the channel's residual persists "
            "within units (p = %.3g), so a disturbance that persists may be read in part as the "
            "drift, or the drift's form may miss the plant's; the tube's 0.95 a step is not "
            "assured",
            persistence.p_value,
            extra={
                "chc_event": "tube",
                "noise_persistence": persistence.correlation,
                "noise_persistence_p": persistence.p_value,
            },
        )

    def solve(lo: Array, hi: Array) -> CausalPlan:
        lo, hi = pin(lo, hi)
        # Held, the solve audits at the whole box's largest actions, which ``held`` carries, not at
        # its own box's, which a candidate of the greedy selection narrows: greedy ranks candidates
        # on, and the certificate reports, that one audit, so the two cannot disagree.
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
            noise=step_noise,
            noise_persistence=step_carry,
            tolerance=float("inf") if tolerance is None else tolerance,
            steps=_PLAN_STEPS,
            # the budgets' rows last: :meth:`Prescription.budget_prices` reads them there
            constraints=(rate, *relation_rows, *spend_rows),
            barrier=held,
        )
        # the field's error read where the plan goes, at the state each of its steps starts from
        return _carried_along(
            solved,
            np.asarray(lipschitz, dtype=np.float64)
            if slopes is None
            else slopes(np.asarray(solved.actions, dtype=np.float64)),
            _field_errors(
                fit, distance, np.asarray(solved.trajectory, dtype=np.float64)[:-1], driver_reads
            )
            if model_error > 0.0
            else model_error,
            step_noise,
            step_carry,
            dt,
            float("inf") if tolerance is None else tolerance,
            field_scale,
            step_read,
        )

    def regret(solved: CausalPlan) -> PlanRegretBound:
        return plan_regret_bound(solved, model, start, planning_cost, dt, u_lo, u_hi, probes=4)

    selection: LeverSelection | None = None
    if max_levers is None:
        plan = solve(u_lo, u_hi)
    else:
        idle = total_cost(model, start, jnp.zeros((horizon, n_levers)), dt, planning_cost)
        plan, steps = _select_levers(levers, max_levers, solve, regret, held=frozenset(unmoved))
        selection = LeverSelection(idle_cost=float(idle), steps=steps)
    run = replace(
        run,
        solve_dtype=str(plan.actions.dtype),
        solver_status=plan.solver_status,
        solver_iterations=plan.solver_iterations,
    )

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

    _warn_actions_outside_logged_range(
        panel, lever_names, np.asarray(plan.actions, dtype=np.float64), skip=rule_levers
    )
    if held is not None:
        safety = plan.safety  # solve() already audited it at the whole box's largest actions
    elif margins:
        safety = _audit(plan, model, margins, dt, 1.0, gamma, 1.0, reach, (u_lo, u_hi))
    else:
        safety = None
    gap = regret(plan)
    directions, response = _absorbed(
        fit,
        data["x"],
        data["u"],
        jnp.concatenate(
            [
                jax.vmap(control_affine_features, in_axes=(0, None))(
                    data["x"], fit.residual.degree
                ),
                driver_read,
            ],
            axis=1,
        ),
    )
    first_loaded = _first_loaded(
        model, plan, dt, fit, directions, response, None if driven is None else driven.drivers
    )
    estimability = (
        "estimable"
        if not per_state
        else ("held_to_log" if first_loaded is None else "not_estimable")
    )
    (_log.info if estimability != "not_estimable" else _log.warning)(
        "what the log determines of the plan's path",
        extra={
            "chc_event": "estimability",
            "estimability": estimability,
            "identification_rank": rank,
            "unmoved_directions": per_state,
            "first_loaded_step": first_loaded,
        },
    )
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
        regret_bound=gap.bound,
        gamma=None if safety is None else float(gamma),
        regret_status=gap.status,
        unmoved_levers=unmoved_levers,
        tube_rate=None if plan.certified_horizon is None else tube_rate,
        tube_noise=None
        if plan.certified_horizon is None
        else math.sqrt(max(float(np.linalg.eigvalsh(step_noise)[-1]), 0.0)),
        tube_persistence=None if plan.certified_horizon is None else step_persistence,
        tube_autoregression=None
        if plan.certified_horizon is None
        else tuple(tuple(float(value) for value in row) for row in step_carry),
        tube_mode=None
        if plan.certified_horizon is None
        else "bound"
        if slopes is None
        else "exact",
        tube_state_noise=None
        if plan.certified_horizon is None or slopes is None
        else _state_noise(
            slopes(np.asarray(plan.actions, dtype=np.float64)),
            step_noise,
            step_carry,
            dt,
            horizon,
            step_read,
            integrator=integrator,
        ),
        error_clustered_by=clustered_by,
        error_clusters=groups if clustered_by is not None else None,
        error_periods=periods if two_way else None,
        error_cosines=fit.error_cosines,
        noise_persistence=None if persistence is None else _finite(persistence.correlation),
        noise_persistence_p=None if persistence is None else _finite(persistence.p_value),
        estimability=estimability,
        identification_rank=rank,
        unmoved_directions=per_state,
        first_loaded_step=first_loaded,
        relations=kept.relations,
        rule_levers=rule_levers,
        stationarity=plan.stationarity,
        stationarity_tolerance=plan.stationarity_tolerance,
        constraint_violation=plan.constraint_violation,
        violation_tolerance=plan.violation_tolerance,
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
            "regret_status": certificate.regret_status,
            "unmoved_levers": list(certificate.unmoved_levers),
            "tube_rate": certificate.tube_rate,
            "tube_mode": certificate.tube_mode,
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
        interference=interference,
        budgets=tuple(budgets),
        start=start_state,
        run=run,
        _columns=columns,
        _start=start,
        _alone=alone,
        _budget_rows=tuple(rows.matrix.shape[0] for rows in spend_rows),
        _magnitudes=_ruled_schedule(model, plan),
    )


# ---- wiring ---------------------------------------------------------------------------------


def _select_levers(
    levers: Sequence[Lever],
    max_levers: int,
    solve: Callable[[Array, Array], CausalPlan],
    price: Callable[[CausalPlan], PlanRegretBound],
    held: frozenset[int] = frozenset(),
) -> tuple[CausalPlan, tuple[SelectionStep, ...]]:
    """Greedy forward selection: add the lever whose plan ranks first, ``max_levers`` times at most.

    Each candidate is a cold solve on its own box, the levers not in it pinned to ``[0, 0]``, so its
    plan is the one :func:`prescribe` would make for that set alone, whatever the path to it; once
    every lever is in, the plan is the one ``max_levers=None`` makes. Ties go to the lever listed
    first. The levers in ``held`` are never candidates: ``solve`` holds them where they are.
    """
    chosen: list[int] = []
    plans: list[CausalPlan] = []
    steps: list[SelectionStep] = []
    for _ in range(min(max_levers, len(levers) - len(held))):
        started = time.perf_counter()
        candidates: dict[int, CausalPlan] = {}
        for index in range(len(levers)):
            if index in chosen or index in held:
                continue
            keep = {*chosen, index}
            lo = jnp.array([lever.lo if i in keep else 0.0 for i, lever in enumerate(levers)])
            hi = jnp.array([lever.hi if i in keep else 0.0 for i, lever in enumerate(levers)])
            candidates[index] = solve(lo, hi)
        best = min(candidates, key=lambda index: _rank(candidates[index]))
        chosen.append(best)
        plans.append(candidates[best])
        gap = price(plans[-1])
        steps.append(SelectionStep(levers[best].name, plans[-1].task_cost, gap.bound, gap.status))
        _log.info(
            "lever selected",
            extra={
                "chc_event": "selection",
                "step": len(steps),
                "lever": steps[-1].lever,
                "task_cost": steps[-1].task_cost,
                "regret_bound": steps[-1].regret_bound,
                "regret_status": steps[-1].regret_status,
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


@dataclass(frozen=True)
class _Kept:
    """What a plan keeps to of what the log did along the directions it never moved (ADR 0054)."""

    rules: tuple[int, ...] = ()  # whole levers the log set from the state alone
    relations: tuple[LeverRelation, ...] = ()  # combinations of the rest it kept at one level
    refusal: str | None = None  # why no plan keeps to what the log did
    logged: _LoggedRelations | None = None


def _keep_to_log(
    fit: CausalDynamicsFit,
    data: dict[str, Array],
    unmoved: tuple[int, ...],
    covariates: Array,
    covariate_names: Sequence[str],
    lever_names: Sequence[str],
    n_states: int,
) -> _Kept:
    """How a plan keeps to the log along the directions it never moved, or why none can.

    A lever whose whole channel the log never moved is held at its level where the log kept it
    there, and follows its rule where the log set it from the state alone. One the log set from a
    column outside the state has a rule no plan can read: it is refused. Among the other levers, a
    combination the log kept at one level is kept there by an equality row; one it set from
    anything else is refused, since no row of the plan's actions holds it."""
    if fit.unmoved is None or fit.unmoved.shape[1] == 0:
        return _Kept()
    x, u = data["x"], data["u"]
    logged = _logged_relations(u, x, covariates, _NUISANCE_DEGREE)
    precision = float(np.sqrt(np.finfo(np.asarray(u).dtype).eps))
    identity = np.eye(len(lever_names))
    rules = tuple(
        index
        for index in unmoved
        if not _within(logged.constant, identity[index], precision)
        and _within(logged.state, identity[index], precision)
    )
    outside = [index for index in unmoved if not _within(logged.state, identity[index], precision)]
    if outside:
        read = "; ".join(
            f"`{lever_names[index]}` from "
            + ", ".join(
                f"`{name}`"
                for name in _read_from(u[:, index], covariates, covariate_names, n_states)
            )
            for index in outside
        )
        return _Kept(
            refusal=f"the log set {read}, outside the plan's state, so no plan keeps to what "
            "the log did: leave them out of the levers, where they stay on that rule, or log them "
            "moving apart from what they read",
            logged=logged,
        )
    free = [index for index in range(len(lever_names)) if index not in unmoved]
    if not free:  # every lever is held or on its rule: no combination of the rest is left to keep
        return _Kept(rules=rules, logged=logged)
    among = _logged_relations(u[:, jnp.array(free)], x, covariates, _NUISANCE_DEGREE)

    def named(inner: np.ndarray, outer: np.ndarray) -> str:
        left = outer - inner @ (inner.T @ outer)
        column = left[:, int(np.argmax(np.linalg.norm(left, axis=0)))] / among.size
        column = column if column[np.argmax(np.abs(column))] > 0.0 else -column
        weights = np.zeros(len(lever_names))
        weights[free] = column / np.linalg.norm(column)
        return _show_relation(LeverRelation(tuple(weights), 0.0), lever_names).removesuffix(" = 0")

    if among.covariates.shape[1] > among.state.shape[1]:
        combination = named(among.state, among.covariates)
        return _Kept(
            refusal=f"the log set {combination} from columns outside the plan's state, which no "
            "plan reads",
            logged=logged,
        )
    if among.state.shape[1] > among.constant.shape[1]:
        combination = named(among.constant, among.state)
        return _Kept(
            refusal=f"the log set {combination} from the state, which no row of the plan's "
            "actions holds",
            logged=logged,
        )
    spread = np.sqrt(np.mean(np.asarray(u[:, jnp.array(free)], dtype=np.float64) ** 2, axis=0))
    relations = []
    for column in among.constant.T:
        column = column / among.size  # the weights on the levers as logged
        column = column / np.linalg.norm(column)
        column = column if column[np.argmax(np.abs(column))] > 0.0 else -column
        level = float(column @ among.means)
        if abs(level) <= precision * float(np.abs(column) @ spread):
            level = 0.0  # a combination the log kept at zero, read through rounding
        weights = np.zeros(len(lever_names))
        weights[free] = column
        relations.append(LeverRelation(tuple(float(weight) for weight in weights), level))
    return _Kept(rules=rules, relations=tuple(relations), logged=logged)


def _within(basis: np.ndarray, vector: np.ndarray, precision: float) -> bool:
    """Whether ``vector`` lies in the span of ``basis``'s orthonormal columns."""
    return float(np.linalg.norm(vector - basis @ (basis.T @ vector))) <= precision


def _read_from(
    action: Array, covariates: Array, names: Sequence[str], n_states: int
) -> tuple[str, ...]:
    """The columns outside the state an action's logged rule reads: each one without which the
    others no longer predict it to the precision :func:`chc.dynamics_id._logged_relations` reads,
    or every one, where none is needed alone."""
    precision = float(jnp.sqrt(jnp.finfo(action.dtype).eps))
    size = float(jnp.linalg.norm(action))
    needed = []
    for column in range(n_states, covariates.shape[1]):
        rest = jnp.delete(covariates, column, axis=1)
        features = _polynomial_features(_units.standardised(rest), _NUISANCE_DEGREE)
        left = action - features @ jnp.linalg.lstsq(features, action)[0]
        if not float(jnp.linalg.norm(left)) <= precision * size:
            needed.append(names[column])
    return tuple(needed) or tuple(names[n_states:])


def _relation_rows(
    relation: LeverRelation, horizon: int, u_lo: Array, u_hi: Array
) -> LinearConstraint:
    """``relation`` held by an equality row at every step: at its level, or at the nearest level
    the boxes reach where they do not reach it, which leaves the log's relation, and the
    estimability test reads that."""
    weights = np.asarray(relation.weights)
    lo, hi = np.asarray(u_lo, dtype=np.float64), np.asarray(u_hi, dtype=np.float64)
    moved = weights != 0.0  # a free lever's unbounded side must not read as 0 * inf
    low = float(np.minimum(weights * lo, weights * hi)[moved].sum())
    high = float(np.maximum(weights * lo, weights * hi)[moved].sum())
    level = min(max(relation.level, low), high)
    levels = np.full(horizon, level)
    return LinearConstraint(np.kron(np.eye(horizon), weights), levels, levels)


def _alone(fit: CausalDynamicsFit, data: dict[str, Array], start: Array) -> tuple[bool, ...]:
    """Per lever, whether the log identifies its own entry of the channel on the target's row at
    ``start``: whether that entry, a linear functional of the channel's coefficients, is
    orthogonal to every direction the log never moved (:attr:`CausalDynamicsFit.unmoved`).

    Read with each coefficient scaled by its column of the channel's design on the log's raw
    actions, as the fit splits the directions, so the test reads the same in any units. In raw
    units a lever logged in 1e9 of its units shares a direction with another only 2e-9 of the way
    along it, under the square root of the precision."""
    states, actions, features = fit.residual.channel.shape
    if fit.unmoved is None or fit.unmoved.shape[1] == 0:
        return (True,) * actions
    # every state's channel is moved along the same directions, so the target's block holds them
    width = actions * features
    directions = fit.unmoved[:width, : fit.unmoved.shape[1] // states]
    size = _units.norms(_channel_design(data["u"], data["x"], fit.residual.channel_degree))
    basis = jnp.linalg.qr(directions * size[:, None])[0]
    phi = control_affine_features(start, fit.residual.channel_degree)
    precision = float(jnp.sqrt(jnp.finfo(basis.dtype).eps))
    alone = []
    for action in range(actions):
        functional = jnp.zeros((actions, features)).at[action].set(phi).ravel() / size
        norm = float(jnp.linalg.norm(functional))
        leaning = float(jnp.linalg.norm(basis.T @ functional))
        alone.append(leaning <= precision * norm)
    return tuple(alone)


def _ruled_schedule(model: Dynamics, plan: CausalPlan) -> Array | None:
    """The plan's actions with each ruled lever read off its rule along the predicted path, a
    step at a time as :meth:`PrescribedPolicy.actions_at` reads it; None where no lever follows a
    rule."""
    if not isinstance(model, _Ruled):
        return None
    path, actions = np.asarray(plan.trajectory[:-1]), np.asarray(plan.actions)
    return jnp.asarray(
        np.stack([_act(model.rule, x, u) for x, u in zip(path, actions, strict=True)])
    )


def _act(rule: _Rule | None, x: NDArray[Any], u: NDArray[Any]) -> NDArray[np.float64]:
    """The action taken at ``x`` for the schedule's ``u``: ``u`` with each ruled lever on its
    rule, read in the floats the rule holds."""
    if rule is None:
        return np.array(u, dtype=np.float64)
    dtype = rule.centre.dtype
    action = rule.read(jnp.asarray(x, dtype=dtype), jnp.asarray(u, dtype=dtype))
    return np.asarray(action, dtype=np.float64)


def _first_loaded(
    model: Dynamics,
    plan: CausalPlan,
    dt: float,
    fit: CausalDynamicsFit,
    directions: Array,
    response: Array,
    drivers: Callable[[Array], Array] | None,
) -> int | None:
    """The first step at which a fit the log cannot tell from the fitted one predicts another
    path, or None where none does (ADR 0054).

    Each such fit is the fitted one moved along a column of ``directions`` with the drift
    regression's ``response`` (:func:`chc.dynamics_id._absorbed`), by any amount. The field is
    linear in the parameters, so where a move leaves the field as it was at every point the step
    reads it at, RK4's four or the Euler map's one, the step lands where it did whatever the
    amount, and the path is the same, not only to first order. A step that sets ruled levers holds
    them over the step at their level
    at its start, as the log held them over a period, so inside the step the move does not cancel,
    in the log's own steps either: such a step is one of the log's own where the move cancels at
    the state it starts from and the action it holds. The move counts as zero where the terms that
    make it up cancel to the square root of the working precision; a nan does not."""
    if directions.shape[1] == 0 or plan.actions.shape[0] == 0:
        return None
    precision = jnp.sqrt(jnp.finfo(directions.dtype).eps)

    def clear(t: Array, x: Array, action: Array) -> Array:
        design = _channel_design(action[None, :], x[None, :], fit.residual.channel_degree)[0]
        features = control_affine_features(x, fit.residual.degree)
        if drivers is not None:
            features = jnp.concatenate([features, drivers(t)])
        moved = design @ directions - features @ response
        size = jnp.abs(design) @ jnp.abs(directions) + jnp.abs(features) @ jnp.abs(response)
        return jnp.all(jnp.abs(moved) <= precision * size)

    def step(t: Array, x: Array, u: Array) -> Array:
        if isinstance(model, _SetAtStep):
            return clear(t, x, model.at_step(x, u)[1])
        if isinstance(model, _EulerStep):
            return clear(t, x, u)
        k1 = model(t, x, u)
        k2 = model(t + 0.5 * dt, x + 0.5 * dt * k1, u)
        k3 = model(t + 0.5 * dt, x + 0.5 * dt * k2, u)
        return (
            clear(t, x, u)
            & clear(t + 0.5 * dt, x + 0.5 * dt * k1, u)
            & clear(t + 0.5 * dt, x + 0.5 * dt * k2, u)
            & clear(t + dt, x + dt * k3, u)
        )

    times = dt * jnp.arange(plan.actions.shape[0], dtype=plan.trajectory.dtype)
    clean = np.asarray(jax.vmap(step)(times, plan.trajectory[:-1], plan.actions))
    loaded = np.flatnonzero(~clean)
    return int(loaded[0]) if loaded.size else None


def _resolve_adjustment(
    adjustment: CausalGraph | Sequence[str],
    *,
    panel: Panel,
    states: tuple[str, ...],
    levers: tuple[str, ...],
) -> AdjustmentSet:
    """Derive the set from a graph, or take the caller's word and record that that is what it is.

    The fit reads the channel into every state off one set, so a graph's set is derived for the
    levers' effect on all of them together: a set that identifies the effect on the target alone
    leaves the channel into a constrained state confounded, and the barrier and the tube priced on
    it."""
    if isinstance(adjustment, CausalGraph):
        adjustment.require_columns(panel.names)
        joint = adjustment.adjustment_set(treatment=levers, outcome=states)
        if joint.status == "identified" or len(states) == 1:
            return joint
        confounded = [
            state
            for state in states
            if adjustment.adjustment_set(treatment=levers, outcome=state).status != "identified"
        ]
        return AdjustmentSet(
            (),
            "not_identified",
            f"{joint.reason}; the fit reads the channel into every state of {list(states)} off one "
            "set, and "
            + (
                f"none identifies the levers' effect on {confounded}"
                if confounded
                else "no one set identifies the levers' effect on all of them"
            ),
        )
    named = tuple(str(name) for name in adjustment)
    missing = sorted(set(named) - set(panel.names))
    if missing:
        raise KeyError(f"asserted adjustment set names columns not in the panel: {missing}")
    acted = sorted(set(named) & set(levers))
    if acted:
        raise DecisionError(
            f"the asserted adjustment set names the levers {acted}: a lever is the action, and "
            "adjusting for it leaves the log no move of its own to learn from"
        )
    return AdjustmentSet(
        named,
        "identified",
        "asserted by the caller, not derived from a graph; nothing here checked it",
    )


def _period_steps(panel: Panel) -> NDArray[np.int64]:
    """Each row's period as a step on the panel's time grid: two periods are one step apart only
    where no period lies between them.

    The grid is the panel's declared ``frequency``, a calendar's or a step's, where it has one, so
    a period no unit logged still parts the two on either side of it; ``"observed"`` ranks the
    periods as :meth:`chc.panel.Panel.codes` ranks them. Undeclared, periods that are numbers or
    dates sit on the grid of their smallest spacing where every period falls on it. Periods off
    such a grid, calendar months stamped as dates among them, and text are ranked: the periods
    logged are the grid, and a period no unit logged is not seen. A number read as a grid point and
    not as a position, such as 202412 for a month, puts each year's turn 89 steps from the month
    before it, so no transition crosses it.

    An inferred grid is refused where most of the periods a unit logs in a row are more than one of
    its steps apart: the least spacing was then set by a period off the log's own step, one stray
    row at 3.5 in a log of whole periods, and on its grid every other transition parts as a gap.

    Raises:
        PanelError: for such a grid, naming the two periods that set its step and a row of each.
    """
    column = np.asarray(panel[panel.time])
    frequency = panel.provenance.frequency
    if frequency is not None and frequency != "observed":
        return _ordinals(column, frequency)
    steps = None if frequency == "observed" else _grid_steps(column)
    if steps is None:
        return panel.codes()[1]
    units, _ = panel.codes()
    order = np.lexsort((steps, units))
    within = units[order][1:] == units[order][:-1]
    gaps = np.diff(steps[order])[within]
    apart = int(np.count_nonzero(gaps > 1))
    if 2 * apart > gaps.size:
        distinct = np.unique(steps)
        least = int(np.flatnonzero(np.diff(distinct) == 1)[0])
        rows = [int(np.flatnonzero(steps == distinct[least + k])[0]) for k in (0, 1)]
        stamps, units_logged = column.tolist(), panel[panel.unit].tolist()
        read = [f"{stamps[row]!r} (row {row}, unit {units_logged[row]!r})" for row in rows]
        raise PanelError(
            f"time column {panel.time!r} is read on the grid of its least spacing, set by "
            f"{read[0]} and {read[1]}, and on that grid {apart} of the {gaps.size} periods units "
            "log in a row are more than one step apart, so each would part its two rows as a gap; "
            "declare the periods' step with frequency=, or frequency='observed' to read them in "
            "their sorted order"
        )
    return steps


def _transitions(
    panel: Panel,
    *,
    states: tuple[str, ...],
    levers: tuple[str, ...],
    adjust_for: tuple[str, ...],
    drivers: tuple[str, ...] = (),
) -> tuple[dict[str, Array], NDArray[Any]]:
    """``(x, u, x_next)`` over consecutive periods within a unit, plus the adjustment columns, and
    each driver at both ends of the transition (``name`` and ``f"{name}_next"``); beside them, two
    labels a transition, ``(N, 3)`` codes: its cluster, the panel's cluster column at its first
    period where the panel declares one and its unit where not, the period it starts in, in steps,
    and its unit.

    Gaps are dropped, not interpolated: a unit missing period ``t`` contributes the transitions on
    either side of the hole and nothing across it, and so does a period no unit logged, where the
    periods are numbers (:func:`_period_steps`). That is why an unbalanced panel is allowed here
    while :meth:`chc.panel.Panel.wide` refuses one --- a transition needs two adjacent rows, not a
    rectangle.
    """
    unit_codes, _ = panel.codes()
    time_codes = _period_steps(panel)
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

    data: dict[str, Array] = {
        "x": _stacked(panel, states, current),
        "u": _stacked(panel, levers, current),
        "x_next": _stacked(panel, states, following),
    }
    holds = {"x": "the states", "u": "the levers", "x_next": "the states a period on"}

    def put(key: str, name: str, rows: NDArray[np.int64], what: str) -> None:
        # The fit reads x, u and x_next by role and every other column by its name, all from this
        # one dict: a name that is already a key would be read in two roles.
        if key in holds:
            raise DecisionError(
                f"{what} would be read as {key!r}, which already holds {holds[key]}; rename the "
                "panel's column"
            )
        holds[key] = what
        data[key] = _stacked(panel, (name,), rows)

    for name in adjust_for:
        if name == "x" and states == ("x",):
            continue  # the one state, adjusted for: "x" already holds its column
        put(name, name, current, f"the covariate {name!r}")
    for name in drivers:
        if name not in adjust_for:  # a driver may be a covariate as well: one column, read once
            put(name, name, current, f"the driver {name!r}")
        put(f"{name}_next", name, following, f"the driver {name!r} a period on")
    cluster = (
        unit_codes
        if panel.cluster is None
        else np.unique(np.asarray(panel[panel.cluster]), return_inverse=True)[1].reshape(-1)
    )
    return data, np.column_stack(
        [cluster[current], time_codes[current], unit_codes[current]]
    ).astype(np.int64)


def _named(argument: object) -> tuple[str, ...]:
    """``argument`` as the column names it holds: itself where it names one, else none."""
    return (argument,) if isinstance(argument, str) else ()


def _group_of_each_unit(panel: Panel, column: str) -> NDArray[np.int64]:
    """Each row's group code in ``column``, refused where a unit's rows are in two groups."""
    units, _ = panel.codes()
    groups = np.unique(np.asarray(panel[column]), return_inverse=True)[1].reshape(-1)
    pairs = np.unique(np.column_stack([units, groups]), axis=0)
    split = pairs[1:, 0][pairs[1:, 0] == pairs[:-1, 0]]
    if split.size:
        raise DecisionError(
            f"unit {panel.units[int(split[0])]!r} is in more than one group of {column!r}; a "
            "unit's group holds over all its rows"
        )
    return groups


def _number_column(panel: Panel, name: str) -> bool:
    """Whether ``name`` holds real numbers: a boolean, integer or floating column, or an object
    column of numbers (:func:`chc.frames._not_numbers`)."""
    column = panel[name]
    kind = column.dtype.kind
    return kind in "biuf" or (kind == "O" and _not_numbers(column) is None)


def _one_value_per_group(panel: Panel, name: str, groups: NDArray[np.int64]) -> bool:
    """Whether column ``name`` holds one value over each group's rows."""
    codes = np.unique(np.asarray(panel[name]), return_inverse=True)[1].reshape(-1)
    return np.unique(np.column_stack([groups, codes]), axis=0).shape[0] == np.unique(groups).size


def _cluster_panel(
    panel: Panel, column: str, numbers: tuple[str, ...], labels: tuple[str, ...] = ()
) -> Panel:
    """The panel with one row per group of ``column`` and period, each numeric column the mean of
    the group's rows logged that period; ``numbers`` are read as numbers or refused, and
    ``labels`` are carried as they are, refused where a group's rows hold two values.

    The group's label is the unit, and the period label of the group's first row the time, which
    the panel's ``frequency`` reads to the same period. The declared ``cluster`` is kept where it
    holds one value over each group's rows."""
    groups = _group_of_each_unit(panel, column)
    _, periods = panel.codes()
    key = groups * (int(periods.max()) + 1) + periods
    _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    counts = np.bincount(inverse)
    data: dict[str, NDArray[Any]] = {
        column: np.asarray(panel[column])[first],
        panel.time: np.asarray(panel[panel.time])[first],
    }
    for name in labels:
        if name in data:
            continue
        if not _one_value_per_group(panel, name, groups):
            raise DecisionError(
                f"a group of {column!r} is in more than one group of {name!r}; a group's rows "
                "are one unit, which is in one group throughout"
            )
        data[name] = np.asarray(panel[name])[first]
    declared = panel.cluster if panel.cluster not in (None, column, *labels) else None
    if declared is not None and _one_value_per_group(panel, declared, groups):
        data[declared] = np.asarray(panel[declared])[first]
    else:
        declared = panel.cluster if panel.cluster in labels else None
    for name in panel.names:
        if name in data or name in (panel.unit, panel.cluster):
            continue
        if name in numbers or _number_column(panel, name):
            data[name] = np.bincount(inverse, weights=panel._numbers(name)) / counts
    return Panel.from_frame(
        data,
        unit=column,
        time=panel.time,
        frequency=panel.provenance.frequency,
        cluster=declared,
        seed=panel.provenance.seed,
    )


def _exposure(
    panel: Panel,
    levers: tuple[str, ...],
    neighbours: Mapping[Any, Iterable[Any]] | str,
    labels: NDArray[np.int64],
) -> NDArray[np.float64]:
    """For each transition :func:`_transitions` labels, each lever's mean over the unit's
    neighbours logged in the period it starts in; nan where none is."""
    unit_codes, _ = panel.codes()
    if isinstance(neighbours, str):
        groups = _group_of_each_unit(panel, neighbours)
        group_of = dict(zip(unit_codes.tolist(), groups.tolist(), strict=True))
        members: dict[int, list[int]] = {}
        for unit, group in sorted(group_of.items()):
            members.setdefault(group, []).append(unit)
        adjacency = {
            unit: [other for other in members[group] if other != unit]
            for unit, group in group_of.items()
        }
    else:
        index = {label: code for code, label in enumerate(panel.units)}
        named = {unit: tuple(others) for unit, others in neighbours.items()}
        unknown = [
            label
            for label in (*named, *(other for others in named.values() for other in others))
            if label not in index
        ]
        if unknown:
            raise DecisionError(
                f"neighbours name {unknown[0]!r}, which is not a unit of the panel; units are "
                f"{list(panel.units)[:10]}"
            )
        adjacency = {
            index[unit]: sorted({index[other] for other in others} - {index[unit]})
            for unit, others in named.items()
        }
    steps = _period_steps(panel)
    row_of = {
        (unit, step): row
        for row, (unit, step) in enumerate(zip(unit_codes.tolist(), steps.tolist(), strict=True))
    }
    values = np.column_stack([panel._numbers(name) for name in levers])
    exposure = np.full((labels.shape[0], len(levers)), np.nan)
    for row, (step, unit) in enumerate(labels[:, 1:].tolist()):
        rows = [
            row_of[(other, step)] for other in adjacency.get(unit, ()) if (other, step) in row_of
        ]
        if rows:
            exposure[row] = np.mean(values[rows], axis=0)
    return exposure


def _stacked(panel: Panel, names: tuple[str, ...], rows: NDArray[np.int64]) -> Array:
    """The columns ``names`` at ``rows``, one column each, as the fit reads them."""
    if not names:
        return jnp.zeros((rows.size, 0))
    return jnp.stack([jnp.asarray(panel._numbers(name)[rows]) for name in names], axis=1)


def _panel_start(panel: Panel, states: tuple[str, ...]) -> tuple[Array, StartState]:
    """The mean over units of each unit's states at its latest period, the units in code order, in
    one pass over the rows; and its record."""
    unit_codes, _ = panel.codes()
    time_codes = _period_steps(panel)
    latest: dict[int, tuple[int, int]] = {}
    for row, (unit, period) in enumerate(
        zip(unit_codes.tolist(), time_codes.tolist(), strict=True)
    ):
        if unit not in latest or period > latest[unit][0]:
            latest[unit] = (period, row)
    rows = np.array([latest[unit][1] for unit in sorted(latest)], dtype=np.int64)
    last = _stacked(panel, states, rows)
    start = jnp.mean(last, axis=0)
    record = StartState(
        "panel",
        states,
        tuple(float(value) for value in np.asarray(start, dtype=float)),
        units=int(rows.size),
        periods=tuple(sorted(set(np.asarray(panel[panel.time])[rows].tolist()))),
        spread=tuple(float(value) for value in np.std(np.asarray(last, dtype=float), axis=0)),
    )
    return start, record


# The class :func:`chc.dynamics_id.fit_causal_residual` regresses the levers on by default.
_LOGGER_CHECK_DEGREE = 2
# The degree of the nuisances' polynomial prescribe fits with, and so of the rules of the state the
# log's actions are read against (ADR 0054).
_NUISANCE_DEGREE = 2
_LOGGER_CHECK_ALPHA = 0.05  # the level at which a check is logged as a warning
_PERSISTENCE_ALPHA = 0.05  # the level at which the report names the noise's persistence
_JACKKNIFE_GROUPS = 50  # the most groups of units the tube's noise read is jackknifed over
_INTERFERENCE_ALPHA = 0.05  # the level at which a spillover is logged and named in the report
# The fit's ridge and each descent's budget of steps, the fit's and the planner's defaults, passed
# by name so that the run's record holds what the calls were given (ADR 0063).
_RIDGE = 1e-6
_PLAN_STEPS = 10_000
# The distributions whose versions a run records: the library and those that compute its numbers.
_VERSIONED = ("causal-hybrid-control", "jax", "jaxlib", "numpy", "scipy")


def _versions() -> tuple[tuple[str, str], ...]:
    """Each of :data:`_VERSIONED` at the version installed, read from its metadata.

    A module's ``__version__`` is a second copy of that, which can lag, as ``chc.__version__`` did
    (:func:`chc.panel.installed_version`). ``"unknown"`` where a distribution has no metadata, as a
    source tree that was never installed has none.
    """
    found = []
    for name in _VERSIONED:
        try:
            found.append((name, metadata.version(name)))
        except metadata.PackageNotFoundError:
            found.append((name, "unknown"))
    return tuple(found)


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _readable(panel: Panel, name: str) -> bool:
    """Whether the logger check reads column ``name``: a column the panel holds of a boolean, an
    integer, a floating, a complex or a duration dtype, or an object column the rule reads as
    numbers (:func:`chc.frames._not_numbers`). The reader refuses complex numbers and durations
    (:meth:`chc.panel.Panel._numbers`); text, dates and an object column of other values are a
    panel's labels, and are not read."""
    if name not in panel.columns:
        return False
    column = panel[name]
    kind = column.dtype.kind
    return kind in "biufcm" or (kind == "O" and _not_numbers(column) is None)


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
    unit_codes, _ = panel.codes()
    time_codes = _period_steps(panel)
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
        return np.column_stack([panel._numbers(name)[rows] for name in names])

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
        "the levers read more than the state and their recorded parents, or read those through "
        "more than a quadratic"
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
    """The windows :meth:`Prescription.evaluate` reads: ``x (E, H + 1, n)``, ``u (E, H, m)``,
    ``units (E,)``, each window's unit as :meth:`chc.panel.Panel.codes` codes it, and ``starts
    (E,)``, each window's first period as a step of the panel's grid."""

    x: NDArray[np.float64]
    u: NDArray[np.float64]
    units: NDArray[np.int64]
    starts: NDArray[np.int64]


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
    unit_codes, _ = panel.codes()
    time_codes = _period_steps(panel)
    order = np.lexsort((time_codes, unit_codes))
    if time_zero == "calendar":
        # Sorted by unit and period, a unit's periods rise strictly: H + 1 rows in a row of one
        # unit whose periods differ by H are H + 1 consecutive periods.
        units, times = unit_codes[order], time_codes[order]
        last = int(time_codes.max())
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
        return np.stack([panel._numbers(name) for name in names], axis=1)

    return _Episodes(
        stack(states)[rows],
        stack(levers)[rows[:, :-1]],
        unit_codes[rows[:, 0]],
        time_codes[rows[:, 0]],
    )


@eqx.filter_jit
def _one_step(model: Dynamics, x: Array, u: Array, dt: float) -> tuple[Array, Array, Array]:
    """``model``'s step over ``dt`` from ``(x, u)`` (:func:`chc.integrate.rk4_step`) and its
    Jacobians there, compiled: an evaluation linearises once for every draw of its bootstrap."""
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
        logged = panel._numbers(driver.name)
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


def _warn_actions_outside_logged_range(
    panel: Panel, levers: tuple[str, ...], actions: NDArray[np.float64], *, skip: tuple[str, ...]
) -> None:
    """Warn of each lever the plan sets outside the range the panel logged it in: the fitted
    channel is extrapolated there, and its standard error, read at the log's actions, does not
    price that. A lever on its logged rule (``skip``) is set by the state, not by the plan."""
    for index, name in enumerate(levers):
        if name in skip:
            continue
        logged = panel._numbers(name)
        low, high = float(logged.min()), float(logged.max())
        planned = actions[:, index]
        outside = np.flatnonzero((planned < low) | (planned > high))
        if outside.size:
            _log.warning(
                "the plan sets %r outside the range the panel logged, [%.4g, %.4g], first at step "
                "%d of %d; the fitted channel is extrapolated there, and the tube's error, read at "
                "the log's actions, does not price that",
                name,
                low,
                high,
                int(outside[0]),
                planned.size,
                extra={
                    "chc_event": "lever_range",
                    "lever": name,
                    "logged": [low, high],
                    "planned": [float(planned.min()), float(planned.max())],
                    "steps": outside.tolist(),
                },
            )


def _cost(
    states: tuple[str, ...],
    levers: Sequence[Lever],
    target: Target,
    horizon: int,
    rule: _Rule | None = None,
) -> QuadraticCost:
    """Weight the target's own coordinate and price each lever; constrained states are free.

    A constrained state gets weight zero rather than a small one: its bound is enforced by the
    barrier and priced by the certificate, and adding a quadratic pull towards zero would be a
    second, unstated objective.

    A schedule becomes one target per state. The start's row repeats the first level; no action
    reaches the start, so that row moves the reported cost and nothing the plan does.

    A lever on ``rule`` is priced at the level the rule sets, the action the field takes.
    """
    weights = jnp.array([target.weight] + [0.0] * (len(states) - 1))
    if np.ndim(target.value) == 0:
        goal = jnp.array([target.value] + [0.0] * (len(states) - 1))
    else:
        levels = jnp.asarray(target.value)
        path = jnp.concatenate([levels[:1], levels])
        goal = jnp.zeros((horizon + 1, len(states))).at[:, 0].set(path)
    q = jnp.diag(weights)
    r = jnp.diag(jnp.array([lever.unit_cost for lever in levers]))
    if rule is None:
        return QuadraticCost(Q=q, R=r, Qf=q, x_target=goal)
    return _RuledCost(Q=q, R=r, Qf=q, x_target=goal, rule=rule)


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


def _rate(model: Dynamics, x0: Array, u_lo: Array, u_hi: Array) -> tuple[float, TubeRate]:
    """The tube's rate, a norm-Lipschitz bound on the field in the state, and where it holds.

    The norm of the field's slope in the state at the box's centre, plus half each lever's width
    times the norm of what a unit of that lever adds to the slope. A control-affine field's slope
    is affine in the actions, so over the box it is within that sum of the centre's. On a field
    affine in the state (:func:`_slope_free_of_state`) the slope is the same at every state and the
    rate is ``global``; elsewhere it is read at ``x0`` and is ``local``. A log-norm is no such
    bound: it can be negative, which turned the tube's radii negative and certified every step.
    """
    centre = (u_lo + u_hi) / 2.0
    slope, _ = linearize_continuous(model, x0, centre)
    spread = 0.0
    for lever in range(centre.size):
        moved, _ = linearize_continuous(model, x0, centre.at[lever].add(1.0))
        spread += float(u_hi[lever] - u_lo[lever]) / 2.0 * float(jnp.linalg.norm(moved - slope, 2))
    rate = float(jnp.linalg.norm(slope, 2)) + spread
    return rate, "global" if _slope_free_of_state(model) else "local"


def _slope_free_of_state(model: Dynamics) -> bool:
    """Whether the field's slope in the state is the same at every state, by its structure alone:
    affine in the state for each action, a control channel at most affine in it. A drift past
    degree 1, a channel past degree 1, or any field not named here is not taken to be so."""
    if isinstance(model, LinearDynamics | DampedOscillator | ZeroResidual):
        return True
    if isinstance(model, ControlAffineResidual):
        return model.degree <= 1 and model.channel_degree <= 1
    if isinstance(model, HybridDynamics):
        return _slope_free_of_state(model.known) and _slope_free_of_state(model.residual)
    if isinstance(model, DrivenDynamics):
        return _slope_free_of_state(model.dynamics)
    if isinstance(model, _EulerStep):
        return _slope_free_of_state(model.field)
    return False


def _slopes(model: Dynamics, x0: Array) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """The field's slope in the state at each of a plan's actions, the ``(H, n, n)`` matrices the
    tube is carried by on a field affine in the state (:func:`_slope_free_of_state`), read at
    ``x0``, where it is the slope at every state. A control channel affine in the state moves the
    slope with the action, by about 0.04 a unit on the quickstart's fitted field, so each step
    reads its own."""
    slope = jax.vmap(lambda u: jax.jacobian(lambda x: model(0.0, x, u))(x0))

    def along(actions: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(slope(jnp.asarray(actions, dtype=x0.dtype)), dtype=np.float64)

    return along


def _state_noise(
    slopes: NDArray[np.float64],
    noise: NDArray[np.float64],
    persistence: float | NDArray[np.float64],
    dt: float,
    horizon: int,
    read: _NoiseRead | None = None,
    *,
    integrator: Integrator = "rk4",
) -> tuple[tuple[float, ...], ...]:
    """:attr:`DecisionCertificate.tube_state_noise`: each state's 0.95 radius of the noise's part
    of the gap at each step, ``sqrt(chi2_1(0.95) P_k[i, i])``, with the ``read``'s own error
    carried as the tube carries it, by the steps of ``integrator``."""
    level = float(chi2.ppf(_TUBE_COVERAGE, 1))
    return tuple(
        tuple(math.sqrt(level * max(float(variance), 0.0)) for variance in np.diag(covariance))
        for covariance in _read_covariances(
            slopes, noise, persistence, dt, horizon, read, integrator=integrator
        )
    )


def _field_error(fit: CausalDynamicsFit, reach: NDArray[np.float64]) -> float:
    """The fitted field's error over the log, a standard error: the channel's at each lever's
    ``reach`` (:func:`_model_error`), and the drift's own at the log's states and mean action,
    ``drift_error``, in quadrature. Its root mean square over the log's rows is what the tube
    reads at each planned state (:func:`_field_errors`), and the plan is solved with it.

    The drift absorbed the channel at the log's mean action (ADR 0091), so the field's error at an
    action is the drift's error there plus the channel's times the action's distance from it.
    Read at that centre the two are uncorrelated to first order, as an intercept read at the
    regressor's mean is with its slope, and so add in quadrature. Up to 0.17.0 the tube carried the
    channel's part alone: on the quickstart 0.016, where the drift's is 0.039, the demand the fit
    adjusts for and the plan does not forecast being in the drift's residual (ADR 0096). A fit with
    no channel error carries no tube, the drift's error with it.

    The tube bounds the gap's Euclidean norm over the states, so each part is the scale of that
    norm: ``drift_error``, a root mean over the states, times the root of their number.
    """
    channel = _model_error(fit, reach)
    if fit._lever_errors is None or fit.drift_error is None:
        return channel
    return math.hypot(channel, math.sqrt(fit.residual.channel.shape[0]) * fit.drift_error)


def _field_errors(
    fit: CausalDynamicsFit,
    reach: NDArray[np.float64],
    states: NDArray[np.float64],
    drivers: NDArray[np.float64],
) -> NDArray[np.float64]:
    """:func:`_field_error` read at each of the plan's ``(H, n)`` states rather than over the log's
    rows: at the state each step starts from, ``drivers`` ``(H, d)`` as the drift reads them over
    that step. The channel's error is each lever's at that state priced at its ``reach``
    (:func:`_model_errors`), the drift's the standard error of its fitted value there, from the
    coefficients' covariance ``drift_error`` reads at the log's rows
    (``CausalDynamicsFit._drift_blocks``), the two in quadrature as :func:`_field_error` adds them.

    A fitted value's error grows with its distance from the log's states, and a plan goes where
    its target is, not where the log was: on 100 one-unit logs of 60 transitions of an AR(1) plant
    whose state spreads 0.52 about 0, a plan to 1 reads the drift's error 1.45 to 1.63 times its
    root mean over the rows from its fourth step on, and the channel's 1.39 to 1.57 times. Read
    over the rows, the field's error the tube carried fell short of the plant's own departure from
    the plan at its late steps (ADR 0104). A fit that keeps no blocks reads its root means at
    every step. Each part is the scale of the error's Euclidean norm over the states, as
    :func:`_field_error` reads it (ADR 0100): the drift's mean variance over the states' blocks,
    one every state shares under Euler, times their number.
    """
    channel = _model_errors(fit, reach, states)
    if fit._lever_errors is None or fit.drift_error is None:
        return channel
    count = fit.residual.channel.shape[0]
    if fit._drift_blocks is None:
        return np.hypot(channel, math.sqrt(count) * fit.drift_error)
    features = np.concatenate(
        [_features(states, fit.residual.degree), np.asarray(drivers, dtype=np.float64)], axis=1
    )
    variances = np.einsum("kf,gfh,kh->kg", features, fit._drift_blocks, features)
    return np.hypot(channel, np.sqrt(count * np.maximum(np.mean(variances, axis=1), 0.0)))


def _model_errors(
    fit: CausalDynamicsFit, reach: NDArray[np.float64], states: NDArray[np.float64]
) -> NDArray[np.float64]:
    """:func:`_model_error` read at each of ``states`` rather than over the log's rows: each lever's
    error is the root mean over its column of the channel's entries of their standard errors at
    the state, from the coefficients' covariances ``_lever_errors`` reads at the rows
    (``CausalDynamicsFit._channel_blocks``), priced at its ``reach`` and summed as
    :func:`_model_error` sums them, the scale of the Euclidean norm over the states. A fit that
    keeps no blocks reads :func:`_model_error` at every state."""
    rows = np.atleast_2d(np.asarray(states, dtype=np.float64))
    if fit._lever_errors is None:
        return np.zeros(rows.shape[0])
    if fit._channel_blocks is None:
        return np.full(rows.shape[0], _model_error(fit, reach))
    levers = fit._lever_errors.shape[1]
    features = _features(rows, fit.residual.channel_degree)
    # (r, H, n m): each covariance's variance of each entry at each state, entry (i, j) at i m + j
    variances = np.einsum("kf,refh,kh->rke", features, fit._channel_blocks, features)
    by_lever = np.sqrt(
        np.maximum(
            np.stack([np.mean(variances[:, :, j::levers], axis=2) for j in range(levers)], 2), 0.0
        )
    )
    rates = by_lever * reach
    count = fit.residual.channel.shape[0]
    return np.max(np.sqrt(count * np.sum(rates * rates, axis=2)), axis=0)


def _features(states: NDArray[np.float64], degree: int) -> NDArray[np.float64]:
    """``(H, F)``: :func:`chc.residual.control_affine_features` of each of ``states``."""
    return np.asarray(
        jax.vmap(control_affine_features, in_axes=(0, None))(jnp.asarray(states), degree),
        dtype=np.float64,
    )


def _field_scale(fit: CausalDynamicsFit) -> float:
    """How many of its standard errors the tube reads :func:`_field_error` at, in units of the
    normal's: ``t_nu(0.975) / z(0.975)``, ``nu`` the degrees of freedom the channel's error is read
    with, the independent terms its spread sums less one, its cosines or its clusters
    (:func:`chc.dynamics_id._degrees_of_freedom`, the ``t`` its tests are sized against). The
    drift's error is read from its regression's residuals as if each were independent, which on a
    log whose noise persists or clusters they are not; the channel's count is the robust one, and
    the tube reads the field's whole error at it. Welch--Satterthwaite's count of the two parts,
    the drift's at its residual degrees of freedom, read the tube shorter on short logs and the
    same on long ones (ADR 0104).

    The field's error at a step is ``f`` times Student's ``t`` along a direction the fit does not
    give. :func:`chc.plan._error_tube` places it along the noise's widest, where it widens the
    radius most, as a Gaussian of standard deviation ``t / z`` times ``f``: beside the noise, that
    radius holds it with probability 0.9496 to 0.9523 at ``nu`` 3 to 52 over every ratio of the two
    on one state, 0.9500 to 0.9514 on two where it lies along the widest direction, and more where
    it does not: 0.960 and 0.972 along the least, at ``nu`` 3 and 52
    (``validation/tube_level.mac``). The noise's radius plus ``f``, as the tube read it up to
    0.18.0, holds 0.758 at ``nu = 3`` and 0.846 at ``nu = 52`` where the field's error is four
    times the noise's.
    """
    assert fit.defect is not None  # every fit keeps its defect
    degrees = _degrees_of_freedom(int(np.asarray(fit.defect).shape[0]), fit.clusters, fit.periods)
    t_level = float(student_t.ppf(0.5 + _TUBE_COVERAGE / 2.0, degrees))
    return t_level / math.sqrt(float(chi2.ppf(_TUBE_COVERAGE, 1)))


def _model_error(fit: CausalDynamicsFit, reach: NDArray[np.float64]) -> float:
    """Turn the channel's standard error into the per-step rate error the Gronwall tube wants.

    ``channel_error`` is the scale of ``B_hat(x) - B(x)`` per entry at the log's states. Entry
    ``(i, j)`` is in the states' units per unit of lever ``j``, and ``reach[j]``, which
    :func:`prescribe` passes as the largest distance lever ``j``'s box allows from the log's mean
    action, is in lever ``j``'s units: their product is the rate error that action induces, in the
    states' units, whatever lever ``j``'s units are. The distance, not the action: the fit's drift
    absorbs the channel at the log's action, so its error moves the field by the action's distance
    from there, and an origin of a lever moved with its log is the same field (ADR 0091). Up to
    0.17.0 ``reach[j]`` was the largest action from the lever's zero: on a log of one state and two
    levers in ``[-2, 2]``, ``u1``'s zero moved by 100, its log and box with it, ended the tube at
    0.713 where it ends at 0.538, and certified one step where it certifies two.

    The tube bounds the gap's Euclidean norm over the states, and the rate error at an action
    ``d`` is the vector ``(B_hat - B) d``, whose mean square norm is the sum over the entries of
    each one's mean square error times ``d_j^2``, where the entries' errors are uncorrelated. This
    returns its root, ``sqrt(n sum_j (e_j reach_j)^2)`` for ``n`` states, ``e_j`` lever ``j``'s
    read of a covariance ``channel_error`` is read off (``CausalDynamicsFit._lever_errors``), a
    root mean over the states, the largest read taken once the levers are priced (ADR 0087).
    Up to 0.18.0 it was the root mean over the entries, ``sqrt(n m)`` times smaller for ``m``
    levers: over 200 logs of two states and two levers, the field error's root mean square norm
    read 1.93 times its root mean square, where it reads 0.96 times this. With one state and one
    lever it is ``channel_error`` times ``reach``, bit for
    bit; with levers in one unit whose ``reach`` is one number, that times ``sqrt(n m)`` to
    rounding. Up to 0.16.0 it was ``channel_error``
    times the largest action of any lever, which mixed the levers' units: on a log of one state and
    two levers in ``[-2, 2]``, either lever in units 1e3 or 1e-3 times its own, its box with it,
    ended the tube 7 times as wide and certified neither of the two steps it certifies. A lever
    whose ``reach`` is zero, held at the log's mean action, adds no error, and ``reach`` zero for
    every lever leaves the channel nothing to move, and no tube.

    It is a standard error, not coverage: ``channel_error`` is the robust standard error,
    root-mean over the entries and the log's states, calibrated on average but scattering from one
    log to the next, so one fit's tube is as wide as that fit's reading; the tube reads it at the
    quantile of Student's ``t`` at the degrees of freedom it is read with (:func:`_field_scale`).
    The docstring of :class:`chc.dynamics_id.CausalDynamicsFit` says by how much.
    """
    if fit._lever_errors is None:
        return 0.0
    rates = fit._lever_errors * reach
    states = fit.residual.channel.shape[0]
    return float(np.max(np.sqrt(states * np.sum(rates * rates, axis=1))))


def _step_noise(
    fit: CausalDynamicsFit, dt: float, labels: NDArray[Any]
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """The plant's noise as the tube reads it: the plan's one-step error on the log, ``dt`` times
    the fit's defect, its covariance ``S`` in the states' units, the mean of its outer products;
    the VAR(1) coefficient that carries it from step to step (:func:`_autoregression`); and its
    lag-1 correlation within units, pooled over the states as
    :func:`~chc.dynamics_id.persistence_check` pools them, which the report prints. The two are
    nan, unknown, where the log has noise and no unit has two consecutive transitions to read them
    from, and the tube is then not evaluated: up to 0.18.0 it read 0, a tube holding white noise
    where the noise may persist. Where the log has no noise they are 0, which moves nothing.
    ``labels`` are :func:`_transitions`' group, period and unit of each transition. The defect is
    the fit's integrator's, ``rk4`` by default, the step the plan rolls out by."""
    assert fit.defect is not None  # every fit keeps its defect
    error = dt * np.asarray(fit.defect, dtype=np.float64)
    covariance = error.T @ error / error.shape[0]
    if not covariance.any():
        return covariance, np.zeros_like(covariance), 0.0
    correlation = _persistence(
        error, labels[:, 2], labels[:, 1], fit.periods is not None
    ).correlation
    return covariance, _autoregression(error, covariance, labels[:, 2], labels[:, 1]), correlation


def _noise_read(fit: CausalDynamicsFit, dt: float, labels: NDArray[Any]) -> _NoiseRead | None:
    """The sampling error of :func:`_step_noise`'s ``S`` and ``Phi``, which the tube carries
    (:func:`chc.uncertainty._read_covariances`): both read again with each of up to
    :data:`_JACKKNIFE_GROUPS` groups of the log's units left out, and the covariance of the
    noise's mean over the log, ``G / (G - 1) sum_g s_g s_g' / N^2``, ``s_g`` unit ``g``'s sum of
    its ``N`` rows' errors, robust to a noise that persists within units. The units are
    independent, so the log's own spread over them is its error. None on a log of one unit, which
    reads no spread over units, and where leaving a group out leaves no pair of consecutive
    periods.

    The fitted drift absorbs the noise's mean on the log, and with it the mean of a confounder the
    fit adjusts for and the plan does not forecast: the plant's noise misses it by that mean's
    error. On the quickstart's plant with its demand an AR(0.8), recorded and adjusted, the plan's
    mean departure from the plant correlated -0.99 with the log's mean demand over 16 logs, 0.010
    at the first step and 0.105 at the fifteenth, where the field's error the tube carries reads
    0.002 and 0.065. Over 16 logs ``S`` read 0.82 to 1.12 times its own on the supply and ``Phi``
    0.757 to 0.832 for 0.796, each unbiased, and a tube that takes them for the plant's held 0.944
    a step. The drift's own error at the log's mean reads the part of the mean's error that is the
    drift's residual, and that part is counted twice: on that plant its variance is 0.5 % of the
    mean's (ADR 0105)."""
    assert fit.defect is not None  # every fit keeps its defect
    error = dt * np.asarray(fit.defect, dtype=np.float64)
    units = np.unique(np.asarray(labels[:, 2]).reshape(-1), return_inverse=True)[1].reshape(-1)
    count = int(units.max()) + 1
    if count < 2:
        return None
    sums = np.zeros((count, error.shape[1]))
    np.add.at(sums, units, error)
    level = count / (count - 1) * (sums.T @ sums) / error.shape[0] ** 2
    groups = units % min(count, _JACKKNIFE_GROUPS)
    earlier, later = _consecutive(labels[:, 2], labels[:, 1])
    covariances, carries = [], []
    for group in range(int(groups.max()) + 1):
        rows, pairs = groups != group, groups[earlier] != group
        if not pairs.any():
            return None
        kept = error[rows]
        covariance = kept.T @ kept / kept.shape[0]
        lagged = error[later[pairs]].T @ error[earlier[pairs]] / int(pairs.sum())
        covariances.append(covariance)
        carries.append(_yule_walker(lagged, covariance))
    return _NoiseRead(np.stack(covariances), np.stack(carries), level)


def _autoregression(
    error: NDArray[np.float64],
    covariance: NDArray[np.float64],
    units: NDArray[Any],
    periods: NDArray[Any],
) -> NDArray[np.float64]:
    """``Phi``, the ``(n, n)`` coefficient of the VAR(1) the tube carries the one-step ``error`` by,
    ``(N, n)`` rows of ``units`` at whole-number ``periods``: Yule--Walker's ``Gamma_1 S^-1``,
    ``Gamma_1`` the mean of ``w_t w_{t-1}'`` over the pairs of one unit's consecutive periods and
    ``S`` the ``covariance`` of every row, so that ``Cov(w_{k+1}, w_k) = Phi S`` as the log reads
    it, in any units and any basis of the states. Directions ``S`` does not span carry no noise
    and persist at 0. Where the pairs read a lag-1 covariance no stationary noise of covariance
    ``S`` has, its whitened ``S^{-1/2} Gamma_1 S^{-1/2}`` past a contraction, it is scaled back to
    one, the noise persisting fully along its most persistent direction. nan where no unit has two
    consecutive periods.

    A pooled correlation, one ``rho`` carrying ``Cov(w_i, w_j) = rho^|i - j| S``, reads a noise
    persistent on one state and white on another as half as persistent on both: on the
    quickstart's plant with its demand an AR(0.8), each state counted alike in its own units, it
    read 0.41 where the supply's error persists at 0.80 and the wait's at 0.02, and the plant left
    the tube from its second week on, 0.79 a week by the fifteenth (ADR 0105).
    """
    earlier, later = _consecutive(units, periods)
    states = covariance.shape[0]
    if earlier.size == 0:
        return np.full((states, states), math.nan)
    return _yule_walker(error[later].T @ error[earlier] / earlier.size, covariance)


def _consecutive(
    units: NDArray[Any], periods: NDArray[Any]
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """The rows of each pair of one unit's consecutive periods, the earlier's and the later's."""
    codes = np.unique(np.asarray(units).reshape(-1), return_inverse=True)[1].reshape(-1)
    steps = np.asarray(periods).reshape(-1)
    order = np.lexsort((steps, codes))
    earlier, later = order[:-1], order[1:]
    paired = (codes[earlier] == codes[later]) & (steps[later] - steps[earlier] == 1)
    return earlier[paired], later[paired]


def _yule_walker(
    lagged: NDArray[np.float64], covariance: NDArray[np.float64]
) -> NDArray[np.float64]:
    """:func:`_autoregression`'s ``Gamma_1 S^-1`` from the lag-1 covariance ``lagged``, its
    whitened form scaled back to a contraction."""
    values, vectors = np.linalg.eigh(covariance)
    live = values > 1e-12 * float(values[-1])
    basis, roots = vectors[:, live], np.sqrt(values[live])
    whitened = (basis.T @ lagged @ basis) / np.outer(roots, roots)
    norm = float(np.linalg.norm(whitened, 2))
    if norm > 1.0:
        whitened = whitened / norm
    return basis @ (roots[:, None] * whitened / roots[None, :]) @ basis.T


def _show(value: object) -> str:
    if value is None:
        return "not evaluated"
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def _strict(value: Any) -> Any:
    """``value`` with every float that is not finite read as None, which JSON can carry."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strict(item) for item in value]
    return value


def _json_label(label: Any) -> Any:
    """A period's label as JSON carries it: a number or text as it is, a date or a time in ISO
    8601, anything else as its text."""
    if isinstance(label, np.generic):
        label = label.item()
    if isinstance(label, int | float | str):
        return label
    if isinstance(label, datetime.date | datetime.time):
        return label.isoformat()
    return str(label)


def _show_start(start: StartState) -> str:
    named = [f"`{name}` {value:.4g}" for name, value in zip(start.states, start.value, strict=True)]
    if start.source == "given":
        return f"Start: `x0` as given, {', '.join(named)}."
    periods = start.periods
    logged = (
        f"period {periods[0]}"
        if len(periods) == 1
        else "periods " + ", ".join(str(period) for period in periods)
        if len(periods) <= 3
        else f"{len(periods)} periods, {periods[0]} to {periods[-1]}"
    )
    if start.units == 1:
        return f"Start: the one unit's last logged state, from {logged}: {', '.join(named)}."
    assert start.spread is not None  # a panel's start keeps its spread
    values = ", ".join(
        f"{text} ({spread:.4g})" for text, spread in zip(named, start.spread, strict=True)
    )
    return (
        f"Start: the mean (standard deviation) over {start.units} units of each unit's last "
        f"logged state, from {logged}: {values}."
    )


def _show_relation(relation: LeverRelation, names: Sequence[str]) -> str:
    terms = " ".join(
        f"{'-' if weight < 0.0 else '+'} {abs(weight):.4g} `{name}`"
        for weight, name in zip(relation.weights, names, strict=True)
        if weight != 0.0
    )
    head = "" if terms.startswith("+") else "-"
    return f"{head}{terms[2:]} = {relation.level:.4g}"


def _show_price(price: RowPrice, share: float = 1.0) -> str:
    if price.price is None:
        return f"[{share * price.interval[0]:.4g}, {share * price.interval[1]:.4g}]"
    return f"{share * price.price:.4g}"
