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
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from jax import Array

from chc.control import SolverStatus
from chc.cost import QuadraticCost
from chc.dynamics import Dynamics, HybridDynamics, LinearDynamics
from chc.dynamics_id import CausalDynamicsFit, Integrator, fit_causal_residual
from chc.graph import AdjustmentSet, CausalGraph
from chc.lqr import linearize_continuous
from chc.panel import Panel, Provenance
from chc.plan import CausalPlan, CertificateStatus, causal_plan, certify_safety

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
at --- ``precision``, ``adjustment``, ``fit``, ``abort``, ``plan``, ``certificate`` --- so a JSON
formatter downstream can route on one field rather than parse a sentence. The library installs no
handler and sets no level: that is the application's call, and a library that reaches for
``basicConfig`` takes it away.

The two records that are not ``INFO`` are the two worth waking someone for: identifying in single
precision, and a graph that says the effect is not identified at all.
"""

IdentificationStatus = Literal["identified", "asserted", "not_identified"]
"""How the set was arrived at. ``asserted`` means a caller named it and nothing here checked it."""


@dataclass(frozen=True)
class Lever:
    """One actionable column and the box it may move in.

    ``unit_cost`` is the quadratic price of using it, and defaults to free: with a box constraint a
    free lever is still bounded, so the default is safe rather than merely convenient.
    """

    name: str
    lo: float
    hi: float
    unit_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise DecisionError(f"lever {self.name!r} has lo={self.lo} above hi={self.hi}")


@dataclass(frozen=True)
class Target:
    """The column to steer and where to steer it."""

    name: str
    value: float
    weight: float = 1.0


@dataclass(frozen=True)
class Constraint:
    """A state that must stay inside a range --- what the safety certificate is priced against."""

    state: str
    lo: float | None = None
    hi: float | None = None

    def __post_init__(self) -> None:
        if self.lo is None and self.hi is None:
            raise DecisionError(f"constraint on {self.state!r} bounds nothing")
        if self.lo is not None and self.hi is not None and self.lo > self.hi:
            raise DecisionError(f"constraint on {self.state!r} has lo={self.lo} above hi={self.hi}")


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
    gamma_star: float | None  # weakest step's confounding ceiling (§40); None if unconstrained
    # None when the effect is not identified: no solve was attempted, which is a third answer and
    # not one of the solver's three. Same convention as the certified-horizon fields above.
    solver_status: SolverStatus | None
    solver_iterations: int

    @property
    def trustworthy_steps(self) -> int:
        """The prefix that survives *both* axes --- the number an operator can act on.

        Zero whenever the effect is not identified, whatever the tube says, and never longer than
        the shorter of the two certified prefixes. ``None`` on either axis means "not evaluated",
        which contributes zero here rather than infinity.
        """
        if self.identification == "not_identified":
            return 0
        limits = [self.certified_horizon, self.barrier_certified_steps]
        measured = [limit for limit in limits if limit is not None]
        return min(measured) if measured else 0


@dataclass(frozen=True)
class Prescription:
    """The decision, the evidence for it, and what it took to get there."""

    levers: tuple[Lever, ...]
    target: str
    plan: CausalPlan | None  # None iff the effect is not identified, so no plan was made
    certificate: DecisionCertificate
    model_fit: CausalDynamicsFit
    provenance: Provenance

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
        lines += [
            "## Certificate",
            "",
            f"- identification: **{certificate.identification}** ({certificate.adjustment.reason})",
            f"- adjusted for: {list(certificate.adjustment.covariates) or 'nothing'}",
            f"- channel standard error: {_show(certificate.identification_radius)}",
            f"- overlap (residualised action variance): {certificate.overlap:.4g}",
            f"- error tube: **{certificate.certificate_status}**, "
            f"certified horizon {_show(certificate.certified_horizon)}",
            f"- barrier: certified steps {_show(certificate.barrier_certified_steps)}, "
            f"gamma* {_show(certificate.gamma_star)}",
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
                "gamma_star": certificate.gamma_star,
                "solver_status": certificate.solver_status,
                "solver_iterations": certificate.solver_iterations,
                "trustworthy_steps": certificate.trustworthy_steps,
            },
            "provenance": self.provenance.to_json(),
        }


def prescribe(
    panel: Panel,
    *,
    levers: Sequence[Lever],
    target: Target,
    horizon: int,
    adjustment: CausalGraph | Sequence[str],
    constraints: Sequence[Constraint] = (),
    known: Dynamics | None = None,
    dt: float = 1.0,
    gamma: float = 1.0,
    tolerance: float | None = None,
    x0: Array | None = None,
    folds: int = 2,
    seed: int = 0,
    integrator: Integrator = "rk4",
) -> Prescription:
    """Fit the causal control channel from ``panel`` and plan a certified schedule on it.

    Args:
        panel: long-format logs. Consecutive periods within a unit become the ``(x, u, x_next)``
            transitions the channel is fitted on; gaps are dropped rather than interpolated, so an
            unbalanced panel is fine and a silently invented row is not.
        levers, target, constraints: the decision, in the domain's own names. States are the target
            column followed by each constrained column, in that order.
        adjustment: a :class:`~chc.graph.CausalGraph` to *derive* the adjustment set from, or a
            sequence of column names to *assert* it. Required, and deliberately so --- omitting it
            would default to adjusting for nothing, which is a causal claim, not an absence of one.
        known: physics kept fixed and not estimated. ``None`` means nothing is known and the whole
            vector field is the fitted residual.
        dt: the time step one panel period represents.
        gamma: the sensitivity level the barrier is priced at (§40), passed to
            :func:`chc.plan.certify_safety`. It prices the plan; it does not change it.
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

    Returns:
        A :class:`Prescription`. Read :attr:`DecisionCertificate.identification` before
        :attr:`Prescription.schedule`, which raises when the effect is not identified.

    Raises:
        DecisionError: the decision is mis-specified --- no lever, a column named as both target and
            constraint, or a panel with no consecutive pair of periods to fit a transition on.
        KeyError: a lever, target, constraint or asserted covariate names a column the panel does
            not have. The message lists the panel's columns.

    Each decision point emits one ``logging`` record on ``chc.decision``, keyed by ``chc_event``
    (see :data:`_log`). Nothing is configured here; a caller that wants them calls
    ``logging.basicConfig`` itself. An unidentified effect and a single-precision panel are the two
    that come through at ``WARNING``.
    """
    if not levers:
        raise DecisionError(
            "prescribe needs at least one lever; there is nothing to decide otherwise"
        )
    states = (target.name, *(constraint.state for constraint in constraints))
    if len(set(states)) != len(states):
        raise DecisionError(f"a column is both target and constraint: {states}")
    lever_names = tuple(lever.name for lever in levers)
    for name in (*states, *lever_names):
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
    _log.info(
        "adjustment resolved",
        extra={
            "chc_event": "adjustment",
            "source": "graph" if isinstance(adjustment, CausalGraph) else "asserted",
            "status": resolved.status,
            "covariates": list(resolved.covariates),
        },
    )
    data = _transitions(panel, states=states, levers=lever_names, adjust_for=resolved.covariates)
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
            ),
            model_fit=fit,
            provenance=panel.provenance,
        )

    if not fit.identified:
        resolved = AdjustmentSet(
            resolved.covariates,
            resolved.status,
            resolved.reason
            + "; the estimator was handed no covariate and no instrument, so its channel is the "
            "observational fit, which is the interventional one only if the lever is unconfounded",
        )

    model = HybridDynamics(known=base, residual=fit.residual)
    start = jnp.asarray(data["x0"]) if x0 is None else jnp.asarray(x0)
    u_lo = jnp.array([lever.lo for lever in levers])
    u_hi = jnp.array([lever.hi for lever in levers])
    u_max = float(jnp.max(jnp.maximum(jnp.abs(u_lo), jnp.abs(u_hi))))

    started = time.perf_counter()
    plan = causal_plan(
        model,
        start,
        _cost(states, levers, target),
        dt,
        horizon,
        u_lo,
        u_hi,
        lipschitz=_log_norm(model, start, n_levers),
        model_error=0.0 if tolerance is None else _model_error(fit, u_max),
        tolerance=float("inf") if tolerance is None else tolerance,
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
            "seconds": time.perf_counter() - started,
        },
    )

    barrier = _barrier(states, constraints)
    safety = (
        None
        if barrier is None
        else certify_safety(plan, model, barrier, dt, gamma=gamma, u_max=u_max)
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
    )
    _log.info(
        "decision certified",
        extra={
            "chc_event": "certificate",
            "identification": certificate.identification,
            "gamma_star": certificate.gamma_star,
            "barrier_certified_steps": certificate.barrier_certified_steps,
            "trustworthy_steps": certificate.trustworthy_steps,
        },
    )
    return Prescription(
        levers=tuple(levers),
        target=target.name,
        plan=plan,
        certificate=certificate,
        model_fit=fit,
        provenance=panel.provenance,
    )


# ---- wiring ---------------------------------------------------------------------------------


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
    panel: Panel, *, states: tuple[str, ...], levers: tuple[str, ...], adjust_for: tuple[str, ...]
) -> dict[str, Array]:
    """``(x, u, x_next)`` over consecutive periods within a unit, plus the adjustment columns.

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

    units = sorted({unit for unit, _ in row_of})
    final_rows = np.array(
        [row_of[(unit, max(p for u, p in row_of if u == unit))] for unit in units], dtype=np.int64
    )
    data["x0"] = jnp.mean(stack(states, final_rows), axis=0)
    return data


def _cost(states: tuple[str, ...], levers: Sequence[Lever], target: Target) -> QuadraticCost:
    """Weight the target's own coordinate and price each lever; constrained states are free.

    A constrained state gets weight zero rather than a small one: its bound is enforced by the
    barrier and priced by the certificate, and adding a quadratic pull towards zero would be a
    second, unstated objective.
    """
    weights = jnp.array([target.weight] + [0.0] * (len(states) - 1))
    goal = jnp.array([target.value] + [0.0] * (len(states) - 1))
    q = jnp.diag(weights)
    return QuadraticCost(
        Q=q, R=jnp.diag(jnp.array([lever.unit_cost for lever in levers])), Qf=q, x_target=goal
    )


def _barrier(
    states: tuple[str, ...], constraints: Sequence[Constraint]
) -> Callable[[Array], Array] | None:
    """``h(x) = min_j (bound margins)``, safe where ``h >= 0``; ``None`` when nothing is bounded.

    The minimum is non-smooth where two constraints bind at once, and ``certify_safety`` reads a
    gradient off it. At such a point the gradient is one of the active constraints' rather than a
    convex combination, which is a valid subgradient but not the tightest channel; with a single
    active constraint --- the ordinary case --- it is exact.
    """
    if not constraints:
        return None
    index = {name: position for position, name in enumerate(states)}
    terms = [(index[c.state], c.lo, c.hi) for c in constraints]

    def barrier(x: Array) -> Array:
        margins = []
        for position, lo, hi in terms:
            if lo is not None:
                margins.append(x[position] - lo)
            if hi is not None:
                margins.append(hi - x[position])
        return jnp.min(jnp.stack(margins))

    return barrier


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
    ``channel_error`` is the homoskedastic sandwich diagonal and runs optimistic on the IV path, so
    the tube inherits that optimism and the docstring of :class:`chc.dynamics_id.CausalDynamicsFit`
    is the place that says by how much.
    """
    return 0.0 if fit.channel_error is None else float(fit.channel_error) * max(u_max, 1.0)


def _show(value: object) -> str:
    if value is None:
        return "not evaluated"
    return f"{value:.4g}" if isinstance(value, float) else str(value)
