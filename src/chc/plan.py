"""The one-call spine: plan a control sequence and get its guarantees attached to it.

Everything here already existed -- constrained OC (:mod:`chc.control`), the offline pessimism and
uncertainty penalties (:mod:`chc.support`, :mod:`chc.uncertainty`), and the certified Gronwall
error tube with its safe horizon. What was missing was a single object that carries a plan
*together with* the evidence about where it may be trusted, so a caller cannot walk away with the
actions and leave the certificate behind. See ``plans/21`` §D.

    plan = causal_plan(model, x0, cost, dt=0.1, horizon=20, u_lo=-5.0, u_hi=5.0,
                       lipschitz=0.8, model_error=0.05, tolerance=0.5)
    plan.actions              # the full sequence
    plan.certified_actions    # only the prefix whose error tube is inside tolerance
    plan.certified_horizon    # where that prefix ends

With no safety arguments this is exactly :func:`chc.control.projected_gradient_control` with the
trajectory and cost packaged; each safety argument switches on one existing layer, so the defaults
promise nothing that was not asked for.

**Three modes, deliberately named apart, because "safety" alone does not say which one you get:**

* **plan** -- :func:`causal_plan`. Box constraints in the solve -- optionally intersected with
  linear rows over the whole sequence, such as a budget or a rate limit -- plus an *a-priori*
  Gronwall error tube that says how far ahead the plan may be trusted. The tube is computed from
  ``lipschitz`` and ``model_error``; it does not enter the objective and does not move a single
  action. A :class:`BarrierConstraint` does: passed as ``barrier=``, the audit's own condition is
  held in the solve by augmented-Lagrangian rounds (``docs/adr/0002-barrier-in-the-solve.md``).
* **audit** -- :func:`certify_safety`. Given a barrier and a sensitivity level ``Gamma``, it prices
  a *finished* plan against §40: where along it the safety guarantee survives unmeasured
  confounding, and the largest ``Gamma`` the whole plan tolerates. Read-only by construction.
* **filter** -- :func:`chc.barrier.robust_safety_filter`. It clips one nominal action into the
  certified interval at one state, online, scalar control.

A held barrier binds the *answer*, not every iterate as the box and the rows do, and the solver's
word is not taken for it: a solve stopped by its budget, or a condition no admissible action can
meet, comes back short of the condition, and :attr:`CausalPlan.safety` -- the audit, run on the
finished plan -- says so. The tube is still never imposed inside the optimisation, so a plan can
leave tolerance at step 3; that is why :attr:`CausalPlan.certified_actions` exists.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from chc.barrier import barrier_gamma_star, identification_radius_threshold
from chc.control import (
    Bound,
    LinearConstraint,
    SolverResult,
    SolverStatus,
    _constraint_blocks,
    broadcast_box,
    check_box,
    projected_gradient_solve,
)
from chc.cost import QuadraticCost, total_cost
from chc.dynamics import Dynamics
from chc.integrate import rollout
from chc.support import PenaltyModel, SupportModel, _pessimistic_loop, pessimistic_solve
from chc.uncertainty import (
    certified_horizon,
    confounding_robust_inflation,
    time_varying_rollout_bound,
)

CertificateStatus = Literal["not_evaluated", "uncertified", "partial", "certified"]

_BARRIER_BACKOFF = 1e-6  # how far inside the condition the solve aims, relative to its terms
_BARRIER_ROUNDS = 30  # augmented-Lagrangian rounds, each one descent of at most ``steps``
_PENALTY_GROWTH = 10.0  # applied when a round fails to halve the shortfall
_PENALTY_RANGE = 1e8  # how far the penalty may grow past its start before the rounds give up

_log = logging.getLogger(__name__)
"""Barrier rounds of :func:`causal_plan`, keyed by ``chc_event`` as in :mod:`chc.decision`."""

# Compiled once here: an eager ``lax.scan`` is traced and compiled afresh on every call, which a
# replanning loop would pay at every step.
_trajectory = eqx.filter_jit(rollout)


@dataclass(frozen=True)
class CausalPlan:
    """A control sequence together with the certificate that says how far to trust it.

    ``None`` in the certificate fields means *no error model was supplied*, which is a different
    statement from a certificate that came back empty: the first is "unknown", the second is
    "checked, and nothing holds". Earlier versions returned an all-zero tube and a full
    ``certified_horizon`` in the first case, which reads as a proof of safety over the whole plan
    when nothing was proved at all. Read :attr:`certificate_status` before either field.

    :attr:`solver_status` answers a *different* question from :attr:`certificate_status`, and the
    two are independent. The certificate is about the model: how far the plan can be trusted given
    the error budget. The solver status is about the optimisation: whether the descent reached its
    own stopping rule or merely ran out of budget. A fully certified plan built on an unfinished
    solve is a trustworthy tube around a suboptimal action, and nothing in the tube says so.
    """

    actions: Array  # (horizon, m)
    trajectory: Array  # (horizon + 1, n) nominal rollout under the planning model
    task_cost: float  # cost of the plan, penalties excluded, so weights stay comparable
    uncertainty_tube: Array | None  # (horizon + 1,) per-step error radius; None if not evaluated
    certified_horizon: int | None  # last step inside ``tolerance``; None if not evaluated
    solver_status: SolverStatus  # why the descent stopped -- see :data:`chc.control.SolverStatus`
    solver_iterations: int  # accepted descent steps
    # :func:`certify_safety` run on this plan against the barrier it was solved under; None when no
    # barrier was given. The verdict to read, not the solve's: the rounds aim inside the condition,
    # but a budget-stopped or infeasible solve can come back short of it.
    safety: SafetyCertificate | None = None

    @property
    def certificate_status(self) -> CertificateStatus:
        """Whether the tube was evaluated at all, and if so how much of the plan it covers.

        Derived rather than stored: a status field that can disagree with the horizon it summarises
        is a worse footgun than the one this replaces.
        """
        if self.certified_horizon is None:
            return "not_evaluated"
        if self.certified_horizon == 0:
            return "uncertified"
        return "certified" if self.certified_horizon == self.actions.shape[0] else "partial"

    @property
    def certified_actions(self) -> Array:
        """The prefix of the plan the tube still covers -- what a cautious caller should execute.

        Raises:
            ValueError: if no error model was supplied. Slicing by ``None`` would hand back the
                whole sequence, and an empty one would claim the plan had been checked and failed;
                neither is true, so the question has no answer to return.
        """
        if self.certified_horizon is None:
            raise ValueError(
                "no error model was supplied, so no prefix is certified; pass model_error to "
                "causal_plan (see CausalPlan.certificate_status)"
            )
        return self.actions[: self.certified_horizon]


@dataclass(frozen=True)
class BarrierConstraint:
    """``grad h . xdot >= -alpha * h`` at every planned step, against every effect the data allow.

    The condition :func:`certify_safety` audits, handed to :func:`causal_plan` to hold in the solve
    rather than to price after it. ``barrier`` is ``h`` -- safe where ``h >= 0``, a JAX-traceable
    scalar function; ``alpha`` is the class-K gain; ``gamma`` and ``cvar_gap`` set the
    identification radius ``(gamma-1)/(gamma+1) * cvar_gap`` on the effect, so the condition is the
    worst case over the identified set and ``gamma = 1`` is the ordinary CBF condition. The defaults
    and the calibration burden on ``cvar_gap`` are :func:`certify_safety`'s.

    Raises:
        ValueError: on ``gamma < 1``, which is not a sensitivity level, or ``cvar_gap <= 0``, which
            scales no radius -- here, rather than after the solve that would have used them.
    """

    barrier: Callable[[Array], Array]
    alpha: float = 1.0
    gamma: float = 1.0
    cvar_gap: float = 1.0

    def __post_init__(self) -> None:
        if not self.gamma >= 1.0:
            raise ValueError(f"gamma is a sensitivity level and must be >= 1, got {self.gamma}")
        if not self.cvar_gap > 0.0:
            raise ValueError(
                f"cvar_gap must be positive to scale a sensitivity radius, got {self.cvar_gap}"
            )


def causal_plan(
    model: Dynamics,
    x0: Array,
    cost: QuadraticCost,
    dt: float,
    horizon: int,
    u_lo: Bound,
    u_hi: Bound,
    *,
    support: SupportModel | None = None,
    lam_supp: float = 0.0,
    uncertainty: PenaltyModel | None = None,
    lam_unc: float = 0.0,
    lipschitz: float = 0.0,
    model_error: float = 0.0,
    tolerance: float = float("inf"),
    steps: int = 10_000,
    constraints: Sequence[LinearConstraint] = (),
    barrier: BarrierConstraint | None = None,
    warm_start: Array | None = None,
) -> CausalPlan:
    """Plan under box constraints, optional offline pessimism, and a certified error tube.

    Args:
        support: offline ``(x, u)`` support model; supplying it switches the solve from plain
            projected-gradient OC to :func:`chc.support.pessimistic_control`.
        uncertainty: a ``PenaltyModel`` (ensemble, Wasserstein, confounding radius) weighted by
            ``lam_unc``. Requires ``support`` -- the pessimistic solver evaluates both terms.
        lipschitz, model_error: feed the discrete-Gronwall tube ``e_{k+1} = (1+L*dt)e_k + dt*eps``.
            ``model_error`` is what switches certification on: left at its default the tube would be
            identically zero, so the plan reports ``certificate_status == "not_evaluated"`` and both
            certificate fields come back ``None`` rather than a vacuous full-horizon pass. A
            negative ``lipschitz`` is allowed and meaningful -- it is a contractive log-norm (§30),
            and the tube then shrinks.
        tolerance: tube radius above which the plan stops being certified.
        constraints: linear rows over the whole action sequence
            (:class:`~chc.control.LinearConstraint`), held by every iterate of either solver, not
            only by the answer.
        barrier: a condition to hold at every planned step (:class:`BarrierConstraint`), by
            augmented-Lagrangian rounds around the same descent. Unlike ``constraints`` it binds the
            answer, not every iterate, and the plan comes back with :attr:`CausalPlan.safety` --
            :func:`certify_safety` run on the finished plan, the verdict to read. Its ``gamma_star``
            is priced at the largest magnitude any lever may take, the budget :func:`chc.prescribe`
            uses. A barrier the unconstrained plan already clears changes no action, even when
            that solve stopped on its budget. ``steps`` then caps each descent, of which there are
            at most ``1 + _BARRIER_ROUNDS``.
        warm_start: the ``(horizon, m)`` actions the descent starts from, zeros when omitted --
            typically the last plan shifted one step, which :class:`chc.mpc.RecedingHorizon` passes
            along with the barrier's multipliers. It is projected onto the box and the rows before
            the first step, so it need not be admissible. It moves where the solve starts, and so
            what a solve stopped by ``steps`` returns; a solve that converges on a problem with one
            minimiser lands on it from any start.

    Raises:
        ValueError: if an uncertainty penalty is given without a support model, which would
            silently drop it -- the pessimistic solver is the only consumer of that argument; if
            ``model_error`` is negative, which is not an error budget; if a barrier is given with
            a box that admits no action but zero, which leaves nothing to price it with; or if
            ``warm_start`` is not a finite ``(horizon, m)`` array.
    """
    plan, _ = _plan(
        model,
        x0,
        cost,
        dt,
        horizon,
        u_lo,
        u_hi,
        support=support,
        lam_supp=lam_supp,
        uncertainty=uncertainty,
        lam_unc=lam_unc,
        lipschitz=lipschitz,
        model_error=model_error,
        tolerance=tolerance,
        steps=steps,
        constraints=constraints,
        barrier=barrier,
        warm_start=warm_start,
        multipliers=None,
    )
    return plan


def _plan(
    model: Dynamics,
    x0: Array,
    cost: QuadraticCost,
    dt: float,
    horizon: int,
    u_lo: Bound,
    u_hi: Bound,
    *,
    support: SupportModel | None,
    lam_supp: float,
    uncertainty: PenaltyModel | None,
    lam_unc: float,
    lipschitz: float,
    model_error: float,
    tolerance: float,
    steps: int,
    constraints: Sequence[LinearConstraint],
    barrier: BarrierConstraint | None,
    warm_start: Array | NDArray[np.float64] | None,
    multipliers: NDArray[np.float64] | None,
) -> tuple[CausalPlan, NDArray[np.float64] | None]:
    """:func:`causal_plan`, with the barrier's multipliers passed in and handed back.

    What a receding horizon carries from one plan to the next besides the actions: ``multipliers``
    seed the barrier rounds in place of zeros, and the rounds' last come back -- zeros where the
    barrier was slack, ``None`` with no barrier.
    """
    if uncertainty is not None and support is None:
        raise ValueError("uncertainty penalty requires a support model; it is unused without one")
    if model_error < 0.0:
        raise ValueError(
            f"model_error is a per-step error budget and cannot be negative: {model_error}"
        )
    authority = float(np.max(np.maximum(np.abs(np.asarray(u_lo)), np.abs(np.asarray(u_hi)))))
    if barrier is not None and not authority > 0.0:
        raise ValueError(
            f"a barrier needs a box that admits a nonzero action; the largest is {authority}"
        )

    guess = jnp.zeros((horizon, cost.R.shape[0]))
    if warm_start is not None:
        start = np.asarray(warm_start, dtype=np.float64)
        if start.shape != guess.shape:
            raise ValueError(
                f"warm_start has shape {start.shape}, but a plan is {guess.shape}: one row per "
                "step, one column per lever"
            )
        if not np.isfinite(start).all():
            raise ValueError("warm_start has a non-finite entry, and the descent would start there")
        guess = jax.device_put(start.astype(guess.dtype))  # converted on the host: no compilation
    if support is None:
        solve = projected_gradient_solve(
            model, x0, guess, dt, cost, u_lo, u_hi, steps=steps, constraints=constraints
        )
    else:
        solve = pessimistic_solve(
            model,
            x0,
            guess,
            dt,
            cost,
            support,
            lam_supp,
            u_lo,
            u_hi,
            steps=steps,
            uncertainty=uncertainty,
            lam_unc=lam_unc,
            constraints=constraints,
        )
    actions, status, iterations = solve.actions, solve.status, solve.iterations
    if barrier is not None:
        actions, status, iterations, multipliers = _hold_barrier(
            model,
            x0,
            cost,
            dt,
            u_lo,
            u_hi,
            constraints,
            barrier,
            solve,
            support=support,
            lam_supp=lam_supp,
            uncertainty=uncertainty,
            lam_unc=lam_unc,
            steps=steps,
            multipliers=multipliers,
        )

    lipschitz_seq, error_seq = [lipschitz] * horizon, [model_error] * horizon
    evaluated = model_error > 0.0
    plan = CausalPlan(
        actions=actions,
        trajectory=_trajectory(model, x0, actions, dt),
        task_cost=float(total_cost(model, x0, actions, dt, cost)),
        uncertainty_tube=(
            time_varying_rollout_bound(lipschitz_seq, error_seq, dt) if evaluated else None
        ),
        certified_horizon=(
            certified_horizon(lipschitz_seq, error_seq, dt, tolerance) if evaluated else None
        ),
        solver_status=status,
        solver_iterations=iterations,
    )
    if barrier is None:
        return plan, None
    audit = certify_safety(
        plan,
        model,
        barrier.barrier,
        dt,
        alpha=barrier.alpha,
        gamma=barrier.gamma,
        cvar_gap=barrier.cvar_gap,
        u_max=authority,
    )
    return replace(plan, safety=audit), multipliers


class _HeldBarrier(eqx.Module):
    """The barrier condition as a Powell-Hestenes-Rockafellar term: one multiplier per step.

    It is a :class:`~chc.support.PenaltyModel`, so the penalised descent of :mod:`chc.support`
    minimises it unchanged. That descent has one penalty slot besides the support model's, so
    ``rest`` carries the uncertainty penalty the plan may already have.
    """

    model: Dynamics
    barrier: Callable[[Array], Array] = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    delta: float = eqx.field(static=True)
    multipliers: Array
    weight: Array
    backoff: Array
    rest: PenaltyModel | None = None
    rest_weight: float = eqx.field(static=True, default=0.0)

    def shortfall(self, states: Array, actions: Array) -> Array:
        """``-alpha h - (a + <w, u> - d ||u||)`` per step: positive where the audit fails."""
        h_values, grad_norm, drift, channel = _barrier_terms(
            self.model, self.barrier, states, actions, self.dt
        )
        return -self.alpha * h_values - _guaranteed(drift, channel, grad_norm, actions, self.delta)

    def scale(self, states: Array, actions: Array) -> Array:
        """The largest term of the condition over the plan: what rounding in it is relative to."""
        h_values, grad_norm, drift, channel = _barrier_terms(
            self.model, self.barrier, states, actions, self.dt
        )
        push = jnp.einsum("km,km->k", channel, actions)
        spread = self.delta * grad_norm * _norm(actions)
        return jnp.max(jnp.abs(self.alpha * h_values) + jnp.abs(drift) + jnp.abs(push) + spread)

    def penalty_trajectory(self, xs: Array, us: Array) -> Array:
        excess = self.shortfall(xs, us) + self.backoff
        shifted = jnp.maximum(0.0, self.multipliers + self.weight * excess)
        term = jnp.sum(shifted**2 - self.multipliers**2) / (2.0 * self.weight)
        if self.rest is not None:
            term = term + self.rest_weight * self.rest.penalty_trajectory(xs, us)
        return term


@eqx.filter_jit
def _barrier_state(
    model: Dynamics, x0: Array, actions: Array, dt: float, held: _HeldBarrier
) -> tuple[Array, Array]:
    """``(scale, shortfall)`` of the condition along the plan -- one compiled call per round."""
    states = rollout(model, x0, actions, dt)[:-1]
    return held.scale(states, actions), held.shortfall(states, actions)


def _hold_barrier(
    model: Dynamics,
    x0: Array,
    cost: QuadraticCost,
    dt: float,
    u_lo: Bound,
    u_hi: Bound,
    constraints: Sequence[LinearConstraint],
    barrier: BarrierConstraint,
    start: SolverResult,
    *,
    support: SupportModel | None,
    lam_supp: float,
    uncertainty: PenaltyModel | None,
    lam_unc: float,
    steps: int,
    multipliers: NDArray[np.float64] | None,
) -> tuple[Array, SolverStatus, int, NDArray[np.float64]]:
    """Hold ``barrier`` by augmented-Lagrangian rounds started from the unconstrained ``start``.

    Each round minimises the plan's own objective plus ``sum(max(0, lam + rho c)^2 - lam^2)/2 rho``
    over the same box and rows by the same descent, ``c`` being the per-step shortfall of the
    condition :func:`certify_safety` audits; then ``lam <- max(0, lam + rho c)``, and ``rho`` grows
    tenfold whenever a round fails to halve ``max |max(c, -lam/rho)|``, the measure of infeasibility
    and complementarity together. Scheme and starting penalty are the safeguarded ones of Birgin &
    Martinez (2014); rounds stop once the measure is within half the back-off and the round's
    descent stopped on its own rule, or when ``rho`` would pass ``_PENALTY_RANGE`` times its start.

    ``c`` carries a back-off of ``_BARRIER_BACKOFF`` times the condition's largest term at
    ``start``, so the rounds settle inside the condition: an active step solved exactly sits on it,
    and rounding alone would then fail the audit about half the time. A ``start`` that clears the
    shifted condition is returned untouched, which is what makes a slack barrier free.

    ``multipliers`` seed the rounds in place of zeros -- a receding horizon's last, shifted. The
    penalty is chosen afresh all the same: one grown for the last state left the first round
    ill-conditioned, and cost more descent steps than warm multipliers saved.

    Returns:
        The actions, the status -- ``converged`` for the rounds' own rule, ``max_iterations``
        otherwise -- the accepted descent steps over every descent, ``start``'s included, and the
        multipliers the rounds ended with: zeros when ``start`` came back untouched.
    """
    actions = start.actions
    lo = broadcast_box(u_lo, actions.shape, "u_lo", actions.dtype)
    hi = broadcast_box(u_hi, actions.shape, "u_hi", actions.dtype)
    check_box(lo, hi)
    blocks = _constraint_blocks(constraints, lo, hi, actions.dtype)
    held = _HeldBarrier(
        model=model,
        barrier=barrier.barrier,
        dt=dt,
        alpha=barrier.alpha,
        delta=confounding_robust_inflation(barrier.cvar_gap, 0.0, barrier.gamma),
        multipliers=jnp.zeros(actions.shape[0], actions.dtype),
        weight=jnp.asarray(1.0, actions.dtype),
        backoff=jnp.asarray(0.0, actions.dtype),
        rest=uncertainty,
        rest_weight=lam_unc,
    )
    scale, shortfall = _barrier_state(model, x0, actions, dt, held)
    backoff = _BARRIER_BACKOFF * float(scale)
    excess = np.asarray(shortfall, dtype=np.float64) + backoff
    if float(np.max(excess)) <= 0.0:
        return actions, start.status, start.iterations, np.zeros_like(excess)

    task = float(np.asarray(start.cost_history)[-1])  # on the host: indexing compiles per length
    weight = float(
        np.clip(
            10.0
            * max(1.0, abs(task))
            / max(1.0, 0.5 * float(np.sum(np.maximum(excess, 0.0) ** 2))),
            1e-8,
            1e8,
        )
    )
    ceiling = weight * _PENALTY_RANGE
    multipliers = np.zeros_like(excess) if multipliers is None else multipliers
    status: SolverStatus = "max_iterations"
    iterations, rounds = start.iterations, 0
    measure, previous = math.inf, math.inf
    for rounds in range(1, _BARRIER_ROUNDS + 1):
        held = eqx.tree_at(
            lambda term: (term.multipliers, term.weight, term.backoff),
            held,
            (
                jnp.asarray(multipliers, actions.dtype),
                jnp.asarray(weight, actions.dtype),
                jnp.asarray(backoff, actions.dtype),
            ),
        )
        # lr0 and tol are the solvers' defaults, which the unconstrained solve above also used.
        actions, _, taken = _pessimistic_loop(
            model,
            x0,
            actions,
            dt,
            cost,
            support,
            lam_supp,
            lo,
            hi,
            steps,
            0.2,
            1e-9,
            held,
            1.0,
            blocks,
        )
        iterations += int(taken)
        _, shortfall = _barrier_state(model, x0, actions, dt, held)
        excess = np.asarray(shortfall, dtype=np.float64) + backoff
        measure = float(np.max(np.abs(np.maximum(excess, -multipliers / weight))))
        multipliers = np.maximum(0.0, multipliers + weight * excess)
        _log.debug(
            "barrier round",
            extra={
                "chc_event": "barrier_round",
                "round": rounds,
                "penalty": weight,
                "measure": measure,
                "worst_shortfall": float(np.max(excess)) - backoff,
                "descent_steps": int(taken),
            },
        )
        if measure <= 0.5 * backoff and int(taken) < steps:
            status = "converged"
            break
        if measure > 0.5 * previous:
            if weight * _PENALTY_GROWTH > ceiling:
                break
            weight *= _PENALTY_GROWTH
        previous = measure
    _log.info(
        "barrier held" if status == "converged" else "barrier rounds stopped short",
        extra={
            "chc_event": "barrier",
            "status": status,
            "rounds": rounds,
            "penalty": weight,
            "measure": measure,
            "backoff": backoff,
            "active_steps": int(np.sum(multipliers > 0.0)),
        },
    )
    return actions, status, iterations, multipliers


@dataclass(frozen=True)
class SafetyCertificate:
    """§40 evaluated along a finished plan: how much confounding its safety guarantee survives.

    Two questions that must not be conflated, so both are reported:

    * ``planned_certified`` -- does the action the planner actually chose still clear the barrier
      once the adversary moves the effect inside the identified set? This is about *this* plan.
    * ``gamma_star`` -- does *any* admissible action clear it, and up to which sensitivity level?
      This is about the problem, and is the number an operator can act on. A plan can fail the
      first while the second is comfortable, which says the planner, not the confounding, is at
      fault; the reverse says no controller would have helped.
    """

    barrier_values: Array  # h(x_k) along the nominal trajectory, k = 0 .. horizon-1
    guaranteed_derivative: Array  # a_k + <w_k, u_k> - d_k*||u_k||, the worst case AT the plan
    required: Array  # -alpha * h(x_k); certification asks guaranteed >= required
    planned_certified: Array  # bool per step
    certified_steps: int  # leading certified PREFIX, as in ``certified_horizon``, not a pass count
    gamma_star: float  # weakest step's §40 ceiling; nan if some step certifies at no Gamma at all
    step_gamma_star: tuple[float, ...]  # per-step ceilings, so the weak link is locatable
    radius: float  # largest d_k = Delta(Gamma)*||grad h(x_k)|| applied along the plan


def _norm(vectors: Array) -> Array:
    """Row norms whose gradient at a zero row is zero, a subgradient, where ``jnp.linalg.norm``'s is
    nan. Zero rows are masked *before* the norm, since a ``where`` after it still differentiates
    both branches; the rest go through ``jnp.linalg.norm`` itself, whose fused kernel rounds once
    less than a written-out square root of a sum does."""
    nonzero = jnp.any(vectors != 0.0, axis=-1)
    return jnp.where(
        nonzero, jnp.linalg.norm(jnp.where(nonzero[..., None], vectors, 1.0), axis=-1), 0.0
    )


def _barrier_terms(
    model: Dynamics, barrier: Callable[[Array], Array], states: Array, actions: Array, dt: float
) -> tuple[Array, Array, Array, Array]:
    """``h``, ``||grad h||``, drift ``a = grad h . f(x, 0)``, channel ``w = B^T grad h``, per step.

    The one place the barrier condition is formed: :func:`certify_safety` audits it and
    :func:`causal_plan` holds it through the same code, so the solve cannot drift from its audit.
    """
    times = dt * jnp.arange(states.shape[0])
    zeros = jnp.zeros_like(actions)

    def scalar_h(x: Array) -> Array:
        return jnp.squeeze(barrier(x))

    h_values = jax.vmap(scalar_h)(states)
    grad_h = jax.vmap(jax.grad(scalar_h))(states)
    drift = jnp.einsum("kn,kn->k", grad_h, jax.vmap(model)(times, states, zeros))
    channel = jnp.einsum(  # w = B^T grad h, per step
        "kn,knm->km", grad_h, jax.vmap(jax.jacobian(model, argnums=2))(times, states, zeros)
    )
    return h_values, _norm(grad_h), drift, channel


def _guaranteed(
    drift: Array, channel: Array, grad_norm: Array, actions: Array, delta: float
) -> Array:
    """``a + <w, u> - d ||u||``, ``d = delta ||grad h||``: the worst case at the planned action."""
    return drift + jnp.einsum("km,km->k", channel, actions) - delta * grad_norm * _norm(actions)


def certify_safety(
    plan: CausalPlan,
    model: Dynamics,
    barrier: Callable[[Array], Array],
    dt: float,
    *,
    alpha: float = 1.0,
    gamma: float = 1.0,
    cvar_gap: float = 1.0,
    u_max: float | None = None,
) -> SafetyCertificate:
    """Price a plan's barrier guarantee against a partially identified control effect (§40).

    The plan was made under a *point* estimate of the effect. If that estimate came from confounded
    logs the effect matrix is only set-identified, and given an **operator-norm** identification
    radius ``||B_hat - B||_op <= Delta`` the channel ``w = B^T grad h`` is pinned within
    ``d = Delta * ||grad h||``, so the guaranteed barrier derivative at action ``u`` drops to
    ``a + <w, u> - d*||u||`` (:mod:`chc.barrier`). This walks the nominal trajectory and reports
    where that guarantee survives.

    Args:
        barrier: ``h(x)``, safe where ``h >= 0``. Differentiated with :func:`jax.grad`, so it must
            be a JAX-traceable scalar function.
        alpha: the class-K gain in ``grad h . xdot >= -alpha*h``.
        gamma, cvar_gap: the sensitivity level and the gap it scales, combined into
            ``Delta = (gamma-1)/(gamma+1) * cvar_gap``. ``gamma = 1`` is exact identification --
            zero radius, and this degenerates to the ordinary CBF check. NOTE the calibration
            burden: ``Delta`` is used here as an **operator-norm radius on the effect matrix**, and
            §32's scalar CVaR gap does not by itself establish one. Supplying ``cvar_gap`` in
            matrix operator-norm units is the caller's externally calibrated input, exactly as
            ``gamma`` is.
        u_max: actuation limit, used for ``gamma_star`` only -- that is the *best admissible action*
            question, which needs a budget the plan's own choice does not define. Defaults to the
            largest action norm the plan actually uses, answering the narrower "how much confounding
            does this plan's own authority tolerate".

    Assumes the plant is control-affine and the uncertainty set isotropic (see :mod:`chc.barrier`;
    ``d*||u||`` is exact for a Euclidean ball, an outer bound otherwise), and audits the *model's*
    trajectory pointwise. Forward invariance (Nagumo/Brezis) is assumed as throughout the CBF
    literature; what is certified is the pointwise condition.

    Raises:
        ValueError: if ``cvar_gap`` is non-positive, which would make the radius meaningless and
            :func:`chc.barrier.barrier_gamma_star` uninvertible; or if the actuation budget backing
            ``gamma_star`` is non-positive, which asks how much confounding a controller with no
            authority tolerates.
    """
    if cvar_gap <= 0.0:
        raise ValueError(f"cvar_gap must be positive to scale a sensitivity radius, got {cvar_gap}")

    states, actions = plan.trajectory[:-1], plan.actions
    h_values, grad_norm, drift, channel = _barrier_terms(model, barrier, states, actions, dt)
    delta = confounding_robust_inflation(cvar_gap, 0.0, gamma)
    guaranteed = _guaranteed(drift, channel, grad_norm, actions, delta)
    required = -alpha * h_values
    certified = guaranteed >= required

    authority = u_max if u_max is not None else float(jnp.max(jnp.linalg.norm(actions, axis=1)))
    if authority <= 0.0:
        source = "u_max" if u_max is not None else "the plan's largest action (it never acts)"
        raise ValueError(f"gamma_star needs a positive actuation budget; {source} is {authority}")
    per_step = tuple(
        barrier_gamma_star(
            identification_radius_threshold(
                drift=float(a), channel=float(c), u_max=authority, alpha_h=float(alpha * h)
            ),
            cvar_gap,
            float(n),
        )
        for a, c, h, n in zip(
            drift, jnp.linalg.norm(channel, axis=1), h_values, grad_norm, strict=True
        )
    )
    return SafetyCertificate(
        barrier_values=h_values,
        guaranteed_derivative=guaranteed,
        required=required,
        planned_certified=certified,
        certified_steps=int(jnp.argmin(certified)) if not bool(jnp.all(certified)) else len(states),
        # nan is the weakest possible link -- a step no Gamma certifies must not be skipped by a
        # nanmin and let the plan report a comfortable ceiling it does not have.
        gamma_star=float("nan") if any(np.isnan(per_step)) else min(per_step),
        step_gamma_star=per_step,
        radius=delta * float(jnp.max(grad_norm)),
    )


ModulusSource = Literal["supplied", "measured"]


@dataclass(frozen=True)
class PlanRegretBound:
    """How far a finished plan can be from the best one the same box allows (Result 69, L3.2).

    Result 6's self-certifying bound ``J(U) - J* <= |grad J|^2/(2 mu)`` needs no optimum, which is
    what makes it a certificate rather than a diagnostic. It does not survive a box: at a lever the
    gradient holds against its own bound ``grad J`` is nonzero while the true regret is zero, so
    that bound charges regret to a coordinate that cannot move. Result 13 read the same fact from
    the other side -- an active cap freezes the control and the regret's curvature collapses.

    What replaces it is the same quantity maximised over the *feasible* moves only. Per coordinate
    the certified gain of a move ``d`` is ``-g d - (mu/2) d^2``, and its maximum over
    ``[lo - U, hi - U]`` is the unconstrained one minus a perfect square
    (``validation/constrained_plan_regret.mac`` STEP 3a): never larger, exactly zero at a lever the
    gradient pins, and still finite as ``mu -> 0``, where it degrades to :attr:`frank_wolfe_gap`
    rather than to infinity.

    The bound is on the **planning objective**, which is the question the solver was asked. How far
    the planning model itself is from the plant is the error tube's question
    (:attr:`CausalPlan.uncertainty_tube`), and the two must not be added.
    """

    bound: float  # J(U) - min over the box of J, certified; inf when nothing certifies it
    unconstrained_bound: float  # Result 6's |grad J|^2/(2 mu), for the comparison
    frank_wolfe_gap: float  # the mu-free fallback the box guarantees; bound <= this
    modulus: float  # the mu actually used
    modulus_source: ModulusSource
    gradient_norm: float
    per_lever: tuple[float, ...]  # the bound split by lever; sums to ``bound``
    pinned_actions: int  # coordinates the gradient holds against a bound: exactly free
    ok: bool  # the objective was convex enough over the box for the bound to mean anything


def _objective_modulus(
    objective: Callable[[Array], Array],
    actions: Array,
    lo: Array,
    hi: Array,
    probes: int,
    seed: int,
) -> float:
    """Smallest Hessian eigenvalue of ``J`` over the box, at the plan and at random feasible points.

    A local modulus, and the certificate says so: for a plant affine in the action ``J`` is exactly
    quadratic and one evaluation is the global answer, while for a nonlinear plant this is a sample
    and a caller who can bound the curvature should pass ``modulus`` instead. Sampling is what makes
    a non-convex objective *visible* -- a single evaluation at the plan sits at a solver's stopping
    point, which is the least likely place to find the negative curvature.
    """
    flat = actions.reshape(-1)
    drawn = jax.random.uniform(
        jax.random.PRNGKey(seed),
        (max(probes, 0), flat.size),
        minval=lo.reshape(-1),
        maxval=hi.reshape(-1),
    )

    def curvature(point: Array) -> Array:
        hessian = jax.hessian(lambda v: objective(v.reshape(actions.shape)))(point)
        return jnp.min(jnp.linalg.eigvalsh(0.5 * (hessian + hessian.T)))

    # vmapped rather than looped: the Hessian of a rollout is expensive to TRACE, and a Python
    # loop retraces it once per probe. Batching pays that cost once for the whole sample.
    return float(jnp.min(jax.vmap(curvature)(jnp.concatenate([flat[None, :], drawn]))))


def plan_regret_bound(
    plan: CausalPlan,
    model: Dynamics,
    x0: Array,
    cost: QuadraticCost,
    dt: float,
    u_lo: Bound,
    u_hi: Bound,
    *,
    modulus: float | None = None,
    probes: int = 16,
    seed: int = 0,
) -> PlanRegretBound:
    """RESULT 69 (L3.2) -- the optimality gap of a finished plan, certified from its own gradient.

    Derived in ``validation/constrained_plan_regret.mac``, proved in
    ``proofs/constrained_plan_regret.v``. Reads the plan's actions, not the solver's internals, so
    it prices a plan that came from anywhere -- including one an operator edited by hand.

    Args:
        modulus: the strong-convexity modulus of ``J`` over the box. ``None`` measures it (see
            :func:`_objective_modulus`). A *smaller* modulus gives a *larger* bound and is never
            invalid (STEP 4d), so a conservative one is the safe input; for a plant affine in the
            action ``lambda_min(R)`` is always valid and needs no eigenvalue solve on ``J``.
        probes: random feasible points added to the curvature sample when ``modulus`` is measured.

    The three numbers to read together: :attr:`~PlanRegretBound.bound` is what the box certifies,
    :attr:`~PlanRegretBound.unconstrained_bound` is what Result 6 would have reported at the same
    plan, and :attr:`~PlanRegretBound.frank_wolfe_gap` is what survives if the modulus goes to zero.
    ``ok`` is ``False`` -- and the bound ``inf`` -- exactly when the measured curvature is negative,
    because then no convexity argument applies and a finite number would be a fabrication.

    Raises:
        ValueError: if a supplied ``modulus`` is negative, which is not a curvature.
    """
    if modulus is not None and modulus < 0.0:
        raise ValueError(f"modulus is a curvature and cannot be negative: {modulus}")

    actions = plan.actions
    lo = broadcast_box(u_lo, actions.shape, "u_lo", actions.dtype)
    hi = broadcast_box(u_hi, actions.shape, "u_hi", actions.dtype)

    def objective(us: Array) -> Array:
        return total_cost(model, x0, us, dt, cost)

    gradient = jax.grad(objective)(actions)
    mu = (
        float(modulus)
        if modulus is not None
        else _objective_modulus(objective, actions, lo, hi, probes, seed)
    )
    source: ModulusSource = "supplied" if modulus is not None else "measured"

    below, above = lo - actions, hi - actions  # the feasible moves, as offsets from the plan
    # max{-g d : d in [below, above]}: a line on an interval is largest at an endpoint, and both
    # candidates are >= 0 because 0 is feasible.
    linear = jnp.maximum(-gradient * below, -gradient * above)
    if mu > 0.0:
        step = jnp.clip(-gradient / mu, below, above)
        certified = -gradient * step - 0.5 * mu * step**2
    else:
        certified = linear

    per_lever = tuple(float(v) for v in jnp.sum(certified, axis=0))
    total = float(jnp.sum(certified))
    return PlanRegretBound(
        bound=total if mu >= 0.0 else math.inf,
        unconstrained_bound=(float(jnp.sum(gradient**2)) / (2.0 * mu) if mu > 0.0 else math.inf),
        frank_wolfe_gap=float(jnp.sum(linear)),
        modulus=mu,
        modulus_source=source,
        gradient_norm=float(jnp.linalg.norm(gradient)),
        per_lever=per_lever if mu >= 0.0 else tuple(math.inf for _ in per_lever),
        pinned_actions=int(jnp.sum((certified <= 0.0) & (gradient != 0.0))),
        ok=bool(mu >= 0.0),
    )
