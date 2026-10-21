"""The one-call spine: plan a control sequence and get its guarantees attached to it.

Everything here already existed -- constrained OC (:mod:`chc.control`), the offline pessimism and
uncertainty penalties (:mod:`chc.support`, :mod:`chc.uncertainty`), and the certified Gronwall
error tube with its safe horizon. What was missing was a single object that carries a plan
*together with* the evidence about where it may be trusted, so a caller cannot walk away with the
actions and leave the certificate behind.

    plan = causal_plan(model, x0, cost, dt=0.1, horizon=20, u_lo=-5.0, u_hi=5.0,
                       lipschitz=0.8, model_error=0.05, tolerance=0.5)
    plan.actions              # the full sequence
    plan.certified_actions    # only the prefix whose error tube is inside tolerance
    plan.certified_horizon    # where that prefix ends
    plan.shadow_prices()      # what each constraint row is worth to the plan

With no safety arguments this is exactly :func:`chc.control.projected_gradient_control` with the
trajectory and cost packaged; each safety argument switches on one existing layer, so the defaults
promise nothing that was not asked for.

**Three modes, deliberately named apart, because "safety" alone does not say which one you get:**

* **plan** -- :func:`causal_plan`. Box constraints in the solve -- optionally intersected with
  linear rows over the whole sequence, such as a budget or a rate limit -- plus an *a-priori*
  Gronwall error tube that says how far ahead the plan may be trusted. The tube is computed from
  ``lipschitz`` and ``model_error``, and from the plant's own one-step ``noise`` where it is given;
  it does not enter the objective and does not move a single
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
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import null_space
from scipy.optimize import linprog, lsq_linear, minimize

from chc.adjoint import control_gradient_adjoint
from chc.barrier import barrier_gamma_star, identification_radius_threshold
from chc.causal import _polynomial_features
from chc.control import (
    Blocks,
    Bound,
    LinearConstraint,
    SolverResult,
    SolverStatus,
    _action_units,
    _constraint_blocks,
    _stationarity_tolerance,
    _step_norms,
    broadcast_box,
    check_box,
    projected_gradient_solve,
)
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
from chc.integrate import rk4_step, rollout
from chc.residual import ControlAffineResidual, ZeroResidual, control_affine_features
from chc.response import relax
from chc.support import (
    PenaltyModel,
    SupportModel,
    _augmented,
    _penalised_gradient,
    _pessimistic_loop,
    _radius_norms,
    _smooth,
    pessimistic_solve,
)
from chc.uncertainty import (
    _check_tolerance,
    _linear_tube,
    _noise_covariance,
    _noise_persistence,
    _noise_tube,
    _NoiseRead,
    _tube,
    _within,
    confounding_robust_inflation,
)

if TYPE_CHECKING:
    from chc.dynamics_id import CausalDynamicsFit, Integrator

CertificateStatus = Literal["not_evaluated", "no_tolerance", "uncertified", "partial", "certified"]
"""How far a plan's error tube trusts it: ``not_evaluated`` where no error model gave it a tube,
``no_tolerance`` where the tube was read against an infinite tolerance, which holds every finite
radius and so certifies nothing, and otherwise ``uncertified``, ``partial`` or ``certified`` as
none, some or all of its steps keep the tube within the tolerance."""

_TOLERANCE_DEFAULT = 1.0
"""The ``tolerance`` :func:`causal_plan` and :class:`chc.mpc.RecedingHorizon` take from 0.20.0 when
none is given: one unit of the state, the radius a caller who logs each state in units of the error
accepted on it passes (ADR 0091)."""
PriceStatus = Literal["exact", "weakly_active", "degenerate", "inactive"]
DiagnosticStatus = Literal["within", "beyond", "not_finite", "not_evaluated"]
"""How a plan's stationarity or constraint violation reads against its tolerance: ``within`` it,
``beyond`` it, ``not_finite`` where the number is nan or infinite, as where the rollout left the
float range, and ``not_evaluated`` where it was not read: a plan built by hand."""

_BARRIER_BACKOFF = 1e-6  # how far inside the condition the solve aims, relative to its terms
_BARRIER_ROUNDS = 30  # augmented-Lagrangian rounds, whose descents share a budget of ``steps``
_PENALTY_GROWTH = 10.0  # applied when a round fails to halve the shortfall
_PENALTY_RANGE = 1e8  # how far the penalty may grow past its start before the rounds give up
_DESCENT_TOL = 1e-14  # the share of the cost at the guess a step must buy: the descents' default
_LOOSE_TOL = 1e-9  # a barrier round's ``tol`` while the last one ended short of the condition
_RESTORE_STEPS = 20  # Gauss-Newton projections onto the held condition before it is given up

_log = logging.getLogger(__name__)
"""Barrier rounds of :func:`causal_plan`, keyed by ``chc_event`` as in :mod:`chc.decision`."""

# Compiled once here: an eager ``lax.scan`` is traced and compiled afresh on every call, which a
# replanning loop would pay at every step.
_trajectory = eqx.filter_jit(rollout)
_descent_units = eqx.filter_jit(_action_units)
_objective = eqx.filter_jit(_augmented)
_task_cost = eqx.filter_jit(total_cost)


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

    :meth:`shadow_prices` answers a third: what each constraint row costs the plan, read off the
    problem the plan was solved for.
    """

    actions: Array  # (horizon, m)
    trajectory: Array  # (horizon + 1, n) nominal rollout under the planning model
    task_cost: float  # cost of the plan, penalties excluded, so weights stay comparable
    # (horizon + 1,) per-step error radius, the model error's plus, where the plant's one-step
    # noise was given, the radius holding the noise's part of the gap at each step with probability
    # 0.95; None if not evaluated
    uncertainty_tube: Array | None
    # last step inside ``tolerance``, and before the predicted path's first state that is not a
    # number; None if not evaluated
    certified_horizon: int | None
    solver_status: SolverStatus  # why the descent stopped -- see :data:`chc.control.SolverStatus`
    solver_iterations: int  # accepted descent steps
    # :func:`certify_safety` run on this plan against the barrier it was solved under; None when no
    # barrier was given. The verdict to read, not the solve's: the rounds aim inside the condition,
    # but a budget-stopped or infeasible solve can come back short of it.
    safety: SafetyCertificate | None = None
    # The task cost of the same problem planned on the concave envelope of every response curve in
    # the model that starts convex, the plan this one's descent started from; None when the model
    # holds no such curve or the caller gave a warm start. Where a larger response never costs more
    # and the relaxed problem is convex -- a budget spread over curves of spend -- no plan costs
    # less, so ``task_cost - relaxed_cost`` bounds how far this plan is from the best one.
    relaxed_cost: float | None = None
    # How near the plan is to a first-order optimum of the problem it was solved for, and how far
    # it breaks that problem's box and rows, each read at the returned actions in the descent's own
    # variables, so that each reads the same in any units of the levers, the states, the rows or the
    # cost (:attr:`stationarity_status`, :attr:`violation_status`). None where not read.
    stationarity: float | None = None
    stationarity_tolerance: float | None = None
    constraint_violation: float | None = None
    violation_tolerance: float | None = None
    # the radius ``certified_horizon`` was read against; None on a plan built by hand
    tolerance: float | None = None
    _problem: _PlanProblem | None = field(default=None, repr=False, compare=False)

    @property
    def certificate_status(self) -> CertificateStatus:
        """Whether the tube was evaluated at all, and if so how much of the plan it covers
        (:data:`CertificateStatus`).

        Derived rather than stored: a status field that can disagree with the horizon it summarises
        is a worse footgun than the one this replaces. An infinite :attr:`tolerance` reads
        ``no_tolerance``: up to 0.18.0 it read ``certified`` on every plan whose tube stayed finite,
        a pass no tube could fail.
        """
        if self.certified_horizon is None:
            return "not_evaluated"
        if self.tolerance is not None and not math.isfinite(self.tolerance):
            return "no_tolerance"
        if self.certified_horizon == 0:
            return "uncertified"
        return "certified" if self.certified_horizon == self.actions.shape[0] else "partial"

    @property
    def stationarity_status(self) -> DiagnosticStatus:
        """How :attr:`stationarity` reads against :attr:`stationarity_tolerance`.

        :attr:`stationarity` is the plan's KKT residual (ADR 0097). The descent steps in
        ``v = sigma * u`` (ADR 0059), where ``sigma_kj^2`` is the cost's curvature along lever ``j``
        at step ``k`` over ``|J|`` at the guess, zeros clipped to the box: a pure number in any
        units of the lever and the cost. There the gradient is read as a share of ``|J|``, each row
        is divided by its norm, and the residual is the least norm of the gradient plus a push of
        the binding rows and box sides with the signs their bounds allow, over
        ``max(1, |gradient on the free actions|)``: zero exactly at a KKT point, positive where a
        multiplier has the wrong sign. Off the constraints its square is about twice the share of
        ``|J|`` that one more Newton step along each action alone would buy. What binds is
        :meth:`shadow_prices`' rule, read in ``v``.

        The tolerance is what the descent can see: it stops where no step buys more than
        ``max(1e-14, 4 eps)`` of ``|J|``, ``eps`` the actions' precision, which leaves a residual of
        about the root of twice that; ten times it allows for a curvature off one along the
        residual. Where the curvature is further off, as under a steep penalty, the descent reads
        this residual itself and stops only within the tolerance, or where the gradients at the
        ends of every trial read no fall either. ``beyond`` it, the descent stopped on its budget
        or short of the optimum, and the plan, its prices and its regret are a stopped solve's.
        On a plan held under a barrier, each step whose condition the rounds ended with a positive
        multiplier on binds too, and pushes along the gradient of its shortfall in ``v``: the
        residual is the augmented Lagrangian's, ``grad J + sum_k mu_k grad c_k`` with ``mu >= 0``,
        read at the condition the rounds held, a relative 1e-6 inside the audit's.
        """
        return _diagnostic_status(self.stationarity, self.stationarity_tolerance)

    @property
    def violation_status(self) -> DiagnosticStatus:
        """How :attr:`constraint_violation` reads against :attr:`violation_tolerance`.

        :attr:`constraint_violation` is the largest distance in ``v``, the variables of
        :attr:`stationarity_status`, from the plan to a side of its box or to a bound of one of its
        ``constraints`` rows that it is past: a row's excess over the row's norm in ``v``. Zero
        where the plan meets every one. Its tolerance is the one the descent's projection holds
        each side and row to, ``256 eps (1 + max|v|)``, times four for the plan's own rounding and
        a projection started from another guess. The barrier's condition is not among them:
        :attr:`safety` audits it.
        """
        return _diagnostic_status(self.constraint_violation, self.violation_tolerance)

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

    def shadow_prices(self, tolerance: float | None = None) -> ShadowPrices:
        """What each row of the plan's ``constraints`` is worth: its KKT multiplier at this plan.

        The gradient is of the objective the solve minimised -- the task cost, plus the pessimism
        penalties when a support model was given -- at the returned actions. A row binds within
        ``tolerance * (1 + |bound|)`` of its bound, and so does a box bound. The box's multipliers
        absorb every action at a bound, so the rows' multipliers solve the stationarity conditions
        on the free actions alone, by least squares. Where the binding rows are dependent there,
        the multiplier is not unique, and two linear programs give each row's range instead.

        Read ``residual`` and ``dual_feasible`` before the prices. A solve stopped by its budget is
        not at a KKT point, and its prices answer a question that has none. A price is the marginal
        fall in the optimum per unit its row is relaxed: of the optimum, on a plan convex in its
        actions -- a linear model and a quadratic cost -- and of the local optimum the solve found
        otherwise. These are the stated rows' multipliers, not the barrier rounds' that
        :class:`chc.mpc.RecedingHorizon` carries between steps.

        Computed when asked: one gradient, a least-squares solve and, for a degenerate row, two
        linear programs. ``tolerance`` defaults to ``max(1e-6, 1e3 * eps)`` of the actions' dtype.

        Raises:
            ValueError: on a plan held under a barrier, whose condition is a constraint of its own,
                with multipliers that would enter every row's price; or on a plan that carries no
                problem, as one built by hand does.
        """
        problem = self._problem
        if problem is None:
            raise ValueError(
                "this plan carries no problem to price; causal_plan and RecedingHorizon attach one"
            )
        if problem.barrier:
            raise ValueError(
                "the plan was held under a barrier, a constraint whose multipliers would enter "
                "every row's price; the prices of a barrier-held plan are not built"
            )
        shape, dtype = self.actions.shape, self.actions.dtype
        if tolerance is None:
            tolerance = max(1e-6, 1e3 * float(jnp.finfo(dtype).eps))
        present = [row for row in problem.constraints if row.matrix.shape[0]]
        size = int(np.prod(shape))
        lo = broadcast_box(problem.u_lo, shape, "u_lo", dtype)
        hi = broadcast_box(problem.u_hi, shape, "u_hi", dtype)
        return _row_prices(
            problem.gradient(self.actions),
            np.asarray(self.actions, dtype=np.float64).ravel(),
            np.vstack([row.matrix for row in present]) if present else np.zeros((0, size)),
            np.concatenate([row.lower for row in present]) if present else np.zeros(0),
            np.concatenate([row.upper for row in present]) if present else np.zeros(0),
            np.asarray(lo, np.float64).ravel(),
            np.asarray(hi, np.float64).ravel(),
            tolerance,
            problem.gradient(_guess(lo, hi)),
        )

    def decision_weight(self, tolerance: float | None = None) -> DecisionWeight:
        """What an error in the one-step channel costs this plan, to second order. *Experimental.*

        If the plant's one-step map is the planning model's plus ``E u``, each entry of the channel
        off by ``E`` as :func:`chc.gate.channel_drift_evalues` watches them, the plan loses
        ``vec(E)' W vec(E) / 2`` against the plan that knew ``E``, up to a term cubic in ``E``. By
        the envelope theorem ``W = J_Eu M^-1 J_uE``, with ``M`` the Hessian of the task cost in
        the directions the plan may move and ``J_uE`` its mixed derivative in those directions and
        ``E``. The directions are the actions off the box, along the binding rows, with
        :meth:`shadow_prices`' rule for what binds. ``u`` is the action the field takes: a lever
        :func:`chc.decision.prescribe` keeps on the log's rule of the state enters at its rule's
        level, not at its column of :attr:`actions`.

        ``W`` is the regret's curvature at ``E = 0``, and three things bound how far it reaches:

        * **The active set.** ``W`` holds while the bounds and rows that bind stay binding. At one
          met with a zero multiplier, counted in ``weakly_active``, the regret is piecewise
          quadratic, and ``W``, with the bound held, is its lower branch: freeing a direction can
          only raise it.
        * **The cubic term.** On the scalar plan of ``validation/plan_decision_weight.mac``, whose
          one-step channel is 0.117, the regret's third derivative in ``E`` is -280 where ``W`` is
          9.2, and the quadratic is within 10% of the exact regret only for ``E`` in
          ``[-0.0096, 0.0102]``, about 8% of the channel.
        * **A stationary plan.** ``residual`` is the stationarity miss along the free directions.
          A solve stopped by its budget carries a first-order term that ``W`` does not.

        ``W`` vanishes where the plan's response to the channel is flat, as on one step at
        ``r = gamma^2 qf``, and the regret there is fourth order, not nil.

        Computed when asked, in the plan's precision: one Hessian of the task cost in the actions,
        ``(H m)^2`` entries, and one mixed derivative. ``tolerance`` is :meth:`shadow_prices`'.

        Raises:
            ValueError: on a plan held under a barrier, whose condition moves with the channel; on
                one solved with pessimism penalties, whose regret on the task cost has a
                first-order term; on one that carries no problem, as one built by hand does; or
                where ``M`` is not positive definite, so the plan is not a strict local minimum
                and its regret is not quadratic in ``E``; or where the task cost's derivatives at
                the plan are not finite.
        """
        problem = _weighable(self._problem)
        shape = self.actions.shape
        states, actions = problem.x0.shape[0], shape[1]

        def cost(u: Array, change: Array) -> Array:
            return _perturbed_task_cost(problem, u.reshape(shape), change.reshape(states, actions))

        matrix, free, weakly_active, residual = _regret_curvature(
            problem, self.actions, cost, states * actions, tolerance
        )
        return DecisionWeight(
            matrix=matrix,
            channel_shape=(states, actions),
            free=free,
            weakly_active=weakly_active,
            residual=residual,
        )

    def planned_cost_bias(
        self,
        covariance: ArrayLike,
        *,
        error_mean: ArrayLike | None = None,
        tolerance: float | None = None,
    ) -> PlannedCostBias:
        """How far the plan's own cost reads from what it costs the plant, in expectation over the
        estimate's error, and the cost corrected for it. *Experimental.*

        The plan was made on an estimate of its one-step channel, off the plant's by ``e`` with
        ``covariance = S = E[e e']``, ``e`` the estimate less the plant's channel, raveled as
        :meth:`decision_weight` ravels it. Its planned cost, the task cost on the estimate, misses
        what the plan pays on the plant by

            E[J_true - J_model] = tr(W S) - tr(J_ee S) / 2 - J_e . E[e]

        to second order (``validation/planned_cost_bias.mac``), with ``W`` the decision weight,
        ``J_ee`` the task cost's Hessian in the channel at the fixed plan and ``J_e`` its gradient.
        The first term is the optimiser's curse and is never negative: the plan fits its estimate's
        error. The second is the curvature of the planned cost at the fixed plan, never positive
        where that cost is convex in the channel. So the sum has no fixed sign. On one step of
        ``x1 = a x0 + b u`` at the cost ``q x1^2 + r u^2`` it is ``a^2 q^2 x0^2 (2 r^2 - 5 b^2 q r
        + b^4 q^2) / (r + b^2 q)^3`` per unit of ``S``, which is negative, a planned cost that
        reads high, for ``q b^2 / r`` between ``(5 - sqrt 17)/2 = 0.438`` and ``(5 + sqrt 17)/2 =
        4.562``, and positive outside. The third term is the estimate's own bias, ``error_mean``,
        0 where it is not given; a cross-fitted channel's is of the order of ``S``.

        Each term is read at the plan and the estimate, not at the plant's channel, which the
        expansion is about: the difference is of higher order. It holds while the plan's active set
        does not move with the error, so a plan that meets a bound or a row with a zero multiplier
        is refused; and as far as ``W`` does, for an error small against the channel (see
        :meth:`decision_weight`). ``residual`` is the stationarity miss along the free directions:
        a solve stopped by its budget carries a first-order term that ``W`` does not.

        :func:`chc.misspecification.planned_cost_bias` reads ``S`` off a fitted channel's
        influence, over every parameter the fit holds, the drift's included.

        Raises:
            ValueError: on a ``covariance`` that is missing, not a finite ``(n m, n m)`` matrix, or
                not symmetric positive semidefinite; an ``error_mean`` that is not a finite ``(n,
                m)`` matrix; a plan that meets a bound or a row with a zero multiplier; and whatever
                :meth:`decision_weight` refuses.
        """
        problem = _weighable(self._problem)
        shape = self.actions.shape
        states, actions = problem.x0.shape[0], shape[1]

        def cost(u: Array, change: Array) -> Array:
            return _perturbed_task_cost(problem, u.reshape(shape), change.reshape(states, actions))

        matrix = _covariance(covariance, states * actions)
        mean = None
        if error_mean is not None:
            mean = np.asarray(error_mean, dtype=np.float64)
            if mean.shape != (states, actions) or not np.all(np.isfinite(mean)):
                raise ValueError(
                    f"error_mean must be a finite {(states, actions)} matrix, the one-step "
                    f"channel's shape, got shape {mean.shape}"
                )
        terms = _bias_terms(problem, self.actions, cost, states * actions, tolerance)
        return terms.bias(matrix, mean)


def _covariance(covariance: ArrayLike | None, size: int) -> NDArray[np.float64]:
    """``covariance`` as a symmetric ``(size, size)`` matrix, refused where it is missing, not
    finite, not symmetric to rounding, or has an eigenvalue below rounding of its largest."""
    if covariance is None:
        raise ValueError(
            "no covariance of the channel's error was given, so the bias has no S to read"
        )
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"covariance must be a finite ({size}, {size}) matrix, got shape {matrix.shape}"
        )
    scale = float(np.abs(matrix).max(initial=0.0))
    slack = 1e3 * np.finfo(np.float64).eps * scale
    if float(np.abs(matrix - matrix.T).max(initial=0.0)) > slack:
        raise ValueError("covariance must be symmetric, and it is not to rounding")
    matrix = (matrix + matrix.T) / 2.0
    if size and float(np.linalg.eigvalsh(matrix)[0]) < -slack:
        raise ValueError(
            "covariance must be positive semidefinite, and it has an eigenvalue below zero"
        )
    return matrix


@dataclass(frozen=True)
class _BiasTerms:
    """What :class:`PlannedCostBias` reads off a plan, before ``S``: the decision weight, the task
    cost's gradient and Hessian in the perturbation at the fixed plan, and the cost itself."""

    weight: NDArray[np.float64]
    gradient: NDArray[np.float64]
    hessian: NDArray[np.float64]
    planned_cost: float
    free: int
    residual: float

    def optimism(self, covariance: NDArray[np.float64]) -> float:
        return float(np.trace(self.weight @ covariance))

    def bias(
        self,
        covariance: NDArray[np.float64],
        mean: NDArray[np.float64] | None,
        unseen: int | None = None,
    ) -> PlannedCostBias:
        return PlannedCostBias(
            planned_cost=self.planned_cost,
            optimism=self.optimism(covariance),
            curvature=-0.5 * float(np.trace(self.hessian @ covariance)),
            first_order=0.0 if mean is None else -float(self.gradient @ mean.ravel()),
            free=self.free,
            residual=self.residual,
            unseen=unseen,
        )


def _bias_terms(
    problem: _PlanProblem,
    actions: Array,
    cost: Callable[[Array, Array], Array],
    size: int,
    tolerance: float | None,
) -> _BiasTerms:
    """:class:`_BiasTerms` for a plan at ``actions`` whose task cost under a perturbation of
    ``size`` entries is ``cost(actions.ravel(), change)``; refused where the plan meets a bound or a
    row with a zero multiplier, whose regret is piecewise and whose bias is then not one
    expansion."""
    weight, free, weakly_active, residual = _regret_curvature(
        problem, actions, cost, size, tolerance
    )
    if weakly_active:
        raise ValueError(
            f"the plan meets {weakly_active} bound(s) or row(s) with a zero multiplier, so its "
            "active set moves with the channel's error, either way, and the planned cost's bias "
            "is not one expansion"
        )
    flat = actions.ravel()
    still = jnp.zeros(size, dtype=actions.dtype)
    gradient = np.asarray(jax.grad(cost, argnums=1)(flat, still), dtype=np.float64)
    hessian = np.asarray(jax.hessian(cost, argnums=1)(flat, still), dtype=np.float64)
    planned = float(cost(flat, still))
    if not (np.all(np.isfinite(gradient)) and np.all(np.isfinite(hessian))):
        raise ValueError(
            "the task cost's derivatives in the channel at the plan are not finite: the planned "
            "cost has no bias to read"
        )
    return _BiasTerms(
        weight=weight,
        gradient=gradient,
        hessian=(hessian + hessian.T) / 2.0,
        planned_cost=planned,
        free=free,
        residual=residual,
    )


def _weighable(problem: _PlanProblem | None) -> _PlanProblem:
    """The plan's problem, refused where its regret in an error of the model is not quadratic."""
    if problem is None:
        raise ValueError(
            "this plan carries no problem to weigh; causal_plan and RecedingHorizon attach one"
        )
    if problem.barrier:
        raise ValueError(
            "the plan was held under a barrier, whose condition moves with the channel; the "
            "decision weight of a barrier-held plan is not built"
        )
    weighted = problem.lam_supp != 0.0 or (
        problem.uncertainty is not None and problem.lam_unc != 0.0
    )
    if problem.support is not None and weighted:
        raise ValueError(
            "the plan minimised the task cost plus pessimism penalties, so an error in the "
            "channel costs its task cost at first order and the regret is not quadratic"
        )
    return problem


def _regret_curvature(
    problem: _PlanProblem,
    actions: Array,
    cost: Callable[[Array, Array], Array],
    size: int,
    tolerance: float | None,
) -> tuple[NDArray[np.float64], int, int, float]:
    """The regret's Hessian in a perturbation of the model, by the envelope theorem, for a plan at
    ``actions`` whose task cost under a perturbation of ``size`` entries is
    ``cost(actions.ravel(), change)``: ``W = J' M^-1 J`` over the directions the plan may move.

    Returns ``W``, how many directions the plan may move in, how many bounds and rows it meets with
    a zero multiplier, and its stationarity miss along the free directions, relative.
    """
    shape, dtype = actions.shape, actions.dtype
    if tolerance is None:
        tolerance = max(1e-6, 1e3 * float(jnp.finfo(dtype).eps))
    flat = actions.ravel()
    still = jnp.zeros(size, dtype=dtype)
    gradient = np.asarray(jax.grad(cost)(flat, still), dtype=np.float64)
    hessian = np.asarray(jax.hessian(cost)(flat, still), dtype=np.float64)
    mixed = np.asarray(jax.jacfwd(jax.grad(cost), argnums=1)(flat, still), dtype=np.float64)
    # numpy's eigvalsh reads a nan matrix as finite, and LAPACK fails on an infinite one
    if not all(np.isfinite(d).all() for d in (gradient, hessian, mixed)):
        raise ValueError(
            "the task cost's derivatives at the plan are not finite, as where the model's rollout "
            "leaves the float range: the regret has no curvature to read"
        )
    present = [row for row in problem.constraints if row.matrix.shape[0]]
    n_actions = flat.size
    matrix = np.vstack([row.matrix for row in present]) if present else np.zeros((0, n_actions))
    box_lo = broadcast_box(problem.u_lo, shape, "u_lo", dtype)
    box_hi = broadcast_box(problem.u_hi, shape, "u_hi", dtype)
    lo, hi = np.asarray(box_lo, np.float64).ravel(), np.asarray(box_hi, np.float64).ravel()
    kkt = _kkt(
        gradient,
        np.asarray(flat, dtype=np.float64),
        matrix,
        np.concatenate([row.lower for row in present]) if present else np.zeros(0),
        np.concatenate([row.upper for row in present]) if present else np.zeros(0),
        lo,
        hi,
        tolerance,
        np.asarray(jax.grad(cost)(_guess(box_lo, box_hi).ravel(), still), dtype=np.float64),
    )
    free = kkt.free
    # A box of zero width fixes its action: no change of the channel moves it off, so it is no
    # branch of the regret, whatever its pull.
    pinned = (kkt.at_hi | kkt.at_lo) & (hi > lo)
    weak_rows = np.abs(kkt.multipliers) * np.linalg.norm(kkt.columns[free], axis=0)
    weakly_active = int(np.sum(np.abs(kkt.remainder[pinned]) <= kkt.slack)) + int(
        np.sum(weak_rows <= kkt.slack)
    )
    # The directions the plan may move: the free actions, less what keeps a binding row bound.
    basis = np.eye(n_actions)[:, free]
    if kkt.active.size and basis.shape[1]:
        basis = basis @ null_space(matrix[kkt.active][:, free])
    curvature = basis.T @ hessian @ basis
    eigenvalues = np.linalg.eigvalsh(curvature) if basis.shape[1] else np.zeros(0)
    # a share of the largest eigenvalue, not of at least 1: a cost in units that put its whole
    # curvature below 2e-13 was refused at a strict minimum
    if eigenvalues.size and eigenvalues[0] <= 1e3 * np.finfo(np.float64).eps * float(
        np.abs(eigenvalues).max()
    ):
        raise ValueError(
            f"the task cost's Hessian along the plan's free directions has eigenvalue "
            f"{eigenvalues[0]:.4g}: the plan is not a strict local minimum, and its regret is "
            "not quadratic in the channel"
        )
    reduced = basis.T @ mixed
    weight = reduced.T @ np.linalg.solve(curvature, reduced) if basis.shape[1] else None
    pull = gradient[free]
    return (
        np.zeros((size, size)) if weight is None else (weight + weight.T) / 2,
        int(basis.shape[1]),
        weakly_active,
        float(np.linalg.norm(basis.T @ gradient)) / max(1.0, float(np.linalg.norm(pull))),
    )


class _Rule(eqx.Module):
    """Levers set from the state alone, on a rule: a polynomial of the standardised state, clipped
    to their boxes, as :func:`chc.decision.prescribe` reads one off a log (ADR 0054). The plan's
    field, its price and its decision weight read the action through :meth:`read`, so each sees the
    action the plan takes."""

    centre: Array  # (n,)
    shift: Array  # (n,)
    factor: Array  # (n,)
    coefficients: Array  # (features, r)
    lo: Array  # (r,)
    hi: Array  # (r,)
    levers: tuple[int, ...] = eqx.field(static=True)
    degree: int = eqx.field(static=True)

    def levels(self, x: Array) -> Array:
        """The ruled levers' levels at ``x``."""
        standard = ((x - self.centre - self.shift) * self.factor)[None, :]
        rule = _polynomial_features(standard, self.degree)[0] @ self.coefficients
        return jnp.clip(rule, self.lo, self.hi)

    def read(self, x: Array, u: Array) -> Array:
        """The action taken at ``x``: ``u`` with the ruled levers on their rule."""
        return u.at[jnp.array(self.levers)].set(self.levels(x))


class _Ruled(_SetAtStep):
    """A field whose ruled levers follow their rule. Their columns of the plan's actions are not
    read. A step sets them at the state it starts from and holds them over the step, as the log
    held them over a period; a call reads the rule at the state it is given."""

    dynamics: Dynamics
    rule: _Rule

    def read(self, x: Array, u: Array) -> Array:
        """The action the field reads at ``x``."""
        return self.rule.read(x, u)

    def at_step(self, x: Array, u: Array) -> tuple[Dynamics, Array]:
        return self.dynamics, self.rule.read(x, u)

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return self.dynamics(t, x, self.rule.read(x, u))


class _RuledCost(QuadraticCost):
    """The cost at the action :class:`_Ruled` reads: a ruled lever is priced at its rule's level,
    not at its column of the plan, which nothing reads."""

    rule: _Rule

    def running(self, x: Array, u: Array, target: Array | None = None) -> Array:
        return QuadraticCost.running(self, x, self.rule.read(x, u), target)


def _taken(model: Dynamics, x: Array, u: Array) -> Array:
    """The action ``model``'s field takes at ``x`` for the plan's ``u``."""
    return model.read(x, u) if isinstance(model, _Ruled) else u


def _integrator(model: Dynamics) -> Integrator:
    """The one-step map :func:`chc.integrate.rk4_step` steps ``model`` by: the Euler map's where
    the field is fitted to it (:class:`~chc.dynamics._EulerStep`), RK4's elsewhere. The tube reads
    it here, so it is carried by the step the plan rolls out by."""
    inner = model.dynamics if isinstance(model, _Ruled) else model
    return "euler" if isinstance(inner, _EulerStep) else "rk4"


def _perturbed_task_cost(problem: _PlanProblem, actions: Array, change: Array) -> Array:
    """The task cost of ``actions`` when each step's one-step map is the model's plus
    ``change @ u``, ``u`` the action the field takes: the rollout :func:`chc.integrate.rollout`
    makes, with the channel moved."""
    targets = problem.cost.targets(actions.shape[0])

    def body(carry: tuple[Array, Array], u: Array) -> tuple[tuple[Array, Array], Array]:
        t, x = carry
        x_next = rk4_step(problem.model, t, x, u, problem.dt) + change @ _taken(problem.model, x, u)
        return (t + problem.dt, x_next), x_next

    start = (jnp.asarray(0.0, dtype=problem.x0.dtype), problem.x0)
    _, xs = jax.lax.scan(body, start, actions)
    xs = jnp.concatenate([problem.x0[None, :], xs], axis=0)
    running = jnp.sum(jax.vmap(problem.cost.running)(xs[:-1], actions, targets[:-1]))
    return running + problem.cost.terminal(xs[-1], targets[-1])


@dataclass(frozen=True)
class RowPrice:
    """What one constraint row is worth to the plan: how much its objective falls per unit the
    row is relaxed -- an upper bound raised, a lower bound lowered.

    ``status`` says how far ``price`` can be read:

    * ``exact`` -- the row binds, and the binding rows are independent on the actions the box
      leaves free (LICQ), so the multiplier is unique.
    * ``weakly_active`` -- the row sits at its bound with a multiplier of zero: relaxing it buys
      nothing to first order, and tightening it costs nothing to first order.
    * ``degenerate`` -- the binding rows are dependent on the free actions, so the multiplier is a
      set: ``price`` is ``None`` and ``interval`` its range. Relaxing the bound buys
      ``interval[0]`` a unit and tightening it costs ``interval[1]``, both widened by the
      stationarity slack.
    * ``inactive`` -- the row has room, and is worth nothing at the margin.

    An equality row's price takes either sign: the fall per unit its value rises.
    """

    status: PriceStatus
    price: float | None
    interval: tuple[float, float]  # (price, price) unless degenerate


@dataclass(frozen=True)
class ShadowPrices:
    """The rows' prices at a plan, with what it takes to trust them.

    See :meth:`CausalPlan.shadow_prices`.
    """

    rows: tuple[RowPrice, ...]  # one per row of the plan's constraints, stacked in the order given
    residual: float  # stationarity miss on the free actions, over max(1, |gradient there|)
    # Of the binding rows on the free actions: 0 when they are dependent, inf when none binds.
    smallest_singular_value: float
    dual_feasible: bool  # every multiplier has the sign its bound allows, to the tolerance


@dataclass(frozen=True)
class DecisionWeight:
    """What an error in the one-step channel costs a plan, to second order. *Experimental.*

    See :meth:`CausalPlan.decision_weight`.
    """

    # (n m, n m), symmetric positive semidefinite: the channel's entry (i, j), state i and action j,
    # is index i m + j, the order chc.gate.channel_drift_evalues reads the entries in
    matrix: NDArray[np.float64]
    channel_shape: tuple[int, int]  # (n, m)
    free: int  # the directions the plan may move in: actions off the box, less the binding rows
    # bounds and rows the plan meets with a zero multiplier; a box of zero width fixes its action
    # and is none
    weakly_active: int
    residual: float  # stationarity miss along the free directions, over max(1, |gradient there|)

    def regret(self, change: ArrayLike) -> float:
        """``vec(E)' W vec(E) / 2`` at ``change = E``: what the plan loses, to second order,
        against the plan that knew its one-step channel was off by ``E``.

        Raises:
            ValueError: on a ``change`` that is not a finite ``channel_shape`` matrix.
        """
        entries = np.asarray(change, dtype=np.float64)
        if entries.shape != self.channel_shape or not np.all(np.isfinite(entries)):
            raise ValueError(
                f"change must be a finite {self.channel_shape} matrix, the one-step channel's "
                f"shape, got shape {entries.shape}"
            )
        flat = entries.ravel()
        return float(flat @ self.matrix @ flat) / 2.0


CostBiasDirection = Literal["optimistic", "pessimistic", "unbiased", "unknown"]
"""Which way a planned cost reads against what the plan pays the plant, in expectation:
``optimistic`` where it reads low, a positive bias, ``pessimistic`` where it reads high, and
``unknown`` where the bias is nan."""


@dataclass(frozen=True)
class PlannedCostBias:
    """A plan's planned cost, its bias against what the plan pays the plant, with its sign, and the
    cost corrected for it, to second order in the estimate's error. *Experimental.*

    See :meth:`CausalPlan.planned_cost_bias`.
    """

    planned_cost: float  # the task cost the plan reads on the model it was made on
    # tr(W S): the optimiser's curse, what the plan pays for fitting its estimate's error; >= 0
    optimism: float
    # -tr(J_ee S) / 2: the planned cost's curvature in the channel at the fixed plan; <= 0 where it
    # is convex there
    curvature: float
    # -J_e . E[e]: the estimate's own bias carried at first order; 0 if not given
    first_order: float
    free: int  # the directions the plan may move in: actions off the box, less the binding rows
    residual: float  # stationarity miss along the free directions, over max(1, |gradient there|)
    # directions the log's actions never moved that the plan's regret weighs, where the error was
    # read off a fit (:func:`chc.misspecification.planned_cost_bias`): the fit's covariance holds no
    # error along them, so the bias leaves them out. None where no fit was read.
    unseen: int | None = None

    @property
    def bias(self) -> float:
        """``E[J_true - J_model]``: what the plan pays the plant less what it reads, in
        expectation; positive where the planned cost reads low."""
        return self.optimism + self.curvature + self.first_order

    @property
    def corrected_cost(self) -> float:
        """The planned cost plus its bias: what the plan pays the plant, in expectation."""
        return self.planned_cost + self.bias

    @property
    def direction(self) -> CostBiasDirection:
        """Which way the planned cost reads (:data:`CostBiasDirection`). A nan bias, as from a term
        a plan that left the float range reads, is ``unknown``: up to 0.18.0 it read
        ``unbiased``, failing both comparisons."""
        bias = self.bias
        if math.isnan(bias):
            return "unknown"
        return "optimistic" if bias > 0.0 else "pessimistic" if bias < 0.0 else "unbiased"


@dataclass(frozen=True)
class _PlanProblem:
    """What :meth:`CausalPlan.shadow_prices` needs of the solve, kept by the plan it produced."""

    model: Dynamics
    x0: Array
    cost: QuadraticCost
    dt: float
    u_lo: Bound
    u_hi: Bound
    constraints: tuple[LinearConstraint, ...]
    support: SupportModel | None
    lam_supp: float
    uncertainty: PenaltyModel | None
    lam_unc: float
    barrier: bool
    # the condition the barrier rounds held, with their last multipliers; None without a barrier
    held: _HeldBarrier | None = None

    def gradient(self, actions: Array) -> NDArray[np.float64]:
        """The solve's objective, differentiated at ``actions`` and flattened as the rows are: at a
        step the descent holds on the confounding radius' kink, the least-norm element of its
        subdifferential (:func:`chc.support._penalised_gradient`)."""
        if self.support is None:
            gradient = control_gradient_adjoint(self.model, self.x0, actions, self.dt, self.cost)
        else:
            gradient = _penalised_gradient(
                self.model,
                self.x0,
                actions,
                self.dt,
                self.cost,
                self.support,
                self.lam_supp,
                self.uncertainty,
                self.lam_unc,
                any(row.matrix.shape[0] for row in self.constraints),
            )
        return np.asarray(gradient, dtype=np.float64).ravel()

    def value(self, actions: Array) -> Array:
        """The solve's objective at ``actions``, as the penalised descent reads it: the confounding
        radius' norm unsmoothed where it takes the norm by its proximal map."""
        if self.support is None:
            return total_cost(self.model, self.x0, actions, self.dt, self.cost)
        rows = any(row.matrix.shape[0] for row in self.constraints)
        norms = _radius_norms(self.uncertainty, self.lam_unc, actions, rows)
        value = _objective(
            self.model,
            self.x0,
            actions,
            self.dt,
            self.cost,
            self.support,
            self.lam_supp,
            *_smooth(self.uncertainty, self.lam_unc, norms),
        )
        if norms is None:
            return value
        return value + _step_norms(actions.astype(value.dtype), norms, jnp.ones((), value.dtype))


@dataclass(frozen=True)
class _Kkt:
    """What binds at a plan, and the multipliers that make it stationary, as prices and weights
    read them."""

    at_hi: NDArray[np.bool_]  # actions on their upper bound
    at_lo: NDArray[np.bool_]  # actions on their lower bound
    active: NDArray[np.intp]  # the binding rows' indices
    equality: NDArray[np.bool_]  # per binding row: at both of its bounds
    columns: NDArray[np.float64]  # (actions, binding rows): each row's push, signed by its side
    multipliers: NDArray[np.float64]  # the rows', by least squares on the free actions
    singular: NDArray[np.float64]  # the binding rows' singular values on the free actions
    residual: float  # stationarity miss on the free actions, over ``scale``
    scale: float  # max(1, |gradient on the free actions|)
    slack: float  # what counts as zero in a multiplier, a share of the larger of the gradient's
    # norm on the free actions and the guess's
    remainder: NDArray[np.float64]  # the gradient less the rows' push: the box's multipliers

    @property
    def free(self) -> NDArray[np.bool_]:
        return ~(self.at_hi | self.at_lo)


def _guess(lo: Array, hi: Array) -> Array:
    """The actions the descent starts from where it is given none: zeros, clipped to the box."""
    return jnp.clip(jnp.zeros_like(lo), lo, hi)


def _kkt(
    gradient: NDArray[np.float64],
    actions: NDArray[np.float64],
    matrix: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    lo: NDArray[np.float64],
    hi: NDArray[np.float64],
    tolerance: float,
    reference: NDArray[np.float64],
) -> _Kkt:
    """The binding bounds and rows at ``actions``, within ``tolerance * (1 + |bound|)``, and the
    rows' multipliers from the gradient on the actions left off the box.

    What counts as zero in a multiplier is a share of the larger of the gradient's norm on the
    free actions and ``reference``'s, the gradient at the guess the descent starts from,
    zeros clipped to the box. Both are in the cost's units per unit of an action, so a multiplier
    reads as zero or not the same in any units the actions share and in any units of the cost.
    The guess's gradient is the cost's slope before the plan moves, which is there to compare a
    row's push with when every row's is zero and the plan's own gradient is rounding."""
    with np.errstate(invalid="ignore"):  # an infinite bound is never binding
        at_hi = np.isfinite(hi) & (actions >= hi - tolerance * (1.0 + np.abs(hi)))
        at_lo = np.isfinite(lo) & (actions <= lo + tolerance * (1.0 + np.abs(lo))) & ~at_hi
        values = matrix @ actions
        up = np.isfinite(upper) & (values >= upper - tolerance * (1.0 + np.abs(upper)))
        down = np.isfinite(lower) & (values <= lower + tolerance * (1.0 + np.abs(lower)))
    free = ~(at_hi | at_lo)
    active = np.flatnonzero(up | down)
    # A row at its upper bound pushes back along +a, at its lower along -a; an equality row, either.
    columns = (matrix[active] * np.where(down[active] & ~up[active], -1.0, 1.0)[:, None]).T
    local, pull = columns[free], gradient[free]
    scale = max(1.0, float(np.linalg.norm(pull)))
    size = max(float(np.linalg.norm(pull)), float(np.linalg.norm(reference)))
    multipliers, singular = np.zeros(active.size), np.zeros(0)
    if active.size and local.shape[0]:
        multipliers = np.linalg.lstsq(local, -pull, rcond=None)[0]
        singular = np.linalg.svd(local, compute_uv=False)
    return _Kkt(
        at_hi=at_hi,
        at_lo=at_lo,
        active=active,
        equality=up[active] & down[active],
        columns=columns,
        multipliers=multipliers,
        singular=singular,
        residual=float(np.linalg.norm(pull + local @ multipliers)) / scale,
        scale=scale,
        slack=1e3 * tolerance * size,
        remainder=gradient + columns @ multipliers,
    )


def _diagnostic_status(value: float | None, tolerance: float | None) -> DiagnosticStatus:
    if value is None or tolerance is None:
        return "not_evaluated"
    if not math.isfinite(value):
        return "not_finite"
    return "within" if value <= tolerance else "beyond"


def _diagnostics(problem: _PlanProblem, actions: Array) -> tuple[float | None, float, float, float]:
    """``(stationarity, its tolerance, constraint violation, its tolerance)`` of ``actions`` on
    ``problem``, as :attr:`CausalPlan.stationarity_status` and :attr:`CausalPlan.violation_status`
    read them. Under a barrier, each step whose condition the rounds ended with a positive
    multiplier on binds, and pushes along the gradient of its shortfall."""
    shape, dtype = actions.shape, actions.dtype
    eps = float(jnp.finfo(dtype).eps)
    box_lo = broadcast_box(problem.u_lo, shape, "u_lo", dtype)
    box_hi = broadcast_box(problem.u_hi, shape, "u_hi", dtype)
    guess = _guess(box_lo, box_hi)
    value = problem.value(guess)
    sigma, scale, _ = _descent_units(
        problem.model, problem.x0, guess, problem.dt, problem.cost, box_lo, box_hi, value
    )
    factor = np.asarray(sigma, dtype=np.float64).ravel()
    size = factor.size
    present = [row for row in problem.constraints if row.matrix.shape[0]]
    matrix = np.vstack([row.matrix for row in present]) / factor if present else np.zeros((0, size))
    norms = np.linalg.norm(matrix, axis=1)
    lower = np.concatenate([row.lower for row in present]) / norms if present else np.zeros(0)
    upper = np.concatenate([row.upper for row in present]) / norms if present else np.zeros(0)
    matrix = matrix / norms[:, None]
    point = factor * np.asarray(actions, dtype=np.float64).ravel()
    lo = factor * np.asarray(box_lo, dtype=np.float64).ravel()
    hi = factor * np.asarray(box_hi, dtype=np.float64).ravel()
    level = matrix @ point
    with np.errstate(invalid="ignore"):  # an absent side is infinite, and nan actions break all
        excess = np.concatenate([lo - point, point - hi, lower - level, level - upper])
    # np.max keeps a nan, where max() drops it
    violation = float(np.max(np.maximum(excess, 0.0), initial=0.0))
    violation_tolerance = 1024 * eps * (1.0 + float(np.max(np.abs(point), initial=0.0)))
    stationarity_tolerance = _stationarity_tolerance(_DESCENT_TOL, dtype)
    share = factor * float(scale)
    gradient = problem.gradient(actions) / share
    kkt = _kkt(
        gradient,
        point,
        matrix,
        lower,
        upper,
        lo,
        hi,
        max(1e-6, 1e3 * eps),
        problem.gradient(guess) / share,
    )
    capped, floored = np.flatnonzero(kkt.at_hi), np.flatnonzero(kkt.at_lo)
    sides = np.zeros((size, capped.size + floored.size))
    sides[capped, np.arange(capped.size)] = 1.0
    sides[floored, capped.size + np.arange(floored.size)] = -1.0
    pushes = np.hstack([kkt.columns, sides, _condition_pushes(problem, actions, factor)])
    miss = gradient
    if pushes.shape[1]:
        # an equality row and a box of zero width, which fixes its action, push either way; an
        # inequality row and a side of a wider box only outward
        fixed = hi <= lo
        either = np.concatenate([kkt.equality, fixed[capped], fixed[floored]])
        either = np.concatenate([either, np.zeros(pushes.shape[1] - either.size, bool)])
        least = np.where(either, -np.inf, 0.0)
        fit = lsq_linear(pushes, -gradient, bounds=(least, np.inf), method="bvls")
        miss = gradient + pushes @ fit.x
    return (
        float(np.linalg.norm(miss)) / kkt.scale,
        stationarity_tolerance,
        violation,
        violation_tolerance,
    )


@eqx.filter_jit
def _shortfall_jacobian(
    held: _HeldBarrier, model: Dynamics, x0: Array, actions: Array, dt: float
) -> Array:
    """The held condition's shortfall at each step, differentiated in the actions."""

    def shortfall(us: Array) -> Array:
        return held.shortfall(rollout(model, x0, us, dt)[:-1], us)

    return jax.jacobian(shortfall)(actions)


def _round_metric(
    model: Dynamics,
    x0: Array,
    actions: Array,
    dt: float,
    cost: QuadraticCost,
    lo: Array,
    hi: Array,
    held: _HeldBarrier,
    excess: NDArray[np.float64],
) -> Array:
    """Per action, the square root of a barrier round's Gauss-Newton curvature along it at
    ``actions``: the cost's (:func:`chc.control._action_units`) plus the penalty's, ``rho`` times
    the squared slope in the action of each step's shortfall whose term is on there,
    ``lam + rho (c + backoff) > 0``, ``excess`` being ``c + backoff``. A lever in other units, or
    the cost, moves both alike, so the steps it scales read neither."""
    sigma, size, _ = _descent_units(model, x0, actions, dt, cost, lo, hi)
    slopes = np.asarray(_shortfall_jacobian(held, model, x0, actions, dt), dtype=np.float64)
    weight = float(held.weight)
    on = (np.asarray(held.multipliers, dtype=np.float64) + weight * excess).ravel() > 0.0
    penalty = weight * np.sum(slopes.reshape(-1, *actions.shape)[on] ** 2, axis=0)
    curvature = np.asarray(sigma, dtype=np.float64) ** 2 * float(size) + penalty
    return jnp.asarray(np.sqrt(curvature), actions.dtype)


def _condition_pushes(
    problem: _PlanProblem, actions: Array, factor: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``(actions, binding steps)``: the gradient in ``v`` of the shortfall of each step's barrier
    condition the rounds ended with a positive multiplier on, each column of norm 1; no columns
    without a barrier or with none binding."""
    held = problem.held
    if held is None:
        return np.zeros((factor.size, 0))
    binding = np.flatnonzero(np.asarray(held.multipliers, dtype=np.float64).ravel() > 0.0)
    if not binding.size:
        return np.zeros((factor.size, 0))

    jacobian = np.asarray(
        _shortfall_jacobian(held, problem.model, problem.x0, actions, problem.dt), dtype=np.float64
    )
    columns = jacobian.reshape(-1, factor.size)[binding].T / factor[:, None]
    norms = np.linalg.norm(columns, axis=0)
    return columns / np.where(norms > 0.0, norms, 1.0)


def _row_prices(
    gradient: NDArray[np.float64],
    actions: NDArray[np.float64],
    matrix: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    lo: NDArray[np.float64],
    hi: NDArray[np.float64],
    tolerance: float,
    reference: NDArray[np.float64],
) -> ShadowPrices:
    """The rows' multipliers at ``actions``, from the gradient on the actions left off the box;
    ``reference`` is :func:`_kkt`'s."""
    kkt = _kkt(gradient, actions, matrix, lower, upper, lo, hi, tolerance, reference)
    at_hi, at_lo, active, equality = kkt.at_hi, kkt.at_lo, kkt.active, kkt.equality
    columns, multipliers, singular = kkt.columns, kkt.multipliers, kkt.singular
    residual, scale, slack, remainder = kkt.residual, kkt.scale, kkt.slack, kkt.remainder
    local = columns[kkt.free]
    independent = active.size == 0 or (
        singular.size == active.size
        and singular[-1] > singular[0] * max(local.shape) * np.finfo(np.float64).eps
    )
    dual_feasible = bool(
        np.all(multipliers[~equality] >= -slack)
        and np.all(remainder[at_hi] <= slack)
        and np.all(remainder[at_lo] >= -slack)
    )
    rows: list[RowPrice] = []
    for index in range(matrix.shape[0]):
        found = np.flatnonzero(active == index)
        if not found.size:
            rows.append(RowPrice("inactive", 0.0, (0.0, 0.0)))
            continue
        column = int(found[0])
        if independent:
            price = float(multipliers[column])
            weak = abs(price) * float(np.linalg.norm(local[:, column])) <= slack
            rows.append(RowPrice("weakly_active" if weak else "exact", price, (price, price)))
            continue
        low, high, feasible = _multiplier_range(
            gradient,
            columns,
            equality,
            at_hi,
            at_lo,
            column,
            10.0 * max(residual, tolerance) * scale,
        )
        dual_feasible = dual_feasible and feasible
        rows.append(RowPrice("degenerate", None, (low, high)))
    return ShadowPrices(
        rows=tuple(rows),
        residual=residual,
        smallest_singular_value=(
            math.inf if not active.size else float(singular[-1]) if independent else 0.0
        ),
        dual_feasible=dual_feasible,
    )


def _multiplier_range(
    gradient: NDArray[np.float64],
    columns: NDArray[np.float64],
    equality: NDArray[np.bool_],
    at_hi: NDArray[np.bool_],
    at_lo: NDArray[np.bool_],
    column: int,
    slack: float,
) -> tuple[float, float, bool]:
    """The least and the greatest multiplier of one binding row over every set of multipliers that
    meets the stationarity conditions to within ``slack``, and whether any does."""
    size, active = columns.shape
    capped, floored = np.flatnonzero(at_hi), np.flatnonzero(at_lo)
    system = np.zeros((size, active + capped.size + floored.size))
    system[:, :active] = columns
    system[capped, active + np.arange(capped.size)] = 1.0
    system[floored, active + capped.size + np.arange(floored.size)] = -1.0
    bounds = [(None, None) if both else (0.0, None) for both in equality]
    bounds += [(0.0, None)] * (capped.size + floored.size)
    ends, feasible = [], True
    for sense in (1.0, -1.0):
        objective = np.zeros(system.shape[1])
        objective[column] = sense
        result = linprog(
            objective,
            A_ub=np.vstack([system, -system]),
            b_ub=np.concatenate([slack - gradient, slack + gradient]),
            bounds=bounds,
            method="highs",
        )
        feasible = feasible and result.status != 2
        ends.append(sense * result.fun if result.status == 0 else -sense * math.inf)
    return float(ends[0]), float(ends[1]), feasible


Barrier = Callable[[Array], Array] | tuple[Callable[[Array], Array], ...]
"""``h``, safe where ``h >= 0``; or a tuple of them, safe where each is, each its own condition."""

_GAMMA_LIMIT = (3.0 + math.sqrt(5.0)) / 2.0  # where (Gamma-1)^2/Gamma reaches 1


@dataclass(frozen=True, eq=False)
class ChannelSensitivity:
    """An absolute radius on a fitted channel's error, state by state, under a marginal sensitivity
    model of level ``Gamma`` on the levers given a hidden confounder.

    The model: given the state and the adjustment set, a hidden confounder ``Z`` moves the levers'
    density by a factor within ``[1/Gamma, Gamma]`` (the continuous marginal sensitivity model,
    Jesson et al. 2022), and nothing bounds how much of the rate's noise ``Z`` carries. Then the
    share of an action residual's variance ``Z`` explains is at most ``rho^2 = (Gamma-1)^2/Gamma``,
    and the least-squares channel on the action residuals is biased by at most

        ``||(B_hat(x) - B(x)) D||_2 <= kappa(Gamma) * confounding * nu(x)``,
        ``kappa(Gamma) = (Gamma-1) / sqrt(3 Gamma - Gamma^2 - 1)``,

    at every state ``x`` (``validation/channel_bias_bound.mac``), where ``confounding`` is
    ``sqrt(lambda_max(R'R))``, ``R`` the fit's moment residual, and
    ``nu(x)^2 = lambda_max(D Phi(x)' V^-1 Phi(x) D)``, ``V`` the channel's design on the action
    residuals ``gram``, ``Phi(x) a`` the design row of an action ``a`` at ``x`` and ``D`` each
    lever's largest action over the largest (:func:`certify_safety`'s ``u_max``). The bound grows
    without limit as ``Gamma`` reaches ``(3 + sqrt 5)/2``: from there the levers may be a function
    of the confounder, and no channel is identified. Beside the bias the channel scatters by
    ``sampling * nu(x)``, ``sampling`` the norm of :func:`chc.dynamics_id.channel_region`'s radii
    at ``level``. The radius at ``x`` is

        ``Delta(x, Gamma) = nu(x) * (sampling + kappa(Gamma) * confounding)``,

    in the channel's own units, an operator-norm ball around the estimate whose width reads the
    data: the scatter of the log's actions and of its noise, and how far ``x`` lies from the states
    the log moved the levers at. ``Gamma = 1`` is the sampling error alone. Read at each planned
    step's predicted state, it widens where the plan leaves the log behind.

    Premises: the fit's channel is least squares on the action residuals (an identified Euler fit
    by adjustment, as :func:`chc.dynamics_id.channel_region` takes); the confounder's push on the
    rate is independent of the rest of its noise. The bound holds for any distribution of the
    actions; the sampling part is the ellipsoid's, whose noise must be mean zero given the
    actions, and under a confounder is read at the least-squares limit, to first order. An
    ``rk4`` fit is not covered: its fixed point carries the drift's confounding, which no bound
    on the levers limits, into the channel.
    """

    gram: NDArray[np.float64]  # (q, q): V, the channel's design on the action residuals, D'D
    channel_degree: int
    sampling: float  # the ellipsoid's radii's norm at ``level``, in V's norm
    confounding: float  # sqrt(lambda_max(R'R)), R the moment residual, in the rate's units
    level: float

    def __post_init__(self) -> None:
        gram = np.asarray(self.gram, dtype=np.float64)
        if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not np.isfinite(gram).all():
            raise ValueError(f"gram must be a finite square matrix, got shape {gram.shape}")
        if not np.all(np.linalg.eigvalsh(0.5 * (gram + gram.T)) > 0.0):
            raise ValueError(
                "gram must be positive definite: some direction of the channel is unmoved"
            )
        if not (0.0 <= self.sampling < math.inf and 0.0 <= self.confounding < math.inf):
            raise ValueError(
                f"sampling={self.sampling} and confounding={self.confounding} must be finite and "
                "not negative"
            )
        if not 0.0 < self.level < 1.0:
            raise ValueError(f"level={self.level!r} must be in (0, 1)")
        object.__setattr__(self, "gram", gram)

    @classmethod
    def from_fit(cls, fit: CausalDynamicsFit, level: float = 0.95) -> ChannelSensitivity:
        """The fit's own: its channel's design, its moment residual and its region at ``level``.

        Raises:
            ValueError: where :func:`chc.dynamics_id.channel_region` refuses the fit.
        """
        from chc.dynamics_id import channel_region

        region = channel_region(fit, level)
        solved = fit._channel_gram
        if solved is None:  # channel_region refuses such a fit first
            raise ValueError("the fit keeps no least-squares Gram of its channel to read")
        gram = np.asarray(solved.matrix, dtype=np.float64)
        ridge = np.asarray(solved.ridge, dtype=np.float64)
        residual = np.asarray(fit.moment_residual, dtype=np.float64)
        return cls(
            gram=gram - np.diag(ridge),  # D'D: the ridge's share would shrink the width
            channel_degree=fit.residual.channel_degree,
            sampling=float(np.linalg.norm(region.radius)),
            confounding=math.sqrt(
                max(float(np.max(np.linalg.eigvalsh(residual.T @ residual))), 0.0)
            ),
            level=float(level),
        )

    @staticmethod
    def factor(gamma: float) -> float:
        """``kappa(Gamma)``: the bias per unit of ``confounding * nu(x)``; ``inf`` from
        ``Gamma = (3 + sqrt 5)/2`` on.

        Raises:
            ValueError: on a ``gamma`` below 1 or not a number, which is no sensitivity level.
        """
        if not gamma >= 1.0:
            raise ValueError(f"gamma is a sensitivity level and must be >= 1, got {gamma}")
        spread = 3.0 * gamma - gamma * gamma - 1.0
        return (gamma - 1.0) / math.sqrt(spread) if spread > 0.0 else math.inf

    @staticmethod
    def level_at(factor: float) -> float:
        """The ``Gamma`` whose :meth:`factor` is ``factor``: below ``(3 + sqrt 5)/2``."""
        if factor == math.inf:
            return _GAMMA_LIMIT
        square = factor * factor
        return (2.0 + 3.0 * square + factor * math.sqrt(4.0 + 5.0 * square)) / (
            2.0 * (1.0 + square)
        )

    def radius(self, state: ArrayLike, gamma: float, u_max: ArrayLike | None = None) -> float:
        """``Delta(x, Gamma)`` at ``state``, the levers read in units of their own largest action
        where ``u_max`` gives one per lever, as :func:`certify_safety` reads them."""
        point = jnp.asarray(state, dtype=jnp.result_type(float))
        levers = self.gram.shape[0] // int(control_affine_features(point, self.channel_degree).size)
        shape = None if u_max is None else _action_shape(_reach(u_max, levers))[0]
        width = _ChannelWidth.build(self, levers, shape, gamma, point.dtype)
        return float(width(point[None, :])[0])


class _ChannelWidth(eqx.Module):
    """``Delta(x, Gamma)`` of a :class:`ChannelSensitivity` at each of a batch of states,
    JAX-traceable, so the held solve and the audit read one radius."""

    inverse: Array  # V^-1
    shape: Array | None  # D, each lever's largest action over the largest; None, the identity
    scale: Array  # sampling + kappa(Gamma) * confounding
    degree: int = eqx.field(static=True)
    levers: int = eqx.field(static=True)

    @classmethod
    def build(
        cls,
        sensitivity: ChannelSensitivity,
        levers: int,
        shape: NDArray[np.float64] | None,
        gamma: float,
        dtype: jnp.dtype,
    ) -> _ChannelWidth:
        """Raises:
        ValueError: on a ``gamma`` at or past ``(3 + sqrt 5)/2``, whose bias is unbounded, or a
            ``gram`` whose size is not the levers' times the channel's features."""
        kappa = ChannelSensitivity.factor(gamma)
        if not math.isfinite(kappa):
            raise ValueError(
                f"gamma={gamma} is at or past (3 + sqrt 5)/2 = {_GAMMA_LIMIT:.6f}: under the "
                "channel's sensitivity model the levers may then be a function of the confounder, "
                "and no radius bounds the channel"
            )
        if sensitivity.gram.shape[0] % levers:
            raise ValueError(
                f"the sensitivity's gram has size {sensitivity.gram.shape[0]}, which is not "
                f"{levers} levers times the channel's features"
            )
        return cls(
            inverse=jnp.asarray(np.linalg.inv(sensitivity.gram), dtype),
            shape=None if shape is None else jnp.asarray(shape, dtype),
            scale=jnp.asarray(sensitivity.sampling + kappa * sensitivity.confounding, dtype),
            degree=sensitivity.channel_degree,
            levers=levers,
        )

    def __call__(self, states: Array) -> Array:
        return self.scale * self.widths(states)

    def widths(self, states: Array) -> Array:
        """``nu(x)`` at each state: the radius per unit of ``sampling + kappa * confounding``."""

        def one(x: Array) -> Array:
            features = control_affine_features(x, self.degree)
            if features.size * self.levers != self.inverse.shape[0]:
                raise ValueError(
                    f"the sensitivity's gram has size {self.inverse.shape[0]}; {self.levers} "
                    f"levers at {features.size} features each need {features.size * self.levers}"
                )
            lifted = jnp.kron(jnp.eye(self.levers, dtype=x.dtype), features[:, None])
            metric = lifted.T @ self.inverse @ lifted
            if self.shape is not None:
                metric = metric * self.shape[:, None] * self.shape[None, :]
            return jnp.sqrt(jnp.maximum(jnp.linalg.eigvalsh(metric)[-1], 0.0))

        return jax.vmap(one)(states)


@dataclass(frozen=True)
class BarrierConstraint:
    """``grad h . xdot >= -alpha * h`` at every planned step, against every effect the data allow.

    The condition :func:`certify_safety` audits, handed to :func:`causal_plan` to hold in the solve
    rather than to price after it. ``barrier`` is ``h`` -- safe where ``h >= 0``, a JAX-traceable
    scalar function; ``alpha`` is the class-K gain; ``gamma`` and ``cvar_gap`` set the
    identification radius ``(gamma-1)/(gamma+1) * cvar_gap * ||B_hat(x0)||_2`` on the effect,
    ``cvar_gap`` a share of the channel's operator norm at the plan's start, so the condition is
    the worst case over the identified set and ``gamma = 1`` is the ordinary CBF condition. The
    defaults and the calibration burden on ``cvar_gap`` are :func:`certify_safety`'s.

    Several bounds are a tuple of barriers, safe where every one is ``>= 0``, and each holds its own
    condition at every step. Their least is no substitute: its condition jumps where two tie,
    reading on each side that side's gradient, so a solve that ends on the tie cannot step off it,
    and the condition it holds asks nothing of a bound until that bound is the nearest.

    ``u_max``, one number per lever, each lever's largest action in its own units, reads the
    condition in units of each lever's own largest action, as :func:`certify_safety` does with the
    same numbers, and the audit after the solve reads them; ``None`` reads it in the levers' own
    units, one ball for all of them, as before. :func:`chc.prescribe` passes its box's. A lever
    whose number is 0 must have a box that holds it at 0.

    ``sensitivity`` (:class:`ChannelSensitivity`) replaces the radius's share of the channel by an
    absolute radius at each step's predicted state, as :func:`certify_safety` reads it, held in the
    solve at the same steps; ``gamma`` must then be below ``(3 + sqrt 5)/2``.

    ``u_box``, ``(u_lo, u_hi)`` with one number per lever, is the box the audit after the solve
    reads ``gamma_star`` on; ``None`` reads it on the box the plan is solved in. A solve in a
    narrower box than the levers have -- :func:`chc.prescribe`'s lever selection pins the levers it
    leaves out at zero -- is then audited as if planned on all of them, with the same box as
    ``u_max``.

    Raises:
        ValueError: on a ``gamma`` below 1 or not finite, which is not a sensitivity level, a
            ``cvar_gap`` not positive or not finite, which scales no radius, a ``u_max`` with an
            entry negative or nan, or none positive, which is no action's size, or an empty
            tuple of barriers, which holds nothing -- here, rather than after the solve that would
            have used them; or, with a ``sensitivity``, a ``gamma`` at or past ``(3 + sqrt 5)/2``.
    """

    barrier: Barrier
    alpha: float = 1.0
    gamma: float = 1.0
    cvar_gap: float = 1.0
    u_max: tuple[float, ...] | None = None
    u_box: tuple[tuple[float, ...], tuple[float, ...]] | None = None
    sensitivity: ChannelSensitivity | None = None

    def __post_init__(self) -> None:
        if isinstance(self.barrier, tuple) and not self.barrier:
            raise ValueError("an empty tuple of barriers holds no condition")
        if self.u_max is not None:
            object.__setattr__(self, "u_max", tuple(float(v) for v in _reach(self.u_max)))
        if self.u_box is not None:
            lo, hi = (tuple(float(v) for v in np.ravel(end)) for end in self.u_box)
            object.__setattr__(self, "u_box", (lo, hi))
        if not 1.0 <= self.gamma < math.inf:
            raise ValueError(
                "gamma is a marginal sensitivity model level and must be >= 1 and finite, got "
                f"{self.gamma}"
            )
        if not 0.0 < self.cvar_gap < math.inf:
            raise ValueError(
                "cvar_gap must be positive and finite to scale a sensitivity radius, got "
                f"{self.cvar_gap}"
            )
        if self.sensitivity is not None and not self.gamma < _GAMMA_LIMIT:
            raise ValueError(
                f"gamma={self.gamma} is at or past (3 + sqrt 5)/2 = {_GAMMA_LIMIT:.6f}, where the "
                "channel's sensitivity model bounds no bias, and no condition can be held"
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
    lipschitz: ArrayLike = 0.0,
    model_error: float = 0.0,
    noise: ArrayLike = 0.0,
    noise_persistence: float | ArrayLike = 0.0,
    tolerance: float | None = None,
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
        lipschitz, model_error: feed the error tube of the plan's RK4 rollout, the field off by at
            most ``model_error`` wherever a step reads it. ``lipschitz`` is either a norm-Lipschitz
            bound ``L >= 0`` on the field in the state, for the recursion
            ``e_{k+1} = e_k + dt (L e_k + eps) phi(L dt)``, ``phi(z) = 1 + z/2 + z^2/6 + z^3/24``
            (:func:`chc.uncertainty.time_varying_rollout_bound`), or, for a field affine in the
            state, its ``(n, n)`` state matrix ``A``, whose RK4
            propagators carry the tube (:func:`chc.uncertainty.linear_rollout_bound`) and shrink it
            where ``A`` contracts. ``model_error`` or ``noise`` is what switches certification on:
            with neither the tube would be identically zero, so the plan reports
            ``certificate_status == "not_evaluated"`` and both certificate fields come back ``None``
            rather than a vacuous full-horizon pass.
        noise, noise_persistence: the plant's own error at each step, which no model error
            covers: ``w_k = x_{k+1} - step(x_k, u_k)``, the plant's next state less the plan's RK4
            step of the model from the plant's state, Gaussian, AR(1) from one step to the next,
            ``Cov(w_i, w_j) = noise_persistence^|i - j| noise``. ``noise`` is its covariance in the
            state's units, a number for each state's variance with none shared, or an ``(n, n)``
            matrix; ``noise_persistence`` its lag-1 correlation, in ``[-1, 1]``, or the ``(n, n)``
            coefficient ``Phi`` of a VAR(1), ``w_{k+1} = Phi w_k + eta``, where the states' noises
            persist at rates of their own, ``Cov(w_{k+1}, w_k) = Phi noise``, a stationary noise's,
            ``noise - Phi noise Phi'`` with no negative eigenvalue. The tube adds to
            the model error's radius the one within which the noise's part of the gap stays at each
            step with probability 0.95, carried through the same ``lipschitz``: through ``R(dt A)``
            on a state matrix, exactly to first order, and on a norm bound ``L`` through
            ``R(L dt)``, which bounds it (``validation/predictive_tube.mac``). On a state matrix
            the gap's covariance is exact, and the radius is the exact 0.95 quantile of the gap's
            distance from the plan, a sum of the covariance's eigenvalues times independent
            ``chi2_1`` (Imhof 1961). On a norm bound it is that quantile at the first step, where
            the covariance is ``noise`` itself, and past it ``sqrt(chi2_n(0.95))`` times the root
            of the bound on the largest variance along any direction, which holds with
            probability 0.95 on one state and more on several. Up to 0.17.0 that was the radius
            on a state matrix too, up to ``sqrt(chi2_n(0.95) / chi2_1(0.95))`` too wide where one
            direction's variance dominates, 1.249 times on two states. Zero by default, which
            leaves the tube the model error's alone, as before.
        tolerance: tube radius above which the plan stops being certified, in the state's units.
            A step whose predicted state is not a finite number, an overflow's or a nan's, is
            certified at no tolerance. Not given, it is infinite, which holds every finite radius:
            the plan's :attr:`~CausalPlan.certificate_status` then reads ``no_tolerance`` rather
            than ``certified``, and where an error model gives the plan a tube a ``FutureWarning``
            says that from 0.20.0 the default is 1.0, one unit of the state, the radius ADR 0091
            asks of a caller who logs each state in units of the error accepted on it. Pass it to
            choose the radius; ``inf`` keeps today's reading without the warning.
        constraints: linear rows over the whole action sequence
            (:class:`~chc.control.LinearConstraint`), held by every iterate of either solver, not
            only by the answer.
        barrier: a condition to hold at every planned step (:class:`BarrierConstraint`), or one
            per barrier of a tuple, by augmented-Lagrangian rounds around the same descent, with
            ``h >= 0`` at each step's next state where the path would leave the set, as the audit
            certifies a step by. Unlike
            ``constraints`` it binds the answer, not every iterate, and the plan comes back with
            :attr:`CausalPlan.safety` -- :func:`certify_safety` run on the finished plan, the
            verdict to read. Its ``gamma_star`` is priced at the largest magnitude any lever may
            take, the budget :func:`chc.prescribe` uses, drawn back to the box where the best
            action of that size leaves it. A barrier the unconstrained plan already
            clears changes no action, even when that solve stopped on its budget. ``steps`` then
            caps the descent before the rounds, and the rounds' descents on each set of next
            states they hold together, so a held plan takes at most twice ``steps``, and
            ``steps`` more for each time a next state joins the held ones, besides the plan on the
            concave envelope it may start from; it used to cap each of up to 31 descents. The
            solver status reads ``converged`` only where the rounds hold the condition and the
            plan's :attr:`~CausalPlan.stationarity_status` reads ``within``; ``infeasible`` where
            the rounds' penalty reached its ceiling with some step still short of the condition.
        warm_start: the ``(horizon, m)`` actions the descent starts from, zeros when omitted --
            typically the last plan shifted one step, which :class:`chc.mpc.RecedingHorizon` passes
            along with the barrier's multipliers. It is projected onto the box and the rows before
            the first step, so it need not be admissible. It moves where the solve starts, and so
            what a solve stopped by ``steps`` returns; a solve that converges on a problem with one
            minimiser lands on it from any start. Omitted on a model holding a
            :class:`chc.response.Saturation` that starts convex, the start is instead the plan of
            the same problem on each such curve's concave envelope (:func:`chc.response.relax`),
            and the plan reports that problem's cost as :attr:`CausalPlan.relaxed_cost`: such a
            curve has no slope at zero spend, so zeros are a stationary point the descent would stop
            at, reporting convergence.

    Raises:
        ValueError: if an uncertainty penalty is given without a support model, which would
            silently drop it -- the pessimistic solver is the only consumer of that argument; if
            ``model_error`` is negative, infinite or nan, which is not an error budget; if
            ``lipschitz`` is a negative, infinite or nan number, which no norm of a field's slope
            is, or neither a number nor a square matrix of the state's size; if ``noise`` is not
            finite, not a number or a matrix of the state's size, or not a covariance, which is
            symmetric with no negative eigenvalue, or ``noise_persistence`` is neither a
            correlation in ``[-1, 1]`` nor such a VAR(1) coefficient;
            if ``tolerance`` is
            negative or nan, or ``dt`` is not a finite positive step; if ``x0`` has an entry that
            is not a finite number, from which no step is predicted; if a barrier is given with
            a box that admits no action but zero, which leaves nothing to price it with; if a
            barrier is given and the path from the descent's start or from the rounds' start is
            not a finite number at some state, or the model is not affine in the action there or
            at a finite side of the box, which :func:`certify_safety` refuses of the finished plan
            and is now refused before the solve; or if ``warm_start`` is not a finite
            ``(horizon, m)`` array.
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
        noise=noise,
        noise_persistence=noise_persistence,
        tolerance=tolerance,
        steps=steps,
        constraints=constraints,
        barrier=barrier,
        warm_start=warm_start,
        multipliers=None,
    )
    return plan


def _error_tube(
    carried: NDArray[np.float64],
    model_error: float | NDArray[np.float64],
    covariance: NDArray[np.float64],
    persistence: float | NDArray[np.float64],
    dt: float,
    horizon: int,
    field_scale: float | None = None,
    read: _NoiseRead | None = None,
    *,
    integrator: Integrator = "rk4",
) -> NDArray[np.float64] | None:
    """The plan's error tube carried by ``carried`` -- a norm bound, one state matrix, or one a step
    -- or None where neither the model error nor the noise gives it a radius, of rollouts by
    ``integrator``, the step the plan's model is stepped by (:func:`_integrator`).

    ``model_error`` is a bound on the field's error at each step, one number for every step or one
    a step, and the tube adds what it carries to the noise's radius, unless ``field_scale`` is
    given: then ``model_error`` is a standard error, and what it carries, ``f_k``, the standard
    deviation of the norm of an error independent of the noise, read at ``field_scale`` times
    itself where its spread is estimated (:func:`chc.decision._field_scale`). Its direction is not
    known, and it is placed along the noise's widest, where it widens the radius most: the radius
    is that of the noise's covariance with ``(field_scale f_k)^2`` added to its largest
    eigenvalue, on one state ``sqrt(r_k^2 + (field_scale z f_k)^2)``, ``z`` the normal's two-sided
    quantile. ``read`` carries the noise's own error where it was read off a log
    (:func:`chc.uncertainty._read_covariances`)."""
    reads = np.asarray(model_error, dtype=np.float64)
    errors = [float(reads)] * horizon if reads.ndim == 0 else [float(error) for error in reads]
    if not (max(errors, default=0.0) > 0.0 or covariance.any()):
        return None
    tube = (
        _tube([float(carried)] * horizon, errors, dt, integrator)
        if carried.ndim == 0
        else _linear_tube(carried, errors, dt, integrator=integrator)
    )
    if field_scale is not None:
        return _noise_tube(
            carried,
            covariance,
            persistence,
            dt,
            horizon,
            field_scale * tube,
            read,
            integrator=integrator,
        )
    if covariance.any():
        tube = tube + _noise_tube(
            carried, covariance, persistence, dt, horizon, read=read, integrator=integrator
        )
    return tube


def _carried_along(
    plan: CausalPlan,
    state_matrices: NDArray[np.float64],
    model_error: float | NDArray[np.float64],
    noise: ArrayLike,
    noise_persistence: float | NDArray[np.float64],
    dt: float,
    tolerance: float,
    field_scale: float | None = None,
    read: _NoiseRead | None = None,
) -> CausalPlan:
    """``plan`` with its tube carried by the field's ``(horizon, n, n)`` state matrix at each of its
    steps rather than by the one ``lipschitz`` it was solved with: a field affine in the state whose
    slope moves with the action, as :func:`chc.prescribe`'s fitted one does; or, given one number,
    by that norm bound as the plan was. ``model_error`` is one number for every step or one a step,
    and ``field_scale`` reads it as a standard error, and ``read`` carries the noise's own error
    (:func:`_error_tube`). The tube is read off the finished actions and moves no action, and is
    carried by the step the plan's model is stepped by."""
    horizon = len(plan.actions)
    covariance = _noise_covariance(noise, plan.trajectory.shape[-1])
    tube = _error_tube(
        state_matrices,
        model_error,
        covariance,
        noise_persistence,
        dt,
        horizon,
        field_scale,
        read,
        integrator="rk4" if plan._problem is None else _integrator(plan._problem.model),
    )
    if tube is None:
        return plan
    return replace(
        plan,
        uncertainty_tube=jnp.asarray(tube),
        certified_horizon=_certified(tube, tolerance, plan.trajectory),
    )


def _certified(tube: NDArray[np.float64], tolerance: float, trajectory: Array) -> int:
    """The steps ``tube`` keeps within ``tolerance`` and whose predicted state is a number: a tube
    around a nan or an overflowed state bounds its distance to nothing."""
    finite = np.all(np.isfinite(np.asarray(trajectory[1:])), axis=-1)
    reached = int(np.argmin(finite)) if not finite.all() else finite.size
    return min(_within(tube, tolerance), reached)


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
    lipschitz: ArrayLike,
    model_error: float,
    tolerance: float | None,
    steps: int,
    constraints: Sequence[LinearConstraint],
    barrier: BarrierConstraint | None,
    warm_start: Array | NDArray[np.float64] | None,
    multipliers: NDArray[np.float64] | None,
    noise: ArrayLike = 0.0,
    noise_persistence: float | ArrayLike = 0.0,
) -> tuple[CausalPlan, NDArray[np.float64] | None]:
    """:func:`causal_plan`, with the barrier's multipliers passed in and handed back.

    What a receding horizon carries from one plan to the next besides the actions: ``multipliers``
    seed the barrier rounds in place of zeros, and the rounds' last come back -- zeros where the
    barrier was slack, ``None`` with no barrier. One row per step, two columns per barrier: its
    condition's and its next state's. A ``tolerance`` of None is infinite, and warns where the
    plan has a tube, at the caller of the caller's frame.
    """
    if uncertainty is not None and support is None:
        raise ValueError("uncertainty penalty requires a support model; it is unused without one")
    # A nan budget passed `model_error < 0` and then read as no budget, `model_error > 0` being
    # false too: the plan came back unevaluated with no word of why.
    if not 0.0 <= model_error < math.inf:
        raise ValueError(
            "model_error is a per-step error budget and cannot be negative, infinite or nan: "
            f"{model_error}"
        )
    if not 0.0 < dt < math.inf:
        raise ValueError(f"dt={dt} is not a step, which is finite and positive")
    if not np.all(np.isfinite(np.asarray(x0))):
        raise ValueError(
            f"x0={np.asarray(x0).tolist()} is not a state: every entry must be a number"
        )
    # In JAX's precision: a NumPy float64 start carried as it is made the float32 solve ask for
    # float64 arrays of x0's dtype, which JAX truncates with a warning.
    x0 = jnp.asarray(x0)
    unset = tolerance is None
    tolerance = math.inf if tolerance is None else tolerance
    _check_tolerance(tolerance)
    rate = np.asarray(lipschitz, dtype=float)
    if rate.ndim == 0 and not rate >= 0.0:
        raise ValueError(
            f"lipschitz={float(rate)} bounds the norm of the field's slope, which is never "
            "negative; a contracting field affine in the state passes its state matrix instead"
        )
    if rate.ndim == 0 and rate == math.inf:
        raise ValueError(
            "lipschitz=inf bounds nothing: a norm of the field's slope is finite, and the tube's "
            "first step would read inf * 0"
        )
    square = (np.shape(x0)[-1],) * 2
    if rate.ndim != 0 and rate.shape != square:
        raise ValueError(
            f"lipschitz is a number or the field's state matrix of shape {square}, not an array of "
            f"shape {rate.shape}"
        )
    covariance = _noise_covariance(noise, square[0])
    persistence = _noise_persistence(noise_persistence, covariance)
    authority = float(np.max(np.maximum(np.abs(np.asarray(u_lo)), np.abs(np.asarray(u_hi)))))
    if barrier is not None and not authority > 0.0:
        raise ValueError(
            f"a barrier needs a box that admits a nonzero action; the largest is {authority}"
        )

    first = None
    if warm_start is None and (relaxed := relax(model)) is not model:
        first, _ = _plan(
            relaxed,
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
            warm_start=None,
            multipliers=multipliers,
            noise=noise,
            noise_persistence=noise_persistence,
        )
        warm_start = first.actions
        _log.info(
            "planned on the concave envelope first",
            extra={
                "chc_event": "plan_relaxed",
                "relaxed_cost": first.task_cost,
                "status": first.solver_status,
                "descent_steps": first.solver_iterations,
            },
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
    if barrier is not None:  # before the descent, as well as before the rounds
        box = (
            broadcast_box(u_lo, guess.shape, "u_lo", guess.dtype),
            broadcast_box(u_hi, guess.shape, "u_hi", guess.dtype),
        )
        _refuse_unheldable(model, x0, dt, barrier.barrier, guess, box)
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
    held = None
    if barrier is not None:
        actions, status, iterations, multipliers, held = _hold_barrier(
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

    if first is not None:
        iterations += first.solver_iterations
        if status == "no_progress":
            # the answer is where the relaxed descent stopped, and it stopped for its own reason
            status = first.solver_status

    tube = _error_tube(
        rate, model_error, covariance, persistence, dt, horizon, integrator=_integrator(model)
    )
    problem = _PlanProblem(
        model=model,
        x0=x0,
        cost=cost,
        dt=dt,
        u_lo=u_lo,
        u_hi=u_hi,
        constraints=tuple(constraints),
        support=support,
        lam_supp=lam_supp,
        uncertainty=uncertainty,
        lam_unc=lam_unc,
        barrier=barrier is not None,
        held=held,
    )
    stationarity, stationarity_tolerance, violation, violation_tolerance = _diagnostics(
        problem, actions
    )
    trajectory = _trajectory(model, x0, actions, dt)
    plan = CausalPlan(
        actions=actions,
        trajectory=trajectory,
        task_cost=float(total_cost(model, x0, actions, dt, cost)),
        uncertainty_tube=None if tube is None else jnp.asarray(tube),
        certified_horizon=None if tube is None else _certified(tube, tolerance, trajectory),
        solver_status=status,
        solver_iterations=iterations,
        relaxed_cost=None if first is None else first.task_cost,
        stationarity=stationarity,
        stationarity_tolerance=stationarity_tolerance,
        constraint_violation=violation,
        violation_tolerance=violation_tolerance,
        tolerance=tolerance,
        _problem=problem,
    )
    if unset and tube is not None:
        warnings.warn(
            "the plan's error tube was read against no tolerance, which holds every finite radius, "
            "so its certificate_status reads 'no_tolerance'; from 0.20.0 the default tolerance is "
            f"{_TOLERANCE_DEFAULT:g}, one unit of the state (ADR 0091). Pass tolerance to choose "
            "the radius, or inf to keep this reading.",
            FutureWarning,
            stacklevel=3,
        )
    if barrier is None:
        return plan, None
    audit = _audit(
        plan,
        model,
        barrier.barrier,
        dt,
        barrier.alpha,
        barrier.gamma,
        barrier.cvar_gap,
        authority if barrier.u_max is None else barrier.u_max,
        (u_lo, u_hi)
        if barrier.u_box is None
        else (jnp.asarray(barrier.u_box[0]), jnp.asarray(barrier.u_box[1])),
        barrier.sensitivity,
    )
    return replace(plan, safety=audit), multipliers


class _HeldBarrier(eqx.Module):
    """The barrier condition as a Powell-Hestenes-Rockafellar term: one multiplier per step.

    It is a :class:`~chc.support.PenaltyModel`, so the penalised descent of :mod:`chc.support`
    minimises it unchanged. That descent has one penalty slot besides the support model's, so
    ``rest`` carries the uncertainty penalty the plan may already have.

    Each barrier holds two conditions at every step, the two the audit certifies a step by beyond
    ``h >= 0`` where the plan starts: the barrier condition, and ``h - ||grad h|| e >= 0`` at the
    step's next state, ``e`` the deviation tube a plant whose channel is in the set can reach
    (:func:`_deviation_tube`), where its path would otherwise leave the set: between the points the
    condition is read at, or away from the model's states, which it is read at. The shortfall, and
    the multipliers with it, have those two columns per barrier, in turn,
    ``(steps, 2 * barriers)``. Each column is multiplied by its ``factors`` entry: a condition's,
    the largest condition's scale over its own; a next state's, the same scale over the largest
    ``|h|`` along the path; so one back-off and one penalty read every column alike, whatever the
    units of the state it bounds.
    """

    model: Dynamics
    # not static: a barrier that is a module keeps its arrays traced, so a replanning loop whose
    # bounds move compiles once; filter_jit keys on a plain function either way
    barrier: Barrier
    dt: float = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    delta: Array  # a value, not a program: it moves with the start's channel in a replanning loop
    multipliers: Array
    weight: Array
    backoff: Array
    factors: Array | None = None  # one per column of the shortfall; None reads them as ones
    # per lever, what an action is multiplied by before its norm: see _action_shape; None, ones
    inverse: Array | None = None
    # (steps, barriers): 1 where a next state's column is held, 0 where it reads -inf and holds
    # nothing; None holds every one
    paths: Array | None = None
    # the radius at each step's state where a ChannelSensitivity sets it; None, ``delta`` at each
    width: _ChannelWidth | None = None
    rest: PenaltyModel | None = None
    rest_weight: float = eqx.field(static=True, default=0.0)

    def shortfall(self, states: Array, actions: Array) -> Array:
        """Per step and barrier, ``-alpha h - (a + <w, u> - d ||u||)`` and ``-(h - ||grad h|| e)``
        at the step's next state, ``e`` the deviation tube there (:func:`_deviation_tube`), each
        times its factor: positive where the audit fails."""
        reached = self._next_states(states, actions)
        deviation = _deviation_tube(
            self.model, states, actions, self.dt, self._delta(states), self.inverse
        )[1:]
        columns = []
        for index, barrier in enumerate(self._barriers()):
            left = -_barrier_room(barrier, reached, deviation)
            if self.paths is not None:
                left = jnp.where(self.paths[:, index] > 0.0, left, -jnp.inf)
            columns += [self._shortfall(barrier, states, actions), left]
        stacked = jnp.stack(columns, axis=1)
        return stacked if self.factors is None else stacked * self.factors

    def _barriers(self) -> tuple[Callable[[Array], Array], ...]:
        return self.barrier if isinstance(self.barrier, tuple) else (self.barrier,)

    def _next_states(self, states: Array, actions: Array) -> Array:
        """``x_1 .. x_H``: the states after ``x_0 .. x_{H-1}``, the last one's by its own step."""
        last = rk4_step(
            self.model, self.dt * (states.shape[0] - 1), states[-1], actions[-1], self.dt
        )
        return jnp.concatenate([states[1:], last[None, :]])

    def _shortfall(self, barrier: Callable[[Array], Array], states: Array, actions: Array) -> Array:
        h_values, grad_norm, drift, channel = _barrier_terms(
            self.model, barrier, states, actions, self.dt
        )
        return -self.alpha * h_values - _guaranteed(
            drift, channel, grad_norm, actions, self._delta(states), self.inverse
        )

    def _delta(self, states: Array) -> Array:
        return self.delta if self.width is None else self.width(states)

    def scale(self, states: Array, actions: Array) -> Array:
        """Per column of :meth:`shortfall`, before ``factors``, what rounding in it is relative to:
        a condition's largest term over the plan, and a next state's largest ``|h|`` along the
        path, the start's included."""
        path = jnp.concatenate([states, self._next_states(states, actions)[-1:]])
        return jnp.stack(
            [
                size
                for barrier in self._barriers()
                for size in (
                    self._scale(barrier, states, actions),
                    jnp.max(jnp.abs(jax.vmap(lambda x, h=barrier: jnp.squeeze(h(x)))(path))),
                )
            ]
        )

    def _scale(self, barrier: Callable[[Array], Array], states: Array, actions: Array) -> Array:
        h_values, grad_norm, drift, channel = _barrier_terms(
            self.model, barrier, states, actions, self.dt
        )
        push = jnp.einsum("km,km->k", channel, actions)
        spread = self._delta(states) * grad_norm * _norm(_shaped(actions, self.inverse))
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
) -> tuple[Array, SolverStatus, int, NDArray[np.float64], _HeldBarrier]:
    """Hold ``barrier`` by augmented-Lagrangian rounds started from the unconstrained ``start``.

    Each round minimises the plan's own objective plus ``sum(max(0, lam + rho c)^2 - lam^2)/2 rho``
    over the same box and rows by the same descent, ``c`` being the per-step shortfall of the
    condition :func:`certify_safety` audits, and ``-h`` at the next states held (below); then
    ``lam <- max(0, lam + rho c)``, and ``rho`` grows tenfold whenever a round fails to halve
    ``max |max(c, -lam/rho)|``, the measure of infeasibility and complementarity together. The
    scheme is Birgin & Martinez's (2014), and so is the starting penalty, ten times the cost at
    ``start`` over half the squared shortfall, less their floors of 1, which read the units of the
    cost and of the condition. Rounds stop once the measure is within half the back-off and the
    plan's stationarity, read with the multipliers the round ends with, is within its tolerance,
    ``converged``, whether or not the budget cut the round's descent: the descent stopping short
    of a stationary point read ``converged`` with a residual 28 times its tolerance on a SIR plant,
    and under the deviation tube a round the budget cut at a stationary point left a plan under a
    ``gamma = 2`` floor ``max_iterations``. They stop as well where ``rho`` would pass
    ``_PENALTY_RANGE`` times its start, ``infeasible`` if some step still falls short of the
    condition itself, the back-off aside, and ``max_iterations`` otherwise, once
    ``_BARRIER_ROUNDS`` are spent, or once their descents on one set of held next states have
    taken ``steps`` between them. ``rho`` grows only on a round whose descent stopped on its own
    rule and whose measure is beyond half the back-off: grown on rounds the budget cut, it reached
    3e6 on a feasible oscillator, where a round's descent stepped to a plan 134% above the
    optimum.

    ``c`` carries a back-off of ``_BARRIER_BACKOFF`` times the condition's largest term at
    ``start``, so the rounds settle inside the condition: an active step solved exactly sits on it,
    and rounding alone would then fail the audit about half the time. A ``start`` that clears the
    shifted condition is returned untouched, which is what makes a slack barrier free.

    Each round is the penalised descent of :mod:`chc.support` in variables scaled action by action
    by the round's own Gauss-Newton curvature (:func:`_round_metric`): the cost's, as the planner
    reads it, plus the penalty's on the steps whose term is on, so a unit step is the Newton step
    along each action alone however stiff the penalty has grown. Its stopping rule is the
    planner's, a fall relative to the round's objective with the first-order check
    (:func:`chc.control._descend`); a round that starts while the last one ended outside the
    back-off stops at a relative fall of ``_LOOSE_TOL`` instead, the inexact early solves of
    augmented-Lagrangian methods, and cannot end the rounds. The rounds start from ``start``
    itself. So no number of theirs carries a unit, and a plan held in other units of a lever or of
    the cost is the same plan to the rounds' tolerance. They used to run that descent in the
    caller's units, from a descent in the caller's units, with a step of 0.2 and a fall of 1e-9
    that read both: on the tests' oscillator under a velocity floor, the force in units 1e3 times
    its own ran out of its 310 000 steps 14% above the optimum, and the cost in units 1e-3 times
    its own stopped as converged 2.3% above it.

    The audit certifies a step only where ``h >= 0`` at its next state as well, and at a coarse step
    the condition can hold at every sample while the path leaves the set between them. The next
    states join as an active set: the rounds hold the conditions alone, and where the path then
    falls short at a next state, by the back-off, that state joins and the rounds go on, until none
    not yet held falls short. Held from the first round, they stiffened the rounds where none binds:
    at ``dt = 1`` and ``alpha = 0.2`` a plan whose path never leaves the set stopped as converged
    38% above the one the conditions alone give.

    A diagonal metric does not solve an ill-conditioned round, and rounds that each run out of
    descent steps end where rounding leaves them: on an unstable plant at ``dt = 1`` a warm start
    moved by 1e-15 moved the task cost from 12.20 to 14.16, against an optimum of 11.39, and one
    such end fell short of the condition by 5e-5 at a step, so the audit certified 1 step of 12.
    Where the rounds end with a held column short of the audit's condition itself, the plan is
    projected onto the held columns at the back-off (:func:`_restore`), and its status reads
    ``max_iterations``. Where no projection holds them, the status stays the rounds',
    ``max_iterations`` or ``infeasible``: a projection that fails from a far point of a curved
    condition proves nothing of the condition, and :attr:`CausalPlan.safety` reports the steps
    it fails. The plan is then the one, of the rounds' iterates and the projection onto the columns
    of the longest leading run of steps a projection holds, that meets the audit's condition over
    the most leading steps, the rounds' last where none does better: under a deviation tube the
    rounds on a confounded integrator ran out of steps with the last iterate meeting the condition
    at no step, while a projection held it at 7 of 12. A plan the rounds did not end as
    ``converged`` that meets the condition at every step is the cheapest that does.

    ``multipliers`` seed the rounds in place of zeros -- a receding horizon's last, shifted. The
    penalty is chosen afresh all the same: one grown for the last state left the first round
    ill-conditioned, and cost more descent steps than warm multipliers saved.

    Returns:
        The actions, the status as above, the accepted descent steps over every descent,
        ``start``'s included, the multipliers the rounds ended with, zeros when ``start`` came back
        untouched, and the condition they held, with those multipliers.
    """
    actions = start.actions
    lo = broadcast_box(u_lo, actions.shape, "u_lo", actions.dtype)
    hi = broadcast_box(u_hi, actions.shape, "u_hi", actions.dtype)
    check_box(lo, hi)
    _refuse_unheldable(model, x0, dt, barrier.barrier, actions, (lo, hi))
    blocks = _constraint_blocks(constraints, lo, hi, actions.dtype, disjoint=True)
    several = isinstance(barrier.barrier, tuple)
    columns = (2 * len(barrier.barrier),) if several else (2,)
    shape, inverse = None, None
    if barrier.u_max is not None:
        reach = _reach(barrier.u_max, actions.shape[1])
        shape, inverse, _ = _action_shape(reach)
        moved = np.max(np.maximum(np.abs(np.asarray(lo)), np.abs(np.asarray(hi))), axis=0)
        if inverse is not None:
            _act_within(reach, moved > 0.0, "the box lets it move")
    held = _HeldBarrier(
        model=model,
        barrier=barrier.barrier,
        dt=dt,
        alpha=barrier.alpha,
        delta=jnp.asarray(
            confounding_robust_inflation(barrier.cvar_gap, 0.0, barrier.gamma)
            * _channel_size(model, x0, actions.shape[1], shape),
            actions.dtype,
        ),
        inverse=None if inverse is None else jnp.asarray(inverse, actions.dtype),
        width=None
        if barrier.sensitivity is None
        else _ChannelWidth.build(
            barrier.sensitivity, actions.shape[1], shape, barrier.gamma, actions.dtype
        ),
        multipliers=jnp.zeros(actions.shape[:1] + columns, actions.dtype),
        weight=jnp.asarray(1.0, actions.dtype),
        backoff=jnp.asarray(0.0, actions.dtype),
        factors=jnp.ones(columns, actions.dtype),
        rest=uncertainty,
        rest_weight=lam_unc,
    )

    def excess_at(actions: Array) -> tuple[float, NDArray[np.float64]]:
        nonlocal held
        scale, _ = _barrier_state(model, x0, actions, dt, held)
        scales = np.asarray(scale, dtype=np.float64)
        conditions, paths = scales[0::2], scales[1::2]
        largest = float(np.max(conditions))
        # a column with no term at all has no units to read, and keeps the caller's; one barrier's
        # condition keeps its own, as before the next states joined it
        reference = np.where(np.arange(scales.size) % 2 == 0, largest, largest or np.max(paths))
        factors, positive = np.ones_like(scales), (scales > 0.0) & (reference > 0.0)
        if not several:
            positive[0] = False
        factors[positive] = reference[positive] / scales[positive]
        held = eqx.tree_at(lambda term: term.factors, held, jnp.asarray(factors, actions.dtype))
        _, shortfall = _barrier_state(model, x0, actions, dt, held)
        backoff = _BARRIER_BACKOFF * largest
        return backoff, np.asarray(shortfall, dtype=np.float64) + backoff

    # The next states join where the path leaves the set, as rows join an active set: holding them
    # all from the start stiffened the rounds where no next state binds, and a plan at dt = 1 under
    # alpha = 0.2 stopped as converged 38% above the one the condition alone gives, whose path never
    # leaves the set.
    count = len(barrier.barrier) if several else 1
    held = eqx.tree_at(
        lambda term: term.paths,
        held,
        jnp.zeros((actions.shape[0], count), actions.dtype),
        is_leaf=lambda leaf: leaf is None,
    )
    backoff, excess = excess_at(actions)
    if multipliers is not None:  # a receding horizon's next states that bound stay held
        held = _held_paths(held, np.asarray(multipliers)[:, 1::2] > 0.0)
        _, shortfall = _barrier_state(model, x0, actions, dt, held)
        excess = np.asarray(shortfall, dtype=np.float64) + backoff
    status: SolverStatus = start.status
    iterations, rounds, weight = start.iterations, 0, math.nan
    measure = math.inf
    multipliers = np.zeros_like(excess) if multipliers is None else multipliers
    problem = _PlanProblem(
        model=model,
        x0=x0,
        cost=cost,
        dt=dt,
        u_lo=u_lo,
        u_hi=u_hi,
        constraints=tuple(constraints),
        support=support,
        lam_supp=lam_supp,
        uncertainty=uncertainty,
        lam_unc=lam_unc,
        barrier=True,
    )
    horizon = actions.shape[0]
    best: tuple[int, float, Array] | None = None

    def score(at: Array) -> tuple[int, float, Array]:
        """The leading steps at which ``at`` meets the audit's condition, its cost, and itself."""
        every = eqx.tree_at(lambda term: term.paths, held, jnp.ones_like(held.multipliers[:, 1::2]))
        _, shortfall = _barrier_state(model, x0, at, dt, every)
        met = np.all(np.asarray(shortfall) <= 0.0, axis=1)
        prefix = int(met.size if met.all() else np.argmin(met))
        return prefix, float(_task_cost(model, x0, at, dt, cost)), at

    def better(one: tuple[int, float, Array], other: tuple[int, float, Array] | None) -> bool:
        """More leading steps certified, or every step and a lower cost."""
        if other is None:
            return True
        return one[0] > other[0] or (one[0] == other[0] == horizon and one[1] < other[1])

    def keep(at: Array) -> None:
        nonlocal best
        if better(scored := score(at), best):
            best = scored

    cleared, spent = False, 0
    while True:
        if float(np.max(excess)) <= 0.0:
            left = _left_set(model, x0, actions, dt, held, backoff)
            if not left.any():
                cleared = True
                break
            held = _held_paths(held, left)
            _, shortfall = _barrier_state(model, x0, actions, dt, held)
            excess = np.asarray(shortfall, dtype=np.float64) + backoff
        if spent >= steps:  # the rounds on one set of held columns share one budget
            status = "max_iterations"
            break
        if math.isnan(weight):
            # Birgin & Martinez's start, the penalty term as large as the cost, without their
            # floors of 1 on the cost and on the shortfall: a floor reads their units, and set the
            # penalty of a cost in thousandths a thousand times too high. A cost of zero has no
            # size to read, and keeps 1. On the host: indexing compiles per length.
            task = abs(float(np.asarray(start.cost_history)[-1]))
            level = task if 0.0 < task < math.inf else 1.0
            weight = 10.0 * level / (0.5 * float(np.sum(np.maximum(excess, 0.0) ** 2)))
            ceiling = weight * _PENALTY_RANGE
        actions, status, taken, rounds, weight, measure, excess, multipliers, held = _rounds(
            model,
            x0,
            dt,
            cost,
            lo,
            hi,
            blocks,
            actions,
            held,
            excess,
            multipliers,
            weight,
            ceiling,
            backoff,
            steps=steps,
            budget=steps - spent,
            support=support,
            lam_supp=lam_supp,
            problem=problem,
            keep=keep,
        )
        iterations += taken
        # rounds that held their columns leave a new problem where a next state joins them
        spent = 0 if status == "converged" else spent + taken
        left = _left_set(model, x0, actions, dt, held, backoff)
        if not left.any():
            break
        held = _held_paths(held, left)
        _, shortfall = _barrier_state(model, x0, actions, dt, held)
        excess = np.asarray(shortfall, dtype=np.float64) + backoff
    if cleared and math.isnan(weight):
        return actions, start.status, start.iterations, np.zeros_like(excess), held
    _, shortfall = _barrier_state(model, x0, actions, dt, held)
    restored = False
    if float(np.max(shortfall)) > 0.0:  # short of the audit's condition, not only of the aim
        while (
            moved := _restore(model, x0, dt, cost, lo, hi, blocks, actions, held, backoff)
        ) is not None:
            actions, restored = moved, True
            left = _left_set(model, x0, actions, dt, held, backoff)
            if not left.any():
                break
            held = _held_paths(held, left)
        _, shortfall = _barrier_state(model, x0, actions, dt, held)
        if restored and float(np.max(shortfall)) <= 0.0:
            status = "max_iterations"
        else:  # the longest leading run of steps a projection holds
            every = eqx.tree_at(
                lambda term: term.paths, held, jnp.ones_like(held.multipliers[:, 1::2])
            )
            low, high = 0, horizon - 1
            while low < high:
                middle = (low + high + 1) // 2
                moved = _restore(
                    model, x0, dt, cost, lo, hi, blocks, actions, every, backoff, upto=middle
                )
                if moved is None:
                    high = middle - 1
                else:
                    low = middle
                    keep(moved)
    if status != "converged" and best is not None and better(best, score(actions)):
        actions = best[2]
        if best[0] == horizon:
            status = "max_iterations"
    _log.info(
        "barrier held" if status == "converged" else "barrier rounds stopped short",
        extra={
            "chc_event": "barrier",
            "status": status,
            "rounds": rounds,
            "penalty": weight,
            "measure": measure,
            "backoff": backoff,
            "restored": restored,
            "active_steps": int(np.sum(np.any(multipliers.reshape(len(multipliers), -1) > 0.0, 1))),
        },
    )
    held = eqx.tree_at(lambda term: term.multipliers, held, jnp.asarray(multipliers, actions.dtype))
    return actions, status, iterations, multipliers, held


def _held_paths(held: _HeldBarrier, more: NDArray[np.bool_]) -> _HeldBarrier:
    """``held`` with the next states ``more``, ``(steps, barriers)``, held as well."""
    paths = np.maximum(np.asarray(held.paths, dtype=np.float64), more.astype(np.float64))
    return eqx.tree_at(lambda term: term.paths, held, jnp.asarray(paths, held.weight.dtype))


def _left_set(
    model: Dynamics, x0: Array, actions: Array, dt: float, held: _HeldBarrier, backoff: float
) -> NDArray[np.bool_]:
    """``(steps, barriers)``: the next states not yet held whose ``h``, scaled as its column,
    falls short of the back-off."""
    every = eqx.tree_at(lambda term: term.paths, held, jnp.ones_like(held.multipliers[:, 1::2]))
    _, shortfall = _barrier_state(model, x0, actions, dt, every)
    reached = np.asarray(shortfall, dtype=np.float64)[:, 1::2] + backoff
    return (reached > 0.0) & (np.asarray(held.paths) <= 0.0)


def _restore(
    model: Dynamics,
    x0: Array,
    dt: float,
    cost: QuadraticCost,
    lo: Array,
    hi: Array,
    blocks: Blocks,
    actions: Array,
    held: _HeldBarrier,
    backoff: float,
    upto: int | None = None,
) -> Array | None:
    """``actions`` moved least, in the planner's variables ``v = sigma * u`` read there
    (:func:`chc.control._action_units`), to where every column ``held`` holds is within the
    back-off, the box and the rows kept; None where no move holds them. ``upto`` holds the columns
    of the steps before it alone.

    Each step is the projection onto the columns linearised at the last one, a least-distance
    program solved exactly by Lawson & Hanson's reduction to one bounded least squares (1974,
    ch. 23); on a plant affine in the state and the action with an affine barrier and no radius
    the columns are affine, and one step is the answer. Steps stop once every column is within
    half the back-off, or after ``_RESTORE_STEPS``, where the audit's condition decides.
    """
    shape, dtype = actions.shape, actions.dtype
    sigma = np.asarray(_descent_units(model, x0, actions, dt, cost, lo, hi)[0], np.float64).ravel()
    low, high = np.asarray(lo, np.float64).ravel(), np.asarray(hi, np.float64).ravel()
    eye = np.eye(sigma.size)
    current = actions

    def held_at(at: Array) -> NDArray[np.float64]:
        shortfall = np.array(_barrier_state(model, x0, at, dt, held)[1], np.float64)
        if upto is not None:
            shortfall[upto:] = -np.inf
        return shortfall.ravel()

    for _ in range(_RESTORE_STEPS):
        excess = held_at(current) + backoff
        if float(np.max(excess)) <= 0.5 * backoff:
            return current
        flat = np.asarray(current, np.float64).ravel()
        on = np.isfinite(excess)
        slopes = np.asarray(_shortfall_jacobian(held, model, x0, current, dt), np.float64)
        # the move d holds G d >= g, row by row: the columns, the box's finite sides, the rows'
        rows = [-slopes.reshape(excess.size, -1)[on]]
        sides = [excess[on]]
        rows += [eye[np.isfinite(low)], -eye[np.isfinite(high)]]
        sides += [(low - flat)[np.isfinite(low)], (flat - high)[np.isfinite(high)]]
        for matrix, lower, upper, _ in blocks:
            matrix = np.asarray(matrix, np.float64)
            lower, upper = np.asarray(lower, np.float64), np.asarray(upper, np.float64)
            rows += [matrix[np.isfinite(lower)], -matrix[np.isfinite(upper)]]
            sides += [(lower - matrix @ flat)[np.isfinite(lower)]]
            sides += [(matrix @ flat - upper)[np.isfinite(upper)]]
        scaled = np.vstack(rows) / sigma  # in w = sigma * d, the program is argmin |w|
        stacked = np.vstack([scaled.T, np.concatenate(sides)[None, :]])
        target = np.zeros(sigma.size + 1)
        target[-1] = 1.0
        weights = lsq_linear(stacked, target, bounds=(0.0, np.inf), method="bvls").x
        residual = stacked @ weights - target
        if not abs(residual[-1]) > 1e-12:  # the linearised columns, box and rows have no point
            return None
        moved = np.clip(flat - residual[:-1] / residual[-1] / sigma, low, high)
        current = jnp.asarray(moved.reshape(shape), dtype)
    return current if float(np.max(held_at(current))) <= 0.0 else None


def _rounds(
    model: Dynamics,
    x0: Array,
    dt: float,
    cost: QuadraticCost,
    lo: Array,
    hi: Array,
    blocks: Blocks,
    actions: Array,
    held: _HeldBarrier,
    excess: NDArray[np.float64],
    multipliers: NDArray[np.float64],
    weight: float,
    ceiling: float,
    backoff: float,
    *,
    steps: int,
    budget: int,
    support: SupportModel | None,
    lam_supp: float,
    problem: _PlanProblem,
    keep: Callable[[Array], None],
) -> tuple[
    Array,
    SolverStatus,
    int,
    int,
    float,
    float,
    NDArray[np.float64],
    NDArray[np.float64],
    _HeldBarrier,
]:
    """:func:`_hold_barrier`'s augmented-Lagrangian rounds on the columns ``held`` holds: the
    actions, the status, the descent steps taken, the rounds, the penalty, the measure, the
    excess, the multipliers and the condition they end with.

    The rounds' descents take at most ``budget`` steps between them, each one compiled for
    ``steps`` and given half of what is left, so one stalled round leaves the next ones a share: a
    first round on the oscillator under a ``gamma = 2`` floor took all 10 000 steps it was given,
    where the nine after it took 238. A round its share cut grows no penalty: it says nothing of
    whether the condition can be met, and the penalty grown on such rounds stiffened the next one,
    which then stepped to a far point of the condition. They stop as ``converged`` only where the
    plan's stationarity (:attr:`CausalPlan.stationarity_status`) is within its tolerance as well,
    read on ``problem`` with the multipliers they end with, and a round its share cut stops them so
    too, since that check reads the plan, not the descent: under a deviation tube on that
    oscillator a cut round ended within both, and the plan read ``max_iterations``."""
    status: SolverStatus = "max_iterations"
    iterations, rounds = 0, 0
    measure, previous = math.inf, math.inf
    for rounds in range(1, _BARRIER_ROUNDS + 1):
        if iterations >= budget:
            rounds -= 1
            break
        cap = max(1, (budget - iterations) // 2)
        held = eqx.tree_at(
            lambda term: (term.multipliers, term.weight, term.backoff),
            held,
            (
                jnp.asarray(multipliers, actions.dtype),
                jnp.asarray(weight, actions.dtype),
                jnp.asarray(backoff, actions.dtype),
            ),
        )
        loose = measure > backoff
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
            1.0,
            _LOOSE_TOL if loose else _DESCENT_TOL,
            held,
            1.0,
            blocks,
            False,
            _round_metric(model, x0, actions, dt, cost, lo, hi, held, excess),
            jnp.asarray(cap),
        )
        iterations += int(taken)
        exhausted = int(taken) >= cap
        keep(actions)
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
        within = measure <= 0.5 * backoff
        if within and not loose and _stationary(problem, held, actions, multipliers):
            status = "converged"
            break
        if exhausted:  # neither a solve nor a sign of the condition: no growth
            pass
        elif measure > 0.5 * previous and not within:
            if weight * _PENALTY_GROWTH > ceiling:
                if float(np.max(excess)) > backoff:  # short of the audit's condition, not the aim
                    status = "infeasible"
                break
            weight *= _PENALTY_GROWTH
        previous = measure
    return actions, status, iterations, rounds, weight, measure, excess, multipliers, held


def _stationary(
    problem: _PlanProblem, held: _HeldBarrier, actions: Array, multipliers: NDArray[np.float64]
) -> bool:
    """Whether the plan's stationarity at ``actions``, under the condition ``held`` holds with
    ``multipliers``, is within its tolerance, as the finished plan reports it."""
    held = eqx.tree_at(lambda term: term.multipliers, held, jnp.asarray(multipliers, actions.dtype))
    stationarity, tolerance, _, _ = _diagnostics(replace(problem, held=held), actions)
    return _diagnostic_status(stationarity, tolerance) == "within"


def _refuse_unheldable(
    model: Dynamics,
    x0: Array,
    dt: float,
    barrier: Barrier,
    actions: Array,
    box: tuple[Array, Array],
) -> None:
    """Refuse, before the rounds spend a descent on it, what :func:`certify_safety` refuses of
    the finished plan: a path from ``actions``, where the rounds start, that leaves the float
    range, which leaves the condition nothing to read; and a model whose field is not affine in
    the action at ``actions`` or at either side of the ``box`` where it is finite
    (:func:`_refuse_curved`). The audit raised both only after the rounds: 20 s of them on a Hill
    curve, and on a path that overflows, rounds whose every shortfall was nan."""
    trajectory = _trajectory(model, x0, actions, dt)
    finite = np.all(np.isfinite(np.asarray(trajectory)), axis=-1)
    if not finite.all():
        raise ValueError(
            f"the plan's predicted state {int(np.argmin(finite))} is not a finite number, and no "
            "barrier condition reads there"
        )
    probes = [actions] + [jnp.where(jnp.isfinite(side), side, 0.0) for side in box]
    for one in barrier if isinstance(barrier, tuple) else (barrier,):
        _refuse_curved(model, one, trajectory[:-1], probes, dt)


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

    So ``gamma_star = inf`` beside few or no certified steps is no contradiction: where the drift
    alone holds every step, or the radius has no part a level widens (a sensitivity with no
    confounding), the ceiling is ``inf``, while a certified step needs this plan's own action to
    clear the condition and its path, widened by ``deviation``, to stay in the set.
    """

    barrier_values: Array  # h(x_k) along the nominal trajectory, k = 0 .. horizon-1
    # a_k + <w_k, u_k> - d_k*||u_k||, the worst case AT the plan; with several barriers, the one
    # with the least room's at each step, as is ``required``
    guaranteed_derivative: Array
    required: Array  # -alpha * h(x_k); certification asks guaranteed >= required
    # bool per step: the condition holds, and the path is in the safe set at the step's state and
    # at the next
    planned_certified: Array
    certified_steps: int  # leading certified PREFIX, as in ``certified_horizon``, not a pass count
    gamma_star: float  # weakest step's §40 ceiling; nan if some step certifies at no Gamma at all
    step_gamma_star: tuple[float, ...]  # per-step ceilings, so the weak link is locatable
    # largest d_k = Delta(Gamma)*||grad h(x_k)|| along the plan, Delta in the channel's units
    radius: float
    # what the radius is: ``relative``, a share of the fitted channel's own size, which no level
    # takes past the estimate's size from it; ``absolute``, a sensitivity model's bound on the
    # channel's error (``sensitivity=``), which can hold a channel far from its estimate
    radius_model: Literal["relative", "absolute"]
    # e_0 .. e_H: how far the path of a plant whose channel is within the radius of the model's
    # can be from the model's predicted states under the plan's actions; a step is certified only
    # where h less ||grad h|| times it is not negative at the step's state and at the next
    deviation: Array


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


def _channel_size(
    model: Dynamics, x0: Array, levers: int, shape: NDArray[np.float64] | None = None
) -> float:
    """``||B_hat(x0) D||_2``, the fitted channel's operator norm at the plan's first state, each
    lever's column times its entry of ``shape`` (:func:`_action_shape`), ``D`` the identity where
    it is None: what ``cvar_gap`` is a share of, so the identification radius is in the channel's
    own units and ``gamma_star`` reads the same in any units of the state or of the levers.

    Raises:
        ValueError: if it is not a finite number, which scales no radius.
    """
    channel = jax.jacobian(model, argnums=2)(0.0, x0, jnp.zeros(levers, x0.dtype))
    if shape is not None:
        channel = channel * jnp.asarray(shape, channel.dtype)
    size = float(jnp.linalg.norm(channel, 2))
    if not math.isfinite(size):
        raise ValueError(
            f"the fitted channel at the plan's first state has norm {size}, which scales no radius"
        )
    return size


def _guaranteed(
    drift: Array,
    channel: Array,
    grad_norm: Array,
    actions: Array,
    delta: float | Array,
    inverse: Array | NDArray[np.float64] | None = None,
) -> Array:
    """``a + <w, u> - d ||D^+ u||``, ``d = delta ||grad h||``: the worst case at the planned action,
    ``D^+`` the levers' ``inverse`` (:func:`_action_shape`), the identity where it is None."""
    spread = delta * grad_norm * _norm(_shaped(actions, inverse))
    return drift + jnp.einsum("km,km->k", channel, actions) - spread


def _shaped(actions: Array, inverse: Array | NDArray[np.float64] | None) -> Array:
    return actions if inverse is None else actions * jnp.asarray(inverse, actions.dtype)


def _spectral(matrix: Array) -> Array:
    """``||M||_2``, differentiated as the top singular pair's ``u1' dM v1``: a subgradient where the
    top singular value repeats, as on the identity, where an SVD's derivative is nan."""
    _, vectors = jnp.linalg.eigh(jax.lax.stop_gradient(matrix.T @ matrix))
    return _norm(matrix @ vectors[:, -1])


def _step_gains(
    model: Dynamics, t: Array, x: Array, u: Array, dt: float
) -> tuple[Array, Array, Array]:
    """``||dF/dx||``, ``||dF/dc||`` and the action the field takes, of the one-step map ``F`` that
    :func:`chc.integrate.rk4_step` steps ``model`` by from ``x`` under ``u``, ``c`` a constant added
    to the field at every stage: how a gap in the state, and a channel error ``E u`` held over the
    step, reach the next state."""

    def stepped(x: Array, shift: Array) -> Array:
        field, taken = model.at_step(x, u) if isinstance(model, _SetAtStep) else (model, u)
        if isinstance(field, _EulerStep):
            return x + dt * (field(t, x, taken) + shift)
        k1 = field(t, x, taken) + shift
        k2 = field(t + 0.5 * dt, x + 0.5 * dt * k1, taken) + shift
        k3 = field(t + 0.5 * dt, x + 0.5 * dt * k2, taken) + shift
        k4 = field(t + dt, x + dt * k3, taken) + shift
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    zero = jnp.zeros_like(x)
    growth = jax.jacfwd(stepped, argnums=0)(x, zero)
    gain = jax.jacfwd(stepped, argnums=1)(x, zero)
    taken = model.at_step(x, u)[1] if isinstance(model, _SetAtStep) else u
    return _spectral(growth), _spectral(gain), taken


def _deviation_tube(
    model: Dynamics,
    states: Array,
    actions: Array,
    dt: float,
    delta: float | Array,
    inverse: Array | NDArray[np.float64] | None = None,
) -> Array:
    """``e_0 .. e_H``: how far from the model's states ``states`` (``x_0 .. x_{H-1}``) the path of a
    plant whose channel is within ``delta`` of the model's at each step can be, stepped by the same
    one-step map under the same ``actions``:

        ``e_{k+1} = ||dF/dx|| e_k + ||dF/dc|| delta_k ||D^+ u_k||``,  ``e_0 = 0``

    (:func:`_step_gains`), ``D^+`` the levers' ``inverse``. On a model affine in the state with a
    channel error that does not move with it, it bounds the gap, and on one state, one lever and
    actions of one sign the plant at the radius's edge reaches it
    (``validation/barrier_deviation_tube.mac``); elsewhere it reads the map's slope and the radius
    at the model's state, a bound to first order in the gap."""
    steps = states.shape[0]
    radii = jnp.broadcast_to(jnp.asarray(delta, states.dtype), (steps,))

    def carried(_: None) -> Array:
        times = dt * jnp.arange(steps, dtype=states.dtype)
        growth, gain, taken = jax.vmap(lambda t, x, u: _step_gains(model, t, x, u, dt))(
            times, states, actions
        )
        pushes = gain * radii * _norm(_shaped(taken, inverse))

        def carry(gap: Array, step: tuple[Array, Array]) -> tuple[Array, Array]:
            rate, push = step
            reached = rate * gap + push
            return reached, reached

        reached = jax.lax.scan(carry, jnp.zeros((), states.dtype), (growth, pushes))[1]
        return jnp.concatenate([jnp.zeros((1,), states.dtype), reached])

    # a radius of zero, the ordinary barrier's at gamma = 1, opens no tube: skip the slopes
    return jax.lax.cond(
        jnp.any(radii != 0.0),
        carried,
        lambda _: jnp.zeros((steps + 1,), states.dtype),
        None,
    )


# the audit runs eagerly, and an eager scan or cond compiles anew on every call: a receding horizon
# would recompile the tube at every step
_compiled_tube = eqx.filter_jit(_deviation_tube)


def _barrier_room(barrier: Callable[[Array], Array], states: Array, deviation: Array) -> Array:
    """``h(x_k) - ||grad h(x_k)|| e_k`` at each state: the least ``h`` a plant ``e_k`` from ``x_k``
    can have, exactly on an affine ``h`` and to first order in ``e_k`` otherwise."""

    def scalar_h(x: Array) -> Array:
        return jnp.squeeze(barrier(x))

    values = jax.vmap(scalar_h)(states)
    return values - _norm(jax.vmap(jax.grad(scalar_h))(states)) * deviation


def _reach(u_max: ArrayLike, levers: int | None = None) -> NDArray[np.float64]:
    """Each lever's largest action, checked: none nan or negative, one positive at least, and one
    per lever where ``levers`` is given. ``inf`` is a free side's.

    Raises:
        ValueError: otherwise, since no action has such a size.
    """
    reach = np.asarray(u_max, dtype=np.float64)
    if reach.ndim != 1 or (levers is not None and reach.shape[0] != levers):
        expected = "one per lever" if levers is None else f"{levers}, one per lever"
        raise ValueError(
            f"u_max has shape {reach.shape}; each lever's largest action is {expected}"
        )
    if not (np.all(reach >= 0.0) and np.any(reach > 0.0)):  # nan compares false
        raise ValueError(
            f"u_max={reach.tolist()} is no set of actions' sizes: each is a number not negative, "
            "and one is positive at least"
        )
    return reach


def _act_within(reach: NDArray[np.float64], moved: NDArray[np.bool_], where: str) -> None:
    """Refuse a lever whose largest action is zero but which ``moved``: no action set holds it."""
    stuck = np.flatnonzero((reach == 0.0) & moved)
    if stuck.size:
        raise ValueError(
            f"lever {int(stuck[0])} has a largest action of 0 in u_max, but {where}; its actions "
            "lie outside the set the radius is read on"
        )


def _action_shape(
    reach: NDArray[np.float64],
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float]:
    """The action set ``||D^+ u|| <= U`` whose semi-axes are each lever's largest action: ``D``,
    each lever's ``reach`` over the largest ``U``, its inverse ``D^+``, zero where a lever has no
    action, and ``U``. A free side has no largest action to read its lever's units by: ``D`` and
    ``D^+`` are then None, the levers' own units read as one, and ``U`` is ``inf``, as before.

    With lever ``j`` in units ``c_j`` times its own, its reach moves as ``c_j``, its column of the
    channel as ``1 / c_j``, and ``U`` with whichever lever is the largest: the audit's every
    product of the channel by ``D``, and of ``D^+`` by an action, then moves by ``U``'s factor
    alone, which the audit reads none of (ADR 0085). One lever, or levers whose reaches are equal,
    have ``D = D^+ = 1`` exactly, ``x / x`` being 1 in binary floating point, and the ball of
    radius ``U`` they had.
    """
    largest = float(np.max(reach))
    if largest == math.inf:
        return None, None, largest
    shape = reach / largest
    inverse = np.divide(largest, reach, out=np.zeros_like(reach), where=reach > 0.0)
    return shape, inverse, largest


def certify_safety(
    plan: CausalPlan,
    model: Dynamics,
    barrier: Barrier,
    dt: float,
    *,
    alpha: float = 1.0,
    gamma: float = 1.0,
    cvar_gap: float = 1.0,
    u_max: ArrayLike | None = None,
    sensitivity: ChannelSensitivity | None = None,
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
            be a JAX-traceable scalar function. A tuple of them is audited each on its own and
            read as one: a step is certified where every barrier's condition holds there, and
            the step's ``barrier_values`` are the least barrier's, its ``guaranteed_derivative``
            and ``required`` the pair of the barrier with the least room, and its
            ``step_gamma_star`` the least of theirs -- an upper bound rather than the step's own
            ceiling, since each is reached by its own best action and no one action need reach
            the least.
        alpha: the class-K gain in ``grad h . xdot >= -alpha*h``.
        gamma, cvar_gap: the marginal sensitivity model's level and the gap its factor scales,
            combined into ``Delta = (gamma-1)/(gamma+1) * cvar_gap * ||B_hat(x0)||_2``:
            ``cvar_gap`` is a *share* of the model's channel, its operator norm at the plan's first
            state, so the radius is in the channel's units and ``gamma_star`` reads the same in any
            units of the state or of the levers. The default, 1, is the channel's own size, the
            calibration ``cvar_gap := b_hat`` of :mod:`chc.research.sensitivity`. ``gamma = 1`` is
            exact identification -- zero radius, and this degenerates to the ordinary CBF check.
            NOTE the calibration burden: ``Delta`` is used here as an **operator-norm radius on the
            effect matrix**, and §32's scalar CVaR gap does not by itself establish one. The share
            is the caller's externally calibrated input, exactly as ``gamma`` is; an absolute
            radius ``r`` is the share ``r / ||B_hat(x0)||_2``. The radius is a share of the
            *fitted* channel, so at every ``gamma``, however large, it stays within
            ``cvar_gap * ||B_hat(x0)||_2`` of the estimate: ``gamma_star = inf`` reads every level
            of that set, and nothing of an effect further off. With the default share an estimate
            of 0.004 for an effect of 1 is certified at every level along a plan the effect itself
            takes out of the set. The factor ``(gamma-1)/(gamma+1)`` is the bounded-density-ratio
            model's (:func:`chc.uncertainty.confounding_robust_inflation`), but a channel radius
            spent as ``d*||u||`` is a structural sensitivity model of its own, not that model's
            bound. Read so, ``gamma`` and ``gamma_star`` are a relative margin, not a level of
            confounding; ``sensitivity`` gives the radius a sensitivity model's meaning.
        u_max: actuation limit, used for ``gamma_star`` -- that is the *best admissible action*
            question, which needs a budget the plan's own choice does not define. Defaults to the
            largest action norm the plan actually uses, answering the narrower "how much confounding
            does this plan's own authority tolerate". One number per lever, each lever's largest
            action in its own units, reads the whole audit in units of each lever's own largest
            action: the action set is the ellipsoid ``||D^+ u|| <= U`` whose semi-axes are those
            numbers, ``U`` the largest and ``D`` each over ``U``, the radius ``Delta`` is on
            ``(B_hat - B) D`` and sized by ``||B_hat(x0) D||_2``, and an action's norm is
            ``||D^+ u||``. That reads the same in any units of each lever, where one number reads
            the levers' units as one: on a log of one state and two levers in ``[-2, 2]``, either
            lever logged in units 1e3 or 1e-3 times its own, its box with it, the barrier's
            certified steps fell from 3 to 0. One lever, or levers whose numbers are equal, read
            the ball of radius ``U`` they read with ``u_max = U``, bit for bit. A lever whose
            number is 0 has no action in the set, and the plan must not move it. A number ``inf``,
            a free side's, leaves no units to read that lever by, and the audit reads the levers'
            own units, as ``u_max = inf`` does. The set is symmetric about zero: where the
            levers' box is not, it admits actions the box does not, and ``gamma_star`` read on it
            can name a level no admissible action holds. :func:`causal_plan` and
            :func:`chc.prescribe` read it on their box as well.
        sensitivity: an absolute sensitivity model of the channel (:class:`ChannelSensitivity`),
            in place of ``cvar_gap``'s share: the radius at step ``k`` is
            ``Delta(x_k, gamma) = nu(x_k) (sampling + kappa(gamma) confounding)``, read at the
            step's predicted state, a bound on the channel's error there when the levers' density
            given a hidden confounder moves by a factor within ``[1/gamma, gamma]``, at the
            sensitivity's ``level``. ``gamma = 1`` is then the sampling error alone, not exact
            identification, and ``gamma_star`` is each step's largest ``gamma`` under that model:
            below ``(3 + sqrt 5)/2``, or ``inf`` where the drift alone keeps the step. On logs
            whose levers follow a confounder at ``gamma = 1.5``, a plan held and certified under
            the fit's own sensitivity at that level kept the true plant in the set over its
            certified steps on 20 logs of 20, on a stable drift and on an unstable one (ADR 0111,
            ADR 0119). ``gamma`` must be below ``(3 + sqrt 5)/2``. The certificate's
            ``radius_model`` reads ``"absolute"`` with a ``sensitivity`` and ``"relative"``
            without one.

    Assumes the plant is control-affine, and checks it: the condition reads the drift and the
    channel at zero action, so a model whose field, seen through ``grad h``, is not affine in the
    action at the planned actions or at those ``gamma_star`` reads is refused. Assumes the
    uncertainty set isotropic (see :mod:`chc.barrier`; ``d*||u||`` is exact for a Euclidean ball,
    an outer bound otherwise).

    What a certified step promises: every plant whose channel is within the radius of the
    model's, its drift the model's, stepped by the model's one-step map under the plan's own
    actions from the plan's start, is in the safe set at that step's state and the next. The
    condition is read at the *model's* states, and such a plant's path leaves them: under an
    exact absolute radius of 0.1004 on a drift of 0.3, a plan certified on all 60 steps by the
    condition alone took the plant at the radius's edge to ``h = -0.132``, and on an integrator at
    ``dt = 1`` one certified on all 12 took it to ``h = -0.5``. So a step is certified where the
    condition holds at its state and ``h - ||grad h|| e >= 0`` at its state and at the next, ``e``
    the deviation tube ``e_{k+1} = ||dF/dx|| e_k + ||dF/dc|| Delta_k ||D^+ u_k||`` that the
    channel's radius can open between the two paths (``deviation``,
    ``validation/barrier_deviation_tube.mac``), ``F`` the one-step map, ``c`` an error added to
    the field. The tube bounds the gap where the model is affine in the state, the channel's error
    does not move with the state (a relative radius, or a sensitivity of ``channel_degree`` 0) and
    ``h`` is affine; elsewhere it reads the map's slope, the radius and ``grad h`` at the model's
    states, and holds to first order in its own width. It is about the channel alone: an error in
    the drift, and a plant stepped by another map than the model's, are outside it. Reading ``h``
    at both ends of a step also catches a path that leaves the set between the points the
    condition is read at: at ``dt = 1`` and ``alpha = 1`` a plan that met it at every step of
    twelve ended outside the set. Between the steps nothing is read.

    Raises:
        ValueError: if ``cvar_gap`` is not positive or not finite, which would make the radius
            meaningless and :func:`chc.barrier.barrier_gamma_star` uninvertible, or the channel at
            the plan's first state is not a finite number; if ``gamma`` is
            below 1 or not finite, which is not a sensitivity level; if the actuation budget
            backing ``gamma_star`` is non-positive, which asks how much confounding a controller
            with no authority tolerates; if ``u_max`` per lever has an entry negative or nan,
            none positive, not one per lever, or a 0 at a lever the plan moves; if a state of the
            plan's predicted path is not a finite number; if the model is not affine in the
            action where the condition is read; or, with a ``sensitivity``, if ``gamma`` is at
            or past ``(3 + sqrt 5)/2`` or the sensitivity's channel has another size.
    """
    return _audit(plan, model, barrier, dt, alpha, gamma, cvar_gap, u_max, None, sensitivity)


def _audit(
    plan: CausalPlan,
    model: Dynamics,
    barrier: Barrier,
    dt: float,
    alpha: float,
    gamma: float,
    cvar_gap: float,
    u_max: ArrayLike | None,
    box: tuple[Bound, Bound] | None,
    sensitivity: ChannelSensitivity | None = None,
) -> SafetyCertificate:
    """:func:`certify_safety`, with ``gamma_star`` read on the actions ``box`` admits too.

    ``box`` is the plan's ``(u_lo, u_hi)``, or None for ``u_max``'s set alone. That set is
    symmetric about zero, so a box that is not -- a lever that may only rise, as spend does --
    admits less of it: where a step's best action of the set leaves the box, it is drawn back along
    its ray to the box's edge, and the step's ceiling read at that action's size. On one lever that
    is the box's best action, so the ceiling is exact; on several it is a lower bound.
    """
    if isinstance(barrier, tuple):
        if not barrier:
            raise ValueError("an empty tuple of barriers holds no condition")
        audits = [
            _audit(plan, model, one, dt, alpha, gamma, cvar_gap, u_max, box, sensitivity)
            for one in barrier
        ]
        return _every_barrier(audits)
    if not 0.0 < cvar_gap < math.inf:
        raise ValueError(
            f"cvar_gap must be positive and finite to scale a sensitivity radius, got {cvar_gap}"
        )
    finite = np.all(np.isfinite(np.asarray(plan.trajectory)), axis=-1)
    if not finite.all():
        raise ValueError(
            f"the plan's predicted state {int(np.argmin(finite))} is not a finite number, and no "
            "barrier condition reads there"
        )
    states, actions = plan.trajectory[:-1], plan.actions
    shape, inverse = None, None
    if u_max is None or np.ndim(u_max) == 0:
        authority = (
            float(np.asarray(u_max, dtype=np.float64))
            if u_max is not None
            else float(jnp.max(jnp.linalg.norm(actions, axis=1)))
        )
        reach = np.full(actions.shape[1], authority)
    else:
        reach = _reach(u_max, actions.shape[1])
        shape, inverse, authority = _action_shape(reach)
        if inverse is not None:
            _act_within(reach, np.any(np.asarray(actions) != 0.0, axis=0), "the plan moves it")
    if sensitivity is None:
        size = _channel_size(model, states[0], actions.shape[1], shape)
        delta: float | Array = confounding_robust_inflation(cvar_gap, 0.0, gamma) * size
        width = None
    else:
        width = _ChannelWidth.build(sensitivity, actions.shape[1], shape, gamma, states.dtype)
        delta = width(states)
    h_values, grad_norm, drift, channel = _barrier_terms(model, barrier, states, actions, dt)
    guaranteed = _guaranteed(drift, channel, grad_norm, actions, delta, inverse)
    required = -alpha * h_values
    # The condition is read at the model's states, and a plant whose channel is in the set is
    # elsewhere: under the plan's own actions its path drifts from the model's by up to the
    # deviation tube, and it is in the set where h less the tube's reach is. Read at the step's
    # state and at the next, since a step of length dt can also leave the set between the points
    # the condition is read at: at dt = 1 a plan that met it at every step of twelve ended outside.
    deviation = _compiled_tube(
        model,
        states,
        actions,
        dt,
        jnp.asarray(delta, states.dtype),
        None if inverse is None else jnp.asarray(inverse, states.dtype),
    )
    room = _barrier_room(barrier, plan.trajectory, deviation)
    certified = (guaranteed >= required) & (room[:-1] >= 0.0) & (room[1:] >= 0.0)

    if not authority > 0.0:  # a nan passed `authority <= 0` and priced every step's threshold nan
        source = "u_max" if u_max is not None else "the plan's largest action (it never acts)"
        raise ValueError(f"gamma_star needs a positive actuation budget; {source} is {authority}")
    scaled = channel if shape is None else channel * jnp.asarray(shape, channel.dtype)
    lengths = jnp.linalg.norm(scaled, axis=1)
    # the action the set's best one is along, at size one in the set's norm: U D c / ||c|| over U
    unit = jnp.where(
        lengths[:, None] > 0.0, scaled / jnp.where(lengths > 0.0, lengths, 1.0)[:, None], 0.0
    )
    if shape is not None:
        unit = unit * jnp.asarray(shape, unit.dtype)
    budgets = _box_budgets(np.asarray(unit, dtype=np.float64), authority, reach, box)
    probes = [actions] + (
        [unit * jnp.asarray(budgets, unit.dtype)[:, None]] if np.isfinite(budgets).all() else []
    )
    _refuse_curved(model, barrier, states, probes, dt)
    thresholds = [
        _threshold(float(a), float(c), float(budget), float(alpha * h))
        for a, c, h, budget in zip(drift, lengths, h_values, budgets, strict=True)
    ]
    if sensitivity is None or width is None:
        per_step = tuple(
            barrier_gamma_star(
                threshold,
                cvar_gap,
                # the gap is a share of the channel's size: carried with the gradient, where a
                # zero channel reads as a flat barrier does, a radius of zero at every level
                size * float(n),
            )
            for threshold, n in zip(thresholds, grad_norm, strict=True)
        )
    else:
        per_step = tuple(
            _sensitivity_gamma_star(threshold, float(n) * float(w), sensitivity)
            for threshold, n, w in zip(
                thresholds, grad_norm, np.asarray(width.widths(states)), strict=True
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
        radius=float(jnp.max(delta * grad_norm)),
        radius_model="relative" if sensitivity is None else "absolute",
        deviation=deviation,
    )


def _sensitivity_gamma_star(
    threshold: float, unit: float, sensitivity: ChannelSensitivity
) -> float:
    """A step's largest ``Gamma`` under ``sensitivity``: the radius ``threshold`` it can absorb, on
    ``d = unit * (sampling + kappa(Gamma) * confounding)``, ``unit`` its ``||grad h|| nu(x)``.

    ``nan`` where the sampling part alone exceeds it, ``1.0`` where it takes all of it, ``inf``
    where the threshold is (the drift alone keeps the step) or the radius cannot reach it (no
    channel error at the state, or no confounding to scale), and otherwise below
    ``(3 + sqrt 5)/2`` (``validation/channel_bias_bound.mac`` STEP 3c)."""
    if unit == 0.0:
        return math.inf if threshold >= 0.0 else math.nan
    room = threshold / unit - sensitivity.sampling
    if math.isnan(room) or room < 0.0:
        return math.nan
    if room == math.inf or sensitivity.confounding == 0.0:
        return math.inf
    if room == 0.0:
        return 1.0
    return ChannelSensitivity.level_at(room / sensitivity.confounding)


def _threshold(drift: float, channel: float, budget: float, alpha_h: float) -> float:
    """:func:`chc.barrier.identification_radius_threshold` at an action of size ``budget``; at
    size 0, no action but zero, which holds the step at every radius or at none."""
    if budget > 0.0:
        return identification_radius_threshold(
            drift=drift, channel=channel, u_max=budget, alpha_h=alpha_h
        )
    return math.inf if -alpha_h - drift <= 0.0 else math.nan


def _box_budgets(
    unit: NDArray[np.float64],
    authority: float,
    reach: NDArray[np.float64],
    box: tuple[Bound, Bound] | None,
) -> NDArray[np.float64]:
    """Per step, the largest ``t <= authority`` with ``t * unit[k]`` inside ``box``.

    Only a side of the box shorter than the set's semi-axis on its lever, ``reach``, bounds ``t``:
    a box symmetric about zero holds the whole set, and leaves ``authority`` as it was, exactly.
    A side that does not reach zero admits no step along it.
    """
    budgets = np.full(unit.shape[0], authority)
    if box is None:
        return budgets
    lo, hi = (np.broadcast_to(np.asarray(end, dtype=np.float64), unit.shape) for end in box)
    side = np.where(unit > 0.0, hi, np.where(unit < 0.0, -lo, np.inf))
    short = side < reach
    if short.any():
        ratio = np.full(unit.shape, np.inf)
        ratio[short] = np.maximum(side[short], 0.0) / np.abs(unit[short])
        budgets = np.minimum(budgets, ratio.min(axis=1))
    return budgets


def _refuse_curved(
    model: Dynamics,
    barrier: Callable[[Array], Array],
    states: Array,
    probes: Sequence[Array],
    dt: float,
) -> None:
    """Refuse a model whose field, seen through ``grad h``, is not affine in the action at
    ``probes``: the plan's actions, and the actions ``gamma_star`` reads.

    The condition is ``a + <w, u>``, the drift and the channel read at zero action, which is the
    field at ``u`` only where it is affine in ``u``. On a Hill curve of slope 2 or on ``u + u^3``
    the audit passed steps the field at the planned action failed. The defect allowed is the
    rounding of an affine field, ``sqrt(eps)`` of the terms' size.

    Raises:
        ValueError: at the first step and probe where the defect is larger.
    """
    times = dt * jnp.arange(states.shape[0])
    grad_h = jax.vmap(jax.grad(lambda x: jnp.squeeze(barrier(x))))(states)
    rest = jax.vmap(model)(times, states, jnp.zeros_like(probes[0]))
    allowed = float(jnp.sqrt(jnp.finfo(rest.dtype).eps))

    def pushed(t: Array, x: Array, u: Array) -> Array:
        return jax.jvp(lambda v: model(t, x, v), (jnp.zeros_like(u),), (u,))[1]

    for probe in probes:
        moved = jax.vmap(model)(times, states, probe)
        linear = jax.vmap(pushed)(times, states, probe)
        defect = jnp.abs(jnp.einsum("kn,kn->k", grad_h, moved - rest - linear))
        size = jnp.einsum(
            "kn,kn->k", jnp.abs(grad_h), jnp.abs(moved) + jnp.abs(rest) + jnp.abs(linear)
        )
        curved = np.flatnonzero(np.asarray(defect > allowed * size))
        if curved.size:
            raise ValueError(
                f"the model is not affine in the action at step {int(curved[0])}: the barrier "
                "condition reads the field's drift and channel at zero action, which is the field "
                "at another action only on a control-affine plant, and would certify steps the "
                "field there fails"
            )


def _every_barrier(audits: Sequence[SafetyCertificate]) -> SafetyCertificate:
    """Several barriers' audits read as one, a step certified where each is: see certify_safety."""
    room = jnp.stack([audit.guaranteed_derivative - audit.required for audit in audits])
    least = jnp.argmin(room, axis=0)[None]  # nan room is least, and certifies nothing

    def binding(values: Array) -> Array:
        return jnp.take_along_axis(values, least, axis=0)[0]

    certified = jnp.all(jnp.stack([audit.planned_certified for audit in audits]), axis=0)
    step_gamma_star = tuple(
        float(np.min(column))  # a nan in any barrier's column is the step's
        for column in np.asarray([audit.step_gamma_star for audit in audits]).T
    )
    return SafetyCertificate(
        barrier_values=jnp.min(jnp.stack([audit.barrier_values for audit in audits]), axis=0),
        guaranteed_derivative=binding(jnp.stack([audit.guaranteed_derivative for audit in audits])),
        required=binding(jnp.stack([audit.required for audit in audits])),
        planned_certified=certified,
        certified_steps=(
            int(jnp.argmin(certified)) if not bool(jnp.all(certified)) else int(certified.shape[0])
        ),
        gamma_star=(
            math.nan if any(math.isnan(g) for g in step_gamma_star) else min(step_gamma_star)
        ),
        step_gamma_star=step_gamma_star,
        radius=max(audit.radius for audit in audits),
        radius_model=audits[0].radius_model,  # one sensitivity audits every barrier
        deviation=audits[0].deviation,  # the plant's path, which no barrier moves
    )


ModulusSource = Literal["supplied", "measured"]
"""Where :attr:`PlanRegretBound.modulus` comes from: the caller, or ``J``'s Hessian. Measured, it is
the one Hessian of an objective the model's structure makes quadratic, which is the whole box's, or
the least eigenvalue at the plan and at random points of the box, a sample that cannot bound the
box's least; :attr:`PlanRegretBound.status` tells the two apart."""

RegretStatus = Literal["certified", "diagnostic", "refused"]
"""What :attr:`PlanRegretBound.bound` is. ``certified`` on a supplied modulus or a quadratic
objective's; ``diagnostic`` on a sampled modulus, which a pocket of negative curvature between the
samples breaks; ``refused``, the bound ``inf``, where the modulus came out negative or not a finite
number, or the bound itself did not come out finite."""


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

    The moves are measured lever by lever, each in its own scale (STEP 8 of the same file): the
    modulus is the curvature in the norm ``|D^-1 d|``, ``D`` one scale per lever that moves with
    that lever's units, and a move's curvature along lever ``j`` is ``mu / s_j^2``. In the plain
    norm the modulus is the cheapest lever's curvature, which charges a stiff lever's moves at a
    cheap one's, and the bound read the levers' units: lever ``u1`` in units 1e3 times its own
    took one plan's bound from 0.0266 to 0.0576, where it now reads 0.0214 in both. Measured, the
    scales come from the Hessian's diagonal (:func:`_objective_modulus`); a supplied modulus is
    the plain norm's, every scale 1.

    The bound is on the **planning objective**, which is the question the solver was asked. How far
    the planning model itself is from the plant is the error tube's question
    (:attr:`CausalPlan.uncertainty_tube`), and the two must not be added.

    It is a certificate only where the modulus is one for the whole box: supplied, or read off an
    objective the model's structure makes quadratic (:attr:`status`). A modulus sampled at the plan
    and a few points of the box is not: a pocket of negative curvature between the samples can hide
    a plan much cheaper than this one, and the bound then reads small with nothing behind it.
    """

    bound: float  # J(U) - min over the box of J where ``status`` certifies it; inf when refused
    unconstrained_bound: float  # Result 6's |grad J|^2/(2 mu), for the comparison
    frank_wolfe_gap: float  # the mu-free fallback the box guarantees; bound <= this
    modulus: float  # the mu actually used, over the actions the box lets move, in its norm
    modulus_source: ModulusSource
    gradient_norm: float  # in the dual norm, so unconstrained_bound = gradient_norm^2/(2 mu)
    per_lever: tuple[float, ...]  # the bound split by lever; sums to ``bound``
    pinned_actions: int  # coordinates the gradient holds against a bound: exactly free
    status: RegretStatus


def _objective_modulus(
    objective: Callable[[Array], Array],
    actions: Array,
    lo: Array,
    hi: Array,
    probes: int,
    seed: int,
) -> tuple[float, Array]:
    """Smallest Hessian eigenvalue of ``J`` at the plan and at ``probes`` random feasible points,
    over the actions the box lets move, each lever's actions in its own scale; and those scales,
    one per lever.

    Lever ``j``'s scale is ``s_j = (h / h_j)^(1/2)``, ``h_j`` the mean of the Hessian's diagonal
    over its moving actions at the plan and ``h`` the stiffest lever's. The modulus is the least
    eigenvalue of ``D H D``, ``D`` the scales along the actions: the curvature in the norm
    ``|D^-1 d|``, which charges each lever's moves at its own curvature. Each scale moves with its
    own lever's units, so the bound reads none of them; the Hessian's least eigenvalue is the
    cheapest lever's curvature, and in other units of one lever the bound read those units. One
    lever, or levers of equal curvature, have the scale 1 and the plain Hessian's least
    eigenvalue. A lever with no positive curvature keeps the scale 1: some diagonal entry of the
    block is then not positive, so no scale makes the modulus positive, and it reads the
    Frank-Wolfe gap or a refusal whatever the units.

    A box of zero width fixes its action, so every feasible move is zero there and the bound needs
    ``J``'s curvature along the other actions alone: the least eigenvalue of the Hessian's block on
    them, never below the whole Hessian's (Cauchy interlacing). The whole Hessian's reads an action
    nothing reads, a lever on the log's rule, as a direction of no curvature, and the bound was
    refused. With no action free every move is zero, and the modulus is 0. A lever none of whose
    actions moves takes its scale from all its actions, which only the unconstrained bound reads.

    Where the objective is quadratic (:func:`_quadratic`) the Hessian is one matrix and the plan's
    is the answer for the whole box, so no point is drawn. Elsewhere this is a sample, and a caller
    who can bound the curvature should pass ``modulus`` instead. Sampling is what makes a
    non-convex objective *visible* -- a single evaluation at the plan sits at a solver's stopping
    point, which is the least likely place to find the negative curvature -- but it proves nothing
    about the points it did not draw. The scales are the plan's at every point drawn: any positive
    scales give a valid bound, and only their tightness depends on where they are read.
    """
    flat = actions.reshape(-1)
    mask = np.asarray(hi > lo).reshape(actions.shape)
    moving = np.flatnonzero(mask.reshape(-1))
    if moving.size == 0:
        return 0.0, jnp.ones(actions.shape[1:], dtype=actions.dtype)
    drawn = jax.random.uniform(
        jax.random.PRNGKey(seed),
        (max(probes, 0), flat.size),
        minval=lo.reshape(-1),
        maxval=hi.reshape(-1),
    )

    def hessian(point: Array) -> Array:
        return jax.hessian(lambda v: objective(v.reshape(actions.shape)))(point)

    # vmapped rather than looped: the Hessian of a rollout is expensive to TRACE, and a Python
    # loop retraces it once per probe. Batching pays that cost once for the whole sample.
    hessians = jax.vmap(hessian)(jnp.concatenate([flat[None, :], drawn]))
    diagonal = np.asarray(jnp.diagonal(hessians[0])).reshape(actions.shape)
    read = np.where(mask.any(axis=0), mask, True)  # a lever that cannot move reads all its own
    curvature = np.sum(np.where(read, diagonal, 0.0), axis=0) / np.sum(read, axis=0)
    stiff = curvature > 0.0  # a nan curvature is not, and leaves its nan in the Hessian
    reference = stiff & mask.any(axis=0)
    if reference.any():
        stiffest = np.max(curvature[reference])
        scale = np.where(stiff, np.sqrt(stiffest / np.where(stiff, curvature, 1.0)), 1.0)
    else:  # no lever that moves has a positive curvature: no scale makes the modulus positive
        scale = np.ones_like(curvature)
    along = jnp.asarray(np.broadcast_to(scale, actions.shape).reshape(-1)[moving], actions.dtype)
    block = hessians[:, moving][:, :, moving] * along[None, :, None] * along[None, None, :]
    least = jnp.min(jnp.linalg.eigvalsh(0.5 * (block + jnp.swapaxes(block, 1, 2))))
    return float(least), jnp.asarray(scale, dtype=actions.dtype)


def _quadratic(model: Dynamics) -> bool:
    """Whether the cost is quadratic in the actions by the model's structure alone.

    A field affine in the state and the action together makes each RK4 step affine in both, so the
    rollout is affine in the actions and the quadratic cost quadratic in them: its Hessian is one
    matrix over the whole box. A drift past degree 1, a channel that reads the state, or any field
    not named here may bend the rollout, and is not taken to be quadratic.
    """
    if isinstance(model, LinearDynamics | DampedOscillator | ZeroResidual):
        return True
    if isinstance(model, ControlAffineResidual):
        return model.degree <= 1 and model.channel_degree == 0
    if isinstance(model, HybridDynamics):
        return _quadratic(model.known) and _quadratic(model.residual)
    if isinstance(model, DrivenDynamics):
        return _quadratic(model.dynamics)
    if isinstance(model, _EulerStep):
        return _quadratic(model.field)
    return False


def _stacked_rows(
    constraints: Sequence[LinearConstraint], actions: Array
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """``(G, b - G U)`` with ``G d <= b - G U`` every finite side of ``constraints`` over a move
    ``d`` from the plan ``U``: an upper side as its row, a lower one as the row negated. None where
    no side is finite.

    Raises:
        ValueError: on a row not as wide as the action sequence.
    """
    plan = np.asarray(actions, dtype=np.float64).reshape(-1)
    matrices, rights = [], []
    for constraint in constraints:
        if constraint.matrix.shape[1] != plan.size:
            raise ValueError(
                f"a constraint row has {constraint.matrix.shape[1]} columns; the plan's action "
                f"sequence has {plan.size} entries"
            )
        reached = constraint.matrix @ plan
        upper, lower = np.isfinite(constraint.upper), np.isfinite(constraint.lower)
        matrices += [constraint.matrix[upper], -constraint.matrix[lower]]
        rights += [
            constraint.upper[upper] - reached[upper],
            reached[lower] - constraint.lower[lower],
        ]
    matrix = np.concatenate([np.zeros((0, plan.size)), *matrices])
    if matrix.shape[0] == 0:
        return None
    return matrix, np.concatenate(rights)


def _row_bound(
    gradient: NDArray[np.float64],
    curvature: NDArray[np.float64],
    below: NDArray[np.float64],
    above: NDArray[np.float64],
    matrix: NDArray[np.float64],
    right: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, float]:
    """The largest certified gain ``-g d - sum_i c_i d_i^2 / 2`` over moves ``d`` in the box and
    ``G d <= right``, bounded from above by its Lagrange dual, and the linear program's, ``c = 0``.

    The dual function ``q(lam) = sum_i max_{d_i in [below_i, above_i]} (-t_i d_i - c_i d_i^2 / 2)
    + lam' right``, ``t = g + G' lam``, is at least the primal's largest at every ``lam >= 0``
    (``validation/regret_bound_rows.mac``). It is separable, each term's maximiser the clip of
    ``-t_i / c_i`` to the box, or the box's end ``-t_i`` points to where ``c_i = 0``. The linear
    program, solved by HiGHS, gives the ``c = 0`` multipliers; where ``c > 0`` they start
    L-BFGS-B's descent of ``q``, whose gradient is ``right - G d(lam)``. Each bound is ``q`` at
    the multipliers its solver ended on, so a solver stopped short loosens it and never voids it.

    Returns:
        Each action's gain at the dual's maximiser ``d(lam)``, the dual bound, and the linear
        program's.

    Raises:
        ValueError: where no move in the box meets the rows.
    """
    flat_g, flat_c = gradient.reshape(-1), np.broadcast_to(curvature, gradient.shape).reshape(-1)
    low, high = below.reshape(-1), above.reshape(-1)

    def move(lam: NDArray[np.float64], c: NDArray[np.float64]) -> NDArray[np.float64]:
        tilt = flat_g + matrix.T @ lam
        bent = c > 0.0
        flat = np.where(tilt > 0.0, low, np.where(tilt < 0.0, high, 0.0))
        return np.where(bent, np.clip(-tilt / np.where(bent, c, 1.0), low, high), flat)

    def dual(lam: NDArray[np.float64], c: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        d = move(lam, c)
        value = float(np.sum(-(flat_g + matrix.T @ lam) * d - 0.5 * c * d**2) + lam @ right)
        return value, right - matrix @ d

    program = linprog(
        flat_g, A_ub=matrix, b_ub=right, bounds=list(zip(low, high, strict=True)), method="highs"
    )
    if program.status == 2:
        raise ValueError("no plan in the box meets the constraints: there is no optimum to compare")
    start = (
        np.maximum(-np.asarray(program.ineqlin.marginals, dtype=np.float64), 0.0)
        if program.status == 0
        else np.zeros(matrix.shape[0])
    )
    linear = dual(start, np.zeros_like(flat_c))[0]
    lam = start
    if np.any(flat_c > 0.0):
        descent = minimize(
            lambda value: dual(value, flat_c),
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, None)] * matrix.shape[0],
            options={"maxiter": 10_000, "ftol": 1e-15, "gtol": 1e-12},
        )
        lam = np.maximum(np.asarray(descent.x, dtype=np.float64), 0.0)
    total = dual(lam, flat_c)[0]
    d = move(lam, flat_c)
    gains = (-flat_g * d - 0.5 * flat_c * d**2).reshape(gradient.shape)
    return gains, total, linear


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
    constraints: Sequence[LinearConstraint] = (),
) -> PlanRegretBound:
    """RESULT 69 (L3.2) -- the optimality gap of a finished plan, certified from its own gradient.

    Derived in ``validation/constrained_plan_regret.mac``, proved in
    ``proofs/constrained_plan_regret.v``. Reads the plan's actions, not the solver's internals, so
    it prices a plan that came from anywhere -- including one an operator edited by hand.

    The feasible set is the box, and the linear rows passed as ``constraints``. Without them the
    bound is on the gap to the box's optimum, which is never below the gap to the optimum under
    the rows the plan was solved under, so it stays valid and is loose by what the rows cost: a
    plan at the exact optimum under a budget row that binds read 0.0885 where its own gap is
    zero. Passed the rows, it reads 4e-13 there.

    Args:
        modulus: the strong-convexity modulus of ``J`` over the box, which makes the bound a
            certificate. ``None`` reads it off ``J``'s Hessian (see :func:`_objective_modulus`):
            once at the plan where the model is linear, the Hessian then being one matrix, and
            otherwise at the plan and ``probes`` random points, a sample that cannot certify. A
            *smaller* modulus gives a *larger* bound and is never invalid (STEP 4d), so a
            conservative one is the safe input; for a linear plant with positive semidefinite ``Q``
            and ``Qf``, ``lambda_min(R)`` is always valid and needs no eigenvalue solve on ``J``.
            A supplied modulus is the plain norm's, in the levers' own units, and the bound then
            reads those units; read off the Hessian, each lever's moves are in its own scale.
        probes: random feasible points added to the curvature sample when ``modulus`` is sampled.
        constraints: linear rows over the whole action sequence
            (:class:`~chc.control.LinearConstraint`) that bound the plans compared with this one,
            beside the box. The certified gain over the box and the rows has no closed form; it is
            read off its Lagrange dual, ``q(lam) = sum_i max_{d_i in box} (-(g + G'lam)_i d_i -
            c_i d_i^2 / 2) + lam'(b - G U)``, at the multipliers a dual descent ends on, and any
            multipliers bound it from above (weak duality, ``validation/regret_bound_rows.mac``),
            so the descent's stopping point loosens the bound and never voids it. The modulus and
            its sample still read the box alone. :attr:`~PlanRegretBound.frank_wolfe_gap` is the
            linear program over the same set, read off its dual too, and
            :attr:`~PlanRegretBound.per_lever` the gain of each lever's moves at the dual's
            maximiser, which sums to ``bound`` less the multipliers' slack in the rows.

    The three numbers to read together: :attr:`~PlanRegretBound.bound` is what the box certifies,
    :attr:`~PlanRegretBound.unconstrained_bound` is what Result 6 would have reported at the same
    plan, and :attr:`~PlanRegretBound.frank_wolfe_gap` is what survives if the modulus goes to zero.
    :attr:`~PlanRegretBound.status` says what the bound is: ``certified`` on a supplied modulus or
    a linear model's, ``diagnostic`` on a sampled one, and ``refused`` -- the bound ``inf`` -- when
    the modulus is negative, because then no convexity argument applies and a finite number would be
    a fabrication, and when the modulus or the bound is not a finite number.

    Raises:
        ValueError: if a supplied ``modulus`` is negative, infinite or nan, which is not a
            curvature; or if no plan in the box meets ``constraints``, or a row is not as wide as
            the action sequence.
    """
    if modulus is not None and not 0.0 <= modulus < math.inf:
        raise ValueError(
            f"modulus is a curvature and cannot be negative, infinite or nan: {modulus}"
        )

    actions = plan.actions
    lo = broadcast_box(u_lo, actions.shape, "u_lo", actions.dtype)
    hi = broadcast_box(u_hi, actions.shape, "u_hi", actions.dtype)

    def objective(us: Array) -> Array:
        return total_cost(model, x0, us, dt, cost)

    gradient = jax.grad(objective)(actions)
    source: ModulusSource
    sampled = modulus is None and not _quadratic(model)
    if modulus is not None:
        mu, source = float(modulus), "supplied"
        scale = jnp.ones(actions.shape[1:], dtype=actions.dtype)
    else:
        mu, scale = _objective_modulus(objective, actions, lo, hi, probes if sampled else 0, seed)
        source = "measured"

    below, above = lo - actions, hi - actions  # the feasible moves, as offsets from the plan
    rows = _stacked_rows(constraints, actions)
    if rows is not None:
        certified, total, fallback = _row_bound(
            np.asarray(gradient, dtype=np.float64),
            np.asarray(mu / scale**2 if mu > 0.0 else jnp.zeros_like(scale), dtype=np.float64),
            np.asarray(below, dtype=np.float64),
            np.asarray(above, dtype=np.float64),
            *rows,
        )
    else:
        # max{-g d : d in [below, above]}: a line on an interval is largest at an endpoint, and
        # both candidates are >= 0 because 0 is feasible.
        linear = jnp.maximum(-gradient * below, -gradient * above)
        if mu > 0.0:
            curvature = mu / scale**2  # the modulus read back in each lever's units (STEP 8b)
            step = jnp.clip(-gradient / curvature, below, above)
            certified = -gradient * step - 0.5 * curvature * step**2
        else:
            certified = linear
        total = float(jnp.sum(certified))

    per_lever = tuple(float(v) for v in jnp.sum(certified, axis=0))
    # Refused unless every number the bound rests on is finite: a nan modulus read neither below 0
    # nor certifying nothing, and certified an infinite bound; a nan gradient certified a nan one.
    refused = not (0.0 <= mu < math.inf and math.isfinite(total))
    status: RegretStatus = "refused" if refused else ("diagnostic" if sampled else "certified")
    return PlanRegretBound(
        bound=math.inf if refused else total,
        unconstrained_bound=(
            float(jnp.sum((scale * gradient) ** 2)) / (2.0 * mu) if mu > 0.0 else math.inf
        ),
        frank_wolfe_gap=float(jnp.sum(linear)) if rows is None else fallback,
        modulus=mu,
        modulus_source=source,
        gradient_norm=float(jnp.linalg.norm(scale * gradient)),
        per_lever=tuple(math.inf for _ in per_lever) if refused else per_lever,
        pinned_actions=int(jnp.sum((certified <= 0.0) & (gradient != 0.0))),
        status=status,
    )
