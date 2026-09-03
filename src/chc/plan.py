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

* **plan** -- :func:`causal_plan`. Box constraints in the solve, plus an *a-priori* Gronwall error
  tube that says how far ahead the plan may be trusted. The tube is computed from ``lipschitz`` and
  ``model_error``; it does not enter the objective and does not move a single action.
* **audit** -- :func:`certify_safety`. Given a barrier and a sensitivity level ``Gamma``, it prices
  a *finished* plan against §40: where along it the safety guarantee survives unmeasured
  confounding, and the largest ``Gamma`` the whole plan tolerates. Read-only by construction.
* **filter** -- :func:`chc.barrier.robust_safety_filter`. The only mode that changes an action: it
  clips one nominal action into the certified interval at one state, online, scalar control.

What does **not** exist here is a state-constrained solve: no barrier, tube or ``h(x) >= 0``
requirement is imposed *inside* the optimisation, so :func:`causal_plan` can return a plan whose
audit fails and whose tube leaves tolerance at step 3. That is why the audit is a separate call and
why :attr:`CausalPlan.certified_actions` exists -- the enforcement happens after the fact, by
truncation or by the filter, not by the planner. A CBF-QP or barrier-penalised solve is future work.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from chc.barrier import barrier_gamma_star, identification_radius_threshold
from chc.control import Bound, projected_gradient_control
from chc.cost import QuadraticCost, total_cost
from chc.dynamics import Dynamics
from chc.integrate import rollout
from chc.support import PenaltyModel, SupportModel, pessimistic_control
from chc.uncertainty import (
    certified_horizon,
    confounding_robust_inflation,
    time_varying_rollout_bound,
)

CertificateStatus = Literal["not_evaluated", "uncertified", "partial", "certified"]


@dataclass(frozen=True)
class CausalPlan:
    """A control sequence together with the certificate that says how far to trust it.

    ``None`` in the certificate fields means *no error model was supplied*, which is a different
    statement from a certificate that came back empty: the first is "unknown", the second is
    "checked, and nothing holds". Earlier versions returned an all-zero tube and a full
    ``certified_horizon`` in the first case, which reads as a proof of safety over the whole plan
    when nothing was proved at all. Read :attr:`certificate_status` before either field.
    """

    actions: Array  # (horizon, m)
    trajectory: Array  # (horizon + 1, n) nominal rollout under the planning model
    task_cost: float  # cost of the plan, penalties excluded, so weights stay comparable
    uncertainty_tube: Array | None  # (horizon + 1,) per-step error radius; None if not evaluated
    certified_horizon: int | None  # last step inside ``tolerance``; None if not evaluated

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

    Raises:
        ValueError: if an uncertainty penalty is given without a support model, which would
            silently drop it -- the pessimistic solver is the only consumer of that argument; or if
            ``model_error`` is negative, which is not an error budget.
    """
    if uncertainty is not None and support is None:
        raise ValueError("uncertainty penalty requires a support model; it is unused without one")
    if model_error < 0.0:
        raise ValueError(
            f"model_error is a per-step error budget and cannot be negative: {model_error}"
        )

    guess = jnp.zeros((horizon, cost.R.shape[0]))
    if support is None:
        actions, _ = projected_gradient_control(model, x0, guess, dt, cost, u_lo, u_hi, steps=steps)
    else:
        actions, _ = pessimistic_control(
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
        )

    lipschitz_seq, error_seq = [lipschitz] * horizon, [model_error] * horizon
    evaluated = model_error > 0.0
    return CausalPlan(
        actions=actions,
        trajectory=rollout(model, x0, actions, dt),
        task_cost=float(total_cost(model, x0, actions, dt, cost)),
        uncertainty_tube=(
            time_varying_rollout_bound(lipschitz_seq, error_seq, dt) if evaluated else None
        ),
        certified_horizon=(
            certified_horizon(lipschitz_seq, error_seq, dt, tolerance) if evaluated else None
        ),
    )


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
    grad_norm = jnp.linalg.norm(grad_h, axis=1)

    delta = confounding_robust_inflation(cvar_gap, 0.0, gamma)
    guaranteed = (
        drift
        + jnp.einsum("km,km->k", channel, actions)
        - delta * grad_norm * jnp.linalg.norm(actions, axis=1)
    )
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
