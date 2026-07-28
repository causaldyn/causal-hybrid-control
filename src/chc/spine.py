"""The four layers on one story: confounded logs -> causal effect -> plan -> safety certificate.

Every layer of this library already had its own demo; none of them ran end to end. This is the
whole spine on a single decision, so the pieces are forced to compose:

1. **fit** -- an incentive response estimated from logs whose behaviour policy chased a confounder
   (:mod:`chc.causal`), once naively and once with the backdoor adjustment;
2. **plan** -- constrained OC with a certified Gronwall error tube (:func:`chc.plan.causal_plan`);
3. **certify** -- the plan priced against a partially identified effect
   (:func:`chc.plan.certify_safety`);
4. **audit** -- each plan then executed on the *true* plant, so the numbers a caller would have
   trusted offline can be compared with what actually happened.

The plant is two zones of a mobile driver pool. Incentivising zone A pulls drivers *out of* zone B
(the ``[+b, -b]`` control column is driver conservation, i.e. the interference channel), zone B
drains on its own, and the barrier is a supply floor there. So the objective and the constraint pull
against each other through the only lever available, which is what makes the certificate
load-bearing rather than decorative.

Deliberately control-affine: :func:`chc.plan.certify_safety` reads the channel ``B^T grad h`` off
the Jacobian at ``u = 0``, exact for an affine plant and only a linearisation otherwise -- the
softmax-equilibrium market in :mod:`chc.marketplace` is *not* affine, so certifying a plan on it
would quietly evaluate the channel at the wrong action.

    uv run python scripts/spine_demo.py
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from chc.causal import ConfoundedLinearSystem, estimate_control_effect
from chc.cost import QuadraticCost, total_cost
from chc.dynamics import HybridDynamics, LinearDynamics
from chc.estimators import BackdoorOLS, CausalEffectEstimator
from chc.integrate import rollout
from chc.plan import CausalPlan, SafetyCertificate, causal_plan, certify_safety
from chc.residual import ZeroResidual


def two_zone_market(
    effect: float, *, decay: float = 0.6, coupling: float = 0.3, drain: float = 0.25
) -> HybridDynamics:
    """Driver supply in two zones under one incentive lever, ``xdot = A x + B u``.

    ``A`` is the known mechanism: zone A relaxes back to its free-flow level at rate ``decay``,
    drivers exchange between the zones at ``coupling``, and zone B loses drivers at ``drain`` on its
    own. ``B = effect * [+1, -1]`` is the learned part -- the incentive gain, whose sign and size
    are exactly what a confounded log gets wrong.
    """
    a_matrix = jnp.array([[-decay, coupling], [coupling, drain]])
    b_matrix = effect * jnp.array([[1.0], [-1.0]])
    return HybridDynamics(known=LinearDynamics(a_matrix, b_matrix), residual=ZeroResidual(2))


@dataclass(frozen=True)
class SpineArm:
    """One controller: what it believed, what it planned, and what that plan actually cost."""

    name: str
    effect: float  # the incentive gain this arm estimated and planned with
    plan: CausalPlan
    certificate: SafetyCertificate
    true_trajectory: Array  # the plan executed on the TRUE plant, full horizon
    true_cost: float  # its cost there, not under its own model
    true_barrier_min: float  # smallest h on the true plant over the certified prefix only


@dataclass(frozen=True)
class SpineReport:
    """The two arms plus the ground truth neither of them was allowed to see."""

    effect_true: float
    supply_floor: float
    arms: tuple[SpineArm, ...]

    def arm(self, name: str) -> SpineArm:
        return next(a for a in self.arms if a.name == name)


def run_spine(
    *,
    n_data: int = 20_000,
    horizon: int = 25,
    dt: float = 0.1,
    target: float = 1.0,
    supply_floor: float = -0.4,
    u_max: float = 3.0,
    gamma: float = 1.6,
    cvar_gap: float = 1.0,
    seed: int = 0,
    estimator: CausalEffectEstimator | None = None,
) -> SpineReport:
    """Run both arms through all four layers and report them side by side.

    Args:
        gamma, cvar_gap: the assumed sensitivity level and the gap it scales, passed straight to
            :func:`chc.plan.certify_safety`. They price the *plan*; they do not change it.
        supply_floor: the barrier is ``h(x) = x_B - supply_floor``, safe where ``h >= 0``.
        u_max: the actuation budget backing ``gamma_star`` -- the box bound, not the plan's own
            largest action, since ``gamma_star`` asks what the *best admissible* action could hold.
    """
    system = ConfoundedLinearSystem()
    data = system.sample(n_data, jax.random.key(seed))
    effects = {
        "naive": float(estimate_control_effect(data, adjust_for=())),
        "causal": float((estimator or BackdoorOLS()).estimate(data, covariates=("x", "z")).effect),
    }

    truth = two_zone_market(system.b_true)
    x0 = jnp.array([0.0, -0.1])
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.6])),
        R=jnp.array([[0.05]]),
        Qf=jnp.diag(jnp.array([2.0, 0.6])),
        x_target=jnp.array([target, 0.0]),
    )

    def barrier(x: Array) -> Array:
        return x[1] - supply_floor

    arms = []
    for name, effect in effects.items():
        believed = two_zone_market(effect)
        plan = causal_plan(
            believed,
            x0,
            cost,
            dt,
            horizon,
            -u_max,
            u_max,
            lipschitz=0.9,
            model_error=0.02,
            tolerance=0.5,
        )
        # certified against the model the operator actually has; gamma covers the gap to truth
        certificate = certify_safety(
            plan, believed, barrier, dt, gamma=gamma, cvar_gap=cvar_gap, u_max=u_max
        )
        executed = rollout(truth, x0, plan.actions, dt)
        certified = executed[: certificate.certified_steps + 1]
        arms.append(
            SpineArm(
                name=name,
                effect=effect,
                plan=plan,
                certificate=certificate,
                true_trajectory=executed,
                true_cost=float(total_cost(truth, x0, plan.actions, dt, cost)),
                true_barrier_min=float(jnp.min(jax.vmap(barrier)(certified))),
            )
        )
    return SpineReport(effect_true=system.b_true, supply_floor=supply_floor, arms=tuple(arms))
