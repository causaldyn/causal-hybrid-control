"""The one-call spine: a plan that carries its own certificate (plans/21 §D)."""

from itertools import pairwise

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from chc.barrier import robust_barrier_margin
from chc.control import Bound, projected_gradient_control, projected_gradient_solve
from chc.cost import QuadraticCost, total_cost
from chc.dynamics import DampedOscillator, HybridDynamics, LinearDynamics
from chc.plan import causal_plan, certify_safety, plan_regret_bound
from chc.residual import MLPResidual, ZeroResidual
from chc.support import SupportModel
from chc.uncertainty import ConfoundingRobustPenalty, confounding_robust_inflation

_A = jnp.array([[-0.5, 1.0], [0.0, -0.3]])
_B = jnp.array([[0.0], [1.0]])
_MODEL = HybridDynamics(known=LinearDynamics(_A, _B), residual=ZeroResidual(2))
_COST = QuadraticCost(
    Q=jnp.diag(jnp.array([1.0, 0.1])),
    R=jnp.array([[0.05]]),
    Qf=jnp.diag(jnp.array([5.0, 1.0])),
    x_target=jnp.zeros(2),
)
_X0 = jnp.array([1.0, 0.0])
_ARGS = (_MODEL, _X0, _COST, 0.1, 12, -5.0, 5.0)


def test_bare_plan_matches_projected_gradient_control() -> None:
    """With no safety arguments the spine must be the existing solver, not a new one."""
    plan = causal_plan(*_ARGS)
    # Both at their own default budget: pinning one of them here would test the budgets agreeing
    # rather than the solvers being the same solver, which is what the claim is.
    reference, _ = projected_gradient_control(
        _MODEL, _X0, jnp.zeros((12, 1)), 0.1, _COST, -5.0, 5.0
    )
    assert float(jnp.max(jnp.abs(plan.actions - reference))) < 1e-6
    assert plan.trajectory.shape == (13, 2)
    assert plan.actions.shape == (12, 1)


def test_tube_and_certified_horizon_cut_the_plan_where_the_error_leaves_tolerance() -> None:
    plan = causal_plan(*_ARGS, lipschitz=0.8, model_error=0.05, tolerance=0.03)
    assert plan.uncertainty_tube is not None
    assert plan.certified_horizon is not None
    assert plan.uncertainty_tube.shape == (13,)
    assert float(plan.uncertainty_tube[0]) == 0.0
    assert bool(jnp.all(jnp.diff(plan.uncertainty_tube) >= 0.0))  # the tube only grows
    assert 0 < plan.certified_horizon < 12  # a real cut, not all-or-nothing
    assert plan.certificate_status == "partial"
    assert plan.certified_actions.shape == (plan.certified_horizon, 1)
    assert float(plan.uncertainty_tube[plan.certified_horizon]) <= 0.03


def test_no_error_model_reports_not_evaluated_instead_of_a_full_horizon_pass() -> None:
    """The footgun this replaces: a zero tube used to certify all 12 steps having proved nothing."""
    plan = causal_plan(*_ARGS, tolerance=1e-9)
    assert plan.certificate_status == "not_evaluated"
    assert plan.uncertainty_tube is None
    assert plan.certified_horizon is None
    with pytest.raises(ValueError, match="no error model"):
        _ = plan.certified_actions


def test_an_error_model_inside_tolerance_certifies_the_whole_plan() -> None:
    """ "certified" must stay reachable -- the fix must not make every plan look unevaluated."""
    plan = causal_plan(*_ARGS, lipschitz=0.8, model_error=1e-6, tolerance=1.0)
    assert plan.certificate_status == "certified"
    assert plan.certified_horizon == 12
    assert plan.certified_actions.shape == (12, 1)


def test_an_error_model_that_busts_tolerance_immediately_certifies_nothing() -> None:
    plan = causal_plan(*_ARGS, lipschitz=0.8, model_error=10.0, tolerance=1e-6)
    assert plan.certificate_status == "uncertified"
    assert plan.certified_horizon == 0
    assert plan.certified_actions.shape == (0, 1)


def test_a_negative_error_budget_is_rejected_rather_than_shrinking_the_tube() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        causal_plan(*_ARGS, lipschitz=0.8, model_error=-0.05)


def test_support_penalty_pulls_the_plan_toward_the_logged_cloud() -> None:
    key = jax.random.key(0)
    logged_x = 0.05 * jax.random.normal(key, (200, 2))
    logged_u = 0.05 * jax.random.normal(jax.random.key(1), (200, 1))
    support = SupportModel.fit(logged_x, logged_u)

    free = causal_plan(*_ARGS)
    pessimistic = causal_plan(*_ARGS, support=support, lam_supp=1.0)
    assert float(jnp.linalg.norm(pessimistic.actions)) < float(jnp.linalg.norm(free.actions))
    assert pessimistic.task_cost > free.task_cost  # pessimism costs task performance, by design


def test_uncertainty_without_support_is_rejected_rather_than_silently_dropped() -> None:
    with pytest.raises(ValueError, match="requires a support model"):
        causal_plan(*_ARGS, uncertainty=ConfoundingRobustPenalty(radius=0.1), lam_unc=1.0)


_PLAN = causal_plan(*_ARGS)


def _mixed(x: Array) -> Array:
    """Relative-degree-1 CBF for this plant: the control reaches it in one differentiation."""
    return 0.5 * x[0] + x[1] + 0.4


def _position(x: Array) -> Array:
    """Relative degree 2 -- ``B^T grad h == 0``, so no action moves this barrier directly."""
    return x[0]


def test_exact_identification_reduces_to_the_ordinary_barrier_check() -> None:
    """At ``gamma = 1`` the radius vanishes and the guarantee must be plain ``grad h . f(x, u)``."""
    cert = certify_safety(_PLAN, _MODEL, _mixed, 0.1, alpha=5.0, gamma=1.0, u_max=5.0)
    grad = jnp.array([0.5, 1.0])  # constant for an affine barrier
    nominal = jax.vmap(lambda x, u: grad @ (_A @ x + _B @ u))(_PLAN.trajectory[:-1], _PLAN.actions)

    assert cert.radius == 0.0
    assert float(jnp.max(jnp.abs(cert.guaranteed_derivative - nominal))) < 1e-5
    assert float(jnp.max(jnp.abs(cert.required + 5.0 * cert.barrier_values))) < 1e-6


def test_a_wider_sensitivity_only_ever_shrinks_the_guarantee() -> None:
    certs = [
        certify_safety(_PLAN, _MODEL, _mixed, 0.1, alpha=5.0, gamma=g, u_max=5.0)
        for g in (1.0, 1.5, 2.0, 3.0, 5.0)
    ]
    radii = [c.radius for c in certs]
    steps = [c.certified_steps for c in certs]

    assert radii == sorted(radii)
    assert radii[0] < radii[-1]
    assert steps == sorted(steps, reverse=True)
    # A real cut at both ends, not all-or-nothing. The exact count is a property of the plan, not
    # of the guarantee: a better-converged plan drives harder, leaves tolerance sooner and certifies
    # fewer steps, so pinning the number would make this a test of the solver's budget.
    assert 0 < steps[0] < 12
    assert steps[-1] == 0
    for lo, hi in pairwise(certs):
        assert bool(jnp.all(hi.guaranteed_derivative <= lo.guaranteed_derivative + 1e-9))


def test_gamma_star_prices_the_problem_and_does_not_move_with_the_assumed_level() -> None:
    """``gamma_star`` asks whether *any* action holds, so the assumed ``gamma`` is not an input."""
    ceilings = {
        certify_safety(_PLAN, _MODEL, _mixed, 0.1, alpha=5.0, gamma=g, u_max=5.0).gamma_star
        for g in (1.0, 2.0, 4.0)
    }
    assert len(ceilings) == 1

    thin = certify_safety(_PLAN, _MODEL, _mixed, 0.1, alpha=5.0).gamma_star  # plan's own authority
    assert 1.0 < thin < ceilings.pop()  # less authority buys less tolerance


def test_gamma_star_is_sharp_at_the_weakest_step() -> None:
    """At exactly ``gamma_star`` the best admissible action meets the barrier; past it it cannot."""
    cert = certify_safety(_PLAN, _MODEL, _mixed, 0.1, alpha=5.0, gamma=1.0, u_max=5.0)
    k = int(jnp.argmin(jnp.asarray(cert.step_gamma_star)))
    x = _PLAN.trajectory[k]
    grad = np.array([0.5, 1.0])
    drift = float(grad @ np.asarray(_A @ x))
    channel = float(abs(grad @ np.asarray(_B).ravel()))
    required = -5.0 * float(_mixed(x))

    def best_margin(gamma: float) -> float:
        radius = confounding_robust_inflation(1.0, 0.0, gamma) * float(np.linalg.norm(grad))
        return robust_barrier_margin(drift, channel, radius, 5.0)

    assert best_margin(cert.gamma_star) == pytest.approx(required, abs=1e-6)
    assert best_margin(cert.gamma_star * 0.999) > required
    assert best_margin(cert.gamma_star * 1.001) < required


def test_a_barrier_the_control_cannot_reach_is_uncertifiable_rather_than_infinitely_robust() -> (
    None
):
    """The plan-level form of the round-eleven bug: a ``nan`` step must sink the whole plan.

    ``B^T grad h == 0`` with a positive deficit is nominally infeasible -- no radius helps, because
    no action helps. Skipping those steps (a ``nanmin``) would report the *other* steps' comfortable
    ceiling as the plan's, which is the inverse of the truth.
    """
    cert = certify_safety(_PLAN, _MODEL, _position, 0.1, alpha=0.2, gamma=1.0, u_max=5.0)
    assert all(np.isnan(g) for g in cert.step_gamma_star)
    assert np.isnan(cert.gamma_star)


def test_a_plan_can_fail_a_guarantee_the_problem_itself_tolerates() -> None:
    """The two questions must not be collapsed: ``gamma_star`` indicts the problem, not the plan."""

    def slack(x: Array) -> Array:
        return 0.5 * x[0] + x[1] + 1.0

    cert = certify_safety(_PLAN, _MODEL, slack, 0.1, alpha=2.0, gamma=1.5, u_max=5.0)
    assert cert.gamma_star == float("inf")  # standing still is safe at every sensitivity level
    assert cert.certified_steps == 0  # yet the planned action is not, from the first step on
    # and the per-step flags recover later, which is why ``certified_steps`` is a prefix length --
    # like ``CausalPlan.certified_horizon`` -- rather than a count of the steps that happen to pass.
    assert bool(jnp.any(cert.planned_certified))


def test_a_non_positive_cvar_gap_is_rejected_rather_than_inverted() -> None:
    with pytest.raises(ValueError, match="cvar_gap must be positive"):
        certify_safety(_PLAN, _MODEL, _mixed, 0.1, cvar_gap=0.0)


def test_a_plan_that_never_acts_names_itself_rather_than_the_inner_threshold() -> None:
    """The default budget is the plan's own action, so an idle plan must say *that*, not "u_max"."""
    idle = causal_plan(_MODEL, jnp.zeros(2), _COST, 0.1, 12, -5.0, 5.0)
    with pytest.raises(ValueError, match="it never acts"):
        certify_safety(idle, _MODEL, _mixed, 0.1)


def test_a_plan_says_whether_its_own_solve_finished() -> None:
    # The certificate is about the model; the solver status is about the optimisation. A fully
    # certified plan built on a truncated solve is a trustworthy tube around a suboptimal action,
    # and before this the plan had no way to say so.
    model = HybridDynamics(
        known=DampedOscillator(omega=1.0, zeta=0.1), residual=ZeroResidual(out_dim=2)
    )
    cost = QuadraticCost(
        Q=jnp.eye(2),
        R=jnp.array([[0.01]]),
        Qf=10.0 * jnp.eye(2),
        x_target=jnp.zeros(2),
    )
    x0 = jnp.array([1.0, 0.0])

    truncated = causal_plan(model, x0, cost, 0.1, 15, -5.0, 5.0, steps=4)
    assert truncated.solver_status == "max_iterations"
    assert truncated.solver_iterations == 4

    finished = causal_plan(model, x0, cost, 0.1, 15, -5.0, 5.0, steps=50_000)
    assert finished.solver_status == "converged"
    assert finished.task_cost < truncated.task_cost

    # Independent axes: both plans carry the same certificate verdict, and only one is optimised.
    assert truncated.certificate_status == finished.certificate_status == "not_evaluated"


def test_a_boxed_plan_certifies_its_own_optimality_gap_where_the_pl_bound_charges_regret() -> None:
    # Result 69 (L3.2), validation/constrained_plan_regret.mac, proofs/constrained_plan_regret.v.
    # Result 6's |grad J|^2/(2 mu) needs no optimum, which is what makes it a certificate -- and it
    # does not survive a box: at a lever the gradient holds against its own bound grad J is nonzero
    # while the true regret is zero.
    dt, horizon = 0.1, 12
    plant = LinearDynamics(jnp.array([[0.0, 1.0], [-2.0, -0.3]]), jnp.array([[0.0], [1.0]]))
    cost = QuadraticCost(
        Q=jnp.eye(2), R=0.05 * jnp.eye(1), Qf=3.0 * jnp.eye(2), x_target=jnp.zeros(2)
    )
    x0 = jnp.array([1.5, 0.0])

    def reached(model: object, lo: Bound, hi: Bound, m: int, seed: int) -> float:
        """Best cost the box allows, from several starts -- the truth the bound is checked on."""
        best = jnp.inf
        for key in jax.random.split(jax.random.PRNGKey(seed), 5):
            start = jax.random.uniform(key, (horizon, m), minval=lo, maxval=hi)
            solved = projected_gradient_solve(
                model, x0, start, dt, cost, lo, hi, steps=40_000, tol=1e-14
            )
            best = jnp.minimum(best, total_cost(model, x0, solved.actions, dt, cost))
        return float(best)

    # 1. THE HEADLINE. A box tight enough to clip the optimum away leaves every action pinned,
    #    so the plan IS optimal -- and the bound says exactly zero where Result 6's says 56.
    pinned = causal_plan(plant, x0, cost, dt, horizon, -0.2, 0.2, steps=20_000)
    tight = plan_regret_bound(pinned, plant, x0, cost, dt, -0.2, 0.2, probes=8)
    assert tight.pinned_actions == pinned.actions.size  # the gradient holds all of them
    assert tight.bound == 0.0  # exactly, not approximately: the maximiser is d = 0
    assert tight.unconstrained_bound > 50.0  # ... and Result 6 charges regret to the optimum
    assert pinned.task_cost - reached(plant, -0.2, 0.2, 1, 0) <= 1e-9

    # 2. THE GATE THAT CAN FAIL: on genuinely unconverged plans the bound must sit ABOVE the
    #    realised gap, and it is worth having only if it does so without being vacuous.
    for lo, hi, steps, ceiling in ((-2.0, 2.0, 3, 2.0), (-2.0, 2.0, 200, 1.5), (-0.4, 0.4, 3, 1.2)):
        plan = causal_plan(plant, x0, cost, dt, horizon, lo, hi, steps=steps)
        curve = plan_regret_bound(plan, plant, x0, cost, dt, lo, hi, probes=8)
        realised = plan.task_cost - reached(plant, lo, hi, 1, 0)
        assert curve.bound >= realised - 1e-12  # valid
        assert curve.bound <= ceiling * realised  # and tight enough to act on
        assert curve.bound <= curve.unconstrained_bound + 1e-12  # the box never loosens it
        assert curve.bound <= curve.frank_wolfe_gap + 1e-12  # ... and the mu-free fallback caps it
        assert sum(curve.per_lever) == pytest.approx(curve.bound, rel=1e-9, abs=1e-12)

    # 3. A SECOND PLANT, nonlinear: the same certificate, with the modulus no longer a constant.
    hybrid = HybridDynamics(
        DampedOscillator(1.0, 0.2), MLPResidual(2, 1, 2, key=jax.random.PRNGKey(3))
    )
    plan = causal_plan(hybrid, x0, cost, dt, horizon, -0.5, 0.5, steps=20_000)
    curve = plan_regret_bound(hybrid_plan := plan, hybrid, x0, cost, dt, -0.5, 0.5, probes=24)
    assert curve.ok
    assert curve.modulus > 0.0
    assert curve.bound >= hybrid_plan.task_cost - reached(hybrid, -0.5, 0.5, 1, 2) - 1e-12

    # 4. WHERE THE MODULUS COMES FROM. For a plant affine in the action J is exactly quadratic, so
    #    the measured curvature is the global one -- and lambda_min(R) is a valid floor under it
    #    (STEP 7b), which STEP 4d says can only make the bound larger.
    slack = causal_plan(plant, x0, cost, dt, horizon, -2.0, 2.0, steps=20_000)
    measured = plan_regret_bound(slack, plant, x0, cost, dt, -2.0, 2.0, probes=16)
    hessian = jax.hessian(lambda v: total_cost(plant, x0, v.reshape(horizon, 1), dt, cost))(
        slack.actions.reshape(-1)
    )
    assert measured.modulus == pytest.approx(
        float(jnp.min(jnp.linalg.eigvalsh(hessian))), rel=1e-10
    )
    assert measured.modulus >= float(jnp.min(jnp.linalg.eigvalsh(cost.R)))
    conservative = plan_regret_bound(slack, plant, x0, cost, dt, -2.0, 2.0, modulus=0.05)
    assert conservative.modulus_source == "supplied"
    assert conservative.bound >= measured.bound  # a smaller modulus is looser, never invalid

    # 5. AND IT REFUSES TO CERTIFY WHAT IT CANNOT. A residual large enough to make J non-convex
    #    over the box gets inf, because there no convexity argument applies and a finite number
    #    would be a fabrication.
    loud = jax.tree_util.tree_map(
        lambda a: 4.0 * a if eqx.is_array(a) else a,
        MLPResidual(2, 1, 2, key=jax.random.PRNGKey(3)),
    )
    wild = HybridDynamics(DampedOscillator(1.0, 0.2), loud)
    rough = causal_plan(wild, x0, cost, dt, horizon, -6.0, 6.0, steps=4_000)
    verdict = plan_regret_bound(rough, wild, x0, cost, dt, -6.0, 6.0, probes=48)
    assert not verdict.ok
    assert verdict.modulus < 0.0
    assert np.isinf(verdict.bound)

    with pytest.raises(ValueError, match="cannot be negative"):
        plan_regret_bound(slack, plant, x0, cost, dt, -2.0, 2.0, modulus=-1.0)
