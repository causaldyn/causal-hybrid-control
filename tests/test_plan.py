"""The one-call spine: a plan that carries its own certificate."""

import math
from collections.abc import Callable
from dataclasses import replace
from itertools import pairwise, product

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from chc.barrier import robust_barrier_margin
from chc.control import Bound, projected_gradient_control, projected_gradient_solve
from chc.cost import QuadraticCost, total_cost
from chc.dynamics import DampedOscillator, DrivenDynamics, Dynamics, HybridDynamics, LinearDynamics
from chc.integrate import rk4_step, rollout
from chc.plan import (
    BarrierConstraint,
    CausalPlan,
    _quadratic,
    causal_plan,
    certify_safety,
    plan_regret_bound,
)
from chc.residual import ControlAffineResidual, MLPResidual, ZeroResidual
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


def test_the_plans_tube_is_its_rk4_rollouts_and_certifies_no_step_their_gap_has_passed() -> None:
    """``x' = x`` planned at ``dt = 1`` against the field off by 0.1: the plan's RK4 rollout and
    the off field's part by 0.1708 after one step, past a tolerance of 0.12 that Euler's recursion,
    reading 0.1, certified the step under."""
    model = LinearDynamics(jnp.array([[1.0]]), jnp.zeros((1, 1)))
    cost = QuadraticCost(Q=jnp.zeros((1, 1)), R=jnp.eye(1), Qf=jnp.eye(1), x_target=jnp.zeros(1))
    plan = causal_plan(
        model, jnp.zeros(1), cost, 1.0, 3, -1.0, 1.0, lipschitz=1.0, model_error=0.1, tolerance=0.12
    )
    off = DrivenDynamics(model, jnp.array([[0.1]]), jnp.ones((4, 1)), 1.0)
    gaps = np.abs(np.asarray(rollout(off, jnp.zeros(1), plan.actions, 1.0) - plan.trajectory))[:, 0]
    assert plan.uncertainty_tube is not None
    np.testing.assert_allclose(np.asarray(plan.uncertainty_tube), gaps, rtol=1e-14, atol=0.0)
    assert plan.certified_horizon == 0
    assert plan.certificate_status == "uncertified"


def test_a_contracting_state_matrix_shrinks_the_tube_its_norm_would_widen() -> None:
    """The plan's own matrix as ``lipschitz``: RK4's propagators carry the tube, below the tube of
    the matrix's norm at every step, so at least as many steps are certified."""
    by_matrix = causal_plan(*_ARGS, lipschitz=_A, model_error=0.05, tolerance=0.03)
    by_norm = causal_plan(
        *_ARGS, lipschitz=float(jnp.linalg.norm(_A, 2)), model_error=0.05, tolerance=0.03
    )
    assert by_matrix.uncertainty_tube is not None
    assert by_norm.uncertainty_tube is not None
    assert bool(jnp.all(by_matrix.uncertainty_tube[1:] < by_norm.uncertainty_tube[1:]))
    assert by_matrix.certified_horizon is not None
    assert by_norm.certified_horizon is not None
    assert by_matrix.certified_horizon >= by_norm.certified_horizon


@pytest.mark.parametrize(
    ("lipschitz", "match"),
    [
        (-0.3, r"lipschitz=-0.3 bounds the norm of the field's slope, which is never negative"),
        (float("nan"), r"lipschitz=nan bounds the norm"),
        (jnp.eye(3), r"state matrix of shape \(2, 2\), not an array of shape \(3, 3\)"),
        (jnp.ones(2), r"not an array of shape \(2,\)"),
    ],
)
def test_a_rate_that_is_neither_a_norm_nor_the_state_matrix_is_refused(lipschitz, match) -> None:
    """A negative log-norm turned the tube's radii negative and certified every step; it is refused
    whether or not a tube is asked for."""
    with pytest.raises(ValueError, match=match):
        causal_plan(*_ARGS, lipschitz=lipschitz, model_error=0.05, tolerance=0.03)
    with pytest.raises(ValueError, match=match):
        causal_plan(*_ARGS, lipschitz=lipschitz)


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


def test_a_plan_resting_where_the_barrier_is_flat_is_certified_at_every_gamma() -> None:
    """A smooth barrier has flat points -- the centre of a ball -- and a safe plan may rest on one.

    There the radius is zero at every ``Gamma`` and ``0 >= -alpha h`` holds, so the ceiling is
    ``inf``. The audit used to raise instead, because it inverted the radius at a zero gradient.
    """
    plan = causal_plan(_MODEL, jnp.zeros(2), _COST, 0.1, 12, -5.0, 5.0)
    assert float(jnp.max(jnp.abs(plan.trajectory))) == 0.0

    def ball(x: Array) -> Array:
        return 1.0 - jnp.sum(x**2)

    cert = certify_safety(plan, _MODEL, ball, 0.1, alpha=5.0, u_max=5.0)
    assert cert.gamma_star == float("inf")
    assert cert.certified_steps == 12


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


# ---- the identification radius is a share of the channel ----

_DRIFT, _GAIN, _FLOOR, _ALPHA, _BUDGET = -1.0, 0.3, 0.5, 0.5, 4.0
_COST_1 = QuadraticCost(Q=jnp.eye(1), R=jnp.eye(1), Qf=jnp.eye(1), x_target=jnp.zeros(1))


def _scalar(state_in: float, lever_in: float) -> tuple[CausalPlan, LinearDynamics, Callable]:
    """``x' = -x + 0.3 u`` from ``x = 1`` under a fixed schedule, its floor ``x >= 0.5``: the
    state in ``state_in`` times its units and the lever in ``lever_in`` times its own."""
    model = LinearDynamics(jnp.array([[_DRIFT]]), jnp.array([[_GAIN * state_in / lever_in]]))
    actions = lever_in * jnp.linspace(4.0, -1.0, 12)[:, None]
    plan = causal_plan(model, jnp.array([state_in]), _COST_1, 0.1, 12, -5.0, 5.0)
    plan = replace(
        plan, actions=actions, trajectory=rollout(model, jnp.array([state_in]), actions, 0.1)
    )

    def floor(x: Array) -> Array:
        return x[0] - _FLOOR * state_in

    return plan, model, floor


def test_gamma_star_is_the_closed_form_in_the_channel_s_units() -> None:
    """At the default share, the channel's own size, a floor ``x - lo`` on ``x' = a x + b u`` has
    the ceiling ``2 b U / D - 1`` at a step whose deficit is ``0 < D < b U``
    (``validation/gamma_star_units.mac`` (4)). With a gap of 1 in the channel's units, ``b = 0.3``
    read as a channel a third of the radius's unit, and the weakest step's ceiling was 1.25 where
    it is 2.15."""
    plan, model, floor = _scalar(1.0, 1.0)
    cert = certify_safety(plan, model, floor, 0.1, alpha=_ALPHA, gamma=2.0, u_max=_BUDGET)
    x = np.asarray(plan.trajectory[:-1, 0])
    deficit = -_ALPHA * (x - _FLOOR) - _DRIFT * x
    assert np.all((deficit > 0.0) & (deficit < _GAIN * _BUDGET))  # every step in the finite case
    np.testing.assert_allclose(
        cert.step_gamma_star, 2.0 * _GAIN * _BUDGET / deficit - 1.0, rtol=1e-12, atol=0.0
    )
    assert cert.radius == pytest.approx(
        confounding_robust_inflation(1.0, 0.0, 2.0) * _GAIN, rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize(
    ("state_in", "lever_in"), [(1e-3, 1.0), (1e3, 1.0), (1.0, 1e-3), (1.0, 1e3), (1e3, 1e-3)]
)
def test_gamma_star_and_the_certified_steps_read_the_same_in_any_units(
    state_in: float, lever_in: float
) -> None:
    """The same plan posed with the state in ``state_in`` times its units and the lever in
    ``lever_in`` times its own, the budget with it. The radius was a gap of 1 in the channel's
    units, which move with both: in thousandths of the state the channel, 3e-4, ceded every level
    above 1.0002, and in thousands, 300, it held every level and certified four steps of twelve
    where one is."""
    plan, model, floor = _scalar(1.0, 1.0)
    one = certify_safety(plan, model, floor, 0.1, alpha=_ALPHA, gamma=2.0, u_max=_BUDGET)
    plan, model, floor = _scalar(state_in, lever_in)
    other = certify_safety(
        plan, model, floor, 0.1, alpha=_ALPHA, gamma=2.0, u_max=lever_in * _BUDGET
    )
    assert 0 < one.certified_steps < 12  # a real cut
    assert other.certified_steps == one.certified_steps
    np.testing.assert_array_equal(other.planned_certified, one.planned_certified)
    np.testing.assert_allclose(other.step_gamma_star, one.step_gamma_star, rtol=1e-9, atol=0.0)
    assert other.radius == pytest.approx(one.radius * state_in / lever_in, rel=1e-9, abs=0.0)


def test_the_radius_is_a_share_of_the_channel_s_operator_norm() -> None:
    """Two levers whose columns are not orthogonal: the operator norm, ``1 + sqrt 2``, not the
    Frobenius norm, ``sqrt 6``, is the channel's size."""
    model = LinearDynamics(-jnp.eye(2), jnp.array([[1.0, 2.0], [0.0, 1.0]]))
    cost = QuadraticCost(Q=_COST.Q, R=0.05 * jnp.eye(2), Qf=_COST.Qf, x_target=_COST.x_target)
    plan = causal_plan(model, jnp.ones(2), cost, 0.1, 12, -5.0, 5.0)
    cert = certify_safety(plan, model, _mixed, 0.1, gamma=3.0, cvar_gap=0.5, u_max=5.0)
    expected = confounding_robust_inflation(0.5, 0.0, 3.0) * (1.0 + math.sqrt(2.0))
    assert cert.radius == pytest.approx(expected * math.hypot(0.5, 1.0), rel=1e-12, abs=0.0)


class _Gain(eqx.Module):
    """``x' = -x + (1 + x) u``: a channel that grows with the state."""

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return -x + (1.0 + x) * u


def test_the_radius_is_read_at_the_plan_s_first_state() -> None:
    """A channel that moves with the state is sized once, where the plan starts: ``1 + x0``."""
    plan = causal_plan(_Gain(), jnp.array([2.0]), _COST_1, 0.1, 12, -1.0, 1.0)
    assert float(jnp.ptp(plan.trajectory)) > 0.1  # the channel moves along the plan

    def floor(x: Array) -> Array:
        return x[0] + 1.0

    cert = certify_safety(plan, _Gain(), floor, 0.1, gamma=2.0, u_max=1.0)
    assert cert.radius == pytest.approx(
        confounding_robust_inflation(1.0, 0.0, 2.0) * 3.0, rel=1e-12, abs=0.0
    )


def test_a_channel_that_is_not_a_number_scales_no_radius() -> None:
    model = LinearDynamics(-jnp.eye(1), jnp.array([[math.nan]]))
    plan = _scalar(1.0, 1.0)[0]
    with pytest.raises(ValueError, match="scales no radius"):
        certify_safety(plan, model, lambda x: x[0], 0.1, gamma=2.0, u_max=1.0)


def test_a_non_positive_cvar_gap_is_rejected_rather_than_inverted() -> None:
    with pytest.raises(ValueError, match="cvar_gap must be positive"):
        certify_safety(_PLAN, _MODEL, _mixed, 0.1, cvar_gap=0.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"cvar_gap": math.nan}, "must be positive and finite to scale a sensitivity radius"),
        ({"cvar_gap": math.inf}, "must be positive and finite to scale a sensitivity radius"),
        ({"gamma": math.nan}, "Gamma must be >= 1 and finite"),
        ({"gamma": math.inf}, "Gamma must be >= 1 and finite"),
        ({"u_max": math.nan}, "needs a positive actuation budget; u_max is nan"),
    ],
)
def test_a_sensitivity_that_is_no_number_is_refused_rather_than_priced(
    kwargs: dict[str, float], match: str
) -> None:
    """A nan passed both checks, and an infinite level read inf/inf: either way the radius was nan,
    and the audit certified no step, where the limit of an infinite level is the whole gap."""
    with pytest.raises(ValueError, match=match):
        certify_safety(_PLAN, _MODEL, _mixed, 0.1, **kwargs)


def test_a_plan_that_never_acts_names_itself_rather_than_the_inner_threshold() -> None:
    """The default budget is the plan's own action, so an idle plan must say *that*, not "u_max"."""
    idle = causal_plan(_MODEL, jnp.zeros(2), _COST, 0.1, 12, -5.0, 5.0)
    with pytest.raises(ValueError, match="it never acts"):
        certify_safety(idle, _MODEL, _mixed, 0.1)


# ---- what the audit reads, and where it refuses ----


def test_a_step_whose_path_leaves_the_set_is_not_certified() -> None:
    """At ``dt = 1`` the condition read at each step's state holds at all twelve steps, with room,
    while the RK4 step from the first state lands outside ``{h >= 0}``: ``x' = x/2 + u`` under the
    ceiling ``h = 1 - x`` and ``u_k = 0.95 - 1.5 x_k`` reaches ``x_1 = 1.232``. The audit read the
    condition alone and certified all twelve steps."""
    model = LinearDynamics(jnp.array([[0.5]]), jnp.array([[1.0]]))
    states, actions = [jnp.array([0.0])], []
    for _ in range(12):
        actions.append(0.95 - 1.5 * states[-1])
        states.append(rk4_step(model, 0.0, states[-1], actions[-1], 1.0))
    plan = replace(
        causal_plan(model, jnp.array([0.0]), _COST_1, 1.0, 12, -5.0, 5.0),
        actions=jnp.stack(actions),
        trajectory=jnp.stack(states),
    )

    def ceiling(x: Array) -> Array:
        return 1.0 - x[0]

    cert = certify_safety(plan, model, ceiling, 1.0, alpha=1.0, u_max=5.0)
    inside = 1.0 - np.asarray(plan.trajectory[:, 0]) >= 0.0
    assert bool(jnp.all(cert.guaranteed_derivative >= cert.required))  # the condition alone holds
    assert not inside[1]
    np.testing.assert_array_equal(cert.planned_certified, inside[:-1] & inside[1:])
    assert cert.certified_steps == 0


class _Hill(eqx.Module):
    """``x' = -x + 3 u^2 / (1 + u^2)``: a Hill curve of slope 2, no slope at zero action."""

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return -x + 3.0 * u**2 / (1.0 + u**2)


class _Cubic(eqx.Module):
    """``x' = -x + u + u^3``: a channel of 1 at zero action, of 13 at ``u = 2``."""

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return -x + u + u**3


@pytest.mark.parametrize("model", [_Hill(), _Cubic()], ids=["hill", "cubic"])
def test_a_field_not_affine_in_the_action_is_refused_rather_than_read_at_zero_action(
    model: eqx.Module,
) -> None:
    """The condition reads the drift and the channel at ``u = 0``. On these fields the audit
    certified the planned steps where the field at the planned action fails the condition at
    every one: on ``u + u^3`` from ``x = 0.5`` at ``u = 1`` the path leaves ``{x <= 1}`` by the
    third step."""
    actions = jnp.ones((10, 1))
    plan = replace(
        causal_plan(_Cubic(), jnp.array([0.5]), _COST_1, 0.1, 10, -2.0, 2.0),
        actions=actions,
        trajectory=rollout(model, jnp.array([0.5]), actions, 0.1),
    )
    with pytest.raises(ValueError, match="not affine in the action at step 0"):
        certify_safety(plan, model, lambda x: 1.0 - x[0], 0.1, u_max=2.0)


def test_a_field_affine_in_the_action_whose_channel_moves_with_the_state_is_not_refused() -> None:
    """``x' = -x + (1 + x) u`` is affine in the action at every state, though its channel moves
    with the state: the check reads the field at each step's own state."""
    plan = causal_plan(_Gain(), jnp.array([2.0]), _COST_1, 0.1, 12, -1.0, 1.0)
    cert = certify_safety(plan, _Gain(), lambda x: x[0] + 1.0, 0.1, gamma=2.0, u_max=1.0)
    assert len(cert.step_gamma_star) == 12


def _one_sided(lo: float, hi: float) -> tuple[CausalPlan, Callable[[Array], Array]]:
    """``x' = x + u`` from ``x = 1`` under the ceiling ``x <= 1.1``, which only a negative action
    keeps, held in the solve on the box ``[lo, hi]``."""
    model = LinearDynamics(jnp.array([[1.0]]), jnp.array([[1.0]]))

    def ceiling(x: Array) -> Array:
        return 1.1 - x[0]

    cost = QuadraticCost(Q=jnp.eye(1), R=jnp.eye(1), Qf=jnp.eye(1), x_target=jnp.array([1.0]))
    barrier = BarrierConstraint(ceiling, alpha=1.0)
    plan = causal_plan(model, jnp.array([1.0]), cost, 0.1, 10, lo, hi, barrier=barrier, steps=500)
    return plan, ceiling


def test_gamma_star_reads_the_box_not_the_largest_action_either_way() -> None:
    """On the box ``[0, 5]`` no admissible action lowers ``x``, so no step's condition holds at
    any level, where the audit read the box as ``[-5, 5]`` and found ceilings from 10.1 down to
    1.04. On ``[-1, 5]`` the best action is ``-1``, and each step's ceiling is the one read with
    ``u_max = 1``, exactly: one lever's best action in a box is at an end of it."""
    plan, ceiling = _one_sided(0.0, 5.0)
    assert plan.safety is not None
    assert all(math.isnan(g) for g in plan.safety.step_gamma_star)

    plan, ceiling = _one_sided(-1.0, 5.0)
    assert plan.safety is not None
    model = LinearDynamics(jnp.array([[1.0]]), jnp.array([[1.0]]))
    one = certify_safety(plan, model, ceiling, 0.1, alpha=1.0, u_max=1.0)
    assert plan.safety.step_gamma_star == one.step_gamma_star
    assert any(math.isfinite(g) for g in one.step_gamma_star)


def test_a_start_that_is_not_a_state_is_refused() -> None:
    """A nan start predicted a nan path, and its tube certified all ten steps."""
    with pytest.raises(ValueError, match="is not a state"):
        causal_plan(_MODEL, jnp.array([1.0, jnp.nan]), _COST, 0.1, 10, -5.0, 5.0, model_error=0.01)


def test_a_tube_certifies_no_step_past_the_path_s_overflow() -> None:
    """RK4 multiplies ``x' = 200 x`` by 8221 a step at ``dt = 0.1``, past float64's range after 78
    steps of 100; a tube carried at no rate and at a tolerance of ``inf`` covered all of them."""
    model = LinearDynamics(jnp.array([[200.0]]), jnp.array([[1.0]]))
    plan = causal_plan(
        model, jnp.array([1.0]), _COST_1, 0.1, 100, -1.0, 1.0, model_error=1e-300, steps=1
    )
    finite = np.isfinite(np.asarray(plan.trajectory[:, 0]))
    assert not finite.all()
    assert plan.certified_horizon == int(np.argmin(finite)) - 1
    assert plan.certificate_status == "partial"


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
    #    realised gap, and it is worth having only if it does so without being vacuous. The plans
    #    are steepest descent's, in the caller's units and cut short: the planner's own steps,
    #    scaled by the cost's curvature, take the 0.4 box's plan to its optimum in one, and leave
    #    the 2.0 box's 0.29 from it after three, where the bound measured 3.5 times that.
    from chc.control import _projected_gradient_solve

    for lo, hi, steps, ceiling in ((-2.0, 2.0, 3, 2.0), (-2.0, 2.0, 200, 1.5), (-0.4, 0.4, 3, 1.2)):
        descent = _projected_gradient_solve(
            plant, x0, jnp.zeros((horizon, 1)), dt, cost, lo, hi, steps, 0.2, 1e-9, (), scaled=False
        )
        plan = causal_plan(
            plant, x0, cost, dt, horizon, lo, hi, steps=0, warm_start=descent.actions
        )
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
    assert curve.status == "diagnostic"  # a sampled modulus bounds nothing between its samples
    assert curve.modulus_source == "measured"
    assert curve.modulus > 0.0
    assert curve.bound >= hybrid_plan.task_cost - reached(hybrid, -0.5, 0.5, 1, 2) - 1e-12

    # 4. WHERE THE MODULUS COMES FROM. For a linear plant J is exactly quadratic, so the curvature
    #    at the plan is the global one -- and with Q and Qf positive semidefinite lambda_min(R) is a
    #    valid floor under it (STEP 7b), which STEP 4d says can only make the bound larger.
    slack = causal_plan(plant, x0, cost, dt, horizon, -2.0, 2.0, steps=20_000)
    measured = plan_regret_bound(slack, plant, x0, cost, dt, -2.0, 2.0, probes=16)
    assert measured.modulus_source == "measured"
    assert measured.status == "certified"
    hessian = jax.hessian(lambda v: total_cost(plant, x0, v.reshape(horizon, 1), dt, cost))(
        slack.actions.reshape(-1)
    )
    assert measured.modulus == pytest.approx(
        float(jnp.min(jnp.linalg.eigvalsh(hessian))), rel=1e-10
    )
    assert measured.modulus >= float(jnp.min(jnp.linalg.eigvalsh(cost.R)))
    conservative = plan_regret_bound(slack, plant, x0, cost, dt, -2.0, 2.0, modulus=0.05)
    assert conservative.modulus_source == "supplied"
    assert conservative.status == "certified"
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
    assert verdict.status == "refused"
    assert verdict.modulus < 0.0
    assert np.isinf(verdict.bound)

    with pytest.raises(ValueError, match="cannot be negative"):
        plan_regret_bound(slack, plant, x0, cost, dt, -2.0, 2.0, modulus=-1.0)


def _box_minimum(hessian: np.ndarray, linear: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """``min 0.5 x'Hx + c'x`` over a box, ``H`` positive definite, exactly: each action free, at
    its lower bound or at its upper one, the free ones solving their rows. The optimum's own
    pattern is among them and every feasible candidate is a point of the box, so the least
    feasible value is the minimum."""
    best = math.inf
    for pattern in product((0, 1, 2), repeat=linear.size):
        at = np.array(pattern)
        x = np.where(at == 1, lo, np.where(at == 2, hi, 0.0))
        if not np.all(np.isfinite(x)):
            continue  # pinned to a side the box leaves open
        free, held = at == 0, at != 0
        if free.any():
            rows = -linear[free] - hessian[np.ix_(free, held)] @ x[held]
            x[free] = np.linalg.solve(hessian[np.ix_(free, free)], rows)
        if np.all(x >= lo) and np.all(x <= hi):
            best = min(best, float(0.5 * x @ hessian @ x + linear @ x))
    return best


def _exact_minimum(
    plant: LinearDynamics,
    x0: Array,
    cost: QuadraticCost,
    dt: float,
    horizon: int,
    lo: np.ndarray,
    hi: np.ndarray,
) -> float:
    """The least cost a linear plant's box allows: its cost is quadratic in the actions, read off
    at zero and minimised by :func:`_box_minimum`."""
    levers = lo.size

    def objective(us: Array) -> Array:
        return total_cost(plant, x0, us.reshape(horizon, levers), dt, cost)

    zero = jnp.zeros(horizon * levers)
    least = _box_minimum(
        np.asarray(jax.hessian(objective)(zero)),
        np.asarray(jax.grad(objective)(zero)),
        np.tile(lo, horizon),
        np.tile(hi, horizon),
    )
    return least + float(objective(zero))


def test_the_regret_bound_reads_no_lever_s_units_and_holds_on_the_exact_optimum() -> None:
    """Levers priced 1e-3 to 10 a unit, boxes of unlike widths, one side left open on some: the
    bound is at least the exact regret, the box's quadratic programme solved by enumeration, and
    reads the same with any lever in units 1e-3 or 1e3 times its own, alone or with the others.
    Its modulus was the Hessian's least eigenvalue, the cheapest lever's curvature in whatever
    units the levers came in, and the bound read those units."""
    rng = np.random.default_rng(86)
    dt = 0.1
    for _ in range(12):
        levers = int(rng.integers(2, 4))
        horizon = 6 // levers
        states = int(rng.integers(1, 4))
        drift = jnp.asarray(rng.normal(scale=0.7, size=(states, states)))
        channel = jnp.asarray(rng.normal(size=(states, levers)))
        plant = LinearDynamics(drift, channel)
        prices = np.exp(rng.uniform(np.log(1e-3), np.log(10.0), size=levers))
        cost = QuadraticCost(
            Q=jnp.eye(states),
            R=jnp.diag(jnp.asarray(prices)),
            Qf=2.0 * jnp.eye(states),
            x_target=jnp.asarray(rng.normal(size=states)),
        )
        x0 = jnp.asarray(rng.normal(size=states))
        lo = -np.exp(rng.uniform(np.log(0.1), np.log(3.0), size=levers))
        hi = np.exp(rng.uniform(np.log(0.1), np.log(3.0), size=levers))
        start = rng.uniform(lo, hi, size=(horizon, levers))
        hi[rng.random(levers) < 0.3] = np.inf
        plan = causal_plan(plant, x0, cost, dt, horizon, lo, hi, steps=0, warm_start=start)
        gap = plan_regret_bound(plan, plant, x0, cost, dt, lo, hi)
        assert gap.status == "certified"

        best = _exact_minimum(plant, x0, cost, dt, horizon, lo, hi)
        regret = plan.task_cost - best
        assert gap.bound >= regret - 1e-12 * plan.task_cost

        for units in ((1e3, 1.0), (1.0, 1e-3), (1e-3, 1e3)):
            c = np.resize(np.asarray(units), levers)  # u = c u', so B u = (B / c) u'
            moved = LinearDynamics(drift, channel / c)
            priced = QuadraticCost(
                Q=cost.Q, R=cost.R / np.outer(c, c), Qf=cost.Qf, x_target=cost.x_target
            )
            replanned = causal_plan(
                moved, x0, priced, dt, horizon, c * lo, c * hi, steps=0, warm_start=c * start
            )
            other = plan_regret_bound(replanned, moved, x0, priced, dt, c * lo, c * hi)
            assert other.status == "certified"
            assert other.bound == pytest.approx(gap.bound, rel=1e-9, abs=0.0), units
            np.testing.assert_allclose(other.per_lever, gap.per_lever, rtol=1e-9, atol=0.0)
            assert other.unconstrained_bound == pytest.approx(
                gap.unconstrained_bound, rel=1e-9, abs=0.0
            )
        assert gap.unconstrained_bound == pytest.approx(
            gap.gradient_norm**2 / (2.0 * gap.modulus), rel=1e-12, abs=0.0
        )


def test_one_lever_that_moves_has_the_plain_norm_s_regret_bound_to_the_bit() -> None:
    """One lever that moves has scale 1, beside a lever its box pins, however stiff: the measured
    modulus is the least eigenvalue of the moving lever's Hessian, and the bound is the one that
    modulus gives supplied, in the plain norm, bit for bit, as before the levers were scaled."""
    plant = LinearDynamics(
        jnp.array([[0.0, 1.0], [-2.0, -0.3]]), jnp.array([[0.0, 0.5], [1.0, 0.2]])
    )
    cost = QuadraticCost(
        Q=jnp.eye(2),
        R=jnp.diag(jnp.array([0.05, 30.0])),
        Qf=3.0 * jnp.eye(2),
        x_target=jnp.zeros(2),
    )
    x0 = jnp.array([1.5, 0.0])
    lo, hi = jnp.array([-2.0, 0.7]), jnp.array([2.0, 0.7])
    plan = causal_plan(plant, x0, cost, 0.1, 12, lo, hi, steps=3)
    measured = plan_regret_bound(plan, plant, x0, cost, 0.1, lo, hi)
    plain = plan_regret_bound(plan, plant, x0, cost, 0.1, lo, hi, modulus=measured.modulus)
    hessian = jax.hessian(lambda v: total_cost(plant, x0, v.reshape(12, 2), 0.1, cost))(
        plan.actions.reshape(-1)
    )
    moving = np.asarray(hessian)[0::2, 0::2]
    assert measured.modulus == pytest.approx(np.linalg.eigvalsh(moving)[0], rel=1e-12, abs=0.0)
    assert measured.bound > 0.0
    assert (measured.bound, measured.per_lever) == (plain.bound, plain.per_lever)


class _Convex(eqx.Module):
    """A rate convex in the cheap lever: the objective's curvature along it moves over the box."""

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return jnp.array([-x[0] + u[0] + 0.3 * u[0] ** 2 + 0.5 * u[1]])


def test_a_sampled_modulus_reads_every_point_drawn_in_the_plan_s_scales() -> None:
    """The levers' scales come from the Hessian's diagonal at the plan and stay fixed over the
    sample: the modulus is the least eigenvalue of ``D H D`` over the plan and every point drawn,
    one ``D``. Scales read at another point are as valid but measure each point in other units."""
    model = _Convex()
    cost = QuadraticCost(
        Q=jnp.eye(1),
        R=jnp.diag(jnp.array([0.01, 1.0])),
        Qf=2.0 * jnp.eye(1),
        x_target=jnp.array([-3.0]),
    )
    x0, dt, horizon, probes = jnp.zeros(1), 0.1, 3, 4
    lo, hi = jnp.array([-1.0, -0.5]), jnp.array([1.0, 1.5])
    plan = causal_plan(model, x0, cost, dt, horizon, lo, hi, steps=0, warm_start=jnp.zeros((3, 2)))
    hessian = jax.hessian(lambda v: total_cost(model, x0, v.reshape(horizon, 2), dt, cost))
    for seed in range(3):
        gap = plan_regret_bound(plan, model, x0, cost, dt, lo, hi, probes=probes, seed=seed)
        assert gap.status == "diagnostic"
        drawn = jax.random.uniform(
            jax.random.PRNGKey(seed),
            (probes, 2 * horizon),
            minval=jnp.tile(lo, horizon),
            maxval=jnp.tile(hi, horizon),
        )
        points = jnp.concatenate([plan.actions.reshape(1, -1), drawn])
        curvature = np.diag(np.asarray(hessian(points[0]))).reshape(horizon, 2).mean(axis=0)
        scale = np.tile(np.sqrt(curvature.max() / curvature), horizon)
        least = min(
            np.linalg.eigvalsh(scale[:, None] * np.asarray(hessian(point)) * scale[None, :])[0]
            for point in points
        )
        assert least > 0.0
        assert gap.modulus == pytest.approx(least, rel=1e-12, abs=0.0)


class _HiddenPocket(eqx.Module):
    """A rate of 1 at every action but a pocket 0.02 wide at 1.3, where it falls to 0."""

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return jnp.array([1.0 - jnp.exp(-(((u[0] - 1.3) / 0.02) ** 2))])


def test_a_sampled_curvature_is_a_diagnostic_never_a_certificate() -> None:
    """The review's hidden pocket: the plan at 0 costs 1, the pocket's action 0.00845, so the plan's
    regret is at least 0.99, while the curvature sampled at the plan and 16 points reads 0.01, flat,
    and the bound 0. It was reported certified; it is that number still, and a diagnostic."""
    model = _HiddenPocket()
    cost = QuadraticCost(
        Q=jnp.zeros((1, 1)), R=jnp.array([[0.01]]), Qf=jnp.array([[2.0]]), x_target=jnp.zeros(1)
    )
    x0 = jnp.zeros(1)
    plan = causal_plan(model, x0, cost, 1.0, 1, -2.0, 2.0, steps=10, warm_start=jnp.zeros((1, 1)))
    pocket = float(total_cost(model, x0, jnp.array([[1.3]]), 1.0, cost))
    assert plan.task_cost - pocket > 0.99
    gap = plan_regret_bound(plan, model, x0, cost, 1.0, -2.0, 2.0)
    assert gap.bound < plan.task_cost - pocket  # the samples miss the pocket
    assert gap.modulus_source == "measured"
    assert gap.status == "diagnostic"
    # a modulus the caller supplies is the caller's claim, and it certifies
    claimed = plan_regret_bound(plan, model, x0, cost, 1.0, -2.0, 2.0, modulus=gap.modulus)
    assert claimed.status == "certified"


@pytest.mark.parametrize("driven", [False, True])
@pytest.mark.parametrize(
    ("degree", "channel_degree", "status"),
    [(1, 0, "certified"), (0, 0, "certified"), (2, 0, "diagnostic"), (1, 1, "diagnostic")],
)
def test_the_objective_is_quadratic_only_where_the_field_is_affine(
    degree: int, channel_degree: int, status: str, driven: bool
) -> None:
    """A drift of degree 1 beside a constant channel keeps every RK4 step affine, so one Hessian is
    the box's; a quadratic drift or a channel that reads the state bends the rollout, however small
    the coefficient that does it. A driver's push reads neither the state nor the action, and
    changes neither."""
    features = {0: 1, 1: 3, 2: 6}  # monomials of two states up to each degree, bias first
    drift = 0.01 * jnp.ones((2, features[degree]))
    channel = 0.01 * jnp.ones((2, 1, features[channel_degree]))
    known = LinearDynamics(jnp.array([[0.0, 1.0], [-2.0, -0.3]]), jnp.array([[0.0], [1.0]]))
    model: Dynamics = HybridDynamics(
        known, ControlAffineResidual(drift, channel, degree, channel_degree)
    )
    if driven:
        model = DrivenDynamics(model, jnp.ones((2, 1)), jnp.linspace(0.0, 0.3, 7)[:, None], 0.1)
    cost = QuadraticCost(Q=jnp.eye(2), R=0.05 * jnp.eye(1), Qf=jnp.eye(2), x_target=jnp.zeros(2))
    x0 = jnp.array([1.0, 0.0])
    plan = causal_plan(model, x0, cost, 0.1, 6, -1.0, 1.0, steps=200)
    gap = plan_regret_bound(plan, model, x0, cost, 0.1, -1.0, 1.0, probes=4)
    assert gap.modulus_source == "measured"
    assert gap.status == status


@pytest.mark.parametrize(
    ("model", "quadratic"),
    [
        (LinearDynamics(_A, _B), True),
        (DampedOscillator(1.0, 0.2), True),
        (_MODEL, True),  # a linear plant with a zero residual
        (
            HybridDynamics(
                DampedOscillator(1.0, 0.2), MLPResidual(2, 1, 2, key=jax.random.PRNGKey(3))
            ),
            False,
        ),
        (_HiddenPocket(), False),
    ],
    ids=["linear", "oscillator", "zero-residual", "mlp-residual", "unnamed"],
)
def test_only_a_field_named_affine_makes_the_objective_quadratic(
    model: Dynamics, quadratic: bool
) -> None:
    assert _quadratic(model) is quadratic


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lipschitz": math.inf, "model_error": 0.1}, r"lipschitz=inf bounds nothing"),
        ({"lipschitz": 1.0, "model_error": math.nan}, r"infinite or nan: nan"),
        ({"lipschitz": 1.0, "model_error": math.inf}, r"infinite or nan: inf"),
        ({"lipschitz": 1.0, "model_error": 0.1, "tolerance": math.nan}, r"tolerance=nan is not"),
        ({"lipschitz": 1.0, "model_error": 0.1, "tolerance": -0.1}, r"tolerance=-0\.1 is not"),
    ],
)
def test_a_plan_refuses_what_its_tube_cannot_be_read_on(kwargs: dict, match: str) -> None:
    """``lipschitz=inf`` read the tube ``[0, nan, nan, nan]`` and certified 3 steps of 3 at a
    tolerance of 0.12; a nan budget read no tube at all, and said nothing of why."""
    model = LinearDynamics(jnp.array([[1.0]]), jnp.zeros((1, 1)))
    cost = QuadraticCost(Q=jnp.zeros((1, 1)), R=jnp.eye(1), Qf=jnp.eye(1), x_target=jnp.zeros(1))
    with pytest.raises(ValueError, match=match):
        causal_plan(
            model, jnp.zeros(1), cost, 1.0, 3, -1.0, 1.0, steps=2, **{"tolerance": 0.12, **kwargs}
        )


@pytest.mark.parametrize("dt", [0.0, -1.0, math.nan, math.inf])
def test_a_plan_refuses_a_step_that_is_no_step(dt: float) -> None:
    with pytest.raises(ValueError, match=r"is not a step"):
        causal_plan(_MODEL, _X0, _COST, dt, 12, -5.0, 5.0)


@pytest.mark.parametrize("modulus", [math.nan, math.inf])
def test_a_modulus_that_is_no_curvature_is_refused(modulus: float) -> None:
    """``modulus=nan`` read ``certified`` with the bound ``inf``, and ``modulus=inf`` ``certified``
    with the bound nan."""
    plan = causal_plan(*_ARGS)
    with pytest.raises(ValueError, match=r"cannot be negative, infinite or nan"):
        plan_regret_bound(plan, _MODEL, _X0, _COST, 0.1, -5.0, 5.0, modulus=modulus)


def test_a_bound_read_off_numbers_that_are_not_finite_is_refused() -> None:
    """A nan in the model makes the Hessian's least eigenvalue nan, which is neither below 0 nor a
    curvature: it read ``certified`` with an infinite bound. On a supplied modulus the gradient is
    nan, and so was the certified bound."""
    plan = causal_plan(*_ARGS)
    broken = HybridDynamics(
        known=LinearDynamics(_A.at[0, 0].set(jnp.nan), _B), residual=ZeroResidual(2)
    )
    for modulus in (None, 0.05):
        verdict = plan_regret_bound(plan, broken, _X0, _COST, 0.1, -5.0, 5.0, modulus=modulus)
        assert verdict.status == "refused"
        assert math.isinf(verdict.bound)
        assert all(math.isinf(share) for share in verdict.per_lever)
