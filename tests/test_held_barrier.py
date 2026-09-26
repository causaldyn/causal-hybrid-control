"""A barrier held inside the solve: the audit's own condition, reached by augmented Lagrangian.

The oracle shares nothing with the solver. On a linear plant with an affine barrier the condition
``grad h . f(x_k, u_k) - d |u_k| >= -alpha h(x_k)`` is two linear rows per step in the actions --
one per sign of ``u_k`` -- so the program is a strictly convex QP, solved exactly by a Cholesky
change of variables and Lawson & Hanson's least-distance program, one NNLS call.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from scipy.optimize import nnls

from chc import (
    BarrierConstraint,
    CausalPlan,
    DampedOscillator,
    QuadraticCost,
    SupportModel,
    causal_plan,
    certify_safety,
)
from chc.cost import total_cost
from chc.dynamics import HybridDynamics, LinearDynamics
from chc.epidemic import SIRDynamics
from chc.integrate import rollout
from chc.residual import ZeroResidual
from chc.uncertainty import confounding_robust_inflation

DT = 0.1
OMEGA, ZETA = 1.0, 0.1
OSCILLATOR = DampedOscillator(omega=OMEGA, zeta=ZETA)
X0 = jnp.array([1.0, 0.0])
COST = QuadraticCost(
    Q=jnp.diag(jnp.array([1.0, 0.0])),
    R=jnp.array([[0.01]]),
    Qf=jnp.diag(jnp.array([10.0, 1.0])),
    x_target=jnp.zeros(2),
)
HORIZON = 20
U_MAX = 5.0
# At this floor the optimum binds on 13 steps of 20 at gamma = 1 and 14 at gamma = 2; at 0.3 it
# binds on all of them, and the condition alone would then pin the plan whatever the objective.
FLOOR, ALPHA = 0.8, 2.0
# The rounds settle a relative 1e-6 inside the condition; that back-off, not their inexactness, is
# most of the price, and it measured 2.2e-5 at worst on the instances in the ADR.
GAP = 1e-4


def _velocity_floor(x: Array) -> Array:
    """Safe while the oscillator moves towards the origin no faster than ``FLOOR``."""
    return x[1] + FLOOR


def _least_distance(e: np.ndarray, f: np.ndarray) -> np.ndarray:
    """``argmin |w|`` subject to ``e @ w >= f`` (Lawson & Hanson 1974, ch. 23), by one NNLS."""
    n = e.shape[1]
    stacked = np.vstack([e.T, f[None, :]])
    target = np.zeros(n + 1)
    target[-1] = 1.0
    weights, _ = nnls(stacked, target, maxiter=100 * max(e.shape))
    residual = stacked @ weights - target
    assert abs(residual[-1]) > 1e-12, "the oracle's feasible set is empty"
    return -residual[:n] / residual[-1]


def _floor_rows(flat: Array, gamma: float, floor: float) -> Array:
    """A velocity floor's condition at ``gamma`` as two rows per step, each ``>= 0`` when it holds.

    Written out from the oscillator's equations here, not taken from the library.
    """
    spread = confounding_robust_inflation(1.0, 0.0, gamma)  # ||grad h|| = 1 for this barrier
    us = flat.reshape(HORIZON, 1)
    xs = rollout(OSCILLATOR, X0, us, DT)[:-1]
    drift = -(OMEGA**2) * xs[:, 0] - 2.0 * ZETA * OMEGA * xs[:, 1]
    required = -ALPHA * (xs[:, 1] + floor)
    # the worst case of -spread*|u| is the smaller of the two branches, so both must hold
    return jnp.concatenate(
        [
            drift + (1.0 - spread) * us[:, 0] - required,
            drift + (1.0 + spread) * us[:, 0] - required,
        ]
    )


def _exact_floor_optimum(
    gamma: float, floor: float = FLOOR, penalty: Callable[[Array, Array], Array] | None = None
) -> tuple[np.ndarray, float]:
    """The exact optimum under a velocity floor's condition at ``gamma``, and its objective.

    ``penalty(xs, us)`` joins the task cost, and must stay quadratic in the actions -- as a support
    model's is on this linear plant -- for the Hessian taken at zero to be the program's own.
    """

    def value(flat: Array) -> Array:
        us = flat.reshape(HORIZON, 1)
        task = total_cost(OSCILLATOR, X0, us, DT, COST)
        if penalty is None:
            return task
        return task + penalty(rollout(OSCILLATOR, X0, us, DT)[:-1], us)

    def condition(flat: Array) -> Array:
        return _floor_rows(flat, gamma, floor)

    zero = jnp.zeros(HORIZON)
    hessian = np.asarray(jax.hessian(value)(zero))
    hessian = 0.5 * (hessian + hessian.T)
    gradient = np.asarray(jax.grad(value)(zero))
    rows = np.asarray(jax.jacobian(condition)(zero))
    lower = -np.asarray(condition(zero))
    g = np.vstack([-rows, np.eye(HORIZON), -np.eye(HORIZON)])
    h = np.concatenate([-lower, np.full(HORIZON, U_MAX), np.full(HORIZON, U_MAX)])
    probe = np.random.default_rng(0).normal(size=HORIZON)
    quadratic = float(value(zero)) + gradient @ probe + 0.5 * probe @ hessian @ probe
    assert abs(float(value(jnp.asarray(probe))) - quadratic) <= 1e-9 * abs(quadratic), "not a QP"
    free = -np.linalg.solve(hessian, gradient)
    back = np.linalg.inv(np.linalg.cholesky(hessian)).T  # u = free + back @ w, |w|^2 = 2 J + c
    optimum = free + back @ _least_distance(-g @ back, g @ free - h)
    assert float(np.min(np.asarray(condition(jnp.asarray(optimum))))) > -1e-10
    return optimum, float(value(jnp.asarray(optimum)))


@pytest.mark.parametrize("gamma", [1.0, 2.0])
def test_the_held_condition_lands_on_the_exact_optimum(gamma: float) -> None:
    """``gamma = 2`` makes the worst case bind: ``d |u|`` has a kink the rounds must not stall."""
    free = causal_plan(OSCILLATOR, X0, COST, DT, HORIZON, -U_MAX, U_MAX)
    plan = causal_plan(
        OSCILLATOR,
        X0,
        COST,
        DT,
        HORIZON,
        -U_MAX,
        U_MAX,
        barrier=BarrierConstraint(_velocity_floor, alpha=ALPHA, gamma=gamma),
    )
    _, optimum = _exact_floor_optimum(gamma)

    assert plan.safety is not None
    assert plan.solver_status == "converged"
    assert plan.safety.certified_steps == HORIZON
    unheld = certify_safety(free, OSCILLATOR, _velocity_floor, DT, alpha=ALPHA, gamma=gamma)
    assert unheld.certified_steps < HORIZON  # the free plan breaks it, so the barrier binds
    assert plan.task_cost >= optimum * (1.0 - 1e-12)  # feasible, so it cannot beat the optimum
    assert (plan.task_cost - optimum) / optimum < GAP


def test_the_log_counts_the_steps_the_exact_optimum_binds_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``active_steps`` counts positive multipliers, which name the optimum's active set.

    A penalty with no multipliers reaches the same plan here, back-off and all, and would log zero.
    """
    with caplog.at_level(logging.INFO, logger="chc.plan"):
        causal_plan(
            OSCILLATOR,
            X0,
            COST,
            DT,
            HORIZON,
            -U_MAX,
            U_MAX,
            barrier=BarrierConstraint(_velocity_floor, alpha=ALPHA),
        )
    optimum, _ = _exact_floor_optimum(1.0)
    rows = np.asarray(_floor_rows(jnp.asarray(optimum), 1.0, FLOOR))
    slack = np.minimum(rows[:HORIZON], rows[HORIZON:])
    [held] = [r for r in caplog.records if getattr(r, "chc_event", None) == "barrier"]
    assert np.min(slack[slack > 1e-9]) > 0.1  # the active set is unambiguous at this floor
    assert getattr(held, "active_steps", None) == int(np.sum(slack <= 1e-9)) == 13


@pytest.mark.parametrize("steps", [10_000, 5])
def test_a_barrier_the_free_plan_already_clears_changes_no_action(steps: int) -> None:
    """``steps = 5`` stops the free solve short; a slack barrier must not resume the descent."""
    free = causal_plan(OSCILLATOR, X0, COST, DT, HORIZON, -U_MAX, U_MAX, steps=steps)
    held = causal_plan(
        OSCILLATOR,
        X0,
        COST,
        DT,
        HORIZON,
        -U_MAX,
        U_MAX,
        steps=steps,
        barrier=BarrierConstraint(lambda x: x[1] + 100.0, alpha=ALPHA),
    )
    assert np.array_equal(np.asarray(free.actions), np.asarray(held.actions))
    assert (held.solver_status, held.solver_iterations) == (
        free.solver_status,
        free.solver_iterations,
    )
    assert held.safety is not None
    assert held.safety.certified_steps == HORIZON


def test_the_plan_carries_the_audit_itself_rather_than_the_solvers_report() -> None:
    """Every argument must reach the audit, the box's budget included.

    The floor is tighter than ``FLOOR``: there standing still clears the condition at every step,
    so no step needs any authority and ``gamma_star`` is ``inf`` on every budget. Here some steps
    need it, and the budget shows.
    """

    def tight_floor(x: Array) -> Array:
        return x[1] + 0.3

    requirement = BarrierConstraint(tight_floor, alpha=ALPHA, gamma=1.5, cvar_gap=0.5)
    plan = causal_plan(OSCILLATOR, X0, COST, DT, HORIZON, -U_MAX, U_MAX, barrier=requirement)
    audit = certify_safety(
        plan, OSCILLATOR, tight_floor, DT, alpha=ALPHA, gamma=1.5, cvar_gap=0.5, u_max=U_MAX
    )
    own_budget = certify_safety(
        plan, OSCILLATOR, tight_floor, DT, alpha=ALPHA, gamma=1.5, cvar_gap=0.5
    )
    assert own_budget.gamma_star != audit.gamma_star  # the budget is visible to this test
    assert plan.safety is not None
    for field in ("barrier_values", "guaranteed_derivative", "required", "planned_certified"):
        assert np.array_equal(
            np.asarray(getattr(plan.safety, field)), np.asarray(getattr(audit, field))
        )
    assert plan.safety.certified_steps == audit.certified_steps
    assert plan.safety.gamma_star == audit.gamma_star
    assert plan.safety.step_gamma_star == audit.step_gamma_star
    assert plan.safety.radius == audit.radius


def test_a_capacity_barrier_holds_on_a_nonlinear_epidemic() -> None:
    """``I <= 0.1`` on SIR, whose channel ``beta s i`` moves with the state: nothing is linear."""
    model = SIRDynamics(beta=0.6, gamma=0.1)
    start = jnp.array([0.99, 0.01])
    effort = QuadraticCost(
        Q=jnp.zeros((2, 2)), R=jnp.array([[1.0]]), Qf=jnp.zeros((2, 2)), x_target=jnp.zeros(2)
    )
    horizon = 60

    def capacity(x: Array) -> Array:
        return 0.1 - x[1]

    free = causal_plan(model, start, effort, 1.0, horizon, 0.0, 0.9)
    plan = causal_plan(
        model, start, effort, 1.0, horizon, 0.0, 0.9, barrier=BarrierConstraint(capacity, alpha=0.5)
    )
    assert plan.safety is not None
    assert (
        certify_safety(free, model, capacity, 1.0, alpha=0.5, u_max=0.9).certified_steps < horizon
    )
    assert plan.solver_status == "converged"
    assert plan.safety.certified_steps == horizon
    assert float(jnp.max(plan.trajectory[:, 1])) <= 0.1


def test_an_unreachable_barrier_is_reported_by_the_audit_rather_than_raised() -> None:
    """Position is relative degree 2 here: at the first step no action moves the condition."""
    plant = HybridDynamics(
        known=LinearDynamics(jnp.array([[-0.5, 1.0], [0.0, -0.3]]), jnp.array([[0.0], [1.0]])),
        residual=ZeroResidual(2),
    )
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.1])),
        R=jnp.array([[0.05]]),
        Qf=jnp.diag(jnp.array([5.0, 1.0])),
        x_target=jnp.zeros(2),
    )
    plan = causal_plan(
        plant,
        jnp.array([1.0, 0.0]),
        cost,
        DT,
        12,
        -5.0,
        5.0,
        steps=400,
        barrier=BarrierConstraint(lambda x: x[0], alpha=0.2),
    )
    assert plan.safety is not None
    assert plan.solver_status == "max_iterations"
    assert plan.safety.certified_steps == 0
    assert np.isnan(plan.safety.step_gamma_star[0])  # no radius and no action certifies step 0


def test_the_rounds_keep_the_plans_own_pessimism() -> None:
    """The rounds must minimise what the plan minimised, penalties included, not the task alone.

    A second support model stands in for the uncertainty penalty: on this linear plant both are
    quadratic in the actions, so the pessimistic program keeps an exact optimum. Its own floor makes
    the rounds run with 10 steps of 20 binding: at ``FLOOR`` the pessimistic plan already clears the
    condition, and at 0.3 every step binds and the condition alone fixes the plan.
    """
    floor = 0.5
    support = SupportModel.fit(
        0.3 * jax.random.normal(jax.random.key(0), (200, 2)),
        0.3 * jax.random.normal(jax.random.key(1), (200, 1)),
    )
    lean = SupportModel.fit(
        0.3 * jax.random.normal(jax.random.key(2), (200, 2)),
        0.5 + 0.3 * jax.random.normal(jax.random.key(3), (200, 1)),
    )

    def penalty(xs: Array, us: Array) -> Array:
        return 0.1 * support.penalty_trajectory(xs, us) + lean.penalty_trajectory(xs, us)

    def lower_floor(x: Array) -> Array:
        return x[1] + floor

    def pessimistic(barrier: BarrierConstraint | None) -> CausalPlan:
        return causal_plan(
            OSCILLATOR,
            X0,
            COST,
            DT,
            HORIZON,
            -U_MAX,
            U_MAX,
            support=support,
            lam_supp=0.1,
            uncertainty=lean,
            lam_unc=1.0,
            barrier=barrier,
        )

    free = pessimistic(None)
    plan = pessimistic(BarrierConstraint(lower_floor, alpha=ALPHA))
    _, optimum = _exact_floor_optimum(1.0, floor, penalty)
    value = plan.task_cost + float(penalty(plan.trajectory[:-1], plan.actions))

    assert plan.safety is not None
    assert certify_safety(free, OSCILLATOR, lower_floor, DT, alpha=ALPHA).certified_steps < HORIZON
    assert plan.solver_status == "converged"
    assert plan.safety.certified_steps == HORIZON
    assert value >= optimum * (1.0 - 1e-12)
    # the optimum of the same program without the uncertainty term scores 3.6% above this one
    assert (value - optimum) / optimum < GAP


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"gamma": 0.9}, "gamma is a sensitivity level"), ({"cvar_gap": 0.0}, "cvar_gap must be")],
)
def test_a_barrier_constraint_refuses_what_certify_safety_could_not_price(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BarrierConstraint(_velocity_floor, **kwargs)


def test_a_barrier_needs_a_box_that_admits_an_action() -> None:
    with pytest.raises(ValueError, match="admits a nonzero action"):
        causal_plan(
            OSCILLATOR, X0, COST, DT, HORIZON, 0.0, 0.0, barrier=BarrierConstraint(_velocity_floor)
        )
