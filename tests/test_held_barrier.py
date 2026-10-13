"""A barrier held inside the solve: the audit's own condition, reached by augmented Lagrangian.

The oracle shares nothing with the solver. On a linear plant with an affine barrier the condition
``grad h . f(x_k, u_k) - d |u_k| >= -alpha h(x_k)`` is two linear rows per step in the actions --
one per sign of ``u_k`` -- so the program is a strictly convex QP, solved exactly by a Cholesky
change of variables and Lawson & Hanson's least-distance program, one NNLS call.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from scipy.optimize import nnls

from chc import (
    BarrierConstraint,
    CausalPlan,
    QuadraticCost,
    SupportModel,
    causal_plan,
    certify_safety,
)
from chc.control import LinearConstraint
from chc.cost import total_cost
from chc.dynamics import DampedOscillator, HybridDynamics, LinearDynamics
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


def _ceiling_rows(flat: Array, gamma: float, ceiling: float) -> Array:
    """A velocity ceiling ``ceiling - x[1]``'s condition at ``gamma``, as :func:`_floor_rows`."""
    spread = confounding_robust_inflation(1.0, 0.0, gamma)
    us = flat.reshape(HORIZON, 1)
    xs = rollout(OSCILLATOR, X0, us, DT)[:-1]
    drift = OMEGA**2 * xs[:, 0] + 2.0 * ZETA * OMEGA * xs[:, 1]
    required = -ALPHA * (ceiling - xs[:, 1])
    return jnp.concatenate(
        [
            drift - (1.0 - spread) * us[:, 0] - required,
            drift - (1.0 + spread) * us[:, 0] - required,
        ]
    )


def _exact_floor_optimum(
    gamma: float,
    floor: float = FLOOR,
    penalty: Callable[[Array, Array], Array] | None = None,
    ceiling: float | None = None,
) -> tuple[np.ndarray, float]:
    """The exact optimum under a velocity floor's condition at ``gamma``, and its objective.

    ``penalty(xs, us)`` joins the task cost, and must stay quadratic in the actions -- as a support
    model's is on this linear plant -- for the Hessian taken at zero to be the program's own. A
    ``ceiling`` adds its own condition to the floor's.
    """

    def value(flat: Array) -> Array:
        us = flat.reshape(HORIZON, 1)
        task = total_cost(OSCILLATOR, X0, us, DT, COST)
        if penalty is None:
            return task
        return task + penalty(rollout(OSCILLATOR, X0, us, DT)[:-1], us)

    def condition(flat: Array) -> Array:
        rows = _floor_rows(flat, gamma, floor)
        if ceiling is None:
            return rows
        return jnp.concatenate([rows, _ceiling_rows(flat, gamma, ceiling)])

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


@pytest.mark.parametrize("unit", [1.0, 1e-6])
@pytest.mark.parametrize("gamma", [1.0, 2.0])
def test_a_band_holds_each_bound_s_condition_and_lands_on_the_exact_optimum(
    gamma: float, unit: float, caplog: pytest.LogCaptureFixture
) -> None:
    """Under the floor and a ceiling ``0.1 - v`` the velocity falls through the band's midpoint, and
    the optimum binds the ceiling's condition on steps where the floor is the nearer bound, as it
    brakes: the least of the two margins asks nothing of the ceiling there. Each barrier of the
    tuple holds its own condition, and the log counts a step where any of them binds.

    The ceiling in units a millionth of the floor's is held as in its own. Read in its own units,
    the rounds' one back-off, a millionth of the floor's terms, was as large as the ceiling's own
    terms, and the rounds stopped short with the ceiling held on 16 or 17 steps of 20."""
    ceiling = 0.1

    def velocity_ceiling(x: Array) -> Array:
        return unit * (ceiling - x[1])

    with caplog.at_level(logging.INFO, logger="chc.plan"):
        plan = causal_plan(
            OSCILLATOR,
            X0,
            COST,
            DT,
            HORIZON,
            -U_MAX,
            U_MAX,
            barrier=BarrierConstraint(
                (velocity_ceiling, _velocity_floor), alpha=ALPHA, gamma=gamma
            ),
        )
    actions, optimum = _exact_floor_optimum(gamma, ceiling=ceiling)
    floor_rows = np.asarray(_floor_rows(jnp.asarray(actions), gamma, FLOOR))
    rows = np.asarray(_ceiling_rows(jnp.asarray(actions), gamma, ceiling))
    slack = np.minimum(
        np.minimum(floor_rows[:HORIZON], floor_rows[HORIZON:]),
        np.minimum(rows[:HORIZON], rows[HORIZON:]),
    )
    binds = np.minimum(rows[:HORIZON], rows[HORIZON:]) <= 1e-9
    speed = np.asarray(rollout(OSCILLATOR, X0, jnp.asarray(actions).reshape(-1, 1), DT))[:-1, 1]
    assert np.any(binds & (speed + FLOOR < ceiling - speed))
    [held] = [r for r in caplog.records if getattr(r, "chc_event", None) == "barrier"]
    assert np.min(slack[slack > 1e-9]) > 1e-3  # the active set is unambiguous here
    assert getattr(held, "active_steps", None) == int(np.sum(slack <= 1e-9))

    assert plan.solver_status == "converged"
    assert plan.safety is not None
    assert plan.safety.certified_steps == HORIZON
    for barrier in (_velocity_floor, velocity_ceiling):
        own = certify_safety(plan, OSCILLATOR, barrier, DT, alpha=ALPHA, gamma=gamma)
        assert own.certified_steps == HORIZON
    assert plan.task_cost >= optimum * (1.0 - 1e-12)
    assert (plan.task_cost - optimum) / optimum < GAP


@pytest.mark.parametrize("unit", [0.1, 10.0])
def test_the_held_condition_lands_on_the_exact_optimum_in_any_units_of_the_force(
    unit: float,
) -> None:
    """The force in ``unit`` times its own units, its box and its price with it: the plan is the
    same one, read in those units. The radius was ``(gamma-1)/(gamma+1)`` in the channel's units,
    ``1/unit`` here, so tenths held the floor at a tenth of the radius the audit prices and tens at
    ten times it."""
    drift = jnp.array([[0.0, 1.0], [-(OMEGA**2), -2.0 * ZETA * OMEGA]])
    model = LinearDynamics(drift, jnp.array([[0.0], [1.0 / unit]]))
    cost = QuadraticCost(Q=COST.Q, R=COST.R / unit**2, Qf=COST.Qf, x_target=COST.x_target)
    plan = causal_plan(
        model,
        X0,
        cost,
        DT,
        HORIZON,
        -U_MAX * unit,
        U_MAX * unit,
        barrier=BarrierConstraint(_velocity_floor, alpha=ALPHA, gamma=2.0),
    )
    _, optimum = _exact_floor_optimum(2.0)

    assert plan.safety is not None
    assert plan.safety.certified_steps == HORIZON
    assert plan.task_cost >= optimum * (1.0 - 1e-12)
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


def test_a_barrier_given_twice_counts_each_step_once(caplog: pytest.LogCaptureFixture) -> None:
    """Both columns of a repeated barrier bind on the same steps: the log counts steps, not the
    pairs of a step and a barrier, so it reads the optimum's 13, as the barrier given once does."""
    with caplog.at_level(logging.INFO, logger="chc.plan"):
        plan = causal_plan(
            OSCILLATOR,
            X0,
            COST,
            DT,
            HORIZON,
            -U_MAX,
            U_MAX,
            barrier=BarrierConstraint((_velocity_floor, _velocity_floor), alpha=ALPHA),
        )
    [held] = [r for r in caplog.records if getattr(r, "chc_event", None) == "barrier"]
    assert getattr(held, "active_steps", None) == 13
    assert plan.safety is not None
    assert plan.safety.certified_steps == HORIZON


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


@pytest.mark.parametrize("pinned", [0, 1])
@pytest.mark.parametrize("scale", [1.0, 1e6])
def test_rounds_from_a_plan_the_rows_and_the_box_pin_to_one_point_stay_on_it(
    pinned: int, scale: float
) -> None:
    """``3 u1 = u2`` at every step, with one lever's box pinned at zero, leaves the zero plan alone
    feasible, and the projection holds the row only to its tolerance; a barrier no plan clears
    sends it to the rounds. Their descent took a trial that moved the plan by that tolerance alone
    as a step where it lowered the penalised cost by more than ``1e-9``, and as the penalty grew
    the plan walked 1.9e-9 to 3.1e-5 off zero, across the row. A move only across the constraints
    held at both of its ends is not a step."""
    horizon = 3
    cost = QuadraticCost(
        Q=scale * jnp.eye(1),
        R=scale * jnp.diag(jnp.array([0.01, 1.0])),
        Qf=scale * jnp.eye(1),
        x_target=jnp.array([1.0]),
    )
    tie = LinearConstraint(np.kron(np.eye(horizon), [[3.0, -1.0]]) / np.sqrt(10.0), 0.0, 0.0)
    lo, hi = np.full((horizon, 2), -2.0), np.full((horizon, 2), 2.0)
    lo[:, pinned] = hi[:, pinned] = 0.0
    plan = causal_plan(
        LinearDynamics(jnp.array([[-0.5]]), jnp.array([[0.8, 0.1]])),
        jnp.ones(1),
        cost,
        DT,
        horizon,
        lo,
        hi,
        constraints=(tie,),
        barrier=BarrierConstraint(lambda x: x[0] - 2.0, alpha=1.0),
    )
    assert plan.solver_status == "max_iterations"  # no plan clears it, and none is claimed to
    assert np.asarray(plan.actions).tolist() == [[0.0, 0.0]] * horizon


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
    [
        ({"gamma": 0.9}, "gamma is a marginal sensitivity model level"),
        ({"gamma": math.nan}, "gamma is a marginal sensitivity model level"),
        ({"gamma": math.inf}, "gamma is a marginal sensitivity model level"),
        ({"cvar_gap": 0.0}, "cvar_gap must be"),
        ({"cvar_gap": math.nan}, "cvar_gap must be"),
        ({"cvar_gap": math.inf}, "cvar_gap must be"),
    ],
)
def test_a_barrier_constraint_refuses_what_certify_safety_could_not_price(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BarrierConstraint(_velocity_floor, **kwargs)


def test_a_tuple_is_audited_barrier_by_barrier_and_read_as_one() -> None:
    """Each step reads every barrier: certified where all are, its value the least, its pair the
    one with the least room, its ceiling the least; the radius is the widest, here the ceiling's,
    whose gradient is twice the floor's."""
    plan = causal_plan(OSCILLATOR, X0, COST, DT, HORIZON, -U_MAX, U_MAX)

    def velocity_ceiling(x: Array) -> Array:
        return 2.0 * (0.1 - x[1])

    barriers = (_velocity_floor, velocity_ceiling)
    kwargs = {"alpha": ALPHA, "gamma": 2.0, "u_max": U_MAX}
    floor, ceiling = (certify_safety(plan, OSCILLATOR, b, DT, **kwargs) for b in barriers)
    both = certify_safety(plan, OSCILLATOR, barriers, DT, **kwargs)

    held = np.asarray(floor.planned_certified) & np.asarray(ceiling.planned_certified)
    assert held.any()
    assert not held.all()
    np.testing.assert_array_equal(np.asarray(both.planned_certified), held)
    assert both.certified_steps == min(floor.certified_steps, ceiling.certified_steps)
    np.testing.assert_array_equal(
        np.asarray(both.barrier_values),
        np.minimum(np.asarray(floor.barrier_values), np.asarray(ceiling.barrier_values)),
    )
    rooms = [np.asarray(a.guaranteed_derivative - a.required) for a in (floor, ceiling)]
    pick = np.where(rooms[0] <= rooms[1], 0, 1)
    assert 0 < pick.sum() < HORIZON  # each barrier has the least room somewhere
    for field in ("guaranteed_derivative", "required"):
        values = np.stack([np.asarray(getattr(a, field)) for a in (floor, ceiling)])
        np.testing.assert_array_equal(
            np.asarray(getattr(both, field)), values[pick, np.arange(HORIZON)]
        )
    assert both.step_gamma_star == tuple(
        min(a, b) for a, b in zip(floor.step_gamma_star, ceiling.step_gamma_star, strict=True)
    )
    assert both.gamma_star == min(floor.gamma_star, ceiling.gamma_star)
    assert both.radius == ceiling.radius == 2.0 * floor.radius > 0.0


def test_an_empty_tuple_of_barriers_is_refused() -> None:
    with pytest.raises(ValueError, match="empty tuple of barriers"):
        BarrierConstraint(())
    plan = causal_plan(OSCILLATOR, X0, COST, DT, HORIZON, -U_MAX, U_MAX, steps=1)
    with pytest.raises(ValueError, match="empty tuple of barriers"):
        certify_safety(plan, OSCILLATOR, (), DT, u_max=U_MAX)


def test_a_barrier_needs_a_box_that_admits_an_action() -> None:
    with pytest.raises(ValueError, match="admits a nonzero action"):
        causal_plan(
            OSCILLATOR, X0, COST, DT, HORIZON, 0.0, 0.0, barrier=BarrierConstraint(_velocity_floor)
        )


# ---- each lever read in units of its own largest action ----

# the oscillator pushed by two forces, the second at half the first's gain, in boxes of 5 and 1
TWO_DRIFT = jnp.array([[0.0, 1.0], [-(OMEGA**2), -2.0 * ZETA * OMEGA]])
TWO_CHANNEL = jnp.array([[0.0, 0.0], [1.0, 0.5]])
TWO_REACH = np.array([U_MAX, 1.0])


def _two_forces(units: tuple[float, float] = (1.0, 1.0)) -> tuple[LinearDynamics, QuadraticCost]:
    """The two-force oscillator with force ``j`` in ``units[j]`` times its own units, its price
    with it."""
    scale = jnp.asarray(units)
    cost = QuadraticCost(
        Q=COST.Q,
        R=jnp.diag(jnp.array([0.01, 0.02]) / scale**2),
        Qf=COST.Qf,
        x_target=COST.x_target,
    )
    return LinearDynamics(TWO_DRIFT, TWO_CHANNEL / scale), cost


def _same_audit(one: object, other: object) -> None:
    """Every field of two audits, bit for bit, nan where both are."""
    for name in ("barrier_values", "guaranteed_derivative", "required", "planned_certified"):
        np.testing.assert_array_equal(
            np.asarray(getattr(one, name)), np.asarray(getattr(other, name))
        )
    for name in ("certified_steps", "gamma_star", "step_gamma_star", "radius"):
        np.testing.assert_array_equal(
            np.asarray(getattr(one, name)), np.asarray(getattr(other, name))
        )


def test_one_lever_s_largest_action_reads_the_ball_it_read_bit_for_bit() -> None:
    """One lever's largest action is the whole action set's: the audit and the held condition read
    the ball of that radius they read before, to the bit, held or not; so do two levers whose
    largest actions are equal, and a free side, which leaves no units to read."""
    plan = causal_plan(OSCILLATOR, X0, COST, DT, HORIZON, -U_MAX, U_MAX)
    kwargs = {"alpha": ALPHA, "gamma": 2.0}
    ball = certify_safety(plan, OSCILLATOR, _velocity_floor, DT, u_max=U_MAX, **kwargs)
    assert ball.certified_steps < HORIZON  # the free plan breaks the floor
    _same_audit(
        certify_safety(plan, OSCILLATOR, _velocity_floor, DT, u_max=(U_MAX,), **kwargs), ball
    )

    held = [
        causal_plan(
            OSCILLATOR,
            X0,
            COST,
            DT,
            HORIZON,
            -U_MAX,
            U_MAX,
            barrier=BarrierConstraint(_velocity_floor, u_max=u_max, **kwargs),
        )
        for u_max in (None, (U_MAX,))
    ]
    np.testing.assert_array_equal(np.asarray(held[1].actions), np.asarray(held[0].actions))
    _same_audit(held[1].safety, held[0].safety)

    model, cost = _two_forces()
    both = causal_plan(model, X0, cost, DT, HORIZON, -U_MAX, U_MAX)
    ball = certify_safety(both, model, _velocity_floor, DT, u_max=U_MAX, **kwargs)
    _same_audit(
        certify_safety(both, model, _velocity_floor, DT, u_max=(U_MAX, U_MAX), **kwargs), ball
    )
    free = certify_safety(both, model, _velocity_floor, DT, u_max=math.inf, **kwargs)
    _same_audit(
        certify_safety(both, model, _velocity_floor, DT, u_max=(math.inf, 1.0), **kwargs), free
    )


@pytest.mark.parametrize("units", [(1e3, 1.0), (1.0, 1e-3), (1e-3, 1e3)])
def test_the_audit_reads_the_same_in_any_units_of_each_lever(units: tuple[float, float]) -> None:
    """Each force in units of its own, its box with it: the same plan, read in those units, clears
    the same steps by the same guaranteed derivative, and every step's ceiling is the same. With
    one number for both forces, the largest action of either, the ball and the radius read their
    units as one: in ``(1e3, 1)`` the plan, which clears 14 steps of 20, cleared none, and
    ``gamma*`` read 2779 where it reads 4.59. Held in the solve, whose rounds step each force in
    units of its largest action, every step clears at one cost in every units; stepped in the
    caller's, ``(1e3, 1)`` held none."""
    kwargs = {"alpha": ALPHA, "gamma": 2.0}
    model, cost = _two_forces()
    plan = causal_plan(model, X0, cost, DT, HORIZON, -TWO_REACH, TWO_REACH)
    one = certify_safety(plan, model, _velocity_floor, DT, u_max=TWO_REACH, **kwargs)
    assert one.certified_steps < HORIZON  # the free plan breaks the floor
    assert int(np.sum(np.asarray(one.planned_certified))) == 14
    assert any(1.0 < value < math.inf for value in one.step_gamma_star)

    scale = np.asarray(units)
    other_model, _ = _two_forces(units)
    moved = replace(plan, actions=plan.actions * jnp.asarray(scale))
    other = certify_safety(
        moved, other_model, _velocity_floor, DT, u_max=TWO_REACH * scale, **kwargs
    )
    assert other.certified_steps == one.certified_steps
    np.testing.assert_array_equal(
        np.asarray(other.planned_certified), np.asarray(one.planned_certified)
    )
    np.testing.assert_allclose(
        np.asarray(other.guaranteed_derivative),
        np.asarray(one.guaranteed_derivative),
        rtol=1e-12,
        atol=1e-15,
    )
    np.testing.assert_allclose(other.step_gamma_star, one.step_gamma_star, rtol=1e-9, atol=0.0)

    held = [
        causal_plan(
            forces,
            X0,
            price,
            DT,
            HORIZON,
            -TWO_REACH * s,
            TWO_REACH * s,
            barrier=BarrierConstraint(_velocity_floor, u_max=tuple(TWO_REACH * s), **kwargs),
        )
        for (forces, price), s in ((_two_forces(), np.ones(2)), (_two_forces(units), scale))
    ]
    for plan in held:
        assert plan.safety is not None
        assert plan.safety.certified_steps == HORIZON
        assert plan.solver_status == "converged"
    assert held[1].task_cost == pytest.approx(held[0].task_cost, rel=1e-5, abs=0.0)


def test_gamma_star_is_where_the_best_action_inside_the_box_just_clears() -> None:
    """At a step's ceiling the best action of the ellipsoid whose semi-axes are each lever's largest
    action, ``U D (cos t, sin t)``, sought over 200 001 angles, meets the condition with no room,
    against the radius on ``(B_hat - B) D`` sized by ``||B_hat D||_2``. The ball of radius 5 that
    one number read had its best action push the second force to 2.2, past its box of 1."""
    model, cost = _two_forces()
    plan = causal_plan(model, X0, cost, DT, HORIZON, -TWO_REACH, TWO_REACH)
    audit = certify_safety(
        plan, model, _velocity_floor, DT, alpha=ALPHA, gamma=2.0, u_max=TWO_REACH
    )
    angles = np.linspace(0.0, 2.0 * np.pi, 200_001)
    ellipse = TWO_REACH[:, None] * np.stack([np.cos(angles), np.sin(angles)])  # inside the box
    channel = np.asarray(TWO_CHANNEL)
    size = float(np.linalg.norm(channel * (TWO_REACH / U_MAX), 2))
    checked = 0
    for state, ceiling in zip(np.asarray(plan.trajectory[:-1]), audit.step_gamma_star, strict=True):
        if not 1.0 < ceiling < math.inf:
            continue
        radius = (ceiling - 1.0) / (ceiling + 1.0) * size  # ||grad h|| = 1
        drift = float((np.asarray(TWO_DRIFT) @ state)[1])
        best = drift + float(np.max(channel[1] @ ellipse)) - radius * U_MAX
        assert best == pytest.approx(-ALPHA * (state[1] + FLOOR), rel=1e-8, abs=1e-12)
        checked += 1
    assert checked > 0
    w = channel[1]
    assert abs(U_MAX * w[1] / np.linalg.norm(w)) > TWO_REACH[1]


@pytest.mark.parametrize(
    ("u_max", "message"),
    [
        ((-1.0, 1.0), "no set of actions' sizes"),
        ((math.nan, 1.0), "no set of actions' sizes"),
        ((0.0, 0.0), "no set of actions' sizes"),
        ((U_MAX,), "one per lever"),
        ((0.0, 1.0), "largest action of 0"),
    ],
)
def test_each_lever_s_largest_action_must_be_one(u_max: tuple[float, ...], message: str) -> None:
    """A negative or nan size, none positive, too few, or 0 at a lever the plan moves: no ellipsoid
    holds the plan's actions, and the radius would read nothing."""
    model, cost = _two_forces()
    plan = causal_plan(model, X0, cost, DT, HORIZON, -TWO_REACH, TWO_REACH)
    with pytest.raises(ValueError, match=message):
        certify_safety(plan, model, _velocity_floor, DT, u_max=u_max)
    if message != "one per lever" and message != "largest action of 0":
        with pytest.raises(ValueError, match=message):
            BarrierConstraint(_velocity_floor, u_max=u_max)
    else:
        with pytest.raises(ValueError, match=message):
            causal_plan(
                model,
                X0,
                cost,
                DT,
                HORIZON,
                -TWO_REACH,
                TWO_REACH,
                barrier=BarrierConstraint(_velocity_floor, gamma=2.0, u_max=u_max),
            )


def _box_ceilings(
    plan: CausalPlan, lo: np.ndarray, hi: np.ndarray, gamma: float = 2.0
) -> tuple[list[float], np.ndarray]:
    """The two-force plan's audit with the box ``[lo, hi]`` as its action set, and the box's points
    on a grid of 1001 by 1001."""
    model, _ = _two_forces()
    audit = certify_safety(plan, model, _velocity_floor, DT, alpha=ALPHA, gamma=gamma, box=(lo, hi))
    axes = [np.linspace(low, high, 1001) for low, high in zip(lo, hi, strict=True)]
    grid = np.stack([side.ravel() for side in np.meshgrid(*axes)])
    return list(audit.step_gamma_star), grid


@pytest.mark.parametrize(
    ("lo", "hi"),
    [((-1.0, 0.0), (5.0, 1.0)), ((0.5, -1.0), (5.0, -0.2)), ((-5.0, -1.0), (5.0, 1.0))],
)
def test_gamma_star_in_the_box_is_where_the_best_action_of_the_box_just_clears(
    lo: tuple[float, float], hi: tuple[float, float]
) -> None:
    """Given the box, each step's ceiling is where the best action of the box itself, sought over a
    grid of a million of its points, meets the condition with no room, against the radius on
    ``(B_hat - B) D`` sized by ``||B_hat D||_2``, ``D`` each lever's largest action in the box over
    the largest. On the symmetric box ``[-5, 5] x [-1, 1]`` the box's corners lie outside the
    ellipsoid, and its ceilings are at least the ellipsoid's."""
    model, cost = _two_forces()
    box_lo, box_hi = np.asarray(lo), np.asarray(hi)
    plan = causal_plan(model, X0, cost, DT, HORIZON, jnp.asarray(box_lo), jnp.asarray(box_hi))
    ceilings, grid = _box_ceilings(plan, box_lo, box_hi)
    reach = np.maximum(np.abs(box_lo), np.abs(box_hi))
    shaped = np.asarray(TWO_CHANNEL) * (reach / reach.max())
    size = float(np.linalg.norm(shaped, 2))
    spread = np.linalg.norm(grid * (reach.max() / reach)[:, None], axis=0)
    channel = np.asarray(TWO_CHANNEL)[1]
    step = float(np.max((box_hi - box_lo) / 1000.0))
    checked = 0
    for state, ceiling in zip(np.asarray(plan.trajectory[:-1]), ceilings, strict=True):
        drift = float((np.asarray(TWO_DRIFT) @ state)[1])
        required = -ALPHA * (state[1] + FLOOR)
        if math.isnan(ceiling):  # no action of the box clears the step even at gamma = 1
            assert drift + float(np.max(channel @ grid)) < required
            continue
        if not 1.0 < ceiling < math.inf:
            continue
        radius = (ceiling - 1.0) / (ceiling + 1.0) * size  # ||grad h|| = 1
        best = float(np.max(drift + channel @ grid - radius * spread))
        assert best == pytest.approx(
            required, abs=3.0 * step * (1.0 + float(np.abs(channel).max()))
        )
        assert best <= required + 1e-12
        checked += 1
    assert checked > 0
    if box_lo[0] == -box_hi[0] and box_lo[1] == -box_hi[1]:
        ellipsoid = certify_safety(
            plan, model, _velocity_floor, DT, alpha=ALPHA, gamma=2.0, u_max=tuple(reach)
        )
        for inside, outer in zip(ellipsoid.step_gamma_star, ceilings, strict=True):
            assert outer >= inside or math.isnan(inside)


# a lever that only pushes the state up, towards its upper bound, under a drift that crosses it
_PUSH = LinearDynamics(jnp.array([[0.3]]), jnp.array([[0.8]]))
_PUSH_COST = QuadraticCost(
    Q=jnp.eye(1), R=jnp.array([[0.01]]), Qf=jnp.eye(1), x_target=jnp.array([3.0])
)


def _below(x: Array) -> Array:
    return 2.05 - x[0]


@pytest.mark.parametrize(("lo", "hi"), [(0.0, 2.0), (1.0, 1.0), (0.5, 2.0)])
def test_a_box_whose_every_action_pushes_into_the_bound_certifies_no_level(
    lo: float, hi: float
) -> None:
    """In ``[0, 2]``, at its level 1 held, or in ``[0.5, 2]``, every action the box allows pushes
    the state into the bound the drift is already crossing, and no level of sensitivity is
    certified at the first step: ``gamma_star`` is nan. The ellipsoid of semi-axis 2 held ``-2``,
    and read 4.82 there in ``[0, 2]`` and ``[0.5, 2]``; at the held level 1 it held ``-1``, and
    read 1.91."""
    plan = causal_plan(_PUSH, jnp.array([2.0]), _PUSH_COST, 0.2, 4, lo, hi)
    reach = (max(abs(lo), abs(hi)),)
    boxed = certify_safety(plan, _PUSH, _below, 0.2, gamma=1.5, box=((lo,), (hi,)))
    assert math.isnan(boxed.step_gamma_star[0])
    assert math.isnan(boxed.gamma_star)
    ellipsoid = certify_safety(plan, _PUSH, _below, 0.2, gamma=1.5, u_max=reach)
    assert 1.0 < ellipsoid.step_gamma_star[0] < math.inf
    held = causal_plan(
        _PUSH,
        jnp.array([2.0]),
        _PUSH_COST,
        0.2,
        4,
        lo,
        hi,
        barrier=BarrierConstraint(_below, gamma=1.5, box=((lo,), (hi,))),
    )
    assert held.safety is not None
    assert math.isnan(held.safety.gamma_star)


def test_a_box_that_holds_zero_and_a_safe_drift_certifies_every_level_and_one_that_does_not_less():
    """Where the drift alone clears a step and the box holds zero, every level is certified, as
    the ellipsoid reads it; where the box holds no zero, every admissible action is moved by the
    effect's error, and the step's ceiling is finite: ``(<w, u> - D) / |u|`` at the box's best
    action, here ``u = 0.5``."""
    safe = LinearDynamics(jnp.array([[-0.3]]), jnp.array([[0.8]]))
    plan = causal_plan(safe, jnp.array([1.9]), _PUSH_COST, 0.2, 2, 0.5, 2.0)
    held_zero = certify_safety(plan, safe, _below, 0.2, gamma=1.5, box=((-1.0,), (2.0,)))
    assert held_zero.step_gamma_star[0] == math.inf
    no_zero = certify_safety(plan, safe, _below, 0.2, gamma=1.5, box=((0.5,), (2.0,)))
    x = float(plan.trajectory[0, 0])
    deficit = -(2.05 - x) - 0.3 * x  # -alpha h - grad h . A x, grad h = -1
    assert deficit < 0.0
    threshold = (-0.8 * 0.5 - deficit) / 0.5  # D = 1 for one lever
    share = threshold / 0.8  # over ||B_hat|| ||grad h||, against the default gap of 1
    assert 0.0 < share < 1.0
    expected = (1.0 + share) / (1.0 - share)
    assert no_zero.step_gamma_star[0] == pytest.approx(expected, rel=1e-12, abs=0.0)
    assert math.isfinite(no_zero.step_gamma_star[0])


def test_one_lever_s_symmetric_box_reads_the_closed_form() -> None:
    """One lever in ``[-U, U]``: the box is the ball of radius ``U``, and each step's ceiling is the
    closed form's, to rounding."""
    plan = causal_plan(OSCILLATOR, X0, COST, DT, HORIZON, -U_MAX, U_MAX)
    kwargs = {"alpha": ALPHA, "gamma": 2.0}
    ball = certify_safety(plan, OSCILLATOR, _velocity_floor, DT, u_max=U_MAX, **kwargs)
    boxed = certify_safety(
        plan, OSCILLATOR, _velocity_floor, DT, box=((-U_MAX,), (U_MAX,)), **kwargs
    )
    assert any(1.0 < value < math.inf for value in ball.step_gamma_star)
    np.testing.assert_allclose(boxed.step_gamma_star, ball.step_gamma_star, rtol=1e-12, atol=0.0)
    assert boxed.certified_steps == ball.certified_steps


def test_the_box_threshold_is_the_supremum_over_the_box() -> None:
    """``_box_threshold`` against the largest ratio ``(<w, u> - D) / ||u||`` over 40 000 random
    points and a grid of each of 600 boxes of one to three levers, some holding zero, some not,
    some with a lever held at one level: never below it, and above it by less than the grid's
    resolution allows."""
    from chc.barrier import _box_threshold

    rng = np.random.default_rng(7)
    for trial in range(600):
        m = int(rng.integers(1, 4))
        lo = 2.0 * rng.normal(size=m)
        hi = lo + rng.exponential(2.0, size=m)
        if trial % 3 == 0:
            lo, hi = -np.abs(lo), np.abs(hi)
        if trial % 7 == 0:
            hi[0] = lo[0]
        w = rng.normal(size=m)
        w[rng.random(m) < 0.1] = 0.0
        drift, alpha_h = float(rng.normal()), float(rng.normal())
        deficit = -alpha_h - drift
        got = _box_threshold(drift, w, lo, hi, alpha_h)
        points = lo + (hi - lo) * rng.random((40_000, m))
        axes = [np.linspace(a, b, 60 if m < 3 else 25) for a, b in zip(lo, hi, strict=True)]
        points = np.vstack([points, np.stack([g.ravel() for g in np.meshgrid(*axes)], 1)])
        size = np.linalg.norm(points, axis=1)
        brute = float(np.max((points[size > 0] @ w - deficit) / size[size > 0]))
        if np.all(lo <= 0.0) and np.all(hi >= 0.0) and deficit <= 0.0:
            assert got == math.inf
        elif brute < 0.0:
            assert math.isnan(got) or got < 1e-6
        else:
            assert got >= brute - 1e-12 * max(1.0, abs(brute))
            assert got <= brute + 0.05 * max(1.0, abs(brute))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"box": ((-1.0,), (1.0,))}, "one per lever"),
        ({"box": ((1.0, 0.0), (0.0, 1.0))}, "a side above the other"),
        ({"box": ((-1.0, -1.0), (1.0, 1.0)), "u_max": 1.0}, "one number"),
        ({"box": ((-1.0, -1.0), (1.0, 1.0)), "u_max": (1.0, 0.0)}, "largest action of 0"),
    ],
)
def test_a_box_must_be_one_pair_per_lever(kwargs: dict[str, Any], message: str) -> None:
    """A box with too few levers, a side above the other, one number for every lever's units, or a
    lever free to move whose largest action is 0: no set of actions the audit can read."""
    model, cost = _two_forces()
    plan = causal_plan(model, X0, cost, DT, HORIZON, 0.0, 0.0)
    with pytest.raises(ValueError, match=message):
        certify_safety(plan, model, _velocity_floor, DT, **kwargs)


@pytest.mark.parametrize("hold", [False, True])
def test_prescribe_seeks_gamma_star_in_its_box(hold: bool) -> None:
    """A lever logged in ``[0, 2]`` that pushes the state up, and a drift already crossing the
    state's upper bound at the start: no action the box allows holds the first step at any level,
    and ``gamma_star`` is nan, held or not. Up to 0.18.0 the audit's ellipsoid held ``u = -2`` and
    read 2.02, as it reads in ``[-2, 2]``, where the box holds that action."""
    from chc import CausalGraph, Constraint, Lever, Panel, Target, prescribe

    rng = np.random.default_rng(0)
    steps = 800
    x = np.zeros(steps + 1)
    u = np.zeros(steps)
    z = rng.normal(size=steps)
    for t in range(steps):
        u[t] = np.clip(0.5 * z[t] + 0.6 * rng.normal() + 1.0, 0.0, 2.0)
        rate = -0.5 * x[t] + 0.8 * u[t] + 0.5 * z[t] + 1.4
        x[t + 1] = x[t] + 0.2 * rate + 0.05 * rng.normal()
    frame = {"unit": np.zeros(steps, int), "time": np.arange(steps), "u": u, "z": z, "x": x[:steps]}
    panel = Panel.from_frame(frame, unit="unit", time="time", seed=0)
    ceilings = {}
    for lo in (0.0, -2.0):
        result = prescribe(
            panel,
            adjustment=CausalGraph.from_edges([("z", "u"), ("z", "x"), ("u", "x")]),
            levers=[Lever("u", lo=lo, hi=2.0)],
            target=Target("x", value=3.0),
            constraints=[Constraint("x", hi=2.05)],
            hold_constraints=hold,
            horizon=6,
            dt=0.2,
            tolerance=0.5,
            x0=[2.0],
            gamma=1.5,
        )
        ceilings[lo] = result.certificate.gamma_star
    assert ceilings[0.0] is not None
    assert math.isnan(ceilings[0.0])
    assert ceilings[-2.0] is not None
    assert 1.0 < ceilings[-2.0] < math.inf


def test_a_box_without_end_reads_the_ratio_s_limit_along_its_last_piece() -> None:
    """One lever in ``[0, inf)`` whose channel along the barrier is ``w = 0.8`` and a deficit
    ``D = 0.5``: ``(w u - D) / u`` rises towards ``w`` and never reaches it, and the threshold is
    that supremum. In ``[0, 10]`` it is the ratio at ``u = 10``."""
    from chc.barrier import _box_threshold

    channel = np.array([0.8])
    assert _box_threshold(
        -0.5, channel, np.array([0.0]), np.array([math.inf]), 0.0
    ) == pytest.approx(0.8, rel=1e-15, abs=0.0)
    assert _box_threshold(-0.5, channel, np.array([0.0]), np.array([10.0]), 0.0) == pytest.approx(
        (0.8 * 10.0 - 0.5) / 10.0, rel=1e-15, abs=0.0
    )
    two = _box_threshold(
        -0.5, np.array([0.8, 0.6]), np.array([0.0, 0.0]), np.array([math.inf, math.inf]), 0.0
    )
    assert two == pytest.approx(1.0, rel=1e-15, abs=0.0)
