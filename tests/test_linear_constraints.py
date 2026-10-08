"""Linear constraints on the action sequence: every iterate is feasible, and the answer is optimal.

The oracle shares nothing with the solver. The plant is linear and the cost quadratic, so the
problem is a strictly convex QP in the actions (RK4 on a linear plant is linear in the actions). It
is solved exactly by a Cholesky change of variables and Lawson & Hanson's least-distance program --
one NNLS call, an active-set method with finite termination -- against the solver's projected
gradient with Dykstra's alternating projections.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from scipy.optimize import nnls

import chc.control
from chc import LinearConstraint, QuadraticCost, SupportModel, causal_plan
from chc.control import (
    _backtrack,
    _constraint_blocks,
    _dykstra,
    _no_duals,
    _polish,
    projected_gradient_control,
    projected_gradient_solve,
)
from chc.cost import total_cost
from chc.dynamics import DampedOscillator, LinearDynamics
from chc.support import pessimistic_solve

DT = 0.1
HORIZON = 30
U_MAX = 5.0


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


def _rows(lo, hi, matrix, lower, upper) -> tuple[np.ndarray, np.ndarray]:
    """The feasible set as ``G u <= h``, keeping only the sides that are present."""
    n = matrix.shape[1]
    has_upper, has_lower = np.isfinite(upper), np.isfinite(lower)
    g = np.vstack([matrix[has_upper], -matrix[has_lower], np.eye(n), -np.eye(n)])
    h = np.concatenate([upper[has_upper], -lower[has_lower], hi, -lo])
    return g, h


def _exact_projection(y, lo, hi, matrix, lower, upper) -> np.ndarray:
    g, h = _rows(lo, hi, matrix, lower, upper)
    return y + _least_distance(-g, g @ y - h)


def _exact_optimum(hessian, gradient, lo, hi, matrix, lower, upper) -> np.ndarray:
    g, h = _rows(lo, hi, matrix, lower, upper)
    free = -np.linalg.solve(hessian, gradient)
    back = np.linalg.inv(np.linalg.cholesky(hessian)).T  # u = free + back @ w, |w|^2 = 2 J + c
    return free + back @ _least_distance(-g @ back, g @ free - h)


def _random_polytope(rng: np.random.Generator, rows: int, size: int):
    """Random slabs, some one-sided, around a point inside the unit box; and a point to project."""
    matrix = rng.normal(size=(rows, size))
    level = matrix @ rng.uniform(-0.5, 0.5, size)
    lower = np.where(rng.random(rows) < 0.3, -np.inf, level - rng.uniform(0.05, 1.0, rows))
    upper = np.where(rng.random(rows) < 0.3, np.inf, level + rng.uniform(0.05, 1.0, rows))
    return matrix, lower, upper, -np.ones(size), np.ones(size), rng.normal(0.0, 3.0, size)


def _problem():
    dyn = DampedOscillator(omega=1.0, zeta=0.1)
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.0])),
        R=jnp.array([[0.01]]),
        Qf=jnp.diag(jnp.array([10.0, 1.0])),
        x_target=jnp.zeros(2),
    )
    x0 = jnp.array([1.0, 0.0])

    def objective(flat):
        return total_cost(dyn, x0, flat.reshape(HORIZON, 1), DT, cost)

    zero = jnp.zeros(HORIZON)
    hessian = np.asarray(jax.hessian(objective)(zero))
    gradient = np.asarray(jax.grad(objective)(zero))
    return dyn, cost, x0, objective, 0.5 * (hessian + hessian.T), gradient


def _binding_constraints(hessian, gradient):
    """A rate band and a budget, both sized off the box-only optimum so that both bind."""
    lo, hi = np.full(HORIZON, -U_MAX), np.full(HORIZON, U_MAX)
    none = np.zeros((0, HORIZON))
    box_only = _exact_optimum(hessian, gradient, lo, hi, none, np.zeros(0), np.zeros(0))
    rate = LinearConstraint.rate_limit(HORIZON, [0.35 * float(np.max(np.abs(np.diff(box_only))))])
    # The box optimum pushes the position back with negative force, so the budget caps how much
    # of it the whole plan may spend.
    budget = LinearConstraint(np.ones((1, HORIZON)), 0.6 * float(box_only.sum()), np.inf)
    return box_only, (rate, budget), lo, hi


def _stacked(constraints):
    return (
        np.vstack([c.matrix for c in constraints]),
        np.concatenate([c.lower for c in constraints]),
        np.concatenate([c.upper for c in constraints]),
    )


def _violation(u, constraints) -> float:
    matrix, lower, upper = _stacked(constraints)
    level = matrix @ np.asarray(u).ravel()
    return float(np.max(np.maximum(lower - level, level - upper)))


def test_the_constrained_solve_reaches_the_exact_optimum() -> None:
    dyn, cost, x0, objective, hessian, gradient = _problem()
    box_only, constraints, lo, hi = _binding_constraints(hessian, gradient)
    assert _violation(box_only, constraints) > 0.1  # the box optimum breaks both, or this is idle

    exact = _exact_optimum(hessian, gradient, lo, hi, *_stacked(constraints))
    best = float(objective(jnp.asarray(exact)))
    result = projected_gradient_solve(
        dyn, x0, jnp.zeros((HORIZON, 1)), DT, cost, -U_MAX, U_MAX, constraints=constraints
    )

    assert result.status == "converged"
    assert (float(objective(result.actions.ravel())) - best) / best < 1e-6
    assert result.constraint_violation < 1e-9
    assert _violation(result.actions, constraints) == pytest.approx(
        result.constraint_violation, abs=1e-12
    )
    assert float(jnp.max(jnp.abs(result.actions))) <= U_MAX  # the box is exact, not approximate
    assert result.stationarity < 1e-3
    matrix, lower, upper = _stacked(constraints)
    active = np.minimum(matrix @ exact - lower, upper - matrix @ exact) < 1e-8
    assert active.sum() >= 3  # several rows bind at the optimum, not a lucky single one


def test_a_solve_stopped_by_its_budget_still_returns_a_feasible_plan() -> None:
    """The property the design was chosen for: an augmented Lagrangian is feasible only at the end.

    So a solve cut short by its step budget is an executable plan here, and would not be there.
    """
    dyn, cost, x0, _, hessian, gradient = _problem()
    _, constraints, _, _ = _binding_constraints(hessian, gradient)
    result = projected_gradient_solve(
        dyn, x0, jnp.zeros((HORIZON, 1)), DT, cost, -U_MAX, U_MAX, steps=3, constraints=constraints
    )
    assert result.status == "max_iterations"
    assert result.constraint_violation < 1e-9
    assert result.stationarity > 1e-2  # unfinished, and it says so


def test_the_initial_guess_is_projected_rather_than_trusted() -> None:
    dyn, cost, x0, _, hessian, gradient = _problem()
    _, constraints, _, _ = _binding_constraints(hessian, gradient)
    wild = jnp.asarray(np.where(np.arange(HORIZON) % 2, 4.0, -4.0)[:, None])  # breaks the band
    assert _violation(wild, constraints) > 1.0
    us, _ = projected_gradient_control(
        dyn, x0, wild, DT, cost, -U_MAX, U_MAX, steps=1, constraints=constraints
    )
    assert _violation(us, constraints) < 1e-9


def test_no_constraints_and_an_empty_one_are_the_box_solve_exactly() -> None:
    dyn, cost, x0, *_ = _problem()
    guess = jnp.zeros((HORIZON, 1))
    plain = projected_gradient_solve(dyn, x0, guess, DT, cost, -U_MAX, U_MAX, steps=200)
    empty = projected_gradient_solve(
        dyn,
        x0,
        guess,
        DT,
        cost,
        -U_MAX,
        U_MAX,
        steps=200,
        constraints=(LinearConstraint.rate_limit(HORIZON, [np.inf]),),
    )
    assert np.array_equal(np.asarray(plain.actions), np.asarray(empty.actions))
    assert plain.constraint_violation == 0.0


def test_the_pessimistic_solver_holds_the_same_rows() -> None:
    dyn, cost, x0, _, hessian, gradient = _problem()
    _, constraints, _, _ = _binding_constraints(hessian, gradient)
    rng = np.random.default_rng(0)
    support = SupportModel.fit(
        jnp.asarray(rng.normal(size=(200, 2))), jnp.asarray(rng.normal(size=(200, 1)))
    )
    result = pessimistic_solve(
        dyn,
        x0,
        jnp.zeros((HORIZON, 1)),
        DT,
        cost,
        support,
        0.01,
        -U_MAX,
        U_MAX,
        steps=400,
        constraints=constraints,
    )
    assert result.constraint_violation < 1e-9


def test_causal_plan_passes_the_rows_through() -> None:
    dyn, cost, x0, _, hessian, gradient = _problem()
    _, constraints, _, _ = _binding_constraints(hessian, gradient)
    plan = causal_plan(dyn, x0, cost, DT, HORIZON, -U_MAX, U_MAX, constraints=constraints)
    assert _violation(plan.actions, constraints) < 1e-9


def test_single_precision_settles_at_its_own_tolerance() -> None:
    dyn, cost, x0, _, hessian, gradient = _problem()
    _, constraints, _, _ = _binding_constraints(hessian, gradient)
    result = projected_gradient_solve(
        dyn,
        x0,
        jnp.zeros((HORIZON, 1), dtype=jnp.float32),
        DT,
        cost,
        -U_MAX,
        U_MAX,
        steps=300,
        constraints=constraints,
    )
    assert result.actions.dtype == jnp.float32
    assert result.constraint_violation < 1e-4


def test_rate_limit_builds_one_band_per_capped_lever() -> None:
    horizon, caps = 4, [0.5, np.inf, 0.0]
    constraint = LinearConstraint.rate_limit(horizon, caps)
    assert constraint.matrix.shape == (2 * (horizon - 1), horizon * 3)
    schedule = np.array([[0.0, 9.0, 1.0], [0.4, -9.0, 1.0], [0.9, 9.0, 1.0], [0.4, 0.0, 1.0]])
    level = constraint.matrix @ schedule.ravel()
    np.testing.assert_allclose(level, [0.4, 0.5, -0.5, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(constraint.upper, [0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
    assert LinearConstraint.rate_limit(horizon, [np.inf]).matrix.shape == (0, horizon)
    assert LinearConstraint.rate_limit(1, [0.5]).matrix.shape == (0, 1)


@pytest.mark.parametrize(
    ("matrix", "lower", "upper", "match"),
    [
        (np.array([[1.0, 0.0], [0.0, 0.0]]), -1.0, 1.0, "row 1 of the constraint matrix is zero"),
        (np.array([[1.0, 0.0]]), 2.0, 1.0, "row 0 has lower 2.0 above upper 1.0"),
        (np.array([[1.0, 0.0]]), np.nan, 1.0, "NaN"),
        (np.array([[np.inf, 0.0]]), -1.0, 1.0, "non-finite coefficient"),
        (np.array([[1.0, 0.0]]), [0.0, 0.0], 1.0, "one per row"),
    ],
)
def test_a_constraint_that_cannot_mean_anything_is_refused(matrix, lower, upper, match) -> None:
    with pytest.raises(ValueError, match=match):
        LinearConstraint(matrix, lower, upper)


def test_an_empty_feasible_set_is_refused_before_the_solve() -> None:
    dyn, cost, x0, *_ = _problem()
    impossible = LinearConstraint(np.ones((1, HORIZON)), 2 * HORIZON * U_MAX, np.inf)
    with pytest.raises(ValueError, match="no action sequence satisfies"):
        projected_gradient_solve(
            dyn, x0, jnp.zeros((HORIZON, 1)), DT, cost, -U_MAX, U_MAX, constraints=(impossible,)
        )
    too_narrow = LinearConstraint(np.ones((1, 7)), -1.0, 1.0)
    with pytest.raises(ValueError, match="flatten to 30"):
        projected_gradient_solve(
            dyn, x0, jnp.zeros((HORIZON, 1)), DT, cost, -U_MAX, U_MAX, constraints=(too_narrow,)
        )


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    shape=st.sampled_from([(3, 4), (6, 5), (9, 6), (12, 8)]),
)
# Vertices where a constraint Dykstra had not yet released made the polished set over-determined:
# plain Dykstra ran out its sweeps on each, 4.6e-4, 1.9e-3 and 2.7e-3 from the projection.
@example(seed=66, shape=(3, 4))
@example(seed=639, shape=(9, 6))
@example(seed=567, shape=(12, 8))
def test_the_projection_is_the_euclidean_one(seed: int, shape: tuple[int, int]) -> None:
    """The projection against the exact least-distance one, on random polytopes with an interior."""
    rows, size = shape
    matrix, lower, upper, lo, hi, y = _random_polytope(np.random.default_rng(seed), rows, size)
    blocks = _constraint_blocks(
        [LinearConstraint(matrix, lower, upper)], jnp.asarray(lo), jnp.asarray(hi), jnp.float64
    )
    projected, _ = _dykstra(
        jnp.asarray(y),
        jnp.asarray(lo),
        jnp.asarray(hi),
        blocks,
        _no_duals(size, blocks, jnp.float64),
    )
    exact = _exact_projection(y, lo, hi, matrix, lower, upper)
    assert np.max(np.abs(np.asarray(projected) - exact)) < 1e-9


@pytest.mark.parametrize("steps", [0, 2, 3])
def test_the_polish_vouches_only_for_the_projection(steps: int) -> None:
    """Wherever the active-set loop stops, a point it accepts is the projection.

    Random multipliers of admissible sign and a starved step budget stop the loop mid-way on
    purpose -- rows still held by multipliers they no longer bind, coordinates released half-way --
    which is where a check that trusted the method's own bookkeeping would vouch for a wrong point.
    """
    polish = jax.jit(_polish, static_argnames="steps")
    accepted = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        rows, size = ((3, 4), (6, 5), (9, 6))[seed % 3]
        matrix, lower, upper, lo, hi, y = _random_polytope(rng, rows, size)
        blocks = _constraint_blocks(
            [LinearConstraint(matrix, lower, upper)], jnp.asarray(lo), jnp.asarray(hi), jnp.float64
        )
        multipliers = []
        for block_rows, block_lower, block_upper, _ in blocks:
            nu = rng.normal(size=block_rows.shape[0]) * (rng.random(block_rows.shape[0]) < 0.6)
            nu = np.where(np.isinf(block_upper) & (nu > 0), 0.0, nu)  # no bound to push down from
            nu = np.where(np.isinf(block_lower) & (nu < 0), 0.0, nu)
            multipliers.append(jnp.asarray(nu))
        tolerance = 256 * np.finfo(np.float64).eps * (1 + np.max(np.abs(y)))
        kkt, x, _ = polish(
            jnp.asarray(y),
            jnp.asarray(lo),
            jnp.asarray(hi),
            blocks,
            (jnp.zeros(size), tuple(multipliers)),
            jnp.asarray(tolerance),
            steps=steps,
        )
        if kkt:
            accepted += 1
            exact = _exact_projection(y, lo, hi, matrix, lower, upper)
            assert np.max(np.abs(np.asarray(x) - exact)) < 1e-9, seed
    assert accepted > 0


def test_rows_orthogonal_only_in_sum_take_a_class_each_for_the_planner() -> None:
    """``(1, 1)`` and ``(1, -1)`` over two levers are orthogonal, so Dykstra may project onto both
    at once, and a budget row of ones joins a rate band's class. The planner's descent scales each
    action's column by its own factor, and scaled apart such rows are not orthogonal, so for it
    rows share a class only where no action enters both. On the plans tried the polish finished
    every projection either way; the classes keep the sweeps exact where it does not."""
    pair = LinearConstraint(
        np.array([[1.0, 1.0], [1.0, -1.0]]), np.array([-1.0, -np.inf]), np.array([np.inf, 0.6])
    )
    lo, hi = jnp.full((1, 2), -5.0), jnp.full((1, 2), 5.0)
    assert len(_constraint_blocks([pair], lo, hi, jnp.float64)) == 1
    assert len(_constraint_blocks([pair], lo, hi, jnp.float64, disjoint=True)) == 2
    rate = LinearConstraint.rate_limit(4, [1.0])
    budget = LinearConstraint(np.ones((1, 4)), -np.inf, 3.0)
    lo, hi = jnp.full((4, 1), -5.0), jnp.full((4, 1), 5.0)
    assert len(_constraint_blocks([rate, budget], lo, hi, jnp.float64)) == 2
    assert len(_constraint_blocks([rate, budget], lo, hi, jnp.float64, disjoint=True)) == 3


def test_the_planner_projects_on_classes_its_scaling_keeps_orthogonal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both entry points hand the descent rows coloured for its scaled variables: in every class
    no action enters two rows, so a rate band and a budget, orthogonal in the caller's units, are
    projected on apart."""
    seen = []
    loop = chc.control._projected_gradient_loop

    def recorded(*args):
        seen.append(args[10])  # the constraint blocks
        return loop(*args)

    monkeypatch.setattr(chc.control, "_projected_gradient_loop", recorded)
    dyn, cost, x0, *_ = _problem()
    rate = LinearConstraint.rate_limit(HORIZON, [0.5])
    budget = LinearConstraint(np.ones((1, HORIZON)), -np.inf, 2.0)
    us0 = jnp.zeros((HORIZON, 1))
    for solve in (projected_gradient_solve, projected_gradient_control):
        solve(dyn, x0, us0, DT, cost, -U_MAX, U_MAX, steps=3, constraints=(rate, budget))
    assert len(seen) == 2
    for blocks in seen:
        assert sum(rows.shape[0] for rows, *_ in blocks) == HORIZON  # the band's 29, the budget
        for rows, *_ in blocks:
            support = (np.abs(np.asarray(rows)) > 0.0).astype(int)
            shared = support @ support.T
            assert np.array_equal(shared, np.diag(np.diag(shared)))


@pytest.mark.parametrize("pinned", [0, 1])
@pytest.mark.parametrize("scale", [1.0, 1e6])
def test_a_plan_the_rows_and_the_box_pin_to_one_point_stays_on_it(
    pinned: int, scale: float
) -> None:
    """``3 u1 = u2`` at every step, with one lever's box pinned at zero, leaves the zero plan alone
    feasible, and the projection holds the row only to its tolerance. With the cost 1e6 times its
    own, the descent in the caller's units took a trial that put the free lever 1e-12 off zero as a
    step, since it lowered the cost by more than ``1e-9``; scaled, with ``tol`` relative, a trial
    5e-14 off, which lowered it by 1.9e-14 and 1.4e-13 of itself, in any units. Either way it
    reported ``converged`` with the plan off the row. A trial that moves the plan by the
    projection's error alone is not a step."""
    horizon = 3
    model = LinearDynamics(jnp.array([[-0.5]]), jnp.array([[0.8, 0.1]]))
    cost = QuadraticCost(
        Q=scale * jnp.eye(1),
        R=scale * jnp.diag(jnp.array([0.01, 1.0])),
        Qf=scale * jnp.eye(1),
        x_target=jnp.array([1.0]),
    )
    tie = LinearConstraint(np.kron(np.eye(horizon), [[3.0, -1.0]]) / np.sqrt(10.0), 0.0, 0.0)
    lo, hi = np.full((horizon, 2), -2.0), np.full((horizon, 2), 2.0)
    lo[:, pinned] = hi[:, pinned] = 0.0
    solve = projected_gradient_solve(
        model, jnp.ones(1), jnp.zeros((horizon, 2)), DT, cost, lo, hi, constraints=(tie,)
    )
    assert solve.status == "no_progress"
    assert np.asarray(solve.actions).tolist() == [[0.0, 0.0]] * horizon


@pytest.mark.parametrize("pinned", [0, 1])
def test_a_search_that_ends_on_an_unsettled_projection_reports_the_cost_of_its_plan(
    pinned: int,
) -> None:
    """On the pinned plan above, the trial the projection moved by its own error alone reads 1.9e-14
    and 1.4e-13 below the start's cost. The line search holds a trial that fell while it halves on
    for one that buys its share of the predicted fall; a trial whose projection did not settle is
    neither a step nor a value it holds, so the solve reports the cost of the plan it returns."""
    horizon = 3
    model = LinearDynamics(jnp.array([[-0.5]]), jnp.array([[0.8, 0.1]]))
    cost = QuadraticCost(
        Q=jnp.eye(1), R=jnp.diag(jnp.array([0.01, 1.0])), Qf=jnp.eye(1), x_target=jnp.array([1.0])
    )
    tie = LinearConstraint(np.kron(np.eye(horizon), [[3.0, -1.0]]) / np.sqrt(10.0), 0.0, 0.0)
    lo, hi = np.full((horizon, 2), -2.0), np.full((horizon, 2), 2.0)
    lo[:, pinned] = hi[:, pinned] = 0.0
    solve = projected_gradient_solve(
        model, jnp.ones(1), jnp.zeros((horizon, 2)), DT, cost, lo, hi, constraints=(tie,)
    )
    at_plan = float(total_cost(model, jnp.ones(1), solve.actions, DT, cost))
    assert solve.iterations == 0
    assert float(solve.cost_history[-1]) == pytest.approx(at_plan, rel=5e-15, abs=0.0)


def test_the_line_search_hands_back_the_duals_of_the_trial_it_takes() -> None:
    """Each trial is projected by Dykstra from the iterate's duals, and the duals of the trial the
    search takes come back with it, to warm-start the next step's projection. Here the first trial
    buys 0.875 of the fall its gradient predicts and is taken, and the budget row binds there, so
    its multiplier is not the zero the search started from."""
    us, lo, hi = jnp.zeros(2), jnp.full(2, -5.0), jnp.full(2, 5.0)
    blocks = _constraint_blocks(
        (LinearConstraint(np.ones((1, 2)), -np.inf, 1.0),), lo, hi, us.dtype
    )
    duals = _no_duals(2, blocks, us.dtype)
    target = jnp.array([2.0, 2.0])

    def value_of(actions: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum((actions - target) ** 2)

    grad = 2.0 * (us - target)
    taken, _, accepted, taken_duals = _backtrack(
        us, value_of(us), grad, lo, hi, 1.0, 0.0, value_of, blocks, duals
    )
    first, first_duals = _dykstra(us - grad, lo, hi, blocks, duals)
    assert bool(accepted)
    np.testing.assert_allclose(taken, first, rtol=1e-12, atol=0.0)
    assert float(first_duals[1][0][0]) != 0.0
    for kept, expected in zip(
        jax.tree.leaves(taken_duals), jax.tree.leaves(first_duals), strict=True
    ):
        np.testing.assert_allclose(kept, expected, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize(
    ("side", "row", "start", "end", "unresolved"),
    [
        # across the row and the side that pin the plan: the polish left this, 10 tolerances out
        ((0.0, 1.0), (1.03, -0.046, 0.0, 0.0), (0.0, 0.0), (0.0, 1.1e-12), True),
        ((1.0, 1.0), (1.0, 1.0, 0.0, 0.0), (0.0, 0.0), (0.3, -0.3), False),  # along a held row
        ((1.0, 1.0), (1.0, 0.0, -np.inf, 0.5), (0.0, 0.0), (0.5, 0.0), False),  # onto a row
        ((1.0, 1.0), (0.0, 1.0, -1.0, 1.0), (0.5, 0.0), (1.0, 0.0), False),  # onto a side
    ],
)
def test_a_move_only_across_what_holds_the_plan_at_both_ends_is_not_a_step(
    side: tuple[float, float],
    row: tuple[float, float, float, float],
    start: tuple[float, float],
    end: tuple[float, float],
    unresolved: bool,
) -> None:
    """Dykstra holds each side and row to its tolerance, which pins a vertex where a row meets a
    side at a narrow angle only to the tolerance over the angle's sine. What a move does beyond the
    constraints held at both of its ends is a step; what it does across them alone is not."""
    from chc.control import _unresolved

    hi = jnp.asarray(side)
    rows = LinearConstraint(np.array([row[:2]]), np.array([row[2]]), np.array([row[3]]))
    blocks = _constraint_blocks([rows], -hi, hi, jnp.float64)
    verdict = _unresolved(jnp.asarray(start), jnp.asarray(end), -hi, hi, blocks, jnp.asarray(1e-13))
    assert bool(verdict) is unresolved


@pytest.mark.parametrize("shift", [0.0, 2.5, -4.0])
def test_a_lever_capped_at_zero_is_held_at_its_clipped_mean(shift: float) -> None:
    """A zero rate cap makes every row an equality; the projection is then ``clip(mean(y))``."""
    rng = np.random.default_rng(7)
    y = shift + rng.normal(0.0, 1.5, HORIZON)
    lo, hi = -np.ones(HORIZON), np.ones(HORIZON)
    blocks = _constraint_blocks(
        [LinearConstraint.rate_limit(HORIZON, [0.0])], jnp.asarray(lo), jnp.asarray(hi), jnp.float64
    )
    projected, _ = _dykstra(
        jnp.asarray(y),
        jnp.asarray(lo),
        jnp.asarray(hi),
        blocks,
        _no_duals(HORIZON, blocks, jnp.float64),
    )
    np.testing.assert_allclose(projected, np.clip(y.mean(), -1.0, 1.0), rtol=0, atol=1e-10)


def test_nan_actions_break_every_row_rather_than_none() -> None:
    """``max`` dropped the nan, so a solve that ended on nan actions reported a
    ``constraint_violation`` of 0."""
    row = LinearConstraint(np.array([[1.0, 1.0]]), np.array([0.0]), np.array([1.0]))
    assert chc.control._violation(jnp.array([0.2, 0.3]), [row]) == 0.0
    assert math.isnan(chc.control._violation(jnp.array([np.nan, 0.3]), [row]))
