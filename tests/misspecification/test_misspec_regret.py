"""The plan's regret bound on a plan made by a model other than the one it is priced on.

:func:`~chc.plan.plan_regret_bound` reads a plan's optimality gap off its gradient on the model it
is given, whatever made the plan, and calls the bound ``certified`` where that model is linear.
The contract read here: a plan solved on a misspecified model, under a box and a budget row,
priced on the plant, is no further from the plant's best plan under the same box and row than the
certified bound says. The plant's best plan is SLSQP's on the exact quadratic, at a tolerance far
below the gaps read.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import optimize

from chc import QuadraticCost, causal_plan
from chc.control import LinearConstraint
from chc.cost import total_cost
from chc.dynamics import DampedOscillator
from chc.plan import plan_regret_bound

pytestmark = pytest.mark.misspecification

HORIZON, DT, BOX = 12, 0.1, 5.0
COST = QuadraticCost(
    Q=jnp.diag(jnp.array([1.0, 0.0])),
    R=jnp.array([[0.01]]),
    Qf=jnp.diag(jnp.array([10.0, 1.0])),
    x_target=jnp.zeros(2),
)
START = jnp.array([1.0, 0.0])
PLANT = DampedOscillator(omega=1.0, zeta=0.1)


def _best(budget: LinearConstraint) -> float:
    """The plant's least cost under the box and the budget row, by SLSQP on its exact gradient."""
    cost = jax.jit(lambda u: total_cost(PLANT, START, u[:, None], DT, COST))
    grad = jax.jit(jax.grad(lambda u: total_cost(PLANT, START, u[:, None], DT, COST)))
    row = np.asarray(budget.matrix)[0]
    found = optimize.minimize(
        lambda u: float(cost(jnp.asarray(u))),
        np.zeros(HORIZON),
        jac=lambda u: np.asarray(grad(jnp.asarray(u))),
        bounds=[(-BOX, BOX)] * HORIZON,
        constraints=[{"type": "ineq", "fun": lambda u: float(budget.upper[0]) - row @ u}],
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    assert found.success, found.message
    return float(found.fun)


@pytest.mark.parametrize("omega", [0.6, 1.0, 1.6])
def test_the_certified_bound_covers_a_plan_made_on_a_wrong_model(omega: float) -> None:
    """Plans made on oscillators of frequency 0.6, 1 and 1.6 where the plant's is 1, held to
    spend at most half of what the plant's free plan spends: the gaps on the plant are 0.195, 0
    and 0.617, the certified bounds 6.11, 0.088 and 6.45. The bound reads the box alone, so where
    the row binds it is loose by what the row costs, as at frequency 1."""
    free = causal_plan(PLANT, START, COST, DT, HORIZON, -BOX, BOX)
    spend = float(np.sum(np.asarray(free.actions)))
    budget = LinearConstraint(np.sign(spend) * np.ones((1, HORIZON)), -np.inf, 0.5 * abs(spend))
    model = DampedOscillator(omega=omega, zeta=0.1)
    plan = causal_plan(model, START, COST, DT, HORIZON, -BOX, BOX, constraints=(budget,))
    bound = plan_regret_bound(plan, PLANT, START, COST, DT, -BOX, BOX)
    gap = float(total_cost(PLANT, START, plan.actions, DT, COST)) - _best(budget)
    assert bound.status == "certified"
    assert gap >= -1e-9
    assert bound.bound >= gap, (bound.bound, gap)


@pytest.mark.parametrize("omega", [0.6, 1.0, 1.6])
def test_the_bound_read_under_the_rows_is_still_valid_and_no_looser(omega: float) -> None:
    """The same plans priced against the box and the budget row they were solved under. The bound
    still covers the gap (1.83, 4e-13 and 6.45 for 0.195, 0 and 0.617) and is never above the
    box's alone but by rounding: at 1.6 the row does not bind, and the two differ in the last bit.
    Where the model is the plant it is the optimum's zero, not the row's 0.088."""
    free = causal_plan(PLANT, START, COST, DT, HORIZON, -BOX, BOX)
    spend = float(np.sum(np.asarray(free.actions)))
    budget = LinearConstraint(np.sign(spend) * np.ones((1, HORIZON)), -np.inf, 0.5 * abs(spend))
    model = DampedOscillator(omega=omega, zeta=0.1)
    plan = causal_plan(model, START, COST, DT, HORIZON, -BOX, BOX, constraints=(budget,))
    box = plan_regret_bound(plan, PLANT, START, COST, DT, -BOX, BOX)
    rows = plan_regret_bound(plan, PLANT, START, COST, DT, -BOX, BOX, constraints=(budget,))
    gap = float(total_cost(PLANT, START, plan.actions, DT, COST)) - _best(budget)
    assert rows.status == "certified"
    assert rows.bound >= gap, (rows.bound, gap)
    assert rows.bound <= box.bound + 4.0 * np.spacing(box.bound), (rows.bound, box.bound)
    if omega == 1.0:
        assert rows.bound <= 1e-9 < box.bound, (rows.bound, box.bound)
