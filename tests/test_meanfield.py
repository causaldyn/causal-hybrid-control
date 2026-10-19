"""Mean-field gate: planning that anticipates the migration lag beats a myopic controller."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from chc.research.meanfield import MeanFieldControl


def test_meanfield_planning_beats_myopic_under_migration_lag() -> None:
    regrets = MeanFieldControl(horizon=14).regrets(steps=200)
    assert regrets["planned-CHC"] < regrets["myopic"]  # anticipating the lag beats reacting to it
    assert regrets["myopic"] < regrets["no-control"]  # some control beats none


def test_the_oracle_is_no_worse_than_slsqp_from_the_myopic_plan() -> None:
    """The planner run three times as long stalled at 43.1 here, where SLSQP from the myopic plan
    reaches 30.3: the oracle must be at least as good as any plan found, and no regret negative."""
    model = MeanFieldControl(horizon=14)
    regrets = model.regrets(steps=200)
    myopic = model._myopic_plan()
    shape = myopic.shape
    budget_rows = np.kron(np.eye(shape[0]), np.ones(shape[1]))
    cost = jax.jit(model.rollout_cost)
    slope = jax.jit(jax.grad(model.rollout_cost))
    found = minimize(
        lambda v: float(cost(jnp.asarray(v.reshape(shape)))),
        np.asarray(myopic, dtype=float).ravel(),
        jac=lambda v: np.asarray(slope(jnp.asarray(v.reshape(shape))), dtype=float).ravel(),
        method="SLSQP",
        bounds=[(0.0, None)] * budget_rows.shape[1],
        constraints=[{"type": "eq", "fun": lambda v: budget_rows @ v - model.budget}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    assert np.max(np.abs(budget_rows @ found.x - model.budget)) < 1e-6
    oracle = float(model.rollout_cost(myopic)) - regrets["myopic"]
    assert oracle <= found.fun + 1e-3
    assert min(regrets.values()) >= 0.0
