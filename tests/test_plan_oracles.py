"""The planner against closed forms: where optimal control has the answer, ``causal_plan`` finds it.

``validation/planner_oracles.mac`` derives both, and a wrong gradient or costate misses either.

* **The golden rule.** Undiscounted, the middle of a long plan on ``MarketingMixSystem`` sits at
  the best steady state, Nerlove and Arrow's turnpike (Weber, (3.73)-(3.75)), whatever the step:
  an RK4 step with the spend held keeps the plant's equilibria and its costate ratio. There
  ``k u = w (S* - S) / d (gamma + beta h'(A) / theta)``: the spend's price against the value of the
  sales it buys, now through ``gamma`` and later through the adstock.
* **Bang, singular, bang.** Harvesting a logistic stock for ``(x - 1) u`` (Anita, Arnautu and
  Capasso 2011, sec. 3.4): no effort until the stock reaches ``x~ = (k + 1) / 2``, the effort that
  holds it there, and full effort from where the stock would end at ``1 + (x~ - 1) u~ / ubar``.
  The only oracle here with a singular arc.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

from chc.cost import QuadraticCost
from chc.mmm import MarketingMixSystem
from chc.plan import causal_plan

SYSTEM = MarketingMixSystem()
PRICE, TARGET = 0.08, 8.0  # the spend's quadratic price, and the sales target at weight 1

# validation/planner_oracles.mac, STEP 4
SINGULAR_STOCK, SINGULAR_EFFORT = 3.0, 0.12
ENTRY, EXIT, VALUE = 2.703100720721096, 9.27765777115768, 2.578920572317573


def _golden_rule() -> tuple[np.ndarray, float, np.ndarray]:
    """The best steady state's spend, sales and adstock, found without the planner."""
    beta, gamma, theta, half = (
        np.asarray(values) for values in (SYSTEM.beta, SYSTEM.gamma, SYSTEM.theta, SYSTEM.half)
    )

    def steady(spend: np.ndarray) -> tuple[float, np.ndarray]:
        adstock = spend / theta
        carried = np.sum(beta * adstock / (half + adstock)) + np.sum(gamma * spend)
        return SYSTEM.base + carried / SYSTEM.decay, adstock

    def cost(spend: np.ndarray) -> float:
        return 0.5 * (steady(spend)[0] - TARGET) ** 2 + 0.5 * PRICE * float(np.sum(spend**2))

    best = minimize(
        cost,
        np.ones(3),
        method="L-BFGS-B",
        bounds=[(SYSTEM.spend_floor, SYSTEM.spend_ceiling)] * 3,
        options={"ftol": 1e-16, "gtol": 1e-13, "maxiter": 10_000},
    )
    sales, adstock = steady(best.x)
    return best.x, sales, adstock


def test_the_best_steady_state_prices_spend_as_nerlove_and_arrow_do() -> None:
    spend, sales, adstock = _golden_rule()
    beta, gamma, theta, half = (
        np.asarray(values) for values in (SYSTEM.beta, SYSTEM.gamma, SYSTEM.theta, SYSTEM.half)
    )
    slope = half / (half + adstock) ** 2
    worth = (TARGET - sales) / SYSTEM.decay * (gamma + beta * slope / theta)
    np.testing.assert_allclose(PRICE * spend, worth, atol=1e-7)


@pytest.mark.parametrize(("dt", "horizon"), [(1.0, 80), (0.5, 160)])
def test_the_middle_of_a_long_plan_is_the_golden_rule_at_any_step(dt: float, horizon: int) -> None:
    spend, sales, _ = _golden_rule()
    weight = jnp.diag(jnp.array([1.0, 0.0, 0.0, 0.0]))
    cost = QuadraticCost(
        Q=weight, R=PRICE * jnp.eye(3), Qf=weight, x_target=jnp.array([TARGET, 0.0, 0.0, 0.0])
    )
    start = jnp.array([SYSTEM.base, 0.0, 0.0, 0.0])
    plan = causal_plan(
        SYSTEM.plant(),
        start,
        cost,
        dt,
        horizon,
        SYSTEM.spend_floor,
        SYSTEM.spend_ceiling,
        steps=100_000,
    )
    middle = horizon // 2
    assert plan.solver_status == "converged"
    assert abs(float(plan.trajectory[middle, 0]) - sales) < 1e-6
    np.testing.assert_allclose(plan.actions[middle], spend, atol=1e-3)


class _Harvest(eqx.Module):
    """A logistic stock fished with effort ``u``, and the revenue ``(x - 1) u`` it has earned."""

    def __call__(self, t: float, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        stock = x[0]
        return jnp.array([0.3 * stock * (1.0 - stock / 5.0) - u[0] * stock, (stock - 1.0) * u[0]])


def test_a_harvest_plan_rides_the_singular_arc_and_earns_its_value() -> None:
    dt, horizon, top = 0.1, 100, 1.5
    # a revenue target far above reach: the cost falls as the revenue rises
    cost = QuadraticCost(
        Q=jnp.zeros((2, 2)),
        R=jnp.zeros((1, 1)),
        Qf=jnp.diag(jnp.array([0.0, 1.0])),
        x_target=jnp.array([0.0, 100.0]),
    )
    plan = causal_plan(
        _Harvest(), jnp.array([2.0, 0.0]), cost, dt, horizon, 0.0, top, steps=200_000
    )
    stock, effort = np.asarray(plan.trajectory)[:, 0], np.asarray(plan.actions)[:, 0]
    assert plan.solver_status == "converged"
    np.testing.assert_allclose(stock[40:60], SINGULAR_STOCK, atol=1e-4)
    np.testing.assert_allclose(effort[40:60], SINGULAR_EFFORT, atol=5e-4)
    # effort held over steps is a restriction of the continuous problem, so it earns no more
    revenue = float(plan.trajectory[-1, 1])
    assert VALUE - 1e-4 < revenue <= VALUE + 1e-6
    entry = dt * int(np.argmax(effort > 1e-6))
    leave = dt * (horizon - int(np.argmax(effort[::-1] < top - 1e-6)))
    assert abs(entry - ENTRY) <= dt
    assert abs(leave - EXIT) <= dt
