"""Model predictive control: receding-horizon optimal control with warm starts.

At each step MPC solves a finite-horizon OC problem from the current measured state (against the
*model*), applies only the first control, advances the *plant*, and re-solves. Planning and
reality are separate objects (``model`` vs ``plant``), so the offline/confounded setting — plan
with the learned hybrid model, act on the true system — needs no change to this loop.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from chc.control import projected_gradient_control
from chc.cost import QuadraticCost
from chc.dynamics import Dynamics
from chc.integrate import rk4_step


def mpc_control(
    model: Dynamics,
    x0: Array,
    cost: QuadraticCost,
    dt: float,
    horizon: int,
    u_lo: float,
    u_hi: float,
    n_steps: int,
    *,
    plant: Dynamics | None = None,
    inner_steps: int = 40,
    warm_start: bool = True,
) -> tuple[Array, Array]:
    """Run closed-loop MPC for ``n_steps``; return the realised trajectory and applied controls.

    Args:
        model: dynamics used for planning (the controller's belief).
        plant: true dynamics the control is applied to (defaults to ``model``).

    Returns:
        ``(xs, us)`` with ``xs`` of shape ``(n_steps + 1, n)`` and ``us`` of shape ``(n_steps, m)``.
    """
    plant = model if plant is None else plant
    control_dim = cost.R.shape[0]
    guess = jnp.zeros((horizon, control_dim))
    x = x0
    states = [x0]
    applied: list[Array] = []

    for _ in range(n_steps):
        us_opt, _ = projected_gradient_control(
            model, x, guess, dt, cost, u_lo, u_hi, steps=inner_steps
        )
        u0 = us_opt[0]
        applied.append(u0)
        x = rk4_step(plant, 0.0, x, u0, dt)
        states.append(x)
        guess = (
            jnp.concatenate([us_opt[1:], us_opt[-1:]], axis=0)
            if warm_start
            else jnp.zeros((horizon, control_dim))
        )

    return jnp.stack(states), jnp.stack(applied)
