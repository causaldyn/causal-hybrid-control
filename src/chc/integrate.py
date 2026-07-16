"""Fixed-step explicit integration and zero-order-hold rollout."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array, lax

from chc.dynamics import Dynamics


def rk4_step(dyn: Dynamics, t: float | Array, x: Array, u: Array, dt: float) -> Array:
    """One classical Runge-Kutta 4 step with the control held constant over the step."""
    k1 = dyn(t, x, u)
    k2 = dyn(t + 0.5 * dt, x + 0.5 * dt * k1, u)
    k3 = dyn(t + 0.5 * dt, x + 0.5 * dt * k2, u)
    k4 = dyn(t + dt, x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rollout(dyn: Dynamics, x0: Array, us: Array, dt: float, t0: float = 0.0) -> Array:
    """Roll the dynamics forward under a control sequence.

    Args:
        us: control sequence, shape ``(H, m)``.

    Returns:
        State trajectory ``xs`` of shape ``(H + 1, n)`` including the initial state.
    """

    def body(carry: tuple[Array, Array], u: Array) -> tuple[tuple[Array, Array], Array]:
        t, x = carry
        x_next = rk4_step(dyn, t, x, u, dt)
        return (t + dt, x_next), x_next

    _, xs = lax.scan(body, (jnp.asarray(t0), x0), us)
    return jnp.concatenate([x0[None, :], xs], axis=0)
