"""Bolza objective (running + terminal quadratic cost) and the trajectory cost functional."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from chc.dynamics import Dynamics
from chc.integrate import rollout


class QuadraticCost(eqx.Module):
    """Bolza quadratic cost: running + terminal penalties on ``x - x_target`` and ``u``."""

    Q: Array
    R: Array
    Qf: Array
    x_target: Array

    def running(self, x: Array, u: Array) -> Array:
        dx = x - self.x_target
        return 0.5 * dx @ self.Q @ dx + 0.5 * u @ self.R @ u

    def terminal(self, x: Array) -> Array:
        dx = x - self.x_target
        return 0.5 * dx @ self.Qf @ dx


@eqx.filter_jit
def total_cost(dyn: Dynamics, x0: Array, us: Array, dt: float, cost: QuadraticCost) -> Array:
    """``J = Σ_{k<H} L(x_k, u_k) + Φ(x_H)`` over the rolled-out trajectory."""
    xs = rollout(dyn, x0, us, dt)
    running = jnp.sum(jax.vmap(cost.running)(xs[:-1], us))
    return running + cost.terminal(xs[-1])
