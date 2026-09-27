"""Bolza objective (running + terminal quadratic cost) and the trajectory cost functional."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from chc.dynamics import Dynamics
from chc.integrate import rollout


class QuadraticCost(eqx.Module):
    """Bolza quadratic cost: running + terminal penalties on ``x - x_target`` and ``u``.

    ``x_target`` is one target for every state, shape ``(n,)``, or one per state of a plan, shape
    ``(H + 1, n)``: row ``k`` is the target of ``x_k`` and the last row the terminal one. A row
    means nothing without its step, so with a per-state target :meth:`running` and
    :meth:`terminal` take the row as ``target``, and :meth:`targets` hands every consumer the rows
    of an ``H``-step plan.
    """

    Q: Array
    R: Array
    Qf: Array
    x_target: Array

    def running(self, x: Array, u: Array, target: Array | None = None) -> Array:
        dx = x - (self._fixed() if target is None else target)
        return 0.5 * dx @ self.Q @ dx + 0.5 * u @ self.R @ u

    def terminal(self, x: Array, target: Array | None = None) -> Array:
        dx = x - (self._fixed() if target is None else target)
        return 0.5 * dx @ self.Qf @ dx

    def targets(self, horizon: int) -> Array:
        """``(horizon + 1, n)``: the target of every state of a ``horizon``-step plan."""
        if self.x_target.ndim < 2:
            return jnp.broadcast_to(self.x_target, (horizon + 1, self.Q.shape[0]))
        if self.x_target.shape[0] != horizon + 1:
            raise ValueError(
                f"x_target has {self.x_target.shape[0]} rows, one per state, but a {horizon}-step "
                f"plan has {horizon + 1} states"
            )
        return self.x_target

    def _fixed(self) -> Array:
        if self.x_target.ndim >= 2:
            raise ValueError("x_target has one row per state; pass the stage's row as target")
        return self.x_target


@eqx.filter_jit
def total_cost(dyn: Dynamics, x0: Array, us: Array, dt: float, cost: QuadraticCost) -> Array:
    """``J = Σ_{k<H} L(x_k, u_k) + Φ(x_H)`` over the rolled-out trajectory."""
    xs = rollout(dyn, x0, us, dt)
    targets = cost.targets(us.shape[0])
    running = jnp.sum(jax.vmap(cost.running)(xs[:-1], us, targets[:-1]))
    return running + cost.terminal(xs[-1], targets[-1])
