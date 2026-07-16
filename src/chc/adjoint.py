"""Discrete adjoint: the control gradient dJ/du via backward-in-time recursion.

Local Jacobians ``∂F/∂x``, ``∂F/∂u`` and the stage-cost gradients come from autodiff, but the
backward recursion is assembled explicitly. That is deliberate: it is the object the gradient-check
gate verifies against autodiff and finite differences, and it is the reference the eventual Rust
runtime will mirror (where no autodiff is available).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from chc.cost import QuadraticCost
from chc.dynamics import Dynamics
from chc.integrate import rk4_step, rollout


@eqx.filter_jit
def control_gradient_adjoint(
    dyn: Dynamics, x0: Array, us: Array, dt: float, cost: QuadraticCost
) -> Array:
    """Return ``dJ/du`` of shape ``(H, m)`` via the discrete adjoint.

    ``λ_H = ∇Φ(x_H)``;  ``λ_k = ∇_x L_k + (∂F/∂x_k)ᵀ λ_{k+1}``;
    ``∇_{u_k} J = ∇_u L_k + (∂F/∂u_k)ᵀ λ_{k+1}``.
    """
    xs = rollout(dyn, x0, us, dt)  # (H + 1, n)
    horizon = us.shape[0]

    def step_fn(x: Array, u: Array) -> Array:
        return rk4_step(dyn, 0.0, x, u, dt)

    f_x = jax.vmap(jax.jacobian(step_fn, argnums=0))(xs[:-1], us)  # (H, n, n)
    f_u = jax.vmap(jax.jacobian(step_fn, argnums=1))(xs[:-1], us)  # (H, n, m)
    l_x = jax.vmap(jax.grad(cost.running, argnums=0))(xs[:-1], us)  # (H, n)
    l_u = jax.vmap(jax.grad(cost.running, argnums=1))(xs[:-1], us)  # (H, m)
    lam_terminal = jax.grad(cost.terminal)(xs[-1])  # (n,)

    def body(lam: Array, k: Array) -> tuple[Array, Array]:
        g_u = l_u[k] + f_u[k].T @ lam
        lam_next = l_x[k] + f_x[k].T @ lam
        return lam_next, g_u

    ks = jnp.arange(horizon - 1, -1, -1)
    _, grads_reversed = jax.lax.scan(body, lam_terminal, ks)
    return grads_reversed[::-1]  # reorder to k = 0 .. H-1
