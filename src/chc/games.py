"""Game-theoretic control: agent equilibria + differentiable Stackelberg (leader) control.

Marketplaces have strategic, mobile agents, so SUTVA fails -- the platform is a Stackelberg *leader*
over the agents' equilibrium, and its action is optimised accounting for the induced best response.
These are the reusable methods (equilibrium solver + bilevel allocator); the benchmark *task* that
scores them (the zone-incentive game) lives in ``causaldyn-bench``. See ``plans/16``.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from jax import Array


def project_simplex(v: Array, z: float) -> Array:
    """Euclidean projection of ``v`` onto ``{u >= 0, sum u = z}`` (Duchi et al. 2008)."""
    n = v.shape[0]
    sorted_v = jnp.sort(v)[::-1]
    cssv = jnp.cumsum(sorted_v) - z
    rho = jnp.count_nonzero(sorted_v - cssv / (jnp.arange(n) + 1) > 0)
    theta = cssv[rho - 1] / rho
    return jnp.maximum(v - theta, 0.0)


def softmax_congestion_equilibrium(
    attract: Array,
    u: Array,
    congestion: float,
    mass: float,
    beta: float = 2.5,
    iters: int = 120,
) -> Array:
    """Agent best-response equilibrium: a softmax congestion fixed point of the mass distribution.

    Agents flow toward higher value ``attract + u - congestion*x/mass`` (crowding lowers it); the
    damped iteration converges to the Wardrop/logit equilibrium ``x`` (summing to ``mass``).
    """
    n = attract.shape[0]

    def body(_: int, x: Array) -> Array:
        value = beta * (attract + u - congestion * x / mass)
        return 0.5 * x + 0.5 * mass * jax.nn.softmax(value)

    return jax.lax.fori_loop(0, iters, body, jnp.full(n, mass / n))


def stackelberg_allocation(
    objective: Callable[[Array], Array],
    n: int,
    budget: float,
    steps: int = 400,
    lr: float = 0.05,
) -> Array:
    """Differentiable-bilevel leader allocation: maximise ``objective(u)`` over the budget simplex.

    ``objective`` is evaluated *through* the equilibrium (so ``jax.grad`` differentiates the bilevel
    problem). Adam handles the scale; the plan is projected onto the budget simplex each step, and
    the best feasible allocation seen is returned.
    """
    u = jnp.full(n, budget / n)
    grad_fn = jax.jit(jax.grad(lambda u: -objective(u)))
    value_fn = jax.jit(objective)
    optimizer = optax.adam(lr)
    state = optimizer.init(u)
    best_u, best_val = u, float(value_fn(u))
    for _ in range(steps):
        updates, state = optimizer.update(grad_fn(u), state)
        u = project_simplex(optax.apply_updates(u, updates), budget)
        val = float(value_fn(u))
        if val > best_val:
            best_u, best_val = u, val
    return best_u
