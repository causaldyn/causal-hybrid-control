"""Control gradient dJ/du: a hand-written discrete adjoint + an adaptive diffrax continuous adjoint.

The discrete adjoint assembles the backward recursion explicitly from autodiff Jacobians -- the
object the gradient-check gate verifies against autodiff and finite differences.
:func:`control_gradient_diffrax` is the modern sibling: each interval is integrated by an adaptive
(stiff-capable) solver, with the continuous ``BacksolveAdjoint`` for O(1)-memory gradients.
"""

from __future__ import annotations

import diffrax
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


def total_cost_diffrax(
    dyn: Dynamics,
    x0: Array,
    us: Array,
    dt: float,
    cost: QuadraticCost,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    adjoint: diffrax.AbstractAdjoint | None = None,
) -> Array:
    """Discrete control cost ``J`` with each interval integrated by an adaptive diffrax solver.

    Same objective and zero-order-hold control as :func:`chc.cost.total_cost`, but each step
    ``x_k -> x_{k+1}`` is an adaptive (stiff-capable) Tsit5 solve of ``dx/dt = dyn(t, x, u_k)``, not
    the fixed-step RK4 of ``chc.integrate.rollout`` -- accuracy set by ``(rtol, atol)``, and the
    continuous ``BacksolveAdjoint`` available (via :func:`control_gradient_diffrax`).
    """
    # diffrax types the vector-field time as RealScalarLike; Dynamics takes the scalar at runtime
    term = diffrax.ODETerm(lambda t, x, u: dyn(t, x, u))  # type: ignore[arg-type]
    solver, controller = diffrax.Tsit5(), diffrax.PIDController(rtol=rtol, atol=atol)
    adjoint = diffrax.RecursiveCheckpointAdjoint() if adjoint is None else adjoint

    def step(carry: tuple[Array, Array], u: Array) -> tuple[tuple[Array, Array], Array]:
        t, x = carry
        solution = diffrax.diffeqsolve(
            term, solver, t, t + dt, dt0=dt / 5, y0=x, args=u,
            stepsize_controller=controller, adjoint=adjoint,
            saveat=diffrax.SaveAt(t1=True), max_steps=10000,
        )
        return (t + dt, solution.ys[-1]), x  # emit the pre-decision state x_k

    (_, x_final), xs = jax.lax.scan(step, (jnp.asarray(0.0), x0), us)
    return jnp.sum(jax.vmap(cost.running)(xs, us)) + cost.terminal(x_final)


def control_gradient_diffrax(
    dyn: Dynamics,
    x0: Array,
    us: Array,
    dt: float,
    cost: QuadraticCost,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    adjoint: diffrax.AbstractAdjoint | None = None,
) -> Array:
    """``dJ/du`` through the adaptive diffrax solve -- the continuous-adjoint sibling of the
    discrete :func:`control_gradient_adjoint`. Pass ``adjoint=diffrax.BacksolveAdjoint()`` for the
    memory-light optimise-then-discretise continuous adjoint; the default backprops the solver.
    """
    return jax.grad(lambda u: total_cost_diffrax(dyn, x0, u, dt, cost, rtol, atol, adjoint))(us)
