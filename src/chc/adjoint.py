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


@eqx.filter_jit
def costate_norms(dyn: Dynamics, x0: Array, us: Array, dt: float, cost: QuadraticCost) -> Array:
    """``||lambda_{k+1}||`` for ``k = 0 .. H-1`` -- the local Lipschitz constant of the cost-to-go.

    The same backward recursion :func:`control_gradient_adjoint` runs, returning the costates
    themselves rather than the gradient they contract with. ``lambda_{k+1}`` is
    ``dJ/dx_{k+1}`` along this trajectory, so ``||lambda_{k+1}||`` is exactly the first-order
    sensitivity of the objective to a perturbation of the state at step ``k+1`` -- which is what
    turns a bound on a per-step *transition* error into a bound on the *objective*
    (:class:`chc.uncertainty.ConfoundingRobustPenalty`, Result 38 (b)).

    Local by construction, and that is the honest reading: these are derivatives at one trajectory,
    so the bound they license is a first-order one about a neighbourhood of it, not a global
    Lipschitz constant of ``J``.
    """
    xs = rollout(dyn, x0, us, dt)
    horizon = us.shape[0]

    def step_fn(x: Array, u: Array) -> Array:
        return rk4_step(dyn, 0.0, x, u, dt)

    f_x = jax.vmap(jax.jacobian(step_fn, argnums=0))(xs[:-1], us)
    l_x = jax.vmap(jax.grad(cost.running, argnums=0))(xs[:-1], us)
    lam_terminal = jax.grad(cost.terminal)(xs[-1])

    def body(lam: Array, k: Array) -> tuple[Array, Array]:
        # The costate reported at k is lambda_{k+1}: the one that multiplies a perturbation
        # ENTERING x_{k+1}, which is the step u_k acts on.
        return l_x[k] + f_x[k].T @ lam, jnp.linalg.norm(lam)

    ks = jnp.arange(horizon - 1, -1, -1)
    _, norms_reversed = jax.lax.scan(body, lam_terminal, ks)
    return norms_reversed[::-1]


@eqx.filter_jit
def perturbation_cost_weights(
    dyn: Dynamics, x0: Array, us: Array, dt: float, cost: QuadraticCost, radius: float
) -> Array:
    """Weights ``w_k`` with ``sum_k w_k * radius * ||u_k|| >= |J(perturbed) - J(reference)|``.

    The object Result 38 (b) asks for. A support model licenses a bound on the *field*:
    ``||f_true(x, u) - f_hat(x, u)|| <= radius * ||u||``. This turns that into a bound on the
    *objective*, at the actions it is handed, in three pieces:

    * **injection.** One RK4 step maps a field offset ``g`` into a state offset
      ``dt (I + dtJ/2 + (dtJ)^2/6 + (dtJ)^3/24) g`` with ``J = df/dx`` -- exact for an offset
      constant over the step, which an unmodelled ``Delta_B u`` at fixed ``u`` is. The spectral
      norm of that polynomial falls either side of 1 depending on the plant -- 0.99990 on the
      certificate's oscillator, so a bare ``dt`` happens to be conservative there and need not be
      elsewhere -- which is reason enough to compute it rather than assume it.
    * **first order.** ``||lambda_{k+1}||`` contracted with that injection (:func:`costate_norms`).
    * **second order.** The first-order term is *asymptotically tight*, so it is not an upper
      bound: the curvature term it drops is positive and O(radius^2). A deviation tube
      ``rho_{k+1} = ||dF/dx|| rho_k + eps_k`` carries the injections forward, and
      ``1/2 sum ||d^2 L/dx^2|| rho^2`` closes the Taylor expansion. Exact for a linear plant with
      a quadratic cost, where ``J`` is exactly quadratic along the perturbation; ``O(radius^3)``
      otherwise.

    ``radius = 0`` is meaningful and is used: it zeroes the tube, so the weights collapse to
    exactly the first-order term, which is how the certificate reports both.

    The curvature term does not factor per step, so it is spread across the same
    ``radius * ||u_k||`` denominator the first-order term uses -- the *sum* is the bound, and only
    the sum is claimed. Weights are computed at one reference trajectory, so the guarantee is
    local to it; :func:`chc.uncertainty.confounding_cost_bound_certificate` measures where it holds.
    """
    xs = rollout(dyn, x0, us, dt)
    horizon = us.shape[0]
    eye = jnp.eye(x0.shape[0], dtype=xs.dtype)

    def step_fn(x: Array, u: Array) -> Array:
        return rk4_step(dyn, 0.0, x, u, dt)

    f_x = jax.vmap(jax.jacobian(step_fn, argnums=0))(xs[:-1], us)  # (H, n, n)
    field_x = jax.vmap(jax.jacobian(lambda x, u: dyn(0.0, x, u), argnums=0))(xs[:-1], us)

    def injection_gain(jac: Array) -> Array:
        m = dt * jac
        m2 = m @ m
        return dt * jnp.linalg.norm(eye + m / 2.0 + m2 / 6.0 + m2 @ m / 24.0, ord=2)

    gains = jax.vmap(injection_gain)(field_x)  # (H,)
    action_norms = jnp.linalg.norm(us, axis=1)
    eps = radius * action_norms * gains

    def tube(rho: Array, k: Array) -> tuple[Array, Array]:
        nxt = jnp.linalg.norm(f_x[k], ord=2) * rho + eps[k]
        return nxt, nxt

    _, rho = jax.lax.scan(tube, jnp.zeros((), dtype=xs.dtype), jnp.arange(horizon))  # rho_{k+1}

    hess_run = jax.vmap(jax.hessian(cost.running, argnums=0))(xs[:-1], us)
    curvature = jax.vmap(lambda h: jnp.linalg.norm(h, ord=2))(hess_run)
    terminal_curvature = jnp.linalg.norm(jax.hessian(cost.terminal)(xs[-1]), ord=2)
    second = 0.5 * (jnp.sum(curvature[1:] * rho[:-1] ** 2) + terminal_curvature * rho[-1] ** 2)

    first_order = costate_norms(dyn, x0, us, dt, cost) * gains
    denominator = radius * jnp.sum(action_norms)
    spread = jnp.where(
        denominator > 0.0, second / jnp.where(denominator > 0.0, denominator, 1.0), 0.0
    )
    return first_order + spread


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
    term = diffrax.ODETerm(lambda t, x, u: dyn(t, x, u))
    solver, controller = diffrax.Tsit5(), diffrax.PIDController(rtol=rtol, atol=atol)
    adjoint = diffrax.RecursiveCheckpointAdjoint() if adjoint is None else adjoint

    def step(carry: tuple[Array, Array], u: Array) -> tuple[tuple[Array, Array], Array]:
        t, x = carry
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t,
            t + dt,
            dt0=dt / 5,
            y0=x,
            args=u,
            stepsize_controller=controller,
            adjoint=adjoint,
            saveat=diffrax.SaveAt(t1=True),
            max_steps=10000,
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
