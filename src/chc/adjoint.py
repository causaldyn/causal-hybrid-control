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
from chc.dynamics import Dynamics, _EulerStep, _SetAtStep
from chc.integrate import rk4_step, rollout

__all__ = [
    "control_gradient_adjoint",
    "control_gradient_diffrax",
    "costate_norms",
    "perturbation_cost_weights",
    "total_cost_diffrax",
]


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
    targets = cost.targets(horizon)  # (H + 1, n)
    times = dt * jnp.arange(horizon)  # the clock the rollout steps: step k starts at k * dt

    def step_fn(t: Array, x: Array, u: Array) -> Array:
        return rk4_step(dyn, t, x, u, dt)

    f_x = jax.vmap(jax.jacobian(step_fn, argnums=1))(times, xs[:-1], us)  # (H, n, n)
    f_u = jax.vmap(jax.jacobian(step_fn, argnums=2))(times, xs[:-1], us)  # (H, n, m)
    l_x = jax.vmap(jax.grad(cost.running, argnums=0))(xs[:-1], us, targets[:-1])  # (H, n)
    l_u = jax.vmap(jax.grad(cost.running, argnums=1))(xs[:-1], us, targets[:-1])  # (H, m)
    lam_terminal = jax.grad(cost.terminal)(xs[-1], targets[-1])  # (n,)

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
    (:class:`chc.uncertainty.ConfoundingRobustPenalty`).

    Local by construction, and that is the honest reading: these are derivatives at one trajectory,
    so the bound they license is a first-order one about a neighbourhood of it, not a global
    Lipschitz constant of ``J``.
    """
    return jnp.linalg.norm(_costates(dyn, x0, us, dt, cost), axis=1)


def _costates(dyn: Dynamics, x0: Array, us: Array, dt: float, cost: QuadraticCost) -> Array:
    """``lambda_{k+1}`` for ``k = 0 .. H-1``, ``(H, n)``: :func:`costate_norms`' vectors."""
    xs = rollout(dyn, x0, us, dt)
    horizon = us.shape[0]
    targets = cost.targets(horizon)
    times = dt * jnp.arange(horizon)

    def step_fn(t: Array, x: Array, u: Array) -> Array:
        return rk4_step(dyn, t, x, u, dt)

    f_x = jax.vmap(jax.jacobian(step_fn, argnums=1))(times, xs[:-1], us)
    l_x = jax.vmap(jax.grad(cost.running, argnums=0))(xs[:-1], us, targets[:-1])
    lam_terminal = jax.grad(cost.terminal)(xs[-1], targets[-1])

    def body(lam: Array, k: Array) -> tuple[Array, Array]:
        # The costate reported at k is lambda_{k+1}: the one that multiplies a perturbation
        # ENTERING x_{k+1}, which is the step u_k acts on.
        return l_x[k] + f_x[k].T @ lam, lam

    ks = jnp.arange(horizon - 1, -1, -1)
    _, reversed_costates = jax.lax.scan(body, lam_terminal, ks)
    return reversed_costates[::-1]


class _Offset(eqx.Module):
    """``field(t, x, u) + offset``, the offset constant in the state, the time and the action."""

    field: Dynamics
    offset: Array

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return self.field(t, x, u) + self.offset


def _offset_step(dyn: Dynamics, t: Array, x: Array, u: Array, offset: Array, dt: float) -> Array:
    """One step of the field plus a constant ``offset``, the action held and the field stepped as
    :func:`chc.integrate.rk4_step` holds and steps it."""
    field, held = _stepped_field(dyn, x, u)
    if isinstance(field, _EulerStep):
        return rk4_step(_EulerStep(_Offset(field.field, offset)), t, x, held, dt)
    return rk4_step(_Offset(field, offset), t, x, held, dt)


def _stepped_field(dyn: Dynamics, x: Array, u: Array) -> tuple[Dynamics, Array]:
    """The field a step from ``x`` steps, and the action it holds over the step
    (:class:`chc.dynamics._SetAtStep`)."""
    return dyn.at_step(x, u) if isinstance(dyn, _SetAtStep) else (dyn, u)


@eqx.filter_jit
def perturbation_cost_weights(
    dyn: Dynamics, x0: Array, us: Array, dt: float, cost: QuadraticCost, radius: float
) -> Array:
    """Weights ``w_k`` with ``sum_k w_k * radius * ||u_k|| >= |J(perturbed) - J(reference)|``.

    The multiplier :meth:`chc.uncertainty.ConfoundingRobustPenalty.certified` needs. A support
    model licenses a bound on the *field*:
    ``||f_true(x, u) - f_hat(x, u)|| <= radius * ||u||``. This turns that into a bound on the
    *objective*, at the actions it is handed, in three pieces:

    * **injection.** One RK4 step maps a field offset ``g`` into a state offset
      ``dt (I + dtJ/2 + (dtJ)^2/6 + (dtJ)^3/24) g`` with ``J = df/dx`` -- exact for an offset
      constant over the step, which an unmodelled ``Delta_B u`` at fixed ``u`` is. The Euler map
      of a field fitted to it (:class:`chc.dynamics._EulerStep`) maps it into ``dt g``. The spectral
      norm of that polynomial falls either side of 1 depending on the plant -- 0.99990 on the
      certificate's oscillator, so a bare ``dt`` happens to be conservative there and need not be
      elsewhere -- which is reason enough to compute it rather than assume it.
    * **first order.** ``||lambda_{k+1}||`` contracted with that injection (:func:`costate_norms`).
    * **second order.** The first-order term is *asymptotically tight*, so it is not an upper
      bound: the curvature term it drops is positive and O(radius^2). A deviation tube
      ``rho_{k+1} = ||dF/dx|| rho_k + eps_k`` carries the injections forward, and
      ``1/2 sum ||d^2 L/dx^2|| rho^2`` adds the cost's curvature along it. The step's own
      curvature enters at the same order: ``lambda_{k+1}`` contracted with the Hessian of the step
      in the state and the field offset ``g_k`` together, ``K_k``, adds
      ``1/2 sum ||K_k|| (rho_k^2 + ||g_k||^2)``, ``||g_k|| <= radius ||u_k||``. Without it a
      cubic drift's bound read 0.97 of the worst case at radius 0.5. On a plant affine in the
      state ``K_k`` is zero and ``J`` is exactly quadratic along the perturbation, so the sum is
      an upper bound; otherwise it is one to second order, its ``O(radius^3)`` remainder not
      bounded, and every derivative read at the reference.

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
    times = dt * jnp.arange(horizon)

    def step_fn(t: Array, x: Array, u: Array) -> Array:
        return rk4_step(dyn, t, x, u, dt)

    f_x = jax.vmap(jax.jacobian(step_fn, argnums=1))(times, xs[:-1], us)  # (H, n, n)
    field_x = jax.vmap(jax.jacobian(dyn, argnums=1))(times, xs[:-1], us)

    def injection_gain(jac: Array) -> Array:
        m = dt * jac
        m2 = m @ m
        return dt * jnp.linalg.norm(eye + m / 2.0 + m2 / 6.0 + m2 @ m / 24.0, ord=2)

    if isinstance(_stepped_field(dyn, x0, jnp.zeros(us.shape[1:], us.dtype))[0], _EulerStep):
        gains = jnp.full((horizon,), dt, dtype=xs.dtype)
    else:
        gains = jax.vmap(injection_gain)(field_x)  # (H,)
    action_norms = jnp.linalg.norm(us, axis=1)
    eps = radius * action_norms * gains

    def tube(rho: Array, k: Array) -> tuple[Array, Array]:
        nxt = jnp.linalg.norm(f_x[k], ord=2) * rho + eps[k]
        return nxt, nxt

    _, rho = jax.lax.scan(tube, jnp.zeros((), dtype=xs.dtype), jnp.arange(horizon))  # rho_{k+1}

    targets = cost.targets(horizon)
    hess_run = jax.vmap(jax.hessian(cost.running, argnums=0))(xs[:-1], us, targets[:-1])
    curvature = jax.vmap(lambda h: jnp.linalg.norm(h, ord=2))(hess_run)
    terminal_curvature = jnp.linalg.norm(jax.hessian(cost.terminal)(xs[-1], targets[-1]), ord=2)
    second = 0.5 * (jnp.sum(curvature[1:] * rho[:-1] ** 2) + terminal_curvature * rho[-1] ** 2)

    def coupling(t: Array, x: Array, u: Array, lam: Array) -> Array:
        """``||K_k||``: the step's Hessian in ``(x, g)`` at ``(x_k, 0)``, contracted with
        ``lambda_{k+1}``."""
        size = x.shape[0]

        def reached(z: Array) -> Array:
            return lam @ _offset_step(dyn, t, z[:size], u, z[size:], dt)

        return jnp.linalg.norm(jax.hessian(reached)(jnp.concatenate([x, jnp.zeros_like(x)])), 2)

    couplings = jax.vmap(coupling)(times, xs[:-1], us, _costates(dyn, x0, us, dt, cost))
    deviation = jnp.concatenate([jnp.zeros((1,), rho.dtype), rho[:-1]])  # rho_k, rho_0 = 0
    second = second + 0.5 * jnp.sum(couplings * (deviation**2 + (radius * action_norms) ** 2))

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
    targets = cost.targets(us.shape[0])
    return jnp.sum(jax.vmap(cost.running)(xs, us, targets[:-1])) + cost.terminal(
        x_final, targets[-1]
    )


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
