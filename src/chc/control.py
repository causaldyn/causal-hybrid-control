"""Offline optimal control: projected gradient with backtracking (monotone descent)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from chc.adjoint import control_gradient_adjoint
from chc.cost import QuadraticCost, total_cost
from chc.dynamics import Dynamics


def project_box(us: Array, lo: float, hi: float) -> Array:
    """Euclidean projection onto the box ``[lo, hi]`` (elementwise clip)."""
    return jnp.clip(us, lo, hi)


def projected_gradient_control(
    dyn: Dynamics,
    x0: Array,
    us0: Array,
    dt: float,
    cost: QuadraticCost,
    u_lo: float,
    u_hi: float,
    steps: int = 200,
    lr0: float = 0.2,
    tol: float = 1e-9,
) -> tuple[Array, Array]:
    """Minimise ``J`` over the control sequence subject to box constraints.

    Uses the discrete adjoint (:func:`chc.adjoint.control_gradient_adjoint`) for the gradient and
    backtracking line search, so every accepted step strictly decreases the cost. Returns the
    optimised controls and the cost history (length ``1 + accepted steps``).
    """
    us = project_box(us0, u_lo, u_hi)
    current = total_cost(dyn, x0, us, dt, cost)
    history = [float(current)]

    for _ in range(steps):
        grad = control_gradient_adjoint(dyn, x0, us, dt, cost)
        lr = lr0
        improved = False
        candidate = us
        candidate_cost = current
        for _ls in range(40):
            candidate = project_box(us - lr * grad, u_lo, u_hi)
            candidate_cost = total_cost(dyn, x0, candidate, dt, cost)
            if candidate_cost < current - tol:
                improved = True
                break
            lr *= 0.5
        if not improved:
            break
        us, current = candidate, candidate_cost
        history.append(float(current))

    return us, jnp.asarray(history)
