"""LQR / AKOR: the r_θ→0 closed-form optimum and Riccati cross-checks.

For the linear(-ised) known system this is the analytic control optimum the learned controller
must approach (a correctness limit), cross-validated in Octave / Maxima (see ``validation/``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.linalg import solve_continuous_are

from chc.dynamics import Dynamics
from chc.integrate import rk4_step


def linearize_continuous(dyn: Dynamics, x: Array, u: Array) -> tuple[Array, Array]:
    """Jacobians ``(A, B) = (∂f/∂x, ∂f/∂u)`` of the continuous vector field at ``(x, u)``."""
    a = jax.jacobian(lambda z: dyn(0.0, z, u))(x)
    b = jax.jacobian(lambda w: dyn(0.0, x, w))(u)
    return a, b


def linearize_discrete(dyn: Dynamics, x: Array, u: Array, dt: float) -> tuple[Array, Array]:
    """Jacobians ``(A_d, B_d)`` of one RK4 step at ``(x, u)``."""
    a = jax.jacobian(lambda z: rk4_step(dyn, 0.0, z, u, dt))(x)
    b = jax.jacobian(lambda w: rk4_step(dyn, 0.0, x, w, dt))(u)
    return a, b


def continuous_lqr(a: Array, b: Array, q: Array, r: Array) -> tuple[Array, Array]:
    """Solve the continuous ARE; return ``(P, K)`` with the optimal feedback ``u = -K x``."""
    p = solve_continuous_are(np.asarray(a), np.asarray(b), np.asarray(q), np.asarray(r))
    k = np.linalg.solve(np.asarray(r), np.asarray(b).T @ p)
    return jnp.asarray(p), jnp.asarray(k)


def finite_horizon_dlqr(
    a: Array, b: Array, q: Array, r: Array, qf: Array, horizon: int
) -> tuple[Array, Array]:
    """Backward Riccati recursion for the finite-horizon discrete LQ problem.

    Returns time-varying gains ``gains`` of shape ``(H, m, n)`` (``u_k = -gains[k] x_k``) and the
    initial cost-to-go ``P_0`` (optimal cost ``0.5 x0ᵀ P_0 x0``).
    """
    a_np = np.asarray(a)
    b_np = np.asarray(b)
    q_np = np.asarray(q)
    r_np = np.asarray(r)
    p = np.asarray(qf, dtype=float).copy()
    gains: list[np.ndarray] = [np.zeros((b_np.shape[1], a_np.shape[0]))] * horizon
    for k in range(horizon - 1, -1, -1):
        s = r_np + b_np.T @ p @ b_np
        gain = np.linalg.solve(s, b_np.T @ p @ a_np)
        gains[k] = gain
        p = q_np + a_np.T @ p @ a_np - a_np.T @ p @ b_np @ gain
    return jnp.asarray(np.stack(gains)), jnp.asarray(p)


def dlqr_feedback_controls(dyn: Dynamics, x0: Array, gains: Array, dt: float) -> Array:
    """Simulate ``u_k = -gains[k] x_k`` and return the realised control sequence ``(H, m)``."""
    x = x0
    controls = []
    for gain in gains:
        u = -gain @ x
        controls.append(u)
        x = rk4_step(dyn, 0.0, x, u, dt)
    return jnp.stack(controls)
