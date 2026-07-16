"""Operator splitting (Lie-Trotter, Strang-Marchuk): integrate the known linear part exactly.

For ``ẋ = A x + r(x)`` the known linear (often stiff) operator is advanced by its exact flow
``exp(A dt)`` and the learned residual ``r`` by RK4. Strang-Marchuk composes them symmetrically for
2nd-order accuracy, so the network need not represent the stiff known dynamics (``plans/01`` §3.1).
This is where the "Marchuk" framing earns its keep.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.scipy.linalg as jsla
from jax import Array

from chc.dynamics import Dynamics
from chc.integrate import rk4_step

Flow = Callable[[Array, float], Array]


def exact_linear_flow(a_matrix: Array) -> Flow:
    """Exact flow of ``ẋ = A x``: returns ``(x, dt) -> expm(A dt) @ x``."""

    def flow(x: Array, dt: float) -> Array:
        return jsla.expm(a_matrix * dt) @ x

    return flow


def residual_flow(field: Dynamics, u: Array) -> Flow:
    """RK4 flow of ``ẋ = field(x, u)`` at a fixed control ``u``."""

    def flow(x: Array, dt: float) -> Array:
        return rk4_step(field, 0.0, x, u, dt)

    return flow


def lie_trotter_step(flow_a: Flow, flow_b: Flow, x: Array, dt: float) -> Array:
    """First-order split: ``S_A(dt) ∘ S_B(dt)``."""
    return flow_a(flow_b(x, dt), dt)


def strang_marchuk_step(flow_a: Flow, flow_b: Flow, x: Array, dt: float) -> Array:
    """Second-order symmetric split: ``S_A(dt/2) ∘ S_B(dt) ∘ S_A(dt/2)``."""
    half = flow_a(x, 0.5 * dt)
    full = flow_b(half, dt)
    return flow_a(full, 0.5 * dt)
