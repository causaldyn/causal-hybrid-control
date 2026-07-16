"""1D Galerkin finite elements with a tridiagonal (Thomas / progonka) solve.

Weak form of ``-u'' = f`` on ``[0,1]`` with ``u(0)=u(1)=0`` and a piecewise-linear hat basis: the 1D
analogue of the user's 2D bilinear coursework (``plans/11`` §5). Hat stiffness assembles to the
tridiagonal stencil ``(1/h)[-1, 2, -1]``, solved by the Thomas sweep (the "progonka" kernel of
Marchuk-Agoshkov projection-grid methods). Seeds the weak-form / Galerkin track (``plans/01`` §3.3).
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array, lax


def thomas_solve(sub: Array, diag: Array, sup: Array, rhs: Array) -> Array:
    """Solve a tridiagonal system (Thomas / progonka). ``sub[0]`` and ``sup[-1]`` are unused."""

    def forward(
        carry: tuple[Array, Array], inp: tuple[Array, Array, Array, Array]
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        c_prev, d_prev = carry
        sub_i, diag_i, sup_i, rhs_i = inp
        m = diag_i - sub_i * c_prev
        c_i = sup_i / m
        d_i = (rhs_i - sub_i * d_prev) / m
        return (c_i, d_i), (c_i, d_i)

    _, (cs, ds) = lax.scan(forward, (jnp.zeros(()), jnp.zeros(())), (sub, diag, sup, rhs))

    def backward(x_next: Array, inp: tuple[Array, Array]) -> tuple[Array, Array]:
        c_i, d_i = inp
        x_i = d_i - c_i * x_next
        return x_i, x_i

    _, x_head = lax.scan(backward, ds[-1], (cs[:-1], ds[:-1]), reverse=True)
    return jnp.concatenate([x_head, ds[-1:]])


def poisson_1d(f: Callable[[Array], Array], n: int) -> tuple[Array, Array]:
    """FEM solution of ``-u'' = f`` on ``[0,1]``, ``u(0)=u(1)=0``, on ``n`` uniform elements.

    Returns the node coordinates (length ``n+1``) and the nodal solution (with the zero boundaries).
    """
    h = 1.0 / n
    nodes = jnp.linspace(0.0, 1.0, n + 1)
    interior = nodes[1:-1]  # n-1 unknowns
    diag = jnp.full(n - 1, 2.0 / h)
    off = jnp.full(n - 1, -1.0 / h)
    sub = off.at[0].set(0.0)
    sup = off.at[-1].set(0.0)
    rhs = f(interior) * h  # lumped load, O(h^2) like the FEM itself
    u_interior = thomas_solve(sub, diag, sup, rhs)
    return nodes, jnp.concatenate([jnp.zeros(1), u_interior, jnp.zeros(1)])
