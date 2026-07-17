"""Deep Galerkin Method -- a neural PDE solver, meeting classical Galerkin/Marchuk FEM on one PDE.

The Deep Galerkin Method (Sirignano-Spiliopoulos) trains a network to satisfy a PDE by minimising
its residual at random points -- a mesh-free Galerkin scheme. Here it solves the same 1-D Poisson
BVP ``-V''(x) = f(x)``, ``V(0)=V(1)=0`` that ``chc.galerkin`` solves with a variational-difference
FEM (progonka), so the *neural* Galerkin can be checked against the analytic and the *classical*
one. The bridge from ``plans/01`` (Marchuk/Galerkin) to learning-based PDE solvers and mean-field
control (cf. Deep Galerkin for MFC, arXiv 2405.13346). Scope: a focused 1-D demo, not high-dim.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array


class ScalarMLP(eqx.Module):
    """A small tanh MLP ``x -> V(x)`` (scalar in, scalar out) for a 1-D field."""

    layers: list

    def __init__(self, width: int, key: Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(1, width, key=k1),
            eqx.nn.Linear(width, width, key=k2),
            eqx.nn.Linear(width, 1, key=k3),
        ]

    def __call__(self, x: Array) -> Array:
        h = jnp.atleast_1d(x)
        for lin in self.layers[:-1]:
            h = jax.nn.tanh(lin(h))
        return self.layers[-1](h)[0]


def solve_poisson_dgm(
    source: Callable[[Array], Array],
    width: int = 32,
    steps: int = 4000,
    n_collocation: int = 128,
    seed: int = 0,
) -> ScalarMLP:
    """Deep Galerkin solve of ``-V''(x) = source(x)`` on ``[0,1]``, ``V(0)=V(1)=0``.

    Minimises the mean-squared PDE residual at random collocation points plus the boundary term.
    """
    key = jax.random.key(seed)
    model = ScalarMLP(width, key)

    def second_derivative(m: ScalarMLP, x: Array) -> Array:
        return jax.grad(jax.grad(m.__call__))(x)

    def loss(m: ScalarMLP, xs: Array) -> Array:
        residual = jax.vmap(lambda x: second_derivative(m, x) + source(x))(xs)  # -V'' = f
        boundary = m(jnp.array(0.0)) ** 2 + m(jnp.array(1.0)) ** 2
        return jnp.mean(residual**2) + boundary

    optimizer = optax.adam(2e-3)
    state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(
        m: ScalarMLP, opt_state: optax.OptState, xs: Array
    ) -> tuple[ScalarMLP, optax.OptState]:
        grads = eqx.filter_grad(loss)(m, xs)
        updates, opt_state = optimizer.update(grads, opt_state)
        return eqx.apply_updates(m, updates), opt_state

    for _ in range(steps):
        key, sample_key = jax.random.split(key)
        xs = jax.random.uniform(sample_key, (n_collocation,))
        model, state = step(model, state, xs)
    return model
