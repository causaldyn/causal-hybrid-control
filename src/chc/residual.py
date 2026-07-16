"""Learned residual backends (Strategy). MLP here; KAN / RBF / GP are future backends."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


class ZeroResidual(eqx.Module):
    """A residual that contributes nothing — recovers the pure known dynamics."""

    out_dim: int = eqx.field(static=True)

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return jnp.zeros((self.out_dim,))


class MLPResidual(eqx.Module):
    """Learned residual ``r_θ(x, u)`` backed by an MLP over the concatenated ``[x, u]``.

    One interchangeable backend; the point of the abstraction is that KAN / RBF / GP slot in here
    without touching dynamics or control.
    """

    mlp: eqx.nn.MLP

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        out_dim: int,
        width: int = 16,
        depth: int = 2,
        *,
        key: Array,
    ) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=state_dim + control_dim,
            out_size=out_dim,
            width_size=width,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return self.mlp(jnp.concatenate([x, u]))
