"""Learned residual backends (Strategy): MLP and an RBF Kolmogorov-Arnold layer; GP is future."""

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


class RBFKANLayer(eqx.Module):
    """One Kolmogorov-Arnold layer with RBF edge functions (FastKAN-style).

    Each input-output edge is a learnable 1D map ``phi(x) = sum_g c_g rbf_g(x) + w silu(x)`` over a
    fixed radial-basis grid; the output is ``bias_j + sum_i phi_{ji}(z_i)``. Each edge is an
    extractable 1D curve (interpretable) and cheap to evaluate.
    """

    in_dim: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    num_grid: int = eqx.field(static=True)
    grid_range: float = eqx.field(static=True)
    coeff: Array
    base_weight: Array
    bias: Array

    def __init__(
        self, in_dim: int, out_dim: int, num_grid: int, grid_range: float, *, key: Array
    ) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_grid = num_grid
        self.grid_range = grid_range
        k_coeff, k_base = jax.random.split(key)
        self.coeff = (in_dim * num_grid) ** -0.5 * jax.random.normal(
            k_coeff, (out_dim, in_dim, num_grid)
        )
        self.base_weight = in_dim**-0.5 * jax.random.normal(k_base, (out_dim, in_dim))
        self.bias = jnp.zeros(out_dim)

    def __call__(self, z: Array) -> Array:
        grid = jnp.linspace(-self.grid_range, self.grid_range, self.num_grid)
        inv_h = (self.num_grid - 1) / (2.0 * self.grid_range)
        phi = jnp.exp(-(((z[:, None] - grid[None, :]) * inv_h) ** 2))  # (in_dim, num_grid)
        spline = jnp.einsum("oig,ig->o", self.coeff, phi)
        return self.bias + spline + self.base_weight @ jax.nn.silu(z)


class KANResidual(eqx.Module):
    """Learned residual ``r_θ(x, u)`` backed by a Kolmogorov-Arnold layer (RBF edges).

    A drop-in :class:`ResidualModel` alternative to :class:`MLPResidual`; interpretable, and the
    same training / adjoint machinery applies unchanged.
    """

    layer: RBFKANLayer

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        out_dim: int,
        num_grid: int = 8,
        grid_range: float = 3.0,
        *,
        key: Array,
    ) -> None:
        self.layer = RBFKANLayer(state_dim + control_dim, out_dim, num_grid, grid_range, key=key)

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return self.layer(jnp.concatenate([x, u]))


class GraphResidual(eqx.Module):
    """Message-passing residual over a fixed graph -- a GNN backend for networked/spatial dynamics.

    The state is ``n_nodes`` blocks of ``node_dim``. Each node's update is an MLP of its features,
    the mean of its neighbours' encoded features (one message-passing round), and the control. It is
    permutation-equivariant and parameter-shared, learning a coupling a pointwise MLP re-learns per
    node. The adjacency is frozen (``stop_gradient``); see ``plans/16``.
    """

    adjacency: Array
    n_nodes: int = eqx.field(static=True)
    node_dim: int = eqx.field(static=True)
    encoder: eqx.nn.MLP
    message: eqx.nn.MLP

    def __init__(
        self, adjacency: Array, node_dim: int, control_dim: int, hidden: int = 16, *, key: Array
    ) -> None:
        k_enc, k_msg = jax.random.split(key)
        degree = jnp.sum(adjacency, axis=1, keepdims=True)
        self.adjacency = adjacency / jnp.maximum(degree, 1.0)  # row-normalised (mean aggregation)
        self.n_nodes = adjacency.shape[0]
        self.node_dim = node_dim
        self.encoder = eqx.nn.MLP(node_dim, hidden, hidden, 1, activation=jax.nn.tanh, key=k_enc)
        self.message = eqx.nn.MLP(
            node_dim + hidden + control_dim, node_dim, hidden, 1, activation=jax.nn.tanh, key=k_msg
        )

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        adjacency = jax.lax.stop_gradient(self.adjacency)  # graph structure is fixed, not trained
        nodes = x.reshape(self.n_nodes, self.node_dim)
        messages = adjacency @ jax.vmap(self.encoder)(nodes)
        control = jnp.broadcast_to(u, (self.n_nodes, u.shape[0]))
        update = jax.vmap(self.message)(jnp.concatenate([nodes, messages, control], axis=1))
        return update.reshape(-1)
