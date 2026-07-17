"""Graph-residual gate: message passing extrapolates a coupling better than a pointwise MLP."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from chc import HybridDynamics, MLPResidual, fit_residual, one_step_mse, rk4_step
from chc.residual import GraphResidual

N = 10
DT = 0.05


def _ring(n: int) -> Array:
    i = jnp.arange(n)
    return jnp.zeros((n, n)).at[i, (i + 1) % n].set(1.0).at[i, (i - 1) % n].set(1.0)


class _Uncoupled(eqx.Module):
    """Known part: ``n`` independent damped oscillators; state is ``n`` blocks of (pos, vel)."""

    n: int = eqx.field(static=True)

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        nodes = x.reshape(self.n, 2)
        return jnp.stack([nodes[:, 1], -(nodes[:, 0]) - 0.1 * nodes[:, 1]], axis=1).reshape(-1)


class _Coupling(eqx.Module):
    """True residual: graph-Laplacian coupling on the acceleration."""

    adjacency: Array
    n: int = eqx.field(static=True)

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        pos = x.reshape(self.n, 2)[:, 0]
        coupling = 0.5 * (self.adjacency @ pos - jnp.sum(self.adjacency, axis=1) * pos)
        return jnp.stack([jnp.zeros(self.n), coupling], axis=1).reshape(-1)


def test_graph_residual_extrapolates_coupling_better_than_mlp() -> None:
    adjacency, known = _ring(N), _Uncoupled(N)
    plant = HybridDynamics(known=known, residual=_Coupling(adjacency, N))
    k = jax.random.split(jax.random.key(0), 3)
    x_tr = 0.4 * jax.random.normal(k[0], (200, 2 * N))  # trained on small amplitude
    u_tr = 0.2 * jax.random.normal(k[1], (200, 1))
    x_te = 2.5 * jax.random.normal(
        jax.random.key(9), (400, 2 * N)
    )  # tested far out (extrapolation)
    u_te = 0.2 * jax.random.normal(jax.random.key(8), (400, 1))
    xn = jax.vmap(lambda x, u: rk4_step(plant, 0.0, x, u, DT))
    data = {"x": x_tr, "u": u_tr, "x_next": xn(x_tr, u_tr)}
    x_next_te = xn(x_te, u_te)

    graph, _ = fit_residual(
        HybridDynamics(known=known, residual=GraphResidual(adjacency, 2, 1, hidden=16, key=k[2])),
        data,
        DT,
        steps=1500,
    )
    mlp, _ = fit_residual(
        HybridDynamics(known=known, residual=MLPResidual(2 * N, 1, 2 * N, width=32, key=k[2])),
        data,
        DT,
        steps=1500,
    )
    graph_mse = float(one_step_mse(graph, x_te, u_te, x_next_te, DT))
    mlp_mse = float(one_step_mse(mlp, x_te, u_te, x_next_te, DT))
    assert graph_mse < mlp_mse  # the graph inductive bias extrapolates the coupling better
