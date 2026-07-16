"""Galerkin/FEM gate: the Thomas (progonka) solve is exact; 1D FEM converges at 2nd order."""

import jax
import jax.numpy as jnp
from jax import Array

from chc.galerkin import poisson_1d, poisson_2d, thomas_solve


def _f(x: Array) -> Array:
    return (jnp.pi**2) * jnp.sin(jnp.pi * x)


def _u_exact(x: Array) -> Array:
    return jnp.sin(jnp.pi * x)


def _f2(x: Array, y: Array) -> Array:
    return 2.0 * (jnp.pi**2) * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)


def _u_exact2(x: Array, y: Array) -> Array:
    return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)


def test_thomas_matches_dense_solve() -> None:
    n = 20
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    sub = jnp.concatenate([jnp.zeros(1), jax.random.uniform(k1, (n - 1,))])
    sup = jnp.concatenate([jax.random.uniform(k2, (n - 1,)), jnp.zeros(1)])
    diag = 3.0 + jax.random.uniform(k3, (n,))  # diagonally dominant
    rhs = jax.random.uniform(k4, (n,))
    dense = jnp.diag(diag) + jnp.diag(sup[:-1], 1) + jnp.diag(sub[1:], -1)
    assert jnp.allclose(thomas_solve(sub, diag, sup, rhs), jnp.linalg.solve(dense, rhs), atol=1e-10)


def test_poisson_1d_fem_second_order() -> None:
    ns = [16, 32, 64, 128]
    errors = [
        float(jnp.max(jnp.abs(u - _u_exact(nodes)))) for nodes, u in (poisson_1d(_f, n) for n in ns)
    ]
    assert errors[-1] < 1e-4
    hs = jnp.array([1.0 / n for n in ns])
    slope = float(jnp.polyfit(jnp.log(hs), jnp.log(jnp.array(errors)), 1)[0])
    assert 1.8 < slope < 2.2


def test_poisson_2d_fem_second_order() -> None:
    ns = [8, 16, 32]
    errors = []
    for n in ns:
        coords, u = poisson_2d(_f2, n)
        xs, ys = jnp.meshgrid(coords, coords, indexing="ij")
        errors.append(float(jnp.max(jnp.abs(u - _u_exact2(xs, ys)))))
    assert errors[-1] < 5e-3
    hs = jnp.array([1.0 / n for n in ns])
    slope = float(jnp.polyfit(jnp.log(hs), jnp.log(jnp.array(errors)), 1)[0])
    assert 1.8 < slope < 2.2
