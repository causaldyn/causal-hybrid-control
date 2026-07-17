"""Deep Galerkin gate: the neural PDE solver recovers the analytic 1-D Poisson solution."""

import jax
import jax.numpy as jnp

from chc.deep_galerkin import solve_poisson_dgm


def test_deep_galerkin_recovers_the_poisson_solution() -> None:
    # -V'' = pi^2 sin(pi x), V(0)=V(1)=0  ->  V(x) = sin(pi x)
    def source(x: jax.Array) -> jax.Array:
        return jnp.pi**2 * jnp.sin(jnp.pi * x)

    model = solve_poisson_dgm(source, steps=3000)
    xs = jnp.linspace(0.0, 1.0, 21)
    predicted = jnp.array([model(x) for x in xs])
    rmse = float(jnp.sqrt(jnp.mean((predicted - jnp.sin(jnp.pi * xs)) ** 2)))
    assert rmse < 0.05  # the neural Galerkin matches the analytic (and FEM) solution
