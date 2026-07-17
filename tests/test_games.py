"""Game primitives: simplex projection, congestion equilibrium, Stackelberg allocation."""

import jax.numpy as jnp

from chc.games import project_simplex, softmax_congestion_equilibrium, stackelberg_allocation


def test_project_simplex_lands_on_the_budget_simplex() -> None:
    projected = project_simplex(jnp.array([2.0, -1.0, 0.5, 3.0]), 1.0)
    assert abs(float(jnp.sum(projected)) - 1.0) < 1e-5
    assert bool(jnp.all(projected >= -1e-6))


def test_equilibrium_conserves_mass_and_favours_attractive_zones() -> None:
    x = softmax_congestion_equilibrium(
        jnp.array([1.0, 0.5, 0.2]), jnp.zeros(3), congestion=1.0, mass=6.0
    )
    assert abs(float(jnp.sum(x)) - 6.0) < 1e-3  # mass is conserved
    assert float(x[0]) > float(x[2])  # the more attractive zone holds more agents


def test_stackelberg_allocation_beats_uniform_on_a_concave_objective() -> None:
    a = jnp.array([0.1, 0.5, 2.0])

    def objective(u: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(jnp.sqrt(a + u))  # concave -> water-filling optimum

    u = stackelberg_allocation(objective, 3, budget=1.0, steps=300)
    assert abs(float(jnp.sum(u)) - 1.0) < 1e-3
    assert float(objective(u)) >= float(objective(jnp.full(3, 1 / 3)))
