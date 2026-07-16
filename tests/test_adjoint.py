"""Gradient-check gate: discrete adjoint == autodiff == finite differences."""

import jax
import jax.numpy as jnp

from chc import (
    DampedOscillator,
    HybridDynamics,
    QuadraticCost,
    ZeroResidual,
    control_gradient_adjoint,
    total_cost,
)

DT = 0.1


def _setup() -> tuple[HybridDynamics, QuadraticCost, jax.Array, jax.Array]:
    dyn = HybridDynamics(
        known=DampedOscillator(omega=1.0, zeta=0.15), residual=ZeroResidual(out_dim=2)
    )
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.05])),
        R=jnp.array([[0.02]]),
        Qf=jnp.diag(jnp.array([5.0, 1.0])),
        x_target=jnp.zeros(2),
    )
    x0 = jnp.array([1.0, 0.0])
    us = 0.5 * jax.random.normal(jax.random.key(1), (30, 1))
    return dyn, cost, x0, us


def test_adjoint_matches_autodiff() -> None:
    dyn, cost, x0, us = _setup()
    g_adjoint = control_gradient_adjoint(dyn, x0, us, DT, cost)
    g_autodiff = jax.grad(lambda u: total_cost(dyn, x0, u, DT, cost))(us)
    assert jnp.allclose(g_adjoint, g_autodiff, atol=1e-8, rtol=1e-6)


def test_adjoint_matches_finite_difference() -> None:
    dyn, cost, x0, us = _setup()
    g_adjoint = control_gradient_adjoint(dyn, x0, us, DT, cost)
    eps = 1e-6
    for k in (0, 7, 15, 29):
        pert = jnp.zeros_like(us).at[k, 0].set(eps)
        fd = (
            total_cost(dyn, x0, us + pert, DT, cost) - total_cost(dyn, x0, us - pert, DT, cost)
        ) / (2 * eps)
        assert jnp.allclose(g_adjoint[k, 0], fd, atol=1e-5, rtol=1e-4)
