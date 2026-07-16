"""Optimal-control gate: projected gradient reduces cost and drives the state toward target."""

import jax.numpy as jnp

from chc import (
    DampedOscillator,
    HybridDynamics,
    QuadraticCost,
    ZeroResidual,
    projected_gradient_control,
    rollout,
)

DT = 0.1


def test_projected_gradient_reduces_cost_and_reaches_target() -> None:
    dyn = HybridDynamics(
        known=DampedOscillator(omega=1.0, zeta=0.1), residual=ZeroResidual(out_dim=2)
    )
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.0])),
        R=jnp.array([[0.01]]),
        Qf=jnp.diag(jnp.array([10.0, 1.0])),
        x_target=jnp.zeros(2),
    )
    x0 = jnp.array([1.0, 0.0])
    us0 = jnp.zeros((50, 1))

    us, history = projected_gradient_control(dyn, x0, us0, DT, cost, u_lo=-5.0, u_hi=5.0, steps=150)

    assert history[-1] < 0.7 * history[0]  # meaningful cost reduction
    assert bool((jnp.abs(us) <= 5.0 + 1e-6).all())  # respects box constraints
    xs = rollout(dyn, x0, us, DT)
    assert abs(float(xs[-1, 0])) < abs(float(x0[0]))  # ends closer to target position
