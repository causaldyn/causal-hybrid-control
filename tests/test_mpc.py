"""MPC gate: closed-loop receding-horizon control regulates the plant and respects constraints."""

import jax.numpy as jnp
import pytest

from chc import (
    DampedOscillator,
    HybridDynamics,
    QuadraticCost,
    ZeroResidual,
    mpc_control,
    rollout,
)

DT = 0.1


def test_mpc_regulates_oscillator() -> None:
    model = HybridDynamics(
        known=DampedOscillator(omega=1.0, zeta=0.1), residual=ZeroResidual(out_dim=2)
    )
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.1])),
        R=jnp.array([[0.05]]),
        Qf=jnp.diag(jnp.array([5.0, 1.0])),
        x_target=jnp.zeros(2),
    )
    x0 = jnp.array([1.0, 0.0])
    n_steps = 40

    xs, us = mpc_control(model, x0, cost, DT, horizon=20, u_lo=-5.0, u_hi=5.0, n_steps=n_steps)

    assert xs.shape == (n_steps + 1, 2)
    assert us.shape == (n_steps, 1)
    assert abs(float(xs[-1, 0])) < 0.15  # regulated to the target
    assert bool((jnp.abs(us) <= 5.0 + 1e-6).all())  # respects the box constraints
    assert float(jnp.max(jnp.abs(us))) > 0.1  # the controller actually acts

    # closed-loop tracking beats the open-loop (do-nothing) response
    xs_free = rollout(model, x0, jnp.zeros((n_steps, 1)), DT)
    assert float(jnp.sum(xs[:, 0] ** 2)) < float(jnp.sum(xs_free[:, 0] ** 2))


def test_mpc_refuses_a_target_that_would_not_move_with_the_loop() -> None:
    model = HybridDynamics(
        known=DampedOscillator(omega=1.0, zeta=0.1), residual=ZeroResidual(out_dim=2)
    )
    cost = QuadraticCost(
        Q=jnp.eye(2), R=jnp.array([[0.05]]), Qf=jnp.eye(2), x_target=jnp.zeros((21, 2))
    )
    with pytest.raises(ValueError, match="same window at every step"):
        mpc_control(model, jnp.array([1.0, 0.0]), cost, DT, 20, -5.0, 5.0, n_steps=3)
