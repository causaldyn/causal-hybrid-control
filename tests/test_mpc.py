"""MPC gate: closed-loop receding-horizon control regulates the plant and respects constraints."""

import equinox as eqx
import jax.numpy as jnp
import pytest
from jax import Array

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


class _Sine(eqx.Module):
    """``x' = -x + u + 2 sin(3 t)``: a plant that moves in time."""

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return -x + u + 2.0 * jnp.sin(3.0 * t) * jnp.ones_like(x)


class _Onset(eqx.Module):
    """``x' = u + p(t)``, a push rising from 0 to 5 around ``t = 1``: invisible to a window of
    0.5 s until the loop is within half a second of it."""

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return u + 2.5 * (1.0 + jnp.tanh((t - 1.0) / 0.05)) * jnp.ones_like(x)


SCALAR = QuadraticCost(Q=jnp.eye(1), R=jnp.array([[0.01]]), Qf=jnp.eye(1), x_target=jnp.zeros(1))


def test_mpc_steps_its_plant_on_the_loops_clock() -> None:
    # The realised trajectory is the plant's own rollout of the applied actions exactly when step k
    # is taken at t = k * dt; stepping every action at t = 0 left the forcing at sin(0) = 0.
    x0 = jnp.array([0.5])
    xs, us = mpc_control(_Sine(), x0, SCALAR, DT, horizon=5, u_lo=-50.0, u_hi=50.0, n_steps=12)
    assert float(jnp.max(jnp.abs(xs - rollout(_Sine(), x0, us, DT)))) < 1e-10


def test_mpc_plans_each_window_from_the_time_it_starts() -> None:
    xs, us = mpc_control(
        _Onset(), jnp.zeros(1), SCALAR, DT, horizon=5, u_lo=-50.0, u_hi=50.0, n_steps=14
    )
    assert abs(float(us[0, 0])) < 1e-3  # at t = 0 the push is a second away, past the window
    assert float(us[7, 0]) < -0.3  # at t = 0.7 the window reaches it, and the plan leans into it
    assert float(xs[8, 0]) < -0.05  # the state moves before the push, which is 0.002 at t = 0.8
