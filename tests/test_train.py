"""System-ID gate: fitting the residual recovers hidden physics the known model misses."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from chc import DampedOscillator, HybridDynamics, MLPResidual, ZeroResidual, rk4_step, rollout
from chc.train import fit_residual, fit_residual_multistep, one_step_mse, rollout_mse

DT = 0.05


class CubicResidual(eqx.Module):
    """Ground-truth hidden term: a cubic stiffening force in the acceleration channel."""

    beta: float

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return jnp.array([0.0, -self.beta * x[0] ** 3])


def test_fit_residual_recovers_hidden_physics() -> None:
    known = DampedOscillator(omega=1.0, zeta=0.1)
    true_plant = HybridDynamics(known=known, residual=CubicResidual(beta=0.5))

    k_x, k_u = jax.random.split(jax.random.key(0))
    xs = jax.random.normal(k_x, (2000, 2))
    us = 0.5 * jax.random.normal(k_u, (2000, 1))
    x_next = jax.vmap(lambda x, u: rk4_step(true_plant, 0.0, x, u, DT))(xs, us)
    data = {"x": xs, "u": us, "x_next": x_next}

    known_only = HybridDynamics(known=known, residual=ZeroResidual(out_dim=2))
    mse_known = float(one_step_mse(known_only, xs, us, x_next, DT))

    init = HybridDynamics(
        known=known,
        residual=MLPResidual(
            state_dim=2, control_dim=1, out_dim=2, width=32, key=jax.random.key(1)
        ),
    )
    trained, history = fit_residual(init, data, DT, steps=2000, lr=1e-2)
    mse_trained = float(one_step_mse(trained, xs, us, x_next, DT))

    assert float(history[-1]) < float(history[0])  # training reduced the loss
    assert mse_trained < 0.2 * mse_known  # learning the residual cuts error >5x


def test_multistep_training_reduces_rollout_drift() -> None:
    known = DampedOscillator(omega=1.0, zeta=0.1)
    true_plant = HybridDynamics(known=known, residual=CubicResidual(beta=0.5))
    horizon, n_traj = 15, 300

    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    x0s = 0.8 * jax.random.normal(k0, (n_traj, 2))
    us = 0.3 * jax.random.normal(k1, (n_traj, horizon, 1))
    xs_clean = jax.vmap(lambda x0, u: rollout(true_plant, x0, u, DT))(x0s, us)
    xs = xs_clean + 0.02 * jax.random.normal(k2, xs_clean.shape)  # observation noise induces drift

    def fresh() -> HybridDynamics:
        return HybridDynamics(
            known=known, residual=MLPResidual(2, 1, 2, width=32, key=jax.random.key(1))
        )

    one_step_data = {
        "x": xs[:, :-1, :].reshape(-1, 2),
        "u": us.reshape(-1, 1),
        "x_next": xs[:, 1:, :].reshape(-1, 2),
    }
    m_one, _ = fit_residual(fresh(), one_step_data, DT, steps=1500, lr=1e-2)
    m_multi, _ = fit_residual_multistep(
        fresh(), {"x0": xs[:, 0, :], "us": us, "xs": xs}, DT, steps=1500, lr=1e-2
    )

    # evaluate on held-out CLEAN trajectories
    e0, e1 = jax.random.split(jax.random.key(9))
    ex0 = 0.8 * jax.random.normal(e0, (100, 2))
    eus = 0.3 * jax.random.normal(e1, (100, horizon, 1))
    exs = jax.vmap(lambda x0, u: rollout(true_plant, x0, u, DT))(ex0, eus)

    err_one = float(rollout_mse(m_one, ex0, eus, exs, DT))
    err_multi = float(rollout_mse(m_multi, ex0, eus, exs, DT))
    assert err_multi < 0.5 * err_one  # multi-step training reduces rollout drift
