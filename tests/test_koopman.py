"""Koopman gate: an EDMD lift makes the nonlinear system linear enough to predict and control."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chc import DampedOscillator, HybridDynamics, rk4_step
from chc.koopman import KoopmanModel, koopman_controller, koopman_lqr_gain

DT = 0.05


class _Cubic(eqx.Module):
    beta: float

    def __call__(self, t: float, x: jax.Array, u: jax.Array) -> jax.Array:
        return jnp.array([0.0, -self.beta * x[0] ** 3])


def _plant() -> HybridDynamics:
    return HybridDynamics(known=DampedOscillator(omega=1.0, zeta=0.1), residual=_Cubic(beta=0.5))


def _transitions(n: int = 3000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_x, k_u = jax.random.split(jax.random.key(0))
    xs = jax.random.normal(k_x, (n, 2))
    us = 0.5 * jax.random.normal(k_u, (n, 1))
    x_next = jax.vmap(lambda x, u: rk4_step(_plant(), 0.0, x, u, DT))(xs, us)
    return np.asarray(xs), np.asarray(us), np.asarray(x_next)


def test_koopman_predicts_the_lifted_nonlinear_dynamics() -> None:
    xs, us, x_next = _transitions()
    model = KoopmanModel(degree=3).fit(xs, us, x_next)
    rmse = float(np.sqrt(np.mean((model.predict(xs, us) - x_next) ** 2)))
    assert rmse < 0.01  # the polynomial lift makes the cubic oscillator near-linear


def test_koopman_lqr_regulates_the_true_system_to_target() -> None:
    xs, us, x_next = _transitions()
    model = KoopmanModel(degree=3).fit(xs, us, x_next)
    gain = koopman_lqr_gain(model, np.diag([10.0, 1.0]), np.array([[0.1]]))
    control = koopman_controller(model, gain, np.array([1.0, 0.0]))
    plant, x = _plant(), np.array([0.0, 0.0])
    for _ in range(80):
        u = np.clip(control(x), -10.0, 10.0)
        x = np.asarray(rk4_step(plant, 0.0, jnp.asarray(x), jnp.asarray(u), DT))
    assert (
        x[0] > 0.7
    )  # LQR on the Koopman matrices drives the true plant toward target position 1.0
