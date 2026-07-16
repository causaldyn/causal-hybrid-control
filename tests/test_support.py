"""Pessimism gate: support penalty keeps control near the data; greedy control extrapolates."""

import jax
import jax.numpy as jnp

from chc import (
    DampedOscillator,
    HybridDynamics,
    QuadraticCost,
    ZeroResidual,
    projected_gradient_control,
    rollout,
)
from chc.support import SupportModel, pessimistic_control

DT = 0.1


def test_pessimism_keeps_control_in_support() -> None:
    k_x, k_u = jax.random.split(jax.random.key(0))
    xs_data = jax.random.normal(k_x, (2000, 2))  # x ~ N(0, I)
    us_data = 0.3 * jax.random.normal(k_u, (2000, 1))  # u ~ N(0, 0.3^2): narrow support
    support = SupportModel.fit(xs_data, us_data)

    model = HybridDynamics(
        known=DampedOscillator(omega=1.0, zeta=0.1), residual=ZeroResidual(out_dim=2)
    )
    # cheap control + a far target ⇒ the greedy optimum slams the actuator far beyond the u-support
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.0])),
        R=jnp.array([[0.001]]),
        Qf=jnp.diag(jnp.array([10.0, 1.0])),
        x_target=jnp.array([-3.0, 0.0]),
    )
    x0 = jnp.zeros(2)
    us0 = jnp.zeros((20, 1))

    us_greedy, _ = projected_gradient_control(model, x0, us0, DT, cost, -5.0, 5.0, steps=200)
    us_pess, _ = pessimistic_control(
        model, x0, us0, DT, cost, support, lam_supp=20.0, u_lo=-5.0, u_hi=5.0, steps=200
    )

    greedy_max = float(jnp.max(jnp.abs(us_greedy)))
    pess_max = float(jnp.max(jnp.abs(us_pess)))
    assert greedy_max > 2.0  # greedy extrapolates far past the u-support (~0.3)
    assert pess_max < 0.6 * greedy_max  # pessimism shrinks control toward the data

    xs_greedy = rollout(model, x0, us_greedy, DT)
    xs_pess = rollout(model, x0, us_pess, DT)
    assert float(jnp.max(jnp.abs(xs_pess[:, 0]))) < float(jnp.max(jnp.abs(xs_greedy[:, 0])))
