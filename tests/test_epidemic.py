"""Epidemic flagship gate: optimal NPI flattens the curve under the capacity constraint."""

import jax.numpy as jnp

from chc.epidemic import SIRDynamics, optimal_npi
from chc.integrate import rollout

DT = 1.0
I_MAX = 0.1
U_MAX = 0.9


def test_optimal_npi_flattens_the_curve() -> None:
    model = SIRDynamics(beta=0.6, gamma=0.1)  # R0 = 6
    x0 = jnp.array([0.99, 0.01])
    horizon = 100

    peak_uncontrolled = float(jnp.max(rollout(model, x0, jnp.zeros((horizon, 1)), DT)[:, 1]))
    us = optimal_npi(model, x0, DT, horizon, I_MAX, u_max=U_MAX, steps=400)
    peak_controlled = float(jnp.max(rollout(model, x0, us, DT)[:, 1]))

    assert peak_uncontrolled > 2 * I_MAX  # uncontrolled epidemic overshoots capacity
    assert peak_controlled <= I_MAX * 1.15  # control flattens the curve to ~capacity
    assert float(jnp.sum(us)) > 0.0  # it actually intervenes
    assert bool((us >= 0.0).all())  # within the lower box bound
    assert bool((us <= U_MAX + 1e-6).all())  # within the upper box bound
