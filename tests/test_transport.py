"""chc.transport: mass is conserved, Strang-Marchuk is 2nd-order, and control steers the density."""

import math

import jax
import jax.numpy as jnp

from chc.splitting import lie_trotter_step
from chc.transport import (
    MeanFieldTransport,
    _advect_flow,
    _react_flow,
    solve_transport,
)


def test_advection_conserves_total_mass() -> None:
    n = 64
    dx = 1.0 / n
    x = (jnp.arange(n) + 0.5) * dx
    rho0 = jnp.exp(-0.5 * ((x - 0.5) / 0.1) ** 2)
    v = 0.5 + 0.3 * jnp.sin(2 * jnp.pi * x)  # spatially varying velocity
    dt = 0.5 * dx / float(jnp.max(jnp.abs(v)))  # CFL-safe
    zero = jnp.zeros(n)
    final, _ = solve_transport(rho0, v, zero, zero, dx, dt, steps=200)
    assert abs(float(jnp.sum(final) - jnp.sum(rho0))) < 1e-9 * float(jnp.sum(rho0))


def test_strang_marchuk_is_second_order_in_time() -> None:
    n = 128
    dx = 1.0 / n
    x = (jnp.arange(n) + 0.5) * dx
    rho0 = jnp.exp(-0.5 * ((x - 0.5) / 0.1) ** 2)
    v = jnp.full(n, 0.4)  # constant advection...
    k = 1.0 + 0.5 * jnp.sin(2 * jnp.pi * x)  # ...but spatially varying reaction -> [A, R] != 0
    s = jnp.zeros(n)
    horizon_time = 0.4

    def strang(dt: float) -> jax.Array:
        final, _ = solve_transport(rho0, v, k, s, dx, dt, round(horizon_time / dt))
        return final

    def lie(dt: float) -> jax.Array:
        react, advect = _react_flow(k, s), _advect_flow(v, dx)

        def step(rho: jax.Array, _: jax.Array) -> tuple[jax.Array, None]:
            return lie_trotter_step(react, advect, rho, dt), None

        final, _ = jax.lax.scan(step, rho0, jnp.arange(round(horizon_time / dt)))
        return final

    reference = strang(0.00125)  # near-exact fine-dt reference on the same grid
    err = lambda sol: float(jnp.linalg.norm(sol - reference))  # noqa: E731
    strang_order = math.log2(err(strang(0.01)) / err(strang(0.005)))
    lie_order = math.log2(err(lie(0.01)) / err(lie(0.005)))
    assert strang_order > 1.7  # Strang-Marchuk splitting is 2nd-order in time
    assert strang_order > lie_order + 0.5  # and clearly beats Lie-Trotter's ~1st order


def test_mean_field_transport_steers_the_density_to_target() -> None:
    result = MeanFieldTransport(horizon=40, n_cells=64).mismatches(steps=200)
    assert result["controlled-CHC"] < 0.4 * result["uncontrolled"]  # control moves mass to target
