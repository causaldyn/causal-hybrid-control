"""chc.transport: mass is conserved, Strang-Marchuk is 2nd-order, and control steers the density."""

import math

import jax
import jax.numpy as jnp

from chc.splitting import lie_trotter_step
from chc.toeplitz import circulant_matvec
from chc.transport import (
    MeanFieldTransport,
    _advect_flow,
    _react_flow,
    advection_diffusion_field,
    advection_diffusion_kernel,
    advection_diffusion_propagator,
    advection_diffusion_symbol,
    periodic_smoothing_kernel,
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


def test_advection_diffusion_field_matches_a_single_analytic_mode() -> None:
    # The spectral operator is exact, so on one Fourier mode it must reproduce the closed form
    # -c k cos(kx) - nu k^2 sin(kx) with no discretisation error to hide behind.
    n, length, speed, viscosity = 64, 1.0, 0.8, 0.01
    x = jnp.linspace(0.0, length, n, endpoint=False)
    k = 2.0 * jnp.pi / length
    field = advection_diffusion_field(jnp.sin(k * x), length, speed=speed, viscosity=viscosity)
    analytic = -speed * k * jnp.cos(k * x) - viscosity * k**2 * jnp.sin(k * x)
    assert float(jnp.max(jnp.abs(field - analytic))) < 1e-4 * float(speed * k)


def test_pure_advection_propagator_is_an_exact_translation() -> None:
    # validation/spectral_circulant.mac STEP 5e: at nu = 0 every mode has |lambda| = 1 and the
    # propagator is an isometry -- it moves the wave and leaves the amplitude alone.
    n, length, speed, dt = 128, 1.0, 0.8, 0.25
    x = jnp.linspace(0.0, length, n, endpoint=False)
    k = 2.0 * jnp.pi / length
    moved = advection_diffusion_propagator(
        jnp.sin(k * x), length, speed=speed, viscosity=0.0, dt=dt
    )
    assert float(jnp.max(jnp.abs(moved - jnp.sin(k * (x - speed * dt))))) < 1e-5


def test_advection_diffusion_kernel_realises_the_field_as_a_circulant() -> None:
    # The plant IS a circulant, which is what puts the truth inside SpectralResidual's class.
    n, length, speed, viscosity = 64, 1.0, 0.8, 0.01
    kernel = advection_diffusion_kernel(n, length, speed=speed, viscosity=viscosity)
    u = jax.random.normal(jax.random.PRNGKey(0), (n,))
    direct = advection_diffusion_field(u, length, speed=speed, viscosity=viscosity)
    assert float(jnp.max(jnp.abs(circulant_matvec(kernel, u) - direct))) < 1e-3


def test_the_nyquist_advection_bin_is_zero_on_an_even_grid() -> None:
    # validation/spectral_circulant.mac STEP 7: the Nyquist mode is (-1)^j, whose exact derivative
    # vanishes at every grid point, so the correct discrete symbol of d/dx there is 0. That is also
    # what keeps the operator a REAL circulant, since a real first column forces a real eigenvalue.
    n, length = 64, 1.0
    advection_only = advection_diffusion_symbol(n, length, speed=0.8, viscosity=0.0)
    assert float(jnp.abs(advection_only[-1])) < 1e-12
    assert float(jnp.abs(advection_only[n // 2 - 1])) > 1.0  # and the bin below it is not zero
    both = advection_diffusion_symbol(n, length, speed=0.8, viscosity=0.01)
    assert (
        float(jnp.abs(jnp.imag(both[-1]))) < 1e-12
    )  # real at Nyquist, as a real circulant must be
    assert float(jnp.real(both[-1])) < -100.0  # diffusion keeps its Nyquist bin, and it is large


def test_smoothing_kernel_is_a_nonlocal_unit_gain_actuator() -> None:
    # A diagonal control matrix cannot spread; a circulant one can, which is why the certificate
    # gives the control channel a kernel rather than a scalar.
    n, length = 64, 1.0
    kernel = periodic_smoothing_kernel(n, length, 0.04)
    assert abs(float(jnp.sum(kernel)) - 1.0) < 1e-5  # unit gain at k = 0: no net mass injected
    assert float(kernel[1]) > 0.05  # genuinely spreads to the neighbour
    impulse = jnp.zeros(n).at[n // 2].set(1.0)
    spread = circulant_matvec(kernel, impulse)
    assert int(jnp.sum(spread > 1e-3)) > 3  # one cell of control reaches several cells of state
