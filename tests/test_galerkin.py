"""Galerkin/FEM gate: the Thomas (progonka) solve is exact; 1D FEM converges at 2nd order."""

import jax
import jax.numpy as jnp
from jax import Array

from chc.galerkin import (
    convection_diffusion_1d,
    convection_diffusion_certificate,
    convection_diffusion_exact,
    optimal_upwind,
    poisson_1d,
    poisson_2d,
    thomas_solve,
)


def _f(x: Array) -> Array:
    return (jnp.pi**2) * jnp.sin(jnp.pi * x)


def _u_exact(x: Array) -> Array:
    return jnp.sin(jnp.pi * x)


def _f2(x: Array, y: Array) -> Array:
    return 2.0 * (jnp.pi**2) * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)


def _u_exact2(x: Array, y: Array) -> Array:
    return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)


def test_thomas_matches_dense_solve() -> None:
    n = 20
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    sub = jnp.concatenate([jnp.zeros(1), jax.random.uniform(k1, (n - 1,))])
    sup = jnp.concatenate([jax.random.uniform(k2, (n - 1,)), jnp.zeros(1)])
    diag = 3.0 + jax.random.uniform(k3, (n,))  # diagonally dominant
    rhs = jax.random.uniform(k4, (n,))
    dense = jnp.diag(diag) + jnp.diag(sup[:-1], 1) + jnp.diag(sub[1:], -1)
    assert jnp.allclose(thomas_solve(sub, diag, sup, rhs), jnp.linalg.solve(dense, rhs), atol=1e-10)


def test_poisson_1d_fem_second_order() -> None:
    ns = [16, 32, 64, 128]
    errors = [
        float(jnp.max(jnp.abs(u - _u_exact(nodes)))) for nodes, u in (poisson_1d(_f, n) for n in ns)
    ]
    assert errors[-1] < 1e-4
    hs = jnp.array([1.0 / n for n in ns])
    slope = float(jnp.polyfit(jnp.log(hs), jnp.log(jnp.array(errors)), 1)[0])
    assert 1.8 < slope < 2.2


def test_poisson_2d_fem_second_order() -> None:
    ns = [8, 16, 32]
    errors = []
    for n in ns:
        coords, u = poisson_2d(_f2, n)
        xs, ys = jnp.meshgrid(coords, coords, indexing="ij")
        errors.append(float(jnp.max(jnp.abs(u - _u_exact2(xs, ys)))))
    assert errors[-1] < 5e-3
    hs = jnp.array([1.0 / n for n in ns])
    slope = float(jnp.polyfit(jnp.log(hs), jnp.log(jnp.array(errors)), 1)[0])
    assert 1.8 < slope < 2.2


def test_convection_diffusion_matches_the_exact_solution_when_resolved() -> None:
    # Below the cell-Peclet threshold the unstabilised scheme is an ordinary second-order FEM, so
    # everything the oscillation tests below say is about Pe, not about the discretisation.
    ns = [40, 80, 160]
    errors = []
    for n in ns:
        nodes, u = convection_diffusion_1d(0.1, 1.0, n)
        errors.append(float(jnp.max(jnp.abs(u - convection_diffusion_exact(nodes, 0.1, 1.0)))))
    hs = jnp.array([1.0 / n for n in ns])
    slope = float(jnp.polyfit(jnp.log(hs), jnp.log(jnp.array(errors)), 1)[0])
    assert 1.8 < slope < 2.2


def test_central_differencing_oscillates_above_the_cell_peclet_threshold() -> None:
    # proofs/convection_diffusion.v: oscillation_above_threshold / monotone_below_threshold. The
    # discriminator is the SIGN of consecutive forward differences, which is the proved quantity --
    # not the error, which is large on both sides for a different reason.
    def alternations(eps: float, n: int) -> int:
        _, u = convection_diffusion_1d(eps, 1.0, n)
        d = jnp.diff(u)
        return int(jnp.sum(jnp.sign(d[:-1]) * jnp.sign(d[1:]) < 0))

    assert alternations(0.01, 20) > 0  # Pe = 2.5
    assert alternations(0.1, 20) == 0  # Pe = 0.25, the same mesh and the same scheme


def test_full_upwind_is_monotone_but_only_first_order() -> None:
    # It buys stability with a truncation of exp(2*Pe) after two terms (upwind_first_order), so it
    # must be monotone AND must fail to reach second order.
    _, u = convection_diffusion_1d(0.01, 1.0, 20, alpha=1.0)
    assert float(jnp.min(u)) >= -1e-12
    assert float(jnp.max(u)) <= 1.0 + 1e-12
    ns = [40, 80, 160]
    errors = []
    for n in ns:
        nodes, u_n = convection_diffusion_1d(0.1, 1.0, n, alpha=1.0)
        errors.append(float(jnp.max(jnp.abs(u_n - convection_diffusion_exact(nodes, 0.1, 1.0)))))
    hs = jnp.array([1.0 / n for n in ns])
    slope = float(jnp.polyfit(jnp.log(hs), jnp.log(jnp.array(errors)), 1)[0])
    assert 0.85 < slope < 1.15


def test_optimal_upwind_stays_inside_the_proved_stability_range() -> None:
    # stability_range: alpha > 1 - 1/Pe is non-oscillatory; optimal_alpha_below_full_upwind: the
    # optimal choice is strictly less diffusive than alpha = 1. Both, across four decades of Pe.
    for peclet in (0.05, 0.5, 2.5, 20.0, 500.0):
        alpha = optimal_upwind(peclet)
        assert 1.0 - 1.0 / peclet <= alpha < 1.0
        _, u = convection_diffusion_1d(1.0 / (2.0 * peclet * 20), 1.0, 20, alpha=alpha)
        assert float(jnp.min(u)) >= -1e-12  # the operational consequence: no oscillation
    # The bound is strict in exact arithmetic, and measurably strict only while the gap
    # coth(Pe) - 1 = 2/(exp(2*Pe) - 1) exceeds double precision -- i.e. below Pe ~ 19.
    for peclet in (0.05, 0.5, 2.5, 15.0):
        assert 1.0 - 1.0 / peclet < optimal_upwind(peclet)


def test_optimal_upwind_series_branch_agrees_with_the_closed_form() -> None:
    # coth(Pe) - 1/Pe subtracts two quantities that both diverge as Pe -> 0, so the closed form
    # loses every significant digit there. The two branches must agree where they meet.
    crossover = 1e-3
    series = crossover / 3.0 - crossover**3 / 45.0
    assert abs(optimal_upwind(crossover) - series) < 1e-9 * abs(series)
    assert abs(optimal_upwind(1e-6) / 1e-6 - 1.0 / 3.0) < 1e-12  # the exact Pe -> 0 limit


def test_convection_diffusion_certificate_locates_the_threshold_and_prices_the_cure() -> None:
    # Result 47. Each block can fail on its own, and the certificate is run at whatever precision
    # is in force -- which is why nodal exactness is stated in ULPs and not in absolute error.
    curve = convection_diffusion_certificate()
    # (1) THE INSTABILITY. Above the threshold the scheme undershoots and alternates.
    assert curve.peclet > 1.0
    assert curve.bubnov_undershoot < -0.01
    assert curve.bubnov_sign_changes > 0
    # (2) THE THRESHOLD, measured by bisection rather than quoted: threshold_is_exactly_one.
    assert abs(curve.measured_threshold - 1.0) < 0.01
    assert curve.fine_peclet < 1.0
    assert curve.fine_bubnov_undershoot >= -1e-12  # same scheme, same code path, no oscillation
    # (3) FULL UPWIND: monotone, and first-order because it truncates exp(2*Pe) after two terms.
    assert curve.upwind_undershoot >= -1e-12
    assert abs(curve.upwind_order - 1.0) < 0.15
    assert curve.upwind_error < curve.bubnov_error  # stable, but still visibly wrong
    # (4) THE CURE. Nodally exact, in units of the working machine epsilon so the assertion holds
    # under both float32 and float64 -- the absolute error moves by seven orders between them.
    assert curve.supg_error_ulps < 10.0
    assert curve.supg_error < 1e-6 * curve.upwind_error
    assert 1.0 - 1.0 / curve.peclet < curve.alpha_optimal < 1.0
    assert curve.ok
