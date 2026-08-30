"""Deep Galerkin gates: the analytic 1-D Poisson solution, and the LQ mean-field equilibrium."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chc.deep_galerkin import (
    LQMeanFieldGame,
    lq_mean_field_certificate,
    solve_mfg_dgm,
    solve_poisson_dgm,
)


def test_deep_galerkin_recovers_the_poisson_solution() -> None:
    # -V'' = pi^2 sin(pi x), V(0)=V(1)=0  ->  V(x) = sin(pi x)
    def source(x: jax.Array) -> jax.Array:
        return jnp.pi**2 * jnp.sin(jnp.pi * x)

    model = solve_poisson_dgm(source, steps=3000)
    xs = jnp.linspace(0.0, 1.0, 21)
    predicted = jnp.array([model(x) for x in xs])
    rmse = float(jnp.sqrt(jnp.mean((predicted - jnp.sin(jnp.pi * xs)) ** 2)))
    assert rmse < 0.05  # the neural Galerkin matches the analytic (and FEM) solution


def test_lq_mean_field_closed_form_annihilates_both_pdes() -> None:
    game = LQMeanFieldGame(
        a=-0.5,
        b=1.0,
        q=1.0,
        r=1.0,
        coupling=0.5,
        terminal_coupling=0.5,
        sigma=0.7,
        horizon=1.0,
        mean_initial=1.0,
        variance_initial=0.25,
    )
    solution = game.solve()
    rng = np.random.default_rng(0)
    t = rng.uniform(0.0, 1.0, 500)
    x = rng.uniform(-5.0, 5.0, 500)
    assert np.abs(solution.hjb_residual(t, x)).max() < 1e-10
    assert np.abs(solution.fokker_planck_residual(t, x)).max() < 1e-10
    assert solution.mean[0] == pytest.approx(game.mean_initial)
    assert solution.variance[0] == pytest.approx(game.variance_initial)
    # A(T) closes the loop even though a alone does not fix the sign.
    assert game.closed_loop_rate < 0.0
    assert game.closed_loop_rate**2 == pytest.approx(game.a**2 + game.q * game.b**2 / game.r)


def test_obstruction_horizon_matches_the_bisected_denominator() -> None:
    game = LQMeanFieldGame(
        a=-0.5,
        b=1.0,
        q=1.0,
        r=1.0,
        coupling=3.0,
        terminal_coupling=3.0,
        sigma=0.7,
        horizon=1.0,
        mean_initial=1.0,
        variance_initial=0.25,
    )
    assert game.lambda_squared < 0.0 < game.branch_threshold < game.coupling
    predicted = game.obstruction_horizon()
    low, high = 0.1, 1.2
    for _ in range(80):
        middle = 0.5 * (low + high)
        if game.fixed_point_denominator(low) * game.fixed_point_denominator(middle) <= 0.0:
            high = middle
        else:
            low = middle
    assert 0.5 * (low + high) == pytest.approx(predicted, rel=1e-9)
    assert game.fixed_point_denominator(predicted) == pytest.approx(0.0, abs=1e-12)


def test_monotone_coupling_has_no_obstruction_at_any_horizon() -> None:
    game = LQMeanFieldGame(
        a=-0.5,
        b=1.0,
        q=1.0,
        r=1.0,
        coupling=0.5,
        terminal_coupling=0.5,
        sigma=0.7,
        horizon=1.0,
        mean_initial=1.0,
        variance_initial=0.25,
    )
    assert game.obstruction_gain <= 0.0 < game.lambda_squared
    assert math.isinf(game.obstruction_horizon())
    for horizon in (0.5, 5.0, 50.0, 500.0):
        assert game.fixed_point_denominator(horizon) >= 1.0


def test_quadrature_nodes_are_not_trainable() -> None:
    # The density carries its initial condition exactly, so the mean at t=0 is m0 by
    # construction -- unless the optimiser has been moving the integration grid underneath it.
    game = LQMeanFieldGame(
        a=-0.5,
        b=1.0,
        q=1.0,
        r=1.0,
        coupling=3.0,
        terminal_coupling=3.0,
        sigma=0.7,
        horizon=0.35,
        mean_initial=1.0,
        variance_initial=0.25,
    )
    model = solve_mfg_dgm(game, steps=200)
    expected = jnp.linspace(-model.half_width, model.half_width, model.n_quadrature)
    assert float(jnp.abs(model.quadrature - expected).max()) == 0.0
    assert float(model.mean(jnp.asarray(0.0))) == pytest.approx(game.mean_initial, abs=1e-9)


def test_lq_mean_field_certificate_holds() -> None:
    curve = lq_mean_field_certificate()
    assert curve.ok
    # The gate is exact, so any DGM error is the DGM's.
    assert curve.closed_form_hjb_residual < 1e-10
    assert curve.horizon_relative_error < 1e-9
    assert curve.pole_exponent == pytest.approx(-1.0, abs=0.05)
    # The neural solve reproduces the equilibrium on the monotone instance.
    assert curve.dgm_control_error < 0.02
    # And the arm that must be able to fail: near the obstruction the error grows while the
    # residual a stopping rule would watch actually shrinks.
    assert curve.near_control_error > 5.0 * curve.far_control_error
    assert curve.near_residual < curve.far_residual
    assert curve.residual_blindness > 5.0
