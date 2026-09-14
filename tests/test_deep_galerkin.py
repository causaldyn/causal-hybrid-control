"""Deep Galerkin gates: the analytic 1-D Poisson solution, and the LQ mean-field equilibrium."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chc.deep_galerkin import (
    CongestedMeanFieldGame,
    LQMeanFieldGame,
    adjoint_weighted_error,
    dual_weighted_error_estimate,
    finite_population_gap_certificate,
    lq_mean_field_certificate,
    nonlinear_dwr_certificate,
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


def test_dual_weighted_estimate_recovers_the_error_without_the_closed_form() -> None:
    # Result 55: a deliberately under-trained model, so the error is large and the estimate has
    # something to find. The estimator sees only the network and the cost parameters.
    game = LQMeanFieldGame(
        a=-0.5,
        b=1.0,
        q=1.0,
        r=1.0,
        coupling=3.0,
        terminal_coupling=3.0,
        sigma=0.7,
        horizon=0.5,
        mean_initial=1.0,
        variance_initial=0.25,
    )
    model = solve_mfg_dgm(game, steps=150, half_width=10.0, seed=0)
    mean_terminal = model.mean(jnp.asarray(game.horizon))
    fitted = float(
        jax.grad(lambda x: model.value_at(jnp.asarray(0.0), x, mean_terminal))(jnp.asarray(0.0))
    )
    truth = abs(fitted - float(game.solve().value_s[0]))
    assert truth > 0.1  # the model really is wrong, so the estimate is not trivially zero
    assert dual_weighted_error_estimate(game, model) == pytest.approx(truth, rel=0.1)


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
    # Result 55: the dual-weighted estimator, built from the model's own linearisation and never
    # from the closed form, recovers the error the raw residual inverts -- and does so to within
    # a few percent at a horizon where the true error is two orders larger.
    assert curve.near_value_error > 5.0 * curve.far_value_error
    assert curve.near_dual_weighted > curve.far_dual_weighted
    assert curve.dual_weighted_accuracy < 0.1


# ---------------------------------------------------------------------------------------------
# The finite-population gap: the Fokker-Planck density is the N -> infinity limit.
# ---------------------------------------------------------------------------------------------


def _monotone_game() -> LQMeanFieldGame:
    return LQMeanFieldGame(
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


def test_the_gap_halves_when_the_population_quadruples() -> None:
    """The exponent is an identity, so it holds with no simulation and no fitted slope."""
    solution = _monotone_game().solve()
    small, large = solution.finite_population_rms(64), solution.finite_population_rms(256)
    assert np.allclose(large, small / 2.0)


def test_a_population_needs_at_least_one_agent() -> None:
    solution = _monotone_game().solve()
    with pytest.raises(ValueError, match="at least one agent"):
        solution.finite_population_rms(0)
    with pytest.raises(ValueError, match="at least one agent"):
        solution.simulate_population(0)


def test_a_simulated_population_averages_to_the_mean_field_mean() -> None:
    """``E[m_N] = m`` exactly under a mean-field feedback -- no bias to correct for."""
    solution = _monotone_game().solve()
    _, paths = solution.simulate_population(64, replicates=400, seed=3, n_step=200)
    gap = abs(float(paths[:, -1].mean()) - float(solution.mean[-1]))
    assert gap < 3.0 * float(solution.finite_population_rms(64)[-1]) / math.sqrt(400)


def test_a_common_shock_is_what_breaks_the_identity() -> None:
    """Independence, not large N, is what makes the gap shrink; share the noise and it stops."""
    solution = _monotone_game().solve()
    truth = float(solution.mean[-1])

    def rms(n_agents: int, noise: str) -> float:
        _, paths = solution.simulate_population(
            n_agents, replicates=200, seed=11, n_step=200, noise=noise
        )
        return float(np.sqrt(((paths[:, -1] - truth) ** 2).mean()))

    independent = rms(4, "independent") / rms(256, "independent")
    shared = rms(4, "common") / rms(256, "common")
    assert independent > 5.0  # 8x is the closed form; Monte-Carlo error is the slack
    assert shared == pytest.approx(1.0, abs=0.15)


def test_finite_population_gap_certificate_holds() -> None:
    gap = finite_population_gap_certificate()
    assert gap.ok
    # The closed form, not a fitted rate: the measured RMS matches sqrt(v(T)/N) at every N.
    assert gap.worst_relative_error < 0.08
    assert gap.measured_exponent == pytest.approx(-0.5, abs=0.05)
    # The falsification arm: with one shared Brownian path the exponent is nowhere near -1/2.
    assert gap.common_shock_exponent > -0.15
    # Coupling the agents moves the CONSTANT to the rate A + kappa and leaves the exponent alone,
    # so the unmoved constant has to be wrong -- otherwise the arm proves nothing.
    assert gap.coupled_variance_ratio == pytest.approx(1.0, abs=0.08)
    assert gap.coupled_wrong_rate_ratio > 1.5
    # And at the kappa = c/N an N-player Nash actually has, the bias is O(1/N): it matches the
    # leading-order constant and stays well under the O(1/sqrt N) fluctuation.
    assert gap.measured_bias_constant == pytest.approx(gap.predicted_bias_constant, rel=0.05)
    assert gap.bias_to_fluctuation < 0.3


# ---------------------------------------------------------------------------------------------
# A16: the adjoint as a linearisation, so the estimator survives a non-quadratic game.
# ---------------------------------------------------------------------------------------------


def test_the_congested_solve_reproduces_the_closed_form_when_there_is_no_congestion() -> None:
    """At ``congestion = 0`` the shooting solve and the LQ closed form are the same problem."""
    base = _monotone_game()
    _, path = CongestedMeanFieldGame(base=base, congestion=0.0).solve(n_time=801)
    reference = base.solve(n_time=801)
    assert path[0, 1] == pytest.approx(float(reference.value_s[0]), abs=1e-7)
    assert np.abs(path[:, 0] - reference.mean).max() < 1e-7


def test_congestion_moves_the_answer_enough_to_be_worth_estimating() -> None:
    """A gate the affine adjoint could pass by accident is not a gate."""
    base = _monotone_game()
    affine = CongestedMeanFieldGame(base=base, congestion=0.0).solve(n_time=401)[1]
    congested = CongestedMeanFieldGame(base=base, congestion=0.6).solve(n_time=401)[1]
    assert abs(congested[0, 1] / affine[0, 1] - 1.0) > 0.5


def test_the_adjoint_estimate_is_exact_on_an_affine_field() -> None:
    """The manufactured error is ``eta p_S(0)``; on an affine field the estimator returns it."""
    game = CongestedMeanFieldGame(base=_monotone_game(), congestion=0.0)
    times, path = game.solve(n_time=801)
    rate = np.asarray(jax.vmap(game.reduced_field)(jnp.asarray(path)))
    shape = np.stack(
        [np.sin(np.pi * times), 1.0 + 0.4 * np.cos(np.pi * times)], axis=1
    )  # p_m(0) = 0
    shape_rate = np.stack(
        [np.pi * np.cos(np.pi * times), -0.4 * np.pi * np.sin(np.pi * times)], axis=1
    )
    for size in (0.05, 0.0125):
        got = adjoint_weighted_error(
            game.reduced_field,
            times,
            path + size * shape,
            rate + size * shape_rate,
            game.terminal_row,
        )
        assert got == pytest.approx(size * shape[0, 1], rel=1e-5)


def test_the_estimator_rejects_a_rate_that_does_not_match_the_trajectory() -> None:
    game = CongestedMeanFieldGame(base=_monotone_game(), congestion=0.3)
    times, path = game.solve(n_time=201)
    with pytest.raises(ValueError, match="must agree"):
        adjoint_weighted_error(game.reduced_field, times, path, path[:-1], game.terminal_row)


def test_nonlinear_dwr_certificate_holds() -> None:
    curve = nonlinear_dwr_certificate()
    assert curve.ok
    # An affine field leaves no remainder, so the relative error is FLAT in the perturbation size
    # and sits at the quadrature floor -- which is a floor, because halving the step quarters it.
    assert max(curve.affine_relative_errors) < 1e-5
    assert max(curve.affine_relative_errors) / min(curve.affine_relative_errors) < 1.1
    assert curve.affine_quadrature_ratio > 3.0
    # And the adjoint it builds by autodiff agrees with Result 55's closed-form denominator.
    assert curve.denominator_against_result_55 < 1e-6
    # On the congested field the remainder is the second variation: second order in the defect,
    # first order in the relative error, on an arm where that error is percent-scale not noise.
    assert curve.congested_exponent == pytest.approx(2.0, abs=0.15)
    assert curve.congested_relative_error > 0.01
    # The right adjoint is worth exactly one order; the affine model, which is what the Result 55
    # estimator would apply here, stops tracking the error altogether.
    assert curve.affine_adjoint_exponent == pytest.approx(1.0, abs=0.15)
    assert curve.affine_model_exponent == pytest.approx(0.0, abs=0.15)
    assert curve.affine_adjoint_errors[-1] > 100.0 * curve.congested_errors[-1]
