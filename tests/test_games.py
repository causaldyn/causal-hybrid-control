"""Game primitives: simplex projection, certified congestion equilibrium, Stackelberg allocation."""

import jax
import jax.numpy as jnp
import pytest

from chc.games import (
    EquilibriumMonotonicityCertificate,
    EquilibriumTransferCertificate,
    congestion_contraction_certificate,
    congestion_contraction_modulus,
    congestion_damping,
    equilibrium_monotonicity_certificate,
    equilibrium_transfer_certificate,
    project_simplex,
    softmax_congestion_equilibrium,
    stackelberg_allocation,
)
from chc.marketplace import SharedStateMarket


@pytest.fixture(scope="module")
def transfer() -> EquilibriumTransferCertificate:
    """Result 39's evidence -- ~9 s of leader solves, so it is built once for the module."""
    return equilibrium_transfer_certificate()


@pytest.fixture(scope="module")
def monotonicity() -> EquilibriumMonotonicityCertificate:
    """Result 39's GLOBAL half: moduli over a box plus finite-perturbation displacements."""
    return equilibrium_monotonicity_certificate()


def test_project_simplex_lands_on_the_budget_simplex() -> None:
    projected = project_simplex(jnp.array([2.0, -1.0, 0.5, 3.0]), 1.0)
    assert abs(float(jnp.sum(projected)) - 1.0) < 1e-5
    assert bool(jnp.all(projected >= -1e-6))


def test_equilibrium_conserves_mass_and_favours_attractive_zones() -> None:
    solution = softmax_congestion_equilibrium(
        jnp.array([1.0, 0.5, 0.2]), jnp.zeros(3), congestion=1.0, mass=6.0
    )
    assert abs(float(jnp.sum(solution.x)) - 6.0) < 1e-3  # mass is conserved
    assert float(solution.x[0]) > float(solution.x[2])  # the more attractive zone holds more agents


def test_equilibrium_reports_convergence_inside_the_certified_region() -> None:
    solution = softmax_congestion_equilibrium(
        jnp.array([1.0, 0.5, 0.2]), jnp.zeros(3), congestion=1.0, mass=6.0
    )
    assert bool(solution.converged)
    assert float(solution.residual) <= 1e-6


def test_equilibrium_reports_failure_instead_of_returning_a_non_equilibrium() -> None:
    """The regression test for the bug this solver replaced: a fixed trip count returned a
    2-cycle point (residual 3.21 on mass 6.0, most mass on the LEAST attractive zone) with no
    way for the caller to notice. Non-convergence must now be visible in the return value.
    """
    solution = softmax_congestion_equilibrium(
        jnp.array([1.0, 0.5, 0.2]), jnp.zeros(3), congestion=8.0, mass=6.0, beta=25.0
    )
    assert not bool(solution.converged)
    assert float(solution.residual) > 0.1  # relative: the absolute gap is 3.2 on a mass of 6
    assert congestion_contraction_modulus(25.0, 8.0) < 0.0  # and the certificate refuses it


_ATTRACT = jnp.array([1.0, 0.5, 0.2, -0.4])
_WEIGHTS = jnp.array([0.3, -1.1, 0.7, 0.5])
_MASS, _CONGESTION, _BETA = 8.0, 1.5, 2.5


def _equilibrium_objective(u: jnp.ndarray) -> jnp.ndarray:
    solution = softmax_congestion_equilibrium(_ATTRACT, u, _CONGESTION, _MASS, _BETA)
    return jnp.dot(_WEIGHTS, solution.x)


def _dense_ift_gradient() -> jnp.ndarray:
    """``w^T (I - dT/dx)^{-1} dT/du`` from an explicit solve -- the formula, not the solver."""

    def step(u: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * x + 0.5 * _MASS * jax.nn.softmax(
            _BETA * (_ATTRACT + u - _CONGESTION * x / _MASS)
        )

    u0 = jnp.zeros(4)
    x_star = softmax_congestion_equilibrium(_ATTRACT, u0, _CONGESTION, _MASS, _BETA).x
    jac_x = jax.jacobian(lambda z: step(u0, z))(x_star)
    jac_u = jax.jacobian(lambda p: step(p, x_star))(u0)
    return _WEIGHTS @ jnp.linalg.solve(jnp.eye(4) - jac_x, jac_u)


def test_implicit_gradient_matches_the_dense_ift_solve() -> None:
    """The bilevel path the module advertises. Unrolling a truncated iteration, or getting the
    adjoint recursion wrong, would miss this by percent; the implicit VJP hits float32 precision.
    """
    analytic = jax.grad(_equilibrium_objective)(jnp.zeros(4))
    assert float(jnp.max(jnp.abs(analytic - _dense_ift_gradient()))) < 1e-5


def test_implicit_gradient_matches_central_differences() -> None:
    """Derivative-free cross-check of the same gradient -- nothing here uses autodiff.

    ``eps = 3e-3`` balances truncation against the float32 round-off the solve leaves in the
    objective; at ``1e-3`` the cancellation error alone is the size of the tolerance.
    """
    analytic = jax.grad(_equilibrium_objective)(jnp.zeros(4))
    eps, u0 = 3e-3, jnp.zeros(4)
    numeric = jnp.array(
        [
            (_equilibrium_objective(u0.at[i].add(eps)) - _equilibrium_objective(u0.at[i].add(-eps)))
            / (2 * eps)
            for i in range(4)
        ]
    )
    relative = float(jnp.max(jnp.abs(numeric - analytic)) / jnp.max(jnp.abs(analytic)))
    assert relative < 1e-3


def test_stackelberg_allocation_improves_the_equilibrium_objective() -> None:
    """Optimise *through* the equilibrium, not through a closed form standing in for it."""
    attract = jnp.array([0.1, 0.5, 2.0])
    weights = jnp.array([1.0, 1.0, 0.2])  # the crowded zone is worth least per driver

    def objective(u: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(weights, softmax_congestion_equilibrium(attract, u, 1.5, 6.0).x)

    u = stackelberg_allocation(objective, 3, budget=1.0, steps=300)
    assert abs(float(jnp.sum(u)) - 1.0) < 1e-3
    assert float(objective(u)) > float(objective(jnp.full(3, 1 / 3)))


def test_contraction_certificate_holds_at_the_solved_equilibrium() -> None:
    certificate = congestion_contraction_certificate(beta=2.5, congestion=2.0)
    assert certificate.certified
    assert certificate.ok
    assert certificate.modulus > 0.0
    assert certificate.measured_operator_norm <= certificate.jacobian_bound + 1e-6


def test_contraction_certificate_is_sufficient_not_necessary() -> None:
    """At ``beta*congestion = 7`` the map still converges but the bound refuses to certify it --
    the bound uses the worst-case softmax eigenvalue 1/2, attained only at a two-point uniform s.
    """
    assert congestion_contraction_modulus(7.0, 1.0) < 0.0  # not certified
    solution = softmax_congestion_equilibrium(
        jnp.array([1.0, 0.5, 0.2]), jnp.zeros(3), congestion=1.0, mass=6.0, beta=7.0
    )
    assert bool(solution.converged)  # yet it converges


def test_shipped_marketplace_configuration_is_certified() -> None:
    """Makes the ``SharedStateMarket._equilibrium`` docstring claim executable, not decorative."""
    market = SharedStateMarket()
    assert congestion_contraction_modulus(market.beta, market.congestion) > 0.0


def test_optimal_damping_solves_a_game_the_default_cannot() -> None:
    """Root-cause regression: the ``beta*congestion < 6`` ceiling belonged to the hard-coded
    damping, not to the game. At ``kappa = 20`` the default 2-cycles; ``congestion_damping``
    contracts and the same equilibrium is reached.
    """
    beta, congestion = 10.0, 2.0
    attract = jnp.array([1.0, 0.5, 0.2, -0.3])
    default = softmax_congestion_equilibrium(attract, jnp.zeros(4), congestion, 6.0, beta)
    assert not bool(default.converged)

    damping = congestion_damping(beta, congestion)
    assert damping < 4.0 / (2.0 + beta * congestion)  # inside the certified region
    fixed = softmax_congestion_equilibrium(
        attract, jnp.zeros(4), congestion, 6.0, beta, damping=damping
    )
    assert bool(fixed.converged)
    assert congestion_contraction_modulus(beta, congestion, damping) > 0.0


def test_equilibrium_conditioning_is_exactly_one_at_every_congestion(
    transfer: EquilibriumTransferCertificate,
) -> None:
    """Result 39: the ambient ``||(I - S')^{-1}||_2 = 1`` uniformly -- the equilibrium never
    *locally* amplifies an operator perturbation, however close the damped solver is to losing
    contraction.
    """
    assert all(abs(c - 1.0) < 1e-3 for c in transfer.conditioning)


def test_the_ambient_norm_is_attained_only_in_the_never_excited_mass_direction(
    transfer: EquilibriumTransferCertificate,
) -> None:
    """The 1 comes from ``J*1 = 0``. Equilibria conserve mass, so displacements live in ``1^perp``,
    where the same operator is *strictly* contractive -- the honest sharp number.
    """
    assert all(0.0 < t < 1.0 for t in transfer.tangent_conditioning)
    assert all(
        t < c for t, c in zip(transfer.tangent_conditioning, transfer.conditioning, strict=True)
    )


def test_equilibrium_transfer_regret_is_second_order_in_operator_error(
    transfer: EquilibriumTransferCertificate,
) -> None:
    """The order survives the equilibrium layer: an interior leader optimum gives slope ~2."""
    assert 1.7 <= transfer.regret_slope <= 2.3
    assert transfer.ok


def test_contraction_modulus_is_not_the_equilibrium_conditioning(
    transfer: EquilibriumTransferCertificate,
) -> None:
    """The correction to the proposed ``C/mu^2`` constant, scoped to what it actually refutes: for
    *this* map the damped solver's contraction margin is not the equilibrium's sensitivity constant.
    ``1/mu`` overstates it by ``4/(6-kappa)``, unboundedly, while the truth does not move off 1.
    """
    assert transfer.looseness[-1] > 50.0  # kappa = 5.96 -> 100x
    assert transfer.looseness[0] < transfer.looseness[-1]  # and it grows with kappa
    assert max(transfer.conditioning) - min(transfer.conditioning) < 1e-3  # truth is flat


def test_the_residual_map_is_one_strongly_monotone_over_the_whole_box(
    monotonicity: EquilibriumMonotonicityCertificate,
) -> None:
    # Not an implicit-function derivative at the equilibrium: a minimum over a box, and the
    # constant is exactly 1 because the softmax Jacobian annihilates the constants.
    assert monotonicity.ambient_modulus == pytest.approx(1.0, abs=1e-9)
    assert monotonicity.ok


def test_monotonicity_holds_between_pairs_not_only_in_the_derivative(
    monotonicity: EquilibriumMonotonicityCertificate,
) -> None:
    # The definition itself, on random pairs. A pointwise Jacobian bound does not imply it without
    # the mean value theorem, so measuring it is a separate check rather than a restatement.
    assert monotonicity.finite_monotonicity >= 1.0 - 1e-6


def test_a_finite_operator_perturbation_moves_the_equilibrium_no_further(
    monotonicity: EquilibriumMonotonicityCertificate,
) -> None:
    # What the local conditioning could not say: at modulus 1 the displacement is bounded by the
    # perturbation one-for-one, at finite size.
    assert monotonicity.displacement_ratio <= 1.0 + 1e-6


def test_the_tangent_improvement_is_local_and_evaporates_over_the_box(
    monotonicity: EquilibriumMonotonicityCertificate,
) -> None:
    # Result 39 (b)'s strictly-better tangent constant is a neighbourhood statement. Near a corner
    # the softmax approaches a vertex, s_min collapses, and the improvement is gone -- which is
    # exactly why quoting it globally is a mistake.
    assert monotonicity.local_tangent_bound > monotonicity.tangent_bound
    assert monotonicity.tangent_bound == pytest.approx(1.0, abs=1e-6)
    assert monotonicity.local_tangent_modulus >= monotonicity.local_tangent_bound - 1e-9


def test_an_active_constraint_costs_an_order_and_a_vertex_costs_the_slope(
    transfer: EquilibriumTransferCertificate,
) -> None:
    # The regularity assumption behind Result 39 (c), made falsifiable. With the budget constraint
    # active the slope drops by nearly a full order; at a vertex the plan does not move at all, so
    # the regret is identically zero and there is no slope to degrade.
    assert transfer.constrained_slope < transfer.regret_slope - 0.5
    assert all(r == 0.0 for r in transfer.vertex_regret)
