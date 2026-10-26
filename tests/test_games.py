"""Game primitives: simplex projection, certified congestion equilibrium, Stackelberg allocation."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chc.games import (
    congestion_contraction_modulus,
    congestion_damping,
    exact_congestion_allocation,
    project_simplex,
    softmax_congestion_equilibrium,
    stackelberg_allocation,
)
from chc.marketplace import SharedStateMarket
from chc.research.games import (
    EquilibriumMonotonicityCertificate,
    EquilibriumTransferCertificate,
    congestion_contraction_certificate,
    equilibrium_monotonicity_certificate,
    equilibrium_transfer_certificate,
)


@pytest.fixture(scope="module")
def transfer() -> EquilibriumTransferCertificate:
    """The transfer evidence -- ~9 s of leader solves, so it is built once for the module."""
    return equilibrium_transfer_certificate()


@pytest.fixture(scope="module")
def monotonicity() -> EquilibriumMonotonicityCertificate:
    """The transfer's GLOBAL half: moduli over a box plus finite-perturbation displacements."""
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


def test_stackelberg_allocation_finds_a_local_maximum_not_the_global_one() -> None:
    """The documented scope: from the even split the ascent ends at the first zone's vertex, worth
    2.973, where the second zone's vertex is worth 3.184."""
    attract = jnp.array([0.1, 0.5, 2.0])
    weights = jnp.array([1.0, 1.0, 0.2])

    def objective(u: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(weights, softmax_congestion_equilibrium(attract, u, 1.5, 6.0).x)

    u = stackelberg_allocation(objective, 3, budget=1.0, steps=300)
    assert float(objective(u)) == pytest.approx(2.973, abs=2e-3)
    assert float(objective(jnp.array([0.0, 1.0, 0.0]))) == pytest.approx(3.184, abs=2e-3)


def test_stackelberg_allocation_starts_where_it_is_told() -> None:
    """The ascent from the even split ends at the first zone's vertex, worth 2.973; started at the
    second zone's vertex it keeps its 3.184. A start off the simplex is projected onto it."""
    attract = jnp.array([0.1, 0.5, 2.0])
    weights = jnp.array([1.0, 1.0, 0.2])

    def objective(u: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(weights, softmax_congestion_equilibrium(attract, u, 1.5, 6.0).x)

    u = stackelberg_allocation(
        objective, 3, budget=1.0, steps=300, start=jnp.array([0.0, 1.0, 0.0])
    )
    assert float(objective(u)) >= float(objective(jnp.array([0.0, 1.0, 0.0])))
    assert float(objective(u)) == pytest.approx(3.184, abs=2e-3)
    held = stackelberg_allocation(
        objective, 3, budget=1.0, steps=0, start=jnp.array([1.0, 2.0, 0.0])
    )
    np.testing.assert_allclose(held, [0.0, 1.0, 0.0], rtol=0.0, atol=1e-12)
    with pytest.raises(ValueError, match="start has shape"):
        stackelberg_allocation(objective, 3, budget=1.0, start=jnp.ones(2))


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
    """The ambient ``||(I - S')^{-1}||_2 = 1`` uniformly -- the equilibrium never
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
    # The strictly-better tangent constant is a neighbourhood statement. Near a corner
    # the softmax approaches a vertex, s_min collapses, and the improvement is gone -- which is
    # exactly why quoting it globally is a mistake.
    assert monotonicity.local_tangent_bound > monotonicity.tangent_bound
    assert monotonicity.tangent_bound == pytest.approx(1.0, abs=1e-6)
    assert monotonicity.local_tangent_modulus >= monotonicity.local_tangent_bound - 1e-9


def test_an_active_constraint_costs_an_order_and_a_vertex_costs_the_slope(
    transfer: EquilibriumTransferCertificate,
) -> None:
    # The regularity assumption of the order transfer, made falsifiable. With the budget constraint
    # active the slope drops by nearly a full order; at a vertex the plan does not move at all, so
    # the regret is identically zero and there is no slope to degrade.
    assert transfer.constrained_slope < transfer.regret_slope - 0.5
    assert all(r == 0.0 for r in transfer.vertex_regret)


def _rides(value, demand, beta, congestion, mass, u):
    """The rides an incentive ``u`` completes, through the solver the market uses, at float64."""
    shares = softmax_congestion_equilibrium(
        jnp.asarray(value), u, congestion, mass, beta, tol=1e-13, max_iter=5000
    )
    return jnp.sum(jnp.minimum(jnp.asarray(demand), shares.x)), shares.converged


@pytest.mark.parametrize(
    ("seed", "congestion"), [(0, 2.0), (1, 2.0), (2, 2.0), (3, 0.0)], ids=["a", "b", "c", "free"]
)
def test_no_split_of_the_budget_completes_more_than_the_exact_allocation(seed, congestion) -> None:
    """Every split of the budget on a grid of step ``budget/120`` over three zones, solved by the
    fixed point the market uses: none completes more rides than the allocation returned, and the
    fixed point at that allocation completes what it says. The last market has no crowding, where
    a zone's floor share is an exponential, not Lambert's ``W``."""
    rng = np.random.default_rng(seed)
    demand = rng.uniform(0.4, 2.0, 3)
    value = 1.5 * demand + 0.3 * rng.standard_normal(3)
    beta, mass, budget = 2.5, 3.0, 1.5
    exact = exact_congestion_allocation(value, demand, beta, congestion, mass, budget)
    steps = 120
    grid = np.array([(i, j, steps - i - j) for i in range(steps + 1) for j in range(steps + 1 - i)])
    rides, converged = jax.jit(
        jax.vmap(lambda u: _rides(value, demand, beta, congestion, mass, u))
    )(jnp.asarray(budget * grid / steps))
    assert bool(jnp.all(converged))
    at_exact, solved = _rides(value, demand, beta, congestion, mass, jnp.asarray(exact.allocation))
    assert bool(solved)
    # the solves stop at a residual of 1e-13 of the drivers: 1e-9 is a thousand times their error
    assert float(at_exact) == pytest.approx(exact.value, abs=1e-9)
    assert float(jnp.max(rides)) <= exact.value + 1e-9


def test_the_exact_allocation_carries_its_certificate() -> None:
    """On ``SharedStateMarket(seed=4)``: the zones it names hold their demand, the unpaid zones sit
    at its level, the budget reaches that level and falls short just below it."""
    market = SharedStateMarket(seed=4)
    demand, value = (np.asarray(column, np.float64) for column in market._base())
    beta, congestion, mass = market.beta, market.congestion, market.mass
    exact = exact_congestion_allocation(value, demand, beta, congestion, mass, market.budget)
    assert np.all(exact.allocation >= 0.0)
    assert math.fsum(exact.allocation) == pytest.approx(market.budget, rel=1e-15, abs=0.0)
    assert math.fsum(exact.shares) == pytest.approx(1.0, rel=1e-15, abs=0.0)
    full = list(exact.at_demand)
    assert np.max(np.abs(mass * exact.shares[full] - demand[full])) <= 1e-12 * mass
    level = np.log(exact.shares) / beta + congestion * exact.shares - value
    unpaid = [i for i in range(demand.size) if i not in full and i != exact.partial]
    assert np.max(np.abs(level[unpaid] - exact.level)) <= 1e-12
    paid = exact.allocation - exact.allocation[unpaid[0]]
    assert np.max(np.abs(level - exact.level - paid)) <= 1e-12  # Berry's inversion
    assert exact.spent <= market.budget < exact.cost_below
    assert exact.level_below < exact.level
    assert exact.value == pytest.approx(
        math.fsum(np.minimum(demand, mass * exact.shares)), rel=1e-15, abs=0.0
    )


def test_with_no_budget_the_exact_allocation_is_the_free_equilibrium() -> None:
    market = SharedStateMarket(seed=4)
    demand, value = (np.asarray(column, np.float64) for column in market._base())
    exact = exact_congestion_allocation(
        value, demand, market.beta, market.congestion, market.mass, 0.0
    )
    assert np.all(exact.allocation == 0.0)
    free, solved = _rides(value, demand, market.beta, market.congestion, market.mass, jnp.zeros(12))
    assert bool(solved)
    assert exact.value == pytest.approx(float(free), abs=1e-9)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"value": np.zeros(21), "demand": np.ones(21)}, "at most 20 zones"),
        ({"demand": np.ones(4)}, "both are one vector over the zones"),
        ({"demand": np.array([1.0, -0.5, 1.0])}, "a zone's demand is non-negative"),
        ({"value": np.array([1.0, np.nan, 1.0])}, "value is not finite"),
        ({"beta": 0.0}, "beta is 0.0; it is one finite number, positive"),
        ({"congestion": -1.0}, "congestion is -1.0; it is one finite number, non-negative"),
        ({"mass": math.inf}, "mass is inf"),
        ({"budget": -1.0}, "budget is -1.0"),
    ],
    ids=["zones", "shapes", "demand", "nan", "beta", "congestion", "mass", "budget"],
)
def test_inputs_outside_the_exact_allocation_s_contract_are_refused(change, message) -> None:
    """21 zones would read up to 22 million vertices a level."""
    given = {
        "value": np.ones(3),
        "demand": np.ones(3),
        "beta": 2.5,
        "congestion": 2.0,
        "mass": 3.0,
        "budget": 1.0,
    } | change
    with pytest.raises(ValueError, match=message):
        exact_congestion_allocation(**given)
