"""The research half of :mod:`chc.games`: certificates that price its solver and its leader.

Each solves generated logit congestion games with :func:`~chc.games.fixed_point` and
:func:`~chc.games.softmax_congestion_equilibrium` and measures one claim on them: the contraction
bound at the equilibrium returned (:func:`congestion_contraction_certificate`), the global
strong-monotonicity modulus of the equilibrium map and the bound it buys
(:func:`equilibrium_monotonicity_certificate`), and how a model error in the game transfers to the
leader's regret (:func:`equilibrium_transfer_certificate`). They sit outside the surface 0.20.0
freezes (:mod:`chc.research`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from chc.games import (
    congestion_contraction_modulus,
    congestion_damping,
    fixed_point,
    project_simplex,
    softmax_congestion_equilibrium,
)


@dataclass(frozen=True)
class CongestionContractionCertificate:
    """Numeric evidence that the certified Jacobian bound holds at the solved equilibrium."""

    kappa: float  # beta * congestion
    damping: float  # the iteration's damping d
    jacobian_bound: float  # certified max(1-d, d*(1 + kappa/2) - 1)
    modulus: float  # mu = 1 - the UNIFORM jacobian bound; > 0 certifies contraction (sufficient)
    certified: bool  # 0 < d < 4/(2 + kappa)
    measured_operator_norm: float  # ||T'(x*)||_2 at the solved point
    measured_residual: float  # relative fixed-point residual at the returned point
    ok: bool  # measured norm under the bound, and (if certified) the solve converged


def _congestion_step(
    attract: Array, congestion: float, mass: float, beta: float, damping: float
) -> Callable[[Array], Array]:
    """The damped best-response map at zero incentive -- what both certificates differentiate."""

    def step(x: Array) -> Array:
        value = beta * (attract - congestion * x / mass)
        return (1.0 - damping) * x + damping * mass * jax.nn.softmax(value)

    return step


def congestion_contraction_certificate(
    beta: float = 2.5,
    congestion: float = 2.0,
    n_zones: int = 12,
    seed: int = 0,
    damping: float = 0.5,
) -> CongestionContractionCertificate:
    """Solve one congestion equilibrium and confirm the contraction bound at the point returned."""
    attract = jax.random.normal(jax.random.key(seed), (n_zones,))
    mass = float(n_zones)
    solution = softmax_congestion_equilibrium(
        attract, jnp.zeros(n_zones), congestion, mass, beta, damping=damping
    )
    step = _congestion_step(attract, congestion, mass, beta, damping)
    measured = float(jnp.linalg.norm(jax.jacobian(step)(solution.x), 2))
    bound = max(1.0 - damping, damping * (1.0 + beta * congestion / 2.0) - 1.0)
    certified = damping < 4.0 / (2.0 + beta * congestion)
    return CongestionContractionCertificate(
        kappa=beta * congestion,
        damping=damping,
        jacobian_bound=bound,
        modulus=congestion_contraction_modulus(beta, congestion, damping),
        certified=certified,
        measured_operator_norm=measured,
        measured_residual=float(solution.residual),
        ok=measured <= bound + 1e-6 and (not certified or bool(solution.converged)),
    )


@dataclass(frozen=True)
class EquilibriumMonotonicityCertificate:
    """Global strong monotonicity of ``F = I - S``, and the finite-perturbation bound it licenses.

    Result 39 (b) bounds ``||(I - S')^{-1}||`` at the equilibrium -- an implicit-function
    derivative, so it prices *infinitesimal* perturbations and says as much. These are the global
    quantities: a modulus that holds at every point of a box, and a displacement measured at a
    perturbation of finite size.
    """

    kappa: float
    ambient_modulus: float  # min over the box of lambda_min of the symmetric part of F'; exactly 1
    tangent_modulus: float  # the same restricted to the fixed-mass tangent space 1-perp
    tangent_bound: float  # 1 + kappa*n*s_min^2 at the worst sampled s -- the certified bound
    box_bound: float  # 1 + kappa*exp(-2*spread)/n, computable from the BOX with no solve at all
    local_tangent_modulus: float  # the tangent modulus on a ball around the equilibrium instead
    local_tangent_bound: float  # and its certified bound, where the improvement over 1 is real
    jacobian_agreement: float  # max |autodiff F' - the closed form|; a precision check, not a gate
    finite_monotonicity: float  # min over random pairs of <F(x)-F(y), x-y>/||x-y||^2
    displacement_ratio: float  # max ||x*(S+g)-x*(S)||/||g||; monotonicity caps it at 1/mu
    ok: bool


def _tangent_basis(n: int) -> Array:
    """Orthonormal basis of ``1-perp``: the directions a mass-conserving equilibrium can move."""
    basis = jnp.eye(n) - jnp.ones((n, n)) / n
    q, _ = jnp.linalg.qr(basis)
    return q[:, : n - 1]


def equilibrium_monotonicity_certificate(
    beta: float = 2.5,
    congestion: float = 2.0,
    n_zones: int = 6,
    mass: float = 6.0,
    samples: int = 400,
    perturbations: int = 24,
    seed: int = 0,
) -> EquilibriumMonotonicityCertificate:
    """Measure the GLOBAL modulus Result 39 (b) left open, and the bound it buys.

    ``F'(x) = I + kappa*J(s(x))`` with ``J = diag(s) - s s^T`` a covariance matrix, so ``F' >= I``
    at *every* ``x``, not only at the equilibrium: ``F`` is 1-strongly monotone globally, and
    ``1`` cannot be improved because ``J`` annihilates the constants. The direction that saturates
    it is the one mass conservation forbids, so on the fixed-mass tangent space the modulus is
    strictly larger, bounded below by ``1 + kappa*n*s_min^2`` -- exactly attained at the two-point
    uniform ``s``, the same configuration that makes Result 39 (a)'s Popoviciu bound sharp
    (``validation/equilibrium_monotonicity.mac``, ``proofs/equilibrium_monotonicity.v``; z3 and cvc5
    both return unsat on the negation at ``n = 3``).

    Measurements, each able to fail. ``ambient_modulus`` and ``tangent_modulus`` are minimised over
    a box rather than evaluated at a point; ``tangent_bound`` and ``box_bound`` are the certified
    lower bounds, the second needing neither a sample nor a solve. ``finite_monotonicity`` tests the
    definition itself on random *pairs*, which no derivative argument implies.
    ``displacement_ratio`` perturbs the operator by a finite amount and measures how far the
    equilibrium moves: strong monotonicity at ``mu`` caps it at ``1/mu``, one-for-one at ``mu = 1``.

    Reported at two scales on purpose, because the answers differ and only one of them is the
    headline. Over the whole box the tangent improvement **evaporates**: near a corner the softmax
    approaches a vertex, ``s_min`` collapses, and both the measured tangent modulus and its bound
    fall back to 1. The strictly-better constant of Result 39 (b) is a statement about a
    neighbourhood of an interior equilibrium, and ``local_tangent_modulus`` is where it has teeth --
    quoting it globally is exactly the mistake this certificate exists to prevent.
    """
    key = jax.random.key(seed)
    k_attract, k_box, k_pairs, k_perturb = jax.random.split(key, 4)
    attract = jax.random.normal(k_attract, (n_zones,))
    kappa = beta * congestion

    def response(x: Array) -> Array:
        return mass * jax.nn.softmax(beta * (attract - congestion * x / mass))

    def residual_map(x: Array) -> Array:
        return x - response(x)

    q = np.asarray(_tangent_basis(n_zones), dtype=np.float64)

    def moduli(points: Array) -> tuple[float, float, float, float]:
        """Ambient and tangent moduli over a sample, the certified bound, and autodiff agreement.

        ``F'(x) = I + kappa*(diag(s) - s s^T)`` in closed form, and the eigen-solves run on that in
        float64 -- because the quantity under test is *exactly* 1, and taking it from a float32
        autodiff Jacobian misses by ~2e-7, which is larger than any tolerance worth calling a gate.
        The autodiff Jacobian is still computed and compared, so the shipped map is checked rather
        than assumed to be the map the algebra describes; that comparison is a precision question
        and carries a precision tolerance, which the modulus does not.
        """
        probs = np.asarray(jax.vmap(lambda x: response(x) / mass)(points), dtype=np.float64)
        # Renormalising is not a guard, it is the definition: `s` is a probability vector, and a
        # float32 softmax sums to 1 only to ~1e-7. That residual mass is exactly what stops `J`
        # from annihilating the constants, and it shows up as a modulus 2e-7 BELOW 1 -- an
        # artefact of the sample, not of the algebra the certificate is testing.
        probs = probs / probs.sum(axis=1, keepdims=True)
        gram = np.einsum("ni,nj->nij", probs, probs)
        jac = np.eye(n_zones) + kappa * (np.einsum("ij,ni->nij", np.eye(n_zones), probs) - gram)
        autodiff = np.asarray(jax.vmap(jax.jacobian(residual_map))(points), dtype=np.float64)
        agreement = float(np.max(np.abs(autodiff - jac)))
        ambient = float(np.min(np.linalg.eigvalsh(jac)[:, 0]))
        projected = np.einsum("ji,njk,kl->nil", q, jac, q)
        tangent = float(np.min(np.linalg.eigvalsh(projected)[:, 0]))
        return ambient, tangent, 1.0 + kappa * n_zones * float(np.min(probs)) ** 2, agreement

    xs = jax.random.uniform(k_box, (samples, n_zones), minval=0.0, maxval=mass)
    ambient, tangent, bound, agreement = moduli(xs)

    # The box bound needs no solve and no sample: over x in [0, mass]^n the logits span at most
    # beta*(ptp(attract) + congestion), and softmax turns that into s_min >= exp(-spread)/n.
    spread = float(beta * (jnp.max(attract) - jnp.min(attract) + congestion))
    box = 1.0 + kappa * float(jnp.exp(-2.0 * spread)) / n_zones

    pairs = jax.random.uniform(k_pairs, (2, samples, n_zones), minval=0.0, maxval=mass)
    gaps = pairs[0] - pairs[1]
    deltas = jax.vmap(residual_map)(pairs[0]) - jax.vmap(residual_map)(pairs[1])
    finite = float(jnp.min(jnp.sum(deltas * gaps, axis=1) / jnp.sum(gaps * gaps, axis=1)))

    damping = congestion_damping(beta, congestion)
    base = fixed_point(
        lambda _, x: (1.0 - damping) * x + damping * response(x),
        None,
        jnp.full(n_zones, mass / n_zones),
    )
    k_local = jax.random.fold_in(k_box, 1)
    local = base.x + 0.05 * mass * jax.random.normal(k_local, (samples, n_zones))
    _, local_tangent, local_bound, _ = moduli(local)
    offsets = 0.05 * mass * jax.random.normal(k_perturb, (perturbations, n_zones))

    def displaced(offset: Array) -> Array:
        moved = fixed_point(
            lambda g, x: (1.0 - damping) * x + damping * (response(x) + g),
            offset,
            base.x,
        )
        return jnp.linalg.norm(moved.x - base.x) / jnp.linalg.norm(offset)

    ratio = float(jnp.max(jax.vmap(displaced)(offsets)))
    return EquilibriumMonotonicityCertificate(
        kappa=kappa,
        ambient_modulus=ambient,
        tangent_modulus=tangent,
        tangent_bound=bound,
        box_bound=box,
        local_tangent_modulus=local_tangent,
        local_tangent_bound=local_bound,
        jacobian_agreement=agreement,
        finite_monotonicity=finite,
        displacement_ratio=ratio,
        ok=(
            abs(ambient - 1.0) < 1e-9  # exactly 1: attained on the constants, never below
            and tangent >= bound - 1e-9
            and local_tangent >= local_bound - 1e-9
            and local_bound > bound  # the local improvement is real where the global one is not
            and bound >= box - 1e-12
            and finite >= 1.0 - 1e-6  # the definition, on pairs, not a derivative
            and ratio <= 1.0 + 1e-6  # 1/mu at mu = 1
            and agreement < 1e-4  # the shipped map IS I + kappa*J, to the working precision
            and bool(base.converged)
        ),
    )


@dataclass(frozen=True)
class EquilibriumTransferCertificate:
    """Result 39: the equilibrium layer transfers the estimation-error ORDER and is non-expansive.

    ``conditioning`` is the ambient ``||(I - S')^{-1}||_2`` for the UNDAMPED best response ``S`` --
    the local (implicit-function) constant relating an infinitesimal operator perturbation to the
    equilibrium displacement. It is exactly ``1`` for every ``kappa``, but only because the softmax
    Jacobian annihilates the constants: the value is attained in the mass direction ``1``, which
    equilibria never move along (``sum x = mass`` is conserved). ``tangent_conditioning`` is the
    number that actually binds -- the same operator restricted to the fixed-mass tangent space
    ``1^perp``, equal to ``1 / (1 + kappa * lambda_min^+(J))`` and *strictly* below 1 for an
    interior distribution, though it approaches 1 as the equilibrium degenerates toward a vertex.

    The contraction modulus of the damped solver is a different quantity again; ``naive_bounds``
    records ``1/mu`` and ``looseness`` how far it overstates the ambient truth (infinite once the
    solver stops contracting, while the equilibrium itself does not move).
    """

    kappas: tuple[float, ...]
    conditioning: tuple[float, ...]  # ambient ||(I - S')^{-1}||_2 -- exactly 1, via the mass mode
    tangent_conditioning: tuple[float, ...]  # the same on 1^perp: 1/(1 + kappa*lambda_min^+(J)) < 1
    naive_bounds: tuple[float, ...]  # 1/mu from the damped contraction modulus (inf if mu <= 0)
    looseness: tuple[float, ...]  # naive / ambient
    operator_errors: tuple[float, ...]  # perturbation sizes fed to the leader problem
    regrets: tuple[float, ...]  # leader regret on the TRUE game, interior optimum
    regret_slope: float  # log-log slope of regret vs error; theory says 2
    constrained_slope: float  # the same with the budget constraint ACTIVE -- the order collapses
    vertex_regret: tuple[float, ...]  # regret at a vertex optimum; identically 0, so no slope
    ok: bool  # ambient conditioning at 1, tangent strictly inside it, slope quadratic


def _simplex_leader(
    objective: Callable[[Array], Array], n: int, budget: float, steps: int, lr: float
) -> Array:
    """Projected Adam on the budget simplex -- :func:`_unconstrained_leader`'s constrained sibling.

    Exists to make the regularity assumption of Result 39 (c) falsifiable: the interior arm is what
    the order claim is about, and this is the arm where it breaks.
    """
    u = project_simplex(jnp.full(n, budget / n), budget)
    optimizer = optax.adam(lr)
    state = optimizer.init(u)
    grad_fn = jax.grad(lambda v: -objective(v))

    @jax.jit
    def step(u: Array, state: Any) -> tuple[Array, Any]:
        updates, state = optimizer.update(grad_fn(u), state)
        return project_simplex(jnp.asarray(optax.apply_updates(u, updates)), budget), state

    for _ in range(steps):
        u, state = step(u, state)
    return u


def _log_slope(errors: Sequence[float], regrets: Sequence[float]) -> float:
    """Log-log slope of regret against operator error; the floor keeps an exact zero finite."""
    logged = jnp.log(jnp.maximum(jnp.array(regrets), 1e-16))
    return float(jnp.polyfit(jnp.log(jnp.array(errors)), logged, 1)[0])


def _unconstrained_leader(
    objective: Callable[[Array], Array], n: int, steps: int, lr: float
) -> Array:
    """Adam ascent with NO constraint -- the interior optimum the quadratic order needs.

    ``stackelberg_allocation`` projects onto the budget simplex, and at an active constraint the
    curvature collapses: measured, the regret slope drops from 2 to ~0.9, and at a vertex it is
    identically 0 because the argmax is locally constant in the operator. The order claim is about
    the interior case, so the certificate must measure the interior case.
    """
    u = jnp.zeros(n)
    grad_fn = jax.jit(jax.grad(lambda v: -objective(v)))
    optimizer = optax.adam(lr)
    state = optimizer.init(u)
    for _ in range(steps):
        updates, state = optimizer.update(grad_fn(u), state)
        u = jnp.asarray(optax.apply_updates(u, updates))
    return u


def equilibrium_transfer_certificate(
    n_zones: int = 6,
    seed: int = 0,
    incentive_cost: float = 2.0,
    budget: float = 3.0,
    steps: int = 900,
    errors: Sequence[float] = (0.4, 0.2, 0.1, 0.05),
    kappas: Sequence[float] = (4.0, 5.0, 5.5, 5.8, 5.96),
) -> EquilibriumTransferCertificate:
    """Measure the two halves of Result 39 on the logit congestion Stackelberg game.

    Half one, across ``kappas``: the local conditioning ``||(I - S')^{-1}||_2`` -- both the ambient
    norm and its restriction to the fixed-mass tangent space, which is the one an equilibrium
    displacement can actually excite -- against the naive contraction constant ``1/mu``. Half two,
    across ``errors``: perturb the agents' operator by ``e``, re-solve the leader problem through
    the perturbed equilibrium, and score the resulting plan on the TRUE game -- the regret must be
    second order in ``e``.

    Both conditioning numbers are *local*: they are implicit-function derivatives at the
    equilibrium, so they bound the response to infinitesimal perturbations. The global
    inverse-Lipschitz statement -- strong monotonicity of ``F(x) = x - S(x)`` -- is
    :func:`equilibrium_monotonicity_certificate`, which measures it over a box and at a finite
    perturbation size rather than deducing it from these derivatives.

    The order half carries a **binding regularity assumption**, and the last two fields make it
    falsifiable rather than merely stated. The quadratic order needs the leader's optimum to move
    smoothly with the operator, which an active constraint destroys: with the budget constraint
    active the measured slope falls from ``1.97`` to ``1.14``, and at a **vertex** the argmax is
    locally constant, so the regret is *identically zero* across the whole sweep and the log-log
    slope does not exist. The vertex case is not a degraded rate but the absence of one, and any
    statement of the form "the equilibrium transfers the order" is vacuous there.
    """
    key_a, key_w, key_p = jax.random.split(jax.random.key(seed), 3)
    attract = jax.random.normal(key_a, (n_zones,))
    weights = jnp.abs(jax.random.normal(key_w, (n_zones,))) + 0.2
    direction = jax.random.normal(key_p, (n_zones,))
    direction = direction / jnp.linalg.norm(direction)
    mass, congestion = float(n_zones), 2.0

    # orthonormal basis of the fixed-mass tangent space 1^perp, where equilibrium moves live
    tangent_basis = jnp.linalg.qr(jnp.eye(n_zones) - jnp.ones((n_zones, n_zones)) / n_zones)[0][
        :, : n_zones - 1
    ]

    conditioning, tangent, naive, loose = [], [], [], []
    for kappa in kappas:
        beta = kappa / congestion
        solution = softmax_congestion_equilibrium(
            attract, jnp.zeros(n_zones), congestion, mass, beta
        )
        undamped = _congestion_step(attract, congestion, mass, beta, 1.0)
        jac = jax.jacobian(undamped)(solution.x)
        resolvent = jnp.linalg.inv(jnp.eye(n_zones) - jac)
        exact = float(jnp.linalg.norm(resolvent, 2))
        modulus = congestion_contraction_modulus(beta, congestion)
        conditioning.append(exact)
        tangent.append(float(jnp.linalg.norm(tangent_basis.T @ resolvent @ tangent_basis, 2)))
        naive.append(1.0 / modulus if modulus > 0.0 else float("inf"))
        loose.append(naive[-1] / exact)

    beta = 2.5

    def leader_value(operator: Array, u: Array) -> Array:
        equilibrium = softmax_congestion_equilibrium(operator, u, congestion, mass, beta).x
        return jnp.dot(weights, equilibrium) - 0.5 * incentive_cost * jnp.dot(u, u)

    def plan(operator: Array) -> Array:
        return _unconstrained_leader(lambda u: leader_value(operator, u), n_zones, steps, lr=0.02)

    best = float(leader_value(attract, plan(attract)))
    regrets = [best - float(leader_value(attract, plan(attract + e * direction))) for e in errors]

    # Two arms that can falsify the regularity the order claim needs. On the budget simplex with the
    # sum constraint active the plan still moves, so a slope exists and is measurably worse than 2;
    # dropping the quadratic incentive cost sends the optimum to a vertex, where it stops moving at
    # all and the notion of a slope goes with it.
    def constrained_plan(operator: Array) -> Array:
        return _simplex_leader(lambda u: leader_value(operator, u), n_zones, budget, steps, lr=0.05)

    def vertex_value(operator: Array, u: Array) -> Array:
        equilibrium = softmax_congestion_equilibrium(operator, u, congestion, mass, beta).x
        return jnp.dot(weights, equilibrium)

    def vertex_plan(operator: Array) -> Array:
        return _simplex_leader(lambda u: vertex_value(operator, u), n_zones, budget, steps, lr=0.05)

    constrained_best = float(leader_value(attract, constrained_plan(attract)))
    constrained = [
        constrained_best - float(leader_value(attract, constrained_plan(attract + e * direction)))
        for e in errors
    ]
    vertex_best = float(vertex_value(attract, vertex_plan(attract)))
    vertex = [
        vertex_best - float(vertex_value(attract, vertex_plan(attract + e * direction)))
        for e in errors
    ]
    slope = _log_slope(errors, regrets)
    return EquilibriumTransferCertificate(
        kappas=tuple(float(k) for k in kappas),
        conditioning=tuple(conditioning),
        tangent_conditioning=tuple(tangent),
        naive_bounds=tuple(naive),
        looseness=tuple(loose),
        operator_errors=tuple(float(e) for e in errors),
        regrets=tuple(regrets),
        regret_slope=slope,
        constrained_slope=_log_slope(errors, constrained),
        vertex_regret=tuple(vertex),
        ok=all(abs(c - 1.0) < 1e-3 for c in conditioning)
        and all(t <= c + 1e-9 for t, c in zip(tangent, conditioning, strict=True))
        and 1.7 <= slope <= 2.3
        and _log_slope(errors, constrained) < slope - 0.5  # an active constraint costs an order
        and all(r == 0.0 for r in vertex),  # a vertex plan does not move, so there is no slope
    )
