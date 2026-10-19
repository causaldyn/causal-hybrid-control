"""Game-theoretic control: agent equilibria + differentiable Stackelberg (leader) control.

Marketplaces have strategic, mobile agents, so SUTVA fails -- the platform is a Stackelberg *leader*
over the agents' equilibrium, and its action is optimised accounting for the induced best response.
These are the reusable methods (equilibrium solver + bilevel allocator); the benchmark *task* that
scores them (the zone-incentive game) lives in ``causaldyn-bench``.

The solver is a **certified** fixed point, not a fixed iteration count. :func:`fixed_point` runs to
a tolerance under ``lax.while_loop`` and differentiates by the implicit function theorem, so
(a) non-convergence is reported in :class:`EquilibriumSolution` instead of returned silently,
(b) the backward pass costs one adjoint solve rather than a tape over every iterate, and
(c) one contraction certificate covers both passes at once: the adjoint iteration
``w <- w_bar + (dT/dx)^T w`` is the Neumann series for ``(I - dT/dx)^{-1}``, so it converges
exactly when the forward map contracts.

The three certificates that price this solver and the leader on generated congestion games --
:func:`~chc.research.games.congestion_contraction_certificate`, which checks (c),
:func:`~chc.research.games.equilibrium_monotonicity_certificate` and
:func:`~chc.research.games.equilibrium_transfer_certificate` -- moved to :mod:`chc.research.games`
in 0.19.0.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from chc import _moved


def project_simplex(v: Array, z: float) -> Array:
    """Euclidean projection of ``v`` onto ``{u >= 0, sum u = z}`` (Duchi et al. 2008)."""
    n = v.shape[0]
    sorted_v = jnp.sort(v)[::-1]
    cssv = jnp.cumsum(sorted_v) - z
    rho = jnp.count_nonzero(sorted_v - cssv / (jnp.arange(n) + 1) > 0)
    theta = cssv[rho - 1] / rho
    return jnp.maximum(v - theta, 0.0)


class EquilibriumSolution(eqx.Module):
    """A solved fixed point together with the evidence that it *is* one.

    ``converged`` is a traced boolean, so it cannot raise inside ``jit`` -- checking it (or feeding
    it to a certificate) is the caller's job. That is the point: a non-equilibrium can no longer
    leave this module disguised as an equilibrium.
    """

    x: Array
    residual: Array  # ||x - T(x)|| / max(1, ||x||) at the returned point
    converged: Array  # residual <= tol


def _relative_residual(step_gap: Array, scale: Array) -> Array:
    """``||x - T(x)||`` measured against the size of ``x`` -- the scale-free convergence test.

    An ABSOLUTE tolerance is unusable here: the residual carries the units of ``x`` (a driver
    headcount), and float32 rounding puts a floor of about ``eps*||x||`` on it, so ``tol=1e-8``
    is unreachable at ``mass=12`` and reports a converged solve as failed. Relative-with-an-
    absolute-floor keeps one default honest from ``mass=1`` to ``mass=1e5``.
    """
    return jnp.linalg.norm(step_gap) / jnp.maximum(1.0, jnp.linalg.norm(scale))


def _iterate(
    step: Callable[[Array], Array], x_init: Array, tol: float, max_iter: int
) -> tuple[Array, Array]:
    """Iterate ``x <- step(x)`` until the relative residual is ``<= tol``, or ``max_iter``."""
    dtype = jnp.asarray(x_init).dtype

    def cond(carry: tuple[Array, Array, Array]) -> Array:
        _, residual, i = carry
        return (residual > tol) & (i < max_iter)

    def body(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        x, _, i = carry
        x_next = step(x)
        return x_next, _relative_residual(x_next - x, x_next).astype(dtype), i + 1

    init = (x_init, jnp.array(jnp.inf, dtype), jnp.array(0, jnp.int32))
    x, residual, _ = jax.lax.while_loop(cond, body, init)
    return x, residual


@partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4))
def _solve(
    f: Callable[[Any, Array], Array], params: Any, x_init: Array, tol: float, max_iter: int
) -> Array:
    return _iterate(lambda x: f(params, x), x_init, tol, max_iter)[0]


def _solve_fwd(
    f: Callable[[Any, Array], Array], params: Any, x_init: Array, tol: float, max_iter: int
) -> tuple[Array, tuple[Any, Array]]:
    x = _iterate(lambda z: f(params, z), x_init, tol, max_iter)[0]
    return x, (params, x)


def _solve_bwd(
    f: Callable[[Any, Array], Array],
    tol: float,
    max_iter: int,
    res: tuple[Any, Array],
    cotangent: Array,
) -> tuple[Any, Array]:
    """Implicit-function-theorem VJP: ``p_bar = (dT/dp)^T (I - dT/dx)^{-T} x_bar``.

    The inverse is applied as its own fixed point ``w <- x_bar + (dT/dx)^T w`` (the Neumann series),
    which converges under the same contraction that makes the forward solve well posed -- so one
    certificate covers both passes and neither materialises the Jacobian.
    """
    params, x = res
    _, vjp_x = jax.vjp(lambda z: f(params, z), x)
    w, _ = _iterate(lambda w: cotangent + vjp_x(w)[0], cotangent, tol, max_iter)
    _, vjp_params = jax.vjp(lambda p: f(p, x), params)
    return vjp_params(w)[0], jnp.zeros_like(x)


_solve.defvjp(_solve_fwd, _solve_bwd)


def fixed_point(
    f: Callable[[Any, Array], Array],
    params: Any,
    x_init: Array,
    *,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> EquilibriumSolution:
    """Solve ``x = f(params, x)`` to a tolerance, differentiable in ``params`` (implicit, not tape).

    ``tol`` is on the RELATIVE residual (see :func:`_relative_residual`); the default sits an order
    of magnitude above the float32 rounding floor, so a converged solve is never reported as failed.
    Reverse-mode only: ``custom_vjp`` defines no JVP, so ``jax.jacfwd`` through the solve will fail.
    The gradient is the exact one at the fixed point, independent of how many iterations the forward
    solve took -- which is what makes the tolerance-based ``lax.while_loop`` usable at all, since
    JAX has no reverse rule for it.
    """
    x = _solve(f, params, x_init, tol, max_iter)
    residual = _relative_residual(f(params, x) - x, x)
    return EquilibriumSolution(x=x, residual=residual, converged=residual <= tol)


def softmax_congestion_equilibrium(
    attract: Array,
    u: Array,
    congestion: float,
    mass: float,
    beta: float = 2.5,
    *,
    damping: float = 0.5,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> EquilibriumSolution:
    """Agent best-response equilibrium: the damped softmax congestion fixed point of the mass.

    Agents flow toward higher value ``attract + u - congestion*x/mass`` (crowding lowers it); the
    iteration ``T(x) = (1-d)*x + d*mass*softmax(beta*value)`` converges to the Wardrop/logit
    equilibrium (``sum x = mass``, since the softmax Jacobian annihilates the constants). The
    uniform Jacobian bound certifies contraction when ``0 < damping < 4/(2 + beta*congestion)`` --
    at the default ``d = 1/2`` that is exactly ``beta*congestion < 6``. Above it the iteration
    *may* 2-cycle (the bound is sufficient, not necessary), which is why the result
    carries its own residual; the remedy is a smaller ``damping``, not a different game, since the
    equilibrium itself stays non-expansive to operator perturbations
    (:func:`equilibrium_transfer_certificate`).
    :func:`congestion_damping` returns the fastest contracting choice for a given game.
    """

    def step(params: tuple[Array, Array], x: Array) -> Array:
        a, incentive = params
        value = beta * (a + incentive - congestion * x / mass)
        return (1.0 - damping) * x + damping * mass * jax.nn.softmax(value)

    x_init = jnp.full(attract.shape[0], mass / attract.shape[0])
    return fixed_point(step, (attract, u), x_init, tol=tol, max_iter=max_iter)


def congestion_damping(beta: float, congestion: float) -> float:
    """Modulus-optimal damping ``4/(4 + beta*congestion)`` for the logit congestion iteration.

    The two extreme eigenvalues of ``T'`` are ``1-d`` and ``1 - d*(1 + kappa/2)``; equalising their
    magnitudes gives ``d* = 4/(4+kappa)``, and any ``0 < d < 4/(2+kappa)`` clears the uniform bound.
    So EVERY game has a contracting damping -- the ``kappa < 6`` ceiling is a property of the
    hard-coded ``d = 1/2``, not of the game. Kept opt-in: the default stays ``1/2`` so shipped
    numbers do not move.
    """
    return 4.0 / (4.0 + beta * congestion)


def congestion_contraction_modulus(beta: float, congestion: float, damping: float = 0.5) -> float:
    """Certified contraction modulus ``mu = 1 - ||T'||_2`` of the damped logit congestion map.

    ``T'(x) = (1-d)*I - d*beta*c*(diag(s) - s s^T)``, and ``v^T (diag(s) - s s^T) v = Var_s(v)``, so
    by Popoviciu the softmax Jacobian has spectrum in ``[0, 1/2]`` (sharp at ``s = (1/2,1/2,0..)``).
    ``T'`` is symmetric, hence ``||T'||_2 <= max(1-d, d*(1 + beta*c/2) - 1)``, and that UNIFORM
    bound leaves a positive modulus iff ``0 < d < 4/(2 + beta*c)`` -- at ``d = 1/2`` exactly
    ``beta*c < 6``. Both sides of the interval bind: at ``d = 0`` the map is the identity and the
    ``1-d`` branch is already 1. Sharp over the class, SUFFICIENT for a fixed game: the worst-case
    eigenvalue is attained only at a two-point uniform ``s``; measured, the default map still
    converges at ``beta*c = 7`` and 2-cycles from ``8``. Derived in
    ``validation/congestion_contraction.mac``, cross-checked in ``.py``;
    ``proofs/congestion_contraction.v`` proves the ``d = 1/2`` case, ``beta*c < 6``, and the
    general-``d`` interval is Maxima's.

    This is the SOLVER's rate, not the equilibrium's conditioning -- see
    :func:`equilibrium_transfer_certificate`, where using ``1/mu`` as a sensitivity constant is
    measured to overstate the truth by up to 200x.
    """
    return 1.0 - max(1.0 - damping, damping * (1.0 + beta * congestion / 2.0) - 1.0)


def stackelberg_allocation(
    objective: Callable[[Array], Array],
    n: int,
    budget: float,
    steps: int = 400,
    lr: float = 0.05,
) -> Array:
    """Differentiable-bilevel leader allocation: maximise ``objective(u)`` over the budget simplex.

    ``objective`` is evaluated *through* the equilibrium (so ``jax.grad`` differentiates the bilevel
    problem). Adam handles the scale; the plan is projected onto the budget simplex each step, and
    the best feasible allocation seen is returned.

    The ascent starts at the even split and finds a local maximum, not the global one: the
    objective through an equilibrium need not be concave in ``u``. Three zones of attraction
    ``(0.1, 0.5, 2.0)``, worth ``(1, 1, 0.2)`` a driver, at congestion 1.5 and mass 6, end with the
    whole budget of 1 on the first zone, worth 2.973, where the whole budget on the second is worth
    3.184. Compare the result with the vertices or other starts where the global plan matters.
    """
    u = jnp.full(n, budget / n)
    grad_fn = jax.jit(jax.grad(lambda u: -objective(u)))
    value_fn = jax.jit(objective)
    optimizer = optax.adam(lr)
    state = optimizer.init(u)
    best_u, best_val = u, float(value_fn(u))
    for _ in range(steps):
        updates, state = optimizer.update(grad_fn(u), state)
        u = project_simplex(jnp.asarray(optax.apply_updates(u, updates)), budget)
        val = float(value_fn(u))
        if val > best_val:
            best_u, best_val = u, val
    return best_u


# The research half of this module moved to chc.research.games in 0.19.0. Each of its names
# is served from here with a DeprecationWarning until 0.20.0 (docs/adr/0117).
_MOVED = frozenset(
    {
        "congestion_contraction_certificate",
        "CongestionContractionCertificate",
        "equilibrium_monotonicity_certificate",
        "equilibrium_transfer_certificate",
        "EquilibriumMonotonicityCertificate",
        "EquilibriumTransferCertificate",
    }
)

if TYPE_CHECKING:
    # isort: off
    from chc.research.games import (
        congestion_contraction_certificate as congestion_contraction_certificate,
        CongestionContractionCertificate as CongestionContractionCertificate,
        equilibrium_monotonicity_certificate as equilibrium_monotonicity_certificate,
        equilibrium_transfer_certificate as equilibrium_transfer_certificate,
        EquilibriumMonotonicityCertificate as EquilibriumMonotonicityCertificate,
        EquilibriumTransferCertificate as EquilibriumTransferCertificate,
    )
    # isort: on
else:
    __getattr__ = _moved.serve_names(__name__, "chc.research.games", _MOVED)
