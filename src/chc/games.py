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

The leader's ascent, :func:`stackelberg_allocation`, finds a local maximum of any objective. For
the logit congestion game with completed rides as the objective,
:func:`exact_congestion_allocation` finds the global one, and the vertex that certifies it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array
from jax.typing import ArrayLike
from numpy.typing import NDArray

from chc.frames import _real_numbers


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
    (:func:`~chc.research.games.equilibrium_transfer_certificate`).
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
    """Differentiable-bilevel leader allocation: an ascent on ``objective(u)`` over the budget
    simplex.

    ``objective`` is evaluated *through* the equilibrium (so ``jax.grad`` differentiates the bilevel
    problem). Adam handles the scale; the plan is projected onto the budget simplex each step, and
    the best feasible allocation seen is returned.

    The ascent starts at the even split and ends at a local maximum at best, not the global one:
    the objective through an equilibrium need not be concave in ``u``, and ``steps`` may end it
    short of either. Three zones of attraction ``(0.1, 0.5, 2.0)``, worth ``(1, 1, 0.2)`` a driver,
    at congestion 1.5 and mass 6, end with the whole budget of 1 on the first zone, worth 2.973,
    where the whole budget on the second is worth 3.184. Compare the result with the vertices or
    other starts where the global plan matters. Where the objective is the rides a logit
    congestion equilibrium completes, :func:`exact_congestion_allocation` finds the global plan:
    on the 30 markets ``SharedStateMarket(seed=0..29)`` in float64, 2500 steps of this ascent
    ended short of it on 17, by up to 0.215 rides.
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


_EPS = float(np.finfo(np.float64).eps)

_EXACT_ZONES = 20
"""The most zones :func:`exact_congestion_allocation` enumerates. A level reads ``k * 2**(k - 1)``
vertices over the ``k`` zones whose floor lies below their demand, and a solve at most
``2*j + 51`` levels, where the search below the free level doubles its reach ``j`` times: at 20
zones a level reads up to 10.5 million vertices, at 21 twice that."""

_NEWTON_STEPS = 64
"""Newton steps at most for a floor share, whose start lies within 1 of the root: quadratic
convergence reaches the last bit in about six."""


@dataclass(frozen=True)
class CongestionAllocation:
    """The global optimum :func:`exact_congestion_allocation` returns, and its certificate.

    The allocation completes ``value`` rides. At its equilibrium the zones ``at_demand`` hold
    exactly their demand, ``partial`` holds more drivers than its floor and fewer than its demand,
    and every other zone sits at its floor share at ``level``, funded by nothing but its share of
    the budget left over, which moves no driver. Reaching that costs ``spent``, at most the budget.
    No allocation completes more: at ``level_below``, a level just below ``level``, every vertex
    costs ``cost_below`` or more, above the budget, or no vertex fills the deficit without a zone
    passing its demand (``cost_below`` is inf); and an allocation that completed more would reach
    a vertex of a level that low (the exchange step of :func:`exact_congestion_allocation`).
    """

    allocation: NDArray[np.float64]  # (n,) the incentive per zone: non-negative, summing to budget
    value: float  # the rides completed at the allocation's equilibrium
    level: float  # mu: log(s)/beta + congestion*s - value of each zone at its floor
    shares: NDArray[np.float64]  # (n,) the drivers' shares at the allocation's equilibrium
    at_demand: tuple[int, ...]  # the zones funded to complete their whole demand
    partial: int | None  # the funded zone between its floor and its demand; None if there is none
    spent: float  # the least budget that reaches ``level``; the rest is spread evenly
    level_below: float  # the level, just below ``level``, at which the budget falls short
    cost_below: float  # the least cost of a vertex at ``level_below``: above the budget, or inf
    vertices: int  # the vertices whose cost the solve evaluated, over every level it read


def _floor_log_shares(
    level: float, value: NDArray[np.float64], beta: float, congestion: float
) -> NDArray[np.float64]:
    """``log(g_i)``: the log share at which a zone's level ``log(s)/beta + congestion*s - value_i``
    reads ``level``.

    With ``t = log(s)``, ``k = beta*congestion`` and ``z = beta*(level + value_i)`` the equation is
    ``t + k*exp(t) = z``, whose root is ``z - W(k*exp(z))`` (Lambert's ``W``), increasing and
    convex in ``t``, so Newton's iterates fall monotonically onto it from the right and overshoot
    to the right from the left. The start, ``z`` where ``L = log(k) + z <= 1`` and
    ``log(L) - log(k)`` past it, lies within 1 of the root: ``0 <= W(e**L) <= 1`` on the first
    side, and ``log(y) - log(log(y)) <= W(y) <= log(y) - log(log(y)) + 0.582`` for ``y >= e``
    (Hoorfar and Hassani 2008) on the second. Read in logarithms no step overflows, and a share
    below the smallest float keeps its level.
    """
    z = beta * (level + value)
    if congestion == 0.0:
        return z
    log_k = math.log(beta * congestion)
    reach = log_k + z
    t = np.where(reach > 1.0, np.log(np.maximum(reach, 1.0)) - log_k, z)
    for _ in range(_NEWTON_STEPS):
        grow = np.exp(t + log_k)
        step = (t + grow - z) / (1.0 + grow)
        t = t - step
        if np.all(np.abs(step) <= 4.0 * _EPS * np.maximum(1.0, np.abs(t))):
            break
    return t


def _subset_sums(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """The sum of ``values`` over every subset, at the index whose bit ``i`` holds element ``i``."""
    sums = np.zeros(1)
    for entry in values:
        sums = np.concatenate([sums, sums + entry])
    return sums


def _cheapest_vertex(
    level: float,
    value: NDArray[np.float64],
    demand_share: NDArray[np.float64],
    beta: float,
    congestion: float,
) -> tuple[float, NDArray[np.float64] | None, NDArray[np.float64] | None, tuple[int, ...], int]:
    """The least budget that holds every unfunded zone at its floor share at ``level`` and fills the
    deficit with no zone past its demand, over every vertex of that box cut by the deficit's
    hyperplane; with the vertex's shares and incentives, the zones it funds in full, and the count
    of vertices read. The cost is inf where the zones below their demand cannot hold the deficit.
    """
    t = _floor_log_shares(level, value, beta, congestion)
    floor = np.exp(t)
    deficit = 1.0 - math.fsum(floor)
    n = floor.size
    if deficit <= 0.0:
        return 0.0, floor / math.fsum(floor), np.zeros(n), (), 0
    room = demand_share - floor
    open_zones = np.flatnonzero(room > 0.0)
    k = open_zones.size
    slack = 2.0 * max(k, 1) * _EPS  # the rounding of a sum of up to k shares
    if k == 0 or math.fsum(room[open_zones]) < deficit - slack:
        return math.inf, None, None, (), 0
    caps, logs = room[open_zones], t[open_zones]
    with np.errstate(divide="ignore"):
        # phi(floor + lift) - level, in logarithms: exact where the lift is zero or the floor tiny
        def cost(zone: int | NDArray[np.int64], lift: NDArray[np.float64]) -> NDArray[np.float64]:
            return (np.logaddexp(logs[zone], np.log(lift)) - logs[zone]) / beta + congestion * lift

        fill = cost(np.arange(k), caps)
    cap_sums, fill_sums = _subset_sums(caps), _subset_sums(fill)
    best, choice = math.inf, None
    for zone in range(k):
        # the subsets that leave ``zone`` out, ``zone`` taking what they leave of the deficit
        keep = (slice(None), 0, slice(None))
        others = cap_sums.reshape(-1, 2, 1 << zone)[keep].ravel()
        need = deficit - others
        fits = np.flatnonzero((need >= -slack) & (need <= caps[zone] + slack))
        if not fits.size:
            continue
        lift = np.clip(need[fits], 0.0, caps[zone])
        with np.errstate(divide="ignore"):
            part = cost(zone, lift)
        total = fill_sums.reshape(-1, 2, 1 << zone)[keep].ravel()[fits] + part
        at = int(np.argmin(total))
        if total[at] < best:
            best, choice = float(total[at]), (zone, int(fits[at]), float(lift[at]), float(part[at]))
    vertices = k << (k - 1)
    if choice is None:
        return math.inf, None, None, (), vertices
    zone, position, lift_zone, part_zone = choice
    subset = (position >> zone << (zone + 1)) | (position & ((1 << zone) - 1))
    members = [i for i in range(k) if subset >> i & 1]
    shares, incentive = floor.copy(), np.zeros(n)
    shares[open_zones[members]] += caps[members]
    incentive[open_zones[members]] = fill[members]
    shares[open_zones[zone]] += lift_zone
    incentive[open_zones[zone]] = part_zone
    full = members + ([zone] if lift_zone >= caps[zone] else [])
    return best, shares, incentive, tuple(sorted(int(open_zones[i]) for i in full)), vertices


def _parameter(name: str, number: float, *, positive: bool) -> float:
    read = np.asarray(_real_numbers(number, name), dtype=np.float64)
    if read.ndim or not np.isfinite(read) or (read <= 0.0 if positive else read < 0.0):
        bound = "positive" if positive else "non-negative"
        raise ValueError(f"{name} is {number!r}; it is one finite number, {bound}")
    return float(read)


def exact_congestion_allocation(
    value: ArrayLike,
    demand: ArrayLike,
    beta: float,
    congestion: float,
    mass: float,
    budget: float,
) -> CongestionAllocation:
    """The split of a lump-sum incentive budget that completes the most rides through a logit
    congestion equilibrium, found exactly, with the vertex that certifies it.

    The game is :class:`chc.marketplace.SharedStateMarket`'s. ``mass`` drivers settle at the
    shares ``s = x / mass = softmax(beta * (value + u - congestion * s))``, the incentive
    ``u >= 0`` with ``sum(u) = budget`` adds to each zone's value, and zone ``i`` completes
    ``min(demand_i, mass * s_i)`` rides. ``value`` is a zone's value to a driver before crowding
    and incentive: the market's ``attract``, a fitted equilibrium's ``attract * demand``.

    - ``s`` is the equilibrium under ``u`` exactly where ``phi_i(s_i) = log(s_i)/beta +
      congestion*s_i - value_i`` equals ``u_i`` plus one constant shared by every zone (Berry
      1994's inversion). So the least budget that reaches ``s`` pays each zone
      ``phi_i(s_i) - min_j phi_j(s_j)``, and what is left of the budget, spread evenly, moves no
      driver.
    - The zones paid nothing share one level ``mu`` and sit at their floor shares ``g_i(mu)``,
      ``W(beta*congestion*exp(beta*(mu + value_i)))/(beta*congestion)`` with ``W`` Lambert's,
      increasing and convex in ``mu``. An allocation whose least level is ``mu`` holds every zone
      at its floor or above, so it completes at most ``U(mu) = mass - sum_i (mass*g_i(mu) -
      demand_i)+``, which falls as ``mu`` rises.
    - ``U(mu)`` is reached where the deficit ``1 - sum(g(mu))`` can be filled, with no zone past
      its demand, for at most the budget. ``phi`` is concave, so the least such cost is a concave
      minimisation over a box cut by a hyperplane, and lies at a vertex (Rockafellar 1970,
      §32): every funded zone at its demand but at most one. That cost falls as
      ``mu`` rises, so bisection finds the least level the budget reaches.
    - That level's ``U`` is the global maximum: given an allocation with a funded zone past its
      demand, raise the level, holding the other funded zones and letting that zone give up the
      drivers the floors take. The bill falls, at ``-#funded - phi_k' * sum(g')``, and the drivers
      wasted do not grow, until no funded zone is past its demand: a vertex of a level the budget
      reaches.

    Global separable concave minimisation is NP-hard (Moré and Vavasis 1991), and the enumeration
    is the price of the certificate: a level reads ``k * 2**(k - 1)`` vertices over the ``k``
    zones whose floor lies below their demand, at most 24,576 on the 12 zones of
    ``SharedStateMarket``, and a solve at most ``2*j + 51`` levels, where the search below the free
    level doubles its reach ``j`` times: 49 on each of the markets ``SharedStateMarket(seed=0..29)``
    at either precision, 0.3 to 1.2 million vertices in all. Past 20 zones it refuses. Exact where
    the objective is completed rides and the budget a lump sum paid into the drivers' value;
    ``validation/exact_congestion_allocation.mac`` checks the inversion, the floor, the bound and
    a vertex's cost at a rational point. In float64 throughout, so the optimum does not move with
    the caller's precision; an equilibrium solved to a tolerance reproduces ``value`` to that
    tolerance.

    Args:
        value: ``(n,)`` each zone's value to a driver before crowding and incentive, finite.
        demand: ``(n,)`` the rides each zone can complete, finite and non-negative.
        beta: the drivers' sensitivity to value, positive.
        congestion: the crowding penalty, non-negative.
        mass: the drivers, positive.
        budget: the incentive to split, non-negative.

    Raises:
        ValueError: on a ``value`` or ``demand`` that is not a finite vector of numbers of one
            length, a negative demand, parameters outside their ranges, or more than 20 zones.
    """
    read = {
        name: np.asarray(_real_numbers(column, name), dtype=np.float64)
        for name, column in (("value", value), ("demand", demand))
    }
    worth, wanted = read["value"], read["demand"]
    if worth.ndim != 1 or not worth.size or wanted.shape != worth.shape:
        raise ValueError(
            f"value has shape {worth.shape} and demand {wanted.shape}: both are one vector over the"
            " zones"
        )
    for name, column in read.items():
        if not np.all(np.isfinite(column)):
            raise ValueError(f"{name} is not finite")
    if np.any(wanted < 0.0):
        raise ValueError(f"demand reaches {float(np.min(wanted))}; a zone's demand is non-negative")
    beta = _parameter("beta", beta, positive=True)
    congestion = _parameter("congestion", congestion, positive=False)
    mass = _parameter("mass", mass, positive=True)
    budget = _parameter("budget", budget, positive=False)
    n = worth.size
    if n > _EXACT_ZONES:
        raise ValueError(
            f"{n} zones: a level reads up to n * 2**(n - 1) = {n << (n - 1)} vertices, and the"
            f" exact allocation enumerates at most {_EXACT_ZONES} zones"
        )
    demand_share = wanted / mass
    scale = 1.0 / beta + congestion + float(np.max(np.abs(worth))) + budget

    def resolved(low: float, high: float) -> bool:
        return high - low <= 4.0 * _EPS * max(abs(low), abs(high), scale)

    # the free level, with no incentive: the floor shares sum to 1. At the low end they sum to at
    # most sum(exp(beta*(level + value))) = 1, at the high end the most valued zone's alone is 1
    low = (
        -(float(np.max(beta * worth)) + math.log(math.fsum(np.exp(beta * (worth - worth.max())))))
        / beta
    )
    high = congestion - float(np.max(worth))
    while not resolved(low, high):
        middle = 0.5 * (low + high)
        floor = np.exp(_floor_log_shares(middle, worth, beta, congestion))
        low, high = (low, middle) if math.fsum(floor) >= 1.0 else (middle, high)
    free = high

    vertices = 0

    def cheapest(level: float) -> tuple[Any, ...]:
        nonlocal vertices
        found = _cheapest_vertex(level, worth, demand_share, beta, congestion)
        vertices += found[-1]
        return found

    # a level the budget does not reach, below the free one: the cost grows without bound below
    reach = 1.0 / beta + congestion
    low, below = free - reach, cheapest(free - reach)
    while not below[0] > budget:
        reach *= 2.0
        low, below = free - reach, cheapest(free - reach)
    high, at = free, None
    while not resolved(low, high):
        middle = 0.5 * (low + high)
        found = cheapest(middle)
        if found[0] <= budget:
            high, at = middle, found
        else:
            low, below = middle, found
    if at is None:  # no level below the free one is within the budget: no incentive moves a driver
        floor = np.exp(_floor_log_shares(free, worth, beta, congestion))
        at = (0.0, floor / math.fsum(floor), np.zeros(n), (), 0)
    spent, shares, incentive, full, _ = at
    funded = np.flatnonzero(incentive > 0.0)
    partial = [int(i) for i in funded if int(i) not in full]
    return CongestionAllocation(
        allocation=incentive + (budget - spent) / n,
        value=math.fsum(np.minimum(wanted, mass * shares)),
        level=high,
        shares=shares,
        at_demand=full,
        partial=partial[0] if partial else None,
        spent=spent,
        level_below=low,
        cost_below=below[0],
        vertices=vertices,
    )
