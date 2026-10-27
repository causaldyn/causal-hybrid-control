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
    *,
    start: Array | None = None,
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

    ``start``, where given, replaces the even split: the ascent starts at its projection onto the
    budget simplex.
    """
    if start is not None and jnp.shape(start) != (n,):
        raise ValueError(f"start has shape {jnp.shape(start)}; the allocation has {n} entries")
    u = jnp.full(n, budget / n) if start is None else project_simplex(jnp.asarray(start), budget)
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
    ``shadow_price`` is what one more unit of budget completes there, the budget's Lagrange
    multiplier: on ``SharedStateMarket(seed=0..7)`` at budgets from 0.5 to 60 it equals central
    differences of ``value`` in the budget to 1e-8, and it is 0 where more budget completes
    nothing.
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
    # the rides one more unit of budget completes, dV*/dB: 0 where more budget completes nothing,
    # nan at a vertex with no partial zone where drivers are wasted (:func:`_budget_shadow_price`)
    shadow_price: float = math.nan


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


def _budget_shadow_price(
    shares: NDArray[np.float64],
    demand: NDArray[np.float64],
    at_demand: tuple[int, ...],
    partial: int | None,
    cost_below: float,
    beta: float,
    congestion: float,
    mass: float,
) -> float:
    """The optimum's rides per unit of budget,
    ``dV*/dB = mass*beta*q_p*Q_O / ((|T| + 1)*q_p + Q_R)``, with ``q = s / (1 + beta*congestion*s)``
    at the optimum's shares, ``T`` the zones held at their demand, ``p`` the partial zone, ``Q_R``
    the sum of ``q`` over the other zones, at their floor, and ``Q_O`` over those of them past their
    demand.

    One more unit of budget lowers the level ``mu`` the unfunded zones share by ``1/(|T| + 1 +
    Q_R/q_p)``: each zone at demand and the partial zone pays one unit per unit of level, and the
    partial zone takes the drivers the floors give up, ``beta*q_j`` per unit of level each, at
    ``1/(beta*q_p)`` per driver share. The rides gained are the over-supplied floors' waste, which
    falls at ``mass*beta*Q_O`` per unit of level. 0 where more budget completes nothing: no zone
    at its floor is past its demand, so no driver is wasted, or the zones below their demand cannot
    hold the deficit at any lower level (``cost_below`` is inf). nan where drivers are wasted and
    the optimum has no partial zone, a vertex at which the rides have no derivative in the budget,
    as at a budget of zero. ``validation/marketplace_shadow_price.mac`` derives it.
    """
    if not math.isfinite(cost_below):
        return 0.0
    rest = np.ones(shares.size, dtype=bool)
    rest[list(at_demand)] = False
    if partial is not None:
        rest[partial] = False
    over = rest & (mass * shares > demand)
    if not over.any():
        return 0.0
    if partial is None:
        return math.nan
    q = shares / (1.0 + beta * congestion * shares)
    return (
        mass
        * beta
        * q[partial]
        * math.fsum(q[over])
        / ((len(at_demand) + 1) * q[partial] + math.fsum(q[rest]))
    )


def _parameter(name: str, number: float, *, positive: bool) -> float:
    read = np.asarray(_real_numbers(number, name), dtype=np.float64)
    if read.ndim or not np.isfinite(read) or (read <= 0.0 if positive else read < 0.0):
        bound = "positive" if positive else "non-negative"
        raise ValueError(f"{name} is {number!r}; it is one finite number, {bound}")
    return float(read)


def _read_game(
    value: ArrayLike,
    demand: ArrayLike,
    beta: float,
    congestion: float,
    mass: float,
    budget: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float, float, float]:
    """A congestion game's zones and parameters read in float64, refused outside their ranges."""
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
    return (
        worth,
        wanted,
        _parameter("beta", beta, positive=True),
        _parameter("congestion", congestion, positive=False),
        _parameter("mass", mass, positive=True),
        _parameter("budget", budget, positive=False),
    )


def _resolved(low: float, high: float, scale: float) -> bool:
    return high - low <= 4.0 * _EPS * max(abs(low), abs(high), scale)


def _free_level(worth: NDArray[np.float64], beta: float, congestion: float, scale: float) -> float:
    """The level of the equilibrium with no incentive, where the floor shares sum to 1. At the low
    end they sum to at most ``sum(exp(beta*(level + value))) = 1``, at the high end the most valued
    zone's alone is 1."""
    low = (
        -(float(np.max(beta * worth)) + math.log(math.fsum(np.exp(beta * (worth - worth.max())))))
        / beta
    )
    high = congestion - float(np.max(worth))
    while not _resolved(low, high, scale):
        middle = 0.5 * (low + high)
        floor = np.exp(_floor_log_shares(middle, worth, beta, congestion))
        low, high = (low, middle) if math.fsum(floor) >= 1.0 else (middle, high)
    return high


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
    worth, wanted, beta, congestion, mass, budget = _read_game(
        value, demand, beta, congestion, mass, budget
    )
    n = worth.size
    if n > _EXACT_ZONES:
        raise ValueError(
            f"{n} zones: a level reads up to n * 2**(n - 1) = {n << (n - 1)} vertices, and the"
            f" exact allocation enumerates at most {_EXACT_ZONES} zones"
        )
    demand_share = wanted / mass
    scale = 1.0 / beta + congestion + float(np.max(np.abs(worth))) + budget

    def resolved(low: float, high: float) -> bool:
        return _resolved(low, high, scale)

    free = _free_level(worth, beta, congestion, scale)

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
    middle_zone = partial[0] if partial else None
    return CongestionAllocation(
        allocation=incentive + (budget - spent) / n,
        value=math.fsum(np.minimum(wanted, mass * shares)),
        level=high,
        shares=shares,
        at_demand=full,
        partial=middle_zone,
        spent=spent,
        level_below=low,
        cost_below=below[0],
        vertices=vertices,
        shadow_price=_budget_shadow_price(
            shares, wanted, full, middle_zone, below[0], beta, congestion, mass
        ),
    )


@dataclass(frozen=True)
class PerDriverAllocation:
    """The optimum :func:`per_driver_congestion_allocation` returns, and the bound certifying it.

    Each driver in zone ``i`` is paid ``allocation[i]``, so the money paid is
    ``mass * sum(shares * allocation)``, ``spent``, at most the budget to the solver's tolerance.
    The allocation completes ``value`` rides; no allocation whose money paid is within the budget
    completes more than ``bound``, the Lagrangian dual's value at the multipliers read off the
    optimum, so ``gap`` bounds the rides the allocation leaves undone. ``shadow_price`` is the
    budget's multiplier, what one more unit of money completes there.
    """

    allocation: NDArray[np.float64]  # (n,) the incentive per driver in each zone, the least zero
    value: float  # the rides completed at the allocation's equilibrium
    level: float  # log(s)/beta + congestion*s - value of the zones paid nothing
    shares: NDArray[np.float64]  # (n,) the drivers' shares at the allocation's equilibrium
    spent: float  # the money paid, mass * sum(shares * allocation)
    bound: float  # the dual's value: no allocation within the budget completes more
    gap: float  # bound - value, to rounding at least zero
    shadow_price: float  # the budget's multiplier: the rides one more unit of money completes


_ACTIVE = 1e-7
"""The relative distance within which :func:`per_driver_congestion_allocation` reads a zone as at
its floor or at its demand when it reads the multipliers off the optimum. A misread costs only
bound: the dual's value at any multipliers bounds the optimum."""

_LOG_SHARE_FLOOR = -700.0
"""The least log share the dual's per-zone supremum searches, short of the float64 underflow."""


def _per_driver_dual(
    price: float,
    floor_prices: NDArray[np.float64],
    shift: float,
    worth: NDArray[np.float64],
    wanted: NDArray[np.float64],
    beta: float,
    congestion: float,
    mass: float,
    budget: float,
) -> float:
    """The Lagrangian dual of the per-driver program at ``(lambda, mu, nu)`` = ``(price,
    floor_prices, shift)``, ``mu >= 0`` with ``sum(mu) = lambda * mass``:
    ``lambda*B + nu + sum_i sup_{0 < s <= 1} h_i(s)``,
    ``h_i(s) = min(d_i, m*s) - lambda*m*s*phi_i(s) + mu_i*phi_i(s) - nu*s``. Each ``h_i`` is
    concave, so its supremum is where its left derivative changes sign, found by bisection on the
    log share."""

    def phi(share: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.log(share) / beta + congestion * share - worth

    def rising(log_share: NDArray[np.float64]) -> NDArray[np.bool_]:
        share = np.exp(log_share)
        slope = 1.0 / (beta * share) + congestion
        left = mass * (mass * share < wanted) - price * mass * (phi(share) + share * slope)
        left = left + np.where(floor_prices > 0.0, floor_prices * slope, 0.0) - shift
        return left > 0.0

    low = np.full(worth.size, _LOG_SHARE_FLOOR)
    high = np.zeros(worth.size)
    whole = rising(high)
    for _ in range(128):
        middle = 0.5 * (low + high)
        up = rising(middle)
        low, high = np.where(up, middle, low), np.where(up, high, middle)
    share = np.where(whole, 1.0, np.exp(high))
    level = phi(share)
    best = (
        np.minimum(wanted, mass * share)
        - price * mass * share * level
        + np.where(floor_prices > 0.0, floor_prices * level, 0.0)
        - shift * share
    )
    return price * budget + shift + math.fsum(best)


def _per_driver_multipliers(
    log_shares: NDArray[np.float64],
    levels: NDArray[np.float64],
    allocation: NDArray[np.float64],
    wanted: NDArray[np.float64],
    beta: float,
    congestion: float,
    mass: float,
    slack: bool,
) -> tuple[float, NDArray[np.float64], float]:
    """The multipliers ``(lambda, mu, nu)`` of the per-driver program, read off its optimum.

    The KKT conditions: for every zone ``m*w_i - lambda*m*psi_i + mu_i*phi_i' - nu = 0``, with
    ``psi_i = phi_i + s_i*phi_i'`` the slope of ``s*phi(s)``, ``w_i`` the rides' supergradient (1
    below demand, 0 past it, anywhere in [0, 1] at it), ``mu_i = 0`` off the floor, and
    ``sum(mu) = lambda*m``. Each zone off both its floor and its demand gives one equation in
    ``(lambda, nu)``, and the floor zones' ``mu_i``, solved from theirs, give one more through
    their sum. Where those leave a line, as with no budget, the point on it nearest the least
    squares solution that keeps ``lambda``, every ``mu_i`` and every ``w_i`` at a zone's demand in
    range is taken. Where the budget is ``slack``, ``lambda`` is 0 (complementary slackness) and
    ``nu`` the least squares solution of the rest. ``1/phi_i' = beta*s_i/(1 + beta*congestion*s_i)``
    is read from the log share, so a share below the smallest float reads 0, not a division by
    zero."""
    shares = np.exp(log_shares)
    level = float(np.min(levels))
    inverse_slope = beta * shares / (1.0 + beta * congestion * shares)
    tilt = levels + 1.0 / beta + congestion * shares
    floor = allocation <= _ACTIVE * max(1.0, abs(level))
    kink = np.abs(mass * shares - wanted) <= _ACTIVE * np.maximum(1.0, wanted)
    below = (mass * shares < wanted).astype(np.float64)
    loose = ~floor & ~kink
    held = floor & ~kink
    by_zone = np.stack([mass * tilt, np.ones(shares.size)], axis=1)
    rows = np.concatenate(
        [by_zone[loose], (by_zone[held] * inverse_slope[held, None]).sum(axis=0, keepdims=True)]
    )
    rows[-1, 0] -= mass
    targets = np.concatenate(
        [mass * below[loose], [float(np.sum(mass * below[held] * inverse_slope[held]))]]
    )
    solution, _, rank, _ = np.linalg.lstsq(rows, targets, rcond=None)
    if slack:
        shift, *_ = np.linalg.lstsq(rows[:, 1:], targets, rcond=None)
        solution = np.array([0.0, float(shift[0])])
    elif rank < 2:
        # every point solution + z*null solves the equations; the inequalities a.x >= b bound z
        null = np.linalg.svd(rows)[2][-1]
        rules = [(np.array([1.0, 0.0]), 0.0)]
        rules += [(by_zone[i], mass * below[i]) for i in np.flatnonzero(held)]
        at_demand = np.flatnonzero(kink & ~floor)
        rules += [(by_zone[i], 0.0) for i in at_demand]
        rules += [(-by_zone[i], -mass) for i in at_demand]
        low, high = -math.inf, math.inf
        for a, b in rules:
            rate, room = float(a @ null), float(a @ solution) - b
            if rate > 0.0:
                low = max(low, -room / rate)
            elif rate < 0.0:
                high = min(high, -room / rate)
        if low <= high:
            solution = solution + min(max(0.0, low), high) * null
    price, shift = (float(part) for part in solution)
    floor_prices = np.where(held, (shift + price * mass * tilt - mass * below) * inverse_slope, 0.0)
    floor_prices = np.maximum(floor_prices, 0.0)
    return math.fsum(floor_prices) / mass, floor_prices, shift


def _water_fill(
    level: float,
    worth: NDArray[np.float64],
    demand_share: NDArray[np.float64],
    beta: float,
    congestion: float,
    scale: float,
) -> tuple[float, NDArray[np.float64], NDArray[np.bool_]]:
    """The least money that holds every zone at or above its floor share at ``level`` and fills the
    deficit with no zone past its demand, the log shares that do it, and the zones it lifts off
    their floor, the zones it pays; inf where the zones below their demand cannot hold the deficit.
    Where the floors leave no deficit, the level is the free one, and the shares are the floors
    normalised, paid nothing.

    A zone at share ``s`` costs ``mass*s*(phi(s) - level)``, convex and rising from its floor, at
    the margin ``mass*(psi(s) - level)``, ``psi(s) = log(s)/beta + 2*congestion*s - value + 1/beta``
    (the slope of ``s*phi(s)``). The least cost fills to one margin ``theta``: each zone with room
    takes ``psi^-1(theta)``, held between its floor and its demand. ``psi = theta`` is the floor's
    equation at twice the congestion, so ``psi^-1`` is :func:`_floor_log_shares`', and ``theta`` is
    found by bisection on the filled sum. The returned cost is in units of ``1/mass``."""
    floor = _floor_log_shares(level, worth, beta, congestion)
    shares = np.exp(floor)
    deficit = 1.0 - math.fsum(shares)
    unpaid = np.zeros(worth.size, dtype=bool)
    if deficit <= 0.0:
        return 0.0, floor - math.log(math.fsum(shares)), unpaid
    with np.errstate(divide="ignore"):
        cap = np.log(demand_share)
    room = cap > floor
    if not room.any() or math.fsum(np.exp(cap[room]) - shares[room]) < deficit:
        return math.inf, floor, unpaid
    value = worth[room]

    def filled(margin: float) -> NDArray[np.float64]:
        lifted = _floor_log_shares(margin - 1.0 / beta, value, beta, 2.0 * congestion)
        return np.clip(lifted, floor[room], cap[room])

    def margin_of(log_share: NDArray[np.float64]) -> NDArray[np.float64]:
        share = np.exp(log_share)
        return log_share / beta + 2.0 * congestion * share - value + 1.0 / beta

    low, high = float(np.min(margin_of(floor[room]))), float(np.max(margin_of(cap[room])))
    while not _resolved(low, high, scale):
        middle = 0.5 * (low + high)
        gained = math.fsum(np.exp(filled(middle)) - shares[room])
        low, high = (middle, high) if gained < deficit else (low, middle)
    log_shares = floor.copy()
    log_shares[room] = filled(high)
    paid = log_shares > floor
    share = np.exp(log_shares[paid])
    worth_paid = worth[paid]
    cost = math.fsum(share * (log_shares[paid] / beta + congestion * share - worth_paid - level))
    return cost, log_shares, paid


def per_driver_congestion_allocation(
    value: ArrayLike,
    demand: ArrayLike,
    beta: float,
    congestion: float,
    mass: float,
    budget: float,
) -> PerDriverAllocation:
    """The incentive per driver that completes the most rides through a logit congestion
    equilibrium when the money paid, not the sum of the incentives, is the budget, with the dual
    bound that certifies it.

    The game is :func:`exact_congestion_allocation`'s: ``mass`` drivers settle at the shares
    ``s = softmax(beta * (value + u - congestion * s))`` and zone ``i`` completes
    ``min(demand_i, mass * s_i)``. Here each driver in zone ``i`` is paid ``u_i >= 0``, so the money
    paid is ``sum_i u_i * mass * s_i`` (Ong, Freund and Crapis 2021, Lyft's repositioning
    incentives). By Berry's (1994) inversion the cheapest ``u`` reaching ``s`` pays
    ``phi_i(s_i) - min_j phi_j(s_j)``, ``phi_i(s) = log(s)/beta + congestion*s - value_i``, so the
    money that reaches ``s`` is ``mass * (sum_i s_i*phi_i(s_i) - min_j phi_j(s_j))``. ``s*phi(s)``
    is convex, its second derivative ``(2*beta*congestion*s + 1)/(beta*s)``, and ``phi`` is
    concave, so ``-min phi`` is convex: the shares the budget reaches are a convex set, and the
    rides, concave in the shares, are maximised by a convex program. Paid as a lump sum,
    ``sum(u) = budget``, the reachable shares are not convex, and
    :func:`exact_congestion_allocation` enumerates vertices instead.

    The program is solved through its KKT conditions. The zones paid nothing share one level ``t``
    at their floor shares, as in :func:`exact_congestion_allocation`; the paid zones below their
    demand share one margin ``psi_i(s_i) = phi_i(s_i) + s_i*phi_i'(s_i)``, the slope of their
    cost, since a zone's cost ``mass*s*(phi(s) - t)`` is convex: the deficit the floors leave is
    filled to one margin, the least money that fills it, with no zone past its demand. That least
    money falls as ``t`` rises, and the rides a level allows,
    ``mass - sum_i (mass*g_i(t) - demand_i)+``, fall too, so bisection finds the least level the
    budget reaches, as for the lump sum. Where the drivers outnumber the demand, the search stops
    at the level that fills every zone to its demand. The multipliers are read off the optimum
    and the Lagrangian dual is evaluated there, zone by zone, each zone's term a one-dimensional
    concave supremum: ``bound`` is at least the rides of any allocation within the budget (weak
    duality), whatever the multipliers, and ``gap`` is the certificate. On the markets
    ``SharedStateMarket(seed=0..5)`` at budgets from 0 to 20 it reads below 1e-12.
    ``validation/per_driver_allocation.mac`` checks the convexity, the cost of the shares, the
    margin's inverse, the dual's per-zone supremum and the KKT reading. In float64 throughout.

    Args:
        value: ``(n,)`` each zone's value to a driver before crowding and incentive, finite.
        demand: ``(n,)`` the rides each zone can complete, finite and non-negative.
        beta: the drivers' sensitivity to value, positive.
        congestion: the crowding penalty, non-negative.
        mass: the drivers, positive.
        budget: the money to pay, non-negative.

    Raises:
        ValueError: on a ``value`` or ``demand`` that is not a finite vector of numbers of one
            length, a negative demand, or parameters outside their ranges.
    """
    worth, wanted, beta, congestion, mass, budget = _read_game(
        value, demand, beta, congestion, mass, budget
    )
    demand_share = wanted / mass
    scale = 1.0 / beta + congestion + float(np.max(np.abs(worth))) + budget / mass
    free = _free_level(worth, beta, congestion, scale)
    money = budget / mass

    def cost(level: float) -> tuple[float, NDArray[np.float64], NDArray[np.bool_]]:
        return _water_fill(level, worth, demand_share, beta, congestion, scale)

    # below the level at which the zones' floors and demands together hold every driver, the
    # deficit no longer fits under the demands; there every zone is filled to its demand
    def held(level: float) -> float:
        floor = np.exp(_floor_log_shares(level, worth, beta, congestion))
        return math.fsum(np.maximum(floor, demand_share))

    reach = 1.0 / beta + congestion
    low = free - reach
    while held(low) > 1.0 and not cost(low)[0] > money:
        reach *= 2.0
        low = free - reach
    high = free
    if held(low) <= 1.0:  # the drivers outnumber the demand: the least level that fills it
        top = free
        while not _resolved(low, top, scale):
            middle = 0.5 * (low + top)
            low, top = (middle, top) if held(middle) <= 1.0 else (low, middle)
        low = top
        if cost(low)[0] <= money:
            high = low
    while not _resolved(low, high, scale):
        middle = 0.5 * (low + high)
        low, high = (low, middle) if cost(middle)[0] <= money else (middle, high)
    # the rides stay at the mass up to the first level at which a floor reaches its demand: the
    # cheapest of those optima pays no more than that level costs
    if np.all(wanted > 0.0):
        flat = min(
            float(np.min(np.log(demand_share) / beta + congestion * demand_share - worth)), free
        )
        high = max(high, flat)
    _, log_shares, paid = cost(high)
    shares = np.exp(log_shares)
    levels = log_shares / beta + congestion * shares - worth
    allocation = np.where(paid, np.maximum(levels - high, 0.0), 0.0)
    completed = math.fsum(np.minimum(wanted, mass * shares))
    spent = mass * math.fsum(shares * allocation)
    price, floor_prices, shift = _per_driver_multipliers(
        log_shares,
        levels,
        allocation,
        wanted,
        beta,
        congestion,
        mass,
        spent < budget - _ACTIVE * max(1.0, budget),
    )
    bound = _per_driver_dual(
        price, floor_prices, shift, worth, wanted, beta, congestion, mass, budget
    )
    return PerDriverAllocation(
        allocation=allocation,
        value=completed,
        level=high,
        shares=shares,
        spent=spent,
        bound=bound,
        gap=bound - completed,
        shadow_price=price,
    )
