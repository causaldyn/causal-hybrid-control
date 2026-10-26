"""Offline causal decision-making under equilibrium interference.

The problem, end to end: learn an incentive policy from *confounded,
switchback-logged* marketplace data where (a) the logging policy is confounded (operators already
incentivise busy zones, so demand drives both treatment and outcome), and (b) SUTVA fails because
drivers are mobile -- incentivising a zone pulls drivers from its neighbours through a shared
equilibrium (Munro-Wager-Xu market-equilibrium interference; Wager-Xu supply cannibalisation;
shared-state DML, arXiv 2504.08836). Three failure modes, each a real baseline:

* **predictive / MOPO** fits an outcome model on the confounded logs -- confounding inflates the
  response of already-busy zones, so it over-allocates to them (where drivers only cannibalise).
* **naive causal** adjusts each zone for its own demand (a backdoor) but assumes SUTVA -- it
  allocates by local uplift, double-counting drivers the market cannot supply (predicts a lift it
  never gets). The adjustment removes part of the confounding, not all: the logging policy splits
  one budget by every zone's demand, so a zone's incentive also follows the others' demand, which
  draws drivers from it too. Against the logs rerun with incentives that do not follow demand, it
  removed 28-91 % of the predictive response's excess on 30 markets (median 63 %).
* **CHC** adjusts each zone for its own demand and the shared exposure *and* plans through the
  structural equilibrium (``chc.games``) with an offline-pessimism margin (a Wasserstein/DRO-style
  shrink by the estimate's influence-function SE), so cannibalisation is built in and
  extrapolation is not trusted.

What the comparison measures: :func:`pessimistic_equilibrium_allocation` plans through the
equilibrium :func:`fit_equilibrium` estimates from the logs, passed as ``equilibrium=`` (ADR 0114),
never through the simulator's own base demand, attractiveness and equilibrium map, which no log
supplies: through those, a response of zero planned as close to the oracle as the adjusted
one, so the comparison measured knowing the equilibrium. The SUTVA baselines get no such
structure. The plan's value is the fitted equilibrium's: on 30 markets the adjusted
response's tilt moved the share of the best plan's lift by -0.022 +- 0.012 on 400 blocks against
a response of zero (mean +- standard error; -0.17 to +0.12), and by -0.057 +- 0.025 on 50 (-0.41
to +0.21), the best plan being the exact oracle's (:meth:`SharedStateMarket.oracle_allocation`).
Where the market lies outside the fitted class, its attractiveness fixed while the demand moves,
the tilt cost share too: 0.73 of the lift against 0.79 with a response of zero, over 100 logs of
400 blocks (ADR 0134; through a fit that stopped early it had read 0.72 against 0.40). The fit
warns on such a market, and the planner refuses a fit whose drivers do not tell an incentive from
none (ADR 0123).

Everything is differentiable and reuses the equilibrium solver in :mod:`chc.games`.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray
from scipy import special, stats
from scipy.optimize import least_squares, lsq_linear, minimize

from chc import _units
from chc.frames import _real_entries
from chc.games import (
    exact_congestion_allocation,
    fixed_point,
    softmax_congestion_equilibrium,
    stackelberg_allocation,
)


@dataclass(frozen=True)
class SharedStateMarket:
    """Mobile drivers reach a congestion equilibrium; the platform allocates a fixed budget.

    The shared state is the equilibrium driver distribution ``x(u)`` (``sum x = mass`` is conserved,
    so an incentive to one zone is a driver *taken from* the others -- the interference channel).
    ``attract`` is aligned with ``demand`` so the zones a naive allocator wants to feed (high local
    uplift) are the already-saturated ones, where extra drivers only cannibalise.
    """

    n_zones: int = 12
    mass: float = 12.0  # total mobile drivers
    congestion: float = 2.0  # crowding penalty (more drivers per zone -> lower per-driver value)
    beta: float = 2.5  # driver price-sensitivity
    budget: float = 6.0
    confounding: float = 2.0  # how strongly the logging policy chases demand (the confounder)
    seed: int = 0

    def _base(self) -> tuple[Array, Array]:
        """Reference (planning-time) demand and attractiveness."""
        k = jax.random.split(jax.random.key(self.seed), 2)
        demand = 0.4 + 1.6 * jax.random.uniform(k[0], (self.n_zones,))
        attract = 1.5 * demand + 0.3 * jax.random.normal(k[1], (self.n_zones,))
        return demand, attract

    def _equilibrium(self, attract: Array, u: Array) -> Array:
        """Solved driver distribution. ``beta*congestion = 5 < 6`` here, so the map is certified
        contracting (:func:`chc.research.games.congestion_contraction_certificate`) and the solve
        converges.
        """
        return softmax_congestion_equilibrium(attract, u, self.congestion, self.mass, self.beta).x

    def completions(self, u: Array) -> Array:
        """Total realised rides = sum_i min(demand_i, drivers_i) at the equilibrium."""
        demand, attract = self._base()
        return jnp.sum(jnp.minimum(demand, self._equilibrium(attract, u)))

    def value(self, u: Array) -> float:
        return float(self.completions(u))

    def oracle_allocation(self, steps: int | None = None) -> Array:
        """The allocation that completes the most rides at the market's equilibrium, with full
        model access: the regret reference, exact (:func:`chc.games.exact_congestion_allocation`).

        Its value moves with the machine only through the equilibrium solve, to about its
        tolerance of 1e-6. The ascent it ran before, a local maximum at best, stays
        :func:`chc.games.stackelberg_allocation`; ``steps``, its length, is ignored, and leaves in
        0.23.0.

        Warns:
            DeprecationWarning: where ``steps`` is given.
        """
        if steps is not None:
            warnings.warn(
                "SharedStateMarket.oracle_allocation(steps=...) is ignored and leaves in 0.23.0:"
                " the oracle is the exact optimum, chc.games.exact_congestion_allocation. The"
                " ascent it ran is chc.games.stackelberg_allocation(market.completions,"
                " market.n_zones, market.budget, steps=steps)",
                DeprecationWarning,
                stacklevel=2,
            )
        demand, attract = self._base()
        exact = exact_congestion_allocation(
            attract, demand, self.beta, self.congestion, self.mass, self.budget
        )
        return jnp.asarray(exact.allocation, dtype=demand.dtype)

    def generate_logs(self, n_blocks: int, key: Array) -> dict[str, Array]:
        """Confounded switchback logs, shape ``(n_blocks, n_zones)``: each block draws a demand
        season and a budget scale (switchback), assigns a demand-chasing (confounded) allocation,
        and records the shared-state exposure aggregate and realised completions.
        """
        base_demand, base_attract = self._base()

        def block(carry: None, k: Array) -> tuple[None, dict[str, Array]]:
            k_season, k_scale, k_noise = jax.random.split(k, 3)
            season = 0.5 + jax.random.uniform(k_season, (self.n_zones,))  # within-zone demand shift
            demand = base_demand * season
            attract = base_attract * season
            scale = 0.4 + 1.2 * jax.random.uniform(k_scale)  # switchback block-level budget scale
            u = scale * self.budget * jax.nn.softmax(self.confounding * demand)  # chases demand
            x = self._equilibrium(attract, u)
            y = jnp.minimum(demand, x) + 0.02 * jax.random.normal(k_noise, (self.n_zones,))
            agg = jnp.full(self.n_zones, jnp.sum(u))  # shared-state exposure (total incentive)
            return None, {"u": u, "demand": demand, "aggregate": agg, "y": y}

        _, logs = jax.lax.scan(block, None, jax.random.split(key, n_blocks))
        return logs  # each (n_blocks, n_zones)


@dataclass(frozen=True)
class ExposureResponse:
    """A calibrated per-zone incentive response ``dY/du`` plus its influence-function SE.

    ``se`` is each zone's HC0 sandwich, White's heteroskedasticity-robust standard error with no
    small-sample correction: right as the log grows, short in a short one. On
    ``SharedStateMarket(seed=0)`` read by :func:`calibrate_naive_causal`, ``marginal +- 1.96 se``
    held the slope of three logs of 40,000 blocks in 91.3% of the twelve zones' reads over 300 logs
    of 400 blocks (MC SE about 0.5%), 79% in the worst zone, whose slope spread 1.18 times its mean
    ``se``; about the reads' own mean, 91.8%. HC3's leverage correction raised that to only 92.1%.
    Over 150 logs of 4,000 blocks it held in 94.2%, and in 95.0% over another 150. A pessimism
    ``radius`` in units of ``se`` (:func:`sutva_allocation`,
    :func:`pessimistic_equilibrium_allocation`) discounts less than its normal quantile says in
    logs that short.
    """

    marginal: Array  # (n_zones,) estimated marginal completion response to an incentive
    se: Array  # (n_zones,) HC0 heteroskedasticity-robust SE of ``marginal``


def _zone_slope(u: Array, y: Array, extra: Array) -> tuple[Array, Array]:
    """Per-zone OLS slope of ``y`` on ``u`` controlling ``extra`` (blocks x k), + robust SE. Returns
    the incentive slope and its sandwich SE.

    The intercept is free, and the ridge on each other coefficient is scaled by its column's
    variance, so that the slope reads the same in any units and from any origin of the incentive
    and the covariates. Added as a constant, the ridge outweighed an incentive logged in millionths
    of its units and set every zone's slope to zero, and a zone logged at one incentive throughout
    read a response no log can identify. The slopes and their sandwich are solved about the
    columns' means, where the intercept drops out of both: in the Gram of the raw columns a
    column's spread drowns in the rounding of its offset, and an incentive constant but for 1e-11
    of its size read a slope of zero. A column whose spread is rounding (:mod:`chc._units`) is
    zeroed about its mean, so an incentive logged at one level reads a slope and an SE of exactly
    zero, and its ridge scale is 1, which keeps the solve regular and moves nothing else. The
    deviations' own mean is taken off them too: in float32, JAX's default, a mean is rounded to
    about 1e-7 of its size, which an incentive near constant carries into the residuals through its
    large slope; without it, one whose spread is about 100 epsilons of its size read an SE 1.2e-3
    off.
    """
    columns = _units.centred(jnp.concatenate([u[:, None], extra], axis=1))
    centred = columns.deviations
    outcome = y - jnp.mean(y)
    gram_inv = jnp.linalg.inv(centred.T @ centred + jnp.diag(1e-4 * columns.scales))
    coef = gram_inv @ centred.T @ outcome
    resid = outcome - centred @ coef
    meat = (centred * resid[:, None]).T @ (centred * resid[:, None])
    cov = gram_inv @ meat @ gram_inv
    return coef[0], jnp.sqrt(jnp.maximum(cov[0, 0], 0.0))


def _calibrate(logs: dict[str, Array], covariates: tuple[str, ...]) -> ExposureResponse:
    read = _real_entries(logs, ("u", "y", *covariates), "logs")
    # A zone's columns are taken by a traced index under vmap, which a NumPy array cannot take.
    logs = {name: jnp.asarray(read[name]) for name in ("u", "y", *covariates)}
    n_zones = logs["u"].shape[1]
    # In the incentive's dtype: under x64 a default block is float64, and joined to a float32 log
    # it would decide the zone's solve, not the log.
    empty = jnp.zeros((logs["u"].shape[0], 0), dtype=logs["u"].dtype)

    def one_zone(i: Array) -> tuple[Array, Array]:
        extra = jnp.stack([logs[c][:, i] for c in covariates], axis=1) if covariates else empty
        return _zone_slope(logs["u"][:, i], logs["y"][:, i], extra)

    marginal, se = jax.vmap(one_zone)(jnp.arange(n_zones))
    return ExposureResponse(marginal=marginal, se=se)


def calibrate_predictive(logs: dict[str, Array]) -> ExposureResponse:
    """MOPO-style: regress completions on the incentive per zone, ignoring the demand confounder."""
    return _calibrate(logs, ())


def calibrate_naive_causal(logs: dict[str, Array]) -> ExposureResponse:
    """Backdoor per zone: adjust each zone for its own demand, and assume SUTVA (no shared state).

    The adjustment leaves the confounding that runs through the other zones' demand (see the
    module).
    """
    return _calibrate(logs, ("demand",))


def calibrate_shared_state(logs: dict[str, Array]) -> ExposureResponse:
    """Shared-state calibration: adjust each zone for its own demand *and* the shared exposure
    aggregate, the block's total incentive, so the marginal reads the zone's response at a fixed
    total, net of the cannibalisation the total absorbs.

    It is not de-confounded. A block's incentives are its total split by a softmax of every zone's
    demand, so at a fixed total and a fixed own demand a zone's incentive moves only with the other
    zones' demand, which draws drivers from it too. Holding the total takes out the randomised
    budget and keeps that part. Against its logs rerun with incentives that do not follow demand, it
    read 0.20 to 0.59 higher on 30 markets at either precision, where :func:`calibrate_naive_causal`
    read up to 0.33 higher. :func:`pessimistic_equilibrium_allocation` reads it only through its
    tilt, ``-min(max(marginal, 0), radius * se)``.
    """
    return _calibrate(logs, ("demand", "aggregate"))


@dataclass(frozen=True)
class BudgetScaleResponse:
    """The completions' slope on the block's total incentive: what logs whose only randomised move
    is the budget's scale identify without a model (:func:`calibrate_budget_scale`)."""

    marginal: Array  # (n_zones,) each zone's completions per unit of the total, its demand held
    se: Array  # (n_zones,) HC0 heteroskedasticity-robust SE of ``marginal``
    total: float  # the block's total completions per unit of the total, every zone's demand held
    total_se: float  # HC0 SE of ``total``
    # the same slope read on the fitted equilibrium's completions at the logged blocks; nan where
    # no equilibrium was given
    fitted_total: float
    z: float  # ``(total - fitted_total) / total_se``; nan where no equilibrium was given


def calibrate_budget_scale(
    logs: dict[str, Array], *, equilibrium: FittedEquilibrium | None = None
) -> BudgetScaleResponse:
    """The one response the logs identify with no model: completions per unit of the block's total
    incentive, along the split the logging policy made.

    :meth:`SharedStateMarket.generate_logs` draws each block's budget scale independently of its
    demand and splits it by a softmax of every zone's demand. Given the demand, the scale is the
    only randomised move, so the slope of a zone's completions on the block's total, its own demand
    held, is unconfounded for ``sum_j (dY_i/du_j) pi_j``, the response along the logged split
    ``pi``, averaged over the blocks with the weights an OLS slope puts on a uniform regressor
    (Yitzhaki 1996). The demand is held only to narrow the error. Neither the ``n x n`` response
    nor a zone's response net of what it draws from the others is identified from one randomised
    number per block (Berry and Haile 2014); only a model reads them, as :func:`fit_equilibrium`'s
    does. :func:`calibrate_shared_state` holds this very total fixed, which takes out the
    randomised move.

    With ``equilibrium``, :attr:`BudgetScaleResponse.z` compares the total's slope with the one the
    fit's own completions read at the logged blocks: a check of the model along the one randomised
    direction, and a weak one. The slope is one number per market, and a fit of the wrong class can
    match it.

    Args:
        logs: ``u``, ``demand`` and ``y`` of shape ``(n_blocks, n_zones)``, finite.
        equilibrium: a fit to check against the logs, :func:`fit_equilibrium`'s.

    Raises:
        ValueError: on columns that are not finite matrices of one shape, or an equilibrium whose
            zones are not the logs'.
    """
    read = _real_entries(logs, ("u", "demand", "y"), "logs")
    u, demand, y = (jnp.asarray(read[name]) for name in ("u", "demand", "y"))
    for name, column in (("u", u), ("demand", demand), ("y", y)):
        if column.ndim != 2 or column.shape != u.shape:
            raise ValueError(
                f"logs[{name!r}] has shape {column.shape}; the logs are (n_blocks, n_zones) and"
                f" logs['u'] is {u.shape}"
            )
        if not bool(jnp.all(jnp.isfinite(column))):
            raise ValueError(f"logs[{name!r}] is not finite")
    dtype = jnp.result_type(u, demand, y, float)
    u, demand, y = (column.astype(dtype) for column in (u, demand, y))
    total_incentive = jnp.sum(u, axis=1)
    marginal, se = jax.vmap(lambda i: _zone_slope(total_incentive, y[:, i], demand[:, i][:, None]))(
        jnp.arange(u.shape[1])
    )
    total, total_se = _zone_slope(total_incentive, jnp.sum(y, axis=1), demand)
    fitted_total = z = math.nan
    if equilibrium is not None:
        if equilibrium.demand.shape != (u.shape[1],):
            raise ValueError(
                f"the equilibrium has {equilibrium.demand.shape[0]} zones and the logs {u.shape[1]}"
            )
        beta, congestion, mass = (jnp.asarray(part, dtype) for part in equilibrium._parameters())
        parameters = (beta, congestion, mass)
        attract = jnp.asarray(equilibrium.attract, dtype)
        drivers = jax.vmap(lambda d, v: _fitted_drivers(attract * d, v, parameters))(demand, u)
        completed = jnp.sum(jnp.minimum(demand, drivers), axis=1)
        fitted_total = float(_zone_slope(total_incentive, completed, demand)[0])
        z = (float(total) - fitted_total) / float(total_se)
    return BudgetScaleResponse(
        marginal=marginal,
        se=se,
        total=float(total),
        total_se=float(total_se),
        fitted_total=fitted_total,
        z=z,
    )


def sutva_allocation(market: SharedStateMarket, response: ExposureResponse, radius: float) -> Array:
    """Baseline allocation: fund zones by their pessimistic local uplift (the SUTVA assumption)."""
    pess = jnp.maximum(response.marginal - radius * response.se, 0.0)  # W-DRO shrink of a reward
    total = jnp.sum(pess)
    # Shares of the budget, so the uplift's units cancel. A floor of 1e-9 added to the total was
    # absolute in them: completions logged in millionths of their units moved the shares by 2e-4
    # of the largest. With no uplift positive the total is zero, and so is every share.
    return market.budget * pess / jnp.where(total > 0.0, total, 1.0)


@dataclass(frozen=True)
class FittedEquilibrium:
    """A congestion equilibrium estimated from logs by :func:`fit_equilibrium`.

    Drivers settle at ``x = mass * softmax(beta * (attract * demand + u - congestion * x / mass))``
    and a zone completes ``min(demand, x)``, the model class :class:`SharedStateMarket` simulates,
    with each zone's attractiveness proportional to its demand. The planning demand is each zone's
    mean logged demand, the typical block's.
    """

    demand: Array  # (n_zones,) the demand planned at: each zone's mean over the logged blocks
    attract: Array  # (n_zones,) attractiveness per unit of demand
    # (n_zones,) its standard error from the Gauss-Newton covariance at the fit; inf where the logs
    # do not identify it (:func:`_attract_se`)
    attract_se: Array
    beta: float  # drivers' sensitivity to value
    congestion: float  # crowding penalty
    mass: float  # total drivers
    rmse: float  # root mean square of the fit's residual over the logged blocks and zones
    # whether the fit stopped at Bates and Watts' relative offset of 1e-3, or where no Gauss-Newton
    # step lowers its residual; False where it ran out of steps first
    converged: bool
    # p-value of an F test of the residuals on each zone's demand and incentive, to the second
    # order: small where the logs follow some other law than the fitted class
    lack_of_fit: float
    # how far the logged incentives improve the fit: Vuong's z of the blocks' squared residuals
    # against the same fit with no incentive; below the one-sided 5% quantile the drivers' answer
    # to an incentive is not told from none, and the planner refuses the fit
    incentive_z: float
    # (n_zones + 3, n_zones + 3) the block-clustered sandwich of the parameters ``(attract,
    # log beta, log congestion, log mass)``, read over the zones :func:`_identified` keeps and the
    # three market parameters; nan in a held zone's row and column, and everywhere where the
    # Gauss-Newton matrix of the kept parameters is singular (:func:`_sandwich`). None in a fit
    # built by hand without it
    covariance: Array | None = None
    # (n_zones, n_zones) the covariance of ``demand``: the logged blocks' over their count
    demand_covariance: Array | None = None

    def drivers(self, u: Array) -> Array:
        """The equilibrium driver distribution at the planning demand under incentive ``u``."""
        return _fitted_drivers(self.attract * self.demand, u, self._parameters())

    def completions(self, u: Array) -> Array:
        """Total completions ``sum_i min(demand_i, x_i(u))`` at the planning demand."""
        return jnp.sum(jnp.minimum(self.demand, self.drivers(u)))

    def _parameters(self) -> tuple[Array, Array, Array]:
        dtype = self.demand.dtype
        return (
            jnp.asarray(self.beta, dtype),
            jnp.asarray(self.congestion, dtype),
            jnp.asarray(self.mass, dtype),
        )


def _fitted_step(params: tuple[Array, ...], x: Array) -> Array:
    """One damped logit best response, the damping ``4/(4 + beta*congestion)`` at which the map
    contracts for every game (:func:`chc.games.congestion_damping`): a fit passes through values
    of ``beta*congestion`` past the ceiling of 6 that the fixed damping of 1/2 needs."""
    attract, u, beta, congestion, mass = params
    damping = 4.0 / (4.0 + beta * congestion)
    value = beta * (attract + u - congestion * x / mass)
    return (1.0 - damping) * x + damping * mass * jax.nn.softmax(value)


def _fitted_drivers(attract: Array, u: Array, parameters: tuple[Array, Array, Array]) -> Array:
    beta, congestion, mass = parameters
    start = jnp.full(attract.shape[0], 1.0 / attract.shape[0], dtype=attract.dtype) * mass
    # every parameter goes in through the solve's params: its gradient is implicit in them alone
    return fixed_point(_fitted_step, (attract, u, beta, congestion, mass), start).x


_INCENTIVE_Z = float(special.ndtri(0.95))
"""Vuong's ``z`` below which a fit's incentive response is not told from none, one-sided 5%."""

_LACK_OF_FIT = 0.01
"""The lack-of-fit p-value below which :func:`fit_equilibrium` warns."""

_OFFSET = 1e-3
"""Bates and Watts' (1981) relative offset at which a least-squares fit has converged: the Gauss-
Newton step left is a thousandth of the radius of the parameters' confidence region."""

_LEVENBERG_MARQUARDT_CALLS = 100
"""Residual evaluations at most in the Levenberg-Marquardt polish."""

_GAUSS_NEWTON_STEPS = 20
"""Gauss-Newton steps at most after the Levenberg-Marquardt polish."""


def _polish(
    start: NDArray[np.float64],
    drivers: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    jacobian: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    demand: NDArray[np.float64],
    y: NDArray[np.float64],
    eps: float,
) -> tuple[NDArray[np.float64], bool]:
    """The least-squares fit of ``y = min(demand, drivers(p))`` from ``start``, and whether it
    converged: Bates and Watts' relative offset is at most ``_OFFSET``, or no Gauss-Newton step
    lowers the sum of squares at all, the floor of the arithmetic.

    Levenberg-Marquardt (MINPACK's, Moré 1978) polishes first. The residual is not smooth where a
    zone's drivers meet its demand, and the method stalls on such a kink, so Gauss-Newton steps
    finish, halved until they lower the sum of squares. On a zone within the equilibrium solve's
    tolerance of its demand they take the generalised gradient (Clarke 1983) of least norm in the
    Gauss-Newton metric, any weight in ``[0, 1]`` on the zone's row, and so move along the kink;
    the offset reads that gradient. A direction no column moves past ``eps`` of the largest is
    dropped: a parameter that runs off to infinity, such as a crowding penalty fitted to zero on
    its logarithm, has no offset to read. ``validation/relative_offset_stop.mac`` checks the
    Gauss-Newton step's drop, the rule's reading as a ceiling on the offset, and the least
    gradient's step along a kink.
    """
    n_parameters = start.size

    def residual(p: NDArray[np.float64]) -> NDArray[np.float64]:
        return (y - np.minimum(demand, drivers(p))).ravel()

    def rows(p: NDArray[np.float64]) -> NDArray[np.float64]:
        below = drivers(p) < demand
        return -np.where(below[..., None], jacobian(p), 0.0).reshape(-1, n_parameters)

    p = least_squares(
        residual,
        start,
        jac=rows,
        method="lm",
        x_scale="jac",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=_LEVENBERG_MARQUARDT_CALLS,
    ).x
    for _ in range(_GAUSS_NEWTON_STEPS):
        x, slope = drivers(p), -jacobian(p).reshape(-1, n_parameters)
        left = (y - np.minimum(demand, x)).ravel()
        sse = float(left @ left)
        reach = 1e-6 * np.maximum(1.0, np.linalg.norm(x, axis=1, keepdims=True))
        corner = (np.abs(x - demand) <= reach).ravel()
        below = (x < demand).ravel() & ~corner
        # the metric reads a corner's row as below its demand; the gradient weighs it by theta
        _, singular, basis = np.linalg.svd(
            np.where((below | corner)[:, None], slope, 0.0), full_matrices=False
        )
        rank = int(np.sum(singular > singular[0] * max(slope.shape) * eps))
        if rank == 0:
            return p, True
        metric = basis[:rank].T / singular[:rank]
        gradient = slope[below].T @ left[below]
        weighted = (slope[corner] * left[corner, None]).T
        if corner.any():
            theta = lsq_linear(metric.T @ weighted, -(metric.T @ gradient), bounds=(0.0, 1.0)).x
            gradient = gradient + weighted @ theta
        reduced = metric.T @ gradient
        # the full step's drop, ||Q1'r||^2, against what it leaves: the offset, squared
        drop = float(reduced @ reduced)
        if drop * max(left.size - rank, 1) <= _OFFSET**2 * rank * max(sse - drop, 0.0):
            return p, True
        step = -(metric @ reduced)
        size = 1.0
        while size >= 2.0**-20:
            trial = residual(p + size * step)
            if float(trial @ trial) < sse:
                p = p + size * step
                break
            size /= 2.0
        else:
            return p, True
    return p, False


def _attract_se(
    p: NDArray[np.float64],
    drivers: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    jacobian: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    demand: NDArray[np.float64],
    y: NDArray[np.float64],
    eps: float,
    count: int,
) -> NDArray[np.float64]:
    """The standard errors of the first ``count`` parameters, the zones' attractiveness, from the
    Gauss-Newton covariance ``s**2 (J'J)^+`` at ``p``: ``J`` the drivers' Jacobian on the rows
    below their demand or within the solve's tolerance of it (the rows :func:`_polish`'s metric
    reads), ``s**2`` the sum of squares over the residuals less the rank.

    The pseudo-inverse keeps the singular directions above ``eps`` times the largest times the
    larger dimension, as :func:`_polish` does. A zone whose unit direction leaves that row space by
    a squared share above the same bound is not identified by the logs: its error is ``inf``, not
    the pseudo-inverse's finite value, which reads only the identified part of it.
    """
    n_parameters = p.size
    x = drivers(p)
    slope = -jacobian(p).reshape(-1, n_parameters)
    left = (y - np.minimum(demand, x)).ravel()
    reach = 1e-6 * np.maximum(1.0, np.linalg.norm(x, axis=1, keepdims=True))
    read = ((x < demand) | (np.abs(x - demand) <= reach)).ravel()
    _, singular, basis = np.linalg.svd(np.where(read[:, None], slope, 0.0), full_matrices=False)
    bound = max(slope.shape) * eps
    rank = int(np.sum(singular > singular[0] * bound)) if singular[0] > 0.0 else 0
    variance = float(left @ left) / max(left.size - rank, 1)
    kept = basis[:rank, :count]
    spread = np.sqrt(variance * np.sum((kept / singular[:rank, None]) ** 2, axis=0))
    outside = 1.0 - np.sum(kept**2, axis=0)
    return np.where(outside > bound, np.inf, spread)


def _identified(attract: NDArray[np.float64], attract_se: NDArray[np.float64]) -> NDArray[np.intp]:
    """The zones whose attractiveness the logs identify: an error finite and within its size. The
    planner holds every other zone at zero incentive."""
    return np.flatnonzero(np.isfinite(attract_se) & (attract_se <= np.abs(attract)))


def _sandwich(
    p: NDArray[np.float64],
    drivers: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    jacobian: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    demand: NDArray[np.float64],
    y: NDArray[np.float64],
    eps: float,
    kept: NDArray[np.intp],
) -> NDArray[np.float64]:
    """The block-clustered sandwich ``H^-1 M H^-1`` of the least-squares parameters ``p`` over the
    indices ``kept``, the others held at their fit: ``H = sum_b J_b'J_b``, the Gauss-Newton matrix,
    and ``M = sum_b (J_b'r_b)(J_b'r_b)'``, each block's score once, so the zones' residuals within a
    block may correlate (Cameron and Miller 2015, no small-sample factor). ``J_b`` is the block's
    drivers' Jacobian on the rows :func:`_attract_se` reads, ``r_b`` its residual.

    nan in every entry where ``H`` is singular at the cut :func:`_attract_se` makes, a singular
    value of the stacked Jacobian at or below ``eps`` times the largest times its larger dimension:
    the logs do not read some combination of the kept parameters, and a pseudo-inverse would report
    its variance as zero. In a held parameter's row and column, nan."""
    n_parameters = p.size
    x = drivers(p)
    reach = 1e-6 * np.maximum(1.0, np.linalg.norm(x, axis=1, keepdims=True))
    read = (x < demand) | (np.abs(x - demand) <= reach)
    rows = np.where(read[..., None], -jacobian(p), 0.0)[..., kept]
    residual = np.where(read, y - np.minimum(demand, x), 0.0)
    stacked = rows.reshape(-1, kept.size)
    covariance = np.full((n_parameters, n_parameters), np.nan)
    _, singular, basis = np.linalg.svd(stacked, full_matrices=False)
    if not singular[0] > 0.0 or singular[-1] <= singular[0] * max(stacked.shape) * eps:
        return covariance
    # H^-1 = V diag(1/s^2) V' from the stacked Jacobian's SVD: no Gram is formed, whose condition
    # is the Jacobian's squared
    bread = basis.T / singular**2 @ basis
    scores = np.einsum("bip,bi->bp", rows, residual)
    meat = scores.T @ scores
    covariance[np.ix_(kept, kept)] = bread @ meat @ bread
    return covariance


def _lack_of_fit(
    residual: NDArray[np.float64], demand: NDArray[np.float64], u: NDArray[np.float64]
) -> float:
    """The p-value of an F test of ``residual`` on each zone's own intercept, demand and incentive
    to the second order, every zone's terms together. Under the fitted class the residual is the
    logs' noise and follows none of them; the fit's own parameters, already projected out, make the
    test conservative. A residual of exactly zero fits, at 1."""
    blocks, zones = residual.shape
    explained = kept = 0.0
    terms = 0
    for zone in range(zones):
        columns = [
            (column - column.mean()) / (column.std() or 1.0)
            for column in (demand[:, zone], u[:, zone])
        ]
        d, v = columns
        design = np.stack([np.ones(blocks), d, v, d * d, v * v, d * v], axis=1)
        coefficients, _, rank, _ = np.linalg.lstsq(design, residual[:, zone], rcond=None)
        left = residual[:, zone] - design @ coefficients
        explained += float(residual[:, zone] @ residual[:, zone] - left @ left)
        kept += float(left @ left)
        terms += int(rank)
    total = blocks * zones
    if explained + kept == 0.0:
        return 1.0
    if kept == 0.0:
        return 0.0
    statistic = (explained / terms) / (kept / (total - terms))
    return float(stats.f.sf(statistic, terms, total - terms))


def _incentive_z(with_u: NDArray[np.float64], without_u: NDArray[np.float64]) -> float:
    """Vuong's ``z`` of the blocks' summed squared residuals without the incentive less with it:
    positive where the incentive explains the completions."""
    gain = np.sum(without_u**2, axis=1) - np.sum(with_u**2, axis=1)
    spread = float(np.std(gain, ddof=1))
    if spread == 0.0:
        return 0.0 if float(np.mean(gain)) == 0.0 else math.copysign(math.inf, float(np.mean(gain)))
    return math.sqrt(gain.size) * float(np.mean(gain)) / spread


def fit_equilibrium(logs: dict[str, Array], *, max_iter: int = 500) -> FittedEquilibrium:
    """Estimate the congestion equilibrium the logs were generated under, by least squares.

    The fitted model is :class:`FittedEquilibrium`'s: per zone, an attractiveness per unit of
    demand, and for the market, the drivers' sensitivity, the crowding penalty and the drivers'
    total, ``n_zones + 3`` parameters, fitted to each block's completions ``y`` through each
    block's equilibrium under its logged incentive ``u`` and demand. The positive parameters are
    fitted on their logarithms, by L-BFGS on gradients implicit in the equilibrium; the solve
    starts from an attractiveness of 1, a sensitivity and a penalty of 1, and a total 1.2 times
    the largest block's completions, since a block completes no more rides than it has drivers.
    Levenberg-Marquardt and Gauss-Newton steps on the blocks' residuals then polish it until the
    step left is a thousandth of the radius of the parameters' confidence region (Bates and Watts'
    relative offset, 1981). L-BFGS alone stopped on the relative fall of its objective, 2.2e-9 of
    1 where the mean squared residual is 4e-4, and said it converged: on markets 0-3 at offsets of
    0.08 to 2.6, its mean squared residual up to 2.1% above the polished fit's.

    The demand is logged and enters the model, so the logging policy's chase of it does not
    confound the fit. What is assumed: the model class, which is the simulator's, and an
    attractiveness that moves with the demand. Neither is proved, but a market that breaks them
    leaves its mark on the residuals, and :attr:`FittedEquilibrium.lack_of_fit` reads it: an F test
    of each zone's residual on its own demand and incentive to the second order. Below 0.01 the
    fit warns. On 20 logs of 400 blocks from five markets whose attractiveness stays put while the
    demand moves, the fit converged on every one with the sensitivity 0.54 too high on average
    (1.17 at most), its residual 2.4 to 3.1 times the noise's, and nothing said; the test read
    below 3e-226 on each, and below 1e-15 on 20 logs of 50 blocks. On 50 logs of the class itself
    it read under 0.01 once (0.009) at 400 blocks, and never at 50.

    The fit is then fitted again with every incentive at zero, and
    :attr:`FittedEquilibrium.incentive_z` is Vuong's ``z`` of what the incentives gain, block by
    block: a fit whose drivers do not tell an incentive from none plans nothing, and
    :func:`pessimistic_equilibrium_allocation` refuses it. On the same markets' logs with the
    incentives shuffled across the blocks, the planner had spent the whole budget through the fit,
    and on some markets lost completions by it; its ``z`` read -5.0 to 1.0 on 400 blocks and at
    most 1.28 on 50, under the one-sided 5% quantile 1.645 on all 40, where the logs themselves
    read at least 11.7 and 3.3, and the markets outside the class at least 14.2 and 3.8.

    The fit reads its ``n_zones + 3`` parameters from at least as many blocks: on five markets of
    12 zones, 3 blocks read the sensitivity 1.7 off at the median and 94 at worst, and 15 blocks
    0.055 and 0.26. A zone that draws
    almost no drivers in any logged block tells little of its attractiveness: the default market's
    zone 8 averages 0.0008 drivers on a demand of 0.50, and its attractiveness read 0.73 for 0.63
    on 400 blocks and 0.67 on 6400, where moving it from 0.63 to 0.95 moves the mean squared
    residual by 1e-10 of its 4e-4. On short logs such a zone's attractiveness is read by little
    but the fit's own path: on 50 blocks of markets 0-29, four fits read a zone's more than 1 off
    and one ran it to -4.8e4, the logs' drivers there no fewer for it.

    :attr:`FittedEquilibrium.attract_se` is each zone's standard error from the Gauss-Newton
    covariance ``s**2 (J'J)^+`` at the fit, ``J`` the drivers' Jacobian in the parameters on the
    blocks where a zone is below its demand, ``s**2`` the mean square of the residual over the
    blocks' zones less the rank. A zone whose direction the pseudo-inverse drops, for want of any
    block that moves its drivers, reads ``inf``. Of the 60 fits in float64 five read a zone's
    error infinite or above its attractiveness's size: those four, and on 400 blocks one zone read
    -0.01 with an error of 0.40. :func:`pessimistic_equilibrium_allocation` holds such a zone at
    zero incentive.

    :attr:`FittedEquilibrium.covariance` is the parameters' block-clustered sandwich
    ``H^-1 M H^-1`` (Cameron and Miller 2015): ``H`` the Gauss-Newton matrix on the same rows, ``M``
    the blocks' scores, so the zones' errors within a block may correlate. It is read over the zones
    the planner does not hold and the three market parameters, the held zones' attractiveness
    fixed at the fit, and is nan where the logs leave a combination of those unread.
    :attr:`FittedEquilibrium.demand_covariance` is the planning demand's. The planner's expected
    objective reads both; the tilted one reads neither. Over 100 logs of 400 blocks of market 0
    (market 1), its 95% intervals held the sensitivity, the penalty and the total in 99% (96%),
    98% (96%) and 96% (97%) of the logs, and the median zone's attractiveness in 96% (94%), the
    worst zone's in 83% (87%). On 50 blocks of market 0, in 94%, 95% and 89%, and 96% at the median
    zone; a zone that draws almost no drivers, held on 64 of those logs, in 64% of the rest.

    Args:
        logs: :meth:`SharedStateMarket.generate_logs`'s, or any with ``u``, ``demand`` and ``y``
            of shape ``(n_blocks, n_zones)``, ``n_blocks`` at least the ``n_zones + 3`` parameters
            and the demand positive.
        max_iter: L-BFGS iterations at most, before the polish.

    Raises:
        ValueError: on columns that are not finite matrices of one shape, fewer blocks than
            parameters, or a demand that is not positive.

    Warns:
        UserWarning: where :attr:`FittedEquilibrium.lack_of_fit` is below 0.01.
    """
    read = _real_entries(logs, ("u", "demand", "y"), "logs")
    u, demand, y = (jnp.asarray(read[name]) for name in ("u", "demand", "y"))
    for name, column in (("u", u), ("demand", demand), ("y", y)):
        if column.ndim != 2 or column.shape != u.shape:
            raise ValueError(
                f"logs[{name!r}] has shape {column.shape}; the logs are (n_blocks, n_zones) and"
                f" logs['u'] is {u.shape}"
            )
        if not bool(jnp.all(jnp.isfinite(column))):
            raise ValueError(f"logs[{name!r}] is not finite")
    n_blocks, n_zones = u.shape
    if n_blocks < n_zones + 3:
        raise ValueError(
            f"{n_blocks} blocks for {n_zones} zones: the fit has n_zones + 3 = {n_zones + 3}"
            " parameters and reads them from at least as many blocks"
        )
    if not bool(jnp.all(demand > 0.0)):
        raise ValueError(
            f"logs['demand'] reaches {float(jnp.min(demand))}; a zone's demand is positive"
        )
    dtype = jnp.result_type(u, demand, y, float)
    u, demand, y = (column.astype(dtype) for column in (u, demand, y))

    def unpack(p: Array) -> tuple[Array, tuple[Array, Array, Array]]:
        return p[:n_zones], (jnp.exp(p[n_zones]), jnp.exp(p[n_zones + 1]), jnp.exp(p[n_zones + 2]))

    def one_block(p: Array, d: Array, v: Array) -> Array:
        attract, parameters = unpack(p)
        return _fitted_drivers(attract * d, v, parameters)

    def drivers(p: Array, incentive: Array) -> Array:
        return jax.vmap(lambda d, v: one_block(p, d, v))(demand, incentive)

    def residual(p: Array, incentive: Array) -> Array:
        return y - jnp.minimum(demand, drivers(p, incentive))

    value_and_grad = jax.jit(
        jax.value_and_grad(lambda p, incentive: jnp.mean(residual(p, incentive) ** 2))
    )
    drivers_at = jax.jit(drivers)
    # each block's drivers in the parameters, reverse mode through the fixed point
    jacobians = jax.jit(jax.vmap(jax.jacrev(one_block), in_axes=(None, 0, 0)))
    mass = 1.2 * float(jnp.max(jnp.sum(y, axis=1)))
    start = np.concatenate([np.ones(n_zones), [0.0, 0.0, math.log(mass)]])
    eps = float(jnp.finfo(dtype).eps)
    demand_64, y_64 = np.asarray(demand, np.float64), np.asarray(y, np.float64)

    def solve(incentive: Array) -> tuple[NDArray[np.float64], bool, NDArray[np.float64]]:
        def objective(p: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            value, grad = value_and_grad(jnp.asarray(p, dtype=dtype), incentive)
            return float(value), np.asarray(grad, dtype=np.float64)

        found = minimize(
            objective, start, jac=True, method="L-BFGS-B", options={"maxiter": max_iter}
        )
        p, converged = _polish(
            found.x,
            lambda p: np.asarray(drivers_at(jnp.asarray(p, dtype=dtype), incentive), np.float64),
            lambda p: np.asarray(
                jacobians(jnp.asarray(p, dtype=dtype), demand, incentive), np.float64
            ),
            demand_64,
            y_64,
            eps,
        )
        left = np.asarray(residual(jnp.asarray(p, dtype=dtype), incentive), np.float64)
        return p, converged, left

    fitted, converged, left = solve(u)
    _, _, idle = solve(jnp.zeros_like(u))
    attract_se = _attract_se(
        fitted,
        lambda p: np.asarray(drivers_at(jnp.asarray(p, dtype=dtype), u), np.float64),
        lambda p: np.asarray(jacobians(jnp.asarray(p, dtype=dtype), demand, u), np.float64),
        demand_64,
        y_64,
        eps,
        n_zones,
    )
    kept = np.concatenate([_identified(fitted[:n_zones], attract_se), n_zones + np.arange(3)])
    covariance = _sandwich(
        fitted,
        lambda p: np.asarray(drivers_at(jnp.asarray(p, dtype=dtype), u), np.float64),
        lambda p: np.asarray(jacobians(jnp.asarray(p, dtype=dtype), demand, u), np.float64),
        demand_64,
        y_64,
        eps,
        kept.astype(np.intp),
    )
    lack_of_fit = _lack_of_fit(left, np.asarray(demand, np.float64), np.asarray(u, np.float64))
    attract, (beta, congestion, total) = unpack(jnp.asarray(fitted, dtype=dtype))
    rmse = math.sqrt(float(np.mean(left**2)))
    if lack_of_fit < _LACK_OF_FIT:
        warnings.warn(
            f"fit_equilibrium: the residuals still follow the zones' demand and incentive (F test"
            f" p = {lack_of_fit:.2g}, rmse {rmse:.3g}): the logs do not follow a logit congestion"
            " equilibrium whose attractiveness moves with the demand, and a plan through this fit"
            " reads the nearest one",
            UserWarning,
            stacklevel=2,
        )
    return FittedEquilibrium(
        demand=jnp.mean(demand, axis=0),
        attract=attract,
        attract_se=jnp.asarray(attract_se, dtype=dtype),
        beta=float(beta),
        congestion=float(congestion),
        mass=float(total),
        rmse=rmse,
        converged=converged,
        lack_of_fit=lack_of_fit,
        incentive_z=_incentive_z(left, idle),
        covariance=jnp.asarray(covariance, dtype=dtype),
        demand_covariance=jnp.asarray(
            np.atleast_2d(np.cov(demand_64, rowvar=False)) / n_blocks, dtype=dtype
        ),
    )


def pessimistic_equilibrium_allocation(
    market: SharedStateMarket,
    response: ExposureResponse | None,
    *,
    radius: float = 1.0,
    steps: int = 600,
    equilibrium: FittedEquilibrium | None = None,
    objective: PlanObjective = "tilted",
) -> Array:
    """CHC allocation: plan the budget *through* the congestion equilibrium (``chc.games``) so the
    driver-conservation (cannibalisation) channel the SUTVA baselines assume away is structural. The
    adjusted response gates the plan through a W-DRO pessimism tilt (``radius * SE``): where
    the offline estimate is weak or unsupported, the incentive is discounted rather than trusted.

    A zone whose attractiveness the fit does not identify, its
    :attr:`FittedEquilibrium.attract_se` infinite or above the attractiveness's size, is held at
    zero incentive, and the budget is planned over the rest: the logs do not say what an incentive
    there buys. Such a zone draws almost no drivers in any logged block; on 50 blocks of market 2
    the plan had spent 3.1 of the budget of 6 on one and won 0.15 of the exact oracle's lift.

    ``equilibrium`` is the one planned through, and is required: :func:`fit_equilibrium`'s,
    estimated from the logs. ``market`` gives only the budget and the zones; its own base demand,
    attractiveness and solver are the simulator's truth, which no log supplies, and the planner does
    not read them. ``response`` enters only through the tilt.

    Raises:
        TypeError: when ``equilibrium`` is not given.
        ValueError: on a response or an equilibrium whose zones are not ``market``'s, an
            equilibrium whose drivers do not tell an incentive from none
            (:attr:`FittedEquilibrium.incentive_z`): every plan through it reads the same, or one
            that identifies no zone's attractiveness.
    """
    for name, part in (("marginal", response.marginal), ("se", response.se)):
        if jnp.shape(part) != (market.n_zones,):
            raise ValueError(
                f"response.{name} has shape {jnp.shape(part)}; the market has {market.n_zones}"
                " zones"
            )
    if equilibrium is None:
        raise TypeError(
            "pessimistic_equilibrium_allocation plans through an equilibrium estimated from the"
            " logs: pass equilibrium=fit_equilibrium(logs). The market's own equilibrium is the"
            " simulator's truth, which no log supplies"
        )
    if equilibrium.demand.shape != (market.n_zones,):
        raise ValueError(
            f"the equilibrium has {equilibrium.demand.shape[0]} zones and the market"
            f" {market.n_zones}"
        )
    if not equilibrium.incentive_z >= _INCENTIVE_Z:
        raise ValueError(
            f"the fitted equilibrium's drivers do not tell an incentive from none (z ="
            f" {equilibrium.incentive_z:.2f}, under {_INCENTIVE_Z:.3f}): the logged"
            " incentives explain no more of the completions than none, so no plan through it"
            " is better than another"
        )
    # The worst response within radius SEs that is not negative, less the estimate projected onto
    # the same prior: -min(max(m, 0), radius * se), never a bonus (validation/marketplace_tilt.mac).
    # Read against the raw estimate, a negative one earned its zone a bonus of |m| per unit.
    pess = jnp.maximum(response.marginal - radius * response.se, 0.0)
    tilt = pess - jnp.maximum(response.marginal, 0.0)
    completions = equilibrium.completions
    se, attract = np.asarray(equilibrium.attract_se), np.asarray(equilibrium.attract)
    free = np.flatnonzero(np.isfinite(se) & (se <= np.abs(attract)))
    if not free.size:
        raise ValueError(
            "the fitted equilibrium identifies no zone's attractiveness: every attract_se is"
            " infinite or above the attractiveness's size, so no zone's incentive is read"
        )
    zero = jnp.zeros(market.n_zones, dtype=equilibrium.attract.dtype)

    def surrogate(v: Array) -> Array:
        u = zero.at[free].set(v)
        return completions(u) + jnp.dot(tilt, u)

    planned = stackelberg_allocation(surrogate, free.size, market.budget, steps=steps)
    return zero.at[free].set(planned)


def interference_bias(
    market: SharedStateMarket, response: ExposureResponse, allocation: Array
) -> float:
    """SUTVA over-count: the lift a per-zone (additive) model predicts minus the lift realised at
    the equilibrium -- the interference the naive planner ignores (positive = over-prediction).
    """
    predicted = float(jnp.dot(response.marginal, allocation))
    realised = market.value(allocation) - market.value(jnp.zeros(market.n_zones))
    return predicted - realised
