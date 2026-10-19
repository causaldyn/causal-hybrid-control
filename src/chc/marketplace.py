"""Offline causal decision-making under equilibrium interference.

The problem, end to end: learn an incentive policy from *confounded,
switchback-logged* marketplace data where (a) the logging policy is confounded (operators already
incentivise busy zones, so demand drives both treatment and outcome), and (b) SUTVA fails because
drivers are mobile -- incentivising a zone pulls drivers from its neighbours through a shared
equilibrium (Munro-Wager-Xu market-equilibrium interference; Wager-Xu supply cannibalisation;
shared-state DML, arXiv 2504.08836). Three failure modes, each a real baseline:

* **predictive / MOPO** fits an outcome model on the confounded logs -- confounding inflates the
  response of already-busy zones, so it over-allocates to them (where drivers only cannibalise).
* **naive causal** de-confounds per zone (backdoor on demand) but assumes SUTVA -- it allocates by
  local uplift, double-counting drivers the market cannot supply (predicts a lift it never gets).
* **CHC** de-confounds *and* plans through the structural equilibrium (``chc.games``) with an
  offline-pessimism margin (a Wasserstein/DRO-style shrink by the estimate's influence-function SE),
  so cannibalisation is built in and extrapolation is not trusted.

What the comparison measures: :func:`pessimistic_equilibrium_allocation` is handed the simulator's
own base demand, attractiveness and equilibrium map, which no estimate from the logs supplies, and
the logs reach it only through the pessimism tilt, at most ``radius`` standard errors per zone. Its
gap to the oracle is therefore the value of knowing the equilibrium, not of de-confounding: on the
default market the plan from a response of zero lands as close to the oracle as the plan from the
de-confounded one. The SUTVA baselines get no such structure. :func:`fit_equilibrium` estimates
the equilibrium from the logs instead, and the planner takes it as ``equilibrium=`` (ADR 0114).

Everything is differentiable and reuses the equilibrium solver in :mod:`chc.games`.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray
from scipy.optimize import minimize

from chc import _units
from chc.frames import _real_entries
from chc.games import fixed_point, softmax_congestion_equilibrium, stackelberg_allocation


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
        contracting (:func:`chc.games.congestion_contraction_certificate`) and the solve converges.
        """
        return softmax_congestion_equilibrium(attract, u, self.congestion, self.mass, self.beta).x

    def completions(self, u: Array) -> Array:
        """Total realised rides = sum_i min(demand_i, drivers_i) at the equilibrium."""
        demand, attract = self._base()
        return jnp.sum(jnp.minimum(demand, self._equilibrium(attract, u)))

    def value(self, u: Array) -> float:
        return float(self.completions(u))

    def oracle_allocation(self, steps: int = 1200) -> Array:
        """Equilibrium-aware optimum with full model access (the regret reference)."""
        return stackelberg_allocation(self.completions, self.n_zones, self.budget, steps=steps)

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
    """Backdoor per zone: adjust for demand (de-confound) but assume SUTVA (no shared state)."""
    return _calibrate(logs, ("demand",))


def calibrate_shared_state(logs: dict[str, Array]) -> ExposureResponse:
    """AIPW-style shared-state calibration: de-confound *and* control the shared exposure aggregate,
    so the marginal is the direct response net of the cannibalisation the aggregate absorbs.
    """
    return _calibrate(logs, ("demand", "aggregate"))


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
    beta: float  # drivers' sensitivity to value
    congestion: float  # crowding penalty
    mass: float  # total drivers
    rmse: float  # root mean square of the fit's residual over the logged blocks and zones
    converged: bool  # whether the least-squares solve met its tolerance

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


def fit_equilibrium(logs: dict[str, Array], *, max_iter: int = 500) -> FittedEquilibrium:
    """Estimate the congestion equilibrium the logs were generated under, by least squares.

    The fitted model is :class:`FittedEquilibrium`'s: per zone, an attractiveness per unit of
    demand, and for the market, the drivers' sensitivity, the crowding penalty and the drivers'
    total, ``n_zones + 3`` parameters, fitted to each block's completions ``y`` through each
    block's equilibrium under its logged incentive ``u`` and demand. The positive parameters are
    fitted on their logarithms, by L-BFGS on gradients implicit in the equilibrium; the solve
    starts from an attractiveness of 1, a sensitivity and a penalty of 1, and a total 1.2 times
    the largest block's completions, since a block completes no more rides than it has drivers.

    The demand is logged and enters the model, so the logging policy's chase of it does not
    confound the fit. What is assumed and not tested: the model class, which is the simulator's
    (a market whose drivers follow another law is fitted by the nearest logit equilibrium), and an
    attractiveness that moves with the demand. The parameters' sampling error is not carried into
    the plan; :attr:`FittedEquilibrium.rmse` against the logs' noise says how well the class fits.

    Args:
        logs: :meth:`SharedStateMarket.generate_logs`'s, or any with ``u``, ``demand`` and ``y``
            of shape ``(n_blocks, n_zones)``.
        max_iter: L-BFGS iterations at most.
    """
    read = _real_entries(logs, ("u", "demand", "y"), "logs")
    u, demand, y = (jnp.asarray(read[name]) for name in ("u", "demand", "y"))
    dtype = jnp.result_type(u, demand, y, float)
    u, demand, y = (column.astype(dtype) for column in (u, demand, y))
    n_zones = u.shape[1]

    def unpack(p: Array) -> tuple[Array, tuple[Array, Array, Array]]:
        return p[:n_zones], (jnp.exp(p[n_zones]), jnp.exp(p[n_zones + 1]), jnp.exp(p[n_zones + 2]))

    def loss(p: Array) -> Array:
        attract, parameters = unpack(p)
        drivers = jax.vmap(lambda d, v: _fitted_drivers(attract * d, v, parameters))(demand, u)
        return jnp.mean((y - jnp.minimum(demand, drivers)) ** 2)

    value_and_grad = jax.jit(jax.value_and_grad(loss))

    def objective(p: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        value, grad = value_and_grad(jnp.asarray(p, dtype=dtype))
        return float(value), np.asarray(grad, dtype=np.float64)

    mass = 1.2 * float(jnp.max(jnp.sum(y, axis=1)))
    start = np.concatenate([np.ones(n_zones), [0.0, 0.0, math.log(mass)]])
    result = minimize(objective, start, jac=True, method="L-BFGS-B", options={"maxiter": max_iter})
    attract, (beta, congestion, total) = unpack(jnp.asarray(result.x, dtype=dtype))
    return FittedEquilibrium(
        demand=jnp.mean(demand, axis=0),
        attract=attract,
        beta=float(beta),
        congestion=float(congestion),
        mass=float(total),
        rmse=math.sqrt(float(result.fun)),
        converged=bool(result.success),
    )


def pessimistic_equilibrium_allocation(
    market: SharedStateMarket,
    response: ExposureResponse,
    *,
    radius: float = 1.0,
    steps: int = 600,
    equilibrium: FittedEquilibrium | None = None,
) -> Array:
    """CHC allocation: plan the budget *through* the congestion equilibrium (``chc.games``) so the
    driver-conservation (cannibalisation) channel the SUTVA baselines assume away is structural. The
    de-confounded response gates the plan through a W-DRO pessimism tilt (``radius * SE``): where
    the offline estimate is weak or unsupported, the incentive is discounted rather than trusted.

    ``equilibrium`` is the one planned through: :func:`fit_equilibrium`'s, estimated from the logs.
    Without it the plan goes through ``market``'s own, its base demand, its attractiveness and its
    solver, the simulator's truth that no log supplies, with a ``FutureWarning``: a future release
    will require ``equilibrium``. ``market`` gives the budget and the zones either way.
    ``response`` enters only through the tilt, so through the simulator's equilibrium the plan is
    close to the oracle whatever the response says.
    """
    # The worst response within radius SEs that is not negative, less the estimate projected onto
    # the same prior: -min(max(m, 0), radius * se), never a bonus (validation/marketplace_tilt.mac).
    # Read against the raw estimate, a negative one earned its zone a bonus of |m| per unit.
    pess = jnp.maximum(response.marginal - radius * response.se, 0.0)
    tilt = pess - jnp.maximum(response.marginal, 0.0)
    if equilibrium is None:
        warnings.warn(
            "pessimistic_equilibrium_allocation plans through the simulator's own equilibrium, "
            "which no log supplies; pass equilibrium=fit_equilibrium(logs) to plan on one "
            "estimated from the logs. A future release will require it",
            FutureWarning,
            stacklevel=2,
        )
        demand, attract = market._base()

        def completions(u: Array) -> Array:
            return jnp.sum(jnp.minimum(demand, market._equilibrium(attract, u)))

    else:
        completions = equilibrium.completions

    def surrogate(u: Array) -> Array:
        return completions(u) + jnp.dot(tilt, u)

    return stackelberg_allocation(surrogate, market.n_zones, market.budget, steps=steps)


def interference_bias(
    market: SharedStateMarket, response: ExposureResponse, allocation: Array
) -> float:
    """SUTVA over-count: the lift a per-zone (additive) model predicts minus the lift realised at
    the equilibrium -- the interference the naive planner ignores (positive = over-prediction).
    """
    predicted = float(jnp.dot(response.marginal, allocation))
    realised = market.value(allocation) - market.value(jnp.zeros(market.n_zones))
    return predicted - realised
