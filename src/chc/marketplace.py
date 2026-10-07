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

Everything is differentiable and reuses the equilibrium solver in :mod:`chc.games`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from chc import _units
from chc.frames import _real_entries
from chc.games import softmax_congestion_equilibrium, stackelberg_allocation


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
    """A calibrated per-zone incentive response ``dY/du`` plus its influence-function SE."""

    marginal: Array  # (n_zones,) estimated marginal completion response to an incentive
    se: Array  # (n_zones,) heteroskedastic-robust SE of ``marginal``


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
    logs = _real_entries(logs, ("u", "y", *covariates), "logs")
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


def pessimistic_equilibrium_allocation(
    market: SharedStateMarket, response: ExposureResponse, *, radius: float = 1.0, steps: int = 600
) -> Array:
    """CHC allocation: plan the budget *through* the congestion equilibrium (``chc.games``) so the
    driver-conservation (cannibalisation) channel the SUTVA baselines assume away is structural. The
    de-confounded response gates the plan through a W-DRO pessimism tilt (``radius * SE``): where
    the offline estimate is weak or unsupported, the incentive is discounted rather than trusted.
    """
    demand, attract = market._base()
    pess = jnp.maximum(response.marginal - radius * response.se, 0.0)
    tilt = pess - response.marginal  # <= 0: pessimistic discount where the response is uncertain

    def surrogate(u: Array) -> Array:
        return jnp.sum(jnp.minimum(demand, market._equilibrium(attract, u))) + jnp.dot(tilt, u)

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
