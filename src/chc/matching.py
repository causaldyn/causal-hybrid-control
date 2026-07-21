"""Kantorovich optimal transport for marketplace matching -- dispatch plan + dual surge prices.

The discrete, marketplace-native sibling of the continuum :mod:`chc.transport`: driver->rider
dispatch is Kantorovich's transportation problem (1939; Nobel 1975) -- move drivers/zone
(``supply``) to riders/zone (``demand``) at least travel ``cost``. The **dual potentials are the
market-clearing prices**: the demand dual ``g`` is the surge signal (higher where demand outstrips
supply), free as the dual of the same optimisation -- which no dispatch heuristic gives. Solved by
entropic (log-domain, differentiable) Sinkhorn so pricing is optimisable; ``eps -> 0`` recovers the
exact LP (Octave `glpk` cross-check: Kantorovich-Rubinstein gap = 0). NumPy baselines, JAX OT core.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import logsumexp

from chc.games import project_simplex


@dataclass(frozen=True)
class SinkhornResult:
    """A solved entropic OT: plan, dual potentials (surge prices), cost, and duality gap."""

    plan: Array  # (m, n) transport / matching plan
    potentials_f: Array  # (m,) supply-side dual potentials
    potentials_g: Array  # (n,) demand-side dual potentials = market-clearing (surge) prices
    transport_cost: float  # <plan, cost>
    duality_gap: float  # |cost - dual|; -> 0 as eps -> 0 (Kantorovich-Rubinstein)


def sinkhorn(
    cost: Array, supply: Array, demand: Array, *, eps: float = 0.05, iters: int = 1000
) -> SinkhornResult:
    """Entropic (log-domain, stable) Kantorovich OT; returns the plan and dual potentials (surge).

    ``cost`` is ``(m, n)`` dispatch cost; ``supply``/``demand`` are the marginals (equal totals).
    Recovers the exact transportation LP as ``eps -> 0``; the demand potentials ``g`` are the
    market-clearing surge prices.
    """
    cost = jnp.asarray(cost)
    a, b = jnp.asarray(supply), jnp.asarray(demand)
    log_a, log_b = jnp.log(a), jnp.log(b)

    def step(carry: tuple[Array, Array], _: Array) -> tuple[tuple[Array, Array], None]:
        f, g = carry
        f = eps * (log_a - logsumexp((g[None, :] - cost) / eps, axis=1))
        g = eps * (log_b - logsumexp((f[:, None] - cost) / eps, axis=0))
        return (f, g), None

    (f, g), _ = jax.lax.scan(step, (jnp.zeros_like(a), jnp.zeros_like(b)), None, length=iters)
    plan = jnp.exp((f[:, None] + g[None, :] - cost) / eps)
    transport = float(jnp.sum(plan * cost))
    dual = float(jnp.dot(f, a) + jnp.dot(g, b))
    return SinkhornResult(plan, f, g, transport, abs(transport - dual))


def _nearest_local(cost: np.ndarray, supply: np.ndarray, demand: np.ndarray) -> tuple[float, float]:
    """Naive: each rider zone served only by its single nearest driver zone; rest is stranded."""
    a, b = supply.copy(), demand.copy()
    served_cost, served = 0.0, 0.0
    for j in np.argsort(-b):  # high-demand zones first
        i = int(np.argmin(cost[:, j]))
        take = min(a[i], b[j])
        served_cost += take * cost[i, j]
        a[i] -= take
        served += take
    return served_cost, served / float(demand.sum())


def _nearest_reroute(cost: np.ndarray, supply: np.ndarray, demand: np.ndarray) -> float:
    """Myopic but complete: each rider zone takes nearest available supply, rerouting to fill."""
    a, b = supply.copy(), demand.copy()
    total = 0.0
    for j in np.argsort(-b):
        for i in np.argsort(cost[:, j]):
            if b[j] <= 0.0:
                break
            take = min(a[i], b[j])
            total += take * cost[i, j]
            a[i] -= take
            b[j] -= take
    return total


@dataclass(frozen=True)
class MarketplaceMatching:
    """Zones with driver ``supply``, rider ``demand``, and ``cost`` (travel) between them."""

    supply: Array  # (Z,) drivers per zone
    demand: Array  # (Z,) riders per zone (equal total to supply)
    cost: Array  # (Z, Z) zone-to-zone dispatch cost (travel distance)
    eps: float = 0.02
    iters: int = 2000

    @classmethod
    def synthetic_city(cls, n_zones: int = 8, seed: int = 0) -> MarketplaceMatching:
        """Drivers cluster centrally, riders spread to the suburbs -- a supply-demand mismatch."""
        rng = np.random.default_rng(seed)
        positions = rng.uniform(-2.0, 2.0, (n_zones, 2))
        radius = np.linalg.norm(positions, axis=1)
        supply = np.exp(-radius)  # drivers concentrate near the centre (small radius)
        demand = 0.3 + radius  # riders concentrate in the periphery
        supply *= demand.sum() / supply.sum()  # balance totals
        cost = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
        return cls(jnp.asarray(supply), jnp.asarray(demand), jnp.asarray(cost))

    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.asarray(self.cost), np.asarray(self.supply), np.asarray(self.demand)

    def optimal(self) -> SinkhornResult:
        """The Kantorovich OT dispatch (100% served at min travel) + the dual surge prices."""
        return sinkhorn(self.cost, self.supply, self.demand, eps=self.eps, iters=self.iters)

    def surge_prices(self) -> Array:
        """Market-clearing surge = the demand-side dual potentials (high in undersupplied zones)."""
        return self.optimal().potentials_g

    def nearest_local(self) -> tuple[float, float]:
        """Naive local dispatch: ``(served_cost, coverage)`` -- strands demand it cannot reach."""
        return _nearest_local(*self._arrays())

    def nearest_reroute_cost(self) -> float:
        """Myopic complete dispatch cost (100% served, but locally greedy -- costlier than OT)."""
        return _nearest_reroute(*self._arrays())

    def surge_rebalance(self, step: float = 0.5) -> tuple[float, float]:
        """Apply surge as a driver incentive; drivers best-respond toward high-price zones.

        Returns supply-demand imbalance ``(before, after)``: the surge prices, projected onto the
        supply simplex (a best-response via :func:`chc.games.project_simplex`), pull drivers toward
        high-price zones -- surge as an *intervention* whose equilibrium effect the dual predicts.
        """
        g = self.surge_prices()
        s, d = jnp.asarray(self.supply), jnp.asarray(self.demand)
        before = float(jnp.sum(jnp.abs(s - d)))
        s_new = project_simplex(s + step * (g - jnp.mean(g)), float(jnp.sum(s)))
        return before, float(jnp.sum(jnp.abs(s_new - d)))


def marketplace_report(matching: MarketplaceMatching) -> str:
    """Kantorovich OT dispatch vs two naive failure modes, plus the surge/equilibrium win."""
    opt = matching.optimal()
    _, coverage = matching.nearest_local()
    reroute = matching.nearest_reroute_cost()
    before, after = matching.surge_rebalance()
    total = float(jnp.sum(matching.demand))
    saved = (reroute - opt.transport_cost) / reroute * 100
    stranded = (1.0 - coverage) * total
    return "\n".join(
        [
            f"Kantorovich OT: 100% served at min cost {opt.transport_cost:.2f} + surge prices.",
            f"naive local-only dispatch strands {stranded:.1f} of {total:.1f} riders "
            f"({(1.0 - coverage) * 100:.0f}%) where demand exceeds local supply.",
            f"naive reroute (100% served): cost {reroute:.2f} -> OT is {saved:.1f}% cheaper.",
            f"surge -> driver reallocation cuts imbalance {before:.2f} -> {after:.2f} "
            f"({(before - after) / before * 100:.0f}%).",
            f"Kantorovich-Rubinstein duality gap = {opt.duality_gap:.2e} (surge = free dual).",
        ]
    )
