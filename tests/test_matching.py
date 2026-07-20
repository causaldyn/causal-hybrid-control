"""Kantorovich OT matching: exact-LP recovery, strong duality, and dispatch beating naive."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chc.matching import MarketplaceMatching, marketplace_report, sinkhorn

# the 3x3 transportation instance cross-checked in Octave glpk (LP optimum = 18)
COST = jnp.array([[1.0, 2, 3], [4, 1, 2], [3, 2, 1]])
SUPPLY = jnp.array([4.0, 5, 3])
DEMAND = jnp.array([6.0, 3, 3])


def test_sinkhorn_marginals_match_supply_and_demand() -> None:
    res = sinkhorn(COST, SUPPLY, DEMAND, eps=0.05, iters=1000)
    assert jnp.allclose(res.plan.sum(axis=1), SUPPLY, atol=1e-3)  # rows = supply
    assert jnp.allclose(res.plan.sum(axis=0), DEMAND, atol=1e-3)  # cols = demand


def test_sinkhorn_recovers_the_exact_lp_as_eps_shrinks() -> None:
    res = sinkhorn(COST, SUPPLY, DEMAND, eps=0.01, iters=4000)
    assert res.transport_cost == pytest.approx(18.0, abs=0.1)  # the glpk LP optimum


def test_kantorovich_rubinstein_strong_duality() -> None:
    coarse = sinkhorn(COST, SUPPLY, DEMAND, eps=0.1, iters=2000).duality_gap
    fine = sinkhorn(COST, SUPPLY, DEMAND, eps=0.01, iters=4000).duality_gap
    assert fine < coarse  # the entropic gap shrinks toward 0 as eps -> 0 (exact in the limit)
    assert fine < 0.15  # already tight


def test_optimal_dispatch_beats_the_naive_baselines() -> None:
    city = MarketplaceMatching.synthetic_city(n_zones=10, seed=1)
    _, coverage = city.nearest_local()
    assert city.optimal().transport_cost < city.nearest_reroute_cost()  # OT cheaper than myopic
    assert coverage < 1.0  # local-only dispatch strands some demand


def test_surge_rebalancing_cuts_imbalance() -> None:
    before, after = MarketplaceMatching.synthetic_city(n_zones=10, seed=2).surge_rebalance()
    assert after < before  # drivers responding to surge reduce supply-demand imbalance


def test_report_renders_the_kantorovich_story() -> None:
    text = marketplace_report(MarketplaceMatching.synthetic_city(seed=3))
    assert "Kantorovich" in text  # the lineage
    assert "surge" in text  # the dual output
