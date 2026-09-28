"""Kantorovich OT matching: exact-LP recovery, strong duality, and dispatch beating naive."""

from __future__ import annotations

import logging

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


def _sinkhorn_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if getattr(r, "chc_event", None) == "sinkhorn"]


def test_sinkhorn_reports_and_warns_when_its_iterations_stop_short(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # At eps = 0.01 this city's residual sits on a plateau near 1e-2 for the first thousand
    # iterations; the result used to hand back that plan, and prices read off it, with nothing said.
    city = MarketplaceMatching.synthetic_city(n_zones=10, seed=1)
    with caplog.at_level(logging.WARNING, logger="chc.matching"):
        res = sinkhorn(city.cost, city.supply, city.demand, eps=0.01, iters=1000)
    rows = jnp.sum(jnp.abs(res.plan.sum(axis=1) - city.supply))
    cols = jnp.sum(jnp.abs(res.plan.sum(axis=0) - city.demand))
    assert res.marginal_residual == pytest.approx(float((rows + cols) / city.supply.sum()))
    assert res.marginal_residual > 1e-3
    (record,) = _sinkhorn_warnings(caplog)
    assert record.levelno == logging.WARNING
    assert getattr(record, "marginal_residual", None) == res.marginal_residual


def test_sinkhorn_is_quiet_once_its_marginals_hold(caplog: pytest.LogCaptureFixture) -> None:
    city = MarketplaceMatching.synthetic_city(n_zones=10, seed=1)
    with caplog.at_level(logging.WARNING, logger="chc.matching"):
        res = sinkhorn(city.cost, city.supply, city.demand, eps=0.01, iters=16000)
    assert res.marginal_residual < 1e-12
    assert not _sinkhorn_warnings(caplog)


def test_sinkhorn_refuses_a_tolerance_that_cannot_be_met() -> None:
    with pytest.raises(ValueError, match="tol must be positive"):
        sinkhorn(COST, SUPPLY, DEMAND, tol=0.0)
