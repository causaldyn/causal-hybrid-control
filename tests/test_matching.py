"""Kantorovich OT matching: exact-LP recovery, strong duality, and dispatch beating naive."""

from __future__ import annotations

import logging
import math

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special
from scipy.optimize import linprog

from chc.matching import (
    MarketplaceMatching,
    marketplace_report,
    shadow_price_effect,
    shadow_price_interval,
    sinkhorn,
)

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


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"eps": math.nan}, "eps must be positive and finite"),
        ({"eps": -0.5}, "eps must be positive and finite"),
        ({"eps": 0.0}, "eps must be positive and finite"),
        ({"eps": math.inf}, "eps must be positive and finite"),
        ({"tol": math.nan}, "tol must be positive and finite"),
        ({"tol": math.inf}, "tol must be positive and finite"),
        ({"cost": COST.at[0, 1].set(jnp.nan)}, "cost must be finite"),
        ({"supply": SUPPLY.at[0].set(jnp.nan)}, "supply must be finite"),
        ({"supply": SUPPLY.at[0].set(jnp.inf), "demand": DEMAND.at[0].set(jnp.inf)}, "finite"),
        ({"supply": SUPPLY.at[0].set(-4.0), "demand": DEMAND.at[0].set(-2.0)}, "negative"),
    ],
)
def test_sinkhorn_refuses_a_problem_that_is_no_number(change: dict, match: str) -> None:
    """At ``eps = -0.5`` the solve returned the coupling that maximises the cost, its marginals
    met to 1e-16; a nan or infinite ``eps`` or mass gave a plan of nan, and a nan ``tol`` warned on
    every call."""
    problem = {"cost": COST, "supply": SUPPLY, "demand": DEMAND}
    with pytest.raises(ValueError, match=match):
        sinkhorn(**{**problem, **change})


# ---- the global effect of an experiment on a matching, read off its rents ----


def _market(seed: int, rows: int = 4, columns: int = 3):
    """The lab's random market: row types and zones on a plane, a value that falls with distance,
    and a treatment direction scaled to at most 1 in any cell."""
    rng = np.random.default_rng(seed)
    supply = rng.dirichlet(2.0 * np.ones(rows))
    demand = rng.dirichlet(2.0 * np.ones(columns))
    zones = rng.uniform(-1.0, 1.0, (columns, 2))
    types = rng.uniform(-1.0, 1.0, (rows, 2))
    distance = np.linalg.norm(types[:, None, :] - zones[None, :, :], axis=2)
    values = (
        rng.normal(0.0, 0.5, columns)[None, :] - distance + 0.3 * rng.normal(size=(rows, columns))
    )
    direction = rng.normal(size=(rows, columns))
    return values, supply, demand, direction / np.max(np.abs(direction))


def _split(values, supply, demand, treatment, share):
    """The experiment's market with every row type split into a treated and a control copy."""
    rows = values.shape[0]
    return (
        -np.vstack([values + treatment, values]),
        np.concatenate([share * supply, (1.0 - share) * supply]),
        demand,
        np.concatenate([np.ones(rows, dtype=bool), np.zeros(rows, dtype=bool)]),
    )


def _solve(values, supply, demand, eps):
    """``max <P, values> - eps KL(P | supply x demand)``, each row's plan per unit of its mass, and
    the column prices, by a log-domain Sinkhorn in NumPy run until its potentials stop moving: an
    engine apart from :func:`sinkhorn`."""
    log_kernel = values / eps + np.log(supply)[:, None] + np.log(demand)[None, :]
    f, g = np.zeros(len(supply)), np.zeros(len(demand))
    for _ in range(200_000):
        f = np.log(supply) - special.logsumexp(log_kernel + g[None, :], axis=1)
        g_next = np.log(demand) - special.logsumexp(log_kernel + f[:, None], axis=0)
        done = np.max(np.abs(g_next - g)) < 1e-15
        g = g_next
        if done:
            break
    f = np.log(supply) - special.logsumexp(log_kernel + g[None, :], axis=1)
    plan = np.exp(log_kernel + f[:, None] + g[None, :])
    welfare = np.sum(plan * values) - eps * np.sum(plan * np.log(plan / np.outer(supply, demand)))
    return float(welfare), plan / supply[:, None], -eps * g


def _welfare(values, supply, demand, eps):
    return _solve(values, supply, demand, eps)[0]


def _split_welfare(values, supply, demand, treatment, share, eps):
    cost, masses, capacities, _ = _split(values, supply, demand, treatment, share)
    return _welfare(-cost, masses, capacities, eps)


def _global_effect(values, supply, demand, treatment, eps):
    return _welfare(values + treatment, supply, demand, eps) - _welfare(values, supply, demand, eps)


def _slope(scales, errors) -> float:
    return float(np.polyfit(np.log(scales), np.log(np.abs(errors)), 1)[0])


@pytest.mark.parametrize(("seed", "mass"), [(0, 1.0), (1, 1.0), (2, 2.7)])
def test_the_effect_is_the_slope_of_welfare_in_the_treated_share(seed: int, mass: float) -> None:
    """Each row's rent is the derivative of welfare in its own mass, so the treated-minus-control
    difference of rents is ``f'(p)``: checked against a five-point difference of the welfare an
    independent solver gives at nearby shares, on markets of total mass 1 and 2.7."""
    values, supply, demand, direction = _market(seed)
    supply, demand = mass * supply, mass * demand
    eps, treatment, step = 0.3, 0.8 * direction, 1e-3
    for share in (0.2, 0.5, 0.8):
        estimate = shadow_price_effect(
            *_split(values, supply, demand, treatment, share),
            eps=eps,
            iters=5000,
            tol=1e-12,
        )
        near = [
            _split_welfare(values, supply, demand, treatment, share + k * step, eps)
            for k in (-2, -1, 1, 2)
        ]
        slope = (near[0] - 8.0 * near[1] + 8.0 * near[2] - near[3]) / (12.0 * step)
        assert estimate.effect == pytest.approx(slope, rel=1e-7, abs=1e-10)
        assert estimate.treated_share == pytest.approx(share)


def test_a_row_outside_the_experiment_is_priced_and_not_counted() -> None:
    """An idle pool of supply, one more row that nobody randomises, and an outside option for the
    rows, one more column: the effect is still the slope of welfare in the treated share."""
    values, supply, demand, direction = _market(3)
    values = np.column_stack([np.vstack([values, np.zeros(3)]), np.zeros(5)])
    values[-1, -1] = -5.0  # idle supply never takes the outside option
    supply = np.append(supply, 0.3)
    demand = np.append(demand, 0.3)
    treatment = np.column_stack([np.vstack([0.8 * direction, np.zeros(3)]), np.zeros(5)])
    eps, share, step = 0.3, 0.3, 1e-3

    def market(p):
        cost = -np.vstack([values[:4] + treatment[:4], values[:4], values[4:]])
        masses = np.concatenate([p * supply[:4], (1.0 - p) * supply[:4], supply[4:]])
        return cost, masses

    cost, masses = market(share)
    treated = np.arange(9) < 4
    estimate = shadow_price_effect(
        cost,
        masses,
        demand,
        treated,
        eps=eps,
        randomised=np.arange(9) < 8,
        iters=5000,
        tol=1e-12,
    )
    near = [
        _welfare(-market(share + k * step)[0], market(share + k * step)[1], demand, eps)
        for k in (-2, -1, 1, 2)
    ]
    slope = (near[0] - 8.0 * near[1] + 8.0 * near[2] - near[3]) / (12.0 * step)
    assert estimate.effect == pytest.approx(slope, rel=1e-7)
    counted = shadow_price_effect(cost, masses, demand, treated, eps=eps, iters=5000, tol=1e-12)
    assert abs(counted.effect - estimate.effect) > 0.01


def test_the_other_side_is_randomised_by_transposing_the_market() -> None:
    """Treating some of the columns: their prices are their rents, and the transposed market reads
    them as the slope of welfare in the columns' treated share."""
    values, supply, demand, direction = _market(4, rows=3, columns=4)
    eps, share, step, treatment = 0.3, 0.4, 1e-3, 0.8 * direction

    def welfare(p):
        split = np.hstack([values + treatment, values])
        return _welfare(split, supply, np.concatenate([p * demand, (1.0 - p) * demand]), eps)

    estimate = shadow_price_effect(
        -np.hstack([values + treatment, values]).T,
        np.concatenate([share * demand, (1.0 - share) * demand]),
        supply,
        np.arange(8) < 4,
        eps=eps,
        iters=5000,
        tol=1e-12,
    )
    near = [welfare(share + k * step) for k in (-2, -1, 1, 2)]
    assert estimate.effect == pytest.approx(
        (near[0] - 8.0 * near[1] + 8.0 * near[2] - near[3]) / (12.0 * step), rel=1e-7
    )


def test_the_standard_error_is_a_spread_within_each_arm_in_the_effects_units() -> None:
    """Adding the same amount to all of the treated rows' values moves no plan and no price: the
    effect moves by the randomised mass times it, and the standard error, a spread within each arm,
    does not move. Scaling every mass scales welfare, so the effect, its standard error and the
    curvature scale with it, and the prices do not move."""
    worth, lift, masses, capacities, _ = _zone_market(40)
    treated = np.append(np.random.default_rng(1).random(40) < 0.5, False)

    def read(shift: float = 0.0, scale: float = 1.0):
        return shadow_price_effect(
            -(worth + (lift + shift) * treated[:, None]),
            scale * masses,
            scale * capacities,
            treated,
            eps=0.3,
            randomised=np.arange(41) < 40,
            iters=5000,
            tol=1e-12,
        )

    base, shifted, scaled = read(), read(shift=0.7), read(scale=2.7)
    assert shifted.effect - base.effect == pytest.approx(0.7, rel=1e-9)
    assert shifted.standard_error == pytest.approx(base.standard_error, rel=1e-9)
    assert shifted.curvature == pytest.approx(base.curvature, rel=1e-9)
    for field in ("effect", "standard_error", "curvature"):
        assert getattr(scaled, field) == pytest.approx(2.7 * getattr(base, field), rel=1e-9)
    assert scaled.nu_hat == pytest.approx(base.nu_hat, rel=1e-9)


def test_post_stratified_the_effect_does_not_see_how_the_coins_mixed_each_arm() -> None:
    """With ``strata`` the effect weights each stratum's treated-minus-control difference by the
    stratum's mass. Adding an amount to every row of one stratum, treated or not, moves no plan and
    no price, so it moves neither that effect, nor its standard error, nor the interval at
    ``eps = 0``; the plain difference moves by the amount times the stratum's imbalance between the
    arms. Adding an amount to every treated row moves the effect by the randomised mass times it,
    and renaming the strata changes nothing."""
    worth, lift, masses, capacities, kind = _zone_market(80)
    treated = np.append(np.random.default_rng(2).random(80) < 0.5, False)
    randomised = np.arange(81) < 80
    strata = np.append(kind, -1)
    in_stratum = np.append(kind == 0, False)
    values = worth + lift * treated[:, None]
    moved = values + 0.9 * in_stratum[:, None]

    def read(values, labels):
        return shadow_price_effect(
            -values,
            masses,
            capacities,
            treated,
            eps=0.3,
            randomised=randomised,
            strata=labels,
            iters=5000,
            tol=1e-12,
        )

    def interval(values):
        return shadow_price_interval(
            -values, masses, capacities, treated, randomised=randomised, strata=strata
        )

    imbalance = [
        masses[arm & in_stratum].sum() / masses[arm].sum()
        for arm in (treated, randomised & ~treated)
    ]
    assert imbalance[0] - imbalance[1] > 0.05
    plain = read(moved, None).effect - read(values, None).effect
    assert plain == pytest.approx(0.9 * (imbalance[0] - imbalance[1]), rel=1e-9)
    base, after = read(values, strata), read(moved, strata)
    assert base.treated_share == pytest.approx(masses[treated].sum() / masses[randomised].sum())
    assert after.effect == pytest.approx(base.effect, abs=1e-12)
    assert after.standard_error == pytest.approx(base.standard_error, rel=1e-9)
    assert interval(moved) == pytest.approx(interval(values), abs=1e-7)
    lifted = read(values + 0.7 * treated[:, None], strata)
    assert lifted.effect - base.effect == pytest.approx(0.7, rel=1e-9)
    renamed = read(values, np.append(3 - kind, -1))
    for field in ("effect", "standard_error", "curvature", "nu_hat"):
        assert getattr(renamed, field) == pytest.approx(getattr(base, field), rel=1e-9)


@pytest.mark.parametrize("seed", [0, 5])
def test_the_effect_falls_with_the_share_and_brackets_the_global_effect(
    seed: int,
) -> None:
    """Welfare is concave in the treated share, so its slope falls, and the slopes at no row and
    every row treated bracket the global effect, even for a treatment far outside the second-order
    regime."""
    values, supply, demand, direction = _market(seed)
    eps, treatment = 0.3, 2.0 * direction
    shares = np.concatenate([[1e-6], np.linspace(0.1, 0.9, 9), [1.0 - 1e-6]])
    effects = [
        shadow_price_effect(
            *_split(values, supply, demand, treatment, share),
            eps=eps,
            iters=5000,
            tol=1e-12,
        ).effect
        for share in shares
    ]
    assert np.all(np.diff(effects) <= 1e-9)
    assert effects[-1] < _global_effect(values, supply, demand, treatment, eps) < effects[0]


def test_the_effect_misses_the_global_effect_at_second_order_and_at_one_half_at_third() -> None:
    """``f'(p) - (f(1) - f(0)) = (1/2 - p) K delta^2 / eps + O(delta^3)``, with ``K = s' C^+ s``
    read off the untreated market: ``s = sum_i a_i J(pi_i) d_i`` and ``C = sum_i a_i J(pi_i)``.
    The naive difference misses at first order."""
    values, supply, demand, direction = _market(7)
    eps, share = 0.3, 0.2
    _, plan, _ = _solve(values, supply, demand, eps)
    jacobians = [np.diag(row) - np.outer(row, row) for row in plan]
    push = sum(a * j @ d for a, j, d in zip(supply, jacobians, direction, strict=True))
    hessian = sum(a * j for a, j in zip(supply, jacobians, strict=True))
    k_r = float(push[:-1] @ np.linalg.solve(hessian[:-1, :-1], push[:-1]))

    scales = np.array([0.1, 0.05, 0.025])
    off_half, at_half, naive = [], [], []
    for scale in scales:
        truth = _global_effect(values, supply, demand, scale * direction, eps)
        estimate = shadow_price_effect(
            *_split(values, supply, demand, scale * direction, share),
            eps=eps,
            iters=5000,
            tol=1e-12,
        )
        off_half.append(estimate.effect - truth)
        naive.append(estimate.naive - truth)
        at_half.append(
            shadow_price_effect(
                *_split(values, supply, demand, scale * direction, 0.5),
                eps=eps,
                iters=5000,
                tol=1e-12,
            ).effect
            - truth
        )
    assert off_half[-1] / ((0.5 - share) * k_r * scales[-1] ** 2 / eps) == pytest.approx(
        1.0, abs=0.01
    )
    assert _slope(scales, off_half) == pytest.approx(2.0, abs=0.05)
    assert _slope(scales, at_half) == pytest.approx(3.0, abs=0.1)
    assert _slope(scales, naive) == pytest.approx(1.0, abs=0.05)


def test_the_naive_difference_misses_by_the_displacement_it_prices() -> None:
    """To first order the naive difference misses by ``(1/eps) sum_i a_i Cov_pi_i(beta, d_i)``:
    each row's treatment, priced at the untreated market's prices under its own plan."""
    values, supply, demand, direction = _market(7)
    eps, scale = 0.3, 0.01
    _, plan, prices = _solve(values, supply, demand, eps)
    covariance = [
        row @ (prices * d) - (row @ prices) * (row @ d)
        for row, d in zip(plan, direction, strict=True)
    ]
    predicted = scale * float(supply @ np.array(covariance)) / eps
    estimate = shadow_price_effect(
        *_split(values, supply, demand, scale * direction, 0.2),
        eps=eps,
        iters=5000,
        tol=1e-12,
    )
    truth = _global_effect(values, supply, demand, scale * direction, eps)
    assert (estimate.naive - truth) / predicted == pytest.approx(1.0, abs=0.03)


@pytest.mark.parametrize("seed", [7, 8])
def test_the_gauss_legendre_pair_leaves_a_fifth_order_bias(seed: int) -> None:
    """Averaged over ``p = 1/2 -+ 1/(2 sqrt 3)``, the effect's miss has zero coefficients through
    the fourth order, since each order's coefficient is a polynomial in ``p`` of one degree less,
    with mean zero on ``[0, 1]``."""
    values, supply, demand, direction = _market(seed)
    eps = 0.3
    pair = (0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0))
    misses = {"pair": [], "half": []}
    for scale in (0.2, 0.1):
        truth = _global_effect(values, supply, demand, scale * direction, eps)

        def effect(share, scale=scale):
            return shadow_price_effect(
                *_split(values, supply, demand, scale * direction, share),
                eps=eps,
                iters=5000,
                tol=1e-12,
            ).effect

        misses["pair"].append(0.5 * (effect(pair[0]) + effect(pair[1])) - truth)
        misses["half"].append(effect(0.5) - truth)
    assert np.log2(misses["pair"][0] / misses["pair"][1]) == pytest.approx(5.0, abs=0.15)
    assert np.log2(misses["half"][0] / misses["half"][1]) == pytest.approx(3.0, abs=0.2)


def _zone_market(units: int, seed: int = 2026):
    """``units`` request rows of four types against three zones, an outside option for them (the
    last column), and a row of idle supply (the last row): the shape of a platform's experiment,
    many randomised rows against few columns. Returns each unit's type too."""
    rng = np.random.default_rng(seed)
    values, supply, demand, direction = _market(0)
    kind = rng.choice(4, size=units, p=supply)
    worth = np.zeros((units + 1, 4))
    worth[:units, :3] = 1.0 + values[kind] + 0.25 * rng.normal(size=(units, 3))
    lift = np.zeros_like(worth)
    lift[:units, :3] = 0.4 * direction[kind] + 0.2 * rng.normal(size=units)[:, None]
    masses = np.append(np.full(units, 1.0 / units), 0.25)
    capacities = np.append(demand, 0.25)
    return worth, lift, masses, capacities, kind


def test_over_assignments_the_error_matches_the_spread_and_the_curvature_its_value() -> None:
    """A hundred assignments of 320 units at ``p = 0.2``: the effect scatters as its standard error
    says, and its mean lands on the global effect. The curvature, noisy in any one experiment, lands
    on ``-f''(p)`` on average, where the plug-in ``eps g' C^+ g`` read it 3.6 times too large, all
    of the excess the assignment's noise in ``g``. Post-stratified by the units' types, all of that
    still holds, and the effect scatters about half as much."""
    worth, lift, masses, capacities, kind = _zone_market(320)
    units, eps, share = 320, 0.3, 0.2
    randomised = np.arange(units + 1) < units
    reads = {"plain": [], "post": []}
    for replicate in range(100):
        rng = np.random.default_rng(7000 + replicate)
        treated = (rng.random(units + 1) < share) & randomised
        for name, strata in (("plain", None), ("post", np.append(kind, -1))):
            reads[name].append(
                shadow_price_effect(
                    -(worth + lift * treated[:, None]),
                    masses,
                    capacities,
                    treated,
                    eps=eps,
                    randomised=randomised,
                    strata=strata,
                    iters=5000,
                    tol=1e-10,
                )
            )

    def split_welfare(p):
        rows = np.vstack([worth[:units] + lift[:units], worth[:units], worth[units:]])
        split = np.concatenate([p * masses[:units], (1.0 - p) * masses[:units], masses[units:]])
        return _welfare(rows, split, capacities, eps)

    step = 1e-2
    near = [split_welfare(share + k * step) for k in (-2, -1, 0, 1, 2)]
    fluid = -(-near[0] + 16.0 * near[1] - 30.0 * near[2] + 16.0 * near[3] - near[4]) / (
        12 * step**2
    )
    truth = _welfare(worth + lift, masses, capacities, eps) - _welfare(
        worth, masses, capacities, eps
    )

    spreads = {}
    for name, estimates in reads.items():
        effects = [estimate.effect for estimate in estimates]
        spreads[name] = float(np.std(effects, ddof=1))
        errors = [estimate.standard_error for estimate in estimates]
        assert 0.8 < spreads[name] / float(np.mean(errors)) < 1.25
        assert abs(np.mean(effects) - truth) < 3.0 * spreads[name] / np.sqrt(len(effects))
        assert 0.5 < float(np.mean([estimate.curvature for estimate in estimates])) / fluid < 1.6
    assert spreads["post"] < 0.7 * spreads["plain"]


def test_near_the_lp_limit_the_second_order_claim_is_refused(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Where the treatment moves the prices by more than ``eps`` across the shares, the bias is
    no longer second order, so the estimate says so instead of pricing it. On one market, the
    prices move 1.2 ``eps`` at ``eps = 0.5`` and 0.6 at ``eps = 1``."""
    values, supply, demand, direction = _market(7)
    market = _split(values, supply, demand, 0.8 * direction, 0.2)
    with caplog.at_level(logging.WARNING, logger="chc.matching"):
        near_lp = shadow_price_effect(*market, eps=0.5, iters=20_000, tol=1e-9)
    assert near_lp.marginal_residual <= 1e-9
    assert 1.0 < near_lp.nu_hat < 1.5
    assert near_lp.second_order_bias is None
    assert "LP limit" in near_lp.reason
    (record,) = [
        r for r in caplog.records if getattr(r, "chc_event", None) == "shadow_price_effect"
    ]
    assert getattr(record, "nu_hat", None) == near_lp.nu_hat

    smooth = shadow_price_effect(*market, eps=1.0, iters=5000, tol=1e-9)
    assert 0.5 < smooth.nu_hat < 1.0
    assert smooth.reason == ""
    assert smooth.second_order_bias == pytest.approx((0.5 - 0.2) * smooth.curvature)


def test_a_solve_that_stops_short_makes_no_second_order_claim() -> None:
    values, supply, demand, direction = _market(7)
    estimate = shadow_price_effect(
        *_split(values, supply, demand, 0.3 * direction, 0.2),
        eps=0.3,
        iters=2,
        tol=1e-9,
    )
    assert estimate.marginal_residual > 1e-9
    assert estimate.second_order_bias is None
    assert "raise iters" in estimate.reason


def _lp_welfare(values, supply, demand) -> float:
    rows, columns = values.shape
    marginals = np.vstack(
        [
            np.kron(np.eye(rows), np.ones(columns)),
            np.kron(np.ones(rows), np.eye(columns)),
        ]
    )
    result = linprog(
        -values.ravel(),
        A_eq=marginals,
        b_eq=np.concatenate([supply, demand]),
        method="highs",
    )
    return -float(result.fun)


def test_at_eps_zero_the_effect_is_the_superdifferential_of_welfare_in_the_share() -> None:
    """The exact LP's welfare is concave and piecewise linear in the treated share. At a kink the
    rents range over ``[f'(p+), f'(p-)]``, and the interval is that range; between kinks it is a
    point, the slope."""
    values, supply, demand, direction = _market(300)
    treatment = 0.5 * direction

    def welfare(p):
        cost, masses, capacities, _ = _split(values, supply, demand, treatment, p)
        return _lp_welfare(-cost, masses, capacities)

    grid = np.linspace(0.05, 0.95, 19)
    step = 1e-7
    right = [(welfare(p + step) - welfare(p)) / step for p in grid]
    kinks = [k for k in range(len(grid) - 1) if right[k] - right[k + 1] > 1e-6]
    assert kinks
    k = kinks[0]
    # welfare is linear on either side of an isolated kink: the kink is where the two lines meet
    kink = (
        welfare(grid[k + 1]) - welfare(grid[k]) + right[k] * grid[k] - right[k + 1] * grid[k + 1]
    ) / (right[k] - right[k + 1])
    low, high = shadow_price_interval(*_split(values, supply, demand, treatment, kink))
    assert low == pytest.approx(right[k + 1], abs=1e-6)
    assert high == pytest.approx(right[k], abs=1e-6)
    assert high - low > 1e-3

    between = 0.5 * (grid[k + 1] + kink) if grid[k + 1] - kink > 0.02 else 0.5 * (grid[k] + kink)
    low, high = shadow_price_interval(*_split(values, supply, demand, treatment, between))
    slope = (welfare(between + step) - welfare(between)) / step
    assert high - low < 1e-9
    assert low == pytest.approx(slope, abs=1e-6)


def test_a_market_of_infinite_mass_is_refused() -> None:
    """``inf - inf`` is nan, so infinite masses passed the test that compares the totals."""
    values, supply, demand, direction = _market(0)
    cost, masses, capacities, treated = _split(values, supply, demand, 0.5 * direction, 0.5)
    masses[0], capacities[0] = np.inf, np.inf
    with pytest.raises(ValueError, match="positive and finite"):
        shadow_price_interval(cost, masses, capacities, treated)
    with pytest.raises(ValueError, match="positive and finite"):
        shadow_price_effect(cost, masses, capacities, treated, eps=0.3)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"eps": 0.0}, "see shadow_price_interval"),
        ({"eps": math.nan}, "positive and finite; at eps = 0"),
        ({"eps": math.inf}, "positive and finite; at eps = 0"),
        ({"demand": np.array([0.5, 0.3, 0.1])}, "equal totals"),
        ({"treated": np.array([1, 1, 0, 0, 1, 0, 0, 0])}, "one bool per row"),
        ({"treated": np.arange(8) < 1}, "at least two treated rows"),
        (
            {"randomised": np.arange(8) < 5},
            "at least two treated rows and two control rows",
        ),
        ({"cost": np.zeros((8, 2))}, "cost must be"),
        ({"strata": np.zeros(7, dtype=int)}, "one label per row"),
        ({"strata": np.array([0, 0, 0, 1, 0, 0, 0, 1])}, "two control rows in stratum 1"),
    ],
)
def test_an_experiment_it_cannot_read_is_refused(change: dict, message: str) -> None:
    values, supply, demand, direction = _market(0)
    cost, masses, capacities, treated = _split(values, supply, demand, 0.5 * direction, 0.5)
    arguments = {"cost": cost, "demand": capacities, "treated": treated, "eps": 0.3}
    arguments.update(change)
    with pytest.raises(ValueError, match=message):
        shadow_price_effect(
            arguments["cost"],
            masses,
            arguments["demand"],
            arguments["treated"],
            eps=arguments["eps"],
            randomised=arguments.get("randomised"),
            strata=arguments.get("strata"),
        )
