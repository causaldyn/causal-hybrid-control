"""Offline causal decision under equilibrium interference: the logging policy is confounded and
SUTVA fails, so the predictive (MOPO) and naive-causal planners capture little of the oracle's lift,
while the CHC allocation, which plans through an equilibrium fitted from the logs, recovers most of
it.

The thresholds are read on one market, ``SharedStateMarket(seed=0)``, among the hardest for the
SUTVA planners. Over markets 0-29 on logs of 400 blocks, the CHC allocation had the least regret of
the three on every one in float64 and on 29 in float32; the naive plan won a median of 0.35 of the
exact oracle's lift in float64, up to 0.79.
"""

import dataclasses
import functools
import importlib
import inspect
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, stats

from chc.games import (
    exact_congestion_allocation,
    fixed_point,
    per_driver_congestion_allocation,
    softmax_congestion_equilibrium,
)
from chc.marketplace import (
    ExposureResponse,
    FittedEquilibrium,
    SharedStateMarket,
    _attract_se,
    _driver_jacobians,
    _expected_completions,
    _fitted_completions,
    _fitted_drivers,
    _fitted_step,
    _sandwich,
    _zone_slope,
    calibrate_budget_scale,
    calibrate_naive_causal,
    calibrate_predictive,
    calibrate_shared_state,
    dither_check,
    dither_size,
    fit_equilibrium,
    interference_bias,
    pessimistic_equilibrium_allocation,
    sutva_allocation,
)

# Public as jax.enable_x64 from jax 0.8.0; the floor, 0.4.30, has only jax.experimental.enable_x64,
# which later releases deprecate.
if hasattr(jax, "enable_x64"):
    enable_x64 = jax.enable_x64
else:  # pragma: no cover - the lowest supported jax
    enable_x64 = importlib.import_module("jax.experimental").enable_x64


@pytest.fixture(scope="module")
def env() -> dict:
    market = SharedStateMarket(seed=0)
    logs = market.generate_logs(400, jax.random.key(1))
    zero = jnp.zeros(market.n_zones)
    oracle_value = market.value(market.oracle_allocation())
    base_value = market.value(zero)
    return {
        "market": market,
        "logs": logs,
        "oracle": oracle_value,
        "base": base_value,
        "fitted": fit_equilibrium(logs),
    }


def _rerun(market: SharedStateMarket, logs: dict, u: jax.Array) -> dict:
    """The logged blocks under the incentives ``u``: each keeps its demand, the attractiveness that
    moves with it and its completions noise, and its drivers settle anew."""
    base_demand, base_attract = market._base()
    attract = base_attract * logs["demand"] / base_demand
    settle = jax.vmap(market._equilibrium)
    noise = logs["y"] - jnp.minimum(logs["demand"], settle(attract, logs["u"]))
    y = jnp.minimum(logs["demand"], settle(attract, u)) + noise
    return {"u": u, "demand": logs["demand"], "y": y}


def test_the_logging_policy_s_chase_of_demand_confounds_the_responses(env: dict) -> None:
    """The reference is a response no log gives: the naive-causal regression on the logged blocks
    rerun with each block's incentives another block's, which no longer follow its demand. Against
    it the predictive response reads high, the logging policy having chased demand. Adjusting each
    zone for its own demand removes part of that, not all: the policy splits one budget by a
    softmax of every zone's demand, so a zone's incentive also follows the others' demand, which
    draws drivers from it too. Rerun with each zone's incentive chasing its own logged demand and
    another block's elsewhere, where adjusting for its own demand is complete, the naive-causal
    response still reads above that rerun's. The test was the responses' correlation with demand,
    the naive-causal one's near zero, which is no test: the randomised response follows demand, and
    over seeds 0-29 the naive-causal one read 0.2 or more in size on 15.

    Over seeds 0-29 in float64 the predictive excess read 0.30 to 0.54, the share of it the
    adjustment keeps 0.09 to 0.72, the excess over the complete adjustment 0.13 to 0.30 and the
    randomised response's correlation with demand 0.39 to 0.87. Each bound is their mean less or
    plus three SDs, rounded outward."""
    market, logs = env["market"], env["logs"]
    demand, _ = market._base()
    total = logs["aggregate"][:, :1]

    def chase(chased: jax.Array) -> jax.Array:
        """generate_logs's logging policy: the block's total, split by a softmax of demand."""
        return total * jax.nn.softmax(market.confounding * chased, axis=1)

    # the twelve-term sum and two products round to at most about 8 epsilons; twice that
    rtol = 16 * float(jnp.finfo(logs["u"].dtype).eps)
    np.testing.assert_allclose(chase(logs["demand"]), logs["u"], rtol=rtol, atol=0.0)
    order = np.random.default_rng(0).permutation(logs["u"].shape[0])
    randomised = calibrate_naive_causal(_rerun(market, logs, logs["u"][order])).marginal
    complete = []
    for zone in range(market.n_zones):
        chased = logs["demand"][order].at[:, zone].set(logs["demand"][:, zone])
        complete.append(calibrate_naive_causal(_rerun(market, logs, chase(chased))).marginal[zone])
    predictive = calibrate_predictive(logs).marginal
    naive = calibrate_naive_causal(logs).marginal
    excess = float(jnp.mean(predictive - randomised))
    assert excess > 0.25
    assert float(jnp.mean(naive - randomised)) < 0.88 * excess
    assert float(jnp.mean(naive - jnp.stack(complete))) > 0.07
    assert float(jnp.corrcoef(randomised, demand)[0, 1]) > 0.28


def test_sutva_planners_capture_almost_none_of_the_oracle_lift(env: dict) -> None:
    market, base, oracle = env["market"], env["base"], env["oracle"]
    pred_alloc = sutva_allocation(market, calibrate_predictive(env["logs"]), radius=1.0)
    naive_alloc = sutva_allocation(market, calibrate_naive_causal(env["logs"]), radius=1.0)
    lift = oracle - base  # the completions the oracle wins over doing nothing
    # SUTVA over-allocates into zones whose drivers only cannibalise: on this market both plans
    # win less than no incentive; the naive one stays under 0.3 of the lift on half of markets 0-29
    assert market.value(pred_alloc) - base < 0.3 * lift
    assert market.value(naive_alloc) - base < 0.3 * lift


def test_chc_recovers_the_oracle_where_baselines_do_not(env: dict) -> None:
    market, oracle = env["market"], env["oracle"]
    shared = calibrate_shared_state(env["logs"])
    chc = pessimistic_equilibrium_allocation(market, shared, radius=1.0, equilibrium=env["fitted"])
    naive = sutva_allocation(market, calibrate_naive_causal(env["logs"]), radius=1.0)
    pred = sutva_allocation(market, calibrate_predictive(env["logs"]), radius=1.0)
    # the least regret of the three, as on every one of markets 0-29 in float64
    assert market.value(chc) > max(market.value(naive), market.value(pred))
    # through an equilibrium fitted from the logs: 0.102 short in float32, 0.087 in float64
    assert oracle - market.value(chc) < 0.3
    assert oracle - market.value(naive) > 0.8  # SUTVA leaves large regret on the table
    assert oracle - market.value(pred) > 0.8


def test_the_planner_refuses_to_plan_without_a_fitted_equilibrium(env: dict) -> None:
    """Handed no equilibrium, the planner went through the simulator's own, which no log supplies,
    and planned as close to the oracle from a response of zero as from the adjusted one: the
    comparison measured knowing the equilibrium. The fit is a keyword with no default, so its
    signature says so and a type checker flags the omission: annotated optional, it was not."""
    market = env["market"]
    parameter = inspect.signature(pessimistic_equilibrium_allocation).parameters["equilibrium"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    assert parameter.annotation == "FittedEquilibrium"
    with pytest.raises(TypeError, match="required keyword-only argument: 'equilibrium'"):
        pessimistic_equilibrium_allocation(market, calibrate_shared_state(env["logs"]))  # type: ignore[call-arg]


@pytest.mark.parametrize("radius", [math.nan, math.inf, -5.0])
def test_both_planners_refuse_a_radius_that_is_no_count_of_standard_errors(
    env: dict, radius: float
) -> None:
    """A nan radius made the SUTVA plan nan and sent the equilibrium planner back to the even split
    without a word (10.94 rides where radius 1 plans 11.91); a negative one turned the pessimism
    into a bonus."""
    market, logs = env["market"], env["logs"]
    with pytest.raises(ValueError, match="radius is"):
        sutva_allocation(market, calibrate_naive_causal(logs), radius)
    with pytest.raises(ValueError, match="radius is"):
        pessimistic_equilibrium_allocation(
            market, calibrate_shared_state(logs), radius=radius, equilibrium=env["fitted"]
        )


def test_the_fitted_equilibrium_recovers_the_market_s_parameters(env: dict) -> None:
    """Drivers' sensitivity 2.5, crowding 2 and 12 drivers, read off 400 blocks whose completions
    carry noise of SD 0.02: the fit lands within 0.014, 0.004 and 0.04 of them in either precision,
    and its residual is the noise's."""
    fitted = env["fitted"]
    assert fitted.converged
    assert fitted.beta == pytest.approx(2.5, abs=0.05)
    assert fitted.congestion == pytest.approx(2.0, abs=0.05)
    assert fitted.mass == pytest.approx(12.0, abs=0.1)
    assert fitted.rmse == pytest.approx(0.02, rel=0.1)


def _relative_offset(fitted, logs: dict) -> tuple[float, float]:
    """Bates and Watts' (1981) relative offset of the least-squares fit at ``fitted``'s parameters,
    and the least distance of a block's drivers from its demand, where the residual has a kink.

    The Jacobian of the drivers in the fit's parameters, the attractiveness and the logarithms of
    the sensitivity, the penalty and the total, by reverse mode through each block's equilibrium;
    a zone past its demand completes its demand, and its row is zero."""
    u, demand, y = (jnp.asarray(logs[name]) for name in ("u", "demand", "y"))
    n = u.shape[1]

    def one_block(q: jax.Array, d: jax.Array, v: jax.Array) -> jax.Array:
        return _fitted_drivers(q[:n] * d, v, (jnp.exp(q[n]), jnp.exp(q[n + 1]), jnp.exp(q[n + 2])))

    q = jnp.concatenate(
        [fitted.attract, jnp.log(jnp.array([fitted.beta, fitted.congestion, fitted.mass]))]
    )
    drivers = jax.vmap(lambda d, v: one_block(q, d, v))(demand, u)
    below = np.asarray(drivers < demand).ravel()
    jacobian = np.asarray(jax.vmap(jax.jacrev(one_block), in_axes=(None, 0, 0))(q, demand, u))
    residual = np.asarray(y - jnp.minimum(demand, drivers)).ravel()
    basis, _ = np.linalg.qr(jacobian.reshape(-1, q.size)[below])
    fitted_part = basis.T @ residual[below]
    explained, total, size = fitted_part @ fitted_part, residual @ residual, q.size
    offset = math.sqrt(explained / size) / math.sqrt((total - explained) / (residual.size - size))
    return offset, float(jnp.min(jnp.abs(drivers - demand)))


def test_the_fit_converges_where_no_gauss_newton_step_is_left() -> None:
    """On market 2's logs of 400 blocks the fit stopped on scipy's default tolerance on the change
    in its objective, 2.2e-9 of 1 where the objective is 4e-4, and said it converged. A Gauss-Newton
    step from there was 2.62 times the radius of the parameters' confidence region in its own
    metric (Bates and Watts' relative offset), and lowered the mean squared residual by 2.1%. The
    fit now stops at an offset of 1e-3 at most; no zone sits at its demand, where the residual has
    a kink and the offset is read along it."""
    market = SharedStateMarket(seed=2)
    logs = market.generate_logs(400, jax.random.key(3))
    fitted = fit_equilibrium(logs)
    offset, gap = _relative_offset(fitted, logs)
    assert gap > 1e-3
    assert fitted.converged
    assert offset <= 1e-3


@pytest.mark.parametrize(("seed", "rides"), [(4, 11.5314154095), (1, 11.2246544642)])
def test_the_oracle_is_the_exact_optimum(seed: int, rides: float) -> None:
    """The oracle was 1200 steps of an ascent from the even split, a local maximum at best: on
    markets 4 and 1 it completed 11.32 and 10.55 rides where the exact allocation completes 11.53
    and 11.22. The market's own equilibrium solve, to its tolerance of 1e-6 of the drivers, now
    reads the exact allocation's rides to 2e-6."""
    market = SharedStateMarket(seed=seed)
    demand, attract = market._base()
    exact = exact_congestion_allocation(
        attract, demand, market.beta, market.congestion, market.mass, market.budget
    )
    assert exact.value == pytest.approx(rides, abs=1e-9)
    assert market.value(market.oracle_allocation()) == pytest.approx(rides, abs=2e-6)


def test_past_the_certified_crowding_the_market_damps_its_solve_and_reads_the_equilibrium() -> None:
    """At ``beta*congestion = 20`` the solve damped by 1/2 2-cycled at a residual of 0.26, and the
    market read 11.43 rides at the oracle's incentive where its equilibrium completes 12.0. Damped
    by ``4/(4 + beta*congestion)`` it converges, and reads the rides of the exact allocation's own
    equilibrium, found by inverting the shares rather than iterating."""
    market = SharedStateMarket(seed=0, congestion=8.0)
    demand, attract = market._base()
    found = exact_congestion_allocation(
        attract, demand, market.beta, market.congestion, market.mass, market.budget
    )
    assert market.value(jnp.asarray(found.allocation)) == pytest.approx(found.value, abs=2e-5)


def test_a_solve_that_ends_unconverged_raises_rather_than_return_drivers() -> None:
    """Drivers this price-sensitive leave the least attractive zones almost empty, and there the
    damping ``4/(4 + beta*congestion)`` at ``beta*congestion = 10**4`` closes 0.04% of the gap a
    step: 500 steps leave the residual far above its tolerance. The market raises, eagerly and
    inside the logs' scan, where it returned the unconverged drivers as an equilibrium."""
    market = SharedStateMarket(seed=0, beta=1e4, congestion=1.0)
    with pytest.raises(RuntimeError, match="did not converge"):
        market.value(jnp.zeros(market.n_zones))
    with pytest.raises(RuntimeError, match="did not converge"):
        market.generate_logs(3, jax.random.key(0))


@pytest.mark.parametrize(
    ("seed", "payment", "dither"),
    [(0, "lump_sum", 0.0), (1, "lump_sum", 0.3), (2, "per_driver", 0.0)],
)
def test_below_the_certified_crowding_the_logs_keep_their_bits(
    seed: int, payment: str, dither: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The published markets, ``beta*congestion = 5``, log what the plain solve damped by 1/2 logs,
    bit for bit, with and without a dither and paid either way."""
    market = SharedStateMarket(seed=seed, payment=payment)  # type: ignore[arg-type]
    logs = market.generate_logs(40, jax.random.key(seed + 5), dither=dither)
    monkeypatch.setattr(
        SharedStateMarket,
        "_equilibrium",
        lambda self, attract, u: (
            softmax_congestion_equilibrium(attract, u, self.congestion, self.mass, self.beta).x
        ),
    )
    plain = market.generate_logs(40, jax.random.key(seed + 5), dither=dither)
    assert sorted(logs) == sorted(plain)
    for name in logs:
        np.testing.assert_array_equal(np.asarray(logs[name]), np.asarray(plain[name]))


def test_the_oracle_takes_no_step_count() -> None:
    """``steps`` was ignored with a warning in 0.22.0 and is gone: the oracle is the exact optimum,
    and the ascent it ran is ``chc.games.stackelberg_allocation``."""
    with pytest.raises(TypeError, match="steps"):
        SharedStateMarket(seed=4).oracle_allocation(steps=1200)  # type: ignore[call-arg]


def test_an_attractiveness_no_block_moves_reads_an_infinite_error() -> None:
    """Drivers linear in three parameters, the third moving no block: the first two read the
    Gauss-Newton errors ``s * sqrt(diag((J'J)^-1))`` of the two-column fit, the third ``inf``,
    not the pseudo-inverse's zero."""
    rng = np.random.default_rng(0)
    slope = np.column_stack([rng.normal(size=40), rng.normal(size=40), np.zeros(40)])
    p = np.array([0.3, -0.2, 5.0])
    y = slope @ p + 0.1 * rng.normal(size=40)
    se = _attract_se(
        p,
        lambda q: (slope @ q).reshape(8, 5),
        lambda q: slope.reshape(8, 5, 3),
        np.full((8, 5), 1e9),
        y.reshape(8, 5),
        float(np.finfo(np.float64).eps),
        3,
    )
    left = y - slope @ p
    variance = left @ left / (40 - 2)
    expected = np.sqrt(variance * np.diag(np.linalg.inv(slope[:, :2].T @ slope[:, :2])))
    np.testing.assert_allclose(se[:2], expected, rtol=1e-12, atol=0.0)
    assert np.isinf(se[2])


def test_the_planner_holds_a_zone_the_logs_do_not_identify_at_zero() -> None:
    """On market 2's logs of 50 blocks zones 8 and 10 draw almost no drivers: the fit reads zone 8's
    attractiveness -4.3 with a standard error of 9e2, and zone 10's not at all. Planned through,
    the fit spent 3.1 of the budget of 6 on zone 8 and won 0.15 of the exact oracle's lift over no
    incentive; held at zero, the plan spends it on the zones the logs read."""
    market = SharedStateMarket(seed=2)
    logs = market.generate_logs(50, jax.random.key(3))
    fitted = fit_equilibrium(logs)
    se, attract = np.asarray(fitted.attract_se), np.asarray(fitted.attract)
    assert se[8] > abs(attract[8])
    assert np.isinf(se[10])
    assert np.all(np.isfinite(se[[0, 1, 2, 3, 4, 5, 6, 7, 9, 11]]))
    plan = pessimistic_equilibrium_allocation(
        market, calibrate_shared_state(logs), radius=1.0, equilibrium=fitted
    )
    assert float(plan[8]) == 0.0
    assert float(plan[10]) == 0.0
    base = market.value(jnp.zeros(market.n_zones))
    oracle = market.value(market.oracle_allocation())
    assert market.value(plan) - base > 0.5 * (oracle - base)


def test_a_fit_on_logs_that_carry_no_response_is_refused_by_the_planner(env: dict) -> None:
    """The fitted equilibrium is the logs': with each zone's incentives shuffled across the blocks
    they no longer move the completions they are logged beside, and the fit reads drivers that do
    not follow an incentive. Planned through, it spent the budget anyway, and the plan won none of
    the oracle's lift over doing nothing; on other markets it lost completions. Fitted again with
    no incentive at all, the logs read as well or better, so the planner refuses the fit, where the
    logs' own plans and wins most of the lift."""
    market, logs, base, oracle = env["market"], env["logs"], env["base"], env["oracle"]
    shared = calibrate_shared_state(logs)
    order = np.random.default_rng(0).permutation(logs["u"].shape[0])
    scrambled = fit_equilibrium(logs | {"u": logs["u"][order]})
    assert scrambled.beta < 0.1
    assert scrambled.incentive_z < 1.645 < env["fitted"].incentive_z
    with pytest.raises(ValueError, match="do not tell an incentive from none"):
        pessimistic_equilibrium_allocation(market, shared, equilibrium=scrambled)
    plan = pessimistic_equilibrium_allocation(market, shared, equilibrium=env["fitted"])
    assert market.value(plan) - base > 0.8 * (oracle - base)


def _fixed_attractiveness_logs(market: SharedStateMarket, n_blocks: int, key: jax.Array) -> dict:
    """Logs of a market outside the fitted class: each block's demand moves with its season, and
    the zones' attractiveness stays at its base."""
    base_demand, base_attract = market._base()

    def block(carry: None, k: jax.Array) -> tuple[None, dict]:
        k_season, k_scale, k_noise = jax.random.split(k, 3)
        demand = base_demand * (0.5 + jax.random.uniform(k_season, (market.n_zones,)))
        scale = 0.4 + 1.2 * jax.random.uniform(k_scale)
        u = scale * market.budget * jax.nn.softmax(market.confounding * demand)
        x = market._equilibrium(base_attract, u)
        y = jnp.minimum(demand, x) + 0.02 * jax.random.normal(k_noise, (market.n_zones,))
        return None, {
            "u": u,
            "demand": demand,
            "aggregate": jnp.full(market.n_zones, jnp.sum(u)),
            "y": y,
        }

    return jax.lax.scan(block, None, jax.random.split(key, n_blocks))[1]


def test_a_market_outside_the_fitted_class_is_warned_of(env: dict) -> None:
    """Attractiveness that stays put while the demand moves: the fit converged, with its
    sensitivity half a unit off and nothing said. Its residuals follow each zone's demand and
    incentive, and the F test on them reads it, where on the class's own logs it does not."""
    market = env["market"]
    with pytest.warns(UserWarning, match=r"residuals still follow the zones' demand.*p = "):
        fitted = fit_equilibrium(_fixed_attractiveness_logs(market, 400, jax.random.key(1)))
    assert fitted.converged
    assert fitted.lack_of_fit < 1e-10
    assert env["fitted"].lack_of_fit > 0.01
    assert fitted.incentive_z > 1.645  # the incentive still moves the drivers


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda logs: logs | {"demand": logs["demand"][:, :11]}, r"logs\['demand'\] has shape"),
        (lambda logs: logs | {"u": logs["u"][:, 0]}, r"logs\['u'\] has shape \(400,\)"),
        (lambda logs: logs | {"y": logs["y"].at[3, 2].set(jnp.nan)}, r"logs\['y'\] is not finite"),
        (lambda logs: logs | {"demand": logs["demand"].at[5, 1].set(-0.3)}, "demand is positive"),
        (lambda logs: {name: column[:14] for name, column in logs.items()}, "14 blocks for 12"),
    ],
    ids=["zones", "vector", "nan", "negative demand", "blocks"],
)
def test_logs_outside_the_fit_s_contract_are_refused(env: dict, edit, message: str) -> None:
    """11 zones of demand beside 12 of incentive failed in JAX's broadcasting, a vector of
    incentives on an index, and a negative demand, or 3 blocks, fitted and "converged" (the
    sensitivity 3.57 for 2.5)."""
    with pytest.raises(ValueError, match=message):
        fit_equilibrium(edit(env["logs"]))


def test_the_planner_refuses_zones_that_are_not_the_market_s(env: dict) -> None:
    """A 6-zone equilibrium on a 12-zone market failed in JAX's broadcasting."""
    market, logs = env["market"], env["logs"]
    shared = calibrate_shared_state(logs)
    six = SharedStateMarket(n_zones=6, seed=0)
    small = fit_equilibrium(six.generate_logs(60, jax.random.key(4)))
    with pytest.raises(ValueError, match="the equilibrium has 6 zones and the market 12"):
        pessimistic_equilibrium_allocation(market, shared, equilibrium=small)
    short = ExposureResponse(marginal=shared.marginal[:6], se=shared.se[:6])
    with pytest.raises(ValueError, match=r"response.marginal has shape \(6,\)"):
        pessimistic_equilibrium_allocation(market, short, equilibrium=env["fitted"])


def test_naive_over_predicts_while_chc_delivers_more(env: dict) -> None:
    market = env["market"]
    naive_resp = calibrate_naive_causal(env["logs"])
    naive_alloc = sutva_allocation(market, naive_resp, radius=1.0)
    chc_alloc = pessimistic_equilibrium_allocation(
        market, calibrate_shared_state(env["logs"]), radius=1.0, equilibrium=env["fitted"]
    )
    # the naive per-zone (additive) model predicts a lift the equilibrium never realises...
    assert interference_bias(market, naive_resp, naive_alloc) > 1.0
    # ...and CHC, planning through that equilibrium, realises strictly more completions than naive:
    # 3.40 more on this market in float32, 1.24 in float64, and from 0.02 to 3.40 over markets 0-29
    assert market.value(chc_alloc) > market.value(naive_alloc) + 1.0


def test_a_negative_response_earns_its_zone_no_bonus(env: dict) -> None:
    """The tilt was the pessimistic response less the raw estimate, ``max(m - r se, 0) - m``, so a
    zone whose estimate is negative earned ``|m|`` per unit of incentive. Against the estimate
    projected onto the same non-negative prior, a negative estimate tilts nothing, and the plan is
    the zero response's."""
    market = env["market"]
    zero = jnp.zeros(market.n_zones)
    negative = ExposureResponse(marginal=zero.at[0].set(-0.5), se=zero)
    fitted = env["fitted"]
    one = pessimistic_equilibrium_allocation(
        market, ExposureResponse(marginal=zero, se=zero), equilibrium=fitted
    )
    other = pessimistic_equilibrium_allocation(market, negative, equilibrium=fitted)
    _assert_close(other, one, 1e-6)


UNITS = [1e-9, 1e-6, 1e-3, 1e3, 1e6]
CALIBRATIONS = [calibrate_predictive, calibrate_naive_causal, calibrate_shared_state]


def _assert_close(other: jax.Array, one: jax.Array, rel: float, part: str = "") -> None:
    """Agreement to ``rel`` of the reference's largest entry."""
    reference = np.asarray(one)
    np.testing.assert_allclose(
        np.asarray(other),
        reference,
        rtol=0.0,
        atol=rel * float(np.max(np.abs(reference))),
        err_msg=part,
    )


@pytest.mark.parametrize("units", UNITS)
@pytest.mark.parametrize("calibrate", CALIBRATIONS, ids=lambda f: f.__name__)
def test_the_calibrated_response_reads_the_same_with_the_incentive_in_any_units(
    env: dict, calibrate, units: float
) -> None:
    """Each zone's slope was solved under a constant ridge of 1e-4: with the incentive and its
    aggregate logged in thousandths of their units, the response per original unit was 0.93 to
    0.97 off, and in millionths it read zero in every zone."""
    logs = env["logs"]
    scaled = logs | {"u": logs["u"] * units, "aggregate": logs["aggregate"] * units}
    one, other = calibrate(logs), calibrate(scaled)
    _assert_close(other.marginal * units, one.marginal, 1e-9, "marginal")
    _assert_close(other.se * units, one.se, 1e-9, "se")


@pytest.mark.parametrize("units", UNITS)
@pytest.mark.parametrize(
    "calibrate", [calibrate_naive_causal, calibrate_shared_state], ids=lambda f: f.__name__
)
def test_the_calibrated_response_reads_the_same_with_the_demand_in_any_units(
    env: dict, calibrate, units: float
) -> None:
    """With the demand logged in millionths of its units the de-confounded responses were 0.69 and
    0.38 off: the ridge took the adjustment away, and the naive causal response read the
    unadjusted, predictive one."""
    logs = env["logs"]
    one, other = calibrate(logs), calibrate(logs | {"demand": logs["demand"] * units})
    _assert_close(other.marginal, one.marginal, 1e-9, "marginal")
    _assert_close(other.se, one.se, 1e-9, "se")


@pytest.mark.parametrize("units", UNITS)
@pytest.mark.parametrize("columns", [("y",), ("u", "aggregate")], ids=["outcome", "incentive"])
def test_the_sutva_allocation_reads_the_same_in_any_units_of_the_log(
    env: dict, columns: tuple[str, ...], units: float
) -> None:
    """The budget is split in proportion to each zone's pessimistic uplift, whose units cancel, but
    the total was floored by adding 1e-9 in those units: with the completions logged in millionths
    of their units the allocation moved by 2.3e-4 of the largest, and in billionths by 0.19."""
    market, logs = env["market"], env["logs"]
    scaled = logs | {column: logs[column] * units for column in columns}
    for calibrate in (calibrate_predictive, calibrate_naive_causal):
        one = sutva_allocation(market, calibrate(logs), radius=1.0)
        other = sutva_allocation(market, calibrate(scaled), radius=1.0)
        _assert_close(other, one, 1e-9, calibrate.__name__)


@pytest.mark.parametrize("level", [0.0, 0.4], ids=["zeros", "constant"])
def test_a_zone_whose_incentive_never_moved_reads_a_response_of_zero(
    env: dict, level: float
) -> None:
    """An incentive that never moves has no variance to scale the ridge by, and a constant one is
    the intercept's column again, so without a term of its own the zone's Gram is singular. It
    counts as constant: its centred values are zeroed, the zone reads a response and an SE of
    zero, and every other zone reads what it did. With the ridge on the intercept too, a zone
    logged at 0.4 throughout read 0.367, -0.776 and -0.729, a response the log cannot identify.
    The constant carries a relative jitter of 1e-14, 45 epsilons, under the 64 at which a column's
    spread is rounding; with its centred values kept it left a response of 4e-9 of the largest and
    an SE of 6e-9, and solved on the Gram of the raw columns, 1.3e-7 and 5e-3."""
    logs = env["logs"]
    jitter = 1.0 + 1e-14 * np.random.default_rng(3).standard_normal(logs["u"].shape[0])
    idle = logs | {"u": logs["u"].at[:, 0].set(jnp.asarray(level * jitter))}
    for calibrate in CALIBRATIONS:
        one, other = calibrate(logs), calibrate(idle)
        assert (float(other.marginal[0]), float(other.se[0])) == (0.0, 0.0), calibrate.__name__
        _assert_close(other.marginal[1:], one.marginal[1:], 1e-12, calibrate.__name__)


FAR = [
    (calibrate_predictive, "u"),
    (calibrate_naive_causal, "u"),
    (calibrate_naive_causal, "demand"),
    (calibrate_shared_state, "u"),
    (calibrate_shared_state, "demand"),
    (calibrate_shared_state, "aggregate"),
]


@pytest.mark.parametrize(
    ("calibrate", "column"), FAR, ids=[f"{f.__name__}-{column}" for f, column in FAR]
)
def test_the_calibrated_response_reads_the_same_with_a_column_far_from_zero(
    env: dict, calibrate, column: str
) -> None:
    """Each zone's regression has an intercept, so a column's origin carries no information.
    Shifted by a thousand of its spreads, the incentive moved the first zone's predictive response
    to 0.658 where it reads 0.823, the demand its naive-causal one to 0.542 where 0.359, and the
    aggregate its shared-state one to 0.519 where 0.577, as the ridge was on the intercept too; in
    millionths of their units as well, they read 0, 0.823 and 0.359. A ridge scaled by each
    column's mean square moves them as far or further. Solved about the columns' means, the shift
    costs no digits, and rounding moves a response by 2e-14 of the largest, where on the Gram of
    the raw columns by 5e-9."""
    logs = env["logs"]
    raw = logs[column]
    moved = 1e-6 * (raw + 1e3 * jnp.std(raw, axis=0))
    one, other = calibrate(logs), calibrate(logs | {column: moved})
    units = 1e-6 if column == "u" else 1.0
    _assert_close(other.marginal * units, one.marginal, 1e-11, "marginal")
    _assert_close(other.se * units, one.se, 1e-11, "se")


@pytest.mark.parametrize("calibrate", CALIBRATIONS, ids=lambda f: f.__name__)
def test_numpy_logs_calibrate_as_the_same_logs_in_jax(env: dict, calibrate) -> None:
    """Each zone's columns are taken by a traced index under ``jax.vmap``, which a NumPy array
    cannot take: logs of NumPy arrays failed with JAX's ``TracerArrayConversionError``."""
    logs = env["logs"]
    one = calibrate(logs)
    other = calibrate({name: np.asarray(column) for name, column in logs.items()})
    for part in ("marginal", "se"):
        left, right = np.asarray(getattr(other, part)), np.asarray(getattr(one, part))
        assert (left.dtype, left.tobytes()) == (right.dtype, right.tobytes()), part


def _plane(n: int = 1_000) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``y = 1 + 2 x + 0.5 z`` and a tenth of a unit of noise."""
    x, z, noise = np.random.default_rng(0).standard_normal((3, n))
    return jnp.asarray(x), jnp.asarray(z), jnp.asarray(1.0 + 2.0 * x + 0.5 * z + 0.1 * noise)


@pytest.mark.parametrize("offset", [1e-11, 1e-13], ids=["1e-11", "1e-13"])
def test_a_column_constant_but_for_1e_11_or_1e_13_of_its_size_reads_its_slope(
    offset: float,
) -> None:
    """``1 + s x`` is ``x`` in units of ``s``, 45,000 or 450 epsilons of its size, so it is data:
    as the incentive its slope and SE are x's over ``s``, and as a covariate it leaves the
    incentive's as they are. The column holds x to five significant digits at 1e-11 and three at
    1e-13, so they agree to 1e-15 / s: the slopes to 1e-7 and 1.4e-5, the SEs, over residuals a
    tenth of a unit, to 1e-6 and 4.3e-4. In the Gram of the raw columns its spread drowned in the
    rounding of its offset, and so it did with the ridge a constant: as the incentive its slope
    read zero, and as a covariate it moved the incentive's slope by 1.6e-2 and its SE fourfold. A
    floor of 1e-12 of its size counted the 1e-13 column as constant, with the same readings."""
    x, z, y = _plane()
    slope, se = _zone_slope(x, y, z[:, None])
    rel = 1e-15 / offset
    for part, units, (other_slope, other_se) in (
        ("incentive", offset, _zone_slope(1.0 + offset * x, y, z[:, None])),
        ("covariate", 1.0, _zone_slope(x, y, (1.0 + offset * z)[:, None])),
    ):
        assert float(other_slope) * units == pytest.approx(float(slope), rel=rel, abs=0.0), part
        assert float(other_se) * units == pytest.approx(float(se), rel=rel, abs=0.0), part


def test_a_column_constant_to_4e_15_of_its_size_reads_no_slope() -> None:
    """4e-15 of its size is 18 epsilons, under the 64 at which a column's spread is rounding, so
    the column counts as constant: its centred values are zeroed. As the incentive it reads a slope
    and an SE of exactly zero, and as a covariate it leaves the incentive's as with the column
    exactly constant. Kept, its centred values read 1.5e-10 with an SE of 1.2e-10 as the
    incentive, the residual's sum along their 4e-15 over the ridge of 1e-4; scaled by its
    variance, 1.6e-29 of that, the incentive took up the noise with a slope of 9.5e11; with the
    ridge a constant on the intercept too, it read half the intercept, 0.498."""
    x, z, y = _plane()
    signs = np.where(np.random.default_rng(1).random(1_000) < 0.5, -1.0, 1.0)
    still = jnp.asarray(1.0 + 4e-15 * signs)
    slope, se = _zone_slope(still, y, jnp.stack([x, z], axis=1))
    assert (float(slope), float(se)) == (0.0, 0.0)
    near = _zone_slope(x, y, jnp.stack([z, still], axis=1))
    reference = _zone_slope(x, y, jnp.stack([z, jnp.ones_like(z)], axis=1))
    np.testing.assert_array_equal(np.asarray(near), np.asarray(reference))


def test_a_column_constant_to_1e_6_of_its_size_is_rounding_in_float32() -> None:
    """In float32 the 64 epsilons at which a column's spread is rounding are 7.6e-6 of its size,
    so a column constant but for 1e-6 of it, 8 epsilons, counts as constant: its centred values are
    zeroed. As the incentive it reads a slope and an SE of exactly zero, and as a covariate it
    leaves the incentive's as with the column exactly constant. Kept, its centred values read
    0.037 with an SE of 0.030 as the incentive, the residual's sum along their 1e-6 over the ridge
    of 1e-4; counted as data, by a floor of 1e-24 of its mean square or by float64's epsilon, the
    incentive took up the noise with a slope of 3862 and an SE of 3150."""
    x, z, y = (a.astype(jnp.float32) for a in _plane())
    signs = np.where(np.random.default_rng(1).random(1_000) < 0.5, -1.0, 1.0)
    still = jnp.asarray(1.0 + 1e-6 * signs, dtype=jnp.float32)
    slope, se = _zone_slope(still, y, jnp.stack([x, z], axis=1))
    assert slope.dtype == se.dtype == jnp.float32
    assert (float(slope), float(se)) == (0.0, 0.0)
    near = _zone_slope(x, y, jnp.stack([z, still], axis=1))
    reference = _zone_slope(x, y, jnp.stack([z, jnp.ones_like(z)], axis=1))
    np.testing.assert_array_equal(np.asarray(near), np.asarray(reference))


@pytest.mark.parametrize("calibrate", CALIBRATIONS, ids=lambda f: f.__name__)
def test_a_float32_log_is_calibrated_in_float32(env: dict, calibrate) -> None:
    """Every calibration of a float32 log solves and reports in float32, so the float32 rule
    applies: a zone whose incentive is constant but for 1e-6 of it, 8 epsilons, counts as constant
    and reads a response and an SE of exactly zero. The predictive calibration has no covariates,
    and its empty block of them was float64 under x64: it promoted each zone's solve to float64,
    where 8 float32 epsilons are data, and that zone read a response of -42576 with an SE of
    49767."""
    logs = env["logs"]
    single = {name: column.astype(jnp.float32) for name, column in logs.items()}
    level = float(jnp.mean(logs["u"][:, 0]))
    signs = np.where(np.random.default_rng(1).random(logs["u"].shape[0]) < 0.5, -1.0, 1.0)
    still = jnp.asarray(level * (1.0 + 1e-6 * signs), dtype=jnp.float32)
    response = calibrate(single | {"u": single["u"].at[:, 0].set(still)})
    assert response.marginal.dtype == response.se.dtype == jnp.float32
    assert (float(response.marginal[0]), float(response.se[0])) == (0.0, 0.0)


def test_a_zone_whose_incentive_is_rounding_in_float32_takes_no_budget(env: dict) -> None:
    """Logged in float32, a zone whose incentive is constant but for 1e-6 of it, 8 epsilons, reads
    a response and an SE of zero under every calibration, and takes no share of the budget
    whichever way its jitter falls; the other zones share all of it. Kept, its centred values read
    a response of 0.13 and 0.14 with SEs of 0.099 and 0.098 under the calibrations that adjust for
    covariates, and with the jitter one way round the zone took 0.053 and 0.031 of the budget of
    6."""
    market, logs = env["market"], env["logs"]
    single = {name: column.astype(jnp.float32) for name, column in logs.items()}
    level = float(jnp.mean(logs["u"][:, 0]))
    signs = np.where(np.random.default_rng(1).random(logs["u"].shape[0]) < 0.5, -1.0, 1.0)
    for jitter in (1.0 + 1e-6 * signs, 1.0 - 1e-6 * signs):
        still = jnp.asarray(level * jitter, dtype=jnp.float32)
        idle = single | {"u": single["u"].at[:, 0].set(still)}
        for calibrate in CALIBRATIONS:
            allocation = sutva_allocation(market, calibrate(idle), radius=1.0)
            assert allocation.dtype == jnp.float32, calibrate.__name__
            assert float(allocation[0]) == 0.0, calibrate.__name__
            assert float(jnp.sum(allocation)) == pytest.approx(market.budget, rel=1e-6, abs=0.0)


def test_an_incentive_100_epsilons_from_constant_is_data_in_float32() -> None:
    """In float32 an incentive whose spread is about 100 epsilons of its size is data, above the 64
    at which it is rounding, and its slope is large: ``0.4 (1 + s x)`` reads 2 / (0.4 s). The
    deviations' own mean is taken off the centred columns, so its slope and SE read what float64
    reads from the same values, to 2.4e-7 and 9.4e-8. Centred about its rounded mean alone, the
    incentive carried that rounding, a few epsilons of its size, into the residuals through its
    slope: the slope read 3.2e-5 off and the SE 1.2e-3."""
    x, z, y = _plane()
    s = 100 * float(jnp.finfo(jnp.float32).eps)
    single = tuple(a.astype(jnp.float32) for a in (0.4 * (1.0 + s * x), y, z[:, None]))
    slope, se = _zone_slope(*single)
    assert slope.dtype == se.dtype == jnp.float32
    double = _zone_slope(*(a.astype(jnp.float64) for a in single))
    np.testing.assert_allclose(np.asarray([slope, se]), np.asarray(double), rtol=1e-5, atol=0.0)


def test_a_zone_whose_incentive_never_moved_reads_a_response_of_zero_in_float32() -> None:
    """JAX computes in float32 unless x64 is on, and there a mean is rounded to about 1e-7 of its
    size, so a constant incentive deviates from its computed mean by that rounding on every row.
    Solved about that mean alone, a zone logged at 0.4 throughout read a response of 1.3e-7 and an
    SE of 1.0e-2, twice that of the zone with its incentive moving. It counts as constant, its
    centred values are zeroed, and the response and its SE are zero."""
    x, z, y = (a.astype(jnp.float32) for a in _plane(400))
    slope, se = _zone_slope(jnp.full(400, 0.4, dtype=jnp.float32), y, jnp.stack([x, z], axis=1))
    assert slope.dtype == se.dtype == jnp.float32
    assert (float(slope), float(se)) == (0.0, 0.0)


def test_a_response_with_no_positive_uplift_funds_no_zone(env: dict) -> None:
    """With every pessimistic uplift at zero the shares have no total, and the allocation is zero
    in every zone rather than nan."""
    market = env["market"]
    flat = ExposureResponse(marginal=jnp.zeros(market.n_zones), se=jnp.ones(market.n_zones))
    assert np.all(np.asarray(sutva_allocation(market, flat, radius=1.0)) == 0.0)


def _linear_sandwich_case() -> tuple:
    """Drivers linear in three parameters over 8 blocks of 5 zones, every zone below its demand,
    and the residuals of one block sharing a draw, so the blocks' scores carry what the rows'
    do not."""
    rng = np.random.default_rng(1)
    slope = rng.normal(size=(8, 5, 3))
    p = np.array([0.3, -0.2, 0.5])
    shared = np.repeat(rng.normal(size=(8, 1)), 5, axis=1)
    y = np.einsum("bip,p->bi", slope, p) + 0.1 * rng.normal(size=(8, 5)) + 0.1 * shared
    return slope, p, y


def _clustered(slope: np.ndarray, p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``H^-1 M H^-1`` written out: ``H`` the stacked rows' Gram, ``M`` the blocks' scores'."""
    rows = slope.reshape(-1, slope.shape[-1])
    bread = np.linalg.inv(rows.T @ rows)
    meat = sum(
        np.outer(block.T @ (y_b - block @ p), block.T @ (y_b - block @ p))
        for block, y_b in zip(slope, y, strict=True)
    )
    return bread @ meat @ bread


def test_the_fit_s_sandwich_is_the_block_clustered_one() -> None:
    """The fit carried no covariance of its parameters: its sampling error was not read anywhere.
    On a model linear in its parameters the sandwich is Cameron and Miller's ``H^-1 M H^-1`` with
    each block's score once; a parameter held is nan in its row and column, and the others read
    the sandwich of the fit without it."""
    slope, p, y = _linear_sandwich_case()
    eps = float(np.finfo(np.float64).eps)

    def read(kept: list[int]) -> np.ndarray:
        return _sandwich(
            p,
            lambda q: np.einsum("bip,p->bi", slope, q),
            lambda q: slope,
            np.full((8, 5), 1e9),
            y,
            eps,
            np.array(kept),
        )

    np.testing.assert_allclose(read([0, 1, 2]), _clustered(slope, p, y), rtol=1e-12, atol=0.0)
    held = read([0, 2])
    assert np.all(np.isnan(held[1]))
    assert np.all(np.isnan(held[:, 1]))
    without = _clustered(slope[..., [0, 2]], p[[0, 2]], y - slope[..., 1] * p[1])
    np.testing.assert_allclose(held[np.ix_([0, 2], [0, 2])], without, rtol=1e-12, atol=0.0)


def test_a_sandwich_the_logs_do_not_read_is_nan() -> None:
    """A parameter no block moves leaves the Gauss-Newton matrix singular: a pseudo-inverse would
    read its variance as zero, so the sandwich reads nan in every entry."""
    slope, p, y = _linear_sandwich_case()
    slope = slope.copy()
    slope[..., 2] = 0.0
    covariance = _sandwich(
        p,
        lambda q: np.einsum("bip,p->bi", slope, q),
        lambda q: slope,
        np.full((8, 5), 1e9),
        y,
        float(np.finfo(np.float64).eps),
        np.arange(3),
    )
    assert np.all(np.isnan(covariance))


def test_the_fit_carries_its_sandwich_and_its_demand_s_covariance(env: dict) -> None:
    """Market 0's 400 blocks: every parameter is read, the sandwich is symmetric and positive
    definite, and the planning demand's covariance is the blocks' over their count."""
    fitted, logs = env["fitted"], env["logs"]
    covariance = np.asarray(fitted.covariance, np.float64)
    assert covariance.shape == (15, 15)
    assert np.all(np.isfinite(covariance))
    np.testing.assert_allclose(covariance, covariance.T, rtol=0.0, atol=1e-12 * covariance.max())
    assert np.linalg.eigvalsh(covariance).min() > 0.0
    demand = np.asarray(logs["demand"], np.float64)
    np.testing.assert_allclose(
        np.asarray(fitted.demand_covariance, np.float64),
        np.cov(demand, rowvar=False) / demand.shape[0],
        rtol=1e-12,
        atol=0.0,
    )


def test_the_driver_jacobians_are_the_solve_s_in_closed_form(env: dict) -> None:
    """The closed forms of the drivers' Jacobians, in the fit's parameters and in the planning
    demand, against reverse mode through the equilibrium solved to 1e-13."""
    fitted = env["fitted"]
    n = fitted.demand.shape[0]
    u = jnp.linspace(0.0, 1.0, n)
    q0 = jnp.concatenate(
        [fitted.attract, jnp.log(jnp.array([fitted.beta, fitted.congestion, fitted.mass]))]
    )

    def solved(q: jax.Array, demand: jax.Array) -> jax.Array:
        params = (q[:n] * demand, u, jnp.exp(q[n]), jnp.exp(q[n + 1]), jnp.exp(q[n + 2]))
        start = jnp.full(n, jnp.exp(q[n + 2]) / n)
        return fixed_point(_fitted_step, params, start, tol=1e-13, max_iter=20_000).x

    drivers = solved(q0, fitted.demand)
    by_parameters, by_demand = _driver_jacobians(
        drivers, u, fitted.attract, fitted.demand, fitted._parameters()
    )
    np.testing.assert_allclose(
        by_parameters, jax.jacrev(solved)(q0, fitted.demand), rtol=0.0, atol=1e-9
    )
    np.testing.assert_allclose(
        by_demand, jax.jacrev(solved, argnums=1)(q0, fitted.demand), rtol=0.0, atol=1e-9
    )


def test_the_expected_completions_are_the_normal_loss_function() -> None:
    """``E[min(d, X)]``, X normal, against quadrature zone by zone, and ``min(d, x)`` where the
    variance is zero."""
    drivers = jnp.array([0.8, 1.3, 2.0, 0.5])
    demand = jnp.array([1.0, 1.0, 1.5, 0.5])
    variance = jnp.array([0.04, 0.09, 0.0, 0.01])
    by_zone = []
    for x, d, v in zip(drivers.tolist(), demand.tolist(), variance.tolist(), strict=True):
        if v == 0.0:
            by_zone.append(min(x, d))
            continue
        sd = math.sqrt(v)

        def density(t: float, x: float = x, sd: float = sd) -> float:
            return math.exp(-0.5 * ((t - x) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))

        below, _ = integrate.quad(lambda t: t * density(t), x - 40 * sd, d, epsabs=1e-14)
        above, _ = integrate.quad(density, d, x + 40 * sd, epsabs=1e-14)
        by_zone.append(below + d * above)
    assert float(_expected_completions(drivers, demand, variance)) == pytest.approx(
        math.fsum(by_zone), rel=1e-12, abs=0.0
    )


def test_the_expected_objective_plans_market_18_where_the_tilted_ascent_stops_short() -> None:
    """Float32 market 18, logs of 400 blocks: the tilted plan's 600-step ascent from the even split
    ended short on the fitted equilibrium, and the naive plan won more of the oracle's lift. The
    expected objective hedges each zone's kink at its demand under the fit's own uncertainty, and
    starts a second ascent from the fitted game's exact optimum: it wins most of the lift."""
    with enable_x64(False):
        market = SharedStateMarket(seed=18)
        logs = market.generate_logs(400, jax.random.key(19))
        fitted = fit_equilibrium(logs)
        plan = pessimistic_equilibrium_allocation(
            market, None, equilibrium=fitted, objective="expected"
        )
        naive = sutva_allocation(market, calibrate_naive_causal(logs), 1.0)
        base = market.value(jnp.zeros(market.n_zones))
        lift = market.value(market.oracle_allocation()) - base
        assert plan.dtype == jnp.float32
        assert float(jnp.sum(plan)) == pytest.approx(market.budget, rel=1e-6, abs=0.0)
        assert market.value(plan) - base > 0.9 * lift
        assert market.value(plan) > market.value(naive)


def test_the_expected_objective_wins_more_than_the_tilted_one(env: dict) -> None:
    """Market 0 in float64: the tilted plan won 0.918 of the exact oracle's lift, the expected one
    all of it, holding no zone and spending the budget."""
    market, base, oracle = env["market"], env["base"], env["oracle"]
    tilted = pessimistic_equilibrium_allocation(
        market, calibrate_shared_state(env["logs"]), equilibrium=env["fitted"]
    )
    expected = pessimistic_equilibrium_allocation(
        market, None, equilibrium=env["fitted"], objective="expected"
    )
    assert float(jnp.sum(expected)) == pytest.approx(market.budget, rel=1e-12, abs=0.0)
    assert market.value(expected) - base > 0.99 * (oracle - base)
    assert market.value(expected) > market.value(tilted)


def test_the_expected_objective_holds_a_zone_the_logs_do_not_identify() -> None:
    """Market 2's 50 blocks, where zones 8 and 10 draw almost no drivers: the expected objective
    holds them at zero, as the tilted one does, and reads the sandwich over the rest."""
    market = SharedStateMarket(seed=2)
    logs = market.generate_logs(50, jax.random.key(3))
    fitted = fit_equilibrium(logs)
    covariance = np.asarray(fitted.covariance)
    assert np.all(np.isnan(covariance[[8, 10]]))
    plan = pessimistic_equilibrium_allocation(
        market, None, equilibrium=fitted, objective="expected"
    )
    assert float(plan[8]) == 0.0
    assert float(plan[10]) == 0.0
    base = market.value(jnp.zeros(market.n_zones))
    assert market.value(plan) - base > 0.5 * (market.value(market.oracle_allocation()) - base)


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        ({"covariance": None}, ValueError, "carries none"),
        ({"covariance": jnp.full((15, 15), jnp.nan)}, ValueError, "sandwich is singular"),
    ],
    ids=["no covariance", "singular"],
)
def test_the_expected_objective_refuses_a_fit_whose_uncertainty_it_cannot_read(
    env: dict, change: dict, error: type, message: str
) -> None:
    fitted = dataclasses.replace(env["fitted"], **change)
    with pytest.raises(error, match=message):
        pessimistic_equilibrium_allocation(
            env["market"], None, equilibrium=fitted, objective="expected"
        )


def test_the_planner_s_objective_is_one_it_knows_and_the_tilt_needs_a_response(env: dict) -> None:
    with pytest.raises(ValueError, match="'tilted' or 'expected'"):
        pessimistic_equilibrium_allocation(
            env["market"], None, equilibrium=env["fitted"], objective="robust"
        )
    with pytest.raises(TypeError, match="pass a response"):
        pessimistic_equilibrium_allocation(env["market"], None, equilibrium=env["fitted"])


def test_the_budget_scale_s_slope_is_read_and_the_fit_matches_it(env: dict) -> None:
    """Market 0's logs: the block's total incentive is randomised independently of the demand, so
    its slope is a causal response along the logged split, and the fitted equilibrium's own
    completions read the same total slope within 0.2 of its SE."""
    read = calibrate_budget_scale(env["logs"], equilibrium=env["fitted"])
    assert read.marginal.shape == read.se.shape == (12,)
    assert np.all(np.asarray(read.se) > 0.0)
    assert read.total < 0.0 < read.total_se
    assert abs(read.z) < 0.2
    assert read.z == pytest.approx((read.total - read.fitted_total) / read.total_se, rel=1e-12)
    bare = calibrate_budget_scale(env["logs"])
    assert math.isnan(bare.fitted_total)
    assert math.isnan(bare.z)
    assert bare.total == read.total


def test_the_budget_scale_s_slope_is_the_one_ols_reads() -> None:
    """Completions linear in the block's total and the demand: each zone's slope on the total,
    its own demand held, and the total's, every zone's demand held, are the coefficients, to the
    ridge's 1e-4 of the columns' variance."""
    rng = np.random.default_rng(3)
    blocks, zones = 400, 3
    demand = rng.uniform(0.5, 1.5, size=(blocks, zones))
    total = rng.uniform(1.0, 3.0, size=blocks)
    u = total[:, None] * np.full(zones, 1.0 / zones)
    gain = np.array([0.2, -0.1, 0.05])
    y = gain * total[:, None] + 0.7 * demand + 0.01 * rng.normal(size=(blocks, zones))
    read = calibrate_budget_scale({"u": u, "demand": demand, "y": y})
    np.testing.assert_allclose(read.marginal, gain, rtol=0.0, atol=5e-3)
    assert read.total == pytest.approx(gain.sum(), abs=5e-3)


def test_the_budget_scale_refuses_logs_outside_its_contract(env: dict) -> None:
    logs = env["logs"]
    with pytest.raises(ValueError, match=r"logs\['y'\] has shape"):
        calibrate_budget_scale(logs | {"y": logs["y"][:, :11]})
    with pytest.raises(ValueError, match="zones and the logs"):
        calibrate_budget_scale(
            {name: logs[name][:, :11] for name in ("u", "demand", "y")}, equilibrium=env["fitted"]
        )


def _published_logs(market: SharedStateMarket, n_blocks: int, key: jax.Array) -> dict:
    """The logs as 0.22.0 generated them, its method's body copied: with no dither the method
    gives these, bit for bit."""
    base_demand, base_attract = market._base()

    def block(carry: None, k: jax.Array) -> tuple[None, dict]:
        k_season, k_scale, k_noise = jax.random.split(k, 3)
        season = 0.5 + jax.random.uniform(k_season, (market.n_zones,))
        demand = base_demand * season
        attract = base_attract * season
        scale = 0.4 + 1.2 * jax.random.uniform(k_scale)
        u = scale * market.budget * jax.nn.softmax(market.confounding * demand)
        x = market._equilibrium(attract, u)
        y = jnp.minimum(demand, x) + 0.02 * jax.random.normal(k_noise, (market.n_zones,))
        agg = jnp.full(market.n_zones, jnp.sum(u))
        return None, {"u": u, "demand": demand, "aggregate": agg, "y": y}

    _, logs = jax.lax.scan(block, None, jax.random.split(key, n_blocks))
    return logs


@pytest.mark.parametrize("x64", [True, False], ids=["float64", "float32"])
def test_logs_with_no_dither_are_the_published_logs_bit_for_bit(x64: bool) -> None:
    with enable_x64(x64):
        market = SharedStateMarket(seed=3)
        want = _published_logs(market, 60, jax.random.key(4))
        for logs in (
            market.generate_logs(60, jax.random.key(4)),
            market.generate_logs(60, jax.random.key(4), dither=0.0),
        ):
            assert sorted(logs) == sorted(want)
            for name in want:
                np.testing.assert_array_equal(np.asarray(logs[name]), np.asarray(want[name]))


@pytest.mark.parametrize("zones", [12, 5])
def test_a_dither_moves_the_logits_and_holds_each_block_s_total(zones: int) -> None:
    """Each block's logits move by +-h with as many signs of each kind (a 0 on the odd zone out):
    the incentives stay positive, the block's total holds, the demand is the undithered logs', and
    each rate is the undithered one times ``exp(shift)``, renormalised."""
    market = SharedStateMarket(n_zones=zones, seed=3)
    plain = market.generate_logs(60, jax.random.key(4))
    logs = market.generate_logs(60, jax.random.key(4), dither=0.3)
    shift = np.asarray(logs["dither"])
    assert np.all((shift == 0.3).sum(axis=1) == zones // 2)
    assert np.all((shift == -0.3).sum(axis=1) == zones // 2)
    assert np.all((shift == 0.0).sum(axis=1) == zones % 2)
    assert len({tuple(row) for row in shift}) > 20  # the signs' order is drawn block by block
    np.testing.assert_array_equal(np.asarray(logs["demand"]), np.asarray(plain["demand"]))
    u, u0 = np.asarray(logs["u"]), np.asarray(plain["u"])
    assert np.all(u > 0.0)
    np.testing.assert_allclose(u.sum(axis=1), u0.sum(axis=1), rtol=1e-14, atol=0.0)
    moved = np.log(u / u0) - shift
    np.testing.assert_allclose(moved - moved[:, :1], 0.0, rtol=0.0, atol=1e-13)


def _true_fit(market: SharedStateMarket, logs: dict) -> FittedEquilibrium:
    """The market's own parameters as a fit: attractiveness per unit of demand is the base's."""
    demand, attract = market._base()
    return FittedEquilibrium(
        demand=jnp.mean(logs["demand"], axis=0),
        attract=attract / demand,
        attract_se=jnp.zeros(market.n_zones),
        beta=market.beta,
        congestion=market.congestion,
        mass=market.mass,
        rmse=0.02,
        converged=True,
        lack_of_fit=1.0,
        incentive_z=math.inf,
    )


@pytest.mark.parametrize("key", [11, 12, 13])
def test_the_dither_check_reads_a_gradient_error_planted_in_the_completions(key: int) -> None:
    """Completions that follow the true equilibrium plus ``e . du``, ``du`` the dither's first-order
    move of the rates: the check, read through the true parameters, recovers the zero-sum part of
    ``e`` within 4 of its own standard errors in every zone (each 12 zones' worth of normal reads
    passes so with probability 1 - 8e-4), and rejects. Unplanted, it does not."""
    market = SharedStateMarket(seed=3)
    logs = market.generate_logs(400, jax.random.key(key), dither=0.3)
    truth = _true_fit(market, logs)
    u, shift = np.asarray(logs["u"]), np.asarray(logs["dither"])
    total = u.sum(axis=1, keepdims=True)
    weights = u * np.exp(-shift)
    base = total * weights / weights.sum(axis=1, keepdims=True)
    move = base * (shift - (base / total * shift).sum(axis=1, keepdims=True))
    planted = np.linspace(-1.5, 1.5, market.n_zones)
    y = np.asarray(logs["y"]).copy()
    y[:, 0] += move @ planted
    read = dither_check(logs | {"y": y}, truth)
    se = np.sqrt(np.diag(read.covariance))
    assert np.all(np.abs(read.error - (planted - planted.mean())) <= 4.0 * se)
    assert read.p_value == 1.0 / 2000.0  # no bootstrap draw reaches it
    assert dither_check(logs | {"y": y}, truth, reference="f").p_value < 1e-6
    assert abs(math.fsum(read.error)) <= 1e-12
    assert dither_check(logs, truth).p_value > 0.01


def _hc3_wald(design: np.ndarray, residual: np.ndarray) -> float:
    """The HC3 Wald statistic of the slopes, every coefficient but the intercept, by least squares
    and the sandwich written out: the check's statistic read another way."""
    coefficients, *_ = np.linalg.lstsq(design, residual, rcond=None)
    bread = np.linalg.inv(design.T @ design)
    leverage = np.array([row @ bread @ row for row in design])
    scaled = (residual - design @ coefficients) / (1.0 - leverage)
    meat = sum(np.outer(row, row) * e**2 for row, e in zip(design, scaled, strict=True))
    covariance = (bread @ meat @ bread)[1:, 1:]
    return float(coefficients[1:] @ np.linalg.solve(covariance, coefficients[1:]))


def test_the_dither_check_refers_its_wald_to_a_wild_bootstrap_by_default() -> None:
    """Referred to F, the HC3 Wald rejected the true fit on 14.5% of logs of 25 blocks at 5%, on
    12.5% of 50 (market 3): the sandwich's own noise, with 12 regressors, is not small there. By
    default the p-value is now the share of a wild bootstrap's statistics at or above the logs'
    (Rademacher signs on the residuals about their mean, the null's fit), ``(1 + count) / (1 +
    draws)``: here rebuilt draw by draw, each statistic by least squares, from the same signs."""
    market = SharedStateMarket(seed=3)
    logs = market.generate_logs(48, jax.random.key(8), dither=0.3)
    truth = _true_fit(market, logs)
    draws, seed = 99, 5
    read = dither_check(logs, truth, draws=draws, seed=seed)
    u, shift = np.asarray(logs["u"]), np.asarray(logs["dither"])
    total = u.sum(axis=1, keepdims=True)
    weights = u * np.exp(-shift)
    base = total * weights / weights.sum(axis=1, keepdims=True)
    move = base * (shift - (base / total * shift).sum(axis=1, keepdims=True))
    completed = _fitted_completions(truth)
    predicted = np.asarray(jax.vmap(completed)(logs["demand"], logs["u"]), np.float64)
    residual = np.asarray(logs["y"]).sum(axis=1) - predicted
    design = np.column_stack([np.ones(len(residual)), move[:, :-1]])
    wald = _hc3_wald(design, residual)
    assert read.wald == pytest.approx(wald, rel=1e-9, abs=0.0)
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(draws, len(residual)))
    centred = residual - residual.mean()
    boot = [_hc3_wald(design, residual.mean() + sign * centred) for sign in signs]
    beyond = sum(b >= wald * (1.0 - 1e-9) for b in boot)
    assert sum(b >= wald * (1.0 + 1e-9) for b in boot) == beyond  # no tie within rounding
    assert read.p_value == (1 + beyond) / (draws + 1)
    f = dither_check(logs, truth, reference="f")
    assert f.p_value == pytest.approx(float(stats.f.sf(wald / 11, 11, 48 - 12)), rel=1e-9, abs=0.0)
    assert f.wald == read.wald
    assert dither_check(logs, truth, draws=draws, seed=seed).p_value == read.p_value


def test_the_dither_check_refuses_logs_it_cannot_read(env: dict) -> None:
    market = SharedStateMarket(seed=3)
    logs = market.generate_logs(44, jax.random.key(4), dither=0.3)
    truth = _true_fit(market, logs)
    with pytest.raises(ValueError, match="carry no dither"):
        dither_check(env["logs"], truth)
    with pytest.raises(ValueError, match="carry no dither"):
        dither_check(logs | {"dither": jnp.zeros_like(logs["dither"])}, truth)
    with pytest.raises(ValueError, match="blocks for 12 zones"):
        dither_check({name: column[:12] for name, column in logs.items()}, truth)
    with pytest.raises(ValueError, match="below 4 blocks a direction"):
        dither_check({name: column[:43] for name, column in logs.items()}, truth)
    with pytest.raises(ValueError, match="zones and the logs"):
        dither_check({name: column[:, :11] for name, column in logs.items()}, truth)
    with pytest.raises(ValueError, match="0 < level < power < 1"):
        dither_check(logs, truth, level=0.9, power=0.8)
    with pytest.raises(ValueError, match="reference is"):
        dither_check(logs, truth, reference="chi2")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="draws is"):
        dither_check(logs, truth, draws=0)


@pytest.fixture(scope="module")
def pilot() -> dict:
    market = SharedStateMarket(seed=3)
    logs = market.generate_logs(200, jax.random.key(5))
    return {"market": market, "logs": logs, "truth": _true_fit(market, logs)}


def test_the_dither_s_size_falls_as_the_root_of_the_blocks_and_as_the_error(pilot: dict) -> None:
    """``h = sqrt(lambda * rho / N) / error``: four times the blocks halve it, as does twice the
    error, where it is affordable."""
    read = functools.partial(dither_size, pilot["logs"], pilot["truth"], max_cost=1.0)
    base = read(error=1.0, blocks=400)
    assert base.affordable
    assert base.blocks == 400
    assert read(error=1.0, blocks=1600).size == pytest.approx(base.size / 2.0, rel=1e-12, abs=0.0)
    assert read(error=2.0, blocks=400).size == pytest.approx(base.size / 2.0, rel=1e-12, abs=0.0)


def test_the_dither_is_not_sized_for_a_study_the_check_would_refuse(pilot: dict) -> None:
    with pytest.raises(ValueError, match="reads at least 4 blocks a direction, 44 here"):
        dither_size(pilot["logs"], pilot["truth"], error=1.0, blocks=43)


def test_a_dither_that_costs_too_much_is_cut_and_the_blocks_grow(pilot: dict) -> None:
    """Where the size that finds the error costs more than the cap, the size is cut to what costs
    the cap and the blocks grow by the square of the cut: a capped perturbation is a rate
    constraint, so the remedy is duration."""
    free = dither_size(pilot["logs"], pilot["truth"], error=0.5, blocks=400, max_cost=1.0)
    cap = 0.25 * free.cost
    capped = dither_size(pilot["logs"], pilot["truth"], error=0.5, blocks=400, max_cost=cap)
    assert not capped.affordable
    assert capped.size < free.size
    assert 0.9 * cap <= capped.cost <= cap
    assert capped.blocks == math.ceil(400 * (free.size / capped.size) ** 2)


def test_the_dither_s_size_reads_the_detectable_error_it_was_sized_for(pilot: dict) -> None:
    """Logs dithered at the size :func:`dither_size` returns for an error of 0.5 at 400 blocks:
    the check on them reads a least detectable error near 0.5. Over five logs of each of markets
    3 and 7 it read 0.51 to 0.70, above the target by 14% at the median: the rule reads the pilot's
    undithered residuals, the check the dithered logs' HC3 sandwich."""
    market = pilot["market"]
    design = dither_size(pilot["logs"], pilot["truth"], error=0.5, blocks=400, max_cost=1.0)
    logs = market.generate_logs(400, jax.random.key(6), dither=design.size)
    read = dither_check(logs, _true_fit(market, logs))
    assert 0.4 <= read.minimum_detectable <= 0.75


def test_the_per_driver_market_pays_each_driver_and_its_oracle_is_the_convex_optimum() -> None:
    """Paid per driver, the money is ``sum(u * x)``; the oracle is
    :func:`chc.games.per_driver_congestion_allocation`'s, whose drivers at the market's own
    equilibrium are paid the budget at most (to the solve's tolerance of 1e-6 of the drivers) and
    complete the rides it says; the lump-sum oracle on the same market completes more for the
    same money as a sum of rates."""
    market = SharedStateMarket(seed=1, payment="per_driver")
    demand, attract = market._base()
    u = jnp.linspace(0.1, 0.5, market.n_zones)
    x = market._equilibrium(attract, u)
    assert float(market.spend(u)) == pytest.approx(float(jnp.dot(u, x)), rel=1e-15, abs=0.0)
    lump = dataclasses.replace(market, payment="lump_sum")
    assert float(lump.spend(u)) == pytest.approx(float(jnp.sum(u)), rel=1e-15, abs=0.0)
    found = per_driver_congestion_allocation(
        attract, demand, market.beta, market.congestion, market.mass, market.budget
    )
    oracle = market.oracle_allocation()
    np.testing.assert_array_equal(np.asarray(oracle), found.allocation)
    assert float(market.spend(oracle)) <= market.budget * (1.0 + 1e-5)
    assert market.value(oracle) == pytest.approx(found.value, abs=2e-5)
    assert market.value(lump.oracle_allocation()) > found.value


def test_per_driver_logs_pay_each_block_its_scale_of_the_budget() -> None:
    """Each block's rates are scaled so that its drivers are paid the block's budget scale times
    the budget, logged as ``spend``; the lump-sum logs carry no ``spend``."""
    market = SharedStateMarket(seed=1, payment="per_driver")
    key = jax.random.key(7)
    logs = market.generate_logs(30, key)
    scale = jnp.stack(
        [
            0.4 + 1.2 * jax.random.uniform(jax.random.split(k, 3)[1])
            for k in jax.random.split(key, 30)
        ]
    )
    np.testing.assert_allclose(logs["spend"][:, 0], scale * market.budget, rtol=1e-5, atol=0.0)
    np.testing.assert_allclose(logs["aggregate"][:, 0], jnp.sum(logs["u"], axis=1), rtol=1e-14)
    assert "spend" not in dataclasses.replace(market, payment="lump_sum").generate_logs(3, key)


def test_a_market_s_payment_is_one_it_knows_and_the_lump_sum_planners_refuse_the_other(
    env: dict,
) -> None:
    with pytest.raises(ValueError, match="'lump_sum' or 'per_driver'"):
        SharedStateMarket(payment="per_ride")  # type: ignore[arg-type]
    market = dataclasses.replace(env["market"], payment="per_driver")
    response = calibrate_shared_state(env["logs"])
    with pytest.raises(ValueError, match="pays per driver"):
        pessimistic_equilibrium_allocation(market, response, equilibrium=env["fitted"])
    with pytest.raises(ValueError, match="pays per driver"):
        sutva_allocation(market, response, 1.0)
    with pytest.raises(ValueError, match="finite non-negative logit shift"):
        env["market"].generate_logs(3, jax.random.key(0), dither=-0.1)
