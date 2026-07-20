"""Offline causal decision under equilibrium interference: the logging policy is confounded and
SUTVA fails, so predictive (MOPO) and naive-causal planners underperform even no incentive, while
the equilibrium-aware, de-confounded, pessimistic CHC allocation recovers the oracle.
"""

import jax
import jax.numpy as jnp
import pytest

from chc.marketplace import (
    SharedStateMarket,
    calibrate_naive_causal,
    calibrate_predictive,
    calibrate_shared_state,
    interference_bias,
    pessimistic_equilibrium_allocation,
    sutva_allocation,
)


@pytest.fixture(scope="module")
def env() -> dict:
    market = SharedStateMarket(seed=0)
    logs = market.generate_logs(400, jax.random.key(1))
    zero = jnp.zeros(market.n_zones)
    oracle_value = market.value(market.oracle_allocation(steps=1200))
    base_value = market.value(zero)
    return {"market": market, "logs": logs, "oracle": oracle_value, "base": base_value}


def test_confounded_calibration_biases_toward_demand(env: dict) -> None:
    demand, _ = env["market"]._base()
    pred = calibrate_predictive(env["logs"]).marginal
    naive = calibrate_naive_causal(env["logs"]).marginal
    pred_corr = float(jnp.corrcoef(pred, demand)[0, 1])
    naive_corr = float(jnp.corrcoef(naive, demand)[0, 1])
    assert pred_corr > 0.25  # confounding: the logging policy chased demand, inflating busy zones
    assert abs(naive_corr) < 0.2  # backdoor on demand de-confounds the response


def test_sutva_planners_capture_almost_none_of_the_oracle_lift(env: dict) -> None:
    market, base, oracle = env["market"], env["base"], env["oracle"]
    pred_alloc = sutva_allocation(market, calibrate_predictive(env["logs"]), radius=1.0)
    naive_alloc = sutva_allocation(market, calibrate_naive_causal(env["logs"]), radius=1.0)
    lift = oracle - base  # the completions the oracle wins over doing nothing
    # SUTVA over-allocates into zones whose drivers only cannibalise -> captures <30% (often < 0)
    assert market.value(pred_alloc) - base < 0.3 * lift
    assert market.value(naive_alloc) - base < 0.3 * lift


def test_chc_recovers_the_oracle_where_baselines_do_not(env: dict) -> None:
    market, oracle = env["market"], env["oracle"]
    shared = calibrate_shared_state(env["logs"])
    chc = pessimistic_equilibrium_allocation(market, shared, radius=1.0)
    naive = sutva_allocation(market, calibrate_naive_causal(env["logs"]), radius=1.0)
    pred = sutva_allocation(market, calibrate_predictive(env["logs"]), radius=1.0)
    assert oracle - market.value(chc) < 0.3  # equilibrium-aware + de-confounded ~ recovers oracle
    assert oracle - market.value(naive) > 0.8  # SUTVA leaves large regret on the table
    assert oracle - market.value(pred) > 0.8


def test_naive_over_predicts_while_chc_delivers_more(env: dict) -> None:
    market = env["market"]
    naive_resp = calibrate_naive_causal(env["logs"])
    naive_alloc = sutva_allocation(market, naive_resp, radius=1.0)
    chc_alloc = pessimistic_equilibrium_allocation(
        market, calibrate_shared_state(env["logs"]), radius=1.0
    )
    # the naive per-zone (additive) model predicts a lift the equilibrium never realises...
    assert interference_bias(market, naive_resp, naive_alloc) > 1.0
    # ...and CHC, planning through that equilibrium, realises strictly more completions than naive
    assert market.value(chc_alloc) > market.value(naive_alloc) + 1.0
