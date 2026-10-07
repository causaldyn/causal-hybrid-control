"""Offline causal decision under equilibrium interference: the logging policy is confounded and
SUTVA fails, so predictive (MOPO) and naive-causal planners underperform even no incentive, while
the equilibrium-aware, de-confounded, pessimistic CHC allocation recovers the oracle.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chc.marketplace import (
    ExposureResponse,
    SharedStateMarket,
    _zone_slope,
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
    to 0.658 where it reads 0.823, the demand its de-confounded one to 0.542 where 0.359, and the
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
