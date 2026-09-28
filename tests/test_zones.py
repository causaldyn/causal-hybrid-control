"""The zone market: the closed forms it is parameterised by, its bridges, and the confounding its
logs carry.

Six logs were read before any threshold on a fit was written. Against one period's response, the
fit that adjusts for the shock read the price at 0.78-1.16 and the incentive at 0.90-1.08 of it,
zone by zone; the fit that does not read the price at -0.71 to 0.21 of it (a price rise raising
demand in 18 of the 24 zones) and the incentive at -0.08 to 0.63, 0.18-0.39 averaged over the city.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chc.dynamics_id import fit_causal_residual
from chc.integrate import rk4_step, rollout
from chc.zones import ZoneMarketSystem

K = 4


def _transitions(system: ZoneMarketSystem, logs: dict[str, np.ndarray]) -> dict[str, jax.Array]:
    """Consecutive periods of a day as ``(x, u, x_next)``, plus each zone's shock."""
    day = logs["day"]
    current = np.flatnonzero(np.r_[day[1:] == day[:-1], False])
    x = np.stack([logs[name] for name in system.state_columns], 1)
    u = np.stack([logs[name] for name in system.lever_columns], 1)
    data = {"x": jnp.asarray(x[current]), "u": jnp.asarray(u[current])}
    data["x_next"] = jnp.asarray(x[current + 1])
    for name in system.shock_columns:
        data[name] = jnp.asarray(logs[name][current][:, None])
    return data


@pytest.mark.parametrize("matching", ["harmonic", "linear"])
def test_the_do_nothing_point_is_the_steady_state_under_both_laws(matching: str) -> None:
    system = ZoneMarketSystem(matching=matching)  # type: ignore[arg-type]
    rate = system.plant()(0.0, system.do_nothing, jnp.zeros(2 * K))
    assert float(jnp.max(jnp.abs(rate))) < 1e-12


def test_the_linear_law_is_the_harmonic_tangent_and_passes_through_zero() -> None:
    harmonic, linear = ZoneMarketSystem().plant(), ZoneMarketSystem(matching="linear").plant()
    x0 = ZoneMarketSystem().do_nothing
    assert jnp.allclose(jax.jacfwd(harmonic.trips)(x0), jax.jacfwd(linear.trips)(x0), atol=1e-14)
    assert jnp.allclose(harmonic.trips(x0), linear.trips(x0), atol=1e-14)
    assert float(jnp.max(jnp.abs(linear.trips(jnp.zeros(3 * K))))) == 0.0


def test_the_plant_is_control_affine_with_the_channel_it_states() -> None:
    system = ZoneMarketSystem()
    plant = system.plant(shock=jnp.array([0.2, -0.1, 0.0, 0.3]))
    transfer, recruit = np.asarray(plant.transfer), np.asarray(plant.recruit)
    channel = np.zeros((3 * K, 2 * K))
    channel[:K, :K] = transfer * recruit
    channel[K : 2 * K, K:] = -np.diag(np.asarray(plant.demand * plant.elasticity))
    channel[2 * K :, :K] = np.eye(K)
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = jnp.asarray(np.asarray(system.do_nothing) + rng.uniform(0.0, 3.0, 3 * K))
        u = jnp.asarray(rng.uniform(0.0, 1.0, 2 * K))
        moved = plant(0.0, x, u) - plant(0.0, x, jnp.zeros(2 * K))
        assert np.allclose(np.asarray(moved), channel @ np.asarray(u), atol=1e-12)


def test_the_city_keeps_one_minus_spill_of_what_an_incentive_recruits() -> None:
    """Interference, conserved: a zone gains its recruits and its two neighbours lose ``spill`` of
    them between them, so every column of the supply channel sums to ``(1 - spill) recruit``."""
    for spill in (0.0, 0.5, 1.0):
        system = ZoneMarketSystem(spill=spill)
        plant = system.plant()
        supply_channel = np.asarray(plant.transfer) * np.asarray(plant.recruit)
        assert np.allclose(supply_channel.sum(axis=0), (1.0 - spill) * np.asarray(system.recruit))
        assert np.allclose(np.diag(supply_channel), system.recruit)


def test_the_known_rows_are_the_plant_s_own_stock_rows() -> None:
    system = ZoneMarketSystem()
    known, plant = system.stock_dynamics(), system.plant()
    x, u = system.do_nothing + 1.0, jnp.full(2 * K, 0.4)
    stock = slice(2 * K, 3 * K)
    assert jnp.allclose(jax.jacfwd(plant, argnums=1)(0.0, x, u)[stock], known.a_matrix[stock])
    assert jnp.allclose(jax.jacfwd(plant, argnums=2)(0.0, x, u)[stock], known.b_matrix[stock])
    assert float(jnp.max(jnp.abs(known.a_matrix[: 2 * K]))) == 0.0
    assert float(jnp.max(jnp.abs(known.b_matrix[: 2 * K]))) == 0.0


def _settled_trips(system: ZoneMarketSystem, incentive: np.ndarray) -> np.ndarray:
    u = jnp.concatenate([jnp.asarray(incentive), jnp.zeros(K)])
    trajectory = rollout(system.plant(), system.do_nothing, jnp.tile(u, (600, 1)), 1.0)
    assert float(jnp.max(jnp.abs(trajectory[-1] - trajectory[-2]))) < 1e-12  # settled
    return np.asarray(system.plant().trips(trajectory[-1]))


def test_the_one_shot_decision_is_the_settled_trips_under_linear_matching() -> None:
    system = ZoneMarketSystem(matching="linear")
    decision = system.zone_decision(
        target=np.zeros(K), state_weight=np.eye(K), action_weight=np.eye(K)
    )
    incentive = np.array([1.0, 0.5, 0.0, 0.8])
    predicted = decision.baseline + (decision.coupling * system.incentive_channel) @ incentive
    assert np.allclose(_settled_trips(system, incentive), predicted, atol=1e-12)


def test_the_one_shot_decision_is_first_order_under_harmonic_matching() -> None:
    """Its error, relative to the move it predicts, halves with the incentive: 3.5%, 6.8% and 12.8%
    at 0.025, 0.05 and 0.1 held in three zones."""
    system = ZoneMarketSystem()
    decision = system.zone_decision(
        target=np.zeros(K), state_weight=np.eye(K), action_weight=np.eye(K)
    )
    errors = []
    for level in (0.025, 0.05, 0.1):
        incentive = level * np.array([1.0, 0.5, 0.0, 0.8])
        move = (decision.coupling * system.incentive_channel) @ incentive
        gap = _settled_trips(system, incentive) - decision.baseline - move
        errors.append(np.max(np.abs(gap)) / np.max(np.abs(move)))
    assert errors[0] < 0.05
    assert 1.7 < errors[1] / errors[0] < 2.3
    assert 1.7 < errors[2] / errors[1] < 2.3


def test_the_linear_gaussian_plant_is_one_period_of_the_log_under_linear_matching() -> None:
    """With no supply or queue noise, what the plant's ``a``, ``b`` and ``offset`` leave of each
    logged period is the shock's response and nothing else, and its covariance is ``noise``."""
    system = ZoneMarketSystem(matching="linear", noise=0.0)
    plant = system.linear_gaussian(1.0)
    data = _transitions(system, system.sample(n_days=10, seed=3))
    x, u, x_next = (np.asarray(data[name]) for name in ("x", "u", "x_next"))
    left = x_next - x @ plant.a.T - u @ plant.b.T - plant.offset
    shocks = np.hstack([np.asarray(data[name]) for name in system.shock_columns])
    response, *_ = np.linalg.lstsq(shocks, left, rcond=None)
    assert np.max(np.abs(left - shocks @ response)) < 1e-12
    assert np.allclose(plant.noise, system.shock_sd**2 * response.T @ response, atol=1e-12)


def test_the_logged_operator_chased_each_zone_s_shock() -> None:
    system = ZoneMarketSystem()
    logs = system.sample(n_days=10, seed=0)
    assert set(logs) == {"day", "period", *system.state_columns, *system.lever_columns} | set(
        system.shock_columns
    )
    assert all(logs[name].shape == (10 * 48,) for name in logs)
    for i in range(K):
        for lever in (f"incentive_{i}", f"price_{i}"):
            assert logs[lever].min() >= 0.0
            assert logs[lever].max() <= 1.0
            assert np.corrcoef(logs[lever], logs[f"shock_{i}"])[0, 1] > 0.8


def test_the_graph_adjusts_every_lever_for_every_zone_s_shock() -> None:
    system = ZoneMarketSystem()
    for outcome in ("queue_0", "supply_2"):
        adjustment = system.graph().adjustment_set(treatment=system.lever_columns, outcome=outcome)
        assert adjustment.status == "identified"
        assert sorted(adjustment.covariates) == sorted(system.shock_columns)


@pytest.fixture(scope="module")
def fits() -> dict[str, np.ndarray]:
    """The fitted channel at the do-nothing point over one period's response, both ways."""
    system = ZoneMarketSystem()
    data = _transitions(system, system.sample(seed=0))
    step = system.linear_gaussian(1.0).b
    ratios = {}
    for name, adjust_for in (("naive", ()), ("adjusted", system.shock_columns)):
        fit = fit_causal_residual(
            system.stock_dynamics(), data, 1.0, adjust_for=adjust_for, integrator="euler", seed=0
        )
        channel = np.asarray(fit.residual.control_channel(system.do_nothing))
        ratios[f"{name}_price"] = np.diag(channel[K : 2 * K, K:]) / np.diag(step[K : 2 * K, K:])
        ratios[f"{name}_incentive"] = np.diag(channel[:K, :K]) / np.diag(step[:K, :K])
    return ratios


def test_a_fit_that_ignores_the_shock_all_but_loses_the_price_and_the_incentive(
    fits: dict[str, np.ndarray],
) -> None:
    assert np.all(fits["naive_price"] < 0.35)
    assert fits["naive_price"].mean() < 0.2
    assert fits["naive_incentive"].mean() < 0.5


def test_a_fit_that_adjusts_for_the_shock_reads_one_period_s_response(
    fits: dict[str, np.ndarray],
) -> None:
    assert np.all((fits["adjusted_price"] > 0.7) & (fits["adjusted_price"] < 1.3))
    assert np.all((fits["adjusted_incentive"] > 0.8) & (fits["adjusted_incentive"] < 1.2))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"queue": (4.0, 3.0, 3.0)}, "share one non-zero length"),
        ({"supply": (8.0, -6.0, 6.0, 4.0)}, "supply must be positive"),
        ({"elasticity": (0.5, 1.0, 0.5, 0.5)}, "elasticity must lie in"),
        ({"spill": 1.5}, "spill must lie in"),
        ({"matching": "cobb-douglas"}, "matching must be"),
        (
            {"supply": (8.0,), "queue": (4.0,), "recruit": (2.0,), "carry": (0.6,)}
            | {"elasticity": (0.5,)},
            "no neighbour",
        ),
    ],
)
def test_a_market_that_cannot_exist_is_refused(change: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ZoneMarketSystem(**change)  # type: ignore[arg-type]


def test_the_rk4_step_the_log_uses_is_the_one_the_bridge_differentiates() -> None:
    """``linear_gaussian`` differentiates one RK4 period of this plant; one period of the log with
    no noise and no shock is that same step."""
    system = ZoneMarketSystem(noise=0.0, shock_sd=0.0)
    logs = system.sample(n_days=1, n_periods=2, seed=0)
    x = jnp.asarray([logs[name][0] for name in system.state_columns])
    u = jnp.asarray([logs[name][0] for name in system.lever_columns])
    x_next = np.array([logs[name][1] for name in system.state_columns])
    assert np.allclose(np.asarray(rk4_step(system.plant(), 0.0, x, u, 1.0)), x_next, atol=1e-12)
