"""Exogenous drivers: a push the plan cannot move but can see coming.

The load-bearing tests are the two oracles. On the true plant, the plan made against the forecast is
the minimiser of a quadratic program written out here by hand -- the RK4 map with the driver moving
linearly across each step, stacked and solved in closed form, sharing no code with
:class:`~chc.dynamics.DrivenDynamics` -- and the same forecast read one step late is not. Through
the façade, the schedule prescribed from the logs with the forecast costs what the oracle costs on
the true plant, and the one prescribed without it costs 1.7-2x as much.

The logs come from zones heated against the weather: the logging policy chases the outdoor
temperature, which also pushes the zone, so the weather confounds the heater *and* drives the
state -- the case L8.1 found `prescribe` could not state.
"""

from __future__ import annotations

import logging
import re

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from chc import (
    CausalGraph,
    Constraint,
    DecisionError,
    Driver,
    Lever,
    Panel,
    QuadraticCost,
    RecedingHorizon,
    Target,
    causal_plan,
    fit_causal_residual,
    prescribe,
)
from chc.adjoint import control_gradient_adjoint
from chc.cost import total_cost
from chc.decision import Prescription, _cost
from chc.dynamics import DrivenDynamics, Dynamics, LinearDynamics
from chc.plan import _barrier_terms

DT = 0.1
DECAY, CHANNEL, PUSH = -0.5, 0.8, 1.2  # temp' = DECAY temp + CHANNEL heat + PUSH outdoor
PERIOD = 2.0  # the weather's cycle, in model time
HORIZON = 20
X0 = 0.2
EDGES = [("outdoor", "heat"), ("outdoor", "temp"), ("heat", "temp")]


def _weather(phase: float | np.ndarray, n_levels: int) -> np.ndarray:
    return np.sin(2.0 * np.pi * DT * np.arange(n_levels) / PERIOD + phase)


def _forecast() -> np.ndarray:
    """The weather over the plan, one level at every grid time: ``HORIZON + 1`` of them."""
    return _weather(0.3, HORIZON + 1)


def _step(temp: np.ndarray, heat: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """One RK4 step of the true plant, the heat held and the weather moving from start to end."""
    middle = 0.5 * (start + end)

    def rate(state: np.ndarray, outdoor: np.ndarray) -> np.ndarray:
        return DECAY * state + CHANNEL * heat + PUSH * outdoor

    k1 = rate(temp, start)
    k2 = rate(temp + 0.5 * DT * k1, middle)
    k3 = rate(temp + 0.5 * DT * k2, middle)
    k4 = rate(temp + DT * k3, end)
    return temp + DT / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _simulate(
    n_units: int = 100, n_periods: int = 24, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(temp, heat, outdoor)``, a row per zone: temp and outdoor at every grid time, heat over
    every step, set by a policy that heats against the weather it sees."""
    rng = np.random.default_rng(seed)
    outdoor = _weather(rng.uniform(0.0, 2.0 * np.pi, (n_units, 1)), n_periods + 1)
    temp = np.empty((n_units, n_periods + 1))
    temp[:, 0] = rng.normal(0.0, 0.3, n_units)
    heat = np.empty((n_units, n_periods))
    for k in range(n_periods):
        heat[:, k] = -0.7 * outdoor[:, k] + rng.normal(0.0, 0.5, n_units)
        temp[:, k + 1] = _step(temp[:, k], heat[:, k], outdoor[:, k], outdoor[:, k + 1])
        temp[:, k + 1] += rng.normal(0.0, 0.01, n_units)
    return temp, heat, outdoor


def _panel(seed: int = 0) -> Panel:
    temp, heat, outdoor = _simulate(seed=seed)
    n_units, n_periods = heat.shape
    frame = {
        "unit": np.repeat(np.arange(n_units), n_periods),
        "time": np.tile(np.arange(n_periods), n_units),
        "temp": temp[:, :-1].ravel(),
        "heat": heat.ravel(),
        "outdoor": outdoor[:, :-1].ravel(),
    }
    return Panel.from_frame(frame, unit="unit", time="time", seed=0)


@pytest.fixture(scope="module")
def panel() -> Panel:
    return _panel()


def _true_plant(forecast: np.ndarray, start: float = 0.0) -> DrivenDynamics:
    base = LinearDynamics(jnp.array([[DECAY]]), jnp.array([[CHANNEL]]))
    return DrivenDynamics(base, jnp.array([[PUSH]]), jnp.asarray(forecast)[:, None], DT, start)


# ---- the hold ----


def test_the_hold_passes_through_every_level_and_is_linear_between_and_beyond_them() -> None:
    levels = jnp.array([[0.0, 1.0], [2.0, -1.0], [1.0, 0.0]])
    still = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    plant = DrivenDynamics(still, jnp.ones((1, 2)), levels, 0.5, start=1.0)
    for k in range(3):
        assert jnp.allclose(plant.drivers(1.0 + 0.5 * k), levels[k])
    assert jnp.allclose(plant.drivers(1.25), (levels[0] + levels[1]) / 2)
    assert jnp.allclose(plant.drivers(1.6), levels[1] + 0.2 * (levels[2] - levels[1]))
    # past either end the nearest segment carries on, so a forecast one level short extrapolates
    assert jnp.allclose(plant.drivers(0.5), levels[0] - (levels[1] - levels[0]))
    assert jnp.allclose(plant.drivers(2.5), levels[2] + (levels[2] - levels[1]))


def test_the_drivers_move_the_drift_and_never_the_channel() -> None:
    """What ``gamma*`` is priced on: the forecast lands in the barrier's drift term at each step's
    own time, and the channel ``B^T grad h`` is the undriven plant's."""
    plant = _true_plant(_forecast())
    base = plant.dynamics
    states = jnp.linspace(-0.5, 1.5, HORIZON)[:, None]
    actions = jnp.linspace(1.0, -1.0, HORIZON)[:, None]

    def barrier(x: Array) -> Array:
        return 1.5 - x[0] ** 2

    _, _, drift, channel = _barrier_terms(plant, barrier, states, actions, DT)
    _, _, base_drift, base_channel = _barrier_terms(base, barrier, states, actions, DT)
    push = PUSH * jnp.asarray(_forecast()[:HORIZON])
    assert jnp.allclose(channel, base_channel, atol=1e-12)
    assert jnp.allclose(drift - base_drift, -2.0 * states[:, 0] * push, atol=1e-12)


@pytest.mark.parametrize(
    ("gain", "levels", "dt", "match"),
    [
        (jnp.ones((1, 1)), jnp.ones((1, 1)), 0.1, "K >= 2"),
        (jnp.ones((1, 1)), jnp.ones(3), 0.1, "K >= 2"),
        (jnp.ones((1, 2)), jnp.ones((3, 1)), 0.1, "one column per driver"),
        (jnp.ones((1, 1)), jnp.ones((3, 1)), 0.0, "not a positive grid spacing"),
    ],
)
def test_a_hold_that_cannot_be_read_is_refused(
    gain: Array, levels: Array, dt: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        DrivenDynamics(LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1))), gain, levels, dt)


class _Cubic(eqx.Module):
    """``x' = -0.5 x - 0.4 x^3 + u``: nonlinear, so the RK4 stages' Jacobians feel the push."""

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return -0.5 * x - 0.4 * x**3 + u


def test_the_adjoint_matches_central_differences_on_a_driven_plant() -> None:
    """The ``_forced`` check of ``tests/test_adjoint.py`` with the push read off a forecast: each
    step is differentiated at its own time, since on a nonlinear plant the forecast moves the RK4
    stages and a step read at another time is a different map."""
    cost = QuadraticCost(Q=jnp.eye(1), R=0.05 * jnp.eye(1), Qf=jnp.eye(1), x_target=jnp.ones(1))
    plant = DrivenDynamics(
        _Cubic(), jnp.array([[1.5]]), jnp.asarray(3.0 * _forecast())[:, None], DT
    )
    x0 = jnp.array([0.4])
    us = 0.3 * jax.random.normal(jax.random.key(0), (HORIZON, 1))
    gradient = control_gradient_adjoint(plant, x0, us, DT, cost)
    autodiff = jax.grad(lambda actions: total_cost(plant, x0, actions, DT, cost))(us)
    assert jnp.allclose(gradient, autodiff, atol=1e-10)
    eps = 1e-6
    for k in (0, 7, 13, 19):
        bump = jnp.zeros_like(us).at[k, 0].set(eps)
        up, down = (
            total_cost(plant, x0, us + bump, DT, cost),
            total_cost(plant, x0, us - bump, DT, cost),
        )
        assert abs(float(gradient[k, 0]) - float((up - down) / (2 * eps))) < 1e-7


# ---- the fit ----


def _transitions(temp: np.ndarray, heat: np.ndarray, outdoor: np.ndarray) -> dict[str, Array]:
    def column(values: np.ndarray) -> Array:
        return jnp.asarray(values.reshape(-1, 1))

    return {
        "x": column(temp[:, :-1]),
        "u": column(heat),
        "x_next": column(temp[:, 1:]),
        "outdoor": column(outdoor[:, :-1]),
        "outdoor_next": column(outdoor[:, 1:]),
    }


@pytest.mark.parametrize("units", [1e-15, 1e-6, 1e6, 1e15])
def test_a_driver_s_gain_reads_the_same_in_any_units(units: float) -> None:
    """The driver joins the drift regression in its raw units, under a ridge that was a constant:
    logged in millionths of its units, the driver read a gain of 0.0013 where it reads 1.198, and
    the channel 0.778 where 0.796."""
    data = _transitions(*_simulate())
    scaled = dict(data, outdoor=data["outdoor"] * units, outdoor_next=data["outdoor_next"] * units)
    zero = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    one, other = (
        fit_causal_residual(
            zero, log, DT, adjust_for=("outdoor",), integrator="rk4", drivers=("outdoor",)
        )
        for log in (data, scaled)
    )
    assert one.driver_gain is not None
    assert other.driver_gain is not None
    np.testing.assert_allclose(
        np.asarray(other.driver_gain) * units, np.asarray(one.driver_gain), rtol=0.0, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(other.residual.channel), np.asarray(one.residual.channel), rtol=0.0, atol=1e-6
    )


def test_the_fit_recovers_the_push_and_the_decay_it_used_to_absorb() -> None:
    """Without the driver the drift carries its correlation with the state: over eight seeds the
    decay came back between -0.18 and -0.005 against a true -0.5, and the defect sat at 8x the
    noise floor. With it the gain, the decay and the channel land within 0.012, 0.014 and 0.010 of
    the truth on every seed, and the defect is the floor. Thresholds sit at about twice that."""
    data = _transitions(*_simulate())
    zero = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    driven = fit_causal_residual(
        zero, data, DT, adjust_for=("outdoor",), integrator="rk4", drivers=("outdoor",)
    )
    blind = fit_causal_residual(zero, data, DT, adjust_for=("outdoor",), integrator="rk4")

    assert driven.drivers == ("outdoor",)
    assert driven.driver_gain is not None
    assert blind.driver_gain is None
    assert float(driven.driver_gain[0, 0]) == pytest.approx(PUSH, abs=0.025)
    assert float(driven.residual.drift[0, 1]) == pytest.approx(DECAY, abs=0.03)
    assert float(driven.residual.channel[0, 0, 0]) == pytest.approx(CHANNEL, abs=0.02)
    assert driven.integrator_defect == pytest.approx(0.1, abs=0.01)  # 0.01 noise over dt

    assert abs(float(blind.residual.drift[0, 1]) - DECAY) > 0.25
    assert blind.integrator_defect is not None
    assert blind.integrator_defect > 0.5


@pytest.mark.parametrize(
    ("names", "match"),
    [
        (
            {"adjust_for": ("outdoor",), "drivers": ("outdoor", "outdoor")},
            r"drivers named more than once: \['outdoor'\]",
        ),
        (
            {"drivers": ("outdoor",), "instrument": "outdoor"},
            "the instrument 'outdoor' is also a covariate or a driver",
        ),
        (
            {"drivers": ("outdoor",), "instrument": "outdoor_next"},
            "the instrument 'outdoor_next' is also a covariate or a driver",
        ),
    ],
)
def test_the_fit_refuses_a_driver_named_twice_or_as_the_instrument(names: dict, match: str) -> None:
    """Named twice, a driver was fitted as two, its gain split between the copies; as the
    instrument, it was partialled out with the drivers, at the step's start or its end, and lent
    the action no move of its own."""
    data = _transitions(*_simulate())
    zero = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    with pytest.raises(ValueError, match=match):
        fit_causal_residual(zero, data, DT, integrator="rk4", **names)


def test_a_policy_that_reads_the_forecast_is_adjusted_through_the_driver_s_next_level() -> None:
    """A logger that heats against where the weather is going sets the action on the driver's
    next level, which pushes the zone across the whole step: read at the start alone, the driver
    leaves that path open. Under ``euler``, with no correction to hide it, the channel came back
    0.656 that way over six seeds, against the Euler map's own 0.780 with both levels adjusted."""
    rng = np.random.default_rng(0)
    n_units, n_periods = 100, 24
    walk = rng.normal(0.0, 0.3, (n_units, n_periods))
    outdoor = np.concatenate([rng.normal(0.0, 1.0, (n_units, 1)), walk], axis=1).cumsum(axis=1)
    temp = np.empty((n_units, n_periods + 1))
    temp[:, 0] = rng.normal(0.0, 0.3, n_units)
    heat = np.empty((n_units, n_periods))
    for k in range(n_periods):
        heat[:, k] = -0.7 * outdoor[:, k + 1] + rng.normal(0.0, 0.5, n_units)
        temp[:, k + 1] = _step(temp[:, k], heat[:, k], outdoor[:, k], outdoor[:, k + 1])
        temp[:, k + 1] += rng.normal(0.0, 0.01, n_units)
    zero = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    fit = fit_causal_residual(
        zero, _transitions(temp, heat, outdoor), DT, adjust_for=("outdoor",), drivers=("outdoor",)
    )
    z = DECAY * DT
    euler_channel = CHANNEL * (1.0 + z / 2 + z**2 / 6 + z**3 / 24)  # RK4's gain on u, read per dt
    assert float(fit.residual.channel[0, 0, 0]) == pytest.approx(euler_channel, abs=0.015)


# ---- the plan ----


def _oracle(
    forecast: np.ndarray, x0: float, weight: float, price: float, goal: float
) -> np.ndarray:
    """The unconstrained optimum on the true plant, from the RK4 map written out and stacked.

    The map is affine, ``x_{k+1} = phi x_k + gamma u_k + c_k``, so the states are ``f + M u`` and
    the Bolza cost is a quadratic in ``u`` whose normal equations are solved directly.
    """
    zeros = np.zeros(1)
    phi = _step(np.ones(1), zeros, zeros, zeros)[0]
    gamma = _step(zeros, np.ones(1), zeros, zeros)[0]
    steps = len(forecast) - 1
    offset = [
        _step(zeros, zeros, forecast[k : k + 1], forecast[k + 1 : k + 2])[0] for k in range(steps)
    ]
    reach, free = np.zeros((steps + 1, steps)), np.zeros(steps + 1)
    free[0] = x0
    for k in range(steps):
        reach[k + 1] = phi * reach[k]
        reach[k + 1, k] += gamma
        free[k + 1] = phi * free[k] + offset[k]
    weights = np.diag(np.full(steps + 1, weight))  # running on x_0..x_{H-1}, terminal on x_H
    hessian = reach.T @ weights @ reach + price * np.eye(steps)
    return np.linalg.solve(hessian, -reach.T @ weights @ (free - goal))


def _tracking_cost() -> QuadraticCost:
    return QuadraticCost(Q=jnp.eye(1), R=0.05 * jnp.eye(1), Qf=jnp.eye(1), x_target=jnp.ones(1))


def test_a_known_forecast_plans_the_oracle_schedule() -> None:
    """The plan agrees with the hand-written optimum to the solver's tolerance (0.001 measured,
    against actions up to 2.3), and each way of misreading the forecast misses it by far more:
    one step late by 0.27, not at all by 0.98."""
    oracle = _oracle(_forecast(), X0, 1.0, 0.05, 1.0)
    x0, cost = jnp.array([X0]), _tracking_cost()

    def gap(plant: Dynamics) -> float:
        plan = causal_plan(plant, x0, cost, DT, HORIZON, -50.0, 50.0)
        return float(np.max(np.abs(np.asarray(plan.actions)[:, 0] - oracle)))

    assert gap(_true_plant(_forecast())) < 0.01
    assert gap(_true_plant(_forecast(), start=DT)) > 0.1
    assert gap(_true_plant(_forecast()).dynamics) > 0.5


def test_a_window_started_at_t_is_the_tail_of_the_driven_plan() -> None:
    """Bellman through the forecast: from the full plan's own state at step ``j``, the window over
    the steps left, started at ``t = j dt``, reads the forecast from level ``j`` on and returns the
    full plan's tail. Started at the default ``t = 0`` it reads the forecast from the top."""
    plant, cost, j = _true_plant(_forecast()), _tracking_cost(), 7
    full = causal_plan(plant, jnp.array([X0]), cost, DT, HORIZON, -50.0, 50.0)
    window = RecedingHorizon(plant, cost, DT, HORIZON - j, -50.0, 50.0)
    tail = window.step(full.trajectory[j], t=j * DT)
    assert float(jnp.max(jnp.abs(tail.actions - full.actions[j:]))) < 0.01
    unclocked = RecedingHorizon(plant, cost, DT, HORIZON - j, -50.0, 50.0).step(full.trajectory[j])
    assert float(jnp.max(jnp.abs(unclocked.actions - full.actions[j:]))) > 0.5


# ---- the façade ----

LEVER = Lever("heat", lo=-3.0, hi=3.0, unit_cost=0.05)
TARGET = Target("temp", value=1.0)


def _prescribe(panel: Panel, **kwargs: object):
    return prescribe(
        panel,
        levers=[LEVER],
        target=TARGET,
        horizon=HORIZON,
        adjustment=CausalGraph.from_edges(EDGES),
        dt=DT,
        x0=jnp.array([X0]),
        **kwargs,  # type: ignore[arg-type]
    )


def test_the_forecast_pays_on_the_true_plant(panel: Panel) -> None:
    """Both schedules scored on the true plant under the weather that arrives. Over four seeds the
    one made with the forecast cost 1.0001-1.0036x the oracle and the one made without it
    1.69-1.98x."""
    cost = _cost(("temp",), [LEVER], TARGET, HORIZON)
    truth, x0 = _true_plant(_forecast()), jnp.array([X0])
    oracle = causal_plan(truth, x0, cost, DT, HORIZON, jnp.array([-3.0]), jnp.array([3.0]))

    def regret(result: Prescription) -> float:
        assert result.plan is not None
        actions = result.plan.actions
        return float(
            total_cost(truth, x0, actions, DT, cost)
            / total_cost(truth, x0, oracle.actions, DT, cost)
        )

    driven = _prescribe(panel, drivers=[Driver("outdoor", _forecast())])
    assert driven.certificate.identification == "identified"
    assert regret(driven) < 1.02
    assert regret(_prescribe(panel)) > 1.5


def test_prescribe_states_all_three_things_l8_1_could_not(panel: Panel) -> None:
    """A band on the steered state (ADR 0005), a target that moves within the horizon (ADR 0006)
    and the weather in the forecast, in one call, with the barrier priced on the driven drift."""
    rising = np.linspace(0.4, 1.0, HORIZON)
    result = prescribe(
        panel,
        levers=[LEVER],
        target=Target("temp", value=rising),
        constraints=[Constraint("temp", lo=-0.5, hi=1.2)],
        horizon=HORIZON,
        adjustment=CausalGraph.from_edges(EDGES),
        dt=DT,
        x0=jnp.array([X0]),
        drivers=[Driver("outdoor", _forecast())],
    )
    assert result.certificate.identification == "identified"
    assert result.plan is not None
    assert result.certificate.gamma_star is not None
    trajectory = np.asarray(result.plan.trajectory)[:, 0]
    assert trajectory.min() >= -0.5 - 1e-6
    assert trajectory.max() <= 1.2 + 1e-6
    pushes = re.search(r"push on `temp`'s rate: `outdoor` ([+-][0-9.]+)", result.report())
    assert pushes is not None
    assert float(pushes.group(1)) == pytest.approx(PUSH, abs=0.03)
    assert result.to_json()["drivers"] == [{"name": "outdoor", "forecast": _forecast().tolist()}]


@pytest.mark.parametrize(
    ("drivers", "error", "match"),
    [
        ([Driver("outdoor", _forecast())] * 2, DecisionError, "more than once"),
        ([Driver("heat", _forecast())], DecisionError, "a lever or a state"),
        ([Driver("temp", _forecast())], DecisionError, "a lever or a state"),
        ([Driver("outdoor", _forecast()[:-1])], DecisionError, "21 for this horizon"),
        ([Driver("outdoor", np.full(HORIZON + 1, np.nan))], DecisionError, "not finite"),
        ([Driver("wind", _forecast())], KeyError, "'wind' is not in the panel"),
    ],
)
def test_a_driver_the_decision_cannot_read_is_refused_before_the_fit(
    panel: Panel, drivers: list[Driver], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        _prescribe(panel, drivers=drivers)


def test_a_driver_whose_column_the_transitions_hold_in_another_role_is_refused() -> None:
    """The fit reads each driver by its column's name and a period on as ``f"{name}_next"``, from
    the dict that holds the covariates by their names and the levers as ``u``: a covariate named
    ``outdoor_next`` was replaced by the weather a period on, and a driver named ``u`` replaced the
    lever's column."""
    temp, heat, outdoor = _simulate()
    n_units, n_periods = heat.shape
    frame = {
        "unit": np.repeat(np.arange(n_units), n_periods),
        "time": np.tile(np.arange(n_periods), n_units),
        "temp": temp[:, :-1].ravel(),
        "heat": heat.ravel(),
        "outdoor": outdoor[:, :-1].ravel(),
    }
    noise = np.random.default_rng(1).normal(size=heat.size)
    beside = Panel.from_frame({**frame, "outdoor_next": noise}, unit="unit", time="time")
    kwargs = {"levers": [LEVER], "target": TARGET, "horizon": HORIZON, "dt": DT}
    with pytest.raises(
        DecisionError,
        match="the driver 'outdoor' a period on would be read as 'outdoor_next', which already "
        "holds the covariate 'outdoor_next'",
    ):
        prescribe(
            beside,
            adjustment=("outdoor", "outdoor_next"),
            drivers=[Driver("outdoor", _forecast())],
            **kwargs,  # type: ignore[arg-type]
        )
    frame["u"] = frame.pop("outdoor")
    renamed = Panel.from_frame(frame, unit="unit", time="time")
    with pytest.raises(DecisionError, match="the driver 'u' would be read as 'u', which already"):
        prescribe(renamed, adjustment=(), drivers=[Driver("u", _forecast())], **kwargs)  # type: ignore[arg-type]


def test_a_forecast_past_the_logged_range_is_logged_as_extrapolation(
    panel: Panel, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="chc.decision"):
        _prescribe(panel, drivers=[Driver("outdoor", 2.0 * _forecast())])
    warned = [
        record for record in caplog.records if getattr(record, "chc_event", "") == "driver_range"
    ]
    assert len(warned) == 1
    assert warned[0].levelno == logging.WARNING
    assert getattr(warned[0], "forecast", None) == pytest.approx([-2.0 * 1.0, 2.0 * 1.0], abs=0.05)
