"""A discrete-time panel, fitted to its one-step map and planned by it (ADR 0118).

A log whose outcome keeps 0.1 of itself from one period to the next has no field whose RK4 step
reproduces it: RK4's growth factor never falls below 0.2704 (``validation/euler_tube.mac``), and
the ``rk4`` fit raises. Under ``integrator="euler"`` the fit reads the log's one-step map, and the
plan, its tube, its barrier's audit, its regret bound and its rollouts step the field by
``x + dt f(x, u)``, the map it was fitted to: never by RK4, the bias an Euler fit planned by RK4
carried up to 0.18.0.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from chc import CausalGraph, Constraint, Driver, Lever, Panel, Target, prescribe
from chc.adjoint import costate_norms, perturbation_cost_weights
from chc.cost import QuadraticCost
from chc.dynamics import HybridDynamics, LinearDynamics, _SetAtStep
from chc.integrate import rk4_step, rollout
from chc.misspecification import _parameter_cost

GRAPH = CausalGraph.from_edges([("z", "u"), ("z", "x"), ("u", "x")])


def _log(persistence: float, seed: int, units: int = 40, periods: int = 12) -> Panel:
    """``x' = persistence x + 0.5 u + 0.3 z + e`` per period, ``z`` a logged confounder and
    ``u = 0.5 z`` plus a draw of its own, ``e`` of spread 0.1."""
    rng = np.random.default_rng(seed)
    rows: dict[str, list[float]] = {name: [] for name in ("unit", "time", "x", "u", "z")}
    for unit in range(units):
        x = rng.normal()
        for period in range(periods):
            z = rng.normal()
            u = 0.5 * z + rng.normal()
            for name, value in zip(rows, (unit, period, x, u, z), strict=True):
                rows[name].append(value)
            x = persistence * x + 0.5 * u + 0.3 * z + rng.normal(0.0, 0.1)
    return Panel.from_frame(
        {name: np.asarray(values) for name, values in rows.items()},
        unit="unit",
        time="time",
        seed=seed,
    )


def _decide(panel: Panel, **kwargs: object):
    return prescribe(
        panel,
        levers=[Lever("u", lo=-2.0, hi=2.0)],
        target=Target("x", value=1.0),
        adjustment=GRAPH,
        dt=1.0,
        horizon=6,
        tolerance=2.0,
        x0=jnp.zeros(1),
        **kwargs,  # type: ignore[arg-type]
    )


def _euler_path(field, x0, actions, dt: float = 1.0) -> np.ndarray:
    states = [np.asarray(x0, dtype=np.float64)]
    for k, u in enumerate(np.asarray(actions)):
        states.append(states[-1] + dt * np.asarray(field(k * dt, jnp.asarray(states[-1]), u)))
    return np.stack(states)


def test_a_log_that_barely_persists_is_fitted_and_planned_by_its_one_step_map() -> None:
    """The rk4 fit raises on the log and names the remedy. The Euler fit reads the map: the
    channel 0.5 within four of its standard errors, the decay ``0.1 - 1``. The plan is the Euler
    map's rollout of the fitted field, 0.995 at its first step where RK4's rollout of the same
    field reads 0.656, and its tube is carried by the field's own step matrices."""
    panel = _log(0.1, seed=0)
    with pytest.raises(ValueError, match=r"prescribe\(integrator='euler'\)"):
        _decide(panel)
    decided = _decide(panel, integrator="euler")
    fit, plan, certificate = decided.model_fit, decided.plan, decided.certificate
    assert plan is not None
    assert fit.integrator == "euler"
    assert decided.run.integrator == "euler"
    assert fit.channel_error is not None
    channel = np.asarray(fit.residual.channel)
    assert abs(float(channel[0, 0, 0]) - 0.5) < 4.0 * fit.channel_error
    assert float(np.asarray(fit.residual.drift)[0, 1]) == pytest.approx(-0.9, abs=0.05)

    known = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    field = HybridDynamics(known=known, residual=fit.residual)
    planned = np.asarray(plan.trajectory, dtype=np.float64)
    np.testing.assert_allclose(
        planned, _euler_path(field, jnp.zeros(1), plan.actions), rtol=1e-12, atol=1e-14
    )
    by_rk4 = np.asarray(rollout(field, jnp.zeros(1), plan.actions, 1.0))
    assert abs(planned[1, 0] - by_rk4[1, 0]) > 0.3
    assert (certificate.tube_rate, certificate.tube_mode) == ("global", "exact")
    assert certificate.certified_horizon is not None
    assert certificate.regret_bound is not None
    assert math.isfinite(certificate.regret_bound)


def test_a_log_sampled_from_a_slow_plant_is_still_planned_by_rk4() -> None:
    """The default is unchanged: on a log that keeps 0.8 a period the rk4 fit converges, and its
    plan is RK4's rollout of the fitted field, not the Euler map's."""
    decided = _decide(_log(0.8, seed=1))
    plan, fit = decided.plan, decided.model_fit
    assert plan is not None
    assert fit.integrator == "rk4"
    known = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    field = HybridDynamics(known=known, residual=fit.residual)
    planned = np.asarray(plan.trajectory, dtype=np.float64)
    by_rk4 = np.asarray(rollout(field, jnp.zeros(1), plan.actions, 1.0))
    np.testing.assert_allclose(planned, by_rk4, rtol=1e-12, atol=1e-14)
    assert np.max(np.abs(planned - _euler_path(field, jnp.zeros(1), plan.actions))) > 0.01


def test_prescribe_refuses_an_integrator_it_does_not_know() -> None:
    with pytest.raises(Exception, match=r"integrator='midpoint' is neither 'rk4'"):
        _decide(_log(0.1, seed=0), integrator="midpoint")


def _coverage(seed: int, draws: int) -> list[float]:
    """The share of ``draws`` runs of the plant inside the tube at each step of the plan made on
    the log of ``seed``, run on from the plan's start under its actions, the confounder drawn
    afresh each period: the fit adjusts for it and the plan does not forecast it."""
    plan = _decide(_log(0.1, seed=seed), integrator="euler").plan
    assert plan is not None
    assert plan.uncertainty_tube is not None
    tube, planned = np.asarray(plan.uncertainty_tube), np.asarray(plan.trajectory)
    rng = np.random.default_rng(seed)
    x = np.full(draws, planned[0, 0])
    inside = []
    for k, action in enumerate(np.asarray(plan.actions)[:, 0]):
        x = 0.1 * x + 0.5 * action + 0.3 * rng.standard_normal(draws) + rng.normal(0.0, 0.1, draws)
        inside.append(float(np.mean(np.abs(x - planned[k + 1, 0]) <= tube[k + 1])))
    return inside


def test_the_tube_holds_a_plant_that_barely_persists_at_its_level_at_each_step() -> None:
    """Over 16 logs of 2000 runs the plant stays inside the tube at each of six steps 0.95 of the
    time within four standard errors of the mean over the logs, 0.0023 to 0.0028: 0.9475 to
    0.9515. Over 64 logs of 4000 it held 0.9499 to 0.9505, each within 0.0012, where one log's
    share ranges 0.92 to 0.97. The same tube carried by RK4's step matrix of the fitted slope,
    which keeps 0.41 of the gap a step where the Euler map keeps 0.1, held 0.965 to 0.969 from
    the second step on, six of its standard errors and more above the level."""
    shares = np.array([_coverage(seed, 2000) for seed in range(100, 116)])
    mean = shares.mean(axis=0)
    error = shares.std(axis=0, ddof=1) / math.sqrt(shares.shape[0])
    assert np.all(np.abs(mean - 0.95) <= 4.0 * error), (mean, error)


def test_a_forecast_driver_is_read_at_each_step_s_start_as_the_euler_map_reads_it() -> None:
    """With a driver ``w`` pushing ``0.4 w`` a period, the Euler fit reads its gain off the log's
    one-step map, and the plan steps it at the level the forecast holds at each step's start, all
    the Euler map reads, not at its mean over the step."""
    rng = np.random.default_rng(0)
    rows: dict[str, list[float]] = {name: [] for name in ("unit", "time", "x", "u", "z", "w")}
    for unit in range(40):
        x = rng.normal()
        for period in range(12):
            z, w = rng.normal(), rng.normal()
            u = 0.5 * z + rng.normal()
            for name, value in zip(rows, (unit, period, x, u, z, w), strict=True):
                rows[name].append(value)
            x = 0.1 * x + 0.5 * u + 0.3 * z + 0.4 * w + rng.normal(0.0, 0.1)
    forecast = np.array([0.0, 2.0, -1.0, 0.0, 1.0, 0.0, 0.0])
    decided = prescribe(
        Panel.from_frame({k: np.asarray(v) for k, v in rows.items()}, unit="unit", time="time"),
        levers=[Lever("u", lo=-2.0, hi=2.0)],
        target=Target("x", value=1.0),
        adjustment=CausalGraph.from_edges([("z", "u"), ("z", "x"), ("u", "x"), ("w", "x")]),
        dt=1.0,
        horizon=6,
        tolerance=2.0,
        x0=jnp.zeros(1),
        integrator="euler",
        drivers=[Driver("w", forecast)],
    )
    fit, plan = decided.model_fit, decided.plan
    assert plan is not None
    assert fit.driver_gain is not None
    gain = np.asarray(fit.driver_gain)
    assert float(gain[0, 0]) == pytest.approx(0.4, abs=0.03)
    known = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    field = HybridDynamics(known=known, residual=fit.residual)
    path = [np.zeros(1)]
    for k, u in enumerate(np.asarray(plan.actions)):
        rate = np.asarray(field(k * 1.0, jnp.asarray(path[-1]), jnp.asarray(u)))
        path.append(path[-1] + rate + gain[:, 0] * forecast[k])
    np.testing.assert_allclose(np.asarray(plan.trajectory), np.stack(path), rtol=1e-12, atol=1e-14)


def test_a_barrier_held_plan_steps_by_the_map_it_audits() -> None:
    """Held under a bound, the solve's next-state rows and the audit read the plan's own Euler
    path: the bound holds at every step of it."""
    decided = prescribe(
        _log(0.1, seed=0),
        levers=[Lever("u", lo=-2.0, hi=2.0)],
        target=Target("x", value=1.0),
        constraints=[Constraint("x", hi=0.6)],
        hold_constraints=True,
        adjustment=GRAPH,
        dt=1.0,
        horizon=6,
        tolerance=2.0,
        x0=jnp.zeros(1),
        integrator="euler",
    )
    plan, certificate = decided.plan, decided.certificate
    assert plan is not None
    known = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    field = HybridDynamics(known=known, residual=decided.model_fit.residual)
    planned = np.asarray(plan.trajectory, dtype=np.float64)
    np.testing.assert_allclose(
        planned, _euler_path(field, jnp.zeros(1), plan.actions), rtol=1e-12, atol=1e-14
    )
    assert np.all(planned[:, 0] <= 0.6 + 1e-9)
    assert certificate.barrier_certified_steps == 6


def test_the_misspecification_envelope_steps_the_field_as_the_plan_does() -> None:
    """The envelope a plan's regret in its fit's error is read through rebuilds the model with the
    residual moved: under the Euler map it is stepped by that map, so unmoved it prices the plan's
    own task cost."""
    decided = _decide(_log(0.1, seed=0), integrator="euler")
    plan = decided.plan
    assert plan is not None
    _, cost, size = _parameter_cost(plan, decided.model_fit.residual)
    priced = float(cost(jnp.asarray(plan.actions).ravel(), jnp.zeros(size)))
    assert priced == pytest.approx(plan.task_cost, rel=1e-12, abs=0.0)


class _Mirrored(_SetAtStep):
    """A field whose one action is set at the step's start to minus the state."""

    dynamics: LinearDynamics

    def at_step(self, x, u):
        return self.dynamics, -x

    def __call__(self, t, x, u):
        return self.dynamics(t, x, -x)


def test_a_step_steps_an_euler_field_by_its_map_with_the_action_set_at_its_start() -> None:
    from chc.dynamics import _EulerStep

    lin = LinearDynamics(jnp.array([[-0.9]]), jnp.array([[0.5]]))
    x, u = jnp.array([0.7]), jnp.array([1.3])
    np.testing.assert_allclose(
        rk4_step(_EulerStep(lin), 0.0, x, u, 0.5), x + 0.5 * lin(0.0, x, u), rtol=1e-15
    )
    np.testing.assert_allclose(
        rk4_step(_Mirrored(_EulerStep(lin)), 0.0, x, u, 0.5),
        x + 0.5 * lin(0.0, x, -x),
        rtol=1e-15,
    )


def test_the_injection_of_a_field_error_under_the_euler_map_is_dt() -> None:
    """An offset ``g`` in the field moves the Euler step by ``dt g``: the first-order weights are
    the costates' norms times ``dt``, where RK4's polynomial in ``dt J`` reads otherwise."""
    from chc.dynamics import _EulerStep

    lin = LinearDynamics(jnp.array([[-0.9]]), jnp.array([[0.5]]))
    cost = QuadraticCost(Q=jnp.eye(1), R=jnp.eye(1), Qf=jnp.eye(1), x_target=jnp.array([1.0]))
    x0, us, dt = jnp.zeros(1), jnp.full((4, 1), 1.5), 1.0
    euler = _EulerStep(lin)
    weights = perturbation_cost_weights(euler, x0, us, dt, cost, 0.0)
    np.testing.assert_allclose(weights, costate_norms(euler, x0, us, dt, cost) * dt, rtol=1e-12)
    rk4 = perturbation_cost_weights(lin, x0, us, dt, cost, 0.0)
    assert not np.allclose(rk4 / costate_norms(lin, x0, us, dt, cost), dt, rtol=1e-3, atol=0.0)
