"""A prescription fitted as forward Euler's one-step map rolls out by that map.

``prescribe(integrator="euler")`` fits the field so that ``x + dt f`` is the panel's one-step map,
for a panel that is genuinely discrete-time. Up to 0.18.0 the plan then rolled the field out by
RK4 all the same: its path, its tube and its certificate read a map the fit was never made
consistent with.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chc import CausalGraph, Constraint, Lever, Panel, Target, prescribe
from chc.cost import QuadraticCost
from chc.decision import Prescription
from chc.dynamics import HybridDynamics, LinearDynamics, _ForwardEuler
from chc.dynamics_id import Integrator
from chc.integrate import rk4_step, rollout
from chc.plan import _integrator, _quadratic, _Rule, _Ruled, causal_plan
from chc.uncertainty import (
    _linear_tube,
    _noise_covariances,
    _noise_radius,
    _noise_tube,
    _step_parts,
    _tube,
)

_DT = 0.5


def _bilinear_log(periods: int, seed: int) -> Panel:
    """A discrete-time plant of two states, its channel affine in the state: ``x + dt f`` is its
    one-step map, with ``u = 0.5 z - 0.3 x1 + 0.4 e`` confounded by ``z``."""
    rng = np.random.default_rng(seed)
    x = np.zeros(2)
    rows: dict[str, list[float]] = {name: [] for name in ("unit", "time", "x1", "x2", "u", "z")}
    for period in range(periods + 60):
        z = rng.normal()
        u = 0.5 * z - 0.3 * x[0] + 0.4 * rng.normal()
        if period >= 60:
            for name, value in zip(rows, (0, period - 60, x[0], x[1], u, z), strict=True):
                rows[name].append(float(value))
        rate = np.array(
            [
                -0.4 * x[0] + 0.2 * x[1] + (0.8 + 0.3 * x[0]) * u + 0.2 * z,
                -0.3 * x[1] + (0.5 - 0.2 * x[1]) * u + 0.1 * z,
            ]
        )
        x = x + _DT * rate + 0.1 * rng.normal(size=2)
    return Panel.from_frame(
        {name: np.asarray(values) for name, values in rows.items()}, unit="unit", time="time"
    )


def _prescribe(panel: Panel, integrator: Integrator) -> Prescription:
    edges = [("z", "u"), ("z", "x1"), ("z", "x2"), ("u", "x1"), ("u", "x2"), ("x2", "x1")]
    return prescribe(
        panel,
        adjustment=CausalGraph.from_edges(edges),
        levers=[Lever("u", lo=-2.0, hi=2.0)],
        target=Target("x1", value=1.0),
        constraints=[Constraint("x2", hi=10.0)],
        horizon=12,
        dt=_DT,
        tolerance=0.4,
        integrator=integrator,
    )


@pytest.fixture(scope="module")
def euler() -> Prescription:
    return _prescribe(_bilinear_log(400, 3), "euler")


def test_the_plan_rolls_out_by_the_map_the_fit_was_made_consistent_with(
    euler: Prescription,
) -> None:
    """Each step of the plan's path is ``x + dt f(x, u)`` of the fitted field, to rounding, where
    RK4's step of the same field moves the first step by more than a hundredth of the log's noise;
    the plan's task cost is the Euler rollout's."""
    plan, fit = euler.plan, euler.model_fit
    field = HybridDynamics(
        known=LinearDynamics(jnp.zeros((2, 2)), jnp.zeros((2, 1))), residual=fit.residual
    )
    path = np.asarray(plan.trajectory)
    actions = np.asarray(plan.actions)
    steps = np.array(
        [
            np.asarray(state + _DT * field(0.0, jnp.asarray(state), jnp.asarray(action)))
            for state, action in zip(path[:-1], actions, strict=True)
        ]
    )
    np.testing.assert_allclose(path[1:], steps, rtol=1e-12, atol=1e-12)
    rk4 = np.asarray(rollout(field, jnp.asarray(path[0]), jnp.asarray(actions), _DT))
    assert float(np.max(np.abs(rk4[1] - path[1]))) > 1e-3
    problem = plan._problem
    assert problem is not None
    assert _integrator(problem.model) == "euler"


def test_the_tube_is_carried_by_euler_s_step_matrices(monkeypatch: pytest.MonkeyPatch) -> None:
    """The plan's tube is carried by Euler's step, ``1 + dt A_k`` at each planned action with one
    stage, its noise's covariance by ``P_{k+1} = M_k P_k M_k' + M_k C_k + C_k' M_k' + S``,
    ``C_{k+1} = rho (M_k C_k + S)``, and each state's noise radius reads those covariances."""
    import chc.plan

    calls: list[tuple] = []
    carried = chc.plan._error_tube

    def spy(*args: object) -> object:
        calls.append(args)
        return carried(*args)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(chc.plan, "_error_tube", spy)
    result = _prescribe(_bilinear_log(400, 3), "euler")
    assert calls[-1][-1] == "euler"
    np.testing.assert_array_equal(np.asarray(result.plan.uncertainty_tube), carried(*calls[-1]))
    slopes, _, noise, persistence = calls[-1][:4]
    assert slopes.shape == (12, 2, 2)
    eye = np.eye(2)
    covariance, cross = np.zeros((2, 2)), np.zeros((2, 2))
    expected = [covariance]
    for slope in slopes:
        step = eye + _DT * slope
        covariance, cross = (
            step @ covariance @ step.T + step @ cross + cross.T @ step.T + noise,
            persistence * (step @ cross + noise),
        )
        expected.append(covariance)
    covariances = _noise_covariances(slopes, noise, persistence, _DT, 12, "euler")
    np.testing.assert_allclose(np.array(covariances), np.array(expected), rtol=1e-12, atol=0.0)
    state_noise = result.certificate.tube_state_noise
    assert state_noise is not None
    np.testing.assert_allclose(
        np.array(state_noise[3]),
        np.sqrt(3.841458820694124 * np.diag(expected[3])),
        rtol=1e-9,
        atol=0.0,
    )
    for slope in slopes:
        growth, weights = _step_parts(_DT * slope, eye, "euler")
        np.testing.assert_array_equal(growth, eye + _DT * slope)
        assert len(weights) == 1
    # one step's error moves the next state by dt times itself, Euler's one stage
    assert _linear_tube(slopes, [0.1] * 12, _DT, "euler")[1] == pytest.approx(0.1 * _DT, rel=1e-15)
    assert not np.allclose(
        _noise_tube(slopes, noise, persistence, _DT, 12)[2:],
        _noise_tube(slopes, noise, persistence, _DT, 12, "euler")[2:],
        rtol=1e-6,
        atol=0.0,
    )


def test_rk4_step_takes_euler_s_step_for_a_field_fitted_as_that_map() -> None:
    """Marked, a field steps by ``x + dt f``, also inside a rule that sets a lever from the state;
    unmarked, by RK4 as before."""
    field = LinearDynamics(jnp.array([[-0.4, 0.2], [0.1, -0.3]]), jnp.array([[0.8], [0.5]]))
    x, u = jnp.array([0.3, -0.2]), jnp.array([0.7])
    marked = _ForwardEuler(field)
    np.testing.assert_array_equal(
        np.asarray(rk4_step(marked, 0.0, x, u, _DT)), np.asarray(x + _DT * field(0.0, x, u))
    )
    assert not np.allclose(
        np.asarray(rk4_step(field, 0.0, x, u, _DT)), np.asarray(x + _DT * field(0.0, x, u))
    )
    rule = _Rule(
        centre=jnp.zeros(2),
        shift=jnp.zeros(2),
        factor=jnp.ones(2),
        coefficients=jnp.array([[0.2], [0.5], [0.0]]),
        lo=jnp.array([-1.0]),
        hi=jnp.array([1.0]),
        levers=(0,),
        degree=1,
    )
    ruled = _Ruled(marked, rule)
    taken = rule.read(x, u)
    np.testing.assert_array_equal(
        np.asarray(rk4_step(ruled, 0.0, x, u, _DT)), np.asarray(x + _DT * field(0.0, x, taken))
    )
    assert _integrator(ruled) == "euler"
    assert _integrator(_Ruled(field, rule)) == "rk4"
    assert _integrator(field) == "rk4"


def test_an_rk4_prescription_is_unmarked() -> None:
    """The default fit rolls out by RK4, as before: its model carries no mark."""
    result = _prescribe(_bilinear_log(200, 4), "rk4")
    problem = result.plan._problem
    assert problem is not None
    assert _integrator(problem.model) == "rk4"


def test_a_plan_on_a_marked_field_carries_its_norm_bound_tube_by_euler_s_step() -> None:
    """Under a norm bound ``L`` the field's part grows by ``1 + L dt`` a step and the noise's
    largest variance by ``p' = (1 + L dt)^2 p + q``, its radius ``sqrt(chi2_n(0.95) p)`` past the
    first step, which is the noise's own exact radius; the path is Euler's, and a field affine in
    the state and the action is still read as quadratic in the actions."""
    field = LinearDynamics(jnp.array([[-0.4, 0.2], [0.1, -0.3]]), jnp.array([[0.8], [0.5]]))
    marked = _ForwardEuler(field)
    x0 = jnp.array([0.3, -0.2])
    cost = QuadraticCost(
        Q=jnp.eye(2), R=0.1 * jnp.eye(1), Qf=jnp.eye(2), x_target=jnp.array([1.0, 0.0])
    )
    noise = np.array([[0.02, 0.005], [0.005, 0.01]])
    plan = causal_plan(
        marked, x0, cost, _DT, 6, -1.0, 1.0, lipschitz=0.7, model_error=0.05, noise=noise
    )
    actions = np.asarray(plan.actions)
    path = [np.asarray(x0)]
    for action in actions:
        state = jnp.asarray(path[-1])
        path.append(np.asarray(state + _DT * field(0.0, state, jnp.asarray(action))))
    np.testing.assert_allclose(np.asarray(plan.trajectory), np.array(path), rtol=0.0, atol=1e-12)
    growth = 1.0 + 0.7 * _DT
    largest = float(np.linalg.eigvalsh(noise)[-1])
    variance, expected_noise = 0.0, [0.0]
    for _ in range(6):
        variance = growth * growth * variance + largest
        expected_noise.append(float(np.sqrt(5.991464547107979 * variance)))
    expected_noise[1] = _noise_radius(noise)
    np.testing.assert_allclose(
        _noise_tube(np.array(0.7), noise, 0.0, _DT, 6, "euler"), expected_noise, rtol=1e-12
    )
    expected = _tube([0.7] * 6, [0.05] * 6, _DT, "euler") + np.array(expected_noise)
    assert plan.uncertainty_tube is not None
    np.testing.assert_allclose(np.asarray(plan.uncertainty_tube), expected, rtol=1e-12, atol=0.0)
    assert _quadratic(marked)
