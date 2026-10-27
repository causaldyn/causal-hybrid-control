"""Under RK4 the tube reads the field's error where each of RK4's four stages reads the field.

One RK4 step of a field affine in the state moves with the field's error ``H(x, u) theta`` by
``dt sum_i c_i(z) H(x_i, u) theta``, ``x_i`` the step's stage states (``validation/
rk4_stage_term.mac``). Read at the step's start alone, the box's form drops a term of the order of
``dt^2``; read at each stage, each carried by its own weight, it bounds the step's error by
Minkowski's inequality, as it bounds the steps' sum (ADR 0141). The Euler map has one stage, the
step's start, and reads as before.
"""

from __future__ import annotations

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chc import CausalGraph, Lever, Panel, Target, decision, prescribe
from chc.decision import _field_errors, _plan_field, _slopes, _stage_states, _Stages
from chc.dynamics import HybridDynamics, LinearDynamics
from chc.dynamics_id import fit_causal_residual
from chc.integrate import rk4_step, rollout
from chc.plan import _error_tube
from chc.uncertainty import _linear_tube

jax.config.update("jax_enable_x64", True)

_DT, _RATE, _GAIN, _SIGMA, _ROWS, _HORIZON = 0.2, -0.5, 1.0, 0.05, 200, 15
_KNOWN = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
_PLANS = {
    "held": np.full(_HORIZON, 0.6),
    "crossed": np.r_[np.ones(5), -np.ones(10)],
    "alternating": np.array([(-1.0) ** step for step in range(_HORIZON)]),
}


def _flow_log(seed: int) -> dict[str, jnp.ndarray]:
    """One log of the exact flow of ``x' = a x + b u``, ``u = -x/2 + n/2`` held over each step:
    an RK4 fit's field is the plant's to the order of ``dt^4``, so its miss is its error."""
    rng = np.random.default_rng(seed)
    growth = math.exp(_RATE * _DT)
    x, rows = 0.0, []
    for _ in range(_ROWS):
        u = -0.5 * x + 0.5 * rng.normal()
        following = growth * x + (growth - 1.0) / _RATE * _GAIN * u + _SIGMA * rng.normal()
        rows.append((x, u, following))
        x = following
    xs, us, nexts = (jnp.asarray([row[j] for row in rows])[:, None] for j in range(3))
    return {"x": xs, "u": us, "x_next": nexts}


def _forms(
    seed: int, actions: np.ndarray, **options: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The box's form at the plan's own actions read at each step's start, the same read at
    RK4's stages, and the plan's form, steps 1 to the horizon."""
    fit = fit_causal_residual(
        _KNOWN, _flow_log(seed), _DT, unconfounded=True, folds=1, integrator="rk4", **options
    )
    model = HybridDynamics(known=_KNOWN, residual=fit.residual)
    start, moves = jnp.zeros(1), jnp.asarray(actions)[:, None]
    fitted = np.asarray(rollout(model, start, moves, _DT))
    stages = _Stages(model, _DT, np.zeros((1, 4, 0)))
    at_start, at_stages = (
        np.array(
            [
                _field_errors(
                    fit,
                    np.zeros(1),
                    fitted[step : step + 1],
                    np.zeros((1, 0)),
                    (np.array([actions[step]]), np.array([actions[step]])),
                    read,
                )[0]
                for step in range(len(actions))
            ]
        )
        for read in (None, stages)
    )
    matrices = _slopes(model, start)(np.asarray(moves))
    return (
        _linear_tube(matrices, at_start, _DT, integrator="rk4")[1:],
        _linear_tube(matrices, at_stages, _DT, integrator="rk4")[1:],
        _plan_field(fit, model, start, moves, _DT)[1:],
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_stages_read_is_never_narrower_than_the_plan_s_form_under_rk4(seed: int) -> None:
    """At the plan's own actions the box's form read at RK4's stages is never narrower than the
    plan's form, at any step of a held, a crossed or an alternating plan: the gains' norms and the
    stages' bound the steps' sum and each step's stages by Minkowski's inequality. Read at each
    step's start, it fell short of the plan's form at the first step of every plan here, the
    stages' term pushing the field's error outward: on 400 logs its misses had a root mean square
    of 1.19 of it there."""
    for actions in _PLANS.values():
        at_start, at_stages, plan = _forms(seed, actions)
        assert np.all(at_stages >= plan * (1.0 - 1e-12)), np.round(at_stages / plan, 4)
        assert at_start[0] < plan[0], (at_start[0], plan[0])


def test_the_stage_states_are_the_states_rk4_reads() -> None:
    """The four states the tube reads the field's error at are those RK4 reads the field at: the
    step from them is :func:`chc.integrate.rk4_step`'s, to rounding."""
    fit = fit_causal_residual(
        _KNOWN, _flow_log(0), _DT, unconfounded=True, folds=1, integrator="rk4"
    )
    model = HybridDynamics(known=_KNOWN, residual=fit.residual)
    states = np.array([[0.0], [0.4], [-1.3]])
    corners = np.array([[-2.0], [2.0]])
    moved = _stage_states(model, states, corners, _DT)
    assert moved.shape == (3, 2, 4, 1)
    weights = np.array([1.0, 2.0, 2.0, 1.0]) / 6.0
    offsets = np.array([0.0, 0.5, 0.5, 1.0]) * _DT
    for step, vertex in itertools.product(range(3), range(2)):
        time = step * _DT
        action = jnp.asarray(corners[vertex])
        slopes = [
            float(model(time + offset, jnp.asarray(moved[step, vertex, stage]), action)[0])
            for stage, offset in enumerate(offsets)
        ]
        stepped = float(rk4_step(model, time, jnp.asarray(states[step]), action, _DT)[0])
        assert states[step, 0] + _DT * float(weights @ slopes) == pytest.approx(
            stepped, rel=1e-13, abs=1e-15
        )
        assert moved[step, vertex, 0] == pytest.approx(states[step], rel=0.0, abs=0.0)


def test_a_stage_s_error_is_largest_at_a_vertex_where_the_channel_does_not_read_the_state() -> None:
    """Where the channel does not read the state, a stage's state moves affinely with the action
    and its error's variance is a convex quadratic in it (``rk4_stage_term.mac`` (6)): over the box
    ``[-2, 2]`` each stage's read at the vertices is the largest over 401 actions inside it."""
    fit = fit_causal_residual(
        _KNOWN,
        _flow_log(1),
        _DT,
        unconfounded=True,
        folds=1,
        integrator="rk4",
        channel_degree=0,
    )
    model = HybridDynamics(known=_KNOWN, residual=fit.residual)
    stages = _Stages(model, _DT, np.zeros((1, 4, 0)))
    for state in (-1.0, 0.0, 1.5):
        rows = np.array([[state]])
        box = _field_errors(
            fit, np.zeros(1), rows, np.zeros((1, 0)), (np.array([-2.0]), np.array([2.0])), stages
        )[0]
        inside = np.array(
            [
                _field_errors(
                    fit, np.zeros(1), rows, np.zeros((1, 0)), (np.array([u]), np.array([u])), stages
                )[0]
                for u in np.linspace(-2.0, 2.0, 401)
            ]
        )
        assert np.all(box >= inside.max(axis=0) * (1.0 - 1e-12))
        assert np.any(box > inside[200] * 1.01)


@pytest.mark.parametrize("matrices", [np.array([[-0.4]]), np.array([[[-0.4]], [[0.1]], [[-1.0]]])])
@pytest.mark.parametrize("integrator", ["rk4", "euler"])
def test_equal_stage_errors_carry_the_one_error_s_tube(
    matrices: np.ndarray, integrator: str
) -> None:
    """Every stage's error equal to the step's reads the one error's tube, to rounding: the
    stages' weights are what the one error is carried by."""
    errors = np.array([0.3, 0.1, 0.7])
    stages = 4 if integrator == "rk4" else 1
    one = _linear_tube(matrices, errors, 0.5, integrator=integrator)
    staged = _linear_tube(
        matrices, np.repeat(errors[:, None], stages, axis=1), 0.5, integrator=integrator
    )
    np.testing.assert_allclose(staged, one, rtol=1e-14, atol=0.0)
    with pytest.raises(ValueError, match="one a stage"):
        _linear_tube(matrices, np.ones((3, 5 - stages)), 0.5, integrator=integrator)
    with pytest.raises(ValueError, match="per-step error budget"):
        _linear_tube(matrices, np.full((3, stages), -1.0), 0.5, integrator=integrator)


def test_a_norm_bound_reads_the_stages_largest() -> None:
    """A norm bound holds where the field is off by at most its budget wherever the step reads it,
    so given each stage's error it reads their largest."""
    errors = np.array([[0.1, 0.4, 0.2, 0.3], [0.5, 0.1, 0.1, 0.1]])
    covariance = np.array([[0.01]])
    staged = _error_tube(np.float64(0.8), errors, covariance, 0.0, 0.1, 2, 1.1, None)
    largest = _error_tube(np.float64(0.8), errors.max(axis=1), covariance, 0.0, 0.1, 2, 1.1, None)
    assert staged is not None
    assert largest is not None
    np.testing.assert_array_equal(staged, largest)


def _bilinear_log(seed: int) -> dict[str, np.ndarray]:
    """One unit of the Euler map of ``x' = -0.4 x + (0.8 + 0.3 x) u + 0.2 z``, ``dt = 0.5``."""
    rng = np.random.default_rng(seed)
    rows: dict[str, list[float]] = {name: [] for name in ("unit", "time", "x", "u", "z")}
    x = 0.0
    for period in range(240):
        z = rng.normal()
        u = 0.5 * z + rng.normal(0.0, 0.4)
        for name, value in (("unit", 0), ("time", period), ("x", x), ("u", u), ("z", z)):
            rows[name].append(value)
        x = x + 0.5 * (-0.4 * x + (0.8 + 0.3 * x) * u + 0.2 * z) + rng.normal(0.0, 0.1)
    return {name: np.asarray(values) for name, values in rows.items()}


def _decide(integrator: str):
    log = _bilinear_log(2)
    return prescribe(
        Panel.from_frame(log, unit="unit", time="time", seed=0),
        levers=[Lever("u", lo=-2.0, hi=2.0)],
        target=Target("x", value=1.0),
        adjustment=CausalGraph.from_edges([("z", "u"), ("z", "x"), ("u", "x")]),
        dt=0.5,
        x0=[float(log["x"][-1])],
        horizon=8,
        tolerance=10.0,
        integrator=integrator,  # type: ignore[arg-type]
    )


def test_the_euler_map_reads_no_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Euler map has one stage, the step's start: its tube never reads the stages, and stays
    0.22.0's bit for bit. Under ``rk4`` the same decision reads them."""

    def refused(*args: object, **kwargs: object) -> None:
        raise AssertionError("the stages were read")

    euler = _decide("euler")
    assert euler.plan is not None
    assert euler.certificate.tube_mode == "exact"
    monkeypatch.setattr(decision, "_stage_variances", refused)
    again = _decide("euler")
    assert again.plan is not None
    np.testing.assert_array_equal(
        np.asarray(again.plan.uncertainty_tube), np.asarray(euler.plan.uncertainty_tube)
    )
    with pytest.raises(AssertionError, match="the stages were read"):
        _decide("rk4")
