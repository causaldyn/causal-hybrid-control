"""The tube's field part read at the plan's own actions, and the per-log spread beside the tube.

The box's form reads each step's field error alone, at the worst vertex of the levers' box, and
adds the gains' norms: where the plan's actions change sign the steps' errors cancel, which it
cannot see. ``tube_actions="plan"`` carries the fit's error through the plan's own rollout as one
vector, ``S_k Sigma S_k'`` (ADR 0138). The certificate reports, beside the tube, the spread over
logs of the share of paths a tube calibrated on average holds.
"""

from __future__ import annotations

import logging
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

from chc import CausalGraph, Lever, Panel, Target, prescribe
from chc.decision import (
    DecisionError,
    _field_errors,
    _per_log_spread,
    _plan_field,
    _slopes,
)
from chc.dynamics import HybridDynamics, LinearDynamics, _EulerStep
from chc.dynamics_id import fit_causal_residual
from chc.integrate import rollout
from chc.uncertainty import _linear_tube

jax.config.update("jax_enable_x64", True)

_DT, _RATE, _GAIN, _SIGMA, _ROWS, _HORIZON = 0.2, -0.5, 1.0, 0.05, 200, 15
_KNOWN = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))


def _closed_loop(seed: int) -> dict[str, jnp.ndarray]:
    """One log of the Euler map ``x+ = x + dt (a x + b u) + e``, ``u = -x/2 + n/2``."""
    rng = np.random.default_rng(seed)
    x, rows = 0.0, []
    for _ in range(_ROWS):
        u = -0.5 * x + 0.5 * rng.normal()
        following = x + _DT * (_RATE * x + _GAIN * u) + _SIGMA * rng.normal()
        rows.append((x, u, following))
        x = following
    xs, us, nexts = (jnp.asarray([row[j] for row in rows])[:, None] for j in range(3))
    return {"x": xs, "u": us, "x_next": nexts}


def _truth(actions: np.ndarray) -> np.ndarray:
    path = [0.0]
    for u in actions:
        path.append(path[-1] + _DT * (_RATE * path[-1] + _GAIN * u))
    return np.asarray(path)


def _forms(seed: int, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The fitted rollout's miss from the truth, the box's form at the plan's own actions and the
    plan's form, steps 1 to the horizon."""
    fit = fit_causal_residual(_KNOWN, _closed_loop(seed), _DT, unconfounded=True, folds=1)
    model = _EulerStep(HybridDynamics(known=_KNOWN, residual=fit.residual))
    start, moves = jnp.zeros(1), jnp.asarray(actions)[:, None]
    fitted = np.asarray(rollout(model, start, moves, _DT))[:, 0]
    errors = np.array(
        [
            _field_errors(
                fit,
                np.zeros(1),
                fitted[step : step + 1, None],
                np.zeros((1, 0)),
                (np.array([actions[step]]), np.array([actions[step]])),
            )[0]
            for step in range(len(actions))
        ]
    )
    box = _linear_tube(_slopes(model, start)(np.asarray(moves)), errors, _DT, integrator="euler")
    plan = _plan_field(fit, model, start, moves, _DT)
    return fitted[1:] - _truth(actions)[1:], box[1:], plan[1:]


def test_the_first_step_is_the_box_s_form_at_the_plan_s_action() -> None:
    """One step reads one error, the field's at the plan's first action: the two forms agree."""
    actions = np.array([(-1.0) ** step for step in range(_HORIZON)])
    for seed in range(3):
        _, box, plan = _forms(seed, actions)
        assert plan[0] == pytest.approx(box[0], rel=1e-12, abs=0.0)
        assert np.all(plan <= box * (1.0 + 1e-12))


def test_a_held_plan_from_rest_reads_close_to_the_box_s_form() -> None:
    """A held action's errors share their sign, so the forms differ only as the state moves the
    error's direction: within 5 % at every step (``validation/plan_action_tube.mac``, (5): 1.039,
    1.025, 1.014 at the population covariance)."""
    actions = np.full(_HORIZON, 0.6)
    for seed in range(3):
        _, box, plan = _forms(seed, actions)
        assert np.all(box / plan >= 1.0 - 1e-12)
        assert np.all(box / plan <= 1.05)


@pytest.mark.parametrize("seed", [0, 1])
def test_an_alternating_plan_is_covered_by_its_own_form_where_the_box_s_is_too_wide(
    seed: int,
) -> None:
    """60 logs a seed, actions ``+-1`` in turn: the box's form is about 1.75 times the plan's at
    the plan's actions (``validation/plan_action_tube.mac``: 1.95 to 2.15 at the population
    covariance), and the misses over the plan's form have a root mean square of about one, where
    over the box's they are about 0.6. A root mean square of 60 standard normals is within 0.25 of
    one at about four of its standard errors, ``1 / sqrt(120)``; 0.6 lies 4.4 standard errors
    below."""
    logs = 60
    actions = np.array([(-1.0) ** step for step in range(_HORIZON)])
    forms = [_forms(1000 * seed + log, actions) for log in range(logs)]
    miss, box, plan = (np.array([form[j] for form in forms]) for j in range(3))
    for step in (4, 9, 14):
        assert 0.75 <= math.sqrt(float(np.mean((miss[:, step] / plan[:, step]) ** 2))) <= 1.25
        assert float(np.median(box[:, step] / plan[:, step])) >= 1.5


def _bilinear_log(seed: int, periods: int = 480) -> dict[str, np.ndarray]:
    """One unit of ``x' = -0.4 x + (0.8 + 0.3 x) u + 0.2 z`` at half-hour steps, ``z`` a logged
    confounder and ``u = 0.5 z`` plus a draw of its own (``tests/test_exact_tube.py``'s log)."""
    rng = np.random.default_rng(seed)
    rows: dict[str, list[float]] = {name: [] for name in ("unit", "time", "x", "u", "z")}
    x = 0.0
    for period in range(periods):
        z = rng.normal()
        u = 0.5 * z + rng.normal(0.0, 0.4)
        for name, value in (("unit", 0), ("time", period), ("x", x), ("u", u), ("z", z)):
            rows[name].append(value)

        def field(state: float, u: float = u, z: float = z) -> float:
            return -0.4 * state + (0.8 + 0.3 * state) * u + 0.2 * z

        k1 = field(x)
        k2 = field(x + 0.25 * k1)
        k3 = field(x + 0.25 * k2)
        k4 = field(x + 0.5 * k3)
        x = x + 0.5 / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4) + rng.normal(0.0, 0.1)
    return {name: np.asarray(values) for name, values in rows.items()}


def _decide(integrator: str, **options: object):
    log = _bilinear_log(3)
    return prescribe(
        Panel.from_frame(log, unit="unit", time="time", seed=0),
        levers=[Lever("u", lo=-2.0, hi=2.0)],
        target=Target("x", value=1.0),
        adjustment=CausalGraph.from_edges([("z", "u"), ("z", "x"), ("u", "x")]),
        dt=0.5,
        x0=[float(log["x"][-1])],
        horizon=12,
        tolerance=10.0,
        integrator=integrator,  # type: ignore[arg-type]
        **options,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("integrator", ["euler", "rk4"])
def test_prescribe_reads_the_plan_s_form_where_asked_and_names_it(integrator: str) -> None:
    """The same plan, its tube's field part at the plan's actions: on this log never wider than
    the box's, which reads the box's worst vertex, and narrower past the first steps; the noise's
    part the same; the certificate and its JSON name the form. The default stays the box's."""
    box = _decide(integrator)
    plan = _decide(integrator, tube_actions="plan")
    assert box.plan is not None
    assert plan.plan is not None
    assert box.certificate.tube_actions == "box"
    assert plan.certificate.tube_actions == "plan"
    assert plan.to_json()["certificate"]["tube_actions"] == "plan"
    np.testing.assert_array_equal(np.asarray(plan.plan.actions), np.asarray(box.plan.actions))
    wide, narrow = np.asarray(box.plan.uncertainty_tube), np.asarray(plan.plan.uncertainty_tube)
    assert np.all(narrow <= wide * (1.0 + 1e-12))
    assert np.any(narrow[2:] < 0.99 * wide[2:])
    assert plan.certificate.tube_state_noise == box.certificate.tube_state_noise


def test_prescribe_refuses_a_form_it_does_not_know() -> None:
    with pytest.raises(DecisionError, match="tube_actions"):
        _decide("rk4", tube_actions="vertex")


class _Bent(eqx.Module):
    """Known physics whose slope in the state changes from one state to the next."""

    def __call__(self, t: float | jax.Array, x: jax.Array, u: jax.Array) -> jax.Array:
        return -0.3 * jnp.tanh(x)


def test_the_plan_s_form_falls_back_to_the_box_s_where_the_tube_is_a_bound(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Known physics bent in the state has no exact tube: asked for the plan's form, the decision
    warns and reads the box's, and says so."""

    log = _bilinear_log(5, periods=240)
    with caplog.at_level(logging.WARNING, logger="chc"):
        decided = prescribe(
            Panel.from_frame(log, unit="unit", time="time", seed=0),
            known=_Bent(),
            levers=[Lever("u", lo=-2.0, hi=2.0)],
            target=Target("x", value=1.0),
            adjustment=CausalGraph.from_edges([("z", "u"), ("z", "x"), ("u", "x")]),
            dt=0.5,
            x0=[float(log["x"][-1])],
            horizon=6,
            tolerance=10.0,
            integrator="rk4",
            tube_actions="plan",
        )
    assert decided.certificate.tube_mode == "bound"
    assert decided.certificate.tube_actions == "box"
    assert decided.certificate.tube_per_log_spread is None
    assert any("not at the plan's" in record.getMessage() for record in caplog.records)


# the spreads at kappa 1/2, 1 and 2 to 19 digits (``validation/tube_per_log_content.gp``)
_SPREADS = {0.5: 0.0324434054809161837, 1.0: 0.0821814200157158550, 2.0: 0.1394345691998676063}


def test_the_per_log_spread_is_its_closed_form() -> None:
    for kappa, spread in _SPREADS.items():
        assert _per_log_spread(kappa) == pytest.approx(spread, rel=1e-12, abs=0.0)
    assert _per_log_spread(0.0) == 0.0
    assert _per_log_spread(math.inf) == pytest.approx(math.sqrt(0.95 * 0.05), rel=1e-15, abs=0.0)
    assert _per_log_spread(1e4) == pytest.approx(math.sqrt(0.95 * 0.05), rel=1e-3, abs=0.0)


def test_the_per_log_spread_by_simulation() -> None:
    """At ``kappa = 1``, 4000 logs of a log's share of its paths: the spread is within four of its
    standard errors of the closed form, the standard error from the share's fourth moment."""
    rng = np.random.default_rng(11)
    kappa, z = 1.0, 1.959963984540054
    shift = kappa * rng.normal(size=4000)
    radius = z * math.sqrt(1.0 + kappa**2)
    share = norm.cdf(radius - shift) - norm.cdf(-radius - shift)
    centred = share - np.mean(share)
    variance = float(np.mean(centred**2))
    error = math.sqrt((float(np.mean(centred**4)) - variance**2) / share.size) / (
        2.0 * math.sqrt(variance)
    )
    assert abs(math.sqrt(variance) - _SPREADS[1.0]) <= 4.0 * error


@pytest.mark.parametrize("tube_actions", ["box", "plan"])
def test_the_certificate_s_spread_reads_the_tube_s_own_parts(tube_actions: str) -> None:
    """On one state the tube is ``z sqrt(sigma_k^2 + s_k^2)``, ``z sigma_k`` the state's own noise
    radius: the spread reported at each step is the closed form at ``kappa = s_k / sigma_k``."""
    decided = _decide("rk4", tube_actions=tube_actions)
    certificate = decided.certificate
    assert decided.plan is not None
    assert certificate.tube_per_log_spread is not None
    assert certificate.tube_state_noise is not None
    tube = np.asarray(decided.plan.uncertainty_tube)
    noise = np.array([radii[0] for radii in certificate.tube_state_noise])
    assert certificate.tube_per_log_spread[0] == 0.0
    for step in range(1, tube.size):
        kappa = math.sqrt((tube[step] / noise[step]) ** 2 - 1.0)
        assert certificate.tube_per_log_spread[step] == pytest.approx(
            _per_log_spread(kappa), rel=1e-9, abs=0.0
        )
    assert max(certificate.tube_per_log_spread) > 0.0
    assert decided.to_json()["certificate"]["tube_per_log_spread"] == list(
        certificate.tube_per_log_spread
    )
    (line,) = [line for line in decided.report().splitlines() if "spreads about 0.95" in line]
    assert ("plan's own actions" in line) == (tube_actions == "plan")
