"""The facade runs the whole chain, and the chain's failures survive being wrapped in it.

The load-bearing test is the three-arm comparison: the same logs, read three ways. With the
confounder adjusted for the channel is recovered; with an empty adjustment asserted it is nearly
three times too large; with the confounder declared latent there is no schedule at all. If wrapping
the layers in one call ever loses that separation, this file fails.
"""

from __future__ import annotations

import dataclasses
import datetime
import functools
import importlib
import inspect
import json
import logging
import math
import types
from decimal import Decimal
from importlib import metadata
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chc.cost import QuadraticCost
from chc.decision import (
    Constraint,
    DecisionCertificate,
    DecisionError,
    Driver,
    Lever,
    NotIdentifiedError,
    Prescription,
    RunProvenance,
    StartState,
    Target,
    _barrier,
    _certify,
    _episodes,
    _json_label,
    _linearised,
    _margins,
    _model_error,
    _panel_start,
    _rate,
    _show_start,
    _transitions,
    prescribe,
)
from chc.dynamics import (
    DampedOscillator,
    DrivenDynamics,
    Dynamics,
    HybridDynamics,
    LinearDynamics,
)
from chc.dynamics_id import fit_causal_residual
from chc.graph import AdjustmentSet, CausalGraph
from chc.integrate import rollout
from chc.mpc import PeriodBudget
from chc.panel import Panel, PanelError
from chc.plan import CausalPlan, causal_plan, certify_safety
from chc.residual import ControlAffineResidual

# Public as jax.enable_x64 from jax 0.8.0; the floor, 0.4.30, has only jax.experimental.enable_x64,
# which jax 0.11 no longer has.
if hasattr(jax, "enable_x64"):
    enable_x64 = jax.enable_x64
else:
    enable_x64 = importlib.import_module("jax.experimental").enable_x64

DT = 0.1
B_TRUE = 0.8  # the incentive's true effect on supply, the number every arm is judged against
WAIT_CHANNEL = -0.4

EDGES = [
    ("demand", "incentive"),  # the policy chases the demand shock: this is the confounding
    ("demand", "supply"),
    ("incentive", "supply"),
    ("incentive", "wait"),
    ("supply", "wait"),
]


def _logs(
    n_units: int = 200, n_periods: int = 12, seed: int = 0, sticky: float = 0.0
) -> dict[str, np.ndarray]:
    """Two zones of a driver pool under one incentive, logged by a policy that chases demand, and
    that keeps ``sticky`` of its last incentive."""
    rng = np.random.default_rng(seed)
    rows: dict[str, list[float]] = {
        name: [] for name in ("unit", "time", "supply", "wait", "incentive", "demand")
    }
    for unit in range(n_units):
        supply, wait = rng.normal(0.0, 0.2), rng.normal(0.0, 0.2)
        incentive = 0.0
        for period in range(n_periods):
            demand = rng.normal(0.0, 1.0)
            incentive = 0.9 * demand + sticky * incentive + rng.normal(0.0, 0.5)
            rows["unit"].append(unit)
            rows["time"].append(period)
            rows["supply"].append(supply)
            rows["wait"].append(wait)
            rows["incentive"].append(incentive)
            rows["demand"].append(demand)
            supply_next = supply + DT * (
                -0.6 * supply + 0.3 * wait + B_TRUE * incentive + 1.5 * demand
            )
            wait_next = wait + DT * (0.25 * wait + WAIT_CHANNEL * incentive)
            supply = supply_next + rng.normal(0.0, 0.01)
            wait = wait_next + rng.normal(0.0, 0.01)
    return {name: np.asarray(values) for name, values in rows.items()}


def _panel(**kwargs: int) -> Panel:
    return Panel.from_frame(_logs(**kwargs), unit="unit", time="time", seed=0)


def _prescribe(panel: Panel, adjustment: object, **kwargs: object):
    return prescribe(
        panel,
        levers=[Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05)],
        target=Target("supply", value=1.0),
        constraints=[Constraint("wait", hi=0.5)],
        adjustment=adjustment,  # type: ignore[arg-type]
        horizon=15,
        dt=DT,
        tolerance=0.5,
        **kwargs,  # type: ignore[arg-type]
    )


def _channel(result, state: int = 0) -> float:
    """The constant term of the fitted control channel on one state's row."""
    return float(np.asarray(result.model_fit.residual.channel)[state, 0, 0])


def test_the_graph_derived_arm_recovers_the_channel_the_confounded_one_gets_wrong() -> None:
    panel = _panel()
    graph = CausalGraph.from_edges(EDGES)

    adjusted = _prescribe(panel, graph)
    assert adjusted.certificate.adjustment.covariates == ("demand",)
    assert abs(_channel(adjusted) - B_TRUE) < 0.05
    assert abs(_channel(adjusted, state=1) - WAIT_CHANNEL) < 0.05

    asserted_empty = _prescribe(panel, ())
    assert asserted_empty.certificate.identification == "asserted"
    assert _channel(asserted_empty) > 2.0  # 2.6x the truth: the whole reason the graph is required
    assert "observational fit" in asserted_empty.certificate.adjustment.reason


def test_a_latent_confounder_produces_no_schedule_at_all() -> None:
    graph = CausalGraph.from_edges(EDGES, latent=("demand",))
    panel = _panel()
    result = _prescribe(panel, graph)
    assert result.certificate.identification == "not_identified"
    # where the plan would have started
    assert result.start == _panel_start(panel, ("supply", "wait"))[1]
    assert result.plan is None
    assert result.certificate.trustworthy_steps == 0
    assert result.certificate.solver_status is None
    with pytest.raises(NotIdentifiedError, match="not identified"):
        _ = result.schedule
    report = result.report()
    assert "no schedule" in report.lower()
    assert "\n\nStart: the mean (standard deviation) over 200 units of " in report
    assert "- solver: not run" in report
    assert "None" not in report


def _policy_logs(
    offset: float, random_lever: float, reads: tuple[float, float] = (0.9, -0.3)
) -> dict[str, np.ndarray]:
    """Eighty units of twelve periods whose lever ``u`` the policy sets from the confounder and the
    state alone, ``offset + reads[0] z + reads[1] y``, beside a lever ``v`` drawn at random that
    pushes ``y`` by ``random_lever``."""
    rng = np.random.default_rng(0)
    rows: dict[str, list[float]] = {name: [] for name in ("unit", "time", "y", "u", "v", "z")}
    for unit in range(80):
        y = rng.normal()
        for period in range(12):
            z = rng.normal()
            u, v = offset + reads[0] * z + reads[1] * y, rng.normal()
            for name, value in zip(rows, (unit, period, y, u, v, z), strict=True):
                rows[name].append(value)
            rate = -0.5 * y + 0.8 * u + random_lever * v + 1.5 * z
            y = y + DT * rate + rng.normal(0.0, 0.01)
    return {name: np.asarray(values) for name, values in rows.items()}


def _prescribe_policy(logs: dict[str, np.ndarray], levers: list[Lever], **kwargs: object):
    edges = [("z", "u"), ("z", "y"), *((lever.name, "y") for lever in levers)]
    return prescribe(
        Panel.from_frame(logs, unit="unit", time="time", seed=0),
        levers=levers,
        target=Target("y", value=1.0),
        adjustment=CausalGraph.from_edges(edges),
        horizon=3,
        dt=DT,
        tolerance=0.5,
        **kwargs,  # type: ignore[arg-type]
    )


def test_a_lever_the_log_never_moved_gives_no_schedule(caplog) -> None:
    """A review's case: the policy sets the lever from the confounder and the state, so nothing of
    it is left once they are adjusted for, and the moment has no data on its channel. Up to 0.13 the
    ridge read it as near zero where the truth is 0.8, and the plan on that channel was certified
    over every step; now there is none."""
    with caplog.at_level(logging.WARNING, logger="chc.decision"):
        result = _prescribe_policy(
            _policy_logs(0.0, 0.0), [Lever("u", lo=-2.0, hi=2.0, unit_cost=0.05)]
        )
    certificate = result.certificate
    assert (certificate.identification, certificate.unmoved_levers) == ("not_identified", ("u",))
    assert certificate.adjustment.status == "not_identified"
    assert result.plan is None
    assert (certificate.trustworthy_steps, certificate.solver_status) == (0, None)
    with pytest.raises(NotIdentifiedError, match=r"the log never moves \['u'\]"):
        _ = result.schedule
    assert result.to_json()["certificate"]["unmoved_levers"] == ["u"]
    assert "**No schedule.**" in result.report()
    (abort,) = [r for r in caplog.records if getattr(r, "chc_event", None) == "abort"]
    assert abort.getMessage() == "no schedule: the log never moves a lever"


@pytest.mark.parametrize(
    ("max_levers", "most"),
    [(None, 2.0), (1, 2.0), (2, 2.0), (None, 0.25)],
    ids=["every lever", "one lever kept", "two levers kept", "a box below the logged level"],
)
def test_a_lever_the_log_never_moved_is_held_at_its_logged_level(caplog, max_levers, most) -> None:
    """Beside a lever drawn at random, the one the policy kept at one level is held there, or at its
    box's end where the level lies past it, at every step; greedy selection never offers it. The
    certificate, the JSON, the report and a warning name it, and the random lever is planned. Held
    at its box's end the plan leaves what the log did from the first step, and says so."""
    logs = _policy_logs(0.5, 0.6, reads=(0.0, 0.0))
    levers = [Lever("u", lo=-2.0, hi=most, unit_cost=0.05), Lever("v", lo=-2.0, hi=2.0)]
    with caplog.at_level(logging.INFO, logger="chc.decision"):
        result = _prescribe_policy(logs, levers, max_levers=max_levers)
    certificate = result.certificate
    assert (certificate.identification, certificate.unmoved_levers) == ("identified", ("u",))
    assert result.plan is not None
    actions = np.asarray(result.plan.actions)
    mean = float(logs["u"].reshape(80, 12)[:, :-1].mean())  # each unit's last row starts nothing
    assert mean > 0.25
    logged = min(mean, most)
    np.testing.assert_allclose(actions[:, 0], logged, rtol=1e-12, atol=0.0)
    assert np.max(np.abs(actions[:, 1])) > 0.1
    if max_levers is not None:
        assert result.selection is not None
        assert [step.lever for step in result.selection.steps] == ["v"]
        offered = [r.candidates for r in caplog.records if r.getMessage() == "lever selected"]
        assert [sorted(candidates) for candidates in offered] == [["v"]]
    assert result.to_json()["certificate"]["unmoved_levers"] == ["u"]
    assert "- never moved by the log, so held at their logged level: `u`" in result.report()
    (held,) = [r for r in caplog.records if getattr(r, "chc_event", None) == "unmoved"]
    assert held.levers == ["u"]
    assert held.levels == pytest.approx([logged], rel=1e-12, abs=0.0)
    inside = most >= mean
    assert certificate.estimability == ("held_to_log" if inside else "not_estimable")
    assert certificate.first_loaded_step == (None if inside else 0)
    assert (certificate.trustworthy_steps > 0) == inside


def test_a_lever_set_from_a_column_outside_the_state_gives_no_schedule(caplog) -> None:
    """The policy sets ``u`` from the confounder as well as the state: no level is that rule, and
    the plan cannot read the confounder, so there is no schedule, and the reason names it. The
    mean hold, the answer up to 0.13, kept the confounder's push only where it is independent of
    the state, and moved the state's part to a level the log never ran."""
    with caplog.at_level(logging.WARNING, logger="chc.decision"):
        result = _prescribe_policy(
            _policy_logs(0.5, 0.6),
            [Lever("u", lo=-2.0, hi=2.0, unit_cost=0.05), Lever("v", lo=-2.0, hi=2.0)],
        )
    certificate = result.certificate
    assert result.plan is None
    assert (certificate.identification, certificate.estimability) == (
        "not_identified",
        "not_estimable",
    )
    assert certificate.unmoved_levers == ("u",)
    assert "the log set `u` from `z`, outside the plan's state" in certificate.adjustment.reason
    with pytest.raises(NotIdentifiedError, match="no plan keeps to what the log did"):
        _ = result.schedule
    (abort,) = [r for r in caplog.records if getattr(r, "chc_event", None) == "abort"]
    assert abort.getMessage() == "no schedule: no plan keeps to what the log did"


def test_a_latent_parent_the_panel_does_not_hold_leaves_the_logger_unchecked() -> None:
    """The check conditions on the levers' parents, so it cannot run without one; the rest of the
    unidentified path goes on as before, whether or not the panel holds a column by that name."""
    graph = CausalGraph.from_edges(EDGES, latent=("demand",))
    logs = _logs()
    logs.pop("demand")
    for panel in (_panel(), Panel.from_frame(logs, unit="unit", time="time", seed=0)):
        result = _prescribe(panel, graph)
        assert result.certificate.identification == "not_identified"
        assert result.logger_check is None
        assert "- logger check: not run, the levers' parents ['demand'] are not logged" in (
            result.report()
        )


def test_a_plan_whose_levers_were_logged_on_a_column_outside_its_state_is_not_evaluated() -> None:
    # The policy chased demand, which the plan's state does not carry: no policy of the state is
    # the logger, and importance weights would be wrong however many episodes there were.
    panel = _panel()
    chased = _prescribe(panel, CausalGraph.from_edges(EDGES))
    asserted = _prescribe(panel, ("demand",))

    with pytest.raises(DecisionError, match=r"logged on \['demand'\], outside the plan's state"):
        chased.evaluate(panel)
    with pytest.raises(DecisionError, match=r"logged on \['demand'\]"):
        asserted.evaluate(panel)
    with pytest.raises(DecisionError, match="does not record"):
        dataclasses.replace(chased, _columns=None).evaluate(panel)
    with pytest.raises(DecisionError, match="driver forecasts"):
        dataclasses.replace(chased, drivers=(Driver("demand", np.zeros(16)),)).evaluate(panel)
    with pytest.raises(NotIdentifiedError, match="no plan to evaluate"):
        _prescribe(panel, CausalGraph.from_edges(EDGES, latent=("demand",))).evaluate(panel)


def test_the_episodes_are_windows_cut_back_from_each_units_latest_period() -> None:
    # Unit 0 has periods 0-6; unit 1 has 0-2 and 4-8, a hole at 3 that no window may cross.
    rows = [(0, t) for t in range(7)] + [(1, t) for t in (0, 1, 2, 4, 5, 6, 7, 8)]
    panel = Panel.from_frame(
        {
            "unit": np.array([unit for unit, _ in rows]),
            "time": np.array([t for _, t in rows]),
            "x": np.array([10.0 * unit + t for unit, t in rows]),
            "a": np.array([100.0 + 10.0 * unit + t for unit, t in rows]),
        },
        unit="unit",
        time="time",
    )

    episodes = _episodes(panel, states=("x",), levers=("a",), horizon=2, time_zero="unit")

    assert episodes.x[..., 0].tolist() == [
        [4, 5, 6],
        [2, 3, 4],
        [0, 1, 2],
        [10, 11, 12],
        [16, 17, 18],
        [14, 15, 16],
    ]
    assert episodes.u[..., 0].tolist() == [
        [104, 105],
        [102, 103],
        [100, 101],
        [110, 111],
        [116, 117],
        [114, 115],
    ]
    assert episodes.units.tolist() == [0, 0, 0, 1, 1, 1]
    with pytest.raises(DecisionError, match="needs two windows of 7 consecutive periods"):
        _episodes(panel, states=("x",), levers=("a",), horizon=6, time_zero="unit")


def test_the_episodes_start_on_one_calendar_for_every_unit() -> None:
    # Periods 0-8, so windows of three periods start at 6, 4, 2 and 0. Unit 0 leaves after period
    # 5, unit 1 misses period 3, unit 2 arrives at period 1: each gives the windows it was
    # observed throughout. Cut back from its latest period, unit 0's would start at 3 and 1.
    rows = (
        [(0, t) for t in range(6)]
        + [(1, t) for t in (0, 1, 2, 4, 5, 6, 7, 8)]
        + [(2, t) for t in range(1, 9)]
    )
    panel = Panel.from_frame(
        {
            "unit": np.array([unit for unit, _ in rows]),
            "time": np.array([t for _, t in rows]),
            "x": np.array([10.0 * unit + t for unit, t in rows]),
            "a": np.array([100.0 + 10.0 * unit + t for unit, t in rows]),
        },
        unit="unit",
        time="time",
    )

    episodes = _episodes(panel, states=("x",), levers=("a",), horizon=2, time_zero="calendar")
    cut = _episodes(panel, states=("x",), levers=("a",), horizon=2, time_zero="unit")

    assert episodes.x[..., 0].tolist() == [
        [2, 3, 4],
        [0, 1, 2],
        [16, 17, 18],
        [14, 15, 16],
        [10, 11, 12],
        [26, 27, 28],
        [24, 25, 26],
        [22, 23, 24],
    ]
    assert episodes.u[..., 0].tolist() == [
        [102, 103],
        [100, 101],
        [116, 117],
        [114, 115],
        [110, 111],
        [126, 127],
        [124, 125],
        [122, 123],
    ]
    assert episodes.units.tolist() == [0, 0, 1, 1, 1, 2, 2, 2]
    assert cut.x[:2, :, 0].tolist() == [[3, 4, 5], [1, 2, 3]]
    with pytest.raises(DecisionError, match="needs two windows of 9 consecutive periods"):
        _episodes(panel, states=("x",), levers=("a",), horizon=8, time_zero="calendar")


def test_a_calendar_window_is_one_units() -> None:
    # Unit 0 is observed at periods 0-4 and unit 1 at 5-8: unit 0's period 4, then unit 1's 5 and
    # 6, span a window's periods, and are two units'.
    rows = [(0, t) for t in range(5)] + [(1, t) for t in range(5, 9)]
    panel = Panel.from_frame(
        {
            "unit": np.array([unit for unit, _ in rows]),
            "time": np.array([t for _, t in rows]),
            "x": np.array([10.0 * unit + t for unit, t in rows]),
            "a": np.zeros(len(rows)),
        },
        unit="unit",
        time="time",
    )

    episodes = _episodes(panel, states=("x",), levers=("a",), horizon=2, time_zero="calendar")

    assert episodes.x[..., 0].tolist() == [[2, 3, 4], [0, 1, 2], [16, 17, 18]]
    assert episodes.units.tolist() == [0, 0, 1]


def test_a_linear_models_linearisation_is_its_own_step_and_its_noise_the_residuals() -> None:
    # RK4 on x' = A x + B u is the degree-4 Taylor polynomial of the step, written out here.
    a, b, dt = np.array([[-0.5, 0.2], [0.0, -0.3]]), np.array([[1.0], [0.5]]), 0.1
    ha = dt * a
    powers = [np.linalg.matrix_power(ha, k) for k in range(5)]
    step = sum(p / math.factorial(k) for k, p in enumerate(powers))
    channel = dt * sum(p / math.factorial(k + 1) for k, p in enumerate(powers[:4])) @ b
    rng = np.random.default_rng(3)
    x, u = np.empty((50, 4, 2)), rng.normal(size=(50, 3, 1))
    x[:, 0] = rng.normal(size=(50, 2))
    shocks = 0.1 * rng.normal(size=(50, 3, 2))
    for t in range(3):
        x[:, t + 1] = x[:, t] @ step.T + u[:, t] @ channel.T + shocks[:, t]

    plant = _linearised(LinearDynamics(jnp.asarray(a), jnp.asarray(b)), x, u, dt)

    assert np.allclose(plant.a, step, rtol=0.0, atol=1e-14)
    assert np.allclose(plant.b, channel, rtol=0.0, atol=1e-14)
    assert np.allclose(plant.offset, 0.0, rtol=0.0, atol=1e-13)
    assert np.allclose(plant.noise, np.cov(shocks.reshape(-1, 2), rowvar=False), atol=1e-12)


def test_a_nonlinear_model_is_linearised_at_the_episodes_mean_state_and_action() -> None:
    def pendulum(t: float, x: jax.Array, u: jax.Array) -> jax.Array:
        return jnp.stack([x[1], -jnp.sin(x[0]) - 0.2 * x[1] + u[0]])

    def step(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """RK4 over 0.1, written here."""

        def f(z: np.ndarray) -> np.ndarray:
            return np.array([z[1], -np.sin(z[0]) - 0.2 * z[1] + u[0]])

        k1 = f(x)
        k2 = f(x + 0.05 * k1)
        k3 = f(x + 0.05 * k2)
        k4 = f(x + 0.1 * k3)
        return x + 0.1 / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    rng = np.random.default_rng(4)
    x, u = rng.normal(1.0, 0.3, size=(40, 3, 2)), rng.normal(0.5, 0.2, size=(40, 2, 1))
    x_bar, u_bar = x[:, :-1].reshape(-1, 2).mean(axis=0), u.reshape(-1, 1).mean(axis=0)

    plant = _linearised(pendulum, x, u, 0.1)

    h = 1e-6
    jacobian = np.stack(
        [(step(x_bar + h * e, u_bar) - step(x_bar - h * e, u_bar)) / (2.0 * h) for e in np.eye(2)],
        axis=1,
    )
    assert np.allclose(plant.a, jacobian, rtol=0.0, atol=1e-8)
    assert np.allclose(plant.a @ x_bar + plant.b @ u_bar + plant.offset, step(x_bar, u_bar))


def test_the_schedule_respects_the_box_and_its_windows_agree_with_the_magnitudes() -> None:
    result = _prescribe(_panel(), CausalGraph.from_edges(EDGES))
    schedule = result.schedule
    magnitudes = np.asarray(schedule.magnitudes)
    assert magnitudes.shape == (15, 1)
    assert magnitudes.min() >= -2.0 - 1e-6
    assert magnitudes.max() <= 2.0 + 1e-6

    window = schedule.windows()["incentive"]
    assert window is not None
    active = np.flatnonzero(np.abs(magnitudes[:, 0]) > 1e-6)
    assert window == (int(active[0]), int(active[-1]))


def test_the_two_certificate_axes_are_reported_separately() -> None:
    """A plan certified over a channel nothing identifies must not read as a certified decision."""
    identified = _prescribe(_panel(), CausalGraph.from_edges(EDGES))
    assert identified.certificate.certificate_status == "certified"
    assert identified.certificate.certified_horizon == 15
    assert identified.certificate.barrier_certified_steps is not None
    assert identified.certificate.trustworthy_steps == 15

    blocked = _prescribe(_panel(), CausalGraph.from_edges(EDGES, latent=("demand",)))
    assert blocked.certificate.certificate_status == "not_evaluated"
    assert blocked.certificate.trustworthy_steps == 0


def test_omitting_the_tolerance_switches_the_tube_off_rather_than_setting_it_to_infinity() -> None:
    panel, graph = _panel(), CausalGraph.from_edges(EDGES)
    without = prescribe(
        panel,
        levers=[Lever("incentive", lo=-2.0, hi=2.0)],
        target=Target("supply", value=1.0),
        adjustment=graph,
        horizon=8,
        dt=DT,
    )
    assert without.certificate.certificate_status == "not_evaluated"
    assert without.certificate.certified_horizon is None
    assert without.certificate.tube_rate is None
    assert without.certificate.trustworthy_steps == 0  # no constraint either, so nothing is proved
    assert without.plan is not None  # the plan exists; only its tube was not evaluated


class _Bent(eqx.Module):
    """Known physics whose slope in the state changes from one state to the next."""

    def __call__(self, t: float | jax.Array, x: jax.Array, u: jax.Array) -> jax.Array:
        return -0.3 * jnp.tanh(x)


def test_the_tube_says_whether_its_rate_holds_at_every_state() -> None:
    """A field affine in the state has one slope in it at every state, bounded over the levers'
    box, so its tube bounds the rollout from anywhere; bent known physics has its slope read at the
    start alone. The certificate, its JSON and its report say which."""
    graph = CausalGraph.from_edges(EDGES)
    affine = _prescribe(_panel(), graph)
    assert affine.certificate.tube_rate == "global"
    assert affine.to_json()["certificate"]["tube_rate"] == "global"
    (line,) = [line for line in affine.report().splitlines() if "error tube" in line]
    assert line.endswith(", global rate")

    bent = _prescribe(_panel(), graph, known=_Bent())
    assert bent.certificate.certified_horizon is not None
    assert bent.certificate.tube_rate == "local"


_BILINEAR = ControlAffineResidual(drift=jnp.array([[0.0, 0.1]]), channel=jnp.array([[[0.8, 0.5]]]))


def test_the_rate_is_the_steepest_slope_the_levers_box_allows() -> None:
    """``x' = 0.1 x + (0.8 + 0.5 x) u`` has the slope ``0.1 + 0.5 u`` in the state: 0.1 at no
    action, 0.6 at the centre of the box ``[0, 2]`` and 1.1 at its edge, where the plan is free to
    sit."""
    rate, where = _rate(_BILINEAR, jnp.array([2.0]), jnp.array([0.0]), jnp.array([2.0]))
    assert rate == pytest.approx(1.1, rel=1e-15, abs=0.0)
    assert where == "global"


@pytest.mark.parametrize(
    ("field", "where"),
    [
        (LinearDynamics(jnp.array([[0.2]]), jnp.array([[1.0]])), "global"),
        (DampedOscillator(omega=2.0, zeta=0.1), "global"),
        (
            HybridDynamics(
                known=LinearDynamics(jnp.ones((1, 1)), jnp.zeros((1, 1))), residual=_BILINEAR
            ),
            "global",
        ),
        (DrivenDynamics(_BILINEAR, jnp.ones((1, 1)), jnp.ones((2, 1)), 0.1), "global"),
        (
            ControlAffineResidual(drift=jnp.zeros((1, 3)), channel=jnp.zeros((1, 1, 3)), degree=2),
            "local",
        ),
        (
            ControlAffineResidual(
                drift=jnp.zeros((1, 2)), channel=jnp.zeros((1, 1, 3)), degree=1, channel_degree=2
            ),
            "local",
        ),
        (HybridDynamics(known=_Bent(), residual=_BILINEAR), "local"),
    ],
)
def test_a_rate_holds_at_every_state_only_on_a_field_affine_in_it(
    field: Dynamics, where: str
) -> None:
    """A drift or a channel past degree 1 in the state, or physics the check does not name, has its
    slope read at the start alone."""
    states = 2 if isinstance(field, DampedOscillator) else 1
    _, read = _rate(field, jnp.full(states, 2.0), jnp.array([-1.0]), jnp.array([1.0]))
    assert read == where


def _growing_panel() -> Panel:
    """One state that grows by a fifth each period, under a lever that chases the driver ``w``."""
    rng = np.random.default_rng(0)
    rows: dict[str, list[float]] = {name: [] for name in ("unit", "time", "x", "u", "w")}
    for unit in range(100):
        x = rng.normal()
        for period in range(6):
            w = rng.normal()
            u = 0.9 * w + rng.normal(0.0, 0.5)
            for name, value in (("unit", unit), ("time", period), ("x", x), ("u", u), ("w", w)):
                rows[name].append(value)
            x = 1.2 * x + 0.8 * u + 1.5 * w + rng.normal(0.0, 0.01)
    logs = {name: np.asarray(values) for name, values in rows.items()}
    return Panel.from_frame(logs, unit="unit", time="time", seed=0)


def test_the_tube_holds_the_gap_its_model_error_opens_on_a_growing_field() -> None:
    """The fitted field pushed by its model error along the way it grows: the plan's RK4 rollouts
    part by more than Euler's recursion's ``dt * error`` after one step, and by no more than the
    tube at any step."""
    dt, horizon = 1.0, 6
    result = prescribe(
        _growing_panel(),
        levers=[Lever("u", lo=-1.0, hi=1.0)],
        target=Target("x", value=0.5),
        adjustment=CausalGraph.from_edges([("w", "u"), ("w", "x"), ("u", "x")]),
        horizon=horizon,
        dt=dt,
        tolerance=10.0,
        x0=jnp.zeros(1),
    )
    assert result.plan is not None
    assert result.plan.uncertainty_tube is not None
    known = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    model = HybridDynamics(known=known, residual=result.model_fit.residual)
    error = _model_error(result.model_fit, 1.0)
    pushed = DrivenDynamics(model, jnp.array([[error]]), jnp.ones((horizon + 1, 1)), dt)
    moved = rollout(pushed, jnp.zeros(1), result.plan.actions, dt)
    gaps = np.abs(np.asarray(moved - result.plan.trajectory))[:, 0]
    tube = np.asarray(result.plan.uncertainty_tube)
    assert gaps[1] > dt * error
    assert np.all(gaps <= tube * (1.0 + 1e-9))
    assert result.certificate.tube_rate == "global"


@pytest.mark.parametrize(
    ("tube", "barrier", "expected"),
    [
        (15, 9, 9),  # both evaluated: the shorter prefix binds
        (15, None, 15),  # nothing is bounded, so there is no safety prefix to respect
        (None, 9, 0),  # the barrier cleared a trajectory whose error nothing bounds
        (None, None, 0),
    ],
)
def test_an_unevaluated_tube_vouches_for_no_step_whatever_the_barrier_says(
    tube: int | None, barrier: int | None, expected: int
) -> None:
    certificate = DecisionCertificate(
        identification="asserted",
        adjustment=AdjustmentSet((), "identified", "asserted by the caller"),
        identification_radius=None,
        overlap=1.0,
        certificate_status="not_evaluated" if tube is None else "certified",
        certified_horizon=tube,
        barrier_certified_steps=barrier,
        gamma_star=None,
        solver_status="converged",
        solver_iterations=1,
        regret_bound=None,
    )
    assert certificate.trustworthy_steps == expected


def test_a_state_in_the_middle_of_a_two_sided_bound_is_audited_rather_than_skipped() -> None:
    """At the midpoint the two margins tie with opposite gradients, and ``jnp.min`` averages tied
    gradients, so the barrier's was zero: that step's check vanished and ``gamma_star`` raised. A
    plan that starts in the middle of a two-sided bound sits exactly there at its first step. The
    audit now reads each margin; a held solve still reads one gradient, and it is a margin's."""
    bounds = [Constraint("wait", lo=-0.5, hi=0.5)]
    barrier = _barrier(_margins(("supply", "wait"), bounds))
    assert np.allclose(jax.grad(barrier)(jnp.zeros(2)), [0.0, 1.0])  # the first tied margin, lo's

    result = prescribe(
        _panel(n_units=40, n_periods=8),
        levers=[Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05)],
        target=Target("supply", value=1.0),
        constraints=bounds,
        adjustment=CausalGraph.from_edges(EDGES),
        horizon=15,
        dt=DT,
        tolerance=0.5,
        x0=jnp.zeros(2),
    )
    assert result.certificate.barrier_certified_steps is not None
    assert result.certificate.gamma_star is not None


def _dash() -> tuple[LinearDynamics, CausalPlan]:
    """``wait' = wait + u`` planned from 1 towards 5 in a box of 2: full push from the start."""
    model = LinearDynamics(jnp.ones((1, 1)), jnp.ones((1, 1)))
    cost = QuadraticCost(
        Q=jnp.eye(1), R=jnp.array([[1e-3]]), Qf=jnp.eye(1), x_target=jnp.array([5.0])
    )
    plan = causal_plan(model, jnp.ones(1), cost, DT, 5, jnp.array([-2.0]), jnp.array([2.0]))
    assert float(plan.actions[0, 0]) == pytest.approx(2.0)
    return model, plan


def test_a_tie_is_certified_only_if_every_tied_margin_is() -> None:
    """From the middle of ``[0.5, 1.5]`` both margins are the minimum, and a state leaving through
    ``hi`` at speed 3 clears ``lo``'s condition while failing its own: checked on the first tied
    margin alone, that step would be certified. Its ceiling is the weaker margin's, not the first's:
    the drift pushes towards ``hi``, so ``lo`` holds with no action at all."""
    model, plan = _dash()
    lo, hi = _margins(("wait",), [Constraint("wait", lo=0.5, hi=1.5)])
    first, second = (certify_safety(plan, model, margin, DT, u_max=2.0) for margin in (lo, hi))
    assert bool(first.planned_certified[0])
    assert not bool(second.planned_certified[0])

    audit = _certify(plan, model, (lo, hi), DT, gamma=1.0, u_max=2.0)
    assert not bool(audit.planned_certified[0])
    assert audit.certified_steps == 0
    assert np.isinf(first.step_gamma_star[0])
    # hi: a deficit of 0.5 on a channel of 1 at u_max 2 leaves a radius 0.75: (1 + 0.75)/(1 - 0.75)
    assert audit.step_gamma_star[0] == second.step_gamma_star[0] == pytest.approx(7.0)


def test_off_a_tie_the_audit_is_certify_safety_on_the_barrier_itself() -> None:
    """With ``lo`` the only minimum at every step, reading margins one by one changes nothing."""
    model, plan = _dash()
    margins = _margins(("wait",), [Constraint("wait", lo=0.5, hi=20.0)])
    combined = _certify(plan, model, margins, DT, gamma=2.0, u_max=2.0)
    direct = certify_safety(plan, model, _barrier(margins), DT, gamma=2.0, u_max=2.0)
    for field in dataclasses.fields(direct):
        np.testing.assert_array_equal(getattr(combined, field.name), getattr(direct, field.name))


def test_reach_prices_a_lever_by_its_box_and_not_by_its_coefficient_alone() -> None:
    panel, graph = _panel(), CausalGraph.from_edges(EDGES)
    wide = prescribe(
        panel,
        levers=[Lever("incentive", lo=-2.0, hi=2.0)],
        target=Target("supply", value=1.0),
        adjustment=graph,
        horizon=5,
        dt=DT,
    )
    narrow = prescribe(
        panel,
        levers=[Lever("incentive", lo=-0.1, hi=0.1)],
        target=Target("supply", value=1.0),
        adjustment=graph,
        horizon=5,
        dt=DT,
    )
    assert abs(wide.reach()["incentive"]) > 10 * abs(narrow.reach()["incentive"])
    assert "incentive" in wide.explain()


@functools.cache
def _on_two_scales() -> tuple[Prescription, Prescription]:
    """One log and one decision, the second with each state's zero moved: supply 100 units down
    and the wait 50, as a log in kelvin sits 273.15 from the same log in degrees Celsius."""
    logs = _logs()
    shifted = {**logs, "supply": logs["supply"] + 100.0, "wait": logs["wait"] + 50.0}
    graph = CausalGraph.from_edges(EDGES)
    here = _prescribe(Panel.from_frame(logs, unit="unit", time="time", seed=0), graph)
    there = prescribe(
        Panel.from_frame(shifted, unit="unit", time="time", seed=0),
        levers=[Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05)],
        target=Target("supply", value=101.0),
        constraints=[Constraint("wait", hi=50.5)],
        adjustment=graph,
        horizon=15,
        dt=DT,
        tolerance=0.5,
    )
    return here, there


def test_the_certificate_does_not_move_with_the_zero_of_the_state_scale() -> None:
    """The plan is the same on both scales, to the solver's tolerance, and so is how far it is
    trusted. The radius was the root mean of the channel's coefficients' variances, its value at
    ``x = 0`` among them, which the moved log reaches only by extrapolation: 0.0125 here and 0.946
    there, which certified 15 steps here and 2 there."""
    here, there = _on_two_scales()
    assert here.plan is not None
    assert there.plan is not None
    assert np.asarray(there.plan.actions) == pytest.approx(np.asarray(here.plan.actions), abs=1e-5)
    # the ridge weighs the moved log's larger coefficients: 8e-5 apart
    assert there.certificate.identification_radius == pytest.approx(
        here.certificate.identification_radius, rel=1e-3
    )
    assert there.certificate.certified_horizon == here.certificate.certified_horizon


def test_reach_reads_the_channel_at_the_state_the_plan_starts_from() -> None:
    """The channel is affine in the state, so its constant term is its value at ``x = 0``, where a
    log need never have been. Read there, the reach moved with the state's zero: 3.35 here and
    9.16 there. It is read at the start, where the plan acts first."""
    here, there = _on_two_scales()
    assert here.plan is not None
    start = here.plan.trajectory[0]
    channel = here.model_fit.residual.control_channel(start)
    assert here.reach()["incentive"] == pytest.approx(4.0 * float(channel[0, 0]), rel=1e-12)
    assert there.reach()["incentive"] == pytest.approx(here.reach()["incentive"], rel=1e-5)


def test_a_period_no_unit_logged_parts_the_periods_on_either_side_of_it() -> None:
    """Hours 22, 23, 46 and 47 are two pairs a day apart. Ranked, they were four periods in a row,
    so a home logged at those hours alone gave a transition from 19.9 to 15.0 across the day, and
    beside a home logged every hour it did not. Whole-number periods sit on the grid of their
    smallest spacing, so logs taken every other week stay consecutive."""
    alone = {
        "home": np.zeros(4, dtype=int),
        "hour": np.array([22, 23, 46, 47]),
        "temperature": np.array([20.0, 19.9, 15.0, 15.1]),
        "heater": np.ones(4),
    }
    data, _ = _transitions(
        Panel.from_frame(alone, unit="home", time="hour"),
        states=("temperature",),
        levers=("heater",),
        adjust_for=(),
    )
    assert np.asarray(data["x"])[:, 0].tolist() == [20.0, 15.0]
    assert np.asarray(data["x_next"])[:, 0].tolist() == [19.9, 15.1]

    logs = _logs(n_units=40)
    every_other = Panel.from_frame({**logs, "time": 2 * logs["time"]}, unit="unit", time="time")
    data, _ = _transitions(every_other, states=("supply",), levers=("incentive",), adjust_for=())
    assert data["x"].shape == (40 * 11, 1)


@pytest.mark.parametrize("resolution", ["ns", "us", "D"])
def test_dated_periods_part_at_a_missing_week_and_calendar_months_stay_in_a_row(
    resolution: str,
) -> None:
    """Dates sit on the grid of their smallest spacing at every resolution, though ``periods``
    holds integers at ``ns`` and dates above it: ten weeks with the fifth missing give seven
    transitions, not eight. Calendar months, 29 to 31 days apart here, lie on no such grid and are
    ranked, so twelve give eleven; on the grid of their greatest common spacing, a day, they gave
    none."""

    def count(periods: np.ndarray) -> int:
        frame = {
            "home": np.zeros(periods.size, dtype=int),
            "period": periods.astype(f"datetime64[{resolution}]"),
            "temperature": np.linspace(20.0, 15.0, periods.size),
            "heater": np.ones(periods.size),
        }
        panel = Panel.from_frame(frame, unit="home", time="period")
        data, _ = _transitions(panel, states=("temperature",), levers=("heater",), adjust_for=())
        return int(data["x"].shape[0])

    weeks = np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(10)
    assert count(np.delete(weeks, 4)) == 7
    with pytest.warns(FutureWarning, match="frequency"):
        assert count(np.arange("2024-01", "2025-01", dtype="datetime64[M]")) == 11


def _monthly(written: str) -> dict[str, np.ndarray]:
    """Two homes over the sixteen months from November 2023 to February 2025, stamped on each
    month's last day, February 2024's the 29th; home ``a`` misses March 2024, and both miss July
    2024. A home's temperature is its month's count since 1970, plus 1000 for home ``b``, so a
    transition between consecutive months moves it by one."""
    rows = [
        (home, month)
        for home in ("a", "b")
        for month in np.arange("2023-11", "2025-03", dtype="datetime64[M]").tolist()
        if month != datetime.date(2024, 7, 1) and (home, month) != ("a", datetime.date(2024, 3, 1))
    ]
    home = np.array([home for home, _ in rows])
    month = np.array([month for _, month in rows], dtype="datetime64[M]")
    last_day = (month + np.timedelta64(1, "M")).astype("datetime64[D]") - np.timedelta64(1, "D")
    return {
        "home": home,
        "month": month if written == "M" else last_day.astype(f"datetime64[{written}]"),
        "temperature": month.astype(np.int64) + 1000.0 * (home == "b"),
        "heater": np.ones(home.size),
    }


def _steps_taken(panel: Panel) -> np.ndarray:
    data, _ = _transitions(panel, states=("temperature",), levers=("heater",), adjust_for=())
    return np.asarray(data["x_next"])[:, 0] - np.asarray(data["x"])[:, 0]


@pytest.mark.parametrize("written", ["M", "D", "ns"])
def test_one_monthly_calendar_gives_the_same_transitions_however_it_is_written(
    written: str,
) -> None:
    """Declared monthly, the periods are months whether stamped as months, days or nanoseconds. A
    transition joins two consecutive months: across both years' ends and from the leap day, and
    over no month a home missed. Undeclared, month ends are ranked: July, which no home logged,
    is not seen, and June to August reads as one step for each home."""
    frame = _monthly(written)
    data, _ = _transitions(
        Panel.from_frame(frame, unit="home", time="month", frequency="M"),
        states=("temperature",),
        levers=("heater",),
        adjust_for=(),
    )
    x = np.asarray(data["x"])[:, 0]
    assert np.all(np.asarray(data["x_next"])[:, 0] - x == 1.0)
    december_2023, february_2024, june_2024, december_2024 = 647, 649, 653, 659
    assert sorted(x.tolist()) == sorted(
        [*range(646, 649), *range(651, 653), *range(655, 661)]
        + [1000 + month for month in [*range(646, 653), *range(655, 661)]]
    )
    assert {december_2023, december_2024, 1000 + december_2023, 1000 + december_2024} <= set(x)
    assert 1000 + february_2024 in x  # b's leap day to its March 31st
    assert february_2024 not in x  # a missed March
    assert june_2024 not in x  # no home logged July
    assert 1000 + june_2024 not in x
    if written != "M":
        with pytest.warns(FutureWarning, match="frequency"):
            ranked = Panel.from_frame(frame, unit="home", time="month")
        assert sorted(_steps_taken(ranked).tolist()) == [1.0] * 24 + [2.0] * 2


@pytest.mark.parametrize(
    ("frequency", "stamps", "transitions"),
    [
        ("D", ["2024-02-28", "2024-02-29", "2024-03-01", "2024-03-03"], 2),
        ("W", ["2024-01-01", "2024-01-08", "2024-01-22", "2024-01-29"], 2),
        ("Q", ["2023-12-31", "2024-03-31", "2024-06-30", "2024-12-31"], 2),
        ("Y", ["2022-12-31", "2023-12-31", "2024-12-31", "2026-12-31"], 2),
    ],
)
def test_each_calendar_unit_parts_its_periods_at_a_missing_one(
    frequency: str, stamps: list[str], transitions: int
) -> None:
    """A day, a week, a quarter across a year's end, a year across a leap year: two of three
    consecutive pairs are a step apart, and the third is two."""
    frame = {
        "home": np.zeros(4, dtype=int),
        "month": np.array(stamps, dtype="datetime64[D]"),
        "temperature": np.arange(4.0),
        "heater": np.ones(4),
    }
    panel = Panel.from_frame(frame, unit="home", time="month", frequency=frequency)
    assert _steps_taken(panel).size == transitions


def test_a_declared_step_or_the_order_logged_replaces_the_least_spacing() -> None:
    """Weeks 0, 2, 4 and 8 sit, undeclared, on a grid of two: two transitions, 4 to 8 being two
    steps. Declared a step of one, no two are consecutive; of two, as undeclared; ``"observed"``
    ranks them, and 4 to 8 is one step."""
    frame = {
        "home": np.zeros(4, dtype=int),
        "week": np.array([0, 2, 4, 8]),
        "temperature": np.arange(4.0),
        "heater": np.ones(4),
    }

    def taken(frequency: object) -> int:
        panel = Panel.from_frame(frame, unit="home", time="week", frequency=frequency)  # type: ignore[arg-type]
        return int(_steps_taken(panel).size)

    assert [taken(None), taken(2), taken("observed")] == [2, 2, 3]
    with pytest.raises(DecisionError, match="no unit has two consecutive periods"):
        taken(1)


def _renamed_confounder(name: str) -> tuple[Panel, CausalGraph]:
    """The tests' world with its confounder's column, ``demand``, named ``name``."""
    logs = _logs(n_units=40)
    logs[name] = logs.pop("demand")
    edges = [(name if a == "demand" else a, name if b == "demand" else b) for a, b in EDGES]
    return Panel.from_frame(logs, unit="unit", time="time"), CausalGraph.from_edges(edges)


@pytest.mark.parametrize("name", ["u", "x", "x_next"])
def test_a_covariate_whose_name_the_transitions_hold_in_another_role_is_refused(name: str) -> None:
    """The fit reads the levers, the states and the states a period on as ``u``, ``x`` and
    ``x_next``, and each covariate by its column's name, from one dict. A confounder named ``u``
    replaced the lever's column there: the channel read 2.28 where the confounder's other names
    read 0.84, and the decision was refused as a lever the log never moved. Named ``x`` or
    ``x_next`` it failed inside the fit."""
    panel, graph = _renamed_confounder(name)
    with pytest.raises(DecisionError, match=f"the covariate '{name}' would be read as '{name}'"):
        _prescribe(panel, graph)


def test_a_covariate_named_x0_reads_as_under_any_other_name() -> None:
    """The start was kept in the same dict as ``x0``, after the covariates, so a confounder of that
    name failed inside the fit; the start is now computed apart from it."""
    plain = _prescribe(*_renamed_confounder("demand"))
    renamed = _prescribe(*_renamed_confounder("x0"))
    assert plain.plan is not None
    assert renamed.plan is not None
    assert np.array_equal(
        np.asarray(renamed.model_fit.residual.channel), np.asarray(plain.model_fit.residual.channel)
    )
    assert np.array_equal(np.asarray(renamed.plan.actions), np.asarray(plain.plan.actions))
    assert np.array_equal(
        np.asarray(renamed.plan.trajectory[0]), np.asarray(plain.plan.trajectory[0])
    )


def test_an_asserted_adjustment_set_that_names_a_lever_is_refused() -> None:
    """Adjusted for, the action leaves no move of its own: the decision was refused as a lever the
    log never moved, which was not the reason."""
    with pytest.raises(DecisionError, match=r"names the levers \['incentive'\]"):
        _prescribe(_panel(n_units=40), ("demand", "incentive"))


LEVERS_IN_TWO_ROLES = {
    "a lever twice": (
        {"levers": [Lever("incentive", -2.0, 2.0, 0.05), Lever("incentive", -2.0, 2.0, 0.05)]},
        r"levers named more than once: \['incentive'\]",
    ),
    "a lever as the target": (
        {"target": Target("incentive", value=1.0)},
        r"columns \['incentive'\] are named as levers and as the target or a constraint",
    ),
    "a lever under a constraint": (
        {"constraints": [Constraint("incentive", hi=1.0)]},
        r"columns \['incentive'\] are named as levers and as the target or a constraint",
    ),
}


@pytest.mark.parametrize(
    ("names", "match"), LEVERS_IN_TWO_ROLES.values(), ids=LEVERS_IN_TWO_ROLES.keys()
)
def test_a_lever_named_twice_or_as_a_state_is_refused(names: dict, match: str) -> None:
    """A lever named twice was planned as two levers on one column, each with its own action and
    cost; as the target or under a constraint, the fit's RK4 fixed point did not converge."""
    decision = {
        "levers": [Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05)],
        "target": Target("supply", value=1.0),
        "constraints": [Constraint("wait", hi=0.5)],
    } | names
    with pytest.raises(DecisionError, match=match):
        prescribe(
            _panel(n_units=40),
            adjustment=("demand",),
            horizon=15,
            dt=DT,
            tolerance=0.5,
            **decision,  # type: ignore[arg-type]
        )


def test_the_start_is_each_units_latest_state_on_a_panel_whose_units_end_apart() -> None:
    """The mean over units of each unit's state at its own latest period, whatever the rows'
    order."""
    logs = _logs(n_units=3, n_periods=4)
    kept = ~((logs["unit"] == 1) & (logs["time"] == 3))
    order = np.random.default_rng(1).permutation(int(kept.sum()))
    panel = Panel.from_frame(
        {name: column[kept][order] for name, column in logs.items()}, unit="unit", time="time"
    )
    last = [
        logs["supply"][(logs["unit"] == unit) & (logs["time"] == period)][0]
        for unit, period in ((0, 3), (1, 2), (2, 3))
    ]
    start, record = _panel_start(panel, ("supply",))
    (value,) = np.asarray(start).tolist()
    assert value == pytest.approx(sum(last) / 3.0, rel=1e-14, abs=0.0)
    assert (record.source, record.value, record.units, record.periods) == (
        "panel",
        (value,),
        3,
        (2, 3),
    )
    assert record.spread == pytest.approx((float(np.std(last)),), rel=1e-12, abs=0.0)


def test_the_record_says_where_the_plan_starts_and_over_how_many_units() -> None:
    """A mean over units that sit far apart is a start no unit is at: the record keeps the mean the
    plan starts from, how many units it is over, when they were logged and how far apart they
    sit."""
    result = _small()
    logs = _logs(n_units=40)
    last = np.stack([logs[name][logs["time"] == 11] for name in ("supply", "wait")], axis=1)
    start = result.start
    assert start is not None
    assert (start.source, start.states, start.units, start.periods) == (
        "panel",
        ("supply", "wait"),
        40,
        (11,),
    )
    assert start.value == pytest.approx(tuple(last.mean(axis=0)), rel=1e-12, abs=0.0)
    assert start.spread == pytest.approx(tuple(last.std(axis=0)), rel=1e-12, abs=0.0)
    assert np.asarray(result._start).tolist() == list(start.value)


def test_one_unit_s_start_is_its_own_state_and_sits_apart_from_nothing() -> None:
    _, record = _panel_start(_panel(n_units=1, n_periods=5), ("supply", "wait"))
    assert (record.units, record.periods, record.spread) == (1, (4,), (0.0, 0.0))
    assert _show_start(record).startswith("Start: the one unit's last logged state, from period 4:")


def test_a_given_start_is_recorded_as_given_and_an_integer_one_is_read_as_floats() -> None:
    """An integer start failed inside the fit's linearisation, which differentiates only floats."""
    result = _small(x0=[1, 0])
    assert result.start == StartState("given", ("supply", "wait"), (1.0, 0.0))
    assert np.asarray(result._start).tolist() == [1.0, 0.0]
    assert "\n\nStart: `x0` as given, `supply` 1, `wait` 0.\n" in result.report()
    assert result.to_json()["start"] == {
        "source": "given",
        "states": ["supply", "wait"],
        "value": [1.0, 0.0],
        "units": None,
        "periods": [],
        "spread": None,
    }


@pytest.mark.parametrize(
    ("x0", "match"),
    [
        ([0.0], r"x0 has shape \(1,\); the plan starts from one value per state, 2 for "),
        ([0.0, 0.0, 0.0], r"x0 has shape \(3,\)"),
        ([[0.0, 0.0]], r"x0 has shape \(1, 2\)"),
        (0.0, r"x0 has shape \(\)"),
        ([math.nan, 0.0], r"x0 is \[nan, 0\.0\], and a start must be finite"),
        ([0.0, -math.inf], r"x0 is \[0\.0, -inf\]"),
    ],
)
def test_a_start_that_is_not_one_finite_value_per_state_is_refused(x0: object, match: str) -> None:
    """A start of the wrong shape failed inside jax, and a nan or an infinite one was refused as a
    drift or a channel that is not a number."""
    with pytest.raises(DecisionError, match=match):
        _small(x0=x0)


@pytest.mark.parametrize(
    ("periods", "logged"),
    [
        ((7,), "from period 7: "),
        ((6, 7), "from periods 6, 7: "),
        ((5, 6, 7), "from periods 5, 6, 7: "),
        ((4, 5, 6, 7), "from 4 periods, 4 to 7: "),
    ],
)
def test_the_report_names_the_periods_the_units_last_states_were_logged_at(
    periods: tuple[int, ...], logged: str
) -> None:
    record = StartState("panel", ("supply",), (0.5,), units=9, periods=periods, spread=(0.25,))
    assert _show_start(record) == (
        "Start: the mean (standard deviation) over 9 units of each unit's last logged state, "
        f"{logged}`supply` 0.5 (0.25)."
    )


def test_a_dated_start_s_periods_go_to_json_as_iso_8601_or_as_the_panel_holds_them() -> None:
    """``periods`` holds integers at ``ns`` and dates or datetimes above it; JSON has neither."""
    result = _small()
    days = np.datetime64("2024-03-01") + np.arange(4).astype("timedelta64[D]")
    for resolution, label in (
        ("D", "2024-03-04"),
        ("s", "2024-03-04T00:00:00"),
        ("ns", 1_709_510_400_000_000_000),
    ):
        panel = Panel.from_frame(
            {
                "home": np.zeros(4, dtype=int),
                "day": days.astype(f"datetime64[{resolution}]"),
                "temperature": np.linspace(20.0, 15.0, 4),
            },
            unit="home",
            time="day",
        )
        _, record = _panel_start(panel, ("temperature",))
        written = dataclasses.replace(result, start=record).to_json()["start"]
        assert json.loads(json.dumps(written, allow_nan=False))["periods"] == [label]


@pytest.mark.parametrize(
    ("period", "label"),
    [
        (np.int64(3), 3),
        (2.5, 2.5),
        ("2024-W10", "2024-W10"),
        (datetime.date(2024, 3, 4), "2024-03-04"),
        (datetime.datetime(2024, 3, 4, 6, 30), "2024-03-04T06:30:00"),
        (datetime.timedelta(days=1), "1 day, 0:00:00"),
    ],
)
def test_a_period_s_label_is_a_json_number_or_text(period: object, label: object) -> None:
    assert _json_label(period) == label


def test_an_evaluation_window_does_not_span_a_period_no_unit_logged() -> None:
    logs = _logs(n_units=40)
    kept = logs["time"] != 5
    panel = Panel.from_frame(
        {name: column[kept] for name, column in logs.items()}, unit="unit", time="time"
    )
    for time_zero in ("calendar", "unit"):
        episodes = _episodes(
            panel, states=("supply",), levers=("incentive",), horizon=3, time_zero=time_zero
        )
        held = {float(value) for value in logs["supply"][logs["time"] == 4]}
        after = {float(value) for value in logs["supply"][logs["time"] == 6]}
        for window in episodes.x[:, :, 0]:
            pairs = set(zip(window[:-1].tolist(), window[1:].tolist(), strict=True))
            assert not any(a in held and b in after for a, b in pairs), time_zero


def test_the_report_and_the_json_carry_the_same_decision() -> None:
    result = _prescribe(_panel(), CausalGraph.from_edges(EDGES))
    payload = json.loads(json.dumps(result.to_json()))
    assert payload["schema_version"] == 2
    assert payload["target"] == "supply"
    assert payload["levers"] == ["incentive"]
    assert np.allclose(payload["schedule"], np.asarray(result.plan.actions))
    assert payload["certificate"]["adjusted_for"] == ["demand"]
    assert payload["provenance"]["data_sha256"] == result.provenance.data_sha256
    start = result.start
    assert start is not None
    assert start.spread is not None
    assert payload["start"] == {
        "source": "panel",
        "states": ["supply", "wait"],
        "value": list(start.value),
        "units": 200,
        "periods": [11],
        "spread": list(start.spread),
    }
    # the fitted channel reads the state, so the curvature behind the regret bound is a sample
    assert payload["certificate"]["regret_status"] == "diagnostic"
    assert result.certificate.regret_status == "diagnostic"

    report = result.report()
    assert "# Prescription for `supply`" in report
    assert (
        "## Decision\n\nStart: the mean (standard deviation) over 200 units of each unit's last "
        f"logged state, from period 11: `supply` {start.value[0]:.4g} ({start.spread[0]:.4g}), "
        f"`wait` {start.value[1]:.4g} ({start.spread[1]:.4g}).\n\n| lever |" in report
    )
    assert "- regret bound on the fitted model: " in report
    assert (
        report.split("- regret bound on the fitted model: ")[1]
        .split("\n")[0]
        .endswith(", diagnostic")
    )
    assert "Trustworthy prefix: 15 steps" in report
    assert "- logger check: passed (p = " in report
    assert result.provenance.data_sha256[:16] in report
    run = result.run
    assert run is not None
    assert payload["run"] == run.to_json()
    versions = ", ".join(f"{name} {version}" for name, version in run.versions)
    assert (
        f"- run: on {jax.default_backend()} ({jax.devices()[0].device_kind}), x64=True, "
        "fit in float64, solve in float64, seed=0, integrator=rk4\n"
        "- settings: folds=2, degree=1, channel_degree=1, nuisance_degree=2, ridge=1e-06, "
        "steps=10000, tolerance=0.5, hold_constraints=False, max_levers=none\n"
        f"- versions: {versions}"
    ) in report


def _small(panel: Panel | None = None, **kwargs: object) -> Prescription:
    return prescribe(
        _panel(n_units=40) if panel is None else panel,
        levers=[Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05)],
        target=Target("supply", value=1.0),
        horizon=6,
        dt=DT,
        **{  # type: ignore[arg-type]
            "constraints": [Constraint("wait", hi=0.5)],
            "adjustment": CausalGraph.from_edges(EDGES),
            "tolerance": 0.5,
            **kwargs,
        },
    )


def _refused(result: Prescription) -> Prescription:
    """The record of a plan whose regret bound was refused, at the certificate and at each step of
    its selection: an objective not convex where its curvature was read. No small world reaches
    one, so it is set by hand, as a review did."""
    assert result.selection is not None
    return dataclasses.replace(
        result,
        certificate=dataclasses.replace(
            result.certificate, regret_bound=math.inf, regret_status="refused"
        ),
        selection=dataclasses.replace(
            result.selection,
            steps=tuple(
                dataclasses.replace(step, regret_bound=math.inf, regret_status="refused")
                for step in result.selection.steps
            ),
        ),
    )


STATES = {
    "a tube not evaluated": (
        lambda: _small(tolerance=None),
        {"certificate_status": "not_evaluated", "certified_horizon": None},
    ),
    "an uncertified tube": (lambda: _small(tolerance=1e-9), {"certificate_status": "uncertified"}),
    "a diagnostic regret bound": (lambda: _small(), {"regret_status": "diagnostic"}),
    "a refused regret bound": (
        lambda: _refused(_small(max_levers=1)),
        {"regret_bound": None, "regret_status": "refused"},
    ),
    "a finite ceiling": (
        lambda: _small(constraints=[Constraint("wait", hi=0.05)]),
        {"gamma_star_status": "finite"},
    ),
    "an infinite ceiling": (
        lambda: _small(constraints=[Constraint("wait", hi=100.0)]),
        {"gamma_star": None, "gamma_star_status": "every_level"},
    ),
    "a nan ceiling": (
        lambda: _small(constraints=[Constraint("wait", hi=-5.0)]),
        {"gamma_star": None, "gamma_star_status": "no_level"},
    ),
    "no plan": (
        lambda: _small(adjustment=CausalGraph.from_edges(EDGES, latent=("demand",))),
        {"regret_bound": None, "regret_status": None, "gamma_star_status": None},
    ),
    "a plan the log cannot keep to": (
        lambda: _prescribe_policy(
            _policy_logs(0.5, 0.6),
            [Lever("u", lo=-2.0, hi=2.0, unit_cost=0.05), Lever("v", lo=-2.0, hi=2.0)],
        ),
        {"estimability": "not_estimable", "regret_bound": None},
    ),
}


@pytest.mark.parametrize(("build", "reads"), STATES.values(), ids=STATES.keys())
def test_every_state_of_the_record_is_strict_json(build, reads: dict[str, object]) -> None:
    """A number that is not finite is null, and the status beside it says which. Python's ``json``
    wrote ``Infinity`` and ``NaN`` for a refused regret bound and for either ceiling, which JSON
    does not have (ADR 0055)."""
    result = build()
    record = result.to_json()
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert {key: record["certificate"][key] for key in reads} == reads
    assert record["certificate"]["gamma_star_status"] == result.certificate.gamma_star_status
    if reads.get("gamma_star_status") == "finite":
        assert record["certificate"]["gamma_star"] == result.certificate.gamma_star > 1.0
    if reads.get("regret_status") == "refused":  # and so is each step of its selection
        assert [
            (step["regret_bound"], step["regret_status"]) for step in record["selection"]["steps"]
        ] == [(None, "refused")]
    # the run's record rides along in every state, its solve read off the plan, none where none ran
    plan = result.plan
    assert (
        record["run"]["solve_dtype"],
        record["run"]["solver_status"],
        record["run"]["solver_iterations"],
    ) == (
        (None, None, None)
        if plan is None
        else ("float64", plan.solver_status, plan.solver_iterations)
    )


def _recorded(real: Any, calls: list[tuple[inspect.BoundArguments, Any]]) -> Any:
    """``real``, keeping each call bound to its signature with the defaults filled in, beside what
    it returned."""
    signature = inspect.signature(real)

    def call(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        out = real(*args, **kwargs)
        calls.append((bound, out))
        return out

    return call


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[tuple[inspect.BoundArguments, Any]]]:
    """What ``prescribe`` gave the fit and the planner, read off each call rather than off the
    record under test."""
    seen: dict[str, list[tuple[inspect.BoundArguments, Any]]] = {"fit": [], "plan": []}
    monkeypatch.setattr(
        "chc.decision.fit_causal_residual", _recorded(fit_causal_residual, seen["fit"])
    )
    monkeypatch.setattr("chc.decision.causal_plan", _recorded(causal_plan, seen["plan"]))
    return seen


@pytest.fixture(scope="module")
def default_run() -> RunProvenance:
    """The run's record at ``_small``'s defaults, before any test moves a setting."""
    run = _small().run
    assert run is not None
    return run


def _check_against(
    calls: dict[str, list[tuple[inspect.BoundArguments, Any]]], run: RunProvenance
) -> None:
    """Every field of ``run`` but the precision and the versions is what a call was given, or read
    off what it ran on or returned: the device and the dtype of the transitions the fit read, and
    the dtype of the actions of the one solve and why it stopped, after how many steps."""
    ((fit, _),) = calls["fit"]
    given = fit.arguments
    channel_degree = given["degree"] if given["channel_degree"] is None else given["channel_degree"]
    (device,) = given["data"]["x"].devices()
    assert (
        run.backend,
        run.device_kind,
        run.fit_dtype,
        run.seed,
        run.integrator,
        run.folds,
        run.degree,
        run.channel_degree,
        run.nuisance_degree,
        run.ridge,
    ) == (
        device.platform,
        device.device_kind,
        str(given["data"]["x"].dtype),
        given["seed"],
        given["integrator"],
        given["folds"],
        given["degree"],
        channel_degree,
        given["nuisance_degree"],
        given["ridge"],
    )
    solves = [
        (
            bound.arguments["steps"],
            bound.arguments["tolerance"],
            bound.arguments["barrier"] is not None,
            str(plan.actions.dtype),
            plan.solver_status,
            plan.solver_iterations,
        )
        for bound, plan in calls["plan"]
    ]
    tolerance = math.inf if run.tolerance is None else run.tolerance  # no tube is an infinite one
    recorded = (
        run.steps,
        tolerance,
        run.hold_constraints,
        run.solve_dtype,
        run.solver_status,
        run.solver_iterations,
    )
    assert solves == ([] if run.solve_dtype is None else [recorded])


def _without_the_stop(run: RunProvenance) -> RunProvenance:
    """``run`` without why the descent stopped and after how many steps, which a changed fit moves
    as well: :func:`_check_against` reads them off the planner instead."""
    return dataclasses.replace(run, solver_status=None, solver_iterations=None)


@pytest.mark.parametrize("x64", [True, False], ids=["x64", "x32"])
def test_the_run_records_what_the_fit_and_the_solve_were_given(calls, x64: bool) -> None:
    """A review found ``x64`` recorded True beside a fit on float32 transitions: the panel's flag,
    read when it was built, standing in for the run's. The panel is built in float64 here and its
    record keeps saying so; the run's says what the run did, field by field what the fit and the
    planner were given, defaults included, and why the planner stopped as the plan it returned
    says. The ridge and the steps ``prescribe`` passes are the fit's and the planner's own
    defaults, so passing them moved no number."""
    panel = _panel(n_units=40)
    with enable_x64(x64):
        result = _small(panel)
    run = result.run
    assert run is not None
    assert run.x64 is x64
    assert result.provenance.x64 is True
    assert run.fit_dtype == run.solve_dtype == ("float64" if x64 else "float32")
    _check_against(calls, run)
    defaults = (
        inspect.signature(fit_causal_residual).parameters["ridge"].default,
        inspect.signature(causal_plan).parameters["steps"].default,
    )
    assert (run.ridge, run.steps) == defaults
    certificate = result.certificate
    assert (run.solver_status, run.solver_iterations) == (
        certificate.solver_status,
        certificate.solver_iterations,
    )


CHANGES = {
    "folds": ({"folds": 3}, {}, {"folds": 3}),
    "a numpy seed": ({"seed": np.int64(1)}, {}, {"seed": 1}),
    "integrator": ({"integrator": "euler"}, {}, {"integrator": "euler"}),
    "no tube": ({"tolerance": None}, {}, {"tolerance": None}),
    "held constraints": ({"hold_constraints": True}, {}, {"hold_constraints": True}),
    "max_levers": ({"max_levers": 1}, {}, {"max_levers": 1}),
    "ridge": ({}, {"_RIDGE": 1e-4}, {"ridge": 1e-4}),
    "steps": ({}, {"_PLAN_STEPS": 7}, {"steps": 7}),
    "nuisance degree": ({}, {"_NUISANCE_DEGREE": 1}, {"nuisance_degree": 1}),
}


@pytest.mark.parametrize(("arguments", "constants", "moved"), CHANGES.values(), ids=CHANGES.keys())
def test_a_changed_setting_moves_its_own_field_of_the_run_record_and_no_other(
    calls, default_run: RunProvenance, monkeypatch, arguments, constants, moved
) -> None:
    """Whether the caller passes the setting or the library fixes it, it reaches the record as the
    fit or the planner was given it, and the record stays plain JSON: a numpy seed reads as an int,
    which ``json`` can write. The report prints no ``None``: no tube and no cap read ``none``."""
    for name, value in constants.items():
        monkeypatch.setattr(f"chc.decision.{name}", value)
    result = _small(**arguments)
    run = result.run
    assert run is not None
    assert _without_the_stop(run) == _without_the_stop(dataclasses.replace(default_run, **moved))
    _check_against(calls, run)
    assert json.loads(json.dumps(run.to_json(), allow_nan=False)) == run.to_json()
    assert "None" not in result.report()


def test_the_run_reads_the_degrees_off_the_fit(calls, monkeypatch: pytest.MonkeyPatch) -> None:
    """``prescribe`` leaves the drift's and the channel's feature degrees to the fit, so the record
    reads them off the fit it got: one made at other degrees is recorded at those."""
    other = functools.partial(fit_causal_residual, degree=2, channel_degree=1)
    monkeypatch.setattr("chc.decision.fit_causal_residual", _recorded(other, calls["fit"]))
    run = _small().run
    assert run is not None
    assert (run.degree, run.channel_degree) == (2, 1)
    _check_against(calls, run)


def test_a_solve_stopped_by_its_budget_is_recorded_as_stopped_there(
    calls, default_run: RunProvenance, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Why the planner stopped is the plan's own word, set where its descent stopped: by its rule
    on the default budget, and on the budget at 7 steps. The certificate reads the same."""
    monkeypatch.setattr("chc.decision._PLAN_STEPS", 7)
    result = _small()
    run = result.run
    assert run is not None
    assert default_run.solver_status == "converged"
    assert (run.solver_status, run.solver_iterations) == ("max_iterations", 7)
    certificate = result.certificate
    assert (certificate.solver_status, certificate.solver_iterations) == ("max_iterations", 7)
    _check_against(calls, run)


def test_the_run_names_the_device_the_fit_ran_on_and_not_jax_s_default(
    calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inside a ``jax.default_device`` block the fit runs on the block's device, which on a machine
    with a GPU need not be the default backend's. jax is made to name a GPU as its default here, as
    it would there, and the record names the device the fit's transitions sat on."""
    gpu = types.SimpleNamespace(platform="gpu", device_kind="a GPU")
    monkeypatch.setattr(jax, "default_backend", lambda: gpu.platform)
    monkeypatch.setattr(jax, "devices", lambda backend=None: [gpu])
    run = _small().run
    assert run is not None
    assert (run.backend, run.device_kind) != (gpu.platform, gpu.device_kind)
    _check_against(calls, run)


def test_the_run_records_the_installed_versions_of_what_computed_it(
    default_run: RunProvenance,
) -> None:
    """From each distribution's metadata, which the installer writes, and not from a module's
    ``__version__``, a second copy that can lag."""
    names = ("causal-hybrid-control", "jax", "jaxlib", "numpy", "scipy")
    expected = {name: metadata.version(name) for name in names}
    assert dict(default_run.versions) == expected
    assert default_run.to_json()["versions"] == expected


def test_the_run_goes_to_json_as_plain_values(default_run: RunProvenance) -> None:
    """Under ``allow_nan=False``, as ADR 0055 holds the record to, inside the prescription's and on
    its own: the versions an object and an infinite tolerance null, as any number that is not
    finite is; the certificate's status says a tube was evaluated, where no tolerance would have
    evaluated none."""
    result = _small(tolerance=math.inf)
    run = result.run
    assert run is not None
    assert run == dataclasses.replace(default_run, tolerance=math.inf)
    assert result.certificate.certificate_status == "certified"
    record = result.to_json()
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert json.loads(json.dumps(run.to_json(), allow_nan=False)) == record["run"]
    assert result.plan is not None
    assert record["run"] == {
        "x64": True,
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "fit_dtype": "float64",
        "solve_dtype": "float64",
        "solver_status": result.plan.solver_status,
        "solver_iterations": result.plan.solver_iterations,
        "seed": 0,
        "integrator": "rk4",
        "versions": dict(default_run.versions),
        "folds": 2,
        "degree": 1,
        "channel_degree": 1,
        "nuisance_degree": 2,
        "ridge": 1e-6,
        "steps": 10_000,
        "tolerance": None,
        "hold_constraints": False,
        "max_levers": None,
    }


def test_the_reported_gamma_names_its_sensitivity_model() -> None:
    """Rosenbaum's Gamma is another model's, and docs/concepts/gamma.md gives the bracket."""
    report = _prescribe(_panel(), CausalGraph.from_edges(EDGES)).report()
    (line,) = [line for line in report.splitlines() if "gamma*" in line]
    assert line.endswith("(marginal sensitivity model)")


def test_the_certificate_states_the_gamma_its_barrier_prefix_was_audited_at() -> None:
    """The certified steps are counted at the caller's ``gamma``, and ``gamma*`` is where no action
    clears the barrier; printed without the first, the steps read as if nothing had been priced."""
    result = _prescribe(_panel(), CausalGraph.from_edges(EDGES), gamma=1.5)
    assert result.certificate.gamma == 1.5
    (line,) = [line for line in result.report().splitlines() if "gamma*" in line]
    assert " at gamma 1.5, gamma* " in line
    assert result.to_json()["certificate"]["gamma"] == 1.5


def test_no_gamma_is_stated_where_no_bound_was_audited() -> None:
    result = prescribe(
        _panel(),
        levers=[Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05)],
        target=Target("supply", value=1.0),
        adjustment=CausalGraph.from_edges(EDGES),
        horizon=15,
        dt=DT,
        tolerance=0.5,
        gamma=1.5,
    )
    assert result.certificate.barrier_certified_steps is None
    assert result.certificate.gamma is None
    (line,) = [line for line in result.report().splitlines() if "gamma*" in line]
    assert "at gamma" not in line


def test_an_unbalanced_panel_still_yields_the_transitions_on_either_side_of_a_hole() -> None:
    logs = _logs(n_units=40)
    keep = ~((logs["unit"] == 0) & (logs["time"] == 5))
    punched = {name: column[keep] for name, column in logs.items()}
    panel = Panel.from_frame(punched, unit="unit", time="time")
    assert not panel.is_balanced
    result = _prescribe(panel, CausalGraph.from_edges(EDGES))
    assert result.plan is not None  # the hole cost two transitions, not the whole unit


def test_the_arguments_that_cannot_mean_anything_are_refused() -> None:
    panel = _panel(n_units=20, n_periods=4)
    graph = CausalGraph.from_edges(EDGES)
    with pytest.raises(ValueError, match="at least one lever"):
        prescribe(
            panel, levers=[], target=Target("supply", 1.0), adjustment=graph, horizon=3, dt=DT
        )
    with pytest.raises(KeyError, match="not in the panel"):
        prescribe(
            panel,
            levers=[Lever("bonus", -1.0, 1.0)],
            target=Target("supply", 1.0),
            adjustment=graph,
            horizon=3,
            dt=DT,
        )
    with pytest.raises(ValueError, match=r"constrained more than once: \['wait'\]"):
        prescribe(
            panel,
            levers=[Lever("incentive", -1.0, 1.0)],
            target=Target("supply", 1.0),
            constraints=[Constraint("wait", lo=-1.0), Constraint("wait", hi=2.0)],
            adjustment=graph,
            horizon=3,
            dt=DT,
        )
    with pytest.raises(ValueError, match="no constraint was given to hold"):
        prescribe(
            panel,
            levers=[Lever("incentive", -1.0, 1.0)],
            target=Target("supply", 1.0),
            hold_constraints=True,
            adjustment=graph,
            horizon=3,
            dt=DT,
        )
    with pytest.raises(ValueError, match="above hi"):
        Lever("incentive", lo=1.0, hi=-1.0)
    for cap in (-0.1, float("nan")):
        with pytest.raises(ValueError, match="non-negative distance"):
            Lever("incentive", lo=-1.0, hi=1.0, cap_per_step=cap)
    with pytest.raises(ValueError, match="bounds nothing"):
        Constraint("wait")
    with pytest.raises(KeyError, match="not in the panel"):
        _prescribe(panel, ("weather",))


def test_a_rate_limit_moves_the_schedule_rather_than_annotating_it() -> None:
    """L3.1: a capped lever changes what is prescribed, and binds where the free plan jumps."""
    panel = _panel()
    graph = CausalGraph.from_edges(EDGES)
    free = np.asarray(_prescribe(panel, graph).schedule.magnitudes)[:, 0]
    cap = 0.25 * float(np.max(np.abs(np.diff(free))))

    capped = prescribe(
        panel,
        levers=[Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05, cap_per_step=cap)],
        target=Target("supply", value=1.0),
        constraints=[Constraint("wait", hi=0.5)],
        adjustment=graph,
        horizon=15,
        dt=DT,
        tolerance=0.5,
    )
    held = np.abs(np.diff(np.asarray(capped.schedule.magnitudes)[:, 0]))
    assert held.max() <= cap + 1e-9
    assert held.max() == pytest.approx(cap, abs=1e-6)  # it binds: without the cap it is 4x this
    assert capped.certificate.solver_status == "converged"


def test_a_budget_is_priced_apart_from_the_rate_limit_held_beside_it() -> None:
    """The plan's rows are the rate limit's, then the budget's; the budget's price is the planned
    cost's slope in its amount."""
    panel = _panel()
    graph = CausalGraph.from_edges(EDGES)

    def decide(amount: float) -> Prescription:
        return prescribe(
            panel,
            levers=[Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05, cap_per_step=0.3)],
            target=Target("supply", value=1.0),
            constraints=[Constraint("wait", hi=0.5)],
            adjustment=graph,
            horizon=15,
            dt=DT,
            tolerance=0.5,
            budgets=[PeriodBudget(np.ones(1), amount, 15)],
        )

    def cost(amount: float) -> float:
        plan = decide(amount).plan
        assert plan is not None
        return plan.task_cost

    budgeted = decide(10.0)
    assert budgeted.plan is not None
    assert len(budgeted.plan.shadow_prices().rows) == 14 + 1
    assert float(np.sum(budgeted.plan.actions)) == pytest.approx(10.0, rel=1e-9)
    [[price]] = budgeted.budget_prices()
    assert price.status == "exact"
    step = 0.01
    slope = (cost(10.0 - step) - cost(10.0 + step)) / (2 * step)
    assert price.price == pytest.approx(slope, rel=1e-3)


def test_a_budget_held_beside_the_constraints_barrier_is_kept_and_reported_unpriced() -> None:
    held = _prescribe(
        _panel(),
        CausalGraph.from_edges(EDGES),
        hold_constraints=True,
        budgets=[PeriodBudget(np.ones(1), 3.0, 15)],
    )
    assert held.plan is not None
    assert float(np.sum(held.plan.actions)) <= 3.0 + 1e-9
    assert (
        "Budget of 3 per 15 steps: not priced, as the plan was held under the constraints' "
        "barrier." in held.report()
    )
    with pytest.raises(ValueError, match="barrier"):
        held.budget_prices()


def test_no_budget_is_priced_when_the_effect_is_not_identified() -> None:
    blocked = _prescribe(
        _panel(),
        CausalGraph.from_edges(EDGES, latent=("demand",)),
        budgets=[PeriodBudget(np.ones(1), 3.0, 15)],
    )
    with pytest.raises(NotIdentifiedError, match="no budget was priced"):
        blocked.budget_prices()
    assert blocked.to_json()["budgets"] == [
        {"weights": [1.0], "amount": 3.0, "period": 15, "start": 0.0}
    ]


def test_a_schedule_is_steered_for_before_it_moves() -> None:
    """L10: a target that moves inside the horizon reaches the plan before it moves.

    Supply is to stay at 0 for seven steps and be at 0.8 from the eighth. Priced against that
    schedule, its own plan costs less than half of what either constant target a caller could pass
    instead does: 0, which waits for a later call to re-plan, or 0.8, which leaves 0 at once.
    """
    panel = _panel()
    graph = CausalGraph.from_edges(EDGES)
    horizon, step, high, unit_cost = 15, 7, 0.8, 0.05
    schedule = [0.0] * step + [high] * (horizon - step)

    def decide(value: float | list[float]) -> Prescription:
        return prescribe(
            panel,
            levers=[Lever("incentive", lo=-2.0, hi=2.0, unit_cost=unit_cost)],
            target=Target("supply", value=value),
            adjustment=graph,
            horizon=horizon,
            dt=DT,
            tolerance=0.5,
        )

    def against_schedule(result: Prescription) -> float:
        """The Bolza sum with ``value[k]`` on the state after ``k + 1`` actions, written out."""
        assert result.plan is not None
        xs = np.asarray(result.plan.trajectory)[:, 0]
        us = np.asarray(result.plan.actions)[:, 0]
        rows = np.array([schedule[0], *schedule])
        return float(
            0.5 * np.sum((xs[:-1] - rows[:-1]) ** 2)
            + 0.5 * unit_cost * np.sum(us**2)
            + 0.5 * (xs[-1] - rows[-1]) ** 2
        )

    scheduled = decide(schedule)
    assert scheduled.plan is not None
    assert against_schedule(scheduled) == pytest.approx(scheduled.plan.task_cost, rel=1e-9)
    constants = min(against_schedule(decide(0.0)), against_schedule(decide(high)))
    assert against_schedule(scheduled) < 0.5 * constants
    assert float(np.asarray(scheduled.plan.trajectory)[step, 0]) > 0.2  # rising before the move

    flat, scalar = decide([1.0] * horizon), decide(1.0)
    assert flat.plan is not None
    assert scalar.plan is not None
    assert np.array_equal(np.asarray(flat.plan.actions), np.asarray(scalar.plan.actions))
    with pytest.raises(DecisionError, match="one level per step, 15 for this horizon"):
        decide(schedule[1:])


def test_held_constraints_move_the_schedule_rather_than_fail_its_audit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``wait >= -0.1`` binds: the incentive that lifts supply drains wait below it from step 0."""
    panel = _panel()
    graph = CausalGraph.from_edges(EDGES)

    def decide(hold: bool) -> Prescription:
        return prescribe(
            panel,
            levers=[Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05)],
            target=Target("supply", value=1.0),
            constraints=[Constraint("wait", lo=-0.1)],
            hold_constraints=hold,
            adjustment=graph,
            horizon=15,
            dt=DT,
            tolerance=0.5,
        )

    audited = decide(False)
    with caplog.at_level(logging.INFO, logger="chc.decision"):
        held = decide(True)
    assert audited.certificate.barrier_certified_steps == 0
    assert held.certificate.barrier_certified_steps == 15
    assert held.certificate.solver_status == "converged"
    assert held.plan is not None
    assert float(np.min(np.asarray(held.plan.trajectory)[:, 1])) >= -0.1  # states: supply, wait
    plan = caplog.records[_events(caplog).index("plan")]
    assert getattr(plan, "constraints_held", None) is True


def test_a_replanning_loop_compiles_nothing_after_its_first_held_prescription() -> None:
    """A receding-horizon loop calls ``prescribe`` from every state it reaches, and a held bound
    may move between calls, as a comfort band's edge does. The bound reaches the held solve as an
    array, so the calls after the first compile nothing; closed over, it compiled the barrier's
    programs afresh at every call, and a week of half-hour calls held gigabytes of them."""
    panel = _panel()
    graph = CausalGraph.from_edges(EDGES)

    def decide(bound: float) -> Prescription:
        return prescribe(
            panel,
            levers=[Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05)],
            target=Target("supply", value=1.0),
            constraints=[Constraint("wait", lo=bound)],
            hold_constraints=True,
            adjustment=graph,
            horizon=15,
            dt=DT,
            tolerance=0.5,
        )

    jax.clear_caches()
    first = decide(-0.1)
    compiled: list[str] = []

    class Record(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.getMessage().startswith("Compiling "):
                compiled.append(record.getMessage().split(" ")[1])

    handler, logger = Record(), logging.getLogger("jax")
    logger.addHandler(handler)
    try:
        with jax.log_compiles(True):
            later = [decide(bound) for bound in (-0.12, -0.08)]
    finally:
        logger.removeHandler(handler)
    assert compiled == []
    for held, bound in zip([first, *later], (-0.1, -0.12, -0.08), strict=True):
        assert held.plan is not None
        assert held.certificate.barrier_certified_steps == 15
        assert float(np.min(np.asarray(held.plan.trajectory)[:, 1])) >= bound - 1e-6


def test_a_bound_on_the_target_is_held_on_its_own_coordinate() -> None:
    """L10: the steered state can be bounded as well as steered, and gains no second coordinate.

    Steering supply to 1 with ``supply <= 0.5`` asks for "as close as the bound permits". Priced,
    the plan heads for 1 and the audit stops where it would cross; held, it stays under the bound.
    The fit sees one state, so the bound cost the identification nothing.
    """
    panel = _panel()
    graph = CausalGraph.from_edges(EDGES)

    def decide(hold: bool) -> Prescription:
        return prescribe(
            panel,
            levers=[Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05)],
            target=Target("supply", value=1.0),
            constraints=[Constraint("supply", hi=0.5)],
            hold_constraints=hold,
            adjustment=graph,
            horizon=15,
            dt=DT,
            tolerance=0.5,
        )

    priced, held = decide(False), decide(True)
    assert np.asarray(held.model_fit.residual.channel).shape[0] == 1  # supply alone
    assert priced.plan is not None
    assert held.plan is not None
    assert float(np.max(np.asarray(priced.plan.trajectory)[:, 0])) > 0.5
    assert priced.certificate.barrier_certified_steps is not None
    assert priced.certificate.barrier_certified_steps < 15
    assert held.certificate.barrier_certified_steps == 15
    assert held.certificate.gamma_star is not None
    assert float(np.max(np.asarray(held.plan.trajectory)[:, 0])) <= 0.5


# ---- the operational log: a decision nobody can reconstruct afterwards is not auditable ----


def _events(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The `chc_event` of each record, read with `getattr` because `extra` keys are dynamic."""
    return [str(getattr(record, "chc_event", "")) for record in caplog.records]


def test_the_logger_check_flags_a_graph_that_leaves_out_what_the_levers_read() -> None:
    """The graph without the policy's edge from demand says demand was fixed before the incentive
    and not read; the panel says otherwise, and the check says so before anything is fitted."""
    panel = _panel()
    right = _prescribe(panel, CausalGraph.from_edges(EDGES)).logger_check
    missing = CausalGraph.from_edges([edge for edge in EDGES if edge != ("demand", "incentive")])
    prescription = _prescribe(panel, missing)
    wrong = prescription.logger_check
    assert right is not None
    assert wrong is not None
    assert right.given == ("supply", "wait", "demand")
    assert right.test.p_value > 0.05
    assert wrong.given == ("supply", "wait")
    assert wrong.test.p_value <= 0.05
    strongest = np.nanargmax(np.abs(wrong.test.partial_correlation[0]))
    assert wrong.columns[strongest] == "demand"
    report = prescription.report()
    assert "strongest: `incentive` on `demand`" in report
    assert "read more than the record says, or read it through more than a quadratic" in report


def test_the_logger_check_sees_a_policy_that_keeps_part_of_its_last_incentive() -> None:
    """The graph has the right parents; the policy also reads its own past, which the weights of a
    policy of the state do not."""
    panel = Panel.from_frame(_logs(sticky=0.5), unit="unit", time="time", seed=0)
    check = _prescribe(panel, CausalGraph.from_edges(EDGES)).logger_check
    assert check is not None
    assert check.test.p_value <= 0.05
    strongest = np.nanargmax(np.abs(check.test.partial_correlation[0]))
    assert check.columns[strongest] == "incentive[t-1]"


def test_every_decision_point_leaves_a_structured_record(caplog: pytest.LogCaptureFixture) -> None:
    """Each record names its point in `chc_event`, so a handler routes on a field, not on prose."""
    with caplog.at_level(logging.INFO, logger="chc.decision"):
        _prescribe(_panel(n_units=40, n_periods=8), CausalGraph.from_edges(EDGES))
    assert _events(caplog) == [
        "adjustment",
        "logger_check",
        "fit",
        "plan",
        "estimability",
        "certificate",
    ]
    fit = caplog.records[_events(caplog).index("fit")]
    assert getattr(fit, "method", None) == "orthogonal"
    assert getattr(fit, "transitions", 0) > 0
    plan = caplog.records[_events(caplog).index("plan")]
    assert getattr(plan, "rate_limited_levers", None) == []
    assert getattr(plan, "constraints_held", None) is False
    estimability = caplog.records[_events(caplog).index("estimability")]
    assert estimability.levelno == logging.INFO
    assert getattr(estimability, "estimability", None) == "estimable"
    certificate = caplog.records[_events(caplog).index("certificate")]
    assert getattr(certificate, "regret_status", None) == "diagnostic"


def test_the_unidentified_path_warns_rather_than_falling_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No schedule is the loudest thing this library says; it must not be said only in a return."""
    graph = CausalGraph.from_edges(EDGES, latent=("demand",))
    with caplog.at_level(logging.INFO, logger="chc.decision"):
        _prescribe(_panel(n_units=40, n_periods=8), graph)
    events = _events(caplog)
    assert events.count("abort") == 1
    assert "plan" not in events  # nothing was solved, so nothing may claim to have been
    abort = caplog.records[events.index("abort")]
    assert abort.levelno == logging.WARNING
    assert "demand" in str(getattr(abort, "reason", ""))


def test_the_library_installs_no_handler_and_sets_no_level() -> None:
    """Logging configuration belongs to the application; a library that takes it steals it."""
    logger = logging.getLogger("chc.decision")
    assert logger.handlers == []
    assert logger.level == logging.NOTSET


def test_single_precision_is_warned_about_and_not_refused(caplog: pytest.LogCaptureFixture) -> None:
    """JAX is float32 by default and the harm is plant-specific, so this is a warning, not a gate.

    The precision warned about is the run's, which :attr:`Prescription.run` records: a panel built
    in float64 and planned on with ``x64`` off fits on float32 transitions, while the panel's own
    record still reads True. What the log adds is that it reaches an operator who never opens
    either record. ``enable_x64`` sets the flag for its block alone, so nothing leaks into a
    later test.
    """
    panel = _panel(n_units=40, n_periods=8)
    with caplog.at_level(logging.INFO, logger="chc.decision"), enable_x64(False):
        result = _prescribe(panel, CausalGraph.from_edges(EDGES))
    events = _events(caplog)
    assert events[0] == "precision"
    assert caplog.records[0].levelno == logging.WARNING
    assert "plan" in events  # warned, then carried on: the schedule is still produced
    assert result.provenance.x64 is True
    assert result.run is not None
    assert result.run.x64 is False
    report = result.report()
    assert "- panel: chc " in report
    device = f"{jax.default_backend()} ({jax.devices()[0].device_kind})"
    assert (
        f", x64=True, seed=0\n- run: on {device}, x64=False, fit in float32, solve in float32, "
        in report
    )


def test_a_panel_built_in_single_precision_is_not_warned_about_on_a_run_in_double(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The panel's flag says how JAX was set when the panel was built, which moves no number of the
    fit: the panel holds NumPy's columns, and the fit reads them in the run's precision."""
    with enable_x64(False):
        panel = _panel(n_units=40, n_periods=8)
    with caplog.at_level(logging.INFO, logger="chc.decision"):
        result = _prescribe(panel, CausalGraph.from_edges(EDGES))
    assert "precision" not in _events(caplog)
    assert result.provenance.x64 is False
    assert result.run is not None
    assert (result.run.x64, result.run.fit_dtype) == (True, "float64")


def test_a_mis_specified_decision_and_an_unidentified_one_are_different_types() -> None:
    """The fallback to partial identification is triggered by one of these and not by the other.

    Both stay ``ValueError`` subclasses, so nothing that caught the old type stops catching them.
    """
    assert issubclass(NotIdentifiedError, DecisionError)
    assert issubclass(DecisionError, ValueError)
    with pytest.raises(DecisionError, match="above hi"):
        Lever("incentive", lo=1.0, hi=-1.0)
    with pytest.raises(DecisionError, match="at least one lever"):
        prescribe(
            _panel(n_units=10, n_periods=4),
            levers=[],
            target=Target("supply", 1.0),
            adjustment=CausalGraph.from_edges(EDGES),
            horizon=3,
            dt=DT,
        )
    blocked = _prescribe(
        _panel(n_units=40, n_periods=8), CausalGraph.from_edges(EDGES, latent=("demand",))
    )
    with pytest.raises(NotIdentifiedError):
        _ = blocked.schedule


# --- the channel's error, summed within each unit or declared cluster, and each period ----------


def test_the_channel_s_error_sums_within_each_unit_or_declared_cluster_and_each_period() -> None:
    logs = _logs(n_units=40)
    by_unit = _prescribe(Panel.from_frame(logs, unit="unit", time="time"), ["demand"])
    certificate = by_unit.certificate
    grouping = (
        certificate.error_clustered_by,
        certificate.error_clusters,
        certificate.error_periods,
    )
    assert grouping == ("unit", 40, 11)
    clusters = by_unit.model_fit.clusters
    assert clusters is not None
    assert clusters[:, 0].tolist() == np.repeat(np.arange(40), 11).tolist()
    assert clusters[:, 1].tolist() == np.tile(np.arange(11), 40).tolist()
    assert (
        "summed within 40 groups of `unit`, within each of 11 periods, and within both, "
        "whichever reads largest (two-way CR1)" in by_unit.report()
    )
    shown = by_unit.to_json()["certificate"]
    assert (
        shown["error_clustered_by"],
        shown["error_clusters"],
        shown["error_periods"],
    ) == grouping

    regions = {**logs, "region": np.array(["north", "south", "east", "west"])[logs["unit"] % 4]}
    panel = Panel.from_frame(regions, unit="unit", time="time", cluster="region")
    by_region = _prescribe(panel, ["demand"])
    region = by_region.certificate
    assert (region.error_clustered_by, region.error_clusters, region.error_periods) == (
        "region",
        4,
        11,
    )
    assert by_region.model_fit.clusters is not None
    assert by_region.model_fit.clusters[:, 0].tolist() == np.repeat([1, 2, 0, 3] * 10, 11).tolist()
    np.testing.assert_array_equal(
        by_region.model_fit.residual.channel, by_unit.model_fit.residual.channel
    )
    assert region.identification_radius != certificate.identification_radius


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_a_panel_whose_transitions_all_start_in_one_period_sums_within_its_units_alone() -> None:
    result = _prescribe(_panel(n_units=400, n_periods=2), ["demand"])
    certificate = result.certificate
    grouping = (
        certificate.error_clustered_by,
        certificate.error_clusters,
        certificate.error_periods,
    )
    assert grouping == ("unit", 400, None)
    assert result.model_fit.clusters is not None
    assert result.model_fit.clusters.tolist() == list(range(400))
    assert "summed within 400 groups of `unit` (CR1)" in result.report()
    assert (
        "- logger check: nothing to test, every row it can test falls in one period"
        in result.report()
    )
    assert result.to_json()["certificate"]["error_periods"] is None


def test_a_panel_of_one_unit_takes_its_transitions_as_independent() -> None:
    result = _prescribe(_panel(n_units=1, n_periods=400), ["demand"])
    certificate = result.certificate
    assert certificate.identification_radius is not None
    grouping = (
        certificate.error_clustered_by,
        certificate.error_clusters,
        certificate.error_periods,
    )
    assert grouping == (None, None, None)
    assert result.model_fit.clusters is None
    report = result.report()
    assert "each transition taken as independent" in report
    assert "\n\nStart: the one unit's last logged state, from period 399: `supply` " in report


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dt": 0.0}, r"dt=0\.0 is not a step"),
        ({"dt": math.nan}, r"dt=nan is not a step"),
        ({"tolerance": math.nan}, r"tolerance=nan is not a radius"),
        ({"tolerance": -1.0}, r"tolerance=-1\.0 is not a radius"),
        ({"gamma": 0.5}, r"gamma=0\.5 is not a sensitivity level"),
        ({"gamma": math.nan}, r"gamma=nan is not a sensitivity level"),
        ({"gamma": math.inf}, r"gamma=inf is not a sensitivity level"),
    ],
)
def test_a_step_or_a_tolerance_that_is_no_number_is_refused_before_the_fit(
    kwargs: dict, match: str
) -> None:
    with pytest.raises(DecisionError, match=match):
        prescribe(
            _panel(),
            levers=[Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05)],
            target=Target("supply", value=1.0),
            adjustment=CausalGraph.from_edges(EDGES),
            horizon=5,
            **{"dt": DT, "tolerance": 0.5, **kwargs},  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: Lever("incentive", lo=math.nan, hi=2.0), r"lever 'incentive' has lo=nan"),
        (lambda: Lever("incentive", lo=-2.0, hi=math.nan), r"lever 'incentive' has hi=nan"),
        (lambda: Lever("incentive", lo=-2.0, hi=2.0, unit_cost=math.nan), r"unit_cost=nan; a"),
        (lambda: Lever("incentive", lo=-2.0, hi=2.0, unit_cost=math.inf), r"unit_cost=inf; a"),
        (lambda: Target("supply", value=math.nan), r"'supply' has value nan at entry 0"),
        (lambda: Target("supply", value=[1.0, 1.0, math.inf]), r"has value inf at entry 2"),
        (lambda: Target("supply", value=1.0, weight=math.nan), r"'supply' has weight=nan"),
        (lambda: Target("supply", value=1.0, weight=math.inf), r"'supply' has weight=inf"),
        (lambda: Constraint("wait", hi=math.nan), r"constraint on 'wait' has hi=nan"),
        (lambda: Constraint("wait", lo=math.nan, hi=0.5), r"constraint on 'wait' has lo=nan"),
    ],
)
def test_a_lever_a_target_or_a_constraint_of_no_number_is_refused(build, match: str) -> None:
    """Each passed its checks or had none. A nan bound clipped every action to nan; a nan price,
    level or weight made the task cost nan, and the solve stopped where it started, a plan that
    never moved and read certified at every level; a lone nan constraint bound went uncompared."""
    with pytest.raises(DecisionError, match=match):
        build()


def test_a_free_side_is_still_a_bound() -> None:
    lever = Lever("incentive", lo=-math.inf, hi=math.inf)
    assert (lever.lo, lever.hi) == (-math.inf, math.inf)
    assert Constraint("wait", lo=-math.inf).lo == -math.inf


def test_a_tube_whose_rate_is_not_finite_is_not_evaluated(monkeypatch, caplog) -> None:
    """A rate of ``inf`` read the tube ``[0, nan, ...]``, which certified every step."""
    monkeypatch.setattr("chc.decision._rate", lambda *args: (math.inf, "global"))
    with caplog.at_level(logging.WARNING, logger="chc.decision"):
        result = _prescribe(_panel(), CausalGraph.from_edges(EDGES))
    certificate = result.certificate
    assert certificate.certificate_status == "not_evaluated"
    assert certificate.certified_horizon is None
    assert certificate.tube_rate is None
    assert certificate.trustworthy_steps == 0
    (tube,) = [r for r in caplog.records if getattr(r, "chc_event", None) == "tube"]
    assert tube.rate == math.inf


def _as_text(values: np.ndarray) -> np.ndarray:
    """The column as pandas 3 hands over its ``str`` dtype: text among objects."""
    return values.astype(str).astype(object)


def _as_dates(values: np.ndarray) -> np.ndarray:
    return np.datetime64("2024-01-01") + np.arange(values.size)


def _past_float64(values: np.ndarray) -> np.ndarray:
    cells = np.array([Decimal(float(value)) for value in values], dtype=object)
    cells[0] = Decimal("1E+400")
    return cells


@pytest.mark.parametrize(
    ("column", "convert", "why"),
    [
        ("incentive", _as_text, "type str, not a real number"),
        ("supply", _as_dates, r"dtype datetime64\[D\], which holds dates or times"),
        ("wait", _as_text, "type str, not a real number"),
        ("demand", _past_float64, "a finite number past float64's range, where it is an infinity"),
    ],
    ids=["lever", "target", "constraint", "covariate"],
)
def test_a_column_prescribe_reads_as_numbers_is_refused_unless_it_holds_them(
    column: str, convert, why: str
) -> None:
    """The fit and the logger check read each lever, state and covariate with a float64 cast,
    which read text as the numbers it spells, a date as its count of days and a Decimal of 1E+400
    as an infinity: a plan came back from text and from dates."""
    logs = _logs(n_units=20, n_periods=6)
    logs[column] = convert(logs[column])
    panel = Panel.from_frame(logs, unit="unit", time="time")
    with pytest.raises(
        PanelError,
        match=rf"column '{column}' is .+ for unit np\.int64\(0\) at time np\.int64\(0\): {why}",
    ):
        _prescribe(panel, ("demand",))


def test_a_driver_column_of_text_is_refused() -> None:
    logs = _logs(n_units=20, n_periods=6)
    logs["demand"] = _as_text(logs["demand"])
    panel = Panel.from_frame(logs, unit="unit", time="time")
    with pytest.raises(PanelError, match=r"column 'demand' is '.+' for unit np\.int64\(0\) at"):
        _prescribe(panel, (), drivers=[Driver("demand", np.zeros(16))])


def test_the_evaluation_reads_its_panel_as_numbers_only_where_it_holds_them() -> None:
    """The windows were read with a float64 cast, so a lever logged as text was read as the
    numbers it spells, and the evaluation went on to price them."""
    logs = _logs(n_units=40, n_periods=20)
    prescription = _prescribe(Panel.from_frame(logs, unit="unit", time="time"), ())
    logs["incentive"] = _as_text(logs["incentive"])
    panel = Panel.from_frame(logs, unit="unit", time="time")
    with pytest.raises(PanelError, match=r"column 'incentive' is '.+' for unit .+: type str"):
        prescription.evaluate(panel)


def test_the_logger_check_reads_a_column_beside_the_plan_only_where_it_holds_numbers() -> None:
    """The logger check reads the columns the graph names beside the plan's, which nothing else
    reads, and read a complex one as its real part, with a warning at most."""
    logs = _logs(n_units=20, n_periods=6)
    logs["noise"] = logs["demand"] + 1j
    panel = Panel.from_frame(logs, unit="unit", time="time")
    graph = CausalGraph.from_edges([*EDGES, ("noise", "wait")])
    assert graph.adjustment_set(treatment=("incentive",), outcome="supply").covariates == (
        "demand",
    )
    with pytest.raises(PanelError, match=r"column 'noise' is .+: dtype complex128, which holds"):
        _prescribe(panel, graph)


def test_the_episodes_read_a_column_as_numbers_only_where_it_holds_them() -> None:
    logs = _logs(n_units=4, n_periods=6)
    logs["incentive"] = _as_text(logs["incentive"])
    panel = Panel.from_frame(logs, unit="unit", time="time")
    with pytest.raises(
        PanelError,
        match=r"column 'incentive' is '.+' for unit np\.int64\(0\) at time np\.int64\(0\): type",
    ):
        _episodes(panel, states=("supply",), levers=("incentive",), horizon=2, time_zero="unit")
