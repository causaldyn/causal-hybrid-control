"""The facade runs the whole chain, and the chain's failures survive being wrapped in it.

The load-bearing test is the three-arm comparison: the same logs, read three ways. With the
confounder adjusted for the channel is recovered; with an empty adjustment asserted it is nearly
three times too large; with the confounder declared latent there is no schedule at all. If wrapping
the layers in one call ever loses that separation, this file fails.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math

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
    Target,
    _barrier,
    _certify,
    _episodes,
    _linearised,
    _margins,
    prescribe,
)
from chc.dynamics import LinearDynamics
from chc.graph import AdjustmentSet, CausalGraph
from chc.mpc import PeriodBudget
from chc.panel import Panel
from chc.plan import CausalPlan, causal_plan, certify_safety

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
    result = _prescribe(_panel(), graph)
    assert result.certificate.identification == "not_identified"
    assert result.plan is None
    assert result.certificate.trustworthy_steps == 0
    assert result.certificate.solver_status is None
    with pytest.raises(NotIdentifiedError, match="not identified"):
        _ = result.schedule
    assert "no schedule" in result.report().lower()


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
    assert without.certificate.trustworthy_steps == 0  # no constraint either, so nothing is proved
    assert without.plan is not None  # the plan exists; only its tube was not evaluated


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


def test_the_report_and_the_json_carry_the_same_decision() -> None:
    result = _prescribe(_panel(), CausalGraph.from_edges(EDGES))
    payload = json.loads(json.dumps(result.to_json()))
    assert payload["schema_version"] == 1
    assert payload["target"] == "supply"
    assert payload["levers"] == ["incentive"]
    assert np.allclose(payload["schedule"], np.asarray(result.plan.actions))
    assert payload["certificate"]["adjusted_for"] == ["demand"]
    assert payload["provenance"]["data_sha256"] == result.provenance.data_sha256

    report = result.report()
    assert "# Prescription for `supply`" in report
    assert "Trustworthy prefix: 15 steps" in report
    assert "- logger check: passed (p = " in report
    assert result.provenance.data_sha256[:16] in report


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
    assert _events(caplog) == ["adjustment", "logger_check", "fit", "plan", "certificate"]
    fit = caplog.records[_events(caplog).index("fit")]
    assert getattr(fit, "method", None) == "orthogonal"
    assert getattr(fit, "transitions", 0) > 0
    plan = caplog.records[_events(caplog).index("plan")]
    assert getattr(plan, "rate_limited_levers", None) == []
    assert getattr(plan, "constraints_held", None) is False


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

    The precision is on :class:`~chc.panel.Provenance` of every result either way; what the log adds
    is that it reaches an operator who never opens the provenance. Forged here rather than by
    flipping ``jax_enable_x64``, which is process-global and would leak into every later test.
    """
    panel = _panel(n_units=40, n_periods=8)
    single = dataclasses.replace(panel, provenance=dataclasses.replace(panel.provenance, x64=False))
    with caplog.at_level(logging.INFO, logger="chc.decision"):
        result = _prescribe(single, CausalGraph.from_edges(EDGES))
    events = _events(caplog)
    assert events[0] == "precision"
    assert caplog.records[0].levelno == logging.WARNING
    assert "plan" in events  # warned, then carried on: the schedule is still produced
    assert result.provenance.x64 is False


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
