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

import numpy as np
import pytest

from chc.decision import (
    Constraint,
    DecisionError,
    Lever,
    NotIdentifiedError,
    Target,
    prescribe,
)
from chc.graph import CausalGraph
from chc.panel import Panel

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


def _logs(n_units: int = 200, n_periods: int = 12, seed: int = 0) -> dict[str, np.ndarray]:
    """Two zones of a driver pool under one incentive, logged by a policy that chases demand."""
    rng = np.random.default_rng(seed)
    rows: dict[str, list[float]] = {
        name: [] for name in ("unit", "time", "supply", "wait", "incentive", "demand")
    }
    for unit in range(n_units):
        supply, wait = rng.normal(0.0, 0.2), rng.normal(0.0, 0.2)
        for period in range(n_periods):
            demand = rng.normal(0.0, 1.0)
            incentive = 0.9 * demand + rng.normal(0.0, 0.5)
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
    assert result.provenance.data_sha256[:16] in report


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
    with pytest.raises(ValueError, match="both target and constraint"):
        prescribe(
            panel,
            levers=[Lever("incentive", -1.0, 1.0)],
            target=Target("supply", 1.0),
            constraints=[Constraint("supply", hi=2.0)],
            adjustment=graph,
            horizon=3,
            dt=DT,
        )
    with pytest.raises(ValueError, match="above hi"):
        Lever("incentive", lo=1.0, hi=-1.0)
    with pytest.raises(ValueError, match="bounds nothing"):
        Constraint("wait")
    with pytest.raises(KeyError, match="not in the panel"):
        _prescribe(panel, ("weather",))


# ---- the operational log: a decision nobody can reconstruct afterwards is not auditable ----


def _events(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The `chc_event` of each record, read with `getattr` because `extra` keys are dynamic."""
    return [str(getattr(record, "chc_event", "")) for record in caplog.records]


def test_every_decision_point_leaves_a_structured_record(caplog: pytest.LogCaptureFixture) -> None:
    """Each record names its point in `chc_event`, so a handler routes on a field, not on prose."""
    with caplog.at_level(logging.INFO, logger="chc.decision"):
        _prescribe(_panel(n_units=40, n_periods=8), CausalGraph.from_edges(EDGES))
    assert _events(caplog) == ["adjustment", "fit", "plan", "certificate"]
    fit = caplog.records[_events(caplog).index("fit")]
    assert getattr(fit, "method", None) == "orthogonal"
    assert getattr(fit, "transitions", 0) > 0


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
