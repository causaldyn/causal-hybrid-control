"""A prescription runs as a policy: the schedule for the levers the plan moves, the log's rule of
the state for a lever the log set from it, the logged level for one the log never moved.

The logs: one state ``y`` and a confounder ``z`` the graph adjusts for, ``y`` moving by
``0.1 (-0.5 y + 0.8 u1 + 0.1 u2 + 1.5 z)`` a period. ``u1`` is set to ``-0.3 y`` under ``state``
and moved by the confounder and its own draw under ``free``; ``u2`` moves by its own draw; ``u3``
stays at 0.5 throughout, so the plan holds it there.
"""

from __future__ import annotations

import functools
import importlib
import json
from dataclasses import replace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chc.decision import (
    Lever,
    NotIdentifiedError,
    PrescribedPolicy,
    Prescription,
    Target,
    prescribe,
)
from chc.graph import CausalGraph
from chc.panel import Panel

# Public as jax.enable_x64 from jax 0.8.0; the floor, 0.4.30, has only jax.experimental.enable_x64,
# which jax 0.11 no longer has.
if hasattr(jax, "enable_x64"):
    enable_x64 = jax.enable_x64
else:
    enable_x64 = importlib.import_module("jax.experimental").enable_x64

DT = 0.1
EDGES = [("z", "y"), ("u1", "y"), ("u2", "y"), ("u3", "y"), ("z", "u1"), ("z", "u2")]


@functools.cache
def _panel(policy: str) -> Panel:
    rng = np.random.default_rng(57)
    rows: dict[str, list[float]] = {
        name: [] for name in ("unit", "time", "y", "u1", "u2", "u3", "z")
    }
    for unit in range(60):
        y = rng.normal()
        for period in range(12):
            z, noise = rng.normal(), rng.normal()
            u1 = -0.3 * y if policy == "state" else 0.9 * z + 0.5 * rng.normal()
            u2, u3 = 0.6 * noise, 0.5
            for name, value in zip(rows, (unit, period, y, u1, u2, u3, z), strict=True):
                rows[name].append(value)
            y += DT * (-0.5 * y + 0.8 * u1 + 0.1 * u2 + 1.5 * z) + rng.normal(scale=0.01)
    return Panel.from_frame(
        {name: np.asarray(values) for name, values in rows.items()}, unit="unit", time="time"
    )


@functools.cache
def _prescribe(policy: str) -> Prescription:
    return prescribe(
        _panel(policy),
        levers=[
            Lever("u1", lo=-2.0, hi=2.0, unit_cost=0.01),
            Lever("u2", lo=-2.0, hi=2.0, unit_cost=1.0),
            Lever("u3", lo=-2.0, hi=2.0),
        ],
        target=Target("y", value=1.0),
        adjustment=CausalGraph.from_edges(EDGES),
        horizon=3,
        dt=DT,
        tolerance=0.5,
        x0=jnp.ones(1),
    )


def _path(prescription: Prescription) -> np.ndarray:
    assert prescription.plan is not None
    return np.asarray(prescription.plan.trajectory, dtype=np.float64)


@pytest.mark.parametrize("policy", ["state", "free"])
def test_along_the_path_the_plan_predicts_the_policy_takes_the_schedule(policy: str) -> None:
    prescription = _prescribe(policy)
    run, path = prescription.policy(), _path(prescription)
    schedule = np.asarray(prescription.schedule.magnitudes, dtype=np.float64)
    assert run.horizon == schedule.shape[0] == 3
    for step in range(run.horizon):
        assert np.array_equal(run.actions_at(step, path[step]), schedule[step])


def test_off_the_path_a_ruled_lever_answers_the_state_and_the_others_keep_the_schedule() -> None:
    prescription = _prescribe("state")
    run, path = prescription.policy(), _path(prescription)
    assert run.ruled == ("u1",)
    assert prescription.certificate.rule_levers == ("u1",)
    for step in range(run.horizon):
        on, off = run.actions_at(step, path[step]), run.actions_at(step, path[step] + 1.0)
        assert np.array_equal(off[1:], on[1:])
        assert off[0] - on[0] == pytest.approx(-0.3, rel=1e-9, abs=0.0)  # the log's u1 = -0.3 y
        assert run.actions_at(step, [100.0])[0] == -2.0  # the rule's level clipped to the box


@pytest.mark.parametrize("policy", ["state", "free"])
def test_a_lever_the_log_never_moved_keeps_its_logged_level_in_any_state(policy: str) -> None:
    prescription = _prescribe(policy)
    run, path = prescription.policy(), _path(prescription)
    assert "u3" in prescription.certificate.unmoved_levers
    for step in range(run.horizon):
        for state in (path[step], path[step] - 3.0):
            assert run.actions_at(step, state)[2] == pytest.approx(0.5, rel=1e-12, abs=0.0)


def test_a_policy_with_no_ruled_lever_reads_no_rule() -> None:
    run = _prescribe("free").policy()
    assert run.ruled == ()
    assert not np.isnan(run.schedule).any()
    assert run.to_json()["rule"] is None


@pytest.mark.parametrize("policy", ["state", "free"])
def test_the_json_round_trip_reads_the_same_bits(policy: str) -> None:
    prescription = _prescribe(policy)
    run, path = prescription.policy(), _path(prescription)
    text = json.dumps(run.to_json(), allow_nan=False)
    back = PrescribedPolicy.from_json(json.loads(text))
    assert (back.levers, back.states, back.dt, back.ruled) == (
        run.levers,
        run.states,
        run.dt,
        run.ruled,
    )
    np.testing.assert_array_equal(back.schedule, run.schedule)
    for step in range(run.horizon):
        for state in (path[step], path[step] + 2.5, path[step] - 0.7):
            assert np.array_equal(back.actions_at(step, state), run.actions_at(step, state))
    assert json.dumps(back.to_json(), allow_nan=False) == text
    ruled = [row[0] for row in json.loads(text)["schedule"]]
    assert all(level is None for level in ruled) == (policy == "state")


def test_a_state_is_read_in_the_order_of_the_states_or_by_name() -> None:
    run = _prescribe("state").policy()
    assert run.states == ("y",)
    by_order = run.actions_at(1, [0.4])
    assert np.array_equal(run.actions_at(1, {"y": 0.4, "z": 9.0}), by_order)
    assert np.array_equal(run.actions_at(np.int64(1), np.array([0.4])), by_order)


@pytest.mark.parametrize(
    ("step", "state", "message"),
    [
        (-1, [0.4], "runs steps 0 to 2"),
        (3, [0.4], "runs steps 0 to 2"),
        (1.0, [0.4], "a step is an integer"),
        (True, [0.4], "a step is an integer"),
        (1, [float("nan")], "finite numbers"),
        (1, [float("inf")], "finite numbers"),
        (1, [0.4, 0.1], "one value for each of"),
        (1, {"z": 0.4}, r"does not name \['y'\]"),
    ],
)
def test_the_policy_refuses_a_step_or_a_state_it_cannot_run(
    step: Any, state: Any, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _prescribe("state").policy().actions_at(step, state)


def _broken(change: str) -> dict[str, Any]:
    record = json.loads(json.dumps(_prescribe("state").policy().to_json()))
    rule = record["rule"]
    match change:
        case "schema":
            record["schema_version"] = 2
        case "ruled_level":
            record["schedule"][0][0] = 0.1
        case "free_level":
            record["schedule"][1][1] = None
        case "short_row":
            record["schedule"][2] = record["schedule"][2][:2]
        case "no_steps":
            record["schedule"] = []
        case "twice":
            record["levers"] = ["u1", "u1", "u3"]
        case "unknown":
            rule["levers"] = ["u9"]
        case "coefficients":
            rule["coefficients"] = rule["coefficients"][:-1]
        case "factor":
            rule["factor"] = [-1.0]
        case "degree":
            rule["degree"] = 1.5
        case "precision":
            record["precision"] = "float16"
        case "dt":
            record["dt"] = 0.0
        case "box":
            rule["lo"], rule["hi"] = [2.0], [-2.0]
    return record


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("schema", "reads schema_version 1"),
        ("ruled_level", "'u1' reads 0.1"),
        ("free_level", "'u2' reads None"),
        ("short_row", "a row of 3 levels"),
        ("no_steps", "a row of 3 levels"),
        ("twice", "distinct names"),
        ("unknown", r"sets \['u9'\]"),
        ("coefficients", "coefficients is"),
        ("factor", "factor is finite and at least 0"),
        ("degree", "degree is a whole number"),
        ("precision", "float32 or float64"),
        ("dt", "finite step above 0"),
        ("box", "lo lies at or below its hi"),
    ],
)
def test_a_record_that_does_not_run_is_refused(change: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PrescribedPolicy.from_json(_broken(change))


def _with_box(rule: Any, *, lo: float, hi: float) -> Any:
    return eqx.tree_at(
        lambda r: (r.lo, r.hi), rule, (jnp.full_like(rule.lo, lo), jnp.full_like(rule.hi, hi))
    )


def test_an_infinite_bound_of_the_rule_reads_back_as_written() -> None:
    run = _prescribe("state").policy()
    rule = run._rule
    assert rule is not None
    open_box = replace(run, _rule=_with_box(rule, lo=-jnp.inf, hi=jnp.inf))
    record = json.loads(json.dumps(open_box.to_json(), allow_nan=False))
    assert (record["rule"]["lo"], record["rule"]["hi"]) == ([None], [None])
    back = PrescribedPolicy.from_json(record)
    assert back.actions_at(0, [100.0])[0] == open_box.actions_at(0, [100.0])[0] < -2.0


def test_a_rule_read_in_another_precision_than_jax_runs_in_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _prescribe("state").policy().to_json()
    assert record["precision"] == "float64"
    with pytest.raises(ValueError, match="read in float32 and JAX now runs in the other"):
        PrescribedPolicy.from_json({**record, "precision": "float32"})
    monkeypatch.setattr("chc.decision._x64_enabled", lambda: False)
    with pytest.raises(ValueError, match="read in float64 and JAX now runs in the other"):
        PrescribedPolicy.from_json(record)


def test_a_rule_written_in_single_precision_reads_back_to_the_bit_in_single_precision() -> None:
    with enable_x64(False):
        prescription = prescribe(
            _panel("state"),
            levers=[
                Lever("u1", lo=-2.0, hi=2.0, unit_cost=0.01),
                Lever("u2", lo=-2.0, hi=2.0, unit_cost=1.0),
                Lever("u3", lo=-2.0, hi=2.0),
            ],
            target=Target("y", value=1.0),
            adjustment=CausalGraph.from_edges(EDGES),
            horizon=3,
            dt=DT,
            tolerance=0.5,
            x0=jnp.ones(1),
        )
        run, path = prescription.policy(), _path(prescription)
        record = json.loads(json.dumps(run.to_json(), allow_nan=False))
        back = PrescribedPolicy.from_json(record)
        schedule = np.asarray(prescription.schedule.magnitudes, dtype=np.float64)
        for step in range(run.horizon):
            assert np.array_equal(run.actions_at(step, path[step]), schedule[step])
            for state in (path[step], path[step] + 2.5):
                assert np.array_equal(back.actions_at(step, state), run.actions_at(step, state))
        level = run.actions_at(0, [0.4])[0]
    assert record["precision"] == "float32"
    assert level == float(np.float32(level))  # a float32 number, widened


def test_a_prescription_with_no_plan_has_no_policy() -> None:
    with pytest.raises(NotIdentifiedError, match="no policy was computed"):
        replace(_prescribe("free"), plan=None).policy()


def test_a_prescription_prescribe_did_not_build_has_no_policy() -> None:
    with pytest.raises(ValueError, match="build it with prescribe"):
        replace(_prescribe("free"), _columns=None).policy()
