"""A decision record that replays: ``Prescription.to_json`` holds what ``prescribe`` was asked, and
``Prescription.from_json`` asks it again and gives the record back, every number to the bit, or
refuses and names the field.

Three shapes: the quickstart's panel; two levers under caps, a budget, a target schedule, a logged
draw and named neighbours; and a held constraint on a graph whose edges carry their provenance,
with a known model, a driver, clusters and ``max_levers``. Between them every argument that moves
the result is off its default, so a replay that dropped one would not give its record back.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chc.decision as decision
import chc.evaluation as decision_evaluation
from chc.cost import QuadraticCost
from chc.decision import (
    Constraint,
    DecisionError,
    Driver,
    Lever,
    Prescription,
    Target,
    prescribe,
)
from chc.dynamics import LinearDynamics
from chc.evaluation import (
    EVALUATION_SCHEMA_VERSION,
    AffinePolicy,
    LinearGaussianPlant,
    evaluate_plan,
)
from chc.graph import CausalGraph, EdgeProvenance
from chc.misspecification import _clustered_squares, planned_cost_bias
from chc.panel import Panel
from chc.plan import PlannedCostBias
from chc.residual import ControlAffineResidual

DT = 0.1


def _quickstart_logs(regions: int = 200, weeks: int = 12, seed: int = 0) -> dict[str, np.ndarray]:
    """The quickstart's driver pool, ``docs/quickstart.py``'s ``simulate_logs``."""
    rng = np.random.default_rng(seed)
    names = ("region", "week", "supply", "wait", "incentive", "demand")
    rows: dict[str, list[float]] = {name: [] for name in names}
    for region in range(regions):
        supply, wait = rng.normal(0.0, 0.2), rng.normal(0.0, 0.2)
        for week in range(weeks):
            demand = rng.normal(0.0, 1.0)
            incentive = 0.9 * demand + rng.normal(0.0, 0.5)
            logged = (region, week, supply, wait, incentive, demand)
            for name, value in zip(names, logged, strict=True):
                rows[name].append(value)
            supply, wait = (
                supply
                + DT * (-0.6 * supply + 0.3 * wait + 0.8 * incentive + 1.5 * demand)
                + rng.normal(0.0, 0.01),
                wait + DT * (0.25 * wait - 0.4 * incentive) + rng.normal(0.0, 0.01),
            )
    return {name: np.asarray(values) for name, values in rows.items()}


QUICKSTART_EDGES = [
    ("demand", "incentive"),
    ("demand", "supply"),
    ("incentive", "supply"),
    ("incentive", "wait"),
    ("wait", "supply"),
]


def _two_lever_logs(units: int = 40, periods: int = 12, seed: int = 1) -> Panel:
    """Two levers set from a logged confounder ``z``, the state, and a logged draw each; units in
    markets of four; an exogenous ``temp`` that pushes ``x``."""
    rng = np.random.default_rng(seed)
    names = ("unit", "time", "x", "w", "u1", "u2", "z", "eta1", "eta2", "market", "temp")
    rows: dict[str, list[np.ndarray]] = {name: [] for name in names}
    x, w = rng.normal(0.0, 0.3, units), rng.normal(0.0, 0.3, units)
    for t in range(periods):
        z = rng.normal(0.0, 1.0, units)
        eta1, eta2 = rng.normal(0.0, 0.5, (2, units))
        u1 = 0.6 * z + 0.2 * x + eta1
        u2 = -0.4 * z + eta2
        temp = np.full(units, math.sin(t / 2.0)) + rng.normal(0.0, 0.1, units)
        logged = (
            np.arange(units),
            np.full(units, t),
            x,
            w,
            u1,
            u2,
            z,
            eta1,
            eta2,
            np.arange(units) // 4,
            temp,
        )
        for name, value in zip(names, logged, strict=True):
            rows[name].append(np.asarray(value, dtype=float))
        x, w = (
            x
            + DT * (-0.5 * x + 0.8 * u1 + 0.3 * u2 + 1.0 * z + 0.5 * temp)
            + rng.normal(0.0, 0.02, units),
            w + DT * (0.2 * x - 0.3 * w + 0.5 * u2) + rng.normal(0.0, 0.02, units),
        )
    frame = {name: np.concatenate(values) for name, values in rows.items()}
    return Panel.from_frame(frame, unit="unit", time="time", seed=1)


QUICKSTART = Panel.from_frame(_quickstart_logs(), unit="region", time="week", seed=0)
TWO = _two_lever_logs()
KNOWN = LinearDynamics(jnp.array([[-0.5, 0.0], [0.0, -0.3]]), jnp.zeros((2, 2)))
PROVENANCE = {
    ("u1", "x"): EdgeProvenance("discovered", "partial_corr_test", 0.0012),
    ("x", "w"): EdgeProvenance("mechanism"),
}
HELD_GRAPH = CausalGraph.from_edges(
    [("z", "u1"), ("z", "u2"), ("z", "x"), ("u1", "x"), ("u2", "x"), ("u2", "w"), ("x", "w")],
    provenance=PROVENANCE,
)


def _quickstart(**options: Any) -> Prescription:
    return prescribe(
        QUICKSTART,
        adjustment=CausalGraph.from_edges(QUICKSTART_EDGES),
        levers=[Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05)],
        target=Target("supply", value=1.0),
        constraints=[Constraint("wait", hi=0.5)],
        horizon=15,
        dt=DT,
        tolerance=0.5,
        integrator="euler",
        **options,
    )


def _two_levers(**options: Any) -> Prescription:
    arguments: dict[str, Any] = {
        "levers": [
            Lever("u1", lo=-1.5, hi=2.0, unit_cost=0.05, cap_per_step=0.8),
            Lever("u2", lo=-1.0, hi=1.0, unit_cost=0.1),
        ],
        "target": Target("x", value=np.linspace(0.2, 0.6, 6), weight=2.0),
        "constraints": [Constraint("w", lo=-0.6)],
        "horizon": 6,
        "adjustment": ["z"],
        "budgets": [chc_budget()],
        "dt": DT,
        "gamma": 1.5,
        "radius": "relative",
        "tolerance": 0.4,
        "x0": [0.1, -0.05],
        "folds": 3,
        "seed": 5,
        "integrator": "rk4",
        "neighbours": {unit: [unit ^ 1] for unit in range(40)},
        "nuisance_knots": 4,
        "tube_actions": "plan",
        "dither": ["eta1", "eta2"],
        "planned_cost_bias": True,
    }
    arguments.update(options)
    return prescribe(TWO, **arguments)


def chc_budget():
    from chc.mpc import PeriodBudget

    return PeriodBudget(np.array([1.0, 0.5]), amount=3.0, period=3)


def _held(**options: Any) -> Prescription:
    arguments: dict[str, Any] = {
        "levers": [Lever("u1", lo=-1.0, hi=1.5, unit_cost=0.05), Lever("u2", lo=-1.0, hi=1.0)],
        "target": Target("x", value=0.5),
        "constraints": [Constraint("w", lo=-0.3, hi=0.3)],
        "hold_constraints": True,
        "horizon": 5,
        "adjustment": HELD_GRAPH,
        "known": KNOWN,
        "max_levers": 1,
        "dt": DT,
        "gamma": 1.2,
        "radius": "absolute",
        "tolerance": 0.5,
        "drivers": [Driver("temp", np.sin(np.arange(6) / 2.0))],
        "clusters": "market",
        "planned_cost_bias": True,
    }
    arguments.update(options)
    return prescribe(TWO, **arguments)


@pytest.fixture(scope="module")
def captured() -> tuple[Prescription, list[tuple[Any, Any]]]:
    """The quickstart asked for its planned cost's bias, and the plan and the fit it was read
    from."""
    seen: list[tuple[Any, Any]] = []
    real = decision._fit_cost_bias

    def spy(plan: Any, fit: Any) -> PlannedCostBias:
        seen.append((plan, fit))
        return real(plan, fit)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(decision, "_fit_cost_bias", spy)
        made = _quickstart(planned_cost_bias=True)
    return made, seen


@pytest.fixture(scope="module")
def quickstart(captured: tuple[Prescription, list[tuple[Any, Any]]]) -> Prescription:
    return captured[0]


@pytest.fixture(scope="module")
def two_levers() -> Prescription:
    return _two_levers()


@pytest.fixture(scope="module")
def held() -> Prescription:
    return _held()


SHAPES = ("quickstart", "two_levers", "held")


def _text(record: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(record, allow_nan=False))


@pytest.mark.parametrize("shape", SHAPES)
def test_a_record_replays_to_the_bit(shape: str, request: pytest.FixtureRequest) -> None:
    """The replay's record is the record, through JSON text, and its schedule, its plan's actions
    and its certificate are the original's bit for bit."""
    made: Prescription = request.getfixturevalue(shape)
    record = _text(made.to_json())
    replayed = Prescription.from_json(
        record,
        TWO if shape != "quickstart" else QUICKSTART,
        model=KNOWN if shape == "held" else None,
    )
    assert replayed.to_json() == record
    assert made.plan is not None
    assert replayed.plan is not None
    assert np.array_equal(np.asarray(replayed.plan.actions), np.asarray(made.plan.actions))
    assert replayed.certificate == made.certificate


def test_a_record_that_made_no_plan_replays() -> None:
    latent = prescribe(
        QUICKSTART,
        adjustment=CausalGraph.from_edges(QUICKSTART_EDGES, latent=("demand",)),
        levers=[Lever("incentive", lo=-2.0, hi=2.0)],
        target=Target("supply", value=1.0),
        horizon=15,
        dt=DT,
        planned_cost_bias=True,
    )
    record = _text(latent.to_json())
    assert record["schedule"] is None
    assert record["certificate"]["planned_cost_bias_reason"] == (
        "no plan was made: the effect is not identified"
    )
    assert Prescription.from_json(record, QUICKSTART).to_json() == record


# ---- what the record holds ----


def test_the_record_holds_every_input_and_the_checks_the_plan_was_read_with(
    two_levers: Prescription,
) -> None:
    record = _text(two_levers.to_json())
    inputs = record["inputs"]
    assert tuple(inputs) == decision._INPUT_KEYS
    assert inputs["panel_sha256"] == TWO.provenance.data_sha256
    assert inputs["panel_cluster"] is None
    assert inputs["adjustment"] == {"kind": "asserted", "covariates": ["z"]}
    assert inputs["levers"][0] == {
        "name": "u1",
        "lo": -1.5,
        "hi": 2.0,
        "unit_cost": 0.05,
        "cap_per_step": 0.8,
    }
    assert inputs["target"] == {
        "name": "x",
        "value": np.linspace(0.2, 0.6, 6).tolist(),
        "weight": 2.0,
    }
    assert inputs["neighbours"]["units"][:2] == [[0, [1]], [1, [0]]]
    assert (inputs["x0"], inputs["integrator"], inputs["dither"]) == (
        [0.1, -0.05],
        "rk4",
        ["eta1", "eta2"],
    )
    assert inputs["known"] is None
    dither = record["dither_check"]
    assert dither["dither"] == ["eta1", "eta2"]
    assert dither["status"] in ("tested", "void")
    assert dither["relevance"] == [float(v) for v in two_levers.dither_check.relevance]
    check = two_levers.logger_check
    if check is None:
        assert record["logger_check"] is None
    else:
        assert record["logger_check"]["p_value"] == pytest.approx(check.test.p_value, abs=0.0)
    assert record["certificate"]["interference_p"] is not None
    # the bias is read, or None with the reason, never both
    certificate = record["certificate"]
    assert (certificate["planned_cost_bias"] is None) != (
        certificate["planned_cost_bias_reason"] is None
    )


def test_a_graph_s_edges_go_to_the_record_with_their_provenance(held: Prescription) -> None:
    written = _text(held.to_json())["inputs"]
    edges = {(e["parent"], e["child"]): e for e in written["adjustment"]["edges"]}
    assert edges["u1", "x"] == {
        "parent": "u1",
        "child": "x",
        "source": "discovered",
        "test": "partial_corr_test",
        "p_value": 0.0012,
    }
    assert edges["x", "w"]["source"] == "mechanism"
    assert edges["z", "u1"] == {
        "parent": "z",
        "child": "u1",
        "source": "asserted",
        "test": None,
        "p_value": None,
    }
    assert written["known"] == decision._model_digest(KNOWN)
    assert len(written["known"]) == 64
    assert (written["clusters"], written["max_levers"], written["hold_constraints"]) == (
        "market",
        1,
        True,
    )
    assert written["drivers"] == [{"name": "temp", "forecast": np.sin(np.arange(6) / 2.0).tolist()}]


def test_an_infinite_setting_is_text_and_none_is_not_given() -> None:
    """JSON has no infinity, and None in ``inputs`` means not given: ``tolerance=None`` switches
    the tube off where ``tolerance=inf`` evaluates it, and a lever's open side is ``-inf``."""
    assert [decision._setting(v) for v in (math.inf, -math.inf, None, 2, 0.5)] == [
        "inf",
        "-inf",
        None,
        2,
        0.5,
    ]
    assert [decision._read_setting(v) for v in ("inf", "-inf", None, 2, 0.5)] == [
        math.inf,
        -math.inf,
        None,
        2,
        0.5,
    ]
    lever = Lever("u1", lo=-math.inf, hi=1.0)
    written = decision._inputs_record(
        TWO,
        levers=[lever],
        target=Target("x", value=0.0),
        horizon=3,
        adjustment=["z"],
        constraints=[Constraint("w", lo=-math.inf)],
        hold_constraints=False,
        budgets=(),
        max_levers=None,
        known=None,
        dt=DT,
        gamma=1.0,
        radius="auto",
        tolerance=math.inf,
        x0=None,
        folds=2,
        seed=0,
        integrator=None,
        drivers=(),
        neighbours="market",
        clusters=None,
        nuisance_knots="auto",
        tube_actions="box",
        dither="eta1",
        planned_cost_bias=False,
    )
    text = _text(written)
    assert (text["tolerance"], text["levers"][0]["lo"], text["constraints"][0]) == (
        "inf",
        "-inf",
        {"state": "w", "lo": "-inf", "hi": None},
    )
    read = decision._replay_arguments(text, TWO, None)
    assert read["levers"] == [lever]
    assert read["constraints"] == [Constraint("w", lo=-math.inf)]
    assert (read["tolerance"], read["neighbours"], read["dither"]) == (math.inf, "market", "eta1")


# ---- refusals ----


def test_a_record_of_another_schema_or_without_inputs_is_refused(quickstart: Prescription) -> None:
    record = _text(quickstart.to_json())
    with pytest.raises(DecisionError, match=r"^schema_version: this release replays 2, not 3"):
        Prescription.from_json({**record, "schema_version": 3}, QUICKSTART)
    without = {key: value for key, value in record.items() if key != "inputs"}
    with pytest.raises(DecisionError, match=r"^inputs: the record holds no inputs"):
        Prescription.from_json(without, QUICKSTART)
    with pytest.raises(DecisionError, match="a decision record is a mapping"):
        Prescription.from_json([record], QUICKSTART)  # type: ignore[arg-type]


@pytest.mark.parametrize("key", ["gamma", "seed", "dither", "panel_sha256", "planned_cost_bias"])
def test_a_record_missing_a_setting_is_refused_naming_it(
    quickstart: Prescription, key: str
) -> None:
    record = _text(quickstart.to_json())
    del record["inputs"][key]
    with pytest.raises(DecisionError, match=rf"^inputs\.{key}: the record does not hold it"):
        Prescription.from_json(record, QUICKSTART)


def test_a_record_of_another_release_or_precision_is_refused(quickstart: Prescription) -> None:
    record = _text(quickstart.to_json())
    older = copy.deepcopy(record)
    older["run"]["versions"]["causal-hybrid-control"] = "0.0.1"
    with pytest.raises(DecisionError, match=r"^run\.versions: the record was written by "):
        Prescription.from_json(older, QUICKSTART)
    unnamed = copy.deepcopy(record)
    del unnamed["run"]["versions"]["causal-hybrid-control"]
    with pytest.raises(DecisionError, match=r"^run\.versions: the record does not name"):
        Prescription.from_json(unnamed, QUICKSTART)
    single = copy.deepcopy(record)
    single["run"]["x64"] = False
    with pytest.raises(DecisionError, match=r"^run\.x64: the record was made with"):
        Prescription.from_json(single, QUICKSTART)


def test_a_record_replayed_on_other_data_is_refused(quickstart: Prescription) -> None:
    record = _text(quickstart.to_json())
    with pytest.raises(DecisionError, match=r"^inputs\.panel_sha256: the record was made from"):
        Prescription.from_json(record, TWO)


def test_a_known_model_must_be_passed_again_and_be_the_same(
    held: Prescription, quickstart: Prescription
) -> None:
    record = _text(held.to_json())
    with pytest.raises(DecisionError, match=r"^inputs\.known: .* pass it again as model="):
        Prescription.from_json(record, TWO)
    other = LinearDynamics(jnp.array([[-0.5, 0.0], [0.0, -0.31]]), jnp.zeros((2, 2)))
    with pytest.raises(DecisionError, match=r"^inputs\.known: the model given reads digest"):
        Prescription.from_json(record, TWO, model=other)
    with pytest.raises(DecisionError, match=r"^inputs\.known: the record was made with no known"):
        Prescription.from_json(_text(quickstart.to_json()), QUICKSTART, model=KNOWN)


def test_a_model_s_digest_reads_its_arrays_and_its_class() -> None:
    same = LinearDynamics(jnp.array([[-0.5, 0.0], [0.0, -0.3]]), jnp.zeros((2, 2)))
    assert decision._model_digest(same) == decision._model_digest(KNOWN)
    single = LinearDynamics(KNOWN.a_matrix.astype(jnp.float32), KNOWN.b_matrix)
    assert decision._model_digest(single) != decision._model_digest(KNOWN)
    swapped = LinearDynamics(KNOWN.b_matrix, KNOWN.a_matrix)
    assert decision._model_digest(swapped) != decision._model_digest(KNOWN)


def test_a_replay_that_does_not_give_the_record_back_names_the_field(
    quickstart: Prescription,
) -> None:
    record = _text(quickstart.to_json())
    record["certificate"]["channel_error"] *= 1.0 + 2.0**-40
    with pytest.raises(
        DecisionError, match=r"^certificate\.channel_error: the replay does not give the record"
    ):
        Prescription.from_json(record, QUICKSTART)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("gamma",), 1.7),
        (("seed",), 6),
        (("folds",), 2),
        (("nuisance_knots",), 5),
        (("x0",), [0.15, -0.05]),
        (("target", "weight"), 3.0),
        (("levers", 0, "cap_per_step"), 0.5),
        (("budgets", 0, "amount"), 2.5),
    ],
)
def test_each_setting_moves_the_replay(
    two_levers: Prescription, path: tuple[Any, ...], value: Any
) -> None:
    """A record whose setting is changed, and nothing else, does not replay: the setting reaches
    the replay, and moves what it gives."""
    record = _text(two_levers.to_json())
    holder: Any = record["inputs"]
    for step in path[:-1]:
        holder = holder[step]
    assert holder[path[-1]] != value
    holder[path[-1]] = value
    with pytest.raises(DecisionError, match="the replay does not give the record back"):
        Prescription.from_json(record, TWO)


# ---- the planned cost's bias in the certificate ----


def test_the_certificate_s_bias_is_the_fit_s_at_the_quickstart(
    captured: tuple[Prescription, list[tuple[Any, Any]]],
) -> None:
    made, seen = captured
    ((plan, fit),) = seen
    assert plan is made.plan
    assert fit.influence is not None
    # the refit keeping its influence holds the plan's own parameters
    assert np.array_equal(
        np.asarray(fit.residual.channel), np.asarray(made.model_fit.residual.channel)
    )
    assert np.array_equal(np.asarray(fit.residual.drift), np.asarray(made.model_fit.residual.drift))
    bias = made.certificate.planned_cost_bias
    assert bias == planned_cost_bias(plan, fit)
    assert made.certificate.planned_cost_bias_reason is None
    assert bias is not None
    assert bias.planned_cost == made.plan.task_cost
    written = made.to_json()["certificate"]["planned_cost_bias"]
    assert written == {
        "planned_cost": bias.planned_cost,
        "optimism": bias.optimism,
        "curvature": bias.curvature,
        "first_order": 0.0,
        "bias": bias.bias,
        "direction": bias.direction,
        "corrected_cost": bias.corrected_cost,
        "unseen": bias.unseen,
        "free": bias.free,
        "residual": bias.residual,
    }
    assert "planned cost's bias, to second order" in made.report()


def test_the_bias_moves_nothing_else_and_is_off_by_default(quickstart: Prescription) -> None:
    plain = _quickstart()
    assert plain.certificate.planned_cost_bias is None
    assert plain.certificate.planned_cost_bias_reason == decision._BIAS_NOT_ASKED
    assert "planned cost's bias" not in plain.report()
    asked, default = quickstart.to_json(), plain.to_json()
    for record in (asked, default):
        del record["certificate"]["planned_cost_bias"]
        del record["certificate"]["planned_cost_bias_reason"]
        del record["inputs"]["planned_cost_bias"]
    assert asked == default
    assert (
        quickstart.report().replace(quickstart._cost_bias_lines()[0] + "\n", "") == plain.report()
    )


def test_a_bias_the_fit_cannot_read_is_none_with_the_refusal(held: Prescription) -> None:
    """The held plan is made against a driver's forecast, whose gain's error the bias does not
    read."""
    certificate = held.certificate
    assert certificate.planned_cost_bias is None
    assert certificate.planned_cost_bias_reason is not None
    assert "drivers" in certificate.planned_cost_bias_reason
    assert "planned cost's bias: not read, " in held.report()


def _scalar_logs(units: int = 400, seed: int = 11) -> Panel:
    """One transition a unit of ``xdot = -0.5 x + 1.2 u + z``, the lever set from ``z``."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0.0, 1.0, units)
    z = rng.normal(0.0, 1.0, units)
    u = 0.5 * z + rng.normal(0.0, 1.0, units)
    x1 = x0 + DT * (-0.5 * x0 + 1.2 * u + z) + rng.normal(0.0, 0.05, units)
    frame = {
        "unit": np.repeat(np.arange(units), 2),
        "time": np.tile([0, 1], units),
        "x": np.column_stack([x0, x1]).ravel(),
        "u": np.column_stack([u, np.zeros(units)]).ravel(),
        "z": np.column_stack([z, np.zeros(units)]).ravel(),
    }
    return Panel.from_frame(frame, unit="unit", time="time", seed=0)


def test_on_one_scalar_step_the_bias_is_maxima_s_closed_form() -> None:
    """``validation/planned_cost_bias.mac``: one step ``x1 = p + g u`` at the cost
    ``qf (x1 - target)^2 + r u^2`` reads ``y^2 qf^2 (2 r^2 - 5 g^2 qf r + g^4 qf^2) / (2 (r + g^2
    qf)^3)`` per unit of ``S``, the error variance of ``g``, ``y = p - target``. With the drift's
    rows of the fit's influence set to zero, the certificate's bias is that, at the fitted ``p``
    and ``g`` and ``S`` the fit's covariance of ``g``."""
    seen: list[tuple[Any, Any]] = []
    real = decision._fit_cost_bias

    def spy(plan: Any, fit: Any) -> PlannedCostBias:
        seen.append((plan, fit))
        return real(plan, fit)

    q, r, target, start = 2.0, 0.1, 0.3, 1.0
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(decision, "_fit_cost_bias", spy)
        made = prescribe(
            _scalar_logs(),
            levers=[Lever("u", lo=-50.0, hi=50.0, unit_cost=r)],
            target=Target("x", value=target, weight=q),
            adjustment=CausalGraph.from_edges([("z", "u"), ("z", "x"), ("u", "x")]),
            horizon=1,
            dt=DT,
            x0=[start],
            planned_cost_bias=True,
        )
    ((plan, fit),) = seen
    assert made.certificate.planned_cost_bias == planned_cost_bias(plan, fit)
    channel = int(np.asarray(fit.residual.channel).size)
    influence = np.asarray(fit.influence).copy()
    influence[:, :, channel:] = 0.0
    alone = dataclasses.replace(fit, influence=jnp.asarray(influence))
    found = planned_cost_bias(plan, alone)

    residual = fit.residual
    x = jnp.array([start])

    def one_step(theta: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        moved = ControlAffineResidual(
            drift=residual.drift,
            channel=theta.reshape(residual.channel.shape),
            degree=residual.degree,
            channel_degree=residual.channel_degree,
        )
        return x + DT * moved(0.0, x, u)

    theta = jnp.asarray(residual.channel).ravel()
    free = float(one_step(theta, jnp.zeros(1))[0])
    gain = float(jax.jacfwd(one_step, argnums=1)(theta, jnp.zeros(1))[0, 0])
    lift = np.asarray(
        jax.jacfwd(lambda t: jax.jacfwd(one_step, argnums=1)(t, jnp.zeros(1))[0, 0])(theta)
    )
    (covariance,) = _clustered_squares(influence, fit.clusters, fit.periods)
    variance = float(lift @ covariance[:channel, :channel] @ lift)
    y = free - target
    coefficient = (
        y**2
        * q**2
        * (2 * r**2 - 5 * gain**2 * q * r + gain**4 * q**2)
        / (2 * (r + gain**2 * q) ** 3)
    )
    # the solver's optimum is within its stationarity tolerance of the closed form's
    assert found.bias == pytest.approx(coefficient * variance, rel=1e-6, abs=0.0)
    assert (
        found.bias != made.certificate.planned_cost_bias.bias
    )  # the drift's error is the plan's too


# ---- the evaluation's record ----


def _market(zones: int = 400, periods: int = 12, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    supply = np.empty((zones, periods))
    supply[:, 0] = rng.normal(0.0, 1.3, zones)
    incentive = rng.normal(0.0, 1.0, (zones, periods))
    shock = rng.normal(0.0, 1.0, (zones, periods))
    for t in range(periods - 1):
        supply[:, t + 1] = 0.5 * supply[:, t] + 0.5 * incentive[:, t] + shock[:, t]
    zone, time = np.nonzero(np.ones((zones, periods), dtype=bool))
    frame = {
        "zone": zone,
        "time": time,
        "supply": supply[zone, time],
        "incentive": incentive[zone, time],
    }
    return Panel.from_frame(frame, unit="zone", time="time", seed=0)


def test_a_prescription_s_evaluation_goes_to_json() -> None:
    panel = _market()
    made = prescribe(
        panel,
        levers=[Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.1)],
        target=Target("supply", value=0.5),
        adjustment=CausalGraph.from_edges([("incentive", "supply")]),
        horizon=3,
        dt=DT,
        tolerance=0.5,
        integrator="rk4",
    )
    evaluation = made.evaluate(panel, n_resamples=20, seed=3, model_error=0.5)
    record = evaluation.to_json()
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert (record["schema_version"], record["method"], record["level"]) == (
        EVALUATION_SCHEMA_VERSION,
        "pdis",
        0.95,
    )
    assert record["value"] == evaluation.value
    assert record["interval"] == list(evaluation.interval)
    assert record["versus_logger"]["interval"] == list(evaluation.versus_logger.interval)
    assert record["bootstrap"]["resamples"] == 20
    assert record["certificate"]["samples"] == evaluation.certificate.samples
    assert record["settings"] == {
        "entry": "Prescription.evaluate",
        "method": "pdis",
        "smoothing": None,
        "model_error": 0.5,
        "min_effective": 100.0,
        "logger": None,
        "time_zero": "calendar",
        "n_resamples": 20,
        "seed": 3,
        "max_refused": 0.05,
        "panel_sha256": panel.provenance.data_sha256,
    }
    check = evaluation.logger_check
    assert (record["logger_check"] is None) == (check is None)


def test_an_evaluation_of_logs_goes_to_json_with_the_logger_it_was_given() -> None:
    rng = np.random.default_rng(4)
    x, u = np.zeros((2001, 1)), rng.normal(0.0, 1.0, (2000, 1))
    for t in range(2000):
        x[t + 1] = 0.8 * x[t] + 0.5 * u[t] + rng.normal(0.0, 0.2)
    plant = LinearGaussianPlant(
        np.array([[0.8]]), np.array([[0.5]]), np.zeros(1), np.array([[0.04]])
    )
    cost = QuadraticCost(Q=jnp.eye(1), R=jnp.array([[0.1]]), Qf=jnp.eye(1), x_target=jnp.zeros(1))
    logger = AffinePolicy(np.zeros((1, 1)), np.zeros(1), np.eye(1))
    plan = AffinePolicy(np.array([[-0.4]]), np.zeros(1), np.array([[0.25]]))
    evaluation = evaluate_plan({"x": x, "u": u}, plan, "mis", plant=plant, cost=cost, logger=logger)
    record = evaluation.to_json()
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert record["settings"] == {
        "entry": "evaluate_plan",
        "method": "mis",
        "smoothing": None,
        "model_error": 1.0,
        "min_effective": 100.0,
        "logger": {"gain": [[0.0]], "offset": [0.0], "covariance": [[1.0]]},
        "logs_sha256": decision_evaluation._logs_sha256({"x": x, "u": u}),
    }
    assert record["degrees_of_freedom"] == evaluation.degrees_of_freedom
    assert record["bootstrap"] is None
    assert record["versus_logger"] is None
    # the settings are a record, not part of the evaluation's equality
    assert dataclasses.replace(evaluation, _settings=None) == evaluation


def test_an_evaluation_names_the_logs_it_was_read_from_and_takes_array_keywords() -> None:
    """The logs' digest moves with any of their numbers and their shape, and a keyword passed as a
    0-d array is written as the float it holds: up to 0.24.0 ``json.dumps`` refused the record."""
    rng = np.random.default_rng(4)
    x, u = np.zeros((2001, 1)), rng.normal(0.0, 1.0, (2000, 1))
    for t in range(2000):
        x[t + 1] = 0.8 * x[t] + 0.5 * u[t] + rng.normal(0.0, 0.2)
    plant = LinearGaussianPlant(
        np.array([[0.8]]), np.array([[0.5]]), np.zeros(1), np.array([[0.04]])
    )
    cost = QuadraticCost(Q=jnp.eye(1), R=jnp.array([[0.1]]), Qf=jnp.eye(1), x_target=jnp.zeros(1))
    plan = AffinePolicy(np.array([[-0.4]]), np.zeros(1), np.array([[0.25]]))
    evaluation = evaluate_plan(
        {"x": x, "u": u},
        plan,
        "mis",
        plant=plant,
        cost=cost,
        model_error=np.array(0.5),  # type: ignore[arg-type]
        min_effective=np.float64(100.0),  # type: ignore[arg-type]
    )
    record = evaluation.to_json()
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert type(record["settings"]["model_error"]) is float
    assert record["settings"]["model_error"] == 0.5
    digest = decision_evaluation._logs_sha256
    assert record["settings"]["logs_sha256"] == digest({"x": x, "u": u})
    assert digest({"x": x.astype(np.float32).astype(np.float64), "u": u}) != digest(
        {"x": x, "u": u}
    )
    assert digest({"x": x, "u": u.reshape(1, -1)}) != digest({"x": x, "u": u})
    assert digest({"x": x.tolist(), "u": u, "y": 1.0}) == digest({"x": x, "u": u})
    assert decision_evaluation._json_safe({"a": np.array(2.0), "b": np.array(3)}) == {
        "a": 2.0,
        "b": 3,
    }


# ---- what a replay reads, and how it refuses ----


def _iterators(**options: Any) -> Prescription:
    """``_two_levers`` with every collection handed in as a one-shot iterator."""
    return _two_levers(
        levers=iter(
            [
                Lever("u1", lo=-1.5, hi=2.0, unit_cost=0.05, cap_per_step=0.8),
                Lever("u2", lo=-1.0, hi=1.0, unit_cost=0.1),
            ]
        ),
        constraints=iter([Constraint("w", lo=-0.6)]),
        budgets=iter([chc_budget()]),
        adjustment=iter(["z"]),
        neighbours={unit: iter([unit ^ 1]) for unit in range(40)},
        dither=iter(["eta1", "eta2"]),
        **options,
    )


def test_collections_handed_in_as_iterators_decide_and_replay_as_lists(
    two_levers: Prescription,
) -> None:
    """``neighbours`` is a mapping of iterables, and the others are read more than once: up to
    0.24.0 the record spent the neighbours' iterators, the interference check read no neighbour
    and reported not tested, and the record did not replay."""
    made = _iterators()
    assert made.interference is not None
    assert two_levers.interference is not None
    assert made.interference.p_value == two_levers.interference.p_value
    record = _text(made.to_json())
    assert record == _text(two_levers.to_json())
    assert Prescription.from_json(record, TWO).to_json() == record


def test_a_panel_s_declared_cluster_is_part_of_the_record() -> None:
    """The declared cluster groups the channel's error and is not in the data's digest: a replay
    on the same data without it is refused naming it, where it once read as a numerical
    difference in ``certificate.channel_error``."""
    logs = _quickstart_logs(regions=60, weeks=8)
    logs["zone"] = np.asarray([f"z{region // 5}" for region in logs["region"]], dtype=object)
    declared = Panel.from_frame(logs, unit="region", time="week", cluster="zone", seed=0)
    plain = Panel.from_frame(logs, unit="region", time="week", seed=0)
    assert declared.provenance.data_sha256 == plain.provenance.data_sha256
    made = prescribe(
        declared,
        adjustment=CausalGraph.from_edges(QUICKSTART_EDGES),
        levers=[Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05)],
        target=Target("supply", value=1.0),
        horizon=4,
        dt=DT,
        tolerance=0.5,
        integrator="euler",
    )
    record = _text(made.to_json())
    assert record["inputs"]["panel_cluster"] == "zone"
    assert Prescription.from_json(record, declared).to_json() == record
    with pytest.raises(
        DecisionError, match=r"^inputs\.panel_cluster: .* declared cluster 'zone', and this panel"
    ):
        Prescription.from_json(record, plain)


def _tampered(record: dict[str, Any], path: tuple[Any, ...], value: Any) -> dict[str, Any]:
    changed = copy.deepcopy(record)
    holder: Any = changed["inputs"]
    for step in path[:-1]:
        holder = holder[step]
    if value is _APPEND:
        cycle = {"parent": "supply", "child": "demand", "source": "asserted", "test": None}
        holder[path[-1]].append({**cycle, "p_value": None})
    else:
        holder[path[-1]] = value
    return changed


_APPEND = object()


@pytest.mark.parametrize(
    ("path", "value", "named"),
    [
        (("horizon",), "four", "inputs.horizon"),
        (("folds",), 2.5, "inputs.folds"),
        (("adjustment", "edges"), _APPEND, "inputs.adjustment"),
        (("adjustment", "edges", 0, "source"), "guessed", "inputs.adjustment"),
        (("levers", 0, "lo"), 5.0, "inputs.levers"),
        (("target", "weight"), -1.0, "inputs.target"),
        (("gamma",), 0.5, "inputs.gamma"),
        (("radius",), "wide", "inputs.radius"),
        (("tube_actions",), "vertex", "inputs.tube_actions"),
        (("levers", 0, "name"), "bonus", "inputs"),
    ],
    ids=[
        "horizon text",
        "folds fraction",
        "a cycle",
        "an edge's source",
        "a lever's box",
        "a negative weight",
        "gamma below 1",
        "a radius",
        "a tube form",
        "a column not in the panel",
    ],
)
def test_a_record_prescribe_refuses_is_refused_naming_its_input(
    quickstart: Prescription, path: tuple[Any, ...], value: Any, named: str
) -> None:
    """Each refusal is a ``DecisionError`` that opens with the ``inputs`` key it reads, or with
    ``inputs`` where the refusal names none; up to 0.24.0 four of them were a bare ``ValueError``,
    a ``CyclicGraphError`` or prescribe's message with no path, and a negative weight replayed to
    a schedule the record does not hold."""
    record = _tampered(_text(quickstart.to_json()), path, value)
    with pytest.raises(DecisionError, match=rf"^{named.replace('.', chr(92) + '.')}: "):
        Prescription.from_json(record, QUICKSTART)


# ---- the graph's edges ----


def test_an_edge_s_provenance_is_not_part_of_the_graph_s_equality() -> None:
    edges = [("z", "u1"), ("u1", "x"), ("x", "w")]
    bare = CausalGraph.from_edges(edges)
    sourced = CausalGraph.from_edges(edges, provenance=PROVENANCE)
    assert bare == sourced
    assert hash(bare) == hash(sourced)
    assert sourced.edge_provenance("u1", "x") == PROVENANCE["u1", "x"]
    assert sourced.edge_provenance("z", "u1") == EdgeProvenance("asserted")
    assert bare.edge_provenance("x", "w") == EdgeProvenance("asserted")
    with pytest.raises(KeyError, match="is not an edge"):
        bare.edge_provenance("w", "x")


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: EdgeProvenance("guessed"), "asserted', 'discovered' or 'mechanism'"),  # type: ignore[arg-type]
        (lambda: EdgeProvenance("discovered"), "names the test"),
        (lambda: EdgeProvenance("discovered", "gcm", 1.5), r"lies in \[0, 1\]"),
        (lambda: EdgeProvenance("asserted", "gcm"), "names no test"),
        (
            lambda: CausalGraph.from_edges(
                [("a", "b")], provenance={("b", "a"): EdgeProvenance("mechanism")}
            ),
            "not edges of the graph",
        ),
        (
            lambda: CausalGraph.from_edges([("a", "b")], provenance={("a", "b"): "mechanism"}),  # type: ignore[dict-item]
            "is an EdgeProvenance",
        ),
    ],
)
def test_an_edge_s_provenance_is_checked(make: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make()
