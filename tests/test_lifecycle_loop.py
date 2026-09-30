"""The lifecycle as one program written from the top level alone: a panel, a prescription, and its
evaluation from a later panel before anything is deployed.

The loop stops there. The deployment gate after it assumes that a decision's reward does not
depend on earlier decisions, and a plan over a horizon, on a plant with memory, breaks that; no
stage here hands the gate its batches.

The market is linear and Gaussian, and its incentive is randomised and chases nothing, so every
number the evaluation reports is inside :mod:`chc.evaluation`'s scope. Demand moves supply and
nothing else: the graph's adjustment set holds it for precision, and the logger never read it.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

import chc
from chc.decision import _Columns, _linearised

DT = 0.1
STEP = np.array([[0.94, 0.03], [0.0, 1.025]])  # (supply, wait), one period, Euler at DT
CHANNEL = np.array([0.08, -0.04])  # the incentive's push on (supply, wait)
DEMAND = np.array([0.15, 0.0])
SHOCK = 0.01
EDGES = [("demand", "supply"), ("incentive", "supply"), ("incentive", "wait"), ("supply", "wait")]


def _market(
    units: int, seed: int, chase: float = 0.0, shared: float = 0.0, stale: float = 0.0
) -> dict[str, np.ndarray]:
    """Twelve periods of each unit, the incentive drawn afresh every period, sd 1.5, plus
    ``chase`` times the period's demand and ``stale`` times the last period's supply, as a budget
    set on last week's report is. A ``shared`` share of the draw's variance is drawn once a period
    for every unit, as a budget shock is."""
    rng = np.random.default_rng(seed)
    common = np.random.default_rng([seed, 1]).normal(0.0, 1.5, 12)
    names = ("unit", "time", "supply", "wait", "incentive", "demand")
    rows: dict[str, list[float]] = {name: [] for name in names}
    for unit in range(units):
        x = rng.normal(0.0, 0.2, 2)
        before = x
        for period in range(12):
            demand, incentive = rng.normal(), rng.normal(0.0, 1.5)
            incentive = np.sqrt(1 - shared) * incentive + np.sqrt(shared) * common[period]
            incentive += chase * demand + stale * before[0]
            for name, value in zip(names, (unit, period, *x, incentive, demand), strict=True):
                rows[name].append(value)
            before = x
            x = STEP @ x + CHANNEL * incentive + DEMAND * demand + rng.normal(0.0, SHOCK, 2)
    return {name: np.asarray(values) for name, values in rows.items()}


def _value(actions: np.ndarray, starts: np.ndarray, cost: chc.QuadraticCost) -> float:
    """The open-loop schedule's expected running cost on the true market, by covariance
    propagation from the law of ``starts``."""
    q, r = np.asarray(cost.Q), np.asarray(cost.R)
    targets = np.asarray(cost.targets(len(actions)))
    noise = np.outer(DEMAND, DEMAND) + SHOCK**2 * np.eye(2)
    mean, covariance, total = starts.mean(axis=0), np.cov(starts, rowvar=False), 0.0
    for t, u in enumerate(actions):
        gap = mean - targets[t]
        total += 0.5 * float(np.trace(q @ covariance) + gap @ q @ gap + u @ r @ u)
        mean = STEP @ mean + CHANNEL * u[0]
        covariance = STEP @ covariance @ STEP.T + noise
    return total


def _prescription() -> chc.Prescription:
    panel = chc.Panel.from_frame(_market(400, seed=0), unit="unit", time="time", seed=0)
    return chc.prescribe(
        panel,
        levers=[chc.Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05)],
        target=chc.Target("supply", value=0.3),
        constraints=[chc.Constraint("wait", hi=0.5)],
        adjustment=chc.CausalGraph.from_edges(EDGES),
        horizon=3,
        dt=DT,
        tolerance=0.5,
    )


def test_a_prescription_is_evaluated_from_a_later_panel_before_it_is_deployed() -> None:
    prescription = _prescription()
    later = _market(400, seed=1)
    evaluation = prescription.evaluate(
        chc.Panel.from_frame(later, unit="unit", time="time", seed=0)
    )

    # Each unit's windows of four periods start at periods 8, 5 and 2.
    columns = np.stack([later["supply"], later["wait"]], axis=1).reshape(400, 12, 2)
    incentive = later["incentive"].reshape(400, 12, 1)
    logs = {
        "x": np.concatenate([columns[:, s : s + 4] for s in (8, 5, 2)]),
        "u": np.concatenate([incentive[:, s : s + 3] for s in (8, 5, 2)]),
    }
    assert prescription.plan is not None
    problem = prescription.plan._problem
    assert problem is not None
    actions = np.asarray(prescription.plan.actions)
    truth = _value(actions, logs["x"][:, 0], problem.cost)
    assert prescription.certificate.adjustment.covariates == ("demand",)
    assert prescription._columns == _Columns(
        states=("supply", "wait"), logged_on=(), unlogged=("demand",)
    )
    assert evaluation.certificate.certified
    assert evaluation.certificate.certified_horizon == 3
    assert evaluation.interval[0] <= truth <= evaluation.interval[1]
    # What it hands evaluate_plan: these episodes, the plan's actions open loop, its cost.
    direct = chc.evaluate_plan(
        logs,
        chc.AffineSchedule.open_loop(actions, 2),
        "pdis",
        plant=_linearised(problem.model, logs["x"], logs["u"], problem.dt),
        cost=problem.cost,
    )
    assert evaluation.value == pytest.approx(direct.value, rel=1e-9)


def test_the_evaluation_warns_when_a_later_logger_read_what_the_graph_says_it_did_not(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The graph says the incentive chases nothing. A later panel whose logger chased demand, which
    moves supply, or read last period's supply, has another propensity than the weights read, and
    the evaluation says so. A budget shock every unit drew at once is noise, not a column read, and
    passes."""
    prescription = _prescription()

    def evaluate(later: dict[str, np.ndarray]) -> chc.evaluation.PlanEvaluation:
        return prescription.evaluate(chc.Panel.from_frame(later, unit="unit", time="time", seed=0))

    with caplog.at_level(logging.INFO, logger="chc.decision"):
        honest = evaluate(_market(400, seed=1))
        chased = evaluate(_market(400, seed=1, chase=1.0))
        shocked = evaluate(_market(400, seed=2, shared=0.5))
        stale = evaluate(_market(400, seed=1, stale=1.0))
    checks = [
        record for record in caplog.records if getattr(record, "chc_event", "") == "logger_check"
    ]
    assert honest.logger_check is not None
    assert chased.logger_check is not None
    assert shocked.logger_check is not None
    assert stale.logger_check is not None
    assert honest.logger_check.columns[0] == "demand"
    assert honest.logger_check.test.p_value > 0.05
    assert chased.logger_check.test.p_value <= 0.05
    assert shocked.logger_check.test.p_value > 0.05
    assert stale.logger_check.test.p_value <= 0.05
    strongest = np.nanargmax(np.abs(stale.logger_check.test.partial_correlation[0]))
    assert stale.logger_check.columns[strongest] == "supply[t-1]"
    levels = [logging.INFO, logging.WARNING, logging.INFO, logging.WARNING]
    assert [record.levelno for record in checks] == levels
