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


def _market(units: int, seed: int) -> dict[str, np.ndarray]:
    """Twelve periods of each unit, the incentive drawn afresh every period, sd 1.5."""
    rng = np.random.default_rng(seed)
    names = ("unit", "time", "supply", "wait", "incentive", "demand")
    rows: dict[str, list[float]] = {name: [] for name in names}
    for unit in range(units):
        x = rng.normal(0.0, 0.2, 2)
        for period in range(12):
            demand, incentive = rng.normal(), rng.normal(0.0, 1.5)
            for name, value in zip(names, (unit, period, *x, incentive, demand), strict=True):
                rows[name].append(value)
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


def test_a_prescription_is_evaluated_from_a_later_panel_before_it_is_deployed() -> None:
    panel = chc.Panel.from_frame(_market(400, seed=0), unit="unit", time="time", seed=0)
    prescription = chc.prescribe(
        panel,
        levers=[chc.Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05)],
        target=chc.Target("supply", value=0.3),
        constraints=[chc.Constraint("wait", hi=0.5)],
        adjustment=chc.CausalGraph.from_edges(EDGES),
        horizon=3,
        dt=DT,
        tolerance=0.5,
    )
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
    assert prescription._columns == _Columns(states=("supply", "wait"), logged_on=())
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
