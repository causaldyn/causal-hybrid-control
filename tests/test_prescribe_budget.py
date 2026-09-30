"""A spend budget for ``prescribe``: the plan keeps to it, and says what a unit more would buy.

The plant is :mod:`chc.mmm`'s, three channels of spend on a saturating carryover, planned from a
confounded log with the season adjusted for. Unbudgeted, the plan spends where the sales target
pays for it; the budgets here allow 60% of that, so every row binds and every price is a real one.
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

from chc.decision import Constraint, DecisionError, Prescription, Target, prescribe
from chc.mmm import SALES, MarketingMixSystem, adstock_dynamics
from chc.mpc import PeriodBudget
from chc.panel import Panel

SYSTEM = MarketingMixSystem()
HORIZON = 12
SHARE = 0.6  # of the unbudgeted plan's spend


def _prescribe(panel: Panel, budgets: tuple[PeriodBudget, ...] = ()) -> Prescription:
    return prescribe(
        panel,
        levers=SYSTEM.levers(0.08),
        target=Target(SALES, value=8.0),
        constraints=[Constraint(name, lo=0.0) for name in SYSTEM.adstock_columns],
        adjustment=SYSTEM.graph(),
        known=adstock_dynamics(jnp.array(SYSTEM.theta)),
        horizon=HORIZON,
        dt=1.0,
        tolerance=1.0,
        budgets=budgets,
    )


def _spend(prescription: Prescription) -> np.ndarray:
    """What each step of the plan spends."""
    assert prescription.plan is not None
    return np.asarray(prescription.plan.actions).sum(axis=1)


@pytest.fixture(scope="module")
def panel() -> Panel:
    logs = SYSTEM.sample(n_regions=20, n_weeks=20, seed=0)
    return Panel.from_frame(logs, unit="region", time="week", seed=0)


@pytest.fixture(scope="module")
def budget(panel: Panel) -> float:
    return SHARE * float(_spend(_prescribe(panel)).sum())


@pytest.fixture(scope="module")
def held(panel: Panel, budget: float) -> Prescription:
    return _prescribe(panel, (PeriodBudget(np.ones(3), budget, HORIZON),))


def _cost(panel: Panel, budget: float) -> float:
    plan = _prescribe(panel, (PeriodBudget(np.ones(3), budget, HORIZON),)).plan
    assert plan is not None
    assert plan.solver_status == "converged"
    return plan.task_cost


def test_a_budget_for_the_whole_horizon_binds_and_is_priced_at_the_cost_s_slope(
    panel: Panel, budget: float, held: Prescription
) -> None:
    assert held.plan is not None
    assert held.plan.solver_status == "converged"
    assert _spend(held).sum() == pytest.approx(budget, rel=1e-9)
    [[price]] = held.budget_prices()
    assert price.status == "exact"
    assert price.price is not None
    assert price.price > 0.0
    step = 0.05  # a central difference, since the price itself falls as the budget grows
    slope = (_cost(panel, budget - step) - _cost(panel, budget + step)) / (2 * step)
    assert price.price == pytest.approx(slope, rel=1e-3)


@pytest.mark.parametrize(
    ("period", "fraction", "shares"), [(4, 1 / 3, (1.0, 1.0, 1.0)), (8, 1 / 2, (1.0, 0.5))]
)
def test_each_period_keeps_to_its_budget_and_one_the_horizon_cuts_short_to_its_share(
    panel: Panel,
    budget: float,
    held: Prescription,
    period: int,
    fraction: float,
    shares: tuple[float, ...],
) -> None:
    amount = fraction * budget
    prescription = _prescribe(panel, (PeriodBudget(np.ones(3), amount, period),))
    spend = _spend(prescription)
    spent = [spend[first : first + period].sum() for first in range(0, HORIZON, period)]
    np.testing.assert_allclose(spent, amount * np.asarray(shares), rtol=1e-9)
    [prices] = prescription.budget_prices()
    assert [price.status for price in prices] == ["exact"] * len(shares)
    assert all(price.price is not None and price.price > 0.0 for price in prices)
    # the periods' caps sum to at most the whole-horizon budget, so they cannot plan cheaper
    assert prescription.plan is not None
    assert held.plan is not None
    assert prescription.plan.task_cost >= held.plan.task_cost


def test_the_report_and_the_record_carry_the_budget(budget: float, held: Prescription) -> None:
    assert held.plan is not None
    [[price]] = held.budget_prices()
    assert price.price is not None
    line = (
        f"Budget of {budget:.6g} per {HORIZON} steps: a unit more in each period would lower the "
        f"planned cost by {price.price:.4g}."
    )
    assert line in held.report()
    record = json.loads(json.dumps(held.to_json()))
    assert record["budgets"] == [
        {"weights": [1.0, 1.0, 1.0], "amount": budget, "period": HORIZON, "start": 0.0}
    ]


@pytest.mark.parametrize(
    ("budget", "message"),
    [
        (PeriodBudget(np.ones(2), 10.0, HORIZON), "weighs 2 levers and the decision has 3"),
        (PeriodBudget(np.ones(3), 10.0, HORIZON, start=-2.0), "start at t = -2"),
        # every channel's floor spends 0.05 a week, so 12 weeks spend at least 1.8
        (PeriodBudget(np.ones(3), 1.5, HORIZON), "less than the 1.8 the levers' boxes spend"),
    ],
)
def test_a_budget_the_decision_cannot_keep_is_refused(
    panel: Panel, budget: PeriodBudget, message: str
) -> None:
    with pytest.raises(DecisionError, match=message):
        _prescribe(panel, (budget,))
