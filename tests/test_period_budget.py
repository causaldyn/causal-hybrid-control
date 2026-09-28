"""A budget per period on a receding horizon: never overspent, and allocated as a whole plan would.

The baseline is the budget row a caller could already pass in ``constraints``. It caps every window,
and a loop that replans each step spends it again at every step: on the ledger lab's plant, 2.0 and
8.3 times the day's budget. The plant is that lab's, ``x' = -0.5 x + u`` tracked to 2, and every
spend is audited on the applied actions, not the plans.
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np
import pytest

from chc import LinearConstraint, QuadraticCost, RecedingHorizon, causal_plan
from chc.cost import total_cost
from chc.dynamics import LinearDynamics
from chc.integrate import rk4_step
from chc.mpc import PeriodBudget

DT = 0.5
MODEL = LinearDynamics(jnp.array([[-0.5]]), jnp.array([[1.0]]))
COST = QuadraticCost(Q=jnp.eye(1), R=jnp.array([[0.05]]), Qf=jnp.eye(1), x_target=jnp.array([2.0]))
X0 = jnp.array([1.0])
U_HI = 50.0
DAY = 12


def _free_spend(steps: int) -> float:
    """What the plan over ``steps`` spends with no budget at all."""
    return float(jnp.sum(causal_plan(MODEL, X0, COST, DT, steps, 0.0, U_HI).actions))


def _loop(controller: RecedingHorizon, steps: int, period: int | None) -> np.ndarray:
    """The applied actions; with a ``period``, each step reports that period's spend so far."""
    x, applied = X0, []
    for k in range(steps):
        spent = None if period is None else sum(applied[(k // period) * period :])
        plan = controller.step(x, t=k * DT, spent=spent)
        applied.append(float(plan.actions[0, 0]))
        x = rk4_step(MODEL, k * DT, x, plan.actions[0], DT)
    return np.asarray(applied)


def _cost(actions: np.ndarray) -> float:
    return float(total_cost(MODEL, X0, jnp.asarray(actions).reshape(-1, 1), DT, COST))


@pytest.mark.parametrize(("share", "overspend"), [(0.5, 2.006797), (0.1, 8.286766)])
def test_a_budget_row_in_constraints_caps_each_window_and_the_day_overspends(
    share: float, overspend: float
) -> None:
    """Pinned as found, so that the ledger's ``<= B`` below is a change and not a coincidence.

    The ledger lab's configuration: a 48-step day, 8-step windows. Its numbers were 2.006797 and
    8.286766 in CHC's solver and 8.286841 in Octave's ``qp``.
    """
    day, window = 48, 8
    budget = share * _free_spend(day)
    row = LinearConstraint(np.ones((1, window)), -np.inf, budget)
    controller = RecedingHorizon(MODEL, COST, DT, window, 0.0, U_HI, constraints=(row,))
    assert _loop(controller, day, None).sum() / budget == pytest.approx(overspend, abs=1e-3)


@pytest.mark.parametrize(("share", "worst"), [(0.5, 0.01), (0.1, 0.02)])
def test_a_period_budget_is_spent_exactly_every_day_and_close_to_the_three_day_plan(
    share: float, worst: float, caplog: pytest.LogCaptureFixture
) -> None:
    """Three days against the plan made for all three at once, one budget row per day.

    Measured: 0.62 % and 1.38 % of the budget's value (what the oracle gains over spending
    nothing) with a window of one day; 2.0 % and 4.9 % with half a day, which gets a pro-rata
    share of what is left; the window row above overspends 1.8-5.7 times on the same days.
    """
    days = 3
    budget = share * _free_spend(DAY)
    rows = np.kron(np.eye(days), np.ones((1, DAY)))
    oracle = causal_plan(
        MODEL,
        X0,
        COST,
        DT,
        days * DAY,
        0.0,
        U_HI,
        constraints=(LinearConstraint(rows, -np.inf, np.full(days, budget)),),
    )
    best, idle = _cost(np.asarray(oracle.actions).ravel()), _cost(np.zeros(days * DAY))

    def regret(window: int) -> float:
        controller = RecedingHorizon(
            MODEL, COST, DT, window, 0.0, U_HI, budget=PeriodBudget(np.ones(1), budget, DAY)
        )
        applied = _loop(controller, days * DAY, DAY)
        spend = applied.reshape(days, DAY).sum(axis=1)
        assert np.all(spend <= budget * (1.0 + 1e-9))
        assert np.all(spend >= budget * (1.0 - 1e-6))  # the budget binds, and all of it is spent
        return (_cost(applied) - best) / (idle - best)

    with caplog.at_level(logging.WARNING, logger="chc.mpc"):
        whole = regret(DAY)
        assert 0.0 <= whole < worst
        assert regret(DAY // 2) > 2.0 * whole
    # A period spent to its last step leaves rounding, not an overrun, and says nothing.
    assert [r for r in caplog.records if getattr(r, "chc_event", None) == "budget_overrun"] == []


@pytest.mark.parametrize("start", [0.0, 2 * DT, -5 * DT])
def test_the_first_step_of_a_period_plans_that_period_under_its_whole_budget(start: float) -> None:
    controller = RecedingHorizon(
        MODEL, COST, DT, DAY, 0.0, U_HI, budget=PeriodBudget(np.ones(1), 3.0, DAY, start)
    )
    plan = controller.step(X0, t=start + DAY * DT, spent=0.0)  # the second day's first step
    row = LinearConstraint(np.ones((1, DAY)), -np.inf, 3.0)
    reference = causal_plan(MODEL, X0, COST, DT, DAY, 0.0, U_HI, constraints=(row,))
    np.testing.assert_allclose(plan.actions, reference.actions, atol=1e-12)


@pytest.mark.parametrize(
    ("window", "bounds"),
    [(DAY, [2.0, 3.0 * 5 / DAY]), (2 * DAY, [2.0, 3.0, 3.0 * 5 / DAY]), (4, [2.0 * 4 / 7])],
    ids=["a-day", "two-days", "short"],
)
def test_each_period_in_the_window_gets_its_share(
    window: int, bounds: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    """Step 5 of a 12-step day, 1.0 of 3.0 spent: 2.0 is left for the day's 7 remaining steps."""
    controller = RecedingHorizon(
        MODEL, COST, DT, window, 0.0, U_HI, budget=PeriodBudget(np.ones(1), 3.0, DAY)
    )
    with caplog.at_level(logging.DEBUG, logger="chc.mpc"):
        plan = controller.step(X0, t=5 * DT, spent=1.0)
    [record] = [r for r in caplog.records if getattr(r, "chc_event", None) == "budget"]
    assert record.bounds == pytest.approx(bounds)
    edges = np.cumsum([0, 7, DAY, DAY])[: len(bounds) + 1].clip(max=window)
    actions = np.asarray(plan.actions).ravel()
    for bound, first, last in zip(bounds, edges[:-1], edges[1:], strict=True):
        assert actions[first:last].sum() <= bound + 1e-9


def test_an_overspent_period_spends_nothing_more_and_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    controller = RecedingHorizon(
        MODEL, COST, DT, DAY, 0.0, U_HI, budget=PeriodBudget(np.ones(1), 3.0, DAY)
    )
    with caplog.at_level(logging.WARNING, logger="chc.mpc"):
        plan = controller.step(X0, t=5 * DT, spent=4.0)
    [record] = [r for r in caplog.records if getattr(r, "chc_event", None) == "budget_overrun"]
    assert record.left == -1.0
    assert float(jnp.max(jnp.abs(plan.actions[:7]))) < 1e-9
    assert float(jnp.sum(plan.actions[7:])) > 0.0  # the next day's budget is still there to plan


def test_a_box_that_forces_spend_is_obeyed_and_the_overrun_named(
    caplog: pytest.LogCaptureFixture,
) -> None:
    controller = RecedingHorizon(
        MODEL, COST, DT, DAY, 0.2, U_HI, budget=PeriodBudget(np.ones(1), 3.0, DAY)
    )
    with caplog.at_level(logging.WARNING, logger="chc.mpc"):
        plan = controller.step(X0, t=5 * DT, spent=2.0)  # 1.0 left; 7 steps at 0.2 spend 1.4
    [record] = [r for r in caplog.records if getattr(r, "chc_event", None) == "budget_overrun"]
    assert (record.floor, record.allowed) == pytest.approx((1.4, 1.0))
    np.testing.assert_allclose(plan.actions[:7], 0.2, atol=1e-9)


def test_the_window_keeps_its_own_rows_beside_the_budget() -> None:
    rate = LinearConstraint.rate_limit(DAY, [0.3])
    controller = RecedingHorizon(
        MODEL,
        COST,
        DT,
        DAY,
        0.0,
        U_HI,
        constraints=(rate,),
        budget=PeriodBudget(np.ones(1), 3.0, DAY),
    )
    actions = np.asarray(controller.step(X0, t=3 * DT, spent=0.5).actions).ravel()
    assert np.max(np.abs(np.diff(actions))) <= 0.3 + 1e-9
    assert actions[:9].sum() <= 2.5 + 1e-9


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({"weights": np.ones((1, 1))}, "finite"),
        ({"weights": np.array([np.nan])}, "finite"),
        ({"weights": np.zeros(1)}, "caps nothing"),
        ({"amount": -1.0}, "non-negative"),
        ({"amount": float("inf")}, "non-negative"),
        ({"period": 0}, "at least one step"),
        ({"start": float("nan")}, "start"),
    ],
)
def test_a_budget_that_caps_nothing_sensible_is_refused(
    arguments: dict[str, object], match: str
) -> None:
    fields: dict[str, object] = {"weights": np.ones(1), "amount": 3.0, "period": DAY}
    with pytest.raises(ValueError, match=match):
        PeriodBudget(**(fields | arguments))


def test_a_step_without_its_ledger_or_off_the_clock_is_refused() -> None:
    budgeted = RecedingHorizon(
        MODEL, COST, DT, DAY, 0.0, U_HI, budget=PeriodBudget(np.ones(1), 3.0, DAY)
    )
    with pytest.raises(ValueError, match="needs spent"):
        budgeted.step(X0, t=0.0)
    with pytest.raises(ValueError, match="dt grid"):
        budgeted.step(X0, t=0.3, spent=0.0)
    with pytest.raises(ValueError, match="has no budget"):
        RecedingHorizon(MODEL, COST, DT, DAY, 0.0, U_HI).step(X0, spent=0.0)
    two_levers = PeriodBudget(np.ones(2), 3.0, DAY)
    with pytest.raises(ValueError, match="weighs 2 levers"):
        RecedingHorizon(MODEL, COST, DT, DAY, 0.0, U_HI, budget=two_levers).step(X0, spent=0.0)
