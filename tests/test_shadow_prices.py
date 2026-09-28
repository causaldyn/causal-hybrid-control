"""What a constraint row is worth to a plan: its KKT multiplier, read off the plan's own problem.

The separable cases have exact answers -- the water-filling price by bisection, and a two-row case
by finite differences of an SLSQP solve. The plant cases are the ledger lab's, ``x' = -0.5 x + u``
tracked to 2 over a 48-step day, where Octave's ``qp`` gives the budget's multiplier as 1.75366592.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import OptimizeResult, minimize

from chc import (
    BarrierConstraint,
    CausalPlan,
    LinearConstraint,
    QuadraticCost,
    RecedingHorizon,
    SupportModel,
    causal_plan,
)
from chc.dynamics import LinearDynamics
from chc.mpc import PeriodBudget
from chc.plan import _row_prices
from chc.support import _augmented
from chc.uncertainty import ConfoundingRobustPenalty

DT = 0.5
DAY = 48
MODEL = LinearDynamics(jnp.array([[-0.5]]), jnp.array([[1.0]]))
COST = QuadraticCost(Q=jnp.eye(1), R=jnp.array([[0.05]]), Qf=jnp.eye(1), x_target=jnp.array([2.0]))
X0 = jnp.array([1.0])
U_HI = 50.0
HALF_A_DAY = 24.438098  # half of what the day spends with no budget: 48.876197

HOURS = np.arange(24)
PULL = 1.0 + 0.9 * np.cos(2 * np.pi * (HOURS - 0.7 * 24) / 24)


def _water_filling(budget: float, cap: float = np.inf) -> tuple[np.ndarray, float]:
    """``min sum u^2/2 - PULL u`` under ``sum u <= budget`` and ``0 <= u <= cap``, by bisection."""

    def spend(price: float) -> float:
        return float(np.clip(PULL - price, 0.0, cap).sum())

    low, high = 0.0, float(PULL.max())
    if spend(0.0) <= budget:
        return np.clip(PULL, 0.0, cap), 0.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if spend(middle) > budget else (low, middle)
    price = 0.5 * (low + high)
    return np.clip(PULL - price, 0.0, cap), price


def _prices(u: np.ndarray, rows: list[tuple[np.ndarray, float, float]], hi: float = np.inf):
    return _row_prices(
        u - PULL,
        u,
        np.array([row[0] for row in rows]),
        np.array([row[1] for row in rows]),
        np.array([row[2] for row in rows]),
        np.zeros(u.size),
        np.full(u.size, hi),
        1e-7,
    )


def test_the_water_filling_price_is_read_exactly() -> None:
    budget = 0.5 * float(np.clip(PULL, 0.0, None).sum())
    u, price = _water_filling(budget)
    assert int((u == 0.0).sum()) == 9  # the night hours sit on the box, which absorbs them
    prices = _prices(u, [(np.ones(24), -np.inf, budget)])
    [row] = prices.rows
    assert row.status == "exact"
    assert row.price == pytest.approx(price, abs=1e-12)
    assert prices.residual < 1e-14
    assert prices.dual_feasible


def test_a_budget_that_only_just_binds_is_weakly_active() -> None:
    budget = float(np.clip(PULL, 0.0, None).sum())
    u, _ = _water_filling(budget)
    [row] = _prices(u, [(np.ones(24), -np.inf, budget)]).rows
    assert (row.status, row.price) == ("weakly_active", 0.0)


def test_an_equality_row_is_priced_with_either_sign() -> None:
    """A day that must spend half again what it would: every hour rises by the same ``c``, and
    raising the spend costs ``c`` a unit, so the row's price is ``-c``."""
    spend = 1.5 * float(PULL.sum())
    c = (spend - float(PULL.sum())) / 24
    prices = _prices(PULL + c, [(np.ones(24), spend, spend)])
    [row] = prices.rows
    assert row.status == "exact"
    assert row.price == pytest.approx(-c, abs=1e-12)
    assert prices.dual_feasible


@pytest.mark.parametrize(("level", "cap"), [(0.0, np.inf), (2.0, 2.0)])
def test_a_plan_held_on_its_box_against_the_objective_is_at_no_kkt_point(
    level: float, cap: float
) -> None:
    """Every hour on the floor, which the objective would raise, or on a cap of 2, which it would
    lower. Nothing is free, so the residual is zero, and only the box's signs say so."""
    prices = _prices(np.full(24, level), [(np.ones(24), -np.inf, 100.0)], hi=cap)
    assert prices.residual == 0.0
    assert not prices.dual_feasible


def test_a_budget_that_meets_every_cap_has_a_range_of_prices() -> None:
    """Every hour at its cap of 0.4 and the budget at 24 * 0.4: raising the budget buys nothing,
    and lowering it costs what the flattest hour gains from the cap, ``1 - 0.4``."""
    u = np.full(24, 0.4)
    [row] = _row_prices(
        u - 1.0,
        u,
        np.ones((1, 24)),
        np.array([-np.inf]),
        np.array([24 * 0.4]),
        np.zeros(24),
        np.full(24, 0.4),
        1e-7,
    ).rows
    assert (row.status, row.price) == ("degenerate", None)
    assert row.interval == pytest.approx((0.0, 0.6), abs=1e-5)


def test_a_row_given_twice_splits_its_price_any_way() -> None:
    budget = 0.5 * float(np.clip(PULL, 0.0, None).sum())
    u, price = _water_filling(budget)
    first, second = _prices(u, [(np.ones(24), -np.inf, budget)] * 2).rows
    for row in (first, second):
        assert row.status == "degenerate"
        assert row.interval == pytest.approx((0.0, price), abs=1e-5)


def test_two_different_rows_are_each_priced_at_their_own_slope() -> None:
    """A day's budget and a cap on the peak hours, 12 to 17, both binding."""
    peak = np.zeros(24)
    peak[12:18] = 1.0
    budget = 0.4 * float(np.clip(PULL, 0.0, None).sum())

    def solve(budget: float, cap: float) -> tuple[np.ndarray, float]:
        result = minimize(
            lambda u: float(np.sum(0.5 * u**2 - PULL * u)),
            np.full(24, 0.1),
            jac=lambda u: u - PULL,
            bounds=[(0.0, None)] * 24,
            constraints=[
                {"type": "ineq", "fun": lambda u: budget - u.sum()},
                {"type": "ineq", "fun": lambda u: cap - peak @ u},
            ],
            method="SLSQP",
            options={"ftol": 1e-14, "maxiter": 500},
        )
        return result.x, float(result.fun)

    u, _ = solve(budget, 1.2)
    prices = _row_prices(
        u - PULL,
        u,
        np.stack([np.ones(24), peak]),
        np.full(2, -np.inf),
        np.array([budget, 1.2]),
        np.zeros(24),
        np.full(24, np.inf),
        1e-6,
    )
    step = 1e-3
    slopes = (
        (solve(budget - step, 1.2)[1] - solve(budget + step, 1.2)[1]) / (2 * step),
        (solve(budget, 1.2 - step)[1] - solve(budget, 1.2 + step)[1]) / (2 * step),
    )
    assert [row.status for row in prices.rows] == ["exact", "exact"]
    assert [row.price for row in prices.rows] == pytest.approx(slopes, abs=1e-6)


Row = tuple[np.ndarray, float, float]


def _optimum(h: np.ndarray, g: np.ndarray, hi: np.ndarray, rows: list[Row]) -> OptimizeResult:
    """``min sum h u^2/2 - g u`` under ``rows`` and ``0 <= u <= hi``, by SLSQP."""
    constraints = []
    for a, lower, upper in rows:
        if lower == upper:
            constraints.append({"type": "eq", "fun": lambda u, a=a, b=lower: a @ u - b})
            continue
        if np.isfinite(upper):
            constraints.append({"type": "ineq", "fun": lambda u, a=a, b=upper: b - a @ u})
        if np.isfinite(lower):
            constraints.append({"type": "ineq", "fun": lambda u, a=a, b=lower: a @ u - b})
    return minimize(
        lambda u: float(np.sum(0.5 * h * u**2 - g * u)),
        np.clip(g / h, 0.0, np.minimum(hi, 1.0)),
        jac=lambda u: h * u - g,
        bounds=[(0.0, None if np.isinf(cap) else cap) for cap in hi],
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-15, "maxiter": 1000},
    )


def _relaxed(rows: list[Row], index: int, step: float) -> list[Row]:
    """``rows`` with row ``index`` relaxed by ``step``: an upper bound or an equality's value
    raised, a lower bound lowered."""
    a, lower, upper = rows[index]
    if lower == upper:
        moved = (a, lower + step, upper + step)
    elif np.isfinite(upper):
        moved = (a, lower, upper + step)
    else:
        moved = (a, lower - step, upper)
    return [*rows[:index], moved, *rows[index + 1 :]]


def test_every_exact_price_is_the_slope_of_the_optimum() -> None:
    """Random separable problems under a box, with upper, lower and equality rows: each exact
    price against a central difference of the optimum, re-solved with its row relaxed and
    tightened. The optimum is piecewise quadratic in a bound, so the difference is exact inside an
    active set, and what is left is the re-solves' own error over the step: at a step of 1e-5 one
    price missed by 1.2e-6, at 1e-4 by 4.9e-7."""
    rng = np.random.default_rng(11)
    gaps = []
    for _ in range(40):
        n = int(rng.integers(6, 14))
        h, g = rng.uniform(0.5, 2.0, n), rng.uniform(-0.5, 2.0, n)
        hi = np.where(rng.random(n) < 0.4, rng.uniform(0.2, 1.2, n), np.inf)
        rows: list[Row] = []
        for kind in rng.choice(["upper", "lower", "equality"], int(rng.integers(1, 4))):
            a = np.where(rng.random(n) < 0.7, rng.uniform(0.2, 1.5, n), 0.0)
            if not a.any():
                a[0] = 1.0
            value = float(a @ np.clip(g / h, 0.0, hi))
            if kind == "upper":
                rows.append((a, -np.inf, value * rng.uniform(0.4, 0.9)))
            elif kind == "lower":
                rows.append((a, value * rng.uniform(1.05, 1.3), np.inf))
            else:
                level = value * rng.uniform(0.6, 1.2)
                rows.append((a, level, level))
        solved = _optimum(h, g, hi, rows)
        if not solved.success:
            continue
        u = np.clip(solved.x, 0.0, hi)
        prices = _row_prices(
            h * u - g,
            u,
            np.array([row[0] for row in rows]),
            np.array([row[1] for row in rows]),
            np.array([row[2] for row in rows]),
            np.zeros(n),
            hi,
            1e-6,
        )
        assert prices.residual < 1e-6
        assert prices.dual_feasible
        for index, price in enumerate(prices.rows):
            if price.status == "exact":
                tightened, relaxed = (
                    _optimum(h, g, hi, _relaxed(rows, index, step)).fun for step in (-1e-4, 1e-4)
                )
                slope = (tightened - relaxed) / 2e-4
                gaps.append(abs(price.price - slope) / max(1.0, abs(slope)))
    assert len(gaps) >= 50
    assert max(gaps) < 1e-5


def _budgeted(
    budget: float,
    u_hi: float = U_HI,
    *,
    support: SupportModel | None = None,
    lam_supp: float = 0.0,
    uncertainty: ConfoundingRobustPenalty | None = None,
    lam_unc: float = 0.0,
    steps: int = 10_000,
    barrier: BarrierConstraint | None = None,
) -> CausalPlan:
    row = LinearConstraint(np.ones((1, DAY)), -np.inf, budget)
    return causal_plan(
        MODEL,
        X0,
        COST,
        DT,
        DAY,
        0.0,
        u_hi,
        support=support,
        lam_supp=lam_supp,
        uncertainty=uncertainty,
        lam_unc=lam_unc,
        steps=steps,
        constraints=(row,),
        barrier=barrier,
    )


def test_the_budget_price_of_a_plan_is_the_qp_multiplier_and_the_slope_of_its_cost() -> None:
    plan = _budgeted(HALF_A_DAY)
    prices = plan.shadow_prices()
    [row] = prices.rows
    assert row.status == "exact"
    assert row.price == pytest.approx(1.75366592, rel=2e-6)  # Octave's qp, same plant and day
    slope = (_budgeted(HALF_A_DAY - 0.2).task_cost - _budgeted(HALF_A_DAY + 0.2).task_cost) / 0.4
    assert row.price == pytest.approx(slope, rel=2e-6)
    assert prices.dual_feasible
    assert prices.residual < 1e-4


def test_a_budget_that_meets_every_cap_of_a_plan_is_priced_by_its_range() -> None:
    """A cap of 0.4 holds every step at the cap with no budget, so a budget of ``48 * 0.4`` binds
    with nothing free: raising it buys nothing, and lowering it costs the left slope."""
    plan = _budgeted(DAY * 0.4, u_hi=0.4)
    [row] = plan.shadow_prices().rows
    assert (row.status, row.price) == ("degenerate", None)
    low, high = row.interval
    assert low == pytest.approx(0.0, abs=1e-5)
    step = 1e-3
    left = (_budgeted(DAY * 0.4 - step, u_hi=0.4).task_cost - plan.task_cost) / step
    assert high == pytest.approx(left, rel=1e-2)


def test_a_spend_floor_is_priced_by_what_lowering_it_saves() -> None:
    floor = 60.0  # the day spends 48.9 with no budget
    row = LinearConstraint(np.ones((1, DAY)), floor, np.inf)

    def cost(floor: float) -> float:
        row = LinearConstraint(np.ones((1, DAY)), floor, np.inf)
        return causal_plan(MODEL, X0, COST, DT, DAY, 0.0, U_HI, constraints=(row,)).task_cost

    [price] = (
        causal_plan(MODEL, X0, COST, DT, DAY, 0.0, U_HI, constraints=(row,)).shadow_prices().rows
    )
    assert price.status == "exact"
    assert price.price == pytest.approx((cost(floor + 0.2) - cost(floor - 0.2)) / 0.4, rel=1e-5)
    assert price.price > 0.0


def test_a_pessimistic_plan_is_priced_on_the_objective_it_minimised() -> None:
    """The support penalty holds the plan near the logged actions, and the confounding penalty
    charges 0.2 a unit of action, so the budget's price is the slope of the penalised objective.
    Priced on the task cost alone, the budget's multiplier would leave a residual of 0.1 on the
    free steps. Priced without the confounding penalty, it would be 0.2 too high with the same
    residual, since the penalty is flat in the direction the row is: only the slope says so."""
    logged_x = 1.5 + 0.3 * jax.random.normal(jax.random.key(0), (400, 1))
    logged_u = 0.3 + 0.1 * jax.random.normal(jax.random.key(1), (400, 1))
    support = SupportModel.fit(logged_x, logged_u)
    penalty = ConfoundingRobustPenalty(radius=0.2)

    def plan(budget: float) -> CausalPlan:
        return _budgeted(budget, support=support, lam_supp=1.0, uncertainty=penalty, lam_unc=1.0)

    def objective(budget: float) -> float:
        actions = plan(budget).actions
        return float(_augmented(MODEL, X0, actions, DT, COST, support, 1.0, penalty, 1.0))

    prices = plan(10.0).shadow_prices()
    [row] = prices.rows
    assert row.status == "exact"
    assert row.price == pytest.approx((objective(9.8) - objective(10.2)) / 0.4, rel=1e-6)
    assert prices.residual < 1e-4
    assert prices.dual_feasible


def test_an_unfinished_solve_says_its_prices_are_not_to_be_trusted() -> None:
    early = _budgeted(HALF_A_DAY, steps=3)
    assert early.solver_status == "max_iterations"
    assert early.shadow_prices().residual > 1e3 * _budgeted(HALF_A_DAY).shadow_prices().residual


def test_a_budgeted_step_prices_its_budget() -> None:
    """The first step of a day plans that day under one row, so it prices the day's budget."""
    controller = RecedingHorizon(
        MODEL, COST, DT, DAY, 0.0, U_HI, budget=PeriodBudget(np.ones(1), HALF_A_DAY, DAY)
    )
    [row] = controller.step(X0, t=0.0, spent=0.0).shadow_prices().rows
    assert row.status == "exact"
    assert row.price == pytest.approx(1.75366592, rel=2e-6)


def test_a_plan_without_rows_has_nothing_to_price() -> None:
    prices = causal_plan(MODEL, X0, COST, DT, DAY, 0.0, U_HI).shadow_prices()
    assert prices.rows == ()
    assert prices.smallest_singular_value == np.inf


def test_a_barrier_held_plan_or_one_built_by_hand_is_not_priced() -> None:
    barrier = BarrierConstraint(lambda x: x[0] - 0.5)
    with pytest.raises(ValueError, match="barrier"):
        _budgeted(HALF_A_DAY, barrier=barrier).shadow_prices()
    plan = _budgeted(HALF_A_DAY)
    by_hand = CausalPlan(plan.actions, plan.trajectory, plan.task_cost, None, None, "converged", 1)
    with pytest.raises(ValueError, match="no problem"):
        by_hand.shadow_prices()
