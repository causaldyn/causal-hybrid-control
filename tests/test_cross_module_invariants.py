"""One executed action, one set of units and one precision, read the same by every module.

A prescription is read by several modules at once: the plan's cost and path, the regret bound, the
decision weight, the error tube, the budgets' prices, the policy and the JSON record. Each test
here ties two of those readings together, or a reading to the same decision posed another way:
the cost reported is the cost of running the schedule, the record's schedule is the schedule, a
column nothing reads moves nothing, and a problem posed in other units reads the same. The
allocation planner is held to the last two in currency and in outcome units.

The logs are one state ``y``, two levers and a confounder ``z``, with ``y`` moving by
``0.1 (-0.5 y + 0.8 u1 + 0.1 u2 + 1.5 z)`` a period, under three logging policies: every lever
moved (``free``), ``u2 = 2 u1`` (``together``), and ``u1`` set from the state (``state``).
"""

from __future__ import annotations

import functools
import importlib
import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from chc.allocation import ReturnTarget, allocate, budget_for
from chc.control import LinearConstraint
from chc.cost import QuadraticCost, total_cost
from chc.decision import (
    Constraint,
    Lever,
    PrescribedPolicy,
    Prescription,
    Target,
    _rate,
    prescribe,
)
from chc.dynamics import LinearDynamics
from chc.graph import CausalGraph
from chc.integrate import rollout
from chc.mpc import PeriodBudget
from chc.panel import Panel
from chc.plan import _row_prices, _Ruled, causal_plan, plan_regret_bound
from chc.response import (
    Channel,
    DelayedAdstock,
    GeometricAdstock,
    Hill,
    Tanh,
    WeibullAdstock,
)
from chc.uncertainty import _noise_tube, _tube

# Public as jax.enable_x64 from jax 0.8.0; the floor, 0.4.30, has only jax.experimental.enable_x64.
if hasattr(jax, "enable_x64"):
    enable_x64 = jax.enable_x64
else:  # pragma: no cover - the dependency floor's spelling
    enable_x64 = importlib.import_module("jax.experimental").enable_x64

DT = 0.1
HORIZON = 3
EDGES = [("z", "y"), ("u1", "y"), ("u2", "y"), ("z", "u1"), ("z", "u2")]
POLICIES = ["free", "together", "state"]


@functools.cache
def _columns(policy: str) -> dict[str, np.ndarray]:
    """A hundred units of fifteen periods logged under ``policy``."""
    rng = np.random.default_rng(303)
    rows: dict[str, list[float]] = {name: [] for name in ("unit", "time", "y", "u1", "u2", "z")}
    for unit in range(100):
        y = rng.normal()
        for period in range(15):
            z, noise = rng.normal(), rng.normal()
            if policy == "together":
                u1 = 0.9 * z - 0.3 * y + 0.5 * noise
                u2 = 2.0 * u1
            elif policy == "state":
                u1, u2 = -0.3 * y, 0.6 * noise
            else:
                u1, u2 = 0.9 * z + 0.5 * rng.normal(), 0.6 * noise
            for name, value in zip(rows, (unit, period, y, u1, u2, z), strict=True):
                rows[name].append(value)
            y += DT * (-0.5 * y + 0.8 * u1 + 0.1 * u2 + 1.5 * z) + rng.normal(scale=0.01)
    return {name: np.asarray(values) for name, values in rows.items()}


@functools.cache
def _decide(
    policy: str,
    *,
    levers_in: tuple[float, float] = (1.0, 1.0),
    levers_at: tuple[float, float] = (0.0, 0.0),
    state_in: float = 1.0,
    unread: bool = False,
    budget: bool = False,
    max_levers: int | None = None,
    floor: float | None = None,
    hold: bool = False,
) -> Prescription:
    """``policy``'s log and problem, each lever logged in ``levers_in`` times its units and the
    state in ``state_in`` times its own: the boxes and a budget's weights with the levers, the
    target, the start and the tube's tolerance with the state, and the prices so the cost is the
    same number. ``unread`` adds a column no lever, target or covariate names. ``budget`` holds
    the plan to spend at most 0.6 over the horizon, ``u1`` at 1 a unit and ``u2`` at 3, on boxes
    from 0, where it binds. ``floor`` bounds the state to ``[floor, 1.1]`` in its own units, audited
    at ``gamma = 1.5``, and ``hold`` holds the bounds in the solve. ``levers_at`` moves each
    lever's zero, the log and the box with it."""
    columns = dict(_columns(policy))
    first, second = levers_in
    columns["u1"] = first * columns["u1"] + levers_at[0]
    columns["u2"] = second * columns["u2"] + levers_at[1]
    columns["y"] = state_in * columns["y"]
    if unread:
        columns["w"] = np.random.default_rng(7).normal(scale=1e6, size=columns["y"].size)
    low = 0.0 if budget else -2.0
    budgets = (
        [PeriodBudget(weights=np.array([1.0 / first, 3.0 / second]), amount=0.6, period=HORIZON)]
        if budget
        else []
    )
    return prescribe(
        Panel.from_frame(columns, unit="unit", time="time"),
        levers=[
            Lever(
                "u1",
                lo=low * first + levers_at[0],
                hi=2.0 * first + levers_at[0],
                unit_cost=0.01 * (state_in / first) ** 2,
            ),
            Lever(
                "u2",
                lo=low * second + levers_at[1],
                hi=2.0 * second + levers_at[1],
                unit_cost=(state_in / second) ** 2,
            ),
        ],
        target=Target("y", value=(2.0 if budget else 1.0) * state_in),
        adjustment=CausalGraph.from_edges(EDGES),
        budgets=budgets,
        max_levers=max_levers,
        constraints=(
            [] if floor is None else [Constraint("y", lo=floor * state_in, hi=1.1 * state_in)]
        ),
        hold_constraints=hold,
        gamma=1.0 if floor is None else 1.5,
        horizon=HORIZON,
        dt=DT,
        tolerance=0.5 * state_in,
        x0=jnp.full(1, state_in),
    )


def _schedule(result: Prescription) -> np.ndarray:
    return np.asarray(result.schedule.magnitudes)


# ---- the cost reported is the cost of running the schedule ----


@pytest.mark.parametrize("policy", POLICIES)
def test_the_plan_s_path_and_cost_are_its_schedule_s_on_the_fitted_field(policy: str) -> None:
    """The schedule, each level held over its step, run on the fitted field and priced at the
    task's cost, is the plan's path and its cost: what the regret bound, the tube and the record
    read is what the schedule does. A lever on the log's rule takes its schedule's level as the
    plan's field reads it, so the field without the rule runs the same path."""
    result = _decide(policy)
    plan = result.plan
    assert plan is not None
    problem = plan._problem
    assert problem is not None
    field = problem.model.dynamics if isinstance(problem.model, _Ruled) else problem.model
    price = QuadraticCost(problem.cost.Q, problem.cost.R, problem.cost.Qf, problem.cost.x_target)
    schedule = jnp.asarray(_schedule(result))
    np.testing.assert_allclose(
        np.asarray(rollout(field, problem.x0, schedule, problem.dt)),
        np.asarray(plan.trajectory),
        rtol=1e-12,
        atol=0.0,
    )
    cost = float(total_cost(field, problem.x0, schedule, problem.dt, price))
    assert cost == pytest.approx(plan.task_cost, rel=1e-12, abs=0.0)


def test_the_selection_s_last_step_is_the_plan_and_its_idle_cost_is_doing_nothing() -> None:
    """With ``max_levers``, the step the selection kept is the plan the prescription returns, at
    its cost and its regret bound, in the object and in the record; the idle cost is the
    schedule of zeros run on the plan's field."""
    result = _decide("free", max_levers=1)
    plan, selection = result.plan, result.selection
    assert plan is not None
    assert selection is not None
    problem = plan._problem
    assert problem is not None
    last = selection.steps[-1]
    assert last.task_cost == plan.task_cost
    assert last.regret_bound == result.certificate.regret_bound
    idle = total_cost(
        problem.model, problem.x0, jnp.zeros_like(plan.actions), problem.dt, problem.cost
    )
    assert selection.idle_cost == float(idle)
    record = result.to_json()["selection"]
    assert record["idle_cost"] == selection.idle_cost
    assert record["steps"][-1]["task_cost"] == plan.task_cost


# ---- the record's schedule is the schedule, in either precision ----


@pytest.mark.parametrize("double", [True, False], ids=["float64", "float32"])
@pytest.mark.parametrize("policy", ["free", "state"])
def test_the_record_and_the_policy_carry_the_schedule_to_the_bit(policy: str, double: bool) -> None:
    """The record's schedule is the schedule, number for number; the policy run along the
    predicted path takes the schedule's actions, and so does the policy read back from its own
    record. In float32 as in float64: the schedule's floats are the solve's, and each reader
    carries them unrounded."""
    with enable_x64(double):
        # float32 is fitted afresh: the cache holds float64's
        result = _decide(policy) if double else _decide.__wrapped__(policy)
        plan = result.plan
        assert plan is not None
        schedule = _schedule(result)
        assert schedule.dtype == (np.float64 if double else np.float32)
        record = json.loads(json.dumps(result.to_json(), allow_nan=False))
        assert np.array_equal(np.asarray(record["schedule"], dtype=np.float64), schedule)
        path = np.asarray(plan.trajectory)
        policy_run = result.policy()
        again = PrescribedPolicy.from_json(
            json.loads(json.dumps(policy_run.to_json(), allow_nan=False))
        )
        for runner in (policy_run, again):
            actions = np.stack([runner.actions_at(step, path[step]) for step in range(HORIZON)])
            assert np.array_equal(actions, schedule)


# ---- a column nothing reads moves nothing ----


@pytest.mark.parametrize("policy", ["free", "state"])
def test_a_column_the_problem_does_not_name_moves_nothing(policy: str) -> None:
    """A column no lever, target or covariate names, a million times the others' size, leaves the
    schedule, the cost and every number of the certificate where they were, to the bit."""
    one, other = _decide(policy), _decide(policy, unread=True)
    assert one.plan is not None
    assert other.plan is not None
    assert np.array_equal(_schedule(other), _schedule(one))
    assert other.plan.task_cost == one.plan.task_cost
    assert other.to_json()["certificate"] == one.to_json()["certificate"]


def test_the_column_of_a_lever_on_the_log_s_rule_leaves_the_regret_bound_certified() -> None:
    """The plan keeps a column for ``u1``, which follows the log's rule, and nothing reads it: its
    box has zero width and the field, the price and the channel read the rule's level. The regret
    bound's curvature was read off the whole Hessian, whose row for that column is zero, so its
    least eigenvalue was 0 to rounding, -2.2e-16 here, and the bound was refused, ``inf``. It now
    reads the curvature along the actions the box lets move, and the column adds nothing to it."""
    result = _decide("state")
    plan = result.plan
    assert plan is not None
    problem = plan._problem
    assert problem is not None
    certificate = result.certificate
    assert certificate.rule_levers == ("u1",)
    assert certificate.regret_status == "diagnostic"
    assert 0.0 <= certificate.regret_bound < 1e-6 * plan.task_cost

    gap = plan_regret_bound(
        plan,
        problem.model,
        problem.x0,
        problem.cost,
        problem.dt,
        problem.u_lo,
        problem.u_hi,
        probes=4,
    )
    assert gap.bound == certificate.regret_bound
    assert gap.modulus > 0.0
    assert gap.per_lever[0] == 0.0


@given(
    a=st.floats(min_value=-1.0, max_value=0.5),
    b=st.floats(min_value=0.2, max_value=2.0),
    price=st.floats(min_value=0.01, max_value=1.0),
    level=st.floats(min_value=-5.0, max_value=5.0),
    side=st.floats(min_value=0.05, max_value=2.0),
    horizon=st.integers(min_value=1, max_value=4),
)
@settings(deadline=None, max_examples=20)
def test_an_action_nothing_reads_moves_no_regret_bound(
    a: float, b: float, price: float, level: float, side: float, horizon: int
) -> None:
    """A second lever the field does not read, at no price, its box pinned at any level: the plan
    of the first lever, in a box of any width, with that column beside it has the one-lever plan's
    regret bound, modulus and status, the modulus being the least eigenvalue of the one-lever
    cost's Hessian. The whole Hessian has a zero row there, and its least eigenvalue, 0 or a
    rounding below it, gave the Frank-Wolfe gap or a refusal."""
    one = LinearDynamics(jnp.array([[a]]), jnp.array([[b]]))
    two = LinearDynamics(jnp.array([[a]]), jnp.array([[b, 0.0]]))
    cost_one = QuadraticCost(
        Q=jnp.eye(1), R=price * jnp.eye(1), Qf=2.0 * jnp.eye(1), x_target=jnp.array([1.0])
    )
    cost_two = QuadraticCost(
        Q=jnp.eye(1),
        R=jnp.diag(jnp.array([price, 0.0])),
        Qf=2.0 * jnp.eye(1),
        x_target=jnp.array([1.0]),
    )
    x0 = jnp.array([0.3])
    plan = causal_plan(one, x0, cost_one, DT, horizon, -side, side)
    alone = plan_regret_bound(plan, one, x0, cost_one, DT, -side, side)
    beside = jnp.concatenate([plan.actions, jnp.full((horizon, 1), level)], axis=1)
    lo, hi = jnp.array([-side, level]), jnp.array([side, level])
    paired = plan_regret_bound(replace(plan, actions=beside), two, x0, cost_two, DT, lo, hi)
    assert paired.status == alone.status == "certified"
    hessian = jax.hessian(lambda us: total_cost(one, x0, us, DT, cost_one))(plan.actions)
    least = float(np.linalg.eigvalsh(np.asarray(hessian).reshape(horizon, horizon))[0])
    assert alone.modulus == pytest.approx(least, rel=1e-9, abs=0.0)
    assert paired.modulus == pytest.approx(alone.modulus, rel=1e-12, abs=0.0)
    assert paired.bound == pytest.approx(alone.bound, rel=1e-9, abs=1e-15)
    assert paired.per_lever[1] == 0.0


@pytest.mark.parametrize("units", [1e-3, 1.0, 1e3])
def test_a_row_s_multiplier_reads_as_zero_against_the_rows_pushes_in_any_units(
    units: float,
) -> None:
    """``(u1^2 + u2^2) / 2`` held to ``u1 >= 1`` and ``u2 >= 1e-4``, the cost in ``units`` times
    its own: the second row pushes a ten-thousandth of the first, and reads as met with no pull;
    the first is a price, in any units of the cost. The guess, zeros, is the cost's own minimum, so
    its slope says nothing, and the rows' pushes are the scale. Against at least 1, the first row
    read as no pull with the cost in thousandths."""
    actions = np.array([1.0, 1e-4])
    prices = _row_prices(
        units * actions,
        actions,
        np.eye(2),
        np.array([1.0, 1e-4]),
        np.full(2, np.inf),
        np.full(2, -np.inf),
        np.full(2, np.inf),
        1e-6,
        np.zeros(2),
    )
    assert [row.status for row in prices.rows] == ["exact", "weakly_active"]
    assert prices.rows[0].price == pytest.approx(units, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("units", [1e-3, 1.0, 1e3])
def test_a_row_met_with_no_pull_reads_so_in_its_price_and_its_weight_in_any_units(
    units: float,
) -> None:
    """The scalar plan held to ``u_0 + u_1 >= c``, ``c`` the free plan's own sum, the cost in
    ``units`` times its own: the row binds with a zero multiplier, and its price and the decision
    weight both read it so. At the plan the gradient is rounding; the cost's slope at the guess is
    the scale its multiplier is read against."""
    plant = LinearDynamics(jnp.array([[-0.5]]), jnp.array([[1.2]]))
    cost = QuadraticCost(
        Q=units * jnp.eye(1),
        R=units * jnp.array([[0.1]]),
        Qf=units * jnp.array([[2.0]]),
        x_target=jnp.array([0.0]),
    )
    optimum = np.array([-2.023822333873833, -1.248583272170075])
    row = LinearConstraint(np.ones((1, 2)), optimum.sum(), np.inf)
    plan = causal_plan(plant, jnp.ones(1), cost, DT, 2, -10.0, 10.0, constraints=(row,))
    at = replace(plan, actions=jnp.asarray(optimum[:, None]))
    assert [price.status for price in at.shadow_prices().rows] == ["weakly_active"]
    assert at.decision_weight().weakly_active == 1


def _logged_mean(
    policy: str,
    levers_in: tuple[float, float] = (1.0, 1.0),
    levers_at: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Each lever's mean over the log's transitions, every period but each unit's last."""
    columns = _columns(policy)
    rows = columns["time"] < columns["time"].max()
    return np.array(
        [
            scale * float(np.mean(columns[name][rows])) + at
            for name, scale, at in zip(("u1", "u2"), levers_in, levers_at, strict=True)
        ]
    )


def _tube_at(result: Prescription, reach: np.ndarray) -> np.ndarray:
    """The tube the plan's own rate, noise and persistence carry, each lever's error read off each
    covariance and priced at ``reach``, the largest read taken."""
    plan = result.plan
    assert plan is not None
    problem = plan._problem
    assert problem is not None
    errors = result.model_fit._lever_errors
    assert errors is not None
    rates = errors * reach
    error = float(np.max(np.sqrt(np.mean(rates * rates, axis=1))))
    noise, persistence = result.certificate.tube_noise, result.certificate.tube_persistence
    assert noise is not None
    assert persistence is not None
    rate, _ = _rate(problem.model, problem.x0, jnp.asarray(problem.u_lo), jnp.asarray(problem.u_hi))
    return _tube([rate] * HORIZON, [error] * HORIZON, DT, "rk4") + _noise_tube(
        np.asarray(rate), np.array([[noise**2]]), persistence, DT, HORIZON
    )


@pytest.mark.parametrize(
    "levers_in",
    [(1.0, 1.0), (1e-3, 1e-3), (1e3, 1e3), (1e-3, 1.0), (1e3, 1.0), (1.0, 1e3), (1e-3, 1e3)],
)
def test_the_tube_prices_each_lever_s_error_at_its_largest_distance_from_the_log_s_action(
    levers_in: tuple[float, float],
) -> None:
    """Each lever in units of its own, its box with it. Lever ``j``'s share of the channel's
    standard error, per unit of it, times the largest distance its box allows from the log's mean
    action, ``2 levers_in[j]`` give or take that mean, reads the same in any units of the lever,
    and the tube's rate error is its root mean over the channel's entries, the largest read
    taken. Before the fix, priced at the largest action of either box, ``u1`` in units 1e3 times
    its own ended the tube at 3.89 where it ends at 0.538; in thousandths, at 4.04. Priced at each
    lever's largest action from its zero, the reading ADR 0087 left, the drift's absorbing the
    channel at the log's mean action went unread."""
    result = _decide("free", levers_in=levers_in)
    plan = result.plan
    assert plan is not None
    mean = _logged_mean("free", levers_in)
    reach = np.maximum(
        np.abs(-2.0 * np.asarray(levers_in) - mean), np.abs(2.0 * np.asarray(levers_in) - mean)
    )
    assert plan.uncertainty_tube is not None
    np.testing.assert_allclose(
        np.asarray(plan.uncertainty_tube), _tube_at(result, reach), rtol=1e-12, atol=0.0
    )
    assert result.certificate.certified_horizon == _decide("free").certificate.certified_horizon


def test_a_lever_the_plan_holds_at_its_logged_level_adds_no_error() -> None:
    """``state``'s log sets ``u1`` from the state alone, and the plan holds that lever's column at
    its mean logged level, 0.028: the action the drift absorbed the channel at, so ``u1``'s share
    of the channel's error moves the field by nothing there, and the tube prices ``u2``'s alone,
    at its largest distance from its own mean. Priced from the lever's zero, at 0.028 and 2, the
    tube ended at 0.548948 where it ends at 0.548963; at the box's largest action, at 0.551."""
    result = _decide("state")
    plan = result.plan
    assert plan is not None
    problem = plan._problem
    assert problem is not None
    lo, hi = np.asarray(problem.u_lo), np.asarray(problem.u_hi)
    assert lo[0] == hi[0]
    assert 0.02 < lo[0] < 0.03
    assert result.model_fit._lever_errors is not None
    assert result.model_fit._lever_errors.shape == (3, 2)
    mean = _logged_mean("state")
    reach = np.array([0.0, max(abs(-2.0 - mean[1]), abs(2.0 - mean[1]))])
    assert plan.uncertainty_tube is not None
    np.testing.assert_allclose(
        np.asarray(plan.uncertainty_tube), _tube_at(result, reach), rtol=1e-12, atol=0.0
    )


@pytest.mark.parametrize("levers_at", [(3.0, 0.0), (0.0, -5.0), (100.0, 0.0)])
@pytest.mark.parametrize("policy", ["free", "state"])
def test_the_tube_reads_the_same_wherever_a_lever_s_zero_is(
    policy: str, levers_at: tuple[float, float]
) -> None:
    """Each lever's zero moved, the log and the box with it. The fit's drift absorbs the channel at
    the log's action, so the field it fits is the same one, read at the moved action, and so is its
    error at each action. The tube priced each lever's error from its zero: ``free``'s ended at
    0.5431 with ``u1`` moved by 3, where it ends at 0.5384, and at 0.7126 moved by 100, where it
    certified one step where it certifies two. It now prices it from the log's mean action, which
    moves with the zero."""
    one = _decide(policy).plan
    other = _decide(policy, levers_at=levers_at).plan
    assert one is not None
    assert other is not None
    assert one.uncertainty_tube is not None
    assert other.uncertainty_tube is not None
    np.testing.assert_allclose(
        np.asarray(other.uncertainty_tube), np.asarray(one.uncertainty_tube), rtol=1e-4, atol=0.0
    )
    assert other.certified_horizon == one.certified_horizon


def test_a_plan_whose_every_box_is_pinned_has_no_regret() -> None:
    """Every action fixed by its box: no move is feasible, so the bound is 0, certified, whatever
    the cost's curvature."""
    model = LinearDynamics(jnp.array([[-0.5]]), jnp.array([[1.0, 0.0]]))
    cost = QuadraticCost(
        Q=jnp.eye(1),
        R=jnp.diag(jnp.array([0.1, 0.0])),
        Qf=jnp.eye(1),
        x_target=jnp.array([1.0]),
    )
    lo = hi = jnp.array([0.4, -3.0])
    plan = causal_plan(model, jnp.zeros(1), cost, DT, 2, lo, hi)
    gap = plan_regret_bound(plan, model, jnp.zeros(1), cost, DT, lo, hi)
    assert (gap.bound, gap.modulus, gap.status) == (0.0, 0.0, "certified")


# ---- a problem posed in other units reads the same ----


def _certificate_in(
    result: Prescription, levers_in: float | tuple[float, float], state_in: float
) -> dict[str, object]:
    """What the certificate reads, each number brought back to the problem's own units: the cost
    and the regret in the cost's, which the prices keep, the tube in the state's, the schedule in
    each lever's."""
    plan = result.plan
    assert plan is not None
    certificate = result.certificate
    weight = plan.decision_weight()
    tube = plan.uncertainty_tube
    return {
        "certified_horizon": certificate.certified_horizon,
        "regret_status": certificate.regret_status,
        "weight": (weight.free, weight.weakly_active),
        "estimability": certificate.estimability,
        "task_cost": plan.task_cost / state_in**2,
        "regret_bound": certificate.regret_bound / state_in**2,
        "tube": None if tube is None else np.asarray(tube) / state_in,
        "prices": [[row.price / state_in**2 for row in rows] for rows in result.budget_prices()],
        "schedule": _schedule(result) / levers_in,
    }


def _same(other: dict[str, object], one: dict[str, object], scale: float) -> None:
    for key in ("certified_horizon", "regret_status", "weight", "estimability"):
        assert other[key] == one[key], key
    assert other["task_cost"] == pytest.approx(one["task_cost"], rel=1e-9, abs=0.0)
    # a bound at the plan's rounding reads rounding in any units: held to the cost's precision
    assert other["regret_bound"] == pytest.approx(one["regret_bound"], rel=1e-6, abs=1e-12 * scale)
    np.testing.assert_allclose(other["tube"], one["tube"], rtol=1e-9, atol=0.0)  # type: ignore[arg-type]
    np.testing.assert_allclose(other["prices"], one["prices"], rtol=1e-9, atol=0.0)  # type: ignore[arg-type]
    np.testing.assert_allclose(other["schedule"], one["schedule"], rtol=0.0, atol=1e-9)  # type: ignore[arg-type]


@pytest.mark.parametrize("units", [1e-3, 1e3])
@pytest.mark.parametrize("policy", POLICIES)
def test_the_certificate_reads_the_same_in_any_units_the_levers_share(
    policy: str, units: float
) -> None:
    """Both levers logged in ``units`` times their own, their boxes with them and their prices
    over its square. The tube's rate error, the channel's error per unit of a lever times the
    largest action, was that error times at least 1, a floor in the levers' units: in thousandths
    the tube ended at 2.2 where it ends at 0.0044, and certified no step of three. A row's push
    read as zero below 1e-3 of at least one unit of the cost per unit of an action: in thousands,
    the three rows that keep ``u2 = 2 u1``, pushing 8.9e-6 to 1.7e-5, read as met with no pull,
    and the decision weight counted three weakly active rows where it counts none. The schedule
    reads in the levers' units; the cost, the regret bound, the decision weight, the tube and the
    certified horizon read the same."""
    one = _certificate_in(_decide(policy), 1.0, 1.0)
    other = _certificate_in(_decide(policy, levers_in=(units, units)), units, 1.0)
    _same(other, one, _decide(policy).plan.task_cost)  # type: ignore[union-attr]


@pytest.mark.parametrize("levers_in", [(1e3, 1.0), (1.0, 1e-3), (1e-3, 1e3)])
@pytest.mark.parametrize("policy", POLICIES)
def test_the_certificate_reads_the_same_in_any_units_of_each_lever(
    policy: str, levers_in: tuple[float, float]
) -> None:
    """Each lever logged in units of its own, its box with it and its price over their square. The
    tube read one lever's units: with ``u1`` in units 1e3 times its own, ``free``'s tube ended at
    3.89 and ``together``'s at 1.40, where both end near 0.54, and neither certified a step, where
    both certify two; on ``state``'s log, whose ``u1`` the plan holds to the log's rule, with
    ``u1`` in thousandths and ``u2`` in thousands it ended at 4.2e3. Every number now reads the
    same: the schedule in each lever's units, the cost, the regret bound, the decision weight, the
    tube and the certified horizon in the problem's own."""
    one = _certificate_in(_decide(policy), 1.0, 1.0)
    other = _certificate_in(_decide(policy, levers_in=levers_in), levers_in, 1.0)
    _same(other, one, _decide(policy).plan.task_cost)  # type: ignore[union-attr]
    assert other["certified_horizon"] == 2


@pytest.mark.parametrize("levers_in", [(1e3, 1.0), (1.0, 1e-3), (1e-3, 1e3)])
@pytest.mark.parametrize(
    ("policy", "budget"), [("free", False), ("together", False), ("free", True)]
)
def test_the_regret_bound_reads_the_same_in_any_units_of_each_lever(
    policy: str, budget: bool, levers_in: tuple[float, float]
) -> None:
    """Each lever in units of its own, the other left as it was or in others. The bound's modulus
    was the Hessian's least eigenvalue, the cheapest lever's curvature in whatever units the
    levers came in, which charged both levers' moves at it: with ``u1`` in units 1e3 times its
    own, ``together``'s bound read 0.0576 where it read 0.0266, and ``free``'s 5.4e-12 where it
    read 1.8e-16. Each lever's moves are now measured in its own scale, and it reads 0.0214 and
    1.7e-16 in any units."""
    reference = _decide(policy, budget=budget)
    one = reference.certificate
    other = _decide(policy, levers_in=levers_in, budget=budget).certificate
    assert other.regret_status == one.regret_status == "diagnostic"
    assert reference.plan is not None
    assert other.regret_bound == pytest.approx(
        one.regret_bound, rel=1e-9, abs=1e-12 * reference.plan.task_cost
    )


@pytest.mark.parametrize("units", [1e-3, 1e3])
def test_the_certificate_reads_the_same_in_any_units_of_the_state(units: float) -> None:
    """The state logged in ``units`` times its own, the target, the start and the tolerance with
    it, the prices times its square. Every number reads the same in the problem's own units. In
    thousandths the cost is a millionth of its size, and the rows that keep ``u2 = 2 u1`` read as
    met with no pull, as with the levers in thousands."""
    one = _certificate_in(_decide("together"), 1.0, 1.0)
    other = _certificate_in(_decide("together", state_in=units), 1.0, units)
    _same(other, one, _decide("together").plan.task_cost)  # type: ignore[union-attr]


@pytest.mark.parametrize(("levers_in", "state_in"), [((1e-3, 1e-3), 1.0), ((1.0, 1.0), 1e3)])
def test_a_budget_s_price_reads_the_same_in_any_units(
    levers_in: tuple[float, float], state_in: float
) -> None:
    """A budget that binds, its weights in currency per unit of each lever: in other units of the
    levers or of the state its price, in the cost's units per currency unit, is the same."""
    reference = _decide("free", budget=True)
    one = _certificate_in(reference, 1.0, 1.0)
    assert one["prices"][0][0] > 0.0  # type: ignore[index]
    other = _certificate_in(
        _decide("free", levers_in=levers_in, state_in=state_in, budget=True),
        levers_in[0],
        state_in,
    )
    _same(other, one, reference.plan.task_cost)  # type: ignore[union-attr]


OTHER_UNITS = [((1e-3, 1e-3), 1.0), ((1e3, 1e3), 1.0), ((1.0, 1.0), 1e-3), ((1.0, 1.0), 1e3)]


@pytest.mark.parametrize(("levers_in", "state_in"), OTHER_UNITS)
def test_gamma_star_and_the_barrier_s_steps_read_the_same_in_any_units(
    levers_in: tuple[float, float], state_in: float
) -> None:
    """The state bounded to ``[0.8, 1.1]`` and audited at ``gamma = 1.5``. The identification radius
    was ``(gamma-1)/(gamma+1)`` times a gap of 1 in the channel's units, which are the state's per
    the levers': with the levers in thousands or the state in thousandths, gamma* read 1.0014
    where it reads 10.9 and the barrier cleared none of the two steps it clears; the other way,
    every level. The gap is now a share of the channel's own size."""
    one = _decide("free", floor=0.8).certificate
    other = _decide("free", levers_in=levers_in, state_in=state_in, floor=0.8).certificate
    assert one.gamma_star_status == "finite"
    assert one.barrier_certified_steps is not None
    assert 0 < one.barrier_certified_steps < HORIZON  # a real cut
    assert other.barrier_certified_steps == one.barrier_certified_steps
    assert other.gamma_star == pytest.approx(one.gamma_star, rel=1e-9, abs=0.0)


@pytest.mark.parametrize(("levers_in", "state_in"), OTHER_UNITS)
def test_a_held_barrier_clears_every_step_in_any_units(
    levers_in: tuple[float, float], state_in: float
) -> None:
    """The bounds held in the solve, at the radius the audit prices. In the channel's units that
    radius made the levers in thousands or the state in thousandths stand still and clear no step
    of three. The plans themselves are not compared: the rounds descend in the caller's units."""
    one = _decide("free", floor=0.8, hold=True).certificate
    other = _decide("free", levers_in=levers_in, state_in=state_in, floor=0.8, hold=True)
    assert one.barrier_certified_steps == HORIZON
    assert other.certificate.barrier_certified_steps == HORIZON
    assert float(np.max(np.abs(_schedule(other)))) > 0.0


EACH_LEVER = [(1e3, 1.0), (1.0, 1e-3), (1e-3, 1e3)]


@pytest.mark.parametrize("levers_in", EACH_LEVER)
@pytest.mark.parametrize(("policy", "floor"), [(p, 0.2) for p in POLICIES] + [("free", 0.8)])
def test_gamma_star_and_the_barrier_s_steps_read_the_same_in_any_units_of_each_lever(
    policy: str, floor: float, levers_in: tuple[float, float]
) -> None:
    """Each lever logged in units of its own, its box with it. The audit read the levers through
    one number, the largest action of either box, and one ball in the levers' units: bounded to
    ``[0.2, 1.1]``, ``free``'s and ``together``'s barrier cleared three steps of three, and none
    with ``u1`` in thousands or ``u2`` in thousandths, so the trusted steps fell from two to none;
    on ``state``'s log ``gamma*`` read 118 where it reads 6.17. It now reads each lever in units of
    its own largest action."""
    one = _decide(policy, floor=floor).certificate
    other = _decide(policy, levers_in=levers_in, floor=floor).certificate
    assert one.barrier_certified_steps is not None
    assert other.barrier_certified_steps == one.barrier_certified_steps
    assert other.trustworthy_steps == one.trustworthy_steps
    assert other.gamma_star_status == one.gamma_star_status
    assert other.gamma_star == pytest.approx(one.gamma_star, rel=1e-9, abs=0.0)


@pytest.mark.parametrize("levers_in", EACH_LEVER)
def test_a_held_barrier_clears_every_step_in_any_units_of_each_lever(
    levers_in: tuple[float, float],
) -> None:
    """``state``'s log, the bounds ``[0.2, 1.1]`` held in the solve. The held condition read the
    levers' units as one: with ``u1`` in thousands, or ``u2`` in thousandths, the held plan cleared
    no step of three. It now holds, and the audit reads, each lever in units of its own largest
    action, and every step clears."""
    one = _decide("state", floor=0.2, hold=True).certificate
    other = _decide("state", levers_in=levers_in, floor=0.2, hold=True).certificate
    assert one.barrier_certified_steps == HORIZON
    assert other.barrier_certified_steps == HORIZON


@pytest.mark.parametrize(("levers_in", "state_in"), [*OTHER_UNITS, ((1e-3, 1e3), 1e-3)])
@pytest.mark.parametrize("policy", ["free", "together"])
def test_the_plan_s_first_order_check_reads_the_same_in_any_units(
    policy: str, levers_in: tuple[float, float], state_in: float
) -> None:
    """Under a budget that binds, the plan's stationarity and its largest breach of its box and
    rows, each beside its tolerance, are read in the descent's own variables: each reads the same
    with the levers and the state in any units, and the certificate and the record carry the
    plan's."""
    one = _decide(policy, budget=True)
    other = _decide(policy, levers_in=levers_in, state_in=state_in, budget=True)
    for result in (one, other):
        plan, certificate = result.plan, result.certificate
        assert plan is not None
        assert (certificate.stationarity_status, certificate.violation_status) == ("within",) * 2
        read = (
            certificate.stationarity,
            certificate.stationarity_tolerance,
            certificate.constraint_violation,
            certificate.violation_tolerance,
        )
        assert read == (
            plan.stationarity,
            plan.stationarity_tolerance,
            plan.constraint_violation,
            plan.violation_tolerance,
        )
        record = result.to_json()["certificate"]
        assert (record["stationarity"], record["stationarity_status"]) == (read[0], "within")
        assert (record["constraint_violation"], record["violation_status"]) == (read[2], "within")
    first, second = one.certificate, other.certificate
    # converged to the descent's own resolution, so equal to rounding where the plan is rounding
    assert second.stationarity == pytest.approx(first.stationarity, rel=1e-6, abs=1e-13)
    assert second.stationarity_tolerance == first.stationarity_tolerance
    assert second.violation_tolerance == pytest.approx(first.violation_tolerance, rel=1e-9, abs=0.0)


# ---- the allocation planner: currency, outcome units, and a channel no plan can buy ----

PERIODS = 13
S_CHANNELS = (
    Channel(GeometricAdstock(0.3, length=6, normalized=True), Tanh(250.0), 1100.0),
    Channel(DelayedAdstock(0.6, 1.5, length=8, normalized=True), Hill(110.0, 2.5), 900.0),
    Channel(WeibullAdstock(1.2, 3.0, length=10, normalized=True), Hill(200.0, 3.0), 1400.0),
)
HISTORY = np.random.default_rng(3).gamma(4.0, 20.0, size=(30, 3))
LOWER = np.array([40.0, 25.0, 15.0])
UPPER = np.array([160.0, 100.0, 60.0])
BUDGET = PERIODS * 150.0


def _in(channel: Channel, currency: float, outcome: float) -> Channel:
    """``channel`` with spend in ``currency`` times its units and return in ``outcome`` times."""
    curve = channel.curve
    scaled = (
        Hill(currency * curve.scale, curve.slope)
        if isinstance(curve, Hill)
        else type(curve)(currency * curve.scale)  # type: ignore[call-arg, attr-defined]
    )
    return Channel(channel.kernel, scaled, outcome * channel.coefficient)


def _returned(channels, spend: np.ndarray) -> float:
    """What a plan returns, each channel run over history, plan and tail as one series."""
    total = 0.0
    for column, (channel, rate) in enumerate(zip(channels, spend, strict=True)):
        tail = np.zeros(channel.kernel.length - 1)
        series = np.concatenate([HISTORY[:, column], np.full(PERIODS, rate), tail])
        total += float(np.sum(np.asarray(channel(series))[HISTORY.shape[0] :]))
    return total


@functools.cache
def _allocated():
    return allocate(S_CHANNELS, BUDGET, PERIODS, lower=LOWER, upper=UPPER, history=HISTORY)


@pytest.mark.parametrize(
    ("currency", "outcome"), [(1e-3, 1.0), (1e3, 1.0), (1.0, 1e-3), (1.0, 1e3), (1e3, 1e-3)]
)
def test_a_plan_on_s_curves_reads_the_same_in_any_currency_and_outcome_units(
    currency: float, outcome: float
) -> None:
    """On S-shaped curves, searched in boxes: spend in ``currency`` times its units and return in
    ``outcome`` times. The spend reads in the currency, the worth and the bound in the outcome,
    the price in their ratio, and the search plans the same boxes. The worth is what the channels
    return on the spend."""
    plan = _allocated()
    assert plan.boxes > 1
    assert plan.worth == pytest.approx(_returned(S_CHANNELS, plan.spend), rel=1e-12, abs=0.0)
    moved = allocate(
        tuple(_in(channel, currency, outcome) for channel in S_CHANNELS),
        currency * BUDGET,
        PERIODS,
        lower=currency * LOWER,
        upper=currency * UPPER,
        history=currency * HISTORY,
    )
    np.testing.assert_allclose(moved.spend, currency * plan.spend, rtol=1e-12, atol=0.0)
    assert moved.worth == pytest.approx(outcome * plan.worth, rel=1e-12, abs=0.0)
    assert moved.bound == pytest.approx(outcome * plan.bound, rel=1e-12, abs=0.0)
    assert moved.price == pytest.approx(outcome / currency * plan.price, rel=1e-12, abs=0.0)
    assert (moved.boxes, moved.stopped) == (plan.boxes, plan.stopped)


def test_a_budget_for_a_return_on_s_curves_reads_the_same_in_any_units() -> None:
    plan = budget_for(
        S_CHANNELS,
        ReturnTarget(0.5 * _allocated().gain),
        PERIODS,
        lower=LOWER,
        upper=UPPER,
        history=HISTORY,
    )
    currency, outcome = 1e3, 1e-3
    moved = budget_for(
        tuple(_in(channel, currency, outcome) for channel in S_CHANNELS),
        ReturnTarget(outcome * 0.5 * _allocated().gain),
        PERIODS,
        lower=currency * LOWER,
        upper=currency * UPPER,
        history=currency * HISTORY,
    )
    assert moved.budget == pytest.approx(currency * plan.budget, rel=1e-12, abs=0.0)
    np.testing.assert_allclose(moved.spend, currency * plan.spend, rtol=1e-12, atol=0.0)
    assert moved.gain == pytest.approx(outcome * plan.gain, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("level", [0.0, 30.0])
def test_a_channel_no_plan_can_buy_moves_nothing(level: float) -> None:
    """A fourth channel, the steepest, whose box holds it at one rate: the others' spend, the
    worth less that channel's own fixed return, the bound and the price are the three-channel
    plan's."""
    extra = Channel(GeometricAdstock(0.5, length=4, normalized=True), Tanh(50.0), 3000.0)
    channels = (*S_CHANNELS, extra)
    history = np.hstack([HISTORY, np.full((HISTORY.shape[0], 1), level)])
    plan = _allocated()
    moved = allocate(
        channels,
        BUDGET + PERIODS * level,
        PERIODS,
        lower=np.append(LOWER, level),
        upper=np.append(UPPER, level),
        history=history,
    )
    np.testing.assert_allclose(moved.spend[:3], plan.spend, rtol=1e-12, atol=0.0)
    assert moved.spend[3] == level
    tail = np.zeros(extra.kernel.length - 1)
    own = float(
        np.sum(
            np.asarray(extra(np.concatenate([history[:, 3], np.full(PERIODS, level), tail])))[
                history.shape[0] :
            ]
        )
    )
    assert moved.worth - own == pytest.approx(plan.worth, rel=1e-12, abs=0.0)
    assert moved.bound - own == pytest.approx(plan.bound, rel=1e-12, abs=0.0)
    assert moved.price == pytest.approx(plan.price, rel=1e-12, abs=0.0)


@functools.cache
def _two_states(supply_in: float, wait_in: float) -> Prescription:
    """The quickstart's problem on sixty regions, ``supply`` and ``wait`` logged in ``supply_in``
    and ``wait_in`` times their units, the target, the bound and the price with them, and the
    tolerance in ``supply``'s."""
    rng = np.random.default_rng(0)
    names = ("region", "week", "supply", "wait", "incentive", "demand")
    rows: dict[str, list[float]] = {name: [] for name in names}
    for region in range(60):
        supply, wait = rng.normal(0.0, 0.2), rng.normal(0.0, 0.2)
        for week in range(12):
            demand = rng.normal(0.0, 1.0)
            incentive = 0.9 * demand + rng.normal(0.0, 0.5)
            for name, value in zip(
                names, (region, week, supply, wait, incentive, demand), strict=True
            ):
                rows[name].append(value)
            supply, wait = (
                supply
                + DT * (-0.6 * supply + 0.3 * wait + 0.8 * incentive + 1.5 * demand)
                + rng.normal(0.0, 0.01),
                wait + DT * (0.25 * wait - 0.4 * incentive) + rng.normal(0.0, 0.01),
            )
    columns = {name: np.asarray(values) for name, values in rows.items()}
    columns["supply"] = supply_in * columns["supply"]
    columns["wait"] = wait_in * columns["wait"]
    edges = [
        ("demand", "incentive"),
        ("demand", "supply"),
        ("incentive", "supply"),
        ("incentive", "wait"),
        ("wait", "supply"),
    ]
    return prescribe(
        Panel.from_frame(columns, unit="region", time="week", seed=0),
        adjustment=CausalGraph.from_edges(edges),
        levers=[Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05 * supply_in**2)],
        target=Target("supply", value=1.0 * supply_in),
        constraints=[Constraint("wait", hi=0.5 * wait_in)],
        horizon=5,
        dt=DT,
        tolerance=0.5 * supply_in,
    )


def test_the_tube_is_one_norm_over_states_in_the_units_they_share() -> None:
    """The tube is a norm over the states, and its rate error a root mean over them: it reads the
    same, by the factor, in any units the states share, and states in units unlike each other mix
    in it. ``tolerance`` is one number in the states' units, and a caller who logs each state in
    units of the error accepted on it gives each state its own scale."""
    one = _two_states(1.0, 1.0).plan
    shared = _two_states(1e3, 1e3).plan
    apart = _two_states(1.0, 1e3).plan
    assert one is not None
    assert shared is not None
    assert apart is not None
    assert one.uncertainty_tube is not None
    np.testing.assert_allclose(
        np.asarray(shared.uncertainty_tube) / 1e3,
        np.asarray(one.uncertainty_tube),
        rtol=1e-9,
        atol=0.0,
    )
    assert apart.uncertainty_tube is not None
    assert float(apart.uncertainty_tube[-1]) > 10.0 * float(one.uncertainty_tube[-1])
