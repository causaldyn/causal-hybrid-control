"""A receding horizon warm-started from its last plan: the answers of cold solves, for less work.

The reference is :func:`causal_plan` solved cold at every state the controller visits, which
``test_held_barrier.py`` holds to an exact QP oracle. What the warm start may change is the number
of descent steps, never the plan beyond the solver's tolerance.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from chc import (
    BarrierConstraint,
    CausalPlan,
    DampedOscillator,
    LinearConstraint,
    QuadraticCost,
    RecedingHorizon,
    SupportModel,
    causal_plan,
)
from chc.integrate import rk4_step
from chc.plan import _plan

DT = 0.1
OSCILLATOR = DampedOscillator(omega=1.0, zeta=0.1)
X0 = jnp.array([1.0, 0.0])
COST = QuadraticCost(
    Q=jnp.diag(jnp.array([1.0, 0.0])),
    R=jnp.array([[0.01]]),
    Qf=jnp.diag(jnp.array([10.0, 1.0])),
    x_target=jnp.zeros(2),
)
HORIZON = 12
U_MAX = 5.0
LOOP = 10
# At this floor the condition binds on every step of the first plan, so each step needs the rounds.
FLOOR, ALPHA = 0.3, 2.0
# Relative task-cost gap between two solves of one problem; measured at 1.0e-6 at worst.
GAP = 1e-4


def _velocity_floor(x: Array) -> Array:
    return x[1] + FLOOR


BARRIER = BarrierConstraint(_velocity_floor, alpha=ALPHA)
# ``_plan`` takes every argument by keyword and defaults none of them: these are causal_plan's own.
DEFAULTS: dict[str, Any] = {
    name: parameter.default
    for name, parameter in inspect.signature(causal_plan).parameters.items()
    if parameter.default is not inspect.Parameter.empty and name != "warm_start"
}


def _shifted(actions: Array) -> np.ndarray:
    held = np.asarray(actions)
    return np.concatenate([held[1:], held[-1:]])


class Loop(NamedTuple):
    states: list[Array]  # every state the controller planned from, and the one it ended in
    plans: list[CausalPlan]  # the controller's
    cold: list[CausalPlan]  # causal_plan from zeros at the same states
    actions_only: list[CausalPlan]  # causal_plan from the controller's own warm actions, no more


@pytest.fixture(scope="module")
def loop() -> Loop:
    controller = RecedingHorizon(OSCILLATOR, COST, DT, HORIZON, -U_MAX, U_MAX, barrier=BARRIER)
    states, plans, cold, actions_only = [X0], [], [], []
    for _ in range(LOOP):
        x = states[-1]
        plan = controller.step(x)
        warm = None if not plans else _shifted(plans[-1].actions)
        cold.append(causal_plan(OSCILLATOR, x, COST, DT, HORIZON, -U_MAX, U_MAX, barrier=BARRIER))
        actions_only.append(
            causal_plan(
                OSCILLATOR, x, COST, DT, HORIZON, -U_MAX, U_MAX, barrier=BARRIER, warm_start=warm
            )
        )
        plans.append(plan)
        states.append(rk4_step(OSCILLATOR, 0.0, x, plan.actions[0], DT))
    return Loop(states, plans, cold, actions_only)


def test_the_controller_takes_every_argument_causal_plan_does() -> None:
    """A new ``causal_plan`` argument the controller does not carry would be silently unusable."""
    planned = {
        name: parameter.default
        for name, parameter in inspect.signature(causal_plan).parameters.items()
        if name not in ("x0", "warm_start")
    }
    carried = {
        field.name: field.default for field in dataclasses.fields(RecedingHorizon) if field.init
    }
    assert carried.keys() == planned.keys()
    assert {
        name: default for name, default in carried.items() if default is not dataclasses.MISSING
    } == {
        name: default for name, default in planned.items() if default is not inspect.Parameter.empty
    }


def test_each_step_is_the_plan_causal_plan_makes_from_that_state(loop: Loop) -> None:
    gaps = [
        abs(plan.task_cost - cold.task_cost) / abs(cold.task_cost)
        for plan, cold in zip(loop.plans, loop.cold, strict=True)
    ]
    assert max(gaps) <= GAP
    assert all(plan.solver_status == "converged" for plan in loop.plans)


def test_the_warm_start_saves_descent_steps_and_the_multipliers_save_most(loop: Loop) -> None:
    """Measured at 32 083 steps warm, 48 286 from the warm actions alone, 48 240 cold."""
    warm = sum(plan.solver_iterations for plan in loop.plans)
    actions_only = sum(plan.solver_iterations for plan in loop.actions_only)
    cold = sum(plan.solver_iterations for plan in loop.cold)
    assert warm < 0.8 * cold
    assert warm < 0.8 * actions_only


def test_the_closed_loop_stays_safe_and_every_applied_action_is_certified(loop: Loop) -> None:
    assert min(float(_velocity_floor(x)) for x in loop.states) > 0.0
    for plan in loop.plans:
        assert plan.safety is not None
        assert bool(plan.safety.planned_certified[0])


def _rounds(caplog: pytest.LogCaptureFixture) -> list[int]:
    return [
        int(getattr(record, "rounds", -1))
        for record in caplog.records
        if getattr(record, "chc_event", None) == "barrier"
    ]


def test_handed_its_own_multipliers_back_the_rounds_finish_sooner(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The multipliers go in and come out of the solve: seeded with its own, it needs fewer rounds.

    Not one round, because the penalty starts afresh and the inner descents are inexact: measured
    at 6 rounds against 13 from zero.
    """
    arguments = {**DEFAULTS, "barrier": BARRIER, "warm_start": None}
    with caplog.at_level(logging.INFO, logger="chc.plan"):
        plan, multipliers = _plan(
            OSCILLATOR, X0, COST, DT, HORIZON, -U_MAX, U_MAX, multipliers=None, **arguments
        )
        again, _ = _plan(
            OSCILLATOR, X0, COST, DT, HORIZON, -U_MAX, U_MAX, multipliers=multipliers, **arguments
        )
    cold_rounds, warm_rounds = _rounds(caplog)
    assert multipliers is not None
    assert int(np.sum(multipliers > 0.0)) > 0
    assert warm_rounds < cold_rounds
    assert abs(again.task_cost - plan.task_cost) <= GAP * abs(plan.task_cost)


def test_a_step_allowed_no_descent_returns_the_last_plan_moved_one_step_on() -> None:
    controller = RecedingHorizon(OSCILLATOR, COST, DT, HORIZON, -U_MAX, U_MAX)
    first = controller.step(X0)
    controller.steps = 0
    second = controller.step(rk4_step(OSCILLATOR, 0.0, X0, first.actions[0], DT))
    np.testing.assert_array_equal(np.asarray(second.actions), _shifted(first.actions))


def test_a_step_hands_the_next_its_plan_and_multipliers_moved_one_step_on() -> None:
    """Bit for bit: the second step is ``_plan`` from the first's actions and multipliers, shifted.

    Solves land on the same plan from nearby starts, so only the exact starts can tell a shifted
    hand-over from an unshifted one.
    """
    controller = RecedingHorizon(OSCILLATOR, COST, DT, HORIZON, -U_MAX, U_MAX, barrier=BARRIER)
    first = controller.step(X0)
    x = rk4_step(OSCILLATOR, 0.0, X0, first.actions[0], DT)
    second = controller.step(x)
    arguments = {**DEFAULTS, "barrier": BARRIER}
    _, multipliers = _plan(
        OSCILLATOR,
        X0,
        COST,
        DT,
        HORIZON,
        -U_MAX,
        U_MAX,
        warm_start=None,
        multipliers=None,
        **arguments,
    )
    assert multipliers is not None
    expected, _ = _plan(
        OSCILLATOR,
        x,
        COST,
        DT,
        HORIZON,
        -U_MAX,
        U_MAX,
        warm_start=_shifted(first.actions),
        multipliers=_shifted(multipliers),
        **arguments,
    )
    np.testing.assert_array_equal(np.asarray(second.actions), np.asarray(expected.actions))
    assert second.solver_iterations == expected.solver_iterations


def test_a_slack_barrier_hands_on_no_multipliers() -> None:
    """A step the barrier-free plan already makes safe passes zeros on, whatever it was handed."""
    arguments = {**DEFAULTS, "barrier": BarrierConstraint(lambda x: x[1] + 10.0, alpha=ALPHA)}
    plan, multipliers = _plan(
        OSCILLATOR,
        X0,
        COST,
        DT,
        HORIZON,
        -U_MAX,
        U_MAX,
        warm_start=None,
        multipliers=np.ones(HORIZON),
        **arguments,
    )
    assert plan.safety is not None
    assert plan.safety.certified_steps == HORIZON
    np.testing.assert_array_equal(multipliers, np.zeros(HORIZON))


def test_a_warm_start_moves_the_start_not_the_answer() -> None:
    cold = causal_plan(OSCILLATOR, X0, COST, DT, HORIZON, -U_MAX, U_MAX)
    start = np.random.default_rng(0).uniform(-U_MAX, U_MAX, (HORIZON, 1))
    elsewhere = causal_plan(OSCILLATOR, X0, COST, DT, HORIZON, -U_MAX, U_MAX, warm_start=start)
    at_the_answer = causal_plan(
        OSCILLATOR, X0, COST, DT, HORIZON, -U_MAX, U_MAX, warm_start=cold.actions
    )
    assert cold.solver_status == elsewhere.solver_status == "converged"
    assert abs(elsewhere.task_cost - cold.task_cost) <= GAP * abs(cold.task_cost)
    assert at_the_answer.solver_iterations < cold.solver_iterations // 10


def test_a_warm_start_need_not_be_admissible() -> None:
    budget = LinearConstraint(np.ones((1, HORIZON)), -np.inf, 3.0)
    plan = causal_plan(
        OSCILLATOR,
        X0,
        COST,
        DT,
        HORIZON,
        -U_MAX,
        U_MAX,
        steps=0,
        constraints=(budget,),
        warm_start=np.full((HORIZON, 1), 2.0 * U_MAX),
    )
    actions = np.asarray(plan.actions)
    assert np.all(np.abs(actions) <= U_MAX)
    assert float(np.sum(actions)) <= 3.0 + 1e-9


@pytest.mark.parametrize(
    "start",
    [
        np.zeros((HORIZON - 1, 1)),
        np.zeros(HORIZON),
        np.where(np.arange(HORIZON)[:, None] == 3, np.nan, 0.0),
        np.full((HORIZON, 1), np.inf),
    ],
    ids=["short", "flat", "nan", "inf"],
)
def test_a_warm_start_that_is_not_a_finite_plan_is_refused(start: np.ndarray) -> None:
    with pytest.raises(ValueError, match="warm_start"):
        causal_plan(OSCILLATOR, X0, COST, DT, HORIZON, -U_MAX, U_MAX, warm_start=start)


@contextmanager
def _compilations() -> Iterator[list[str]]:
    compiled: list[str] = []

    class Record(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.getMessage().startswith("Compiling "):
                compiled.append(record.getMessage().split(" ")[1])

    handler, logger = Record(), logging.getLogger("jax")
    logger.addHandler(handler)
    try:
        with jax.log_compiles(True):
            yield compiled
    finally:
        logger.removeHandler(handler)


SUPPORT = SupportModel.fit(
    jax.random.normal(jax.random.PRNGKey(0), (200, 2)),
    jax.random.normal(jax.random.PRNGKey(1), (200, 1)),
)


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"barrier": BARRIER, "lipschitz": 0.5, "model_error": 0.01, "tolerance": 1.0},
        {
            "barrier": BARRIER,
            "support": SUPPORT,
            "lam_supp": 0.1,
            "constraints": (
                LinearConstraint.rate_limit(HORIZON, [1.0]),
                LinearConstraint(np.ones((1, HORIZON)), -np.inf, 30.0),
            ),
        },
    ],
    ids=["free", "barrier-and-tube", "barrier-support-and-rows"],
)
def test_a_step_compiles_nothing_once_its_programs_exist(arguments: dict[str, Any]) -> None:
    """Every program the first step compiled is reused: a replan costs no compilation.

    The caches are cleared first, so nothing an earlier test compiled can stand in for the first
    step. That step runs every path here -- the barrier binds from ``X0`` -- so the steps after it
    have nothing left to compile; what would compile again is a new shape, dtype or function.
    """
    jax.clear_caches()
    controller = RecedingHorizon(OSCILLATOR, COST, DT, HORIZON, -U_MAX, U_MAX, **arguments)
    plan = controller.step(X0)
    x = rk4_step(OSCILLATOR, 0.0, X0, plan.actions[0], DT)
    with _compilations() as compiled:
        for _ in range(3):
            plan = controller.step(x)
            x = rk4_step(OSCILLATOR, 0.0, x, plan.actions[0], DT)
    assert compiled == []
