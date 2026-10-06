"""Model predictive control: receding-horizon optimal control with warm starts.

At each step MPC solves a finite-horizon OC problem from the current measured state (against the
*model*), applies only the first control, advances the *plant*, and re-solves. Planning and
reality are separate objects (``model`` vs ``plant``), so the offline/confounded setting — plan
with the learned hybrid model, act on the true system — needs no change to this loop.

:func:`mpc_control` runs that loop against a simulated plant. :class:`RecedingHorizon` is its
online form -- the caller owns the plant and the clock -- and each of its steps is a whole
:func:`chc.plan.causal_plan`, certificate included. :class:`PeriodBudget` caps what that loop
spends per period, which no single window can see.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from chc.control import Bound, LinearConstraint, broadcast_box, projected_gradient_control
from chc.cost import QuadraticCost
from chc.dynamics import Dynamics
from chc.integrate import rk4_step
from chc.plan import BarrierConstraint, CausalPlan, _plan
from chc.support import PenaltyModel, SupportModel

_log = logging.getLogger(__name__)


def mpc_control(
    model: Dynamics,
    x0: Array,
    cost: QuadraticCost,
    dt: float,
    horizon: int,
    u_lo: Bound,
    u_hi: Bound,
    n_steps: int,
    *,
    plant: Dynamics | None = None,
    inner_steps: int = 40,
    warm_start: bool = True,
) -> tuple[Array, Array]:
    """Run closed-loop MPC for ``n_steps``; return the realised trajectory and applied controls.

    Unlike the one-shot solvers, ``inner_steps`` here is a **per-decision latency budget** and is
    deliberately far below their default: a warm start hands each replan a nearly-optimal iterate,
    so the truncation buys latency rather than hiding an unconverged answer. Priced rather than
    assumed -- against the same loop run to convergence over 25 replans, 40 steps costs 0.3% of
    closed-loop cost on a known plant and 0.4% with an MLP residual, for 1.5x and 2.4x less time.
    Raise it if the loop is offline; the solve stops on its own once the line search fails.

    The loop keeps one clock: step ``k`` applies its action to the plant at ``t = k * dt``, and the
    plan it solves there reads the model from ``t = k * dt`` on. Autonomous plants never notice; a
    plant with a time-varying term is planned and stepped on the time it is at.

    Args:
        model: dynamics used for planning (the controller's belief).
        plant: true dynamics the control is applied to (defaults to ``model``).

    Returns:
        ``(xs, us)`` with ``xs`` of shape ``(n_steps + 1, n)`` and ``us`` of shape ``(n_steps, m)``.
    """
    _refuse_moving_target(cost, "mpc_control")
    plant = model if plant is None else plant
    control_dim = cost.R.shape[0]
    guess = jnp.zeros((horizon, control_dim))
    x = x0
    states = [x0]
    applied: list[Array] = []

    for k in range(n_steps):
        t = k * dt
        us_opt, _ = projected_gradient_control(
            _Shifted(model, jnp.asarray(t)), x, guess, dt, cost, u_lo, u_hi, steps=inner_steps
        )
        u0 = us_opt[0]
        applied.append(u0)
        x = rk4_step(plant, t, x, u0, dt)
        states.append(x)
        guess = (
            jnp.concatenate([us_opt[1:], us_opt[-1:]], axis=0)
            if warm_start
            else jnp.zeros((horizon, control_dim))
        )

    return jnp.stack(states), jnp.stack(applied)


class _Shifted(eqx.Module):
    """``dynamics`` on the loop's clock: time ``t`` inside a window is ``t0 + t`` outside it.

    ``t0`` is an array leaf, so a new window start is a new value, not a new program.
    """

    dynamics: Dynamics
    t0: Array

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return self.dynamics(self.t0 + t, x, u)


def _refuse_moving_target(cost: QuadraticCost, where: str) -> None:
    if cost.x_target.ndim >= 2:
        raise ValueError(
            f"{where} replans with one cost, so a per-state x_target would hold the same window at "
            "every step instead of moving with the loop; plan each step with causal_plan and the "
            "rows for that step"
        )


def _shift(sequence: NDArray[np.float64]) -> NDArray[np.float64]:
    """One step on: drop the first entry, repeat the last."""
    return np.concatenate([sequence[1:], sequence[-1:]])


@dataclass(frozen=True)
class PeriodBudget:
    """At most ``amount`` spent per period of ``period`` steps; a step spends ``weights @ u``.

    Periods start at ``t = start`` on the loop's clock and every ``period * dt`` after. Unspent
    budget does not carry over: each period starts with ``amount``.

    Raises:
        ValueError: on ``weights`` that are not a finite 1-D vector with a non-zero entry, a
            negative or non-finite ``amount``, a ``period`` below one step, or a non-finite
            ``start``.
    """

    weights: NDArray[np.float64]  # (m,) the spend of one unit of each lever
    amount: float
    period: int
    start: float = 0.0

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        if weights.ndim != 1 or not np.isfinite(weights).all():
            raise ValueError(f"weights must be a finite (m,) vector, got shape {weights.shape}")
        if not weights.any():
            raise ValueError("every weight is zero, so the budget caps nothing")
        if not (math.isfinite(self.amount) and self.amount >= 0.0):
            raise ValueError(f"amount must be finite and non-negative, got {self.amount}")
        if self.period < 1:
            raise ValueError(f"period must be at least one step, got {self.period}")
        if not math.isfinite(self.start):
            raise ValueError(f"start must be finite, got {self.start}")
        object.__setattr__(self, "weights", weights)


@dataclass(eq=False)
class RecedingHorizon:
    """:func:`chc.plan.causal_plan` from each measured state, warm-started from the last plan.

    ::

        controller = RecedingHorizon(model, cost, dt=0.1, horizon=20, u_lo=-5.0, u_hi=5.0,
                                     barrier=BarrierConstraint(h, alpha=2.0))
        while running:
            plan = controller.step(measure())
            apply(plan.actions[0])

    :meth:`step` returns the whole :class:`~chc.plan.CausalPlan`, so the certificate reaches the
    caller with the action it covers; read ``plan.safety`` and ``plan.solver_status`` before
    applying ``plan.actions[0]``. The fields are :func:`~chc.plan.causal_plan`'s arguments, with
    the same meaning and defaults.

    Each step starts the descent from the last plan shifted one step, its final action repeated,
    and a held barrier's rounds from the last multipliers shifted the same way, under a penalty
    chosen afresh. That moves where the solves start, not what they minimise.

    **Compiled once.** Each program compiles the first time a step needs it -- the barrier's rounds
    the first time the barrier binds -- and is reused after, so steps stop compiling. What compiles
    again is a new shape, dtype or Python function: build the barrier's function once, outside the
    loop, as this object does by holding it; a fresh ``lambda`` per step compiles the descent per
    step. Across processes, set ``jax_compilation_cache_dir`` and lower
    ``jax_persistent_cache_min_compile_time_secs`` to 0: at its default of one second JAX wrote
    none of a first step's programs on the tests' oscillator, and at 0 a second process loaded
    every one of them from disk.

    Each step is one :func:`~chc.plan.causal_plan`, so nothing outside the horizon carries over:
    ``constraints`` bind each horizon afresh -- a budget row caps every window, not the run's
    total, and a rate limit does not reach back to the action already applied -- and ``steps``
    caps each descent, of which a held barrier runs up to ``1 + _BARRIER_ROUNDS`` a step. For the
    same reason a per-state ``x_target`` is refused: every step would read the same window.

    **A budget per period** is ``budget``, and the ledger is the caller's: :meth:`step` takes what
    the current period has spent so far, as measured. Each period the window touches gets one row
    over its steps in the window: what is left of the current period, pro rata to the share of its
    remaining steps the window holds, and a later period's ``amount``, pro rata the same way. So a
    ``horizon`` of at least ``period`` sees the rest of every period it plans in, and its row is
    exactly what is left; a shorter one gets a flat share of it. How far past the period's end to
    look is the plant's to say. Against the plan made for the whole run at once, on the ledger lab's
    plant, whose memory is a few steps, a window of one day lost 0.4-0.8 % of the budget's value
    over three days and one of a sixth of a day 4.6-8.2 %; on the marketing-mix plant, whose slowest
    adstock keeps 37 % of a spend four weeks on, a window of two four-week periods lost 0.24-0.67 %
    over six and one of one period 0.45-1.25 % (``docs/adr/0008-a-budget-per-period.md``). No step's
    row lets the period overspend, except where the box forces a spend; then the row is the least
    the box allows, and a warning says by how much the period goes over
    (``chc_event="budget_overrun"``).

    One controller per control loop: :meth:`step` updates the warm start in place, unguarded. A
    new controller over the same model, cost and barrier is a cold start that reuses the compiled
    programs.
    """

    model: Dynamics
    cost: QuadraticCost
    dt: float
    horizon: int
    u_lo: Bound
    u_hi: Bound
    support: SupportModel | None = None
    lam_supp: float = 0.0
    uncertainty: PenaltyModel | None = None
    lam_unc: float = 0.0
    lipschitz: float = 0.0
    model_error: float = 0.0
    tolerance: float = float("inf")
    steps: int = 10_000
    constraints: Sequence[LinearConstraint] = ()
    barrier: BarrierConstraint | None = None
    budget: PeriodBudget | None = None
    _warm: tuple[NDArray[np.float64], NDArray[np.float64] | None] | None = field(
        default=None, init=False, repr=False
    )

    def step(self, x: Array, t: float = 0.0, spent: float | None = None) -> CausalPlan:
        """Plan from the measured state ``x``, and keep the plan as the next step's start.

        ``t`` is the loop's clock at ``x``, and the window is planned from it: step ``k`` of the
        plan reads the model at ``t + k * dt``. An autonomous model ignores it; a model with a
        time-varying term needs it, since the default plans every window as if it started at 0.

        ``spent`` is what the period ``t`` falls in has spent so far, as measured, and a budgeted
        step needs it: the budget's rows are set from it and from where ``t`` falls in the period.

        Raises:
            ValueError: on a ``t`` that is not finite; on ``spent`` without a budget, or a budget
                without ``spent``; on a budgeted step whose ``t`` is not on the ``dt`` grid from the
                budget's start.
        """
        _refuse_moving_target(self.cost, "RecedingHorizon")
        if not np.isfinite(t):
            raise ValueError(f"t is the loop's clock and must be finite, got {t}")
        constraints = tuple(self.constraints)
        if self.budget is not None:
            constraints = (*constraints, self._budget_rows(self.budget, t, spent))
        elif spent is not None:
            raise ValueError("spent is a budget's ledger, and this controller has no budget")
        actions, multipliers = (None, None) if self._warm is None else self._warm
        plan, multipliers = _plan(
            _Shifted(self.model, jnp.asarray(t)),
            x,
            self.cost,
            self.dt,
            self.horizon,
            self.u_lo,
            self.u_hi,
            support=self.support,
            lam_supp=self.lam_supp,
            uncertainty=self.uncertainty,
            lam_unc=self.lam_unc,
            lipschitz=self.lipschitz,
            model_error=self.model_error,
            tolerance=self.tolerance,
            steps=self.steps,
            constraints=constraints,
            barrier=self.barrier,
            warm_start=actions,
            multipliers=multipliers,
        )
        self._warm = (
            _shift(np.asarray(plan.actions, dtype=np.float64)),
            None if multipliers is None else _shift(multipliers),
        )
        return plan

    def _budget_rows(self, budget: PeriodBudget, t: float, spent: float | None) -> LinearConstraint:
        if spent is None or not math.isfinite(spent):
            raise ValueError(
                "a budgeted step needs spent: what the period t falls in has spent so far"
            )
        return _period_rows(
            budget,
            t,
            spent,
            dt=self.dt,
            horizon=self.horizon,
            u_lo=self.u_lo,
            u_hi=self.u_hi,
            levers=self.cost.R.shape[0],
        )


def _period_rows(
    budget: PeriodBudget,
    t: float,
    spent: float,
    *,
    dt: float,
    horizon: int,
    u_lo: Bound,
    u_hi: Bound,
    levers: int,
) -> LinearConstraint:
    """One row per period a plan of ``horizon`` steps from ``t`` touches, over that period's steps
    in the plan, when the period ``t`` falls in has spent ``spent``."""
    offset = (t - budget.start) / dt
    index = round(offset)
    if abs(offset - index) > 1e-6:
        raise ValueError(
            f"t = {t} is {offset:.6g} steps from the budget's start; a budgeted step reads its "
            "place in the period off the clock, so t must be on the dt grid"
        )
    position = index % budget.period
    if budget.weights.shape[0] != levers:
        raise ValueError(
            f"the budget weighs {budget.weights.shape[0]} levers, the plan has {levers}"
        )
    shape = (horizon, levers)
    dtype = jnp.result_type(float)
    lo = np.asarray(broadcast_box(u_lo, shape, "u_lo", dtype), dtype=np.float64)
    hi = np.asarray(broadcast_box(u_hi, shape, "u_hi", dtype), dtype=np.float64)
    spends = budget.weights != 0.0  # a free lever's unbounded side must not read as 0 * inf
    weights = budget.weights[spends]
    least = np.minimum(weights * lo[:, spends], weights * hi[:, spends]).sum(axis=1)
    left = budget.amount - spent
    # A row the solve holds to rounding comes back that far over; that is not an overrun.
    rounding = 1e3 * float(jnp.finfo(dtype).eps) * max(1.0, budget.amount)
    matrix, upper, first = [], [], 0
    while first < horizon:
        share = budget.period - position if first == 0 else budget.period
        count = min(share, horizon - first)
        bound = (left if first == 0 else budget.amount) * count / share
        floor = float(least[first : first + count].sum())
        if bound < floor:
            if first == 0 and floor - bound > rounding:
                _log.warning(
                    "this period's %d steps in the window spend at least %.6g, and their "
                    "row allowed %.6g of the %.6g left, so the plan spends the least the box "
                    "allows",
                    count,
                    floor,
                    bound,
                    left,
                    extra={
                        "chc_event": "budget_overrun",
                        "position": position,
                        "left": left,
                        "floor": floor,
                        "allowed": bound,
                    },
                )
            bound = floor
        row = np.zeros(shape)
        row[first : first + count] = budget.weights
        matrix.append(row.ravel())
        upper.append(bound)
        first += count
    _log.debug(
        "budget rows for step %d of the period: %s",
        position,
        [round(value, 9) for value in upper],
        extra={"chc_event": "budget", "position": position, "left": left, "bounds": upper},
    )
    return LinearConstraint(np.stack(matrix), np.full(len(upper), -np.inf), np.asarray(upper))
