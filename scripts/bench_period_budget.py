"""The numbers behind docs/adr/0008-a-budget-per-period.md. Costs and spends only, no wall time.

    lab  the ledger lab's plant, ``x' = -0.5 x + u`` tracked to 2 over 48-step days, with a daily
         budget of a half and a tenth of what an unbudgeted day spends.
    mmm  the marketing-mix case's true plant over four-week periods, with a budget per period of
         a half and a quarter of what the unbudgeted twelve-week plan spends a period, planned on
         the true plant and on the adjusted arm's fit.

Every loop acts on the true plant and is scored against the plan made for the whole run at once,
one budget row per period. The arms: the budget row a caller could already pass in
``constraints``, which caps each window; ``PeriodBudget`` at three window lengths; and, in the lab,
the rejected design, a window that shrinks to the end of its period. The same loops without a
budget are the control: against the unbudgeted plan they lose only what a receding horizon loses
by not knowing where the run ends, and by planning on a fitted model.

Each case runs three periods and then six, at the same budget. A regret that stays put when the
run doubles is paid once, at its end, by a window that plans past it; one that doubles is paid
every period. Regret is in the cost's units and as a share of the budget's value, what the
whole-run plan gains over spending the box's floor throughout, and only for loops that kept the
budget: a loop that overspends is not scored against a plan that did not.

Run: uv run python scripts/bench_period_budget.py {lab,mmm} > out.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from chc import LinearConstraint, QuadraticCost, RecedingHorizon, causal_plan  # noqa: E402
from chc.cost import total_cost  # noqa: E402
from chc.dynamics import Dynamics, HybridDynamics, LinearDynamics  # noqa: E402
from chc.integrate import rk4_step  # noqa: E402
from chc.mmm import MarketingMixSystem, adstock_dynamics, run_marketing_mix  # noqa: E402
from chc.mpc import PeriodBudget  # noqa: E402

Step = Callable[[jax.Array, int, list[np.ndarray]], np.ndarray]


@dataclass(frozen=True)
class Case:
    truth: Dynamics
    x0: jax.Array
    cost: QuadraticCost
    dt: float
    box: tuple[float, float]
    period: int
    periods: int

    @property
    def steps(self) -> int:
        return self.period * self.periods

    @property
    def levers(self) -> int:
        return int(self.cost.R.shape[0])

    def audit(self, actions: np.ndarray) -> float:
        u = jnp.asarray(actions).reshape(self.steps, self.levers)
        return float(total_cost(self.truth, self.x0, u, self.dt, self.cost))

    def best(self, constraints: tuple[LinearConstraint, ...] = ()) -> float:
        plan = causal_plan(
            self.truth, self.x0, self.cost, self.dt, self.steps, *self.box, constraints=constraints
        )
        return self.audit(np.asarray(plan.actions))

    def loop(self, act: Step) -> np.ndarray:
        x, applied = self.x0, []
        for k in range(self.steps):
            applied.append(act(x, k, applied))
            x = rk4_step(self.truth, k * self.dt, x, jnp.asarray(applied[-1]), self.dt)
        return np.asarray(applied)

    def cap(self, steps: int, amount: float) -> LinearConstraint:
        return LinearConstraint(
            np.ones((1, steps * self.levers)), np.full(1, -np.inf), np.full(1, amount)
        )


def _plain(controller: RecedingHorizon, dt: float) -> Step:
    return lambda x, k, applied: np.asarray(controller.step(x, t=k * dt).actions[0])


def _ledger(controller: RecedingHorizon, dt: float, budget: PeriodBudget) -> Step:
    def act(x: jax.Array, k: int, applied: list[np.ndarray]) -> np.ndarray:
        opened = k - k % budget.period
        spent = float(np.sum(np.asarray(applied[opened:]) @ budget.weights)) if k > opened else 0.0
        return np.asarray(controller.step(x, t=k * dt, spent=spent).actions[0])

    return act


def _shrinking(case: Case, model: Dynamics, amount: float) -> Step:
    """Plan to the end of the period with what is left of it, and nothing past it."""

    def act(x: jax.Array, k: int, applied: list[np.ndarray]) -> np.ndarray:
        position = k % case.period
        left = amount - float(np.sum(applied[k - position :])) if position else amount
        steps = case.period - position
        row = case.cap(steps, max(left, steps * case.levers * case.box[0]))
        plan = causal_plan(model, x, case.cost, case.dt, steps, *case.box, constraints=(row,))
        return np.asarray(plan.actions[0])

    return act


def _run(
    case: Case,
    models: dict[str, Dynamics],
    windows: tuple[int, ...],
    free_spend: float,
    shares: tuple[float, ...],
    cap_window: int,
    shrinking: bool,
) -> dict[str, object]:
    floor = case.audit(np.full((case.steps, case.levers), case.box[0]))

    def score(applied: np.ndarray, best: float, amount: float | None) -> dict[str, object]:
        row: dict[str, object] = {}
        if amount is not None:
            spend = applied.reshape(case.periods, -1).sum(axis=1) / amount
            row["spend_over_budget"] = spend.tolist()
            if np.any(spend > 1.0 + 1e-6):
                return row
        regret = case.audit(applied) - best
        return row | {"regret": regret, "regret_share": regret / (floor - best)}

    best = case.best()
    control = {
        f"{label}_{window}": score(
            case.loop(
                _plain(RecedingHorizon(model, case.cost, case.dt, window, *case.box), case.dt)
            ),
            best,
            None,
        )
        for label, model in models.items()
        for window in windows
    }
    out: dict[str, object] = {
        "floor_cost": floor,
        "free_period_spend": free_spend,
        "unbudgeted": {"oracle_cost": best, **control},
    }
    for share in shares:
        amount = share * free_spend
        rows = LinearConstraint(
            np.kron(np.eye(case.periods), np.ones((1, case.period * case.levers))).astype(float),
            np.full(case.periods, -np.inf),
            np.full(case.periods, amount),
        )
        best = case.best((rows,))
        cap = RecedingHorizon(
            case.truth,
            case.cost,
            case.dt,
            cap_window,
            *case.box,
            constraints=(case.cap(cap_window, amount),),
        )
        arms = {f"window_row_{cap_window}": score(case.loop(_plain(cap, case.dt)), best, amount)}
        if shrinking:
            arms["shrinking"] = score(case.loop(_shrinking(case, case.truth, amount)), best, amount)
        for label, model in models.items():
            for window in windows:
                budget = PeriodBudget(np.ones(case.levers), amount, case.period)
                controller = RecedingHorizon(
                    model, case.cost, case.dt, window, *case.box, budget=budget
                )
                applied = case.loop(_ledger(controller, case.dt, budget))
                arms[f"{label}_{window}"] = score(applied, best, amount)
        out[f"budget_{share}"] = {"amount": amount, "oracle_cost": best, **arms}
    return out


def lab() -> dict[str, object]:
    model = LinearDynamics(jnp.array([[-0.5]]), jnp.array([[1.0]]))
    cost = QuadraticCost(
        Q=jnp.eye(1), R=jnp.array([[0.05]]), Qf=jnp.eye(1), x_target=jnp.array([2.0])
    )
    case = Case(model, jnp.array([1.0]), cost, dt=0.5, box=(0.0, 50.0), period=48, periods=3)
    day = causal_plan(model, case.x0, cost, case.dt, case.period, *case.box)
    return {
        f"{periods}_periods": _run(
            replace(case, periods=periods),
            {"true": model},
            windows=(8, 48, 96),
            free_spend=float(jnp.sum(day.actions)),
            shares=(0.5, 0.1),
            cap_window=8,
            shrinking=True,
        )
        for periods in (3, 6)
    }


def mmm() -> dict[str, object]:
    system = MarketingMixSystem()
    levers = len(system.channels)
    weight = jnp.diag(jnp.array([1.0] + [0.0] * levers))
    cost = QuadraticCost(
        Q=weight, R=0.08 * jnp.eye(levers), Qf=weight, x_target=jnp.array([8.0] + [0.0] * levers)
    )
    case = Case(
        system.plant(0.0),
        jnp.array([system.base] + [0.0] * levers),
        cost,
        dt=1.0,
        box=(system.spend_floor, system.spend_ceiling),
        period=4,
        periods=3,
    )
    adjusted = next(arm for arm in run_marketing_mix().arms if arm.name == "adjusted")
    assert adjusted.prescription is not None
    fitted = HybridDynamics(
        known=adstock_dynamics(jnp.array(system.theta)),
        residual=adjusted.prescription.model_fit.residual,
    )
    run = causal_plan(case.truth, case.x0, cost, case.dt, case.steps, *case.box)
    return {
        f"{periods}_periods": _run(
            replace(case, periods=periods),
            {"true": case.truth, "fitted": fitted},
            windows=(4, 8, 12),
            free_spend=float(jnp.sum(run.actions)) / case.periods,
            shares=(0.5, 0.25),
            cap_window=4,
            shrinking=False,
        )
        for periods in (3, 6)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("case", choices=("lab", "mmm"))
    print(json.dumps(lab() if parser.parse_args().case == "lab" else mmm(), indent=1))
