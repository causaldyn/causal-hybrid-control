"""The measurements behind docs/adr/0019-re-reading-a-moved-channel.md. Estimates, intervals and
regrets only, no wall time.

    lab      The drift monitor's plant (``scripts/bench_drift.py``): ``x' = 0.8 x + 0.15 + b u +
             eps``, ``eps`` Laplace of standard deviation 0.5, under ``u = -0.5 x + dither xi``,
             against a model ``x' = 0.9 x + 0.05 + u`` whose drift is off in slope and intercept.
             With ``b = 1.07`` the move is 0.07: its estimate's mean, its standard error against
             its spread, and the share of intervals ``estimate +- 1.96 SE`` that cover it, at two
             dither scales and two log lengths.
    moved    The same plant with ``b`` at 1.07 for 3000 decisions and 1.4 for the last 1000: what
             the estimate follows with and without forgetting, and how often its interval covers
             that target.
    plan     The budgeted two-state, two-lever plan of ``tests/test_decision_weight.py``, run over
             and over with a dither on a plant whose one-step map is the model's plus ``E u``,
             ``E`` 10-15% of the one-step channel's entries. Each log's estimate is priced with
             the plan's decision weight, and the plan is re-solved on it by SLSQP. Per log size:
             the realised regrets of keeping and of re-planning against the plan that knew ``E``,
             the two prices, and three rules a caller might build on them (see :func:`plan`).

Run: JAX_ENABLE_X64=1 uv run python scripts/bench_channel_move.py [--logs 200] > out.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from chc.control import LinearConstraint
from chc.cost import QuadraticCost
from chc.dynamics import LinearDynamics
from chc.gate import ChannelMove, DecisionLog, channel_move
from chc.integrate import rk4_step
from chc.plan import CausalPlan, _perturbed_task_cost, causal_plan

SEED = 20260929
NOISE = 0.5

MODEL = LinearDynamics(jnp.array([[-0.3, 0.2], [0.1, -0.5]]), jnp.array([[1.0, 0.3], [-0.2, 0.8]]))
COST = QuadraticCost(
    Q=jnp.diag(jnp.array([1.0, 0.5])),
    R=jnp.diag(jnp.array([0.2, 0.1])),
    Qf=jnp.diag(jnp.array([2.0, 1.0])),
    x_target=jnp.array([1.5, -1.0]),
)
START = jnp.array([0.2, 0.4])
DT, HORIZON, BOX = 0.2, 5, 1.5
BUDGET = LinearConstraint(np.tile([1.0, 0.0], HORIZON)[None, :], -np.inf, 4.0)
# the plant's one-step channel less the model's
PLAN_MOVE = np.array([[0.03, -0.015], [0.02, 0.015]])
PLAN_DITHER = np.array([0.3, 0.2])
PLAN_NOISE = 0.05


def _log(action: np.ndarray, dither: np.ndarray) -> DecisionLog:
    size = action.shape[0]
    return DecisionLog(action, np.ones(size), np.zeros(size, dtype=bool), dither)


def _lab(
    paths: int, channel: np.ndarray, dither: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = rng.normal(0.0, 0.7, paths)
    action, drawn, residual = (np.empty((channel.size, paths)) for _ in range(3))
    for t, b in enumerate(channel):
        drawn[t] = dither * rng.standard_normal(paths)
        action[t] = -0.5 * x + drawn[t]
        noise = rng.laplace(scale=NOISE / math.sqrt(2.0), size=paths)
        after = 0.8 * x + 0.15 + b * action[t] + noise
        residual[t] = after - (0.9 * x + 0.05 + action[t])
        x = after
    return action, drawn, residual


def _read(
    plant: tuple[np.ndarray, np.ndarray, np.ndarray], dither: float, forgetting: float
) -> tuple[np.ndarray, np.ndarray]:
    action, drawn, residual = plant
    moves = [
        channel_move(
            _log(action[:, p], drawn[:, p]),
            residual[:, p],
            dither_scale=dither,
            forgetting=forgetting,
        )
        for p in range(action.shape[1])
    ]
    estimate = np.array([m.estimate[0, 0] for m in moves])
    error = np.array([math.sqrt(m.covariance[0, 0]) for m in moves])
    return estimate, error


def _summary(estimate: np.ndarray, error: np.ndarray, target: float) -> dict[str, float]:
    return {
        "target": target,
        "mean": float(estimate.mean()),
        "bias_z": float((estimate.mean() - target) / (estimate.std() / math.sqrt(estimate.size))),
        "spread_over_error": float(estimate.std() / error.mean()),
        "covered": float(np.mean(np.abs(estimate - target) <= 1.96 * error)),
    }


def lab(paths: int) -> dict[str, object]:
    rng = np.random.default_rng(SEED)
    out: dict[str, object] = {}
    for dither in (0.3, 1.0):
        for decisions in (500, 2000):
            plant = _lab(paths, np.full(decisions, 1.07), dither, rng)
            out[f"dither {dither}, {decisions} decisions"] = _summary(
                *_read(plant, dither, 1.0), 0.07
            )
    return out


def moved(paths: int) -> dict[str, object]:
    rng = np.random.default_rng(SEED + 1)
    channel = np.where(np.arange(4000) < 3000, 1.07, 1.4)
    plant = _lab(paths, channel, 0.3, rng)
    out: dict[str, object] = {}
    for forgetting in (1.0, 0.995):
        weights = forgetting ** np.arange(channel.size - 1, -1, -1)
        target = float(weights @ (channel - 1.0) / weights.sum())
        estimate, error = _read(plant, 0.3, forgetting)
        out[f"forgetting {forgetting}"] = _summary(estimate, error, target) | {
            "effective_size": float(weights.sum() ** 2 / np.sum(weights**2))
        }
    return out


class _Problem(NamedTuple):
    """The budgeted plan at its SLSQP optimum, and the perturbed problem to re-solve it on."""

    plan: CausalPlan
    kept: np.ndarray  # (HORIZON, 2): the optimum with the model's channel
    solve: Callable[[np.ndarray, np.ndarray], np.ndarray]  # (change, start) -> optimum
    value: Callable[[jax.Array, jax.Array], jax.Array]  # (actions, change) -> task cost


def _problem() -> _Problem:
    solved = causal_plan(
        MODEL, START, COST, DT, HORIZON, -BOX, BOX, constraints=(BUDGET,), steps=20_000
    )
    problem = solved._problem
    assert problem is not None
    value = jax.jit(lambda u, change: _perturbed_task_cost(problem, u.reshape(HORIZON, 2), change))
    gradient = jax.jit(jax.grad(value))

    def solve(change: np.ndarray, start: np.ndarray) -> np.ndarray:
        e = jnp.asarray(change)
        result = minimize(
            lambda u: float(value(jnp.asarray(u), e)),
            start.ravel(),
            jac=lambda u: np.asarray(gradient(jnp.asarray(u), e)),
            bounds=[(-BOX, BOX)] * start.size,
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda u: BUDGET.upper - BUDGET.matrix @ u,
                    "jac": lambda u: -BUDGET.matrix,
                }
            ],
            method="SLSQP",
            options={"ftol": 1e-15, "maxiter": 2000},
        )
        assert result.success, result.message
        return result.x.reshape(HORIZON, 2)

    kept = solve(np.zeros((2, 2)), np.asarray(solved.actions, dtype=np.float64))
    return _Problem(replace(solved, actions=jnp.asarray(kept)), kept, solve, value)


def _plan_log(
    runs: int, kept: np.ndarray, phi: np.ndarray, gamma: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``runs`` runs of the kept plan with a dither, on the plant whose one-step map is the model's
    plus ``PLAN_MOVE u`` and noise: per run and step, the action, the dither and the residual."""
    drawn = PLAN_DITHER * rng.standard_normal((runs, HORIZON, 2))
    action = kept + drawn
    x = np.broadcast_to(np.asarray(START, dtype=np.float64), (runs, 2))
    residual = np.empty((runs, HORIZON, 2))
    for t in range(HORIZON):
        model_step = x @ phi.T + action[:, t] @ gamma.T
        after = (
            model_step + action[:, t] @ PLAN_MOVE.T + PLAN_NOISE * rng.standard_normal((runs, 2))
        )
        residual[:, t] = after - model_step
        x = after
    return action, drawn, residual


def _read_runs(action: np.ndarray, drawn: np.ndarray, residual: np.ndarray) -> ChannelMove:
    return channel_move(
        _log(action.reshape(-1, 2), drawn.reshape(-1, 2)),
        residual.reshape(-1, 2),
        dither_scale=PLAN_DITHER,
    )


def plan(logs: int) -> dict[str, object]:
    """Per log size, the realised regrets of keeping and of re-planning on the estimate, the two
    prices, and three rules a caller might build on them:

    * ``compared``: re-plan when the keep price exceeds the re-plan price, both off one log;
    * ``shrunk``: re-plan on the estimate times ``max(0, 1 - tr(W S) / dh' W dh)``, the linear
      shrinkage that is optimal in ``W``'s metric when ``d' W d`` is known, with it estimated;
    * ``split``: compare the first half's keep price with the second half's re-plan price, and
      re-plan on the second half's estimate, so the choice does not select on its error.
    """
    base, kept, solve, value = _problem()
    weight = base.decision_weight()
    truth = jnp.asarray(PLAN_MOVE)
    best = float(value(jnp.asarray(solve(PLAN_MOVE, kept)), truth))

    def regret(actions: np.ndarray) -> float:
        return float(value(jnp.asarray(actions), truth)) - best

    keep_regret = regret(kept)
    phi, gamma = (
        np.asarray(m, dtype=np.float64)
        for m in jax.jacfwd(lambda x, u: rk4_step(MODEL, 0.0, x, u, DT), argnums=(0, 1))(
            START, jnp.zeros(2)
        )
    )
    rng = np.random.default_rng(SEED + 2)
    out: dict[str, object] = {
        "keep_regret": keep_regret,
        "keep_price_at_the_true_move": weight.regret(PLAN_MOVE),
        "weakly_active": weight.weakly_active,
        "free": weight.free,
    }
    for runs in (6, 20, 80, 320):
        rows: dict[str, list[float]] = {
            name: []
            for name in ("keep_price", "replan_price", "replan", "compared", "shrunk", "split")
        }
        for _ in range(logs):
            action, drawn, residual = _plan_log(runs, kept, phi, gamma, rng)
            move = _read_runs(action, drawn, residual)
            price = move.price(weight)
            replanned = regret(solve(move.estimate, kept))
            rows["keep_price"].append(price.keep)
            rows["replan_price"].append(price.replan)
            rows["replan"].append(replanned)
            rows["compared"].append(replanned if price.keep > price.replan else keep_regret)
            d = move.estimate.ravel()
            alpha = max(0.0, 1.0 - 2.0 * price.replan / float(d @ weight.matrix @ d))
            rows["shrunk"].append(
                regret(solve(alpha * move.estimate, kept)) if alpha else keep_regret
            )
            half = runs // 2
            first = _read_runs(action[:half], drawn[:half], residual[:half])
            second = _read_runs(action[half:], drawn[half:], residual[half:])
            rows["split"].append(
                regret(solve(second.estimate, kept))
                if first.price(weight).keep > second.price(weight).replan
                else keep_regret
            )
        summary: dict[str, float] = {"always_keep": keep_regret}
        for name, values in rows.items():
            array = np.array(values)
            summary[f"{name}_mean"] = float(array.mean())
            summary[f"{name}_se"] = float(array.std() / math.sqrt(logs))
        out[f"{runs * HORIZON} decisions"] = summary
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=2000)
    parser.add_argument("--logs", type=int, default=200)
    args = parser.parse_args()
    out = {
        "paths": args.paths,
        "logs": args.logs,
        "x64": bool(jax.config.jax_enable_x64),
        "lab": lab(args.paths),
        "moved": moved(args.paths),
        "plan": plan(args.logs),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
