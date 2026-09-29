"""The measurements behind :mod:`chc.misspecification`. Regrets, prices and rates only, no wall
time.

The plant has two states and one action, ``rate = A x + b(x) u + c z``, with a channel that moves
with the first state, ``b(x) = b0 + slope x_0 b1``. The log's action is confounded through ``z``,
``u = mean + 0.9 z - 0.3 x_0 + N(0, 0.5^2)``, and ``z`` is adjusted for. Each log is fitted twice by
the constant-channel class, unweighted and weighted by ``exp(x_0)``, and the plan is a 12-step
tracking plan made on the unweighted fit. At ``slope = 0`` the class holds the truth.

    metric       On logs of --rows transitions whose actions average -1, 0 and +1 (slope 1), and
                 one whose channel is constant: the regret of planning on the unweighted fit when
                 the weighted fit's model holds, both plans re-solved; the gate's quadratic
                 ``d' W d / 2`` and what it reports; and the price of the channel's move alone, the
                 drift held, which :meth:`chc.plan.CausalPlan.decision_weight` reads, next to that
                 move's own regret.
    reach        The same logs with the difference scaled by t: the regret against ``t^2 d' W d/2``.
    calibration  --logs whole logs of --small transitions, redrawn (states, actions, ``z`` and
                 noise), at slope 1 and slope 0: the cost's z-score against the quadratic at the
                 population difference (from --population transitions), the noise against the
                 difference's realised spread in ``W``, and the p-value's rejection rates.
    power        The rejection rate at 5% over --logs logs as the slope grows from 0.
    unseen       Two actions whose log moves only the second at twice the first, under a constant
                 channel: what the gate reads, how many directions it reports it cannot see, and
                 what planning on the fit costs against the truth. Then the same with the plan held
                 to the log's own ratio, where the unmoved direction stops mattering.

Run: JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run python scripts/bench_misspecification.py > out.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from chc.control import LinearConstraint
from chc.cost import QuadraticCost, total_cost
from chc.dynamics import HybridDynamics
from chc.dynamics_id import CausalDynamicsFit, fit_causal_residual
from chc.integrate import rk4_step
from chc.misspecification import _parameters, _residual, misspecification_cost
from chc.plan import CausalPlan, causal_plan
from chc.residual import ControlAffineResidual

SEED = 20260930
DT, HORIZON, BOX = 0.1, 12, 5.0
A = np.array([[-0.5, 0.2], [0.0, -0.3]])
B0 = np.array([0.8, -0.4])
B1 = np.array([0.6, 0.0])
C = np.array([1.5, 0.0])
START = jnp.zeros(2)
COST = QuadraticCost(
    Q=jnp.diag(jnp.array([1.0, 0.1])),
    R=jnp.array([[0.05]]),
    Qf=jnp.diag(jnp.array([1.0, 0.1])),
    x_target=jnp.array([1.0, 0.0]),
)
FIT = {"adjust_for": ("z",), "degree": 1, "channel_degree": 0, "nuisance_degree": 2, "seed": 0}
# the unseen section's constant channel, one column per action
PAIR = np.array([[0.8, 0.3], [-0.4, 0.5]])


def _known(t: jax.Array, x: jax.Array, u: jax.Array) -> jax.Array:
    return jnp.zeros_like(x)


def _log(rows: int, seed: int, slope: float, mean: float) -> dict[str, jax.Array]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (rows, 2))
    z = rng.normal(0.0, 1.0, rows)
    u = mean + 0.9 * z - 0.3 * x[:, 0] + rng.normal(0.0, 0.5, rows)
    b = B0[None, :] + slope * np.outer(x[:, 0], B1)
    rate = x @ A.T + b * u[:, None] + np.outer(z, C)
    x_next = x + DT * rate + rng.normal(0.0, 0.01, (rows, 2))
    return {
        "x": jnp.asarray(x),
        "u": jnp.asarray(u[:, None]),
        "x_next": jnp.asarray(x_next),
        "z": jnp.asarray(z[:, None]),
    }


def _fits(data: dict[str, jax.Array], influence: bool = True) -> tuple[CausalDynamicsFit, ...]:
    reference = fit_causal_residual(_known, data, DT, influence=influence, **FIT)
    alternative = fit_causal_residual(
        _known, data, DT, weights=lambda s: jnp.exp(s[:, 0]), influence=influence, **FIT
    )
    return reference, alternative


def _plan(residual: ControlAffineResidual, cost: QuadraticCost = COST, **kwargs) -> CausalPlan:
    model = HybridDynamics(known=_known, residual=residual)
    return causal_plan(model, START, cost, DT, HORIZON, -BOX, BOX, steps=20_000, **kwargs)


def _regret(residual: ControlAffineResidual, actions: jax.Array, **kwargs) -> float:
    """What ``actions`` lose under ``residual``'s model against the plan made on it."""
    model = HybridDynamics(known=_known, residual=residual)
    cost = kwargs.get("cost", COST)
    best = _plan(residual, **kwargs).actions
    return float(
        total_cost(model, START, actions, DT, cost) - total_cost(model, START, best, DT, cost)
    )


def _quadratic(plan: CausalPlan, reference: CausalDynamicsFit, change: np.ndarray) -> float:
    """``change' W change / 2``, ``W`` the gate's own at ``plan``: a fit of ``reference``'s
    parameters plus ``change`` that keeps ``reference``'s influence differs from it with a zero
    covariance, so the gate's cost is the bare quadratic."""
    moved = _residual(jnp.asarray(_parameters(reference.residual) + change), reference.residual)
    return misspecification_cost(
        plan, reference, dataclasses.replace(reference, residual=moved)
    ).cost


def _channel_only(plan: CausalPlan, reference: CausalDynamicsFit, alternative: CausalDynamicsFit):
    """The alternative's channel beside the reference's drift, its one-step channel's move against
    the reference's under RK4, and what :meth:`CausalPlan.decision_weight` prices for it."""
    moved = dataclasses.replace(reference.residual, channel=alternative.residual.channel)

    def one_step(residual: ControlAffineResidual) -> jax.Array:
        model = HybridDynamics(known=_known, residual=residual)
        return jax.jacfwd(lambda u: rk4_step(model, 0.0, START, u, DT))(jnp.zeros(1))

    change = one_step(moved) - one_step(reference.residual)
    return moved, plan.decision_weight().regret(change)


def metric(rows: int) -> dict[str, dict[str, float]]:
    out = {}
    for slope, mean in ((1.0, -1.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)):
        reference, alternative = _fits(_log(rows, SEED, slope, mean))
        plan = _plan(reference.residual)
        regret = _regret(alternative.residual, plan.actions)
        gate = misspecification_cost(plan, reference, alternative)
        moved, channel_price = _channel_only(plan, reference, alternative)
        channel_regret = _regret(moved, plan.actions)
        out[f"slope {slope:g}, mean action {mean:+g}"] = {
            "regret": regret,
            "quadratic": gate.cost + gate.noise,
            "quadratic_over_regret": (gate.cost + gate.noise) / regret,
            "cost": gate.cost,
            "cost_error": gate.cost_error,
            "noise": gate.noise,
            "p_value": gate.p_value,
            "unseen": gate.unseen,
            "channel_price": channel_price,
            "channel_price_over_regret": channel_price / regret,
            "channel_regret": channel_regret,
            "channel_regret_over_regret": channel_regret / regret,
            "plan_mean_action": float(jnp.mean(plan.actions)),
        }
    return out


def reach(rows: int) -> dict[str, dict[str, float]]:
    out = {}
    for mean in (-1.0, 0.0, 1.0):
        reference, alternative = _fits(_log(rows, SEED, 1.0, mean))
        plan = _plan(reference.residual)
        own = _parameters(reference.residual)
        d = _parameters(alternative.residual) - own
        quadratic = _quadratic(plan, reference, d)
        for t in (1.0, 0.3, 0.1):
            regret = _regret(_residual(jnp.asarray(own + t * d), reference.residual), plan.actions)
            out[f"mean action {mean:+g}, t {t:g}"] = {
                "regret": regret,
                "quadratic": t * t * quadratic,
                "quadratic_over_regret": t * t * quadratic / regret,
            }
    return out


def _population(slope: float, rows: int) -> np.ndarray:
    reference, alternative = _fits(_log(rows, SEED - 1, slope, 0.0), influence=False)
    return _parameters(alternative.residual) - _parameters(reference.residual)


def _rates(p_values: np.ndarray) -> dict[str, float]:
    return {f"reject_at_{level:g}": float(np.mean(p_values < level)) for level in (0.01, 0.05, 0.1)}


def calibration(logs: int, small: int, population: int) -> dict[str, dict[str, float]]:
    out = {}
    for slope in (1.0, 0.0):
        truth = _population(slope, population)
        z, targets, noise, realised, p_values = [], [], [], [], []
        for index in range(logs):
            reference, alternative = _fits(_log(small, SEED + 1 + index, slope, 0.0))
            plan = _plan(reference.residual)
            gate = misspecification_cost(plan, reference, alternative)
            d = _parameters(alternative.residual) - _parameters(reference.residual)
            target = _quadratic(plan, reference, truth)
            z.append((gate.cost - target) / gate.cost_error)
            targets.append(target)
            noise.append(gate.noise)
            realised.append(_quadratic(plan, reference, d - truth))
            p_values.append(gate.p_value)
        z_array, p_array = np.asarray(z), np.asarray(p_values)
        out[f"slope {slope:g}"] = {
            "logs": logs,
            "rows": small,
            "target_mean": float(np.mean(targets)),
            "z_mean": float(z_array.mean()),
            "z_mean_se": float(z_array.std(ddof=1) / math.sqrt(logs)),
            "z_sd": float(z_array.std(ddof=1)),
            "covered_at_95": float(np.mean(np.abs(z_array) <= 1.959964)),
            "noise_mean": float(np.mean(noise)),
            "realised_noise_mean": float(np.mean(realised)),
            "realised_noise_se": float(np.std(realised, ddof=1) / math.sqrt(logs)),
            **_rates(p_array),
            "ks_uniform": float(
                np.max(np.abs(np.sort(p_array) - (np.arange(1, logs + 1) - 0.5) / logs))
            ),
        }
    return out


def power(logs: int, small: int) -> dict[str, dict[str, float]]:
    out = {}
    for slope in (0.0, 0.005, 0.01, 0.02, 0.03):
        p_values = []
        for index in range(logs):
            reference, alternative = _fits(_log(small, SEED + 1 + index, slope, 0.0))
            plan = _plan(reference.residual)
            p_values.append(misspecification_cost(plan, reference, alternative).p_value)
        out[f"slope {slope:g}"] = {"logs": logs, "rows": small, **_rates(np.asarray(p_values))}
    return out


def _pair_log(rows: int, seed: int) -> dict[str, jax.Array]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (rows, 2))
    z = rng.normal(0.0, 1.0, rows)
    first = 0.9 * z - 0.3 * x[:, 0] + rng.normal(0.0, 0.5, rows)
    u = np.stack([first, 2.0 * first], axis=1)
    rate = x @ A.T + u @ PAIR.T + np.outer(z, C)
    x_next = x + DT * rate + rng.normal(0.0, 0.01, (rows, 2))
    return {
        "x": jnp.asarray(x),
        "u": jnp.asarray(u),
        "x_next": jnp.asarray(x_next),
        "z": jnp.asarray(z[:, None]),
    }


def unseen(rows: int) -> dict[str, dict[str, float]]:
    cost = dataclasses.replace(COST, R=0.05 * jnp.eye(2))
    truth = ControlAffineResidual(
        drift=jnp.asarray(np.concatenate([np.zeros((2, 1)), A], axis=1)),
        channel=jnp.asarray(PAIR[:, :, None]),
        degree=1,
        channel_degree=0,
    )
    reference, alternative = _fits(_pair_log(rows, SEED))
    # the second action held at twice the first, step by step: the log's own ratio
    ratio = LinearConstraint(np.kron(np.eye(HORIZON), [[-2.0, 1.0]]), 0.0, 0.0)
    out = {}
    for name, constraints in (("free", ()), ("held to the log's ratio", (ratio,))):
        plan = _plan(reference.residual, cost, constraints=constraints)
        gate = misspecification_cost(plan, reference, alternative)
        out[name] = {
            "unmoved": int(np.shape(reference.unmoved)[1]),
            "unseen": gate.unseen,
            "cost": gate.cost,
            "cost_error": gate.cost_error,
            "p_value": gate.p_value,
            "regret_against_truth": _regret(
                truth, plan.actions, cost=cost, constraints=constraints
            ),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=40_000)
    parser.add_argument("--small", type=int, default=4000)
    parser.add_argument("--population", type=int, default=400_000)
    parser.add_argument("--logs", type=int, default=200)
    parser.add_argument(
        "--sections", nargs="+", default=["metric", "reach", "calibration", "power", "unseen"]
    )
    args = parser.parse_args()
    sections: dict[str, Callable[[], object]] = {
        "metric": lambda: metric(args.rows),
        "reach": lambda: reach(args.rows),
        "calibration": lambda: calibration(args.logs, args.small, args.population),
        "power": lambda: power(args.logs, args.small),
        "unseen": lambda: unseen(args.rows),
    }
    out: dict[str, object] = {"x64": bool(jax.config.jax_enable_x64), "seed": SEED}
    for name in args.sections:
        out[name] = sections[name]()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
