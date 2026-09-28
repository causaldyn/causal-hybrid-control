"""The coverage behind docs/adr/0009-evaluating-a-plan.md. Costs and coverage only, no wall time.

The loop is the two-sided market of ``tests/test_evaluation.py``: ``x = (imbalance, backlog)``,
``u`` the incentive, a legacy logger ``u = 0.3 x_1 + 0.2 x_2 + 0.1 + noise`` and the LQ plan with
an offset of 0.05, deterministic, so every weighted method evaluates it smoothed and subtracts the
model's ``tau^2 beta_hat``. Each replicate draws fresh logs and fits the model to them by least
squares, or takes the true plant; truths are the plan's closed-form values under the true plant.

    stationary  one logged run of 4000 transitions from the logger's stationary law; "mis", "dr"
                and "fqe" estimate the plan's average cost per step.
    episodes    3000 episodes of 5 steps from ``x_0 ~ N((1, 0.5), 0.2 I)`` under a wider logger,
                ``sigma = 1``; "pdis" estimates the plan's expected cost over them.

    valley      MountainCarContinuous-v0 linearised at its valley floor, its force disturbed with
                sd 0.2 and logged by a lightly damped operator, ``u = -20 v + N(0, 0.3^2)``, for an
                LQR plan whose states are ten times narrower than the logs'. The weight sits in a
                few of the logger's passes through the floor; "mis" and "dr" also report what a
                fixed ``t_39`` would have covered on the same replicates.

``model_error`` is 0 (the model's correction trusted) and 1 (the default: carried in the
interval). A refusal is counted, not scored.

Run: uv run python scripts/bench_evaluation.py {stationary,episodes,valley} [--replicates 500]
"""

from __future__ import annotations

import argparse
import json
import math

import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
from scipy.stats import t as student

from chc import (
    AffinePolicy,
    InfeasibleEvaluation,
    InitialLaw,
    LinearGaussianPlant,
    QuadraticCost,
    evaluate_plan,
)

MARKET = LinearGaussianPlant(
    np.array([[0.85, 0.15], [0.25, 0.75]]),
    np.array([[-0.6], [-0.2]]),
    np.zeros(2),
    np.diag([0.05, 0.03]),
)
Q, R = np.diag([2.0, 1.0]), np.array([[0.4]])
COST = QuadraticCost(Q=jnp.asarray(Q), R=jnp.asarray(R), Qf=jnp.asarray(Q), x_target=jnp.zeros(2))
_P = solve_discrete_are(MARKET.a, MARKET.b, Q, R)
PLAN = AffinePolicy(
    -np.linalg.solve(R + MARKET.b.T @ _P @ MARKET.b, MARKET.b.T @ _P @ MARKET.a),
    np.array([0.05]),
    np.zeros((1, 1)),
)
INITIAL = InitialLaw(np.array([1.0, 0.5]), 0.2 * np.eye(2))

_VALLEY_B = np.array([[0.0015], [0.0015]])
VALLEY = LinearGaussianPlant(
    np.array([[0.9925, 1.0], [-0.0075, 1.0]]),
    _VALLEY_B,
    np.zeros(2),
    0.04 * _VALLEY_B @ _VALLEY_B.T,
)
VALLEY_Q, VALLEY_R = np.diag([400.0, 40000.0]), np.array([[1.0]])
VALLEY_COST = QuadraticCost(
    Q=jnp.asarray(VALLEY_Q),
    R=jnp.asarray(VALLEY_R),
    Qf=jnp.asarray(VALLEY_Q),
    x_target=jnp.zeros(2),
)
_PV = solve_discrete_are(VALLEY.a, VALLEY.b, VALLEY_Q, VALLEY_R)
VALLEY_PLAN = AffinePolicy(
    -np.linalg.solve(VALLEY_R + VALLEY.b.T @ _PV @ VALLEY.b, VALLEY.b.T @ _PV @ VALLEY.a),
    np.array([0.1]),
    np.zeros((1, 1)),
)
VALLEY_LOGGER = AffinePolicy(np.array([[0.0, -20.0]]), np.zeros(1), np.array([[0.09]]))


def _logger(variance: float) -> AffinePolicy:
    return AffinePolicy(np.array([[0.3, 0.2]]), np.array([0.1]), np.array([[variance]]))


def _stage(
    mx: np.ndarray, sx: np.ndarray, policy: AffinePolicy, q: np.ndarray = Q, r: np.ndarray = R
) -> float:
    mu = policy.gain @ mx + policy.offset
    su = policy.gain @ sx @ policy.gain.T + policy.covariance
    return 0.5 * float(np.trace(q @ sx) + mx @ q @ mx + np.trace(r @ su) + mu @ r @ mu)


def _stationary_state(
    policy: AffinePolicy, plant: LinearGaussianPlant = MARKET
) -> tuple[np.ndarray, np.ndarray]:
    f = plant.a + plant.b @ policy.gain
    sx = solve_discrete_lyapunov(f, plant.b @ policy.covariance @ plant.b.T + plant.noise)
    return np.linalg.solve(np.eye(2) - f, plant.b @ policy.offset), sx


def _episode_truth(horizon: int) -> float:
    f = MARKET.a + MARKET.b @ PLAN.gain
    mx, sx, total = INITIAL.mean, INITIAL.covariance, 0.0
    for _ in range(horizon):
        total += _stage(mx, sx, PLAN)
        mx = f @ mx + MARKET.b @ PLAN.offset
        sx = f @ sx @ f.T + MARKET.noise
    return total


def _rollouts(
    policy: AffinePolicy,
    starts: np.ndarray,
    steps: int,
    rng: np.random.Generator,
    plant: LinearGaussianPlant = MARKET,
    noise: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``noise`` maps standard normals to the plant's noise: the Cholesky factor by default, and
    ``b`` times the disturbance's sd for a plant disturbed only through its input."""
    runs = starts.shape[0]
    x, u = np.empty((runs, steps + 1, 2)), np.empty((runs, steps, 1))
    x[:, 0] = starts
    dither = np.linalg.cholesky(policy.covariance)
    root = np.linalg.cholesky(plant.noise) if noise is None else noise
    for t in range(steps):
        u[:, t] = (
            x[:, t] @ policy.gain.T + policy.offset + rng.standard_normal((runs, 1)) @ dither.T
        )
        x[:, t + 1] = (
            x[:, t] @ plant.a.T
            + u[:, t] @ plant.b.T
            + rng.standard_normal((runs, root.shape[1])) @ root.T
        )
    return x, u


def _fitted(x: np.ndarray, u: np.ndarray) -> LinearGaussianPlant:
    """Least squares of ``x'`` on ``(x, u, 1)``, pooled over every transition in the logs."""
    xs, us, xn = x[..., :-1, :].reshape(-1, 2), u.reshape(-1, 1), x[..., 1:, :].reshape(-1, 2)
    design = np.hstack([xs, us, np.ones((xs.shape[0], 1))])
    coef, *_ = np.linalg.lstsq(design, xn, rcond=None)
    residual = xn - design @ coef
    noise = residual.T @ residual / (xs.shape[0] - design.shape[1])
    return LinearGaussianPlant(coef[:2].T, coef[2:3].T, coef[3], 0.5 * (noise + noise.T))


def _summary(errors: list[float], covered: list[bool], widths: list[float], **extra) -> dict:
    e = np.asarray(errors)
    n = e.size
    return {
        "scored": n,
        "coverage": float(np.mean(covered)) if n else None,
        "coverage_se": math.sqrt(0.95 * 0.05 / n) if n else None,
        "mean_error": float(e.mean()) if n else None,
        "mean_error_se": float(e.std(ddof=1) / math.sqrt(n)) if n > 1 else None,
        "rmse": float(math.sqrt(np.mean(e * e))) if n else None,
        "mean_width": float(np.mean(widths)) if n else None,
        **extra,
    }


def _run(case: str, replicates: int) -> dict:
    rng = np.random.default_rng(20260928)
    plant, plan, cost = MARKET, PLAN, COST
    if case == "valley":
        plant, plan, cost = VALLEY, VALLEY_PLAN, VALLEY_COST
        mean, cov = _stationary_state(VALLEY_LOGGER, VALLEY)
        starts = mean + rng.standard_normal((replicates, 2)) @ np.linalg.cholesky(cov).T
        xs, us = _rollouts(VALLEY_LOGGER, starts, 4000, rng, VALLEY, 0.2 * _VALLEY_B)
        truth = _stage(*_stationary_state(VALLEY_PLAN, VALLEY), VALLEY_PLAN, VALLEY_Q, VALLEY_R)
        methods = ("mis", "dr")
    elif case == "stationary":
        logger = _logger(0.25)
        mean, cov = _stationary_state(logger)
        starts = mean + rng.standard_normal((replicates, 2)) @ np.linalg.cholesky(cov).T
        xs, us = _rollouts(logger, starts, 4000, rng)
        truth = _stage(*_stationary_state(PLAN), PLAN)
        methods = ("mis", "dr", "fqe")
    else:
        logger = _logger(1.0)
        horizon, runs = 5, 3000
        starts = (
            INITIAL.mean
            + rng.standard_normal((replicates * runs, 2)) @ np.linalg.cholesky(INITIAL.covariance).T
        )
        x, u = _rollouts(logger, starts, horizon, rng)
        xs, us = (
            x.reshape(replicates, runs, horizon + 1, 2),
            u.reshape(replicates, runs, horizon, 1),
        )
        truth = _episode_truth(horizon)
        methods = ("pdis",)
    out: dict = {"case": case, "replicates": replicates, "truth": truth, "arms": []}
    for method in methods:
        for model_name in ("true", "fitted"):
            for model_error in (0.0, 1.0):
                if method == "fqe" and model_error == 1.0:
                    continue  # nothing is smoothed, so nothing is corrected
                errors, covered, widths, shares, refused = [], [], [], [], 0
                dofs, fixed_t = [], []
                for r in range(replicates):
                    logs = {"x": xs[r], "u": us[r]}
                    model = plant if model_name == "true" else _fitted(xs[r], us[r])
                    try:
                        result = evaluate_plan(
                            logs, plan, method, plant=model, cost=cost, model_error=model_error
                        )
                    except InfeasibleEvaluation:
                        refused += 1
                        continue
                    errors.append(result.value - truth)
                    covered.append(result.interval[0] <= truth <= result.interval[1])
                    widths.append(result.interval[1] - result.interval[0])
                    shares.append(result.model_share)
                    if result.degrees_of_freedom is not None:
                        dofs.append(result.degrees_of_freedom)
                        # The batch part of the half-width, rescaled from t_dof to t_39.
                        sampling = 0.5 * widths[-1] - model_error * result.model_correction
                        ratio = student.ppf(0.975, 39) / student.ppf(0.975, dofs[-1])
                        fixed_t.append(
                            abs(errors[-1])
                            <= model_error * result.model_correction + ratio * sampling
                        )
                out["arms"].append(
                    {
                        "method": method,
                        "model": model_name,
                        "model_error": model_error,
                        **_summary(
                            errors,
                            covered,
                            widths,
                            refused=refused,
                            mean_model_share=float(np.mean(shares)) if shares else None,
                            degrees_of_freedom=(
                                [float(np.min(dofs)), float(np.median(dofs)), float(np.max(dofs))]
                                if dofs
                                else None
                            ),
                            coverage_fixed_t39=float(np.mean(fixed_t)) if fixed_t else None,
                        ),
                    }
                )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=["stationary", "episodes", "valley"])
    parser.add_argument("--replicates", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(_run(args.case, args.replicates), indent=2))


if __name__ == "__main__":
    main()
