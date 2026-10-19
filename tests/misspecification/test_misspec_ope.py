"""Off-policy intervals where the model is wrong and where the plan leaves the logged actions.

:func:`~chc.evaluation.evaluate_plan` reports a nominal 95 % interval for the stationary cost of a
plan, from one run logged by another policy, or refuses with
:class:`~chc.evaluation.InfeasibleEvaluation`. The contract read here: on 40 logs of 20 000
steps, the interval covers the plan's true cost on 95 % of them or more, less three Monte Carlo
standard deviations (``sqrt(0.95 * 0.05 / 40) = 0.034``), or the evaluation refuses.

The market is ``tests/test_evaluation.py``'s: two states, one incentive, its logger
``u = 0.3 x_0 + 0.2 x_1 + 0.1`` plus noise of variance 0.25, the plan the LQR gain with an offset
of 0.05 and a dither of variance 0.04.
"""

from __future__ import annotations

import functools
import math

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

from chc.cost import QuadraticCost
from chc.evaluation import AffinePolicy, InfeasibleEvaluation, LinearGaussianPlant, evaluate_plan

pytestmark = pytest.mark.misspecification

MARKET = LinearGaussianPlant(
    np.array([[0.85, 0.15], [0.25, 0.75]]),
    np.array([[-0.6], [-0.2]]),
    np.zeros(2),
    np.diag([0.05, 0.03]),
)
Q, R = np.diag([2.0, 1.0]), np.array([[0.4]])
COST = QuadraticCost(Q=jnp.asarray(Q), R=jnp.asarray(R), Qf=jnp.asarray(Q), x_target=jnp.zeros(2))
_P = solve_discrete_are(MARKET.a, MARKET.b, Q, R)
GAIN = -np.linalg.solve(R + MARKET.b.T @ _P @ MARKET.b, MARKET.b.T @ _P @ MARKET.a)
LOGGER = AffinePolicy(np.array([[0.3, 0.2]]), np.array([0.1]), np.array([[0.25]]))
LOGS, STEPS = 40, 20_000
FLOOR = 0.95 - 3.0 * math.sqrt(0.95 * 0.05 / LOGS)


def _plan(offset: float) -> AffinePolicy:
    return AffinePolicy(GAIN, np.array([offset]), np.array([[0.04]]))


def _value(policy: AffinePolicy) -> float:
    """The stationary expected stage cost of ``policy`` on the market, in closed form."""
    loop = MARKET.a + MARKET.b @ policy.gain
    drive = MARKET.b @ policy.covariance @ MARKET.b.T + MARKET.noise
    covariance = solve_discrete_lyapunov(loop, drive)
    mean = np.linalg.solve(np.eye(2) - loop, MARKET.b @ policy.offset + MARKET.offset)
    action_mean = policy.gain @ mean + policy.offset
    action_covariance = policy.gain @ covariance @ policy.gain.T + policy.covariance
    return 0.5 * float(
        np.trace(Q @ covariance)
        + mean @ Q @ mean
        + np.trace(R @ action_covariance)
        + action_mean @ R @ action_mean
    )


@functools.cache
def _log(seed: int) -> dict[str, np.ndarray]:
    """One run of the logger from its stationary law, drawn as ``tests/test_evaluation.py`` draws
    it, on seeds 100 to 139."""
    rng = np.random.default_rng(100 + seed)
    loop = MARKET.a + MARKET.b @ LOGGER.gain
    covariance = solve_discrete_lyapunov(
        loop, MARKET.b @ LOGGER.covariance @ MARKET.b.T + MARKET.noise
    )
    mean = np.linalg.solve(np.eye(2) - loop, MARKET.b @ LOGGER.offset + MARKET.offset)
    noise, dither = np.linalg.cholesky(MARKET.noise), np.linalg.cholesky(LOGGER.covariance)
    x, u = np.empty((STEPS + 1, 2)), np.empty((STEPS, 1))
    x[0] = (mean + rng.standard_normal((1, 2)) @ np.linalg.cholesky(covariance).T)[0]
    for t in range(STEPS):
        u[t] = x[t] @ LOGGER.gain.T + LOGGER.offset + (rng.standard_normal((1, 1)) @ dither.T)[0]
        x[t + 1] = (
            x[t] @ MARKET.a.T
            + u[t] @ MARKET.b.T
            + MARKET.offset
            + (rng.standard_normal((1, 2)) @ noise.T)[0]
        )
    return {"x": x, "u": u}


def _held_or_refused(method: str, plan: AffinePolicy, model: LinearGaussianPlant) -> float:
    """The share of logs whose interval covers the plan's true cost or which refuse."""
    truth = _value(plan)
    kept = 0
    for seed in range(LOGS):
        try:
            low, high = evaluate_plan(_log(seed), plan, method, plant=model, cost=COST).interval  # type: ignore[arg-type]
        except InfeasibleEvaluation:
            kept += 1
            continue
        kept += low <= truth <= high
    return kept / LOGS


def _wrong(shift: float) -> LinearGaussianPlant:
    """The market's model with its first state's own persistence off by ``shift``."""
    return LinearGaussianPlant(
        MARKET.a + np.array([[shift, 0.0], [0.0, 0.0]]), MARKET.b, MARKET.offset, MARKET.noise
    )


@pytest.mark.parametrize("method", ["dr", "fqe"])
def test_the_interval_covers_under_a_wrong_model(method: str) -> None:
    """The model's ``A[0, 0]`` is 0.95 where the market's is 0.85: ``"dr"`` covers on 40 of 40,
    ``"fqe"`` on 36."""
    assert _held_or_refused(method, _plan(0.05), _wrong(0.1)) >= FLOOR


def test_the_mis_interval_covers_under_a_wrong_model() -> None:
    """``"mis"`` weighs by the model's stationary law, so the model's error moves its estimate to
    first order; its interval reaches to ``"dr"``'s, which reads that error off the log. It covers
    on 40 of 40 at an ``A[0, 0]`` error of 0.1 (0.95 and 0.975 at 0 and 0.05)."""
    assert _held_or_refused("mis", _plan(0.05), _wrong(0.1)) >= FLOOR


@pytest.mark.parametrize("method", ["mis", "dr", "fqe"])
@pytest.mark.parametrize("offset", [1.0, 3.0])
def test_a_plan_outside_the_logged_actions_is_covered_or_refused(
    method: str, offset: float
) -> None:
    """The plan's offset is 1 or 3, where the logger's is 0.1 and its dither's sd 0.5: at 1 each
    method covers on 38 to 40 logs, at 3 each refuses on all 40."""
    assert _held_or_refused(method, _plan(offset), MARKET) >= FLOOR
