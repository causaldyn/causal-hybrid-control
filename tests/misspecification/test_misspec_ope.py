"""Off-policy intervals where the model is wrong and where the plan leaves the logged actions.

:func:`~chc.evaluation.evaluate_plan` reports a nominal 95 % interval for the stationary cost of a
plan, from one run logged by another policy, or refuses with
:class:`~chc.evaluation.InfeasibleEvaluation`. The contract read here: the interval covers the
plan's true cost on 95 % of logs of 20 000 steps, or the evaluation refuses.

Each test reads 200 logs from its block of seeds (:mod:`_misspec_counts`) and counts the logs
covered and refused apart. The logs covered or refused have the floor 179, below which a count of
``Binomial(200, 0.95)`` falls with probability at most ``alpha = 1/1000``, or 171, that of
``Binomial(200, 0.92)``, where a method is measured to cover short of 0.95 and its test says so.
Where the plan stays among the logged actions, the logs refused have the ceiling 21 of
``Binomial(200, 0.05)``, so an evaluation that refuses every log fails. Under a wrong model,
``"dr"``'s ``z = 1.96 (centre - cost) / half-width`` is near normal, and the sum of ``z^2`` over the
logs is at most 283.06, the ``1 - alpha / 10`` quantile of chi-square with 200 degrees.

Against an interval ``k`` times its own about its centre, a normal ``z`` is covered with probability
``p1 = erf(1.96 k / sqrt 2)``, 0.883 at ``k = 0.8``: the floor at 0.95 fails a block with
probability 0.651 there and the sum with 0.826, and each fails with probability 0.8 below
``k = 0.778`` and ``k = 0.804``, its minimal detectable defect (``validation/misspec_power.mac``).
Each test below gives the counts of block 0, which CI reads, and the rates over blocks 0 to 4.

The market is ``tests/test_evaluation.py``'s: two states, one incentive, its logger
``u = 0.3 x_0 + 0.2 x_1 + 0.1`` plus noise of variance 0.25, the plan the LQR gain with an offset
of 0.05 and a dither of variance 0.04.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest
from _misspec_counts import ALPHA, LOGS, at_least, at_most, seeds
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
from scipy.stats import chi2

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
STEPS = 20_000
REFUSED = at_most("0.05")
Z_SQUARED = float(chi2.ppf(1.0 - float(ALPHA) / 10.0, LOGS))


class Outcomes(NamedTuple):
    covered: np.ndarray
    refused: np.ndarray
    z: np.ndarray

    def __repr__(self) -> str:
        return (
            f"of {self.covered.size}: covered {self.covered.sum()}, refused "
            f"{self.refused.sum()}, sum z^2 {np.sum(self.z**2):.1f}"
        )


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
    it, from the seed ``100 + seed``."""
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


def _read(method: str, plan: AffinePolicy, model: LinearGaussianPlant) -> Outcomes:
    """Per log of the block, whether the interval covers the plan's true cost, whether the
    evaluation refuses, and, where it does not, the interval's own ``z``."""
    truth = _value(plan)
    covered, refused, z = [], [], []
    for seed in seeds(0):
        try:
            low, high = evaluate_plan(_log(seed), plan, method, plant=model, cost=COST).interval  # type: ignore[arg-type]
        except InfeasibleEvaluation:
            covered.append(False)
            refused.append(True)
            continue
        covered.append(low <= truth <= high)
        refused.append(False)
        z.append(1.96 * (0.5 * (low + high) - truth) / (0.5 * (high - low)))
    return Outcomes(np.array(covered), np.array(refused), np.array(z))


def _holds_or_refuses(found: Outcomes, p0: str) -> None:
    assert np.sum(found.covered | found.refused) >= at_least(p0), found


def _wrong(shift: float) -> LinearGaussianPlant:
    """The market's model with its first state's own persistence off by ``shift``."""
    return LinearGaussianPlant(
        MARKET.a + np.array([[shift, 0.0], [0.0, 0.0]]), MARKET.b, MARKET.offset, MARKET.noise
    )


def test_the_dr_interval_covers_under_a_wrong_model() -> None:
    """The model's ``A[0, 0]`` is 0.95 where the market's is 0.85, ``p0 = 0.95``. Block 0: covered
    189, refused none, sum of ``z^2`` 197.7. Over 1000 logs, covered 0.949. An interval 0.8 times
    its own covers ``p1 = 0.883``: the floor fails a block with probability 0.65 there, and the
    sum, resampled from the 1000, 0.89."""
    found = _read("dr", _plan(0.05), _wrong(0.1))
    _holds_or_refuses(found, "0.95")
    assert found.refused.sum() <= REFUSED, found
    assert np.sum(found.z**2) <= Z_SQUARED, found


def test_the_fqe_interval_covers_less_under_a_wrong_model() -> None:
    """The same model: ``"fqe"`` fits the plan's value on the log and does not read the model's
    error, and its interval covers short of 0.95, so its floor is that of ``p0 = 0.92``. Block 0:
    covered 186, refused none. Over 1000 logs, covered 0.935. An interval 0.8 times its own covers
    ``p1 = 0.863``, where the floor fails a block with probability 0.33."""
    found = _read("fqe", _plan(0.05), _wrong(0.1))
    _holds_or_refuses(found, "0.92")
    assert found.refused.sum() <= REFUSED, found


def test_the_mis_interval_covers_under_a_wrong_model() -> None:
    """``"mis"`` weighs by the model's stationary law, so the model's error moves its estimate to
    first order; its interval reaches to ``"dr"``'s, which reads that error off the log, and holds
    the estimate, so it is wider than a normal one and the sum of ``z^2`` reads nothing of it.
    ``p0 = 0.95``. Block 0: covered 194, refused none. Over 1000 logs, covered 0.974; an interval
    0.8 times its own covers ``p1 = 0.932``, where the floor fails a block with probability 0.02."""
    found = _read("mis", _plan(0.05), _wrong(0.1))
    _holds_or_refuses(found, "0.95")
    assert found.refused.sum() <= REFUSED, found


@pytest.mark.parametrize("method", ["mis", "dr", "fqe"])
@pytest.mark.parametrize("offset", [1.0, 3.0])
def test_a_plan_outside_the_logged_actions_is_covered_or_refused(
    method: str, offset: float
) -> None:
    """The plan's offset is 1 or 3, where the logger's is 0.1 and its dither's sd 0.5. At 1, over
    1000 logs, ``"mis"`` covers 0.949, ``"dr"`` 0.936 and ``"fqe"`` 0.927, the last two short of
    0.95, so their floors are those of ``p0 = 0.92`` (block 0: 196, 193 and 189); at 3 each refuses
    every log. An interval 0.8 times its own covers ``p1`` = 0.886, 0.867 and 0.865 at 1, where the
    floors fail a block with probability 0.60, 0.27 and 0.30."""
    found = _read(method, _plan(offset), MARKET)
    _holds_or_refuses(found, "0.95" if method == "mis" else "0.92")
