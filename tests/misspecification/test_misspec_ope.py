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
import math
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest
from _misspec_counts import ALPHA, LOGS, at_least, at_most, seeds
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
from scipy.stats import chi2

from chc.cost import QuadraticCost
from chc.decision import Lever, Target, prescribe
from chc.evaluation import AffinePolicy, InfeasibleEvaluation, LinearGaussianPlant, evaluate_plan
from chc.graph import CausalGraph
from chc.panel import Panel

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


# ---- a plan made on groups, evaluated by Prescription.evaluate

# Units in zones of four, x' = 0.8 x + 0.5 u + SPILL m + N(0, 0.1^2), m the mean lever of the unit's
# zone-mates, each lever drawn from its unit's state alone, u = -0.3 x + N(0, 0.7^2).
SPILL, ZONE_SIZE, ZONES, ZONE_PERIODS, ZONE_PANELS = 0.4, 4, 150, 24, 40
ZONE_GOAL, ZONE_PRICE = 0.3, 0.5


def _zone_panel(seed: int) -> Panel:
    """A log of the zones, the units starting where 40 periods of the logger leave them."""
    rng = np.random.default_rng(seed)
    n = ZONE_SIZE * ZONES

    def step(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        mine = u.reshape(ZONES, ZONE_SIZE)
        mates = (mine.sum(axis=1, keepdims=True) - mine) / (ZONE_SIZE - 1)
        return 0.8 * x + 0.5 * u + SPILL * mates.ravel() + 0.1 * rng.standard_normal(n)

    x = np.zeros(n)
    for _ in range(40):
        x = step(x, -0.3 * x + 0.7 * rng.standard_normal(n))
    rows: dict[str, list[np.ndarray]] = {name: [] for name in ("unit", "zone", "time", "x", "u")}
    for period in range(ZONE_PERIODS):
        u = -0.3 * x + 0.7 * rng.standard_normal(n)
        for name, values in (
            ("unit", np.arange(n)),
            ("zone", np.arange(n) // ZONE_SIZE),
            ("time", np.full(n, period)),
            ("x", x),
            ("u", u),
        ):
            rows[name].append(values)
        x = step(x, u)
    frame = {name: np.concatenate(values) for name, values in rows.items()}
    return Panel.from_frame(frame, unit="unit", time="time", seed=0)


def _zone_mean_cost(schedule: np.ndarray) -> float:
    """The plan's cost over its steps, terminal term left out, read on each zone's mean state when
    every unit follows ``schedule``, from the law the logger leaves the zones' means at. Every
    unit's lever moves its zone's mean by ``0.5 + SPILL``, so the mean follows
    ``x' = 0.8 x + 0.9 a + N(0, 0.01 / 4)``, and under the logger
    ``x' = 0.53 x + 0.63 eta + N(0, 0.01 / 4)``, ``eta`` of variance 1/4. Checked against a
    simulation of 500 000 zones at the schedule the plan below makes, 0.238700 (SE 0.000383)
    against 0.238769."""
    channel = 0.5 + SPILL
    loop = 0.8 - 0.3 * channel
    variance = (channel**2 * 0.49 + 0.01) / ZONE_SIZE / (1.0 - loop**2)
    mean, total = 0.0, 0.0
    for action in schedule:
        total += 0.5 * ((mean - ZONE_GOAL) ** 2 + variance) + 0.5 * ZONE_PRICE * action**2
        mean, variance = 0.8 * mean + channel * action, 0.64 * variance + 0.01 / ZONE_SIZE
    return total


def test_a_plan_made_on_groups_is_read_at_its_groups_mean_cost() -> None:
    """A plan made on the zones' means is evaluated on 40 later panels of 150 zones at
    ``min_effective=30``, where the certificate accepts the zones' windows (at the default 100 it
    refuses every panel), and ``model_error=0``, so the interval is the bootstrap's alone. Its
    value, less the normalised weights' first-order bias it reports, centres on the zones' mean
    cost within three Monte Carlo standard errors, and its interval covers that cost at least
    ``0.85 - 3 sqrt(0.85 * 0.15 / 40) = 0.68`` of the time. On 194 panels the value was 0.23404
    (MC SE 0.00186) against 0.23877, the reported bias -0.0037 a panel, and the interval covered
    0.861 (0.025): the bounds' false-red rates are 0.6 % at that centre and 0.4 % at a coverage of
    0.85. Up to 1.1.0 the units were read one by one: the value was 0.145, and the interval covered
    1 of 40."""
    plan = prescribe(
        _zone_panel(0),
        levers=[Lever("u", lo=-2.0, hi=2.0, unit_cost=ZONE_PRICE)],
        target=Target("x", value=ZONE_GOAL),
        adjustment=CausalGraph.from_edges([("u", "x")]),
        horizon=4,
        dt=1.0,
        integrator="euler",
        clusters="zone",
    )
    truth = _zone_mean_cost(np.asarray(plan.schedule.magnitudes).ravel())
    centred, covered = [], 0
    for panel in range(ZONE_PANELS):
        try:
            evaluation = plan.evaluate(
                _zone_panel(1000 + panel),
                seed=panel,
                model_error=0.0,
                min_effective=30.0,
                n_resamples=100,
            )
        except InfeasibleEvaluation:
            continue
        assert evaluation.normalisation_bias is not None
        centred.append(evaluation.value - evaluation.normalisation_bias)
        covered += evaluation.interval[0] <= truth <= evaluation.interval[1]
    errors = np.asarray(centred) - truth
    assert abs(errors.mean()) <= 3.0 * errors.std(ddof=1) / math.sqrt(errors.size)
    assert covered / ZONE_PANELS >= 0.85 - 3.0 * math.sqrt(0.85 * 0.15 / ZONE_PANELS)
