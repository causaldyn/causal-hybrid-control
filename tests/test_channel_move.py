"""The one-step channel's move, read off a log's dither, and its price in a plan's own cost.

The identities are ``validation/dither_channel_move.mac``'s and ``proofs/dither_channel_move.v``'s.
Here they are checked on logs: a plant whose model has the drift wrong, a policy that reads the
state, and noise that is Laplace and grows with the state, so that everything but the dither
conspires against the estimate. The last tests read the move of a real plan's channel and price it
with the plan's own :meth:`chc.plan.CausalPlan.decision_weight`.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chc import DecisionLog, LinearConstraint, QuadraticCost, causal_plan, channel_move
from chc.dynamics import LinearDynamics
from chc.gate import ChannelMove
from chc.integrate import rk4_step
from chc.plan import DecisionWeight

SIGMA = np.array([0.4, 0.8])  # the two actions' dither scales
MOVE = np.array([[0.15, -0.05], [0.0, 0.1]])  # the plant's one-step channel less the model's
DRIFT_ERROR = np.array([[-0.1, 0.1], [-0.1, 0.1]])  # the plant's drift less the model's
POLICY = np.array([[-0.5, 0.1], [0.2, -0.3]])  # the nominal action reads the state
WEIGHT = np.array(
    [[2.0, 0.3, 0.0, 0.1], [0.3, 1.0, -0.2, 0.0], [0.0, -0.2, 1.5, 0.4], [0.1, 0.0, 0.4, 0.8]]
)


def _log(action: np.ndarray, dither: np.ndarray) -> DecisionLog:
    size = action.shape[0]
    return DecisionLog(action, np.ones(size), np.zeros(size, dtype=bool), dither)


def _plant(
    paths: int, decisions: int, rng: np.random.Generator, move: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per decision and path: the action as applied, its dither and the residual against a model
    whose drift is off by ``DRIFT_ERROR`` and an offset of 0.1, under ``u = POLICY x + dither``.
    ``move`` holds the channel's move per decision, ``MOVE`` throughout by default. The noise is
    Laplace with a standard deviation of ``0.1 + 0.2 |x|``, so it moves with the state."""
    moves = np.broadcast_to(MOVE if move is None else move, (decisions, 2, 2))
    x = rng.normal(0.0, 0.5, (paths, 2))
    action, drawn, residual = (np.empty((decisions, paths, 2)) for _ in range(3))
    for t in range(decisions):
        drawn[t] = SIGMA * rng.standard_normal((paths, 2))
        action[t] = x @ POLICY.T + drawn[t]
        scale = (0.1 + 0.2 * np.abs(x)) / math.sqrt(2.0)
        residual[t] = x @ DRIFT_ERROR.T + 0.1 + action[t] @ moves[t].T + rng.laplace(scale=scale)
        x = 0.6 * x + residual[t] + 0.3 * action[t]
    return action, drawn, residual


def _moves(
    plant: tuple[np.ndarray, np.ndarray, np.ndarray], forgetting: float = 1.0
) -> list[ChannelMove]:
    action, drawn, residual = plant
    return [
        channel_move(
            _log(action[:, p], drawn[:, p]),
            residual[:, p],
            dither_scale=SIGMA,
            forgetting=forgetting,
        )
        for p in range(action.shape[1])
    ]


def test_a_decision_log_reads_as_the_docstring_writes_it() -> None:
    """Three decisions, two states, one action, forgetting at a half: the products, their weighted
    mean and the scaled weighted spread, transcribed entry by entry."""
    drawn = np.array([0.3, -0.5, 0.2])
    residual = np.array([[0.4, -1.0], [0.1, 0.6], [-0.2, 0.3]])
    moved = channel_move(_log(drawn + 0.1, drawn), residual, dither_scale=0.5, forgetting=0.5)
    w = np.array([0.25, 0.5, 1.0])
    z = residual * (drawn / 0.5)[:, None] / 0.5
    mean = w @ z / w.sum()
    others = np.array([w.sum() - wt for wt in w])
    others_squared = np.array([np.sum(w**2) - wt**2 for wt in w])
    scale = np.sum(w**2) / np.sum(w**2 * (others**2 + others_squared))
    spread = sum(w[t] ** 2 * np.outer(z[t] - mean, z[t] - mean) for t in range(3)) * scale
    np.testing.assert_allclose(moved.estimate, mean[:, None], rtol=1e-13)
    np.testing.assert_allclose(moved.covariance, spread, rtol=1e-12)
    assert moved.effective_size == pytest.approx(w.sum() ** 2 / np.sum(w**2), rel=1e-13)
    with pytest.raises(ValueError, match="read-only"):
        moved.estimate[0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        moved.covariance[0, 0] = 1.0


def test_the_move_is_read_without_bias_whatever_the_model_gets_wrong() -> None:
    """Entry by entry, with and without forgetting, the estimates average to the move: the drift
    error, the policy's action and the noise add to the product's spread and not to its mean."""
    plant = _plant(1500, 300, np.random.default_rng(30))
    for forgetting in (1.0, 0.98):
        estimates = np.array([move.estimate for move in _moves(plant, forgetting)])
        z = (estimates.mean(axis=0) - MOVE) / (estimates.std(axis=0) / math.sqrt(len(estimates)))
        assert np.abs(z).max() < 4.0, forgetting


@pytest.mark.parametrize("forgetting", [1.0, 0.98])
def test_the_covariance_is_the_estimates_spread_over_logs(forgetting: float) -> None:
    """The average covariance against the estimates' own over 1500 logs: variances within 12%, and
    correlations within 0.08, with noise that moves with the state and weights that are not
    equal."""
    moves = _moves(_plant(1500, 300, np.random.default_rng(31)), forgetting)
    estimates = np.array([move.estimate.ravel() for move in moves])
    spread = np.cov(estimates, rowvar=False)
    covariance = np.mean([move.covariance for move in moves], axis=0)
    np.testing.assert_allclose(np.diag(covariance) / np.diag(spread), 1.0, atol=0.12)

    def correlation(m: np.ndarray) -> np.ndarray:
        root = np.sqrt(np.diag(m))
        return m / np.outer(root, root)

    np.testing.assert_allclose(correlation(covariance), correlation(spread), atol=0.08)


def test_the_covariance_is_unbiased_for_products_that_share_one() -> None:
    """With i.i.d. products the scaling is exact at any weights: over 40 000 logs of eight decisions
    forgotten at 0.7, the average covariance meets the estimates' spread to Monte Carlo error. The
    rough ``kish / (kish - 1)`` scaling reads it 13% low here."""
    rng = np.random.default_rng(32)
    logs, size, forgetting = 40_000, 8, 0.7
    drawn = rng.standard_normal((logs, size))
    residual = 0.5 + 1.0 * drawn + rng.standard_normal((logs, size))
    moves = [
        channel_move(_log(drawn[p], drawn[p]), residual[p], dither_scale=1.0, forgetting=forgetting)
        for p in range(logs)
    ]
    estimates = np.array([move.estimate[0, 0] for move in moves])
    covariance = np.array([move.covariance[0, 0] for move in moves])
    ratio = covariance.mean() / estimates.var(ddof=1)
    assert ratio == pytest.approx(1.0, abs=4 * math.sqrt(2 / logs) + 0.01)
    kish = moves[0].effective_size
    w = forgetting ** np.arange(size - 1, -1, -1)
    rough = [
        np.sum(w**2 * (z - np.sum(w * z) / w.sum()) ** 2) / w.sum() ** 2 * kish / (kish - 1)
        for z in residual * drawn
    ]
    assert np.mean(rough) / estimates.var(ddof=1) < 0.95


@pytest.mark.parametrize(
    ("move", "wide"),
    [(MOVE, (0.85, 1.15)), (np.zeros((2, 2)), (1.0, 1.45))],
    ids=["moved", "unmoved"],
)
def test_the_keep_price_is_unbiased_and_replanning_costs_the_estimates_noise(
    move: np.ndarray, wide: tuple[float, float]
) -> None:
    """Over 1500 logs: the keep price averages ``d' W d / 2``, and the re-plan price the weighted
    square of the estimates' own error, ``E[(dh - d)' W (dh - d)] / 2``. The keep price's error
    reads 1.04 of its spread over the logs when the channel moved, and 1.31 when it did not, where
    the floor on ``d' W S W d`` leans it wide."""
    moves = _moves(_plant(1500, 300, np.random.default_rng(33), move))
    weight = DecisionWeight(WEIGHT, (2, 2), free=4, weakly_active=0, residual=0.0)
    prices = [moved.price(weight) for moved in moves]
    keep = np.array([price.keep for price in prices])
    target = move.ravel() @ WEIGHT @ move.ravel() / 2
    assert abs(keep.mean() - target) < 4 * keep.std() / math.sqrt(len(keep))
    errors = np.array([moved.estimate.ravel() - move.ravel() for moved in moves])
    realised = np.einsum("pi,ij,pj->p", errors, WEIGHT, errors) / 2
    replan = np.array([price.replan for price in prices])
    assert replan.mean() == pytest.approx(realised.mean(), rel=0.1)
    ratio = np.mean([price.keep_error for price in prices]) / keep.std()
    assert wide[0] <= ratio <= wide[1]


@pytest.mark.parametrize(("forgetting", "size"), [(1.0, 7), (0.9, 5), (0.99, 1000), (0.5, 60)])
def test_forgetting_is_worth_what_the_derivation_says(forgetting: float, size: int) -> None:
    """STEP 3a: ``(1 + l)(1 - l^n) / ((1 - l)(1 + l^n))``, and ``n`` without forgetting."""
    drawn = np.random.default_rng(34).standard_normal(size)
    moved = channel_move(_log(drawn, drawn), drawn + 0.3, dither_scale=1.0, forgetting=forgetting)
    expected = (
        size
        if forgetting == 1.0
        else (1 + forgetting) * (1 - forgetting**size) / ((1 - forgetting) * (1 + forgetting**size))
    )
    assert moved.effective_size == pytest.approx(expected, rel=1e-12)


def test_an_estimate_that_forgets_follows_a_channel_that_moved() -> None:
    """The channel moves by ``MOVE`` after 300 of 400 decisions. Without forgetting the estimate
    averages the whole log, a quarter of the move; forgetting at 0.98 it averages the weights'
    share of the decisions since, 0.87 of it. Each lands on its own target."""
    decisions, change = 400, 300
    path = np.where((np.arange(decisions) >= change)[:, None, None], MOVE, 0.0)
    plant = _plant(1500, decisions, np.random.default_rng(35), path)
    for forgetting, share in ((1.0, 0.25), (0.98, 1 - 0.98**100)):
        estimates = np.array([move.estimate for move in _moves(plant, forgetting)])
        z = (estimates.mean(axis=0) - share * MOVE) / (
            estimates.std(axis=0) / math.sqrt(len(estimates))
        )
        assert np.abs(z).max() < 4.0, forgetting


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"forgetting": 0.0}, r"forgetting must lie in \(0, 1\], got 0.0"),
        ({"forgetting": 1.5}, r"forgetting must lie in \(0, 1\]"),
        ({"forgetting": math.nan}, r"forgetting must lie in \(0, 1\]"),
        ({"forgetting": 1e-200}, "leaves all the weight on the last decision"),
        ({"dither_scale": -0.3}, "dither_scale must be positive"),
        ({"dither_scale": np.ones(2)}, r"dither_scale must be a scalar or of shape \(1,\)"),
        ({"dither_scale": 0.09}, "dither of action 0"),
        ({"residual": np.ones(39)}, r"residual must have shape \(40,\) or \(40, states\)"),
        ({"residual": np.full(40, np.inf)}, "residual is not finite"),
    ],
)
def test_a_move_outside_the_contract_is_refused(change: dict[str, object], match: str) -> None:
    drawn = 0.3 * np.random.default_rng(36).standard_normal(40)
    arguments: dict[str, object] = {"residual": np.zeros(40), "dither_scale": 0.3} | change
    residual = arguments.pop("residual")
    with pytest.raises(ValueError, match=match):
        channel_move(_log(drawn, drawn), residual, **arguments)  # type: ignore[arg-type]


def test_a_log_that_cannot_carry_the_estimate_is_refused() -> None:
    drawn = 0.3 * np.random.default_rng(37).standard_normal(40)
    log = _log(drawn, drawn)
    with pytest.raises(ValueError, match="1 decisions: the estimate's covariance needs two"):
        channel_move(_log(drawn[:1], drawn[:1]), np.zeros(1), dither_scale=0.3)
    with pytest.raises(ValueError, match=r"\(the first is decision 7\)"):
        channel_move(
            dataclasses.replace(log, saturated=np.arange(40) == 7), np.zeros(40), dither_scale=0.3
        )
    with pytest.raises(ValueError, match="records no dither"):
        channel_move(dataclasses.replace(log, dither=None), np.zeros(40), dither_scale=0.3)


def test_a_price_needs_a_weight_on_the_moves_channel() -> None:
    drawn = np.random.default_rng(38).standard_normal(40)
    moved = channel_move(_log(drawn, drawn), np.ones((40, 2)), dither_scale=1.0)
    wrong = DecisionWeight(np.eye(4), (1, 4), free=4, weakly_active=0, residual=0.0)
    with pytest.raises(ValueError, match=r"of shape \(1, 4\), the move is of shape \(2, 1\)"):
        moved.price(wrong)


# ---- a real plan's channel, moved: the estimate in the entries its decision weight reads ----

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
PLAN_MOVE = np.array(
    [[0.02, -0.01], [0.015, 0.01]]
)  # the plant's one-step channel less the model's


def test_a_plans_moved_channel_is_read_in_the_entries_its_weight_prices() -> None:
    """The budgeted plan of ``tests/test_decision_weight.py``, run 40 times over with a dither on a
    plant whose one-step map is the model's RK4 step plus ``PLAN_MOVE u``, a drift the model does
    not have and noise. Over 400 such logs the estimate lands on ``PLAN_MOVE`` entry by entry, and
    the keep price averages ``DecisionWeight.regret(PLAN_MOVE)``: the two read the channel in one
    order and one unit."""
    plan = causal_plan(
        MODEL, START, COST, DT, HORIZON, -BOX, BOX, constraints=(BUDGET,), steps=20_000
    )
    weight = plan.decision_weight()
    phi, gamma = (
        np.asarray(m, dtype=np.float64)
        for m in jax.jacfwd(lambda x, u: rk4_step(MODEL, 0.0, x, u, DT), argnums=(0, 1))(
            START, jnp.zeros(2)
        )
    )
    nominal = np.asarray(plan.actions, dtype=np.float64)
    rng = np.random.default_rng(39)
    logs, runs, sigma = 400, 40, np.array([0.3, 0.2])
    estimates, keep = [], []
    for _ in range(logs):
        drawn = sigma * rng.standard_normal((runs, HORIZON, 2))
        action = nominal + drawn
        x = np.broadcast_to(np.asarray(START, dtype=np.float64), (runs, 2))
        residual = np.empty((runs, HORIZON, 2))
        for t in range(HORIZON):
            model_step = x @ phi.T + action[:, t] @ gamma.T
            after = (
                model_step
                + action[:, t] @ PLAN_MOVE.T
                + 0.05 * np.tanh(x)
                + 0.02 * rng.standard_normal((runs, 2))
            )
            residual[:, t] = after - model_step
            x = after
        moved = channel_move(
            _log(action.reshape(-1, 2), drawn.reshape(-1, 2)),
            residual.reshape(-1, 2),
            dither_scale=sigma,
        )
        estimates.append(moved.estimate)
        keep.append(moved.price(weight).keep)
    estimates_array = np.array(estimates)
    z = (estimates_array.mean(axis=0) - PLAN_MOVE) / (estimates_array.std(axis=0) / math.sqrt(logs))
    assert np.abs(z).max() < 4.0
    keep_array = np.array(keep)
    target = weight.regret(PLAN_MOVE)
    assert target > 0
    assert abs(keep_array.mean() - target) < 4 * keep_array.std() / math.sqrt(logs)
