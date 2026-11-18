"""The capped exploration's stopping mass is the discrete optimum, not the constant-cap relaxation.

``validation/capped_exploration_schedule.mac`` STEP 9. Along the greedy fill the cost reads the
INTEGER round that delivers the mass. The constant-cap quadratic
``A w^2 + (K/cap) w = K c T + K I0/cap`` reads that round as ``mass/cap``, a continuous
relaxation, so its root is not the optimum. At ``A = 1``, ``K = 4``, ``c = 1``, ``I0 = 1`` and
``T = 3`` the exact masses are closed forms, and on short horizons a brute-force minimum over the
box of per-round actions checks the rest.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy.optimize import minimize

from chc.regret import capped_exploration_policy

# A = b^2 + rr = 1, K = A (du*/db)^2 = 4 with du*/db = -xt (rr - b^2)/(rr + b^2)^2 = -2,
# c = eta/sigma^2 = 1, I0 = 1: every one exact in binary
UNIT = {"b": 0.5, "rr": 0.75, "xt": 4.0, "sigma": 1.0, "eta": 1.0, "i0": 1.0}


def _constants(b: float, rr: float, xt: float, sigma: float, eta: float, i0: float) -> tuple:
    a = b * b + rr
    sensitivity = -xt * (rr - b * b) / (rr + b * b) ** 2
    return a, a * sensitivity * sensitivity, eta / sigma**2, i0


def _relaxed_mass(horizon: int, cap: float, a: float, k: float, c: float, i0: float) -> float:
    q = k / cap
    w = (-q + math.sqrt(q * q + 4.0 * a * (k * c * horizon + k * i0 / cap))) / (2.0 * a)
    return (w - i0) / c


def _cost(v: np.ndarray, a: float, k: float, c: float, i0: float) -> np.ndarray:
    """The regret of each row of ``v``: ``A sum_t v_t + K sum_t 1/(I0 + c S_(t-1))``."""
    before = np.cumsum(v, axis=-1) - v
    return a * v.sum(axis=-1) + (k / (i0 + c * before)).sum(axis=-1)


def _gradient(v: np.ndarray, a: float, k: float, c: float, i0: float) -> np.ndarray:
    weights = k / (i0 + c * (np.cumsum(v) - v)) ** 2  # round t's weight; v_j moves every t > j
    later = np.concatenate([np.cumsum(weights[::-1])[::-1][1:], [0.0]])
    return a - c * later


def _fill_cost(mass: float, caps: np.ndarray, a: float, k: float, c: float, i0: float) -> float:
    fill = np.diff(np.concatenate([[0.0], np.minimum(np.cumsum(caps), mass)]))
    return float(_cost(fill, a, k, c, i0))


def _box_minimum(caps: np.ndarray, a: float, k: float, c: float, i0: float) -> tuple[float, float]:
    """The least regret over ``0 <= v <= caps`` and the mass that attains it, by brute force.

    Every point of a grid of nine levels per round is priced; the five cheapest are polished by a
    bounded quasi-Newton descent on the full program, which assumes neither a greedy fill nor a
    stopping round.
    """
    grid = np.array(list(itertools.product(*(np.linspace(0.0, cap, 9) for cap in caps))))
    costs = _cost(grid, a, k, c, i0)
    polished = [
        minimize(
            lambda v: float(_cost(v, a, k, c, i0)),
            seed,
            jac=lambda v: _gradient(v, a, k, c, i0),
            method="L-BFGS-B",
            bounds=[(0.0, float(cap)) for cap in caps],
            options={"ftol": 1e-15, "gtol": 1e-13, "maxiter": 10_000},
        )
        for seed in grid[np.argsort(costs)[:5]]
    ]
    best = min(polished, key=lambda result: result.fun)
    assert best.fun <= costs.min()
    return float(best.fun), float(np.sum(best.x))


def test_a_loose_cap_stops_inside_round_one_below_the_relaxed_root() -> None:
    # STEP 9g: at cap 100 round 1 balances, A (I0 + c S)^2 = K c (T - 1) = 8, far inside the cap
    a, k, c, i0 = _constants(**UNIT)
    policy = capped_exploration_policy(horizon=3, cap=100.0, **UNIT)
    exact = math.sqrt(8.0) - 1.0
    assert policy.predicted_mass == pytest.approx(exact, rel=1e-12, abs=0.0)
    assert policy.exploration_mass == pytest.approx(exact, rel=1e-12, abs=0.0)
    assert policy.block_rounds == 1
    assert policy.cost == pytest.approx(3.0 + 4.0 * math.sqrt(2.0), rel=1e-12, abs=0.0)
    # the relaxation's root and the cap-free form, to Maxima's bfloat digits, are well above it,
    # and delivering the relaxation's mass costs more
    relaxed = _relaxed_mass(3, 100.0, a, k, c, i0)
    assert relaxed == pytest.approx(2.449927953142543, rel=1e-12, abs=0.0)
    assert math.sqrt(12.0) - 1.0 == pytest.approx(2.464101615137755, rel=1e-12, abs=0.0)
    assert relaxed - policy.predicted_mass > 0.6
    assert _fill_cost(relaxed, np.full(3, 100.0), a, k, c, i0) > policy.cost + 0.1


def test_a_tight_cap_stops_inside_round_two_below_the_relaxed_root() -> None:
    # STEP 9h: at cap 4/5 round 2 balances, A (I0 + c S)^2 = K c (T - 2) = 4, so the mass is 1
    a, k, c, i0 = _constants(**UNIT)
    policy = capped_exploration_policy(horizon=3, cap=0.8, **UNIT)
    assert policy.predicted_mass == pytest.approx(1.0, rel=1e-12, abs=0.0)
    assert policy.exploration_mass == pytest.approx(1.0, rel=1e-12, abs=0.0)
    assert policy.block_rounds == 2
    assert policy.last_round == pytest.approx(0.2, rel=1e-12, abs=0.0)  # the schedule (4/5, 1/5, 0)
    assert policy.cost == pytest.approx(83.0 / 9.0, rel=1e-12, abs=0.0)
    relaxed = _relaxed_mass(3, 0.8, a, k, c, i0)
    assert relaxed == pytest.approx(1.321825380496478, rel=1e-12, abs=0.0)
    assert _fill_cost(relaxed, np.full(3, 0.8), a, k, c, i0) > policy.cost + 0.01


def test_a_balance_that_skips_a_round_boundary_stops_on_the_boundary() -> None:
    # STEP 9i: at cap 3/2 round 1 balances at sqrt(8) - 1 > 3/2 and round 2 at 1 < 3/2. The cost
    # falls up to 3/2 and rises after it, so the optimum is the boundary; the balance of round 2,
    # 1, is below it and not optimal
    a, k, c, i0 = _constants(**UNIT)
    policy = capped_exploration_policy(horizon=3, cap=1.5, **UNIT)
    assert policy.predicted_mass == pytest.approx(1.5, rel=1e-12, abs=0.0)
    assert policy.exploration_mass == pytest.approx(1.5, rel=1e-12, abs=0.0)
    assert policy.cost == pytest.approx(8.7, rel=1e-12, abs=0.0)
    assert _fill_cost(1.0, np.full(3, 1.5), a, k, c, i0) > policy.cost + 0.01


SETTINGS = (
    UNIT,  # A = 1, K = 4, c = 1, I0 = 1
    {"b": 1.0, "rr": 2.0, "xt": 6.0, "sigma": 0.8, "eta": 0.5, "i0": 0.5},
    {"b": 0.5, "rr": 0.75, "xt": 10.0, "sigma": 1.0, "eta": 2.0, "i0": 2.0},  # K = 25, c = 2
)
SCHEDULES = (
    (0.7,),
    (100.0, 100.0, 100.0),
    (0.8, 0.8, 0.8),
    (1.5, 1.5, 1.5),
    (0.0, 1.0, 2.0),
    (2.0, 0.5, 0.0, 1.0),
    (0.4, 0.4, 0.4, 0.4, 0.4),
    (0.3, 0.6, 1.2, 2.4, 4.8),
    (3.0, 1.0, 0.5, 0.25, 0.1),
)


@pytest.mark.parametrize("setting", range(len(SETTINGS)))
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_the_stopping_mass_is_the_brute_force_optimum(
    schedule: tuple[float, ...], setting: int
) -> None:
    params = SETTINGS[setting]
    a, k, c, i0 = _constants(**params)
    caps = np.array(schedule)
    horizon = caps.size
    least, mass = _box_minimum(caps, a, k, c, i0)
    policy = capped_exploration_policy(horizon=horizon, cap=list(schedule), **params)
    assert policy.cost == pytest.approx(least, rel=1e-10, abs=0.0)
    assert policy.exploration_mass == pytest.approx(mass, rel=0.0, abs=1e-6)
    assert policy.predicted_mass == pytest.approx(mass, rel=0.0, abs=1e-6)
    assert _fill_cost(policy.predicted_mass, caps, a, k, c, i0) == pytest.approx(
        least, rel=1e-10, abs=0.0
    )
    # a budget below that mass is spent in full, and one above it changes nothing
    if policy.predicted_mass > 0.0:
        for budget in (0.5 * policy.predicted_mass, 2.0 * policy.predicted_mass):
            bounded = capped_exploration_policy(
                horizon=horizon, cap=list(schedule), budget=budget, **params
            )
            assert bounded.exploration_mass == pytest.approx(
                min(budget, policy.predicted_mass), rel=1e-12, abs=0.0
            )
