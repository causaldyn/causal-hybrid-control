"""The tube carries the plant's noise as a VAR(1), each direction persisting at its own rate
(ADR 0105).

The tube read the noise's persistence as one lag-1 correlation, pooled over the states with each
counted alike in its own units. On the quickstart's plant with its demand an AR(0.8), the supply's
one-step error persists at 0.80 and the wait's at 0.02, and the pooled correlation read 0.41: the
plant left the tube from the second week on, 0.79 a week by the fifteenth. The VAR(1) coefficient
``Phi``, Yule--Walker's ``Gamma_1 S^-1``, carries ``Cov(w_{k+1}, w_k) = Phi S`` through
``C_{k+1} = (M C_k + S) Phi'`` (``validation/var_noise_tube.mac``).
"""

from __future__ import annotations

import logging
import math

import jax.numpy as jnp
import numpy as np
import pytest

from chc import CausalGraph, Constraint, Lever, Panel, Target, prescribe
from chc.cost import QuadraticCost
from chc.decision import _autoregression
from chc.dynamics import LinearDynamics
from chc.plan import causal_plan
from chc.uncertainty import _noise_covariances, _noise_radius, _noise_tube

_DT = 0.1


def _var_noise(phi: np.ndarray, covariance: np.ndarray, units: int, periods: int, seed: int):
    """``units`` stationary VAR(1) series of ``periods`` steps, ``Cov(w) = covariance``."""
    rng = np.random.default_rng(seed)
    innovation = covariance - phi @ covariance @ phi.T
    root, start = np.linalg.cholesky(innovation), np.linalg.cholesky(covariance)
    w = rng.standard_normal((units, covariance.shape[0])) @ start.T
    rows = [w]
    for _ in range(periods - 1):
        w = w @ phi.T + rng.standard_normal(w.shape) @ root.T
        rows.append(w)
    error = np.stack(rows, 1).reshape(-1, covariance.shape[0])
    return error, np.repeat(np.arange(units), periods), np.tile(np.arange(periods), units)


def test_the_coefficient_reads_each_state_s_persistence() -> None:
    """On 400 units of 25 periods of a VAR(1) whose first state persists at 0.8 and pushes the
    second, which persists at 0.1, the coefficient comes back within 0.02 of each entry, where a
    pooled correlation reads one number for both. Each entry's Monte Carlo standard error over
    9600 pairs is under 0.007."""
    phi = np.array([[0.8, 0.0], [0.3, 0.1]])
    innovation = np.array([[1.0, 0.2], [0.2, 0.5]])
    covariance = innovation
    for _ in range(200):
        covariance = phi @ covariance @ phi.T + innovation
    error, units, periods = _var_noise(phi, covariance, 400, 25, 0)
    sample = error.T @ error / error.shape[0]
    np.testing.assert_allclose(
        _autoregression(error, sample, units, periods), phi, rtol=0.0, atol=0.02
    )


def test_the_coefficient_moves_with_any_basis_of_the_states() -> None:
    """Read in other units or another basis of the states, ``w -> T w``, the coefficient is
    ``T Phi T^-1``: the tube it carries is the same noise's."""
    phi = np.array([[0.7, 0.1], [0.0, 0.2]])
    error, units, periods = _var_noise(phi, np.eye(2), 50, 20, 1)
    transform = np.array([[1e3, 2.0], [-0.5, 1e-2]])
    moved = error @ transform.T
    read = _autoregression(error, error.T @ error / error.shape[0], units, periods)
    read_moved = _autoregression(moved, moved.T @ moved / moved.shape[0], units, periods)
    np.testing.assert_allclose(
        read_moved, transform @ read @ np.linalg.inv(transform), rtol=1e-8, atol=1e-10
    )


def test_a_lag_no_stationary_noise_has_is_scaled_back_to_full_persistence() -> None:
    """Pairs that read a lag-1 covariance past the noise's own, past a contraction once whitened,
    are scaled back to one: the noise persists fully along its most persistent direction, and the
    innovation's covariance ``S - Phi S Phi'`` has no negative eigenvalue. A unit that never
    steps twice leaves the coefficient unknown, and a state with no noise persists at 0."""
    error = np.array([[1.0, 0.0], [2.0, 0.0], [4.0, 0.0], [8.0, 0.0]])
    units, periods = np.zeros(4, dtype=int), np.arange(4)
    covariance = np.diag([10.0, 0.0])
    phi = _autoregression(error, covariance, units, periods)
    assert phi[1].tolist() == [0.0, 0.0]
    assert phi[:, 1].tolist() == [0.0, 0.0]
    assert phi[0, 0] == pytest.approx(1.0, rel=1e-12, abs=0.0)  # read (2 + 8 + 32) / 3 / 10 > 1
    remainder = covariance - phi @ covariance @ phi.T
    assert np.linalg.eigvalsh(remainder)[0] >= -1e-12
    lonely = _autoregression(error, covariance, np.arange(4), np.zeros(4, dtype=int))
    assert np.isnan(lonely).all()


def _double_sum(steps: np.ndarray, phi: np.ndarray, noise: np.ndarray, k: int) -> np.ndarray:
    """``Cov(e_k)`` for ``e_{j+1} = M_j e_j + w_j``, summed over every pair of the noise's steps."""

    def carry(i: int) -> np.ndarray:
        matrix = np.eye(noise.shape[0])
        for step in steps[i + 1 : k]:
            matrix = step @ matrix
        return matrix

    def cov(i: int, j: int) -> np.ndarray:
        if i >= j:
            return np.linalg.matrix_power(phi, i - j) @ noise
        return noise @ np.linalg.matrix_power(phi.T, j - i)

    return sum(carry(i) @ cov(i, j) @ carry(j).T for i in range(k) for j in range(k))


def test_the_recursion_is_the_covariance_of_the_gap_and_reduces_to_ar1() -> None:
    """``_noise_covariances`` under a VAR(1) is the gap's covariance summed over every pair of
    steps, through step matrices that move each step; at ``Phi = rho I`` it is the AR(1)'s."""
    rng = np.random.default_rng(2)
    slopes = rng.normal(0.0, 0.8, (6, 2, 2))
    noise = np.array([[0.04, 0.01], [0.01, 0.02]])
    phi = np.array([[0.7, 0.2], [-0.1, 0.3]])
    dt = 0.3
    eye = np.eye(2)
    steps = []
    for slope in slopes:
        z = dt * slope
        steps.append(eye + z + z @ z / 2 + z @ z @ z / 6 + z @ z @ z @ z / 24)
    covariances = _noise_covariances(slopes, noise, phi, dt, 6)
    for k in range(1, 7):
        np.testing.assert_allclose(
            covariances[k], _double_sum(np.asarray(steps), phi, noise, k), rtol=1e-12, atol=1e-15
        )
    scalar = _noise_covariances(slopes, noise, 0.4, dt, 6)
    matrix = _noise_covariances(slopes, noise, 0.4 * eye, dt, 6)
    for one, other in zip(scalar, matrix, strict=True):
        np.testing.assert_allclose(other, one, rtol=1e-15, atol=0.0)


def test_the_radius_holds_a_var1_noise_at_each_step() -> None:
    """Simulated, the gap of a plant whose noise is the VAR(1) stays inside the radius at each of
    eight steps on 0.95 of 40,000 runs, within four Monte Carlo standard errors of 0.0011 each;
    the pooled correlation's AR(1) radius at its own read holds less where one state persists."""
    phi = np.array([[0.85, 0.0], [0.0, 0.0]])
    noise = np.diag([1.0, 1.0])
    state = np.array([[-0.2, 0.05], [0.0, -0.4]])
    dt, horizon, draws = 0.5, 8, 40_000
    radii = _noise_tube(state, noise, phi, dt, horizon)
    z = dt * state
    step = np.eye(2) + z + z @ z / 2 + z @ z @ z / 6 + z @ z @ z @ z / 24
    rng = np.random.default_rng(3)
    innovation = np.linalg.cholesky(noise - phi @ noise @ phi.T)
    w = rng.standard_normal((draws, 2))
    gap = np.zeros((draws, 2))
    inside = []
    for k in range(1, horizon + 1):
        gap = gap @ step.T + w
        inside.append(float(np.mean(np.linalg.norm(gap, axis=1) <= radii[k])))
        w = w @ phi.T + rng.standard_normal((draws, 2)) @ innovation.T
    error = math.sqrt(0.95 * 0.05 / draws)
    np.testing.assert_allclose(inside, 0.95, rtol=0.0, atol=4.0 * error)
    pooled = _noise_tube(state, noise, 0.425, dt, horizon)  # each state alike: (0.85 + 0) / 2
    assert np.all(pooled[2:] < radii[2:])
    assert radii[1] == pytest.approx(_noise_radius(noise), rel=1e-12, abs=0.0)


def test_a_norm_bound_carries_the_coefficient_by_its_largest_singular_value() -> None:
    """Under a norm bound the cross term is carried by ``||Phi||``, which bounds ``(M C + S) Phi'``
    as ``|rho|`` bounds the AR(1)'s."""
    noise = np.array([[0.02, 0.005], [0.005, 0.01]])
    phi = np.array([[0.5, 0.3], [0.0, 0.2]])
    norm = float(np.linalg.norm(phi, 2))
    np.testing.assert_allclose(
        _noise_tube(np.array(0.4), noise, phi, 0.5, 6),
        _noise_tube(np.array(0.4), noise, norm, 0.5, 6),
        rtol=1e-15,
        atol=0.0,
    )


def test_causal_plan_takes_a_coefficient_of_a_stationary_noise_and_refuses_others() -> None:
    """A coefficient whose innovation ``S - Phi S Phi'`` has a negative eigenvalue is no stationary
    noise's, and one of another size is no state's."""
    model = LinearDynamics(jnp.array([[0.0, 1.0], [-1.0, -0.3]]), jnp.array([[0.0], [1.0]]))
    cost = QuadraticCost(Q=jnp.eye(2), R=0.1 * jnp.eye(1), Qf=jnp.eye(2), x_target=jnp.zeros(2))
    noise = np.diag([0.01, 0.02])

    def plan(persistence: object):
        return causal_plan(
            model,
            jnp.ones(2),
            cost,
            0.2,
            6,
            -1.0,
            1.0,
            lipschitz=np.asarray(model.a_matrix),
            noise=noise,
            noise_persistence=persistence,
            tolerance=1.0,
        )

    held = plan(np.array([[0.6, 0.0], [0.0, 0.1]]))
    assert held.uncertainty_tube is not None
    with pytest.raises(ValueError, match="stationary noise"):
        plan(np.array([[1.2, 0.0], [0.0, 0.0]]))
    with pytest.raises(ValueError, match="VAR"):
        plan(np.eye(3) * 0.5)
    with pytest.raises(ValueError, match="correlation"):
        plan(1.5)


def _quickstart_log(
    seed: int, phi: float, disturbance: float = 0.0, carry: float = 0.0
) -> dict[str, np.ndarray]:
    """The quickstart's log with its demand an AR(``phi``) within each region, stationary, and an
    unlogged AR(``carry``) disturbance of standard deviation ``disturbance`` on supply's rate."""
    rng = np.random.default_rng(seed)
    names = ("region", "week", "supply", "wait", "incentive", "demand")
    rows: dict[str, list[float]] = {name: [] for name in names}
    for region in range(200):
        supply, wait = rng.normal(0.0, 0.2), rng.normal(0.0, 0.2)
        demand = rng.normal()
        unlogged = disturbance * rng.normal()
        for week in range(12):
            incentive = 0.9 * demand + rng.normal(0.0, 0.5)
            for name, value in zip(
                names, (region, week, supply, wait, incentive, demand), strict=True
            ):
                rows[name].append(value)
            supply, wait = (
                supply
                + _DT * (-0.6 * supply + 0.3 * wait + 0.8 * incentive + 1.5 * demand + unlogged)
                + rng.normal(0.0, 0.01),
                wait + _DT * (0.25 * wait - 0.4 * incentive) + rng.normal(0.0, 0.01),
            )
            demand = phi * demand + math.sqrt(1.0 - phi**2) * rng.normal()
            unlogged = carry * unlogged + disturbance * math.sqrt(1.0 - carry**2) * rng.normal()
    return {name: np.asarray(values) for name, values in rows.items()}


_EDGES = [
    ("demand", "incentive"),
    ("demand", "supply"),
    ("incentive", "supply"),
    ("incentive", "wait"),
    ("wait", "supply"),
]


def _decide(logs: dict[str, np.ndarray]):
    return prescribe(
        Panel.from_frame(logs, unit="region", time="week", seed=0),
        adjustment=CausalGraph.from_edges(_EDGES),
        levers=[Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05)],
        target=Target("supply", value=1.0),
        constraints=[Constraint("wait", hi=0.5)],
        horizon=15,
        dt=_DT,
        tolerance=0.5,
    )


def test_a_disturbance_that_persists_unlogged_is_warned_of(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unlogged AR(0.9) disturbance of standard deviation 2 on supply's rate is read in part as
    supply's own drift, so the one-step error on the log is smaller than the plant's and the
    noise the tube reads may not be the plant's. The channel's residual persists (p below 1e-20),
    and a ``tube`` warning and the report say so. On the same log without the disturbance the
    check reads p = 0.63 and nothing is warned of; on such logs it rejects at about its size, 3
    of 40 at 0.05."""

    def warned(records: list[logging.LogRecord]) -> list[logging.LogRecord]:
        return [
            r
            for r in records
            if getattr(r, "chc_event", None) == "tube" and "noise_persistence_p" in r.__dict__
        ]

    with caplog.at_level(logging.WARNING, logger="chc.decision"):
        decided = _decide(_quickstart_log(34, 0.0, disturbance=2.0, carry=0.9))
    certificate = decided.certificate
    assert certificate.noise_persistence_p is not None
    assert certificate.noise_persistence_p < 1e-20
    assert certificate.certified_horizon is not None
    assert len(warned(caplog.records)) == 1
    assert "the noise the tube reads may not be the plant's" in decided.report()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="chc.decision"):
        clean = _decide(_quickstart_log(34, 0.0))
    assert clean.certificate.noise_persistence_p is not None
    assert clean.certificate.noise_persistence_p > 0.05
    assert not warned(caplog.records)


def test_the_quickstart_s_tube_holds_a_persistent_demand_at_each_week() -> None:
    """With the quickstart's demand an AR(0.8), run on from the plan's start under its actions,
    the demand drawn from its own AR(0.8), the plant stays inside the tube at each of 15 weeks on
    0.95 of 4000 runs within four Monte Carlo standard errors of 0.0034, where the pooled
    correlation's tube held 0.956 at the first week and 0.78 at the fifteenth. The coefficient the
    certificate reports reads the supply's persistence near 0.8 and the wait's near 0."""
    decided = _decide(_quickstart_log(0, 0.8))
    plan, certificate = decided.plan, decided.certificate
    assert plan is not None
    assert plan.uncertainty_tube is not None
    assert certificate.tube_autoregression is not None
    phi = np.asarray(certificate.tube_autoregression)
    assert phi[0, 0] == pytest.approx(0.8, abs=0.05)
    assert abs(phi[1, 1]) < 0.1
    assert certificate.tube_persistence == pytest.approx(0.41, abs=0.05)
    tube, planned = np.asarray(plan.uncertainty_tube), np.asarray(plan.trajectory)
    rng = np.random.default_rng(12)
    draws = 4000
    supply = np.full(draws, planned[0, 0])
    wait = np.full(draws, planned[0, 1])
    demand = rng.standard_normal(draws)
    inside = []
    for k, action in enumerate(np.asarray(plan.actions)[:, 0]):
        supply, wait = (
            supply
            + _DT * (-0.6 * supply + 0.3 * wait + 0.8 * action + 1.5 * demand)
            + rng.normal(0.0, 0.01, draws),
            wait + _DT * (0.25 * wait - 0.4 * action) + rng.normal(0.0, 0.01, draws),
        )
        inside.append(
            np.mean(np.hypot(supply - planned[k + 1, 0], wait - planned[k + 1, 1]) <= tube[k + 1])
        )
        demand = 0.8 * demand + 0.6 * rng.standard_normal(draws)
    error = math.sqrt(0.95 * 0.05 / draws)
    assert min(inside) >= 0.95 - 4.0 * error, inside
