"""Evaluating a plan from logs: the certificate against an independent route and the loops that
fixed its statement, and the estimates against values in closed form.

The independent route, ``_xi_log_moment``, writes ``alpha log W_H`` as a quadratic form in the
standard-normal vector of every draw along a logged trajectory and integrates it in one step. It
uses no recursion, no starred loop and no Renyi formula, and takes a schedule as readily as a
policy, since the plan enters one step at a time. The loops are an external verifier's:

* CE1 passes the small-gain test and the one-step gate, and its trajectory weight has an infinite
  second moment from ``h = 6`` when it starts from the logger's stationary law.
* CE3 shifts no gain on an unstable loop, and is finite at every horizon.
* E1-E3 set the one-step gate against the stationary condition, in both directions.
* M1 and M2 make a spectral-radius condition wrong in both directions for matrices.

Truths are computed here, by Lyapunov solves and covariance propagation, not by the module.
"""

from __future__ import annotations

import logging
import math

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
from scipy.stats import norm
from scipy.stats import t as student

from chc import QuadraticCost
from chc.evaluation import (
    _GRID,
    AffinePolicy,
    AffineSchedule,
    EvaluationMethod,
    InfeasibleEvaluation,
    InitialLaw,
    LinearGaussianPlant,
    PlanEvaluation,
    _batch_half_width,
    _episode_smoothing,
    _evaluate_by_unit,
    _smoothed_log_moments,
    _StageCost,
    certify_evaluation,
    evaluate_plan,
    fit_logger,
)

# A two-sided market: x = (demand-supply imbalance, unserved backlog), u = the incentive level.
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
LOGGER = AffinePolicy(np.array([[0.3, 0.2]]), np.array([0.1]), np.array([[0.25]]))
# A plan that tightens its feedback over five steps while its offset walks down.
RAMP = AffineSchedule(
    np.stack([(0.4 + 0.3 * t) * PLAN.gain for t in range(5)]),
    np.linspace(0.3, -0.2, 5)[:, None],
    np.zeros((1, 1)),
)


def _scalar_plant(a: float, b: float, noise: float, offset: float = 0.0) -> LinearGaussianPlant:
    return LinearGaussianPlant(
        np.array([[a]]), np.array([[b]]), np.array([offset]), np.array([[noise]])
    )


def _scalar_policy(gain: float, offset: float, variance: float) -> AffinePolicy:
    return AffinePolicy(np.array([[gain]]), np.array([offset]), np.array([[variance]]))


def _stationary_state(
    plant: LinearGaussianPlant, policy: AffinePolicy
) -> tuple[np.ndarray, np.ndarray]:
    f = plant.a + plant.b @ policy.gain
    covariance = solve_discrete_lyapunov(f, plant.b @ policy.covariance @ plant.b.T + plant.noise)
    mean = np.linalg.solve(np.eye(f.shape[0]) - f, plant.b @ policy.offset + plant.offset)
    return mean, covariance


def _stage_cost(
    mx: np.ndarray, sx: np.ndarray, policy: AffinePolicy, target: np.ndarray | None = None
) -> float:
    """``E [(x - target)'Q(x - target) + u'Ru] / 2`` when ``x ~ N(mx, sx)`` and ``u`` follows
    ``policy``; the target is zero by default."""
    dx = mx if target is None else mx - target
    mu = policy.gain @ mx + policy.offset
    su = policy.gain @ sx @ policy.gain.T + policy.covariance
    return 0.5 * float(np.trace(Q @ sx) + dx @ Q @ dx + np.trace(R @ su) + mu @ R @ mu)


def _stationary_value(plant: LinearGaussianPlant, policy: AffinePolicy) -> float:
    return _stage_cost(*_stationary_state(plant, policy), policy)


def _episode_value(
    plant: LinearGaussianPlant,
    plan: AffinePolicy | AffineSchedule,
    initial: InitialLaw,
    horizon: int,
    targets: np.ndarray | None = None,
) -> float:
    mx, sx, total = initial.mean, initial.covariance, 0.0
    for t in range(horizon):
        policy = plan if isinstance(plan, AffinePolicy) else plan.step(t)
        f = plant.a + plant.b @ policy.gain
        total += _stage_cost(mx, sx, policy, None if targets is None else targets[t])
        mx = f @ mx + plant.b @ policy.offset + plant.offset
        sx = f @ sx @ f.T + plant.b @ policy.covariance @ plant.b.T + plant.noise
    return total


def _rollouts(
    plant: LinearGaussianPlant,
    policy: AffinePolicy,
    starts: np.ndarray,
    steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """``x`` ``(R, steps + 1, n)`` and ``u`` ``(R, steps, m)``, one rollout from each start."""
    replicates, n = starts.shape
    m = plant.actions
    noise = np.linalg.cholesky(plant.noise)
    dither = np.linalg.cholesky(policy.covariance)
    x, u = np.empty((replicates, steps + 1, n)), np.empty((replicates, steps, m))
    x[:, 0] = starts
    for t in range(steps):
        u[:, t] = (
            x[:, t] @ policy.gain.T
            + policy.offset
            + rng.standard_normal((replicates, m)) @ dither.T
        )
        x[:, t + 1] = (
            x[:, t] @ plant.a.T
            + u[:, t] @ plant.b.T
            + plant.offset
            + rng.standard_normal((replicates, n)) @ noise.T
        )
    return x, u


def _stationary_logs(
    plant: LinearGaussianPlant, logger: AffinePolicy, steps: int, replicates: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Logs started from the logger's stationary law, so that no burn-in is needed."""
    rng = np.random.default_rng(seed)
    mean, covariance = _stationary_state(plant, logger)
    starts = (
        mean + rng.standard_normal((replicates, mean.shape[0])) @ np.linalg.cholesky(covariance).T
    )
    return _rollouts(plant, logger, starts, steps, rng)


def _unit_windows(
    plant: LinearGaussianPlant,
    logger: AffinePolicy,
    units: int,
    windows: int,
    horizon: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``x``, ``u`` and each episode's unit: every unit one run of the logger from its stationary
    law, cut into ``windows`` consecutive windows of ``horizon`` steps, so that a unit's windows
    share its state."""
    mean, covariance = _stationary_state(plant, logger)
    starts = mean + rng.standard_normal((units, mean.shape[0])) @ np.linalg.cholesky(covariance).T
    x, u = _rollouts(plant, logger, starts, windows * horizon, rng)
    xs = np.stack([x[:, k * horizon : (k + 1) * horizon + 1] for k in range(windows)], axis=1)
    us = np.stack([u[:, k * horizon : (k + 1) * horizon] for k in range(windows)], axis=1)
    return (
        xs.reshape(units * windows, horizon + 1, -1),
        us.reshape(units * windows, horizon, -1),
        np.repeat(np.arange(units), windows),
    )


def _log_density(policy: AffinePolicy | AffineSchedule, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """``log N(u; gain x + offset, covariance)`` at every episode and step, up to a constant."""
    if isinstance(policy, AffineSchedule):
        mean = np.einsum("tmn,etn->etm", policy.gains, x) + policy.offsets
    else:
        mean = x @ policy.gain.T + policy.offset
    residual = u - mean
    precision = np.linalg.inv(policy.covariance)
    return -0.5 * (
        np.einsum("eti,ij,etj->et", residual, precision, residual)
        + np.linalg.slogdet(policy.covariance)[1]
    )


def _xi_log_moment(
    plant: LinearGaussianPlant,
    logger: AffinePolicy,
    plan: AffinePolicy | AffineSchedule,
    initial: InitialLaw,
    horizon: int,
    alpha: float = 2.0,
) -> float:
    """``log E_b[W_H^alpha]``, with ``alpha log W_H`` a quadratic form in the standard normals.

    ``xi = (e_0, nu_0..nu_{H-1}, w_0..w_{H-2})``: the initial draw, the logger's action draws and
    the plant's noise. Every state and action is affine in ``xi``, the log weight is quadratic in
    it, and ``E exp(xi'G xi + g'xi + g0) = det(I - 2G)^(-1/2) exp(g0 + g'(I - 2G)^(-1) g / 2)``,
    finite iff ``I - 2G > 0``."""
    n, m = plant.b.shape
    size = n + horizon * m + (horizon - 1) * n
    eigenvalues, vectors = np.linalg.eigh(initial.covariance)
    root0 = vectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None))) @ vectors.T
    root_b = np.linalg.cholesky(logger.covariance)
    root_w = np.linalg.cholesky(plant.noise)
    cx, lx = initial.mean.astype(float), np.zeros((n, size))
    lx[:, :n] = root0
    g_mat, g_vec, g0 = np.zeros((size, size)), np.zeros(size), 0.0
    column = n
    for t in range(horizon):
        step = plan if isinstance(plan, AffinePolicy) else plan.step(t)
        precision = np.linalg.inv(step.covariance)
        d_gain, d_offset = step.gain - logger.gain, step.offset - logger.offset
        log_det = np.linalg.slogdet(logger.covariance)[1] - np.linalg.slogdet(step.covariance)[1]
        pick = np.zeros((m, size))
        pick[:, column : column + m] = np.eye(m)
        column += m
        # u_t - plan's mean = -d_gain x_t - d_offset + root_b nu_t; 2 log(pi / b) is quadratic in it
        error = -d_gain @ lx + root_b @ pick
        shift = -d_gain @ cx - d_offset
        g_mat += pick.T @ pick - error.T @ precision @ error
        g_vec += -2.0 * error.T @ precision @ shift
        g0 += -shift @ precision @ shift + log_det
        if t < horizon - 1:
            noise = np.zeros((n, size))
            noise[:, column : column + n] = np.eye(n)
            column += n
            lu, cu = logger.gain @ lx + root_b @ pick, logger.gain @ cx + logger.offset
            lx = plant.a @ lx + plant.b @ lu + root_w @ noise
            cx = plant.a @ cx + plant.b @ cu + plant.offset
    scale = alpha / 2.0
    core = np.eye(size) - 2.0 * scale * g_mat
    core = 0.5 * (core + core.T)
    if np.min(np.linalg.eigvalsh(core)) <= 0.0:
        return math.inf
    g = scale * g_vec
    return float(scale * g0 - 0.5 * np.linalg.slogdet(core)[1] + 0.5 * g @ np.linalg.solve(core, g))


def _random_case(
    rng: np.random.Generator,
) -> tuple[LinearGaussianPlant, AffinePolicy, AffinePolicy, InitialLaw, int]:
    n, m = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    a = rng.normal(size=(n, n)) * 0.5
    b = rng.normal(size=(n, m))
    root = rng.normal(size=(n, n)) * 0.4
    plant = LinearGaussianPlant(a, b, rng.normal(size=n) * 0.3, root @ root.T + 0.05 * np.eye(n))
    root_b = rng.normal(size=(m, m)) * 0.5
    logger_cov = root_b @ root_b.T + 0.6 * np.eye(m)
    logger = AffinePolicy(rng.normal(size=(m, n)) * 0.4, rng.normal(size=m) * 0.3, logger_cov)
    plan_cov = logger_cov * rng.uniform(0.3, 1.6)
    plan = AffinePolicy(
        logger.gain + rng.normal(size=(m, n)) * 0.3, rng.normal(size=m) * 0.3, plan_cov
    )
    root0 = rng.normal(size=(n, n)) * 0.5
    initial = InitialLaw(rng.normal(size=n) * 0.5, root0 @ root0.T)
    return plant, logger, plan, initial, int(rng.integers(1, 7))


def _pdis(
    plant: LinearGaussianPlant,
    logger: AffinePolicy,
    plan: AffinePolicy,
    initial: InitialLaw,
    horizon: int,
    samples: int = 10**6,
):
    return certify_evaluation(
        plant, logger, plan, "pdis", samples, horizon=horizon, initial=initial
    )


# ------------------------------------------------------------------------ the trajectory weight


def test_the_trajectory_moment_matches_the_whole_trajectory_integral() -> None:
    rng = np.random.default_rng(11)
    finite = 0
    for _ in range(80):
        plant, logger, plan, initial, horizon = _random_case(rng)
        certified = _pdis(plant, logger, plan, initial, horizon).log_second_moment
        oracle = _xi_log_moment(plant, logger, plan, initial, horizon)
        assert math.isfinite(certified) == math.isfinite(oracle)
        if math.isfinite(oracle):
            finite += 1
            assert certified == pytest.approx(oracle, rel=1e-9, abs=1e-9)
    assert 20 < finite < 80  # both branches were exercised


def test_a_schedules_trajectory_moment_matches_the_whole_trajectory_integral() -> None:
    rng = np.random.default_rng(12)
    finite = 0
    for _ in range(80):
        plant, logger, plan, initial, horizon = _random_case(rng)
        m, n = plan.gain.shape
        schedule = AffineSchedule(
            plan.gain + rng.normal(size=(horizon, m, n)) * 0.3,
            plan.offset + rng.normal(size=(horizon, m)) * 0.3,
            plan.covariance,
        )
        certified = certify_evaluation(
            plant, logger, schedule, "pdis", 10**6, initial=initial
        ).log_second_moment
        oracle = _xi_log_moment(plant, logger, schedule, initial, horizon)
        assert math.isfinite(certified) == math.isfinite(oracle)
        if math.isfinite(oracle):
            finite += 1
            assert certified == pytest.approx(oracle, rel=1e-9, abs=1e-9)
    assert 20 < finite < 80  # both branches were exercised


def test_the_smoothing_grid_reads_each_level_as_the_whole_trajectory_integral_does() -> None:
    """Every level of a grid in one recursion, each against the integral for the plan smoothed by
    that level alone, at every horizon. Past the logger's room a level diverges from the first
    step; within it a level can still diverge at a later one, and is ``inf`` from there on."""
    rng = np.random.default_rng(13)
    finite = first = later = 0
    for case in range(40):
        plant, logger, plan, initial, horizon = _random_case(rng)
        m, n = plan.gain.shape
        schedule = AffineSchedule(
            plan.gain + rng.normal(size=(horizon, m, n)) * 0.3,
            plan.offset + rng.normal(size=(horizon, m)) * 0.3,
            plan.covariance if case % 2 else np.zeros((m, m)),
        )
        room = np.min(np.linalg.eigvalsh(2.0 * logger.covariance - schedule.covariance))
        levels = math.sqrt(room) * np.geomspace(0.05, 1.5, 8)

        moments = _smoothed_log_moments(plant, logger, schedule, initial, 2.0, levels)

        for k, tau in enumerate(levels):
            covariance = schedule.covariance + tau * tau * np.eye(m)
            for h in range(1, horizon + 1):
                head = AffineSchedule(schedule.gains[:h], schedule.offsets[:h], covariance)
                oracle = _xi_log_moment(plant, logger, head, initial, h)
                assert math.isfinite(moments[k, h - 1]) == math.isfinite(oracle)
                if math.isfinite(oracle):
                    assert moments[k, h - 1] == pytest.approx(oracle, rel=1e-9, abs=1e-9)
            row = np.isfinite(moments[k])
            finite += bool(row.all())
            first += not row[0]
            later += bool(row[0] and not row.all())
    # every branch was exercised
    assert finite > 0
    assert first > 0
    assert later > 0


def test_a_schedule_that_holds_one_policy_is_certified_as_that_policy() -> None:
    initial = InitialLaw(np.array([0.3, -0.2]), 0.2 * np.eye(2))
    held = AffineSchedule(
        np.broadcast_to(PLAN.gain, (5, 1, 2)), np.broadcast_to(PLAN.offset, (5, 1)), PLAN.covariance
    )

    assert certify_evaluation(MARKET, LOGGER, held, "pdis", 10**4, initial=initial) == (
        certify_evaluation(MARKET, LOGGER, PLAN, "pdis", 10**4, horizon=5, initial=initial)
    )
    ramp = certify_evaluation(MARKET, LOGGER, RAMP, "pdis", 10**4, initial=initial, smoothing=0.3)
    margins = [
        certify_evaluation(
            MARKET, LOGGER, RAMP.step(t), "pdis", 10**4, horizon=5, initial=initial, smoothing=0.3
        ).one_step_margin
        for t in range(5)
    ]
    assert all(margin is not None for margin in margins)
    assert ramp.one_step_margin == min(margin for margin in margins if margin is not None)


def test_the_small_gain_test_and_the_one_step_gate_pass_on_a_loop_that_escapes_at_h_6() -> None:
    # CE1: 2 S_b - S_pi - 2 D^2 R_x = 55/52 > 0, and r = 9/70 < 1 - |a*| = 31/140. From the
    # logger's stationary law, R_x = 200/13, the x_0 term diverges once the recursion's iterate
    # passes 1 / (2 R_x); from a fixed start it never does.
    plant = _scalar_plant(19 / 20, 1.0, 1 / 2)
    logger, plan = _scalar_policy(0.0, 0.0, 1.0), _scalar_policy(-3 / 20, 0.0, 1 / 4)

    stationary_start = _pdis(
        plant, logger, plan, InitialLaw(np.zeros(1), np.array([[200 / 13]])), 10
    )
    fixed_start = _pdis(plant, logger, plan, InitialLaw(np.zeros(1), np.zeros((1, 1))), 400)

    assert stationary_start.one_step_margin == pytest.approx(55 / 52, rel=1e-12)
    assert stationary_start.escape_horizon == 6
    assert not stationary_start.certified
    assert "infinite from h = 6" in stationary_start.reason
    assert fixed_start.escape_horizon is None


def test_a_plan_that_shifts_no_gain_is_finite_at_every_horizon_on_an_unstable_loop() -> None:
    # CE3: D = 0, so E_b[W_H^2] = exp(H (log c0 + d^2 / (2 S_b - S_pi))) whatever the loop does.
    plant = _scalar_plant(1.5, 1.0, 0.3)
    logger, plan = _scalar_policy(-0.3, 0.1, 1.0), _scalar_policy(-0.3, -0.2, 0.64)

    certificate = _pdis(plant, logger, plan, InitialLaw(np.zeros(1), np.eye(1)), 60)

    closed = 60 * (-0.5 * math.log(0.64) - 0.5 * math.log(1.36) + 0.09 / 1.36)
    assert certificate.log_second_moment == pytest.approx(8.134660321708, rel=1e-11)
    assert certificate.log_second_moment == pytest.approx(closed, rel=1e-12)
    assert certificate.one_step_margin is None  # the logger's loop has no stationary law


def test_the_certifiable_horizon_grows_like_log_n_over_the_growth_rate() -> None:
    # The lab's two-state loop, with offsets and a correlated initial law: lambda = 0.290715 a
    # step, so each tenfold n buys log(10) / lambda = 7.9 steps.
    plant = LinearGaussianPlant(
        np.array([[0.95, 0.10], [0.0, 0.85]]),
        np.array([[0.0], [0.5]]),
        np.zeros(2),
        np.array([[0.05, 0.01], [0.01, 0.08]]),
    )
    logger = AffinePolicy(np.array([[-0.2, -0.5]]), np.array([0.1]), np.array([[0.8]]))
    plan = AffinePolicy(np.array([[-0.4, -0.9]]), np.array([-0.1]), np.array([[0.3]]))
    initial = InitialLaw(np.array([0.2, -0.1]), np.array([[0.3, 0.05], [0.05, 0.4]]))

    horizons = [
        _pdis(plant, logger, plan, initial, 40, samples).certified_horizon
        for samples in (10**3, 10**4, 10**5, 10**6)
    ]
    growth = _pdis(plant, logger, plan, initial, 201).log_second_moment - (
        _pdis(plant, logger, plan, initial, 200).log_second_moment
    )
    stationary = certify_evaluation(plant, logger, plan, "mis", 10**4)

    assert horizons == [7, 15, 23, 31]
    assert growth == pytest.approx(0.290715, abs=5e-7)
    assert math.exp(stationary.log_second_moment) == pytest.approx(1.9023, abs=5e-5)


def test_a_spectral_radius_test_is_wrong_in_both_directions_for_matrices() -> None:
    # B = I, S_b = S_pi = I, W = 0.1 I, K_b = 0, K_pi = D, so the starred loop is A + 2 D.
    # M1: the naive radius condition holds, r = 0.4 <= 1 - rho(A*) = 0.5, and the moment escapes.
    # M2: it fails, r = 0.5 > 0.1, and the moment stays finite.
    def loop(
        a_star: np.ndarray, d: np.ndarray
    ) -> tuple[LinearGaussianPlant, AffinePolicy, AffinePolicy]:
        plant = LinearGaussianPlant(a_star - 2.0 * d, np.eye(2), np.zeros(2), 0.1 * np.eye(2))
        return (
            plant,
            AffinePolicy(np.zeros((2, 2)), np.zeros(2), np.eye(2)),
            AffinePolicy(d, np.zeros(2), np.eye(2)),
        )

    fixed = InitialLaw(np.zeros(2), np.zeros((2, 2)))
    m1 = loop(np.array([[0.5, 2.0], [0.0, 0.5]]), np.diag([math.sqrt(0.16 / 2.2), 0.0]))
    m2 = loop(np.diag([0.9, 0.1]), np.diag([0.0, math.sqrt(0.25 / 2.2)]))

    assert _pdis(*m1, fixed, 10).escape_horizon == 4
    assert _pdis(*m2, fixed, 3000).escape_horizon is None


# ----------------------------------------------------------------------- the stationary weight


@pytest.mark.parametrize(
    ("loop", "gate", "state", "action", "second"),
    [
        # E1: the one-step gate passes, and the plan spreads the state past twice the logger's.
        ((1.0, 0.5, 0.2, -1.0, 0.5, -0.4, 0.6), 0.088, -0.105556, None, None),
        # E2: the gate fails, and the stationary weight is finite.
        ((0.95, 1.0, 0.5, -0.05, 1.0, -0.95, 0.5), -11.289474, 14.789474, 0.635231, 3.642590),
        # E3: the gate passes, and so does 2 R_x - P_x; the action block fails.
        ((0.8, 1.0, 0.2, -0.6, 1.0, 0.0, 0.5), 0.6, 0.555556, -1.65, None),
    ],
    ids=["E1", "E2", "E3"],
)
def test_the_one_step_gate_is_neither_necessary_nor_sufficient_for_the_stationary_weight(
    loop: tuple[float, ...],
    gate: float,
    state: float,
    action: float | None,
    second: float | None,
) -> None:
    a, b, noise, logger_gain, logger_var, plan_gain, plan_var = loop
    certificate = certify_evaluation(
        _scalar_plant(a, b, noise),
        _scalar_policy(logger_gain, 0.0, logger_var),
        _scalar_policy(plan_gain, 0.0, plan_var),
        "mis",
        10**5,
    )

    assert certificate.one_step_margin == pytest.approx(gate, abs=1e-6)
    assert certificate.state_margin == pytest.approx(state, abs=1e-6)
    if action is not None:
        assert certificate.action_margin == pytest.approx(action, abs=1e-6)
    assert certificate.certified == (second is not None)
    if second is not None:
        assert math.exp(certificate.log_second_moment) == pytest.approx(second, abs=1e-6)


def test_the_smoothing_interval_is_exactly_zero_to_tau_max() -> None:
    limit = certify_evaluation(MARKET, LOGGER, PLAN, "mis", 10**5).smoothing_limit
    assert limit is not None

    inside = certify_evaluation(MARKET, LOGGER, PLAN, "mis", 10**5, smoothing=0.999 * limit)
    outside = certify_evaluation(MARKET, LOGGER, PLAN, "mis", 10**5, smoothing=1.001 * limit)

    assert all(m is not None and m > 0.0 for m in (inside.state_margin, inside.action_margin))
    assert not outside.certified
    assert any(m is not None and m <= 0.0 for m in (outside.state_margin, outside.action_margin))


def test_ci_reliable_reads_the_fourth_moment_not_the_second() -> None:
    # Same gain, a wider plan: 2 Sigma_b - Sigma_pi > 0 holds, 4 Sigma_b - 3 Sigma_pi does not.
    plant = _scalar_plant(0.5, 1.0, 1.0)
    logger = _scalar_policy(-0.2, 0.0, 1.0)

    narrow = certify_evaluation(plant, logger, _scalar_policy(-0.2, 0.0, 1.2), "mis", 10**5)
    wide = certify_evaluation(plant, logger, _scalar_policy(-0.2, 0.0, 1.4), "mis", 10**5)

    assert (narrow.certified, narrow.ci_reliable) == (True, True)
    assert (wide.certified, wide.ci_reliable) == (True, False)


# ---------------------------------------------------------------------------------- estimates


def test_the_loop_the_one_step_gate_misjudges_is_refused_by_weights_and_evaluated_by_fqe() -> None:
    # The loop of test_the_one_step_estimand_is_not_the_deployed_value: the candidate spreads the
    # state to 4.28 against the logger's 1.47, so every stationary weight has infinite variance,
    # while the restricted chi-square over quadratic features is 3.30.
    plant = _scalar_plant(0.9, 1.0, 0.1)
    logger, candidate = _scalar_policy(-0.4, 0.0, 1.0), _scalar_policy(-0.1, 0.0, 1.44)
    cost = QuadraticCost(
        Q=jnp.array([[2.0]]), R=jnp.array([[2.0]]), Qf=jnp.array([[2.0]]), x_target=jnp.zeros(1)
    )
    x, u = _stationary_logs(plant, logger, 20_000, 1, seed=31)
    logs = {"x": x[0], "u": u[0]}

    refusals = {}
    for method in ("mis", "dr"):
        with pytest.raises(InfeasibleEvaluation) as caught:
            evaluate_plan(logs, candidate, method, plant=plant, cost=cost, logger=logger)
        refusals[method] = caught.value.certificate
    fqe = evaluate_plan(logs, candidate, "fqe", plant=plant, cost=cost, logger=logger)

    deployed = (1.54 / 0.36) * (1 + 0.1**2) + 1.44  # 5.761, the candidate's own average cost
    one_step = (1.1 / 0.75) * (1 + 0.1**2) + 1.44  # 2.921, what one-step weights converge to
    assert all(c.state_margin is not None and c.state_margin < 0.0 for c in refusals.values())
    assert math.expm1(fqe.certificate.log_second_moment) == pytest.approx(3.30, abs=0.01)
    assert fqe.interval[0] <= deployed <= fqe.interval[1]
    assert not fqe.interval[0] <= one_step <= fqe.interval[1]


def test_smoothing_costs_exactly_tau_squared_beta() -> None:
    x, u = _stationary_logs(MARKET, LOGGER, 4000, 1, seed=5)
    stationary = evaluate_plan({"x": x[0], "u": u[0]}, PLAN, "mis", plant=MARKET, cost=COST)
    initial = InitialLaw(np.array([1.0, 0.5]), 0.2 * np.eye(2))
    starts = initial.mean + np.random.default_rng(6).normal(size=(3000, 2)) @ np.sqrt(
        initial.covariance
    )
    xe, ue = _rollouts(
        MARKET,
        AffinePolicy(LOGGER.gain, LOGGER.offset, np.eye(1)),
        starts,
        5,
        np.random.default_rng(7),
    )
    episodic = evaluate_plan({"x": xe, "u": ue}, PLAN, "pdis", plant=MARKET, cost=COST)

    def smoothed(tau: float) -> AffinePolicy:
        return AffinePolicy(PLAN.gain, PLAN.offset, np.array([[tau * tau]]))

    tau = stationary.certificate.smoothing
    assert tau > 0.0
    assert stationary.model_correction == pytest.approx(
        _stationary_value(MARKET, smoothed(tau)) - _stationary_value(MARKET, PLAN), rel=1e-9
    )
    tau = episodic.certificate.smoothing
    law = InitialLaw(xe[:, 0].mean(axis=0), np.cov(xe[:, 0], rowvar=False))
    assert episodic.model_correction == pytest.approx(
        _episode_value(MARKET, smoothed(tau), law, 5) - _episode_value(MARKET, PLAN, law, 5),
        rel=1e-9,
    )


@pytest.mark.parametrize("method", ["mis", "dr", "fqe"])
def test_the_stationary_intervals_cover_the_deployed_value(method: EvaluationMethod) -> None:
    truth = _stationary_value(MARKET, PLAN)
    x, u = _stationary_logs(MARKET, LOGGER, 4000, 100, seed=2026)

    covered, errors = [], []
    for r in range(x.shape[0]):
        result = evaluate_plan(
            {"x": x[r], "u": u[r]}, PLAN, method, plant=MARKET, cost=COST, model_error=0.0
        )
        covered.append(result.interval[0] <= truth <= result.interval[1])
        errors.append(result.value - truth)
        assert (result.degrees_of_freedom is None) == (method == "fqe")

    errors_ = np.asarray(errors)
    assert 0.88 <= np.mean(covered) <= 1.0  # nominal 0.95, three binomial SDs at 100 replicates
    assert abs(errors_.mean()) < 3.0 * errors_.std(ddof=1) / math.sqrt(errors_.size)


@pytest.mark.parametrize(("carriers", "dof"), [(range(40), 39.0), ((3, 11, 20, 37), 4.0)])
def test_the_batch_interval_has_as_many_degrees_of_freedom_as_batches_carry_mass(
    carriers: tuple[int, ...] | range, dof: float
) -> None:
    # Constant within a batch, so each batch's sum of squares is exactly 100 or 0: forty equal
    # batches keep t_39, and four carrying batches leave t_4.
    contributions = np.zeros(4000)
    for sign, batch in enumerate(carriers):
        contributions[100 * batch : 100 * (batch + 1)] = (-1.0) ** sign
    means = contributions.reshape(40, 100).mean(axis=1)

    half, got = _batch_half_width(contributions)

    assert got == pytest.approx(dof, rel=1e-12)
    assert half == pytest.approx(
        student.ppf(0.975, dof) * means.std(ddof=1) / math.sqrt(40), rel=1e-12, abs=0.0
    )


def test_weights_spent_in_a_few_of_the_loggers_excursions_leave_the_interval_few_degrees() -> None:
    # MountainCarContinuous-v0 linearised at the valley floor, its force disturbed and logged by a
    # lightly damped operator, evaluated for a tight LQR plan whose states are ten times narrower
    # than the logs': the weight sits in the logger's rare passes through the floor. On 40 logs
    # the degrees of freedom never exceeded 13.4, and a fixed t_39 covered 0.91 of 500 replicates.
    # The market's weights spread over every batch; its lowest of 40 logs was 31.6.
    a, b = np.array([[0.9925, 1.0], [-0.0075, 1.0]]), np.array([[0.0015], [0.0015]])
    plant = LinearGaussianPlant(a, b, np.zeros(2), 0.04 * b @ b.T)
    logger = AffinePolicy(np.array([[0.0, -20.0]]), np.zeros(1), np.array([[0.09]]))
    q, r = np.diag([400.0, 40000.0]), np.array([[1.0]])
    p = solve_discrete_are(a, b, q, r)
    plan = AffinePolicy(
        -np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a), np.array([0.1]), np.zeros((1, 1))
    )
    cost = QuadraticCost(
        Q=jnp.asarray(q), R=jnp.asarray(r), Qf=jnp.asarray(q), x_target=jnp.zeros(2)
    )
    rng = np.random.default_rng(5)
    mean, covariance = _stationary_state(plant, logger)
    x, u = np.empty((4001, 2)), np.empty((4000, 1))
    x[0] = mean + np.linalg.cholesky(covariance) @ rng.standard_normal(2)
    for t in range(4000):
        u[t] = logger.gain @ x[t] + 0.3 * rng.standard_normal(1)
        x[t + 1] = a @ x[t] + b @ (u[t] + 0.2 * rng.standard_normal(1))
    xm, um = _stationary_logs(MARKET, LOGGER, 4000, 1, seed=2026)

    spiky = evaluate_plan({"x": x, "u": u}, plan, "dr", plant=plant, cost=cost, model_error=0.0)
    spread = evaluate_plan(
        {"x": xm[0], "u": um[0]}, PLAN, "mis", plant=MARKET, cost=COST, model_error=0.0
    )

    assert spiky.degrees_of_freedom is not None
    assert spiky.degrees_of_freedom < 15.0
    assert spread.degrees_of_freedom is not None
    assert spread.degrees_of_freedom > 25.0


def test_the_episodic_interval_covers_the_deployed_value() -> None:
    initial = InitialLaw(np.array([1.0, 0.5]), 0.2 * np.eye(2))
    logger = AffinePolicy(LOGGER.gain, LOGGER.offset, np.array([[1.0]]))
    rng = np.random.default_rng(77)

    covered = []
    for _ in range(60):
        starts = initial.mean + rng.normal(size=(3000, 2)) @ np.sqrt(initial.covariance)
        x, u = _rollouts(MARKET, logger, starts, 5, rng)
        result = evaluate_plan(
            {"x": x, "u": u}, PLAN, "pdis", plant=MARKET, cost=COST, model_error=0.0
        )
        truth = _episode_value(MARKET, PLAN, initial, 5)
        covered.append(result.interval[0] <= truth <= result.interval[1])

    assert np.mean(covered) >= 0.85  # nominal 0.95; 60 replicates


@pytest.mark.parametrize("kind", ["feedback", "open loop"])
def test_the_episodic_interval_covers_a_schedules_deployed_value(kind: str) -> None:
    # Against a target that moves every step: the schedule's truth reads each step's own row.
    initial = InitialLaw(np.array([1.0, 0.5]), 0.2 * np.eye(2))
    logger = AffinePolicy(LOGGER.gain, LOGGER.offset, np.array([[1.0]]))
    targets = np.linspace([0.5, 0.0], [-0.3, 0.4], 6)
    cost = QuadraticCost(Q=COST.Q, R=COST.R, Qf=COST.Qf, x_target=jnp.asarray(targets))
    schedule = (
        RAMP
        if kind == "feedback"
        else AffineSchedule.open_loop(np.linspace(0.4, -0.2, 5)[:, None], states=2)
    )
    truth = _episode_value(MARKET, schedule, initial, 5, targets)
    rng = np.random.default_rng(78)

    covered, errors = [], []
    for _ in range(60):
        starts = initial.mean + rng.normal(size=(3000, 2)) @ np.sqrt(initial.covariance)
        x, u = _rollouts(MARKET, logger, starts, 5, rng)
        result = evaluate_plan(
            {"x": x, "u": u}, schedule, "pdis", plant=MARKET, cost=cost, model_error=0.0
        )
        covered.append(result.interval[0] <= truth <= result.interval[1])
        errors.append(result.value - truth)

    errors_ = np.asarray(errors)
    assert np.mean(covered) >= 0.85  # nominal 0.95; 60 replicates
    assert abs(errors_.mean()) < 3.0 * errors_.std(ddof=1) / math.sqrt(errors_.size)


def test_a_schedules_smoothing_costs_exactly_tau_squared_beta() -> None:
    initial = InitialLaw(np.array([1.0, 0.5]), 0.2 * np.eye(2))
    rng = np.random.default_rng(6)
    starts = initial.mean + rng.normal(size=(3000, 2)) @ np.sqrt(initial.covariance)
    logger = AffinePolicy(LOGGER.gain, LOGGER.offset, np.eye(1))
    x, u = _rollouts(MARKET, logger, starts, 5, rng)

    result = evaluate_plan({"x": x, "u": u}, RAMP, "pdis", plant=MARKET, cost=COST)

    tau = result.certificate.smoothing
    smoothed = AffineSchedule(RAMP.gains, RAMP.offsets, np.array([[tau * tau]]))
    law = InitialLaw(x[:, 0].mean(axis=0), np.cov(x[:, 0], rowvar=False))
    assert tau > 0.0
    assert result.model_correction == pytest.approx(
        _episode_value(MARKET, smoothed, law, 5) - _episode_value(MARKET, RAMP, law, 5), rel=1e-9
    )


def test_the_smoothing_is_the_level_a_search_one_level_at_a_time_would_choose() -> None:
    """The smoothing's search scores its grid at once, each level's value read as the plan's own
    plus ``tau^2 beta``. Here each level is scored alone: its weights' second moment by the
    whole-trajectory integral, its value by covariance propagation of the smoothed plan. The level
    chosen scores the least, to rounding, or no level passes and none is chosen."""
    rng = np.random.default_rng(15)
    stages = _StageCost.steps(COST, 2, 1, 5)
    chosen = 0
    for _ in range(12):
        logger = AffinePolicy(LOGGER.gain, LOGGER.offset, np.array([[rng.uniform(0.2, 1.5)]]))
        schedule = AffineSchedule.open_loop(rng.normal(0.0, 0.4, (5, 1)), states=2)
        initial = InitialLaw(rng.normal(size=2) * 0.5, rng.uniform(0.05, 0.4) * np.eye(2))
        samples, model_error = int(rng.integers(300, 5000)), float(rng.choice([0.0, 0.5, 1.0]))

        got = _episode_smoothing(
            MARKET, logger, schedule, stages, initial, samples, model_error, 100.0
        )

        limit = math.sqrt(2.0 * logger.covariance[0, 0])
        grid = np.geomspace(1e-4 * limit, 0.999 * limit, _GRID)
        base = _episode_value(MARKET, schedule, initial, 5)
        scores = np.full(_GRID, math.inf)
        for i, tau in enumerate(grid):
            smoothed = AffineSchedule(schedule.gains, schedule.offsets, np.array([[tau * tau]]))
            l2 = _xi_log_moment(MARKET, logger, smoothed, initial, 5)
            if l2 <= math.log(samples / 100.0):
                value = _episode_value(MARKET, smoothed, initial, 5)
                variance = math.expm1(l2) * value * value / samples
                scores[i] = (model_error * (value - base)) ** 2 + variance
        if not np.isfinite(scores).any():
            assert got is None
            continue
        chosen += 1
        assert got is not None
        assert scores[np.argmin(np.abs(grid - got))] == pytest.approx(scores.min(), rel=1e-9)
    assert 0 < chosen < 12


def test_pdis_scores_each_step_against_its_own_target() -> None:
    # The plan is the logger, so every weight is one and the estimate is the logs' mean cost, each
    # step against its own row; no step reads the last, the terminal state's.
    logger = AffinePolicy(LOGGER.gain, LOGGER.offset, np.array([[1.0]]))
    rng = np.random.default_rng(21)
    x, u = _rollouts(MARKET, logger, rng.normal(size=(400, 2)), 5, rng)
    targets = np.linspace([1.0, -0.5], [-0.5, 0.8], 6)
    cost = QuadraticCost(Q=COST.Q, R=COST.R, Qf=COST.Qf, x_target=jnp.asarray(targets))
    schedule = AffineSchedule(
        np.broadcast_to(logger.gain, (5, 1, 2)),
        np.broadcast_to(logger.offset, (5, 1)),
        logger.covariance,
    )

    result = evaluate_plan(
        {"x": x, "u": u}, schedule, "pdis", plant=MARKET, cost=cost, logger=logger
    )

    dx = x[:, :-1] - targets[:-1]
    logged = 0.5 * (np.einsum("eti,ij,etj->et", dx, Q, dx) + np.einsum("eti,ij,etj->et", u, R, u))
    assert result.value == pytest.approx(logged.mean(axis=0).sum(), rel=1e-12)


def test_dr_removes_the_first_order_error_of_a_misspecified_model() -> None:
    # A[0, 0] off by 0.1. The model's own value, and MIS, whose weights come from the model, carry
    # its first-order error; DR's is the product of the two nuisances' errors. The plan carries
    # its dither into deployment, so no smoothing correction adds a first-order term of its own.
    dithered = AffinePolicy(PLAN.gain, PLAN.offset, np.array([[0.04]]))
    truth = _stationary_value(MARKET, dithered)
    wrong = LinearGaussianPlant(
        MARKET.a + np.array([[0.1, 0.0], [0.0, 0.0]]), MARKET.b, MARKET.offset, MARKET.noise
    )
    x, u = _stationary_logs(MARKET, LOGGER, 200_000, 1, seed=3)
    logs = {"x": x[0], "u": u[0]}

    direct = _stationary_value(wrong, dithered) - truth
    mis = evaluate_plan(logs, dithered, "mis", plant=wrong, cost=COST).value - truth
    dr = evaluate_plan(logs, dithered, "dr", plant=wrong, cost=COST).value - truth

    assert direct > 0.003
    assert abs(mis - direct) < 0.25 * direct
    assert abs(dr) < 0.25 * direct


def test_a_randomised_plan_is_evaluated_as_it_is() -> None:
    dithered = AffinePolicy(PLAN.gain, PLAN.offset, np.array([[0.04]]))
    x, u = _stationary_logs(MARKET, LOGGER, 4000, 1, seed=9)

    result = evaluate_plan({"x": x[0], "u": u[0]}, dithered, "dr", plant=MARKET, cost=COST)

    assert result.certificate.smoothing == 0.0
    assert (result.model_correction, result.model_share) == (0.0, 0.0)
    assert result.interval[0] <= _stationary_value(MARKET, dithered) <= result.interval[1]


def test_the_model_correction_is_carried_in_the_interval() -> None:
    x, u = _stationary_logs(MARKET, LOGGER, 4000, 1, seed=12)
    logs = {"x": x[0], "u": u[0]}

    trusted = evaluate_plan(logs, PLAN, "mis", plant=MARKET, cost=COST, model_error=0.0)
    counted = evaluate_plan(
        logs,
        PLAN,
        "mis",
        plant=MARKET,
        cost=COST,
        model_error=1.0,
        smoothing=trusted.certificate.smoothing,
    )

    width = counted.interval[1] - counted.interval[0]
    assert width == pytest.approx(
        trusted.interval[1] - trusted.interval[0] + 2.0 * trusted.model_correction,
        rel=1e-12,
        abs=0.0,
    )
    assert counted.model_share == pytest.approx(
        trusted.model_correction / (trusted.value + trusted.model_correction), rel=1e-12, abs=0.0
    )


def test_a_cost_and_its_negative_read_one_interval() -> None:
    # a reward written as a cost need not be positive semidefinite, and smoothing then lowers it:
    # the model's correction is negative, and its error no smaller for that
    x, u = _stationary_logs(MARKET, LOGGER, 4000, 1, seed=12)
    logs = {"x": x[0], "u": u[0]}
    reward = QuadraticCost(Q=-COST.Q, R=-COST.R, Qf=-COST.Qf, x_target=COST.x_target)

    cost = evaluate_plan(logs, PLAN, "mis", plant=MARKET, cost=COST)
    negated = evaluate_plan(logs, PLAN, "mis", plant=MARKET, cost=reward)

    assert negated.model_correction == -cost.model_correction < 0.0
    assert negated.value == -cost.value
    assert negated.interval == (-cost.interval[1], -cost.interval[0])
    assert negated.model_share == cost.model_share > 0.0


def test_a_cost_and_its_negative_read_one_interval_over_the_units() -> None:
    logger = AffinePolicy(LOGGER.gain, LOGGER.offset, np.eye(1))
    x, u, units = _unit_windows(MARKET, logger, 20, 3, 5, np.random.default_rng(61))
    reward = QuadraticCost(Q=-COST.Q, R=-COST.R, Qf=-COST.Qf, x_target=COST.x_target)

    cost, negated = (
        _evaluate_by_unit(
            x,
            u,
            units,
            RAMP,
            plant=lambda xs, us: MARKET,
            cost=scored,
            logger=None,
            smoothing=None,
            model_error=1.0,
            min_effective=5.0,
            resamples=20,
            seed=3,
        )
        for scored in (COST, reward)
    )

    assert cost.versus_logger is not None
    assert negated.versus_logger is not None
    assert negated.model_correction == -cost.model_correction < 0.0
    assert negated.interval == (-cost.interval[1], -cost.interval[0])
    assert negated.versus_logger.interval == (
        -cost.versus_logger.interval[1],
        -cost.versus_logger.interval[0],
    )


def test_fit_logger_recovers_the_logging_policy() -> None:
    x, u = _stationary_logs(MARKET, LOGGER, 50_000, 1, seed=4)

    fitted = fit_logger(x[0, :-1], u[0])

    assert np.allclose(fitted.gain, LOGGER.gain, atol=0.02)
    assert np.allclose(fitted.offset, LOGGER.offset, atol=0.01)
    assert np.allclose(fitted.covariance, LOGGER.covariance, rtol=0.03)


# ------------------------------------------------------------------------- the units' bootstrap


def test_the_plan_against_the_logger_is_read_off_the_difference_within_each_draw() -> None:
    """With the plant, the logger and the smoothing given, a draw of the units recomputes two
    means, and their bootstrap spread is the cluster-robust one: each episode's influence summed
    within its unit (Liang and Zeger 1986). The difference's spread is the difference's
    influence's, a third below the two spreads in quadrature, since both values read the same
    episodes."""
    logger = AffinePolicy(LOGGER.gain, LOGGER.offset, np.eye(1))
    x, u, units = _unit_windows(MARKET, logger, 150, 4, 5, np.random.default_rng(31))
    tau = 0.6

    result = _evaluate_by_unit(
        x,
        u,
        units,
        RAMP,
        plant=lambda xs, us: MARKET,
        cost=COST,
        logger=logger,
        smoothing=tau,
        model_error=0.0,
        min_effective=20.0,
        resamples=4000,
        seed=3,
    )

    smoothed = AffineSchedule(RAMP.gains, RAMP.offsets, np.array([[tau * tau]]))
    xs = x[:, :-1]
    cost = 0.5 * (np.einsum("eti,ij,etj->et", xs, Q, xs) + np.einsum("eti,ij,etj->et", u, R, u))
    log_w = np.cumsum(_log_density(smoothed, xs, u) - _log_density(logger, xs, u), axis=1)
    w = np.exp(log_w - log_w.max(axis=0))
    w /= w.mean(axis=0)
    plan = np.sum(w * (cost - np.mean(w * cost, axis=0)), axis=1)
    logged = np.sum(cost - cost.mean(axis=0), axis=1)

    def spread(influence: np.ndarray) -> float:
        return math.sqrt(np.sum(np.bincount(units, weights=influence) ** 2)) / units.size

    comparison, bootstrap = result.versus_logger, result.bootstrap
    assert comparison is not None
    assert bootstrap is not None
    assert (bootstrap.units, bootstrap.resamples, bootstrap.refused) == (150, 4000, 0)
    assert comparison.logger_value == pytest.approx(cost.mean(axis=0).sum(), rel=1e-12)
    assert comparison.difference == pytest.approx(result.value - comparison.logger_value, rel=1e-12)
    half = (result.interval[1] - result.interval[0]) / 2.0
    paired = (comparison.interval[1] - comparison.interval[0]) / 2.0
    assert half == pytest.approx(1.959964 * spread(plan), rel=0.06)
    assert paired == pytest.approx(1.959964 * spread(plan - logged), rel=0.06)
    assert spread(plan - logged) < 0.75 * math.hypot(spread(plan), spread(logged))


def test_each_draw_of_the_units_is_evaluated_as_a_panel_of_its_own() -> None:
    """Replaying the draws from the seed, each drawn panel is handed to ``evaluate_plan`` with a
    plant fitted to it, so its logger is fitted, its starts' law read and its smoothing chosen on
    it alone. The intervals are those draws' spread, the plan's model correction added as
    ``evaluate_plan`` adds it, about the estimate on every unit."""
    logger = AffinePolicy(LOGGER.gain, LOGGER.offset, np.eye(1))
    x, u, units = _unit_windows(MARKET, logger, 40, 3, 5, np.random.default_rng(61))

    def plant(xs: np.ndarray, us: np.ndarray) -> LinearGaussianPlant:
        """The market's step fitted by least squares to the episodes given."""
        z = np.concatenate([xs[:, :-1], us, np.ones_like(us)], axis=-1).reshape(-1, 4)
        y = xs[:, 1:].reshape(-1, 2)
        fit = np.linalg.lstsq(z, y, rcond=None)[0]
        noise = np.cov(y - z @ fit, rowvar=False)
        return LinearGaussianPlant(fit[:2].T, fit[2:3].T, fit[3], noise)

    def logged(xs: np.ndarray, us: np.ndarray) -> float:
        stage = np.einsum("eti,ij,etj->et", xs[:, :-1], Q, xs[:, :-1])
        return float(np.mean(np.sum(0.5 * (stage + np.einsum("eti,ij,etj->et", us, R, us)), 1)))

    result = _evaluate_by_unit(
        x,
        u,
        units,
        RAMP,
        plant=plant,
        cost=COST,
        logger=None,
        smoothing=None,
        model_error=0.5,
        min_effective=5.0,
        resamples=50,
        seed=7,
    )

    rng = np.random.default_rng(7)
    values, differences = [], []
    for _ in range(50):
        drawn_units = np.bincount(rng.integers(0, 40, 40), minlength=40)
        rows = np.repeat(np.arange(units.size), drawn_units[units])
        xs, us = x[rows], u[rows]
        drawn = evaluate_plan(
            {"x": xs, "u": us},
            RAMP,
            "pdis",
            plant=plant(xs, us),
            cost=COST,
            model_error=0.5,
            min_effective=5.0,
        )
        values.append(drawn.value)
        differences.append(drawn.value - logged(xs, us))
    whole = evaluate_plan(
        {"x": x, "u": u},
        RAMP,
        "pdis",
        plant=plant(x, u),
        cost=COST,
        model_error=0.5,
        min_effective=5.0,
    )
    widened = 0.5 * whole.model_correction
    z = float(norm.ppf(0.975))

    comparison, bootstrap = result.versus_logger, result.bootstrap
    assert comparison is not None
    assert bootstrap is not None
    assert (bootstrap.units, bootstrap.resamples, bootstrap.refused) == (40, 50, 0)
    assert whole.certificate.smoothing > 0.0
    assert result.value == pytest.approx(whole.value, rel=1e-12)
    assert comparison.difference == pytest.approx(whole.value - logged(x, u), rel=1e-12)
    for (lo, hi), centre, draws in (
        (result.interval, whole.value, values),
        (comparison.interval, comparison.difference, differences),
    ):
        assert (lo + hi) / 2.0 == pytest.approx(centre, rel=1e-12)
        assert (hi - lo) / 2.0 == pytest.approx(z * np.std(draws, ddof=1) + widened, rel=1e-9)


def test_the_units_bootstrap_covers_where_windows_taken_as_independent_do_not() -> None:
    """Each of 60 units is one run of a logger whose loop keeps 98% of its slower mode a step, cut
    into ten consecutive windows, so a unit's windows share its state. Over 40 replicates the
    episodes' own interval, which takes them as independent, is little more than half as wide as
    the estimate's spread and covers 28 of 40; resampling the units, the analysis run again on each
    draw, covers 37 (the test asks 35, the binomial's 1.4% tail at the nominal 0.95) and is as wide
    as the spread."""
    logger = AffinePolicy(np.array([[0.02, 0.02]]), np.zeros(1), np.eye(1))
    schedule = AffineSchedule.open_loop(np.linspace(0.2, -0.1, 5)[:, None], states=2)
    truth = _episode_value(MARKET, schedule, InitialLaw(*_stationary_state(MARKET, logger)), 5)
    rng = np.random.default_rng(41)

    errors, independent, by_unit = [], [], []
    for replicate in range(40):
        x, u, units = _unit_windows(MARKET, logger, 60, 10, 5, rng)
        alone = evaluate_plan(
            {"x": x, "u": u}, schedule, "pdis", plant=MARKET, cost=COST, model_error=0.0
        )
        drawn = _evaluate_by_unit(
            x,
            u,
            units,
            schedule,
            plant=lambda xs, us: MARKET,
            cost=COST,
            logger=None,
            smoothing=None,
            model_error=0.0,
            min_effective=100.0,
            resamples=100,
            seed=replicate,
        )
        errors.append(drawn.value - truth)
        independent.append(alone.interval)
        by_unit.append(drawn.interval)

    def coverage(intervals: list[tuple[float, float]]) -> float:
        return float(np.mean([lo <= truth <= hi for lo, hi in intervals]))

    def width(intervals: list[tuple[float, float]]) -> float:
        return float(np.mean([hi - lo for lo, hi in intervals])) / 2.0

    spread = 1.959964 * float(np.std(errors, ddof=1))
    assert coverage(by_unit) >= 35 / 40
    assert coverage(independent) <= 0.75
    assert width(by_unit) == pytest.approx(spread, rel=0.25)
    assert width(independent) < 0.7 * spread


def test_draws_the_certificate_refuses_are_counted_and_too_many_refuse_the_evaluation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """At the certificate's edge, a draw of the units whose starts spread wider than the panel's
    needs more episodes than it has. Such draws are left out, counted, and logged as a warning;
    when all but one are, no spread is left to read and the evaluation is refused."""
    logger = AffinePolicy(LOGGER.gain, LOGGER.offset, np.eye(1))
    x, u, units = _unit_windows(MARKET, logger, 100, 3, 5, np.random.default_rng(51))
    starts = x[:, 0]
    initial = InitialLaw(starts.mean(axis=0), np.cov(starts, rowvar=False))
    edge = certify_evaluation(
        MARKET, logger, RAMP, "pdis", x.shape[0], smoothing=0.5, initial=initial
    ).effective_samples

    def evaluate(resamples: int) -> PlanEvaluation:
        return _evaluate_by_unit(
            x,
            u,
            units,
            RAMP,
            plant=lambda xs, us: MARKET,
            cost=COST,
            logger=logger,
            smoothing=0.5,
            model_error=0.0,
            min_effective=0.999 * edge,
            resamples=resamples,
            seed=0,
        )

    with caplog.at_level(logging.INFO, logger="chc.evaluation"):
        result = evaluate(40)
    records = [r for r in caplog.records if getattr(r, "chc_event", "") == "unit_bootstrap"]

    assert result.bootstrap is not None
    assert 0 < result.bootstrap.refused < 40
    assert [(r.levelno, r.refused) for r in records] == [
        (logging.WARNING, result.bootstrap.refused)
    ]
    with pytest.raises(InfeasibleEvaluation, match="which leaves no spread to read"):
        evaluate(2)


# ------------------------------------------------------------------------------ what is refused


def test_unstable_loops_are_refused_with_their_spectral_radius() -> None:
    plant = _scalar_plant(1.1, 1.0, 0.1)

    logger_unstable = certify_evaluation(
        plant, _scalar_policy(0.0, 0.0, 1.0), _scalar_policy(-0.5, 0.0, 1.0), "mis", 10**4
    )
    plan_unstable = certify_evaluation(
        plant, _scalar_policy(-0.5, 0.0, 1.0), _scalar_policy(0.0, 0.0, 1.0), "fqe", 10**4
    )

    assert "logger's closed loop has spectral radius 1.1" in logger_unstable.reason
    assert "plan's closed loop has spectral radius 1.1" in plan_unstable.reason
    assert not logger_unstable.certified
    assert not plan_unstable.certified


def test_arguments_that_do_not_fit_the_method_are_refused() -> None:
    initial = InitialLaw(np.zeros(2), np.eye(2))
    dithered = AffinePolicy(PLAN.gain, PLAN.offset, np.array([[0.04]]))

    with pytest.raises(ValueError, match="only 'pdis' reads"):
        certify_evaluation(MARKET, LOGGER, PLAN, "mis", 100, horizon=5, initial=initial)
    with pytest.raises(ValueError, match="pass their horizon"):
        certify_evaluation(MARKET, LOGGER, PLAN, "pdis", 100, horizon=5)
    with pytest.raises(ValueError, match="smoothing applies"):
        certify_evaluation(MARKET, LOGGER, PLAN, "fqe", 100, smoothing=0.1)
    with pytest.raises(ValueError, match="smoothing applies"):
        certify_evaluation(MARKET, LOGGER, dithered, "mis", 100, smoothing=0.1)
    with pytest.raises(ValueError, match="method must be one of"):
        certify_evaluation(MARKET, LOGGER, PLAN, "snips", 100)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="gain has shape"):
        certify_evaluation(MARKET, LOGGER, _scalar_policy(0.0, 0.0, 1.0), "mis", 100)


@pytest.mark.parametrize("method", ["mis", "dr", "fqe"])
def test_a_schedule_is_refused_by_every_stationary_method(method: EvaluationMethod) -> None:
    x, u = _stationary_logs(MARKET, LOGGER, 200, 1, seed=1)

    with pytest.raises(ValueError, match="only 'pdis' evaluates one"):
        certify_evaluation(MARKET, LOGGER, RAMP, method, 100)
    with pytest.raises(ValueError, match="only 'pdis' evaluates one"):
        evaluate_plan({"x": x[0], "u": u[0]}, RAMP, method, plant=MARKET, cost=COST)


def test_a_schedule_that_does_not_fit_the_episodes_is_refused() -> None:
    initial = InitialLaw(np.zeros(2), np.eye(2))
    x, u = _rollouts(MARKET, LOGGER, np.zeros((10, 2)), 4, np.random.default_rng(3))
    per_state = QuadraticCost(Q=COST.Q, R=COST.R, Qf=COST.Qf, x_target=jnp.zeros((5, 2)))

    with pytest.raises(ValueError, match="the schedule runs 5 steps and the episodes 4"):
        certify_evaluation(MARKET, LOGGER, RAMP, "pdis", 100, horizon=4, initial=initial)
    with pytest.raises(ValueError, match="the schedule runs 5 steps and the episodes 4"):
        evaluate_plan({"x": x, "u": u}, RAMP, "pdis", plant=MARKET, cost=COST)
    with pytest.raises(ValueError, match="gain has shape \\(1, 3\\)"):
        certify_evaluation(
            MARKET,
            LOGGER,
            AffineSchedule.open_loop(np.zeros((5, 1)), 3),
            "pdis",
            100,
            initial=initial,
        )
    with pytest.raises(
        ValueError, match="x_target has 5 rows, one per state, but 3 steps have 4 states"
    ):
        evaluate_plan({"x": x[:, :-1], "u": u[:, :-1]}, PLAN, "pdis", plant=MARKET, cost=per_state)
    fits = evaluate_plan(
        {"x": x, "u": u},
        LOGGER,
        "pdis",
        plant=MARKET,
        cost=per_state,
        logger=LOGGER,
        min_effective=1.0,
    )
    assert fits.certificate.effective_samples == pytest.approx(10.0, rel=1e-12)


def test_a_schedule_is_checked_like_a_policy() -> None:
    with pytest.raises(ValueError, match="gains must have shape \\(H, m, n\\)"):
        AffineSchedule(np.zeros((1, 2)), np.zeros((1, 1)), np.zeros((1, 1)))
    with pytest.raises(ValueError, match="offsets must have shape \\(5, 1\\)"):
        AffineSchedule(RAMP.gains, np.zeros((4, 1)), np.zeros((1, 1)))
    with pytest.raises(ValueError, match="not positive semidefinite"):
        AffineSchedule(RAMP.gains, RAMP.offsets, -np.eye(1))
    with pytest.raises(ValueError, match="actions must have shape \\(H, m\\)"):
        AffineSchedule.open_loop(np.zeros(5), 2)
    open_loop = AffineSchedule.open_loop(np.ones((3, 2)), 4, covariance=0.1 * np.eye(2))
    assert open_loop.horizon == 3
    assert open_loop.gains.shape == (3, 2, 4)
    assert not np.any(open_loop.gains)
    assert np.array_equal(open_loop.step(2).covariance, 0.1 * np.eye(2))


def test_logs_that_do_not_fit_the_method_are_refused() -> None:
    x, u = _stationary_logs(MARKET, LOGGER, 200, 1, seed=1)

    with pytest.raises(KeyError, match="missing \\['u'\\]"):
        evaluate_plan({"x": x[0]}, PLAN, "mis", plant=MARKET, cost=COST)
    with pytest.raises(ValueError, match="reads x \\(T \\+ 1, n\\)"):
        evaluate_plan({"x": x[0, 1:], "u": u[0]}, PLAN, "mis", plant=MARKET, cost=COST)
    with pytest.raises(ValueError, match="reads x \\(E, H \\+ 1, n\\)"):
        evaluate_plan({"x": x[0], "u": u[0]}, PLAN, "pdis", plant=MARKET, cost=COST)
    per_step = QuadraticCost(Q=COST.Q, R=COST.R, Qf=COST.Qf, x_target=jnp.zeros((201, 2)))
    with pytest.raises(ValueError, match="one row per state"):
        evaluate_plan({"x": x[0], "u": u[0]}, PLAN, "mis", plant=MARKET, cost=per_step)


def test_an_infeasible_evaluation_carries_its_certificate() -> None:
    plant = _scalar_plant(1.0, 0.5, 0.2)
    logger, candidate = _scalar_policy(-1.0, 0.0, 0.5), _scalar_policy(-0.4, 0.0, 0.6)
    x, u = _stationary_logs(plant, logger, 2000, 1, seed=8)
    cost = QuadraticCost(
        Q=jnp.array([[2.0]]), R=jnp.array([[2.0]]), Qf=jnp.array([[2.0]]), x_target=jnp.zeros(1)
    )

    with pytest.raises(InfeasibleEvaluation, match="2 R_x - P_x") as caught:
        evaluate_plan(
            {"x": x[0], "u": u[0]}, candidate, "mis", plant=plant, cost=cost, logger=logger
        )

    assert caught.value.certificate.one_step_margin == pytest.approx(0.088, abs=1e-9)
