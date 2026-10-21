"""The tube reads the fitted field's error where the plan goes, at its studentised quantile,
beside the noise.

The field's error the tube carries is a standard error the log estimates, from the cosines or the
clusters its spread sums and from the drift regression's residuals. Up to 0.18.0 the tube added
one of it, carried, to the radius that holds the plant's noise at 0.95, and read it as a root mean
over the log's rows: on one-unit logs of 30 transitions of an AR(1) plant, where the channel's
error is read from three cosines and the plan heads for a target 1.7 of the log's spreads away,
the plant left the tube at its eighth step on 16 % of runs. The tube is now the 0.95 radius of the
noise's covariance with the field's carried error added along its widest direction as a Gaussian
of standard deviation ``kappa f_k``, ``kappa = t_nu(0.975) / z(0.975)`` at the degrees of freedom
the channel's error is read with: on one state ``sqrt(r_k^2 + t_nu^2 f_k^2)``
(``validation/tube_level.mac``); and ``f_k`` carries the field's error read at the state each of
the plan's steps starts from.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm, t

from chc import CausalGraph, Constraint, Lever, Panel, Target, prescribe
from chc.decision import (
    _field_error,
    _field_errors,
    _field_scale,
    _model_error,
    _model_errors,
    _slopes,
)
from chc.dynamics import HybridDynamics, LinearDynamics
from chc.dynamics_id import CausalDynamicsFit, _cosine_count, fit_causal_residual
from chc.plan import _error_tube
from chc.residual import ControlAffineResidual
from chc.uncertainty import _linear_tube, _noise_covariances, _noise_radius, _noise_tube, _tube


def _one_state_log(seed: int, periods: int) -> dict[str, np.ndarray]:
    """One unit of ``x' = -0.4 x + (0.8 + 0.3 x) u + 0.2 z`` at half-hour Euler steps, ``z`` a
    logged confounder and ``u = 0.5 z`` plus a draw of its own, an error of spread 0.1 a step."""
    rng = np.random.default_rng(seed)
    rows: dict[str, list[float]] = {name: [] for name in ("unit", "time", "x", "u", "z")}
    x = 0.0
    for period in range(periods):
        z = rng.normal()
        u = 0.5 * z + rng.normal()
        for name, value in (("unit", 0), ("time", period), ("x", x), ("u", u), ("z", z)):
            rows[name].append(value)
        x = x + 0.5 * (-0.4 * x + (0.8 + 0.3 * x) * u + 0.2 * z) + rng.normal(0.0, 0.1)
    return {name: np.asarray(values) for name, values in rows.items()}


def _two_state_log(seed: int, periods: int) -> dict[str, np.ndarray]:
    """One unit of two states whose channel is affine in the state, at half-hour Euler steps."""
    rng = np.random.default_rng(seed)
    x = np.zeros(2)
    rows: dict[str, list[float]] = {name: [] for name in ("unit", "time", "x1", "x2", "u", "z")}
    for period in range(periods):
        z = rng.normal()
        u = 0.5 * z - 0.3 * x[0] + 0.4 * rng.normal()
        for name, value in zip(rows, (0, period, x[0], x[1], u, z), strict=True):
            rows[name].append(float(value))
        rate = np.array(
            [
                -0.4 * x[0] + 0.2 * x[1] + (0.8 + 0.3 * x[0]) * u + 0.2 * z,
                -0.3 * x[1] + (0.5 - 0.2 * x[1]) * u + 0.1 * z,
            ]
        )
        x = x + 0.5 * rate + 0.1 * rng.normal(size=2)
    return {name: np.asarray(values) for name, values in rows.items()}


def _read_at(
    fit: CausalDynamicsFit, log: dict[str, np.ndarray], states: list[str], reach: np.ndarray, at
) -> np.ndarray:
    """The field's error at each of the ``(H, n)`` states ``at``, read here from an Euler fit on
    ``log`` without the tube's reader: the drift's as its own least squares' standard error at
    each state, ``sigma^2 phi' (X'X)^-1 phi``, its features ``[1, x]`` on the log's rows and
    ``sigma^2`` the defect's mean square over ``(N - F) n``, the defect being that regression's
    residual under Euler; the channel's as its coefficients' covariance read at the state, its
    root mean over the log's rows the channel's error the fit reports at ``reach``. The two in
    quadrature."""
    assert fit.integrator == "euler"
    assert fit.defect is not None
    assert fit._channel_blocks is not None
    rows = np.stack([log[name][:-1] for name in states], 1)
    design = np.concatenate([np.ones((rows.shape[0], 1)), rows], 1)
    defect = np.asarray(fit.defect, dtype=np.float64)
    sigma2 = np.sum(defect**2) / ((design.shape[0] - design.shape[1]) * defect.shape[1])
    gram = np.linalg.inv(design.T @ design)
    points = np.concatenate([np.ones((len(at), 1)), np.asarray(at)], 1)
    drift = np.sqrt(sigma2 * np.einsum("kf,fh,kh->k", points, gram, points))
    (blocks,) = fit._channel_blocks  # one covariance: one unit's cosines

    def channel(features: np.ndarray) -> np.ndarray:
        variances = np.einsum("kf,efh,kh->ke", features, blocks, features)
        return reach[0] * np.sqrt(np.mean(variances, 1))

    at_rows = channel(design)
    assert math.sqrt(np.mean(at_rows**2)) == pytest.approx(
        _model_error(fit, reach), rel=1e-10, abs=0.0
    )
    return np.hypot(channel(points), drift)


def test_the_field_s_error_at_a_state_is_the_euler_fit_s_own_standard_error_there() -> None:
    """Fitted as Euler's map, the drift is least squares on ``[1, x]``, and its error at a state is
    that regression's standard error there; the channel's is its coefficients' covariance read at
    the state. Read without the tube's reader, the two in quadrature are what the tube reads at
    each of the states a plan of 8 steps starts from, and the plan heads away from the log's
    states: its late steps read more error than the rows' root mean."""
    log = _one_state_log(5, 61)
    known = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    fit = fit_causal_residual(
        known,
        {
            "x": jnp.asarray(log["x"][:-1, None]),
            "u": jnp.asarray(log["u"][:-1, None]),
            "x_next": jnp.asarray(log["x"][1:, None]),
            "z": jnp.asarray(log["z"][:-1, None]),
        },
        0.5,
        adjust_for=("z",),
        periods=np.arange(60),
        integrator="euler",
    )
    mean = float(np.mean(log["u"][:-1]))
    reach = np.array([max(abs(-2.0 - mean), abs(2.0 - mean))])
    away = np.linspace(float(np.mean(log["x"])), 3.0, 8)[:, None]
    errors = _read_at(fit, log, ["x"], reach, away)
    np.testing.assert_allclose(
        _field_errors(fit, reach, away, np.zeros((8, 0))), errors, rtol=1e-6, atol=0.0
    )
    assert np.all(errors[-3:] > _field_error(fit, reach))


def test_one_state_s_tube_is_the_noise_s_radius_and_t_times_the_field_s_in_quadrature() -> None:
    """On one unit of 60 transitions the channel's error is read from six cosines: the tube is
    ``sqrt(r_k^2 + t_6(0.975)^2 f_k^2)``, ``r_k`` the noise's exact radius and ``f_k`` the field's
    standard error at the state each step starts from, carried by the fitted slope at each planned
    action, where it read ``r_k + f_k`` with the field's error a root mean over the log's rows."""
    log = _one_state_log(5, 61)
    horizon, dt = 8, 0.5
    decided = prescribe(
        Panel.from_frame(log, unit="unit", time="time", seed=0),
        levers=[Lever("u", lo=-2.0, hi=2.0)],
        target=Target("x", value=1.0),
        adjustment=CausalGraph.from_edges([("z", "u"), ("z", "x"), ("u", "x")]),
        dt=dt,
        horizon=horizon,
        tolerance=10.0,
    )
    plan, fit, certificate = decided.plan, decided.model_fit, decided.certificate
    assert plan is not None
    assert plan.uncertainty_tube is not None
    assert certificate.tube_mode == "exact"
    assert certificate.tube_noise is not None
    assert certificate.tube_autoregression is not None
    carried = np.asarray(certificate.tube_autoregression)  # the VAR(1) the tube carries (ADR 0105)
    assert fit.drift_error is not None
    problem = plan._problem
    assert problem is not None
    slopes = _slopes(problem.model, problem.x0)(np.asarray(plan.actions, dtype=np.float64))
    mean = float(np.mean(log["u"][:-1]))
    reach = np.array([max(abs(-2.0 - mean), abs(2.0 - mean))])
    assert _cosine_count(60) == 6
    nu = 6.0
    errors = _field_errors(fit, reach, np.asarray(plan.trajectory)[:-1], np.zeros((horizon, 0)))
    field = _linear_tube(slopes, errors, dt)
    noise = _noise_tube(slopes, np.array([[certificate.tube_noise**2]]), carried, dt, horizon)
    expected = np.sqrt(noise**2 + (t.ppf(0.975, nu) * field) ** 2)
    tube = np.asarray(plan.uncertainty_tube)
    np.testing.assert_allclose(tube, expected, rtol=1e-8, atol=0.0)
    assert np.all(tube[1:] < noise[1:] + field[1:] * t.ppf(0.975, nu))
    assert float(t.ppf(0.975, nu)) > 2.2  # a level a normal's 1.96 would read short of


def test_two_states_place_the_field_s_error_along_the_noise_s_widest_direction() -> None:
    """On two states the field's carried error, of a direction the fit does not give, is a
    Gaussian of standard deviation ``kappa f_k`` along the noise's widest direction, where it
    widens the radius most: ``(kappa f_k)^2`` is added to the largest eigenvalue of the noise's
    covariance ``P_k`` before its exact 0.95 radius is taken, ``kappa = t_nu(0.975) / z(0.975)``
    as on one state. The radius lies between the noise's own and that of ``(kappa f_k)^2`` added on
    every state."""
    log = _two_state_log(3, 121)
    horizon, dt = 10, 0.5
    decided = prescribe(
        Panel.from_frame(log, unit="unit", time="time", seed=0),
        levers=[Lever("u", lo=-2.0, hi=2.0)],
        target=Target("x1", value=1.0),
        adjustment=CausalGraph.from_edges(
            [("z", "u"), ("z", "x1"), ("z", "x2"), ("u", "x1"), ("u", "x2"), ("x2", "x1")]
        ),
        constraints=[Constraint("x2", hi=50.0)],
        dt=dt,
        horizon=horizon,
        tolerance=10.0,
    )
    plan, fit = decided.plan, decided.model_fit
    assert decided.certificate.tube_mode == "exact"
    assert plan is not None
    assert plan.uncertainty_tube is not None
    assert fit.drift_error is not None
    assert fit.defect is not None
    problem = plan._problem
    assert problem is not None
    slopes = _slopes(problem.model, problem.x0)(np.asarray(plan.actions, dtype=np.float64))
    mean = float(np.mean(log["u"][:-1]))
    reach = np.array([max(abs(-2.0 - mean), abs(2.0 - mean))])
    kappa = t.ppf(0.975, float(_cosine_count(120))) / norm.ppf(0.975)
    assert _field_scale(fit) == pytest.approx(kappa, rel=1e-12, abs=0.0)
    errors = _field_errors(fit, reach, np.asarray(plan.trajectory)[:-1], np.zeros((horizon, 0)))
    field = _linear_tube(slopes, errors, dt)
    error = dt * np.asarray(fit.defect, dtype=np.float64)
    covariance = error.T @ error / error.shape[0]
    carried = decided.certificate.tube_autoregression
    assert carried is not None
    persistence = np.asarray(carried)  # the VAR(1) the tube carries (ADR 0105)
    covariances = _noise_covariances(slopes, covariance, persistence, dt, horizon)
    expected, noise, everywhere = [], [], []
    for p, spread in zip(covariances, field, strict=True):
        values, vectors = np.linalg.eigh(p)
        widest = vectors[:, -1]
        expected.append(_noise_radius(p + (kappa * spread) ** 2 * np.outer(widest, widest)))
        noise.append(_noise_radius(p))
        everywhere.append(_noise_radius(p + (kappa * spread) ** 2 * np.eye(2)))
        # past the start, a widest direction to place it along
        assert not p.any() or values[0] < 0.9 * values[-1]
    tube = np.asarray(plan.uncertainty_tube)
    np.testing.assert_allclose(tube, expected, rtol=1e-6, atol=0.0)
    assert np.all(tube[1:] > np.asarray(noise)[1:])
    assert np.all(tube[1:] < np.asarray(everywhere)[1:])


def _covered(radius: float, field: float, nu: float) -> float:
    """``P(|n + field T| <= radius)``, ``n`` a standard normal and ``T`` Student's at ``nu``."""

    def inside(value: float) -> float:
        return float(
            (norm.cdf(radius - field * value) - norm.cdf(-radius - field * value))
            * t.pdf(value, nu)
        )

    return integrate.quad(inside, -np.inf, np.inf, epsabs=1e-13, epsrel=1e-12, limit=400)[0]


@pytest.mark.parametrize("nu", [3.0, 9.0, 52.0])
@pytest.mark.parametrize("field", [0.5, 2.0, 8.0])
def test_the_level_holds_a_studentised_error_beside_gaussian_noise(nu: float, field: float) -> None:
    """One step from a start the plan knows, the gap is the noise, of variance 1, plus the field's
    error, ``f`` times Student's ``t``: the tube's radius holds it with probability 0.9495 to
    0.9525 at every ratio, where the noise's radius plus ``f`` holds 0.758 at ``nu = 3`` and
    ``f = 4``."""
    scale = float(t.ppf(0.975, nu) / norm.ppf(0.975))
    tube = _error_tube(np.zeros((1, 1, 1)), field, np.eye(1), 0.0, 1.0, 1, scale)
    assert tube is not None
    assert tube[1] == pytest.approx(
        math.sqrt(norm.ppf(0.975) ** 2 + (t.ppf(0.975, nu) * field) ** 2), rel=1e-12, abs=0.0
    )
    assert 0.9495 <= _covered(float(tube[1]), field, nu) <= 0.9525
    added = _error_tube(np.zeros((1, 1, 1)), 4.0, np.eye(1), 0.0, 1.0, 1)
    assert added is not None
    assert _covered(float(added[1]), 4.0, 3.0) == pytest.approx(0.7583, abs=1e-4)


def test_a_norm_bound_adds_the_field_s_variance_to_the_noise_s_largest() -> None:
    """Under a norm bound the noise's largest variance ``p_k`` is bounded by its recursion, the
    noise read as ``p_k I``, and the field's part, ``s_k = kappa f_k``, joins one of its equal
    directions: the radius of ``diag(p_k + s_k^2, p_k)`` past the first step, and at it that of the
    noise ``S`` with ``s_1^2`` added to its largest eigenvalue. Read as a bound, the field's part is
    added to the noise's radius as before."""
    noise = np.array([[0.02, 0.005], [0.005, 0.01]])
    rate, dt, horizon, error, scale = 0.4, 0.5, 5, 0.03, 1.3
    tube = _error_tube(np.array(rate), error, noise, 0.3, dt, horizon, scale)
    assert tube is not None
    field = _tube([rate] * horizon, [error] * horizon, dt, "rk4")
    z = rate * dt
    growth = 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0
    largest = float(np.linalg.eigvalsh(noise)[-1])
    spread, cross, expected = 0.0, 0.0, [0.0]
    for step in range(1, horizon + 1):
        spread, cross = (
            growth * growth * spread + 2.0 * growth * cross + largest,
            0.3 * (growth * cross + largest),
        )
        expected.append(_noise_radius(np.diag([spread + (scale * field[step]) ** 2, spread])))
    widest = np.linalg.eigh(noise)[1][:, -1]
    expected[1] = _noise_radius(noise + (scale * field[1]) ** 2 * np.outer(widest, widest))
    np.testing.assert_allclose(tube, expected, rtol=1e-10, atol=0.0)
    bound = _error_tube(np.array(rate), error, noise, 0.3, dt, horizon)
    assert bound is not None
    np.testing.assert_allclose(
        bound, field + _noise_tube(np.array(rate), noise, 0.3, dt, horizon), rtol=0.0, atol=0.0
    )


def _fit(channel: float, drift: float | None, rows: int) -> CausalDynamicsFit:
    """A fit of one state and one lever on ``rows`` transitions of one unit, whose channel's error
    at a reach of 1 is ``channel`` and whose drift's is ``drift``, its drift two coefficients."""
    return CausalDynamicsFit(
        residual=ControlAffineResidual(drift=jnp.zeros((1, 2)), channel=jnp.zeros((1, 1, 2))),
        identified=True,
        method="orthogonal",
        folds=2,
        channel_error=channel,
        drift_error=drift,
        action_residual_variance=1.0,
        nuisance_r2_state=0.0,
        nuisance_r2_action=0.0,
        moment_norm=0.0,
        defect=jnp.zeros((rows, 1)),
        periods=np.arange(rows),
        _lever_errors=np.array([[channel]]),
    )


def test_the_scale_reads_the_channel_s_degrees_of_freedom() -> None:
    """The field's error is read at the degrees of freedom of the channel's: its cosines on one
    unit (six on 60 transitions, ADR 0075), its clusters less one on many; the drift's error,
    read from residuals taken as independent, does not raise them, however large its share."""
    for channel, drift in ((0.2, 0.1), (0.2, None), (0.01, 1.0)):
        assert _field_scale(_fit(channel, drift, 60)) == pytest.approx(
            t.ppf(0.975, 6.0) / norm.ppf(0.975), rel=1e-12, abs=0.0
        )
    many = dataclasses.replace(_fit(0.2, 0.1, 60), periods=None, clusters=np.arange(60)[:, None])
    assert _field_scale(many) == pytest.approx(
        t.ppf(0.975, 59.0) / norm.ppf(0.975), rel=1e-12, abs=0.0
    )


def test_the_tube_reads_the_same_in_any_units_of_the_state() -> None:
    """The field's part and the noise's move together with the state's units, and so does their
    radius: a log in thousandths of the state's units reads a tube 1000 times as wide."""
    log = _one_state_log(5, 61)
    tubes = []
    for unit in (1.0, 1e3):
        scaled = dict(log, x=unit * log["x"])
        decided = prescribe(
            Panel.from_frame(scaled, unit="unit", time="time", seed=0),
            levers=[Lever("u", lo=-2.0, hi=2.0)],
            target=Target("x", value=unit),
            adjustment=CausalGraph.from_edges([("z", "u"), ("z", "x"), ("u", "x")]),
            dt=0.5,
            horizon=6,
            tolerance=10.0 * unit,
        )
        assert decided.plan.uncertainty_tube is not None
        tubes.append(np.asarray(decided.plan.uncertainty_tube) / unit)
    np.testing.assert_allclose(tubes[1], tubes[0], rtol=1e-6, atol=0.0)


def test_rk4_steps_the_same_field_through_its_own_matrices() -> None:
    """The tube carries both parts through RK4's step matrices of the slope at each action. The
    field's error is read at each planned state off the
    covariances the fit reads at the log's rows: there, its root mean square over the rows is the
    field's error the fit reports, the channel's at its reach and the drift's beside it."""
    log = _one_state_log(5, 61)
    decided = prescribe(
        Panel.from_frame(log, unit="unit", time="time", seed=0),
        levers=[Lever("u", lo=-2.0, hi=2.0)],
        target=Target("x", value=1.0),
        adjustment=CausalGraph.from_edges([("z", "u"), ("z", "x"), ("u", "x")]),
        dt=0.5,
        horizon=6,
        tolerance=10.0,
    )
    plan, fit, certificate = decided.plan, decided.model_fit, decided.certificate
    assert plan.uncertainty_tube is not None
    assert certificate.tube_noise is not None
    assert certificate.tube_autoregression is not None
    carried = np.asarray(certificate.tube_autoregression)  # the VAR(1) the tube carries (ADR 0105)
    known = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    model = HybridDynamics(known=known, residual=fit.residual)
    x0 = jnp.asarray(decided.start.value)
    slopes = np.array(
        [jax.jacobian(lambda x, u=u: model(0.0, x, u))(x0) for u in jnp.asarray(plan.actions)]
    )
    mean = float(np.mean(log["u"][:-1]))
    reach = np.array([max(abs(-2.0 - mean), abs(2.0 - mean))])
    rows = log["x"][:-1, None]
    at_rows = _field_errors(fit, reach, rows, np.zeros((rows.shape[0], 0)))
    assert math.sqrt(np.mean(at_rows**2)) == pytest.approx(
        _field_error(fit, reach), rel=1e-10, abs=0.0
    )
    planned = np.asarray(plan.trajectory)[:-1]
    errors = _field_errors(fit, reach, planned, np.zeros((6, 0)))
    assert np.all(errors[-2:] > _field_error(fit, reach))
    field = _linear_tube(slopes, errors, 0.5)
    noise = _noise_tube(slopes, np.array([[certificate.tube_noise**2]]), carried, 0.5, 6)
    scale = _field_scale(fit)
    np.testing.assert_allclose(
        np.asarray(plan.uncertainty_tube),
        np.sqrt(noise**2 + (scale * norm.ppf(0.975) * field) ** 2),
        rtol=1e-10,
        atol=0.0,
    )


def test_the_field_s_error_far_from_the_log_is_read_at_its_own_standard_error() -> None:
    """Over 40 logs of 200 transitions of ``x' = -0.5 x + u + 0.8 z``, fitted as Euler's map, the
    fitted field's error at a state 2.5 of the log's spreads from its mean, scaled by the standard
    error read there, has a root mean square near one, the drift's at the log's mean action and the
    channel's alike. Scaled by their root means over the log's rows, as the tube read them up to
    0.18.0, they read about twice that: the error grows away from the log's states."""
    dt, rows = 0.2, 200
    known = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    drift, channel, drift_rows, channel_rows = [], [], [], []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0, 0.5)
        log: dict[str, list[float]] = {name: [] for name in ("x", "u", "z", "x_next")}
        for _ in range(rows):
            z = rng.normal()
            u = -0.5 * x + 0.8 * z + 0.5 * rng.normal()
            after = x + dt * (-0.5 * x + u + 0.8 * z) + 0.05 * rng.normal()
            for name, value in zip(log, (x, u, z, after), strict=True):
                log[name].append(value)
            x = after
        fit = fit_causal_residual(
            known,
            {name: jnp.asarray(values)[:, None] for name, values in log.items()},
            dt,
            adjust_for=("z",),
            integrator="euler",
            periods=np.arange(rows),
        )
        assert fit.drift_error is not None
        centre = float(np.mean(log["u"]))
        far = np.array([[np.mean(log["x"]) + 2.5 * np.std(log["x"])]])

        def rate(u: float, far: np.ndarray = far, fit: CausalDynamicsFit = fit) -> float:
            return float(fit.residual(0.0, jnp.asarray(far[0]), jnp.array([u]))[0])

        off = rate(centre) - (-0.5 * far[0, 0] + centre)
        drift.append(off / _field_errors(fit, np.zeros(1), far, np.zeros((1, 0)))[0])
        drift_rows.append(off / fit.drift_error)
        moved = rate(1.0) - rate(0.0) - 1.0
        channel.append(moved / _model_errors(fit, np.ones(1), far)[0])
        channel_rows.append(moved / _model_error(fit, np.ones(1)))

    def rms(values: list[float]) -> float:
        return math.sqrt(float(np.mean(np.square(values))))

    assert 0.75 < rms(drift) < 1.4
    assert 0.75 < rms(channel) < 1.5
    assert rms(drift_rows) > 1.7
    assert rms(channel_rows) > 2.0


def test_a_fit_that_keeps_no_covariances_reads_its_root_means_at_every_state() -> None:
    """A fit built without the covariances behind its errors reads the field's error as a root
    mean over the log's rows at every state, as the tube read it up to 0.18.0; no channel error,
    no field error."""
    fit = _fit(0.2, 0.1, 60)
    states = np.array([[0.0], [3.0], [-7.0]])
    # np.hypot and math.hypot may differ in the last place across platforms' libm
    np.testing.assert_allclose(
        _field_errors(fit, np.array([2.0]), states, np.zeros((3, 0))),
        np.full(3, math.hypot(0.4, 0.1)),
        rtol=1e-15,
        atol=0.0,
    )
    no_channel = dataclasses.replace(fit, _lever_errors=None)
    np.testing.assert_array_equal(
        _field_errors(no_channel, np.array([2.0]), states, np.zeros((3, 0))), np.zeros(3)
    )


def test_under_rk4_each_state_s_drift_error_reads_its_own_block() -> None:
    """Under ``rk4`` the drift's covariance is one block a state, and its error at a state is the
    scale of the Euclidean norm over them, their mean variance times their number (ADR 0100), as
    the tube reads ``drift_error`` over the log's rows: on two states, the drift's part read at
    each row has the root mean square ``sqrt(2)`` times ``drift_error`` over the rows."""
    log = _two_state_log(3, 241)
    states = np.stack([log["x1"], log["x2"]], 1)
    known = LinearDynamics(jnp.zeros((2, 2)), jnp.zeros((2, 1)))
    fit = fit_causal_residual(
        known,
        {
            "x": jnp.asarray(states[:-1]),
            "u": jnp.asarray(log["u"][:-1, None]),
            "x_next": jnp.asarray(states[1:]),
            "z": jnp.asarray(log["z"][:-1, None]),
        },
        0.5,
        adjust_for=("z",),
        periods=np.arange(240),
        integrator="rk4",
    )
    assert fit.integrator == "rk4"
    assert fit.drift_error is not None
    assert fit._drift_blocks is not None
    assert fit._drift_blocks.shape[0] == 2
    assert not np.allclose(fit._drift_blocks[0], fit._drift_blocks[1], rtol=0.1, atol=0.0)
    drift = _field_errors(fit, np.zeros(1), states[:-1], np.zeros((240, 0)))
    assert math.sqrt(np.mean(drift**2)) == pytest.approx(
        math.sqrt(2.0) * fit.drift_error, rel=1e-10, abs=0.0
    )
