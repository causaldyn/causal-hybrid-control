"""The tube reads the fitted field's error from the log's projected action, through the rk4 fixed
point, and with the push's mean read as the channel's error is (ADR 0133).

On a linear log, with a linear nuisance basis, one fold and no ridge, the Euler and the rk4 fits
are least squares on the one-step map, read as a rate or pushed through the inverse of the RK4
polynomial, so the field's first-order error at a state and an action is the delta method through
that map. Each test reads it by that route, without the fit's own reader: the channel's error, the
drift's own with the channel held, the covariance the two share, and the push's mean, each at its
own small-sample factor, from the rows, the clusters' sums or the periods' cosines. The action
enters through its distance from the log's projected action ``m(x)``, the action at which the
field's error is the drift's own: what the drift's features predict of the logged actions, under
``rk4`` through the drift's block of Newton's matrix.
"""

from __future__ import annotations

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import chi2

from chc.decision import _field_error, _field_errors
from chc.dynamics import HybridDynamics, LinearDynamics
from chc.dynamics_id import CausalDynamicsFit, fit_causal_residual
from chc.integrate import rk4_step

_KNOWN = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))  # no physics: one lever
# the fits the one-step map reads exactly: linear nuisances on one fold and no ridge
_EXACT = {
    "adjust_for": ("z",),
    "channel_degree": 0,
    "nuisance_degree": 1,
    "nuisance_knots": None,
    "folds": 1,
    "ridge": 0.0,
}


def _r(z: float) -> float:
    return 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0


def _s(z: float) -> float:
    return 1.0 + z / 2.0 + z**2 / 6.0 + z**3 / 24.0


def _r_slope(z: float) -> float:
    return 1.0 + z + z**2 / 2.0 + z**3 / 6.0


def _s_slope(z: float) -> float:
    return 0.5 + z / 3.0 + z**2 / 8.0


def _rk4_inverse(growth: float) -> float:
    """``z`` with ``R(z) = growth``, from ``log(growth)`` by Newton's steps."""
    z = math.log(growth)
    for _ in range(60):
        z -= (_r(z) - growth) / _r_slope(z)
    return z


def _flow_log(
    seed: int,
    *,
    dt: float = 1.0,
    decay: float = -0.7,
    gain: float = 0.8,
    push: float = 1.0,
    noise: float = 0.1,
    persistence: float = 0.8,
    rows: int = 200,
    levers: int = 1,
) -> dict[str, np.ndarray]:
    """``rows`` transitions, after 50 of burn-in, of the exact flow of
    ``x' = decay x + gain sum(u) + push z``, ``z`` an AR(``persistence``) series of unit variance
    held over each step, each logged lever ``-0.5 x + 0.8 z`` plus a draw of its own of spread
    0.5, the state's noise of spread ``noise`` a step."""
    rng = np.random.default_rng(seed)
    growth = math.exp(decay * dt)
    held = (growth - 1.0) / decay
    x, z = rng.normal(0.0, 0.5), rng.normal()
    log: dict[str, list] = {name: [] for name in ("x", "u", "z", "x_next")}
    for step in range(50 + rows):
        z = persistence * z + math.sqrt(1.0 - persistence**2) * rng.normal()
        u = -0.5 * x + 0.8 * z + 0.5 * rng.normal(size=levers)
        after = growth * x + held * (gain * float(np.sum(u)) + push * z) + noise * rng.normal()
        if step >= 50:
            for name, value in zip(log, (x, u, z, after), strict=True):
                log[name].append(value)
        x = after
    return {
        "x": np.asarray(log["x"])[:, None],
        "u": np.asarray(log["u"]).reshape(rows, levers),
        "z": np.asarray(log["z"])[:, None],
        "x_next": np.asarray(log["x_next"])[:, None],
    }


def _fit(log: dict[str, np.ndarray], dt: float, integrator: str, **options) -> CausalDynamicsFit:
    data = {name: jnp.asarray(column) for name, column in log.items()}
    known = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, log["u"].shape[1])))
    return fit_causal_residual(known, data, dt, integrator=integrator, **_EXACT, **options)


_READINGS = {
    "rows": {},
    "periods": {"periods": np.arange(200)},
    "clusters": {"clusters": np.arange(200) // 10},
}


def _terms(scores: np.ndarray, reading: str) -> tuple[np.ndarray, int | None]:
    """The scores' independent terms: their projections on the 13 cosines
    ``sqrt(2 / 13) cos(pi j (t + 1/2) / 200)`` of the 200 periods (Lazarus, Lewis, Stock and
    Watson 2018), or their sums over 20 clusters of 10 consecutive transitions; and the
    clusters' count."""
    rows = scores.shape[0]
    if reading == "periods":
        frequencies = np.arange(1, 14)[:, None]
        cosines = math.sqrt(2.0 / 13) * np.cos(
            math.pi * frequencies * (np.arange(rows) + 0.5) / rows
        )
        return cosines @ scores, None
    sums = scores.reshape(rows // 10, 10, -1).sum(axis=1)
    return sums, sums.shape[0]


def _factor(rows: int, coefficients: int, groups: int | None) -> float:
    """``N / (N - k)``, or CR1's ``G / (G - 1) (N - 1) / (N - k)`` over ``G`` clusters."""
    if groups is None:
        return rows / (rows - coefficients)
    return groups / (groups - 1) * (rows - 1) / (rows - coefficients)


def _by_the_one_step_map(
    log: dict[str, np.ndarray],
    dt: float,
    integrator: str,
    reading: str,
    state: float,
    box: tuple[np.ndarray, np.ndarray],
) -> tuple[float, float]:
    """The field's error variance at ``state``, the largest over the box's vertices, and
    ``drift_error``, by the delta method through the one-step map ``x_next = X rho``,
    ``X = [1, x, u, z - z_bar]``: ``theta = h(rho)``, ``J = d theta / d rho``, so each row moves
    the field's coefficients by ``J (X'X)^-1 X_i`` times its residual. Under ``rows`` the drift's
    own error and its covariance with the channel read the residuals' pooled variance, the
    channel's error each row's own, and the push's mean its rows as independent, as the fit
    reads them; otherwise every part is read from the clusters' sums or the periods' cosines."""
    x, u, z, after = (log[name] for name in ("x", "u", "z", "x_next"))
    rows, levers = u.shape
    x, z, after = x[:, 0], z[:, 0] - np.mean(z), after[:, 0]
    design = np.column_stack([np.ones(rows), x, u, z])
    inverse = np.linalg.inv(design.T @ design)
    rho = inverse @ design.T @ after
    residual = after - design @ rho
    drift_design = design[:, [0, 1, levers + 2]]
    weight = np.linalg.solve(drift_design.T @ drift_design, drift_design.T)
    gamma = weight @ u  # (3, m): what the drift's features predict of the logged actions
    if integrator == "euler":
        jacobian, growth, block = np.eye(levers + 3) / dt, 1.0, np.eye(3)
        push_gain = rho[levers + 2] / dt
    else:
        assert levers == 1
        zz = _rk4_inverse(rho[1])
        s, r_slope, s_slope = _s(zz), _r_slope(zz), _s_slope(zz)
        c, b, push_gain = (rho[i] / (dt * s) for i in (0, 2, 3))
        jacobian = np.linalg.inv(
            np.array(
                [
                    [dt * s, dt**2 * s_slope * c, 0.0, 0.0],
                    [0.0, dt * r_slope, 0.0, 0.0],
                    [0.0, dt**2 * s_slope * b, dt * s, 0.0],
                    [0.0, dt**2 * s_slope * push_gain, 0.0, dt * s],
                ]
            )
        )
        growth = s
        # the drift's block of Newton's matrix over (c, a, p), closed (rk4_inherited.mac (3))
        block = np.array(
            [
                [s, dt * s_slope * (c + b * gamma[0, 0]), 0.0],
                [0.0, r_slope + dt * s_slope * b * gamma[1, 0], 0.0],
                [0.0, dt * s_slope * (push_gain + b * gamma[2, 0]), s],
            ]
        )
    unblocked = np.linalg.inv(block)
    projected = np.array([1.0, state, 0.0]) @ unblocked @ (growth * gamma)  # m(x), (m,)
    moves = (jacobian @ inverse @ design.T).T  # (N, p): theta's move with each row's x_next
    channel = moves[:, 2 : 2 + levers] * residual[:, None]
    own = moves @ np.concatenate([[1.0, state], projected, [0.0]]) * residual
    pushed = push_gain * z
    drift = (unblocked @ weight * (residual / dt)).T[:, :2]  # the carried drift's, (c, a)
    if reading == "rows":
        pooled = residual @ residual / (rows - 3)
        own_moves = own / residual
        own_part = pooled * own_moves @ own_moves
        shared = pooled * (channel / residual[:, None]).T @ own_moves
        spread = _factor(rows, levers, None) * channel.T @ channel
        push_part = pushed @ pushed / rows**2
        gram = np.linalg.inv(drift_design.T @ drift_design)
        drift_block = pooled / dt**2 * (unblocked @ gram @ unblocked.T)[:2, :2]
    else:
        terms, groups = _terms(np.column_stack([channel, own, pushed, drift]), reading)
        on_channel, on_own, on_push = terms[:, :levers], terms[:, levers], terms[:, levers + 1]
        on_drift = terms[:, levers + 2 :]
        channel_factor, drift_factor = (_factor(rows, k, groups) for k in (levers, 3))
        own_part = drift_factor * on_own @ on_own
        shared = math.sqrt(channel_factor * drift_factor) * on_channel.T @ on_own
        spread = channel_factor * on_channel.T @ on_channel
        push_part = _factor(rows, 1, groups) * on_push @ on_push / rows**2
        drift_block = drift_factor * on_drift.T @ on_drift
    lo, hi = box
    largest = -math.inf
    for vertex in itertools.product(*zip(lo, hi, strict=True)):
        away = np.asarray(vertex) - projected
        largest = max(largest, own_part + away @ spread @ away + 2.0 * away @ shared)
    features = np.column_stack([np.ones(rows), x])
    drift_error = math.sqrt(np.mean(np.einsum("ia,ab,ib->i", features, drift_block, features)))
    return largest + push_part, drift_error


def test_euler_and_rk4_fits_of_a_linear_log_predict_one_step_alike() -> None:
    """In a linear world both fits are least squares on the one-step map, the rk4 fit's pushed
    through the RK4 polynomial's inverse (``rk4_inherited.mac`` (2)): stepped by its own integrator
    each predicts every transition the other does, to rounding, while their fields differ."""
    log = _flow_log(0)
    states, actions = jnp.asarray(log["x"]), jnp.asarray(log["u"])
    euler, rk4 = (
        HybridDynamics(known=_KNOWN, residual=_fit(log, 1.0, integrator).residual)
        for integrator in ("euler", "rk4")
    )
    by_euler = jax.vmap(lambda x, u: x + euler(0.0, x, u))(states, actions)
    by_rk4 = jax.vmap(lambda x, u: rk4_step(rk4, 0.0, x, u, 1.0))(states, actions)
    scale = float(jnp.max(jnp.abs(by_euler)))
    np.testing.assert_allclose(by_rk4, by_euler, rtol=0.0, atol=1e-12 * scale)
    # and they are not one field: Euler reads the step's map, (exp(a dt) - 1)/dt, not a
    slopes = [float(model.residual.drift[0, 1]) for model in (euler, rk4)]
    assert slopes[0] > -0.55 > -0.65 > slopes[1]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("reading", ["rows", "periods", "clusters"])
@pytest.mark.parametrize("integrator", ["euler", "rk4"])
def test_the_field_s_error_is_the_delta_method_through_the_one_step_map(
    integrator: str, reading: str, seed: int
) -> None:
    """At a state 2.5 of the log's spreads from its mean, the field's error the tube reads over a
    box of the lever is the delta method's through the one-step map at the box's vertex farther
    from the projected action, to rounding: within ``reach`` of the logged mean, and over a box
    not centred there. On this log, ``dt = 1`` and ``a dt = -0.7`` with an AR(0.8) push, up to
    0.21.0 the tube read the channel's error at the distance from the mean and the field's at the
    mean in quadrature, under ``rk4`` with none of what the field inherits, and given periods or
    clusters it read the drift's error and the push's mean from rows taken as independent."""
    log = _flow_log(seed)
    fit = _fit(log, 1.0, integrator, **_READINGS[reading])
    state = float(np.mean(log["x"]) + 2.5 * np.std(log["x"]))
    centre = np.mean(log["u"], axis=0)
    at, none = np.array([[state]]), np.zeros((1, 0))
    variance, drift_error = _by_the_one_step_map(
        log, 1.0, integrator, reading, state, (centre - 2.0, centre + 2.0)
    )
    read = float(_field_errors(fit, np.array([2.0]), at, none)[0])
    assert read**2 == pytest.approx(variance, rel=1e-9, abs=0.0)
    assert fit.drift_error == pytest.approx(drift_error, rel=1e-9, abs=0.0)
    box = (centre - 0.5, centre + 2.5)
    variance, _ = _by_the_one_step_map(log, 1.0, integrator, reading, state, box)
    read = float(_field_errors(fit, np.array([2.5]), at, none, box)[0])
    assert read**2 == pytest.approx(variance, rel=1e-9, abs=0.0)


def test_two_levers_read_the_box_s_vertex_farthest_from_the_projected_action() -> None:
    """With two levers the field's error is ``(u - m(x))' C (u - m(x))`` with the two channels'
    covariance ``C`` whole, beside the drift's own, the covariance they share and the push's mean,
    and the tube reads it at the box's vertex where it is largest, here by the delta method through
    the one-step map; at every planned state too, and over the log's rows as its root mean
    square."""
    log = _flow_log(4, dt=0.5, persistence=0.0, levers=2)
    fit = _fit(log, 0.5, "euler")
    centre = np.mean(log["u"], axis=0)
    box = (centre + np.array([-1.0, -0.5]), centre + np.array([2.0, 1.5]))
    reach = np.maximum(np.abs(box[0] - centre), np.abs(box[1] - centre))
    states = np.linspace(-1.0, 1.5, 5)[:, None]
    expected = [
        _by_the_one_step_map(log, 0.5, "euler", "rows", float(state[0]), box)[0] for state in states
    ]
    read = _field_errors(fit, reach, states, np.zeros((5, 0)), box)
    np.testing.assert_allclose(read**2, expected, rtol=1e-9, atol=0.0)
    rows = np.asarray(log["x"])
    over_rows = _field_errors(fit, reach, rows, np.zeros((rows.shape[0], 0)), box)
    assert _field_error(fit, reach, box) == pytest.approx(
        math.sqrt(np.mean(over_rows**2)), rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("seed_block", [0, 1])
def test_rk4_s_tube_reads_the_field_s_miss_far_from_the_log_at_its_own_error(
    seed_block: int,
) -> None:
    """Over 40 exact-flow logs of 200 transitions of ``x' = -0.7 x + 0.8 u + z`` at ``dt = 1``,
    the logged ``u`` leaning on the state by -0.5, fitted by ``rk4``, the field's miss at a state
    2.5 of the log's spreads from its mean and the log's mean action, against the field whose RK4
    step is the plant's flow, scaled by the error the tube reads there, has a root mean square
    inside the band 40 standard normals hold on 0.999 of draws, ``chi2_40`` at 0.0005 and 0.9995.
    Up to 0.21.0 the tube read the drift's own error there, which the channel's error reaches
    through the fixed point: over 400 logs the miss read 1.88 of it, and ``+-1.96`` held 0.70 of
    them; now 1.03 and 0.935. On these two blocks of 40 it reads 1.14 and 0.85, inside the band
    0.65 to 1.38, where it read 2.05 and 1.53."""
    low, high = (math.sqrt(chi2.ppf(q, 40) / 40) for q in (0.0005, 0.9995))
    growth = math.exp(-0.7)
    z = _rk4_inverse(growth)
    slope, gain = z, (growth - 1.0) / -0.7 * 0.8 / _s(z)
    scaled = []
    for seed in range(40 * seed_block, 40 * seed_block + 40):
        log = _flow_log(seed, persistence=0.0)
        fit = _fit(log, 1.0, "rk4")
        state = float(np.mean(log["x"]) + 2.5 * np.std(log["x"]))
        action = float(np.mean(log["u"]))
        miss = float(fit.residual(0.0, jnp.array([state]), jnp.array([action]))[0]) - (
            slope * state + gain * action
        )
        error = float(_field_errors(fit, np.zeros(1), np.array([[state]]), np.zeros((1, 0)))[0])
        scaled.append(miss / error)
    rms = math.sqrt(float(np.mean(np.square(scaled))))
    assert low < rms < high, rms
