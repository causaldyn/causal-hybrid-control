"""The plan's tube carries the plant's own one-step noise as well as the channel's error (ADR 0080).

Fed the channel's standard error alone, `prescribe`'s tube read about 0.05 K one step ahead on logs
shaped like the BOPTEST case's, where the plant's error one step ahead reached 0.16 to 0.22 K at
95 %: the plant left the tube at the first step on 60 % of draws while the certificate trusted 6 to
13 steps. The tube now adds the
radius that holds the noise's part of the gap at each step with probability 0.95, the noise read
off the fit's one-step defect on the log. These tests measure that rate on plants whose noise is
known, rather than assume it.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import chi2

from chc import CausalGraph, Lever, Panel, Target, prescribe
from chc.cost import QuadraticCost
from chc.dynamics import LinearDynamics
from chc.dynamics_id import fit_causal_residual
from chc.integrate import rk4_step
from chc.mpc import RecedingHorizon
from chc.plan import causal_plan
from chc.uncertainty import _linear_tube, _tube

_COVERAGE = 0.95
_DRAWS = 20_000
# The Monte Carlo standard error of a coverage of 0.95 over that many draws, 0.0015.
_SE = math.sqrt(_COVERAGE * (1.0 - _COVERAGE) / _DRAWS)


def _plan(state_matrix: np.ndarray, lipschitz: object, **noise: object):
    """A plan on the linear model ``x' = A x + u`` from ``x0 = 1``, steering to 0 for 12 steps of
    0.2, its tube the noise's alone."""
    states = state_matrix.shape[0]
    model = LinearDynamics(jnp.asarray(state_matrix), jnp.eye(states, 1))
    cost = QuadraticCost(
        Q=jnp.eye(states), R=0.1 * jnp.eye(1), Qf=jnp.eye(states), x_target=jnp.zeros(states)
    )
    plan = causal_plan(
        model, jnp.ones(states), cost, 0.2, 12, -1.0, 1.0, lipschitz=lipschitz, **noise
    )
    return model, plan


def _realised(model, plan, noise: np.ndarray, persistence: float, seed: int) -> np.ndarray:
    """The plant's distance from the plan at each step over ``_DRAWS`` runs: the model's own RK4
    step from the plant's state under the plan's actions, plus a Gaussian error of covariance
    ``noise``, AR(1) at ``persistence`` and stationary from the first step."""
    rng = np.random.default_rng(seed)
    root = np.linalg.cholesky(noise)
    step = jax.vmap(lambda x, u: rk4_step(model, 0.0, x, u, 0.2), in_axes=(0, None))
    states = np.tile(np.asarray(plan.trajectory[0]), (_DRAWS, 1))
    error = rng.standard_normal((_DRAWS, noise.shape[0])) @ root.T
    distances = []
    for k, action in enumerate(np.asarray(plan.actions)):
        states = np.asarray(step(jnp.asarray(states), jnp.asarray(action))) + error
        distances.append(np.linalg.norm(states - np.asarray(plan.trajectory[k + 1]), axis=1))
        fresh = rng.standard_normal((_DRAWS, noise.shape[0])) @ root.T
        error = persistence * error + math.sqrt(1.0 - persistence**2) * fresh
    return np.stack(distances, axis=1)


@pytest.mark.parametrize("persistence", [0.0, 0.8, -0.6])
def test_carried_by_the_state_matrix_the_tube_holds_one_state_at_its_rate(
    persistence: float,
) -> None:
    """On a plant the model is, the gap is the noise's alone and Gaussian, so the tube through RK4's
    step of the state matrix holds it with probability 0.95 at every step: within four Monte Carlo
    standard errors, persistent noise and noise that alternates included."""
    a_matrix = np.array([[-0.5]])
    noise = np.array([[0.01]])
    model, plan = _plan(a_matrix, a_matrix, noise=noise, noise_persistence=persistence)
    assert plan.uncertainty_tube is not None
    tube = np.asarray(plan.uncertainty_tube)
    inside = np.mean(_realised(model, plan, noise, persistence, seed=1) <= tube[1:], axis=0)
    assert np.all(np.abs(inside - _COVERAGE) <= 4.0 * _SE), inside


@pytest.mark.parametrize("persistence", [0.0, 0.8])
def test_carried_by_a_norm_bound_the_tube_holds_at_least_its_rate_and_on_two_states_its_rate(
    persistence: float,
) -> None:
    """A norm-Lipschitz rate bounds the step's slope: it holds the plant at least 0.95 of the
    time, exactly at the first step, where nothing has been carried yet. Carried by the state
    matrix on two states, the covariance is exact and so is the radius, the quantile of a sum of
    two unequal chi-squares: it holds the plant at 0.95 within four Monte Carlo standard errors at
    every step. Up to 0.17.0 the radius read the largest variance along any direction, and held it
    at 0.979 to 0.986."""
    one = np.array([[-0.5]])
    model, plan = _plan(one, 0.5, noise=0.01, noise_persistence=persistence)
    tube = np.asarray(plan.uncertainty_tube)
    inside = np.mean(_realised(model, plan, np.array([[0.01]]), persistence, seed=2) <= tube[1:], 0)
    assert np.all(inside >= _COVERAGE - 4.0 * _SE), inside
    assert abs(inside[0] - _COVERAGE) <= 4.0 * _SE

    two = np.array([[-0.5, 1.0], [0.0, -0.3]])
    noise = np.array([[0.01, 0.004], [0.004, 0.02]])
    model, plan = _plan(two, two, noise=noise, noise_persistence=persistence)
    tube = np.asarray(plan.uncertainty_tube)
    inside = np.mean(_realised(model, plan, noise, persistence, seed=3) <= tube[1:], axis=0)
    assert np.all(np.abs(inside - _COVERAGE) <= 4.0 * _SE), inside


def test_the_noise_radius_is_the_derivations_covariance() -> None:
    """``validation/predictive_tube.mac``'s values, to 40 digits there: the largest eigenvalue of
    the covariance carried by RK4's step of a state matrix, and its bound carried by a
    norm-Lipschitz rate, which reads the persistence's size whatever its sign."""
    from chc.uncertainty import _noise_covariances, _noise_tube

    matrix, noise = np.array([[-1.0, 0.5], [0.0, -2.0]]), np.array([[0.04, 0.01], [0.01, 0.02]])
    exact = [np.linalg.eigvalsh(p)[-1] for p in _noise_covariances(matrix, noise, 0.5, 0.1, 4)]
    np.testing.assert_allclose(
        exact,
        [
            0.0,
            4.414213562373095e-2,
            1.2060596895879581e-1,
            2.0194292561614788e-1,
            2.774381666236308e-1,
        ],
        rtol=1e-13,
        atol=0.0,
    )
    for persistence in (0.5, -0.5):
        bound = _noise_tube(np.asarray(1.0), np.array([[0.04]]), persistence, 0.1, 4)
        np.testing.assert_allclose(
            bound**2 / chi2.ppf(_COVERAGE, 1),
            [0.0, 0.04, 1.3306293616736111e-1, 2.7115829707010393e-1, 4.533269108698939e-1],
            rtol=1e-13,
            atol=0.0,
        )


@pytest.mark.parametrize("lipschitz", [0.5, np.array([[-0.5]])], ids=["norm", "matrix"])
def test_without_noise_the_tube_is_the_model_errors_alone_to_the_bit(lipschitz: object) -> None:
    """No noise, whatever its persistence, leaves the tube the one the model error alone gives."""
    a_matrix = np.array([[-0.5]])
    _, before = _plan(a_matrix, lipschitz, model_error=0.05)
    _, after = _plan(a_matrix, lipschitz, model_error=0.05, noise=0.0, noise_persistence=0.7)
    errors = [0.05] * 12
    alone = (
        _tube([0.5] * 12, errors, 0.2, "rk4")
        if np.ndim(lipschitz) == 0
        else _linear_tube(a_matrix, errors, 0.2)
    )
    assert before.uncertainty_tube is not None
    assert after.uncertainty_tube is not None
    assert np.array_equal(np.asarray(after.uncertainty_tube), alone)
    assert np.array_equal(np.asarray(before.uncertainty_tube), alone)
    assert after.certified_horizon == before.certified_horizon


def test_noise_alone_switches_the_tube_on() -> None:
    """With no model error the tube is the noise's: 1.96 standard deviations at the first step."""
    _, plan = _plan(np.array([[-0.5]]), 0.5, noise=0.01, tolerance=0.3)
    assert plan.certificate_status == "partial"
    assert plan.uncertainty_tube is not None
    assert float(plan.uncertainty_tube[1]) == pytest.approx(
        math.sqrt(chi2.ppf(_COVERAGE, 1)) * 0.1, rel=1e-14, abs=0.0
    )


def test_the_receding_horizon_controller_plans_with_the_noise() -> None:
    """Each step of the controller is ``causal_plan`` from that state, the noise's tube included."""
    model, plan = _plan(np.array([[-0.5]]), 0.5, noise=0.01, noise_persistence=0.6)
    controller = RecedingHorizon(
        model,
        QuadraticCost(Q=jnp.eye(1), R=0.1 * jnp.eye(1), Qf=jnp.eye(1), x_target=jnp.zeros(1)),
        dt=0.2,
        horizon=12,
        u_lo=-1.0,
        u_hi=1.0,
        lipschitz=0.5,
        noise=0.01,
        noise_persistence=0.6,
    )
    stepped = controller.step(jnp.ones(1))
    assert stepped.uncertainty_tube is not None
    assert plan.uncertainty_tube is not None
    assert np.array_equal(np.asarray(stepped.uncertainty_tube), np.asarray(plan.uncertainty_tube))


@pytest.mark.parametrize(
    ("noise", "persistence", "message"),
    [
        (-0.01, 0.0, "not a covariance"),
        (np.array([[0.01, 0.005], [0.0, 0.01]]), 0.0, "not a covariance"),
        (np.array([[0.01, 0.02], [0.02, 0.01]]), 0.0, "not a covariance"),
        (math.nan, 0.0, "not finite"),
        (np.eye(3), 0.0, r"shape \(2, 2\)"),
        (0.01, 1.5, "not a correlation"),
        (0.01, math.nan, "not a correlation"),
    ],
    ids=["negative", "asymmetric", "indefinite", "nan", "wrong size", "past one", "nan rho"],
)
def test_a_noise_that_is_no_covariance_is_refused(
    noise: object, persistence: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _plan(np.array([[-0.5, 1.0], [0.0, -0.3]]), 1.0, noise=noise, noise_persistence=persistence)


def _one_unit_log(persistence: float, seed: int) -> dict[str, np.ndarray]:
    """960 half-hours of one unit: ``x' = -0.4 x + 0.8 u + 0.2 z``, ``z`` a logged confounder drawn
    afresh each step and ``u = 0.5 z`` plus a draw of its own; each step adds an error of spread
    0.1, AR(1) at ``persistence``."""
    rng = np.random.default_rng(seed)
    rows: dict[str, list[float]] = {name: [] for name in ("unit", "time", "x", "u", "z")}
    x, error = 0.0, rng.normal(0.0, 0.1)
    for period in range(960):
        z = rng.normal()
        u = 0.5 * z + rng.normal(0.0, 0.4)
        for name, value in (("unit", 0), ("time", period), ("x", x), ("u", u), ("z", z)):
            rows[name].append(value)
        x = float(_plant_step(np.array([x]), u, np.array([z]))[0]) + error
        error = persistence * error + math.sqrt(1.0 - persistence**2) * rng.normal(0.0, 0.1)
    return {name: np.asarray(values) for name, values in rows.items()}


def _plant_step(x: np.ndarray, u: float, z: np.ndarray) -> np.ndarray:
    """RK4 over half an hour of ``x' = -0.4 x + 0.8 u + 0.2 z``, ``u`` and ``z`` held."""

    def field(state: np.ndarray) -> np.ndarray:
        return -0.4 * state + 0.8 * u + 0.2 * z

    k1 = field(x)
    k2 = field(x + 0.25 * k1)
    k3 = field(x + 0.25 * k2)
    k4 = field(x + 0.5 * k3)
    return x + 0.5 / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _prescribe(log: dict[str, np.ndarray], **kwargs: object):
    return prescribe(
        Panel.from_frame(log, unit="unit", time="time", seed=0),
        levers=[Lever("u", lo=-2.0, hi=2.0)],
        target=Target("x", value=1.0),
        adjustment=CausalGraph.from_edges([("z", "u"), ("z", "x"), ("u", "x")]),
        dt=0.5,
        x0=[float(log["x"][-1])],
        **kwargs,  # type: ignore[arg-type]
    )


# the confounder's push over a step, 0.5 phi(-0.2) 0.2 z, of spread 0.0906, beside the error's 0.1
_PUSH = 0.5 * (1.0 - 0.1 + 0.04 / 6.0 - 0.008 / 24.0) * 0.2


def test_prescribe_reads_the_noise_off_the_log_and_its_tube_holds_the_plant() -> None:
    """The fitted field leaves the confounder's push and the error in its one-step defect, which
    the certificate reports and the tube carries. Run on from the log's last state under the
    plan's actions, the confounder drawn afresh and the error too, the plant stays inside the tube
    at least 0.95 of the time at every step, within four Monte Carlo standard errors. The defect is
    in-sample: the fitted channel, off the truth by its estimation error, explains a little of the
    push, so the noise reads between the error's spread and the push's and error's together."""
    log = _one_unit_log(0.0, seed=4)
    decided = _prescribe(log, horizon=12, tolerance=10.0)
    assert decided.plan is not None
    tube = np.asarray(decided.plan.uncertainty_tube)
    predicted = np.asarray(decided.plan.trajectory)[:, 0]
    rng = np.random.default_rng(5)
    state = np.full(_DRAWS, predicted[0])
    inside = []
    for k, action in enumerate(np.asarray(decided.plan.actions)[:, 0]):
        error = rng.normal(0.0, 0.1, _DRAWS)
        state = _plant_step(state, float(action), rng.standard_normal(_DRAWS)) + error
        inside.append(np.mean(np.abs(state - predicted[k + 1]) <= tube[k + 1]))
    assert np.all(np.asarray(inside) >= _COVERAGE - 4.0 * _SE), inside

    certificate, fit = decided.certificate, decided.model_fit
    assert fit.defect is not None
    one_step = 0.5 * np.asarray(fit.defect, dtype=np.float64)[:, 0]
    assert certificate.tube_noise == pytest.approx(
        math.sqrt(np.mean(one_step**2)), rel=1e-12, abs=0.0
    )
    assert 0.1 < certificate.tube_noise < math.sqrt(_PUSH**2 + 0.01)
    assert certificate.tube_persistence is not None
    assert abs(certificate.tube_persistence) < 0.1
    record = decided.to_json()["certificate"]
    assert record["tube_noise"] == certificate.tube_noise
    assert record["tube_persistence"] == certificate.tube_persistence
    (line,) = [line for line in decided.report().splitlines() if "the tube's noise" in line]
    assert f"standard deviation {certificate.tube_noise:.4g}" in line


def test_prescribe_reads_the_noise_s_persistence_off_the_log() -> None:
    """An error AR(0.7) under a push drawn afresh persists in the defect at about
    ``0.7 * 0.01 / (push^2 + 0.01)``, 0.385; the drift fitted on a persistent error reads a little
    of it off the state (ADR 0064), so the defect's own reads lower. Coverage under persistence is
    measured on many logs, not this one: a fresh draw of the error is not this log's continuation,
    and the plant's continuation is one draw of its history."""
    decided = _prescribe(_one_unit_log(0.7, seed=4), horizon=12, tolerance=10.0)
    certificate = decided.certificate
    assert certificate.tube_persistence is not None
    expected = 0.7 * 0.01 / (_PUSH**2 + 0.01)
    assert certificate.tube_persistence == pytest.approx(expected, rel=0.0, abs=0.1)
    assert decided.to_json()["certificate"]["tube_persistence"] == certificate.tube_persistence
    (line,) = [line for line in decided.report().splitlines() if "the tube's noise" in line]
    assert f"lag-1 autocorrelation {certificate.tube_persistence:+.2f}" in line


def test_no_tolerance_reads_no_noise() -> None:
    decided = _prescribe(_one_unit_log(0.0, seed=6), horizon=4)
    certificate = decided.certificate
    assert (certificate.certified_horizon, certificate.tube_noise) == (None, None)
    assert certificate.tube_persistence is None
    assert not [line for line in decided.report().splitlines() if "the tube's noise" in line]


def test_a_fit_with_no_channel_error_reads_no_noise_and_trusts_no_step() -> None:
    """Asserted with nothing adjusted for, the fit reports no channel error, so nothing bounds the
    field the plan rolls out; the noise alone must not switch on a tube that would trust it."""
    log = _one_unit_log(0.0, seed=8)
    decided = prescribe(
        Panel.from_frame(log, unit="unit", time="time", seed=0),
        levers=[Lever("u", lo=-2.0, hi=2.0)],
        target=Target("x", value=1.0),
        adjustment=(),
        dt=0.5,
        horizon=4,
        tolerance=10.0,
        x0=[float(log["x"][-1])],
    )
    assert decided.model_fit.channel_error is None
    certificate = decided.certificate
    assert (certificate.certified_horizon, certificate.tube_noise) == (None, None)
    assert (certificate.tube_persistence, certificate.trustworthy_steps) == (None, 0)


def test_the_fit_keeps_its_defect_row_by_row() -> None:
    """``integrator_defect`` is the root mean square of the rows the fit now keeps."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(400, 1))
    z = rng.normal(size=(400, 1))
    u = 0.5 * z + rng.normal(0.0, 0.4, (400, 1))
    x_next = x + 0.1 * (-0.4 * x + 0.8 * u + 0.6 * z) + rng.normal(0.0, 0.01, (400, 1))
    data = {"x": jnp.asarray(x), "u": jnp.asarray(u), "x_next": jnp.asarray(x_next)}
    data["z"] = jnp.asarray(z)
    fit = fit_causal_residual(
        LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1))), data, 0.1, adjust_for=("z",)
    )
    assert fit.defect is not None
    assert fit.defect.shape == (400, 1)
    assert fit.integrator_defect == float(jnp.sqrt(jnp.mean(jnp.asarray(fit.defect) ** 2)))


def _load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "bench_boptest_tube.py"
    spec = importlib.util.spec_from_file_location("bench_boptest_tube", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_on_logs_like_the_boptest_case_it_trusts_no_step_the_plant_leaves() -> None:
    """The case's first call on three logs of each plant: a one-state zone and a zone fed through
    a floor, the noise persisting AR(0.9). The plant run on from each log's end under the plan
    stays within the tolerance of 0.5 K, at 95 % of the draws over the logs, for as many steps as
    its error allows; the certificate trusts at least one step and none past them, where it trusted
    6 to 13 when its tube read the channel's error alone."""
    script = _load_script()
    cells = script.cover("first", ["one,0.0", "slab,0.9"], logs=3, seed=20261012, draws=100)
    for cell, read in cells.items():
        assert read["trusted"]["min"] >= 1, cell
        assert read["trusted"]["max"] <= read["allowed"], (cell, read["trusted"], read["allowed"])
