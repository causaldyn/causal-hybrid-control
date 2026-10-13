"""The tube carried by the fitted field's own step matrices along the plan, at its exact radius.

On a field affine in the state RK4 steps the gap between the plant and the plan by ``R(dt A_k)``,
``A_k`` the field's slope at the action of step ``k``. ``prescribe`` carried the tube by a bound on
the slope's norm over the levers' box instead, which grew it by 1.065 a week on the quickstart's
fitted field where the step matrices grow it by 1.027 at most, and read the noise's radius as
``sqrt(chi2_n(0.95) lambda_max)``, which held the quickstart's plant at 0.986 at the first week
where the tube states 0.95. The quickstart trusted 1 week of the 3 its plant allows. The norm
bound's slack also hid the drift's own error, which the tube did not carry: it now carries it
beside the channel's (ADR 0096).
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import chi2

from chc import CausalGraph, Constraint, Lever, Panel, Target, prescribe
from chc.cost import QuadraticCost
from chc.decision import _field_error, _model_error, _slopes, _state_noise
from chc.dynamics import HybridDynamics, LinearDynamics
from chc.dynamics_id import CausalDynamicsFit
from chc.integrate import rk4_step, rollout
from chc.plan import _carried_along, causal_plan
from chc.residual import ControlAffineResidual
from chc.uncertainty import _linear_tube, _noise_covariances, _noise_radius, _noise_tube

_COVERAGE = 0.95
_DRAWS = 20_000
# The Monte Carlo standard error of a coverage of 0.95 over that many draws, 0.0015.
_SE = math.sqrt(_COVERAGE * (1.0 - _COVERAGE) / _DRAWS)


@pytest.mark.parametrize("states", [1, 2, 3])
def test_a_repeated_state_matrix_carries_the_one_matrix_s_tube_to_the_bit(states: int) -> None:
    rng = np.random.default_rng(states)
    a_matrix = rng.normal(size=(states, states)) - np.eye(states)
    errors = rng.uniform(0.01, 0.1, 9).tolist()
    repeated = np.repeat(a_matrix[None], 9, axis=0)
    assert np.array_equal(_linear_tube(repeated, errors, 0.3), _linear_tube(a_matrix, errors, 0.3))
    root = rng.normal(size=(states, states))
    noise = root @ root.T / states
    assert np.array_equal(
        _noise_tube(repeated, noise, 0.4, 0.3, 9), _noise_tube(a_matrix, noise, 0.4, 0.3, 9)
    )


def test_a_sequence_s_tube_carries_each_step_s_error_through_the_later_steps_matrices() -> None:
    """Step ``j``'s error reaches step ``k`` through ``R(z_{k-1}) ... R(z_{j+1}) c_i(z_j)``,
    ``z = dt A``: the tube is the sum of those gains' norms times each step's error, read here off
    the products written out step by step, on matrices that do not commute."""
    rng = np.random.default_rng(5)
    dt, steps = 0.3, 6
    matrices = rng.normal(size=(steps, 2, 2))
    errors = rng.uniform(0.01, 0.1, steps)
    eye = np.eye(2)

    def step(z: np.ndarray) -> np.ndarray:
        return eye + z + z @ z / 2.0 + z @ z @ z / 6.0 + z @ z @ z @ z / 24.0

    def weights(z: np.ndarray) -> list[np.ndarray]:
        return [
            (eye + z + z @ z / 2.0 + z @ z @ z / 4.0) / 6.0,
            (2.0 * eye + z + z @ z / 2.0) / 6.0,
            (2.0 * eye + z) / 6.0,
            eye / 6.0,
        ]

    expected = [0.0]
    for k in range(1, steps + 1):
        total = 0.0
        for j in range(k):
            carried = eye
            for later in range(j + 1, k):
                carried = step(dt * matrices[later]) @ carried
            gain = sum(np.linalg.norm(carried @ w, 2) for w in weights(dt * matrices[j]))
            total += errors[j] * dt * gain
        expected.append(total)
    tube = _linear_tube(matrices, errors.tolist(), dt)
    np.testing.assert_allclose(tube, expected, rtol=1e-12, atol=0.0)
    reversed_order = _linear_tube(matrices[::-1], errors.tolist(), dt)
    assert np.max(np.abs(reversed_order - tube)) > 1e-3 * tube[-1]


@pytest.mark.parametrize(
    "a_matrix", [np.array([[-0.5]]), np.array([[-0.5, 1.0], [0.0, -0.3]])], ids=["one", "two"]
)
def test_prescribe_s_slopes_on_a_linear_model_give_causal_plan_s_matrix_tube_to_the_bit(
    a_matrix: np.ndarray,
) -> None:
    """``prescribe`` reads the slope at each planned action off the model; on a linear model it is
    the state matrix at every action, and the plan's tube is ``causal_plan``'s matrix tube to the
    bit, the model error's part and the noise's."""
    states = a_matrix.shape[0]
    model = LinearDynamics(jnp.asarray(a_matrix), jnp.eye(states, 1))
    cost = QuadraticCost(
        Q=jnp.eye(states), R=0.1 * jnp.eye(1), Qf=jnp.eye(states), x_target=jnp.zeros(states)
    )
    x0 = jnp.ones(states)
    noise = np.array([[0.01]]) if states == 1 else np.array([[0.02, 0.003], [0.003, 0.002]])
    common = {"model_error": 0.05, "noise": noise, "noise_persistence": 0.6, "tolerance": 0.4}
    public = causal_plan(model, x0, cost, 0.2, 12, -1.0, 1.0, lipschitz=a_matrix, **common)
    by_norm = causal_plan(model, x0, cost, 0.2, 12, -1.0, 1.0, lipschitz=1.0, **common)
    private = _carried_along(
        by_norm,
        _slopes(model, x0)(np.asarray(by_norm.actions, dtype=np.float64)),
        0.05,
        noise,
        0.6,
        0.2,
        0.4,
    )
    assert np.array_equal(np.asarray(private.actions), np.asarray(by_norm.actions))
    assert public.uncertainty_tube is not None
    assert private.uncertainty_tube is not None
    assert np.array_equal(np.asarray(private.uncertainty_tube), np.asarray(public.uncertainty_tube))
    assert private.certified_horizon == public.certified_horizon


def _field(states: int):
    """``x' = A x + (b + C x) u``: one lever whose push moves with the state, so the slope in the
    state, ``A + u C``, moves with the action."""
    if states == 1:
        drift, channel = np.array([[0.0, -0.5]]), np.array([[[1.0, 0.8]]])
    else:
        drift = np.array([[0.0, -0.5, 1.0], [0.0, 0.0, -0.3]])
        channel = np.array([[[1.0, 0.6, 0.0]], [[0.5, 0.4, -0.5]]])
    known = LinearDynamics(jnp.zeros((states, states)), jnp.zeros((states, 1)))
    residual = ControlAffineResidual(drift=jnp.asarray(drift), channel=jnp.asarray(channel))
    return HybridDynamics(known=known, residual=residual)


@pytest.mark.parametrize("persistence", [0.0, 0.7])
@pytest.mark.parametrize("states", [1, 2])
def test_the_tube_holds_a_bilinear_plant_at_its_rate_under_switching_actions(
    states: int, persistence: float
) -> None:
    """On a plant the model is, under actions that switch the slope from growing to contracting,
    the gap is Gaussian with the covariance the step matrices carry, so the tube holds it with
    probability 0.95 at every step, within four Monte Carlo standard errors, on one state and on
    two, where the radius is the exact quantile of a sum of two unequal chi-squares."""
    model, dt = _field(states), 0.2
    actions = np.array([1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 0.5, -1.0, 1.0, -0.5, 1.0, -1.0])[:, None]
    x0 = jnp.full(states, 0.5)
    noise = np.array([[0.01]]) if states == 1 else np.array([[0.02, 0.003], [0.003, 0.002]])
    slopes = _slopes(model, x0)(actions)
    assert np.ptp(slopes[:, 0, 0]) > 1.0  # the slope switches with the action
    tube = _noise_tube(slopes, noise, persistence, dt, actions.shape[0])
    planned = np.asarray(rollout(model, x0, jnp.asarray(actions), dt))
    rng = np.random.default_rng(11 + states)
    root = np.linalg.cholesky(noise)
    step = jax.vmap(lambda x, u: rk4_step(model, 0.0, x, u, dt), in_axes=(0, None))
    plant = np.tile(np.asarray(x0), (_DRAWS, 1))
    error = rng.standard_normal((_DRAWS, states)) @ root.T
    inside = []
    for k, action in enumerate(actions):
        plant = np.asarray(step(jnp.asarray(plant), jnp.asarray(action))) + error
        inside.append(np.mean(np.linalg.norm(plant - planned[k + 1], axis=1) <= tube[k + 1]))
        fresh = rng.standard_normal((_DRAWS, states)) @ root.T
        error = persistence * error + math.sqrt(1.0 - persistence**2) * fresh
    assert np.all(np.abs(np.asarray(inside) - _COVERAGE) <= 4.0 * _SE), inside


def test_the_noise_radius_is_the_exact_quantile_of_the_gaussian_s_distance() -> None:
    """``validation/exact_tube.mac``: at each pair of eigenvalues the radius holds the Gaussian with
    probability 0.95 to 1.4e-11 there, by a one-dimensional integral; at the quickstart's first
    week the old ``chi2_2(0.95) lambda_max`` held it with probability 0.986. On one state and on
    equal eigenvalues the radius is ``sqrt(chi2_n(0.95) lambda)`` to the bit; it reads only the
    eigenvalues, and moves as the root of a scale of the covariance."""
    for values, squared in (
        ((0.5, 1.0), 4.628283006555677),
        ((0.01, 1.0), 3.851522417726763),
        ((9.98697012e-5, 2.12624274e-2), 0.08177890584991816),
    ):
        radius = _noise_radius(np.diag(values))
        assert radius**2 == pytest.approx(squared, rel=1e-12, abs=0.0), values
        assert math.sqrt(chi2.ppf(_COVERAGE, 1) * max(values)) < radius
        assert radius < math.sqrt(chi2.ppf(_COVERAGE, 2) * max(values))
        turn = np.array([[0.6, -0.8], [0.8, 0.6]])
        turned = _noise_radius(turn @ np.diag(values) @ turn.T)
        assert turned == pytest.approx(radius, rel=1e-12, abs=0.0)
        for scale in (1e-6, 1e6):
            scaled = _noise_radius(scale * np.diag(values))
            assert scaled == pytest.approx(math.sqrt(scale) * radius, rel=1e-12, abs=0.0)
    assert _noise_radius(np.array([[0.04]])) == math.sqrt(chi2.ppf(_COVERAGE, 1) * 0.04)
    for states in (2, 3):
        assert _noise_radius(0.7 * np.eye(states)) == math.sqrt(chi2.ppf(_COVERAGE, states) * 0.7)
    assert _noise_radius(np.zeros((2, 2))) == 0.0
    # on a norm bound the covariance is exact at the first step alone, where it is the noise
    noise = np.array([[0.02, 0.003], [0.003, 0.002]])
    bound = _noise_tube(np.asarray(0.5), noise, 0.0, 0.2, 3)
    assert bound[1] == _noise_radius(noise)
    growth = 1.0 + 0.1 + 0.01 / 2.0 + 0.001 / 6.0 + 0.0001 / 24.0
    largest = float(np.linalg.eigvalsh(noise)[-1])
    assert bound[2] == pytest.approx(
        math.sqrt(chi2.ppf(_COVERAGE, 2) * largest * (1.0 + growth**2)), rel=1e-14, abs=0.0
    )


def test_a_state_in_other_units_moves_the_noise_s_covariance_to_d_p_d() -> None:
    """``validation/exact_tube.mac`` (3): with state ``i`` in units ``d_i`` times its own, each
    step matrix reads ``D M D^-1`` and the noise ``D S D``, so the covariance carried along the
    plan is ``D P D`` at every step, and each state's own radius moves by its ``d_i`` alone."""
    model = _field(2)
    actions = np.array([1.0, -1.0, 0.5, -1.0, 1.0, 1.0])[:, None]
    slopes = _slopes(model, jnp.zeros(2))(actions)
    noise = np.array([[0.02, 0.003], [0.003, 0.002]])
    units = np.diag([1e3, 1e-2])
    moved = units @ slopes @ np.linalg.inv(units)
    before = _noise_covariances(slopes, noise, 0.5, 0.2, 6)
    after = _noise_covariances(moved, units @ noise @ units, 0.5, 0.2, 6)
    for old, new in zip(before, after, strict=True):
        np.testing.assert_allclose(new, units @ old @ units, rtol=1e-12, atol=0.0)
    own = np.asarray(_state_noise(slopes, noise, 0.5, 0.2, 6))
    marginal = np.array([np.diag(covariance) for covariance in before])
    np.testing.assert_allclose(own, np.sqrt(chi2.ppf(_COVERAGE, 1) * marginal), rtol=1e-14, atol=0)
    moved_own = np.asarray(_state_noise(moved, units @ noise @ units, 0.5, 0.2, 6))
    np.testing.assert_allclose(moved_own, own * np.diag(units), rtol=1e-12, atol=0.0)


def _fit(errors: np.ndarray | None, drift_error: float | None) -> CausalDynamicsFit:
    """A fit of one state and two levers whose channel's error, read lever by lever off each
    covariance, is ``errors``, and whose drift's is ``drift_error``; nothing else of it is read."""
    return CausalDynamicsFit(
        residual=ControlAffineResidual(drift=jnp.zeros((1, 2)), channel=jnp.zeros((1, 2, 2))),
        identified=errors is not None,
        method="orthogonal",
        folds=2,
        channel_error=None if errors is None else float(np.sqrt(np.mean(errors**2, 1)).max()),
        drift_error=drift_error,
        action_residual_variance=1.0,
        nuisance_r2_state=0.0,
        nuisance_r2_action=0.0,
        moment_norm=0.0,
        _lever_errors=errors,
    )


def test_the_field_s_error_adds_the_drift_s_own_to_the_channel_s_in_quadrature() -> None:
    """The drift absorbed the channel at the log's mean action, so the field's error at an action
    is the drift's there plus the channel's times the distance: read at that centre they are
    uncorrelated to first order, and add in quadrature. A fit with no channel error carries no
    tube, whatever its drift's error."""
    errors = np.array([[3.0, 0.0], [0.0, 1.0], [1.0, 2.0]])
    reach = np.array([2.0, 5.0])
    channel = _model_error(_fit(errors, None), reach)
    assert channel == pytest.approx(math.sqrt(52.0), rel=1e-15, abs=0.0)
    assert _field_error(_fit(errors, None), reach) == channel
    assert _field_error(_fit(errors, 5.0), reach) == math.hypot(channel, 5.0)
    assert _field_error(_fit(errors, 5.0), np.zeros(2)) == 5.0
    assert _field_error(_fit(None, 5.0), reach) == 0.0


def _bilinear_log(seed: int, periods: int = 480) -> dict[str, np.ndarray]:
    """One unit of ``x' = -0.4 x + (0.8 + 0.3 x) u + 0.2 z`` at half-hour steps, ``z`` a logged
    confounder drawn afresh each step and ``u = 0.5 z`` plus a draw of its own, with an error of
    spread 0.1 a step."""
    rng = np.random.default_rng(seed)
    rows: dict[str, list[float]] = {name: [] for name in ("unit", "time", "x", "u", "z")}
    x = 0.0
    for period in range(periods):
        z = rng.normal()
        u = 0.5 * z + rng.normal(0.0, 0.4)
        for name, value in (("unit", 0), ("time", period), ("x", x), ("u", u), ("z", z)):
            rows[name].append(value)

        def field(state: float, u: float = u, z: float = z) -> float:
            return -0.4 * state + (0.8 + 0.3 * state) * u + 0.2 * z

        k1 = field(x)
        k2 = field(x + 0.25 * k1)
        k3 = field(x + 0.25 * k2)
        k4 = field(x + 0.5 * k3)
        x = x + 0.5 / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4) + rng.normal(0.0, 0.1)
    return {name: np.asarray(values) for name, values in rows.items()}


def test_prescribe_carries_its_tube_by_the_slope_at_each_planned_action() -> None:
    """On a field affine in the state whose channel moves with it, each step's matrix is the
    fitted slope at that step's action, read off the fitted field here; the tube's rate error is
    the channel's at the lever's distance from the log's mean action with the drift's own beside
    it. The certificate names the radii ``exact``, in its JSON and its report, and gives each
    state's own 0.95 radius of the noise, which on one state is the tube's noise part."""
    log = _bilinear_log(3)
    horizon, dt = 12, 0.5
    decided = prescribe(
        Panel.from_frame(log, unit="unit", time="time", seed=0),
        levers=[Lever("u", lo=-2.0, hi=2.0)],
        target=Target("x", value=1.0),
        adjustment=CausalGraph.from_edges([("z", "u"), ("z", "x"), ("u", "x")]),
        dt=dt,
        x0=[float(log["x"][-1])],
        horizon=horizon,
        tolerance=10.0,
    )
    plan, fit, certificate = decided.plan, decided.model_fit, decided.certificate
    assert plan is not None
    assert plan.uncertainty_tube is not None
    assert certificate.tube_noise is not None
    assert certificate.tube_persistence is not None
    known = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    model = HybridDynamics(known=known, residual=fit.residual)
    x0 = jnp.asarray([float(log["x"][-1])])
    slopes = np.array(
        [jax.jacobian(lambda x, u=u: model(0.0, x, u))(x0) for u in jnp.asarray(plan.actions)]
    )
    mean = float(np.mean(log["u"][:-1]))
    distance = np.array([max(abs(-2.0 - mean), abs(2.0 - mean))])
    error = _field_error(fit, distance)
    assert error > _model_error(fit, distance)
    noise = np.array([[certificate.tube_noise**2]])
    persistence = certificate.tube_persistence
    expected = _linear_tube(slopes, [error] * horizon, dt)
    expected += _noise_tube(slopes, noise, persistence, dt, horizon)
    tube = np.asarray(plan.uncertainty_tube)
    np.testing.assert_allclose(tube, expected, rtol=1e-12, atol=0.0)
    # the slope at the box's centre, the same at every step, is another tube: the action matters
    at_centre = np.asarray(jax.jacobian(lambda x: model(0.0, x, jnp.zeros(1)))(x0))
    centre = np.repeat(at_centre[None], horizon, axis=0)
    held = _linear_tube(centre, [error] * horizon, dt)
    held += _noise_tube(centre, noise, persistence, dt, horizon)
    assert np.max(np.abs(held - tube) / tube.clip(1e-300)) > 1e-3

    assert certificate.tube_mode == "exact"
    assert certificate.tube_rate == "global"
    record = decided.to_json()["certificate"]
    assert record["tube_mode"] == "exact"
    noise_part = _noise_tube(slopes, noise, persistence, dt, horizon)
    assert certificate.tube_state_noise is not None
    assert len(certificate.tube_state_noise) == horizon + 1
    np.testing.assert_allclose(
        [radii[0] for radii in certificate.tube_state_noise], noise_part, rtol=1e-12, atol=0.0
    )
    assert record["tube_state_noise"] == [list(radii) for radii in certificate.tube_state_noise]
    (line,) = [line for line in decided.report().splitlines() if "error tube" in line]
    assert "exact radii" in line


class _Bent(eqx.Module):
    """Known physics whose slope in the state changes from one state to the next."""

    def __call__(self, t: float | jax.Array, x: jax.Array, u: jax.Array) -> jax.Array:
        return -0.3 * jnp.tanh(x)


def test_a_field_whose_slope_moves_with_the_state_keeps_the_norm_bound() -> None:
    """Bent known physics has its slope read at the start alone: the tube stays carried by the norm
    bound, its radii a ``bound``, and no state's own radius is reported, since the bound holds the
    largest variance alone. Without a tolerance no tube is evaluated and no mode is named."""
    log = _bilinear_log(5, periods=240)
    common: dict[str, object] = {
        "levers": [Lever("u", lo=-2.0, hi=2.0)],
        "target": Target("x", value=1.0),
        "adjustment": CausalGraph.from_edges([("z", "u"), ("z", "x"), ("u", "x")]),
        "dt": 0.5,
        "x0": [float(log["x"][-1])],
        "horizon": 6,
    }
    panel = Panel.from_frame(log, unit="unit", time="time", seed=0)
    bent = prescribe(panel, known=_Bent(), tolerance=10.0, **common)  # type: ignore[arg-type]
    certificate = bent.certificate
    assert (certificate.tube_rate, certificate.tube_mode) == ("local", "bound")
    assert certificate.tube_state_noise is None
    record = bent.to_json()["certificate"]
    assert (record["tube_mode"], record["tube_state_noise"]) == ("bound", None)
    (line,) = [line for line in bent.report().splitlines() if "error tube" in line]
    assert "bound radii" in line
    assert not [line for line in bent.report().splitlines() if "own 0.95 radius" in line]
    off = prescribe(panel, **common).certificate  # type: ignore[arg-type]
    assert (off.tube_mode, off.tube_state_noise) == (None, None)


_DT = 0.1


def _quickstart_logs(seed: int = 0) -> dict[str, np.ndarray]:
    """The quickstart's logs (``docs/quickstart.py``)."""
    rng = np.random.default_rng(seed)
    names = ("region", "week", "supply", "wait", "incentive", "demand")
    rows: dict[str, list[float]] = {name: [] for name in names}
    for region in range(200):
        supply, wait = rng.normal(0.0, 0.2), rng.normal(0.0, 0.2)
        for week in range(12):
            demand = rng.normal(0.0, 1.0)
            incentive = 0.9 * demand + rng.normal(0.0, 0.5)
            for name, value in zip(
                names, (region, week, supply, wait, incentive, demand), strict=True
            ):
                rows[name].append(value)
            supply, wait = (
                supply
                + _DT * (-0.6 * supply + 0.3 * wait + 0.8 * incentive + 1.5 * demand)
                + rng.normal(0.0, 0.01),
                wait + _DT * (0.25 * wait - 0.4 * incentive) + rng.normal(0.0, 0.01),
            )
    return {name: np.asarray(values) for name, values in rows.items()}


def _quickstart(logs: dict[str, np.ndarray], wait_units: float = 1.0):
    edges = [
        ("demand", "incentive"),
        ("demand", "supply"),
        ("incentive", "supply"),
        ("incentive", "wait"),
        ("wait", "supply"),
    ]
    return prescribe(
        Panel.from_frame(logs, unit="region", time="week", seed=0),
        adjustment=CausalGraph.from_edges(edges),
        levers=[Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05)],
        target=Target("supply", value=1.0),
        constraints=[Constraint("wait", hi=0.5 * wait_units)],
        horizon=15,
        dt=_DT,
        tolerance=0.5,
    )


def test_the_quickstart_trusts_the_weeks_its_plant_allows_and_holds_it_over_them() -> None:
    """Run on from the plan's start under its actions, demand drawn afresh each week as on the
    log, the quickstart's plant stays within the tolerance of 0.5 at 95 % for 3 weeks: the
    certificate trusts those 3, where the norm bound's tube trusted 1, and the plant is inside the
    tube at each of them at least 0.95 of the time, within four Monte Carlo standard errors."""
    decided = _quickstart(_quickstart_logs())
    plan, certificate = decided.plan, decided.certificate
    assert plan is not None
    assert plan.uncertainty_tube is not None
    assert certificate.tube_mode == "exact"
    assert certificate.trustworthy_steps == 3
    tube = np.asarray(plan.uncertainty_tube)
    planned = np.asarray(plan.trajectory)
    rng = np.random.default_rng(12)
    supply = np.full(_DRAWS, planned[0, 0])
    wait = np.full(_DRAWS, planned[0, 1])
    inside, reach = [], []
    for k, action in enumerate(np.asarray(plan.actions)[:, 0]):
        demand = rng.standard_normal(_DRAWS)
        supply, wait = (
            supply
            + _DT * (-0.6 * supply + 0.3 * wait + 0.8 * action + 1.5 * demand)
            + rng.normal(0.0, 0.01, _DRAWS),
            wait + _DT * (0.25 * wait - 0.4 * action) + rng.normal(0.0, 0.01, _DRAWS),
        )
        distance = np.hypot(supply - planned[k + 1, 0], wait - planned[k + 1, 1])
        inside.append(np.mean(distance <= tube[k + 1]))
        reach.append(np.quantile(distance, _COVERAGE))
    allowed = int(np.argmax(np.asarray(reach) > 0.5))
    assert allowed == certificate.trustworthy_steps
    assert np.all(np.asarray(inside[:allowed]) >= _COVERAGE - 4.0 * _SE), inside


def test_each_state_s_own_radius_moves_with_that_state_s_units_alone() -> None:
    """With ``wait`` logged in thousandths, its bound with it, the fit and the plan read the same
    problem (ADR 0059): each state's own radius of the noise moves by that state's units alone,
    ``supply``'s not at all. The tube itself is one Euclidean radius over the states and moves
    otherwise, as the tolerance does (ADR 0091)."""
    logs = _quickstart_logs()
    base = _quickstart(logs).certificate
    scaled_logs = dict(logs, wait=logs["wait"] * 1e3)
    scaled = _quickstart(scaled_logs, wait_units=1e3).certificate
    assert base.tube_state_noise is not None
    assert scaled.tube_state_noise is not None
    before, after = np.asarray(base.tube_state_noise), np.asarray(scaled.tube_state_noise)
    np.testing.assert_allclose(after[1:, 0], before[1:, 0], rtol=1e-6, atol=0.0)
    np.testing.assert_allclose(after[1:, 1], 1e3 * before[1:, 1], rtol=1e-6, atol=0.0)
