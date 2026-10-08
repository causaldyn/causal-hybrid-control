"""Causal identification of a residual's control channel: earn the claim, do not assert it.

Every test here is two-sided. "The orthogonal fit recovers B" passes vacuously if the plant is not
actually confounded, so each recovery test also pins down what the *un*-adjusted fit does on the
same rows.
"""

import dataclasses
import functools
import importlib
import itertools
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg
import scipy.stats

from chc import _units
from chc.control import projected_gradient_control
from chc.cost import QuadraticCost, total_cost
from chc.dynamics import HybridDynamics, LinearDynamics
from chc.dynamics_id import (
    CausalDynamicsFit,
    ConfoundedControlAffineSystem,
    _absorbed,
    _channel_design,
    _clustered_squares,
    _less_below_nothing,
    _logged_relations,
    _ridge_inverse,
    _ridge_weight,
    _robust_spreads,
    _solve_ridge,
    _unmoved_actions,
    _unmoved_directions,
    fit_causal_residual,
    omitted_confounder_bound,
    persistence_check,
    solve_channel_moment,
)
from chc.integrate import rk4_step
from chc.residual import ControlAffineResidual, control_affine_features
from chc.train import fit_residual

# Public as jax.enable_x64 from jax 0.8.0; the floor, 0.4.30, has only jax.experimental.enable_x64,
# which jax 0.11 no longer has.
if hasattr(jax, "enable_x64"):
    enable_x64 = jax.enable_x64
else:
    enable_x64 = importlib.import_module("jax.experimental").enable_x64

DRIFT = jnp.array([[-0.5, 0.1], [0.0, -0.3]])
CHANNEL = jnp.array([[1.0], [0.5]])  # the estimand
CONFOUNDER_TO_RATE = jnp.array([[2.0], [1.0]])
CONFOUNDER_TO_ACTION = jnp.array([[-1.5]])


def _known(t: float | jax.Array, x: jax.Array, u: jax.Array) -> jax.Array:
    """Physics that contributes nothing, so the residual owns the whole rate."""
    return jnp.zeros_like(x)


def _system(**kw) -> ConfoundedControlAffineSystem:
    return ConfoundedControlAffineSystem(
        drift=DRIFT,
        channel=CHANNEL,
        confounder_to_rate=CONFOUNDER_TO_RATE,
        confounder_to_action=CONFOUNDER_TO_ACTION,
        **kw,
    )


def _channel_of(fit: CausalDynamicsFit) -> jax.Array:
    """The constant part of ``B_θ``, the channel at ``x = 0``. At ``degree=1`` the fitted channel
    is affine in the state; the plants here have a constant one, so its slope should be noise."""
    return fit.residual.channel[:, :, 0]


def _error(channel: jax.Array) -> float:
    return float(jnp.linalg.norm(channel - CHANNEL))


def test_prediction_error_training_lands_on_the_observational_channel() -> None:
    """The premise of this module: ``chc.train`` fits the wrong object under confounding.

    Not an assumption to state in a docstring -- gradient-descent MSE training and the
    unadjusted moment must agree on the *same* wrong channel, and both must be far from the truth.
    """
    system = _system(instrument_to_action=jnp.array([[0.8]]))
    data = system.sample(4000, jax.random.key(0), _known)
    init = ControlAffineResidual(drift=jnp.zeros((2, 3)), channel=jnp.zeros((2, 1, 3)))
    trained, _ = fit_residual(
        HybridDynamics(known=_known, residual=init), data, system.dt, steps=800, lr=1e-1
    )
    mse_channel = trained.residual.channel[:, :, 0]
    observational = _channel_of(fit_causal_residual(_known, data, system.dt))

    assert float(jnp.linalg.norm(mse_channel - observational)) < 0.01  # measured gap 0.0018
    assert _error(mse_channel) > 0.9  # and both are wrong by 1.09; true B has norm 1.12


def test_the_unadjusted_fit_is_biased_and_the_orthogonal_fit_is_not() -> None:
    system = _system()
    data = system.sample(4000, jax.random.key(0), _known)

    orthogonal = fit_causal_residual(_known, data, system.dt, adjust_for=("z",))
    unadjusted = fit_causal_residual(_known, data, system.dt)

    assert _error(_channel_of(orthogonal)) < 0.05  # measured 0.008
    assert _error(_channel_of(unadjusted)) > 1.0  # measured 1.35, and sign-flipped: K < 0
    assert bool(jnp.all(_channel_of(unadjusted) < 0.0))
    assert orthogonal.identified
    assert not unadjusted.identified


def test_channel_error_is_second_order_in_nuisance_error() -> None:
    """Orthogonality is present, not merely intended.

    The perturbation directions are *functions of the conditioning variables*, which is what a
    mis-specified nuisance actually looks like; perturbing with independent noise would make the
    first-order terms vanish for the trivial reason. The exponent alone is not the whole claim --
    ``error/eps`` must also collapse across the sweep, since a first-order score holds it constant.
    """
    system = _system(noise_scale=1e-4)
    data = system.sample(40_000, jax.random.key(1), _known)
    x, z, u = data["x"], data["z"], data["u"]
    y = (data["x_next"] - x) / system.dt

    action_nuisance = z @ CONFOUNDER_TO_ACTION.T  # m(x,z) = E[u | x,z], exactly
    state_nuisance = x @ DRIFT.T + action_nuisance @ CHANNEL.T + z @ CONFOUNDER_TO_RATE.T
    covariates = jnp.concatenate([x, z], axis=1)
    k_state, k_action = jax.random.split(jax.random.key(101))
    towards_state = covariates @ jax.random.normal(k_state, (covariates.shape[1], y.shape[1]))
    towards_action = covariates @ jax.random.normal(k_action, (covariates.shape[1], u.shape[1]))
    towards_state /= jnp.std(towards_state)
    towards_action /= jnp.std(towards_action)

    eps = np.array([0.1, 0.05, 0.025, 0.0125])
    errors = np.array(
        [
            _error(
                solve_channel_moment(
                    y - state_nuisance - e * towards_state,
                    u - action_nuisance - e * towards_action,
                    x,
                )[:, :, 0]
            )
            for e in eps
        ]
    )

    slope = float(np.polyfit(np.log(eps), np.log(errors), 1)[0])
    assert 1.75 < slope < 2.25  # measured 2.05; a first-order score would sit at 1
    ratios = errors / eps
    assert ratios[0] / ratios[-1] > 4.0  # an 8x eps range must shrink error/eps, not preserve it


def test_own_sample_residualisation_collapses_the_channel_for_a_saturated_learner() -> None:
    """What the fold machinery is actually for.

    With the default ridge-polynomial nuisances, own-sample partialling-out is unbiased (see
    :func:`~chc.dynamics_id.fit_causal_residual`'s ``folds`` note), so this failure needs a learner
    that can memorise: 1-nearest-neighbour predicts each row from itself, both residuals vanish,
    and the moment has nothing left to identify. Excluding the row itself repairs it.
    """
    system = _system()
    data = system.sample(1500, jax.random.key(0), _known)
    x, u = data["x"], data["u"]
    y = (data["x_next"] - x) / system.dt
    covariates = jnp.concatenate([x, data["z"]], axis=1)
    square_distance = jnp.sum((covariates[:, None, :] - covariates[None, :, :]) ** 2, axis=-1)

    def nearest_neighbour(target: jax.Array, *, own_row: bool) -> jax.Array:
        masked = square_distance if own_row else square_distance + 1e9 * jnp.eye(len(covariates))
        return target[jnp.argmin(masked, axis=1)]

    def channel(*, own_row: bool) -> jax.Array:
        return solve_channel_moment(
            y - nearest_neighbour(y, own_row=own_row),
            u - nearest_neighbour(u, own_row=own_row),
            x,
        )[:, :, 0]

    assert _error(channel(own_row=True)) > 1.0  # collapses to zero, error = ||B||
    assert _error(channel(own_row=False)) < 0.2  # measured 0.137


def test_the_ridge_polynomial_fit_barely_moves_with_the_fold_count() -> None:
    """Frisch-Waugh-Lovell, stated as a regression guard rather than as folklore.

    Residualising ``y`` and ``u`` by the same projection whose span contains both nuisances is
    exactly unbiased in-sample, so the fold count is a variance knob here and nothing more. This
    test exists to catch a change to the fold bookkeeping that quietly moves the point estimate.
    """
    system = _system()
    data = system.sample(4000, jax.random.key(0), _known)
    channels = [
        _channel_of(fit_causal_residual(_known, data, system.dt, adjust_for=("z",), folds=k))
        for k in (1, 2, 4, 8)
    ]
    spread = max(float(jnp.linalg.norm(c - channels[0])) for c in channels[1:])
    assert spread < 0.005  # measured 0.0004 across folds 1/2/4/8
    assert all(_error(c) < 0.05 for c in channels)


def test_control_regret_collapses_for_the_causal_fit() -> None:
    """The payoff test: identification has to change the *action*, not just the coefficient."""
    system = _system(instrument_to_action=jnp.array([[0.8]]))
    data = system.sample(4000, jax.random.key(0), _known)
    truth = HybridDynamics(
        known=_known,
        residual=ControlAffineResidual(
            drift=jnp.concatenate([jnp.zeros((2, 1)), DRIFT], axis=1),
            channel=jnp.concatenate([CHANNEL[:, :, None], jnp.zeros((2, 1, 2))], axis=2),
        ),
    )
    cost = QuadraticCost(
        Q=jnp.eye(2), R=0.1 * jnp.eye(1), Qf=5.0 * jnp.eye(2), x_target=jnp.array([1.0, 0.0])
    )
    x0, us0 = jnp.zeros(2), jnp.zeros((25, 1))

    def realised_cost(model: HybridDynamics) -> float:
        us, _ = projected_gradient_control(model, x0, us0, system.dt, cost, -5.0, 5.0, steps=120)
        return float(total_cost(truth, x0, us, system.dt, cost))

    def planner(**kw) -> HybridDynamics:
        return HybridDynamics(
            known=_known, residual=fit_causal_residual(_known, data, system.dt, **kw).residual
        )

    oracle = realised_cost(truth)
    causal_regret = realised_cost(planner(adjust_for=("z",))) - oracle
    unadjusted_regret = realised_cost(planner()) - oracle

    assert causal_regret < 0.05  # measured 0.014
    assert unadjusted_regret > 1.0  # measured 6.20 against a 6.10 oracle cost
    assert unadjusted_regret > 50.0 * max(causal_regret, 1e-6)


def test_reports_non_identification_when_nothing_in_the_log_identifies_the_channel() -> None:
    system = _system()
    data = system.sample(2000, jax.random.key(0), _known)
    fit = fit_causal_residual(_known, data, system.dt)

    assert fit.identified is False
    assert fit.method == "observational"
    assert fit.channel_error is None  # no standard error on an unidentified quantity
    assert fit.drift_error is None  # nor on a drift conditional on a channel that means nothing
    assert fit.action_residual_variance > 0.0  # diagnostics still populated for comparison


def test_the_fit_does_not_depend_on_the_units_the_caller_logged_in() -> None:
    """A ridge on an unstandardised polynomial basis penalises each monomial by the caller's units.

    Not a hypothetical. On a 20-day log from a real building emulator (``causaldyn-bench`` Track
    D-causal) the zone entered in Celsius at ~21 beside already-standardised weather columns; the
    degree-2 nuisance Gram came out at condition number 1.4e11, past what float32 carries, and the
    fit returned ``nan`` -- while the same rows in float64 fitted fine, which is what identified it
    as conditioning rather than data.

    Offsetting a covariate reproduces both failure modes in float32: reverting the standardisation,
    ``z + 1e4`` returns ``nan`` and ``z + 1e5`` returns a *finite* channel wrong by 1.34 against a
    baseline error of 0.009 -- a silently wrong answer, which no caller can detect.

    This suite cannot assert that directly, and the reason is worth knowing: ``tests/conftest.py``
    turns on ``jax_enable_x64``, so **no test here can fail from float32 conditioning**, which is
    exactly how the defect survived the whole suite. What float64 still shows is the underlying
    cause -- the unstandardised fit drifts by 2e-4 to 4.5e-4 as the offset grows, because the ridge
    is penalising monomials of order ``1e10``. Standardising makes the estimate *exactly* invariant,
    so that is what this asserts, at a tolerance 20x under the drift it is there to catch.
    """
    system = _system()
    data = system.sample(3000, jax.random.key(0), _known)
    baseline = _channel_of(fit_causal_residual(_known, data, system.dt, adjust_for=("z",)))
    assert _error(baseline) < 0.05  # the fit has to be good before invariance means anything

    for name, shifted in (
        ("kelvin-like offset", {**data, "z": data["z"] + 273.15}),
        ("milli-unit rescale", {**data, "z": data["z"] * 1e3}),
        ("offset that returned nan", {**data, "z": data["z"] + 1e4}),
        ("offset that returned a silently wrong channel", {**data, "z": data["z"] + 1e5}),
    ):
        moved = _channel_of(fit_causal_residual(_known, shifted, system.dt, adjust_for=("z",)))
        assert jnp.all(jnp.isfinite(moved)), name
        assert float(jnp.linalg.norm(moved - baseline)) < 1e-5, name


@pytest.mark.parametrize("integrator", ["euler", "rk4"])
def test_the_errors_do_not_move_with_the_zero_of_the_state_scale(integrator: str) -> None:
    """Logs in degrees Celsius and in kelvin are one log. Moving the state's zero moves the
    channel's and the drift's coefficients, and their values at the logged states not at all; the
    errors are those values'. Read off the coefficients' own variances, the value at ``x = 0``
    among them, the channel's error was 0.0091 on these logs and 0.579 with the states moved."""
    system = _system()
    data = system.sample(2000, jax.random.key(0), _known)
    shift = jnp.array([100.0, -50.0])
    moved = {**data, "x": data["x"] + shift, "x_next": data["x_next"] + shift}

    here, there = (
        fit_causal_residual(_known, log, system.dt, adjust_for=("z",), integrator=integrator)
        for log in (data, moved)
    )

    # the ridge weighs the moved log's larger coefficients: 8e-6 apart
    assert there.channel_error == pytest.approx(here.channel_error, rel=1e-4)
    assert there.drift_error == pytest.approx(here.drift_error, rel=1e-4)


def test_the_drift_carries_its_own_scale_and_it_shrinks_like_root_n() -> None:
    """``drift_error`` has to be a standard error, not a decorative number.

    A real one falls as ``1/sqrt(N)``. This pins that, because the drift scale exists so a consumer
    can compare it against ``channel_error`` and see which half of the model its horizon is really
    waiting on -- a comparison that is worthless if either side is arbitrary.
    """
    system = _system()
    errors = []
    for n in (2000, 8000, 32_000):
        data = system.sample(n, jax.random.key(0), _known)
        fit = fit_causal_residual(_known, data, system.dt, adjust_for=("z",))
        assert fit.drift_error is not None
        errors.append(fit.drift_error)

    for coarse, fine in itertools.pairwise(errors):
        ratio = coarse / fine
        assert 1.6 < ratio < 2.5, errors  # 4x the sample should halve it


def test_the_drift_jacobian_reads_the_drift_where_control_channel_reads_the_channel() -> None:
    """The two accessors must index the same parameter block the estimator wrote."""
    drift = jnp.array([[0.7, -0.3, 0.1], [-0.2, 0.4, -0.5]])  # columns: bias, x0, x1
    residual = ControlAffineResidual(drift=drift, channel=jnp.zeros((2, 1, 3)), degree=1)
    x = jnp.array([1.5, -2.0])

    assert jnp.allclose(residual.drift_jacobian(x), drift[:, 1:])  # constant at degree 1
    assert residual.drift_jacobian(x).shape == (2, 2)

    quadratic = ControlAffineResidual(
        drift=jnp.array([[0.0, 0.0, 1.0]]), channel=jnp.zeros((1, 1, 3)), degree=2
    )  # features [1, x, x^2] on a single state, so the drift is x^2 and the Jacobian is 2x
    assert jnp.allclose(quadratic.drift_jacobian(jnp.array([3.0])), jnp.array([[6.0]]))


def test_the_drift_jacobian_alone_does_not_decide_stability_when_the_channel_moves() -> None:
    """A plant that decays everywhere its actuator can reach, reported as runaway by the drift.

    The coefficients are the ones BOPTEST's ``bestest_hydronic`` produced: an actuator that reports
    its action as a setpoint in ``[15, 25] °C`` puts the drift's zero 15 K outside the range the
    plant ever visits, and the drift absorbs that offset times the state-dependent channel. So
    ``drift_jacobian`` reads ``+6.42`` while the vector field the horizon integrates decays at
    ``-1.40`` at the setpoint actually held, and at every setpoint above 17.0 °C.
    """
    residual = ControlAffineResidual(
        drift=jnp.array([[0.0, 6.4197]]),
        channel=jnp.array([[[9.2965, -0.3779]]]),
        degree=1,
    )
    x = jnp.array([21.0])

    assert float(residual.drift_jacobian(x)[0, 0]) > 0.0
    held = float(residual.closed_loop_jacobian(x, jnp.array([20.685]))[0, 0])
    assert abs(held + 1.397) < 1e-3
    assert float(residual.closed_loop_jacobian(x, jnp.array([15.0]))[0, 0]) > 0.0  # low end grows

    constant = ControlAffineResidual(
        drift=jnp.array([[0.0, -0.05]]), channel=jnp.array([[[1.2, 0.0]]]), degree=1
    )  # a channel that does not move with the state: the two questions coincide again
    assert jnp.allclose(
        constant.drift_jacobian(x), constant.closed_loop_jacobian(x, jnp.array([20.685]))
    )


def test_a_fitted_residual_exposes_the_stability_its_horizon_depends_on() -> None:
    """The failure mode Track D-causal hit on a real building, reduced to an assertion.

    The estimator identifies the channel and leaves the drift to least squares, so nothing stops a
    fitted drift from being unstable. That is documented scope, not a bug -- but it has to be
    *visible*, because an MPC plans on the drift too.
    """
    system = _system()
    data = system.sample(4000, jax.random.key(0), _known)
    fit = fit_causal_residual(_known, data, system.dt, adjust_for=("z",))

    jacobian = fit.residual.drift_jacobian(jnp.zeros(2))
    assert jacobian.shape == (2, 2)
    recovered = float(jnp.max(jnp.real(jnp.linalg.eigvals(jacobian))))
    truth = float(jnp.max(jnp.real(jnp.linalg.eigvals(DRIFT))))
    assert abs(recovered - truth) < 0.15 * abs(truth)  # the spectrum is the drift's, not noise


def test_the_two_fits_agree_when_there_is_nothing_to_correct() -> None:
    """Adjusting must not charge a premium on an unconfounded log."""
    system = ConfoundedControlAffineSystem(
        drift=DRIFT,
        channel=CHANNEL,
        confounder_to_rate=jnp.zeros((2, 1)),  # z still drives the action, but not the rate
        confounder_to_action=CONFOUNDER_TO_ACTION,
    )
    data = system.sample(4000, jax.random.key(3), _known)
    adjusted = _channel_of(fit_causal_residual(_known, data, system.dt, adjust_for=("z",)))
    unadjusted = _channel_of(fit_causal_residual(_known, data, system.dt))

    assert float(jnp.linalg.norm(adjusted - unadjusted)) < 0.02  # measured 0.011
    assert _error(adjusted) < 0.05
    assert _error(unadjusted) < 0.05


def test_iv_recovers_the_channel_when_the_confounder_is_latent() -> None:
    """``z`` is never passed to the estimator; only the exogenous action shifter is.

    Asserted across seeds rather than on one, because the shifter explains only ~18% of the action's
    variance: the estimate is consistent but an order of magnitude noisier than adjusting for a
    logged confounder, so a tight single-seed bound would be a coin flip dressed as a gate.
    """
    system = _system(instrument_to_action=jnp.array([[0.8]]))
    fits = [
        fit_causal_residual(
            _known, system.sample(8000, jax.random.key(s), _known), system.dt, instrument="w"
        )
        for s in range(5)
    ]
    errors = sorted(_error(_channel_of(f)) for f in fits)

    assert all(f.method == "iv" for f in fits)
    assert all(f.identified for f in fits)
    assert all(f.channel_error is not None and f.channel_error > 0.0 for f in fits)
    assert errors[-1] < 0.15  # measured worst 0.091 over these seeds
    assert errors[len(errors) // 2] < 0.05  # median comfortably better
    # ...and still nowhere near the 1.09 an unidentified fit lands at on the same DGP
    unadjusted = fit_causal_residual(
        _known, system.sample(8000, jax.random.key(0), _known), system.dt
    )
    assert errors[-1] < 0.2 * _error(_channel_of(unadjusted))


@pytest.mark.parametrize("channel_degree", [None, 0], ids=["affine", "constant"])
def test_the_reported_channel_error_is_calibrated_on_both_identified_paths(
    channel_degree: int | None,
) -> None:
    """The standard error has to track the spread it claims to describe.

    This is the test that catches the defect it was written for: reporting the ordinary
    least-squares SE of a regression on the *projected* action, with sigma^2 taken from that same
    regression's residual, understated the IV path by ~5x -- worse than reporting nothing, because
    ``chc.sensitivity`` consumes exactly this number as a radius. The robust sandwich on the
    structural residual brings both paths into band: over 200 logs it came to 1.08 and 1.05 of the
    channel's error against the truth, where the homoskedastic one ran optimistic on the IV path.
    Both are read where the error is, at the log's states: the channel's value there against the
    truth's.
    """
    system = _system(instrument_to_action=jnp.array([[0.8]]))
    for keywords in ({"adjust_for": ("z",)}, {"instrument": "w"}):
        deviations, reported = [], []
        for seed in range(24):
            data = system.sample(2000, jax.random.key(seed), _known)
            fit = fit_causal_residual(
                _known, data, system.dt, channel_degree=channel_degree, **keywords
            )
            fitted = jax.vmap(fit.residual.control_channel)(data["x"])
            deviations.append(float(jnp.mean((fitted - CHANNEL) ** 2)))
            reported.append(fit.channel_error)
        ratio = float(np.sqrt(np.mean(deviations)) / np.mean(reported))
        assert 0.5 < ratio < 2.0, f"{keywords} SE off by {ratio:.2f}x"


@pytest.mark.parametrize(
    ("degree", "channel_degree", "drift_features", "channel_features"),
    [(2, None, 6, 6), (1, 0, 3, 1), (0, 1, 1, 3)],
    ids=["shared", "constant_channel", "constant_drift"],
)
def test_the_residual_channel_is_the_jacobian_the_safety_layer_reads(
    degree: int, channel_degree: int | None, drift_features: int, channel_features: int
) -> None:
    """``control_channel`` and ``d r / d u`` must be the same object.

    :func:`chc.plan.certify_safety` recovers the channel by differentiating at ``u = 0``; if the
    estimated parameter and that derivative could disagree, identification and certification would
    be talking about different plants.
    """
    residual = ControlAffineResidual(
        drift=jax.random.normal(jax.random.key(1), (2, drift_features)),
        channel=jax.random.normal(jax.random.key(2), (2, 1, channel_features)),
        degree=degree,
        channel_degree=channel_degree,
    )
    x, u = jnp.array([0.3, -0.7]), jnp.array([1.2])
    jacobian = jax.jacobian(lambda action: residual(0.0, x, action))(u)

    assert jnp.allclose(jacobian, residual.control_channel(x))


# ---- D15: which one-step map the fit is consistent with ----


def _rk4_amplification(z: float) -> float:
    """RK4's one-step growth factor for ``y' = z/dt * y`` -- the exponential's Taylor sum to 4."""
    return 1.0 + z + z**2 / 2 + z**3 / 6 + z**4 / 24


def _rk4_generated_log(
    theta: float,
    n: int = 4000,
    dt: float = 1.0,
    seed: int = 0,
    spread: float = 1.0,
    noise_seed: int | None = None,
    noise_growth: float = 0.0,
):
    """One-step pairs from ``x' = -theta x + 0.8 u``, integrated the way the planner integrates.

    ``noise_seed`` redraws only the observation noise, on the rows ``seed`` drew; its scale grows
    as ``exp(noise_growth x)``."""
    kx, kz, ku, kn = jax.random.split(jax.random.PRNGKey(seed), 4)
    x = jax.random.normal(kx, (n, 1))
    z = jax.random.normal(kz, (n, 1))
    u = spread * (0.9 * z + 0.4 * jax.random.normal(ku, (n, 1)))
    truth = LinearDynamics(jnp.array([[-theta]]), jnp.array([[0.8]]))
    x_next = jax.vmap(lambda xi, ui: rk4_step(truth, 0.0, xi, ui, dt))(x, u)
    kn = kn if noise_seed is None else jax.random.PRNGKey(noise_seed)
    noise = 0.01 * jnp.exp(noise_growth * x) * jax.random.normal(kn, (n, 1))
    return {"x": x, "u": u, "z": z, "x_next": x_next + noise}


def _exact_log(theta: float, n: int = 2000, dt: float = 1.0, seed: int = 0):
    """The same plant's log with each step solved exactly, the action held, instead of by RK4."""
    kx, kz, ku, kn = jax.random.split(jax.random.PRNGKey(seed), 4)
    x = jax.random.normal(kx, (n, 1))
    z = jax.random.normal(kz, (n, 1))
    u = 0.9 * z + 0.4 * jax.random.normal(ku, (n, 1))
    factor = float(np.exp(-theta * dt))
    x_next = factor * x + 0.8 * u * (1.0 - factor) / theta
    return {"x": x, "u": u, "z": z, "x_next": x_next + 0.01 * jax.random.normal(kn, (n, 1))}


def test_the_euler_fit_recovers_the_euler_field_and_the_rk4_fit_recovers_the_true_one() -> None:
    """The default reads a forward difference; the plan integrates with RK4. That is a real bias.

    Two-sided in the same way as everything else here. It would be no achievement for ``rk4`` to
    land on ``-theta`` if ``euler`` did too, so the ``euler`` arm is pinned to the *closed-form*
    place it is supposed to land: the RK4 amplification ``1 + z + z^2/2 + z^3/6 + z^4/24`` divided
    back into a rate. The channel is the number that matters -- it is what the optimiser moves
    along -- and it is off by 28% at ``theta*dt = 0.7``.
    """
    dt = 1.0
    base = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    for theta in (0.7, 0.4, 0.25):
        data = _rk4_generated_log(theta, dt=dt)
        euler = fit_causal_residual(base, data, dt, adjust_for=("z",), seed=0)
        rk4 = fit_causal_residual(base, data, dt, adjust_for=("z",), seed=0, integrator="rk4")

        # feature 0 is the bias, feature 1 the state itself
        assert float(np.asarray(euler.residual.drift)[0, 1]) == pytest.approx(
            (_rk4_amplification(-theta * dt) - 1.0) / dt, abs=0.01
        )
        assert float(np.asarray(rk4.residual.drift)[0, 1]) == pytest.approx(-theta, abs=0.01)
        assert float(np.asarray(rk4.residual.channel)[0, 0, 0]) == pytest.approx(0.8, abs=0.01)
        assert euler.integrator == "euler"
        assert rk4.integrator == "rk4"

    # the sharpest case, stated as the bias it removes rather than as a tolerance
    data = _rk4_generated_log(0.7, dt=dt)
    euler = fit_causal_residual(base, data, dt, adjust_for=("z",), seed=0)
    rk4 = fit_causal_residual(base, data, dt, adjust_for=("z",), seed=0, integrator="rk4")
    off = abs(float(np.asarray(euler.residual.channel)[0, 0, 0]) - 0.8)
    assert off > 0.2  # 0.574 against 0.800: the mismatch is a quarter of the control channel
    assert abs(float(np.asarray(rk4.residual.channel)[0, 0, 0]) - 0.8) < 0.1 * off


def test_the_defect_lands_at_the_noise_floor_under_whichever_integrator_was_asked_for() -> None:
    """``integrator_defect`` is measured against the map the fit was made consistent with.

    Both settings therefore reach the same floor -- the 0.01 observation noise divided by ``dt`` --
    on the same log, and that is the point: it says the fixed point converged, not that one
    integrator fits better than the other. A number that only ever went down for ``rk4`` would be
    measuring two different things and calling it an improvement.
    """
    dt = 1.0
    base = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    data = _rk4_generated_log(0.7, dt=dt)
    for integrator in ("euler", "rk4"):
        fit = fit_causal_residual(base, data, dt, adjust_for=("z",), seed=0, integrator=integrator)
        assert fit.integrator_defect is not None
        assert fit.integrator_defect == pytest.approx(0.01, abs=0.003)


def test_the_correction_runs_when_the_gap_it_closes_hides_under_the_noise() -> None:
    """At ``theta*dt = 0.05`` with an action that moves the state little, the Euler gap raises the
    defect's RMS by well under 1%. A 1% bar refused the first correction on eight seeds of eight:
    ``"rk4"`` handed back the Euler fit, digit for digit, and reported the noise floor as its
    defect. The gap is 2.5% of the channel, three standard errors of it on this log.

    Paired on one log, so the noise both fits share cancels: the Euler arm lands on the closed-form
    Euler reading and the ``rk4`` arm sits the whole gap above it.
    """
    dt, theta = 0.1, 0.5
    z = -theta * dt
    base = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    data = _rk4_generated_log(theta, n=20_000, dt=dt, spread=0.3)
    euler = fit_causal_residual(base, data, dt, adjust_for=("z",), seed=0)
    rk4 = fit_causal_residual(base, data, dt, adjust_for=("z",), seed=0, integrator="rk4")

    def slope(fit: CausalDynamicsFit) -> float:
        return float(np.asarray(fit.residual.drift)[0, 1])

    def channel(fit: CausalDynamicsFit) -> float:
        return float(np.asarray(fit.residual.channel)[0, 0, 0])

    assert slope(euler) == pytest.approx((_rk4_amplification(z) - 1.0) / dt, abs=0.003)
    assert slope(rk4) == pytest.approx(-theta, abs=0.003)
    euler_gap = 0.8 * (1.0 - (_rk4_amplification(z) - 1.0) / z)  # 0.0197
    assert channel(rk4) - channel(euler) == pytest.approx(euler_gap, abs=0.003)


def test_the_rk4_channel_error_carries_the_rk4_gain_the_spread_carries() -> None:
    """Replicates of one log that differ only in their noise scatter the ``rk4`` channel 1.64x as
    far as the Euler channel, and ``channel_error`` says 1.70x. That factor is the RK4 map's gain on
    the estimate, which the fixed point's own error carries.

    Read against the Euler fit on the same replicates, because the two share every noise draw:
    sixteen replicates pin the ratio of their spreads to a few percent, where either spread alone
    moves by a quarter (0.75 to 1.15 of its error over three sets of sixteen). Errors that leave the
    gain out sit near the Euler fit's and fail the second assertion by the gain itself; read off
    the corrected target, as up to 0.6.0, they were six times the spread.
    """
    dt, replicates = 1.0, 16
    base = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    spread, reported = {}, {}
    for integrator in ("euler", "rk4"):
        channels, errors = [], []
        for replicate in range(replicates):
            data = _rk4_generated_log(0.7, n=1000, dt=dt, noise_seed=replicate)
            fit = fit_causal_residual(
                base, data, dt, adjust_for=("z",), seed=0, integrator=integrator
            )
            assert fit.channel_error is not None
            # the replicates share their states, where the channel's value is read
            channels.append(np.asarray(jax.vmap(fit.residual.control_channel)(data["x"])).ravel())
            errors.append(fit.channel_error)
        spread[integrator] = float(np.sqrt(np.mean(np.var(np.array(channels), axis=0, ddof=1))))
        reported[integrator] = float(np.mean(errors))

    assert spread["rk4"] / reported["rk4"] == pytest.approx(1.0, abs=0.35)
    gain = spread["rk4"] / spread["euler"]
    assert gain / (reported["rk4"] / reported["euler"]) == pytest.approx(1.0, abs=0.15)


def test_the_rk4_fit_reads_an_exact_log_as_rk4_does_and_refuses_one_rk4_cannot_read() -> None:
    """On a log whose steps were solved exactly rather than by RK4, the ``rk4`` fit is the field
    whose RK4 step reproduces it: the decay ``z*`` at which RK4's growth factor ``R`` equals
    ``exp(-theta dt)``, and the channel ``0.8 z* / (-theta dt)``. Not the plant's own ``-theta``,
    which RK4 steps to a different log.

    ``R`` bottoms out at 0.2704, at ``z = -1.596``, so there is no such field once ``theta dt``
    passes ``-log(0.2704) = 1.308``. Up to 0.6.0 the fit returned one anyway. It iterated until the
    defect stopped falling and kept where it was: ``-1.279`` against this fixed point's ``-1.370``
    at ``theta dt = 1.25``, and at 1.5 a decay of ``-1.57`` with the defect at five times the noise.
    """
    dt, theta = 1.0, 1.25
    base = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    fit = fit_causal_residual(
        base, _exact_log(theta, dt=dt), dt, adjust_for=("z",), seed=0, integrator="rk4"
    )
    roots = np.roots([1 / 24, 1 / 6, 1 / 2, 1, 1 - np.exp(-theta * dt)])
    z_star = max(root.real for root in roots if abs(root.imag) < 1e-9)  # the branch R falls on
    assert float(np.asarray(fit.residual.drift)[0, 1]) == pytest.approx(z_star / dt, abs=0.005)
    assert float(np.asarray(fit.residual.channel)[0, 0, 0]) == pytest.approx(
        0.8 * z_star / (-theta * dt), abs=0.005
    )
    assert fit.integrator_defect == pytest.approx(0.01, abs=0.001)

    with pytest.raises(ValueError, match="rk4 fixed point did not converge"):
        fit_causal_residual(
            base, _exact_log(1.5, dt=dt), dt, adjust_for=("z",), seed=0, integrator="rk4"
        )


# ---- ID1: which transitions the channel moment weighs ----

DECISION_MEAN, DECISION_VARIANCE = -0.5, 0.25  # Q, the law of the states the decision is taken at


def _decision_log(
    n: int, curvature: float, seed: int, noise: float = 1.0, nuisances: str = "polynomial"
) -> dict[str, jax.Array]:
    """A scalar state whose channel ``1 + x/2 + curvature x^2`` the linear class misses unless the
    curvature is zero. The confounder ``z`` leaves the action a variance ``exp(x/2)`` and the rate
    a noise of variance ``exp(-x/2)``, so each weight reads its own channel off the same rows.
    ``"trigonometric"`` nuisances lie outside the fits' polynomial sieve."""
    kx, kz, kv, ke = jax.random.split(jax.random.key(seed), 4)
    x = jax.random.normal(kx, (n, 1))
    z = jax.random.normal(kz, (n, 1))
    if nuisances == "polynomial":
        action, drift = 0.5 * x + 0.1 * x**2, 1.0 + 0.5 * x - 0.2 * x**2
    else:
        action, drift = 0.5 * x + 0.3 * jnp.sin(2.0 * x), jnp.cos(x) + 0.5 * x
    u = action + 0.8 * z + jnp.exp(0.25 * x) * jax.random.normal(kv, (n, 1))
    rate = (1.0 + 0.5 * x + curvature * x**2) * u + drift + z
    rate = rate + noise * jnp.exp(-0.25 * x) * jax.random.normal(ke, (n, 1))
    return {"x": x, "z": z, "u": u, "x_next": x + rate}


def _decision_weight(states: jax.Array) -> jax.Array:
    """``kappa (dQ/dP) / s^2`` at ``kappa = 1``: the log's states are standard normal, and the
    action's leftover variance is ``exp(x/2)``."""
    x = states[:, 0]
    log_ratio = -((x - DECISION_MEAN) ** 2) / (2 * DECISION_VARIANCE) + x**2 / 2
    return jnp.exp(log_ratio - 0.5 * x) / jnp.sqrt(DECISION_VARIANCE)


def _tilted_channel(tilt: float, curvature: float) -> np.ndarray:
    """The linear fit to ``1 + x/2 + curvature x^2`` under ``N(tilt, 1)``, which is what a weight
    ``w`` with ``w s^2`` proportional to ``exp(tilt x)`` turns the log's standard normal into."""
    return np.array([1.0 + curvature * (1.0 - tilt**2), 0.5 + 2.0 * curvature * tilt])


def _decision_regret(channel: np.ndarray, curvature: float) -> float:
    """``E_Q[(b - c0 - c1 x)^2] / 2``: acting on the fitted channel ``c`` at states drawn from Q,
    each paid ``b(x) u - u^2 / 2``. From Q's moments, independently of the fit."""
    c0, c1, c2 = 1.0 - channel[0], 0.5 - channel[1], curvature
    mu, var = DECISION_MEAN, DECISION_VARIANCE
    m1, m2, m3 = mu, mu**2 + var, mu**3 + 3 * mu * var
    m4 = mu**4 + 6 * mu**2 * var + 3 * var**2
    square = c0**2 + 2 * c0 * c1 * m1 + (c1**2 + 2 * c0 * c2) * m2 + 2 * c1 * c2 * m3 + c2**2 * m4
    return 0.5 * square


def _decision_variance(curvature: float) -> tuple[np.ndarray, np.ndarray]:
    """``G = E_Q[phi phi']`` and ``n Cov`` of the decision-weighted line, ``G^-1 E[psi psi'] G^-1``,
    by Gauss-Hermite under Q. The score is ``w phi u_res (r u_res + eps)``, with ``r`` the line's
    miss under Q, so ``E[psi psi'] = E_Q[(q/p) phi phi' (3 r^2 + var(eps)/s^2)]``."""
    nodes, masses = np.polynomial.hermite.hermgauss(80)
    x = DECISION_MEAN + np.sqrt(2.0 * DECISION_VARIANCE) * nodes
    under_q = masses / np.sqrt(np.pi)
    density_ratio = np.exp(-((x - DECISION_MEAN) ** 2) / (2 * DECISION_VARIANCE) + x**2 / 2)
    density_ratio /= np.sqrt(DECISION_VARIANCE)
    miss = curvature * ((x - DECISION_MEAN) ** 2 - DECISION_VARIANCE)
    features = np.stack([np.ones_like(x), x])
    gram = (under_q * features) @ features.T
    meat = (under_q * density_ratio * (3.0 * miss**2 + np.exp(-x)) * features) @ features.T
    return gram, np.linalg.solve(gram, np.linalg.solve(gram, meat).T)


@functools.cache
def _replicated_decision_fits() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """A hundred fresh logs of 4000 rows from a class that misses the truth, fitted under the
    decision weight, and the first forty under ones as well: the lines, the errors the fits
    reported, and each log's mean square of the line's features ``(1, x)``, the states the errors
    are read at. A regret is a quadratic form in the line's error, so its mean over forty logs
    moves by a fifth; the hundred are for that."""
    weights = {"ones": lambda states: jnp.ones(states.shape[0]), "decision": _decision_weight}
    lines: dict[str, list[np.ndarray]] = {name: [] for name in weights}
    errors: dict[str, list[float]] = {name: [] for name in weights}
    moments: dict[str, list[np.ndarray]] = {name: [] for name in weights}
    for seed in range(100):
        data = _decision_log(4000, 0.4, seed=100 + seed)
        features = np.concatenate([np.ones_like(data["x"]), np.asarray(data["x"])], axis=1)
        for name, weight in weights.items():
            if name == "ones" and seed >= 40:
                continue
            fit = _decision_fit(data, weight)
            assert fit.channel_error is not None
            lines[name].append(_line(fit))
            errors[name].append(fit.channel_error)
            moments[name].append(features.T @ features / features.shape[0])
    return {
        name: (np.array(lines[name]), np.array(errors[name]), np.array(moments[name]))
        for name in weights
    }


def _decision_fit(data: dict[str, jax.Array], weights) -> CausalDynamicsFit:
    return fit_causal_residual(
        _known, data, 1.0, adjust_for=("z",), nuisance_degree=4, weights=weights
    )


def _line(fit: CausalDynamicsFit) -> np.ndarray:
    """The fitted channel's intercept and slope in the scalar state."""
    return np.asarray(fit.residual.channel)[0, 0]


def test_a_weight_on_the_state_keeps_the_moment_orthogonal_and_one_on_the_action_does_not() -> None:
    """Why the weight is a function of the state alone. On a state weight the channel's error is
    second order in the nuisances' error; on a weight that reads the action it is first order,
    because ``E[w u_res | x, z]`` is no longer zero."""
    data = _decision_log(40_000, 0.0, seed=1, noise=1e-4)
    x, z, u = data["x"], data["z"], data["u"]
    y = data["x_next"] - x
    action_nuisance = 0.5 * x + 0.1 * x**2 + 0.8 * z  # E[u | x, z], exactly
    state_nuisance = (1.0 + 0.5 * x) * action_nuisance + 1.0 + 0.5 * x - 0.2 * x**2 + z
    towards_state = x**2 - 1.0 + 0.7 * z
    towards_action = x - 0.5 * x * z
    towards_state /= jnp.std(towards_state)
    towards_action /= jnp.std(towards_action)
    eps = np.array([0.1, 0.05, 0.025, 0.0125])

    def slope(weights: jax.Array) -> float:
        errors = [
            np.linalg.norm(
                np.asarray(
                    solve_channel_moment(
                        y - state_nuisance - e * towards_state,
                        u - action_nuisance - e * towards_action,
                        x,
                        weights=weights,
                    )
                )[0, 0]
                - np.array([1.0, 0.5])
            )
            for e in eps
        ]
        return float(np.polyfit(np.log(eps), np.log(errors), 1)[0])

    assert 1.75 < slope(_decision_weight(x)) < 2.25
    assert 0.75 < slope(jnp.exp(0.5 * u[:, 0])) < 1.25


def test_every_weight_reads_the_channel_when_the_class_contains_it() -> None:
    """Only the variance moves. The inverse of the rate's noise, ``exp(x/2)``, is the efficient
    weight, whose error is ``0.873`` of the unweighted fit's in the limit; here it reports 0.93,
    and over 60 such logs the channel scattered 0.87 as far. Over 200 logs of 4000 rows it
    scattered 1.15 as far: it loads the right tail, where the degree-4 nuisances extrapolate."""
    data = _decision_log(64_000, 0.0, seed=2)
    ones = _decision_fit(data, lambda states: jnp.ones(states.shape[0]))
    inverse_noise = _decision_fit(data, lambda states: jnp.exp(0.5 * states[:, 0]))
    for fit in (ones, inverse_noise, _decision_fit(data, _decision_weight)):
        assert np.allclose(_line(fit), [1.0, 0.5], atol=0.03)

    limit = np.sqrt((2.0 + 1.0) * np.exp(-0.5) / ((1.8125 + 1.25) * np.exp(-0.25)))
    assert inverse_noise.channel_error is not None
    assert ones.channel_error is not None
    assert limit < inverse_noise.channel_error / ones.channel_error < 0.97


def test_each_weight_reads_its_own_projection_when_the_class_misses_the_channel() -> None:
    """Under a class that misses the truth the weight chooses the estimand: the linear fit to the
    true channel under ``w s^2`` times the log's law of states. Checked on three tilts and on the
    decision weight, whose target is the fit under Q, ``(1.0, 0.1)`` here -- four channels at
    least 0.3 apart, read off the same rows."""
    curvature = 0.4
    data = _decision_log(64_000, curvature, seed=3)
    targets = {
        -0.5: _tilted_channel(-0.5, curvature),
        0.5: _tilted_channel(0.5, curvature),  # the unweighted fit, (1.3, 0.9)
        1.0: _tilted_channel(1.0, curvature),  # the inverse noise, (1.0, 1.3)
    }
    for tilt, target in targets.items():
        fit = _decision_fit(data, lambda states, t=tilt: jnp.exp((t - 0.5) * states[:, 0]))
        assert np.allclose(_line(fit), target, atol=0.05), (tilt, _line(fit), target)
    unweighted = _decision_fit(data, None)
    assert np.allclose(_line(unweighted), targets[0.5], atol=0.05)

    decision = _decision_fit(data, _decision_weight)
    q_fit = np.array(
        [
            1.0 + curvature * (DECISION_VARIANCE - DECISION_MEAN**2),
            0.5 + 2.0 * curvature * DECISION_MEAN,
        ]
    )
    assert np.allclose(_line(decision), q_fit, atol=0.05)
    channels = [*targets.values(), q_fit]
    assert min(np.linalg.norm(a - b) for a, b in itertools.combinations(channels, 2)) > 0.3


@pytest.mark.parametrize("nuisances", ["polynomial", "trigonometric"])
def test_the_decision_weight_takes_the_regret_to_the_floor_of_its_class(nuisances: str) -> None:
    """The payoff: acting on the decision-weighted fit costs what the best line costs, the
    curvature's share ``curvature^2 var_Q^2 = 0.01`` that no line removes, and the unweighted fit
    costs nine times that. The regret is computed from Q's moments, not from the fit. Nuisances
    outside the fits' polynomial sieve leave the decision weight at the floor, since its moment
    stays orthogonal to their error and it reads the states where the sieve is good; the
    unweighted fit, which reads the tails, moved to 7.9 times it."""
    curvature = 0.4
    data = _decision_log(64_000, curvature, seed=4, nuisances=nuisances)
    floor = curvature**2 * DECISION_VARIANCE**2
    decision = _decision_regret(_line(_decision_fit(data, _decision_weight)), curvature)
    unweighted = _decision_regret(_line(_decision_fit(data, None)), curvature)

    assert decision == pytest.approx(floor, rel=0.05)
    assert unweighted > 6.0 * decision  # 9.9 and 7.9 times


def test_a_weighted_fit_reports_an_error_its_spread_matches() -> None:
    """The weighted fit's standard error is robust, since a weight is chosen because the rows
    differ. The error is the line's at the log's states, so the spread is read there too: each
    fit's distance from the mean line, at its own log's states. Over fresh logs of a class that
    misses the truth, where the score's variance moves with the state, the spread came to 1.00 of
    the robust error weighted by ones, and to 1.02 of the decision weight's. Up to 0.12 both were
    read on the line's coefficients, where the ratios were 1.02 and 0.96, and the homoskedastic
    error the unweighted fit reported up to 0.7.0 came to 0.40."""
    for name, (lines, errors, moments) in _replicated_decision_fits().items():
        apart = lines - lines.mean(axis=0)
        at_states = np.einsum("ra,rab,rb->r", apart, moments, apart)
        spread = float(np.sqrt(at_states.sum() / (len(lines) - 1)))
        assert spread / float(np.mean(errors)) == pytest.approx(1.0, abs=0.25), name


def test_the_unweighted_fit_reports_the_robust_error_a_weight_of_ones_does() -> None:
    """Every fit's channel error is the robust one (D28). On this class the homoskedastic error the
    unweighted fit reported up to 0.7.0 came to 0.40 of the channel's spread over 800 logs of 4000
    rows, and the robust one, which a weight of ones always reported, to 0.96."""
    data = _decision_log(4000, 0.4, seed=100)
    plain = fit_causal_residual(_known, data, 1.0, adjust_for=("z",), nuisance_degree=4)
    ones = _decision_fit(data, lambda states: jnp.ones(states.shape[0]))
    assert plain.channel_error is not None
    assert plain.channel_error == ones.channel_error


def test_the_decision_weights_regret_is_what_its_variance_predicts() -> None:
    """Above the floor, the decision-weighted fit's regret is ``tr(G V) / 2n``, with ``V`` the
    decision weight's variance worked out by quadrature from the log's law, not read off a fit:
    ``2.98 / n`` here, against the lab's ``2.91 / n`` on the class that contains the truth. The
    bias term of the corrected expansion vanishes for the decision weight, which is what makes it
    the one weight whose regret this predicts. Over 400 logs the ratio was 1.14 +- 0.06: at 4000
    rows the line scatters 5% wider than its limit."""
    curvature, rows = 0.4, 4000
    lines = _replicated_decision_fits()["decision"][0]
    floor = curvature**2 * DECISION_VARIANCE**2
    excess = np.mean([_decision_regret(line, curvature) for line in lines]) - floor
    gram, variance = _decision_variance(curvature)
    predicted = 0.5 * float(np.trace(gram @ variance)) / rows
    assert 0.75 < excess / predicted < 1.55


@pytest.mark.parametrize("tilt", [-1.0, 1.0], ids=["quiet_rows", "noisy_rows"])
def test_the_weighted_error_carries_each_rows_noise_and_the_nuisances_error_at_the_rows_it_loads(
    tilt: float,
) -> None:
    """On one log whose noise grows as ``exp(x)``, redraws of the noise alone scatter the channel by
    what the fit's own linear map says, whichever end of the log the weight loads. The spread is
    the channel's at the log's states, where the error is read.

    Weighed by ``exp(-x)``, towards the quiet rows, the spread came to 0.98 and 1.00 of the error
    under the two integrators over sixteen redraws; weighed by ``exp(x)``, towards the noisy rows,
    to 1.05 and 1.05. Up to 0.12 both were read on the channel's coefficients, where the ratios
    were 1.04 and 1.02, then 0.93 and 0.85. There a sandwich on the moment alone, which leaves out
    what the cross-fitted nuisances pass on from the rows the weight loads, came to 0.67 on the
    quiet rows; an error that pools the noise over the rows came to 0.16 under ``rk4`` on the noisy
    rows, and agrees with the robust one on average under ``exp(-x)``, so only the noisy rows can
    tell them apart."""
    dt = 1.0
    base = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    for integrator in ("euler", "rk4"):
        values, errors = [], []
        for replicate in range(16):
            data = _rk4_generated_log(0.7, n=1000, dt=dt, noise_seed=replicate, noise_growth=1.0)
            fit = fit_causal_residual(
                base,
                data,
                dt,
                adjust_for=("z",),
                integrator=integrator,
                weights=lambda states: jnp.exp(tilt * states[:, 0]),
            )
            assert fit.channel_error is not None
            values.append(np.asarray(jax.vmap(fit.residual.control_channel)(data["x"])).ravel())
            errors.append(fit.channel_error)
        spread = float(np.sqrt(np.mean(np.var(np.array(values), axis=0, ddof=1))))
        assert spread / float(np.mean(errors)) == pytest.approx(1.0, abs=0.3), integrator


def test_the_weight_is_read_off_the_states_alone_and_its_scale_does_not_matter() -> None:
    """Ones reproduce the unweighted channel exactly; a weight a billion times smaller gives the
    same fit, since the weights are scaled to mean 1 before the ridge sees them; the moment the
    weighted fit solves is the weighted one."""
    data = _decision_log(4000, 0.4, seed=5)
    seen: list[jax.Array] = []

    def ones(states: jax.Array) -> jax.Array:
        seen.append(states)
        return jnp.ones(states.shape[0])

    unweighted = _decision_fit(data, None)
    even = _decision_fit(data, ones)
    assert len(seen) == 1
    assert np.array_equal(np.asarray(seen[0]), np.asarray(data["x"]))
    assert np.allclose(_line(even), _line(unweighted), rtol=0.0, atol=1e-12)
    assert even.weighted
    assert not unweighted.weighted
    assert even.effective_sample_size == pytest.approx(4000.0)

    decision = _decision_fit(data, _decision_weight)
    tiny = _decision_fit(data, lambda states: 1e-9 * _decision_weight(states))
    assert np.allclose(_line(tiny), _line(decision), rtol=1e-9, atol=0.0)
    assert tiny.channel_error == pytest.approx(decision.channel_error, rel=1e-9)
    assert decision.moment_norm < 1e-8
    assert decision.effective_sample_size is not None
    assert 0.3 * 4000 < decision.effective_sample_size < 0.8 * 4000


def test_a_zero_weight_drops_its_rows_from_the_moment() -> None:
    """Weights of zero and one solve the moment on the rows weighted one, and count them."""
    data = _decision_log(4000, 0.4, seed=6)
    x = data["x"]
    rng = np.random.default_rng(0)
    y_res, u_res = jnp.asarray(rng.normal(size=(4000, 1))), jnp.asarray(rng.normal(size=(4000, 1)))
    kept = np.asarray(x[:, 0] > 0.0)
    weighted = solve_channel_moment(y_res, u_res, x, weights=jnp.asarray(kept, dtype=x.dtype))
    subset = solve_channel_moment(y_res[kept], u_res[kept], x[kept])
    assert np.allclose(np.asarray(weighted), np.asarray(subset), rtol=1e-10, atol=0.0)

    fit = _decision_fit(data, lambda states: (states[:, 0] > 0.0).astype(states.dtype))
    assert fit.effective_sample_size == pytest.approx(float(kept.sum()))


@pytest.mark.parametrize(
    ("weights", "error", "message"),
    [
        (lambda states: jnp.ones((states.shape[0], 1)), ValueError, "one value per transition"),
        (lambda states: -jnp.ones(states.shape[0]), ValueError, "finite and non-negative"),
        (lambda states: jnp.full(states.shape[0], jnp.nan), ValueError, "finite and non-negative"),
        (lambda states: jnp.zeros(states.shape[0]), ValueError, "zero at every state"),
        ("efficient", TypeError, "a function of the state, or None"),
    ],
)
def test_a_weight_is_refused_unless_it_is_one_finite_non_negative_value_per_state(
    weights, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        _decision_fit(_decision_log(200, 0.0, seed=7), weights)


def test_the_rk4_fixed_point_takes_the_weights_too() -> None:
    """Ones give the unweighted ``rk4`` fit to the digit, and a weight that the class makes
    harmless leaves the channel where the unweighted one finds it, at the noise floor."""
    dt = 1.0
    base = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    data = _rk4_generated_log(0.7, dt=dt)

    def fit(weights) -> CausalDynamicsFit:
        return fit_causal_residual(
            base, data, dt, adjust_for=("z",), integrator="rk4", weights=weights
        )

    unweighted, even = fit(None), fit(lambda states: jnp.ones(states.shape[0]))
    assert np.allclose(np.asarray(even.residual.channel), np.asarray(unweighted.residual.channel))
    assert np.allclose(np.asarray(even.residual.drift), np.asarray(unweighted.residual.drift))
    weighted = fit(lambda states: jnp.exp(-states[:, 0]))
    assert float(np.asarray(weighted.residual.channel)[0, 0, 0]) == pytest.approx(0.8, abs=0.01)
    assert weighted.integrator_defect == pytest.approx(0.01, abs=0.003)


# ---- D29: a constant channel beside an affine drift ----


def test_a_constant_channel_is_n_m_coefficients_beside_the_drifts_own_basis() -> None:
    """``channel_degree=0`` estimates the channel §18/§19 cover and leaves the drift affine.

    Two-sided like the rest: adjusted, it lands on the truth; unadjusted, it is as wrong as the
    affine channel's constant term, since the class does not identify anything the moment does
    not.
    """
    system = _system()
    data = system.sample(4000, jax.random.key(0), _known)
    constant = fit_causal_residual(_known, data, system.dt, adjust_for=("z",), channel_degree=0)
    unadjusted = fit_causal_residual(_known, data, system.dt, channel_degree=0)

    residual = constant.residual
    assert residual.channel.shape == (2, 1, 1)  # n m, where the affine channel has n m (n + 1)
    assert residual.drift.shape == (2, 3)  # the drift keeps its bias and its two slopes
    assert _error(residual.channel[:, :, 0]) < 0.05  # measured 0.008
    assert _error(unadjusted.residual.channel[:, :, 0]) > 1.0  # measured 1.35

    x, elsewhere, u = jnp.array([0.3, -0.7]), jnp.array([-2.0, 1.5]), jnp.array([0.4])
    assert jnp.array_equal(residual.control_channel(x), residual.control_channel(elsewhere))
    assert jnp.allclose(residual.closed_loop_jacobian(x, u), residual.drift_jacobian(x))


def test_the_rk4_fixed_point_reads_a_constant_channel() -> None:
    """The ``rk4`` fixed point carries the channel's own basis, not the drift's.

    Pinned the way the affine fit is: Euler to the RK4 amplification's reading of the channel,
    ``rk4`` to the true one, both at the noise floor of what they were made consistent with.
    """
    dt = 1.0
    base = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    data = _rk4_generated_log(0.7, dt=dt)
    euler, rk4 = (
        fit_causal_residual(
            base, data, dt, adjust_for=("z",), integrator=integrator, channel_degree=0
        )
        for integrator in ("euler", "rk4")
    )

    assert rk4.residual.channel.shape == (1, 1, 1)
    assert float(np.asarray(rk4.residual.channel)[0, 0, 0]) == pytest.approx(0.8, abs=0.01)
    assert float(np.asarray(rk4.residual.drift)[0, 1]) == pytest.approx(-0.7, abs=0.01)
    assert float(np.asarray(euler.residual.channel)[0, 0, 0]) == pytest.approx(0.574, abs=0.01)
    for fit in (euler, rk4):
        assert fit.integrator_defect == pytest.approx(0.01, abs=0.003)


def test_a_negative_channel_degree_is_refused() -> None:
    system = _system()
    data = system.sample(200, jax.random.key(0), _known)
    with pytest.raises(ValueError, match="channel_degree must be a non-negative integer"):
        fit_causal_residual(_known, data, system.dt, adjust_for=("z",), channel_degree=-1)


def _policy_log(
    n: int, actions: Callable[[np.random.Generator, np.ndarray, np.ndarray], np.ndarray]
) -> dict[str, jax.Array]:
    """A two-state log under a logging policy of the caller's, the confounder ``z`` adjusted for:
    ``actions(rng, x, z)`` returns ``(n, m)``, and each action's channel is the matching column of
    ``[[0.8, 0.1], [-0.4, 0.3]]``."""
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, (n, 2))
    z = rng.normal(0.0, 1.0, n)
    u = actions(rng, x, z)
    channel = np.array([[0.8, 0.1], [-0.4, 0.3]])[:, : u.shape[1]]
    rate = x @ np.array([[-0.5, 0.2], [0.0, -0.3]]).T + u @ channel.T + np.outer(z, [1.5, 0.0])
    return {
        "x": jnp.asarray(x),
        "u": jnp.asarray(u),
        "x_next": jnp.asarray(x + 0.1 * rate + rng.normal(0.0, 0.01, (n, 2))),
        "z": jnp.asarray(z[:, None]),
    }


def _dithered(rng: np.random.Generator, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    return (0.9 * z - 0.3 * x[:, 0] + rng.normal(0.0, 0.5, z.size))[:, None]


def _determined(
    scale: float,
) -> Callable[[np.random.Generator, np.ndarray, np.ndarray], np.ndarray]:
    return lambda rng, x, z: scale * (0.9 * z - 0.3 * x[:, 0])[:, None]


@pytest.mark.parametrize(
    ("actions", "rows", "channel_degree", "unmoved"),
    [
        (_dithered, 4000, 0, 0),
        (_dithered, 4000, 1, 0),
        (lambda rng, x, z: _dithered(rng, x, z) * 1e-6 + _determined(1.0)(rng, x, z), 4000, 0, 0),
        (_determined(1.0), 4000, 0, 2),
        (_determined(1.0), 4000, 1, 6),
        (_determined(1.0), 60, 0, 2),
        (_determined(1e-4), 4000, 0, 2),
        (lambda rng, x, z: _dithered(rng, x, z) * np.array([1.0, 2.0]), 4000, 0, 2),
        (lambda rng, x, z: _dithered(rng, x, z) * np.array([1.0, 2.0]), 4000, 1, 6),
        (lambda rng, x, z: _dithered(rng, x, z) * np.array([1.0, 0.0]), 4000, 0, 2),
    ],
    ids=[
        "dithered",
        "dithered, affine channel",
        "a dither of one part in a million",
        "determined by the covariates",
        "determined, affine channel",
        "determined, on 60 rows",
        "determined, at a ten-thousandth of the scale",
        "the second action twice the first",
        "twice the first, affine channel",
        "the second action never used",
    ],
)
def test_the_fit_names_the_directions_its_log_never_moves(
    actions: Callable[[np.random.Generator, np.ndarray, np.ndarray], np.ndarray],
    rows: int,
    channel_degree: int,
    unmoved: int,
) -> None:
    """One direction per state for each combination of the channel's coefficients the action
    residuals never reach, however small the log or the actions' units. A dither, however faint,
    moves its direction: the fit then reads it with an error to match."""
    fit = fit_causal_residual(
        _known,
        _policy_log(rows, actions),
        0.1,
        adjust_for=("z",),
        nuisance_degree=2,
        channel_degree=channel_degree,
    )
    assert fit.unmoved is not None
    assert fit.unmoved.shape[1] == unmoved


@pytest.mark.parametrize(
    ("rows", "actions", "nuisance_degree", "unmoved"),
    [(3, 3, 2, 9), (2, 1, 0, 1)],
    ids=["a nuisance that spans the rows", "a log of two transitions"],
)
def test_a_log_shorter_than_the_channel_names_every_direction_it_cannot_reach(
    rows: int, actions: int, nuisance_degree: int, unmoved: int
) -> None:
    """Fewer transitions than channel coefficients a state, two states and an affine channel: three
    rows under a nuisance of degree 2, which spans them and leaves nothing of the three actions, so
    all nine directions are unmoved; two rows under the mean alone, which move two directions of
    three and leave one. A thin SVD read only as many directions as rows, and returned three and
    none."""
    states = jax.random.normal(jax.random.key(0), (rows, 2))
    taken = jax.random.normal(jax.random.key(1), (rows, actions))
    covariates = jnp.concatenate([states, jnp.ones((rows, 1))], axis=1)
    null = _unmoved_directions(taken, states, covariates, nuisance_degree, 1)
    assert null.shape == (3 * actions, unmoved)
    np.testing.assert_allclose(np.linalg.norm(null, axis=0), 1.0, rtol=1e-12)


def _both(
    first: Callable[[np.random.Generator, np.ndarray, np.ndarray], np.ndarray],
    second: Callable[[np.random.Generator, np.ndarray, np.ndarray], np.ndarray],
) -> Callable[[np.random.Generator, np.ndarray, np.ndarray], np.ndarray]:
    return lambda rng, x, z: np.concatenate([first(rng, x, z), second(rng, x, z)], axis=1)


@pytest.mark.parametrize(
    ("actions", "channel_degree", "unmoved"),
    [
        (_both(_dithered, _dithered), 0, ()),
        (_both(_determined(1.0), _dithered), 0, (0,)),
        (_both(_dithered, _determined(1.0)), 1, (1,)),
        (_both(_determined(1.0), _determined(2.0)), 1, (0, 1)),
        (lambda rng, x, z: _dithered(rng, x, z) * np.array([1.0, 0.0]), 0, (1,)),
        (lambda rng, x, z: _dithered(rng, x, z) * np.array([1.0, 2.0]), 1, ()),
    ],
    ids=[
        "both dithered",
        "the first determined",
        "the second determined, affine channel",
        "both determined",
        "the second never used",
        "the second twice the first",
    ],
)
def test_an_action_is_unmoved_where_every_direction_of_its_own_channel_is(
    actions: Callable[[np.random.Generator, np.ndarray, np.ndarray], np.ndarray],
    channel_degree: int,
    unmoved: tuple[int, ...],
) -> None:
    """Two states and two actions: an action the covariates determine, or one never used, leaves
    its channel unmoved on both states. Two actions that move together leave unmoved a direction
    that both share and neither owns, so neither is unmoved."""
    fit = fit_causal_residual(
        _known,
        _policy_log(4000, actions),
        0.1,
        adjust_for=("z",),
        nuisance_degree=2,
        channel_degree=channel_degree,
    )
    assert _unmoved_actions(fit) == unmoved


@pytest.mark.parametrize(
    ("actions", "column", "units", "unmoved"),
    [
        (_both(_determined(1.0), _dithered), "x", 1e-12, (0,)),
        (_both(_determined(1.0), _dithered), "x", 1e12, (0,)),
        (_both(_determined(1.0), _dithered), "u", 1e9, (0,)),
        (lambda rng, x, z: _dithered(rng, x, z) * np.array([1.0, 2.0]), "u", 1e-9, ()),
        (lambda rng, x, z: _dithered(rng, x, z) * np.array([1.0, 2.0]), "u", 1e9, ()),
    ],
    ids=[
        "the first determined, the state at 1e-12",
        "the first determined, the state at 1e12",
        "the first determined, logged at 1e9",
        "the second twice the first, the first at 1e-9",
        "the second twice the first, the first at 1e9",
    ],
)
def test_an_action_is_unmoved_in_any_units_of_the_state_and_of_the_actions(
    actions: Callable[[np.random.Generator, np.ndarray, np.ndarray], np.ndarray],
    column: str,
    units: float,
    unmoved: tuple[int, ...],
) -> None:
    """The state, or the first action, logged in ``units`` times its own, with an affine channel.
    The fit's directions are unit columns in raw coefficient units, where the state's features part
    from the constant's, and one action's columns from the other's, by those units. Read there, the
    first action's channel lay 3.7e-5 off their span with the state at 1e-12 of its units, 5.2e-5
    at 1e12, and 1.6e-7 with the action at 1e9, each past the square root of the precision. Where
    the second is twice the first, the direction the two share lies 2e-9 of the way along the
    first at 1e9, and read as the second's own; at 1e-9, as the first's."""
    log = _policy_log(4000, actions)
    if column == "x":
        log = dict(log, x=log["x"] * units, x_next=log["x_next"] * units)
    else:
        log = dict(log, u=log["u"] * jnp.array([units, 1.0]))
    fit = fit_causal_residual(
        _known, log, 0.1, adjust_for=("z",), nuisance_degree=2, channel_degree=1
    )
    assert _unmoved_actions(fit) == unmoved


@pytest.mark.parametrize("units", [1e-12, 1.0, 1e12])
def test_the_unmoved_directions_are_orthogonal_in_the_units_the_fit_read_them(
    units: float,
) -> None:
    """The covariates determine the first action, so its channel on the constant and on each of the
    two states is unmoved: three directions. Each coefficient times the size the fit keeps beside
    them, they are the orthogonal directions the fit read, in any units of the state. In raw
    coefficient units the state's units part its features from the constant's: at 1e-12 and 1e12
    of them the three directions are independent only to 3.5e-12 and 1.9e-12, their least singular
    value."""
    log = _policy_log(4000, _both(_determined(1.0), _dithered))
    log = dict(log, x=log["x"] * units, x_next=log["x_next"] * units)
    fit = fit_causal_residual(
        _known, log, 0.1, adjust_for=("z",), nuisance_degree=2, channel_degree=1
    )
    states, actions, features = fit.residual.channel.shape
    assert fit.unmoved is not None
    assert fit._unmoved_size is not None
    directions = fit.unmoved[: actions * features, : fit.unmoved.shape[1] // states]
    scaled = np.asarray(directions * fit._unmoved_size[:, None])
    assert scaled.shape[1] == 3
    unit = scaled / np.linalg.norm(scaled, axis=0)
    np.testing.assert_allclose(unit.T @ unit, np.eye(3), rtol=0.0, atol=1e-12)


def test_a_policy_the_covariates_determine_reads_as_a_confident_wrong_channel() -> None:
    """Why the field exists: with nothing left of the action once the covariates are taken out, the
    moment has no data, and the log's rates set the channel by least squares beside the drift, at
    2.47 where the truth is 0.8: the confounder's push, read as the action's. Its error says 0.0018.
    Only ``unmoved`` tells the caller that every direction of the channel is unread."""
    fit = fit_causal_residual(
        _known,
        _policy_log(4000, _determined(1.0)),
        0.1,
        adjust_for=("z",),
        nuisance_degree=2,
        channel_degree=0,
    )
    assert fit.channel_error is not None
    assert fit.channel_error < 0.01
    assert abs(float(fit.residual.channel[0, 0, 0]) - 0.8) > 0.5
    assert fit.unmoved is not None
    assert fit.unmoved.shape[1] == fit.residual.channel.size


RULED = {"adjust_for": ("z",), "nuisance_degree": 2, "channel_degree": 1}


def _ruled(scale: float) -> Callable[[np.random.Generator, np.ndarray, np.ndarray], np.ndarray]:
    return lambda rng, x, z: scale * (-0.3 * x[:, 0])[:, None]


def test_where_the_moment_has_no_data_the_fit_keeps_to_the_log_in_any_units() -> None:
    """A lever the first state sets, ``-0.3 x0``, leaves its whole affine channel unmoved. Its push
    along the constant feature, ``-0.3 x0``, the drift takes up, and the fit holds the channel at
    zero there; along ``x0`` and ``x1`` it cannot, so the log's rates rule out all but one value,
    and the fit takes it. The moment's ridge set all three: a slope of 0.2 in ``x0`` added to the
    log moved the fit by 2e-11, and with the lever logged in millionths the fit's rates missed the
    log's by 1.6e6, against 1.06 (ADR 0054)."""
    fits = {}
    for scale in (1.0, 1e6):
        log = _policy_log(4000, _ruled(scale))
        slope = 0.1 * 0.2 * log["x"][:, :1] * log["u"] / scale * jnp.array([1.0, 0.0])
        sloped = dict(log, x_next=log["x_next"] + slope)
        fits[scale] = [fit_causal_residual(_known, data, 0.1, **RULED) for data in (log, sloped)]
    for scale, (fit, other) in fits.items():
        np.testing.assert_allclose(np.asarray(fit.residual.channel)[:, 0, 0], 0.0, atol=1e-12)
        change = (np.asarray(other.residual.channel) - np.asarray(fit.residual.channel)) * scale
        np.testing.assert_allclose(
            change, [[[0.0, 0.2, 0.0]], [[0.0, 0.0, 0.0]]], rtol=0.0, atol=1e-9
        )
    # At 1e6 the log's rates carry a term of 2e5 that the drift absorbs, which float64 rounds at
    # about 5e-11. From that alone the fits differed by 4.5e-10 on CI, and by 5.6e-10 at 2**20.
    np.testing.assert_allclose(
        np.asarray(fits[1e6][0].residual.channel) * 1e6,
        np.asarray(fits[1.0][0].residual.channel),
        rtol=0.0,
        atol=2e-8,
    )
    assert fits[1e6][0].integrator_defect == pytest.approx(
        fits[1.0][0].integrator_defect, rel=1e-6, abs=0.0
    )


@pytest.mark.parametrize("units", [1e-12, 1.0, 1e12])
def test_the_drift_takes_up_each_absorbed_move_in_any_units_of_the_state(units: float) -> None:
    """``_absorbed`` returns the moves of the fit no transition tells from it: unmoved directions
    whose push on the log the drift's features take up, with the drift regression's response, which
    a plan's test reads to tell whether a move cancels at the plan's points. On the ruled log, the
    lever's push along the constant feature is ``-0.3 x0``, which the drift's ``x0`` takes up. Read
    by least squares on the raw drift design, with the state at 1e-12 of its units, the response
    left 1.6e-3 of the push, where the plan's test reads a move as none only within 1.5e-8 of its
    terms; at 1e-13 the rank cutoff dropped the state's columns, and it left all of it."""
    log = _policy_log(4000, _ruled(1.0))
    log = dict(log, x=log["x"] * units, x_next=log["x_next"] * units)
    fit = fit_causal_residual(_known, log, 0.1, **RULED)
    design = jax.vmap(control_affine_features, in_axes=(0, None))(log["x"], fit.residual.degree)
    directions, response = _absorbed(fit, log["x"], log["u"], design)
    assert directions.shape[1] == 1
    raw = _channel_design(log["u"], log["x"], fit.residual.channel_degree)
    push = np.asarray(raw @ directions)
    left = np.linalg.norm(push - np.asarray(design @ response), axis=0)
    assert float(np.max(left / np.linalg.norm(push, axis=0))) < 1e-12


def test_the_representer_carries_what_the_log_reads_where_the_moment_has_no_data() -> None:
    """On the ruled log no direction is moved, so the channel is what the log's rates read along
    the directions the drift cannot take up, linear in the rates and in nothing the nuisances fit.
    A row's weight on it, the representer, is then the change one row makes, to rounding."""
    log = _policy_log(400, _ruled(1.0))
    fit = fit_causal_residual(_known, log, 0.1, influence=True, **RULED)
    assert fit.representer is not None
    for row, state in ((3, 0), (250, 1)):
        moved = dict(log, x_next=log["x_next"].at[row, state].add(1e-3))
        change = np.asarray(fit_causal_residual(_known, moved, 0.1, **RULED).residual.channel)
        change = (change - np.asarray(fit.residual.channel)).ravel() / (1e-3 / 0.1)
        np.testing.assert_allclose(
            np.asarray(fit.representer)[row, state] / 400, change, rtol=0.0, atol=1e-8
        )
    assert float(np.max(np.abs(np.asarray(fit.representer)))) > 1.0


def test_a_channel_the_log_cannot_split_is_split_the_same_in_any_units() -> None:
    """The second action twice the first: the log fixes ``B1 + 2 B2`` on each feature and nothing
    of how it splits. The fit splits it at zero along the unmoved direction once each action is
    scaled to its own size, so logging the second in thousandths moves its coefficients by that
    factor and nothing else."""
    log = _policy_log(4000, lambda rng, x, z: _dithered(rng, x, z) * np.array([1.0, 2.0]))
    milli = dict(log, u=log["u"] * jnp.array([1.0, 1e3]))
    one, other = (fit_causal_residual(_known, data, 0.1, **RULED) for data in (log, milli))
    np.testing.assert_allclose(
        np.asarray(other.residual.channel) * np.array([1.0, 1e3])[None, :, None],
        np.asarray(one.residual.channel),
        rtol=1e-6,
        atol=1e-12,
    )


def _twice(rng: np.random.Generator, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Two actions, the second twice the first, so the moment is solved on the moved directions."""
    return _dithered(rng, x, z) * np.array([1.0, 2.0])


@pytest.mark.parametrize("units", [1e-15, 1e-9, 1e-6, 1e-3, 1e3, 1e6, 1e15])
@pytest.mark.parametrize("actions", [_dithered, _twice], ids=["moved", "split"])
def test_the_channel_reads_the_same_with_the_actions_in_any_units(
    actions: Callable[[np.random.Generator, np.ndarray, np.ndarray], np.ndarray], units: float
) -> None:
    """The moment's ridge was a constant on a Gram in the actions' units squared: logged in
    millionths of their units, the actions read a channel of 0.0008 and -0.0004 where the fit reads
    0.805 and -0.398, the log's being 0.8 and -0.4."""
    log = _policy_log(4000, actions)
    scaled = dict(log, u=log["u"] * units)
    one, other = (fit_causal_residual(_known, data, 0.1, **RULED) for data in (log, scaled))
    np.testing.assert_allclose(
        np.asarray(other.residual.channel) * units,
        np.asarray(one.residual.channel),
        rtol=0.0,
        atol=1e-6,
    )
    assert other.channel_error is not None
    assert other.channel_error * units == pytest.approx(one.channel_error, rel=1e-6, abs=0.0)


@pytest.mark.parametrize("units", [1e-13, 1e-6, 1e-3, 1e3, 1e6, 1e15])
@pytest.mark.parametrize("integrator", ["euler", "rk4"])
def test_the_fit_reads_the_same_with_the_state_in_any_units(integrator: str, units: float) -> None:
    """The drift regression's ridge was a constant on a Gram of the raw state's monomials: logged
    in millionths of its units, the state read a drift slope of -0.0020 where the fit reads -0.475,
    the log's being -0.5. Below 1e-12 of its units the nuisances took the state for a constant and
    left it unscaled under their ridge: at 1e-13 the channel moved by 0.46. A rate scales with the
    state, so the drift's and the channel's intercepts do, and so do the errors, which are read at
    the log's states; the slopes in the state do not."""
    log = _policy_log(4000, _dithered)
    scaled = dict(log, x=log["x"] * units, x_next=log["x_next"] * units)
    one, other = (
        fit_causal_residual(_known, data, 0.1, integrator=integrator, **RULED)
        for data in (log, scaled)
    )
    per_feature = np.array([units, 1.0, 1.0])
    for part in ("drift", "channel"):
        np.testing.assert_allclose(
            np.asarray(getattr(other.residual, part)) / per_feature,
            np.asarray(getattr(one.residual, part)),
            rtol=0.0,
            atol=1e-6,
            err_msg=part,
        )
    for error in ("drift_error", "channel_error"):
        assert getattr(other, error) / units == pytest.approx(
            getattr(one, error), rel=1e-6, abs=0.0
        ), error


@pytest.mark.parametrize(("offset", "tolerance"), [(1e2, 1e-12), (1e4, 1e-10)])
@pytest.mark.parametrize("integrator", ["euler", "rk4"])
def test_the_fit_reads_the_same_with_the_state_far_from_zero(
    integrator: str, offset: float, tolerance: float
) -> None:
    """The drift's bias takes up a state's offset, so the drift's slopes and the channel do not
    move, and its intercept moves by the slopes times the offset. The ridge scaled the slopes'
    terms by the state's mean square, which grows with the offset, and penalised the bias, whose
    size does: with the state a hundred spreads from zero the slopes moved by 3.8e-6 of the
    largest, and ten thousand away by 0.036; the channel, under the RK4 map, by 7.6e-8 and 7.3e-4.
    Solved on the uncentred Gram the slopes moved by 1.7e-11 and 4.3e-7, the rounding at the
    offset's square; with the columns centred, by 5.9e-15 and 4.5e-13."""
    log = _policy_log(4000, _dithered)
    shifted = dict(log, x=log["x"] + offset, x_next=log["x_next"] + offset)
    settings = {"adjust_for": ("z",), "nuisance_degree": 2, "channel_degree": 0}
    one, other = (
        fit_causal_residual(_known, data, 0.1, integrator=integrator, **settings)
        for data in (log, shifted)
    )
    drift, moved = np.asarray(one.residual.drift), np.asarray(other.residual.drift)
    np.testing.assert_allclose(moved[:, 1:], drift[:, 1:], rtol=0.0, atol=tolerance)
    np.testing.assert_allclose(
        moved[:, 0],
        drift[:, 0] - offset * drift[:, 1:].sum(axis=1),
        rtol=0.0,
        atol=tolerance * offset,
    )
    channel = np.asarray(one.residual.channel)
    np.testing.assert_allclose(
        np.asarray(other.residual.channel), channel, rtol=0.0, atol=tolerance
    )


def test_a_ridge_solve_reads_the_same_with_a_column_far_from_zero() -> None:
    """With its bias free and each other column's term scaled by its variance, a ridge solve's fit
    does not see a column's origin, at any ridge: at 0.1 on 50 rows, with the bias penalised and
    the column's term scaled by its mean square, a column a thousand spreads from zero lost 0.9994
    of its slope."""
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 1.0, 50)
    target = jnp.asarray((2.0 + 3.0 * x + rng.normal(0.0, 0.1, 50))[:, None])
    fits = []
    for offset in (0.0, 1e3):
        design = jnp.stack([jnp.ones(50), jnp.asarray(x + offset)], axis=1)
        fits.append(np.asarray(design @ _solve_ridge(design, target, 0.1)))
    np.testing.assert_allclose(fits[1], fits[0], rtol=1e-6, atol=0.0)


@pytest.mark.parametrize("units", [1e-6, 1e6])
@pytest.mark.parametrize("integrator", ["euler", "rk4"])
def test_each_row_s_influence_reads_the_same_with_the_state_in_any_units(
    integrator: str, units: float
) -> None:
    """A row's influence reads the drift's direct response to it through the drift regression,
    under that regression's ridge: logged in millionths of its units, the state moved the influence
    by its own size. Each parameter's influence scales as the parameter does."""
    log = _policy_log(4000, _dithered)
    scaled = dict(log, x=log["x"] * units, x_next=log["x_next"] * units)
    one, other = (
        fit_causal_residual(_known, data, 0.1, integrator=integrator, influence=True, **RULED)
        for data in (log, scaled)
    )
    assert one.influence is not None
    assert other.influence is not None
    per_feature = np.array([units, 1.0, 1.0])
    channel = np.broadcast_to(per_feature, np.asarray(one.residual.channel).shape).ravel()
    # the drift's rows are its features, each beside every state's coefficient on it
    drift = np.repeat(per_feature, np.asarray(one.residual.drift).shape[0])
    psi = np.asarray(one.influence)
    np.testing.assert_allclose(
        np.asarray(other.influence) / np.concatenate([channel, drift]),
        psi,
        rtol=0.0,
        atol=1e-8 * float(np.max(np.abs(psi))),
    )


@pytest.mark.parametrize(("units", "offset"), [(1e-15, 0.0), (1e15, 0.0), (1e-6, 1e3)])
def test_the_channel_reads_the_same_with_the_adjustment_set_in_any_units(
    units: float, offset: float
) -> None:
    """The nuisances standardise the adjustment set, but took a column whose spread was under 1e-12
    in the caller's units for a constant and left it unscaled under their ridge: logged at 1e-13 of
    its units, the confounder was not adjusted for, and the channel moved by 1.27 and its error 2.6
    times. A column is rounding now when its spread is at most 64 eps of its own size (ADR 0057),
    which an offset of the size kelvin carry does not make it; the offset's rounding of the column
    moves the channel by 2.5e-9."""
    log = _policy_log(4000, _dithered)
    one, other = (
        fit_causal_residual(_known, data, 0.1, **RULED)
        for data in (log, dict(log, z=log["z"] * units + offset))
    )
    np.testing.assert_allclose(
        np.asarray(other.residual.channel), np.asarray(one.residual.channel), rtol=0.0, atol=1e-8
    )
    assert other.channel_error == pytest.approx(one.channel_error, rel=1e-8, abs=0.0)


@pytest.mark.parametrize("integrator", ["euler", "rk4"])
@pytest.mark.parametrize(("state", "action"), [(1e-6, 1e6), (1e6, 1e-6)])
def test_the_channel_reads_the_same_in_odd_units_far_from_zero_in_float32(
    integrator: str, state: float, action: float
) -> None:
    """In float32, JAX's default, with the state and the confounder in ``state`` of their units, the
    action in ``action`` of its own, and each column 300 of its units from zero, the channel and
    its error read as at unit scale, to the log's own rounding. In float32 an entry 300 spreads
    from zero is rounded by up to 1.6e-5 of the spread, and that rounding alone, fitted in float64,
    moved the channel by up to 7.7e-5 on these logs; the float32 fit moved it by up to 6.9e-5, and
    its error by 1.4e-4 of itself. The bounds, 2e-4, a sixteenth of the channel's error, and 1e-3,
    are about three and seven times those. Under ``rk4`` the Newton steps were solved in the
    caller's units: the channel moved by 4.4e-4, and its error by 3.4% of itself, and by 30 times
    itself with the columns at their own origin."""
    settings = {"adjust_for": ("z",), "nuisance_degree": 2, "channel_degree": 0}
    log = {name: np.asarray(column) for name, column in _policy_log(4000, _dithered).items()}
    moved = {
        "x": state * (log["x"] + 300.0),
        "x_next": state * (log["x_next"] + 300.0),
        "u": action * (log["u"] + 300.0),
        "z": state * (log["z"] + 300.0),
    }
    with enable_x64(False):
        one, other = (
            fit_causal_residual(
                _known,
                {name: jnp.asarray(column, dtype=jnp.float32) for name, column in data.items()},
                0.1,
                integrator=integrator,
                **settings,
            )
            for data in (log, moved)
        )
    assert other.residual.channel.dtype == jnp.float32
    per_unit = state / action
    np.testing.assert_allclose(
        np.asarray(other.residual.channel) / per_unit,
        np.asarray(one.residual.channel),
        rtol=0.0,
        atol=2e-4,
    )
    assert other.channel_error is not None
    assert other.channel_error / per_unit == pytest.approx(one.channel_error, rel=1e-3, abs=0.0)


@pytest.mark.parametrize("units", [1e-15, 1e15])
def test_the_rule_the_log_set_from_the_state_reads_the_same_in_any_units(units: float) -> None:
    """A plan follows the rule the log set a lever by, read on the state standardised as the log
    was. A state whose spread was under 1e-12 in the caller's units was left unscaled: at 1e-15 of
    its units the rule read 0.30 off, and the state predicted no combination of the actions."""
    log = _policy_log(4000, _ruled(1.0))

    def read(scale: float):
        x = log["x"] * scale
        return _logged_relations(log["u"], x, jnp.concatenate([x, log["z"]], axis=1), 2)

    one, other = read(1.0), read(units)
    np.testing.assert_allclose(np.asarray(other.rule), np.asarray(one.rule), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(other.factor) * units, np.asarray(one.factor), rtol=1e-12, atol=0.0
    )
    for span in ("constant", "state", "covariates"):
        mine, theirs = np.asarray(getattr(other, span)), np.asarray(getattr(one, span))
        assert mine.shape == theirs.shape, span
        np.testing.assert_allclose(mine @ mine.T, theirs @ theirs.T, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("units", [1e-12, 1e-9, 1e9, 1e12])
@pytest.mark.parametrize(
    "actions",
    [
        _both(_ruled(1.0), _dithered),
        lambda rng, x, z: _dithered(rng, x, z) * np.array([1.0, 2.0]),
    ],
    ids=["the first set from the state", "the second twice the first"],
)
def test_what_the_log_kept_reads_the_same_with_one_action_in_other_units(
    actions: Callable[[np.random.Generator, np.ndarray, np.ndarray], np.ndarray], units: float
) -> None:
    """The first action logged in ``units`` times its own. Whether an action is kept at one level,
    or set from the state or the covariates, is whether its axis lies in that span, to the square
    root of the precision; the distance reads the same in any units of the actions. Orthonormalised
    over the raw actions, the span the state predicts lay 3.0e-7 off the first action's axis at
    1e12, and the log read as having set it from outside the state; and the combination kept at
    one level, ``u2 = 2 u1``, lay 2e-12 of the way along ``u1``, so the second action's axis read
    as in that span, and the relation's weight on ``u1`` as -1.99996e-12 where -2e-12."""
    log = _policy_log(4000, actions)
    covariates = jnp.concatenate([log["x"], log["z"]], axis=1)

    def off(scale: float) -> np.ndarray:
        logged = _logged_relations(
            log["u"] * jnp.array([scale, 1.0]), log["x"], covariates, nuisance_degree=2
        )
        distances = []
        for span in ("constant", "state", "covariates"):
            basis = np.asarray(getattr(logged, span))
            distances.append(
                [np.linalg.norm(axis - basis @ (basis.T @ axis)) for axis in np.eye(2)]
            )
        return np.array(distances)

    np.testing.assert_allclose(off(units), off(1.0), rtol=0.0, atol=1e-12)


def test_an_action_the_log_never_used_is_each_span_it_kept() -> None:
    """The second action never used: its column of the log is zeros, whose size reads 1, and its
    axis is the whole of each span, the one kept at one level first, in any units of the first."""
    log = _policy_log(4000, lambda rng, x, z: _dithered(rng, x, z) * np.array([1.0, 0.0]))
    covariates = jnp.concatenate([log["x"], log["z"]], axis=1)
    for units in (1e-12, 1.0, 1e12):
        logged = _logged_relations(
            log["u"] * jnp.array([units, 1.0]), log["x"], covariates, nuisance_degree=2
        )
        for span in ("constant", "state", "covariates"):
            basis = np.abs(np.asarray(getattr(logged, span)))
            np.testing.assert_allclose(basis, [[0.0], [1.0]], rtol=0.0, atol=1e-12, err_msg=span)


@pytest.mark.parametrize("units", [1e-15, 1e-12, 1e15])
def test_where_the_moment_has_no_data_the_fit_reads_the_same_with_the_state_in_any_units(
    units: float,
) -> None:
    """On the ruled log the fit splits the channel by least squares on the drift's design, whose
    cutoff is relative to its largest singular value: with the state at 1e-12 of its units, the
    state's columns fell under it beside the constant's, and the channel moved by 50; at 1e15 the
    constant's did, and it moved by 0.02 (ADR 0054)."""
    log = _policy_log(4000, _ruled(1.0))
    scaled = dict(log, x=log["x"] * units, x_next=log["x_next"] * units)
    one, other = (fit_causal_residual(_known, data, 0.1, **RULED) for data in (log, scaled))
    per_feature = np.array([units, 1.0, 1.0])
    for part in ("drift", "channel"):
        np.testing.assert_allclose(
            np.asarray(getattr(other.residual, part)) / per_feature,
            np.asarray(getattr(one.residual, part)),
            rtol=0.0,
            atol=1e-9,
            err_msg=part,
        )


@pytest.mark.parametrize("units", [1e-15, 1e-6, 1e6, 1e15])
def test_the_channel_reads_the_same_with_the_instrument_in_any_units(units: float) -> None:
    """The instrument's first stage regressed the action on the raw monomials of the state and the
    instrument under a constant ridge: logged in millionths of its units, the instrument moved the
    channel by up to 0.36."""
    system = _system(instrument_to_action=jnp.array([[0.8]]))
    data = system.sample(4000, jax.random.key(0), _known)
    scaled = dict(data, w=data["w"] * units)
    one, other = (
        fit_causal_residual(_known, log, system.dt, instrument="w") for log in (data, scaled)
    )
    np.testing.assert_allclose(
        np.asarray(other.residual.channel), np.asarray(one.residual.channel), rtol=0.0, atol=1e-6
    )


def test_a_column_of_zeros_keeps_a_ridge_term_and_reads_a_coefficient_of_zero() -> None:
    """A state or a driver logged at zero in every row has no mean square to scale the ridge by,
    and without a term of its own the Gram is singular there."""
    design = jnp.stack([jnp.ones(50), jnp.linspace(-1.0, 1.0, 50), jnp.zeros(50)], axis=1)
    coefficients = _solve_ridge(design, 2.0 * design[:, 1:2] + 1.0, 1e-6)
    np.testing.assert_allclose(
        np.asarray(coefficients).ravel(), [1.0, 2.0, 0.0], rtol=0.0, atol=1e-6
    )


def test_the_centred_ridge_parts_are_the_raw_gram_s_own() -> None:
    """Put together from the centred columns, the inverse, the rows' weights and the solve are the
    raw Gram's under the same ridge, where that Gram is well conditioned."""
    rng = np.random.default_rng(7)
    design = jnp.asarray(np.column_stack([np.ones(40), rng.normal(2.0, 1.0, (40, 3))]))
    target = jnp.asarray(rng.normal(size=(40, 2)))
    scales = jnp.concatenate([jnp.zeros(1), _units.centred(design[:, 1:]).scales])
    gram = design.T @ design + 0.3 * jnp.diag(scales)
    inverse = np.linalg.inv(np.asarray(gram))
    weight = inverse @ np.asarray(design).T
    np.testing.assert_allclose(_ridge_inverse(design, 0.3), inverse, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(_ridge_weight(design, 0.3), weight, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(
        _solve_ridge(design, target, 0.3), weight @ np.asarray(target), rtol=1e-10, atol=1e-13
    )


def _near_constant(column: np.ndarray, noise: float = 0.01) -> tuple[jax.Array, jax.Array]:
    """A ridge solve's coefficients and fitted values on a bias, a slope and ``column``, with the
    target moving with the slope by 2 and with a standard normal ``z`` by 0.5."""
    rng = np.random.default_rng(4)
    x = np.linspace(-1.0, 1.0, 50)
    z = rng.normal(0.0, 1.0, 50)
    target = jnp.asarray((1.0 + 2.0 * x + 0.5 * z + rng.normal(0.0, noise, 50))[:, None])
    design = jnp.stack([jnp.ones(50), jnp.asarray(x), jnp.asarray(column)], axis=1)
    coefficients = _solve_ridge(design, target, 1e-6)
    return coefficients, design @ coefficients


def test_a_column_close_to_a_constant_is_one_far_from_zero() -> None:
    """A column whose spread is 1e-11 of its size is a column in small units far from zero, and the
    solve reads it as it reads the same column standardised: its slope over its spread. On the
    uncentred Gram it was the bias's twin to rounding, under a term scaled to its spread, and the
    solve read nan."""
    z = np.random.default_rng(4).normal(0.0, 1.0, 50)  # the target's z, as _near_constant draws it
    near, near_fitted = _near_constant(1.0 + 1e-11 * z)
    standard, standard_fitted = _near_constant(z)
    assert np.all(np.isfinite(np.asarray(near)))
    np.testing.assert_allclose(np.asarray(near_fitted), np.asarray(standard_fitted), atol=1e-4)
    assert float(near[2, 0]) * 1e-11 == pytest.approx(float(standard[2, 0]), rel=1e-4, abs=0.0)


@pytest.mark.parametrize("level", [1.0, 1e12])
def test_a_column_constant_but_for_rounding_is_zeroed(level: float) -> None:
    """A column whose spread is rounding, here 16 eps of its size, is zeroed about its mean: its
    coefficient is exactly zero, and the bias and the slope are the solve's without it. Kept beside
    the bias it repeats under a ridge scaled by its mean square, it read -2.2e-8 and moved them by
    as much; under a term scaled to its spread, it took up the target's noise with a coefficient of
    -3.0e13, which moved the fitted values by 0.12. With its row and column of the inverse 0 but
    its deviations kept, at 1e12 they moved the slope by 1.2e-3 through the other columns' block."""
    rounding = 16.0 * np.finfo(np.float64).eps * np.random.default_rng(5).choice([-1.0, 1.0], 50)
    coefficients, fitted = _near_constant(level * (1.0 + rounding))
    without, fitted_without = _near_constant(np.zeros(50))
    assert float(coefficients[2, 0]) == 0.0
    assert bool(jnp.array_equal(coefficients[:2], without[:2]))
    assert bool(jnp.array_equal(fitted, fitted_without))


@pytest.mark.parametrize(
    "column",
    [
        np.full(50, 21.3),
        1.0 + 16.0 * np.finfo(np.float64).eps * np.random.default_rng(5).choice([-1.0, 1.0], 50),
        np.zeros(50),
    ],
    ids=["21.3 in every row", "constant but for rounding", "zeros"],
)
def test_a_rounding_column_leaves_the_bias_s_variance_as_without_it(column: np.ndarray) -> None:
    """A zeroed column's row and column of the solve's inverse are 0: its coefficient is exactly 0,
    with no variance, and the bias and the slope read the variances of the solve without it. Kept
    beside the bias it repeats, under a ridge scaled by its mean square, a column at 21.3 or at 1
    gave the bias a variance of 1e6, the inverse of the ridge, where the solve without it reads
    0.02; a column of zeros kept a variance of its own of 1e6."""
    x = jnp.linspace(-1.0, 1.0, 50)
    without = _ridge_inverse(jnp.stack([jnp.ones(50), x], axis=1), 1e-6)
    inverse = _ridge_inverse(jnp.stack([jnp.ones(50), x, jnp.asarray(column)], axis=1), 1e-6)
    np.testing.assert_allclose(inverse[:2, :2], without, rtol=0.0, atol=1e-15)
    assert bool(jnp.all(inverse[2] == 0.0))
    assert bool(jnp.all(inverse[:, 2] == 0.0))


def test_the_public_moment_reads_a_state_logged_at_zero_as_no_slope() -> None:
    """A state logged at zero in every row makes its slope's column of the moment zero; the
    moment's penalty keeps a term of 1 there, without which its Gram is singular."""
    rng = np.random.default_rng(6)
    u_res = jnp.asarray(rng.normal(size=(500, 1)))
    y_res = 0.8 * u_res + 0.1 * jnp.asarray(rng.normal(size=(500, 1)))
    channel = np.asarray(solve_channel_moment(y_res, u_res, jnp.zeros((500, 1))))
    assert channel[0, 0, 1] == 0.0
    assert channel[0, 0, 0] == pytest.approx(0.8, abs=0.02)


def test_the_public_moment_reads_the_same_in_any_units() -> None:
    """``solve_channel_moment`` scales its ridge by the regressor's own columns, the only size it
    is given: residuals in millionths of their units read a channel of 0.0016 and 0.0007 where they
    read 0.80 and 0.40."""
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.normal(size=(2000, 1)))
    u_res = jnp.asarray(rng.normal(size=(2000, 1)))
    y_res = 0.8 * u_res * (1.0 + 0.5 * x) + 0.1 * jnp.asarray(rng.normal(size=(2000, 1)))
    one = np.asarray(solve_channel_moment(y_res, u_res, x))
    small = np.asarray(solve_channel_moment(y_res, 1e-6 * u_res, x))
    np.testing.assert_allclose(small * 1e-6, one, rtol=1e-9, atol=0.0)


# ---- a channel error summed within clusters ----

CLUSTERED = {"adjust_for": ("z",), "channel_degree": 0, "nuisance_degree": 1, "folds": 1}


def _clustered_log(rows: int = 100, seed: int = 23) -> dict[str, jax.Array]:
    """One state and one lever, confounded by ``z``, the lever's own part ``a`` and the rate's
    noise ``e`` drawn afresh on each row."""
    rng = np.random.default_rng(seed)
    x, z, a, e = rng.normal(size=(4, rows))
    u = z + a
    after = x + 0.1 * (-0.5 * x + 0.8 * u + 1.5 * z + 0.2 * e)
    return {
        "x": jnp.asarray(x[:, None]),
        "u": jnp.asarray(u[:, None]),
        "z": jnp.asarray(z[:, None]),
        "x_next": jnp.asarray(after[:, None]),
    }


def _repeated(data: dict[str, jax.Array], times: int) -> dict[str, jax.Array]:
    return {name: jnp.repeat(column, times, axis=0) for name, column in data.items()}


def test_rows_repeated_and_clustered_by_their_original_read_the_original_error() -> None:
    """Each row of a log of 100 repeated 16 times: row by row the error reads a quarter of the
    original's, as 1600 independent rows would; summed within each row's copies it reads the
    original's, since ``(N - 1) / (N - k)`` is 1 at one coefficient and ``G / (G - 1)`` is the
    original's ``N / (N - k)``."""
    data = _clustered_log()
    original = fit_causal_residual(_known, data, 0.1, **CLUSTERED)
    copies = _repeated(data, 16)
    rows = fit_causal_residual(_known, copies, 0.1, **CLUSTERED)
    clustered = fit_causal_residual(
        _known, copies, 0.1, **CLUSTERED, clusters=np.repeat(np.arange(100), 16)
    )
    assert original.channel_error is not None
    assert rows.channel_error is not None
    assert rows.channel_error / original.channel_error == pytest.approx(0.25, abs=0.002)
    assert clustered.channel_error == pytest.approx(original.channel_error, rel=1e-6, abs=0.0)
    assert clustered.residual.channel == pytest.approx(original.residual.channel, abs=1e-7)


@pytest.mark.parametrize("integrator", ["euler", "rk4"])
def test_each_transition_its_own_cluster_is_the_row_by_row_error_on_one_state(
    integrator: str,
) -> None:
    data = _clustered_log(400, seed=4)
    options = {**CLUSTERED, "integrator": integrator, "influence": True}
    rows = fit_causal_residual(_known, data, 0.1, **options)
    own = fit_causal_residual(_known, data, 0.1, **options, clusters=np.arange(400))
    assert own.channel_error == pytest.approx(rows.channel_error, rel=1e-12, abs=0.0)
    np.testing.assert_allclose(own.influence, rows.influence, rtol=1e-12, atol=0.0)
    assert own.clusters is not None
    assert own.clusters.tolist() == list(range(400))
    assert rows.clusters is None


def test_under_euler_a_row_s_states_summed_leave_every_state_s_channel_error_as_it_was() -> None:
    """Under Euler each state's channel reads its own column of rates, so summing a row's states
    within its cluster adds only the covariances between two states' channels, which the error does
    not read: each transition its own cluster reads the row-by-row error on every state."""
    data = _system().sample(600, jax.random.key(3), _known)
    rows = fit_causal_residual(_known, data, 0.1, adjust_for=("z",))
    own = fit_causal_residual(_known, data, 0.1, adjust_for=("z",), clusters=np.arange(600))
    assert own.channel_error == pytest.approx(rows.channel_error, rel=1e-12, abs=0.0)


def test_the_clustered_error_is_cr1_by_hand() -> None:
    """Own-sample partialling-out on a linear basis with no ridge is Frisch-Waugh-Lovell, so the
    channel is ``u_res' y / u_res' u_res`` and each row moves it by ``u_res_i / u_res' u_res``: the
    error is CR1's over the clusters' summed scores, whatever their sizes."""
    rows = 300
    data = _clustered_log(rows, seed=5)
    labels = np.array([f"g{g:02d}" for g in np.random.default_rng(1).integers(0, 30, rows)])
    fit = fit_causal_residual(_known, data, 0.1, **CLUSTERED, ridge=0.0, clusters=labels)

    x, z, u = (np.asarray(data[name])[:, 0] for name in ("x", "z", "u"))
    y = (np.asarray(data["x_next"])[:, 0] - x) / 0.1
    basis = np.column_stack([np.ones(rows), x, z])
    projector = basis @ np.linalg.pinv(basis)
    u_res, y_res = u - projector @ u, y - projector @ y
    channel = u_res @ y_res / (u_res @ u_res)
    score = u_res / (u_res @ u_res) * (y_res - channel * u_res)
    _, codes = np.unique(labels, return_inverse=True)
    summed = np.bincount(codes, weights=score)
    groups = summed.size  # (N - 1) / (N - k) is 1 at one coefficient
    variance = groups / (groups - 1) * np.sum(summed**2)
    assert float(fit.residual.channel[0, 0, 0]) == pytest.approx(channel, rel=1e-9, abs=0.0)
    assert fit.channel_error == pytest.approx(math.sqrt(variance), rel=1e-9, abs=0.0)


@pytest.mark.parametrize("integrator", ["euler", "rk4"])
def test_clusters_move_the_error_and_nothing_else(integrator: str) -> None:
    data = _clustered_log(400, seed=6)
    options = {**CLUSTERED, "integrator": integrator, "influence": True}
    rows = fit_causal_residual(_known, data, 0.1, **options)
    clustered = fit_causal_residual(_known, data, 0.1, **options, clusters=np.arange(400) // 8)
    assert clustered.channel_error != rows.channel_error
    # each row's influence is the same but for the factor: CR1's over 50 clusters, not N / (N - k)
    factor = math.sqrt((50 / 49 * 399 / 399) / (400 / 399))
    np.testing.assert_allclose(clustered.influence, factor * rows.influence, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(clustered.residual.channel, rows.residual.channel)
    np.testing.assert_array_equal(clustered.residual.drift, rows.residual.drift)
    np.testing.assert_array_equal(clustered.representer, rows.representer)
    np.testing.assert_array_equal(clustered.moment_residual, rows.moment_residual)
    moved = {"channel_error", "influence", "clusters"}
    for field in dataclasses.fields(CausalDynamicsFit):
        if field.name not in moved | {"residual", "representer", "moment_residual", "unmoved"}:
            assert getattr(clustered, field.name) == getattr(rows, field.name), field.name


def test_clusters_are_refused_unless_they_label_every_transition_and_name_two() -> None:
    data = _clustered_log(50)
    with pytest.raises(ValueError, match="label each of the 50 transitions once"):
        fit_causal_residual(_known, data, 0.1, **CLUSTERED, clusters=np.arange(49))
    with pytest.raises(ValueError, match="label each of the 50 transitions once"):
        fit_causal_residual(_known, data, 0.1, **CLUSTERED, clusters=np.zeros((50, 1)))
    with pytest.raises(ValueError, match="name one cluster"):
        fit_causal_residual(_known, data, 0.1, **CLUSTERED, clusters=np.full(50, "store"))
    with pytest.raises(ValueError, match="label each of the 50 transitions once"):
        fit_causal_residual(_known, data, 0.1, **CLUSTERED, clusters=np.zeros((50, 3)))
    one_period = np.column_stack([np.arange(50) % 5, np.zeros(50)])
    with pytest.raises(ValueError, match="name one cluster in one of their two dimensions"):
        fit_causal_residual(_known, data, 0.1, **CLUSTERED, clusters=one_period)


def test_the_influence_s_square_is_its_rows_or_its_clusters_sums_squared() -> None:
    rows = np.arange(24.0).reshape(4, 2, 3)
    flat = rows.reshape(8, 3)
    (independent,) = _clustered_squares(rows, None)
    np.testing.assert_array_equal(independent, flat.T @ flat)
    summed = np.array([rows[0].sum(0) + rows[2].sum(0), rows[1].sum(0) + rows[3].sum(0)])
    (clustered,) = _clustered_squares(rows, np.array([0, 1, 0, 1]))
    np.testing.assert_array_equal(clustered, summed.T @ summed)
    assert _clustered_squares(rows[:, :, 0], np.array([1, 0, 0, 0])) == (81**2 + 3**2,)


# ---- clustered two ways ----


def _two_way_sums(rows: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The two dimensions' sums squared less their cells', nothing read as nothing."""

    def square(codes: np.ndarray) -> np.ndarray:
        group = np.unique(codes, axis=0, return_inverse=True)[1].reshape(-1)
        summed = np.zeros((group.max() + 1, rows.shape[1]))
        np.add.at(summed, group, rows)
        return summed.T @ summed

    return square(labels[:, 0]) + square(labels[:, 1]) - square(labels)


def test_two_ways_read_their_sums_and_each_way_s_alone() -> None:
    """Units (0, 0, 0, 1, 1, 1) over periods (0, 1, 2, 0, 1, 2), the values 1 to 6: the units'
    sums 6 and 15 square to 261, the periods' 5, 7 and 9 to 155, and the cells' to 91, so two
    ways read 261 + 155 - 91 = 325 (Cameron, Gelbach and Miller 2011). The values carry CR1's
    factor for the 2 units, ``2 / 1``, as an influence does; the 3 periods' own is ``3 / 2``, so
    they alone read 155 * 0.75."""
    codes = np.column_stack([np.repeat([0, 1], 3), np.tile([0, 1, 2], 2)])
    values = np.arange(1.0, 7.0)[:, None]
    assert _clustered_squares(values, codes) == (325.0, 261.0, 116.25)
    assert _clustered_squares(values, codes[:, ::-1]) == (325.0, 116.25, 261.0)


def test_the_two_way_error_is_the_largest_of_three_by_hand() -> None:
    """Two labels a transition, a unit and a period say. Two ways read the units' summed scores
    squared, plus the periods', less the cells', which both count, at CR1's factor for the smaller
    dimension's ``G``; each way alone reads its own sums at its own. The error is the largest of
    the three (MacKinnon, Nielsen and Webb 2024), here the two ways'. The channel is the
    Frisch-Waugh-Lovell one of the one-way test."""
    rows = 300
    data = _clustered_log(rows, seed=5)
    rng = np.random.default_rng(7)
    labels = np.column_stack([rng.integers(0, 30, rows), rng.integers(0, 12, rows)])
    fit = fit_causal_residual(_known, data, 0.1, **CLUSTERED, ridge=0.0, clusters=labels)

    x, z, u = (np.asarray(data[name])[:, 0] for name in ("x", "z", "u"))
    y = (np.asarray(data["x_next"])[:, 0] - x) / 0.1
    basis = np.column_stack([np.ones(rows), x, z])
    projector = basis @ np.linalg.pinv(basis)
    u_res, y_res = u - projector @ u, y - projector @ y
    channel = u_res @ y_res / (u_res @ u_res)
    score = u_res / (u_res @ u_res) * (y_res - channel * u_res)

    def squared(codes: np.ndarray) -> float:
        group = np.unique(codes, axis=0, return_inverse=True)[1].reshape(-1)
        return float(np.sum(np.bincount(group, weights=score) ** 2))

    units, periods = (np.unique(labels[:, k]).size for k in (0, 1))
    assert (units, periods) == (30, 12)  # (N - 1) / (N - k) is 1 at one coefficient
    two_way = squared(labels[:, 0]) + squared(labels[:, 1]) - squared(labels)
    reads = (
        periods / (periods - 1) * two_way,
        units / (units - 1) * squared(labels[:, 0]),
        periods / (periods - 1) * squared(labels[:, 1]),
    )
    assert len(set(reads)) == 3
    assert max(reads) == reads[0]
    assert fit.clusters is not None
    assert fit.clusters.shape == (rows, 2)
    assert fit.channel_error == pytest.approx(math.sqrt(max(reads)), rel=1e-9, abs=0.0)


@pytest.mark.parametrize("integrator", ["euler", "rk4"])
def test_a_second_way_that_holds_each_transition_alone_reads_the_first_way(
    integrator: str,
) -> None:
    """Periods each one transition's own add each row's square and take it away as the cells', so
    two ways read the units' sums, and in either order of the two. Each of 50 transitions is
    logged 8 times, a unit its copies, so the rows alone read less than the units do."""
    data = _repeated(_clustered_log(50, seed=6), 8)
    options = {**CLUSTERED, "integrator": integrator, "influence": True}
    units, alone = np.arange(400) // 8, np.arange(400)
    one_way = fit_causal_residual(_known, data, 0.1, **options, clusters=units)
    rows = fit_causal_residual(_known, data, 0.1, **options, clusters=alone)
    assert rows.channel_error < one_way.channel_error
    for labels in (np.column_stack([units, alone]), np.column_stack([alone, units])):
        two_way = fit_causal_residual(_known, data, 0.1, **options, clusters=labels)
        assert two_way.channel_error == pytest.approx(one_way.channel_error, rel=1e-12, abs=0.0)
        np.testing.assert_allclose(two_way.influence, one_way.influence, rtol=1e-12, atol=0.0)
        np.testing.assert_array_equal(two_way.residual.channel, one_way.residual.channel)


def test_a_two_way_direction_read_below_nothing_is_read_as_nothing() -> None:
    """Scores that cancel within each unit and each period but not within a cell: two ways read
    nothing and the cells 4, so that direction reads -4 and is read as 0, while a direction the
    same in every row keeps its 8 + 8 - 4, and each way alone reads 8 there. The fit's spreads
    are the same, with CR1's factor."""
    codes = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    cancelling = np.array([1.0, -1.0, -1.0, 1.0])
    assert _clustered_squares(cancelling[:, None], codes) == (0.0, 0.0, 0.0)
    rows = np.column_stack([cancelling, np.ones(4)])
    expected = (np.diag([0.0, 12.0]), np.diag([0.0, 8.0]), np.diag([0.0, 8.0]))
    for got, want in zip(_clustered_squares(rows[:, None, :], codes), expected, strict=True):
        np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)
    sensitivity = jnp.asarray(rows.T[:, :, None])  # (coefficients, transitions, states)
    spreads = _robust_spreads(sensitivity, jnp.ones((4, 1)), 1, codes)
    for got, want in zip(spreads, expected, strict=True):
        np.testing.assert_allclose(got, 2.0 * want, rtol=0.0, atol=1e-12)  # G / (G - 1) = 2


def test_the_part_read_as_nothing_moves_with_the_coefficients() -> None:
    """The scores' columns mixed, ``rows @ B``, mix every covariance as ``B' V B``, the part read
    as nothing included, since it is found against the three sums together, which mix the same
    way. Clipped along the two ways' own eigenvectors instead (Cameron, Gelbach and Miller 2011),
    the mixed sums ``[[8, 12], [12, 12]]`` read ``[[9.26, 10.93], [10.93, 12.90]]``, not
    ``B' diag(0, 12) B``. A covariance with no part below nothing keeps every bit."""
    codes = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    rows = np.column_stack([[1.0, -1.0, -1.0, 1.0], np.ones(4)])
    mix = np.array([[1.0, 0.0], [1.0, 1.0]])
    plain = _clustered_squares(rows[:, None, :], codes)
    mixed = _clustered_squares((rows @ mix)[:, None, :], codes)
    for one, other in zip(plain, mixed, strict=True):
        np.testing.assert_allclose(other, mix.T @ one @ mix, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(mixed[0], np.full((2, 2), 12.0), rtol=1e-12, atol=0.0)
    definite = np.array([[2.0, 1.0], [1.0, 3.0]])
    np.testing.assert_array_equal(_less_below_nothing(definite, 3.0 * definite), definite)


def test_the_part_below_nothing_is_measured_against_the_three_sums_together() -> None:
    """``every``, the three sums with every sign a plus, bounds the two ways' sums on both sides,
    so the generalised eigenproblem ``spread v = l every v``, ``v' every v = 1``, has its ``l`` in
    ``[-1, 1]``, and the part below nothing is ``every V min(L, 0) V' every``. SciPy's
    generalised solver, a route of its own, reads the same. Like ``every``, the cells' sums alone
    and the units' and periods' together are each ``a spread + b cells``, so they have its
    eigenvectors and take away the same part; the units' alone do not."""
    codes = np.column_stack([np.repeat(np.arange(3), 4), np.tile(np.arange(4), 3)])
    rows = np.random.default_rng(0).normal(size=(12, 3))

    def square(labels: np.ndarray) -> np.ndarray:
        group = np.unique(labels, axis=0, return_inverse=True)[1].reshape(-1)
        summed = np.zeros((group.max() + 1, 3))
        np.add.at(summed, group, rows)
        return summed.T @ summed

    first, second, cells = square(codes[:, 0]), square(codes[:, 1]), square(codes)
    spread, every = first + second - cells, first + second + cells
    shares, vectors = scipy.linalg.eigh(spread, every)
    assert shares[0] < 0.0 < shares[1]
    assert shares[0] >= -1.0
    assert shares[-1] <= 1.0
    lifted = every @ vectors
    expected = spread - (lifted * np.minimum(shares, 0.0)) @ lifted.T
    np.testing.assert_allclose(_clustered_squares(rows[:, None, :], codes)[0], expected, rtol=1e-10)
    for alike in (cells, first + second):
        np.testing.assert_allclose(_less_below_nothing(spread, alike), expected, rtol=1e-10)
    assert not np.allclose(_less_below_nothing(spread, first), expected, rtol=1e-3)


def test_two_ways_that_read_nothing_read_the_larger_way_alone() -> None:
    """On 3 units over 4 periods the two ways' sums read below nothing in every direction of the
    channel, so read as nothing they would say the channel is known exactly. The error is the
    larger way's alone, the units'."""
    data = _clustered_log(60, seed=0)
    labels = np.column_stack([np.arange(60) // 20, np.arange(60) % 4])
    options = {**CLUSTERED, "channel_degree": 1, "ridge": 0.0, "influence": True}
    both, units, periods = (
        fit_causal_residual(_known, data, 0.1, **options, clusters=clusters)
        for clusters in (labels, labels[:, 0], labels[:, 1])
    )
    size = both.residual.channel.size
    influence = np.asarray(both.influence)[:, :, :size]
    assert np.all(np.linalg.eigvalsh(_two_way_sums(influence.sum(axis=1), labels)) < 0.0)
    squares = _clustered_squares(influence, labels)
    np.testing.assert_allclose(squares[0], 0.0, rtol=0.0, atol=1e-12 * np.abs(squares[1]).max())
    assert units.channel_error > periods.channel_error
    assert both.channel_error == pytest.approx(units.channel_error, rel=1e-12, abs=0.0)


def test_two_ways_whose_sums_read_less_than_a_way_alone_bound_as_that_way() -> None:
    """On 3 units over 4 periods the two ways' sums of the channel's influence read 2.8e-4, the
    periods' 7.7e-4 and the units' 1.1e-3, so the error, and the omitted confounder's interval
    about the bias bounds, are the units'."""
    data = _clustered_log(60, seed=0)
    labels = np.column_stack([np.arange(60) // 20, np.arange(60) % 4])
    options = {**CLUSTERED, "ridge": 0.0, "influence": True}
    both, units = (
        fit_causal_residual(_known, data, 0.1, **options, clusters=clusters)
        for clusters in (labels, labels[:, 0])
    )
    sums, by_unit, by_period = _clustered_squares(np.asarray(both.influence)[:, :, :1], labels)
    assert 0.0 < sums < by_period < by_unit
    assert both.channel_error == pytest.approx(units.channel_error, rel=1e-12, abs=0.0)
    shares = {"cf_y": 0.05, "cf_d": 0.05}
    got, want = (
        omitted_confounder_bound(fit, np.ones((1, 1, 1)), **shares) for fit in (both, units)
    )
    for name in ("ci_lower", "ci_upper"):
        assert getattr(got, name) == pytest.approx(getattr(want, name), rel=1e-12, abs=0.0), name


def test_a_two_way_error_does_not_move_with_the_zero_of_the_state() -> None:
    """The channel affine in the state, moving the state's zero mixes its coefficients. On 4
    units over 3 periods the two ways' sums read a direction below nothing, and still more than
    either way alone, so the error is theirs, and it does not move."""
    data = _clustered_log(60, seed=1)
    labels = np.column_stack([np.arange(60) // 15, np.arange(60) % 3])
    options = {**CLUSTERED, "channel_degree": 1, "ridge": 0.0, "influence": True}
    here = fit_causal_residual(_known, data, 0.1, **options, clusters=labels)
    moved = {**data, "x": data["x"] + 3.0, "x_next": data["x_next"] + 3.0}
    there = fit_causal_residual(_known, moved, 0.1, **options, clusters=labels)
    influence = np.asarray(here.influence)[:, :, : here.residual.channel.size].sum(axis=1)
    assert np.linalg.eigvalsh(_two_way_sums(influence, labels))[0] < 0.0
    alone = (
        fit_causal_residual(_known, data, 0.1, **options, clusters=labels[:, k]).channel_error
        for k in (0, 1)
    )
    assert here.channel_error > max(alone)
    assert there.channel_error == pytest.approx(here.channel_error, rel=1e-12, abs=0.0)


FIT_SECOND_ROLES = {
    "the action as a covariate": ({"adjust_for": ("z", "u")}, "the covariate 'u' is the action"),
    "the outcome as a covariate": (
        {"adjust_for": ("z", "x_next")},
        "the covariate 'x_next' is the next state",
    ),
    "the action as its instrument": ({"instrument": "u"}, "the instrument 'u' is the action"),
    "the state as an instrument": ({"instrument": "x"}, "the instrument 'x' is the state"),
    "an instrument adjusted for": (
        {"adjust_for": ("z", "w"), "instrument": "w"},
        "the instrument 'w' is also a covariate",
    ),
    "the action as a driver": ({"adjust_for": ("z",), "drivers": ("u",)}, "the driver 'u' is"),
}


@pytest.mark.parametrize(
    ("kwargs", "match"), FIT_SECOND_ROLES.values(), ids=FIT_SECOND_ROLES.keys()
)
def test_a_column_named_in_a_second_role_is_refused(kwargs: dict, match: str) -> None:
    """The fit reads the state, the action and the next state as ``x``, ``u`` and ``x_next`` and
    every other column by its name, from one dict. Adjusted for the action as well as the
    confounder, the channel came back as the unadjusted fit's: -0.209 where the truth is 1.0."""
    system = _system(instrument_to_action=jnp.array([[0.8]]))
    data = system.sample(500, jax.random.key(0), _known)
    with pytest.raises(ValueError, match=match):
        fit_causal_residual(_known, data, system.dt, **kwargs)


# ---- the instrument's rank ----

IV_SYSTEM = _system(instrument_to_action=jnp.array([[0.8]]))
ONE_FEATURE = {"nuisance_degree": 1, "channel_degree": 0}


def _iv_log(rows: int = 2000) -> dict[str, jax.Array]:
    return IV_SYSTEM.sample(rows, jax.random.key(0), _known)


def _assert_not_identified(fit: CausalDynamicsFit, without: CausalDynamicsFit) -> None:
    """``fit`` reads as a fit with no instrument and no adjustment does: no error, and the channel
    of ``without``, the same fit with no instrument, kept to compare."""
    assert fit.identified is False
    assert fit.method == "observational"
    assert fit.channel_error is None
    assert fit.drift_error is None
    np.testing.assert_array_equal(
        np.asarray(fit.residual.channel), np.asarray(without.residual.channel)
    )


def test_the_review_s_zero_instrument_identifies_nothing() -> None:
    """No state, a lever that alternates in sign, and an instrument of zeros: its moment holds at
    every channel. Before the rank check the fit read identified, the channel an exact 0.0, its
    error 0.0 and the moment's norm 0.0, where the log's own channel is 0.8. Its nuisance is of
    degree 0, whose first stage is a constant that no instrument enters; the plant's logs below
    read a zero instrument under first stages that one would enter."""
    rows = 200
    u = np.tile([-1.0, 1.0, -1.0, 1.0], rows // 4)
    e = np.tile([-1.0, -1.0, 1.0, 1.0], rows // 4)
    data = {
        "x": jnp.zeros((rows, 1)),
        "u": jnp.asarray(u[:, None]),
        "x_next": jnp.asarray((0.1 * (0.8 * u + 0.1 * e))[:, None]),
        "iv": jnp.zeros((rows, 1)),
    }
    keywords = {"degree": 0, "channel_degree": 0, "nuisance_degree": 0, "folds": 1}
    fit = fit_causal_residual(_known, data, 0.1, instrument="iv", **keywords)
    _assert_not_identified(fit, fit_causal_residual(_known, data, 0.1, **keywords))
    assert fit.instrument_rank == 0
    np.testing.assert_array_equal(np.asarray(fit.instrument_relevance), [0.0])


@pytest.mark.parametrize(
    ("rows", "keywords"),
    [
        (2000, {}),
        (2000, ONE_FEATURE),
        (2000, {"integrator": "rk4"}),
        (2000, {"adjust_for": ("z",)}),
        (200, {"nuisance_degree": 3}),
    ],
    ids=[
        "affine channel",
        "constant channel",
        "rk4",
        "beside an adjustment set",
        "200 rows, a nuisance of degree 3",
    ],
)
def test_a_zero_instrument_leaves_the_fit_unidentified(rows: int, keywords: dict) -> None:
    """The confounded plant with its instrument set to zero. Before the rank check the affine
    channel read identified at ``[0.81, -0.06]`` with an error of 6.0, where the truth is
    ``[1.0, 0.5]``. Beside an adjustment set, the instrument names a confounder the covariates
    leave, so the fit does not fall back on them. On 200 rows under a nuisance of degree 3 the
    push of zeros is rounding of up to 6 eps of the lever's size, which reads as none."""
    data = {**_iv_log(rows), "w": jnp.zeros((rows, 1))}
    fit = fit_causal_residual(
        _known, data, IV_SYSTEM.dt, instrument="w", influence=True, **keywords
    )
    _assert_not_identified(fit, fit_causal_residual(_known, data, IV_SYSTEM.dt, **keywords))
    assert fit.influence is None
    assert fit.instrument_rank == 0
    np.testing.assert_array_equal(np.asarray(fit.instrument_relevance), 0.0)


def test_an_instrument_orthogonal_to_the_action_identifies_nothing() -> None:
    """A column of noise less its least-squares projection on the state and the action, under a
    first stage linear in the state and the instrument, so the action moves with no part of it."""
    data = _iv_log()
    noise = jax.random.normal(jax.random.key(7), (2000, 1))
    design = jnp.concatenate([jnp.ones((2000, 1)), data["x"], data["u"]], axis=1)
    data["w"] = noise - design @ jnp.linalg.lstsq(design, noise)[0]
    keywords = {"nuisance_degree": 1}
    fit = fit_causal_residual(_known, data, IV_SYSTEM.dt, instrument="w", **keywords)
    _assert_not_identified(fit, fit_causal_residual(_known, data, IV_SYSTEM.dt, **keywords))
    assert fit.instrument_rank == 0
    np.testing.assert_array_equal(np.asarray(fit.instrument_relevance), [0.0, 0.0, 0.0])


def _two_lever_log(second: str, rows: int = 2000) -> dict[str, jax.Array]:
    """One state and two levers, confounded by a latent ``z``, and two instruments that move the
    first lever. The second they move as well (``"moved"``), or not at all, its part they would
    explain taken out (``"blind"``), or the log never uses it (``"unused"``)."""
    rng = np.random.default_rng(11)
    x, z = rng.normal(size=(2, rows, 1))
    w = rng.normal(size=(rows, 2))
    first = -1.5 * z + 0.5 * rng.normal(size=(rows, 1)) + w @ np.array([[0.6], [0.4]])
    other = z + 0.5 * rng.normal(size=(rows, 1)) + w @ np.array([[0.3], [-0.5]])
    if second == "blind":
        state = np.column_stack([np.ones(rows), x])
        shifted = w - state @ np.linalg.lstsq(state, w, rcond=None)[0]
        other = other - shifted @ np.linalg.lstsq(shifted, other, rcond=None)[0]
    elif second == "unused":
        other = np.zeros((rows, 1))
    u = np.column_stack([first, other])
    rate = -0.5 * x + u @ np.array([[0.8], [-0.4]]) + 1.5 * z
    return {
        "x": jnp.asarray(x),
        "u": jnp.asarray(u),
        "x_next": jnp.asarray(x + 0.1 * rate + 0.01 * rng.normal(size=(rows, 1))),
        "w": jnp.asarray(w),
    }


def test_two_instruments_that_move_one_lever_leave_the_moment_short_of_rank() -> None:
    """Two channel coefficients and two instruments, both of which move the first lever alone: the
    moment has rank 1. Where they move the second lever as well, it has rank 2, and the fit reads
    both levers' channels."""
    blind = _two_lever_log("blind")
    fit = fit_causal_residual(_known, blind, 0.1, instrument="w", **ONE_FEATURE)
    _assert_not_identified(fit, fit_causal_residual(_known, blind, 0.1, **ONE_FEATURE))
    assert fit.instrument_rank == 1
    relevance = np.asarray(fit.instrument_relevance)
    assert relevance.shape == (2,)
    assert relevance[0] > 0.1
    assert relevance[1] == 0.0

    moved = fit_causal_residual(_known, _two_lever_log("moved"), 0.1, instrument="w", **ONE_FEATURE)
    assert moved.identified
    assert moved.method == "iv"
    assert moved.instrument_rank == 2
    assert moved.channel_error is not None
    np.testing.assert_allclose(np.asarray(moved.residual.channel)[0, :, 0], [0.8, -0.4], atol=0.1)


def test_a_lever_the_log_never_used_leaves_the_rank_to_the_levers_it_did() -> None:
    """The second lever never used: its channel is unmoved, as it is under an adjustment set, and
    the rank counts the directions the log moves, which the instruments move in full."""
    fit = fit_causal_residual(_known, _two_lever_log("unused"), 0.1, instrument="w", **ONE_FEATURE)
    assert fit.unmoved is not None
    assert fit.unmoved.shape[1] == 1
    assert fit.identified
    assert fit.method == "iv"
    assert fit.instrument_rank == 1
    assert np.asarray(fit.instrument_relevance).shape == (1,)


def test_an_instrument_identifies_nothing_where_the_log_moves_no_direction() -> None:
    """A lever the state sets, ``-0.3 x0``: the log moves no direction of its channel, so no
    instrument moves one either. Adjusted, the fit keeps its flag and names every direction
    unmoved (ADR 0054); through an instrument it is not identified, whatever the instrument."""
    log = _policy_log(4000, _ruled(1.0))
    log["w"] = jax.random.normal(jax.random.key(3), (4000, 1))
    keywords = {"nuisance_degree": 2, "channel_degree": 1}
    fit = fit_causal_residual(_known, log, 0.1, instrument="w", **keywords)
    _assert_not_identified(fit, fit_causal_residual(_known, log, 0.1, **keywords))
    assert fit.unmoved is not None
    assert fit.unmoved.shape[1] == fit.residual.channel.size
    assert fit.instrument_rank == 0
    assert np.asarray(fit.instrument_relevance).shape == (0,)
    assert fit_causal_residual(_known, log, 0.1, **RULED).identified


def _weakened_log(gain: float, rows: int = 2000) -> tuple[dict[str, jax.Array], float]:
    """One state and one lever, confounded by a latent ``z``, and an instrument that moves the
    lever by ``gain``, the rest of the lever made blind to it. With the first stage linear and the
    channel constant, the relevance is the lever's partial correlation with the instrument given
    the state, ``gain a / hypot(b, gain a)``, ``a`` and ``b`` the instrument's and the rest's
    norms less their projection on the state; returned beside the log."""
    rng = np.random.default_rng(4)
    x, z, w = rng.normal(size=(3, rows, 1))
    state = np.column_stack([np.ones(rows), x])

    def less_state(columns: np.ndarray) -> np.ndarray:
        return columns - state @ np.linalg.lstsq(state, columns, rcond=None)[0]

    shifted = less_state(w)
    rest = -1.5 * z + 0.5 * rng.normal(size=(rows, 1))
    rest = rest - shifted @ np.linalg.lstsq(shifted, rest, rcond=None)[0]
    u = rest + gain * w
    rate = -0.5 * x + u + 2.0 * z
    data = {
        "x": jnp.asarray(x),
        "u": jnp.asarray(u),
        "x_next": jnp.asarray(x + 0.1 * rate + 0.01 * rng.normal(size=(rows, 1))),
        "w": jnp.asarray(w),
    }
    a, b = np.linalg.norm(shifted), np.linalg.norm(less_state(rest))
    return data, float(gain * a / np.hypot(b, gain * a))


def test_a_weakening_instrument_reads_a_falling_relevance_and_stays_identified() -> None:
    """The relevance falls with the instrument's gain on the lever, as the partial correlation in
    closed form does, and the fit stays identified as long as the gain is not zero."""
    relevances = []
    for gain in (1.0, 0.1, 0.01, 1e-4, 1e-8):
        data, closed = _weakened_log(gain)
        fit = fit_causal_residual(_known, data, 0.1, instrument="w", **ONE_FEATURE)
        assert fit.identified, gain
        assert fit.method == "iv"
        assert fit.instrument_rank == 1
        assert fit.channel_error is not None
        (relevance,) = np.asarray(fit.instrument_relevance)
        # Near a right angle, the push's rounding turns the correlation by its share of the push,
        # so it grows as the gain's inverse square: 5.6e-12 of the value at 1e-4, 3.7e-3 at 1e-8.
        assert relevance == pytest.approx(closed, rel=1e-9 if gain >= 1e-4 else 0.05, abs=0.0)
        relevances.append(relevance)
    assert all(weak < strong for strong, weak in itertools.pairwise(relevances))

    data, closed = _weakened_log(0.0)
    fit = fit_causal_residual(_known, data, 0.1, instrument="w", **ONE_FEATURE)
    assert closed == 0.0
    _assert_not_identified(fit, fit_causal_residual(_known, data, 0.1, **ONE_FEATURE))
    assert fit.instrument_rank == 0


@pytest.mark.parametrize(
    ("integrator", "channel", "errors"),
    [
        (
            "euler",
            [
                [1.0401683819451792, -0.02860024480332746, 0.10613137707962442],
                [0.5189711722550808, -0.016148790583162567, 0.050102200625694565],
            ],
            (0.08806213130166844, 0.06558687121445604),
        ),
        (
            "rk4",
            [
                [1.0508991256123608, -0.029127363934056676, 0.10820857328551325],
                [0.5222761403167667, -0.016414417710067162, 0.05087340795001539],
            ],
            (0.08991264604299758, 0.06674476953031243),
        ),
    ],
)
def test_a_relevant_instrument_fits_as_it_did(
    integrator: str, channel: list[list[float]], errors: tuple[float, float]
) -> None:
    """The positive control: an instrument that moves the lever, as on the reference plant, leaves
    the fit identified, near the truth, and as the fit read it before the rank check, pinned here
    to 1e-9. The ridge read in each column's own units moved these by at most 9e-9 of each."""
    fit = fit_causal_residual(
        _known, _iv_log(), IV_SYSTEM.dt, instrument="w", integrator=integrator
    )
    assert fit.identified
    assert fit.method == "iv"
    assert fit.instrument_rank == 3
    relevance = np.asarray(fit.instrument_relevance)
    assert relevance.shape == (3,)
    assert np.all(np.diff(relevance) <= 0.0)
    assert relevance[-1] > 0.3  # measured 0.36 to 0.46
    assert relevance[0] < 0.6
    assert _error(_channel_of(fit)) < 0.1
    np.testing.assert_allclose(
        np.asarray(fit.residual.channel)[:, 0, :], channel, rtol=1e-9, atol=0.0
    )
    assert (fit.channel_error, fit.drift_error) == pytest.approx(errors, rel=1e-9, abs=0.0)


@pytest.mark.parametrize("log", ["relevant", "short of rank"])
def test_the_instrument_s_rank_and_relevance_read_the_same_in_any_units(log: str) -> None:
    """The instrument, the levers and the state each logged in units a millionth of their own and
    a million times them."""
    data, dt, keywords = (
        (_iv_log(), IV_SYSTEM.dt, {})
        if log == "relevant"
        else (_two_lever_log("blind"), 0.1, ONE_FEATURE)
    )
    base = fit_causal_residual(_known, data, dt, instrument="w", **keywords)
    for column, factor in itertools.product(("w", "u", "x"), (1e-6, 1e6)):
        moved = {**data, column: data[column] * factor}
        if column == "x":
            moved["x_next"] = data["x_next"] * factor
        fit = fit_causal_residual(_known, moved, dt, instrument="w", **keywords)
        assert (fit.identified, fit.instrument_rank) == (base.identified, base.instrument_rank)
        np.testing.assert_allclose(
            np.asarray(fit.instrument_relevance),
            np.asarray(base.instrument_relevance),
            rtol=1e-9,
            atol=0.0,
            err_msg=f"{column} * {factor:g}",
        )


def test_a_weight_that_drops_the_rows_the_instrument_moves_leaves_no_rank() -> None:
    """The rank is the weighted moment's. The state sits at one of two levels and the instrument
    moves the lever at the second alone; a weight of zero there leaves the moment no row the
    instrument moves."""
    rows = 2000
    rng = np.random.default_rng(5)
    x = (rng.random((rows, 1)) < 0.5).astype(float)
    z, e = rng.normal(size=(2, rows, 1))
    w = rng.normal(size=(rows, 1)) * x
    u = -1.5 * z + 0.5 * e + 0.8 * w
    rate = -0.5 * x + u + 2.0 * z
    data = {
        "x": jnp.asarray(x),
        "u": jnp.asarray(u),
        "x_next": jnp.asarray(x + 0.1 * rate + 0.01 * rng.normal(size=(rows, 1))),
        "w": jnp.asarray(w),
    }

    def first_level(states: jax.Array) -> jax.Array:
        return (states[:, 0] == 0.0).astype(states.dtype)

    unweighted = fit_causal_residual(_known, data, 0.1, instrument="w", **ONE_FEATURE)
    assert unweighted.identified
    assert unweighted.instrument_rank == 1
    weighted = fit_causal_residual(
        _known, data, 0.1, instrument="w", weights=first_level, **ONE_FEATURE
    )
    _assert_not_identified(
        weighted, fit_causal_residual(_known, data, 0.1, weights=first_level, **ONE_FEATURE)
    )
    assert weighted.instrument_rank == 0


def _persistent_log(
    noise: float, units: int = 100, periods: int = 30, seed: int = 0
) -> tuple[dict[str, jax.Array], np.ndarray, np.ndarray]:
    """``x' = 0.95 x + 0.1 (0.8 u + 1.5 z) + e``, ``u = a + 0.9 z``, ``a`` AR(0.7) and ``e``
    AR(``noise``) within each unit, ``z`` drawn afresh; rows period-major, with each transition's
    unit and period."""
    rng = np.random.default_rng(seed)

    def ar(rho: float, spread: float) -> np.ndarray:
        out = np.empty((periods, units))
        out[0] = rng.normal(0.0, spread, units)
        for t in range(1, periods):
            out[t] = rho * out[t - 1] + rng.normal(0.0, spread * math.sqrt(1.0 - rho**2), units)
        return out

    z = rng.normal(size=(periods, units))
    u = ar(0.7, 0.5) + 0.9 * z
    e = ar(noise, 0.05)
    x = np.empty((periods + 1, units))
    x[0] = rng.normal(size=units)
    for t in range(periods):
        x[t + 1] = 0.95 * x[t] + 0.1 * (0.8 * u[t] + 1.5 * z[t]) + e[t]
    data = {
        name: jnp.asarray(values.reshape(-1, 1))
        for name, values in (("x", x[:-1]), ("u", u), ("z", z), ("x_next", x[1:]))
    }
    labels = np.tile(np.arange(units), periods)
    steps = np.repeat(np.arange(periods), units)
    return data, labels, steps


def test_the_persistence_check_reads_noise_that_persists_within_units() -> None:
    """With the noise AR(0.7) within each unit, the moment's residual reads its persistence; drawn
    afresh, it reads none. Each of the 100 units pairs 29 transitions."""
    for noise, low, high in ((0.7, 0.6, 0.75), (0.0, -0.1, 0.1)):
        data, units, periods = _persistent_log(noise)
        fit = fit_causal_residual(_known, data, 0.1, adjust_for=("z",), channel_degree=0)
        check = persistence_check(fit, units, periods)
        assert (check.pairs, check.units) == (100 * 29, 100)
        assert low < check.correlation < high, noise
    assert check.p_value > 1e-3
    data, units, periods = _persistent_log(0.7)
    fit = fit_causal_residual(_known, data, 0.1, adjust_for=("z",), channel_degree=0)
    assert persistence_check(fit, units, periods).p_value < 1e-12


def test_the_persistence_check_pairs_a_unit_s_consecutive_periods_by_hand() -> None:
    """Units of any sortable kind, rows in any order, and a gap: ``c``'s periods 1 and 3 make no
    pair. Each state is read in its own units, and the units' sums of products are tested against
    ``t(G - 1)`` with CR1's factor, each sum less its pairs' share of the total."""
    data, _, _ = _persistent_log(0.0, units=4, periods=10)
    fit = fit_causal_residual(_known, data, 0.1, adjust_for=("z",), channel_degree=0)
    rows = 7
    residual = np.random.default_rng(3).normal(size=(rows, 2)) * np.array([1.0, 1e6])
    units = np.array(["b", "a", "b", "a", "b", "c", "c"])
    periods = np.array([3, 1, 1, 2, 2, 1, 3])
    pairs = [(1, 3), (2, 4), (4, 0)]  # (earlier row, later row): a 1-2, b 1-2, b 2-3
    unit = np.array([units[later] for _, later in pairs])
    scaled = residual / np.sqrt(np.mean(residual**2, axis=0))
    products = np.array([scaled[later] @ scaled[earlier] for earlier, later in pairs])
    earlier = np.array([scaled[row] for row, _ in pairs])
    later = np.array([scaled[row] for _, row in pairs])
    correlation = products.sum() / math.sqrt(np.sum(later**2) * np.sum(earlier**2))
    sums = np.array([products[unit == name].sum() for name in ("a", "b")])
    centred = sums - np.array([1.0, 2.0]) * products.sum() / 3.0
    statistic = sums.sum() / math.sqrt(2.0 * np.sum(centred**2))
    p_value = 2.0 * scipy.stats.t.sf(abs(statistic), 1)
    for order in (np.arange(rows), np.random.default_rng(4).permutation(rows)):
        moved = dataclasses.replace(fit, moment_residual=jnp.asarray(residual[order]))
        check = persistence_check(moved, units[order], periods[order])
        assert (check.pairs, check.units) == (3, 2)
        assert check.correlation == pytest.approx(correlation, rel=1e-12, abs=0.0)
        assert check.p_value == pytest.approx(p_value, rel=1e-10, abs=0.0)


def test_the_persistence_check_on_units_alike_is_student_s_one_sample_t() -> None:
    """Where every unit pairs alike, the statistic is Student's one-sample t on the units' sums,
    exact against ``t(G - 1)`` where those sums are normal."""
    data, units, periods = _persistent_log(0.3, units=5, periods=12)
    fit = fit_causal_residual(_known, data, 0.1, adjust_for=("z",), channel_degree=0)
    residual = np.asarray(fit.moment_residual)
    scaled = (residual / np.sqrt(np.mean(residual**2, axis=0))).reshape(12, 5, -1)
    sums = np.einsum("tgn,tgn->g", scaled[1:], scaled[:-1])
    expected = float(scipy.stats.ttest_1samp(sums, 0.0).pvalue)
    check = persistence_check(fit, units, periods)
    assert (check.pairs, check.units) == (55, 5)
    assert check.p_value == pytest.approx(expected, rel=1e-10, abs=0.0)


def test_the_persistence_check_reads_each_state_in_its_own_units() -> None:
    data, units, periods = _persistent_log(0.3, units=30, periods=12)
    fit = fit_causal_residual(_known, data, 0.1, adjust_for=("z",), channel_degree=0)
    residual = np.asarray(fit.moment_residual)
    two = np.column_stack([residual[:, 0], np.roll(residual[:, 0], 7)])
    one = persistence_check(
        dataclasses.replace(fit, moment_residual=jnp.asarray(two)), units, periods
    )
    for scale in (1e-9, 1e9):
        moved = dataclasses.replace(fit, moment_residual=jnp.asarray(two * np.array([1.0, scale])))
        other = persistence_check(moved, units, periods)
        assert other.correlation == pytest.approx(one.correlation, rel=1e-12, abs=0.0)
        assert other.p_value == pytest.approx(one.p_value, rel=1e-10, abs=0.0)


def test_the_persistence_check_on_one_unit_or_no_pair_reads_no_test() -> None:
    data, _, _ = _persistent_log(0.0, units=4, periods=10)
    fit = fit_causal_residual(_known, data, 0.1, adjust_for=("z",), channel_degree=0)
    rows = np.asarray(fit.moment_residual).shape[0]
    one = persistence_check(fit, np.zeros(rows, dtype=int), np.arange(rows))
    assert (one.pairs, one.units) == (rows - 1, 1)
    assert math.isfinite(one.correlation)
    assert math.isnan(one.p_value)
    none = persistence_check(fit, np.arange(rows), np.zeros(rows, dtype=int))
    assert (none.pairs, none.units) == (0, 0)
    assert math.isnan(none.correlation)
    assert math.isnan(none.p_value)


def test_the_persistence_check_refuses_what_it_cannot_pair() -> None:
    data, units, periods = _persistent_log(0.0, units=4, periods=10)
    fit = fit_causal_residual(_known, data, 0.1, adjust_for=("z",), channel_degree=0)
    with pytest.raises(ValueError, match="not identified"):
        persistence_check(dataclasses.replace(fit, moment_residual=None), units, periods)
    with pytest.raises(ValueError, match="one each"):
        persistence_check(fit, units[1:], periods)
    with pytest.raises(ValueError, match="whole numbers"):
        persistence_check(fit, units, periods.astype(float))
    with pytest.raises(ValueError, match="two transitions starting in one period"):
        persistence_check(fit, units, np.zeros_like(periods))


def test_an_identified_fit_keeps_its_moment_residual_without_its_influence() -> None:
    data, _, _ = _persistent_log(0.0, units=10, periods=10)
    options = {"adjust_for": ("z",), "channel_degree": 0}
    for integrator in ("euler", "rk4"):
        lean = fit_causal_residual(_known, data, 0.1, **options, integrator=integrator)
        full = fit_causal_residual(
            _known, data, 0.1, **options, integrator=integrator, influence=True
        )
        assert lean.influence is None
        assert lean.moment_residual is not None
        np.testing.assert_array_equal(lean.moment_residual, full.moment_residual)
    blind = fit_causal_residual(_known, data, 0.1, channel_degree=0)
    assert not blind.identified
    assert blind.moment_residual is None
