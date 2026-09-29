"""Causal identification of a residual's control channel: earn the claim, do not assert it.

Every test here is two-sided. "The orthogonal fit recovers B" passes vacuously if the plant is not
actually confounded, so each recovery test also pins down what the *un*-adjusted fit does on the
same rows.
"""

import functools
import itertools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chc.control import projected_gradient_control
from chc.cost import QuadraticCost, total_cost
from chc.dynamics import HybridDynamics, LinearDynamics
from chc.dynamics_id import (
    CausalDynamicsFit,
    ConfoundedControlAffineSystem,
    fit_causal_residual,
    solve_channel_moment,
)
from chc.integrate import rk4_step
from chc.residual import ControlAffineResidual
from chc.train import fit_residual

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
    assert unadjusted_regret > 1.0  # measured 6.41 against a 6.10 oracle cost
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
    """
    system = _system(instrument_to_action=jnp.array([[0.8]]))
    for keywords in ({"adjust_for": ("z",)}, {"instrument": "w"}):
        deviations, reported = [], []
        for seed in range(24):
            fit = fit_causal_residual(
                _known,
                system.sample(2000, jax.random.key(seed), _known),
                system.dt,
                channel_degree=channel_degree,
                **keywords,
            )
            deviations.extend(
                abs(np.asarray(_channel_of(fit)).ravel() - np.asarray(CHANNEL).ravel())
            )
            reported.append(fit.channel_error)
        ratio = float(np.sqrt(np.mean(np.square(deviations))) / np.mean(reported))
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
            channels.append(np.asarray(fit.residual.channel).ravel())
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
def _replicated_decision_fits() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """A hundred fresh logs of 4000 rows from a class that misses the truth, fitted under the
    decision weight, and the first forty under ones as well: the lines, and the errors the fits
    reported. A regret is a quadratic form in the line's error, so its mean over forty logs moves
    by a fifth; the hundred are for that."""
    weights = {"ones": lambda states: jnp.ones(states.shape[0]), "decision": _decision_weight}
    lines: dict[str, list[np.ndarray]] = {name: [] for name in weights}
    errors: dict[str, list[float]] = {name: [] for name in weights}
    for seed in range(100):
        data = _decision_log(4000, 0.4, seed=100 + seed)
        for name, weight in weights.items():
            if name == "ones" and seed >= 40:
                continue
            fit = _decision_fit(data, weight)
            assert fit.channel_error is not None
            lines[name].append(_line(fit))
            errors[name].append(fit.channel_error)
    return {name: (np.array(lines[name]), np.array(errors[name])) for name in weights}


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
    differ. Over fresh logs of a class that misses the truth, where the score's variance moves
    with the state, the robust error weighted by ones came to 1.02 of its spread, and the decision
    weight's to 0.96. The homoskedastic error the unweighted fit reported up to 0.7.0 came to
    0.40."""
    for name, (lines, errors) in _replicated_decision_fits().items():
        spread = float(np.sqrt(np.mean(np.var(lines, axis=0, ddof=1))))
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
    lines, _ = _replicated_decision_fits()["decision"]
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
    what the fit's own linear map says, whichever end of the log the weight loads.

    Weighed by ``exp(-x)``, towards the quiet rows, the error came to 1.04 and 1.02 of the spread
    under the two integrators over sixteen redraws; a sandwich on the moment alone leaves out what
    the cross-fitted nuisances pass on from the rows the weight loads, and came to 0.67. Weighed by
    ``exp(x)``, towards the noisy rows, it came to 0.93 and 0.85; an error that pools the noise
    over the rows came to 0.16 under ``rk4``, and agrees with the robust one on average under
    ``exp(-x)``, so only the noisy rows can tell them apart."""
    dt = 1.0
    base = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    for integrator in ("euler", "rk4"):
        channels, errors = [], []
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
            channels.append(np.asarray(fit.residual.channel).ravel())
            errors.append(fit.channel_error)
        spread = float(np.sqrt(np.mean(np.var(np.array(channels), axis=0, ddof=1))))
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


def test_a_policy_the_covariates_determine_reads_as_a_confident_wrong_channel() -> None:
    """Why the field exists: with nothing left of the action once the covariates are taken out, the
    ridge sets the channel, near zero where the truth is 0.8, and its error says 0.0013. Only
    ``unmoved`` tells the caller that every direction of the channel is unread."""
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
