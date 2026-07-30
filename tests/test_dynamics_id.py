"""Causal identification of a residual's control channel: earn the claim, do not assert it.

Every test here is two-sided. "The orthogonal fit recovers B" passes vacuously if the plant is not
actually confounded, so each recovery test also pins down what the *un*-adjusted fit does on the
same rows.
"""

import jax
import jax.numpy as jnp
import numpy as np

from chc.control import projected_gradient_control
from chc.cost import QuadraticCost, total_cost
from chc.dynamics import HybridDynamics
from chc.dynamics_id import (
    CausalDynamicsFit,
    ConfoundedControlAffineSystem,
    fit_causal_residual,
    solve_channel_moment,
)
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
    """The constant part of ``B_θ``, which at ``degree=1`` is the whole of it."""
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
    assert fit.action_residual_variance > 0.0  # diagnostics still populated for comparison


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


def test_the_reported_channel_error_is_calibrated_on_both_identified_paths() -> None:
    """The standard error has to track the spread it claims to describe.

    This is the test that catches the defect it was written for: reporting the ordinary
    least-squares SE of a regression on the *projected* action, with sigma^2 taken from that same
    regression's residual, understated the IV path by ~5x -- worse than reporting nothing, because
    ``chc.sensitivity`` consumes exactly this number as a radius. The 2SLS sandwich with sigma^2
    from the structural residual brings both paths into band.
    """
    system = _system(instrument_to_action=jnp.array([[0.8]]))
    for keywords in ({"adjust_for": ("z",)}, {"instrument": "w"}):
        deviations, reported = [], []
        for seed in range(24):
            fit = fit_causal_residual(
                _known, system.sample(2000, jax.random.key(seed), _known), system.dt, **keywords
            )
            deviations.extend(
                abs(np.asarray(_channel_of(fit)).ravel() - np.asarray(CHANNEL).ravel())
            )
            reported.append(fit.channel_error)
        ratio = float(np.sqrt(np.mean(np.square(deviations))) / np.mean(reported))
        assert 0.5 < ratio < 2.0, f"{keywords} SE off by {ratio:.2f}x"


def test_the_residual_channel_is_the_jacobian_the_safety_layer_reads() -> None:
    """``control_channel`` and ``d r / d u`` must be the same object.

    :func:`chc.plan.certify_safety` recovers the channel by differentiating at ``u = 0``; if the
    estimated parameter and that derivative could disagree, identification and certification would
    be talking about different plants.
    """
    residual = ControlAffineResidual(
        drift=jax.random.normal(jax.random.key(1), (2, 6)),
        channel=jax.random.normal(jax.random.key(2), (2, 1, 6)),
        degree=2,
    )
    x, u = jnp.array([0.3, -0.7]), jnp.array([1.2])
    jacobian = jax.jacobian(lambda action: residual(0.0, x, action))(u)

    assert jnp.allclose(jacobian, residual.control_channel(x))
