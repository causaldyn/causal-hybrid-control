"""Causal identification of the residual's *control channel*, not just of a scalar effect.

``chc.train`` fits a residual by prediction error. Under a confounded logging policy that is the
wrong object: if the historical action was chosen from a covariate that also drives the state,
then ``d(dx/dt)/du`` fitted on ``(x, u, x_next)`` is the **observational** response, and a
controller that optimises against it moves the state the wrong way. That is the same failure the
scalar headline benchmark shows -- and until now nothing protected the residual from it.

This module closes that gap with Robinson partialling-out lifted from a scalar effect to a
state-dependent matrix. With nuisances ``g(x,z) = E[y | x,z]`` and ``m(x,z) = E[u | x,z]``, where
``y = (x_next - x)/dt - f_known(t, x, u)`` is the part the residual must explain::

    y - g(x,z)  =  B_θ(x) (u - m(x,z))  +  eps,     E[eps | x, z, u - m] = 0

so the estimator regresses the **residualised state rate on the residualised action**, with
K-fold cross-fitting of both nuisances. Neyman orthogonality is what buys the guarantee: the
channel error is *second* order in nuisance error, which is exactly Results 18 (order transfer
``p -> 2p``) and 19 (debias every channel via cross-fit Robinson DML). This module is those
results' missing consumer, not a new one.

HONEST SCOPE, three limits worth stating before the code:

* Only the **channel** ``B_θ`` is interventional. ``a_θ`` is fitted on the remainder and therefore
  absorbs whatever the omitted confounder contributes to the drift in-sample, so it is an
  *observational-conditional* drift. Planning is unbiased in the direction the optimiser moves;
  the predicted trajectory *level* still shifts if the confounder's distribution does.
* The residual must be control-affine (:class:`chc.residual.ControlAffineResidual`). A general
  ``r_θ(x, u)`` has no partialling-out moment and gets no guarantee here.
* With no adjustment set and no instrument nothing in the log identifies the channel. The
  estimator reports ``identified=False`` rather than returning a confident wrong answer -- that
  case belongs to :mod:`chc.sensitivity`, which prices the radius instead of pretending it away.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from chc.causal import _polynomial_features, _ridge_predict
from chc.dynamics import Dynamics
from chc.residual import ControlAffineResidual, control_affine_features

_NOT_IDENTIFIED = (
    "no adjustment set and no instrument: the control channel is not identified from this log. "
    "Use chc.sensitivity to price the identification radius instead of trusting the estimate."
)


@dataclass(frozen=True)
class ConfoundedControlAffineSystem:
    """Euler-logged ``dx/dt = f_known(x,u) + A x + B_true u + C z`` under ``u = K z + eta``.

    The vector-state analogue of :class:`chc.causal.ConfoundedLinearSystem` and the DGP
    :func:`fit_causal_residual` is scored against. ``z`` is the confounder: it drives the logged
    action through ``K`` *and* the state rate through ``C``, so with ``z`` withheld the response of
    the rate to the action is ``B_true + C sigma_z^2 K^T / var(u)`` rather than ``B_true``, where
    ``var(u) = K K^T sigma_z^2 + G G^T + sigma_eta^2`` collects every source of action variance.

    A scalar channel with ``B=1, C=2, K=-1.5, sigma_z=1, sigma_eta=0.5`` and **no instrument**
    gives ``1 - 2*1.5/2.5 = -0.2``: the sign flip of the headline benchmark, reproduced per
    channel. Turning the instrument on adds ``G G^T`` to that denominator and *dilutes* the bias
    toward zero -- the exogenous shifter makes the log more informative, so the observational fit
    looks less wrong while being no more identified. Worth knowing before reading a number off it.

    The Euler discretisation is deliberately the one :func:`fit_causal_residual` inverts, so a
    failing recovery test means the identification is wrong, not that the integrator disagreed.
    """

    drift: Array  # (n, n) state -> rate
    channel: Array  # (n, m) causal control channel, the estimand
    confounder_to_rate: Array  # (n, d)
    confounder_to_action: Array  # (m, d) the logging policy
    instrument_to_action: Array | None = None  # (m, d) exogenous shifter, None = no instrument
    dt: float = 0.05
    z_scale: float = 1.0
    eta_scale: float = 0.5
    noise_scale: float = 0.01

    def sample(self, n: int, key: Array, known: Dynamics) -> dict[str, Array]:
        """Draw ``n`` transitions: ``x (n,d_x)``, ``z``, ``u``, ``x_next``, ``w`` (instrument)."""
        state_dim, control_dim = self.channel.shape
        conf_dim = self.confounder_to_rate.shape[1]
        k_x, k_z, k_eta, k_w, k_noise = jax.random.split(key, 5)
        x = jax.random.normal(k_x, (n, state_dim))
        z = self.z_scale * jax.random.normal(k_z, (n, conf_dim))
        eta = self.eta_scale * jax.random.normal(k_eta, (n, control_dim))
        w = jax.random.normal(k_w, (n, conf_dim))
        u = z @ self.confounder_to_action.T + eta
        if self.instrument_to_action is not None:
            u = u + w @ self.instrument_to_action.T
        rate = (
            jax.vmap(lambda xi, ui: known(0.0, xi, ui))(x, u)
            + x @ self.drift.T
            + u @ self.channel.T
            + z @ self.confounder_to_rate.T
        )
        noise = self.noise_scale * jax.random.normal(k_noise, (n, state_dim))
        return {"x": x, "z": z, "u": u, "w": w, "x_next": x + self.dt * rate + noise}


@dataclass(frozen=True)
class CausalDynamicsFit:
    """A fitted causal residual next to what is and is not known about it.

    ``identified`` is the load-bearing field: ``False`` means the log cannot pin the channel down
    at all, and the residual is then the observational fit, kept only so the caller can compare.
    """

    residual: ControlAffineResidual
    identified: bool
    method: str  # "orthogonal" | "iv" | "observational"
    folds: int
    # root-mean diagonal of the 2SLS sandwich, None when not identified. Homoskedastic, so on the
    # ``iv`` path with a weak first stage it runs optimistic -- measured ~1.25x at N=4000 with a
    # first-stage R^2 of 0.18. A test pins the calibration band; do not read it as exact.
    channel_error: float | None
    action_residual_variance: (
        float  # overlap proxy: 0 => a deterministic policy, nothing to regress
    )
    nuisance_r2_state: float
    nuisance_r2_action: float
    moment_norm: float  # ||mean(design * residual)|| at the solution; should be ~0


def _r_squared(target: Array, prediction: Array) -> float:
    centred = target - jnp.mean(target, axis=0, keepdims=True)
    total = float(jnp.sum(centred**2))
    if total == 0.0:
        return 0.0
    return float(1.0 - jnp.sum((target - prediction) ** 2) / total)


def _cross_fit_residuals(
    target: Array,
    action: Array,
    covariates: Array,
    *,
    degree: int,
    folds: int,
    ridge: float,
    seed: int,
) -> tuple[Array, Array, Array, Array]:
    """Vector-valued cross-fitted partialling-out; the matrix form of ``_dml_residuals``.

    Kept separate rather than generalising that function: it has three callers on a scalar
    contract, and :func:`chc.causal._ridge_predict` already accepts a matrix target, so the only
    thing this adds is the fold bookkeeping over two dimensions.
    """
    n = target.shape[0]
    chunks = jnp.array_split(jax.random.permutation(jax.random.key(seed), n), folds)
    target_hat = jnp.zeros_like(target)
    action_hat = jnp.zeros_like(action)
    for k in range(folds):
        test = chunks[k]
        train = (
            jnp.concatenate([chunks[j] for j in range(folds) if j != k]) if folds > 1 else chunks[0]
        )
        phi_train = _polynomial_features(covariates[train], degree)
        phi_test = _polynomial_features(covariates[test], degree)
        target_hat = target_hat.at[test].set(
            _ridge_predict(phi_train, target[train], phi_test, ridge)
        )
        action_hat = action_hat.at[test].set(
            _ridge_predict(phi_train, action[train], phi_test, ridge)
        )
    return target - target_hat, action - action_hat, target_hat, action_hat


def _solve_ridge(design: Array, target: Array, ridge: float) -> Array:
    gram = design.T @ design + ridge * jnp.eye(design.shape[1])
    return jnp.linalg.solve(gram, design.T @ target)


def _channel_design(action_residual: Array, states: Array, degree: int) -> Array:
    """Row ``i`` is ``vec(u_res_i (x) phi(x_i))``, so a coefficient block ``C[j, k, l]`` means
    ``B_θ(x)[j, k] = sum_l C[j, k, l] phi_l(x)``."""
    phi = jax.vmap(control_affine_features, in_axes=(0, None))(states, degree)
    n, control_dim = action_residual.shape
    return (action_residual[:, :, None] * phi[:, None, :]).reshape(n, control_dim * phi.shape[1])


def _channel_coefficients(
    state_residual: Array,
    regressor: Array,
    instrument: Array,
    ridge: float,
) -> Array:
    """Solve the just-identified moment ``Z'(y_res - D c) = 0`` for ``c``, ridge-stabilised.

    ``Z is D`` reduces to ordinary least squares; a different ``Z`` is two-stage least squares.
    Writing it as one solve rather than "regress on the projection" matters: the two agree only for
    a scalar action with no state features, because the in-sample identity ``Z'u = Z'Z`` does not
    survive multiplication by ``phi(x)``.
    """
    gram = instrument.T @ regressor + ridge * jnp.eye(regressor.shape[1])
    return jnp.linalg.solve(gram, instrument.T @ state_residual)


def _sandwich_error(
    state_residual: Array,
    regressor: Array,
    instrument: Array,
    coeffs: Array,
    ridge: float,
) -> float:
    """Root-mean diagonal of ``sigma^2 (Z'D)^-1 (Z'Z) (D'Z)^-1`` -- the 2SLS sandwich.

    ``sigma^2`` comes from the **structural** residual ``y_res - D c``, i.e. against the actual
    action rather than against the instrument. Using the instrument's own fitted residual instead
    understates it badly (measured ~5x on the IV path) because the projection has already dropped
    the endogenous variation the structural error is made of.
    """
    n, n_coeff = regressor.shape
    score = state_residual - regressor @ coeffs
    sigma2 = jnp.sum(score**2) / (max(n - n_coeff, 1) * state_residual.shape[1])
    bread = jnp.linalg.inv(instrument.T @ regressor + ridge * jnp.eye(n_coeff))
    covariance = sigma2 * bread @ (instrument.T @ instrument) @ bread.T
    return float(jnp.sqrt(jnp.mean(jnp.diag(covariance))))


def solve_channel_moment(
    state_residual: Array,
    action_residual: Array,
    states: Array,
    *,
    instrument_action: Array | None = None,
    degree: int = 1,
    ridge: float = 1e-6,
) -> Array:
    """Solve ``E[(y_res - B_θ(x) u_res) (x) (z (x) phi(x))] = 0`` for the channel.

    The Robinson moment on its own, separated from the nuisance estimation
    :func:`fit_causal_residual` wraps around it. Public for two reasons: cross-fitted
    ridge-polynomial nuisances are a default rather than a commitment, and the point of a debiased
    score is that ``g`` and ``m`` may come from any learner -- gradient boosting, a neural net, a
    model the caller already had -- so the caller needs a way in that does not go through ours.

    Args:
        state_residual: ``y - g(x, z)``, shape ``(N, n)``.
        action_residual: ``u - m(x, z)``, shape ``(N, m)`` -- the regressor.
        states: ``x``, shape ``(N, n)`` -- the channel's own feature argument, which is *not*
            residualised: ``B_θ`` is allowed to depend on the state.
        instrument_action: the action-shaped variable ``z`` that enters the moment, when it differs
            from the regressor: the projection of the action on an exogenous shifter, for the case
            where the confounder is latent. ``None`` means the action residual instruments itself,
            which is the orthogonal (adjusted) case.
        degree: monomial degree of ``B_θ``'s dependence on the state.
        ridge: Tikhonov term on the moment's Gram matrix.

    Returns:
        The channel coefficients, shape ``(n, m, n_features)``, consumable directly as
        :attr:`chc.residual.ControlAffineResidual.channel`.
    """
    regressor = _channel_design(action_residual, states, degree)
    moment = (
        regressor
        if instrument_action is None
        else _channel_design(instrument_action, states, degree)
    )
    coeffs = _channel_coefficients(state_residual, regressor, moment, ridge)
    n_features = regressor.shape[1] // action_residual.shape[1]
    return coeffs.T.reshape(state_residual.shape[1], action_residual.shape[1], n_features)


def fit_causal_residual(
    known: Dynamics,
    data: dict[str, Array],
    dt: float,
    *,
    adjust_for: tuple[str, ...] = (),
    instrument: str | None = None,
    degree: int = 1,
    nuisance_degree: int = 2,
    folds: int = 2,
    ridge: float = 1e-6,
    seed: int = 0,
) -> CausalDynamicsFit:
    """Fit a :class:`ControlAffineResidual` whose channel is the *interventional* control response.

    Args:
        known: the physics kept fixed; its own control dependence is known, not estimated.
        data: columns ``x (N,n)``, ``u (N,m)``, ``x_next (N,n)``, plus any named in ``adjust_for``
            and ``instrument``.
        adjust_for: observed confounders. Empty *and* no instrument => ``identified=False``.
        instrument: name of an exogenous action shifter, for when the confounder is latent. It
            enters as the moment's instrument, not as the regressor, so this is real 2SLS. Expect a
            **variance premium**, not a free lunch: identification here rides on however much of
            the action the shifter explains, and on the reference DGP that is 18%, which costs
            roughly an order of magnitude in channel error against adjusting for a logged
            confounder (0.10 vs 0.002 at ``N=4000``). Still ~10x better than not identifying at all.
        degree: channel/drift feature degree. ``1`` keeps the constant channel that §18/§19 cover.
        nuisance_degree: flexibility of ``g`` and ``m``. Richer nuisances are the whole point of
            cross-fitting -- orthogonality is what makes their error enter only at second order.
        folds: cross-fitting folds. ``1`` fits the nuisances on the same rows it residualises.
            That is *not* biased for the default nuisances and should not be sold as such:
            residualising ``y`` and ``u`` by the same linear projection whose span contains the
            truth is Frisch-Waugh-Lovell, so own-sample partialling-out is exactly unbiased and
            out-of-fold prediction only adds variance -- measurably worse at small ``N``.
            Cross-fitting earns its keep against learners whose fit is adaptive to the sample
            (feature selection, trees, early stopping) or saturated enough to memorise it, where
            own-sample residuals collapse; ``folds>=2`` is the safe default for that reason.

    Returns:
        A :class:`CausalDynamicsFit`. Read ``identified`` before ``residual``.
    """
    x, u, x_next = data["x"], data["u"], data["x_next"]
    rate = (x_next - x) / dt
    y = rate - jax.vmap(lambda xi, ui: known(0.0, xi, ui))(x, u)

    identified = bool(adjust_for) or instrument is not None
    covariates = jnp.concatenate([x, *[data[name] for name in adjust_for]], axis=1)
    y_res, u_res, y_hat, u_hat = _cross_fit_residuals(
        y, u, covariates, degree=nuisance_degree, folds=folds, ridge=ridge, seed=seed
    )

    method = "orthogonal" if adjust_for else ("iv" if instrument else "observational")
    instrument_action: Array | None = None
    if instrument is not None:
        # 2SLS lifted to the matrix: the part of the action the exogenous shifter explains becomes
        # the moment's instrument -- not the regressor -- so the confounder's contribution drops out
        # of E[z (x) eps] even though z is never observed. u_res stays the regressor.
        w = data[instrument]
        design_w = _polynomial_features(jnp.concatenate([x, w], axis=1), nuisance_degree)
        projected = design_w @ _solve_ridge(design_w, u, ridge)
        instrument_action = projected - jnp.mean(projected, axis=0, keepdims=True)
        method = "iv"

    channel = solve_channel_moment(
        y_res, u_res, x, instrument_action=instrument_action, degree=degree, ridge=ridge
    )
    regressor = _channel_design(u_res, x, degree)
    moment = (
        regressor if instrument_action is None else _channel_design(instrument_action, x, degree)
    )
    coeffs = channel.reshape(x.shape[1], -1).T

    # a_θ mops up the rest of y at the fitted channel; see this module's scope note on why that
    # makes the drift observational-conditional while the channel stays interventional.
    phi_x = jax.vmap(control_affine_features, in_axes=(0, None))(x, degree)
    fitted = jax.vmap(lambda c, ui: (channel @ c) @ ui)(phi_x, u)
    drift = _solve_ridge(phi_x, y - fitted, ridge).T
    residual = ControlAffineResidual(drift=drift, channel=channel, degree=degree)

    score = y_res - regressor @ coeffs
    error = _sandwich_error(y_res, regressor, moment, coeffs, ridge) if identified else None

    return CausalDynamicsFit(
        residual=residual,
        identified=identified,
        method=method,
        folds=folds,
        channel_error=error,
        action_residual_variance=float(jnp.mean(u_res**2)),
        nuisance_r2_state=_r_squared(y, y_hat),
        nuisance_r2_action=_r_squared(u, u_hat),
        moment_norm=float(jnp.linalg.norm(moment.T @ score / moment.shape[0])),
    )
