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
  the predicted trajectory *level* still shifts if the confounder's distribution does. On real
  data this bites harder than it sounds: a confounder that trends *with* the state gets charged to
  positive feedback in ``x``, so the fitted drift can come back **unstable** even where the channel
  is clean -- and an MPC uses the drift too. Check the drift's spectrum, and give measured exogenous
  drivers a place in the drift as regressors instead of leaving them only in ``adjust_for``.
* The residual must be control-affine (:class:`chc.residual.ControlAffineResidual`). A general
  ``r_θ(x, u)`` has no partialling-out moment and gets no guarantee here.
* With no adjustment set and no instrument nothing in the log identifies the channel. The
  estimator reports ``identified=False`` rather than returning a confident wrong answer -- that
  case belongs to :mod:`chc.sensitivity`, which prices the radius instead of pretending it away.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
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
    # Root-mean diagonal of the drift stage's own homoskedastic OLS covariance, or None when the
    # channel is not identified (the drift is then conditional on a meaningless channel). It is a
    # DIFFERENT object from ``channel_error``: conditional on the fitted channel, whose uncertainty
    # it does not propagate, and homoskedastic -- a scale, not coverage. Reported because on a real
    # plant the drift, not the channel, dominated closed-loop cost.
    drift_error: float | None
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


def _standardised(covariates: Array) -> Array:
    """Centre and scale each covariate, so the nuisance basis is conditioned in any caller's units.

    Without this the polynomial basis inherits whatever units the caller happened to use, and the
    ridge penalises each monomial accordingly -- ``ridge=1e-6`` means something entirely different
    against a column of order 1 than against its square of order 400. Measured on a 20-day log from
    a real building emulator (``causaldyn-bench`` Track D-causal), where the zone enters in Celsius
    at ~21 and the weather columns are already standardised: the degree-2 Gram came out at condition
    number **1.4e11**, past what float32 can carry, and the fit returned ``nan``. The same rows in
    float64 fitted fine, which is what identified this as conditioning rather than data. Scaling
    first drops it to order 10 -- on a reproducible stand-in with that geometry (768 rows, zone at
    21 +- 0.9 beside three standardised columns) the degree-2 Gram goes from ``2.7e10`` to
    ``2.4e1`` -- and removes the precision dependence.

    Full-sample statistics rather than per-fold ones. Centring and scaling the *inputs* is a linear
    reparametrisation, so the span of the fitted basis -- and hence the partialled-out residual --
    is unchanged by it; only the ridge's meaning moves, and pinning that to the data's own scale is
    the point. Per-fold statistics would make the penalty fold-dependent for no gain.

    A zero-variance column keeps scale 1, because dividing a constant column by its own zero spread
    is how a conditioning fix becomes a ``nan`` of its own.
    """
    centre = jnp.mean(covariates, axis=0, keepdims=True)
    spread = jnp.std(covariates, axis=0, keepdims=True)
    return (covariates - centre) / jnp.where(spread > 1e-12, spread, 1.0)


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
    thing this adds is the fold bookkeeping over two dimensions -- and the standardisation below.
    """
    covariates = _standardised(covariates)
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


def _ols_error(target: Array, design: Array, coeffs: Array, ridge: float) -> float:
    """Root-mean diagonal of ``sigma^2 (X'X)^-1`` -- the plain homoskedastic OLS covariance.

    Used for the drift stage, and deliberately not the sandwich :func:`_sandwich_error` computes:
    the drift is fitted by least squares on the remainder, so there is no instrument and no
    endogenous regressor to sandwich against. What it does share is the caveat that it is a scale
    rather than a coverage statement.
    """
    n, n_coeff = design.shape
    score = target - design @ coeffs
    sigma2 = jnp.sum(score**2) / (max(n - n_coeff, 1) * target.shape[1])
    covariance = sigma2 * jnp.linalg.inv(design.T @ design + ridge * jnp.eye(n_coeff))
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
    drift_scale = _ols_error(y - fitted, phi_x, drift.T, ridge) if identified else None

    return CausalDynamicsFit(
        residual=residual,
        identified=identified,
        method=method,
        folds=folds,
        channel_error=error,
        drift_error=drift_scale,
        action_residual_variance=float(jnp.mean(u_res**2)),
        nuisance_r2_state=_r_squared(y, y_hat),
        nuisance_r2_action=_r_squared(u, u_hat),
        moment_norm=float(jnp.linalg.norm(moment.T @ score / moment.shape[0])),
    )


# --- Result 41 (A7): what a tracked log identifies, and what it does not ---


@dataclass(frozen=True)
class ClosedLoopAttribution:
    """Which coefficients of a control-affine fit the log identifies, on a tracked plant."""

    manifold_slope: float  # m, from regressing the state on the action
    manifold_r2: float  # 1 means exact tracking: the log lies on an affine manifold
    implied_gain: float  # -1/m, the proportional gain of the loop that produced the log
    action_curvature: float  # C, the u^2 coefficient of the response along the manifold
    predicted_interaction: float  # C/m = -gain*C, the identity
    fitted_interaction: float  # b1 from the four-term least squares
    fitted_drift: float  # a from the same fit -- reported so its instability is visible
    design_condition: float  # cond of the standardised design; large means d, a, b0 are not split
    exploration_budget: float  # variance off the manifold, as a share of the state's variance


def closed_loop_gain_attribution(
    states: Array, actions: Array, rates: Array
) -> ClosedLoopAttribution:
    """Attribute a control-affine fit's coefficients to the controller and to the plant.

    Result 41 left one item open: *why* the interaction coefficient ``b1`` of
    ``dx/dt = d + a*x + (b0 + b1*x)*u`` comes out large and negative on a setpoint-tracked zone.
    The answer is a property of the log, not of the plant. A proportional loop
    ``u = gain*(setpoint - x) + u0`` puts every sample on an affine manifold ``x = c + m*u`` with
    ``m = -1/gain``, and restricted to that manifold the four-term class collapses to a quadratic in
    the action::

        d + a*x + (b0 + b1*x)*u  =  (d + a*c) + (a*m + b0 + b1*c)*u + (b1*m)*u^2

    Only ``b1`` reaches the ``u^2`` term, so **``b1`` is the identified coefficient and the pole is
    not** -- the inverse of the usual reading, and the reason Result 41 (a) found the reported pole
    to be a units artefact while ``lambda(u) = a + b1*u`` held. Matching against an observed
    response ``A + B*u + C*u^2`` gives

        ``b1 = C/m = -gain*C``,

    so the sign is decided by the curvature of the response in the action and the **magnitude by the
    controller**: a tighter tracker reports a bigger interaction from identical physics.
    ``proofs/closed_loop_attribution.v`` proves the matching identity and exhibits the explicit
    one-parameter family that leaves ``d``, ``a`` and ``b0`` free.

    This also **refutes** the standing guess that ``b1 = -1/gain``. That is the *manifold slope*,
    not the interaction; equating them forces ``C = 1/gain^2``, a constraint on the plant rather
    than an identity (``guess_is_a_constraint_not_an_identity``).

    ``exploration_budget`` is what buys the rest back: variation off the manifold restores the
    design's rank, and with it the separation of ``a`` from ``b0``. Reading a pole off a fit whose
    budget is ~0 is reading the regulariser.
    """
    x = jnp.asarray(states, dtype=jnp.float64).ravel()
    u = jnp.asarray(actions, dtype=jnp.float64).ravel()
    y = jnp.asarray(rates, dtype=jnp.float64).ravel()

    affine = jnp.stack([jnp.ones_like(u), u], axis=1)
    manifold = jnp.linalg.lstsq(affine, x, rcond=None)[0]
    slope = float(manifold[1])
    residual = x - affine @ manifold

    quadratic = jnp.stack([jnp.ones_like(u), u, u**2], axis=1)
    response = jnp.linalg.lstsq(quadratic, y, rcond=None)[0]
    curvature = float(response[2])

    design = jnp.stack([jnp.ones_like(x), x, u, x * u], axis=1)
    coefficients = jnp.linalg.lstsq(design, y, rcond=None)[0]
    scale = jnp.linalg.norm(design, axis=0)
    condition = float(jnp.linalg.cond(design / jnp.where(scale > 0.0, scale, 1.0)))

    state_variance = float(jnp.var(x))
    return ClosedLoopAttribution(
        manifold_slope=slope,
        manifold_r2=_r_squared(x, affine @ manifold),
        # A loop with zero gain leaves the state unexplained by the action; reporting an infinite
        # gain is the honest answer, not a clamped one.
        implied_gain=float("inf") if slope == 0.0 else -1.0 / slope,
        action_curvature=curvature,
        predicted_interaction=float("nan") if slope == 0.0 else curvature / slope,
        fitted_interaction=float(coefficients[3]),
        fitted_drift=float(coefficients[1]),
        design_condition=condition,
        exploration_budget=(
            0.0 if state_variance == 0.0 else float(jnp.var(residual)) / state_variance
        ),
    )


@dataclass(frozen=True)
class ClosedLoopAttributionCertificate:
    """Two arms: an interaction the plant has, and one the tracking loop manufactures."""

    gains: tuple[float, ...]
    true_interaction: float  # the plant's own b1 in arm A
    recovered: tuple[float, ...]  # fitted b1 under exact tracking -- should equal it at every gain
    drift_error: tuple[float, ...]  # |fitted a - true a| under the same fits; should NOT be small
    curvature: float  # the u^2 term arm B's plant has and the fitted class cannot represent
    spurious: tuple[float, ...]  # fitted b1 in arm B: pure artefact, -gain*curvature
    spurious_predicted: tuple[float, ...]  # -gain*curvature
    refuted_guess: tuple[float, ...]  # -1/gain, the form plans/24 carried
    exploration: tuple[float, ...]  # off-manifold noise levels for the recovery sweep
    drift_error_by_exploration: tuple[float, ...]  # |fitted a - true a| as exploration grows
    condition_by_exploration: tuple[float, ...]
    ok: bool


def closed_loop_attribution_certificate(
    gains: Sequence[float] = (0.5, 1.0, 2.6, 8.0),
    exploration: Sequence[float] = (0.0, 0.05, 0.25, 1.0),
    samples: int = 4000,
    seed: int = 0,
) -> ClosedLoopAttributionCertificate:
    """Separate the interaction a plant HAS from the one a tracking loop MANUFACTURES.

    Arm A: the plant really is ``d + a*x + (b0 + b1*x)*u``. Under exact tracking the design is
    singular, yet ``b1`` comes back exactly while the drift ``a`` does not -- the non-identification
    is real and it is the pole that suffers, not the interaction.

    Arm B: the plant has **no** interaction, but its response carries a ``u^2`` term the fitted
    class cannot represent. The fit answers with ``b1 = -gain*curvature``: a coefficient that is
    entirely an artefact of misspecification amplified by the loop, growing linearly with the
    controller's gain. That is the mechanism behind Result 41's large negative ``b1``, and why
    the number cannot be read as authority-falls-with-temperature without checking the budget.

    The exploration sweep is the remedy: variation off the manifold restores the design's rank and
    the drift with it. All three claims are measured here, and the middle one is the one that would
    embarrass the entry if ``b1`` turned out to be as unstable as ``a``.
    """
    rng = np.random.default_rng(seed)
    true_drift, true_offset, true_channel, true_interaction = -0.35, 1.0, 0.40, -0.30
    curvature = 0.1454

    def tracked(gain: float, noise: float) -> tuple[Array, Array]:
        actions = 20.0 + 2.0 * rng.standard_normal(samples)
        states = 17.0 - actions / gain + noise * rng.standard_normal(samples)
        return jnp.asarray(states), jnp.asarray(actions)

    recovered, drift_error, spurious = [], [], []
    for gain in gains:
        states, actions = tracked(gain, 0.0)
        rates = (
            true_offset + true_drift * states + (true_channel + true_interaction * states) * actions
        )
        fit = closed_loop_gain_attribution(states, actions, rates)
        recovered.append(fit.fitted_interaction)
        drift_error.append(abs(fit.fitted_drift - true_drift))

        curved = true_offset + true_drift * states + true_channel * actions + curvature * actions**2
        spurious.append(closed_loop_gain_attribution(states, actions, curved).fitted_interaction)

    drift_by_noise, condition_by_noise = [], []
    for noise in exploration:
        states, actions = tracked(2.6, noise)
        rates = (
            true_offset + true_drift * states + (true_channel + true_interaction * states) * actions
        )
        fit = closed_loop_gain_attribution(states, actions, rates)
        drift_by_noise.append(abs(fit.fitted_drift - true_drift))
        condition_by_noise.append(fit.design_condition)

    predicted = tuple(-g * curvature for g in gains)
    return ClosedLoopAttributionCertificate(
        gains=tuple(float(g) for g in gains),
        true_interaction=true_interaction,
        recovered=tuple(recovered),
        drift_error=tuple(drift_error),
        curvature=curvature,
        spurious=tuple(spurious),
        spurious_predicted=predicted,
        refuted_guess=tuple(-1.0 / g for g in gains),
        exploration=tuple(float(e) for e in exploration),
        drift_error_by_exploration=tuple(drift_by_noise),
        condition_by_exploration=tuple(condition_by_noise),
        ok=(
            all(abs(b - true_interaction) < 1e-6 for b in recovered)
            and max(drift_error) > 1e-2  # the pole is NOT recovered, and that is the point
            and all(abs(s - p) < 1e-6 for s, p in zip(spurious, predicted, strict=True))
            and drift_by_noise[-1] < 1e-6  # exploration buys the drift back
            and condition_by_noise[-1] < condition_by_noise[0]
        ),
    )
