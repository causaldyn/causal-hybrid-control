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

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from chc.causal import _polynomial_features, _ridge_predict
from chc.dynamics import DrivenDynamics, Dynamics, HybridDynamics
from chc.integrate import rk4_step
from chc.residual import ControlAffineResidual, control_affine_features

Integrator = Literal["euler", "rk4"]
"""Which one-step map the fit is asked to be consistent with. See :func:`fit_causal_residual`."""

_MAX_NEWTON_STEPS = 20
"""Cap on Newton's method for the ``rk4`` fixed point. From the Euler fit it converges in three to
six steps up to ``|A|dt = 1.2``, so reaching the cap means there is no fixed point to reach."""

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
    # Root-mean diagonal of the channel's robust covariance, None when not identified: each row's
    # squared structural residual carried through the fit's own linear map, cross-fitted nuisances
    # included. Robust since 0.8.0; the homoskedastic error before it read 0.80x the channel's
    # spread over 800 logs whose noise moves with the state, and 0.40x when the class missed the
    # truth, where the robust one's root mean square reads 0.96x and 1.04x. On a confounded plant
    # it came to 1.08x the channel's error against the truth on the adjustment path and 1.05x on
    # the ``iv`` path (200 logs of 2000 rows). One log's value scatters, by 24% and 40% of itself
    # on those 800 logs of 4000 rows: a scale, not coverage. Under ``rk4`` it is the fixed point's
    # own, the noise carried through the RK4 map's gain on the estimate: 1.70x the Euler fit's at
    # ``theta*dt = 0.7``, where 200 noise draws on one log scattered the channel 1.73x as far.
    channel_error: float | None
    # Root-mean diagonal of the drift stage's own homoskedastic OLS covariance, or None when the
    # channel is not identified (the drift is then conditional on a meaningless channel). It is a
    # DIFFERENT object from ``channel_error``: conditional on the fitted channel, whose uncertainty
    # it does not propagate, and homoskedastic -- a scale, not coverage. Reported because on a real
    # plant the drift, not the channel, dominated closed-loop cost. Its noise is the drift
    # regression's residual, so it counts what the model class leaves out: on the marketing-mix
    # plant, 3.4x the drift's scatter over noise draws. Under ``rk4`` it goes through the drift's
    # block of the RK4 map's gain, and the map couples the drift to the channel it is conditional
    # on: on a log the model class fits, the drift scattered 1.26x it at ``theta*dt = 0.7`` over
    # 400 seeds, against 1.02x under Euler.
    drift_error: float | None
    action_residual_variance: (
        float  # overlap proxy: 0 => a deterministic policy, nothing to regress
    )
    nuisance_r2_state: float
    nuisance_r2_action: float
    moment_norm: float  # ||mean(design * residual)|| at the solution; should be ~0
    integrator: Integrator = "euler"  # which one-step map the fit was made consistent with
    # RMS one-step defect in rate units, ``||x_next - step(F, x, u, dt)|| / dt``, under THAT
    # integrator. Comparable across the two settings. It floors at the observation noise, never at
    # zero; above that it is what the model class leaves out, since an ``rk4`` fit that cannot
    # reach its fixed point raises instead.
    integrator_defect: float | None = None
    drivers: tuple[str, ...] = ()  # the exogenous columns fitted into the drift, in gain order
    # (n, d): each driver's push on each state's rate, fitted jointly with the drift; None without
    # drivers. :class:`chc.dynamics.DrivenDynamics` replays it against a forecast.
    driver_gain: Array | None = None
    weighted: bool = False  # whether a weight on the state weighed the channel moment
    # Kish's (sum w)^2 / sum w^2: how many equally weighted transitions the moment is worth
    effective_sample_size: float | None = None


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
    weights: Array | None = None,
) -> Array:
    """Solve the just-identified moment ``Z'(y_res - D c) = 0`` for ``c``, ridge-stabilised.

    ``Z is D`` reduces to ordinary least squares; a different ``Z`` is two-stage least squares.
    Writing it as one solve rather than "regress on the projection" matters: the two agree only for
    a scalar action with no state features, because the in-sample identity ``Z'u = Z'Z`` does not
    survive multiplication by ``phi(x)``.
    """
    if weights is not None:
        instrument = instrument * weights[:, None]
    gram = instrument.T @ regressor + ridge * jnp.eye(regressor.shape[1])
    return jnp.linalg.solve(gram, instrument.T @ state_residual)


def _robust_spread(sensitivity: Array, score: Array, n_coeff: int) -> Array:
    """``sum_i J_i diag(e_i^2) J_i'`` for a fit linear in its target ``y``, ``J = d coeffs / d y``.

    Each row's own squared residual stands in for its noise, so it holds when the noise differs
    across rows, and ``J`` runs through the cross-fitted nuisances as well as the moment. A weight
    that loads a few rows loads the nuisance fits' error at them too, which a sandwich on the moment
    alone misses: on a log whose noise grew as ``exp(x)``, weighed by ``exp(-x)``, that sandwich
    came to 0.67 of the channel's spread over sixteen redraws of the noise, and this to 1.04.
    """
    n = score.shape[0]
    scale = n / max(n - n_coeff, 1)
    return jnp.einsum("pis,is,qis->pq", sensitivity, scale * score**2, sensitivity)


def _state_weights(weights: Callable[[Array], Array], states: Array) -> Array:
    """The caller's weight at each state, checked and scaled to mean 1, so ``ridge`` keeps its
    meaning whatever the weight's units."""
    if not callable(weights):
        raise TypeError(f"weights must be a function of the state, or None; got {weights!r}")
    values = jnp.asarray(weights(states), dtype=states.dtype)
    if values.shape != (states.shape[0],):
        raise ValueError(
            f"the weight must give one value per transition, shape ({states.shape[0]},), from the "
            f"states alone; got shape {values.shape}"
        )
    if not bool(jnp.all(jnp.isfinite(values))) or bool(jnp.any(values < 0.0)):
        raise ValueError("the weight must be finite and non-negative at every state")
    if float(jnp.sum(values)) <= 0.0:
        raise ValueError("the weight is zero at every state")
    return values / jnp.mean(values)


def _ols_error(target: Array, design: Array, coeffs: Array, ridge: float) -> float:
    """Root-mean diagonal of ``sigma^2 (X'X)^-1`` -- the plain homoskedastic OLS covariance.

    Used for the drift stage, which is fitted by least squares on the remainder, so there is no
    instrument and no endogenous regressor to sandwich against. Unlike the channel's robust error it
    pools the noise over the rows, and it is a scale rather than a coverage statement.
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
    weights: Array | None = None,
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
        weights: one weight per transition, shape ``(N,)``, on its moment; ``None`` weighs all
            alike. A weight that is a function of the state alone keeps the moment orthogonal (see
            :func:`fit_causal_residual`). Experimental: it may change or be withdrawn in any
            release.

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
    coeffs = _channel_coefficients(state_residual, regressor, moment, ridge, weights)
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
    channel_degree: int | None = None,
    nuisance_degree: int = 2,
    folds: int = 2,
    ridge: float = 1e-6,
    seed: int = 0,
    integrator: Integrator = "euler",
    drivers: tuple[str, ...] = (),
    weights: Callable[[Array], Array] | None = None,
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
        degree: the feature degree of the drift, and of the channel unless ``channel_degree``
            says otherwise. ``1`` fits an affine drift and, by default, a channel affine in the
            state, which contains the constant channel §18/§19 cover without being restricted to it.
        channel_degree: the feature degree of the channel; ``None`` is ``degree``. ``0`` fits the
            constant channel §18/§19 cover beside a drift of degree ``degree``: ``n m``
            coefficients where the affine channel has ``n m (n + 1)``, and as many fewer tangents
            in the ``rk4`` fixed point. An affine channel is as accurate as a constant one only
            where the log is centred. On eight logs of :class:`chc.zones.ZoneMarketSystem`, against
            one RK4 period's response (``scripts/bench_channel_degree.py``), the constant channel
            was 3.6% (linear matching) and 4.6% (harmonic) off in relative RMS wherever it was
            read; the affine one was as close at the logs' mean state, 16% off at the do-nothing
            point and 159% and 123% at ``x = 0``, where :meth:`chc.decision.Prescription.reach`
            reads it.
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
        integrator: the one-step map the fitted field is made consistent with.

            ``"euler"`` (default, and what every release so far did) reads the rate off the log as
            ``(x_next - x)/dt``, which is exactly right if the caller treats the result as a
            *discrete-time* model and steps it the same way. It is **not** right if the caller then
            integrates with :func:`chc.integrate.rk4_step`, as :func:`chc.plan.causal_plan` and
            :func:`chc.integrate.rollout` do: the two disagree by the RK4 amplification factor
            ``1 + z + z^2/2 + z^3/6 + z^4/24`` at ``z = A dt``. On a linear plant at ``theta*dt =
            0.7`` that is a decay fitted at ``-0.502`` against a true ``-0.700`` and, more to the
            point, a control channel fitted at ``0.574`` against a true ``0.800``.

            ``"rk4"`` makes the field consistent with RK4 instead, without inverting RK4 in closed
            form, which does not exist for a nonlinear field. Coefficients read a rate off the log:
            their own fitted rate plus what their RK4 step leaves of ``x_next``. The ``rk4`` fit is
            the one the estimator returns when handed its own reading, found by Newton's method
            from the Euler fit in three to six steps between ``|A|dt = 0.05`` and ``1.2``. Measured
            on the same plant, it recovers ``-0.699`` and ``0.800``. Its standard errors are the
            fixed point's own (:attr:`CausalDynamicsFit.channel_error`).

            Up to 0.6.0 the fixed point was iterated instead -- add the leftover to the target,
            refit -- and each state stopped once its defect's RMS stopped falling. That is not where
            the fixed point is: the RMS flattens under the noise first, and the iteration contracts
            the drift only by ``1 - (1 + z + z^2/2 + z^3/6)`` a pass, 0.51 at ``z = -0.7``. Even run
            until the RMS stopped falling at all, it stopped 0.9 standard errors of the channel and
            1.4 of the decay short at ``theta*dt = 0.7`` and ``N = 1000``, and 0.09 of the decay
            short at 1.25 on a log solved exactly.

            A log that falls by more in one step than RK4 can follow has no ``rk4`` reading at all.
            RK4's growth factor bottoms out at 0.2704, at ``z = -1.596``, and an exact linear mode
            falls below that past ``|A|dt = 1.308``. The fit raises there rather than return a field
            that is neither reading. A channel that reads the state can run out sooner: on the plant
            in :mod:`chc.mmm` with its fastest channel decaying at ``theta*dt = 1.2``, one seed in
            eight stalls in that channel's dependence on its own adstock, at a gap a damped Newton
            cannot close either, at both precisions. ``integrator_defect`` floors at the noise under
            both.

            The default stays ``"euler"`` so no shipped fit changes meaning.
            :func:`chc.decision.prescribe` defaults the other way, because it *knows* the field goes
            to the planner.
        drivers: exogenous columns that push the state and that nothing the plan does can move --
            weather, a demand forecast. ``data`` carries each at the start of the transition under
            its own name and at the end under ``f"{name}_next"``, and between the two it moves
            linearly, which is how :class:`chc.dynamics.DrivenDynamics` replays a forecast and what
            the ``rk4`` fit steps. Each enters the drift regression **jointly** with the
            state features, so the drift stops carrying the driver's correlation with the state
            (this module's first scope note): at its mean over the step under ``rk4``, the push a
            linear path puts into a step, and at the start under ``euler``, all the Euler map
            reads. Both ends join the channel's nuisance covariates. A logging policy that reads
            the forecast sets the action on where the driver is going, and with the start alone
            that is confounding: on a random-walk driver logged that way the ``euler`` channel came
            back ``0.656`` against the Euler map's own ``0.780``. Partialling out is safe for any
            driver, since a column nothing downstream of the action moves can be neither a mediator
            nor a collider of its effect. The fitted gain is :attr:`CausalDynamicsFit.driver_gain`;
            its standard error is inside ``drift_error``.
        weights: a weight on each transition's channel moment, a function of its state alone.
            Experimental: it may change or be withdrawn in any release. A callable is called with
            ``x (N, n)`` and nothing else, since a weight that reads the
            action or the next state biases the channel at first order, while any weight on the
            state keeps the moment orthogonal. The weights are scaled to mean 1. The channel's
            standard error is robust, as for every fit, and here it has to be: a weight that loads
            a few rows loads the nuisances' error at them too. ``effective_sample_size`` says how
            few.

            Under a channel class that contains the truth every weight estimates the same channel,
            and only the variance moves. The inverse of the rate's noise variance is the efficient
            weight in the limit, but it loads the states where the noise is small, which can be
            where the nuisances extrapolate: on a log whose rate noise ran ``exp(-x/2)`` against
            the action's ``exp(x/2)``, the channel scattered 0.87 as far as the unweighted fit's at
            ``N = 64000``, its limit, and 1.15 as far at ``N = 4000``.

            Under a class that misses the truth, each weight estimates a different channel: the
            projection of the true one under ``w s^2`` times the log's law of states, where
            ``s^2(x)`` is the variance the adjustment set leaves in the action at ``x``. The weight
            then chooses where the fit is right. For a one-shot decision on a scalar action, made
            at states drawn from ``Q`` with regret curvature ``kappa(x)`` in the channel,
            ``w = kappa (dQ/dP) / s^2`` makes the fit the best of its class for that decision. A
            weight that has to be estimated first -- ``s^2``, or the inverse noise -- carries its
            own error into the channel at first order there, since the estimand moves with the
            weight. No weight on the state alone serves a dynamic plan, which reads the channel's
            slope along its path as well as its level: there the costate weight left 36 to 178
            times the regret of the best fit in the class.

    Returns:
        A :class:`CausalDynamicsFit`. Read ``identified`` before ``residual``.
    """
    channel_degree = degree if channel_degree is None else channel_degree
    if channel_degree < 0:
        raise ValueError(f"channel_degree must be a non-negative integer; got {channel_degree}")
    x, u, x_next = data["x"], data["u"], data["x_next"]
    known_rate = jax.vmap(lambda xi, ui: known(0.0, xi, ui))(x, u)

    identified = bool(adjust_for) or instrument is not None
    driver_start = jnp.concatenate([x[:, :0], *[data[name] for name in drivers]], axis=1)
    driver_end = jnp.concatenate([x[:, :0], *[data[f"{name}_next"] for name in drivers]], axis=1)
    # What a transition's rate sees of a driver: its mean over the step, exact for the linear path
    # between the two ends, or its value at the start, which is all the Euler map reads.
    driver_read = driver_start if integrator == "euler" else 0.5 * (driver_start + driver_end)
    unadjusted = [data[name] for name in drivers if name not in adjust_for]
    covariates = jnp.concatenate(
        [x, *[data[name] for name in adjust_for], *unadjusted, driver_end], axis=1
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

    # a_θ mops up the rest of y at the fitted channel; see this module's scope note on why that
    # makes the drift observational-conditional while the channel stays interventional. The drivers
    # sit in the same regression, read the way the integrator reads them.
    phi_x = jax.vmap(control_affine_features, in_axes=(0, None))(x, degree)
    phi_c = jax.vmap(control_affine_features, in_axes=(0, None))(x, channel_degree)
    design = jnp.concatenate([phi_x, driver_read], axis=1)

    row_weight = None if weights is None else _state_weights(weights, x)

    def solve(y: Array) -> tuple[Array, Array, tuple[Array, Array, Array, Array, Array]]:
        """The channel and the drift regression's coefficients a target ``y`` fits to, with the
        residualisation behind them. Linear in ``y``, which the ``rk4`` fixed point relies on."""
        y_res, u_res, y_hat, u_hat = _cross_fit_residuals(
            y, u, covariates, degree=nuisance_degree, folds=folds, ridge=ridge, seed=seed
        )
        channel = solve_channel_moment(
            y_res,
            u_res,
            x,
            instrument_action=instrument_action,
            degree=channel_degree,
            ridge=ridge,
            weights=row_weight,
        )
        fitted = jax.vmap(lambda c, ui: (channel @ c) @ ui)(phi_c, u)
        remainder = _solve_ridge(design, y - fitted, ridge)  # (features + drivers, n)
        return channel, remainder, (y_res, u_res, y_hat, u_hat, fitted)

    def fit_to(rate: Array) -> CausalDynamicsFit:
        """Everything downstream of the target rate, so the ``rk4`` fixed point can re-run it."""
        y = rate - known_rate
        channel, remainder, (y_res, u_res, y_hat, u_hat, fitted) = solve(y)
        regressor = _channel_design(u_res, x, channel_degree)
        moment = (
            regressor
            if instrument_action is None
            else _channel_design(instrument_action, x, channel_degree)
        )
        coeffs = channel.reshape(x.shape[1], -1).T
        drift = remainder[: phi_x.shape[1]].T
        gain = remainder[phi_x.shape[1] :].T

        score = y_res - regressor @ coeffs
        weighted_moment = moment if row_weight is None else moment * row_weight[:, None]
        return CausalDynamicsFit(
            residual=ControlAffineResidual(
                drift=drift, channel=channel, degree=degree, channel_degree=channel_degree
            ),
            identified=identified,
            method=method,
            folds=folds,
            # the channel's comes from the fit's own sensitivity, which only the caller knows how
            # to take: through the cross-fit under Euler, through the fixed point under ``rk4``
            channel_error=None,
            drift_error=_ols_error(y - fitted, design, remainder, ridge) if identified else None,
            action_residual_variance=float(jnp.mean(u_res**2)),
            nuisance_r2_state=_r_squared(y, y_hat),
            nuisance_r2_action=_r_squared(u, u_hat),
            moment_norm=float(jnp.linalg.norm(weighted_moment.T @ score / moment.shape[0])),
            integrator=integrator,
            drivers=tuple(drivers),
            driver_gain=gain if drivers else None,
            weighted=row_weight is not None,
            effective_sample_size=float(x.shape[0])
            if row_weight is None
            else float(jnp.sum(row_weight) ** 2 / jnp.sum(row_weight**2)),
        )

    def predicted(residual: ControlAffineResidual, gain: Array) -> Array:
        """``step(F, x, u, dt)`` on every transition, under the integrator the fit is made for."""
        model = HybridDynamics(known=known, residual=residual)

        def step(xi: Array, ui: Array, start: Array, end: Array) -> Array:
            # The transition's own stretch of the drivers' path, held the way the planner holds a
            # forecast: one implementation of the hold, so the fit and the plan cannot drift apart.
            field = DrivenDynamics(model, gain, jnp.stack([start, end]), dt)
            if integrator == "euler":
                return xi + dt * field(0.0, xi, ui)
            return rk4_step(field, 0.0, xi, ui, dt)

        return jax.vmap(step)(x, u, driver_start, driver_end)

    def defect(fit: CausalDynamicsFit) -> Array:
        """``(x_next - step(F, x, u, dt)) / dt`` -- what the fitted field leaves of the log."""
        gain = jnp.zeros((x.shape[1], 0)) if fit.driver_gain is None else fit.driver_gain
        return (x_next - predicted(fit.residual, gain)) / dt

    def rk4_fixed_point(start: CausalDynamicsFit) -> CausalDynamicsFit:
        """The fit that is its own ``rk4`` reading of the log, by Newton's method from ``start``.

        Coefficients ``theta`` read a rate off the log: their own fitted rate ``Phi theta`` plus
        what their RK4 step leaves over, ``y(theta) = Phi theta + (x_next - RK4_theta) / dt``.
        Under Euler that is ``(x_next - x) / dt`` whatever ``theta`` is. The fixed point is
        ``theta = G y(theta)``, with ``G`` the fit, which is linear in its target.

        A noise ``e`` in ``x_next`` moves it by ``K^-1 G e / dt``, where ``K = I - G dy/dtheta`` is
        the matrix Newton steps with, the identity under Euler. The drift's error stays conditional
        on the channel, as it is under Euler: its own regression, through the drift's block of K.
        """
        shape, size = start.residual.channel.shape, start.residual.channel.size

        def unpack(theta: Array) -> tuple[ControlAffineResidual, Array, Array]:
            rows = theta[size:].reshape(-1, x.shape[1])  # the drift regression's: drift, then gain
            channel = theta[:size].reshape(shape)
            residual = ControlAffineResidual(
                drift=rows[: phi_x.shape[1]].T,
                channel=channel,
                degree=degree,
                channel_degree=channel_degree,
            )
            return residual, rows[phi_x.shape[1] :].T, rows

        def reading(theta: Array) -> Array:
            residual, gain, rows = unpack(theta)
            own = jax.vmap(lambda c, ui: (residual.channel @ c) @ ui)(phi_c, u) + design @ rows
            return own + (x_next - predicted(residual, gain)) / dt

        def coefficients(y: Array) -> Array:
            channel, remainder, _ = solve(y)
            return jnp.concatenate([channel.ravel(), remainder.ravel()])

        def newton_matrix(theta: Array) -> tuple[Array, Array]:
            sensitivity = jax.jacfwd(reading)(theta)  # (N, n, p)
            gained = jnp.einsum("pis,isq->pq", fit_map, sensitivity)
            return jnp.eye(theta.size) - gained, sensitivity

        fit_map = jax.jacrev(coefficients)(jnp.zeros_like(x_next))  # (p, N, n), exact: linear
        gain = jnp.zeros((x.shape[1], 0)) if start.driver_gain is None else start.driver_gain
        remainder = jnp.concatenate([start.residual.drift.T, gain.T])
        theta = jnp.concatenate([start.residual.channel.ravel(), remainder.ravel()])
        tolerance = float(jnp.sqrt(jnp.finfo(theta.dtype).eps))
        for _ in range(_MAX_NEWTON_STEPS):
            gap = jnp.einsum("pis,is->p", fit_map, reading(theta)) - theta
            step = jnp.linalg.solve(newton_matrix(theta)[0], gap)
            theta = theta + step
            if float(jnp.max(jnp.abs(step))) <= tolerance * float(jnp.max(jnp.abs(theta))):
                break
        else:
            raise ValueError(
                f"the rk4 fixed point did not converge in {_MAX_NEWTON_STEPS} Newton steps (last "
                f"step {float(jnp.max(jnp.abs(step))):.3g}): no field stepped by RK4 at dt={dt} "
                "reproduces this log. On a linear mode that happens once |A|dt passes 1.31, where "
                "the log decays by more in one step than any RK4 step can. Log at a finer dt, or "
                "fit with integrator='euler' and treat the result as the discrete-time map it is."
            )
        target = reading(theta)
        fit = fit_to(known_rate + target)
        if not identified:
            return fit

        # Only the noise is random given the log, and the channel reads it where its own sandwich
        # does, after the adjustment set has taken out what it explains: a push the model leaves
        # out but the adjustment set carries -- a season -- is in the defect and is not noise.
        channel, _, (y_res, u_res, *_) = solve(target)
        regressor = _channel_design(u_res, x, channel_degree)
        score = y_res - regressor @ channel.reshape(x.shape[1], -1).T
        newton, sensitivity = newton_matrix(theta)
        spread = _robust_spread(fit_map, score, regressor.shape[1])
        covariance = jnp.linalg.solve(newton, jnp.linalg.solve(newton, spread).T)
        # The drift keeps the defect, which is its own regression's residual, as under Euler.
        noise = jnp.sum(defect(fit) ** 2, axis=0) / max(x.shape[0] - design.shape[1], 1)
        gram = jnp.linalg.inv(design.T @ design + ridge * jnp.eye(design.shape[1]))
        drift_size = theta.size - size
        drift_gained = jnp.einsum("fi,isq->fsq", gram @ design.T, sensitivity[:, :, size:])
        drift_newton = jnp.eye(drift_size) - drift_gained.reshape(drift_size, drift_size)
        drift_spread = jnp.kron(gram, jnp.diag(noise))
        drift_covariance = jnp.linalg.solve(
            drift_newton, jnp.linalg.solve(drift_newton, drift_spread).T
        )
        return dataclasses.replace(
            fit,
            channel_error=float(jnp.sqrt(jnp.mean(jnp.diag(covariance)[:size]))),
            drift_error=float(jnp.sqrt(jnp.mean(jnp.diag(drift_covariance)))),
        )

    fit = fit_to((x_next - x) / dt)
    if integrator == "rk4":
        fit = rk4_fixed_point(fit)
    elif identified:
        y = (x_next - x) / dt - known_rate
        channel, _, (y_res, u_res, *_) = solve(y)
        regressor = _channel_design(u_res, x, channel_degree)
        score = y_res - regressor @ channel.reshape(x.shape[1], -1).T
        sensitivity = jax.jacrev(lambda target: solve(target)[0].ravel())(jnp.zeros_like(y))
        spread = _robust_spread(sensitivity, score, regressor.shape[1])
        fit = dataclasses.replace(fit, channel_error=float(jnp.sqrt(jnp.mean(jnp.diag(spread)))))
    return dataclasses.replace(fit, integrator_defect=float(jnp.sqrt(jnp.mean(defect(fit) ** 2))))


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
    refuted_guess: tuple[float, ...]  # -1/gain: the manifold slope, once guessed to be b1
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
