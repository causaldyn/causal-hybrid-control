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

HONEST SCOPE, four limits worth stating before the code:

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
* With no adjustment set and no instrument nothing in the log identifies the channel, and an
  instrument identifies it only along the directions it moves the action. The estimator reports
  ``identified=False`` in both cases rather than returning a confident wrong answer -- that case
  belongs to :mod:`chc.sensitivity`, which prices the radius instead of pretending it away.
* The premise is Markov: given ``x``, ``z`` and ``u``, a transition's noise does not depend on the
  past. Where the noise persists within a unit and a lever persists too, the state is a common
  effect of the past lever and the past noise, so ``E[eps | x, z, u - m]`` is not 0 and the channel
  is biased by an amount ``channel_error`` does not cover. :func:`persistence_check` reads the
  noise's persistence off the moment's residual; adjusting for the state, the levers and ``z`` a
  period earlier removes the bias where the noise is AR(1) (ADR 0064).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.stats import t as student_t

from chc import _units
from chc.causal import (
    _ACTION_AND_OUTCOME,
    _TRANSITION,
    _polynomial_features,
    _refuse_reread,
    _ridge_predict,
    _stream_key,
)
from chc.dynamics import DrivenDynamics, Dynamics, HybridDynamics
from chc.frames import _real_entries, _real_numbers
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

    ``identified`` is the load-bearing field: ``False`` means nothing the fit was given pins the
    channel down from the log -- no adjustment set and no instrument, or an instrument whose moment
    falls short of rank (``instrument_rank``) -- and the residual is then the observational fit,
    kept only so the caller can compare.
    """

    residual: ControlAffineResidual
    identified: bool
    method: str  # "orthogonal" | "iv" | "observational"
    folds: int
    # The standard error of the fitted channel's value B(x) at the log's states, root mean square
    # over the states and the channel's entries, None when not identified. It is read off the
    # channel's robust covariance: each row's squared structural residual carried through the fit's
    # own linear map, cross-fitted nuisances included, or under ``clusters`` each cluster's summed
    # score squared (CR1; two-way, the largest of three reads). Up to 0.12 it was that covariance's
    # root-mean diagonal, whose constant term is the value at x = 0, so it moved with the state's
    # zero: one log read 0.0091, and 0.579 with its states moved by 100 and -50. The figures that
    # follow compared that version with the spread of the same coefficients. Robust since 0.8.0;
    # the homoskedastic error before it read 0.80x the channel's spread over 800 logs whose noise
    # moves with the state, and 0.40x when the class missed the truth, where the robust one's root
    # mean square reads 0.96x and 1.04x. On a confounded plant it came to 1.08x the channel's error
    # against the truth on the adjustment path and 1.05x on the ``iv`` path (200 logs of 2000
    # rows). One log's value scatters, by 24% and 40% of itself on those 800 logs of 4000 rows: a
    # scale, not coverage. Under ``rk4`` it is the fixed point's own, the noise carried through the
    # RK4 map's gain on the estimate: 1.70x the Euler fit's at ``theta*dt = 0.7``, where 200 noise
    # draws on one log scattered the channel 1.73x as far. Read from ``G`` clusters' sums, a test
    # read off it sizes better against ``t(G - 1)``'s quantile than the normal's (ADR 0062).
    channel_error: float | None
    # The standard error of the drift regression's fitted value at the log's rows, root mean square
    # over the rows and the states, from the drift stage's own homoskedastic OLS covariance; None
    # when the channel is not identified (the drift is then conditional on a meaningless channel).
    # Up to 0.12 it was that covariance's root-mean diagonal, which the figures here were read on.
    # It is a DIFFERENT object from ``channel_error``: conditional on the fitted channel, whose
    # uncertainty it does not propagate, and homoskedastic -- a scale, not coverage. Reported
    # because on a real plant the drift, not the channel, dominated closed-loop cost. Its noise is
    # the drift regression's residual, so it counts what the model class leaves out: on the
    # marketing-mix plant, 3.4x the drift's scatter over noise draws. Under ``rk4`` it goes through
    # the drift's block of the RK4 map's gain, and the map couples the drift to the channel it is
    # conditional on: on a log the model class fits, the drift scattered 1.26x it at
    # ``theta*dt = 0.7`` over 400 seeds, against 1.02x under Euler.
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
    # (N, n, p), kept when the fit was asked for it: each transition's share, per state, of the
    # error in the fitted parameters when the whole log is drawn again -- the channel, raveled, then
    # the drift regression's rows (the drift's transpose, then the drivers' gain's), raveled. A row
    # moves the channel through the moment's residual, and the drift through the channel and through
    # its own regression's residual, so the sum of psi psi' over both is their robust covariance,
    # the drift's carrying the channel's error. Under Euler its channel block is
    # ``channel_error``'s; under ``rk4`` the RK4 map also carries the drift's residual into the
    # channel, which ``channel_error``, read with the noise alone random, leaves out.
    influence: Array | None = None
    # (p, r), in ``influence``'s order: the directions the log's actions never move. Each moves one
    # state's channel where the action residuals, scaled to the raw actions' size, fall below the
    # square root of the working precision, with the drift regression's response to it on this log.
    # The moment has no data there. Where the drift's features take up a direction's push on the
    # log, no rate of the log tells it apart, and the channel is held at zero along it, in the
    # actions' scaled units; where they cannot, the log's rates rule out all but one value, and the
    # channel takes it by least squares beside the drift: what the log did, not an effect, whatever
    # ``channel_error`` says (ADR 0054). ``r`` is 0 when the log moves every direction.
    unmoved: Array | None = None
    # (N, n, q), kept with ``influence``, ``q`` the channel's size in its raveled order: ``N`` times
    # each transition's weight, per state, in each channel coefficient through the channel's moment
    # with the cross-fitted nuisances held, less the drift regression's own weight on the row, which
    # reads the state alone and so carries no omitted confounder's push. It is the plug-in Riesz
    # representer of every linear functional of the channel, ``N (D'D)^-1 D_i`` under Euler, which
    # :func:`omitted_confounder_bound` reads. Under ``rk4`` it goes through the fixed point, to
    # first order, as ``influence`` does.
    representer: Array | None = None
    # (N, n), where the channel is identified: its moment's residual, ``y_res - D c``, per state,
    # which :func:`persistence_check` reads.
    moment_residual: Array | None = None
    # (N,), each transition's cluster as a code from 0 where the fit was given ``clusters``, or
    # (N, 2) where it was given two, a code a dimension: the channel's covariance then sums each
    # cluster's scores over its transitions and states before squaring them (CR1), two-way the
    # largest read of the two ways and of each alone, and a reader of ``influence`` sums its rows
    # the same way (:func:`_clustered_squares`). None where each transition's each state is its own.
    clusters: np.ndarray | None = None
    # Experimental: it may change or be withdrawn in any release. Under ``instrument``, ``(p,)``:
    # how far the instrument moves each of the ``p`` directions a state of the channel the log's
    # actions move, all but ``unmoved``'s. They are the canonical correlations, largest first,
    # between the first stage's push on the actions, fitted without the ridge, and the actions,
    # each less its least-squares projection on the nuisance's features, on the channel's features
    # and with the moment's weights: the unregularised moment's singular values with each side
    # whitened, read the same in any units of the instrument, the actions and the state. A push of
    # at most 64 eps of the raw actions' size reads 0, as does a correlation of at most 64 eps. A 0
    # leaves the moment short of rank, and the fit not identified; so does ``p = 0``, a log that
    # moves no direction, which leaves the instrument none to move. The smallest says how weak the
    # instrument is where it is weakest; it is not a test. None without an instrument.
    instrument_relevance: Array | None = None
    # Experimental, as ``instrument_relevance``: how many of its entries are above 0, the rank of
    # the instrument's moment along the directions the log moves. None without an instrument.
    instrument_rank: int | None = None


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
    thing this adds is the fold bookkeeping over two dimensions -- and the standardisation below.

    Each covariate is centred and scaled first (:func:`chc._units.standardised`), so the nuisance
    basis is conditioned in any caller's units. Without this the polynomial basis inherits whatever
    units the caller happened to use, and the ridge penalises each monomial accordingly --
    ``ridge=1e-6`` means something entirely different against a column of order 1 than against its
    square of order 400. Measured on a 20-day log from a real building emulator (``causaldyn-bench``
    Track D-causal), where the zone enters in Celsius at ~21 and the weather columns are already
    standardised: the degree-2 Gram came out at condition number **1.4e11**, past what float32 can
    carry, and the fit returned ``nan``. The same rows in float64 fitted fine, which is what
    identified this as conditioning rather than data. Scaling first drops it to order 10 -- on a
    reproducible stand-in with that geometry (768 rows, zone at 21 +- 0.9 beside three
    standardised columns) the degree-2 Gram goes from ``2.7e10`` to ``2.4e1`` -- and removes the
    precision dependence.

    Full-sample statistics rather than per-fold ones. Centring and scaling the *inputs* is a linear
    reparametrisation, so the span of the fitted basis -- and hence the partialled-out residual --
    is unchanged by it; only the ridge's meaning moves, and pinning that to the data's own scale is
    the point. Per-fold statistics would make the penalty fold-dependent for no gain. A column whose
    spread is rounding reads 0 in every row: divided by its own spread it would be ``nan``, or its
    rounding blown up to unit size, and a floor of 1e-12 in the caller's units took a column logged
    in small units for a constant and left it unscaled under the ridge.
    """
    covariates = _units.standardised(covariates)
    n = target.shape[0]
    chunks = jnp.array_split(jax.random.permutation(_stream_key(seed, "folds"), n), folds)
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


def _ridge_parts(design: Array, ridge: float) -> tuple[Array, Array, Array, Array]:
    """The columns after the bias, centred twice, their means and second-pass shifts
    (:func:`chc._units.centred`), and the inverse of their Gram under the ridge: a ridge solve of
    ``X'X + ridge diag(s)`` written with the columns centred, where the bias is apart from the rest,
    for a design whose first column is its bias, as :func:`chc.residual.control_affine_features`
    and :func:`chc.causal._polynomial_features` put it. ``s`` is 0 on the bias, left free to take
    up each column's mean, and each other column's variance about its mean (ADR 0057). Scaled by
    its mean square, a column far from zero, such as a temperature in kelvin, had its slope shrunk
    by the square of its offset over its spread.

    The solve is the same, and a column far from zero costs it no digits: on the raw Gram, a column
    1e-11 of its size from constant was the bias's twin to rounding, under a term too small to tell
    them apart, and read nan. Added as a constant, the ridge outweighed a column logged in units a
    millionth of its own, and set its coefficient near zero. A column whose spread is rounding is
    zeroed and its row and column of the inverse are 0: its coefficient is exactly 0, with no error
    of its own, and the bias carries its level. Kept with a ridge scaled by its mean square, beside
    the bias it repeats, it left the bias an error of the order of the inverse of the ridge."""
    columns = _units.centred(design[:, 1:])
    centred = columns.deviations
    inner = jnp.linalg.inv(centred.T @ centred + ridge * jnp.diag(columns.scales))
    kept = ~columns.rounding
    inner = jnp.where(kept[:, None] & kept[None, :], inner, 0.0)
    return centred, columns.centre, columns.shift, inner


def _solve_ridge(design: Array, target: Array, ridge: float) -> Array:
    centred, centre, shift, inner = _ridge_parts(design, ridge)
    slopes = inner @ (centred.T @ target)
    bias = jnp.mean(target, axis=0) - centre @ slopes - shift @ slopes
    return jnp.concatenate([bias[None], slopes])


def _ridge_weight(design: Array, ridge: float) -> Array:
    """``(X'X + ridge diag(s))^-1 X'``: each row's weight on each coefficient of the solve."""
    centred, centre, shift, inner = _ridge_parts(design, ridge)
    slopes = inner @ centred.T
    bias = 1.0 / design.shape[0] - centre @ slopes - shift @ slopes
    return jnp.concatenate([bias[None], slopes])


def _ridge_inverse(design: Array, ridge: float) -> Array:
    """``(X'X + ridge diag(s))^-1``, put together from the centred columns' own."""
    _, centre, shift, inner = _ridge_parts(design, ridge)
    mean = centre + shift
    corner = 1.0 / design.shape[0] + mean @ inner @ mean
    edge = -(inner @ mean)
    return jnp.block([[corner[None, None], edge[None, :]], [edge[:, None], inner]])


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
    penalty: Array,
    weights: Array | None = None,
) -> Array:
    """Solve the just-identified moment ``Z'(y_res - D c) = 0`` for ``c``, ridge-stabilised: the
    ridge times ``penalty`` added to ``Z'D``. The penalty scales each coefficient to its column's
    size, so the ridge reads the same in any units of the actions and the state.

    ``Z is D`` reduces to ordinary least squares; a different ``Z`` is two-stage least squares.
    Writing it as one solve rather than "regress on the projection" matters: the two agree only for
    a scalar action with no state features, because the in-sample identity ``Z'u = Z'Z`` does not
    survive multiplication by ``phi(x)``.
    """
    if weights is not None:
        instrument = instrument * weights[:, None]
    gram = instrument.T @ regressor + ridge * penalty
    return jnp.linalg.solve(gram, instrument.T @ state_residual)


def _small_sample(rows: int, n_coeff: int, clusters: int | None) -> float:
    """The robust covariance's small-sample factor: ``N / (N - k)`` over rows, and CR1's
    ``G / (G - 1) (N - 1) / (N - k)`` over ``G`` clusters, which is the same where each row is its
    own."""
    if clusters is None:
        return rows / max(rows - n_coeff, 1)
    return clusters / (clusters - 1) * (rows - 1) / max(rows - n_coeff, 1)


def _robust_spreads(
    sensitivity: Array, score: Array, n_coeff: int, clusters: np.ndarray | None = None
) -> tuple[Array, ...]:
    """``sum_i J_i diag(e_i^2) J_i'`` for a fit linear in its target ``y``, ``J = d coeffs / d y``;
    under ``clusters``, ``sum_g s_g s_g'`` with ``s_g = sum_{i in g} J_i e_i``. Two-way, the three
    spreads of :func:`_clustered_squares`, and an error read off them is the largest of their reads.

    Each row's own squared residual stands in for its noise, so it holds when the noise differs
    across rows, and ``J`` runs through the cross-fitted nuisances as well as the moment. A weight
    that loads a few rows loads the nuisance fits' error at them too, which a sandwich on the moment
    alone misses: on a log whose noise grew as ``exp(x)``, weighed by ``exp(-x)``, that sandwich
    came to 0.67 of the channel's spread over sixteen redraws of the noise, and this to 1.04. Summed
    within a cluster first, it holds as well when the scores of one cluster's rows and states
    depend on one another in any way.
    """
    n = score.shape[0]
    if clusters is None:
        scale = _small_sample(n, n_coeff, None)
        return (jnp.einsum("pis,is,qis->pq", sensitivity, scale * score**2, sensitivity),)
    scores = jnp.einsum("pis,is->ip", sensitivity, score)
    factor = _small_sample(n, n_coeff, _cluster_count(clusters))
    if clusters.ndim == 1:
        summed = jax.ops.segment_sum(
            scores, jnp.asarray(clusters), num_segments=int(clusters.max()) + 1
        )
        return (factor * summed.T @ summed,)
    squares = _clustered_squares(np.asarray(scores)[:, None], clusters)
    return tuple(factor * jnp.asarray(square) for square in squares)


def _cluster_count(clusters: np.ndarray) -> int:
    """``G`` in CR1's factor: the clusters' count, or two-way the smaller dimension's, which bounds
    how many independent terms the error is read from (Cameron, Gelbach and Miller 2011)."""
    return int(np.min(clusters.max(axis=0))) + 1


def _clustered_squares(rows: np.ndarray, clusters: np.ndarray | None) -> tuple[np.ndarray, ...]:
    """``rows``' covariance, ``(N, n, ...)`` per transition and state as
    :attr:`CausalDynamicsFit.influence` is: the sum of the outer products of its independent
    terms, each a transition and state, or under ``clusters`` a cluster's transitions and states
    summed.

    Two-way, three covariances, and a reader takes the one it reads largest (MacKinnon, Nielsen
    and Webb 2024): the two dimensions' sums less their intersection's, which both count (Cameron,
    Gelbach and Miller 2011), less any part that reads below nothing (:func:`_less_below_nothing`);
    then each dimension's sums alone, moved from the smaller dimension's CR1 factor, which
    ``rows`` carry, to its own. The two-way sums alone can read less than either dimension does:
    on a log of 3 units over 4 periods they read nothing of a channel affine in the state.
    """
    if clusters is None:
        terms = rows.reshape(-1, *rows.shape[2:])
        return (np.tensordot(terms, terms, axes=(0, 0)),)
    per_row = rows.sum(axis=1)

    def square(codes: np.ndarray) -> np.ndarray:
        summed = np.zeros((int(codes.max()) + 1, *per_row.shape[1:]))
        np.add.at(summed, codes, per_row)
        return np.tensordot(summed, summed, axes=(0, 0))

    if clusters.ndim == 1:
        return (square(clusters),)
    first, second = square(clusters[:, 0]), square(clusters[:, 1])
    cells = square(np.unique(clusters, axis=0, return_inverse=True)[1].reshape(-1))
    spread, every = first + second - cells, first + second + cells
    both = np.maximum(spread, 0.0) if spread.ndim == 0 else _less_below_nothing(spread, every)
    smaller = _cluster_count(clusters)
    alone = []
    for one, codes in ((first, clusters[:, 0]), (second, clusters[:, 1])):
        count = int(codes.max()) + 1
        # the factor whole, so a dimension of the smaller count keeps its sums bit for bit
        alone.append(one * ((count / (count - 1)) / (smaller / (smaller - 1))))
    return (both, *alone)


def _less_below_nothing(spread: np.ndarray, every: np.ndarray) -> np.ndarray:
    """``spread`` less the part it reads below nothing, measured against ``every``, its sums with
    every sign a plus: ``spread - T Q min(L, 0) Q' T'``, ``every = T T'`` and ``Q L Q'`` the
    eigendecomposition of ``T^+ spread T^+'``. ``-every <= spread <= every``, so ``L`` lies in
    ``[-1, 1]``, and a spread with no part below nothing keeps every bit. ``every`` is
    ``spread`` plus twice the cells' sums, and any reference ``a spread + b cells``, the cells'
    sums alone or the units' and the periods' together among them, has the same generalised
    eigenvectors ``v``; each part taken away, ``spread v v' spread / v' spread v``, depends on
    ``v`` alone, so they all take away the same part.

    Both move with the coefficients as the scores do, so a change of their units or of the
    state's zero changes no value read off the result. The eigenvalues of ``spread`` itself, which
    Cameron, Gelbach and Miller (2011) clip, move with them: on a panel of 200 regions over 11
    periods, moving the state's zero moved the channel's error from 0.008164 to 0.009408.
    """
    values, vectors = np.linalg.eigh(every)
    kept = values > values[-1] * values.size * np.finfo(np.float64).eps
    root, whitening = (
        vectors[:, kept] * np.sqrt(values[kept]),
        vectors[:, kept] / np.sqrt(values[kept]),
    )
    shares, turns = np.linalg.eigh(whitening.T @ spread @ whitening)
    lifted = root @ turns
    return spread - (lifted * np.minimum(shares, 0.0)) @ lifted.T


def _error_at_rows(blocks: Array, features: Array) -> float:
    """Root mean square, over the log's rows and the ``G`` blocks, of the standard error of a value
    ``features[i] @ c_g`` whose coefficients ``c_g`` have the covariance ``blocks[g]``:
    ``sqrt(mean_g tr(blocks[g] M))``, ``M`` the features' mean square over the rows.

    A coefficient's own variance moves with the state's zero: the constant term is the value at
    ``x = 0``, however far from it the log was taken. A value at the log's rows does not move.
    """
    moment = features.T @ features / features.shape[0]
    # a covariance two ways read nothing in reads its zero only to rounding, either side of it
    return float(jnp.sqrt(jnp.maximum(jnp.mean(jnp.einsum("gab,ba->g", blocks, moment)), 0.0)))


def _channel_blocks(covariance: Array, channel_shape: tuple[int, ...]) -> Array:
    """``(n m, F, F)``: each channel entry's coefficients' covariance, out of the raveled
    ``(n, m, F)`` channel's."""
    entries, features = channel_shape[0] * channel_shape[1], channel_shape[2]
    every = jnp.arange(entries)
    return covariance.reshape(entries, features, entries, features)[every, :, every, :]


def _influence(
    sensitivity: Array,
    score: Array,
    direct: Array,
    drift_score: Array,
    n_coeff: int,
    clusters: np.ndarray | None = None,
) -> Array:
    """``psi[i, s] = sqrt(n / (n - k)) (J[:, i, s] e[i, s] + D[:, i, s] (r[i, s] - e[i, s]))``,
    shape ``(N, n, p)``: each row's and state's share of the error. ``J`` is the whole fit's
    response to the row's rate and ``D`` the part of it that reaches the drift directly, not
    through the channel, so the channel reads the moment's residual ``e`` and the drift's direct
    part its own regression's residual ``r``, which holds what the drift's features leave. Under
    ``clusters`` the factor is CR1's (:func:`_small_sample`), for sums within each cluster."""
    n = score.shape[0]
    count = None if clusters is None else _cluster_count(clusters)
    return jnp.sqrt(_small_sample(n, n_coeff, count)) * (
        jnp.einsum("pis,is->isp", sensitivity, score)
        + jnp.einsum("pis,is->isp", direct, drift_score - score)
    )


def _representer(sensitivity: Array) -> Array:
    """``(q, N, n)`` weights of the channel's coefficients on each row's rate, as ``(N, n, q)``
    times ``N``, the scale whose mean square is the representer's second moment."""
    return sensitivity.shape[1] * jnp.transpose(sensitivity, (1, 2, 0))


def _kept(residual: Array, scale: Array) -> Array:
    """Unit columns ``(k, r)``, in the units ``scale`` divides out: the combinations of
    ``residual``'s columns that, scaled, keep less than the square root of the working precision."""
    residual = residual * scale
    # A log of fewer transitions than coefficients has fewer singular values than directions, and a
    # thin SVD returns no row for the rest, which no transition moves: zero rows bring them back
    # with a singular value of 0.
    short = residual.shape[1] - residual.shape[0]
    if short > 0:
        residual = jnp.concatenate([residual, jnp.zeros((short, residual.shape[1]))])
    _, singular, rows = jnp.linalg.svd(residual, full_matrices=False)
    null = rows[singular <= jnp.sqrt(jnp.finfo(singular.dtype).eps)].T * scale[:, None]
    return null / jnp.linalg.norm(null, axis=0)


def _unmoved_directions(
    actions: Array, states: Array, covariates: Array, nuisance_degree: int, channel_degree: int
) -> Array:
    """The channel's coefficient directions, unit columns ``(m k, r)``, along which the log's
    actions, less their least-squares projection on the nuisance's features, keep less than the
    square root of the working precision of their raw size.

    Projected without the cross-fit's ridge and folds, a policy the covariates determine leaves
    rounding, where the ridge would leave a bias that grows as the log shrinks. Scaled to the raw
    actions, the test reads the same in any units. An action the log never used has a raw column of
    zeros, and its coefficients come back whole."""
    features = _polynomial_features(_units.standardised(covariates), nuisance_degree)
    left = actions - features @ jnp.linalg.lstsq(features, actions)[0]
    size = jnp.linalg.norm(_channel_design(actions, states, channel_degree), axis=0)
    scale = 1.0 / jnp.where(size > 0.0, size, 1.0)
    return _kept(_channel_design(left, states, channel_degree), scale)


def _instrument_relevance(
    actions: Array,
    states: Array,
    covariates: Array,
    shifter: Array,
    nuisance_degree: int,
    channel_degree: int,
    free: Array | None,
    weights: Array | None,
) -> Array:
    """How far the instrument ``shifter`` moves each direction of the channel: ``(p,)``, along
    ``free``'s ``p`` columns, or every coefficient where it is None. They are the canonical
    correlations, largest first, between the first stage's push on the actions and the actions,
    each less its least-squares projection on the nuisance's features, on the channel's features,
    each row weighed by ``weights``: the singular values of the unregularised moment with each side
    whitened, so the count of those above 0 is its rank.

    The push is the actions' least-squares fit on the first stage's features of the state and the
    instrument, without the ridge. A push of at most ``chc._units.ROUNDING`` eps of the raw
    actions' size along a direction is none, and so is a correlation of at most
    ``chc._units.ROUNDING`` eps: both read 0. Scaled to the raw actions, on features standardised
    as the nuisance's are, and whitened, the reading is the same in any units of the instrument,
    the actions and the state."""
    raw = _channel_design(actions, states, channel_degree)
    size = jnp.linalg.norm(raw, axis=0)
    along = jnp.diag(1.0 / jnp.where(size > 0.0, size, 1.0)) if free is None else free
    correlations = jnp.zeros(along.shape[1], dtype=raw.dtype)
    if along.shape[1] == 0:
        return correlations
    rounding = _units.ROUNDING * jnp.finfo(raw.dtype).eps
    features = _polynomial_features(_units.standardised(covariates), nuisance_degree)
    first = _polynomial_features(
        _units.standardised(jnp.concatenate([states, shifter], axis=1)), nuisance_degree
    )
    root = jnp.ones(raw.shape[0], dtype=raw.dtype) if weights is None else jnp.sqrt(weights)

    def span(columns: Array) -> Array:
        left = columns - features @ jnp.linalg.lstsq(features, columns)[0]
        design = root[:, None] * _channel_design(left, states, channel_degree) @ along
        out, singular, _ = jnp.linalg.svd(design, full_matrices=False)
        return out[:, singular > rounding]

    pushed, moved = span(first @ jnp.linalg.lstsq(first, actions)[0]), span(actions)
    if pushed.shape[1] and moved.shape[1]:
        found = jnp.linalg.svd(pushed.T @ moved, compute_uv=False)
        correlations = correlations.at[: found.size].set(found)
    return jnp.where(correlations > rounding, correlations, 0.0)


def _unmoved_parameters(
    directions: Array, features: Array, actions: Array, design: Array, ridge: float, states: int
) -> Array:
    """Each unmoved direction of the channel, state by state, as a move of every parameter in
    :attr:`CausalDynamicsFit.influence`'s order: the channel's own, and the drift regression's
    response to it on this log."""
    shape = (states, actions.shape[1], features.shape[1])
    moves = []
    for state in range(states):
        for direction in directions.T:
            change = jnp.zeros(shape).at[state].set(direction.reshape(shape[1:]))
            pushed = jax.vmap(lambda c, u, change=change: (change @ c) @ u)(features, actions)
            response = _solve_ridge(design, pushed, ridge)
            moves.append(jnp.concatenate([change.ravel(), -response.ravel()]))
    size = int(np.prod(shape)) + design.shape[1] * states
    return jnp.stack(moves, axis=1) if moves else jnp.zeros((size, 0))


@dataclass(frozen=True)
class _Unmoved:
    """The span of the channel's unmoved directions, split by what the drift regression does with
    their push on the log's raw actions (ADR 0054).

    ``basis`` ``(m k, r)`` spans it and ``free`` ``(m k, m k - r)`` the rest of the channel, the
    columns of both orthonormal once scaled to the raw actions' size, so the split reads the same
    in any units. The channel's moment is solved on ``free`` alone, where it has data. ``taken``
    ``(r, t)`` holds the combinations of ``basis`` whose push the drift's features take up exactly,
    by least squares, so no rate of the log tells them apart, and the fit leaves the channel at
    zero along them. ``pinned`` ``(m k, N)`` reads the rest off a rate, as least squares beside the
    drift's features does: there the log rules out all but one value."""

    raw: Array  # (N, m k): the channel's design on the log's raw actions
    basis: Array
    free: Array
    taken: Array
    pinned: Array

    def hold(self, channel: Array, rate: Array) -> Array:
        """``channel`` ``(n, m, k)``, solved on ``free``, with what the log's ``rate`` ``(N, n)``
        reads along ``basis`` beside the drift added. Linear in both."""
        rows = channel.reshape(channel.shape[0], -1)
        rows = rows + (self.pinned @ (rate - self.raw @ rows.T)).T
        return rows.reshape(channel.shape)


def _split_unmoved(directions: Array, raw: Array, design: Array) -> _Unmoved:
    """:class:`_Unmoved` for the unit columns ``directions`` ``(m k, r)``, with ``raw`` the
    channel's design on the log's raw actions and ``design`` the drift regression's."""
    size = jnp.linalg.norm(raw, axis=0)
    scale = 1.0 / jnp.where(size > 0.0, size, 1.0)
    unit = scale[:, None] * jnp.linalg.qr(directions / scale[:, None], mode="complete")[0]
    basis, free = unit[:, : directions.shape[1]], unit[:, directions.shape[1] :]
    push = raw @ basis
    # in each column's own units, so the rank cutoff falls at one share of every column: on the
    # raw columns, relative to the largest singular value, it dropped a column logged in units far
    # from the others'
    left = push - design @ _units.least_squares(design, push)
    # as in _kept: a log of fewer transitions than directions leaves the rest a zero singular value
    short = left.shape[1] - left.shape[0]
    padded = jnp.concatenate([left, jnp.zeros((short, left.shape[1]))]) if short > 0 else left
    out, singular, rows = jnp.linalg.svd(padded, full_matrices=False)
    seen = singular > jnp.sqrt(jnp.finfo(singular.dtype).eps)
    return _Unmoved(
        raw=raw,
        basis=basis,
        free=free,
        taken=rows[~seen].T,
        pinned=basis @ (rows[seen].T / singular[seen]) @ out[: left.shape[0], seen].T,
    )


def _unmoved_actions(fit: CausalDynamicsFit) -> tuple[int, ...]:
    """The actions whose whole channel the log never moved: every coefficient of theirs, on every
    feature of the channel, in the span of :attr:`CausalDynamicsFit.unmoved`'s directions."""
    states, actions, features = fit.residual.channel.shape
    if fit.unmoved is None or fit.unmoved.shape[1] == 0:
        return ()
    # every state's channel is moved along the same directions, so the first state's block of its
    # first moves holds them all
    width = actions * features
    basis = jnp.linalg.qr(fit.unmoved[:width, : fit.unmoved.shape[1] // states])[0]
    moved = jnp.eye(width) - basis @ basis.T
    precision = float(jnp.sqrt(jnp.finfo(basis.dtype).eps))
    return tuple(
        action
        for action in range(actions)
        if float(jnp.linalg.norm(moved[:, action * features : (action + 1) * features], 2))
        <= precision
    )


@dataclass(frozen=True)
class _LoggedRelations:
    """What the log's actions kept to: orthonormal bases ``(m, r)``, in the actions' own units, of
    the combinations it kept at one level, of those the state alone predicts, and of those its
    covariates predict, each span holding the one before; and each action's least-squares rule of
    the state, the nuisance's polynomial of the state standardised as the log's was."""

    constant: np.ndarray
    state: np.ndarray
    covariates: np.ndarray
    means: np.ndarray  # (m,): a constant combination's level is its weights times these
    # (n,) each: the state standardised as (x - centre - shift) * factor (chc._units.standardising)
    centre: Array
    shift: Array
    factor: Array
    rule: Array  # (features, m)


def _logged_relations(
    actions: Array, states: Array, covariates: Array, nuisance_degree: int
) -> _LoggedRelations:
    """The combinations of the log's actions that a constant, the state or the covariates predict
    to the precision :func:`_unmoved_directions` reads, projected the same way."""
    size = jnp.linalg.norm(actions, axis=0)
    scale = 1.0 / jnp.where(size > 0.0, size, 1.0)

    def kept(features: Array) -> np.ndarray:
        left = actions - features @ jnp.linalg.lstsq(features, actions)[0]
        columns = np.asarray(_kept(left, scale), dtype=np.float64)
        return np.linalg.qr(columns)[0] if columns.shape[1] else columns

    centre, shift, factor = _units.standardising(states)
    features = _polynomial_features((states - centre - shift) * factor, nuisance_degree)
    return _LoggedRelations(
        constant=kept(jnp.ones((actions.shape[0], 1), dtype=actions.dtype)),
        state=kept(features),
        covariates=kept(_polynomial_features(_units.standardised(covariates), nuisance_degree)),
        means=np.asarray(jnp.mean(actions, axis=0), dtype=np.float64),
        centre=centre,
        shift=shift,
        factor=factor,
        rule=jnp.linalg.lstsq(features, actions)[0],
    )


def _absorbed(
    fit: CausalDynamicsFit, states: Array, actions: Array, drift_design: Array
) -> tuple[Array, Array]:
    """The moves of the fit no transition of the log tells from it (ADR 0054): the unmoved channel
    directions (:attr:`CausalDynamicsFit.unmoved`) whose push on the log's own actions the drift
    regression takes up exactly, ``(m k, r)``, and that regression's response, ``(features +
    drivers, r)``.

    The response is by least squares, not the fit's ridge, so each move leaves every predicted rate
    of the log as it was to rounding. A direction whose push the drift cannot take up moves the
    log's own predicted rates, so within the model class the log rules it out, though the channel's
    moment has no data along it, and the fit reads it off the log's rate (:class:`_Unmoved`, the
    same split). Scaled to the push its coefficients would make on actions of the log's raw size,
    the test reads the same in any units."""
    states_n, levers, features = fit.residual.channel.shape
    width = levers * features
    if fit.unmoved is None or fit.unmoved.shape[1] == 0:
        return jnp.zeros((width, 0)), jnp.zeros((drift_design.shape[1], 0))
    raw = _channel_design(actions, states, fit.residual.channel_degree)
    split = _split_unmoved(
        fit.unmoved[:width, : fit.unmoved.shape[1] // states_n], raw, drift_design
    )
    taken = split.basis @ split.taken
    push = raw @ taken
    # A push that cancels to the working precision of its terms is none: its regression would read
    # the rounding as a response, which no action at the plan's points could cancel.
    precision = jnp.sqrt(jnp.finfo(push.dtype).eps)
    size = jnp.linalg.norm(jnp.abs(raw) @ jnp.abs(taken), axis=0)
    none = jnp.linalg.norm(push, axis=0) <= precision * size
    return taken, jnp.linalg.lstsq(drift_design, jnp.where(none, 0.0, push))[0]


def _nuisance_inputs(
    data: dict[str, Array],
    adjust_for: Sequence[str],
    drivers: Sequence[str],
    integrator: Integrator,
) -> tuple[Array, Array, Array, Array]:
    """``(covariates, start, end, read)``: what the nuisances read -- the state, the adjustment
    set, each driver not in it at the transition's start, and every driver at its end -- and the
    drivers at the transition's two ends and as its rate reads them."""
    x = data["x"]
    start = jnp.concatenate([x[:, :0], *[data[name] for name in drivers]], axis=1)
    end = jnp.concatenate([x[:, :0], *[data[f"{name}_next"] for name in drivers]], axis=1)
    # What a transition's rate sees of a driver: its mean over the step, exact for the linear path
    # between the two ends, or its value at the start, which is all the Euler map reads.
    read = start if integrator == "euler" else 0.5 * (start + end)
    unadjusted = [data[name] for name in drivers if name not in adjust_for]
    covariates = jnp.concatenate(
        [x, *[data[name] for name in adjust_for], *unadjusted, end], axis=1
    )
    return covariates, start, end, read


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
    """The fitted value's standard error at the log's rows under ``sigma^2 (X'X)^-1``, the plain
    homoskedastic OLS covariance, root mean square over the rows (:func:`_error_at_rows`).

    Used for the drift stage, which is fitted by least squares on the remainder, so there is no
    instrument and no endogenous regressor to sandwich against. Unlike the channel's robust error it
    pools the noise over the rows and the states; a scale rather than a coverage statement.
    """
    n, n_coeff = design.shape
    score = target - design @ coeffs
    sigma2 = jnp.sum(score**2) / (max(n - n_coeff, 1) * target.shape[1])
    covariance = sigma2 * _ridge_inverse(design, ridge)
    return _error_at_rows(covariance[None], design)


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

    It solves the moment it is handed and checks no rank. Along a direction of the channel that
    ``instrument_action`` does not move, the moment holds at every channel, and the ridge picks
    one. Whether an instrument identifies the channel is the caller's to settle before the solve,
    which sees neither the raw shifter nor the covariates the nuisances took out, and a rank that
    decides it reads both.

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
        ridge: Tikhonov term on the moment's Gram matrix, on each coefficient scaled by its
            regressor column's mean square, weighted as the rows are, so that it reads the same in
            any units of the actions and the state. Added as a constant, it outweighed actions
            logged in units a millionth of their own, and set their channel near zero.
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
    penalty = jnp.diag(_units.mean_squares(regressor, weights))
    coeffs = _channel_coefficients(state_residual, regressor, moment, ridge, penalty, weights)
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
    influence: bool = False,
    clusters: np.ndarray | Array | None = None,
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

            It identifies the channel only where its moment has rank: where the first stage's push
            on the actions moves with the actions along every direction of the channel the actions
            move (all but ``unmoved``'s), each less what the covariates' features predict, read
            without the ridge and in any units (:attr:`CausalDynamicsFit.instrument_relevance`),
            and the actions move one at least. Short of that rank, the moment holds at every
            channel along a direction it misses, and the fit reads as one with no adjustment and
            no instrument: ``identified=False``, ``method="observational"``, the channel of the
            action residuals as their own instrument, kept to compare, and no error. It does so
            beside ``adjust_for`` too: naming an instrument says the covariates leave a
            confounder, so the fit does not fall back on them. Without the check an instrument of
            zeros read identified, the channel set by the moment's ridge and noise: on the
            reference plant ``[0.81, -0.06]`` with an error of 6.0, where the truth is
            ``[1.0, 0.5]``. At ``nuisance_degree=0`` the first stage is a constant, which no
            instrument enters, so no instrument has rank there.

            The rank does not grade an instrument: a column of noise drawn apart from the action
            keeps it, and reads identified with a channel that misses. Its relevance grades it: on
            the reference plant such columns read 0.002 to 0.035, the medians of eight at 500,
            2000 and 8000 rows, of the order of ``1 / sqrt(N)``, where the plant's instrument read
            0.39 to 0.48. No inference here is robust to a weak instrument.
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
            read it up to 0.12.
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
        ridge: the Tikhonov term of the nuisances, the channel's moment, the drift regression and
            the instrument's first stage. The nuisances read standardised covariates. The drift
            regression and the first stage scale each slope's term by its column's variance,
            beside a free bias; the moment, which has no bias, scales each coefficient's by its
            column's mean square, on the log's raw actions. So the ridge reads the same in any
            units of the state, the actions, the drivers and the instrument. Added as a constant,
            it outweighed a column logged in units a millionth of its own: such actions read a
            channel near zero, and such a state a drift slope near zero.
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
        influence: whether to keep each transition's influence on the fitted parameters,
            :attr:`CausalDynamicsFit.influence`, which
            :func:`chc.misspecification.misspecification_cost` reads to compare two fits of one
            log, and beside it the channel's representer, which :func:`omitted_confounder_bound`
            reads with the moment residual. Off by default: under ``euler`` it costs a
            reverse pass per parameter rather than per channel coefficient, and it is ``N n p``
            numbers the fit then carries, and ``N n q`` more.
        clusters: each transition's cluster, ``N`` labels of any one sortable kind, at least two
            distinct. The channel's covariance then sums each cluster's scores over its transitions
            and states before squaring them, CR1 with the factor ``G / (G - 1) (N - 1) / (N - k)``
            over ``G`` clusters and ``k`` coefficients a state, so it holds whatever the dependence
            within a cluster; and a reader of ``influence`` sums its rows the same way. ``None``
            takes every transition's every state as independent of the rest, which a log of units
            followed over time breaks wherever the model leaves out something that persists: the
            transitions of one unit then share it. The estimate does not move; only its error does.
            On panels of 5 to 80 units whose noise persists within each unit, a 5 % test of a zero
            channel rejects 12.5 % to 35.5 % of the time row by row and 3.75 % to 14.75 % by
            unit: CR1 still over-rejects with few clusters (ADR 0053). Two labels a transition,
            shape ``(N, 2)``, cluster it two ways, by its unit and by its period say. The error is
            then the largest of three (MacKinnon, Nielsen and Webb 2024): the two dimensions' sums
            less their intersection's (Cameron, Gelbach and Miller 2011), at CR1's factor for the
            smaller dimension's ``G`` and less any part that reads below nothing; and each
            dimension's sums alone, at its own. A shock every unit shares in a period, met by
            levers every unit moves together, makes one period's scores move together across
            units, which a sum within units leaves out (ADR 0061).

    Returns:
        A :class:`CausalDynamicsFit`. Read ``identified`` before ``residual``.

    Raises:
        ValueError: on a negative ``channel_degree``; on ``clusters`` that do not label every
            transition once or once in each of two dimensions, or name fewer than two clusters in
            a dimension; on an ``rk4`` fixed point that does
            not converge; on a covariate named ``u`` or ``x_next``, an instrument or a driver named
            ``x``, ``u`` or ``x_next``, a driver named twice, or an instrument that is also a
            covariate or a driver, at the step's start or its end. Each would read a column the fit
            reads in another role: adjusted for the action, the channel came back as the
            confounded regression's.
    """
    _refuse_reread("covariate", adjust_for, _ACTION_AND_OUTCOME)
    _refuse_reread("driver", drivers, _TRANSITION)
    twice = sorted({name for name in drivers if drivers.count(name) > 1})
    if twice:
        raise ValueError(
            f"drivers named more than once: {twice}; read twice, a driver's gain is split between "
            "its two copies"
        )
    if instrument is not None:
        _refuse_reread("instrument", (instrument,), _TRANSITION)
        if instrument in {*adjust_for, *drivers, *(f"{name}_next" for name in drivers)}:
            raise ValueError(
                f"the instrument {instrument!r} is also a covariate or a driver, at the step's "
                "start or its end, and partialled out it lends the action no move of its own"
            )
    channel_degree = degree if channel_degree is None else channel_degree
    if channel_degree < 0:
        raise ValueError(f"channel_degree must be a non-negative integer; got {channel_degree}")
    shifter = () if instrument is None else (instrument,)
    ends = tuple(f"{name}_next" for name in drivers)
    data = _real_entries(data, ("x", "u", "x_next", *adjust_for, *drivers, *ends, *shifter))
    x, u, x_next = data["x"], data["u"], data["x_next"]
    codes = None if clusters is None else _cluster_codes(clusters, x.shape[0])
    known_rate = jax.vmap(lambda xi, ui: known(0.0, xi, ui))(x, u)

    identified = bool(adjust_for) or instrument is not None
    covariates, driver_start, driver_end, driver_read = _nuisance_inputs(
        data, adjust_for, drivers, integrator
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
        instrument_action = _units.centred(projected).deviations
        method = "iv"

    # a_θ mops up the rest of y at the fitted channel; see this module's scope note on why that
    # makes the drift observational-conditional while the channel stays interventional. The drivers
    # sit in the same regression, read the way the integrator reads them.
    phi_x = jax.vmap(control_affine_features, in_axes=(0, None))(x, degree)
    phi_c = jax.vmap(control_affine_features, in_axes=(0, None))(x, channel_degree)
    design = jnp.concatenate([phi_x, driver_read], axis=1)

    row_weight = None if weights is None else _state_weights(weights, x)
    raw = _channel_design(u, x, channel_degree)
    # the moment's ridge, scaled to the raw actions' size, as the unmoved directions are
    penalty = jnp.diag(_units.mean_squares(raw, row_weight))
    directions = _unmoved_directions(u, x, covariates, nuisance_degree, channel_degree)
    unmoved = _unmoved_parameters(directions, phi_c, u, design, ridge, x.shape[1])
    # The moment has no data along these directions, and its ridge would set the channel there by
    # the ratio of two roundings, which grows as the square of the actions' units: the fit then
    # missed the log's own rates by ten times their noise in units a thousand times larger.
    split = _split_unmoved(directions, raw, design) if directions.shape[1] else None
    relevance: Array | None = None
    rank: int | None = None
    if instrument is not None:
        relevance = _instrument_relevance(
            u,
            x,
            covariates,
            data[instrument],
            nuisance_degree,
            channel_degree,
            None if split is None else split.free,
            row_weight,
        )
        rank = int(jnp.sum(relevance > 0.0))
        if rank == 0 or rank < relevance.shape[0]:
            # The moment has no rank along a direction the instrument does not move: its ridge and
            # its noise set the channel there, and an instrument of zeros read an exact zero, error
            # zero. Where the log moves no direction, the instrument identifies none.
            identified, method, instrument_action = False, "observational", None

    def moment(y_res: Array, u_res: Array) -> Array:
        regressor = _channel_design(u_res, x, channel_degree)
        instrument = (
            regressor
            if instrument_action is None
            else _channel_design(instrument_action, x, channel_degree)
        )
        if split is None:
            coeffs = _channel_coefficients(y_res, regressor, instrument, ridge, penalty, row_weight)
        else:
            # Only where the moment has data: an unmoved direction's regressor is what the
            # nuisance's ridge left of an action the covariates determine, and solved beside the
            # rest it moved them as far as the actions' units made it.
            coeffs = split.free @ _channel_coefficients(
                y_res,
                regressor @ split.free,
                instrument @ split.free,
                ridge,
                split.free.T @ penalty @ split.free,
                row_weight,
            )
        return coeffs.T.reshape(y_res.shape[1], u_res.shape[1], -1)

    def solve(y: Array) -> tuple[Array, Array, tuple[Array, Array, Array, Array, Array]]:
        """The channel and the drift regression's coefficients a target ``y`` fits to, with the
        residualisation behind them. Linear in ``y``, which the ``rk4`` fixed point relies on."""
        y_res, u_res, y_hat, u_hat = _cross_fit_residuals(
            y, u, covariates, degree=nuisance_degree, folds=folds, ridge=ridge, seed=seed
        )
        channel = moment(y_res, u_res)
        if split is not None:
            channel = split.hold(channel, y)
        fitted = jax.vmap(lambda c, ui: (channel @ c) @ ui)(phi_c, u)
        remainder = _solve_ridge(design, y - fitted, ridge)  # (features + drivers, n)
        return channel, remainder, (y_res, u_res, y_hat, u_hat, fitted)

    def _parameters(solved: tuple[Array, Array, object]) -> Array:
        """The channel, then the drift regression's rows: the order the ``rk4`` fixed point uses."""
        return jnp.concatenate([solved[0].ravel(), solved[1].ravel()])

    def drift_direct(channel_size: int) -> Array:
        """``(p, N, n)``: how a row's rate moves the drift regression's rows other than through the
        channel -- the regression's own weight on the row, in the row's state's column."""
        states = x.shape[1]
        weight = _ridge_weight(design, ridge)  # (features + drivers, N)
        rows = jnp.einsum("fi,ts->ftis", weight, jnp.eye(states))
        rows = rows.reshape(weight.shape[0] * states, x.shape[0], states)
        return jnp.concatenate([jnp.zeros((channel_size, x.shape[0], states)), rows])

    def held(u_res: Array) -> Array:
        """``(p, N, n)``: how a row's rate moves the parameters through the channel's moment alone,
        with the cross-fitted nuisances held, less the drift regression's own weight on the row.
        Letting the nuisances move with the row as well, as ``influence`` does, adds their own
        error to the representer, and so an upward bias of the order of the nuisance features over
        ``N`` to its second moment."""

        def respond(y_res: Array) -> Array:
            channel = moment(y_res, u_res)
            if split is not None:  # the row's rate moves y_res and the rate alike
                channel = split.hold(channel, y_res)
            fitted = jax.vmap(lambda c, ui: (channel @ c) @ ui)(phi_c, u)
            return jnp.concatenate([channel.ravel(), -_solve_ridge(design, fitted, ridge).ravel()])

        return jax.jacrev(respond)(jnp.zeros_like(x_next))

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
            unmoved=unmoved,
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
        # Each parameter in its own unit: the power of two nearest the size of the fit's weight on
        # the log's rates times theirs, which is in the parameter's units whichever state's rates
        # it reads. K's entries are ratios of the parameters' units, so in the caller's units LU
        # compared entries that are not comparable: with the state in millionths and the actions
        # in millions, in float32, the rounding in a drift row outweighed the channel's own entry,
        # LU pivoted on it, and the channel's error read 30 times its own.
        rates = _units.root_mean_square(reading(theta))
        weight = (fit_map * rates).reshape(theta.size, -1).T
        unit = 1.0 / _units.power_of_two(_units.root_mean_square(weight))

        def unitless(matrix: Array, units: Array) -> Array:
            return matrix * units[None, :] / units[:, None]

        tolerance = float(jnp.sqrt(jnp.finfo(theta.dtype).eps))
        for _ in range(_MAX_NEWTON_STEPS):
            gap = jnp.einsum("pis,is->p", fit_map, reading(theta)) - theta
            step = unit * jnp.linalg.solve(unitless(newton_matrix(theta)[0], unit), gap / unit)
            theta = theta + step
            if float(jnp.max(jnp.abs(step / unit))) <= tolerance * float(
                jnp.max(jnp.abs(theta / unit))
            ):
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
        newton = unitless(newton, unit)
        covariances = [
            unit[:, None]
            * jnp.linalg.solve(newton, jnp.linalg.solve(newton, spread / unit[:, None] / unit).T)
            * unit
            for spread in _robust_spreads(fit_map, score, regressor.shape[1], codes)
        ]

        # A row reaches the fixed point through K^-1 G, as the covariance says the noise does.
        def carry(rows: Array) -> Array:
            own = rows.reshape(theta.size, -1) / unit[:, None]
            return (unit[:, None] * jnp.linalg.solve(newton, own)).reshape(rows.shape)

        # The drift keeps the defect, which is its own regression's residual, as under Euler.
        noise = jnp.sum(defect(fit) ** 2, axis=0) / max(x.shape[0] - design.shape[1], 1)
        gram = _ridge_inverse(design, ridge)
        drift_size = theta.size - size
        drift_gained = jnp.einsum(
            "fi,isq->fsq", _ridge_weight(design, ridge), sensitivity[:, :, size:]
        )
        drift_unit = unit[size:]
        drift_newton = unitless(
            jnp.eye(drift_size) - drift_gained.reshape(drift_size, drift_size), drift_unit
        )
        drift_spread = jnp.kron(gram, jnp.diag(noise)) / drift_unit[:, None] / drift_unit
        drift_covariance = (
            drift_unit[:, None]
            * jnp.linalg.solve(drift_newton, jnp.linalg.solve(drift_newton, drift_spread).T)
            * drift_unit
        )
        # the drift's rows are its features, then its drivers, each state's coefficient beside the
        # others': state s's coefficients are every n-th, from s
        rows, states = design.shape[1], x.shape[1]
        every_state = jnp.arange(states)
        drift_blocks = drift_covariance.reshape(rows, states, rows, states)[
            :, every_state, :, every_state
        ]
        return dataclasses.replace(
            fit,
            channel_error=max(
                _error_at_rows(_channel_blocks(covariance[:size, :size], channel.shape), phi_c)
                for covariance in covariances
            ),
            drift_error=_error_at_rows(drift_blocks, design),
            influence=_influence(
                carry(fit_map),
                score,
                carry(drift_direct(size)),
                defect(fit),
                regressor.shape[1],
                codes,
            )
            if influence
            else None,
            representer=_representer(carry(held(u_res))[:size]) if influence else None,
            moment_residual=score,
        )

    fit = fit_to((x_next - x) / dt)
    if integrator == "rk4":
        fit = rk4_fixed_point(fit)
    elif identified:
        y = (x_next - x) / dt - known_rate
        channel, remainder, (y_res, u_res, _, _, fitted) = solve(y)
        regressor = _channel_design(u_res, x, channel_degree)
        score = y_res - regressor @ channel.reshape(x.shape[1], -1).T
        if influence:
            sensitivity = jax.jacrev(lambda target: _parameters(solve(target)))(jnp.zeros_like(y))
        else:
            sensitivity = jax.jacrev(lambda target: solve(target)[0].ravel())(jnp.zeros_like(y))
        spreads = _robust_spreads(sensitivity[: channel.size], score, regressor.shape[1], codes)
        fit = dataclasses.replace(
            fit,
            channel_error=max(
                _error_at_rows(_channel_blocks(spread, channel.shape), phi_c) for spread in spreads
            ),
            influence=_influence(
                sensitivity,
                score,
                drift_direct(channel.size),
                y - fitted - design @ remainder,
                regressor.shape[1],
                codes,
            )
            if influence
            else None,
            representer=_representer(held(u_res)[: channel.size]) if influence else None,
            moment_residual=score,
        )
    return dataclasses.replace(
        fit,
        integrator_defect=float(jnp.sqrt(jnp.mean(defect(fit) ** 2))),
        clusters=codes,
        instrument_relevance=relevance,
        instrument_rank=rank,
    )


def _cluster_codes(clusters: np.ndarray | Array, rows: int) -> np.ndarray:
    """Each transition's cluster as a code from 0, in the labels' sorted order, or two-way a code
    a dimension, ``(N, 2)``."""
    labels = np.asarray(clusters)
    if labels.shape not in ((rows,), (rows, 2)):
        raise ValueError(
            f"clusters label each of the {rows} transitions once, shape ({rows},), or once in each "
            f"of two dimensions, shape ({rows}, 2); got an array of shape {labels.shape}"
        )
    codes = []
    for column in labels.reshape(rows, -1).T:
        distinct, code = np.unique(column, return_inverse=True)
        if distinct.size < 2:
            raise ValueError(
                "clusters name one cluster"
                + ("" if labels.ndim == 1 else " in one of their two dimensions")
                + "; a covariance summed within clusters needs two at least, since one cluster's "
                "sum leaves no spread between clusters to read"
            )
        codes.append(code.astype(np.int64))
    return codes[0] if labels.ndim == 1 else np.column_stack(codes)


# --- MM7: how strong a confounder the adjustment set left out would have to be ---


@dataclass(frozen=True)
class OmittedConfounderBound:
    """How far a confounder the adjustment set left out could move a linear functional of the
    channel, and how strong it would have to be to move it to ``null``. *Experimental.*

    ``bias_scale`` is identified; ``strength`` is the analyst's, as ``Gamma`` is
    (docs/concepts/gamma.md), in partial R^2s rather than odds.
    """

    estimate: float
    # sum over states of sigma nu: the bias at strength 1, DoubleML's ``max_bias``
    bias_scale: float
    strength: float  # |rho| sqrt(cf_y cf_d / (1 - cf_d))
    lower: float  # estimate - strength * bias_scale
    upper: float
    # with the sampling error of the estimate and of bias_scale, each one-sided at ``level``
    ci_lower: float
    ci_upper: float
    # cf_y = cf_d at which the bound reaches ``null``: the design sensitivity, which the confidence
    # bound's own value, below, reaches as the log grows. 1 when no share moves the bound.
    robustness_value: float
    robustness_value_ci: float


def omitted_confounder_bound(
    fit: CausalDynamicsFit,
    functional: Array,
    *,
    cf_y: float,
    cf_d: float,
    rho: float = 1.0,
    level: float = 0.95,
    null: float = 0.0,
) -> OmittedConfounderBound:
    """Bound a linear functional of the channel against a confounder ``adjust_for`` left out.
    *Experimental.*

    Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis (2022) bound the bias of a linear
    functional of a partially linear fit by ``|rho| sqrt(cf_y cf_d / (1 - cf_d)) sigma nu``:
    ``sigma^2`` the rate's residual variance after the fit, ``nu^2`` the second moment of the
    functional's Riesz representer, both identified, and two shares that are not. ``cf_y`` is the
    share of that residual variance the latent explains; ``cf_d`` the share of the long
    representer's second moment the fit's short one misses, which for one lever is the latent's
    partial R^2 with it. ``rho`` is the correlation between the two gaps, 1 at the worst. On a
    linear Gaussian plant with one latent the bound is attained, for any functional of any number
    of levers, and with two latents it is strict unless they move the lever and the rate in
    proportion (``validation/omitted_confounder_bound.mac``). Over several states the bound sums
    ``sigma_s nu_s``, taking the shares as common to them; under Euler a functional of one
    state's channel reads that state alone.

    ``functional`` weighs the channel's coefficients, in its shape ``(n, m, n_features)``. One
    lever's effect on one state is a single 1. Moving spend from lever ``a`` to lever ``b`` is
    ``+1`` at ``b`` and ``-1`` at ``a``, and since its value is linear in the channel, its
    ``robustness_value`` is the confounding at which the move stops paying.

    The sampling error follows DoubleML's: each confidence bound is one-sided at ``level``, and
    carries the estimate's influence and ``bias_scale``'s, whose ``nu^2`` part is ``nu^2 -
    alpha^2`` for a representer ``alpha``. Its sign convention for the estimate differs from this
    one in the cross term of the two influences, which vanishes in expectation. A fit given
    ``clusters`` has both influences summed within each cluster, as its channel's error is. The
    bounds take a ``t``'s quantile with ``G - 1`` degrees of freedom, ``G`` the clusters, two-way
    the smaller dimension's count, or the rows where there are none. At 5 units, a 5 % test of
    the channel read off its error rejected 9.75 % and 14.75 % of panels against the normal's
    quantile, and 3.25 % and 5.75 % against ``t(4)``'s (ADR 0062). DoubleML takes the normal's.

    Args:
        fit: an identified fit of :func:`fit_causal_residual`, by adjustment, unweighted, made
            with ``influence=True``.
        functional: the weight on each channel coefficient.
        cf_y: the share of the residual variance the latent explains, in ``[0, 1)``.
        cf_d: the share of the long representer's second moment the short one misses, in
            ``[0, 1)``.
        rho: the correlation of the two gaps, in ``[-1, 1]``.
        level: each confidence bound's one-sided level, in ``[0.5, 1)``.
        null: the value the robustness values measure the distance to.

    Raises:
        ValueError: on a fit whose channel is not identified by adjustment -- an instrument's
            representer is not the moment's -- or that is weighted, or kept no representer; on a
            functional of another shape; on shares, ``rho`` or ``level`` outside their ranges.
    """
    if fit.method != "orthogonal":
        raise ValueError(
            "the bound is derived for a channel identified by adjustment; this fit's method is "
            f"{fit.method!r}"
        )
    if fit.weighted:
        raise ValueError("the bound is derived for the unweighted moment; this fit is weighted")
    if fit.representer is None or fit.moment_residual is None or fit.influence is None:
        raise ValueError("the fit kept no representer; fit it with influence=True")
    for name, share in (("cf_y", cf_y), ("cf_d", cf_d)):
        if not 0.0 <= share < 1.0:
            raise ValueError(f"{name} is a share of variance and must lie in [0, 1); got {share}")
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho is a correlation and must lie in [-1, 1]; got {rho}")
    if not 0.5 <= level < 1.0:
        raise ValueError(
            f"level is a one-sided confidence level and must lie in [0.5, 1); got {level}"
        )
    channel = np.asarray(fit.residual.channel, dtype=np.float64)
    weights = np.asarray(functional, dtype=np.float64)
    if weights.shape != channel.shape:
        raise ValueError(
            f"the functional weighs a channel of shape {weights.shape}; the fit's is "
            f"{channel.shape}"
        )
    w = weights.ravel()
    estimate = float(w @ channel.ravel())
    alpha = np.asarray(fit.representer, dtype=np.float64) @ w  # (N, n)
    residual = np.asarray(fit.moment_residual, dtype=np.float64)
    psi = np.asarray(fit.influence, dtype=np.float64)[:, :, : w.size] @ w  # the estimate's
    rows = residual.shape[0]
    clusters = fit.clusters
    sigma2, nu2 = np.mean(residual**2, axis=0), np.mean(alpha**2, axis=0)
    product = np.sqrt(sigma2 * nu2)
    scale = float(np.sum(product))
    # d(sigma nu) = (nu^2 d sigma^2 + sigma^2 d nu^2) / (2 sigma nu), in the mean's scaling
    scale_psi = np.divide(
        nu2 * (residual**2 - sigma2) + sigma2 * (nu2 - alpha**2),
        2.0 * product,
        out=np.zeros_like(residual),
        where=product > 0.0,
    )
    # one degree of freedom fewer than the independent terms the spreads sum (ADR 0062)
    count = rows if clusters is None else _cluster_count(clusters)
    quantile = float(student_t.ppf(level, count - 1))

    def bounds(strength: float) -> tuple[float, float, float, float]:
        low, high = estimate - strength * scale, estimate + strength * scale
        moved = strength * scale_psi / rows
        spread_low = float(np.sqrt(max(_clustered_squares(psi - moved, clusters))))
        spread_high = float(np.sqrt(max(_clustered_squares(psi + moved, clusters))))
        return low, high, low - quantile * spread_low, high + quantile * spread_high

    def at_share(share: float) -> float:
        """The strength at ``cf_y = cf_d = share``."""
        return abs(rho) * share / float(np.sqrt(1.0 - share))

    strength = abs(rho) * float(np.sqrt(cf_y * cf_d / (1.0 - cf_d)))
    lower, upper, ci_lower, ci_upper = bounds(strength)

    gap = abs(estimate - null)
    facing = 2 if estimate > null else 3  # the confidence bound on the null's side

    def reach(share: float) -> float:
        """How far the confidence bound facing the null still is from it."""
        bound = bounds(at_share(share))[facing]
        return bound - null if estimate > null else null - bound

    if gap == 0.0:
        robustness, robustness_ci = 0.0, 0.0
    elif abs(rho) * scale == 0.0:  # no share moves the bound; the sampling error alone may
        robustness, robustness_ci = 1.0, (0.0 if reach(0.0) <= 0.0 else 1.0)
    else:
        ratio = (gap / (abs(rho) * scale)) ** 2
        robustness = 0.5 * (float(np.sqrt(ratio**2 + 4.0 * ratio)) - ratio)
        robustness_ci = (
            0.0
            if reach(0.0) <= 0.0
            else float(brentq(reach, 0.0, robustness, xtol=1e-12, rtol=4.0 * np.finfo(float).eps))
        )
    return OmittedConfounderBound(
        estimate=estimate,
        bias_scale=scale,
        strength=strength,
        lower=lower,
        upper=upper,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        robustness_value=robustness,
        robustness_value_ci=robustness_ci,
    )


@dataclass(frozen=True)
class PersistenceCheck:
    """Whether a fit's transition noise persists within units, read off its channel moment's
    residual by :func:`persistence_check`. *Experimental.*

    The fit's premise is Markov: given the state, the levers and the adjustment set, a transition's
    noise does not depend on the past. Where the noise persists within a unit and a lever persists
    too, the state is a common effect of the past lever and the past noise, so given the state this
    period's lever and noise move together, and the channel is biased by an amount its error does
    not cover (ADR 0064).
    """

    correlation: float  # lag-1, pooled over the pairs and the states, each state in its own units
    p_value: float  # two-sided against none, off ``t(G - 1)``; nan below two units
    pairs: int  # transitions whose unit's transition a period earlier is in the log
    units: int  # ``G``: the units with a pair


def persistence_check(
    fit: CausalDynamicsFit, units: ArrayLike, periods: ArrayLike
) -> PersistenceCheck:
    """The lag-1 autocorrelation of ``fit``'s moment residual within units. *Experimental.*

    ``units`` and ``periods`` label the fit's transitions, ``N`` each: the unit, of any one
    sortable kind, and the period the transition starts in, a whole number of steps. Two
    transitions of one unit whose periods differ by one are a pair. Each state's residual is
    divided by its root mean square, so each state counts alike in any units. The pairs' products,
    summed over the states, are summed within each unit before the test, since the transitions of
    one unit need not be independent; the statistic is their total over its CR1 spread, each
    unit's sum less its pairs' share of the total, against ``t(G - 1)`` (ADR 0062).

    A rejection says the premise fails, not by how much the channel is off: that depends on how
    far a lever persists as well (:class:`PersistenceCheck`). Adjusting the fit for the state, the
    levers and the adjustment set a period earlier removes the bias where the noise is AR(1): on
    panels of 200 units over 100 periods whose noise and lever are AR(0.7), a channel of 0.8 read
    0.0130 low, 0.0007 low with them, and 0.0250 low with the state's lag alone (ADR 0064).

    Raises:
        ValueError: the fit kept no moment residual, as where its channel is not identified; the
            labels are not one a transition; a period is not a whole number; or a unit has two
            transitions starting in one period.
    """
    if fit.moment_residual is None:
        raise ValueError("the fit kept no moment residual: its channel is not identified")
    residual = np.asarray(fit.moment_residual, dtype=np.float64)
    labels, steps = np.asarray(units).reshape(-1), np.asarray(periods).reshape(-1)
    if labels.size != residual.shape[0] or steps.size != residual.shape[0]:
        raise ValueError(
            f"units and periods label the fit's {residual.shape[0]} transitions, one each; got "
            f"{labels.size} and {steps.size}"
        )
    if not np.issubdtype(steps.dtype, np.integer):
        raise ValueError(f"periods are whole numbers of steps; got dtype {steps.dtype}")
    codes = np.unique(labels, return_inverse=True)[1].reshape(-1)
    order = np.lexsort((steps, codes))
    earlier, later = order[:-1], order[1:]
    same = codes[earlier] == codes[later]
    if np.any(same & (steps[earlier] == steps[later])):
        raise ValueError("a unit has two transitions starting in one period")
    paired = same & (steps[later] - steps[earlier] == 1)
    earlier, later = earlier[paired], later[paired]
    peak = np.max(np.abs(residual), axis=0)
    live = peak > 0.0
    scaled = residual[:, live] / peak[live]
    scaled = scaled / np.sqrt(np.mean(scaled**2, axis=0))
    products = np.sum(scaled[later] * scaled[earlier], axis=1)
    pairs, count = int(products.size), int(np.unique(codes[later]).size)
    if pairs == 0 or not np.any(live):
        return PersistenceCheck(math.nan, math.nan, pairs, count)
    correlation = float(
        np.sum(products)
        / math.sqrt(float(np.sum(scaled[later] ** 2)) * float(np.sum(scaled[earlier] ** 2)))
    )
    present = np.unique(codes[later])
    sums = np.bincount(codes[later], weights=products)[present]
    # centred, as CR1 centres a regression's scores: uncentred, the statistic stays under
    # sqrt(G - 1), so at six units or fewer it never clears t(G - 1)'s 5 % quantile
    centred = sums - np.bincount(codes[later])[present] * (float(np.sum(products)) / pairs)
    spread = math.sqrt(count / (count - 1) * float(np.sum(centred**2))) if count > 1 else 0.0
    p_value = (
        float(2.0 * student_t.sf(abs(float(np.sum(sums))) / spread, count - 1))
        if spread > 0.0
        else math.nan
    )
    return PersistenceCheck(correlation, p_value, pairs, count)


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
    x = jnp.asarray(_real_numbers(states, "states"), dtype=jnp.float64).ravel()
    u = jnp.asarray(_real_numbers(actions, "actions"), dtype=jnp.float64).ravel()
    y = jnp.asarray(_real_numbers(rates, "rates"), dtype=jnp.float64).ravel()

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
