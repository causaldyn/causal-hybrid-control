"""The tube's reads: its rate, the field's slopes and its error at the plan's states and
RK4 stages, the plant's noise and the sampling error of its read, and the per-log spread."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray
from scipy.special import owens_t
from scipy.stats import chi2, norm
from scipy.stats import t as student_t

from chc._robust_cov import robust_cov
from chc.dynamics import (
    DampedOscillator,
    DrivenDynamics,
    Dynamics,
    HybridDynamics,
    LinearDynamics,
    _EulerStep,
    _SetAtStep,
)
from chc.dynamics_id import CausalDynamicsFit, Integrator, _degrees_of_freedom, _persistence
from chc.integrate import rollout
from chc.lqr import linearize_continuous
from chc.plan import CausalPlan, _certified, _integrator
from chc.residual import ControlAffineResidual, ZeroResidual, control_affine_features
from chc.uncertainty import (
    _TUBE_COVERAGE,
    _chi_square_mixture_survival,
    _linear_tube,
    _noise_covariance,
    _noise_covariances,
    _noise_tube,
    _NoiseRead,
    _read_covariances,
    _step_parts,
    _studentised,
)

if TYPE_CHECKING:
    from chc.decision import TubeRate


_JACKKNIFE_GROUPS = 50  # the most groups of units the tube's noise read is jackknifed over


def _rate(model: Dynamics, x0: Array, u_lo: Array, u_hi: Array) -> tuple[float, TubeRate]:
    """The tube's rate, a norm-Lipschitz bound on the field in the state, and where it holds.

    The norm of the field's slope in the state at the box's centre, plus half each lever's width
    times the norm of what a unit of that lever adds to the slope. A control-affine field's slope
    is affine in the actions, so over the box it is within that sum of the centre's. On a field
    affine in the state (:func:`_slope_free_of_state`) the slope is the same at every state and the
    rate is ``global``; elsewhere it is read at ``x0`` and is ``local``. A log-norm is no such
    bound: it can be negative, which turned the tube's radii negative and certified every step.
    """
    centre = (u_lo + u_hi) / 2.0
    slope, _ = linearize_continuous(model, x0, centre)
    spread = 0.0
    for lever in range(centre.size):
        moved, _ = linearize_continuous(model, x0, centre.at[lever].add(1.0))
        spread += float(u_hi[lever] - u_lo[lever]) / 2.0 * float(jnp.linalg.norm(moved - slope, 2))
    rate = float(jnp.linalg.norm(slope, 2)) + spread
    return rate, "global" if _slope_free_of_state(model) else "local"


def _slope_free_of_state(model: Dynamics) -> bool:
    """Whether the field's slope in the state is the same at every state, by its structure alone:
    affine in the state for each action, a control channel at most affine in it. A drift past
    degree 1, a channel past degree 1, or any field not named here is not taken to be so."""
    if isinstance(model, LinearDynamics | DampedOscillator | ZeroResidual):
        return True
    if isinstance(model, ControlAffineResidual):
        return model.degree <= 1 and model.channel_degree <= 1
    if isinstance(model, HybridDynamics):
        return _slope_free_of_state(model.known) and _slope_free_of_state(model.residual)
    if isinstance(model, DrivenDynamics):
        return _slope_free_of_state(model.dynamics)
    if isinstance(model, _EulerStep):
        return _slope_free_of_state(model.field)
    return False


def _slopes(model: Dynamics, x0: Array) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """The field's slope in the state at each of a plan's actions, the ``(H, n, n)`` matrices the
    tube is carried by on a field affine in the state (:func:`_slope_free_of_state`), read at
    ``x0``, where it is the slope at every state. A control channel affine in the state moves the
    slope with the action, by about 0.04 a unit on the quickstart's fitted field, so each step
    reads its own."""
    slope = jax.vmap(lambda u: jax.jacobian(lambda x: model(0.0, x, u))(x0))

    def along(actions: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(slope(jnp.asarray(actions, dtype=x0.dtype)), dtype=np.float64)

    return along


def _state_noise(
    slopes: NDArray[np.float64],
    noise: NDArray[np.float64],
    persistence: float | NDArray[np.float64],
    dt: float,
    horizon: int,
    read: _NoiseRead | None = None,
    *,
    integrator: Integrator = "rk4",
) -> tuple[tuple[float, ...], ...]:
    """:attr:`DecisionCertificate.tube_state_noise`: each state's 0.95 radius of the noise's part
    of the gap at each step, ``sqrt(chi2_1(0.95) P_k[i, i])``, with the ``read``'s own error
    carried as the tube carries it, by the steps of ``integrator``."""
    level = float(chi2.ppf(_TUBE_COVERAGE, 1))
    return tuple(
        tuple(math.sqrt(level * max(float(variance), 0.0)) for variance in np.diag(covariance))
        for covariance in _read_covariances(
            slopes, noise, persistence, dt, horizon, read, integrator=integrator
        )
    )


def _per_log_spread(kappa: float, radius: float | None = None, read: float = 0.0) -> float:
    """The standard deviation over logs of the share of the plant's paths a radius holds, on one
    state whose noise has standard deviation ``sigma`` and on which each log's fitted path is off
    by an offset of standard deviation ``kappa sigma`` across logs, which all of that log's paths
    share: the field's error, and the noise's mean the fitted drift absorbed.

    ``radius`` is the radius over ``sigma``, ``rho``; by default ``z sqrt(1 + kappa^2)``, at which
    it holds the two together with probability 0.95, ``z`` the normal's two-sided 0.95 quantile.
    One log's offset is ``kappa sigma W``, ``W`` standard normal, so its share is ``C(W) =
    Phi(rho - kappa W) - Phi(-rho - kappa W)``, ``2 Phi(h) - 1`` on average over logs, ``h = rho /
    sqrt(1 + kappa^2)``, spread by

        ``Var C(W) = 4 (2 T(h, 1) - T(h, a) - T(h, 1 / a))``,  ``a = sqrt(1 + 2 kappa^2)``,

    ``T`` Owen's (1956): ``E C(W)^2`` is the chance that two paths of one log both lie within the
    radius, a bivariate normal's at correlation ``kappa^2 / (1 + kappa^2)``, and ``2 T(h, 1) =
    Phi(h) Phi(-h)``. At the default radius ``h = z``: 0 at ``kappa = 0``; 0.0324, 0.0822 and
    0.1394 at ``kappa`` 0.5, 1 and 2; and ``sqrt(0.95 * 0.05) = 0.218`` as ``kappa`` grows, where a
    log's paths all lie within the radius or all without (``validation/tube_per_log_content.mac``,
    ``tube_per_log_content.gp``).

    ``read`` is the standard deviation over logs of ``rho`` itself: each log reads the noise and
    the field's error the radius is made of, each with its own sampling error. With ``rho`` read
    as ``rho + read D``, ``D`` standard normal independent of ``W``, the variance is, by the law of
    total variance, ``E_D Var(C | D) + Var_D E(C | D)``: the closed form's mean over ``D`` and
    the spread of ``2 Phi(h_D) - 1``, both over ``D`` by Gauss-Hermite quadrature at
    :data:`_READ_NODES` nodes (``validation/tube_per_log_read.mac``)."""
    if kappa == 0.0 and read == 0.0:
        return 0.0
    if math.isinf(kappa):
        return math.sqrt(_TUBE_COVERAGE * (1.0 - _TUBE_COVERAGE))
    z = math.sqrt(float(chi2.ppf(_TUBE_COVERAGE, 1)))
    rho = z * math.sqrt(1.0 + kappa * kappa) if radius is None else radius
    a = math.sqrt(1.0 + 2.0 * kappa * kappa)
    if read == 0.0:
        level = rho / math.sqrt(1.0 + kappa * kappa)
        variance = 4.0 * (
            2.0 * float(owens_t(level, 1.0))
            - float(owens_t(level, a))
            - float(owens_t(level, 1.0 / a))
        )
        return math.sqrt(max(variance, 0.0))
    nodes, weights = np.polynomial.hermite_e.hermegauss(_READ_NODES)
    weights = weights / math.sqrt(2.0 * math.pi)
    levels = np.maximum(rho + read * nodes, 0.0) / math.sqrt(1.0 + kappa * kappa)
    within = 4.0 * (2.0 * owens_t(levels, 1.0) - owens_t(levels, a) - owens_t(levels, 1.0 / a))
    means = 2.0 * norm.cdf(levels) - 1.0
    between = means - float(weights @ means)
    variance = float(weights @ within) + float(weights @ (between * between))
    return math.sqrt(max(variance, 0.0))


_READ_NODES = 40  # Gauss-Hermite nodes over a log's read of its radius (_per_log_spread)


def _per_log_spreads(
    field: NDArray[np.float64],
    tube_field: NDArray[np.float64],
    tube: NDArray[np.float64],
    slopes: NDArray[np.float64],
    noise: NDArray[np.float64],
    persistence: NDArray[np.float64],
    dt: float,
    read: _NoiseRead | None,
    degrees: float,
    integrator: Integrator,
) -> tuple[float, ...]:
    """:attr:`DecisionCertificate.tube_per_log_spread`: :func:`_per_log_spread` at each step, on
    one state along the widest direction ``v`` of the noise's part of the gap, ``P_k`` read
    plug-in (:func:`chc.uncertainty._noise_covariances`), ``sigma^2 = v' P_k v``.

    The offset every path of a log shares is the field's error at the plan's own actions,
    ``field`` (:func:`_plan_field`, ``(H + 1,)``), whatever actions the tube was read for, placed
    along ``v`` as the tube places it, and the noise's mean the fitted drift absorbed, of
    covariance ``O_k = D_k V D_k'`` (``read.level``, :func:`chc.uncertainty._read_covariances`):
    ``kappa^2 = (field_k^2 + v' O_k v) / sigma^2``. The radius, ``tube`` (``(H + 1,)``), holds a
    share ``m_k`` of the paths on average over logs, a Gaussian of covariance ``P_k + O_k +
    field_k^2 v v'`` within it (:func:`chc.uncertainty._chi_square_mixture_survival`): 0.95 where
    the tube reads the field's error as it is, more where it reads a bound, as the box's form
    does. On one state that is the radius ``rho = h sqrt(1 + kappa^2)`` over ``sigma``, ``2 Phi(h)
    - 1 = m_k``, which :func:`_per_log_spread` is read at. The radius's own read over logs is
    carried at the share of ``rho`` that the radius's parts move it by to first order: ``z^2 /
    (2 r^2) sqrt(N^2 q + t^4 2 / degrees)``, ``r^2 = (N + v' O_k v + t^2) z^2`` the tube's own
    radius on one state, ``N = c^2 sigma^2`` the noise part at its studentised factor ``c``
    (:func:`chc.uncertainty._studentised`), ``q`` the jackknife variance of ``log sigma^2`` over
    ``read``'s replicates, and ``t`` the field part the tube carried, ``tube_field``, whose
    variance is read on ``degrees`` degrees of freedom. A step whose share rounds to 1 reads 0;
    one with no noise reads ``sqrt(m (1 - m))``, each log's paths all in or all out.

    Against the spread observed over 20 logs on six cells of two plants, Euler and RK4, either
    form, it read 0.7 to 1.4 of it wherever the mean share is below 0.98. Past 0.99, under the
    box's form, it overstates, the plug-in of one log's radius over its own noise read moving a
    share that steep: 0.0074 for 0.0013 at the last step of one cell. Up to 0.24.0 ``kappa`` was
    the tube's field part over ``sigma`` and the radius was taken to hold 0.95 exactly: under the
    box's form, a bound rather than a standard deviation, it read 0.125 there, and it left out the
    radius's own read, which spreads the share where the noise dominates: 0.0005 for 0.0061 at
    the quickstart's third step (ADR 0138)."""
    horizon = field.shape[0] - 1
    covariances = _noise_covariances(slopes, noise, persistence, dt, horizon, integrator=integrator)
    rates = slopes if slopes.ndim == 3 else np.broadcast_to(slopes, (horizon, *slopes.shape))
    eye = np.eye(noise.shape[0])
    offset, offsets = np.zeros_like(noise), [np.zeros_like(noise)]
    for slope in rates[:horizon]:
        offset = _step_parts(dt * np.asarray(slope), eye, integrator)[0] @ offset + eye
        offsets.append(offset @ read.level @ offset.T if read is not None else np.zeros_like(noise))
    replicates = (
        None
        if read is None
        else [
            _noise_covariances(slopes, np.asarray(cov), carry, dt, horizon, integrator=integrator)
            for cov, carry in zip(read.covariances, read.carries, strict=True)
        ]
    )
    spreads = [0.0]
    for step in range(1, horizon + 1):
        values, vectors = np.linalg.eigh(np.atleast_2d(covariances[step]))
        widest = vectors[:, -1]
        variance = max(float(values[-1]), 0.0)
        shifted = max(float(widest @ offsets[step] @ widest), 0.0)
        shared = float(field[step]) ** 2 + shifted
        gap = covariances[step] + offsets[step] + float(field[step]) ** 2 * np.outer(widest, widest)
        spread = np.clip(np.linalg.eigvalsh(gap), 0.0, None)
        if not spread[-1] > 0.0:
            spreads.append(0.0)
            continue
        missed = _chi_square_mixture_survival(
            float(tube[step]) ** 2 / float(spread[-1]), spread / spread[-1]
        )
        if variance == 0.0:
            spreads.append(math.sqrt(max(missed * (1.0 - missed), 0.0)))
            continue
        level = float(norm.isf(missed / 2.0))
        if not math.isfinite(level):
            spreads.append(0.0)
            continue
        logs, scale = 0.0, 1.0
        if replicates is not None:
            sizes = np.array([float(widest @ replicate[step] @ widest) for replicate in replicates])
            scale = float(_studentised(sizes[:, None])[0])
            with np.errstate(divide="ignore", invalid="ignore"):
                centred = np.log(sizes) - np.mean(np.log(sizes))
                logs = (sizes.size - 1) / sizes.size * float(np.sum(centred * centred))
            logs = logs if math.isfinite(logs) else 0.0
        own = scale * scale * variance
        carried = float(tube_field[step]) ** 2
        moved = math.sqrt(own * own * logs + carried * carried * 2.0 / degrees) / (
            2.0 * (own + shifted + carried)
        )
        kappa = math.sqrt(shared / variance)
        rho = level * math.sqrt(1.0 + kappa * kappa)
        spreads.append(_per_log_spread(kappa, rho, rho * moved))
    return tuple(spreads)


def _shared_and_carried(
    field: NDArray[np.float64], scale: float, box: NDArray[np.float64] | None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The field's error at the plan's actions, and the field part the tube carried: the same at
    the tube's studentised ``scale`` under the plan's form, the box's bound ``box`` at it under
    the box's."""
    return field, scale * (field if box is None else box)


def _box_field(
    fit: CausalDynamicsFit,
    reach: NDArray[np.float64],
    plan: CausalPlan,
    state_matrices: NDArray[np.float64],
    drivers: NDArray[np.float64],
    box: tuple[NDArray[np.float64], NDArray[np.float64]],
    dt: float,
    integrator: Integrator,
    stages: _Stages | None,
) -> NDArray[np.float64]:
    """``(H + 1,)``: what the tube carries of the field's error over the levers' box, each step's
    read at its worst vertex, at each of RK4's stages given ``stages`` (:func:`_field_errors`),
    and carried by its gains' norms (:func:`chc.uncertainty._linear_tube`), as
    :func:`prescribe`'s solve carries it."""
    errors = _field_errors(
        fit, reach, np.asarray(plan.trajectory, dtype=np.float64)[:-1], drivers, box, stages
    )
    return _linear_tube(
        state_matrices,
        errors if errors.ndim == 2 else [float(error) for error in errors],
        dt,
        integrator=integrator,
    )


class _FieldJacobian(eqx.Module):
    """The field's gradient in the fit's error ``theta`` at ``(t, x, u)`` (:func:`_plan_field`):
    a module rather than a closure, so that one fit's rollout traces and compiles once whatever
    call reads it."""

    projection: Array
    driven: DrivenDynamics | None
    degree: int = eqx.field(static=True)
    channel_degree: int = eqx.field(static=True)

    def __call__(self, t: Array, x: Array, u: Array) -> Array:
        own = control_affine_features(x, self.degree)
        if self.driven is not None:
            own = jnp.concatenate([own, self.driven.drivers(t)])
        lifted = jnp.outer(u, control_affine_features(x, self.channel_degree)).ravel()
        eye = jnp.eye(self.projection.shape[0])
        channel = jnp.kron(eye, lifted[None, :]) - jnp.einsum("sca,c->sa", self.projection, own)
        return jnp.concatenate([channel, jnp.kron(eye, own[None, :]), eye], axis=1)


class _MovedField(eqx.Module):
    """``field`` with the fitted coefficients off by ``theta``, to first order: ``field(t, x, u) +
    jacobian(t, x, u) theta`` (:func:`_plan_field`)."""

    field: Dynamics
    jacobian: _FieldJacobian
    theta: Array

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return self.field(t, x, u) + self.jacobian(jnp.asarray(t), x, u) @ self.theta


@eqx.filter_jit
def _path_sensitivity(
    inner: Dynamics,
    jacobian: _FieldJacobian,
    start: Array,
    actions: Array,
    dt: float,
    euler: bool,
    zero: Array,
) -> Array:
    """The plan's path differentiated in the fit's error ``theta`` at zero (:func:`_plan_field`),
    compiled once per model's structure: up to 0.24.0 each call compiled its rollout afresh."""

    def path(theta: Array) -> Array:
        moved = _MovedField(inner, jacobian, theta)
        return rollout(_EulerStep(moved) if euler else moved, start, actions, dt)

    return jax.jacfwd(path)(zero)


def _plan_field(
    fit: CausalDynamicsFit, model: Dynamics, start: Array, actions: Array, dt: float
) -> NDArray[np.float64]:
    """``(H + 1,)``: the standard deviation of the gap the fitted field's error opens at each of a
    plan's steps, at the plan's own ``actions``, to first order: ``sqrt(tr(S_k Sigma S_k'))``.

    The fit's error is one vector ``theta = (e, d_1 .. d_n, p)``: the channel's coefficients',
    each state's drift's own and the push's mean (:class:`chc.dynamics_id._FieldTerms`), of
    covariance ``Sigma`` (:meth:`~chc.dynamics_id._FieldTerms.joints`, the largest read at each
    step). It moves the field at ``(x, u)`` by ``H(x, u) theta``, ``H``'s row ``s`` the field's
    gradient in them: ``e_s (x) g(x, u) - projection[s]' phi(x)``, ``phi(x)`` and a one. ``S_k`` is
    the plan's state at step ``k`` differentiated in ``theta`` through ``model``'s own rollout
    (``jax.jacfwd``), RK4 or the Euler map, the field read where each stage reads it, so one
    error's move at every step is carried with its sign, and the steps' errors cancel where the
    actions change sign (Lutkepohl 2005, section 3.5: the estimated coefficients' share of a
    forecast's error). :func:`_field_errors` instead reads each step's error alone, at the box's
    worst vertex, and :func:`chc.uncertainty._linear_tube` adds their gains' norms. Under the
    Euler map that is never less than this at the plan's actions, by Minkowski's inequality, and
    equal on one state where the steps' errors are one (``validation/plan_action_tube.mac``).
    Under RK4 the stages read the error at states the field has moved, a term of order ``dt^2``;
    the box's form reads each stage's error at its own state and carries it by its own weight
    (ADR 0141), and is never less than this either, by the same inequality
    (``validation/rk4_stage_term.mac`` (5)). Up to 0.22.0 it read the step's start alone and could
    be the narrower: on 400 logs of an exact flow, 0.88 of this at the first step (median), its
    misses' root mean square 1.19 standard deviations, this one's 1.04; read at the stages, 1.02
    to 1.04 of this (median; never below 1.001 at any step of three plans) and 1.00 to 1.03."""
    terms = fit._field_terms
    assert terms is not None
    euler = isinstance(model, _EulerStep)
    inner = model.field if isinstance(model, _EulerStep) else model
    driven = inner if isinstance(inner, DrivenDynamics) else None
    jacobian = _FieldJacobian(
        jnp.asarray(terms.projection),
        driven,
        fit.residual.degree,
        fit.residual.channel_degree,
    )
    joints = terms.joints()
    zero = jnp.zeros(joints[0].shape[0], dtype=jnp.result_type(start))
    sensitivity = np.asarray(
        _path_sensitivity(inner, jacobian, start, jnp.asarray(actions), dt, euler, zero),
        dtype=np.float64,
    )
    variances = np.max(
        [np.einsum("kip,pq,kiq->k", sensitivity, joint, sensitivity) for joint in joints], axis=0
    )
    return np.sqrt(np.maximum(variances, 0.0))


def _carried_at_plan(
    plan: CausalPlan,
    fit: CausalDynamicsFit,
    model: Dynamics,
    state_matrices: NDArray[np.float64],
    noise: NDArray[np.float64],
    persistence: NDArray[np.float64],
    dt: float,
    tolerance: float,
    field_scale: float,
    read: _NoiseRead | None,
) -> CausalPlan:
    """``plan`` with its tube's field part read at its own actions (:func:`_plan_field`), at
    ``field_scale`` times itself and along the noise's widest direction, as
    :func:`chc.plan._carried_along` reads the box's (:func:`chc.uncertainty._noise_tube`). The
    tube moves no action."""
    field = _plan_field(fit, model, plan.trajectory[0], plan.actions, dt)
    tube = _noise_tube(
        state_matrices,
        _noise_covariance(noise, plan.trajectory.shape[-1]),
        persistence,
        dt,
        len(plan.actions),
        field_scale * field,
        read,
        integrator=_integrator(model),
    )
    return replace(
        plan,
        uncertainty_tube=jnp.asarray(tube),
        certified_horizon=_certified(tube, tolerance, plan.trajectory),
    )


def _field_error(
    fit: CausalDynamicsFit,
    reach: NDArray[np.float64],
    box: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
) -> float:
    """The fitted field's error over the log, a standard error: the root mean square, over the
    log's rows, of what :func:`_field_errors` reads at each, the largest over the actions of
    ``box``, or of each lever ``j`` within ``reach[j]`` of its mean logged action where no box is
    given. The plan is solved with it.

    Up to 0.17.0 the tube carried the channel's part alone: on the quickstart 0.016, where the
    drift's is 0.039, the demand the fit adjusts for and the plan does not forecast being in the
    drift's residual (ADR 0096). A fit that keeps no ``_field_terms``, as one built without its
    covariances, reads the channel's error at each lever's ``reach`` (:func:`_model_error`) and
    ``drift_error`` in quadrature. A fit with no channel error carries no tube, the drift's error
    with it.

    The tube bounds the gap's Euclidean norm over the states, so each part is the scale of that
    norm: ``drift_error``, a root mean over the states, times the root of their number.
    """
    terms = fit._field_terms
    if terms is not None and fit._lever_errors is not None and fit.drift_error is not None:
        errors = _field_errors(fit, reach, terms.states, terms.drivers, box)
        return float(np.sqrt(np.mean(errors**2)))
    channel = _model_error(fit, reach)
    if fit._lever_errors is None or fit.drift_error is None:
        return channel
    return math.hypot(channel, math.sqrt(fit.residual.channel.shape[0]) * fit.drift_error)


@dataclass(frozen=True)
class _Stages:
    """Where RK4 reads the field over each of a plan's steps (:func:`_field_errors`): ``model``,
    the field the plan is rolled out by, its step ``dt``, and ``drivers``, ``(H, 4, d)``, the
    drivers each stage reads, at the step's start, its middle twice and its end."""

    model: Dynamics
    dt: float
    drivers: NDArray[np.float64]


def _field_errors(
    fit: CausalDynamicsFit,
    reach: NDArray[np.float64],
    states: NDArray[np.float64],
    drivers: NDArray[np.float64],
    box: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    stages: _Stages | None = None,
) -> NDArray[np.float64]:
    """The fitted field's error at each of the plan's ``(H, n)`` states, a standard error: at the
    state each step starts from, ``drivers`` ``(H, d)`` as the drift reads them over that step,
    the largest over the actions of ``box``, ``(lo, hi)``, or where no box is given of each lever
    ``j`` within ``reach[j]`` of its mean logged action.

    Given ``stages``, the plan is rolled out by RK4, and the error is read where each of its four
    stages reads the field, ``(H, 4)``, each at its own state and drivers, under each vertex at the
    stage's state that vertex moves it to, and :func:`chc.uncertainty._linear_tube` carries each
    by its own weight (ADR 0141). One RK4 step of a field affine in the state moves with the
    field's error ``H(x, u) theta`` by ``dt sum_i c_i(z) H(x_i, u) theta``, ``x_i`` its stages'
    states, exactly; read at the step's start alone, as up to 0.22.0, it drops
    ``dt^2 S'(z) e1 f(x, u)`` on one state, the field's move over the step times the error's slope
    in the state ``e1``: of the order of ``dt^2``, not of the terms the ``rk4`` fit's error
    inherits, which every stage reads (``validation/rk4_stage_term.mac``). Each stage read on its
    own bounds the step's error by Minkowski's inequality, within 2 % of it at that file's point,
    where the start's read is 0.86 to 1.48 of it. On 400 logs of an exact flow (``x' = -x/2 + u``,
    ``dt = 0.2``) the first step's misses had a root mean square of up to 1.19 of the start's read
    at a plan's first action, ``+-1.96`` of it holding 0.89 of the logs; read at the stages, 1.00
    to 1.03, holding 0.93 to 0.94, and over a held plan's 15 steps 0.96 to 1.05, holding 0.93 to
    0.955, where the start's read held 0.915 to 0.943.
    Where the channel does not read the state, a stage's state moves affinely with the action and
    its variance is a convex quadratic in it, largest at a vertex; under a channel that does, the
    third stage's variance is of degree 6 in the action (``rk4_stage_term.mac`` (6)), and the
    vertices are read. The Euler map has one stage, the step's start, and reads as before.

    The drift regression reads its target at the fitted channel, so a channel error ``e`` moves the
    field at an action ``u`` by ``e' (g(x, u) - Gamma' phi(x))``: the channel's error times the
    action's distance from the log's projected action ``Gamma' phi(x)``, what the drift's features
    predict of the logged actions, not from their mean (``CausalDynamicsFit._field_terms``, Frisch
    and Waugh 1933). Its square is convex in ``u``, so its largest over the box is at a vertex,
    and the tube reads the field's error there, beside the drift's own and the push's mean, the
    largest over the channel's covariances. Up to 0.21.0 the tube read the channel's error times
    the action's distance from the logged mean and the field's error at the mean, in quadrature,
    which drops ``2 (u - u_bar)' C (u_bar - Gamma' phi)``: beyond the mean, away from the
    projected action, it read up to ``sqrt 2`` too little, toward it too much
    (``validation/tube_projected_action.mac``). On 200 logs of 200 transitions each of the exact
    flows of ``x' = -0.5 x + u + 0.8 z`` at ``dt = 0.2`` and ``x' = -0.7 x + 0.8 u + z`` at
    ``dt = 1``, the logged ``u`` leaning on the state by -0.5, fitted under Euler, at a state 2.5
    of the log's spreads from its mean and the vertex of ``u_bar +- 2`` farther from the projected
    action, the field's miss read 1.18 to 1.19 of the tube's error in root mean square and
    ``+-1.96`` of it held 0.89 to 0.90 of the logs; read from the projected action, 1.04 and 0.93
    to 0.94. Under ``rk4`` the drift reaches the channel through the fixed point: the field's
    error is read through the drift's block of its Newton matrix, and holds the covariance the two
    share, which is not nothing as it is under Euler (``validation/rk4_inherited.mac``). Up to
    0.21.0 an ``rk4`` fit's tube read the drift's own error alone: on 400 logs each of the same
    two flows, at the log's mean action, the miss read 1.53 and 1.88 of it, ``+-1.96`` holding
    0.80 and 0.70 of the logs; now 1.02 and 1.03, holding 0.95 and 0.935, and at the far vertex
    1.04, holding 0.93 to 0.94, where it read 1.40 and 1.87. Given ``periods`` on the first flow
    with an AR(0.8) push, the push's mean read from the periods' cosines (ADR 0133), the miss at
    the mean action reads 1.12 of the tube's error under either integrator, ``+-t_13`` of it
    holding 0.94 of 300 logs, where the push's mean read from the rows held 0.75 under Euler and
    0.55 under ``rk4``.

    A fitted value's error grows with its distance from the log's states, and a plan goes where
    its target is, not where the log was: on 100 one-unit logs of 60 transitions of an AR(1) plant
    whose state spreads 0.52 about 0, a plan to 1 reads the drift's error 1.45 to 1.63 times its
    root mean over the rows from its fourth step on, and the channel's 1.39 to 1.57 times. Read
    over the rows, the field's error the tube carried fell short of the plant's own departure from
    the plan at its late steps (ADR 0104). Each part is the scale of the error's Euclidean norm
    over the states, as :func:`_field_error` reads it (ADR 0100): the variances' sum over them. A
    fit that keeps no ``_field_terms`` reads the channel's error at each lever's ``reach``
    (:func:`_model_errors`) and the drift's at each state, in quadrature; one that keeps no blocks
    reads its root means at every step.
    """
    if fit._lever_errors is None or fit.drift_error is None:
        return _model_errors(fit, reach, states)
    drivers = np.asarray(drivers, dtype=np.float64)
    if fit._field_terms is not None:
        variances = (
            _field_variances(fit, reach, states, drivers, box)
            if stages is None
            else _stage_variances(fit, reach, states, stages, box)
        )
        return np.sqrt(np.maximum(variances, 0.0))
    channel = _model_errors(fit, reach, states)
    count = fit.residual.channel.shape[0]
    if fit._drift_blocks is None:
        return np.hypot(channel, math.sqrt(count) * fit.drift_error)
    features = np.concatenate([_features(states, fit.residual.degree), drivers], axis=1)
    variances = np.mean(np.einsum("kf,gfh,kh->kg", features, fit._drift_blocks, features), axis=1)
    return np.hypot(channel, np.sqrt(count * np.maximum(variances, 0.0)))


# The box's vertices :func:`_field_variances` reads at once: 2 ** 12, so a box of many levers is
# read in blocks of a bounded size rather than all at once.
_CORNERS_AT_ONCE = 4096


def _field_variances(
    fit: CausalDynamicsFit,
    reach: NDArray[np.float64],
    states: NDArray[np.float64],
    drivers: NDArray[np.float64],
    box: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
) -> NDArray[np.float64]:
    """``(H,)``: the field's error variance summed over the states at each of ``states``, the
    largest over the box's vertices and the channel's covariances, read off
    ``fit._field_terms``. With ``L_s(u) = e_s (x) g(x, u) - P_s' phi`` the field's move with the
    channel's error at state ``s``, ``g(x, u) = u (x) psi(x)`` linear in ``u``, the sum is
    ``u' A u + 2 b' u + c`` for each covariance, convex, and read at every vertex."""
    rows = np.atleast_2d(np.asarray(states, dtype=np.float64))
    curvature, linear, constant = _field_quadratic(fit, rows, drivers)
    corners = _box_corners(fit, reach, box)
    largest = np.full(rows.shape[0], -np.inf)
    for start in range(0, corners.shape[0], _CORNERS_AT_ONCE):
        block = corners[start : start + _CORNERS_AT_ONCE]
        values = (
            np.einsum("vi,rkij,vj->rkv", block, curvature, block)
            + 2.0 * np.einsum("vi,rki->rkv", block, linear)
            + constant[:, :, None]
        )
        largest = np.maximum(largest, np.max(values, axis=(0, 2)))
    return largest


def _box_corners(
    fit: CausalDynamicsFit,
    reach: NDArray[np.float64],
    box: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
) -> NDArray[np.float64]:
    """``(V, m)``: the vertices of ``box``, ``(lo, hi)``, or where no box is given of each lever
    ``j`` within ``reach[j]`` of its mean logged action; a lever whose ends meet once."""
    terms = fit._field_terms
    assert terms is not None
    if box is None:
        lo, hi = terms.mean_action - reach, terms.mean_action + reach
    else:
        lo, hi = (np.asarray(end, dtype=np.float64).reshape(-1) for end in box)
    ends = [
        np.unique(np.array([low, high], dtype=np.float64)) for low, high in zip(lo, hi, strict=True)
    ]
    return np.stack(np.meshgrid(*ends, indexing="ij"), axis=-1).reshape(-1, lo.size)


def _field_quadratic(
    fit: CausalDynamicsFit, rows: NDArray[np.float64], drivers: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """The field's error variance summed over the states at each of ``rows`` ``(K, n)`` as a
    quadratic in the action, one a channel's covariance: ``(r, K, m, m)``, ``(r, K, m)`` and
    ``(r, K)``, ``A``, ``b`` and ``c`` of ``u' A u + 2 b' u + c`` (:func:`_field_variances`)."""
    terms = fit._field_terms
    assert terms is not None
    own = np.concatenate([_features(rows, fit.residual.degree), drivers], axis=1)
    basis = _features(rows, fit.residual.channel_degree)
    points, count, levers = rows.shape[0], terms.push.shape[1], terms.mean_action.size
    entries = terms.channel.shape[1] // count
    reads = terms.channel.shape[0]
    channel = terms.channel.reshape(reads, count, entries, count, entries)
    cross = terms.cross.reshape(terms.cross.shape[0], count, entries, count, own.shape[1])
    # an action u moves state s's own channel entries (j, l) by u_j psi_l(x)
    moves = np.einsum("ij,kl->kijl", np.eye(levers), basis).reshape(points, levers, entries)
    # each state's move at the action zero, over every state's channel entries
    offset = -np.einsum("sca,kc->ksa", terms.projection, own).reshape(points, count, count, entries)
    every = np.arange(count)
    pulled = np.einsum("ruatb,kstb->rksua", channel, offset)
    shared = np.einsum("rsasc,kc->rksa", cross, own)
    curvature = np.einsum("kia,srab,kjb->rkij", moves, channel[:, every, :, every, :], moves)
    linear = np.einsum("kia,rksa->rki", moves, pulled[:, :, every, every, :] + shared)
    constant = (
        np.einsum("ksua,rksua->rk", offset, pulled)
        + 2.0 * np.einsum("ksua,ruasc,kc->rk", offset, cross, own)
        + count * np.mean(np.einsum("kc,rgcd,kd->rkg", own, terms.drift, own), axis=2)
        + np.sum(terms.push, axis=1)[:, None]
    )
    return curvature, linear, constant


def _stage_variances(
    fit: CausalDynamicsFit,
    reach: NDArray[np.float64],
    states: NDArray[np.float64],
    stages: _Stages,
    box: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
) -> NDArray[np.float64]:
    """``(H, 4)``: :func:`_field_variances` read where each of RK4's four stages reads the field
    over the step from each of ``states``: at the stage's own state, which moves with the action,
    the largest over the box's vertices and the channel's covariances, each stage on its own."""
    rows = np.atleast_2d(np.asarray(states, dtype=np.float64))
    corners = _box_corners(fit, reach, box)
    moved = _stage_states(stages.model, rows, corners, stages.dt)  # (H, V, 4, n)
    points, size = rows.shape[0], max(1, _CORNERS_AT_ONCE // rows.shape[0])
    largest = np.full((points, moved.shape[2]), -np.inf)
    for start in range(0, corners.shape[0], size):
        block = corners[start : start + size]
        actions = np.tile(block, (points, 1))
        for stage in range(moved.shape[2]):
            curvature, linear, constant = _field_quadratic(
                fit,
                moved[:, start : start + size, stage].reshape(points * block.shape[0], -1),
                np.repeat(stages.drivers[:, stage], block.shape[0], axis=0),
            )
            values = (
                np.einsum("ki,rkij,kj->rk", actions, curvature, actions)
                + 2.0 * np.einsum("ki,rki->rk", actions, linear)
                + constant
            ).reshape(-1, points, block.shape[0])
            largest[:, stage] = np.maximum(largest[:, stage], np.max(values, axis=(0, 2)))
    return largest


def _stage_states(
    model: Dynamics, states: NDArray[np.float64], corners: NDArray[np.float64], dt: float
) -> NDArray[np.float64]:
    """``(H, V, 4, n)``: the states RK4's four stages read ``model``'s field at over the step from
    each of ``states``, step ``k`` at ``t = k dt``, under each of ``corners`` held, as
    :func:`chc.integrate.rk4_step` reads them: the step's start, its two midpoints and its end."""
    rows = jnp.asarray(states)

    def stages(t: Array, x: Array, u: Array) -> Array:
        field, held = model.at_step(x, u) if isinstance(model, _SetAtStep) else (model, u)
        k1 = field(t, x, held)
        second = x + 0.5 * dt * k1
        k2 = field(t + 0.5 * dt, second, held)
        third = x + 0.5 * dt * k2
        k3 = field(t + 0.5 * dt, third, held)
        return jnp.stack([x, second, third, x + dt * k3])

    over = jax.vmap(jax.vmap(stages, in_axes=(None, None, 0)), in_axes=(0, 0, None))
    times = dt * jnp.arange(rows.shape[0], dtype=rows.dtype)
    return np.asarray(over(times, rows, jnp.asarray(corners, dtype=rows.dtype)), dtype=np.float64)


def _model_errors(
    fit: CausalDynamicsFit, reach: NDArray[np.float64], states: NDArray[np.float64]
) -> NDArray[np.float64]:
    """:func:`_model_error` read at each of ``states`` rather than over the log's rows: each lever's
    error is the root mean over its column of the channel's entries of their standard errors at
    the state, from the coefficients' covariances ``_lever_errors`` reads at the rows
    (``CausalDynamicsFit._channel_blocks``), priced at its ``reach`` and summed as
    :func:`_model_error` sums them, the scale of the Euclidean norm over the states. A fit that
    keeps no blocks reads :func:`_model_error` at every state."""
    rows = np.atleast_2d(np.asarray(states, dtype=np.float64))
    if fit._lever_errors is None:
        return np.zeros(rows.shape[0])
    if fit._channel_blocks is None:
        return np.full(rows.shape[0], _model_error(fit, reach))
    levers = fit._lever_errors.shape[1]
    features = _features(rows, fit.residual.channel_degree)
    # (r, H, n m): each covariance's variance of each entry at each state, entry (i, j) at i m + j
    variances = np.einsum("kf,refh,kh->rke", features, fit._channel_blocks, features)
    by_lever = np.sqrt(
        np.maximum(
            np.stack([np.mean(variances[:, :, j::levers], axis=2) for j in range(levers)], 2), 0.0
        )
    )
    rates = by_lever * reach
    count = fit.residual.channel.shape[0]
    return np.max(np.sqrt(count * np.sum(rates * rates, axis=2)), axis=0)


def _features(states: NDArray[np.float64], degree: int) -> NDArray[np.float64]:
    """``(H, F)``: :func:`chc.residual.control_affine_features` of each of ``states``."""
    return np.asarray(
        jax.vmap(control_affine_features, in_axes=(0, None))(jnp.asarray(states), degree),
        dtype=np.float64,
    )


def _field_scale(fit: CausalDynamicsFit) -> float:
    """How many of its standard errors the tube reads :func:`_field_error` at, in units of the
    normal's: ``t_nu(0.975) / z(0.975)``, ``nu`` the degrees of freedom the channel's error is read
    with, the independent terms its spread sums less one, its cosines or its clusters
    (:func:`chc._robust_cov._degrees_of_freedom`, the ``t`` its tests are sized against). Given
    clusters or periods, the drift's own error and the push's mean are read from the same sums or
    cosines as the channel's (ADR 0133), so the field's whole error is read with that count; given
    neither, every part is read from the rows, the drift's own error pooled over them.
    Welch--Satterthwaite's count of the two parts, the drift's at its residual degrees of freedom,
    read the tube shorter on short logs and the same on long ones (ADR 0104).

    The field's error at a step is ``f`` times Student's ``t`` along a direction the fit does not
    give. :func:`chc.plan._error_tube` places it along the noise's widest, where it widens the
    radius most, as a Gaussian of standard deviation ``t / z`` times ``f``: beside the noise, that
    radius holds it with probability 0.9496 to 0.9523 at ``nu`` 3 to 52 over every ratio of the two
    on one state, 0.9500 to 0.9514 on two where it lies along the widest direction, and more where
    it does not: 0.960 and 0.972 along the least, at ``nu`` 3 and 52
    (``validation/tube_level.mac``). The noise's radius plus ``f``, as the tube read it up to
    0.18.0, holds 0.758 at ``nu = 3`` and 0.846 at ``nu = 52`` where the field's error is four
    times the noise's.
    """
    assert fit.defect is not None  # every fit keeps its defect
    degrees = _degrees_of_freedom(int(np.asarray(fit.defect).shape[0]), fit.clusters, fit.periods)
    t_level = float(student_t.ppf(0.5 + _TUBE_COVERAGE / 2.0, degrees))
    return t_level / math.sqrt(float(chi2.ppf(_TUBE_COVERAGE, 1)))


def _model_error(fit: CausalDynamicsFit, reach: NDArray[np.float64]) -> float:
    """Turn the channel's standard error into the per-step rate error the Gronwall tube wants.

    ``channel_error`` is the scale of ``B_hat(x) - B(x)`` per entry at the log's states. Entry
    ``(i, j)`` is in the states' units per unit of lever ``j``, and ``reach[j]``, which
    :func:`prescribe` passes as the largest distance lever ``j``'s box allows from the log's mean
    action, is in lever ``j``'s units: their product is the rate error that action induces, in the
    states' units, whatever lever ``j``'s units are. The distance, not the action: the fit's drift
    absorbs the channel at the log's action, so its error moves the field by the action's distance
    from there, and an origin of a lever moved with its log is the same field (ADR 0091). Up to
    0.17.0 ``reach[j]`` was the largest action from the lever's zero: on a log of one state and two
    levers in ``[-2, 2]``, ``u1``'s zero moved by 100, its log and box with it, ended the tube at
    0.713 where it ends at 0.538, and certified one step where it certifies two.

    The tube bounds the gap's Euclidean norm over the states, and the rate error at an action
    ``d`` is the vector ``(B_hat - B) d``, whose mean square norm is the sum over the entries of
    each one's mean square error times ``d_j^2``, where the entries' errors are uncorrelated. This
    returns its root, ``sqrt(n sum_j (e_j reach_j)^2)`` for ``n`` states, ``e_j`` lever ``j``'s
    read of a covariance ``channel_error`` is read off (``CausalDynamicsFit._lever_errors``), a
    root mean over the states, the largest read taken once the levers are priced (ADR 0087).
    Up to 0.18.0 it was the root mean over the entries, ``sqrt(n m)`` times smaller for ``m``
    levers: over 200 logs of two states and two levers, the field error's root mean square norm
    read 1.93 times its root mean square, where it reads 0.96 times this. With one state and one
    lever it is ``channel_error`` times ``reach``, bit for
    bit; with levers in one unit whose ``reach`` is one number, that times ``sqrt(n m)`` to
    rounding. Up to 0.16.0 it was ``channel_error``
    times the largest action of any lever, which mixed the levers' units: on a log of one state and
    two levers in ``[-2, 2]``, either lever in units 1e3 or 1e-3 times its own, its box with it,
    ended the tube 7 times as wide and certified neither of the two steps it certifies. A lever
    whose ``reach`` is zero, held at the log's mean action, adds no error, and ``reach`` zero for
    every lever leaves the channel nothing to move, and no tube.

    It is a standard error, not coverage: ``channel_error`` is the robust standard error,
    root-mean over the entries and the log's states, calibrated on average but scattering from one
    log to the next, so one fit's tube is as wide as that fit's reading; the tube reads it at the
    quantile of Student's ``t`` at the degrees of freedom it is read with (:func:`_field_scale`).
    The docstring of :class:`chc.dynamics_id.CausalDynamicsFit` says by how much.
    """
    if fit._lever_errors is None:
        return 0.0
    rates = fit._lever_errors * reach
    states = fit.residual.channel.shape[0]
    return float(np.max(np.sqrt(states * np.sum(rates * rates, axis=1))))


def _step_noise(
    fit: CausalDynamicsFit, dt: float, labels: NDArray[Any]
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """The plant's noise as the tube reads it: the plan's one-step error on the log, ``dt`` times
    the fit's defect, its covariance ``S`` in the states' units, the mean of its outer products;
    the VAR(1) coefficient that carries it from step to step (:func:`_autoregression`); and its
    lag-1 correlation within units, pooled over the states as
    :func:`~chc.dynamics_id.persistence_check` pools them, which the report prints. The two are
    nan, unknown, where the log has noise and no unit has two consecutive transitions to read them
    from, and the tube is then not evaluated: up to 0.18.0 it read 0, a tube holding white noise
    where the noise may persist. Where the log has no noise they are 0, which moves nothing.
    ``labels`` are :func:`_transitions`' group, period and unit of each transition. The defect is
    the fit's integrator's, the step the plan rolls out by."""
    assert fit.defect is not None  # every fit keeps its defect
    error = dt * np.asarray(fit.defect, dtype=np.float64)
    covariance = error.T @ error / error.shape[0]
    if not covariance.any():
        return covariance, np.zeros_like(covariance), 0.0
    correlation = _persistence(
        error, labels[:, 2], labels[:, 1], fit.periods is not None
    ).correlation
    return covariance, _autoregression(error, covariance, labels[:, 2], labels[:, 1]), correlation


def _noise_read(fit: CausalDynamicsFit, dt: float, labels: NDArray[Any]) -> _NoiseRead | None:
    """The sampling error of :func:`_step_noise`'s ``S`` and ``Phi``, which the tube carries
    (:func:`chc.uncertainty._read_covariances`): both read again with each of up to
    :data:`_JACKKNIFE_GROUPS` groups of the log's units left out, and the covariance of the
    noise's mean over the log, ``G / (G - 1) sum_g s_g s_g' / N^2``, ``s_g`` unit ``g``'s sum of
    its ``N`` rows' errors, robust to a noise that persists within units. The units are
    independent, so the log's own spread over them is its error. None on a log of one unit, which
    reads no spread over units, and where leaving a group out leaves no pair of consecutive
    periods.

    The fitted drift absorbs the noise's mean on the log, and with it the mean of a confounder the
    fit adjusts for and the plan does not forecast: the plant's noise misses it by that mean's
    error. On the quickstart's plant with its demand an AR(0.8), recorded and adjusted, the plan's
    mean departure from the plant correlated -0.99 with the log's mean demand over 16 logs, 0.010
    at the first step and 0.105 at the fifteenth, where the field's error the tube carries reads
    0.002 and 0.065. Over 16 logs ``S`` read 0.82 to 1.12 times its own on the supply and ``Phi``
    0.757 to 0.832 for 0.796, each unbiased, and a tube that takes them for the plant's held 0.944
    a step. The drift's own error at the log's mean reads the part of the mean's error that is the
    drift's residual, and that part is counted twice: on that plant its variance is 0.5 % of the
    mean's (ADR 0105)."""
    assert fit.defect is not None  # every fit keeps its defect
    error = dt * np.asarray(fit.defect, dtype=np.float64)
    units = np.unique(np.asarray(labels[:, 2]).reshape(-1), return_inverse=True)[1].reshape(-1)
    count = int(units.max()) + 1
    if count < 2:
        return None
    level = robust_cov(error, None, units).cov / error.shape[0] ** 2
    groups = units % min(count, _JACKKNIFE_GROUPS)
    earlier, later = _consecutive(labels[:, 2], labels[:, 1])
    covariances, carries = [], []
    for group in range(int(groups.max()) + 1):
        rows, pairs = groups != group, groups[earlier] != group
        if not pairs.any():
            return None
        kept = error[rows]
        covariance = kept.T @ kept / kept.shape[0]
        lagged = error[later[pairs]].T @ error[earlier[pairs]] / int(pairs.sum())
        covariances.append(covariance)
        carries.append(_yule_walker(lagged, covariance))
    return _NoiseRead(np.stack(covariances), np.stack(carries), level)


def _autoregression(
    error: NDArray[np.float64],
    covariance: NDArray[np.float64],
    units: NDArray[Any],
    periods: NDArray[Any],
) -> NDArray[np.float64]:
    """``Phi``, the ``(n, n)`` coefficient of the VAR(1) the tube carries the one-step ``error`` by,
    ``(N, n)`` rows of ``units`` at whole-number ``periods``: Yule--Walker's ``Gamma_1 S^-1``,
    ``Gamma_1`` the mean of ``w_t w_{t-1}'`` over the pairs of one unit's consecutive periods and
    ``S`` the ``covariance`` of every row, so that ``Cov(w_{k+1}, w_k) = Phi S`` as the log reads
    it, in any units and any basis of the states. Directions ``S`` does not span carry no noise
    and persist at 0. Where the pairs read a lag-1 covariance no stationary noise of covariance
    ``S`` has, its whitened ``S^{-1/2} Gamma_1 S^{-1/2}`` past a contraction, it is scaled back to
    one, the noise persisting fully along its most persistent direction. nan where no unit has two
    consecutive periods.

    A pooled correlation, one ``rho`` carrying ``Cov(w_i, w_j) = rho^|i - j| S``, reads a noise
    persistent on one state and white on another as half as persistent on both: on the
    quickstart's plant with its demand an AR(0.8), each state counted alike in its own units, it
    read 0.41 where the supply's error persists at 0.80 and the wait's at 0.02, and the plant left
    the tube from its second week on, 0.79 a week by the fifteenth (ADR 0105).
    """
    earlier, later = _consecutive(units, periods)
    states = covariance.shape[0]
    if earlier.size == 0:
        return np.full((states, states), math.nan)
    return _yule_walker(error[later].T @ error[earlier] / earlier.size, covariance)


def _consecutive(
    units: NDArray[Any], periods: NDArray[Any]
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """The rows of each pair of one unit's consecutive periods, the earlier's and the later's."""
    codes = np.unique(np.asarray(units).reshape(-1), return_inverse=True)[1].reshape(-1)
    steps = np.asarray(periods).reshape(-1)
    order = np.lexsort((steps, codes))
    earlier, later = order[:-1], order[1:]
    paired = (codes[earlier] == codes[later]) & (steps[later] - steps[earlier] == 1)
    return earlier[paired], later[paired]


def _yule_walker(
    lagged: NDArray[np.float64], covariance: NDArray[np.float64]
) -> NDArray[np.float64]:
    """:func:`_autoregression`'s ``Gamma_1 S^-1`` from the lag-1 covariance ``lagged``, its
    whitened form scaled back to a contraction."""
    values, vectors = np.linalg.eigh(covariance)
    live = values > 1e-12 * float(values[-1])
    basis, roots = vectors[:, live], np.sqrt(values[live])
    whitened = (basis.T @ lagged @ basis) / np.outer(roots, roots)
    norm = float(np.linalg.norm(whitened, 2))
    if norm > 1.0:
        whitened = whitened / norm
    return basis @ (roots[:, None] * whitened / roots[None, :]) @ basis.T
