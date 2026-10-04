"""Evaluating a plan from logs before it is deployed, certified before any cost is read.

:func:`chc.offpolicy.off_policy_value` weights one step at a time, so it estimates a candidate's
value on the logger's own states. A feedback plan on a plant with memory moves the states, and the
value of deploying it is a property of the loop it closes. This module estimates that value, on a
linear-Gaussian loop, and first says whether the logs can:

* :func:`certify_evaluation` reads the model and the logger, never the logs' costs, and returns an
  :class:`EvaluationCertificate`: the exact second moment of the weights a method needs, the
  effective sample size it predicts, and the margin of each condition that keeps it finite.
* :func:`evaluate_plan` certifies, raises :class:`InfeasibleEvaluation` naming the failing
  condition when the certificate refuses, and otherwise returns a :class:`PlanEvaluation`.

Four methods:

* ``"pdis"`` -- per-decision importance sampling over episodes, each step's weights normalised to
  mean one, for the expected cost over the logged horizon from the logs' initial states. The
  second moment of its weight obeys an exact risk-sensitive Riccati recursion along a "starred"
  loop (Jacobson 1973), started from the logs' own initial-state law. The certifiable horizon is
  the longest at which ``n / E_b[W_h^2]`` stays above ``min_effective``. A small-gain test in
  closed form is no substitute: it omits the initial-state law, and a loop that passes it has an
  infinite second moment from ``h = 6`` when it starts from the logger's stationary law.
* ``"mis"`` -- marginalised importance sampling on the stationary state-action laws, for the
  average cost per step. Its weight has a finite second moment iff ``2 Sigma_b^z - Sigma_pi^z > 0``,
  which splits into a state part and an action part. The one-step gate is neither necessary nor
  sufficient for it: a loop for each direction is in the tests.
* ``"dr"`` -- ``"mis"`` with the model's relative value as a control variate, so that its error is
  the product of the weights' error and the model's.
* ``"fqe"`` -- fitted Q evaluation with quadratic features, which is exact on this class. It is
  certified by the restricted chi-square over those features, evaluates a deterministic plan
  directly, and works where every weight has infinite variance.

A plan is an :class:`AffinePolicy`, which holds for ever, or an :class:`AffineSchedule`, whose gain
and offset change with the step and which ends at its horizon; a prescription's schedule is one,
open loop. Only ``"pdis"`` evaluates a schedule: ``"mis"`` and ``"dr"`` read stationary laws and
``"fqe"`` here is average-cost, so each refuses one. The starred loop then changes with the step,
and the recursion carries the law of the state forward, which serves every horizon in one pass
whether the plan changes or not. ``"pdis"`` also scores each step against its own target when the
cost has one per state.

A deterministic plan has no density, so the weights evaluate it smoothed, ``N(K x + k, tau^2 I)``,
and smoothing raises its average cost by exactly ``tau^2 beta``, ``beta = tr(R + B' P B)`` (over a
horizon, ``sum_t tr(R + B' P_{t+1} B)``). The value reported is the deployed plan's: the model's
``tau^2 beta_hat`` is subtracted, and because its error ``tau^2 (beta - beta_hat)`` is first order
in the model's, it is also added to the interval, times ``model_error``. The report names it, and
the share of the weighted estimate it removed. A randomised plan -- one that carries its dither into
deployment -- is evaluated as it is, and nothing is subtracted. The smoothing is chosen against the
cost by predicted mean squared error; to design a logger, sweep its covariance through
:func:`certify_evaluation`, which is cheap and reads no data.

Scope: every number here is exact for ``x' = a x + b u + offset + w`` and an affine plan, and no
more. A fitted CHC plant enters linearised; a receding-horizon LQ controller whose constraints do
not bind is an affine plan. A clipped or saturating logger, a logger whose randomisation is
confounded with the plant's noise, and a plan whose constraints bind are outside the class, and the
certificate does not see them. Stationary effective sample sizes count transitions as if they were
independent, and a slowly mixing logger has fewer; the stationary interval reads its degrees of
freedom off the batches the weights actually used, so it widens where the certificate cannot see.
The intervals are fixed-``n``, not anytime-valid.

What maintained packages already do: SCOPE-RL has the sequential estimators, doubly robust and DICE
among them, for gym environments. What none of them computes is the certificate, which is why this
module exists.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Literal, TypeVar, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import t as student

from chc.cost import QuadraticCost
from chc.independence import GcmTest

_log = logging.getLogger(__name__)

EvaluationMethod = Literal["dr", "mis", "fqe", "pdis"]

_Array = NDArray[np.float64]

_Z95 = 1.959963984540054
_BATCHES = 40  # contiguous batches for a stationary interval under the logs' Markov dependence
_GRID = 200  # geometric grid over (0, tau_max) for the smoothing


def _matrix(value: ArrayLike, name: str, shape: tuple[int, ...]) -> _Array:
    array = np.array(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} is not finite")
    array.setflags(write=False)
    return array


def _psd(value: _Array, name: str) -> None:
    scale = max(1.0, float(np.abs(value).max()))
    if not np.allclose(value, value.T, rtol=0.0, atol=1e-12 * scale):
        raise ValueError(f"{name} is not symmetric")
    if _min_eig(value) < -1e-12 * scale:
        raise ValueError(f"{name} is not positive semidefinite")


@dataclass(frozen=True)
class LinearGaussianPlant:
    """One step of the loop, ``x' = a x + b u + offset + w`` with ``w ~ N(0, noise)``.

    Raises:
        ValueError: on shapes that do not agree, a non-finite entry, or a ``noise`` that is not a
            symmetric positive semidefinite matrix.
    """

    a: NDArray[np.float64]  # (n, n)
    b: NDArray[np.float64]  # (n, m)
    offset: NDArray[np.float64]  # (n,)
    noise: NDArray[np.float64]  # (n, n)

    def __post_init__(self) -> None:
        a = np.asarray(self.a, dtype=np.float64)
        b = np.asarray(self.b, dtype=np.float64)
        if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
            raise ValueError(f"a must be a square matrix, got shape {a.shape}")
        n = a.shape[0]
        if b.ndim != 2 or b.shape[0] != n or b.shape[1] == 0:
            raise ValueError(f"b must have shape ({n}, m), got {b.shape}")
        object.__setattr__(self, "a", _matrix(a, "a", (n, n)))
        object.__setattr__(self, "b", _matrix(b, "b", b.shape))
        object.__setattr__(self, "offset", _matrix(self.offset, "offset", (n,)))
        object.__setattr__(self, "noise", _matrix(self.noise, "noise", (n, n)))
        _psd(self.noise, "noise")

    @property
    def states(self) -> int:
        return self.a.shape[0]

    @property
    def actions(self) -> int:
        return self.b.shape[1]


@dataclass(frozen=True)
class AffinePolicy:
    """``u | x ~ N(gain x + offset, covariance)``; a zero covariance is a deterministic plan.

    Raises:
        ValueError: on shapes that do not agree, a non-finite entry, or a ``covariance`` that is not
            a symmetric positive semidefinite matrix.
    """

    gain: NDArray[np.float64]  # (m, n)
    offset: NDArray[np.float64]  # (m,)
    covariance: NDArray[np.float64]  # (m, m)

    def __post_init__(self) -> None:
        gain = np.asarray(self.gain, dtype=np.float64)
        if gain.ndim != 2 or 0 in gain.shape:
            raise ValueError(f"gain must be an (m, n) matrix, got shape {gain.shape}")
        m = gain.shape[0]
        object.__setattr__(self, "gain", _matrix(gain, "gain", gain.shape))
        object.__setattr__(self, "offset", _matrix(self.offset, "offset", (m,)))
        object.__setattr__(self, "covariance", _matrix(self.covariance, "covariance", (m, m)))
        _psd(self.covariance, "covariance")


@dataclass(frozen=True)
class AffineSchedule:
    """A plan that changes with the step and ends: ``u_t | x_t ~ N(gains[t] x_t + offsets[t],
    covariance)`` for ``t < horizon``; a zero covariance is a deterministic schedule. A
    prescription's schedule is open loop, every gain zero, and :meth:`open_loop` builds one.

    Raises:
        ValueError: on shapes that do not agree, a non-finite entry, or a ``covariance`` that is not
            a symmetric positive semidefinite matrix.
    """

    gains: NDArray[np.float64]  # (H, m, n)
    offsets: NDArray[np.float64]  # (H, m)
    covariance: NDArray[np.float64]  # (m, m)

    def __post_init__(self) -> None:
        gains = np.asarray(self.gains, dtype=np.float64)
        if gains.ndim != 3 or 0 in gains.shape:
            raise ValueError(f"gains must have shape (H, m, n), got {gains.shape}")
        horizon, m, _ = gains.shape
        object.__setattr__(self, "gains", _matrix(gains, "gains", gains.shape))
        object.__setattr__(self, "offsets", _matrix(self.offsets, "offsets", (horizon, m)))
        object.__setattr__(self, "covariance", _matrix(self.covariance, "covariance", (m, m)))
        _psd(self.covariance, "covariance")

    @property
    def horizon(self) -> int:
        return self.gains.shape[0]

    def step(self, t: int) -> AffinePolicy:
        """The plan at step ``t``."""
        return AffinePolicy(self.gains[t], self.offsets[t], self.covariance)

    @classmethod
    def open_loop(
        cls, actions: ArrayLike, states: int, covariance: ArrayLike | None = None
    ) -> AffineSchedule:
        """``u_t = actions[t]`` whatever the state, over ``states`` states: deterministic unless a
        ``covariance`` is given.

        Raises:
            ValueError: on ``actions`` that are not an ``(H, m)`` matrix, ``states`` below one, and
                every error the constructor raises.
        """
        u = np.asarray(actions, dtype=np.float64)
        if u.ndim != 2 or 0 in u.shape or states < 1:
            raise ValueError(
                f"actions must have shape (H, m) and states be at least 1, got {u.shape} and "
                f"{states}"
            )
        horizon, m = u.shape
        spread = np.zeros((m, m)) if covariance is None else np.asarray(covariance, np.float64)
        return cls(np.zeros((horizon, m, states)), u, spread)


@dataclass(frozen=True)
class InitialLaw:
    """``x_0 ~ N(mean, covariance)``, the law the episodes start from; the covariance may be
    singular, and is zero for a fixed start.

    Raises:
        ValueError: on shapes that do not agree, a non-finite entry, or a ``covariance`` that is not
            a symmetric positive semidefinite matrix.
    """

    mean: NDArray[np.float64]  # (n,)
    covariance: NDArray[np.float64]  # (n, n)

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        if mean.ndim != 1 or mean.shape[0] == 0:
            raise ValueError(f"mean must be a vector, got shape {mean.shape}")
        n = mean.shape[0]
        object.__setattr__(self, "mean", _matrix(mean, "mean", (n,)))
        object.__setattr__(self, "covariance", _matrix(self.covariance, "covariance", (n, n)))
        _psd(self.covariance, "covariance")


def fit_logger(x: ArrayLike, u: ArrayLike) -> AffinePolicy:
    """Least-squares fit of ``u ~ N(K x + k, S)``, with a full covariance ``S``.

    Where the propensity was logged at decision time, pass that policy instead: a fitted one is a
    model of the logger, and :func:`chc.offpolicy.fit_behavior_policy` says what its misfit costs.

    Raises:
        ValueError: on ``x`` and ``u`` that are not row-aligned matrices, or no more rows than
            coefficients per action.
    """
    xs, us = np.asarray(x, dtype=np.float64), np.asarray(u, dtype=np.float64)
    if xs.ndim != 2 or us.ndim != 2 or xs.shape[0] != us.shape[0]:
        raise ValueError(
            f"x (N, n) and u (N, m) must be row-aligned, got {xs.shape} and {us.shape}"
        )
    design = np.hstack([xs, np.ones((xs.shape[0], 1))])
    if xs.shape[0] <= design.shape[1]:
        raise ValueError(f"{xs.shape[0]} rows cannot fit {design.shape[1]} coefficients per action")
    coef, *_ = np.linalg.lstsq(design, us, rcond=None)
    residual = us - design @ coef
    covariance = residual.T @ residual / (xs.shape[0] - design.shape[1])
    return AffinePolicy(coef[:-1].T, coef[-1], _sym(covariance))


@dataclass(frozen=True)
class EvaluationCertificate:
    """What the model and the logger say about evaluating a plan, before any cost is read.

    ``log_second_moment`` is ``log(1 + chi^2)`` of the weights the method uses -- for ``"pdis"``
    the trajectory weight's at the logged horizon, for ``"fqe"`` the restricted chi-square over its
    features -- and ``effective_samples`` is ``samples`` divided by ``1 + chi^2``. A margin is a
    smallest eigenvalue, and its condition holds when it is positive. ``one_step_margin`` is the
    one-step gate's, averaged over the logger's stationary states, and for a schedule the smallest
    over its steps. It is reported for comparison only: it is neither necessary nor sufficient for
    any method here, and is ``None`` when the logger's loop has no stationary law.
    """

    method: EvaluationMethod
    certified: bool
    reason: str
    samples: int  # transitions for the stationary methods, episodes for "pdis"
    log_second_moment: float
    effective_samples: float
    ci_reliable: bool  # the fourth Renyi moment is finite, so the interval's variance is consistent
    smoothing: float  # the tau the weights use; 0 for a randomised plan and for "fqe"
    one_step_margin: float | None
    smoothing_limit: float | None = None  # tau_max, for a plan the weights smooth
    state_margin: float | None = None  # 2 R_x - P_x, the stationary methods
    action_margin: float | None = (
        None  # its action block's Schur complement, the stationary methods
    )
    certified_horizon: int | None = None  # "pdis": the longest h min_effective episodes still hold
    escape_horizon: int | None = None  # "pdis": the first h whose second moment is infinite


@dataclass(frozen=True)
class PlanEvaluation:
    """A plan's estimated cost, with its certificate and what the model supplied.

    ``value`` is the average cost per step (stationary methods) or the expected cost over the
    logged horizon from the logs' initial states (``"pdis"``) of the plan as given. For a plan the
    weights smoothed, ``model_correction`` is the model's ``tau^2 beta_hat``, subtracted from the
    weighted estimate and added to the interval's half-width times ``model_error``, and
    ``model_share`` is the share of the weighted estimate it removed; both are 0 otherwise.
    ``effective_samples`` is the weights' own, ``(sum w)^2 / sum w^2``, beside the certificate's
    prediction, and ``None`` for ``"fqe"``. ``degrees_of_freedom`` is the stationary interval's
    Student ``t``: well below 39 when a few of the logger's excursions carry the weights, and
    ``None`` for ``"fqe"`` and ``"pdis"``, whose intervals are normal. From
    :meth:`chc.decision.Prescription.evaluate` the interval is normal too, its spread read from a
    bootstrap over the units the episodes came from (``bootstrap``) rather than across episodes.
    """

    value: float
    interval: tuple[float, float]
    certificate: EvaluationCertificate
    model_correction: float
    model_share: float
    effective_samples: float | None
    degrees_of_freedom: float | None
    # whether the logs' actions read anything but the plan's state: filled by
    # :meth:`chc.decision.Prescription.evaluate`, which knows the columns' names, and None from
    # :func:`evaluate_plan`, which does not
    logger_check: LoggerCheck | None = None
    # the plan against the policy that logged the episodes, and the bootstrap over units both its
    # interval and ``interval`` were read from: filled by
    # :meth:`chc.decision.Prescription.evaluate`, which knows each episode's unit, and None from
    # :func:`evaluate_plan`, which takes its episodes as independent
    versus_logger: LoggerComparison | None = None
    bootstrap: UnitBootstrap | None = None


@dataclass(frozen=True)
class LoggerComparison:
    """The plan against the policy that logged the episodes, on the same episodes.

    ``logger_value`` is the logs' own mean cost over the horizon: the logger's value from the
    episodes' initial states, what deploying the plan would replace. ``difference`` is the plan's
    value less it. ``interval`` is read off the difference within each resample of the units, both
    values computed on the same draw, so what the two share --- which units were drawn, the states
    they started from --- cancels in it; two marginal intervals combined would count it twice. The
    plan's model correction widens it as it widens the plan's own interval.

    A plan's value alone does not say that switching to it helps: the policy in place can do better
    than the plan, and better than any other fixed alternative a trial would compare it with
    (Hernán and Robins, *Causal Inference: What If*, Fine Points 22.7 and 22.8). The difference is
    the switch's own estimate.
    """

    logger_value: float
    difference: float  # the plan's value less the logger's
    interval: tuple[float, float]


@dataclass(frozen=True)
class UnitBootstrap:
    """How the intervals of a :class:`PlanEvaluation` were read: from ``resamples`` draws of the
    units with replacement, every episode of a drawn unit kept, the whole analysis run again on
    each draw.

    ``refused`` counts the draws whose certificate refused; the intervals read the others, so a
    count well above zero says the evaluation sits near the certificate's edge.
    """

    units: int  # the units that contributed an episode
    resamples: int
    refused: int


@dataclass(frozen=True)
class LoggerCheck:
    """Whether the logs' actions read ``given`` and a randomisation of their own alone, tested
    against ``columns`` by :func:`chc.independence.gcm_test`. *Experimental.*

    The weights read the logger's propensity given the plan's state. A logger that also read
    another column before it acted, or the state's past, has another propensity, which the weights
    do not see. ``test.p_value`` is the chance of a dependence this large if it read nothing else.
    A pass is not a confirmation: ``test.detectable`` is the partial correlation each pair could
    have shown. Nor is a rejection a measured cost. Over 400 units and 12 periods, a logger that
    chased demand was flagged on every panel while the estimate's mean error stayed within 0.56 of
    its standard error; one that kept half its last incentive was flagged too, and its estimate sat
    1.13 standard errors high (ADR 0028).
    """

    levers: tuple[str, ...]  # the rows of ``test``'s arrays
    given: tuple[str, ...]  # what the levers and ``columns`` were regressed on
    columns: tuple[str, ...]  # the arrays' columns; a column one period back is ``name[t-1]``
    test: GcmTest


class InfeasibleEvaluation(ValueError):
    """The certificate refused: a weight has infinite variance, or too little mass is left."""

    def __init__(self, certificate: EvaluationCertificate) -> None:
        super().__init__(certificate.reason)
        self.certificate = certificate


# ----------------------------------------------------------------------------------- linear algebra


def _sym(m: _Array) -> _Array:
    return 0.5 * (m + m.T)


def _min_eig(m: _Array) -> float:
    return float(np.min(np.linalg.eigvalsh(_sym(m))))


def _sqrt_psd(m: _Array) -> _Array:
    w, v = np.linalg.eigh(_sym(m))
    return (v * np.sqrt(np.clip(w, 0.0, None))) @ v.T


def _stack_sym(m: _Array) -> _Array:
    return 0.5 * (m + np.swapaxes(m, -1, -2))


def _stack_min_eig(m: _Array) -> _Array:
    return np.min(np.linalg.eigvalsh(_stack_sym(m)), axis=-1)


def _stack_mv(m: _Array, v: _Array) -> _Array:
    """``m[k] @ v[k]`` for every ``k``; a single matrix or vector broadcasts."""
    return np.einsum("...ij,...j->...i", m, v)


def _log_alpha_moment(m1: _Array, s1: _Array, m0: _Array, s0: _Array, alpha: float) -> float:
    """``log E_{N(m0, s0)}[(N(m1, s1) / N(m0, s0))^alpha]``, ``inf`` when it diverges."""
    sa = alpha * s0 + (1.0 - alpha) * s1
    if _min_eig(sa) <= 0.0 or _min_eig(s1) <= 0.0:
        return math.inf
    dm = m1 - m0
    return float(
        (1.0 - alpha) / 2.0 * np.linalg.slogdet(s1)[1]
        + alpha / 2.0 * np.linalg.slogdet(s0)[1]
        - np.linalg.slogdet(sa)[1] / 2.0
        - alpha * (1.0 - alpha) / 2.0 * dm @ np.linalg.solve(sa, dm)
    )


def _log_normal(z: _Array, mean: _Array, covariance: _Array) -> _Array:
    chol = np.linalg.cholesky(covariance)
    scaled = np.linalg.solve(chol, (z - mean).T)
    log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
    return -0.5 * np.sum(scaled * scaled, axis=0) - 0.5 * (
        log_det + z.shape[-1] * math.log(2 * math.pi)
    )


# ------------------------------------------------------------------------------------ closed loops


_Plan = TypeVar("_Plan", AffinePolicy, AffineSchedule)


def _has_density(policy: AffinePolicy | AffineSchedule) -> bool:
    return _min_eig(policy.covariance) > 0.0


def _smoothed(policy: _Plan, tau: float) -> _Plan:
    m = policy.covariance.shape[0]
    return replace(policy, covariance=policy.covariance + tau * tau * np.eye(m))


def _held(plan: AffinePolicy | AffineSchedule, method: EvaluationMethod) -> AffinePolicy:
    """The policy a stationary method evaluates. A schedule changes with the step and ends, so its
    loop has no stationary law and no average cost."""
    if isinstance(plan, AffineSchedule):
        raise ValueError(
            f"{method!r} evaluates a stationary loop's average cost, and a schedule changes with "
            "the step and ends: only 'pdis' evaluates one"
        )
    return plan


def _schedule(plan: AffinePolicy | AffineSchedule, horizon: int) -> AffineSchedule:
    """``plan`` over ``horizon`` steps: a policy held for every one, or a schedule of that many."""
    if isinstance(plan, AffineSchedule):
        if plan.horizon != horizon:
            raise ValueError(
                f"the schedule runs {plan.horizon} steps and the episodes {horizon}: 'pdis' "
                "evaluates a schedule over the logged horizon"
            )
        return plan
    return AffineSchedule(
        np.broadcast_to(plan.gain, (horizon, *plan.gain.shape)),
        np.broadcast_to(plan.offset, (horizon, *plan.offset.shape)),
        plan.covariance,
    )


def _spectral_radius(plant: LinearGaussianPlant, policy: AffinePolicy) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(plant.a + plant.b @ policy.gain))))


def _joint(mx: _Array, sx: _Array, policy: AffinePolicy) -> tuple[_Array, _Array]:
    """The law of ``z = (x, u)`` when ``x ~ N(mx, sx)`` and ``u`` follows ``policy``."""
    k = policy.gain
    sz = np.block([[sx, sx @ k.T], [k @ sx, k @ sx @ k.T + policy.covariance]])
    return np.concatenate([mx, k @ mx + policy.offset]), _sym(sz)


def _stationary(plant: LinearGaussianPlant, policy: AffinePolicy) -> tuple[_Array, _Array, _Array]:
    """Stationary mean and covariance of ``z = (x, u)``, and the state covariance, of a loop whose
    spectral radius is below one."""
    f = plant.a + plant.b @ policy.gain
    drive = plant.b @ policy.covariance @ plant.b.T + plant.noise
    sx = _sym(solve_discrete_lyapunov(f, drive))
    # astype, not a cast: numpy's stubs type solve and cholesky on float input as floating[Any]
    # before 2.5, the numpy Python 3.11 gets, so the float64 promise is made true of the value.
    mx = np.linalg.solve(np.eye(plant.states) - f, plant.b @ policy.offset + plant.offset).astype(
        np.float64, copy=False
    )
    mz, sz = _joint(mx, sx, policy)
    return mz, sz, sx


def _trajectory_log_moments(
    plant: LinearGaussianPlant,
    logger: AffinePolicy,
    plan: AffineSchedule,
    initial: InitialLaw,
    alpha: float,
) -> _Array:
    """``log E_b[W_h^alpha]`` for ``h = 1..plan.horizon``, ``inf`` from the first ``h`` at which it
    diverges.

    Integrating one step's action against ``b (pi_t / b)^alpha`` leaves a Gaussian factor in the
    state and an action law tilted toward the plan; the next state under that law is the "starred"
    loop. The recursion carries the law of the state forward along it, unnormalised: each step's
    factor tilts the law and adds its log mass, which after ``h`` steps is ``log E_b[W_h^alpha]``.
    One pass serves every ``h``, and the loop may change with the step.
    """
    out = np.full(plan.horizon, math.inf)
    sb, sp = logger.covariance, plan.covariance
    gap = alpha * sb - (alpha - 1.0) * sp
    if _min_eig(gap) <= 0.0 or _min_eig(sp) <= 0.0 or _min_eig(sb) <= 0.0:
        return out
    potential = 0.5 * alpha * (alpha - 1.0) * np.linalg.inv(gap)
    log_c0 = float(
        0.5 * alpha * np.linalg.slogdet(sb)[1]
        - 0.5 * (alpha - 1.0) * np.linalg.slogdet(sp)[1]
        - 0.5 * np.linalg.slogdet(gap)[1]
    )
    sbi, spi = np.linalg.inv(sb), np.linalg.inv(sp)
    s_star = np.linalg.inv(alpha * spi - (alpha - 1.0) * sbi)
    v_star = plant.b @ s_star @ plant.b.T + plant.noise
    mean, cov, log_mass = initial.mean, initial.covariance, 0.0
    for t, (gain, offset) in enumerate(zip(plan.gains, plan.offsets, strict=True)):
        d_gain, d_offset = gain - logger.gain, offset - logger.offset
        p = d_gain.T @ potential @ d_gain
        q = d_gain.T @ potential @ d_offset
        # E exp(x' p x + 2 q' x) under the state's law N(mean, cov), and the law it tilts that one
        # into, N(mean + 2 tilt g, tilt) with tilt = (cov^-1 - 2 p)^-1, written through cov's root
        # so that a singular cov, a fixed start's among them, needs no inverse.
        root = _sqrt_psd(cov)
        core = _sym(np.eye(p.shape[0]) - 2.0 * root @ p @ root)
        if _min_eig(core) <= 0.0:
            return out
        tilt = root @ np.linalg.solve(core, root)
        g = p @ mean + q
        log_mass += (
            log_c0
            + float(d_offset @ potential @ d_offset)
            + float(mean @ p @ mean)
            + 2.0 * float(q @ mean)
            + 2.0 * float(g @ tilt @ g)
            - 0.5 * float(np.linalg.slogdet(core)[1])
        )
        out[t] = log_mass
        a_star = plant.a + plant.b @ s_star @ (
            alpha * spi @ gain - (alpha - 1.0) * sbi @ logger.gain
        )
        b_star = (
            plant.b @ s_star @ (alpha * spi @ offset - (alpha - 1.0) * sbi @ logger.offset)
            + plant.offset
        )
        mean = a_star @ (mean + 2.0 * tilt @ g) + b_star
        cov = _sym(a_star @ tilt @ a_star.T + v_star)
    return out


def _smoothed_log_moments(
    plant: LinearGaussianPlant,
    logger: AffinePolicy,
    plan: AffineSchedule,
    initial: InitialLaw,
    alpha: float,
    taus: _Array,
) -> _Array:
    """:func:`_trajectory_log_moments` of ``plan`` smoothed by each of ``taus``, as ``(K, H)``.

    The same recursion with a leading axis over the smoothing, so a grid of ``K`` levels takes
    ``H`` batched steps rather than ``K`` recursions of its own. A level leaves the batch at the
    step its moment diverges, and its row is ``inf`` from there, as the scalar recursion's is.
    """
    n, m = plant.states, plant.actions
    out = np.full((taus.shape[0], plan.horizon), math.inf)
    sb = logger.covariance
    sp = plan.covariance + (taus * taus)[:, None, None] * np.eye(m)
    gap = alpha * sb - (alpha - 1.0) * sp
    if _min_eig(sb) <= 0.0:
        return out
    live = np.flatnonzero((_stack_min_eig(gap) > 0.0) & (_stack_min_eig(sp) > 0.0))
    if live.size == 0:
        return out
    sp, gap = sp[live], gap[live]
    potential = 0.5 * alpha * (alpha - 1.0) * np.linalg.inv(gap)
    log_c0 = (
        0.5 * alpha * np.linalg.slogdet(sb)[1]
        - 0.5 * (alpha - 1.0) * np.linalg.slogdet(sp)[1]
        - 0.5 * np.linalg.slogdet(gap)[1]
    )
    sbi, spi = np.linalg.inv(sb), np.linalg.inv(sp)
    s_star = np.linalg.inv(alpha * spi - (alpha - 1.0) * sbi)
    v_star = plant.b @ s_star @ plant.b.T + plant.noise
    mean = np.broadcast_to(initial.mean, (live.size, n))
    cov = np.broadcast_to(initial.covariance, (live.size, n, n))
    log_mass = np.zeros(live.size)
    for t, (gain, offset) in enumerate(zip(plan.gains, plan.offsets, strict=True)):
        d_gain, d_offset = gain - logger.gain, offset - logger.offset
        p = d_gain.T @ potential @ d_gain
        q = d_gain.T @ potential @ d_offset
        w, v = np.linalg.eigh(_stack_sym(cov))
        root = (v * np.sqrt(np.clip(w, 0.0, None))[:, None, :]) @ np.swapaxes(v, -1, -2)
        core = _stack_sym(np.eye(n) - 2.0 * root @ p @ root)
        keep = _stack_min_eig(core) > 0.0
        if not keep.all():
            live = live[keep]
            if live.size == 0:
                return out
            potential, log_c0, spi, s_star, v_star = (
                potential[keep],
                log_c0[keep],
                spi[keep],
                s_star[keep],
                v_star[keep],
            )
            mean, cov, log_mass = mean[keep], cov[keep], log_mass[keep]
            p, q, root, core = p[keep], q[keep], root[keep], core[keep]
        tilt = root @ np.linalg.solve(core, root)
        g = _stack_mv(p, mean) + q
        log_mass = log_mass + (
            log_c0
            + d_offset @ potential @ d_offset
            + np.einsum("ki,kij,kj->k", mean, p, mean)
            + 2.0 * np.einsum("ki,ki->k", q, mean)
            + 2.0 * np.einsum("ki,kij,kj->k", g, tilt, g)
            - 0.5 * np.linalg.slogdet(core)[1]
        )
        out[live, t] = log_mass
        a_star = plant.a + plant.b @ s_star @ (
            alpha * spi @ gain - (alpha - 1.0) * sbi @ logger.gain
        )
        b_star = (
            _stack_mv(
                plant.b @ s_star,
                alpha * spi @ offset - (alpha - 1.0) * sbi @ logger.offset,
            )
            + plant.offset
        )
        mean = _stack_mv(a_star, mean + 2.0 * _stack_mv(tilt, g)) + b_star
        cov = _stack_sym(a_star @ tilt @ np.swapaxes(a_star, -1, -2) + v_star)
    return out


# ---------------------------------------------------------------------------------- cost and value


@dataclass(frozen=True)
class _StageCost:
    """``cost.running`` as a quadratic in ``z = (x, u)``,
    ``z' matrix z + 2 linear' z + constant``."""

    matrix: _Array
    linear: _Array
    constant: float

    @classmethod
    def of(cls, cost: QuadraticCost, states: int, actions: int) -> _StageCost:
        if np.ndim(cost.x_target) != 1:
            raise ValueError(
                "the cost's x_target has one row per state; a stationary loop is scored against "
                "one target, and only 'pdis' scores each step against its own"
            )
        return cls.steps(cost, states, actions, 1)[0]

    @classmethod
    def steps(
        cls, cost: QuadraticCost, states: int, actions: int, horizon: int
    ) -> tuple[_StageCost, ...]:
        """Each of ``horizon`` steps' cost, against the cost's one target or against the step's
        row of one per state, which :meth:`QuadraticCost.targets` orders."""
        q, r = np.asarray(cost.Q, dtype=np.float64), np.asarray(cost.R, dtype=np.float64)
        # Read here, not through targets(): jax would round a float64 target without x64.
        target = np.asarray(cost.x_target, dtype=np.float64)
        if (
            q.shape != (states, states)
            or r.shape != (actions, actions)
            or target.ndim not in (1, 2)
            or target.shape[-1] != states
        ):
            raise ValueError(
                f"the cost is over {q.shape[0]} states and {r.shape[0]} actions, with x_target of "
                f"shape {target.shape}; the plant is over {states} and {actions}"
            )
        if target.ndim == 2 and target.shape[0] != horizon + 1:
            raise ValueError(
                f"x_target has {target.shape[0]} rows, one per state, but {horizon} steps have "
                f"{horizon + 1} states"
            )
        rows = np.broadcast_to(target, (horizon, states)) if target.ndim == 1 else target[:-1]
        zeros = np.zeros((states, actions))
        matrix = _sym(0.5 * np.block([[q, zeros], [zeros.T, r]]))
        return tuple(
            cls(
                matrix,
                np.concatenate([-0.5 * q @ row, np.zeros(actions)]),
                0.5 * float(row @ q @ row),
            )
            for row in rows
        )

    def __call__(self, z: _Array) -> _Array:
        quadratic = np.einsum("...i,ij,...j->...", z, self.matrix, z)
        return quadratic + 2.0 * z @ self.linear + self.constant

    def mean(self, m: _Array, s: _Array) -> float:
        return float(
            np.trace(self.matrix @ s) + m @ self.matrix @ m + 2.0 * self.linear @ m + self.constant
        )

    def second_moment_about(self, m: _Array, s: _Array, value: float) -> float:
        """``E (c(z) - value)^2`` for ``z ~ N(m, s)``."""
        g = self.matrix @ m + self.linear
        cs = self.matrix @ s
        variance = 2.0 * float(np.trace(cs @ cs)) + 4.0 * float(g @ s @ g)
        return variance + (self.mean(m, s) - value) ** 2


@dataclass(frozen=True)
class _RelativeValue:
    """The plan's relative value, ``h(x) = x' p x + 2 linear' x``, from the average-cost Bellman
    equation of the plant, and its curvature in the action, ``beta = tr(C_uu + b' p b)``:
    smoothing the plan by ``tau^2 I`` costs ``tau^2 beta`` a step. Neither depends on the plan's
    covariance."""

    p: _Array
    linear: _Array
    beta: float

    @classmethod
    def of(cls, plant: LinearGaussianPlant, plan: AffinePolicy, cost: _StageCost) -> _RelativeValue:
        n = plant.states
        cx, cu = cost.matrix[:n, :n], cost.matrix[n:, n:]
        k = plan.gain
        f = plant.a + plant.b @ k
        p = _sym(solve_discrete_lyapunov(f.T, cx + k.T @ cu @ k))
        stage_linear = cost.linear[:n] + k.T @ cu @ plan.offset
        drift = plant.b @ plan.offset + plant.offset
        linear = np.linalg.solve(np.eye(n) - f.T, stage_linear + f.T @ p @ drift).astype(
            np.float64, copy=False
        )
        return cls(p, linear, float(np.trace(cu + plant.b.T @ p @ plant.b)))

    def __call__(self, x: _Array) -> _Array:
        return np.einsum("...i,ij,...j->...", x, self.p, x) + 2.0 * x @ self.linear


def _horizon_beta(plant: LinearGaussianPlant, plan: AffineSchedule, cost: _StageCost) -> float:
    """``sum_t tr(C_uu + b' P_{t+1} b)`` with ``P_horizon = 0``: smoothing's cost over the
    horizon. Only the cost's curvature enters, which every step's target shares."""
    n = plant.states
    cx, cu = cost.matrix[:n, :n], cost.matrix[n:, n:]
    p = np.zeros((n, n))
    total = 0.0
    for k in plan.gains[::-1]:
        total += float(np.trace(cu + plant.b.T @ p @ plant.b))
        f = plant.a + plant.b @ k
        p = f.T @ p @ f + cx + k.T @ cu @ k
    return total


def _episode_value(
    plant: LinearGaussianPlant,
    plan: AffineSchedule,
    costs: tuple[_StageCost, ...],
    initial: InitialLaw,
) -> float:
    """The model's expected cost over the schedule from ``initial``, step ``t`` scored by
    ``costs[t]``."""
    drive = plant.b @ plan.covariance @ plant.b.T + plant.noise
    mx, sx = initial.mean, initial.covariance
    total = 0.0
    for t, cost in enumerate(costs):
        policy = plan.step(t)
        total += cost.mean(*_joint(mx, sx, policy))
        f = plant.a + plant.b @ policy.gain
        mx = f @ mx + plant.b @ policy.offset + plant.offset
        sx = _sym(f @ sx @ f.T + drive)
    return total


# ------------------------------------------------------------------------------------- certificate


def _quadratic_features(z: _Array) -> _Array:
    """``(1, z, z_i z_j for i <= j)``, row-wise."""
    i, j = np.triu_indices(z.shape[-1])
    return np.concatenate([np.ones((*z.shape[:-1], 1)), z, z[..., i] * z[..., j]], axis=-1)


def _feature_moments(mean: _Array, cov: _Array) -> tuple[_Array, _Array]:
    """``E phi`` and ``E phi phi'`` of :func:`_quadratic_features` for ``z ~ N(mean, cov)``,
    exactly (Isserlis, to the fourth moment)."""
    mu, s = mean, cov
    m2 = s + np.outer(mu, mu)
    m3 = (
        np.einsum("i,j,k->ijk", mu, mu, mu)
        + np.einsum("i,jk->ijk", mu, s)
        + np.einsum("j,ik->ijk", mu, s)
        + np.einsum("k,ij->ijk", mu, s)
    )
    m4 = (
        np.einsum("i,j,k,l->ijkl", mu, mu, mu, mu)
        + np.einsum("i,j,kl->ijkl", mu, mu, s)
        + np.einsum("i,k,jl->ijkl", mu, mu, s)
        + np.einsum("i,l,jk->ijkl", mu, mu, s)
        + np.einsum("j,k,il->ijkl", mu, mu, s)
        + np.einsum("j,l,ik->ijkl", mu, mu, s)
        + np.einsum("k,l,ij->ijkl", mu, mu, s)
        + np.einsum("ij,kl->ijkl", s, s)
        + np.einsum("ik,jl->ijkl", s, s)
        + np.einsum("il,jk->ijkl", s, s)
    )
    d = mu.shape[0]
    iu, ju = np.triu_indices(d)
    first = np.concatenate([[1.0], mu, m2[iu, ju]])
    second = np.empty((first.shape[0], first.shape[0]))
    second[0], second[:, 0] = first, first
    second[1 : 1 + d, 1 : 1 + d] = m2
    second[1 : 1 + d, 1 + d :] = m3[:, iu, ju]
    second[1 + d :, 1 : 1 + d] = m3[:, iu, ju].T
    second[1 + d :, 1 + d :] = m4[iu, ju][:, iu, ju]
    return first, second


def _smoothing_limit(sb: _Array, s0: _Array, s1: _Array) -> float:
    """``tau_max``, with ``2 sb - (s0 + tau^2 (s1 - s0)) > 0`` exactly on ``(0, tau_max)``.

    The plan's stationary covariance is linear in ``tau^2``, and its growth ``s1 - s0`` has the
    identity in its action block, so ``tau_max`` is finite, and 0 when ``2 sb - s0`` is not
    positive definite."""
    room = _sym(2.0 * sb - s0)
    if _min_eig(room) <= 0.0:
        return 0.0
    chol = np.linalg.cholesky(room)
    scaled = np.linalg.solve(chol, np.linalg.solve(chol, _sym(s1 - s0)).T).astype(
        np.float64, copy=False
    )
    return 1.0 / math.sqrt(float(np.max(np.linalg.eigvalsh(_sym(scaled)))))


def _schur_margins(
    rx: _Array, px: _Array, logger: AffinePolicy, target: AffinePolicy
) -> tuple[float, float]:
    """``2 Sigma_b^z - Sigma_pi^z > 0`` split into its state block, ``2 R_x - P_x``, and the
    action block's Schur complement, ``2 S_b - S_pi - D (P_x + P_x (2 R_x - P_x)^-1 P_x) D'``."""
    state_block = _sym(2.0 * rx - px)
    state = _min_eig(state_block)
    if state <= 0.0:
        return state, -math.inf
    d = target.gain - logger.gain
    kernel = px + px @ np.linalg.solve(state_block, px)
    return state, _min_eig(2.0 * logger.covariance - target.covariance - d @ kernel @ d.T)


def _one_step_margin(rx: _Array, logger: AffinePolicy, target: AffinePolicy) -> float:
    """``2 S_b - S_pi - 2 D R_x D'``: the one-step weight's second moment, averaged over the
    logger's stationary states, is finite iff it is positive definite."""
    d = target.gain - logger.gain
    return _min_eig(2.0 * logger.covariance - target.covariance - 2.0 * d @ rx @ d.T)


def _tilted_second_moment(
    mp: _Array, sp: _Array, mb: _Array, sb: _Array, cost: _StageCost, value: float
) -> float:
    """``E_b[w^2 (c - value)^2] / E_b[w^2]``: the cost's second moment under the law the squared
    weight tilts the logger's toward, ``N(mq, (2 sp^-1 - sb^-1)^-1)``."""
    sq = _sym(np.linalg.inv(2.0 * np.linalg.inv(sp) - np.linalg.inv(sb)))
    mq = sq @ (2.0 * np.linalg.solve(sp, mp) - np.linalg.solve(sb, mb))
    return cost.second_moment_about(mq, sq, value)


def _grid(limit: float) -> _Array:
    """The geometric grid over ``(0, limit)`` the smoothing is chosen on."""
    return np.geomspace(1e-4 * limit, 0.999 * limit, _GRID)


def _grid_minimum(grid: _Array, scores: _Array) -> float | None:
    """The grid point of the first smallest score; ``None`` when no score is finite."""
    best = int(np.argmin(scores))
    return float(grid[best]) if math.isfinite(scores[best]) else None


def _best_on_grid(limit: float, score: Callable[[float], float]) -> float | None:
    """The minimiser of ``score`` on a geometric grid over ``(0, limit)``; ``None`` when it is
    nowhere finite."""
    grid = _grid(limit)
    return _grid_minimum(grid, np.array([score(float(tau)) for tau in grid]))


def _check_policies(plant: LinearGaussianPlant, **policies: AffinePolicy | AffineSchedule) -> None:
    shape = (plant.actions, plant.states)
    for name, policy in policies.items():
        gain = policy.gain.shape if isinstance(policy, AffinePolicy) else policy.gains.shape[1:]
        if gain != shape:
            raise ValueError(f"the {name}'s gain has shape {gain}, the plant's loop needs {shape}")


def certify_evaluation(
    plant: LinearGaussianPlant,
    logger: AffinePolicy,
    plan: AffinePolicy | AffineSchedule,
    method: EvaluationMethod,
    samples: int,
    *,
    smoothing: float | None = None,
    horizon: int | None = None,
    initial: InitialLaw | None = None,
    min_effective: float = 100.0,
) -> EvaluationCertificate:
    """Whether ``samples`` logs of ``logger`` on ``plant`` can evaluate ``plan`` by ``method``.

    Reads the model and the two policies, and nothing the plan will be scored on. ``samples``
    counts transitions for the stationary methods and episodes for ``"pdis"``, which also needs
    the logged ``horizon``, a schedule's own by default, and the episodes' ``initial`` law. A plan
    without a density is evaluated by the weights smoothed by ``smoothing``, and by default by the
    ``tau`` at which the logs support the most effective samples; :func:`evaluate_plan` chooses its
    own against the cost. The certificate refuses when a weight the method needs has infinite
    variance, or leaves fewer than ``min_effective`` effective samples (``"pdis"``: at the logged
    horizon).

    Raises:
        ValueError: on an unknown method, a policy whose shape does not fit the plant, a schedule
            for a method other than ``"pdis"`` or over another horizon than the episodes', a
            ``samples`` or ``min_effective`` below one, ``horizon`` and ``initial`` missing for
            ``"pdis"`` or passed to another method, or a ``smoothing`` that is not positive, or is
            passed for ``"fqe"`` or a plan with a density.
    """
    if method not in get_args(EvaluationMethod):
        raise ValueError(f"method must be one of {get_args(EvaluationMethod)}, got {method!r}")
    _check_policies(plant, logger=logger, plan=plan)
    if samples < 1 or not min_effective >= 1.0:
        raise ValueError(
            f"samples and min_effective must be at least 1, got {samples}, {min_effective}"
        )
    if smoothing is not None:
        if not (math.isfinite(smoothing) and smoothing > 0.0):
            raise ValueError(f"smoothing must be positive, got {smoothing}")
        if method == "fqe" or _has_density(plan):
            raise ValueError(
                "smoothing applies to a plan without a density, evaluated by weights; 'fqe' "
                "evaluates a deterministic plan directly, and a plan with a density as it is"
            )
    if method != "pdis":
        held = _held(plan, method)
        if horizon is not None or initial is not None:
            raise ValueError("horizon and initial describe episodes, which only 'pdis' reads")
        return _certify_stationary(plant, logger, held, method, samples, smoothing, min_effective)
    if horizon is None and isinstance(plan, AffineSchedule):
        horizon = plan.horizon
    if horizon is None or initial is None:
        raise ValueError("'pdis' reads episodes: pass their horizon and initial law")
    if horizon < 1 or initial.mean.shape != (plant.states,):
        raise ValueError(
            f"horizon must be at least 1 and initial over {plant.states} states, got "
            f"{horizon} and {initial.mean.shape[0]}"
        )
    return _certify_episodes(
        plant, logger, _schedule(plan, horizon), samples, smoothing, initial, min_effective
    )


def _certify_stationary(
    plant: LinearGaussianPlant,
    logger: AffinePolicy,
    plan: AffinePolicy,
    method: EvaluationMethod,
    samples: int,
    smoothing: float | None,
    min_effective: float,
) -> EvaluationCertificate:
    def refuse(
        reason: str,
        one_step: float | None = None,
        limit: float | None = None,
        state: float | None = None,
        action: float | None = None,
    ) -> EvaluationCertificate:
        return EvaluationCertificate(
            method,
            False,
            reason,
            samples,
            math.inf,
            0.0,
            False,
            0.0,
            one_step,
            smoothing_limit=limit,
            state_margin=state,
            action_margin=action,
        )

    for name, policy, meaning in (
        ("logger", logger, "its logs have no stationary law"),
        ("plan", plan, "it has no average cost"),
    ):
        radius = _spectral_radius(plant, policy)
        if radius >= 1.0:
            return refuse(f"the {name}'s closed loop has spectral radius {radius:.4g}: {meaning}")
    mb, sb, rx = _stationary(plant, logger)
    if _min_eig(sb) <= 0.0:
        return refuse("the logger does not excite every direction of (x, u): Sigma_b^z is singular")
    if method == "fqe":
        mp, sp, _ = _stationary(plant, plan)
        nu, _ = _feature_moments(mp, sp)
        _, gram = _feature_moments(mb, sb)
        log_second = math.log(float(nu @ np.linalg.solve(gram, nu)))
        effective = samples * math.exp(-log_second)
        certified = effective >= min_effective
        return EvaluationCertificate(
            method,
            certified,
            (
                f"certified: 1 + restricted chi^2 = {math.exp(log_second):.4g}, "
                f"{effective:.1f} effective of {samples} transitions"
                if certified
                else f"{effective:.1f} effective transitions < min_effective = {min_effective}"
            ),
            samples,
            log_second,
            effective,
            True,
            0.0,
            _one_step_margin(rx, logger, plan),
        )

    limit = None
    target, tau = plan, 0.0
    if not _has_density(plan):
        mp0, s0, _ = _stationary(plant, plan)
        growth = _stationary(plant, _smoothed(plan, 1.0))[1] - s0
        limit = _smoothing_limit(sb, s0, s0 + growth)
        if limit <= 0.0:
            return refuse(
                "no smoothing makes the stationary weight square-integrable: 2 Sigma_b^z - "
                f"Sigma_pi^z has smallest eigenvalue {_min_eig(2.0 * sb - s0):.4g} at tau = 0",
                _one_step_margin(rx, logger, plan),
                limit=0.0,
            )
        if smoothing is None:
            chosen = _best_on_grid(
                limit, lambda t: _log_alpha_moment(mp0, s0 + t * t * growth, mb, sb, 2.0)
            )
            if chosen is None:
                return refuse(
                    f"no smoothing in (0, {limit:.4g}) gives the stationary weight a finite "
                    "second moment in floating point",
                    _one_step_margin(rx, logger, plan),
                    limit=limit,
                )
            smoothing = chosen
        tau = smoothing
        target = _smoothed(plan, tau)
    mp, sp, px = _stationary(plant, target)
    state, action = _schur_margins(rx, px, logger, target)
    one_step = _one_step_margin(rx, logger, target)
    at = "" if limit is None else f" (tau = {tau:.4g}, tau_max = {limit:.4g})"
    if state <= 0.0:
        return refuse(
            "the stationary weight has infinite variance: 2 R_x - P_x has smallest eigenvalue "
            f"{state:.4g}, so the plan spreads the state past twice the logger's variance{at}",
            one_step,
            limit,
            state,
            action,
        )
    if action <= 0.0:
        return refuse(
            "the stationary weight has infinite variance: the action block's Schur complement "
            f"has smallest eigenvalue {action:.4g}{at}",
            one_step,
            limit,
            state,
            action,
        )
    log_second = _log_alpha_moment(mp, sp, mb, sb, 2.0)
    effective = samples * math.exp(-log_second)
    certified = effective >= min_effective
    return EvaluationCertificate(
        method,
        certified,
        (
            f"certified: 1 + chi^2 = {math.exp(log_second):.4g}, {effective:.1f} effective of "
            f"{samples} transitions{at}"
            if certified
            else f"{effective:.1f} effective transitions < min_effective = {min_effective}{at}"
        ),
        samples,
        log_second,
        effective,
        math.isfinite(_log_alpha_moment(mp, sp, mb, sb, 4.0)),
        tau,
        one_step,
        smoothing_limit=limit,
        state_margin=state,
        action_margin=action,
    )


def _certify_episodes(
    plant: LinearGaussianPlant,
    logger: AffinePolicy,
    plan: AffineSchedule,
    samples: int,
    smoothing: float | None,
    initial: InitialLaw,
    min_effective: float,
) -> EvaluationCertificate:
    horizon = plan.horizon
    rx = _stationary(plant, logger)[2] if _spectral_radius(plant, logger) < 1.0 else None

    def one_step(target: AffineSchedule) -> float | None:
        if rx is None:
            return None
        return min(_one_step_margin(rx, logger, target.step(t)) for t in range(horizon))

    def refuse(reason: str, limit: float | None = None) -> EvaluationCertificate:
        return EvaluationCertificate(
            "pdis", False, reason, samples, math.inf, 0.0, False, 0.0, one_step(plan), limit
        )

    if not _has_density(logger):
        return refuse("the logger has no density: its covariance is singular")
    room = _min_eig(2.0 * logger.covariance - plan.covariance)
    if room <= 0.0:
        return refuse(
            f"2 S_b - S_pi has smallest eigenvalue {room:.4g}: every step's weight has infinite "
            "variance"
        )
    limit = None
    target, tau = plan, 0.0
    if not _has_density(plan):
        limit = math.sqrt(room)
        if smoothing is None:

            def second_moment(tau: float) -> float:
                smoothed = _smoothed(plan, tau)
                return float(_trajectory_log_moments(plant, logger, smoothed, initial, 2.0)[-1])

            chosen = _best_on_grid(limit, second_moment)
            if chosen is None:
                return refuse(
                    f"no smoothing in (0, {limit:.4g}) keeps E_b[W_h^2] finite through "
                    f"h = {horizon}",
                    limit,
                )
            smoothing = chosen
        tau = smoothing
        target = _smoothed(plan, tau)
    moments = _trajectory_log_moments(plant, logger, target, initial, 2.0)
    infinite = np.flatnonzero(~np.isfinite(moments))
    escape = int(infinite[0]) + 1 if infinite.size else None
    short = np.flatnonzero(moments > math.log(samples / min_effective))
    certified_horizon = int(short[0]) if short.size else horizon
    log_second = float(moments[-1])
    effective = samples * math.exp(-log_second) if escape is None else 0.0
    at = "" if limit is None else f" (tau = {tau:.4g})"
    if escape is not None:
        reason = f"E_b[W_h^2] is infinite from h = {escape}{at}"
    elif certified_horizon < horizon:
        reason = (
            f"the certifiable horizon is {certified_horizon} and the logs run {horizon} steps: "
            f"n / E_b[W_h^2] falls below min_effective = {min_effective} after it{at}"
        )
    else:
        reason = (
            f"certified through h = {horizon}: {effective:.1f} effective of {samples} episodes{at}"
        )
    fourth = _trajectory_log_moments(plant, logger, target, initial, 4.0)[-1]
    return EvaluationCertificate(
        "pdis",
        escape is None and certified_horizon >= horizon,
        reason,
        samples,
        log_second,
        effective,
        bool(math.isfinite(fourth)),
        tau,
        one_step(target),
        smoothing_limit=limit,
        certified_horizon=certified_horizon,
        escape_horizon=escape,
    )


# -------------------------------------------------------------------------------------- estimators


def _parse_logs(
    logs: Mapping[str, ArrayLike], method: EvaluationMethod, plant: LinearGaussianPlant
) -> tuple[_Array, _Array]:
    missing = sorted({"x", "u"} - set(logs))
    if missing:
        raise KeyError(f"logs need 'x' and 'u'; missing {missing}")
    x = np.asarray(logs["x"], dtype=np.float64)
    u = np.asarray(logs["u"], dtype=np.float64)
    n, m = plant.states, plant.actions
    if method == "pdis":
        ok = x.ndim == 3 and u.ndim == 3 and x.shape[0] == u.shape[0] >= 2
        want = "x (E, H + 1, n) and u (E, H, m) with E >= 2 episodes"
    else:
        ok = x.ndim == 2 and u.ndim == 2 and u.shape[0] >= _BATCHES
        want = f"x (T + 1, n) and u (T, m) with T >= {_BATCHES} transitions"
    ok = ok and x.shape[-2] == u.shape[-2] + 1 and x.shape[-1] == n and u.shape[-1] == m
    if not ok:
        raise ValueError(f"{method!r} reads {want}, n = {n}, m = {m}; got {x.shape} and {u.shape}")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(u))):
        raise ValueError("the logs are not finite")
    return x, u


def _batch_half_width(contributions: _Array) -> tuple[float, float]:
    """The half-width of Student's ``t`` over contiguous batch means, and its degrees of freedom:
    Satterthwaite's ``(sum_k s_k)^2 / sum_k s_k^2`` over each batch's sum of squares ``s_k``, at
    most 39. Weights that spend their mass in a few of the logger's excursions leave the batch
    variance fewer degrees of freedom than batches, and a fixed ``t_39`` then undercovers."""
    batches = np.array_split(contributions, _BATCHES)
    means = np.array([batch.mean() for batch in batches])
    squares = np.array([float(batch @ batch) for batch in batches])
    total = float(squares.sum())
    if total == 0.0:
        return 0.0, _BATCHES - 1.0
    dof = min(_BATCHES - 1.0, total * total / float(squares @ squares))
    half = float(student.ppf(0.975, dof)) * float(means.std(ddof=1)) / math.sqrt(_BATCHES)
    return half, dof


def _normalised(log_w: _Array, axis: int = 0) -> _Array:
    w = np.exp(log_w - np.max(log_w, axis=axis, keepdims=True))
    return w / w.mean(axis=axis, keepdims=True)


def _effective(w: _Array) -> float:
    return float(w.sum() ** 2 / (w * w).sum())


def _stationary_estimate(
    x: _Array,
    u: _Array,
    plant: LinearGaussianPlant,
    plan: AffinePolicy,
    target: AffinePolicy,
    cost: _StageCost,
    method: EvaluationMethod,
) -> tuple[float, float, float, float]:
    """The weighted estimate of ``target``'s average cost, its half-width, the weights' own
    effective sample size and the interval's degrees of freedom. The denominator is the logs' own
    law, fitted, not the model's."""
    xs, xn = x[:-1], x[1:]
    z = np.hstack([xs, u])
    mp, sp, _ = _stationary(plant, target)
    spread = np.atleast_2d(np.cov(z, rowvar=False))
    w = _normalised(_log_normal(z, mp, sp) - _log_normal(z, z.mean(axis=0), spread))
    if method == "mis":
        g = cost(z)
    else:
        h = _RelativeValue.of(plant, plan, cost)
        predicted = xs @ plant.a.T + u @ plant.b.T + plant.offset
        g = cost.mean(mp, sp) + h(xn) - h(predicted) - float(np.trace(h.p @ plant.noise))
    value = float(np.mean(w * g))
    half, dof = _batch_half_width(w * (g - value))
    return value, half, _effective(w), dof


def _fqe_estimate(
    x: _Array, u: _Array, plan: AffinePolicy, cost: _StageCost
) -> tuple[float, float]:
    """Average-cost LSTD-Q with quadratic features, the average cost in place of the constant
    feature, and a sandwich interval: the TD error is a martingale difference when the Q function
    is quadratic, as it is on this class."""
    xs, xn = x[:-1], x[1:]
    n, m = xs.shape[1], u.shape[1]
    z = np.hstack([xs, u])
    phi = _quadratic_features(z)[:, 1:]
    following = _quadratic_features(np.hstack([xn, xn @ plan.gain.T + plan.offset]))[:, 1:]
    i, j = np.triu_indices(n + m)
    actions = (i >= n) & (j >= n)
    following[:, n + m + np.flatnonzero(actions)] += plan.covariance[i[actions] - n, j[actions] - n]
    c = cost(z)
    rows = z.shape[0]
    instruments = np.hstack([phi, np.ones((rows, 1))])
    moved = np.hstack([phi - following, np.ones((rows, 1))])
    system = instruments.T @ moved / rows
    solution = np.linalg.solve(system, instruments.T @ c / rows)
    value = float(solution[-1])
    residual = c - moved @ solution
    meat = (instruments * residual[:, None] ** 2).T @ instruments / rows
    inverse = np.linalg.inv(system)
    variance = float((inverse @ meat @ inverse.T)[-1, -1]) / rows
    return value, _Z95 * math.sqrt(variance)


def _log_policy(policy: AffinePolicy | AffineSchedule, x: _Array, u: _Array) -> _Array:
    """``log policy(u | x)`` over the leading axes of ``x`` and ``u``; a schedule's step is the
    axis before the last."""
    if isinstance(policy, AffinePolicy):
        mean = x @ policy.gain.T + policy.offset
    else:
        mean = np.einsum("tmn,...tn->...tm", policy.gains, x) + policy.offsets
    residual = (u - mean).reshape(-1, u.shape[-1])
    zero = np.zeros(u.shape[-1])
    return _log_normal(residual, zero, policy.covariance).reshape(u.shape[:-1])


def _stage_costs(x: _Array, u: _Array, costs: tuple[_StageCost, ...]) -> _Array:
    """``(E, H)``: each episode's logged cost at each step, step ``t`` scored by ``costs[t]``."""
    z = np.concatenate([x[:, :-1], u], axis=-1)
    return np.stack([cost(z[:, t]) for t, cost in enumerate(costs)], axis=1)


def _episodes_estimate(
    x: _Array,
    u: _Array,
    logger: AffinePolicy,
    target: AffineSchedule,
    costs: tuple[_StageCost, ...],
) -> tuple[float, float, float]:
    """Per-decision importance sampling, each step's weights normalised to mean one: the estimate
    of ``target``'s expected cost over the horizon, step ``t`` scored by ``costs[t]``, its
    half-width by the delta method over episodes, and the last step's effective sample size."""
    xs = x[:, :-1]
    c = _stage_costs(x, u, costs)
    log_w = np.cumsum(_log_policy(target, xs, u) - _log_policy(logger, xs, u), axis=1)
    w = _normalised(log_w)
    per_step = np.mean(w * c, axis=0)
    influence = np.sum(w * (c - per_step), axis=1)
    half = _Z95 * float(influence.std(ddof=1)) / math.sqrt(x.shape[0])
    return float(per_step.sum()), half, _effective(w[:, -1])


def _episode_smoothing(
    plant: LinearGaussianPlant,
    logger: AffinePolicy,
    plan: AffineSchedule,
    costs: tuple[_StageCost, ...],
    initial: InitialLaw,
    samples: int,
    model_error: float,
    min_effective: float,
) -> float | None:
    """The ``tau`` minimising the predicted mean squared error of the reported value -- the
    model's unverified share of the correction, squared, plus the weighted estimate's variance --
    among those the certificate passes. The variance is approximated by the weights' own,
    ``(E_b[W_H^2] - 1) J_H^2 / n``, which leaves out the cost's spread. ``None`` when no smoothing
    passes.

    The grid is read at once, by :func:`_smoothed_log_moments`, and the model's value of the plan
    smoothed by ``tau`` is the plan's own plus ``tau^2 beta``: the smoothing enters the value
    linearly in ``tau^2``, and ``beta`` is that slope (:func:`_horizon_beta`)."""
    budget = math.log(samples / min_effective)
    room = _min_eig(2.0 * logger.covariance - plan.covariance)
    if room <= 0.0 or not _has_density(logger):
        return None
    beta = _horizon_beta(plant, plan, costs[0])
    grid = _grid(math.sqrt(room))
    l2 = _smoothed_log_moments(plant, logger, plan, initial, 2.0, grid)[:, -1]
    within = l2 <= budget
    tau, l2 = grid[within], l2[within]
    value = _episode_value(plant, plan, costs, initial) + tau * tau * beta
    scores = np.full(grid.shape, math.inf)
    scores[within] = (model_error * tau * tau * beta) ** 2 + np.expm1(l2) * value * value / samples
    return _grid_minimum(grid, scores)


def _stationary_smoothing(
    plant: LinearGaussianPlant,
    logger: AffinePolicy,
    plan: AffinePolicy,
    cost: _StageCost,
    samples: int,
    model_error: float,
    min_effective: float,
) -> float | None:
    """:func:`_episode_smoothing`'s ``tau`` for a stationary method, whose weighted estimate's
    variance is exactly ``E_b[w^2 (c - J)^2] / n``."""
    budget = math.log(samples / min_effective)
    if max(_spectral_radius(plant, logger), _spectral_radius(plant, plan)) >= 1.0:
        return None
    mb, sb, _ = _stationary(plant, logger)
    mp, s0, _ = _stationary(plant, plan)
    growth = _stationary(plant, _smoothed(plan, 1.0))[1] - s0
    if _min_eig(sb) <= 0.0:
        return None
    limit = _smoothing_limit(sb, s0, s0 + growth)
    if limit <= 0.0:
        return None
    beta = _RelativeValue.of(plant, plan, cost).beta

    def stationary_mse(tau: float) -> float:
        sp = s0 + tau * tau * growth
        l2 = _log_alpha_moment(mp, sp, mb, sb, 2.0)
        if not l2 <= budget:
            return math.inf
        spread = _tilted_second_moment(mp, sp, mb, sb, cost, cost.mean(mp, sp))
        return (model_error * tau * tau * beta) ** 2 + math.exp(l2) * spread / samples

    return _best_on_grid(limit, stationary_mse)


def _episode_certificate(
    x: _Array,
    plant: LinearGaussianPlant,
    logger: AffinePolicy,
    schedule: AffineSchedule,
    stages: tuple[_StageCost, ...],
    smoothing: float | None,
    model_error: float,
    min_effective: float,
) -> EvaluationCertificate:
    """:func:`certify_evaluation` by ``"pdis"`` on the episodes ``x``, from their starts' law, at
    ``smoothing`` or else at the ``tau`` :func:`_episode_smoothing` chooses: the verdict
    :func:`evaluate_plan` admits, before it is logged."""
    samples = x.shape[0]
    starts = x[:, 0]
    initial = InitialLaw(starts.mean(axis=0), np.atleast_2d(np.cov(starts, rowvar=False)))
    if smoothing is None and not _has_density(schedule):
        smoothing = _episode_smoothing(
            plant, logger, schedule, stages, initial, samples, model_error, min_effective
        )
    return certify_evaluation(
        plant,
        logger,
        schedule,
        "pdis",
        samples,
        smoothing=smoothing,
        initial=initial,
        min_effective=min_effective,
    )


def _episode_estimate(
    x: _Array,
    u: _Array,
    plant: LinearGaussianPlant,
    logger: AffinePolicy,
    schedule: AffineSchedule,
    stages: tuple[_StageCost, ...],
    tau: float,
) -> tuple[float, float, float, float]:
    """``"pdis"``'s weighted estimate with the weights smoothed by ``tau``, its half-width and the
    weights' effective sample size (:func:`_episodes_estimate`), and the model's correction
    ``tau^2 beta_hat``."""
    correction = tau * tau * _horizon_beta(plant, schedule, stages[0]) if tau > 0.0 else 0.0
    weighted, half, effective = _episodes_estimate(
        x, u, logger, _smoothed(schedule, tau) if tau > 0.0 else schedule, stages
    )
    return weighted, half, effective, correction


def _admit(certificate: EvaluationCertificate) -> EvaluationCertificate:
    """Log the certificate's verdict, and raise :class:`InfeasibleEvaluation` when it refuses."""
    event = {
        "chc_event": "plan_evaluation",
        "method": certificate.method,
        "certified": certificate.certified,
        "samples": certificate.samples,
        "effective_samples": certificate.effective_samples,
        "smoothing": certificate.smoothing,
    }
    if not certificate.certified:
        _log.info("evaluation refused", extra=event | {"reason": certificate.reason})
        raise InfeasibleEvaluation(certificate)
    _log.info("evaluation certified", extra=event)
    return certificate


def evaluate_plan(
    logs: Mapping[str, ArrayLike],
    plan: AffinePolicy | AffineSchedule,
    method: EvaluationMethod,
    *,
    plant: LinearGaussianPlant,
    cost: QuadraticCost,
    logger: AffinePolicy | None = None,
    smoothing: float | None = None,
    model_error: float = 1.0,
    min_effective: float = 100.0,
) -> PlanEvaluation:
    """Estimate the cost of deploying ``plan``, from ``logs`` of another policy, if the logs can.

    ``logs`` has ``"x"`` and ``"u"``: for the stationary methods one logged run, ``x`` of shape
    ``(T + 1, n)`` and ``u`` of shape ``(T, m)``; for ``"pdis"`` ``E`` episodes, ``(E, H + 1, n)``
    and ``(E, H, m)``, whose first states give the initial law. ``plan`` is a policy, or for
    ``"pdis"`` a schedule over the logged horizon. The cost is ``cost.running``: one target for
    every state, or for ``"pdis"`` one for each of the episodes' ``H + 1`` states, the last of
    which no step reads. ``plant`` is the model: of the loop's law for every method but
    ``"fqe"``, of the relative value for ``"dr"``, and of the smoothing correction. ``logger`` is
    the policy that logged, fitted by :func:`fit_logger` when omitted.

    For a plan without a density, ``smoothing`` is the ``tau`` the weights use, by default the one
    that minimises the predicted mean squared error of the reported value, and ``model_error`` the
    share of the model's correction ``tau^2 beta_hat`` counted as unverified, which the interval
    carries: 1, the default, counts all of it.

    The interval is a nominal 95 % one: from 40 contiguous batch means and Student's ``t`` with
    Satterthwaite's degrees of freedom for ``"mis"`` and ``"dr"``, a sandwich for ``"fqe"``, and
    the delta method over episodes for ``"pdis"``.

    Raises:
        InfeasibleEvaluation: when the certificate refuses; it carries the certificate.
        KeyError: on logs without ``"x"`` or ``"u"``.
        ValueError: on logs of the wrong shape, non-finite logs, a negative ``model_error``, a
            cost with a target per state for a stationary method or with other than ``H + 1`` of
            them for ``"pdis"``, and every error :func:`certify_evaluation` raises.
    """
    if method not in get_args(EvaluationMethod):
        raise ValueError(f"method must be one of {get_args(EvaluationMethod)}, got {method!r}")
    if not (math.isfinite(model_error) and model_error >= 0.0):
        raise ValueError(f"model_error must be non-negative, got {model_error}")
    _check_policies(plant, plan=plan)
    held = None if method == "pdis" else _held(plan, method)
    x, u = _parse_logs(logs, method, plant)
    n, m = plant.states, plant.actions
    if logger is None:
        logger = fit_logger(x[..., :-1, :].reshape(-1, n), u.reshape(-1, m))
    samples = u.shape[0]
    effective: float | None
    dof: float | None = None
    if held is None:
        schedule = _schedule(plan, u.shape[1])
        stages = _StageCost.steps(cost, n, m, schedule.horizon)
        certificate = _admit(
            _episode_certificate(
                x, plant, logger, schedule, stages, smoothing, model_error, min_effective
            )
        )
        weighted, half, effective, correction = _episode_estimate(
            x, u, plant, logger, schedule, stages, certificate.smoothing
        )
    else:
        stage = _StageCost.of(cost, n, m)
        if smoothing is None and method != "fqe" and not _has_density(held):
            smoothing = _stationary_smoothing(
                plant, logger, held, stage, samples, model_error, min_effective
            )
        certificate = _admit(
            certify_evaluation(
                plant,
                logger,
                held,
                method,
                samples,
                smoothing=smoothing,
                min_effective=min_effective,
            )
        )
        tau = certificate.smoothing
        correction = tau * tau * _RelativeValue.of(plant, held, stage).beta if tau > 0.0 else 0.0
        if method == "fqe":
            weighted, half = _fqe_estimate(x, u, held, stage)
            effective = None
        else:
            weighted, half, effective, dof = _stationary_estimate(
                x, u, plant, held, _smoothed(held, tau) if tau > 0.0 else held, stage, method
            )
    value = weighted - correction
    half += model_error * correction
    share = correction / abs(weighted) if correction > 0.0 else 0.0
    return PlanEvaluation(
        value, (value - half, value + half), certificate, correction, share, effective, dof
    )


def _logged_cost(x: _Array, u: _Array, costs: tuple[_StageCost, ...]) -> float:
    """The logs' own mean cost over the horizon: the value of the policy that logged them."""
    return float(_stage_costs(x, u, costs).mean(axis=0).sum())


def _evaluate_by_unit(
    x: _Array,
    u: _Array,
    units: NDArray[np.int64],
    plan: AffineSchedule,
    *,
    plant: Callable[[_Array, _Array], LinearGaussianPlant],
    cost: QuadraticCost,
    logger: AffinePolicy | None,
    smoothing: float | None,
    model_error: float,
    min_effective: float,
    resamples: int,
    seed: int,
) -> PlanEvaluation:
    """:func:`evaluate_plan` by ``"pdis"`` on the episodes ``x``, ``u`` of ``units``, beside the
    plan against the logger on them, both intervals read from a bootstrap over the units.

    A unit's episodes are dependent through its state, so a draw takes as many units as there are,
    with replacement, and every episode of a unit drawn. Every step that reads the episodes runs
    again on each draw: ``plant`` builds the model from the episodes drawn, the logger is fitted
    again unless it is given, and the initial law, the smoothing, the certificate and both values
    are computed again. A draw the certificate refuses is counted and left out. Each interval is
    its estimate plus or minus 1.96 standard deviations of its draws, widened by the plan's model
    correction as :func:`evaluate_plan` widens its own; the difference's draws are the difference
    within each draw. ``resamples`` is at least two, the caller's to check.

    Raises:
        InfeasibleEvaluation: when the certificate refuses the episodes, or refuses all draws but
            one, which leaves no spread to read.
    """
    evaluation = evaluate_plan(
        {"x": x, "u": u},
        plan,
        "pdis",
        plant=plant(x, u),
        cost=cost,
        logger=logger,
        smoothing=smoothing,
        model_error=model_error,
        min_effective=min_effective,
    )
    n, m = x.shape[-1], u.shape[-1]
    stages = _StageCost.steps(cost, n, m, plan.horizon)
    labels, members = np.unique(units, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws: list[tuple[float, float]] = []
    refusal: EvaluationCertificate | None = None
    for _ in range(resamples):
        drawn = np.bincount(rng.integers(0, labels.size, labels.size), minlength=labels.size)
        rows = np.repeat(np.arange(x.shape[0]), drawn[members])
        xs, us = x[rows], u[rows]
        model = plant(xs, us)
        fitted = (
            logger
            if logger is not None
            else fit_logger(xs[:, :-1].reshape(-1, n), us.reshape(-1, m))
        )
        certificate = _episode_certificate(
            xs, model, fitted, plan, stages, smoothing, model_error, min_effective
        )
        if not certificate.certified:
            refusal = certificate
            continue
        weighted, _, _, correction = _episode_estimate(
            xs, us, model, fitted, plan, stages, certificate.smoothing
        )
        draws.append((weighted - correction, weighted - correction - _logged_cost(xs, us, stages)))
    refused = resamples - len(draws)
    _log.log(
        logging.WARNING if refused else logging.INFO,
        "the certificate refused resamples of the units, and the intervals read the rest"
        if refused
        else "units resampled",
        extra={
            "chc_event": "unit_bootstrap",
            "units": int(labels.size),
            "resamples": resamples,
            "refused": refused,
        },
    )
    if refusal is not None and len(draws) < 2:
        raise InfeasibleEvaluation(
            replace(
                refusal,
                reason=f"the certificate refused {refused} of {resamples} resamples of the units, "
                f"which leaves no spread to read; the last because {refusal.reason}",
            )
        )
    spread = np.std(np.asarray(draws), axis=0, ddof=1)
    widened = model_error * evaluation.model_correction
    half, half_difference = (_Z95 * float(deviation) + widened for deviation in spread)
    logged = _logged_cost(x, u, stages)
    difference = evaluation.value - logged
    return replace(
        evaluation,
        interval=(evaluation.value - half, evaluation.value + half),
        versus_logger=LoggerComparison(
            logged, difference, (difference - half_difference, difference + half_difference)
        ),
        bootstrap=UnitBootstrap(int(labels.size), resamples, refused),
    )
