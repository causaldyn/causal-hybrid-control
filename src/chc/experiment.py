"""Which experiment, if any, to run before a decision: units per zone, the regret they remove, and a
verdict to deploy now, experiment first, or abstain.

The decision (:class:`ZoneDecision`) is one-shot, over ``K`` zones. Zone ``k``'s lever ``u_k`` moves
the zones' states through its own channel ``b_k``, and a known coupling spreads the effect::

    x = x0 + coupling diag(b) u,   J(u; b) = (x - x*)' Q (x - x*) / 2 + u' R u / 2,   lo <= u <= hi.

The channels are known up to a Gaussian posterior (:class:`ChannelPrior`), and acting now costs
regret: what the best action under it, :meth:`ZoneDecision.bayes_action`, loses against the one
that knew ``b``. An experiment
(:class:`ZoneExperiment`) sets zone ``k``'s lever to ``+probe_k`` or ``-probe_k`` at random on
``n_k`` units, each costing ``unit_cost_k`` and each adding ``probe_k^2 / noise_sd_k^2`` to the
precision of ``b_k``. :func:`design_experiment` chooses the ``n_k`` that minimise the regret left
after the experiment, times the number of times the decision will be used, plus what the experiment
spends, within a budget. Then it measures the regret before and after by Monte Carlo, re-solving the
decision on every draw: the regrets, the value, its ``error`` and ``regret_bound`` are that
measurement. ``pinned`` and ``weights`` are not: they are the local model's, below.

The local model it optimises (the lab's value-of-information thread, independently verified):

* **The decision weight** ``W = J_bu M^-1 J_ub`` over the free levers, ``M`` the Hessian of ``J``
  in ``u``. It is the Hessian of the certainty-equivalent regret in the channel estimate, so a
  posterior with covariance ``S`` leaves an expected regret of ``tr(W S) / 2`` to leading order.
  It is averaged over the prior's sigma points, because at a knife edge -- ``q b^2 = r`` in one
  zone -- the plug-in ``W`` is zero and the regret is fourth order, not nil.
* **A pinned lever**, held at its bound by a positive multiplier, has no local weight. It costs
  regret only when the channel lies past its activation point: ``s G(z) / (2 S)`` for a multiplier
  ``z`` standard deviations from zero, with ``G(z) = E (Z - z)_+^2``. What an experiment can buy
  there is at most ``delta^2 phi(z) / (S z^3)`` for every ``z > 0``, ``delta^2`` the multiplier's
  preposterior variance: a zone far from activation is not worth testing, and one near it is.
* **The allocation** starts from reverse water-filling at the price of information,
  ``unit_cost_k noise_sd_k^2 / probe_k^2``, and minimises the full objective from there. With a flat
  prior that start is decision-Neyman, ``n_k`` proportional to
  ``noise_sd_k sqrt(W_kk / unit_cost_k) / probe_k``; on the lab's four-zone market classical Neyman,
  which ignores ``W``, left 1.5 times its regret.

The action, before the experiment and after it, is the Bayes one: the best action under the prior
now and under the posterior after. The expected cost under ``b ~ N(m, S)`` is
``J(u; m) + u' ((coupling' Q coupling) o S) u / 2``, ``o`` the elementwise product
(``validation/experiment_bayes_action.mac``), so the Bayes action is the certainty-equivalent one
with the action weight raised by ``(coupling' Q coupling) o S``, and it holds back where the
channel is uncertain. The posterior's covariance does not depend on what the experiment reads, so
neither does the raise after it. Against the certainty-equivalent action an experiment was
credited with what acting well now would have saved without it: on one zone, ``Q = 1``,
``R = 0.1``, a target 1 from a baseline 0 and ``b ~ N(0.5, 1)``, acting on the mean lost 1.01 and
the Bayes action 0.27; at a unit cost of 0.03 the experiment recommended against the first, worth
0.63 there, left 0.35 of regret and spent 0.03, and so lost 0.11 against the second. So
``"experiment"`` says experimenting beats acting well now.

What the local model gets wrong, and what is done about it:

* **A wide prior.** At a prior sd of 37-50% of ``b`` the lab's local model put the regret after the
  experiment 32% low. The numbers reported are Monte Carlo's whatever the model says;
  ``model_gap`` is how far the model was from them, and a gap above 25% is logged as a warning.
  The model then only proposes: at every spend from the budget down by halves to one unit's, its
  best experiment that spends it, and none; Monte Carlo scores each on draws of its own and the
  best is reported on the main draws. The same search runs where the model's experiment does not
  pay, which the model read at that one allocation only: on one zone, ``R = 0.1``,
  ``b ~ N(0.5, 1)`` and a unit costing 0.07, it chose none, and one unit was worth 0.033.
* **A channel whose sign is unknown.** When a zone that matters stays within two posterior standard
  deviations of zero after the experiment the budget buys, and the experiment does not pay, the
  verdict is ``"abstain"``: the sign decides which way to act.
* **Outside the model, and not checked:** a response nonlinear in the lever, interference the
  coupling does not state, a probe that moves the state it measures, heavy-tailed noise, and a
  pinned lever's far bound, which the tail law ignores and so overstates what testing it buys.

A channel the logs do not identify -- :class:`chc.decision.NotIdentifiedError` -- enters with the
prior its owner is prepared to state, a wide variance about the best guess; the design then says
how many units in which zones identifying it is worth, or ``"abstain"`` when nothing affordable is.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, special
from scipy.linalg import solve_discrete_lyapunov

from chc.regret import _gain_jacobian, dlqr

__all__ = [
    "ChannelPrior",
    "DesignVerdict",
    "DitherDesign",
    "DitherMethod",
    "ExperimentDesign",
    "PinnedZone",
    "ProbeDesign",
    "ZoneDecision",
    "ZoneExperiment",
    "design_experiment",
    "dither_covariance",
    "probe_design",
]

_log = logging.getLogger(__name__)

DesignVerdict = Literal["deploy", "experiment", "abstain"]

_Array = NDArray[np.float64]

_KAPPA = 0.5  # sigma points sit sqrt(K + kappa) prior standard deviations from the mean
_SIGNED = 2.0  # a channel's sign is known once its mean is this many posterior sds from zero
_RELEVANT = 1e-3  # a zone matters when its decision weight is this share of the largest
_GAP_WARNING = 0.25
_MIN_DRAWS = 100
_SEARCH_DRAWS = 1000  # draws per allocation when Monte Carlo searches the allocations
_Z95 = 1.959963984540054


def _vector(value: ArrayLike, name: str, size: int | None = None) -> _Array:
    array = np.array(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or (size is not None and array.shape != (size,)):
        wanted = "a non-empty vector" if size is None else f"shape ({size},)"
        raise ValueError(f"{name} must have {wanted}, got shape {array.shape}")
    return array


def _matrix(value: ArrayLike, name: str, size: int) -> _Array:
    array = np.array(value, dtype=np.float64)
    if array.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size}), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} has a non-finite entry")
    return array


def _finite(array: _Array, name: str) -> None:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} has a non-finite entry")


def _symmetric_root(m: _Array, name: str, *, definite: bool) -> _Array:
    """The symmetric square root of ``m``, after checking it is symmetric and (semi)definite."""
    if not np.allclose(m, m.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    values, vectors = np.linalg.eigh(0.5 * (m + m.T))
    floor = 1e-12 * max(1.0, float(np.max(np.abs(values))))
    smallest = float(np.min(values))
    if smallest < -floor or (definite and smallest <= floor):
        kind = "positive definite" if definite else "positive semidefinite"
        raise ValueError(f"{name} must be {kind}, smallest eigenvalue {smallest:.3g}")
    return (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.T


def _frozen(array: _Array) -> _Array:
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ZoneDecision:
    """A one-shot decision over ``K`` zones, ``x = baseline + coupling diag(b) u`` and
    ``J(u; b) = (x - target)' state_weight (x - target) / 2 + u' action_weight u / 2`` on the box
    ``lo <= u <= hi``.

    ``coupling[i, k]`` is how much of zone ``k``'s channel reaches zone ``i``'s state: the identity
    when zones do not interact, ``I - P`` when a lever pulls a share ``P[i, k]`` of its effect from
    its neighbours. ``baseline`` is the state with every lever at zero. ``action_weight`` must be
    positive definite, so every channel vector has one best action; a bound may be infinite.

    Raises:
        ValueError: on shapes that do not agree, a non-finite entry other than a bound, a
            ``state_weight`` that is not symmetric positive semidefinite, an ``action_weight``
            that is not symmetric positive definite, or a box with ``lo >= hi``.
    """

    coupling: _Array  # (K, K)
    baseline: _Array  # (K,)
    target: _Array  # (K,)
    state_weight: _Array  # (K, K)
    action_weight: _Array  # (K, K)
    lo: _Array  # (K,)
    hi: _Array  # (K,)
    _state_root: _Array = field(init=False, repr=False, compare=False)
    _action_root: _Array = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        coupling = np.array(self.coupling, dtype=np.float64)
        if coupling.ndim != 2 or coupling.shape[0] != coupling.shape[1] or coupling.size == 0:
            raise ValueError(f"coupling must be a square matrix, got shape {coupling.shape}")
        k = coupling.shape[0]
        _finite(coupling, "coupling")
        fields = {
            "coupling": coupling,
            "baseline": _vector(self.baseline, "baseline", k),
            "target": _vector(self.target, "target", k),
            "state_weight": _matrix(self.state_weight, "state_weight", k),
            "action_weight": _matrix(self.action_weight, "action_weight", k),
            "lo": _vector(self.lo, "lo", k),
            "hi": _vector(self.hi, "hi", k),
        }
        _finite(fields["baseline"], "baseline")
        _finite(fields["target"], "target")
        if np.any(np.isnan(fields["lo"])) or np.any(np.isnan(fields["hi"])):
            raise ValueError("a bound is NaN")
        if np.any(fields["lo"] >= fields["hi"]):
            zones = np.flatnonzero(fields["lo"] >= fields["hi"]).tolist()
            raise ValueError(f"zones {zones} have lo >= hi; a lever needs room to move")
        state_root = _symmetric_root(fields["state_weight"], "state_weight", definite=False)
        _symmetric_root(fields["action_weight"], "action_weight", definite=True)
        for name, value in fields.items():
            object.__setattr__(self, name, _frozen(value))
        object.__setattr__(self, "_state_root", _frozen(state_root))
        # astype, not a cast: numpy's stub for cholesky on float input is floating[Any] before 2.5,
        # the numpy Python 3.11 gets, so the float64 promise is made true of the value.
        action_root = np.linalg.cholesky(fields["action_weight"]).T.astype(np.float64, copy=False)
        object.__setattr__(self, "_action_root", _frozen(action_root))

    @property
    def zones(self) -> int:
        return self.coupling.shape[0]

    def cost(self, u: ArrayLike, b: ArrayLike) -> float:
        """``J(u; b)``."""
        return _cost(self, _vector(u, "u", self.zones), _vector(b, "b", self.zones))

    def solve(self, b: ArrayLike) -> _Array:
        """The best action for channels ``b``, on the box."""
        b = _vector(b, "b", self.zones)
        _finite(b, "b")
        return _solve(self, b).action

    def regret(self, u: ArrayLike, b: ArrayLike) -> float:
        """What ``u`` loses against the best action when the channels are ``b``."""
        b = _vector(b, "b", self.zones)
        _finite(b, "b")
        return self.cost(u, b) - _cost(self, _solve(self, b).action, b)

    def bayes_action(self, prior: ChannelPrior) -> _Array:
        """The best action, on the box, for channels known only as ``prior``: the action that
        minimises the expected cost ``J(u; m) + u' ((coupling' Q coupling) o S) u / 2``
        (``validation/experiment_bayes_action.mac``), the certainty-equivalent one with the action
        weight raised by ``(coupling' Q coupling) o S``.

        Raises:
            ValueError: on a prior whose length is not the decision's zones.
        """
        if prior.mean.size != self.zones:
            raise ValueError(f"the decision has {self.zones} zones, the prior {prior.mean.size}")
        return _solve(_cautious(self, prior.covariance), prior.mean).action


def _cautious(d: ZoneDecision, covariance: _Array) -> ZoneDecision:
    """``d`` with its action weight raised by ``(coupling' Q coupling) o covariance``: its
    certainty-equivalent action at the mean of ``b ~ N(mean, covariance)`` is the Bayes action."""
    raised = d.action_weight + (d.coupling.T @ d.state_weight @ d.coupling) * covariance
    return dataclasses.replace(d, action_weight=0.5 * (raised + raised.T))


def _cost(d: ZoneDecision, u: _Array, b: _Array) -> float:
    x = d.baseline + (d.coupling * b) @ u - d.target
    return 0.5 * float(x @ d.state_weight @ x + u @ d.action_weight @ u)


@dataclass(frozen=True)
class _Solution:
    action: _Array
    hessian: _Array  # M = G' Q G + R
    gradient: _Array  # of J in u, at the action
    pinned: NDArray[np.bool_]  # held at a bound by a strictly positive multiplier


def _solve(d: ZoneDecision, b: _Array) -> _Solution:
    """Bounded least squares, ``|Q^1/2 (x0 + G u - x*)|^2 + |R^1/2 u|^2``, by BVLS, whose last
    iterate is already the exact least-squares solve on its free set: solving the free levers again
    on that set moved it by at most 1.3e-13 relative over 4000 random instances."""
    k = d.zones
    g = d.coupling * b
    hessian = g.T @ d.state_weight @ g + d.action_weight
    linear = g.T @ d.state_weight @ (d.baseline - d.target)
    result = optimize.lsq_linear(
        np.vstack([d._state_root @ g, d._action_root]),
        np.concatenate([d._state_root @ (d.target - d.baseline), np.zeros(k)]),
        bounds=(d.lo, d.hi),
        method="bvls",
        tol=1e-12,
    )
    u = np.clip(result.x, d.lo, d.hi)
    gradient = hessian @ u + linear
    reach = 1e-9 * (1.0 + np.abs(u))
    scale = 1e-10 * (1.0 + float(np.abs(linear).max()) + float(np.abs(hessian).max()))
    at_lo, at_hi = u <= d.lo + reach, u >= d.hi - reach
    pinned = (at_lo & (gradient > scale)) | (at_hi & (gradient < -scale))
    return _Solution(u, hessian, gradient, pinned)


@dataclass(frozen=True)
class _Pin:
    """A lever held at its bound: its multiplier, the multiplier's gradient in ``b`` and the Schur
    curvature that turns a violated multiplier into regret."""

    zone: int
    multiplier: float  # |dJ/du_k| at the bound, > 0
    gradient: _Array  # d multiplier / d b, with the free levers re-optimised
    curvature: float


@dataclass(frozen=True)
class _Local:
    action: _Array
    weight: _Array
    pins: tuple[_Pin, ...]


def _local(d: ZoneDecision, b: _Array) -> _Local:
    """The decision weight over the free levers, and each pinned lever's multiplier, at ``b``."""
    solution = _solve(d, b)
    u, m = solution.action, solution.hessian
    g = d.coupling * b
    qx = d.state_weight @ (d.baseline + g @ u - d.target)
    # column j: the derivative in b_j of the gradient in u
    j_ub = np.diag(d.coupling.T @ qx) + (g.T @ d.state_weight @ d.coupling) * u
    free, held = np.flatnonzero(~solution.pinned), np.flatnonzero(solution.pinned)
    weight = np.zeros((d.zones, d.zones))
    response = np.zeros((0, d.zones))
    if free.size:
        response = np.linalg.solve(m[np.ix_(free, free)], j_ub[free])  # -du_F/db
        weight = j_ub[free].T @ response
    pins = []
    for k in held:
        if free.size:
            coupled = np.linalg.solve(m[np.ix_(free, free)], m[free, k])
            gradient = j_ub[k] - m[k, free] @ response
            curvature = float(m[k, k] - m[k, free] @ coupled)
        else:
            gradient, curvature = j_ub[k].copy(), float(m[k, k])
        pins.append(_Pin(int(k), abs(float(solution.gradient[k])), gradient, curvature))
    return _Local(u, 0.5 * (weight + weight.T), tuple(pins))


@dataclass(frozen=True)
class ChannelPrior:
    """What is known about the channels now, ``b ~ N(mean, covariance)``: a fit's estimate and its
    covariance, say. A zero variance is a channel known exactly, and no experiment is spent on it.

    Raises:
        ValueError: on shapes that do not agree, a non-finite entry, or a covariance that is not
            symmetric positive semidefinite.
    """

    mean: _Array
    covariance: _Array
    _root: _Array = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        mean = _vector(self.mean, "mean")
        _finite(mean, "mean")
        covariance = _matrix(self.covariance, "covariance", mean.size)
        root = _symmetric_root(covariance, "covariance", definite=False)
        object.__setattr__(self, "mean", _frozen(mean))
        object.__setattr__(self, "covariance", _frozen(0.5 * (covariance + covariance.T)))
        object.__setattr__(self, "_root", _frozen(root))


@dataclass(frozen=True)
class ZoneExperiment:
    """What a unit of experiment costs and tells, per zone.

    A unit in zone ``k`` sets the lever to ``+probe_k`` or ``-probe_k`` at random and observes the
    zone's response with noise sd ``noise_sd_k``, so it adds ``probe_k^2 / noise_sd_k^2`` to the
    precision of ``b_k``. It costs ``unit_cost_k``, in the units of the decision's cost.

    Raises:
        ValueError: on vectors of different lengths, or an entry that is not positive and finite.
    """

    unit_cost: _Array
    noise_sd: _Array
    probe: _Array

    def __post_init__(self) -> None:
        size = _vector(self.unit_cost, "unit_cost").size
        for name in ("unit_cost", "noise_sd", "probe"):
            value = _vector(getattr(self, name), name, size)
            if not np.all(np.isfinite(value) & (value > 0.0)):
                raise ValueError(f"{name} must be positive and finite, got {value.tolist()}")
            object.__setattr__(self, name, _frozen(value))

    @property
    def information(self) -> _Array:
        """The precision one unit adds to each channel."""
        return self.probe**2 / self.noise_sd**2


@dataclass(frozen=True)
class PinnedZone:
    """A lever held at its bound at the prior mean: how far its multiplier is from releasing it, in
    prior standard deviations, and the expected regret the pin costs now, times ``uses``.

    ``regret`` is the local model's Gaussian-tail law, ``s G(z) / (2 S)``, not a measurement: on one
    lever pinned at zero, ``x = -1 + b u`` with ``b ~ N(-0.3, s^2)``, it read 14% under the exact
    regret at ``z = 3``, 6% under at ``z = 1.5``, 8% over at ``z = 1`` and 52% over at ``z = 0.6``.
    """

    zone: int
    z: float
    regret: float


@dataclass(frozen=True)
class ExperimentDesign:
    """What :func:`design_experiment` recommends, and the numbers behind it.

    The regrets are expectations over the prior, times ``uses``: ``regret_now`` of the Bayes
    action under the prior, :meth:`ZoneDecision.bayes_action`, and ``regret_after`` of the Bayes
    action under the posterior once ``sample_sizes`` units have run (see :mod:`chc.experiment`).
    Both are Monte Carlo's, over draws of the channels with the decision re-solved on every draw,
    and ``error`` is a half-width that holds their expected difference with probability at least
    0.95: the larger side of Hall's (1992) skewness-corrected interval, since the difference is
    skewed to the right. ``value`` is what the experiment removes less what it spends.
    ``regret_bound`` is the ``level`` quantile of the regret of acting now: with that probability
    under the prior, deploying now loses at most this much. It is the draws' order statistic of
    rank ``ceil(level (draws + 1))``, which a fresh draw stays at or below with probability at
    least ``level`` at any number of draws, and ``inf`` where ``draws`` is below
    ``level / (1 - level)``, too few to show it.
    ``model_gap`` is the local model's largest relative miss against the two Monte Carlo regrets;
    above 25%, or where its experiment does not pay, the allocation is the best Monte Carlo
    scored of the model's proposals, and ``reason`` says which. ``weights`` is the prior-averaged
    decision weight ``W``.
    """

    verdict: DesignVerdict
    reason: str
    sample_sizes: tuple[int, ...]
    spend: float
    regret_now: float
    regret_after: float
    value: float
    error: float
    regret_bound: float
    model_gap: float
    weights: _Array
    pinned: tuple[PinnedZone, ...]


def _second_partial_moment(z: float) -> float:
    """``E (Z - z)_+^2`` for a standard normal ``Z``."""
    return max(
        (1.0 + z * z) * float(special.ndtr(-z))
        - z * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi),
        0.0,
    )


def _pin_regret(pin: _Pin, variance: float) -> float:
    """The expected regret of holding ``pin`` at its bound when its multiplier has ``variance``."""
    if variance <= 0.0:
        return 0.0
    return (
        variance
        * _second_partial_moment(pin.multiplier / math.sqrt(variance))
        / (2.0 * pin.curvature)
    )


def _posterior_covariance(prior: ChannelPrior, information: _Array) -> _Array:
    """``(S0^-1 + diag(information))^-1``, in a form that allows a singular ``S0``."""
    root = prior._root
    inner = np.eye(root.shape[0]) + (root * information) @ root
    return root @ np.linalg.solve(inner, root)


def _expected_regret(
    weight: _Array, pins: tuple[_Pin, ...], prior: ChannelPrior, information: _Array
) -> float:
    """The local model: ``tr(W S1) / 2``, plus each pinned lever's regret now less what the
    experiment's preposterior variance buys back."""
    after = _posterior_covariance(prior, information)
    total = 0.5 * float(np.sum(weight * after))
    for pin in pins:
        now = float(pin.gradient @ prior.covariance @ pin.gradient)
        left = float(pin.gradient @ after @ pin.gradient)
        total += _pin_regret(pin, now) - _pin_regret(pin, max(now - left, 0.0))
    return total


def _sigma_weight(d: ZoneDecision, prior: ChannelPrior) -> _Array:
    """``W`` averaged over the prior's sigma points."""
    k = d.zones
    spread = math.sqrt(k + _KAPPA)
    total = _KAPPA / (k + _KAPPA) * _local(d, prior.mean).weight
    for column in prior._root.T:
        for sign in (1.0, -1.0):
            total = (
                total + 0.5 / (k + _KAPPA) * _local(d, prior.mean + sign * spread * column).weight
            )
    return 0.5 * (total + total.T)


def _water_filling(
    weight: _Array, prior: ChannelPrior, experiment: ZoneExperiment, budget: float, uses: float
) -> _Array:
    """Reverse water-filling on the diagonal of ``W`` at the price of information, spending at
    most ``budget``: the exact optimum for a diagonal ``W`` and a diagonal prior."""
    information = experiment.information
    price = experiment.unit_cost / information
    variance = np.diag(prior.covariance)
    with np.errstate(divide="ignore"):
        precision = np.where(variance > 0.0, 1.0 / variance, np.inf)
    w = np.clip(np.diag(weight), 0.0, None)

    def units(log_price_multiplier: float) -> _Array:
        level = np.sqrt(uses * w / (2.0 * math.exp(log_price_multiplier) * price))
        return np.maximum(level - precision, 0.0) / information

    n = units(0.0)
    if experiment.unit_cost @ n > budget:
        high = 1.0
        while experiment.unit_cost @ units(high) > budget:
            high *= 2.0
        n = units(optimize.brentq(lambda t: experiment.unit_cost @ units(t) - budget, 0.0, high))
    return n


def _allocate(
    risk: Callable[[_Array], float],
    start: _Array,
    unit_cost: _Array,
    budget: float,
    *,
    price: float = 1.0,
) -> _Array:
    """Minimise ``risk(n) + price unit_cost . n`` over ``n >= 0`` with ``unit_cost . n <= budget``,
    from the water-filling start and from an even split, in spend shares so every variable is O(1).
    At ``price = 0`` it is the model's best experiment that spends ``budget``, paid or not."""
    k = unit_cost.size
    scale = max(risk(np.zeros(k)), np.finfo(float).tiny)

    def objective(share: _Array) -> float:
        n = np.clip(share, 0.0, None) * budget / unit_cost
        return (risk(n) + price * float(unit_cost @ n)) / scale

    candidates = [np.clip(start * unit_cost / budget, 0.0, 1.0), np.full(k, 0.5 / k)]
    best = candidates[0]
    for x0 in candidates:
        result = optimize.minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * k,
            constraints=[{"type": "ineq", "fun": lambda share: 1.0 - float(np.sum(share))}],
            options={"ftol": 1e-12, "maxiter": 500},
        )
        for share in (np.clip(result.x, 0.0, 1.0), x0):
            if np.sum(share) <= 1.0 + 1e-9 and objective(share) < objective(best):
                best = share
    return np.clip(best, 0.0, None) * budget / unit_cost


def _whole_units(n: _Array, unit_cost: _Array, budget: float) -> NDArray[np.int64]:
    whole = np.floor(n + 1e-9).astype(np.int64)
    while unit_cost @ whole > budget:
        whole[int(np.argmax(whole * unit_cost))] -= 1
    return whole


@dataclass(frozen=True)
class _Worlds:
    """Channels drawn from the prior, with what an experiment would read in each and the regret of
    acting now; shared by every allocation scored on them."""

    b: _Array  # (draws, K)
    shocks: _Array  # (draws, K) standard normal, one reading error per zone, scaled per allocation
    best: _Array  # per draw, the cost of the action that knows b
    now: _Array  # per draw, the regret of the Bayes action under the prior


def _worlds(
    d: ZoneDecision, prior: ChannelPrior, action: _Array, draws: int, seed: int | tuple[int, int]
) -> _Worlds:
    rng = np.random.default_rng(seed)
    b = prior.mean + rng.standard_normal((draws, d.zones)) @ prior._root.T
    shocks = rng.standard_normal((draws, d.zones))
    best = np.array([_cost(d, _solve(d, row).action, row) for row in b])
    now = np.array([_cost(d, action, row) for row in b]) - best
    return _Worlds(b, shocks, best, now)


def _regret_after(
    d: ZoneDecision,
    prior: ChannelPrior,
    worlds: _Worlds,
    units: NDArray[np.int64],
    information: _Array,
) -> _Array:
    """Per draw, the regret of the Bayes action under the posterior that ``units`` leave."""
    observed = np.flatnonzero(units > 0)
    if not observed.size:
        return worlds.now
    noise = 1.0 / np.sqrt(units[observed] * information[observed])
    innovation = prior.covariance[np.ix_(observed, observed)] + np.diag(noise**2)
    gain = np.linalg.solve(innovation, prior.covariance[observed]).T
    # The posterior covariance does not depend on the reading, so neither does the raised weight.
    cautious = _cautious(d, _posterior_covariance(prior, units * information))
    readings = worlds.b[:, observed] + worlds.shocks[:, observed] * noise
    means = prior.mean + (readings - prior.mean[observed]) @ gain.T
    after = np.empty(worlds.b.shape[0])
    for i in range(after.size):
        after[i] = _cost(d, _solve(cautious, means[i]).action, worlds.b[i]) - worlds.best[i]
    return after


def _candidates(
    risk: Callable[[_Array], float],
    weight: _Array,
    prior: ChannelPrior,
    experiment: ZoneExperiment,
    budget: float,
    uses: float,
) -> list[NDArray[np.int64]]:
    """No experiment, and at each spend from the budget down by halves to one unit's, the local
    model's best experiment that spends it, in whole units: the model's shape, Monte Carlo's
    scale."""
    unit_cost = experiment.unit_cost
    found = [np.zeros(unit_cost.size, dtype=np.int64)]
    spend = budget
    while spend >= float(unit_cost.min()):
        start = _water_filling(weight, prior, experiment, spend, uses)
        units = _whole_units(_allocate(risk, start, unit_cost, spend, price=0.0), unit_cost, spend)
        if not any(np.array_equal(units, seen) for seen in found):
            found.append(units)
        spend *= 0.5
    return found


def _pinned_zone(pin: _Pin, prior: ChannelPrior, uses: float) -> PinnedZone:
    variance = float(pin.gradient @ prior.covariance @ pin.gradient)
    z = pin.multiplier / math.sqrt(variance) if variance > 0.0 else math.inf
    return PinnedZone(pin.zone, z, uses * _pin_regret(pin, variance))


def _upper_quantile(values: _Array, level: float) -> float:
    """The order statistic of rank ``ceil(level (n + 1))`` of ``n`` draws. A further draw of their
    law lies at or below the rank-``k`` statistic with probability ``k / (n + 1)``, more where the
    law has atoms, so this is the least rank that holds ``level``. The interpolated quantile fell
    short: 0.941 at 100 draws."""
    rank = math.ceil(level * (values.size + 1))
    return float(np.sort(values)[rank - 1]) if rank <= values.size else math.inf


def _half_width(values: _Array) -> float:
    """The larger side of Hall's (1992) skewness-corrected 95% interval for the mean of ``values``,
    as a half-width on either side of it. A regret is skewed to the right and its spread rises with
    it, so where the mean reads low the spread does too, and 1.96 standard errors held the mean
    0.915 to 0.926 of the time at 100 draws on three designs. Hall's transformation
    ``g(t) = t + b t^2 + b^2 t^3 / 3 + b / 2``, ``b = skew / (3 sqrt(n))``, removes the
    studentised mean's skew to first order, and its larger side held 0.950 to 0.969 from 100 to
    1000 draws: at least 95%, at 1.6 times the normal width at 100 draws and 1.1 times at 1000."""
    n = values.size
    se = float(np.std(values, ddof=1)) / math.sqrt(n)
    if se == 0.0:
        return 0.0
    centred = values - values.mean()
    b = float(np.mean(centred**3) / np.mean(centred**2) ** 1.5) / (3.0 * math.sqrt(n))

    def inverse(y: float) -> float:
        # g^-1(y) = (c - 1) / b with c^3 = 1 + 3 b (y - b / 2), written without the cancellation
        c = float(np.cbrt(1.0 + 3.0 * b * (y - 0.5 * b)))
        return 3.0 * (y - 0.5 * b) / (c * c + c + 1.0)

    return se * max(inverse(_Z95), -inverse(-_Z95))


def _relative_gap(model: float, measured: float) -> float:
    if measured == 0.0:
        return 0.0 if model == 0.0 else math.inf
    return abs(model / measured - 1.0)


def design_experiment(  # noqa: C901, PLR0912, PLR0915
    decision: ZoneDecision,
    prior: ChannelPrior,
    experiment: ZoneExperiment,
    budget: float,
    *,
    uses: float = 1.0,
    level: float = 0.95,
    draws: int = 10_000,
    seed: int = 0,
) -> ExperimentDesign:
    """How many units to run in each zone before deciding, and whether to run any.

    ``uses`` is how many times the decision will be taken on what is known after the experiment,
    so the regret it leaves counts ``uses`` times and the spend once. The verdict:

    * ``"experiment"``: the regret the experiment removes, against the Bayes action now, exceeds
      its spend by more than the Monte Carlo error;
    * ``"abstain"``: no experiment scored does, and a zone that matters stays within two posterior
      standard deviations of zero, so the sign of its channel -- which way to act there -- is still
      unknown;
    * ``"deploy"``: no experiment scored does, and acting now, on
      :meth:`ZoneDecision.bayes_action`, costs ``regret_now`` on average and at most
      ``regret_bound`` with probability ``level``.

    Raises:
        ValueError: on a prior or an experiment whose length is not the decision's zones, a
            ``budget`` that is negative or not finite, ``uses`` that is not positive and finite,
            a ``level`` outside ``(0, 1)``, or fewer than 100 ``draws``.
    """
    k = decision.zones
    if prior.mean.size != k or experiment.unit_cost.size != k:
        raise ValueError(
            f"the decision has {k} zones, the prior {prior.mean.size} and the experiment "
            f"{experiment.unit_cost.size}"
        )
    if not (math.isfinite(budget) and budget >= 0.0):
        raise ValueError(f"budget must be non-negative and finite, got {budget}")
    if not (math.isfinite(uses) and uses > 0.0):
        raise ValueError(f"uses must be positive and finite, got {uses}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level}")
    if draws < _MIN_DRAWS:
        raise ValueError(f"draws must be at least {_MIN_DRAWS}, got {draws}")

    information = experiment.information
    local = _local(decision, prior.mean)
    weight = _sigma_weight(decision, prior)
    action = decision.bayes_action(prior)

    def risk(n: _Array) -> float:
        return uses * _expected_regret(weight, local.pins, prior, n * information)

    def gap(units: NDArray[np.int64], now: float, after: float) -> float:
        return max(
            _relative_gap(risk(np.zeros(k)), now),
            _relative_gap(risk(units.astype(np.float64)), after),
        )

    units = np.zeros(k, dtype=np.int64)
    if budget > 0.0:
        start = _water_filling(weight, prior, experiment, budget, uses)
        units = _whole_units(
            _allocate(risk, start, experiment.unit_cost, budget), experiment.unit_cost, budget
        )
    worlds = _worlds(decision, prior, action, draws, seed)
    regret_now = uses * float(worlds.now.mean())
    after = _regret_after(decision, prior, worlds, units, information)
    model_gap = gap(units, regret_now, uses * float(after.mean()))
    pays = units.sum() > 0 and regret_now - uses * float(after.mean()) - float(
        experiment.unit_cost @ units
    ) > uses * _half_width(worlds.now - after)
    searched = 0
    affordable = budget >= float(experiment.unit_cost.min())
    if affordable and (model_gap > _GAP_WARNING or not pays):
        # The model is too far off to allocate on, or what it chose does not pay, which it read
        # only at that one allocation: score the model's shapes at every scale by Monte Carlo, on
        # draws of their own, so the allocation chosen is not the one its own draws flatter, and
        # report on the main draws.
        candidates = _candidates(risk, weight, prior, experiment, budget, uses)
        search = _worlds(decision, prior, action, min(draws, _SEARCH_DRAWS), (seed, 1))
        scores = [
            uses * float(_regret_after(decision, prior, search, n, information).mean())
            + float(experiment.unit_cost @ n)
            for n in candidates
        ]
        searched = len(candidates)
        chosen = candidates[int(np.argmin(scores))]
        if not np.array_equal(chosen, units):
            units = chosen
            after = _regret_after(decision, prior, worlds, units, information)
            model_gap = gap(units, regret_now, uses * float(after.mean()))
    spend = float(experiment.unit_cost @ units)
    regret_after = uses * float(after.mean())
    error = uses * _half_width(worlds.now - after)
    regret_bound = uses * _upper_quantile(worlds.now, level)
    value = regret_now - regret_after - spend
    if model_gap > _GAP_WARNING:
        _log.warning(
            "the local regret model misses Monte Carlo by %.0f%%; the numbers reported are Monte "
            "Carlo's, and %s",
            100.0 * model_gap,
            f"the allocation is the best of {searched} that Monte Carlo scored"
            if searched
            else "the allocation was optimised on the model",
            extra={"chc_event": "design_gap", "model_gap": model_gap, "searched": searched},
        )

    diagonal = np.diag(weight)
    relevant = (
        diagonal > _RELEVANT * float(diagonal.max()) if diagonal.max() > 0.0 else diagonal > 0.0
    )
    sd_after = np.sqrt(np.diag(_posterior_covariance(prior, units * information)))
    unsigned = relevant & (np.abs(prior.mean) < _SIGNED * sd_after)
    plan = ", ".join(f"{n} in zone {z}" for z, n in enumerate(units.tolist()) if n > 0)
    if not affordable:
        found = "the budget buys not one unit"
    elif searched:
        found = (
            f"of the {searched} experiments Monte Carlo scored, from none through one unit's spend "
            "to the whole budget, the best does not pay more than its Monte Carlo error"
        )
    else:
        chose = f"runs {plan}" if plan else "runs none"
        found = (
            f"the experiment the local model chose, within {100.0 * model_gap:.0f}% of Monte "
            f"Carlo, {chose} and does not pay more than its Monte Carlo error"
        )
    verdict: DesignVerdict
    if units.sum() > 0 and value > error:
        verdict = "experiment"
        reason = (
            f"run {plan}: it removes {regret_now - regret_after:.3g} of regret for a spend of "
            f"{spend:.3g}, a value of {value:.3g} +- {error:.3g}"
        )
    elif unsigned.any():
        verdict = "abstain"
        reason = (
            f"zones {np.flatnonzero(unsigned).tolist()} matter and the sign of their channel is "
            f"unknown, and {found}"
        )
    else:
        verdict = "deploy"
        reason = (
            f"{found}: acting now costs {regret_now:.3g} of regret on average, and at most "
            f"{regret_bound:.3g} with probability {level:g}"
        )
    pinned = tuple(_pinned_zone(pin, prior, uses) for pin in local.pins)
    _log.info(
        "experiment design: %s",
        verdict,
        extra={
            "chc_event": "design",
            "verdict": verdict,
            "units": units.tolist(),
            "spend": spend,
            "value": value,
            "error": error,
        },
    )
    return ExperimentDesign(
        verdict=verdict,
        reason=reason,
        sample_sizes=tuple(int(n) for n in units),
        spend=spend,
        regret_now=regret_now,
        regret_after=regret_after,
        value=value,
        error=error,
        regret_bound=regret_bound,
        model_gap=model_gap,
        weights=_frozen(weight.copy()),
        pinned=pinned,
    )


DitherMethod = Literal["closed_form", "numeric"]


@dataclass(frozen=True)
class DitherDesign:
    """What :func:`dither_covariance` recommends: the dither's covariance over the levers, and
    what it and two simpler designs leave.

    ``value`` is ``tr(C V) + tr(Q (P0 + V)^-1)`` at ``covariance``: what the dither costs the plan
    plus the regret its information leaves. ``gap`` bounds how far ``value`` is above the least
    value any dither reaches: 0 for the closed form, which is exact; on the numeric route, the
    duality gap the stationarity of the solution certifies. ``diagonal`` and ``white`` are the
    best designs that dither each lever on its own, ``diag(diagonal)``, and every lever alike,
    ``white I``, with their values.
    """

    covariance: _Array  # (K, K), positive semidefinite
    value: float
    method: DitherMethod
    gap: float
    diagonal: _Array  # (K,)
    diagonal_value: float
    white: float
    white_value: float


_DITHER_PSD = 1e-10  # V*'s least eigenvalue, relative to its largest, above which it is a dither


def dither_covariance(
    cost: ArrayLike, weight: ArrayLike, precision: ArrayLike | None = None
) -> DitherDesign:
    """The covariance of a dither over several levers that buys the most for what it costs.

    Over ``K`` levers, a dither of covariance ``V`` costs the plan ``tr(C V)`` and leaves a regret
    of ``tr(Q (P0 + V)^-1)``: ``C`` is ``cost``, ``Q`` is ``weight`` and ``P0`` is ``precision``,
    zero where not given. :func:`dither_covariance` minimises their sum over ``V >= 0``.

    **Where the three come from.** A dither drawn afresh each period with covariance ``V``, run for
    ``n`` periods on a channel that does not move with the state (or a plant whose drift is known),
    with noise covariance ``Sigma``, adds ``n V (x) Sigma^-1`` to the precision of the channel; a
    plan whose regret in the channel's error is ``e' W e / 2`` (the decision weight,
    :meth:`chc.plan.CausalPlan.decision_weight`, or :func:`design_experiment`'s) then leaves
    ``tr(Q V^-1)``, ``Q`` the weight traced against ``Sigma`` over ``2 n``, in the units ``P0``
    holds the prior's precision in. ``C`` is half the sum, over the dither's periods, of the plan's
    action Hessian's diagonal blocks: what a dither costs a quadratic task, exactly, when the
    dither is open loop. This is the criterion of Wagenmaker, Simchowitz and Jamieson (ICML 2021)
    in its static case.

    **The closed form.** The sum is convex in ``V``, and stationary where ``C = M^-1 Q M^-1``,
    ``M = P0 + V``: at the matrix geometric mean of ``C^-1`` and ``Q`` (Pusz and Woronowicz, Rep.
    Math. Phys. 1975; Ando, Linear Algebra Appl. 1979; Bhatia, Positive Definite Matrices, 2007,
    ch. 4),

        M* = C^-1 # Q = C^-1/2 (C^1/2 Q C^1/2)^1/2 C^-1/2,

    the one positive definite solution of ``M C M = Q``. Where ``V* = M* - P0`` is positive
    semidefinite it is the optimum, with value ``2 tr((C^1/2 Q C^1/2)^1/2) - tr(C P0)``
    (``validation/dither_covariance.mac``). A diagonal ``C``, ``Q`` and ``P0`` give each lever the
    precision ``sqrt(Q_kk / C_kk)``, the level :func:`design_experiment`'s water-filling starts
    from. Where ``V*`` is not, the prior already holds more than the closed form asks along a
    direction, the bound ``V >= 0`` binds, and the optimum has no closed form: it is found over a
    Cholesky factor of ``V`` by L-BFGS from three starts, and ``gap`` bounds what that leaves.

    On ``C = [[2, 1], [1, 1]]``, ``Q = [[1, -1/2], [-1/2, 2]]`` and no prior the value is
    ``2 sqrt(3 + sqrt 7) = 4.7522``: 16 % below the best diagonal dither, ``4 sqrt 2 = 5.6569``,
    and 21 % below the best white one, 6. A correlated dither moves the levers together where the
    channel's error costs the plan least to learn.

    It is a certainty-equivalent design: ``Q`` is read at the estimate, and the information is the
    least-squares fit's. Where the dither also moves a state the plan reads, the state's own
    excitation adds information this leaves out.

    Raises:
        ValueError: on matrices that are not square of one size, a non-finite entry, a ``cost``
            or a ``weight`` that is not symmetric positive definite, or a ``precision`` that is not
            symmetric positive semidefinite.
    """
    c = np.array(cost, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.size == 0:
        raise ValueError(f"cost must be a square matrix, got shape {c.shape}")
    k = c.shape[0]
    c = _matrix(c, "cost", k)
    q = _matrix(weight, "weight", k)
    p0 = np.zeros((k, k)) if precision is None else _matrix(precision, "precision", k)
    c_root = _symmetric_root(c, "cost", definite=True)
    _symmetric_root(q, "weight", definite=True)
    _symmetric_root(p0, "precision", definite=False)
    c, q, p0 = (0.5 * (m + m.T) for m in (c, q, p0))
    c_inverse_root = np.linalg.inv(c_root)
    # In C's whitened coordinates, V = C^-1/2 Vw C^-1/2, the cost is tr(Vw) and the regret
    # tr(Qw (Pw + Vw)^-1): the optimum's precision is Qw^1/2.
    qw = _symmetric(c_root @ q @ c_root)
    pw = _symmetric(c_root @ p0 @ c_root)
    mw = _symmetric_root(qw, "weight", definite=True)
    vw = _symmetric(mw - pw)
    values = np.linalg.eigvalsh(vw)
    scale = float(np.max(np.abs(np.linalg.eigvalsh(mw))))
    if values[0] >= -_DITHER_PSD * scale:
        method: DitherMethod = "closed_form"
        vw = _psd(vw)
        gap = 0.0
    else:
        method = "numeric"
        vw, gap = _dither_numeric(qw, pw, vw, scale)
    covariance = _symmetric(c_inverse_root @ vw @ c_inverse_root)
    value = (
        2.0 * float(np.trace(mw)) - float(np.sum(c * p0))
        if method == "closed_form"
        else _dither_value(covariance, c, q, p0)
    )
    diagonal = _diagonal_dither(c, q, p0)
    white = _white_dither(c, q, p0)
    design = DitherDesign(
        covariance=_frozen(covariance),
        value=value,
        method=method,
        gap=gap,
        diagonal=_frozen(diagonal),
        diagonal_value=_dither_value(np.diag(diagonal), c, q, p0),
        white=white,
        white_value=_dither_value(white * np.eye(k), c, q, p0),
    )
    _log.info(
        "dither covariance: %s",
        method,
        extra={
            "chc_event": "dither_design",
            "method": method,
            "value": design.value,
            "gap": gap,
            "diagonal_value": design.diagonal_value,
            "white_value": design.white_value,
        },
    )
    return design


def _symmetric(m: _Array) -> _Array:
    return 0.5 * (m + m.T)


def _psd(m: _Array) -> _Array:
    """``m`` with its negative eigenvalues, rounding's, set to zero."""
    values, vectors = np.linalg.eigh(_symmetric(m))
    return _symmetric((vectors * np.clip(values, 0.0, None)) @ vectors.T)


def _dither_value(v: _Array, c: _Array, q: _Array, p0: _Array) -> float:
    """``tr(C V) + tr(Q (P0 + V)^-1)``; ``inf`` where ``P0 + V`` is singular."""
    try:
        factor = np.linalg.cholesky(_symmetric(p0 + v))
    except np.linalg.LinAlgError:
        return math.inf
    inverse = np.linalg.solve(factor.T, np.linalg.solve(factor, np.eye(v.shape[0])))
    return float(np.sum(c * v) + np.sum(q * inverse))


def _dither_numeric(qw: _Array, pw: _Array, start: _Array, scale: float) -> tuple[_Array, float]:
    """The optimum of ``tr(V) + tr(Qw (Pw + V)^-1)`` over ``V >= 0``, where the bound binds: by
    L-BFGS over a lower-triangular ``L``, ``V = L L'``, from the closed form's positive part, the
    best white and the best diagonal dither, at the scale where ``Qw^1/2``'s largest eigenvalue is
    1; with the duality gap its gradient certifies, in ``Qw``'s units.

    At a ``V >= 0`` with gradient ``G = I - M^-1 Qw M^-1``, convexity gives every ``V' >= 0`` a
    value at least ``f(V) + tr(G (V' - V))``, and the optimum has ``tr(V') <= f(V)``: so the gap
    is at most ``tr(G V) - f(V) min(lambda_min(G), 0)``."""
    k = qw.shape[0]
    q, p = qw / scale**2, pw / scale
    rows, cols = np.tril_indices(k)
    eye = np.eye(k)

    def unpack(theta: _Array) -> _Array:
        lower = np.zeros((k, k))
        lower[rows, cols] = theta
        return lower

    def objective(theta: _Array) -> tuple[float, _Array]:
        lower = unpack(theta)
        m = _symmetric(p + lower @ lower.T)
        try:
            factor = np.linalg.cholesky(m)
        except np.linalg.LinAlgError:
            return math.inf, np.zeros_like(theta)
        inverse = np.linalg.solve(factor.T, np.linalg.solve(factor, eye))
        gradient = eye - inverse @ q @ inverse
        value = float(np.trace(lower @ lower.T) + np.sum(q * inverse))
        return value, (2.0 * gradient @ lower)[rows, cols]

    def factor_of(v: _Array) -> _Array:
        values, vectors = np.linalg.eigh(_symmetric(v))
        root = vectors * np.sqrt(np.clip(values, 0.0, None))
        # a full-rank start, so the factor can turn toward any direction
        return np.linalg.cholesky(root @ root.T + 1e-3 * eye)[rows, cols]

    diag_q, diag_p = np.diag(q), np.diag(p)
    starts = [
        _psd(start / scale),
        np.diag(np.maximum(np.sqrt(diag_q) - diag_p, 0.0)),
        max(float(np.sqrt(np.trace(q) / k)) - float(np.trace(p)) / k, 0.0) * eye,
    ]
    best: tuple[float, _Array] = (math.inf, starts[0])
    for v0 in starts:
        result = optimize.minimize(
            objective,
            factor_of(v0),
            jac=True,
            method="L-BFGS-B",
            options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10_000, "maxcor": 30},
        )
        lower = unpack(result.x)
        value = objective(result.x)[0]
        if value < best[0]:
            best = (value, lower @ lower.T)
    value, v = best
    m = _symmetric(p + v)
    inverse = np.linalg.inv(m)
    gradient = _symmetric(eye - inverse @ q @ inverse)
    least = float(np.linalg.eigvalsh(gradient)[0])
    gap = max(float(np.sum(gradient * v)) - value * min(least, 0.0), 0.0)
    return scale * _psd(v), scale * gap


def _diagonal_dither(c: _Array, q: _Array, p0: _Array) -> _Array:
    """The best ``diag(v)``, ``v >= 0``: in closed form where ``P0`` is diagonal, the lever's
    precision ``sqrt(Q_kk / C_kk)`` less the prior's, and by L-BFGS-B from there otherwise."""
    start = np.maximum(np.sqrt(np.diag(q) / np.diag(c)) - np.diag(p0), 0.0)
    if np.array_equal(p0, np.diag(np.diag(p0))):
        return start
    k = c.shape[0]

    def objective(v: _Array) -> tuple[float, _Array]:
        m = p0 + np.diag(v)
        try:
            inverse = np.linalg.inv(np.linalg.cholesky(m))
        except np.linalg.LinAlgError:
            return math.inf, np.zeros(k)
        inverse = inverse.T @ inverse
        return (
            float(np.diag(c) @ v + np.sum(q * inverse)),
            np.diag(c) - np.diag(inverse @ q @ inverse),
        )

    result = optimize.minimize(
        objective,
        start + 1e-3 * float(np.max(start, initial=1.0)),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * k,
        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10_000},
    )
    return np.clip(result.x, 0.0, None)


def _white_dither(c: _Array, q: _Array, p0: _Array) -> float:
    """The best ``v I``, ``v >= 0``: the root of the value's derivative in ``v``, which grows with
    ``v``; ``sqrt(tr Q / tr C)`` with no prior.

    The derivative is ``tr C - sum_k q_k / (l_k + v)^2`` over the prior's eigenvalues ``l_k``,
    ``q_k`` the weight's diagonal in their basis, all positive. Below ``sqrt(q_0 / tr C) - l_0``,
    ``l_0`` the least, its first term alone keeps it negative, so the root is at least that, and
    the search starts there, where no term divides by zero (``validation/dither_covariance.mac``).
    """
    trace_c = float(np.trace(c))
    if not p0.any():
        return math.sqrt(float(np.trace(q)) / trace_c)
    values, vectors = np.linalg.eigh(p0)
    q_rotated = np.diag(vectors.T @ q @ vectors)

    def slope(v: float) -> float:
        return trace_c - float(np.sum(q_rotated / (values + v) ** 2))

    low = max(0.0, math.sqrt(q_rotated[0] / trace_c) - float(values[0]))
    if slope(low) >= 0.0:  # 0, or with one lever the bound itself up to rounding
        return low
    high = max(math.sqrt(float(np.trace(q)) / trace_c), low)
    while slope(high) < 0.0:
        high *= 2.0
    return float(optimize.brentq(slope, low, high, xtol=1e-300))


@dataclass(frozen=True)
class ProbeDesign:
    """What :func:`probe_design` recommends: the probe's covariance over the levers, what the gain
    re-solved after it is expected to lose, and the same for a white probe of the same power.

    ``regret`` is the expected excess of the re-solved gain's average cost per period over the
    optimal gain's, to first order in ``1 / periods``: ``Phi(U) / periods`` at ``covariance``.
    ``gap`` bounds how far it is above the least any probe of that power reaches, in the same
    units: 0 for the closed form, the duality gap the gradient certifies on the numeric route.
    ``white_regret`` is ``Phi`` of ``power / m`` times the identity over ``periods``, and ``ratio``
    ``regret / white_regret``, at most 1. ``cost`` and ``white_cost`` are what each probe adds to
    the plan's average cost per period while it runs, exactly: ``tr((R + B' P B) U)``, ``P`` the
    cost-to-go of ``gain``, the gain the probe runs under.
    """

    covariance: _Array  # (m, m), positive semidefinite, of trace `power`
    regret: float
    method: DitherMethod
    gap: float
    white_regret: float
    ratio: float
    cost: float
    white_cost: float
    gain: _Array  # (m, n), u = -gain x + probe


def probe_design(
    a: ArrayLike,
    b: ArrayLike,
    q: ArrayLike,
    r: ArrayLike,
    *,
    noise: ArrayLike,
    power: float,
    periods: int,
    weight: ArrayLike | None = None,
    gain: ArrayLike | None = None,
) -> ProbeDesign:
    """The covariance of a probe, at a cap on its power, that leaves the LQR gain re-solved after
    it the least expected regret.

    The plant is ``x_{t+1} = A x_t + B u_t + w_t``, ``w_t ~ N(0, noise)`` independent over ``t``,
    and the decision is the LQR gain of the average cost per period ``E[x' Q x + u' R u]``,
    ``u = -K x``. For ``periods`` periods the plant runs under ``gain``, the LQR gain on ``(A, B)``
    unless given, plus a probe drawn afresh each period, ``u_t = -gain x_t + eta_t``,
    ``eta_t ~ N(0, U)``; ``[A B]`` is then fitted by least squares on ``(x_t, u_t)`` and the gain
    re-solved on the fit. :func:`probe_design` chooses the ``U >= 0`` with ``tr U = power`` that
    minimises what the re-solved gain is expected to lose.

    **The criterion** is the task-optimal one of Wagenmaker, Simchowitz and Jamieson (ICML 2021):
    the trace of the decision weight against the inverse information. To second order in the fit's
    error ``e`` over ``theta = (A.ravel(), B.ravel())``, the re-solved gain loses ``e' W e / 2`` per
    period, ``W`` the decision weight: ``weight``, by default the LQR's own,
    ``W = 2 J' (G (x) Sigma*) J``, ``J`` the gain's Jacobian in ``theta``, ``G = R + B' P* B`` and
    ``Sigma*`` the closed loop's stationary state covariance under ``noise``
    (:func:`chc.regret.ce_regret_law`'s second-order law, doubled to a Hessian). To first order the
    fit's error in ``[A B]``, row by row, is ``N(0, noise (x) Gamma^-1 / periods)``, ``Gamma`` the
    stationary covariance of ``z = (x, u)`` in the probed loop, so the expected regret is
    ``Phi(U) / periods`` with ``Phi(U) = tr(Wn Gamma(U)^-1) / 2``, ``Wn`` the weight traced against
    ``noise`` over the plant's rows, ``(n + m) x (n + m)``: the re-plan price ``tr(W S) / 2`` of
    :meth:`chc.gate.ChannelMove.price` at the fit's covariance ``S``.

    **Where the probe's information goes.** ``Gamma^-1 = E Sx^-1 E' + F U^-1 F'``, ``E = [I; 0]``
    and ``F = [gain'; I]``: the fit reads ``B`` off the probe alone, ``U``, and the closed loop
    ``A - B gain`` off the state's excitation, ``Sx(U)``, the stationary solution of
    ``Sx = A_cl Sx A_cl' + B U B' + noise`` (``validation/dynamic_probe_design.mac``). So

        Phi(U) = (tr(Wn_xx Sx(U)^-1) + tr(F' Wn F U^-1)) / 2,

    convex in ``U``, since ``Sx`` is affine in it, and decreasing, so the cap binds. Where the
    weight holds no state block the first term vanishes and the optimum is in closed form,
    ``U* = power M / tr M`` with ``M = (F' Wn F)^1/2``. That is the static case: a weight on ``B``
    alone, ``n m`` square and ravelled as ``B.ravel()`` (as
    :meth:`chc.plan.CausalPlan.decision_weight` ravels the one-step channel), gives
    :func:`dither_covariance`'s covariance with ``cost`` the identity and ``weight``
    ``F' Wn F / (2 periods)`` at the power that is its trace, and ``regret`` its regret term. A cap
    on ``tr(C U)`` in place of ``tr U``, such as the probe's own cost, ``C = R + B' P B``, is this
    one on the levers scaled by ``C^1/2``: pass ``B C^-1/2`` and ``C^-1/2 R C^-1/2``. Otherwise the
    optimum is found over a Cholesky factor of ``U`` scaled to the cap, by L-BFGS from the white
    probe and from the closed form of the second term alone, and ``gap`` is
    ``tr(D U) - power lambda_min(D)``, ``D`` the gradient of ``Phi``, over ``periods``: the
    convex problem's Frank-Wolfe gap, a bound on how far the probe is from the best.

    On ``A = [[0.9, 0.2], [-0.1, 0.7]]``, ``B = [[1, 0.1], [0.3, 0.2]]``, ``Q = I``,
    ``R = diag(1, 0.05)``, unit noise and power 1, the gain leans on the weak, cheap second lever;
    the design gives it 0.83 of the power and leaves 0.753 of the white probe's regret. At power 10,
    where the state's excitation counts, it leaves 0.942, where the closed form of the second term
    alone leaves 1.004.

    **What it leaves out.** It is first order in ``1 / periods``: the fit's error is not exactly
    Gaussian, a fit of a few periods inflates its covariance, by ``periods / (periods - n - m -
    1)`` for Gaussian regressors, and the exact regret's mean is infinite, since a fit can give a
    gain that does not stabilise the plant (:func:`chc.regret.ce_regret_law` reads its law). It
    is a certainty-equivalent design: the weight and ``Gamma`` are read at the ``(A, B)`` given,
    not at the truth. The probe is drawn afresh each period; a probe correlated over time, whose
    spectrum Wagenmaker et al. design, can excite the state more at the same power, and is not
    built. The fit is of all of ``[A B]``: a fit that holds ``A`` known reads ``B`` off ``gain x``
    as well, and learns more than this credits. The cap is the caller's: what the probe costs over
    the run, against the regret it removes over the uses after it, is not traded here.

    Raises:
        ValueError: on matrices of shapes that do not match, a non-finite entry, a ``q`` that is
            not symmetric positive semidefinite, an ``r`` or a ``noise`` that is not symmetric
            positive definite, a plant with no stabilising LQR gain, a ``gain`` whose closed loop
            is not stable, a ``weight`` that is not ``n (n + m)`` or ``n m`` square, not symmetric
            positive semidefinite or zero, a ``power`` that is not positive and finite, or
            ``periods`` of at most ``n + m + 1``.
    """
    if not (math.isfinite(power) and power > 0.0):
        raise ValueError(f"power must be positive and finite, got {power}")
    loop = _probed_loop(a, b, q, r, noise, gain)
    n, m = loop.b.shape
    if periods <= n + m + 1:
        raise ValueError(f"periods must exceed n + m + 1 = {n + m + 1}, got {periods}")
    w = _probe_weight(loop, weight)
    traced = _traced_weight(w, loop.noise, n, m)
    f = np.vstack([loop.gain.T, np.eye(m)])
    h_state, h_probe = traced[:n, :n], _symmetric(f.T @ traced @ f)

    def value(u: _Array) -> tuple[float, _Array]:
        return _probe_value(u, loop.closed, loop.b, loop.noise, h_state, h_probe)

    white_covariance = (power / m) * np.eye(m)
    white_value = value(white_covariance)[0]
    method: DitherMethod = "closed_form"
    gap = 0.0
    if m == 1:
        covariance, phi = white_covariance, white_value
    elif not np.any(h_state):
        root = _psd_root(h_probe)
        covariance = power * root / float(np.trace(root))
        phi = float(np.trace(root)) ** 2 / (2.0 * power)
    else:
        method = "numeric"
        covariance, phi, gap = _probe_numeric(value, h_probe, power, white_value)
    cost_to_go = solve_discrete_lyapunov(loop.closed.T, loop.q + loop.gain.T @ loop.r @ loop.gain)
    probe_price = _symmetric(loop.r + loop.b.T @ _symmetric(cost_to_go) @ loop.b)
    design = ProbeDesign(
        covariance=_frozen(_symmetric(covariance)),
        regret=float(phi) / periods,
        method=method,
        gap=gap / periods,
        white_regret=white_value / periods,
        ratio=float(phi) / white_value,
        cost=float(np.sum(probe_price * covariance)),
        white_cost=power / m * float(np.trace(probe_price)),
        gain=_frozen(loop.gain.copy()),
    )
    _log.info(
        "probe design: %s",
        method,
        extra={
            "chc_event": "probe_design",
            "method": method,
            "regret": design.regret,
            "gap": design.gap,
            "white_regret": design.white_regret,
            "ratio": design.ratio,
        },
    )
    return design


@dataclass(frozen=True)
class _ProbedLoop:
    """The plant, its LQR task and optimum, and the loop the probe runs in."""

    a: _Array
    b: _Array
    q: _Array
    r: _Array
    noise: _Array
    optimal: _Array  # K*, u = -K* x
    cost_to_go: _Array  # P*
    gain: _Array
    closed: _Array  # A - B gain


def _probed_loop(
    a: ArrayLike, b: ArrayLike, q: ArrayLike, r: ArrayLike, noise: ArrayLike, gain: ArrayLike | None
) -> _ProbedLoop:
    b_matrix = np.array(b, dtype=np.float64)
    if b_matrix.ndim != 2 or b_matrix.size == 0:
        raise ValueError(f"b must be a non-empty matrix, got shape {b_matrix.shape}")
    _finite(b_matrix, "b")
    n, m = b_matrix.shape
    a_matrix = _matrix(a, "a", n)
    q_matrix, r_matrix, noise_matrix = (
        _matrix(q, "q", n),
        _matrix(r, "r", m),
        _matrix(noise, "noise", n),
    )
    _symmetric_root(q_matrix, "q", definite=False)
    _symmetric_root(r_matrix, "r", definite=True)
    _symmetric_root(noise_matrix, "noise", definite=True)
    q_matrix, r_matrix, noise_matrix = (_symmetric(x) for x in (q_matrix, r_matrix, noise_matrix))
    try:
        optimal, cost_to_go = dlqr(a_matrix, b_matrix, q_matrix, r_matrix)
    except (np.linalg.LinAlgError, ValueError) as error:
        raise ValueError("the plant has no stabilising LQR gain") from error
    running = optimal if gain is None else np.array(gain, dtype=np.float64)
    if running.shape != (m, n):
        raise ValueError(f"gain must have shape ({m}, {n}), got {running.shape}")
    _finite(running, "gain")
    closed = a_matrix - b_matrix @ running
    radius = float(np.max(np.abs(np.linalg.eigvals(closed))))
    if radius >= 1.0:
        raise ValueError(
            f"the loop the probe runs in is not stable, spectral radius {radius:.4g}: it has no"
            " stationary covariance"
        )
    return _ProbedLoop(
        a_matrix, b_matrix, q_matrix, r_matrix, noise_matrix, optimal, cost_to_go, running, closed
    )


def _probe_weight(loop: _ProbedLoop, weight: ArrayLike | None) -> _Array:
    """The decision weight over ``(A.ravel(), B.ravel())``: the LQR's own, or ``weight``, one over
    ``B.ravel()`` alone padded with zeros."""
    n, m = loop.b.shape
    d = n * (n + m)
    if weight is None:
        jacobian = _gain_jacobian(loop.a, loop.b, loop.optimal, loop.cost_to_go, loop.r)
        jacobian = jacobian.reshape(d, m * n)
        spread = solve_discrete_lyapunov(loop.a - loop.b @ loop.optimal, loop.noise)
        curvature = loop.r + loop.b.T @ loop.cost_to_go @ loop.b
        w = 2.0 * jacobian @ np.kron(_symmetric(curvature), _symmetric(spread)) @ jacobian.T
    else:
        w = np.array(weight, dtype=np.float64)
        if w.shape not in ((d, d), (n * m, n * m)):
            raise ValueError(
                f"weight must have shape ({d}, {d}) over (A.ravel(), B.ravel()) or"
                f" ({n * m}, {n * m}) over B.ravel(), got {w.shape}"
            )
        _finite(w, "weight")
        _symmetric_root(w, "weight", definite=False)
        if w.shape[0] != d:
            full = np.zeros((d, d))
            full[n * n :, n * n :] = w
            w = full
    if not np.any(w):
        raise ValueError("weight is zero: no probe changes the regret")
    return _symmetric(w)


def _psd_root(m: _Array) -> _Array:
    """The symmetric square root of ``m``'s positive part."""
    values, vectors = np.linalg.eigh(_symmetric(m))
    return _symmetric((vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.T)


def _traced_weight(w: _Array, noise: _Array, n: int, m: int) -> _Array:
    """``Wn_jl = sum_ik W_(i,j),(k,l) noise_ki`` over the regressors ``z = (x, u)``, ``(i, j)`` the
    entry of ``[A B]`` in row ``i`` and column ``j``: ``tr(W (noise (x) X)) = tr(Wn X)``."""
    width = n + m
    order = np.array(
        [i * n + j if j < n else n * n + i * m + (j - n) for i in range(n) for j in range(width)]
    )
    rows = w[np.ix_(order, order)].reshape(n, width, n, width)
    return _symmetric(np.einsum("ijkl,ki->jl", rows, noise))


def _probe_value(
    u: _Array, closed: _Array, b: _Array, noise: _Array, h_state: _Array, h_probe: _Array
) -> tuple[float, _Array]:
    """``Phi(U) = (tr(Wn_xx Sx(U)^-1) + tr(F' Wn F U^-1)) / 2`` and its gradient in ``U``,
    ``-(B' L*(Sx^-1 Wn_xx Sx^-1) B + U^-1 F' Wn F U^-1) / 2``, ``L*`` the adjoint Lyapunov map;
    ``inf`` where ``U`` is singular and the second term is not zero."""
    state = _symmetric(solve_discrete_lyapunov(closed, noise + b @ u @ b.T))
    state_inverse = np.linalg.inv(state)
    pull = _symmetric(state_inverse @ h_state @ state_inverse)
    gradient = -0.5 * b.T @ solve_discrete_lyapunov(closed.T, pull) @ b
    total = float(np.sum(h_state * state_inverse))
    if np.any(h_probe):
        try:
            factor = np.linalg.cholesky(_symmetric(u))
        except np.linalg.LinAlgError:
            return math.inf, np.zeros_like(u)
        inverse = np.linalg.solve(factor.T, np.linalg.solve(factor, np.eye(u.shape[0])))
        total += float(np.sum(h_probe * inverse))
        gradient = gradient - 0.5 * inverse @ h_probe @ inverse
    return 0.5 * total, _symmetric(gradient)


def _probe_numeric(
    value: Callable[[_Array], tuple[float, _Array]], h_probe: _Array, power: float, scale: float
) -> tuple[_Array, float, float]:
    """The least ``Phi(U)`` over ``U >= 0``, ``tr U = power``: by L-BFGS over a lower-triangular
    ``L``, ``U = power L L' / tr(L L')``, from the white probe and the second term's closed form,
    with ``Phi`` scaled by the white probe's; and the Frank-Wolfe gap its gradient ``D``
    certifies, ``tr(D U) - power lambda_min(D)``, which bounds ``Phi(U) - Phi*`` by convexity."""
    k = h_probe.shape[0]
    rows, cols = np.tril_indices(k)
    eye = np.eye(k)

    def covariance_of(theta: _Array) -> tuple[_Array, _Array, float]:
        lower = np.zeros((k, k))
        lower[rows, cols] = theta
        square = lower @ lower.T
        return lower, square, float(np.trace(square))

    def objective(theta: _Array) -> tuple[float, _Array]:
        lower, square, size = covariance_of(theta)
        if not size > 0.0:
            return math.inf, np.zeros_like(theta)
        phi, gradient = value(power * square / size)
        if not math.isfinite(phi):
            return math.inf, np.zeros_like(theta)
        tangent = (power / size) * (gradient - (float(np.sum(gradient * square)) / size) * eye)
        return phi / scale, (2.0 * tangent @ lower / scale)[rows, cols]

    root = _psd_root(h_probe)
    starts = [eye]
    if np.trace(root) > 0.0:
        starts.append(root / np.trace(root) + 1e-3 * eye)
    best: tuple[float, _Array] = (math.inf, eye)
    for start in starts:
        result = optimize.minimize(
            objective,
            np.linalg.cholesky(_symmetric(start))[rows, cols],
            jac=True,
            method="L-BFGS-B",
            options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10_000, "maxcor": 30},
        )
        _, square, size = covariance_of(result.x)
        candidate = power * square / size
        phi = value(candidate)[0]
        if phi < best[0]:
            best = (phi, candidate)
    phi, covariance = best
    gradient = value(covariance)[1]
    least = float(np.linalg.eigvalsh(gradient)[0])
    gap = max(float(np.sum(gradient * covariance)) - power * least, 0.0)
    return covariance, phi, gap
