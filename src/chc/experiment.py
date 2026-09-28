"""Which experiment, if any, to run before a decision: units per zone, the regret they remove, and a
verdict to deploy now, experiment first, or abstain.

The decision (:class:`ZoneDecision`) is one-shot, over ``K`` zones. Zone ``k``'s lever ``u_k`` moves
the zones' states through its own channel ``b_k``, and a known coupling spreads the effect::

    x = x0 + coupling diag(b) u,   J(u; b) = (x - x*)' Q (x - x*) / 2 + u' R u / 2,   lo <= u <= hi.

The channels are known up to a Gaussian posterior (:class:`ChannelPrior`), and acting now on its
mean costs regret: what the action loses against the one that knew ``b``. An experiment
(:class:`ZoneExperiment`) sets zone ``k``'s lever to ``+probe_k`` or ``-probe_k`` at random on
``n_k`` units, each costing ``unit_cost_k`` and each adding ``probe_k^2 / noise_sd_k^2`` to the
precision of ``b_k``. :func:`design_experiment` chooses the ``n_k`` that minimise the regret left
after the experiment, times the number of times the decision will be used, plus what the experiment
spends, within a budget. Then it measures the regret before and after by Monte Carlo, re-solving the
decision on every draw, and every number it reports is that measurement.

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

What the local model gets wrong, and what is done about it:

* **A wide prior.** At a prior sd of 37-50% of ``b`` the lab's local model put the regret after the
  experiment 32% low. The numbers reported are Monte Carlo's whatever the model says;
  ``model_gap`` is how far the model was from them, and a gap above 25% is logged as a warning,
  because the allocation was optimised on the model.
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

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, special

_log = logging.getLogger(__name__)

DesignVerdict = Literal["deploy", "experiment", "abstain"]

_Array = NDArray[np.float64]

_KAPPA = 0.5  # sigma points sit sqrt(K + kappa) prior standard deviations from the mean
_SIGNED = 2.0  # a channel's sign is known once its mean is this many posterior sds from zero
_RELEVANT = 1e-3  # a zone matters when its decision weight is this share of the largest
_GAP_WARNING = 0.25
_MIN_DRAWS = 100


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
        object.__setattr__(
            self, "_action_root", _frozen(np.linalg.cholesky(fields["action_weight"]).T)
        )

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
    prior standard deviations, and the expected regret the pin costs now, times ``uses``."""

    zone: int
    z: float
    regret: float


@dataclass(frozen=True)
class ExperimentDesign:
    """What :func:`design_experiment` recommends, and the numbers behind it.

    The regrets are expectations over the prior, times ``uses``: ``regret_now`` of acting on the
    prior mean, ``regret_after`` of acting on the posterior mean once ``sample_sizes`` units have
    run. Both are Monte Carlo's, over draws of the channels with the decision re-solved on every
    draw, and ``error`` is the 95% half-width of their difference. ``value`` is what the experiment
    removes less what it spends. ``regret_bound`` is the ``level`` quantile of the regret of acting
    now: with that probability under the prior, deploying now loses at most this much.
    ``model_gap`` is the local model's largest relative miss against the two Monte Carlo regrets,
    and the allocation was optimised on that model. ``weights`` is the prior-averaged decision
    weight ``W``.
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
) -> _Array:
    """Minimise ``risk(n) + unit_cost . n`` over ``n >= 0`` with ``unit_cost . n <= budget``, from
    the water-filling start and from an even split, in spend shares so every variable is O(1)."""
    k = unit_cost.size
    scale = max(risk(np.zeros(k)), np.finfo(float).tiny)

    def objective(share: _Array) -> float:
        n = np.clip(share, 0.0, None) * budget / unit_cost
        return (risk(n) + float(unit_cost @ n)) / scale

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
class _MonteCarlo:
    now: _Array  # per draw, the regret of acting on the prior mean
    after: _Array  # per draw, the regret of acting on the posterior mean after the experiment


def _monte_carlo(
    d: ZoneDecision,
    prior: ChannelPrior,
    action: _Array,
    units: NDArray[np.int64],
    information: _Array,
    draws: int,
    seed: int,
) -> _MonteCarlo:
    rng = np.random.default_rng(seed)
    b = prior.mean + rng.standard_normal((draws, d.zones)) @ prior._root.T
    observed = np.flatnonzero(units > 0)
    noise = 1.0 / np.sqrt(units[observed] * information[observed])
    shocks = rng.standard_normal((draws, observed.size)) * noise
    gain = np.zeros((d.zones, observed.size))
    if observed.size:
        innovation = prior.covariance[np.ix_(observed, observed)] + np.diag(noise**2)
        gain = np.linalg.solve(innovation, prior.covariance[observed]).T
    now, after = np.empty(draws), np.empty(draws)
    for i in range(draws):
        best = _cost(d, _solve(d, b[i]).action, b[i])
        now[i] = _cost(d, action, b[i]) - best
        if observed.size:
            mean = prior.mean + gain @ (b[i, observed] + shocks[i] - prior.mean[observed])
            after[i] = _cost(d, _solve(d, mean).action, b[i]) - best
        else:
            after[i] = now[i]
    return _MonteCarlo(now, after)


def _pinned_zone(pin: _Pin, prior: ChannelPrior, uses: float) -> PinnedZone:
    variance = float(pin.gradient @ prior.covariance @ pin.gradient)
    z = pin.multiplier / math.sqrt(variance) if variance > 0.0 else math.inf
    return PinnedZone(pin.zone, z, uses * _pin_regret(pin, variance))


def _relative_gap(model: float, measured: float) -> float:
    if measured == 0.0:
        return 0.0 if model == 0.0 else math.inf
    return abs(model / measured - 1.0)


def design_experiment(
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

    * ``"experiment"``: the regret the experiment removes exceeds its spend by more than the Monte
      Carlo error;
    * ``"abstain"``: it does not, and a zone that matters stays within two posterior standard
      deviations of zero, so the sign of its channel -- which way to act there -- is still unknown;
    * ``"deploy"``: it does not, and acting now costs ``regret_now`` on average and at most
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

    def risk(n: _Array) -> float:
        return uses * _expected_regret(weight, local.pins, prior, n * information)

    units = np.zeros(k, dtype=np.int64)
    if budget > 0.0:
        start = _water_filling(weight, prior, experiment, budget, uses)
        units = _whole_units(
            _allocate(risk, start, experiment.unit_cost, budget), experiment.unit_cost, budget
        )
    spend = float(experiment.unit_cost @ units)
    measured = _monte_carlo(decision, prior, local.action, units, information, draws, seed)
    regret_now = uses * float(measured.now.mean())
    regret_after = uses * float(measured.after.mean())
    error = uses * 1.96 * float(np.std(measured.now - measured.after, ddof=1)) / math.sqrt(draws)
    regret_bound = uses * float(np.quantile(measured.now, level))
    value = regret_now - regret_after - spend
    model_gap = max(
        _relative_gap(risk(np.zeros(k)), regret_now),
        _relative_gap(risk(units.astype(np.float64)), regret_after),
    )
    if model_gap > _GAP_WARNING:
        _log.warning(
            "the local regret model misses Monte Carlo by %.0f%%; the numbers reported are Monte "
            "Carlo's, and the allocation was optimised on the model",
            100.0 * model_gap,
            extra={"chc_event": "design_gap", "model_gap": model_gap},
        )

    diagonal = np.diag(weight)
    relevant = (
        diagonal > _RELEVANT * float(diagonal.max()) if diagonal.max() > 0.0 else diagonal > 0.0
    )
    sd_after = np.sqrt(np.diag(_posterior_covariance(prior, units * information)))
    unsigned = relevant & (np.abs(prior.mean) < _SIGNED * sd_after)
    plan = ", ".join(f"{n} in zone {z}" for z, n in enumerate(units.tolist()) if n > 0)
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
            "unknown, and no experiment the budget buys both identifies it and pays"
        )
    else:
        verdict = "deploy"
        reason = (
            f"no experiment the budget buys pays: acting now costs {regret_now:.3g} of regret on "
            f"average, and at most {regret_bound:.3g} with probability {level:g}"
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
