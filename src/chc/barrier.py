"""Safety certificates when the control effect is only PARTIALLY IDENTIFIED.

The pessimism line (:mod:`chc.support`, :mod:`chc.uncertainty`) spends an identification radius on
*performance*: a worse effect estimate buys a wider penalty and a costlier plan. This module spends
the same radius on *safety*, and the accounting is different in a way that matters operationally.

For a control-affine plant ``xdot = f(x) + B u`` and a safe set ``{h >= 0}``, the barrier condition
is ``grad h . (f + B u) >= -alpha * h``. Write ``a = grad h . f`` for the drift term, ``g`` for the
control channel and ``U`` for the actuation limit. If the effect matrix is known only up to a radius
``Delta`` -- here an *identification* radius from the bounded-density-ratio model at level
``Gamma``, not a disturbance bound -- the guaranteed barrier derivative is ``a + g*u - d*|u|`` with
``d = Delta * ||grad h||``, and:

* the best guaranteed margin is ``a + (|g| - d)_+ * U``, so the margin lost to an identification
  radius ``d`` is ``U * min(d, |g|)`` -- **first order in ``d`` while control authority lasts**,
  then flat: past ``d = |g|`` there is nothing left to lose;
* at ``d >= |g|`` the robust-optimal action is exactly **zero**: with the sign of the channel
  unidentified, acting cannot improve the guarantee (the safety analogue of the sign threshold of
  :func:`chc.research.regret.partial_id_control_certificate`);
* certification is sharp at ``d* = |g| - D/U``, where ``D = -alpha*h - a`` is the deficit the drift
  leaves for the controller, which inverts to a largest tolerable sensitivity ``Gamma*``. Three
  regimes, and only the third is the interesting one: ``D <= 0`` (the drift alone is safe, no radius
  can break it), ``D > U*|g|`` (nominally infeasible -- not certified even at ``d = 0``), and
  ``0 < D <= U*|g|`` (``d*`` in ``[0, |g|)``, the case the threshold formula describes).

The tightening itself is the standard robust-CBF one (Jankovic 2018; Kolathaya-Ames 2019). What is
new here is the source of the radius and the resulting **order dichotomy**: the same effect error
costs performance regret only at second order (``L_reg * e**2``): an interior optimum has a
vanishing gradient, while a *binding constraint* has no envelope protection and pays at first
order. Objectives are protected; constraints are not. That comparison lives in ``d < |g|``, which is
also where any usable controller lives -- once the radius exceeds the channel the safety loss is
capped at ``U*|g|`` simply because the guarantee is already gone. Within that band, and below
``e = U / L_reg``, safety is the larger loss by a ratio that diverges as the error shrinks.

**Uncertainty-set assumption.** ``a + <w, u> - d*||u||`` is the worst case for a *symmetric,
isotropic* set: ``||u||_2 <= U`` for the action and a dual-norm ball of radius ``d`` for the channel
error. It is exact there (Cauchy-Schwarz is attained). For box actuation limits or anisotropic
channel uncertainty the worst case is the corresponding support function and the closed forms below
are only an outer bound. For an *asymmetric* scalar interval ``g_true in [g - d_lo, g + d_hi]`` the
two branch slopes differ (``g - d_lo`` right, ``g + d_hi`` left) and the zero-action rule reads in
its general form: **acting is pointless exactly when the identified interval contains zero.**

NumPy, like the rest of :mod:`chc.research.sensitivity`, so it is independent of the JAX x64 flag.
Maxima ``validation/barrier_feasibility.mac``, Rocq ``proofs/barrier_feasibility.v``.

**The plant's noise, read apart** (experimental). :func:`exit_probability` bounds the chance that
a discrete-time closed loop leaves a :class:`QuadraticBarrier`'s safe set within ``K`` steps under
its noise, where the barrier holds in expectation, ``E[h(x') | x] >= alpha h(x) + slack``: Ville's
inequality on Kushner's supermartingale, as Cosner, Culbertson, Taylor and Ames state it (RSS 2023,
Thm 5). It reads a feedback law and refuses an open-loop plan, and it is no part of any audit of
the deterministic condition above. Maxima ``validation/exit_probability.mac``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from chc.evaluation import AffinePolicy, LinearGaussianPlant
from chc.uncertainty import confounding_robust_inflation

__all__ = [
    "BarrierConfoundingCurve",
    "ExitProbability",
    "QuadraticBarrier",
    "SafetyFilterBenchmark",
    "admissible_action_interval",
    "barrier_confounding_certificate",
    "barrier_gamma_star",
    "control_channel",
    "exit_probability",
    "identification_radius_threshold",
    "robust_barrier_margin",
    "robust_safe_action",
    "robust_safety_filter",
    "safety_filter_benchmark",
]


def _numbers(**values: float | NDArray[np.float64]) -> None:
    """Refuse a nan among ``values``. Every comparison with nan is false, so given a nan radius,
    channel, drift or rate the filter would pass the nominal action unfiltered, and
    :func:`robust_safe_action` would grant full authority."""
    named = [name for name, value in values.items() if np.isnan(value).any()]
    if named:
        raise ValueError(f"{', '.join(named)} must be numbers, not nan")


def control_channel(grad_h: NDArray[np.float64], b_hat: NDArray[np.float64]) -> float:
    """``||B^T grad h||`` -- how strongly the control can move the barrier, in the best direction.

    The multivariate stand-in for the scalar ``|g|``. Cauchy-Schwarz is tight at
    ``u = U * w / ||w||``, so every bound below is attained rather than conservative.
    """
    return float(np.linalg.norm(np.atleast_2d(b_hat).T @ np.asarray(grad_h, dtype=np.float64)))


def robust_barrier_margin(drift: float, channel: float, radius: float, u_max: float) -> float:
    """Best barrier derivative guaranteed against every effect inside the identified set.

    ``drift + (channel - radius)_+ * u_max``. The clip at zero is the whole content: past
    ``radius = channel`` extra actuation buys nothing, because the adversary can cancel it. So the
    margin lost relative to a perfectly identified channel is ``u_max * min(radius, channel)`` --
    first order in the radius until the authority runs out, then constant. Exact for a Euclidean
    action ball and an isotropic channel radius (see the module docstring).
    """
    _numbers(drift=drift, channel=channel, radius=radius, u_max=u_max)
    if u_max < 0.0:
        raise ValueError(f"actuation limit must be nonnegative, got {u_max}")
    gap = channel - radius
    return drift + gap * u_max if gap > 0.0 else drift


def robust_safe_action(
    grad_h: NDArray[np.float64],
    b_hat: NDArray[np.float64],
    radius: float,
    u_max: float,
) -> NDArray[np.float64]:
    """The action attaining :func:`robust_barrier_margin`: full authority, or nothing at all.

    Returns exactly ``0`` when ``radius >= ||B^T grad h||``. That is not a numerical artefact and
    not conservatism -- it is the optimum: with the channel's sign unidentified, every nonzero
    action has a worst case at least as bad as standing still.
    """
    _numbers(grad_h=np.asarray(grad_h), b_hat=np.asarray(b_hat), radius=radius, u_max=u_max)
    w = np.atleast_2d(b_hat).T @ np.asarray(grad_h, dtype=np.float64)
    norm_w = float(np.linalg.norm(w))
    if norm_w <= radius or norm_w == 0.0:
        return np.zeros_like(w)
    return u_max * w / norm_w


def identification_radius_threshold(
    drift: float, channel: float, u_max: float, alpha_h: float
) -> float:
    """Largest radius ``d*`` whose safety certificate still holds: ``channel - D / u_max``.

    ``D = -alpha_h - drift`` is the deficit the drift leaves for the controller to cover, and the
    answer is piecewise in it -- the formula alone is a trap, because it happily returns a
    *negative* "largest radius" for a problem no radius can save:

    * ``D <= 0`` -- the uncontrolled system already satisfies the barrier. No radius can break it,
      so the threshold is vacuous and ``inf`` says so rather than inviting a comparison.
    * ``D > u_max * channel`` -- nominally infeasible: even a perfectly identified channel at full
      authority cannot cover the deficit, so nothing is certified, not even at ``radius = 0``.
      ``nan`` (the module's empty-set convention, as in :func:`admissible_action_interval`).
    * ``0 < D <= u_max * channel`` -- the case worth having a threshold for: ``d*`` lands in
      ``[0, channel)`` and certification holds **iff** ``radius <= d*``.
    """
    # a nan passed `u_max <= 0` and returned nan, which here says "infeasible", not "no number"
    if not u_max > 0.0:
        raise ValueError(f"actuation limit must be positive to have a threshold, got {u_max}")
    _numbers(drift=drift, channel=channel, alpha_h=alpha_h)
    deficit = -alpha_h - drift
    if deficit <= 0.0:
        return float("inf")
    if deficit > u_max * channel:
        return float("nan")
    return channel - deficit / u_max


def barrier_gamma_star(threshold_radius: float, cvar_gap: float, grad_norm: float) -> float:
    """Largest MSM ``Gamma`` a safety certificate survives: ``(gap + c) / (gap - c)``.

    Inverts ``Delta(Gamma) = (Gamma-1)/(Gamma+1) * cvar_gap`` at ``threshold_radius``, with
    ``c = threshold_radius / grad_norm`` the effect-space radius the barrier can absorb. The closed
    form is only valid on ``0 < c < cvar_gap``; outside it the four cases are genuinely different
    answers rather than edge-case noise, and the denominator would silently change sign:

    * ``c < 0`` (or ``nan``) -- nothing is certified, not even at ``Gamma = 1`` where the radius is
      zero. ``nan``, because no ``Gamma >= 1`` works and reporting ``1.0`` would claim the opposite.
    * ``c == 0`` -- exact identification and nothing more: ``1.0``.
    * ``0 < c < cvar_gap`` -- the finite ceiling ``(gap + c) / (gap - c)``.
    * ``c >= cvar_gap`` -- ``Delta`` saturates at ``cvar_gap`` as ``Gamma -> inf``, so the threshold
      is beyond anything the sensitivity model can produce: ``inf``.

    ``grad_norm == 0`` is a state where the barrier is flat, such as the centre of a ball, and there
    the radius ``Delta * grad_norm`` is zero at every ``Gamma``: the step holds at every level or at
    none, ``inf`` when ``threshold_radius >= 0`` and ``nan`` otherwise.
    """
    # a nan gap passed `cvar_gap <= 0` and read no Gamma at all, and an infinite one inf/inf; a nan
    # gradient norm read nan, and an infinite one a radius of 0, exact identification
    if not (0.0 < cvar_gap < math.inf and 0.0 <= grad_norm < math.inf):
        raise ValueError(
            "cvar_gap must be positive and finite and grad_norm nonnegative and finite to invert "
            f"the radius, got cvar_gap={cvar_gap}, grad_norm={grad_norm}"
        )
    if grad_norm == 0.0:
        return float("inf") if threshold_radius >= 0.0 else float("nan")
    c = threshold_radius / grad_norm
    if np.isnan(c) or c < 0.0:
        return float("nan")
    if c == 0.0:
        return 1.0
    if c >= cvar_gap:
        return float("inf")
    return (cvar_gap + c) / (cvar_gap - c)


def admissible_action_interval(
    channel: float, radius: float, u_max: float, drift: float, alpha_h: float
) -> tuple[float, float]:
    """The scalar actions whose *guaranteed* barrier derivative clears ``-alpha_h``.

    ``channel`` is the **signed** ``grad h . B``, unlike the norm taken by
    :func:`robust_barrier_margin` -- which way to push is the whole question here. The guaranteed
    derivative ``drift + channel*u - radius*|u|`` is concave in ``u``, so the admissible set is an
    interval; combined with ``|u| <= u_max`` it stays one. Returns an empty interval as
    ``(nan, nan)`` when no admissible action exists, which the caller must handle rather than
    silently clip into.

    A non-positive deficit means ``u = 0`` is admissible -- it does **not** mean every action is.
    The control that serves the task is exactly the one that can spend the drift's slack and cross
    the boundary, so both ends are computed from the branch slopes in every case.
    """
    _numbers(channel=channel, radius=radius, u_max=u_max, drift=drift, alpha_h=alpha_h)
    if u_max < 0.0:
        raise ValueError(f"actuation limit must be nonnegative, got {u_max}")
    deficit = -alpha_h - drift
    right, left = channel - radius, channel + radius  # slopes of the u >= 0 and u <= 0 branches
    if deficit > 0.0:  # u = 0 is inadmissible: exactly one branch can carry the deficit
        if abs(channel) <= radius:
            return float("nan"), float("nan")  # sign unidentified: nothing raises the guarantee
        lo, hi = (deficit / right, u_max) if channel > 0.0 else (-u_max, deficit / left)
    else:  # u = 0 is admissible; the slack bounds how far the task may push each way
        hi = u_max if right >= 0.0 else min(u_max, deficit / right)
        lo = -u_max if left <= 0.0 else max(-u_max, deficit / left)
    return (lo, hi) if lo <= hi else (float("nan"), float("nan"))


def robust_safety_filter(
    u_nominal: float, channel: float, radius: float, u_max: float, drift: float, alpha_h: float
) -> float:
    """Least-restrictive admissible action: the nominal, clipped into the certified interval.

    Minimal intervention in the exact sense -- among all actions whose guaranteed margin clears, the
    one closest to what the task controller asked for. Scalar control only: for ``m > 1`` inputs the
    same set is convex but its projection is no longer a clip, and a filter that quietly over-brakes
    would misreport the price of safety. Use :func:`robust_safe_action` there, which maximises the
    margin rather than tracking a nominal.

    When the interval is empty -- the deficit exceeds what the identified channel can deliver -- the
    margin-maximising action ``sign(channel)*u_max`` is returned: it does not certify, and the
    caller should already know that from :func:`identification_radius_threshold`.
    """
    _numbers(u_nominal=u_nominal)
    lo, hi = admissible_action_interval(channel, radius, u_max, drift, alpha_h)
    if np.isnan(lo):
        return 0.0 if abs(channel) <= radius else float(np.sign(channel) * u_max)
    return float(np.clip(u_nominal, lo, hi))


@dataclass(frozen=True)
class BarrierConfoundingCurve:
    """Measured evidence for the first-order safety loss over a sensitivity grid on one problem."""

    gammas: tuple[float, ...]
    radii: tuple[float, ...]  # Delta(Gamma) * ||grad h||
    margins: tuple[float, ...]  # best guaranteed barrier derivative at each Gamma
    certified: tuple[bool, ...]  # margin >= -alpha*h
    actions: tuple[float, ...]  # norm of the robust-optimal action
    gamma_star: float  # the threshold, from the closed form
    last_certified_gamma: float  # the largest grid Gamma that actually certifies
    # log-log slope of the margin robust_barrier_margin loses in the effect error (theory: 1)
    safety_slope: float
    # log-log slope of the regret L_reg * error^2: 2 by construction, the order the theory
    # states, not a measurement of it
    regret_slope: float
    ok: bool


def barrier_confounding_certificate(
    channel: float = 0.6,
    drift: float = -0.9,
    u_max: float = 2.0,
    alpha_h: float = 0.5,
    cvar_gap: float = 1.0,
    grad_norm: float = 1.0,
    l_reg: float = 3.0,
    gammas: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 5.0, 9.0),
) -> BarrierConfoundingCurve:
    """Sweep the sensitivity level and check the threshold, the zero-action rule and the orders.

    The defaults are chosen so the sweep crosses every regime rather than sitting inside one: the
    drift is in deficit (``D = 0.4 > 0``) so the controller is load-bearing, ``Gamma* = 7/3`` falls
    inside the grid so certification genuinely fails partway, and the radius overtakes the channel
    at ``Gamma = 4`` so the tail of the grid exercises the zero-action rule. A configuration with
    ``D <= 0`` would certify everything and prove nothing.
    """
    radii = tuple(
        confounding_robust_inflation(cvar_gap, 0.0, gamma) * grad_norm for gamma in gammas
    )
    margins = tuple(robust_barrier_margin(drift, channel, d, u_max) for d in radii)
    certified = tuple(m >= -alpha_h - 1e-12 for m in margins)
    actions = tuple(0.0 if d >= channel else u_max for d in radii)

    threshold = identification_radius_threshold(drift, channel, u_max, alpha_h)
    gamma_star = barrier_gamma_star(threshold, cvar_gap, grad_norm)
    certified_gammas = [g for g, c in zip(gammas, certified, strict=True) if c]
    last_certified = max(certified_gammas) if certified_gammas else float("nan")

    errors = np.array([0.4, 0.2, 0.1, 0.05, 0.025])
    # nominal margin minus robust margin, on the identified side, read off the margin itself: up
    # to 0.18.0 it was errors * u_max, the theory's answer, a slope of 1 whatever the margin did
    nominal = robust_barrier_margin(drift, channel, 0.0, u_max)
    safety_loss = np.array(
        [nominal - robust_barrier_margin(drift, channel, float(e), u_max) for e in errors]
    )
    regret = l_reg * errors**2
    safety_slope = float(np.polyfit(np.log(errors), np.log(safety_loss), 1)[0])
    regret_slope = float(np.polyfit(np.log(errors), np.log(regret), 1)[0])

    grid_agrees = all(
        c == (gamma <= gamma_star + 1e-9) for gamma, c in zip(gammas, certified, strict=True)
    )
    ok = (
        grid_agrees
        and abs(safety_slope - 1.0) < 1e-6
        and abs(regret_slope - 2.0) < 1e-6
        and all((act == 0.0) == (d >= channel) for act, d in zip(actions, radii, strict=True))
    )
    return BarrierConfoundingCurve(
        gammas=tuple(gammas),
        radii=radii,
        margins=margins,
        certified=certified,
        actions=actions,
        gamma_star=gamma_star,
        last_certified_gamma=last_certified,
        safety_slope=safety_slope,
        regret_slope=regret_slope,
        ok=ok,
    )


@dataclass(frozen=True)
class SafetyFilterBenchmark:
    """Closed-loop evidence for the claim that a regret-sized budget is not a safety budget."""

    controllers: tuple[str, ...]
    violation_rate: tuple[float, ...]  # fraction of steps outside the safe set on the TRUE plant
    worst_violation: tuple[float, ...]  # deepest excursion past the limit (0.0 if never)
    tracking_cost: tuple[float, ...]  # mean squared distance to the reference
    gamma_star: float  # the certified sensitivity ceiling for this configuration
    ok: bool


def safety_filter_benchmark(
    b_true: float = 1.0,
    kappa: float = -0.5,
    confounding: float = 1.0,
    gamma: float = 3.0,
    decay: float = 0.3,
    x0: float = 0.0,
    x_ref: float = 3.0,
    x_limit: float = 2.0,
    alpha: float = 1.0,
    dt: float = 0.1,
    n_steps: int = 60,
    u_max: float = 6.0,
    n_data: int = 4000,
    seed: int = 0,
) -> SafetyFilterBenchmark:
    """Drive a confounded plant at a reference *beyond* a safety limit, four ways.

    The scalar plant ``x' = decay*x + b_true*u`` must track ``x_ref`` while respecting
    ``x <= x_limit``, and the two are in conflict by construction -- the reference sits past the
    limit, so nothing but the constraint stops the controller. The drift is *unstable* (``decay >
    0``), which is what makes the threshold bite: near the limit the plant pushes outward on its
    own, so the controller has a real deficit to cover and ``Gamma*`` is finite. The actuator gain
    is calibrated from an observational log whose action responded to a latent driver (``kappa < 0``
    attenuates it), so the controller believes a weaker actuator than it has and over-commands.

    The four rows are the point:

    * ``oracle`` filters with the true gain and a zero radius -- the best achievable safe tracking;
    * ``safety_calibrated`` filters with the biased gain and the MSM radius at ``gamma``;
    * ``regret_calibrated`` shrinks its action by the second-order regret accounting instead -- the
      radius spent on the *objective*, which is what the pessimism line does;
    * ``greedy`` believes the biased gain and does nothing about it.

    Returns violation rate, worst excursion and tracking cost for each. Read the tracking column
    with its floor in mind: a controller that respects the limit cannot beat
    ``(x_limit - x_ref)**2`` on this plant, so the violators' lower cost is *bought* with the
    excursions in the other two columns and is not evidence that they control better. This is a
    *demonstration on one synthetic plant*, not a theorem: it grounds the order argument rather
    than proving it.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_data)
    u_log = kappa * z + rng.standard_normal(n_data)
    x_log = rng.standard_normal(n_data)
    x_next = decay * x_log + b_true * u_log + confounding * z + 0.1 * rng.standard_normal(n_data)
    design = np.column_stack([x_log, u_log])
    b_hat = float(np.linalg.lstsq(design, x_next, rcond=None)[0][1])  # z omitted: confounded

    radius = confounding_robust_inflation(abs(b_hat), 0.0, gamma)
    # the tightest point is the boundary itself: h = 0, drift = -decay*x_limit pushing outward
    threshold = identification_radius_threshold(
        drift=-decay * x_limit, channel=abs(b_hat), u_max=u_max, alpha_h=0.0
    )
    gamma_star = barrier_gamma_star(threshold, abs(b_hat), 1.0)

    def nominal(x: float, gain: float) -> float:
        """One-step certainty-equivalent tracker: reach ``x_ref`` next step if the gain is right."""
        return float(np.clip(((x_ref - x) / dt - decay * x) / gain, -u_max, u_max))

    def run(mode: str) -> tuple[float, float, float]:
        gain = b_true if mode == "oracle" else b_hat
        filter_radius = 0.0 if mode == "oracle" else radius
        x, violations, worst, cost = x0, 0, 0.0, 0.0
        for _ in range(n_steps):
            u = nominal(x, gain)
            if mode in ("oracle", "safety_calibrated"):
                # h = x_limit - x, grad h = -1: drift = -decay*x, channel = -gain
                u = robust_safety_filter(
                    u,
                    channel=-gain,
                    radius=filter_radius,
                    u_max=u_max,
                    drift=-decay * x,
                    alpha_h=alpha * (x_limit - x),
                )
            elif mode == "regret_calibrated":
                u *= abs(b_hat) / (abs(b_hat) + radius)  # the regret's effect-error shrink
            x = x + dt * (decay * x + b_true * u)
            excursion = x - x_limit
            violations += excursion > 0.0
            worst = max(worst, excursion)
            cost += (x - x_ref) ** 2
        return violations / n_steps, worst, cost / n_steps

    modes = ("oracle", "safety_calibrated", "regret_calibrated", "greedy")
    rows = [run(mode) for mode in modes]
    rates, worsts, costs = (tuple(row[i] for row in rows) for i in range(3))
    filtered_are_safe = rates[0] == 0.0 and rates[1] == 0.0
    unfiltered_violate = rates[2] > 0.0 and rates[3] > 0.0
    return SafetyFilterBenchmark(
        controllers=modes,
        violation_rate=rates,
        worst_violation=worsts,
        tracking_cost=costs,
        gamma_star=gamma_star,
        ok=filtered_are_safe and unfiltered_violate,
    )


@dataclass(frozen=True)
class QuadraticBarrier:
    """``h(x) = upper - (x - centre)' shape (x - centre)``, safe where ``h >= 0``.

    ``shape`` is symmetric positive semidefinite, so ``h`` exceeds ``upper`` nowhere: the bound
    :func:`exit_probability` needs on the whole state space. Positive definite, the safe set is an
    ellipsoid; of lower rank, a slab or a cylinder, unbounded along ``shape``'s null space. An
    affine barrier has no such bound and is not one of these; a half-space is read through a slab
    inside it, which the loop leaves whenever it leaves the half-space.

    Raises:
        ValueError: on an ``upper`` that is not positive and finite, a ``centre`` that is not a
            vector, a ``shape`` that is not ``(n, n)``, a non-finite entry, or a ``shape`` that is
            not symmetric positive semidefinite.
    """

    upper: float
    centre: NDArray[np.float64]  # (n,)
    shape: NDArray[np.float64]  # (n, n)

    def __post_init__(self) -> None:
        upper = float(self.upper)
        if not 0.0 < upper < math.inf:
            raise ValueError(f"upper must be positive and finite, got {self.upper}")
        centre = np.array(self.centre, dtype=np.float64)
        if centre.ndim != 1 or centre.size == 0:
            raise ValueError(f"centre must be a vector, got shape {centre.shape}")
        n = centre.size
        shape = np.array(self.shape, dtype=np.float64)
        if shape.shape != (n, n):
            raise ValueError(f"shape must have shape ({n}, {n}), got {shape.shape}")
        if not (np.all(np.isfinite(centre)) and np.all(np.isfinite(shape))):
            raise ValueError("centre and shape must be finite")
        scale = max(1.0, float(np.abs(shape).max()))
        if not np.allclose(shape, shape.T, rtol=0.0, atol=1e-12 * scale):
            raise ValueError("shape is not symmetric")
        shape = 0.5 * (shape + shape.T)
        if float(np.linalg.eigvalsh(shape).min()) < -1e-12 * scale:
            raise ValueError("shape is not positive semidefinite, so h has no upper bound")
        centre.setflags(write=False)
        shape.setflags(write=False)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "centre", centre)
        object.__setattr__(self, "shape", shape)

    def __call__(self, x: ArrayLike) -> float:
        """``h`` at the state ``x``."""
        gap = np.asarray(x, dtype=np.float64) - self.centre
        return self.upper - float(gap @ self.shape @ gap)


@dataclass(frozen=True)
class ExitProbability:
    """A bound on the chance that a closed loop leaves ``{h >= -level}`` within ``steps`` steps.

    ``bound`` is at least ``P(min_{k <= steps} h(x_k) < -level)`` from the start, by Ville's
    inequality on Kushner's supermartingale, under ``E[h(x_{k+1}) | x_k] >= alpha h(x_k) + slack``
    and ``h <= upper``; it is clipped at 1, where the theorem says nothing. Read ``scope`` before
    the number. ``"everywhere"``: the drift condition holds at every state, exactly, on an affine
    loop. ``"states"``: it was read at the states given, and the bound holds only if no state the
    loop visits before it leaves has a smaller margin -- nothing checks that between them.
    """

    bound: float
    steps: int  # K: the bound covers k = 0 .. K
    alpha: float
    # the least E[h(x') | x] - alpha h(x) over the scope, capped at upper * (1 - alpha)
    slack: float
    level: float
    initial: float  # h(x0)
    upper: float  # the barrier's bound
    # "inward": slack >= -level (1 - alpha), so the mean step from the edge of {h >= -level} stays
    # in it, and the bound is Cosner et al.'s geometric form; "outward": below that, the summed one
    branch: Literal["inward", "outward"]
    scope: Literal["everywhere", "states"]


def _exit_bound(
    initial: float, steps: int, alpha: float, upper: float, slack: float, level: float
) -> tuple[float, Literal["inward", "outward"]]:
    """Cosner, Culbertson, Taylor and Ames's Theorem 5 (RSS 2023), clipped at 1."""
    spread = upper + level
    if slack >= -level * (1.0 - alpha):
        rate = (upper * alpha + level + slack) / spread
        return min(1.0, 1.0 - (initial + level) / spread * rate**steps), "inward"
    deficit = upper * (1.0 - alpha) - slack
    summed = (1.0 - alpha**steps) / (1.0 - alpha)
    value = (upper - initial) / spread * alpha**steps + deficit / spread * summed
    return min(1.0, value), "outward"


def _least_affine_margin(
    barrier: QuadraticBarrier,
    matrix: NDArray[np.float64],
    shift: NDArray[np.float64],
    noise_cost: float,
    alpha: float,
) -> float:
    """``inf_x E[h(x') | x] - alpha h(x)`` on the loop ``x' = matrix x + shift + w``.

    With ``y = x - centre`` and ``e = matrix centre + shift - centre`` the margin is
    ``r0 - 2 q'y + y'Q y``, with ``F = matrix`` and ``P`` the barrier's shape:
    ``Q = alpha P - F'P F``, ``q = F'P e`` and ``r0 = (1 - alpha) upper - noise_cost - e'P e``.
    Bounded below iff ``Q >= 0`` with ``q`` in its range, and then its least value is
    ``r0 - q'Q^+ q`` (``validation/exit_probability.mac``, STEP 6).

    Raises:
        ValueError: where the margin falls without bound, so the drift condition holds at no slack.
    """
    shape = barrier.shape
    pull = matrix @ barrier.centre + shift - barrier.centre
    contraction = matrix.T @ shape @ matrix
    q_matrix = alpha * shape - contraction
    q = matrix.T @ shape @ pull
    r0 = (1.0 - alpha) * barrier.upper - noise_cost - float(pull @ shape @ pull)
    eigenvalues, vectors = np.linalg.eigh(0.5 * (q_matrix + q_matrix.T))
    tolerance = 1e-10 * max(float(np.abs(alpha * shape).max()), float(np.abs(contraction).max()))
    along = vectors.T @ q
    flat = np.abs(eigenvalues) <= tolerance
    falls = (eigenvalues < -tolerance) | (flat & (np.abs(along) > 1e-9 * np.linalg.norm(q)))
    if np.any(falls):
        direction = vectors[:, int(np.argmax(falls))]
        raise ValueError(
            f"the drift condition E[h(x') | x] >= alpha h(x) + slack holds at no slack on this "
            f"loop: at alpha = {alpha}, E[h(x') | x] - alpha h(x) falls without bound along "
            f"{np.round(direction, 6).tolist()}, where alpha * shape - F' shape F has the "
            f"eigenvalue {float(eigenvalues[int(np.argmax(falls))]):.3g}. A larger alpha may "
            "hold; none below 1 holds on a loop that does not contract the barrier's shape"
        )
    curved = ~flat
    return r0 - float(np.sum(along[curved] ** 2 / eigenvalues[curved]))


def _refuse_open_loop(policy: object) -> None:
    """A plan of actions fixed in advance has no action that reads the state the noise moved."""
    if isinstance(policy, AffinePolicy):
        if not np.any(policy.gain):
            raise ValueError(
                "an AffinePolicy whose gain is zero is an open-loop plan: no action reads the "
                "state, and the exit probability reads a closed loop"
            )
        return
    if not callable(policy):
        raise TypeError(
            "policy must be a feedback law, an AffinePolicy or a callable of the state, got "
            f"{type(policy).__name__}: a plan of actions fixed in advance (an AffineSchedule, a "
            "CausalPlan) is open loop, and the exit probability reads a closed loop"
        )


def _checked_start(
    barrier: QuadraticBarrier,
    plant: LinearGaussianPlant,
    x0: ArrayLike,
    *,
    steps: int,
    alpha: float,
    level: float,
) -> NDArray[np.float64]:
    """``x0`` as a vector, once the theorem's numbers and the dimensions are checked."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError(f"steps must be an integer of at least 1, got {steps!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if not 0.0 <= level < math.inf:
        raise ValueError(f"level must be nonnegative and finite, got {level}")
    n = plant.states
    if barrier.centre.size != n:
        raise ValueError(f"the barrier is on {barrier.centre.size} states, the plant on {n}")
    start = np.asarray(x0, dtype=np.float64)
    if start.shape != (n,) or not np.all(np.isfinite(start)):
        raise ValueError(f"x0 must be a finite vector of {n} states, got {start!r}")
    return start


def _affine_slack(
    barrier: QuadraticBarrier,
    plant: LinearGaussianPlant,
    policy: AffinePolicy,
    alpha: float,
    states: ArrayLike | None,
) -> float:
    """The margin's least value over the whole state space, on the loop the policy closes."""
    if policy.gain.shape != (plant.actions, plant.states):
        raise ValueError(
            f"the policy's gain must have shape ({plant.actions}, {plant.states}), got "
            f"{policy.gain.shape}"
        )
    if states is not None:
        raise ValueError(
            "states reads a callable policy; an AffinePolicy's condition is read everywhere"
        )
    noise = plant.noise + plant.b @ policy.covariance @ plant.b.T
    return _least_affine_margin(
        barrier,
        plant.a + plant.b @ policy.gain,
        plant.b @ policy.offset + plant.offset,
        float(np.sum(barrier.shape * noise)),
        alpha,
    )


def _callable_slack(
    barrier: QuadraticBarrier,
    plant: LinearGaussianPlant,
    policy: Callable[[NDArray[np.float64]], ArrayLike],
    start: NDArray[np.float64],
    alpha: float,
    states: ArrayLike | None,
) -> float:
    """The margin's least value over ``start`` and ``states``, the policy read at each."""
    n, m = plant.states, plant.actions
    if states is None:
        raise ValueError("a callable policy needs the states at which its condition is read")
    read = np.asarray(states, dtype=np.float64)
    if read.ndim != 2 or read.shape[0] == 0 or read.shape[1] != n:
        raise ValueError(f"states must have shape (N, {n}) with N >= 1, got {read.shape}")
    if not np.all(np.isfinite(read)):
        raise ValueError("states must be finite")
    noise_cost = float(np.sum(barrier.shape * plant.noise))
    margins = []
    for x in np.vstack([start, read]):
        u = np.asarray(policy(x), dtype=np.float64).reshape(-1)
        if u.size != m or not np.all(np.isfinite(u)):
            raise ValueError(f"the policy must return {m} finite actions, got {u!r} at {x}")
        mean = plant.a @ x + plant.b @ u + plant.offset
        margins.append(barrier(mean) - noise_cost - alpha * barrier(x))
    return min(margins)


def exit_probability(
    barrier: QuadraticBarrier,
    plant: LinearGaussianPlant,
    policy: AffinePolicy | Callable[[NDArray[np.float64]], ArrayLike],
    x0: ArrayLike,
    *,
    steps: int,
    alpha: float,
    level: float = 0.0,
    states: ArrayLike | None = None,
) -> ExitProbability:
    """How likely a closed loop is to leave a barrier's safe set within ``steps`` steps: a bound.

    The loop is ``plant`` under the state feedback ``policy``, ``x' = a x + b u(x) + offset + w``,
    the noise ``w`` independent of the past and of the state, with mean 0 and covariance
    ``plant.noise``, plus ``b Sigma b'`` for an affine policy's own covariance ``Sigma``. On a
    quadratic barrier the next state's mean barrier needs that covariance alone,
    ``E[h(x') | x] = h(m(x)) - tr(shape Cov(w))`` with ``m(x)`` the loop's mean step, so the drift
    condition ``E[h(x') | x] >= alpha h(x) + slack`` is read exactly, without sampling the noise:

    * an :class:`~chc.evaluation.AffinePolicy` closes an affine loop, ``x' = F x + f + w``, on
      which the margin ``E[h(x') | x] - alpha h(x)`` is a quadratic in ``x``. Its least value over
      the whole state space is ``slack`` (``scope="everywhere"``), the premise as Cosner et al.
      state it. Where the quadratic falls without bound -- ``alpha shape - F' shape F`` has a
      negative direction, or a flat one along which the margin still slopes -- no slack holds
      and the call refuses.
    * a callable ``u = policy(x)``, a safety filter around a nominal controller say, is read at
      ``x0`` and at each row of ``states`` (``scope="states"``), and ``slack`` is the least margin
      there. The bound then holds if no state the loop visits before it leaves
      ``{h >= -level}`` has a smaller margin, which nothing checks between the states given.

    ``slack`` is capped at ``upper (1 - alpha)``, as the theorem asks. With
    ``h0 = h(x0)`` and ``phi = upper (1 - alpha) - slack``, the bound is Cosner, Culbertson,
    Taylor and Ames's (RSS 2023, Thm 5, a form of Kushner's): where
    ``slack >= -level (1 - alpha)`` (``branch="inward"``),

        1 - (h0 + level)/(upper + level) * ((upper alpha + level + slack)/(upper + level))^steps,

    and below it (``"outward"``)

        (upper - h0)/(upper + level) alpha^steps
            + phi/(upper + level) (1 - alpha^steps)/(1 - alpha),

    each clipped at 1. At ``slack = level = 0`` it is ``1 - alpha^steps h0 / upper``. Maxima
    ``validation/exit_probability.mac`` derives both forms from the supermartingale and sets the
    bound against the exact exit probability of a scalar Gaussian loop, where it is several times
    the truth: 0.2025 against 0.0263 at one step, 0.896 against 0.420 at ten.

    Scope. A closed loop only: the condition is a property of the feedback at the states the noise
    moves the loop to, and an open-loop plan -- an :class:`~chc.evaluation.AffineSchedule`, a
    :class:`chc.plan.CausalPlan`, an affine policy whose gain is zero -- is refused. The bound is
    a reading of the loop under its noise, not a certificate of the deterministic barrier:
    :func:`chc.plan.certify_safety` neither reads nor returns it. It is free of the dimension, and
    it reads the noise through its covariance alone; a noise whose law depends on the state is
    outside it, and so is one that persists from step to step, an AR(1) say, until it is carried
    as a state.

    Args:
        barrier: ``h`` and its bound ``upper``; safe where ``h >= 0``.
        plant: the step ``x' = a x + b u + offset + w``.
        policy: the state feedback that closes the loop.
        x0: the known start.
        steps: ``K``, at least 1.
        alpha: the discrete-time factor in ``(0, 1)`` of ``E[h(x') | x] >= alpha h(x) + slack``,
            not :func:`chc.plan.certify_safety`'s continuous-time gain, which is a rate.
        level: ``gamma >= 0``: the bound is on leaving ``{h >= -level}``.
        states: ``(N, n)``, the states besides ``x0`` at which a callable policy's condition is
            read. Refused with an affine policy, whose condition is read everywhere.

    Raises:
        TypeError: on a ``barrier`` that is not a :class:`QuadraticBarrier`, a ``plant`` that is
            not a :class:`~chc.evaluation.LinearGaussianPlant`, or a ``policy`` that is not a
            feedback law.
        ValueError: on an open-loop policy, dimensions that do not agree, a non-finite entry,
            ``steps`` below 1, an ``alpha`` outside ``(0, 1)``, a negative ``level``, ``states``
            with an affine policy or none with a callable one, an action of the wrong size, and a
            drift condition that holds at no slack.
    """
    if not isinstance(barrier, QuadraticBarrier):
        raise TypeError(
            f"barrier must be a QuadraticBarrier, got {type(barrier).__name__}: the bound needs h "
            "bounded above everywhere, and its mean under the noise"
        )
    if not isinstance(plant, LinearGaussianPlant):
        raise TypeError(f"plant must be a LinearGaussianPlant, got {type(plant).__name__}")
    _refuse_open_loop(policy)
    start = _checked_start(barrier, plant, x0, steps=steps, alpha=alpha, level=level)
    scope: Literal["everywhere", "states"]
    if isinstance(policy, AffinePolicy):
        slack, scope = _affine_slack(barrier, plant, policy, alpha, states), "everywhere"
    else:
        slack, scope = _callable_slack(barrier, plant, policy, start, alpha, states), "states"
    upper = barrier.upper
    slack = min(slack, upper * (1.0 - alpha))
    initial = barrier(start)
    bound, branch = _exit_bound(initial, steps, alpha, upper, slack, level)
    return ExitProbability(
        bound=bound,
        steps=steps,
        alpha=alpha,
        slack=slack,
        level=level,
        initial=initial,
        upper=upper,
        branch=branch,
        scope=scope,
    )
