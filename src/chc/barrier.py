"""Safety certificates when the control effect is only PARTIALLY IDENTIFIED (§40).

The pessimism line (:mod:`chc.support`, :mod:`chc.uncertainty`) spends an identification radius on
*performance*: a worse effect estimate buys a wider penalty and a costlier plan. This module spends
the same radius on *safety*, and the accounting is different in a way that matters operationally.

For a control-affine plant ``xdot = f(x) + B u`` and a safe set ``{h >= 0}``, the barrier condition
is ``grad h . (f + B u) >= -alpha * h``. Write ``a = grad h . f`` for the drift term, ``g`` for the
control channel and ``U`` for the actuation limit. If the effect matrix is known only up to a radius
``Delta`` -- here an *identification* radius from §32's bounded-density-ratio model at level
``Gamma``, not a disturbance bound -- the guaranteed barrier derivative is ``a + g*u - d*|u|`` with
``d = Delta * ||grad h||``, and:

* the best guaranteed margin is ``a + (|g| - d)_+ * U``, so the margin lost to an identification
  radius ``d`` is ``U * min(d, |g|)`` -- **first order in ``d`` while control authority lasts**,
  then flat: past ``d = |g|`` there is nothing left to lose;
* at ``d >= |g|`` the robust-optimal action is exactly **zero**: with the sign of the channel
  unidentified, acting cannot improve the guarantee (the safety analogue of §11's sign threshold);
* certification is sharp at ``d* = |g| - D/U``, where ``D = -alpha*h - a`` is the deficit the drift
  leaves for the controller, which inverts to a largest tolerable sensitivity ``Gamma*``. Three
  regimes, and only the third is the interesting one: ``D <= 0`` (the drift alone is safe, no radius
  can break it), ``D > U*|g|`` (nominally infeasible -- not certified even at ``d = 0``), and
  ``0 < D <= U*|g|`` (``d*`` in ``[0, |g|)``, the case the threshold formula describes).

The tightening itself is the standard robust-CBF one (Jankovic 2018; Kolathaya-Ames 2019). What is
new here is the source of the radius and the resulting **order dichotomy**: the same effect error
costs performance regret only at second order (§33, ``L_reg * e**2``): an interior optimum has a
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

NumPy, like the rest of the §32-§40 sensitivity line, so it is independent of the JAX x64 flag.
Maxima ``validation/barrier_feasibility.mac``, Rocq ``proofs/barrier_feasibility.v``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chc.uncertainty import confounding_robust_inflation

__all__ = [
    "BarrierConfoundingCurve",
    "SafetyFilterBenchmark",
    "admissible_action_interval",
    "barrier_confounding_certificate",
    "barrier_gamma_star",
    "control_channel",
    "identification_radius_threshold",
    "robust_barrier_margin",
    "robust_safe_action",
    "robust_safety_filter",
    "safety_filter_benchmark",
]


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
    if u_max < 0.0:
        raise ValueError(f"actuation limit must be nonnegative, got {u_max}")
    return drift + max(0.0, channel - radius) * u_max


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
    if u_max <= 0.0:
        raise ValueError(f"actuation limit must be positive to have a threshold, got {u_max}")
    deficit = -alpha_h - drift
    if deficit <= 0.0:
        return float("inf")
    if deficit > u_max * channel:
        return float("nan")
    return channel - deficit / u_max


def barrier_gamma_star(threshold_radius: float, cvar_gap: float, grad_norm: float) -> float:
    """Largest sensitivity ``Gamma`` a safety certificate survives: ``(gap + c) / (gap - c)``.

    Inverts §32's ``Delta(Gamma) = (Gamma-1)/(Gamma+1) * cvar_gap`` at ``threshold_radius``, with
    ``c = threshold_radius / grad_norm`` the effect-space radius the barrier can absorb. The closed
    form is only valid on ``0 < c < cvar_gap``; outside it the four cases are genuinely different
    answers rather than edge-case noise, and the denominator would silently change sign:

    * ``c < 0`` (or ``nan``) -- nothing is certified, not even at ``Gamma = 1`` where the radius is
      zero. ``nan``, because no ``Gamma >= 1`` works and reporting ``1.0`` would claim the opposite.
    * ``c == 0`` -- exact identification and nothing more: ``1.0``.
    * ``0 < c < cvar_gap`` -- the finite ceiling ``(gap + c) / (gap - c)``.
    * ``c >= cvar_gap`` -- ``Delta`` saturates at ``cvar_gap`` as ``Gamma -> inf``, so the threshold
      is beyond anything the sensitivity model can produce: ``inf``.
    """
    if cvar_gap <= 0.0 or grad_norm <= 0.0:
        raise ValueError("cvar_gap and grad_norm must be positive to invert the radius")
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
    lo, hi = admissible_action_interval(channel, radius, u_max, drift, alpha_h)
    if np.isnan(lo):
        return 0.0 if abs(channel) <= radius else float(np.sign(channel) * u_max)
    return float(np.clip(u_nominal, lo, hi))


@dataclass(frozen=True)
class BarrierConfoundingCurve:
    """Measured evidence for §40 over a sensitivity grid on one confounded safety problem."""

    gammas: tuple[float, ...]
    radii: tuple[float, ...]  # Delta(Gamma) * ||grad h||
    margins: tuple[float, ...]  # best guaranteed barrier derivative at each Gamma
    certified: tuple[bool, ...]  # margin >= -alpha*h
    actions: tuple[float, ...]  # norm of the robust-optimal action
    gamma_star: float  # the threshold, from the closed form
    last_certified_gamma: float  # the largest grid Gamma that actually certifies
    safety_slope: float  # log-log slope of margin loss in the effect error (theory: 1)
    regret_slope: float  # log-log slope of the §33 performance regret (theory: 2)
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
    safety_loss = errors * u_max  # nominal margin minus robust margin, on the identified side
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
    """Closed-loop evidence for the §40 claim that a regret-sized budget is not a safety budget."""

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
    * ``safety_calibrated`` filters with the biased gain and the §32 radius at ``gamma``;
    * ``regret_calibrated`` shrinks its action by the §33 accounting instead -- the radius spent on
      the *objective*, which is what this library did everywhere before §40;
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
                u *= abs(b_hat) / (abs(b_hat) + radius)  # shrink by the §33 effect-error accounting
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
