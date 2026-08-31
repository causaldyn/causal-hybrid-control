"""Delayed dynamics as an ordinary ``chc`` vector field: the linear-chain delay line.

A discrete delay ``x(t - tau)`` is not a finite-dimensional vector field -- the history is a
function, so a delay differential equation lives in an infinite-dimensional state space and cannot
be a :class:`~chc.dynamics.Dynamics`. The ``m``-stage **linear chain** is one:

    ``x' = f(t, x, b_m, u)``,  ``b_1' = (x - b_1) m/tau``,  ``b_i' = (b_{i-1} - b_i) m/tau``

Each stage is a first-order lag of time constant ``tau/m``; equivalently the buffer is a first-order
upwind discretisation of transport along the delay line. The point of paying that price is that the
result is an ordinary vector field on ``z = [x, b_1, ..., b_m]``, so the *entire* rest of the
library -- ``rollout``, ``control_gradient_adjoint``, ``projected_gradient_control``,
``pessimistic_control``, ``causal_plan``, ``certify_safety``, ``mpc_control`` -- runs on a delayed
plant with no change at all. A method-of-steps DDE solver would have needed every one of them
rebuilt against it, and ``diffrax`` (already a dependency) does not solve DDEs.

**What the approximation actually costs, derived rather than asserted** (``validation/
delay_chain.mac``; the chain's transfer function ``(1 + s tau/m)^-m`` is the Laplace transform of
its kernel, so the moments come straight off it):

* The kernel is Erlang ``(m, m/tau)``: mass 1 and mean exactly ``tau``, but variance ``tau^2/m``.
  The applied delay is *smeared*, with relative spread ``1/sqrt(m)`` -- 10% at ``m = 100``.
* On the scalar loop ``x' = -a x(t - tau)`` the chain's Hopf boundary is
  ``a tau = m tan(pi/2m) sec^m(pi/2m)``, exactly (characteristic residual 0 in Maxima), which tends
  to the true discrete-delay value ``pi/2`` from **above**, with relative excess ``pi^2/(8m)``.

The direction of that second error is the one that matters and is why the margin is not read off
this object: the chain is **optimistic** about stability. It is the right tool for *simulating* a
delayed plant, and the wrong tool for *certifying* one -- ``chc.delay`` supplies the plant, and the
delay margin is computed from the exact discrete-delay characteristic equation.

The two errors also converge at different speeds -- ``O(1/m)`` for the margin against
``O(1/sqrt m)`` for the kernel shape -- so a stability question needs far fewer stages than a
reproduce-the-waveform question. Use :func:`stages_for_spread` to size the second.

Stages are not free to add, and the binding constraint is not the one the eigenvalues suggest.
The buffer block is *defective* -- a single Jordan block with the one eigenvalue ``-m/tau`` repeated
``m`` times -- so its spectrum says almost nothing about what an explicit integrator does to it.
Read it as advection instead: the symbol of first-order upwind covers a disc of radius ``m/tau``
centred at ``-m/tau``, so it reaches ``-2m/tau``, twice as far out as the eigenvalue, and the
constraint is a **CFL number** ``m dt/tau <= 1.3926`` rather than the RK4 real-axis limit
``2.7853``. Using the eigenvalue would permit exactly twice as many stages as are safe. Measured:
at ``tau/dt = 50`` the buffer is bounded through ``m = 75`` and reaches ``2.4e1`` at ``m = 80`` and
``8.8e30`` at ``m = 100`` -- silent, then catastrophic, which is why :func:`max_stages` is derived
rather than tuned into.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.optimize import brentq
from scipy.special import lambertw

from chc.cost import QuadraticCost

# Largest CFL number for RK4 on first-order upwind: the biggest c with |R(c(e^{-i xi} - 1))| <= 1
# for every xi, R the RK4 stability polynomial. Computed by bisection over the symbol, not quoted.
# The binding wavenumber is xi = pi, where the symbol sits at -2c, so this is exactly half the RK4
# real-axis limit 2.7853 -- the factor of two that an eigenvalue argument silently loses.
_UPWIND_CFL = 1.3926467817

#: A delayed vector field: ``(t, x, x_delayed, u) -> dx/dt``.
DelayedField = Callable[[float | Array, Array, Array, Array], Array]


def max_stages(tau: float, dt: float) -> int:
    """Largest chain length RK4 can integrate at step ``dt`` -- a CFL cap, not a spectral one.

    ``m dt/tau <= 1.3926``, from the upwind symbol rather than from the buffer's (defective)
    eigenvalue; see the module docstring for why the two differ by a factor of two.

    Deliberately the conservative side of the observed cliff: this is the von Neumann bound for the
    *periodic* problem, and a finite chain with an inflow boundary tolerates a little more (onset
    measured near ``1.50`` at 4000 steps, against the ``1.3926`` returned here). Conservative is the
    right direction when the failure mode is a buffer that looks fine for hundreds of steps and then
    reaches ``1e30``.
    """
    return int(_UPWIND_CFL * tau / dt)


def stages_for_spread(relative_spread: float) -> int:
    """Chain length whose applied delay has standard deviation ``relative_spread * tau``.

    Inverts the Erlang variance ``tau^2/m``. Sizing by the *kernel* is the conservative choice: the
    stability boundary converges as ``1/m`` and so is already far more accurate at the same ``m``.
    """
    if not 0.0 < relative_spread < 1.0:
        raise ValueError(f"relative_spread must lie in (0, 1); got {relative_spread}")
    return int(jnp.ceil(1.0 / relative_spread**2))


class DelayedDynamics(eqx.Module):
    """``x' = core(t, x, x(t - tau), u)`` presented as a plain :class:`~chc.dynamics.Dynamics`.

    The augmented state is ``z = [x, b_1, ..., b_stages]`` of width ``state_dim * (stages + 1)``;
    ``b_stages`` is the delayed signal the core sees. Build the initial ``z`` with
    :func:`augment_state` (constant history, the usual DDE initial condition) and read results back
    with :func:`state_of` / :func:`state_trajectory`.

    Only the *state* is delayed. A delayed control is the same object with the control appended to
    the delayed signal, and is deliberately not special-cased: in an MPC loop the control history is
    already known to the caller, so delaying it is a shift of ``us``, not a change of plant.
    """

    core: DelayedField
    tau: float = eqx.field(static=True)
    stages: int = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)

    def __init__(self, core: DelayedField, tau: float, stages: int, state_dim: int) -> None:
        if tau <= 0.0:
            raise ValueError(
                f"tau must be positive; a plant with no delay is just `core`. Got {tau}"
            )
        if stages < 1:
            raise ValueError(f"stages must be at least 1; got {stages}")
        if state_dim < 1:
            raise ValueError(f"state_dim must be at least 1; got {state_dim}")
        self.core, self.tau, self.stages, self.state_dim = core, tau, stages, state_dim

    def __call__(self, t: float | Array, z: Array, u: Array) -> Array:
        buffer = z[self.state_dim :].reshape(self.stages, self.state_dim)
        x, delayed = z[: self.state_dim], buffer[-1]
        # Each stage relaxes towards the one upstream of it; the first relaxes towards x itself.
        upstream = jnp.concatenate([x[None, :], buffer[:-1]], axis=0)
        return jnp.concatenate(
            [self.core(t, x, delayed, u), ((upstream - buffer) * (self.stages / self.tau)).ravel()]
        )


def augment_state(x0: Array, stages: int) -> Array:
    """Initial augmented state for a **constant** history ``x(s) = x0`` for all ``s <= 0``.

    The constant history is the standard DDE initial condition and the only one a caller holding a
    single ``x0`` can honestly supply. Pass a non-constant history by building ``z`` directly:
    ``jnp.concatenate([x0, *history])`` with the oldest slot last.
    """
    return jnp.concatenate([x0] + [x0] * stages)


def state_of(z: Array, state_dim: int) -> Array:
    """The undelayed state ``x`` out of an augmented ``z``."""
    return z[:state_dim]


def delayed_of(z: Array, state_dim: int) -> Array:
    """The delayed signal ``x(t - tau)`` out of an augmented ``z`` -- the last buffer slot."""
    return z[-state_dim:]


def state_trajectory(zs: Array, state_dim: int) -> Array:
    """The ``x`` columns of an augmented rollout ``(H + 1, state_dim * (stages + 1))``."""
    return zs[:, :state_dim]


def lift_cost(cost: QuadraticCost, stages: int) -> QuadraticCost:
    """Embed a cost on ``x`` into one on the augmented ``z``, charging nothing for the buffer.

    Without this the "nothing downstream changes" claim would be hollow: every solver takes a
    :class:`~chc.cost.QuadraticCost` whose ``Q`` is sized to the state it is given. The buffer
    carries no cost because it is bookkeeping, not a physical coordinate -- penalising it would
    penalise the plant's own history.
    """
    width = cost.Q.shape[0] * (stages + 1)

    def embed(block: Array) -> Array:
        return (
            jnp.zeros((width, width), dtype=block.dtype)
            .at[: block.shape[0], : block.shape[1]]
            .set(block)
        )

    return QuadraticCost(
        Q=embed(cost.Q),
        R=cost.R,
        Qf=embed(cost.Qf),
        x_target=augment_state(cost.x_target, stages),
    )


def exact_delayed_rollout(
    core: DelayedField, x0: Array, us: Array, dt: float, lag: int, t0: float = 0.0
) -> Array:
    """Reference rollout with an **exact integer lag**, by carrying the history explicitly.

    This is the oracle the chain is priced against, not a shipping path: it is exact for a delay of
    ``lag * dt`` under explicit Euler, but it is a discrete map rather than a vector field, so
    nothing else in the library can consume it. Used by the tests to measure what
    :class:`DelayedDynamics` costs, and by callers who want the discrete-delay ground truth.

    The history before ``t0`` is constant at ``x0``, matching :func:`augment_state`.
    """
    if lag < 0:
        raise ValueError(f"lag must be non-negative; got {lag}")
    history = jnp.broadcast_to(x0, (lag + 1, x0.shape[0]))

    def body(carry: tuple[Array, Array], u: Array) -> tuple[tuple[Array, Array], Array]:
        t, past = carry
        x, delayed = past[-1], past[0]
        x_next = x + dt * core(t, x, delayed, u)
        return (t + dt, jnp.concatenate([past[1:], x_next[None, :]])), x_next

    _, xs = jax.lax.scan(body, (jnp.asarray(t0), history), us)
    return jnp.concatenate([x0[None, :], xs], axis=0)


@dataclass(frozen=True)
class DelayMarginCertificate:
    """Evidence that the delay margin is where the closed form says it is, on both sides."""

    pole: float
    gain: float
    critical_delay: float
    stable_delays: tuple[float, ...]  # multiples of the critical delay that stayed bounded
    unstable_delays: tuple[float, ...]  # multiples that grew
    largest_stable_ratio: float  # the biggest tau/tau_c that stayed bounded
    smallest_unstable_ratio: float  # the smallest that did not
    ok: bool


def delay_margin(pole: float, gain: float) -> float:
    """Largest measurement delay the loop ``x' = pole*x - gain*x(t - tau)`` tolerates.

    ``tau_c = arccos(pole/gain) / sqrt(gain^2 - pole^2)``, from the imaginary-axis crossing of
    ``lambda - pole + gain exp(-lambda tau)`` (``validation/delay_margin.mac``; both real and
    imaginary residuals are exactly ``0``). At ``pole = 0`` this is ``pi/(2 gain)``, the textbook
    ``K tau = pi/2``.

    Computed from the *exact* characteristic equation, deliberately not from the delay line of
    :class:`DelayedDynamics`: the chain's own boundary sits **above** this one by ``pi^2/(8m)``,
    so reading a margin off the simulator would call a loop safe past the point where it is not.

    ``gain <= pole`` raises: with ``pole > 0`` a gain that small does not stabilise the loop even
    at zero delay, so "the largest tolerable delay" names nothing. The formula also carries its own
    fundamental limit -- ``tau_c -> 1/pole`` as ``gain -> pole+``, and it *decreases* in ``gain``
    from there, so an unstable pole admits **no** gain past ``tau = 1/pole``.
    """
    if gain <= abs(pole):
        raise ValueError(
            f"gain must exceed |pole| for the delay-free loop to be stable; got gain={gain}, "
            f"pole={pole}"
        )
    return float(np.arccos(pole / gain) / np.sqrt(gain**2 - pole**2))


def delay_margin_certificate(
    pole: float = 0.0,
    gain: float = 1.0,
    dt: float = 0.002,
    ratios: tuple[float, ...] = (0.5, 0.8, 0.95, 1.05, 1.3, 2.0),
    horizon: float = 400.0,
) -> DelayMarginCertificate:
    """Sweep ``tau`` across :func:`delay_margin` and check the loop really changes behaviour there.

    Simulated with :func:`exact_delayed_rollout` rather than the chain, for two reasons: the lag is
    exact at ``tau/dt`` steps, and explicit Euler errs *conservative* -- its own boundary sits
    ``1/(2m)`` **below** the continuous one, against the chain's ``+pi^2/(8m)`` above. A
    conservative simulator that still shows instability past ``tau_c`` is evidence; an optimistic
    one showing stability just inside it would not be.

    That leaves a band of width ``~1/(2m)`` around ``tau_c`` where the discretisation, not the
    plant, decides the verdict, so ``ratios`` deliberately steps over it rather than into it.
    """
    critical = delay_margin(pole, gain)

    def core(t: float | Array, x: Array, x_delayed: Array, u: Array) -> Array:
        return pole * x - gain * x_delayed

    x0 = jnp.array([1.0])
    stable: list[float] = []
    unstable: list[float] = []
    for ratio in ratios:
        tau = ratio * critical
        lag = max(1, round(tau / dt))
        steps = int(horizon / dt)
        xs = exact_delayed_rollout(core, x0, jnp.zeros((steps, 1)), dt, lag)
        early = float(jnp.max(jnp.abs(xs[: steps // 4])))
        late = float(jnp.max(jnp.abs(xs[-steps // 4 :])))
        (unstable if late > early else stable).append(ratio)

    largest_stable = max(stable) if stable else 0.0
    smallest_unstable = min(unstable) if unstable else float("inf")
    return DelayMarginCertificate(
        pole=pole,
        gain=gain,
        critical_delay=critical,
        stable_delays=tuple(stable),
        unstable_delays=tuple(unstable),
        largest_stable_ratio=largest_stable,
        smallest_unstable_ratio=smallest_unstable,
        # The boundary must sit inside (largest stable, smallest unstable), and both sides must be
        # non-empty -- a sweep where nothing destabilised would prove nothing about the margin.
        ok=bool(stable) and bool(unstable) and largest_stable < 1.0 < smallest_unstable,
    )


# Design constant of the decay-rate-optimal scalar delayed feedback: K* = _OPTIMAL_GAIN_CONSTANT/tau
# maximises the spectral abscissa of x' = -K x(t - tau), giving sigma* = 1/tau. Classical; what is
# derived in validation/delay_ball.mac is everything that follows from the root there being DOUBLE.
_OPTIMAL_GAIN_CONSTANT = float(1.0 / np.e)

# The design stabilises the true plant iff K* tau < pi/2, i.e. iff tau_hat/tau exceeds this.
STABILISING_RATIO_FLOOR = float(2.0 / (np.pi * np.e))


@dataclass(frozen=True)
class DelayBall:
    """The set of delay estimates that still stabilise the true plant. It is a **half-line**.

    ``ratio_floor`` is the smallest ``tau_hat/tau`` that stabilises; there is no ceiling, so the
    field is named for what it is rather than paired with an absent ``ratio_ceiling``. In absolute
    terms the admissible estimates are ``(shortest_safe_estimate, inf)``.

    Contrast the ball in dynamics error that Result 44 gives, which is symmetric and pairs with a
    regret quadratic in its radius. Neither property survives the move to delay space, and both fail
    for the same reason: the decay-optimal design sits at a *defective* characteristic root.
    """

    tau: float
    gain: float
    ratio_floor: float
    shortest_safe_estimate: float
    relative_radius: float


def optimal_delay_gain(tau: float) -> float:
    """Gain maximising the decay rate of ``x' = -K x(t - tau)``: ``K* = 1/(e tau)``.

    The decay rate it achieves is ``sigma* = 1/tau``, and no gain does better.

    At this gain the characteristic root ``s = -1`` (in ``s = lambda tau``) is **double**. That is
    the source of every asymmetry in :func:`delay_ball` and :func:`delay_design_loss`: perturbing a
    parameter away from a defective root moves it like a square root, not linearly.
    """
    if tau <= 0.0:
        raise ValueError(f"tau must be positive; got {tau}")
    return _OPTIMAL_GAIN_CONSTANT / tau


def delay_ball(tau: float) -> DelayBall:
    """Which delay estimates keep the true ``tau``-plant stable under the decay-optimal design.

    Designing ``K_hat = 1/(e tau_hat)`` and running it against the true delay ``tau`` puts the loop
    gain at ``kappa = K_hat tau = 1/(e r)`` with ``r = tau_hat/tau``. The exact boundary is
    ``kappa = pi/2`` (``delay_margin`` at ``a = 0``), and ``kappa`` is antitone in ``r``, so the
    admissible set is ``r > 2/(pi e) = 0.2342``: **under**-estimating the delay by more than 76% of
    it destabilises, and **over**-estimating never does, at any magnitude.

    That asymmetry is not a modelling choice -- it follows from the boundary being an upper bound on
    loop gain while the design's gain is inversely proportional to the assumed delay. Its practical
    reading is uncomfortable: the safe direction for stability is the expensive one for performance
    (see :func:`delay_design_loss`), so "be conservative about the delay" is not free advice.

    No Lyapunov-Krasovskii functional is constructed, deliberately. An LKF would deliver a
    *sufficient* condition with an unquantified gap; the characteristic equation gives the exact
    boundary, so an LKF here would be strictly weaker evidence, not stronger.
    """
    if tau <= 0.0:
        raise ValueError(f"tau must be positive; got {tau}")
    return DelayBall(
        tau=tau,
        gain=optimal_delay_gain(tau),
        ratio_floor=STABILISING_RATIO_FLOOR,
        shortest_safe_estimate=STABILISING_RATIO_FLOOR * tau,
        relative_radius=1.0 - STABILISING_RATIO_FLOOR,
    )


def delay_design_loss(ratio: float) -> float:
    """Decay rate given up by designing for ``tau_hat`` when the true delay is ``tau``.

    ``ratio`` is ``tau_hat/tau``; the return is ``tau * (sigma* - sigma)``, dimensionless, zero at
    ``ratio = 1`` and ``1.0`` at the stability boundary. Computed from the exact rightmost
    characteristic root, not from the expansion -- the expansion is what explains the shape, the
    exact root is what a caller should be given.

    **The two sides are not the same order.** Writing ``eps = (ratio - 1)/ratio``, the root's real
    part near the optimum is ``sqrt(2 eps) - 2 eps/3`` for an over-estimate and ``-2 eps/3`` for an
    under-estimate (``validation/delay_ball.mac``, matched to the exact root to ``1e-3`` at
    ``|eps| = 0.05``). Over-estimating opens as a **square root**, under-estimating only linearly,
    so at ``|eps| = 0.05`` -- the same absolute error either way -- an over-estimate costs ``0.287``
    against an under-estimate's ``0.033``, 8.8 times as much. No symmetric radius can describe both,
    which is why :class:`DelayBall` carries a floor and no ceiling and this is not a norm of
    ``eps``.

    So the two failure directions want opposite things: stability wants ``tau_hat`` large, the decay
    rate wants it small. :func:`robust_delay_design` resolves that against an interval.
    """
    if ratio <= 0.0:
        raise ValueError(f"ratio must be positive; got {ratio}")
    if ratio <= STABILISING_RATIO_FLOOR:
        return float("inf")  # the design does not stabilise the plant at all; no rate to give up
    kappa = _OPTIMAL_GAIN_CONSTANT / ratio
    if abs(kappa - _OPTIMAL_GAIN_CONSTANT) < 1e-14:
        return 0.0  # the double root itself, where both branches meet
    if kappa < _OPTIMAL_GAIN_CONSTANT:  # real branch: two real roots, the rightmost is a Lambert W
        return float(1.0 + np.real(lambertw(-kappa, 0)))
    # complex branch: s = p + i q with p = -q cot(q) from the imaginary part, and the real part then
    # collapsing to kappa = exp(p) q / sin(q). Both exact; derived in validation/delay_ball.mac.
    q = brentq(
        lambda y: np.exp(-y * np.cos(y) / np.sin(y)) * y / np.sin(y) - kappa,
        1e-12,
        np.pi / 2 - 1e-12,
    )
    return float(1.0 - q * np.cos(q) / np.sin(q))  # Re u = 1 + Re s, Re s = -q cot q


@dataclass(frozen=True)
class RobustDelayDesign:
    """The delay to design for, given only an interval containing the true one."""

    tau_design: float
    gain: float
    worst_case_loss: float
    stabilises_interval: bool


def robust_delay_design(lo: float, hi: float) -> RobustDelayDesign:
    """Minimax delay design over an interval: which ``tau_hat`` to build the controller for.

    Feed it the ends of a delay interval -- :attr:`chc.irf.DelayEstimate.lo` and ``.hi`` are exactly
    that -- and it returns the ``tau_hat`` minimising the worst-case :func:`delay_design_loss` over
    every true delay the interval allows. Because the loss is unimodal in ``tau_hat/tau``, the worst
    case is attained at an end of the interval, and the minimax equalises the two ends.

    **The answer is not the centre of the interval.** Over-estimating costs a square root and
    under-estimating only a linear term, so equalising the two ends pushes the design *down*: on
    ``[0.8, 1.25]`` the minimax lands at ``0.837`` against a geometric mean of ``1.0``, a 16% shift,
    and it halves the worst-case loss, ``0.528`` to ``0.270``. Designing for the middle of a delay
    interval is a symmetric answer to an asymmetric problem.

    The shift is downward only while the interval is informative. The ratio depends on ``hi/lo``
    alone (the problem is scale-free in ``tau``), it deepens to ``0.754`` of the geometric mean near
    ``hi/lo = 3.2``, then relaxes and **crosses back above at hi/lo = 13.25**: past there the low
    end is close to the stabilising floor, its loss saturates near 1, and it dominates instead. A
    13x-wide delay interval is barely information, so the rule holds wherever it is worth applying
    -- but it is a regime, not a law, and the crossover is where it ends.

    The direction is worth naming because it opposes the safety instinct: a *longer* assumed delay
    buys stability margin (:func:`delay_ball`), a *shorter* one buys decay rate. Both are exact, and
    only the interval decides the trade.

    Do not pass a censored estimate. :attr:`chc.irf.DelayEstimate.censored` means the peak sat on
    the edge of the horizon, so ``hi`` bounds nothing -- the true delay may be far larger, and that
    is the direction :func:`delay_ball` says is unsafe to under-shoot.
    """
    if not 0.0 < lo <= hi:
        raise ValueError(f"need 0 < lo <= hi; got lo={lo}, hi={hi}")
    if lo == hi:
        return RobustDelayDesign(lo, optimal_delay_gain(lo), 0.0, True)
    # L(t/hi) - L(t/lo) is positive at t = lo (only the far end is mis-specified) and negative at
    # t = hi, so the equalising design is bracketed by the interval itself. The lower end is lifted
    # off the stabilising floor first, for an interval wider than the ball's dynamic range.
    left = max(lo, hi * STABILISING_RATIO_FLOOR * (1.0 + 1e-9))
    gap = brentq(lambda t: delay_design_loss(t / hi) - delay_design_loss(t / lo), left, hi)
    return RobustDelayDesign(
        tau_design=float(gap),
        gain=optimal_delay_gain(float(gap)),
        worst_case_loss=float(delay_design_loss(gap / hi)),
        stabilises_interval=bool(gap / hi > STABILISING_RATIO_FLOOR),
    )


@dataclass(frozen=True)
class DelayBallCertificate:
    """Evidence that the delay ball's floor is where the derivation says, and that the loss is too.

    ``largest_unstable`` and ``smallest_stable`` are multiples of the derived floor, so they bracket
    where the loop *actually* changes behaviour: the derivation is confirmed when the bracket
    straddles 1. Result 44's ball came from a Lyapunov *sufficient* condition and ran 15.5x
    conservative; this one comes from the exact characteristic equation, so the bracket is tight to
    the sweep grid and there is no slack to report.
    """

    tau: float
    ratio_floor: float
    stable_ratios: tuple[float, ...]
    unstable_ratios: tuple[float, ...]
    largest_unstable: float
    smallest_stable: float
    worst_loss_error: float
    ok: bool


def delay_ball_certificate(
    tau: float = 1.0,
    dt: float = 0.001,
    ratios: tuple[float, ...] = (0.25, 0.6, 0.9, 1.1, 1.6, 3.0, 6.0),
    loss_ratios: tuple[float, ...] = (0.8, 0.9, 0.95, 1.05, 1.1, 1.25),
    horizon: float = 60.0,
) -> DelayBallCertificate:
    """Sweep the delay estimate across :func:`delay_ball`'s floor and check both halves of it.

    ``ratios`` are multiples of the floor ``2/(pi e)``, so they run from a quarter of it to six
    times; ``loss_ratios`` are values of ``tau_hat/tau`` near 1, where the derived
    :func:`delay_design_loss` is checked against the decay rate the simulation actually shows.

    Simulated with :func:`exact_delayed_rollout` for the same reason as
    :func:`delay_margin_certificate`: explicit Euler's own boundary sits *below* the continuous one,
    so instability it reports past the floor is evidence rather than discretisation.

    The certificate can fail in four independent ways: no ratio destabilises, no ratio stabilises,
    the floor does not separate them, or the measured loss departs from the closed form by more than
    the discretisation can account for.
    """
    if tau <= 0.0:
        raise ValueError(f"tau must be positive; got {tau}")
    floor = STABILISING_RATIO_FLOOR
    x0 = jnp.array([1.0])
    steps = int(horizon / dt)
    lag = max(1, round(tau / dt))

    def simulate(ratio: float) -> Array:
        gain = optimal_delay_gain(ratio * tau)  # designed for tau_hat, run against tau

        def core(t: float | Array, x: Array, x_delayed: Array, u: Array) -> Array:
            return -gain * x_delayed

        return exact_delayed_rollout(core, x0, jnp.zeros((steps, 1)), dt, lag)

    stable: list[float] = []
    unstable: list[float] = []
    for ratio in ratios:
        xs = simulate(ratio * floor)
        early = float(jnp.max(jnp.abs(xs[: steps // 4])))
        late = float(jnp.max(jnp.abs(xs[-steps // 4 :])))
        (unstable if late > early else stable).append(ratio)

    worst_loss_error = 0.0
    for ratio in loss_ratios:
        xs = np.asarray(simulate(ratio))[:, 0]
        measured = 1.0 - tau * _envelope_decay_rate(xs, dt)
        worst_loss_error = max(worst_loss_error, abs(measured - delay_design_loss(ratio)))

    largest_unstable = max(unstable) if unstable else 0.0
    smallest_stable = min(stable) if stable else float("inf")
    return DelayBallCertificate(
        tau=tau,
        ratio_floor=floor,
        stable_ratios=tuple(stable),
        unstable_ratios=tuple(unstable),
        largest_unstable=largest_unstable,
        smallest_stable=smallest_stable,
        worst_loss_error=worst_loss_error,
        ok=bool(
            stable
            and unstable
            and largest_unstable < 1.0 < smallest_stable
            # explicit Euler costs 0.93 * dt/tau of decay rate, measured flat over an 8x range of
            # dt; twice that leaves margin for the fit without admitting a genuine gap
            and worst_loss_error < 2.0 * dt / tau
        ),
    )


def _envelope_decay_rate(xs: np.ndarray, dt: float) -> float:
    """Spectral abscissa read off the tail of ``|x|``, through its peaks when the response rings."""
    envelope = np.abs(xs)
    half = xs.size // 2
    peaks = np.where((envelope[1:-1] > envelope[:-2]) & (envelope[1:-1] >= envelope[2:]))[0] + 1
    peaks = peaks[(peaks > half) & (envelope[peaks] > 0.0)]
    if peaks.size < 2:  # monotone decay: no ringing to track, so the tail itself is the envelope
        peaks = np.array([half, xs.size - 1])
    times = peaks * dt
    return float(-np.polyfit(times, np.log(envelope[peaks]), 1)[0])
