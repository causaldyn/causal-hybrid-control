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
