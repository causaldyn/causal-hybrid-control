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

import equinox as eqx
import jax
import jax.numpy as jnp
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
