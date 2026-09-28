"""Dynamics: the known mechanism, the learned residual, and their additive hybrid."""

from __future__ import annotations

from typing import Protocol

import equinox as eqx
import jax.numpy as jnp
from jax import Array


class Dynamics(Protocol):
    """A vector field f(t, x, u) -> dx/dt."""

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array: ...


class DampedOscillator(eqx.Module):
    """Known 2-state linear system: a driven damped harmonic oscillator.

    State ``x = [position, velocity]``, control ``u = [force]``:
        ``ẍ + 2ζω ẋ + ω² x = u``.

    ``omega`` and ``zeta`` are kept as (dynamic) leaves so they can later be identified as
    physical parameters ``p`` rather than hard-coded.
    """

    omega: float
    zeta: float

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        pos, vel = x[0], x[1]
        acc = -(self.omega**2) * pos - 2.0 * self.zeta * self.omega * vel + u[0]
        return jnp.stack([vel, acc])


class LinearDynamics(eqx.Module):
    """Known linear system ``ẋ = A x + B u``; the exactly-integrable part used in splitting."""

    a_matrix: Array  # (n, n)
    b_matrix: Array  # (n, m)

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return self.a_matrix @ x + self.b_matrix @ u


class HybridDynamics(eqx.Module):
    """``f(x, u, t) = f_known(x, u, t) + r_θ(x, u, t)``.

    The residual carries only the unknown part. With a :class:`~chc.residual.ZeroResidual` this
    reduces exactly to the known dynamics.

    Fitting the residual by prediction error (:mod:`chc.train`) makes its control channel the
    *observational* response, which is the wrong object under a confounded logging policy;
    :func:`chc.dynamics_id.fit_causal_residual` is the identified alternative, restricted to a
    control-affine residual.
    """

    known: Dynamics
    residual: Dynamics

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return self.known(t, x, u) + self.residual(t, x, u)


class DrivenDynamics(eqx.Module):
    """``f(t, x, u) + G w(t)``: a plant pushed by exogenous drivers whose path the plan can see.

    ``levels[k]`` is the drivers' value at ``t = start + k * dt``, and between two points they move
    linearly -- a first-order hold. A zero-order hold would not do: RK4's last stage of step ``k``
    is the first of step ``k + 1``, both at ``t = (k + 1) * dt``, so no function of ``t`` alone can
    hold one level through a step. Past either end the nearest segment is extended, so a plan
    longer than the forecast extrapolates it; a ``horizon``-step plan wants ``horizon + 1`` levels.

    The term reads neither ``x`` nor ``u``. It moves the drift and never the control channel, so
    what is priced on the channel -- the barrier's ``gamma*``, the tube's model error -- is left
    alone, and the forecast enters the barrier's drift term at every step.
    """

    dynamics: Dynamics
    gain: Array  # (n, d): how each driver pushes each state's rate
    levels: Array  # (K, d), K >= 2: the drivers at start, start + dt, ...
    dt: float
    start: float = 0.0

    def __check_init__(self) -> None:
        if self.levels.ndim != 2 or self.levels.shape[0] < 2:
            raise ValueError(
                f"levels has shape {self.levels.shape}; it needs (K, d) with K >= 2 points, one "
                "per grid time, for a path to interpolate"
            )
        if self.gain.ndim != 2 or self.gain.shape[1] != self.levels.shape[1]:
            raise ValueError(
                f"gain has shape {self.gain.shape} and levels {self.levels.shape}; the gain needs "
                "one column per driver"
            )
        if not self.dt > 0.0:
            raise ValueError(f"dt={self.dt} is not a positive grid spacing")

    def drivers(self, t: float | Array) -> Array:
        """The drivers at time ``t``, linear between the two grid points around it."""
        position = (t - self.start) / self.dt
        segment = jnp.clip(jnp.floor(position), 0, self.levels.shape[0] - 2)
        index = segment.astype(jnp.int32)
        share = position - segment
        return self.levels[index] + share * (self.levels[index + 1] - self.levels[index])

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return self.dynamics(t, x, u) + self.gain @ self.drivers(t)
