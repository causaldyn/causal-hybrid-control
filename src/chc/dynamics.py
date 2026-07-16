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


class HybridDynamics(eqx.Module):
    """``f(x, u, t) = f_known(x, u, t) + r_θ(x, u, t)``.

    The residual carries only the unknown part; restricting its inputs (a feature map) is where
    causal design will live. With a :class:`~chc.residual.ZeroResidual` this reduces exactly to the
    known dynamics.
    """

    known: Dynamics
    residual: Dynamics

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return self.known(t, x, u) + self.residual(t, x, u)
