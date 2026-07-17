"""Continuum mean-field control: the density transport PDE ``rho_t + (rho v)_x = s`` (plans/16).

The continuous-space sibling of :class:`chc.meanfield.MeanFieldControl` (which lives on a finite
zone graph). Agents are a density ``rho(x, t)`` on a periodic 1-D domain; a velocity field ``v``
advects them and a source ``s`` injects/removes mass. Two pieces, both honest and validated:

* a **conservative** finite-volume solver (upwind flux) that conserves total mass to machine
  precision -- the physical invariant of a continuity equation -- advanced by **Strang-Marchuk**
  operator splitting (advection & reaction), reusing :func:`chc.splitting.strang_marchuk_step`. This
  is where the Marchuk framing earns its keep on a genuine PDE; the split is 2nd-order in time;
* :class:`MeanFieldTransport`, which picks the velocity field steering ``rho`` from an initial to a
  target profile at least transport effort ``sum rho v^2`` -- a discretised **dynamic optimal
  transport / mean-field game**, differentiable straight through the conservative solver.

plans/16 deferred this continuum model (the discrete zone graph gets ~90% of the value at ~10% of
the cost); it is built here as the elegant limit, with the mass-conservation and splitting-order
gates that make it trustworthy rather than merely elegant.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from jax import Array

from chc.splitting import strang_marchuk_step


def _upwind_divergence(rho: Array, v: Array, dx: float) -> Array:
    """Semi-discrete conservative upwind flux divergence ``-d(rho v)/dx`` on a periodic grid.

    Fluxes at cell faces telescope under the periodic ``roll``, so ``sum(divergence) == 0`` exactly:
    the mass-conservation invariant of the continuity equation, independent of the time integrator.
    """
    v_face = 0.5 * (v + jnp.roll(v, -1))  # velocity at the face between cell i and i+1
    rho_face = jnp.where(v_face >= 0.0, rho, jnp.roll(rho, -1))  # upwind density
    flux = v_face * rho_face  # F_{i+1/2}
    return -(flux - jnp.roll(flux, 1)) / dx  # -(F_{i+1/2} - F_{i-1/2}) / dx


def _advect(rho: Array, v: Array, dx: float, dt: float) -> Array:
    """One conservative advection sub-step, RK2 (Heun) in time to keep the operator split 2nd order.

    Explicit Euler would cap the whole scheme at 1st order however cleverly it is split; each RK2
    stage is conservative, so total mass is still invariant. Stable under CFL ``max|v| dt <= dx``.
    """
    k1 = _upwind_divergence(rho, v, dx)
    k2 = _upwind_divergence(rho + dt * k1, v, dx)
    return rho + 0.5 * dt * (k1 + k2)


def _react(rho: Array, k: Array, s: Array, dt: float) -> Array:
    """One local reaction/source sub-step ``rho_t = -k rho + s`` solved exactly per cell."""
    decay = jnp.exp(-k * dt)
    return rho * decay + jnp.where(k > 0.0, s / jnp.where(k > 0.0, k, 1.0) * (1.0 - decay), s * dt)


def _advect_flow(v: Array, dx: float):
    """The advection flow ``(rho, dt) -> _advect(...)`` for operator splitting."""
    return lambda rho, dt: _advect(rho, v, dx, dt)


def _react_flow(k: Array, s: Array):
    """The reaction flow ``(rho, dt) -> _react(...)`` for operator splitting."""
    return lambda rho, dt: _react(rho, k, s, dt)


def transport_step(rho: Array, v: Array, k: Array, s: Array, dx: float, dt: float) -> Array:
    """One Strang-Marchuk step of ``rho_t + (rho v)_x = -k rho + s`` (reaction & advection)."""
    return strang_marchuk_step(_react_flow(k, s), _advect_flow(v, dx), rho, dt)


def solve_transport(
    rho0: Array, v: Array, k: Array, s: Array, dx: float, dt: float, steps: int
) -> tuple[Array, Array]:
    """Roll the Strang-Marchuk step ``steps`` times; return ``(final_density, trajectory)``."""

    def step(rho: Array, _: Array) -> tuple[Array, Array]:
        rho_next = transport_step(rho, v, k, s, dx, dt)
        return rho_next, rho_next

    return jax.lax.scan(step, rho0, jnp.arange(steps))


def _gaussian(x: Array, center: float, width: float) -> Array:
    return jnp.exp(-0.5 * ((x - center) / width) ** 2)


@dataclass(frozen=True)
class MeanFieldTransport:
    """Steer a density ``rho(x,t)`` to a target by choosing the velocity field (continuum MFC).

    Minimises terminal mismatch ``||rho_T - target||^2`` plus transport effort ``sum rho v^2`` over
    a time-varying velocity field, advecting mass with the conservative solver. The velocity is
    CFL-capped so the explicit scheme stays stable no matter what the optimiser proposes.
    """

    n_cells: int = 64
    length: float = 1.0
    horizon: int = 40
    dt: float = 0.02
    effort_weight: float = 0.01
    source: float = 0.3  # where mass starts (fraction of the domain)
    target_center: float = 0.7  # where it should end up
    width: float = 0.08
    seed: int = 0

    @property
    def dx(self) -> float:
        return self.length / self.n_cells

    @property
    def v_cfl(self) -> float:
        return 0.9 * self.dx / self.dt  # keep the explicit advection stable

    def _initial_target(self) -> tuple[Array, Array]:
        """Unit-mass initial and target bumps (same mass -> steerable by advection alone)."""
        x = (jnp.arange(self.n_cells) + 0.5) * self.dx
        rho0 = _gaussian(x, self.source * self.length, self.width)
        target = _gaussian(x, self.target_center * self.length, self.width)
        return rho0 / (jnp.sum(rho0) * self.dx), target / (jnp.sum(target) * self.dx)

    def rollout_cost(self, v_seq: Array) -> Array:
        """Terminal mismatch + transport effort for a velocity plan ``v_seq`` (horizon, n)."""
        rho0, target = self._initial_target()

        def step(rho: Array, v: Array) -> tuple[Array, Array]:
            v = jnp.clip(v, -self.v_cfl, self.v_cfl)
            rho_next = _advect(rho, v, self.dx, self.dt)
            effort = self.effort_weight * jnp.sum(rho * v**2) * self.dx
            return rho_next, effort

        rho_final, efforts = jax.lax.scan(step, rho0, v_seq)
        mismatch = jnp.sum((rho_final - target) ** 2) * self.dx
        return mismatch + jnp.sum(efforts)

    def plan(self, steps: int = 300, lr: float = 0.05) -> Array:
        """Adam on the velocity field through the conservative solver; keep the best-seen plan."""
        v = jnp.zeros((self.horizon, self.n_cells))
        grad_fn = jax.jit(jax.grad(self.rollout_cost))
        cost_fn = jax.jit(self.rollout_cost)
        optimizer = optax.adam(lr)
        state = optimizer.init(v)
        best_v, best_cost = v, float(cost_fn(v))
        for _ in range(steps):
            updates, state = optimizer.update(grad_fn(v), state)
            v = optax.apply_updates(v, updates)
            cost = float(cost_fn(v))
            if cost < best_cost:
                best_v, best_cost = v, cost
        return best_v

    def mismatches(self, steps: int = 300) -> dict[str, float]:
        """Terminal mismatch with no control vs the planned velocity field (lower is better)."""
        rho0, target = self._initial_target()
        zero = jnp.zeros(self.n_cells)
        no_control, _ = solve_transport(rho0, zero, zero, zero, self.dx, self.dt, self.horizon)
        rho_final = rho0
        for v in self.plan(steps=steps):
            rho_final = _advect(rho_final, jnp.clip(v, -self.v_cfl, self.v_cfl), self.dx, self.dt)
        return {
            "uncontrolled": float(jnp.sum((no_control - target) ** 2) * self.dx),
            "controlled-CHC": float(jnp.sum((rho_final - target) ** 2) * self.dx),
        }
