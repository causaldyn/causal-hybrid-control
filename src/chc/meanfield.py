"""Sequential mean-field control: steer a distribution of agents over a zone graph toward a target.

The "recommend a zone, agents migrate, new distribution" loop: agents adjust *gradually* toward the
softmax response to the per-tick incentives, so an action plays out over several ticks -- a planner
that anticipates the migration lag beats a myopic controller that only reacts to the mismatch.
Builds on the game response in ``chc.games``. See ``plans/16`` (Phase 3).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from jax import Array

from chc.games import project_simplex


@dataclass(frozen=True)
class MeanFieldControl:
    """Agents migrate gradually toward incentivised zones; the platform steers them to a target."""

    n_zones: int = 6
    mass: float = 6.0
    budget: float = 3.0
    horizon: int = 20
    adjust: float = 0.4  # migration speed (<1 -> a lag that rewards planning)
    beta: float = 2.5
    congestion: float = 1.0
    control_weight: float = 0.02
    seed: int = 0

    def _target_attract(self) -> tuple[Array, Array]:
        """Demand ``target`` (where agents should be) and natural ``attract`` (where they drift)."""
        k = jax.random.split(jax.random.key(self.seed), 2)
        target = self.mass * jax.nn.softmax(jax.random.normal(k[0], (self.n_zones,)))
        attract = jax.random.normal(k[1], (self.n_zones,))  # natural pull, unrelated to demand
        return target, attract

    def _response(self, rho: Array, u: Array, attract: Array) -> Array:
        """One migration step: agents move a fraction ``adjust`` toward the softmax pull."""
        pull = self.mass * jax.nn.softmax(
            self.beta * (attract + u - self.congestion * rho / self.mass)
        )
        return (1.0 - self.adjust) * rho + self.adjust * pull

    def rollout_cost(self, u_seq: Array) -> Array:
        """Cumulative cost of a plan ``u_seq``: distance-to-target + control effort."""
        target, attract = self._target_attract()

        def step(rho: Array, u: Array) -> tuple[Array, Array]:
            rho_next = self._response(rho, u, attract)
            cost = jnp.sum((rho_next - target) ** 2) + self.control_weight * jnp.sum(u**2)
            return rho_next, cost

        rho0 = jnp.full(self.n_zones, self.mass / self.n_zones)
        _, costs = jax.lax.scan(step, rho0, u_seq)
        return jnp.sum(costs)

    def _myopic_cost(self) -> float:
        """Closed-loop myopic policy: each tick spend the budget on the current demand deficit."""
        target, attract = self._target_attract()

        def step(rho: Array, _: int) -> tuple[Array, Array]:
            deficit = jnp.maximum(target - rho, 0.0)
            u = self.budget * deficit / (jnp.sum(deficit) + 1e-8)
            rho_next = self._response(rho, u, attract)
            cost = jnp.sum((rho_next - target) ** 2) + self.control_weight * jnp.sum(u**2)
            return rho_next, cost

        rho0 = jnp.full(self.n_zones, self.mass / self.n_zones)
        _, costs = jax.lax.scan(step, rho0, jnp.arange(self.horizon))
        return float(jnp.sum(costs))

    def plan(self, steps: int = 400, lr: float = 0.05) -> Array:
        """Projected-gradient MPC: optimise the whole ``u_seq`` through the migration rollout."""
        u = jnp.full((self.horizon, self.n_zones), self.budget / self.n_zones)
        grad_fn = jax.jit(jax.grad(self.rollout_cost))
        cost_fn = jax.jit(self.rollout_cost)
        project = jax.vmap(lambda row: project_simplex(row, self.budget))
        optimizer = optax.adam(lr)
        state = optimizer.init(u)
        best_u, best_cost = u, float(cost_fn(u))
        for _ in range(steps):
            updates, state = optimizer.update(grad_fn(u), state)
            u = project(optax.apply_updates(u, updates))
            cost = float(cost_fn(u))
            if cost < best_cost:
                best_u, best_cost = u, cost
        return best_u

    def regrets(self, steps: int = 400) -> dict[str, float]:
        """Regret vs a well-planned oracle for no-control / myopic / planned mean-field control."""
        no_control = float(self.rollout_cost(jnp.zeros((self.horizon, self.n_zones))))
        planned = float(self.rollout_cost(self.plan(steps=steps)))
        oracle = float(self.rollout_cost(self.plan(steps=steps * 3)))
        return {
            "no-control": no_control - oracle,
            "myopic": self._myopic_cost() - oracle,
            "planned-CHC": planned - oracle,
        }
