"""Sequential mean-field control: steer a distribution of agents over a zone graph toward a target.

The "recommend a zone, agents migrate, new distribution" loop: agents adjust *gradually* toward the
softmax response to the per-tick incentives, so an action plays out over several ticks -- a planner
that anticipates the migration lag beats a myopic controller that only reacts to the mismatch.
Builds on the game response in ``chc.games``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array
from scipy.optimize import minimize

from chc.games import project_simplex


@partial(jax.jit, static_argnums=0)
def _meanfield_cost(planner: MeanFieldControl, u_seq: Array) -> Array:
    """``planner.rollout_cost`` jitted at module level, so the cache is not rebuilt per call.

    The planner is a frozen dataclass of scalars, hence hashable, hence a legal static argument;
    binding it with ``jax.jit`` inside :meth:`MeanFieldControl.plan` gave every call an
    empty cache and recompiled the rollout on every plan.
    """
    return planner.rollout_cost(u_seq)


@partial(jax.jit, static_argnums=0)
def _meanfield_grad(planner: MeanFieldControl, u_seq: Array) -> Array:
    """``d/du_seq`` of :func:`_meanfield_cost`, jitted at module level for the same reason."""
    return jax.grad(lambda plan: planner.rollout_cost(plan))(u_seq)


@partial(jax.jit, static_argnums=(0, 3))
def _descended(planner: MeanFieldControl, u_seq: Array, step: float, steps: int) -> Array:
    """``steps`` projected-gradient steps of fixed size ``step`` on the budget's simplices."""
    project = jax.vmap(lambda row: project_simplex(row, planner.budget))

    def body(_: int, plan: Array) -> Array:
        return project(plan - step * jax.grad(planner.rollout_cost)(plan))

    return jax.lax.fori_loop(0, steps, body, u_seq)


_DESCENT_STEPS = (1.0, 0.1, 0.01)
_DESCENT_LENGTH = 2000


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

    def _myopic_plan(self) -> Array:
        """Myopic policy: each tick spend the budget on the deficit; returns its open-loop plan."""
        target, attract = self._target_attract()

        def step(rho: Array, _: Array) -> tuple[Array, Array]:
            deficit = jnp.maximum(target - rho, 0.0)
            u = self.budget * deficit / (jnp.sum(deficit) + 1e-8)
            return self._response(rho, u, attract), u

        rho0 = jnp.full(self.n_zones, self.mass / self.n_zones)
        _, u_seq = jax.lax.scan(step, rho0, jnp.arange(self.horizon))
        return u_seq

    def plan(self, steps: int = 400, lr: float = 0.05, init: Array | None = None) -> Array:
        """Projected-gradient MPC on ``u_seq`` through the rollout (warm-start from ``init``)."""
        u = (
            init
            if init is not None
            else jnp.full((self.horizon, self.n_zones), self.budget / self.n_zones)
        )
        grad_fn = partial(_meanfield_grad, self)
        cost_fn = partial(_meanfield_cost, self)
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

    def _polished(self, start: Array) -> Array:
        """``start`` refined by SLSQP over the plans that spend the budget at every tick."""
        shape = start.shape
        rows = np.kron(np.eye(shape[0]), np.ones(shape[1]))

        def cost(v: np.ndarray) -> float:
            return float(_meanfield_cost(self, jnp.asarray(v.reshape(shape))))

        def grad(v: np.ndarray) -> np.ndarray:
            return np.asarray(_meanfield_grad(self, jnp.asarray(v.reshape(shape))), dtype=float)

        result = minimize(
            cost,
            np.asarray(start, dtype=float).ravel(),
            jac=lambda v: grad(v).ravel(),
            method="SLSQP",
            bounds=[(0.0, None)] * rows.shape[1],
            constraints=[
                {"type": "eq", "fun": lambda v: rows @ v - self.budget, "jac": lambda v: rows}
            ],
            options={"maxiter": 500, "ftol": 1e-12},
        )
        project = jax.vmap(lambda row: project_simplex(row, self.budget))
        return project(jnp.asarray(result.x.reshape(shape)))

    def regrets(self, steps: int = 400) -> dict[str, float]:
        """Regret vs the best plan found for no-control / myopic / planned mean-field control.

        The planner warm-starts from the myopic open-loop plan, so it refines (never worsens) it.
        The oracle is the least cost among the myopic and planned plans, the planner run three
        times as long, SLSQP's refinements of the myopic and the long plan, and plain projected
        gradient from the myopic, planned and long plans at the fixed steps 1, 0.1 and 0.01,
        2000 steps each, with SLSQP's refinement of the best: the best of local searches, not a
        proved optimum, and never above a plan it is compared with. The cost is not convex, and
        each search ends in the basin its start and step lead to. The planner run longer alone
        stalls: at the defaults, in float64, it costs 67.94; SLSQP from the myopic plan reaches
        45.69, and projected gradient from it at the step 0.1 reaches 42.81, so against SLSQP
        alone every regret read 2.88 low.
        """
        myopic_seq = self._myopic_plan()
        planned_seq = self.plan(steps=steps, init=myopic_seq)
        long_seq = self.plan(steps=steps * 3, init=myopic_seq)
        no_control = float(self.rollout_cost(jnp.zeros((self.horizon, self.n_zones))))
        myopic = float(self.rollout_cost(myopic_seq))
        planned = float(self.rollout_cost(planned_seq))
        descents = [
            _descended(self, start, step, _DESCENT_LENGTH)
            for start in (myopic_seq, planned_seq, long_seq)
            for step in _DESCENT_STEPS
        ]
        descended = min(descents, key=lambda u: float(_meanfield_cost(self, u)))
        searched = (
            long_seq,
            self._polished(myopic_seq),
            self._polished(long_seq),
            descended,
            self._polished(descended),
        )
        oracle = min([myopic, planned, *(float(self.rollout_cost(u)) for u in searched)])
        return {
            "no-control": no_control - oracle,
            "myopic": myopic - oracle,
            "planned-CHC": planned - oracle,
        }
