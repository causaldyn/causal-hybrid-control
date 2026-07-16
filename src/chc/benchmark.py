"""Benchmark v0: confounded, constrained control tasks with oracle regret — the moat (``plans/06``).

Each task ships a confounded offline dataset, a true plant with a computable oracle controller, and
an evaluation reporting **regret vs oracle**, **constraint violations**, and **out-of-support action
rate**. The point is to measure *where* causal control beats predictive control — and to be honest
where it does not. v0 has the pricing task (confounded linear steering); more tasks slot into the
same ``TaskResult`` / ``leaderboard`` shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from chc.causal import ConfoundedLinearSystem, estimate_control_effect
from chc.flagship import closed_loop


@dataclass(frozen=True)
class TaskResult:
    """One controller's score on one task."""

    controller: str
    cost: float
    regret: float  # cost - oracle_cost (>= 0; the oracle knows the true effect)
    constraint_violations: float  # fraction of steps outside the safe state set
    ood_rate: float  # fraction of actions outside the logged action support


@dataclass(frozen=True)
class PricingTask:
    """Confounded linear steering: drive x to a target; effect of u is confounded in the logs."""

    x0: float = 0.0
    x_target: float = 2.0
    n_steps: int = 30
    u_lo: float = -10.0
    u_hi: float = 10.0
    x_safe: float = 6.0  # state constraint |x| <= x_safe
    control_weight: float = 0.01
    n_data: int = 20_000
    kappa: float = -1.5  # confounding strength; 0.0 = randomised logs (no confounding)

    def _closed_loop_cost(
        self, system: ConfoundedLinearSystem, b_hat: float, key: Array
    ) -> tuple[Array, Array, float]:
        xs, us = closed_loop(
            system,
            b_hat,
            jnp.asarray(self.x0),
            self.x_target,
            self.n_steps,
            self.u_lo,
            self.u_hi,
            key,
        )
        cost = float(jnp.sum((xs - self.x_target) ** 2) + self.control_weight * jnp.sum(us**2))
        return xs, us, cost

    def _score(
        self,
        system: ConfoundedLinearSystem,
        name: str,
        b_hat: float,
        oracle_cost: float,
        u_support: tuple[float, float],
        key: Array,
    ) -> TaskResult:
        xs, us, cost = self._closed_loop_cost(system, b_hat, key)
        lo, hi = u_support
        return TaskResult(
            controller=name,
            cost=cost,
            regret=cost - oracle_cost,
            constraint_violations=float(jnp.mean(jnp.abs(xs) > self.x_safe)),
            ood_rate=float(jnp.mean((us < lo) | (us > hi))),
        )

    def run(self, seed_data: int = 0, seed_eval: int = 1) -> list[TaskResult]:
        """Fit the effect (oracle / causal / predictive) from logs and score each controller."""
        system = ConfoundedLinearSystem(kappa=self.kappa)
        data = system.sample(self.n_data, jax.random.key(seed_data))
        u_support = (
            float(jnp.quantile(data["u"], 0.01)),
            float(jnp.quantile(data["u"], 0.99)),
        )
        key = jax.random.key(seed_eval)
        _, _, oracle_cost = self._closed_loop_cost(system, system.b_true, key)
        controllers = {
            "oracle": system.b_true,
            "causal-CHC": float(estimate_control_effect(data, adjust_for=("z",))),
            "predictive": float(estimate_control_effect(data, adjust_for=())),
        }
        return [
            self._score(system, name, b_hat, oracle_cost, u_support, key)
            for name, b_hat in controllers.items()
        ]


def leaderboard(results: list[TaskResult]) -> str:
    """Format task results as a table sorted by regret (best first)."""
    header = f"{'controller':<14}{'cost':>12}{'regret':>12}{'viol':>8}{'ood':>8}"
    rows = [
        f"{r.controller:<14}{r.cost:>12.2f}{r.regret:>12.2f}"
        f"{r.constraint_violations:>8.2f}{r.ood_rate:>8.2f}"
        for r in sorted(results, key=lambda r: r.regret)
    ]
    return "\n".join([header, *rows])
