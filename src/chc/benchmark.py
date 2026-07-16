"""Benchmark v0: confounded, constrained control tasks with oracle regret — the moat (``plans/06``).

Each task ships a confounded offline dataset, a true plant with a computable oracle controller, and
an evaluation reporting **regret vs oracle**, **constraint violations**, and **out-of-support action
rate**. The point is to measure *where* causal control beats predictive control — and to be honest
where it does not. v0 has pricing (steering), inventory (newsvendor), and support-shift (pessimism)
tasks, all in the same ``TaskResult`` / ``leaderboard`` shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.stats
from jax import Array

from chc.causal import ConfoundedLinearSystem, estimate_control_effect
from chc.control import projected_gradient_control
from chc.cost import QuadraticCost, total_cost
from chc.dynamics import HybridDynamics, LinearDynamics
from chc.flagship import closed_loop
from chc.residual import ZeroResidual
from chc.support import SupportModel, pessimistic_control


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


@dataclass(frozen=True)
class InventoryTask:
    """Newsvendor ordering under a confounded demand-response model (holding / stockout costs).

    A fixed-intensity promo lifts demand; in the logs the promo was correlated with a demand driver
    ``z`` (a confounder), so the promo effect is biased. The retailer orders to a newsvendor level
    from its estimated demand model, so a wrong estimate systematically over- or under-orders.
    """

    d0: float = 5.0  # base demand
    promo: float = 1.0  # fixed promo intensity
    sigma_d: float = 1.0  # demand noise std
    holding: float = 0.5  # per-unit holding cost
    stockout: float = 2.0  # per-unit stockout cost (asymmetric: shortages hurt more)
    kappa: float = -1.0  # confounding strength (sign chosen so the naive fit under-orders)
    n_data: int = 20_000
    n_eval: int = 5000

    def _order(self, b_hat: float) -> float:
        critical_ratio = self.stockout / (self.stockout + self.holding)
        z = float(jax.scipy.stats.norm.ppf(critical_ratio))
        return self.d0 + b_hat * self.promo + self.sigma_d * z

    def run(self, seed_data: int = 0, seed_eval: int = 1) -> list[TaskResult]:
        """Estimate demand response (oracle / causal / predictive) and score the induced order."""
        system = ConfoundedLinearSystem(a=0.0, b_true=1.0, c=2.0, kappa=self.kappa)
        data = system.sample(self.n_data, jax.random.key(seed_data))
        demand = (
            self.d0
            + system.b_true * self.promo
            + self.sigma_d * jax.random.normal(jax.random.key(seed_eval), (self.n_eval,))
        )

        def cost_of(order: float) -> float:
            over = jnp.maximum(order - demand, 0.0)
            under = jnp.maximum(demand - order, 0.0)
            return float(jnp.mean(self.holding * over + self.stockout * under))

        oracle_cost = cost_of(self._order(system.b_true))
        controllers = {
            "oracle": system.b_true,
            "causal-CHC": float(estimate_control_effect(data, adjust_for=("z",))),
            "predictive": float(estimate_control_effect(data, adjust_for=())),
        }
        results = []
        for name, b_hat in controllers.items():
            order = self._order(b_hat)
            results.append(
                TaskResult(
                    controller=name,
                    cost=cost_of(order),
                    regret=cost_of(order) - oracle_cost,
                    constraint_violations=float(jnp.mean(demand > order)),  # stockout rate
                    ood_rate=0.0,  # single fixed-promo order; action support not applicable
                )
            )
        return results


class _BumpActuator(eqx.Module):
    """Plant whose control effectiveness peaks then decays: ``effect(u) = u·exp(-(u/u_sat)^2)``.

    Near ``u=0`` the effect is ~linear (a linear model is right on-support); for ``|u| >> u_sat``
    the actuator loses effectiveness, so extrapolating to large actions yields almost no effect.
    """

    a_matrix: Array
    b_matrix: Array
    u_sat: float

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        effect = u * jnp.exp(-((u / self.u_sat) ** 2))
        return self.a_matrix @ x + self.b_matrix @ effect


@dataclass(frozen=True)
class SupportShiftTask:
    """Model exploitation under support shift — where *pessimism*, not causality, is the safeguard.

    A linear model matches the true plant on the offline action support, but the plant's control
    effectiveness collapses for large actions. The greedy controller extrapolates off-support to
    chase gains the model promises and stalls; pessimism keeps actions in-support and stays safe.
    """

    x0: float = 2.0  # start far from target so the controller wants a big push
    x_target: float = 0.0
    dt: float = 0.1
    horizon: int = 25
    u_lo: float = -8.0
    u_hi: float = 8.0
    u_sat: float = 0.8  # actuator sweet-spot scale
    control_weight: float = 0.001
    lam_supp: float = 5.0
    n_data: int = 4000
    inner_steps: int = 300

    def run(self, seed_data: int = 0) -> list[TaskResult]:
        """Optimise on the model (greedy/pessimistic) and the plant (oracle); score on the plant."""
        a = jnp.array([[0.0, 1.0], [-1.0, -0.2]])
        b = jnp.array([[0.0], [1.0]])
        model = HybridDynamics(
            known=LinearDynamics(a_matrix=a, b_matrix=b), residual=ZeroResidual(2)
        )
        plant = _BumpActuator(a_matrix=a, b_matrix=b, u_sat=self.u_sat)

        k_x, k_u = jax.random.split(jax.random.key(seed_data))
        xs_data = jax.random.normal(k_x, (self.n_data, 2))
        us_data = 0.4 * jax.random.normal(k_u, (self.n_data, 1))  # narrow action support
        support = SupportModel.fit(xs_data, us_data)
        u_support = float(jnp.quantile(jnp.abs(us_data), 0.99))

        cost = QuadraticCost(
            Q=jnp.diag(jnp.array([1.0, 0.0])),
            R=jnp.array([[self.control_weight]]),
            Qf=jnp.diag(jnp.array([10.0, 1.0])),
            x_target=jnp.array([self.x_target, 0.0]),
        )
        x0 = jnp.array([self.x0, 0.0])
        us0 = jnp.zeros((self.horizon, 1))

        us_greedy, _ = projected_gradient_control(
            model, x0, us0, self.dt, cost, self.u_lo, self.u_hi, steps=self.inner_steps
        )
        us_pess, _ = pessimistic_control(
            model,
            x0,
            us0,
            self.dt,
            cost,
            support,
            self.lam_supp,
            self.u_lo,
            self.u_hi,
            steps=self.inner_steps,
        )
        us_oracle, _ = projected_gradient_control(
            plant, x0, us0, self.dt, cost, self.u_lo, self.u_hi, steps=self.inner_steps
        )

        def true_cost(us: Array) -> float:
            return float(total_cost(plant, x0, us, self.dt, cost))

        def ood(us: Array) -> float:
            return float(jnp.mean(jnp.abs(us) > u_support))

        oracle_cost = true_cost(us_oracle)
        controllers = (("oracle", us_oracle), ("pessimistic", us_pess), ("greedy", us_greedy))
        return [
            TaskResult(
                controller=name,
                cost=true_cost(us),
                regret=true_cost(us) - oracle_cost,
                constraint_violations=0.0,
                ood_rate=ood(us),
            )
            for name, us in controllers
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
