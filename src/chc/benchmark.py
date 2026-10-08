"""Benchmark v0: confounded, constrained control tasks with oracle regret.

Each task ships a confounded offline dataset, a true plant with a computable oracle controller, and
an evaluation reporting **regret vs oracle**, **constraint violations**, and **out-of-support action
rate**. The point is to measure *where* causal control beats predictive control — and to be honest
where it does not. v0 has pricing (steering), inventory (newsvendor), support-shift (pessimism),
model-uncertainty (calibrated ensemble) and confounding-robust (sensitivity radius under a *hidden*
confounder) tasks, plus causal-dynamics (the confounding sits in the plant's own control channel),
all in the same ``TaskResult`` / ``leaderboard`` shape.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.stats
import numpy as np
from jax import Array

from chc.causal import ConfoundedLinearSystem, estimate_control_effect
from chc.control import lbfgs_box_control, projected_gradient_control
from chc.cost import QuadraticCost, total_cost
from chc.delay import exact_delayed_rollout, robust_delay_design
from chc.dynamics import Dynamics, HybridDynamics, LinearDynamics
from chc.dynamics_id import ConfoundedControlAffineSystem, fit_causal_residual
from chc.estimators import BackdoorOLS, CausalEffectEstimator
from chc.flagship import closed_loop
from chc.integrate import rk4_step, rollout
from chc.irf import delay_estimate
from chc.residual import ControlAffineResidual, ZeroResidual
from chc.support import SupportModel, pessimistic_control
from chc.uncertainty import (
    ConfoundingRobustPenalty,
    EnsembleResidual,
    EnsembleUncertainty,
    fit_ensemble,
)


@dataclass(frozen=True)
class TaskResult:
    """One controller's score on one task."""

    controller: str
    cost: float
    # cost - oracle_cost. The oracle is the same controller on the true effect, a reference rather
    # than a floor, so on one noise path a controller can land a little below it.
    regret: float
    constraint_violations: float  # fraction of steps outside the safe state set
    ood_rate: float  # fraction of actions outside the logged action support


@dataclass(frozen=True)
class MultiSeedResult:
    """One controller's regret aggregated across seeds, with a bootstrap confidence interval."""

    controller: str
    regret_mean: float
    regret_lo: float  # 95% percentile-bootstrap CI lower bound
    regret_hi: float  # 95% percentile-bootstrap CI upper bound
    regret_std: float  # across-seed standard deviation
    ood_mean: float  # mean out-of-support action rate (the safety signal)
    n_seeds: int


@dataclass(frozen=True)
class PricingTask:
    """Confounded linear steering: drive x to a target; effect of u is confounded in the logs.

    The oracle is the controllers' own rule, the one-step certainty-equivalent action, at the true
    effect and on the same noise path, so a regret is what the estimate costs under that rule. It
    is not the best policy: the rule ignores ``control_weight``, which the finite-horizon LQR policy
    weighs, and over 4 000 noise paths that policy costs 0.00098 less on average (standard error
    0.0001), 0.0095 less on the path the task draws, in float64. A rule tuned to the drawn path, an
    effect of 1.03, costs 0.029 less there and 0.018 more on average, and a plan that reads the
    path's noise 0.29 less; neither is open to a controller. A controller can therefore land a
    little below the oracle on one path.
    """

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

    def run(
        self,
        seed_data: int = 0,
        seed_eval: int = 1,
        estimator: CausalEffectEstimator | None = None,
    ) -> list[TaskResult]:
        """Fit the effect (oracle / causal / predictive) from logs and score each controller.

        The CHC controller uses the pluggable ``estimator`` (default ``BackdoorOLS``, the linear
        adjustment); pass ``DoubleML()`` / ``EconMLDoubleML()`` to swap the causal backend. The
        predictive baseline stays the fixed naive (unadjusted) fit.
        """
        estimator = estimator or BackdoorOLS()
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
            "causal-CHC": float(estimator.estimate(data, covariates=("x", "z")).effect),
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

    The oracle orders the critical fractile of the true demand, the order of least expected cost,
    and is scored on the same ``n_eval`` draws as every controller. On those 5 000 draws the order
    at their own fractile costs 5.7e-5 less in float64, an order that reads the draws it is scored
    on.
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

    def run(
        self,
        seed_data: int = 0,
        seed_eval: int = 1,
        estimator: CausalEffectEstimator | None = None,
    ) -> list[TaskResult]:
        """Estimate demand response (oracle / causal / predictive) and score the induced order.

        ``estimator`` is the pluggable causal backend for the CHC order (default ``BackdoorOLS``).
        """
        estimator = estimator or BackdoorOLS()
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
            "causal-CHC": float(estimator.estimate(data, covariates=("x", "z")).effect),
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

    The oracle is the planner's descent from zero on the true plant, and it is the best plan. Past
    ``u_sat / sqrt(2)`` an action buys an effect a smaller one gives for less, and inside it the
    effect rises with the action and the action's cost is convex in the effect, while the plant is
    linear in the effect: there the problem is convex. The oracle's actions all lie inside, so
    where it stops on its rule, its stationarity 8.4e-7 in float64, is the problem's minimum.
    Descents from 73 starts -- zero, each side of the box throughout, at each step, and for the
    first 1, 2, 3, 5 or 10 steps, the other two plans and 8 seeded draws -- end on 39 plans, none
    below the oracle by more than 2.2e-12.
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
    inner_steps: int = 10_000

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


class _CubicDragActuator(eqx.Module):
    """Plant whose control effect saturates then reverses: ``effect(u) = u - drag·u^3``.

    Near ``u=0`` the effect is ~linear (the known model is right on the offline support); for large
    ``|u|`` the cubic drag dominates and the effect turns negative, so a controller that trusts a
    linear extrapolation and pushes hard backfires on the true plant.
    """

    a_matrix: Array
    b_matrix: Array
    drag: float

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return self.a_matrix @ x + self.b_matrix @ (u - self.drag * u**3)


# The oracle's starts on a plant whose effect reverses: zero, then each side of the box held for the
# first k of these steps. The edge is where the reversed effect is strongest.
_EDGE_PREFIXES = (1, 2, 3)


def _best_descent(
    plant: Dynamics,
    x0: Array,
    cost: QuadraticCost,
    dt: float,
    horizon: int,
    u_lo: float,
    u_hi: float,
    steps: int,
) -> Array:
    """The cheapest plan on ``plant`` found by descents from a fixed set of starts.

    The starts are zero, then the upper and the lower side of the box held for the first ``k``
    steps and zero after, for each ``k`` in ``_EDGE_PREFIXES``: seven, in that order. From each the
    planner descends, L-BFGS-B goes on from where it stopped, and the planner descends again to its
    own rule; each runs to ``steps`` at most. The planner alone crawls along this plant's ravines:
    from the upper side held for two steps it ended 10 000 steps at 3.926 and 200 000 at 3.773716,
    its stationarity 3.3e-3, where the three in turn stop on the planner's rule at 3.773697. A
    later plan replaces an earlier one only where it costs strictly less, so a tie keeps the
    earlier start's plan.
    """
    levers = cost.R.shape[0]
    starts = [jnp.zeros((horizon, levers))]
    for k in _EDGE_PREFIXES:
        starts += [jnp.zeros((horizon, levers)).at[:k].set(side) for side in (u_hi, u_lo)]
    best, best_cost = starts[0], math.inf
    for start in starts:
        us, _ = projected_gradient_control(plant, x0, start, dt, cost, u_lo, u_hi, steps=steps)
        us, _ = lbfgs_box_control(plant, x0, us, dt, cost, u_lo, u_hi, steps=steps)
        us, _ = projected_gradient_control(plant, x0, us, dt, cost, u_lo, u_hi, steps=steps)
        value = float(total_cost(plant, x0, us, dt, cost))
        if value < best_cost:
            best, best_cost = us, value
    return best


@functools.cache
def _cubic_drag_oracle(task: ModelUncertaintyTask, dtype: np.dtype) -> Array:
    """:class:`ModelUncertaintyTask`'s oracle, searched once per task and precision.

    It reads no data, so every seed of a task shares it; ``dtype`` keys the precision it ran in.
    """
    del dtype
    a = jnp.array([[0.0, 1.0], [-1.0, -0.2]])
    b = jnp.array([[0.0], [1.0]])
    plant = _CubicDragActuator(a_matrix=a, b_matrix=b, drag=task.drag)
    return _best_descent(
        plant,
        jnp.array([task.x0, 0.0]),
        task._cost(),
        task.dt,
        task.horizon,
        task.u_lo,
        task.u_hi,
        task.inner_steps,
    )


@dataclass(frozen=True)
class ModelUncertaintyTask:
    """Model exploitation where the safeguard is calibrated model uncertainty, not support distance.

    A residual is fit from offline data on a narrow action support; its deep-ensemble members agree
    there and **disagree** off it. The greedy controller trusts the ensemble mean and pushes into
    that high-uncertainty region, where the true plant's cubic drag backfires; calibrated pessimism
    penalises the ensemble disagreement and stays where the learned model is trustworthy -- the
    ``chc.uncertainty`` ``U`` term, complementing the density-distance ``D`` of ``SupportShift``.

    The oracle is the cheapest of seven descents on the true plant, from zero and from each side of
    the box held for the first one, two or three steps (``_best_descent``), searched once per task
    and shared by its seeds. The plant acts through ``u - drag u^3``, which turns over at
    ``|u| = 2.58`` and reaches -68.8 on the box's edge, so the best plan pushes through the reversed
    effect: at 3.773697 in float64 its first four actions are 8, 6.63, -8 and -6.81, and the rest
    stay inside the turn. The descent from zero alone stops at the effect's peak, at 15.520831, and
    regrets read against it were 11.75 too low. No plan found costs less: descents from 152 starts,
    each run to 210 000 steps and polished by L-BFGS-B, and a dynamic programme over the state, end
    on many plans, none below 3.773697. Every plan costs at least 3.76084, the minimum of the
    problem relaxed to the effects, in which the plant is linear and the action's cost is replaced
    by its convex envelope (``tests/test_benchmark.py``), so the oracle is within 0.0129 of the best
    plan there is. Against it the calibrated plan's regret is 12.40 and greedy's 1360.58: the row
    measures what planning on the learned model costs against a plan that knows the plant's turn,
    which no model fit on the logged support can see.
    """

    x0: float = 2.0
    x_target: float = 0.0
    dt: float = 0.1
    horizon: int = 25
    u_lo: float = -8.0
    u_hi: float = 8.0
    drag: float = 0.15
    control_weight: float = 0.001
    lam_unc: float = 1000.0
    n_data: int = 2000
    n_members: int = 5
    fit_steps: int = 1000
    inner_steps: int = 10_000

    def _cost(self) -> QuadraticCost:
        return QuadraticCost(
            Q=jnp.diag(jnp.array([1.0, 0.0])),
            R=jnp.array([[self.control_weight]]),
            Qf=jnp.diag(jnp.array([10.0, 1.0])),
            x_target=jnp.array([self.x_target, 0.0]),
        )

    def run(self, seed_data: int = 0) -> list[TaskResult]:
        """Fit an ensemble on the support, then score greedy/calibrated/oracle on the plant."""
        a = jnp.array([[0.0, 1.0], [-1.0, -0.2]])
        b = jnp.array([[0.0], [1.0]])
        known = HybridDynamics(
            known=LinearDynamics(a_matrix=a, b_matrix=b), residual=ZeroResidual(2)
        )
        plant = _CubicDragActuator(a_matrix=a, b_matrix=b, drag=self.drag)

        k_x, k_u = jax.random.split(jax.random.key(seed_data))
        xs_data = jax.random.normal(k_x, (self.n_data, 2))
        us_data = 0.4 * jax.random.normal(k_u, (self.n_data, 1))  # narrow action support
        x_next = jax.vmap(lambda x, u: rk4_step(plant, 0.0, x, u, self.dt))(xs_data, us_data)
        data = {"x": xs_data, "u": us_data, "x_next": x_next}
        support = SupportModel.fit(xs_data, us_data)
        u_support = float(jnp.quantile(jnp.abs(us_data), 0.99))

        model_ens, _ = fit_ensemble(
            known, data, self.dt, n_members=self.n_members, steps=self.fit_steps, seed=seed_data + 1
        )
        uncertainty = EnsembleUncertainty(ensemble=cast(EnsembleResidual, model_ens.residual))

        cost = self._cost()
        x0 = jnp.array([self.x0, 0.0])
        us0 = jnp.zeros((self.horizon, 1))

        us_greedy, _ = projected_gradient_control(
            model_ens, x0, us0, self.dt, cost, self.u_lo, self.u_hi, steps=self.inner_steps
        )
        us_cal, _ = pessimistic_control(
            model_ens,
            x0,
            us0,
            self.dt,
            cost,
            support,
            0.0,
            self.u_lo,
            self.u_hi,
            steps=self.inner_steps,
            uncertainty=uncertainty,
            lam_unc=self.lam_unc,
        )
        us_oracle = _cubic_drag_oracle(self, jax.dtypes.canonicalize_dtype(jnp.float64))

        def true_cost(us: Array) -> float:
            return float(total_cost(plant, x0, us, self.dt, cost))

        def ood(us: Array) -> float:
            return float(jnp.mean(jnp.abs(us) > u_support))

        oracle_cost = true_cost(us_oracle)
        controllers = (("oracle", us_oracle), ("calibrated", us_cal), ("greedy", us_greedy))
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


@dataclass(frozen=True)
class ConfoundingRobustTask:
    """Control under HIDDEN confounding: the safeguard is a sensitivity radius, not adjustment.

    The actuator gain is calibrated from an observational log whose action was driven by a
    **latent** disturbance that also moved the transition. It is never recorded and there is no
    instrument, so the gain is only *partially identified* -- the adjustment fix of ``PricingTask``
    is unavailable by construction, which is why this task takes no estimator argument. Here the
    confounding **attenuates** the gain, so the greedy plan believes a weak actuator, over-commands,
    overshoots the target on the true plant, and demands actions far outside the logged range. The
    safeguard is ``chc.uncertainty.ConfoundingRobustPenalty`` in the ``lam_unc`` channel: the
    partial-identification radius times the action magnitude, bounding the transition error a
    confounded gain can inject per step.

    HONEST TRAPS. (1) The penalty is ONE-SIDED -- it can only shrink actions, so it rescues an
    understated gain and strictly *hurts* when the confounding inflates it instead (a test pins
    this). (2) ``gamma`` and the ``cvar_gap := b_hat`` calibration are the analyst's inputs, not
    learned; the row scores the controller at the *assumed* sensitivity, not an oracle one.
    (3) With no confounding the robust controller pays a strict premium, and it only wins beyond a
    problem-dependent confounding threshold -- at half the default confounding it still loses.
    (4) This is an OPEN-LOOP plan scored on the plant; the closed-loop counterpart is
    ``chc.sensitivity.confounding_robust_tracking_benchmark``.

    The oracle is the planner's descent from zero on the true plant. The plant is linear, so the
    cost is a convex quadratic in the plan and the descent's plan is the best one: the exact
    minimiser over the box, by bounded-variable least squares (Stark and Parker 1995), is 6.9e-11
    below it in float64, and descents from 73 starts end on the same plan.
    """

    x0: float = 2.0
    x_target: float = 0.0
    dt: float = 0.1
    horizon: int = 25
    u_lo: float = -8.0
    u_hi: float = 8.0
    b_gain: float = 1.0  # TRUE actuator gain; the log's b_true
    kappa: float = -0.5  # behaviour policy's response to the LATENT driver (<0 attenuates b_hat)
    confounding: float = 1.0  # the latent driver's direct push on the logged transition
    gamma: float = 5.0  # assumed sensitivity -- the ONE safeguard knob (covers the default bias)
    # Units bridge, not a tuned constant: the penalty bounds a VECTOR-FIELD error, so a step injects
    # dt*radius*||u||, and the quadratic cost converts a position error e at rate dJ/de ~ 2|x0-x*|.
    # Hence lam_unc ~ dt * 2|x0 - x*| = 0.1 * 4 ~ 0.2 (order of magnitude; the row is sensitive to
    # it -- the mechanism helps across ~[0.02, 0.4] and over-shrinks into a loss by 1.0).
    lam_unc: float = 0.2
    control_weight: float = 0.001
    overshoot_tol: float = 0.25  # blowing this far past the target counts as a violation
    n_data: int = 4000
    inner_steps: int = 10_000

    def run(self, seed_data: int = 0) -> list[TaskResult]:
        """Calibrate the gain on a confounded log, then score greedy/robust/oracle on the plant."""
        a = jnp.array([[0.0, 1.0], [-1.0, -0.2]])
        unit_b = jnp.array([[0.0], [1.0]])
        plant = HybridDynamics(
            known=LinearDynamics(a_matrix=a, b_matrix=self.b_gain * unit_b),
            residual=ZeroResidual(2),
        )

        # the log is the actuator channel (u enters only the velocity row) as a scalar response
        system = ConfoundedLinearSystem(
            a=0.0, b_true=self.b_gain, c=self.confounding, kappa=self.kappa, eta_scale=1.0
        )
        sampled = system.sample(self.n_data, jax.random.key(seed_data))
        logs = {name: sampled[name] for name in ("x", "u", "x_next")}  # z LATENT: never logged
        b_hat = float(estimate_control_effect(logs, adjust_for=()))  # float: hashable static field

        model = HybridDynamics(
            known=LinearDynamics(a_matrix=a, b_matrix=b_hat * unit_b), residual=ZeroResidual(2)
        )
        penalty = ConfoundingRobustPenalty.from_sensitivity(cvar_gap=b_hat, gamma=self.gamma)

        # inert at lam_supp=0.0 but evaluated unconditionally, so it must stay shape-valid
        support = SupportModel.fit(
            jax.random.normal(jax.random.key(seed_data + 1), (self.n_data, 2)),
            logs["u"][:, None],
        )
        u_support = float(jnp.quantile(jnp.abs(logs["u"]), 0.99))

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
        us_robust, _ = pessimistic_control(
            model,
            x0,
            us0,
            self.dt,
            cost,
            support,
            0.0,
            self.u_lo,
            self.u_hi,
            steps=self.inner_steps,
            uncertainty=penalty,
            lam_unc=self.lam_unc,
        )
        us_oracle, _ = projected_gradient_control(
            plant, x0, us0, self.dt, cost, self.u_lo, self.u_hi, steps=self.inner_steps
        )

        def true_cost(us: Array) -> float:
            return float(total_cost(plant, x0, us, self.dt, cost))

        def overshoot(us: Array) -> float:
            xs = rollout(plant, x0, us, self.dt)
            return float(jnp.mean(xs[:, 0] < self.x_target - self.overshoot_tol))

        def ood(us: Array) -> float:
            return float(jnp.mean(jnp.abs(us) > u_support))

        oracle_cost = true_cost(us_oracle)
        controllers = (("oracle", us_oracle), ("robust", us_robust), ("greedy", us_greedy))
        return [
            TaskResult(
                controller=name,
                cost=true_cost(us),
                regret=true_cost(us) - oracle_cost,
                constraint_violations=overshoot(us),
                ood_rate=ood(us),
            )
            for name, us in controllers
        ]


@dataclass(frozen=True)
class CausalDynamicsTask:
    """Confounding inside the *dynamics model*, not beside it -- the ``chc.dynamics_id`` consumer.

    Every other task here estimates a scalar effect and hands it to a controller. In this one the
    confounded object is the plant's own control channel: the log's action was chosen from a
    covariate that also moved the state rate, so a residual fitted by prediction error learns the
    *observational* response and the planner inherits it. Three ways in, scored on the same plant:

    * ``mse-id`` -- unadjusted, which is where ``chc.train``'s prediction-error fit lands.
    * ``causal-id`` -- the confounder is logged and adjusted for (the orthogonal moment).
    * ``causal-iv`` -- the confounder is *latent*; only an exogenous action shifter is available.

    HONEST TRAPS. (1) The failure is **silent on the safety channels**. The confounding attenuates
    the channel to ~0.02 of its true 1.0, so ``mse-id`` prices the actuator as useless against the
    ``control_weight`` penalty and gives up: measured ``max|u|`` 0.235 against the oracle's 2.516,
    so it reaches ``x = 0.135`` instead of 0.712. It never approaches the box, never leaves the
    logged action support (99th percentile of ``|u|`` is 4.4), and reports ``viol = ood = 0`` while
    conceding most of the achievable improvement. Regret is the only column that sees it -- a
    reminder that constraint and support diagnostics do not detect a mis-scaled channel. Shrinking
    ``control_weight`` flips the same bias into over-commanding, where those columns *do* fire.
    (2) The two identified rows are **not interchangeable**. Adjusting for a logged confounder gives
    regret 0.014; the instrument gives 0.131, because it explains only ~18% of the action's variance
    and identification rides on that share alone. Both beat the 6.20 of not identifying at all, but
    an instrument is a weaker substitute for the confounder than the word "identified" suggests.
    (3) Only the **channel** is identified; ``a_θ`` stays an observational-conditional drift, so
    this row scores planning, not forecasting. (4) The plant is control-affine by construction,
    which is the class the estimator and :func:`chc.plan.certify_safety` share -- a general
    nonlinear residual gets no orthogonality guarantee and no row here.

    The oracle is the planner's descent from zero on the true plant. The plant is linear in the
    state and the action, so the cost is a convex quadratic in the plan and the descent's plan is
    the best one: the exact minimiser over the box, by bounded-variable least squares, is 3.4e-13
    below it in float64, and descents from 74 starts end on the same plan.
    """

    x_target: float = 1.0
    dt: float = 0.05
    horizon: int = 25
    u_lo: float = -5.0
    u_hi: float = 5.0
    x_safe: float = 2.5  # |x[0]| <= x_safe; saturating on a mis-scaled channel blows through it
    control_weight: float = 0.1
    n_data: int = 4000
    inner_steps: int = 10_000

    def run(self, seed_data: int = 0) -> list[TaskResult]:
        """Fit the channel three ways from one confounded log, then score each plan on the plant."""
        drift = jnp.array([[-0.5, 0.1], [0.0, -0.3]])
        channel = jnp.array([[1.0], [0.5]])
        system = ConfoundedControlAffineSystem(
            drift=drift,
            channel=channel,
            confounder_to_rate=jnp.array([[2.0], [1.0]]),
            confounder_to_action=jnp.array([[-1.5]]),
            instrument_to_action=jnp.array([[0.8]]),
            dt=self.dt,
        )

        def known(t: float | Array, x: Array, u: Array) -> Array:
            return jnp.zeros_like(x)

        data = system.sample(self.n_data, jax.random.key(seed_data), known)
        u_support = float(jnp.quantile(jnp.abs(data["u"]), 0.99))

        plant = HybridDynamics(
            known=known,
            residual=ControlAffineResidual(
                drift=jnp.concatenate([jnp.zeros((2, 1)), drift], axis=1),
                channel=jnp.concatenate([channel[:, :, None], jnp.zeros((2, 1, 2))], axis=2),
            ),
        )
        cost = QuadraticCost(
            Q=jnp.eye(2),
            R=self.control_weight * jnp.eye(1),
            Qf=5.0 * jnp.eye(2),
            x_target=jnp.array([self.x_target, 0.0]),
        )
        x0, us0 = jnp.zeros(2), jnp.zeros((self.horizon, 1))

        def plan(model: HybridDynamics) -> Array:
            us, _ = projected_gradient_control(
                model, x0, us0, self.dt, cost, self.u_lo, self.u_hi, steps=self.inner_steps
            )
            return us

        def fitted(
            *, adjust_for: tuple[str, ...] = (), instrument: str | None = None
        ) -> HybridDynamics:
            fit = fit_causal_residual(
                known, data, self.dt, adjust_for=adjust_for, instrument=instrument
            )
            return HybridDynamics(known=known, residual=fit.residual)

        plans = {
            "oracle": plan(plant),
            "causal-id": plan(fitted(adjust_for=("z",))),
            "causal-iv": plan(fitted(instrument="w")),
            "mse-id": plan(fitted()),
        }
        costs = {
            name: float(total_cost(plant, x0, us, self.dt, cost)) for name, us in plans.items()
        }
        return [
            TaskResult(
                controller=name,
                cost=costs[name],
                regret=costs[name] - costs["oracle"],
                constraint_violations=float(
                    jnp.mean(jnp.abs(rollout(plant, x0, us, self.dt)[:, 0]) > self.x_safe)
                ),
                ood_rate=float(jnp.mean(jnp.abs(us) > u_support)),
            )
            for name, us in plans.items()
        ]


@dataclass(frozen=True)
class DelayOscillationTask:
    """A delay-blind controller walks into a Hopf bifurcation -- ``chc.delay``'s consumer.

    Generic marketplace shape: an incentive moves supply, but participants respond ``tau`` later, so
    the plant is ``x' = channel * u(t - tau)`` -- the delay sits on the **actuation** path. Under
    proportional feedback ``u = -K x`` that closes to ``x' = -channel*K*x(t - tau)``, whose exact
    stability boundary is ``channel*K*tau = pi/2`` (:func:`chc.delay.delay_margin` at pole 0). Three
    controllers, all minimising the *same* quadratic cost by the *same* grid search, differing only
    in the delay they assume:

    * ``delay-blind`` -- assumes ``tau = 0``. Its optimum is the memoryless ``sqrt(q/r)``, which for
      the shipped weights is ``3.10`` against an analytic ``sqrt(q/r) = 3.162`` -- the 2% gap is
      explicit Euler, whose effective decay ``-ln(1 - dt K)/dt`` exceeds ``K`` and so shifts the
      optimum down. The arm fails on its own terms, not because it was handed a bad gain:
      ``3.10 * channel * tau`` is **1.97x past** ``pi/2``, so the loop rings up instead of down.
    * ``delay-aware`` -- estimates the delay from the log with :func:`chc.irf.delay_estimate`, turns
      the interval into a design with :func:`chc.delay.robust_delay_design`, then runs the same grid
      search at that delay.
    * ``oracle`` -- the same grid search at the true ``tau``.

    HONEST NOTES. (1) **This is a row where the safety columns fire.** ``CausalDynamicsTask``
    documents the opposite trap -- a mis-scaled channel that concedes regret while reporting
    ``viol = ood = 0``. Here the failure is loud: the blind arm leaves the safe set and the logged
    action support on most steps. Both traps are real; neither column is a general detector.
    (2) The blind arm's **cost magnitude is an artefact of ``state_cap``**, which clips the
    diverging trajectory so the number stays finite and precision-independent. Read it as a verdict,
    not a quantity; the ordering and the constraint columns carry the content.
    (3) The delay is estimated from the **rate**, not the level. This plant is an integrator, so the
    level's impulse response is a *step* -- a plateau with no peak, on which ``delay_estimate``
    correctly returns an interval spanning it rather than inventing a mode. Differencing turns the
    step back into a spike. That is a modelling obligation on the caller, not a fallback.
    (4) The log is sampled every ``observe_every`` plant steps, putting ``tau`` at 3.33 samples --
    deliberately **off** the observation grid, because an on-grid delay makes the estimate exact and
    the row uninformative. That is also the case ``peak_lag(refine=True)`` exists for, so this task
    turns it on; it recovers ``0.947`` of a true ``1.0`` where the integer argmax can only say
    ``0.900``. The residual 5.3% is not noise -- it is the parabola's documented shrinkage, which
    for a fraction ``f = 1/3`` predicts ``0.950``.
    (5) The estimate lands **below** the truth, which by ``chc.delay.delay_ball`` is the
    destabilising direction. It is harmless here only because that ball has 76% of relative slack;
    on a plant with a tighter one the same 5.3% would matter.

    The oracle's gain is the best of the grid: on its 400 gains the cost has one minimum, at
    0.7175, and 20 001 gains between that one's neighbours find 0.7199, 1.6e-5 cheaper, less than
    the 8.3e-5 the cost rises to the nearer neighbour. It is the best proportional gain to the
    grid's resolution; no other feedback law is searched.
    """

    tau: float = 1.0
    channel: float = 1.0
    dt: float = 0.01  # plant integration step
    observe_every: int = 30  # log sampling: tau is 3.33 samples, off the grid on purpose
    score_horizon: float = 30.0
    state_weight: float = 1.0
    control_weight: float = 0.1
    n_obs: int = 1200
    log_noise: float = 0.4
    x_safe: float = 3.0
    state_cap: float = 20.0  # the plant saturates; keeps the diverging row finite and legible
    gain_grid: int = 400

    def _log(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Open-loop log under a zero-order-held random incentive, observed coarsely."""
        rng = np.random.default_rng(seed)
        lag, steps = round(self.tau / self.dt), self.n_obs * self.observe_every
        coarse = rng.standard_normal(self.n_obs)
        held = np.repeat(coarse, self.observe_every)
        x = np.zeros(steps + 1)
        noise = self.log_noise * np.sqrt(self.dt) * rng.standard_normal(steps)
        for t in range(steps):
            drive = self.channel * held[t - lag] if t >= lag else 0.0
            x[t + 1] = x[t] + self.dt * drive + noise[t]
        return coarse, x[:: self.observe_every][: self.n_obs]

    def _sweep(self, gains: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
        """Closed-loop states and actions for every gain, on a plant with delay ``tau``.

        ``tau = 0`` gives ``lag = 0``, where ``exact_delayed_rollout`` reads the delayed state as
        the current one -- the memoryless plant, under the *same* Euler scheme as every other arm.
        The blind controller is therefore not handed a different numerical model, only a different
        assumed delay.
        """
        lag, steps = round(tau / self.dt), int(self.score_horizon / self.dt)

        def core(t: float | Array, x: Array, x_delayed: Array, u: Array) -> Array:
            return (
                -self.channel * u[0] * x_delayed
            )  # the gain rides in as data, so one trace serves

        def one(gain: Array) -> Array:
            return exact_delayed_rollout(
                core, jnp.array([1.0]), jnp.full((steps, 1), gain), self.dt, lag
            )[:, 0]

        states = np.asarray(jax.vmap(one)(jnp.asarray(gains)))
        states = np.clip(np.nan_to_num(states, nan=self.state_cap), -self.state_cap, self.state_cap)
        # the controller pays for what it ISSUES, u(t) = -K x(t), which is also what ``ood_rate``
        # compares against -- the log records issued incentives, not the ones arriving tau later
        return states, -gains[:, None] * states[:, :steps]

    def _costs(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self.dt * (
            self.state_weight * np.sum(states[:, :-1] ** 2, axis=1)
            + self.control_weight * np.sum(actions**2, axis=1)
        )

    def _best_gain(self, assumed_tau: float) -> float:
        """The gain minimising the cost, computed on a plant with the ASSUMED delay."""
        grid = np.geomspace(0.05, 6.0, self.gain_grid)
        return float(grid[int(np.argmin(self._costs(*self._sweep(grid, assumed_tau))))])

    def run(self, seed_data: int = 0) -> list[TaskResult]:
        """Estimate the delay from one log, then score three gains on the true delayed plant."""
        coarse, observed = self._log(seed_data)
        estimate = delay_estimate(
            {"x": np.diff(observed), "u": coarse[:-1]},
            horizon=12,
            dt=self.dt * self.observe_every,
            adjust_for=(),
            refine=True,  # the delay is off the observation grid -- what refinement is for
            seed=seed_data,
        )
        design = robust_delay_design(max(estimate.lo, self.dt), estimate.hi)

        gains = {
            "delay-blind": self._best_gain(0.0),
            "delay-aware": self._best_gain(design.tau_design),
            "oracle": self._best_gain(self.tau),
        }
        states, actions = self._sweep(np.array(list(gains.values())), self.tau)
        costs = self._costs(states, actions)
        support = float(np.quantile(np.abs(coarse), 0.99))
        oracle_cost = float(costs[-1])
        return [
            TaskResult(
                controller=name,
                cost=float(costs[row]),
                regret=float(costs[row] - oracle_cost),
                constraint_violations=float(np.mean(np.abs(states[row]) > self.x_safe)),
                ood_rate=float(np.mean(np.abs(actions[row]) > support)),
            )
            for row, name in enumerate(gains)
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


class BenchmarkTask(Protocol):
    """A benchmark task: one data seed in, one :class:`TaskResult` per controller out."""

    def run(self, seed_data: int = ...) -> list[TaskResult]: ...


def _bootstrap_ci(
    values: np.ndarray, *, level: float = 0.95, n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """Percentile-bootstrap confidence interval for the mean of ``values`` (NumPy only)."""
    if values.size < 2:
        point = float(values.mean()) if values.size else float("nan")
        return point, point
    rng = np.random.default_rng(seed)
    resampled = values[rng.integers(0, values.size, size=(n_boot, values.size))]
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(resampled.mean(axis=1), [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def run_multiseed(task: BenchmarkTask, seeds: Sequence[int]) -> list[MultiSeedResult]:
    """Aggregate ``task`` across seeds into per-controller regret with a bootstrap CI.

    ``task.run(seed_data=s)`` is called once per seed and the results grouped by controller. The
    interval is a percentile bootstrap over the seeds -- honest error bars on "does this controller
    actually win", replacing a single-seed point regret that could be luck of the draw.
    """
    regrets: dict[str, list[float]] = {}
    oods: dict[str, list[float]] = {}
    order: list[str] = []
    for seed in seeds:
        for result in task.run(seed_data=int(seed)):
            if result.controller not in regrets:
                regrets[result.controller], oods[result.controller] = [], []
                order.append(result.controller)
            regrets[result.controller].append(result.regret)
            oods[result.controller].append(result.ood_rate)
    summaries = []
    for controller in order:
        arr = np.asarray(regrets[controller], dtype=np.float64)
        lo, hi = _bootstrap_ci(arr)
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        ood = float(np.mean(oods[controller]))
        summaries.append(MultiSeedResult(controller, float(arr.mean()), lo, hi, std, ood, arr.size))
    return summaries


def leaderboard_multiseed(results: list[MultiSeedResult]) -> str:
    """Format multi-seed results sorted by mean regret (best first), with 95% bootstrap CIs."""
    header = f"{'controller':<14}{'regret':>10}{'95% CI':>22}{'ood':>7}{'seeds':>7}"
    rows = [
        f"{r.controller:<14}{r.regret_mean:>10.2f}"
        f"{f'[{r.regret_lo:.2f}, {r.regret_hi:.2f}]':>22}{r.ood_mean:>7.2f}{r.n_seeds:>7d}"
        for r in sorted(results, key=lambda r: r.regret_mean)
    ]
    return "\n".join([header, *rows])
