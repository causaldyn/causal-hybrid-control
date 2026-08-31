"""Benchmark v0: confounded, constrained control tasks with oracle regret — the moat (``plans/06``).

Each task ships a confounded offline dataset, a true plant with a computable oracle controller, and
an evaluation reporting **regret vs oracle**, **constraint violations**, and **out-of-support action
rate**. The point is to measure *where* causal control beats predictive control — and to be honest
where it does not. v0 has pricing (steering), inventory (newsvendor), support-shift (pessimism),
model-uncertainty (calibrated ensemble) and confounding-robust (sensitivity radius under a *hidden*
confounder) tasks, plus causal-dynamics (the confounding sits in the plant's own control channel),
all in the same ``TaskResult`` / ``leaderboard`` shape.
"""

from __future__ import annotations

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
from chc.control import projected_gradient_control
from chc.cost import QuadraticCost, total_cost
from chc.dynamics import HybridDynamics, LinearDynamics
from chc.dynamics_id import ConfoundedControlAffineSystem, fit_causal_residual
from chc.estimators import BackdoorOLS, CausalEffectEstimator
from chc.flagship import closed_loop
from chc.integrate import rk4_step, rollout
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
    regret: float  # cost - oracle_cost (>= 0; the oracle knows the true effect)
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


@dataclass(frozen=True)
class ModelUncertaintyTask:
    """Model exploitation where the safeguard is calibrated model uncertainty, not support distance.

    A residual is fit from offline data on a narrow action support; its deep-ensemble members agree
    there and **disagree** off it. The greedy controller trusts the ensemble mean and pushes into
    that high-uncertainty region, where the true plant's cubic drag backfires; calibrated pessimism
    penalises the ensemble disagreement and stays where the learned model is trustworthy -- the
    ``chc.uncertainty`` ``U`` term, complementing the density-distance ``D`` of ``SupportShift``.
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

        cost = QuadraticCost(
            Q=jnp.diag(jnp.array([1.0, 0.0])),
            R=jnp.array([[self.control_weight]]),
            Qf=jnp.diag(jnp.array([10.0, 1.0])),
            x_target=jnp.array([self.x_target, 0.0]),
        )
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
        us_oracle, _ = projected_gradient_control(
            plant, x0, us0, self.dt, cost, self.u_lo, self.u_hi, steps=self.inner_steps
        )

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
    ``control_weight`` penalty and gives up: measured ``max|u|`` 0.22 against the oracle's 2.49, so
    it reaches ``x = 0.12`` instead of 0.71. It never approaches the box, never leaves the logged
    action support (99th percentile of ``|u|`` is 4.4), and reports ``viol = ood = 0`` while
    conceding most of the achievable improvement. Regret is the only column that sees it -- a
    reminder that constraint and support diagnostics do not detect a mis-scaled channel. Shrinking
    ``control_weight`` flips the same bias into over-commanding, where those columns *do* fire.
    (2) The two identified rows are **not interchangeable**. Adjusting for a logged confounder gives
    regret 0.014; the instrument gives 0.132, because it explains only ~18% of the action's variance
    and identification rides on that share alone. Both beat the 6.41 of not identifying at all, but
    an instrument is a weaker substitute for the confounder than the word "identified" suggests.
    (3) Only the **channel** is identified; ``a_θ`` stays an observational-conditional drift, so
    this row scores planning, not forecasting. (4) The plant is control-affine by construction,
    which is the class the estimator and :func:`chc.plan.certify_safety` share -- a general
    nonlinear residual gets no orthogonality guarantee and no row here.
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
    header = f"{'controller':<14}{'regret':>10}{'95% CI':>20}{'ood':>7}{'seeds':>7}"
    rows = [
        f"{r.controller:<14}{r.regret_mean:>10.2f}"
        f"{f'[{r.regret_lo:.2f}, {r.regret_hi:.2f}]':>20}{r.ood_mean:>7.2f}{r.n_seeds:>7d}"
        for r in sorted(results, key=lambda r: r.regret_mean)
    ]
    return "\n".join([header, *rows])
