"""Support / pessimism: keep offline-trained control inside the region the data justifies.

``SupportModel`` scores how far a state-action pair ``(x, u)`` sits from the offline data cloud
(squared Mahalanobis distance ``D``); ``pessimistic_control`` penalises leaving that support, so the
controller does not exploit the model where it was never trained. This is the offline-safety layer
(``plans/02`` §3): the objective is ``J_task + λ_unc·Σ U + λ_supp·Σ D``, where the calibrated
predictive-uncertainty term ``U`` comes from ``chc.uncertainty`` (deep ensemble / conformal) and
this module supplies ``D`` and the controller that combines them through the ``PenaltyModel``.
"""

from __future__ import annotations

from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from chc.cost import QuadraticCost, total_cost
from chc.dynamics import Dynamics
from chc.integrate import rollout


class PenaltyModel(Protocol):
    """Anything that scores a trajectory penalty ``penalty_trajectory(xs, us) -> scalar``.

    Both ``SupportModel`` (density distance ``D``) and the ``chc.uncertainty`` scorers (calibrated
    predictive uncertainty ``U``) satisfy it, so they are interchangeable penalty channels.
    """

    def penalty_trajectory(self, xs: Array, us: Array) -> Array: ...


class SupportModel(eqx.Module):
    """Gaussian support of the offline ``(x, u)`` cloud; scores squared Mahalanobis distance."""

    mean: Array
    precision: Array  # inverse covariance of concatenated (x, u)

    @classmethod
    def fit(cls, xs: Array, us: Array, ridge: float = 1e-3) -> SupportModel:
        z = jnp.concatenate([xs, us], axis=1)
        mean = jnp.mean(z, axis=0)
        centered = z - mean
        cov = (centered.T @ centered) / z.shape[0]
        precision = jnp.linalg.inv(cov + ridge * jnp.eye(z.shape[1]))
        return cls(mean=mean, precision=precision)

    def squared_distance(self, x: Array, u: Array) -> Array:
        d = jnp.concatenate([x, u]) - self.mean
        return d @ self.precision @ d

    def penalty_trajectory(self, xs: Array, us: Array) -> Array:
        """Total off-support penalty over the visited (x, u) pairs (xs: (H,n), us: (H,m))."""
        return jnp.sum(jax.vmap(self.squared_distance)(xs, us))


def pessimistic_control(
    model: Dynamics,
    x0: Array,
    us0: Array,
    dt: float,
    cost: QuadraticCost,
    support: SupportModel,
    lam_supp: float,
    u_lo: float,
    u_hi: float,
    steps: int = 200,
    lr0: float = 0.2,
    tol: float = 1e-9,
    uncertainty: PenaltyModel | None = None,
    lam_unc: float = 0.0,
) -> tuple[Array, Array]:
    """Projected-gradient OC with an offline-safety penalty ``λ_supp·Σ D + λ_unc·Σ U``.

    ``support`` supplies the density-distance term ``D``; the optional ``uncertainty`` scorer gives
    the calibrated predictive-uncertainty term ``U`` (a ``chc.uncertainty`` ensemble/conformal).
    Uses autodiff for the augmented-objective gradient (validated equal to the discrete adjoint in
    ``01 §4.1``). Returns the optimised controls and the **task**-cost history (penalties excluded,
    so runs at different weights are comparable).
    """

    def task(us: Array) -> Array:
        return total_cost(model, x0, us, dt, cost)

    def augmented(us: Array) -> Array:
        xs = rollout(model, x0, us, dt)
        penalty = lam_supp * support.penalty_trajectory(xs[:-1], us)
        if uncertainty is not None:
            penalty = penalty + lam_unc * uncertainty.penalty_trajectory(xs[:-1], us)
        return task(us) + penalty

    grad_aug = eqx.filter_jit(jax.grad(augmented))
    aug = eqx.filter_jit(augmented)
    task_jit = eqx.filter_jit(task)

    us = jnp.clip(us0, u_lo, u_hi)
    current = aug(us)
    history = [float(task_jit(us))]

    for _ in range(steps):
        grad = grad_aug(us)
        lr = lr0
        improved = False
        candidate = us
        candidate_cost = current
        for _ls in range(40):
            candidate = jnp.clip(us - lr * grad, u_lo, u_hi)
            candidate_cost = aug(candidate)
            if candidate_cost < current - tol:
                improved = True
                break
            lr *= 0.5
        if not improved:
            break
        us, current = candidate, candidate_cost
        history.append(float(task_jit(us)))

    return us, jnp.asarray(history)
