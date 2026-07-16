"""Support / pessimism: keep offline-trained control inside the region the data justifies.

``SupportModel`` scores how far a state-action pair ``(x, u)`` sits from the offline data cloud
(squared Mahalanobis distance); ``pessimistic_control`` penalises leaving that support, so the
controller does not exploit the model where it was never trained. This is the offline-safety layer
(``plans/02`` §3) in a minimal first form — density-distance only; ensembles / DAREK bounds later.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from chc.cost import QuadraticCost, total_cost
from chc.dynamics import Dynamics
from chc.integrate import rollout


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
) -> tuple[Array, Array]:
    """Projected-gradient OC with a support penalty ``λ·Σ D²(x, u)``.

    Uses autodiff for the augmented-objective gradient (validated equal to the discrete adjoint in
    ``01 §4.1``). Returns the optimised controls and the **task**-cost history (penalty excluded, so
    runs at different ``lam_supp`` are comparable).
    """

    def task(us: Array) -> Array:
        return total_cost(model, x0, us, dt, cost)

    def augmented(us: Array) -> Array:
        xs = rollout(model, x0, us, dt)
        return task(us) + lam_supp * support.penalty_trajectory(xs[:-1], us)

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
