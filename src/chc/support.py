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
import numpy as np
from jax import Array

from chc.control import _backtrack
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


@eqx.filter_jit
def _pessimistic_loop(
    model: Dynamics,
    x0: Array,
    us0: Array,
    dt: float,
    cost: QuadraticCost,
    support: SupportModel,
    lam_supp: float,
    u_lo: float,
    u_hi: float,
    steps: int,
    lr0: float,
    tol: float,
    uncertainty: PenaltyModel | None,
    lam_unc: float,
) -> tuple[Array, Array, Array]:
    """The penalised descent as one XLA program, mirroring :func:`chc.control`'s shape exactly.

    Module level for the same reason: the jitted objective and its gradient used to be built inside
    :func:`pessimistic_control`, so each call got an empty compilation cache and recompiled the
    augmented gradient every solve instead of once per problem shape.

    Acceptance is on the *augmented* cost and the recorded history is the *task* cost, so both are
    carried through the loop -- runs at different penalty weights stay comparable.
    """

    def task(us: Array) -> Array:
        return total_cost(model, x0, us, dt, cost)

    def augmented(us: Array) -> Array:
        xs = rollout(model, x0, us, dt)
        penalty = lam_supp * support.penalty_trajectory(xs[:-1], us)
        if uncertainty is not None:
            penalty = penalty + lam_unc * uncertainty.penalty_trajectory(xs[:-1], us)
        return task(us) + penalty

    grad_aug = jax.grad(augmented)
    us = jnp.clip(us0, u_lo, u_hi)
    initial = augmented(us)
    values = jnp.zeros((steps + 1,), dtype=initial.dtype).at[0].set(task(us))

    def descending(carry: tuple[Array, Array, Array, Array, Array]) -> Array:
        taken, _, _, _, alive = carry
        return jnp.logical_and(taken < steps, alive)

    def descend(
        carry: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array]:
        taken, us, current, values, _ = carry
        us, current, accepted = _backtrack(
            us, current, grad_aug(us), u_lo, u_hi, lr0, tol, augmented
        )
        taken = jnp.where(accepted, taken + 1, taken)
        return taken, us, current, values.at[taken].set(task(us)), accepted

    taken, optimised, _, values, _ = jax.lax.while_loop(
        descending, descend, (jnp.asarray(0), us, initial, values, jnp.asarray(True))
    )
    return optimised, values, taken


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
    steps: int = 10_000,
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

    optimised, values, taken = _pessimistic_loop(
        model,
        x0,
        us0,
        dt,
        cost,
        support,
        lam_supp,
        u_lo,
        u_hi,
        steps,
        lr0,
        tol,
        uncertainty,
        lam_unc,
    )
    return optimised, jnp.asarray(np.asarray(values)[: int(taken) + 1].tolist())
