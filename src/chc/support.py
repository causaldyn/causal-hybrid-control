"""Support / pessimism: keep offline-trained control inside the region the data justifies.

``SupportModel`` scores how far a state-action pair ``(x, u)`` sits from the offline data cloud
(squared Mahalanobis distance ``D``); ``pessimistic_control`` penalises leaving that support, so the
controller does not exploit the model where it was never trained. This is the offline-safety layer
(``plans/02`` §3): the objective is ``J_task + λ_unc·Σ U + λ_supp·Σ D``, where the calibrated
predictive-uncertainty term ``U`` comes from ``chc.uncertainty`` (deep ensemble / conformal) and
this module supplies ``D`` and the controller that combines them through the ``PenaltyModel``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from chc.control import (
    Blocks,
    Bound,
    Duals,
    LinearConstraint,
    SolverResult,
    _backtrack,
    _constraint_blocks,
    _dykstra,
    _no_duals,
    _polytope_stationarity,
    _status,
    _violation,
    broadcast_box,
    check_box,
    project_box,
)
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
    support: SupportModel | None,
    lam_supp: float,
    u_lo: Array,
    u_hi: Array,
    steps: int,
    lr0: float,
    tol: float,
    uncertainty: PenaltyModel | None,
    lam_unc: float,
    blocks: Blocks = (),
) -> tuple[Array, Array, Array]:
    """The penalised descent as one XLA program, mirroring :func:`chc.control`'s shape exactly.

    Module level for the same reason: the jitted objective and its gradient used to be built inside
    :func:`pessimistic_control`, so each call got an empty compilation cache and recompiled the
    augmented gradient every solve instead of once per problem shape.

    Acceptance is on the *augmented* cost and the recorded history is the *task* cost, so both are
    carried through the loop -- runs at different penalty weights stay comparable. ``support`` may
    be ``None``: :func:`chc.plan.causal_plan` runs its barrier rounds through this descent with a
    penalty of their own and no support model.
    """

    def task(us: Array) -> Array:
        return total_cost(model, x0, us, dt, cost)

    def augmented(us: Array) -> Array:
        xs = rollout(model, x0, us, dt)
        penalty = 0.0 if support is None else lam_supp * support.penalty_trajectory(xs[:-1], us)
        if uncertainty is not None:
            penalty = penalty + lam_unc * uncertainty.penalty_trajectory(xs[:-1], us)
        return task(us) + penalty

    grad_aug = jax.grad(augmented)
    duals = _no_duals(us0.size, blocks, us0.dtype)
    if blocks:
        flat, duals = _dykstra(us0.ravel(), u_lo.ravel(), u_hi.ravel(), blocks, duals)
        us = flat.reshape(us0.shape)
    else:
        us = jnp.clip(us0, u_lo, u_hi)
    initial = augmented(us)
    values = jnp.zeros((steps + 1,), dtype=initial.dtype).at[0].set(task(us))

    def descending(carry: tuple[Array, Array, Array, Array, Array, Duals]) -> Array:
        taken, _, _, _, alive, _ = carry
        return jnp.logical_and(taken < steps, alive)

    def descend(
        carry: tuple[Array, Array, Array, Array, Array, Duals],
    ) -> tuple[Array, Array, Array, Array, Array, Duals]:
        taken, us, current, values, _, duals = carry
        us, current, accepted, duals = _backtrack(
            us, current, grad_aug(us), u_lo, u_hi, lr0, tol, augmented, blocks, duals
        )
        taken = jnp.where(accepted, taken + 1, taken)
        return taken, us, current, values.at[taken].set(task(us)), accepted, duals

    taken, optimised, _, values, _, _ = jax.lax.while_loop(
        descending, descend, (jnp.asarray(0), us, initial, values, jnp.asarray(True), duals)
    )
    return optimised, values, taken


def pessimistic_solve(
    model: Dynamics,
    x0: Array,
    us0: Array,
    dt: float,
    cost: QuadraticCost,
    support: SupportModel,
    lam_supp: float,
    u_lo: Bound,
    u_hi: Bound,
    steps: int = 10_000,
    lr0: float = 0.2,
    tol: float = 1e-9,
    uncertainty: PenaltyModel | None = None,
    lam_unc: float = 0.0,
    *,
    constraints: Sequence[LinearConstraint] = (),
) -> SolverResult:
    """:func:`pessimistic_control`, returning why it stopped as well as where.

    The stationarity residual is of the **augmented** objective, not the task cost: the penalties
    are what the descent actually minimised, so a residual measured on the task alone would be
    non-zero at the very point the solver was right to stop. With ``constraints`` it projects onto
    the box and the rows together, as :func:`chc.control.projected_gradient_solve` does.
    """
    lo = broadcast_box(u_lo, us0.shape, "u_lo", us0.dtype)
    hi = broadcast_box(u_hi, us0.shape, "u_hi", us0.dtype)
    check_box(lo, hi)
    blocks = _constraint_blocks(constraints, lo, hi, us0.dtype)
    optimised, values, taken = _pessimistic_loop(
        model,
        x0,
        us0,
        dt,
        cost,
        support,
        lam_supp,
        lo,
        hi,
        steps,
        lr0,
        tol,
        uncertainty,
        lam_unc,
        blocks,
    )
    iterations = int(taken)

    def augmented(us: Array) -> Array:
        xs = rollout(model, x0, us, dt)
        penalty = lam_supp * support.penalty_trajectory(xs[:-1], us)
        if uncertainty is not None:
            penalty = penalty + lam_unc * uncertainty.penalty_trajectory(xs[:-1], us)
        return total_cost(model, x0, us, dt, cost) + penalty

    gradient = jax.grad(augmented)(optimised)
    return SolverResult(
        actions=optimised,
        cost_history=jnp.asarray(np.asarray(values)[: iterations + 1].tolist()),
        status=_status(iterations, steps),
        iterations=iterations,
        stationarity=(
            _polytope_stationarity(optimised, gradient, lo, hi, blocks)
            if blocks
            else float(jnp.linalg.norm(optimised - project_box(optimised - gradient, lo, hi)))
        ),
        constraint_violation=_violation(optimised, constraints),
    )


def pessimistic_control(
    model: Dynamics,
    x0: Array,
    us0: Array,
    dt: float,
    cost: QuadraticCost,
    support: SupportModel,
    lam_supp: float,
    u_lo: Bound,
    u_hi: Bound,
    steps: int = 10_000,
    lr0: float = 0.2,
    tol: float = 1e-9,
    uncertainty: PenaltyModel | None = None,
    lam_unc: float = 0.0,
    *,
    constraints: Sequence[LinearConstraint] = (),
) -> tuple[Array, Array]:
    """Projected-gradient OC with an offline-safety penalty ``λ_supp·Σ D + λ_unc·Σ U``.

    ``support`` supplies the density-distance term ``D``; the optional ``uncertainty`` scorer gives
    the calibrated predictive-uncertainty term ``U`` (a ``chc.uncertainty`` ensemble/conformal).
    Uses autodiff for the augmented-objective gradient (validated equal to the discrete adjoint in
    ``01 §4.1``). Returns the optimised controls and the **task**-cost history (penalties excluded,
    so runs at different weights are comparable).

    ``u_lo`` / ``u_hi`` take the same scalar, per-lever or full-schedule forms, and
    ``constraints`` the same linear rows, that :func:`chc.control.projected_gradient_control`
    accepts.
    """
    lo = broadcast_box(u_lo, us0.shape, "u_lo", us0.dtype)
    hi = broadcast_box(u_hi, us0.shape, "u_hi", us0.dtype)
    check_box(lo, hi)
    blocks = _constraint_blocks(constraints, lo, hi, us0.dtype)
    optimised, values, taken = _pessimistic_loop(
        model,
        x0,
        us0,
        dt,
        cost,
        support,
        lam_supp,
        lo,
        hi,
        steps,
        lr0,
        tol,
        uncertainty,
        lam_unc,
        blocks,
    )
    return optimised, jnp.asarray(np.asarray(values)[: int(taken) + 1].tolist())
