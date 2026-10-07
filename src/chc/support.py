"""Support / pessimism: keep offline-trained control inside the region the data justifies.

``SupportModel`` scores how far a state-action pair ``(x, u)`` sits from the offline data cloud
(squared Mahalanobis distance ``D``); ``pessimistic_control`` penalises leaving that support, so the
controller does not exploit the model where it was never trained. This is the offline-safety layer:
the objective is ``J_task + λ_unc·Σ U + λ_supp·Σ D``, where the calibrated
predictive-uncertainty term ``U`` comes from ``chc.uncertainty`` (deep ensemble / conformal) and
this module supplies ``D`` and the controller that combines them through the ``PenaltyModel``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from chc import _units
from chc.control import (
    Blocks,
    Bound,
    Duals,
    LinearConstraint,
    SolverResult,
    _backtrack,
    _constraint_blocks,
    _dykstra,
    _history,
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
from chc.frames import _real_numbers
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
        """The log's mean and the inverse of its covariance, each variance raised by ``ridge`` of
        itself.

        Raising each variance by a share of itself shrinks the correlations and counts the distance
        in each coordinate's own standard deviations: a point ``k`` of them off the log, along a
        coordinate the log moved apart from the others, scores about ``k**2 / (1 + ridge)`` in any
        units and from any origin, and coordinates the log moved in step keep a finite distance
        across their line. A ridge added in the caller's units outweighed a coordinate logged in
        small ones: at 1e-3 of its units, a point 4 standard deviations off scored 0.016 where it
        scored 16.0.

        Raises:
            ValueError: on a coordinate the log never moved, or moved by rounding alone (a spread
                about its mean of at most 64 eps of its root mean square, eps its precision's),
                whose distance is unbounded in any units: a finite stand-in would be set by the
                units it was logged in, or by its rounding. And on a coordinate whose spread the
                precision cannot hold in the caller's units, where the inverse of its square
                overflows or underflows the precision's dtype and every point would read as off
                the support, or on it.
        """
        xs, us = _real_numbers(xs, "xs"), _real_numbers(us, "us")
        z = jnp.concatenate([xs, us], axis=1)
        n = xs.shape[1]
        # each column as a share of its largest entry, so that no square overflows or underflows,
        # centred twice (chc._units)
        peak = jnp.max(jnp.abs(z), axis=0)
        columns = _units.centred(z / jnp.where(peak == 0.0, 1.0, peak))
        held = [k for k, still in enumerate(columns.rounding.tolist()) if still]
        if held:
            names = ", ".join(_label(k, n) for k in held)
            raise ValueError(
                f"the log never moved {names}: every row holds one value, up to rounding (a "
                "spread about the mean of at most 64 eps of its size), so a deviation along it has "
                "no logged spread to be measured in, and any finite distance would be set by the "
                "units it was logged in; leave it out of the problem, or build "
                "SupportModel(mean, precision) with the spread it should be measured in"
            )
        centred = columns.deviations
        cov = (centred.T @ centred) / z.shape[0]
        spread = jnp.sqrt(jnp.diag(cov))
        inner = jnp.linalg.inv(cov / jnp.outer(spread, spread) + ridge * jnp.eye(z.shape[1]))
        # back in the caller's units one coordinate at a time, so that an entry leaves the range
        # only where the precision itself cannot hold it
        spread = peak * spread
        precision = inner / spread[:, None] / spread[None, :]
        holds = jnp.all(jnp.isfinite(precision), axis=1) & (jnp.diagonal(precision) > 0.0)
        outside = [k for k, ok in enumerate(holds.tolist()) if not ok]
        if outside:
            names = ", ".join(f"{_label(k, n)} (spread {float(spread[k]):.3g})" for k in outside)
            raise ValueError(
                f"the log's spread along {names} is outside what the precision can hold in these "
                f"units: the inverse of its square overflows or underflows {precision.dtype}, so "
                "every point would read as off the support, or on it; rescale the column, and the "
                "points scored against it, nearer a spread of 1, and every distance stays the same"
            )
        return cls(mean=jnp.mean(z, axis=0), precision=precision)

    def squared_distance(self, x: Array, u: Array) -> Array:
        d = jnp.concatenate([x, u]) - self.mean
        return d @ self.precision @ d

    def penalty_trajectory(self, xs: Array, us: Array) -> Array:
        """Total off-support penalty over the visited (x, u) pairs (xs: (H,n), us: (H,m))."""
        return jnp.sum(jax.vmap(self.squared_distance)(xs, us))


def _label(column: int, states: int) -> str:
    return f"state {column}" if column < states else f"action {column - states}"


def _augmented(
    model: Dynamics,
    x0: Array,
    us: Array,
    dt: float,
    cost: QuadraticCost,
    support: SupportModel | None,
    lam_supp: float,
    uncertainty: PenaltyModel | None,
    lam_unc: float,
) -> Array:
    """The task cost plus the weighted penalties: what the pessimistic descent minimises."""
    xs = rollout(model, x0, us, dt)
    penalty = 0.0 if support is None else lam_supp * support.penalty_trajectory(xs[:-1], us)
    if uncertainty is not None:
        penalty = penalty + lam_unc * uncertainty.penalty_trajectory(xs[:-1], us)
    return total_cost(model, x0, us, dt, cost) + penalty


# Compiled once at module level: a gradient of a closure built per solve re-traced the rollout's
# scan and compiled it afresh on every call.
_augmented_gradient = eqx.filter_jit(jax.grad(_augmented, argnums=2))


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
        return _augmented(model, x0, us, dt, cost, support, lam_supp, uncertainty, lam_unc)

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
    gradient = _augmented_gradient(
        model, x0, optimised, dt, cost, support, lam_supp, uncertainty, lam_unc
    )
    return SolverResult(
        actions=optimised,
        cost_history=_history(values, iterations),
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
    return optimised, _history(values, int(taken))
