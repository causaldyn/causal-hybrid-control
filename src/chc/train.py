"""System identification: fit the learned residual to trajectory data (learn what physics misses).

Trains only the residual parameters (the known mechanism is frozen) to minimise one-step or
multi-step (rollout) prediction error via autodiff through the RK4 step + Optax. This is the "hybrid
learns the unknown part" step (``plans/08`` Milestone B/D); the known dynamics are never re-learned.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from chc.dynamics import Dynamics, HybridDynamics
from chc.integrate import rk4_step, rollout


def _train(
    known: Dynamics,
    residual: Dynamics,
    loss_and_grad: Callable[[Dynamics], tuple[Array, Any]],
    steps: int,
    lr: float,
) -> tuple[HybridDynamics, Array]:
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(residual, eqx.is_inexact_array))

    @eqx.filter_jit
    def update(
        residual: Dynamics, opt_state: optax.OptState
    ) -> tuple[Dynamics, optax.OptState, Array]:
        loss, grads = loss_and_grad(residual)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(residual, eqx.is_inexact_array)
        )
        return eqx.apply_updates(residual, updates), opt_state, loss

    history = []
    for _ in range(steps):
        residual, opt_state, loss = update(residual, opt_state)
        history.append(float(loss))
    return HybridDynamics(known=known, residual=residual), jnp.asarray(history)


def one_step_mse(model: Dynamics, xs: Array, us: Array, x_next: Array, dt: float) -> Array:
    """Mean squared one-step prediction error of ``model`` over a batch of transitions."""
    pred = jax.vmap(lambda x, u: rk4_step(model, 0.0, x, u, dt))(xs, us)
    return jnp.mean((pred - x_next) ** 2)


def fit_residual(
    model: HybridDynamics,
    data: dict[str, Array],
    dt: float,
    steps: int = 2000,
    lr: float = 1e-2,
) -> tuple[HybridDynamics, Array]:
    """Fit the residual of ``model`` to one-step transitions ``data = {x, u, x_next}``.

    Returns the trained model (same frozen ``known``, updated ``residual``) and the loss history.
    """
    known = model.known
    residual = model.residual
    xs, us, x_next = data["x"], data["u"], data["x_next"]

    @eqx.filter_value_and_grad
    def loss_fn(residual: Dynamics) -> Array:
        return one_step_mse(HybridDynamics(known=known, residual=residual), xs, us, x_next, dt)

    return _train(known, residual, loss_fn, steps, lr)


def rollout_mse(model: Dynamics, x0s: Array, us: Array, xs: Array, dt: float) -> Array:
    """Mean squared multi-step rollout error over a batch of trajectories.

    ``x0s`` (N, n), ``us`` (N, H, m), ``xs`` (N, H+1, n): realised states incl. the initial one.
    """
    preds = jax.vmap(lambda x0, u_seq: rollout(model, x0, u_seq, dt))(x0s, us)
    return jnp.mean((preds - xs) ** 2)


def fit_residual_multistep(
    model: HybridDynamics,
    data: dict[str, Array],
    dt: float,
    steps: int = 1500,
    lr: float = 1e-2,
) -> tuple[HybridDynamics, Array]:
    """Fit the residual to *trajectories* by minimising multi-step rollout error.

    ``data = {x0, us, xs}`` with shapes (N, n), (N, H, m), (N, H+1, n). Unlike one-step fitting this
    directly penalises rollout drift (``plans/08``). Returns the trained model and loss history.
    """
    known = model.known
    residual = model.residual
    x0s, us, xs = data["x0"], data["us"], data["xs"]

    @eqx.filter_value_and_grad
    def loss_fn(residual: Dynamics) -> Array:
        return rollout_mse(HybridDynamics(known=known, residual=residual), x0s, us, xs, dt)

    return _train(known, residual, loss_fn, steps, lr)
