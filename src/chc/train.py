"""System identification: fit the learned residual to trajectory data (learn what physics misses).

Trains only the residual parameters (the known mechanism is frozen) to minimise one-step prediction
error, via autodiff through the RK4 step + Optax. This is the "hybrid learns the unknown part" step
(``plans/08`` Milestone B/D); the known dynamics are never re-learned.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from chc.dynamics import Dynamics, HybridDynamics
from chc.integrate import rk4_step


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

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(residual, eqx.is_inexact_array))

    @eqx.filter_jit
    def update(
        residual: Dynamics, opt_state: optax.OptState
    ) -> tuple[Dynamics, optax.OptState, Array]:
        loss, grads = loss_fn(residual)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(residual, eqx.is_inexact_array)
        )
        return eqx.apply_updates(residual, updates), opt_state, loss

    history = []
    for _ in range(steps):
        residual, opt_state, loss = update(residual, opt_state)
        history.append(float(loss))

    return HybridDynamics(known=known, residual=residual), jnp.asarray(history)
