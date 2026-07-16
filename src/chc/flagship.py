"""Flagship: causal vs predictive control under confounding — the sign-flip catastrophe.

Offline data is logged under a behaviour policy that ties the action to a confounder ``z`` which
also drives the outcome. A controller that fits the action effect **without** adjusting for ``z``
learns the wrong sign and drives the true plant away from target; adjusting for ``z`` recovers the
effect and control succeeds. This is the "causal != predictive for control" figure (``plans/05``):
plan with the learned effect, act on the true system — the model/plant split of ``chc.mpc``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from chc.causal import ConfoundedLinearSystem, estimate_control_effect


def certainty_equivalent_control(
    a: float, b_hat: float, x: Array, x_target: float, u_lo: float, u_hi: float
) -> Array:
    """One-step control to reach ``x_target`` under the believed dynamics ``x' = a x + b_hat u``."""
    return jnp.clip((x_target - a * x) / b_hat, u_lo, u_hi)


def closed_loop(
    system: ConfoundedLinearSystem,
    b_hat: float,
    x0: Array,
    x_target: float,
    n_steps: int,
    u_lo: float,
    u_hi: float,
    key: Array,
) -> tuple[Array, Array]:
    """Roll the TRUE plant under the certainty-equivalent controller that believes ``b_hat``."""

    def step(x: Array, k: Array) -> tuple[Array, tuple[Array, Array]]:
        u = certainty_equivalent_control(system.a, b_hat, x, x_target, u_lo, u_hi)
        noise = system.noise_scale * jax.random.normal(k, ())
        x_next = (
            system.a * x + system.b_true * u + noise
        )  # true interventional plant (z not acting)
        return x_next, (x, u)

    x_last, (xs, us) = jax.lax.scan(step, x0, jax.random.split(key, n_steps))
    return jnp.append(xs, x_last), us


def run_flagship(
    *,
    n_data: int = 20_000,
    x0: float = 0.0,
    x_target: float = 2.0,
    n_steps: int = 30,
    u_lo: float = -10.0,
    u_hi: float = 10.0,
    seed_data: int = 0,
    seed_run: int = 1,
) -> dict[str, Any]:
    """Estimate the effect two ways from confounded logs, then control the true plant with each."""
    system = ConfoundedLinearSystem()
    data = system.sample(n_data, jax.random.key(seed_data))
    b_naive = float(estimate_control_effect(data, adjust_for=()))
    b_causal = float(estimate_control_effect(data, adjust_for=("z",)))
    run = jax.random.key(seed_run)
    xs_naive, us_naive = closed_loop(
        system, b_naive, jnp.asarray(x0), x_target, n_steps, u_lo, u_hi, run
    )
    xs_causal, us_causal = closed_loop(
        system, b_causal, jnp.asarray(x0), x_target, n_steps, u_lo, u_hi, run
    )
    return {
        "b_true": system.b_true,
        "b_naive": b_naive,
        "b_causal": b_causal,
        "x_target": x_target,
        "xs_naive": xs_naive,
        "xs_causal": xs_causal,
        "us_naive": us_naive,
        "us_causal": us_causal,
    }
