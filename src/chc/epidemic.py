"""Scientific flagship: SIR epidemic control — flatten the curve under a capacity constraint.

Nonlinear known dynamics (compartmental SIR) with an NPI control ``u`` that scales transmission
``beta -> beta*(1-u)``. Optimal control keeps infections under a hospital-capacity threshold with
minimal intervention — constrained OC on a classic nonlinear population model (the Bazykin /
Riznichenko / Marchuk-immunology lineage). In observational logs the intervention effect is
confounded (policy reacts to case counts); here the plant is the true system and control is planned
against it.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from chc.integrate import rollout


class SIRDynamics(eqx.Module):
    """Normalised SIR; state ``[S, I]``, control ``u in [0,1]`` scales ``beta -> beta*(1-u)``."""

    beta: float
    gamma: float

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        s, i = x[0], x[1]
        force = self.beta * (1.0 - u[0]) * s * i
        return jnp.stack([-force, force - self.gamma * i])


@eqx.filter_jit
def epidemic_cost(
    model: SIRDynamics,
    x0: Array,
    us: Array,
    dt: float,
    i_max: float,
    w_npi: float,
    w_peak: float,
) -> Array:
    """Intervention effort plus a squared penalty for exceeding the capacity ``i_max``."""
    infections = rollout(model, x0, us, dt)[:, 1]
    over_capacity = jnp.sum(jnp.maximum(infections - i_max, 0.0) ** 2)
    intervention = jnp.sum(us[:, 0] ** 2)
    return w_peak * over_capacity + w_npi * intervention


@eqx.filter_jit
def _epidemic_cost_grad(
    model: SIRDynamics,
    x0: Array,
    us: Array,
    dt: float,
    i_max: float,
    w_npi: float,
    w_peak: float,
) -> Array:
    """``d/dus`` of :func:`epidemic_cost`, jitted at module level so the cache outlives the call."""
    return jax.grad(epidemic_cost, argnums=2)(model, x0, us, dt, i_max, w_npi, w_peak)


def optimal_npi(
    model: SIRDynamics,
    x0: Array,
    dt: float,
    horizon: int,
    i_max: float,
    u_max: float = 0.9,
    w_npi: float = 1.0,
    w_peak: float = 1.0e4,
    steps: int = 400,
    lr0: float = 0.5,
) -> Array:
    """Open-loop optimal NPI: least intervention that keeps ``I <= i_max`` (projected gradient)."""

    def obj(us: Array) -> Array:
        return epidemic_cost(model, x0, us, dt, i_max, w_npi, w_peak)

    def grad_fn(us: Array) -> Array:
        return _epidemic_cost_grad(model, x0, us, dt, i_max, w_npi, w_peak)

    us = jnp.zeros((horizon, 1))
    current = obj(us)
    for _ in range(steps):
        grad = grad_fn(us)
        lr = lr0
        improved = False
        candidate = us
        for _ in range(30):
            candidate = jnp.clip(us - lr * grad, 0.0, u_max)
            candidate_cost = obj(candidate)
            if candidate_cost < current - 1e-12:
                improved = True
                break
            lr *= 0.5
        if not improved:
            break
        us, current = candidate, candidate_cost
    return us
