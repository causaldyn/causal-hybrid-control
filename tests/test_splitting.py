"""Splitting gate: Strang-Marchuk is 2nd-order; Lie-Trotter is 1st-order."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from chc.dynamics import HybridDynamics, LinearDynamics
from chc.integrate import rk4_step
from chc.splitting import exact_linear_flow, lie_trotter_step, residual_flow, strang_marchuk_step

A = jnp.array([[0.0, 1.0], [-1.0, -0.2]])
B = jnp.array([[0.0], [1.0]])
X0 = jnp.array([1.0, 0.0])
U = jnp.zeros(1)
T = 1.0


class CubicField(eqx.Module):
    """Autonomous nonlinear residual r(x) = [0, -beta x0^3]."""

    beta: float

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return jnp.array([0.0, -self.beta * x[0] ** 3])


def _reference() -> Array:
    """Near-exact solution of ẋ = A x + r(x) by very fine RK4."""
    true_sys = HybridDynamics(
        known=LinearDynamics(a_matrix=A, b_matrix=B), residual=CubicField(0.3)
    )

    def body(x: Array, _: None) -> tuple[Array, None]:
        return rk4_step(true_sys, 0.0, x, U, T / 8192), None

    x, _ = jax.lax.scan(body, X0, None, length=8192)
    return x


def _slope(step_fn) -> float:
    flow_a = exact_linear_flow(A)
    flow_b = residual_flow(CubicField(0.3), U)
    x_ref = _reference()
    n_grid = [8, 16, 32, 64]
    errors = []
    for n in n_grid:
        x = X0
        for _ in range(n):
            x = step_fn(flow_a, flow_b, x, T / n)
        errors.append(float(jnp.linalg.norm(x - x_ref)))
    dts = jnp.array([T / n for n in n_grid])
    return float(jnp.polyfit(jnp.log(dts), jnp.log(jnp.array(errors)), 1)[0])


def test_strang_marchuk_is_second_order() -> None:
    assert 1.7 < _slope(strang_marchuk_step) < 2.3


def test_lie_trotter_is_first_order() -> None:
    assert 0.7 < _slope(lie_trotter_step) < 1.3
