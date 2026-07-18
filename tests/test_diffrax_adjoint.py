"""Diffrax adjoint: an adaptive continuous-adjoint control gradient matching FD and RK4 adjoint."""

import diffrax
import jax
import jax.numpy as jnp

from chc.adjoint import control_gradient_adjoint, control_gradient_diffrax, total_cost_diffrax
from chc.cost import QuadraticCost, total_cost
from chc.dynamics import DampedOscillator, HybridDynamics
from chc.residual import ZeroResidual

DT = 0.1


def _setup() -> tuple[HybridDynamics, QuadraticCost, jnp.ndarray, jnp.ndarray]:
    dyn = HybridDynamics(known=DampedOscillator(omega=1.0, zeta=0.15), residual=ZeroResidual(2))
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.0])),
        R=jnp.array([[0.05]]),
        Qf=jnp.diag(jnp.array([5.0, 1.0])),
        x_target=jnp.array([0.0, 0.0]),
    )
    x0 = jnp.array([1.0, 0.0])
    us = 0.3 * jax.random.normal(jax.random.key(1), (30, 1))
    return dyn, cost, x0, us


def test_diffrax_cost_matches_the_fixed_step_rollout() -> None:
    dyn, cost, x0, us = _setup()
    diffrax_cost = float(total_cost_diffrax(dyn, x0, us, DT, cost))
    rk4_cost = float(total_cost(dyn, x0, us, DT, cost))
    assert abs(diffrax_cost - rk4_cost) < 1e-3  # adaptive solve and RK4 approximate the same cost


def test_diffrax_adjoint_matches_finite_difference() -> None:
    dyn, cost, x0, us = _setup()
    gradient = control_gradient_diffrax(dyn, x0, us, DT, cost)
    eps = 1e-6
    for k in (0, 7, 15, 29):
        pert = jnp.zeros_like(us).at[k, 0].set(eps)
        plus = total_cost_diffrax(dyn, x0, us + pert, DT, cost)
        minus = total_cost_diffrax(dyn, x0, us - pert, DT, cost)
        assert jnp.allclose(gradient[k, 0], (plus - minus) / (2 * eps), atol=1e-5, rtol=1e-4)


def test_continuous_backsolve_adjoint_matches_backprop_through_solver() -> None:
    dyn, cost, x0, us = _setup()
    backprop = control_gradient_diffrax(dyn, x0, us, DT, cost)
    continuous = control_gradient_diffrax(dyn, x0, us, DT, cost, adjoint=diffrax.BacksolveAdjoint())
    assert jnp.allclose(backprop, continuous, atol=1e-6)  # the two adjoint routes agree


def test_diffrax_adjoint_agrees_with_the_hand_written_rk4_adjoint() -> None:
    dyn, cost, x0, us = _setup()
    diffrax_grad = control_gradient_diffrax(dyn, x0, us, DT, cost)
    rk4_grad = control_gradient_adjoint(dyn, x0, us, DT, cost)
    assert jnp.allclose(diffrax_grad, rk4_grad, atol=1e-4)  # adaptive ~= fixed-step at tight rtol
