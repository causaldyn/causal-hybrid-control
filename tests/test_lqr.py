"""LQR / AKOR gates: cost-to-go identity, PG-OC vs the LQ optimum, CARE residual + stability."""

import jax.numpy as jnp

from chc import (
    DampedOscillator,
    HybridDynamics,
    QuadraticCost,
    ZeroResidual,
    continuous_lqr,
    dlqr_feedback_controls,
    finite_horizon_dlqr,
    linearize_continuous,
    linearize_discrete,
    projected_gradient_control,
    total_cost,
)

DT = 0.1


def _dyn() -> HybridDynamics:
    return HybridDynamics(
        known=DampedOscillator(omega=1.0, zeta=0.1), residual=ZeroResidual(out_dim=2)
    )


def _weights() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return jnp.diag(jnp.array([1.0, 0.05])), jnp.array([[0.1]]), jnp.diag(jnp.array([2.0, 1.0]))


def test_dlqr_cost_identity() -> None:
    """Rolling out the LQR feedback costs exactly the cost-to-go 0.5 x0ᵀ P0 x0."""
    dyn = _dyn()
    q, r, qf = _weights()
    x0 = jnp.array([1.0, 0.5])
    horizon = 40
    a_d, b_d = linearize_discrete(dyn, jnp.zeros(2), jnp.zeros(1), DT)
    gains, p0 = finite_horizon_dlqr(a_d, b_d, q, r, qf, horizon)
    us = dlqr_feedback_controls(dyn, x0, gains, DT)
    cost = QuadraticCost(Q=q, R=r, Qf=qf, x_target=jnp.zeros(2))
    j = total_cost(dyn, x0, us, DT, cost)
    j_star = 0.5 * x0 @ p0 @ x0
    assert jnp.allclose(j, j_star, rtol=1e-6, atol=1e-6)


def test_projected_gradient_reaches_lqr_optimum() -> None:
    """Unconstrained projected-gradient OC cannot beat the LQ optimum and converges close to it."""
    dyn = _dyn()
    q, r, qf = _weights()
    x0 = jnp.array([1.0, 0.5])
    horizon = 40
    a_d, b_d = linearize_discrete(dyn, jnp.zeros(2), jnp.zeros(1), DT)
    _, p0 = finite_horizon_dlqr(a_d, b_d, q, r, qf, horizon)
    j_star = float(0.5 * x0 @ p0 @ x0)
    cost = QuadraticCost(Q=q, R=r, Qf=qf, x_target=jnp.zeros(2))
    us0 = jnp.zeros((horizon, 1))
    _, history = projected_gradient_control(dyn, x0, us0, DT, cost, u_lo=-1e3, u_hi=1e3, steps=500)
    assert float(history[-1]) + 1e-6 >= j_star  # LQR is the optimum
    assert float(history[-1]) <= 1.10 * j_star  # PG converges close to it


def test_continuous_care_residual_and_stability() -> None:
    dyn = _dyn()
    a, b = linearize_continuous(dyn, jnp.zeros(2), jnp.zeros(1))
    q = jnp.eye(2)
    r = jnp.array([[1.0]])
    p, k = continuous_lqr(a, b, q, r)
    residual = a.T @ p + p @ a - p @ b @ jnp.linalg.solve(r, b.T @ p) + q
    assert jnp.allclose(residual, jnp.zeros((2, 2)), atol=1e-8)
    closed_loop_eigs = jnp.linalg.eigvals(a - b @ k)
    assert bool((closed_loop_eigs.real < 0).all())
