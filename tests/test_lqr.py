"""LQR / AKOR gates: cost-to-go identity, PG-OC vs the LQ optimum, CARE residual + stability."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import solve_continuous_are

from chc import HybridDynamics, QuadraticCost
from chc.control import projected_gradient_control
from chc.cost import total_cost
from chc.dynamics import DampedOscillator
from chc.lqr import (
    continuous_lqr,
    dlqr_feedback_controls,
    finite_horizon_dlqr,
    linearize_continuous,
    linearize_discrete,
)
from chc.residual import ZeroResidual

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


def _two_levers() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = np.array([[0.0, 1.0, 0.0], [-2.0, -0.3, 0.5], [0.1, 0.0, -1.0]])
    b = np.array([[0.3, -1.2], [1.0, 0.4], [-0.5, 0.8]])
    return a, b, np.diag([10.0, 1.0, 0.5]), np.array([[0.1, 0.02], [0.02, 0.3]])


@pytest.mark.parametrize("units", [(2.0**-40, 2.0**20), (1e-12, 1.0), (1e-9, 1e6), (1e12, 1e-3)])
def test_continuous_lqr_reads_one_gain_in_any_units_of_each_action(
    units: tuple[float, float],
) -> None:
    """Each lever in units `s` divides its column of B by `s` and R by `s s'`: one problem.

    SciPy's CARE reads B and R apart: at a billionth of the units it read a gain off by 2e-5, and
    at 1e-12 it refused the problem as too close to the imaginary axis. In powers of two the
    problem is now the same to the bit.
    """
    a, b, q, r = _two_levers()
    p, k = (np.asarray(m) for m in continuous_lqr(a, b, q, r))
    s = np.array(units)
    p_units, k_units = (np.asarray(m) for m in continuous_lqr(a, b / s, q, r / np.outer(s, s)))
    if np.all(np.log2(s) == np.round(np.log2(s))):
        np.testing.assert_array_equal(k_units / s[:, None], k)
        np.testing.assert_array_equal(p_units, p)
    else:
        assert np.max(np.abs(k_units / s[:, None] - k)) <= 1e-12 * np.max(np.abs(k))
        assert np.max(np.abs(p_units - p)) <= 1e-12 * np.max(np.abs(p))


@pytest.mark.parametrize("cost_units", [1e-12, 1e12])
def test_continuous_lqr_reads_one_gain_in_any_units_of_the_cost(cost_units: float) -> None:
    """Q and R in other units of the cost give the same gain, and P in those units."""
    a, b, q, r = _two_levers()
    p, k = (np.asarray(m) for m in continuous_lqr(a, b, q, r))
    p_units, k_units = (np.asarray(m) for m in continuous_lqr(a, b, cost_units * q, cost_units * r))
    assert np.max(np.abs(k_units - k)) <= 1e-12 * np.max(np.abs(k))
    assert np.max(np.abs(p_units / cost_units - p)) <= 1e-12 * np.max(np.abs(p))


def test_continuous_lqr_solves_an_r_with_a_zero_entry_as_scipy_does() -> None:
    """No unit rescales a zero cost, so that action keeps its units, and SciPy's solve."""
    a, b, q, _ = _two_levers()
    r = np.array([[0.0, 0.5], [0.5, 1.0]])
    p, _ = continuous_lqr(a, b, q, r)
    np.testing.assert_array_equal(np.asarray(p), solve_continuous_are(a, b, q, r))


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
