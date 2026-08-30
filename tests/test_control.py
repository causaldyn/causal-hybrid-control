"""Optimal-control gate: projected gradient reduces cost and drives the state toward target."""

import jax.numpy as jnp

from chc import (
    DampedOscillator,
    HybridDynamics,
    QuadraticCost,
    ZeroResidual,
    lbfgs_box_control,
    nlp_solver_certificate,
    projected_gradient_control,
    rollout,
)

DT = 0.1


def test_projected_gradient_reduces_cost_and_reaches_target() -> None:
    dyn = HybridDynamics(
        known=DampedOscillator(omega=1.0, zeta=0.1), residual=ZeroResidual(out_dim=2)
    )
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.0])),
        R=jnp.array([[0.01]]),
        Qf=jnp.diag(jnp.array([10.0, 1.0])),
        x_target=jnp.zeros(2),
    )
    x0 = jnp.array([1.0, 0.0])
    us0 = jnp.zeros((50, 1))

    us, history = projected_gradient_control(dyn, x0, us0, DT, cost, u_lo=-5.0, u_hi=5.0, steps=150)

    assert history[-1] < 0.7 * history[0]  # meaningful cost reduction
    assert bool((jnp.abs(us) <= 5.0 + 1e-6).all())  # respects box constraints
    xs = rollout(dyn, x0, us, DT)
    assert abs(float(xs[-1, 0])) < abs(float(x0[0]))  # ends closer to target position


def test_lbfgs_matches_the_projected_gradient_objective_and_beats_it() -> None:
    certificate = nlp_solver_certificate()
    assert certificate.ok
    assert certificate.box_respected
    assert certificate.worst_lbfgs_stationarity < 1e-3
    # Both directions: the first-order budget is short where the problem is ill conditioned and
    # adequate where it is not, so the gap is a statement about conditioning, not about the plant.
    assert certificate.worst_relative_gap > 0.05
    assert certificate.best_relative_gap < 0.005


def test_lbfgs_box_control_preserves_the_caller_dtype() -> None:
    # SciPy is a float64 boundary; a float32 caller must not be silently promoted on the way back.
    dyn = HybridDynamics(
        known=DampedOscillator(omega=1.0, zeta=0.1), residual=ZeroResidual(out_dim=2)
    )
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.0])),
        R=jnp.array([[0.01]]),
        Qf=jnp.diag(jnp.array([10.0, 1.0])),
        x_target=jnp.zeros(2),
    )
    us0 = jnp.zeros((20, 1), dtype=jnp.float32)
    us, history = lbfgs_box_control(dyn, jnp.array([1.0, 0.0]), us0, DT, cost, -5.0, 5.0)
    assert us.dtype == us0.dtype
    assert float(history[-1]) < float(history[0])
    assert bool((jnp.abs(us) <= 5.0 + 1e-6).all())
