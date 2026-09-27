"""Gradient-check gate: discrete adjoint == autodiff == finite differences."""

import jax
import jax.numpy as jnp
import pytest

from chc import (
    DampedOscillator,
    HybridDynamics,
    QuadraticCost,
    ZeroResidual,
    control_gradient_adjoint,
    rollout,
    total_cost,
)
from chc.adjoint import costate_norms, perturbation_cost_weights

DT = 0.1


def _setup() -> tuple[HybridDynamics, QuadraticCost, jax.Array, jax.Array]:
    dyn = HybridDynamics(
        known=DampedOscillator(omega=1.0, zeta=0.15), residual=ZeroResidual(out_dim=2)
    )
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.05])),
        R=jnp.array([[0.02]]),
        Qf=jnp.diag(jnp.array([5.0, 1.0])),
        x_target=jnp.zeros(2),
    )
    x0 = jnp.array([1.0, 0.0])
    us = 0.5 * jax.random.normal(jax.random.key(1), (30, 1))
    return dyn, cost, x0, us


def test_adjoint_matches_autodiff() -> None:
    dyn, cost, x0, us = _setup()
    g_adjoint = control_gradient_adjoint(dyn, x0, us, DT, cost)
    g_autodiff = jax.grad(lambda u: total_cost(dyn, x0, u, DT, cost))(us)
    assert jnp.allclose(g_adjoint, g_autodiff, atol=1e-8, rtol=1e-6)


def test_adjoint_matches_finite_difference() -> None:
    dyn, cost, x0, us = _setup()
    g_adjoint = control_gradient_adjoint(dyn, x0, us, DT, cost)
    eps = 1e-6
    for k in (0, 7, 15, 29):
        pert = jnp.zeros_like(us).at[k, 0].set(eps)
        fd = (
            total_cost(dyn, x0, us + pert, DT, cost) - total_cost(dyn, x0, us - pert, DT, cost)
        ) / (2 * eps)
        assert jnp.allclose(g_adjoint[k, 0], fd, atol=1e-5, rtol=1e-4)


def _ramp(cost: QuadraticCost, horizon: int) -> QuadraticCost:
    """The same cost, its position target ramping from 0 to 1 over the plan's states."""
    ramp = jnp.linspace(0.0, 1.0, horizon + 1)
    rows = jnp.stack([ramp, jnp.zeros_like(ramp)], axis=1)
    return QuadraticCost(Q=cost.Q, R=cost.R, Qf=cost.Qf, x_target=rows)


def test_a_target_per_state_is_the_bolza_sum_written_out() -> None:
    """Row ``k`` is ``x_k``'s target and the last row the terminal one, term by term."""
    dyn, cost, x0, us = _setup()
    moving = _ramp(cost, us.shape[0])
    xs = rollout(dyn, x0, us, DT)
    written = 0.0
    for k in range(us.shape[0]):
        dx = xs[k] - moving.x_target[k]
        written += 0.5 * dx @ cost.Q @ dx + 0.5 * us[k] @ cost.R @ us[k]
    dx = xs[-1] - moving.x_target[-1]
    written += 0.5 * dx @ cost.Qf @ dx
    assert abs(float(total_cost(dyn, x0, us, DT, moving)) - float(written)) < 1e-10


def test_adjoint_matches_autodiff_under_a_moving_target() -> None:
    dyn, cost, x0, us = _setup()
    moving = _ramp(cost, us.shape[0])
    g_adjoint = control_gradient_adjoint(dyn, x0, us, DT, moving)
    g_autodiff = jax.grad(lambda u: total_cost(dyn, x0, u, DT, moving))(us)
    assert jnp.allclose(g_adjoint, g_autodiff, atol=1e-8, rtol=1e-6)
    assert not jnp.allclose(g_adjoint, control_gradient_adjoint(dyn, x0, us, DT, cost), atol=1e-3)


def test_a_target_repeated_on_every_state_is_the_fixed_target() -> None:
    """Rows that all agree are the fixed target, to the bit, through every consumer of the cost."""
    dyn, cost, x0, us = _setup()
    rows = QuadraticCost(Q=cost.Q, R=cost.R, Qf=cost.Qf, x_target=cost.targets(us.shape[0]))
    assert rows.x_target.shape == (us.shape[0] + 1, 2)
    for consumer in (total_cost, control_gradient_adjoint, costate_norms):
        assert jnp.array_equal(consumer(dyn, x0, us, DT, rows), consumer(dyn, x0, us, DT, cost))
    assert jnp.array_equal(
        perturbation_cost_weights(dyn, x0, us, DT, rows, 0.1),
        perturbation_cost_weights(dyn, x0, us, DT, cost, 0.1),
    )


def test_a_target_per_state_needs_its_row_and_its_horizon() -> None:
    dyn, cost, x0, us = _setup()
    moving = _ramp(cost, us.shape[0])
    with pytest.raises(ValueError, match="pass the stage's row as target"):
        moving.running(x0, us[0])
    with pytest.raises(ValueError, match="31 rows, one per state, but a 20-step plan has 21"):
        total_cost(dyn, x0, us[:20], DT, moving)
