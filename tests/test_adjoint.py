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
from chc.integrate import rk4_step

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


def _forced(t: jax.Array, x: jax.Array, u: jax.Array) -> jax.Array:
    """A scalar plant whose forcing and whose decay both move in time."""
    return (-0.5 + 0.3 * t) * x + u + jnp.sin(3.0 * t)


def _forced_setup() -> tuple[QuadraticCost, jax.Array, jax.Array]:
    cost = QuadraticCost(Q=jnp.eye(1), R=0.1 * jnp.eye(1), Qf=jnp.eye(1), x_target=jnp.ones(1))
    return cost, jnp.array([0.2]), 0.3 * jax.random.normal(jax.random.key(0), (20, 1))


def test_the_adjoint_steps_the_clock_the_rollout_steps() -> None:
    """On a plant that moves in time each step's Jacobian is taken at that step's time."""
    cost, x0, us = _forced_setup()
    g_adjoint = control_gradient_adjoint(_forced, x0, us, DT, cost)
    g_autodiff = jax.grad(lambda u: total_cost(_forced, x0, u, DT, cost))(us)
    assert jnp.allclose(g_adjoint, g_autodiff, atol=1e-8, rtol=1e-6)


def test_the_costates_are_cost_to_go_gradients_on_a_plant_that_moves_in_time() -> None:
    cost, x0, us = _forced_setup()
    xs = rollout(_forced, x0, us, DT)
    horizon = us.shape[0]

    def cost_to_go(x: jax.Array, k: int) -> jax.Array:
        """``J`` from state ``x`` entering step ``k``, on the actions that remain."""
        tail = rollout(_forced, x, us[k:], DT, t0=k * DT)
        running = sum(cost.running(tail[j], us[k + j]) for j in range(horizon - k))
        return running + cost.terminal(tail[-1])

    norms = costate_norms(_forced, x0, us, DT, cost)
    for k in (0, 7, horizon - 1):
        expected = jnp.linalg.norm(jax.grad(cost_to_go)(xs[k + 1], k + 1))
        assert abs(float(norms[k]) - float(expected)) < 1e-8


def test_the_perturbation_weights_read_each_step_at_its_own_time() -> None:
    """Both terms, written out on a scalar plant: the injection gain and the tube's transition."""
    cost, x0, us = _forced_setup()
    xs = rollout(_forced, x0, us, DT)
    times = DT * jnp.arange(us.shape[0])
    step = DT * jax.vmap(jax.jacobian(_forced, argnums=1))(times, xs[:-1], us)[:, 0, 0]
    gains = DT * jnp.abs(1.0 + step / 2.0 + step**2 / 6.0 + step**3 / 24.0)
    first = costate_norms(_forced, x0, us, DT, cost) * gains
    at_zero = perturbation_cost_weights(_forced, x0, us, DT, cost, 0.0)
    assert jnp.allclose(at_zero, first, rtol=1e-12)

    radius = 0.1

    def one_step(t: jax.Array, x: jax.Array, u: jax.Array) -> jax.Array:
        return rk4_step(_forced, t, x, u, DT)

    transition = jax.vmap(jax.jacobian(one_step, argnums=1))(times, xs[:-1], us)[:, 0, 0]
    rho, tube = 0.0, []
    for k in range(us.shape[0]):
        rho = abs(float(transition[k])) * rho + radius * abs(float(us[k, 0])) * float(gains[k])
        tube.append(rho)
    second = 0.5 * (sum(r**2 for r in tube[:-1]) + tube[-1] ** 2)  # Q = Qf = 1
    spread = second / (radius * float(jnp.sum(jnp.abs(us))))
    weights = perturbation_cost_weights(_forced, x0, us, DT, cost, radius)
    assert jnp.allclose(weights, first + spread, rtol=1e-10)
