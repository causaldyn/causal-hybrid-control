"""Delay gate: the linear-chain delay line is a plain Dynamics, and the whole stack runs on it."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chc import QuadraticCost, causal_plan, mpc_control, projected_gradient_control, rollout
from chc.adjoint import control_gradient_adjoint
from chc.cost import total_cost
from chc.delay import (
    DelayedDynamics,
    augment_state,
    delayed_of,
    exact_delayed_rollout,
    lift_cost,
    max_stages,
    stages_for_spread,
    state_of,
    state_trajectory,
)

_TAU, _DT = 1.0, 0.02
_LAG = round(_TAU / _DT)


def _core(t, x, x_delayed, u):
    """Scalar delayed loop: the control acts now, the state feeds back one delay ago."""
    return -0.8 * x_delayed + 0.5 * u


def _cost() -> QuadraticCost:
    return QuadraticCost(
        Q=jnp.eye(1), R=0.05 * jnp.eye(1), Qf=5.0 * jnp.eye(1), x_target=jnp.zeros(1)
    )


def test_the_chain_converges_to_an_exact_integer_lag() -> None:
    """More stages must mean less error, at the rate the Erlang variance predicts.

    The chain applies a delay of mean tau and variance tau^2/m, so the gap to a sharp lag should
    close like 1/sqrt(m) -- asserting the *rate* and not merely "it got smaller" is what makes this
    a test of the derivation rather than of monotonicity.
    """
    x0, us = jnp.array([1.0]), jnp.zeros((300, 1))
    exact = exact_delayed_rollout(_core, x0, us, _DT, _LAG)
    scaled = []
    previous = np.inf
    for stages in (5, 10, 20, 40):
        dyn = DelayedDynamics(_core, tau=_TAU, stages=stages, state_dim=1)
        zs = rollout(dyn, augment_state(x0, stages), us, _DT)
        gap = float(jnp.max(jnp.abs(state_trajectory(zs, 1) - exact)))
        assert gap < previous
        previous = gap
        scaled.append(gap * np.sqrt(stages))
    assert max(scaled) / min(scaled) < 2.0  # gap * sqrt(m) is bounded, i.e. the rate is 1/sqrt(m)


def test_the_discrete_adjoint_is_exact_on_the_augmented_plant() -> None:
    """The load-bearing claim: no gradient machinery needed changing for a delayed plant."""
    stages = 40
    dyn = DelayedDynamics(_core, tau=_TAU, stages=stages, state_dim=1)
    z0, us, cost = augment_state(jnp.array([1.0]), stages), jnp.zeros((60, 1)), _cost()
    lifted = lift_cost(cost, stages)

    analytic = control_gradient_adjoint(dyn, z0, us, _DT, lifted)
    autodiff = jax.grad(lambda u: total_cost(dyn, z0, u, _DT, lifted))(us)
    assert float(jnp.max(jnp.abs(analytic - autodiff))) < 1e-12

    eps = 1e-6
    base = float(total_cost(dyn, z0, us, _DT, lifted))
    for k in (0, 17, 59):
        bumped = us.at[k, 0].add(eps)
        finite = (float(total_cost(dyn, z0, bumped, _DT, lifted)) - base) / eps
        assert abs(float(analytic[k, 0]) - finite) < 1e-5


def test_every_solver_runs_unchanged_on_a_delayed_plant() -> None:
    """Augmentation earns its place only if nothing downstream needs a delayed variant."""
    stages = 30
    dyn = DelayedDynamics(_core, tau=_TAU, stages=stages, state_dim=1)
    z0, lifted = augment_state(jnp.array([1.0]), stages), lift_cost(_cost(), stages)

    _, history = projected_gradient_control(dyn, z0, jnp.zeros((60, 1)), _DT, lifted, -3.0, 3.0)
    assert float(history[-1]) < float(history[0])

    plan = causal_plan(dyn, z0, lifted, _DT, 60, -3.0, 3.0)
    assert plan.actions.shape == (60, 1)
    assert plan.trajectory.shape == (61, z0.shape[0])

    xs, us = mpc_control(dyn, z0, lifted, _DT, horizon=40, u_lo=-3.0, u_hi=3.0, n_steps=20)
    assert us.shape == (20, 1)
    assert abs(float(state_of(xs[-1], 1)[0])) < abs(float(state_of(xs[0], 1)[0]))


def test_the_buffer_really_lags_by_tau() -> None:
    """``delayed_of`` must track ``x`` shifted by tau, or the object is not a delay at all."""
    stages = 40
    dyn = DelayedDynamics(lambda t, x, xd, u: u, tau=_TAU, stages=stages, state_dim=1)
    x0 = jnp.array([0.0])
    ramp = jnp.ones((200, 1))  # x(t) = t exactly, so the delayed signal must read t - tau
    zs = rollout(dyn, augment_state(x0, stages), ramp, _DT)
    for step in (100, 150, 199):
        delayed = float(delayed_of(zs[step], 1)[0])
        assert abs(delayed - max(0.0, step * _DT - _TAU)) < 0.1


def test_max_stages_is_a_cliff_and_not_a_suggestion() -> None:
    """Inside the CFL cap the buffer is bounded; far outside it must actually blow up.

    A cap nothing ever violates is not evidence that the cap is real, so the far side is asserted
    as loudly as the near side. The failure is silent at first -- hence the long horizon.
    """
    cap = max_stages(_TAU, _DT)
    frozen = DelayedDynamics(lambda t, x, xd, u: jnp.zeros_like(x), _TAU, cap, 1)
    perturbed = augment_state(jnp.array([1.0]), cap) + 1e-8 * jax.random.normal(
        jax.random.key(0), (cap + 1,)
    )
    inside = rollout(frozen, perturbed, jnp.zeros((4000, 1)), _DT)
    assert float(jnp.max(jnp.abs(inside))) < 2.0

    over = 2 * cap
    outside_dyn = DelayedDynamics(lambda t, x, xd, u: jnp.zeros_like(x), _TAU, over, 1)
    outside = rollout(
        outside_dyn,
        augment_state(jnp.array([1.0]), over)
        + 1e-8 * jax.random.normal(jax.random.key(0), (over + 1,)),
        jnp.zeros((4000, 1)),
        _DT,
    )
    assert float(jnp.max(jnp.abs(outside))) > 1e3


def test_lift_cost_charges_nothing_for_the_buffer() -> None:
    """The buffer is bookkeeping; pricing it would price the plant's own history."""
    stages, cost = 6, _cost()
    x = jnp.array([1.7])
    lifted = lift_cost(cost, stages)
    z = augment_state(x, stages)
    u = jnp.array([0.3])
    assert abs(float(lifted.running(z, u)) - float(cost.running(x, u))) < 1e-12
    assert abs(float(lifted.terminal(z)) - float(cost.terminal(x))) < 1e-12
    # and a buffer far from the state changes nothing
    z_wild = z.at[1:].set(50.0)
    assert abs(float(lifted.terminal(z_wild)) - float(cost.terminal(x))) < 1e-12


def test_stages_for_spread_inverts_the_erlang_variance() -> None:
    assert stages_for_spread(0.1) == 100
    assert stages_for_spread(0.5) == 4
    for bad in (0.0, 1.0, -0.2, 3.0):
        with pytest.raises(ValueError, match="relative_spread"):
            stages_for_spread(bad)


def test_a_plant_with_no_delay_is_rejected_rather_than_silently_accepted() -> None:
    """``tau = 0`` is not a delay line with zero lag, it is a division by zero waiting to happen."""
    for tau in (0.0, -1.0):
        with pytest.raises(ValueError, match="tau must be positive"):
            DelayedDynamics(_core, tau=tau, stages=4, state_dim=1)
    with pytest.raises(ValueError, match="stages must be at least 1"):
        DelayedDynamics(_core, tau=1.0, stages=0, state_dim=1)
    with pytest.raises(ValueError, match="state_dim must be at least 1"):
        DelayedDynamics(_core, tau=1.0, stages=4, state_dim=0)
    with pytest.raises(ValueError, match="lag must be non-negative"):
        exact_delayed_rollout(_core, jnp.array([1.0]), jnp.zeros((3, 1)), _DT, lag=-1)
