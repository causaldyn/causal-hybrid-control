"""The expert path: a hybrid model fitted by hand, then controlled, without `prescribe`.

Run: uv run python docs/expert_path.py
"""

from __future__ import annotations

# --8<-- [start:fit]
import equinox as eqx
import jax
import jax.numpy as jnp

from chc import BarrierConstraint, HybridDynamics, QuadraticCost, RecedingHorizon
from chc.dynamics import DampedOscillator
from chc.integrate import rk4_step
from chc.mpc import mpc_control
from chc.residual import KANResidual
from chc.train import fit_residual

DT = 0.1


class Stiffening(eqx.Module):
    """The physics the known model leaves out: a spring that stiffens as it stretches."""

    def __call__(self, t, x, u):
        return jnp.array([0.0, -0.5 * x[0] ** 3])


known = DampedOscillator(omega=1.0, zeta=0.1)
plant = HybridDynamics(known=known, residual=Stiffening())  # the system itself, unknown to us

# logged transitions (x, u, x_next) from the plant, under actions drawn at random
k_x, k_u, k_r = jax.random.split(jax.random.key(0), 3)
xs = jax.random.normal(k_x, (2000, 2))
us = 0.5 * jax.random.normal(k_u, (2000, 1))
x_next = jax.vmap(lambda x, u: rk4_step(plant, 0.0, x, u, DT))(xs, us)

# the known oscillator plus a learnable (KAN) residual, swappable for an MLP or a linear term
model = HybridDynamics(known=known, residual=KANResidual(2, 1, 2, grid_range=4.0, key=k_r))
model, _ = fit_residual(model, {"x": xs, "u": us, "x_next": x_next}, DT, steps=2000)
# --8<-- [end:fit]

# --8<-- [start:mpc]
cost = QuadraticCost(
    Q=jnp.diag(jnp.array([1.0, 0.1])),
    R=jnp.array([[0.05]]),
    Qf=jnp.diag(jnp.array([5.0, 1.0])),
    x_target=jnp.zeros(2),
)
states, actions = mpc_control(
    model,
    jnp.array([1.0, 0.0]),
    cost,
    dt=DT,
    horizon=20,
    u_lo=-5.0,
    u_hi=5.0,
    n_steps=40,
    plant=plant,
)  # plans on the fitted model, acts on the plant; u_lo/u_hi also take one bound per lever
# --8<-- [end:mpc]
print(
    f"mpc_control: {actions.shape[0]} steps on the plant, from x = (1, 0) to "
    f"x = ({float(states[-1, 0]):+.3f}, {float(states[-1, 1]):+.3f})"
)

# --8<-- [start:receding]
floor = BarrierConstraint(lambda x: x[1] + 0.8, alpha=2.0)  # velocity >= -0.8; built once
controller = RecedingHorizon(model, cost, dt=DT, horizon=20, u_lo=-5.0, u_hi=5.0, barrier=floor)

x = jnp.array([1.0, 0.0])  # the state just measured
plan = controller.step(x)  # every step: read plan.safety, then apply plan.actions[0]
# --8<-- [end:receding]
print(
    f"RecedingHorizon.step: first action {float(plan.actions[0, 0]):+.3f}, "
    f"barrier certified on {plan.safety.certified_steps} of {plan.actions.shape[0]} steps"
)
