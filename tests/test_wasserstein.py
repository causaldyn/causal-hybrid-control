"""Wasserstein-1 DRO penalty: linear in the radius, equals radius*Lipschitz, upper-bounds the
worst-case residual shift (the W-DRO certificate), and steers closed-loop control off the fragile
(high-gradient) region of the learned residual.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from chc.cost import QuadraticCost
from chc.dynamics import DampedOscillator, HybridDynamics
from chc.integrate import rollout
from chc.support import SupportModel, pessimistic_control
from chc.uncertainty import WassersteinPenalty

DT = 0.1
H = 25


class _LinearResidual(eqx.Module):
    a: Array  # (out, n); dr/dx = a everywhere, so the Lipschitz constant is ||a||_F

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return self.a @ x


class _TanhResidual(eqx.Module):
    """A learned correction with a steep, fragile feature at ``x[0] = 0`` (max ||dr/dx|| there)."""

    amp: float
    width: float

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return jnp.array([self.amp * jnp.tanh(x[0] / self.width), 0.0])


def test_penalty_is_linear_in_the_radius() -> None:
    res = _LinearResidual(a=jnp.array([[1.0, 0.3]]))
    xs = jax.random.normal(jax.random.key(1), (4, 2))
    us = jnp.zeros((4, 1))
    p1 = WassersteinPenalty(residual=res, radius=0.2).penalty_trajectory(xs, us)
    p2 = WassersteinPenalty(residual=res, radius=0.4).penalty_trajectory(xs, us)
    assert jnp.allclose(p2, 2.0 * p1)  # penalty = radius * (path-fixed Lipschitz sum)


def test_penalty_equals_radius_times_lipschitz_for_a_linear_residual() -> None:
    a = jnp.array([[0.5, -1.0], [0.0, 2.0]])
    pen = WassersteinPenalty(residual=_LinearResidual(a=a), radius=0.3)
    xs = jax.random.normal(jax.random.key(0), (5, 2))
    us = jnp.zeros((5, 1))
    expected = 0.3 * 5 * jnp.linalg.norm(a)  # ||dr/dx||_F = ||a||_F at every state
    assert jnp.allclose(pen.penalty_trajectory(xs, us), expected)  # mean + radius*Lipschitz


def test_penalty_bounds_the_worst_case_residual_shift() -> None:
    # W-DRO certificate: any shift ||delta|| <= radius moves the summed residual by at most the
    # penalty; the top-singular-vector shift attains H*sigma1*radius (Frobenius upper-bounds it).
    a = jnp.array([[0.5, -1.0], [0.2, 2.0]])
    radius, n = 0.25, 6
    pen = WassersteinPenalty(residual=_LinearResidual(a=a), radius=radius)
    xs = jax.random.normal(jax.random.key(2), (n, 2))
    us = jnp.zeros((n, 1))
    _, sing, vt = jnp.linalg.svd(a)
    delta = radius * vt[0]  # worst-case aligned shift: max ||a @ delta|| = radius * sigma1
    base = jnp.sum(jax.vmap(lambda x: a @ x)(xs), axis=0)
    shifted = jnp.sum(jax.vmap(lambda x: a @ (x + delta))(xs), axis=0)
    realized = float(jnp.linalg.norm(shifted - base))
    penalty = float(pen.penalty_trajectory(xs, us))
    assert realized <= penalty + 1e-4  # the penalty certifies the worst-case degradation
    assert jnp.allclose(realized, n * sing[0] * radius, atol=1e-4)  # operator-norm shift is tight


def _regulate_toward_the_fragile_target() -> tuple[
    HybridDynamics, QuadraticCost, Array, Array, SupportModel
]:
    model = HybridDynamics(
        known=DampedOscillator(omega=1.0, zeta=0.1), residual=_TanhResidual(amp=0.4, width=0.5)
    )
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.0])),
        R=jnp.array([[0.01]]),
        Qf=jnp.diag(jnp.array([5.0, 1.0])),
        x_target=jnp.zeros(2),  # the target sits in the residual's steep, fragile region
    )
    x0 = jnp.array([1.5, 0.0])
    us0 = jnp.zeros((H, 1))
    k_x, k_u = jax.random.split(jax.random.key(0))
    support = SupportModel.fit(jax.random.normal(k_x, (500, 2)), jax.random.normal(k_u, (500, 1)))
    return model, cost, x0, us0, support


def _run(model, cost, x0, us0, support, penalty, lam_unc) -> tuple[float, float]:
    us, _ = pessimistic_control(
        model, x0, us0, DT, cost, support, lam_supp=0.0, u_lo=-5.0, u_hi=5.0,
        steps=150, uncertainty=penalty, lam_unc=lam_unc,
    )
    xs = rollout(model, x0, us, DT)
    settled = float(jnp.mean(jnp.abs(xs[H // 2 :, 0])))  # how close it settles to the fragile x0=0
    accrued = float(penalty.penalty_trajectory(xs[:-1], us))  # residual-Lipschitz along the path
    return settled, accrued


def test_wdro_control_backs_off_the_fragile_region() -> None:
    model, cost, x0, us0, support = _regulate_toward_the_fragile_target()
    robust = WassersteinPenalty.from_model(model, radius=1.0)

    naive_settled, naive_accrued = _run(model, cost, x0, us0, support, robust, lam_unc=0.0)
    robust_settled, robust_accrued = _run(model, cost, x0, us0, support, robust, lam_unc=6.0)

    assert naive_settled < 0.25  # ignoring the shift, control parks on the fragile target
    assert robust_settled > 0.5  # W-DRO holds out of the region where the residual is steep
    assert robust_accrued < 0.4 * naive_accrued  # cutting accrued residual-Lipschitz sharply

    # the penalty channel -- not lam_unc alone -- drives it: radius=0 recovers the naive trajectory
    zero_radius = WassersteinPenalty.from_model(model, radius=0.0)
    guard_settled, _ = _run(model, cost, x0, us0, support, zero_radius, lam_unc=6.0)
    assert jnp.allclose(guard_settled, naive_settled, atol=1e-3)
