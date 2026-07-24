"""ConfoundingRobustPenalty: linear in the radius, equals radius*Sigma||u_t|| (the §34 dimensional
bound on the confounded effect error), from_sensitivity matches the §32 MSM inflation, and it
shrinks the closed-loop action magnitude in pessimistic_control as the assumed confounding grows.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from chc.cost import QuadraticCost
from chc.dynamics import DampedOscillator, HybridDynamics
from chc.support import SupportModel, pessimistic_control
from chc.uncertainty import ConfoundingRobustPenalty, confounding_robust_inflation

DT = 0.1
H = 25


class _ZeroResidual(eqx.Module):
    """No learned correction: the penalty depends on the control alone, not the dynamics."""

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return jnp.zeros_like(x)


def test_penalty_is_linear_in_the_radius() -> None:
    us = jnp.array([[1.0], [-2.0], [0.5]])
    xs = jnp.zeros((3, 2))
    p1 = ConfoundingRobustPenalty(radius=0.3).penalty_trajectory(xs, us)
    p2 = ConfoundingRobustPenalty(radius=0.6).penalty_trajectory(xs, us)
    assert jnp.allclose(p2, 2.0 * p1)  # penalty = radius * (path-fixed control-norm sum)


def test_penalty_equals_radius_times_summed_control_norm() -> None:
    us = jnp.array([[3.0, 4.0], [0.0, 1.0]])  # per-step L2 norms 5, 1 (the §34 ||u_t|| factor)
    xs = jnp.zeros((2, 3))
    pen = ConfoundingRobustPenalty(radius=0.5)
    assert jnp.allclose(pen.penalty_trajectory(xs, us), 0.5 * (5.0 + 1.0))


def test_from_sensitivity_radius_matches_the_msm_inflation() -> None:
    pen = ConfoundingRobustPenalty.from_sensitivity(cvar_gap=1.3, gamma=2.5)
    assert pen.radius == confounding_robust_inflation(1.3, 0.0, 2.5)  # §32 half-width Delta(Gamma)


def test_zero_radius_penalty_is_zero_regardless_of_the_controls() -> None:
    us = jax.random.normal(jax.random.key(3), (H, 1))
    xs = jnp.zeros((H, 2))
    pen = ConfoundingRobustPenalty(radius=0.0)
    assert (
        pen.penalty_trajectory(xs, us) == 0.0
    )  # Gamma=1 -> no confounding -> the penalty vanishes


def _regulate_toward_a_reachable_target() -> tuple[
    HybridDynamics, QuadraticCost, Array, Array, SupportModel
]:
    model = HybridDynamics(known=DampedOscillator(omega=1.0, zeta=0.1), residual=_ZeroResidual())
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.0])),
        R=jnp.array([[0.01]]),  # cheap control: greedy uses sizeable actions to reach the target
        Qf=jnp.diag(jnp.array([5.0, 1.0])),
        x_target=jnp.array([1.0, 0.0]),
    )
    x0 = jnp.zeros(2)
    us0 = jnp.zeros((H, 1))
    k_x, k_u = jax.random.split(jax.random.key(0))
    support = SupportModel.fit(jax.random.normal(k_x, (500, 2)), jax.random.normal(k_u, (500, 1)))
    return model, cost, x0, us0, support


def _action_magnitude(model, cost, x0, us0, support, radius) -> float:
    us, _ = pessimistic_control(
        model,
        x0,
        us0,
        DT,
        cost,
        support,
        lam_supp=0.0,
        u_lo=-5.0,
        u_hi=5.0,
        steps=150,
        uncertainty=ConfoundingRobustPenalty(radius=radius),
        lam_unc=1.0,
    )
    return float(jnp.sum(jnp.abs(us)))


def test_more_confounding_shrinks_the_closed_loop_action() -> None:
    model, cost, x0, us0, support = _regulate_toward_a_reachable_target()
    none = _action_magnitude(model, cost, x0, us0, support, radius=0.0)  # trust the effect fully
    mild = _action_magnitude(model, cost, x0, us0, support, radius=0.5)
    strong = _action_magnitude(model, cost, x0, us0, support, radius=1.0)
    # a wider sensitivity radius = more distrust of the confounded effect = smaller, safer actions
    assert strong < mild < none
