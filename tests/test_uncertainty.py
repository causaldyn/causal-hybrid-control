"""Calibrated pessimism: the ensemble flags out-of-support states, split conformal hits nominal
coverage, and penalising that uncertainty avoids the model exploitation a greedy controller hits.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from chc import SplitConformal, fit_ensemble
from chc.benchmark import ModelUncertaintyTask
from chc.dynamics import HybridDynamics, LinearDynamics
from chc.integrate import rk4_step
from chc.residual import ZeroResidual

DT = 0.1


class _CubicDrag(eqx.Module):
    a: Array
    b: Array
    drag: float

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return self.a @ x + self.b @ (u - self.drag * u**3)


def _fit_ensemble_on_support(seed: int = 0) -> tuple[HybridDynamics, _CubicDrag]:
    a = jnp.array([[0.0, 1.0], [-1.0, -0.2]])
    b = jnp.array([[0.0], [1.0]])
    known = HybridDynamics(known=LinearDynamics(a_matrix=a, b_matrix=b), residual=ZeroResidual(2))
    plant = _CubicDrag(a=a, b=b, drag=0.15)
    k_x, k_u = jax.random.split(jax.random.key(seed))
    xs = jax.random.normal(k_x, (1500, 2))
    us = 0.4 * jax.random.normal(k_u, (1500, 1))  # narrow action support
    x_next = jax.vmap(lambda x, u: rk4_step(plant, 0.0, x, u, DT))(xs, us)
    model, _ = fit_ensemble(known, {"x": xs, "u": us, "x_next": x_next}, DT, n_members=4, steps=700)
    return model, plant


def _sample(plant: _CubicDrag, key: Array, n: int, scale: float, shift: float = 0.0) -> dict:
    k_x, k_u = jax.random.split(key)
    xs = jax.random.normal(k_x, (n, 2))
    us = shift + scale * jax.random.normal(k_u, (n, 1))
    x_next = jax.vmap(lambda x, u: rk4_step(plant, 0.0, x, u, DT))(xs, us)
    return {"x": xs, "u": us, "x_next": x_next}


def test_ensemble_disagreement_flags_out_of_support() -> None:
    model, plant = _fit_ensemble_on_support()
    ensemble = model.residual

    def mean_disagreement(data: dict) -> float:
        per_point = jax.vmap(lambda x, u: ensemble.disagreement(0.0, x, u))(data["x"], data["u"])
        return float(jnp.mean(per_point))

    in_region = mean_disagreement(_sample(plant, jax.random.key(5), 400, 0.4))
    out_region = mean_disagreement(_sample(plant, jax.random.key(6), 400, 1.0, shift=4.0))
    assert out_region > 5.0 * in_region  # the ensemble knows where it is extrapolating


def test_split_conformal_hits_nominal_coverage() -> None:
    model, plant = _fit_ensemble_on_support()
    for alpha in (0.1, 0.2):
        calib = _sample(plant, jax.random.key(7), 800, 0.4)
        test = _sample(plant, jax.random.key(8), 800, 0.4)
        conformal = SplitConformal.calibrate(model, calib, DT, alpha=alpha)
        assert abs(conformal.coverage(test) - (1.0 - alpha)) < 0.05  # finite-sample coverage holds


def test_calibrated_pessimism_avoids_model_exploitation() -> None:
    task = ModelUncertaintyTask(n_members=4, fit_steps=700, n_data=1500, inner_steps=250)
    results = {r.controller: r for r in task.run(seed_data=0)}
    assert results["oracle"].regret == 0.0
    assert results["calibrated"].regret < 0.1 * results["greedy"].regret  # penalising U avoids it
    assert results["greedy"].ood_rate > 0.3  # greedy pushes into the high-uncertainty region
    assert results["calibrated"].ood_rate < 0.1  # calibrated stays where the model is trustworthy
