"""R-learner: recovers a heterogeneous effect tau(x) under confounding; a naive regression can't."""

import jax
import jax.numpy as jnp

from chc import RLearner


def _heterogeneous_confounded(n: int, seed: int) -> tuple[dict, jnp.ndarray]:
    keys = jax.random.split(jax.random.key(seed), 5)
    x0 = jax.random.normal(keys[0], (n,))
    x1 = jax.random.normal(keys[1], (n,))
    z = jax.random.normal(keys[2], (n,))  # confounder: drives both treatment and outcome
    u = 1.5 * z + 0.5 * jax.random.normal(keys[3], (n,))  # treatment confounded by z
    true_tau = 1.0 + 0.8 * x0  # the effect of u depends on x0 (heterogeneous)
    x_next = true_tau * u + 2.0 * z + 0.1 * jax.random.normal(keys[4], (n,))
    return {"x0": x0, "x1": x1, "z": z, "u": u, "x_next": x_next}, true_tau


def test_rlearner_recovers_heterogeneous_effect_under_confounding() -> None:
    data, true_tau = _heterogeneous_confounded(8000, seed=0)
    covariates = ("x0", "x1", "z")
    estimate = RLearner(degree=3, cate_degree=1).estimate(data, covariates=covariates)
    covs = jnp.stack([data[c] for c in covariates], axis=1)
    predicted = estimate.cate(covs)
    assert abs(estimate.effect - 1.0) < 0.1  # ATE recovered (true 1.0)
    assert float(jnp.corrcoef(predicted, true_tau)[0, 1]) > 0.98  # the CATE tracks the truth
    assert float(jnp.sqrt(jnp.mean((predicted - true_tau) ** 2))) < 0.1  # low pointwise CATE error


def test_rlearner_beats_naive_treatment_regression() -> None:
    data, _ = _heterogeneous_confounded(8000, seed=1)
    y, u = data["x_next"], data["u"]
    naive_ate = float(jnp.sum((y - y.mean()) * (u - u.mean())) / jnp.sum((u - u.mean()) ** 2))
    r_ate = RLearner().estimate(data, covariates=("x0", "x1", "z")).effect
    assert abs(r_ate - 1.0) < abs(naive_ate - 1.0)  # residualisation de-confounds; naive does not
