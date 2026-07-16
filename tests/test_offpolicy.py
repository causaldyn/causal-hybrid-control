"""Off-policy gate: accurate value under overlap; flag when the target leaves the support."""

import jax
import jax.numpy as jnp

from chc.offpolicy import GaussianPolicy, fit_behavior_policy, off_policy_value


def _bandit(behavior: GaussianPolicy, n: int, key: jax.Array) -> dict[str, jax.Array]:
    k_x, k_u, k_r = jax.random.split(key, 3)
    xs = jax.random.normal(k_x, (n, 1))
    means = jax.vmap(behavior.mean)(xs)
    us = means + jnp.exp(behavior.log_std) * jax.random.normal(k_u, (n, 1))
    rewards = -((us[:, 0] - xs[:, 0]) ** 2) + 0.1 * jax.random.normal(k_r, (n,))  # optimal u = x
    return {"x": xs, "u": us, "r": rewards}


def test_ope_recovers_value_under_overlap() -> None:
    behavior = GaussianPolicy(
        weight=jnp.array([[0.5]]), bias=jnp.array([0.0]), log_std=jnp.array([0.0])
    )  # u ~ N(0.5 x, 1), broad coverage
    data = _bandit(behavior, 5000, jax.random.key(0))
    target = GaussianPolicy(
        weight=jnp.array([[1.0]]), bias=jnp.array([0.0]), log_std=jnp.array([jnp.log(0.3)])
    )  # u ~ N(x, 0.3), near-optimal

    result = off_policy_value(data, target, behavior)
    assert result["overlap_ok"]
    # near-optimal policy value ≈ -Var(u - x) = -0.09
    assert abs(result["snips_value"] - (-0.09)) < 0.05


def test_ope_flags_no_overlap() -> None:
    behavior = GaussianPolicy(
        weight=jnp.array([[0.5]]), bias=jnp.array([0.0]), log_std=jnp.array([0.0])
    )
    data = _bandit(behavior, 5000, jax.random.key(1))
    off_support = GaussianPolicy(
        weight=jnp.array([[1.0]]), bias=jnp.array([10.0]), log_std=jnp.array([jnp.log(0.1)])
    )  # actions ~10 away from anything logged

    result = off_policy_value(data, off_support, behavior)
    assert not result["overlap_ok"]
    assert result["ess_fraction"] < 0.1


def test_fit_behavior_policy_recovers_parameters() -> None:
    truth = GaussianPolicy(
        weight=jnp.array([[0.7]]), bias=jnp.array([-0.2]), log_std=jnp.array([jnp.log(0.5)])
    )
    data = _bandit(truth, 8000, jax.random.key(2))
    fitted = fit_behavior_policy(data["x"], data["u"])
    assert abs(float(fitted.weight[0, 0]) - 0.7) < 0.05
    assert abs(float(fitted.bias[0]) - (-0.2)) < 0.05
    assert abs(float(jnp.exp(fitted.log_std[0])) - 0.5) < 0.05
