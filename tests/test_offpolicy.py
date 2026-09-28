"""Off-policy gate: accurate value under overlap; flag when the target leaves the support."""

import jax
import jax.numpy as jnp
import numpy as np

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


def test_the_one_step_estimand_is_not_the_deployed_value() -> None:
    # Logs from x' = 0.9 x + u + w, w ~ N(0, 0.1), under u = -0.4 x + N(0, 1). The candidate
    # u = -0.1 x + N(0, 1.2^2) spreads the state wider than the logger: its stationary variance is
    # 1.54/0.36 = 4.28 against the logger's 1.1/0.75 = 1.47, past twice it, so the stationary
    # state-action ratio a deployed plan needs has infinite variance. The weights here are
    # one-step ones, and SNIPS converges to the candidate's value on the LOGGER's states.
    rng = np.random.default_rng(31)
    steps, burn = 60_000, 1_000
    x = np.zeros(steps + 1)
    u = np.zeros(steps)
    for t in range(steps):
        u[t] = -0.4 * x[t] + rng.standard_normal()
        x[t + 1] = 0.9 * x[t] + u[t] + np.sqrt(0.1) * rng.standard_normal()
    xs, us = jnp.asarray(x[burn:steps, None]), jnp.asarray(u[burn:, None])
    rewards = -(xs[:, 0] ** 2 + us[:, 0] ** 2)
    target = GaussianPolicy(
        weight=jnp.array([[-0.1]]), bias=jnp.array([0.0]), log_std=jnp.array([np.log(1.2)])
    )

    result = off_policy_value({"x": xs, "u": us, "r": rewards}, target, fit_behavior_policy(xs, us))

    one_step = -((1.1 / 0.75) * (1 + 0.1**2) + 1.2**2)  # E_{x~d_b} E_{u~pi}[r] = -2.921
    deployed = -((1.54 / 0.36) * (1 + 0.1**2) + 1.2**2)  # the candidate's own stationary value
    assert result["overlap_ok"]  # the flag cannot see it
    assert abs(result["snips_value"] - one_step) < 0.15
    assert abs(result["snips_value"] - deployed) > 0.4 * abs(deployed)
