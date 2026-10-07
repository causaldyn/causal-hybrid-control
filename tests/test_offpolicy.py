"""Off-policy gate: accurate value under overlap; flag when the target leaves the support."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chc.offpolicy import GaussianPolicy, fit_behavior_policy, off_policy_value

# Public as jax.enable_x64 from jax 0.8.0; the floor, 0.4.30, has only jax.experimental.enable_x64,
# which jax 0.11 no longer has.
if hasattr(jax, "enable_x64"):
    enable_x64 = jax.enable_x64
else:
    enable_x64 = importlib.import_module("jax.experimental").enable_x64


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


# --- the logger's fit reads the same in whatever units the actions and the states were logged in --


def _logs(key: jax.Array, noise: float) -> tuple[jax.Array, jax.Array, jax.Array]:
    """4000 rows: two states, an action linear in them with spread ``noise``, and its reward."""
    k_x, k_u = jax.random.split(key)
    xs = jax.random.normal(k_x, (4000, 2))
    us = xs @ jnp.array([[0.6], [-0.3]]) + 0.2 + noise * jax.random.normal(k_u, (4000, 1))
    return xs, us, -jnp.sum((us - 0.5 * xs[:, :1]) ** 2, axis=1)


def test_the_logger_s_fit_and_a_target_s_value_read_the_same_in_any_action_units() -> None:
    """The fitted spread was ``std + 1e-8`` in the actions' units, so the 1e-8 grew against a spread
    logged in small units: at 1e-6 of them the spread read 0.5183 where it read 0.5083 and the IPS
    value -0.2499 where it read -0.2518; at 1e-9 of them the spread read 21 times its size."""
    xs, us, rewards = _logs(jax.random.key(0), noise=0.5)

    def read(s: float) -> np.ndarray:
        behaviour = fit_behavior_policy(xs, s * us)
        target = GaussianPolicy(
            weight=s * jnp.array([[0.7, -0.2]]),
            bias=s * jnp.array([0.1]),
            log_std=jnp.log(s * jnp.array([0.4])),
        )
        value = off_policy_value({"x": xs, "u": s * us, "r": rewards}, target, behaviour)
        fitted = [*np.asarray(behaviour.weight).ravel(), *np.asarray(behaviour.bias)]
        spread = float(jnp.exp(behaviour.log_std[0]))
        values = [value["ips_value"], value["snips_value"], value["ess"]]
        return np.array([*np.divide(fitted, s), spread / s, *values])

    reference = read(1.0)
    for s in (1e-9, 1e-6, 1e-3, 1e3, 1e6):
        np.testing.assert_allclose(read(s), reference, rtol=1e-12, atol=0.0, err_msg=f"{s}")


def test_a_logger_that_set_its_action_from_the_state_alone_reads_a_spread_of_its_own_size() -> None:
    """Its residual is rounding, so its spread is the floor, which was 1e-8 in the actions' units:
    logged at 1e-6 of these units, the actions read a spread of 1e-2 of their spread where they
    read 1e-8. The floor is now 1e-8 of the actions' spread about their mean, in any units and from
    any origin; 1e-8 of their root mean square read 1e-5 of their spread with the actions 1e3 of it
    off zero."""
    xs, us, _ = _logs(jax.random.key(1), noise=0.0)
    us = (us - jnp.mean(us)) / jnp.std(us)
    for s in (1.0, 1e-9, 1e-6, 1e-3, 1e3, 1e6):
        for origin in (0.0, 1e3):
            spread = float(jnp.exp(fit_behavior_policy(xs, s * (us + origin)).log_std[0])) / s
            assert spread == pytest.approx(1e-8, rel=1e-6, abs=0.0), (s, origin)


def test_the_fitted_spread_is_the_residual_s_own_wherever_the_log_has_one() -> None:
    """The floor stays a floor: the 1e-8 that was added to every spread moved this one by 2e-8 of
    itself, and by 2e-2 of it at 1e-6 of the actions' units."""
    xs, us, _ = _logs(jax.random.key(2), noise=0.5)
    design = np.column_stack([np.asarray(xs), np.ones(xs.shape[0])])
    for s in (1.0, 1e-6, 1e6):
        actions = s * np.asarray(us)
        residual = actions - design @ np.linalg.lstsq(design, actions, rcond=None)[0]
        spread = float(jnp.exp(fit_behavior_policy(xs, s * us).log_std[0]))
        assert spread == pytest.approx(float(np.std(residual)), rel=1e-12, abs=0.0), s


@pytest.mark.parametrize("level", [0.0, 0.7], ids=["at zero", "at 0.7"])
@pytest.mark.parametrize("double", [True, False], ids=["double", "single"])
def test_a_state_the_log_held_at_one_value_reads_no_weight(level: float, double: bool) -> None:
    """A state the log held at one value cannot be told from the intercept. Centred, it is zero or
    its mean's rounding, under lstsq's cut, so it reads no weight and the rest of the fit is the
    one without it. Uncentred, the intercept was split between them: a state held at 0.7 read a
    weight of 0.092, and the intercept 0.131 where it read 0.196."""
    with enable_x64(double):
        xs, us, _ = _logs(jax.random.key(3), noise=0.5)
        held = fit_behavior_policy(xs.at[:, 1].set(level), us)
        without = fit_behavior_policy(xs[:, :1], us)
    # single precision centres a state held at 0.7 to its mean's rounding, 6e-8 of it, which lstsq
    # reads as a weight of 4.5e-8 and an intercept moved by 2e-7 of itself
    weight, rel = (1e-12, 1e-12) if double else (1e-6, 1e-6)
    assert float(held.weight[0, 1]) == pytest.approx(0.0, abs=weight)
    assert float(held.weight[0, 0]) == pytest.approx(float(without.weight[0, 0]), rel=rel, abs=0.0)
    assert float(held.bias[0]) == pytest.approx(float(without.bias[0]), rel=rel, abs=0.0)


def test_actions_the_logger_never_took_read_a_spread_of_1e_8() -> None:
    """A column of zeros has neither a spread nor a size for the floor to be a share of."""
    xs, us, _ = _logs(jax.random.key(3), noise=0.5)
    idle = fit_behavior_policy(xs, jnp.zeros_like(us))
    assert float(idle.log_std[0]) == pytest.approx(float(np.log(1e-8)), rel=1e-12, abs=0.0)


def _with_a_driver(key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """4000 rows: a state ``x``, a driver ``z``, and an action that follows both."""
    k_x, k_z, k_u = jax.random.split(key, 3)
    x, z = jax.random.normal(k_x, (4000, 1)), jax.random.normal(k_z, (4000, 1))
    return x, z, 0.6 * x + 0.5 * z + 0.2 + 0.3 * jax.random.normal(k_u, (4000, 1))


@pytest.mark.parametrize(("scale", "atol"), [(1e-11, 1e-4), (1e-13, 1e-2)], ids=["1e-11", "1e-13"])
def test_a_state_in_small_units_far_from_zero_reads_its_slope(scale: float, atol: float) -> None:
    """``1 + 1e-11 z`` is not a constant but ``z`` in small units far from zero, so the fit reads
    the slope ``z`` has, times 1e11, and the same fitted actions. Divided by its spread but not
    centred, it took the intercept's place and read none of ``z``'s slope, and the fitted actions
    moved by 2.1. ``1 + 1e-13 z`` moves by 450 eps of its size, which is data too: a floor of 1e-12
    of its size counted it as rounding, and the fitted actions moved by 2.1 again."""
    x, z, us = _with_a_driver(jax.random.key(6))
    reference = fit_behavior_policy(jnp.concatenate([x, z], axis=1), us)
    design = jnp.concatenate([x, 1.0 + scale * z], axis=1)
    fit = fit_behavior_policy(design, us)
    assert np.all(np.isfinite(np.asarray(fit.weight)))
    # the column keeps about 5 of z's digits at 1e-11 and 3 at 1e-13, 2.2e-16 of 1 against the
    # scale of z; the fitted actions moved by 1.5e-5 and 1.7e-3
    slope = float(fit.weight[0, 1]) * scale
    assert slope == pytest.approx(float(reference.weight[0, 1]), rel=1e-4, abs=0.0)
    fitted = jax.vmap(fit.mean)(design)
    expected = jax.vmap(reference.mean)(jnp.concatenate([x, z], axis=1))
    np.testing.assert_allclose(fitted, expected, rtol=0.0, atol=atol)


@pytest.mark.parametrize(
    ("double", "jitter"), [(True, 4e-15), (False, 1e-6)], ids=["double", "single"]
)
def test_a_state_held_but_for_rounding_reads_no_weight(double: bool, jitter: float) -> None:
    """``1 + 4e-15`` either way, 18 eps, is a constant up to rounding in double precision, as is
    ``1 + 1e-6`` either way, 8 eps, in single: a spread of at most 64 eps of its size counts as
    none, so the state reads no weight and the rest of the fit is the one with the state at zero.
    The old fit split the intercept with it and read a weight of 0.097, and 0.101 in single
    precision. Counted as a spread, it took up the actions' noise with a weight of 1.4e12, and 345
    in single precision, where a floor of 1e-12 of its size counted it so."""
    with enable_x64(double):
        x, _, us = _with_a_driver(jax.random.key(6))
        signs = jnp.where(jax.random.bernoulli(jax.random.key(7), 0.5, (4000, 1)), 1.0, -1.0)
        fit = fit_behavior_policy(jnp.concatenate([x, 1.0 + jitter * signs], axis=1), us)
        zero = fit_behavior_policy(jnp.concatenate([x, jnp.zeros_like(signs)], axis=1), us)
    # single precision fits the rest to 7e-7 of itself here
    rel = 1e-6 if double else 1e-5
    assert abs(float(fit.weight[0, 1])) < 1e-6
    assert float(fit.weight[0, 0]) == pytest.approx(float(zero.weight[0, 0]), rel=rel, abs=0.0)
    assert float(fit.bias[0]) == pytest.approx(float(zero.bias[0]), rel=rel, abs=0.0)


def test_a_state_logged_in_other_units_keeps_its_weight_in_single_precision() -> None:
    """lstsq cuts singular values below ``eps * max(N, p)`` of the largest, 4.8e-4 here in float32,
    so a state logged at 1e-6 of the other columns' units fell under the cut: its weight read
    exactly 0, the fitted spread 0.791 where it read 0.514, and the IPS value of a target restated
    in those units -1.487 where it read -0.252. A state logged at 1e6 of them put the other state
    under the cut instead, and the value read -0.486."""
    with enable_x64(False):
        xs, us, rewards = _logs(jax.random.key(0), noise=0.5)
        assert xs.dtype == jnp.float32

        def read(s: float) -> np.ndarray:
            units = jnp.array([s, 1.0])
            behaviour = fit_behavior_policy(xs * units, us)
            target = GaussianPolicy(
                weight=jnp.array([[0.7, -0.2]]) / units,
                bias=jnp.array([0.1]),
                log_std=jnp.log(jnp.array([0.4])),
            )
            value = off_policy_value({"x": xs * units, "u": us, "r": rewards}, target, behaviour)
            fitted = [*np.asarray(behaviour.weight[0] * units), float(behaviour.bias[0])]
            spread = float(jnp.exp(behaviour.log_std[0]))
            return np.array([*fitted, spread, value["ips_value"], value["ess"]])

        reference = read(1.0)
        for s in (1e-9, 1e-6, 1e-3, 1e3, 1e6):
            # single precision: a column rescaled to within a factor 1.4 of its size at s = 1
            # rounds otherwise in the solve, by 2e-6 of the weight on these logs
            np.testing.assert_allclose(read(s), reference, rtol=2e-5, atol=0.0, err_msg=f"{s}")


def test_a_state_logged_far_from_zero_keeps_its_weight_in_single_precision() -> None:
    """A state 1e4 of its spread off zero is all but collinear with the intercept, so lstsq's cut,
    4.8e-4 of the largest singular value here, dropped it: with the state also at 1e-6 of the
    other's units, its weight read 1.9e-9 where it read 0.601 and the fitted spread 0.791 where it
    read 0.514. The old fit lost it at 1e3 of its spread already. Centred, the intercept carries
    the level."""
    with enable_x64(False):
        xs, us, _ = _logs(jax.random.key(0), noise=0.5)
        spread = float(jnp.std(xs[:, 0]))

        def read(scale: float, origin: float) -> np.ndarray:
            moved = xs.at[:, 0].set(scale * (xs[:, 0] + origin * spread))
            fit = fit_behavior_policy(moved, us)
            weight = np.asarray(fit.weight[0]) * np.array([scale, 1.0])
            return np.array([*weight, float(jnp.exp(fit.log_std[0]))])

        reference = read(1.0, 0.0)
        for scale, origin in ((1.0, 1e4), (1e-6, 1e4), (1e-6, 1e3)):
            # single precision keeps a state 1e4 of its spread off zero to 1e-3 of that spread in
            # each row; over 4000 rows the fit reads it to about 1e-5 (measured 6e-6)
            np.testing.assert_allclose(
                read(scale, origin), reference, rtol=1e-4, atol=0.0, err_msg=f"{scale} {origin}"
            )
