"""Causal gate (H1): confounded fit is sign-flipped; adjusted fit recovers the true effect."""

import jax
import jax.numpy as jnp

from chc.causal import (
    ConfoundedLinearSystem,
    estimate_control_effect,
    estimate_effect_dml,
    estimate_effect_iv,
    sensitivity_analysis,
)


def _data() -> dict[str, jax.Array]:
    return ConfoundedLinearSystem().sample(20_000, jax.random.key(0))


def test_naive_estimate_is_confounded() -> None:
    system = ConfoundedLinearSystem()
    b_naive = float(estimate_control_effect(_data(), adjust_for=()))
    assert b_naive < 0.0  # sign-flipped relative to the true +1.0 effect
    assert abs(b_naive - system.b_true) > 0.5  # substantially biased


def test_adjusted_estimate_recovers_true_effect() -> None:
    system = ConfoundedLinearSystem()
    b_causal = float(estimate_control_effect(_data(), adjust_for=("z",)))
    assert abs(b_causal - system.b_true) < 0.05  # recovers the true interventional effect


def test_confounding_flips_the_control_decision() -> None:
    """The whole point: acting on the naive estimate pushes the control the wrong way."""
    system = ConfoundedLinearSystem()
    data = _data()
    b_naive = estimate_control_effect(data, adjust_for=())
    b_causal = estimate_control_effect(data, adjust_for=("z",))
    assert jnp.sign(b_causal) == jnp.sign(system.b_true)  # causal: correct direction
    assert jnp.sign(b_naive) != jnp.sign(system.b_true)  # naive: wrong direction


def test_iv_recovers_effect_with_latent_confounder() -> None:
    system = ConfoundedLinearSystem(gamma=1.0)  # instrument active; z is treated as latent
    data = system.sample(40_000, jax.random.key(0))
    b_naive = float(estimate_control_effect(data, adjust_for=()))  # cannot adjust for latent z
    b_iv = float(estimate_effect_iv(data, instrument="w"))
    assert abs(b_naive - system.b_true) > 0.3  # naive is biased by the latent confounder
    assert abs(b_iv - system.b_true) < 0.1  # 2SLS recovers the true effect


def test_sensitivity_robustness_value() -> None:
    data = ConfoundedLinearSystem(gamma=1.0).sample(40_000, jax.random.key(0))
    robust = sensitivity_analysis(data, adjust_for=("z",))  # correctly adjusted -> strong effect
    fragile = sensitivity_analysis(data, adjust_for=())  # confounded -> fragile estimate
    for report in (robust, fragile):
        assert 0.0 <= report["robustness_value"] <= 1.0
    assert robust["robustness_value"] > 0.8  # the true effect is hard to explain away
    assert fragile["robustness_value"] < 0.4  # the confounded estimate is fragile


def test_dml_recovers_effect_under_nonlinear_confounding() -> None:
    k = jax.random.split(jax.random.key(1), 4)
    n = 20_000
    x = jax.random.normal(k[0], (n,))
    z = jax.random.normal(k[1], (n,))
    eta = jax.random.normal(k[2], (n,))
    noise = 0.1 * jax.random.normal(k[3], (n,))
    u = z**2 + eta  # action depends nonlinearly on the confounder
    y = 0.5 * x + 1.0 * u + 1.5 * z**2 + noise  # confounding enters through z^2
    data = {"x": x, "z": z, "u": u, "x_next": y}

    b_adjust = float(estimate_control_effect(data, adjust_for=("z",)))
    b_dml = float(estimate_effect_dml(data, covariates=("x", "z"), degree=3))
    assert abs(b_adjust - 1.0) > 0.3  # linear adjustment is biased by the z^2 confounding
    assert abs(b_dml - 1.0) < 0.1  # DML recovers the true effect
