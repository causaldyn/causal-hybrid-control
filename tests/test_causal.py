"""Causal gate (H1): confounded fit is sign-flipped; adjusted fit recovers the true effect."""

import jax
import jax.numpy as jnp

from chc.causal import ConfoundedLinearSystem, estimate_control_effect


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
