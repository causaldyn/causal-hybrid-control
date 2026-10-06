"""Causal gate (H1): confounded fit is sign-flipped; adjusted fit recovers the true effect."""

import importlib

import jax
import jax.numpy as jnp
import pytest

from chc import causal
from chc.causal import (
    ConfoundedLinearSystem,
    dml_point_and_se,
    e_value,
    estimate_control_effect,
    estimate_effect_dml,
    estimate_effect_iv,
    refute_effect,
    sensitivity_analysis,
)

# Public as jax.enable_x64 from jax 0.8.0; the floor, 0.4.30, has only jax.experimental.enable_x64,
# which jax 0.11 no longer has.
if hasattr(jax, "enable_x64"):
    enable_x64 = jax.enable_x64
else:
    enable_x64 = importlib.import_module("jax.experimental").enable_x64


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


def test_e_value_grows_with_effect_and_bottoms_out_at_null() -> None:
    assert e_value(0.0)["e_value"] == 1.0  # a null effect needs no confounding to explain away
    assert e_value(1.0)["e_value"] > e_value(0.3)["e_value"] > e_value(0.0)["e_value"]  # monotone


def test_e_value_ci_is_one_when_the_interval_covers_the_null() -> None:
    wide = e_value(0.2, std_error=0.5)  # 95% interval spans the null
    tight = e_value(0.6, std_error=0.05)  # interval clears the null
    assert wide["e_value_ci"] == 1.0  # confounding need not be invoked
    assert tight["e_value_ci"] > 1.0  # a genuine bound survives the confidence limit
    assert tight["e_value_ci"] < tight["e_value"]  # the CI limit is the more conservative bound


def test_sensitivity_analysis_reports_the_e_value() -> None:
    data = ConfoundedLinearSystem(gamma=1.0).sample(40_000, jax.random.key(0))
    robust = sensitivity_analysis(data, adjust_for=("z",))
    assert robust["e_value"] > 1.0  # the adjusted effect resists confounding (risk-ratio scale)
    assert robust["e_value_ci"] >= 1.0  # confidence-limit E-value is always defined


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


def test_cross_fitting_folds_ignore_data_drawn_on_the_same_seed() -> None:
    # Folds drawn from key(seed) itself put 1000 float32 rows in the order of z, which the sampler
    # draws from that key's second child, and the estimate's error grew sixteenfold.
    system = ConfoundedLinearSystem()
    with enable_x64(False):
        errors = [
            float(estimate_effect_dml(system.sample(1_000, jax.random.key(seed)), seed=seed))
            - system.b_true
            for seed in range(6)
        ]
    assert max(abs(error) for error in errors) < 0.02


def test_the_random_common_cause_is_drawn_apart_from_the_data(monkeypatch) -> None:
    # It was drawn from the key the sampler drew z from, so it was z.
    seen: list[jax.Array] = []

    def spy(data: dict[str, jax.Array], adjust_for: tuple[str, ...] = ()) -> jax.Array:
        if "_rcc" in data:
            seen.append(data["_rcc"])
        return estimate_control_effect(data, adjust_for)

    monkeypatch.setattr(causal, "estimate_control_effect", spy)
    data = _data()
    refute_effect(data, adjust_for=("z",))
    (common_cause,) = seen
    for name, column in data.items():
        assert abs(float(jnp.corrcoef(common_cause, column)[0, 1])) < 0.05, name


def test_refutation_passes_for_adjusted_estimate() -> None:
    data = ConfoundedLinearSystem().sample(20_000, jax.random.key(0))
    report = refute_effect(data, adjust_for=("z",))
    assert report["passes"]
    assert abs(report["placebo"]) < 0.05  # permuting the treatment collapses the effect
    assert abs(report["random_common_cause"] - report["original"]) < 0.05  # stable
    assert abs(report["subset"] - report["original"]) < 0.1  # stable


SECOND_ROLES = {
    "the action as a covariate": (
        lambda d: estimate_control_effect(d, ("z", "u")),
        "the covariate 'u' is the action",
    ),
    "the outcome as a covariate": (
        lambda d: estimate_control_effect(d, ("z", "x_next")),
        "the covariate 'x_next' is the next state",
    ),
    "the action, for its sensitivity": (
        lambda d: sensitivity_analysis(d, ("z", "u")),
        "the covariate 'u' is the action",
    ),
    "the state, for its sensitivity": (
        lambda d: sensitivity_analysis(d, ("z", "x")),
        "the covariate 'x' is the state",
    ),
    "a covariate twice, for its sensitivity": (
        lambda d: sensitivity_analysis(d, ("z", "z")),
        r"covariates named more than once: \['z'\]",
    ),
    "the action as its instrument": (
        lambda d: estimate_effect_iv(d, "u"),
        "the instrument 'u' is the action",
    ),
    "the state as an instrument": (
        lambda d: estimate_effect_iv(d, "x"),
        "the instrument 'x' is the state",
    ),
    "the action, partialled out": (
        lambda d: estimate_effect_dml(d, ("x", "z", "u")),
        "the covariate 'u' is the action",
    ),
    "the outcome, partialled out": (
        lambda d: dml_point_and_se(d, ("x", "z", "x_next")),
        "the covariate 'x_next' is the next state",
    ),
}


@pytest.mark.parametrize(("call", "match"), SECOND_ROLES.values(), ids=SECOND_ROLES.keys())
def test_a_column_named_in_a_second_role_is_refused(call, match: str) -> None:
    """The state, the action and the outcome are ``x``, ``u`` and ``x_next``, and every other column
    is read by its name, from one dict. With the action among the covariates the effect read 0.5005
    against a true 1.0, its coefficient split between two copies; as its own instrument, 2SLS was
    the confounded regression; and a column read twice left the sensitivity's design singular, its
    error nan and its interval's E-value 1."""
    with pytest.raises(ValueError, match=match):
        call(_data())


def test_the_state_stays_a_covariate_where_it_repeats_no_regressor() -> None:
    """Partialling out reads no column but the covariates, so the state is one of them there; and in
    the adjusted regression a second copy of the state moves only its own coefficient."""
    data = _data()
    assert float(estimate_effect_dml(data, ("x", "z"))) == pytest.approx(1.0, abs=0.05)
    alone = float(estimate_control_effect(data, ("z",)))
    assert float(estimate_control_effect(data, ("z", "x"))) == pytest.approx(alone, rel=1e-9)


def test_the_random_common_cause_takes_a_name_the_data_does_not_hold() -> None:
    """It was added as ``_rcc``: a confounder of that name was replaced by the random column, and
    the refutation failed an estimate that was right."""
    data = _data()
    named = {**{k: v for k, v in data.items() if k != "z"}, "_rcc": data["z"]}
    report = refute_effect(named, adjust_for=("_rcc",))
    assert report == refute_effect(data, adjust_for=("z",))
