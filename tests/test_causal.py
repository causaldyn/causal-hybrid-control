"""Causal gate (H1): confounded fit is sign-flipped; adjusted fit recovers the true effect."""

import importlib
import itertools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
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


# ---- the instrument's rank ----

NOT_MOVED = "does not move the action beyond what the state explains"


def _iv_data() -> dict[str, jax.Array]:
    return ConfoundedLinearSystem(gamma=1.0).sample(40_000, jax.random.key(0))


def _less(column: jax.Array, *on: jax.Array) -> jax.Array:
    """``column`` less its least-squares projection on a constant and the columns ``on``, each
    centred first: the same projection, which least squares rounds less beside a column far from
    zero."""
    on_centred = (np.asarray(c) - np.mean(np.asarray(c)) for c in on)
    design = np.column_stack([np.ones(column.shape[0]), *on_centred])
    values = np.asarray(column)
    return jnp.asarray(values - design @ np.linalg.lstsq(design, values, rcond=None)[0])


def _with_action(data: dict[str, jax.Array], action: jax.Array) -> dict[str, jax.Array]:
    """The log with another action, and the outcome moved by it at the effect, 1.0."""
    return {**data, "u": action, "x_next": data["x_next"] + action - data["u"]}


def _state_copy(data: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """An action that moves with the state as well, and an instrument that is the state's affine
    copy: it moves the action as the state does, and no further."""
    return {**_with_action(data, data["u"] + 0.8 * data["x"]), "w": 2.0 * data["x"] + 3.0}


def _single_state_copy(data: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """:func:`_state_copy`'s log in single precision, its copy of the state computed there."""
    log = {name: column.astype(jnp.float32) for name, column in _state_copy(data).items()}
    return {**log, "w": 2.0 * log["x"] + 3.0}


def _single_state(data: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """:func:`_state_copy`'s log with the state alone in single precision: the copy, computed in
    double, differs from it by the state's rounding."""
    log = _state_copy(data)
    return {**log, "x": log["x"].astype(jnp.float32)}


NO_RANK: dict[str, Callable[[dict[str, jax.Array]], dict[str, jax.Array]]] = {
    "an instrument of zeros": lambda d: {**d, "w": jnp.zeros_like(d["w"])},
    "a constant instrument": lambda d: {**d, "w": jnp.full_like(d["w"], 5.0)},
    "noise less its projection on the state and the action": lambda d: {
        **d,
        "w": _less(jax.random.normal(jax.random.key(7), d["w"].shape), d["x"], d["u"]),
    },
    "the state's copy, where the action moves with the state": _state_copy,
    "the state's copy, in single precision": _single_state_copy,
    "the state in single precision, its copy in double": _single_state,
    "an action the state sets": lambda d: _with_action(d, 0.7 * d["x"] + 2.0),
}


@pytest.mark.parametrize("log", NO_RANK.values(), ids=NO_RANK.keys())
def test_an_instrument_that_moves_the_action_nowhere_beyond_the_state_is_refused(log) -> None:
    """2SLS's moment has rank only where the instrument moves the action beyond what the state and
    a constant explain. Short of it the second stage's design is collinear, and least squares
    returned its minimum-norm coefficient where the truth is 1.0: -0.0032 for an instrument of
    zeros, a constant or noise less its projection on the state and the action; 0.63 for the
    state's affine copy where the action moves with the state, though the copy correlates with the
    action at 0.39; 2.61 for that copy computed in single precision, and 0.15 for one computed in
    double beside the state in single, each of which leaves the copy the state's rounding; and 0.88
    where the state sets the action, which no instrument then moves."""
    with pytest.raises(ValueError, match=NOT_MOVED):
        estimate_effect_iv(log(_iv_data()), instrument="w")


@pytest.mark.parametrize("level", [0.0, 4.0])
def test_a_state_that_never_moves_leaves_the_instrument_alone_to_move_the_action(
    level: float,
) -> None:
    """A state that never moves explains nothing a constant does not, so the plant's instrument
    still estimates the effect, and an instrument of zeros is still refused: it read 5.1e-5 beside
    a state of zeros and 3.0e-6 beside one of fours, where the truth is 1.0."""
    data = {**_iv_data(), "x": jnp.full((40_000,), level)}
    assert float(estimate_effect_iv(data, instrument="w")) == pytest.approx(1.0, abs=0.01)
    with pytest.raises(ValueError, match=NOT_MOVED):
        estimate_effect_iv({**data, "w": jnp.zeros_like(data["w"])}, instrument="w")


def test_a_relevant_instrument_estimates_as_it_did() -> None:
    """The positive control: the plant's instrument, whose estimate is the unfixed release's two
    stages bit for bit, on any machine, and 1.0085126809414267 where they were measured."""
    data = _iv_data()
    x, u, w, y = data["x"], data["u"], data["w"], data["x_next"]
    ones = jnp.ones((x.shape[0], 1))
    first = jnp.concatenate([jnp.stack([x, w], axis=1), ones], axis=1)
    pushed = first @ jnp.linalg.lstsq(first, u, rcond=None)[0]
    second = jnp.concatenate([jnp.stack([x, pushed], axis=1), ones], axis=1)
    unfixed = float(jnp.linalg.lstsq(second, y, rcond=None)[0][1])
    effect = float(estimate_effect_iv(data, instrument="w"))
    assert effect == unfixed
    assert effect == pytest.approx(1.0085126809414267, rel=1e-12, abs=0.0)


UNITS = list(itertools.product(("w", "u", "x"), (1e-6, 1e6, -1.0)))


@pytest.mark.parametrize(("column", "factor"), UNITS)
def test_a_relevant_instrument_estimates_in_any_units(column: str, factor: float) -> None:
    """The instrument, the action and the state each logged in a millionth of their units, in a
    million times them and with the sign flipped: the plant's instrument still estimates the
    effect, in the action's units."""
    data = _iv_data()
    base = float(estimate_effect_iv(data, instrument="w"))
    moved = {**data, column: data[column] * factor}
    assert float(estimate_effect_iv(moved, instrument="w")) == pytest.approx(
        base / factor if column == "u" else base, rel=1e-9, abs=0.0
    )


FAR_UNITS = list(itertools.product(("w", "u", "x"), (1e-12, 1e-6, 1e6, 1e12, -1.0)))


@pytest.mark.parametrize(
    ("log", "column", "factor"),
    [(log, *units) for log, units in itertools.product(NO_RANK.values(), FAR_UNITS)],
    ids=[f"{name}, {c} * {f:g}" for name, (c, f) in itertools.product(NO_RANK, FAR_UNITS)],
)
def test_an_instrument_short_of_rank_is_refused_in_any_units(
    log, column: str, factor: float
) -> None:
    """Each log short of rank, with the instrument, the action or the state logged in a millionth
    or a million millionth of its units, a million or a million million times them, or with the
    sign flipped. The state is scaled to unit spread before its projection, so least squares does
    not drop it as rounding, as it drops a column 1e-12 the size of the constant."""
    short = log(_iv_data())
    with pytest.raises(ValueError, match=NOT_MOVED):
        estimate_effect_iv({**short, column: short[column] * factor}, instrument="w")


LEVELS = list(itertools.product(("w", "u", "x"), (1e6, -1e6)))


@pytest.mark.parametrize(
    ("log", "column", "level"),
    [(log, *at) for log, at in itertools.product(NO_RANK.values(), LEVELS)],
    ids=[f"{name}, {c} + {lv:g}" for name, (c, lv) in itertools.product(NO_RANK, LEVELS)],
)
def test_an_instrument_short_of_rank_is_refused_at_any_level(
    log, column: str, level: float
) -> None:
    """Each log short of rank, with the instrument or the action logged a million above or below
    zero, or drawn from a state that far from it. A column so far from zero is known only to its
    rounding there, which moves the product of the two by itself: noise less its projection, a
    million above zero, moves the action by 4.4e-13 of a correlation, its rounding's alone. A
    state so far out is all but collinear with the constant, and least squares on the two as they
    come leaves its copy a spread beyond them; centred first, the state is read in full. It is
    shifted before the log is drawn from it: shifted after, its rounding there is information the
    copy keeps and the state does not."""
    data = _iv_data()
    if column == "x":
        short = log({**data, "x": data["x"] + level})
    else:
        drawn = log(data)
        short = {**drawn, column: drawn[column] + level}
    with pytest.raises(ValueError, match=NOT_MOVED):
        estimate_effect_iv(short, instrument="w")
