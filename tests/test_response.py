"""The response curves' contract: each family's bounds, bend, tangency and nestings.

The inflections and tangencies are derived in ``validation/response_curves.mac``; the anchors below
are its double-precision roots.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from chc.response import (
    Algebraic,
    Arctan,
    BetaCDF,
    BurrXII,
    ChapmanRichards,
    Envelope,
    Exponential,
    GammaCDF,
    Gompertz,
    HalfNormal,
    Hill,
    Kumaraswamy,
    Logarithmic,
    Logistic,
    LogNormalCDF,
    MichaelisMenten,
    Power,
    Response,
    Richards,
    Ricker,
    Saturation,
    Tanh,
    Weibull,
)

CONCAVE = [
    MichaelisMenten(2.0),
    Exponential(2.0),
    Tanh(2.0),
    Arctan(2.0),
    Algebraic(2.0),
    HalfNormal(2.0),
]
S_SHAPED = [
    Hill(2.0, 2.5),
    Weibull(2.0, 1.8),
    Logistic(2.0, 4.0),
    Gompertz(2.0, 5.0),
    Richards(2.0, 3.0, 0.5),
    ChapmanRichards(2.0, 3.0),
    GammaCDF(2.0, 3.0),
    LogNormalCDF(2.0, 0.5),
    BurrXII(2.0, 4.0, 1.5),
    BetaCDF(2.0, 3.0, 2.0),
    Kumaraswamy(2.0, 3.0, 2.0),
]
BOUNDED = CONCAVE + S_SHAPED
# the S-shaped families at shapes where they are concave from zero spend
CONCAVE_SHAPES = [
    Hill(2.0, 0.7),
    Weibull(2.0, 1.0),
    Gompertz(2.0, 0.8),
    ChapmanRichards(2.0, 0.5),
    GammaCDF(2.0, 1.0),
    BurrXII(2.0, 1.0, 3.0),
    BetaCDF(2.0, 1.0, 2.0),
    Kumaraswamy(2.0, 0.8, 2.0),
]


def name(curve: Response) -> str:
    return type(curve).__name__


def slope(curve: Response, spend: float) -> float:
    return float(jax.grad(curve)(jnp.asarray(spend)))


def curvature(curve: Response, spend: float) -> float:
    return float(jax.grad(jax.grad(curve))(jnp.asarray(spend)))


def test_the_library_ships_seventeen_bounded_families_and_three_others() -> None:
    assert len({name(curve) for curve in BOUNDED}) == 17
    assert not any(isinstance(curve, Saturation) for curve in (Logarithmic(1.0), Power(1.0, 0.5)))
    assert not isinstance(Ricker(1.0), Saturation)


@pytest.mark.parametrize("curve", BOUNDED + CONCAVE_SHAPES, ids=name)
def test_every_saturation_curve_starts_at_zero_rises_and_approaches_one(curve: Saturation) -> None:
    spend = jnp.linspace(0.0, 40.0, 2001)
    values = np.asarray(curve(spend))
    slopes = np.asarray(jax.vmap(jax.grad(curve))(spend[1:]))
    assert values[0] == 0.0
    assert np.all(np.diff(values) >= 0.0)
    assert np.all(values <= 1.0)
    assert np.all(np.isfinite(slopes))
    assert np.all(slopes >= 0.0)
    assert float(curve(1e300)) == pytest.approx(1.0, abs=4e-16)


@pytest.mark.parametrize(
    ("curve", "expected"),
    [
        (MichaelisMenten(2.0), 0.5),
        (Exponential(2.0), 0.5),
        (Tanh(2.0), 0.5),
        (Arctan(2.0), 1.0 / math.pi),
        (Algebraic(2.0), 0.5),
        (HalfNormal(2.0), 1.0 / math.sqrt(math.pi)),
        (Hill(2.0, 1.0), 0.5),
        (Weibull(2.0, 1.0), 0.5),
        (ChapmanRichards(2.0, 1.0), 0.5),
        (GammaCDF(2.0, 1.0), 0.5),
        (BurrXII(2.0, 1.0, 2.0), 1.0),
        (BetaCDF(2.0, 1.0, 2.0), 1.0),
        (Kumaraswamy(2.0, 1.0, 3.0), 1.5),
        (Logistic(2.0, 4.0), 4.0 / (1.0 + math.exp(4.0)) / 2.0),
        (Hill(2.0, 2.5), 0.0),
        (LogNormalCDF(2.0, 0.5), 0.0),
    ],
    ids=lambda value: name(value) if isinstance(value, Response) else "",
)
def test_the_slope_at_zero_spend_is_the_curves_own(curve: Saturation, expected: float) -> None:
    # a planner started at zero spend reads this number; JAX's own gamma and beta CDFs return nan
    # there at shape 1, and a flat 0 would have been as wrong
    assert slope(curve, 0.0) == pytest.approx(expected, rel=1e-14, abs=1e-300)


@pytest.mark.parametrize("curve", S_SHAPED, ids=name)
def test_the_inflection_is_where_the_curvature_changes_sign(curve: Saturation) -> None:
    bend = curve.inflection()
    assert bend > 0.0
    peak = max(abs(curvature(curve, spend)) for spend in np.linspace(0.05, 3.0, 60) * bend)
    assert abs(curvature(curve, bend)) <= 1e-9 * peak
    assert curvature(curve, 0.9 * bend) > 0.0
    assert curvature(curve, 1.1 * bend) < 0.0


@pytest.mark.parametrize("curve", CONCAVE + CONCAVE_SHAPES, ids=name)
def test_a_concave_curve_reports_no_inflection_and_no_tangency(curve: Saturation) -> None:
    assert curve.inflection() == 0.0
    assert curve.tangency() == 0.0
    assert all(curvature(curve, spend) <= 1e-12 for spend in np.linspace(0.02, 6.0, 60))


@pytest.mark.parametrize("curve", S_SHAPED, ids=name)
def test_the_tangent_from_the_origin_touches_the_curve_at_its_tangency(curve: Saturation) -> None:
    touch = curve.tangency()
    assert touch > curve.inflection()
    assert float(curve(touch)) == pytest.approx(touch * slope(curve, touch), rel=1e-12)


@pytest.mark.parametrize(
    ("curve", "anchor"),
    [
        (Logistic(2.0, 4.0), 1.384939756667759),
        (Gompertz(2.0, 5.0), 2.559040989350351),
        (Richards(2.0, 3.0, 0.5), 1.454459495565034),
        (GammaCDF(2.0, 3.0), 3.383634282853181),
        (LogNormalCDF(2.0, 0.5), 1.295577262006302),
        (BurrXII(2.0, 4.0, 1.5), 1.17342542916607),
        (BurrXII(2.0, 3.0, 2.0), 1.0),
        (Kumaraswamy(2.0, 3.0, 2.0), 0.9283177667225558),
        (BetaCDF(2.0, 3.0, 2.0), 8.0 / 9.0),
        (Weibull(2.0, 2.0), 1.120906422778534),
        (ChapmanRichards(2.0, 3.0), 1.903813694440384),
    ],
    ids=lambda value: name(value) if isinstance(value, Response) else "",
)
def test_the_tangencies_match_maximas_roots(curve: Saturation, anchor: float) -> None:
    assert curve.tangency() == pytest.approx(2.0 * anchor, rel=1e-13)


@pytest.mark.parametrize(
    "curve", [Hill(2.0, 2.5), Weibull(2.0, 1.8), ChapmanRichards(2.0, 3.0)], ids=name
)
def test_a_closed_form_tangency_agrees_with_the_root_it_replaces(curve: Saturation) -> None:
    assert curve._standard_tangency() == pytest.approx(
        Saturation._standard_tangency(curve), rel=1e-13
    )


def test_a_curve_at_its_supports_end_touches_its_envelope_at_the_kink() -> None:
    # at b = 1 the beta and Kumaraswamy CDFs are z^a up to K and flat after: convex to the corner
    for curve in (BetaCDF(2.0, 3.0, 1.0), Kumaraswamy(2.0, 3.0, 1.0)):
        assert curve.inflection() == 2.0
        assert curve.tangency() == 2.0
        assert float(Envelope(curve)(1.0)) == pytest.approx(0.5, rel=1e-14)


@pytest.mark.parametrize("curve", BOUNDED + CONCAVE_SHAPES, ids=name)
def test_the_envelope_is_concave_and_never_below_its_curve(curve: Saturation) -> None:
    envelope = Envelope(curve)
    touch = curve.tangency()
    spend = jnp.linspace(0.0, 4.0 * max(touch, 2.0), 1601)
    above, on = np.asarray(envelope(spend)), np.asarray(curve(spend))
    assert np.all(above >= on - 1e-15)
    assert np.all(np.diff(above, 2) <= 1e-12)
    beyond = np.asarray(spend) >= touch
    np.testing.assert_array_equal(above[beyond], on[beyond])
    if touch > 0.0:
        chord = float(curve(touch)) / touch
        np.testing.assert_allclose(above[~beyond], chord * np.asarray(spend)[~beyond], rtol=1e-14)
    assert envelope.inflection() == 0.0
    assert envelope.tangency() == 0.0


def test_the_envelopes_slope_in_a_curve_parameter_is_exact_with_the_tangency_held() -> None:
    spend, n, step = 1.0, 3.0, 1e-6
    envelope = Envelope(Hill(2.0, n))
    held = jax.grad(lambda slope: eqx.tree_at(lambda e: e.curve.slope, envelope, slope)(spend))
    moved = (Envelope(Hill(2.0, n + step))(spend) - Envelope(Hill(2.0, n - step))(spend)) / 2 / step
    assert float(held(jnp.asarray(n))) == pytest.approx(float(moved), rel=1e-7)


def test_the_families_nest_where_their_definitions_say() -> None:
    spend = jnp.linspace(0.0, 12.0, 241)

    def same(one: Response, other: Response, rel: float = 1e-14) -> None:
        np.testing.assert_allclose(one(spend), other(spend), rtol=rel, atol=1e-300)

    same(Hill(2.0, 1.0), MichaelisMenten(2.0))
    same(BurrXII(2.0, 3.5, 1.0), Hill(2.0, 3.5))
    for shape_one in (Weibull(2.0, 1.0), GammaCDF(2.0, 1.0), ChapmanRichards(2.0, 1.0)):
        same(shape_one, Exponential(2.0))
    same(Richards(2.0, 3.0, 1.0), Logistic(2.0, 3.0))
    # nu -> 0: Gompertz with b = e^s at scale K / s, the gap shrinking in proportion to nu
    gompertz = Gompertz(2.0 / 3.0, math.exp(3.0))
    gaps = [np.max(np.abs(Richards(2.0, 3.0, nu)(spend) - gompertz(spend))) for nu in (1e-4, 1e-5)]
    assert gaps[0] < 2e-2
    assert gaps[1] == pytest.approx(gaps[0] / 10.0, rel=0.05)


@settings(max_examples=60, deadline=None)
@given(
    family=st.sampled_from([*BOUNDED, Logarithmic(2.0), Power(2.0, 0.6), Ricker(2.0)]),
    currency=st.floats(min_value=1e-3, max_value=1e3),
    share=st.floats(min_value=0.0, max_value=10.0),
)
def test_a_change_of_currency_moves_the_scale_and_nothing_else(
    family: Response, currency: float, share: float
) -> None:
    spend = share * 2.0
    moved = eqx.tree_at(lambda curve: curve.scale, family, family.scale * currency)
    assert float(moved(spend * currency)) == pytest.approx(
        float(family(spend)), rel=1e-12, abs=1e-15
    )
    assert slope(moved, spend * currency) * currency == pytest.approx(
        slope(family, spend), rel=1e-11, abs=1e-15
    )


@pytest.mark.parametrize("curve", S_SHAPED, ids=name)
def test_the_inflection_and_the_tangency_are_in_spend(curve: Saturation) -> None:
    moved = eqx.tree_at(lambda moved: moved.scale, curve, curve.scale * 1000.0)
    assert moved.inflection() == pytest.approx(1000.0 * curve.inflection(), rel=1e-14)
    assert moved.tangency() == pytest.approx(1000.0 * curve.tangency(), rel=1e-12)


@pytest.mark.parametrize("curve", [Logarithmic(2.0), Power(2.0, 0.5), Power(2.0, 1.0)], ids=str)
def test_the_unbounded_baselines_rise_concave_without_a_ceiling(curve: Response) -> None:
    spend = jnp.linspace(0.0, 40.0, 801)
    values = np.asarray(curve(spend))
    assert values[0] == 0.0
    assert np.all(np.diff(values) > 0.0)
    assert np.all(np.diff(values, 2) <= 1e-12)
    assert float(curve(2.0e8)) > 10.0


def test_ricker_peaks_at_its_scale_and_falls_after() -> None:
    curve = Ricker(2.0)
    spend = jnp.linspace(0.0, 20.0, 2001)
    values = np.asarray(curve(spend))
    assert values[0] == 0.0
    assert float(spend[np.argmax(values)]) == pytest.approx(2.0)
    assert float(curve(2.0)) == pytest.approx(1.0, rel=1e-15)
    assert slope(curve, 2.0) == pytest.approx(0.0, abs=1e-15)
    assert np.all(np.diff(values[np.asarray(spend) > 2.0]) < 0.0)


@pytest.mark.parametrize(
    ("build", "field"),
    [
        (lambda: Hill(0.0, 2.0), "scale"),
        (lambda: Hill(-1.0, 2.0), "scale"),
        (lambda: Hill(math.nan, 2.0), "scale"),
        (lambda: Hill(math.inf, 2.0), "scale"),
        (lambda: Hill(jnp.ones(2), 2.0), "scale"),
        (lambda: Hill(1.0, 0.0), "slope"),
        (lambda: Richards(1.0, 3.0, 0.0), "asymmetry"),
        (lambda: Power(1.0, 1.5), "exponent"),
        (lambda: Power(1.0, 0.0), "exponent"),
        (lambda: BetaCDF(1.0, 2.0, 0.5), "b"),
        (lambda: Kumaraswamy(1.0, 2.0, 0.9), "b"),
    ],
)
def test_a_parameter_out_of_range_is_refused(build, field: str) -> None:
    with pytest.raises(ValueError, match=f"^{field}"):
        build()


def test_integer_parameters_become_floats_a_fit_can_move() -> None:
    curve = Hill(2, 3)
    assert jnp.issubdtype(curve.scale.dtype, jnp.floating)
    assert jnp.issubdtype(curve.slope.dtype, jnp.floating)


@pytest.mark.parametrize(
    "curve", [curve for curve in BOUNDED if not isinstance(curve, BetaCDF)], ids=name
)
def test_every_parameter_has_a_finite_slope_down_to_zero_spend(curve: Saturation) -> None:
    spend = jnp.linspace(0.0, 10.0, 41)
    grads = eqx.filter_grad(lambda moved: jnp.sum(moved(spend)))(curve)
    assert all(np.all(np.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(grads))


def test_the_beta_cdf_refuses_a_slope_in_its_shapes_and_names_the_alternative() -> None:
    with pytest.raises(ValueError, match="Kumaraswamy"):
        jax.grad(lambda a: BetaCDF(1.0, a, 2.0)(0.5))(jnp.asarray(2.0))


def test_a_curve_built_inside_a_trace_fits_by_gradient() -> None:
    spend = jnp.linspace(0.0, 6.0, 25)
    target = Hill(2.0, 3.0)(spend)

    @jax.jit
    def loss(log_parameters: jax.Array) -> jax.Array:
        scale, slope = jnp.exp(log_parameters)
        return jnp.sum((Hill(scale, slope)(spend) - target) ** 2)

    at_truth = jnp.log(jnp.array([2.0, 3.0]))
    assert float(loss(at_truth)) == 0.0
    np.testing.assert_allclose(jax.grad(loss)(at_truth), 0.0, atol=1e-15)
    assert np.all(np.isfinite(jax.grad(loss)(at_truth + 0.1)))
