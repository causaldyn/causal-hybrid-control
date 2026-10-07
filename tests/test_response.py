"""The response curves' contract: each family's bounds, bend, tangency and nestings; the planner's
start on them; and the carryover kernels and the return per unit read through them.

The inflections and tangencies are derived in ``validation/response_curves.mac``; the anchors below
are its double-precision roots.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import minimize_scalar
from scipy.stats import weibull_min

from chc.control import LinearConstraint
from chc.cost import QuadraticCost
from chc.mpc import RecedingHorizon
from chc.plan import CausalPlan, causal_plan
from chc.response import (
    Adstock,
    Algebraic,
    Arctan,
    BetaCDF,
    BurrXII,
    Channel,
    ChapmanRichards,
    DelayedAdstock,
    Envelope,
    Exponential,
    GammaCDF,
    GeometricAdstock,
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
    WeibullAdstock,
    _touches,
    contribution,
    marginal_roi,
    relax,
    roi,
    steady_state_marginal_roi,
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
    assert float(curve(touch)) == pytest.approx(touch * slope(curve, touch), rel=1e-12, abs=0.0)


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
    assert curve.tangency() == pytest.approx(2.0 * anchor, rel=1e-13, abs=0.0)


@pytest.mark.parametrize(
    "curve", [Hill(2.0, 2.5), Weibull(2.0, 1.8), ChapmanRichards(2.0, 3.0)], ids=name
)
def test_a_closed_form_tangency_agrees_with_the_root_it_replaces(curve: Saturation) -> None:
    assert curve._standard_tangency() == pytest.approx(
        Saturation._standard_tangency(curve), rel=1e-13, abs=0.0
    )


def test_a_curve_at_its_supports_end_touches_its_envelope_at_the_kink() -> None:
    # at b = 1 the beta and Kumaraswamy CDFs are z^a up to K and flat after: convex to the corner
    for curve in (BetaCDF(2.0, 3.0, 1.0), Kumaraswamy(2.0, 3.0, 1.0)):
        assert curve.inflection() == 2.0
        assert curve.tangency() == 2.0
        assert float(Envelope(curve)(1.0)) == pytest.approx(0.5, rel=1e-14, abs=0.0)


@pytest.mark.parametrize("a", [1.0000000000006113, 1.0 + 2.0**-52, 1.000000002])
def test_a_beta_cdf_at_b_one_turns_at_its_corner_however_close_a_is_to_one(a: float) -> None:
    # a + b - 2 cancels near a = b = 1: these read the inflection 3.6e-4 past the corner, where the
    # tangency then sat and the chord ran below the curve; a zero to divide by; and 1.1e-7 short
    curve = BetaCDF(2.0, a, 1.0)
    assert curve.inflection() == 2.0
    assert curve.tangency() == 2.0
    chord = float(curve(2.0)) / 2.0  # the slope on the left is the chord's to rounding near a = 1
    assert slope(curve, 2.0) <= chord <= slope(curve, math.nextafter(2.0, 0.0)) * (1.0 + 1e-12)


@pytest.mark.parametrize(
    "curve",
    [
        # past b = 1 by less than a double resolves, the curve turns concave within e^(-1/(b - 1))
        # of the corner: the search stopped 1e-15 short of it, where the chord cuts the curve
        BetaCDF(2.0, 1.0000558545842515, 1.000000000149455),
        Kumaraswamy(2.0, 2.999941587134027, 1.000000000000001),
    ],
    ids=name,
)
def test_a_curve_that_turns_only_at_its_corner_touches_there(curve: Saturation) -> None:
    assert curve.tangency() == 2.0


def test_a_gap_that_turns_inside_the_support_is_found_there() -> None:
    # a = b = 1 + 1.07e-4 turns at 0.797 of K; taken at the corner instead, its chord would leave
    # the curve 1.6e-5 above it
    curve = Kumaraswamy(2.0, 1.0001074911103465, 1.0001074911103458)
    touch = curve.tangency()
    assert touch < 2.0
    assert float(curve(touch)) == pytest.approx(touch * slope(curve, touch), rel=1e-12, abs=0.0)


def test_a_tangency_is_the_first_double_past_its_root() -> None:
    # the root is 0.99999999999986329319 (validation/response_curves.mac, STEP 11), 1.4e-13 short of
    # the corner, where the slope turns by 2.6e-4 an ulp: a root search's four ulps or more put the
    # chord's slope outside the curve's on both sides
    curve = Kumaraswamy(1.0, 1000.0, 1.3162271367087017)
    touch = curve.tangency()
    assert touch == 0.9999999999998633
    chord = float(curve(touch)) / touch
    assert slope(curve, touch) <= chord <= slope(curve, math.nextafter(touch, 0.0))


@pytest.mark.parametrize(
    "curve", [*S_SHAPED, BetaCDF(2.0, 3.0, 1.0), Kumaraswamy(2.0, 3.0, 1.0)], ids=name
)
def test_the_touch_from_zero_spend_is_the_tangency(curve: Saturation) -> None:
    # the bisection across many starts closes on the double the search from the origin closes on,
    # and on the corner of a curve convex up to the end of its support
    touch = float(_touches(curve, jnp.zeros(1))[0])
    assert touch == pytest.approx(Saturation._standard_tangency(curve), rel=1e-14, abs=0.0)


@pytest.mark.parametrize(
    ("curve", "start", "anchor"),
    [
        (Hill(1.0, 3.0), 0.5, 0.9590789410597730953),
        (Hill(1.0, 2.0), 0.25, 0.7807764064044151375),
        (Weibull(1.0, 3.0), 0.4, 1.0979963117018852953),
        (Logistic(1.0, 6.0), 0.5, 1.2107061755031192551),
    ],
    ids=lambda value: name(value) if isinstance(value, Response) else "",
)
def test_the_touch_from_a_start_matches_maximas_root(
    curve: Saturation, start: float, anchor: float
) -> None:
    # validation/envelope_on_interval.mac, STEPs 1, 2 and 5
    touch = float(_touches(curve, jnp.asarray([start]))[0])
    assert touch == pytest.approx(anchor, rel=1e-14, abs=0.0)


@pytest.mark.parametrize("curve", S_SHAPED, ids=name)
def test_the_chord_from_each_start_touches_the_curve_nearer_as_the_start_nears_the_bend(
    curve: Saturation,
) -> None:
    bend = curve._standard_inflection()
    starts = np.array([0.0, 0.3, 0.9, 1.0, 1.5]) * bend
    touches = np.asarray(_touches(curve, jnp.asarray(starts)))
    np.testing.assert_array_equal(touches[3:], starts[3:])  # from the bend on, each is its own
    assert np.all(touches[:3] > bend)
    assert np.all(np.diff(touches[:3]) < 0.0)
    g = curve.standard
    for start, touch in zip(starts[:3], touches[:3], strict=True):
        chord = (float(g(touch)) - float(g(start))) / (touch - start)
        assert chord == pytest.approx(float(jax.grad(g)(touch)), rel=1e-12, abs=0.0)


@pytest.mark.parametrize(
    ("curve", "spend", "expected", "rel"),
    [
        (Exponential(1.0), 40.0, math.exp(-40.0), 1e-14),
        (Tanh(1.0), 20.0, 4.0 * math.exp(-40.0) / (1.0 + math.exp(-40.0)) ** 2, 1e-14),
        (Weibull(1.0, 2.0), 10.0, 20.0 * math.exp(-100.0), 1e-14),
        (ChapmanRichards(1.0, 3.0), 40.0, 3.0 * math.exp(-40.0) * math.expm1(-40.0) ** 2, 1e-14),
        (
            Gompertz(1.0, 5.0),
            40.0,
            5.0 * math.exp(-40.0 - 5.0 * math.exp(-40.0)) / -math.expm1(-5.0),
            1e-14,
        ),
        (BurrXII(1.0, 2.0, 20.0), 10.0, 400.0 * 101.0**-21.0, 1e-13),
        # STEP 11 of validation/response_curves.mac; 1 - z^a is rounded first, 7e-11 of itself
        (
            Kumaraswamy(1.0, 1000.0, 1.3162271367087017),
            0.9999999999998632,
            1.000168068085653,
            1e-10,
        ),
    ],
    ids=lambda value: name(value) if isinstance(value, Response) else "",
)
def test_a_slope_in_the_tail_is_the_closed_forms(
    curve: Saturation, spend: float, expected: float, rel: float
) -> None:
    # jax reads expm1's slope as expm1(x) + 1, which cancels once expm1(x) is -1 to rounding: each
    # of these read 0, half its slope, 0.7 % or 1.8e-4 off
    assert slope(curve, spend) == pytest.approx(expected, rel=rel, abs=0.0)


@pytest.mark.parametrize("family", [BetaCDF, Kumaraswamy], ids=lambda family: family.__name__)
@pytest.mark.parametrize(("a", "anchor"), [(2.0, 0.25), (3.0, 0.3849001794597505)])
def test_a_curve_convex_up_to_its_corner_has_the_corners_nonconvexity(
    family: type[Saturation], a: float, anchor: float
) -> None:
    # validation/envelope_nonconvexity.mac STEP 7: z^a under the chord to the corner, 2/(3 sqrt(3))
    # at a = 3. The slope at the inflection read the ceiling's 0 there, and gave no excess at all
    assert family(2.0, a, 1.0).nonconvexity() == pytest.approx(anchor, rel=1e-14, abs=0.0)


def test_kumaraswamys_nonconvexity_runs_on_to_the_corners_as_b_falls_to_one() -> None:
    # validation/envelope_nonconvexity.mac STEP 7: d rho/d b = (1 - z^a) log(1 - z^a) at the peak
    rho, slope_ = 0.3849001794597505, -0.1726141304007209
    for eps in (1e-9, 1e-6):
        assert Kumaraswamy(2.0, 3.0, 1.0 + eps).nonconvexity() == pytest.approx(
            rho + slope_ * eps, rel=1e-12, abs=0.0
        )


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


@pytest.mark.parametrize(
    ("curve", "anchor"),
    [
        (Hill(2.0, 2.0), 0.06744224886812294),
        (Hill(2.0, 3.0), 0.1547005383792515),
        (Hill(2.0, 5.0), 0.292002901391939),
        (Weibull(2.0, 2.0), 0.1082013977031811),
        (Logistic(2.0, 4.0), 0.2027652730511883),
        (Gompertz(2.0, 5.0), 0.1130473059774493),
        (Richards(2.0, 3.0, 0.5), 0.1893298084759638),
        (GammaCDF(2.0, 3.0), 0.1141875232209258),
        (LogNormalCDF(2.0, 0.5), 0.1876091739327744),
        (BurrXII(2.0, 4.0, 1.5), 0.2529083420922978),
        (Kumaraswamy(2.0, 3.0, 2.0), 0.292002901391939),
        (BetaCDF(2.0, 3.0, 2.0), 0.2434505799448503),
        (ChapmanRichards(2.0, 3.0), 0.1025153459085434),
    ],
    ids=lambda value: name(value) if isinstance(value, Response) else "",
)
def test_the_nonconvexity_matches_maximas(curve: Saturation, anchor: float) -> None:
    # validation/envelope_nonconvexity.mac STEP 3
    assert curve.nonconvexity() == pytest.approx(anchor, rel=1e-12, abs=0.0)


@pytest.mark.parametrize(
    ("slope_", "polynomial"),
    [(2.0, [4, 12, 14, -1]), (3.0, [3, 6, -1]), (5.0, [125, 375, 375, -35, -32])],
)
def test_hills_nonconvexity_at_a_whole_slope_is_its_polynomials_root(
    slope_: float, polynomial: list[int]
) -> None:
    # validation/envelope_nonconvexity.mac STEP 2; at slope 3 the root is 2/sqrt(3) - 1
    (root,) = [r.real for r in np.roots(polynomial) if abs(r.imag) < 1e-12 and 0.0 < r.real < 1.0]
    assert Hill(2.0, slope_).nonconvexity() == pytest.approx(root, rel=1e-13, abs=0.0)


@pytest.mark.parametrize("curve", S_SHAPED, ids=name)
def test_the_nonconvexity_is_the_envelopes_largest_excess(curve: Saturation) -> None:
    spend = jnp.linspace(0.0, curve.tangency(), 20001)
    excess = float(jnp.max(Envelope(curve)(spend) - curve(spend)))
    assert excess <= curve.nonconvexity() * (1 + 1e-12)
    assert excess >= curve.nonconvexity() * (1 - 1e-6)


@pytest.mark.parametrize("curve", CONCAVE + CONCAVE_SHAPES, ids=name)
def test_a_concave_curve_has_no_nonconvexity(curve: Saturation) -> None:
    assert curve.nonconvexity() == 0.0
    assert Envelope(curve).nonconvexity() == 0.0


@pytest.mark.parametrize("a", [1.5, 2.0, 3.0, 4.0])
def test_kumaraswamy_at_b_two_has_hills_nonconvexity_at_slope_two_a_less_one(a: float) -> None:
    # validation/envelope_nonconvexity.mac STEP 5: different curves, one excess
    assert Kumaraswamy(2.0, a, 2.0).nonconvexity() == pytest.approx(
        Hill(5.0, 2.0 * a - 1.0).nonconvexity(), rel=1e-12, abs=0.0
    )


def test_the_nonconvexity_does_not_move_with_the_currency() -> None:
    assert Hill(2000.0, 3.0).nonconvexity() == pytest.approx(
        Hill(2.0, 3.0).nonconvexity(), rel=1e-14, abs=0.0
    )


@pytest.mark.parametrize(
    ("curve", "anchor"),
    [
        (Weibull(1.0, 1.0 + 1e-3), 0.0020111162423181239),
        (Weibull(1.0, 1.0 + 1e-4), 0.00020015705713263540),
        (Weibull(1.0, 1.0 + 1e-6), 2.0000249113597843e-06),
        (Weibull(1.0, 1.0 + 1e-9), 2.0000002042076526e-09),
        (Weibull(1.0, 1.0 + 1e-12), 2.0001778012172336e-12),
        (ChapmanRichards(1.0, 1.0 + 1e-3), 0.0019986677767711026),
        (ChapmanRichards(1.0, 1.0 + 1e-4), 0.00019998666777765502),
        (ChapmanRichards(1.0, 1.0 + 1e-6), 1.9999986665032447e-06),
        (ChapmanRichards(1.0, 1.0 + 1e-9), 2.0000001641474084e-09),
        (ChapmanRichards(1.0, 1.0 + 1e-12), 2.0001778011633485e-12),
        (Weibull(1.0, 1e4), 1.0002457076775969),
        (ChapmanRichards(1.0, 1e4), 11.667123907124678),
    ],
    ids=lambda value: name(value) if isinstance(value, Response) else "",
)
def test_a_lambert_tangency_matches_paris_root_near_shape_one_and_far_from_it(
    curve: Saturation, anchor: float
) -> None:
    # validation/envelope_nonconvexity.gp, e^u - 1 = k u at 80 digits: near k = 1 the root is near
    # 2 (k - 1), where both of Lambert W's real branches round to -1
    # pytest.approx's default abs of 1e-12 would pass any of the small anchors
    assert curve.tangency() == pytest.approx(anchor, rel=1e-14, abs=0.0)


# a (1 - a), for a the root of a = e^(2a - 2) below 1 (validation/envelope_nonconvexity.mac STEP 6)
THRESHOLD = 0.16190255947297871
PAST_ONE = 1.0 + 1e-6


@pytest.mark.parametrize(
    ("curve", "law"),
    [
        (Hill(2.0, PAST_ONE), 1.0),
        (Weibull(2.0, PAST_ONE), 2.0),
        (GammaCDF(2.0, PAST_ONE), 2.0),
        (ChapmanRichards(2.0, PAST_ONE), 2.0),
        (BurrXII(2.0, PAST_ONE, 1.5), 1.2),
        (Kumaraswamy(2.0, PAST_ONE, 3.0), 3.0),
        (BetaCDF(2.0, PAST_ONE, 2.0), 4.0),
    ],
    ids=lambda value: name(value) if isinstance(value, Response) else "",
)
def test_the_nonconvexity_vanishes_as_the_square_of_the_shape_past_one(
    curve: Saturation, law: float
) -> None:
    # a curve that starts alpha u - beta u^2 in u = z^(1 + eps) has (alpha^2/beta) a (1 - a) eps^2
    eps = PAST_ONE - 1.0
    assert curve.nonconvexity() == pytest.approx(law * THRESHOLD * eps**2, rel=1e-4, abs=0.0)


def test_gompertzs_nonconvexity_vanishes_as_the_cube_of_its_displacement_past_one() -> None:
    # it starts with a slope, 1/(e - 1) at b = 1, which its convex stretch only bends
    eps = 1e-4
    assert Gompertz(2.0, 1.0 + eps).nonconvexity() == pytest.approx(
        eps**3 / (12 * (math.e - 1)), rel=1e-3, abs=0.0
    )


@pytest.mark.parametrize("curve", [Gompertz(2.0, 1.0 + 1e-12), Logistic(2.0, 1e-8)], ids=name)
def test_a_convex_stretch_within_rounding_of_none_has_no_nonconvexity(curve: Saturation) -> None:
    assert curve.tangency() > 0.0
    assert curve.nonconvexity() == 0.0


def test_the_envelopes_slope_in_a_curve_parameter_is_exact_with_the_tangency_held() -> None:
    spend, n, step = 1.0, 3.0, 1e-6
    envelope = Envelope(Hill(2.0, n))
    held = jax.grad(lambda slope: eqx.tree_at(lambda e: e.curve.slope, envelope, slope)(spend))
    moved = (Envelope(Hill(2.0, n + step))(spend) - Envelope(Hill(2.0, n - step))(spend)) / 2 / step
    assert float(held(jnp.asarray(n))) == pytest.approx(float(moved), rel=1e-7)


def test_envelopes_with_different_tangencies_share_one_tree_structure() -> None:
    # so a program compiled for one posterior draw of an S-shaped curve serves every other draw
    one, other = Envelope(Hill(2.0, 3.0)), Envelope(Hill(1.5, 4.5))
    assert float(one.touch) != float(other.touch)
    assert jax.tree_util.tree_structure(one) == jax.tree_util.tree_structure(other)


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


def test_richards_keeps_its_floor_and_slope_where_nu_e_to_the_s_is_past_a_double() -> None:
    # 1000 e^709 at zero spend is 8.2e310, and its floor 0.49 (validation/response_curves.mac,
    # STEP 8): read as 0, the curve at half its scale was 0.697
    assert float(Richards(1.0, 709.0, 1000.0)(0.5)) == pytest.approx(
        0.4067401462626451, rel=1e-14, abs=0.0
    )
    # a logistic at every steepness, with the logistic's slope of 0 at zero spend, not nan
    steep = Richards(1.0, 1000.0, 1.0)
    assert slope(steep, 0.0) == 0.0
    assert steep.nonconvexity() == pytest.approx(
        Logistic(1.0, 1000.0).nonconvexity(), rel=1e-14, abs=0.0
    )


@pytest.mark.parametrize(
    ("steepness", "asymmetry", "anchors"),
    [
        # the floor 0.971: in a vector the curve read 3.8e-15 at zero spend, where alone it read 0
        (
            0.001000053526891461,
            176.627704067942,
            [1.891480872416736e-16, 1.891480872416741e-10, 1.891480872422056e-7],
        ),
        # the floor 1 - 1.4e-5: the curve read 0 up to 1e-6 and erred by 7.8e-4 at 1e-3
        (1e-6, 1e6, [7.238183079151661e-20, 7.238183079151661e-14, 7.238183079151662e-11]),
    ],
    ids=["floor 0.971", "floor 1 - 1.4e-5"],
)
def test_richards_keeps_its_relative_accuracy_near_zero_spend_where_its_floor_nears_one(
    steepness: float, asymmetry: float, anchors: list[float]
) -> None:
    # the curve less its floor, over 1 less it, carried the floor's rounding up as 1 / (1 - floor)
    # (validation/response_curves.mac, STEP 12)
    curve = Richards(1.0, steepness, asymmetry)
    values = np.asarray(curve(jnp.asarray([0.0, 1e-12, 1e-6, 1e-3])))
    assert values[0] == 0.0
    np.testing.assert_allclose(values[1:], anchors, rtol=4e-15, atol=0.0)


@pytest.mark.parametrize("spend", [2.10546875, 4.0])
def test_weibulls_slope_is_zero_not_nan_where_z_to_the_k_is_past_a_double(spend: float) -> None:
    # at k = 947, k z^(k - 1) passes a double from z = 2.10 and z^k from 2.12; the curve is its
    # ceiling there, and its slope e^-inf
    curve = Weibull(1.0, 947.4635256553754)
    assert float(curve(spend)) == 1.0
    assert slope(curve, spend) == 0.0


def test_weibulls_slope_short_of_its_ceiling_is_the_closed_forms() -> None:
    z, k = 1.001, 850.0
    assert slope(Weibull(1.0, k), z) == pytest.approx(
        k * z ** (k - 1.0) * math.exp(-(z**k)), rel=1e-12, abs=0.0
    )


def test_the_beta_cdfs_slope_is_finite_where_its_normaliser_is_past_a_double() -> None:
    # 1/B(1000, 1000) = e^1388.5 (validation/response_curves.mac, STEP 10): at zero spend it met
    # z^999 = 0 into nan
    curve = BetaCDF(1.0, 1000.0, 1000.0)
    assert slope(curve, 0.0) == 0.0
    assert slope(curve, 0.5) == pytest.approx(35.67802229170864, rel=1e-12, abs=0.0)


def test_burr_xii_keeps_its_value_and_slope_where_z_to_the_c_is_past_a_double() -> None:
    # at c = 850, z^c passes a double from z = 2.30, where the curve is 0.51
    # (validation/response_curves.mac, STEP 9): read as its ceiling, g(4) was 1 and the slope nan,
    # and no tangency was found
    curve = BurrXII(1.0, 850.0, 0.001)
    assert float(curve(4.0)) == pytest.approx(0.6922138966637709, rel=1e-14, abs=0.0)
    assert slope(curve, 4.0) == pytest.approx(0.06540454695894868, rel=1e-13, abs=0.0)
    assert curve.tangency() == pytest.approx(2.062147283815243, rel=1e-13, abs=0.0)


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
    assert float(curve(2.0)) == pytest.approx(1.0, rel=1e-15, abs=0.0)
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
    "curve",
    [
        curve
        for curve in [*BOUNDED, *CONCAVE_SHAPES, Power(2.0, 0.5)]
        if not isinstance(curve, BetaCDF)
    ],
    ids=name,
)
def test_every_parameter_has_a_finite_slope_down_to_zero_spend(curve: Response) -> None:
    spend = jnp.linspace(0.0, 10.0, 41)
    grads = eqx.filter_grad(lambda moved: jnp.sum(moved(spend)))(curve)
    assert all(np.all(np.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(grads))
    # the curve is 0 at zero spend whatever its parameters, and so is every slope in them there
    at_zero = eqx.filter_grad(lambda moved: moved(0.0))(curve)
    assert not any(np.any(leaf) for leaf in jax.tree_util.tree_leaves(at_zero))


# the families that rise from zero spend like a power of it, at shapes below 1, at 1, between 1 and
# 2, and past 2
AT_ZERO = [
    pytest.param(curve, id=f"{name(curve)}-{shape}")
    for shape in (0.5, 1.0, 1.5, 2.5)
    for curve in (
        Hill(2.0, shape),
        Weibull(2.0, shape),
        ChapmanRichards(2.0, shape),
        BurrXII(2.0, shape, 2.0),
        Kumaraswamy(2.0, shape, 2.0),
        GammaCDF(2.0, shape),
        BetaCDF(2.0, shape, 2.0),
        *((Power(2.0, shape),) if shape <= 1.0 else ()),
    )
]


@pytest.mark.parametrize("curve", AT_ZERO)
def test_a_slope_at_zero_spend_is_finite_and_nothing_where_no_spend_reaches(
    curve: Response,
) -> None:
    # a period no spend reaches, as after a plan through a kernel with no carryover, reads the
    # curve at zero spend through a zero weight, and so does the slope of spend / K in K at zero
    # spend; below slope 1 the curve's slope there is infinite, and 0 * inf was nan
    assert math.isfinite(slope(curve, 0.0))
    assert math.isfinite(curvature(curve, 0.0))

    def unreached(rate: jax.Array) -> jax.Array:
        return curve(rate * 0.0)

    def at_scale(scale: jax.Array) -> jax.Array:
        return eqx.tree_at(lambda moved: moved.scale, curve, scale)(0.0)

    for read in (unreached, at_scale):
        assert float(jax.grad(read)(jnp.asarray(2.0))) == 0.0
        assert float(jax.grad(jax.grad(read))(jnp.asarray(2.0))) == 0.0


@pytest.mark.parametrize(
    ("curve", "expected"),
    [
        (Hill(2.0, 1.0), -0.5),
        (Hill(2.0, 2.0), 0.5),
        (Weibull(2.0, 1.0), -0.25),
        (Weibull(2.0, 2.0), 0.5),
        (ChapmanRichards(2.0, 1.0), -0.25),
        (ChapmanRichards(2.0, 2.0), 0.5),
        (BurrXII(2.0, 1.0, 2.0), -1.5),
        (BurrXII(2.0, 2.0, 3.0), 1.5),
        (Kumaraswamy(2.0, 1.0, 3.0), -1.5),
        (Kumaraswamy(2.0, 2.0, 3.0), 1.5),
        (GammaCDF(2.0, 1.0), -0.25),
        (GammaCDF(2.0, 2.0), 0.25),
        (BetaCDF(2.0, 1.0, 2.0), -0.5),
        (BetaCDF(2.0, 2.0, 3.0), 3.0),
        (Power(2.0, 1.0), 0.0),
    ],
    ids=lambda value: name(value) if isinstance(value, Response) else "",
)
def test_the_curvature_at_zero_spend_is_the_curves_own_where_it_is_finite(
    curve: Response, expected: float
) -> None:
    # Maxima's g''(0) over K^2 (validation/response_curves.mac, STEP 6): at shape 1 JAX's own
    # second slope of the power is 1 * 0 * 0^-1, and it and the gamma and beta densities' slopes
    # at every shape were nan
    assert curvature(curve, 0.0) == pytest.approx(expected, rel=1e-14, abs=1e-300)


@pytest.mark.parametrize(
    "curve",
    [
        Hill(2.0, 0.7),
        Weibull(2.0, 0.5),
        ChapmanRichards(2.0, 0.5),
        BurrXII(2.0, 0.5, 2.0),
        Kumaraswamy(2.0, 0.8, 2.0),
        GammaCDF(2.0, 0.5),
        BetaCDF(2.0, 0.5, 2.0),
        Power(2.0, 0.5),
    ],
    ids=name,
)
def test_below_slope_one_the_slope_at_zero_spend_is_the_steepest_a_plan_reads(
    curve: Response,
) -> None:
    # the slope there is infinite; read a machine epsilon of a scale off zero it is finite and
    # steeper than anywhere from there on, so a plan at zero spend still moves off it
    spend = 2.0 * jnp.geomspace(1e-15, 1e2, 200)
    beyond = np.asarray(jax.vmap(jax.grad(curve))(spend))
    assert math.isfinite(slope(curve, 0.0))
    assert slope(curve, 0.0) > beyond.max()


@pytest.mark.parametrize(
    ("curve", "shape", "f", "r"),
    [
        (Hill(2.0, 1.0), lambda curve: curve.slope, 1.0, 1.0),
        (Weibull(2.0, 1.0), lambda curve: curve.shape, 1.0, 1.0),
        (ChapmanRichards(2.0, 1.0), lambda curve: curve.power, 1.0, 1.0),
        (BurrXII(2.0, 1.0, 2.0), lambda curve: curve.slope, 2.0, 2.0),
        (Kumaraswamy(2.0, 1.0, 2.0), lambda curve: curve.a, 2.0, 2.0),
        (GammaCDF(2.0, 1.0), lambda curve: curve.shape, 1.0, np.euler_gamma),
        (Power(2.0, 1.0), lambda curve: curve.exponent, 1.0, 1.0),
    ],
    ids=lambda value: name(value) if isinstance(value, Response) else "",
)
def test_a_slope_infinite_at_zero_spend_is_read_a_machine_epsilon_of_the_scale_off_it(
    curve: Response, shape: Callable[[Response], jax.Array], f: float, r: float
) -> None:
    # at shape 1 the slope moved by the shape is (f log z + r) / K, -inf at zero spend
    # (validation/response_curves.mac, STEP 7), and read at z = eps; where a power of 0 at zero
    # spend was not read off it, its log was 0, and the slope moved by the shape r / K
    def moved(value: jax.Array) -> jax.Array:
        return jax.grad(eqx.tree_at(shape, curve, value))(jnp.asarray(0.0))

    at_eps = (f * math.log(np.finfo(np.float64).eps) + r) / 2.0
    assert float(jax.grad(moved)(jnp.asarray(1.0))) == pytest.approx(at_eps, rel=1e-14, abs=0.0)


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
    # not 0 exactly: jax 0.4.30's exp(log(2)) is a last bit off 2
    assert float(loss(at_truth)) < 1e-30
    np.testing.assert_allclose(jax.grad(loss)(at_truth), 0.0, atol=1e-15)
    assert np.all(np.isfinite(jax.grad(loss)(at_truth + 0.1)))


def test_relax_swaps_each_curve_that_starts_convex_and_leaves_the_rest() -> None:
    concave, rate = MichaelisMenten(1.0), jnp.ones(2)
    model = (Hill(1.0, 3.0), concave, {"inner": Weibull(1.0, 2.0)}, rate)
    relaxed = relax(model)
    assert isinstance(relaxed[0], Envelope)
    assert isinstance(relaxed[2]["inner"], Envelope)
    assert relaxed[1] is concave
    assert relaxed[3] is rate
    assert relax(relaxed) is relaxed
    only_concave = (concave, Hill(1.0, 0.8), Ricker(1.0))
    assert relax(only_concave) is only_concave


class _Revenue(eqx.Module):
    """A budget split between an S-curve and a Michaelis-Menten channel, as a one-step plant.

    The revenue's rate does not read the state, so one RK4 step of ``dt = 1`` is the static
    allocation exactly, and a target above any reachable revenue makes the cheapest plan the one
    with the most revenue.
    """

    curve: Saturation
    other: Saturation = MichaelisMenten(100.0)

    def __call__(self, t: float | jax.Array, x: jax.Array, u: jax.Array) -> jax.Array:
        return jnp.array([1000.0 * self.curve(u[0]) + 300.0 * self.other(u[1])])


REVENUE_COST = QuadraticCost(
    Q=jnp.zeros((1, 1)), R=jnp.zeros((2, 2)), Qf=jnp.eye(1), x_target=jnp.array([1e4])
)
BUDGET = LinearConstraint(np.ones((1, 2)), np.array([-np.inf]), np.array([300.0]))


def _best_split(model: _Revenue) -> tuple[float, float]:
    """The most revenue a budget of 300 buys, and the S-curve's share of it: a grid, refined."""
    grid = np.linspace(0.0, 300.0, 30001)
    revenue = np.asarray(1000.0 * model.curve(grid) + 300.0 * model.other(300.0 - grid))
    start = grid[np.argmax(revenue)]
    refined = minimize_scalar(
        lambda spend: -float(1000.0 * model.curve(spend) + 300.0 * model.other(300.0 - spend)),
        bounds=(max(start - 0.02, 0.0), min(start + 0.02, 300.0)),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return max(-float(refined.fun), float(revenue.max())), float(refined.x)


def _revenue(model: _Revenue, plan: CausalPlan) -> float:
    return float(model(0.0, jnp.zeros(1), plan.actions[0])[0])


@pytest.mark.parametrize(
    "curve",
    [
        Hill(100.0, 3.0),
        Weibull(100.0, 1.8),
        Logistic(100.0, 4.0),
        Gompertz(100.0, 5.0),
        Richards(100.0, 3.0, 0.5),
        ChapmanRichards(100.0, 3.0),
        GammaCDF(100.0, 3.0),
        LogNormalCDF(100.0, 0.5),
        BurrXII(100.0, 4.0, 1.5),
        BetaCDF(200.0, 3.0, 2.0),
        Kumaraswamy(200.0, 3.0, 2.0),
    ],
    ids=name,
)
def test_a_plan_from_zero_spend_reaches_the_best_split_on_every_s_curve(curve: Saturation) -> None:
    model = _Revenue(curve)
    best, share = _best_split(model)
    plan = causal_plan(model, jnp.zeros(1), REVENUE_COST, 1.0, 1, 0.0, 300.0, constraints=(BUDGET,))
    assert plan.solver_status == "converged"
    assert _revenue(model, plan) == pytest.approx(best, rel=1e-9)
    assert plan.relaxed_cost is not None
    gap = plan.task_cost - plan.relaxed_cost
    assert gap >= -1e-12 * plan.task_cost
    if share >= curve.tangency():
        # the best split spends where the envelope is the curve, so the relaxation is tight
        assert gap <= 1e-12 * plan.task_cost
    else:
        assert gap > 0.0


def test_zero_spend_alone_stops_at_the_greedy_corner_of_a_hill_curve() -> None:
    # the trap the envelope start removes: h'(0) = 0, so the descent from zeros reports convergence
    model = _Revenue(Hill(100.0, 3.0))
    alone = causal_plan(
        model,
        jnp.zeros(1),
        REVENUE_COST,
        1.0,
        1,
        0.0,
        300.0,
        constraints=(BUDGET,),
        warm_start=jnp.zeros((1, 2)),
    )
    assert alone.solver_status == "converged"
    assert _revenue(model, alone) == pytest.approx(225.0, rel=1e-12)
    assert alone.relaxed_cost is None


def test_a_receding_horizon_relaxes_its_cold_start_only() -> None:
    model = _Revenue(Hill(100.0, 3.0))
    controller = RecedingHorizon(model, REVENUE_COST, 1.0, 1, 0.0, 300.0, constraints=(BUDGET,))
    cold = controller.step(jnp.zeros(1))
    assert cold.relaxed_cost is not None
    assert _revenue(model, cold) == pytest.approx(_best_split(model)[0], rel=1e-9)
    assert controller.step(jnp.zeros(1)).relaxed_cost is None


def test_a_model_with_no_curve_to_relax_plans_as_before() -> None:
    model = _Revenue(MichaelisMenten(100.0))
    plan = causal_plan(model, jnp.zeros(1), REVENUE_COST, 1.0, 1, 0.0, 300.0, constraints=(BUDGET,))
    assert plan.relaxed_cost is None


KERNELS = [
    GeometricAdstock(0.6, length=8, normalized=False),
    GeometricAdstock(0.6, length=8, normalized=True),
    DelayedAdstock(0.7, 2.5, length=10, normalized=False),
    DelayedAdstock(0.7, 2.5, length=10, normalized=True),
    WeibullAdstock(1.5, 3.0, length=12, normalized=False),
    WeibullAdstock(0.8, 2.0, length=12, normalized=True),
]
SPEND = jnp.asarray(np.random.default_rng(7).gamma(2.0, 50.0, size=60))


def kernel_name(kernel: Adstock) -> str:
    return f"{type(kernel).__name__}-{'normalized' if kernel.normalized else 'raw'}"


@pytest.mark.parametrize("kernel", KERNELS, ids=kernel_name)
def test_a_kernel_reads_its_own_length_of_spend_and_no_more(kernel: Adstock) -> None:
    weights, spend = np.asarray(kernel.weights()), np.asarray(SPEND)
    by_hand = [
        sum(weights[lag] * spend[t - lag] for lag in range(min(t + 1, len(weights))))
        for t in range(len(spend))
    ]
    np.testing.assert_allclose(kernel(SPEND), by_hand, rtol=1e-13)
    # a shorter series is a prefix: the kernel's length is its own, not the series'
    np.testing.assert_array_equal(kernel(SPEND[:23]), kernel(SPEND)[:23])
    both = kernel(jnp.stack([SPEND, 3.0 * SPEND], 1))
    np.testing.assert_allclose(both[:, 1], 3.0 * kernel(SPEND), rtol=1e-14)


def test_the_kernels_weights_are_their_definitions() -> None:
    geometric = GeometricAdstock(0.6, length=8, normalized=False).weights()
    np.testing.assert_allclose(geometric, 0.6 ** np.arange(8), rtol=1e-14)
    delayed = DelayedAdstock(0.7, 2.0, length=6, normalized=False).weights()
    np.testing.assert_allclose(delayed, 0.7 ** ((np.arange(6) - 2.0) ** 2), rtol=1e-15)
    assert int(np.argmax(delayed)) == 2
    # the survival of each lag from SciPy's Weibull, rather than the running sum the kernel keeps
    survival = 1.0 - weibull_min.cdf(np.arange(1, 12), 1.5, scale=3.0)
    np.testing.assert_allclose(
        WeibullAdstock(1.5, 3.0, length=12, normalized=False).weights(),
        np.cumprod(np.concatenate([[1.0], survival])),
        rtol=1e-13,
    )
    for kernel in KERNELS:
        if kernel.normalized:
            assert float(jnp.sum(kernel.weights())) == pytest.approx(1.0, rel=1e-14, abs=0.0)


def test_a_kernel_without_carryover_has_a_finite_slope_in_its_retention() -> None:
    def total(retention: jax.Array) -> jax.Array:
        return jnp.sum(GeometricAdstock(retention, length=4, normalized=False)(SPEND))

    weights = GeometricAdstock(0.0, length=4, normalized=False).weights()
    np.testing.assert_array_equal(weights, [1.0, 0.0, 0.0, 0.0])
    # at r = 0 only the first lag moves: each period's adstock gains the spend before it
    assert float(jax.grad(total)(0.0)) == pytest.approx(float(jnp.sum(SPEND[:-1])), rel=1e-14)


def test_a_linear_channel_returns_its_coefficient_times_the_kernels_sum() -> None:
    kernel = GeometricAdstock(0.6, length=8, normalized=False)
    channel = Channel(kernel, Power(1.0, 1.0), 2.0)
    padded = jnp.concatenate([SPEND, jnp.zeros(7)])  # every period's carryover inside the series
    whole = 2.0 * float(jnp.sum(kernel.weights()))
    assert roi(channel, padded, slice(10, 20)) == pytest.approx(whole, rel=1e-13, abs=0.0)
    assert marginal_roi(channel, padded, slice(10, 20)) == pytest.approx(whole, rel=1e-13, abs=0.0)
    assert steady_state_marginal_roi(channel, 40.0) == pytest.approx(whole, rel=1e-13, abs=0.0)


def test_a_contribution_is_the_return_in_its_periods_from_all_the_spend_before() -> None:
    channel = Channel(DelayedAdstock(0.7, 1.5, length=10, normalized=True), Hill(80.0, 2.5), 3.0)
    # over the whole series it is the ROI times what was spent, since no spend returns nothing
    whole = roi(channel, SPEND) * float(jnp.sum(SPEND))
    assert contribution(channel, SPEND) == pytest.approx(whole, rel=1e-13)
    # a window that spends nothing still carries over what came before it
    quiet = SPEND.at[30:40].set(0.0)
    assert contribution(channel, quiet, slice(30, 40)) > 0.0
    assert contribution(channel, quiet, slice(30, 40), revenue_per_kpi=4.0) == pytest.approx(
        4.0 * contribution(channel, quiet, slice(30, 40)), rel=1e-15, abs=0.0
    )


def test_the_marginal_roi_is_the_limit_of_scaling_the_windows_spend() -> None:
    channel = Channel(DelayedAdstock(0.7, 1.5, length=10, normalized=True), Hill(80.0, 2.5), 3.0)
    window, step = slice(20, 35), 1e-6
    inside = jnp.zeros(60).at[window].set(1.0)
    scaled = SPEND * (1.0 + step * inside)
    moved = float(jnp.sum(channel(scaled) - channel(SPEND)))
    moved /= step * float(jnp.sum(SPEND * inside))
    assert marginal_roi(channel, SPEND, window) == pytest.approx(moved, rel=1e-6)


def test_the_steady_state_marginal_roi_is_the_slope_of_a_long_rollout() -> None:
    # the recursion a_t = s_t + r a_{t-1} rolled out at a constant level, the last period's return
    # differenced in the level; and the kernel's closed form b h'(s / (1 - r)) / (1 - r)
    r, beta, curve, level, step = 0.6, 2.0, Hill(300.0, 2.0), 40.0, 1e-4

    def settled(spend: float) -> float:
        stock = 0.0
        for _ in range(400):
            stock = spend + r * stock
        return beta * float(curve(stock))

    moved = (settled(level + step) - settled(level - step)) / (2.0 * step)
    channel = Channel(GeometricAdstock(r, length=400, normalized=False), curve, beta)
    closed = beta * float(jax.grad(curve)(level / (1.0 - r))) / (1.0 - r)
    assert steady_state_marginal_roi(channel, level) == pytest.approx(closed, rel=1e-12, abs=0.0)
    assert steady_state_marginal_roi(channel, level) == pytest.approx(moved, rel=1e-7)


@pytest.mark.parametrize("currency", [1e-3, 0.37, 1e3])
def test_every_return_per_unit_is_invariant_to_the_currency(currency: float) -> None:
    def channel(rate: float) -> Channel:
        curve = LogNormalCDF(90.0 * rate, 0.6)
        return Channel(WeibullAdstock(1.5, 3.0, length=12, normalized=False), curve, 5.0 * rate)

    home, abroad, window = channel(1.0), channel(currency), slice(10, 40)
    assert roi(abroad, SPEND * currency, window) == pytest.approx(
        roi(home, SPEND, window), rel=1e-12, abs=0.0
    )
    assert marginal_roi(abroad, SPEND * currency, window) == pytest.approx(
        marginal_roi(home, SPEND, window), rel=1e-12, abs=0.0
    )
    assert steady_state_marginal_roi(abroad, 40.0 * currency) == pytest.approx(
        steady_state_marginal_roi(home, 40.0), rel=1e-12, abs=0.0
    )


def test_revenue_per_kpi_turns_a_kpi_return_into_revenue() -> None:
    channel = Channel(GeometricAdstock(0.5, length=6, normalized=True), Tanh(100.0), 1.5)
    assert roi(channel, SPEND, revenue_per_kpi=4.0) == pytest.approx(
        4.0 * roi(channel, SPEND), rel=1e-15, abs=0.0
    )
    assert marginal_roi(channel, SPEND, revenue_per_kpi=4.0) == pytest.approx(
        4.0 * marginal_roi(channel, SPEND), rel=1e-15, abs=0.0
    )
    assert steady_state_marginal_roi(channel, 40.0, revenue_per_kpi=4.0) == pytest.approx(
        4.0 * steady_state_marginal_roi(channel, 40.0), rel=1e-15, abs=0.0
    )


def test_a_return_per_unit_is_refused_where_it_is_undefined() -> None:
    channel = Channel(GeometricAdstock(0.5, length=6, normalized=True), Tanh(100.0), 1.5)
    with pytest.raises(ValueError, match=r"spends 0\.0"):
        roi(channel, SPEND.at[10:20].set(0.0), slice(10, 20))
    with pytest.raises(ValueError, match="one channel at a time"):
        marginal_roi(channel, jnp.stack([SPEND, SPEND], 1))
    with pytest.raises(ValueError, match=r"^level"):
        steady_state_marginal_roi(channel, -1.0)
    with pytest.raises(ValueError, match="a kernel reads"):
        channel.kernel(jnp.ones((4, 3, 2)))


@pytest.mark.parametrize(
    ("build", "field"),
    [
        (lambda: GeometricAdstock(0.6, length=0, normalized=True), "length"),
        (lambda: GeometricAdstock(0.6, length=8.0, normalized=True), "length"),
        (lambda: GeometricAdstock(1.5, length=8, normalized=True), "retention"),
        (lambda: GeometricAdstock(-0.1, length=8, normalized=True), "retention"),
        (lambda: DelayedAdstock(0.0, 1.0, length=8, normalized=True), "retention"),
        (lambda: DelayedAdstock(0.5, -1.0, length=8, normalized=True), "delay"),
        (lambda: WeibullAdstock(0.0, 2.0, length=8, normalized=True), "shape"),
        (lambda: WeibullAdstock(1.0, 0.0, length=8, normalized=True), "scale"),
        (lambda: Channel(KERNELS[0], Tanh(1.0), math.nan), "coefficient"),
    ],
)
def test_a_kernel_or_channel_out_of_range_is_refused(build, field: str) -> None:
    with pytest.raises(ValueError, match=f"^{field}"):
        build()


def test_a_channel_fits_through_its_kernel_and_curve() -> None:
    truth = Channel(GeometricAdstock(0.6, length=8, normalized=True), Hill(120.0, 2.0), 3.0)
    observed = truth(SPEND)

    @jax.jit
    def loss(parameters: jax.Array) -> jax.Array:
        retention, log_scale, log_slope, coefficient = parameters
        model = Channel(
            GeometricAdstock(retention, length=8, normalized=True),
            Hill(jnp.exp(log_scale), jnp.exp(log_slope)),
            coefficient,
        )
        return jnp.sum((model(SPEND) - observed) ** 2)

    at_truth = jnp.array([0.6, math.log(120.0), math.log(2.0), 3.0])
    assert float(loss(at_truth)) == pytest.approx(0.0, abs=1e-20)
    np.testing.assert_allclose(jax.grad(loss)(at_truth), 0.0, atol=1e-10)
    assert np.all(np.isfinite(jax.grad(loss)(at_truth + 0.05)))
