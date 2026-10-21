"""chc.allocation.decision_weight: what a plan made on wrong channels loses, to second order.

Checked against Maxima's closed forms (``validation/allocation_decision_weight.mac``) for two
exponential and two Michaelis-Menten channels, and on channels with carryover against the loss
itself: the plan re-made on perturbed channels and scored on the true ones, each channel run over
the history, the plan and the tail as one series.
"""

import importlib

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from chc.allocation import allocate, decision_weight
from chc.response import Channel, Exponential, GeometricAdstock, Hill, MichaelisMenten, Tanh

# Public as jax.enable_x64 from jax 0.8.0; the floor, 0.4.30, has only jax.experimental.enable_x64,
# which jax 0.11 no longer has.
if hasattr(jax, "enable_x64"):
    enable_x64 = jax.enable_x64
else:
    enable_x64 = importlib.import_module("jax.experimental").enable_x64

ONE = GeometricAdstock(0.0, length=1, normalized=False)  # no carryover
PERIODS = 13
KERNELS = (
    GeometricAdstock(0.3, length=6, normalized=True),
    GeometricAdstock(0.6, length=8, normalized=True),
    GeometricAdstock(0.1, length=4, normalized=True),
)
HISTORY = np.random.default_rng(5).gamma(4.0, 20.0, size=(30, 3))
LOWER = np.array([30.0, 20.0, 10.0])
UPPER = np.array([400.0, 300.0, 250.0])
BUDGET = PERIODS * 180.0
CHANNELS = (
    Channel(KERNELS[0], Tanh(250.0), 900.0),
    Channel(KERNELS[1], MichaelisMenten(80.0), 700.0),
    Channel(KERNELS[2], Exponential(90.0), 450.0),
)
# W at b1 = 900, K1 = 50, b2 = 600, K2 = 80, B = 150, rows and columns b1 K1 b2 K2 (STEP 1c, 2c)
MAXIMA = {
    Exponential: [
        [1.258400722377557e-4, 1.568814282940989e-3, -1.887601083566335e-4, 2.58892876105136e-4],
        [1.568814282940989e-3, 1.955798507258982e-2, -2.353221424411483e-3, 3.227547748208865e-3],
        [-1.887601083566335e-4, -2.353221424411483e-3, 2.831401625349503e-4, -3.88339314157704e-4],
        [2.58892876105136e-4, 3.227547748208865e-3, -3.88339314157704e-4, 5.32624625098393e-4],
    ],
    MichaelisMenten: [
        [1.024598768837941e-4, 5.053350379546509e-4, -1.536898153256912e-4, 1.437521242082376e-4],
        [5.053350379546509e-4, 2.492326834183604e-3, -7.580025569319765e-4, 7.089895806259867e-4],
        [-1.536898153256912e-4, -7.580025569319765e-4, 2.305347229885369e-4, -2.156281863123564e-4],
        [1.437521242082376e-4, 7.089895806259867e-4, -2.156281863123564e-4, 2.016855167395683e-4],
    ],
}


@pytest.mark.parametrize("family", [Exponential, MichaelisMenten])
def test_the_weight_is_maxima_s_hessian_of_the_loss(family) -> None:
    channels = (Channel(ONE, family(50.0), 900.0), Channel(ONE, family(80.0), 600.0))
    weight = decision_weight(channels, 150.0, 1, lower=[0.0, 0.0], upper=[150.0, 150.0])
    order = ["0.coefficient", "0.curve.scale", "1.coefficient", "1.curve.scale"]
    index = [weight.parameters.index(name) for name in order]
    np.testing.assert_allclose(weight.matrix[np.ix_(index, index)], MAXIMA[family], rtol=1e-8)
    unused = [weight.parameters.index(f"{c}.kernel.retention") for c in (0, 1)]
    assert not np.any(weight.matrix[unused])  # a kernel of one period reads no retention
    assert weight.pinned == ()


@pytest.mark.parametrize("family", [MichaelisMenten, lambda scale: Hill(scale, 1.0)])
def test_a_kernel_with_no_carryover_weighs_as_one_with_no_tail(family) -> None:
    # at retention 0 the periods after the plan see no spend, and the curvature reads the curve
    # there through a zero weight; at slope 1 a Hill's curvature at zero spend was 0 * inf = nan
    kernel = GeometricAdstock(0.0, length=8, normalized=False)
    channels = (Channel(kernel, family(50.0), 900.0), Channel(kernel, family(80.0), 600.0))
    weight = decision_weight(channels, 150.0, 1, lower=[0.0, 0.0], upper=[150.0, 150.0])
    order = ["0.coefficient", "0.curve.scale", "1.coefficient", "1.curve.scale"]
    index = [weight.parameters.index(name) for name in order]
    np.testing.assert_allclose(
        weight.matrix[np.ix_(index, index)], MAXIMA[MichaelisMenten], rtol=1e-8
    )
    assert np.all(np.isfinite(weight.matrix))


def _worth(channels, rates) -> float:
    """What ``rates`` return on ``channels``, each run over the history, the plan and its tail."""
    total = 0.0
    for column, (channel, rate) in enumerate(zip(channels, rates, strict=True)):
        series = np.concatenate(
            [HISTORY[:, column], np.full(PERIODS, rate), np.zeros(channel.kernel.length - 1)]
        )
        total += float(jnp.sum(channel(jnp.asarray(series))[HISTORY.shape[0] :]))
    return total


def _moved(channels, direction, step):
    """The channels with every inexact parameter moved by ``step * direction``, in order."""
    moved, start = [], 0
    for channel in channels:
        parameters, static = eqx.partition(channel, eqx.is_inexact_array)
        flat, unravel = ravel_pytree(parameters)
        stop = start + flat.size
        moved.append(eqx.combine(unravel(flat + step * direction[start:stop]), static))
        start = stop
    return tuple(moved)


def _loss(channels, direction, step) -> float:
    """The worth lost by the plan made on the moved channels, scored on the true ones."""
    box = {"lower": LOWER, "upper": UPPER, "history": HISTORY}
    best = allocate(channels, BUDGET, PERIODS, **box).spend
    planned = allocate(_moved(channels, direction, step), BUDGET, PERIODS, **box).spend
    return _worth(channels, best) - _worth(channels, planned)


def _directions():
    """The weight on ``CHANNELS``, and three random directions scaled to each parameter's size."""
    weight = decision_weight(CHANNELS, BUDGET, PERIODS, lower=LOWER, upper=UPPER, history=HISTORY)
    size = np.abs(ravel_pytree([eqx.filter(c, eqx.is_inexact_array) for c in CHANNELS])[0])
    rng = np.random.default_rng(11)
    return weight, [rng.standard_normal(size.size) * np.asarray(size) for _ in range(3)]


def test_the_weight_is_the_second_order_loss_of_a_plan_on_moved_channels() -> None:
    """Twice the loss over the step squared is d' W d plus a term linear in the step, from the
    loss's third order, which one Richardson step removes."""
    with enable_x64(True):
        weight, directions = _directions()
        assert weight.pinned == ()
        for direction in directions:
            step = 4e-3  # a smaller one reads the allocation's tolerance
            coarse = 2.0 * _loss(CHANNELS, direction, step) / step**2
            fine = 2.0 * _loss(CHANNELS, direction, step / 2.0) / (step / 2.0) ** 2
            extrapolated = 2.0 * fine - coarse
            predicted = float(direction @ weight.matrix @ direction)
            assert extrapolated == pytest.approx(predicted, rel=1e-4)


def test_the_quadratic_reads_the_loss_within_a_tenth_for_errors_a_tenth_of_each_parameter() -> None:
    """How far the second order reaches on these channels: each parameter off by a tenth of its
    size, in a standard normal's units, and the plan re-made."""
    with enable_x64(True):
        weight, directions = _directions()
        for direction in directions:
            predicted = 0.5 * float(direction @ weight.matrix @ direction) * 0.1**2
            assert _loss(CHANNELS, direction, 0.1) == pytest.approx(predicted, rel=0.1)


def test_a_channel_at_an_end_of_its_box_carries_no_weight() -> None:
    capped = UPPER.copy()
    capped[2] = 25.0  # the exponential channel's cap binds
    with enable_x64(True):
        weight = decision_weight(
            CHANNELS, BUDGET, PERIODS, lower=LOWER, upper=capped, history=HISTORY
        )
    assert weight.allocation.spend[2] == 25.0
    assert weight.pinned == (2,)
    held = [k for k, name in enumerate(weight.parameters) if name.startswith("2.")]
    assert not np.any(weight.matrix[held])
    first = weight.parameters.index("0.coefficient")
    assert weight.matrix[first, first] > 0.0


# the exponential channel held at a cap of 25 a period, or a floor of 80, where it would run at 39
PINNED = {"upper": 25.0, "lower": 80.0}


def _pinned_weight(end: str):
    box = {"lower": LOWER.copy(), "upper": UPPER.copy(), "history": HISTORY}
    box[end][2] = PINNED[end]
    return decision_weight(CHANNELS, BUDGET, PERIODS, **box), box


def test_a_pinned_channel_s_margin_and_its_use_are_maxima_s() -> None:
    """A third exponential channel at a floor of 0 returns ``b3/K3`` on a unit, below the price the
    two free channels share, ``lambda = (b1/K1) exp(-u1/K1)``: its margin is the difference, and an
    error uses it up at the gradient of ``b3/K3`` less that of ``lambda``
    (``validation/allocation_decision_weight.mac``, STEP 3)."""
    channels = (
        Channel(ONE, Exponential(50.0), 900.0),
        Channel(ONE, Exponential(80.0), 600.0),
        Channel(ONE, Exponential(100.0), 100.0),
    )
    weight = decision_weight(channels, 150.0, 1, lower=[0.0] * 3, upper=[150.0] * 3)
    assert weight.pinned == (2,)
    assert weight.margin.tolist() == pytest.approx([2.312739901658918], rel=1e-8)
    order = [f"{c}.{name}" for c in range(3) for name in ("coefficient", "curve.scale")]
    used = [
        -1.415700812674751e-3,
        -1.764916068308613e-2,
        -3.397681950419403e-3,
        4.660071769892447e-3,
        1e-2,
        -1e-2,
    ]
    index = [weight.parameters.index(name) for name in order]
    np.testing.assert_allclose(weight.margin_gradient[0, index], used, rtol=1e-8)
    unused = [weight.parameters.index(f"{c}.kernel.retention") for c in range(3)]
    assert not np.any(weight.margin_gradient[0, unused])


@pytest.mark.parametrize("end", ["upper", "lower"])
def test_a_pinned_channel_s_margin_is_its_return_on_a_unit_from_the_price(end: str) -> None:
    """At its cap the exponential channel returns more on a currency unit than the budget's price,
    at its floor less, which holds it there; its margin is the difference, the slope read by a
    central difference of its worth over the history, the plan and the tail. Moving the
    parameters along a direction uses up the margin at ``margin_gradient``'s rate, against the
    margin of the weight on the moved channels."""
    with enable_x64(True):
        weight, box = _pinned_weight(end)
        assert weight.pinned == (2,)
        spend = weight.allocation.spend
        assert spend[2] == PINNED[end]
        step = 1e-3
        ups = [_worth(CHANNELS, spend + k * step * np.eye(3)[2]) for k in (-1, 1)]
        slope = (ups[1] - ups[0]) / (2.0 * step) / PERIODS
        side = 1.0 if end == "upper" else -1.0
        assert weight.margin.shape == (1,)
        assert weight.margin[0] == pytest.approx(side * (slope - weight.allocation.price), rel=1e-6)
        assert weight.margin[0] > 0.0
        assert weight.margin_gradient.shape == (1, len(weight.parameters))
        for direction in _directions()[1]:
            h = 1e-5
            moved = [
                decision_weight(_moved(CHANNELS, direction, k * h), BUDGET, PERIODS, **box).margin
                for k in (-1, 1)
            ]
            used = -(moved[1][0] - moved[0][0]) / (2.0 * h)
            assert used == pytest.approx(float(weight.margin_gradient[0] @ direction), rel=1e-5)


@pytest.mark.parametrize("end", ["upper", "lower"])
@pytest.mark.parametrize(("share", "held"), [(0.99, True), (1.01, False)])
def test_errors_that_use_up_the_margin_move_the_pinned_channel(
    end: str, share: float, held: bool
) -> None:
    """The weight reads the loss while the errors leave the pinned channel at its end. An error in
    the pinned channel's coefficient scales its slope and moves no price, so it uses up the margin
    exactly at ``margin_gradient``'s rate: one that uses up 99% of it leaves the plan on the moved
    channels at the end, one that uses up 101% moves it off, a loss the weight does not read."""
    with enable_x64(True):
        weight, box = _pinned_weight(end)
        error = np.zeros(len(weight.parameters))
        coefficient = weight.parameters.index("2.coefficient")
        error[coefficient] = share * weight.margin[0] / weight.margin_gradient[0, coefficient]
        plan = allocate(_moved(CHANNELS, error, 1.0), BUDGET, PERIODS, **box)
        assert bool(plan.spend[2] == PINNED[end]) is held
        assert abs(plan.spend[2] - PINNED[end]) < 1.0


def test_the_margin_in_standard_deviations_reads_the_covariance() -> None:
    """``margin / sqrt(g' S g)``; a channel whose box is a point, and a budget that fills every
    box, hold their channels whatever the errors, so their margins are infinite; a plan that holds
    one channel at its cap and the other at its floor with none inside prices the budget at an
    interval, and its margins are nil."""
    with enable_x64(True):
        weight, box = _pinned_weight("upper")
        covariance = np.diag(np.linspace(1.0, 2.0, len(weight.parameters))) * 1e-2
        gradient = weight.margin_gradient[0]
        np.testing.assert_allclose(
            weight.margin_sds(covariance),
            weight.margin / np.sqrt(gradient @ covariance @ gradient),
            rtol=1e-12,
        )
        with pytest.raises(ValueError, match="covariance has shape"):
            weight.margin_sds([[1.0]])
        # a nan or a negative variance read as an infinite margin, with a warning from the root
        size = len(weight.parameters)
        for bad, why in [
            (np.full((size, size), np.nan), "not finite"),
            (-np.eye(size), "not positive semidefinite"),
            (np.eye(size) + np.triu(np.ones((size, size)), 1), "not symmetric"),
        ]:
            for reading in (weight.margin_sds, weight.expected_regret):
                with pytest.raises(ValueError, match=why):
                    reading(bad)
        # a semidefinite covariance that the errors leave in one direction is read as given
        direction = gradient / np.linalg.norm(gradient)
        np.testing.assert_allclose(
            weight.margin_sds(np.outer(direction, direction)),
            weight.margin / np.linalg.norm(gradient),
            rtol=1e-12,
        )
        point = decision_weight(
            CHANNELS, BUDGET, PERIODS, lower=[30.0, 20.0, 25.0], upper=box["upper"], history=HISTORY
        )
        assert point.pinned == (2,)
        assert point.margin.tolist() == [np.inf]
        assert not np.any(point.margin_gradient)
        assert point.margin_sds(covariance).tolist() == [np.inf]
        box = {"lower": LOWER, "upper": UPPER, "history": HISTORY}
        full = decision_weight(CHANNELS, PERIODS * float(UPPER.sum()), PERIODS, **box)
        assert full.pinned == (0, 1, 2)
        assert full.margin.tolist() == [np.inf] * 3
    channels = (Channel(ONE, Exponential(50.0), 900.0), Channel(ONE, Exponential(80.0), 10.0))
    corner = decision_weight(channels, 10.0, 1, lower=[0.0, 0.0], upper=[10.0, 10.0])
    assert corner.allocation.spend.tolist() == [10.0, 0.0]
    assert corner.pinned == (0, 1)
    assert corner.margin.tolist() == [0.0, 0.0]
    assert corner.margin_sds(np.eye(len(corner.parameters))).tolist() == [0.0, 0.0]


def test_the_expected_regret_is_half_the_weight_s_trace_against_the_covariance() -> None:
    channels = (Channel(ONE, Exponential(50.0), 900.0), Channel(ONE, Exponential(80.0), 600.0))
    weight = decision_weight(channels, 150.0, 1, lower=[0.0, 0.0], upper=[150.0, 150.0])
    covariance = np.diag(np.linspace(1.0, 2.0, len(weight.parameters)))
    assert weight.expected_regret(covariance) == pytest.approx(
        0.5 * np.trace(weight.matrix @ covariance)
    )
    with pytest.raises(ValueError, match="covariance has shape"):
        weight.expected_regret([[1.0]])  # would broadcast


def test_what_the_weight_cannot_read_is_refused() -> None:
    s_shaped = (Channel(ONE, Hill(50.0, 2.0), 900.0), Channel(ONE, Exponential(80.0), 600.0))
    with pytest.raises(ValueError, match="S-shaped"):
        decision_weight(s_shaped, 150.0, 1, lower=[0.0, 0.0], upper=[150.0, 150.0])
