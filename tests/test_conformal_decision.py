"""The conformal controller moving a plan's pessimism: :class:`chc.gate.ConformalPessimism` against
every adversary on short sequences and random ones on long, and :class:`chc.gate.ConformalHorizon`
feeding its weight to a :class:`chc.mpc.RecedingHorizon` on a plant whose channel drifts.

The bound's identities, every adversary in exact rationals, and the sequence a loss that rises again
with the weight plays are in ``validation/conformal_decision.mac``; the values pinned here are its.
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from chc import (
    ConformalHorizon,
    ConformalPessimism,
    QuadraticCost,
    RecedingHorizon,
    SupportModel,
)
from chc.dynamics import LinearDynamics
from chc.integrate import rk4_step
from chc.uncertainty import ConfoundingRobustPenalty

# Steps 4-7 of the Maxima file: sums binary floating point holds exactly, so a replay here takes
# every branch the rationals there take.
STEP, TARGET, BOUND, SAFE = 0.5, 0.125, 1.0, 1.0

# An adversary: given the weights issued so far, the last the current one, and the losses played,
# the losses it may play next.
Adversary = Callable[[list[float], list[float]], tuple[float, ...]]


def _pessimism(**given: float) -> ConformalPessimism:
    arguments = {"target": TARGET, "step": STEP, "bound": BOUND, "safe": SAFE} | given
    initial = arguments.pop("initial", 0.0)
    return ConformalPessimism(initial, **arguments)  # type: ignore[arg-type]


def _ratio(pessimism: ConformalPessimism) -> float:
    """How much of the bound the mean loss's excess over the target takes."""
    return (pessimism.mean_loss - pessimism.target) / pessimism.excess_bound


def _worst(
    pessimism: ConformalPessimism, losses: list[float], adversary: Adversary, depth: int
) -> float:
    """The largest share of the bound any play of ``adversary`` takes at any decision, over
    ``depth`` decisions after ``losses``; every play keeps the cap uncut."""
    worst = _ratio(pessimism) if pessimism.steps else 0.0
    if depth == 0:
        return worst
    for loss in adversary(list(pessimism.path), losses):
        child = copy.deepcopy(pessimism)
        child.update(loss)
        assert child.capped == 0
        worst = max(worst, _worst(child, [*losses, loss], adversary, depth - 1))
    return worst


def _one_step_safe(weights: list[float], losses: list[float]) -> tuple[float, ...]:
    return (0.0, TARGET) if weights[-1] >= SAFE else (0.0, TARGET, BOUND)


def _two_step_safe(weights: list[float], losses: list[float]) -> tuple[float, ...]:
    """A loss is refused where it and the one before, both at weights of at least ``safe``,
    average more than the target."""
    n = len(losses)
    both = n >= 1 and weights[n - 1] >= SAFE and weights[-1] >= SAFE
    return tuple(
        loss for loss in (0.0, TARGET, BOUND) if not (both and (losses[-1] + loss) / 2 > TARGET)
    )


@pytest.mark.parametrize(
    ("window", "adversary", "depth", "maxima"),
    [(1, _one_step_safe, 10, Fraction(22, 23)), (2, _two_step_safe, 8, Fraction(14, 15))],
)
def test_every_adversary_keeps_the_mean_within_the_bound(
    window: int, adversary: Adversary, depth: int, maxima: Fraction
) -> None:
    """Every play of losses 0, the target and the bound that an eventually safe adversary may
    make, the floor at 0, as Maxima's STEP 4 enumerates them: the cap is never cut, and the mean
    loss never passes the bound. The worst play takes the share of it Maxima found."""
    pessimism = _pessimism(window=window)
    assert pessimism.cap == SAFE + window * STEP * (BOUND - TARGET)
    worst = _worst(pessimism, [], adversary, depth)
    assert worst == pytest.approx(float(maxima), rel=1e-12, abs=0.0)


def test_the_bound_is_attained_up_to_the_last_gap_below_safe() -> None:
    """The adversary that plays the bound below ``safe`` and the target at or above it: three
    rises of 7/16 carry the weight from 0 past 1, and the mean's excess is then ``21 / (8 T)``
    against a bound of ``23 / (8 T)``. With ``safe`` just above the last weight below it, the gap
    is ``(safe - 7/8) / (step T)``, as small as ``safe`` is close to 7/8."""
    for safe, bound in ((1.0, Fraction(23, 160)), (7 / 8 + 1 / 1024, Fraction(269, 2048))):
        pessimism = _pessimism(safe=safe)
        for _ in range(20):
            pessimism.update(TARGET if pessimism.weight >= safe else BOUND)
        assert pessimism.mean_loss - TARGET == pytest.approx(21 / 160, rel=1e-15, abs=0.0)
        assert pessimism.excess_bound == pytest.approx(float(bound), rel=1e-15, abs=0.0)
        assert pessimism.capped == pessimism.floored == 0


def test_the_weight_is_the_published_update_held_to_its_floor_and_cap() -> None:
    """``w_(t+1) = w_t + eta (l_t - eps)``: Lekeufack et al.'s update (5), with their ``lambda``
    as ``-w``, held to ``[lower, cap]``."""
    rng = np.random.default_rng(51)
    pessimism = ConformalPessimism(0.3, target=0.2, step=0.4, bound=2.0, safe=1.5, lower=0.1)
    assert pessimism.cap == pytest.approx(1.5 + 0.4 * 1.8, rel=1e-15, abs=0.0)
    w, capped, floored = 0.3, 0, 0
    losses = rng.choice([0.0, 2.0], size=400, p=[0.85, 0.15]) * rng.uniform(0.5, 1.0, 400)
    for loss in losses:
        moved = w + 0.4 * (loss - 0.2)
        capped += moved > pessimism.cap
        floored += moved < 0.1
        w = min(max(moved, 0.1), pessimism.cap)
        assert pessimism.update(loss) == w
    assert pessimism.path.shape == (401,)
    assert np.all((pessimism.path >= 0.1) & (pessimism.path <= pessimism.cap))
    assert (pessimism.capped, pessimism.floored) == (capped, floored)
    assert capped > 0  # this sequence meets both ends
    assert floored > 0
    assert pessimism.mean_loss == pytest.approx(float(np.mean(losses)), rel=1e-12, abs=0.0)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize(("initial", "lower"), [(0.0, 0.0), (SAFE, 0.5), (SAFE, -50.0)])
def test_random_losses_safe_past_safe_keep_the_bound_at_every_decision(
    seed: int, initial: float, lower: float
) -> None:
    """Losses drawn afresh at every decision from a law that falls as the weight grows: a share
    ``1 - weight / safe`` of them uniform on ``[0, bound]``, the rest on ``[0, target]``, so the
    target is met from ``safe`` on. The cap is never cut and the bound holds at every decision;
    while the floor has lifted nothing, the mean also stays above ``target - (initial - lower) /
    (step n)``. The slack is rounding: the two ends are identities in exact arithmetic."""
    rng = np.random.default_rng([52, seed])
    pessimism = _pessimism(initial=initial, lower=lower)
    for _ in range(3000):
        share = min(1.0, max(0.0, 1.0 - pessimism.weight / SAFE))
        loss = BOUND * rng.uniform() if rng.uniform() < share else TARGET * rng.uniform()
        pessimism.update(loss)
        assert pessimism.mean_loss - TARGET <= pessimism.excess_bound + 1e-12
        if pessimism.floored == 0:
            least = -(initial - lower) / (STEP * pessimism.steps)
            assert pessimism.mean_loss - TARGET >= least - 1e-12
    assert pessimism.capped == 0
    assert abs(pessimism.mean_loss - TARGET) < 0.01  # it settles on the target, not under it
    if lower == -50.0:
        assert pessimism.floored == 0


def test_losses_that_ignore_the_weight_reach_the_cap_and_say_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Losses of mean 1/2 whatever the weight: no weight meets the target of 1/8, the weight
    reaches its cap, the first cut logs a warning, and the mean passes the bound."""
    rng = np.random.default_rng(53)
    pessimism = _pessimism()
    with caplog.at_level(logging.WARNING, logger="chc.gate"):
        for loss in rng.uniform(0.0, 1.0, 400):
            pessimism.update(loss)
    cuts = [r for r in caplog.records if getattr(r, "chc_event", None) == "conformal_cap"]
    assert len(cuts) == 1
    assert pessimism.capped > 100
    assert pessimism.path.max() == pessimism.cap
    assert pessimism.mean_loss - TARGET > pessimism.excess_bound


def test_a_loss_that_rises_again_with_the_weight_breaks_the_bound() -> None:
    """The loss is the bound below 1, 0 on [1, 2) and the bound again from 2: a plan too cautious
    misses as one too bold does. No weight is safe, and at a claimed ``safe = 3/2`` with a step of 2
    the weight jumps the band, climbs to the cap and stays: Maxima's STEP 7 path, the cap cut 14 of
    20 times, and a mean excess of 27/40 against a bound of 13/160."""

    def loss(w: float) -> float:
        return 0.0 if 1.0 <= w < 2.0 else 1.0

    pessimism = ConformalPessimism(0.0, target=TARGET, step=2.0, bound=1.0, safe=1.5)
    for _ in range(20):
        pessimism.update(loss(pessimism.weight))
    maxima = [0, 7 / 4, 3 / 2, 5 / 4, 1, 3 / 4, 5 / 2] + [13 / 4] * 14
    assert pessimism.path.tolist() == maxima
    assert pessimism.capped == 14
    assert pessimism.mean_loss - TARGET == pytest.approx(27 / 40, rel=1e-15, abs=0.0)
    assert pessimism.excess_bound == pytest.approx(13 / 160, rel=1e-15, abs=0.0)


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({"bound": 0.0}, "bound must be positive and finite"),
        ({"step": math.inf}, "step must be positive and finite"),
        ({"target": 1.0}, r"target must lie in \[0, bound = 1.0\)"),
        ({"target": -0.1}, r"target must lie in \[0, bound = 1.0\)"),
        ({"safe": math.nan}, "safe must be finite"),
        ({"lower": 2.0}, "lower must not exceed safe = 1.0"),
        ({"window": 0}, "window must be at least 1"),
        ({"initial": -0.1}, r"initial must lie in \[lower, safe \+ step \(bound - target\)\]"),
        ({"initial": 1.5}, r"= \[0.0, 1.4375\], got 1.5"),
    ],
)
def test_a_pessimism_outside_the_contract_is_refused(
    arguments: dict[str, float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _pessimism(**arguments)


def test_losses_outside_the_contract_are_refused() -> None:
    pessimism = _pessimism()
    for loss in (-0.1, 1.5, math.nan, math.inf):
        with pytest.raises(ValueError, match=r"loss must be one number in \[0, bound = 1.0\]"):
            pessimism.update(loss)
    with pytest.raises(ValueError, match="one number"):
        pessimism.update([0.1, 0.2])  # type: ignore[arg-type]
    assert pessimism.steps == 0  # a refused loss moves nothing
    assert math.isnan(pessimism.mean_loss)
    assert math.isinf(pessimism.excess_bound)
    pessimism.update(BOUND)
    assert pessimism.excess_bound == pytest.approx(1.4375 / STEP, rel=1e-15, abs=0.0)


# --- ConformalHorizon ----------------------------------------------------------------------------

DT, HORIZON, DRIFT, TOLERANCE = 0.1, 10, 0.05, 0.01
MODEL = LinearDynamics(jnp.array([[-0.5]]), jnp.array([[1.0]]))
COST = QuadraticCost(Q=jnp.eye(1), R=0.01 * jnp.eye(1), Qf=jnp.eye(1), x_target=jnp.array([1.0]))
_LOG = np.random.default_rng(54)
SUPPORT = SupportModel.fit(
    jnp.asarray(_LOG.normal(size=(200, 1))), jnp.asarray(_LOG.normal(size=(200, 1)))
)


class _DriftingGain(eqx.Module):
    """The model's plant with a lever whose gain grows by ``DRIFT`` per unit of time: the model's
    channel is right at 0 and wrong ever after."""

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return -0.5 * x + (1.0 + DRIFT * t) * u


PLANT = _DriftingGain()


def _controller(lam_unc: float = 0.0) -> RecedingHorizon:
    return RecedingHorizon(
        MODEL,
        COST,
        DT,
        HORIZON,
        -3.0,
        3.0,
        support=SUPPORT,
        uncertainty=ConfoundingRobustPenalty(radius=0.5),
        lam_unc=lam_unc,
    )


def _missed(x: Array, t: float, u: Array) -> tuple[Array, float]:
    """The plant's next state, and the loss: 1 where it lies more than ``TOLERANCE`` from the
    model's prediction, 0 within it. The miss grows with the action, as the model's channel
    error does."""
    reached = rk4_step(PLANT, t, x, u, DT)
    predicted = rk4_step(MODEL, t, x, u, DT)
    return reached, float(abs(float(reached[0] - predicted[0])) > TOLERANCE)


@contextmanager
def _compilations() -> Iterator[list[str]]:
    compiled: list[str] = []

    class Record(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.getMessage().startswith("Compiling "):
                compiled.append(record.getMessage().split(" ")[1])

    handler, logger = Record(), logging.getLogger("jax")
    logger.addHandler(handler)
    try:
        with jax.log_compiles(True):
            yield compiled
    finally:
        logger.removeHandler(handler)


def test_on_a_drifting_plant_the_loop_keeps_its_mean_miss_within_the_bound() -> None:
    """120 decisions on the plant whose gain drifts from the model's: at a fixed ``lam_unc = 0``
    the model's prediction misses on more than half of them, and from midway on every one. Moved
    by the conformal controller toward a target of 0.2, the share that misses keeps within the
    bound, 0.24, each plan is made at the weight issued for it, and no weight reaches the cap: at
    ``safe = 2`` the plan holds every action at zero, where the prediction is exact."""
    pessimism = ConformalPessimism(0.0, target=0.2, step=0.5, bound=1.0, safe=2.0)
    loop = ConformalHorizon(_controller(), pessimism)
    fixed = _controller()
    x, x_fixed, misses_fixed, planned_at = jnp.zeros(1), jnp.zeros(1), [], []
    for k in range(120):
        t = k * DT
        plan = loop.step(x, t)
        planned_at.append(loop.controller.lam_unc)
        x, loss = _missed(x, t, plan.actions[0])
        loop.update(loss)
        x_fixed, loss = _missed(x_fixed, t, fixed.step(x_fixed, t).actions[0])
        misses_fixed.append(loss)
    assert planned_at == pessimism.path[:-1].tolist()
    assert all(type(w) is np.float64 for w in planned_at)
    assert pessimism.capped == 0
    assert pessimism.target + pessimism.excess_bound == pytest.approx(0.24, rel=1e-12, abs=0.0)
    assert pessimism.mean_loss <= 0.24
    assert np.mean(misses_fixed) > 0.5
    assert np.mean(misses_fixed[60:]) == 1.0
    assert pessimism.mean_loss > 0.0  # it met misses
    assert max(planned_at) > 0.5  # and moved


@pytest.mark.parametrize("initial", [0.0, 0.75])
def test_the_first_plan_is_the_controllers_own_at_that_weight(initial: float) -> None:
    """The weight goes in as a numpy float, which the descent reads as an input rather than a
    constant of its program; the plan is the one a Python float gives. Bit for bit on Linux, but
    XLA may fold a constant weight of 0 out of the program, and macOS's runner then read the last
    bits otherwise: the same plan, to rounding."""
    pessimism = ConformalPessimism(initial, target=0.2, step=0.5, bound=1.0, safe=2.0)
    plan = ConformalHorizon(_controller(), pessimism).step(jnp.array([0.2]))
    plain = _controller(initial).step(jnp.array([0.2]))
    np.testing.assert_allclose(
        np.asarray(plan.actions), np.asarray(plain.actions), rtol=1e-9, atol=1e-12
    )


def test_a_moving_weight_compiles_nothing_after_the_first_two_steps() -> None:
    """The first step at a weight of 0 and the second off it take the two paths the radius' norm
    has, and every step after reuses them. A Python float is a constant of the program instead:
    the same controller planning at a new one compiles the descent again."""
    jax.clear_caches()
    pessimism = ConformalPessimism(0.0, target=0.2, step=0.5, bound=1.0, safe=2.0)
    loop = ConformalHorizon(_controller(), pessimism)
    x = jnp.zeros(1)

    def step(k: int) -> None:
        nonlocal x
        plan = loop.step(x, k * DT)
        x, _ = _missed(x, k * DT, plan.actions[0])
        loop.update(BOUND if k < 4 else 0.0)  # weights 0.4, 0.8, 1.2, 1.6, 1.5, 1.4

    step(0)
    step(1)
    with _compilations() as compiled:
        for k in range(2, 6):
            step(k)
    assert len(set(pessimism.path.tolist())) == 7
    assert compiled == []
    loop.controller.lam_unc = 1.3
    with _compilations() as compiled:
        loop.controller.step(x, 6 * DT)
    assert compiled != []


def test_plans_and_losses_alternate() -> None:
    pessimism = ConformalPessimism(0.0, target=0.2, step=0.5, bound=1.0, safe=2.0)
    loop = ConformalHorizon(_controller(), pessimism)
    with pytest.raises(ValueError, match="no plan awaits a loss"):
        loop.update(0.0)
    loop.step(jnp.zeros(1))
    with pytest.raises(ValueError, match="the last plan's loss has not been scored"):
        loop.step(jnp.zeros(1))
    with pytest.raises(ValueError, match="loss must be one number"):
        loop.update(2.0)
    assert loop.update(1.0) == pytest.approx(0.4, rel=1e-15, abs=0.0)  # the refused loss kept it
    loop.step(jnp.zeros(1), DT)
    assert loop.controller.lam_unc == pessimism.weight


def test_a_horizon_outside_the_contract_is_refused() -> None:
    pessimism = ConformalPessimism(0.0, target=0.2, step=0.5, bound=1.0, safe=2.0)
    bare = RecedingHorizon(MODEL, COST, DT, HORIZON, -3.0, 3.0, support=SUPPORT)
    with pytest.raises(ValueError, match="no uncertainty model"):
        ConformalHorizon(bare, pessimism)
    below = ConformalPessimism(0.0, target=0.2, step=0.5, bound=1.0, safe=2.0, lower=-1.0)
    with pytest.raises(ValueError, match="must not be negative"):
        ConformalHorizon(_controller(), below)
