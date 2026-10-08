"""Whether a plan pays for its model class being wrong: two fits of one class on one log, their
difference priced in the plan's regret and tested against the law it has when the class holds.

The plant's channel moves with the first state, ``b(x) = b0 + slope x_0 b1``, and the fits are of
the constant-channel class, unweighted and weighted by ``exp(x_0)``: at ``slope = 0`` the class
holds the truth. The measurements the docstrings quote are ``scripts/bench_misspecification.py``'s;
these tests hold the properties those numbers rest on.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, stats

from chc import BarrierConstraint, LinearConstraint, QuadraticCost, causal_plan
from chc.cost import total_cost
from chc.dynamics import HybridDynamics
from chc.dynamics_id import CausalDynamicsFit, fit_causal_residual
from chc.integrate import rk4_step
from chc.misspecification import (
    MisspecificationCost,
    _chi_square_mixture_survival,
    _mixture_weights,
    _parameter_weight,
    _parameters,
    _residual,
    misspecification_cost,
)
from chc.plan import CausalPlan
from chc.residual import ControlAffineResidual

DT, HORIZON, BOX = 0.1, 12, 5.0
A = np.array([[-0.5, 0.2], [0.0, -0.3]])
B0 = np.array([0.8, -0.4])
B1 = np.array([0.6, 0.0])
C = np.array([1.5, 0.0])
START = jnp.zeros(2)
COST = QuadraticCost(
    Q=jnp.diag(jnp.array([1.0, 0.1])),
    R=jnp.array([[0.05]]),
    Qf=jnp.diag(jnp.array([1.0, 0.1])),
    x_target=jnp.array([1.0, 0.0]),
)
FIT = {"adjust_for": ("z",), "degree": 1, "channel_degree": 0, "nuisance_degree": 2, "seed": 0}
PAIR = np.array([[0.8, 0.3], [-0.4, 0.5]])


def _known(t: float | jax.Array, x: jax.Array, u: jax.Array) -> jax.Array:
    return jnp.zeros_like(x)


def _log(rows: int, seed: int, slope: float = 1.0, mean: float = 0.0) -> dict[str, jax.Array]:
    """A whole log: states, the adjustment column, the confounded action and the noise, redrawn."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (rows, 2))
    z = rng.normal(0.0, 1.0, rows)
    u = mean + 0.9 * z - 0.3 * x[:, 0] + rng.normal(0.0, 0.5, rows)
    b = B0[None, :] + slope * np.outer(x[:, 0], B1)
    rate = x @ A.T + b * u[:, None] + np.outer(z, C)
    x_next = x + DT * rate + rng.normal(0.0, 0.01, (rows, 2))
    return {
        "x": jnp.asarray(x),
        "u": jnp.asarray(u[:, None]),
        "x_next": jnp.asarray(x_next),
        "z": jnp.asarray(z[:, None]),
    }


def _fits(
    data: dict[str, jax.Array], **kwargs: object
) -> tuple[CausalDynamicsFit, CausalDynamicsFit]:
    options = {**FIT, "influence": True, **kwargs}
    reference = fit_causal_residual(_known, data, DT, **options)
    alternative = fit_causal_residual(
        _known,
        data,
        DT,
        weights=lambda states: jnp.exp(states[:, 0]),
        **options,
    )
    return reference, alternative


def _model(residual: ControlAffineResidual) -> HybridDynamics:
    return HybridDynamics(known=_known, residual=residual)


def _plan(
    residual: ControlAffineResidual, cost: QuadraticCost = COST, **kwargs: object
) -> CausalPlan:
    return causal_plan(
        _model(residual),
        START,
        cost,
        DT,
        HORIZON,
        -BOX,
        BOX,
        steps=20_000,
        **kwargs,
    )


def _optimum(residual: ControlAffineResidual, start: jax.Array) -> jax.Array:
    """The unconstrained optimum under ``residual``'s model, by Newton's method from ``start``: the
    regret of a small move needs more digits than a first-order solve leaves."""
    model = _model(residual)

    def cost(flat: jax.Array) -> jax.Array:
        return total_cost(model, START, flat.reshape(start.shape), DT, COST)

    flat = start.ravel()
    for _ in range(6):
        flat = flat - jnp.linalg.solve(jax.hessian(cost)(flat), jax.grad(cost)(flat))
    assert float(jnp.max(jnp.abs(flat))) < BOX  # the box does not bind: the optimum is interior
    return flat.reshape(start.shape)


def _regret(residual: ControlAffineResidual, actions: jax.Array) -> float:
    model = _model(residual)
    best = _optimum(residual, actions)
    return float(
        total_cost(model, START, actions, DT, COST) - total_cost(model, START, best, DT, COST)
    )


@functools.cache
def _case(
    rows: int, seed: int, slope: float, mean: float
) -> tuple[CausalDynamicsFit, CausalDynamicsFit, CausalPlan]:
    reference, alternative = _fits(_log(rows, seed, slope, mean))
    plan = _plan(reference.residual)
    return (
        reference,
        alternative,
        dataclasses.replace(plan, actions=_optimum(reference.residual, plan.actions)),
    )


# --- the fit's influence --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("integrator", "tolerance"), [("euler", 1e-9), ("rk4", 0.05)], ids=["euler", "rk4"]
)
def test_the_influence_s_channel_block_sums_to_the_channel_error(
    integrator: str, tolerance: float
) -> None:
    """Under Euler the channel's influence is the one its error is read from. Under ``rk4`` the RK4
    map also carries the drift's residual into the channel, which the error, read with the noise
    alone random, leaves out: here that moved it by half a percent."""
    fit = fit_causal_residual(
        _known,
        _log(2000, 0),
        DT,
        integrator=integrator,
        influence=True,
        **FIT,
    )
    assert fit.influence is not None
    assert fit.channel_error is not None
    size = fit.residual.channel.size
    rows = np.asarray(fit.influence).reshape(-1, fit.influence.shape[-1])[:, :size]
    assert math.sqrt(np.mean(np.diag(rows.T @ rows))) == pytest.approx(
        fit.channel_error, rel=tolerance
    )


@pytest.mark.parametrize("integrator", ["euler", "rk4"])
def test_asking_for_the_influence_leaves_the_fit_as_it_was(integrator: str) -> None:
    data = _log(2000, 0)
    plain, kept = (
        fit_causal_residual(_known, data, DT, integrator=integrator, influence=flag, **FIT)
        for flag in (False, True)
    )
    assert plain.influence is None
    assert kept.influence is not None
    assert kept.influence.shape == (2000, 2, _parameters(kept.residual).size)
    np.testing.assert_array_equal(_parameters(plain.residual), _parameters(kept.residual))
    assert (plain.channel_error, plain.drift_error) == (kept.channel_error, kept.drift_error)


@functools.cache
def _redraws(slope: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Sixty whole logs of 1000 rows -- states, actions, the adjustment column and the noise, all
    redrawn: the two fits' differences, and the mean covariance their influences give."""
    differences, covariances = [], []
    for seed in range(60):
        reference, alternative = _fits(_log(1000, 1000 + seed, slope))
        differences.append(_parameters(alternative.residual) - _parameters(reference.residual))
        spread = np.asarray(alternative.influence) - np.asarray(reference.influence)
        spread = spread.reshape(-1, differences[-1].size)
        covariances.append(spread.T @ spread)
    return np.array(differences), np.mean(covariances, axis=0), reference.residual.channel.size


def _spread_over_predicted(slope: float, block: slice) -> float:
    differences, predicted, _ = _redraws(slope)
    realised = np.cov(differences, rowvar=False)
    return float(np.trace(realised[block, block]) / np.trace(predicted[block, block]))


def test_the_drift_s_difference_spreads_over_whole_logs_as_its_influence_says() -> None:
    """Two fits' drifts differ by the log's projection of their channels' difference on the drift's
    features, and that projection moves when the log is drawn again: the drift's influence reads
    it through its own regression's residual. Read through the moment's residual instead, as it
    first was, the drift's spread came to 2.2 of what the influences said when the class missed."""
    size = _redraws(1.0)[2]
    assert _spread_over_predicted(1.0, slice(size, None)) == pytest.approx(1.0, abs=0.25)


def test_when_the_class_holds_the_difference_spreads_as_its_influence_says() -> None:
    assert _spread_over_predicted(0.0, slice(None)) == pytest.approx(1.0, abs=0.2)


# --- the price ------------------------------------------------------------------------------------


def test_the_quadratic_is_the_regret_between_re_solved_plans_to_second_order() -> None:
    """``d' W d / 2`` against the regret of the plan made on the reference when the model moved by
    ``t d`` holds, the plan re-solved on it: the gap closes in proportion to ``t``."""
    reference, alternative, plan = _case(10_000, 0, 1.0, 0.0)
    weight = _parameter_weight(plan, reference.residual, None)
    own = _parameters(reference.residual)
    d = _parameters(alternative.residual) - own
    ratios = []
    for t in (0.1, 0.03):
        regret = _regret(_residual(jnp.asarray(own + t * d), reference.residual), plan.actions)
        ratios.append(t * t * float(d @ weight @ d) / 2.0 / regret)
    assert abs(1.0 - ratios[0]) < 0.15
    assert abs(1.0 - ratios[1]) < 0.4 * abs(1.0 - ratios[0])


@pytest.mark.parametrize(
    ("mean", "channel_low", "channel_high"),
    [(-1.0, 0.0, 0.3), (1.0, 2.0, 4.0)],
    ids=["actions averaging -1", "actions averaging +1"],
)
def test_the_channel_alone_misprices_what_the_drift_carries(
    mean: float, channel_low: float, channel_high: float
) -> None:
    """Two fits whose channels differ have drifts that differ by the log's projection of the
    channel's move, so the models differ by ``dB (u - uhat(x))``. The channel's price, the drift
    held, reads that far off the regret between the two fits' plans; the quadratic over every
    parameter reads about half of it at this size of move, from below."""
    reference, alternative, plan = _case(10_000, 0, 1.0, mean)
    regret = _regret(alternative.residual, plan.actions)
    moved = dataclasses.replace(reference.residual, channel=alternative.residual.channel)

    def one_step(residual: ControlAffineResidual) -> jax.Array:
        return jax.jacfwd(lambda u: rk4_step(_model(residual), 0.0, START, u, DT))(jnp.zeros(1))

    channel = plan.decision_weight().regret(one_step(moved) - one_step(reference.residual))
    gate = misspecification_cost(plan, reference, alternative)
    assert channel_low < channel / regret < channel_high
    assert 0.4 < (gate.cost + gate.noise) / regret < 0.65


def test_the_cost_is_the_quadratic_less_the_noise_s_share() -> None:
    reference, alternative, plan = _case(10_000, 0, 1.0, 0.0)
    weight = _parameter_weight(plan, reference.residual, None)
    d = _parameters(alternative.residual) - _parameters(reference.residual)
    spread = np.asarray(alternative.influence) - np.asarray(reference.influence)
    spread = spread.reshape(-1, d.size)
    gate = misspecification_cost(plan, reference, alternative)
    assert gate.noise == pytest.approx(float(np.trace(weight @ spread.T @ spread)) / 2, rel=1e-9)
    assert gate.cost == pytest.approx(float(d @ weight @ d) / 2 - gate.noise, rel=1e-9)
    assert 0.0 < gate.noise < 1e-2 * gate.cost
    assert gate.p_value < 1e-6
    assert gate.unseen == 0


# --- the test -------------------------------------------------------------------------------------


def test_p_values_are_uniform_over_whole_logs_when_the_class_holds() -> None:
    """``W`` from one plan, the statistic and its law from each of 80 whole logs redrawn under a
    class that holds the truth: the p-values spread as a uniform does."""
    reference, _, plan = _case(10_000, 0, 0.0, 0.0)
    weight = _parameter_weight(plan, reference.residual, None)
    p_values = []
    for seed in range(80):
        ours, theirs = _fits(_log(1000, 2000 + seed, 0.0))
        d = _parameters(theirs.residual) - _parameters(ours.residual)
        spread = np.asarray(theirs.influence) - np.asarray(ours.influence)
        spread = spread.reshape(-1, d.size)
        p_values.append(
            _chi_square_mixture_survival(
                float(d @ weight @ d), _mixture_weights(weight, spread.T @ spread)
            )
        )
    assert stats.kstest(p_values, "uniform").pvalue > 0.01


def test_a_class_that_holds_passes_and_one_that_misses_is_caught() -> None:
    held = misspecification_cost(*_gate_arguments(_case(4000, 7, 0.0, 0.0)))
    missed = misspecification_cost(*_gate_arguments(_case(4000, 7, 0.3, 0.0)))
    assert held.p_value > 1e-3
    assert abs(held.cost) < 3.0 * held.cost_error
    assert missed.p_value < 1e-4
    assert missed.cost > 3.0 * missed.cost_error


def _gate_arguments(
    case: tuple[CausalDynamicsFit, CausalDynamicsFit, CausalPlan],
) -> tuple[CausalPlan, CausalDynamicsFit, CausalDynamicsFit]:
    reference, alternative, plan = case
    return plan, reference, alternative


# --- what it cannot see ---------------------------------------------------------------------------


def _pair_log(rows: int, seed: int) -> dict[str, jax.Array]:
    """Two actions, the second always twice the first: the log never moves ``2 b_1 - b_2``."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (rows, 2))
    z = rng.normal(0.0, 1.0, rows)
    first = 0.9 * z - 0.3 * x[:, 0] + rng.normal(0.0, 0.5, rows)
    u = np.stack([first, 2.0 * first], axis=1)
    rate = x @ A.T + u @ PAIR.T + np.outer(z, C)
    x_next = x + DT * rate + rng.normal(0.0, 0.01, (rows, 2))
    return {
        "x": jnp.asarray(x),
        "u": jnp.asarray(u),
        "x_next": jnp.asarray(x_next),
        "z": jnp.asarray(z[:, None]),
    }


@pytest.mark.parametrize(
    ("held", "unseen"), [(False, 2), (True, 0)], ids=["free", "held to the log's ratio"]
)
def test_a_direction_the_log_never_moved_is_reported_not_priced(held: bool, unseen: int) -> None:
    """The class holds the truth, so the gate has nothing to find where the log looked, and it does
    not fire. Where the log never looked, one direction per state, both fits hold the channel at
    zero: the gate cannot see a miss there, and says so for each direction the plan weighs. A
    plan held to the log's own ratio of the two actions weighs none of them."""
    cost = dataclasses.replace(COST, R=0.05 * jnp.eye(2))
    reference, alternative = _fits(_pair_log(4000, 0))
    constraints = (
        (LinearConstraint(np.kron(np.eye(HORIZON), [[-2.0, 1.0]]), 0.0, 0.0),) if held else ()
    )
    plan = _plan(reference.residual, cost, constraints=constraints)
    gate = misspecification_cost(plan, reference, alternative)
    assert np.shape(reference.unmoved) == (reference.influence.shape[-1], 2)
    assert gate.unseen == unseen
    assert gate.p_value > 1e-3


@functools.cache
def _gate_in(states: tuple[float, float], lever: float, held: bool) -> MisspecificationCost:
    """The gate on the pair log with each state and the first action logged in units of their own,
    the cost, the target, the box, the ratio held and the alternative's weights with them: one
    problem in other units."""
    x_units, u_units = jnp.asarray(states), jnp.array([lever, 1.0])
    data = _pair_log(4000, 0)
    data = dict(data, x=data["x"] * x_units, x_next=data["x_next"] * x_units, u=data["u"] * u_units)
    options = {**FIT, "influence": True}
    reference = fit_causal_residual(_known, data, DT, **options)
    alternative = fit_causal_residual(
        _known, data, DT, weights=lambda x: jnp.exp(x[:, 0] / states[0]), **options
    )
    over = jnp.diag(1.0 / x_units)
    cost = QuadraticCost(
        Q=over @ COST.Q @ over,
        R=0.05 * jnp.diag(1.0 / u_units**2),
        Qf=over @ COST.Qf @ over,
        x_target=COST.x_target * x_units,
    )
    constraints = (
        (LinearConstraint(np.kron(np.eye(HORIZON), [[-2.0 / lever, 1.0]]), 0.0, 0.0),)
        if held
        else ()
    )
    plan = causal_plan(
        _model(reference.residual),
        START * x_units,
        cost,
        DT,
        HORIZON,
        -BOX * u_units,
        BOX * u_units,
        steps=20_000,
        constraints=constraints,
    )
    return misspecification_cost(plan, reference, alternative)


IN_OTHER_UNITS = pytest.mark.parametrize(
    ("states", "lever"),
    [
        ((1e-12, 1e-12), 1.0),
        ((1e12, 1e12), 1.0),
        ((1e-6, 1.0), 1.0),
        ((1e12, 1.0), 1.0),
        ((1.0, 1e12), 1.0),
        ((1.0, 1.0), 1e-3),
        ((1.0, 1.0), 1e6),
    ],
    ids=[
        "both states at 1e-12",
        "both states at 1e12",
        "the first state at 1e-6",
        "the first state at 1e12",
        "the second state at 1e12",
        "the first action at 1e-3",
        "the first action at 1e6",
    ],
)


@IN_OTHER_UNITS
@pytest.mark.parametrize(
    ("held", "unseen"), [(False, 2), (True, 0)], ids=["free", "held to the log's ratio"]
)
def test_what_the_gate_cannot_see_reads_the_same_in_any_units(
    states: tuple[float, float], lever: float, held: bool, unseen: int
) -> None:
    """Read in raw parameter units, where the units set the parameters' sizes apart, the regret's
    curvature along a direction the plan weighs fell under the floor: with both states at 1e12 of
    their units, either state alone at 1e12 or the first action at 1e6, the free plan's two unseen
    directions read none, and with the first state at 1e-6, one. Read at each parameter's scale on
    the log, the two read 5.7e5 and 1.5e7 times the floor in every unit here."""
    assert _gate_in(states, lever, held).unseen == unseen


@IN_OTHER_UNITS
@pytest.mark.parametrize("held", [False, True], ids=["free", "held to the log's ratio"])
def test_the_gate_s_p_value_reads_the_same_in_any_units(
    states: tuple[float, float], lever: float, held: bool
) -> None:
    """The p-value is the tail of a chi-square mixture weighted by the eigenvalues of ``W S``, which
    are the same in any units. Read off ``S``'s square root in raw parameter units, where its small
    eigenvalues were its largest's rounding, the weights summed to 31.7 against ``tr(W S)`` of
    3.3e-5 with both states at 1e-12, and the free plan's p-value read 0.5 against 0.983 at unit
    scale; with the first state at 1e-6 it read 0.998. Read with each parameter at its scale on the
    log, every unit here reads the unit scale's p-value to 1e-12."""
    at_unit_scale = _gate_in((1.0, 1.0), 1.0, held).p_value
    assert abs(_gate_in(states, lever, held).p_value - at_unit_scale) <= 1e-10


# --- refusals -------------------------------------------------------------------------------------


def test_fits_without_their_influence_are_refused() -> None:
    reference, alternative, plan = _case(4000, 7, 0.0, 0.0)
    for bare in (
        dataclasses.replace(reference, influence=None),
        dataclasses.replace(reference, unmoved=None),
    ):
        with pytest.raises(ValueError, match="influence=True"):
            misspecification_cost(plan, bare, alternative)


def test_fits_of_two_classes_or_two_logs_are_refused() -> None:
    reference, _, plan = _case(4000, 7, 0.0, 0.0)
    other_class = fit_causal_residual(
        _known,
        _log(4000, 7, 0.0),
        DT,
        influence=True,
        **{**FIT, "channel_degree": 1},
    )
    with pytest.raises(ValueError, match="different classes"):
        misspecification_cost(plan, reference, other_class)
    other_log, _ = _fits(_log(3000, 7, 0.0))
    with pytest.raises(ValueError, match="one log's rows"):
        misspecification_cost(plan, reference, other_log)


def test_fits_that_sum_over_different_clusters_are_refused() -> None:
    reference, alternative, plan = _case(4000, 7, 0.0, 0.0)
    blocks = np.arange(4000) // 40
    clustered, other = _fits(_log(4000, 7, 0.0), clusters=blocks)
    with pytest.raises(ValueError, match="different clusters"):
        misspecification_cost(plan, reference, other)
    with pytest.raises(ValueError, match="different clusters"):
        misspecification_cost(plan, clustered, alternative)
    shifted = dataclasses.replace(other, clusters=(blocks + 1) % 100)
    with pytest.raises(ValueError, match="different clusters"):
        misspecification_cost(plan, clustered, shifted)


def test_repeated_rows_clustered_by_their_original_price_as_the_original_does() -> None:
    """Each row of a log of 1000 repeated 4 times, the class missing the truth: summed within each
    row's copies, the difference's covariance is the original's summed within each row, so the
    cost, its error, the noise and the p-value are the original's; summed row by row and state by
    state the noise reads a quarter of the original's, as 4000 independent rows would."""
    data = _log(1000, 11, 1.0)
    copies = {name: jnp.repeat(column, 4, axis=0) for name, column in data.items()}
    original = _fits(data, folds=1, clusters=np.arange(1000))
    repeated = _fits(copies, folds=1, clusters=np.repeat(np.arange(1000), 4))

    def priced(fits: tuple[CausalDynamicsFit, CausalDynamicsFit]):
        plan = _plan(fits[0].residual)
        plan = dataclasses.replace(plan, actions=_optimum(fits[0].residual, plan.actions))
        return misspecification_cost(plan, *fits)

    expected, got = priced(original), priced(repeated)
    for name in ("cost", "cost_error", "noise", "p_value"):
        assert getattr(got, name) == pytest.approx(getattr(expected, name), rel=1e-5), name
    assert got.unseen == expected.unseen
    rows = priced(_fits(copies, folds=1))
    each = priced(_fits(data, folds=1))
    assert rows.noise == pytest.approx(each.noise / 4.0, rel=0.01)


def test_two_way_clusters_price_as_each_way_reads_them() -> None:
    """Each of 100 rows logged 10 times, a unit its copies, so the rows alone read less than the
    units: a second way that holds each row alone prices as the first way does, in either order."""
    data = {name: jnp.repeat(column, 10, axis=0) for name, column in _log(100, 11, 1.0).items()}
    units, alone = np.arange(1000) // 10, np.arange(1000)

    def priced(clusters: np.ndarray):
        fits = _fits(data, folds=1, clusters=clusters)
        plan = _plan(fits[0].residual)
        plan = dataclasses.replace(plan, actions=_optimum(fits[0].residual, plan.actions))
        return misspecification_cost(plan, *fits)

    expected = priced(units)
    for labels in (np.column_stack([units, alone]), np.column_stack([alone, units])):
        got = priced(labels)
        for name in ("cost", "cost_error", "noise", "p_value"):
            assert getattr(got, name) == pytest.approx(
                getattr(expected, name), rel=1e-9, abs=0.0
            ), name


def test_two_ways_whose_sums_read_less_noise_than_a_way_alone_price_as_that_way() -> None:
    """On 10 units over 20 periods the units' sums read the most noise, and two ways price as
    the units alone do."""
    data = _log(400, 11, 1.0)
    labels = np.column_stack([np.arange(400) // 40, np.arange(400) % 20])

    def priced(clusters: np.ndarray):
        fits = _fits(data, folds=1, clusters=clusters)
        plan = _plan(fits[0].residual)
        plan = dataclasses.replace(plan, actions=_optimum(fits[0].residual, plan.actions))
        return misspecification_cost(plan, *fits)

    both, units, periods = priced(labels), priced(labels[:, 0]), priced(labels[:, 1])
    assert periods.noise < units.noise
    for name in ("cost", "cost_error", "noise", "p_value"):
        assert getattr(both, name) == pytest.approx(getattr(units, name), rel=1e-9, abs=0.0), name


def test_a_plan_made_on_another_model_is_refused() -> None:
    reference, alternative, plan = _case(4000, 7, 0.0, 0.0)
    with pytest.raises(ValueError, match="reference fit's model"):
        misspecification_cost(plan, alternative, reference)


def test_a_plan_held_under_a_barrier_is_refused() -> None:
    reference, alternative, _ = _case(4000, 7, 0.0, 0.0)
    barrier = BarrierConstraint(lambda x: 5.0 - x[0])
    plan = _plan(reference.residual, barrier=barrier)
    with pytest.raises(ValueError, match="barrier"):
        misspecification_cost(plan, reference, alternative)


# --- the null law's tail --------------------------------------------------------------------------


@pytest.mark.parametrize("terms", [1, 2, 5])
@pytest.mark.parametrize("q", [0.5, 3.0, 12.0, 30.0])
def test_equal_weights_give_the_chi_square_tail(terms: int, q: float) -> None:
    """With one or two terms a plain rule on ``[0, inf)`` reads the tail at 30 as 0.0036 against
    0.00053: the integrand decays like ``u^(-1 - k/2)`` and oscillates."""
    got = _chi_square_mixture_survival(q, np.full(terms, 2.5))
    assert got == pytest.approx(stats.chi2.sf(q / 2.5, terms), rel=1e-7, abs=1e-12)


def _two_term_tail(q: float, first: float, second: float) -> float:
    """``P(first chi2_1 + second chi2_1 > q)`` by one convolution integral, ``x = t^2`` taking the
    density's singularity out."""
    edge = math.sqrt(q / first)
    inside, _ = integrate.quad(
        lambda t: (
            math.sqrt(2.0 / math.pi)
            * math.exp(-0.5 * t * t)
            * stats.chi2.sf((q - first * t * t) / second, 1)
        ),
        0.0,
        edge,
        epsabs=1e-14,
        epsrel=1e-12,
    )
    return float(stats.chi2.sf(q / first, 1)) + inside


@pytest.mark.parametrize("q", [0.2, 1.0, 4.0, 10.0, 20.0])
def test_unequal_weights_give_the_convolution_s_tail(q: float) -> None:
    got = _chi_square_mixture_survival(q, np.array([3.0, 0.7, 0.0]))
    assert got == pytest.approx(_two_term_tail(q, 3.0, 0.7), rel=1e-7, abs=1e-12)


def test_no_weight_is_a_point_mass_at_zero() -> None:
    assert _chi_square_mixture_survival(0.5, np.zeros(3)) == 0.0
    assert _chi_square_mixture_survival(0.0, np.zeros(3)) == 1.0
