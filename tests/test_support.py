"""Pessimism gate: support penalty keeps control near the data; greedy control extrapolates."""

import importlib
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chc import HybridDynamics, QuadraticCost
from chc.control import projected_gradient_control
from chc.cost import total_cost
from chc.dynamics import DampedOscillator
from chc.integrate import rollout
from chc.residual import ZeroResidual
from chc.support import SupportModel, pessimistic_control

# Public as jax.enable_x64 from jax 0.8.0; the floor, 0.4.30, has only jax.experimental.enable_x64,
# which jax 0.11 no longer has.
if hasattr(jax, "enable_x64"):
    enable_x64 = jax.enable_x64
else:
    enable_x64 = importlib.import_module("jax.experimental").enable_x64

DT = 0.1


def test_pessimism_keeps_control_in_support() -> None:
    k_x, k_u = jax.random.split(jax.random.key(0))
    xs_data = jax.random.normal(k_x, (2000, 2))  # x ~ N(0, I)
    us_data = 0.3 * jax.random.normal(k_u, (2000, 1))  # u ~ N(0, 0.3^2): narrow support
    support = SupportModel.fit(xs_data, us_data)

    model = HybridDynamics(
        known=DampedOscillator(omega=1.0, zeta=0.1), residual=ZeroResidual(out_dim=2)
    )
    # cheap control + a far target ⇒ the greedy optimum slams the actuator far beyond the u-support
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.0])),
        R=jnp.array([[0.001]]),
        Qf=jnp.diag(jnp.array([10.0, 1.0])),
        x_target=jnp.array([-3.0, 0.0]),
    )
    x0 = jnp.zeros(2)
    us0 = jnp.zeros((20, 1))

    us_greedy, _ = projected_gradient_control(model, x0, us0, DT, cost, -5.0, 5.0, steps=200)
    us_pess, _ = pessimistic_control(
        model, x0, us0, DT, cost, support, lam_supp=20.0, u_lo=-5.0, u_hi=5.0, steps=200
    )

    greedy_max = float(jnp.max(jnp.abs(us_greedy)))
    pess_max = float(jnp.max(jnp.abs(us_pess)))
    assert greedy_max > 2.0  # greedy extrapolates far past the u-support (~0.3)
    assert pess_max < 0.6 * greedy_max  # pessimism shrinks control toward the data

    xs_greedy = rollout(model, x0, us_greedy, DT)
    xs_pess = rollout(model, x0, us_pess, DT)
    assert float(jnp.max(jnp.abs(xs_pess[:, 0]))) < float(jnp.max(jnp.abs(xs_greedy[:, 0])))


# --- the penalised descent must reproduce the loop it replaced ----------------------------------

_ULP_BUDGET = 500.0


def _naive_pessimistic(
    model: object,
    x0: jnp.ndarray,
    us0: jnp.ndarray,
    cost: QuadraticCost,
    support: SupportModel,
    lam_supp: float,
    u_lo: float,
    u_hi: float,
    steps: int,
    lr0: float = 0.2,
    tol: float = 1e-9,
) -> tuple[jnp.ndarray, list[float]]:
    """The plain Python recursion for the penalised objective, kept as the oracle."""

    def task(us: jnp.ndarray) -> jnp.ndarray:
        return total_cost(model, x0, us, DT, cost)

    def augmented(us: jnp.ndarray) -> jnp.ndarray:
        xs = rollout(model, x0, us, DT)
        return task(us) + lam_supp * support.penalty_trajectory(xs[:-1], us)

    grad_aug = jax.grad(augmented)
    us = jnp.clip(us0, u_lo, u_hi)
    current = augmented(us)
    history = [float(task(us))]
    for _ in range(steps):
        grad = grad_aug(us)
        lr, improved, candidate, candidate_cost = lr0, False, us, current
        for _ls in range(40):
            candidate = jnp.clip(us - lr * grad, u_lo, u_hi)
            candidate_cost = augmented(candidate)
            if candidate_cost < current - tol:
                improved = True
                break
            lr *= 0.5
        if not improved:
            break
        us, current = candidate, candidate_cost
        history.append(float(task(us)))
    return us, history


def _ulp_gap(a: jnp.ndarray, b: jnp.ndarray) -> float:
    left, right = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    spacing = np.spacing(np.maximum(np.abs(left), np.abs(right)))
    return float(np.max(np.abs(left - right) / np.maximum(spacing, np.finfo(np.float64).tiny)))


def test_compiled_pessimistic_descent_matches_the_python_recursion() -> None:
    k_x, k_u = jax.random.split(jax.random.key(4))
    support = SupportModel.fit(
        jax.random.normal(k_x, (500, 2)), 0.3 * jax.random.normal(k_u, (500, 1))
    )
    model = HybridDynamics(
        known=DampedOscillator(omega=1.0, zeta=0.1), residual=ZeroResidual(out_dim=2)
    )
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.0])),
        R=jnp.array([[0.001]]),
        Qf=jnp.diag(jnp.array([10.0, 1.0])),
        x_target=jnp.array([-3.0, 0.0]),
    )
    x0, us0 = jnp.zeros(2), jnp.zeros((15, 1))

    us_ref, history_ref = _naive_pessimistic(
        model, x0, us0, cost, support, 20.0, -5.0, 5.0, steps=60
    )
    us, history = pessimistic_control(
        model, x0, us0, DT, cost, support, lam_supp=20.0, u_lo=-5.0, u_hi=5.0, steps=60
    )

    assert len(history) == len(history_ref)
    assert _ulp_gap(us, us_ref) < _ULP_BUDGET
    assert _ulp_gap(history, jnp.asarray(history_ref)) < _ULP_BUDGET


def test_the_compiled_pessimistic_solver_amortises_across_calls() -> None:
    # The regression this guards: the jitted augmented objective and its gradient used to be built
    # inside pessimistic_control, so every solve recompiled them. Odd sizes keep the compilation
    # key off every other test's.
    k_x, k_u = jax.random.split(jax.random.key(5))
    support = SupportModel.fit(
        jax.random.normal(k_x, (300, 2)), 0.3 * jax.random.normal(k_u, (300, 1))
    )
    model = HybridDynamics(
        known=DampedOscillator(omega=1.3, zeta=0.1), residual=ZeroResidual(out_dim=2)
    )
    cost = QuadraticCost(
        Q=jnp.eye(2), R=jnp.array([[0.01]]), Qf=jnp.eye(2), x_target=jnp.array([-1.0, 0.0])
    )
    x0, us0 = jnp.zeros(2), jnp.zeros((17, 1))

    def solve_seconds() -> float:
        start = time.perf_counter()
        jax.block_until_ready(
            pessimistic_control(
                model, x0, us0, DT, cost, support, lam_supp=3.0, u_lo=-5.0, u_hi=5.0, steps=41
            )
        )
        return time.perf_counter() - start

    cold = solve_seconds()
    warm = min(solve_seconds(), solve_seconds())
    assert warm < 0.5 * cold


# --- the distance is counted in the log's own spread, whatever units the log was kept in --------


def _cloud() -> tuple[jax.Array, jax.Array]:
    """Two states of unit spread and one action of spread 0.4: the support-shift task's log."""
    k_x, k_u = jax.random.split(jax.random.key(0))
    return jax.random.normal(k_x, (2000, 2)), 0.4 * jax.random.normal(k_u, (2000, 1))


@pytest.mark.parametrize("coordinate", [0, 2], ids=["a state", "the action"])
def test_a_point_off_the_log_scores_the_same_in_any_units(coordinate: int) -> None:
    """The ridge was ``1e-3 * I`` in the caller's units, so a coordinate logged in small ones was
    outweighed by it: at 1e-3 of the state's units a point 4 standard deviations off along it
    scored 0.016 where it scored 16.0, and at 1e-6 of the action's units 2.7e-9 where it scored
    15.9, so a plan off the logged support read as on it."""
    xs, us = _cloud()
    z = np.asarray(jnp.concatenate([xs, us], axis=1))
    mean, spread = z.mean(axis=0), z.std(axis=0)
    off = np.zeros(3)
    off[coordinate] = 4.0
    points = mean + np.array([[0.5, -0.3, 0.1], off, [-1.0, 2.0, -1.5]]) * spread

    def scores(s: float) -> np.ndarray:
        units = np.ones(3)
        units[coordinate] = s
        model = SupportModel.fit(xs * units[:2], us * units[2:])
        return np.array(
            [float(model.squared_distance(p[:2], p[2:])) for p in jnp.asarray(points * units)]
        )

    reference = scores(1.0)
    assert 15.0 < reference[1] < 17.0  # about 16 / (1 + ridge): the log barely couples them
    for s in (1e-9, 1e-6, 1e-3, 1e3, 1e6):
        assert scores(s) == pytest.approx(reference, rel=1e-12, abs=0.0), s


@pytest.mark.parametrize("coordinate", [0, 2], ids=["a state", "the action"])
def test_a_point_off_the_log_scores_the_same_with_a_coordinate_far_from_zero(
    coordinate: int,
) -> None:
    """The distance is counted from the log's own mean, in each coordinate's spread about it, so a
    coordinate raised by 1e3 of its spread and logged at 1e-6 of its units scores the same. With
    the ridge in the caller's units, a point 4 standard deviations off along the state read 1.6e-8
    there where it read 16.0, and along the action 2.7e-9 where it read 15.9."""
    xs, us = _cloud()
    z = np.asarray(jnp.concatenate([xs, us], axis=1))
    mean, spread = z.mean(axis=0), z.std(axis=0)
    off = np.zeros(3)
    off[coordinate] = 4.0
    points = mean + np.array([[0.5, -0.3, 0.1], off, [-1.0, 2.0, -1.5]]) * spread

    def scores(scale: float, origin: float) -> np.ndarray:
        shift, units = np.zeros(3), np.ones(3)
        shift[coordinate], units[coordinate] = origin * spread[coordinate], scale
        moved = (z + shift) * units
        model = SupportModel.fit(jnp.asarray(moved[:, :2]), jnp.asarray(moved[:, 2:]))
        return np.array(
            [
                float(model.squared_distance(p[:2], p[2:]))
                for p in jnp.asarray((points + shift) * units)
            ]
        )

    # a level 1e3 spreads off zero keeps 1e-13 of the spread in each entry; measured 2e-13
    assert scores(1e-6, 1e3) == pytest.approx(scores(1.0, 0.0), rel=1e-11, abs=0.0)


def test_coordinates_the_log_moved_in_step_keep_a_finite_distance_across_their_line() -> None:
    """An action that copied a state leaves the covariance singular along their difference, and the
    ridge alone keeps a distance across that line: half a standard deviation off it along each
    scores ``0.5 / ridge``, 500, in any units. The absolute ridge outweighed an action logged in
    small units and dropped the line: with the action at 1e-6 of the state's units the same point
    scored 0.25, at 1 of them 511, and at 1e3 of them 1022."""
    xs, _ = _cloud()
    z = np.asarray(xs)
    mean, spread = z.mean(axis=0), z.std(axis=0)
    off = np.array([mean[0] + 0.5 * spread[0], mean[1], mean[0] - 0.5 * spread[0]])

    def score(s: float) -> float:
        model = SupportModel.fit(xs, s * xs[:, :1])
        point = jnp.asarray(off * np.array([1.0, 1.0, s]))
        return float(model.squared_distance(point[:2], point[2:]))

    reference = score(1.0)
    assert 400.0 < reference < 600.0  # about 0.5 / ridge: the log barely couples the other state
    for s in (1e-9, 1e-6, 1e-3, 1e3, 1e6):
        assert score(s) == pytest.approx(reference, rel=1e-9, abs=0.0), s


@pytest.mark.parametrize(
    ("held", "named"),
    [(("action", 0.0), "action 1"), (("action", 0.3), "action 1"), (("state", 1.5), "state 0")],
    ids=["an action held at zero", "an action held at 0.3", "a state held at 1.5"],
)
def test_a_coordinate_the_log_never_moved_is_refused(held: tuple[str, float], named: str) -> None:
    """A coordinate the log never moved has no spread for a deviation to be counted in, so its
    distance is unbounded in any units, and no ridge relative to its own spread can stand in for
    one. The absolute ridge stood in: one unit off an action held at 0.3 scored 1000, in whatever
    units it was logged in. Raised by each coordinate's own variance instead, the precision came
    out nan for an action held at 0, and for one held at 0.3 the mean's rounding alone scored 1.0
    at the held level itself and 3e32 one unit off it."""
    xs, us = _cloud()
    kind, level = held
    if kind == "action":
        us = jnp.concatenate([us, jnp.full((us.shape[0], 1), level)], axis=1)
    else:
        xs = xs.at[:, 0].set(level)
    with pytest.raises(ValueError, match=f"the log never moved {named}:"):
        SupportModel.fit(xs, us)


def _states_and_signs() -> tuple[jax.Array, jax.Array]:
    """The support-shift task's two states, and an action of +-1 drawn at random, in the precision
    JAX is set to."""
    k_x, k_s = jax.random.split(jax.random.key(0))
    signs = jnp.where(jax.random.bernoulli(k_s, 0.5, (2000, 1)), 1.0, -1.0)
    return jax.random.normal(k_x, (2000, 2)), signs


_ROUNDING = [
    *[(True, level) for level in (1e-300, 1e-6, 1.0, 1e6, 1e300)],
    *[(False, level) for level in (1e-30, 1.0, 1e30)],
]


@pytest.mark.parametrize(
    ("double", "level"),
    _ROUNDING,
    ids=[f"{'double' if double else 'single'} at {level:g}" for double, level in _ROUNDING],
)
def test_a_coordinate_the_log_moved_by_rounding_alone_is_refused_in_any_units(
    double: bool, level: float
) -> None:
    """An action at one level but for rounding, 18 eps of it either way in double precision and 8
    in single, never moved: a spread of at most 64 eps of its size is rounding, so it is refused as
    a constant one is. Fitted, its distance was counted in its rounding: a point 1e3 of those
    jitters off its level, 4e-12 of it, scored 1e6, and in single precision a point 1e-3 off. At
    1e-300 and 1e300 of its units, and at 1e-30 and 1e30 in single precision, the precision came
    out nan."""
    jitter = 4e-15 if double else 1e-6
    with enable_x64(double):
        xs, signs = _states_and_signs()
        action = level * (1.0 + jitter * signs)
        with pytest.raises(ValueError, match="never moved action 0: every row holds one value, up"):
            SupportModel.fit(xs, action)


_DATA = [
    *[(True, level) for level in (1e-100, 1.0, 1e100, 1e160)],
    *[(False, level) for level in (1e-10, 1.0, 1e10, 1e20)],
]


@pytest.mark.parametrize(
    ("double", "level"),
    _DATA,
    ids=[f"{'double' if double else 'single'} at {level:g}" for double, level in _DATA],
)
def test_a_coordinate_apart_by_1e3_eps_of_its_size_is_counted_in_its_own_spread(
    double: bool, level: float
) -> None:
    """An action at one level but for 1e3 eps of it either way, 16 times the bound, is data: it
    fits, and a point 4 of its standard deviations off along it scores what it scores with the
    action logged as the +-1 it was drawn as, about 16. The bound is read on each column as a share
    of its largest entry, so at 1e160 of its units, and 1e20 in single precision, where the
    column's own squares overflow, it is still data."""
    eps = float(np.finfo(np.float64 if double else np.float32).eps)
    with enable_x64(double):
        xs, signs = _states_and_signs()
        action = level * (1.0 + 1e3 * eps * signs)

        def score(logged: jax.Array) -> float:
            model = SupportModel.fit(xs, logged)
            point = jnp.mean(logged) + 4.0 * jnp.std(logged)
            return float(model.squared_distance(jnp.mean(xs, axis=0), jnp.reshape(point, (1,))))

        # the action keeps about 3 digits of its +-1, 1 ulp against 1e3: measured 1.5e-3 at most
        assert score(action) == pytest.approx(score(signs), rel=1e-2, abs=0.0)


_OUTSIDE = [(True, 1e200), (True, 1e-200), (False, 1e25), (False, 1e-25)]


@pytest.mark.parametrize(
    ("double", "units"),
    _OUTSIDE,
    ids=[f"{'double' if double else 'single'} at {units:g}" for double, units in _OUTSIDE],
)
def test_a_spread_the_precision_cannot_hold_in_the_caller_s_units_is_refused(
    double: bool, units: float
) -> None:
    """In the caller's units the precision along the action is the inverse of its spread squared:
    6e-400 at 1e200 of the action's units, which rounds to 0 and reads every point as on the
    support, and 6e400 at 1e-200, which overflows. Formed from the deviations in those units, the
    covariance itself overflowed or underflowed and the precision read nan. With the ridge
    absolute, it read 0 at 1e200, and at 1e-200 a point 4 standard deviations off scored 9e-34, as
    if on the support. In single precision the same holds at 1e25 and 1e-25 of them, where the
    inverse square is 6e-50 and 6e50."""
    with enable_x64(double):
        xs, us = _cloud()
        with pytest.raises(
            ValueError, match=r"spread along action 0 \(spread .*\) is outside what"
        ):
            SupportModel.fit(xs, units * us)


_EDGES = [(True, 1.5e154), (True, 2e-154), (False, 2e19), (False, 2e-19)]


@pytest.mark.parametrize(
    ("double", "units"),
    _EDGES,
    ids=[f"{'double' if double else 'single'} at {units:g}" for double, units in _EDGES],
)
def test_a_spread_at_the_edge_of_what_the_precision_holds_scores_as_at_unit_scale(
    double: bool, units: float
) -> None:
    """The covariance is formed from each column as a share of its largest entry, so only the
    precision's own range bounds the spreads that fit: from 7.5e-155 to 6.7e153 in double
    precision, and from 5.4e-20 to 9.2e18 in single. Near both ends the scores are the unit-scale
    ones, to 4.4e-16 in double precision and 3.6e-7 in single. Formed from the deviations in the
    caller's units, the covariance overflowed from a spread of about 3e152 in double precision, as
    the sum of 2000 squares, and at the other end its squares fell below the smallest normal
    number: the precision read nan at 8e-155, and the scores were 18% off at 1.6e-154."""
    with enable_x64(double):
        xs, us = _cloud()
        z = np.asarray(jnp.concatenate([xs, us], axis=1), dtype=np.float64)
        mean, spread = z.mean(axis=0), z.std(axis=0)
        points = mean + np.array([[0.5, -0.3, 0.1], [0.0, 0.0, 4.0], [-1.0, 2.0, -1.5]]) * spread

        def scores(s: float) -> np.ndarray:
            model = SupportModel.fit(xs, s * us)
            moved = jnp.asarray(points * np.array([1.0, 1.0, s]), dtype=xs.dtype)
            return np.array([float(model.squared_distance(p[:2], p[2:])) for p in moved])

        rel = 1e-12 if double else 1e-5
        assert scores(units) == pytest.approx(scores(1.0), rel=rel, abs=0.0)
