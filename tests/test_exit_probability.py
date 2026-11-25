"""The K-step exit probability of a closed loop (``chc.barrier.exit_probability``).

Each bound is set against the loop it bounds: the exact exit probability of a scalar Gaussian
loop, by quadrature on the survival recursion, and the exit frequency of seeded Monte Carlo on a
scalar loop, a double integrator and a safety filter. The algebra is
``validation/exit_probability.mac``'s.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable
from itertools import pairwise

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import minimize
from scipy.stats import norm

import chc
from chc.barrier import (
    ExitProbability,
    QuadraticBarrier,
    _exit_bound,
    exit_probability,
    robust_safety_filter,
)
from chc.dynamics import HybridDynamics, LinearDynamics
from chc.evaluation import AffinePolicy, AffineSchedule, LinearGaussianPlant
from chc.plan import CausalPlan, SafetyCertificate, causal_plan, certify_safety
from chc.residual import ZeroResidual

# x' = 0.9 x + 0.5 u + w under u = -0.6 x: the loop's factor is 0.6, w ~ N(0, 0.45^2)
_SCALAR_PLANT = LinearGaussianPlant([[0.9]], [[0.5]], [0.0], [[0.45**2]])
_SCALAR_POLICY = AffinePolicy([[-0.6]], [0.0], [[0.0]])
_CORRIDOR = QuadraticBarrier(1.0, [0.0], [[1.0]])  # |x| <= 1

# a double integrator, dt = 0.1, under u = -4 x1 - 3 x2 + nu, nu ~ N(0, 25); the ellipse is the
# loop's Lyapunov form, centred off the loop's rest point
_DT = 0.1
_A2 = np.array([[1.0, _DT], [0.0, 1.0]])
_B2 = np.array([[_DT**2 / 2], [_DT]])
_GAIN2 = np.array([[-4.0, -3.0]])
_PLANT2 = LinearGaussianPlant(_A2, _B2, np.zeros(2), np.zeros((2, 2)))
_POLICY2 = AffinePolicy(_GAIN2, [0.0], [[25.0]])
_LYAPUNOV = solve_discrete_lyapunov((_A2 + _B2 @ _GAIN2).T, np.eye(2))
_ELLIPSE = QuadraticBarrier(1.0, [0.1, 0.0], _LYAPUNOV / _LYAPUNOV[0, 0])


def _h(barrier: QuadraticBarrier, x: np.ndarray) -> np.ndarray:
    gap = x - barrier.centre
    return barrier.upper - np.einsum("...i,ij,...j->...", gap, barrier.shape, gap)


def _exit_frequency(
    step: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    barrier: QuadraticBarrier,
    x0: np.ndarray,
    steps: int,
    paths: int,
    seed: int,
    level: float = 0.0,
) -> np.ndarray:
    """The share of seeded paths with ``h(x_k) < -level`` at some ``k <= K``, for each ``K``."""
    rng = np.random.default_rng(seed)
    x = np.tile(np.asarray(x0, dtype=np.float64), (paths, 1))
    left = _h(barrier, x) < -level
    shares = []
    for _ in range(steps):
        x = step(x, rng)
        left |= _h(barrier, x) < -level
        shares.append(left.mean())
    return np.array(shares)


def _affine_step(
    plant: LinearGaussianPlant, policy: AffinePolicy
) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    def step(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n, m = plant.states, plant.actions
        nu = rng.multivariate_normal(np.zeros(m), policy.covariance, size=len(x))
        w = rng.multivariate_normal(np.zeros(n), plant.noise, size=len(x))
        u = x @ policy.gain.T + policy.offset + nu
        return x @ plant.a.T + u @ plant.b.T + plant.offset + w

    return step


def _scalar_exact_exit(
    rho: float, sd: float, x0: float, steps: int, nodes: int = 400
) -> list[float]:
    """``P(max_{k <= K} |x_k| > 1)`` for ``x' = rho x + N(0, sd^2)``, by Gauss-Legendre
    quadrature on the survival recursion ``S_k(x) = int_{-1}^{1} p(y | x) S_{k-1}(y) dy``."""
    y, w = np.polynomial.legendre.leggauss(nodes)
    kernel = norm.pdf(y[None, :], loc=rho * y[:, None], scale=sd) * w[None, :]
    start = norm.pdf(y, loc=rho * x0, scale=sd) * w
    survival = np.ones(nodes)
    exits = []
    for _ in range(steps):
        exits.append(1.0 - float(start @ survival))
        survival = kernel @ survival
    return exits


def test_the_bound_holds_the_exact_exit_probability_of_a_scalar_loop() -> None:
    """At alpha = 0.6^2 the margin 0.64 - 0.45^2 - (0.36 - 0.6^2) x^2 is flat, so the slack is
    0.4375 everywhere and the bound 1 - (1 - 0.45^2)^K; the truth is several times smaller."""
    exact = _scalar_exact_exit(0.6, 0.45, 0.0, 10)
    bounds = [
        exit_probability(_CORRIDOR, _SCALAR_PLANT, _SCALAR_POLICY, [0.0], steps=k, alpha=0.36).bound
        for k in range(1, 11)
    ]
    assert all(e <= b for e, b in zip(exact, bounds, strict=True))
    for k, e, b in ((1, 0.0263, 0.2025), (10, 0.420, 0.896)):  # the docstring's numbers
        assert exact[k - 1] == pytest.approx(e, rel=0.0, abs=5e-4)
        assert bounds[k - 1] == pytest.approx(b, rel=0.0, abs=5e-4)
    assert bounds[0] == pytest.approx(1.0 - (1.0 - 0.45**2), rel=1e-12, abs=0.0)
    reading = exit_probability(
        _CORRIDOR, _SCALAR_PLANT, _SCALAR_POLICY, [0.0], steps=10, alpha=0.36
    )
    assert reading.slack == pytest.approx(0.4375, rel=1e-12, abs=0.0)
    assert (reading.branch, reading.scope) == ("inward", "everywhere")


@pytest.mark.parametrize(
    ("barrier", "plant", "policy", "x0", "alpha", "steps", "seed"),
    [
        (_CORRIDOR, _SCALAR_PLANT, _SCALAR_POLICY, [0.0], 0.36, 10, 11),
        (_CORRIDOR, _SCALAR_PLANT, _SCALAR_POLICY, [0.5], 0.8, 6, 12),
        (_ELLIPSE, _PLANT2, _POLICY2, [0.3, 0.0], 0.95, 20, 13),
    ],
    ids=["scalar-inward", "scalar-outward", "double-integrator"],
)
def test_the_exit_frequency_of_an_affine_loop_stays_under_its_bound(
    barrier: QuadraticBarrier,
    plant: LinearGaussianPlant,
    policy: AffinePolicy,
    x0: list[float],
    alpha: float,
    steps: int,
    seed: int,
) -> None:
    frequency = _exit_frequency(
        _affine_step(plant, policy), barrier, np.array(x0), steps, 20_000, seed
    )
    for k in (1, steps // 2, steps):
        reading = exit_probability(barrier, plant, policy, x0, steps=k, alpha=alpha)
        assert frequency[k - 1] <= reading.bound
    assert frequency[-1] > 0.02  # neither side is trivial
    assert reading.bound < 1.0


def test_the_two_dimensional_slack_is_the_exact_one() -> None:
    """``validation/exit_probability.mac`` STEP 6 reads -2357033/2420101800 in exact rationals,
    the outward branch. A numerical search for the margin's least value finds it, and the margin
    read at a grid of states by the callable path never falls below it."""
    reading = exit_probability(_ELLIPSE, _PLANT2, _POLICY2, [0.3, 0.0], steps=20, alpha=0.95)
    assert reading.slack == pytest.approx(-2357033 / 2420101800, rel=1e-12, abs=0.0)
    assert reading.branch == "outward"
    noise = _B2 @ _POLICY2.covariance @ _B2.T
    loop = _A2 + _B2 @ _GAIN2

    def margin(x: np.ndarray) -> float:
        noise_cost = float(np.trace(_ELLIPSE.shape @ noise))
        return _ELLIPSE(loop @ x) - noise_cost - 0.95 * _ELLIPSE(x)

    searched = min(
        minimize(margin, start, method="BFGS", options={"gtol": 1e-12}).fun
        for start in ([0.0, 0.0], [1.0, -2.0], [-1.0, 3.0])
    )
    assert searched == pytest.approx(reading.slack, rel=1e-8, abs=0.0)
    plant = LinearGaussianPlant(_A2, _B2, np.zeros(2), noise)
    axis = np.linspace(-1.5, 1.5, 61)
    grid = np.stack(np.meshgrid(axis, 3 * axis), axis=-1).reshape(-1, 2)
    gridded = exit_probability(
        _ELLIPSE, plant, lambda x: _GAIN2 @ x, [0.3, 0.0], steps=20, alpha=0.95, states=grid
    )
    assert gridded.scope == "states"
    assert gridded.slack == pytest.approx(min(margin(x) for x in grid), rel=1e-12, abs=0.0)
    assert reading.slack < gridded.slack


def test_leaving_a_wider_set_is_bounded_more_tightly() -> None:
    frequency = _exit_frequency(
        _affine_step(_SCALAR_PLANT, _SCALAR_POLICY), _CORRIDOR, np.zeros(1), 10, 20_000, 14, 0.2
    )
    plain = exit_probability(_CORRIDOR, _SCALAR_PLANT, _SCALAR_POLICY, [0.0], steps=10, alpha=0.5)
    wider = exit_probability(
        _CORRIDOR, _SCALAR_PLANT, _SCALAR_POLICY, [0.0], steps=10, alpha=0.5, level=0.2
    )
    assert frequency[-1] <= wider.bound < plain.bound
    assert frequency[-1] > 0.0


def test_the_bound_tightens_with_the_start_s_room_and_loosens_with_the_horizon() -> None:
    def bound(x0: float, steps: int) -> float:
        return exit_probability(
            _CORRIDOR, _SCALAR_PLANT, _SCALAR_POLICY, [x0], steps=steps, alpha=0.36
        ).bound

    by_start = [bound(x0, 5) for x0 in (0.9, 0.6, 0.3, 0.0)]
    assert all(a > b for a, b in pairwise(by_start))
    by_horizon = [bound(0.3, k) for k in (1, 2, 5, 10, 20)]
    assert all(a < b for a, b in pairwise(by_horizon))


@pytest.mark.parametrize("alpha", [0.5, 0.7, 0.9])
def test_a_loop_that_meets_the_condition_with_no_slack_reads_the_plain_form(alpha: float) -> None:
    """``x' = sqrt(alpha) x + w``, ``Var w = 1 - alpha`` on ``h = 1 - x^2``: ``E[h(x')|x] =
    alpha h(x)`` exactly, and the bound is ``1 - alpha^K h0``, tighter as ``alpha`` grows."""
    plant = LinearGaussianPlant([[1.0]], [[1.0]], [0.0], [[1.0 - alpha]])
    policy = AffinePolicy([[np.sqrt(alpha) - 1.0]], [0.0], [[0.0]])
    reading = exit_probability(_CORRIDOR, plant, policy, [0.4], steps=8, alpha=alpha)
    assert reading.slack == pytest.approx(0.0, rel=0.0, abs=1e-14)
    assert reading.bound == pytest.approx(1.0 - alpha**8 * (1.0 - 0.4**2), rel=1e-12, abs=0.0)


def test_the_closed_form_tightens_with_alpha_and_the_start_as_stated() -> None:
    """At a fixed slack: ``1 - alpha^K h0/M`` and its slack-bearing forms fall as alpha or h0
    rises, on either branch; the branches meet at slack = -level (1 - alpha)."""
    for slack, level in ((0.0, 0.0), (0.05, 0.0), (0.02, 0.3)):
        by_alpha = [_exit_bound(0.6, 10, a, 1.0, slack, level)[0] for a in (0.5, 0.7, 0.9)]
        assert all(a > b for a, b in pairwise(by_alpha))
    for slack, level in ((0.0, 0.0), (-0.01, 0.0), (0.05, 0.2)):
        by_start = [_exit_bound(h0, 10, 0.8, 1.0, slack, level)[0] for h0 in (0.2, 0.5, 0.9)]
        assert all(a > b for a, b in pairwise(by_start))
    seam = -0.3 * (1.0 - 0.8)
    inward = _exit_bound(0.6, 7, 0.8, 1.0, seam, 0.3)
    outward = _exit_bound(0.6, 7, 0.8, 1.0, seam - 1e-12, 0.3)
    assert (inward[1], outward[1]) == ("inward", "outward")
    assert inward[0] == pytest.approx(outward[0], rel=1e-9, abs=0.0)


def test_a_start_outside_the_set_has_left_it() -> None:
    reading = exit_probability(_CORRIDOR, _SCALAR_PLANT, _SCALAR_POLICY, [1.2], steps=3, alpha=0.5)
    assert reading.initial < 0.0
    assert reading.bound == 1.0


def _filter_loop() -> tuple[LinearGaussianPlant, Callable[[np.ndarray], np.ndarray], Callable]:
    """``x' = x + dt (0.3 x + u) + w`` driven toward 3 and filtered at 1.6 by
    ``robust_safety_filter``, inside the corridor ``[-1, 2]``; the filter in closed form beside it
    for the Monte Carlo."""
    drift, dt, x_ref, x_filter, u_max = 0.3, 0.1, 3.0, 1.6, 6.0
    plant = LinearGaussianPlant([[1.0 + dt * drift]], [[dt]], [0.0], [[0.1**2]])

    def nominal(x: np.ndarray) -> np.ndarray:
        return np.clip((x_ref - x) / dt - drift * x, -u_max, u_max)

    def policy(x: np.ndarray) -> np.ndarray:
        state = float(x[0])
        u = robust_safety_filter(
            float(nominal(x)[0]),
            channel=-1.0,
            radius=0.0,
            u_max=u_max,
            drift=-drift * state,
            alpha_h=x_filter - state,
        )
        return np.array([u])

    def closed_form(x: np.ndarray) -> np.ndarray:
        return np.clip(np.minimum(nominal(x), x_filter - x - drift * x), -u_max, u_max)

    return plant, policy, closed_form


def test_the_exit_frequency_of_a_filtered_loop_stays_under_its_bound() -> None:
    plant, policy, closed_form = _filter_loop()
    corridor = QuadraticBarrier(1.5**2, [0.5], [[1.0]])
    states = np.linspace(-1.0, 2.0, 301)[:, None]
    for x in np.linspace(-4.0, 7.0, 221):
        assert closed_form(np.array([x]))[0] == policy(np.array([x]))[0]

    def step(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        noise = 0.1 * rng.standard_normal(x.shape)
        return x @ plant.a.T + closed_form(x) @ plant.b.T + noise

    frequency = _exit_frequency(step, corridor, np.zeros(1), 40, 20_000, 15)
    for k in (10, 20, 40):
        reading = exit_probability(
            corridor, plant, policy, [0.0], steps=k, alpha=0.95, states=states
        )
        assert reading.scope == "states"
        assert frequency[k - 1] <= reading.bound < 1.0
    assert frequency[-1] > 0.05


def test_a_callable_is_read_only_at_the_states_given() -> None:
    """States that miss the filter's edge read a larger slack than states that cover it: the
    scope says so, and the caller chooses what to cover."""
    plant, policy, _ = _filter_loop()
    corridor = QuadraticBarrier(1.5**2, [0.5], [[1.0]])
    covering = np.linspace(-1.0, 2.0, 301)[:, None]
    near_start = np.linspace(-0.2, 0.2, 5)[:, None]
    wide = exit_probability(corridor, plant, policy, [0.0], steps=20, alpha=0.95, states=covering)
    narrow = exit_probability(
        corridor, plant, policy, [0.0], steps=20, alpha=0.95, states=near_start
    )
    assert narrow.slack > wide.slack
    assert narrow.bound < wide.bound
    # the slack is the least margin over x0 and every state given, read here apart
    read = np.vstack([np.zeros((1, 1)), covering])
    least = min(
        corridor(plant.a @ x + plant.b @ policy(x)) - 0.1**2 - 0.95 * corridor(x) for x in read
    )
    assert wide.slack == pytest.approx(least, rel=1e-12, abs=0.0)


_MODEL = HybridDynamics(
    known=LinearDynamics(jnp.array([[-0.5, 1.0], [0.0, -0.3]]), jnp.array([[0.0], [1.0]])),
    residual=ZeroResidual(2),
)


def _plan() -> CausalPlan:
    cost = chc.QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.1])),
        R=jnp.array([[0.05]]),
        Qf=jnp.diag(jnp.array([5.0, 1.0])),
        x_target=jnp.zeros(2),
    )
    return causal_plan(_MODEL, jnp.array([1.0, 0.0]), cost, 0.1, 12, -5.0, 5.0)


def test_an_open_loop_plan_is_refused() -> None:
    plant = LinearGaussianPlant(_A2, _B2, np.zeros(2), np.eye(2) * 1e-3)
    schedule = AffineSchedule.open_loop(np.ones((5, 1)), states=2)
    for plan in (schedule, _plan(), np.ones((5, 1))):
        with pytest.raises(TypeError, match="open loop"):
            exit_probability(_ELLIPSE, plant, plan, [0.3, 0.0], steps=5, alpha=0.9)  # type: ignore[arg-type]
    held = AffinePolicy(np.zeros((1, 2)), [1.0], [[0.0]])
    with pytest.raises(ValueError, match="open-loop plan"):
        exit_probability(_ELLIPSE, plant, held, [0.3, 0.0], steps=5, alpha=0.9)


def test_a_drift_condition_that_holds_at_no_slack_is_refused() -> None:
    # an unstable loop: alpha - 1.2^2 < 0 at every alpha
    unstable = AffinePolicy([[0.6]], [0.0], [[0.0]])
    for alpha in (0.5, 0.99):
        with pytest.raises(ValueError, match="holds at no slack"):
            exit_probability(_CORRIDOR, _SCALAR_PLANT, unstable, [0.0], steps=5, alpha=alpha)
    # a stable loop read below its contraction, alpha < 0.6^2
    with pytest.raises(ValueError, match="holds at no slack"):
        exit_probability(_CORRIDOR, _SCALAR_PLANT, _SCALAR_POLICY, [0.0], steps=5, alpha=0.3)
    # the position alone on a double integrator: the velocity moves it, det Q = -alpha F12^2
    position = QuadraticBarrier(1.0, [0.0, 0.0], [[1.0, 0.0], [0.0, 0.0]])
    for alpha in (0.5, 0.9, 0.999):
        with pytest.raises(ValueError, match="holds at no slack"):
            exit_probability(position, _PLANT2, _POLICY2, [0.3, 0.0], steps=5, alpha=alpha)


def test_a_barrier_without_an_upper_bound_is_refused() -> None:
    with pytest.raises(TypeError, match="QuadraticBarrier"):
        exit_probability(
            lambda x: 1.0 - x[0],  # type: ignore[arg-type]
            _SCALAR_PLANT,
            _SCALAR_POLICY,
            [0.0],
            steps=5,
            alpha=0.5,
        )
    with pytest.raises(ValueError, match="no upper bound"):
        QuadraticBarrier(1.0, [0.0, 0.0], [[1.0, 0.0], [0.0, -0.1]])
    with pytest.raises(ValueError, match="positive"):
        QuadraticBarrier(0.0, [0.0], [[1.0]])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"steps": 0, "alpha": 0.5}, "steps"),
        ({"steps": 5, "alpha": 1.0}, "alpha"),
        ({"steps": 5, "alpha": float("nan")}, "alpha"),
        ({"steps": 5, "alpha": 0.5, "level": -0.1}, "level"),
        ({"steps": 5, "alpha": 0.5, "states": [[0.0]]}, "states reads a callable"),
    ],
)
def test_arguments_outside_the_theorem_are_refused(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        exit_probability(_CORRIDOR, _SCALAR_PLANT, _SCALAR_POLICY, [0.0], **kwargs)


def test_a_callable_policy_needs_states() -> None:
    with pytest.raises(ValueError, match="needs the states"):
        exit_probability(_CORRIDOR, _SCALAR_PLANT, lambda x: -0.6 * x, [0.0], steps=5, alpha=0.5)


def test_certify_safety_neither_reads_nor_returns_the_exit_probability() -> None:
    plan = _plan()

    def barrier(x: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * x[0] + x[1] + 0.4

    before = certify_safety(plan, _MODEL, barrier, 0.1, alpha=5.0, u_max=5.0)
    exit_probability(_ELLIPSE, _PLANT2, _POLICY2, [0.3, 0.0], steps=10, alpha=0.95)
    after = certify_safety(plan, _MODEL, barrier, 0.1, alpha=5.0, u_max=5.0)
    for field in dataclasses.fields(SafetyCertificate):
        assert np.array_equal(
            np.asarray(getattr(before, field.name)), np.asarray(getattr(after, field.name))
        ), field.name
    names = {f.name for f in dataclasses.fields(SafetyCertificate)}
    names |= set(inspect.signature(certify_safety).parameters)
    assert not names & {f.name for f in dataclasses.fields(ExitProbability)} - {"alpha"}
    assert "exit_probability" not in inspect.getsource(chc.plan)
