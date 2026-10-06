"""chc.uncertainty: certified-Lipschitz rollout-error bound (discrete Gronwall) -- sound + honest.

Ties the shipped LipschitzResidual's certified constant to a machine-checked pessimism radius.
"""

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from jax import Array

from chc.dynamics import DrivenDynamics, Dynamics, LinearDynamics
from chc.integrate import rollout
from chc.uncertainty import (
    certified_horizon,
    contractive_rollout_bound,
    contractive_rollout_certificate,
    linear_rollout_bound,
    lipschitz_rollout_bound,
    lipschitz_rollout_certificate,
    time_varying_rollout_bound,
)


def test_certificate_deviation_stays_under_the_bound() -> None:
    cert = lipschitz_rollout_certificate(seed=0)
    assert cert.ok  # measured rollout deviation <= the certified Gronwall bound
    assert cert.measured_deviation <= cert.certified_bound + 1e-9
    assert cert.measured_deviation > 0.0  # a non-trivial deviation was actually produced


def test_bound_holds_across_seeds() -> None:
    for seed in range(6):
        cert = lipschitz_rollout_certificate(seed=seed)
        assert (
            cert.measured_deviation <= cert.certified_bound + 1e-9
        )  # the guarantee is not seed-luck


def test_bound_is_not_vacuous_on_a_short_horizon() -> None:
    cert = lipschitz_rollout_certificate(seed=1, horizon=8, dt=0.05)
    # short horizon / bounded L: the certified radius is within an order of magnitude of the truth
    assert cert.measured_deviation >= 0.2 * cert.certified_bound


def test_contractive_certificate_confirms_negative_log_norm_and_flat_radius() -> None:
    cert = contractive_rollout_certificate(seed=0)
    assert cert.ok
    assert cert.contraction_rate > 0.0  # certified |mu| > 0
    assert cert.empirical_one_sided <= -cert.contraction_rate + 1e-6  # genuinely contracting
    assert cert.measured_deviation <= cert.bounded_radius + 1e-6  # under the flat ceiling
    assert cert.bounded_radius < cert.lipschitz_blowup  # contraction beats the e^{L*T} envelope


def test_contractive_radius_stays_capped_while_lipschitz_explodes() -> None:
    # the explicit-Euler ceiling eps*dt/(1-q) is horizon-INDEPENDENT, unlike norm-Lipschitz e^{L*T}
    short = contractive_rollout_bound(1.0, 2.0, 0.1, 0.05, 800)  # rate=1, L=2, dt<2c/L^2=0.5
    long = contractive_rollout_bound(1.0, 2.0, 0.1, 0.05, 5000)
    lipschitz = lipschitz_rollout_bound(2.0, 0.1, 0.05, 5000)
    assert long == pytest.approx(short, rel=1e-6)  # both saturated: the cap is horizon-independent
    assert long < 1e-6 * lipschitz  # while the norm-Lipschitz bound blows up exponentially


def test_contractive_bound_returns_inf_when_step_too_large() -> None:
    # dt >= 2c/L^2 breaks the explicit-Euler contraction (q >= 1); the bound is invalid
    assert contractive_rollout_bound(1.0, 2.0, 0.1, 5.0, 8) == float("inf")  # dt=5 >> 2c/L^2


def test_vanishing_lipschitz_gives_the_linear_envelope() -> None:
    # L -> 0: the Gronwall closed form degrades to the linear eps*dt*H (no exponential blow-up)
    assert lipschitz_rollout_bound(0.0, 0.1, 0.05, 8) == pytest.approx(0.1 * 0.05 * 8)


def test_bound_matches_the_gronwall_closed_form() -> None:
    lipschitz, model_error, dt, horizon = 1.5, 0.2, 0.05, 10
    expected = model_error * ((1.0 + lipschitz * dt) ** horizon - 1.0) / lipschitz
    assert lipschitz_rollout_bound(lipschitz, model_error, dt, horizon) == pytest.approx(expected)


def test_bound_is_monotone_in_model_error_and_horizon() -> None:
    small = lipschitz_rollout_bound(1.0, 0.1, 0.05, 8)
    more_error = lipschitz_rollout_bound(1.0, 0.2, 0.05, 8)
    longer = lipschitz_rollout_bound(1.0, 0.1, 0.05, 16)
    assert more_error > small  # larger per-step error -> larger certified radius
    assert longer > small  # longer horizon -> larger certified radius


def test_bound_grows_exponentially_with_lipschitz_horizon_product() -> None:
    # HONEST SCOPE: the bound is exp(L*T); it must blow up for large L*T (documented, not hidden)
    short = lipschitz_rollout_bound(2.0, 0.1, 0.05, 8)  # L*T = 2*0.4 = 0.8
    long_horizon = lipschitz_rollout_bound(2.0, 0.1, 0.05, 200)  # L*T = 2*10 = 20
    assert long_horizon > 100.0 * short  # exponential premium at large L*T
    assert np.isfinite(long_horizon)


def test_time_varying_certificate_is_tighter_than_constant_and_has_a_cutoff() -> None:
    from chc.uncertainty import time_varying_rollout_certificate

    cert = time_varying_rollout_certificate(seed=0)
    assert cert.ok
    assert cert.varying_final <= cert.constant_final + 1e-9  # per-step L is never looser than max-L
    assert cert.certified_until_step >= 0  # a valid honest planning horizon
    assert cert.safe_until_step >= 1  # some prefix of the plan is robustly feasible


def test_certified_horizon_shrinks_as_tolerance_tightens() -> None:
    from chc.uncertainty import certified_horizon

    lipschitz = [1.0] * 40
    error = [0.1] * 40
    loose = certified_horizon(lipschitz, error, 0.05, tolerance=1.0)
    tight = certified_horizon(lipschitz, error, 0.05, tolerance=0.1)
    assert tight <= loose  # a stricter tolerance certifies fewer steps


def test_constraint_tightening_flags_the_true_trajectory_safe() -> None:
    from chc.uncertainty import constraint_margin

    # nominal margin -0.3 to g<=0, L_g=1, growing error radius: safe until the tube eats the margin
    radii = np.array([0.0, 0.1, 0.2, 0.35, 0.5])
    margin = constraint_margin(-0.3 * np.ones(5), 1.0, radii)
    assert bool(margin[0] <= 0.0)  # at k=0 the true trajectory is certified feasible
    assert bool(margin[-1] > 0.0)  # once L_g*e_k exceeds the nominal margin, feasibility is lost


def test_closed_loop_radius_exceeds_open_loop_when_replanning() -> None:
    from chc.uncertainty import closed_loop_rollout_bound

    open_loop = lipschitz_rollout_bound(1.0, 0.1, 0.05, 10)
    closed = closed_loop_rollout_bound(1.0, 0.5, 2.0, 0.1, 0.05, 10)  # L_pi=2 policy sensitivity
    assert closed > open_loop  # re-planning feeds state error through the policy -> larger tube


class _Off(eqx.Module):
    """A field off by exactly ``eps`` wherever it is read, in a direction that turns with the time
    and the state, so each RK4 stage reads an error of its own."""

    field: Dynamics
    eps: float
    first: Array
    second: Array

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        turn = self.first * jnp.cos(3.0 * t) + self.second * jnp.sin(jnp.sum(x))
        return self.field(t, x, u) + self.eps * turn / jnp.linalg.norm(turn)


class _Tanh(eqx.Module):
    """``A x + W2 tanh(W1 x) + B u``, ``||A|| + ||W2|| ||W1||``-Lipschitz in the state."""

    a: Array
    w1: Array
    w2: Array
    b: Array

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return self.a @ x + self.w2 @ jnp.tanh(self.w1 @ x) + self.b @ u


def _gaps(field: Dynamics, off: Dynamics, x0: Array, actions: Array, dt: float) -> np.ndarray:
    return np.linalg.norm(
        np.asarray(rollout(off, x0, actions, dt) - rollout(field, x0, actions, dt)), axis=1
    )


def test_the_tube_follows_rk4_where_eulers_recursion_reads_low() -> None:
    """``x' = x`` against ``x' = x + 0.1`` at ``dt = 1``: the RK4 rollouts every plan makes part by
    0.1708, 0.6335 and 1.8866, which the RK4 tube reads exactly; Euler's recursion read 0.1, 0.3
    and 0.7, so a tolerance of 0.12 was certified for a step the gap had already passed."""
    field = LinearDynamics(jnp.array([[1.0]]), jnp.zeros((1, 1)))
    off = DrivenDynamics(field, jnp.array([[0.1]]), jnp.ones((4, 1)), 1.0)
    gaps = _gaps(field, off, jnp.zeros(1), jnp.zeros((3, 1)), 1.0)
    tube = np.asarray(time_varying_rollout_bound([1.0] * 3, [0.1] * 3, 1.0))
    np.testing.assert_allclose(tube, [0.0, 41 / 240, 3649 / 5760, 260801 / 138240], rtol=1e-15)
    np.testing.assert_allclose(gaps, tube, rtol=1e-14, atol=0.0)
    euler = np.asarray(time_varying_rollout_bound([1.0] * 3, [0.1] * 3, 1.0, integrator="euler"))
    np.testing.assert_allclose(euler, [0.0, 0.1, 0.3, 0.7], rtol=1e-15)
    assert certified_horizon([1.0] * 3, [0.1] * 3, 1.0, 0.12) == 0
    assert certified_horizon([1.0] * 3, [0.1] * 3, 1.0, 0.12, integrator="euler") == 1


def test_a_radius_at_the_tolerance_is_within_it() -> None:
    tube = time_varying_rollout_bound([1.0] * 3, [0.1] * 3, 1.0)
    assert certified_horizon([1.0] * 3, [0.1] * 3, 1.0, float(tube[2])) == 2


@pytest.mark.parametrize("seed", range(12))
def test_the_rk4_tube_holds_every_gap_of_a_lipschitz_field(seed: int) -> None:
    """Two RK4 rollouts of a random ``tanh`` field, one off by ``eps`` at every stage, stay inside
    the tube built on the field's Lipschitz bound, at every step of the horizon."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 5))
    a = rng.normal(size=(n, n)) * rng.uniform(0.1, 1.5)
    w1, w2 = rng.normal(size=(8, n)), rng.normal(size=(n, 8))
    field = _Tanh(
        jnp.asarray(a), jnp.asarray(w1), jnp.asarray(w2), jnp.asarray(rng.normal(size=(n, 2)))
    )
    lipschitz = float(np.linalg.norm(a, 2) + np.linalg.norm(w2, 2) * np.linalg.norm(w1, 2))
    eps, dt, steps = rng.uniform(0.01, 0.5), rng.uniform(0.01, 0.5), int(rng.integers(3, 30))
    off = _Off(field, eps, jnp.asarray(rng.normal(size=n)), jnp.asarray(rng.normal(size=n)))
    actions = jnp.asarray(rng.normal(size=(steps, 2)))
    gaps = _gaps(field, off, jnp.asarray(rng.normal(size=n)), actions, dt)
    tube = np.asarray(time_varying_rollout_bound([lipschitz] * steps, [eps] * steps, dt))
    assert np.all(gaps <= tube * (1.0 + 1e-12))


@pytest.mark.parametrize("seed", range(12))
def test_the_matrix_tube_holds_every_gap_and_shrinks_where_the_state_matrix_contracts(
    seed: int,
) -> None:
    """On a field affine in the state the tube read off RK4's propagators holds every gap of two
    rollouts, one off by ``eps`` at every stage, and on a contracting matrix it stays below the
    tube of the matrix's norm, which can only grow."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 5))
    a = rng.normal(size=(n, n))
    a -= (np.max(np.linalg.eigvalsh((a + a.T) / 2)) + rng.uniform(0.1, 2.0)) * np.eye(n)
    field = LinearDynamics(jnp.asarray(a), jnp.asarray(rng.normal(size=(n, 2))))
    eps, dt, steps = rng.uniform(0.01, 0.5), rng.uniform(0.01, 0.3), int(rng.integers(3, 40))
    off = _Off(field, eps, jnp.asarray(rng.normal(size=n)), jnp.asarray(rng.normal(size=n)))
    actions = jnp.asarray(rng.normal(size=(steps, 2)))
    gaps = _gaps(field, off, jnp.asarray(rng.normal(size=n)), actions, dt)
    tube = np.asarray(linear_rollout_bound(a, [eps] * steps, dt))
    assert np.all(gaps <= tube * (1.0 + 1e-12))
    norm = np.asarray(time_varying_rollout_bound([np.linalg.norm(a, 2)] * steps, [eps] * steps, dt))
    assert np.all(tube[1:] < norm[1:])


def test_the_matrix_tube_of_a_scalar_rate_is_the_rk4_tube() -> None:
    """On ``A = L >= 0`` every weight the stages put on the error is nonnegative, so the matrix
    tube and the Lipschitz one are the same recursion."""
    scalar = np.asarray(linear_rollout_bound([[0.7]], [0.2] * 25, 0.3))
    np.testing.assert_allclose(
        scalar, np.asarray(time_varying_rollout_bound([0.7] * 25, [0.2] * 25, 0.3)), rtol=1e-13
    )


@pytest.mark.parametrize("rate", [-0.5, float("nan")])
@pytest.mark.parametrize("integrator", ["rk4", "euler"])
def test_a_rate_no_norm_takes_is_refused(rate: float, integrator: str) -> None:
    """A negative log-norm in either recursion turned the radii negative, and every step then
    passed the tolerance: neither tube takes one."""
    with pytest.raises(ValueError, match=r"is not a norm-Lipschitz bound"):
        time_varying_rollout_bound([rate] * 4, [0.1] * 4, 0.5, integrator=integrator)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"is not a norm-Lipschitz bound"):
        certified_horizon([rate] * 4, [0.1] * 4, 0.5, 1.0, integrator=integrator)  # type: ignore[arg-type]


def test_what_is_neither_an_integrator_nor_a_square_matrix_is_refused() -> None:
    with pytest.raises(ValueError, match=r"neither 'rk4' nor 'euler'"):
        time_varying_rollout_bound([1.0], [0.1], 0.5, integrator="heun")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"is not square"):
        linear_rollout_bound(np.ones((2, 3)), [0.1], 0.5)
    with pytest.raises(ValueError, match=r"not finite"):
        linear_rollout_bound([[np.inf]], [0.1], 0.5)


@pytest.mark.parametrize(
    ("rate", "budget", "dt", "match"),
    [
        (math.inf, 0.1, 1.0, r"lipschitz=inf is not a norm-Lipschitz bound"),
        (1.0, math.nan, 1.0, r"model_error=nan is not a per-step error budget"),
        (1.0, -0.1, 1.0, r"model_error=-0\.1 is not a per-step error budget"),
        (1.0, math.inf, 1.0, r"model_error=inf is not a per-step error budget"),
        (1.0, 0.1, 0.0, r"dt=0\.0 is not a step"),
        (1.0, 0.1, -1.0, r"dt=-1\.0 is not a step"),
        (1.0, 0.1, math.nan, r"dt=nan is not a step"),
        (1.0, 0.1, math.inf, r"dt=inf is not a step"),
    ],
)
@pytest.mark.parametrize("integrator", ["rk4", "euler"])
def test_what_no_tube_can_be_read_on_is_refused(
    rate: float, budget: float, dt: float, match: str, integrator: str
) -> None:
    """An infinite rate read ``inf * 0`` at the first step, a nan budget a nan radius, and a
    negative budget negative radii: each tube certified all 3 steps at a tolerance of 0.12."""
    with pytest.raises(ValueError, match=match):
        time_varying_rollout_bound([rate] * 3, [budget] * 3, dt, integrator=integrator)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=match):
        certified_horizon([rate] * 3, [budget] * 3, dt, 0.12, integrator=integrator)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("budget", "dt", "match"),
    [
        (math.nan, 0.5, r"model_error=nan"),
        (-0.1, 0.5, r"model_error=-0\.1"),
        (math.inf, 0.5, r"model_error=inf"),
        (0.1, 0.0, r"dt=0\.0"),
        (0.1, math.nan, r"dt=nan"),
    ],
)
def test_the_matrix_tube_refuses_what_no_tube_can_be_read_on(
    budget: float, dt: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        linear_rollout_bound([[-0.5]], [budget] * 3, dt)


@pytest.mark.parametrize("tolerance", [math.nan, -0.1])
def test_a_tolerance_that_is_no_radius_is_refused(tolerance: float) -> None:
    """A nan tolerance compared false with every radius and certified all 3 steps."""
    with pytest.raises(ValueError, match=r"is not a radius"):
        certified_horizon([1.0] * 3, [0.1] * 3, 1.0, tolerance)


def test_a_radius_that_overflows_is_within_no_tolerance() -> None:
    """``inf`` keeps every finite radius and no other. A budget of 1e308 over a step of 10
    overflows to ``inf`` at the first step, and with a rate of 0 the next reads ``0 * inf``, a nan:
    the tube ``[0, inf, nan, nan]`` was certified for all 3 steps under an infinite tolerance."""
    assert certified_horizon([1.0] * 3, [0.1] * 3, 1.0, math.inf) == 3
    tube = np.asarray(time_varying_rollout_bound([0.0] * 3, [1e308] * 3, 10.0))
    assert np.isinf(tube[1])
    assert np.isnan(tube[2])
    assert certified_horizon([0.0] * 3, [1e308] * 3, 10.0, math.inf) == 0
    assert certified_horizon([0.0] * 3, [1e307] * 3, 1.0, math.inf) == 3


@given(
    rates=st.lists(st.floats(allow_nan=True, allow_infinity=True), min_size=1, max_size=6),
    budget=st.floats(allow_nan=True, allow_infinity=True),
    dt=st.floats(allow_nan=True, allow_infinity=True),
    tolerance=st.floats(allow_nan=True, allow_infinity=True),
    integrator=st.sampled_from(["rk4", "euler"]),
)
def test_a_certified_step_is_a_finite_radius_within_the_tolerance(
    rates: list[float], budget: float, dt: float, tolerance: float, integrator: str
) -> None:
    """Whatever the inputs, either they are refused or every step counted has a radius that is a
    finite number within the tolerance, and the first step not counted has not."""
    budgets = [budget] * len(rates)
    try:
        steps = certified_horizon(rates, budgets, dt, tolerance, integrator=integrator)  # type: ignore[arg-type]
    except ValueError:
        return
    tube = np.asarray(time_varying_rollout_bound(rates, budgets, dt, integrator=integrator))  # type: ignore[arg-type]
    kept = tube[1 : steps + 1]
    assert np.all(np.isfinite(kept))
    assert np.all(kept <= tolerance)
    if steps < len(rates):
        assert not (np.isfinite(tube[steps + 1]) and tube[steps + 1] <= tolerance)
