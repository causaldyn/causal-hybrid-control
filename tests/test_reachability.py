"""HJ backward reachable tube under a set-identified effect (``chc.reachability``)."""

from itertools import pairwise

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from chc.barrier import robust_barrier_margin
from chc.reachability import backward_reachable_tube, barrier_reachability_gap, robust_hamiltonian

# The plant `chc.spine` plans on: two zones, one incentive column that moves drivers between them.
# `h = x1 + 0.4` is a supply floor on the second zone.
ZONE_A = jnp.array([[-0.6, 0.3], [0.3, 0.25]])
ZONE_B = jnp.array([[1.0], [-1.0]])


def _zone_drift(x: Array) -> Array:
    return ZONE_A @ x


def _supply_floor(x: Array) -> Array:
    return x[1] + 0.4


def _double_integrator(x: Array) -> Array:
    return jnp.array([x[1], 0.0])


DI_B = jnp.array([[0.0], [1.0]])


def _position(x: Array) -> Array:
    return x[0]


@pytest.mark.parametrize(
    ("drift_x", "u_max", "radius"),
    [(-0.8, 0.3, 0.0), (-1.0, 0.5, 0.2), (-0.4, 1.0, 1.5), (0.2, 0.5, 0.0), (-0.6, 0.4, 0.9)],
)
def test_tube_matches_the_analytic_sliding_level_set(
    drift_x: float, u_max: float, radius: float
) -> None:
    """With a constant field and ``h = x0`` the gradient never turns, so ``V`` slides rigidly.

    ``grad V = (1, 0)`` for all time, hence ``H = a + U*(1 - Delta)_+`` everywhere and
    ``V(x, T) = x0 + T*min(0, H)``. Any error is the scheme's, which is what this measures.
    """
    horizon = 1.0
    tube = backward_reachable_tube(
        _position,
        lambda _: jnp.array([drift_x, 0.0]),
        jnp.array([[1.0], [0.0]]),
        lower=(-1.0, -1.0),
        upper=(1.0, 1.0),
        resolution=(61, 61),
        horizon=horizon,
        steps=400,
        u_max=u_max,
        radius=radius,
    )
    speed = min(0.0, drift_x + u_max * max(0.0, 1.0 - radius))
    grid_x = tube.axes[0][:, None] * jnp.ones_like(tube.axes[1])[None, :]
    expected = grid_x + horizon * speed
    interior = np.asarray(tube.final)[3:-3, 3:-3] - np.asarray(expected)[3:-3, 3:-3]
    assert np.max(np.abs(interior)) < 0.02


def test_tube_matches_the_double_integrator_braking_parabola() -> None:
    """Relative degree 2: the true invariant set is ``x0 >= x1^2/(2U)`` for approaching states."""
    u_max = 1.0
    tube = backward_reachable_tube(
        _position,
        _double_integrator,
        DI_B,
        lower=(-1.0, -2.0),
        upper=(3.0, 2.0),
        resolution=(121, 121),
        horizon=4.0,
        steps=1800,
        u_max=u_max,
    )
    xs = np.asarray(tube.axes[0])[:, None]
    vs = np.asarray(tube.axes[1])[None, :]
    braking = np.where(vs < 0.0, vs**2 / (2.0 * u_max), 0.0)
    analytic = (xs >= 0.0) & (xs >= braking)
    numeric = np.asarray(tube.final) >= 0.0
    # a band around the parabola is where a first-order scheme is allowed to disagree
    off_boundary = np.abs(xs - braking) > 0.1
    assert np.mean(numeric[off_boundary] == analytic[off_boundary]) > 0.99


def test_hamiltonian_is_exactly_nonincreasing_in_the_identification_radius() -> None:
    """The PDE-level claim: more identification slack cannot buy more guaranteed safety."""
    b_matrix = jnp.array([[1.0], [-1.0]])
    for p in (jnp.array([0.0, 1.0]), jnp.array([1.0, -0.5]), jnp.array([-0.7, -0.7])):
        values = [
            float(robust_hamiltonian(p, jnp.array([0.2, -0.1]), b_matrix, 3.0, radius))
            for radius in (0.0, 0.3, 0.6, 1.0, 1.5, 2.0)
        ]
        assert values == sorted(values, reverse=True)


def test_tube_shrinks_as_the_effect_becomes_less_identified() -> None:
    """The discrete claim, which a first-order scheme only holds to ``O(dx)``.

    The zero level set is located to within a cell, so the sweep is checked against a one-row
    tolerance; the drop it has to clear is several times that.
    """
    fractions = [
        backward_reachable_tube(
            _supply_floor,
            _zone_drift,
            ZONE_B,
            lower=(-1.5, -1.5),
            upper=(1.5, 1.5),
            resolution=(61, 61),
            horizon=2.0,
            steps=1200,
            u_max=3.0,
            radius=radius,
        ).safe_fraction()
        for radius in (0.0, 0.3, 0.6, 1.0, 1.5, 2.0)
    ]
    one_row = 1.0 / 61
    assert all(b <= a + one_row for a, b in pairwise(fractions))
    assert fractions[0] - fractions[-1] > 4 * one_row  # the sweep is not degenerate


def test_tube_is_nonincreasing_in_the_horizon() -> None:
    """``V(., T)`` can only lose points as ``T`` grows: demanding more time is a stronger ask."""
    tube = backward_reachable_tube(
        _position,
        _double_integrator,
        DI_B,
        lower=(-1.0, -2.0),
        upper=(3.0, 2.0),
        resolution=(61, 61),
        horizon=2.0,
        steps=500,
        u_max=1.0,
    )
    assert bool(jnp.all(jnp.diff(tube.values, axis=0) <= 1e-9))
    assert tube.safe_fraction(0) > tube.safe_fraction(-1)


def test_cfl_violation_raises_instead_of_reporting_an_optimistic_set() -> None:
    """The one failure mode a safety tool must not have is a *larger* answer than the truth."""
    with pytest.raises(ValueError, match="CFL number"):
        backward_reachable_tube(
            _supply_floor,
            _zone_drift,
            ZONE_B,
            lower=(-1.5, -1.5),
            upper=(1.5, 1.5),
            resolution=(121, 121),
            horizon=2.0,
            steps=50,
            u_max=3.0,
            radius=0.6,
        )


@pytest.mark.parametrize("radius", [0.0, 0.4, 1.2])
def test_hamiltonian_is_the_barrier_margin_with_a_solved_for_gradient(radius: float) -> None:
    """The two modules run the same algebra; only the gradient's provenance differs."""
    p = jnp.array([0.6, -0.8])
    f = jnp.array([0.2, -0.5])
    b_matrix = jnp.array([[1.0, 0.0], [0.3, 0.9]])
    u_max = 1.7
    grad_norm = float(jnp.linalg.norm(p))
    expected = robust_barrier_margin(
        drift=float(jnp.dot(p, f)),
        channel=float(jnp.linalg.norm(b_matrix.T @ p)),
        radius=radius * grad_norm,
        u_max=u_max,
    )
    assert float(robust_hamiltonian(p, f, b_matrix, u_max, radius)) == pytest.approx(expected)


def test_a_valid_barrier_certificate_makes_the_tube_the_whole_safe_set() -> None:
    """The CBF theorem, executable: condition everywhere on ``{h >= 0}`` implies invariance."""
    gap = barrier_reachability_gap(
        _supply_floor,
        _zone_drift,
        ZONE_B,
        lower=(-1.5, -1.5),
        upper=(1.5, 1.5),
        resolution=(61, 61),
        horizon=2.0,
        steps=1200,
        u_max=3.0,
        radius=0.3,
    )
    assert gap.valid_cbf
    assert gap.reachable_fraction == pytest.approx(gap.safe_fraction, abs=1e-9)
    assert gap.certified_but_unreachable == 0.0
    assert gap.ok


def test_identification_slack_destroys_the_certificate_at_the_zero_action_threshold() -> None:
    """Past ``radius = ||B^T grad h||`` the §40 rule zeroes the action, and the floor gives way."""
    common = {
        "lower": (-1.5, -1.5),
        "upper": (1.5, 1.5),
        "resolution": (61, 61),
        "horizon": 2.0,
        "steps": 1200,
        "u_max": 3.0,
    }
    identified = barrier_reachability_gap(_supply_floor, _zone_drift, ZONE_B, radius=0.6, **common)
    unidentified = barrier_reachability_gap(
        _supply_floor, _zone_drift, ZONE_B, radius=1.2, **common
    )
    assert identified.valid_cbf
    assert not unidentified.valid_cbf
    assert unidentified.reachable_fraction < unidentified.safe_fraction
    assert unidentified.certified_but_unreachable > 0.0
    assert unidentified.ok  # a false antecedent, not a passed test of invariance


def test_pointwise_certification_traps_a_relative_degree_two_barrier() -> None:
    """``B^T grad h == 0`` makes the §40 condition blind to the actuator, so it certifies too much.

    The barrier verdict is then *identical* at every radius while the truth shrinks -- the sharpest
    statement of why :func:`chc.plan.certify_safety`'s per-step prefix is a filter, not a proof.
    """
    common = {
        "lower": (-1.0, -2.0),
        "upper": (3.0, 2.0),
        "resolution": (61, 61),
        "horizon": 2.0,
        "steps": 500,
        "u_max": 1.0,
    }
    tight = barrier_reachability_gap(_position, _double_integrator, DI_B, radius=0.0, **common)
    loose = barrier_reachability_gap(_position, _double_integrator, DI_B, radius=0.5, **common)

    assert (
        not tight.valid_cbf
    )  # x1 < -alpha*x0 is safe-but-uncertified, so the theorem never applies
    assert loose.barrier_fraction == pytest.approx(tight.barrier_fraction)
    assert loose.reachable_fraction < tight.reachable_fraction
    assert loose.certified_but_unreachable > 10 * tight.certified_but_unreachable


def test_interpolation_reproduces_grid_values_and_scores_states_off_grid() -> None:
    tube = backward_reachable_tube(
        _supply_floor,
        _zone_drift,
        ZONE_B,
        lower=(-1.5, -1.5),
        upper=(1.5, 1.5),
        resolution=(61, 61),
        horizon=1.0,
        steps=600,
        u_max=3.0,
    )
    node = jnp.array([tube.axes[0][17], tube.axes[1][40]])
    assert float(tube.interpolate(node)) == pytest.approx(float(tube.final[17, 40]), abs=1e-5)

    mid = 0.5 * (tube.axes[0][17] + tube.axes[0][18])
    between = float(tube.interpolate(jnp.array([mid, tube.axes[1][40]])))
    ends = sorted([float(tube.final[17, 40]), float(tube.final[18, 40])])
    assert ends[0] - 1e-6 <= between <= ends[1] + 1e-6
