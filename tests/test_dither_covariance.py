"""dither_covariance: the dither covariance that minimises tr(C V) + tr(Q (P0 + V)^-1) over V >= 0.

Its numbers are validation/dither_covariance.mac's.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from chc.experiment import ChannelPrior, ZoneExperiment, _water_filling, dither_covariance

C2 = np.array([[2.0, 1.0], [1.0, 1.0]])
Q2 = np.array([[1.0, -0.5], [-0.5, 2.0]])


def _value(c: np.ndarray, q: np.ndarray, p0: np.ndarray, v: np.ndarray) -> float:
    return float(np.trace(c @ v) + np.trace(q @ np.linalg.inv(p0 + v)))


def test_the_two_lever_example() -> None:
    design = dither_covariance(C2, Q2)
    assert design.method == "closed_form"
    assert design.gap == 0.0
    assert design.value == pytest.approx(2.0 * math.sqrt(3.0 + math.sqrt(7.0)), rel=1e-13, abs=0.0)
    assert design.diagonal_value == pytest.approx(4.0 * math.sqrt(2.0), rel=1e-9, abs=0.0)
    assert design.white_value == pytest.approx(6.0, rel=1e-9, abs=0.0)
    assert 1.0 - design.value / design.diagonal_value == pytest.approx(0.15993, abs=5e-6)
    assert 1.0 - design.value / design.white_value == pytest.approx(0.20797, abs=5e-6)
    m = design.covariance
    np.testing.assert_allclose(m @ C2 @ m, Q2, rtol=0.0, atol=1e-13)


def test_a_prior_below_the_optimum_is_subtracted() -> None:
    p0 = np.array([[0.2, 0.05], [0.05, 0.1]])
    design = dither_covariance(C2, Q2, p0)
    assert design.method == "closed_form"
    m = design.covariance + p0
    np.testing.assert_allclose(m @ C2 @ m, Q2, rtol=0.0, atol=1e-12)
    assert design.value == pytest.approx(
        2.0 * math.sqrt(3.0 + math.sqrt(7.0)) - float(np.trace(C2 @ p0)), rel=1e-12, abs=0.0
    )
    assert design.value == pytest.approx(4.15216, abs=5e-6)


def test_a_binding_prior_matches_the_forty_digit_solution() -> None:
    # validation/dither_covariance.mac STEP 7: P0 = diag(1, 1/20) holds more than the closed form
    # asks along a direction, so V >= 0 binds and the optimum is rank one
    design = dither_covariance(C2, Q2, np.diag([1.0, 0.05]))
    assert design.method == "numeric"
    assert design.value == pytest.approx(2.91763610926513659848, rel=1e-8, abs=0.0)
    assert 0.0 <= design.gap <= 1e-7
    eigenvalues = np.linalg.eigvalsh(design.covariance)
    assert eigenvalues[0] == pytest.approx(0.0, abs=1e-5)
    assert eigenvalues[1] > 2.0


def test_a_prior_with_a_zero_eigenvalue_gives_the_white_dither_without_a_warning() -> None:
    # validation/dither_covariance.mac STEP 8b: a prior that holds one lever only; the best white
    # dither is the root of 2 - 1/(1 + v)^2 - 1/v^2, above the search's lower bound sqrt(1/2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        design = dither_covariance(np.eye(2), np.eye(2), np.diag([1.0, 0.0]))
    assert design.white == pytest.approx(0.77122987841870627168, rel=1e-13, abs=0.0)
    assert design.white_value == pytest.approx(3.40366947504161167615, rel=1e-13, abs=0.0)


def test_one_lever_s_white_dither_is_the_search_s_lower_bound() -> None:
    # validation/dither_covariance.mac STEP 8c: on one lever the root is the bound
    # sqrt(Q / C) - P0 itself, where the derivative is zero up to rounding
    design = dither_covariance([[1.0]], [[4.0]], [[1.0]])
    assert design.white == pytest.approx(1.0, rel=1e-15, abs=0.0)
    assert design.white == pytest.approx(design.diagonal[0], rel=1e-15, abs=0.0)


@pytest.mark.parametrize("seed", range(4))
def test_the_diagonal_case_is_water_filling(seed: int) -> None:
    rng = np.random.default_rng(seed)
    k = 4
    weight = np.diag(rng.uniform(0.5, 3.0, k))
    variance = rng.uniform(0.05, 2.0, k)
    variance[0] = 1e-3  # this lever's prior already holds more than the level: clamped at zero
    experiment = ZoneExperiment(
        unit_cost=rng.uniform(0.5, 2.0, k), noise_sd=rng.uniform(0.5, 2.0, k), probe=np.ones(k)
    )
    prior = ChannelPrior(np.zeros(k), np.diag(variance))
    uses = 50.0
    units = _water_filling(weight, prior, experiment, budget=1e12, uses=uses)
    price = experiment.unit_cost / experiment.information
    design = dither_covariance(np.diag(price), uses * weight / 2.0, np.diag(1.0 / variance))
    assert units[0] == 0.0
    assert design.method == "numeric"
    np.testing.assert_allclose(
        np.diag(design.covariance), experiment.information * units, rtol=0.0, atol=1e-5
    )
    np.testing.assert_allclose(design.diagonal, experiment.information * units, rtol=0.0, atol=1e-5)
    off = design.covariance - np.diag(np.diag(design.covariance))
    assert np.max(np.abs(off)) <= 1e-5


def test_the_diagonal_case_without_a_clamp_is_the_level() -> None:
    price, w = np.array([3.0, 2.0 / 7.0]), np.array([5.0, 11.0 / 3.0])
    design = dither_covariance(np.diag(price), np.diag(w))
    assert design.method == "closed_form"
    np.testing.assert_allclose(np.diag(design.covariance), np.sqrt(w / price), rtol=1e-14, atol=0.0)
    np.testing.assert_array_equal(design.covariance - np.diag(np.diag(design.covariance)), 0.0)


@pytest.mark.parametrize("seed", range(6))
def test_no_dither_nearby_does_better(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    k = 3
    a, b, r = (rng.standard_normal((k, k)) for _ in range(3))
    c, q = a @ a.T + 0.3 * np.eye(k), b @ b.T + 0.3 * np.eye(k)
    p0 = r @ r.T * rng.uniform(0.0, 2.0)
    design = dither_covariance(c, q, p0)
    v = design.covariance
    assert np.linalg.eigvalsh(v)[0] >= -1e-9
    assert design.value == pytest.approx(_value(c, q, p0, v), rel=1e-10, abs=0.0)
    assert design.value <= design.diagonal_value + 1e-9
    assert design.value <= design.white_value + 1e-9
    for _ in range(200):
        step = rng.standard_normal((k, k)) * 0.05
        candidate = v + step @ step.T + 0.01 * (step + step.T)
        values, vectors = np.linalg.eigh(candidate)
        candidate = (vectors * np.clip(values, 0.0, None)) @ vectors.T
        assert _value(c, q, p0, candidate) >= design.value - design.gap - 1e-10


@pytest.mark.parametrize(
    ("cost", "weight", "precision", "message"),
    [
        (np.ones((2, 3)), np.eye(2), None, "square"),
        (np.diag([1.0, 0.0]), np.eye(2), None, "cost"),
        (C2, np.diag([1.0, -1.0]), None, "weight"),
        (C2, Q2, np.diag([1.0, -0.1]), "precision"),
        (C2, np.eye(3), None, "weight must have shape"),
        (np.array([[1.0, np.nan], [np.nan, 1.0]]), np.eye(2), None, "non-finite"),
    ],
)
def test_refusals(cost, weight, precision, message) -> None:
    with pytest.raises(ValueError, match=message):
        dither_covariance(cost, weight, precision)
