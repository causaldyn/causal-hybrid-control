"""chc.regret / chc.network_causal: folds on BOTH axes of a delayed-network panel (§59).

Result 51 showed the delayed-network covariance is separable only at ``lag = 0``; Result 52 designed
folds on the space axis and said so. These pin the two-axis structure: the covariance is the
simulator's, its Kronecker rank follows the commutation law, and the price of holding a fold
constant in time is a function of the delay -- small at ``lag = 1`` and not small beyond it.

``validation/space_time_folds.m`` re-derives the rank numbers in Octave from its own shell
construction, so the law is not being checked against the implementation that produced it.
"""

import itertools

import numpy as np
import pytest

from chc.network_causal import (
    ar1_innovations,
    cycle_shells,
    graph_shells,
    kronecker_spectrum,
    panel_covariance,
    propagate_shells,
)
from chc.regret import optimal_fold_partition, space_time_fold_certificate

GAMMAS = (1.0, 0.7, 0.4)


def _rank(sigma: np.ndarray, units: int, times: int) -> int:
    spectrum = kronecker_spectrum(sigma, units, times)
    return int((spectrum > 1e-9 * spectrum[0]).sum())


def test_the_panel_covariance_is_the_simulators() -> None:
    m, p, lag, phi = 6, 4, 1, 0.7
    shells = cycle_shells(m, len(GAMMAS) - 1)
    rng = np.random.default_rng(0)
    innovations = ar1_innovations(rng, (120_000, m), 40 + p + lag * (len(GAMMAS) - 1), phi)
    drawn = propagate_shells(innovations, shells, GAMMAS, lag, p).reshape(-1, m * p)
    predicted = panel_covariance(shells, GAMMAS, phi, lag, p)
    assert np.abs(np.cov(drawn, rowvar=False) - predicted).max() < 0.05 * np.abs(predicted).max()


@pytest.mark.parametrize("dmax", [1, 2, 3, 4])
def test_commuting_shells_give_the_smaller_kronecker_rank(dmax: int) -> None:
    m, p = 8, dmax + 2  # long enough that every shift lands inside the window
    weights = np.array([0.9**k for k in range(dmax + 1)])
    path = np.diag(np.ones(m - 1), 1) + np.diag(np.ones(m - 1), -1)
    cycle_rank = _rank(panel_covariance(cycle_shells(m, dmax), weights, 0.6, 1, p), m, p)
    path_rank = _rank(panel_covariance(graph_shells(path, dmax), weights, 0.6, 1, p), m, p)
    assert cycle_rank == dmax + 1
    assert path_rank == max(2 * dmax, 2)
    assert path_rank < 2 * dmax + 1  # the q=0 and q=dmax factors are symmetric on every graph


def test_zero_delay_is_exactly_separability() -> None:
    m, p = 8, 6
    shells = cycle_shells(m, 4)
    weights = np.array([0.9**k for k in range(5)])
    assert _rank(panel_covariance(shells, weights, 0.6, 0, p), m, p) == 1
    assert _rank(panel_covariance(shells, weights, 0.6, 1, p), m, p) == 5


def test_a_panel_too_short_cannot_see_the_shift() -> None:
    # once lag*q passes p-1 every remaining temporal factor is a multiple of the same matrix
    m, dmax = 8, 4
    weights = np.array([0.9**k for k in range(dmax + 1)])
    shells = cycle_shells(m, dmax)
    resolved = _rank(panel_covariance(shells, weights, 0.6, 1, dmax + 2), m, dmax + 2)
    saturated = _rank(panel_covariance(shells, weights, 0.6, 1, dmax), m, dmax)
    assert saturated < resolved == dmax + 1


def test_time_axis_none_leaves_the_one_axis_design_untouched() -> None:
    shells = cycle_shells(8, 2)
    baseline = optimal_fold_partition(shells, GAMMAS, 0.6, lag=2)
    assert baseline.fold.shape == (8,)
    assert baseline.exhaustive
    panel = optimal_fold_partition(shells, GAMMAS, 0.6, lag=2, time_axis=4, restarts=20)
    assert panel.fold.shape == (32,)


def test_the_wrong_weight_and_the_wrong_axis_cross_over() -> None:
    # two different mistakes: Result 52's cross-sectional WEIGHT, and a frozen time AXIS
    m, p = 8, 4
    shells = cycle_shells(m, 2)
    slice_penalty, axis_price = [], []
    for lag in (1, 4):
        sigma = panel_covariance(shells, GAMMAS, 0.6, lag, p)

        def scored(fold: np.ndarray, sigma: np.ndarray = sigma) -> float:
            return float(sigma[fold[:, None] == fold[None, :]].sum())

        cross_section = np.repeat(optimal_fold_partition(shells, GAMMAS, 0.6, lag=lag).fold, p)
        best_constant = min(
            scored(np.repeat(np.asarray(unit), p))
            for unit in itertools.product((0, 1), repeat=m)
            if sum(unit) == m // 2
        )
        free = optimal_fold_partition(
            shells, GAMMAS, 0.6, lag=lag, time_axis=p, restarts=25
        ).objective
        slice_penalty.append((scored(cross_section) - best_constant) / best_constant)
        axis_price.append((best_constant - free) / best_constant)
    assert slice_penalty[0] > axis_price[0]  # short delay: the weight is the thing to fix
    assert axis_price[1] > slice_penalty[1]  # long delay: the axis is


def test_the_panel_weight_is_not_the_single_slice_weight() -> None:
    # validation/space_time_folds.mac (6): w_1/w_0 vs phi**lag, equal only at the endpoints
    phi, p = 0.6, 3
    gaps = np.arange(p)[:, None] - np.arange(p)[None, :]
    w0 = float((phi ** np.abs(gaps)).sum())
    w1 = float((phi ** np.abs(gaps - 1)).sum())
    assert w0 == pytest.approx(2 * phi**2 + 4 * phi + 3)
    assert w1 == pytest.approx(phi**3 + 2 * phi**2 + 4 * phi + 2)
    assert w1 / w0 > phi


def test_the_certificate_passes_every_gate() -> None:
    certificate = space_time_fold_certificate()
    assert certificate.ok
    assert certificate.commuting_rank == tuple(d + 1 for d in certificate.dmax_grid)
    assert certificate.generic_rank == tuple(max(2 * d, 2) for d in certificate.dmax_grid)
    assert certificate.separable_rank == 1
    assert certificate.exhaustive_ratio == pytest.approx(1.0)
    assert np.all(np.diff(certificate.price_of_one_axis) > 0.0)
