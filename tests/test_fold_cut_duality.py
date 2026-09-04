"""Result 61 -- the fold design is a MAX-bisection, and the exact route reaches it at any m."""

import itertools
from math import comb

import numpy as np
import pytest

from chc.network_causal import cycle_shells
from chc.regret import (
    _circulant_band,
    _exact_banded_bisection,
    _fold_weight_matrix,
    fold_exactness_certificate,
    optimal_fold_partition,
)

GAMMAS = (1.0, 0.7, 0.4)


def _designs(m: int):
    for chosen in itertools.combinations(range(1, m), m // 2 - 1):
        fold = np.ones(m, dtype=np.int_)
        fold[0] = 0
        fold[list(chosen)] = 0
        yield fold


@pytest.mark.parametrize("x", [0.01, 0.3, 0.9])
def test_the_same_fold_mass_is_the_complement_of_the_cut(x: float) -> None:
    m = 10
    q = _fold_weight_matrix(cycle_shells(m, 2), GAMMAS, x)
    total = float(np.ones(m) @ q @ np.ones(m))
    for fold in _designs(m):
        same = fold[:, None] == fold[None, :]
        internal = float(q[same].sum())
        cut = float(q[~same].sum()) / 2.0
        assert internal == pytest.approx(total - 2.0 * cut, abs=1e-12)


@pytest.mark.parametrize("x", [0.01, 0.3, 0.9])
def test_minimising_the_mass_maximises_the_cut_not_the_other_way(x: float) -> None:
    m = 10
    q = _fold_weight_matrix(cycle_shells(m, 2), GAMMAS, x)
    values = []
    for fold in _designs(m):
        same = fold[:, None] == fold[None, :]
        values.append((float(q[same].sum()), float(q[~same].sum()) / 2.0))
    best_cut = min(values)[1]
    assert best_cut == pytest.approx(max(v[1] for v in values), abs=1e-12)
    assert best_cut > min(v[1] for v in values) + 1e-9  # the min-cut design is a different one


def test_the_optimum_keeps_half_the_neighbours_together() -> None:
    """The 2-walk max-cut is NOT the adjacency max-cut: parity is twice as separating."""
    m = 12
    design = optimal_fold_partition(cycle_shells(m, 2), GAMMAS, 0.3, lag=1)
    adjacency = cycle_shells(m, 1)[1]
    cross = design.fold[:, None] != design.fold[None, :]
    parity = np.arange(m) % 2
    parity_cross = parity[:, None] != parity[None, :]
    assert float(adjacency[cross].sum()) / 2.0 == pytest.approx(m / 2)
    assert float(adjacency[parity_cross].sum()) / 2.0 == pytest.approx(m)


@pytest.mark.parametrize("m", [10, 12, 14, 16, 18])
@pytest.mark.parametrize("x", [0.01, 0.3, 0.9])
def test_the_dynamic_program_reproduces_enumeration(m: int, x: float) -> None:
    shells = cycle_shells(m, 2)
    enumerated = optimal_fold_partition(shells, GAMMAS, x, lag=1, exhaustive_limit=10**6)
    dp = optimal_fold_partition(shells, GAMMAS, x, lag=1, exhaustive_limit=0)
    assert enumerated.route == "enumeration"
    assert dp.route == "banded-dp"
    assert dp.exhaustive
    assert dp.objective == pytest.approx(enumerated.objective, abs=1e-9)
    q = _fold_weight_matrix(shells, GAMMAS, x)
    same = dp.fold[:, None] == dp.fold[None, :]
    assert float(q[same].sum()) == pytest.approx(dp.objective, abs=1e-9)
    assert int(dp.fold.sum()) == m // 2


def test_the_dynamic_program_reaches_sizes_enumeration_cannot() -> None:
    m = 120
    design = optimal_fold_partition(cycle_shells(m, 2), GAMMAS, 0.3, lag=1)
    assert design.route == "banded-dp"
    assert comb(m - 1, m // 2 - 1) > 10**30  # what the enumeration would have had to walk
    assert design.lower_bound <= design.objective + 1e-9


def test_the_band_is_set_by_the_truncation_not_by_m() -> None:
    for m in (24, 60, 120):
        for dmax in (1, 2, 3):
            profile = _circulant_band(
                _fold_weight_matrix(cycle_shells(m, dmax), (1.0,) * (dmax + 1), 0.3)
            )
            assert profile is not None
            assert profile[1] == 2 * dmax


def test_a_non_circulant_weight_falls_back_to_the_search() -> None:
    rng = np.random.default_rng(0)
    root = rng.normal(size=(30, 30))
    profile = _circulant_band(root @ root.T)
    assert profile is None


def test_the_dynamic_program_refuses_the_antipodal_size() -> None:
    with pytest.raises(ValueError, match="m > 2\\*band"):
        _exact_banded_bisection(np.ones(5), 4, 8)


@pytest.mark.parametrize("shift", [0.37, -1.5])
def test_a_uniform_off_diagonal_shift_moves_every_design_alike(shift: float) -> None:
    m = 10
    q = _fold_weight_matrix(cycle_shells(m, 2), GAMMAS, 0.3)
    moved = q + shift * (np.ones((m, m)) - np.eye(m))
    gaps = []
    for fold in _designs(m):
        cross = fold[:, None] != fold[None, :]
        gaps.append(float(moved[cross].sum()) / 2.0 - float(q[cross].sum()) / 2.0)
    assert np.ptp(gaps) < 1e-9
    assert gaps[0] == pytest.approx(shift * m * m / 4.0, abs=1e-9)


def test_a_diagonal_shift_does_not_reach_the_cut_at_all() -> None:
    m = 10
    q = _fold_weight_matrix(cycle_shells(m, 2), GAMMAS, 0.3)
    moved = q + 2.5 * np.eye(m)
    for fold in _designs(m):
        cross = fold[:, None] != fold[None, :]
        assert float(moved[cross].sum()) == pytest.approx(float(q[cross].sum()), abs=1e-12)


def test_the_polymake_rational_optimum_is_reproduced() -> None:
    """polymake 4.15, exact rational LP over the 35 balanced cut vectors at m = 8: max cut 188/25.

    gammas = (1, 7/10, 2/5), x = 3/10 makes Q rational, 1'Q1 = 5256/125, so the same-fold mass at
    the optimum is 5256/125 - 2*188/25 = 3376/125.
    """
    design = optimal_fold_partition(cycle_shells(8, 2), GAMMAS, 0.3, lag=1)
    assert design.objective == pytest.approx(3376 / 125, abs=1e-12)


def test_normaliz_counts_the_feasible_set() -> None:
    """Normaliz 3.11.1 on the hypersimplex {x in [0,1]^m : sum x = m/2}: 70, 252, 924, 3432."""
    assert [comb(m, m // 2) for m in (8, 10, 12, 14)] == [70, 252, 924, 3432]
    assert [comb(m - 1, m // 2 - 1) for m in (8, 10, 12, 14)] == [35, 126, 462, 1716]


def test_fold_exactness_certificate() -> None:
    curve = fold_exactness_certificate(sizes=(24,), small=(10, 12))
    assert curve.ok
    assert curve.enumeration_match < 1e-9
    assert curve.duality_residual < 1e-9
    assert curve.cut_is_maximal
    assert curve.antipodal_guard
    assert curve.shift_spread < 1e-9
    assert curve.diagonal_shift_moves_cut < 1e-9
    # the two gaps are different quantities: the design can be exact while the bound is loose
    assert curve.heuristic_shortfall.min() == pytest.approx(0.0, abs=1e-9)
    assert curve.bound_looseness.min() > 0.005
    # and what a caller sees is the two stacked, so it never under-states the shortfall
    assert curve.gap_dominates_shortfall
    assert np.all(curve.certified_gap >= curve.heuristic_shortfall - 1e-12)
    assert np.all(curve.certified_gap > curve.bound_looseness - 1e-12)
