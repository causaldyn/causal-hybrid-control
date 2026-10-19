"""The fold design's diagonally shifted Ky Fan bound (``validation/fold_shifted_bound.mac``)."""

import itertools

import numpy as np
import pytest

from chc.network_causal import cycle_shells, graph_shells
from chc.research.regret import (
    _fold_weight_matrix,
    _named_topologies,
    fold_exactness_certificate,
    fold_heuristic_certificate,
    optimal_fold_partition,
)

GAMMAS = (1.0, 0.7, 0.4)
DECAYS = (0.01, 0.6, 0.9)


def _three_way_optimum(q: np.ndarray) -> float:
    """Least same-fold mass over every balanced 3-partition, by enumeration (5775 at m = 12)."""
    m = q.shape[0]
    size = m // 3
    labels = []
    for first in itertools.combinations(range(1, m), size - 1):
        rest = [v for v in range(1, m) if v not in first]
        for second in itertools.combinations(rest[1:], size - 1):
            fold = np.full(m, 2)
            fold[[0, *first]] = 0
            fold[[rest[0], *second]] = 1
            labels.append(fold)
    onehot = np.eye(3)[np.asarray(labels)]
    return float(np.einsum("nik,ij,njk->n", onehot, q, onehot).min())


@pytest.mark.parametrize(("m", "k_folds"), [(12, 2), (16, 2), (12, 3)])
def test_the_shifted_bound_lies_between_ky_fan_and_the_optimum(m: int, k_folds: int) -> None:
    for x in DECAYS:
        for name, adjacency in _named_topologies(m).items():
            shells = graph_shells(adjacency, 2)
            design = optimal_fold_partition(shells, GAMMAS, x, lag=1, k_folds=k_folds)
            if k_folds == 2:
                assert design.route == "enumeration"
                optimum = design.objective
            else:
                optimum = _three_way_optimum(_fold_weight_matrix(shells, GAMMAS, x))
            assert design.kyfan_bound <= design.lower_bound, (name, x)
            assert design.lower_bound <= optimum + 1e-9, (name, x)
            assert design.gap == pytest.approx(
                (design.objective - design.lower_bound) / abs(design.lower_bound), abs=0.0
            )


def _bisection_sdp(q: np.ndarray, gap: float = 1e-10) -> tuple[float, float]:
    """``(low, high)`` around ``min tr(QX)`` over ``diag X = 1``, ``X1 = 0``, ``X`` psd.

    A log-barrier Newton method on the dual, ``max sum(y)`` with ``V'(Q - diag(y))V`` psd, ``V``
    an orthonormal basis of the mean-free subspace from a QR factorisation. Every iterate is dual
    feasible, so ``sum(y)`` is a lower bound; on the central path the gap is ``mu (m - 1)``, and
    ``high`` allows twice that. The path stops once ``mu (m - 1)`` is ``gap`` of ``max|Q|``.
    """
    m = q.shape[0]
    basis = np.linalg.qr(np.column_stack([np.ones(m), np.eye(m)[:, : m - 1]]))[0][:, 1:]
    compressed = basis.T @ q @ basis
    y = np.full(m, np.linalg.eigvalsh(compressed)[0] - 1.0)
    scale = float(np.abs(q).max())
    mu = scale

    def slack(dual: np.ndarray) -> np.ndarray:
        return compressed - basis.T @ (dual[:, None] * basis)

    while True:
        for _ in range(100):
            inverse = basis @ np.linalg.inv(slack(y)) @ basis.T
            grad = 1.0 - mu * np.diag(inverse)
            step = np.linalg.solve(mu * inverse * inverse, grad)
            decrement = float(grad @ step) / mu
            t = 1.0 if decrement <= 0.25 else 1.0 / (1.0 + np.sqrt(decrement))
            while np.linalg.eigvalsh(slack(y + t * step))[0] <= 0.0:
                t *= 0.5
            y = y + t * step
            if decrement < 1e-20:
                break
        if mu * (m - 1) <= gap * scale:
            return float(y.sum()), float(y.sum()) + 2.0 * mu * (m - 1)
        mu *= 0.2


@pytest.mark.parametrize("m", [12, 16])
def test_two_folds_certify_with_the_bisection_sdp(m: int) -> None:
    # At K = 2 the maximised shifted bound is (1'Q1 + SDP)/2 (Poljak and Rendl 1995; STEP 5 of
    # the .mac file). The library's smoothed search stops short of it by at most 1.55e-6 of the
    # bound on these instances, measured on a 16-unit grid at x = 0.9.
    for x in DECAYS:
        for name, adjacency in _named_topologies(m).items():
            shells = graph_shells(adjacency, 2)
            q = _fold_weight_matrix(shells, GAMMAS, x)
            low, high = (float(q.sum()) + side for side in _bisection_sdp(q))
            low, high = low / 2.0, high / 2.0
            design = optimal_fold_partition(shells, GAMMAS, x, lag=1)
            assert design.route == "enumeration"
            assert low - 2e-6 * abs(low) <= design.lower_bound <= high + 1e-9, (name, x)


def test_on_a_circulant_the_shift_buys_nothing() -> None:
    # The bound is concave in the shift and blind to a uniform one, so averaging any shift over
    # a transitive automorphism group gives a uniform shift at least as good: the plain bound.
    from chc.research.regret import _shifted_kyfan_bound

    rng = np.random.default_rng(3)
    row = rng.random(16)
    row = row + row[(-np.arange(16)) % 16]
    shifts = np.arange(16)
    weights = {"random circulant": row[(shifts[:, None] - shifts[None, :]) % 16]}
    for m in (24, 60):
        for x in DECAYS:
            weights[f"cycle {m} x={x}"] = _fold_weight_matrix(cycle_shells(m, 2), GAMMAS, x)
    for name, q in weights.items():
        for k_folds in (2, 3, 4):
            if q.shape[0] % k_folds:
                continue
            shifted, plain = _shifted_kyfan_bound(q, k_folds)
            assert shifted == pytest.approx(plain, rel=0.0, abs=1e-12), (name, k_folds)

    curve = fold_exactness_certificate(sizes=(24,), small=(10,))
    assert curve.ok
    assert curve.shift_gain < 1e-12
    assert np.all(curve.shifted >= curve.kyfan)


def _gaps(name: str, m: int, x: float) -> tuple[float, float]:
    design = optimal_fold_partition(graph_shells(_named_topologies(m)[name], 2), GAMMAS, x, lag=1)
    assert design.exhaustive
    kyfan = (design.objective - design.kyfan_bound) / abs(design.kyfan_bound)
    return kyfan, design.gap


def test_the_shift_closes_the_gap_off_the_cycle() -> None:
    kyfan, shifted = _gaps("path", 16, 0.01)
    assert kyfan == pytest.approx(0.04397, rel=1e-3, abs=0.0)
    assert shifted <= 0.0040  # measured 0.34%
    for name in ("random0", "random1", "random2", "random3"):
        kyfan, shifted = _gaps(name, 16, 0.01)
        assert shifted <= 0.5 * kyfan, name  # measured ratios 0.36-0.47


def test_a_degenerate_spectrum_does_not_stall_the_shift() -> None:
    # At u = 0 the star's and the barbell's smallest mean-free eigenvalues are repeated, so the
    # exact bound is not differentiable there and a quasi-Newton step on it never leaves zero.
    for name, x in (("star", 0.01), ("star", 0.6), ("star", 0.9), ("barbell", 0.01)):
        kyfan, shifted = _gaps(name, 16, x)
        assert kyfan > 1e-3, (name, x)
        assert shifted < 1e-7, (name, x)  # the bound certifies the enumerated optimum
    kyfan, shifted = _gaps("barbell", 16, 0.6)
    assert shifted < 0.5 * kyfan  # measured 0.16% against 0.45%


def test_the_shifted_bound_is_deterministic() -> None:
    from chc.research.regret import _shifted_kyfan_bound

    for name in ("path", "random3"):
        q = _fold_weight_matrix(graph_shells(_named_topologies(16)[name], 2), GAMMAS, 0.6)
        assert _shifted_kyfan_bound(q, 2) == _shifted_kyfan_bound(q, 2)
        assert _shifted_kyfan_bound(q, 4) == _shifted_kyfan_bound(q, 4)


def test_the_heuristic_certificate_reports_both_gaps() -> None:
    curve = fold_heuristic_certificate(m=12)
    assert curve.ok
    star = curve.names.index("star")
    assert curve.kyfan_gap[star] == pytest.approx(0.0077, rel=1e-2, abs=0.0)
    assert curve.shifted_gap[star] < 1e-7
    assert np.all(curve.shifted_gap <= curve.kyfan_gap)
    assert np.all(curve.shifted_gap >= -1e-12)
    cycle = curve.names.index("cycle")
    assert curve.shifted_gap[cycle] == pytest.approx(curve.kyfan_gap[cycle], rel=1e-9, abs=0.0)
