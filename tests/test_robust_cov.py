"""chc._robust_cov: the library's one robust covariance, over rows, within clusters, two ways and
across one group's periods, with its kind, ``G``, factor and degrees of freedom.

Checked on random scores and regressions: each row its own cluster reads the rows, at CR1's factor
HC1 and at ``G / (G - 1)`` alone HC0 times ``N / (N - 1)``; permuting the rows with their labels
moves nothing beyond rounding; scaling the scores by ``c`` scales the covariance by ``c^2`` and a
regression's design by ``c`` its coefficients' by ``1 / c^2``; two ways is the two dimensions' sums
less their intersection's, with its part below nothing taken away, which adds a covariance to them
and leaves the read none, and whole the sums read the covariance on average where the clipped ones
read above it; HC3 and CR1 against numpy by hand; NumPy and JAX scores read alike. Matrices are
compared on the columns' own scales, where rounding is of one size.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from chc._robust_cov import (
    _clustered_squares,
    _cosine_count,
    _less_below_nothing,
    robust_cov,
    sandwich,
)


def _draw(data: st.DataObject) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores ``(N, p)`` with columns of scales 0.1 to 10, and codes from 0 of one dimension and
    of two, every code holding a row. Each dimension has more codes than the scores have columns,
    so its sums span them: with fewer, the two ways' part below nothing is read off directions that
    are rounding, and the rows' order moves it. The part is read off an eigendecomposition in the
    columns' raw units, whose rounding grows with their spread of scales: at 1e-3 to 1e3, the
    rows' order moved a two-way read by 1e-6 of itself."""
    width = data.draw(st.integers(1, 4), label="width")
    rows = data.draw(st.integers(4 * (width + 1), 48), label="rows")
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1), label="seed"))
    scores = rng.standard_normal((rows, width)) * 10.0 ** rng.uniform(-1.0, 1.0, width)
    codes = []
    for _ in range(2):
        count = data.draw(st.integers(width + 1, rows // 2), label="count")
        labels = np.concatenate([np.arange(count), rng.integers(0, count, rows - count)])
        codes.append(rng.permutation(labels))
    return scores, codes[0], np.column_stack(codes)


def _close(got: np.ndarray, want: np.ndarray, rel: float, size: float | None = None) -> None:
    """Entrywise to ``rel`` of ``size``, by default the larger matrix's largest entry: a
    covariance's off-diagonal entries can be rounding of its diagonal's."""
    if size is None:
        size = max(float(np.max(np.abs(want))), float(np.max(np.abs(got))))
    np.testing.assert_allclose(got, want, rtol=0.0, atol=rel * size)


def _scales(scores: np.ndarray) -> np.ndarray:
    """``d d'``, ``d`` each column's root sum of squares: a covariance over it reads each column on
    its own scale."""
    size = np.sqrt(np.sum(scores**2, axis=0))
    return np.outer(size, size)


def _bound(scores: np.ndarray) -> float:
    """The largest entry of ``(sum |s_i|)(sum |s_i|)'``, which bounds every entry of every sum of
    the clusters' outer products, however the rows are grouped: a two-way read that is nothing is
    nothing to the rounding of these sums, not of itself."""
    total = np.abs(scores).sum(axis=0)
    return float(np.max(np.outer(total, total)))


@settings(max_examples=200, deadline=None)
@given(data=st.data())
def test_each_row_its_own_cluster_reads_the_rows(data: st.DataObject) -> None:
    """CR1's ``G / (G - 1) (N - 1) / (N - k)`` at ``G = N`` is HC1's ``N / (N - k)``, and
    ``G / (G - 1)`` alone is HC0 times ``N / (N - 1)``; ``G`` and ``df`` are the rows'."""
    scores, _, _ = _draw(data)
    rows = scores.shape[0]
    n_coeff = data.draw(st.integers(0, 3), label="n_coeff")
    own = np.arange(rows)
    hc1 = robust_cov(scores, n_coeff)
    hc0 = robust_cov(scores, None)
    assert (hc1.kind, hc0.kind) == ("HC1", "HC0")
    assert hc1.factor == pytest.approx(rows / (rows - n_coeff), rel=1e-15, abs=0.0)
    assert hc0.factor == 1.0
    _close(robust_cov(scores, n_coeff, own).cov, hc1.cov, 1e-14)
    _close(robust_cov(scores, None, own).cov, hc0.cov * rows / (rows - 1), 1e-14)
    for read in (hc1, hc0, robust_cov(scores, n_coeff, own)):
        assert (read.groups, read.df) == (rows, rows - 1)
    _close(hc0.cov, scores.T @ scores, 1e-14)


@settings(max_examples=200, deadline=None)
@given(data=st.data())
def test_permuting_the_rows_with_their_labels_moves_nothing(data: st.DataObject) -> None:
    """Over rows, one way and two ways; each cluster's sum is a sum, whatever the rows' order."""
    scores, one, two = _draw(data)
    order = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1))).permutation(len(scores))
    scales = _scales(scores)
    size = 2.0 * _bound(scores / np.sqrt(np.diag(scales)))
    for clusters in (None, one, two):
        first = robust_cov(scores, 2, clusters)
        moved = robust_cov(scores[order], 2, None if clusters is None else clusters[order])
        assert (moved.kind, moved.groups, moved.df, moved.factor) == (
            first.kind,
            first.groups,
            first.df,
            first.factor,
        )
        for got, want in zip(moved.reads, first.reads, strict=True):
            _close(got / scales, want / scales, 1e-9, size)


@settings(max_examples=200, deadline=None)
@given(data=st.data(), power=st.integers(-20, 20))
def test_scaling_the_scores_scales_the_covariance_by_its_square(
    data: st.DataObject, power: int
) -> None:
    """``c = 2^power``, so the scaling is exact: the scores' covariance by ``c^2`` and a
    regression's design by ``c``, its residual held, its coefficients' by ``1 / c^2``."""
    scores, one, two = _draw(data)
    scale = 2.0**power
    for clusters in (None, one, two):
        first, scaled = robust_cov(scores, 1, clusters), robust_cov(scale * scores, 1, clusters)
        for got, want in zip(scaled.reads, first.reads, strict=True):
            _close(got, scale**2 * want, 1e-12)
    design = (
        scores[:, :2]
        if scores.shape[1] > 1
        else np.column_stack([scores[:, 0], np.ones(len(scores))])
    )
    residual = np.random.default_rng(1).standard_normal(len(scores))
    for clusters in (None, one, two):
        bread = np.linalg.inv(design.T @ design)
        moved = scale * design
        first = sandwich(bread, design, residual, clusters, 2)
        scaled = sandwich(np.linalg.inv(moved.T @ moved), moved, residual, clusters, 2)
        for got, want in zip(scaled.reads, first.reads, strict=True):
            _close(got, want / scale**2, 1e-9)


@settings(max_examples=200, deadline=None)
@given(data=st.data())
def test_two_ways_are_both_dimensions_less_their_intersection(data: st.DataObject) -> None:
    """``c (A + B - I)``, ``A`` and ``B`` each dimension's sums and ``I`` the cells', at CR1's
    factor ``c`` at the smaller dimension's ``G``, its part below nothing taken away, which adds
    a covariance and leaves none; each dimension alone at its own factor; ``df = min(G) - 1``.
    Unclipped, the sums whole, as a bias linear in the covariance reads them."""
    scores, _, two = _draw(data)
    rows = scores.shape[0]
    cells = np.unique(two, axis=0, return_inverse=True)[1].reshape(-1)
    raw = [robust_cov(scores, None, codes) for codes in (two[:, 0], two[:, 1], cells)]
    first, second, both = (read.cov / read.factor for read in raw)
    counts = [int(codes.max()) + 1 for codes in two.T]
    smaller = min(counts)
    read = robust_cov(scores, 3, two)
    factor = smaller / (smaller - 1) * (rows - 1) / (rows - 3)
    assert (read.kind, read.groups, read.df) == ("two-way", smaller, smaller - 1)
    assert read.factor == pytest.approx(factor, rel=1e-15, abs=0.0)
    spread, every = first + second - both, first + second + both
    scales = _scales(scores)
    size = 2.0 * _bound(scores / np.sqrt(np.diag(scales)))
    _close(read.cov / scales, factor * _less_below_nothing(spread, every) / scales, 1e-9, size)
    assert np.min(np.linalg.eigvalsh((read.cov - factor * spread) / scales)) >= -1e-9 * size
    assert np.min(np.linalg.eigvalsh(read.cov / scales)) >= -1e-9 * size
    rows3 = scores[:, None]
    sums = [_clustered_squares(rows3, codes)[0] for codes in (two[:, 0], two[:, 1], cells)]
    whole = _clustered_squares(rows3, two, clip=False)
    np.testing.assert_array_equal(whole[0], sums[0] + sums[1] - sums[2])
    for kept, clipped in zip(whole[1:], _clustered_squares(rows3, two)[1:], strict=True):
        np.testing.assert_array_equal(kept, clipped)
    for alone, square, count in zip(read.alone, (first, second), counts, strict=True):
        _close(alone, count / (count - 1) * (rows - 1) / (rows - 3) * square, 1e-12)


def test_the_whole_two_ways_read_the_covariance_and_the_clipped_read_above_it() -> None:
    """Independent rows of covariance ``I`` on a 3 by 4 grid: each of ``A``, ``B`` and ``I``
    reads ``N I`` on average, so the whole sums do, and taking their part below nothing away
    reads above it. Over 3000 logs the whole sums' trace sits within 4 standard errors of
    ``N p`` and the clipped one more than 2 above it."""
    codes = np.stack(np.meshgrid(np.arange(3), np.arange(4), indexing="ij"), -1).reshape(-1, 2)
    rng = np.random.default_rng(0)
    traces = []
    for _ in range(3000):
        rows = rng.standard_normal((12, 1, 2))
        whole = _clustered_squares(rows, codes, clip=False)[0]
        clipped = _clustered_squares(rows, codes)[0]
        traces.append((np.trace(whole), np.trace(clipped)))
    whole_mean, clipped_mean = np.mean(traces, axis=0)
    whole_se, clipped_se = np.std(traces, axis=0, ddof=1) / np.sqrt(len(traces))
    assert abs(whole_mean - 24.0) < 4.0 * whole_se
    assert clipped_mean - 24.0 > 2.0 * clipped_se


def test_a_second_way_of_one_row_a_cell_reads_the_first_way_alone() -> None:
    """Each row its own period: the periods' sums are the cells', so two ways read the units'
    sums, at the smaller dimension's factor, with nothing below nothing to take away."""
    rng = np.random.default_rng(3)
    scores = rng.standard_normal((30, 3))
    units = np.arange(30) % 6
    two = np.column_stack([units, np.arange(30)])
    read = robust_cov(scores, 2, two)
    one = robust_cov(scores, 2, units)
    _close(read.cov, one.cov, 1e-13)
    assert (read.groups, one.groups, read.df) == (6, 6, 5)


@pytest.mark.parametrize("clusters", [None, "one", "two"])
def test_hc3_and_cr1_read_as_numpy_by_hand(clusters: str | None) -> None:
    """HC3 each residual over one less its leverage, against ``t(N - 1)``; CR1 the clusters'
    summed scores at ``G / (G - 1) (N - 1) / (N - k)``, against ``t(G - 1)``."""
    rng = np.random.default_rng(5)
    design = np.column_stack([rng.standard_normal(50), np.ones(50)])
    residual = rng.standard_normal(50) * (1.0 + np.abs(design[:, 0]))
    bread = np.linalg.inv(design.T @ design)
    units = np.arange(50) % 7
    codes = {None: None, "one": units, "two": np.column_stack([units, np.arange(50) % 5])}
    read = sandwich(bread, design, residual, codes[clusters], 2)
    if clusters is None:
        leverage = np.einsum("ij,jk,ik->i", design, bread, design)
        scores = design * (residual / (1.0 - leverage))[:, None]
        assert (read.kind, read.groups, read.df, read.factor) == ("HC3", 50, 49, 1.0)
        _close(read.cov, bread @ scores.T @ scores @ bread, 1e-13)
        return
    sums = np.stack([(design * residual[:, None])[units == g].sum(axis=0) for g in range(7)])
    by_units = 7 / 6 * 49 / 48 * bread @ sums.T @ sums @ bread
    if clusters == "one":
        assert (read.kind, read.groups, read.df) == ("CR1", 7, 6)
        _close(read.cov, by_units, 1e-13)
    else:
        # the units alone, at their own factor; the periods, 5, are the smaller dimension
        assert (read.kind, read.groups, read.df) == ("two-way", 5, 4)
        _close(read.alone[0], by_units, 1e-13)


@pytest.mark.parametrize("clusters", [None, "one", "two", "periods"])
def test_numpy_and_jax_scores_read_alike(clusters: str | None) -> None:
    """JAX scores are read in JAX, NumPy scores by their float64 sums: the same covariance, kind,
    ``G`` and ``df``; under periods ``G`` and ``df`` are the cosines' count."""
    rng = np.random.default_rng(11)
    scores = rng.standard_normal((60, 3))
    units = np.arange(60) % 6
    periods = np.repeat(np.arange(10), 6)
    codes = {
        None: (None, None),
        "one": (units, None),
        "two": (np.column_stack([units, periods]), None),
        "periods": (None, periods),
    }[clusters]
    plain = robust_cov(scores, 2, *codes)
    traced = robust_cov(jnp.asarray(scores), 2, *codes)
    assert (traced.kind, traced.groups, traced.df, traced.factor) == (
        plain.kind,
        plain.groups,
        plain.df,
        plain.factor,
    )
    for got, want in zip(traced.reads, plain.reads, strict=True):
        _close(np.asarray(got), want, 1e-13)
    if clusters == "periods":
        assert plain.kind == "cosines"
        assert plain.groups == plain.df == _cosine_count(10)
