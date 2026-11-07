"""Robust covariances of a sum of scores, each small-sample factor and degrees of freedom stated
once: over independent rows, within clusters (CR1), two ways (Cameron, Gelbach and Miller 2011),
and across one group's periods by cosines (Lazarus, Lewis, Stock and Watson 2018), as ADR 0053,
ADR 0061, ADR 0062 and ADR 0075 read them.

:func:`robust_cov` reads a :class:`RobustCov` off the rows' scores and :func:`sandwich` off a
regression's design and residual: the covariance, its kind, ``G``, the independent terms it sums,
the factor it carries and ``df``, the degrees of freedom of the ``t`` a test read off it is sized
against. The kinds and their factors, over ``N`` rows, ``k`` coefficients and ``G`` clusters:

- ``"HC1"``, rows: ``N / (N - k)``; ``"HC0"`` where ``k`` is not given, 1.
- ``"HC3"``, rows: each residual over one less its leverage, no factor (MacKinnon and White 1985).
- ``"CR1"``, clusters: ``G / (G - 1) (N - 1) / (N - k)``; where ``k`` is not given,
  ``G / (G - 1)`` alone, as the residual checks read it, which for a mean, ``k = 1``, is the same.
- ``"two-way"``: CR1's factor at the smaller dimension's ``G``, and each dimension alone at its own.
- ``"cosines"``: the rows' factor.

``df`` is the rows or the clusters less one, two ways the smaller dimension's count less one, or
over one group's periods the cosines' count ``nu``, whose cosines leave out the mean the moment
sets to nothing. This module imports nothing from ``chc``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

Kind = Literal["HC0", "HC1", "HC3", "CR1", "two-way", "cosines"]


@dataclass(frozen=True)
class RobustCov:
    """A robust covariance and what a test read off it is sized against."""

    # the covariance, ``(p, p)``, a JAX array for JAX scores; two ways, the two dimensions' sums
    # less their intersection's, less any part below nothing (:func:`_less_below_nothing`)
    cov: Any
    df: int  # the ``t``'s degrees of freedom
    kind: Kind
    # ``G``: the independent terms the covariance sums, the rows, the clusters, two ways the
    # smaller dimension's, or the cosines
    groups: int
    factor: float  # the small-sample factor ``cov`` carries; 1 under HC3
    # two ways, each dimension's sums alone at its own factor, which a reader of the channel's
    # error takes the largest of with ``cov`` (MacKinnon, Nielsen and Webb 2024); () otherwise
    alone: tuple[Any, ...] = ()

    @property
    def reads(self) -> tuple[Any, ...]:
        """``cov``, then each dimension's alone."""
        return (self.cov, *self.alone)


def robust_cov(
    scores: Any,
    n_coeff: int | None,
    clusters: np.ndarray | None = None,
    periods: np.ndarray | None = None,
) -> RobustCov:
    """The covariance of the sum of ``scores``' rows, ``(N, p)``: the sum of their outer products,
    or under ``clusters`` of their sums within each cluster, codes from 0, ``(N,)`` or two ways
    ``(N, 2)``, or under ``periods``, one group's, across the cosines of :func:`_cosines`; times
    the small-sample factor of :func:`_small_sample` for ``n_coeff`` coefficients ``k``.

    JAX scores are read in JAX at their own dtype, the factor taken on the scores before the
    product, and two ways by :func:`_clustered_squares`; NumPy scores in float64 by
    :func:`_clustered_squares`, the factor taken on its sums. ``periods`` come before ``clusters``.
    """
    n = scores.shape[0]
    count = None if clusters is None or periods is not None else _cluster_count(clusters)
    factor = _small_sample(n, n_coeff, count)
    if periods is not None:
        kind: Kind = "cosines"
        groups = _period_cosines(periods)
    elif clusters is None:
        kind, groups = ("HC0" if n_coeff is None else "HC1"), n
    else:
        assert count is not None
        kind, groups = ("CR1" if clusters.ndim == 1 else "two-way"), count
    if not isinstance(scores, jax.Array):
        squares = _clustered_squares(np.asarray(scores)[:, None], clusters, periods)
        reads = tuple(factor * square for square in squares)
    elif periods is not None:
        projected = jnp.asarray(_cosines(periods)) @ scores
        reads = (factor * projected.T @ projected,)
    elif clusters is None:
        reads = (factor * scores.T @ scores,)
    elif clusters.ndim == 1:
        summed = jax.ops.segment_sum(
            scores, jnp.asarray(clusters), num_segments=int(clusters.max()) + 1
        )
        reads = (factor * summed.T @ summed,)
    else:
        squares = _clustered_squares(np.asarray(scores)[:, None], clusters)
        reads = tuple(factor * jnp.asarray(square) for square in squares)
    df = _degrees_of_freedom(n, clusters, periods)
    return RobustCov(reads[0], df, kind, groups, factor, reads[1:])


def sandwich(
    bread: Any,
    design: Any,
    residual: Any,
    clusters: np.ndarray | None = None,
    n_coeff: int | None = None,
) -> RobustCov:
    """``bread meat bread``, the covariance of a regression's coefficients on ``design``
    ``(N, k)``, given each row's ``residual`` ``(N,)`` and ``bread`` the inverse of the design's
    Gram. Without ``clusters``, HC3: each row's residual over one less its leverage
    ``h_i = x_i' bread x_i`` (MacKinnon and White 1985), against ``t(N - 1)``. Under ``clusters``
    the meat is :func:`robust_cov` of the rows' scores ``x_i e_i``, CR1 at ``n_coeff``'s factor;
    two ways, each dimension's alone too."""
    if clusters is None:
        xp = jnp if isinstance(design, jax.Array) else np
        leverage = xp.einsum("ij,jk,ik->i", design, bread, design)
        weighted = design * (residual / (1.0 - leverage))[:, None]
        rows = design.shape[0]
        return RobustCov(bread @ (weighted.T @ weighted) @ bread, rows - 1, "HC3", rows, 1.0)
    read = robust_cov(design * residual[:, None], n_coeff, clusters)
    return RobustCov(
        bread @ read.cov @ bread,
        read.df,
        read.kind,
        read.groups,
        read.factor,
        tuple(bread @ square @ bread for square in read.alone),
    )


def _cluster_factor(clusters: int) -> float:
    """CR1's ``G / (G - 1)`` over ``G`` clusters."""
    return clusters / (clusters - 1)


def _small_sample(rows: int, n_coeff: int | None, clusters: int | None) -> float:
    """The robust covariance's small-sample factor: ``N / (N - k)`` over rows, and CR1's
    ``G / (G - 1) (N - 1) / (N - k)`` over ``G`` clusters, which is the same where each row is its
    own. ``n_coeff`` None counts no coefficients: 1 over rows, ``G / (G - 1)`` alone over
    clusters."""
    if n_coeff is None:
        return 1.0 if clusters is None else _cluster_factor(clusters)
    if clusters is None:
        return rows / max(rows - n_coeff, 1)
    return _cluster_factor(clusters) * (rows - 1) / max(rows - n_coeff, 1)


def _cluster_count(clusters: np.ndarray) -> int:
    """``G`` in CR1's factor: the clusters' count, or two-way the smaller dimension's, which bounds
    how many independent terms the error is read from (Cameron, Gelbach and Miller 2011)."""
    return int(np.min(clusters.max(axis=0))) + 1


def _cosine_count(rows: int) -> int:
    """``nu``, the cosines ``T`` periods are read with: the most whole ones at or below
    ``0.4 T^(2/3)``, the count Lazarus, Lewis, Stock and Watson (2018) recommend, and one at the
    least. Read in whole numbers, ``1000 nu^3 <= 64 T^2``, since in floating point
    ``0.4 * 1000 ** (2 / 3)`` falls short of 40."""
    count = max(int(0.4 * rows ** (2.0 / 3.0)), 1)
    while 1000 * (count + 1) ** 3 <= 64 * rows**2:
        count += 1
    while count > 1 and 1000 * count**3 > 64 * rows**2:
        count -= 1
    return count


def _period_cosines(periods: np.ndarray) -> int:
    """``nu`` for transitions labelled ``periods``: :func:`_cosine_count` of the distinct periods,
    which on one unit are its transitions."""
    return _cosine_count(int(np.unique(periods).size))


def _cosines(periods: np.ndarray) -> np.ndarray:
    """``(nu, N)``: ``sqrt(2 / nu) cos(pi j (p_i + 1/2) / P)`` for ``j`` from 1 to ``nu``
    (:func:`_period_cosines`), at each transition's period ``p_i`` counted from the log's first,
    over the ``P`` periods it spans, a period it lacks holding no score, as Parzen (1963) reads a
    series with gaps.

    ``sum_j c_j c_j'`` with ``c_j = sum_i w_ji s_i`` is the equal-weighted cosine estimate of the
    covariance of the scores' sum (Lazarus, Lewis, Stock and Watson 2018). Over a span without
    gaps each cosine sums to nothing, so a mean of the scores, which the moment sets to nothing,
    moves none of it. Where the scores are independent each ``c_j c_j'`` reads their covariance;
    where they persist, the cosines' low frequencies read the long-run one. As the log grows at a
    fixed ``nu`` the ``c_j`` read as independent normals, so a ``t`` statistic read off the
    estimate is a ``t(nu)``. The transitions of one period share its weights, so ``c_j`` reads the
    sums of the scores within each period, whatever their dependence across the transitions of
    one period: Driscoll and Kraay's (1998) reading, here with the cosines' kernel.
    """
    count = _period_cosines(periods)
    span = int(periods.max()) + 1
    frequencies = np.arange(1, count + 1)[:, None]
    return math.sqrt(2.0 / count) * np.cos(math.pi * frequencies * (periods[None, :] + 0.5) / span)


def _degrees_of_freedom(rows: int, clusters: np.ndarray | None, periods: np.ndarray | None) -> int:
    """The ``t`` a test read off the channel's error is sized against: one fewer than the
    independent terms its spread sums, the rows or the clusters (ADR 0062), or ``nu`` over one
    group's periods, whose cosines leave out the mean the moment sets to nothing (ADR 0075)."""
    if periods is not None:
        return _period_cosines(periods)
    return (rows if clusters is None else _cluster_count(clusters)) - 1


def _clustered_squares(
    rows: np.ndarray, clusters: np.ndarray | None, periods: np.ndarray | None = None
) -> tuple[np.ndarray, ...]:
    """``rows``' covariance, ``(N, n, ...)`` per transition and state as
    :attr:`chc.dynamics_id.CausalDynamicsFit.influence` is: the sum of the outer products of its
    independent terms, each a transition's states summed, or under ``clusters`` a cluster's
    transitions and states summed. Under ``periods``, one group's, each transition's states summed
    and projected on the cosines of :func:`_cosines`, which sums each period's transitions, as the
    channel's error is read.

    Two-way, three covariances, and a reader takes the one it reads largest (MacKinnon, Nielsen
    and Webb 2024): the two dimensions' sums less their intersection's, which both count (Cameron,
    Gelbach and Miller 2011), less any part that reads below nothing (:func:`_less_below_nothing`);
    then each dimension's sums alone, moved from the smaller dimension's CR1 factor, which
    ``rows`` carry, to its own. The two-way sums alone can read less than either dimension does:
    on a log of 3 units over 4 periods they read nothing of a channel affine in the state.
    """
    if periods is not None:
        projected = np.tensordot(_cosines(periods), rows.sum(axis=1), axes=(1, 0))
        return (np.tensordot(projected, projected, axes=(0, 0)),)
    per_row = rows.sum(axis=1)
    if clusters is None:
        return (np.tensordot(per_row, per_row, axes=(0, 0)),)

    def square(codes: np.ndarray) -> np.ndarray:
        summed = np.zeros((int(codes.max()) + 1, *per_row.shape[1:]))
        np.add.at(summed, codes, per_row)
        return np.tensordot(summed, summed, axes=(0, 0))

    if clusters.ndim == 1:
        return (square(clusters),)
    first, second = square(clusters[:, 0]), square(clusters[:, 1])
    cells = square(np.unique(clusters, axis=0, return_inverse=True)[1].reshape(-1))
    spread, every = first + second - cells, first + second + cells
    both = np.maximum(spread, 0.0) if spread.ndim == 0 else _less_below_nothing(spread, every)
    smaller = _cluster_count(clusters)
    alone = []
    for one, codes in ((first, clusters[:, 0]), (second, clusters[:, 1])):
        count = int(codes.max()) + 1
        # the factor whole, so a dimension of the smaller count keeps its sums bit for bit
        alone.append(one * (_cluster_factor(count) / _cluster_factor(smaller)))
    return (both, *alone)


def _less_below_nothing(spread: np.ndarray, every: np.ndarray) -> np.ndarray:
    """``spread`` less the part it reads below nothing, measured against ``every``, its sums with
    every sign a plus: ``spread - T Q min(L, 0) Q' T'``, ``every = T T'`` and ``Q L Q'`` the
    eigendecomposition of ``T^+ spread T^+'``. ``-every <= spread <= every``, so ``L`` lies in
    ``[-1, 1]``, and a spread with no part below nothing keeps every bit. ``every`` is
    ``spread`` plus twice the cells' sums, and any reference ``a spread + b cells``, the cells'
    sums alone or the units' and the periods' together among them, has the same generalised
    eigenvectors ``v``; each part taken away, ``spread v v' spread / v' spread v``, depends on
    ``v`` alone, so they all take away the same part.

    Both move with the coefficients as the scores do, so a change of their units or of the
    state's zero changes no value read off the result. The eigenvalues of ``spread`` itself, which
    Cameron, Gelbach and Miller (2011) clip, move with them: on a panel of 200 regions over 11
    periods, moving the state's zero moved the channel's error from 0.008164 to 0.009408.
    """
    values, vectors = np.linalg.eigh(every)
    kept = values > values[-1] * values.size * np.finfo(np.float64).eps
    root, whitening = (
        vectors[:, kept] * np.sqrt(values[kept]),
        vectors[:, kept] / np.sqrt(values[kept]),
    )
    shares, turns = np.linalg.eigh(whitening.T @ spread @ whitening)
    lifted = root @ turns
    return spread - (lifted * np.minimum(shares, 0.0)) @ lifted.T
