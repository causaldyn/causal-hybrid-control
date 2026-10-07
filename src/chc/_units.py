"""Each column read in its own units: one rounding rule, its centring, and power-of-two scaling.

A column whose spread about its mean, the population standard deviation, is at most ``ROUNDING``
eps of its dtype times its root mean square moved by rounding alone. A column of zeros is
rounding. Both sizes are taken relative to the column's largest entry, so no square under- or
overflows. A centred ridge or least-squares solve zeroes a rounding column's centred values, so its
coefficient is exactly 0 and the intercept carries its level; its ridge scale is 1 (ADR 0057).

Columns are centred twice, the deviations' own mean taken off them. A mean is rounded, in float32
to about 1e-7 of its size, and NumPy sums a 2-D array's columns row by row, so the rounding of its
column means grows with the rows. This module imports nothing from ``chc``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

ROUNDING = 64
"""A spread at most this many eps of its column's root mean square is rounding. Centred once by
JAX, a column whose entries are all equal measured a spread of at most 4.13 eps in float32 and 6.03
eps in float64, from 2 to 1e7 rows; centred once by NumPy, 119 eps at 1e3 rows and 8.5e5 at 1e7.
Centred twice, at most 2.8e-7 eps in float32 and 0 in float64."""


class Centred(NamedTuple):
    """Columns centred for a solve beside a free intercept."""

    centre: Array  # each column's mean
    shift: Array  # the deviations' own mean, taken off them in a second pass
    deviations: Array  # the columns less ``centre`` and ``shift``; 0 in a rounding column
    scales: Array  # each column's variance; 1 in a rounding column
    rounding: Array  # whether the column moved by rounding alone


class CentredNp(NamedTuple):
    """:class:`Centred` in NumPy."""

    centre: NDArray[np.float64]
    shift: NDArray[np.float64]
    deviations: NDArray[np.float64]
    scales: NDArray[np.float64]
    rounding: NDArray[np.bool_]


def root_mean_square(columns: Array) -> Array:
    """Each column's root mean square, taken relative to its largest entry; 0 for a column of
    zeros."""
    peak = jnp.max(jnp.abs(columns), axis=0)
    unit = jnp.where(peak > 0.0, peak, 1.0)
    return unit * jnp.sqrt(jnp.mean((columns / unit) ** 2, axis=0))


def power_of_two(size: Array) -> Array:
    """The power of two nearest the reciprocal of each ``size``, 1 where it is 0. ``jnp.ldexp``
    makes it exactly; ``jnp.exp2`` is inexact on XLA's CPU backend."""
    exponent = jnp.round(jnp.log2(jnp.where(size > 0.0, size, 1.0))).astype(jnp.int32)
    return jnp.ldexp(jnp.ones_like(size), -exponent)


def least_squares(design: Array, target: Array) -> Array:
    """Least-squares coefficients of ``target`` on ``design``, each column scaled first by
    :func:`power_of_two` of its root mean square, so the rank cutoff of ``rcond=None`` falls at one
    share of every column. The scaling is exact."""
    scale = power_of_two(root_mean_square(design))
    coeffs, *_ = jnp.linalg.lstsq(design * scale, target, rcond=None)
    return coeffs * scale.reshape(-1, *(1,) * (coeffs.ndim - 1))


def _measured(columns: Array) -> tuple[Array, Array, Array, Array, Array]:
    """Each column's mean, the first deviations' mean, the deviations centred twice (exactly 0 in a
    column whose entries are all equal), their spread, and whether it is rounding."""
    centre = jnp.mean(columns, axis=0)
    first = columns - centre
    shift = jnp.mean(first, axis=0)
    constant = jnp.max(columns, axis=0) == jnp.min(columns, axis=0)
    deviations = jnp.where(constant, 0.0, first - shift)
    spread = root_mean_square(deviations)
    rounding = spread <= ROUNDING * jnp.finfo(spread.dtype).eps * root_mean_square(columns)
    return centre, shift, deviations, spread, rounding


def spread_and_rounding(columns: Array) -> tuple[Array, Array]:
    """Each column's standard deviation, and whether it is rounding."""
    _, _, _, spread, rounding = _measured(columns)
    return spread, rounding


def spread(columns: Array) -> Array:
    """Each column's standard deviation, or 1 where it is rounding."""
    size, rounding = spread_and_rounding(columns)
    return jnp.where(rounding, 1.0, size)


def centred(columns: Array) -> Centred:
    """The columns centred twice, a rounding column zeroed, and each column's ridge scale."""
    centre, shift, deviations, spread, rounding = _measured(columns)
    return Centred(
        centre=centre,
        shift=shift,
        deviations=jnp.where(rounding, 0.0, deviations),
        scales=jnp.where(rounding, 1.0, spread**2),
        rounding=rounding,
    )


def standardising(columns: Array) -> tuple[Array, Array, Array]:
    """``(centre, shift, factor)`` that standardise each column as ``(x - centre - shift) *
    factor``: its mean, the second pass's, and the reciprocal of its spread, or 0 where the spread
    is rounding, so a rounding column reads 0 in every row."""
    centre, shift, _, spread, rounding = _measured(columns)
    return centre, shift, jnp.where(rounding, 0.0, 1.0 / jnp.where(rounding, 1.0, spread))


def standardised(columns: Array) -> Array:
    """The columns standardised by :func:`standardising`."""
    centre, shift, factor = standardising(columns)
    return (columns - centre - shift) * factor


def mean_squares(columns: Array, weights: Array | None = None) -> Array:
    """Each column's mean square, its rows weighted by ``weights``, and 1 for a column of zeros:
    the ridge's scale on a coefficient of a design with no intercept."""
    squares = columns**2 if weights is None else weights[:, None] * columns**2
    rows = columns.shape[0] if weights is None else jnp.sum(weights)
    size = jnp.sum(squares, axis=0) / rows
    return jnp.where(size > 0.0, size, 1.0)


def root_mean_square_np(columns: NDArray[np.float64]) -> NDArray[np.float64]:
    """:func:`root_mean_square` in NumPy."""
    peak = np.max(np.abs(columns), axis=0)
    unit = np.where(peak > 0.0, peak, 1.0)
    return unit * np.sqrt(np.mean((columns / unit) ** 2, axis=0))


def power_of_two_np(size: NDArray[np.float64]) -> NDArray[np.float64]:
    """:func:`power_of_two` in NumPy."""
    exponent = np.round(np.log2(np.where(size > 0.0, size, 1.0))).astype(int)
    return np.ldexp(np.ones_like(size), -exponent)


def _measured_np(
    columns: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
]:
    """:func:`_measured` in NumPy."""
    centre = np.mean(columns, axis=0)
    first = columns - centre
    shift = np.mean(first, axis=0)
    constant = np.max(columns, axis=0) == np.min(columns, axis=0)
    deviations = np.where(constant, 0.0, first - shift)
    spread = root_mean_square_np(deviations)
    rounding = spread <= ROUNDING * np.finfo(spread.dtype).eps * root_mean_square_np(columns)
    return centre, shift, deviations, spread, rounding


def spread_and_rounding_np(
    columns: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """:func:`spread_and_rounding` in NumPy."""
    _, _, _, spread, rounding = _measured_np(columns)
    return spread, rounding


def centred_np(columns: NDArray[np.float64]) -> CentredNp:
    """:func:`centred` in NumPy."""
    centre, shift, deviations, spread, rounding = _measured_np(columns)
    return CentredNp(
        centre=centre,
        shift=shift,
        deviations=np.where(rounding, 0.0, deviations),
        scales=np.where(rounding, 1.0, spread**2),
        rounding=rounding,
    )


def standardised_np(columns: NDArray[np.float64]) -> NDArray[np.float64]:
    """:func:`standardised` in NumPy."""
    _, _, deviations, spread, rounding = _measured_np(columns)
    return deviations * np.where(rounding, 0.0, 1.0 / np.where(rounding, 1.0, spread))


def mean_squares_np(columns: NDArray[np.float64]) -> NDArray[np.float64]:
    """:func:`mean_squares` in NumPy, unweighted."""
    size = np.mean(columns**2, axis=0)
    return np.where(size > 0.0, size, 1.0)
