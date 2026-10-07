"""Autocorrelation-robust conditional-independence testing (the MCI idea, borrowed from tigramite).

A linear partial-correlation (ParCorr) test of ``x ⊥ y | z``: residualise ``x`` and ``y`` on the
conditioning set ``z``, then Fisher-z test the residual correlation. In serially correlated data a
naive ``corr(x, y)`` is badly miscalibrated -- autocorrelation inflates the effective variance of
the estimator, so unrelated series look linked. Conditioning on the lagged parents (tigramite's
*momentary conditional independence*) whitens the residuals and restores calibration. This is the CI
primitive ``chc.discovery`` screens lagged parents with. The method (partial
correlation) is standard; only the autocorrelation-aware *usage* is borrowed -- no tigramite code.

:func:`gcm_test` is the nonlinear, many-pair form: Shah and Peters' generalised covariance measure,
with a polynomial regression on ``z`` and a sign-change randomisation over clusters of rows.

Computed in NumPy float64, not JAX: a p-value threshold is precision-sensitive near ``alpha``, and a
statistical test must not silently depend on the global ``jax_enable_x64`` flag (float32 would flip
borderline edges in discovery). It is not on any differentiated or jitted path.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from chc import _units

_EPS = 1e-12
# A residual whose variance is below this share of its column's has nothing left to test: ``z``
# determines the column, as the current state determines the last one in a discretised ODE.
_DETERMINED = 1e-8
_POWER_QUANTILE = 0.8416212335729143  # the standard normal's 0.8 quantile: ``detectable``'s power
_DRAW_CHUNK = 256  # sign draws per matrix product, so the draws never hold draws x clusters at once


def _own_units(columns: NDArray[np.float64]) -> NDArray[np.float64]:
    """Each column times the power of two nearest the reciprocal of its spread, or 1 where the
    spread is rounding (:mod:`chc._units`), so a constant column is left as it is.

    The rescaling is exact, so a test here reads a column logged in any units as it reads it in
    these: its absolute floors sit at one share of every column, and no square overflows or
    underflows. A column whose spread is already near 1 is left as it is, bit for bit.
    """
    spread, rounding = _units.spread_and_rounding_np(columns)
    return columns * _units.power_of_two_np(np.where(rounding, 1.0, spread))


def _residualize(target: ArrayLike, conditioning: ArrayLike | None) -> tuple[np.ndarray, int]:
    """Residual of ``target`` after linear regression on ``[1, conditioning]``; returns (resid, k).

    ``target`` and each column of ``conditioning`` are read in their own units (:func:`_own_units`),
    so the residual is in ``target``'s, a power of two times the one in the units given. A
    ``target`` whose spread is rounding (:mod:`chc._units`) has a residual of zeros: what is left of
    it is the regression's rounding, at the scale of its level. ``k`` is the number of conditioning
    columns (0 when ``conditioning`` is ``None``) -- the degrees-of-freedom correction for the
    Fisher-z statistic. A ``(n,)`` or ``(n, k)`` conditioning set is accepted; a ``(k, n)`` one is
    transposed to rows-are-samples.
    """
    target = _own_units(np.asarray(target, dtype=np.float64).ravel()[:, None])[:, 0]
    if _units.spread_and_rounding_np(target[:, None])[1][0]:
        target = np.zeros_like(target)
    if conditioning is None:
        return target - target.mean(), 0
    cond = np.atleast_2d(np.asarray(conditioning, dtype=np.float64))
    if cond.shape[0] != target.shape[0]:
        cond = cond.T
    design = np.column_stack([np.ones(target.shape[0]), _own_units(cond)])
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    return target - design @ coeffs, cond.shape[1]


def partial_corr_test(
    x: ArrayLike, y: ArrayLike, z: ArrayLike | None = None
) -> tuple[float, float]:
    """Test ``x ⊥ y | z`` by partial correlation + Fisher-z; returns ``(partial_corr, p_value)``.

    With ``z=None`` this is the plain marginal-correlation test -- the miscalibrated one under
    autocorrelation. Pass the lagged parents as ``z`` for the calibrated MCI variant. ``z`` may be a
    single covariate ``(n,)`` or several stacked as ``(n, k)``. The p-value is two-sided, and the
    same in whatever units each column is logged. A column that moved by rounding alone, such as
    one logged at a single value, reads a correlation of 0 and a p-value of 1.
    """
    residual_x, k = _residualize(x, z)
    residual_y, _ = _residualize(y, z)
    n = residual_x.shape[0]
    denom = math.sqrt(float(np.sum(residual_x**2)) * float(np.sum(residual_y**2))) + _EPS
    rho = float(np.sum(residual_x * residual_y)) / denom
    rho = min(max(rho, -1.0 + _EPS), 1.0 - _EPS)
    dof = max(n - k - 3, 1)  # Fisher-z uses sqrt(n - |z| - 3)
    stat = math.atanh(rho) * math.sqrt(dof)
    p_value = math.erfc(abs(stat) / math.sqrt(2.0))  # = 2 * (1 - Phi(|stat|)), two-sided
    return rho, p_value


@dataclass(frozen=True)
class GcmTest:
    """``x ⊥ y | z`` by the generalised covariance measure, every pair at once. *Experimental.*

    See :func:`gcm_test`.
    """

    # the largest pair's |sum of products| over the root sum of its clusters' squared sums
    statistic: float
    p_value: float  # from sign changes over the clusters; nan when no pair is left to test
    # (p, q): each pair's residual correlation, which is the partial correlation when the regression
    # on ``z`` is the conditional mean; nan where ``z`` determines either side
    partial_correlation: NDArray[np.float64]
    # (p, q): the partial correlation at which the pair alone would reject at ``alpha`` with
    # probability 0.8, on rows like these; inf where too few clusters let no dependence reject. A
    # pass says nothing about dependence below it
    detectable: NDArray[np.float64]
    clusters: int  # how many blocks the randomisation flipped


def gcm_test(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike | None = None,
    *,
    clusters: ArrayLike | None = None,
    degree: int = 2,
    alpha: float = 0.05,
    draws: int = 1999,
    seed: int = 0,
) -> GcmTest:
    """Test ``x ⊥ y | z`` for every pair of a column of ``x`` and one of ``y`` at once, by the
    generalised covariance measure (Shah and Peters 2020). *Experimental.*

    Each column of ``x`` and of ``y`` is regressed, by least squares in the same rows, on every
    monomial of the standardised ``z`` up to ``degree``. A pair's statistic is the sum of its
    residuals' products over the root sum of their squares, summed first within each cluster. The
    test takes the largest over the pairs, so it is one test however many pairs there are, and its
    p-value counts how often flipping the signs of the clusters' sums reaches it (Canay, Romano and
    Shaikh 2017).

    When the test is valid: the clusters' sums must be independent of one another and symmetric
    about zero when ``x ⊥ y | z``. When ``x`` is a logged action and ``z`` what the logger read,
    that holds for rows grouped by period with any ``y`` fixed before the action, however the state
    is autocorrelated. A logger that read ``z`` and drew fresh noise makes each period's sum a
    martingale difference, and the noise it shares across units stays inside the period's sum.
    And the regression on ``z`` must hold the conditional mean of ``x``, or of each column of
    ``y``, as the monomials up to ``degree`` can: otherwise each pair's sum drifts from zero by the
    product of the two misfits, a drift that grows with the rows (Shah and Peters 2020), and on
    enough rows the test rejects a pair that is independent.

    Why neither cross-fitting nor clusters by unit, as a panel usually has. On one geo or five, a
    fold's regression differs from the rest's and five units are too few clusters: over 500 panels
    a case, clusters by unit rejected 27% at 5% on five geos, cross-fitting by unit 16% there and
    by time 14% on one. Noise the units share within a period ties their rows together however many
    units there are; with it, those and the plain GCM over rows rejected 20-88%. This test stayed
    within two points of 5% in every case (``scripts/bench_logger_check.py``). In-sample
    regressions need the class to be small beside the rows, which ``degree`` sets.

    Args:
        x: ``(n,)`` or ``(n, p)``, the columns tested, such as the levers.
        y: ``(n,)`` or ``(n, q)``, the columns each is tested against.
        z: ``(n,)`` or ``(n, d)``, what both are regressed on; ``None`` tests marginal independence.
        clusters: ``(n,)`` labels. Rows that share one are summed before the randomisation, such as
            a panel's periods; by default every row is its own.
        degree: the largest degree of the monomials of ``z``; 2 is the nuisance class
            :func:`chc.dynamics_id.fit_causal_residual` regresses the levers on. At 1 the pairs'
            correlations are :func:`partial_corr_test`'s.
        alpha: the level at which ``detectable`` is read.
        draws: the sign changes drawn; the p-value's resolution is ``1 / (draws + 1)``.
        seed: of the sign changes.

    Returns:
        A :class:`GcmTest`. A column ``z`` determines, such as the lagged state of a discretised
        ODE given the current one, has no residual left to test, nor has one that moved by rounding
        alone, such as a lever the log never moved. Their pairs read nan and stay out of the
        maximum.

    Raises:
        ValueError: on rows of different lengths, a value that is not finite, ``clusters`` of the
            wrong length, fewer rows than twice the regression's terms, or ``degree``, ``alpha`` or
            ``draws`` out of range.
    """
    if degree < 1:
        raise ValueError(f"degree must be at least 1; got {degree}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha}")
    if draws < 1:
        raise ValueError(f"draws must be positive; got {draws}")
    left, right = _own_units(_as_columns(x, "x")), _own_units(_as_columns(y, "y"))
    rows = left.shape[0]
    if right.shape[0] != rows:
        raise ValueError(f"x has {rows} rows and y {right.shape[0]}")
    design = _monomials(None if z is None else _own_units(_as_columns(z, "z")), degree, rows)
    if rows < 2 * design.shape[1]:
        raise ValueError(
            f"{rows} rows for a regression with {design.shape[1]} terms: the in-sample residuals "
            "need at least twice as many rows as terms. Lower degree or condition on fewer columns"
        )
    if clusters is None:
        labels, blocks = np.arange(rows), rows
    else:
        given = np.asarray(clusters)
        if given.shape != (rows,):
            raise ValueError(f"clusters has shape {given.shape}; expected ({rows},)")
        names, labels = np.unique(given, return_inverse=True)
        blocks = len(names)

    residual_x = _residual(left, design)
    residual_y = _residual(right, design)
    # A column of one value has a variance of 0, which no residual's rounding falls below
    _, flat_x = _units.spread_and_rounding_np(left)
    _, flat_y = _units.spread_and_rounding_np(right)
    live = np.outer(
        ~flat_x & (residual_x.var(axis=0) > _DETERMINED * left.var(axis=0)),
        ~flat_y & (residual_y.var(axis=0) > _DETERMINED * right.var(axis=0)),
    )
    size_x = np.sqrt(np.mean(residual_x**2, axis=0))
    size_y = np.sqrt(np.mean(residual_y**2, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = (residual_x.T @ residual_y) / rows / np.outer(size_x, size_y)
    correlation = np.where(live, correlation, np.nan)

    products = (residual_x[:, :, None] * residual_y[:, None, :]).reshape(rows, -1)
    sums = np.zeros((blocks, products.shape[1]))
    np.add.at(sums, labels, products)
    scale = np.sqrt(np.sum(sums**2, axis=0))
    tested = live.ravel() & (scale > 0.0)
    if not tested.any():
        empty = np.full(live.shape, np.nan)
        return GcmTest(math.nan, math.nan, correlation, empty, blocks)

    sums, scale = sums[:, tested], scale[tested]
    statistic = float(np.max(np.abs(sums.sum(axis=0)) / scale))
    rng = np.random.default_rng(seed)
    flipped = np.empty(draws)
    for start in range(0, draws, _DRAW_CHUNK):
        signs = rng.choice((-1.0, 1.0), size=(min(_DRAW_CHUNK, draws - start), blocks))
        flipped[start : start + signs.shape[0]] = np.max(np.abs(signs @ sums) / scale, axis=1)
    p_value = float((1 + np.count_nonzero(flipped >= statistic)) / (draws + 1))

    # The spread is centred: the scale above carries the dependence itself once there is any.
    critical = float(np.quantile(flipped, 1.0 - alpha, method="higher"))
    spread = np.sqrt(np.sum((sums - sums.mean(axis=0)) ** 2, axis=0))
    reach = _reach(critical, blocks)
    detectable = np.full(live.size, np.nan)
    # too few clusters let no dependence reject, however little their sums spread: one cluster's
    # spread is nothing, and inf times nothing is no number
    detectable[tested] = reach * spread / rows if math.isfinite(reach) else math.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        detectable = detectable.reshape(live.shape) / np.outer(size_x, size_y)
    return GcmTest(statistic, p_value, correlation, detectable, blocks)


def _reach(critical: float, clusters: int) -> float:
    """How far, in the clusters' centred spread, a pair's sum must sit from zero to clear
    ``critical`` with probability 0.8.

    The statistic normalises itself: it is ``|S| / sqrt(W + S^2 / C)`` for the clusters' sum ``S``
    and their centred sum of squares ``W``. So it clears ``critical`` exactly where Student's t on
    the ``C`` sums clears ``q = critical sqrt((C - 1) / (C - critical^2))``, and never once
    ``critical^2 >= C``. That t clears ``q`` with probability 0.8 at a noncentrality of about
    ``q + 0.84 sqrt(1 + q^2 / 2(C - 1))``, a normal approximation to the noncentral t. Over 10 to
    40 clusters the test caught a pair at the ``detectable`` this gives 77-82% of the time; reading
    the statistic as a normal instead gave one it caught 63-74% of the time
    (``scripts/bench_logger_check.py detectable``).
    """
    if clusters < 2 or critical**2 >= clusters:
        return math.inf
    q = critical * math.sqrt((clusters - 1) / (clusters - critical**2))
    widening = math.sqrt(1.0 + q**2 / (2 * (clusters - 1)))
    return (q + _POWER_QUANTILE * widening) * math.sqrt(clusters / (clusters - 1))


def _as_columns(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be (n,) or (n, k); got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} holds a value that is not finite")
    return array


def _monomials(z: NDArray[np.float64] | None, degree: int, rows: int) -> NDArray[np.float64]:
    """``[1, z, z (x) z, ...]`` up to ``degree``, over ``z`` standardised so the powers stay
    conditioned in any units. A column of ``z`` whose spread is rounding (:mod:`chc._units`) is
    the intercept's and is dropped: tested for a spread above zero, a column holding one value
    kept its mean's rounding, which divided by its own spread entered as a column of unit size."""
    columns = [np.ones(rows)]
    if z is not None:
        _, rounding = _units.spread_and_rounding_np(z)
        kept = _units.standardised_np(z[:, ~rounding])
        columns.extend(
            np.prod(kept[:, list(combination)], axis=1)
            for power in range(1, degree + 1)
            for combination in itertools.combinations_with_replacement(range(kept.shape[1]), power)
        )
    return np.column_stack(columns)


def _residual(target: NDArray[np.float64], design: NDArray[np.float64]) -> NDArray[np.float64]:
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return target - design @ coefficients
