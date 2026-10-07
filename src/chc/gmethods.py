"""Robins' g-methods for time-varying treatment under time-varying confounding.

When a confounder ``L_t`` is itself affected by past treatment (``A_{t-1} -> L_t -> Y`` and
``L_t -> A_t``), ordinary regression adjustment is biased both ways: conditioning on ``L_t`` blocks
the ``A_{t-1} -> L_t -> Y`` path (the earlier treatment's total effect is lost) and opens collider
bias. The g-formula instead *standardises* over the confounder's post-treatment distribution. This
implements the iterated-conditional-expectation (sequential-regression) g-computation estimator
(Robins 1986; Bang & Robins 2005) of a treatment *regime* ``a = (a_0, ..., a_T)`` on the final
outcome, with K-fold cross-fitting of the nuisance regressions (the Double-ML honesty).

A statistical estimator, NumPy float64 throughout (like :mod:`chc.did` / :mod:`chc.scm`) -- x64-flag
independent. Continuous or binary treatments; ridge nuisances.

Only the columns a call names are read, each as float64 where it holds real numbers: a boolean, an
integer or a floating dtype, or ``bool``, ``int``, ``float``, ``Decimal`` or ``Fraction`` values. A
column of text, dates, times, durations, complex numbers or other objects is refused, and so is a
number finite in its own type that is an infinity in float64, naming the column and the row; a
float64 cast read the text ``"0.5"`` as 0.5 and a date as its count of days. A nan is read as it
is, and the effect reads nan.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from chc.frames import ColumnData, _numbers, _refuse_shared_names, as_columns

Data = ColumnData
Vector = NDArray[np.float64]


def _float64_columns(data: Data, names: Iterable[str]) -> dict[str, Vector]:
    """The columns ``names`` of any accepted frame as float64 -- this module's contract, whatever
    the x64 flag -- each read only where it holds real numbers (:func:`chc.frames._numbers`). A
    column the call does not name is not read, so a label column of text is no obstacle."""
    columns = as_columns(data)
    return {name: _numbers(columns[name], name) for name in dict.fromkeys(names)}


def _ridge_scales(columns: NDArray[np.float64]) -> tuple[Vector, NDArray[np.bool_]]:
    """Each column's variance about its mean, the scale of its ridge beside a free intercept, and
    which columns count as constant. A column whose spread about its mean is at most 64 epsilons
    of its dtype times its root mean square counts as constant: its centred values are zeroed, so
    its coefficient is exactly zero, and its ridge is scaled by 1, which keeps the solve regular
    and moves nothing else. The spread is the corrected two-pass variance of Chan, Golub and
    LeVeque (1983), the deviations' own mean taken off their mean square: a constant column
    deviates from its computed mean by that mean's rounding alone, which squared would read as a
    spread past 1e5 rows."""
    size = np.mean(columns**2, axis=0)
    deviation = columns - np.mean(columns, axis=0)
    variance = np.mean(deviation**2, axis=0) - np.mean(deviation, axis=0) ** 2
    rounding = (64.0 * np.finfo(size.dtype).eps) ** 2
    constant = variance <= rounding * size
    return np.where(constant, 1.0, variance), constant


def _ridge_fit(design: NDArray[np.float64], target: Vector, ridge: float) -> Vector:
    """Ridge regression on a free intercept and ``design``, the ridge on each column's coefficient
    scaled by the column's variance, so that the fit reads the same in any units of a column and
    from any origin. Added as a constant, the ridge outweighed a treatment logged in millionths of
    its units and set its coefficient near zero; on the intercept, or scaled by a mean square, it
    would shrink the coefficient of a column that sits far from zero. The slopes are solved about
    the columns' means and the intercept recovered from them: in the Gram of the raw columns a
    column's spread drowns in the rounding of its offset, and one constant but for 1e-11 of its
    size left it singular. A column that counts as constant is zeroed about its mean, so its
    coefficient is exactly zero and the intercept carries its level."""
    mean = np.mean(design, axis=0)
    level = np.mean(target)
    scales, constant = _ridge_scales(design)
    centred = np.where(constant, 0.0, design - mean)
    gram = centred.T @ centred + np.diag(ridge * scales)
    slopes = np.linalg.solve(gram, centred.T @ (target - level))
    return np.concatenate([[level - mean @ slopes], slopes])


def _ridge_predict(beta: Vector, design: NDArray[np.float64]) -> Vector:
    return np.column_stack([np.ones(design.shape[0]), design]) @ beta


def _folds(n: int, k: int, seed: int) -> list[NDArray[np.intp]]:
    order = np.random.default_rng(seed).permutation(n)
    return [order[i::k] for i in range(k)]  # deterministic k-way split of a shuffled index


def _pooled_confounders(
    treatments: tuple[str, ...], confounders: tuple[tuple[str, ...], ...], outcome: str
) -> tuple[str, ...]:
    """Every confounder any set names, once each, after refusing an outcome named among the
    treatments or the confounders and a treatment named twice. A confounder may sit in several
    sets: a baseline is measured before every treatment."""
    pooled = tuple(dict.fromkeys(name for block in confounders for name in block))
    _refuse_shared_names(
        {
            "the outcome": (outcome,),
            "a treatment": treatments,
            "a confounder": tuple(name for name in pooled if name not in treatments),
        }
    )
    return pooled


def sequential_g_formula(
    data: Data,
    *,
    treatments: tuple[str, ...],
    confounders: tuple[tuple[str, ...], ...],
    outcome: str,
    regime: tuple[float, ...],
    baseline: tuple[float, ...],
    ridge: float = 1e-3,
    folds: int = 2,
    seed: int = 0,
) -> float:
    """Effect ``E[Y^regime] - E[Y^baseline]`` of a treatment regime under time-varying confounding.

    ``treatments`` is time-ordered and ``confounders[t]`` names the covariates measured before
    treatment ``t`` (same length as ``treatments``). ``regime`` / ``baseline`` set each treatment's
    value under the two interventions. Iterated conditional expectation: regress the running
    pseudo-outcome on the history through time ``t``, set ``A_t`` to the regime value, then recurse
    to ``t = 0``, standardising over each confounder's realised post-treatment distribution rather
    than conditioning on it. The nuisance regressions are cross-fitted over ``folds`` folds, each
    row predicted by the fit on the others, so there are at least two and at most one a row.
    ``ridge`` is their Tikhonov term, on each coefficient but the intercept's in units of its
    column's variance, so the effect reads the same in any units and from any origin of the
    treatments and the confounders, the regime and the baseline taken in the treatments' own.

    A treatment may be a confounder of a later one, measured before it; one set at its own step
    and also among the confounders measured before that step would be held at its logged value
    where it is set.

    Raises:
        ValueError: when the treatments, confounders, regime and baseline differ in length,
            ``folds`` is not a whole number from 2 to the rows, the outcome is named among the
            treatments or the confounders, a treatment is named twice, or a treatment is among
            the confounders of its own step or an earlier one; or when a column it reads does
            not hold real numbers, or is a masked array that masks a cell, naming the column and
            the row.
    """
    if not len(treatments) == len(confounders) == len(regime) == len(baseline):
        msg = "treatments, confounders, regime, and baseline must share one length (the horizon)"
        raise ValueError(msg)
    pooled = _pooled_confounders(treatments, confounders, outcome)
    for step, treatment in enumerate(treatments):
        for earlier in range(step + 1):
            if treatment in confounders[earlier]:
                msg = (
                    f"the treatment {treatment!r} is set at step {step} and is a confounder "
                    f"measured before step {earlier}, so it would be held at its logged value "
                    "where it is set; a treatment may confound only the later ones"
                )
                raise ValueError(msg)
    horizon = len(treatments)
    columns = _float64_columns(data, (outcome, *treatments, *pooled))
    n = int(columns[outcome].shape[0])
    if not isinstance(folds, int | np.integer) or not 2 <= folds <= n:
        msg = (
            f"folds={folds!r}: cross-fitting predicts each row from a fit on the other folds, so "
            f"it needs a whole number of folds from 2 to the rows ({n})"
        )
        raise ValueError(msg)
    fold_indices = _folds(n, folds, seed)

    def g_value(values: tuple[float, ...]) -> float:
        pseudo = columns[outcome].copy()
        for t in range(horizon - 1, -1, -1):
            treat = [columns[treatments[j]] for j in range(t + 1)]
            covariates = [columns[c] for j in range(t + 1) for c in confounders[j]]
            design = np.column_stack([*treat, *covariates])
            intervened = design.copy()
            intervened[:, t] = values[t]  # set A_t to the regime value; earlier treatments observed
            next_pseudo = np.empty(n)
            for held_out in fold_indices:
                train = np.setdiff1d(np.arange(n), held_out, assume_unique=False)
                beta = _ridge_fit(design[train], pseudo[train], ridge)
                next_pseudo[held_out] = _ridge_predict(beta, intervened[held_out])
            pseudo = next_pseudo
        return float(pseudo.mean())

    return g_value(regime) - g_value(baseline)


def naive_pooled_effect(
    data: Data,
    *,
    treatments: tuple[str, ...],
    confounders: tuple[tuple[str, ...], ...],
    outcome: str,
) -> float:
    """The biased baseline: one pooled regression of ``outcome`` on all treatments and confounders,
    summing the treatment coefficients. Wrong under time-varying confounding -- it conditions on the
    post-treatment confounders that the g-formula standardises over.

    Raises:
        ValueError: when the outcome is named among the treatments or the confounders, a treatment
            is named twice, or a treatment is among the confounders: pooled, its coefficient
            would be split with its own copy; or when a column it reads does not hold real
            numbers, or is a masked array that masks a cell, naming the column and the row.
    """
    pooled = _pooled_confounders(treatments, confounders, outcome)
    clash = sorted(set(pooled) & set(treatments))
    if clash:
        msg = (
            f"the treatments {clash} are also confounders; one pooled regression would split each "
            "one's coefficient with its own copy, and the sum would miss the copy's share"
        )
        raise ValueError(msg)
    columns = _float64_columns(data, (outcome, *treatments, *pooled))
    treat = [columns[a] for a in treatments]
    covariates = [columns[c] for block in confounders for c in block]
    beta = _ridge_fit(np.column_stack([*treat, *covariates]), columns[outcome], 1e-6)
    return float(sum(beta[1 + i] for i in range(len(treatments))))  # skip the intercept
