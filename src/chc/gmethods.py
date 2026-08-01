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
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chc.frames import ColumnData, as_columns

Data = ColumnData
Vector = NDArray[np.float64]


def _float64_columns(data: Data) -> dict[str, Vector]:
    """Any accepted frame as float64 columns -- this module's contract, whatever the x64 flag."""
    return {name: np.asarray(column, dtype=np.float64) for name, column in as_columns(data).items()}


def _ridge_fit(design: NDArray[np.float64], target: Vector, ridge: float) -> Vector:
    augmented = np.column_stack([np.ones(design.shape[0]), design])
    gram = augmented.T @ augmented + ridge * np.eye(augmented.shape[1])
    return np.linalg.solve(gram, augmented.T @ target)


def _ridge_predict(beta: Vector, design: NDArray[np.float64]) -> Vector:
    return np.column_stack([np.ones(design.shape[0]), design]) @ beta


def _folds(n: int, k: int, seed: int) -> list[NDArray[np.intp]]:
    order = np.random.default_rng(seed).permutation(n)
    return [order[i::k] for i in range(k)]  # deterministic k-way split of a shuffled index


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
    than conditioning on it. The nuisance regressions are cross-fitted.
    """
    if not len(treatments) == len(confounders) == len(regime) == len(baseline):
        msg = "treatments, confounders, regime, and baseline must share one length (the horizon)"
        raise ValueError(msg)
    horizon = len(treatments)
    columns = _float64_columns(data)
    n = int(columns[outcome].shape[0])
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
    """
    columns = _float64_columns(data)
    treat = [columns[a] for a in treatments]
    covariates = [columns[c] for block in confounders for c in block]
    beta = _ridge_fit(np.column_stack([*treat, *covariates]), columns[outcome], 1e-6)
    return float(sum(beta[1 + i] for i in range(len(treatments))))  # skip the intercept
