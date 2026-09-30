"""Synthetic control and augmented synthetic control for a single treated unit.

The synthetic control method (SCM, Abadie-Diamond-Hainmueller) builds a counterfactual for one
treated unit as a convex combination of donor units matched on the pre-treatment outcome path; the
treatment effect is the treated-minus-synthetic gap in the post period. Its weakness is the simplex
constraint: when the treated unit lies outside the donors' convex hull no weighting balances the
pre-period, so the estimate is biased.

Augmented SCM (ASCM, Ben-Michael-Feller-Rothstein) corrects exactly this: it keeps the SCM donor mix
but adds a ridge outcome-model correction for the residual pre-period imbalance, which is allowed to
extrapolate. When SCM already balances, the correction is ~0 and ASCM reduces to SCM.

A statistical estimator, so NumPy float64 throughout (like :mod:`chc.did`), independent of the JAX
``x64`` flag -- which must not change an estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls

Outcomes = NDArray[np.float64]
Vector = NDArray[np.float64]


@dataclass(frozen=True)
class SyntheticControlResult:
    """A fitted synthetic control: post-period effect path, donor mix, and pre-period fit."""

    att: Vector  # (T1,) treated-minus-synthetic gap for each post-treatment period
    overall: float  # mean post-treatment ATT
    weights: Vector  # (J,) donor weights, aligned with the units minus the treated one
    pre_rmspe: float  # pre-period root-mean-squared prediction error (smaller = better match)


def _scm_weights(donor_pre: Outcomes, treated_pre: Vector, steps: int) -> Vector:
    """Simplex weights minimising ``||treated_pre - donor_pre.T @ w||``, exactly.

    The synthetic control is the point of the donors' hull nearest the treated unit, so with
    ``Z = donor_pre.T - treated_pre`` its weights give the point of the hull of ``Z``'s columns
    nearest the origin. For ``u >= 0`` minimising ``||Z u||^2 + (1'u - 1)^2``, ``u / 1'u`` is that
    point: writing ``u = s w`` with ``w`` on the simplex, the best ``s`` leaves ``a / (1 + a)``
    with ``a = ||Z w||^2``, which grows with ``a``. That is one non-negative least squares, which
    Lawson and Hanson's active set solves in finitely many steps, at most ``steps``. ``Z`` is
    scaled first: the weights do not depend on its scale, and the appended row's weight does.
    """
    z = donor_pre.T - treated_pre[:, None]  # (T0, J)
    scale = max(float(np.sqrt(np.mean(z**2))), np.finfo(np.float64).tiny)
    design = np.vstack([z / scale, np.ones(z.shape[1])])
    target = np.zeros(design.shape[0])
    target[-1] = 1.0
    u, _ = nnls(design, target, maxiter=steps)
    return u / u.sum()


def _split(
    outcomes: Outcomes, treated_unit: int, n_pre: int
) -> tuple[Outcomes, Outcomes, Vector, Vector]:
    outcomes = np.asarray(outcomes, dtype=np.float64)
    n_units, n_periods = outcomes.shape
    if not 0 <= treated_unit < n_units:
        msg = f"treated_unit {treated_unit} out of range for {n_units} units"
        raise ValueError(msg)
    if not 1 <= n_pre < n_periods:
        msg = f"n_pre must be in [1, {n_periods - 1}], got {n_pre}"
        raise ValueError(msg)
    donors = np.delete(outcomes, treated_unit, axis=0)
    if donors.shape[0] < 1:
        msg = "need at least one donor unit"
        raise ValueError(msg)
    treated = outcomes[treated_unit]
    return donors[:, :n_pre], donors[:, n_pre:], treated[:n_pre], treated[n_pre:]


def synthetic_control(
    outcomes: Outcomes, treated_unit: int, n_pre: int, *, steps: int = 5000
) -> SyntheticControlResult:
    """Classic simplex synthetic control for one treated unit against the remaining donor units.

    ``outcomes`` is ``(N, T)``; treatment starts at period ``n_pre`` (so periods ``0..n_pre-1`` are
    pre-treatment). Donor weights lie on the probability simplex; ``att[k]`` is the treated-minus-
    synthetic gap in post period ``k``. The weights are the exact optimum, whatever the outcomes'
    units; ``steps`` caps the solver's iterations, and running out raises ``RuntimeError``.
    """
    donor_pre, donor_post, treated_pre, treated_post = _split(outcomes, treated_unit, n_pre)
    w = _scm_weights(donor_pre, treated_pre, steps)
    pre_rmspe = float(np.sqrt(np.mean((treated_pre - donor_pre.T @ w) ** 2)))
    att = treated_post - donor_post.T @ w
    return SyntheticControlResult(att, float(att.mean()), w, pre_rmspe)


def augmented_synthetic_control(
    outcomes: Outcomes, treated_unit: int, n_pre: int, *, ridge: float = 1.0, steps: int = 5000
) -> SyntheticControlResult:
    """Ridge-augmented synthetic control (Ben-Michael-Feller-Rothstein).

    Starts from the SCM donor weights, then de-biases each post period by the residual pre-period
    imbalance passed through a ridge outcome model fit on the donors:
    ``Y_1(0)_t = w' Y_post_t + (treated_pre - w' donor_pre)' theta_t`` with
    ``theta_t = (Z'Z + ridge*I)^{-1} Z' Y_post_t`` and ``Z`` the donor pre-period matrix. The
    correction vanishes when SCM already balances the pre-period; ``ridge`` controls how far the
    outcome model may extrapolate. ``weights`` are the (interpretable) SCM donor weights; the
    augmentation is an additive outcome correction, not folded into them.
    """
    donor_pre, donor_post, treated_pre, treated_post = _split(outcomes, treated_unit, n_pre)
    w = _scm_weights(donor_pre, treated_pre, steps)
    imbalance = treated_pre - donor_pre.T @ w  # (T0,) pre-period residual SCM cannot balance
    pre_rmspe = float(np.sqrt(np.mean(imbalance**2)))
    n_pre_periods = donor_pre.shape[1]
    gram = donor_pre.T @ donor_pre + ridge * np.eye(n_pre_periods)  # (T0, T0)
    theta = np.linalg.solve(gram, donor_pre.T @ donor_post)  # (T0,T1): one ridge model per period
    counterfactual = donor_post.T @ w + theta.T @ imbalance  # SCM + ridge bias correction
    att = treated_post - counterfactual
    return SyntheticControlResult(att, float(att.mean()), w, pre_rmspe)
