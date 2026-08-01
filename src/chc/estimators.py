"""Pluggable causal-effect estimators: one Strategy interface, thin backend adapters.

CHC's identity is control + pessimism + benchmark *on top of* causal effects, not a home-grown
causal library. A controller only needs a scalar interventional slope ``d x_next / d u``; this
module lets that number come from whichever backend is best, behind a single interface
(``CausalEffectEstimator``):

* **built-ins** (``BackdoorOLS``, ``IV2SLS``, ``DoubleML``) — zero-dependency, thin wrappers over
  ``chc.causal``; the default when you do not want extra installs;
* **adapters** over mature libraries (``EconMLDoubleML`` -> EconML) — *lazy-imported*, never a hard
  dependency. They are heavy and hostile to a modern stack: ``econml>=0.16`` with ``pandas>=3``
  *resolves* fine on 3.12 and 3.14 and then fails to **install**, because resolution picks
  ``numba 0.53.1`` (via ``econml -> sparse 0.19 -> numba``) whose build refuses anything outside
  ``>=3.6,<3.10``. Pinning it would therefore break the lockfile rather than the import, so ``chc``
  does not: install it yourself in a compatible environment and pass the class in.

Richer backends may also return a heterogeneous ``cate(x)`` and ``diagnostics``; the controller
ignores what it does not need, so every backend is swappable without touching the control code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from chc.causal import (
    _ols_with_se,
    _polynomial_features,
    _ridge_predict,
    dml_point_and_se,
    estimate_effect_iv,
)
from chc.frames import ColumnData, as_columns

Data = ColumnData


@dataclass(frozen=True)
class EffectEstimate:
    """What a controller needs (``effect``), plus optional heterogeneity and diagnostics.

    ``effect`` is the average interventional slope ``d x_next / d u`` consumed by
    certainty-equivalent or MPC control. ``cate`` maps covariates to a per-unit effect where the
    backend supplies it (EconML); ``diagnostics`` carries backend extras (t-stat, robustness, Qini).
    """

    effect: float
    std_error: float | None = None
    cate: Callable[[Array], Array] | None = None
    diagnostics: Mapping[str, float] = field(default_factory=dict)


@runtime_checkable
class CausalEffectEstimator(Protocol):
    """Estimate the interventional effect of ``treatment`` on ``outcome`` given ``covariates``.

    ``covariates`` is the conditioning set (backdoor adjustment set / DML nuisances / EconML X).
    """

    def estimate(
        self,
        data: Data,
        *,
        treatment: str = "u",
        outcome: str = "x_next",
        covariates: tuple[str, ...] = ("x", "z"),
    ) -> EffectEstimate: ...


def _columns(data: Data) -> dict[str, Array]:
    """Any accepted frame as JAX arrays -- the estimators' dtype, which follows the x64 flag."""
    return {name: jnp.asarray(column) for name, column in as_columns(data).items()}


def _alias(data: Data, treatment: str, outcome: str) -> dict[str, Array]:
    """View of ``data`` with ``treatment``/``outcome`` also exposed as builtin ``u``/``x_next``."""
    cols = _columns(data)
    return {**cols, "u": cols[treatment], "x_next": cols[outcome]}


@dataclass(frozen=True)
class BackdoorOLS:
    """Backdoor adjustment: OLS of ``outcome`` on ``[treatment, *covariates]`` and an intercept."""

    def estimate(
        self,
        data: Data,
        *,
        treatment: str = "u",
        outcome: str = "x_next",
        covariates: tuple[str, ...] = ("x", "z"),
    ) -> EffectEstimate:
        columns = _columns(data)
        design = [columns[treatment]] + [columns[c] for c in covariates]
        beta, se, _ = _ols_with_se(jnp.stack(design, axis=1), columns[outcome])
        effect, stderr = float(beta[0]), float(se[0])  # treatment is column 0
        return EffectEstimate(effect, stderr, diagnostics={"t_stat": effect / stderr})


@dataclass(frozen=True)
class IV2SLS:
    """Two-stage least squares using ``instrument`` for a latent confounder (built-in)."""

    instrument: str = "w"

    def estimate(
        self,
        data: Data,
        *,
        treatment: str = "u",
        outcome: str = "x_next",
        covariates: tuple[str, ...] = ("x", "z"),
    ) -> EffectEstimate:
        effect = float(
            estimate_effect_iv(_alias(data, treatment, outcome), instrument=self.instrument)
        )
        return EffectEstimate(effect)


@dataclass(frozen=True)
class DoubleML:
    """Cross-fitted, Neyman-orthogonal Double ML with polynomial nuisances (built-in)."""

    degree: int = 2
    folds: int = 5
    ridge: float = 1.0

    def estimate(
        self,
        data: Data,
        *,
        treatment: str = "u",
        outcome: str = "x_next",
        covariates: tuple[str, ...] = ("x", "z"),
    ) -> EffectEstimate:
        effect, se = dml_point_and_se(
            _alias(data, treatment, outcome),
            covariates=covariates,
            degree=self.degree,
            folds=self.folds,
            ridge=self.ridge,
        )
        lo, hi = effect - 1.96 * se, effect + 1.96 * se
        t_stat = effect / se if se > 0.0 else float("inf")
        return EffectEstimate(
            effect, std_error=se, diagnostics={"t_stat": t_stat, "ci95_low": lo, "ci95_high": hi}
        )


@dataclass(frozen=True)
class RLearner:
    """Nie-Wager R-learner for heterogeneous effects ``tau(x)`` (built-in; continuous treatment).

    Cross-fits nuisances ``m(x)=E[outcome|x]`` and ``e(x)=E[treatment|x]`` (polynomial-ridge), then
    fits a linear-in-features ``tau`` minimising the R-loss ``sum (y_res - tau(x)*t_res)^2`` on the
    residualised data -- Neyman-orthogonal, so it recovers the CATE even under *nonlinear*
    confounding where a naive treatment-on-outcome regression is biased. Returns the average effect
    as ``effect`` and ``tau(x)`` as ``cate`` (call it on the covariate matrix). ``cate_degree=1`` is
    linear heterogeneity; raise it for nonlinear ``tau``.
    """

    degree: int = 3  # nuisance flexibility
    cate_degree: int = 1  # tau(x) feature degree
    folds: int = 2
    ridge: float = 1e-2
    seed: int = 0

    def estimate(
        self,
        data: Data,
        *,
        treatment: str = "u",
        outcome: str = "x_next",
        covariates: tuple[str, ...] = ("x", "z"),
    ) -> EffectEstimate:
        columns = _columns(data)
        y, t = columns[outcome], columns[treatment]
        covs = jnp.stack([columns[c] for c in covariates], axis=1)
        n = y.shape[0]
        chunks = jnp.array_split(jax.random.permutation(jax.random.key(self.seed), n), self.folds)
        y_res, t_res = jnp.zeros(n), jnp.zeros(n)
        for k in range(self.folds):  # cross-fit the nuisances out of fold k
            test = chunks[k]
            train = jnp.concatenate([chunks[j] for j in range(self.folds) if j != k])
            phi_tr = _polynomial_features(covs[train], self.degree)
            phi_te = _polynomial_features(covs[test], self.degree)
            m_hat = _ridge_predict(phi_tr, y[train], phi_te, self.ridge)
            e_hat = _ridge_predict(phi_tr, t[train], phi_te, self.ridge)
            y_res = y_res.at[test].set(y[test] - m_hat)
            t_res = t_res.at[test].set(t[test] - e_hat)

        features = _polynomial_features(covs, self.cate_degree)
        design = features * t_res[:, None]  # R-loss: regress y_res on tau-features scaled by t_res
        theta = jnp.linalg.solve(
            design.T @ design + self.ridge * jnp.eye(design.shape[1]), design.T @ y_res
        )
        return EffectEstimate(
            effect=float(jnp.mean(features @ theta)),
            cate=lambda covs_q: _polynomial_features(jnp.asarray(covs_q), self.cate_degree) @ theta,
        )


_ECONML_HINT = (
    "EconMLDoubleML requires the 'econml' package, which is NOT a chc dependency "
    "(it does not resolve on Python 3.12 + pandas 3). Install it in a compatible env: "
    "pip install econml."
)


@dataclass(frozen=True)
class EconMLDoubleML:
    """Adapter over ``econml.dml.LinearDML`` (lazy import; requires econml installed).

    Pass a pre-constructed EconML estimator via ``model`` (e.g. ``CausalForestDML(...)``) to use any
    EconML DML variant; the default is ``LinearDML``. Returns the ATE as ``effect`` and exposes
    EconML's heterogeneous ``effect(X)`` as ``cate``.
    """

    model: Any = None

    def estimate(
        self,
        data: Data,
        *,
        treatment: str = "u",
        outcome: str = "x_next",
        covariates: tuple[str, ...] = ("x", "z"),
    ) -> EffectEstimate:
        try:
            from econml.dml import LinearDML
        except ImportError as exc:  # pragma: no cover - exercised only without econml
            raise ImportError(_ECONML_HINT) from exc
        columns = as_columns(data)
        y = np.asarray(columns[outcome])
        t = np.asarray(columns[treatment])
        x = np.column_stack([np.asarray(columns[c]) for c in covariates])
        est = self.model if self.model is not None else LinearDML(random_state=0)
        est.fit(y, t, X=x)
        return EffectEstimate(
            effect=float(np.mean(est.effect(x))),
            cate=lambda xq: jnp.asarray(est.effect(np.asarray(xq))),
        )


_DOWHY_HINT = (
    "DoWhyEstimator requires the 'dowhy' package, which is NOT a chc dependency. "
    "Install it in a compatible environment: pip install dowhy."
)


@dataclass(frozen=True)
class DoWhyEstimator:
    """Adapter over DoWhy's identify -> estimate workflow (lazy import; requires dowhy installed).

    Builds a ``CausalModel`` with ``common_causes = covariates``, identifies the estimand, and
    estimates it (backdoor linear regression by default). Pass another ``method_name`` for a
    different DoWhy estimator (e.g. propensity-score matching / weighting).
    """

    method_name: str = "backdoor.linear_regression"

    def estimate(
        self,
        data: Data,
        *,
        treatment: str = "u",
        outcome: str = "x_next",
        covariates: tuple[str, ...] = ("x", "z"),
    ) -> EffectEstimate:
        try:
            import pandas as pd
            from dowhy import CausalModel
        except ImportError as exc:  # pragma: no cover - exercised only without dowhy
            raise ImportError(_DOWHY_HINT) from exc
        columns = as_columns(data)
        frame = pd.DataFrame(
            {
                treatment: np.asarray(columns[treatment]),
                outcome: np.asarray(columns[outcome]),
                **{c: np.asarray(columns[c]) for c in covariates},
            }
        )
        model = CausalModel(
            data=frame, treatment=treatment, outcome=outcome, common_causes=list(covariates)
        )
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(estimand, method_name=self.method_name)
        return EffectEstimate(effect=float(estimate.value))
