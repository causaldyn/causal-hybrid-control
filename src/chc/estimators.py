"""Pluggable causal-effect estimators: one Strategy interface, thin backend adapters.

CHC's identity is control + pessimism + benchmark *on top of* causal effects, not a home-grown
causal library. A controller only needs a scalar interventional slope ``d x_next / d u``; this
module lets that number come from whichever backend is best, behind a single interface
(``CausalEffectEstimator``):

* **built-ins** (``BackdoorOLS``, ``IV2SLS``, ``DoubleML``) — zero-dependency, thin wrappers over
  ``chc.causal``; the default when you do not want extra installs;
* **adapters** over mature libraries (``EconMLDoubleML`` -> EconML) — *lazy-imported*, never a hard
  dependency. They are heavy and resolver-hostile in a modern stack (EconML 0.16 will not resolve
  on Python 3.12 + pandas 3 -- it pulls numba 0.53, capped at <3.10), so ``chc`` never pins them:
  install them yourself in a compatible environment and pass the class in.

Richer backends may also return a heterogeneous ``cate(x)`` and ``diagnostics``; the controller
ignores what it does not need, so every backend is swappable without touching the control code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import jax.numpy as jnp
import numpy as np
from jax import Array

from chc.causal import _ols_with_se, estimate_effect_dml, estimate_effect_iv

Data = Mapping[str, Array]


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


def _alias(data: Data, treatment: str, outcome: str) -> dict[str, Array]:
    """View of ``data`` with ``treatment``/``outcome`` also exposed as builtin ``u``/``x_next``."""
    return {**data, "u": data[treatment], "x_next": data[outcome]}


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
        cols = [jnp.asarray(data[treatment])] + [jnp.asarray(data[c]) for c in covariates]
        beta, se, _ = _ols_with_se(jnp.stack(cols, axis=1), jnp.asarray(data[outcome]))
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
        effect = float(
            estimate_effect_dml(
                _alias(data, treatment, outcome),
                covariates=covariates,
                degree=self.degree,
                folds=self.folds,
                ridge=self.ridge,
            )
        )
        return EffectEstimate(effect)


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
        y = np.asarray(data[outcome])
        t = np.asarray(data[treatment])
        x = np.column_stack([np.asarray(data[c]) for c in covariates])
        est = self.model if self.model is not None else LinearDML(random_state=0)
        est.fit(y, t, X=x)
        return EffectEstimate(
            effect=float(np.mean(est.effect(x))),
            cate=lambda xq: jnp.asarray(est.effect(np.asarray(xq))),
        )
