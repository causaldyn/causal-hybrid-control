"""Gradient-boosted surrogate dynamics: the tabular baseline for Track A/B prediction.

Trees (LightGBM/CatBoost) are the strong tabular competitor the hybrid model must be measured
against on one-step (Track A) and rollout (Track B) error. They are **not differentiable**, so this
is a forward predictor / simulator -- not a drop-in JAX residual for adjoint control; it drives
gradient-free control baselines and the prediction leaderboard. With a ``known`` physics map it
becomes a *residual* learner (physics + tree correction); without one it is a pure black-box model.

Optional: requires the ``trees`` extra (``pip install 'causal-hybrid-control[trees]'``); the tree
library is imported lazily so the core stays pure-JAX.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_TREES_HINT = (
    "GradientBoostedDynamics needs the 'trees' extra: pip install 'causal-hybrid-control[trees]'."
)

KnownMap = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _make_regressor(backend: str, params: dict[str, Any]) -> Any:
    """Construct a single-output tree regressor for the chosen backend (lazy import)."""
    if backend == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError(_TREES_HINT) from exc
        return LGBMRegressor(**{"verbosity": -1, "n_estimators": 300, **params})
    if backend == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError(_TREES_HINT) from exc
        return CatBoostRegressor(**{"verbose": False, "iterations": 300, **params})
    raise ValueError(f"unknown backend {backend!r}; use 'lightgbm' or 'catboost'")


@dataclass
class GradientBoostedDynamics:
    """Fit ``x_next ~ trees(x, u)`` with one regressor per state dimension (Track A/B baseline).

    If ``known`` is given, the trees learn the *residual* ``x_next - known(x, u)``; otherwise they
    model the full next state (black box).
    """

    backend: str = "lightgbm"
    params: dict[str, Any] = field(default_factory=dict)
    known: KnownMap | None = None
    _models: list[Any] = field(default_factory=list, init=False, repr=False)

    def fit(self, x: np.ndarray, u: np.ndarray, x_next: np.ndarray) -> GradientBoostedDynamics:
        x, u, y = np.asarray(x, float), np.asarray(u, float), np.asarray(x_next, float)
        target = y - self.known(x, u) if self.known is not None else y
        feats = np.column_stack([x, u])
        self._models = []
        for j in range(target.shape[1]):
            model = _make_regressor(self.backend, self.params)
            model.fit(feats, target[:, j])
            self._models.append(model)
        return self

    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        if not self._models:
            raise RuntimeError("call fit() before predict()")
        x, u = np.atleast_2d(np.asarray(x, float)), np.atleast_2d(np.asarray(u, float))
        feats = np.column_stack([x, u])
        with warnings.catch_warnings():
            # lightgbm's sklearn wrapper stores feature names at fit and warns on unnamed predict
            # input; documented benign noise, suppressed at the adapter boundary (not in tests).
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            pred = np.column_stack([model.predict(feats) for model in self._models])
        return pred + self.known(x, u) if self.known is not None else pred

    def rollout(self, x0: np.ndarray, us: np.ndarray) -> np.ndarray:
        """Autoregressive rollout: ``x0`` (d_x,), ``us`` (H, d_u) -> states (H+1, d_x)."""
        x = np.asarray(x0, float)
        us = np.asarray(us, float)
        states = [x]
        for t in range(us.shape[0]):
            x = self.predict(x[None, :], us[t][None, :])[0]
            states.append(x)
        return np.stack(states)
