"""Gradient-boosted surrogate dynamics predict next state (Track A/B baseline; trees extra)."""

import numpy as np
import pytest

pytest.importorskip("lightgbm")

from chc.surrogate import GradientBoostedDynamics

A = np.array([[1.0, 0.1], [-0.2, 0.9]])
B = np.array([[0.0], [0.5]])


def _transitions(n: int = 3000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    u = rng.normal(size=(n, 1))
    x_next = x @ A.T + u @ B.T + 0.05 * rng.normal(size=(n, 2))
    return x, u, x_next


def test_predicts_next_state_on_heldout_transitions() -> None:
    x, u, x_next = _transitions()
    model = GradientBoostedDynamics(backend="lightgbm").fit(x[:2500], u[:2500], x_next[:2500])
    pred = model.predict(x[2500:], u[2500:])
    assert pred.shape == (500, 2)
    assert np.mean((pred - x_next[2500:]) ** 2) < 0.1  # black-box trees fit the dynamics


def test_residual_mode_with_known_physics_is_more_accurate() -> None:
    x, u, x_next = _transitions()
    known = lambda xb, ub: xb @ A.T + ub @ B.T  # noqa: E731 - the true physics; trees learn the residual
    model = GradientBoostedDynamics(known=known).fit(x[:2500], u[:2500], x_next[:2500])
    pred = model.predict(x[2500:], u[2500:])
    assert np.mean((pred - x_next[2500:]) ** 2) < 0.02  # only the small noise is left to model


def test_rollout_shape() -> None:
    x, u, x_next = _transitions(n=1000)
    model = GradientBoostedDynamics().fit(x, u, x_next)
    states = model.rollout(np.zeros(2), np.zeros((10, 1)))
    assert states.shape == (11, 2)


def test_unknown_backend_raises() -> None:
    x, u, x_next = _transitions(n=50)
    with pytest.raises(ValueError, match="unknown backend"):
        GradientBoostedDynamics(backend="xgboost").fit(x, u, x_next)
