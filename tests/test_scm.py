"""Augmented SCM de-biases the synthetic control when the treated unit is outside the donor hull."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from chc.scm import augmented_synthetic_control, synthetic_control

TAU = 2.0  # post-treatment effect added to the treated unit
N_PRE = 25
Loading = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def _factor_panel(seed: int, treated_loading: Loading) -> np.ndarray:
    """Latent-factor panel; unit 0 is treated (effect TAU post), the rest are donors."""
    rng = np.random.default_rng(seed)
    n_donors, n_post, rank = 30, 10, 3
    n_periods = N_PRE + n_post
    factors = rng.normal(0.0, 1.0, (n_periods, rank))
    donor_loadings = rng.normal(0.0, 1.0, (n_donors, rank))
    treated = treated_loading(donor_loadings, rng) @ factors.T + rng.normal(0.0, 0.1, n_periods)
    treated = treated.copy()
    treated[N_PRE:] += TAU
    donors = donor_loadings @ factors.T + rng.normal(0.0, 0.1, (n_donors, n_periods))
    return np.vstack([treated, donors])


def _in_hull(loadings: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    weights = rng.uniform(0.0, 1.0, loadings.shape[0])
    return (weights / weights.sum()) @ loadings  # a convex combo is inside the donor hull


def _outside_hull(loadings: np.ndarray, _: np.random.Generator) -> np.ndarray:
    center = loadings.mean(axis=0)
    return center + 2.5 * (loadings[0] - center)  # extreme extrapolation: outside the convex hull


def test_synthetic_control_recovers_effect_when_treated_in_hull() -> None:
    outcomes = _factor_panel(seed=0, treated_loading=_in_hull)
    result = synthetic_control(outcomes, treated_unit=0, n_pre=N_PRE)
    assert result.pre_rmspe < 0.2  # the donor mix balances the pre-period
    assert abs(result.overall - TAU) < 0.2  # effect recovered


def test_augmented_scm_debiases_when_treated_outside_hull() -> None:
    outcomes = _factor_panel(seed=1, treated_loading=_outside_hull)
    scm = synthetic_control(outcomes, treated_unit=0, n_pre=N_PRE)
    ascm = augmented_synthetic_control(outcomes, treated_unit=0, n_pre=N_PRE)
    assert scm.pre_rmspe > 0.5  # the simplex cannot balance a treated unit outside the hull
    assert abs(scm.overall - TAU) > 0.2  # so SCM is biased
    assert abs(ascm.overall - TAU) < 0.2  # the ridge augmentation removes most of the bias
    assert abs(ascm.overall - TAU) < abs(scm.overall - TAU) / 3.0  # >=3x bias reduction


def test_augmented_reduces_to_scm_when_pre_period_is_balanced() -> None:
    outcomes = _factor_panel(seed=2, treated_loading=_in_hull)
    scm = synthetic_control(outcomes, treated_unit=0, n_pre=N_PRE)
    ascm = augmented_synthetic_control(outcomes, treated_unit=0, n_pre=N_PRE)
    assert abs(ascm.overall - scm.overall) < 0.1  # correction ~0 when SCM already balances


def test_scm_weights_form_a_valid_simplex() -> None:
    outcomes = _factor_panel(seed=3, treated_loading=_in_hull)
    weights = synthetic_control(outcomes, treated_unit=0, n_pre=N_PRE).weights
    assert weights.shape == (30,)  # one weight per donor
    assert (weights >= -1e-9).all()  # non-negative
    assert abs(float(weights.sum()) - 1.0) < 1e-6  # sums to one


def test_invalid_arguments_raise() -> None:
    outcomes = _factor_panel(seed=4, treated_loading=_in_hull)
    with pytest.raises(ValueError, match="treated_unit"):
        synthetic_control(outcomes, treated_unit=99, n_pre=N_PRE)
    with pytest.raises(ValueError, match="n_pre"):
        synthetic_control(outcomes, treated_unit=0, n_pre=0)
