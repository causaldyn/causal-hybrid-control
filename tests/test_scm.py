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


def _many_donors(seed: int, n_donors: int = 80, n_pre: int = 30) -> np.ndarray:
    """More donors than pre-periods, on trending factors; unit 0 is the treated one."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n_pre + 5, 3)).cumsum(axis=0)
    loadings = rng.uniform(0.0, 1.0, (n_donors + 1, 3))
    noise = rng.normal(0.0, 0.5, (n_donors + 1, n_pre + 5))
    return loadings @ factors.T + noise + rng.normal(size=(n_donors + 1, 1))


def test_the_weights_are_the_optimum_when_the_donors_outnumber_the_pre_period() -> None:
    """The Frank-Wolfe gap ``g'w - min_j g_j``, with ``g`` the gradient of half the squared
    pre-period error, bounds how far the error is above its minimum, whatever found the weights.
    Projected gradient's 5000 steps left it at 8.5% of the error on this panel."""
    outcomes = _many_donors(0)
    weights = synthetic_control(outcomes, treated_unit=0, n_pre=30).weights
    donors = outcomes[1:, :30]
    residual = donors.T @ weights - outcomes[0, :30]
    gradient = donors @ residual
    assert float(gradient @ weights - gradient.min()) <= 1e-9 * 0.5 * float(residual @ residual)


def test_the_weights_do_not_move_when_the_outcomes_change_units_or_origin() -> None:
    """A synthetic control's weights are the same in any units, tiny ones included, and from any
    origin."""
    outcomes = _many_donors(1)
    weights = synthetic_control(outcomes, treated_unit=0, n_pre=30).weights
    for moved in (1e3 * outcomes - 7.0, 1e-12 * outcomes):
        np.testing.assert_allclose(
            synthetic_control(moved, treated_unit=0, n_pre=30).weights, weights, rtol=0.0, atol=1e-9
        )


def test_invalid_arguments_raise() -> None:
    outcomes = _factor_panel(seed=4, treated_loading=_in_hull)
    with pytest.raises(ValueError, match="treated_unit"):
        synthetic_control(outcomes, treated_unit=99, n_pre=N_PRE)
    with pytest.raises(ValueError, match="n_pre"):
        synthetic_control(outcomes, treated_unit=0, n_pre=0)
