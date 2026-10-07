"""Augmented SCM de-biases the synthetic control when the treated unit is outside the donor hull."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from chc.scm import augmented_synthetic_control, synthetic_control

TAU = 2.0  # post-treatment effect added to the treated unit
N_PRE = 25
Loading = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def _factor_panel(seed: int, treated_loading: Loading, *, trend: bool = False) -> np.ndarray:
    """Latent-factor panel; unit 0 is treated (effect TAU post), the rest are donors. With
    ``trend`` the factors are random walks."""
    rng = np.random.default_rng(seed)
    n_donors, n_post, rank = 30, 10, 3
    n_periods = N_PRE + n_post
    factors = rng.normal(0.0, 1.0, (n_periods, rank))
    if trend:
        factors = factors.cumsum(axis=0)
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


def test_the_augmented_estimate_is_the_same_in_any_units() -> None:
    """The ridge was 1.0 in the outcomes' squared units, so it outweighed donors logged in small
    units: on a treated unit outside the donors' hull, at 1e-3 of the outcomes' units the effect
    read 2.266, plain SCM's, where it read 1.966 against a truth of 2.0, its path moved by 1.14 of
    its size, and at 1e3 of them it read 1.946."""
    outcomes = _factor_panel(seed=1, treated_loading=_outside_hull)
    reference = augmented_synthetic_control(outcomes, treated_unit=0, n_pre=N_PRE).att
    for s in (1e-9, 1e-6, 1e-3, 1e3, 1e6):
        att = augmented_synthetic_control(s * outcomes, treated_unit=0, n_pre=N_PRE).att
        np.testing.assert_allclose(att / s, reference, rtol=1e-10, atol=0.0, err_msg=f"{s}")


@pytest.mark.parametrize(
    ("absolute", "origin"),
    [(0.1, 0.0), (1.0, 0.0), (10.0, 0.0), (1.0, 10.0)],
    ids=["0.1", "1.0", "10.0", "1.0 from another origin"],
)
def test_the_outcome_model_is_a_ridge_with_a_free_intercept(absolute: float, origin: float) -> None:
    """Ben-Michael, Feller and Rothstein's outcome model: the donors' outcomes after treatment
    regressed on theirs before it, with an intercept that is not penalised, here a column of its
    own, and ``ridge = lam / v`` a ridge of ``lam`` in the outcomes' units, ``v`` the donors'
    variance about each period's mean. Without the intercept, every outcome raised by 10 moved the
    estimate from 1.966 to 1.988."""
    outcomes = _factor_panel(seed=1, treated_loading=_outside_hull) + origin
    donors, treated = outcomes[1:], outcomes[0]
    pre, post = donors[:, :N_PRE], donors[:, N_PRE:]
    weights = synthetic_control(outcomes, treated_unit=0, n_pre=N_PRE).weights
    design = np.column_stack([np.ones(len(pre)), pre])
    penalty = np.diag([0.0] + [absolute] * N_PRE)
    theta = np.linalg.solve(design.T @ design + penalty, design.T @ post)[1:]
    expected = treated[N_PRE:] - post.T @ weights - theta.T @ (treated[:N_PRE] - pre.T @ weights)
    share = absolute / float(np.mean((pre - pre.mean(axis=0)) ** 2))
    result = augmented_synthetic_control(outcomes, treated_unit=0, n_pre=N_PRE, ridge=share)
    np.testing.assert_allclose(result.att, expected, rtol=1e-10, atol=0.0)


@pytest.mark.parametrize("moved", ["every period", "one period"])
@pytest.mark.parametrize(
    ("spreads", "rtol"), [(1e3, 1e-11), (1e13, 1e-2)], ids=["1e3 spreads", "1e13 spreads"]
)
def test_the_augmented_estimate_is_the_same_with_the_outcomes_far_from_zero(
    spreads: float, rtol: float, moved: str
) -> None:
    """Raised by 1e3 of the donors' spread and logged at 1e-6 of their units, the outcomes read the
    same effect: the slopes are solved for centred, so the level costs them no digits, and the
    ridge is a share of a variance, which the level does not move. Without the intercept, and with
    the ridge in the outcomes' units, the effect read 2.266 where it read 1.966. Raised by 1e13 of
    it in every period, the donors still move apart by 444 eps of their size, which is data, not
    rounding; a floor of 1e-12 of their size counted it as rounding, and the effect read 2.268, the
    synthetic control's."""
    outcomes = _factor_panel(seed=1, treated_loading=_outside_hull)
    reference = augmented_synthetic_control(outcomes, treated_unit=0, n_pre=N_PRE).att
    shift = np.zeros(outcomes.shape[1])
    shift[slice(None) if moved == "every period" else 3] = spreads * np.std(outcomes[1:, :N_PRE])
    att = augmented_synthetic_control(1e-6 * (outcomes + shift), treated_unit=0, n_pre=N_PRE).att
    # a level d spreads off zero keeps d * eps of the spread in each entry: 2e-13 at 1e3 spreads
    # (measured 4e-13 here), and 2e-3 at 1e13 (measured 2.3e-3)
    np.testing.assert_allclose(att / 1e-6, reference, rtol=rtol, atol=0.0)


@pytest.mark.parametrize(
    ("donors", "rtol"),
    [("at zero", 0.0), ("all alike", 0.0), ("apart by 4e-15 of their level", 1e-9)],
    ids=["at zero", "all alike", "apart by rounding"],
)
def test_donors_that_do_not_move_apart_before_treatment_leave_the_synthetic_control(
    donors: str, rtol: float
) -> None:
    """With nothing before treatment to tell the donors apart the outcome model has nothing to fit,
    so the correction is zero and the estimate is the synthetic control's. Donors all alike centre
    to their mean's rounding, a deviation of 1.7 eps of their size here, and donors apart by 4e-15
    of their level, 18 eps, move apart by rounding alone: a deviation of at most 64 eps counts as
    none. A ridge scaled by the deviation let the slopes fit the rounding, and one period's effect
    moved by 206, and by 1e15."""
    outcomes = _factor_panel(seed=2, treated_loading=_in_hull)
    outcomes[1:, :N_PRE] = 0.0 if donors == "at zero" else outcomes[1, :N_PRE] + 0.1
    if donors == "apart by 4e-15 of their level":
        signs = np.where(np.random.default_rng(8).random((30, N_PRE)) < 0.5, -1.0, 1.0)
        outcomes[1:, :N_PRE] *= 1.0 + 4e-15 * signs
    scm = synthetic_control(outcomes, treated_unit=0, n_pre=N_PRE)
    ascm = augmented_synthetic_control(outcomes, treated_unit=0, n_pre=N_PRE)
    # apart by rounding, the slopes are the rounding over the donors' mean square: 7e-12 here
    np.testing.assert_allclose(ascm.att, scm.att, rtol=rtol, atol=0.0)


def test_a_period_the_donors_barely_moved_in_counts_as_left_out() -> None:
    """Every period is the one outcome in one unit, so the ridge is a share of the donors' variance
    pooled over the periods, and a period in which they sit at one level but for 1e-11 of it adds
    nothing at a ridge fixed in the outcomes' units. A share of each period's own variance let that
    period's slope grow as its spread shrank, and with the treated unit 10 below the donors there
    the effect read -1.6e9, and -13.6 at a spread of 1e-3, against a truth of 2."""
    outcomes = _factor_panel(seed=1, treated_loading=_outside_hull)
    rng = np.random.default_rng(5)
    outcomes[1:, 7] = 3.7 * (1.0 + 1e-11 * rng.standard_normal(outcomes.shape[0] - 1))

    def effect(panel: np.ndarray, n_pre: int) -> np.ndarray:
        pre = panel[1:, :n_pre]
        share = 1.0 / float(np.mean((pre - pre.mean(axis=0)) ** 2))
        return augmented_synthetic_control(panel, treated_unit=0, n_pre=n_pre, ridge=share).att

    kept = effect(outcomes, N_PRE)
    left_out = effect(np.delete(outcomes, 7, axis=1), N_PRE - 1)
    # the period's slope is its 1e-11 spread over the ridge, times the treated unit's 10 off it:
    # 1.5e-10 of the effect here
    np.testing.assert_allclose(kept, left_out, rtol=1e-8, atol=0.0)


@pytest.mark.parametrize(
    ("trend", "first_seed"),
    [(False, 1000), (True, 2000)],
    ids=["white-noise factors", "random-walk factors"],
)
def test_the_default_ridge_errs_within_a_tenth_of_the_best_fixed_share(
    trend: bool, first_seed: int
) -> None:
    """The donors' whole variance was the best share tried where the factors are white noise. Where
    they are random walks it missed a truth of 2 by a root mean square of 0.359 over 200 panels, 3.4
    times the best share's 0.105, at 0.03. The default tenth missed by 0.0526 and 0.110, against
    0.0502 and 0.105 at the best; plain SCM by 0.805 and 11.9. Indexed to their first period, the
    random-walk panels missed by 0.142 at a tenth, 1.13 times the best share's 0.126, at 0.01."""
    panels = [
        _factor_panel(seed, treated_loading=_outside_hull, trend=trend)
        for seed in range(first_seed, first_seed + 200)
    ]

    def rmse(effects: list[float]) -> float:
        return float(np.sqrt(np.mean((np.asarray(effects) - TAU) ** 2)))

    default = rmse([augmented_synthetic_control(p, 0, N_PRE).overall for p in panels])
    best = min(
        rmse([augmented_synthetic_control(p, 0, N_PRE, ridge=share).overall for p in panels])
        for share in (0.01, 0.03, 0.1, 0.3, 1.0)
    )
    assert default <= 1.1 * best
