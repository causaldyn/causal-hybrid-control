"""Property-based (hypothesis) tests for the §32/§35 sensitivity invariants.

The Rocq proofs assert these for symbolic inputs; here hypothesis hammers the shipped NumPy code
with random valid inputs -- the numeric complement catching float / edge-case drift hand-picked
examples miss. Properties: §32 radius never optimistic / tight at Gamma=1 / monotone; §35 gain
shift is a sign dichotomy, symmetric loss recovers CE, analytic improvement matches the numeric one.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from chc.sensitivity import (
    asymmetric_control_improvement,
    certainty_equivalence_control,
    confounding_robust_control,
    confounding_robust_inflation,
    confounding_robust_radius,
    msm_worst_case_mean,
    worst_case_asymmetric_loss,
)


def finite(lo: float, hi: float) -> st.SearchStrategy[float]:
    """Bounded, non-NaN, non-inf floats (keeps the sensitivity primitives in their valid domain)."""
    return st.floats(min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False)


gammas = finite(1.0, 50.0)
positive = finite(0.1, 20.0)
outcomes_st = st.lists(finite(-100.0, 100.0), min_size=8, max_size=400).map(np.asarray)


# ---- §32: bounded-density-ratio radius ----


@given(cvar_lo=finite(-50.0, 50.0), gap=positive, gamma=gammas)
def test_inflation_is_nonnegative_and_monotone_in_gamma(
    cvar_lo: float, gap: float, gamma: float
) -> None:
    cvar_up = cvar_lo + gap
    infl = confounding_robust_inflation(cvar_up, cvar_lo, gamma)
    assert infl >= -1e-9  # never optimistic (a nonnegative inflation over the point estimate)
    bigger = confounding_robust_inflation(cvar_up, cvar_lo, gamma + 1.0)
    assert bigger >= infl - 1e-9  # more assumed confounding -> a weakly larger radius


@given(cvar_lo=finite(-50.0, 50.0), gap=positive)
def test_inflation_is_zero_at_gamma_one(cvar_lo: float, gap: float) -> None:
    assert (
        confounding_robust_inflation(cvar_lo + gap, cvar_lo, 1.0) == 0.0
    )  # tight under no confounding


@given(outcomes=outcomes_st, gamma=gammas)
@settings(deadline=None)
def test_worst_case_mean_is_never_optimistic_and_monotone(
    outcomes: np.ndarray, gamma: float
) -> None:
    mu = float(np.mean(outcomes))
    wc = msm_worst_case_mean(outcomes, gamma)
    scale = 1.0 + abs(mu)
    assert wc >= mu - 1e-9 * scale  # the pessimistic worst-case never sits below the sample mean
    assert msm_worst_case_mean(outcomes, gamma + 2.0) >= wc - 1e-9 * scale  # monotone in Gamma


@given(outcomes=outcomes_st)
@settings(deadline=None)
def test_worst_case_mean_reduces_to_the_mean_at_gamma_one(outcomes: np.ndarray) -> None:
    assert msm_worst_case_mean(outcomes, 1.0) == float(np.mean(outcomes))  # Gamma=1 -> point ID


@given(outcomes=outcomes_st, base=positive, gamma=gammas)
@settings(deadline=None)
def test_robust_radius_never_below_nominal(outcomes: np.ndarray, base: float, gamma: float) -> None:
    assert confounding_robust_radius(base, outcomes, gamma) >= base - 1e-9  # pessimism only grows


# ---- §35: minimax controller under asymmetric loss ----

# b_hat > D > 0 (identified effect sign): draw b_hat, then D as a strict fraction of it.
_bhat = finite(0.5, 20.0)
_frac = finite(0.01, 0.95)


@given(b_hat=_bhat, frac=_frac, target=positive, a=positive)
def test_symmetric_loss_recovers_certainty_equivalence(
    b_hat: float, frac: float, target: float, a: float
) -> None:
    u_ce = certainty_equivalence_control(b_hat, target)
    u_rob = confounding_robust_control(b_hat, frac * b_hat, target, a, a)  # alpha == beta
    # symmetric -> radius does not move the gain (Rocq: kappa=0); float division is 1-ulp, not exact
    assert u_rob == pytest.approx(u_ce, rel=1e-12)


@given(b_hat=_bhat, frac=_frac, target=positive, a=positive, b=positive)
def test_gain_shift_follows_the_sign_dichotomy(
    b_hat: float, frac: float, target: float, a: float, b: float
) -> None:
    u_ce = certainty_equivalence_control(b_hat, target)
    u_rob = confounding_robust_control(b_hat, frac * b_hat, target, a, b)
    tol = 1e-9 * (1.0 + abs(u_ce))
    if a >= b:  # overshoot costlier -> conservative (gain shifted down, shift_factor_nonneg)
        assert u_rob <= u_ce + tol
    else:  # undershoot costlier -> aggressive (gain shifted up, shift_factor_nonpos)
        assert u_rob >= u_ce - tol


@given(b_hat=_bhat, frac=_frac, target=positive, a=positive, b=positive)
def test_analytic_improvement_matches_numeric_and_is_nonnegative(
    b_hat: float, frac: float, target: float, a: float, b: float
) -> None:
    d = frac * b_hat
    analytic = asymmetric_control_improvement(b_hat, d, target, a, b)
    u_ce = certainty_equivalence_control(b_hat, target)
    u_rob = confounding_robust_control(b_hat, d, target, a, b)
    w_ce = worst_case_asymmetric_loss(u_ce, b_hat, d, target, a, b)
    w_rob = worst_case_asymmetric_loss(u_rob, b_hat, d, target, a, b)
    scale = 1.0 + abs(analytic)
    assert analytic >= -1e-9 * scale  # the robust controller is never worse in worst-case loss
    assert analytic == (w_ce - w_rob) or abs(analytic - (w_ce - w_rob)) <= 1e-7 * scale


@given(b_hat=_bhat, frac=_frac, target=positive, a=positive)
def test_symmetric_improvement_is_zero(b_hat: float, frac: float, target: float, a: float) -> None:
    # alpha=beta: the (a-b) factor is literally 0.0, so the improvement is exactly zero
    assert asymmetric_control_improvement(b_hat, frac * b_hat, target, a, a) == 0.0
