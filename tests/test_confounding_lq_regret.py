"""chc.regret: confounding-robust LQ regret -- MSM radius through the CE order-doubling.

The §32 confounding half-width becomes a control-regret floor L_reg*Delta^2 that is SECOND order in
the confounding (Rocq ``confounding_lq_regret.v``): control is quadratically more robust than
estimation.
"""

import pytest

from chc.regret import (
    confounding_robust_lq_regret,
    confounding_robust_lq_regret_certificate,
    lq_regret_sensitivity,
)


def test_certificate_confirms_order_doubling_and_quadratic_floor() -> None:
    cert = confounding_robust_lq_regret_certificate()
    assert cert.ok
    assert cert.order_doubling_ratio == pytest.approx(
        1.0, abs=1e-3
    )  # L_reg is the exact leading coeff
    assert cert.floor_quadratic_ratio == pytest.approx(4.0, abs=1e-9)  # floor ~ Delta^2
    assert cert.regret_bounds[0] == pytest.approx(cert.statistical)  # Gamma=1 == statistical regret


def test_regret_bound_is_monotone_nondecreasing_in_gamma() -> None:
    cert = confounding_robust_lq_regret_certificate()
    b = cert.regret_bounds
    assert all(
        b[i] <= b[i + 1] + 1e-15 for i in range(len(b) - 1)
    )  # more confounding -> larger bound


def test_bound_matches_the_closed_form() -> None:
    # L_reg*(eps + Delta)^2 exactly
    assert confounding_robust_lq_regret(2.0, 0.1, 0.3) == pytest.approx(2.0 * (0.1 + 0.3) ** 2)
    assert confounding_robust_lq_regret(2.0, 0.1, 0.0) == pytest.approx(2.0 * 0.1**2)  # Delta=0


def test_pure_confounding_floor_is_below_the_linear_effect_bias() -> None:
    # Rocq floor_below_linear: for a half-width Delta in [0,1] the regret floor L*Delta^2 <= L*Delta
    l_reg = lq_regret_sensitivity(1.3, 0.4, 1.0)
    for delta in (0.1, 0.3, 0.5, 0.9):
        floor = confounding_robust_lq_regret(l_reg, 0.0, delta)
        assert floor <= l_reg * delta + 1e-12  # control regret floor below the effect-bias floor


def test_regret_sensitivity_is_nonnegative_and_vanishes_at_the_flat_point() -> None:
    assert lq_regret_sensitivity(1.3, 0.4, 1.0) > 0.0
    # r = b^2 makes u* locally flat in the effect -> the leading regret coefficient is zero
    assert lq_regret_sensitivity(1.0, 1.0, 1.0) == pytest.approx(0.0)
