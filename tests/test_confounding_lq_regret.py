"""chc.regret: confounding-robust LQ regret -- MSM radius through the CE order-doubling.

The §32 confounding half-width becomes a control-regret floor L_reg*Delta^2 that is SECOND order in
the confounding (Rocq ``confounding_lq_regret.v``): control is quadratically more robust than
estimation.
"""

import numpy as np
import pytest

from chc.regret import (
    confounding_regret_floor_certificate,
    confounding_robust_lq_regret,
    confounding_robust_lq_regret_certificate,
    confounding_robust_lq_regret_matrix,
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


# --- Result 36: empirical Delta^2 floor on synthetic confounded data ---


def test_empirical_floor_recovers_the_quadratic_slope() -> None:
    cert = confounding_regret_floor_certificate()
    assert cert.ok
    assert cert.exponent == pytest.approx(2.0, abs=0.15)  # regret ~ bias^2 on real confounded data
    assert cert.analytic_ratio == pytest.approx(1.0, abs=0.1)  # matches the analytic L_reg*Delta^2


def test_empirical_bias_and_regret_grow_with_confounding() -> None:
    cert = confounding_regret_floor_certificate()
    b = cert.biases
    g = cert.regrets
    assert all(b[i] <= b[i + 1] for i in range(len(b) - 1))  # more confounding -> larger bias
    assert all(g[i] <= g[i + 1] for i in range(len(g) - 1))  # ...and larger control regret


# --- Multivariate §33: matrix Frobenius lift (order-doubling via §21) ---


def test_matrix_regret_reduces_to_the_scalar_in_1x1() -> None:
    h = np.array([[2.0]])
    r = confounding_robust_lq_regret_matrix(h, np.array([[0.1]]), np.array([[0.3]]))
    assert r == pytest.approx(confounding_robust_lq_regret(2.0, 0.1, 0.3))  # == L_reg*(eps+Delta)^2


def test_matrix_confounding_floor_is_frobenius_quadratic() -> None:
    h = np.array([[2.0, 0.0], [0.0, 3.0]])
    delta = np.array([[0.2, 0.1], [0.0, 0.3]])
    zero = np.zeros((2, 2))
    floor = confounding_robust_lq_regret_matrix(h, zero, delta)
    doubled = confounding_robust_lq_regret_matrix(h, zero, 2.0 * delta)
    assert doubled == pytest.approx(
        4.0 * floor
    )  # tr(Delta^T H Delta) ~ ||Delta||_F^2, second order


def test_matrix_regret_is_nonnegative_for_psd_curvature() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((3, 3))
    h = a.T @ a  # PSD curvature
    e = rng.standard_normal((3, 3))
    assert confounding_robust_lq_regret_matrix(h, e, np.zeros((3, 3))) >= 0.0
