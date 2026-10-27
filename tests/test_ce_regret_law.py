"""The certainty-equivalent LQ regret's law under a Gaussian model error, against Maxima's
second- and fourth-order terms (``validation/ce_regret_law.mac``) and against the exact identity
drawn independently."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from chc.regret import (
    _chi_square_mixture_quantile,
    _gain_jacobian,
    ce_regret_law,
    dlqr,
)

# validation/ce_regret_law.mac STEP 3: (a, b, q, r), c2 and c4 / c2 under s^2 I, unit noise
PLANTS = {
    "A": ((1.0, 1.0, 1.0, 1.0), 2.3416407864998736, 4.441640786499874),
    "B": ((1.2, 0.5, 1.0, 1.0), 20.19088293082349, 17.25368408981332),
    "C": ((0.5, 1.0, 1.0, 0.1), 1.10000088760953, 5.097218989712963),
}


def _scalar(a: float, b: float, q: float, r: float):
    return tuple(np.array([[v]]) for v in (a, b, q, r))


def _scalar_regret(a, b, q, r, a_hat, b_hat):
    """The scalar plant's exact regret in closed form (unit noise), +inf where unstable."""
    big = r - q * b_hat**2 - a_hat**2 * r
    p_hat = (np.sqrt(big**2 + 4.0 * b_hat**2 * q * r) - big) / (2.0 * b_hat**2)
    k_hat = a_hat * b_hat * p_hat / (r + b_hat**2 * p_hat)
    big = r - q * b**2 - a**2 * r
    p = (math.sqrt(big**2 + 4.0 * b**2 * q * r) - big) / (2.0 * b**2)
    k = a * b * p / (r + b**2 * p)
    closed = a - b * k_hat
    with np.errstate(divide="ignore"):
        value = (r + b**2 * p) * (k_hat - k) ** 2 / (1.0 - closed**2)
    return np.where(np.abs(closed) >= 1.0, np.inf, value)


@pytest.mark.parametrize("plant", sorted(PLANTS))
def test_the_second_and_fourth_order_terms_are_maximas(plant: str) -> None:
    (a, b, q, r), c2, share = PLANTS[plant]
    s = 0.05
    law = ce_regret_law(*_scalar(a, b, q, r), s * s * np.eye(2), np.eye(1), draws=1)
    assert law.mean == pytest.approx(c2 * s * s, rel=1e-12, abs=0.0)
    # one direction moves the scalar gain: the law is c2 s^2 chi2_1
    assert law.weights[1] <= 1e-14 * law.weights[0]
    assert law.fourth_order_share == pytest.approx(share * s * s, rel=1e-6, abs=0.0)


@pytest.mark.parametrize(
    ("covariance", "c2", "share"),
    [
        (np.eye(2), 87.0 / 32.0, 33333.0 / 7424.0),
        (np.array([[1.0, 0.5], [0.5, 2.0]]), 69.0 / 32.0, 30525.0 / 5888.0),
    ],
)
def test_a_rational_point_reads_maximas_exact_terms(covariance, c2, share) -> None:
    # (a, b, q, r) = (1, 1, 4/3, 1): p* = 2, k* = 2/3, every term a rational (STEP 2)
    s = 0.04
    law = ce_regret_law(*_scalar(1.0, 1.0, 4.0 / 3.0, 1.0), s * s * covariance, np.eye(1), draws=1)
    assert law.mean == pytest.approx(c2 * s * s, rel=1e-12, abs=0.0)
    assert law.fourth_order_share == pytest.approx(share * s * s, rel=1e-6, abs=0.0)


# validation/ce_regret_law.mac STEP 5: A = [[1, 1/2], [0, 9/10]], B = [0, 1]', Q = I, R = 1,
# W = [[1, 1/4], [1/4, 2]]
A5 = np.array([[1.0, 0.5], [0.0, 0.9]])
B5 = np.array([[0.0], [1.0]])
W5 = np.array([[1.0, 0.25], [0.25, 2.0]])
G5 = np.array(
    [
        [1.831106701173259, 0.8514760480944277],
        [-0.19216899571939858, 0.5940958076222892],
        [0.9873724351102247, 0.13682413798275656],
        [-0.12614709354567552, 0.7904010493036041],
        [0.020625790776803873, -0.4702459771368426],
        [-0.2554552592546742, -0.5552640979974994],
    ]
)
M5 = np.array(
    [
        [50.50801002952906, -6.719200074563537, 27.903982282310537, -5.241815822098227,
         1.5648496160020189, -6.140381682170055],
        [-6.719200074563537, 6.6532305178118945, -6.428183931998162, 7.8522116618363595,
         -4.251678048139397, -2.861234964261922],
        [27.903982282310537, -6.428183931998162, 16.696874866987173, -6.270091189053777,
         2.7713986220093814, -1.6578010395902778],
        [-5.241815822098227, 7.8522116618363595, -6.270091189053777, 9.432548631058822,
         -5.18566590701954, -3.9320669329213533],
        [1.5648496160020189, -4.251678048139397, 2.7713986220093814, -5.18566590701954,
         2.8873245316893303, 2.3920626990287257],
        [-6.140381682170055, -2.861234964261922, -1.6578010395902778, -3.9320669329213533,
         2.3920626990287257, 3.095452033322389],
    ]
)  # fmt: skip


def test_the_gain_jacobian_is_maximas_at_sixty_digits() -> None:
    k, p = dlqr(A5, B5, np.eye(2), np.eye(1))
    jacobian = _gain_jacobian(A5, B5, k, p, np.eye(1)).reshape(6, 2)
    np.testing.assert_allclose(jacobian, G5, rtol=0.0, atol=1e-12)


def test_the_weights_are_maximas_second_order_matrix() -> None:
    s = 0.01
    law = ce_regret_law(A5, B5, np.eye(2), np.eye(1), s * s * np.eye(6), W5, draws=1)
    expected = np.sort(np.linalg.eigvalsh(M5))[::-1] * s * s
    np.testing.assert_allclose(law.weights, expected, rtol=1e-10, atol=1e-12 * expected[0])
    assert law.mean == pytest.approx(np.trace(M5) * s * s, rel=1e-11, abs=0.0)


def test_the_fourth_order_extrapolation_reads_the_second_order_term_it_divides_by() -> None:
    # the rule's own c2, extrapolated from three scales, against the closed-form trace: the share
    # rests on it, so a rule or a step that misreads the quadratic term shows here first
    s = 0.01
    law = ce_regret_law(A5, B5, np.eye(2), np.eye(1), s * s * np.eye(6), W5, draws=1)
    assert 0.0 < law.fourth_order_share < 1.0
    small = ce_regret_law(A5, B5, np.eye(2), np.eye(1), 0.25 * s * s * np.eye(6), W5, draws=1)
    # c4 / c2 scales as the covariance does
    assert small.fourth_order_share == pytest.approx(
        0.25 * law.fourth_order_share, rel=1e-5, abs=0.0
    )


@pytest.mark.parametrize("weights", [(1.0,), (2.0, 2.0), (3.0, 1.0, 0.25)])
def test_the_mixture_quantile_is_the_mixtures(weights) -> None:
    w = np.array(weights)
    levels = (0.5, 0.9, 0.99)
    found = [_chi_square_mixture_quantile(level, w) for level in levels]
    if len(set(weights)) == 1:
        expected = [weights[0] * stats.chi2.ppf(level, len(weights)) for level in levels]
        np.testing.assert_allclose(found, expected, rtol=1e-9, atol=0.0)
        return
    draws = 400_000
    sample = (np.random.default_rng(3).standard_normal((draws, w.size)) ** 2) @ w
    for level, quantile in zip(levels, found, strict=True):
        share = float((sample <= quantile).mean())
        assert abs(share - level) <= 4.5 * math.sqrt(level * (1.0 - level) / draws)


def test_the_exact_quantiles_are_the_identity_on_the_same_draws() -> None:
    (a, b, q, r), _, _ = PLANTS["B"]
    s, draws, seed = 0.15, 3000, 11
    covariance = s * s * np.eye(2)
    levels = (0.1, 0.5, 0.9, 0.99, 0.999)
    law = ce_regret_law(
        *_scalar(a, b, q, r), covariance, np.eye(1), levels=levels, draws=draws, seed=seed
    )
    values, vectors = np.linalg.eigh(covariance)
    root = vectors * np.sqrt(values)
    errors = np.random.default_rng(seed).standard_normal((draws, 2)) @ root.T
    exact = np.sort(_scalar_regret(a, b, q, r, a + errors[:, 0], b + errors[:, 1]))
    picks = np.ceil(np.array(levels) * draws).astype(int) - 1
    np.testing.assert_allclose(law.exact_quantiles, exact[picks], rtol=1e-8, atol=0.0)
    assert law.destabilise_probability == float(np.isinf(exact).mean())


def test_the_destabilising_share_is_the_plants() -> None:
    (a, b, q, r), _, _ = PLANTS["B"]
    s, draws = 0.2, 4000
    law = ce_regret_law(*_scalar(a, b, q, r), s * s * np.eye(2), np.eye(1), draws=draws, seed=5)
    reference_draws = 2_000_000
    errors = s * np.random.default_rng(6).standard_normal((reference_draws, 2))
    with np.errstate(invalid="ignore"):
        reference = float(
            np.isinf(_scalar_regret(a, b, q, r, a + errors[:, 0], b + errors[:, 1])).mean()
        )
    spread = math.sqrt(reference * (1.0 - reference) * (1.0 / draws + 1.0 / reference_draws))
    assert 0.03 < reference < 0.05
    assert abs(law.destabilise_probability - reference) <= 4.5 * spread
    assert law.destabilise_upper >= law.destabilise_probability


def test_the_law_matches_the_exact_median_where_it_is_valid() -> None:
    (a, b, q, r), _, _ = PLANTS["A"]
    s, draws = 0.05, 4000
    law = ce_regret_law(*_scalar(a, b, q, r), s * s * np.eye(2), np.eye(1), draws=draws, seed=2)
    assert law.valid
    assert law.destabilise_probability == 0.0
    # the median's standard error over the draws, through chi2_1's density at its median
    median = stats.chi2.ppf(0.5, 1)
    spread = math.sqrt(0.25 / draws) / stats.chi2.pdf(median, 1) / median
    ratio = law.exact_quantiles[0] / law.quantiles[0]
    assert abs(ratio - 1.0) <= 0.01 + 4.5 * spread


@pytest.mark.parametrize(("plant", "s", "valid"), [("A", 0.1, True), ("B", 0.1, False)])
def test_validity_is_the_fourth_order_share_against_the_tolerance(plant, s, valid) -> None:
    (a, b, q, r), _, _ = PLANTS[plant]
    law = ce_regret_law(*_scalar(a, b, q, r), s * s * np.eye(2), np.eye(1), draws=1)
    assert law.valid is valid


def test_the_quantiles_scale_the_chi_square_on_one_direction() -> None:
    (a, b, q, r), c2, _ = PLANTS["C"]
    s = 0.03
    law = ce_regret_law(*_scalar(a, b, q, r), s * s * np.eye(2), np.eye(1), draws=1)
    expected = c2 * s * s * stats.chi2.ppf(law.levels, 1)
    np.testing.assert_allclose(law.quantiles, expected, rtol=1e-9, atol=0.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"covariance": np.eye(3)}, "covariance must be"),
        ({"covariance": np.array([[1.0, 2.0], [2.0, 1.0]])}, "covariance must be symmetric"),
        ({"state_moment": -np.eye(1)}, "state_moment must be symmetric"),
        ({"levels": (0.5, 1.0)}, "levels must lie"),
        ({"draws": 0}, "draws must be"),
    ],
)
def test_the_law_refuses_what_it_cannot_read(kwargs, match) -> None:
    arguments = {"covariance": 0.01 * np.eye(2), "state_moment": np.eye(1)} | kwargs
    a, b, q, r = _scalar(1.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match=match):
        ce_regret_law(
            a,
            b,
            q,
            r,
            arguments.pop("covariance"),
            arguments.pop("state_moment"),
            **arguments,
        )
