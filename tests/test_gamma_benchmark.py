"""chc.uncertainty: calibrating the MSM sensitivity Gamma instead of only assuming it (§32).

Two calibrations, and both can fail here. ``benchmark_gamma`` prices Gamma in units of the
confounding the OBSERVED covariates carry -- which forces a logarithmic scale, because odds ratios
compose multiplicatively. ``negative_control_gamma`` inverts a known-null outcome for the smallest
Gamma that reconciles it, which is a LOWER bound on the confounding actually present.

The load-bearing test is ``test_the_two_endpoints_read_opposite_tails``: the MSM interval is not
symmetric about the mean, so a positive estimate is reconciled by the lower endpoint and a negative
one by the upper. Reusing the upper tail for both is the bug this file fences off, and Rocq
``symmetric_reflex_wrong_verdict`` shows it can turn ``inf`` into a finite 2.
"""

import numpy as np
import pytest

from chc.uncertainty import (
    _top_tail_mean,
    benchmark_gamma,
    gamma_benchmark_certificate,
    msm_worst_case_mean,
    negative_control_gamma,
)


def _design(n: int, beta: list[float], seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, len(beta)))
    probability = 1.0 / (1.0 + np.exp(-(x @ np.asarray(beta))))
    return (rng.uniform(size=n) < probability).astype(np.float64), x


def test_the_benchmark_ranks_covariates_by_their_true_strength() -> None:
    treated, covariates = _design(4000, [1.5, 0.4, 0.0])
    benchmark = benchmark_gamma(treated, covariates, 3.0, names=("strong", "weak", "null"))
    strong, weak, null = benchmark.implied_gamma
    assert strong > weak > null
    assert benchmark.strongest == "strong"
    assert benchmark.strongest_gamma == pytest.approx(strong)


def test_multiples_of_strongest_is_an_exponent_not_a_ratio() -> None:
    treated, covariates = _design(4000, [1.5, 0.4])
    benchmark = benchmark_gamma(treated, covariates, 7.0)
    # Gamma = Gamma_strongest ** k is the definition; Gamma / Gamma_strongest is a different claim.
    assert benchmark.strongest_gamma**benchmark.multiples_of_strongest == pytest.approx(7.0)


def test_a_covariate_that_moves_nothing_sets_no_scale() -> None:
    treated, covariates = _design(500, [1.5])
    dead = np.column_stack([covariates[:, 0], np.zeros(covariates.shape[0])])
    benchmark = benchmark_gamma(treated, dead[:, 1:], 3.0)
    assert benchmark.strongest_gamma == pytest.approx(1.0)
    assert np.isinf(benchmark.multiples_of_strongest)


def test_the_sup_grows_with_the_sample_and_the_quantile_does_not() -> None:
    sup, quantile = [], []
    for n in (500, 32000):
        treated, covariates = _design(n, [1.5, 0.4], seed=3)
        sup.append(benchmark_gamma(treated, covariates, 3.0, quantile=1.0).strongest_gamma)
        quantile.append(benchmark_gamma(treated, covariates, 3.0, quantile=0.95).strongest_gamma)
    assert sup[1] > 1.5 * sup[0]  # an extreme order statistic under an unbounded covariate
    assert 0.7 < quantile[1] / quantile[0] < 1.4


def test_the_quantile_orders_the_reported_sensitivity() -> None:
    treated, covariates = _design(2000, [1.2, 0.3])
    scores = [
        benchmark_gamma(treated, covariates, 3.0, quantile=q).strongest_gamma
        for q in (0.5, 0.9, 0.99, 1.0)
    ]
    assert scores == sorted(scores)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"assumed_gamma": 0.5}, "Gamma must be >= 1"),
        ({"assumed_gamma": 3.0, "quantile": 0.0}, "quantile must lie"),
        ({"assumed_gamma": 3.0, "quantile": 1.5}, "quantile must lie"),
        ({"assumed_gamma": 3.0, "names": ("only-one",)}, "names for"),
    ],
)
def test_the_benchmark_refuses_incoherent_inputs(kwargs: dict, message: str) -> None:
    treated, covariates = _design(200, [1.0, 0.5])
    with pytest.raises(ValueError, match=message):
        benchmark_gamma(treated, covariates, **kwargs)


def test_the_benchmark_needs_a_two_dimensional_design_with_a_column() -> None:
    treated, covariates = _design(200, [1.0])
    with pytest.raises(ValueError, match="must be 2-D"):
        benchmark_gamma(treated, covariates.ravel(), 3.0)
    with pytest.raises(ValueError, match="at least one observed covariate"):
        benchmark_gamma(treated, covariates[:, :0], 3.0)


def test_the_shipped_bound_is_the_blend_maxima_derives() -> None:
    # validation/gamma_benchmark.mac (1): the three-constant form collapses to one blend weight.
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(50):
        outcomes = rng.standard_normal(int(rng.integers(20, 400)))
        gamma = float(rng.uniform(1.0, 12.0))
        mean = float(outcomes.mean())
        tail = _top_tail_mean(outcomes, 1.0 / (gamma + 1.0))
        collapsed = mean + (1.0 - 1.0 / gamma) * (tail - mean)
        worst = max(worst, abs(msm_worst_case_mean(outcomes, gamma) - collapsed))
    assert worst < 1e-12


def test_the_two_endpoints_read_opposite_tails() -> None:
    rng = np.random.default_rng(11)
    outcomes = np.concatenate([rng.standard_normal(400), 8.0 + rng.standard_normal(20)]) + 0.4
    mean = float(outcomes.mean())
    gamma = negative_control_gamma(outcomes)
    assert -msm_worst_case_mean(-outcomes, gamma) == pytest.approx(0.0, abs=1e-8)

    def reflex_endpoint(g: float) -> float:
        """The symmetric-interval reflex: the TOP tail used for the LOWER endpoint."""
        return mean - (1.0 - 1.0 / g) * (_top_tail_mean(outcomes, 1.0 / (g + 1.0)) - mean)

    low, high = 1.0, 1e6
    while high - low > 1e-9 * low:
        mid = 0.5 * (low + high)
        low, high = (low, mid) if reflex_endpoint(mid) <= 0.0 else (mid, high)
    # the heavy upper tail reaches zero sooner, so the reflex declares the null reconciled at a
    # sensitivity where the true interval has not yet covered it -- it understates the confounding
    assert high < gamma
    assert -msm_worst_case_mean(-outcomes, high) > 0.0


def test_the_calibration_is_invariant_to_the_sign_of_the_null_estimate() -> None:
    rng = np.random.default_rng(5)
    outcomes = rng.standard_normal(1500) + 0.3
    assert negative_control_gamma(outcomes) == pytest.approx(negative_control_gamma(-outcomes))


def test_a_null_that_never_crosses_zero_refutes_the_model_class() -> None:
    rng = np.random.default_rng(2)
    assert np.isinf(negative_control_gamma(np.abs(rng.standard_normal(500)) + 1.0))
    assert negative_control_gamma(np.zeros(10)) == pytest.approx(1.0)


def test_a_larger_planted_bias_needs_a_larger_gamma() -> None:
    rng = np.random.default_rng(13)
    noise = rng.standard_normal(3000)
    gammas = [negative_control_gamma(noise + bias) for bias in (0.05, 0.15, 0.35)]
    assert gammas == sorted(gammas)
    assert gammas[0] > 1.0


def test_the_certificate_passes_every_gate() -> None:
    certificate = gamma_benchmark_certificate()
    assert certificate.ok
    assert certificate.monotone_in_strength
    assert certificate.ranks_with_truth
    assert certificate.quantile_growth < certificate.sup_growth
    assert certificate.null_floor_scaled < 12.0
    assert certificate.endpoint_residual < 1e-6
    assert certificate.unreconcilable_is_infinite
