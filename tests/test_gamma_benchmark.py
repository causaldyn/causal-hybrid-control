"""chc.uncertainty: calibrating the MSM sensitivity Gamma instead of only assuming it (§32).

Two calibrations, and both can fail here. ``benchmark_gamma`` prices Gamma in units of the
confounding the OBSERVED covariates carry -- which forces a logarithmic scale, because odds ratios
compose multiplicatively. ``negative_control_gamma`` inverts a known-null outcome for the smallest
Gamma that reconciles it, which is a LOWER bound on the confounding actually present.

The load-bearing test is ``test_the_two_endpoints_read_opposite_tails``: the MSM interval is not
symmetric about the mean, so a positive estimate is reconciled by the lower endpoint and a negative
one by the upper. Reusing the upper tail for both is the bug this file fences off, and Rocq
``symmetric_reflex_wrong_verdict`` shows it can turn ``inf`` into a finite 2.

Both read Gamma in the marginal sensitivity model's units, which are not Rosenbaum's: on a binary
confounder whose pull on the treatment and on the outcome is known, the benchmark of the confounder
itself is its MSM Gamma, Rosenbaum's lies between that and its square, and the Gamma a known-null
outcome needs climbs from 1 to the benchmark as the confounder's pull on the outcome grows.
"""

import math

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
    # EXACTLY 1.0, not approx: two independent fits leave a rounding residue, and approx would let
    # 1 + 2.2e-16 through -- which log() then turns into a reported confounding strength of 5e15.
    # The floor lives in the producer, so this holds on every interpreter and numpy build.
    assert benchmark.strongest_gamma == 1.0
    assert np.isinf(benchmark.multiples_of_strongest)


def test_the_noise_floor_does_not_swallow_a_weak_but_real_covariate() -> None:
    # The other side of the floor that makes the test above platform-independent: it is set at the
    # backward-stability scale of the logit matvec, so a genuinely weak confounder must survive it.
    # A coefficient of 0.05 is far weaker than anything a sensitivity analysis would call material
    # and is still eleven orders of magnitude above the floor.
    treated, covariates = _design(4000, [0.05])
    benchmark = benchmark_gamma(treated, covariates, 3.0)
    assert benchmark.strongest_gamma > 1.0
    assert np.isfinite(benchmark.multiples_of_strongest)


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


def _odds(share: float) -> float:
    return share / (1.0 - share)


def _filled(n: int, share: float) -> np.ndarray:
    """``n`` units, the first ``round(share * n)`` of them ones: a stratum filled exactly."""
    ones = round(share * n)
    return np.repeat([1.0, 0.0], [ones, n - ones])


def _binary_confounder(pull: float, n: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    """A log with one confounder ``u``, 30% ones, which multiplies the odds of treatment by
    ``pull`` from 1 to 4 at ``u = 0``: Rosenbaum's Gamma is ``pull``, up to the strata's
    rounding."""
    exposed = round(0.3 * n)
    u = np.repeat([1.0, 0.0], [exposed, n - exposed])
    odds = 0.25 * pull
    treated = np.concatenate([_filled(exposed, odds / (1.0 + odds)), _filled(n - exposed, 0.2)])
    return treated, u


def _outcome(treated: np.ndarray, u: np.ndarray, pull: float) -> np.ndarray:
    """An outcome the treatment leaves alone and ``u`` multiplies the odds of by ``pull``, about
    even odds, each stratum of ``(u, treated)`` filled exactly."""
    root = math.sqrt(pull)
    y = np.empty_like(u)
    for level, share in ((1.0, root / (1.0 + root)), (0.0, 1.0 / (1.0 + root))):
        for arm in (1.0, 0.0):
            stratum = (u == level) & (treated == arm)
            y[stratum] = _filled(int(stratum.sum()), share)
    return y


@pytest.mark.parametrize("pull", [1.5, 3.0, 10.0, 50.0])
def test_a_confounders_benchmark_is_its_msm_gamma_inside_rosenbaums_bracket(pull: float) -> None:
    """Rosenbaum(Gamma) within MSM(Gamma) within Rosenbaum(Gamma^2) (Zhao, Small and Bhattacharya
    2019, Prop. 7.1). Rosenbaum's model bounds the odds of treatment of two units against each
    other and the MSM each unit's against the stratum's, so the benchmark, which drops the
    confounder from the propensity, must report the second."""
    treated, u = _binary_confounder(pull)
    stratum = _odds(treated.mean())
    msm = max(_odds(treated[u == 1].mean()) / stratum, stratum / _odds(treated[u == 0].mean()))
    rosenbaum = _odds(treated[u == 1].mean()) / _odds(treated[u == 0].mean())
    benchmark = benchmark_gamma(treated, u[:, None], 2.0, quantile=1.0).strongest_gamma
    assert benchmark == pytest.approx(msm, rel=1e-6)
    assert benchmark < rosenbaum < benchmark**2


@pytest.mark.parametrize("pull", [1.5, 3.0, 10.0, 50.0])
def test_a_null_outcome_needs_the_benchmark_only_when_the_confounder_is_the_outcome(
    pull: float,
) -> None:
    """The benchmark prices the treatment side alone. How much Gamma a confounder forces also
    depends on its pull on the outcome, as in Rosenbaum and Silber's amplification: none when it
    leaves the outcome alone, more the more it moves it, and the benchmark itself, the sharp end
    of the MSM's interval, when the outcome is the confounder."""
    treated, u = _binary_confounder(pull)
    benchmark = benchmark_gamma(treated, u[:, None], 2.0, quantile=1.0).strongest_gamma
    needed = []
    for outcome_pull in (1.0, 1.5, 3.0, 10.0, 100.0, 1e4):
        y = _outcome(treated, u, outcome_pull)
        needed.append(negative_control_gamma(y[treated == 1] - y[treated == 0].mean()))
    assert needed[0] == pytest.approx(1.0, abs=1e-3)
    assert needed == sorted(needed)
    assert needed[-1] < benchmark
    sharp = negative_control_gamma(u[treated == 1] - u[treated == 0].mean())
    assert sharp == pytest.approx(benchmark, rel=1e-6)


def test_a_nan_covariate_is_refused_rather_than_read_as_no_confounding() -> None:
    """One nan covariate read every implied Gamma as 1.0, where they are 6.1 and 3.2, and an
    assumed Gamma of 2 as infinitely many times the strongest covariate."""
    treated, covariates = _design(400, [1.0, 0.5])
    covariates[3, 1] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        benchmark_gamma(treated, covariates, assumed_gamma=2.0)
    with pytest.raises(ValueError, match="must be >= 1"):
        benchmark_gamma(treated, np.nan_to_num(covariates), assumed_gamma=float("nan"))


def test_a_nan_negative_control_is_refused_rather_than_calibrated_at_the_ceiling() -> None:
    """A nan outcome, or a nan ``tol``, returned ``gamma_max``, 1e6, as the confounding the
    negative control measured, where it measures 1.37."""
    outcomes = np.random.default_rng(1).normal(0.3, 1.0, 200)
    with pytest.raises(ValueError, match="must be finite"):
        negative_control_gamma(np.where(np.arange(200) == 5, np.nan, outcomes))
    with pytest.raises(ValueError, match="tol must be positive"):
        negative_control_gamma(outcomes, tol=float("nan"))
    with pytest.raises(ValueError, match="gamma_max at least 1"):
        negative_control_gamma(outcomes, gamma_max=float("nan"))
    with pytest.raises(ValueError, match="gamma_max at least 1 and finite"):
        negative_control_gamma(outcomes, gamma_max=float("inf"))
