"""External LaLonde benchmark: CHC estimators recover the randomized ATE from confounded data."""

from __future__ import annotations

import urllib.error

import pytest

from chc.estimators import BackdoorOLS, DoubleML
from chc.lalonde import LalondeData, lalonde_ate, load_lalonde


@pytest.fixture(scope="module")
def data() -> LalondeData:
    try:
        return load_lalonde()
    except (urllib.error.URLError, OSError) as exc:  # offline / mirror down -> gated, like BOPTEST
        pytest.skip(f"LaLonde data unavailable: {exc}")


def test_experimental_ate_is_the_published_benchmark(data: LalondeData) -> None:
    assert data.experimental_ate == pytest.approx(1794, abs=50)  # Dehejia-Wahba randomized effect


def test_naive_observational_estimate_is_catastrophically_biased(data: LalondeData) -> None:
    assert data.naive_ate < -5000  # CPS controls out-earn the treated -> wrong sign, ~ -$8500


def test_backdoor_adjustment_recovers_the_sign(data: LalondeData) -> None:
    backdoor = lalonde_ate(data, BackdoorOLS())
    assert backdoor > 0.0  # adjusting for covariates flips the sign back positive
    assert abs(backdoor - data.experimental_ate) < abs(data.naive_ate - data.experimental_ate)


def test_flexible_double_ml_recovers_most_of_the_effect(data: LalondeData) -> None:
    backdoor = lalonde_ate(data, BackdoorOLS())
    dml = lalonde_ate(data, DoubleML(degree=3, folds=5))
    assert dml > backdoor  # cross-fitted flexible nuisances beat linear adjustment
    assert abs(dml - data.experimental_ate) < 0.25 * abs(data.naive_ate - data.experimental_ate)
