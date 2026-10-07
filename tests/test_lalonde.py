"""External LaLonde benchmark: CHC estimators recover the randomized ATE from confounded data."""

from __future__ import annotations

import urllib.error

import numpy as np
import pytest

from chc.estimators import BackdoorOLS, DoubleML, EffectEstimate
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


class _TreatedShare:
    """An estimator with no check of its own, as a caller's may be: the refusal is lalonde_ate's."""

    def estimate(self, data, *, treatment="u", outcome="x_next", covariates=()):
        return EffectEstimate(float(np.mean(np.asarray(data[treatment]))))


@pytest.mark.parametrize("name", ["treat", "re78"])
def test_a_covariate_named_for_the_treatment_or_the_outcome_is_refused(name: str) -> None:
    """The estimator reads the treatment and the outcome as ``treat`` and ``re78``, from the dict
    the covariates are written into after them: a covariate of either name replaced it."""
    rng = np.random.default_rng(0)
    data = LalondeData(
        treatment=(rng.random(50) < 0.5).astype(float),
        outcome=rng.normal(5000.0, 1000.0, 50),
        covariates={"age": rng.normal(30.0, 5.0, 50), name: rng.normal(0.0, 1.0, 50)},
        experimental_ate=0.0,
    )
    with pytest.raises(ValueError, match=f"the column '{name}' is read as the"):
        lalonde_ate(data, _TreatedShare())


class _Recorded:
    """An estimator that keeps the covariates it is handed, as lalonde_ate standardised them."""

    def __init__(self) -> None:
        self.covariates: dict[str, np.ndarray] = {}

    def estimate(self, data, *, treatment="u", outcome="x_next", covariates=()):
        self.covariates = {name: np.asarray(data[name]) for name in covariates}
        return EffectEstimate(0.0)


@pytest.mark.parametrize("units", [1e-13, 1.0, 1e6])
def test_each_covariate_reaches_the_estimator_at_unit_spread_in_any_units(units: float) -> None:
    """Each covariate is standardised in its own units: a floor of 1e-9 added to its spread was in
    the caller's units, so earnings with a spread of 5,000 dollars, logged in units of 1e13
    dollars, reached the estimator at a third of their spread, and ages at 1 - 2e-10 of theirs. A
    covariate at one level, 0.1 here, reaches it at 0 in every row, where its mean's rounding over
    that floor left it at 3.5e-6; so does one apart from that level by 16 eps, its rounding, which
    divided by its own spread would reach it at unit size."""
    rng = np.random.default_rng(1)
    rows = 2000
    eps = np.finfo(np.float64).eps
    data = LalondeData(
        treatment=(rng.random(rows) < 0.3).astype(float),
        outcome=rng.normal(5000.0, 1000.0, rows),
        covariates={
            "re74": units * rng.normal(1e4, 5e3, rows),
            "age": rng.normal(30.0, 5.0, rows),
            "still": np.full(rows, 0.1),
            "rounding": 0.1 * (1.0 + 16 * eps * np.where(rng.random(rows) < 0.5, -1.0, 1.0)),
        },
        experimental_ate=0.0,
    )
    recorded = _Recorded()
    lalonde_ate(data, recorded)
    for name in ("re74", "age"):
        column = recorded.covariates[name]
        assert float(np.mean(column)) == pytest.approx(0.0, abs=1e-12), name
        assert float(np.std(column)) == pytest.approx(1.0, rel=1e-12, abs=0.0), name
    for name in ("still", "rounding"):
        assert np.all(recorded.covariates[name] == 0.0), name
