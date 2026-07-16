"""Estimator adapters: one Strategy interface, swappable causal backends recover the true effect."""

import jax
import pytest

from chc.causal import ConfoundedLinearSystem
from chc.estimators import (
    IV2SLS,
    BackdoorOLS,
    CausalEffectEstimator,
    DoubleML,
    EconMLDoubleML,
    EffectEstimate,
)


def _data(**kw) -> dict[str, jax.Array]:
    return ConfoundedLinearSystem(**kw).sample(20_000, jax.random.key(0))


def test_builtins_satisfy_the_estimator_protocol() -> None:
    for est in (BackdoorOLS(), IV2SLS(), DoubleML()):
        assert isinstance(est, CausalEffectEstimator)


def test_backdoor_ols_recovers_effect_when_adjusting_for_confounder() -> None:
    result = BackdoorOLS().estimate(_data(), covariates=("x", "z"))
    assert isinstance(result, EffectEstimate)
    assert abs(result.effect - 1.0) < 0.05  # true b_true = +1.0
    assert result.std_error is not None and result.std_error > 0.0


def test_backdoor_ols_is_confounded_without_the_confounder() -> None:
    result = BackdoorOLS().estimate(_data(), covariates=("x",))  # omit z
    assert result.effect < 0.0  # sign-flipped, like the naive fit


def test_double_ml_recovers_effect() -> None:
    result = DoubleML().estimate(_data(), covariates=("x", "z"))
    assert abs(result.effect - 1.0) < 0.1


def test_iv_recovers_effect_with_latent_confounder() -> None:
    result = IV2SLS(instrument="w").estimate(_data(gamma=1.0))
    assert abs(result.effect - 1.0) < 0.1


def test_backends_are_swappable_behind_one_interface() -> None:
    """The point of the refactor: control loops over estimators, blind to which backend."""
    data = _data()
    estimators: list[CausalEffectEstimator] = [BackdoorOLS(), DoubleML()]
    effects = [e.estimate(data, covariates=("x", "z")).effect for e in estimators]
    assert all(abs(b - 1.0) < 0.1 for b in effects)


def test_econml_adapter_raises_actionable_error_when_uninstalled() -> None:
    """The optional adapter must fail loudly with an install hint, never a hard dependency."""
    with pytest.raises(ImportError, match="econml"):
        EconMLDoubleML().estimate(_data())
