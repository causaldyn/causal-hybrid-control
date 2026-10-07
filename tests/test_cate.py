"""R-learner: recovers a heterogeneous effect tau(x) under confounding; a naive regression can't."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chc.estimators import RLearner


def _heterogeneous_confounded(n: int, seed: int) -> tuple[dict, jnp.ndarray]:
    keys = jax.random.split(jax.random.key(seed), 5)
    x0 = jax.random.normal(keys[0], (n,))
    x1 = jax.random.normal(keys[1], (n,))
    z = jax.random.normal(keys[2], (n,))  # confounder: drives both treatment and outcome
    u = 1.5 * z + 0.5 * jax.random.normal(keys[3], (n,))  # treatment confounded by z
    true_tau = 1.0 + 0.8 * x0  # the effect of u depends on x0 (heterogeneous)
    x_next = true_tau * u + 2.0 * z + 0.1 * jax.random.normal(keys[4], (n,))
    return {"x0": x0, "x1": x1, "z": z, "u": u, "x_next": x_next}, true_tau


def test_rlearner_recovers_heterogeneous_effect_under_confounding() -> None:
    data, true_tau = _heterogeneous_confounded(8000, seed=0)
    covariates = ("x0", "x1", "z")
    estimate = RLearner(degree=3, cate_degree=1).estimate(data, covariates=covariates)
    covs = jnp.stack([data[c] for c in covariates], axis=1)
    predicted = estimate.cate(covs)
    assert abs(estimate.effect - 1.0) < 0.1  # ATE recovered (true 1.0)
    assert float(jnp.corrcoef(predicted, true_tau)[0, 1]) > 0.98  # the CATE tracks the truth
    assert float(jnp.sqrt(jnp.mean((predicted - true_tau) ** 2))) < 0.1  # low pointwise CATE error


def test_rlearner_beats_naive_treatment_regression() -> None:
    data, _ = _heterogeneous_confounded(8000, seed=1)
    y, u = data["x_next"], data["u"]
    naive_ate = float(jnp.sum((y - y.mean()) * (u - u.mean())) / jnp.sum((u - u.mean()) ** 2))
    r_ate = RLearner().estimate(data, covariates=("x0", "x1", "z")).effect
    assert abs(r_ate - 1.0) < abs(naive_ate - 1.0)  # residualisation de-confounds; naive does not


@pytest.mark.parametrize("units", [1e-9, 1e-3, 1e3, 1e6])
def test_the_r_learner_reads_the_same_with_the_covariates_in_any_units(units: float) -> None:
    """The ridge was a constant on the Gram of the raw covariates' monomials, in the nuisances and
    in the R-loss: logged in thousandths of their units, the covariates read an average effect of
    2.10 where they read 0.996, and in millionths 2.209, the naive regression's."""
    data, _ = _heterogeneous_confounded(8000, seed=0)
    covariates = ("x0", "x1", "z")
    scaled = {**data, **{name: data[name] * units for name in covariates}}
    one = RLearner().estimate(data, covariates=covariates)
    other = RLearner().estimate(scaled, covariates=covariates)
    assert one.cate is not None
    assert other.cate is not None
    assert other.effect == pytest.approx(one.effect, rel=1e-9, abs=0.0)
    covs = jnp.stack([data[name] for name in covariates], axis=1)
    np.testing.assert_allclose(other.cate(covs * units), one.cate(covs), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("level", [1e3, 1e6])
def test_the_r_learner_reads_the_same_with_a_covariate_at_any_level(level: float) -> None:
    """The bases of the nuisances and of ``tau`` were the raw covariates' monomials, which a
    covariate far from zero makes nearly collinear under the ridge: with the effect modifier logged
    about 1e3 rather than about 0, the effect's slope in it read 0.13 where it reads 0.80, and about
    1e6, 0.0003."""
    data, _ = _heterogeneous_confounded(8000, seed=0)
    covariates = ("x0", "x1", "z")
    one = RLearner().estimate(data, covariates=covariates)
    other = RLearner().estimate({**data, "x0": data["x0"] + level}, covariates=covariates)
    assert one.cate is not None
    assert other.cate is not None
    assert other.effect == pytest.approx(one.effect, rel=1e-9, abs=0.0)
    covs = jnp.stack([data[name] for name in covariates], axis=1)
    shifted = covs.at[:, 0].add(level)
    # logged about the level, the modifier itself is rounded to level * eps
    np.testing.assert_allclose(other.cate(shifted), one.cate(covs), rtol=1e-9, atol=1e-15 * level)


@pytest.mark.parametrize("units", [1e-9, 1e-3, 1e3, 1e6])
def test_the_r_learner_reads_the_same_with_the_treatment_in_any_units(units: float) -> None:
    """The R-loss's ridge was a constant on a Gram in the treatment's units squared: logged in
    thousandths of its units, the treatment read an average effect of 0.167 where it reads 0.996,
    and in millionths 2e-7."""
    data, _ = _heterogeneous_confounded(8000, seed=0)
    covariates = ("x0", "x1", "z")
    one = RLearner().estimate(data, covariates=covariates)
    other = RLearner().estimate({**data, "u": data["u"] * units}, covariates=covariates)
    assert one.cate is not None
    assert other.cate is not None
    assert other.effect * units == pytest.approx(one.effect, rel=1e-9, abs=0.0)
    covs = jnp.stack([data[name] for name in covariates], axis=1)
    np.testing.assert_allclose(other.cate(covs) * units, one.cate(covs), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("name", ["x_next", "u"])
def test_the_r_learner_reads_the_same_with_the_outcome_or_the_treatment_far_from_zero(
    name: str,
) -> None:
    """The nuisances' ridge was on their intercept too, which left a share of a level in the
    residuals: logged 1e3 of its spreads from zero, the outcome read an average effect of 0.99619
    where it reads 0.99607, and the treatment 0.99587. Moved there and into millionths of its
    units, the column keeps its values to 1e3 eps of its spread, 2e-13, and the average effect now
    moves by at most 1e-14, the CATE by 5e-14."""
    data, _ = _heterogeneous_confounded(8000, seed=0)
    covariates = ("x0", "x1", "z")
    column = data[name]
    per_unit = 1e-6 if name == "x_next" else 1e6
    one = RLearner().estimate(data, covariates=covariates)
    other = RLearner().estimate(
        {**data, name: (column + 1e3 * jnp.std(column)) * 1e-6}, covariates=covariates
    )
    assert one.cate is not None
    assert other.cate is not None
    assert other.effect / per_unit == pytest.approx(one.effect, rel=1e-11, abs=0.0)
    covs = jnp.stack([data[c] for c in covariates], axis=1)
    np.testing.assert_allclose(other.cate(covs) / per_unit, one.cate(covs), rtol=1e-11, atol=1e-12)


def test_a_column_of_zeros_among_the_covariates_moves_no_r_learner_estimate() -> None:
    """A covariate logged at zero in every row has no mean square to scale its ridge term by, and
    without a term of its own the R-loss's Gram is singular there."""
    data, _ = _heterogeneous_confounded(8000, seed=0)
    covariates = ("x0", "x1", "z")
    one = RLearner().estimate(data, covariates=covariates)
    other = RLearner().estimate({**data, "zero": jnp.zeros(8000)}, covariates=(*covariates, "zero"))
    assert one.cate is not None
    assert other.cate is not None
    assert other.effect == pytest.approx(one.effect, rel=1e-12, abs=0.0)
    covs = jnp.stack([data[name] for name in covariates], axis=1)
    padded = jnp.concatenate([covs, jnp.zeros((8000, 1))], axis=1)
    np.testing.assert_allclose(other.cate(padded), one.cate(covs), rtol=1e-12, atol=1e-12)
