"""Estimator adapters: one Strategy interface, swappable causal backends recover the true effect."""

import importlib
import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chc.causal import ConfoundedLinearSystem, dml_point_and_se, estimate_effect_iv
from chc.estimators import (
    IV2SLS,
    BackdoorOLS,
    CausalEffectEstimator,
    DoubleML,
    DoWhyEstimator,
    EconMLDoubleML,
    EffectEstimate,
    RLearner,
)

# Public as jax.enable_x64 from jax 0.8.0; the floor, 0.4.30, has only jax.experimental.enable_x64,
# which jax 0.11 no longer has.
if hasattr(jax, "enable_x64"):
    enable_x64 = jax.enable_x64
else:
    enable_x64 = importlib.import_module("jax.experimental").enable_x64


def _data(**kw) -> dict[str, jax.Array]:
    return ConfoundedLinearSystem(**kw).sample(20_000, jax.random.key(0))


def test_builtins_satisfy_the_estimator_protocol() -> None:
    for est in (BackdoorOLS(), IV2SLS(), DoubleML()):
        assert isinstance(est, CausalEffectEstimator)


def test_backdoor_ols_recovers_effect_when_adjusting_for_confounder() -> None:
    result = BackdoorOLS().estimate(_data(), covariates=("x", "z"))
    assert isinstance(result, EffectEstimate)
    assert abs(result.effect - 1.0) < 0.05  # true b_true = +1.0
    assert result.std_error is not None
    assert result.std_error > 0.0


def test_backdoor_ols_is_confounded_without_the_confounder() -> None:
    result = BackdoorOLS().estimate(_data(), covariates=("x",))  # omit z
    assert result.effect < 0.0  # sign-flipped, like the naive fit


def test_double_ml_recovers_effect() -> None:
    result = DoubleML().estimate(_data(), covariates=("x", "z"))
    assert abs(result.effect - 1.0) < 0.1


def test_double_ml_reports_a_covering_confidence_interval() -> None:
    result = DoubleML().estimate(_data(), covariates=("x", "z"))
    assert result.std_error is not None  # ships an influence-function SE
    assert result.std_error > 0.0
    lo, hi = result.diagnostics["ci95_low"], result.diagnostics["ci95_high"]
    assert lo < 1.0 < hi  # the 95% CI covers the true effect b_true = 1.0


def test_dml_influence_function_ci_has_near_nominal_coverage() -> None:
    trials, covered = 30, 0
    for s in range(trials):
        data = ConfoundedLinearSystem().sample(4000, jax.random.key(100 + s))
        theta, se = dml_point_and_se(data, covariates=("x", "z"), degree=2, folds=5)
        covered += abs(theta - 1.0) < 1.96 * se
    assert covered >= 0.8 * trials  # the sandwich influence-function CI ~ nominal 95% coverage


def test_iv_recovers_effect_with_latent_confounder() -> None:
    result = IV2SLS(instrument="w").estimate(_data(gamma=1.0))
    assert abs(result.effect - 1.0) < 0.1


def _iv_data() -> dict[str, jax.Array]:
    return ConfoundedLinearSystem(gamma=1.0).sample(40_000, jax.random.key(0))


def _partial_correlation(state: jax.Array, action: jax.Array, instrument: jax.Array) -> float:
    """The action's and the instrument's correlation, each less its least-squares projection on a
    constant and the state, in absolute value, by numpy."""
    design = np.column_stack([np.ones(state.shape[0]), np.asarray(state)])

    def less_state(column: jax.Array) -> np.ndarray:
        values = np.asarray(column)
        return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]

    a, i = less_state(action), less_state(instrument)
    return float(abs(a @ i) / (np.linalg.norm(a) * np.linalg.norm(i)))


@pytest.mark.parametrize(
    "instrument",
    [
        lambda d: jnp.zeros_like(d["w"]),
        lambda d: 2.0 * d["x"] + 3.0,
    ],
    ids=["an instrument of zeros", "the state's affine copy"],
)
def test_iv_refuses_an_instrument_that_moves_the_treatment_nowhere_beyond_the_state(
    instrument,
) -> None:
    """The adapter reads the estimate through :func:`chc.causal.estimate_effect_iv`, and its
    refusal with it, under the caller's names. It returned -0.0032 for both, where the truth is
    1.0."""
    data = _iv_data()
    named = {"x": data["x"], "spend": data["u"], "sales": data["x_next"], "w": instrument(data)}
    with pytest.raises(ValueError, match="does not move the action beyond what the state explains"):
        IV2SLS().estimate(named, treatment="spend", outcome="sales")


def test_iv_reports_the_instrument_s_relevance_in_any_units() -> None:
    """The relevance is the partial correlation of the treatment and the instrument given the
    state, which the estimate rests on: 0.53 for the plant's instrument. A column of noise drawn
    apart from the action moves it by chance alone, so 2SLS's moment keeps its rank, and the
    estimate reads 0.263 where the truth is 1.0: the relevance, 0.0087, is what tells the two
    apart. It grades an instrument and is not a test. Neither moves with the units of the
    instrument, the treatment or the state, or with their signs."""
    data = _iv_data()
    noise = {**data, "w": jax.random.normal(jax.random.key(5), data["w"].shape)}
    strong, weak = IV2SLS().estimate(data), IV2SLS().estimate(noise)
    assert strong.effect == float(estimate_effect_iv(data, instrument="w"))
    assert weak.effect == pytest.approx(0.263, abs=5e-4)
    assert strong.diagnostics["instrument_relevance"] == pytest.approx(0.5307, abs=1e-4)
    assert weak.diagnostics["instrument_relevance"] == pytest.approx(0.0087, abs=1e-4)
    for log, result in ((data, strong), (noise, weak)):
        relevance = result.diagnostics["instrument_relevance"]
        assert relevance == pytest.approx(
            _partial_correlation(log["x"], log["u"], log["w"]), rel=1e-12, abs=0.0
        )
        for column, factor in itertools.product(("w", "u", "x"), (1e-6, 1e6, -1.0)):
            moved = IV2SLS().estimate({**log, column: log[column] * factor})
            assert moved.diagnostics["instrument_relevance"] == pytest.approx(
                relevance, rel=1e-12, abs=0.0
            ), (column, factor)


@pytest.mark.parametrize("units", [1e-9, 1e-3, 1e3, 1e6])
def test_double_ml_reads_the_same_with_the_covariates_in_any_units(units: float) -> None:
    """The nuisances' ridge, 1.0 here, was a constant on the Gram of the raw covariates' monomials:
    logged in thousandths of their units, the covariates read an effect of -0.201 where they read
    1.004, the unadjusted regression's -0.200."""
    data = ConfoundedLinearSystem().sample(2_000, jax.random.key(0))
    one = DoubleML().estimate(data)
    other = DoubleML().estimate({**data, "x": data["x"] * units, "z": data["z"] * units})
    assert other.effect == pytest.approx(one.effect, rel=1e-9, abs=0.0)
    assert other.std_error == pytest.approx(one.std_error, rel=1e-9, abs=0.0)


@pytest.mark.parametrize("name", ["x_next", "u"])
def test_double_ml_reads_the_same_with_the_outcome_or_the_treatment_far_from_zero(
    name: str,
) -> None:
    """The nuisances' ridge, 1.0 here, was on their intercept too, which left a share of a level in
    the residuals: logged 1e3 of its spreads from zero, the treatment read an effect of 0.107
    where it reads 1.004, and the outcome a standard error 8.3 times its own. Moved there and into
    millionths of its units, the column keeps its values to 1e3 eps of its spread, 2e-13, and the
    estimates now move by at most 2e-13."""
    data = ConfoundedLinearSystem().sample(2_000, jax.random.key(0))
    column = data[name]
    per_unit = 1e-6 if name == "x_next" else 1e6
    one = DoubleML().estimate(data)
    other = DoubleML().estimate({**data, name: (column + 1e3 * column.std()) * 1e-6})
    assert one.std_error is not None
    assert other.std_error is not None
    assert other.effect / per_unit == pytest.approx(one.effect, rel=1e-11, abs=0.0)
    assert other.std_error / per_unit == pytest.approx(one.std_error, rel=1e-11, abs=0.0)


@pytest.mark.parametrize("units", [1e-9, 1e-6, 1e-3, 1e3, 1e6])
def test_backdoor_ols_and_its_error_read_the_same_with_the_treatment_in_any_units_in_float32(
    units: float,
) -> None:
    """In float32 the least-squares cutoff dropped a treatment logged in millionths of its units:
    the effect read 0 where it reads 1.0015, with a standard error ten times its own. Logged in
    millions, the treatment pushed the other columns under the cutoff, and the effect read 0.035."""
    with enable_x64(False):
        data = ConfoundedLinearSystem(gamma=0.8).sample(2_000, jax.random.key(0))
        one = BackdoorOLS().estimate(data)
        other = BackdoorOLS().estimate({**data, "u": data["u"] * units})
    assert one.std_error is not None
    assert other.std_error is not None
    assert other.effect * units == pytest.approx(one.effect, rel=1e-4, abs=0.0)
    assert other.std_error * units == pytest.approx(one.std_error, rel=1e-4, abs=0.0)


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


def test_dowhy_adapter_raises_actionable_error_when_uninstalled() -> None:
    """The DoWhy adapter is lazy-imported too: fail loudly with a hint, never a hard dependency."""
    with pytest.raises(ImportError, match="dowhy"):
        DoWhyEstimator().estimate(_data())


def test_a_nan_standard_error_reads_a_nan_statistic() -> None:
    """``se > 0`` is false for a nan, so a nan outcome read a t-statistic of inf, the strongest
    evidence there is, for an effect that was nan."""
    data = dict(_data())
    data["x_next"] = data["x_next"].at[5].set(float("nan"))
    result = DoubleML().estimate(data, covariates=("x", "z"))
    assert math.isnan(result.diagnostics["t_stat"])


SECOND_ROLES = {
    "the treatment as a covariate": (
        BackdoorOLS(),
        {"covariates": ("x", "z", "u")},
        "the column 'u' is read as the treatment and as a covariate",
    ),
    "the outcome as a covariate": (
        BackdoorOLS(),
        {"covariates": ("x", "z", "x_next")},
        "the column 'x_next' is read as the outcome and as a covariate",
    ),
    "the treatment as the outcome": (
        BackdoorOLS(),
        {"outcome": "u"},
        "the column 'u' is read as the treatment and as the outcome",
    ),
    "a covariate twice": (
        BackdoorOLS(),
        {"covariates": ("x", "z", "z")},
        "the column 'z' is read twice as a covariate",
    ),
    "the treatment as its instrument": (
        IV2SLS(instrument="u"),
        {},
        "the column 'u' is read as the treatment and as the instrument",
    ),
    "2SLS's treatment as a covariate": (
        IV2SLS(),
        {"treatment": "x"},
        "the column 'x' is read as the treatment and as a covariate",
    ),
    "2SLS's instrument as a covariate": (
        IV2SLS(),
        {"covariates": ("x", "w")},
        "the column 'w' is read as the instrument and as a covariate",
    ),
    "2SLS's outcome as a covariate": (
        IV2SLS(),
        {"covariates": ("x", "x_next")},
        "the column 'x_next' is read as the outcome and as a covariate",
    ),
    "DML's treatment as a covariate": (
        DoubleML(),
        {"covariates": ("x", "z", "u")},
        "the column 'u' is read as the treatment and as a covariate",
    ),
    "DML's outcome as a covariate": (
        DoubleML(),
        {"covariates": ("x", "z", "x_next")},
        "the column 'x_next' is read as the outcome and as a covariate",
    ),
    "the R-learner's treatment as a covariate": (
        RLearner(),
        {"covariates": ("x", "z", "u")},
        "the column 'u' is read as the treatment and as a covariate",
    ),
    "the R-learner's outcome as a covariate": (
        RLearner(),
        {"covariates": ("x", "z", "x_next")},
        "the column 'x_next' is read as the outcome and as a covariate",
    ),
    "EconML's treatment as a covariate": (
        EconMLDoubleML(),
        {"covariates": ("x", "z", "u")},
        "the column 'u' is read as the treatment and as a covariate",
    ),
    "DoWhy's outcome as a covariate": (
        DoWhyEstimator(),
        {"covariates": ("x", "z", "x_next")},
        "the column 'x_next' is read as the outcome and as a covariate",
    ),
}


@pytest.mark.parametrize(
    ("estimator", "names", "match"), SECOND_ROLES.values(), ids=SECOND_ROLES.keys()
)
def test_a_column_named_in_two_roles_is_refused(
    estimator: CausalEffectEstimator, names: dict, match: str
) -> None:
    """Each role reads the column its name picks, so a column named in two was read in both: the
    treatment among its covariates was partialled out of itself, and the outcome among them
    explained itself. The refusal comes before an optional backend's import."""
    data = ConfoundedLinearSystem(gamma=1.0).sample(500, jax.random.key(0))
    with pytest.raises(ValueError, match=match):
        estimator.estimate(data, **names)


def _heteroskedastic(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """A confounded log whose noise grows with the treatment: ``y = 0.5 x + u + 2 z + e``, the
    noise's SD ``0.2 + 1.5 |u|``."""
    x, z = rng.standard_normal(n), rng.standard_normal(n)
    u = -z + rng.standard_normal(n)
    y = 0.5 * x + u + 2.0 * z + rng.standard_normal(n) * (0.2 + 1.5 * np.abs(u))
    return {"x": x, "z": z, "u": u, "x_next": y}


def _sandwich(design: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """HC3 standard errors by numpy, an intercept appended to ``design``: each row's residual over
    one less its leverage."""
    full = np.column_stack([design, np.ones(design.shape[0])])
    bread = np.linalg.inv(full.T @ full)
    leverage = np.einsum("ij,jk,ik->i", full, bread, full)
    meat = (full * (residual / (1.0 - leverage))[:, None] ** 2).T @ full
    return np.sqrt(np.diag(bread @ meat @ bread))


def test_backdoor_ols_reports_the_hc3_error() -> None:
    """The error is the heteroskedasticity-robust HC3 sandwich, read against numpy's."""
    data = _heteroskedastic(np.random.default_rng(3), 400)
    design = np.column_stack([data["u"], data["x"], data["z"]])
    full = np.column_stack([design, np.ones(400)])
    residual = data["x_next"] - full @ np.linalg.lstsq(full, data["x_next"], rcond=None)[0]
    result = BackdoorOLS().estimate(data, covariates=("x", "z"))
    assert result.std_error == pytest.approx(_sandwich(design, residual)[0], rel=1e-10, abs=0.0)


def test_backdoor_ols_error_covers_under_heteroskedastic_noise() -> None:
    """Under noise that grows with the treatment the classical error read 0.114 against a Monte
    Carlo spread of 0.164, and its 95% interval covered the effect in 82.5% of 400 draws. Over 200
    draws the coverage's Monte Carlo SD is 0.015 at 0.95, so 0.90 is three of them below."""
    rng = np.random.default_rng(1)
    covered = 0
    for _ in range(200):
        result = BackdoorOLS().estimate(_heteroskedastic(rng, 400), covariates=("x", "z"))
        assert result.std_error is not None
        covered += abs(result.effect - 1.0) <= 1.96 * result.std_error
    assert covered >= 0.90 * 200


def _instrumented(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """An instrument valid only given ``c``: ``w`` moves with ``c``, which drives the outcome, and
    ``z`` confounds the treatment and the outcome unseen. The effect is 1."""
    x, z, c = rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n)
    w = c + rng.standard_normal(n)
    u = 0.8 * w - z + 0.7 * c + 0.5 * rng.standard_normal(n)
    noise = rng.standard_normal(n) * (0.3 + 0.5 * np.abs(w))
    y = 0.5 * x + u + 2.0 * z + 1.5 * c + noise
    return {"x": x, "z": z, "c": c, "w": w, "u": u, "x_next": y}


def test_iv_conditions_both_stages_on_the_covariates_it_is_given() -> None:
    """``covariates`` was not read: both stages conditioned on ``x`` whatever was passed, so an
    instrument valid only given ``c`` read 1.63 with ``c`` named, where the effect is 1."""
    data = _instrumented(np.random.default_rng(4), 20_000)
    adjusted = IV2SLS().estimate(data, covariates=("x", "c"))
    assert adjusted.effect == pytest.approx(1.0, abs=0.05)
    assert IV2SLS().estimate(data, covariates=("x",)).effect > 1.5  # c left out: biased


def test_iv_reports_the_2sls_hc3_error() -> None:
    """The error is the HC3 sandwich on the second stage's design, its leverage read there and its
    residual the outcome less the fit on the treatment itself, read against numpy's; IV2SLS
    reported none."""
    data = _instrumented(np.random.default_rng(5), 1_000)
    controls = np.column_stack([data["x"], data["c"]])
    first = np.column_stack([controls, data["w"], np.ones(1_000)])
    pushed = first @ np.linalg.lstsq(first, data["u"], rcond=None)[0]
    second = np.column_stack([controls, pushed, np.ones(1_000)])
    coeffs = np.linalg.lstsq(second, data["x_next"], rcond=None)[0]
    structural = np.column_stack([controls, data["u"], np.ones(1_000)])
    residual = data["x_next"] - structural @ coeffs
    expected = _sandwich(np.column_stack([controls, pushed]), residual)[2]
    result = IV2SLS().estimate(data, covariates=("x", "c"))
    assert result.std_error == pytest.approx(expected, rel=1e-8, abs=0.0)


def test_iv_error_covers_the_effect() -> None:
    """Over 200 draws of 1,000 rows the 95% interval's coverage has a Monte Carlo SD of 0.015 at
    0.95, so 0.90 is three of them below."""
    rng = np.random.default_rng(6)
    covered = 0
    for _ in range(200):
        result = IV2SLS().estimate(_instrumented(rng, 1_000), covariates=("x", "c"))
        assert result.std_error is not None
        covered += abs(result.effect - 1.0) <= 1.96 * result.std_error
    assert covered >= 0.90 * 200


def _leveraged(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """``y = 0.9 x + 0.8 u + 0.7 z + e`` with ``u = 0.5 x + 0.8 z + a``, ``a`` a t(3) draw over
    its SD and the noise's SD ``|a|``: a few rows carry high leverage and the most noise."""
    x, z, e = rng.standard_normal((3, n))
    a = rng.standard_t(3, n) / np.sqrt(3.0)
    u = 0.5 * x + 0.8 * z + a
    return {"x": x, "z": z, "u": u, "x_next": 0.9 * x + 0.8 * u + 0.7 * z + e * np.abs(a)}


def test_backdoor_ols_error_covers_where_a_few_rows_carry_high_leverage() -> None:
    """A high-leverage row keeps a small residual whatever its noise, so HC1's interval missed the
    effect 15.7 % of 4000 such logs of 100 rows; HC3's, which scales each residual by one less its
    leverage, 8.1 %. Over 300 draws the miss rate's Monte Carlo SD is 0.016 at 0.081, so 0.12 is
    about 2.4 of them above."""
    rng = np.random.default_rng(8)
    misses = 0
    for _ in range(300):
        result = BackdoorOLS().estimate(_leveraged(rng, 100), covariates=("x", "z"))
        assert result.std_error is not None
        misses += abs(result.effect - 0.8) > 1.959964 * result.std_error
    assert misses <= 0.12 * 300
