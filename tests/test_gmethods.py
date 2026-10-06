"""Sequential g-computation recovers a time-varying effect that naive adjustment misses."""

from __future__ import annotations

import numpy as np
import pytest

from chc.gmethods import naive_pooled_effect, sequential_g_formula

TH0, TH1, LAM0, LAM1, GAM, DELTA = 1.0, 1.5, 0.5, 0.8, 1.0, 0.7
TRUE_EFFECT = (TH0 + LAM1 * GAM) + TH1  # (1,1)-(0,0): a0 total (direct + via L1) + a1 = 3.3
SPEC = {"treatments": ("a0", "a1"), "confounders": (("l0",), ("l1",)), "outcome": "y"}


def _time_varying_confounded(n: int, seed: int) -> dict[str, np.ndarray]:
    """A0 affects the confounder L1, which drives A1 and Y -- the g-methods failure mode."""
    rng = np.random.default_rng(seed)
    l0 = rng.normal(0.0, 1.0, n)
    a0 = 0.9 * l0 + rng.normal(0.0, 1.0, n)
    l1 = GAM * a0 + DELTA * l0 + rng.normal(0.0, 1.0, n)  # A0 -> L1 (confounder on the causal path)
    a1 = 1.1 * l1 + 0.3 * a0 + rng.normal(0.0, 1.0, n)  # L1 -> A1 (time-varying confounding)
    y = TH0 * a0 + TH1 * a1 + LAM0 * l0 + LAM1 * l1 + rng.normal(0.0, 0.3, n)
    return {"a0": a0, "a1": a1, "l0": l0, "l1": l1, "y": y}


def test_g_formula_recovers_the_time_varying_effect() -> None:
    data = _time_varying_confounded(40_000, seed=0)
    effect = sequential_g_formula(data, regime=(1.0, 1.0), baseline=(0.0, 0.0), **SPEC)
    assert effect == pytest.approx(TRUE_EFFECT, abs=0.1)  # standardising over L1 recovers the truth


def test_naive_adjustment_is_biased_by_the_mediator_confounder() -> None:
    data = _time_varying_confounded(40_000, seed=1)
    naive = naive_pooled_effect(data, **SPEC)
    assert abs(naive - TRUE_EFFECT) > 0.5  # conditioning on L1 underestimates A0's total effect


def test_g_formula_beats_the_naive_pooled_regression() -> None:
    data = _time_varying_confounded(40_000, seed=2)
    g = sequential_g_formula(data, regime=(1.0, 1.0), baseline=(0.0, 0.0), **SPEC)
    naive = naive_pooled_effect(data, **SPEC)
    assert abs(g - TRUE_EFFECT) < abs(naive - TRUE_EFFECT)  # g-formula is the less-biased estimator


def test_mismatched_horizon_raises() -> None:
    data = _time_varying_confounded(200, seed=3)
    with pytest.raises(ValueError, match="horizon"):
        sequential_g_formula(
            data,
            treatments=("a0", "a1"),
            confounders=(("l0",),),
            outcome="y",
            regime=(1.0, 1.0),
            baseline=(0.0, 0.0),
        )


@pytest.mark.parametrize("folds", [0, 1, 201, 2.0, True])
def test_folds_outside_two_to_the_rows_are_refused(folds) -> None:
    """One fold trains on nothing and returned 0; none left every row unwritten and returned what
    the empty array held."""
    data = _time_varying_confounded(200, seed=4)
    with pytest.raises(ValueError, match=r"folds=.*from 2 to the rows \(200\)"):
        sequential_g_formula(data, regime=(1.0, 1.0), baseline=(0.0, 0.0), folds=folds, **SPEC)


@pytest.mark.parametrize("folds", [2, np.int64(3), 200])
def test_folds_from_two_to_the_rows_are_accepted(folds) -> None:
    data = _time_varying_confounded(200, seed=4)
    effect = sequential_g_formula(data, regime=(1.0, 1.0), baseline=(0.0, 0.0), folds=folds, **SPEC)
    assert np.isfinite(effect)


G_SECOND_ROLES = {
    "a treatment among its own step's confounders": (
        {"confounders": (("l0", "a0"), ("l1",))},
        "the treatment 'a0' is set at step 0 and is a confounder measured before step 0",
    ),
    "a later treatment among an earlier step's confounders": (
        {"confounders": (("l0", "a1"), ("l1",))},
        "the treatment 'a1' is set at step 1 and is a confounder measured before step 0",
    ),
    "the outcome among the confounders": (
        {"confounders": (("l0",), ("l1", "y"))},
        "the column 'y' is read as the outcome and as a confounder",
    ),
    "the outcome as a treatment": (
        {"treatments": ("a0", "y")},
        "the column 'y' is read as the outcome and as a treatment",
    ),
    "a treatment twice": (
        {"treatments": ("a0", "a0")},
        "the column 'a0' is read twice as a treatment",
    ),
}


@pytest.mark.parametrize(("names", "match"), G_SECOND_ROLES.values(), ids=G_SECOND_ROLES.keys())
def test_the_g_formula_refuses_a_column_in_two_roles(names: dict, match: str) -> None:
    """A treatment among the confounders of its own step was set in one column of the design and
    kept at its logged value in the other, and the outcome among them explained itself."""
    data = _time_varying_confounded(200, seed=5)
    spec = SPEC | names
    with pytest.raises(ValueError, match=match):
        sequential_g_formula(data, regime=(1.0, 1.0), baseline=(0.0, 0.0), **spec)


def test_an_earlier_treatment_may_confound_a_later_one() -> None:
    """Measured before the later treatment, the earlier one is one of its confounders: it is set
    at its own step, before the later step's design reads it."""
    data = _time_varying_confounded(40_000, seed=0)
    spec = SPEC | {"confounders": (("l0",), ("l1", "a0"))}
    effect = sequential_g_formula(data, regime=(1.0, 1.0), baseline=(0.0, 0.0), **spec)
    assert effect == pytest.approx(TRUE_EFFECT, abs=0.1)


@pytest.mark.parametrize(
    ("names", "match"),
    [
        (
            {"confounders": (("l0",), ("l1", "a0"))},
            r"the treatments \['a0'\] are also confounders",
        ),
        (
            {"confounders": (("l0", "y"), ("l1",))},
            "the column 'y' is read as the outcome and as a confounder",
        ),
    ],
)
def test_the_pooled_regression_refuses_a_treatment_or_the_outcome_among_the_confounders(
    names: dict, match: str
) -> None:
    """Pooled, a treatment that is also a confounder splits its coefficient with its own copy, and
    the sum of the treatments' coefficients misses the copy's share."""
    data = _time_varying_confounded(200, seed=6)
    with pytest.raises(ValueError, match=match):
        naive_pooled_effect(data, **(SPEC | names))
