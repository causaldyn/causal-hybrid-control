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
            data, treatments=("a0", "a1"), confounders=(("l0",),), outcome="y",
            regime=(1.0, 1.0), baseline=(0.0, 0.0),
        )
