"""A causal-methods leaderboard: each modern estimator vs the naive built-in it is meant to beat.

Every method in the causal frontier (`chc.did`, `chc.scm`, `chc.estimators`, `chc.gmethods`) ships
with a test proving it recovers a *known* effect where a naive baseline is biased. This consolidates
those into one table -- the moat, scored: for each method it draws a self-contained synthetic DGP
with a ground-truth effect, runs the method and its naive baseline, and reports both biases so the
win is visible side by side. NumPy orchestration; the R-learner row builds a JAX payload internally.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from chc.did import callaway_santanna, de_chaisemartin, twoway_fixed_effects_att
from chc.estimators import RLearner
from chc.gmethods import naive_pooled_effect, sequential_g_formula
from chc.scm import augmented_synthetic_control, synthetic_control


@dataclass(frozen=True)
class CausalBenchRow:
    """One method-vs-baseline comparison on a DGP with a known effect."""

    method: str
    baseline: str
    truth: float
    estimate: float
    baseline_estimate: float

    @property
    def method_bias(self) -> float:
        return abs(self.estimate - self.truth)

    @property
    def baseline_bias(self) -> float:
        return abs(self.baseline_estimate - self.truth)


def _staggered_panel(rng: np.random.Generator, delta: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    n_per, n_periods = 300, 8
    group = np.array([3] * n_per + [5] * n_per + [-1] * n_per)
    unit = rng.normal(0.0, 1.0, (group.size, 1))
    trend = (0.3 * np.arange(n_periods))[None, :]
    outcomes = unit + trend + rng.normal(0.0, 0.3, (group.size, n_periods))
    for i, g in enumerate(group):
        if g >= 0:
            for t in range(g, n_periods):
                outcomes[i, t] += delta * (t - g + 1)
    return outcomes, group


def _did_rows(rng: np.random.Generator, delta: float = 0.5) -> list[CausalBenchRow]:
    outcomes, group = _staggered_panel(rng, delta)
    n_periods = outcomes.shape[1]
    post = [delta * (t - g + 1) for i, g in enumerate(group) if g >= 0 for t in range(g, n_periods)]
    twfe = twoway_fixed_effects_att(outcomes, group)
    return [
        CausalBenchRow(
            "Callaway-Sant'Anna",
            "TWFE",
            float(np.mean(post)),
            callaway_santanna(outcomes, group).overall,
            twfe,
        ),
        CausalBenchRow(
            "de Chaisemartin DID_M", "TWFE", delta, de_chaisemartin(outcomes, group), twfe
        ),
    ]


def _scm_row(rng: np.random.Generator, tau: float = 2.0) -> CausalBenchRow:
    n_donors, n_pre, n_post, rank = 30, 25, 10, 3
    factors = rng.normal(0.0, 1.0, (n_pre + n_post, rank))
    loadings = rng.normal(0.0, 1.0, (n_donors, rank))
    center = loadings.mean(axis=0)
    treated_loading = center + 2.5 * (loadings[0] - center)  # outside the donor convex hull
    treated = treated_loading @ factors.T + rng.normal(0.0, 0.1, n_pre + n_post)
    treated[n_pre:] += tau
    donors = loadings @ factors.T + rng.normal(0.0, 0.1, (n_donors, n_pre + n_post))
    panel = np.vstack([treated, donors])
    ascm = augmented_synthetic_control(panel, treated_unit=0, n_pre=n_pre).overall
    scm = synthetic_control(panel, treated_unit=0, n_pre=n_pre).overall
    return CausalBenchRow("augmented SCM", "plain SCM", tau, ascm, scm)


def _rlearner_row(seed: int) -> CausalBenchRow:
    import jax  # local: only this row needs a JAX key

    keys = jax.random.split(jax.random.key(seed), 5)
    x0, x1 = jax.random.normal(keys[0], (8000,)), jax.random.normal(keys[1], (8000,))
    z = jax.random.normal(keys[2], (8000,))  # confounder
    u = 1.5 * z + 0.5 * jax.random.normal(keys[3], (8000,))
    x_next = (1.0 + 0.8 * x0) * u + 2.0 * z + 0.1 * jax.random.normal(keys[4], (8000,))
    data = {"x0": x0, "x1": x1, "z": z, "u": u, "x_next": x_next}
    r_ate = RLearner().estimate(data, covariates=("x0", "x1", "z")).effect
    naive = float(jnp.sum((x_next - x_next.mean()) * (u - u.mean())) / jnp.sum((u - u.mean()) ** 2))
    return CausalBenchRow("R-learner", "naive OLS", 1.0, r_ate, naive)


def _gformula_row(rng: np.random.Generator) -> CausalBenchRow:
    n = 40000
    l0 = rng.normal(0.0, 1.0, n)
    a0 = 0.9 * l0 + rng.normal(0.0, 1.0, n)
    l1 = a0 + 0.7 * l0 + rng.normal(0.0, 1.0, n)  # A0 -> L1 (confounder on the causal path)
    a1 = 1.1 * l1 + 0.3 * a0 + rng.normal(0.0, 1.0, n)
    y = a0 + 1.5 * a1 + 0.5 * l0 + 0.8 * l1 + rng.normal(0.0, 0.3, n)
    data = {"a0": a0, "a1": a1, "l0": l0, "l1": l1, "y": y}
    treatments, confounders = ("a0", "a1"), (("l0",), ("l1",))
    g = sequential_g_formula(
        data,
        treatments=treatments,
        confounders=confounders,
        outcome="y",
        regime=(1.0, 1.0),
        baseline=(0.0, 0.0),
    )
    naive = naive_pooled_effect(data, treatments=treatments, confounders=confounders, outcome="y")
    return CausalBenchRow("g-formula", "naive pooled", 3.3, g, naive)


def run_causal_bench(seed: int = 0) -> list[CausalBenchRow]:
    """Run every frontier causal method vs its naive baseline on a known-effect synthetic DGP."""
    rng = np.random.default_rng(seed)
    return [*_did_rows(rng), _scm_row(rng), _rlearner_row(seed), _gformula_row(rng)]


def causal_bench_report(rows: list[CausalBenchRow]) -> str:
    """Format the causal-methods leaderboard: method and naive baseline bias, side by side."""
    header = f"{'method':<22}{'baseline':<14}{'truth':>8}{'est':>8}{'|bias|':>8}{'base|bias|':>12}"
    lines = [
        f"{r.method:<22}{r.baseline:<14}{r.truth:>8.2f}{r.estimate:>8.2f}"
        f"{r.method_bias:>8.2f}{r.baseline_bias:>12.2f}"
        for r in rows
    ]
    return "\n".join([header, *lines])
