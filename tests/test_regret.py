"""LQ regret guarantee: certainty-equivalence suboptimality is quadratic in the model error."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chc.dynamics import DampedOscillator
from chc.lqr import linearize_discrete, linearized_regret_certificate
from chc.regret import (
    causal_vs_predictive_certificate,
    certainty_equivalence_gap,
    closed_loop_cost,
    dlqr,
    interference_regret_certificate,
    orthogonal_control_certificate,
    pessimism_variance_certificate,
    regret_scaling,
)

A = np.array([[1.0, 0.1], [0.0, 0.95]])
B = np.array([[0.5], [1.0]])
Q = np.eye(2)
R = np.array([[0.5]])
X0 = np.array([1.0, 0.5])


def test_dlqr_solves_the_dare_and_stabilises() -> None:
    k, p = dlqr(A, B, Q, R)
    residual = A.T @ p @ A - p - A.T @ p @ B @ np.linalg.solve(R + B.T @ p @ B, B.T @ p @ A) + Q
    assert np.allclose(residual, 0.0, atol=1e-8)  # P satisfies the discrete algebraic Riccati eqn
    assert np.max(np.abs(np.linalg.eigvals(A - B @ k))) < 1.0  # the optimal loop is stable


def test_closed_loop_cost_matches_the_riccati_value_and_diverges_when_unstable() -> None:
    k, p = dlqr(A, B, Q, R)
    assert np.isclose(closed_loop_cost(A, B, k, Q, R, X0), float(X0 @ p @ X0))  # cost = x0' P x0
    no_control = np.zeros((1, 2))
    assert closed_loop_cost(A, B, no_control, Q, R, X0) == float("inf")  # A is marginally unstable


def test_gap_is_zero_at_the_true_model_and_positive_when_perturbed() -> None:
    assert abs(certainty_equivalence_gap(A, B, Q, R, A, B, X0)) < 1e-8  # exact model -> no regret
    a_hat = A + np.array([[0.0, 0.0], [0.02, -0.03]])
    b_hat = B + np.array([[0.01], [-0.02]])
    assert certainty_equivalence_gap(A, B, Q, R, a_hat, b_hat, X0) > 0.0  # misspecification costs


def test_regret_scales_quadratically_with_model_error() -> None:
    curve = regret_scaling(A, B, Q, R, X0, n_samples=300, seed=0)
    assert 1.7 < curve.exponent < 2.3  # Dean et al.: quadratic suboptimality (theory exponent 2)
    assert (np.diff(curve.gaps) < 0).all()  # gap shrinks monotonically as the error level drops


def test_interference_regret_is_quadratic_in_the_total_error() -> None:
    curve = interference_regret_certificate(A, B, Q, R, X0, interference_ratio=1.0, n_samples=300)
    assert 1.7 < curve.exponent < 2.3  # regret ~ (eid + eint)^2 (proofs/interference_regret.v)
    assert (np.diff(curve.gaps) < 0).all()  # gap shrinks as the total error drops


def test_optimal_pessimism_equals_the_effect_variance() -> None:
    # pessimism in the optimality condition (proofs/pessimistic_optimality.v): the expected-regret-
    # minimising effective-effort rho* tracks the effect-estimate variance s^2, and the pessimistic
    # control beats the certainty-equivalent one under uncertainty
    curve = pessimism_variance_certificate()
    assert np.corrcoef(curve.variances, curve.optimal_rho)[0, 1] > 0.95  # rho* ~ s^2
    for var, rho in zip(curve.variances, curve.optimal_rho, strict=True):
        assert abs(rho - var) < 0.5 * var + 0.03  # optimal pessimism = variance (+ grid step)
    assert (curve.pessimistic_regret < curve.ce_regret).all()  # pessimism beats greedy


def test_predictive_control_is_asymptotically_wrong_under_confounding() -> None:
    # notebook-01 hardened into a theorem (proofs/causal_mpc.v): the observational (predictive)
    # controller inherits a systematic omitted-variable bias, so its regret plateaus at a positive
    # floor that does not vanish with n; the interventional (causal) controller converges to oracle
    curve = causal_vs_predictive_certificate(n_seeds=6)
    floor = curve.predictive_floor
    assert floor > 0.0  # confounding creates a positive control-regret floor
    assert curve.predictive_regret[-1] > 0.7 * floor  # predictive plateaus at it (never vanishes)
    assert curve.causal_regret[-1] < 0.1 * floor  # causal regret vanishes toward the oracle
    assert curve.predictive_regret[-1] > 10 * curve.causal_regret[-1] + 1e-3  # ID closes it


def test_orthogonal_control_is_doubly_debiased() -> None:
    # novel result (proofs/orthogonal_control.v): a controller built on the Neyman-orthogonal DML
    # effect has regret ~ eps^4 in the nuisance error, vs ~ eps^2 for a single-residualisation
    # plug-in -- the DML orthogonality compounds with the certainty-equivalence quadratic regret map
    curve = orthogonal_control_certificate(n=200_000, n_seeds=4)
    assert 1.8 < curve.single_exponent < 2.7  # plug-in control regret ~ eps^2
    assert curve.orthogonal_exponent > 3.3  # orthogonal DML control regret ~ eps^4
    assert curve.orthogonal_exponent > curve.single_exponent + 1.0  # the double-debiasing gap
    assert curve.orthogonal_regret[0] < 0.05 * curve.single_regret[0]  # far lower regret at top eps


def test_interference_strictly_increases_the_regret() -> None:
    # the empirical mirror of the Rocq lemma interference_strictly_worse: adding the exposure-map
    # error channel (eint > 0) enlarges the certificate over the interference-blind (eint = 0) case
    blind = interference_regret_certificate(A, B, Q, R, X0, interference_ratio=0.0, n_samples=300)
    aware = interference_regret_certificate(A, B, Q, R, X0, interference_ratio=1.0, n_samples=300)
    assert aware.gaps[0] > blind.gaps[0]  # interference is not free


def test_linearized_certificate_zero_at_truth_and_positive_under_model_error() -> None:
    dyn = DampedOscillator(omega=1.0, zeta=0.15)
    x_star, u_star = jnp.zeros(2), jnp.zeros(1)
    exact = linearized_regret_certificate(dyn, dyn, x_star, u_star, Q, R, X0, 0.1)
    wrong = DampedOscillator(omega=1.3, zeta=0.05)
    certificate = linearized_regret_certificate(dyn, wrong, x_star, u_star, Q, R, X0, 0.1)
    assert abs(exact) < 1e-8  # a correct model implies no certainty-equivalence regret
    assert certificate > 0.0  # a wrong model implies a positive local suboptimality certificate


def test_linearized_certificate_matches_the_lq_gap_on_the_linearisation() -> None:
    dyn, dyn_hat = DampedOscillator(omega=1.0, zeta=0.15), DampedOscillator(omega=1.2, zeta=0.1)
    x_star, u_star = jnp.zeros(2), jnp.zeros(1)
    a, b = linearize_discrete(dyn, x_star, u_star, 0.1)
    a_hat, b_hat = linearize_discrete(dyn_hat, x_star, u_star, 0.1)
    direct = certainty_equivalence_gap(
        np.asarray(a), np.asarray(b), Q, R, np.asarray(a_hat), np.asarray(b_hat), X0
    )
    cert = linearized_regret_certificate(dyn, dyn_hat, x_star, u_star, Q, R, X0, 0.1)
    assert np.isclose(cert, direct)  # the helper just linearises, then calls the LQ gap
