"""Numeric cross-check of validation/congestion_contraction.mac (Result 39 §B).

A symbolic bound is a hypothesis until it is checked numerically. Three claims:

1. ``lam_max(diag(s) - s s^T) <= 1/2`` for every probability vector ``s`` -- sharp at
   ``s = (1/2, 1/2, 0..)``.
2. ``||T'(x)||_2 <= max(1/2, beta*c/4 - 1/2)`` for the damped logit congestion map, at random
   states.
3. The SUFFICIENT threshold ``beta*c < 6`` is conservative: the map still converges above it, and
   the certificate must never claim contraction where the iteration actually diverges.

Run: ``uv run python validation/congestion_contraction.py``.
"""

from __future__ import annotations

import numpy as np


def softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max())
    return e / e.sum()


def jacobian(s: np.ndarray) -> np.ndarray:
    return np.diag(s) - np.outer(s, s)


def claim_1_softmax_jacobian_spectrum(rng: np.random.Generator) -> float:
    """Worst observed ``lam_max(J)`` over random simplex points of many dimensions."""
    worst = 0.0
    for n in range(2, 40):
        for _ in range(400):
            alpha = 10.0 ** rng.uniform(-2.0, 1.5)  # spans peaked to near-uniform s
            s = rng.dirichlet(np.full(n, alpha))
            worst = max(worst, float(np.linalg.eigvalsh(jacobian(s)).max()))
    sharp = float(np.linalg.eigvalsh(jacobian(np.array([0.5, 0.5]))).max())
    print(f"(1) max lam_max(J) over 15200 random s: {worst:.9f}   (bound 0.5)")
    print(f"    sharp case s=(1/2,1/2):             {sharp:.9f}   (attains the bound)")
    assert worst <= 0.5 + 1e-12
    assert abs(sharp - 0.5) < 1e-12
    return worst


def claim_2_operator_norm(rng: np.random.Generator) -> float:
    """``||T'||_2`` at random states vs the certified bound ``max(1/2, kappa/4 - 1/2)``."""
    worst_slack = np.inf
    for _ in range(4000):
        n = int(rng.integers(2, 12))
        mass = float(10.0 ** rng.uniform(-0.5, 1.5))
        beta = float(10.0 ** rng.uniform(-1.0, 1.5))
        cong = float(10.0 ** rng.uniform(-1.0, 1.5))
        x = rng.dirichlet(np.ones(n)) * mass
        v = beta * (rng.normal(size=n) - cong * x / mass)
        norm = float(
            np.abs(
                np.linalg.eigvalsh(0.5 * np.eye(n) - 0.5 * beta * cong * jacobian(softmax(v)))
            ).max()
        )
        worst_slack = min(worst_slack, max(0.5, beta * cong / 4 - 0.5) - norm)
    print(f"(2) min slack (bound - measured ||T'||) over 4000 states: {worst_slack:.9f}   (>= 0)")
    assert worst_slack >= -1e-12
    return worst_slack


def claim_3_no_false_positive() -> tuple[float, float]:
    """Sweep ``kappa = beta*c``: the certificate must not certify a diverging iteration."""
    attract = np.array([1.0, 0.5, 0.2])
    mass, last_certified, first_divergent = 6.0, 0.0, np.inf
    print("(3) kappa   certified   residual after 400 damped iterations")
    for kappa in np.arange(1.0, 12.01, 0.5):
        beta, cong = kappa, 1.0
        x = np.full(3, mass / 3)
        for _ in range(400):
            x = 0.5 * x + 0.5 * mass * softmax(beta * (attract - cong * x / mass))
        resid = float(
            np.linalg.norm(x - (0.5 * x + 0.5 * mass * softmax(beta * (attract - cong * x / mass))))
        )
        certified = kappa < 6.0
        if certified:
            last_certified = kappa
        if resid > 1e-6 and kappa < first_divergent:
            first_divergent = kappa
        flag = "  <-- certified but the bound is loose" if certified and resid < 1e-6 else ""
        print(f"    {kappa:5.1f}   {certified!s:9s}   {resid:.3e}{flag}")
        assert not (certified and resid > 1e-6), f"FALSE POSITIVE at kappa={kappa}"
    print(f"    last certified kappa {last_certified:.1f}, first divergent {first_divergent:.1f}")
    print("    -> sufficient, not necessary: a real gap where the map converges uncertified")
    return last_certified, float(first_divergent)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    claim_1_softmax_jacobian_spectrum(rng)
    claim_2_operator_norm(rng)
    claim_3_no_false_positive()
    print("\nall three claims hold")
