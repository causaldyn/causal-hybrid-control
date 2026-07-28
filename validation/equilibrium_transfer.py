"""Numeric cross-check of validation/equilibrium_transfer.mac (Result 39), in NumPy not JAX.

Independent of the shipped solver, so it checks the mathematics rather than the implementation:

1. ``||(I + kappa*J)^{-1}||_2 = 1`` exactly, for every probability vector ``s`` and every
   ``kappa >= 0`` -- the equilibrium never amplifies a perturbation of the agents' operator.
2. The damping threshold ``0 < d < 4/(2 + kappa)`` predicts convergence of the damped iteration, and
   ``d* = 4/(4 + kappa)`` is below it for every ``kappa``: no game is out of reach.
3. The naive contraction constant ``1/mu`` at ``d = 1/2`` equals ``4/(6 - kappa)``, so it diverges
   while the true conditioning stays at 1.

Run: ``uv run python validation/equilibrium_transfer.py``.
"""

from __future__ import annotations

import numpy as np


def softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max())
    return e / e.sum()


def claim_1_conditioning_is_one(rng: np.random.Generator) -> float:
    """``||(I - S')^{-1}||_2`` with ``S' = -kappa*J`` -- theory says exactly 1, never more."""
    worst = 0.0
    for _ in range(6000):
        n = int(rng.integers(2, 15))
        s = rng.dirichlet(np.full(n, 10.0 ** rng.uniform(-2.0, 1.5)))
        kappa = float(10.0 ** rng.uniform(-2.0, 2.0))
        jac = np.diag(s) - np.outer(s, s)
        worst = max(worst, float(np.linalg.norm(np.linalg.inv(np.eye(n) + kappa * jac), 2)))
    print(f"(1) max ||(I + kappa*J)^-1||_2 over 6000 draws: {worst:.9f}   (theory: exactly 1)")
    assert worst <= 1.0 + 1e-9
    assert worst >= 1.0 - 1e-9  # attained, because J annihilates the constants
    return worst


def _converges(attract: np.ndarray, kappa: float, damping: float, mass: float = 6.0) -> float:
    x = np.full(attract.size, mass / attract.size)
    for _ in range(2000):
        x = (1 - damping) * x + damping * mass * softmax(kappa * (attract - x / mass))
    step = (1 - damping) * x + damping * mass * softmax(kappa * (attract - x / mass))
    return float(np.linalg.norm(x - step))


def claim_2_damping_threshold() -> None:
    """The certified threshold must never promise convergence the iteration does not deliver."""
    attract = np.array([1.0, 0.5, 0.2, -0.3])
    print("(2) kappa   d=1/2 certified/resid      d*=4/(4+k) certified/resid")
    for kappa in (2.0, 5.0, 6.0, 8.0, 20.0, 100.0):
        star = 4.0 / (4.0 + kappa)
        rows = []
        for d in (0.5, star):
            certified = d < 4.0 / (2.0 + kappa)
            resid = _converges(attract, kappa, d)
            rows.append((certified, resid))
            assert not (certified and resid > 1e-6), f"FALSE POSITIVE kappa={kappa} d={d}"
        assert rows[1][0], f"d* must always be certified (kappa={kappa})"
        print(
            f"    {kappa:6.1f}   {rows[0][0]!s:5s} {rows[0][1]:.2e}          "
            f"{rows[1][0]!s:5s} {rows[1][1]:.2e}"
        )
    print("    -> d* certifies EVERY kappa; the d=1/2 ceiling at 6 is the solver's, not the game's")


def claim_3_looseness() -> None:
    """The naive ``1/mu`` at ``d = 1/2`` is ``4/(6-kappa)``; the truth is 1 throughout."""
    print("(3) kappa    mu      naive 1/mu    4/(6-kappa)    true conditioning")
    for kappa in (5.0, 5.5, 5.8, 5.96, 5.99):
        mu = 1.0 - max(0.5, kappa / 4.0 - 0.5)
        closed = 4.0 / (6.0 - kappa)
        assert abs(1.0 / mu - closed) < 1e-9
        print(f"    {kappa:5.2f}  {mu:7.4f}  {1.0 / mu:10.2f}  {closed:12.2f}                1.0")
    print("    -> the naive constant is unbounded where the conditioning is constant")


if __name__ == "__main__":
    claim_1_conditioning_is_one(np.random.default_rng(0))
    claim_2_damping_threshold()
    claim_3_looseness()
    print("\nall three claims hold")
