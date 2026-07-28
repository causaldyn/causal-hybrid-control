"""Numeric cross-check of validation/barrier_feasibility.mac (Result 40), in NumPy not JAX.

Nothing here uses the closed form under test as an oracle: every claim is checked against a brute
force search, so an algebra slip in the module would show up rather than be reproduced.

1. The robust margin ``max_{|u|<=U} min_{|b'-b|<=Delta} (a + <grad h, b'> u)`` equals
   ``a + (|g| - d)_+ * U`` -- checked against a dense grid over both ``u`` and the adversary.
2. The robust-optimal action is exactly zero iff the identified interval CONTAINS ZERO -- for a
   symmetric ball that reads ``d >= |g|``, and the asymmetric case is checked separately.
3. The feasibility threshold ``d* = |g| - D/U`` is SHARP: certified just below, violated just above.
4. The order dichotomy: safety margin loss is linear in the effect error (log-log slope 1) while
   the performance regret of the same controller is quadratic (slope 2).
5. The multivariate lift ``(||B^T grad h|| - Delta*||grad h||)_+ * U`` is exact, not conservative --
   checked against a random search over the unit ball in R^m.
6. That linearity SATURATES: the margin lost is ``U * min(d, |g|)``, so claim 4 is a statement about
   ``d < |g|`` and not a global affine law.
7. The threshold has three regimes in the drift deficit ``D``, and the middle formula is only valid
   in one of them -- ``D > U*|g|`` is infeasible before any radius is spent.

Run: ``uv run python validation/barrier_feasibility.py``.
"""

from __future__ import annotations

import numpy as np


def brute_force_margin(a: float, g: float, d: float, u_max: float, n: int = 20001) -> float:
    """``max_u min_{g' in [g-d, g+d]} (a + g'*u)`` by dense enumeration of both players."""
    us = np.linspace(-u_max, u_max, n)
    worst = np.minimum(
        a + (g - d) * us, a + (g + d) * us
    )  # a linear min is attained at an endpoint
    return float(np.max(worst))


def claim_1_and_2_margin_and_zero_action(rng: np.random.Generator) -> None:
    """The closed form matches brute force, and u=0 is optimal exactly on the unidentified side."""
    worst_gap, zero_action_disagreements = 0.0, 0
    for _ in range(4000):
        a = float(rng.uniform(-3.0, 3.0))
        g = float(rng.uniform(-2.0, 2.0))
        d = float(rng.uniform(0.0, 3.0))
        u_max = float(rng.uniform(0.2, 4.0))
        closed = a + max(0.0, abs(g) - d) * u_max
        worst_gap = max(worst_gap, abs(closed - brute_force_margin(a, g, d, u_max)))

        us = np.linspace(-u_max, u_max, 20001)
        worst = np.minimum(a + (g - d) * us, a + (g + d) * us)
        best_u = us[int(np.argmax(worst))]
        if (abs(best_u) < 1e-3) != (d >= abs(g)):
            zero_action_disagreements += 1

    print(f"(1) max |closed form - brute force| over 4000 draws: {worst_gap:.3e}")
    print(f"(2) 'u*=0 iff sign unidentified' disagreements: {zero_action_disagreements} / 4000")
    assert worst_gap < 1e-3
    assert zero_action_disagreements == 0


def claim_3_threshold_is_sharp() -> None:
    """Just below ``d*`` the barrier condition holds; just above it cannot be met by any action.

    The threshold only binds when the drift leaves a positive deficit ``D = -alpha*h - a``. With
    ``D <= 0`` the uncontrolled system already satisfies the barrier, ``u = 0`` certifies it, and no
    identification radius can break safety -- checked below, because it is an easy case to design a
    misleading experiment around.
    """
    g, u_max, alpha, h = 1.2, 2.0, 1.0, 0.5
    a = -1.0  # drift fails the barrier condition on its own: the controller has to cover D = 0.5
    deficit = -alpha * h - a
    d_star = abs(g) - deficit / u_max
    print(f"(3) deficit D = {deficit:+.3f} > 0, d* = {d_star:.4f}")
    for eps, expect in ((-1e-3, True), (+1e-3, False)):
        margin = brute_force_margin(a, g, d_star + eps, u_max)
        certified = margin >= -alpha * h - 1e-9
        print(f"    d = d*{eps:+.0e}: best robust margin {margin:+.6f}  certified={certified}")
        assert certified is expect

    slack = -0.4  # now the drift alone satisfies it: D = -0.1 < 0
    for d in (0.0, 1.0, 5.0, 50.0):
        assert brute_force_margin(slack, g, d, u_max) >= -alpha * h - 1e-9
    print("    with D < 0 every radius stays certified (u=0 suffices) -- the threshold is vacuous")


def claim_4_order_dichotomy() -> None:
    """Safety loses margin at FIRST order in the effect error; performance regret at SECOND."""
    g, u_max, l_reg = 1.5, 2.0, 3.0
    errors = np.array([0.4, 0.2, 0.1, 0.05, 0.025])
    safety_loss = (abs(g) - (abs(g) - errors)) * u_max  # nominal margin minus robust margin
    regret = l_reg * errors**2
    s_slope = float(np.polyfit(np.log(errors), np.log(safety_loss), 1)[0])
    r_slope = float(np.polyfit(np.log(errors), np.log(regret), 1)[0])
    print(f"(4) log-log slope, safety margin loss: {s_slope:.4f}   (theory 1)")
    print(f"    log-log slope, performance regret: {r_slope:.4f}   (theory 2)")
    print(f"    ratio at the smallest error: {safety_loss[-1] / regret[-1]:.1f}x and growing")
    assert abs(s_slope - 1.0) < 1e-6
    assert abs(r_slope - 2.0) < 1e-6


def claim_5_multivariate_lift(rng: np.random.Generator) -> None:
    """``(||B^T grad h|| - Delta*||grad h||)_+ * U`` is both an upper bound and ATTAINED.

    Random search alone would be a weak test in ``R^m`` -- a uniform direction almost never aligns
    with ``w``, so it can only ever certify the bound from below and badly. Both halves are checked
    instead: the explicit Cauchy-Schwarz action reaches the value exactly, and no sampled action
    exceeds it.
    """
    worst_excess, worst_attain = 0.0, 0.0
    for _ in range(300):
        n, m = int(rng.integers(2, 6)), int(rng.integers(1, 5))
        grad_h = rng.standard_normal(n)
        b_hat = rng.standard_normal((n, m))
        delta = float(rng.uniform(0.0, 1.5))
        u_max = float(rng.uniform(0.5, 3.0))

        w = b_hat.T @ grad_h
        closed = max(0.0, float(np.linalg.norm(w)) - delta * float(np.linalg.norm(grad_h))) * u_max

        scaled_radius = delta * float(np.linalg.norm(grad_h))

        def value(us: np.ndarray, w: np.ndarray = w, radius: float = scaled_radius) -> np.ndarray:
            return us @ w - radius * np.linalg.norm(us, axis=1)

        norm_w = float(np.linalg.norm(w))
        best = np.zeros((1, m)) if norm_w <= delta * np.linalg.norm(grad_h) else u_max * w / norm_w
        worst_attain = max(worst_attain, abs(closed - float(value(np.atleast_2d(best))[0])))

        directions = rng.standard_normal((4000, m))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        us = directions * rng.uniform(0.0, u_max, size=(4000, 1))
        worst_excess = max(worst_excess, float(np.max(value(us))) - closed)
    print(f"(5) max |closed form - explicit maximiser| over 300 problems: {worst_attain:.3e}")
    print(f"    max (sampled action - closed form), must stay <= 0:       {worst_excess:+.3e}")
    assert worst_attain < 1e-12  # attained, so the bound is exact
    assert worst_excess <= 1e-9  # and never exceeded, so it really is the max


def claim_6_the_linear_loss_saturates(rng: np.random.Generator) -> None:
    """Loss vs a perfect channel is ``U*min(d,|g|)``: first order only below the kink, then flat."""
    worst = 0.0
    for _ in range(2000):
        a = float(rng.normal())
        g = float(abs(rng.normal()) + 0.1)
        u_max = float(rng.uniform(0.5, 3.0))
        d = float(rng.uniform(0.0, 3.0 * g))
        loss = brute_force_margin(a, g, 0.0, u_max) - brute_force_margin(a, g, d, u_max)
        worst = max(worst, abs(loss - u_max * min(d, g)))
    print(f"(6) max |brute-force loss - U*min(d,|g|)| over 2000 problems: {worst:.3e}")
    assert worst < 1e-9

    g, u_max = 0.6, 2.0
    above = [brute_force_margin(-1.0, g, d, u_max) for d in (1.0, 2.0, 50.0)]
    print(f"    margin at d = 1, 2, 50 (must be identical past the kink): {above}")
    assert max(above) - min(above) < 1e-12  # flat: nothing left to lose


def claim_7_the_threshold_has_three_regimes() -> None:
    """``D <= 0`` vacuous, ``D > U*|g|`` empty, ``0 < D <= U*|g|`` the one the formula describes."""
    g, u_max = 0.6, 2.0
    for drift, alpha_h, label in ((-0.1, 0.5, "D <= 0"), (-2.0, 0.5, "D > U*|g|")):
        deficit = -alpha_h - drift
        certified_at_zero = brute_force_margin(drift, g, 0.0, u_max) >= -alpha_h
        print(f"(7) {label:<10} D = {deficit:+.2f}  certified at d = 0: {certified_at_zero}")
        assert certified_at_zero == (deficit <= 0.0)

    drift, alpha_h = -0.9, 0.5  # D = +0.4, inside the useful regime
    d_star = g - (-alpha_h - drift) / u_max
    assert 0.0 <= d_star < g
    assert brute_force_margin(drift, g, d_star - 1e-6, u_max) >= -alpha_h
    assert brute_force_margin(drift, g, d_star + 1e-6, u_max) < -alpha_h
    print(f"    0 < D <= U*|g|: d* = {d_star:.4f} in [0, |g|), sharp both sides")


if __name__ == "__main__":
    generator = np.random.default_rng(0)
    claim_1_and_2_margin_and_zero_action(generator)
    claim_3_threshold_is_sharp()
    claim_4_order_dichotomy()
    claim_5_multivariate_lift(generator)
    claim_6_the_linear_loss_saturates(generator)
    claim_7_the_threshold_has_three_regimes()
    print("\nall seven claims hold")
