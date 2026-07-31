"""Numeric cross-check of validation/actuator_reparametrisation.mac (Result 41).

A symbolic identity is a hypothesis until it is measured. Four claims, on a bilinear plant fitted by
least squares in two actuator coordinate systems:

1. The transformation law ``a -> a + beta*b1``, ``b0 -> alpha*b0``, ``b1 -> alpha*b1``.
2. ``lambda(u) = a + b1*u`` at a fixed physical actuator position is invariant, and recovers the
   truth, while ``a`` alone does not.
3. The Frisch-Waugh-Lovell form of the fitted coefficient.
4. The size of the artefact at the BOPTEST numbers: a plant with a decay of -1.40/h reported as
   +6.42/h purely because its actuator reports a setpoint in ``[15, 25] C``.

Run: ``uv run python validation/actuator_reparametrisation.py``.
"""

from __future__ import annotations

import numpy as np

A_TRUE, D_TRUE, B0_TRUE, B1_TRUE, C_TRUE = -0.05, 1.0, 0.9, -0.03, 0.04
LO, SPAN = 15.0, 10.0  # the two setpoint-actuated BOPTEST cases


def _sample(rng: np.random.Generator, n: int = 4000):
    """A bilinear plant whose actuator lives far from its own zero, as a setpoint does."""
    z = rng.normal(size=n)
    temp = 21.0 + 0.8 * rng.normal(size=n)
    action = 20.0 + 2.5 * rng.normal(size=n)
    rate = (
        A_TRUE * temp
        + D_TRUE
        + C_TRUE * z
        + (B0_TRUE + B1_TRUE * temp) * action
        + 0.01 * rng.normal(size=n)
    )
    return z, temp, action, rate


def _fit(temp: np.ndarray, z: np.ndarray, action: np.ndarray, rate: np.ndarray):
    """OLS on ``[1, T, z, u, T*u]``; returns ``(a, b0, b1)``."""
    design = np.column_stack([np.ones_like(temp), temp, z, action, temp * action])
    coefficients = np.linalg.lstsq(design, rate, rcond=None)[0]
    return coefficients[1], coefficients[3], coefficients[4]


def main() -> None:
    rng = np.random.default_rng(7)
    z, temp, action, rate = _sample(rng)
    travel = (action - LO) / SPAN

    a_raw, b0_raw, b1_raw = _fit(temp, z, action, rate)
    a_frac, b0_frac, b1_frac = _fit(temp, z, travel, rate)

    print("claim 1 -- the transformation law")
    for name, got, want in (
        ("a ", a_frac, a_raw + LO * b1_raw),
        ("b0", b0_frac, SPAN * b0_raw),
        ("b1", b1_frac, SPAN * b1_raw),
    ):
        print(f"  {name} fitted {got:+.8f}  predicted {want:+.8f}  err {abs(got - want):.2e}")

    mean_action, mean_travel = float(action.mean()), float(travel.mean())
    decay_raw = a_raw + b1_raw * mean_action
    decay_frac = a_frac + b1_frac * mean_travel
    truth = A_TRUE + B1_TRUE * mean_action
    print("\nclaim 2 -- lambda(ubar) is invariant and a is not")
    print(
        f"  lambda raw {decay_raw:+.8f}  frac {decay_frac:+.8f}  "
        f"err {abs(decay_raw - decay_frac):.2e}"
    )
    print(f"  truth      {truth:+.8f}   |  a alone: raw {a_raw:+.8f}  frac {a_frac:+.8f}")

    nuisance = np.column_stack([np.ones_like(temp), z, action, temp * action])
    projector = nuisance @ np.linalg.pinv(nuisance)
    temp_residual = temp - projector @ temp
    rate_residual = rate - projector @ rate
    fwl = float(temp_residual @ rate_residual / (temp_residual @ temp_residual))
    print("\nclaim 3 -- Frisch-Waugh-Lovell")
    print(f"  partial ratio {fwl:+.8f}  against fitted {a_raw:+.8f}  err {abs(fwl - a_raw):.2e}")
    print(f"  state identifying budget var(T|W)/var(T) = {temp_residual.var() / temp.var():.6f}")

    print("\nclaim 4 -- the artefact at the BOPTEST numbers (hydronic, fitted on 960 rows)")
    a_hydronic, b1_hydronic, held = 6.4197, -0.3779, 20.685
    print(f"  reported pole (decay at a setpoint of 0 C) {a_hydronic:+.4f}")
    print(f"  decay at the setpoint actually held        {a_hydronic + b1_hydronic * held:+.4f}")
    print(f"  sign changes at a setpoint of              {-a_hydronic / b1_hydronic:.2f} C")


if __name__ == "__main__":
    main()
