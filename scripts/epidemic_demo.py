"""Epidemic flagship: optimal NPI flattens the curve under a hospital-capacity constraint.

Run: uv run python scripts/epidemic_demo.py                  (prints peaks)
     uv run --group viz python scripts/epidemic_demo.py      (also writes outputs/epidemic.png)
"""

from __future__ import annotations

import jax.numpy as jnp

from chc.epidemic import SIRDynamics, optimal_npi
from chc.integrate import rollout


def main() -> None:
    model = SIRDynamics(beta=0.6, gamma=0.1)  # R0 = 6
    x0 = jnp.array([0.99, 0.01])
    dt, horizon, i_max = 1.0, 100, 0.1

    xs_free = rollout(model, x0, jnp.zeros((horizon, 1)), dt)
    us = optimal_npi(model, x0, dt, horizon, i_max, steps=400)
    xs_ctrl = rollout(model, x0, us, dt)

    print(f"capacity I_max      = {i_max:.3f}")
    print(f"uncontrolled peak I = {float(jnp.max(xs_free[:, 1])):.3f}   (overshoots capacity)")
    print(f"controlled peak I   = {float(jnp.max(xs_ctrl[:, 1])):.3f}   (flattened to capacity)")
    print(f"total NPI effort    = {float(jnp.sum(us)):.2f},  max u = {float(jnp.max(us)):.2f}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        from pathlib import Path

        import matplotlib.pyplot as plt

        Path("outputs").mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axhline(i_max, ls="--", c="gray", label="capacity")
        ax.plot(xs_free[:, 1], lw=2, label="no intervention")
        ax.plot(xs_ctrl[:, 1], lw=2, label="optimal NPI")
        ax.plot(us[:, 0], lw=1, ls=":", label="NPI intensity u")
        ax.set_xlabel("day")
        ax.set_ylabel("infected fraction I")
        ax.set_title("Flatten the curve: optimal NPI under a capacity constraint")
        ax.legend()
        fig.tight_layout()
        fig.savefig("outputs/epidemic.png", dpi=130)
        print("wrote outputs/epidemic.png")
    except ImportError:
        print("(add the figure with: uv run --group viz python scripts/epidemic_demo.py)")


if __name__ == "__main__":
    main()
