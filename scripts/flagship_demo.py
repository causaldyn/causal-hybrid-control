"""Flagship demo: causal control reaches the target; predictive control pushes the wrong way.

Run: uv run python scripts/flagship_demo.py                     (prints the summary)
     uv run --group viz python scripts/flagship_demo.py         (also writes outputs/flagship.png)
"""

from __future__ import annotations

from chc.flagship import run_flagship


def main() -> None:
    r = run_flagship()
    print(f"true effect b   = {r['b_true']:+.3f}")
    print(f"naive estimate  = {r['b_naive']:+.3f}   (confounded, wrong sign)")
    print(f"causal estimate = {r['b_causal']:+.3f}   (adjusted for z)")
    print(f"target x*       = {r['x_target']:+.3f}")
    print(f"causal  final x = {float(r['xs_causal'][-1]):+.3f}   -> reaches target")
    print(
        f"naive   final x = {float(r['xs_naive'][-1]):+.3f}   -> wrong way, at the actuator limit"
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        from pathlib import Path

        import matplotlib.pyplot as plt

        Path("outputs").mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axhline(r["x_target"], ls="--", c="gray", label="target")
        ax.plot(r["xs_causal"], lw=2, label=f"causal control ($\\hat b$ = {r['b_causal']:+.2f})")
        ax.plot(r["xs_naive"], lw=2, label=f"predictive control ($\\hat b$ = {r['b_naive']:+.2f})")
        ax.set_xlabel("step")
        ax.set_ylabel("state x")
        ax.set_title("Causal vs predictive control under confounding")
        ax.legend()
        fig.tight_layout()
        fig.savefig("outputs/flagship.png", dpi=130)
        print("wrote outputs/flagship.png")
    except ImportError:
        print("(add the figure with: uv run --group viz python scripts/flagship_demo.py)")


if __name__ == "__main__":
    main()
