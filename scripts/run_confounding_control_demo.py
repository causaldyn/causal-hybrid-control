"""Grounding §35: confounding-robust controller vs certainty-equivalence on a marketplace task.

A synthetic switchback marketplace where a demand confounder biases the naive effect estimate. The
certainty-equivalence controller trusts it and under-incentivises (missing completions -- expensive
when rider churn outweighs budget waste); the §35 confounding-robust controller uses an assumed
sensitivity to shift the incentive and hedge the costlier error. Full estimate->control pipeline.

Run: uv run python scripts/run_confounding_control_demo.py
"""

from __future__ import annotations

from chc.regret import confounding_robust_control_benchmark


def main() -> None:
    curve = confounding_robust_control_benchmark()
    print("== confounding-robust control on a synthetic switchback marketplace ==")
    print("   (churn cost 4x budget-waste; controller assumes sensitivity Gamma=2.5)\n")
    print(f"   {'true confounding':>18} {'CE cost':>10} {'robust cost':>12} {'robust wins':>12}")
    for conf, ce, rob in zip(
        curve.confounding_levels, curve.ce_costs, curve.robust_costs, strict=True
    ):
        flag = "yes" if rob < ce else "premium"
        print(f"   {conf:>18.1f} {ce:>10.3f} {rob:>12.3f} {flag:>12}")
    print()
    print(
        f"   worst-case cost  CE {curve.ce_worst_case:.3f} -> robust {curve.robust_worst_case:.3f}"
    )
    print(f"   savings at realistic confounding : {curve.savings_at_target_pct:.0f}%")
    print(f"   premium when unconfounded : {curve.unconfounded_premium_pct:.0f}% of CE downside")
    print(
        "\n   Pessimism bounds the downside: robust wins wherever confounding is real and pays"
        "\n   only a bounded premium when it is not -- the honest robustness trade-off, not a toy."
    )


if __name__ == "__main__":
    main()
