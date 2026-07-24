"""Dynamic confounding-robust control: the §35 controller in a CLOSED LOOP on a confounded plant.

Lifts the static §35 minimax into a receding-horizon controller on the confounded scalar plant
x' = a*x + b_true*u + noise: a demand confounder biases the offline effect estimate, the CE
controller under-actuates and undershoots the target (expensive when churn outweighs budget waste),
and the §35 controller uses the §32 sensitivity radius to hedge -- over 30 closed-loop steps.

Run: uv run python scripts/run_dynamic_confounding_demo.py
"""

from __future__ import annotations

from chc.regret import confounding_robust_tracking_benchmark


def main() -> None:
    curve = confounding_robust_tracking_benchmark()
    print("== confounding-robust control in CLOSED LOOP on a confounded dynamic plant ==")
    print("   (x' = a*x + b_true*u + noise; churn 4x budget-waste; assumed Gamma=2.5)\n")
    print(f"   {'true confounding':>18} {'CE cost':>10} {'robust cost':>12} {'robust wins':>12}")
    for conf, ce, rob in zip(
        curve.confounding_levels, curve.ce_costs, curve.robust_costs, strict=True
    ):
        flag = "yes" if rob < ce else "premium"
        print(f"   {conf:>18.1f} {ce:>10.3f} {rob:>12.3f} {flag:>12}")
    print()
    ce_wc, rob_wc = curve.ce_worst_case, curve.robust_worst_case
    print(f"   worst-case cost  CE {ce_wc:.3f} -> robust {rob_wc:.3f}")
    print(f"   savings at realistic confounding : {curve.savings_at_target_pct:.0f}%")
    print(f"   premium when unconfounded : {curve.unconfounded_premium_pct:.0f}% of CE downside")
    print(
        "\n   The static §35 minimax is now a closed-loop controller: pessimism bounds the"
        "\n   accumulated downside, wins beyond a confounding threshold, pays a bounded premium."
    )


if __name__ == "__main__":
    main()
