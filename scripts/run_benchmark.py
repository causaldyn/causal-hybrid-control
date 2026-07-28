"""Run the CHC benchmark v0 and print the leaderboard with multi-seed bootstrap CIs.

Run: uv run python scripts/run_benchmark.py
"""

import urllib.error

from chc.benchmark import (
    ConfoundingRobustTask,
    InventoryTask,
    ModelUncertaintyTask,
    PricingTask,
    SupportShiftTask,
    leaderboard_multiseed,
    run_multiseed,
)
from chc.estimators import BackdoorOLS, DoubleML
from chc.lalonde import lalonde_report, load_lalonde

FAST_SEEDS = range(12)
SLOW_SEEDS = range(6)  # the ensemble-fitting task is ~12s/seed; keep it tractable


def main() -> None:
    print("== pricing (confounded linear steering) ==")
    print(leaderboard_multiseed(run_multiseed(PricingTask(), FAST_SEEDS)))
    print("\n== inventory (confounded demand, newsvendor ordering) ==")
    print(leaderboard_multiseed(run_multiseed(InventoryTask(), FAST_SEEDS)))
    print("\n== support-shift (model exploitation; pessimism vs greedy) ==")
    print(leaderboard_multiseed(run_multiseed(SupportShiftTask(), FAST_SEEDS)))
    print("\n== model-uncertainty (calibrated pessimism vs greedy) ==")
    print(leaderboard_multiseed(run_multiseed(ModelUncertaintyTask(), SLOW_SEEDS)))
    print("\n== confounding-robust (HIDDEN confounder; sensitivity radius vs certainty-equiv) ==")
    print(leaderboard_multiseed(run_multiseed(ConfoundingRobustTask(), FAST_SEEDS)))

    print("\n== LaLonde-DW (external: recover the randomized ATE from CPS-confounded data) ==")
    try:
        data = load_lalonde()
    except (urllib.error.URLError, OSError) as exc:
        print(f"skipped (data unavailable offline): {exc}")
    else:
        estimators = {"backdoor-OLS": BackdoorOLS(), "double-ML": DoubleML(degree=3)}
        print(lalonde_report(data, estimators))


if __name__ == "__main__":
    main()
