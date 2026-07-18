"""Run the CHC benchmark v0 and print the leaderboard with multi-seed bootstrap CIs.

Run: uv run python scripts/run_benchmark.py
"""

from __future__ import annotations

from chc.benchmark import (
    InventoryTask,
    ModelUncertaintyTask,
    PricingTask,
    SupportShiftTask,
    leaderboard_multiseed,
    run_multiseed,
)

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


if __name__ == "__main__":
    main()
