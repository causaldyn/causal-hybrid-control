"""Run the CHC benchmark v0 and print the leaderboard.

Run: uv run python scripts/run_benchmark.py
"""

from __future__ import annotations

from chc.benchmark import InventoryTask, PricingTask, SupportShiftTask, leaderboard


def main() -> None:
    print("== pricing (confounded linear steering) ==")
    print(leaderboard(PricingTask().run()))
    print("\n== inventory (confounded demand, newsvendor ordering) ==")
    print(leaderboard(InventoryTask().run()))
    print("\n== support-shift (model exploitation; pessimism vs greedy) ==")
    print(leaderboard(SupportShiftTask().run()))


if __name__ == "__main__":
    main()
