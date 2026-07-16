"""Run the CHC benchmark v0 and print the leaderboard.

Run: uv run python scripts/run_benchmark.py
"""

from __future__ import annotations

from chc.benchmark import PricingTask, leaderboard


def main() -> None:
    print("== pricing (confounded linear steering) ==")
    print(leaderboard(PricingTask().run()))


if __name__ == "__main__":
    main()
