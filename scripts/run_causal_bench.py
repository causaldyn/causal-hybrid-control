"""Run the causal-methods leaderboard: each frontier estimator vs its naive baseline.

Run: uv run python scripts/run_causal_bench.py
"""

from __future__ import annotations

from chc.causal_bench import causal_bench_report, run_causal_bench


def main() -> None:
    print("== causal frontier vs naive built-ins (bias on a known-effect synthetic DGP) ==")
    print(causal_bench_report(run_causal_bench()))


if __name__ == "__main__":
    main()
