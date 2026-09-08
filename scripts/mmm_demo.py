"""Marketing-mix budget scheduling from confounded weekly logs, audited on the true plant.

Run: uv run python scripts/mmm_demo.py
"""

from __future__ import annotations

from chc.mmm import MarketingMixSystem, run_marketing_mix


def main() -> None:
    system = MarketingMixSystem()
    report = run_marketing_mix(system)
    truth = dict(zip(system.channels, system.gamma, strict=True))
    print("== weekly media log, planner chased the season; true incremental returns", truth, "==\n")
    print(report.table())

    adjusted = report.arm("adjusted").prescription
    confounded = report.arm("confounded").prescription
    assert adjusted is not None
    assert confounded is not None

    print("\nwhat each arm believes each channel is worth (channel x box width):")
    header = f"{'channel':16}{'adjusted':>12}{'confounded':>12}{'inflation':>12}"
    print(header)
    print("-" * len(header))
    for name, believed in adjusted.reach().items():
        naive = confounded.reach()[name]
        print(f"{name:16}{believed:>12.3f}{naive:>12.3f}{naive / believed:>11.2f}x")

    flat_gap = report.lift("adjusted") / report.lift("flat") - 1.0
    print(
        f"\nAt the same total budget ({report.arm('adjusted').total_spend:.1f}) the prescribed"
        f" schedule buys {flat_gap:+.1%} more cumulative sales than an equal split -- it"
        " front-loads to build carryover and then tapers, which is why cumulative sales and not"
        " the terminal value is the number reported."
    )
    print(
        f"The confounded arm credits every channel with the season, so it believes it needs less:"
        f" it spends {report.arm('confounded').total_spend:.1f} and buys"
        f" {report.lift('confounded') / report.lift('adjusted'):.0%} of the lift."
    )
    print(
        "Read at return-per-unit-spend it looks BETTER"
        f" ({report.efficiency('confounded'):.3f} against {report.efficiency('adjusted'):.3f}),"
        " because the response saturates and it under-invests. That is why the arms are compared"
        " at matched budget."
    )


if __name__ == "__main__":
    main()
