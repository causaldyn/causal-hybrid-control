"""The whole spine on one decision: confounded logs -> effect -> plan -> safety certificate.

Run: uv run python scripts/spine_demo.py
"""

from __future__ import annotations

from chc.spine import run_spine


def main() -> None:
    report = run_spine()
    print(f"== two-zone driver supply, true incentive gain b = {report.effect_true:+.3f} ==\n")
    header = f"{'arm':8}{'b_hat':>9}{'planned J':>12}{'true J':>10}{'certified':>12}{'Gamma*':>9}"
    print(header)
    print("-" * len(header))
    for arm in report.arms:
        cert = arm.certificate
        prefix = f"{cert.certified_steps}/{len(cert.planned_certified)}"
        print(
            f"{arm.name:8}{arm.effect:>+9.3f}{arm.plan.task_cost:>12.3f}"
            f"{arm.true_cost:>10.3f}{prefix:>12}{cert.gamma_star:>9.3f}"
        )

    naive, causal = report.arm("naive"), report.arm("causal")
    print(
        f"\nthe naive arm plans a cost of {naive.plan.task_cost:.2f} and pays"
        f" {naive.true_cost:.2f} on the real plant; the causal arm's plan is worth what it says"
        f" ({causal.plan.task_cost:.2f} planned, {causal.true_cost:.2f} paid)."
    )
    print(
        f"Gamma* separates them offline, before either acts: the causal plan's supply floor holds"
        f" up to sensitivity {causal.certificate.gamma_star:.2f}, the naive one's only to"
        f" {naive.certificate.gamma_star:.2f} -- a warning that needs no ground truth."
    )
    print(
        f"over the certified prefix the true plant stays safe (min h ="
        f" {causal.true_barrier_min:+.3f}); past it the plan is uncertified, not proven unsafe."
    )


if __name__ == "__main__":
    main()
