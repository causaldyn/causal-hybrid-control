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
        f"Gamma* separates them offline, before either acts: along the causal plan's path an"
        f" admissible action meets the supply floor's barrier condition up to the marginal"
        f" sensitivity model's Gamma {causal.certificate.gamma_star:.2f}, along the naive one's"
        f" only to {naive.certificate.gamma_star:.2f} -- a warning that needs no ground truth."
    )
    below = [
        step
        for step, state in enumerate(causal.true_trajectory)
        if float(state[1]) < report.supply_floor
    ]
    after = (
        f"past them it crosses the floor at step {below[0]}: the certificate audits the floor, and"
        " this solve does not hold it"
        if below
        else "past them it still holds the floor"
    )
    print(
        f"Gamma* is the problem's ceiling along the path, not the plan's: the causal plan's own"
        f" actions are certified for {causal.certificate.certified_steps} steps, over which the"
        f" true plant stays safe (min h = {causal.true_barrier_min:+.3f}); {after}."
    )


if __name__ == "__main__":
    main()
