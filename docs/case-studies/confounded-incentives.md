# Confounded incentives, closed loop

A synthetic **observational** marketplace: a demand confounder drives both the historical incentive
— the past policy raised incentives in busy periods — and the completions it is meant to buy. This
is confounded logging, not a randomised switchback, so the offline effect estimate is biased. The
plant is `x' = a*x + b_true*u + noise`, controlled over a receding horizon toward a target, and
the business cost is **asymmetric**: missing riders (churn) costs more than wasted budget.

Two controllers act on it in closed loop:

- **certainty-equivalence** trusts the biased estimate, under-actuates and undershoots the target;
- **confounding-robust** takes an assumed sensitivity level `Γ`, turns it into a radius on the
  effect, and shifts the incentive to hedge the costlier error.

The radius calibration — the effect's half-width set to `(Γ-1)/(Γ+1)` times the estimate — is a
benchmark assumption, not a general consequence of the sensitivity model. The script sweeps the
*true* confounding strength, which neither controller knows.

```bash
uv run python scripts/run_dynamic_confounding_demo.py
```

What it printed when this site was built:

```text
--8<-- "docs/output/run_dynamic_confounding_demo.txt"
```

Pessimism is not free: near zero confounding the robust controller pays a premium, because there its
conservatism costs more than it saves.
[The sensitivity level Γ](../concepts/gamma.md) explains the radius;
[tutorial 5](../tutorials/05_confounding_robust_control.md) runs the static version step by step.
Code: [`confounding_robust_tracking_benchmark`](../api/regret.md) in `chc.regret`, surfaced by
[`chc.sensitivity`](../api/sensitivity.md).
