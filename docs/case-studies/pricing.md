# Pricing under confounding

The flagship. Offline data is logged under a behaviour policy that ties the action to a confounder
`z` which also drives the outcome — retention teams gave big discounts to the customers most
likely to churn. A controller that fits the action's effect **without** adjusting for `z` learns
the wrong sign and drives the true plant away from its target; adjusting for `z` recovers the effect
and control succeeds. Both controllers plan with their learned effect and act on the true system.

```bash
uv run python scripts/flagship_demo.py
```

What it printed when this site was built:

```text
--8<-- "docs/output/flagship_demo.txt"
```

With matplotlib installed the script also writes `outputs/flagship.png`;
[tutorial 1](../tutorials/01_causal_vs_predictive_control.md) draws the same comparison and scores
it against an oracle. Code: [`chc.flagship`](../api/flagship.md), on
[`chc.causal`](../api/causal.md).
