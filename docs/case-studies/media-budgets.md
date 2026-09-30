# Media budgets

Marketing-mix budget scheduling: which channel to spend on, and when, from confounded logs. Media
spend is *planned against demand* — a marketer raises budget in the weeks a product sells anyway —
so a model fitted on the log credits the channel with the season. Optimise against that model and
the plan over-spends on whichever channel the planner favoured.

The plant is control-affine by construction, so identification and safety read the same object:

```text
d(sales)/dt     = -decay*(sales - base) + sum_c beta_c * hill(adstock_c) + sum_c gamma_c * spend_c
d(adstock_c)/dt = -theta_c * adstock_c + spend_c
```

Each channel acts twice: `gamma_c` is the immediate incremental return and is the **control
channel**; `beta_c` is the carried-over return through a saturating adstock and is part of the
**drift**. Under a policy that chases seasonality it is `gamma_c` that is confounded, and `gamma_c`
is exactly what cross-fit Robinson partialling-out identifies. The adstock rows are mechanical and
go to `prescribe` as `known=`, so the fit has only the sales row to learn.

```bash
uv run python scripts/mmm_demo.py
```

What it printed when this site was built:

```text
--8<-- "docs/output/mmm_demo.txt"
```

The arms are the same optimiser reading the same log through different causal claims, every one
audited by rolling its schedule out on the **true** plant:

- `adjusted` — the graph, so the season is adjusted for and the incremental returns are identified;
- `confounded` — an empty adjustment set *asserted*, which is what fitting the log directly amounts
  to;
- `flat` — an equal split held constant at the `adjusted` arm's total spend, so the head-to-head is
  at matched budget and differs only in allocation;
- `myopic` — the identified fit spent on each week's return alone, at the same budget, which
  isolates what the plan's horizon buys from what identification buys;
- `none` — spend held at the floor, the do-nothing counterfactual every lift is measured against.

Scope, and it bounds what the numbers mean:

- The saturating plant is fitted with a polynomial drift, so the fitted plant is a **local
  surrogate** of the truth. That is why every arm is audited on the true plant, and the sales
  reported are the audited ones rather than the planner's own forecast.
- Spend is priced through `Lever.unit_cost` rather than capped. `prescribe` takes a budget
  (`budgets=`), but the arms here are compared at matched spend instead: the `flat` and `myopic`
  arms are held at the `adjusted` arm's total spend.
- The carryover rates `theta_c` are taken as known. In practice they are fitted (Robyn, Meridian);
  holding them fixed isolates the incremental return, which is what this case study is about.
- What the whole-horizon plan buys over the myopic one is a property of the plant's parameters,
  not of the method. The `myopic` arm is there to measure it rather than assume it.

Code: [`chc.mmm`](../api/mmm.md), through [`prescribe`](../api/decision.md).
