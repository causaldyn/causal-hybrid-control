# Marketplace dispatch

Driver-to-rider dispatch is Kantorovich's transportation problem: move drivers per zone (supply) to
riders per zone (demand) at least travel cost. The **dual potentials are the market-clearing
prices**: the demand dual is the surge signal — higher where demand outstrips supply — and it comes
free as the dual of the same optimisation, which no dispatch heuristic gives. It is solved by
entropic, log-domain Sinkhorn, which is differentiable, so pricing can itself be optimised; as the
entropic regularisation goes to zero it recovers the exact linear program.

The script builds a synthetic city and compares the transport plan with two naive dispatchers: a
local one that strands the demand it cannot reach, and a locally greedy one that serves everyone at
a higher cost. It then applies the surge prices as a driver incentive — drivers best-respond toward
high-price zones — and reports the supply-demand imbalance before and after.

```bash
uv run python scripts/run_marketplace_demo.py
```

What it printed when this site was built:

```text
--8<-- "docs/output/run_marketplace_demo.txt"
```

Code: [`chc.matching`](../api/matching.md).
