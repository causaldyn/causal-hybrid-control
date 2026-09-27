# Case studies

Each one is a script in
[`scripts/`](https://github.com/causaldyn/causal-hybrid-control/tree/main/scripts), and each page
gives the one command that reproduces it and shows what that command printed when this site was
built.

| case study | the decision | command |
|---|---|---|
| [Pricing under confounding](pricing.md) | steer a KPI to target from logs whose action was aimed at the customers most likely to churn | `uv run python scripts/flagship_demo.py` |
| [Flatten the curve](epidemic.md) | the least intervention that keeps an epidemic under hospital capacity | `uv run python scripts/epidemic_demo.py` |
| [Media budgets](media-budgets.md) | which channel to spend on, and when, from logs planned against demand | `uv run python scripts/mmm_demo.py` |
| [Marketplace dispatch](marketplace-dispatch.md) | which drivers to send where, and the surge prices that fall out of the same optimisation | `uv run python scripts/run_marketplace_demo.py` |
| [Driver supply, end to end](driver-supply.md) | an incentive between two zones, from confounded logs to a certified plan run on the true plant | `uv run python scripts/spine_demo.py` |
| [Confounded incentives, closed loop](confounded-incentives.md) | an incentive under hidden confounding, when churn costs more than wasted budget | `uv run python scripts/run_dynamic_confounding_demo.py` |

Run them from a checkout after `uv sync`. None of the scripts enables `x64`, so JAX runs them at its
default precision; see [the dtype policy](../concepts/dtype-policy.md).
