# Tutorials

The notebooks under
[`notebooks/`](https://github.com/causaldyn/causal-hybrid-control/tree/main/notebooks), executed
top to bottom when this site was built: every figure, table and printed number on these pages came
out of that run, not out of the outputs saved in the repository.

| notebook | what it shows |
|---|---|
| [0 · Start here: a decision from logs](00_start_here.md) | the whole loop on one everyday problem: the trap in the logs, the effect, the plan, and how far to trust it |
| [1 · Causal vs predictive control](01_causal_vs_predictive_control.md) | the headline: under confounding, predictive control drives the state the wrong way (to −20, target +2); causal control matches the oracle |
| [2 · Learning what physics misses](02_learn_hidden_physics.md) | hybrid dynamics + system ID: recover an omitted cubic term; multi-step training cuts drift from noisy measurements |
| [3 · The causal inference toolkit](03_causal_inference_toolkit.md) | adjustment · IV/2SLS · Double ML · sensitivity · refutation, side by side |
| [4 · Scientific control & offline safety](04_epidemic_and_pessimism.md) | flatten an epidemic curve down to a capacity cap; pessimism vs a greedy controller |
| [5 · The benchmark scoreboard](05_benchmark_scoreboard.md) | regret vs oracle across three tasks — CHC matches the oracle under confounding and beats the greedy baseline on support-shift |
| [5 · Robust control under hidden confounding](05_confounding_robust_control.md) | when **no adjustment set exists**: a sensitivity level `Γ` → pessimism radius → minimax action, and what it costs when there is no confounding or the bias runs the other way |
| [6 · Adaptive cruise control from fleet logs](06_cruise_control_confounded.md) | relatable end-to-end: confounded fleet logs (Simpson's paradox → adjustment or IV → control) |
| [7 · The LaLonde validation](07_real_data_lalonde.md) | **real data, experimental ground truth**: the naive estimate flips sign; adjustment and Double ML land inside the experiment's 95% interval |
| [8 · Splitting an advertising budget](08_splitting_a_budget.md) | three channels, one budget: why the best split equalises the next euro's return, and when TV switches on |
| [9 · Heating a room on a winter night](09_heating_a_room.md) | a thermostat's logs, the weather that confounds them, and a night plan that keeps the room warm |

## Run them yourself

In a checkout of the repository:

```bash
uv sync --group notebooks
uv run --group notebooks jupyter lab
```

Each `.ipynb` has a paired jupytext `.py` source beside it. Notebook 7 downloads the
Dehejia–Wahba files from NBER on first use into `notebooks/data/`, which git ignores; nothing is
committed. To rebuild these pages, `just docs` executes them all and then builds the site.
