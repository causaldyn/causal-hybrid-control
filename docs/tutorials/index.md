# Tutorials

The eight notebooks under
[`notebooks/`](https://github.com/causaldyn/causal-hybrid-control/tree/main/notebooks), executed
top to bottom when this site was built: every figure, table and printed number on these pages came
out of that run, not out of the outputs saved in the repository.

| notebook | what it shows |
|---|---|
| [1 · Causal vs predictive control](01_causal_vs_predictive_control.md) | the headline: predictive control diverges under confounding, causal control matches the oracle |
| [2 · Learning what physics misses](02_learn_hidden_physics.md) | hybrid dynamics + system ID: recover an omitted cubic term; multi-step training cuts drift |
| [3 · The causal inference toolkit](03_causal_inference_toolkit.md) | adjustment · IV/2SLS · Double ML · sensitivity · refutation, side by side |
| [4 · Scientific control & offline safety](04_epidemic_and_pessimism.md) | flatten an epidemic curve under a capacity cap; pessimism vs a greedy controller |
| [5 · The benchmark scoreboard](05_benchmark_scoreboard.md) | regret vs oracle across three tasks — CHC lands next to the oracle, the baseline blows up |
| [5 · Robust control under hidden confounding](05_confounding_robust_control.md) | when **no adjustment set exists**: a sensitivity level `Γ` → identification radius → minimax action, and its premium when there is no confounding |
| [6 · Adaptive cruise control from fleet logs](06_cruise_control_confounded.md) | relatable end-to-end: confounded fleet logs (Simpson's paradox → IV → control) |
| [7 · The LaLonde validation](07_real_data_lalonde.md) | **real data, experimental ground truth**: the naive estimate flips sign, Double ML recovers the randomised benchmark |

## Run them yourself

In a checkout of the repository:

```bash
uv sync --group notebooks
uv run --group notebooks jupyter lab
```

Each `.ipynb` has a paired jupytext `.py` source beside it. Notebook 7 downloads the
Dehejia–Wahba files from NBER on first use into `notebooks/data/`, which git ignores; nothing is
committed. To rebuild these pages, `just docs` executes all eight and then builds the site.
