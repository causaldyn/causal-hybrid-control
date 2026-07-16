# %% [markdown]
# # 5 · The benchmark scoreboard — where causal control wins, and by how much
#
# One anecdote is not evidence. `chc` ships a small **benchmark suite** of control tasks that each carry
# a *known* ground-truth interventional effect, so we can measure **regret vs an oracle** that plans on
# the true system. Across three structurally different tasks, the same pattern holds: the CHC controller
# lands next to the oracle, while the predictive / greedy baseline is catastrophic.
#
# | task | what the controller decides | the trap |
# |---|---|---|
# | **pricing** | how hard to steer a KPI to target | confounded logs flip the action's sign |
# | **inventory** | how much stock to order (newsvendor) | biased demand-response ⇒ systematic mis-order |
# | **support-shift** | how large an action to take | model is trusted where it was never trained |
#
# Metrics per controller: **cost**, **regret** (`cost − oracle_cost`, lower is better), **constraint
# violations**, and **out-of-support action rate**.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

%matplotlib inline

from chc.benchmark import InventoryTask, PricingTask, SupportShiftTask

TASKS = {
    "pricing": PricingTask(),
    "inventory": InventoryTask(),
    "support-shift": SupportShiftTask(),
}
# the "CHC" controller and the naive baseline differ per task; name them so we can compare fairly
CHC_OF = {"pricing": "causal-CHC", "inventory": "causal-CHC", "support-shift": "pessimistic"}
BASE_OF = {"pricing": "predictive", "inventory": "predictive", "support-shift": "greedy"}

rows = []
for task_name, task in TASKS.items():
    for r in task.run():
        rows.append(
            {
                "task": task_name,
                "controller": r.controller,
                "cost": r.cost,
                "regret": r.regret,
                "violations": r.constraint_violations,
                "ood": r.ood_rate,
            }
        )
board = pd.DataFrame(rows)

# %% [markdown]
# ## The full scoreboard
#
# Read it by task: the **oracle** is the unreachable best, the **CHC** controller sits right next to it,
# and the **baseline** pays a large regret — and, where applicable, violates constraints and acts far
# outside the logged support.

# %%
styled = (
    board.set_index(["task", "controller"])
    .round(3)
    .style.background_gradient(subset=["regret"], cmap="Reds", low=0, high=1)
    .format({"cost": "{:.2f}", "regret": "{:.2f}", "violations": "{:.2f}", "ood": "{:.2f}"})
)
styled

# %% [markdown]
# ## The headline: regret, CHC vs the naive baseline
#
# Regret spans several orders of magnitude, so the bars are on a **log scale** (a floor of `0.01` keeps
# the near-zero oracle/CHC bars visible). The gap *is* the value of the library.

# %%
def regret_of(task: str, controller: str) -> float:
    match = board[(board["task"] == task) & (board["controller"] == controller)]
    return float(match["regret"].iloc[0])


summary = pd.DataFrame(
    {
        "CHC (causal / pessimistic)": [regret_of(t, CHC_OF[t]) for t in TASKS],
        "naive baseline (predictive / greedy)": [regret_of(t, BASE_OF[t]) for t in TASKS],
    },
    index=list(TASKS),
)

fig, ax = plt.subplots(figsize=(8, 4.2))
floor = 0.01
y = np.arange(len(summary))
h = 0.36
ax.barh(y + h / 2, summary.iloc[:, 0].clip(lower=floor), h, color="#54A24B", label=summary.columns[0])
ax.barh(y - h / 2, summary.iloc[:, 1].clip(lower=floor), h, color="#E45756", label=summary.columns[1])
ax.set_yticks(y)
ax.set_yticklabels(summary.index)
ax.set_xscale("log")
ax.set_xlabel("regret vs oracle  (log scale, lower is better)")
ax.set_title("Across every task, CHC stays near the oracle; the baseline blows up")
for i, (chc, base) in enumerate(zip(summary.iloc[:, 0], summary.iloc[:, 1], strict=True)):
    ax.text(max(chc, floor), i + h / 2, f" {chc:.2g}", va="center", fontsize=9)
    ax.text(max(base, floor), i - h / 2, f" {base:.1f}", va="center", fontsize=9, fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## The win in one number per task

# %%
factor = pd.DataFrame(
    {
        "CHC regret": summary.iloc[:, 0].values,
        "baseline regret": summary.iloc[:, 1].values,
        "regret reduction": [
            f"{b / c:,.0f}x lower" if c > 0.05 else "baseline -> ~0 (CHC = oracle)"
            for c, b in zip(summary.iloc[:, 0], summary.iloc[:, 1], strict=True)
        ],
    },
    index=list(TASKS),
).round(3)
factor

# %% [markdown]
# ### Takeaway
# The advantage is not one lucky example. On **confounded steering** (pricing), **biased demand response**
# (inventory), and **off-support model exploitation** (support-shift) — three different failure modes —
# CHC recovers near-oracle control while the predictive/greedy baseline is orders of magnitude worse. The
# benchmark, with its ground-truth effects, is what lets us *quantify* the win instead of asserting it.
