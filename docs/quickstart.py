"""The quickstart: from a panel of confounded logs to a certified schedule, in one call.

Run: uv run python docs/quickstart.py
"""

from __future__ import annotations

# --8<-- [start:precision]
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)  # before any array exists; see the dtype policy

from chc import CausalGraph, Constraint, Lever, Panel, Target, prescribe  # noqa: E402

# --8<-- [end:precision]

# --8<-- [start:logs]
DT = 0.1  # one logged period, in the model's time units


def simulate_logs(regions: int = 200, weeks: int = 12, seed: int = 0) -> dict[str, np.ndarray]:
    """A driver pool logged under a policy that raised the incentive whenever demand spiked."""
    rng = np.random.default_rng(seed)
    names = ("region", "week", "supply", "wait", "incentive", "demand")
    rows: dict[str, list[float]] = {name: [] for name in names}
    for region in range(regions):
        supply, wait = rng.normal(0.0, 0.2), rng.normal(0.0, 0.2)
        for week in range(weeks):
            demand = rng.normal(0.0, 1.0)
            incentive = 0.9 * demand + rng.normal(0.0, 0.5)  # the policy chases demand
            logged = (region, week, supply, wait, incentive, demand)
            for name, value in zip(names, logged, strict=True):
                rows[name].append(value)
            supply, wait = (
                supply
                + DT * (-0.6 * supply + 0.3 * wait + 0.8 * incentive + 1.5 * demand)
                + rng.normal(0.0, 0.01),
                wait + DT * (0.25 * wait - 0.4 * incentive) + rng.normal(0.0, 0.01),
            )
    return {name: np.asarray(values) for name, values in rows.items()}


panel = Panel.from_frame(simulate_logs(), unit="region", time="week", seed=0)
# --8<-- [end:logs]

# --8<-- [start:prescribe]
edges = [
    ("demand", "incentive"),  # the confounding: the policy looked at demand
    ("demand", "supply"),
    ("incentive", "supply"),
    ("incentive", "wait"),
    ("wait", "supply"),
]
for assumption in (
    CausalGraph.from_edges(edges),  # demand was logged, so it can be adjusted for
    CausalGraph.from_edges(edges, latent=("demand",)),  # demand was never logged
):
    out = prescribe(
        panel,
        adjustment=assumption,
        levers=[Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05)],
        target=Target("supply", value=1.0),
        constraints=[Constraint("wait", hi=0.5)],
        horizon=15,
        dt=DT,
        tolerance=0.5,
    )
    print(out.report(), end="\n\n")
# --8<-- [end:prescribe]
