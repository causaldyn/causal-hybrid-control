# Quickstart

From a panel of logs to a schedule and its certificate, in one call. The causal assumption is a
**required** argument, because the default would be "adjust for nothing", which is a claim rather
than its absence — and when the graph says the effect is not identified there is no schedule at all.

Everything below is one script,
[`docs/quickstart.py`](https://github.com/causaldyn/causal-hybrid-control/blob/main/docs/quickstart.py),
and the report at the end is what it printed when this site was built:

```bash
pip install causal-hybrid-control
python quickstart.py
```

## 1. Precision first

<!-- The fences down to section 4 hold snippet includes, which ruff would format as Python. -->
<!-- fmt:off -->
```python
--8<-- "docs/quickstart.py:precision"
```

JAX computes in single precision unless told otherwise, and an identified channel is a number
someone will act on. Set the flag before anything creates an array. `prescribe` does not refuse a
single-precision panel; it warns, and records the flag in the result's provenance. The
[dtype policy](concepts/dtype-policy.md) says why.

## 2. The logs

```python
--8<-- "docs/quickstart.py:logs"
```

A driver pool, logged weekly per region: `supply`, the riders' `wait`, the `incentive` paid, and the
`demand` shock. The policy that wrote the log raised the incentive whenever demand spiked, and
demand also moves supply — so a model fitted on this log credits the incentive with what demand
did.

`Panel.from_frame` takes a pandas frame, a polars frame or a mapping of columns, and neither frame
library is a dependency. It checks the `(unit, time)` index once and names the column *and* the
entity when it refuses.

## 3. The assumption, and the call

```python
--8<-- "docs/quickstart.py:prescribe"
```

<!-- fmt:on -->

The same decision, read under two causal assumptions: once with `demand` logged, once with it
declared latent. `prescribe` composes what is below it and adds no new estimator, solver or
guarantee: the adjustment set comes from the graph, the control channel from cross-fit Robinson
DML (`fit_causal_residual`), the plan from `causal_plan`, the safety price from `certify_safety`,
and how far the plan can be from the best one its own box allows on the fitted model from
`plan_regret_bound`, a diagnostic here: the fitted channel reads the state, so the curvature is
sampled. What each certificate bounds, and on what: [the scopes](concepts/certificates.md#scopes).

## 4. What it printed

```markdown
--8<-- "docs/output/quickstart.txt"
```

Read the first report by its two axes, which are kept apart on purpose:

- **Identification** is about the data and the graph: whether an adjustment set exists, which one
  was used, and the channel's standard error.
- **Certification** is about model error and the barrier: the error tube's certified horizon, over
  the plan's RK4 steps and at the channel's standard error, so a scale rather than a bound; the
  steps whose predicted states clear the constraint at the level of the marginal sensitivity model
  the plan was audited at (`gamma`, when a bound is given), checked pointwise; and `gamma*`, the
  largest level at which every step along the plan's path still has an admissible action that
  meets the barrier, a ceiling for the problem along that path rather than for the plan.

The **trustworthy prefix** is the part of the schedule that survives *both* — the number an
operator can act on. A plan can be fully certified against a channel nothing identifies, which is a
trustworthy tube around a meaningless action, so neither axis is derived from the other.

The second report is the case this library exists for. With `demand` never logged, no observed set
identifies the effect, so there is no schedule: `out.schedule` raises `NotIdentifiedError` rather
than hand back actions that would look like every other schedule and mean nothing. `prescribe` also
logs a `WARNING` on the `chc.decision` logger when that happens. What is left is to price how much
confounding the decision can tolerate — [the sensitivity level Γ](concepts/gamma.md).

`out.to_json()` carries the same schedule, both axes and the provenance as plain JSON values.

## The expert path

`prescribe` wraps [`fit_causal_residual`](api/dynamics_id.md) and [`causal_plan`](api/plan.md),
and both take a hand-built plant, cost and barrier. Below, the model is built by hand: a known
oscillator plus a residual fitted to logged transitions. The actions in this log were drawn at
random, so plain least squares (`fit_residual`) is enough; on a confounded log,
`fit_causal_residual` fits the control channel instead. The script is
[`docs/expert_path.py`](https://github.com/causaldyn/causal-hybrid-control/blob/main/docs/expert_path.py).

<!-- fmt:off -->
```python
--8<-- "docs/expert_path.py:fit"
```

[`mpc_control`](api/mpc.md) then re-plans at every step on the fitted model and applies the first
action to the plant, which `prescribe` does not:

```python
--8<-- "docs/expert_path.py:mpc"
```

Online, where the plant is the world rather than a simulation, `RecedingHorizon` plans from each
measured state and returns the whole `causal_plan`: actions, tube and audit together.

```python
--8<-- "docs/expert_path.py:receding"
```
<!-- fmt:on -->

What it printed when this site was built:

```text
--8<-- "docs/output/expert_path.txt"
```

Each step starts from the last plan and the barrier's multipliers, shifted one step on: 35 % fewer
descent steps than cold solves on an oscillator's velocity floor, for the same closed-loop cost to
`2e-6` (ADR 0003, [receding-horizon warm starts](https://github.com/causaldyn/causal-hybrid-control/blob/main/docs/adr/0003-receding-horizon-warm-starts.md)).
A program compiles the first time a step needs it and is reused after, so steps stop compiling as
long as the model, the cost and the barrier's function stay the same objects: a new `lambda` per
step compiles the descent per step. For a process that restarts, set `jax_compilation_cache_dir`
and lower `jax_persistent_cache_min_compile_time_secs` to 0: at its default of one second JAX wrote
none of a first step's programs to disk.

## Where next

- [Concepts](concepts/identification.md) — what each part of the report means, and what it does
  not promise.
- [Tutorials](tutorials/index.md) — the executed notebooks, from a first decision to real data.
- [What's inside](modules.md) — every module, with what it does and what was measured.
