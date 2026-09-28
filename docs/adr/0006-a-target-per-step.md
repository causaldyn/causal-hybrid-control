# ADR 0006 — A target per step

**Status:** accepted, 2026-09-28.

## Context

`QuadraticCost` had one `x_target` for every state of a plan, and `Target.value` in `prescribe`
was one number. So a set point that moves inside the horizon could reach a plan only through
re-planning, when a later call passes the new level.

On the BOPTEST heat pump (`docs/case-studies/boptest.md`, run L8.1) the lower comfort bound
follows an occupancy schedule, so it moves inside the plan's eight hours. The harness passed the
highest lower bound forecast over those hours, so every plan steered for a higher bound from the
moment it entered the horizon: the case study's "pre-heats up to one horizon early". Passing the
current bound instead would steer for a higher one only once it applied. Neither is the problem as
stated.

## Decision

- `QuadraticCost.x_target` may be `(H + 1, n)`, one row per state of an `H`-step plan: row `k` is
  the target of `x_k`, and the last row the terminal one. A `(n,)` target is one row for every
  state, as before.
- `running` and `terminal` take the row as an optional `target`, and `targets(H)` hands a consumer
  the `(H + 1, n)` rows, broadcasting a fixed target. A row means nothing without its step, so
  `running(x, u)` on a per-state target raises rather than subtract the whole table from `x`.
- Every consumer of the stage costs reads the rows: `total_cost`, the discrete adjoint and its
  siblings `costate_norms` and `perturbation_cost_weights`, `total_cost_diffrax`, and `lift_cost`,
  which lifts each row across the delay line's buffer.
- `mpc_control` and `RecedingHorizon` refuse a per-state target. Both re-plan with one cost, so the
  window would stay where it was while the loop moved on.
- `Target.value` in `prescribe` may be a schedule with one level per step: `value[k]` is the level
  for the state after `k + 1` actions, so a schedule has `horizon` entries, and any other length
  raises `DecisionError`. The start's row repeats the first level. No action reaches the start, so
  that row moves the reported `task_cost` and nothing the plan does.

## Evidence

The panel of `tests/test_decision.py` (200 units, 12 periods, seed 0; `horizon=15`, `dt=0.1`,
`tolerance=0.5`, `unit_cost=0.05`). Supply is to stay at 0 for seven steps and be at 0.8 from the
eighth. Each plan is priced against that schedule, in float64:

| target passed | cost against the schedule | supply after 7 steps | rms off 0, steps 1–7 | rms off 0.8, steps 8–15 | final supply |
|---|---|---|---|---|---|
| the schedule | 0.557 | 0.314 | 0.163 | 0.190 | 0.665 |
| constant 0 | 2.561 | 0.001 | 0.008 | 0.800 | −0.001 |
| constant 0.8 | 1.532 | 0.711 | 0.545 | 0.069 | 0.694 |

- **Every plan converged** and was trusted over all 15 steps, with a regret bound under `5e-8`.
- **The schedule's plan rises before the level moves.** Supply is at 0.314 after the seventh
  step, the last one whose level is 0. Constant 0 does not rise before a later call re-plans;
  constant 0.8, L8.1's choice in miniature, leaves 0 at once. Against the schedule they cost 4.6
  and 2.7 times what the schedule's own plan does.
- **Float32** gives the same table to three decimals except in the constant 0.8 row: 1.529
  against the schedule and 0.544 off 0. Its regret bounds reach `1.6e-5`.
- **Fifteen equal levels are the scalar level.** They plan the same actions to the bit, in both
  precisions. A `(H + 1, n)` cost whose rows all agree gives a fixed target's values to the bit,
  in float64, through `total_cost`, the adjoint, the costate norms and the perturbation weights.
- **Mutation checks.** Reading the rows one state late, in `total_cost` or in the adjoint alone,
  fails `tests/test_adjoint.py`. Mapping `value[k]` to `x_k` instead of `x_{k+1}`, or reading
  only the first level, fails `test_a_schedule_is_steered_for_before_it_moves`, whose Bolza sum
  is written out by hand.

## Consequences

- The stable `cost` module gains an input shape and an optional argument. A fixed target computes
  what it did, to the bit.
- A subclass of `QuadraticCost` whose `running(x, u)` takes no `target` would now be handed a third
  argument by every consumer. The library has none.
- A loop that re-plans with `prescribe` passes the schedule's next `horizon` levels on each call.
  `RecedingHorizon` cannot carry a schedule yet: that needs its `step` to take the window, which is
  a separate decision.
- The regret bound and every certificate are priced on the scheduled objective. Nothing about how
  they are priced changes.

## Alternatives

- **A callable target, `x_target(k)`.** Rejected: a Python function in a module is static, so every
  schedule would compile its own programs, while rows of data share one per shape.
- **A clock coordinate in the state.** Rejected: a target that is a function of a state coordinate
  makes the cost non-quadratic. (This record first gave a second reason, that rollouts pass `t = 0`
  to every step. They do not: `rollout` steps `k * dt`.)
- **A separate `TrackingCost`.** Rejected: every consumer would accept two cost types, and a fixed
  target is one row repeated.
- **Re-planning alone.** Rejected: the table above, and L8.1's early pre-heating.

## References

- Rawlings, J. B., Mayne, D. Q. & Diehl, M. M. (2017). *Model Predictive Control: Theory,
  Computation, and Design*, 2nd ed. Nob Hill Publishing.
