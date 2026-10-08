# ADR 0077 — A benchmark oracle is the best plan found

**Status:** accepted, 2026-10-08.

## Context

Every task in `chc.benchmark` reads its regrets against an oracle on the true plant: a regret is a
controller's cost less the oracle's. That reads the distance from the best plan only where the
oracle is the best plan, and four of the oracles are a local descent from a plan of zeros.

`ModelUncertaintyTask`'s plant acts through `u - 0.15 u^3` in a box of ±8. Past `|u| = 2.58` the
effect reverses, and on the box's edge it is -68.8, so a plan can push harder through the reversed
effect than through the forward one, whose peak is 0.99. The descent from zero stops at that peak,
at 15.520831, and ADR 0059 measured a plan with one action on the edge at 4.797. Its regrets, 0.653
for the calibrated plan and 1348.84 for greedy, read 11.75 too low.

Each oracle, and every reference a regret is read against, was searched for a better plan. The
plan tasks were descended by `projected_gradient_solve` from 73 or 74 starts: zero, each side of the
box throughout, at each single step, and for the first 1, 2, 3, 5 or 10 steps, the task's other
plans and 8 seeded draws. Where the plant is linear the cost is a convex quadratic in the plan, and
its exact minimum over the box was read by bounded-variable least squares (Stark and Parker 1995).
Every figure is in float64, as the suite runs.

| task | oracle | its cost | best found | from | oracle less best |
|---|---|---|---|---|---|
| `PricingTask` | the certainty-equivalent rule at the true effect | 4.606150 | 4.596669, the LQR policy | the finite-horizon Riccati recursion | 0.0095 (0.00098 on average) |
| `InventoryTask` | the critical fractile of the true demand | 0.684746 | 0.684689 | the 5 000 draws' own fractile | 5.7e-5 |
| `SupportShiftTask` | the descent from zero | 20.645062754916 | 20.645062754914 | either side throughout | 2.2e-12 |
| `ModelUncertaintyTask` | the descent from zero | 15.520831 | 3.773697 | the upper side for two steps | 11.747134 |
| `ConfoundingRobustTask` | the descent from zero | 8.306851863451 | 8.306851863382 | the exact box QP | 6.9e-11 |
| `CausalDynamicsTask` | the descent from zero | 6.103079627998 | 6.103079627998 | the exact box QP | 3.4e-13 |
| `DelayOscillationTask` | the best of 400 gains at the true delay | 0.5681766 | 0.5681609 | 20 001 gains between its neighbours | 1.6e-5 |

- **`PricingTask` and `InventoryTask`** have no descent: each oracle is the controllers' own rule
  at the true effect. On pricing the rule ignores the control's weight; over 4 000 noise paths the
  finite-horizon LQR policy, which weighs it, costs 0.00098 less on average, standard error 0.0001.
  The larger gaps on the drawn path come from plans that read it: the rule tuned to the path, an
  effect of 1.03, costs 0.029 less there and 0.018 more on average, and a plan that reads the
  path's noise costs 0.29 less. On inventory the cheaper order is the fractile of the 5 000 draws
  the order is scored on.
- **`SupportShiftTask`**: past `0.8 / sqrt(2)` an action buys an effect a smaller one gives for
  less, and inside it the effect rises with the action and the action's cost is convex in the
  effect, while the plant is linear in the effect. The oracle's actions all lie inside, so it is a
  stationary point of a convex problem, the best plan. The 73 descents end on 39 plans.
- **`ConfoundingRobustTask` and `CausalDynamicsTask`**: linear plants, so one plan; every descent
  ends on it.
- **`DelayOscillationTask`**: the cost has one minimum on the grid, and the refinement's gain
  saves less than the 8.3e-5 the cost rises to the grid's nearer neighbour.
- **`ModelUncertaintyTask`** was searched further. 152 starts -- zero, each side throughout, at one
  step, on opposite sides at two of the first eight steps, and for the first 1, 2, 3, 5 or 10
  steps, the relaxation's plan below read back through its cheapest actions, and 32 seeded draws
  -- were each descended for 10 000 steps, then 200 000, then polished by L-BFGS-B, and again by
  the three solves of the Decision. A dynamic programme over the state, on grids of 71 by 161 and
  141 by 321 states, ended at 3.774214 and 3.774506 once descended. The descents end on many
  plans; none is below 3.773697. Every plan costs at least 3.76084: the plant is linear in the
  effect `e`, the state's cost is a convex quadratic in the effects, and the control's cost is at
  least its weight times the convex envelope of `min{u^2 : u - 0.15 u^3 = e}`. That relaxation is
  convex, and a subgradient at its solution bounds its minimum (Frank and Wolfe 1956). So 3.773697
  is within 0.0129 of the best plan there is.

## Decision

- **An oracle that a better plan beats by more than its solver's tolerance is the best of a fixed,
  documented, deterministic set of starts.** Of the seven tasks only `ModelUncertaintyTask`'s is.
  Its starts are zero, then the upper and the lower side of the box held for the first one, two
  and three steps: seven, in that order, a later plan replacing an earlier one only where it costs
  strictly less. The edge is where the reversed effect is strongest, and the prefixes are where a
  plan that starts at 2 and must reach 0 spends it.
- **Each start is descended, polished and descended again.** The planner alone crawls along this
  plant's ravines: from the upper side held for two steps it ended 10 000 steps at 3.926 and
  200 000 at 3.773716, its stationarity 3.3e-3. `lbfgs_box_control` goes on from where it stopped,
  and the planner descends again to its own rule, at 3.773697, its stationarity 3.7e-6. Each of the
  three runs to the task's `inner_steps` at most.
- **The search runs once per task.** It reads no data, so `run_multiseed`'s seeds share it; it is
  kept per task and per precision.
- **Every other oracle keeps its plan, to the bit**, and each task's docstring says what its
  oracle is and how far from the best it is known to be, with the table's figures.

## Consequences

- **`ModelUncertaintyTask`'s regrets are read against the best plan found.** Its oracle costs
  3.773697, where it cost 15.520831; the calibrated plan's regret is 12.400, where it read 0.653,
  and greedy's 1360.58, where it read 1348.84. The oracle leaves the logged action support at 4 of
  its 25 steps, where it left it at 24. `tests/test_benchmark.py` holds the oracle at or below the
  best plan found, to `1e-9`, and within 0.013 of the relaxation's bound.
- **The calibrated row now reads what staying on the learned model costs.** A regret of 12.4 is
  mostly the 11.75 the reversed effect is worth, which no model fit on the logged support can see. Greedy
  still loses two orders more, and the row's ordering is unchanged.
- **The task costs one search more, once.** The seven solves take 21 s on a CPU under float64, in
  the first seed a task runs; each seed after it takes 25 s, its ensemble fit and its two
  controllers, as before.
- **The six other tasks print what they printed**, so notebooks 04 and 05, which run pricing,
  inventory and support-shift, print the same numbers.

## Alternatives

- **Run every descent oracle from the seven starts.** Rejected: on the three tasks where the oracle
  is already the best the starts change its cost in the last bits, up to 2.2e-12, for nothing.
- **The planner alone, with a larger budget.** Rejected: at 200 000 steps from the best start it is
  still 1.9e-5 above the plan, with a stationarity of 3.3e-3, and from five of the seven starts it
  runs out of steps.
- **L-BFGS-B alone.** Rejected: from the upper side held for two steps it ends at 11.50, a plan at
  the forward effect's peak.
- **The relaxation's plan, read back through the cheapest action for each effect, as the start.**
  Rejected: under the same three solves it ends at 3.774214, a nearby plan 5.2e-4 above the best,
  and it needs a plant linear in a scalar effect, which this task is by construction and a plant
  in general is not.
- **A dynamic programme over the state.** Rejected: its grid's interpolation lands it on nearby
  plans, 3.774214 and 3.774506, and its time grows with the grid.
- **The 152 starts of the search.** Rejected: the seven reach the same plan in 7 of its 152
  solves.
- **LQR as pricing's oracle.** Rejected: a regret there reads what the estimate costs under the
  controllers' rule, and the LQR policy would add the rule's own 0.001 to every row. The rule's
  distance from the best policy is in its docstring.
- **The evaluation draws' own fractile as inventory's oracle**, or **a finer gain grid for the delay
  task.** Rejected: the first reads the draws the order is scored on; the second moves the oracle by
  less than its grid's resolution.
