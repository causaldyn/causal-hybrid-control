# ADR 0076 — Every planner searches S-curves

**Status:** accepted, 2026-10-08. Amends ADR 0051, whose other planners kept the envelope from
zero spend: `budget_for`, `minimax_allocate`, `allocate_geos` and `budget_for_geos` now search as
`allocate` does. Amends ADR 0037 and ADR 0044 (a goal on S-curves is met over searched plans),
ADR 0038 (a regret is read on the curves, against `allocate`'s best), ADR 0043 (S-shaped cells are
searched in boxes) and ADR 0065 (every planner takes the search's settings, and no plan is
`unsearched`).

## Context

ADR 0051 searched S-shaped curves by branch and bound in `allocate`, and ADR 0052 did the same in
`cvar_allocate`. The four other planners kept the plan on the concave envelopes from zero spend.
Since 0.13.0 each such plan said `stopped="unsearched"` and logged its gap. That plan is not the
best, and a goal met along such plans is not met at the least budget:

- `allocate_geos`, on the untied example as one geo (two Hill curves of slope 3 at scales 1 and
  1.01, a budget of 1.6), planned 1.27/0.33 for 0.7048, where `allocate` puts all of it on the
  first for 512/637 = 0.8038.
  A second geo of Hill(0.8, 4) at 0.7 and Hill(1.2, 2.5) at 1.2 joined the first, each cell capped
  at 2. At budgets of 1.6 and 3.2 that grid returned 0.7048 and 1.5523, where the best splits of
  its four cells return 0.8069 and 1.6334. Under totals the planner lost more. At 3.2, with each
  geo's total held at 1.6, it returned 1.4369 where each geo's own best returns 1.6107. With each
  channel's at 2.0 and 1.2 it returned 1.5737 where 1.6294 is the best.
- `budget_for`, on the untied pair, met a gain of 0.7 at 1.5834 and 0.75 at 1.7137. There
  `allocate` gains 0.7988 and 0.8342. All of the budget on the first channel meets them at
  (7/3)^(1/3) = 1.3264 and 3^(1/3) = 1.4422. A floor of 1 on a Hill, short of its tangency, held
  it at the floor for a marginal return of 0.55, where the Hill's own slope is 0.75.
- `budget_for_geos` did the same. On the grid a gain of 1.2 took 2.3175 and 2.4 took 4.7787,
  where `allocate`'s plan of the four cells gains 1.2104 and 2.4612. A gain of 300 from a Hill of
  scale 100 beside a saturating channel was met on a plan that was already the best, but its gap
  of 98.9 was logged.
- `minimax_allocate` read each reading's best as the bound of its plan on the envelopes, and the
  regrets on the envelopes too. The untied pair, read once each way round, was split evenly: a
  worst regret of 0.0024 on the envelopes and of 0.1332 on the curves, where all on either channel
  leaves 0.00475. A Hill reading beside a saturating one, with a budget of 90, put 87.6 on the
  Hill. Against `allocate`'s bests, the worst regret on the curves was 12.42, where 6.65 is the
  least.

The cutting planes of `allocate_geos` and `minimax_allocate` already bounded the envelopes, as
those of `cvar_allocate` do. The bound was not the obstacle; two things were:

- `allocate_geos` makes its plan exact by Newton's method on the prices of the totals that bind,
  with a line search. On an envelope's chord a cell's rate jumps across the chord at the chord's
  slope, and the line search circled the jump. A search of the grid above at 1.6, with that
  polish in each of its 19 boxes, read a cell's slope 2,093,555 times, every one in the polish.
- A goal's bisection on the price follows one path of plans. Plans on S-curves lie on no one
  path.

## Decision

- **`allocate_geos` searches boxes of the cells' rates** as `allocate` does. A box's bound is that
  of the cutting planes on each cell's envelope over the box, read from the duals (ADR 0060), with
  the geos' and channels' totals as the program's rows. The box of the largest bound is cut first.
  The cut is on the cell inside its interval whose envelope stands furthest above its curve at the
  box's plan, at its rate there, so both halves hold the plan and meet the totals. Where no
  envelope stands above its curve, the cut is on the cell whose interval is widest against its
  range. A box whose plan has every cell at an end of its interval is not cut, and its bound
  stands.
- **The tolerance, the cap and the stop reason are ADR 0065's.** `allocate_geos` takes `rtol`,
  `atol` and `max_boxes`. `GeoAllocation` records `stopped`, `boxes`, `gap`, `relative_gap`,
  `tolerance`, `floored`, `limit`, `compile_seconds` and `search_seconds`. A search that the cap
  stops logs `chc_event="allocation_cap"`. Where every curve is concave, one box is planned, as
  before.
- **In a search, Newton's method stops at its first step that does not shrink the totals' miss.**
  The planes' plan then stands, within the box's gap. Where every curve is concave, Newton's method
  runs as before, line search included.
- **A goal on S-curves is met over searched plans.** Each budget's plan is the search of
  `allocate`, or of `allocate_geos`, at that budget. A return target is met by Brent's method on
  the budget, where the plan's gain crosses it. Without totals that gain rises with the budget, so
  the budget is the least. A marginal target is the best plan with the budget free and each unit
  charged the target, searched in boxes. A return on spend is met by Brent's method past that
  plan's budget. Concave curves are met as before.
- **`minimax_allocate` reads each reading's best as the return of `allocate`'s plan**, and each
  regret on the curves. The least worst regret is the worst share of `cvar_allocate` at one
  reading of n, each reading's best in place of a reference's return. It is searched in boxes, as
  `cvar_allocate` searches, from every reading's own plan. Before any plane, a box is bounded by
  the least of the readings' own searches' gaps: no split gains more than that over every
  reading's best.
- **`budget_for`, `budget_for_geos` and `minimax_allocate` take `rtol`, `atol` and `max_boxes`**
  too. `MinimaxAllocation` records the search as `Allocation` does. `SearchStatus` drops
  `unsearched`, and the warning `chc_event="allocation_unsearched"` is gone: no planner returns a
  plan it has not searched.

## Consequences

- Plans change on S-curves only. On the cases above the planners now return, to the tolerance:
  0.8038 for the one geo; 0.8069 and 1.6334 for the grid, `allocate`'s plan of its cells; 1.6107
  and 1.6294 under the totals, the best plan of each geo and of each channel; budgets of 1.3264 and
  1.4422; budgets of 2.3009 and 4.6717 on the grid; a floor-held Hill at 1.2353, where its slope
  meets 0.55; and worst regrets of 0.00475 and 6.65. The goal of 300 returns the same plan, now
  closed. With the guarded Newton steps the search above at 1.6 reads a slope 687 times, in place
  of 2,093,555, for the same worth to the bit.
- A plan that the planes leave inside a cell's interval, short of exact rates, is within its box's
  gap of the best. Without the polish in a box, the grid's rates came back 1e-5 from the best
  split's, though within the tolerance. With the guarded Newton steps the worths of the grid's
  six cases, free and under each kind of total at 1.6 and 3.2, are within 1e-14 of `allocate`'s.
- Concave curves plan as before. Sixteen plans were compared to the bit with the tree before
  this change, every field that existed before: `allocate_geos` free, under totals and on a
  linear cell; each goal over geos and over one geo's channels; `allocate`; `cvar_allocate` on
  concave and on S-shaped readings; and `minimax_allocate` on one reading and on two mirrored
  linear readings. Fifteen are equal. The sixteenth is `minimax_allocate` over the tests' four
  readings with carryover, which now runs `cvar_allocate`'s program: its split moves by 3.1e-8,
  its worst regret falls by 1.3e-9 of 213.2 and its bound rises by 2.4e-12, inside the same gap
  of 1.1e-5.
- A goal on S-curves costs one search a budget its root finder tries: 9 to 11 on the tests'
  goals, each search a few boxes. A marginal target costs one search with the budget free and one
  at its budget.
- `minimax_allocate` logs the gap of a reading's own search that its cap stops, as `allocate`
  logs it, and the gap of its own split.
- `stopped="unsearched"` and `allocation_unsearched` are gone. A caller who branched on them reads
  `stopped="cap"` and `allocation_cap` on the gap that a cap leaves.

## Alternatives

- **The polish with its line search in every box.** It circled each chord's jump: 2,093,555
  reads of a slope where the guarded steps need 687, for the same plan. Rejected.
- **No polish in a box.** The planes' plan is within the gap, but on the grid its rates missed the
  best split's by 1e-5, and its worth by 2.5e-10. Rejected for the guarded Newton steps, which make
  the rates exact where a step helps and stop at the first that does not.
- **A program of its own for `minimax_allocate`.** The least worst regret is the worst share at
  one reading of n, which `cvar_allocate` already searches. Reused.
- **Keep `unsearched` for a goal's plan.** No planner returns such a plan now, so the status would
  be dead. Removed.
- **Bisection on the budget.** The gain need not be concave in the budget, but it is continuous in
  it. Brent's method brackets the same root in fewer searches. Rejected.
