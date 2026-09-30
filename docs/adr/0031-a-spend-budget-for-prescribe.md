# ADR 0031 — A spend budget for `prescribe`

**Status:** proposed, 2026-09-30.

## Context

A media-mix decision is made under a budget: at most so much this quarter, or so much a week. The
question asked of it is what one more unit would buy. `chc.mmm`'s scope note said `prescribe` could
not hold one: `causal_plan` has taken `LinearConstraint` rows since 0.6.0, `RecedingHorizon` a
`PeriodBudget` since 0.7.0, and PL3 prices every row, but the facade a marketer calls had no
argument for any of them, so spend was priced through `Lever.unit_cost` alone.

A budget row is easy to get wrong by hand. `tests/test_period_budget.py` pins what a row of ones in
every window does to a loop: it caps each window, and the day overspends by 2 to 8 times.

## Decision

- **`prescribe(..., budgets=Sequence[PeriodBudget])`.** The type is `RecedingHorizon`'s, and so are
  the rows: the plan is the loop's first step, at `t = 0` with nothing spent. At most `amount` in
  each `period` steps from the plan's first, a step spending `weights @ u`, with each lever's spend
  per unit in the levers' order; one row per period the horizon touches, and a period the horizon
  cuts short gets its share of `amount`, so a budget for the whole horizon has `period=horizon`.
  The rows join the rate limit's, and every iterate of the solve holds them.
- **Refused, as `DecisionError`, before anything is fitted:**
  - weights for another number of levers;
  - a `start` other than 0: a period already under way has spent what a prescription cannot see,
    and it takes no `spent`;
  - an `amount` below what the levers' boxes spend over a period at the least. A plan is made
    offline, so the inputs can be fixed; `RecedingHorizon`, which must act, spends the floor and
    warns (`chc_event="budget_overrun"`).
- **`Prescription.budgets`** holds what the plan kept to, and **`Prescription.budget_prices`**
  says what each period's budget is worth: PL3's multipliers for its rows, one tuple per budget.
  They are the plan's last rows. A plan held under the constraints' barrier has no row prices, so
  `budget_prices` raises there, as `shadow_prices` does, and `report()` states the budget unpriced.
- **`to_json` carries the budgets and not their prices.** A budget is an input the schedule
  depends on, as the drivers are, and a record without it would describe another decision; adding
  a field leaves `schema_version` at 1. A price is an output, and outputs join the schema in its one
  change at 1.0, as ID2's gate does.
- **The regret bound stays priced against the box alone**, as with a rate limit.

## Consequences

- `tests/test_prescribe_budget.py` plans `chc.mmm`'s plant from a confounded log with 60% of the
  unbudgeted plan's spend: the budget binds, and its price matches the central difference of the
  planned cost in the amount. Budgets per four and per eight weeks bind period by period, a period
  the horizon cuts short keeps to its share, and neither plans cheaper than the whole-horizon
  budget they sum within.
- `tests/test_decision.py` prices a budget beside a rate limit's rows, where a price read off the
  wrong rows is an inactive one; keeps a budget under the constraints' barrier and reports it
  unpriced; and refuses to price one when the effect is not identified.
- `tests/test_plan_oracles.py` holds a budget's price to a closed form: goodwill decaying at `d`
  with linear revenue `p G`, Sethi's (1977) setting. A unit of spend at `s` is worth
  `(p / d)(1 - exp(-d (T - s)))` of revenue, falling in `s`, so the plan spends first; the price is
  what the step it leaves inside the box is worth, and a budget of whole steps is degenerate
  between the steps on either side (`validation/planner_oracles.mac`, STEP 5).
- Without a budget nothing moves: the solve gets the same rows, and Track M's matched-budget
  comparison is unchanged.

## Alternatives

- **A `total_budget: float`.** Simplest for the quarter, but a second type for what `PeriodBudget`
  says already, and a weekly budget would need a third. Rejected.
- **`LinearConstraint` rows passed through.** The caller would flatten `(horizon, levers)` by hand,
  the mistake the context describes. Rejected.
- **A `spent` argument for a period under way.** No consumer asks for it, and a loop that tracks
  spend has `RecedingHorizon`. Not built.

## Not built

- **Goal seek and a target ROAS** (plans/27 §2.1). A target on a known function of states and
  levers, such as revenue over spend in a window, has no expression in `prescribe`: every target
  and constraint is a logged column modelled as a state. It needs a known output map, which is a
  change to `prescribe`'s API of its own.
