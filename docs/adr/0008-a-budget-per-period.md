# ADR 0008 — A budget per period

**Status:** accepted, 2026-09-28.

## Context

Budgets are stated per period: a day's energy, a week's or a month's media spend. A budget row in
`constraints` (ADR 0001) caps a plan's total over its horizon, and on a receding horizon (ADR 0003)
each step is a new plan. So the row caps every window, and a loop that re-plans each step spends
the budget again at every step. `RecedingHorizon`'s docstring said so. Nothing in the library could
state the period's budget.

On the ledger lab's plant, `x' = -0.5 x + u` tracked to 2, a 48-step day was planned in 8-step
windows. With a budget row in `constraints`, the day spent 2.007 times its budget when the budget
was half of what an unbudgeted day spends, and 8.287 times when it was a tenth. Octave's `qp` gives
8.2868 as well.

What the solution had to keep:

1. **No period spends more than its budget**, measured on the applied actions, not the plans. The
   one exception is a box that forces a spend, and then the loop says by how much.
2. **The loop's other rows still bind**: a rate limit, or a caller's own row, beside the budget.
3. **Compiled once.** ADR 0003's loop stops compiling after its first steps, and a budget must not
   add a program per step.
4. **An unbudgeted loop computes what it did.**

## Decision

- **`chc.mpc.PeriodBudget(weights, amount, period, start=0.0)`.** Each period of `period` steps
  spends at most `amount` of `weights @ u`, summed over its steps. Periods start at `start` on the
  loop's clock. It is frozen and validated at construction. Unspent budget does not carry over.
- **`RecedingHorizon(budget=...)` and `step(x, t, spent)`.** The ledger is the caller's: `spent` is
  what the current period has spent so far, as measured. An applied action need not be the
  planned one: an actuator clips, an operator overrides. So the spend is a measurement, like the
  state. A step is refused in four cases:
  - a budgeted step without `spent`;
  - a `spent` without a budget;
  - a `t` off the `dt` grid from `start`;
  - a budget over a different number of levers than the plan has.
- **One row per period the window touches**, over that period's steps in the window:
  - The current period's row is what is left, `amount - spent`, pro rata to the share of its
    remaining steps the window holds.
  - A later period's row is `amount`, pro rata the same way.
  - A window of at least one period therefore sees the rest of every period it plans in, and the
    current period's row is exactly what is left.
- **The box before the budget.** Where the box forces a spend above a row, the row is the least the
  box allows. A warning names the overrun (`chc_event="budget_overrun"`, with the step's place in
  the period, what was left, the least spend and what the row allowed). A period that is already
  overspent plans nothing more in it. A row held to rounding is not an overrun, and says nothing.
- **The rows are traced data.** Only their count changes the compiled shape, and the count is the
  number of periods the window touches. That number takes two values over a loop, so a loop builds
  two programs; with a window of one period, both by its second step.
- Each step logs its rows' bounds at DEBUG (`chc_event="budget"`).

## Evidence

The script `scripts/bench_period_budget.py {lab,mmm}` reports costs and spends in float64 and
takes no wall time. Its setup:

- Every loop acts on the true plant and is audited on its applied actions.
- The reference is the plan made for the whole run at once, with one budget row per period.
- Regret is reported in the cost's units and as a share of the budget's value. The value is what
  the whole-run plan gains over spending the box's floor throughout.
- A loop that overspends is not scored against a plan that did not.
- The control is the same loop without a budget, against the unbudgeted whole-run plan.

**Every `PeriodBudget` loop spends every period's budget, to `3e-15` of it either way.** That
holds in both cases, at both budgets, for every window, over three periods and over six. The tests
hold it to `1e-6` below and `1e-9` above.

### The lab plant

The plant runs three 48-step days, with a daily budget of a half and of a tenth of what an
unbudgeted day spends. Over the three days the budget is worth 212.9 and 55.1.

| loop | spend / budget, days 1-3 | regret, share of the budget's value |
|---|---|---|
| a budget row in `constraints`, 8-step window | 2.007, 1.940, 1.940 / 8.287, 8.086, 8.086 | overspent |
| `PeriodBudget`, 8-step window | 1 / 1 | 4.61 % / 8.20 % |
| `PeriodBudget`, a day | 1 / 1 | 0.40 % / 0.83 % |
| `PeriodBudget`, two days | 1 / 1 | 0.67 % / 1.64 % |
| a window that shrinks to the day's end (rejected) | 1 / 1 | 0.54 % / 0.57 % |
| no budget, any of the three windows (control) | | 0.001 % |

### The marketing-mix plant

The plant runs twelve weeks, three four-week periods, with a budget per period of a half and of a
quarter of what the unbudgeted twelve-week plan spends a period. The loop plans on the true plant
and on the adjusted arm's fit. The budget is worth 151.9 and 103.5.

| window | regret on the true plant | regret planned on the fit |
|---|---|---|
| one period | 0.23 % / 0.47 % | 5.66 % / 8.67 % |
| two periods | 0.40 % / 0.87 % | 7.87 % / 12.46 % |
| three periods | 0.62 % / 1.78 % | 8.08 % / 13.84 % |
| no budget (control) | 0.014 % | 0.58-0.63 % |

The budget row in `constraints`, in a window of one period, spends 2.005, 1.378 and 1.374 times the
budget, and at a quarter 2.541, 1.915 and 1.889 times.

### What the numbers say

**The row in `constraints` overspends by a rule, not by noise.** A test pins 2.006797 and 8.286766
as found, so the ledger's `<= B` is a change and not a coincidence.

**A regret that stays put when the run doubles is paid once, at the run's end.** Every loop above
was run again over six periods at the same budget. Regret in the cost's units:

| loop | lab, 3 days | lab, 6 days | marketing mix, 3 periods | marketing mix, 6 periods |
|---|---|---|---|---|
| shrinking to the period's end | 1.14 / 0.32 | 2.98 / 0.73 | | |
| `PeriodBudget`, a sixth of a period | 9.82 / 4.52 | 21.28 / 8.73 | | |
| `PeriodBudget`, one period | 0.84 / 0.46 | 1.01 / 0.50 | 0.35 / 0.49 | 1.41 / 2.70 |
| `PeriodBudget`, two periods | 1.42 / 0.90 | 1.42 / 0.89 | 0.60 / 0.90 | 0.77 / 1.45 |
| `PeriodBudget`, three periods | | | 0.95 / 1.84 | 1.13 / 2.15 |

- A window that plans past the run's last period spends that period as if the run went on. That is
  most of what the longer windows lose over three periods, and it does not grow with the run.
- The shrinking window loses at every period's end. It never sees what the day's end hands to the
  next day, so over six days it loses 0.69 % and 0.66 %, against 0.23 % and 0.45 % for a window of
  a day.
- Without a budget the same end costs 0.003 in the lab's units and 0.025 in the marketing mix's. A
  scarce budget makes where the last spend goes matter.

**How far past the period's end to look is the plant's to say.** The lab plant forgets a spend in a
few steps, so a window of one day loses little at the day's end. The marketing-mix plant's slowest
adstock keeps 37 % of a spend four weeks on, so a window of one period loses at every period's end:
0.35 in three periods and 1.41 in six. Over six periods the window of two periods is best, at 0.24 %
and 0.67 %, against 0.45 % and 1.25 % for one period and 0.36 % and 0.99 % for three.

**Planned on the fit, the loss is the model's, and a budget multiplies it.** Over three periods the
adjusted arm's fit loses 0.58-0.63 % with no budget and 5.7-13.8 % with one: once spend is scarce,
which channel gets it is the fit's call, and the fit's errors cost ten times as much. A longer
window plans further on the wrong model, and here it loses more, over six periods as well as three.

**The first step of a period is the period's own plan.** With a window of one period, the step at
a period's start equals `causal_plan` over that period with one row of the whole budget, to
`1e-12`. That holds for `start` of 0, 1 and -2.5.

**Mutation checks.** Each of these fails at least one test in `tests/test_period_budget.py`:

- dropping the pro-rata factor;
- giving the current period `amount` instead of what is left;
- giving later periods their whole `amount`;
- counting the current period's share from its start rather than from the step;
- reading the step's place in the period one step late;
- dropping the clamp to the box's least spend;
- ignoring `start`;
- warning at a row held to rounding.

## Consequences

- `RecedingHorizon` gains a field and `step` an argument, both defaulting to the unbudgeted loop.
  The parity test holds the other fields to `causal_plan`'s arguments.
- **The ledger is the caller's.** A wrong `spent` is a wrong budget. The controller still keeps no
  record of what was applied; its only state is the warm start.
- **No carry-over.** Unspent budget is lost at the period's end. A budget that carries over is a
  different statement, one row over the run so far, and is not built.
- **A window shorter than a period gets a flat share of what is left**, and loses to one that sees
  the period out: 4.6-8.2 % against 0.4-0.8 % on the lab plant.
- A receding horizon does not know where the run ends, and a budget makes that cost more. The
  controller is not told the end, because a loop that runs on has none.
- `mpc_control` and `prescribe` take no budget.

## Alternatives

- **A budget row in `constraints`.** This is what a caller could already write. Rejected: it caps
  each window, and the loop spends 2.0-8.3 times the day's budget on the lab plant.
- **A window that shrinks to the period's end**, planning the rest of the period with what is left.
  Rejected for three reasons:
  - it plans `period - position` steps, so one day compiles a program for each of its 48 lengths;
  - a caller's rows are built for the window's horizon and do not fit a shrinking one;
  - it never sees past the period's end, and loses there every period.
- **The ledger lab's reserve with a costate terminal.** This window reserves what the
  rest-of-period plan spends past it, and prices its last state with that plan's costate. In the
  lab it matched the plan for the whole day to a regret of `3.2e-8`. Rejected: on a
  linear-quadratic plant, the KKT conditions and the principle of optimality make its first action
  the rest-of-period plan's first action. So it solves the shrinking window's problem and then a
  second one, to recover the answer it already had.
- **A ledger inside the controller**, summing the first actions it planned. Rejected: the applied
  action is the one that spent, and only the caller measures it.
- **Dual pacing**, pricing the budget with a multiplier updated online (Balseiro, Lu & Mirrokni
  2023). Not built: it is the allocation algorithm for requests that arrive unforecast. A plan
  with a model already has the multiplier, from the row.

## References

- Balseiro, S., Lu, H. & Mirrokni, V. (2023). The best of many worlds: dual mirror descent for
  online allocation problems. *Operations Research*. arXiv:2011.10124.
- Rawlings, J. B., Mayne, D. Q. & Diehl, M. M. (2017). *Model Predictive Control: Theory,
  Computation, and Design*, 2nd ed. Nob Hill Publishing.
