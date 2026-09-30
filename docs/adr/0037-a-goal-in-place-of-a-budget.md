# ADR 0037 — A goal in place of a budget

**Status:** proposed, 2026-10-01.

## Context

A media plan is often asked for by its goal rather than its budget: what budget returns a given
amount, how far spend can grow before the next unit returns less than it costs, what budget keeps
the return on ad spend at a target. Robyn's allocator calls the last one target efficiency, and
Meridian's optimiser takes a target ROI or marginal ROI in place of a fixed budget.

`prescribe` cannot answer them. Its targets and constraints are logged columns modelled as states,
and a known part of the plant enters only as a rate term, so a return that is a known function of
the spend has no expression there. `chc.allocation` (ADR 0034) plans discrete channels whose return
is exactly such a function, with the budget's price exact by bisection. What was missing was the
search over the budget.

## Decision

- **`budget_for(channels, goal, periods, *, lower, upper, history=None)`** in `chc.allocation`,
  returning `allocate`'s plan at the budget that meets `goal`.
- **Three goals, one sum type**, `Goal = ReturnTarget | MarginalReturnTarget |
  ReturnOnSpendTarget`:
  - `ReturnTarget(amount)`: the least budget whose plan gains `amount`;
  - `MarginalReturnTarget(per_unit)`: the budget at which one more currency unit returns
    `per_unit`, each channel where its slope meets it or at the end of its box;
  - `ReturnOnSpendTarget(per_unit)`: the most budget whose plan gains `per_unit` for each currency
    unit it spends.
- **Every goal is read on the gain**, `Allocation.gain = worth - idle`, where `idle` is what the
  channels return over the plan's periods with nothing spent in it. `worth` counts the history's
  carryover, which the plan did not buy; a return on spend read on it would be inflated by spend
  already sunk. `Allocation` also carries its `budget`.
- **One path, one bisection.** As the budget's price falls every rate rises, so the plans for all
  budgets lie on one path. A goal is a point on it: the return target where the gain reaches the
  amount, the marginal target at the price itself, and the return on spend where the gain less
  `per_unit` a unit falls through zero. That last is concave in the budget and peaks at the
  marginal target's budget, so the search starts there. Before the peak the average can rise, when
  a floor holds spend on a channel that returns little a unit; the first budget that meets the
  target is then not the most, and `validation/goal_seek.mac` STEP 3 is that case. The bisection is
  `allocate`'s, on the price; where a linear channel's rate jumps, the plans either side are mixed
  to meet the goal exactly.
- **Refused**: a goal whose value is not finite; a gain beyond what the box returns at its caps; a
  return on spend no budget reaches, with the gain less the target at its peak in the message. A
  goal the box binds is met at the end of the box.

## Consequences

- `tests/test_goal_seek.py` holds each goal to its closed form on one Michaelis–Menten and one
  exponential channel (the return on spend through Lambert's W), the floor's two crossings at
  `325 ± 25√13` to the second, a linear channel's jump mixed to the budget between its ends, and
  on three channels with carryover and a history, each goal on the gain the channels return run as
  one series, with a budget a millionth smaller or larger missing it. A change of currency moves
  the budget and leaves the gain. Nine mutations caught.
- On an S-shaped curve the path is the envelope's plans, read on the true curves. The gain still
  rises along it, but a plan off the path may meet a return target for less, and a return on spend
  is a crossing, not proved the most.
- `prescribe` still has no goal. A known output map for it, a function of states and levers that
  targets and constraints could name, is a change to its API of its own.

## Alternatives

- **A goal argument to `allocate`**, its budget a float or a goal. One name for two questions, and
  a budget argument that is sometimes not a budget.
- **A bisection on the budget with `allocate` inside.** The same plans, with a bisection on the
  price inside every step of one on the budget.
- **Newton's method on the budget**, the price as the gain's slope. It needs the price to be
  continuous in the budget, and a linear channel's jump holds it constant over an interval.
- **The return on spend read on `worth`.** Simpler, and wrong by the history's carryover.
