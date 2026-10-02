# ADR 0044 — A goal over geos and channels

**Status:** proposed, 2026-10-02. Experimental, as `chc.allocation` is.

## Context

`budget_for` (ADR 0037) meets a goal over one geo's channels by `allocate`'s bisection on the price:
as the price falls every rate rises, so every budget's plan lies on one path and a goal is a point
on it. `allocate_geos` (ADR 0043) has no such path. Under a cap on a geo and one on a channel a cell
can spend less as the budget grows, and the best plan's gain can fall: past some budget the only
room left is in cells that return less than the spend the caps take from a cell in both.

What stays is enough. The best plan's gain on concave curves is concave in the budget, the value of
a concave program in its right-hand side, and its slope is the budget's price, which `allocate_geos`
returns with every plan.

## Decision

- **`budget_for_geos(cells, goal, periods, *, lower, upper, geo_totals=None, channel_totals=None,
  history=None)`** in `chc.allocation`, returning `allocate_geos`'s plan at the budget that meets
  `goal`, one of `budget_for`'s three, read on the same gain, `worth - idle`.
- **The budgets run between the least and the most** a plan within the boxes and the totals spends,
  two linear programs; boxes and totals no plan meets together are refused there.
- **A return target by Newton's method from the least budget.** A concave gain lies under its
  tangents, so each step stays short of the target and the steps go one way; a step that reaches
  it, which only rounding makes, is the last. The target is refused above the most any plan gains,
  the plan with nothing charged for its spend, not above the most budget's: where the gain falls
  that is less.
- **A marginal target is one plan**, its budget left free and each currency unit it spends charged
  `per_unit` of return: the cutting planes and Newton's method run as for a budget, the budget's row
  replaced by the charge, and the budget is what the plan spends. A negative target is a budget
  past the gain's peak where totals make it fall, and the most budget where they do not. A search
  over the budget for the price costs a plan a step, and at an end of the budgets, where the price
  is any number on one side, the price jumps and the search closes on the end only by halving.
- **A target return on spend by Newton's method from the most budget**, on the gain less `per_unit`
  a unit, concave, whose slope is the price less `per_unit`.
- **On an S-shaped curve** the plans are the envelopes', as `allocate_geos` makes them, and the gain
  is read on the true curves, which is not concave in the budget. A return target is met by Brent's
  method between the least budget and the peak's, and a return on spend between the marginal
  target's budget and the most; neither is proved the least or the most.

## Consequences

- `tests/test_geo_goal_seek.py` holds:
  - one geo to `budget_for`'s budget, spend and gain for each goal, to `1e-12`;
  - on three geos of three channels with carryover, a cap on a geo and a floor on a channel: a
    return target at the least budget, the gain the cells return run as one series, a billionth less
    missing it; a marginal target's plan to `allocate_geos`'s at its budget, the price to `1e-12`; a
    target return on spend at the most budget, a billionth more missing it;
  - goals the totals bind met at the ends of the budgets, worked by hand, among them a marginal
    target just below the price the plans approach at the most budget;
  - at most ten plans for a return target or a return on spend, and none at a budget for a marginal
    target;
  - every geo's total fixed; a change of currency; the refusals;
  - S-shaped curves to `budget_for`'s budget and spend for each goal, and a return on spend the
    most budget meets met there, where Brent's method would find nothing crossed;
  - where two caps make the gain fall past a budget of 10: a return target met short of the peak
    on a concave and on an S-shaped curve, one above refused at the peak, and a negative marginal
    return and a return on spend past it, each to its closed form;
  - a marginal target with a linear cell inside its box, the cutting planes' plan, whose bound
    adds the charge back and is at least the best plan's worth.
- Of the 48 mutations of ADR 0043, four of the seven that fail no test are here, and none is shown
  to change a plan. The best of the cutting planes' plans chosen by its worth, not its worth less
  the charge: they stop when the plan they keep is within the gap, whichever they keep, so only a
  run cut off at 500 rounds could tell. Their gap measured against the worth less the charge: that
  changes when they stop; the polish then makes the plan exact, and where it cannot, the bound
  still holds. A release threshold set without the charge: a difference of a share `1e-9` of it.
  Newton's method on S-shaped curves: beside a concave cell, an S-shaped one met nine targets at
  Brent's budgets, since the most budget a return on spend allows lies past the gain's best ratio,
  where that cell is off its chord; nothing proves it for several S-shaped cells, and a run on two
  was stopped before it finished.
- At an end of the budgets the price is not one number. A marginal target's plan reports the
  target; a budget's plan, whichever price the solver reaches.
- No timing is quoted.

## Alternatives

- **Brent's method on the budget for every goal.** A plan a step for each, where Newton's method
  needs a few and the charged plan one; at an end of the budgets, where the price jumps, a step of
  halving each.
- **A return target refused above the most budget's gain**, as `budget_for` refuses it. Under totals
  that refuses targets a smaller budget meets.
- **A marginal target clamped at nothing**, as `budget_for` clamps it. Without totals the clamp
  changes no plan; under them it hides the budgets past the peak, where the price is negative.
