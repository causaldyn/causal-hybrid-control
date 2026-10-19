# ADR 0110 — A schedule over periods, and a proved last crossing

**Status:** accepted, 2026-10-19. Extends `allocate` (ADR 0051's search in boxes) and amends how
`budget_for` meets a return on spend (ADR 0037).

## Context

`allocate` plans one rate a period for each channel. Where every curve is concave and the periods
are alike, that is the best schedule too. On an S-shaped curve it is not: one Hill curve of slope
3 and scale 1, with no carryover, returns 0.105 on 1.2 spread over four periods and 0.633 on 1.2
spent in one, six times more. The docs said so, and no schedule was searched.

`budget_for` meets a target return on spend at the last budget where the gain less the target a
unit falls through nothing past its peak. On S-shaped curves that surplus can cross nothing
several times. A grid of 16 budgets from the peak to the caps was read down from the caps, and
Brent's method closed the last crossing the grid saw. A window where the goal is met that falls
inside one step was not seen: on a concave channel and a Hill curve of slope 12, the surplus rises
over nothing between about 36.8 and 37.7, inside the step from 36.64 to 41.46, and the goal was met
at 22.83.

## Decision

- **`allocate(schedule=True)`** searches a spend for each period and channel, each in the
  channel's box, the periods together spending the budget. `Allocation.schedule` holds it;
  `spend` is each channel's mean over the periods. The default, `False`, plans one rate a period
  as before.
  - **Exact where every kernel is one period long.** Each period's spend then returns in its own
    period alone, so the worth is a sum of one-variable curves, one a period and channel, and the
    search in boxes that plans the rates plans the cells. The periods of one channel are alike,
    so any order of their spends is as good as another, and a search that bounded each order apart
    took 499 boxes on three channels over six periods without closing. The search reads
    each channel's periods in falling order: a cut of one period's spend at a rate holds the later
    periods at most that rate in the lower half, and the earlier ones at least it in the upper
    half. The same case closes in 13–17 boxes.
  - **A stated local method where a kernel is longer.** The periods' adstocks then share their
    spend, the worth is not a sum of one-variable curves, and no search in boxes applies. The
    schedule is the best of SLSQP ascents on the curves, each polished by Newton's method on the
    face of the box it stops on, from the best rate a period, from the best schedule on the curves'
    concave envelopes and from eight seeded random schedules that spend most of the budget in a few
    cells; never less than the best rate a period. On two Hill curves with carryover over six
    periods it returned 1.150, as the best of 200 random ascents did, against 0.549 flat. Its
    bound is the envelopes' schedule's worth plus its Frank–Wolfe gap: the envelopes are concave
    and above the curves, so no schedule returns more. Where the bound is not within the tolerance
    of the worth, `stopped` is `cap`, `limit` is the new `SearchLimit` `local`, and the call logs
    `chc_event="allocation_local"`. On concave curves the envelopes are the curves, one ascent is
    the best, and on the three concave channels of the allocation tests the gap closes.
- **A proved last crossing.** `budget_for` halves the stretch from the peak to the caps. Its upper
  half is searched in boxes, as the marginal target is, for the plan that gains the most less the
  target a unit with its budget held in the half: a box's split is the rates at the target where
  they spend inside the half, and otherwise the split of the end they pass. Where that search's
  bound falls short of the target, no budget in the half meets it, and the stretch is the lower
  half. Otherwise the stretch starts at that plan's budget, past halfway. Once the stretch is
  narrower than the search's tolerance in the budget, Brent's method closes its crossing to
  rounding. `budget_for_geos`
  holds the budget in the half as one more total of its search, so the proof holds under totals
  too, where the gain need not rise with the budget.

## Consequences

- No default moves. `Allocation` gains a field with a default; `SearchLimit` gains a value.
- On the narrow window above, `budget_for` now returns 37.68 where it returned 22.83. Where the
  grid saw the last crossing the budget is the same to the search's tolerance.
- A proof searches each half once: on the narrow window, 34 searches and 3 plans, where the grid
  read 19 plans. Not timed.
- The local schedule is not proved the best. Its bound on S-shaped curves is loose by the
  envelopes' excess, and the warning says so.

## Alternatives

- **A search in boxes over the adstocks where a kernel is longer.** The worth is a sum of
  one-variable curves of the adstocks, which are linear in the spend, so a search in boxes of the
  adstocks with a concave program in each box would be exact. Rejected for now: each box needs a
  solver for a concave program under linear rows, which the library does not have, and its bound
  by a Frank–Wolfe gap under those rows needs a linear program a box.
- **Enumerate the number of periods each channel spends in.** Where kernels are one period long,
  a best schedule spends a common rate in some periods of each channel and its floor in the
  rest, save one period. Rejected: the counts multiply over channels, and the falling order
  gives the same search without them.
- **A finer grid for the last crossing.** Rejected: any grid misses a window narrower than its
  step.
- **Clear budgets down from the caps by monotonicity.** The best plan's gain rises with the
  budget, so a plan's bound bounds the gain at every smaller budget. Tried first and rejected: near
  a window that barely meets the target each plan clears only a sliver, and the window above took
  over 50 plans without closing; under totals the gain need not rise at all.
