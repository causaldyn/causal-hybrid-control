# ADR 0043 — One budget over geos and channels

**Status:** proposed, 2026-10-02. Experimental, as `chc.allocation` is. Amended 2026-10-07 by
ADR 0060: the bound is read from the program's duals, rounded outward, not from HiGHS's objective.

## Context

`allocate` (ADR 0034) splits one budget over channels. A plan over geos has the same channels in
every geo, each with its own curve, carryover and history, and constraints one budget does not
have: what a geo may spend, a regional cap or a floor a contract holds, and what a channel may spend
across the geos, a platform's commitment. The common way to plan it takes two steps: the budget
split over the geos first, by some reading of each geo's return, then each geo's share over its
channels. The first step decides on averages where the margins decide. Meridian, among the tools
CHC is compared with, splits a budget across channels and holds each channel's split across geos
fixed.

## Decision

- **`allocate_geos(cells, budget, periods, *, lower, upper, geo_totals=None, channel_totals=None,
  history=None)`** in `chc.allocation`, returning a `GeoAllocation`: every cell's spend a period,
  the worth on the curves as given, a bound on their envelopes, the budget's price and a price for
  each geo's and each channel's total. `Totals(least, most)` bounds what each group spends over the
  plan; a fixed spend is `least == most`.
- **One price per constraint.** At the best plan a cell inside its box returns, a currency unit,
  the budget's price plus its geo's and its channel's. A total's price is the return on one more
  unit of room in it: positive where it binds at its most, negative at its least, 0 where it does
  not bind. Where every geo's total is fixed, or every channel's, the budget is fixed with them; its
  price is then 0 and theirs carry it.
- **Cutting planes with a certificate, then Newton's method.** The rates' problem is a separable
  concave maximisation over a polytope: the budget, the boxes and the totals. Kelley's (1960)
  cutting planes bound each cell's worth from above by its tangents and its cap's value; the linear
  program over those bounds (HiGHS) bounds every plan from above while the plans it proposes
  approach from below, and the loop stops at a share `1e-9` of the bound or after 500 rounds. Its
  duals certify the bound and name the totals that bind. With every free cell strictly concave,
  Newton's method on the binding totals' prices then closes the budget and those totals to
  rounding, each cell's rate exact at its price by `allocate`'s root-finder.
- **A Newton step that does not close is replaced by the least of the dual along it.** A budget a hair
  below the most a cap allows leaves every cell outside the capped geo at its cap but one, which
  alone sets the budget's price; the free cells do not see that price, so the step does not move
  it, and the cutting planes' duals are not the plan's there. The residual is the gradient of the
  dual, convex in the prices, so along any step its projection on the step falls; a step that does
  not bring the totals closer is replaced by the least of the dual along it, by Brent's method.
  Where the free cells leave a price undetermined the step is damped, as Levenberg and Marquardt
  damp it, and the search along it moves that price until a cell leaves its cap.
- **The binding totals are settled by the exact plan, not by the duals.** The duals name a total
  that binds only to within the gap, so a cap or floor near what the best plan spends there can be
  named binding when it is not, or free when it binds; the tests' totals a millionth either side of
  the free plan's spend are named both ways. A binding total whose price comes out on the wrong side
  is released and a free total the plan breaks is bound, and Newton's method runs again, a
  primal-dual active-set step. The plan is kept when each binding total's price
  has its side's sign and the free totals hold, the conditions for the best plan on concave worths
  with the rates exact at their prices; where the set keeps changing, after twice as many rounds as
  there are totals, the cutting planes' plan is returned, with the least bound their programs'
  duals gave (ADR 0060).
- **An S-shaped curve is planned on its envelope**, as `allocate` plans it.

## Consequences

- `tests/test_geo_allocation.py` holds:
  - one geo to `allocate`'s plan, worth and price, to `1e-12`;
  - fixed geo budgets to each geo's own `allocate`, each geo's price its plan's;
  - the S-curve counterexample to `allocate`'s plan at a budget past the tangency, 1047.867, and
    at one short of it, on the chord;
  - on three geos of three channels with carryover, a cap on a geo and a floor on a channel that
    bind: SciPy's SLSQP finds nothing better, and every cell's slope meets its three prices;
  - on two geos of two channels, every plan of a 400-step lattice;
  - a plan made in two steps, the geos first, in proportion to their average return, evenly or by
    their boxes: never above the joint plan, and the first leaves more than a percent of the joint
    plan's gain behind;
  - a cap on a geo and a floor on a channel a millionth either side of the free plan's spend: the
    free plan, the total priced at 0, where it does not bind, and where it binds the plan that holds
    it fixed, its price on its side;
  - a budget `1e-9` to `1e-3` below the most a cap on a geo allows: the plan exact, and every cell's
    slope through its whole series meeting its prices to `1e-10`. Without the line search the
    cutting planes' plan came back there;
  - a linear cell, the cutting planes' plan within its gap; a cell held at zero on a square root,
    whose slope there has no tangent; a change of currency.

  Of 48 mutations of `allocate_geos` and of its goals (ADR 0044) 41 fail a test. Three of the
  seven left are here, and none changes a plan. Newton's method started from the fixed totals
  alone, not the totals the duals name: the active-set step binds the others itself, so the duals
  are a warm start. The line search's step halved: the steps after it close the rest. The guard
  against a direction that does not descend the dual dropped: a Newton step on a convex dual
  descends it in exact arithmetic, so only rounding reaches the guard.
- The rates and prices are exact only where the Newton step is. A cell on a straight stretch at
  the best plan, an envelope's chord or a linear curve, leaves the cutting planes' plan: on a linear
  cell beside a Michaelis-Menten one, rates within `2e-5` of the best. Its prices are the last
  program's duals, which certify the bound but need not be the best plan's where the program is
  degenerate. `allocate` mixes the two sides of a linear channel's jump to spend the budget exactly;
  over several totals no such mix is built.
- Each round evaluates every cell once and solves one linear program, whose rows grow by a tangent
  for each cell whose bound is loose. No timing is quoted.
- In single precision, JAX's default, a slope is good to single precision, Newton's method does not
  close to rounding in double, and the cutting planes' plan comes back: its worth and bound are good
  to single precision, its rates only as near as the gap allows. The tests run in double.
- Where every geo's spend and every channel's are fixed, the split of the price between the geos'
  totals and the channels' is the one the solver reaches; the sum each cell meets is the same.

## Alternatives

- **Two steps, the geos first.** One plan of the grid, so never better; worse where the split reads
  averages, by the tests' percent.
- **Coordinate steps on the prices**, each a bisection as `allocate`'s. Each step is exact, but the
  dual is not smooth where a cell meets an end of its box or a straight stretch, and coordinate
  descent can stall there short of the best.
- **A general solver on the rates (SLSQP, trust-region).** No certificate, and no prices that come
  with one.
- **The cutting planes alone.** Certified, but their rates sit at the kinks of the tangents, so a
  gap of `1e-9` in the worth leaves the rates off by about its square root.
