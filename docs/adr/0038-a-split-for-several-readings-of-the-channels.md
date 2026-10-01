# ADR 0038 — A split for several readings of the channels

**Status:** proposed, 2026-10-01.

## Context

`chc.lift` reads a channel from geo tests, and a test that never bent the curve reads its slope
and little of its shape: on causaldyn-bench's Track M v2 the scale's interval stayed open above in
554 of 600 channels. Several saturation families then fit the tests alike and part where a plan
goes, past the spend the tests covered. Choosing one of them, by AIC or by habit, and planning on
it extrapolates that family's shape as if the tests had read it. ADR 0034 left the alternative
unbuilt: a split that holds up under every reading the tests allow.

## Decision

- **`minimax_allocate(readings, budget, periods, *, lower, upper, history=None)`** in
  `chc.allocation`, returning a `MinimaxAllocation`: the split, each reading's best return at the
  budget, the split's regret under each, the worst of them, and a bound below it.
- **Regret, not return.** A reading's regret is its best return at the budget over the split's.
  The split minimises the worst regret over the readings (Savage 1951), not the worst return,
  which the reading with the smallest returns would decide whatever the split.
- **Exact by cutting planes, with a certificate.** Where every curve is concave a reading's
  regret is convex in the split, so the worst regret is too. Kelley's (1960) cutting planes take
  every reading's tangent at every split tried, starting from each reading's own best split; the
  linear program over them (HiGHS) bounds the least worst regret from below, and its solution,
  moved onto the budget, is the next split tried. The loop stops when the best split's worst
  regret is within `1e-9` of the largest best return of the bound, or after 500 rounds, and the
  gap is returned either way.
- **An S-shaped curve is hedged on its envelope**, as `allocate` plans it: a reading's best is
  `allocate`'s bound, and its regret is read on the envelope.
- **The readings are the caller's.** Which families the tests cannot tell apart, and by what rule,
  is a question about the fit, not about the split; causaldyn-bench's Track M v2 measures one rule
  before the library takes any.

## Consequences

- `tests/test_minimax_allocation.py` holds the split to a closed form, two linear readings
  mirrored, hedged by the even split at a worst regret of 50; on three channels with carryover
  under four readings, to every split of a 241-by-241 grid, none with a smaller worst regret,
  while trusting any one reading's best split leaves a worst regret above the hedge's by a tenth;
  one reading, planned as `allocate` plans it; and an S-curve's regret on its envelope's chord.
  Five mutations caught.
- The planes are read in units of the largest best return. In currency, a reading whose fit has
  run to the edge of its family, its coefficient and scale grown together, carries a slope past
  `1e15`, the largest entry HiGHS accepts, and on causaldyn-bench's Track M v2 it refused the
  program. The closed form holds in a unit `1e18` times smaller.
- A reading needs its own `allocate` for its best return, and the planes one tangent a reading a
  round, so the cost grows with the number of readings, which a product of families kept per
  channel makes large.
- It hedges between the readings it is given and against no other. A family none of them is can
  still be the truth.

## Alternatives

- **Averaging the readings.** A split for their mean return: no bound on what it leaves under any
  one of them.
- **The best worst return.** Max-min on the return rather than the regret: the reading with the
  least return decides, however little the split changes it.
- **A general solver on the worst regret's epigraph (SLSQP).** No certificate, and its stopping
  rule sits where the worst regret has kinks, which is where the optimum is.
- **The dual over mixtures of the readings**, `allocate` on each mixture. Its value bounds from
  below too, but at the optimum the split is not unique, and recovering the minimax one takes the
  primal anyway.
