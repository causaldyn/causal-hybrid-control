# ADR 0052 — The worst share's split is searched by branch and bound

**Status:** accepted, 2026-10-05. Amends ADR 0048 (a gain is read on the curves, and on S-shaped
curves the split is searched) and ADR 0051 (`cvar_allocate` searches boxes as `allocate` does).

## Context

`chc.allocation.cvar_allocate` cut its planes on each S-shaped curve's envelope over the whole box
(ADR 0048). Its gains are read on the curves in 0.13.0, so its split never loses to the
reference in the worst share, but the planes' bound was the envelopes', and the split was the best
of the splits the envelopes proposed:

- two Hill curves of slope 3 at scales 1 and 1.01, a budget of 1.6 and the reference all on the
  first: on the envelopes the planes propose 1.27/0.33, which returns 0.705 against the
  reference's 0.804, and the bound stayed open above the reference's 0, where the reference is the
  best split;
- the open bound said how far the split might be from the best, and on S-curves it said nothing
  closer than the envelopes' excess, which `allocate` stopped accepting in ADR 0051.

Reading every reading's channels one at a time cost one dispatch a reading and channel for every
split tried, both on the envelopes and on the curves: a posterior of a thousand draws over five
channels read ten thousand programs a split.

## Decision

- **`cvar_allocate` searches boxes of the rates**, as `allocate` does (Udell and Boyd 2016). In a
  box the planes are cut on each curve's envelope over the box, so the linear program bounds every
  split in the box, and each split it proposes, read on the curves, is a candidate. The box of the
  largest bound is cut first. Boxes within the share `1e-9` of the reference's largest return of
  the best split are settled; the search stops when every box is, or after 500 boxes, and the
  bound returned is the largest left or settled.
- **The channel cut is the one whose envelope stands furthest above its curve, weighed by the
  worst share.** The share's mean gain is a minimum over the risk envelope, `min_q q · gain` with
  `0 <= q_r <= 1 / (level n)` and `sum q = 1`, so at a split its mean on the envelopes exceeds its
  mean on the curves by no more than `q* · (envelope - curve)`, with `q*` the weights the worst
  share puts on the curves' gains there. The box is cut on the channel of the largest weighed
  excess, at its rate at the box's best split on the envelopes, or halfway along its interval
  where that rate is an end. The interval is first narrowed to the rates a split of the budget in
  the box can give the channel, so each half holds such a split. Where no envelope stands above
  its curve there, the channel cut is the one whose interval is widest against its range.
- **A box's planes stop early, and carry over.** In a box the planes stop after 30 rounds, or
  once their bound is within half the box's gap to the best split of its best on the envelopes:
  the rest of the gap is the envelopes' excess, which only a cut closes. A box's planes bound its
  halves too, since the envelope over a half stands below the envelope over the box, so the halves
  start from them; those the last program left slack are dropped, which loosens no bound, since a
  plane slack at a linear program's optimum leaves the optimum where it is.
- **Each channel's readings are read together.** The readings of a channel are grouped by the
  structure of their worths, and each group is stacked on a leading axis and read, or bounded over
  a box, by one compiled program; a split's reading costs a dispatch a group, not a reading. A
  group whose curves are all concave from zero is its own envelope and is never bounded.
- **Concave curves are planned as before.** Every envelope is its curve, one box is searched, and
  its planes run to 500 rounds as before, so the split is the same to the planes' tolerance,
  though not to the bit (below).
- **`CvarAllocation.stopped` and `.boxes`** say whether the bound closed and how many boxes were
  searched, as `Allocation`'s do; a search stopped by its cap logs one warning with the gap
  (`chc_event="allocation_cap"`, `planner="cvar_allocate"`).

## Consequences

- Splits on S-curves change, and only there. Counted in boxes, not timed, since the machine was
  loaded: the two Hill curves close on the reference in 7 boxes, where the bound stood 0.041 above
  it; eight readings of three Hill channels with carryover close in 55 boxes at the level 0.2, on
  the reference, where the bound stood 429 above it, and in 33 at 1, on a split that gains 860.70
  where the envelopes' best gained 855.21 under a bound 396 above it; forty pairs of Hill channels
  drawn at random, one to eight readings each, close in 1 to 47 boxes, 584 in all.
- A posterior of 400 draws over four channels with carryover, half of them Hill curves and half
  tanh, as a Bayesian fit's stacked draws come: at the level 0.1 the search closes on the
  reference in 125 boxes, at 0.5 on a split that gains 106.8 in 231, and at 1 on one that gains
  583.7 in 145, each in a peak resident memory under 490 MiB, the linear programs never past 4572
  rows. Over six such channels at 0.1 the search stopped at its cap, after 501 boxes, a cut adding
  two, with the gap at 0.037 on 353.36, and said so. At 1 the six close in 413 boxes, on a split
  that gains 2678.5, none of the 940 linear programs past 0.6 iterations a row and column; until
  the program's excess costs were written at 1, HiGHS cycled on 87 of them, and the search
  stopped at its cap with no finite bound (the changelog's entry for 0.13.0).
- On concave curves one box is searched, as before. The stacked reading sums each worth's periods
  in another order than a reading alone, by up to 5.5e-12 on returns near 4000, and the channels'
  returns in another order, so a split on a flat optimum moves within the tolerance: on thirty
  readings a tenth apart by 4.5e-7 on rates near 100, its mean gain in the worst share the same to
  2e-11 of itself.
- A box costs up to 30 rounds; each reads every reading on its envelopes and its curves, two
  compiled programs a group and channel, and solves one linear program over the planes kept, one
  row a reading for every split tried. A heap's box holds its cut channel's envelopes for every
  reading, so the search's memory grows with the boxes left and the readings.
- In float32 the gap closes only to float32's rounding, so a search on S-curves runs to its cap:
  the eight readings with carryover at 1 stopped after 501 boxes with a gap of 9.6e-5 on 860.7, and
  said so. The tests plan in float64.
- `tests/test_cvar_allocation.py` holds the search to grids: on the eight readings with carryover
  no split of a 241-by-241 grid of the budget has a larger mean gain in the worst share, at 0.2 or
  at 1; on pairs of Hill readings drawn by a property, none of 2001 splits; on readings whose
  channels mix families, Hill's of slope 1 concave from zero among them, none either, each
  reading's gain its own curves'. A search stopped by its cap, of boxes or of rounds, says so and
  logs its gap; a curve whose tangency the bisection cannot find is refused by name, as `allocate`
  refuses it; the branching weights are the risk envelope's. Of 24 mutants of the search, 20 fail
  a test. Of the four that pass, two only slow it: weighing every reading alike when the channel
  is chosen (614 boxes on the forty pairs, against 584) and keeping every plane for the halves
  (600). The other two move nothing past the tolerance: a bound that forgets the boxes settled
  within it, and a widest interval read unscaled where no envelope stands above its curve, which
  picks another channel only where the ranges differ.

## Alternatives

- **Planes on the envelopes over the whole box, the split the best of those tried** (ADR 0048 as
  amended in 0.13.0): it never loses to the reference, but its bound stays open by as much as the
  envelopes stand above the curves, and a split better than every split the envelopes propose is
  never tried. Rejected.
- **Cutting the channel of the largest unweighed excess**, summed over the readings: it cuts where
  readings the worst share ignores stand furthest above their curves, which closes nothing the
  share reads. Rejected.
- **A box's planes run to the full 500 rounds before it is cut**: on S-curves most of a box's gap
  is its envelopes' excess, which no plane closes, and a cut does. Rejected.
- **Stacking across channels too**: channels of one kernel and family could share a program, but
  channels differ in both as a rule, and the readings are the axis that grows. Left until a case
  needs it.
- **The level 1 written as the mean's program**, a free gain a reading under its planes, with no
  `eta` and no excesses: at 1 the worst share's program has an unbounded optimal face, every
  excess's reduced cost zero, and the mean's has neither. That degeneracy cycled only in HiGHS's
  second, unscaled solve, which the costs at 1 no longer cause: on the 87 programs that cycled,
  their scaled solutions stood within 2.3e-11 of feasible once unscaled, so the second solve
  never ran. Left until a program at 1 stalls with its costs at 1; the iteration limit bounds it.
