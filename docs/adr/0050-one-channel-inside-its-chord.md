# ADR 0050 — One channel inside its chord: S-curve plans take the vertex

**Status:** accepted, 2026-10-04.

## Context

`chc.allocation.allocate` plans S-shaped curves on their concave envelopes (ADR 0029) by bisecting
the budget's price. Where an envelope is linear, its chord from the origin to the tangency, the
channel's rate jumps from 0 to the tangency at one price, and the plan spends the last of the budget
on a mix of the rates either side of that price.

The mix was proportional: one share for every rate that moved within the last bracket. On the
envelopes that is as good as any other split, since every channel that jumps there does so on a
chord of the same slope. On the curves it is the worst. Each channel left between 0 and its tangency
is below its envelope by up to its nonconvexity, and equal curves jump together, so all of them are
left there:
- two Hill curves of slope 3 at a budget of 1.6 scales were split 0.8/0.8 for 0.677;
- one filled first gives 0.705, and the optimum is 0.804.

A budget is one linear constraint. Shapley and Folkman's lemma for one constraint (Aubin and Ekeland
1976; Udell and Boyd 2016) says the relaxed problem has an optimum with at most one channel strictly
inside its chord. Its shortfall on the curves is then at most that channel's coefficient times its
curve's largest excess under the envelope.

## Decision

- **The plan is the vertex.** `allocate`, and `budget_for` through its crossing, fill the rates that
  move in the last bracket one at a time, in the channels' order (`_fill`). At most one channel is
  left between the ends of its jump.
- **Every curve states its nonconvexity.** `Saturation.nonconvexity()` is
  `max (envelope − curve)` on the curve's scale of a ceiling of 1, found at the root of `g' = m`
  below the inflection, with `m` the chord's slope. It is free of scale, so a change of currency
  leaves it. `validation/envelope_nonconvexity.mac` derives Hill's in reduced form and in closed
  form at slopes 2, 3 and 5, and the anchors for every S-shaped family. The `.gp` file checks them
  to 80 digits.
- **The bound before planning is documented where it holds:** kernels of length one, floors at
  zero, caps past the tangencies. There, `bound − worth ≤ max periods · coefficient · nonconvexity`,
  and a property test holds it over random Hill and Weibull channels with ties.

## Consequences

- Plans on tied S-curves change. Two equal Hill(3) channels at 1.6 go from 0.8/0.8 (0.677) to
  1.26/0.34 (0.705), and three at 2.4 from 1.016 to 1.264. Plans with no ties change by at most the
  last bracket's width, a few ulps of the price.
- `allocate_geos` and `minimax_allocate` return a linear program's vertex already and were not
  changed: on the same ties they plan 1.26/0.34. Four equal cells in `allocate_geos` leave a gap of
  0.1544 against the bound 0.1547, so the bound is tight.
- `cvar_allocate` keeps its reference split wherever no split beats it on the envelopes, which is
  its promise. A reference inside the chords therefore stays there: two equal Hill(3) channels
  referenced at 0.8/0.8 keep 0.8/0.8, though the vertex returns more on the curves. That gap is the
  envelope's, like the one below.
- The plan on envelopes is still not the optimum: at scales 1 and 1.01 the two Hill(3) channels plan
  0.705 where 0.804 exists. `bound − worth` reports it, and closing it needs branch-and-bound on the
  chord channel (Udell and Boyd's sigmoidal programming), which is not built.
- The order of filling is the channels' order. Among tied channels, which one is left partly filled
  is therefore a convention, not a choice by worth.

## Alternatives

- **Keep the proportional mix and report the gap.** The gap was honest, but up to the number of tied
  channels times the nonconvexity, where the vertex costs one. Rejected.
- **Choose the partial channel by its worth on the curve.** It is better on some ties and costs one
  evaluation per tied channel. Branch-and-bound subsumes it, so it is left to that.
- **A beam over channel on/off patterns.** It has no bound, and the lemma gives one. Rejected.
