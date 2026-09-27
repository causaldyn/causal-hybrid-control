# ADR 0004 — Which levers: greedy selection under `max_levers`

**Status:** accepted, 2026-09-27.

## Context

`prescribe` plans with every lever it is handed. Which levers to use at all is a question of its
own: a pilot allowed to change two things at once, or an operator who has to answer for every lever
moved, needs a schedule that moves at most `k` of the `m` levers, and a statement of what the others
would have bought. The proposed design was cardinality-constrained planning: a group-L1 penalty on
the action matrix, one group per lever as in the group lasso (Yuan & Lin 2006), inside the
projected-gradient solver, then greedy forward selection with the solve as the inner loop, each
removal priced by the certificate.

Choosing the best `k` of `m` is subset selection, NP-hard already for linear least squares
(Natarajan 1995), and greedy's known guarantees for it rest on a submodularity ratio (Das & Kempe
2011) that nothing here establishes for a constrained optimal-control objective. Every method short
of trying all subsets is a heuristic, and a heuristic is as good as the instances it was checked on.

What the solution had to keep:

1. **`max_levers=None` is today's prescription to the bit.**
2. **A set's plan does not depend on how the set was reached.** It is the plan `prescribe` makes for
   those levers alone, so any answer can be checked against exhaustive search, bit for bit.
3. **An unselected lever has one documented value**, inside its box, at which it does nothing.
4. **What was left out is priced**: each step carries a bound on what the levers not yet chosen,
   and the solve itself, leave on the table.

## Decision

- `prescribe(max_levers=k)` selects by **greedy forward selection**. From no lever, each step plans
  once per lever not yet chosen, with that lever added, and keeps the cheapest plan; under
  `hold_constraints`, the plan whose audit clears the longest prefix comes first and the cost breaks
  the tie. Ties in both go to the lever listed first. `min(k, m)` steps and, for `k ≤ m`,
  `k m − k(k − 1)/2` solves.
- Each candidate is a **cold** `causal_plan` on the levers' own boxes, every lever outside the set
  pinned to `[0, 0]` — the call `prescribe` makes for that set alone. So (2) holds by construction,
  and once every lever is in, the last step's plan is the `None` plan.
- **An unselected lever is held at zero.** Zero is the one level at which leaving a lever out means
  the same thing everywhere it is read: the fitted channel is control-affine, so at zero it credits
  the lever with no effect in any state; the quadratic `unit_cost` charges nothing; and
  `InterventionSchedule.windows` reads zero as inactive. A `known` model is the caller's, and the
  library's own — the marketing mix's adstock — adds nothing at zero spend. A lever whose box
  excludes zero cannot be held off, so `max_levers` refuses it with `DecisionError`, naming the
  lever and its box, as it refuses `max_levers < 1`. Without `max_levers` such a box is planned as
  before.
- `Prescription.selection` is a frozen `LeverSelection` — `idle_cost`, the planned cost with every
  lever at zero, and one `SelectionStep(lever, task_cost, regret_bound)` per step — and `None`
  without `max_levers`. The schedule keeps a column per lever. `to_json` carries the selection under
  `"selection"`; `schema_version` stays `1`, because the module bumps it when a field changes
  meaning, not when one is added, and no existing field changed.
- A step's `regret_bound` is `plan_regret_bound` on that step's plan, **priced against every
  lever's box**. It bounds how far below the step's cost any plan the boxes allow can go — what the
  levers left out could buy, plus what the solve left — and, since the best set of the step's size
  is one such plan, it also bounds greedy's miss against that set. The certificate's `regret_bound`
  is the last step's.
- One `selection` record per step on `chc.decision`: the step, the lever, its cost and bound, every
  candidate's cost and cleared prefix, and the descent steps the candidates took.
- **No group-L1 stage**; the evidence follows, the reasoning is under Alternatives.

## Evidence

Costs and counts only: no wall time was taken. `scripts/bench_max_levers.py instances` plans every
subset of each instance's levers through the public `prescribe`, the others pinned to `[0, 0]`,
and compares greedy with the exhaustive best at every size short of all the levers. Four
alternatives are read off the same table: the group-L1 path's first support of the size, planned
alone (*path*); greedy among the levers of the path's first support larger than the size —
the proposed screen, then greedy (*screen*); greedy followed by best-improvement single swaps
(*swaps*); and backward elimination from the full set (*backward*). The path is solved on
`prescribe`'s planning objective, rebuilt and asserted to reproduce the prescription's plan to the
bit, by an accelerated proximal gradient whose per-column prox is held to L-BFGS-B first (worst
excess `3.6e-15`). A cell shows the set a method picked and its planned cost; `=` is the exhaustive
best.

| instance | size | exhaustive best | greedy | path | screen | swaps | backward |
|---|---|---|---|---|---|---|---|
| driver pool | 1 | boost `1.5526` | = | = | = | = | = |
| | 2 | incentive, boost `1.2446` | = | = | = | = | = |
| marketing mix, spend from zero | 1 | search `47.689` | = | social `53.180` | = | = | = |
| | 2 | search, social `6.7048` | = | = | = | = | = |
| saturation | 1 | wide `0.8947` | = | narrow_a `1.2423` | narrow_a `1.2423` | = | narrow_a `1.2423` |
| | 2 | narrow_a, narrow_b `0.0500` | narrow_a, wide `0.1757` | = | narrow_a, wide `0.1757` | = | = |

The driver pool is `tests/test_lever_selection.py`'s plant; the marketing mix is `chc.mmm`'s case
study with each channel's spend allowed down to zero. Over the six cells greedy finds the best set
in 5, the path in 4, the screen in 4, swaps in 6 and backward elimination in 5.

The saturation plant — one state, three levers — is built for greedy to miss: two cheap levers whose
boxes each stop short of the target, and a wider, dearer one that gets closest alone. Greedy takes
the wide lever first, as the best single lever, and its best partner is then a cheap one, at
`0.1757`; the two cheap levers together plan to `0.0500`, a relative excess for greedy of `2.52`.
The step's regret bound, `0.381`, covers the gap, as it must. The path finds the cheap pair but
enters a cheap lever first, so it misses at size 1 instead. The screen misses at size 1, because the
path's first two-lever support leaves the wide lever out, and at size 2 it admits all three levers
and repeats greedy. Backward elimination is greedy's mirror image, wrong at size 1 and right at
size 2. Swaps repair the miss.

`scripts/bench_max_levers.py random` repeats the comparison on 30 random plants with two states
and four levers: boxes two-sided or from zero, prices and targets drawn at random, and the second
state under a loose bound, so a lever can reach the target directly, through the second state, or
both. On two of them no lever lowers the cost, every set ties at the idle plan, and they are left
out. Over the 84 cells of the other 28, greedy finds the best set in all 84 — in all 28 at size 2
and all 28 at size 3, where it could miss — and so do swaps and backward elimination. The screen
finds it in 83 and the path in 69, skipping a size 6 times where two levers enter together to
within its resolution; with no miss to repair, the two only break hits, 1 and 15. Nine of greedy's
step bounds are infinite, where the measured curvature of the fitted objective is negative and the
bound declines to certify.

A full greedy ranking spent `m(m + 1)/2` solves — 6 on each named plant, 10 on each random one —
and its last step's plan was the unrestricted plan to the bit on every instance. The descent steps
of those solves came to `2.16`–`6.09` times the unrestricted plan's single solve on the named
plants, and `4.03`–`12.72`, median `6.02`, on the random ones.

Without `max_levers` nothing moved: `scripts/bench_max_levers.py fingerprint` hashes four
prescriptions — the driver pool with three levers, with a rate-limited lever under a priced bound,
under a held bound, and the marketing-mix case study as `run_marketing_mix` plans it — over their
JSON, `selection` aside, and the bytes of their actions and trajectories. The four digests are
identical on this branch and on its base, `75432d8`. XLA's floating point is not portable across
machines or JAX versions, so the digests are a check run twice on one machine, not a golden value
for the suite; the suite pins instead the shape that makes the `None` path identical — one cold
solve on the levers' own boxes, no warm start.

The tests were checked by mutation: fifteen edits to `decision.py` — the ranking blind to the held
constraints, a step priced on the box of the levers it uses, one step too many, the dearest
candidate kept, `None` routed through the selection, no zero-in-box check, candidates warm-started
from the last step, the idle cost read off the first step, the log record without its candidates,
the selection left out of the JSON, a step's bound written as its cost in the JSON, earlier choices
forgotten, an unselected lever held at its floor, ties to the lever listed last, and
`max_levers=0` let through. Each fails at least one test. One survived the first run, the ranking
blind to the held constraints: on the held-bound plant, holding the bound makes the plan of the
lever that cannot hold it dear enough for cost alone to pass it over. The test now makes that plan
free by hand, its audit untouched, and the mutant fails it. Read for the same kind of gap, the JSON
test compared only the lever names and now compares the whole selection, and the log test now
checks each candidate's cleared prefix.

## Consequences

- `max_levers=m` or more returns the unrestricted plan with the ranking that led to it, for
  `m(m + 1)/2` solves instead of one.
- Greedy's first step plans every single lever, so its first answer is always the best single
  lever; it can miss from size 2 to `m − 1`, and the saturation plant shows it does. A step's regret
  bound covers the miss, but it covers the levers left out as well, so a large bound does not say
  which of the two it is.
- A step always adds a lever, even one that buys nothing; its `task_cost`, unchanged from the step
  before, says so. Stopping early would need a threshold under which a drop does not count, and
  the solve's own tolerance does not supply one.
- The certificate's other numbers are the final plan's, and the model error and the barrier audit
  are still priced at every lever's authority — the widest box — so they are conservative when
  levers are left out.
- The marketing-mix case study's levers keep a spend floor above zero, and `max_levers` refuses
  them. A lever that must stay above a floor is written as the move above it: its column less the
  floor, its box from zero.

## Alternatives

- **A group-L1 stage** — the rule set before the first run was to add it only if it repairs a
  greedy miss, or if the lever count makes greedy impractical. As the proposed screen it repaired
  no miss, and broke one hit on the named plants and one on the random ones. As the selector itself
  it repaired the saturation miss, broke 2 hits on the named plants and 15 on the random ones, and
  skipped a size 6 times. The first lever to enter the path is the one that pulls hardest at the
  idle plan — the norm of the gradient on its column — and that is not the lever whose plan is
  cheapest: on the marketing mix social pulls hardest, `34.7` against search's `26.5`, and search
  alone plans cheaper. Nor is greedy impractical at these sizes: its cost is a known polynomial,
  `m(m + 1)/2` solves for a full ranking, not a search that can blow up. A screen earns its place
  when `m` solves per step become the bottleneck, and the instance where that happens is the one to
  measure it on. It would also bring a second solver into the library — a group prox, an
  accelerated descent and a `lambda` grid — needing the accuracy gates the projected gradient
  already has.
- **Greedy, then single swaps** — measured: it repaired the saturation miss and broke nothing,
  since it keeps a swap only if the cost drops, for `k (m − k)` solves a pass. Not shipped now,
  because the answer would stop being a chain of additions: `LeverSelection.steps`, each the lever
  a step added and what the set then cost, would need another shape, and the JSON with it. The
  first thing to add if greedy's misses show up in use.
- **Backward elimination** — from every lever, drop the one whose removal costs least. Greedy's
  mirror on the saturation plant, level with it on the random ones, and dearer when the question is
  a few levers out of many: reaching `k` from `m` takes `1 + (m(m + 1) − k(k + 1))/2` solves against
  forward selection's `k m − k(k − 1)/2`.
- **Exhaustive search** — exact, and the oracle of the tests and the bench, for `2^m − 1` solves.
  A caller with few levers can run it through `prescribe` by pinning boxes, as the bench does.
- **Warm-starting each candidate from the last step's plan** — rejected for (2): a candidate's plan
  would depend on the order its set was built in, and could no longer be compared bit for bit with
  the plan of the set alone, which the tests rest on. Its saving was not measured.
- **Holding an unselected lever at the point of its box nearest zero, or at its floor** — the lever
  would still act through the fitted channel and still be charged, so a plan reported as using `k`
  levers would move more than `k`. Refusing is the answer that cannot be misread.

## References

- Natarajan, B. K. (1995). Sparse approximate solutions to linear systems. *SIAM J. Comput.*
  24(2), 227–234.
- Das, A. & Kempe, D. (2011). Submodular meets spectral: greedy algorithms for subset selection,
  sparse approximation and dictionary selection. arXiv:1102.3975.
- Yuan, M. & Lin, Y. (2006). Model selection and estimation in regression with grouped variables.
  *J. R. Stat. Soc. B* 68(1), 49–67.
