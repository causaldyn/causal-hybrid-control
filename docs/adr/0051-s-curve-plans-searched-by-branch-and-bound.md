# ADR 0051 — S-curve plans are searched by branch and bound

**Status:** accepted, 2026-10-05. Amends ADR 0034 (an S-shaped curve is searched, not planned on
its envelope) and ADR 0050 (the vertex is each box's plan, the first box's where the search
starts). Amended 2026-10-05 by ADR 0052: `cvar_allocate` searches boxes too.

## Context

`chc.allocation.allocate` planned S-shaped curves on their concave envelopes from zero spend
(ADR 0029), and since ADR 0050 took the vertex of that plan. The vertex leaves at most one channel
inside its chord, but it is still the envelope's plan, not the curves' best:

- two Hill curves of slope 3 at scales 1 and 1.01 and a budget of 1.6 were planned 1.27/0.33 for
  0.705, where all of it on the first returns 512/637, 0.804;
- the envelope from zero spend ignores the box. A cap short of a curve's tangency is bounded by
  the chord from the origin to the tangency, which stands above the curve at the cap: two Hill(3)
  channels capped at 1 were bounded by 0.815 where the best split returns 0.660;
- with a longer kernel each period runs at its own adstock, so a channel's chord in one period is
  its curve in another, and the plan on the envelopes can be far from the best: 8.04 against 9.32
  for a Hill and a logistic channel with carryover and history (the test below).

`bound - worth` reported each of these honestly, but a planner that reports a gap of 15 % has not
planned.

## Decision

- **`allocate` searches boxes of the rates** (Udell and Boyd 2016, sigmoidal programming). A box's
  bound is the bisection's plan on each channel's envelope over the box, and that plan, read on the
  curves, is a plan. The box of the largest bound is cut first, on the channel whose envelope
  stands furthest above its curve at the plan, at its rate there, or halfway along its interval
  where the rate is an end of it. Boxes within a share `1e-9` of the first box's bound of the best
  plan's worth are settled; the search stops when every box is, or after 500 boxes, and the bound
  returned is the largest left or settled.
- **The envelope over a box.** A box `[l, u]` of a channel's rate is an interval
  `[a_t, b_t]` of each period's adstock. The curve's concave envelope there is the chord from
  `(a_t, h(a_t))` to where it touches the curve, `h(w) - h(a) = (w - a) h'(w)` with `w` past the
  inflection, or to `b_t` where that comes first, and the curve beyond; the curve itself from the
  inflection on. The sum over periods is concave in the rate, lies above the worth, and meets it at
  both ends of the box, so a box cut at a rate is bounded tightly there on both sides, and a plan
  that sits at its channels' ends closes its box.
- **The touch from a start** (`chc.response._touches`) is the tangency from the origin generalised:
  every period's start is bisected at once, under one compiled program, to the least double where
  the gap turns positive, as `_first_turned` closes on the tangency from the origin. From zero
  spend it is that tangency for every S-shaped family, to the double but for one ulp on Burr XII;
  from a start it is Maxima's root to 1e-14 (`validation/envelope_on_interval.mac`). A bisection
  reads only the gap's sign, so the corner where a beta or Kumaraswamy CDF meets its ceiling is no
  trap for it.
- **A chord cut short by the cap is the whole box's envelope.** Read on the curve at the cap, its
  slope would jump to the curve's own, steeper than the chord's, and the bisection's rate test at
  the cap would read a slope the envelope does not have.
- **Concave curves are planned as before, by the same code.** The other planners keep the
  envelope from zero spend: `minimax_allocate` and `cvar_allocate` read regrets and gains on it,
  which the cutting planes need convex and concave; `allocate_geos` cuts planes on it; and a goal
  is met along the path of its plans. `budget_for` therefore returns the plan on that path, not
  `allocate`'s.

## Consequences

- Plans on S-curves change, and only there. The untied example returns 512/637 with no gap; the
  capped pair returns its best split with a bound of the same; the case with carryover returns
  9.32 where the envelopes returned 8.04.
- Each box costs one bisection of the price, as a plan on concave curves does. Counted in boxes,
  not timed, since the machine was loaded: the untied example closes in 5, the capped pair in 5,
  the case with carryover in 7, three channels with carryover in 5, a Kumaraswamy and a Hill in 19,
  and 30 Hill channels drawn at random (scales 0.5 to 2, slopes 1.5 to 4, caps of 3, a budget of
  15) in 6, 8 and 6, each to a gap of 0.
- Sums of S-shaped curves under a budget are NP-hard to plan, so nothing short of the cap bounds
  the count of boxes. A plan that reaches it reports its gap as it stands.
- `Allocation.bound` on S-curves is the search's bound, and `Allocation.price` is read on the
  envelopes of the box the plan was found in.
- ADR 0050's bound on the first box's gap still holds, since with floors at zero and caps past the
  tangencies the first box's envelopes are the envelopes from zero spend; it now bounds where the
  search starts, not what it returns.

## Alternatives

- **A separate exact planner, or a flag on `allocate`.** `allocate` promises the most return, and
  a flag would keep the envelope's plan the default. The cost is bounded by the cap. Rejected.
- **A beam over channel on/off patterns.** It has no bound (ADR 0050). Rejected.
- **The concave envelope of a channel's whole worth in its rate**, in place of the sum of each
  period's. It is tighter with long kernels, but it has no closed form, and the per-period sum
  closes the search on every case tried. Left until a case needs it.
- **Frank–Wolfe meets Shapley–Folkman** (Dubois-Taine and d'Aspremont 2024) returns a vertex of low
  nonconvexity, the plan ADR 0050 already takes, not the best.
- **A mixed-integer second-order cone programme** (arXiv 2101.03663) needs a solver this package
  does not carry. Rejected.
