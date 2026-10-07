# ADR 0065 — The search's settings are the caller's

**Status:** accepted, 2026-10-07. Amends ADR 0051 and ADR 0052: their searches stop at the
caller's tolerance and cap, no cut plans past the cap, and the tolerance has a floor at the rounding
of the dtype the curves are read in.

## Context

`allocate` (ADR 0051) and `cvar_allocate` (ADR 0052) search S-shaped curves by branch and bound.
Both stopped where every box's bound was within a share `1e-9` of a scale of the best plan's
worth, or after 500 boxes. The scale is the first box's bound for `allocate`, and the reference's
largest return for `cvar_allocate`. Both numbers were constants of the module:

- a caller could not trade a wider gap for fewer boxes, or hold a search to a count of boxes;
- the cap was read before a cut, and a cut plans two boxes, so a search the cap stopped planned
  501;
- the share `1e-9` is under float32's epsilon, 1.19e-7, and a gap read in float32 is rounding to
  within a few epsilons of the scale.

The rounding was measured on 300 random S-curve cases: one to four Hill, Weibull or logistic
channels, 1 to 26 periods, kernels of length 1 to 12, and up to 39 periods of history, 2 to 142
terms summed. Each case read its worth, and an envelope over a random box, at 12 random rates, in
float32 and in float64, and kept its largest error of the 12, in epsilons of float32 of the case's
largest value. The worth's was 1.1 in the median case, 3.4 at the 99th percentile and 7.1 at most.
An envelope less the worth, which is what a gap is, was off by 1.3, 3.7 and 6.4. The error did not
grow with the terms: the gaps of the 22 cases of over 100 terms were off by 1.8 at most.

The searches stalled on that rounding. With the default share and a cap of 150 boxes, 4 of 40
random cases of `allocate` ran to the cap in float32, on gaps of 0.23 to 2.5 epsilons of the scale;
float64 closed them in 3, 3, 11 and 3 boxes. Of 21 random cases of `cvar_allocate`, 5 ran to the
cap. Two, on gaps of 0.011 and 0.19 epsilons, float64 closed in 9 and 5 boxes. The other three, on
gaps of 2.1, 359 and 4428 epsilons, are the search's own: float64 closes two of them in 231 and 355
boxes, and leaves the third open at its cap of 500. One reading of two channels with carryover ran
to 501 boxes on a gap of 1.7 epsilons, where float64 closes it in 5. ADR 0052's example, the eight
readings with carryover at the level 1, closed in float32 in 29 boxes before this change, not at
the cap.

A result said how the search ended in `stopped` and `boxes` alone: not its gap, its tolerance, a
program HiGHS left unsolved, or the time.

## Decision

- **`allocate` and `cvar_allocate` take `rtol`, `atol` and `max_boxes`.** The search stops where
  every box's bound is within `max(atol, rtol * scale)` of the best plan's worth, the scale as
  before. The defaults are the constants they replace: `rtol=1e-9`, `atol=0` and
  `max_boxes=500`.
- **No cut plans past the cap.** Before a box is cut, its halves are counted: those that hold a
  split of the budget for `allocate`, one or two for `cvar_allocate`. The box is cut only where the
  count stays within `max_boxes`. A cap of 2 stops at 1 box, and a cap of 4 at 3.
- **The tolerance has a floor: 64 epsilons of the dtype the curves are read in, times the
  scale.** 64 is ten times the largest rounding of a gap measured above, 6.4 epsilons. In float64
  the floor is 1.4e-14 of the scale, under the default share. In float32 it is 7.6e-6, over the
  default share, so a float32 search closes at the floor.
- **A tolerance that is negative or not finite, or a cap that is not a whole number of at least 1,
  is refused** with a `ValueError` that names the argument.
- **A result records how the search ended.** `Allocation` and `CvarAllocation` add `gap`,
  `relative_gap` (the gap over the scale), `tolerance`, `floored` (whether the floor raised the
  tolerance), `limit` (`max_boxes`, `rounds`, `unsolved` or `None`), `compile_seconds` and
  `search_seconds`. `CvarAllocation` also adds `unsolved`, the iterations of each linear program
  HiGHS left unsolved, and `readings`. The compile seconds are the spans of JAX's compile events
  in the planner's thread, a span inside another counted once. Both are recorded, not promised.
- **The planners that do not search take no new arguments.** A goal's plan is judged at the share
  `1e-9` of its bound, with no floor, as before, and records that tolerance and its gap.
  `minimax_allocate`, `allocate_geos` and `budget_for_geos` cut no boxes.

## Consequences

- The defaults plan as before, to the bit. Every result that the eight test files importing
  `chc.allocation` make was recorded before and after the change, 476 of them: the planners' and
  those of `_on_envelopes`, which the tests also call. Every field that existed before was
  compared bit for bit, and all are equal. The old test files run on the new code make the same
  results, but for the two tests that set the module's cap, which is gone; those now pass
  `max_boxes=1`, and plan the same. No test's search plans more than 59 boxes, so the exact cap
  moves nothing at 500.
- In float32 at the floor the nine searches above close: `allocate`'s four in 3, 3, 11 and 3
  boxes, as in float64; `cvar_allocate`'s two on rounding in 9 and 5, as in float64, and its other
  three in 109, 193 and 417. The one-reading case closes in 5. A float32 plan is the best only to
  7.6e-6 of its scale: two searches that closed without the floor at gaps of 0 and 0.002
  epsilons, in 173 and 317 boxes, now stop at 53 and 48 epsilons, in 109 and 193. ADR 0052's
  example closes in 19 boxes, where it took 29, on a split whose mean gain is 860.6974, where it
  was 860.6995.
- In float64 the floor matters only for a share under 1.4e-14. `cvar_allocate`'s bound is a linear
  program's, solved to HiGHS's tolerance of 1e-10, and the floor does not cover that. Asked for no
  gap and with no floor, 9 of 13 searches on the test suite's cases ran to a cap of 120 boxes;
  8 of them on gaps of 6.7e-13 to 3.2e-11 of the scale, over the floor, and their results say
  `limit="max_boxes"`. `allocate` closed all 18 of its cases to a gap of 0.
- A wider tolerance plans fewer boxes: the untied example closes in 3 boxes at `rtol=0.05`, where
  the default plans 5.
- The compile seconds are read from `jax.monitoring`. From jax 0.4.30, the floor, to 0.8.0 it
  publishes no way to remove a listener, and there the module's own
  `_unregister_event_duration_listener_by_callback` removes it. The new tests pass on the floors,
  jax 0.4.30, numpy 2.0 and scipy 1.13 on Python 3.11, as on jax 0.11.2.
- The tests hold caps of 1 to 5 and 7 boxes to their counts and plans, each tolerance to the
  count it stops at, the floor in float64 and in float32, the refusals, the limits, the iterations
  of a stalled program, and the compile seconds. Each of 29 hand-made mutants of the change fails
  a test, and the change unmutated passes them.

## Alternatives

- **`atol + rtol * scale`, as `numpy.isclose` reads them.** A caller who sets both would get a
  looser tolerance than either. Mixed-integer solvers stop at either gap, and so does this.
  Rejected.
- **A relative gap over the best worth or the last bound**, as mixed-integer solvers read it. Both
  move as the search runs; the first box's bound is known before it starts, and it is the scale
  the defaults reproduce. Kept the first box's.
- **A floor of 16, or of 1.** 1 is under the rounding measured, where searches stalled. 16 clears
  the largest, 6.4 epsilons, by 2.5 times, on cases of at most 142 terms; larger problems were not
  measured. At 64 a float32 plan is still the best to 7.6e-6 of its scale. Rejected.
- **A floor at the planes' tolerance for `cvar_allocate`.** It would close the float64 searches
  that ask for less than 1e-10 of the scale, but it is HiGHS's tolerance, not the dtype's. Left:
  where it is the limit, the result says `limit="max_boxes"`.
- **A time limit in seconds.** A search would plan another count of boxes on a loaded machine. A
  cap in boxes plans the same everywhere, and the seconds are recorded. Rejected.
