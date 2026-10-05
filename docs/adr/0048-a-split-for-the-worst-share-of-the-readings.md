# ADR 0048 — A split for the worst share of the readings

**Status:** proposed, 2026-10-03. Amended 2026-10-05 by ADR 0052: a gain is read on the curves,
and on S-shaped curves the split is searched in boxes.

## Context

A Bayesian media-mix fit hands the planner its draws: a posterior of channels, not one reading of
them. Planning on the draws' mean return moves spend where they gain on average, and says nothing
of how many of them it loses under. A plan changed on the strength of a fit is a change somebody
answers for, and without a test to read the channels the fit's own spread is the only evidence of
how it can be wrong. `minimax_allocate` (ADR 0038) answers the worst case among a few readings;
over a posterior's draws the worst case is one draw, which decides alone.

## Decision

- **`cvar_allocate(readings, budget, periods, *, level, against, lower, upper, history=None)`** in
  `chc.allocation`, returning a `CvarAllocation`: the split, each reading's gain on it, the mean
  gain of the worst `level` share of the readings, and a bound above it.
- **Gain over a reference, not return.** A split's return under a reading is mostly the reading's,
  whatever the split; its gain over the plan in place is what the split changes. The reference,
  `against`, spends the budget in the box and gains 0 under every reading, so the split returned
  never does worse in the worst share than keeping the plan.
- **The gain's conditional value at risk.** The split maximises the mean gain over the worst
  `level` share of the readings, weighed alike, the last in part, `max_eta eta - E[(eta -
  gain)_+] / level` (Rockafellar and Uryasev 2000). At `level = 1` it is the mean return's split;
  at `1 / readings` the worst reading's. The level has no default: it is how much of the posterior
  the plan may not lose on, which is the caller's to say.
- **Exact by cutting planes, with a certificate.** Where every curve is concave each reading's gain
  is concave in the split, and so is the share's mean. Kelley's (1960) cutting planes take every
  reading's tangent at every split tried, starting from the reference; the linear program over
  them, the rates, `eta` and one excess a reading (HiGHS, its rows sparse), bounds the most from
  above, and its solution, moved onto the budget, is the next split tried. The loop stops when the
  bound is within `1e-9` of the reference's largest return above the best split's mean gain, or
  after 500 rounds, and the gap is returned either way. The planes are read in units of that
  return, as `minimax_allocate`'s are.
- **An S-shaped curve is planned on its envelope**, as `allocate` plans it, and the gain read there.

## Consequences

- `tests/test_cvar_allocation.py` holds the split to closed forms: two linear readings whose gains
  from the even split are `2 d` and `-d`, which stay at the worst half's level, move all the way at
  0.8 and at 1, to 6.25 and 25, and do so in a unit `1e18` times smaller. On three channels with
  carryover under thirty readings of one family a tenth apart, no split of a 241-by-241 grid has a
  larger mean gain in the worst share at 0.1, 0.3 or 1; at 0.1 the split is the reference, which no
  grid split beats. Seven readings alike are planned as `allocate` plans the one. A Hill's gain is
  its envelope's. A property over readings, levels and references holds the share's mean gain at or
  above 0, equal to the gains' own, and the bound above it by no more than the gap. Fourteen
  mutations caught.
- **It moves only where the worst share gains.** On the thirty readings a tenth apart, the mean's
  split loses under 3 of them, the worst by 120.9, and read in the worst 0.3 share it loses 16.1;
  the split for that share moves 0.345 of the way, gains 5.53 there and loses under 2, the worst by
  29.3. On thirty readings of three families each forty percent apart, the mean's split gains 38.4
  on average and loses under 16 of the thirty, the worst by 310.2; at each of the levels 0.1, 0.2,
  0.3, 0.5 and 0.8 the split is the reference.
- The program carries one excess a reading and adds one plane a reading each round, so its size
  grows with the draws read: a posterior is thinned to the draws the plan reads.
- It is robust to the readings it is given and to no other: draws of a misspecified model are
  misspecified alike.
- Not measured: whether a plan on the worst share serves a decision better than the mean's split
  on worlds where the truth is known.

## Alternatives

- **The mean return** (`level = 1`): on the three families above it gains on average and loses
  under more than half the readings.
- **The least worst regret over the draws** (`minimax_allocate`): over many draws one decides, and
  its regret is against each draw's best split, not the plan in place.
- **A chance constraint**, the share of readings that lose held under a bound: not convex in the
  split, and blind to how much the losing readings lose. The conditional value at risk is its
  convex conservative approximation (Nemirovski and Shapiro 2006).
- **Mean less a multiple of the variance:** penalises a reading's gain above the mean as much as
  one below it, and is not monotone in the gains.
