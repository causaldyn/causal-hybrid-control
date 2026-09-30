# ADR 0034 — A budget allocated over channels

**Status:** proposed, 2026-09-30.

## Context

`chc.response` describes a channel as a kernel, a curve and a coefficient, and `chc.lift` fits one
to geo tests. Nothing planned on them. `prescribe(budgets=...)` (ADR 0031) plans a budget over
`chc.mmm`'s plant, whose adstock is a state in continuous time, which a discrete `Channel` does
not describe. A fitted media-mix model is a set of discrete channels, and the question put to it
is how to spend next quarter's budget across them.

## Decision

- **`allocate(channels, budget, periods, *, lower, upper, history=None)`** in a new module,
  `chc.allocation`, returning an `Allocation`: each channel's spend a period, the plan's worth, a
  bound on the best plan's, and the budget's shadow price.
- **One rate a period for each channel.** The response does not change with the period, so what
  makes periods differ is the history's carryover at the start and the tail at the end, and the
  decision is the split.
- **Carryover counted both ways.** The adstock the history leaves runs into the plan, and the
  plan's spend runs on for each kernel's length after it with nothing spent. A channel with a long
  kernel is then not undervalued for the periods its return outlasts the plan.
- **Exact by bisection on the price.** Each channel's worth is concave in its rate when its curve
  is concave and its coefficient not negative, and the channels add, so every rate is a monotone
  function of one price. A linear channel's rate jumps at one price; the plan mixes the rates
  either side of the jump, both best there, in the proportion that spends the budget exactly.
- **An S-shaped curve is planned on its envelope** (`relax`, ADR 0029), and `bound - worth` bounds
  the plan's shortfall from the best on the true curves; on concave curves it is 0.
- **The box has no default.** It is where the channels were seen; a plan outside it reads the
  curves where nothing measured them.
- **Refused**: `Ricker` and a negative coefficient, on which the bisection proves nothing; a
  budget the box cannot spend; a box, history or channel of the wrong shape or type.

## Consequences

- `tests/test_allocation.py` holds the plan against a second route to each number: its worth
  against the channels run over the history, the plan and the tail as one series; the plan against
  SciPy's SLSQP and 200 random plans in the box; the price against the worth's own slope for every
  channel inside its box; a linear channel against the closed form of where the concave one stops;
  the S-curve counterexample of ADR 0029 at 1047.867 with no gap, and below the tangency a gap that
  brackets the best split found on a grid; and a change of currency that moves the spend and the
  price and leaves the worth.
- causaldyn-bench's Track M v2 plans every arm through it and scores the plans against an oracle
  written there without it; the two agree on the world's own channels.
- The plan is only as good as the channels it is handed. A fitted channel's error goes straight
  into the split.

## Alternatives

- **A schedule, one spend a period for each channel.** A problem of `periods × channels` rates
  whose optimum differs from the flat plan only at the ends, and not what a media plan is asked.
- **`causal_plan` over a plant built from the channels.** Its cost is quadratic in the states, so a
  return would be a target to track rather than a sum to maximise, and its descent has no exact
  treatment of a linear channel's jump.
- **A general solver (SLSQP).** No certificate on an S-curve, and at a linear channel's jump it
  stops where its tolerance does.

## Not built

- **A plan robust to the channels' uncertainty**: a split that pays under every channel in an
  identified set, or the smallest error in a channel's reading that would reverse a move.
