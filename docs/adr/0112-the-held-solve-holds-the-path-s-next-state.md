# ADR 0112 — The held solve holds the path's next state

**Status:** accepted, 2026-10-19. Amends ADR 0101, whose last consequence it closes, and the
barrier rounds of ADR 0102. Amended 2026-10-20: rounds that end short of the condition are
projected onto it.

## Context

ADR 0101 made the audit certify a step only where three things hold: the barrier condition, `h`
at the step's state, and `h` at the next predicted state. The held solve still held the condition
alone. At a coarse step the condition can hold at every sample while the path leaves the set
between them, so a plan held for the audit could fail it.

On `x' = 0.5 x + u`, `dt = 1`, a ceiling `1 - x` at `alpha = 1` and twelve steps, the held plan's
path left the set at 5 of 12 states, least `h` -0.050, and its own audit certified 3 steps.

## Decision

- Each barrier holds two columns per step in the rounds: the condition's shortfall, and `-h` at the
  step's next state. The next state of the last step is its own RK4 step. The multipliers are
  `(steps, 2 * barriers)`.
- A next state's column is scaled to the largest condition's scale, by the largest `|h|` along the
  path, so one back-off and one penalty read every column alike. A condition's column keeps the
  factor it had.
- The next states join the rounds as an active set. The rounds first hold the conditions alone.
  Where the path then falls short of `h >= 0` at a next state not yet held, by the back-off, that
  state joins and the rounds go on. They stop when no next state not yet held falls short. A
  receding horizon's warm multipliers keep the next states that bound.
- Where the rounds end with a held column short of the audit's own condition, the plan is
  projected onto the held columns at the back-off. The projection is the least move in the
  planner's variables `v = sigma u` that holds the columns, linearised, with the box and the rows.
  It is one least-distance program (Lawson & Hanson), iterated up to 20 times on a nonlinear
  plant. Next states that the move takes out of the set join, as in the rounds. The status reads
  `max_iterations`. Where no move holds the columns, the plan and `infeasible` stay.

## Consequences

- The example above certifies 12 of 12 steps, its path's least `h` is 0. The task cost is 11.35.
  The base's 10.04 was the cost of a plan that left the set. SLSQP's optimum under both conditions
  is 8.02, so the rounds stop 41 % above it, at `max_iterations`. That is the diagonal metric on
  an ill-conditioned problem, which ADR 0102 leaves open: the condition alone stopped 28 % above
  its own optimum, 7.84, on the same problem.
- Where the condition alone keeps the path in the set, no next state joins, and the plan is the
  one it was: at `alpha = 0.5` and `0.2` above the costs read 10.274757 and 12.378456 before and
  after, and the step counts 240 000 and 200 000.
- The solver status `infeasible` now also reads a next state the penalty's ceiling left short.
- The rounds do not converge on the example above: every round spends its 10 000 descent steps.
  Where they stop is set by rounding. A warm start moved by 1e-15 moved the task cost at
  `alpha = 0.2` from 12.20 to 14.16; the optimum is 11.39 and binds at all 12 steps. On JAX
  0.4.30 one such end fell 5e-5 short of the condition at step 1, and the audit certified 1 step
  of 12. The projection moves that plan to 12 of 12, its task cost from 14.038 to 14.082. It does
  not move the plan toward the optimum; a block or Newton subproblem (ADR 0102) is still the fix
  for that.

## Alternatives

- **Every next state held from the first round.** The rounds stiffen where no next state binds.
  At `alpha = 0.2` above, the rounds then stopped as `converged` at 17.05, 38 % above the 12.38
  the condition alone gives, whose path never leaves the set.
- **A discrete-time barrier condition**, `h(x_{k+1}) >= (1 - gamma) h(x_k)` against every channel
  in the set. It needs the worst case of `h` over the set's next states, which has no closed form
  for a nonlinear `h`.
- **Hold only `h(x_H) >= 0`.** A path can leave the set and come back.
- **Restart the rounds where they end short.** Birgin and Martinez's start reads the shortfall
  left, and 5e-5 of it sets a penalty above the ceiling that stalled the last round.
