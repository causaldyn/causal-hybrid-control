# ADR 0156 — A held barrier reads `infeasible` only at a first-order point, and counts its penalty's growths

**Status:** accepted, 2026-11-21. Amends the status rule of ADR 0102's barrier rounds and the
ceiling of ADR 0144's outer rules. A public status moves on some plans: a minor release.

## Context

`causal_plan(barrier=...)` holds the barrier condition by augmented-Lagrangian rounds (ADR 0102),
which stop where the penalty would pass 1e8 times its start. There they read `infeasible` if some
step still falls short of the condition after a round that stopped on its own rule, and
`max_iterations` after a round its share of the budget cut (ADR 0144).

A test of `outer="safeguarded"` on an unstable drift, `xdot = 0.5 x + u - 3 Z` with a hidden
confounder at `Gamma = 1.5`, held under the fit's own `ChannelSensitivity`, asserted
`max_iterations`. It passed on this machine and read `infeasible` on a CI runner on the dependency
floors, with the same packages as a run that passed the day before. The condition is feasible
there: SLSQP meets it at every step. Three faults were found, and one property of the test.

1. **The rounds are chaotic in their last bits.** From start states moved from zero by 1e-16 to
   7e-8 (91 values), the safeguarded rounds end in one of two ways. Either the first set of
   rounds runs until the budget is spent, within the back-off, and the projection onto the next
   states then holds every step; or those rounds stop at the penalty's ceiling, the next states
   join (ADR 0112), and the rounds on them start at that ceiling. The second way ends at a far
   point of the condition, a shortfall of 13 to 22, certified on 27 to 30 steps of 30. Each start
   state repeats exactly on one machine; which way it goes differs between CPU classes and
   dependency versions. So the test pinned the status of one path of many.
2. **`infeasible` was read without stationarity.** At a far point the last round's descent can
   stop before its share: it stops where no step buys its share of the round's objective, which
   the penalty at its ceiling inflates. On the floors' jax, one start state's last round took no
   step at a shortfall of 14, at a stationarity of 0.865 against a tolerance of 1.41e-6, and read
   `infeasible`. That is a stall, not a sign of the condition.
3. **The ceiling was decided by rounding.** The rounds stopped where `rho * 10 > rho_0 * 1e8`.
   After seven growths the left side is `rho_0` times ten seven times, times ten; the right is
   `rho_0` times 1e8. They are equal in exact arithmetic, so rounding decided whether an eighth
   growth was taken: of the 44 start states on the floors whose rounds stopped at the ceiling, 25
   stopped after seven growths and 19 after eight.

## Decision

- **`infeasible` needs a first-order point.** At the ceiling, some step short after a round that
  stopped on its own rule, the rounds read `infeasible` only where the plan's stationarity, read
  with the multipliers they end with, is within its tolerance: the check `converged` already
  requires (`_stationary`). Where it is not, they read `no_progress`: the rounds stalled, which
  says nothing of the condition. `SolverStatus`'s docstring gains that reading of `no_progress`
  for the barrier rounds; `infeasible` keeps its own, the sign of a condition no plan meets, not
  a proof of one. A round its share cut still reads `max_iterations`.
- **The ceiling is a count.** `_PENALTY_GROWTHS = 8` tenfold growths, to 1e8 times the start,
  replace `_PENALTY_RANGE` and the comparison. The count carries across the rounds on each set
  of next states, as the ceiling did. The eighth growth, which exact arithmetic allows, is now
  always taken.
- **A relaxed start keeps the rounds' status.** Where the model holds a curve that starts convex,
  `_plan` replaced a final `no_progress` with the relaxed plan's status, since a descent that took
  no step from the relaxed plan stopped for that plan's reason. The rounds now report
  `no_progress` of their own, so the replacement moves before the rounds, onto the descent's own
  result: a stall of the rounds is not read as the relaxed plan's `converged`.
- **The test asserts the rule, from several start states.** It runs from nine start states,
  zero and eight shifts of 7e-16 to 1e-9, and asserts what holds on every path: the penalty first
  grows after a round its share cut; the rounds never converge on this log; a plan reads
  `infeasible` or `no_progress` only where its last round took fewer steps than a full share; the
  audit certifies some steps. About half of the start states end at the far point on any one
  machine, where the rule is exercised. With the rule reverted (a cut round read like a solved
  one) 8 of the 9 fail on this machine, on the lockfile and on the floors alike. The assertion is
  not relaxed to accept `infeasible`.

## Measurements

Statuses on the 91 start states after the change, on this machine (Ryzen 7 5800HS):

| | within the back-off | far point, last round cut | far point, last round stalled |
|---|---:|---:|---:|
| lockfile (jax 0.11.2, Python 3.14) | 49, `max_iterations` | 38, `max_iterations` | 4, `no_progress` |
| floors (jax 0.4.30, Python 3.11) | 58, `max_iterations` | 32, `max_iterations` | 1, `no_progress` |

None converged and none read `infeasible`; each was certified on 27 steps of 30 or more. The five
stalled rounds read stationarity 0.78 to 0.96 against 1.41e-6; the rule before this one read each
of them `infeasible`.

On the library's planning tests, 293 plans held at a barrier (276 under the default rounds, 17
under `"safeguarded"`), each solved by this rule and by the one before it, on the lockfile:

- 286 are the same actions after the same descent steps.
- Six plans move from `infeasible` to `no_progress`, and no other status moves; none moves to or
  from `converged`. Five are of twenty unstable logs held with the drift's error read into the
  room, under the default rounds, each the same plan as before, certified on 22 to 28 steps of
  30, at a stationarity of 0.62 to 1.0 against 1.41e-6. The sixth is the unreachable position
  under Consequences, which three tests build.
- Nine plans stay `infeasible`, each within its tolerance: four that a box and rows pin to one
  point, one whose projection holds no step, one on a box in which no action lowers the state,
  two of a lever that moves the state only through the fit's noise, and a float32 plan at 6.2e-4
  against its 9.8e-3.
- The count moves three plans, those whose rounds rounding had stopped one growth short: the
  unreachable position, by 24 descent steps and a cost higher by 1.5e-9 of itself; and two start
  states of the safeguarded drift above, one certified on 29 steps of 30 where it was on 28, at a
  cost 0.56 % higher, the other 0.10 % cheaper on its 29. None moves to fewer certified steps.

Re-read under `outer="safeguarded"` against the default rounds, the tally the `outer` argument's
docstring states (ADR 0144) now reads: of the 293 it moves 138, 40 to a cheaper plan at the same
certified steps, 73 to a costlier one (by up to 43 %), 7 within 1e-9, 2 to more certified steps and
16 to fewer; 18 that converged end `max_iterations`, and 8 that ended `max_iterations` stop
`no_progress`. The two integrator plans ADR 0144 counted from 12 certified steps to 7 and
`infeasible` now end on 8, `no_progress`, at a stationarity of 0.84: the eighth growth moved them
(13 093 descent steps where they took 9 370), and the rule read the stall.

## The far point: an open item

The diagnosis proposed that a round's objective and the shortfall read after it differ, since a
monotone descent on the penalised cost could not move the plan from a shortfall near 1e-5 to 14 at
a penalty of 8.3e5. Measured, on four start states on the floors (the penalised cost before and
after every round): **it does not hold.** The objective falls in every round. The far point is not
one round's step. The first set of rounds stops at the ceiling 1e-5 short of the conditions; the
next states of steps 9 to 29 then join, short by 50 in the columns' units; the rounds on them
start at the ceiling, where the first moves the plan by 0.45 and the shortfall from 50 to 14.6,
and the second, its multipliers at 1.2e7, moves it by 2e-7, fails to halve the measure and ends
the rounds. The projection onto the held columns then fails from there, and the plan is the
longest leading run a projection holds, 27 to 29 steps. Where the first rounds run until the
budget is spent instead, the same next states join with no rounds left, and the projection from
there holds all 30 steps. ADR 0144 named this join at a high penalty for the third log.

Two remedies were tried under `outer="safeguarded"` alone, against ADR 0144's rule: a fix only if
every held plan moves toward the optimum at no shorter certified prefix.

- **The penalty chosen afresh at a join**, Birgin and Martínez's start on the joined shortfall,
  as for a receding horizon's warm multipliers. On seven start states of the floors it ended
  16 to 88 short with 24 to 29 steps certified, where they had ended with 27 to 29, and one read
  `infeasible`. Rejected.
- **The projection first at a join**, then the rounds from the projected plan where it holds. On
  the 91 start states it certified more leading steps on 36 (lockfile) and 30 (floors), left 53
  and 61 alone, and certified fewer on none; far points fell from 42 to 4 and from 33 to 3. But
  on two start states of the lockfile, both certified on all 30 steps, it moved the plan 0.87 %
  and 0.76 % further from SLSQP's cost (51.055 → 51.501, 51.062 → 51.448, against 50.995). It
  fails the rule on this log alone, and was not run on the corpus.

So the planner is unchanged here, and the far point stays open: under the safeguarded rule, about
a third to a half of the paths near this start end certified on fewer steps than the rest. The
projection at a join is the candidate to measure on the corpus, with the two counterexamples above
to explain.

## Consequences

- A plan that read `infeasible` now reads `no_progress` where the rounds stopped off a first-order
  point. Some true infeasibility moves too: on a position held from `x = 1` at `x >= 0`, relative
  degree 2, no action moves the first step's condition. The lockfile's jax stops its rounds at a
  stationarity of 0.069 and now reads `no_progress`; the floors' stop within it and still read
  `infeasible`. Before the count the lockfile's stopped one growth sooner, within it. The audit,
  `CausalPlan.safety`, reports such a plan either way: no step certified, and the first step's
  `gamma*` not a number.
- Where rounding refused the eighth growth, the rounds now take it, and the plans that reached the
  ceiling move.
- `no_progress` from the rounds is not `no_progress` from a descent: the plan moved. Code that
  reads `no_progress` as "the result is the initial guess" reads it wrongly under a barrier; the
  docstring says so.

## Alternatives

- **Stationarity of the infeasibility measure**, `||max(c, 0)||^2 / 2` over the box, the
  condition Birgin and Martínez's theory gives at an infeasible limit point. Its first-order
  residual, relative to `||c+|| ||J||`, read 1.7e-9 and 2.9e-10 on the position above (the step
  that fails has no gradient), 0 on a velocity floor the box cannot meet, and 1.3e-5 to 2.1e-5 at
  the drift's five stalls. Four orders separate a true sign from a false one, but no tolerance in the
  library sits between them, and one chosen to fit these cases would be fitted, not derived.
  Rejected.
- **A new status, `stalled`.** `SolverStatus` is a public `Literal`; a new value breaks code that
  matches it exhaustively, a harder door than a status that moves within the four. Rejected.
- **Keep `infeasible` and document it as weak.** The docstring already called it "not a proof".
  A status that reads a condition SLSQP meets as one no plan meets is a false sign, and a runner's
  rounding decided it. Rejected.
- **The ceiling as `rho >= rho_0 * 1e8 * (1 - eps)`.** Exact enough, but a tolerance where a
  count states the rule. Rejected.
