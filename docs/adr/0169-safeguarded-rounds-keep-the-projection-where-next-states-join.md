# ADR 0169 — Safeguarded rounds keep the projection where next states join

**Status:** accepted, 2026-11-25. Experimental (`outer="safeguarded"`). Closes the open item of
ADR 0156, the far point, under the safeguarded rule. The default rounds, `outer="halved"`, do not
change, bit for bit.

## Context

ADR 0156 left the far point open. On an unstable drift, `xdot = 0.5 x + u - 3 Z` with a hidden
confounder at `Gamma = 1.5`, held to `x <= 1` under the fit's own `ChannelSensitivity`, the
safeguarded rounds from start states moved from zero by 1e-16 to 7e-8 go one of two ways. Either
the first set of rounds runs until the budget is spent and the projection onto the next states
holds every step; or those rounds stop at the penalty's ceiling, the next states of steps 9 to 29
join short by about 50 in the columns' units, and the rounds on them start at that ceiling. Their
first round moves the plan by about 0.45 and leaves it short by about 15 (ADR 0156), the
projection from there fails, and the plan is the longest leading run a projection holds, 27 to 29
steps of 30.

The candidate ADR 0156 named: at such a join, the projection onto the held columns first, then
the rounds from the projected plan. On the 91 start states it certified more leading steps on 36
(lockfile) and 30 (floors), fewer on none. But on two start states of the lockfile, certified on
all 30 steps either way, it moved the plan further from SLSQP's cost: 51.055 to 51.501 and 51.062
to 51.448, against 50.995. ADR 0144's rule for a change under the safeguarded rule: a fix only if
every held plan moves toward the optimum at no shorter certified prefix. That candidate fails it.

## Why the projection first ends costlier there

On both start states the first rounds stop at the ceiling, `rho = 8.34e6` after eight growths,
their last round cut by its share, and the next states of steps 9 or 10 to 29 join, short by 49.7
and 48.8. Both plans certified on all 30 steps are then projections onto the same held columns
(`_restore`, the least move in the descent's scaled variables); they differ in the point projected
from.

| start | plan | projected from | its cost, certified steps | plan's cost | over SLSQP's |
|---|---|---|---|---:|---:|
| `-3e-12` | before | the iterate after two rounds at the ceiling | 46.556, 17 | 51.055065 | +0.118 % |
| `-3e-12` | projection first | the first rounds' last iterate | 46.695, 9 | 51.501220 | +0.992 % |
| `-2e-9` | before | the iterate after two rounds at the ceiling | 47.078, 18 | 51.061965 | +0.131 % |
| `-2e-9` | projection first, then 14 rounds | the first rounds' last iterate | 46.739, 10 | 51.447976 | +0.888 % |

- **Not a different local minimum.** Neither plan is a first-order point: their stationarity
  reads 0.88 and 0.95 on the first start state, 0.99 and 1.0 on the second, against a tolerance
  of 1.41e-6. SLSQP's plan spends one action, 0.453 at the first step, and rests the other 29 on
  zero, the kink of the tube's norm (ADR 0144); its actions sum to 0.453 in absolute value, the
  plan before's to 0.494, the projection-first plan's to 0.593. The projection is blind to the
  cost: from a plan short by 50 it spreads the correction over every action, where the two
  rounds at the ceiling first moved the plan along the penalised cost, lowering its cost from
  46.695 to 46.556 on the first start state, and left the projection less to move.
- **Nothing after the projection lowers the cost.** On `-3e-12` the projected plan clears every
  held column within the back-off, so the loop reads it as a plan with nothing left to hold and
  runs no round: 4 339 of the 10 000 steps go unspent. Rounds forced from it, at the ceiling,
  move the cost from 51.50122 to 51.49932 in those 4 339 steps. On `-2e-9` the rounds do run, at
  the ceiling: 14 rounds, 4 623 steps, from 51.4744 to 51.4480. At the ceiling the rounds move
  the cost by 0.004 % and 0.05 %. Restarted at its first value, the penalty lets them move, but
  they end at 51.580 and 51.119, still above the plans before.

So the plans before are cheaper because their rounds took two cost-aware steps before the
projection, and the projection from there happened to hold every step; on 40 of the 42 far-point
paths of the lockfile it did not, and those plans stayed at 27 to 29 steps. The five plans' costs
were recomputed in 40 digits from the fitted field (drift `0.0462 + 0.6668 x`, channel `0.2719`)
by the same RK4 rollout, and agree with the library's to 3.1e-16; the percentages above SLSQP and
the actions' sums are read there too (`validation/barrier_far_point_costs.mac`).

## Decision

- **Under `outer="safeguarded"`, the projection at a join is kept, not taken.** Where next states
  join after a set of rounds, `_hold_barrier` projects the plan onto the columns now held
  (`_restore`, the projection it already makes where the rounds end short), and keeps the result
  as a candidate by the rule its other candidates follow: more certified leading steps, or every
  step and a lower cost. The rounds go on from the unprojected plan, step for step as before. The
  plan returned is the best candidate where the rounds do not end `converged`, as before; where
  it meets the condition at every step, its status reads `max_iterations`, as before.
- **It cannot shorten a plan's certified prefix.** A plan changes only where the projection is
  better than every candidate the rounds left before it: more leading steps, or all of them and
  cheaper. The two start states that the projection-first candidate made costlier keep their
  plans, since the projection there (51.501 and 51.474) is costlier than the plan the rounds and
  the final projection give.
- **A debug record at each join.** `chc_event="barrier_join"` carries the next states joined,
  the penalty, its growths and the leading steps the projection certifies, `None` where it holds
  no move. The far point could only be traced by patching the planner; the record says where a
  path went.
- **The default is untouched.** Under `outer="halved"` nothing is projected at a join and no
  record is written: its plans are the same, bit for bit. The projection at a join was not
  measured there, and the contract rule asks for that before a default moves.
- **The test asserts the rule, from the nine start states** of ADR 0156's test: wherever the
  next states join, the plan is certified on no fewer leading steps than the projection there.
  With the projection not kept, 8 of the 9 fail on this machine (29 steps where the projection
  held 30); with it, all 9 pass on the lockfile and on the floors.

## Measurements

**The 91 start states** (`1e-16` to `7e-8` either side of zero, 10 000 steps), each solved once
with the projection kept; the plan the rounds would have returned without it is read off the same
path, and matched the plan before the change on all 91 in both environments (status, certified
steps, cost within 1e-12):

| | lockfile (jax 0.11.2, Python 3.14) | floors (jax 0.4.30, Python 3.11) |
|---|---:|---:|
| more certified leading steps | 36 (27–29 → 30) | 30 (27–29 → 30) |
| the same plan | 55 | 61 |
| fewer certified steps, or costlier at the same | 0 | 0 |
| far-point paths short of 30 steps | 40 → 4 | 33 → 3 |
| statuses moved | 3 `no_progress` → `max_iterations` | 1 `no_progress` → `max_iterations` |
| cost of the 36 / 30 moved plans | 51.053 to 51.518 | 51.065 to 51.563 |
| cost of the plans already on 30 steps | 51.055 to 51.527 | 51.062 to 51.564 |

SLSQP's cost is 50.995133: the moved plans end 0.11 % to 1.1 % above it, the range of the plans
the rounds already held on every step. Before, they cost 50.19 to 50.88, below SLSQP's, being held
on fewer steps. Every plan left short of 30 steps, 7 on the lockfile (4 of them far points) and 9
on the floors (3), is one where the projection at the join holds no move.

**The library's planning tests.** Every plan they hold at a barrier, 293 (276 under the default
rounds, 17 under `"safeguarded"`), solved under `outer="safeguarded"` as before, with the
projection kept, and with the projection first, on the lockfile; the next states joined after a
set of rounds on 154 of them. Read as before against the default rounds, the 293 give ADR 0156's
tally exactly.

| | the projection kept | the projection first |
|---|---:|---:|
| the same plan | 268 | 187 |
| within 1e-9 relative | 0 | 8 |
| more certified leading steps | 12, each 29 → 30 | 12, each 29 → 30 |
| cheaper, every step certified | 13, by 0.07 % to 4.3 % | 30 |
| costlier, every step certified | 0 | 56: 20 `converged` both ways, by up to 1.3 %; 36 others, by up to 3.8 % |
| fewer certified leading steps | 0 | 0 |
| statuses moved | none | 11 `max_iterations` → `converged`, 1 `converged` → `max_iterations`, 1 `max_iterations` → `no_progress` |
| descent steps | 1 673 210, the same | 1 597 823 |

The 12 longer prefixes are ten plans of the unstable drift's log 2 that its tests hold under
`"safeguarded"` (eight start states, and both budgets of the budget test, whose rounds stop at the
ceiling with the same plan) and two of the twenty logs of the misspecification tests, one of them
log 2 again; each goes from 29 steps to 30 at a cost 0.39 % to 1.4 % higher. Held on one more step,
the eleven on log 2 end 0.17 % to 1.03 % above SLSQP's cost, where they ended 0.26 % to 0.45 %
below it. The 13 cheaper plans are of the misspecification tests: the integrator under either
radius (49.107 → 48.962, four plans), the unstable plant under either radius (four plans, by
0.07 % and 2.9 %), and five of the twenty logs, by 0.49 % to 4.3 %.

**The default rounds, bit for bit.** The library with this change, run over the same test files
as each test asks, passes all 472 of their tests, and holds the same 293 plans: the 276 under the
default rounds are the same actions with the same status as before, and the 17 under
`"safeguarded"` are the plans measured above with the projection kept.

## Consequences

- Under `outer="safeguarded"`, a plan whose rounds joined next states after a set of rounds can be
  the projection made at that join: certified on more leading steps than any other candidate, or on
  every step and cheaper. Of the 293 plans of the planning tests, 25 move so, none the other way.
- The rounds' path does not change: the same rounds and the same descent steps on every plan
  measured. The far point is still reached on that path; what changes is that the plan before the
  step to it is kept. A status moves only where the kept projection is returned and holds every
  step, and then reads `max_iterations`, as any such candidate does: 3 and 1 of the 91 start states
  moved from `no_progress`, none of the 293 plans. Where the projection at the join holds no move,
  the plan stays as short as before: 4 of the 91 start states on the lockfile and 3 on the floors.
  That remains open.
- Each join under the safeguarded rule costs one projection, the one the rounds already make where
  they end short; no descent step is added.
- The `outer` docstring's tally of the safeguarded rule against the default rounds moves: of the 293
  it now moves 148 (138 before), 46 to a cheaper plan at the same certified steps (40), 67 to a
  costlier one (73), 7 within 1e-9, 13 to more certified leading steps (2) and 15 to fewer (16),
  two of those still from 12 to 8; the statuses move as ADR 0156 left them. It stays opt-in: it
  still certifies 15 plans on fewer steps than the halved rounds.
- A new debug record, `chc_event="barrier_join"`, under the safeguarded rule only.
- `tests/test_channel_sensitivity.py::test_safeguarded_rounds_keep_the_projection_where_next_states_join`
  asserts the rule from the nine start states. With the projection not kept, 8 of the 9 fail on
  this machine.

## Alternatives

- **The projection first at a join** (ADR 0156's candidate). Rejected: on the two start states
  above it moves plans certified on all 30 steps 0.87 % and 0.76 % further from SLSQP's cost.
  On the planning tests it makes 56 plans certified on every step costlier, 20 of them
  `converged` both ways, by up to 1.3 %, and 36 others by up to 3.8 % (table above).
- **The projection first, the rounds forced on from it.** On `-3e-12` the projected plan clears
  every column, and the loop runs no round from it; forced, the rounds at the ceiling move its cost
  from 51.50122 to 51.49932 in 4 339 steps. Rejected, as above.
- **The projection first, the penalty restarted at its first value.** The rounds then move, to
  51.580 and 51.119 on the two start states, still above the plans before (51.055, 51.062).
  Rejected.
- **The projection first only where the joined shortfall exceeds a multiple of the tolerance.**
  On the lockfile the joined shortfall reads 48.1 to 53.2 on the far-point paths and 48.0 to 53.3
  on the paths whose first rounds ran until the budget was spent; the two start states above read
  49.7 and 48.8. No threshold separates them: it projects on every path or on none.
- **The better of two whole continuations**, with and without the projection first. It passes
  the rule by construction too, and on the moved plans it would keep the rounds the projection-
  first path runs after the projection; on the four moved start states measured those lowered
  the cost by at most 0.07 % (51.0655 to 51.0313), for up to 5 543 more descent steps. Rejected:
  the rounds after a join run twice, for less than a tenth of a percent.
