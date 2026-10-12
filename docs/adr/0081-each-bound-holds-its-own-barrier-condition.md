# ADR 0081 — Each bound holds its own barrier condition

**Status:** accepted, 2026-10-12. Amends ADR 0002: a held barrier may be a tuple, each barrier with
its own condition and multipliers. Amends ADR 0005: a band's two margins no longer tie. Amends ADR
0073: the barrier rounds keep their units for what scaled rounds cost, not for a tie.

## Context

`prescribe` makes a margin of each finite bound of a `Constraint`, `x - lo` or `hi - x`. Since
0.6.0 it held, and its certificate audited, one barrier, their least `h = min_j m_j`, under the
condition `ḣ ≥ -α h` of ADR 0002 at each step. The audit read every margin tied at the minimum,
and the held solve read the first tied margin's gradient. That barrier had three defects.

**Its condition jumps where two margins tie.** On each side of a tie it reads that side's margin's
gradient. The penalised descent of the barrier rounds, run scaled, ended on such a tie on each of
four logs of `scripts/pendulum_demo.py`'s pendulum, whose barrier is `|ω| ≤ 1`. A move of `1e-12`
raised the shortfall there by 2.8 to 6.1, and the rounds took no step. On two of the logs the
barrier was then held for 2 and 0 of its 40 steps. ADR 0073 kept the rounds in the caller's units
for that reason.

**It asks nothing of a bound until that bound is the nearest.** A state may approach the far bound
at any speed while the near one is the least margin. On the pendulum, logs of seeds 0 to 3, the
certificate read all 40 steps held. On each log the upper speed bound's own condition failed on 4
or 5 of them, by up to 1.90 in the condition's units. On `test_decision`'s panel, steering `supply`
from 0 towards 1 inside `[-0.1, 1.5]`, the plan rises faster from its first step than `hi`'s own
condition allows. The priced certificate read 5 steps held and the held one 15, and `hi`'s own
condition failed at the first step of both.

**Which margin is the least depends on the states' units.** Two margins of different states compare
numbers in different units. On the same panel with `supply ≤ 0.8` and `wait` in `[-0.3, 0.3]`, held,
`wait` in units 1e3 times its own planned at a task cost of 4.612 and certified no step, where in
its own units the plan cost 5.409 and certified all 15.

## Decision

- **Each finite bound is a barrier of its own.** `prescribe` hands its margins to
  `BarrierConstraint` as a tuple. `causal_plan` holds each one's condition with a multiplier per
  step and barrier, and `certify_safety` audits each one. A step is certified where every
  barrier's condition holds. The condition reads each bound at every step: approach no bound
  faster than the class-K gain times the distance to it.
- **`BarrierConstraint` and `certify_safety` take a tuple of barriers.** One barrier, as before,
  plans and audits bit for bit as before. The shortfall and the multipliers of a tuple have a
  column per barrier, `(steps, barriers)`, and `RecedingHorizon` shifts them along the steps as it
  did. The log's `active_steps` counts a step where any barrier binds.
- **Each condition is scaled to the largest before the rounds read it.** The rounds' back-off,
  `1e-6` of the condition's largest term at the plan they start from, and their penalty are one
  for all the barriers of a tuple. Read in their own units, a barrier in units a millionth of
  another's got a back-off as large as its own terms. On the oscillator below the rounds then
  stopped short, that barrier held on 16 or 17 of 20 steps. So each barrier's condition is
  multiplied by the largest barrier's term over its own, both read at that plan. A barrier with
  no term at all keeps its own units. One barrier, not a tuple, is held as before.
- **The audit of a tuple reads as one certificate.** `planned_certified` holds where every
  barrier's does. `barrier_values` is the least barrier's. `guaranteed_derivative` and `required`
  are the pair of the barrier with the least room, so `guaranteed >= required` still reads the
  verdict. `step_gamma_star` is the least of the barriers' ceilings, an upper bound on the joint
  one, as ADR 0005 read a tie. `radius` is the largest.
- **The rounds keep their units**, now for their cost. With each bound's own condition, scaled
  rounds hold the pendulum's barrier for all 40 steps on all four logs. They take 83 834 to
  128 435 descent steps, most rounds running out of their 10 000; in the caller's units, 34 349
  to 34 398.

## Evidence

All in float64. The pendulum's adjusted reading, `run_pendulum(episodes=10, steps=60, seed=s)`.
"Own conditions" counts the steps where each bound's own condition holds, lower then upper.

| seed | before: task cost · steps · certified · own conditions | after | after, rounds scaled |
|---|---|---|---|
| 0 | 0.435363 · 32 674 · 40 · 40, 36 | 0.436006 · 34 398 · 40 · 40, 40 | 0.435992 · 91 677 · 40 |
| 1 | 0.457908 · 10 104 · 40 · 40, 35 | 0.436188 · 34 349 · 40 · 40, 40 | 0.436174 · 83 834 · 40 |
| 2 | 0.435418 · 32 674 · 40 · 40, 36 | 0.436059 · 34 383 · 40 · 40, 40 | 0.436045 · 102 136 · 40 |
| 3 | 0.457859 · 10 106 · 40 · 40, 35 | 0.436157 · 34 364 · 40 · 40, 40 | 0.436144 · 128 435 · 40 |

On two of the logs the plan now costs 0.0217 less. Before, the rounds stopped at a plan that
crossed the band's midpoint at 0.19, and failed the upper bound's condition by 0.14. The four
plans now stop within 1.9e-4 of each other.

`test_decision`'s panel, `supply` from 0 towards 1, `horizon = 15`:

| band | priced: certified before / after | held: task cost before / after | held: own conditions before / after |
|---|---|---|---|
| `[-0.1, 1.5]` | 5 / 0 | 2.0783 / 2.2144 | 15, 10 / 15, 15 |
| `[-0.2, 1.2]` | 4 / 0 | 2.1165 / 2.5437 | 15, 11 / 15, 15 |
| `[-0.05, 2.0]` | 15 / 15 | 2.0782 / 2.0782 | 15, 15 / 15, 15 |

The same panel with `supply ≤ hi` and a band on `wait`, `wait` in units `s` times its own, held:

| bounds | before: task cost · certified, `s` = 1 / 1e-3 / 1e3 | after |
|---|---|---|
| `supply ≤ 0.8`, `wait` in `[-0.3, 0.3]` | 5.4092 · 15 / 5.4211 · 15 / 4.6119 · 0 | 5.4092 · 15 in all three |
| `supply ≤ 1.2`, `wait` in `[-0.2, 0.2]` | 6.2291 · 15 / 6.2291 · 14 / 5.3666 · 15 | 6.2291 · 15 in all three |
| `supply ≤ 0.9`, `wait` in `[-0.5, 0.1]` | 3.6965 · 15 / 3.8444 · 15 / 3.4378 · 15 | 4.0732 · 15 in all three |

After, the three units give the same task cost to ten digits. Without the scaling of each
condition to the largest, the held plans read them by up to 0.9 % of the cost: 5.4258 and 4.1090
with `wait` in units 1e-3 times its own.

On the damped oscillator of `test_held_barrier`, the velocity floor `v ≥ -0.8` and a ceiling
`v ≤ 0.1` at `α = 2`: the velocity falls through the band's midpoint, and the exact optimum binds
the ceiling's condition on steps 16 to 19 as it brakes. On 16 and 17, and 18 at `gamma = 2`, the
floor is the nearer bound. The held plan lands
2.5e-6 above that optimum at `gamma = 1` and 4.0e-6 above at `gamma = 2`, relative, with every
step certified. With the ceiling in units 1e-6 times its own it takes as many steps, to the
same cost. The least of the margins would not have held the ceiling there. Without the scaling,
the ceiling in those units was held on 17 and 16 of the 20 steps, the rounds out of their steps.

## Consequences

- **A prescription certifies more, and its held plans can cost more.** The far bound's condition
  limits how fast a plan leaves the near one: on the panel above, 6.5 % and 20 % of the held task
  cost. Where the far bound has room, nothing moves: the third band's plans are unchanged.
- **A plan that the least margin certified can now certify fewer steps**, the priced plans above
  among them. They failed a bound's own condition; the certificate now says so.
- **A prescription no longer reads a constrained state's units**, priced or held: the three
  problems above certify the same steps in all three units, and hold the same plan. Before, the
  third certified 3, 3 and 0 steps priced. `gamma_star` still reads them, for another reason: the
  identification radius is one operator-norm ball on the effect, over states in their own units,
  so the third row's priced `gamma_star` reads 1.202, 1 and `inf`, before and after.
- **The rounds still read the largest barrier's units**, as they read one barrier's: on the
  oscillator, the ceiling in units 1e6 times its own, now the largest, is held on every step
  6.6e-3 and 9.9e-3 above the optimum, after 72 754 and 72 809 descent steps.
- **The rounds no longer need the descent in the caller's units to start from.** Started from the
  scaled plan, as ADR 0059 measured at 0.632 on the pendulum, they now hold the barrier at
  0.436002 to 0.436184 after 33 263 to 33 439 descent steps. The restart is left as it is: it
  moves every binding barrier's plan, one barrier or several, and is a decision of its own.
- **A single bound moves by rounding only.** `prescribe` hands it on as a tuple of one, and the
  held plan under `supply ≤ 0.5` on the tests' panel moves by at most 1.1e-16, with the same steps
  and certificate. `causal_plan` with one barrier, not a tuple, plans bit for bit as before.
- `chc.decision._Barrier` and `_certify`, the least of the margins and its audit, are removed.

## Alternatives

- **A smooth least**, `-log Σ exp(-κ m_j) / κ`. Its condition is continuous, and away from a tie it
  is near the least margin's. But `κ` is a rate per unit of the state, and across states in
  different units no one `κ` reads them alike, as the least does not. It also certifies a barrier
  the caller did not state, whose safe set is smaller than the bounds'.
- **Refuse a tie.** The midpoint of a two-sided bound is where a state at rest inside it starts, as
  the pendulum does: the refusal would refuse the common case.
- **The nonsmooth barrier's almost-active set** (Glotfelter, Cortés & Egerstedt 2017): hold the
  conditions of the margins within `ε` of the least. The condition then jumps where a margin
  enters the set, and `ε` reads the states' units as `κ` does.
- **Hold each bound's condition in the solve, and audit the least.** The solve would hold more than
  the certificate checks: ADR 0002 forms both through the same helpers so they cannot drift.

## References

- Ames, A. D., Xu, X., Grizzle, J. W. & Tabuada, P. (2017). Control barrier function based
  quadratic programs for safety critical systems. *IEEE Trans. Automatic Control* 62(8),
  3861–3876.
- Glotfelter, P., Cortés, J. & Egerstedt, M. (2017). Nonsmooth barrier functions with applications
  to multi-robot systems. *IEEE Control Systems Letters* 1(2), 310–315.
