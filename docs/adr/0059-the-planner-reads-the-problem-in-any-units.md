# ADR 0059 — The planner reads a problem the same in any units

**Status:** accepted, 2026-10-07. Amended 2026-10-07 by ADR 0073: `pessimistic_solve` and
`pessimistic_control` descend in the scaled variables too, their scale raised by the secant of the
penalised cost's curvature as they go, and every penalised descent checks the projection's error;
the barrier's rounds keep their units. `lbfgs_box_control` minimises in the planner's scaled
variables. Amended 2026-10-08: the line search asks a step for a third of the fall its gradient
predicts, where in 0.15.0 it took any fall that beat `tol`; from the zero guess
`SupportShiftTask`'s oracle had settled 0.459 above the plan 0.14.4 reached (Consequences).

## Context

`projected_gradient_solve` and `projected_gradient_control` stepped `u ← P(u - η ∇J)` from
`η = lr0 = 0.2`, and halved `η` until the cost fell by more than `tol = 1e-9`. Both numbers carry
units. `η` is a step in action units squared per unit of cost: with a lever's numbers `s` times
larger (its channel over `s`, `R` over `s²`, its box times `s`), its gradient is `s` times smaller
and the step it needs `s²` times longer. `tol` is a cost: with the cost `c` times larger, a fall of
`1e-9` is `c` times easier to beat. So one problem was planned as many ways as it had units.

Through `causal_plan`, on `test_plan`'s problem (one lever, 12 steps, a box of ±5), whose plan costs
2.902342; on a two-lever variant, whose plan costs 2.559176; on that variant under rows that tie
the levers at every step, 2.767507; and on a budget of 35.5 of 100 steps' spend towards a revenue
out of reach:

| units | before |
|---|---|
| lever × 1 | converged after 615 steps |
| lever × 1e-6, × 1e-3 | converged after 17 and 23 steps, 9.1e-5 and 8.4e-5 of the box from the plan |
| lever × 1e3 | ran out of its 10 000 steps at a cost of 4.4227, 0.27 of the box away |
| lever × 1e6 | took no step |
| cost × 1e-9, × 1e-6 | took no step |
| cost × 1e-3 | ran out of steps at 3.4113 |
| cost × 1e3, × 1e6 | converged after 27 and 40 steps, 9.8e-5 of the box away |
| two levers × 1e-3 and × 1e3 | converged after 23 steps with the second lever moved by 3.2e-11, at 2.9023 |
| two levers × 1e3 and × 1e-3 | ran out of steps at 3.5740 |
| under the rows, × 1e-3 and × 1e3 | converged after 124 steps at 3.1112 |
| under the rows, × 1e3 and × 1e-3 | ran out of steps at 3.5740 |
| the budget, lever × 1e-3 or cost × 1e6 | converged after one step, having spent 68 |
| the budget, lever × 1e3 or cost × 1e-6 | ran out of steps, 0.62 of the box away |

A lever the cost does not feel at the guess (`x1' = x2 u1`, with `x2` zero at the start) never
moved in units 1e3 and 1e6 times its own, at a cost of 8.61 against 3.50.

The budget rows are a second fault, in the projection onto linear rows, that the first exposed: a
trial far outside the box came back from the projection over budget, and the line search took it.
`lbfgs_box_control` has the first fault too (Consequences).

## Decision

- **Each action gets a scale, and the descent runs in scaled variables** `v_kj = σ_kj u_kj`, lever
  `j` at step `k`, with `σ_kj² = c_kj / |J(ū)|`. `ū` is the guess clipped to the box, and `c_kj`
  is the cost's Gauss–Newton curvature along that one action, `R_jj + b_kjᵀ P_{k+1} b_kj`: `b_kj`
  is the action's column of step `k`'s Jacobian, and `P` weighs the states after the step,
  `P_H = Qf` and `P_k = Q + A_kᵀ P_{k+1} A_k`. One pass back over the steps' Jacobians, each the
  Jacobian of one RK4 step at the rollout's state and time, gives every `c_kj`. They are read in
  reverse mode, as the adjoint reads them: a plant whose step has a reverse rule only, a
  `jax.custom_vjp` such as `chc.games.fixed_point`, has no forward one, and in forward mode it
  raised `TypeError` where the old descent planned it. A lever in units `s` times its own moves
  its `c_kj` by `1/s²`, and a cost `c` times its own moves `c_kj` and `|J(ū)|` alike. So `v` is a
  pure number, and the descent in `v` is one iteration in any units.
- **The curvature is Gauss–Newton's, not the Hessian's.** With `Q` and `Qf` positive semidefinite
  it is `R_jj` plus a square, so a non-convex plant never reads as a negative step, and the action
  cost's own diagonal is its floor. On a plant affine in the state and the action it is the
  Hessian's diagonal, and a unit step in `v` is the Newton step along each action alone.
- **Where the curvature gives no size, the box does, then 1.** Where `c_kj / |J(ū)|` is not a
  positive normal number -- an action the cost neither charges for nor feels at `ū` at first order,
  or a rollout that left the float range -- `σ_kj` is one over the action's box side, so that a
  unit step in `v` crosses the box. The box moves with the lever's units as the curvature does.
  Where a side is infinite `σ_kj` is 1, which does not: nothing left in the problem carries the
  lever's scale. The action cost's diagonal needs no rung of its own, since it is inside `c_kj`.
  The floor is the least normal number: XLA on the CPU flushes a subnormal to zero, and a backend
  that kept one would lose the actions' digits in `σ u`.
- **The box, the rows and the projection are carried into `v`.** The box is multiplied by `σ`, a
  row's coefficient on an action divided by that action's `σ`, and Dykstra's projection is
  Euclidean in `v`; the box stays a clip. A step scaled action by action but projected onto the
  rows in the caller's units can stop at points that are not stationary; projected in `v` it
  stops at the problem's KKT points, which are the same in either variables. Rows share a
  projection class only where no action enters both (`_constraint_blocks(disjoint=True)`), which
  keeps them orthogonal under any scaling of the actions.
- **The iterate stays in the caller's units.** A trial is read back as `v / σ`, with each action
  that `v` puts on a side of its box set to the side itself, since `v / σ` can round off it. A plan
  on its box lies on it exactly, and a solve that takes no step returns its start bit for bit.
- **`lr0 = 1`, and `tol` is relative, `1e-14` by default.** The line search starts at the Newton
  step along each action alone, in `v`, and halves: since the amendment, until a step buys a third
  of the fall its gradient predicts, or else to the longest step that counts (Consequences). A step
  counts where it lowers the cost by more than `tol · |J(ū)|`. Where `J(ū)` is zero any strict
  decrease counts, and where it is not finite any finite value does.
- **The projection finishes a trial far outside the box.** A scaled step from a guess far from its
  plan can land many boxes out. Dykstra's row multipliers grow towards such a point by a sliver of
  the box per sweep, and the dual active-set polish that finishes a projection (ADR 0001) moves one
  constraint per step and gave up after 16, so the projection ran out its sweeps over budget. The
  polish may now take `max(16, 2n + p)` steps for `n` actions and `p` rows: a coordinate that the
  projection takes from one side of its box to the other needs a step to leave the first and one to
  be caught at the second. On the budget's first scaled trial, from cold duals, the polish settles
  after 121 to 140 steps; `n + p = 101` is short. `chc.support`'s penalised descent projects
  through the same polish.
- **A trial the projection cannot tell from the plan is not a step.** Dykstra holds each side and
  row to its tolerance, `256 ε (1 + max|y|)` for a trial `y`, and that pins a point only to the
  tolerance over the sine of the angle at which the constraints pinning it meet, and scaling the
  actions can narrow that angle. `prescribe` with `max_levers = 1`, on a log that kept `u2 = 3 u1`,
  pins each candidate's plan to zero by that relation and the other lever's box; in `v` the
  relation's weights are 1.03 and -0.046, where in the caller's units they are 0.95 and -0.32. A
  trial from zero came back with the free lever 1.1e-12 off it, ten times the tolerance, and 2.0e-15
  cheaper, 9.6e-14 of the cost, which `tol = 1e-14` counts. The plan then broke the relation the log
  kept and read `not_estimable` in nearly every unit; the old descent did the same with the cost 1e5
  times its own or more. So the line search now ends, with no step, at a trial within the tolerance
  of a move only across the sides and rows held at both its ends. A move along a held constraint, or
  onto or off one, is a step. No shorter trial moves further, since a projected step lengthens with
  the step (Calamai & Moré 1987). The check is one linear solve in the number of rows per trial, and
  on the 27 problems and the units below it changes no bit.
- **A barrier that binds is held from the caller's units.** `causal_plan` holds a barrier by rounds
  of `chc.support`'s penalised descent, which keeps its units (Consequences). Started from the
  scaled plan on `scripts/pendulum_demo.py`'s pendulum, the rounds held the barrier at a task cost
  of 0.632; started from the descent in the caller's units, as before, at 0.435. So where the
  barrier binds at the scaled plan, the rounds start from the guess descended in the caller's
  units, with the old `lr0 = 0.2` and `tol = 1e-9`.
- **The scale is computed once per solve, inside the compiled program and from arrays,** so a call
  compiles no more programs than before.

## Consequences

- **Every row of the table above converges, to one plan.** `tests/test_plan_units.py` holds each to
  `1e-6` of the box.

  | units | after |
  |---|---|
  | lever × 1, and × 1e-6 to × 1e6 | converged after 41 steps, to 1.7e-16 of the box |
  | cost × 1e-9 to × 1e6 | converged after 41 steps, to 1.1e-16 of the box, its cost to the bit |
  | two levers × 1e-3 and × 1e3 either way, × 1e-6 and × 1e6 | converged after 43 steps, to 4.1e-8 of the box |
  | under the rows, the same units | converged after 44 steps, to 9.4e-17 of the box |
  | the budget, lever × 1e-3 to × 1e3, cost × 1e-6 to × 1e6 | converged after one step, to 5.7e-11 of the box |
  | the lever the cost does not feel at the guess, × 1e-6 to × 1e6 | converged after 44 steps, to the bit |

  The two-lever plans differ by rounding in `σ`, grown over the steps: they stop nearer each other
  than to where the descent goes with `tol = 0`, 8e-8 to 1.2e-7 of the box.
- **Fewer steps, to a lower cost.** On 27 problems that mirror the planner's tests, against the
  least cost any measured run reached, the descent with `tol = 0` among them:

  | problem | steps before | after | cost above the least, before | after |
  |---|---:|---:|---:|---:|
  | one lever, 12 steps | 615 | 41 | 1.6e-8 | 0 |
  | two levers, 20 steps, box ±5 | 5 574 | 4 726 | 4.7e-7 | 5.4e-11 |
  | two levers, box ±1 | 3 690 | 1 158 | 1.5e-7 | 1.0e-11 |
  | eight residual backends | 450–610 | 18–174 | 4.8e-9–1.5e-8 | 0 |
  | a KAN residual | 566 | 65 | 1.2e-8 | 1.3e-5 |
  | `nlp_solver_certificate`'s instances, `R` = 1e-3 / 1e-2 / 1e-1 | 18 690 / 2 609 / 341 | 10 644 / 1 462 / 181 | 1.2e-6 / 1.1e-7 / 5.6e-9 | 3.6e-11 / 4.1e-12 / 0 |
  | boxed plants and a hybrid, 4 boxes | 7–566 | 1–30 | 0–3.3e-9 | 0 |
  | a loud learned residual, box ±6 | 12 355 | 29 927 | 5.1e-7 | 3.1e-10 |
  | revenue, Hill / Michaelis–Menten, under a budget | 18 / 166 | 11 / 16 | 0 / 0 | 0 / 0 |
  | two levers under a rate limit / a budget that never binds | 677 / 5 574 | 43 / 4 726 | 1.9e-8 / 4.7e-7 | 9.4e-12 / 5.4e-11 |
  | the golden rule, `dt` = 1 / 0.5 | 459 / 476 | 147 / 184 | 5.5e-10 / 3.2e-10 | 2.1e-12 / 0 |
  | a harvest, one square of the final state | 4 885 | 18 081 | 4.6e-10 | 4.2e-11 |

  Fewer steps on 25 of the 27, and a lower cost on 21. The costs are relative. The KAN residual
  now stops at another stationary point, 1.3e-5 above the one the old descent found. The loud
  residual and the ill-conditioned instance need more than the default cap of 10 000 either way,
  and the harvest now does too. With the amended line search the ill-conditioned instance converges
  after 8 083 (the consequence on the first step, below).
- **The warm start still saves a share.** On ADR 0003's oscillator loops (horizon 20, 40 steps a
  replan), reproduced before the change, the warm-started controller took 65 317 and 166 115
  descent steps against cold solves' 100 084 and 260 012, under the velocity floors 0.8 and 0.3:
  35 % and 36 % fewer. Now it takes 69 791 and 166 601 against 100 261 and 257 810: 30 % and 35 %
  fewer, its realised cost within 7.7e-7 and 5.2e-8 of the cold loop's. With the amended line
  search it takes 67 097 and 165 909 against 91 390 and 255 980: 27 % and 35 % fewer, within 7.8e-7
  and 1.6e-7.
- **`mpc_control`'s 40 inner steps come nearer the converged loop.** Over 25 replans of a damped
  oscillator they come within 0.001 % of the closed-loop cost of the same loop run to convergence,
  on the known plant and with an MLP residual. Before, they were 1.9 % and 1.2 % above it.
- **The defaults changed meaning.** `lr0` was 0.2 action units per unit of gradient; it is now 1
  Newton step, and no value of it reproduces the old step. `tol` was a fall of `1e-9` in the
  cost's own units; it is now `1e-14` of `|J(ū)|`. A fall `ε` in the cost's own units is
  `tol = ε / |J(ū)|`.
- **`nlp_solver_certificate` defaults to `pg_steps = 50`, not 150.** It exists to show a truncated
  descent's gap on every instance, and its well-conditioned instance now stops on its own rule
  after 181 steps, so at 150 the gap closed and the certificate failed. At 50 the least ratio of
  the descent's stationarity to L-BFGS-B's is 618, where at 150 before it was 242.
- **A constant added to the cost moves the stopping rule**, since `|J(ū)|` moves with it; so does
  the guess. A cost mostly out of a plan's reach stops further from its optimum in what a plan can
  change. The golden-rule oracle is that case: its guess costs 26.5 and 29.8 times its plan. There
  the default sets the plan's mid-horizon sales 8.9e-8 and 7.8e-9 from the golden rule, which the
  test holds to `1e-6`.
- **Actions that act together still take many steps.** The scale is a diagonal. The
  ill-conditioned oscillator needs 10 644 steps, the loud residual 29 927 where it took 12 355,
  and the harvest 18 081 where it took 4 885; with the amended line search, 8 083, 25 761 and
  16 281.
- **A first step read where the plant is linear carried an action past a collapse, until the line
  search asked for sufficient decrease** (amended 2026-10-08). The scale is read at the guess, so
  the first step is the Newton step of the plant as it acts there, along each action alone.
  `SupportShiftTask`'s 25 actions act together: on its linear model, from the zero guess, that step
  is 17 times the step to the cost's minimum along it. Its plant acts through
  `u exp(-(u / 0.8)^2)`, linear at 0 and spent beyond about 2. Halved twice, the first step set
  actions up to 6.2 and still lowered the cost, by 0.4 % of the fall its gradient predicted, and
  the line search, which took any fall that beat `tol`, took it. The descent settled at another
  stationary point: the last five actions near -2.2, where the plant no longer answers them, at a
  true cost of 21.104508; in float32, four, at 21.087929. The descent in the caller's units reached
  20.645, every action at or inside the sweet spot, ±0.565. The task's oracle is that plan, so the
  task measured every regret against a plan 0.459 worse, 0.443 in float32.

  The line search now asks a trial for a third of the fall its gradient predicts for the move,
  `g · (v - v')` to the projected trial `v'`: Armijo's sufficient decrease (Armijo 1966) along the
  projection arc (Bertsekas 1976). It reads a fall against a fall, so it is the same in any units.
  On a quadratic, a step `s` times the one to the minimum along it buys `1 - s/2` of the
  prediction. A first trial with `s ≤ 4/3` is taken whole, the Newton step's `s = 1` among them; a
  longer one is halved into `(2/3, 4/3]`, which leaves at most 1/9 of the fall the line offers. A
  fraction `c` in place of a third leaves `max(c², (1 - 2c)²)`, least at `c = 1/3`. Where no
  halving buys a third, across a kink, the search takes the longest trial that counts, the one it
  took before: at a plan of zeros the confounding radius' smoothed norm holds a trial's share at
  0.034 however short the step. So the descent stops where it stopped before, and `converged`
  means what it meant. The planner, the penalised descent (ADR 0073) and the barrier's rounds share
  the line search, and with it the rule.

  The benchmark tasks' plans, each from the zero guess, at the true cost the task reads (`*`: out
  of the 10 000 steps):

  | plan | steps, any fall | a third | true cost, any fall | a third |
  |---|---:|---:|---:|---:|
  | `SupportShiftTask`'s oracle | 510 | 69 | 21.104508 | 20.645063 |
  | its pessimistic plan | 15 | 14 | 22.888169 | 22.888169 |
  | its greedy plan, on its linear model | 6 962 | 2 845 | 28.005507 | 28.005477 |
  | `ConfoundingRobustTask`'s oracle, the same linear plan | 6 962 | 2 845 | 8.306852 | 8.306852 |
  | its robust plan, at a regret of 3.470570 either way | 1 202 | 1 268 | 11.777422 | 11.777422 |
  | its greedy plan | 1 632 | 1 451 | 14.258292 | 14.258295 |
  | `ModelUncertaintyTask`'s oracle, on its cubic-drag plant | 10 000* | 50 | 14.675420 | 15.520831 |
  | its calibrated plan | 61 | 56 | 16.176229 | 16.176229 |
  | its greedy plan | 7 196 | 6 782 | 1 363.999465 | 1 363.998762 |

  `SupportShiftTask` reads the pessimistic plan's regret as 2.243, where it read 1.784, and
  greedy's as 7.360, where it read 6.901. From seven more guesses, constant at -0.3, 0.3 and 0.6, a
  ramp from -0.5 to 0.5 and three draws of noise of scale 0.3, the bump plant's descent ends at the
  sweet spot from 3 of them, where any fall reached it from 1 and 0.14.4 from 5. The rule keeps
  the descent in the basin its guess reads; it does not search for a lower one. The cubic-drag
  plant's effect, `u - 0.15 u³`, peaks at 1.49 and turns over past 2.58. There any fall's long
  steps reached plans that use the turn, at 14.675 and 14.884, lower than 0.14.4's, from 6 of the
  8 guesses, 2 of them out of steps; with the rule each guess ends at 0.14.4's plan, 15.520831, at
  the peak, in 38 to 57 steps. From every guess on the bump plant the rule ends at or below where
  any fall ended; on the cubic drag, at or above. A plan with an action on the box's edge, where the
  effect is -68.8, costs 4.797: that task's oracle is a local plan under either rule.

  On the 27 problems above, against the least cost any measured run reached:

  | problem | steps, any fall | a third | cost above the least, any fall | a third |
  |---|---:|---:|---:|---:|
  | one lever, 12 steps | 41 | 36 | 6.6e-15 | 0 |
  | two levers, 20 steps, box ±5 | 4 726 | 3 801 | 5.3e-11 | 6.2e-11 |
  | two levers, box ±1 | 1 158 | 1 008 | 1.0e-11 | 9.4e-12 |
  | eight residual backends | 18–174 | 17–77 | ≤ 6.3e-13 | ≤ 6.3e-13 |
  | a KAN residual | 65 | 62 | 1.3e-5 | 1.3e-5 |
  | `nlp_solver_certificate`'s instances, `R` = 1e-3 / 1e-2 / 1e-1 | 10 644 / 1 462 / 181 | 8 083 / 1 360 / 180 | 3.5e-11 / 4.1e-12 / 1.2e-12 | 3.7e-11 / 4.3e-12 / 1.2e-12 |
  | boxed plants and a hybrid, 4 boxes | 1–30 | 1–24 | ≤ 2.3e-12 | ≤ 2.3e-12 |
  | a loud learned residual, box ±6 | 29 927 | 25 761 | 3.1e-10 | 3.0e-10 |
  | revenue, Hill / Michaelis–Menten, under a budget | 11 / 16 | 8 / 12 | 4.6e-13 / 8.0e-15 | 4.7e-13 / 0 |
  | two levers under a rate limit / a budget that never binds | 43 / 4 726 | 31 / 3 801 | 7.9e-12 / 5.3e-11 | 8.6e-12 / 6.2e-11 |
  | the golden rule, `dt` = 1 / 0.5 | 147 / 184 | 149 / 127 | 1.5e-12 / 0 | 1.5e-12 / 2.1e-12 |
  | a harvest, one square of the final state | 18 081 | 16 281 | 4.2e-11 | 4.4e-11 |

  61 030 steps against 71 892: fewer on 22, as many on 4, and more on one, the golden rule at
  `dt` = 1, 149 against 147. Every cost is within 9.6e-12 of the one any fall reached. The searches
  that found a step read the cost 497 601 times against 582 341, and one of them fell back to its
  longest falling trial, on the two levers under a rate limit. The units table above now reads 36,
  36, 35, 17, 1 and 34 steps, row by row, and the two-lever plans stop 9.7e-17 of the box apart,
  where they stopped 4.1e-8 apart. `nlp_solver_certificate`'s least ratio of the descent's
  stationarity to L-BFGS-B's, at its 50 steps, is 147, where it was 214.

  The penalised descent pays in steps. On its 54 solves (ADR 0073) it takes 52 665 against 41 736,
  fewer on 27 and more on 22, and the searches that found a step read `F` 71 604 times against
  50 847. Three solves make the rise. The loud residual under the support penalty takes 9 232 steps
  against 5 858 and ends at the stationary point the descent before ADR 0073 found, 1.3e-4 above
  the other. The ill-conditioned instance under the radius takes 7 368 against 4 077, to 1.4e-10 of
  the least `F` where it came to 2.5e-11. The harvest under the radius runs out of its 10 000 steps
  at the least `F` any run reached, where it stopped after 5 087, 2.0e-8 above it. The other 51
  take 26 065 steps against 26 714. Against the descent before ADR 0073, 218 089 steps, the 54 take
  under a quarter. In all 54 the search fell back to its longest falling trial once, on the boxed
  hybrid under the radius.
- **A truncated plan's regret bound is looser against its regret.** On `test_plan`'s boxed problem,
  three steps leave the plan 0.29 above its optimum, where they left the old descent's 5.46, and
  `plan_regret_bound` reads 1.02, where it read 9.57: 3.5 times the regret, where it was 1.75
  times. The test that holds the bound to twice the regret takes its truncated plans from the
  descent in the caller's units.
- **Holding a binding barrier spends a scaled solve first.** The plan reports the steps of the
  solve in the caller's units that the rounds start from, not the scaled solve's.
- **The penalised descent keeps its units.** `pessimistic_solve`, `pessimistic_control` and the
  barrier rounds call the same line search with `lr0 = 0.2` and `tol = 1e-9` in the caller's
  units, and read a problem in other units differently, as the planner did. The same change applies
  there, with the penalty's curvature added to `c_kj`, and so does the check on the projection's
  error, which its line search skips; both are left to a change of their own.
- **`lbfgs_box_control` keeps its units.** On the one-lever problem it takes no step with the lever
  × 1e-6 or × 1e6, or the cost × 1e-6, and stops 6.0e-3 of the box from the plan with the lever ×
  1e3 or the cost × 1e-3, where in the problem's own units it ends 3.1e-6 away. L-BFGS-B's `pgtol`
  is a gradient in the caller's units, and its `ftol` floors `|f|` at 1.
- **ADR 0001 to 0004 counted the descent in the caller's units.** Their step counts are that
  descent's. `scripts/bench_linear_constraints.py instances` counts it in a copy that must match
  the shipped solver, so that part of the script now stops on its guard.
- **Each solve linearises the rollout once before its first step:** `H` Jacobians of one RK4 step,
  `n` cotangents each, held as `H (n² + n m)` numbers.

## Alternatives considered

Measured on the 27 problems above, at `tol = 1e-14`, against one scale per action.

- **One scale per lever**, from the curvature along the lever moved by one unit at every step,
  over the horizon. It needs `m` tangents through the rollout rather than `H` step Jacobians. But
  it took more steps on 20 of the 27 (the golden rule: 982 where one scale per action takes 147;
  two levers under a rate limit: 773 against 43), fewer on 3 (the harvest: 13 285 against
  18 081), and as many on 4.
- **The exact Hessian's diagonal per action**, one Hessian-vector product per action, `H m` of
  them. More steps on 9 (the golden rule: 1 160 against 147; the Michaelis–Menten revenue: 229
  against 16), fewer on 8 (the port-Hamiltonian backend: 69 against 174), as many on 10. On a
  non-convex plant it can be zero or negative, which hands the action to the box's scale, and a
  Hessian-vector product differentiates the plant in forward mode, which a reverse-only step lacks.
- **The box alone**, `σ` one over the box side. Invariant, and it needs no derivative. But it is
  blind to what the cost feels: more steps on 20 (the golden rule: 22 617 and 42 469 against 147
  and 184; the harvest ran out of 200 000), fewer on 7 (the two-lever problem: 2 693 against
  4 726), and no scale for an unbounded lever.
- **L-BFGS-B as the workhorse.** It crosses the Python boundary on every iteration, so it cannot be
  compiled, and it has the same fault (Consequences).
- **`tol` relative to the current cost, or to the first step's fall.** The first moves the
  threshold as the cost falls; the second shrinks it with how good a warm start is. Neither lets a
  caller convert a fall in the cost's own units before the solve.
- **Stopping on the stationarity residual.** That changes what `converged` means. The residual is
  returned beside the status already.
- **A default `tol` of `1e-12` or `1e-13`.** At `1e-12` the 27 problems take 40 942 steps against
  71 892, but the golden-rule oracle's mid-horizon sales stop up to 9.8e-7 from the golden rule,
  where its test allows `1e-6`; at `1e-13`, up to 3.3e-7. With `tol = 0` they stop 2.7e-8 and 2.8e-8
  away.
- **Holding a binding barrier from the scaled plan**: 0.632 against 0.435 on the pendulum. Rounds
  scaled lever by lever, tried while the scale was one per lever, held it at 0.441 after 10 119
  steps. The rounds belong to `chc.support`.
- **A check on the size of the move alone**, `max |Δv|` within the tolerance. It misses the narrow
  angle: the trial above moved 1.1e-12 against a tolerance of 1.06e-13, and its candidate won the
  selection by the 2.0e-15 it bought.
- **A ridge sized to each row's free part in the polish.** The polish's ridge is sized to the whole
  row, so where the box pins most of a row it biases the free actions by `256 ε |row|² / |free|²`
  of the step, 2.8e-11 above. Sized to the free part, the polish would be exact to rounding on a
  well-posed set. But the narrow angle amplifies any error the projection makes, so this would only
  shrink what the check catches, and the polish is shared with `chc.support`.
- **A larger sweep cap for the projection.** With the old polish, five times the cap took the
  budget's overspend from 98 to 95 of 35.5: the sweeps a far trial needs grow with how far out it
  lands. **Refusing a trial whose projection did not settle** would need `_dykstra` to report it,
  a change to a function `chc.support` shares.

For the line search's rule (amended 2026-10-08), measured on the 27 problems, the penalised
descent's 54 solves (ADR 0073), and the bump and cubic-drag plants from zero and from the eight
guesses:

- **Armijo's rule at another fraction.** At the textbooks' `1e-4` (Nocedal & Wright 2006) the
  bump's oracle takes the step at issue, which buys 0.004, and ends at 21.104508; at `1e-2` it ends
  at 20.746, with actions out to 2.12. Each fraction measured from 0.1 to 0.5 ends at 20.645063.
  At 0.1 the 27 take 68 669 steps, the linear model 6 947 and the 54 penalised solves 46 507; from
  the eight guesses the bump's descent reaches the sweet spot from 2 and the cubic drag's 0.14.4's
  plan from 6, and from zero the drag's oracle ends at 14.884. At 0.25 the 27 take 61 822 steps
  and the linear model 6 157, both without the fallback. At 0.4 the 27 take 57 245, the 54
  47 410, every `F` within 3.2e-8 of the least, and the linear model 2 777 without the fallback,
  and each of the eight guesses ends where it ends at a third: fewer steps than a third on every
  count. The 27's count falls as the fraction rises, and the linear model's up to a half; the 54's
  moves with the paths three solves take, 46 507 at 0.1 and 52 665 at a third. Past a half a
  quadratic's Newton step, which buys a half, is refused. A third is kept for the worst case, not
  for these counts: on a line of a quadratic it leaves at most 1/9 of the fall, where 0.4 leaves
  0.16, and it stays clear of the half a nearly quadratic cost's Newton step buys.
  `test_plan_units`' two levers buy 0.41 at the Newton step, which 0.4 would take by 0.01.
- **Armijo's rule with no fallback.** Where no halving buys a third it takes no step: on the boxed
  hybrid under the radius the descent takes none from its guess of zeros, at the kink, and stops
  6.1e-3 above the least; the 54 take 52 582 steps.
- **A ratio test against the scaled diagonal model**, the test a trust region reads its step by
  (Moré 1983; Conn, Gould & Toint 2000): the fall over `L - scale |Δv|²/2`, `L` the gradient's
  prediction. Along the gradient, at `λ` Newton steps, the model's fall is `L (1 - λ/2)`, so the
  test is Armijo's rule with a fraction that shrinks to half its own as the step grows to the
  Newton step. At 0.25 the 27 take 61 634 steps and the bump's oracle ends at 20.645063; at 0.1 the
  cubic drag's ends at 14.884. It measured no better than Armijo's rule, and it adds a model.
- **A trust region on the first step only**, the step's quadratic term `scale |Δv|²` held to `r²`
  of the cost: at `r` = 0.3 and 0.1 the bump's oracle ends at 20.645063, and at 1 at 21.104508. It
  needs a radius the descent has no reading of, and it guards the first step alone, as the next
  does.
- **Armijo's rule on the first step only**: 69 782 steps on the 27 and 42 203 on the 54; the bump's
  oracle ends at 20.645063 after 106 steps, and from the eight guesses the bump's descent reaches
  the sweet spot from 3 and the cubic drag's 0.14.4's plan from 7. It keeps the long steps after
  the first: the 27 save 3 % of their steps, where the rule saves 15 %.
- **Armijo's rule in the planner's metric only**, the penalised descent's steps from a scale its
  secant raised taking any fall: the planner as with the rule, and 42 386 steps on the 54, every
  `F` within 1.3e-8 of the least. It is two rules for one search, and it takes a step from a raised
  scale on the rule this amendment replaces, though the secant reads the curvature along the last
  step, not along the next.
- **Re-reading the scale at the iterate**, so that each step is the Newton step along each action
  where the plan is: the bump's oracle ends at 21.326 after 8 850 steps, and each step linearises
  the rollout, `H` Jacobians.
- **Barzilai–Borwein steps** (Barzilai & Borwein 1988; Raydan 1997) under a non-monotone line
  search (Grippo, Lampariello & Lucidi 1986; Birgin, Martínez & Raydan 2000). Not measured. The
  first step has no secant to read, so it is the step at issue, and a non-monotone search asks a
  step for a small fraction of its prediction below the highest cost in a window, which the step at
  issue passes.
- **Wolfe's curvature condition** (Wolfe 1969) or **Goldstein's two-sided test** (Goldstein 1965).
  Not measured. Both refuse a step that is too short as well as one too long, which a halving from
  the Newton step does not need, and Wolfe's reads the gradient at each trial, an adjoint pass
  where Armijo's reads a rollout.
