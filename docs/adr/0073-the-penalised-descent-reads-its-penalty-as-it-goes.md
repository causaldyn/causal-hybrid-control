# ADR 0073 — The penalised descent reads its penalty as it goes

**Status:** accepted, 2026-10-07. Amends ADR 0059: `pessimistic_solve` and `pessimistic_control`
descend as the planner does, every penalised descent checks the projection's error, and
`lbfgs_box_control` minimises in the planner's variables.

## Context

ADR 0059 scaled the planner and left the penalised descent as it was. `pessimistic_solve`,
`pessimistic_control` and the barrier rounds inside `causal_plan` stepped `u ← P(u - η ∇F)` from
`η = 0.2`, and counted a step where it lowered `F`, the task cost plus the penalties, by more than
`1e-9`, both in the caller's units. Their line search also took a trial that the projection moved
off the iterate by its own error alone, which the planner's no longer does.

So one penalised problem was solved as many ways as it had units. On `test_plan`'s one-lever
problem under a support penalty, on a two-lever variant, and on that variant under rows that tie
the levers at every step, each against the same problem in its own units:

| units | before |
|---|---|
| lever × 1e-6, × 1e-3 | converged after 17 and 21 steps, 7.0e-6 and 7.4e-6 of the box away |
| lever × 1e3 | ran out of its 10 000 steps 5.8e-2 of the box away |
| lever × 1e6, cost × 1e-6 | took no step |
| cost × 1e-3 | ran out of steps 9.4e-3 of the box away |
| cost × 1e3, × 1e6 | converged after 33 and 36 steps, 5.6e-6 of the box away |
| two levers × 1e-3 and × 1e3, or × 1e-6 and × 1e6 | converged half the box away |
| two levers × 1e3 and × 1e-3 | converged 9.7e-2 of the box away |
| under the rows, × 1e-3 and × 1e3 | converged 0.33 of the box away |
| under the rows, × 1e3 and × 1e-3 | ran out of steps 0.17 of the box away |

Under the confounding radius, `ConfoundingRobustPenalty`, the one-lever problem ran out of its
steps in its own units; with the lever in units 1e-6 to 1e6 times its own, or the cost 1e-6 or 1e6
times, it stopped 1.9e-2 to 0.20 of the box from there, and with the lever 1e6 times or the cost
1e-6 times took no step at all.

`ConfoundingRobustTask`'s robust plan is that case. Its solve ran out of its 10 000 steps with the
stationarity residual at 0.19 and a regret of 3.937, and after 100 000 steps at 0.12 and 3.823. The
problem has a solution. Its model is linear and its cost quadratic, so with the norm unsmoothed it
is a program over the box in the actions' positive and negative parts, whose solution moves 9 of
the 25 actions off zero; from there Newton's method on the exact Hessian reaches the smoothed
problem's KKT point, its residual 2.3e-16, 4.3e-5 from the unsmoothed plan, at a regret of
3.470575. The robust row's regret read 0.662 of greedy's, where it is 0.583.

The step and the stopping rule in the caller's units were not the whole cause. The planner's scale
alone, read at the guess, did worse on that task: after 100 000 steps it stood 0.48 above the
optimal `F`, the old descent 0.28. The cause is the penalty's kink. The radius penalises
`r Σ_t ||u_t||`, the norm smoothed as `sqrt(||u_t||² + ε²)` with `ε` a millionth of the actions'
root mean square, 3.9e-6 at the plan. The plan holds 16 of its 25 actions at zero, where the
curvature of `F` along each is 6.7e2 to 4.5e6 times the cost's; along the other 9 it is the
cost's. Scaled by the cost's curvature, as the planner scales, the Hessian's condition number is
2.2e8; scaled by its own diagonal, 390. At the guess, all zero, `ε` is 1.5e-154, the square root
of the least normal number, so there the kink's curvature is the radius' weight over that. A scale
read once at the guess either misses the kink or reads the guess's.

## Decision

- **The penalised descent is the planner's**, `chc.control._descend`, shared rather than copied:
  steps in `v = σ u`; the box and the rows carried into `v` and projected on there, in classes
  that share no action; a line search from `lr0 = 1` that halves; a step counted where it lowers
  `F` by more than `tol |F(ū)|`, `tol = 1e-14` by default, `ū` being the guess clipped to the box;
  and no step from a trial that the projection moves by its own error alone.
- **Its scale is a floor, raised as the descent goes.** `σ_kj²` starts at the planner's,
  `c_kj / |F(ū)|`, the cost's Gauss–Newton curvature along that action alone over the penalised
  cost. After each accepted step `s`, with `y` the change in `∇F` over it,
  `σ_kj² = max(c_kj, y_kj / s_kj) / |F(ū)|` wherever the action moved and the secant `y_kj / s_kj`
  is positive and finite; elsewhere `σ_kj` keeps its value. That is the curvature of `F` along each
  action, measured where the plan is, so the kink is read once the plan reaches it. A lever in
  units `s` times its own moves `y_kj / s_kj` by `1 / s²`, and `F` in units `c` times its own moves
  it and `|F(ū)|` alike, so `v` stays a pure number.
- **A search that fails in the measured scale is tried again from the floor.** The descent stops
  only where a step in the planner's own scale does not count either, so it stops where the
  planner's rule would, whatever the secants read.
- **The barrier rounds keep their units, and take the projection's check.** A barrier that is the
  least of several margins, as `prescribe` holds a two-sided bound, has a condition that jumps
  where two margins tie, since on each side it reads that side's margin. With the rounds scaled,
  on each of four logs of `scripts/pendulum_demo.py`'s pendulum, seeds 0 to 3, a round ended on a
  tie: a move of `1e-12` raised the shortfall there by 2.8 to 6.1. On the first, neither the new
  descent nor the old took a step from that point. On two of the four the rounds then stopped with
  the barrier held for 2 and 0 of its 40 steps, where unscaled they hold it for all 40 on all
  four. So the rounds run the penalised descent unscaled, with `lr0 = 0.2` and `tol = 1e-9`, and
  where a barrier binds at the planner's scaled plan they start from the descent in the caller's
  units, as ADR 0059 decided.
- **L-BFGS-B minimises in the planner's variables too.** `lbfgs_box_control` hands SciPy
  `J / |J(ū)|` over `v = σ u`, `σ` the planner's read at `ū`, with the gradient `∇J / (σ |J(ū)|)`
  and the box in `v`, and reads the answer back as the planner reads its own, an action on a side
  of its box set to the side. Its `pgtol` and `ftol` were a gradient and a cost in the caller's
  units; in `v` they are pure numbers. It needs no secant of its own: its limited-memory one
  already reads the curvature as it goes.

## Consequences

- **Every row of the table above converges, to one plan.** `tests/test_pessimistic_units.py` holds
  each to `1e-6` of the box.

  | units | after |
  |---|---|
  | lever × 1, and × 1e-6 to × 1e6 | converged after 20 steps, to 1.2e-15 of the box |
  | cost × 1e-6 to × 1e6 | converged after 20 steps, to 5.3e-16 of the box |
  | two levers × 1e-3 and × 1e3 either way, × 1e-6 and × 1e6 | converged after 16 steps, to 2.9e-15 of the box |
  | under the rows, × 1e-3 and × 1e3 either way | converged after 17 steps, to 1.2e-7 of the box |
  | under the radius, lever × 1 and × 1e-6 to × 1e6, cost × 1e-6 and × 1e6 | converged after 102 to 119 steps, to 1.9e-7 of the box |

  Under the rows the two runs take a 17th step that the run in the levers' own units does not
  count: they stop where a step lowers `F` by `1e-14` of it, 1.2e-7 of the box apart.
- **`ConfoundingRobustTask` reports the regret of the optimal plan.** The robust solve converges
  after 1 202 steps, 2.9e-11 above the KKT point's `F` and 1.5e-4 from its plan, at a regret of
  3.470570, 0.583 of greedy's (`tests/test_benchmark.py`).
- **Fewer steps, to a lower penalised cost.** On ADR 0059's 27 problems, each under two penalties:
  a support model fitted on a log spread over half the box, weighted by 1e-2 of `|J(ū)| / H`, and
  the radius 0.3, weighted to put a tenth of `|J(ū)|` on the box's edge. Against the least `F` any
  run reached, the descent with `tol = 0` and a cap of 50 000 among them; `*` marks a run out of its
  10 000 steps:

  | problem | penalty | steps before | after | `F` above the least, before | after |
  |---|---|---:|---:|---:|---:|
  | one lever, 12 steps | support / radius | 601 / 6 463 | 60 / 63 | 1.5e-8 / 4.9e-6 | 2.1e-14 / 1.3e-14 |
  | two levers, 20 steps, box ±5 | support / radius | 2 542 / 10 000* | 2 284 / 629 | 1.6e-7 / 2.1e-2 | 1.3e-11 / 4.3e-12 |
  | two levers, box ±1 | support / radius | 221 / 10 000* | 149 / 113 | 3.3e-9 / 2.5e-3 | 1.5e-13 / 8.5e-15 |
  | nine residual backends | support | 341–439 | 19–58 | 3.5e-9–1.4e-8 | ≤ 4.6e-14 |
  | nine residual backends | radius | 437, the rest 10 000* | 26–72 | 5.9e-9–7.8e-4 | ≤ 1.1e-13 |
  | `nlp_solver_certificate`'s instances, `R` = 1e-3 / 1e-2 / 1e-1 | support | 10 000* / 2 462 / 337 | 10 000* / 2 577 / 229 | 1.9e-6 / 9.9e-8 / 5.4e-9 | 3.7e-10 / 2.4e-12 / 1.3e-13 |
  | the same | radius | 10 000* each | 4 077 / 2 097 / 310 | 2.7e-3 / 2.8e-4 / 1.0e-5 | 2.5e-11 / 1.0e-11 / 4.2e-13 |
  | boxed plants and a hybrid, 4 boxes | support | 7–372 | 7–32 | 4.1e-11–1.8e-9 | ≤ 8.6e-15 |
  | the same | radius | none on box ±0.2, the rest 10 000* | none on box ±0.2, the rest 29–82 | ≤ 3.3e-3 | ≤ 1.9e-14 |
  | a loud learned residual, box ±6 | support / radius | 6 142 / 10 000* | 5 858 / 3 557 | 1.3e-4 / 6.1e-4 | 1.7e-11 / 1.9e-11 |
  | revenue, Hill / Michaelis–Menten, under a budget | support | 328 / 29 | 5 / 12 | 0 / 0 | 2.9e-15 / 7.0e-16 |
  | the same | radius | 8 / 27 | 6 / 28 | 1.8e-16 / 0 | 3.5e-16 / 3.3e-16 |
  | two levers under a rate limit / a budget that never binds | support | 571 / 2 542 | 60 / 2 284 | 1.6e-8 / 1.6e-7 | 3.0e-13 / 1.3e-11 |
  | the same | radius | 251 / 10 000* | 81 / 629 | 9.0e-9 / 2.1e-2 | 2.7e-13 / 4.3e-12 |
  | the golden rule, `dt` = 1 / 0.5 | support | 36 / 39 | 67 / 85 | 2.6e-12 / 1.4e-12 | 3.3e-14 / 2.0e-14 |
  | the same | radius | 156 / 219 | 71 / 95 | 8.2e-11 / 8.2e-11 | 2.7e-13 / 3.6e-13 |
  | a harvest, one square of the final state | support / radius | 53 / 591 | 171 / 5 087 | 5.7e-13 / 2.1e-6 | 4.0e-14 / 9.5e-11 |

  The values are relative. Fewer steps on 45 of the 54, and a lower `F` on 49. On the box ±0.2
  under the radius both take no step, the guess, zero, being where the kink holds every action; on
  the four revenue solves the new `F` is higher by rounding, at most 2.9e-15 of it. The old descent
  ran out of its steps on 19, 18 of them under the radius; the new one on one, the ill-conditioned
  instance under the support penalty, which needs 16 608 steps. 41 736 steps in all, against
  218 089. Where the new descent takes more steps, on 6, the old one stopped on its absolute rule
  at a higher `F`, but for one revenue solve: 28 steps against 27, to the same `F` within 3.3e-16.
- **The defaults changed meaning**, as the planner's did. `lr0` is one Newton step along each
  action alone, in `v`, and `tol` a share of `|F(ū)|`. A fall `ε` in `F`'s own units is
  `tol = ε / |F(ū)|`.
- **The stationarity residual stays in the caller's units**, so it is zero at a KKT point in any
  units and its size is not. At a kink it reads the smoothed gradient: at the box ±0.2 above, a plan
  of zeros that no step leaves reads 0.69.
- **A plan the rows and the box pin to one point stays on it.** From the zero plan that `3 u1 = u2`
  and a box pinned at zero leave alone feasible, `pessimistic_solve` with the cost 1e6 times its own
  took one step, 1.2e-12 off it, and reported `converged`; under a barrier no plan clears, the
  rounds walked it 1.9e-9 to 3.1e-5 off zero as their penalty grew. Both now stay on it
  (`tests/test_pessimistic_units.py`, `tests/test_held_barrier.py`).
- **A failed search off the floor reads one more gradient**, at the point it failed from.
  Otherwise each step reads one gradient, as before.
- **With a support model a barrier's rounds start from the scaled penalised plan**, where they
  started from the old penalised descent's. Only the planner's plan is descended again in the
  caller's units.
- **In float32 the radius' kink is at the plan's rounding.** On the one-lever problem the descent
  converges after 6 172 steps, 1.1e-4 of the box from the float64 plan, where float64 takes 102.
- **The barrier rounds still read other units differently.** Removing the tie's jump, by holding
  each margin's own condition, would hold a stronger condition than the least margin's and change
  what a prescription certifies. That is a decision of its own.
- **`lbfgs_box_control` reads a problem the same in any units.** On the one-lever problem it took
  no step with the lever × 1e-6 or × 1e6, or the cost × 1e-6, and stopped 6.0e-3 of the box from
  the plan with the lever × 1e3 or the cost × 1e-3 (ADR 0059). Now it takes 8 iterations in every
  one of those units, to plans 8.9e-17 of the box apart, 9.3e-6 of the box from the planner's; in
  the lever's own units it ended 3.1e-6 from it (`tests/test_control.py`). Its rules now read the
  scaled gradient and the cost over its value at `ū`, so on `nlp_solver_certificate`'s instances,
  `R` = 1e-3 / 1e-2 / 1e-1, it ends 2.0e-7 / 1.3e-7 / 4.4e-9 above the planner at its cap, where it
  ended 4.1e-8 / 1.1e-8 / 9.2e-10 above, after 104 / 47 / 12 iterations against 82 / 48 / 18, at
  stationarity 8.5e-5 / 1.6e-4 / 7.0e-5 against 1.1e-4 / 6.9e-5 / 2.4e-5. The certificate's least
  ratio of the planner's stationarity to L-BFGS-B's is 214, where it was 618; it asks for 10.

## Alternatives considered

Measured on `ConfoundingRobustTask`'s robust problem from zero, against its optimal `F`:

- **The planner's scale alone, read at the guess.** 100 000 steps, 0.48 above, the residual 0.125:
  the guess shows no kink.
- **The penalty's curvature added to `c_kj` at the guess**, as ADR 0059 suggested. At zero the
  kink's curvature is the radius' weight over 1.5e-154, and so is every action's scale: no step,
  the zero plan, a regret of 17.64. Re-read at each restart of the descent, the same.
- **A weak-secant diagonal**, the quasi-Cauchy update of Zhu, Nazareth & Wolkowicz (1999), which
  fits one secant condition with the whole diagonal at each step: 100 000 steps, 1.8e-4 above,
  and 2.3e-3 above on another seed of the log.
- **L-BFGS-B**, which reached the optimum. It crosses the Python boundary at every iteration, so
  it cannot be compiled, and it takes no rows.
- **The secant without the floor's retry.** It converges here, after 896 steps. But it stops where
  a search fails in the measured scale, which a secant across a kink can set far above the
  curvature elsewhere: on 12 of 40 solves of ADR 0059's problems, with no penalty, the support's
  or the radius', it stopped earlier than with the retry, up to 1.4e-10 above where the retry ends.
- **Scaled barrier rounds.** They end on a margin tie on the pendulum (Decision).
