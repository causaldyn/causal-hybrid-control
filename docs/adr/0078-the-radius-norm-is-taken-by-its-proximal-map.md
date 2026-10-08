# ADR 0078 — The radius' norm is taken by its proximal map

**Status:** accepted, 2026-10-08. Amends ADR 0073: the penalised descent no longer differentiates
the confounding radius' norm, and takes it, with the box, by its proximal map.

## Context

`ConfoundingRobustPenalty(radius=r)` in the `lam_unc` channel of `pessimistic_solve` and
`pessimistic_control` adds `lam_unc r Σ_t L_t ||u_t||` to the penalised cost, `L_t` being 1 or
the certified weights. The norm has a kink at every `u_t = 0`, and a plan under the radius holds
there the actions it does not use. `penalty_trajectory` smooths it as `sqrt(||u_t||² + ε²)`, `ε` a
millionth of the actions' root mean square, so that it has a gradient, and ADR 0073 reads the
curvature this puts at the kink by a secant as the descent goes. Two defects remained. Measured on
`integrate-0.16.0` at the parent of this change, float64 enabled as `tests/conftest.py` does,
`tol` and `steps` at their defaults:

- **Float32 stopped short of the plan.** On `tests/test_pessimistic_units.py`'s one-lever problem
  under the radius 0.3 at a weight of 0.2, from a zero guess, float64 converged after 89 steps and
  float32 after 69, 1.0e-3 of the box from float64's plan, its stationarity residual 2.1e-3
  against 6.5e-7. On the two-lever problem, 102 steps against 55, 1.9e-3 of the box apart. Under
  the support penalty instead, 13 against 14 steps, 2.5e-8 apart. ADR 0073 quoted 102 and 6 172
  steps, 1.1e-4 apart; at the parent of ADR 0059's amendment (the line search's third), this
  measured 104 and 98 steps, 3.1e-5 apart, so neither count reproduces now, while the gap has grown.
  There were two causes. The kink's curvature, the radius' weight over a millionth of the plan's
  size, sits at the plan's rounding in float32. And the penalty was summed in float32: the cost
  reads float32 actions in float64, its weights being float64, but `penalty_trajectory` squares and
  sums them in their own dtype, so the norm term of `F`, 0.33 at the plan, was rounded at 3e-8,
  while the falls near the plan are 1e-9. With the norm taken by its proximal map but still summed
  in float32, the descent stopped after 30 steps, 1.8e-5 of the box away.
- **A zero guess on the kink took no step, where zero is not stationary.** At `u_t = 0` the smoothed
  norm's gradient is zero, so the descent stepped along the task's gradient alone, and every action
  that gradient moved paid the norm's full slope for a fall the gradient predicted without it. The
  units problem in the box ±0.2 does not show it: it converges after 32 steps. Its case is
  `test_plan`'s boxed oscillator, from `(1.5, 0)` over 12 steps in the box ±0.2, under the radius
  0.3 at a weight of 3.1, which puts about a tenth of the cost of doing nothing on the box's edge,
  as ADR 0073's 54 solves weighted it. The descent took no step and reported `no_progress`, its
  residual 0.69: `sqrt(12) × 0.2`, the box-clipped task gradient, since the smoothed norm reads no
  slope at zero. The unsmoothed problem's residual, `||0 - prox(0 - ∇J)||` with the prox of the norm
  and the box, is 0.115 there: three actions' task gradient outweighs the norm's weight, 0.93. With
  a second lever on the position and a weight of 3.25, the same: no step, a residual reported as
  0.92, where the unsmoothed one is 0.186.

## Decision

- **The radius' norm is not differentiated.** Where the `lam_unc` channel holds a
  `ConfoundingRobustPenalty` with a non-zero radius and weight and there are no linear rows, the
  descent minimises `F(u) = f(u) + Σ_t w_t ||u_t||`, `w_t = lam_unc r L_t`, the norm unsmoothed and
  `f` the task cost and the other penalties, whose gradient alone the descent reads.
  (`chc.support._radius_norms`.)
- **Each trial is a proximal step**, a forward-backward step (Combettes & Wajs 2005) in the
  descent's variables `v = σ u`: `v' = prox(v - η ∇_v f)`, the prox of `η Σ_t w_t ||v_t / σ_t||`
  plus the box (`chc.control._shrink`). With one lever a step it is the soft-threshold at
  `η w_t / σ_t` clipped to the box, exact, and an action the threshold reaches is exactly zero. With
  several, `x = clip(y / (1 + μ / σ²))` for the multiplier `μ` at which `μ ||x / σ|| = η w_t`, a
  product that rises with `μ`, found by bisecting `log μ` over the dtype's range.
- **The line search asks for a third of the fall `f`'s gradient and the norm predict together**,
  `∇_v f · (v - v') + h(v) - h(v')`, the rule of Tseng & Yun (2009). By the prox's optimality the
  prediction is at least `||v - v'||² / η`, so a short enough step meets the rule from any point the
  prox moves, a kink included. The fallback to the longest falling trial and the stopping rule are
  ADR 0073's.
- **The norm term is read in the smooth part's precision**, in `F` and in the prediction alike: a
  float32 plan whose cost reads it in float64 sums its norm in float64 too.
- **The secant stays**, and reads `f`'s curvature alone: the kink is no longer in it, but a support
  penalty's curvature, which the floor does not see, still is (Alternatives).
- **`pessimistic_solve`'s residual under the radius is the proximal one**,
  `||u - prox(u - ∇f)||` in the caller's units: zero exactly where the unsmoothed problem is
  stationary, where the smoothed norm's gradient read zero at any action on the kink.
- **Under linear rows nothing changes.** The rows' projection is not the norm's proximal map, so the
  descent there differentiates the smoothed norm as before. A weight of zero and a radius of zero
  also take the descent as it was.

## Consequences

- **Float32 converges as float64 does.** The one-lever problem converges after 45 steps in
  either dtype, its plans 1.4e-7 of the box apart; two levers after 50 and 46, 2.7e-6 apart, where
  the float64 plan itself stops 1.8e-6 of the box from where the descent goes with `tol = 0` and
  the float32 plan 8.3e-7 (`tests/test_pessimistic_radius.py`). With float64 off altogether the
  one-lever problem converges after 20 steps, 2.3e-4 of the box from the float64 plan, its cost
  2.6e-7 above that plan's, under float32's resolution of it, 3.9e-7; it took 59 steps to 4.3e-4.
  Under the support penalty in pure float32 the plans are 4.7e-3 apart.
- **A non-stationary zero guess moves.** The one-lever oscillator converges after 3 steps, three
  actions at the box's edge and the rest at exactly zero, its penalised cost 22.3219 where zero's
  is 22.3485, its residual 0; the two-lever one after 8, its residual 2.7e-8. Where zero is
  stationary the descent still takes no step, and now reports a residual of 0 there.
- **Every unit's plan is one plan.** The units problem under the radius, in its own units, the
  lever in units 1e-6 to 1e6 times its own, and the cost 1e-6 and 1e6 times, converges after 45
  steps in each, to 3.6e-15 of the box, where it took 89 to 97 steps to 2.2e-7
  (`tests/test_pessimistic_units.py`).
- **`ConfoundingRobustTask`'s robust plan** converges after 1 005 steps where it took 1 184, at a
  regret of 3.470573 where it read 3.470583; the KKT point of the smoothed problem reads 3.470575.
  It holds 16 of its 25 actions at exactly zero, the least of the rest 0.52; the regret is 0.583 of
  greedy's, as before (`tests/test_benchmark.py`).
- **Plans without the radius do not move, bit for bit.** Compared before and after, actions, cost
  history, step count, status and residual: the support penalty on the units problems, one lever in
  units 1e-6 to 1e6 and the cost 1e-6 to 1e6, two levers in four pairs of units, two levers under
  rows, one and two levers in float32; the radius at a weight of zero, at a radius of zero, at the
  zero radius `from_sensitivity` gives at `Γ = 1`, under rows and under a budget; a Wasserstein
  penalty on a learned residual at weights 0 and 0.1; `pessimistic_control`; the planner on the
  boxed oscillator in four boxes in either dtype and under a rate limit; and `causal_plan` plain,
  with a support model, and holding a barrier through its rounds. 41 plans, all identical.
- **The descent minimises the unsmoothed norm.** `penalty_trajectory` still returns the smoothed
  one, above it by at most `ε` a step, for every caller that differentiates it; the descent reads
  only its radius and weights.
- **A trial with several levers a step bisects 96 times**, on arrays of the horizon's length; one
  lever a step costs a soft-threshold.
- **Under rows both defects remain.** A plan under the radius and linear rows, from a zero guess on
  the kink, can take no step where zero is not stationary, and its float32 plan stops short.

## Alternatives considered

Measured on three problems, each from a zero guess in float64 and float32: the units problem (box
±5, weight 0.2), and the oscillator in the box ±0.2 (weight 3.1) and ±2 (weight 0.465). The smoothed
rows run the library's descent with the penalty smoothed over `ε` times the plan's root mean
square; the orthant row and its proximal control run a loop with the library's floor scale, third
and fallback, and stopping rule, but no secant. "From the plan" is from the library's proximal plan,
as a share of the box; `F` is the unsmoothed penalised cost.

| problem | method | steps, float64 / float32 | float32 from float64 | `F` above the plan | from the plan |
|---|---|---:|---:|---:|---:|
| units, ±5 | proximal (chosen) | 45 / 45 | 1.4e-7 | 0 | 0 |
| | smoothed, `ε` 1e-6 (before) | 89 / 69 | 1.0e-3 | 9.7e-8 | 1.7e-7 |
| | smoothed, `ε` 1e-4 | 68 / 41 | 1.5e-4 | 9.7e-6 | 1.6e-5 |
| | smoothed, `ε` 1e-2 | 48 / 10 000* | 1.5e-4 | 9.7e-4 | 1.5e-3 |
| | proximal, no secant | 31 / 31 | 1.5e-8 | 8.4e-15 | 7.8e-8 |
| | orthant-wise, no secant | 31 / 31 | 1.8e-8 | 9.3e-15 | 7.8e-8 |
| oscillator, ±0.2 | proximal (chosen) | 3 / 3 | 7.5e-9 | 0 | 0 |
| | smoothed, any `ε` | 0 / 0 | 0 | 2.7e-2 | 0.50 |
| | proximal or orthant-wise, no secant | 2 / 2 | 7.5e-9 | 0 | 0 |
| oscillator, ±2 | proximal (chosen) | 28 / 28 | 4.9e-8 | 0 | 0 |
| | smoothed, `ε` 1e-6 / 1e-4 / 1e-2 | 40 / 32, 38 / 1 420, 30 / 23 | 9.1e-5, 8.9e-5, 6.6e-5 | 2.6e-8, 2.6e-6, 2.6e-4 | 2.2e-6, 8.8e-6, 8.7e-4 |
| | proximal, no secant | 24 / 24 | 5.6e-8 | -5.3e-15 | 7.5e-7 |
| | orthant-wise, no secant | 24 / 24 | 3.5e-8 | -3.6e-15 | 7.5e-7 |

`*` ran out of its steps.

- **A wider smoothing.** It trades the kink's curvature for a bias: the plan moves off the
  unsmoothed one by up to 1.5e-3 of the box, and the float32 plan stays 1e-4 from float64's. From
  a zero guess no `ε` relative to the plan's size helps, since there it is zero: the gradient still
  reads no slope at the kink. An `ε` in the caller's units would read a lever's units, as the fixed
  `1e-6` the smoothing replaced did.
- **An active-set rule at zero**, the orthant-wise step of Andrew & Gao (2007): at an action on the
  kink, step along the least-norm subgradient's negative if it is non-zero, and keep every trial in
  the orthant it started in. With one lever a step it matches the proximal step here, step for
  step. It has no form for a group norm, whose levers leave zero together along a direction rather
  than into an orthant, and it needs a pseudo-gradient and an orthant carried beside the gradient.
  The proximal map is one rule for one lever or several, the box taken in the same map.
- **No secant on the proximal path.** With the radius the only penalty it takes fewer steps, 31
  against 45 on the units problem, 33 against 50 on two levers and 862 against 1 005 on
  `ConfoundingRobustTask`, at a regret of 3.470575. Under a support penalty that weighs, with the
  radius, it takes more: on the units problem at a support weight of 0.5 and 5, 13 and 13 steps
  against 8 and 7, and on two levers 21 and 18 against 12 and 14. The floor is the cost's curvature
  alone, and the secant is what reads a penalty's; kept, as ADR 0073 decided.
- **The proximal map under rows**, by the Dykstra-like splitting of Bauschke & Combettes (2008),
  replacing the box's clip inside `chc.control._dykstra` with the prox of the norm and the box. The
  projection's active-set polish and the test of a trial that moves only by the projection's error
  are a projection's, and would each need their own proximal form. Not built; rows keep the
  smoothed norm.

## References

- Andrew, G. & Gao, J. (2007). Scalable training of L1-regularized log-linear models. *ICML*.
- Bauschke, H. H. & Combettes, P. L. (2008). A Dykstra-like algorithm for two monotone operators.
  *Pacific Journal of Optimization* 4(3), 383–391.
- Combettes, P. L. & Wajs, V. R. (2005). Signal recovery by proximal forward-backward splitting.
  *Multiscale Modeling & Simulation* 4(4), 1168–1200.
- Tseng, P. & Yun, S. (2009). A coordinate gradient descent method for nonsmooth separable
  minimization. *Mathematical Programming* 117, 387–423.
