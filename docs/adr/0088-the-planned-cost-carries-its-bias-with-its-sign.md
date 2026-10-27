# ADR 0088 — The planned cost carries its bias, with its sign

**Status:** accepted, 2026-10-12. Experimental.
Amended by ADR 0145, 2026-10-27: `prescribe(planned_cost_bias=True)` reads the bias into the
certificate's `planned_cost_bias`, opt-in, refitting the channel with its influence.

## Context

A plan made on an estimated channel reports `task_cost`, the cost of its actions on that estimate.
What the plan pays on the plant differs, and not only by noise: in expectation, over the
estimate's error. `plan_regret_bound` bounds how far the plan is from the best one on the planning
model; `decision_weight` prices what an error costs against the plan that knew it. Neither says
whether the planned cost itself reads high or low.

With `e` the estimate's error, `S = E[e e']`, `W` the decision weight and `J_ee` the task cost's
Hessian in the channel at the fixed plan, the second-order delta method gives

```text
E[J_true(û) − J_model(û)] = tr(W S) − tr(J_ee S) / 2 − J_e · E[e] + o(|S|).
```

`tr(W S) ≥ 0` is the optimiser's curse: the plan fits its estimate's error. `−tr(J_ee S)/2` is the
planned cost's own curvature at the fixed plan, `≤ 0` where that cost is convex in the channel.
So no theorem fixes the sign of the sum. On one step of `x1 = a x0 + b u` at the cost
`q x1² + r u²` the bias is `a² q² x0² (2r² − 5b²qr + b⁴q²)/(r + b²q)³` per unit of `S`, negative
between `ρ = q b²/r = (5 − √17)/2 = 0.438` and `(5 + √17)/2 = 4.562`, and positive outside
(`validation/planned_cost_bias.mac`). The expansion is the classical delta method; the
optimiser's-curse framing is Ito, Yabe and Fujimaki (ICML 2018), and Gupta, Huang and
Rusmevichientong (Operations Research 2024) treat linear objectives.

## Decision

- **`CausalPlan.planned_cost_bias(covariance, *, error_mean=None, tolerance=None)`** returns a
  `PlannedCostBias`: the planned cost, the three terms, and as properties the bias, its
  `direction` (`optimistic` where the planned cost reads low, `pessimistic` where it reads high)
  and the corrected cost. The covariance is of the one-step channel, raveled as `decision_weight`
  ravels it and as `ChannelMove.covariance` holds it, so the bias sits beside `W` on the same
  object and in the same coordinates. `W` is `decision_weight`'s, through the same envelope code.
- **`chc.misspecification.planned_cost_bias(plan, fit)`** reads `S` off the fit's influence over
  every parameter the fit holds, the drift's included, in `misspecification_cost`'s envelope: the
  model moved by a residual of the fit's class inside its field. The drift carries the channel's
  error, so the plan pays for both. `S` is summed within clusters and across the periods' cosines
  as the channel's error is. Its `unseen` counts the directions the log never moved that the plan
  weighs, which `S` holds no error along.
- **Refusals.** A plan that meets a bound or a row with a zero multiplier (`weakly_active`) has an
  active set that moves with the error, so its bias is not one expansion: refused. So is a missing
  covariance — `None`, or a fit made without `influence=True` — and one that is not a finite,
  symmetric, positive semidefinite matrix of the channel's size. Plans `decision_weight` refuses
  (a barrier, pessimism penalties, no problem) are refused for the same reasons.
- **The terms are read at the plan and the estimate,** not at the plant's channel, which the
  expansion is about. The difference is of higher order.
- **The fit's own first-order bias is not known** to the fit-based reader, so its `first_order` is
  0; a caller who has an estimate of `E[e]` passes it to the plan's method.

## Consequences

- One step reads Maxima's numbers to 1e-10 (the coefficient 6.170016381437646 at `ρ = 0.274`,
  `W` 9.225094387724484, `J_e` −2.610076308640553), and the closed form to 1e-8 over 25 drawn
  plants. The sign flips at both roots, and at each root the two terms cancel to 1e-9 of the first.
- **A Monte Carlo of refits.** Two states, two levers, three steps; one design of 60 dithered
  decisions, the noise drawn afresh 4000 times, the one-step channel refitted by least squares
  (its error about 12 % of the channel) and the plan re-solved on each fit. The gaps less their
  first-order term averaged 0.01290 ± 0.00027; the plan on the plant's channel predicts 0.01274
  (optimism 0.01552, curvature −0.00278). The optimism alone, the bias's usual stand-in, is ten
  standard errors off.
- On a fit with the drift's influence set to zero, the fit-based bias equals the plan's own at
  `L S L'`, `L` the map from the field's channel to the one-step channel, to 1e-9.
- The bias is O(1/n) and the expansion holds as far as `W` does: while the active set holds, and
  for errors small against the channel. A cross-fitted channel's own bias enters at the same order
  through the third term.

## Alternatives

- **A field on `DecisionCertificate`, filled by `prescribe`.** The certificate is where
  `regret_bound` is read, and it is where the number belongs once `prescribe` has `S`. Today it
  does not: `prescribe` fits without `influence=True`, whose rows cost `N · n · p` floats, and its
  plan may hold levers on the log's rule or read drivers, whose perturbation the fit's class does
  not span. The certificate's record is also a persisted schema (ADR 0055). Adding the field is
  left for its own change.
- **A parametric bootstrap:** draw channels from `N(B̂, S)`, re-plan, and average
  `J(u*(B*); B̂) − J(u*(B*); B*)`. It handles a moving active set, at one solve per draw; the
  expansion costs one Hessian.
- **The optimism alone, `tr(W S)`,** as the planned cost's bias. It has a sign, and the wrong
  number wherever the cost curves in the channel: 22 % high on the Monte Carlo above, and of the
  wrong sign between the roots.
