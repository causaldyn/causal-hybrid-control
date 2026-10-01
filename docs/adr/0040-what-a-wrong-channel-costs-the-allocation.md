# ADR 0040 — What a wrong channel costs the allocation

**Status:** proposed, 2026-10-01.

## Context

A geo test is worth what it saves the plan it informs. `chc.experiment` prices an experiment by
the decision weight `W` of a one-shot decision, and `CausalPlan.decision_weight()` gives the `W` of
a plan over a plant; the budget split `chc.allocation.allocate` makes (ADR 0034) had none. Without
it, where to run a test, and on which channel, is chosen by how precisely a parameter is read
rather than by how much of the plan the parameter moves: a tight interval on a channel the plan
holds at its floor is worth nothing to the plan.

## Decision

- **`decision_weight(channels, budget, periods, *, lower, upper, history=None)`** in
  `chc.allocation`, with `allocate`'s arguments and plan, returning an `AllocationWeight`: each
  parameter's place (`"1.curve.scale"`), the matrix `W`, the plan, and the channels held at an end
  of their box. `AllocationWeight.expected_regret(covariance)` is `tr(W S) / 2`, what a plan made
  on estimates with covariance `S` loses on average, to second order.
- **The parameters are each channel's inexact leaves**: the kernel's, the curve's and the
  coefficient, in the channel's own coordinates, so a fit's covariance in those coordinates can be
  read as it comes.
- **`W` by the implicit function theorem at the plan.** Inside their boxes the rates meet at the
  budget's price, `V_c'(u_c) = periods * price`. An error `d_c` in channel `c` moves its rate by
  `g_c d_c / h_c`, `g_c` the slope's gradient in the channel's parameters and `h_c = -V_c''`, and
  the price moves every free rate back by the share `(1/h_c) / sum(1/h)` of the total, so the
  budget stays spent: `du = D d`, `D = (I - w 1' / sum(w)) A`, `w = 1/h`. The first-order loss is
  nil, the slopes being equal and the moves summing to nothing, and the second is
  `du' diag(h) du / 2`, so `W = D' diag(h) D`. The derivatives are JAX's, of the channel's worth
  over the plan and its tail with the history's carryover in.
- **A channel at an end of its box carries no weight**, as a pinned lever does in `chc.experiment`:
  locally an error does not move it off.
- **Refused**: an S-shaped curve, planned on its envelope, where the rate's response to an error
  is the envelope's and not the curve's; and a channel inside its box whose worth is not strictly
  concave at its rate.
- **Named `AllocationWeight`, not `DecisionWeight`.** `chc.plan.DecisionWeight.regret(change)`
  reads the one-step channel's error; a class of the same name whose method reads a covariance
  would take either matrix without complaint.

## Consequences

- `validation/allocation_decision_weight.mac` derives `W` for two exponential and two
  Michaelis-Menten channels, the split in closed form, and holds it to the Hessian of the realised
  loss: the difference is the zero matrix under Maxima 5.46 and 5.50. The tests hold the library's `W` to
  its numbers to `1e-8`. On three channels with carryover (tanh, Michaelis-Menten, exponential;
  kernels of 6, 8 and 4 periods; a 30-period history) the plan is re-made on channels moved along
  three random directions and scored on the true ones: twice the loss over the step squared, one
  Richardson step taken, meets `d' W d` to `1e-4`, and with each parameter off by a tenth of its
  size the quadratic is within a tenth of the loss. A binding cap pins its channel, whose rows are
  zero. Fourteen mutations caught.
- It is local. A channel pinned with a small multiplier leaves its end under an error the weight
  does not see, and the loss there is more than `W` says, as for `chc.experiment`'s pinned levers.
- It prices a test's covariance; it does not choose the test. On causaldyn-bench's Track M v2 a
  choice of where to test read through it was pre-registered and scored on 500 geo panels
  (`results/track_m2_geo.md` there): choosing the tested geos by the regret `W` expects their test
  to leave cut the plan's regret to 0.0050 per euro of the budget [0.0037, 0.0062], against 0.0143
  for the set that represents the market, Abadie and Zhao's, 0.0164 for the set a synthetic control
  fits best and 0.0241 for a random set; paired, -0.0094 [-0.0128, -0.0059] against the
  representative set. The expected regret, local and at independent errors, under-states the
  realised by about half, 0.0028 against 0.0050: it ranks sets, it does not forecast what a test
  leaves.

## Alternatives

- **Finite differences of the realised loss.** `2 p^2` re-plans, a step long enough to carry the
  third order and one short enough for the quotient to read the allocation's tolerance rather than
  the loss.
- **Differentiating through `allocate`.** Its price is found by bisection and its box by clipping,
  neither differentiable where it matters; the implicit function theorem at the plan is exact there.
- **Reusing `chc.experiment`'s zone decision.** A one-shot quadratic decision: no box per channel,
  no carryover, no budget spent over periods.

## Not built

- **The test's covariance.** What a design leaves the parameters with is the fit's, not the
  weight's; `chc.lift.fit_lift` reports intervals, and a design read before it runs needs a
  prior.
- **Weakly active ends**, counted as `CausalPlan.decision_weight()` counts them, and S-shaped
  channels.
