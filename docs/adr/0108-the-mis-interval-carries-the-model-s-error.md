# ADR 0108 — The "mis" interval carries the model's error

**Status:** accepted, 2026-10-19. Amends the interval `evaluate_plan` reports for `"mis"`.
Amended 2026-10-19: `"dr"` weighs toward the plan's law under the plant fitted to the logs, not
under the model, and both methods refuse a plan that fitted plant's loop cannot hold.

## Context

`"mis"` weighs a logged run toward the plan's stationary state-action law. The weight's numerator
is that law under the model; the logs give only its denominator. So the estimate is the plan's
average cost under the model, `Jm`, and the interval read off the weights holds `Jm`, not the
deployed value `J`. A model error moves `Jm` to first order and the interval does not widen.

On the tests' two-state market, over 400 logs of 4000 steps of a randomised plan, that interval
held `J` 0.958 of the time with the model right, 0.905 with `a[0, 0]` off by 0.05, 0.645 off by
0.1, 0.552 with the offset off by 0.05 and 0.035 with the noise doubled. A model fitted by least
squares to a pilot run of 300 steps left 0.680, and to the same logs 0.915. For a deterministic
plan, which the weights smooth and whose interval already carried the smoothing correction, it
held 0.848 with `b` 0.8 times, 0.870 on the pilot's model and 0.037 with the noise doubled.

The first-order term can be read from the logs. With `h` the plan's relative value under the
model, the performance difference is exact:
`J - Jm = E_d[ E[h(x') | x, u] - Em[h(x') | x, u] ]`, `d` the plan's true stationary law of
`(x, u)`. Weighing the model's Bellman residual toward the model's law in place of `d` leaves an
error of second order (`validation/mis_model_error.mac`, symbolic in every parameter of a scalar
loop).

Second order is a product of two first-order terms, and both were the model's error. An offset
error `delta` moves a slow loop's stationary mean by `delta / (1 - f)`, `f` the closed loop, and
leaves an error `2 p f delta^2 / (1 - f)`, `p` the curvature of `h`
(`validation/dr_logged_weights.mac`). With the tests' market's offset off by 0.05 the estimate
was 0.0104 below `J`, further than `"mis"`'s 0.0080 above it, and its interval held `J` 0.475
and 0.843 of the time for the two plans over 4000 steps, 0.013 and 0.520 over 20 000.

## Decision

- `"mis"` keeps its estimate. Its interval is `"dr"`'s on the same weights, stretched to hold the
  estimate: `[min(value, dr_lo), max(value, dr_hi)]`. `degrees_of_freedom` is `"dr"`'s.
- `"dr"` weighs the model's Bellman residual toward the plan's stationary law under the plant
  fitted to the logs by least squares of `x'` on `(x, u, 1)`, not under the model. The model
  enters only through `Jm` and `h`, so the error is the product of the fit's error and the
  model's, zero when either is (`validation/dr_logged_weights.mac`), and the fit's error shrinks
  with the logs. `"mis"`'s weights keep the model's law in the numerator.
- Where the plan's loop under that fitted plant has spectral radius at least one, both methods
  raise `InfeasibleEvaluation`: the logs give the plan no average cost.
- The smoothing correction enters both as before, so `model_error` keeps its meaning.

## Consequences

- For the randomised plan, with the model right, the `"mis"` interval is about 1.2 times as wide
  and covered 0.950. With `a[0, 0]` off by 0.05 and 0.1 it covered 0.960 and 0.970, off by -0.1
  0.965, with `b` 0.8 times 0.955, with the offset off by 0.05 0.978, with the noise doubled
  0.970, on the pilot's model 0.963 and on the same logs' 0.943. For the deterministic plan it
  covered 0.978 to 0.998 over the same ten models, at about 1.4 times the width with the model
  right. Each figure is over 400 logs, a binomial standard deviation of 0.011 at 0.95.
- Longer logs make the old shortfall worse and leave the new interval at its level: over 20 000
  steps of the randomised plan, with `a[0, 0]` off by 0.05 and 0.1, the interval read off the
  weights covered 0.640 and 0.028, this one 0.968 and 0.975, at 1.23 times the width for 0.05.
- `"dr"`'s interval covered 0.940 to 0.963 over the ten models for the randomised plan and 0.973
  to 0.995 for the deterministic one, over 4000 steps; over 20 000, 0.940 to 0.948 and 0.980 to
  0.995 on the six models run there. With the offset off by 0.05 that is 0.963 and 0.978 over
  4000 steps, 0.943 and 0.980 over 20 000, at 0.78 and 0.79 times the old width: the logs' law
  is closer to the plan's than the model's was, so the weights vary less. With the model right
  the width is unchanged to 0.001.
- Where the model is badly wrong the interval is as wide as the gap between the two estimates. A
  user who sees a wide `"mis"` interval with a narrow `"dr"` one is reading the model's error.
- Both methods fit `n (n + m + 1)` coefficients on the logs, and refuse logs with no more
  transitions than `n + m + 1`. The fit assumes the logs come from a linear-Gaussian plant with an offset, the
  class the model is in; a plant outside it leaves a fit error that does not shrink.
- A plan the model holds and the logs' loop does not is refused where it was evaluated before.
  The model's average cost of such a plan is not one the deployed plant would show.
- An interval for `Jm` alone is gone. Under a model the caller trusts, `Jm` has a closed form and
  needs no logs.

## Alternatives

- **The hull of the two intervals.** Rejected: it covered 0.990 to 1.000 at 1.33 to 2.2 times the
  width. The stretched interval holds at its level with less width.
- **A refit bootstrap over the logs.** Rejected: the caller supplies the model, so a refit sees
  only the sampling error of a model fitted to these logs. A model fitted elsewhere, or a wrong
  one, keeps its bias in every draw.
- **Widen symmetrically by `|dr - mis|`.** Rejected: it adds the gap on the side away from the
  truth as well.
- **Tie the stretch to `model_error`.** Rejected: `model_error` is the share of the smoothing
  correction counted as unverified. A share of the stretch has no meaning as a level.
- **Keep `"dr"`'s weights on the model's law and warn when `"dr"` and `"mis"` disagree.**
  Rejected: the gap does not measure the second-order remainder. With the offset off it was
  0.018 and `"dr"` was 0.0104 off; with the noise doubled it is 0.094 and `"dr"` is right.
- **Put the fitted plant in `"mis"`'s numerator too.** Rejected: `"mis"`'s estimate would become
  the fitted plant's average cost, a different quantity from the one it reports, and its interval
  already carries the model's error.
