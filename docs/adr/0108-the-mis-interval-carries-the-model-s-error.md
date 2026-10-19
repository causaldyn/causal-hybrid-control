# ADR 0108 — The "mis" interval carries the model's error

**Status:** accepted, 2026-10-19. Amends the interval `evaluate_plan` reports for `"mis"`.

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
error of second order. That is what `"dr"` computes (`validation/mis_model_error.mac`, symbolic in
every parameter of a scalar loop).

## Decision

- `"mis"` keeps its estimate. Its interval is `"dr"`'s on the same weights, stretched to hold the
  estimate: `[min(value, dr_lo), max(value, dr_hi)]`. `degrees_of_freedom` is `"dr"`'s.
- `"dr"` is unchanged, bit for bit.
- The smoothing correction enters both as before, so `model_error` keeps its meaning.

## Consequences

- For the randomised plan, with the model right, the interval is about 1.2 times as wide and
  covered 0.950. With `a[0, 0]` off by 0.05 and 0.1 it covered 0.963 and 0.978, off by -0.1
  0.960, with `b` 0.8 times 0.958, with the offset off by 0.05 0.985, with the noise doubled
  0.993, on the pilot's model 0.983 and on the same logs' 0.943. For the deterministic plan it
  covered 0.975 to 1.000 over the same ten models, at about 1.4 times the width with the model
  right. Each figure is over 400 logs, a binomial standard deviation of 0.011 at 0.95.
- Longer logs make the old shortfall worse and leave the new interval at its level: over 20 000
  steps of the randomised plan, with `a[0, 0]` off by 0.05 and 0.1, the interval read off the
  weights covered 0.640 and 0.028, this one 0.975 and 0.985, at 1.23 and 1.58 times the width.
- Where the model is badly wrong the interval is as wide as the gap between the two estimates. A
  user who sees a wide `"mis"` interval with a narrow `"dr"` one is reading the model's error.
- The interval carries the model's error to first order only. `"dr"`'s own second-order
  remainder is not carried: with the offset off by 0.05 `"dr"` covered 0.475 and 0.843 for the
  two plans over 4000 steps, and 0.013 and 0.520 over 20 000. The stretched interval covered there because the truth lay between the two
  estimates; nothing makes that so.
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
