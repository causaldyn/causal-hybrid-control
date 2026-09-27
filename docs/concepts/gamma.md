# The sensitivity level Γ

Offline pessimism assumes the observed transitions *identify* the causal effect. Under hidden
confounding they do not: the effect is only *partially* identified, in a sensitivity interval. `Γ`
is how wide a caller assumes that interval to be, and [`chc.sensitivity`](../api/sensitivity.md) is
the one import that runs the whole line: calibrate `Γ`, turn it into a radius, control against the
radius, and check the result.

## What Γ is

`Γ >= 1` bounds a density ratio: under the bounded density-ratio (marginal) sensitivity model the
density-ratio weight lies in `[1/Γ, Γ]`. The sharp worst-case effect is then a CVaR mixture, and
the gap over the point estimate inflates the pessimism radius by `(Γ-1)/(Γ+1)·(CVaR gap)`. At
`Γ = 1` the radius is zero, and the machinery reduces to the ordinary, fully identified case.

`Γ` is the analyst's **unfalsifiable** input, and a single scalar aggregates over covariates. These
methods robustify pessimism; they do **not** test for confounding.

## Calibrate it before spending it

Unfalsifiable is not the same as uncalibrated. Two functions price `Γ` against the data:

- **`benchmark_gamma`** expresses it in units of the confounding the *observed* covariates carry:
  dropping covariate `j` from the propensity produces exactly the pair of propensities the model
  bounds. It reports the exponent `log Γ / log Γ_strongest`, because odds ratios compose
  multiplicatively — "an unobserved confounder `k` times as strong as the strongest thing we did
  observe". A `k` far below 1 is an assumption nobody should be impressed by; a `k` far above 1 is
  one the analyst has to defend.
- **`negative_control_gamma`** inverts a known-null outcome for the smallest `Γ` that reconciles
  it. On an outcome whose true effect is zero, any nonzero estimate is confounding, so that `Γ` is a
  **lower bound on the confounding actually present**: assuming less is refuted by the data. It
  returns `inf` when no `Γ` reconciles the null — the negative control has then refuted the model
  class instead of calibrating it.

## Spend it on performance

The radius widens a pessimism radius (never optimistic, tight at `Γ = 1`, monotone). The
confounding bias then becomes a control-regret floor that is *second* order in the bias. Under an
asymmetric loss — undershoot costlier than overshoot, or the reverse — the minimax controller
shifts the gain to hedge the costlier error. It bounds the worst-case cost, wins beyond a
problem-dependent confounding threshold, and pays a bounded premium near zero confounding.

## Or on safety

The same radius, spent on a constraint instead of the objective, is the barrier line of
[certificates](certificates.md). There the question inverts: `Γ*` is the **largest**
sensitivity-model level under which the barrier stays certified. It is a model parameter, not a
measured amount of hidden confounding. Once the radius swallows the control channel the
robust-optimal action is exactly zero: with the sign of the channel unidentified, acting cannot
improve the guarantee.

## See it

- [Tutorial 5 (robust control)](../tutorials/05_confounding_robust_control.md) — estimate, radius,
  minimax action, and the trade-off across a confounding sweep.
- The [confounded-incentives case study](../case-studies/confounded-incentives.md) — the same
  controller in closed loop on a confounded plant.
