# The sensitivity level Γ

Offline pessimism assumes the observed transitions *identify* the causal effect. Under hidden
confounding they do not: the effect is only *partially* identified, in a sensitivity interval. `Γ`
is how wide a caller assumes that interval to be, and [`chc.sensitivity`](../api/sensitivity.md) is
the one import that runs the whole line: calibrate `Γ`, turn it into a radius, control against the
radius, and check the result.

## What Γ is

`Γ >= 1` is the level of the **marginal sensitivity model** (MSM, Tan 2006): the odds of treatment
given the covariates and the unobserved confounder lie within a factor `Γ` of the odds given the
covariates alone. For an effect's bounds that is a bound on a density ratio (Dorn and Guo 2023):
the weight that turns the observed outcomes into the unobserved ones lies in `[1/Γ, Γ]`. The sharp
worst-case effect is then a CVaR mixture, and the gap over the point estimate inflates the
pessimism radius by `(Γ-1)/(Γ+1)·(CVaR gap)`. At `Γ = 1` the radius is zero, and the machinery
reduces to the ordinary, fully identified case.

`Γ` is the analyst's **unfalsifiable** input, and a single scalar aggregates over covariates. These
methods robustify pessimism; they do **not** test for confounding.

## Two models, one symbol

Every `Γ` CHC takes or reports is the MSM's, and the report says so. Rosenbaum's `Γ` (2002), the
one matched-design analyses report, bounds a different ratio: the odds of treatment of any two
units with the same covariates. On the models compatible with the data the two nest (Zhao, Small
and Bhattacharya 2019, Prop. 7.1):

```text
Rosenbaum(Γ) ⊆ MSM(Γ) ⊆ Rosenbaum(Γ²)
```

So compare a CHC number with a Rosenbaum-style one through the bracket, not as the same quantity:

- a plan CHC certifies up to `Γ*` holds against every confounder Rosenbaum's model allows at
  `Γ*`, since each is one the MSM allows;
- a study "insensitive to `Γ = 5`" in Rosenbaum's sense covers the MSM only up to `√5 ≈ 2.24`.

`Γ` bounds only how far a confounder moves the treatment. How much bias it produces depends also
on how far it moves the outcome. Rosenbaum and Silber (2009) put the trade in closed form for
Rosenbaum's model: a `Γ` is a confounder that multiplies the odds of treatment by `Λ` and the odds
of a higher response by `Δ`, with `Γ = (ΛΔ + 1)/(Λ + Δ)`, so `Γ = 2` is `Λ = 3` with `Δ = 5`. The
MSM has no such closed form, but the direction carries over, and the tests hold it on a binary
confounder: the `Γ` a known-null outcome needs is 1 when the confounder leaves the outcome alone,
rises with its pull on the outcome, and reaches the confounder's own MSM `Γ` when the outcome *is*
the confounder.

## Calibrate it before spending it

Unfalsifiable is not the same as uncalibrated. Two functions price `Γ` against the data:

- **`benchmark_gamma`** expresses it in units of the confounding the *observed* covariates carry:
  dropping covariate `j` from the propensity produces exactly the pair of propensities the model
  bounds. It reports the exponent `log Γ / log Γ_strongest`, because odds ratios compose
  multiplicatively — "an unobserved confounder `k` times as strong as the strongest thing we did
  observe". A `k` far below 1 is an assumption nobody should be impressed by; a `k` far above 1 is
  one the analyst has to defend. Two limits. It measures the treatment side only, so it is the
  ceiling above: read at `quantile=1.0`, a confounder as strong as covariate `j` in the treatment
  forces at most `Γ_j`, and less if it moves the outcome less. And nothing formal ties an unobserved confounder to an
  observed covariate (Ding 2024, p. 242), so a benchmark sets a scale for `Γ`, not a bound on it.
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
[certificates](certificates.md). There the question inverts: `Γ*` is the **largest** MSM level
under which the barrier stays certified. It is a model parameter, not a
measured amount of hidden confounding. Once the radius swallows the control channel the
robust-optimal action is exactly zero: with the sign of the channel unidentified, acting cannot
improve the guarantee.

## See it

- [Tutorial 5 (robust control)](../tutorials/05_confounding_robust_control.md) — estimate, radius,
  minimax action, and the trade-off across a confounding sweep.
- The [confounded-incentives case study](../case-studies/confounded-incentives.md) — the same
  controller in closed loop on a confounded plant.
