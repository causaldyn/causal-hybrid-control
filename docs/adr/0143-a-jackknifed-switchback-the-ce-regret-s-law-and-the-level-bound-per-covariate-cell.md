# ADR 0143 — A jackknifed switchback, the certainty-equivalent regret's law, and the level bound per covariate cell

**Status:** proposed, 2026-10-27. Amends ADR 0012: its "Not built" bias-corrected reading is built.
Every addition is opt-in or a new function; no default moves, and every existing output is the
same to the bit.

## Context

Three readings report a number whose statement is looser than the mathematics allows.

- **The switchback's plug-in and local projection carry a 1/T bias.** `read_switchback` fits the
  intercept and `y_t` on the periods it reads, so the local projection is biased by
  `-(H - 1) tau_H / (T - H + 1)` to first order, and the plug-in inherits `a_hat`'s bias,
  `-(1 + 3a)/T` for the slope (Kendall 1954). On one zone at `a = 0.9` over 200 periods the steady
  state read -0.68 of its 10. More zones shrink the standard error and leave the bias: on 16 zones
  the reading's Fieller interval covered 0.821 of 0.95. ADR 0012 lists "a bias-corrected `a_hat`
  for short runs near the unit root" as not built.
- **The certainty-equivalent LQ regret has no mean under a Gaussian model error.** Destabilising
  errors have positive probability, and given stability the closed loop's spectral radius has a
  positive density at 1, where the regret `(r + b^2 p)(k_hat - k*)^2/(1 - (a - b k_hat)^2)` has a
  simple pole. `regret_scaling` reads medians without saying why, and nothing in CHC gives the
  regret's law. The literature gives rates (Mania, Tu and Recht, NeurIPS 2019), not a law to
  check a reading against.
- **The MSM bound on a mean reads pooled tails.** `msm_worst_case_mean`, and through it
  `confounding_robust_radius` and `negative_control_gamma`, puts the weight `Gamma` on the
  pooled sample's top `1/(Gamma + 1)`. That is sharp for a weight with `E[w] = 1`. The MSM's weight
  has `E[w | x] = 1` for each covariate value, and its sharp bound reads the tails of `Y | x`
  (Dorn and Guo, JASA 2023). Pooled, the bound is valid and wider by `sd(Y)/sd(Y | X)`: 1.41 at
  `R^2 = 0.5`, 4.47 at 0.95. A negative control's `Gamma` then falls as covariates explain more of
  the control outcome, 1.285 to 1.058 at the same confounding. On a binary lever, `Gamma` read on
  Tan's (2006) odds scale, as `benchmark_gamma` reads it, bounds the weight by
  `[e + (1 - e)/Gamma, e + (1 - e) Gamma]`, not by `[1/Gamma, Gamma]`, which overstates the
  inflation by `1/(1 - e)`.

## Decision

- **`read_switchback(..., jackknife=True)`**, for `"state_aware"` and `"local_projection"` under
  the model standard error.
  - The estimate is the half-panel jackknife (Dhaene and Jochmans, Review of Economic Studies
    2015), `2 tau_T - (tau_1 + tau_2)/2`. Each half is read on its own periods with its own
    intercepts and fit. An odd run's halves share the middle period, which still cancels the 1/T
    term (`validation/half_panel_jackknife.mac`). Dhaene and Juodis (arXiv 2610.26427) show the
    jackknife stays effective for persistent local projections.
  - The standard error is the jackknife's own: each row's influence on the three fits, twice the
    run's less half of each half's, summed within a zone over the lags the reading keeps. To first
    order it is the run's; what it adds are the fits' second-order differences, which the
    run's own standard error misses. It needs no second zone.
  - The steady state's interval is Fieller's at the jackknifed `(a, b)`, with their covariance from
    the same influences. Every other interval is normal.
  - The plug-in's first-order test runs as before. Where one zone's `sd(a_hat)/(1 - a_hat)` passes
    0.1, the reading warns, as the uncorrected one does, with the jackknife's own reason.
- **`chc.regret.ce_regret_law(a, b, q, r, covariance, state_moment)`** returns a `CERegretLaw`.
  - `covariance` is over `(A.ravel(), B.ravel())`. The regret is `tr(W D)`, with `D` that of
    `certainty_equivalence_gap`. `W = x0 x0'` reads it from a start, and a noise covariance reads
    the long-run cost per step.
  - The second-order law is a generalised chi-square. `dK = G e`, with `G` from the Riccati
    equation differentiated with the gain held:
    `dP = A_cl' dP A_cl + E + E'`, `E = A_cl' P (dA - dB K)`, and
    `dK = H^-1 (dB' P A_cl + B' dP A_cl + B' P (dA - dB K))`. Then
    `M = G' (Sigma* (x) H) G`, with `Sigma* = sum_j A_cl^j W A_cl'^j`. The weights are the
    eigenvalues of `S^1/2 M S^1/2`. The quantiles invert Imhof's formula with CHC's routine.
  - The exact regret's quantiles and `P(destabilise)`, with a one-sided 95 % Clopper–Pearson bound,
    are read off draws of the exact identity.
  - `fourth_order_share = c4/c2` says where the law holds. `c4 = (1/8)(S : grad^2)^2 R` is the
    Gaussian mean of the regret's fourth-order Taylor term. A degree-5 Gaussian rule (Stroud 1971)
    reads it on the exact regret at three scales, extrapolated to zero; it works in any dimension.
    `valid` is `fourth_order_share <= tolerance`, 0.05 by default.
  - `regret_scaling`'s docstring says why medians are right.
- **`covariates=` and `propensity=` on `msm_worst_case_mean`, `confounding_robust_radius` and
  `negative_control_gamma`.**
  - `covariates` take a least-squares prediction from `(1, covariates)` out of the tails. It is
    cross-fitted over 5 interleaved folds, and the bound is `mean(Y) + inflation(Y - m(X))`. It
    stays valid for any fixed `m`, since `E[w m(X)] = E[m(X)]` when `E[w | x] = 1`. It is sharp
    where the residual's law does not move with `x`.
  - `propensity` puts each unit's weight in Tan's box. The bound is the largest mean of
    `w (Y - m(X))` over those boxes with `mean(w) = 1`, a fractional knapsack filled greedily. Its
    dual at the fill's threshold gives the per-unit term that `negative_control_gamma`'s `level`
    reads its standard error from. At one `e` the inflation is `(1 - e)` times the box's.
  - `confounding_robust_inflation`'s docstring says "sharp per covariate value", and what pooling
    costs.

## Consequences

The jackknife on `x' = a x + u + eps`, fair coins, one zone unless stated. Bias is of `tau`.
Coverage is of 0.95, each reading's own interval. "se/sd" is the jackknife's mean standard error
over its spread. 1000–6000 runs a row.

| reading | `a` | periods | zones | bias | jackknifed | covered | jackknifed | se/sd |
|---|---|---|---|---|---|---|---|---|
| steady state (`tau` 5) | 0.8 | 200 | 1 | -0.177 | -0.052 | 0.933 | 0.947 | 1.02 |
| steady state (`tau` 10) | 0.9 | 200 | 1 | -0.679 | -0.124 | 0.931 | 0.965 | 1.00 |
| steady state | 0.9 | 400 | 1 | -0.332 | -0.019 | 0.939 | 0.942 | 0.99 |
| steady state | 0.9 | 200 | 4 | -0.731 | -0.067 | 0.908 | 0.925 | 0.96 |
| steady state | 0.9 | 200 | 16 | -0.732 | -0.040 | 0.821 | 0.934 | 0.94 |
| steady state (`tau` 20) | 0.95 | 400 | 1 | -1.471 | -0.306 | 0.928 | 0.965 | 0.97 |
| steady state | 0.95 | 200 | 1 | -2.597 | -1.766 | 0.917 | 0.967 | 1.72 |
| `tau_5` | 0.9 | 200 | 1 | -0.120 | -0.002 | 0.931 | 0.943 | 0.99 |
| local projection `tau_3` | 0.8 | 60 | 1 | -0.082 | +0.031 | 0.895 | 0.935 | 1.06 |
| local projection `tau_3` | 0.8 | 100 | 1 | -0.060 | +0.005 | 0.912 | 0.945 | 1.04 |
| local projection `tau_3` | 0.8 | 200 | 1 | -0.040 | -0.011 | 0.932 | 0.949 | 1.01 |
| local projection `tau_3` | 0.8 | 100 | 8 | -0.036 | +0.009 | 0.929 | 0.950 | 1.01 |
| channel | 0.8 | 200 | 1 | -0.000 | +0.001 | 0.950 | 0.956 | 1.02 |

- The correction costs spread: 3 % to 19 % more where there is a bias to remove, and 0.6 % on the
  channel, which has none.
- At `a = 0.95` over 200 periods each half fits `a_hat` on 100 periods. The halves' readings
  explode, and the jackknife spreads 3.3 times as far as the reading. The warning names this.
- The local projection over 60 periods overcorrects, by +0.031. The steady state on one zone
  over-covers near the unit root (0.965), since Fieller's interval at the jackknifed `(a, b)`
  keeps the ratio's skew.
- `ce_regret_law` on the scalar plant, unit noise, `q = 1`, `S = s^2 I`: `c2/s^2` and
  `c4/s^4` are 2.3416 and 10.401 at `(a, b, r) = (1, 1, 1)`, 20.191 and 348.37 at
  `(1.2, 0.5, 1)`, and 1.1000 and 5.6069 at `(0.5, 1, 0.1)` (`validation/ce_regret_law.mac`). The
  code reads `c2` to `1e-12` and the share to `2e-7`.
  - At the rational point `(1, 1, 4/3, 1)`, where `p* = 2`, Maxima gives `c2 = 87/32` and
    `c4/c2 = 33333/7424` per `s^2`. Under `S = [[1, 1/2], [1/2, 2]] s^2` they are `69/32` and
    `30525/5888`.
  - On a 2-state plant at 60 digits, `G` agrees with a central difference to `1e-40`, and `M` with
    the exact regret's `t^2` coefficient to `1e-12`.
  - Over 2 000 000 draws on the three plants the exact median stayed within 1 % of the law's while
    no draw destabilised. The mean truncated at a stability margin of 0.01 was within 0.3 % of
    `c2 + c4` wherever the share was 0.05 or less. At shares of 0.17 and 0.18 it was 1.29 and 1.33
    times the second-order mean, and `valid` is false there.
- The level bound (`validation/msm_level_bound.mac`):
  - On a two-cell population (`Y | x = m_x -+ 1`, `m = (0, 4)`, `Gamma = 2`) the per-cell sharp
    bound is 5/2 and the pooled one 13/4. The residualised bound is 5/2.
  - With Tan's boxes at `e = (1/4, 3/4)` the per-cell bound is 9/4, the pooled one 41/16 and the
    residualised one 9/4.
  - The greedy fill equals the linear program, in Maxima's simplex and in HiGHS.
  - A negative control of mean 0.2 reads 1.285, 1.194, 1.119 and 1.058 pooled at `R^2` 0, 0.5,
    0.8 and 0.95. With `covariates` it reads 1.285 at each, within sampling error.
  - At `level = 0.95` an unconfounded null at `R^2 = 0.8` read above 1 at the t test's 5 %.
- Every existing output is unchanged. A digest of `read_switchback`'s readings (every analysis,
  one zone and three) and of the three MSM functions without the new arguments is the same on
  this tree and on 0.22.0. The entry-number tests pass unchanged.

## Alternatives considered

- **The jackknife's interval normal for the steady state too.** It covered 0.881 at `a = 0.9`
  over 200 periods on one zone, below the uncorrected Fieller's 0.931, since `b/(1 - a_hat)` is
  skewed. Fieller at the jackknifed `(a, b)` covered 0.965 there and 0.934 on 16 zones.
- **The run's own standard error for the jackknife.** Its normal interval covered 0.839 to 0.930
  where the jackknife's own covered 0.925 to 0.965. It misses the 3 % to 19 % the correction adds.
- **The standard error across zones only** (Dhaene and Juodis's form, units as clusters). It needs
  many zones. The row-wise influence with the reading's own lags gives the same quantity on one
  zone and on many.
- **Jackknifing `(a, b)` and plugging in.** The ratio's own curvature, `b Var(a_hat)/(1 - a)^3`, is
  then left in the estimate. It is of the same order as `a_hat`'s bias and of the other sign.
- **`c4` in closed form, scalar plants only.** The Taylor series in Maxima holds `c4` in radicals
  for one plant at a time. The degree-5 rule reads it for any plant and any dimension, and it is
  checked against the radicals.
- **A quantile regression of `Y` on the covariates at `tau = 1/(Gamma + 1)`** (Dorn and Guo's
  sharp per-`x` form, a linear program). It also captures a residual spread that moves with `x`.
  It needs one fit for each `Gamma` a search visits, and the location model's gain is already in
  the residualised form. Not built.
- **`propensity` with per-cell tails.** With `e` varying, the fill pools the residual's tails
  across cells. That is valid and sharp where `e` and the residual's law do not move with `x`.
  Per-cell tails need the quantile form above.
