# ADR 0099 — The tube's levels are measured beside it

**Status:** accepted, 2026-10-13. Adds two readers beside the tube of `causal_plan` and `prescribe`
(ADR 0080, ADR 0096); it moves none of the tube's numbers, defaults or verdicts.

## Context

The plan's tube has two parts (ADR 0080, ADR 0096), and neither states a level for the whole
plan.

1. **The noise's radius holds each step at 0.95, not every step at once.** ADR 0080 says so. On
   AR(1) noise two steps at 0.95 each hold together 0.9152476 (`validation/tube_levels.mac` (4)),
   and the more steps, the less.
2. **The field's error is a one-standard-error scale, not a bound at a level.** `_model_error`'s
   docstring says "a *scale*, not coverage", and ADR 0096 adds the drift's error in quadrature as a
   scale too. Nothing gives it a probability, nor the certainty-equivalent certificate
   (`ce_explicit_constant_certificate`) whose radius `rho` an estimated channel must be within.

The author decided on 2026-09-29 that levels for the tube are measured and reported beside it,
and that a level is named only where it holds across plants.

## Decision

### 1. `HorizonConformal`: one conformal scale for every step of the horizon

`chc.uncertainty.HorizonConformal.calibrate(errors, profile, delta)` scores each calibration window
by `max_k ||x_{t+k} - x_hat_{t+k}|| / r_k`, `r_k` a fixed profile such as the plan's tube
(`uncertainty_tube[1:]`), and takes `scale`, the `ceil((n + 1)(1 - delta))`-th smallest score
(`_conformal_rank`, the rank `SplitConformal` reads). `band = scale * r_k`.
`horizon_errors(model, states, actions, dt, horizon, stride=)` makes the windows: the model's RK4
open loop from each logged `x_t` under the logged actions.

- **The theorem.** Cleaveland, Lee, Pappas and Lindemann (AAAI 2024, Theorem 3 and Corollary 1):
  on windows drawn i.i.d., with weights not fitted on the calibration windows, a fresh window lies
  within `band` at every step with probability at least `1 - delta`; its scores tied with
  probability 0, at most `1 - delta + 1 / (n + 1)`. Their `C` is the `1 - delta` quantile of the
  calibration scores with `inf` appended, which is the rank above.
- **One series.** Oliveira, Orenstein, Ramos and Romano (JMLR 2024, Theorem 4): on a stationary
  beta-mixing series the level is `1 - delta - eta`, `eta = eps_cal + eps_train + delta_cal`.
  `mixing_penalty(beta, n, train_lag)` takes the least `eta` over the theorem's blocks and a grid of
  `delta_cal`, each a valid penalty. `gaussian_mixing(F, Q, stride=, span=)` bounds `beta` for a
  stable Gaussian VAR(1) by Pinsker's and Jensen's inequalities,
  `beta(k) <= sqrt(-log det(I - S^-1 F^k S F^k')) / 2`, and reads a window lag `j` as the chain's
  `j * stride - span`. `calibrate(..., mixing=)` states `1 - delta - eta` and refuses where
  `eta >= delta`. Their quantile is the empirical one, `ceil(n (1 - delta))`; ours is never lower.
- **What it covers.** Windows of the policy that logged the calibration windows, under that
  policy's actions: not the plan's own actions, which move the plant where the log did not go. The
  docstring says so.

### 2. `channel_region`: the self-normalised ellipsoid of the fitted channel

`chc.dynamics_id.channel_region(fit, level, bound=)` returns a `ChannelRegion`: for each state `s`,
`{c : ||c_hat_s - c||_V <= radius_s}`, `V = D'D + ridge P` the matrix the fit's least squares
solved, `D` the channel's design on the action residuals.

- **The theorem.** Abbasi-Yadkori, Pal and Szepesvari (NeurIPS 2011, Theorems 1 and 2): with the
  regressors predictable and the noise conditionally `R`-sub-Gaussian,
  `||c_hat - c||_V <= R sqrt(2 log(det(V)^(1/2) det(V0)^(-1/2) / delta)) + ||c||_V0` with
  probability at least `1 - delta`, at every sample size at once. Here `V0 = ridge P`, the fit's own
  ridge, and each state is held at `delta / n`, so all of them at once at `1 - delta`. Predictable
  regressors admit a closed-loop log.
- **What it reads.** `R_s` is state `s`'s root mean square of the moment's residual over `N - q`
  rows, a plug-in for the theorem's known `R`. `||c||_V0` is `bound`, or the estimate's own, about
  1e-3 of the coefficients' effect at the default ridge. The fit keeps `V` and the ridge's diagonal
  as the private fields `_channel_gram` and `_channel_ridge`, set on an identified Euler fit by
  adjustment, unweighted, whose actions move every direction; no other field moves.
- **Under the fit's clusters (0.19.0).** Rows are the theorem's terms, so a shock every unit
  shares in a period, met by levers the units move together, adds up across the period's rows
  beyond each row's `R`: on 30 units over 20 periods sharing each period's shock the 0.95 region
  missed the channel on 6.9 % of 1000 logs, and 0.18.0 read a fit given `clusters` the same way.
  The fit now keeps the channel's design `D` beside `V`, and the region reads the fit's
  `clusters`, as the persistence check does: it sums each state's scores `D_i e_i` within them
  and squares them at CR1's factor (two ways, the three reads the fit's error takes), and `R_s^2`
  is the larger of the rows' mean square and the largest generalised eigenvalue of that spread
  against `V`: the clusters' sums are read as the theorem's terms, in the direction they add up
  most. `V`, the ridge and `D` do not read the clusters; the region does. The rows' `R` stays the
  floor, since a spread read from few clusters is noisy and each row's noise needs its own scale.
  The same logs clustered by unit and period missed 0.0 %, the region 2.5 times as wide (median);
  where the rows' read already held, the clustered region held too and widened 1.0 to 1.9 times.
  This keeps the region's guarantee where the fit names its dependence; it does not make the
  theorem's premise hold within a cluster, and a fit given `periods` is read row by row as before.
- **What it gives.** `operator_radius(x)` bounds `||B_hat(x) - B(x)||_2` at the level, at every
  state at once. `rate_error(states, reach)` bounds `||(B_hat(x) - B(x)) a||` over the states and
  the box `|a_j| <= reach_j`, read at its `2^m` corners: the rate error the tube prices as one
  standard error (`_model_error`), read at a level. `stability_level(rho, x)` is the largest level
  at which `operator_radius(x) <= rho`, the probability the region gives a certainty-equivalent
  gain whose stability holds within `rho` (`validation/tube_levels.mac` (5) checks its two
  inversions).
- **It refuses** a fit that is not identified, an instrument's (`iv`), an `rk4` fit, a weighted
  one, one with a direction its actions never move, a zero ridge, and a `level` outside `(0, 1)`.

### 3. Nothing moves

No number of the tube, no default and no certificate verdict reads either of them. They are
reported beside the tube by a caller who asks.

## Consequences

- **On three plants** (`scripts/bench_tube_levels.py`, 100 logs a plant, float64). Each log runs
  `prescribe` and reads its tube `r_1..r_H`; then 399 fresh windows of `H` steps under the logging
  policy, one a unit, calibrate the band, and 4,000 more are read. "Joint" is the share of those
  windows held at every step, averaged over the logs, with its standard error across them; on
  windows drawn i.i.d. its mean is exactly `ceil(400 (1 - delta)) / 400`, 0.95 and 0.90.

  | plant | `H` | the tube, every step | band at 0.95 | its scale | band at 0.90 | Bonferroni at 0.95 | its end, band's end |
  |---|---|---|---|---|---|---|---|
  | AR(1), one state, closed loop | 8 | 0.835 ± 0.002 | 0.9488 ± 0.0012 | 1.25 | 0.8997 ± 0.0017 | 0.973 | 1.01, 0.92 |
  | two states, bilinear channel | 12 | 0.929 ± 0.002 | 0.9506 ± 0.0012 | 1.05 | 0.9006 ± 0.0017 | 0.979 | 0.97, 0.93 |
  | the quickstart's two states | 15 | 0.774 ± 0.002 | 0.9510 ± 0.0012 | 1.33 | 0.8999 ± 0.0014 | 0.979 | 1.30, 1.15 |

  The band holds its level on every plant, within one standard error. The tube itself holds each
  step at 0.95 or more (0.953 to 0.996 a step) and every step at once at 0.77 to 0.93: the joint
  level it does not state is plant by plant, and lower the longer the horizon. To hold 0.95 jointly
  the band scales the tube by 1.05 to 1.33. Bonferroni's per-step radii at `delta / H` hold 0.97 to
  0.98 for a stated 0.95 and end wider than the band; at 0.90 they end within 1 % of it on the
  bilinear plant (0.833 against 0.839) and wider on the other two. So a joint level for the
  logging policy's windows holds across the plants, as the theorem says; the tube's own joint
  coverage does not, and the tube does not state one.
- **On the plan's own actions** nothing is measured here: the band covers the logging policy's
  windows, and the plan moves the plant elsewhere.
- **One series.** On the AR(1) plant, 20,000 windows of 8 steps cut from one series of 320,000
  steps calibrate the band with `gaussian_mixing`'s bound: `eta` reads 0.0628, so the stated level
  is 0.837 at `delta = 0.1` and 0.737 at 0.2; the next 20,000 windows were held on 0.8998 ± 0.0003
  and 0.8001 ± 0.0004 of them. The theorem's penalty is valid and loose, and it is large at sizes a
  log has: with windows of 10 steps on a chain of correlation 0.8 it reads 0.80 at 100 windows,
  0.25 at 1,000, 0.085 at 10,000 and 0.043 at 40,000, so at `delta = 0.05` it refuses below some
  30,000 windows, 300,000 steps.
- **The region holds the channel and misses it under a hidden confounder.** On the same 100 logs a
  plant, fitted under Euler and adjusted for the confounder, the region at 0.9 held the plant's
  channel on every log of every plant (100, 100, 100). Adjusted for a column of noise in its place,
  it held it on none (0, 0, 0). At its level the region is wide, since the theorem holds at every
  sample size at once: its rate error at the log's mean state and each lever's distance from the
  log's mean action is 4.6 times the tube's one-standard-error rate error on the AR(1) plant
  (0.172 against 0.0375), 6.1 times on the bilinear one (0.369 against 0.0604) and 6.5 times on
  the quickstart's (0.0952 against 0.0147). `log det(V) - log det(V0)` is about 40 on the AR(1)
  log, `q = 2`, where the normal ellipsoid's `chi2_2(0.9)` reads 4.6.
- **The tests** (`tests/test_tube_levels.py`): the rank's coverage by exhaustive enumeration for
  `n <= 12` and over all `8!` orderings of eight scores; the band on 20,000 draws of AR(1) windows,
  39 calibrating, 8 steps, `delta = 0.1`, at 36/40 within four standard errors; the tube's two
  steps together at 0.9152476 (`validation/tube_levels.mac` (4)); the region on 60 closed-loop logs
  at 0.9, and its failure on 30 under a hidden confounder; `eta` against the theorem written out
  loop by loop; `beta`'s closed forms on one and two states.
- **No number moves.** `CausalDynamicsFit` gains a private field; every tube, certified horizon and
  certificate is the same. The windows, the band and the region are a caller's to compute.
- **Not covered.** Panel units with common shocks are not independent windows. The region's `R`
  is read off the log, and its regressors use cross-fitted nuisances, which the theorem's premise
  meets to first order only; under persistent noise it needs the noise's lags in the fit
  (`persistence_check`). An `rk4` fit, the default of `prescribe`, has no region; fit under Euler
  to read one. `stability_level`'s `rho` is the true plant's in the certificate; read at the
  estimate it is the estimate's, which the region does not check.

## Alternatives

- **The confounding-robust region: sign-perturbed sums with the logged dither as instrument**
  (MIV-SPS; Szentpeteri and Csaji, Systems & Control Letters 2023). Its level is exactly
  `1 - q/m` at every `N`, and it stays valid under a hidden confounder, where the ellipsoid here
  fails. Its ellipsoidal outer bound needs `m - 1` semidefinite programs, and the library takes no
  SDP solver. Not built.
- **The ellipsoid on an `rk4` fit.** Its error is the fixed point's, `K^-1 G e`, which mixes the
  channel's least squares with the drift's regression; the theorem bounds the first alone. Carrying
  it through `K` would bound neither exactly. Refused, with a message to fit under Euler.
- **The tube at `1 - delta / H` a step (Bonferroni, as Lindemann et al., RA-L 2023).** Valid, and
  it holds 0.97 to 0.98 where 0.95 is stated, ending wider than the band on every plant at 0.95;
  it also needs `(1 - delta / H) / (delta / H)` windows, 299 at `H = 15` and `delta = 0.05`.
- **The normal ellipsoid from `channel_error`'s covariance,** `sqrt(chi2_q(level))` wide: no
  finite-sample level, and `channel_error` is a root mean over states, not a covariance.
- **A field of the certificate, or a default that reads the band.** Not added: the band is a level
  for the logging policy's windows, not the plan's, and the decision of 2026-09-29 names a level
  only where it holds across plants. A caller who wants either reads it beside the certificate.
