# ADR 0064 — Noise that persists within units is read, not removed

**Status:** accepted, 2026-10-07.

## Context

`fit_causal_residual` reads the channel off transitions on a Markov premise: given the state, the
levers and the adjustment set, a transition's noise does not depend on the past. A panel's noise
often persists within a unit, as demand that stays high for some periods does, and a lever often
persists too, as a budget kept for some periods does. Then the state is a common effect of the
past lever and the past noise. Given the state, this period's lever and this period's noise move
together, and the channel is biased by an amount its error, read from the same moment, does not
see.

`validation/persistent_noise_bias.mac` derives the bias for one state and one lever:
`x' = phi x + dt (b u + g z) + e`, `u = a + k z`, the lever's own part `a` AR(`r_u`) of spread
`s_a`, the noise `e` AR(`r_e`) of spread `s_e`, and `z` drawn afresh and adjusted for. At
stationarity the channel reads `b` plus

    -Cov(a, x) Cov(x, e) / (dt Var(x) Var(u~)),
    Cov(a, x) = dt b s_a^2 r_u / (1 - phi r_u),   Cov(x, e) = s_e^2 r_e / (1 - phi r_e),

for `u~` the lever less its projection on the state and `z`. The bias is nothing where the noise
does not persist, where the lever does not, or where the channel is 0. Where `b r_u > 0` its sign
is the opposite of `r_e`'s. At `phi` 0.95, `dt` 0.1, `b` 0.8, `g` 1.5, `k` 0.9, `s_a` 0.5,
`s_e` 0.05 and `r_u` 0.7, it is -0.0028, -0.0123 and -0.0286 at `r_e` 0.3, 0.7 and 0.9: 0.35 %,
1.5 % and 3.6 % of the channel.

`scripts/bench_persistent_noise.py`, section `bias`, draws those plants: 200 units, 50 panels a
cell, the state started at N(0, 1) or 300 periods before the log. Each panel is fitted four ways:
adjusting for `z` alone (*plain*); for `z` and the state a period earlier (*state's lag*); for
`z` and the state, the lever and `z` a period earlier (*all lags*); and adjusting for `z` with
the lever a period earlier as the instrument (*instrument*). The table gives each way's mean
error in the channel, and the persistence check of the plain fit (below): its mean correlation,
and how often it rejected at 5 %.

| `r_e` | periods | start | plain | state's lag | all lags | instrument | correlation | rejected |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 20 | cold | 0.0032 | 0.0033 | 0.0052 | -0.0039 | 0.001 | 0.04 |
| 0.0 | 40 | cold | -0.0031 | -0.0030 | -0.0027 | -0.0065 | 0.003 | 0.04 |
| 0.0 | 100 | cold | 0.0000 | 0.0001 | 0.0012 | -0.0033 | 0.000 | 0.02 |
| 0.3 | 20 | cold | -0.0004 | -0.0075 | -0.0003 | -0.0066 | 0.298 | 1.00 |
| 0.3 | 40 | cold | -0.0048 | -0.0109 | 0.0009 | -0.0260 | 0.297 | 1.00 |
| 0.3 | 100 | cold | -0.0027 | -0.0086 | -0.0006 | -0.0222 | 0.297 | 1.00 |
| 0.7 | 20 | cold | -0.0029 | -0.0154 | 0.0047 | -0.0918 | 0.690 | 1.00 |
| 0.7 | 40 | cold | -0.0122 | -0.0248 | 0.0008 | -0.1024 | 0.693 | 1.00 |
| 0.7 | 100 | cold | -0.0130 | -0.0250 | -0.0007 | -0.1171 | 0.692 | 1.00 |
| 0.9 | 20 | cold | -0.0169 | -0.0306 | 0.0002 | -0.1937 | 0.886 | 1.00 |
| 0.9 | 40 | cold | -0.0245 | -0.0360 | -0.0010 | -0.2399 | 0.884 | 1.00 |
| 0.9 | 100 | cold | -0.0286 | -0.0392 | -0.0016 | -0.2569 | 0.884 | 1.00 |
| 0.7 | 40 | warm | -0.0157 | -0.0275 | 0.0004 | -0.1169 | 0.692 | 1.00 |

The Monte Carlo standard errors are 0.0007 to 0.0048, and 0.0027 to 0.0096 for the instrument.

- *The plain fit matches the closed form* at 100 periods: -0.0027, -0.0130 and -0.0286 against
  -0.0028, -0.0123 and -0.0286; and started 300 periods before the log, -0.0157 ± 0.0028 at 40
  periods against -0.0123. From a cold start at 20 periods it reads less, as the state's start,
  not its past, makes most of its spread.
- *The bias does not shrink with the log.* At 100 periods it is 1.1 and 2.1 times the estimate's
  spread across panels at `r_e` 0.7 and 0.9.
- *All lags remove it*: in every cell within 2.3 Monte Carlo standard errors of the channel, where
  the plain fit is 14 of them off at `r_e` 0.9. The state, the lever and `z` a period earlier, with
  `z` now, span the past noise (the `.mac`'s step 4, determinant -1), so what is left of this
  period's lever and noise are their innovations, which are independent. Where the noise persists
  they also narrow the estimate, to 0.26 to 0.61 of the plain fit's spread; where it does not they
  widen it, by 1.33 to 1.55 times.
- *The state's lag alone makes it worse*, 1.4 to 3.2 times the plain fit's bias from 40 periods:
  it spans three of the five and not the past noise, and is a common effect of the past lever and
  the noise before it.
- *The instrument is invalid*: -0.117 and -0.257 at `r_e` 0.7 and 0.9 over 100 periods, 9 times
  the plain fit's bias. The past lever moves this period's state, so given the state it is tied to
  the past noise.

## Decision

- **`persistence_check(fit, units, periods)`** reads the lag-1 autocorrelation of the channel
  moment's residual within units, *experimental*. Each state's residual is divided by its root mean
  square. The products of a unit's consecutive transitions, summed over the states, are summed
  within the unit, and the units' sums are tested against none: their total over its CR1 spread,
  each unit's sum less its pairs' share of the total, against `t(G - 1)` as in ADR 0062. Left
  uncentred, that statistic stays under `sqrt(G - 1)`, so at six units or fewer it could never
  clear `t(G - 1)`'s 5 % quantile; on the size panels below it rejected 0.5 % at 10 units and
  3.5 % at 40.
- **An identified fit keeps its moment residual**, `CausalDynamicsFit.moment_residual`, with or
  without `influence=True`: `N` by `n` floats.
- **`prescribe` reads it** on the transitions it fits, by unit and the period each starts in.
  `DecisionCertificate.noise_persistence` and `noise_persistence_p` hold the correlation and the
  p-value; the report states them, and at 5 % names noise that persists and the lags to adjust
  for; `to_json` writes both. The schema version stays 2: keys were added, and none changed its
  meaning.
- **No remedy is applied.** The fit is not refitted on the lags: the report says what to adjust
  for, and the caller, who knows which lags the log holds, adds them to the adjustment set.

## Consequences

- *The check's size and power.* `scripts/bench_persistent_noise.py`, section `size`, draws the
  plain fit's panels at 40 periods, 400 a cell, and reads how often the check rejects at 5 %:

  | units | `r_e` 0: rejected | correlation | `r_e` 0.3: rejected | correlation |
  |---|---|---|---|---|
  | 5 | 3.50 % | -0.013 | 63.25 % | 0.256 |
  | 10 | 4.25 % | -0.009 | 99.25 % | 0.276 |
  | 40 | 4.25 % | -0.003 | 100 % | 0.294 |
  | 200 | 4.75 % | -0.001 | 100 % | 0.298 |

  A size's Monte Carlo standard error is 1.1 points near 5 %.
- *A false alarm is a line.* On a log whose noise is fresh, the report names persistence one time
  in twenty. It refuses nothing and moves no plan.
- *What it reads is the noise's persistence, not the bias.* Where no lever persists, persisting
  noise biases nothing and the line still names it; the bias needs both.
- *It reads one lag.* The closed form and the remedy are for AR(1) noise; noise that persists over
  more lags needs those lags in the adjustment set as well, and the check does not say how many.

## Alternatives

- **Fitting on the lags by default.** Rejected: each unit's first transition is lost, the lags
  must exist in the log, and where the noise does not persist they widen the estimate by a third
  to a half for nothing.
- **The lever a period earlier as an instrument.** Rejected: invalid, and biased 9 times as much as
  the plain fit.
- **The state a period earlier alone.** Rejected: it makes the bias worse.
- **A test at several lags at once**, a portmanteau over the pooled residual's autocorrelations.
  Not built: the bias derived here is carried by the first lag.
