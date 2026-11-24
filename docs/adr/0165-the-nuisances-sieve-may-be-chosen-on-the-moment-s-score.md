# ADR 0165 — The nuisances' sieve may be chosen on the moment's score

**Status:** accepted, 2026-11-25. Experimental (`fit_causal_residual(nuisance_choice=)`). Amends
ADR 0120 (the sieve is chosen out of fold) and ADR 0127 (the media-mix plant's 1.07). The
default, `nuisance_choice="prediction"`, does not change, and no value it reads moves.

## Context

ADR 0127 left two readings open: the sieve's choice passed over a spline the channel needed, and
on the media-mix plant of `chc.research.mmm` the spline's channel scattered 1.07 times as far as
the polynomial's, with no measurement of whether the polynomial holds that plant's nuisances.

**The choice passed over the spline on a push the polynomial misses.** On
`x' = -7 x + 8 u + 3 tanh(2 z)`, held over a step of 0.1 and stepped by RK4, `u = -z + 0.5 a`,
noise `1e-3` and 1000 rows, the polynomial's channel error was on average 50 to 55 times the
spline's, and the choice took the spline on 28 of 60 logs. With nuisances linear in their target,
`S y` and `S u`, and `y = C u + g + e`, the target's residual is `C (I - S) u + (I - S)(g + e)`
and its mean square is `C^2` times the action's, the score's and twice their cross product
(`validation/sieve_choice_score.mac`, STEPs 1 and 2). The push the polynomial misses is in the
score `(I - S)(g + e)`: its out-of-fold mean square was 0.49 under the polynomial and 0.00015
under the spline (medians over the 60 logs). In the target it sat beside `C^2` times the action's
noise, 8.2: the target's mean square read 8.70 against 8.44 to 8.55, while the spline's columns
raised the action's by 2.7 %. Each over its column's variance, 52 and 1.24, the two terms moved
by 0.005, in opposite directions, and the polynomial won on 32 logs.

**The media-mix plant, measured against its true nuisances.** 40 regions of 26 weeks, 1000
transitions a log, fitted by Euler with `season` adjusted for and `nuisance_degree=2`; log `k` is
`MarketingMixSystem.sample(seed=k)`, `k` from 0 to 199, and the folds are the fit's own `seed=0`.
Spend is `clip(0.5 + 0.9 s + N(0, 0.35^2), 0.05, 2)`, independent of the state given the season
`s`, so its nuisance is the clipped normal's mean in `s`; the sales row's RK4 increment is a sum
over the channels (to `8.9e-16`), so its nuisance is a sum of one-dimensional integrals over each
channel's spend, read by 64-point Gauss-Legendre quadrature and checked against 400 000 draws on
five rows (every difference within 1.8 Monte Carlo errors). The oracle solves the fit's own
moment on the true nuisances. Each ratio below is a channel entry's standard deviation over the
logs against the oracle's, the median over the 60 entries, its bootstrap SE over logs beside it.

- **The polynomial does not hold the nuisances.** Projected on its span over the log, the truth
  leaves 0.70 of the sales row's noise variance, 0.07 to 0.19 of the adstock rows' and 0.27 of
  the spend's: the spend's mean is flat at the floor below `s = -0.5`, where 31 % of the weeks
  sit, and a quadratic cannot follow the kink. The spline without the powers leaves 0.045, at
  most 0.010 and 0.015. Out of fold, estimation noise included, 0.84 and 0.36 against 0.18 and
  0.14.
- **The polynomial's channel does not read the miss.** It scattered 0.989 times the oracle's
  (0.009), its mean within 0.03 of the oracle's scatter of it (0.32 at the most, two entries past
  three paired errors). The spline, chosen on all 200 logs, scattered 1.079 times the oracle's
  (0.009) and 1.084 times the polynomial's; on logs 0 to 39, 1.068 (0.87 to 1.31), ADR 0127's
  reading.
- **It is not the spline's column count.** Over logs 0 to 99, the polynomial with 40 columns of
  independent noise beside it, 61 as the spline with the powers has, scattered 0.996 times the
  oracle's (0.014); the polynomial with the two true nuisances' columns beside it, 23, which hold
  them, 1.067 (0.009).
- **It is the polynomial's misfit, which the moment reads as the action's variation.** Each
  sieve's two parts apart, over logs 0 to 99: the residuals of the truth's projection on the
  sieve, its approximation without estimation noise, scattered 0.932 times the oracle's (0.008)
  for the polynomial and 0.998 (0.003) for the spline; the oracle's residuals cross-fitted on the
  sieve, its estimation noise without approximation, 1.067 (0.009) and 1.061 (0.012). The two
  products, 0.99 and 1.06, are near the two sieves' own readings on those logs, 1.001 (0.011)
  and 1.075 (0.012). The estimation costs either sieve 6 to 7 %, whether it has 21 columns or
  56; the polynomial wins it back because the part of the spend's mean it misses, a function of
  the season, stays in the action's residual, and the moment reads it as variation in the
  action. That is sound only where the rate's part the action does not move is held: here the
  polynomial's score read 0.93 to 0.99 of the better spline's out-of-fold mean square, state by
  state, medians over the logs. Nothing on the log tells it from the miss of ADR 0120's
  quadratic plant, where the same reading of a misfit is a bias.

## Decision

- **`fit_causal_residual(nuisance_choice="score")`**, opt-in. The three sieves of ADR 0120 are
  cross-fitted as before, and each is read by the sum of the logs of the out-of-fold mean squares
  of the action's residual and of the moment's score, the target's residual less its
  least-squares fit on the action's in the channel's design, column by column; a column that
  moves by rounding alone is left out, and no mean square is read below its column's rounding.
  The least wins, the first on a tie. At `nuisance_knots=None` the check reads its refit's sieve
  by the same rule.
- **Logs, in each column's own units.** Each term is `log(s^2 (1 + r^2))`, `s^2` the column's
  noise and `r` the nuisance's misfit over it; the two sum to at least `2 log(1 + r_m r_g)`
  (STEP 3), and `sqrt(N) r_m r_g` bounds the moment's bias in its own errors under homoskedastic
  noise (STEP 4, Cauchy-Schwarz). The rule reads the two factors of that bound, the action's and
  the score's misfit, each against its own noise.
- **The default stays `"prediction"`**, and `prescribe` reads it. Flipping it is the author's
  call, on the numbers below.

## Consequences

Each design's logs are drawn from `numpy.random.default_rng(k)`, `k` the log's index, as ADR
0120's and ADR 0127's measurements drew them; both rules read the same logs and the same folds,
and the change in scatter carries a paired bootstrap SE.

- **The tanh push** (60 logs): the spline was chosen on 60, against 28. The channel's root mean
  square miss fell from 0.029 to 0.00083 under `euler` and from 0.045 to 0.0012 under `rk4`;
  the 95 % interval covered on 0.967 under either rule and integrator.
- **ADR 0120's designs**: the channel's scatter over the polynomial's, its bias in its mean error
  and its 95 % interval's coverage, the default's reading before the arrow and the score's after,
  one value where the two agree to the digits shown; the change in scatter's paired bootstrap SE
  in brackets.

  | design | logs | same choice | scatter | bias, errors | covers |
  |---|---|---|---|---|---|
  | ADR 0107's plant, 1000 rows | 200 | 168 | 0.737 → 0.742 (0.013) | +0.13 → +0.15 | 0.960 → 0.955 |
  | ADR 0107's plant, 4000 rows | 200 | 170 | 0.540 → 0.541 (0.004) | +0.04 → +0.06 | 0.965 |
  | linear plant of ADR 0107, 1000 rows | 200 | 197 | 1.000 (0.002) | +0.07 | 0.965 |
  | normal covariates, 500 rows | 100 | 97 | 1.012 → 1.000 (0.012) | -0.09 → -0.10 | 0.950 |
  | normal covariates, 2000 rows | 100 | 99 | 0.998 → 1.000 (0.002) | +0.04 | 0.930 |
  | `t(3)` covariates, 500 rows | 100 | 91 | 0.669 → 0.636 (0.024) | +0.09 → +0.13 | 0.980 |
  | `t(3)` covariates, 2000 rows | 100 | 96 | 1.003 → 0.969 (0.038) | +0.15 → +0.19 | 0.970 |
  | lognormal covariates, 500 rows | 100 | 84 | 0.427 → 0.400 (0.025) | -0.17 → -0.10 | 0.970 |
  | lognormal covariates, 2000 rows | 100 | 96 | 0.756 → 0.757 (0.008) | +0.05 → +0.06 | 0.940 → 0.950 |
  | quadratic nuisance, normal, 1000 rows | 100 | 99 | 1.000 | +0.01 | 0.970 |
  | quadratic nuisance, normal, 4000 rows | 100 | 100 | 1.000 | -0.17 | 0.890 |
  | quadratic nuisance, `t(5)`, 1000 rows | 100 | 98 | 0.995 → 1.000 (0.004) | +0.22 → +0.23 | 0.950 |
  | quadratic nuisance, `t(5)`, 4000 rows | 100 | 100 | 1.000 | -0.09 | 0.990 |
  | quadratic nuisance, `t(3)`, 1000 rows | 40 | 40 | 1.000 | -0.17 | 1.000 |
  | quadratic push, `rk4` / `euler` | 60 | 60 | 1.000 | -0.26 / -0.24 | 0.933 / 0.967 |

  No scatter moved by more than 1.4 of its paired errors. The three covariate designs at 500
  rows narrowed by 1.0 to 1.4 errors; the widest widening was 0.005 of the polynomial's, 1.25
  errors, on the quadratic nuisance in a `t(5)` confounder at 1000 rows.
- **The media-mix plant** (logs 0 to 99): the score's rule chose a spline on 96 logs and the
  polynomial on 4, where the default chose a spline on all 100, and the channel scattered 1.079
  times the polynomial's against 1.077, a paired difference of +0.002 (0.006). The polynomial's
  score read 0.93 to 0.99 of the better spline's, its action's residual 1.18 times: the rule
  reads the spend's miss, as it should, and the 1.07 is the price of holding the spend's mean,
  not of the choice.
- A fit under `"score"` costs one least-squares solve of the target's residual on the channel's
  design for each sieve, beside the three cross-fits it already makes.

## Alternatives

- **The estimated variance of the channel**, the plug-in sandwich of each sieve's moment. It
  chose the polynomial on 99 of the 100 media-mix logs, but on the quadratic nuisance in a `t(3)`
  confounder it took the spline without the powers on 7 of 40 logs, a bias of 34 errors, and on
  the normal one the spline with the powers on 45 of 100: it reads a misfit of the action's
  nuisance as variation, the bias with it.
- **The score over the target's variance**, the default's units: it chose the spline on 49 of the
  60 tanh logs, and the channel scattered 0.41 times the polynomial's against 0.024.
- **The logs of the target's mean squares in place of the score's**: the same choice as the
  default on 57 of the 60 tanh logs. The action's noise times the channel is in both.
- **The score's term alone**: as good on these designs, but it drops the action's misfit, the
  other factor of the bias bound.
- **Make it the default.** It moves the default fit wherever the two choices differ: on the tanh
  logs, and on up to 16 % of the logs of ADR 0120's designs. Opt-in first.
