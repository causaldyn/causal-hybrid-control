# ADR 0168 — A switchback's design reading may take a moving baseline

**Status:** accepted, 2026-11-25. Experimental (`chc.switchback`). Amends ADR 0098 (a switchback's
standard error over its coins) and builds its "a baseline that moves with the run". The default,
`baseline="first"`, does not change, and no value it reads moves.

## Context

`read_switchback(..., se="design")` reads `tau_H` by Liang and Recht's moment estimator,
`tau_hat = sum_(s<H) 2 / (T - s) sum_t z_t (y_(t+1+s) - y_0)`, `z = 2u - 1`, with its variance over
the coins, their eq. (2.5): `T^2 var = |M|_F^2 + tr(M M) - 2 sum M_ii^2 + |T_q' h|^2`,
`h = T_g 1 + 2 (e - y_0)`. Everything in the readings the coins did not cause stays in `h`: the
noise's distance from the first reading, and the lever's own mean effect, `T_g 1`. ADR 0098 paid
for it in width, 3.8 to 6.6 times the model's interval on AR(0.8) shocks and 90 times on a trend,
and left a baseline that moves with the run unbuilt: unbiased, but with eq. (2.5) no longer
holding as it stood and its variance not derived.

The moving baseline reads each reading against the one `L` periods back, and the first `L` against
the first reading:

    m_t = y_t - y_(t-L)  for t >= L,      m_t = y_t - y_0  for t < L.

That is the first reading's `y' = y - y_0` passed through `1 - S^L`, `S` the lag on the run, a
lower-triangular Toeplitz filter. Lower-triangular Toeplitz matrices multiply as their sequences
convolve, so `m = T_g~ x + e~` with `g~ = g - S^L g` and `e~ = (1 - S^L)(e - y_0)`: Liang and
Recht's model again, its noise as fixed over the coins as `e`.
`validation/switchback_moving_baseline.mac` derives what follows:

- **STEP 1**: the filter identity, at `T = 6`, `L = 2`, symbolic.
- **STEP 2**: over all 32 designs at `T = 5`, symbolic `g`, `e` and `y_0`, the mean is `tau_H` and the
  variance is eq. (2.5) of `(g~, e~)`, at `(H, L) = (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)`. At
  `L < H` the mean is `tau_H - sum_(L<=s<H) g_(s-L)`: a reading fewer than `H` periods back holds
  the coins the lags below `H` read.
- **STEP 3**: the noise's share over a stationary noise of autocovariance `gam`, the first
  reading's noise included. At the channel, the first reading's expected `T^2 var` less the moving
  baseline's is `8 ((T - L) gam_L - sum_(k=L+1..T) gam_k)`, at `T = 6`, `L = 1, 2, 3`. At any
  `H`, a row `i >= L` whose window the run holds gains
  `4 sum_(s,s'<H) q_s q_s' (gam_0 - gam_(s-s') + gam_(s-s'-L) + gam_(s-s'+L) - gam_(i+s+1) - gam_(i+s'+1))`
  (checked row by row at `T = 7`, `H = 2, 3`). Far from the start of a long run that is `8 gam_L`
  at the channel and `4 (H^2 - H) gam_0` on white noise.
- **STEP 4**: the lever's share on the first-order plant, `g_s = b a^s`, at the channel, a period
  of a long run: `b^2 (a^2 / (1 - a^2) + 1 / (1 - a)^2)` from the first reading,
  `b^2 (2 (1 - a^L) / (1 - a^2) - 1)` from the moving baseline. Their difference is
  `2 a^L b^2 / (1 - a^2) + 2 a b^2 / ((1 - a)^2 (1 + a)) >= 0` on `0 <= a < 1`: the first
  reading carries the lever's mean effect, `b / (1 - a)`, in every row, and the moving baseline
  differences it away.
- **STEP 5**: so at the channel the moving baseline is narrower, per period of a long run, iff
  `8 gam_L` plus STEP 4's difference is positive. On the white-noise plant, `a = 0` and
  `gam_L = 0`, the two tie exactly at every `T`: `T^2 var = T b^2 + 8 T gam_0` for both. It is
  wider where the noise's autocovariance at `L` is negative. Past the channel the first
  reading's own noise enters each of the `H` lags a row sums, `H^2 gam_0` a row, so white noise
  favours the moving baseline. The textbook bar, `gam_L > gam_0 / 2`, compares a difference with
  no baseline at all, which the design does not read: a level would then enter the variance.
- **STEP 6**: the spreads the tests check on `x' = 2u + eps`, `T = 60`, `H = L = 1`, `eps` an AR(rho)
  of unit innovations: 0.4472 for both at `rho = 0`; 0.4954 and 0.5774 at `rho = -1/2`, a ratio of
  1.165.

It passes in Maxima 5.49 (the CI's) and 5.50.

## Decision

- **`read_switchback(..., se="design", baseline="moving")`**, opt-in. `baseline="first"`, the
  default, is ADR 0098's reading, bit for bit.
- **`baseline_lag`**, `L`, is `H` unless given, the shortest lag that keeps the estimate unbiased.
  A longer one pays where the noise's autocovariance peaks there, a weekly cycle in daily readings.
- **The estimate** is ADR 0098's on `m`: `g_s` is unchanged below `L`, and `tau_hat` sums the lags
  below `H <= L`.
- **The variance** is eq. (2.5) of `(g~, e~)`, and its plug-in is ADR 0098's on `m`, with `L` more
  lags, `K = max(H, ceil(3 log T)) + L`, since `g~` carries the response's echo, `-g_(s-L)`, from
  lag `L` on. Where `|g_s| <= C c^s`, `|g~_s| <= C (1 + c^-L) c^s`, so Liang and Recht's
  Proposition 2 holds of it as of `g`.
- **It refuses** a baseline other than `"first"` and `"moving"`; `"moving"` under any other standard
  error than the design's; `baseline_lag` with the first reading; a lag that is not a whole number
  at least `H`; and a run no longer than the `2 K` lags the plug-in reads: at `T = 300` and `H = 3`,
  `L = 131` reads and `L = 132` does not.
- **The log record** `switchback_reading` carries `baseline`.

## Consequences

An exploration, 2000 runs a case, a fair coin each period, 95 % intervals (`0.95 +- 0.0195` at
four standard errors), the plant from rest 400 periods before the run; "first" is ADR 0098's
reading and "moving" this one at `L = H`:

| case | covered, first / moving | spread, first / moving | spread, moving over first | standard error over spread, first / moving | width, first / moving |
|---|---|---|---|---|---|
| `tau_3`, `x' = 0.8 x + 2u + eps`, i.i.d. `eps`, `T = 60` | 0.954 / 0.953 | 2.371 / 1.330 | 0.561 | 1.04 / 1.14 | 9.64 / 5.93 |
| `tau_3`, AR(0.8) shocks, `T = 60` | 0.967 / 0.964 | 6.655 / 2.554 | 0.384 | 0.91 / 1.02 | 23.69 / 10.23 |
| `tau_3`, AR(0.8) shocks, `T = 100` | 0.963 / 0.959 | 4.979 / 1.900 | 0.382 | 0.96 / 1.04 | 18.73 / 7.72 |
| channel, AR(0.8) shocks, `T = 60` | 0.970 / 0.967 | 2.091 / 0.356 | 0.170 | 0.95 / 1.12 | 7.80 / 1.56 |
| channel, one trending noise path, `T = 200` | 0.962 / 0.992 | 3.948 / 0.058 | 0.015 | 1.04 / 1.44 | 16.17 / 0.33 |
| channel, `x' = 2u + eps`, i.i.d. `eps`, `T = 60` | 0.979 / 0.971 | 0.449 / 0.444 | 0.990 | 1.02 / 1.16 | 1.80 / 2.01 |
| channel, the same, `T = 400` | 0.950 / 0.954 | 0.178 / 0.173 | 0.971 | 0.92 / 1.03 | 0.64 / 0.70 |
| `tau_3`, `x' = 2u + eps`, i.i.d. `eps`, `T = 60` | 0.966 / 0.983 | 1.326 / 0.715 | 0.540 | 0.92 / 1.19 | 4.80 / 3.32 |
| channel, `x' = 2u + eps`, `eps` AR(-0.5), `T = 60` | 0.975 / 0.961 | 0.493 / 0.583 | 1.184 | 1.02 / 1.09 | 1.97 / 2.48 |

- **Where the outcome persists, the spread falls to 0.17 to 0.56 of the first reading's**, and the
  interval holds its level. On ADR 0098's AR(0.8) shocks, `tau_3` over 60 periods, the width went
  from 23.69 to 10.23, against the model interval's 6.20, which covered 0.903. On ADR 0098's
  trending noise path the channel's went from 16.17 to 0.33, against the model's 0.18, which
  covered 0.919: differencing at lag 1 turns a linear trend into a constant step, small against
  the trend's distance from the first reading.
- **On white noise at the channel the spreads tie**, as STEP 5 says, and the interval is 9 % to
  12 % wider. The plug-in reads `L` lags more, each with its error. Its standard error over its
  spread was 1.02 to 1.19 on 60 and 100 periods and 1.03 on 400, against the first reading's 0.91
  to 1.04, and 1.44 on the trending path, which it covered 0.992 of.
- **Where the noise alternates the moving baseline is wider**: 1.184 times on `eps` AR(-0.5), with
  STEP 6 predicting 1.165.
- **Past the channel, white noise favours it**: 0.540 of the spread for `tau_3`.
- **The tests** pin:
  - the estimate against its transcription on every design of a run of 10, at `(H, L) = (1, 1),
    (1, 2), (2, 2)`; its standard error against the plug-in transcribed over full matrices; and
    its mean and variance over all 1024 designs against eq. (2.5) of `(g~, e~)`, to `1e-12`;
  - on the persistent plant of the table's first row, 1000 runs: the interval's coverage, the
    estimate's mean, a standard error 1.0 to 1.25 times the spread, each reading's spread against
    eq. (2.5) of its own response and noise averaged over the runs' noise paths, and the moving
    baseline's below 0.6 of the first reading's (0.547; eq. (2.5) predicts 0.543);
  - on `x' = 2u + eps` at the channel, 1000 runs: both spreads against STEP 6's at `rho = 0` and
    `rho = -1/2`;
  - every refusal, the longest lag a run of 300 reads, the default's identity and the log record.

## Alternatives

- **Make it the default.** It moves every design reading, and on a noise that alternates it is
  wider; opt-in first.
- **Read each lever against the reading just before it, `y_(t+1+s) - y_t`**, ADR 0098's note. Its
  baseline moves with the coin, not with the reading, so the readings are no filter of the outcome
  and eq. (2.5) does not hold as it stands. It reads the last lag as the moving baseline at
  `L = H` does and the others against a reading fewer periods back, and on white noise past the
  channel it keeps the first reading's cost, one shared reading in each of the `H` lags a row sums.
- **Regress on the reading `L` back** instead of subtracting it. Its coefficient is read off the
  same coins, so the estimate is biased by `O(1/T)` and eq. (2.5) no longer holds: ADR 0098's
  reason for not centring a zone on its own mean.
- **Choose `L` on the data**, the lag with the least plug-in. The chosen estimate's variance is not
  eq. (2.5) of any one `L`; a lag chosen before the run, or on a pilot, keeps it exact.
- **A Markov design read by a distributed-lag local projection.** Examined before this change
  and not adopted: on the plants tried, the block difference in means read tighter.
