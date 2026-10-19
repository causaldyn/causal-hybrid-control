# ADR 0098 — A switchback's standard error over its coins

**Status:** accepted, 2026-10-13. Extends `read_switchback` (ADR 0012) and closes ADR 0025's
"coverage of the local-projection interval" for a lever drawn by a fair coin each period.

## Context

`read_switchback` reads every effect with a standard error from a model of the outcome's errors:
the regression's, robust to heteroskedasticity, and for the local projection keeping the `H - 1`
lags of its overlapping sums. That error model is right when the outcome's noise is stationary
and the run is long. Two cases break it:

- **A short run whose noise persists.** On the working model, `x' = 0.8 x + 2 u + eps` with `eps`
  AR(0.8), the local projection's interval for `tau_3` covered 0.903 of 1000 runs over 60
  periods, 0.881 over 40 and 0.905 over 100, where it should cover 0.95.
  *Note, 2026-10-13:* the shortfall is the run's length, not the noise's persistence. With
  i.i.d. `eps` the same interval covered 0.849, 0.878 and 0.910 over 40, 60 and 100 periods
  (1000 runs each), and 0.929 over 200 and 0.940 over 400 (200 000 runs each). The regression
  fits its intercept and `y_t` on the same periods, which biases the local projection by
  `-(H - 1) tau_H / (T - H + 1)` to first order on one zone, `-2 tau_3 / (T - 2)` here, less on
  more zones (`validation/local_projection_bias.mac`), and reads its variance low; more HAC lags
  make it worse (0.835, 0.809 and 0.734 at 5, 10 and 20 lags over 60 periods). The local
  projection logs a warning below 400 periods a zone, or where that bias passes 0.3 standard
  errors.
- **A noise that trends.** With one trending noise path held fixed and only the coins redrawn, its
  interval for the channel covered 0.919 of 1000 runs over 200 periods. The trend is in its
  residuals, and its estimate sat 0.42 of its own spread off the channel.

ADR 0025 listed the local projection's coverage as not measured. `randomisation_test` is exact at
any `T`, but it tests a sharp null and needs a range for the persistence to give an interval.

Liang and Recht (Biometrika 2025, "Randomization inference when N equals one") read the impulse
response of a linear time-invariant plant, `y = g * x + e`, off a lever drawn by a fair coin each
period. The noise `e` is fixed and arbitrary: it may trend, persist or shift, as long as the coins
do not move it. Their moment estimator, eq. (2.4), `<(2x - 1) * q, 2y> / T`, is unbiased over the
coins for `<D_T q, g>`, `D_T = diag((T - t)/T)`, and eq. (2.5) is its exact variance over them:

    T^2 var = |M|_F^2 + tr(M M) - 2 sum_i M_ii^2 + |T_q' (T_g 1 + 2 e)|^2,   M = T_q' T_g,

with `T_v` the lower-triangular Toeplitz matrix of `v`. Their Proposition 2 makes the plug-in of
it consistent, with `g` read over `K = Theta(log T)` lags (`2K` in `tr(M M)`) and `e` over the
residuals, when `|g_t| <= C c^t`, `c < 1`.

## Decision

- **`read_switchback(..., se="design")`** reads a local projection with its standard error over
  the coins. The default, `se="model"`, is unchanged, bit for bit.
- **The estimate** is their eq. (2.4) at `q_s = T / (T - s)`, `s < H`:
  `tau_hat = sum_(s<H) 2 / (T - s) sum_t z_t (y_(t+1+s) - y_0)`, `z = 2u - 1`. Each lag is read
  over the `T - s` pairs that reach it, so the target `<D_T q, g>` is `tau_H` itself, and the
  estimate is unbiased for it at every `T`.
- **Each zone is read from its first reading**, `outcome[z, 0]`, which precedes the lever. Any
  baseline fixed before the coins keeps the estimator unbiased and eq. (2.5) exact, with `e`
  replaced by `e - y_0`. Without it, the variance grows with the outcome's level: a zone at 1000
  would read 1000 times wider. With it, a zone's level does not move the reading.
- **The zones are averaged**, and so are their variances over `Z^2`: their coins are independent.
- **The plug-in** reads `K = max(H, ceil(3 log T))` lags. `h = T_g 1 + 2 e` is read as
  `2 y' - z * g_hat`, which equals `1 * g_hat + 2 (y' - x * g_hat)`. `M = T_q' T_g` is banded,
  so eq. (2.5) is summed over its `H + K - 1` diagonals, `O(T H K)`, not `O(T^3)`.
- **It refuses** what it cannot read: another analysis than the local projection, the steady
  state, a lever that is not i.i.d. (the local projection's lag-one test at `5 / sqrt(n)`), a coin
  that is not fair (`|2 u_bar - 1| sqrt(n) > 5`, the same five standard errors), a run of at most
  `2K` periods, and an outcome that never moves off its first reading.
- **`validation/switchback_design_variance.mac`** derives it:
  - STEP 1: over all 16 designs at `T = 4`, symbolic `g` and `e`, `H = 1, 2, 3`, the mean is
    `tau_H` and the variance is eq. (2.5); `e` and the baseline enter only as `e - c`;
  - STEP 2: the channel's variance has no `e_t e_s` term at `t != s`, and `tau_H`'s only those
    with `|t - s| < H`, the readings one lag sums. None is an expectation over the noise, so the
    noise's autocorrelation does not enter;
  - STEP 3: for any `T`, the estimate is `(z'Mz + z'l)/T`, and the variance of that Rademacher
    chaos is `sum_(i != j)(M_ij^2 + M_ij M_ji) + |l|^2`, checked on a generic `M` and `l` at
    `n = 3, 4, 5`; the mean `tr(M)/T` is `tau_H` for every `T`;
  - STEP 4: the plug-in's noise identity.

## Consequences

Measured by `tests/test_switchback_design.py` and the runs behind its comments, 1000 runs a case,
95 % intervals, `0.95 +- 0.028` at four standard errors:

| case | model: covered | design: covered | width, model | width, design |
|---|---|---|---|---|
| `tau_3`, AR(0.8) shocks, `T = 40` | 0.881 | 0.977 | 7.23 | 27.98 |
| `tau_3`, AR(0.8) shocks, `T = 60` | 0.903 | 0.969 | 6.20 | 23.52 |
| `tau_3`, AR(0.8) shocks, `T = 100` | 0.905 | 0.948 | 4.95 | 19.04 |
| channel, one trending noise path, `T = 200` | 0.919 | 0.960 | 0.18 | 16.12 |

- **The design's interval holds its level** where the model's does not, on a noise that persists
  over a short run and on a noise that trends. It errs on the safe side on the shortest runs:
  0.977 over 40 periods and 0.969 over 60, within four standard errors of 0.95. Its plug-in reads
  `h` with the error of `2K` estimated lags in it, which widens it.
- **The model's refusals** are counted out of its coverage: it refused 24 of the 1000 runs over 40
  periods and 5 over 60, whose lag terms summed to a negative variance. The design refused none.
- **It pays in width**: 3.8 to 6.6 times the model's on these AR(0.8) shocks, and 90 times on the
  trend. The local projection controls for `y_t` with an intercept; the moment estimator uses
  neither, so everything in the outcome the lever did not cause stays in its variance. Where the
  model's interval holds, it is the better reading: over 200 and 1000 periods of the same shocks
  both covered 0.947 to 0.973 (an exploration of 300 runs a case), the design's at 3.8 to 6.6 times
  the width, the larger for the channel.
- **The plug-in's lag rule hardly matters here**: at `a = 0.95`, `T = 200`, `K = 1, 3, 6 log T`
  covered 0.948 to 0.965 (an exploration of 1000 runs a case). Eq. (2.5)'s noise term dominates.
- **The tests** pin:
  - the estimate against its transcription on every design of a run of 10, and its mean and
    variance over all 1024 against eq. (2.5), to `1e-12`;
  - the standard error on every design against the plug-in transcribed over full matrices, to
    `1e-10`, at an `H` whose `tr(M M)` reads `g` past the lags its other terms read;
  - the banded sum against the dense eq. (2.5), with `g` and `g_long` of different lengths;
  - the coverage of both intervals in the four cases' first two kinds;
  - a zone's level, the default's identity, every refusal and the log record (`se_over`).

## Not built

- **A baseline that moves with the run.** Subtracting the reading just before each lever,
  `y_t`, is still unbiased: it depends on earlier coins only, which `z_t` is independent of. It
  would remove most of a persistent noise from the variance, and the width with it. But eq. (2.5)
  no longer holds, and its variance is not derived here.
- **Spillover between zones.** A zone's outcome moved by another zone's coins puts terms in the
  variance that eq. (2.5) does not hold. The block difference's standard error keeps them.
- **Markov and block designs.** Eq. (2.5) is for independent fair coins. Those designs keep
  `randomisation_test`.

## Alternatives considered

- **Keep the local projection's estimate and give it a design variance.** Rejected: its
  coefficient is a ratio with `y_t` in its regressors, and its variance over the coins has no
  exact form.
- **Liang and Recht's own `q = 1_(<H)`.** Rejected: its target is `sum_(s<H) (T - s)/T g_s`, which
  is `tau_H` less `sum_s s g_s / T`. At `q_s = T/(T - s)` the target is `tau_H` exactly, and
  eq. (2.5) holds as it stands.
- **Centre each zone on its own mean.** Rejected: the mean moves with the coins, so the estimate
  is biased by `O(1/T)` and eq. (2.5) no longer holds. The first reading precedes the coins.
- **A new analysis, `"moment"`, in place of `se`.** Rejected: `SwitchbackAnalysis` names what
  `design_switchback` plans, and the moment estimator is the local projection read by the design's
  moments.
