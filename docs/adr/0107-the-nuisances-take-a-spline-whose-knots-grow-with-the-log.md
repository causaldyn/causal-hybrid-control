# ADR 0107 — The nuisances take a spline whose knots grow with the log

**Status:** accepted, 2026-10-19. Opt-in; the default is unchanged. Amended 2026-10-19: the
default fit refits once with `"auto"` and warns where that moves its channel by more than three
errors.

## Context

`fit_causal_residual` partials the state and the actions out of the rate with cross-fitted ridge
nuisances, polynomials of fixed degree `nuisance_degree` (2) in the standardised covariates. The
moment is orthogonal, so the nuisances' error moves the channel at second order: the bias is the
product of the two nuisances' approximation errors. A policy or a drift outside the polynomials'
span leaves errors that do not shrink with the log, and so a bias that does not shrink either.
`channel_error` does not cover it.

On a lever set by `-1.5 tanh(2 x)` and a drift with `sin(2 x)` in it (channel 1), the
population bias of the moment is the product of the two projections' residuals under the state's
law. By Gauss–Hermite quadrature it is −0.625 at degree 2, −0.396 at 3, −0.170 at 5, −0.058 at 7
and −0.016 at 9. Measured, degree 2 read −0.615 at 1000 rows and −0.624 at 4000, 26 and 57 of its
errors, its 95 % interval covering on none of 12 logs.

A higher degree does not cure it. A polynomial of high degree fitted on one fold runs off in the
other fold's tails. On 20 logs a cell (`nuisance_degree` 5 / 7 / 9 / 11):

| rows | bias | MC SD / mean error | coverage |
|---|---|---|---|
| 4000 | −0.224 / −0.147 / −0.065 / +0.039 | 2.26 / 2.23 / 0.74 / 0.41 | 0.10 / 0.75 / 1.00 / 1.00 |
| 16000 | −0.184 / −0.108 / −0.056 / +0.043 | 1.27 / 2.01 / 1.96 / 0.73 | 0.10 / 0.40 / 0.85 / 1.00 |

The channel scattered up to 2.3 times as far as its error said, or its error grew to 2.5 times
its scatter. No degree held the 95 %.

## Decision

- **`nuisance_knots`**, opt-in, experimental. It adds a natural cubic spline in each covariate to
  the polynomial of `nuisance_degree`, at that many of the covariate's quantiles. `"auto"` takes
  `ceil(N^(1/3))` knots, at least 3. `None`, the default, adds none.
- A natural cubic spline is linear beyond its outer knots, so a fold's predictions do not run off
  in the other fold's tails. Its error at a smooth nuisance shrinks as the knots grow. At
  `N^(1/3)` knots the product of the two nuisances' errors shrinks faster than the moment's
  `N^(-1/2)`, so the bias shrinks faster than the error.
- The basis is the polynomial's columns and the splines' (truncated powers, Hastie, Tibshirani and
  Friedman 2009, eq. 5.4–5.5), spanned by the full log's left singular vectors at unit mean
  square. The truncated powers are near collinear: the raw columns' condition number was 1.4e6 on
  16000 rows of two covariates at 26 knots. The orthonormal basis has the same span and condition
  number 1 on the log, so a Gram carries it in float32. Full-sample statistics, as the
  standardisation's, so every fold reads one basis.
- The sieve stays a linear smoother of the target. The `rk4` fixed point, the error through the
  cross-fit and the influence read it as they read the polynomial.
- A covariate with fewer than three distinct knots, a flag or a constant, adds no spline.
- **`nuisance_shift(fit, refit, states)`**, experimental: how far a refit moves the channel, the
  root mean square of `B_refit(x) − B_fit(x)` over the log's states and the channel's entries, in
  the fit's `channel_error`. A shift of more than one or two says the fit's nuisances miss.
- **The default fit reads its own shift.** Left at `None`, an identified, unweighted fit without an
  instrument refits the log once with `nuisance_knots="auto"` and reads `nuisance_shift` over the
  log's states. Above three errors it warns (`FutureWarning`) that its nuisances miss the log's, and
  that 0.20.0 reads `"auto"`. The fit it returns is the default's, bit for bit.

## Consequences

- On the plant above, with `nuisance_knots="auto"`:

  | rows | knots | bias | bias / error | MC SD / mean error | coverage (logs) |
  |---|---|---|---|---|---|
  | 1000 | 10 | +0.0036 | +0.18 | 1.13 | 0.955 (200) |
  | 4000 | 16 | +0.0003 | +0.05 | 0.96 | 0.970 (200) |
  | 16000 | 26 | +0.0005 | +0.18 | 0.89 | 0.950 (100) |

- On a linear plant, where the polynomial holds the truth, the spline costs a little variance and
  no coverage: on 200 logs of 1000 rows the interval covered on 0.950, where the polynomial alone
  covered on 0.965, and the channel scattered 11 % wider (0.0070 against 0.0063).
- At 4000 rows of the plant above, the degree-2 fit read 53 to 61 errors from its refit with
  `"auto"` on five logs. The `"auto"` fit read 0.02 to 1.9 from its own refit at twice the knots.
- A fit costs little more: 0.32 s at 1000 rows of the linear plant, against 0.30 s at degree 2;
  1.2 s at 4000 rows and 2.0 s at 16000 on the plant above.
- The spline is additive. A confounder that enters through a nonlinear interaction of two
  covariates is outside the span beyond the polynomial's interactions, and the bias then need not
  vanish. `nuisance_shift` against a fit of higher degree reads it.
- `prescribe` still fits at degree 2 with no spline. Its default, and the fit's, are the author's
  call for 0.20.0: moving either moves every channel a little.
- The default's warning separates the two cases. On the plant above, 100 units of 20 periods, the
  refit moved the channel by 5.8 to 10.4 errors and the fit warned on 40 of 40 logs, where it
  covered on none. On 40 logs each of a linear plant with independent, heteroskedastic and AR(0.8)
  noise, and 20 of the quickstart's panel, the shift was 0.89 errors at most, and it warned on none.
  On one serial log of a linear plant, 200 logs at each length, the shift passed three errors on 2 %
  of logs at 100 rows and on none at 200 or 500; at two, the threshold first tried, it warned on
  6 %, 2 % and 0.5 % of them. Where the noise grew as `exp(x)` it warned on 3 % of 40 logs; under a
  weight `exp(-x)` the shift passed three on 23 %, and under `exp(x)` on 5 %, where the fits covered
  at 1.00 and 0.93: a weight that loads a few rows moves the spline's own second-order error past
  the channel's, so a weighted fit is not checked. The check costs one more fit of each such default
  fit, and of each identified `prescribe`, which about doubles the time of either.

## Alternatives

- **A degree chosen by cross-validation, growing with `N`.** Rejected: a polynomial of high
  degree on Gaussian covariates extrapolates out of fold (table above). Cross-validation would
  pick a degree whose bias stays tens of errors at 4000 rows.
- **Random Fourier features, bandwidth and ridge by cross-validation.** Tried in a prototype:
  at 4000 rows the bias was −0.012 and the channel scattered three times its sandwich error. The
  frequencies are drawn per log, and their approximation error is a bias that no error conditional
  on them reads.
- **A pluggable learner with `fit` and `predict`.** Rejected for now: the fixed point and the
  error read the nuisances as a linear smoother through JAX, which a general learner is not. A
  pluggable feature map is the form that would fit, if a caller needs one.
- **Tensor-product splines.** They hold interactions, but their columns grow as `K^d`.
- **A `FutureWarning` on every default call.** Rejected: it says nothing about the log in hand,
  and it would fire at about 210 calls of `fit_causal_residual` and 210 of `prescribe` in the
  tests, and at every call in the docs and notebooks. `prescribe` takes no `nuisance_knots`, so
  its callers could not silence it.
