# ADR 0163 — A lift fit may read its AR(1) coefficient median-unbiased

**Status:** accepted, 2026-11-24. Experimental (`chc.lift`). Amends ADR 0126 (a lift fit reads
each test's noise as an AR(1) by default). The default, `ar_estimate="least_squares"`, does not
change, and no value it reads moves.

## Context

`fit_lift(serial="ar1")` reads each test's noise as an AR(1) of one coefficient `rho` and fits the
gaps through the Prais-Winsten transform at it. `rho` is the least-squares AR(1) coefficient of the
residuals, each test's periods on their previous one, read again after each refit until it settles
(iterated Cochrane-Orcutt). It reads low: the fit has spent degrees of freedom on the residuals,
and a test's run is short. `fit_lift`'s docstring stated the bias and left it: Kendall's
`(1 + 3 rho) / T`, a correction for a series about its own mean, adds 0.21 at 0.8 where the
reading is 0.023 low. Nothing gave an interval for `rho` itself.

Andrews (*Econometrica* 61, 1993) reads an AR(1) coefficient exactly median-unbiased: where the
law of a statistic depends on `rho` alone and its median rises with `rho`, the `rho` whose law has
the observed statistic as its median is below the truth with probability one half, and its
quantiles give an exact interval. He did it for an AR(1) with a constant and a trend. A lift fit
is a regression with AR(1) errors, and the same argument holds through its residual maker:

- The independent fit's residuals are `r = M nu`, `M = I - J (J'J)^-1 J'` the residual maker of
  the fit's Jacobian `J` and `nu` the noise: exactly for a model linear in its parameters, to
  first order on a curve.
- The least-squares coefficient is `r'Ar / r'Br`, `A` the lag-one products and `B` the lagged
  squares within each test. As `r'Br > 0`, `P(rho^ <= c) = P(nu' M (A - c B) M nu <= 0)`.
- With `nu = L e`, `L` the inverse of the Prais-Winsten transform at `rho` (`W S W' = I`), that is
  `P(sum_k l_k chi2_1 <= 0)`, the `l_k` the eigenvalues of `L' M (A - c B) M L`, of both signs: a
  law of `rho` alone, which Imhof's (*Biometrika* 48, 1961) inversion reads at 0.

The iterated Cochrane-Orcutt reading cannot be inverted so: its residual maker is the
generalised one at its own estimate, so its law is no fixed quadratic form.

`chc.uncertainty._chi_square_mixture_survival`, the library's Imhof inversion, read positive
weights only: it dropped every weight below `1e-12` of the largest and returned 1 at `q <= 0`.

`validation/lift_rho_median_unbiased.mac` holds the identities: the transform whitens the AR(1)
(`T = 4`, symbolic), the two quadratic forms are the statistic's sums, Imhof at `q = 0` with
weights of both signs against the `F(1, 1)` closed form and at `q` of either sign against the
convolution, the Cauchy law of two periods (`P(rho^ <= c) = 1/2 + atan((c - rho) /
sqrt(1 - rho^2)) / pi`, median `rho`), the closed form where two residual degrees of freedom are
left, and six weights by Imhof's integral there. It passes in Maxima 5.46, 5.49 and 5.50.

## Decision

- **`fit_lift(ar_estimate="median_unbiased")`**, opt-in, under `serial="ar1"` only (a
  `ValueError` otherwise). The coefficient is the `rho` at which the law above has the
  least-squares coefficient of the independent fit's residuals as its median, found by Brent's
  method on `[-0.99, 0.99]`, the range the transform is held to; the gaps are then fitted once
  through the transform at it, with no further rounds, which would bring the bias back.
- **`LiftFit.ar_interval`**, the exact equal-tailed interval at the fit's `level`, the `rho` at
  which the statistic is the law's `(1 + level)/2` and `(1 - level)/2` quantiles; an end at -0.99
  or 0.99 is open on that side, as a parameter's end at its bound is. `None` under
  `"least_squares"`.
- **The intervals of the channel's parameters and `check_observational` take the coefficient as
  known**, as they did.
- **`_chi_square_mixture_survival` reads weights of either sign.** The largest weight and the
  cut-off are taken in absolute value; at `q = 0` the integrand does not oscillate, and the head
  runs a decade at a time to 1000 over the smallest weight, the rest integrated in units of that
  cut (QUADPACK's map of `[cut, inf)` to `(0, 1]` has a scale of 1, and on weights -0.0038 and 0.75
  it read -1.4e-13 for 1.8e-8); below `q = 0` it is one less the survival of `-q` under the negated
  weights. The amplitude is the product of `(1 + w^2 u^2)^(-1/4)`, which underflows to 0 where the
  product of their inverses overflows on many weights (two hundred over nine decades passed the
  largest float at `u = 1000 / w_min`). Every caller passes non-negative weights; their values
  move by the rounding of that product only.

## Consequences

On a model linear in its parameters the reading is exact. Four tests of 16 periods, three fitted
columns (a go-dark carryover shape, its slope in the retention and its square, each test's scaled
by `1 + 0.3 j`), stationary AR(1) noise, 2000 draws a coefficient from
`numpy.random.SeedSequence([20261124, 16, k])`, `k` the coefficient's place in the list, read
through the library's own inversion; "low" is the share of draws below the truth (Monte Carlo SE
0.011), the 90% interval's SE 0.0067:

| `rho` | median bias, least squares of the residuals | iterated (the default's reading) | median-unbiased | low, iterated / median-unbiased | 90% interval covers |
|---|---|---|---|---|---|
| 0 | -0.031 | -0.032 | +0.002 | 0.588 / 0.495 | 0.8995 |
| 0.3 | -0.044 | -0.027 | +0.005 | 0.583 / 0.482 | 0.8975 |
| 0.5 | -0.056 | -0.026 | +0.001 | 0.577 / 0.495 | 0.907 |
| 0.8 | -0.062 | -0.014 | -0.003 | 0.570 / 0.511 | 0.903 |

The misses of the interval fall on both sides alike, 4.3 % to 5.4 % each.

On a curve it is first-order. `fit_lift` itself on `tests/test_lift.py`'s worlds, four go-dark
tests of 16 periods through a `Tanh` channel with a geometric carryover, noise of sd 55 an AR(1)
from its stationary law, worlds 0 to 399 for each coefficient, both readings on the same worlds
(Monte Carlo SE 0.025 for the share low, 0.015 and 0.011 for the 90% and 95% intervals):

| `rho` | median bias, default / median-unbiased | mean bias, default / median-unbiased | low, default / median-unbiased | `ar_interval` covers, 90% / 95% | coefficient's 95% interval covers, default / median-unbiased | check rejects at 5%, default / median-unbiased |
|---|---|---|---|---|---|---|
| 0 | -0.017 / +0.016 | -0.021 / +0.007 | 0.535 / 0.463 | 0.870 / 0.932 | 0.953 / 0.955 | 0.065 / 0.052 |
| 0.3 | -0.018 / +0.011 | -0.025 / +0.000 | 0.542 / 0.482 | 0.887 / 0.927 | 0.955 / 0.958 | 0.055 / 0.052 |
| 0.5 | -0.014 / +0.007 | -0.023 / -0.004 | 0.557 / 0.480 | 0.885 / 0.935 | 0.970 / 0.968 | 0.055 / 0.050 |
| 0.8 | -0.006 / -0.002 | -0.015 / -0.007 | 0.530 / 0.507 | 0.892 / 0.935 | 0.950 / 0.950 | 0.045 / 0.043 |

- On these worlds the default's median is already within 0.018 of the truth; the gain is in the
  mean bias, at least halved (-0.0145 to -0.0071 at 0.8) and gone at 0.3, and in an interval for
  `rho` that did not exist. The
  median-unbiased reading leans high at 0 (+0.016, 1.8 Monte Carlo SEs) and its 90% interval
  covers 0.870 there, 2 SEs short: the residual maker is the curve's tangent, not the curve.
- The channel's intervals and the check barely move: the coefficient feeds them only through the
  transform, and both readings are close to the truth on 16 periods a test.
- `tests/test_lift.py` pins the law to Maxima's values (two periods, the Cauchy law; three periods
  with a ramp fitted, the `F(1, 1)` closed form; two tests of four with a constant and a trend,
  six weights) to `1e-10`, and checks on a fitted world that 40 000 draws of the law put the
  statistic at its median at the estimate and at its 0.975 and 0.025 quantiles at the interval's
  ends.
- A fit under `"median_unbiased"` costs about 40 evaluations of the law (39 to 42 on five of
  the worlds), each one eigendecomposition of a square matrix with a row per period of every test
  and one Imhof integral.

## Alternatives

- **Make it the default.** It moves `ar_coefficient`, and with it every interval and the check, on
  every `serial="ar1"` fit; opt-in first.
- **Kendall's correction**, `(1 + 3 rho) / T`: for a series about its own mean, not for residuals
  of a fitted curve, and it over-corrects here by an order of magnitude at 0.8.
- **Correct the mean bias to first order**, solving `tr(M A M S(rho)) / tr(M B M S(rho)) = rho^`:
  a point estimate with no interval and no stated level.
- **Invert the iterated Cochrane-Orcutt reading.** Its residual maker moves with its estimate,
  so it has no fixed law to invert.
- **A parametric bootstrap of the whole fit.** A refit per draw, and it is median-unbiased only
  as far as its draws read the law the inversion reads exactly.
