# ADR 0106 — An effect's error is HC3, and a refutation reads its own

**Status:** accepted, 2026-10-19.

## Context

Three outputs read the noise of a linear regression of the outcome on the action and its
covariates:

- `BackdoorOLS` and `IV2SLS` report the effect's standard error. Up to 0.18.0 `BackdoorOLS` read
  the classical `sigma^2 (X'X)^-1`, which takes the noise's variance as one across rows, and
  `IV2SLS` reported none. Before this release both moved to HC1.
- `sensitivity_analysis` reports a standard error, an E-value interval read off it, and the
  Cinelli–Hazlett robustness value. It read the classical error, and no doc said what it assumes.
  On 4000 logs of 200 rows whose noise grew with the action's own part, its 95 % interval missed
  the effect 22.5 % of the time.
- `refute_effect.passes` asked a placebo to fall within 10 % of the effect, and two refits to
  move it by less than 10 % and 15 %, whatever the log's size. A right linear model failed on its
  noise alone: 31.0 % of 1000 logs of 200 rows, 76.0 % at 50 rows, and 1.0 % at 1000 rows.

The size of a 95 % Wald interval on 4000 logs a cell (seeded, Monte Carlo SD 0.3–0.5 points):

| design | rows | classical | HC1 | HC3 |
|---|---|---|---|---|
| OLS, one noise variance | 100 / 200 / 1000 | 5.0 / 5.5 / 4.9 % | 5.8 / 5.7 / 4.8 % | 4.8 / 5.3 / 4.8 % |
| OLS, variance grows with the action | 100 / 200 / 1000 | 22.9 / 22.5 / 21.7 % | 7.2 / 6.3 / 5.4 % | 5.8 / 5.6 / 5.3 % |
| OLS, t(3) action, noise SD `|a|` | 100 / 200 / 1000 | 50.3 / 52.6 / 63.8 % | 15.7 / 11.9 / 8.7 % | 8.1 / 6.6 / 6.1 % |
| 2SLS, Gaussian instrument | 100 / 200 / 1000 | — | 5.6 / 5.1 / 4.9 % | 4.8 / 4.7 / 4.7 % |
| 2SLS, t(3) instrument | 100 / 200 / 1000 | — | 12.6 / 11.1 / 8.8 % | 7.0 / 7.0 / 5.8 % |

## Decision

- **`BackdoorOLS` and `IV2SLS` read HC3** (MacKinnon and White 1985): each residual over one less
  its row's leverage. For 2SLS the leverage is read on the second stage's design, the instrument's
  push in place of the action. HC3 is at or below HC1's size in every cell above. It holds where a
  few rows carry high leverage, where HC1 does not. The move from the classical error was already
  in this release, so HC3 replaces HC1 outright and adds no option.
- **`sensitivity_analysis(robust=)`**, opt-in. `robust=True` reads `std_error` and `e_value_ci`
  HC3. The default, `None`, reads the classical error as before and emits a `FutureWarning` that
  0.20.0 reads HC3. The robustness value keeps the classical `t` either way. Cinelli and Hazlett's
  closed form is the action's partial `R^2` with the outcome, `t^2 / (t^2 + dof)`, which holds
  for the classical `t` alone.
- **`refute_effect` reads each check against its own sampling error.** The placebo's estimate and
  the two refits' changes are each linear in the outcome. Each gets the standard error of its
  weights on the rows (`jax.grad` through least squares), with each row's own squared residual at
  the factor `N / (N - k)`. `passes` asks all three to lie within the `1 - alpha / 6` quantile of
  `t(N - k)` times their errors (Bonferroni over the three), `alpha=0.05` by default. The report
  carries the three errors.

## Consequences

- `BackdoorOLS.std_error` and `IV2SLS.std_error` move from HC1 to HC3: wider by up to the
  leverage's share, a few per cent on Gaussian designs of 100 rows or more.
- `sensitivity_analysis` warns at its default until 0.20.0. A caller that wants no warning passes
  `robust=False` or `robust=True`.
- A right model fails `refute_effect` 4.4 % of 1000 logs of 200 rows, 3.6 % with heteroskedastic
  noise, 4.9 % at 50 rows, 6.0 % and 3.0 % at 1000 rows. A model that leaves out the confounder
  or reads it in the wrong form fails about as often: 5.4 %, 6.4 % and 6.0 %. The checks read
  stability, not identification, and the docstring says so.

## Alternatives

- **HC1 throughout.** It is the review's choice and Stata's default. Rejected: under high leverage
  it missed 15.7 % where HC3 missed 8.1 %, and HC3 cost nothing where the noise's variance is one.
- **HC2.** Between the two in every cell. HC3 is the one Long and Ervin (2000) recommend below 250
  rows, and it was closer to 5 % here.
- **HC3 in `refute_effect`.** A refit's change is a difference of two estimates' weights. It has
  no one design whose leverage scales its residuals, and the measured size was already at 5 %.
- **Keep the 10 %/15 % shares and document them.** Rejected: the verdict then depends on the log's
  size more than on the model.
