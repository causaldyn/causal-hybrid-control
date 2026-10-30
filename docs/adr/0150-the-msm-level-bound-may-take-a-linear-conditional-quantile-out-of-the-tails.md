# ADR 0150 — The MSM level bound may take a linear conditional quantile out of the tails

**Status:** accepted, 2026-10-30. Builds the quantile regression that ADR 0143 lists as "Not
built", opt-in and experimental. No default moves: every existing output is the same to the bit.

## Context

`msm_worst_case_mean`, `confounding_robust_radius` and `negative_control_gamma` take
`covariates=` and read the MSM bound on a mean as `mean(Y) + inflation(Y - m(X))`, `m` a
cross-fitted least-squares prediction (ADR 0143). That is valid for any `m`, since
`E[w m(X)] = E[m(X)]` when `E[w | x] = 1`, and sharp where the residual's law does not move with
`x`. Where the spread of `Y | x` moves with `x`, the residual pools tails the per-`x` bound reads
apart, and the bound is wider than the sharp one. At `Gamma = 2`
(`validation/msm_quantile_balancing.mac`, Q4):

| population | residualised over per-`x` inflation |
|---|---|
| two equal cells, `N(0, 1)` and `N(0, 9)` | 1.0234 |
| two equal cells, `N(0, 0.09)` and `N(0, 9)` | 1.0640 |
| two equal two-point cells, `m_x -+ s_x`, `m = (0, 4)`, `s = (1, 3)` | 1.25 |
| `Y = 2X + (1 + 2X) eps`, `X ~ U(0, 1)` | bounds 2.0998 against 2.0908 |

Dorn and Guo (JASA 2023) give the sharp form: for any `q(x)` and a weight in `[lo, hi]` with
`E[w | x] = 1`, `E[w Y] <= E[psi]`, `psi = q + lo (Y - q) + (hi - lo) (Y - q)_+`, with equality at
the quantile of `Y | x` at level `Gamma / (Gamma + 1)`. The level is the same for the box
`[1/Gamma, Gamma]` and for Tan's (JASA 2006) box at every propensity
(`validation/msm_level_bound.mac`, L1a).

## Decision

- **`covariate_form="quantile"`** on the three functions, `"residualised"` the default. It needs
  `covariates` and is refused without them; it reads `propensity` as the default does. The
  keyword's entry in each docstring marks it *experimental*, and `docs/api/index.md` lists the
  three.
- **The fit.** `mean(psi) - mean(Y)` is the pinball loss at `Gamma / (Gamma + 1)` weighted by
  `hi - lo` (the Maxima file's Q2), so `q = (1, x)' beta` minimising it is a weighted quantile
  regression (Koenker and Bassett, Econometrica 1978), a linear program in `(beta, u, v)` with
  sparse equality rows, solved by HiGHS through `scipy.optimize.linprog`. Its costs are scaled so
  the least is 1, since HiGHS cycles on degenerate programs whose costs lie below 0.1. A status
  other than optimal raises `RuntimeError`, as `chc.allocation`'s programs do.
- **Cross-fitted, and read on the held units.** Each unit's `q` is fitted on the 5 interleaved
  folds it is not in, the folds the least-squares form uses. The bound is `psi`'s mean on the held
  units and never the program's optimum: `psi` bounds the mean for any `q`, so the solver's
  tolerance costs sharpness and not validity, and the optimum on the fitted rows reads narrower
  than the population's bound.
- **At `level`** the endpoint's standard error is `psi`'s over `sqrt(n)`: the quantile's own
  error moves the endpoint at second order (Dorn, Guo and Kallus, JASA 2025), as the CVaR's
  threshold does for the pooled endpoint (Rockafellar and Uryasev 2000).
- **`negative_control_gamma` refits at each `Gamma` the bisection visits.** `psi` grows with
  `Gamma` at a fixed `q`, but `q` moves with `Gamma`, so the endpoint's monotonicity is checked on
  a grid, not proved. Over 60 samples of 300 outcomes on the `U(0, 1)` population and ten values of
  `Gamma` from 1 to 100, no step fell.
- At `Gamma = 1` the form returns the mean and solves nothing.

## Consequences

- **The gain, measured** at `n = 2000`, `Gamma = 2`, 200 samples, the residualised bound less the
  quantile form's, paired, with its Monte Carlo standard error:

  | population | quantile form | residualised | gap | MC SE of the gap |
  |---|---|---|---|---|
  | two-point cells, `s = (1, 3)`, per-`x` 3 | 3.0028 | 3.2554 | 0.2526 | 0.0010 |
  | `N(0, 0.09)` / `N(0, 9)` cells, per-`x` 0.8999 | 0.8999 | 0.9566 | 0.0567 | 0.0005 |
  | `Y = 2X + (1 + 2X) eps`, per-`x` 2.0908 | 2.0896 | 2.0981 | 0.0085 | 0.0002 |

- **Its price where the spread does not move.** On `Y = 1 + 2X + eps`, `n = 1000`, both forms
  read within 1.8 Monte Carlo standard errors of the per-`x` bound 2.5454 on 6 seeds of 40
  samples, and the quantile form sat 0.0006 to 0.0009 above the residualised one: its intercept is
  cross-fitted where the residualised form's threshold is read in sample.
- **Its price in fitting.** Each fold fits `p + 1` coefficients out of sample. With ten
  covariates that carry nothing and 60 outcomes the cross-fitted bound averaged 0.653 to 0.673
  over 3 seeds of 200 samples, above the population's 0.545, as it must in expectation; a fit on
  the rows it is read on averaged 0.451 to 0.467, 8 to 10 standard errors below it.
- **Valid where the quantile is not linear.** With `sd(x) = 0.5 + 4 (x - 1/2)^2` on a single
  column the form stays at or above the per-`x` bound; the columns must carry the basis
  expansion, as they must for the residualised form.
- **Its level.** Whether `negative_control_gamma(level=0.95)` reads above 1 is settled at
  `Gamma = 1`, where both forms' endpoint is the mean's `t` bound. On `Y = (1 + 2X) eps`, 200
  outcomes, 1000 samples a seed, it read above 1 on 5.9 % and 4.5 %.
- **Its cost** is 5 linear programs of `n` units each, one a fold, at each `Gamma`: one `Gamma`
  for `msm_worst_case_mean` and `confounding_robust_radius`, and about `log2(gamma_max / tol)`
  for `negative_control_gamma`, 50 at the defaults, so 250 programs.
- In sample, on an intercept alone, the form equals the box's closed form and the fill of Tan's
  boxes to 1e-12 relative: HiGHS returns a vertex, and a vertex's `q` is an order statistic. On
  cell dummies it is the cells' closed forms weighted by their shares.

## Alternatives

- **The quantile form as the default.** Rejected: the 1.0 outputs are frozen, the form costs
  linear programs where the default costs a sort, and where the spread does not move with `x` it
  buys nothing and reads slightly wider.
- **Nonparametric conditional quantiles** (kernel, series of a chosen basis, forests), with which
  Dorn, Guo and Kallus's bound is sharp without a linear quantile. Not built: each needs a tuning
  rule and most a dependency the library does not carry. `psi` is valid for any `q`, so a caller's
  basis expansion in `covariates` buys sharpness at no cost in validity, and a quantile model the
  form does not fit leaves it valid.
- **Reporting the program's optimum.** Rejected: read on the rows it was fitted on, it is narrower
  than the population's bound, measured above, and it depends on the solver's tolerance.
- **One fit across `Gamma` for `negative_control_gamma`.** Rejected: the quantile's level moves
  with `Gamma`, and a quantile fitted at one level is sharp at that level only.
- **Per-cell tails for `propensity` alone.** ADR 0143 notes that with `e` varying the fill pools
  the residual's tails across cells. With the propensity's cells among `covariates`, the quantile
  form reads them per cell.
