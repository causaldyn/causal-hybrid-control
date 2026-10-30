# ADR 0148 — A designed test's intervals may carry the AR(1) coefficient's error

**Status:** accepted, 2026-10-30. Experimental. Amends ADR 0146 (a designed test's intervals read
its innovations). The default output of `blank_period_inference`, in either reading, does not
change.

## Context

`chc.scm.blank_period_inference(..., serial="ar1")` forms each post period's interval as the
prediction interval of a `k`-step forecast of an AR(1) with a known coefficient and an unknown
level (ADR 0146). The coefficient is not known: it is Han, Phillips and Sul's panel fully
aggregated estimator (Econometric Theory 30, 2014) over the donors' synthetic-control gaps, the
treated unit left out, and the intervals plug it in. With few donors the estimate is biased low,
and its error enters the centre `c_k = u_k - rho^k u_0 - m a_k` through `rho^k`, the level's sum
`a_k` and the innovations' mean `m`, so coverage falls with the post period. On factor panels
whose factors and errors are AR(1) at 0.95, 15 fitting, 5 blank and 10 post periods, at
`alpha = 0.1`, 2000 panels a case (`scripts/bench_scm_inference.py blank_rho`), the plug-in
intervals covered 0.890 at the first post period falling to 0.852-0.856 at the last on 4 donors,
and 0.912 falling to 0.884-0.886 on 19, where the true coefficient gives 0.884-0.896 and
0.900-0.910. Chernozhukov, Wüthrich and Zhu (JASA 2021, §2.4.2) filter a synthetic control's
residuals by an AR(1) as ADR 0146 does, and Abadie and Zhao (arXiv 2108.02196) justify reading a
designed test from its blank periods; neither carries an estimated coefficient's error into an
interval.

## Decision

- **`blank_period_inference(..., ar_error="plug-in" | "jackknife")`**, `"plug-in"` by default,
  read only under `serial="ar1"`. The name pairs with the result's `ar_coefficient` and
  `ar_check_p_value`: it says how that coefficient's error is read. `"plug-in"` is today's
  reading; `"jackknife"` is refused with the exchangeable reading and with fewer than three donors.
- **The jackknife.** Each donor is left out in turn, the remaining donors' gaps are rebuilt from
  each other alone, each from a synthetic control of the others on the first `ceil(T_fit / 2)`
  periods, and the coefficient is read again; the variance is `(n - 1) / n` times the left-out
  estimates' squared deviations from their mean (Efron and Stein, Annals of Statistics 9, 1981).
  The treated unit never enters it. `BlankPeriodInference.ar_coefficient_variance` records it,
  `None` under the plug-in reading.
- **The widening.** The delta method on each centre: `dc_k / drho = -k rho^(k-1) u_0 + lbar a_k
  - m a_k'`, with `lbar` the mean of the blank innovations' lagged gaps, since `m = mean(u_t) -
  rho mean(u_(t-1))` moves with `rho`, and `a_k' = sum_(j<k) j rho^(j-1)`. Its square times the
  jackknife variance is added to the forecast's variance under the same Student t quantile:
  radius `= hypot(t s sqrt(v_k + a_k^2 / T_B), t |dc_k / drho| sqrt(V))`, which is the plug-in
  radius to the bit at `V = 0`. `validation/scm_ar1_rho_error.mac` derives the derivative,
  checks the jackknife formula on a sample mean, and pins the widened intervals at a sample path.
  The centres, the p-value and every other output stay the plug-in's.
- **Opt-in.** The default output is bit-identical; a test digests every output of the plug-in
  reading as the previous release produced it.

## Consequences

- On the panels above, at `alpha = 0.1`, the jackknife covered 0.895-0.912 at every post period
  on 4 donors and 0.903-0.916 on 19; at `alpha = 0.05` on 4 donors, 0.949-0.954 where the plug-in
  fell to 0.916-0.918. The median radius at the last post period rose 13% on 4 donors and 5% on
  19.
- It overstates the error. Its standard error averaged 0.116 on 4 donors and 0.061 on 19, where
  the estimate's spread across panels was 0.105 and 0.046 about means of 0.893 and 0.916; the
  delete-one jackknife's variance is biased upward in expectation (Efron and Stein's
  inequality), and leaving a donor out also removes it from every other donor's synthetic control.
  On 19 donors the intervals then cover above `1 - alpha` by up to 1.6 points at the last post
  periods, 3.2 points above the plug-in's.
- It widens and does not move the centres, so the estimate's bias stays in them.
- It costs `n_donors - 1` times the plug-in reading's synthetic-control fits of the donors.
- It refuses 2 donors: their gaps are one series and its negative, each donor fitted by the other,
  so leaving either out leaves no gaps, and no resampling of the two sees the estimate's error.
  The plug-in reading's 2-donor shortfall, 0.903 falling to 0.830-0.836, stays.

## Alternatives

- **Dropping each donor's row of gaps without rebuilding the others'.** Cheaper, and at 2 donors
  it gives 0, since the two rows are each other's negative. It treats the rows as independent
  where every row is built from the other donors, and it understated the error: on the first 400
  panels with 4 donors its variance averaged 0.0041, the rebuilding jackknife's 0.0204, where the
  estimate's mean squared error about the true coefficient was 0.0125. On the same 2000 panels at
  `alpha = 0.1` it covered 0.868-0.892 on 4 donors, short of 0.90 by more than two standard
  errors at the late post periods, and 0.894-0.912 on 19.
- **A donor bootstrap.** Resampling donors with replacement repeats a donor, and a repeated donor
  fits itself exactly in the others' synthetic control, which zeroes its gaps; a bootstrap over
  the gap rows has the row jackknife's dependence problem. Either needs more replicates than the
  jackknife's `n_donors` to settle.
- **A bias-corrected estimate.** The jackknife's own bias correction, `n theta - (n - 1)
  mean(theta_(i))`, or a half-panel split, would move the centres toward the true coefficient's,
  but it changes the coefficient the test filters by, and so the test's p-value, and it is noisier
  than the estimate it corrects; an estimate outside `[-1, 1]` would have to be clipped again. It is a separate
  decision from carrying the error.
- **Jackknifing each centre directly**, `c_k(rho_(i))`, in place of the delta method. It keeps the
  centre's curvature in `rho`, at the cost of a variance per post period where the delta method
  states its widening through one variance, of the coefficient, that a caller can read.
