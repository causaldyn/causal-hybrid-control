# ADR 0092 — A designed test is read from the periods it left blank

**Status:** proposed, 2026-10-13. Adds to ADR 0027, whose two tests stay as they are. Amended the
same day: a reading for errors that persist, `serial="ar1"`.

## Context

ADR 0027's two tests do not hold when a design chose the treated unit. A design that picks its
test market chooses the unit its donors fit best. On the bench's factor panels, with that unit
treated and no effect, the in-space placebo rejected 27.8% of the time at 5%. The conformal
interval covered 81.7% at 0.10. The placebo needs the treated unit drawn at random. The conformal
test needs the residuals before treatment to be like those after it, and the design selected them
small.

Abadie and Zhao (*Synthetic Controls for Experimental Design*, arXiv 2108.02196 v5, §3) give a
reading that a design justifies. The weights fit on some periods before treatment, `E`. The other
periods before treatment, `B`, stay blank: the weights never see them. The gaps of the treated
unit from its synthetic control, over `B` and the post periods, are then compared.

- **The test (their eq. 19).** The statistic is the mean absolute gap over the post periods. The
  p-value is the share of the `C(T_B + T_post, T_post)` choices of `T_post` periods from `B` and
  the post periods whose statistic is at least the observed one.
- **Theorem 2.** Its conditions are a linear factor model, noise i.i.d. over time for each unit,
  factors exchangeable over `B` and the post periods and independent of the noise, and a perfect
  fit on the observed covariates. `chc.scm` uses no covariates, so the last condition is empty;
  the fit on the outcomes in `E` need not be perfect. Then the gaps over `B` and the post periods
  are exchangeable, and `alpha - 1/|Π| <= P(p <= alpha) <= alpha`, with `|Π|` the count of
  choices.
- **The interval (their eq. 20-21).** It is split conformal: each post period's gap, plus or minus
  a quantile of the absolute gaps in `B`. Their Theorem 3 gives its coverage as `T_B` and `T_E`
  grow.

## Decision

- **`chc.scm.blank_period_inference(outcomes, treated_unit, n_pre, n_blank, *, alpha=0.05)`**
  (*experimental*) returns a `BlankPeriodInference`. Its fields are the `estimate` fitted on `E`
  alone, the `blank_gaps`, the `p_value`, the per-period `intervals`, `alpha`, and the count of
  `arrangements`. No existing function or default changes.
- **`B` is the last `n_blank` periods before treatment.** Then a season or a trend separates `B`
  from the post periods as little as the panel allows. The caller cannot choose other blank
  periods. A free choice of `B` invites a choice made after seeing the data.
- **The p-value is enumerated exactly.** The function refuses more than a million arrangements.
  Abadie and Zhao allow a random sample of them; that would add a seed and a Monte Carlo error to
  a test whose size is its point.
- **A tie counts as a tie.** Two sums of `T_post` non-negative gaps that are equal in exact
  arithmetic can differ by rounding. Sums within `T_post` units of roundoff of the largest count as
  at least the observed one. This keeps the upper bound, which counts ties against the test.
- **The interval uses split conformal's finite-sample rank.** The radius is the
  `ceil((1 - alpha)(T_B + 1))`-th smallest absolute blank gap. It is infinite when that rank exceeds
  `T_B`. Under exchangeability its coverage is at least `1 - alpha`, exactly `rank / (T_B + 1)`
  without ties. Abadie and Zhao's eq. 20 takes the `ceil((1 - alpha) T_B)`-th. With 5 blank periods
  that is the largest at every `alpha < 0.2`, and covers 5/6 = 83.3% whatever `alpha` asked for.
  The product `(1 - alpha)(T_B + 1)` is rounded to nine decimals before the ceiling. Without that,
  `0.3 * 10` reads `3.0000000000000004` and takes rank 4.

- **`serial="ar1"` reads gaps that persist** (amendment). The default, `"exchangeable"`, is the
  test above. With `"ar1"` the function estimates an AR(1) coefficient `rho` of the gaps from the
  donors, and the result carries it as `ar_coefficient`. The test then ranks the innovations
  `e_t = u_t - rho u_{t-1}` over `B` and the post periods; the first lag is the gap in the last
  period of `E`. Its statistic is `|e_first + (1 - rho) (sum of the other post innovations)|`. A constant
  effect `d` adds `d` to the first post innovation and `(1 - rho) d` to each later one, so this is
  the score of a step. The arrangements are a choice of `T_post` innovations and which of them is
  first: `T_post C(T_B + T_post, T_post)`, 1260 for 5 and 5. The intervals do not change.
  - **Why it holds.** When every unit's errors and the factors are AR(1) with one coefficient, so
    is every synthetic control's gap, whatever its weights. Given the periods of `E`, the
    innovations over `B` and the post periods are then i.i.d., and with the true `rho` the test is
    exact. The design may use `E` as it likes. Chernozhukov, Wüthrich and Zhu (JASA 116, 2021,
    §2.4.2 and §4.4.2, Lemma 7) filter a synthetic control's residuals with a fitted AR model
    before they permute them; this is that filter on Abadie and Zhao's blank periods.
  - **Where `rho` comes from.** Each donor gets a synthetic control from the other donors, without
    the treated unit, fitted on the first `ceil(T_E / 2)` periods. Its gaps on every later period
    are out of its fit. Han, Phillips and Sul's panel fully aggregated estimator (Econometric
    Theory 30, 2014, §2, eq. 6-7) pools them: `y_t - y_s` on `y_{t-1} - y_{s+1}` for every pair
    with `t >= s + 3`. That removes each donor's level and leaves the regressor uncorrelated with
    the error. The estimate is kept in `[-1, 1]`, their parameter range. An effect on the treated
    unit cannot reach it.
  - **What its error costs.** An estimate below `rho` leaves the innovations positively
    correlated, and the test rejects too often; above it, too rarely. On the bench's panels, with
    `rho` fixed 0.1 low, the test rejected 6.5%, 5.8% and 6.1% at 0, 0.5 and 0.8; fixed 0.1 high,
    3.6%, 3.8% and 5.1%.

## Consequences

Measured by `scripts/bench_scm_inference.py blank` on the bench's panels: 19 donors, 20 periods
before treatment and 5 after, `T_E = 15` and `T_B = 5`, so `|Π| = 252`. The factors and the errors
are AR(1) with coefficient 0.5 unless stated. There are 4000 panels per case, and the treated unit
is the one its donors fit best on `E` unless stated. Rates only, no timing. The diagnosis below
ran from scratch scripts beside the bench, on the same panel generator.

| case | rejects at 5% | at 10% | interval covers, `alpha = 0.20` |
|---|---|---|---|
| no effect | 8.7% (s.e. 0.4) | 15.3% | 79.6% |
| no effect, unit chosen on all 20 periods | 18.8% | 31.2% | 67.1% |
| no effect, unit drawn at random | 9.7% | 16.6% | 79.9% |
| no effect, factors and errors white noise | 4.4% (s.e. 0.3) | 9.5% | 83.6% |
| effect 1 | 19.7% | 29.9% | 79.0% |
| effect 2 | 44.3% | 56.4% | 79.7% |
| no effect, 10 blank periods, `|Π| = 3003` | 10.7% | 16.9% | 79.3% |

- **Where Theorem 2's conditions hold, the test holds its level.** With white-noise factors and
  errors, it rejected 4.4% at 5%. The bracket is [4.6%, 5%], 0.7 standard errors away. On the same
  kind of panels, 400 of them, ADR 0027's placebo rejected 23.5% with the unit chosen on `E` and
  28.2% with it chosen on all 20 periods.
- **Persistent errors break it; persistent factors do not.** On the bench's own panels it
  rejected 8.7% at 5%, against the placebo's 27.8%: better, not valid. A further 3000 panels per
  case find the cause. With AR(1) errors and white-noise factors it rejected 9.2%. With AR(1)
  factors and white-noise errors it rejected 4.8%. The rate grows with the errors' coefficient:
  5.5% at 0.2, 8.7% at 0.5 and 21.4% at 0.8. The design adds little to it: with the unit drawn at
  random it rejected 7.9% (errors AR, factors white). Three unused periods between `E` and `B`
  left it at 8.9%, and 40 fitting periods at 8.4%. So the cause is not a leak of the fit into `B`.
  It is the post periods' contiguity: under positive serial correlation the mean over five
  adjacent periods varies more than the mean over five scattered ones.
- **A design that looks at `B` breaks the test.** A unit chosen for its fit on all 20 periods has
  blank gaps selected small. The test then rejected 18.8%.
- **The interval needs `alpha >= 1/(T_B + 1)`.** With 5 blank periods every interval at 0.05 and
  0.10 was unbounded. At 0.20 the intervals covered 83.6% where the noise is white, against the
  exact 5/6. On the bench's panels they covered 79.6%. With 10 blank periods they covered 88.5% at
  0.10, against 90.9% under exchangeability, with median width 5.1.
- **`serial="ar1"` holds the level the default loses.** Same panels, 4000 per case, the factors'
  and the errors' coefficient equal; standard errors 0.3-0.4 points at 5%:

  | coefficient | effect | default at 5% | `"ar1"` at 5% | at 10% | `ar_coefficient`, mean (s.d.) |
  |---|---|---|---|---|---|
  | 0 | 0 | 4.4% | 4.9% | 9.6% | 0.00 (0.08) |
  | 0.5 | 0 | 8.7% | 4.8% | 9.4% | 0.49 (0.08) |
  | 0.8 | 0 | 21.5% | 5.3% | 10.2% | 0.77 (0.07) |
  | 0 | 1 | 13.5% | 27.5% | 41.9% | |
  | 0 | 2 | 48.5% | 76.8% | 88.0% | |
  | 0.5 | 1 | 19.7% | 14.9% | 25.8% | |
  | 0.5 | 2 | 44.3% | 45.2% | 60.4% | |
  | 0.8 | 1 | 28.1% | 12.3% | 21.4% | |
  | 0.8 | 2 | 45.3% | 30.3% | 46.8% | |

  The default's power under persistence is bought with its size. At 0.8 a step is hard to tell
  from the noise's own drift. With 10 blank periods, `|Π| = 15015`, the reading rejected 6.1% on
  the bench's 4000 panels and 5.5% (s.e. 0.2) on 12000; the donors fit on 5 periods. With the
  true `rho` the test rejected 5.2% on the same 12000. With the unit drawn at random it rejected
  5.0%. A design that looked at `B` still breaks it: 7.2%.
- **Persistence breaks the intervals too.** At 0.8 the 80% intervals covered 70.7%. `"ar1"` does
  not change them.
  *2026-10-27, ADR 0146:* under `"ar1"` they are the `k`-step forecast's t prediction intervals
  from the blank periods' innovations, and covered 79.6% at 0.8; the field `ar_check_p_value`
  checks the AR(1) model on the donors' gaps.
- **The blank periods leave the fit.** The weights see 15 periods, not 20. The p-value moves in
  steps of `1/|Π|`: 1/252 with 5 blank and 5 post periods, 1/6 with 1 blank period, so then no test
  at 5% rejects.

## Not built

- **A check of the AR(1) model.** Gaps that follow an AR(2), or factors and errors with
  different coefficients, are filtered only in part. `chc.dynamics_id.persistence_check`
  (ADR 0064) reads persistence in a fitted channel, not in a synthetic control's gaps.
  *2026-10-27, ADR 0146:* built, as a portmanteau on the donors' filtered gaps.
- **Intervals for persistent gaps.** One period's effect is not one innovation's, so the
  split-conformal intervals have no filtered form here.
  *2026-10-27, ADR 0146:* built under `"ar1"`, as the `k`-step forecast's prediction intervals.
- **A random sample of the arrangements** (above).
- **Blank periods other than the last ones before treatment** (above).
- **Abadie and Zhao's designs**, which choose treated and control units together as weighted
  averages. `chc.scm` treats one unit; the caller does the design.
- **Their approximate validity for fixed factors (Theorem OA.1).** It holds as `T_E` grows, at a
  rate the bench's 15 periods do not reach.

## Alternatives considered

The numbers below come from scratch scripts beside the bench, on its panels, 4000 per case unless
stated, at coefficients 0, 0.5 and 0.8.

- **`rho` from the treated unit's gaps over `E`.** Rejected: the fit on `E` absorbs their
  persistence. Their estimate averaged -0.13 at 0.5 and -0.17 at 0.8, and the filtered test
  rejected 10.4% and 23.5%, worse than none. Pooled over every unit's in-sample gaps on `E`, it
  averaged 0.17 and 0.34, and the test rejected 6.8% and 12.3%.
- **The mean absolute innovation, Abadie and Zhao's statistic, after the filter.** Rejected: with
  the true `rho` it held its level, 4.4%, 4.6% and 4.9%, but it reads a constant effect only in
  the first post period. Its power at 0.5 for effects 1 and 2 was 7.6% and 16.5%, at 0.8 6.0% and
  9.1%. The step score's, on 1500 panels: 14.9% and 44.1%, 12.7% and 29.4%.
- **A studentised statistic, a step's GLS t with `rho` from `E`.** Folded in: the step score is
  its numerator, and the permutation replaces its t reference, which would need normal errors.
- **Other estimates of `rho` from the donors fitted on all of `E`**, their gaps over `B` and the
  post periods only. Rejected: ten periods a donor are too few. Least squares without a level
  read each donor's level as persistence, 0.11 at 0: the test rejected 3.9%, 4.1%, 5.2%. With a
  level and Nickell's (Econometrica 1981) bias correction: 4.9%, 5.0%, 5.7%. Han and Phillips's
  first-difference estimator (Econometric Theory 2010): 5.1%, 5.3%, 6.8%. Han, Phillips and
  Sul's: 5.0%, 4.9%, 5.7%. Fitting the donors on half of `E` gives each 17 periods after its fit,
  and at 0.8 cuts the estimate's spread from 0.11 to 0.07.
- **Permuting blocks.** Rejected. Arrangements of disjoint blocks of `T_post` periods, the
  i.i.d. block permutations of Chernozhukov, Wüthrich and Zhu's footnote 6, number
  `T_B / T_post + 1`. A p-value on a grid of 1/20 needs `T_B = 19 T_post`, 95 blank periods for 5
  post ones. Chernozhukov, Wüthrich and Zhu's moving blocks, the cyclic shifts
  of all `T_B + T_post` periods, number `T_B + T_post`: a grid of 1/20 needs `T_B = 15`, which
  leaves 5 of 20 periods to fit. Their validity is approximate, with a bound that falls as
  `(T_post / T_0)^{1/4} log T` (their Theorem 1): at 15 and 5 periods that is 2.3 times an
  unknown constant. Sliding windows that do not wrap are not exchangeable: the post window is
  always the last one, an end of the sequence.

- **Abadie and Zhao's eq. 20 as printed.** Rejected: its coverage reaches `1 - alpha` only as
  `T_B` grows, and with the 5 blank periods a geo test has it does not depend on `alpha` below 0.2.
- **Placing the blank periods first, next to nothing after them.** Rejected: a trend or a season
  then separates them from the post periods by the whole fitting window.
- **A test that permutes all periods before treatment, as ADR 0027's conformal test does.**
  Rejected: the design selected the residuals in `E` small, which is the failure this ADR answers.
