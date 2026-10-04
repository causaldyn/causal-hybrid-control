# ADR 0028 — Asking the panel what the logger read

**Status:** proposed, 2026-09-30.

## Context

`Prescription.evaluate` weights each logged window by the logger's propensity given the plan's
state. That is right only if the logger read the state and a randomisation of its own. The graph
says what the levers were logged on, and `evaluate` refuses a plan whose levers the graph says were
logged on a column outside the state. Nothing asked the panel whether the graph is right, or
whether the logger also read its own past.

Both are testable. A logger that read the state alone leaves each lever independent of every
column fixed before it acted, given the state. Shah and Peters' generalised covariance measure
(2020) tests that on the residuals of two regressions on the state. The usual panel recipe cuts
the regressions' folds by unit and clusters the variance by unit. That assumes many units, and
CHC's cases have few:

- one long run: a pendulum's 4000 steps, one building;
- one to five geos over 104 to 260 weeks, as a media-mix panel has;
- 30 to 300 units over 10 to 20 periods, as a marketplace's zones have.

DoWhy's `falsify_graph` and causal-learn's tests take rows as independent, and neither is aimed at
a decision.

## Decision

- **`chc.independence.gcm_test(x, y, z, *, clusters=None, degree=2, alpha=0.05, draws=1999,
  seed=0)`** (*experimental*) tests `x ⊥ y | z` for every pair of a column of `x` and one of `y`
  at once, and returns a `GcmTest`.
  - Both sides are regressed in-sample, by least squares, on every monomial of the standardised
    `z` up to `degree`. At 2 that is the class `fit_causal_residual` regresses the levers on.
  - A pair's statistic is the sum of its residuals' products over the root sum of their squares,
    summed first within each cluster. The test takes the largest over the pairs.
  - The p-value comes from sign changes of the clusters' sums (Canay, Romano and Shaikh 2017).
  - A column `z` determines has nothing left to test, and stays out of the maximum.
  - `detectable` is the partial correlation at which each pair alone would reject with
    probability 0.8, so a pass says how much it could have seen. The statistic normalises itself,
    so it is read through Student's t on the clusters' sums.
- **Why the periods are the clusters.** When the logger read the state and fresh noise, each
  period's sum is a martingale difference, symmetric about zero, however the state is
  autocorrelated. So the sums need no clusters by unit and no cross-fitting. The noise the units
  share within a period, such as a national budget shock, stays inside the period's sum.
- **`LoggerCheck`** in `chc.evaluation` names the test's rows and columns: the levers, what they
  were regressed on (`given`) and what they were tested against (`columns`).
  - `given` is the plan's states and the levers' recorded parents.
  - `columns` are, at the period, the graph's observed columns that no lever causes, and the
    drivers; and, one period back, `given` and the levers themselves.
- **`Prescription.logger_check`** is computed by `prescribe` before it fits, on the panel it fits.
  **`PlanEvaluation.logger_check`** is computed by `Prescription.evaluate`, on the panel it
  evaluates. Both are `None` when a lever's recorded parent is latent or not a column of the
  panel, since the test conditions on it, or when the panel has fewer rows than twice the
  regression's terms.
- **It reports; it does not refuse.** A p-value at or below 0.05 is logged at `WARNING`
  (`chc_event="logger_check"`), and nothing else changes. `Prescription.report()` prints the
  verdict; the record's JSON does not carry it yet.

## Consequences

Measured by `scripts/bench_logger_check.py` from a snapshot of the library. Its docstring describes
three worlds:

- `panel`: units over periods, an autocorrelated state and a season every unit sees;
- `mm`: geos over weeks, sales that carry over, and a promotion calendar that follows the season;
- `track_o`: one long run of a lightly damped oscillator, dithered by its logger, with a wind the
  logger may read.

In each, the logger read the state and noise of its own. The noise is Gaussian (`homo`), grows with
the state (`hetero`), or has half its variance drawn once a period for every unit (`common`). Rates
only, no timing. The evidence is in `~/.cache/chc-scratch/2026-09-30/gr1/`.

- **Size**, the logger reading the state alone, 1000 panels a case:

  | world | units × periods | noise | rejects at 5% | at 10% |
  |---|---|---|---|---|
  | track_o | 1 × 4000 | homo | 5.3% | 10.2% |
  | track_o | 1 × 4000 | hetero | 5.5% | 11.9% |
  | mm | 1 × 104 | homo | 6.9% | 12.1% |
  | mm | 1 × 156 | homo | 4.8% | 10.7% |
  | mm | 1 × 156 | hetero | 6.0% | 11.1% |
  | mm | 1 × 260 | homo | 5.6% | 10.2% |
  | mm | 5 × 156 | homo | 5.9% | 9.7% |
  | mm | 5 × 156 | hetero | 5.0% | 9.5% |
  | mm | 5 × 156 | common | 4.3% | 8.9% |
  | panel | 30 × 20 | homo | 5.0% | 9.9% |
  | panel | 30 × 20 | common | 5.7% | 10.4% |
  | panel | 100 × 20 | homo | 6.0% | 11.8% |
  | panel | 100 × 20 | common | 5.8% | 11.4% |
  | panel | 300 × 10 | homo | 5.3% | 11.0% |
  | panel | 300 × 10 | common | 6.9% | 12.5% |
  | panel | 4000 × 12 | homo | 5.7% | 10.6% |
  | panel | 4000 × 12 | common | 5.6% | 11.6% |

  Every case is within two points of 5%, so GR1's kill criterion is met at Track O's and the
  media-mix sizes. The widest are one geo over 104 weeks, and 300 zones over 10 periods under a
  shared shock, both at 6.9%: 2.7 standard errors above 5%. At 10% they read 12.1% and 12.5%. The
  two rows of 4000 zones were added on 2026-10-04, and the fifteen above them reproduced exactly.

- **Size where the logger reads the state through more than a quadratic**, 1000 panels a case,
  added on 2026-10-04. In the panel world two loggers read the state alone, with Gaussian noise: one
  clips its lever to [-1.5, 1.5] (`clip`, a box), one switches it at zero, `-tanh(2x)` in place of
  `-0.6x` (`switch`, a threshold rule).

  | world | units × periods | logger | rejects at 5% | at 10% |
  |---|---|---|---|---|
  | panel | 400 × 12 | clip | 4.8% | 10.2% |
  | panel | 4000 × 12 | clip | 7.0% | 13.0% |
  | panel | 400 × 12 | switch | 47.6% | 61.3% |
  | panel | 4000 × 12 | switch | 100.0% | 100.0% |

  Where the logger's mean given the state is linear, as in every world of the table above, the
  lever's regression holds it. Each pair's sum is then the logger's own noise times a residual, and
  the size does not move with the rows. Where the mean is not linear, the quadratic misfits it, and
  each pair's sum drifts from zero by the product of that misfit and the column's, a drift that
  grows with the rows (Shah and Peters 2020). The check then rejects a logger that read nothing
  else. That is still a premise failing, since the logger `evaluate` fits is affine, but not the
  one the warning named: it said the levers read another column, and it now says they read more
  than the record says or read it through more than a quadratic.

- **The alternatives**, the same worlds under the null, 500 panels a case, rejecting at 5%:

  | world | units × periods | noise | this test | GCM over rows | clusters by unit | cross-fit by unit | cross-fit by time |
  |---|---|---|---|---|---|---|---|
  | mm | 1 × 156 | homo | 5.4% | 6.2% | — | 6.0% | 13.8% |
  | mm | 5 × 156 | homo | 5.8% | 5.8% | 26.6% | 16.4% | 7.6% |
  | mm | 5 × 156 | common | 5.2% | 19.6% | 51.0% | 25.6% | 26.2% |
  | panel | 30 × 20 | homo | 4.6% | 4.0% | 6.8% | 5.4% | 5.4% |
  | panel | 100 × 20 | common | 6.6% | 85.2% | 88.2% | 84.2% | 87.6% |
  | track_o | 1 × 4000 | homo | 5.0% | 4.4% | — | 5.0% | 8.0% |

  Each alternative has a Gaussian multiplier over its clusters. Clusters by unit need units, and
  one geo has none to spare. The cross-fitted residuals come from folds whose regressions differ
  from the panel's. A shock the units share ties their rows together within a period, and every
  alternative fails under it, the GCM over rows included. Clusters by period held throughout.

- **Power**, where the logger also read the season, the calendar or the wind, the first column in
  each world, 400 panels a case. The partial correlation it found there, and what it reported
  detectable there, are medians:

  | world | units × periods | reads | partial correlation | detectable | rejects at 5% | at 10% |
  |---|---|---|---|---|---|---|
  | mm | 1 × 156 | 0.2 | 0.139 | 0.258 | 29.5% | 41.0% |
  | mm | 1 × 156 | 0.3 | 0.221 | 0.262 | 65.2% | 74.8% |
  | mm | 1 × 156 | 0.5 | 0.341 | 0.271 | 98.2% | 99.5% |
  | mm | 5 × 156 | 0.1 | 0.077 | 0.116 | 41.5% | 54.8% |
  | mm | 5 × 156 | 0.2 | 0.154 | 0.117 | 97.5% | 98.5% |
  | panel | 100 × 20 | 0.05 | 0.067 | 0.079 | 64.2% | 75.8% |
  | panel | 100 × 20 | 0.1 | 0.136 | 0.087 | 100.0% | 100.0% |
  | track_o | 1 × 4000 | 0.005 | 0.021 | 0.050 | 18.8% | 28.0% |
  | track_o | 1 × 4000 | 0.01 | 0.039 | 0.050 | 57.5% | 70.0% |

  Where the partial correlation it found was a quarter above what it reported detectable, it
  caught 97.5-100% of panels; at 0.85 of it, 64-65%.

- **What `detectable` means**, on independent rows in 10 to 1000 clusters, with the dependence in
  the first of one or three pairs. The target is the mean of what 200 null draws reported; 400
  draws at it were tested:

  | clusters | rows | pairs | detectable | caught | read as a normal | caught |
  |---|---|---|---|---|---|---|
  | 10 | 1000 | 1 | 0.1002 | 82.2% | 0.0810 | 62.7% |
  | 20 | 1000 | 1 | 0.0922 | 77.0% | 0.0837 | 69.2% |
  | 20 | 1000 | 3 | 0.1100 | 77.8% | 0.0955 | 66.5% |
  | 40 | 1000 | 3 | 0.1060 | 80.5% | 0.0990 | 73.5% |
  | 156 | 1560 | 3 | 0.0827 | 79.8% | 0.0813 | 77.8% |
  | 1000 | 1000 | 1 | 0.0884 | 78.5% | 0.0882 | 78.2% |

  The statistic normalises itself, so with few clusters it clears the critical value only where
  Student's t on the clusters' sums clears a higher one. `detectable` is read there. Read as a
  normal, it came out a fifth too small at 10 clusters.

- **What it protects.** `Prescription.evaluate` on the lifecycle test's market, 400 later panels
  a case of 400 units over 12 periods. The plan was fitted where the incentive was drawn afresh.
  In the later panels the logger also chased demand (`strength` times the period's demand), or
  kept `strength` of its last incentive at the same spread. Demand pushes supply by 0.15 in the
  test's market, and by 0.6 in the second. The error is the estimate less the plan's true cost,
  over the interval's standard error at `model_error = 0`. The table was measured again on
  2026-10-04, under the bootstrap over units that ADR 0049 put in the interval's place:

  | logger | strength | demand's push | refused | check rejects | covers, default | covers, `model_error = 0` | its error |
  |---|---|---|---|---|---|---|---|
  | chase | 0 | 0.15 | 0 | 5.2% | 100.0% | 100.0% | -0.05 |
  | chase | 0.25 | 0.15 | 0 | 100.0% | 100.0% | 100.0% | -0.05 |
  | chase | 0.5 | 0.15 | 0 | 100.0% | 99.8% | 100.0% | +0.04 |
  | chase | 1 | 0.15 | 0 | 100.0% | 98.8% | 100.0% | +0.36 |
  | chase | 0 | 0.6 | 0 | 4.5% | 100.0% | 100.0% | -0.05 |
  | chase | 0.25 | 0.6 | 0 | 100.0% | 100.0% | 100.0% | +0.10 |
  | chase | 0.5 | 0.6 | 0 | 100.0% | 100.0% | 100.0% | +0.25 |
  | chase | 1 | 0.6 | 0 | 100.0% | 99.5% | 100.0% | +0.56 |
  | sticky | 0.5 | 0.15 | 0 | 100.0% | 99.8% | 96.2% | +1.13 |
  | sticky | 0.8 | 0.15 | 381 | 100.0% | 68.4% | 15.8% | +2.39 |

  - Where the logger read the state alone, the check passed 95% of the time, and the intervals
    covered every panel where it passed. Both over-cover here: nominally they are 95%.
  - A logger that chased demand was flagged on every panel from a chase of 0.25. The estimate's
    mean error stayed within 0.56 of its standard error, and both intervals still covered
    98.8-100%. A flag is a premise failing, not a measured cost.
  - A logger that kept half its last incentive was flagged on every panel, and its estimate sat
    1.13 standard errors high. The interval at `model_error = 0` covered 96.2%, against every panel
    where the logger read the state alone; the default interval, widened by the model's error,
    covered 99.8%. At 0.8, `evaluate` refused 381 panels. On the 19 it evaluated, the check
    flagged each, and the intervals covered 68% and 16%.
  - The check's rejection rates and the refusals reproduced the first run's exactly. The
    intervals did not: under the earlier one, the interval at `model_error = 0` covered 91.2%
    of the half-sticky panels, its estimate 1.23 standard errors high, and a chase's mean error
    reached 0.71. The run is in `~/.cache/chc-scratch/2026-10-04/logger-bench/`.

  GR1's gate was that coverage must move where the omitted column matters, or the check is not
  wired in. It moved with the lever's own past, which the certificate let through at 0.5. It did
  not move with an omitted column at these strengths. The check is wired in, and its warning
  says what the levers read, not what that cost.

- **Mutations.** Fifteen mutants, each caught by a test. In `gcm_test`: the clusters ignored, a
  column `z` determines tested, the spread read about zero, `z` ignored, the p-value's `1 +`
  dropped, `detectable` at half power, and `detectable` read as a normal. In the wiring: clusters
  by row, the levers' past or the states' past left out, the unlogged columns left out, the parents
  left out of `given`, the levers' descendants tested, a latent parent read, and no warning. The
  harness is in the evidence directory, under `mut/`.

## Not built

- **(1) of GR1: a lever the graph says reaches no steered state.** GR1 also asked for the z-score
  of such a lever's fitted channel. `fit_causal_residual` keeps each transition's influence on the
  channel only when asked for it. Its channel error is a scale, not coverage: one log's value
  scatters by 24-40% of itself (`CausalDynamicsFit.channel_error`). A z-score read from it needs
  its own size bench. GR2's worlds include a lever wrongly declared a non-ancestor, and they
  measure first whether that error goes silent.
- **The decision record.** `Prescription.to_json` does not carry the check. GR3 changes the
  record's schema once, with the graph's provenance.
- **Columns the graph leaves out, one period back.** A logger that read last period's demand is
  seen only through what that left in the state and the levers.
- **The weighted GCM** (Scheidegger, Hörrmann and Bühlmann), for dependence the regressions of
  degree 2 cannot represent. It waits for GR2 to show the need.
- **Telling a logger outside the quadratic from one that read another column.** A regression whose
  class grows with the rows would hold the logger's mean and keep the size; a second test at a
  higher degree would say which of the two readings a rejection is. Either needs its own size table
  on the small panels above, where a larger class costs the rows it is fitted on. Until then the
  warning names both.

## Alternatives considered

- **The design first planned: cross-fitted residuals, folds and clusters by unit, a Gaussian
  multiplier.**
  Rejected on the table above. It assumes many units. CHC's cases have one to five, or share
  shocks within a period.
- **Shah and Peters' GCM over rows.** It held where the rows' noise was their own, and rejected
  20-85% of panels where the units shared a shock.
- **A Gaussian multiplier over the periods' sums, in place of sign changes.** Not built. Sign
  changes are exact when the sums are symmetric and independent, whatever their number. The
  multiplier is asymptotic in the number of clusters, and a marketplace panel here has 10 to 20
  periods.
- **HC2's correction of the residuals for leverage.** Not built. At degree 2 the regressions here
  have 3 to 10 terms against at least 104 rows, so the leverage it corrects is small, and the size
  above held without it.
- **DoWhy's `falsify_graph` and causal-learn's tests.** Both take rows as independent, which the
  table above shows fails with a shared shock, and neither is aimed at a decision.
- **Testing the sequential back-door condition instead** (Bareinboim, Def. 8.2.3). It is weaker:
  the history must block each action's back-door paths to the reward. It licenses weights that
  condition on the history. `evaluate`'s weights condition on the state alone, so they need a
  logger Markov in it. A logger that kept half its last incentive opens no back-door path given the
  history, and it still put the estimate 1.13 standard errors high (above).
- **Refusing on a rejection.** The check flags dependence well below what costs the evaluation
  its coverage: a chase was flagged on every panel while the estimate's mean error stayed within
  0.56 standard errors. A refusal would refuse evaluations that held.
