# ADR 0026 — Inference for a staggered design, through diff-diff

**Status:** proposed, 2026-09-30.

## Context

`chc.did` returns point estimates only: Callaway and Sant'Anna's `ATT(g, t)`, its event study and
its overall ATT. A decision taken on a staggered rollout needs their uncertainty, and a decision
that trusts parallel trends needs to know what a failure of them would do.

The inference is not new. diff-diff (MIT, 3.12.0) carries Callaway and Sant'Anna's multiplier
bootstrap, its uniform band over the event study, and Rambachan and Roth's HonestDiD, with wheels
for Python 3.9 to 3.14. The library's estimators are adapters over maintained backends, not
reimplementations, so the question is whether diff-diff's inference holds its level on the panels
the library serves, and where it does not.

## Decision

- **The `did` extra** is `diff-diff>=3.12,<4`. It brings numpy, pandas and scipy, and nothing the
  core does not already have besides pandas. The upper bound is there because a major release may
  move the results the adapter reads. The `dev` group holds it too, so CI runs the tests.
- **`chc.did.callaway_santanna_inference(outcomes, group, ...)`** (*experimental*) takes
  `callaway_santanna`'s arguments, plus `alpha`, `draws` and `seed`.
  - **It computes `callaway_santanna`'s own estimates.** Then it asks diff-diff for the same
    comparison at the same base period `g - 1`. diff-diff's cells, event study and overall ATT
    must equal the library's to `1e-9` of the largest effect, or it raises `RuntimeError`. So every
    interval is centred on the library's estimate, and a diff-diff release that changed an
    estimand would stop the call instead of shifting its intervals.
  - **It returns an `EventStudyInference`**: the estimate; the event study's standard errors and
    uniform band by relative time, without the base period; the overall ATT's standard error and
    interval.
  - **A unit treated from the first period is dropped**, as R's did drops it. It has no base
    period, so it has no cell and is no one's control. A first-treated period outside the panel is
    refused: within the panel such a unit is never treated, and the caller says so.
  - **diff-diff's warnings of empty cells are filtered by their message.** The base period's cells
    are zero by construction, and some late cells have no clean control; `callaway_santanna`
    leaves both out without a word. Every other warning passes.
- **`EventStudyInference.robust_interval(m)`** is Rambachan and Roth's smoothness restriction
  `Delta^SD(m)` for the event study's average effect after treatment, equally weighted. It is the
  optimal fixed-length interval from diff-diff's HonestDiD, computed on the covariance of an
  analytic fit of the same event study. A bootstrapped event study carries none (below). The
  relative-magnitudes restriction is not exposed (below).

## Consequences

Measured by `scripts/bench_did_inference.py` from a snapshot of the library. The panels have ten
periods, cohorts first treated at periods 4, 6 and 8 and never-treated units, a quarter each; unit
effects that move with the cohort; AR(1) errors within a unit; comparisons with the not-yet-treated.
1000 panels a case, `draws = 999`. Rates only, no timing. The evidence is in
`~/.cache/chc-scratch/2026-09-30/ca1/`.

- **Size**, no effect, at 5%:

  | units | AR(1) | overall ATT rejects | band rejects | pointwise rejects |
  |---|---|---|---|---|
  | 30 | 0 | 7.8% | 24.1% | 6.3-11.2% |
  | 30 | 0.6 | 6.9% | 21.2% | 5.4-11.2% |
  | 100 | 0 | 6.6% | 8.7% | 4.4-7.8% |
  | 100 | 0.6 | 6.0% | 7.9% | 5.2-6.6% |
  | 300 | 0 | 6.5% | 7.2% | 4.7-6.8% |
  | 300 | 0.6 | 5.7% | 6.4% | 4.0-7.1% |

- **Coverage**, a dynamic effect that differs by cohort:

  | units | AR(1) | overall ATT | band | pointwise | overall's median width |
  |---|---|---|---|---|---|
  | 30 | 0 | 0.923 | 0.794 | 0.881-0.951 | 1.36 |
  | 30 | 0.6 | 0.927 | 0.768 | 0.885-0.947 | 1.54 |
  | 100 | 0 | 0.953 | 0.912 | 0.931-0.951 | 0.76 |
  | 100 | 0.6 | 0.931 | 0.925 | 0.925-0.959 | 0.86 |
  | 300 | 0 | 0.951 | 0.951 | 0.938-0.957 | 0.44 |
  | 300 | 0.6 | 0.941 | 0.943 | 0.939-0.964 | 0.50 |

  - **The overall ATT's interval held its level within two points from 100 units**, whatever the
    errors' correlation, which the bootstrap's whole-unit draws are for. At 30 units it rejected
    6.9-7.8%.
  - **The band held from 300 units only.** At 100 it rejected 7.9-8.7% and covered 0.91-0.93;
    at 30, a fifth to a quarter of the null panels fell outside it. Its bootstrap critical value
    shrank with the units (2.65 at 30 against 2.8 at 300, in an exploration on the same design),
    where it should not: with seven or eight units a cohort, the draws' maximum has thin tails.
  - An exploration of 12 000 panels calling diff-diff directly, on other seeds, agreed within two
    standard errors on every rate.

- **`robust_interval(0)`** (`robust`: the treated drift `delta` a period from the never-treated,
  who are the comparison; AR(1) 0.5; 1000 panels a case). The event study's own average after
  treatment was off by 0.17-0.18 at `delta = 0.05`, as `delta` times the mean of `e + 1` says.

  | units | covers, `delta = 0` | covers, `delta = 0.05` | median width |
  |---|---|---|---|
  | 30 | 0.788 | 0.780 | 1.71-1.73 |
  | 100 | 0.924 | 0.924 | 1.12-1.13 |
  | 300 | 0.949 | 0.952 | 0.67 |

  Like the band, it holds from 300 units, and not below.
- **HonestDiD on a bootstrapped event study.** diff-diff's event study from a bootstrapped fit
  carries no covariance, and HonestDiD then reads its effects as independent, with a warning. In
  an exploration of 1000 panels a case on the same design, `Delta^SD(0)`'s interval covered 0.68,
  at 300 units and at 100; at 300 it was 0.40 wide where the analytic covariance makes it 0.67.
  That is why `robust_interval` fits twice, and raises if the analytic event study has no
  covariance.
- **The relative-magnitudes restriction.** diff-diff 3.12's interval for `Delta^RM` is the
  estimated identified set widened by `z` standard errors; its own docstring says the paper's ARP
  hybrid is disabled pending calibration. In the exploration it covered every panel, at 3.4 to 6.3
  times the width of `Delta^SD(0)`'s. It is not R's interval: on the panel below, R's hybrid
  interval on the same event study and covariance was 1.4, 1.6 and 1.8 times as wide at
  `Mbar = 0.5, 1, 2`. R's default grid runs 20 standard errors either side of zero, and cut its
  upper end at 2.81 there, since the estimate sat 15 standard errors above zero; the ratios are on
  a grid centred on the estimate.
- **Against R** (did 2.5.1, HonestDiD 0.2.8, in a scratch library), on one panel of 300 units
  from the same design, comparisons with the not-yet-treated:
  - the 30 cells, the 14 event times and the overall ATT agreed to `6e-15`, and so did their
    analytic standard errors, to `4e-15` relative. The event study's analytic covariance was R's
    influence-function covariance to `3e-15` in every correlation.
  - At 999 draws each, the bootstrap's standard errors were 0.97-1.03 of the analytic ones in
    diff-diff and 0.96-1.11 in R, which reads the scale off the draws' interquartile range: a
    Monte Carlo error near 3.7% at 999 draws, against 2.2% for their standard deviation. The
    band's critical values were 2.78 and 2.80.
  - HonestDiD's `Delta^SD` interval, on the same event study and covariance, agreed with R's to
    `1e-3` at `m = 0` to `0.2`. A test pins `robust_interval` to R's at `m = 0` and `0.1` on
    another panel.
- **Tests** (`tests/test_did_inference.py`, 16): the estimate is `callaway_santanna`'s under both
  comparisons, with a cohort treated from the first period; a panel without never-treated units;
  the band is wider than every pointwise interval; the seed reproduces the bootstrap; the
  standard error is the estimate's spread over sixty panels; a disagreement raises; the
  refusals; the extra is named when it is missing; always-treated units change nothing;
  `robust_interval` widens with `m`, holds a trend the event study's own average misses by more
  than 0.4, is R's HonestDiD on a panel drawn from numpy's frozen `RandomState` stream, and
  refuses a negative `m`. Seven mutations were caught: always-treated units kept, the period
  shift dropped, the standard error doubled, the comparisons swapped, the pointwise interval
  passed off as the band, the relative-magnitudes restriction in place of smoothness, and
  HonestDiD on the bootstrapped event study, which only the test against R catches.

## Not built

- **The relative-magnitudes restriction**, until diff-diff ships ARP's hybrid: its interval is
  not R's (above).
- **A small-sample band.** Below 300 units the band undercovers; nothing here corrects it.
- **de Chaisemartin and D'Haultfoeuille's and Goodman-Bacon's inference, and synthetic DiD.**
  diff-diff has them; no consumer asks yet.
- **Designs that chose their treated units.** The multiplier bootstrap treats the units as drawn;
  a design that chose them (plans/27, CA2 and CA3's note) needs other inference.

## Alternatives considered

- **Implementing the multiplier bootstrap in the library.** Rejected: it would duplicate a
  maintained MIT implementation that reproduces the library's estimates to the last digit.
- **Returning diff-diff's estimates instead of checking them.** Rejected: the check is the only
  place the two implementations meet, and a silent change of estimand is the failure it stops.
- **HonestDiD on the bootstrapped event study.** Rejected on the 0.68 above.
- **moderndid** (MIT), named in plans/27 as the fallback. Not needed: diff-diff held.
