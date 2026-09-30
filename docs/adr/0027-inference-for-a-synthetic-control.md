# ADR 0027 — Inference for a synthetic control, owned

**Status:** proposed, 2026-09-30.

## Context

`chc.scm` returns point estimates only. A decision read off a synthetic control, such as a geo
test or a market launch, needs to know whether the gap after treatment is more than the donors'
misfit.

Two tests are standard.

- **Abadie, Diamond and Hainmueller's in-space placebo** (2010) refits the synthetic control with
  each donor as the treated unit. It ranks the treated unit's ratio of post- to pre-period
  prediction error among all of them. When the treated unit was drawn at random, the rank is
  uniform and the test exact. ADH moved California into the donor pool of every placebo (§3.4,
  p. 501), so each unit's synthetic control comes from all the others.
- **Chernozhukov, Wüthrich and Zhu's conformal test** (2021) tests a sharp null. It takes the
  hypothesised effect out of the treated unit's periods after treatment and refits over every
  period. It then ranks the residuals after treatment against each cyclic shift of the residuals
  in time. So it needs the residuals exchangeable in time, not the assignment random.

The library's causal inference is adapted, not written, where a maintained package does it.
diff-diff (MIT, 3.12, already the `did` extra) carries both tests. Building the adapter found three
things.

- **A defect in `chc.scm` itself, fixed first.** The weights came from 5000 steps of projected
  gradient, which stopped short of the optimum. Over 50 panels a design, the pre-period error
  ended up to 8% above its minimum, and the overall effect moved by up to 0.13 with 80 donors. The
  weights now come from one non-negative least squares, exact to rounding. The reduction is
  `validation/synthetic_control_nnls.mac`.
- **diff-diff's placebo leaves the treated unit out of every placebo's donor pool.** Each placebo
  then has one donor fewer than the treated unit, so the ranks are no longer exchangeable, and
  the test gives up its exact level. The panels where only one construction rejected ran almost
  all one way (below). Reported upstream as
  [diff-diff#837](https://github.com/igerber/diff-diff/issues/837).
- **diff-diff's weights come from Frank-Wolfe.** At its default 10 000 inner steps it left the
  treated unit's fit, or a placebo's, short of convergence on 5 to 11 of 40 generic panels, and
  then refused the test. At 200 000 steps none failed, out of 120. Reported upstream as
  [diff-diff#838](https://github.com/igerber/diff-diff/issues/838).

Over the exact weights, each test is a few lines. The adapter was longer than the methods: a data
frame, a configuration that reproduces the problem, a check that diff-diff reached the optimum, a
Frank-Wolfe budget, and a refusal for each way it can fall short.

## Decision

- **`chc.scm.synthetic_control_inference(outcomes, treated_unit, n_pre, *, alpha=0.05)`**
  (*experimental*) returns a `SyntheticControlInference`: `synthetic_control`'s estimate,
  `placebo_p_value`, `interval` and `alpha`.
- **The placebo is ADH's construction.**
  - Every unit's synthetic control comes from all the others. The p-value is the share of the
    `N` units whose ratio is at least the treated unit's.
  - A pre-period error below `sqrt(eps)` of the panel's spread before treatment counts as that
    much. So units their donors reproduce, as they can when there are more donors than periods
    before treatment, rank by their error after treatment, not by rounding.
  - The ratios are compared cross-multiplied, so a panel with no spread divides by nothing.
- **The interval is the hull of the constant effects the conformal test does not reject** (moving
  blocks, `q = 1`).
  - The search tries 401 effects over ten residual errors either side of the estimate. The error
    is the treated unit's residual, over every period, at the estimate.
  - It pushes the outermost accepted effects outwards until the test rejects, and bisects each end
    to a millionth of that error.
  - Below `alpha = 1 / T` the interval is unbounded; when the test rejects every effect it tries,
    it is `(nan, nan)`.
- **Owned, with diff-diff as the tests' oracle.** A test holds the conformal p-value equal to
  diff-diff's `conformal_test` at 25 effects. Another puts the interval's ends where diff-diff's
  test starts to reject. There is no `scm` extra; diff-diff stays in the `dev` group.

## Consequences

Measured by `scripts/bench_scm_inference.py` from a snapshot of the library, 1000 panels a case.
The panels have a unit level, three AR(1) factors with loadings uniform on `[0, 1]` and AR(1)
errors, with 20 periods before treatment and 5 after. Beside the shipped placebo, every panel also
computes diff-diff's construction, from the same exact weights. Rates only, no timing. The
evidence is in `~/.cache/chc-scratch/2026-09-30/ca2/`.

- **Size**, no effect, the treated unit drawn at random:

  | donors | AR(1) | placebo 5% | placebo 10% | diff-diff's 5% | diff-diff's 10% | only diff-diff's rejects, 10% | only the placebo rejects, 10% |
  |---|---|---|---|---|---|---|---|
  | 9 | 0.5 | 0.0% | 9.6% | 0.0% | 11.1% | 18 | 3 |
  | 19 | 0 | 3.8% | 9.4% | 4.4% | 10.0% | 10 | 4 |
  | 19 | 0.5 | 4.7% | 9.2% | 5.2% | 10.7% | 18 | 3 |

  With 9 donors the smallest p-value is 1/10, so neither rejects at 5%.

- **The interval**, same panels: it covered the effect 97.2-98.4% of the time at `alpha = 0.05`
  and 93.1-94.0% at 0.10. On 25 periods the test's p-values step by 1/25, so its own levels are 4%
  and 8%; the hull covers at least as often as the test. Median widths were 5.4-6.5 at 0.05 and
  4.1-5.3 at 0.10, where the errors' standard deviation is 1 and 1.15. It left the estimate out
  0.4-0.8% of the time at 0.05 and 2.4-4.3% at 0.10. It was never empty at 0.05, and 3 or 4 times
  in 1000 at 0.10. Those ten panels were scanned again on a grid 40 times finer and 4 times wider
  than the search's. On nine of them the test rejected every effect. On the tenth it accepted a
  set about a hundredth of a residual error wide, which the search's grid had stepped over.

- **Power**, 19 donors, AR(1) 0.5:

  | effect | placebo 5% | placebo 10% | diff-diff's 5% | diff-diff's 10% | interval excludes 0, 0.05 | interval excludes 0, 0.10 | interval covers, 0.05 / 0.10 |
  |---|---|---|---|---|---|---|---|
  | 1 | 11.3% | 20.4% | 12.6% | 21.9% | 5.5% | 12.9% | 97.8% / 93.6% |
  | 2 | 30.8% | 46.2% | 33.3% | 48.0% | 21.5% | 34.8% | 98.2% / 93.3% |

  diff-diff's construction rejected 0.5-1.5 points more than ADH's under no effect, and 1.3-2.5
  more under an effect. Most of what it adds is size.

- **A designed test.** With the treated unit chosen as the one its donors fit best, and no effect,
  the placebo rejected 27.8% at 5% and 42.4% at 10% (diff-diff's 28.7% and 43.2%). The interval
  covered 94.4% at 0.05 and 81.7% at 0.10, and was empty 17 times in 1000 at 0.10. Choosing on
  the fit breaks both tests' premises: the ranks are no longer exchangeable, and the residuals
  before treatment are selected small.

- **The conformal p-value is not monotone in the effect.** On a panel whose effect alternates
  between +10 and -10, the test accepts effects near +10 and near -10, at `p = 2/30`, and rejects
  the estimate between them. At 0.05 the hull spans both; a test pins this.

## Not built

- **Firpo and Possebom's confidence sets.** They were unbounded in practice: on the first pilot
  panels, bounded on 7-15% of them at 10%.
- **CWZ's interval for the average effect.** With 20 periods before treatment and 5 after, its
  collapse leaves `T / T1 = 5` blocks, so its smallest p-value is 1/5. It needs `T0 >= 10 T1` at
  10%.
- **Per-period conformal intervals, in-time placebos, leave-one-out.** diff-diff has them; no
  consumer asks yet.
- **Inference for a design that chose its treated unit.** Neither test holds there (above). A
  designed test needs a reading its design justifies, which this does not provide.
- **Covariates and predictor weights.** The synthetic control matches on the pre-period outcomes,
  weighted alike, as `chc.scm` always has.

## Alternatives considered

- **An adapter over diff-diff.** Built first, then rejected: its placebo gives up the exact level,
  its Frank-Wolfe needed 200 000 steps, and the adapter was longer than the tests.
- **pysyncon** (MIT, 1.7.0), the first backend planned. It brings matplotlib and carries nothing
  diff-diff lacks.
- **diff-diff's construction as an option beside ADH's.** Rejected: under no effect it rejected
  0.5-1.5 points more than ADH's, and under an effect 1.3-2.5 more, so what it offers over ADH's
  is mostly size.
- **diff-diff's floor on the placebo's denominator** (`1e-8 max(scale, 1)^2` on the squared
  error). Rejected: through `max(scale, 1)` it depends on the outcomes' units.
