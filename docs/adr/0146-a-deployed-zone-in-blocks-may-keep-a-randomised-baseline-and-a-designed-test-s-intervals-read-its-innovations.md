# ADR 0146 — A deployed zone in blocks may keep a randomised baseline, and a designed test's intervals read its innovations

**Status:** accepted, 2026-10-27. Amends ADR 0090 (the gate reads plants with memory in blocks)
and ADR 0092 (a designed test is read from the periods it left blank).

## Context

Two opt-in readings left a gap their own ADRs named.

- **The gate in blocks has no rollback on evidence.** ADR 0090 made the gate's evidence
  design-based on a plant with memory: each block draws its policy with a known probability, the
  first `b` periods are burned, and the block's read mean, weighted by the inverse of that
  probability, is one increment. Once a zone is deployed it runs the candidate in every block.
  Its rollback e-process then reads the baseline's weight on blocks that never ran the baseline:
  it has no evidence, and only a drift alarm rolls the zone back. A candidate that turns worse
  after its DEPLOY is kept for good.
- **The designed test's intervals assume exchangeable gaps.** ADR 0092's `serial="ar1"` filters
  the treated unit's gaps to their AR(1) innovations, which keeps the test's size at 5% where the
  errors persist. Its split-conformal intervals still ranked the raw blank gaps: at coefficient
  0.8 the 80% intervals covered 70.7%. ADR 0092 listed "intervals for persistent gaps" as not
  built, since one period's effect is not one innovation's, and it listed no check of the AR(1)
  model the filter rests on.

## Decision

- **`GateConfig.rollback_baseline_share`** (*experimental*), unset by default, needs a
  `block_length`, and lies in `(0, 1)`. Set to `q`, a deployed zone's blocks run the baseline with
  probability `q` and log `q`, and the candidate's log `1 - q`; the gate refuses a deployed batch
  logged otherwise. A deployed zone is then an experiment whose candidate share is `1 - q`, and the
  rollback e-process reads that experiment's harm increment.
  - **The weights.** The candidate's weight is `1 / (1 - q)` on its blocks and the baseline's
    `1 / q` on its own, so `D = w_new - w_base` has mean 0 given the past, and `D (1 - R)`, with
    `R` the block's read mean, has mean `V_base - V_new`. It is at least `-1 / q`, and the
    rollback bets on `(D (1 - R) - delta_harm) / (1 / q + delta_harm)`, at least -1 and of mean at
    most 0 while the candidate is not `delta_harm` worse. With `centred`, the increment is ADR
    0135's `x = D (R - c)` at the candidate share `1 - q`, and the rollback bets on
    `(-x - delta_harm) / (B'(c) + delta_harm)`, `B'(c) = max((1 - c) / (1 - q), c / q)`; the level
    `c` continues from the experiment's.
  - **The burn-in** is the shadow blocks' and the experiment's: the first `b` periods of every
    deployed block are not read. Under carryover of order `p <= b` the read periods see only their
    own block's policy, so a ROLLBACK claims, design-based, that the baseline beat the candidate by
    more than `delta_harm` over the blocks read since the DEPLOY, each block given the past.
  - **What it does not change.** Without the share every increment, verdict and log record is as
    before; with it, the shadow and experiment blocks are read as before, and a drift alarm still
    rolls a deployed zone back.
- **`blank_period_inference(serial="ar1")`'s intervals read the innovations.** From the last
  blank gap `u_0`, post period `k`'s gap is `u_k = tau_k + rho^k u_0 + sum_(j=1..k) rho^(k-j)
  e~_j`, the `e~` innovations that carry the gaps' level. The blank periods' `T_B` innovations,
  the first against the last fitting period's gap as in the test, are draws of the same law. With
  their mean `m` and standard deviation `s`, `a_k = sum_(j<k) rho^j` and
  `v_k = sum_(j<k) rho^(2j)`, the interval is
  `u_k - rho^k u_0 - m a_k +- t_(T_B - 1)(1 - alpha / 2) s sqrt(v_k + a_k^2 / T_B)`: the
  prediction interval of a `k`-step forecast of an AR(1) with a known coefficient and an unknown
  level, exact for normal innovations (Box and Jenkins, *Time Series Analysis*, 1970, chapter 5,
  with the level's error added as in a regression's prediction interval). Infinite with fewer than
  two blank periods. The default reading's intervals do not change.
- **`BlankPeriodInference.ar_check_p_value`**, under `serial="ar1"`, `None` otherwise, checks the
  AR(1) model on the donors' gaps the coefficient was estimated from, filtered by it. Each donor's
  innovations less their own mean are multiplied at lags 1 and 2 and summed over the donors period
  by period. Under the model the innovation vectors are i.i.d. over time whatever their covariance
  across donors, which the donors' common factors make far from diagonal, so each lag's total is
  read against its spread over the periods, less its mean under the model, `-(n - h) / n` times
  the squares' total over `n - 1` for a series of `n` demeaned terms. The squares of the two
  ratios, summed, are read against a chi-square with 2 degrees of freedom, a portmanteau in the
  manner of Box and Pierce (JASA 1970) and Ljung and Box (Biometrika 1978) with a variance robust
  to the donors' cross-correlation. It is reported and refuses nothing.
- These are fixes under `serial="ar1"`: the intervals move toward the level they state, and the
  check adds a field. The default reading's every output is digested by a test.

## Consequences

Measured through the shipped code; counts and rates only, no wall time.
`validation/gate_block_rollback.mac` and `validation/scm_ar1_intervals.mac` hold the identities,
the bounds, the second moments, the planning approximation and the sample-point checks;
`tests/test_gate_blocks.py` and `tests/test_scm_blank_inference.py` the tests.

- **A wrong ROLLBACK stays within `alpha_harm`.** On the lab's plant, in blocks of 50 with 40
  burned, read every 5 blocks over 20 000 periods, ADR 0090's candidate, deployed, 0.0096 worse in
  the long run and so inside `delta_harm = 0.02`, was rolled back on none of 1000 paths at `q`
  0.1, 0.2 and 0.3, plain or centred. On a plant of order 1 whose candidate costs 0.2 now and repays
  it a period on, exactly as good sustained, in blocks of 2 with 1 burned, it was rolled back on 0
  to 0.25% of 400 paths of 2000 blocks at the three shares.
- **The burn-in carries the design here too.** On that plant a baseline block after the
  candidate's reads the repayment in its first period, so without a burn-in it reads 0.1 better
  than the candidate. With no period burned the rollback fired on 30%, 51% and 65% of the paths
  plain, and on all of them centred.
- **A candidate that turns worse is rolled back; without the share, never.**
  - On the lab's plant, a deployed candidate that turns, 40 blocks after its DEPLOY, to acting
    `N(1, 1)`, 0.157 worse than the baseline: centred, the rollback fired on 31%, 96.5% and 99.75%
    of 400 paths within the 360 blocks left, after a median of 320, 215 and 155 blocks from the
    turn; plain, on 9%, 46% and 62%. Without the share, on none.
  - On the plant of order 1 with Gaussian noise of sd 0.2, a candidate 0.1 worse sustained from its
    DEPLOY, blocks of 2 with 1 burned: centred, every one of 400 paths rolled back, after a median
    of 790, 410 and 330 blocks at `q` 0.1, 0.2 and 0.3, against 793, 454 and 350 by the planning
    approximation; plain, 36.8%, 71.5% and 82% of the paths within 4000 blocks. Without the share,
    none.
  - The rollback reads the contrast since the DEPLOY, as without blocks. A candidate +0.1 better
    for its first 100 blocks and then 0.1 worse was rolled back centred on 89.5%, 99.5% and 100% of
    400 paths within 1900 blocks of the turn, after a median of 1145, 770 and 610 blocks.
- **The cost.** A deployed zone gives up `q` times the candidate's sustained gain per period: its
  mean reward is `V_new - q (V_new - V_base)`, exactly on a plant whose reward is affine in its
  recent actions, as the tests' plants of order 1 are. At `q = 0.2` and a gain of 0.03, the lab's
  better candidate, that is 0.006 a period. The share is the price of watching the DEPLOY on
  evidence, and a caller who trusts the drift alarm leaves it unset.
- **The intervals cover at their level.** On the bench's designed panels (19 donors, 15 fitting,
  5 blank and 5 post periods, the factors' and errors' coefficient equal), 2000 a case:

  | coefficient | default at 0.2 | `"ar1"` at 0.2 | at 0.1 | at 0.05 | check rejects at 5% |
  |---|---|---|---|---|---|
  | 0 | 83.0% | 79.6% | 89.6% | 94.8% | 2.5% |
  | 0.5 | 79.3% | 79.7% | 89.9% | 94.9% | 3.6% |
  | 0.8 | 70.3% | 79.6% | 89.6% | 94.8% | 4.9% |
  | 0.8, 10 blank | 72.1% | 79.4% | 89.6% | 94.8% | 4.5% |
  | 0.8, effect 2 | 70.1% | 79.4% | 89.5% | 94.6% | 5.9% |

  The default's intervals at 0.1 and 0.05 are unbounded with 5 blank periods. The `"ar1"`
  intervals' median widths at 0.2 are 3.64, 4.38 and 5.38, against the default's 3.64, 3.95 and
  4.36: at 0.8 they are wider by the persistence they now carry. The panel-level standard error
  of a coverage is at most 0.009, so each is within one or two of its level. The estimated
  coefficient, 0.77 for 0.8, accounts for the shortfall that remains: on 2000 other panels of the
  same kind at 0.8, the same intervals covered 80.4% at 0.2 with the true coefficient and 79.7%
  with the estimate.
- **The check sees an AR(2), and little of a mismatch.** On 600 panels whose errors are an AR(2)
  with coefficients 0.5 and 0.3, it rejected at 5% on 62%. On panels whose factors persist at 0.8
  and errors at 0, or the reverse, the gaps are not AR(1), and it rejected on 12% and 8%. A
  single AR(1) coefficient fitted to them filters most of their persistence.
- **Ten mutations of the change were each caught by a test** (listed in the report of the change).

## Alternatives considered

- **A rollback e-process restarted at each read, or a CUSUM over the blocks.** It would catch a
  turn sooner, but the gate's rollback, with or without blocks, reads the contrast since the
  DEPLOY, and its guarantee is stated so; a change of what a ROLLBACK claims is a separate
  decision.
- **Running baseline blocks on a fixed schedule rather than at random.** A schedule is not a
  design: a block's draw must be independent of the past for the weighted contrast to have its
  mean given the past.
- **Shadow evidence in a deployed zone**, the baseline's propensity of the candidate's action. In
  blocks the action is the policy, and the baseline's propensity of the candidate's block is 0.
- **Split-conformal intervals on the innovations, scaled by `sqrt(v_k)`.** Exact at the first
  post period for any law of the innovations, and at every period for normal innovations of mean
  0; but the gaps carry a level, the innovations then have mean `(1 - rho) mu`, and the `k`-step
  error's mean `mu (1 - rho^k)` outgrows the scale. At 0.8 it covered 78.6% at the fifth post
  period, and at 0.05 and 0.1 it is unbounded with 5 blank periods.
- **A centred split-conformal interval.** It leaves the exact rank at every period and is the
  widest of the three.
- **Conformal intervals over ordered tuples of blank innovations.** With 5 blank periods and a
  5-step error the tuples are permutations of the same 5 values, whose weighted sums barely move
  at `rho` near 1; a valid rank over disjoint tuples needs `T_B >= k / alpha` blank periods.
- **A pooled Ljung–Box over the donors as if they were independent.** The donors' gaps share the
  factors, and it rejected 5.3%, 6.8% and 10.7% at lags 1 and 2 where the model holds; read period
  by period, 3.2%, 2.8% and 4.0% on the same 600 panels. Lag 1 alone has little power, since the
  coefficient is fitted to it; three lags rejected 5.3% to 6.2% where it holds, and saw the AR(2)
  no better.
- **A refusal when the check rejects.** Not done: it is a diagnostic beside an estimate whose
  error it cannot repair, as the gate's carryover check is.
