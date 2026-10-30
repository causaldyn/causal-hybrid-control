# ADR 0149 — A worst-share plan may weigh its readings

**Status:** accepted, 2026-10-30. Adds an opt-in argument to `chc.allocation.cvar_allocate`
(experimental).

## Context

`cvar_allocate` chooses the split whose mean gain over a reference in the worst `level` share of
the readings is the most, the gain's conditional value at risk in Rockafellar and Uryasev's form
(2000, Journal of Risk 2(3):21–41), `max_eta eta - E[(eta - gain)_+] / level`, closed on by
Kelley's cutting planes (1960, J. SIAM 8(4):703–712) and, on S-shaped curves, searched in boxes as
`allocate` searches them (Udell and Boyd 2016, Computational Optimization and Applications). The
expectation weighs every reading alike: `1 / n` each.

A posterior that stacks several fits (Yao, Vehtari, Simpson and Gelman 2018, Bayesian Analysis
13(3):917–1007) does not: each fit's draws carry the fit's stacking weight over its draw count, and
the weights are real numbers, not ratios of small integers. The only way to plan over such draws
was to repeat each reading in proportion to its weight. That is

- **inexact**: a weight of 0.137 is 137 copies in 1 000, or a rounding; an irrational weight is
  always a rounding, and the rounding moves the share's boundary;
- **costly**: every copy is a reading, so an excess column in every program and a plane in every
  round of every box. Weights to three digits multiply the readings by up to 1 000.

## Decision

- **`cvar_allocate(..., weights=None)`**: one weight a reading, finite, at least 0, not all 0,
  scaled to sum to 1 (first by the largest, so no sum overflows). A wrong length, a non-finite or
  negative entry and an all-zero vector are refused, each by name.
- **The objective.** `max_eta eta - sum_i w_i (eta - g_i)_+ / level`. Its share's boundary is a
  weighted quantile: the readings in order of gain, worst first, each taking its whole weight until
  the share is full, the reading at the boundary the rest. The planes' excess costs, the reported
  `cvar` and `bound`, the choice of the best split and the box-cut weighing (the risk envelope's
  dual at the box's best split) all read the weights.
- **The programs' costs.** A reading's excess costs `w_i / level` times the factor the readings
  alike use, `max(level n, 1)`, so a reading of weight `1 / n` costs what it costs alike and eta's
  cost is unchanged. Lifting the least weight's cost to 1 instead, as the alike program lifts its
  one cost, raises eta's cost to `level / min w`; measured on concave readings with carryover at
  random levels in `[0.01, 1]`, HiGHS then left 7 of 40 programs unsolved at a spread of `1e7`
  between the largest and least weight, and read programs unbounded past `1e9`. With the alike
  factor, 0 of 40 stopped short at spreads from `1e3` to `1e20`, and 0 of 30 on Hill curves that
  start convex at spreads from `10` to `1e20`.
- **A weight of 0** leaves its reading out of the search: the split is the split without it, to the
  bit against the same positive weights, and its gain is read on the split afterwards and reported
  with the others. The reading is still checked as every reading is.
- **The tolerance** is `rtol` of the largest return on the reference among the readings of positive
  weight; `readings` still counts every reading passed.
- **`weights=None`** takes the path it took before, bit for bit: a digest of the plan's `spend`,
  `gain`, `cvar`, `bound`, `stopped` and `boxes` on three fixed problems (concave, S-shaped, with
  carryover), recorded on 1.0.0 under both resolutions of the lock's dependencies, is asserted.

The weighted forms are checked in exact arithmetic in `validation/cvar_weighted_closed_forms.mac`:
the weighted quantile equals Rockafellar and Uryasev's form on rational gains, weights and levels;
weights 2, 1 and 3 equal the readings repeated twice, once and three times; and the closed forms on
two linear readings at three levels under weights (1, 3) and (3, 1), which the tests read.

## Consequences

- Integer weights plan as the readings repeated: the same `cvar` and `bound` to the search's
  tolerance, a split that gains as much in the repeated readings' share, and the same number of
  boxes, on 10 seeds by 3 levels of concave readings with carryover and of Hill readings.
- Weights scaled by any positive factor plan alike, to the tolerance.
- Weighed alike through `weights=np.ones(n)` is the same problem as `weights=None` but not the same
  bits: `max(level n, 1) (1/n) / level` rounds apart from `max(level n, 1) / (level n)`, and the
  quantile's running sums apart from `floor(level n)`. Both close to the tolerance.
- Two linear readings, A returning 3 and 1 a unit on two channels and B 1 and 2, from the even split
  of 100 at the level 0.8: alike they move the split all onto the first channel, `cvar` 6.25;
  weighed 1 and 3 all onto the second, `cvar` 3.125.

## Alternatives

- **Repeated readings.** Exact only for rational weights with a small common denominator, and every
  copy costs a reading in every program; the weights cost nothing past one number a reading.
- **Importance resampling**: draw `m` readings in proportion to the weights and plan over them
  alike. The share's boundary then carries Monte Carlo error of order `1 / sqrt(m)`, the plan
  changes with the resampling's seed, and the resampled set repeats the heavy readings, which is
  the cost above again.
- **Weights on by default, `np.ones(n)` for the alike case.** It would move the default's bits (see
  above). A new default ships opt-in first, so `None` keeps today's path.
- **Weights for `minimax_allocate`.** Not taken: the worst regret over the readings is the worst
  over every reading of positive weight, whatever the weight, so weights change nothing but which
  readings are in, and leaving a reading out is how a weight of 0 reads there.
