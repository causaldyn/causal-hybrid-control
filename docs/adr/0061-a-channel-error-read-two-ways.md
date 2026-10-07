# ADR 0061 — A channel error read two ways: by unit and by period

**Status:** accepted, 2026-10-07. Amends ADR 0053.

## Context

ADR 0053 sums the channel's scores within each unit, or each declared cluster, before squaring
them. That reads any dependence between the transitions of one unit. It does not read dependence
between units.

A shock every unit shares in a period is such a dependence: a holiday, a price change, the
weather. On its own it is noise in the rate. Where the levers also have a part that every unit
moves together, such as a national campaign, the scores of one period then move together across
units. A sum within units leaves that out, and the error reads too small (Moulton 1990; Cameron,
Gelbach and Miller 2011).

Cameron, Gelbach and Miller's two-way covariance adds the units' summed scores squared to the
periods' and takes away the cells', which both count. That difference has two flaws:

- **It can read a direction below nothing.** On the quickstart's 200 regions over 11 periods it
  read 3 of the channel's 12 directions below nothing. On a log of 3 units over 4 periods it read
  every direction of a channel affine in the state below nothing: clipped, the channel's error
  read 1e-9, or nan where rounding left the read below zero under the root.
- **Their fix moves with the coefficients.** They clip the difference's negative eigenvalues in
  the coefficients' own basis. A change of the coefficients' units or of the state's zero changes
  that basis, so it changes what is clipped: on the quickstart's panel, moving the state's zero
  moved the channel's error from 0.008164 to 0.009408. MacKinnon, Nielsen and Webb (2024) note it:
  the standard error of any coefficient then depends on how the others are written.

`scripts/bench_clustered_error.py`, section `two_way`, measures the errors. The panels have one
state and one lever. The true channel is 0, a confounder is adjusted for, and the lever's own part
and the noise's unit part are AR(0.7) within each unit. In `shared`, every unit also shares a part
of the lever, AR(0.7) over the periods, and a shock in the noise each period. In `none`, neither.
There are 500 panels a cell. The table gives the size of a 5 % test of the channel on each error,
and in brackets the error's root-mean-square over the channel's spread:

| Design | Units | Periods | Row by row | By unit | By period | Two ways | Two ways, `t(G - 1)` |
|---|---|---|---|---|---|---|---|
| shared | 10 | 40 | 0.328 (0.50) | 0.280 (0.58) | 0.078 (0.96) | 0.060 (1.03) | 0.038 |
| shared | 20 | 40 | 0.518 (0.35) | 0.456 (0.40) | 0.086 (0.93) | 0.064 (0.96) | 0.054 |
| shared | 40 | 40 | 0.604 (0.25) | 0.572 (0.29) | 0.076 (0.96) | 0.072 (0.97) | 0.064 |
| shared | 40 | 5 | 0.536 (0.34) | 0.526 (0.36) | 0.222 (0.80) | 0.198 (0.82) | 0.102 |
| shared | 40 | 10 | 0.610 (0.28) | 0.580 (0.30) | 0.144 (0.86) | 0.134 (0.87) | 0.094 |
| shared | 40 | 20 | 0.608 (0.27) | 0.580 (0.30) | 0.090 (0.97) | 0.082 (0.98) | 0.072 |
| none | 10 | 40 | 0.202 (0.66) | 0.074 (1.02) | 0.212 (0.66) | 0.060 (1.04) | 0.040 |
| none | 20 | 40 | 0.236 (0.61) | 0.074 (0.97) | 0.270 (0.59) | 0.064 (0.98) | 0.052 |
| none | 40 | 40 | 0.278 (0.58) | 0.066 (0.94) | 0.298 (0.57) | 0.058 (0.96) | 0.054 |
| none | 40 | 5 | 0.136 (0.76) | 0.058 (1.03) | 0.200 (0.71) | 0.048 (1.09) | 0.012 |
| none | 40 | 10 | 0.222 (0.65) | 0.082 (0.97) | 0.296 (0.61) | 0.068 (1.00) | 0.036 |
| none | 40 | 20 | 0.204 (0.66) | 0.044 (1.05) | 0.248 (0.63) | 0.038 (1.07) | 0.024 |

With 500 panels a cell, a size's Monte Carlo standard error is 1.0 point near 5 %, 1.3 near
10 % and 1.8 near 20 %.

## Decision

- **`fit_causal_residual(clusters=...)` takes two labels a transition**, shape `(N, 2)`: a unit
  and a period, say, each of any one sortable kind, at least two distinct in each dimension.
- **Three covariances.** With `S_k = sum_g s_g s_g'` over the groups of dimension `k`, and `S_12`
  over the cells that both dimensions share:
  - the two ways' sums, `V = c (S_1 + S_2 - S_12)`, `c` CR1's factor at the smaller dimension's
    count `G = min(G_1, G_2)`, less the part below nothing;
  - each dimension's sums alone, `c_k S_k`, at its own CR1 factor.
- **The part below nothing is measured against `R = S_1 + S_2 + S_12`**, the same sums with every
  sign a plus. `-R <= S_1 + S_2 - S_12 <= R`, so the generalised eigenproblem
  `(S_1 + S_2 - S_12) v = l R v` has every `l` in `[-1, 1]`; the part with `l < 0`,
  `R V min(L, 0) V' R` with `V' R V = I`, is taken away. `R` moves with the coefficients as the
  sums do, so a value read off the result does not move with their units or the state's zero. A
  difference with no part below nothing keeps every bit. Any reference
  `a (S_1 + S_2 - S_12) + b S_12`, `S_12` alone or `S_1 + S_2` among them, takes away the same
  part: it has the same generalised eigenvectors `v`, and each part taken away,
  `S v v' S / v' S v` with `S = S_1 + S_2 - S_12`, depends on `v` alone.
- **The error is the largest of the three reads**, the rule MacKinnon, Nielsen and Webb (2024)
  propose. The channel's error, the root-mean-square standard error of its value at the log's
  rows, is the largest of its three reads. `omitted_confounder_bound` takes each spread's largest.
  `misspecification_cost` takes the covariance under which `tr(W S)`, its noise, is largest, and
  reads every number off that one. A read that rounding leaves below nothing reads nothing.
- **The influence carries CR1's factor at `G`**, as one-way; a reader moves each dimension's sums
  to that dimension's own factor.
- **`prescribe` clusters two ways by default.** One dimension is the panel's declared cluster, or
  its unit where it declares none. The other is the period each transition starts in.
  - Where the transitions all start in one period, it sums within the first dimension alone, as
    before.
  - Where they name one group, it takes them as independent, as before.
  - `DecisionCertificate.error_periods` counts the periods where the error is two-way, and is None
    otherwise. The report says "within each of N periods, and within both, whichever reads largest
    (two-way CR1)", and `to_json` writes `error_periods`. The schema version stays 2: a key was
    added, and none changed its meaning.

## Consequences

- *With a shared shock*, by unit the error reads 0.29 to 0.58 of the spread, and the test rejects
  28 % to 58 % of the time. Two ways it reads 0.82 to 1.03, and rejects 6.0 % to 8.2 % from 20
  periods, 13.4 % at 10.
- *Without one*, two ways cost nothing: the error reads 0.96 to 1.09 of the spread, against 0.94
  to 1.05 by unit, and the test rejects 3.8 % to 6.8 %, against 4.4 % to 8.2 % by unit.
- *Which read is the largest.* Under the shared shock, the two ways' sums in 48 % to 83 % of the
  panels and the periods alone in nearly all the rest; without it, the units alone in 55 % to
  65 %, the two ways' sums in most of the rest. In 1 panel of 6000 the two ways' sums read
  nothing, and the error read the larger way alone.
- *The largest of three is conservative.* It never reads below either way alone, so it never
  credits a dependence that one way's sums read as negative. On the pendulum's 20 episodes
  (`scripts/pendulum_demo.py`), whose scores within an episode partly offset one another (ADR
  0053), the steps alone read 4.33e-4, against 3.15e-4 by episode and 2.96e-4 two ways. The error
  reads 4.33e-4, and the adjusted schedule's trusted prefix goes back from 10 steps to 9.
- *Few periods are few clusters.* At 5 periods under the shared shock, two ways rejects 19.8 %,
  and 10.2 % against `t(G - 1)`'s quantile. CR1 at small `G` over-rejects, as ADR 0053 found for
  units. A `t` with `G - 1` degrees of freedom, a wild bootstrap or the cluster jackknife is still
  not built.
- *`prescribe`'s numbers move* on every panel with more than one group and more than one period.
  On the quickstart's 200 regions over 11 transitions each, the channel's error reads 0.008159,
  against 0.007851 by region alone, and nothing else in the report moves.
- *Where the channel is not 0 and both the lever and the noise persist*, the estimate is itself
  biased, which no error repairs: the state is a collider of the past lever and the past noise.
  That is why the channel is 0 here. It is a separate defect.
- *Tests*:
  - the three reads by hand on the Frisch-Waugh-Lovell channel, on labels where the two ways'
    sums read the most, and on six values by hand, at each way's own factor;
  - a second way that holds each transition alone reads the first way's error and influence,
    in either order and under both integrators;
  - a direction below nothing read as nothing; mixing the coefficients mixes every covariance
    the same way; SciPy's generalised eigensolver reads the same part below nothing;
  - two ways that read nothing read the larger way alone; two ways that read less than a way
    alone read, and bound, as that way does; the error does not move with the state's zero;
  - the refusals;
  - the omitted confounder's bound and the misspecification cost read two ways;
  - `prescribe` two ways, by unit and by a declared cluster, one-way on one period, and
    independent on one unit.
- *Mutations*: 28 of 29 mutations fail a test. Among them: the two ways' sums without the
  cells', or with them added; the cells' codes colliding; `G` the larger count; CGM's clip in the
  coefficients' own basis; the part below nothing measured against the first dimension alone; the
  clip removed or inverted; each dimension alone at the smaller one's factor; the two ways' sums
  alone; the least of the three reads under either integrator; the bound or the misspecification
  cost reading the two ways' sums alone, or the cost the least noise; and `prescribe` one-way, or
  losing the periods in the certificate, the report or `to_json()`. The 29th measures the part
  below nothing against the cells' sums alone, which takes away the same part, as above; a test
  holds that.

## Alternatives

- **The eigenvalue fix in the coefficients' own basis** (Cameron, Gelbach and Miller 2011).
  Rejected: it moves with the coefficients, as above.
- **The part below nothing measured against one dimension's sums alone.** Rejected: the error
  would then depend on which dimension comes first.
- **The two ways' sums alone, clipped against `R`.** Rejected: with few clusters they read nothing,
  and the certificate would then say the channel is known exactly; and where one dimension's
  dependence is negative they read below what either way alone reads.
- **Clipping each read at nothing rather than the covariance.** Rejected: a mean of clipped reads
  is not the read of one covariance, and the misspecification cost's chi-square mixture needs a
  covariance.
- **Each sum with its own CR1 factor**, `c_1 S_1 + c_2 S_2 - c_12 S_12`. Not taken: it sized as
  the one factor did, within Monte Carlo error (7.6 % against 7.0 % at 5 periods without a shared
  shock), and one factor is what the influence carries for its readers to move.
- **The two sums without their intersection**, `S_1 + S_2`. It is never below nothing. Rejected: it
  counts each cell twice, and without a shared shock it reads 1.10 to 1.25 of the spread.
- **By period alone.** Rejected: it misses the unit's persistent noise, and without a shared shock
  it rejects 20.0 % to 29.8 %.
- **Two ways on request, by unit by default.** Rejected: by unit alone the test rejects
  28 % to 58 % under a shared shock, and two ways cost nothing without one. The certificate names
  the grouping.
- **A minimum number of periods before two ways.** Rejected, as ADR 0053 rejected a minimum number
  of clusters: by unit alone is not safer at few periods, only narrower. At 5 periods under the
  shared shock it rejects 52.6 %, against 19.8 % two ways.
