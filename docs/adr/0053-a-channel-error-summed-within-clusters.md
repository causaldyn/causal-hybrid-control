# ADR 0053 — A channel error summed within clusters

**Status:** accepted, 2026-10-06.

## Context

`fit_causal_residual`'s channel error is a sandwich over rows: each transition's score, the
fit's sensitivity to its rate times its residual, squared state by state and summed
(`_robust_spread`). That holds where every transition's score is independent of every other's.
Two things break it:

- **Repeated rows.** A log of 100 transitions, each repeated 16 times, moves the channel by 8e-9
  and reads its error at 0.249 of the log's own, as 1600 independent rows would.
- **Units followed over time.** The transitions of one unit share whatever persistent noise the
  model leaves out, and a lever a unit keeps high or low for weeks makes their scores move
  together. The row sandwich then reads too small an error, and the tube that
  `prescribe` builds on it is too narrow.

`Panel.cluster` was declared "for cluster-robust inference" since 0.5.0, and nothing read it:
`fit_causal_residual` had no cluster argument, and `prescribe` passed none.

## Decision

- **`fit_causal_residual(clusters=...)`** takes each transition's cluster, `N` labels of any one
  sortable kind, at least two distinct. The channel's covariance then sums each cluster's scores
  over its transitions and its states before squaring them:
  `V = c sum_g s_g s_g'`, `s_g = sum_{i in g} sum_s J[:, i, s] e[i, s]`, with CR1's factor
  `c = G / (G - 1) (N - 1) / (N - k)` over `G` clusters and `k` coefficients a state. Where every
  transition is its own cluster, `c` is the row sandwich's `N / (N - k)`. Both integrators take
  it: under `rk4` the clustered spread goes through the fixed point's Newton matrix as the row
  one does.
- **The estimate does not move.** The folds, the nuisances, the channel and the drift are the
  same with clusters or without; only the channel's error, the influence's factor and the
  fit's new field `clusters` (the codes, from 0) differ.
- **The influence's readers sum within clusters too.** `CausalDynamicsFit.influence` carries
  CR1's factor where the fit had clusters, and `omitted_confounder_bound` and
  `misspecification_cost` sum its rows within each cluster before squaring them (`_independent`).
  `misspecification_cost` refuses two fits whose clusters differ.
- **`prescribe` clusters by default**: by the panel's declared cluster, or by its unit where it
  declares none, whose transitions share any persistent noise. A panel whose transitions all fall
  in one group keeps the row sandwich: one cluster has no spread between clusters to read.
  `DecisionCertificate.error_clustered_by` and `error_clusters` name the grouping, and the report
  and `to_json()` carry them; the report says when the transitions were taken as independent.
- **`drift_error` stays as it was**, the drift regression's homoskedastic error: a scale that the
  tube leaves out, which a clustered drift error would not change.

## Consequences

- *The repeated rows*, clustered by the row each copy repeats, read the log's own error to
  `2e-8`: at one coefficient `(N - 1) / (N - k)` is 1 and `G / (G - 1)` is the log's own
  `N / (N - k)`.
- *The Wald test's size* (`scripts/bench_clustered_error.py`, section `size`): panels of `G` units
  of 20 transitions whose true channel is 0, the lever's own part and the noise AR(0.7) within
  each unit, or half of each the unit's own, 400 panels a cell; a test at 5 % on each error:

| noise | units `G` | `dt` | size, row by row | size, by unit | error by unit / row by row, median |
|---|---|---|---|---|---|
| AR(0.7) | 5 | 0.1 | 0.1350 | 0.0975 | 1.37 |
| AR(0.7) | 10 | 0.1 | 0.1750 | 0.0925 | 1.42 |
| AR(0.7) | 20 | 0.1 | 0.2000 | 0.0850 | 1.47 |
| AR(0.7) | 40 | 0.1 | 0.2275 | 0.0525 | 1.54 |
| AR(0.7) | 80 | 0.1 | 0.2075 | 0.0650 | 1.57 |
| half the unit's own | 5 | 0.1 | 0.2075 | 0.1475 | 1.24 |
| half the unit's own | 10 | 0.1 | 0.2700 | 0.1125 | 1.50 |
| half the unit's own | 20 | 0.1 | 0.3050 | 0.0925 | 1.69 |
| half the unit's own | 40 | 0.1 | 0.3550 | 0.0950 | 1.86 |
| half the unit's own | 80 | 0.1 | 0.3150 | 0.0375 | 1.99 |
| AR(0.7) | 40 | 1.0 | 0.1250 | 0.0650 | 1.16 |
| half the unit's own | 40 | 1.0 | 0.1425 | 0.0875 | 1.21 |
| independent | 40 | 0.1 | 0.0500 | 0.0650 | 1.00 |
| independent | 40 | 1.0 | 0.0575 | 0.0625 | 0.99 |

- *By unit, the size comes down on every dependent design*, from 12.5 % to 35.5 % row by row to
  3.75 % to 14.75 %. Where the noise is independent, clustering costs little: 6.5 % and 6.25 %
  against 5 % and 5.75 %, within 1.5 standard errors. With 400 panels a cell, a size's Monte Carlo
  standard error is 1.1 points near 5 % and 1.5 near 10 %.
- *CR1 is not exact.* Against the normal's 1.96 its size is above 5 % at 5 and 10 units under
  either noise; under AR noise it is near 5 % from 40 units, but where half the noise is the
  unit's own it stays above 9 % up to 40 units and comes down to 3.75 % only at 80. Few clusters
  read the error from few independent terms. A `t` with `G - 1` degrees of freedom, or a wild
  cluster bootstrap (Cameron, Gelbach and Miller 2008), is the usual remedy; neither is built here.
- *`prescribe`'s numbers move* on every panel of more than one unit: the channel's error, the
  tube's budget built on it, and with them the certified horizon. On the quickstart's 200 regions,
  whose noise is drawn afresh each week, the error reads 0.007851 against 0.007821 row by row, and
  nothing else moves. On the pendulum's 20 episodes (`scripts/pendulum_demo.py`) it reads 3.15e-4
  against 4.63e-4, and the adjusted schedule's trusted prefix goes from 9 steps to 10: on this log
  the scores within an episode partly offset one another, so their sums spread less than they do
  one by one.
  A clustered error is not larger by construction; it is the one that reads the dependence.
- *Tests*: the repeated rows, both ways; each transition its own cluster equal to the row
  sandwich on one state, under both integrators, with the same influence; CR1 by hand on clusters
  of uneven sizes, where own-sample partialling-out on a linear basis is Frisch-Waugh-Lovell;
  the estimate, the drift, the representer and every other field unmoved, the influence moved by
  CR1's factor alone; the refusals; `_independent` on rows and on clusters; the omitted
  confounder's bound and the misspecification cost on the repeated rows; `prescribe`'s
  grouping by unit, by a declared cluster and on one unit.
- *Mutations*: each of 27 mutations of the clustered error fails a test. Among them: CR1's factor
  without `G / (G - 1)`, or with `N` for `N - 1`; the spread summing one state; either integrator
  dropping the codes; the bound or the misspecification cost ignoring the clusters; `prescribe`
  putting every transition in one group, or passing no clusters; and the certificate, the report
  or `to_json()` losing the count.

## Alternatives

- **Clustered folds**, each cluster held out whole by the cross-fit. Rejected for now: it moves
  the estimate, and with this module's nuisances, linear in a fixed basis, a fold that keeps a
  test row's neighbours leaves a bias of the order of the features over `N`, below the error. A
  learner that adapts to its sample would need it.
- **A heteroskedasticity-and-autocorrelation (Newey-West) error within one unit.** Not taken: it
  needs a bandwidth, and a panel of many units carries its dependence in the units already. A
  panel of one unit keeps the row sandwich, and the report says so.
- **The row sandwich with every state's scores summed within the row.** Not the default: under
  `None` the error stays what every release so far reported, each transition's each state
  independent; a caller that wants a row's states summed passes each transition as its own
  cluster.
- **A minimum number of clusters before `prescribe` clusters.** Rejected: the row sandwich is not
  safer at few clusters, only narrower, and the certificate names how many there were.
