# ADR 0075 — One unit's channel error read across its periods by cosines

**Status:** accepted, 2026-10-08. Amends ADR 0053 and ADR 0064.

## Context

ADR 0053 sums the channel's scores within each unit before squaring them, so the error holds
whatever the dependence within a unit; ADR 0061 adds the periods as a second way, and ADR 0062
reads a test off it against `t(G - 1)`. A panel of one unit has no second unit to sum within.
`prescribe` then passed the fit no clusters, and the fit took each transition as independent:
`error_clustered_by` and `error_clusters` read `None`, and the report said "each transition taken
as independent". A transition's score is the lever's residual times the noise. Where the noise
persists and the lever does too, as in ADR 0064's plant, the scores persist, and the sum of their
squares reads less than the spread of their sum. One unit is no corner case: a building, a
pendulum and a single market are each one series.

Every public entry point that reaches one unit, and what it read:

- `fit_causal_residual`: the rows' sandwich; nothing told it its rows were one series.
- `prescribe`: that error, in `identification_radius`, in the tube's model error, in the report
  and in `to_json()`.
- `omitted_confounder_bound`: its confidence bounds from the influence's rows, against `t(N - 1)`.
- `misspecification_cost`: the covariance of the two fits' difference from its rows.
- `persistence_check`: a p-value of `nan` on one unit, since it read its spread from the units'
  sums; `prescribe`'s report said "not tested on one unit".
- The logger check, `gcm_test` clustered by period: on one unit each period holds one row, and
  under its null the statistic's terms are martingale differences in time, so the test holds as it
  stands. ADR 0028 measured its size on one unit.
- `Prescription.evaluate`: one unit's windows read as independent, under a `DeprecationWarning`;
  from 1.0 it raises (ADR 0049).

**The measurement.** ADR 0064's plant on one unit: `x' = 0.95 x + 0.1 (b u + 1.5 z) + e`,
`u = a + 0.9 z`, the lever's own part `a` AR(`r_u`) of spread 0.5, the noise `e` AR(`r_e`) of
spread 0.05, and `z` drawn afresh and adjusted for; 300 periods before the log, then `T`
transitions. Each log is fitted by `fit_causal_residual(adjust_for=("z",), channel_degree=0,
influence=True)`, Euler, two folds, so the channel is one coefficient and `channel_error` is its
standard error. A cell draws 1000 logs from one `SeedSequence`, spawned a log, and each log's seed
spawned again for its data and its bootstrap. Each estimator reads the fit's own scores `s_t`, the
influence's channel column, at the rows' factor `N / (N - k)`:

- *rows*: `sum s_t^2`, the error as it stood, against the normal's quantile.
- *NW, rule*: Newey and West's (1987) Bartlett kernel at the lag `floor(4 (T / 100)^(2/9))` that
  Newey and West (1994) start their selection from, against the normal's.
- *NW, plug-in*: the same kernel at Andrews' (1991) AR(1) plug-in bandwidth,
  `1.1447 (alpha T)^(1/3)` with `alpha = 4 rho^2 / ((1 - rho)^2 (1 + rho)^2)` and `rho` the scores'
  lag-1 autocorrelation, against the normal's.
- *NW, fixed-b*: the same kernel at `1.3 sqrt(T)` (Lazarus, Lewis, Stock and Watson 2018), against
  the fixed-b critical value (Kiefer and Vogelsang 2005), simulated at each `T` from 20 000 series
  of independent normals.
- *cosines*: the equal-weighted cosine estimate, EWC, over `nu = 0.4 T^(2/3)` cosines rounded down,
  against `t(nu)` (Lazarus, Lewis, Stock and Watson 2018).
- *blocks*: `nu + 1` contiguous blocks of periods as clusters, CR1, against `t(nu)` (Bester,
  Conley and Hansen 2011).
- *MBB*: a moving-block bootstrap (Künsch 1989) of the final stage on the cross-fit's residuals,
  blocks of `ceil(T^(1/3))` periods, the order Hall, Horowitz and Jing (1995) find for a variance,
  the basic interval from 499 draws.
- *MBB, PW*: the same at Politis and White's (2004) automatic block length, as Patton, Politis and
  White (2009) correct it.

A refusal has no column: it reads no interval at all. The coverage of the 95 % interval:

| `r_e` | `r_u` | `b` | `T` | rows | NW, rule | NW, plug-in | NW, fixed-b | cosines | blocks | MBB | MBB, PW |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.7 | 0.8 | 50 | 0.967 | 0.960 | 0.964 | 0.982 | 0.969 | 0.981 | 0.769 | 0.759 |
| 0.5 | 0.7 | 0.8 | 50 | 0.907 | 0.916 | 0.909 | 0.952 | 0.954 | 0.946 | 0.708 | 0.698 |
| 0.9 | 0.7 | 0.8 | 50 | 0.829 | 0.852 | 0.852 | 0.914 | 0.935 | 0.923 | 0.640 | 0.626 |
| 0.9 | 0.7 | 0.0 | 50 | 0.842 | 0.861 | 0.857 | 0.927 | 0.935 | 0.935 | 0.665 | 0.652 |
| 0.9 | 0.0 | 0.8 | 50 | 0.971 | 0.968 | 0.969 | 0.980 | 0.978 | 0.978 | 0.762 | 0.778 |
| 0.0 | 0.7 | 0.8 | 200 | 0.963 | 0.961 | 0.963 | 0.974 | 0.964 | 0.970 | 0.908 | 0.917 |
| 0.5 | 0.7 | 0.8 | 200 | 0.860 | 0.921 | 0.920 | 0.948 | 0.950 | 0.948 | 0.870 | 0.857 |
| 0.9 | 0.7 | 0.8 | 200 | 0.703 | 0.852 | 0.859 | 0.918 | 0.925 | 0.918 | 0.798 | 0.795 |
| 0.9 | 0.7 | 0.0 | 200 | 0.683 | 0.864 | 0.873 | 0.918 | 0.928 | 0.913 | 0.804 | 0.804 |
| 0.9 | 0.0 | 0.8 | 200 | 0.954 | 0.952 | 0.954 | 0.958 | 0.954 | 0.958 | 0.889 | 0.908 |
| 0.0 | 0.7 | 0.8 | 1000 | 0.949 | 0.943 | 0.947 | 0.944 | 0.947 | 0.941 | 0.922 | 0.931 |
| 0.5 | 0.7 | 0.8 | 1000 | 0.841 | 0.938 | 0.938 | 0.948 | 0.953 | 0.949 | 0.932 | 0.930 |
| 0.9 | 0.7 | 0.8 | 1000 | 0.651 | 0.895 | 0.914 | 0.939 | 0.941 | 0.932 | 0.899 | 0.905 |
| 0.9 | 0.7 | 0.0 | 1000 | 0.678 | 0.902 | 0.924 | 0.939 | 0.947 | 0.935 | 0.906 | 0.917 |
| 0.9 | 0.0 | 0.8 | 1000 | 0.945 | 0.947 | 0.951 | 0.948 | 0.944 | 0.949 | 0.931 | 0.933 |

A coverage's Monte Carlo standard error is 0.004 to 0.015. The cosines' column is
`fit_causal_residual`'s own error given the periods: at 50 and 200 transitions the simulation's,
which agreed with it to 1e-15 on the simulation's logs, and at 1000 the library's, refitted on the
same logs. There the simulation counted `nu = 39`, `0.4 T^(2/3)` rounded down in floating point,
where the library counts 40, and the two counts' coverages differed by 0.003 at the most.

How far each error reads the estimate's spread across the logs, as the root mean square of the
standard errors over the standard deviation of the estimates:

| `r_e` | `r_u` | `b` | `T` | rows | NW, plug-in | NW, fixed-b | cosines | blocks |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.7 | 0.8 | 50 | 1.132 | 1.133 | 1.107 | 1.213 | 1.208 |
| 0.5 | 0.7 | 0.8 | 50 | 0.955 | 0.989 | 0.982 | 1.086 | 1.070 |
| 0.9 | 0.7 | 0.8 | 50 | 0.814 | 0.872 | 0.877 | 0.985 | 0.954 |
| 0.9 | 0.7 | 0.0 | 50 | 0.825 | 0.880 | 0.887 | 0.998 | 0.971 |
| 0.9 | 0.0 | 0.8 | 50 | 1.186 | 1.175 | 1.130 | 1.245 | 1.226 |
| 0.0 | 0.7 | 0.8 | 200 | 1.064 | 1.067 | 1.064 | 1.100 | 1.105 |
| 0.5 | 0.7 | 0.8 | 200 | 0.760 | 0.914 | 0.952 | 1.000 | 0.989 |
| 0.9 | 0.7 | 0.8 | 200 | 0.544 | 0.812 | 0.855 | 0.913 | 0.885 |
| 0.9 | 0.7 | 0.0 | 200 | 0.545 | 0.815 | 0.857 | 0.911 | 0.889 |
| 0.9 | 0.0 | 0.8 | 200 | 1.032 | 1.032 | 1.007 | 1.038 | 1.044 |
| 0.0 | 0.7 | 0.8 | 1000 | 0.970 | 0.969 | 0.951 | 0.971 | 0.968 |
| 0.5 | 0.7 | 0.8 | 1000 | 0.727 | 0.966 | 0.980 | 1.006 | 0.996 |
| 0.9 | 0.7 | 0.8 | 1000 | 0.525 | 0.959 | 0.981 | 1.015 | 0.988 |
| 0.9 | 0.7 | 0.0 | 1000 | 0.508 | 0.925 | 0.947 | 0.979 | 0.952 |
| 0.9 | 0.0 | 0.8 | 1000 | 1.007 | 1.005 | 0.977 | 0.996 | 0.998 |

## Decision

- **`fit_causal_residual(periods=...)`** takes each transition's period on a log of one unit, a
  whole number of steps, one a transition and no period twice, and reads the channel's covariance
  across them by cosines: `V = N / (N - k) sum_j c_j c_j'`, with
  `c_j = sum_i sqrt(2 / nu) cos(pi j (p_i + 1/2) / P) s_i` for `j` from 1 to `nu`, `s_i` the
  transition's scores summed over its states, `p_i` its period counted from the log's first, and
  `P` the periods the log spans. A period the log lacks holds no score, as Parzen (1963) reads a
  series with gaps. `nu` is the most whole cosines at or below `0.4 N^(2/3)`, read in whole
  numbers, `1000 nu^3 <= 64 N^2`, so that 1000 transitions read 40 and not 39; and one at the
  least. `CausalDynamicsFit.periods` keeps the periods and `error_cosines` gives `nu`. Each row's
  `influence` is the rows' own, factor and all; its readers sum it by the same cosines. `clusters`
  and `periods` together are refused: they are two readings of one dependence.
- **A test read off it takes `t(nu)`.** As the log grows at a fixed `nu`, the `c_j` read as
  independent normals, so the statistic is a `t(nu)` (Sun 2013; Lazarus, Lewis, Stock and Watson
  2018). `omitted_confounder_bound` takes `t(nu)`'s quantile and reads both influences by the
  cosines; `misspecification_cost` reads the difference's covariance by them, and refuses two fits
  read over different periods.
- **`prescribe` passes the periods on a panel of one unit**, the one case with no second group.
  `DecisionCertificate.error_cosines` counts the cosines, `to_json()` writes it, `null` where the
  error is not read so, the fit's log event carries it as `cosines`, and the report reads
  "read across the one unit's periods by 21 cosines of them (EWC)" for 399 transitions.
- **`persistence_check` on one unit reads its pairs across their periods** the same way: the total
  of the pairs' products over the root of their projections on `nu` cosines of the later
  transitions' periods, squared and summed, against `t(nu)`, `nu` counted over the pairs. Below two
  pairs it reads `nan`, and the report says the pairs leave no spread to read.
- **Unchanged:** every panel of several units, to the bit; the logger check; and
  `Prescription.evaluate`, which keeps ADR 0049's path.

**Why the cosines.** Over the fifteen cells they covered 0.925 to 0.978, against a `t` whose
degrees of freedom are a count, not a fit. Fixed-b Bartlett covered 0.914 to 0.982 and the blocks
0.913 to 0.981. Fixed-b needs a critical value for each `T` and bandwidth, simulated or read off a
fitted polynomial. Bartlett at the rule's lag or at the plug-in bandwidth, read against the
normal's quantile, read too little where the scores persist most: 0.852 to 0.924 where `r_e` is
0.9 and `r_u` 0.7. The blocks covered less than the cosines wherever the scores persist, by 0 to
0.015 at the same `nu`, and read 1 to 3 % less of the spread. The bootstrap undercovers even
where the noise is drawn afresh, 0.769 at 50 transitions and 0.908 at 200: it resamples the
final stage alone, the cross-fitted nuisances held, where the error's sensitivity runs through
them too, and it costs 499 draws a fit. A refusal would leave every log of one unit, a
building's or a pendulum's, without an error.

## Consequences

- *On one unit the error widens where the scores persist*: where `r_e` is 0.9 and `r_u` 0.7 it
  reads 0.91 to 1.02 of the estimate's spread, where the rows read 0.51 to 0.82.
- *It widens where the scores do not persist too*, by the price of the `t`: where the noise is
  drawn afresh, the mean interval is 35 % wider than the rows' at 50 transitions, 11 % at 200
  and 2.5 % at 1000. That is the price of not knowing whether the scores persist.
- *`prescribe`'s tube on one unit widens with the error*, so fewer of its steps may be trusted.
  `to_json()`'s schema version stays 2: a key was added, and none changed its meaning.
- *A collider bias stays uncovered.* At `r_e` 0.9, `r_u` 0.7, `b` 0.8 and 1000 transitions the
  channel reads 0.0226 low, 0.39 of its spread (ADR 0064), and the cosines' 0.941 is what is
  left of 95 % beside it.
- *`misspecification_cost`'s chi-square mixture* reads `S` as known; read by `nu` cosines it is
  itself noisy, as at few clusters (ADR 0062), and the analogue for several directions at once is
  not built.
- *Several units in one declared cluster* still read their transitions as independent: there the
  panel names a second dimension the error could read across, and nothing reads it yet.
- *`Prescription.evaluate`* keeps reading one unit's windows as independent, warned of, until 1.0
  refuses them. Its windows are `H + 1` periods long, so a log of a few hundred periods holds a
  few dozen, and cosines over them would read `t(3)` or so; that reading was not measured.
- *Tests*: on one unit of 200 transitions with noise AR(0.9), 200 seeded logs, the cosines covered
  at least 0.87 and the rows at most 0.80; the error by hand on Frisch-Waugh-Lovell's partialling,
  with the periods offset, a gap and shuffled; the periods move the error and nothing else, under
  both integrators; the refusals; the cosine count in whole numbers; the bound's `t(21)`, the
  misspecification's `S`, and the persistence check, by hand; the certificate, the report, the
  JSON and the log event on one unit, and `null` on several.

## Alternatives

- **Newey-West at a consistent bandwidth, against the normal's quantile** (Newey and West 1987;
  Andrews 1991; Newey and West 1994). Rejected: it covered 0.852 to 0.924 where `r_e` is 0.9
  and `r_u` 0.7.
- **Newey-West at `1.3 sqrt(T)` with fixed-b critical values** (Kiefer and Vogelsang 2005;
  Lazarus, Lewis, Stock and Watson 2018). A close second; rejected for its critical values, a
  table or a simulation for each bandwidth over the log, where the cosines read a `t`.
- **Contiguous blocks as clusters** (Bester, Conley and Hansen 2011). It covers nearly as the
  cosines do, and it reuses CR1. Rejected: it covers less wherever the scores persist, and its
  blocks' edges are a choice the cosines do not need.
- **A moving-block bootstrap** (Künsch 1989), at `T^(1/3)` blocks (Hall, Horowitz and Jing 1995)
  or at Politis and White's (2004) length. Rejected: it undercovered at 50 and 200 transitions,
  even where the noise is drawn afresh, and a bootstrap that refits the nuisances costs the fit's
  time times the draws.
- **A refusal on one unit.** Rejected: one unit is a common log, and the cosines read it at the
  coverage the table gives.
- **`periods` read on several units as well**, a unit's own cosines summed. Not built: CR1 within
  units already holds whatever the dependence within a unit, and ADR 0061's second way reads the
  periods' shocks.
