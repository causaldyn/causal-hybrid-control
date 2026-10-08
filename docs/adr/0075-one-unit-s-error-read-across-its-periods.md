# ADR 0075 — One unit's channel error read across its periods by cosines

**Status:** accepted, 2026-10-08. Amends ADR 0053 and ADR 0064. Amended 2026-10-08, before any
release: the units of one declared cluster are read by their period sums across the periods, and
`misspecification_cost` sizes its test given periods against `S` read by the cosines.

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

**Several units in one declared cluster.** ADR 0053 sums the scores within each declared cluster.
A panel whose units all fall in one cluster, as one market's zones do, has no second cluster to
sum within, and its units name each period several times, which the periods above refused.
`prescribe` passed the fit neither, and the fit took each transition as independent, as on one
unit; `fit_causal_residual` refused `clusters` that name one cluster, or one in a dimension of
two. The units of one cluster share its shocks. A shock every unit takes in a period moves that
period's scores together across the units, and where the shock persists and the levers do too, the
period's sums persist across the periods. The rows' error reads neither. What reached it:

- `fit_causal_residual`: the rows' sandwich.
- `prescribe`: that error, in `identification_radius`, in the tube's model error, in the report,
  "each transition taken as independent", and in `to_json()`, whose four `error_*` fields read
  `null`.
- `persistence_check`, in `prescribe`: the units' sums by CR1 against `t(G - 1)`, as if the units
  were independent.
- `omitted_confounder_bound` and `misspecification_cost`: the influence's rows, as the fit read
  them.
- The logger check, `gcm_test` clustered by period: a period's sum holds whatever the dependence
  across its units, and under its null the sums are martingale differences in time, so it holds as
  it stands.
- `Prescription.evaluate`: it resamples the units as independent, as it does on several clusters.

**The measurement, on one cluster.** `G` units of one cluster, each
`x' = 0.95 x + 0.1 (0.8 u + 1.5 z) + e`, `u = a + 0.9 z`, `z` drawn afresh and adjusted for. The
noise is a shock every unit shares, AR(`r_e`), plus each unit's own drawn afresh, each of spread
`0.05 / sqrt(2)`. The lever's own part `a` is AR(0.7) of spread 0.5: each unit's own (*own*), or
one every unit shares and the unit's own, summed over `sqrt(2)` (*shared*). 300 periods before the
log, then `T`. Each log is fitted as above; a cell draws 1000 logs, each from its own seed spawned
from one `SeedSequence`. Each reading reads the influence's channel column at the rows' factor:

- *rows*: as above.
- *period sums*: the scores summed within each period over the units, projected on `nu` cosines of
  the `T` periods, squared and summed, `nu = 0.4 T^(2/3)` rounded down, 5 at 50 periods and 13 at
  200, against `t(nu)`: Driscoll and Kraay's (1998) reading, with the cosines in place of their
  Bartlett kernel. It is `fit_causal_residual`'s own error given the periods, which agreed with it
  to 1e-15 on each cell's first log.
- *units, CR1*: the units as clusters (ADR 0053), against `t(G - 1)`.
- *two-way*: by unit and by period, whichever of the three sums reads largest (ADR 0061), against
  `t(min(G, T) - 1)`: what `prescribe` reads where the panel declares no cluster.

A refusal reads no interval. The coverage of the 95 % interval:

| lever | `G` | `r_e` | `T` | rows | period sums | units, CR1 | two-way |
|---|---|---|---|---|---|---|---|
| own | 3 | 0.0 | 50 | 0.962 | 0.962 | 0.993 | 1.000 |
| own | 3 | 0.5 | 50 | 0.920 | 0.952 | 0.983 | 1.000 |
| own | 3 | 0.9 | 50 | 0.886 | 0.955 | 0.982 | 0.998 |
| own | 3 | 0.0 | 200 | 0.955 | 0.948 | 0.974 | 1.000 |
| own | 3 | 0.5 | 200 | 0.906 | 0.950 | 0.965 | 1.000 |
| own | 3 | 0.9 | 200 | 0.798 | 0.931 | 0.959 | 0.998 |
| own | 10 | 0.0 | 50 | 0.959 | 0.956 | 0.955 | 0.986 |
| own | 10 | 0.5 | 50 | 0.908 | 0.950 | 0.953 | 0.971 |
| own | 10 | 0.9 | 50 | 0.850 | 0.945 | 0.957 | 0.967 |
| own | 10 | 0.0 | 200 | 0.956 | 0.953 | 0.954 | 0.982 |
| own | 10 | 0.5 | 200 | 0.897 | 0.954 | 0.956 | 0.969 |
| own | 10 | 0.9 | 200 | 0.759 | 0.934 | 0.922 | 0.933 |
| shared | 3 | 0.0 | 50 | 0.931 | 0.962 | 0.970 | 1.000 |
| shared | 3 | 0.5 | 50 | 0.817 | 0.924 | 0.946 | 0.999 |
| shared | 3 | 0.9 | 50 | 0.820 | 0.934 | 0.954 | 0.997 |
| shared | 3 | 0.0 | 200 | 0.903 | 0.944 | 0.949 | 1.000 |
| shared | 3 | 0.5 | 200 | 0.787 | 0.946 | 0.916 | 1.000 |
| shared | 3 | 0.9 | 200 | 0.692 | 0.934 | 0.897 | 0.992 |
| shared | 10 | 0.0 | 50 | 0.753 | 0.930 | 0.750 | 0.973 |
| shared | 10 | 0.5 | 50 | 0.634 | 0.929 | 0.694 | 0.921 |
| shared | 10 | 0.9 | 50 | 0.586 | 0.914 | 0.665 | 0.814 |
| shared | 10 | 0.0 | 200 | 0.749 | 0.944 | 0.721 | 0.973 |
| shared | 10 | 0.5 | 200 | 0.575 | 0.938 | 0.625 | 0.913 |
| shared | 10 | 0.9 | 200 | 0.442 | 0.909 | 0.586 | 0.812 |

A coverage's Monte Carlo standard error is 0.006 to 0.016. How far each error reads the estimate's
spread across the logs, as above:

| lever | `G` | `r_e` | `T` | rows | period sums | units, CR1 | two-way |
|---|---|---|---|---|---|---|---|
| own | 3 | 0.0 | 50 | 1.104 | 1.137 | 1.284 | 1.424 |
| own | 3 | 0.5 | 50 | 0.909 | 1.012 | 1.125 | 1.218 |
| own | 3 | 0.9 | 50 | 0.797 | 0.969 | 1.127 | 1.190 |
| own | 3 | 0.0 | 200 | 1.017 | 1.017 | 1.071 | 1.223 |
| own | 3 | 0.5 | 200 | 0.847 | 0.995 | 1.052 | 1.141 |
| own | 3 | 0.9 | 200 | 0.668 | 0.939 | 1.024 | 1.066 |
| own | 10 | 0.0 | 50 | 1.052 | 1.025 | 1.060 | 1.167 |
| own | 10 | 0.5 | 50 | 0.840 | 0.950 | 1.010 | 1.057 |
| own | 10 | 0.9 | 50 | 0.738 | 0.910 | 1.019 | 1.046 |
| own | 10 | 0.0 | 200 | 1.028 | 1.019 | 1.019 | 1.123 |
| own | 10 | 0.5 | 200 | 0.832 | 1.009 | 1.016 | 1.054 |
| own | 10 | 0.9 | 200 | 0.630 | 0.925 | 0.964 | 0.978 |
| shared | 3 | 0.0 | 50 | 0.938 | 1.078 | 1.050 | 1.275 |
| shared | 3 | 0.5 | 50 | 0.712 | 0.905 | 0.802 | 0.960 |
| shared | 3 | 0.9 | 50 | 0.660 | 0.906 | 0.826 | 0.939 |
| shared | 3 | 0.0 | 200 | 0.837 | 0.996 | 0.804 | 1.118 |
| shared | 3 | 0.5 | 200 | 0.665 | 0.991 | 0.697 | 0.922 |
| shared | 3 | 0.9 | 200 | 0.509 | 0.918 | 0.634 | 0.762 |
| shared | 10 | 0.0 | 50 | 0.603 | 0.957 | 0.562 | 1.013 |
| shared | 10 | 0.5 | 50 | 0.474 | 0.946 | 0.494 | 0.806 |
| shared | 10 | 0.9 | 50 | 0.382 | 0.833 | 0.453 | 0.630 |
| shared | 10 | 0.0 | 200 | 0.565 | 0.986 | 0.495 | 1.013 |
| shared | 10 | 0.5 | 200 | 0.413 | 0.947 | 0.419 | 0.759 |
| shared | 10 | 0.9 | 200 | 0.308 | 0.875 | 0.386 | 0.580 |

PART2_CONTEXT

## Decision

- **`fit_causal_residual(periods=...)`** takes each transition's period, a whole number of steps,
  on a log whose transitions fall in one group: one unit's, or the units' of one declared cluster.
  It reads the channel's covariance across them by cosines: `V = N / (N - k) sum_j c_j c_j'`, with
  `c_j = sum_i sqrt(2 / nu) cos(pi j (p_i + 1/2) / P) s_i` for `j` from 1 to `nu`, `s_i` the
  transition's scores summed over its states, `p_i` its period counted from the log's first, and
  `P` the periods the log spans. The transitions of one period share its weights, so `c_j` reads
  the sums of each period's scores, whatever their dependence within the period: Driscoll and
  Kraay's (1998) reading, with the cosines in place of their Bartlett kernel. A period the log
  lacks holds no score, as Parzen (1963) reads a series with gaps. `nu` is the most whole cosines
  at or below `0.4 T^(2/3)` for `T` distinct periods, read in whole numbers, `1000 nu^3 <= 64 T^2`,
  so that 1000 periods read 40 and not 39; and one at the least. On one unit `T` is `N`. Periods
  that name one period are refused, since one period's sum leaves no spread across the periods.
  `CausalDynamicsFit.periods` keeps the periods and `error_cosines` gives `nu`. Each row's
  `influence` is the rows' own, factor and all; its readers sum it by the same cosines. `clusters`
  and `periods` together are refused: they are two readings of one dependence. `clusters` that
  name one cluster, or one in a dimension of two, are refused as before, and the refusal points to
  `periods`.
- **A test read off it takes `t(nu)`.** As the periods grow at a fixed `nu`, the `c_j` read as
  independent normals, so the statistic is a `t(nu)` (Sun 2013; Lazarus, Lewis, Stock and Watson
  2018); a period's sum is one term of a series in the periods, as Vogelsang (2012) reads Driscoll
  and Kraay's sums under fixed-b. `omitted_confounder_bound` takes `t(nu)`'s quantile and reads
  both influences by the cosines; `misspecification_cost` reads the difference's covariance by
  them, sizes its test as the next bullet but one says, and refuses two fits read over different
  periods.
- **`prescribe` passes the periods where every transition falls in one group**, one unit or one
  declared cluster, and they start in two periods at least. `DecisionCertificate.error_cosines`
  counts the cosines, `to_json()` writes it, `null` where the error is not read so, and the fit's
  log event carries it as `cosines`. On one unit `error_clustered_by` stays `None`, and the report
  reads "read across the one unit's periods by 21 cosines of them (EWC)" for 399 transitions. On
  one declared cluster `error_clustered_by` names the cluster's column and `error_clusters` reads
  1, and the report reads "summed within each period over the units of the one group of `market`,
  then read across the periods by 8 cosines of them (Driscoll-Kraay, EWC)" for four units over 100
  periods. One group whose transitions all start in one period takes them as independent, as
  before: one period's sum leaves nothing to read across.
- **`persistence_check` reads one unit's pairs, and a fit's given periods, across their periods**
  the same way: the pairs' products summed within each period, and their total over the root of
  the sums' projections on `nu` cosines of the later transitions' periods, squared and summed,
  against `t(nu)`, `nu` counted over the pairs' periods. A fit given periods says its units share
  one group's noise, so its pairs are read so whatever their units' count; given none, the pairs
  of several units are read by CR1 over the units, as before. Pairs that leave no spread, as
  below two pairs or in one period, read `nan`, and the report says the pairs leave no spread to
  read.
- **`misspecification_cost`'s p-value under `periods` reads `S` as the estimate it is.** With `S`'s
  estimate `sum_j c_j c_j'`, the `c_j` independent `N(0, S / nu)`, as the cosines read as the
  periods grow at a fixed `nu`, `dh' W dh` over the estimate's `tr(W S)` is
  `sum_k w_k X_k / (sum_k w_k Y_k / nu)`: `X_k` chi-square with one degree of freedom, `Y_k` with
  `nu`, all independent, and `w_k` the eigenvalues of `W S`. One weight makes it `F(1, nu)`, `m`
  equal ones `F(m, m nu)`, and as `nu` grows it nears the mixture with `S` known. The p-value is
  its tail at the ratio the log reads, `r`: Imhof's (1961) inversion at 0 of
  `sum_k w_k X_k - (r / nu) sum_k w_k Y_k`, read over `log u`, where the integrand falls
  exponentially both ways; the weights are read off the estimate. `validation/cosine_quadratic_law.mac`
  derives the two F laws in Maxima, reads Imhof's inversion against them and against the values
  the tests hold for unequal weights, and checks the approach to the mixture. `cost`,
  `cost_error`, `noise` and `unseen` do not move.
- **Unchanged:** every panel of two groups or more, to the bit; one unit's error; the logger
  check, whose period sums hold whatever the dependence across one period's units; and
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

**Why the period sums.** Over the 24 cells on one cluster they covered 0.909 to 0.962 and read
0.83 to 1.14 of the estimate's spread. The rows covered 0.442 where ten units share a lever and a
shock AR(0.9) over 200 periods. CR1 over the units takes them as independent, which a shared shock
breaks: where they share the lever too it covered 0.586 to 0.750 at ten units, and at three units
it covered by width alone, its interval read off `t(2)` 1.9 to 3.0 times the rows'. Two-way reads
a shock within each period and the persistence within each unit, but not a shock that persists,
which ties one unit's period to another unit's next: where ten units share the lever and the shock
is AR(0.9) it covered 0.814 at 50 periods and 0.812 at 200, and at three units its interval was 2.6
to 3.3 times the rows'. The period sums read the shock within each period and its
persistence across them, against a `t` whose degrees of freedom are the periods' count.

PART2_WHY

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
- *On one declared cluster the error widens where its units share a lever and a shock that
  persists*: at ten units, the shock AR(0.5) or AR(0.9), it reads 0.83 to 0.95 of the estimate's
  spread, where the rows read 0.31 to 0.47, and its interval is 2.5 to 3.0 times the rows'.
- *It widens where nothing persists too*: where each unit moves its own lever and the shock is
  drawn afresh, the mean interval is 21 to 27 % wider than the rows' at 50 periods and 7 to 8 % at
  200, the price of `t(nu)` with `nu` counted over the periods.
- *`prescribe`'s certificate names the one declared cluster*: `error_clustered_by` its column and
  `error_clusters` 1, where all four `error_*` fields read `null`. No key was added, so
  `to_json()`'s schema version stays 2.
- *`persistence_check` on one declared cluster holds its size*: where the units share a shock
  drawn afresh, the units' CR1 rejected 10.75 % of 400 logs of three units over 200 periods at 5 %
  and 28.5 % of ten, and the period sums 5.75 % and 6.0 %. Where the shock is AR(0.3) the period
  sums caught 69 % and 88 %, and AR(0.7) 99.5 % and all.
- PART2_CONSEQUENCE
- *`Prescription.evaluate`* keeps reading one unit's windows as independent, warned of, until 1.0
  refuses them. Its windows are `H + 1` periods long, so a log of a few hundred periods holds a
  few dozen, and cosines over them would read `t(3)` or so; that reading was not measured. On one
  declared cluster it resamples the units as independent, as it does on several clusters, though
  the units of one cluster share its shocks; that reading was not measured either.
- *Tests*: on one unit of 200 transitions with noise AR(0.9), 200 seeded logs, the cosines covered
  at least 0.87 and the rows at most 0.80; the error by hand on Frisch-Waugh-Lovell's partialling,
  with the periods offset, a gap and shuffled; the periods move the error and nothing else, under
  both integrators; the refusals; the cosine count in whole numbers; the bound's `t(21)`, the
  misspecification's `S`, and the persistence check, by hand; the certificate, the report, the
  JSON and the log event on one unit, and `null` on several. On three units of one cluster over
  200 periods, a shock they share AR(0.9) and a lever they half share AR(0.7), 200 seeded logs,
  the period sums covered at least 0.88 and the rows at most 0.79; the error and the influence's
  square summed within each period, by hand, with `nu` counted over the periods; the bound's
  `t(8)`, the misspecification's `S` and the 8 degrees of freedom of its p-value, and the
  persistence check of four units, by hand; `nan` for pairs in one period; the refusals of one
  period and of one cluster; the certificate, the report, the JSON and the log event on one
  declared cluster, and the rows where its transitions all start in one period. The ratio's law
  against `F(1, nu)` and `F(3, 3 nu)` at ratios from `1e-8` to `1e5`, against Maxima's values for
  unequal weights, and nearing the mixture as `nu` grows.

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
- **The rows on one declared cluster**, as before. Rejected: they covered 0.442 to 0.753 where ten
  units share a lever.
- **CR1 over the units of the one cluster**, each unit a cluster. Rejected: it takes the units as
  independent, which their shared shock breaks, and covered 0.586 to 0.750 where ten units share
  a lever.
- **Two-way, by unit and by period**, as on a panel that declares no cluster (ADR 0061). Rejected:
  it misses a shock that persists, 0.814 and 0.812 where ten units share a lever and the shock is
  AR(0.9), and at three units its interval was 2.6 to 3.3 times the rows'.
- **`clusters` that name one cluster read as the period sums.** Rejected: an argument named for
  clusters would read the periods, which it is not given; the refusal names the argument to give.
- **A refusal on one declared cluster.** Rejected: one market's zones are a common panel, and the
  period sums read it at the coverage the table gives.
PART2_ALTERNATIVES
- **`periods` read on several units as well**, a unit's own cosines summed. Not built: CR1 within
  units already holds whatever the dependence within a unit, and ADR 0061's second way reads the
  periods' shocks.
