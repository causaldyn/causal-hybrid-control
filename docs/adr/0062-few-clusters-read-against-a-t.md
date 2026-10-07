# ADR 0062 — Few clusters read against a t with G - 1 degrees of freedom

**Status:** accepted, 2026-10-07. Amends ADR 0053.

## Context

A channel error summed within clusters is read from `G` independent terms, the clusters' summed
scores. At few clusters its square is itself noisy, roughly a chi-square with `G - 1` degrees of
freedom over `G - 1`, and a test that reads it against the normal's quantile rejects too often.
ADR 0053 measured it and left the remedy unbuilt: at 5 units a 5 % test of the channel rejected
9.75 % of panels under AR noise and 14.75 % where half the noise is the unit's own; at 40 units
under the latter it still rejected 9.5 %. ADR 0061's two-way error inherits the same flaw at few
periods: at 5 periods under a shared shock, two ways rejected 19.8 %.

The one interval the library reads off the clustered spread is `omitted_confounder_bound`'s pair
of confidence bounds, and with them its `robustness_value_ci`. They took the normal's quantile,
as DoubleML does.

`scripts/bench_clustered_error.py`, section `size`, reads ADR 0053's panels again, 400 a cell,
with one more column: the same error summed within units, against `t(G - 1)`'s quantile. The
other two columns reproduce ADR 0053's table to the panel.

| noise | units `G` | `dt` | size, by unit | size, by unit at `t(G - 1)` |
|---|---|---|---|---|
| AR(0.7) | 5 | 0.1 | 0.0975 | 0.0325 |
| AR(0.7) | 10 | 0.1 | 0.0925 | 0.0550 |
| AR(0.7) | 20 | 0.1 | 0.0850 | 0.0675 |
| AR(0.7) | 40 | 0.1 | 0.0525 | 0.0500 |
| AR(0.7) | 80 | 0.1 | 0.0650 | 0.0525 |
| half the unit's own | 5 | 0.1 | 0.1475 | 0.0575 |
| half the unit's own | 10 | 0.1 | 0.1125 | 0.0725 |
| half the unit's own | 20 | 0.1 | 0.0925 | 0.0800 |
| half the unit's own | 40 | 0.1 | 0.0950 | 0.0925 |
| half the unit's own | 80 | 0.1 | 0.0375 | 0.0350 |
| AR(0.7) | 40 | 1.0 | 0.0650 | 0.0525 |
| half the unit's own | 40 | 1.0 | 0.0875 | 0.0800 |
| independent | 40 | 0.1 | 0.0650 | 0.0600 |
| independent | 40 | 1.0 | 0.0625 | 0.0575 |

A size's Monte Carlo standard error is 1.1 points near 5 % and 1.5 near 10 %.

## Decision

- **`omitted_confounder_bound` takes a `t`'s quantile** with `G - 1` degrees of freedom: `G` the
  clusters, two-way the smaller dimension's count, the one ADR 0061's CR1 factor counts; or the
  rows, where the fit has no clusters, as ADR 0053's row sandwich is the clustered one with every
  row its own cluster. A log's repeated rows clustered by their original therefore still bound as
  the original does: 400 rows and 400 clusters both read `t(399)`.
- **The channel's error is unchanged**, and stays a scale: `prescribe` reads no quantile off it.
  Its comment says a test read off it under `clusters` is sized better against `t(G - 1)`.

## Consequences

- *Where clusters are few, the bounds widen*: one-sided at 95 %, by `t(4) / z = 2.132 / 1.645`,
  30 %, at 5 clusters; by 11 % at 10, 5.1 % at 20 and 2.4 % at 40.
- *With no clusters they widen by `t(N - 1) / z`*: 0.23 % at 400 rows, 0.05 % at 2000.
- *The size.* At 5 units the `t` brings it from 9.75 % and 14.75 % to 3.25 % and 5.75 %, and at
  10 from 9.25 % and 11.25 % to 5.5 % and 7.25 %. From 20 units the quantiles differ by 6.8 % or
  less, and the sizes by 1.75 points or less. Where half the noise is the unit's own, 8.0 % at 20
  units and 9.25 % at 40 remain, which the `t` does not repair. Two ways, at 5 periods (ADR 0061's
  table), the `t` brings 19.8 % to 10.2 % under the shared shock, and 4.8 % to 1.2 % without it.
- *`misspecification_cost`'s test* still reads its chi-square mixture off the clustered
  covariance. Its few-cluster analogue is a statistic over several directions at once, not a `t`;
  it is not built.
- *Tests*: two levels' widths stand as `t(G - 1)`'s quantiles, at 4 clusters, at 3 periods two
  ways in either order, and at 400 rows with none, and not as the normal's; the repeated rows
  still bound as their original, every field to 1e-6.

## Alternatives

- **A wild cluster bootstrap** (Cameron, Gelbach and Miller 2008), restricted to the null with
  Rademacher or Webb's weights. It sizes best at few clusters in linear regression. Not built:
  each draw re-estimates the channel under the null through the cross-fitted nuisances, so it
  costs the fit's time times the draws.
- **The cluster jackknife, CV3** (MacKinnon, Nielsen and Webb 2023), which they find more
  reliable than CR1 at few clusters. Not built: it refits the channel `G` times, each time without
  one cluster. The influence's first-order reading of those refits is CR1's sum again; what CV3
  adds is each cluster's leverage, which needs the refit.
- **CR2 with Bell and McCaffrey's degrees of freedom** (Imbens and Kolesár 2016). Not built: it
  needs the fit's whole linear map from the rates to the channel, cross-fitted nuisances included,
  as a matrix of `N` by `N`.
- **The normal's quantile, as DoubleML's.** Rejected: at 5 clusters it rejects twice to three
  times as often as it says.
