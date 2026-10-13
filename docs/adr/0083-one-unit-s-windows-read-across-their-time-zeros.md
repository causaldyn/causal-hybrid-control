# ADR 0083 — One unit's windows read across their time zeros

**Status:** accepted, 2026-10-12. Amends ADR 0049, which deprecated `Prescription.evaluate` on
windows of one unit and would have refused them from 1.0, and ADR 0075, which left that reading
unmeasured. Amended by ADR 0095, 2026-10-13: the shortfall at a horizon of ten steps is the
spread's, not the estimate's bias, which is 0.08 to 0.12 of the half-width there.

## Context

`Prescription.evaluate` reads its intervals from a bootstrap over the units, the whole analysis
run again on each draw (ADR 0049). A panel of one unit has no units to draw. Since 0.12.0 its
windows were read as before the bootstrap: `evaluate_plan` by `"pdis"`, each window taken as
independent, the half-width 1.96 standard deviations of the windows' influences over the square
root of their count. The call warned with a `DeprecationWarning`, logged `chc_event="one_unit"`
at `WARNING`, and from 1.0 was to raise `DecisionError`. No measurement stood behind either the
reading or the refusal.

One unit's windows follow each other. Consecutive windows share a boundary period, the state a
window starts from is where the one before it ended, and noise that persists carries across
windows. Each window's influence is a function of its states, so the influences persist where the
state or the noise does, and the spread of their sum is more than the sum of their squares. One
unit is no corner case: a building, a pendulum and a single market are each one series. ADR 0075
reads one unit's channel error across its periods by cosines and noted that the windows, `H + 1`
periods long, are fewer: a log of a few hundred periods holds a few dozen windows, and cosines
over them read `t(3)` or so.

## The measurement

One unit, one state and one lever: `x' = a x + u + e`, the noise `e` AR(`rho`) of spread 0.5,
logged by `u = -0.1 x + 0.2 + N(0, 1)`, so the loop keeps `a - 0.1` of the state a step: `a` is
0.6 or 0.9, the loop 0.5 or 0.8. The plan is open loop over `H` steps, the logger's mean action at
its stationary mean state plus `0.3 cos(0.7 t)`; the cost is `x^2 / 2 + 0.1 u^2 / 2`, target 0.
A log runs 300 periods before it starts, then `E H + 1` periods, cut into the `E` windows on the
calendar that `Prescription.evaluate` cuts. Each log is read by `evaluate_plan` by `"pdis"` with
the true `a` and `b`, the logger fitted and the smoothing chosen on the log, `model_error=0`: the
smoothing's correction is exact here, so the interval is the spread's reading alone. The truth is
the plan's expected cost over the horizon from the stationary law of the state and the last noise
together, propagated on the system `(x, e)' = [[a, rho], [0, rho]] (x, e) + (u, 0) + (1, 1) d`; at
three cells it agreed with Maxima's exact rational solve to 1e-15 and with 2 000 000 simulated
starts within their error. A cell is 1000 logs, each from its own stream of a `SeedSequence`. Each
reading uses the same influences, the delta method's:

- *independent*: the windows' own interval, `evaluate_plan`'s, against the normal's quantile.
- *cosines*: the equal-weighted cosine estimate over the windows' time zeros, `nu` cosines with
  `nu` the most whole ones at or below `0.4 E^(2/3)` as ADR 0075 counts them, against `t(nu)`
  (Lazarus, Lewis, Stock and Watson 2018).
- *blocks*: `nu + 1` contiguous blocks of windows as clusters, CR1, against `t(nu)` (Bester,
  Conley and Hansen 2011).
- *half the cosines*: the cosines' reading at `max(floor(nu / 2), 1)`.

A refused log, one the certificate refuses, reads no interval and counts in its own column. With
`min_effective=10`, so that 25 to 800 windows can be read:

| loop | noise AR | H | windows | nu | refused | windows as independent | cosines, t(nu) | blocks, t(nu) | width, cosines / independent |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.0 | 4 | 25 | 3 | 79 | 0.899 | 0.953 | 0.945 | 1.66 |
| 0.5 | 0.0 | 4 | 50 | 5 | 0 | 0.929 | 0.955 | 0.961 | 1.37 |
| 0.5 | 0.0 | 4 | 100 | 8 | 0 | 0.944 | 0.966 | 0.967 | 1.24 |
| 0.5 | 0.0 | 4 | 200 | 13 | 0 | 0.966 | 0.981 | 0.979 | 1.17 |
| 0.5 | 0.0 | 4 | 400 | 21 | 0 | 0.976 | 0.984 | 0.988 | 1.13 |
| 0.5 | 0.0 | 4 | 800 | 34 | 0 | 0.986 | 0.988 | 0.989 | 1.11 |
| 0.5 | 0.0 | 10 | 25 | 3 | 426 | 0.944 | 0.953 | 0.965 | 1.57 |
| 0.5 | 0.0 | 10 | 50 | 5 | 6 | 0.950 | 0.965 | 0.971 | 1.32 |
| 0.5 | 0.0 | 10 | 100 | 8 | 0 | 0.958 | 0.964 | 0.967 | 1.18 |
| 0.5 | 0.0 | 10 | 200 | 13 | 0 | 0.972 | 0.972 | 0.978 | 1.11 |
| 0.5 | 0.0 | 10 | 400 | 21 | 0 | 0.967 | 0.972 | 0.976 | 1.08 |
| 0.5 | 0.0 | 10 | 800 | 34 | 0 | 0.980 | 0.985 | 0.982 | 1.06 |
| 0.5 | 0.5 | 4 | 25 | 3 | 66 | 0.895 | 0.938 | 0.940 | 1.68 |
| 0.5 | 0.5 | 4 | 50 | 5 | 0 | 0.903 | 0.933 | 0.935 | 1.41 |
| 0.5 | 0.5 | 4 | 100 | 8 | 0 | 0.942 | 0.961 | 0.957 | 1.27 |
| 0.5 | 0.5 | 4 | 200 | 13 | 0 | 0.957 | 0.971 | 0.970 | 1.19 |
| 0.5 | 0.5 | 4 | 400 | 21 | 0 | 0.957 | 0.982 | 0.975 | 1.15 |
| 0.5 | 0.5 | 4 | 800 | 34 | 0 | 0.971 | 0.982 | 0.976 | 1.12 |
| 0.5 | 0.5 | 10 | 25 | 3 | 409 | 0.920 | 0.954 | 0.956 | 1.63 |
| 0.5 | 0.5 | 10 | 50 | 5 | 7 | 0.927 | 0.952 | 0.950 | 1.30 |
| 0.5 | 0.5 | 10 | 100 | 8 | 0 | 0.937 | 0.949 | 0.956 | 1.19 |
| 0.5 | 0.5 | 10 | 200 | 13 | 0 | 0.951 | 0.964 | 0.965 | 1.12 |
| 0.5 | 0.5 | 10 | 400 | 21 | 0 | 0.954 | 0.956 | 0.958 | 1.08 |
| 0.5 | 0.5 | 10 | 800 | 34 | 0 | 0.965 | 0.972 | 0.968 | 1.06 |
| 0.5 | 0.9 | 4 | 25 | 3 | 74 | 0.752 | 0.870 | 0.887 | 1.91 |
| 0.5 | 0.9 | 4 | 50 | 5 | 0 | 0.788 | 0.875 | 0.874 | 1.61 |
| 0.5 | 0.9 | 4 | 100 | 8 | 0 | 0.840 | 0.908 | 0.908 | 1.47 |
| 0.5 | 0.9 | 4 | 200 | 13 | 0 | 0.870 | 0.928 | 0.928 | 1.39 |
| 0.5 | 0.9 | 4 | 400 | 21 | 0 | 0.912 | 0.959 | 0.953 | 1.33 |
| 0.5 | 0.9 | 4 | 800 | 34 | 0 | 0.901 | 0.955 | 0.956 | 1.32 |
| 0.5 | 0.9 | 10 | 25 | 3 | 413 | 0.860 | 0.930 | 0.927 | 1.71 |
| 0.5 | 0.9 | 10 | 50 | 5 | 8 | 0.865 | 0.932 | 0.917 | 1.43 |
| 0.5 | 0.9 | 10 | 100 | 8 | 0 | 0.881 | 0.923 | 0.926 | 1.27 |
| 0.5 | 0.9 | 10 | 200 | 13 | 0 | 0.912 | 0.946 | 0.945 | 1.20 |
| 0.5 | 0.9 | 10 | 400 | 21 | 0 | 0.939 | 0.956 | 0.961 | 1.16 |
| 0.5 | 0.9 | 10 | 800 | 34 | 0 | 0.931 | 0.953 | 0.956 | 1.13 |
| 0.8 | 0.0 | 4 | 25 | 3 | 259 | 0.792 | 0.911 | 0.930 | 2.07 |
| 0.8 | 0.0 | 4 | 50 | 5 | 7 | 0.826 | 0.913 | 0.914 | 1.67 |
| 0.8 | 0.0 | 4 | 100 | 8 | 0 | 0.855 | 0.932 | 0.934 | 1.53 |
| 0.8 | 0.0 | 4 | 200 | 13 | 0 | 0.892 | 0.961 | 0.960 | 1.42 |
| 0.8 | 0.0 | 4 | 400 | 21 | 0 | 0.900 | 0.953 | 0.952 | 1.36 |
| 0.8 | 0.0 | 4 | 800 | 34 | 0 | 0.936 | 0.977 | 0.975 | 1.33 |
| 0.8 | 0.0 | 10 | 25 | 3 | 887 | 0.912 | 0.947 | 0.947 | 1.81 |
| 0.8 | 0.0 | 10 | 50 | 5 | 502 | 0.904 | 0.934 | 0.938 | 1.44 |
| 0.8 | 0.0 | 10 | 100 | 8 | 86 | 0.898 | 0.927 | 0.925 | 1.25 |
| 0.8 | 0.0 | 10 | 200 | 13 | 2 | 0.913 | 0.931 | 0.926 | 1.19 |
| 0.8 | 0.0 | 10 | 400 | 21 | 0 | 0.933 | 0.950 | 0.951 | 1.13 |
| 0.8 | 0.0 | 10 | 800 | 34 | 0 | 0.941 | 0.958 | 0.960 | 1.11 |
| 0.8 | 0.5 | 4 | 25 | 3 | 336 | 0.795 | 0.917 | 0.917 | 2.08 |
| 0.8 | 0.5 | 4 | 50 | 5 | 23 | 0.814 | 0.907 | 0.902 | 1.67 |
| 0.8 | 0.5 | 4 | 100 | 8 | 0 | 0.856 | 0.916 | 0.918 | 1.51 |
| 0.8 | 0.5 | 4 | 200 | 13 | 0 | 0.879 | 0.938 | 0.936 | 1.39 |
| 0.8 | 0.5 | 4 | 400 | 21 | 0 | 0.893 | 0.945 | 0.941 | 1.35 |
| 0.8 | 0.5 | 4 | 800 | 34 | 0 | 0.911 | 0.966 | 0.962 | 1.31 |
| 0.8 | 0.5 | 10 | 25 | 3 | 925 | 0.893 | 0.960 | 0.933 | 1.76 |
| 0.8 | 0.5 | 10 | 50 | 5 | 599 | 0.840 | 0.920 | 0.908 | 1.41 |
| 0.8 | 0.5 | 10 | 100 | 8 | 139 | 0.821 | 0.878 | 0.872 | 1.23 |
| 0.8 | 0.5 | 10 | 200 | 13 | 1 | 0.879 | 0.910 | 0.912 | 1.18 |
| 0.8 | 0.5 | 10 | 400 | 21 | 0 | 0.899 | 0.937 | 0.934 | 1.13 |
| 0.8 | 0.5 | 10 | 800 | 34 | 0 | 0.927 | 0.947 | 0.950 | 1.10 |
| 0.8 | 0.9 | 4 | 25 | 3 | 425 | 0.637 | 0.850 | 0.845 | 2.34 |
| 0.8 | 0.9 | 4 | 50 | 5 | 61 | 0.685 | 0.838 | 0.824 | 1.86 |
| 0.8 | 0.9 | 4 | 100 | 8 | 0 | 0.715 | 0.854 | 0.852 | 1.68 |
| 0.8 | 0.9 | 4 | 200 | 13 | 0 | 0.762 | 0.886 | 0.876 | 1.55 |
| 0.8 | 0.9 | 4 | 400 | 21 | 0 | 0.824 | 0.920 | 0.917 | 1.47 |
| 0.8 | 0.9 | 4 | 800 | 34 | 0 | 0.824 | 0.918 | 0.915 | 1.43 |
| 0.8 | 0.9 | 10 | 25 | 3 | 976 | 0.667 | 0.792 | 0.708 | 1.84 |
| 0.8 | 0.9 | 10 | 50 | 5 | 827 | 0.711 | 0.821 | 0.832 | 1.48 |
| 0.8 | 0.9 | 10 | 100 | 8 | 454 | 0.755 | 0.821 | 0.835 | 1.32 |
| 0.8 | 0.9 | 10 | 200 | 13 | 67 | 0.763 | 0.823 | 0.816 | 1.20 |
| 0.8 | 0.9 | 10 | 400 | 21 | 0 | 0.769 | 0.817 | 0.819 | 1.15 |
| 0.8 | 0.9 | 10 | 800 | 34 | 0 | 0.813 | 0.848 | 0.845 | 1.11 |

With the default `min_effective=100`, 400 to 1600 windows:

| loop | noise AR | H | windows | nu | refused | windows as independent | cosines, t(nu) | blocks, t(nu) | width, cosines / independent |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.0 | 4 | 400 | 21 | 0 | 0.982 | 0.987 | 0.990 | 1.13 |
| 0.5 | 0.0 | 4 | 800 | 34 | 0 | 0.984 | 0.991 | 0.992 | 1.10 |
| 0.5 | 0.0 | 4 | 1600 | 54 | 0 | 0.985 | 0.985 | 0.990 | 1.09 |
| 0.5 | 0.0 | 10 | 400 | 21 | 0 | 0.979 | 0.980 | 0.978 | 1.09 |
| 0.5 | 0.0 | 10 | 800 | 34 | 0 | 0.994 | 0.993 | 0.992 | 1.06 |
| 0.5 | 0.0 | 10 | 1600 | 54 | 0 | 0.982 | 0.981 | 0.984 | 1.04 |
| 0.5 | 0.5 | 4 | 400 | 21 | 0 | 0.971 | 0.982 | 0.981 | 1.15 |
| 0.5 | 0.5 | 4 | 800 | 34 | 0 | 0.957 | 0.973 | 0.970 | 1.13 |
| 0.5 | 0.5 | 4 | 1600 | 54 | 0 | 0.978 | 0.988 | 0.985 | 1.11 |
| 0.5 | 0.5 | 10 | 400 | 21 | 0 | 0.964 | 0.967 | 0.970 | 1.09 |
| 0.5 | 0.5 | 10 | 800 | 34 | 0 | 0.966 | 0.966 | 0.966 | 1.06 |
| 0.5 | 0.5 | 10 | 1600 | 54 | 0 | 0.979 | 0.984 | 0.981 | 1.05 |
| 0.5 | 0.9 | 4 | 400 | 21 | 0 | 0.891 | 0.959 | 0.956 | 1.35 |
| 0.5 | 0.9 | 4 | 800 | 34 | 0 | 0.904 | 0.958 | 0.953 | 1.32 |
| 0.5 | 0.9 | 4 | 1600 | 54 | 0 | 0.915 | 0.969 | 0.969 | 1.30 |
| 0.5 | 0.9 | 10 | 400 | 21 | 0 | 0.931 | 0.954 | 0.940 | 1.15 |
| 0.5 | 0.9 | 10 | 800 | 34 | 0 | 0.935 | 0.951 | 0.954 | 1.12 |
| 0.5 | 0.9 | 10 | 1600 | 54 | 0 | 0.950 | 0.967 | 0.967 | 1.10 |
| 0.8 | 0.0 | 4 | 400 | 21 | 0 | 0.894 | 0.952 | 0.952 | 1.37 |
| 0.8 | 0.0 | 4 | 800 | 34 | 0 | 0.942 | 0.981 | 0.980 | 1.33 |
| 0.8 | 0.0 | 4 | 1600 | 54 | 0 | 0.948 | 0.984 | 0.986 | 1.31 |
| 0.8 | 0.0 | 10 | 400 | 21 | 595 | 0.968 | 0.978 | 0.980 | 1.14 |
| 0.8 | 0.0 | 10 | 800 | 34 | 0 | 0.925 | 0.953 | 0.950 | 1.12 |
| 0.8 | 0.0 | 10 | 1600 | 54 | 0 | 0.949 | 0.965 | 0.959 | 1.09 |
| 0.8 | 0.5 | 4 | 400 | 21 | 0 | 0.889 | 0.955 | 0.957 | 1.35 |
| 0.8 | 0.5 | 4 | 800 | 34 | 0 | 0.910 | 0.969 | 0.962 | 1.31 |
| 0.8 | 0.5 | 4 | 1600 | 54 | 0 | 0.932 | 0.977 | 0.976 | 1.29 |
| 0.8 | 0.5 | 10 | 400 | 21 | 855 | 0.931 | 0.945 | 0.952 | 1.14 |
| 0.8 | 0.5 | 10 | 800 | 34 | 8 | 0.903 | 0.936 | 0.931 | 1.10 |
| 0.8 | 0.5 | 10 | 1600 | 54 | 0 | 0.903 | 0.926 | 0.925 | 1.08 |
| 0.8 | 0.9 | 4 | 400 | 21 | 0 | 0.794 | 0.909 | 0.906 | 1.50 |
| 0.8 | 0.9 | 4 | 800 | 34 | 0 | 0.841 | 0.936 | 0.937 | 1.41 |
| 0.8 | 0.9 | 4 | 1600 | 54 | 0 | 0.849 | 0.946 | 0.940 | 1.40 |
| 0.8 | 0.9 | 10 | 400 | 21 | 1000 | – | – | – | – |
| 0.8 | 0.9 | 10 | 800 | 34 | 688 | 0.785 | 0.837 | 0.830 | 1.11 |
| 0.8 | 0.9 | 10 | 1600 | 54 | 0 | 0.818 | 0.856 | 0.852 | 1.09 |

A coverage's Monte Carlo standard error is 0.003 to 0.015 where every log is read, more where many
are refused. Of the 98 cells in which at least half the logs were read, the windows taken as
independent covered 0.64 to 0.99 and the cosines 0.82 to 0.99; the cosines covered less in two,
by 0.001, where the noise is fresh and the horizon 10 steps. Where the loop keeps 0.8 and the noise
is AR(0.9), the independent windows covered 0.64 to 0.85 and the cosines 0.82 to 0.95. The
blocks read as the cosines do, within the Monte Carlo error.

The rest of the shortfall at a horizon of 10 steps in the most persistent loop is the estimate's,
not its spread's: its mean error there is 0.07 to 0.33 of its standard deviation below the truth,
the self-normalised weights over ten steps of a persistent state. Halving the cosines adds at most
0.024 from 200 windows up, and below that buys coverage by reading `t(1)` to `t(4)`.

## Decision

- **One unit's windows are read across their time zeros by cosines, against `t(nu)`.** Each
  window's influence on the estimate, the delta method's, is projected on `nu` cosines of the
  windows' time zeros, `sqrt(2 / nu) cos(pi j (p + 1/2) / P)` for `j` from 1 to `nu`, at the
  window's place `p` over the `P` places the windows span; the squared projections summed are the
  equal-weighted cosine estimate of the spread of the influences' sum. The half-width is Student's
  `t(nu)` 97.5 % point times its square root over the window count, widened by `model_error`
  times the model's correction as `evaluate_plan` widens its own. `nu` is ADR 0075's count, over
  the windows: one convention for one unit across the library.
- **A window's place** is its time zero on the grid of `H` periods, counted from the earliest:
  `(start - earliest start) // H`. On the calendar's windows that is exact, and a gap that drops
  windows leaves their places empty, as Parzen (1963) reads a series with gaps. Cut back from each
  run's end (`time_zero="unit"`), windows of different runs start at least `H + 2` periods apart,
  so each keeps a place of its own.
- **The plan against the logger** is read the same way, off the difference's influences: the
  plan's less each window's logged cost about their mean. `PlanEvaluation.versus_logger` is now
  filled on one unit.
- `PlanEvaluation.degrees_of_freedom` holds `nu`, and `bootstrap` stays `None`: nothing was
  resampled. The logger and the smoothing are held at the estimate's, as the delta method holds
  them.
- **No warning and no refusal at 1.0.** The call logs `chc_event="one_unit"` at `INFO`, with the
  windows and the cosines counted.
- The code lives in `chc.evaluation._evaluate_across_time`, beside `_evaluate_by_unit`; the cosines
  are `chc.dynamics_id._cosines`, so the count and the frequencies are ADR 0075's own.

## Consequences

- The interval is wider where the windows are close to independent, by the price of the cosines'
  `t`: where the loop keeps 0.5 and the noise is fresh, 1.66 times at 25 windows of four steps,
  1.24 at 100, 1.11 at 800 and 1.04 to 1.09 at 1600.
- The values do not move: the estimate, the smoothing and the certificate are `evaluate_plan`'s on
  the same windows. On the target-trial test's zone of 1200 windows the values are those pinned
  before, to `1e-9`.
- The coverage still falls short of 95 % where the state and the noise both persist and the
  horizon is long, through the estimate's own bias; the docstring gives the range.
- A unit of few windows reads a wide interval, `t(3)` at 25 windows, where the certificate passes
  it at all: under `min_effective=10` it refused 66 to 976 of 1000 such logs, and under the default
  100 it refused every log of the most persistent loop at 400 windows of ten steps.
- `tests/test_evaluation.py` holds the reading by hand, the weights, the correction, the influences
  and each cosine's projection written out, and the coverage on 200 logs of the most persistent
  loop with 200 windows of four steps: the independent windows covered 160, the cosines 185.
  `tests/test_target_trial.py` holds `Prescription.evaluate` on one zone to the reading by hand on
  windows it cuts itself, on the calendar, across a gap on the calendar, and cut back from each
  run's end, and holds that no warning is raised.
- Of 16 mutations of the change, 15 fail a test. The one left reads each window's place off its
  last period rather than its first, which moves every place by the same `H` periods and so none.
  Reading the places in reverse is a reflection where they are symmetric about their middle, which
  the squared projections cannot see; the gap test drops a period off the middle so they are not.

## Alternatives considered

- **Keeping the deprecation and refusing from 1.0.** Rejected: the windows' influences across their
  time zeros read an interval as close to 95 % as the channel's error reads on one unit, and a
  refusal would leave a single building or market with no evaluation at all.
- **Keeping the independent reading without the warning.** Rejected: it covered as little as 0.64.
- **Half as many cosines.** Rejected: from 200 windows up it adds at most 0.024, below that it
  buys coverage at `t(1)` to `t(4)`, and it would read one unit by a count other than the channel
  error's.
- **Blocks of windows as clusters.** Covered as the cosines do; rejected as a second convention
  for the same reading.
- **A moving-block bootstrap of the windows, the whole analysis inside**, as ADR 0049 runs it over
  units. Not built: ADR 0075 measured the moving-block bootstrap of one unit's transitions at 0.63
  to 0.93, below the cosines in every cell, and each draw would run the analysis again.
