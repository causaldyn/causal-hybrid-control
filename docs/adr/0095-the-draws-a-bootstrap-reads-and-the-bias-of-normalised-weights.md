# ADR 0095 — The draws a bootstrap reads, and the bias of normalised weights

**Status:** accepted, 2026-10-13. Amends ADR 0049, whose bootstrap counted the draws it refused
and said nothing more, and ADR 0083, which put the shortfall of one unit's coverage at a horizon
of ten steps down to the estimate's bias. Corrected 2026-10-13: the coverage at ten steps of
the most persistent unit is 0.83 to 0.85, not 0.86 to 0.87; see the note under *What the table
says*.

## Context

`Prescription.evaluate` reads its intervals from a bootstrap over the units, the whole evaluation
run again on each draw (ADR 0049). A draw the certificate refuses is left out and counted in
`UnitBootstrap.refused`; the intervals read the rest. Near the certificate's edge that is not a
random rest. A draw is refused where its units start wider, or weigh heavier, than the panel's,
so the draws left are the narrow side of the spread, and an interval read off them is conditional
on certification. The record said how many draws were refused and not what that does to the
interval, and no share of refusals made the reading anything else.

`evaluate_plan` by `"pdis"` normalises each step's weights to mean one. The estimate is then a
ratio of two means, biased at order `1 / n`, and nothing reported that bias or its size. ADR 0083
measured it on one unit whose state and noise persist, 0.07 to 0.33 of the estimate's standard
deviation below the truth at a horizon of ten steps, and left it there.

## The measurements

### The bootstrap at the certificate's edge

The tests' market: two states, `x' = a x + b u + w` with `a = [[0.85, 0.15], [0.25, 0.75]]`,
`b = (-0.6, -0.2)`, `w ~ N(0, diag(0.05, 0.03))`, logged by `u = 0.3 x_1 + 0.2 x_2 + 0.1 + N(0, 1)`,
and the cost `x' diag(2, 1) x / 2 + 0.4 u^2 / 2`. The plan is the five-step ramp the tests
evaluate, its feedback `0.4 + 0.3 t` times the LQ gain and its offset walking from 0.3 to -0.2,
smoothed as the certificate chooses. A panel is 60 units, each one run of the logger from its
stationary law cut into five windows of five steps. The plant is the market, `model_error` is 0,
the logger is fitted on every draw, and a bootstrap takes 100 draws. The panel's own edge, the
largest `min_effective` at which it certifies, is found by bisection, and the bootstrap is run at
`c` times it for `c` from 0.5 to 0.98, so that the refused share rises on the same panel. The
truth is the plan's cost from the logger's stationary law, by a Lyapunov solve and covariance
propagation. 300 panels, each from its own stream of a `SeedSequence`:

| refused share | readings | covered | difference covered | half-width / the same panel's with none refused |
|---|---|---|---|---|
| 0 | 838 | 0.926 | 0.939 | 1.000 |
| (0, 0.02] | 217 | 0.931 | 0.926 | 0.987 |
| (0.02, 0.05] | 149 | 0.960 | 0.946 | 0.972 |
| (0.05, 0.10] | 163 | 0.914 | 0.920 | 0.966 |
| (0.10, 0.20] | 165 | 0.952 | 0.909 | 0.945 |
| (0.20, 0.30] | 207 | 0.918 | 0.918 | 0.933 |
| (0.30, 0.50] | 346 | 0.928 | 0.913 | 0.910 |
| (0.50, 1] | 15 | 1.000 | 0.933 | 0.868 |

A coverage over one or two hundred readings has a Monte Carlo error of 0.015 to 0.02, too coarse
to separate a bin from its neighbours. The half-width read against the same panel's is not:
refusing draws narrows the interval, by at most 3.4 % while a tenth of the draws or fewer are
refused, by 5 % to 9 % from a tenth to a half, and by 13 % past a half. An interval 3.4 %
narrower covers about 0.008 less than the same one would; 9 % narrower, about 0.025 less.

120 panels of 30 units of ten windows, the windows of a unit more of them and more dependent,
read the same way, a little narrower: 0.983, 0.969 and 0.958 of the half-width up to 2 %, 5 %
and 10 % refused, 0.936 to 0.886 from a tenth to a half. There the coverage on the same panels,
read at each `c`, fell with the share: 0.925 with none refused, 0.917 at a mean of 3 % refused,
0.908 at 10 %, 0.892 at 27 % and 0.858 at 41 %. On the 60 units it fell from 0.933 to 0.930, 0.930,
0.927 and 0.923 at the same shares.

### The bias of the normalised weights

ADR 0083's unit: `x' = 0.9 x + u + e`, `e` AR(0.9) of spread 0.5, logged by
`u = -0.1 x + 0.2 + N(0, 1)`, so that the loop keeps 0.8 of the state a step; the plan is open
loop, the logger's mean action at its stationary mean plus `0.3 cos(0.7 t)`; the cost is
`x^2 / 2 + 0.1 u^2 / 2`. Each log is one run cut into `E` windows of `H` steps and read as
`Prescription.evaluate` reads one unit, by cosines (ADR 0083), with the plant's `a` and `b`, the
logger fitted, the smoothing chosen and `model_error=0`, under `min_effective=10` so that the long
horizons are read at all. The truth is ADR 0083's, on the state and the last noise together. 600
logs a cell, 300 at 20 steps:

| noise AR | H | windows | read | covered | mean error, sd | mean error, half-width | first order, half-width | measured / first order | flagged | covered, flagged / not |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.9 | 5 | 200 | 600 | 0.882 | -0.194 | -0.094 | -0.022 | 4.2 | 0.04 | 0.62 / 0.89 |
| 0.9 | 5 | 400 | 600 | 0.908 | -0.053 | -0.027 | -0.013 | 2.0 | 0.00 | – / 0.91 |
| 0.9 | 5 | 800 | 600 | 0.925 | -0.171 | -0.084 | -0.009 | 9.0 | 0.00 | – / 0.93 |
| 0.9 | 10 | 400 | 600 | 0.868 | -0.151 | -0.089 | -0.037 | 2.4 | 0.35 | 0.74 / 0.94 |
| 0.9 | 10 | 800 | 600 | 0.865 | -0.214 | -0.123 | -0.023 | 5.4 | 0.03 | 0.63 / 0.87 |
| 0.9 | 10 | 1600 | 600 | 0.862 | -0.113 | -0.082 | -0.014 | 6.0 | 0.00 | – / 0.86 |
| 0.9 | 20 | 3200 | 160 of 300 | 0.625 | -0.593 | -0.404 | -0.106 | 3.8 | 0.94 | 0.60 / 1.00 |
| 0.9 | 20 | 6400 | 300 | 0.633 | -0.271 | -0.302 | -0.073 | 4.1 | 0.83 | 0.58 / 0.92 |
| 0.0 | 10 | 800 | 600 | 0.940 | -0.113 | -0.055 | -0.021 | 2.6 | 0.01 | 0.40 / 0.94 |

*First order* is the bias below, read on each log; *flagged* is the share of logs on which it
passed 0.05 of the half-width. A mean error's Monte Carlo error is 0.04 to 0.08 of the estimate's
standard deviation.

The first-order bias of a ratio of means over `n` independent episodes is
`-sum_t E_b[W_t^2] (E*_t[c_t] - J_t) / n` (Cochran, *Sampling Techniques*, 1977, section 6.8),
where `E*_t` reads step `t`'s state and action under the logger's law tilted by `W_t^2` and `J_t`
is the plan's expected cost at the step. On a linear-Gaussian loop every term is Gaussian, and the
tilted law is the one the certificate's recursion already carries forward for `E_b[W_t^2]`. Two
checks of it, neither through that recursion:

- Two steps from a fixed start, integrated in Maxima over `(u_0, x_1, u_1)` as Gaussians in exact
  rationals: 0.00612466700915665051..., which the library's recursion gives to 3e-16.
- Independent episodes where the model is the plant, `x' = a x + u + N(0, 0.25)`, the same logger
  and plan smoothed by `tau = 0.8`: over 10 000 to 20 000 simulated logs each, the mean error and
  the first order were -0.0140 +- 0.0057 and -0.0165 (`a = 0.6`, five steps, 50 episodes),
  -0.0005 +- 0.0029 and -0.0041 (200 episodes), -0.1005 +- 0.0195 and -0.1113 (`a = 0.9`, 100
  episodes), and -0.2326 +- 0.0395 and -0.2500 (ten steps, 200 episodes).

What the table says:

- At horizons of five and ten steps the bias is 0.03 to 0.12 of the half-width. A bias of a tenth
  of the half-width costs a nominal 95 % interval about half a point, so the coverage of 0.86 to
  0.87 at ten steps is the spread's shortfall, not the bias's: the influences of heavy weights
  read a spread too small. ADR 0083's attribution does not hold.
  *Note, 2026-10-13:* the 600 logs a cell above read high by chance. The same code reproduces
  them to the bit on their own streams, and on 2000 fresh logs a cell it covers 0.830, 0.841 and
  0.850 at 400, 800 and 1600 windows (each +- 0.008), as ADR 0083 measured (0.817 to 0.856). The
  attribution stands: an interval centred on each estimate with the estimate's own spread
  across logs covers 0.966 to 0.968, and the bias alone would leave 0.946 to 0.949. The misses
  are one-sided, 0.15 to 0.17 where the estimate reads low against 0.001 to 0.003 where it reads
  high: the estimate is skewed to the right and the spread it reads rises and falls with it.
- At twenty steps the bias is 0.30 to 0.40 of the half-width and 0.27 to 0.59 of the estimate's
  standard deviation, and the interval covers 0.63.
- The first order is 2 to 9 times smaller than the measured bias on one unit's windows, which are
  not independent and whose weights are heavy-tailed; on independent episodes it is the bias.
- Read on each log, it sorts the logs: those on which it passed 0.05 of the half-width were
  covered 0.40 to 0.74 of the time, the others 0.86 to 1.00. It passed on 83 % to 94 % of the logs
  at twenty steps and on 0 % to 35 % at five and ten.

## Decision

- **`UnitBootstrap` reports the draws it read.** `accepted` and `accepted_share` count them;
  `max_refused` is the bound; `reading` is `"complete"` where no draw was refused, `"conditional"`
  where some were, at most `max_refused` of them, and `"inconclusive"` past it. The
  `unit_bootstrap` event logs them all, at `WARNING` where any draw was refused.
- **`Prescription.evaluate(max_refused=0.05)`** sets the bound, a share in `[0, 1)`. 0.05 stays
  the default: up to it the intervals were at most 3 % narrower than the same panel's with no draw
  refused, and on the smaller panels the coverage fell by 0.008. At a tenth it fell by 0.017
  there, and from a tenth the narrowing doubled.
- **An inconclusive reading is a label and a warning, not a refusal.** The intervals are still
  returned; the caller reads the label.
- **`PlanEvaluation.normalisation_bias`** holds the first-order bias for `"pdis"`, read at the
  smoothing the certificate chose, with the episodes' own starts and count, on the model; `None`
  for the stationary methods.
- **A warning, `chc_event="pdis_bias"`, where it passes 0.05 of the interval's half-width.**
  `evaluate_plan` warns against its own interval; the unit bootstrap and the reading of one unit
  each warn once, against the interval they return.
- **The bias is not subtracted**, and no value moves.
- The recursion behind the certificate's moments became a generator that also yields each step's
  tilted law; the moments it returns are the same to the bit.

## Consequences

- Every value, interval and certificate is unchanged. `UnitBootstrap` takes four more fields, so
  a caller who built one positionally must name them; the library builds it in one place.
- An evaluation near the certificate's edge says so in its record, and one with more than 5 % of
  its draws refused logs that its intervals are inconclusive. The test at the edge in
  `tests/test_evaluation.py`, 40 draws, reads `inconclusive`.
- The warning fires on the long horizons where the bias swamps the interval, and on few episodes.
  It does not fire on most logs at ten steps, whose coverage of 0.83 to 0.85 is the spread's; that
  shortfall is still the docstring's to state.
- `tests/test_evaluation.py` holds the bias against Maxima's two-step integral to `1e-12`, against
  20 000 simulated logs (mean error -0.097 +- 0.020, first order -0.111), the warning on both sides
  of its bound for `evaluate_plan`, the bootstrap and one unit, and the bootstrap's reading on both
  sides of `max_refused`, the bound itself `conditional`. `tests/test_target_trial.py` holds
  `max_refused` reaching the record and refused outside `[0, 1)`.

## Alternatives considered

- **Refusing past the bound**, as the bootstrap's first statement of this had it. Rejected: past
  0.05 the intervals were 3 % to 13 % narrower and covered up to 0.067 less, which is a weaker
  reading and not none; a refusal would leave the caller nothing to weigh.
- **Unnormalised weights**, unbiased where the logger is known. Measured on the same logs, with
  the logger known: the mean errors were within 2.6 of their Monte Carlo errors of 0 and smaller
  than the normalised estimate's in eight cells of nine, and the root mean squared error was 1.30
  to 4.12 times the normalised estimate's.
- **Subtracting the bias.** Estimated from the logs' own weights, as the ratio estimator's
  correction reads it, it removed 0.13 to 0.65 of the bias, 0.14 and 0.24 at twenty steps, and
  raised the root mean squared error by 2 % to 13 %. Read off the model it is the first order
  alone, 2 to 9 times smaller than the bias on one unit, and it would move every value by an
  amount the model's own error makes uncertain.
- **Clipping the weights** (truncated importance sampling) trades the ratio's bias for a clipping
  bias of its own, which has to be measured loop by loop; it corrects nothing.
- **A doubly robust form over episodes** (Jiang and Li 2016), the model's value as a control
  variate. The module holds one only for the stationary `"dr"`; over episodes it is a new
  estimator, not built here.
