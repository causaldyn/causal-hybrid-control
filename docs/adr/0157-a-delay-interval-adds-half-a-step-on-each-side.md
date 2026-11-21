# ADR 0157 — A delay interval adds half a step on each side

**Status:** accepted, 2026-11-21. Evolving tier (`chc.irf`). Changes what
`DelayEstimate.lo` and `DelayEstimate.hi` mean: an interval of the delay, where they were an
interval of the peak's location.

## Context

`delay_estimate` locates the largest response of a local-projection impulse response and gives
`lo` and `hi` from a moving-block percentile bootstrap: every resample is refitted and its peak
located again, at a whole step or, with `refine=True`, at the vertex of the parabola through the
peak and its neighbours. `chc.delay.robust_delay_design` reads those two numbers as the ends of an
interval that holds the true delay, and designs a controller for every delay in it.

A percentile bootstrap carries a statistic's spread about its own limit, not the distance from
that limit to the delay. A transport lag of `c` steps, seen through an input held over each step
and an outcome read as its change over the step, has the samples `1 - |h - c|`: weight `1 - f` at
step `m` and `f` at step `m + 1`, for `c = m + f`. Their largest sample is at `c` rounded, up to
half a step from `c`, and the parabola's vertex is `f / (4 - 6 f)` past `m`, up to a sixth of a
step from `c` (`validation/delay_interval_resolution.mac`). On a sharp peak every resample lands
on the same step, so the interval has no width at all.

`chc.research.benchmark.DelayOscillationTask` logs exactly that plant: a delay of 1.0 time units
at a sampling step of 0.3, so `c = 10/3`. Over 400 seeds each, at the shipped settings
(horizon 12, 400 resamples, level 0.95):

| delay | interval | held the delay | where it sat |
|---|---|---|---|
| 1.0 (3.33 steps) | whole step | 0 of 400 | `[0.9, 0.9]`, below |
| 1.0 | refined | 0 of 400 | median `[0.936, 0.959]`, below |
| 2.0 (6.67 steps) | whole step | 0 of 400 | `[2.1, 2.1]`, above |
| 2.0 | refined | 0 of 400 | median `[2.041, 2.065]`, above |

Below the delay is the side on which a delay-robust design destabilises the plant
(`chc.delay.delay_ball`). The docstring has stated the miss ("0 of 10") since 1.1.1; neither
interval reached its level at either delay.

## Decision

- **`lo` and `hi` are the whole steps at the bootstrap's two percentiles, widened by half a step
  on each side and cut at zero.** The resampled peaks are located at whole steps; the lower end is
  the `(1 - level)/2` quantile taken as the draw below it, and the upper end the draw above the
  `(1 + level)/2` quantile, so each end is a step some resample peaked at and never a value
  interpolated between two steps. Half a step is the farthest the delay sits from the step the
  hat peaks at, a distance the bootstrap cannot show. The cut at zero is causality: no effect
  arrives before its cause.
- **The premise is stated:** the delay lies within half a step of the step the response peaks at.
  It holds for a delay of whole steps and for the hat above. An outcome read at instants under a
  held input peaks up to a whole step late, and the half step does not cover that.
- **The interval reads the whole step whether `refine` is on or not.** `lag` and `delay` keep
  their meaning and values: `refine` still moves them to the vertex, which lies within half a
  step of the whole-step peak.
- **`censored` keeps its meaning.** A peak on the last horizon still bounds nothing above; a peak
  at step 0 now gives `[0, dt/2]`.

## Consequences

The same 400 seeds, after the change:

| delay | `refine` | held the delay | interval |
|---|---|---|---|
| 1.0 | off, on | 400 of 400 each | `[0.75, 1.05]` |
| 2.0 | off, on | 400 of 400 each | `[1.95, 2.25]` |

Over 200 seeds at each of 16 delays, 15 with `c` from 3.0 to 4.0 steps and one at 6.67, the
interval held the delay on 200 of 200 at every one; on test_irf's one-sided plant,
`x_t = phi x_{t-1} + u_{t-5} + e`, whose largest effect is at step 5, on 200 of 200 at
`phi = 0.5` and at `phi = 0.9`. It is one step wide where the peak is sharp, and two at
`c = 3.5`, where the two steps tie and the resamples split between them. The level is held
conservatively: the half step is the worst case, and the interval held the delay on every seed
at every `c` measured.

- A sharply identified lag no longer returns `lo == hi`. On a plant whose delay is a whole number
  of steps, the interval holds that number and no other.
- `DelayOscillationTask`'s delay-aware arm designs for the minimax over `[0.75, 1.05]`, `0.769` of
  the delay: the design leans to the short end, where its loss rises linearly rather than as a
  square root. The interval holds the delay, so `stabilises_interval` now speaks of the true
  plant. The arm's regret moves from 0.007 to 0.11, against the oracle's cost of 0.57; the blind
  arm still diverges, at a regret of 18 953.
- `tests/test_irf.py` checks the interval at both delays and the cut at zero, and
  `tests/test_formal_estimation.py` checks the hat's statements on a simulated held pulse.

## Alternatives

- **The refined peak's percentiles, widened by a sixth of a step.** A sixth is the vertex's
  largest distance from the delay on the hat, at `f = 1/3`. Rejected: the sixth holds on the hat
  alone. On a one-sided response, `phi^h` after the delay, the vertex is `phi / (2 (2 - phi))`
  late, more than a sixth for `phi > 1/2`. Over 200 seeds, at a median width
  of 0.41 steps against 1: 187 on the hat at `c = 10/3` and 192 at `20/3`, 199 on the one-sided
  plant at `phi = 0.5`, where the distance is exactly a sixth, and 0 at `phi = 0.9`.
- **The centroid of the three samples around the peak.** It recovers `c` exactly on the hat
  (`validation/delay_interval_resolution.mac`, step 4). Rejected: its percentile interval held the
  delay on 151 to 170 of 200 seeds at the 13 off-grid delays above, and on a one-sided response it
  is `phi / (1 + phi)` late, later than the vertex (step 5).
- **Keep the interval of the peak and document it.** That was 1.1.1's answer. Rejected: the one
  consumer, `robust_delay_design`, reads the two ends as an interval that holds the delay.
- **A new field for the delay's interval beside the peak's.** Rejected: two intervals on one
  estimate invite reading the narrower one, which is the one that does not hold the delay.
