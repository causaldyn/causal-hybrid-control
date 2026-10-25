# ADR 0104 — The tube reads the field's error where the plan goes, at its studentised quantile

**Status:** accepted, 2026-10-19. Amends the field's part of `prescribe`'s tube (ADR 0096, ADR
0100) and the context of ADR 0099. Amended by ADR 0133, 2026-10-25: given clusters or periods, the
drift's own error and the push's mean are read from them, as the channel's error is, the
alternative rejected below; the degrees of freedom stay the channel's.

## Context

Each radius of `prescribe`'s tube states that the plant stays inside it with probability 0.95. The
radius has two parts: the plant's noise, and the fitted field's error, which is the channel's
error and the drift's error together. On short logs the field's part fell short, for two reasons.

1. **It was one standard error, added to the noise's 0.95 radius.** That is a 68 % reading of
   the error. The error is Student's `t`, because the channel's error is read from as few as 3 to 6
   cosines or clusters. Where the field's error is four times the noise's, the sum holds 0.758 at
   `nu = 3` and 0.846 at `nu = 52` (`validation/tube_level.mac`).
2. **It was a root mean over the log's rows.** A plan heads to its target, away from the log's
   states, and there the fitted field's error grows. On AR(1) logs of 60 transitions, the drift's
   error at a plan's late steps is 1.45 to 1.63 times its root mean over the rows.

On AR(1) logs of 30 transitions the plant stayed inside the tube 0.961 of the time at step 1 and
0.904 at step 8.

## Decision

- The fit keeps the covariances that its drift and channel errors are read from
  (`_drift_blocks`, `_channel_blocks`).
- `prescribe` reads them at the state each planned step starts from. It reads the drivers at
  their forecast's mean over the step, as the `rk4` fit reads a transition. Each part is the scale
  of the Euclidean norm over the states (ADR 0100). The channel stays priced at each lever's
  farthest action in its box (ADR 0091).
- The tube places that error along the noise's widest direction, where it widens the radius
  most, as a Gaussian of standard deviation `t_nu(0.975) / z(0.975)` times the error. `nu` is the
  channel's degrees of freedom. The radius is the exact 0.95 radius of the noise's covariance with
  that variance added to its largest eigenvalue. On one state it is `sqrt(r^2 + t^2 f^2)`.
  `validation/tube_level.mac` checks the level: 0.9496 to 0.9523 at `nu` 3 to 52 on one state,
  0.9500 to 0.9514 on two states, and more where the error lies along another direction.
- ADR 0100's Euclidean norm puts the full mean square of the error's norm along one direction.
  On more than one state that is a bound, not an equality.

## Consequences

Coverage at the plan's first and last step, as `scripts/bench_short_log_tube.py` reads it (200 or
150 logs, 2000 runs a log), before and after:

| plant | transitions | before | after |
|---|---|---|---|
| AR(1) | 30 | 0.961 – 0.904 | 0.968 – 0.968 |
| AR(1) | 60 | 0.961 – 0.950 | 0.947 – 0.976 |
| AR(1) | 120 | 0.963 – 0.969 | 0.949 – 0.973 |
| two levers | 60 | 0.967 – 0.958 | 0.954 – 0.986 |
| bilinear | 60 | 0.985 – 0.939 | 0.999 – 0.989 |
| bilinear | 120 | 0.991 – 0.992 | 0.984 – 0.993 |

- The late steps now read up to 3.6 points over 0.95 on one state and up to 5 on the bilinear
  plant. That is the cost of pricing the channel at the box's farthest action, which also covers
  the fitted drift's bias on short logs.
- On the quickstart's plant (4 logs of 200 units), the well-specified case read 0.950 to 0.958 a
  step before and 0.943 to 0.952 after. Over 16 logs it reads 0.948 to 0.951.
- The tube is narrower where the field's error is small beside the noise, because the sum in
  quadrature is less than the plain sum. It fits worse where the field's error is not an honest
  standard error:
  - Under a sine drift fitted at degree 1, step 15 read 0.895 before and 0.876 after.
  - Under an AR(0.8) confounder on one unit of 120 periods, the minimum read 0.914 before and
    0.889 after.
  - On the bilinear plant under the same confounder, step 1 read 0.927 before and 0.881 after.
  The sine drift is now warned of on every log (ADR 0105). The other two are open. Their
  confounder is logged and adjusted, so the channel's residual does not persist, and nothing
  warns.
- On the pendulum demo (20 logs, `bound` mode) nothing moves. Step 1 reads 0.947 and every
  later step 1.0, before and after, and no step is trustworthy.

## Alternatives

- **Keep one standard error, added to the noise's radius.** Rejected: it does not state a level.
  It held 0.758 at `nu = 3` where the error dominates.
- **Welch-Satterthwaite degrees of freedom for the two parts.** Rejected: with the drift's part at
  its residual degrees of freedom, it read the tube shorter on short logs, and the same on long
  ones.
- **Place the error along the direction the fit does not give.** That direction is unknown. The
  widest direction gives the largest radius, so it is the bound.
- **A drift error robust to clusters and to persistence**, read like the channel's error. Tried
  on the AR(0.8) cells: the minimum moved from 0.890 to 0.889 and stayed at 0.937. Not taken.
