# ADR 0101 — The barrier audit reads the path and the levers' box

**Status:** accepted, 2026-10-19. Amends what a certified step means in `certify_safety` and a
plan's `safety`, and the set `gamma_star` is read on (ADR 0085, ADR 0091).

## Context

The audit certified a step where the continuous barrier condition held at that step's state. Three
cases escaped it:

- **Coarse steps.** At `dt = 1` and `alpha = 1`, a plan certified at all 12 steps ended at
  `h = -0.065`. The condition held at each sample while the path left the set between samples.
- **Asymmetric boxes.** `gamma_star` was read on a ball whose radius is each lever's largest
  magnitude. On a box `[0, 5]` where only a negative action helps, the ball admitted that action,
  and steps reported finite ceilings that no admissible action reaches.
- **Lever selection.** `prescribe` audits a selected plan as if it had been planned on every
  lever. The audit took the box of the narrowed solve, which pins the levers left out at zero.

## Decision

- A step is certified only where the condition holds, `h` at the step's state is at least 0, and
  `h` at the next predicted state is at least 0.
- `gamma_star` is read on the levers' box. Where a step's best action of the ball leaves the box,
  it is drawn back along its ray to the box's edge. This is exact for one lever and a lower bound
  for several. A symmetric box gives the same numbers as before, bit for bit.
- `BarrierConstraint(u_box=)` carries the box the audit reads. `prescribe` passes the box of every
  lever, the same box as `u_max`.
- `certify_safety` refuses a model that is not affine in the action where the condition is read.
  On a Hill curve and on `u + u^3`, the old audit certified 10 of 10 steps, and the field failed
  all 10 at the planned action.
- A non-finite `x0` is refused. The tube certifies no step past a non-finite predicted state.

## Consequences

- Certified prefixes shrink where the path crosses the boundary between samples. One pinned test
  went from 14 steps to 12.
- An asymmetric box can turn a finite `gamma_star` into nan, "no level". That nan is the truth:
  no admissible action keeps the barrier.
- The held solve still holds only the continuous condition. At coarse steps it can plan a path
  that this audit does not certify. Holding `h` at the next state in the solve is open.

## Alternatives

- **Check `h` only at the horizon's end.** Rejected: a path can leave the set and come back.
- **An exact box maximum over several levers.** Deferred: it is a small LP per step. The ray
  draw-back is a lower bound, and a lower bound is enough for a certificate.
