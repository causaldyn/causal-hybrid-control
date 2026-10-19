# ADR 0103 — A Koopman lift's residual reads its spread about the mean

**Status:** accepted, 2026-10-19. Amends the residual scale of ADR 0093.

## Context

ADR 0093 refuses a gain whose lift does not close. It reads each feature's closure residual
against the feature's root mean square. A state offset by a constant, for example temperature in
kelvin, inflates the root mean square. With an offset of 273.15, a lift refused at 0.71 read
0.0024 and was accepted. The accepted gain's closed loop was then 0.70–0.93 away from the plant's.

## Decision

- Each feature's residual is read against its spread about its mean.
- A feature whose spread is zero keeps its root mean square. Otherwise a constant nonzero feature,
  whose mean need not round to its value, reads a residual of 1e6 or more.
- The threshold 0.5 refuses a lift that predicts less than three quarters of some feature's
  variance on held-out transitions, a held-out `R^2` of 0.75, where it read the mean square.

## Consequences

- Lifts with nonzero-mean features read larger residuals. Some gains granted before are refused.
- A closed-loop stability check on the granted gain is not built. It is an option.

## Alternatives

- **Keep the root mean square.** ADR 0093 chose it, and rejected a centred ratio. Rejected now:
  under it a constant offset in the state's units decides the verdict.
