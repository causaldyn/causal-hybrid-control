# ADR 0100 — The tube's rate error is the Euclidean norm over the states

**Status:** accepted, 2026-10-19. Amends the rate error of ADR 0087 and the drift's term of
ADR 0096.

## Context

The tube bounds how far the plant can drift from the plan, at a rate the field's error sets
(ADR 0080). ADR 0087 priced each lever's channel error at that lever's largest action and took
the root mean over the n·m entries of `(B_hat - B) d`. The tube's radius is a Euclidean norm over
the states, so a root mean over entries reads it √(n·m) too small. On a log of 2 states and
2 levers, the true error's norm was 1.88 times the old rate error. The drift's own error
(ADR 0096) entered as a root mean over the n states, √n too small.

## Decision

- The channel part of the rate error is `sqrt(n · Σ_j e_j² r_j²)`. Here `e_j` is lever j's
  channel error and `r_j` its largest action. This is the root mean square of the Euclidean norm
  over the n states of `(B_hat - B) d`.
- The drift's error enters multiplied by √n.
- `validation/tube_lever_units.mac` checks the identities; the ratio of new to old is n·m in
  square.

## Consequences

- Tubes widen. Certified horizons shorten where the model's error dominates the noise. On the
  pinned cases the rate errors double in square (52 → 104). The ends of noise-dominated tubes move
  by under 0.1 %.
- On 40 logs the true error's norm over the new rate error reads 0.93 (SD 0.046), where it read
  1.87 against the old one.

## Alternatives

- **Keep the root mean and document it.** Rejected: the tube is stated as a bound on a Euclidean
  distance, and the root mean does not bound it.
- **The operator norm of `B_hat - B`.** Rejected: the fit gives an error per entry, not a joint
  law, and a bound by the entries' sum is what that error supports.
