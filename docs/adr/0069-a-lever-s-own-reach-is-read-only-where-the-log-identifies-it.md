# ADR 0069 — A lever's own reach is read only where the log identifies it

**Status:** accepted, 2026-10-07.

## Context

`Prescription.reach()` reads, per lever, the fitted channel's entry on the target's row at the
plan's start, times the width of the lever's box, and `explain()` ranks the levers on it. Along a
direction of the channel the log never moved (ADR 0054), the fit holds the channel by a convention:
at zero where the drift takes up the direction's push, or by least squares beside the drift where
it cannot. Neither is an estimate of an effect. A lever whose entry leans on such a direction read
that convention as its reach:

- on a log whose second lever was always twice the first, `u1` read 2.05 and `u2` 1.02, a split of
  their joint push that any other split fits as well;
- a lever the log set from the state alone read 0.2456, though the log never moved its channel.

## Decision

- **A lever's reach is its own where its entry is orthogonal to every unmoved direction.** The
  entry of lever `i` on the target's row at the start `x0` is a linear functional of the channel's
  coefficients, `e_i (x) phi(x0)`. Where it is orthogonal to the fit's unmoved directions of that
  row, no fit the log cannot tell apart moves it, and `reach` reads it. Where it is not, `reach`
  reads `None`.
- **The test reads the log's own units.** Each coefficient is scaled by its column of the
  channel's design on the log's raw actions, as the fit splits the directions. The functional's
  part along the unmoved span is then compared with the square root of the precision times its
  size. `CausalDynamicsFit.unmoved` is in raw coefficient units. There a lever logged in 1e9 of its
  units shares the direction of `u2 = 2 u1` only 2e-9 of the way along it, under the square root of
  the precision, and a test read there would take `u1`'s reach for its own.
- **`explain()` ranks the levers that read a number**, and names the others on one line as levers
  the log set from the state or moved only together with others.
- **`reach()` refuses rather than guesses.** A prescription that does not record which levers the
  log identifies alone, one built other than by `prescribe`, raises `ValueError` where its fit left
  a direction unmoved. Where the fit left none, every reach is the lever's own.
- **`run_marketing_mix`'s myopic arm** ranks the channels by their reach. It raises `ValueError`
  naming a channel whose own return the log does not identify.

## Consequences

- `reach()`'s values are `float | None`. A caller that sorts, divides or compares them reads `None`
  first.
- On the first log above, both levers read `None`. On the second, which set `u1` from the state,
  `u1` reads `None` and `u2` 0.3984, as before. On a log that moved every lever, every reach reads
  as before.
- A test moves the fit along each unmoved direction on six logs. A reach that reads a number does
  not move; one that reads `None` moves with some direction.
- *Left:* the combination the log does identify, `b1 + 2 b2` on the first log, has no reach of its
  own in the API.

## Alternatives considered

- **nan instead of `None`.** It keeps the type, but a sort on nan keys returns some order without
  an error, and a ranking is what `reach` feeds.
- **Refusing the whole call.** It loses the ranking of the levers the log does identify.
- **The test in raw coefficient units.** It does not read the same in any units: see the 2e-9
  above.
- **Reporting the identified combination.** It needs a notion of a group of levers that the API
  does not have.
