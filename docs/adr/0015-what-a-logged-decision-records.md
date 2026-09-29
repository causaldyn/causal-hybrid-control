# ADR 0015 — What a logged decision records

**Status:** proposed, 2026-09-28. Amended 2026-09-29 by ADR 0021, before any release: `dither`
is the draw, before any clip.

## Context

The deployment gate (ADR 0010) keeps its guarantee only on propensities logged when each action was
drawn. In the lab, a behaviour law refitted to the logs took its type-I error from 0.013 to 1.000
(AG4). The gate cannot tell a logged propensity from a refitted one; it can only require the field.
The channel monitor the lab worked out next (AG5) needs the Gaussian dither added to each action,
and needs it as the draw: a clipped or mis-scaled dither breaks its detector.

Both are fields of a record that outlives the process that wrote it. Once a caller stores logs in a
shape, the shape cannot be taken back, so this is a type-1 door.

## Decision

- **`chc.gate.DecisionLog`, one entry per decision**:
  - `action`, as applied, one row per decision for a vector action;
  - `propensity`, computed by the policy that drew the action, when it drew it;
  - `saturated`, a boolean: the action was clipped, so what was applied is not what was drawn;
  - `dither`, optional: the Gaussian perturbation drawn for the policy's action, as drawn. On a
    clipped decision it is the draw, not what the clip left of it (ADR 0021; this said "as
    applied" until then).
- **Stored, a decision is a record** with `decision_log_version` beside those four fields.
  `DecisionLog.to_records()` writes them, and `DecisionLog.from_records(rows)` reads them. The
  version is 1.
- **The reader refuses, and never fills in**:
  - a record with no version, which is anything written before this ADR;
  - a record of another version, or whose version is not an integer;
  - a record without `action`, `propensity` or `saturated`, whatever its version;
  - a `saturated` that is not a boolean, where reading a `0` or a `"false"` would be a guess;
  - a `dither` on some records and not others.
- **Keys the log does not define are the caller's**, such as a timestamp, the reward or the zone,
  and are not read. One row can carry the decision and the caller's own columns.
- **The key is `decision_log_version`, not `version`**, which a caller's rows already use for a
  model or a policy.
- **`ZoneBatch.from_log(log, reward=, candidate=, baseline=)`** takes the logged propensities from
  the log. `ZoneBatch(logged=...)` still takes an array: a log records the field, and cannot make it
  honest.
- **`DecisionLog.dither_draws()`** is how a reader that needs the Gaussian draw gets the dither. It
  refuses a log with no dither, and a log with a clipped decision.

## What a new version requires of stored logs

- **A new version is written when a field changes meaning, or when a reader needs a field that
  version 1 does not have.** A field that no reader needs, and whose absence means what a version-1
  record meant, is added without one.
- **A reader of version 2 reads version 1 only where the missing field has one true value for every
  version-1 record**, and the ADR that makes version 2 writes that value down. Otherwise version-1
  logs are refused, and the decisions have to be logged again under the new version.
- **A version number is never reused.** Version 1 means the fields above, with the meanings above,
  for as long as the library reads it.

## Consequences

- A gate fed from stored logs gets each decision's recorded propensity, by name, or a refusal that
  names what is missing and on which record.
- The stored form survives JSON bit for bit: over 60 generated logs, with actions from 1e-300 to
  1e300, vector actions and empty logs, every field read back equal. An empty log cannot say
  whether it would have carried a dither, and reads back without one.
- A batch read from stored records equals the batch built from the arrays, entry for entry, and the
  gate returns the same verdicts on both.
- A caller who logs a clipped action has to say so. That is the price of a dither reader that
  cannot be fed a clipped draw.
- A caller whose logs predate this ADR cannot feed them to the gate through `from_records`. They can
  still build a `DecisionLog` from arrays, and then vouch for the fields themselves.

## Alternatives considered

- **One version per log, in a columnar form.** Rejected: a log is written one decision at a time,
  into a stream or a table, and a version on each record survives a table that two writers filled.
- **Filling a missing field with a default.** Rejected: a default propensity is the fitted
  propensity that AG4 measured at a type-I error of 1.000, and a default `saturated = false` makes
  a clipped dither look like a draw to the monitor.
- **Typing `ZoneBatch.logged` as a `DecisionLog`.** Rejected: it enforces nothing that the array
  does not, since a `DecisionLog` can be built from any array, and it would make a caller of the
  gate alone record actions and flags the gate never reads.
