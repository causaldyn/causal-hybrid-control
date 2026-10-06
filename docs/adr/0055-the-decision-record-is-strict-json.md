# ADR 0055 — The decision record is strict JSON

**Status:** accepted, 2026-10-06.

## Context

`Prescription.to_json` promises plain JSON-safe values. Three of its numbers can be infinite or
nan in states a decision reaches:

- `regret_bound` is `inf` where the bound is refused (`regret_status` `refused`): the objective
  was not convex where its curvature was read, or a number the bound rests on was not finite.
- `gamma_star` is `inf` where every step's barrier holds however strong the confounding, and nan
  where some step's barrier is kept by no admissible action, not even at `gamma = 1`. A constraint
  the start already keeps with room to spare reads the first; one no action can reach, the second.
- A step of the lever selection carries its own regret bound, `inf` where it is refused.

`json.dumps` writes these as `Infinity` and `NaN`, which JSON does not have (RFC 8259): a strict
parser rejects the whole record. A review of 0.13.0 found that `json.dumps(..., allow_nan=False)`
raises on a refused regret bound and on either ceiling. The first ceiling is the common one: on
the tests' world of a wait that stays near 0, `Constraint("wait", hi=0.5)` reads `inf`, so the
ordinary record of a constrained plan was not JSON.

## Decision

- **No number in the record is infinite or nan.** One that is not finite is written as `null`.
- **Where `null` has more than one reading, a status beside it names it:**
  - `regret_bound` is `null` where no plan was solved, `regret_status` `null` too, and where the
    bound was refused, `regret_status` `refused`;
  - `gamma_star_status` reads `gamma_star`: `finite`; `every_level`, where it is `inf`;
    `no_level`, where it is nan; `null` where no bound was audited, as `gamma_star` is;
  - each selection step carries its `regret_status`, so a step's `null` bound reads `refused`.
- Any other number that is not finite, such as a cost whose predicted path overflowed, is `null`
  as well.
- **`schema_version` is 2**: `regret_bound` and `gamma_star` read `null` in states that wrote a
  number before.
- The Python objects keep `inf` and nan. `DecisionCertificate.gamma_star_status` is a property read
  off `gamma_star`, and `SelectionStep.regret_status` is the step's bound's status.
- A test writes every state the review listed through `json.dumps(..., allow_nan=False)`: a tube
  not evaluated, an uncertified one, a diagnostic and a refused regret bound, both ceilings, no
  plan, and a plan the log does not determine.

## Consequences

- A reader that took `Infinity` from Python's parser reads `null` and the status beside it:
  `regret_status == "refused"` for an infinite regret bound, `gamma_star_status` for the ceiling.
- An output added to the record that can be infinite or nan comes with its status in the same
  change.
- *Left*: the `logging` records keep Python floats, `inf` among them. A JSON log formatter meets
  the same values.

## Alternatives considered

- **`Infinity` and `NaN`, Python's default.** Rejected: not JSON.
- **Strings, `"Infinity"` and `"NaN"`, as protobuf's JSON mapping writes a double.** It keeps the
  value, but the field is then a number or a string, and every reader has to parse both. A status
  is one more field, read the same way as `regret_status`.
- **Refusing to write the record.** Rejected: a refused regret bound is a result, and the record
  is where it is read.
