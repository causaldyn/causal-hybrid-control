# ADR 0070 — A prescription runs as a policy

**Status:** accepted, 2026-10-07.

## Context

`Prescription.schedule` is open loop: a level for each lever at each step. A lever the log set from
the state alone follows the log's rule of the state in the plan's field and in its price (ADR
0054). The schedule's column for it is that rule read along the predicted path, and
`InterventionSchedule.rules` says to set it from the state as it comes, not to those numbers. No
object gave the caller the rule to set it by. Off the predicted path, which is where a plan runs,
the lever's level was not available, and `evaluate` refuses such a plan, since its schedule is
open loop.

## Decision

- **`Prescription.policy()` returns a `PrescribedPolicy`.** Its `actions_at(step, state)` gives
  each lever's level at the step in the state given:
  - a lever the plan moves takes the schedule's level at the step;
  - a lever the log set from the state takes the log's rule at the state, clipped to its box, as
    the plan's field and its price read it;
  - a lever the plan holds to the log, or one the selection left out, keeps the level the schedule
    holds it at.
- **One reader of the rule.** The schedule's ruled columns are read through the same function, a
  step at a time, so along the predicted path the policy's actions are the schedule's to the bit.
- **A state is the states' values in the order of `states`, or a mapping that names each.** A
  name the mapping holds besides is not read. A step that is not an integer from 0 to the horizon
  less one, a state value that is not finite and a state that leaves a name out are refused.
- **The policy is strict JSON** (ADR 0055), with a `schema_version` of its own, 1:
  - a ruled lever's column of the schedule is `null`, as its rule sets it, and `rule` names those
    levers and holds the rule's centre, shift, factor, coefficients, degree and box;
  - an infinite bound of the rule is `null`: `lo`'s reads -inf, `hi`'s inf.
- **`from_json` reads it back to the bit, or refuses it.** It refuses names that are not distinct,
  a schedule that is not a finite number for each lever at each step (a ruled lever's `null`
  excepted), a rule that does not fit the levers and the states, and a box upside down.
- **A rule runs in the precision it was written in.** `precision` names the floats the rule reads
  the state in. `from_json` refuses a rule where JAX runs in the other precision: a rule read in
  float32 runs differently in float64, and the other way round.
- **The name is `PrescribedPolicy`**, apart from the evaluation's policies (`AffinePolicy`,
  `GaussianPolicy`), which are distributions over actions. *Experimental.*

## Consequences

- A plan with a ruled lever can run in closed loop: the caller passes the state each step reaches.
- The schedule's ruled columns are read a step at a time, through the policy's reader, where they
  were read over the whole path at once. The two reads differ in the last bits: in single
  precision a ruled level of the tests' log moved from -0.30000025 to -0.30000028; in double
  precision the tests' plans read the same bits.
- *Left:* `evaluate` still refuses a plan with a ruled lever. The decision record
  (`Prescription.to_json`) names the ruled levers but does not carry the policy: a record kept to
  run later needs the policy's JSON beside it.

## Alternatives considered

- **The rule's coefficients on the schedule.** The caller would rebuild the standardisation, the
  polynomial and the clip: two readers of one rule, which can drift apart.
- **A NumPy copy of the rule for the policy.** It runs without JAX, but its last bits differ from
  the rule the plan's field reads, so the schedule and the policy would disagree along the path.
- **Reading a rule in the other precision.** It runs, but not as written. A refusal names the
  setting to change.
- **The policy inside the decision record now.** It changes the record's schema. It is its own
  decision.
