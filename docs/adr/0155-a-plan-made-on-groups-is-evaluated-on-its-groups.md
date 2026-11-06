# ADR 0155 — A plan made on groups is evaluated on its groups

**Status:** accepted, 2026-11-06. Experimental. Amends ADR 0049's evaluation and ADR 0113's
grouping, and adds `settings.clusters` to the record of ADR 0145.

## Context

`prescribe(clusters=)` (ADR 0113) fits and plans on the groups of a column: a group's row in a
period is the mean of its units' rows logged then, so a spillover inside a group is inside its
rows, and the channel is the effect of moving the group's units together. `Prescription.evaluate`
(ADR 0049) never read that column. It cut its windows from the panel's units, fitted the logger
unit by unit and resampled units, while the plant it linearised was the grouped fit's.

Its weights then read one unit's lever against its group-mates' logged ones: the effect of moving
that unit alone. Its model correction read the grouped channel on the units' windows. On units in
groups of four, `x' = 0.8 x + 0.5 u + 0.4 m + N(0, 0.1^2)` with `m` the mean lever of the unit's
group-mates and each lever drawn from its unit's state, `u = -0.3 x + N(0, 0.7^2)`, a plan made on
150 groups over 24 periods and evaluated on 60 later panels read 0.1441 (MC SE 0.0031). The cost
of each group's mean state under the plan is 0.2388 and the units' mean cost 0.3827. At
`model_error=0` the interval, half-width 0.038, covered neither on any panel. The default
`model_error=1` added the correction, 0.64, to each side, which hid it. No document, ADR or test
covered evaluating a grouped plan, and the grouping function is private, so a caller could not
group the panel either.

## Decision

- **`evaluate` groups the panel as `prescribe` did.** Where the prescription's `inputs.clusters`
  names a column, the panel is grouped by it before the windows are cut and before the logger
  check reads it. The windows, the logger, the plant and the bootstrap's draws are the groups',
  and `min_effective` counts the groups' windows.
- **The estimand is the grouped plan's own cost:** the cost of each group's mean state when every
  unit of the group follows the schedule, the objective the plan was chosen on. The units' mean
  cost under the plan exceeds it by half the target's weight times the variance of the units'
  states about their group's mean, summed over the steps: the mean of a square less the square of
  the mean. No grouping of the panel reads that term.
- **The record names the grouping.** `settings.clusters` holds the column, null where the plan was
  not grouped; `settings.panel_sha256` stays the digest of the panel as handed in, as
  `inputs.panel_sha256` is (ADR 0145).
- **The certificate's floor is unchanged.** A group's lever is the mean of its units' levers, whose
  weights compound; on the design above the default `min_effective=100` refuses the groups'
  windows on every panel and names the horizon it certifies. The caller may read them at a lower
  floor, as the next section measures.

## Consequences

On the design above, 200 later panels of 150 groups over 24 periods, `model_error=0`:

| `min_effective` | refused | value (MC SE) | value less its reported normalisation bias | covers the group-mean cost 0.23877 |
|---|---|---|---|---|
| 100 (default) | 10 of 10 checked: horizon 3 of 4 on 8, 2 on 2 | | | |
| 30 | 6 of 200 | 0.23404 (0.00186) | 0.23776 | 0.861 (0.025) over 194 |

At 30 the effective samples averaged 71 and the half-width 0.040; 85 of the 194 bootstraps read
inconclusive, 58 conditional. The value sits 2.5 standard errors under the cost: that is the
self-normalised weights' ratio bias (ADR 0095), which the evaluation reports, -0.0037 a panel on
average, and the value less it sits 0.5 standard errors under. The cost is the Gaussian moment
recursion of the groups' means, checked against a simulation of 500 000 groups, 0.238700
(SE 0.000383).

- A prescription made without `clusters` is evaluated exactly as before.
- `tests/misspecification/test_misspec_ope.py` reads the coverage on 40 of these panels;
  `tests/test_prescribe_interference.py` checks, deterministically, that the evaluation equals
  the one on the panel grouped by hand, resamples the groups, and refuses at the default floor.

## Alternatives

- **Evaluate the units' mean cost under the plan.** Deferred: it needs a plant of the units with
  their group-mates' levers as a second channel, which no fit holds; ADR 0113 deferred that
  channel for the plan as well.
- **Refuse to evaluate a grouped plan.** Rejected: the grouped evaluation reads the plan's own
  objective, with the same certificate as any other.
- **Make the grouping public and leave it to the caller.** Rejected: a caller who forgot would get
  the units' reading back silently, which is the defect.
- **Lower `min_effective` for a grouped plan.** Rejected: the floor is the caller's statement of
  how many effective windows a reading needs, and does not depend on what a window is.
