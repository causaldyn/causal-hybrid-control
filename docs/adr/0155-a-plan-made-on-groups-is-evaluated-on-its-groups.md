# ADR 0155 — A plan made on groups is evaluated on its groups

**Status:** accepted, 2026-11-06. Experimental. Amends ADR 0049's evaluation and ADR 0113's
grouping, and adds `settings.clusters` to the record of ADR 0145.
Amended 2026-11-07: **no bias-corrected value.** `value` stays the self-normalised estimate, with
`normalisation_bias`, its first-order bias under the model (ADR 0095), reported beside it. The
value less that bias was measured against each design's true cost, and was to become a field only
if it read a smaller mean error on all four designs below and an RMSE higher by no more than its
Monte Carlo error on any. One prescription per design, fitted once in float64 and evaluated on 200
later panels, each from its own seed; the truth is the Gaussian moment recursion of the true plant
from the law the windows start at, each calendar time zero alike:

| design | evaluated | mean reported bias | value: mean error (MC SE) | less the bias: mean error | RMSE: value, less the bias (MC SE) |
|---|---|---|---|---|---|
| the groups below, `min_effective=30`, `model_error=0` | 194 | −0.00372 | −0.00473 (0.00186) | −0.00100 | 0.02632, 0.02597 (0.0016) |
| ADR 0054's held lever: 150 units over 20 periods, the second lever at 0.5 | 200 | +0.00022 | −0.00313 (0.00192) | −0.00335 | 0.02723, 0.02725 (0.0013) |
| the quickstart's plant as `tests/test_lifecycle_loop.py` logs it: 400 units over 12 periods, the incentive drawn afresh, a horizon of 3 | 200 | +0.00007 | −0.00151 (0.00195) | −0.00157 | 0.02756, 0.02756 (0.0012) |
| 150 units, not grouped, over 24 periods: `x' = 0.8 x + 0.5 u + N(0, 0.1^2)`, `u = −0.3 x + N(0, 0.7^2)`, a horizon of 4 | 200 | +0.00029 | −0.00032 (0.00239) | −0.00061 | 0.03365, 0.03364 (0.0018) |

The last three read the defaults (`min_effective=100`, `model_error=1`); the quickstart's own logs
cannot be evaluated, since its incentive chased a demand outside the plan's state. On the groups,
at 71 effective windows, the correction removes four fifths of the mean error and lowers the RMSE
by 0.00035 (paired MC SE 0.00028). On the other three, at 112 to 126 effective windows, the
reported bias is 0.03 to 0.12 of the mean error's Monte Carlo error and of the other sign from the
mean error read, so the corrected value reads the larger mean error on each, and the RMSE moves by
less than 0.1 %. The rule fails, and nothing is added. The warning ADR 0095 logs where the bias
passes 0.05 of the interval's half-width fired on 178 of the 194 grouped panels and on none of the
other 600: it marks the evaluations where the correction moved the reading.

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
