# ADR 0113 — `prescribe` checks for spillover and fits on groups

**Status:** accepted, 2026-10-19.

## Context

`prescribe` fits each unit's transitions on its own levers. Where units spill into each other,
drivers or customers moving between zones, a unit's state moves with its neighbours' levers too.
The fit then reads the effect of moving one unit with its neighbours held. A plan that moves every
unit gets that effect plus the spillover.

On a panel of 50 clusters of four zones over 12 periods, a zone's incentive pushes its own supply
by 0.8 and each cluster-mate's by `-0.4/3`. The incentive chases demand, part of which the cluster
shares. Over 40 seeds the channel read 0.827 on the fit's RK4 field (0.825 for 0.8), where a plan
that moves every zone gets 0.4125. Its 95 % interval covered that on no seed. Nothing in the
certificate said so.

## Decision

- `prescribe(neighbours=)` takes a mapping from a unit to its neighbours, or a column that groups
  the units. The fit's moment residual is regressed on the neighbours' mean levers, both taken
  about their means within each period. Each slope's error is the CR1 sandwich over the groups the
  fit sums its errors in, against `t(G - 1)`. The check's p-value is Bonferroni's over the slopes
  (`chc.dynamics_id.interference_check`, *experimental*).
- A rejection at 5 % is a `WARNING` (`chc_event="interference"`) and a report line naming the
  strongest lever and state and the remedy. `Prescription.interference` holds the check. It
  changes no number, as the logger check does not.
- `prescribe(clusters=)` fits and plans on the groups of a column. A group's row in a period is
  the mean of its units' rows logged then. A spillover inside a group is then inside its rows, and
  the channel is the effect of moving the group's units together. `RunProvenance.clusters` records
  the column; the panel's provenance is the grouped panel's.
- The panel's declared `cluster` is kept where it holds one value over each group's rows.

## Consequences

Over 40 seeds of the panel above (float64):

| spillover | fit | channel | bias against the plan's effect | 95 % coverage | check rejects |
|---|---|---|---|---|---|
| none | zones | 0.8259 | +0.0009 | 1.00 | 0.025 |
| none | groups | 0.8284 | +0.0034 | 1.00 | |
| −0.4 | zones | 0.8268 | +0.4143 | 0.00 | 1.00 |
| −0.4 | groups | 0.4160 | +0.0035 | 1.00 | |
| −0.4, cluster campaigns | zones | 0.6199 | +0.2074 | 0.00 | 1.00 |
| −0.4, cluster campaigns | groups | 0.4145 | +0.0019 | 1.00 | |

With campaigns that raise every zone of a cluster together (SD 0.5), a zone's incentive moves
with its cluster-mates' beyond what the demand predicts, and the zone-by-zone channel absorbs part
of the spillover: it is then neither the direct effect nor the plan's. The zone-by-zone rows are
the same on the code before this change.

- On residuals with a shock every unit shares in a period, chased by the levers, the check rejects
  at 5 % on 0.044 of 2000 draws with 50 clusters and on 0.056 with 10.
- A shock shared by neighbours alone, which their levers chase, reads as spillover: it rejected on
  every draw. Such a shock confounds the unit's own lever as well, unless the adjustment set holds
  it.
- A grouped fit has a row per group and period, so its error reads the groups' spread. Its
  standard error was 0.0147 and 0.0067 against spreads of 0.0101 and 0.0045 over the seeds.
- A spillover between groups is not removed by grouping. `neighbours` keyed by group checks it.

## Alternatives

- **Refuse the plan on a rejection.** Rejected: a 5 % test would refuse one valid panel in twenty,
  and the grouped fit is the remedy, which the report names.
- **Fit the spillover as a second channel** (the neighbours' mean lever as an action). Deferred: the
  plan would then need a lever per unit and its neighbours, which `prescribe`'s pooled schedule does
  not have. `chc.network_causal` estimates both effects where that is the question.
- **Run the check by default on the declared `cluster`.** Rejected: that column is declared for the
  errors' grouping, which need not be the spillover's, and a new warning on every call moves
  existing reports.
