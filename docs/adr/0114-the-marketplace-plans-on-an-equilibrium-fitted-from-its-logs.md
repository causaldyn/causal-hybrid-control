# ADR 0114 — The marketplace plans on an equilibrium fitted from its logs

**Status:** accepted, 2026-10-19. Amends what `pessimistic_equilibrium_allocation` plans through.

## Context

`pessimistic_equilibrium_allocation` planned through the simulator's own equilibrium: its base
demand, its attractiveness and its solver. No log supplies these. The logs reached the plan only
through the pessimism tilt, at most `radius` standard errors per zone. On the default market a
response of zero planned as close to the oracle as the de-confounded one (11.7469 and 11.8274
against 11.8719 in float32). The comparison measured knowing the equilibrium, not learning it.

The tilt had a second defect. It was `max(m - r·se, 0) - m`, so a zone whose estimate `m` was
negative earned a bonus of `|m|` per unit of incentive (`+0.0054` on zone 8 in float64).

## Decision

- `fit_equilibrium(logs)` fits the congestion equilibrium by least squares on each block's
  completions: one attractiveness per unit of demand per zone, and the drivers' sensitivity, the
  crowding and the drivers' total, `n_zones + 3` parameters. The solve is L-BFGS on gradients
  implicit in the equilibrium, the positive parameters on their logarithms. The damping is
  `4/(4 + beta·congestion)`, at which the map contracts for every game, since a fit passes through
  values past the fixed damping's ceiling.
- The planning demand is each zone's mean logged demand.
- `pessimistic_equilibrium_allocation(equilibrium=)` plans through the fitted equilibrium. Without
  it the planner keeps the simulator's, with a `FutureWarning`; a future release will require it.
- The tilt is the worst response within the radius, floored at zero, less the estimate projected
  onto the same floor: `-min(max(m, 0), r·se)`. `validation/marketplace_tilt.mac` checks it.

## Consequences

Five markets, logs of 50, 100 and 400 blocks, completions noise of SD 0.02. The share of the best
plan's lift over no incentive (the best of the oracle's and every plan's, since the oracle's ascent
is local):

| planner | 400 blocks | 50 blocks |
|---|---|---|
| fitted equilibrium, de-confounded response | 0.89–1.00 | 0.59–0.96 |
| simulator's equilibrium, de-confounded response | 0.87–1.00 | 0.79–1.00 |
| fitted equilibrium, incentives shuffled across blocks | ≤ 0 | ≤ 0 |

- On 400 blocks the fit recovers the sensitivity 2.5 within 0.02, the crowding 2 within 0.01 and
  the total 12 within 0.05, with a residual at the noise's SD.
- A fit on shuffled incentives reads a sensitivity below 0.05, and its plan wins nothing, or
  loses completions. The plan now owes its value to the logs.
- The response still enters only through the tilt. The confounded and the de-confounded responses
  plan alike (0.86–0.97 and 0.89–1.00 on 400 blocks). De-confounding enters through the fit, whose
  model holds the logged demand.
- Near the optimum the fitted completions are flat, and the tilt moves the plan along the flat,
  which is not quite the true one. On 400 blocks in float64 the de-confounded response's plan ends
  0.11 completions short of the zero response's on one market and 0.42 ahead on another.
- The parameters' sampling error is not carried into the plan. The model class is the simulator's:
  a market whose drivers follow another law is fitted by the nearest logit equilibrium.

## Alternatives

- **Make `equilibrium=` required now.** Rejected for this release: it breaks every caller at once.
  The `FutureWarning` gives a release to move.
- **A nonparametric response surface over all zones' incentives.** Rejected: 144 cross-effects on
  12 zones, against `n_zones + 3` parameters that carry the conservation of drivers exactly.
- **The tilt against the raw estimate, with the floor removed.** Rejected: a negative response
  would then discount a zone below its estimate, and the floor encodes that an incentive does not
  remove completions in its own zone.
