# ADR 0129 — The Euler fit holds the adjustment set in its drift, and becomes `prescribe`'s default

**Status:** accepted, 2026-10-24. Amends ADR 0118 (`"rk4"` stays the default) and ADR 0125 (the
default fit reads the relative radius): from 0.22.0 `prescribe` fits and plans by `"euler"`.
Amended by ADR 0132 (2026-10-25): a `gamma` at or past `(3 + sqrt 5)/2` no longer raises under
the absolute radius.

## Context

`fit_causal_residual` fits by `"euler"` by default and `prescribe` by `"rk4"` (ADR 0118). Two
things were measured on the misspecification suite's plant, the quickstart's, in seven worlds of
six logs each, the plan scored by its task cost on the true mean plant against an SLSQP oracle with
the same box and bound, and the held barrier at `gamma = 2` with `supply <= 0.8` and `wait <= 0.5`.

**The default certificate reads a margin, not a level of confounding.** `ChannelSensitivity` gives
an `"rk4"` fit no absolute radius: its fixed point carries the drift's confounding into the channel
(ADR 0111). So `radius="auto"` reads the relative radius on every default call (ADR 0125). Where an
unrecorded confounder moves the channel the log reads from 0.8 to about 0.41, the `"rk4"` relative
radius certified all 15 steps on six logs of six, and on each the plant left the set: its least
margin over the certified states was -0.023 to -0.077. The Euler fit's absolute radius certified all
15 steps on each, at margins +0.378 to +0.415.

**The Euler drift left the adjustment set out.** Under `"rk4"` a covariate that is not a driver
joins the drift regression, held over the step, because RK4 carries the drift into the channel
(ADR 0120). Under `"euler"` it did not, and the drift paid twice: a confounder's push that leans on
the state biased it, and one apart from the state widened its error. On `x' = x + (-0.7 x + 0.8 u +
z)` at `dt = 1`, a confounder leaning on the state by 0.5 set the decay at -0.195 over 60 logs of
1000 transitions, its 95 % interval covering -0.7 on none; drawn apart from the state, it left the
drift's error 99 times what it is with the push held. The channel was not touched: it is read off
the partialled-out moment. The plans were: the median regret over the six logs, `"rk4"`, then
`"euler"` as it was, then `"euler"` with the push held:

| world | `"rk4"` | `"euler"`, push left out | `"euler"`, push held |
|---|---|---|---|
| well-specified | 0.0018 | 0.0142 | 0.0018 |
| hidden confounder 0.4 | 0.325 | 0.343 | 0.327 |
| sine drift | 0.0242 | 0.0572 | 0.0238 |
| saturation 1 | 0.248 | 0.252 | 0.249 |
| dead time 1 | 2.588 | 2.728 | 2.593 |
| demand AR(1) 0.8 | 0.0113 | 0.990 | 0.0112 |
| disturbance AR(1) | 0.0084 | 0.0291 | 0.0088 |

With the push held, the Euler fit plans as well as the `"rk4"` fit in each world, and its barrier's
radius is a bound on the channel's error. The quickstart compiles 770 programs under `"euler"`
against 1053 under `"rk4"`, whose fit runs a Newton fixed point the Euler fit does not.

**Under the absolute radius a `gamma` at or past `(3 + sqrt 5)/2` is refused** (ADR 0125). A
default call with constraints at such a `gamma`, which runs today, raises once the default is
`"euler"`.

## Decision

- **The Euler fit holds the adjustment set's push in its drift**, as the `"rk4"` fit does: each
  covariate that is not a driver joins the drift regression in the nuisances' basis, held over the
  step, about its mean on the log, and leaves the returned field afterwards. `drift_error`, the
  drift's blocks and the influence leave the pushes' gains out, as under `"rk4"`.
- **`prescribe`'s default integrator becomes `"euler"` in 0.22.0.** In 0.21.0 `integrator=None`,
  the default, fits and plans by `"rk4"` and raises a `FutureWarning` that names the change; either
  value passed keeps its reading without it. `RunProvenance.integrator` records the map that ran.
- The library's examples pass the integrator: `"euler"` in the quickstart, the README and the
  start-here notebook, whose logs are weekly panels stepped by their one-step map; `"rk4"` in the
  heating notebook, whose room follows Newton's law of cooling in continuous time, and in the
  scripts whose recorded measurements were taken under `"rk4"`.
- `fit_causal_residual` keeps `"euler"`, its default already.

## Consequences

- An Euler fit with an adjustment set reads a different drift and a narrower drift error than in
  0.20: its plans move, by the table above. Its channel and the channel's error do not.
- From 0.22.0 a default `prescribe` call reads the absolute radius where it holds constraints, and
  its held plan mostly pays for it. The median held task cost over the six logs, every step
  certified under either integrator:

  | world | `"rk4"`, relative radius | `"euler"`, absolute radius |
  |---|---|---|
  | well-specified | 4.502 | 4.519 |
  | hidden confounder 0.4 | 2.875, a plan the plant left the set on | 4.418 |
  | sine drift | 4.690 | 4.933 |
  | saturation 1 | 4.170 | 4.705 |
  | dead time 1 | 5.364 | 7.101 |
  | demand AR(1) 0.8 | 4.533 | 4.439 |
  | disturbance AR(1) | 4.520 | 5.543 |

  The held rounds ended on their budget (`max_iterations`) on 13 logs of 42 under `"euler"`, six
  of them under the hidden confounder, and on none under `"rk4"`.
- On a plant in continuous time the Euler fit reads the step's map, not the field. On the heating
  notebook's room at `dt` of one hour, the heater's effect reads 0.97 to 0.98 °C an hour per kW,
  the hour's own effect `(1 - e^-k)/k = 0.975`, where the `"rk4"` fit reads the law's 1.00, and the
  plan's room on the true physics ends at 20.5 °C against 20.6. A caller who reads the fitted field
  as the plant's law passes `"rk4"`.
- A default call with constraints at `gamma >= (3 + sqrt 5)/2` raises from 0.22.0. The error
  names `radius="relative"`; `integrator="rk4"` keeps the reading of 0.21.
- A log that keeps less than 0.27 of a mode from one step to the next, which no field stepped by
  RK4 reproduces (ADR 0118), is planned by default from 0.22.0, where the `"rk4"` fit raises.

## Alternatives

- **Keep `"rk4"` as the default.** The default certificate's `gamma` then stays a margin on the
  estimate, which certified plans the plant left the set on.
- **Make `"euler"` the default in 0.21.0.** Every default plan moves, and some valid calls start to
  raise, in a release that did not say so. A default that moves is announced one minor ahead, as
  0.19.0 announced the defaults of ADRs 0125 to 0128.
- **Warn only where a barrier is held.** The fit and the plan move on every default call, so every
  default call is told.
- **Give the `"rk4"` fit an absolute radius.** It needs a sensitivity model of the fixed point's
  channel, which carries the drift's confounding (ADR 0111). Not done.
