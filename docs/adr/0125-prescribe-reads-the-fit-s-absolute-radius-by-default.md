# ADR 0125 — `prescribe` reads the fit's absolute radius by default, and a plan's tube one unit of the state

**Status:** accepted, 2026-10-21. Amends ADR 0111: `prescribe` no longer keeps the relative margin
where its fit gives an absolute radius. Makes ADR 0091's tolerance of one unit of the state the
default of `causal_plan` and `RecedingHorizon`. Amended by ADR 0129 (2026-10-24): from 0.22.0 the
default fit is Euler's, which gives the absolute radius.

## Context

ADR 0111 gave the barrier an absolute radius on the channel, `ChannelSensitivity`, read at each
step's predicted state, and left it opt-in. `prescribe` passed none. Its audit read the relative
radius, a share of the fitted channel's size at the plan's start: at every level the set stays
within the estimate's own size of the estimate, and `gamma = 1` is a radius of zero. ADR 0119 then
made a certified step carry the deviation a plant in the radius can reach, so the radius decides
what the certificate promises about the plant.

On the misspecification suite's plant, an unrecorded confounder moves the supply channel the log
reads from 0.8 to about 0.41. Fitted by Euler and held at `gamma = 2`, four logs, both radii
certified all 15 steps:

| log | plant's channel from the fit | absolute radius at the start | plant's least margin, absolute | relative |
|---|---|---|---|---|
| 0 | 0.398 | 0.519 | +0.372 | −0.186 |
| 1 | 0.390 | 0.548 | +0.427 | −0.022 |
| 2 | 0.405 | 0.520 | +0.376 | −0.136 |
| 3 | 0.396 | 0.547 | +0.441 | +0.093 |

On three logs of four the relative radius certified every step of a plan the plant left the set
on.

`causal_plan` and `RecedingHorizon` read a plan's tube against an infinite `tolerance` when none
was given. That holds every finite tube, so the plan's status read `no_tolerance`, and 0.19 warned
that the default would become one unit of the state.

## Decision

- **`prescribe(radius="auto")` is the default: the absolute radius where the fit gives one.**
  Where constraints give a barrier, `prescribe` reads `ChannelSensitivity.from_fit(fit)` at 0.95
  and passes it to the held solve (`BarrierConstraint(sensitivity=)`) and to the audit, so both
  read one radius at each step's predicted state. `gamma = 1` is then the channel's sampling
  error.
- **Where the fit gives no absolute radius**, `"auto"` reads the relative one, and an `INFO`
  record says why (`chc_event="radius"`, its `reason` the fit's refusal). That is an `rk4` fit,
  the default integrator's, whose fixed point carries the drift's confounding into the channel
  (ADR 0111); an instrument's fit; and a log that leaves a direction of the channel unmoved. The
  record is not a `WARNING`: the fallback is the default's own reading, and a default run prints
  nothing.
- **`radius="absolute"`** asks for the absolute radius, and on a fit that gives none raises a
  `DecisionError` that names the fit's refusal and the two ways out, `integrator="euler"` and
  `radius="relative"`.
- **`radius="relative"`** keeps the share explicitly, with no record. Any other value is refused.
- **Under the absolute radius**, a `gamma` at or past `(3 + sqrt 5)/2` is refused: the sensitivity
  model bounds no error there. The relative radius still reads it.
- `DecisionCertificate.radius_model` and `RunProvenance.radius`, with its JSON, say which radius
  the audit read, `"absolute"` or `"relative"`, never `"auto"`; None where no barrier was audited.
- **`"rk4"` stays `prescribe`'s default integrator** (ADR 0118). So the default run on a
  continuous-time log reads the relative radius; an absolute one needs `integrator="euler"`.
- **`causal_plan(tolerance=1.0)` and `RecedingHorizon(tolerance=1.0)`** are the defaults, and the
  `FutureWarning` is gone. `inf` still reads `no_tolerance`. `prescribe`'s `tolerance` is
  unchanged: omitted, it evaluates no tube.

## Consequences

- An Euler fit's certificate now reads an absolute radius, and the held plan pays for it: on the
  logs above, the task cost at `gamma = 2` rose from 5.43, 4.94, 5.26, 4.68 to 6.70, 5.92, 6.32,
  5.78, and the rounds more often ended on their budget (`max_iterations`), with every step still certified.
- An `rk4` fit reads the relative radius as before: no number of the quickstart, the README, the
  pendulum case study, the BOPTEST case study or the misspecification suite's `rk4` cases moves.
  Each such run with a constraint logs one `INFO` record more, which prints nothing unless the
  caller configures logging.
- `certify_safety` and `BarrierConstraint` keep the relative radius by default: they take a model,
  not the fit a sensitivity is read from (ADR 0111).
- A plan from `causal_plan` with an error model and no `tolerance` now reads `certified`,
  `partial` or `uncertified` against one unit of the state, where it read `no_tolerance`.

## Alternatives

- **Refuse a fit that gives no absolute radius by default.** Every default `prescribe` with a
  constraint would raise, since the default fit is `rk4`. Only an explicit `radius="absolute"`
  raises.
- **Fall back with a `WARNING`.** Every default run with a constraint would print to stderr about
  a choice the caller did not make.
- **Make `"euler"` the default integrator.** It reads a continuous-time log with Euler's
  discretisation bias (ADR 0118); that is a decision of its own.
- **A per-step share, `||B_hat(x_k) D||`.** Still relative: it reads no error (ADR 0111).
- **Keep the default relative, opt in to the absolute.** It leaves the default certificate's
  `gamma` a margin on the estimate where the fit gives a bound on the channel's error.
