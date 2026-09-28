# ADR 0007 — Exogenous drivers

**Status:** accepted, 2026-09-28.

## Context

Run L8.1 (`docs/case-studies/boptest.md`) found three things `prescribe` could not state. Two have
shipped: a bound on the steered state (ADR 0005) and a target that moves within the horizon
(ADR 0006). The third is a column the plan cannot move but can forecast, such as outdoor
temperature, solar gain or a published demand forecast.

`fit_causal_residual` fitted the drift on the state alone, so a driver's push was averaged into it
together with its correlation with the state. The adjusted arm partialled the weather out of the
channel and still planned for average weather.

## Decision

- **`chc.dynamics.DrivenDynamics(dynamics, gain, levels, dt, start=0.0)`** wraps any `Dynamics` as
  `f(t, x, u) + G w(t)`.
  - `levels[k]` is the drivers' value at `start + k dt`. Between two levels they move linearly
    (a first-order hold), and past either end the nearest segment is extended.
  - A zero-order hold is not expressible: RK4's last stage of step `k` and the first of step
    `k + 1` are both at `(k + 1) dt`, so no function of `t` alone holds a level through a step.
  - The term reads neither `x` nor `u`. So the barrier's channel `B^T grad h` is the undriven
    plant's, and the forecast enters its drift term at each step's own time. `gamma*` is priced
    with the forecast in place.
- **`fit_causal_residual(..., drivers=(...))`** fits the gain jointly with the drift.
  - `data` carries each driver at both ends of a transition, `name` and `f"{name}_next"`.
  - The drift regression reads the driver at its mean over the step under `rk4` (what a linear
    path puts into the step) and at the start under `euler` (all the Euler map reads).
  - Both ends join the channel's nuisance covariates. A driver is exogenous, so partialling it
    out cannot open a path. A logger that acts on where the driver is going is confounded through
    the end level.
  - The `rk4` defect steps each transition with a two-level `DrivenDynamics`. The fit and the plan
    therefore share one implementation of the hold.
  - The gain is `CausalDynamicsFit.driver_gain`.
- **`chc.decision.Driver(name, forecast)` and `prescribe(..., drivers=[...])`.**
  - A forecast has `horizon + 1` levels, one at each `t = k dt`.
  - `prescribe` refuses a driver that is also a lever or a state, one named twice, a forecast of
    the wrong length and a non-finite forecast, all with `DecisionError`. A missing column raises
    `KeyError`.
  - A forecast outside the logged range is logged as extrapolation (`chc_event="driver_range"`).
  - The report prints each driver's push on the target's rate, and `to_json` carries the
    forecasts.
- **Prerequisites, each its own fix.**
  - The discrete adjoint steps its clock as the rollout does (`b3a28b9`).
  - `mpc_control` steps its plant at the loop's time, and `RecedingHorizon.step(x, t)` plans a
    window from `t`. A `DrivenDynamics` built on absolute time is then read at the right level in
    every window.

## Evidence

`tests/test_drivers.py` uses zones heated against the weather: 100 zones, 24 steps, `dt = 0.1`,
0.01 process noise. The true plant is `temp' = −0.5 temp + 0.8 heat + 1.2 outdoor`, with a
periodic outdoor path. The logger heats against the weather it sees. The simulator is RK4 written
out in numpy and shares no code with the library.

| check | measured | threshold |
|---|---|---|
| gain, decay, channel with the driver, over 8 seeds | 1.188–1.205, −0.504 to −0.487, 0.790–0.810 | ±0.025, ±0.03, ±0.02 |
| decay without the driver, over 8 seeds | −0.18 to −0.005, defect 0.80–0.82 against a 0.10 floor | off by > 0.25 |
| plan on the true plant against a hand-written QP | 0.001 (actions up to 2.3) | < 0.01 |
| same, forecast read one step late / not at all | 0.27 / 0.98 | > 0.1 / > 0.5 |
| window started at `t = 7 dt` against the full plan's tail | 0.0009 (1.70 if started at 0) | < 0.01 |
| adjoint against central differences, cubic driven plant | ≤ 6.2e−10 | < 1e−7 |
| `prescribe` cost on the true plant ÷ oracle, 4 seeds | 1.0001–1.0036 with the forecast, 1.69–1.98 without | < 1.02, > 1.5 |
| `euler` channel, logger acting on the next level, 6 seeds | 0.656 with the start alone, 0.780 with both ends | 0.7803 ± 0.015 |

- **Mutation checks.**
  - With the hold read one step early, four tests fail: the hold, the fit, the QP oracle and the
    façade.
  - With the adjoint's clock put back at `t = 0`, the central-difference test fails.
  - With the end level dropped from the nuisance, the anticipating-logger test reads 0.663.
- **The gate.** One call states all three of L8.1's missing things: a band on `temp`, a target
  rising from 0.4 to 1.0, and the weather forecast. The channel is identified, the barrier is
  priced on the driven drift (`gamma*` 5.59), and the report prints the push, `+1.198`.
- **Found along the way, and fixed as its own change.** On these logs the `rk4` correction
  stopped after one or two passes. Its 1% progress bar could not see a bias sitting under the
  noise. That fix lives in `fit_causal_residual`, which now solves the `rk4` fixed point by
  Newton's method instead of stopping on the defect, not here. With it, the variants tried (start
  or mean reading, with or without the end level) reach the same point under `rk4`.
  Under `euler` the end level is what removes the anticipating logger's confounding (the table's
  last row).

## Consequences

- `dynamics`, `dynamics_id` and `decision` are stable modules, and each gains an input. Without
  drivers none of this is on the path: the drift design and the nuisance covariates are the state
  and the adjustment set, as before. (The same release changes every `rk4` fit, now solved to its
  fixed point; that is its own record in the changelog.)
- **Not priced:** the forecast's own error, and the gain's standard error. The tube's model error
  leaves both out, as it leaves out the drift's. A forecast with a stated error belongs to
  scenario or chance constraints, which are not built.
- **A shock held constant through each period is not a driver.** An example is the marketing-mix
  season, drawn once a week. The first-order hold cannot represent it, so it stays an adjustment
  covariate.
- `mpc_control` and `RecedingHorizon` read a driven model at absolute time. The levels must cover
  the whole run; past the last level the forecast is extrapolated linearly.
- L8.1's rerun with all three statements needs the BOPTEST stack, and has not been run.

## Alternatives

- **Drivers as pinned inputs.** Rejected: every lever
  consumer would have to skip them, including `u_max`, rate limits, `max_levers`, the schedule and
  the report. They would also enter the barrier's confounding term `d ||u||`, so `gamma*`'s best
  action would pick the weather.
- **Drivers as DML treatments.** Rejected: a forecast would move the lever's identified channel,
  and a driver in the adjustment set cannot also be a treatment.
- **Drivers as states held by `w' = 0`.** Rejected: persistence cannot carry a forecast that moves.
- **The forecast inside the fitted residual.** Rejected: the residual is the fitted object, and a
  new forecast would then need a new fit rather than a new call.
- **A `t0` threaded through the eight stable functions that roll out.** Rejected: an absolute-time
  hold, read through the controllers' time shift, needs none of them to change.
