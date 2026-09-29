# ADR 0023 — Evaluating a schedule

**Status:** proposed, 2026-09-29.

## Context

`prescribe` returns a plan over a horizon: `CausalPlan.actions`, `(H, m)`, open loop, which
`Prescription.schedule` exposes. ADR 0009's evaluation took one `AffinePolicy`, held for ever. A
prescription could reach `evaluate_plan` only as a stationary policy, which is a different plan: a
media plan's weekly budgets, or a campaign that ends, is its time profile.

Two ways closed the gap: `prescribe` returns a stationary policy, or the evaluation takes a plan
that changes with the step and ends. What each method needs for the second:

- `"pdis"` reads episodes over the logged horizon already. Its weight's second moment needs each
  step's gain and offset, and nothing else changes, if the recursion runs forward. ADR 0009's ran
  backward from the last step, and one pass served every horizon only because the loop did not
  change with the step.
- `"mis"` and `"dr"` read the loop's stationary laws. A schedule that ends has none.
- `"fqe"`, as built, is average-cost LSTD-Q: one Q function for a stationary loop.

## Decision

- **`AffineSchedule(gains, offsets, covariance)`**: `u_t | x_t ~ N(gains[t] x_t + offsets[t],
  covariance)` for `t < H`, with `gains (H, m, n)` and `offsets (H, m)`. Deterministic when the
  covariance is zero. `AffineSchedule.open_loop(actions, states, covariance=None)` builds the
  open-loop kind a prescription is, every gain zero.
  - **One covariance for every step.** A deterministic plan is evaluated through one smoothing,
    chosen by one rule (ADR 0009), and a plan's dither, when it carries one, is one covariance.
    Per-step covariances would need a smoothing limit per step, and no caller asks for them.
- **`certify_evaluation` and `evaluate_plan` take `AffinePolicy | AffineSchedule`.**
  - Only `"pdis"` evaluates a schedule. `"mis"`, `"dr"` and `"fqe"` raise `ValueError`, naming the
    reason: each evaluates a stationary loop's average cost.
  - `horizon` defaults to the schedule's. A schedule over another horizon than the episodes' is
    refused, not truncated: the first `h` steps of a schedule are not the schedule, and a caller
    who means them slices it.
  - A policy is evaluated as the schedule that holds it for every logged step, through the same
    code, so its certificate is the one it was.
- **The recursion runs forward.** It carries the unnormalised law of the state along the starred
  loop: each step's factor, Gaussian in the state, tilts the law and adds its log mass, and after
  `h` steps the mass is `log E_b[W_h^alpha]`. One pass serves every horizon whether the plan
  changes or not.
- **The smoothing correction** over a schedule is `tau^2 sum_t tr(R + B' P_{t+1} B)`, with `P`
  run backward over the schedule's own gains.
- **A target per state** (ADR 0006): `"pdis"` scores step `t` against row `t` of an `(H + 1, n)`
  `x_target`, and no step reads the last row, as no step reads `Qf`. The stationary methods refuse
  such a cost, as before.
- **The one-step margin** of a schedule is the smallest over its steps, reported for comparison
  only, as ADR 0009's is.

## Consequences

- The tests check the schedule against what is independent of the module:
  - on 80 random schedules the moment matches the whole-trajectory integral, which writes the log
    weight as one Gaussian quadratic form in every draw, to a relative 1e-9;
  - a schedule that holds one policy has that policy's certificate, field for field, and the
    verifier's numbers of ADR 0009 hold as they did;
  - a ramped feedback schedule and an open-loop one, against a moving target, are covered by the
    interval at least 0.85 of 60 replicates, with the mean error within three standard errors;
  - the smoothing correction is exact over the schedule's own gains;
  - with the plan equal to the logger, the estimate is the logs' mean cost, each step against its
    own row.
- The public surface grows by one class, `AffineSchedule`, at the top level: `evaluate_plan` takes
  it, and ADR 0016 puts what a lifecycle call takes there.
- A prescription can reach `evaluate_plan` through `AffineSchedule.open_loop`, but the rest of the
  loop does not connect yet: nothing builds the `LinearGaussianPlant` a prescription's model
  linearises to, or the episode arrays `evaluate_plan` reads from a `Panel`.
- Everything is exact on a linear-Gaussian loop and no more, as ADR 0009 says.

## Alternatives considered

- **`prescribe` returns a stationary policy.** Rejected: it would evaluate a plan other than the
  one prescribed.
- **A sequence of `AffinePolicy`, one per step.** Rejected: it allows a covariance per step, which
  one smoothing rule cannot serve, and has no horizon of its own to check the episodes against.
- **Truncating a longer schedule to the logged horizon.** Rejected as a silent default, for the
  reason above.
- **A finite-horizon FQE**, per-step quadratic Q functions fitted backward. It would evaluate a
  schedule where every weight is refused. Deferred: it is an estimator of its own, with its own
  interval, and nothing here depends on it.
