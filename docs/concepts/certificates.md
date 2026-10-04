# Certificates

A plan without the evidence of where it may be trusted is half an answer. `causal_plan` returns
one object that carries both, so a caller cannot walk away with the actions and leave the
certificate behind.

## Plan, audit, filter

"Safety" alone does not say which of three operations you get, so they are named apart:

- **plan** — [`causal_plan`](../api/plan.md). The box and linear constraints on the actions
  (budgets, rate limits) are in the solve, plus an *a-priori* Grönwall error tube that says how far
  ahead the plan may be trusted. The tube is computed from `lipschitz` and `model_error`; it does
  not enter the objective and does not move a single action. A `BarrierConstraint` does: passed as
  `barrier=`, the audit's own condition is held in the solve by augmented-Lagrangian rounds.
- **audit** — `certify_safety`. Given a barrier and a sensitivity level `Γ`, it prices a *finished*
  plan: where along it the safety guarantee survives unmeasured confounding, and the largest `Γ`
  the whole plan tolerates. Read-only by construction.
- **filter** — [`robust_safety_filter`](../api/barrier.md). It clips one nominal action into the
  certified interval at one state, online, for a scalar control.

A held barrier binds the *answer*, not every iterate as the box and the rows do, and the solver's
word is not taken for it: a solve stopped by its budget, or a condition no admissible action can
meet, comes back short of the condition, and `plan.safety` — the audit, run on the finished plan —
says so. The audit's verdict is the one to read, even on a plan solved under the barrier. The
design record is
[ADR 0002](https://github.com/causaldyn/causal-hybrid-control/blob/main/docs/adr/0002-barrier-in-the-solve.md).

## What each certificate says

- **The error tube** — the prefix of the plan whose Grönwall tube stays inside `tolerance`, from
  Lipschitz or contractive-log-norm bounds. `plan.certified_actions` is that prefix and nothing
  more. With no error model supplied the certificate reports `not_evaluated` rather than a vacuous
  full-horizon pass.
- **The barrier** — for a control-affine plant and a safe set `{h >= 0}`, whether the barrier
  condition survives an identification radius on the effect, step by step, and `Γ*`: the largest
  level of the [marginal sensitivity model](gamma.md#two-models-one-symbol) at which every step
  along the plan's path still has an admissible action that meets the condition. It is a ceiling
  for the problem along that path, not for the plan: whether the plan's own actions meet it is
  `planned_certified`, which a plan can fail under a comfortable `Γ*`. The plan-level `Γ*` is the
  weakest step's, so one uncertifiable step sinks the plan. Where the barrier is flat, as at the centre of a ball, the radius is zero at
  every level and the step's `Γ*` is `inf` or `nan`.
  Where two bounds tie, as at the midpoint of a two-sided one, `prescribe` audits each and reports
  the weaker `Γ*`: an upper bound on the joint ceiling, because each bound's is reached by its own
  best action and no single action need reach both.
- **The optimality gap** — `plan_regret_bound` certifies how far a finished plan can be from the
  best one its own box allows, from the plan's own gradient and with no optimum needed. It reads the
  plan's actions, not the solver's internals, so it prices a plan that came from anywhere —
  including one an operator edited by hand. When the measured curvature is negative no convexity
  argument applies, and the bound is `inf` rather than a fabricated number.
- **The regret guarantee** — [`chc.regret`](../api/regret.md) carries the LQ certainty-equivalence
  bound, quadratic in model error (Dean–Mania–Tu–Recht–Matni), and its extensions.

`prescribe` gathers the tube, the barrier and the optimality gap into a `DecisionCertificate`,
beside the identification status. The two axes are never merged: a plan can be fully certified
against a channel nothing identifies, which is a trustworthy tube around a meaningless action.

## Objectives are protected; constraints are not

The same effect error costs the two halves of a decision differently. Performance regret is
*second* order in the effect bias: an interior optimum has a vanishing gradient. The barrier margin
pays at *first* order, because a binding constraint has no envelope to protect it. The comparison
holds while the identification radius is smaller than the control channel, which is where any
usable controller lives; past it the safety guarantee is simply gone. So a budget sized for regret
is the wrong budget for safety.

## What the certificate is worth

`certify_safety` checks the barrier condition pointwise along the model's trajectory. Forward
invariance follows only when the condition holds on the whole safe set.
[`chc.reachability`](../api/reachability.md) computes the Hamilton–Jacobi answer the barrier only
approximates, on the same robust-margin algebra, and prices the difference: where the condition
holds on all of `{h >= 0}` the reachable tube *is* that set, and where it only holds pointwise,
`certified_but_unreachable` says how much the per-step reading over-promises. A per-step certified
prefix is a filter, not a proof.

## See it

- `uv run python scripts/reachability_demo.py` — the pointwise check against the true reachable
  tube, on the two-zone supply floor and on a double integrator where the barrier has relative
  degree 2.
- The [driver-supply case study](../case-studies/driver-supply.md) — `Γ*` tells a confounded plan
  from an adjusted one before either acts.
