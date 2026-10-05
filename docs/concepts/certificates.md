# Certificates

A plan without the evidence of where it may be trusted is half an answer. `causal_plan` returns
one object that carries both, so a caller cannot walk away with the actions and leave the
certificate behind.

## Scopes

Each certificate bounds one thing, on one object, under premises it does not check. The other
pages point here rather than repeat the claims.

| certificate | what it bounds | on what | its premises | proved or measured |
|---|---|---|---|---|
| error tube: `causal_plan`, `prescribe` | the gap between the plan's rollout on the model and on the plant, step by step, and the prefix that stays inside `tolerance` | the plan's own RK4 steps (Euler's on request), on the model and on a plant whose field is within `model_error` of it along the path | a per-step bound on the field's error; a rate that bounds the field's slope in the state, or the state matrix of a field affine in it. Under `prescribe` the error is the channel's standard error, so the tube is a scale, not a bound, and its rate holds at every state (`global`) or at the start alone (`local`) | the recursions bound the error under those premises: Rocq `proofs/lipschitz_rollout.v`, Maxima `validation/rk4_rollout.mac`; the pendulum and BOPTEST case studies measure it |
| identification status: `prescribe` | whether the graph identifies the levers' channel: `identified`, `asserted` or `not_identified` | the panel and the graph passed as `adjustment` | the graph is right; a lever the log never moves is held at its logged level, not identified | the adjustment set is Perković et al.'s canonical one, fuzzed against the criterion |
| barrier audit and `Γ*`: `certify_safety`, `prescribe` | whether the barrier condition holds at each step under an identification radius; `Γ*`, the largest level at which every step still has an admissible action that meets it | the model's predicted path, pointwise: a filter, not a proof that the safe set is invariant | a control-affine plant, a barrier `h`, the marginal sensitivity model | the algebraic core: Rocq `proofs/barrier_feasibility.v`, at any dimension in `proofs/mathcomp/`; what the pointwise check misses: [`chc.reachability`](../api/reachability.md) |
| optimality gap: `plan_regret_bound` | how far the plan's objective is above the best plan its box allows | the planning model, not the plant | a curvature that holds over the whole box: supplied, or a linear model's (`certified`); sampled at the plan and a few points it is a `diagnostic`, and negative it is `refused` | Rocq `proofs/constrained_plan_regret.v`, Maxima `validation/constrained_plan_regret.mac` |
| LQ regret: [`chc.regret`](../api/regret.md) | the certainty-equivalent controller's regret, quadratic in the model's error | a linear-quadratic plant | the model's error inside the ball the bound computes | the gain-gap identity, the ball and the constant: Rocq `proofs/ce_explicit_constants.v`, at any dimension in `proofs/mathcomp/` |
| interference regret: `chc.regret` | regret quadratic in the identification and exposure-map errors together | a smooth cost and its minimiser | the planned policy within `C (e_id + e_int)` of the best one, assumed, not proved | the algebra from that premise: Rocq `proofs/interference_regret.v` |
| evaluation: `certify_evaluation` | whether logs of a given size can evaluate a plan: every weight's variance finite, and enough effective samples, before any cost is read | a linear-Gaussian plant and affine policies | the plant's model | closed forms in [`chc.evaluation`](../api/evaluation.md) |
| conformal width: `SplitConformal` | marginal coverage of at least `1 − α` | data exchangeable with the calibration split | exchangeability; `n ≥ (1 − α)/α` calibration scores, which `calibrate` enforces | the split-conformal argument, cited; `coverage` measures it |
| deployment gate: [`chc.gate`](../api/gate.md) | the expected share of wrong deploys across zones, at most `α` at every read | the zones' logs in shadow | logged propensities, no spillover, no carryover | e-processes and e-BH, cited |
| delay margin: [`chc.delay`](../api/delay.md) | the delay the loop tolerates, `arccos(a/K)/sqrt(K² − a²)` | a scalar loop with one discrete delay | the plant's rate `a`, and a gain `K` larger than `a` in magnitude | the algebraic core: Rocq `proofs/delay_margin.v`; the smallest crossing: Maxima `validation/delay_margin.mac`, checked numerically |

## Plan, audit, filter

"Safety" alone does not say which of three operations you get, so they are named apart:

- **plan** — [`causal_plan`](../api/plan.md). The box and linear constraints on the actions
  (budgets, rate limits) are in the solve, plus an *a-priori* Grönwall error tube that says how far
  ahead the plan may be trusted under its premises ([the scopes](#scopes)). The tube is computed
  from `lipschitz` and `model_error`; it does not enter the objective and does not move a single
  action. A `BarrierConstraint` does: passed as
  `barrier=`, the audit's own condition is held in the solve by augmented-Lagrangian rounds.
- **audit** — `certify_safety`. Given a barrier and a sensitivity level `Γ`, it prices a *finished*
  plan: where along it the safety guarantee survives unmeasured confounding, and `Γ*`, a ceiling
  for the problem along the plan's path rather than for the plan (below). Read-only by
  construction.
- **filter** — [`robust_safety_filter`](../api/barrier.md). It clips one nominal action into the
  certified interval at one state, online, for a scalar control.

A held barrier binds the *answer*, not every iterate as the box and the rows do, and the solver's
word is not taken for it: a solve stopped by its budget, or a condition no admissible action can
meet, comes back short of the condition, and `plan.safety` — the audit, run on the finished plan —
says so. The audit's verdict is the one to read, even on a plan solved under the barrier. The
design record is
[ADR 0002](https://github.com/causaldyn/causal-hybrid-control/blob/main/docs/adr/0002-barrier-in-the-solve.md).

## What each certificate says

- **The error tube** — the prefix of the plan whose error tube stays inside `tolerance`. The tube
  follows the plan's own RK4 steps, at a rate that bounds the norm of the field's slope in the
  state, or at the state matrix of a field affine in the state, whose RK4 propagators carry it.
  `prescribe` says where its rate holds: `global` on a field affine in the state, `local` where
  the slope was read at the plan's start alone. `plan.certified_actions` is that prefix and nothing
  more. With no error model supplied the certificate reports `not_evaluated` rather than a vacuous
  full-horizon pass. The tube is a bound only where `model_error` bounds the field's error; the
  error `prescribe` supplies is the channel's standard error, so there the tube is a scale.
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
- **The optimality gap** — `plan_regret_bound` bounds how far a finished plan can be from the
  best one its own box allows on the planning model, from the plan's own gradient and with no
  optimum needed. It reads the plan's actions, not the solver's internals, so it prices a plan that
  came from anywhere — including one an operator edited by hand. It is a certificate where the
  curvature it divides by holds over the whole box: supplied by the caller, or read off a linear
  model's objective, which is quadratic, so one Hessian is the box's (`status="certified"`).
  Where the curvature is sampled at the plan and a few points of the box, the bound is a diagnostic
  (`"diagnostic"`): a pocket of negative curvature between the samples can hide a much cheaper
  plan. When the curvature read is negative no convexity argument applies, and the bound is `inf`
  rather than a fabricated number (`"refused"`). `prescribe`'s fitted channel reads the state, so
  its bound is a diagnostic.
- **The regret guarantee** — [`chc.regret`](../api/regret.md) carries the LQ certainty-equivalence
  bound, quadratic in model error (Dean–Mania–Tu–Recht–Matni), and its extensions. It holds on a
  linear-quadratic plant whose model's error lies inside the ball the bound computes.

`prescribe` gathers the tube, the barrier and the optimality gap into a `DecisionCertificate`,
beside the identification status. The two axes are never merged: a plan can be fully certified
against a channel nothing identifies, which is a trustworthy tube around a meaningless action.
A lever the log never moves apart from what the states and the covariates predict is not
identified on that log, whatever the graph says: `prescribe` holds it at its logged level and the
certificate names it (`unmoved_levers`), and with every lever so there is no plan.

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
