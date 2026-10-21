# Certificates

A plan without the evidence of where it may be trusted is half an answer. `causal_plan` returns
one object that carries both, so a caller cannot walk away with the actions and leave the
certificate behind.

## Scopes

Each certificate bounds one thing, on one object, under premises it does not check. The other
pages point here rather than repeat the claims.

| certificate | what it bounds | on what | its premises | proved or measured |
|---|---|---|---|---|
| error tube: `causal_plan`, `prescribe` | the gap between the plan's rollout on the model and on the plant, step by step, and the prefix that stays inside `tolerance` | the plan's own RK4 steps (Euler's on request), on the model and on a plant whose field is within `model_error` of it along the path, and, where `noise` is given, whose next state is off the plan's step by a noise of that covariance | a per-step bound on the field's error; a rate that bounds the field's slope in the state, or the state matrix of a field affine in it; a Gaussian noise, AR(1) from step to step, which the tube holds with probability 0.95 at each step. Under `prescribe` the field's error is the channel's standard error, each lever's at that lever's largest distance from the log's mean action, in any units of each lever and wherever its zero is, with the drift's own in quadrature, both read at the state each planned step starts from, a standard error read at Student's `t`, as an error independent of the noise and, on several states, along the noise's widest direction, with which the radius holds it at 0.95, and the noise is the plan's one-step error on the log; its rate holds at every state (`global`), where the fitted slope at each planned action carries the tube, or at the start alone (`local`) | the recursions bound the error under those premises: Rocq `proofs/lipschitz_rollout.v`, Maxima `validation/rk4_rollout.mac`, `validation/predictive_tube.mac`, `validation/exact_tube.mac` and `validation/tube_level.mac`; `tests/test_predictive_tube.py` and `tests/test_exact_tube.py` measure the noise's coverage, and the pendulum and BOPTEST case studies the tube's |
| identification status: `prescribe` | whether the graph identifies the levers' channel: `identified`, `asserted` or `not_identified` | the panel and the graph passed as `adjustment` | the graph is right; a lever the log never moves is held at its logged level, not identified | the adjustment set is Perković et al.'s canonical one, fuzzed against the criterion |
| barrier audit and `Γ*`: `certify_safety`, `prescribe` | whether the barrier condition holds at each step under an identification radius, with the path's state and the next in the safe set by the width of the deviation tube a plant whose channel is in the radius can reach under the plan's actions; `Γ*`, the largest level at which every step still has an admissible action that meets it | the model's predicted path and that tube, at the steps: a filter, not a proof that the safe set is invariant between them; the tube is a bound on a model affine in the state with a channel error that does not move with the state and an affine `h`, first order elsewhere, and covers the channel's error alone, not the drift's | a control-affine plant, checked where the audit reads it; a barrier `h`; under `certify_safety` by default a relative margin, the factor `(Γ-1)/(Γ+1)` on a share of the fitted channel, so even `Γ* = ∞` speaks only for effects within that share of the estimate; with `sensitivity=`, and under `prescribe` by default, an absolute radius under a sensitivity model of the channel, read at each step's state, for an Euler fit by adjustment | the algebraic core: Rocq `proofs/barrier_feasibility.v`, at any dimension in `proofs/mathcomp/`; what the pointwise check misses: [`chc.reachability`](../api/reachability.md) |
| optimality gap: `plan_regret_bound` | how far the plan's objective is above the best plan its box allows | the planning model, not the plant | a curvature that holds over the whole box: supplied, or a linear model's (`certified`); sampled at the plan and a few points it is a `diagnostic`, and negative it is `refused` | Rocq `proofs/constrained_plan_regret.v`, Maxima `validation/constrained_plan_regret.mac` |
| LQ regret: [`chc.regret`](../api/regret.md) | the certainty-equivalent controller's regret, quadratic in the model's error | a linear-quadratic plant | the model's error inside the ball the bound computes | the gain-gap identity, the ball and the constant: Rocq `proofs/ce_explicit_constants.v`, at any dimension in `proofs/mathcomp/` |
| interference regret: [`chc.research.regret`](../api/research/regret.md) | regret quadratic in the identification and exposure-map errors together | a smooth cost and its minimiser | the planned policy within `C (e_id + e_int)` of the best one, assumed, not proved | the algebra from that premise: Rocq `proofs/interference_regret.v` |
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
  full-horizon pass, and with an infinite `tolerance`, which holds every finite tube,
  `no_tolerance`; `causal_plan`'s default `tolerance` is 1, one unit of the state. The tube is a bound only where `model_error` bounds the field's error; the
  error `prescribe` supplies is a standard error, so there that part is a scale. The
  plant's own noise is the other part: given as `noise`, the tube adds the radius that holds it at
  each step with probability 0.95, and `prescribe` reads it off the plan's one-step error on the
  log (`tube_noise`, `tube_persistence`), carried from step to step as a VAR(1) whose coefficient
  is `tube_autoregression`, so that each direction of the noise persists at its own rate
  ([ADR 0105](https://github.com/causaldyn/causal-hybrid-control/blob/main/docs/adr/0105-the-tube-carries-the-noise-as-a-var1.md)).
  Where the channel's residual persists within units (`noise_persistence_p` at most 0.05), the
  noise the tube reads may not be the plant's, and a `tube` warning says so. Without the noise, on logs shaped like the BOPTEST
  case's, a tube of 0.05 K one step ahead trusted 6 to 13 steps, and the plant left it at the
  first step on 60% of draws; with it, the tube trusts 1 or 2 steps, where the plant's error
  allows 2 or 3
  ([ADR 0080](https://github.com/causaldyn/causal-hybrid-control/blob/main/docs/adr/0080-a-plan-s-tube-carries-the-plant-s-noise.md)).
  On a field affine in the state `prescribe` carries both parts by the fitted slope at each
  planned action, and the noise's radius is its exact 0.95 quantile at each step; elsewhere a
  bound on the slope's norm carries them, and the radius holds at least 0.95. `tube_mode` says
  which (`exact` or `bound`), and under `exact` `tube_state_noise` gives each state's own radius
  of the noise, in that state's units. The field's error `prescribe` supplies is the channel's
  and the drift's standard error together, read at the state each planned step starts from, where
  a plan that heads away from the log's states meets more of it, at its studentised quantile
  ([ADR 0104](https://github.com/causaldyn/causal-hybrid-control/blob/main/docs/adr/0104-the-tube-reads-the-field-s-error-where-the-plan-goes.md)).
  The quickstart's plant stays within its tolerance for 3 weeks, and the certificate now trusts 3,
  where it trusted 1
  ([ADR 0096](https://github.com/causaldyn/causal-hybrid-control/blob/main/docs/adr/0096-the-tube-is-carried-by-the-field-s-step-matrices-at-its-exact-radius.md)).
- **The barrier** — for a control-affine plant and a safe set `{h >= 0}`, whether the barrier
  condition survives an identification radius on the effect, step by step, and `Γ*`: the largest
  level of the [marginal sensitivity model](gamma.md#two-models-one-symbol) at which every step
  along the plan's path still has an admissible action that meets the condition. It is a ceiling
  for the problem along that path, not for the plan: whether the plan's own actions meet it is
  `planned_certified`, which a plan can fail under a comfortable `Γ*`. The plan-level `Γ*` is the
  weakest step's, so one uncertifiable step sinks the plan. Its radius is the model's factor
  `(Γ-1)/(Γ+1)` times a share of the fitted channel, so at every level the identified set stays
  within that share of the estimate: `Γ* = ∞` says nothing of a channel the fit has wrong by more.
  Where the barrier is flat, as at the centre of a ball, the radius is zero at
  every level and the step's `Γ*` is `inf` or `nan`.
  Read so, `certify_safety`'s default `Γ*` is a relative margin, not a level of confounding: an
  estimate of 0.004 for an effect of 1 is certified at every level along a plan the effect takes
  out of the set. `sensitivity=ChannelSensitivity.from_fit(fit)` gives it a model's meaning, for an
  Euler fit by adjustment: a hidden confounder moves the levers' density by a factor within
  `[1/Γ, Γ]`, and the radius at each step's predicted state is the channel's sampling error plus
  the bias that allows, in the channel's own units. Over 200 confounded logs whose truth lay inside
  that radius, the relative margin certified a prefix the truth left the set within on 26–28 % of
  them, the absolute radius on none ([ADR 0111](https://github.com/causaldyn/causal-hybrid-control/blob/main/docs/adr/0111-gamma-star-from-a-sensitivity-model-of-the-channel.md)).
  `prescribe` reads its fit's own absolute radius by default, in the held solve and in the audit,
  and the relative margin where the fit gives none, an `rk4` fit's among them, or where
  `radius="relative"` asks for it; the certificate's `radius_model` says which
  ([ADR 0125](https://github.com/causaldyn/causal-hybrid-control/blob/main/docs/adr/0125-prescribe-reads-the-fit-s-absolute-radius-by-default.md)).
  The radius is `(Γ-1)/(Γ+1)·cvar_gap` times the channel's operator norm at the plan's first
  state: `cvar_gap` is a share of the channel, 1 by default, so `Γ*` and the certified steps read
  the same in any units of the state or of the levers
  ([ADR 0085](https://github.com/causaldyn/causal-hybrid-control/blob/main/docs/adr/0085-gamma-star-s-gap-is-a-share-of-the-channel.md)).
  Under `prescribe` the audit reads each lever in units of its own largest action: the action set
  is the ellipsoid inside the box whose semi-axes those are, so `Γ*` and the certified steps read
  the same in any units of each lever
  ([ADR 0091](https://github.com/causaldyn/causal-hybrid-control/blob/main/docs/adr/0091-the-barrier-reads-each-lever-in-units-of-its-own-largest-action.md)).
  Where two bounds tie, as at the midpoint of a two-sided one, `prescribe` audits each and reports
  the weaker `Γ*`: an upper bound on the joint ceiling, because each bound's is reached by its own
  best action and no single action need reach both.
  A plant whose channel is in the radius is not at the model's states: under the plan's actions
  its path drifts from them. A step is certified only where `h - |∇h| e` is not negative at its
  state and the next, `e` the deviation tube that the radius can open between the two paths
  (`SafetyCertificate.deviation`). Read at the model's states alone, an exact radius of 0.1004 on
  a drift of 0.3 certified 60 of 60 steps along which the plant at the radius's edge reached
  `h = -0.132`
  ([ADR 0119](https://github.com/causaldyn/causal-hybrid-control/blob/main/docs/adr/0119-the-barrier-certificate-carries-the-plant-s-deviation.md)).
  `Γ* = ∞` does not read that tube, so it can stand beside no certified step.
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
  its bound is a diagnostic. A plan solved under linear rows, a budget say, is priced against the
  box and those rows when they are passed as `constraints=`: a plan at the optimum under a binding
  budget read 0.0885 against the box alone and 4e-13 against the box and the row.
- **The regret guarantee** — [`chc.regret`](../api/regret.md) carries the LQ certainty-equivalence
  bound, quadratic in model error (Dean–Mania–Tu–Recht–Matni), and
  [`chc.research.regret`](../api/research/regret.md) its extensions. It holds on a linear-quadratic
  plant whose model's error lies inside the ball the bound computes.

`prescribe` gathers the tube, the barrier and the optimality gap into a `DecisionCertificate`,
beside the identification status, with the barrier's `radius_model` and `lever_range_step`: the
first step at which the plan sets a lever outside the range the log moved it in, where the fitted
channel is extrapolated and its error is not read, so the certified horizon ends there. The two axes are never merged: a plan can be fully certified
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
