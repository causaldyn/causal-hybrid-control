# The decision lifecycle

A decision taken from logs goes through six stages:

1. **Identify** what the logs can say about the levers.
2. **Plan** under the limits the business states.
3. **Evaluate** the plan before acting on it.
4. **Experiment** where the logs cannot say enough.
5. **Deploy** one step at a time.
6. **Adapt** when the world moves.

This page files the library by stage. For each stage it lists what is built, where it is
documented, and what is not built yet. Every name below is importable from `chc`.

## 1. Identify: what the logs can say about the levers

- **`Panel` and `prescribe`'s adjustment set.** The causal assumption is a required argument. A
  graph that does not identify the effect raises `NotIdentifiedError` instead of returning a
  schedule. See the [quickstart](quickstart.md) and [identification](concepts/identification.md).
- **`fit_causal_residual`.** It fits the lever's channel in a hybrid model by cross-fitted
  orthogonalisation, or through an instrument when the confounder is latent. When the channel
  class cannot fit every state, `weights` points it at the states a one-shot decision will be taken
  at. See [`chc.dynamics_id`](api/dynamics_id.md).
- **Static effects.** `estimate_effect_dml`, `sensitivity_analysis`, `refute_effect` and
  `e_value`. See [tutorial 3](tutorials/03_causal_inference_toolkit.md), and
  [tutorial 7](tutorials/07_real_data_lalonde.md) against an experimental benchmark.
- **Designs with a comparison group.** Difference-in-differences, in [`chc.did`](api/did.md), and
  synthetic control, in [`chc.scm`](api/scm.md).
- **Effects that take time.** Impulse responses and `delay_estimate`, in
  [`chc.irf`](api/irf.md).
- **No adjustment set at all.** The sensitivity level [Γ](concepts/gamma.md) bounds the effect
  instead of pointing at it; see
  [tutorial 5, robust control](tutorials/05_confounding_robust_control.md).

## 2. Plan: the actions, under what the business states

- **`prescribe`.** From a panel to a schedule with its certificate, in one call. It takes:
  - `Lever`: a box, a limit on how far it moves in one step, and a quadratic price on its use;
  - `Target`: a level, or a schedule of levels;
  - `Constraint`: a bound on a state;
  - `Driver`: a forecast the plan cannot move.

  See the [quickstart](quickstart.md) and [`chc.decision`](api/decision.md).
- **`causal_plan`, the solver underneath.** It takes the box, linear rows over the whole sequence
  (`LinearConstraint`: a budget, a rate limit), a barrier held in the solve (`BarrierConstraint`)
  and offline [pessimism](concepts/pessimism.md). See [`chc.plan`](api/plan.md).
- **`CausalPlan.shadow_prices()`.** What each row costs the plan: the budget's price, per unit.
- **`minimax_action`.** The robust action over an identified interval, minimising the worst cost
  or the worst regret. See [`chc.regret`](api/regret.md).
- **The case studies**, each one decision from its logs to a plan:
  - [pricing under confounding](case-studies/pricing.md);
  - [an epidemic under hospital capacity](case-studies/epidemic.md);
  - [a pendulum from a confounded log](case-studies/pendulum.md);
  - [media budgets](case-studies/media-budgets.md);
  - [marketplace dispatch](case-studies/marketplace-dispatch.md);
  - [driver supply, end to end](case-studies/driver-supply.md);
  - [confounded incentives, closed loop](case-studies/confounded-incentives.md);
  - [a heat pump on a live building emulator](case-studies/boptest.md).

## 3. Evaluate: before acting

- **The error tube.** `CausalPlan.certificate_status` and `CausalPlan.certified_actions` say how
  far ahead the model's error keeps the plan inside tolerance. See
  [certificates](concepts/certificates.md).
- **`certify_safety`.** Where along a finished plan a barrier's guarantee survives unmeasured
  confounding, and the largest Γ the whole plan tolerates.
- **`plan_regret_bound`.** How far the plan's cost can be from the optimum, certified from its own
  gradient.
- **The reachable tube.** [`chc.reachability`](api/reachability.md) computes the backward
  reachable tube under a partially identified effect.
- **`evaluate_plan` and `certify_evaluation`.** They estimate what deploying a plan would cost,
  from logs of another policy, with an interval. Before any cost is read, the certificate says
  whether the logs can, and refuses by name when a weight the method needs has infinite variance.
  Four methods: per-decision importance sampling over episodes, marginalised importance sampling,
  doubly robust, and fitted Q evaluation, which works where every weight is refused. Every number
  is exact for a linear-Gaussian loop and an affine plan, and no more. See
  [`chc.evaluation`](api/evaluation.md).
- **`off_policy_value`.** It weights one step at a time. So it estimates a candidate's value on
  the logger's own states, which is the contextual-bandit value. That is not the value of
  deploying a plan on a plant with memory. See [`chc.offpolicy`](api/offpolicy.md).

## 4. Experiment: where the logs cannot say enough

- **Exploration.** `optimal_exploration_certificate`, `adaptive_exploration_certificate` and
  `capped_exploration_policy` price how much exploration a controller should inject while its
  effect is not identified, and on what schedule. Each is weighed against the control the
  exploration costs. See [`chc.regret`](api/regret.md).
- **`design_experiment`.** It says how many units to run in each zone before a one-shot decision,
  and whether to run any: the regret the experiment removes, against what it spends. The verdict
  is deploy, experiment or abstain, with the reason. Every number it reports is Monte Carlo's,
  re-solving the decision on every draw; its local model only chooses the units. A channel the
  logs do not identify enters as a wide prior. Scope: a static decision over zones, linear in each
  zone's channel, with a box on the levers. See [`chc.experiment`](api/experiment.md).
- **`design_switchback` and `read_switchback`.** Which switchback to run in a zone whose state
  carries the lever's past, for a named effect: the channel, the effect of holding the lever on
  for `H` periods, or the steady state. The design is aligned to the effect, so the persistence
  estimate is orthogonal to it, and the plan quotes each effect's standard error and minimum
  detectable effect at the least favourable persistence. Without a trusted state model only
  model-free readings are planned. `read_switchback` reads the effect off the data, with a
  standard error from the data. Scope: a first-order state per zone, and zones independent. See
  [`chc.switchback`](api/switchback.md).
- **`shadow_price_effect`.** An experiment on a matching market treats some of the rows, and they
  compete with the control rows for the same columns, so the naive difference is not what treating
  every row would do. It reads that global effect off the rows' rents in the experiment's own
  matching, with a standard error, the second-order bias left at the treated share, which vanishes
  at one half, and an alarm near the LP limit, where that bias stops being second order. Strata
  fixed before the assignment post-stratify it. `shadow_price_interval` is the range the exact LP
  leaves at `eps = 0`. See
  [`chc.matching`](api/matching.md).
- **Not built:** an experiment design for a dynamic plan's whole decision, beyond one lever's
  effect.

## 5. Deploy: one step at a time

- **`RecedingHorizon`.** It runs `causal_plan` from each measured state, warm-started, and returns
  the certificate with every step. `PeriodBudget` caps what each period spends, from a ledger the
  caller measures. See [`chc.mpc`](api/mpc.md).
- **`robust_safety_filter`.** It clips one nominal action into the certified interval, at one
  state. See [`chc.barrier`](api/barrier.md).
- **`DeploymentGate`.** It runs a candidate policy in shadow of the baseline, per zone, and after
  each batch says deploy, hold, experiment or roll back. It can be read at any time: at every
  read, the expected share of wrong deploys across zones is at most `alpha`. That holds when the
  propensities were logged at decision time, zones do not spill over, and a decision's reward does
  not depend on earlier ones. See [`chc.gate`](api/gate.md).

## 6. Adapt: after deployment

- **`SplitConformal`.** It calibrates an ensemble's next-state intervals on held-out logs. They
  cover while the logs and the loop are exchangeable, so a run of steps outside them says the
  model no longer describes the plant. Nothing in the library watches for that run. See
  [`chc.uncertainty`](api/uncertainty.md).
- **Not built:** a monitor that notices the effect has moved, or anything that re-identifies it.
