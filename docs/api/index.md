# API reference

One page per public module, rendered from the module's docstrings. Every name is importable from
its module, which is where it stays; the modules that each carry one research result moved into
[`chc.research`](research/index.md) in 0.19.0, and 0.20.0 deleted their old paths. The top-level
`chc` namespace holds the
[lifecycle's](../lifecycle.md) names, the classes a call to one of them takes and the errors they
raise, and its `__all__` lists them. The other names 0.7.0 bound there left it in 1.0: each imports
from its module.

The pages are grouped by stability tier — by what a break would cost you — and each page states its
tier at the top.

## What 1.x promises

From 1.0 the stable tier follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html): a
change that breaks it waits for the next major and gets a deprecation cycle first. The other tiers
promise what the table below says. What applies to every release, and what you can plan against:

- **A minor release may change behaviour outside the stable tier, and the changelog says
  which.** The [changelog](https://github.com/causaldyn/causal-hybrid-control/blob/main/CHANGELOG.md) is written
  to be read before upgrading — it carries scope corrections and retractions alongside the
  additions, because a number that quietly changed meaning is worse than one that broke loudly.
  0.5.0 moved the marketing-mix headline figures by fixing the integrator the fit used; that is the
  kind of thing it records.
- **A patch release adds no feature and changes no signature, and it changes a number only to keep
  a promise these docs make.** Where a result breaks a contract stated here, a patch may fix it, and
  the changelog then gives the counterexample and what to change on upgrading. New models and
  deliberate changes of a default wait for a minor. 0.5.1 fixed a wrong `__version__` and nothing
  else.
- **Renames get one minor of alias.** Removals get one minor of `DeprecationWarning` first, naming
  the replacement in the message. In the stable tier the old name stays until the next major.

## The tiers

Four tiers, by what a break costs you:

| tier | modules | promise |
|---|---|---|
| **stable** | `dynamics` `integrate` `cost` `control` `plan` `barrier` `residual` `lqr` `mpc` `train` `adjoint` `decision` `panel` `graph` `dynamics_id` | the plant/control spine and the façade over it. Breaking changes wait for the next major and get a deprecation cycle |
| **evolving** | the estimator, certificate and domain layers — `causal` `uncertainty` `regret` `irf` `did` `scm` `matching` `marketplace` and their neighbours | may gain keyword arguments in a minor; defaults may change with a changelog entry arguing why |
| **experimental** | modules that exist to carry one research result — `games` `discovery` — and modules built ahead of the release that verifies them — `experiment` `switchback` on a plant CHC did not write, `zones` in the pre-registered study it is the plant of, `dlm`, `lift` and `allocation` on an external media-mix generator, `misspecification` on the zone plant, `response` in the plans it warm-starts, `geo_world` in the geo models scored on it | may change or be withdrawn in any release. Pin an exact version if you depend on one |
| **research** | [`chc.research`](research/index.md): the modules that each carry one research result — `benchmark` `causal_bench` `deep_galerkin` `epidemic` `flagship` `galerkin` `koopman` `lalonde` `meanfield` `mintime` `mmm` `sensitivity` `spine` `surrogate` `symbolic` `transport` — and the research halves of `regret`, `games` and `residual` | outside the surface 0.20.0 freezes: may change or be withdrawn in any release, and `import chc` imports none of it. Pin an exact version if you depend on one |

## Stable

The plant/control spine and the façade over it. Breaking changes wait for the next major and get a
deprecation cycle.

## Evolving

The estimator, certificate and domain layers. They may gain keyword arguments in a minor; defaults
may change with a changelog entry arguing why.

The table above names the stable and the experimental modules one by one, and this tier as
"`causal` `uncertainty` `regret` `irf` `did` `scm` `matching` `marketplace` and their neighbours",
so every module of `chc` it does not name is listed here as evolving.

## Experimental

Modules that exist to carry one research result, and modules built ahead of the release that
verifies them: on a plant CHC did not write, or, for the marketplace plant `zones`, in the
pre-registered study it is the plant of, for the discount model `dlm`, the lift fit `lift` and the
budget's split `allocation` on an external media-mix generator, for the gate `misspecification`
on the zone plant, for the response curves `response` in the plans they warm-start, and for
the geo world `geo_world` in the geo models scored on it. They may change or be withdrawn in
any release. Pin an exact version if you depend on one.

87 entries in modules of other tiers are experimental on the same terms, for the same
reason. An entry is a name, `Class.member` for a field or a method, `function(argument=)` for an
argument, or a value or key in quotes:

- `chc.dynamics_id`: `fit_causal_residual(weights=)`, `solve_channel_moment(weights=)`,
  `fit_causal_residual(influence=)`, `CausalDynamicsFit.influence`, `CausalDynamicsFit.representer`,
  `nuisance_shift`, `CausalDynamicsFit.unmoved`, `CausalDynamicsFit.moment_residual`,
  `CausalDynamicsFit.instrument_relevance`, `CausalDynamicsFit.instrument_rank`,
  `omitted_confounder_bound`, `OmittedConfounderBound`, `persistence_check`, `PersistenceCheck`,
  `interference_check`, `InterferenceCheck`, `lag_check`, `LagCheck`, `channel_region`,
  `ChannelRegion`, `RegionForm`;
- `chc.uncertainty`: `horizon_errors`, `HorizonConformal`, `mixing_penalty`, `gaussian_mixing`;
- `chc.decision`: `prescribe(tube_actions=)`, `prescribe(dither=)`, `prescribe(neighbours=)`,
  `prescribe(planned_cost_bias=)`, `DecisionCertificate.noise_persistence`,
  `DecisionCertificate.noise_persistence_p`, `DecisionCertificate.tube_per_log_spread`,
  `DecisionCertificate.tube_actions`, `DecisionCertificate.planned_cost_bias`,
  `DecisionCertificate.planned_cost_bias_reason`, `TubeActions`, `DitherCheck`, `DitherStatus`,
  `PrescribedPolicy`, `Prescription.logger_check`, `Prescription.interference`,
  `Prescription.dither_check`, `Prescription.from_json`;
- `chc.did`: `callaway_santanna_inference`, `EventStudyInference`;
- `chc.scm`: `synthetic_control_inference`, `SyntheticControlInference`, `blank_period_inference`,
  `BlankPeriodInference`;
- `chc.independence`: `gcm_test`, `GcmTest`;
- `chc.evaluation`: `LoggerCheck`, `PlanEvaluation.logger_check`, `PlanEvaluation.to_json`;
- `chc.matching`: `shadow_price_effect`, `shadow_price_interval`;
- `chc.marketplace`: `dither_check`, `GradientDitherCheck`, `dither_size`, `DitherSize`,
  `SharedStateMarket(payment="per_driver")`;
- `chc.gate`: `channel_drift_evalues`, `DriftAlarm`, `MissRateAlarm`, `OnlineRadius`,
  `channel_move`, `ChannelMove`, `MovePrice`, `BlockBatch`, `GateConfig.alpha_futility`,
  `GateConfig.block_length`, `GateConfig.burn_in`, `GateConfig.check_burn_in`, `GateConfig.centred`,
  `GateConfig.rollback_baseline_share`;
- `chc.estimators`: `IV2SLS.estimate["instrument_relevance"]`;
- `chc.regret`: `minimax_regret_move`, `MinimaxRegretMove`, `MoveRule`;
- `chc.plan`: `CausalPlan.decision_weight`, `DecisionWeight`, `CausalPlan.relaxed_cost`,
  `CausalPlan.planned_cost_bias`, `PlannedCostBias`, `CostBiasDirection`, `causal_plan(inner=)`;
- `chc.mpc`: `RecedingHorizon.inner`.

## Research

The modules of [`chc.research`](research/index.md): each carries one research result — a solver, a
certificate, a benchmark or a case study's plant — rather than a stage of a decision taken from
logs, and the research halves of `regret`, `games` and `residual` sit beside them. They are outside
the surface 0.20.0 freezes: any of them may change or be withdrawn in any release, and `import chc`
imports none of them. The paths they had before 0.19.0 were deleted in 0.20.0: import each name
from its path in `chc.research`.

The certificates that each back one research result inside modules of other tiers are on the same
terms, and stay at their paths for now: the rollout, risk, ensemble and confounding certificates of
`chc.uncertainty`, `barrier_confounding_certificate` and `safety_filter_benchmark` in
`chc.barrier`, `delay_margin_certificate` and `delay_ball_certificate` in `chc.delay`,
`barrier_reachability_gap` and `higher_order_barrier_gap` in `chc.reachability`, and
`linearized_regret_certificate` in `chc.lqr`.

## Roadmap

0.20.0 deleted the paths the research modules left in 0.19.0. In 1.0.0 the stable tier stopped
moving, and the top level holds only the lifecycle's names: the 421 names that warned since 0.8.0
left it. Next, the marketplace study, and the adaptation modules leave the experimental tier.
