# API reference

One page per public module, rendered from the module's docstrings. Every name is importable from
its module, which is where it stays. The top-level `chc` namespace holds the
[lifecycle's](../lifecycle.md) names, the classes a call to one of them takes and the errors they
raise, and its `__all__` lists them. Any other name 0.7.0 bound there still imports from it until
1.0, with a `DeprecationWarning` that names its module.

The pages are grouped by stability tier — by what a break would cost you — and each page states its
tier at the top.

## What "0.x" promises

Pre-1.0, so SemVer's major-version protection does not apply yet. What *does* apply, and what you
can plan against:

- **A minor release may change behaviour, and the changelog says which.** The
  [changelog](https://github.com/causaldyn/causal-hybrid-control/blob/main/CHANGELOG.md) is written
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
  the replacement in the message.

## The tiers

Three tiers, by what a break costs you:

| tier | modules | promise |
|---|---|---|
| **stable** | `dynamics` `integrate` `cost` `control` `plan` `barrier` `residual` `lqr` `mpc` `train` `adjoint` `decision` `panel` `graph` `dynamics_id` | the plant/control spine and the façade over it. Breaking changes wait for 1.0 and get a deprecation cycle |
| **evolving** | the estimator, certificate and domain layers — `causal` `sensitivity` `uncertainty` `regret` `spine` `irf` `did` `scm` `matching` `marketplace` and their neighbours — and `mmm`, the media-budget case study's plant rather than a media-mix model | may gain keyword arguments in a minor; defaults may change with a changelog entry arguing why |
| **experimental** | modules that exist to carry one research result — `deep_galerkin` `galerkin` `transport` `meanfield` `games` `epidemic` `discovery` `symbolic` `koopman` `surrogate` `flagship` `benchmark` `causal_bench` `lalonde` `mintime` — and modules built ahead of the release that verifies them — `experiment` `switchback` on a plant CHC did not write, `zones` in the pre-registered study it is the plant of, `dlm`, `lift` and `allocation` on an external media-mix generator, `misspecification` on the zone plant, `response` in the plans it warm-starts, `geo_world` in the geo models scored on it | may change or be withdrawn in any release. Pin an exact version if you depend on one |

## Stable

The plant/control spine and the façade over it. Breaking changes wait for 1.0 and get a
deprecation cycle.

## Evolving

The estimator, certificate and domain layers. They may gain keyword arguments in a minor; defaults
may change with a changelog entry arguing why.

The table above names the stable and the experimental modules one by one, and this tier as
"`causal` `sensitivity` `uncertainty` `regret` `spine` `irf` `did` `scm` `matching` `marketplace`
and their neighbours", so every module it does not name is listed here as evolving. `mmm` is filed
here as a case-study plant: the saturating-carryover plant of the media-budget case study, not a
media-mix model.

## Experimental

Modules that exist to carry one research result, and modules built ahead of the release that
verifies them: on a plant CHC did not write, or, for the marketplace plant `zones`, in the
pre-registered study it is the plant of, for the discount model `dlm`, the lift fit `lift` and the
budget's split `allocation` on an external media-mix generator, for the gate `misspecification`
on the zone plant, for the response curves `response` in the plans they warm-start, and for
the geo world `geo_world` in the geo models scored on it. They may change or be withdrawn in
any release. Pin an exact version if you depend on one.

Twenty-seven entries in modules of other tiers are experimental on the same terms, for the same
reason:
the `weights` argument of `fit_causal_residual` and `solve_channel_moment`, its `influence` argument
with the fields it keeps, the `unmoved`, `instrument_relevance` and `instrument_rank` fields of
`CausalDynamicsFit`, and `omitted_confounder_bound` and `OmittedConfounderBound` in
`chc.dynamics_id`,
`callaway_santanna_inference` and `EventStudyInference` in `chc.did`, `synthetic_control_inference`
and `SyntheticControlInference` in `chc.scm`, `gcm_test` and `GcmTest` in `chc.independence`,
`LoggerCheck` in `chc.evaluation`, the `logger_check` field of `Prescription` and `PlanEvaluation`,
`shadow_price_effect` and `shadow_price_interval` in `chc.matching`, `channel_drift_evalues`,
`DriftAlarm`, `channel_move`, `ChannelMove`, `MovePrice` and the `alpha_futility` field of
`GateConfig` in `chc.gate`, the `instrument_relevance` diagnostic of `IV2SLS` in `chc.estimators`,
and
`CausalPlan.decision_weight`, `DecisionWeight` and `CausalPlan.relaxed_cost` in `chc.plan`.

## Roadmap

Next, the marketplace study, and the adaptation modules leave the experimental tier. Then 1.0.0:
the stable tier stops moving, and the top level holds only the lifecycle's names. The 421 names that
warn since 0.8.0 leave it then.
