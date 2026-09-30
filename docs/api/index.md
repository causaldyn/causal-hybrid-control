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
- **A patch release changes no signature and no number.**
- **Renames get one minor of alias.** Removals get one minor of `DeprecationWarning` first, naming
  the replacement in the message.

## Stable

The plant/control spine and the façade over it. Breaking changes wait for 1.0 and get a
deprecation cycle.

## Evolving

The estimator, certificate and domain layers. They may gain keyword arguments in a minor; defaults
may change with a changelog entry arguing why.

The README's tier table names the stable and the experimental modules one by one, and this tier as
"`causal` `sensitivity` `uncertainty` `regret` `spine` `irf` `did` `scm` `matching` `marketplace`
`mmm` and their neighbours", so every module it does not name is listed here as evolving.

## Experimental

Modules that exist to carry one research result, and modules built ahead of the release that
verifies them: on a plant CHC did not write, or, for the marketplace plant `zones`, in the
pre-registered study it is the plant of, for the discount model `dlm` on an external media-mix
generator, for the gate `misspecification` on the zone plant, and for the response curves
`response` in the plans they warm-start. They may change or be withdrawn in any release. Pin an
exact version if you depend on one.

Twenty-one entries in modules of other tiers are experimental on the same terms, for the same
reason:
the `weights` argument of `fit_causal_residual` and `solve_channel_moment`, its `influence` argument
with the field of that name, and the `unmoved` field of `CausalDynamicsFit`,
`callaway_santanna_inference` and `EventStudyInference` in `chc.did`, `synthetic_control_inference`
and `SyntheticControlInference` in `chc.scm`, `gcm_test` and `GcmTest` in `chc.independence`,
`LoggerCheck` in `chc.evaluation`, the `logger_check` field of `Prescription` and `PlanEvaluation`,
`shadow_price_effect` and `shadow_price_interval` in `chc.matching`, `channel_drift_evalues`,
`DriftAlarm`, `channel_move`, `ChannelMove` and `MovePrice` in `chc.gate`, and
`CausalPlan.decision_weight`, `DecisionWeight` and `CausalPlan.relaxed_cost` in `chc.plan`.
