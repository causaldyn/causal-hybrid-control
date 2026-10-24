# ADR 0130 — Three more names say what they hold

**Status:** accepted, 2026-10-24. Extends ADR 0128's renames, and dates the end of the decision
record's `identification_radius` key.

## Context

ADR 0128 renamed three names that said more than their values. Three more said more, in
`chc.research`:

- **`confounding_robust_lq_regret`** returns `L_reg (eps + Delta)^2`, the regret to second order.
  Its docstring says so: the exact worst regret over the effect interval exceeds it by 0.9 % at
  `Gamma = 1` and by 5 % to 8 % at `Gamma` from 1.5 to 5. `confounding_robust_lq_regret_matrix`,
  its Frobenius lift `tr(E'HE)`, is the same second-order term. 0.20.0 renamed the curve's field
  that holds the swept values, `ConfoundingRobustLQRegretCurve.second_order_regrets`; the two
  functions that compute them still read as the regret itself.
- **`MultivariateActionFloor.floor`** is `tr(M Psi' G^-1 Psi')` with `Psi'` and `M` read at the
  plant supplied. Van Trees bounds the Bayes risk through the sensitivity averaged over the prior,
  so where `Psi'` and `M` move over the prior a Bayes action goes below the value: at `b = 0`,
  with a prior of SD 1 and five unit-information observations, the floor is 0.333 and the
  posterior's Bayes action regrets 0.075. The name reads as a floor under every estimator.
- **The row `predictive`** of `PricingTask` and `InventoryTask` is the least-squares fit that
  leaves the logged confounder out, beside `predictive-all`, the fit on every logged column. Both
  rows are predictive fits. The label does not say what the row leaves out, and the README, the
  tutorials and the bench's leaderboard (`pricing/predictive`) print it as the predictive
  controller.

`chc.research` is a stability tier of its own (ADR 0117): any of its modules may change in any
release. ADR 0128 still kept each old name until 1.0 with a warning, and this release keeps that
courtesy where a warning is possible.

ADR 0128 also left one key in the decision record: `DecisionCertificate.to_json` writes
`identification_radius` beside `channel_error`, repeating it, so that a reader of a stored record
keeps working. A reader of a JSON key cannot be warned, and nothing said when the key stops.

## Decision

- **The two regret functions are renamed `confounding_robust_lq_second_order_regret` and
  `confounding_robust_lq_second_order_regret_matrix`.** Each old name is a function that warns and
  calls the new one, as `highprob_regret_certificate` does:
  `confounding_robust_lq_regret is renamed confounding_robust_lq_second_order_regret and leaves in
  1.0: it is the regret to second order, not a bound on the worst regret`.
  `confounding_robust_lq_regret_certificate` and `ConfoundingRobustLQRegretCurve` keep their
  names: the certificate checks the second-order term, and the curve's field already says it.
  - `chc.research.sensitivity` lists the new names in its `__all__`, and still imports the old
    ones outside it.
  - The top level keeps the two old names among the 166 research names it serves until 1.0
    (ADR 0117): `chc.confounding_robust_lq_regret` warns that it leaves the top level, and returns
    the old function, which warns again when it is called. The new names do not join the top
    level. Since 0.19.0 a research name lives in `chc.research` alone, as
    `local_highprob_regret_certificate` does.
- **`MultivariateActionFloor.floor` is renamed `local_floor`.** The old name is a property that
  warns and returns it: `MultivariateActionFloor.floor is renamed local_floor and leaves in 1.0:
  it is read at the plant supplied, and a Bayes action can go below it`.
- **The row `predictive` is renamed `predictive-omitted`, with no alias.** A dictionary key cannot
  warn, and `chc.research` may change in any release, so reading `predictive` from a task's
  results is a `KeyError`. `leaderboard` and `leaderboard_multiseed` widen their name column to
  the longest name, 14 at the least, so that the 18 letters of `predictive-omitted` do not push
  its numbers out from under their headings; a table whose names fit in 14 prints as before.
- **pytest turns the new warnings into errors**, under ADR 0128's filter: the library, its tests,
  notebooks and docs use the new names.
- **At 1.0, `to_json` stops writing `identification_radius`, and `SCHEMA_VERSION` moves from 2 to
  3**, in the release that removes the other renamed names. `SCHEMA_VERSION` moves when a field
  changes meaning; a key that leaves changes what a reader of it finds, so its removal moves the
  version too. Until 1.0 the record writes both keys. This release changes no code for it.

## Consequences

- A caller of an old function, or of `floor`, is warned at the line that reads it, and the warning
  names the new one. A caller that builds a `MultivariateActionFloor` by keyword passes
  `local_floor=`: the constructor does not take `floor=`. The library builds it, and nothing in it,
  its tests or notebooks built one by keyword. An object pickled before 0.21.0 unpickles, but
  holds `floor` alone, and reads under neither name.
- A reader of a task's results by the key `predictive` fails at once. The bench's Track D labels
  each row `f"{task}/{r.controller}"`, so it prints `pricing/predictive-omitted` and
  `inventory/predictive-omitted` from 0.21.0 on, and its stored leaderboards keep the old label
  until they are run again.
- The pricing and inventory tables print four characters wider. Their numbers do not move: the
  README's one result reads 4.59 and 13740.08 in float32, 4.61 and 13736.84 under
  `JAX_ENABLE_X64=1`, and `13734.15 [13732.48, 13735.81]` over 12 seeds, as before.
- The library reads no stored decision record. The one reader it has, `PrescribedPolicy.from_json`,
  reads the policy record (`POLICY_SCHEMA_VERSION` 1), which carries no certificate. A reader
  outside the library that takes `certificate.identification_radius` keeps working until 1.0 and
  takes `channel_error` from then; one that checks `schema_version == 2` sees 3 at 1.0, and can
  refuse the record rather than find the key missing. A digest over the whole record, such as
  `scripts/bench_max_levers.py fingerprint` takes, changes at that release.

## Alternatives considered

- **`confounding_robust_lq_leading_regret`**, or `..._regret_approximation`. Rejected for one term
  per concept: the curve's field that holds these values is `second_order_regrets`.
- **Keep `predictive` as a fifth row that repeats `predictive-omitted`.** Rejected: every table
  would print the row twice, `run_multiseed` would aggregate both, and the old label would still
  say less than the row.
- **A results mapping that warns on the old key.** Rejected: a task returns a list of
  `TaskResult`, and each caller builds its own mapping by `controller`.
- **Delete the old regret names and `floor` now**, since `chc.research` may change in any release.
  Rejected: ADR 0128's research renames kept theirs until 1.0, and the courtesy costs two short
  functions and a property.
- **A fixed name column of 18.** Rejected: it would widen every task's table, where the column
  that follows the longest name widens only the tables that need it.
- **Stop writing `identification_radius` now, or move the schema version now.** Rejected: a reader
  of a stored record cannot be warned, and `to_json`'s docstring promises the key until 1.0, the
  release in which every renamed name leaves together.
