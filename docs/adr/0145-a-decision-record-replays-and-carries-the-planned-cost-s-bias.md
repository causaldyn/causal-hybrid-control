# ADR 0145 — A decision record replays, and carries the planned cost's bias

**Status:** accepted, 2026-10-27. Experimental. Adds keys to the decision record of ADR 0055 and
ADR 0130, a field to the certificate for ADR 0088's bias, and a record of `PlanEvaluation`.
Amended 2026-10-27, before its release: `inputs` holds the panel's declared cluster
(`panel_cluster`), which moves the channel's error and is not in `data_sha256`, and a replay on a
panel that declares another is refused naming it; `prescribe` reads each collection it is handed
once, so neighbours given as iterators are recorded and fitted alike; a record whose input does not
read as `prescribe`'s argument, or that `prescribe` refuses, raises a `DecisionError` that opens
with `inputs.<key>`, or `inputs` where the refusal names no argument; and `PlanEvaluation`'s
`settings` name the data: `panel_sha256` from `Prescription.evaluate`, `logs_sha256` (the float64
bytes and shapes of `"x"` and `"u"`) from `evaluate_plan`, with `schema_version` still 1.

## Context

`Prescription.to_json` records the schedule, both certificate axes, the greedy selection, the
drivers, the budgets, the start, the panel's provenance and the run's settings. It does not record
what the decision was made from: the graph, the levers' boxes, costs and caps, the constraints,
the target, every setting of the solve and the tube, `prescribe`'s seed, the known model. A
reader can audit the numbers and cannot make them again. Three more gaps sit beside it:

- **The graph's edges have no source.** A `CausalGraph` holds edges, not where each came from: a
  caller's claim, a test's verdict with its p-value, or a known mechanism. An audit of a decision
  needs to know which of its edges the data were asked about.
- **The logger check and the dither check stay off the record.** `Prescription.logger_check` and
  `dither_check` are reported and logged, and `to_json` writes neither.
- **The planned cost's bias is not on the certificate.** ADR 0088 built
  `CausalPlan.planned_cost_bias` and `chc.misspecification.planned_cost_bias(plan, fit)`, and left
  the certificate's field for its own change: `prescribe` fits without `influence=True`, whose rows
  cost `N n p` floats, and keeping them compiles another program, whose rounding moved the
  channel's error by an ulp on jax 0.4.30. An evaluation (`PlanEvaluation`) has no record at all.

## Decision

- **`to_json` writes `inputs`**: every argument of `prescribe` that moves its result, as plain
  JSON, and the `data_sha256` of the panel as it was handed in, before `clusters` groups it. The
  graph is written with its nodes, its latent nodes, its lags and each edge with its source. A
  known model is any pytree of arrays and code, which JSON does not carry; it is recorded by a
  sha256 of its structure's text and its arrays' dtypes, shapes and bytes, addresses dropped. In
  `inputs` an infinite number is the text `"inf"` or `"-inf"`, so that it stays apart from None,
  which there means not given: `tolerance=None` switches the tube off, `tolerance=inf` does not.
- **`Prescription.from_json(record, panel, *, model=None)`** passes `inputs` to `prescribe` again
  and checks that the new record is the old one, every key and every number. It refuses, with a
  `DecisionError` whose message starts with the field's path:
  - a `schema_version` other than 2, or a record without `inputs` (every record before 0.24.0);
  - a record written by another release of the library (`run.versions`), or under another
    `jax_enable_x64` (`run.x64`);
  - a panel of other data (`inputs.panel_sha256`);
  - a known model missing, of another digest, or given where the record names none
    (`inputs.known`);
  - a field the replay needs that the record does not hold (`inputs.<key>`);
  - and a replay that does not give the record back, naming the first field that differs and the
    versions in `run.versions` (jax, jaxlib, numpy, scipy) that differ from the record's.
  **The rule:** a record replays on the release that wrote it, under its precision. Its other
  dependencies and its device are not asked first: where one differs and moves no number the
  replay stands, and where it moves one the check names it. A replay that returns is the record.
- **`CausalGraph.from_edges(..., provenance=)`** takes an `EdgeProvenance` per edge: `asserted`,
  `discovered` with the test's name and its p-value, or `mechanism`. An edge not named is
  asserted, as every edge was. The field is not part of the graph's equality or hash: two graphs
  with one set of edges answer every path question alike, and no existing comparison moves.
- **`to_json` writes `logger_check` and `dither_check`**: the logger check's statistic, p-value,
  each pair's partial correlation and the one it could detect, a nan correlation or an infinite
  detectable one None; the dither check's draw, status, share, relevances and its contrast's cost,
  error, noise, p-value and unseen directions. None where the check was not run.
- **`prescribe(planned_cost_bias=True)`** fits the channel a second time, keeping its influence,
  and reads `chc.misspecification.planned_cost_bias` on the plan: the certificate's
  `planned_cost_bias` holds the planned cost, the three terms, and through `PlannedCostBias` the
  bias, its direction and the corrected cost. Where it is not asked, where no plan was made, or
  where the function refuses (a plan held under the barrier, one at a bound of its box or a row,
  one made against drivers, a channel that is not identified), the field is None and
  `planned_cost_bias_reason` says which, in the refusal's own words. `report()` prints the bias
  where it was read, and the refusal where it was asked and refused. The second fit is the dither
  check's: asked for both, the channel is fitted again once.
- **`PlanEvaluation.to_json()`** writes the estimate, its interval and its nominal `level` 0.95,
  the model's correction and share, the effective samples, the degrees of freedom, the
  normalisation's bias, the certificate's every field, the comparison with the logger and the
  bootstrap where present, the logger check, and `settings`: the entry point (`evaluate_plan` or
  `Prescription.evaluate`) and its keywords. Its own `schema_version`, `EVALUATION_SCHEMA_VERSION`
  1. No `from_json`: an evaluation holds a `GcmTest`'s arrays, whose infinite and nan entries the
  record writes alike as None, and a reader that rebuilt one would hold a guess.
- **`SCHEMA_VERSION` stays 2.** It moves when a field changes meaning (ADR 0055), and when a key
  leaves (ADR 0130). Every key 0.23.0 wrote is written as before, with the same value; the new
  ones are new keys. ADR 0130's move to 3 at 1.0 is unchanged.

## Consequences

- The three shapes of the tests replay to the bit through JSON text, with every argument off its
  default in one of them: the quickstart's panel (graph, constraint, tube, Euler, the bias read);
  two levers with caps, a budget, a target schedule, a logged draw, named neighbours, `rk4`, the
  relative radius, `tube_actions="plan"`, three folds, seed 5, four knots and a given start; and a
  held constraint at `gamma = 1.2` under the absolute radius, on a graph with a discovered and a
  mechanism edge, with a known model, a driver, clusters and `max_levers=1`. A record whose
  `gamma`, seed, folds, knots, start, target weight, cap or budget is changed does not replay.
- At the quickstart the certificate reads planned cost 2.15002, optimism 8.15e-4, curvature
  -7.87e-4, bias +2.84e-5, optimistic, corrected cost 2.15005 (x64), bit for bit the misspecification
  function's on the refit. On one scalar step with the drift's rows of the influence set to zero,
  the bias is `validation/planned_cost_bias.mac`'s closed form at the fitted free response, gain
  and gain variance, to the solver's tolerance.
- Every key and value 0.23.0 wrote, `report()`'s text and the plans' actions are unchanged, by
  sha256 against 0.23.0's tree, on six prescriptions (the three above, the quickstart's latent
  case, a one-lever market) with the bias not asked, and so are two evaluations, one by each entry
  point.
- A record now holds the graph, the boxes and every setting; a reader that hashed the whole record
  sees a new digest, as ADR 0130 says a reader of the schema should expect at 1.0.
- `Prescription.from_json` costs a whole `prescribe`. The check compares JSON values, so a replay
  on a machine whose kernels round one number otherwise refuses rather than returns a near copy.

## Alternatives

- **Serialise the known model.** Rejected: a `Dynamics` is code as well as arrays, and a model
  read back from JSON would be whatever the reader builds, not what the decision used.
- **Bump `SCHEMA_VERSION` to 3 now.** Rejected: no key's meaning moved, and ADR 0130 ties 3 to the
  removal at 1.0, so a reader sees one move, not two.
- **Refuse any version in `run.versions` that differs.** Rejected: most version moves do not move
  these numbers, and the check after the replay names the ones that did.
- **Return the replay without checking it.** Rejected: a replay that silently differs is the
  failure the record exists to rule out.
- **Read the bias by default.** Rejected for now: it fits the channel a second time on every
  prescription, and the default moves nothing in 0.24.0.
- **The bias by a parametric bootstrap.** ADR 0088's alternative; one solve per draw, against one
  Hessian.
