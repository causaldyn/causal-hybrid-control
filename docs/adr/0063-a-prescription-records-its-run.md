# ADR 0063 — A prescription records its run

**Status:** accepted, 2026-10-07.

## Context

`Prescription.provenance` is the panel's `Provenance`. It is made when the panel is built: the
data's hash, the library's version, and `x64`, JAX's double-precision flag at that moment. Nothing
recorded the run itself.

A review of 0.13.0 found the gap: `x64` recorded as True beside a fit on float32 transitions. The
panel was built with the flag on, and the run had it off. On the tests' world of 40 units, the same
panel planned on in float32 moved the schedule by up to 0.017 in a box of width 4. The fitted
channel moved by 4e-7 at most; the solve stopped after 266 accepted steps, where the float64 solve
took 485. The panel's record read True both times.

The rest of the run went unrecorded too:

- the device JAX ran on;
- the fit's seed, integrator, folds, degrees and ridge;
- the planner's budget of steps, the tube's tolerance, whether the constraints were held, and
  `max_levers`;
- the versions of the packages that compute the numbers: jax, jaxlib, numpy and scipy.

Why the planner stopped, and after how many steps, was on the certificate alone. A record of the
run without them cannot tell a solve that converged from one its budget stopped.

The report's `seed` was the panel's: the generator's seed for simulated data, `None` for observed
data. The fit's seed, which draws the cross-fitting folds, was not printed.

The `precision` warning read the panel's flag as well. A panel built in float64 and planned on in
float32 was not warned about. A panel built in float32 and planned on in float64 was.

## Decision

- **`Prescription.run` is a `RunProvenance`**, a frozen record beside `provenance`. It is `None`
  on a prescription built by hand, as `start` is.
- **Each field is read where the run used it**, so the record cannot disagree with the run:
  - `x64`: `jax_enable_x64` when `prescribe` ran;
  - `backend` and `device_kind`: the platform and the kind of the device the fit's transitions
    sat on, where JAX ran the fit. Not `jax.default_backend()`: inside a `jax.default_device`
    block the fit runs on the block's device, which need not be the default backend's;
  - `fit_dtype`: the dtype of the transitions the fit ran on;
  - `solve_dtype`, `solver_status` and `solver_iterations`: the dtype of the plan's actions, why
    its descent stopped, and its accepted steps. Each is read off the plan the planner returned,
    which `Prescription.plan` holds; under `max_levers`, the plan the selection kept. The
    certificate reads the same status and count off the same plan. All three are `None` where no
    plan was solved; the certificate's count reads 0 there;
  - `seed`, `integrator`, `folds`, `degree`, `channel_degree`, `nuisance_degree` and `ridge`: as
    the fit was given them. `folds`, `integrator` and the two feature degrees are read off the
    fit, which keeps them;
  - `steps`, `tolerance` and `hold_constraints`: as the planner was given them. `tolerance` is
    `None` where no tube was asked for;
  - `max_levers`: the greedy selection's cap;
  - `versions`: causal-hybrid-control, jax, jaxlib, numpy and scipy, each from
    `importlib.metadata.version`, or `"unknown"` for a distribution without metadata. A module's
    `__version__` is a second copy, which can lag: `chc.__version__` sat at 0.3.0 for two
    releases.
- **`prescribe` passes the ridge and the planner's budget by name.** Both were defaults that
  `prescribe` did not pass: 1e-6 for the fit, 10 000 steps for each of the planner's descents.
  `prescribe` now passes the same values, so the record holds what each call was given, and no
  number moves.
- **`to_json` writes the record under `run`**, in strict JSON, as ADR 0055 requires:
  - the versions are an object keyed by distribution;
  - a NumPy integer, such as a seed drawn from an array, is written as an int;
  - an infinite tolerance is `null`, like any number that is not finite. The certificate's status
    says whether a tube was evaluated.
- **The schema version stays 2.** ADR 0055 bumps it when a key changes its meaning. This change
  adds a key, and no key changes its meaning.
- **The report prints the run under the panel.** The panel's line now starts with `panel:`. Three
  lines follow it: the run, its settings and the versions. A tolerance or a cap that was not given
  reads `none`. The stop is printed once, on the certificate's `solver` line.
- **The `precision` warning reads the run's flag**, the one `run.x64` records.

## Consequences

- A record says which precision, device, settings and versions produced it, and how its solve
  ended. A run in float32 on a panel built in float64 reads `x64` False under `run` and True under
  `provenance`.
- The panel's `x64` stays. A redraw of simulated data needs it: the same seed draws a different
  sample at the two settings (`docs/concepts/dtype-policy.md`).
- The fit's seed draws the same folds at both settings: the permutations of 24, 280, 2200 and
  100 000 rows were the same. What `x64` changes in a fit is its arithmetic.
- *Left*:
  - The decision's own terms are not all recorded: the target's level and weight, the
    constraints, `dt` and the known physics.
  - JAX's matmul precision and the versions of equinox, optax and diffrax are not recorded.
  - The logger check's degree and the regret bound's number of probes are fixed in the code, and
    are not recorded.
  - Under `max_levers`, the stop recorded is the kept plan's. The candidates' solves are counted
    only in the `selection` log events.
  - The versions name the installed distributions. A source tree imported ahead of the installed
    one, from a `PYTHONPATH`, is not what they name; `Provenance.chc_version` has the same limit.
- *Tests*:
  - each field is what the fit and the planner were given, or read off what they ran on and
    returned, each call bound to its signature with its defaults, in float64 and in float32;
  - each setting moves its own field and no other: the folds, a NumPy seed, the integrator, no
    tube, held constraints, `max_levers`, the ridge, the steps and the nuisances' degree;
  - a fit made at other feature degrees is recorded at those;
  - a solve on a budget of 7 steps is recorded as stopped by it, after 7, where the default
    budget's solve converged;
  - the device is the transitions', with JAX made to name a GPU as its default;
  - the versions are those `importlib.metadata` reads;
  - the record survives `json.dumps(..., allow_nan=False)` with an infinite tolerance and in each
    state ADR 0055 lists, with no solve's fields where no plan was solved;
  - the report prints the run, and no `None`;
  - the `precision` warning fires on a run in float32, and not on a run in float64 of a panel
    built in float32.
- *Mutations*: each of 37 mutations of the record fails a test. Among them: `x64`, the fit's
  dtype or the `precision` warning read off the panel's flag; the backend or the device's kind
  read off JAX's default; the solve's dtype, status or steps left unset, or the dtype or the steps
  set where no plan was solved; the status read as `converged`, and the steps as the budget,
  whatever the solve did; the seed read off the panel, or a NumPy seed kept as it came; the folds,
  the integrator or a degree recorded as a constant or as another setting; the ridge or the budget
  recorded but not passed; no tube recorded as an infinite one; held constraints, the cap or a
  version dropped; the record or a key of it left out of `to_json()`, its versions written as
  pairs, or its strict pass dropped; the run left out of the report, its line reading the panel's
  `x64`, or a `None` printed; and an unidentified prescription left without its record.

## Alternatives

- **Overwrite the panel's `x64` with the run's.** Rejected: a redraw of the panel's data needs the
  panel's flag, and one field cannot hold two moments.
- **One record for the data and the run.** Rejected: a panel is built once and planned on many
  times, and its record names the data.
- **Read the ridge and the steps off the callees' signatures**, and leave them unpassed. Rejected:
  the call and the record would then each read the value, at two places. Passed by name, one value
  serves both.
- **The backend from `jax.default_backend()`.** Rejected: it names the default backend, and
  inside a `jax.default_device` block the fit runs on the block's device.
- **The stop on the certificate alone.** Rejected: the record of the run would not say how its
  solve ended. The two read the same plan, so they cannot disagree.
- **Versions from `__version__`.** Rejected, as above.
- **A new schema version.** Rejected, by ADR 0055's rule.
- **A status beside the tolerance**, as ADR 0055 asks of an output that can be infinite. Not
  taken: the tolerance is an input, and an infinite tolerance and none part only in whether a tube
  was evaluated, which the certificate's status already says.
