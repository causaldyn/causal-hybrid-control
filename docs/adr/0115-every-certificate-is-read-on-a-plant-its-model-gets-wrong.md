# ADR 0115 — Every certificate is read on a plant its model gets wrong

**Status:** accepted, 2026-10-19.

## Context

The library's certificates were tested where their premises hold: independent noise, an open-loop
lever, nuisances inside the polynomial the fit reads them by, actions inside the logged range, a
plant of the model's own class. No test read a certificate on a plant other than its model. Where
a premise fails, a certificate can still read as valid. Three cases measured here:

- **The tube.** It states 0.95 at each step, with the plant's noise a Gaussian AR(1). Under a
  confounder that persists, AR(0.8) within a unit and adjusted for, the plant left the tube more
  often than stated: coverage was 0.87 at step 4 and 0.77 at step 15. With the noise carried as a
  VAR(1) (ADR 0105) it is 0.942 at its weakest step, where the case with no confounder reads 0.944.
- **The channel's interval.** With the drift and the logging rule outside the fit's polynomial
  sieve, the channel read 0.42 low, about 8 standard errors, and the interval covered on 0 of 40
  logs. The default fit now warns on all 40, and `nuisance_knots="auto"` covers on 38 (ADR 0107).
- **The `"mis"` off-policy interval.** It did not carry the model's error. With the model's
  `A[0, 0]` off by 0.1, it covered on 0 of 40 logs; reaching to `"dr"`'s interval, on 40
  (ADR 0108).

## Decision

- `tests/misspecification/` holds one test per certificate and failed premise. The certificates
  are the tube and its trustworthy steps, the barrier's certified steps and `gamma_star`, the plan's
  regret bound, the channel's standard error, the off-policy intervals, the lift check, and the
  deployment gate's DEPLOY. The premises are a persistent confounder, AR noise, dead time,
  nuisances outside the sieve, actions outside the logged support, and a plant unlike the model.
- Each test asserts one contract. The certificate holds at its stated level within three Monte
  Carlo standard deviations, or it says that the premise fails. A refusal counts as saying so, and
  so does a `WARNING` record, a warning the call gives where the premise fails on the log in hand,
  or a field the certificate reports for that premise, such as the persistence check's p-value or
  `SafetyCertificate.radius_model`. A warning counts on the log it fires on; where a test pools
  its coverage over logs, it counts where it fires on 90 % of them.
- A case that does neither is marked `xfail(strict=True)`. Its reason gives the numbers measured.
  When a fix makes the case hold, the test passes, strict turns that into a failure, and the job
  stays red until the mark is removed.
- The tests carry the marker `misspecification`. `pyproject.toml`'s `addopts` deselects it, so
  every leg of `test` skips the suite. A CI job of its own runs it once, on Python 3.14 and two
  workers. Locally, `just misspecification` runs it.
- The Monte Carlo is seeded and fast: a few logs or worlds per case, read on thousands of
  deployments where the deployments are cheap.

## Consequences

- A change that weakens a certificate on a misspecified plant fails CI. Before, only a change on a
  well-specified plant did.
- The strict xfails are the list of certificates that neither hold nor say so. Each has a measured
  reason, and each stays in the list until a fix lands. At 0.19.0 one remains: the tube under a
  persistent confounder, read over 16 logs: 0.944 by step 13 against a floor of 0.9488, with a
  warning on 1 of 16 logs. The suite reads 33 passed and 1 xfailed.
- Saying so can come late. The gate's carryover warning is counted over the path's whole run, and
  on about four paths in ten it comes after the DEPLOY it would have stopped.
- A pinned contract is a test's own reading of a premise. When a fix redefines a certificate, for
  example the barrier's sensitivity set, the test that reads that certificate is revisited with it.
- The suite takes a few minutes on two workers, and it runs on one Python version only.

## Alternatives

- **Run the suite on every leg.** Rejected: it would add its minutes to each of seven legs. The
  certificates' arithmetic is the same on every version, and the legs already test it.
- **Plain `xfail`, or `skip`.** Rejected: a non-strict xfail lets a fix pass unnoticed, and a skip
  hides the case.
- **A nightly job.** Rejected: it reports a weakened certificate after the merge, not on the pull
  request that weakened it.
