# ADR 0115 — Every certificate is read on a plant its model gets wrong

**Status:** accepted, 2026-10-19.

Amended 2026-11-06: the floors were sized for a false red, not for power. Three Monte Carlo
standard deviations on 40 logs let an interval 20 % too narrow, which covers 0.883, clear the floor
of 0.847 with probability 0.82, and a wrong certificate planted in the library passed every test of
its file: the channel's error or an off-policy interval 0.8 times its own, the dither or the
lagged-lever check's p-value a third of its own; an audit that certified every step passed 15 of
the 16 barrier tests. The tests of those four certificates now read 200 logs from a block of
seeds, `CHC_MISSPEC_BLOCK` (0 in CI), and hold each count to the bound a binomial count at the rate
the test states crosses with probability 1/1000 (`tests/misspecification/_misspec_counts.py`,
reproduced in `validation/misspec_power.mac`). A sum of `z^2` held below a chi-square quantile
reads an error where `z` is near normal, and a check's rejections are held at 0.2 as well as at
0.05. Outcomes are counted apart: where a premise fails, "holds or says so" stays the contract;
where the premises hold, the logs refused, warned or rejected on are false alarms, held to the
ceiling of a rate of 0.05, so a certificate that says so on every log fails. Each test's docstring
states its rate, its rate under the defect it names, and the smallest defect it catches with
probability 0.8. Each barrier family also reads a plan its plant leaves the set under, so a count
that runs past the exit fails. `just mutants` plants the wrong certificates of `tests/mutants/` one
at a time over five blocks: each of the four above fails its file on all five, the barrier's fails
16 of its 32 tests, and no file fails clean on any block. The job's tests take 1.8 times their
time before: 2980 s in one process on a loaded laptop, where the same files took 1690 s.

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
  reason, and each stays in the list until a fix lands. At 0.19.0 none remains. The last was the
  tube under a persistent confounder, read over 16 logs. Its floor counts the logs, not the runs:
  one log's share has a standard deviation of 0.0166 at step 13, where 2000 runs give 0.0049. It
  reads 0.9463 by step 13 there and 0.9484 over 256 logs. The suite reads 34 passed.
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
