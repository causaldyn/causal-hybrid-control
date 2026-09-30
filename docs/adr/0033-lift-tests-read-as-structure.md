# ADR 0033 — Lift tests read as structure

**Status:** proposed, 2026-09-30.

## Context

A media-mix model fitted on logged spend reads the business's own decisions as the channel's
effect when spend follows the business. Heusch (2026b) builds a generator where it does: a
quarterly budget that follows sales, spend raised ahead of promotions, television bursts before
Christmas, and paid shopping bid up after good weeks. On it an observational model reads paid
shopping's return as 10.66 against a true 4.20, and 8.45 with the controls an oracle would choose
(Heusch 2026a).

A geo test removes that: a treated group changes the channel's spend while a control group spends
as planned, and the groups share everything else, so their weekly gap is the channel's own,
`d_t = β[g(x̄ᵀ_t) − g(x̄ᶜ_t)] + ν_t`, `x̄` the adstock. Every parameter of the channel is in it:
the gap's rise and decay carry the carryover, tests at different spend carry the curve's shape, and
its size the coefficient. Reduced to one number, total lift over total spend, a test keeps only the
last. CHC had the channel, `chc.response.Channel`, and nothing that fitted it to a test.

## Decision

- **`fit_lift(tests, channel, *, level=0.95)`** in a new module, `chc.lift`, returning a `LiftFit`:
  the fitted channel, each parameter's estimate and interval, the noise's standard deviation and
  degrees of freedom, and the adstock the readouts covered. A `LiftTest` is a treated and a control
  `GeoArm`, each its spend over the readout and the periods before it, since a period's adstock
  reads the spend of the kernel's length up to it, and its outcome over the readout.
- **Least squares over every period of every test**, with one noise variance, and a
  profile-likelihood interval for each parameter with an `F` cutoff (Bates and Watts 1988): exact
  for a model linear in its parameters, and it follows a likelihood that curves, where a Wald
  interval is symmetric by construction.
- **Coordinates chosen for the two ridges a lift fit meets.** Tests that never bent the curve are
  fitted best by a line, which a saturating family reaches as its scale and coefficient grow
  together; noise the size of the effect can be fitted best by a step, a curve saturated at any
  spend with the carryover gone within a period, which it reaches as the scale and the retention
  fall together. The coefficient is projected out, since the gap is linear in it (variable
  projection, Golub and Pereyra 1973), and the retention, the scales and the shapes are fitted in
  logs, so what is left of each ridge is a line the solver follows.
- **An interval open to a bound is the answer.** A retention's ends are 0 and 1, a scale's or a
  shape's 0 and `inf` past a factor of `e^20`, a coefficient's `±inf`. The estimate on a ridge is
  where the fit stopped on it, within `e^40` of the template.
- **Each arm carries its share of the market**, and its spend and outcome are divided by it
  (Heusch's scaling), so a curve's scale stays in currency.
- **Refused**: no tests; a test with less spend before its readout than the kernel reads, or
  groups over different periods; fewer periods than parameters; a template whose retention, scale
  or shape is not positive; a level outside `(0, 1)`; anything but a `Channel`.

## Consequences

- On causaldyn-bench's Track M v2, Heusch's generator written from his paper, with his four
  go-dark tests of paid shopping, noise of a percent of mean weekly sales in each group as his,
  and a fresh world each time, the retention's 95 % interval covered the truth in 0.954 of 500
  histories (Clopper-Pearson 0.932-0.971), the scale's in 0.976 and the coefficient's in 0.978
  (`results/track_m2_lift.md` there). 81 % of the retention's intervals closed on both sides, 17 %
  of the scale's: his tests bend the curve little, and the scale's interval is open upward.
- The kill is measured, not argued. At three times his noise the retention still covers, 0.958,
  but 15 % of its intervals close: the tests stop identifying the carryover, and the intervals say
  so. At ten times they under-cover, 0.878, and 4 of 500 fits raise rather than converge. On Meta
  at his noise the retention covers 0.918 (0.890-0.941), below the nominal, which the `F` cutoff
  promises exactly only for a model linear in its parameters; on television, 0.958.
- A fit that cannot converge says so: a solve that spends its evaluations goes on from where it
  stopped, and `RuntimeError` is raised when 10 000 evaluations do not converge, or when the
  profile keeps finding points cheaper than the fit. `tests/test_lift.py` holds a history on
  which the fit in natural coordinates never settled.
- The curve is identified over `LiftFit.tested_adstock` alone. A plan that spends past it reads
  the family's shape, not the experiment.

## Alternatives

- **Heusch's Bayesian fit**, with PyMC-Marketing's default priors. The priors close the ridges, so
  where the tests do not inform a parameter its interval is the prior's, and it needs PyMC, which
  the library does not carry.
- **Profiling through `fit_causal_residual` on `chc.mmm`'s plant.** There a test reduced to its
  cumulative lift identifies only the channel's return at the tested spend, not its carryover, and
  the release is measured on Heusch's discrete world.
- **Natural coordinates, the coefficient searched with the rest.** The solver crawled along both
  ridges and ran out of evaluations, and the restarts the profile triggered did not settle.
- **Wald intervals** are symmetric and fail at a bound and on a ridge; a **bootstrap** refits
  every resample and inherits the same ridges.

## Not built

- **The observational channel checked against the readouts**, by a `χ²` on the gaps it predicts,
  and the gap between the two used to calibrate the sensitivity of channels no test reached.
- **A floor under a lift's error measured by placebos.** A gap read through a synthetic control or
  a regression carries an error common to its periods, which one independent noise scale does not
  model; the intervals are then too narrow.
- **Priors calibrated against past lifts.**
