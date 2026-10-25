# ADR 0136 — A floor on the regret itself, and what the logger check cannot see

**Status:** accepted, 2026-10-25. Adds to the floors ADR 0130 named, and amends ADR 0028: the
logger check's rejection is logged at `INFO`.

## Context

The one-step LQ regret of an action `T` is exactly `W (T - psi)^2`, with `W = b^2 + rr` and
`psi = u*(b)` (`validation/action_van_trees.mac`). Two certificates in `chc.research.regret`
bound it from below over a prior, and neither bounds the regret well:

- **`information_lower_bound_certificate`** bounds the Bayes action error by van Trees,
  `(E psi')^2/(I_n + I(lambda))`, and reaches the regret only by multiplying by the least
  curvature `rr`. On a prior `N(0, 1)` with `rr = 0.5`, `xt = 1` and five units of information,
  that floor is 0.0172035 against a Bayes regret of 0.0749503: 0.230 of it. As the information
  grows its ratio tends to 0.216, not to 1, because `rr` stands in for the curvature at every draw.
  The floor that multiplies by the curvature at one point is no bound: a Bayes action goes below
  it, which is why ADR 0130 named it `local_floor`.
- **`multivariate_van_trees_certificate`** has the same gap in matrix form: `bayes_floor` bounds
  the action error, and `local_floor`, which weighs it by `M`, is read at one plant.

The logger check (ADR 0028) logs a rejection at `WARNING`. The check reads the log alone: the
levers against the state and their own past. A confounder drawn afresh each period and Gaussian
leaves no trace there, however much it moves the channel the fit reads, so a `WARNING` reads as a
guard the check is not.

The report's settings line omits `nuisance_knots`, which the run's JSON carries
(`RunProvenance.nuisance_knots`). Since ADR 0127 made the spline the default, a run that asks for
the polynomial alone prints the same line as one that took the default.

## Decision

- **`weighted_bayes_floor` and `best_weighted_bayes_floor` on `InformationLowerBoundCurve`, and
  `weighted_bayes_floor` on `MultivariateVanTreesCurve`,** beside the floors already there. No
  field changes its meaning: ADR 0130 renamed one two minors ago, and a second change would break
  the reading of recorded values.
  - Van Trees weighted by a function `C(b)` (Gill and Levit, Bernoulli 1995): for every
    estimator `T` and every `C` with `C lambda` vanishing at the ends of the prior's support,
    `E[W (T - psi)^2] >= (E[C psi'])^2 / (I_n E[C^2/W] + E[(C' + C (log lambda)')^2/W])`.
    `C = 1` with `W` bounded below by `rr` is the floor already there.
  - `weighted_bayes_floor` takes `C = W psi' = -xt (rr - b^2)/(rr + b^2)`, whose slope
    `C' = 4 xt b rr/(rr + b^2)^2` comes from `psi''` in closed form. The floor is
    `A^2/(I_n A + P)`, `A = E[W psi'^2]`, `P = E[(C' - C (b - b0)/tau^2)^2/W]`: it tends to
    `A/I_n`, the prior's mean of the local floor, and sits below it by `A P/(I_n (I_n A + P))` at
    every `I_n`. `W psi'^2 >= 0`, so nothing cancels at the knife edge `b^2 = rr`.
  - `best_weighted_bayes_floor` is the largest floor over `C` in the span of `W psi'` and the
    Hermite polynomials `He_0 .. He_8` of `(b - b0)/tau`: `a' B^-1 a`, at least the floor of each
    member, `C = 1` and `C = W psi'` among them.
  - Each mean is an adaptive quadrature at a relative `1e-12`, in standard units over 40 prior
    standard deviations, split where `b = 0`.
  - `information_lower_bound_certificate` takes `prior_width`, 0.25 by default, the width its
    van Trees arm always used, and refuses an `rr` or a width that is not positive and finite.
  - In the matrix case `C = M Psi' I^-1` gives `F^2/(F + P)`, `F = E tr(M Psi' I^-1 Psi'^T)`:
    the prior's mean of `local_floor` at the data's information, times `F/(F + P)`, the one factor
    that makes it a floor. The divergence of `C` is a central difference of the action's solves.
    The prior's mean is a tensor Gauss-Hermite rule of 32, 64, 128 and then 256 nodes in each
    direction, the first that agrees with the one before to `1e-8`. Where 256 nodes do not, the
    field is `nan`, not a number that may sit above the regret it bounds.
  - `validation/weighted_van_trees.mac` derives the inequality: its two identities in exact
    Gaussian moments, the second at each `b` so that `W` may move with `b`, the matrix form, the
    closed forms, the best weight, and sample points. `validation/weighted_van_trees.gp` holds
    every number below at 38 digits.
- **`LoggerCheck` records what it cannot see, and its rejection is logged at `INFO`.**
  - For the plant `x' = a x + b u + c w + sg e` and the logger `u = k x + d w + s eta`, with `w`,
    `e` and `eta` independent `N(0, 1)` each period, the step's kernel equals that of a loop with
    no confounder: `x' = (a - beta0 k) x + (b + beta0) u + noise`, `beta0 = c d/(d^2 + s^2)`, the
    logger's noise of variance `d^2 + s^2` and the plant's of `sg^2 + c^2 s^2/(d^2 + s^2)`
    (`validation/confounded_log_equivalence.mac`). The logged path has one law in both worlds,
    so every test that reads the log alone has power equal to its size against `c d != 0`, at
    every number of rows (Darmois-Skitovich: Gaussianity is what makes the worlds one), while the
    fit's channel is off by `beta0`.
  - A confounder that persists makes the lever depend on its own lag given the state, even where
    it does not move the plant (`c = 0`). That is what the check sees: the logger's memory, which
    weights on the state alone assume away, and the columns it read.
  - The check stays, as that diagnostic. Its record keeps its message, its `chc_event` and its
    fields; only the level moves. `Prescription.logger_check`, `PlanEvaluation.logger_check` and
    the report's line are unchanged.
- **The report's settings line names `nuisance_knots`,** after `nuisance_degree`: `auto`, an
  integer, or `none`, as the JSON's value reads.

## Consequences

The regret's floors, at `rr = 0.5` and `xt = 1`, each as a share of the Bayes regret:

| prior | information | Bayes regret | `rr` × action floor | `C = 1` | `weighted_bayes_floor` | `best_weighted_bayes_floor` |
|---|---|---|---|---|---|---|
| N(0, 1) | 1 | 0.212231 | 0.243 | — | 0.390 | 0.761 |
| N(0, 1) | 5 | 0.0749503 | 0.230 | 0.466 | 0.652 | 0.795 |
| N(0, 1) | 20 | 0.0222978 | 0.220 | — | 0.865 | 0.896 |
| N(0, 1) | 100 | 0.00470910 | 0.217 | — | 0.968 | 0.971 |
| N(0, 1) | 10⁴ | 4.77624e-5 | 0.216 | — | 0.99967 | 0.99967 |
| N(1, 0.25²), the certificate's | 32000 | 2.59582e-6 | 0.127 | 0.358 | 0.99985 | 0.99985 |
| N(3, 1) | 1 | 0.0368060 | 0.066 | 0.569 | 0.809 | 0.981 |
| N(√0.5, 0.25²), the knife edge | 20 | 0.00341830 | 0.115 | 0.212 | 0.862 | 0.880 |

At ADR 0130's example, the second row, the regret's floor rises from 0.0172035 to 0.0488867 in
closed form and to 0.0595994 with the best weight, 2.8 and 3.5 times.

The matrix floor, on the multivariate certificate's two decoupled channels (gains 1 and 0.4,
targets 1 and 0.8, a prior of width 0.25 on each), against the sum of the channels' Bayes
regrets:

| setting | `weighted_bayes_floor` | Bayes regret | share |
|---|---|---|---|
| the defaults, `n_probe = 4000` | 3.12299e-5 | 3.12387e-5 | 0.99972 |
| `n_probe = 20` | 0.00487981 | 0.00510040 | 0.957 |
| `rr = 0.1` | 9.70260e-5 | 9.71006e-5 | 0.99923 |
| `prior_width = 0.6` | 4.09055e-5 | 4.09143e-5 | 0.99978 |
| `prior_width = 1.0` | `nan` | | |

- The certificate's rule matches the 38 digits to `3e-14` at the defaults and to `5e-12` at the
  others, and the scalar floors to `1e-15`. On one channel the matrix floor, with its central
  difference, agrees with the scalar one, with its closed form, to `1e-8`.
- Each call of the scalar certificate reads one quadrature and one vector quadrature of 64
  means. Each call of the multivariate one solves the action and its two sensitivities, at the
  point and at a step either side in each direction, at `nodes^2` points for each rule it reads:
  64 nodes at the defaults, 256 at `rr = 0.1` or a width of 0.6.
- A handler set at `WARNING` no longer sees the logger check's rejection. An application that
  alerted on it reads `Prescription.logger_check`, or the `INFO` record, whose message and fields
  did not change.
- A reader that split the report's settings line by position sees one key more.
- Tests: the scalar floors against the 38 digits at five priors, the knife edge among them; their
  order below the Bayes regret, read in closed form, over drawn priors and information; the
  refusals; the matrix floor against the 38 digits at four settings, `nan` on the wide prior, and
  against the scalar floor on one channel. The logger check: the projection that makes the two
  worlds one, its rejection rate under a fresh confounder at its level at 50 and at 400 units
  while the channel read on the state is off by 0.5, and its rejections of a persistent confounder
  that moves nothing. Its record's level, and the settings line with each kind of knots.

## Alternatives considered

- **Replacing `van_trees_action_floor` or `bayes_floor` with the weighted floor.** Rejected:
  readers of recorded values would read a new quantity under an old name, a second time.
- **A basis of Hermite polynomials alone for the best weight.** It reads 0.9994 of the Bayes
  regret at `I_n = 10^4`, where the basis with `W psi'` reads 0.99967. One column more makes the
  best floor at least the closed form's at every prior.
- **Adaptive quadrature in the matrix case.** At the certificate's plant the channels decouple
  and each mean is one-dimensional, but the floor takes any channel directions, where the mean is
  `d`-dimensional. A tensor rule with a gate on its own convergence, `nan` where it does not
  settle, holds there too. Rules finer than 256 nodes are not read: NumPy's Gauss-Hermite weights
  underflow past 360.
- **Keeping the logger check's `WARNING`.** Its power against a fresh confounder that moves the
  channel is its size, so the level read as an alarm the check cannot raise.
- **Removing the check.** It sees persistence, which weights on the state alone assume away
  (ADR 0028), and the columns a logger read.
- **A design-based test of the channel** (Hausman 1978): the channel the adjustment reads against
  the one a logged draw instruments, which a fresh confounder cannot reach. It sees what the log
  alone cannot, but it needs a draw the log records. Not built here.
