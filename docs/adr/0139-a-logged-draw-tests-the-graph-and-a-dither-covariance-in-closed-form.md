# ADR 0139 — A logged draw tests the graph, and a dither covariance in closed form

**Status:** accepted, 2026-10-26. Both additions are experimental and opt-in. Amended 2026-10-27
by ADR 0145: `Prescription.to_json()` carries the dither check, under `dither_check`, so the last
line of the Consequences no longer holds. Amended 2026-10-27: the table's `c = 0.15` and
`c = 0.27` rows were read on 60 and 40 logs, which gave 0.32 and 0.68; on 400 logs of the same
stream they reject at 5 % on 0.39 and 0.875, with binomial standard errors 0.024 and 0.017, and the
rows now hold those. The stream: log `r`, for `r` from 0 to 399, is
`numpy.random.default_rng(r)`, which draws the 100 units' first states and then, each of the 6
periods, the confounder, the logged draw and the plant's noise together as
`standard_normal((3, 100))`, with `d = 1`, the empty graph and float64; the first 60 and 40 of its
logs give the old rows back.

## Context

ADR 0136 recorded what the logger check cannot see. A confounder that is drawn afresh each period
and Gaussian leaves the log of the states and the levers with the law of a loop without it
(`validation/confounded_log_equivalence.mac`). No test that reads only those columns has power
above its size there. The adjusted channel is still biased by `c d/(d^2 + s^2)` on the plant
`x' = a x + b u + c w + sg e` with the logger `u = d w + s eta`.

A logger that adds a random draw `eta` and logs it gives a second estimator. The draw moves the
levers apart from anything the plant reads. So the channel it instruments, conditioned on the
states alone, does not depend on the graph. The difference between the two channels is a
Durbin-Wu-Hausman contrast (Hausman 1978; Donald, Hsu and Lieli 2014).
`misspecification_cost` (ADR 0024) already prices the difference between two fits in the plan's
regret metric and tests it with Imhof's mixture.

`design_experiment` spends a budget by reverse water-filling on the diagonal of the decision
weight. It has no answer for a dither that moves several levers together.

## Decision

- **`prescribe(..., dither=None)`.** `dither` names the column or columns that hold the logged
  draw. With it, `prescribe` refits the channel by `fit_causal_residual(..., instrument=<draw>)`.
  That fit conditions on the states alone and uses the same transitions, clusters, folds, seed,
  integrator and nuisances as the plan's fit. Then `misspecification_cost(plan, plan_fit,
  instrumented_fit)` prices the contrast. The result is `Prescription.dither_check`, a
  `DitherCheck` with the status, the contrast, the share `tr(W S_diff)/tr(W S_iv)`, and the
  instrument's relevance. The plan's fit is refitted with `influence=True` for the contrast,
  not made so once: keeping the influence compiles another program, and on jax 0.4.30 its rounding
  moved the channel's error by an ulp. The plan, the certificate, `to_json()` and every report line but the
  new one are bit-identical with and without `dither`. A test asserts this.
- **The void case is an exact condition, not a threshold on `tr(W S)`.** The check is void when
  two things hold. First, the plan's fit conditions on the states alone. Second, least squares of
  the levers on the nuisances' polynomial of the states and the draw leaves less than `sqrt(eps)`
  of their spread. The two fits then use one variation, so there is nothing to falsify, and
  `contrast` is None. A threshold on the share does not work, because the cross-fitted nuisances
  keep the two fits apart. In the void case the share read 0.001 to 0.013 on 500 transitions of
  the scalar plant.
- **Refusals.** `prescribe` refuses `dither` in these cases:
  - with `hold_constraints` or `drivers`, where the plan has no regret metric in the fit's
    parameters;
  - with levers that follow their logged rule;
  - with an asserted empty adjustment set, whose fit keeps no influence;
  - where a draw is also a lever, a state, a driver or a covariate;
  - where the draw does not identify the channel given the states.
- **The stop rule belongs to the caller.** A rejection at 5 % is logged at `WARNING` with
  `chc_event="dither"`. `prescribe` does not refuse the plan. The contrast's `cost`, read
  against what the caller tolerates, decides together with the p-value.
- **`chc.experiment.dither_covariance(cost, weight, precision=None)`** minimises
  `tr(C V) + tr(Q (P0 + V)^-1)` over `V >= 0`. The optimum precision is the matrix geometric mean
  `C^-1 # Q`. Where `V* = C^-1 # Q - P0` is positive semidefinite, it is the answer, with value
  `2 tr((C^1/2 Q C^1/2)^1/2) - tr(C P0)`. Otherwise the bound `V >= 0` binds. The function then
  runs L-BFGS from three starts over a Cholesky factor of `V` in `C`-whitened coordinates and
  reports a duality-gap bound. `validation/dither_covariance.mac` derives the gradient, the
  convexity, the closed form and its value, and solves the binding case at 40 digits. On diagonal
  data the optimum is `sqrt(Q_kk/C_kk)` per lever, which is `_water_filling`'s level at
  `Q = uses W/2` and `C = price`. A test reproduces `_water_filling`'s allocation with one clamped
  lever. On `C = [[2, 1], [1, 1]]` and `Q = [[1, -1/2], [-1/2, 2]]`, the value is
  `2 sqrt(3 + sqrt 7) = 4.7522`. That is 16 % below the best diagonal dither, `4 sqrt 2`, and 21 %
  below the best white one, 6.

## Consequences

- `validation/dither_hausman.mac` derives the contrast's local noncentrality,
  `c^2 s^2/(sg^2 (d^2 + s^2))` per transition, and its fixed-alternative form. A draw of a tenth
  of the logger's spread needs 50.5 times the transitions of one as large.
- **The check is conservative on short panels.** It reads the difference's variance from the two
  influences, summed two ways and taken at CR1's factor `G/(G - 1)` for the smaller dimension's
  count. On 100 units over 5 transitions that dimension is the 5 periods. The measured standard
  errors read high by these ratios: 1.13 for the instrumented fit, 1.11 for the plan's fit, and
  1.23 for their difference. The cross-term is read closer to its true value than either
  variance, so the difference reads highest. Measured on 500 transitions:

  | Case | Logs | Rejected at 5 % | One-degree closed form |
  |---|---|---|---|
  | Correct graph, `c = 0` | 80 | 0.05 | 0.05 |
  | Correct graph, `c = 0` | 40 | 0.025 | 0.05 |
  | Adjusted graph | 80 | 0 | 0.05 |
  | `c = 0.15` | 400 | 0.39 | 0.64 to 0.66 |
  | `c = 0.27` | 400 | 0.875 | 0.975 to 0.99 |
  | `c = 0.5` | 20 | 1.0 | 1.000 |
  | `c = 0.8` | 10 | 1.0 | 1.000 |
  | `c = 1.2` | 10 | 0.9 | 1.000 |

  The test's power is in the plan's regret metric, not in the channel's own distance. A stronger
  confounder moves the plan as well as the channel, so power is not monotone in `c` on one log: on
  one seed the contrast read p = 0.050 at `c = 0.5` and p = 0.14 at `c = 0.8`. One log in ten
  passed at `c = 1.2`, with p = 0.20.
  The misspecification test holds size within three Monte Carlo standard deviations, and power
  of at least 0.8 at `c = 1/2`.
- `to_json()` does not carry the dither check. A report line does, and only when `dither` is
  given.

## Alternatives

- **A threshold on the share.** Rejected because cross-fitting keeps it off zero in the void
  case, as above.
- **A scalar t-test on the channel's constant.** Rejected because it does not read the plan's
  regret metric. On the scalar plant it rejected 0.325 against the metric's 0.25 at `c = 0.15`,
  and it would not say what the miss costs.
- **An SDP solver for the binding dither case.** Rejected because it needs a new dependency. The
  convex problem is smooth on the factor, and the duality gap certifies the result.
- **Dropping CR1's factor in the contrast.** Not done. It is the channel error's documented
  small-sample correction (ADR 0053, ADR 0062). The contrast reads it as every other consumer
  does.
