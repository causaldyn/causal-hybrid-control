# ADR 0138 — The tube reports its spread over logs, may read the plan's own actions, and the channel's region may read a grid of priors

**Status:** accepted, 2026-10-26. Adds to ADR 0133 (the tube's field part) and ADR 0099 (the
channel's region); moves no default. Amended by ADR 0141, 2026-10-27: the box's form reads each
RK4 stage's error at its own state, so under RK4 too it is never narrower than the plan's form at
the plan's actions; Decision 3's RK4 caveat and the first-step RK4 numbers below predate it.

## Context

Three questions about the tube and the channel's region were open after ADR 0133.

- **What does 0.95 promise one log?** The tube holds the plant with probability 0.95 at each step
  on average over logs. One log's fitted field is off by its own error, and every path planned
  from that log shares it. So one log's share of paths in the tube is not 0.95. It spreads about
  0.95, and the spread grows with the field's part against the noise's. On the misspecification
  suite's dead-time plant one log's share had a standard deviation of 0.077; the floor of that
  test is sized by it. Nothing in the certificate said so.
- **A dead time had no name.** `persistence_check` reads the residual's lag-1 autocorrelation. A
  plant that reads its lever two periods late leaves a residual that moves with the lever two
  periods back. The check cannot name that lag, and `irf.delay_estimate` must be pointed at it by
  hand.
- **The tube's field part is a bound over the levers' box.** `_field_errors` reads each step's
  error alone at the box's worst vertex, and `_linear_tube` adds the gains' norms. One fitted
  coefficient error moves every step, with the sign of each step's action. Where the plan's actions
  change sign, the steps' errors cancel in the plant's gap, and the box's form adds them. On a
  closed-loop log of `x' = -x/2 + u`, at the population covariance, the box's form at the plan's
  own actions is 1.04, 1.02 and 1.01 times the gap's standard deviation at steps 5, 10 and 15 of a
  held plan; 1.61 and 1.89 at steps 10 and 15 of a plan that turns from +1 to -1; 1.95, 2.11 and
  2.15 at steps 5, 10 and 15 of one that alternates (`validation/plan_action_tube.mac`).
- **The channel's region is wide at the fit's ridge.** `channel_region` reads Abbasi-Yadkori, Pal
  and Szepesvari's radius (2011) with `V0` the fit's ridge, 1e-6 of the regressors' scale. Its
  log-determinant `log det(V) / det(V0)` is then the larger part of the radius. Simpson,
  Baumgartner, Koehler and Diehl (arXiv 2603.19073 v2, 2026, Theorem 2) prove a radius that reads
  the log-determinant against any prior `V_bar` fixed before the log, and pay for a `V_bar` near
  `D'D` by `sqrt(1 + rho((V0 + D'D)^-1 V_bar))`.

## Decision

1. **The certificate reports the spread over logs.** On one state with noise `sigma` and a field
   error `kappa sigma W` across logs, `W` standard normal, the radius `z sqrt(1 + kappa^2) sigma`
   holds one log's paths with probability
   `C(W) = Phi(z sqrt(1 + kappa^2) - kappa W) - Phi(-z sqrt(1 + kappa^2) - kappa W)`. Its variance
   over logs is `4 (2 T(z, 1) - T(z, a) - T(z, 1/a))`, `a = sqrt(1 + 2 kappa^2)`, `T` Owen's. It is
   0.0324, 0.0822 and 0.1394 at `kappa` 0.5, 1 and 2, and `sqrt(0.95 * 0.05) = 0.218` as `kappa`
   grows (`validation/tube_per_log_content.mac`, `tube_per_log_content.gp`, 38 digits). At
   `kappa = 1`, 0.136 of logs hold less than 0.90. The dead-time plant's 0.077 is `kappa` 0.94;
   the control plant's 0.0078 is `kappa` 0.22. `DecisionCertificate.tube_per_log_spread` gives
   it at each step, `kappa` the tube's own field part over the root of the noise's largest
   variance, along whose direction the tube places the field. It is a report: it moves no radius.
   It is exact on one state and along the noise's widest direction on several. It is `None` under
   `bound`.
2. **`lag_check`, beside `persistence_check`.** It regresses the fit's moment residual on the
   levers 1 to `lags` periods back, each partialled on the fit's own regressors (Frisch and Waugh),
   with a CR1 error over the units or the fit's clusters. Each state's slopes are tested together
   by Wald against `F(k, G - 1)`, Bonferroni over the states. `lag` names the lag of the largest
   `t`. The docstrings of `lag_check`, `persistence_check` and `irf.delay_estimate` point to one
   another. On the suite's plant, 200 units over 12 periods, a dead time of 2 rejected on 6 logs
   of 6 at `p <= 1e-6`, each naming lag 2; a dead time of 1 named lag 1; with no dead time the
   0.05 test rejected on 5 logs of 60 (1 of the first 6).
3. **`prescribe(tube_actions="plan")`, opt-in.** The fit's error is one vector `theta`: the
   channel's coefficients, each state's drift and the push's mean, with the joint covariance
   `_FieldTerms.joints()` (`drift_joint` and `push_joint` are new fields). It moves the field at
   `(x, u)` by `H(x, u) theta`. `S_k` is the plan's state at step `k` differentiated in `theta`
   through the model's own rollout by `jax.jacfwd`, Euler map or RK4 with the error read where each
   stage reads the field. The field part is `sqrt(tr(S_k Sigma S_k'))`, the largest over the
   covariance's reads (Lutkepohl 2005, section 3.5). Under the Euler map it is never more than the
   box's form at the plan's actions (Minkowski); it is equal at the first step and on one state
   where every step's error is one row. Under RK4 the stages read the error at moved states, a
   term of order `dt^2` the box's form does not carry, and it can be the wider. The guarantee narrows: it holds for the plan's actions, not for any
   action of the box. It needs an `exact` tube and a fit that keeps its field's terms; elsewhere a
   `tube` warning says so and the box's form is read. `DecisionCertificate.tube_actions` names the
   form read.
4. **`channel_region(form="grid")`, opt-in.** The radius is the least over 65 members, `V_bar = 0`
   (Corollary 1) and `V_bar = 2^j V0`, `j = 1 .. 64`, each at `delta / 65`:
   `sqrt(1 + rho_j) sqrt(||c||_V0^2 + R^2 (l_j + 2 log(65 / delta)))`. From the eigenvalues `mu`
   of `D'D` in `V0`'s metric, `rho_j = 2^j / (1 + min mu)` and
   `l_j = sum log(1 + mu / (1 + 2^j))` (`validation/channel_region_grid.mac`).
   `stability_level` inverts it by a root in `log delta`. `ridge` stays the default.

## Consequences

- **The proof.** Theorem 2 needs: noise conditionally `R`-sub-Gaussian given the past; regressors
  predictable; `||c||_V0` bounded; `V0` and `V_bar` fixed before the log. Its steps are a Gaussian
  mixture over `theta ~ N(0, (V0 + V_bar)^-1)` (Lemma 1, an equality for Gaussian noise), the
  supermartingale `exp(<S_t, x> - R^2 ||x||^2_{V_t} / 2)`, Ville's inequality, and
  `||x||^2_{A^-1} <= rho(A^-1 B) ||x||^2_{B^-1}`. `channel_region` already states the first three
  as premises. The fourth is the gap shared with the ridge's form: `V0`, and so the grid, scale
  with the log's mean squares of the regressors, and `R` is a plug-in. The grid adds no premise the
  ridge's form does not have. So it ships, opt-in, and the docstring says the scale is read off
  the log.
- **Numbers, `ridge` → `grid`.** On 2000 closed-loop logs of `x' = 0.9 x + 0.5 u + e`,
  `u = -0.6 x + e'`, 2400 rows, `V0 = 1e-6 I`, `delta = 0.05`, the median radius fell 7.06 → 4.76;
  both held `(0.9, 0.5)` on every log. At `V0 = I`, 5.74 → 4.88. On 100 logs of 1500 rows of
  `tests/test_tube_levels.py`'s feedback log with its confounder adjusted for, the channel's region
  at 0.95 fell 1.69 → 1.16 (median ratio 1.46); both held the channel on every log.
- **Numbers, the plan's form.** On 400 Euler logs of 200 rows of `x+ = x + dt(-x/2 + u) + e`,
  `u = -x/2 + n/2`, the box's form at the plan's own actions over the plan's form, median: a held
  plan 1.000, 1.041, 1.019, 1.009 at steps 1, 5, 10, 15; a turning plan 2.62 and 1.61 at steps 10
  and 15; an alternating one 1.79, 1.75, 1.75 at steps 5, 10, 15. The misses over the plan's form
  had a root mean square of 1.00 to 1.09, over the box's 0.40 to 0.72 where the actions change sign.
  On 400 RK4 logs of the exact flow, the medians were 0.97 to 0.93 on the held plan, 2.57 and
  1.37 at steps 10 and 15 of the turning one, and 2.40, 2.67 and 2.67 at steps 5, 10 and 15 of the
  alternating one; the misses over the plan's form had a root mean square of 0.90 to 1.07. At the
  first step the box's form at the plan's actions was 0.88 of the plan's (median), and its misses
  had a root mean square of 1.19: the box's form at a single action misses RK4's stage term. The
  tube itself reads the box's worst vertex, not the plan's action. On `tests/test_exact_tube.py`'s bilinear log the
  box's tube is up to 1.26 times the plan's under RK4 and 1.19 under Euler at step 12.
- **The misspecification suite with the plan's form.** Every case that reads the tube was run
  with the box's form and the plan's, the same logs and runs. The weakest step's share, box → plan:
  control 0.9494 → 0.9488 (16 logs, floor 0.944); dead time 2, 0.9369 → 0.9297 (16 logs, floor
  0.892); a saturating response 0.518 → 0.460 (held by its `lever_range` warning); persistent
  demand 0.9483 → 0.9468 (floor 0.938); a persistent disturbance 0.9023 → 0.9001 (warned on every
  log); a sine drift 0.888 → 0.874 at step 15 (warned on every log), its trustworthy steps 0.950
  in both. With the noise read's error removed under persistent demand, the weakest step falls by
  0.0018 in both. No test's tolerance would move; the plan's form spends the slack the box's form
  left, most where the plan turns.
- **The spread is not a per-log guarantee.** A per-log level costs a tolerance interval's factor,
  1.13 to 1.19, or a higher conformal rank. `confidence=` is not built; the report states the
  spread and moves nothing.
- **Cost.** The plan's form differentiates one rollout in `theta` by forward mode, `p` columns,
  `p = n (m k + d + 1)`: one Jacobian a solve. `lag_check` is one least squares and one sandwich.
  The grid's radius is 65 closed forms a state.

## Alternatives

- **The plan's form as the default.** Rejected: its guarantee holds for the plan's actions only.
  A schedule changed after the plan, or a lever moved by hand, falls outside it. The box's form
  holds for any action of the box.
- **The delta method's tube through `jax.jacrev` or finite differences.** `jacfwd` is exact for the
  linearised rollout and cheaper for `p` small against the horizon times `n`; finite differences
  need a step size and lose digits.
- **A grid `V_bar = 2^j I`, `j = -20 .. 40`, with `V0 = 0` (ordinary least squares).** On the
  same logs it read 4.75 against 4.76. Rejected: CHC fits with a ridge, and a grid scaled by `V0` needs
  no units of its own.
- **Per-log `confidence=` (tolerance interval or a PAC rank).** Deferred: it widens the tube by a
  factor someone must choose, and the report already says how far one log may stray.
- **Naming the dead time from `delay_estimate` alone.** Rejected: it reads one response's peak and
  needs the treatment and the window. `lag_check` tests the fit's own premise on its own residual.
