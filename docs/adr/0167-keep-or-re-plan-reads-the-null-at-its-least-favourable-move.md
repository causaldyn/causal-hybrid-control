# ADR 0167 — Keep or re-plan reads the null at its least favourable move

**Status:** proposed, 2026-11-25. Experimental (`ChannelMove.keep_or_replan`). Builds the two
things ADR 0019 left out beside `ChannelMove.price`: a re-solve on the move's estimate and a rule
that chooses between keeping and re-planning. Opt-in: nothing read before changes.
`_chi_square_mixture_survival` takes noncentralities, and with none given it computes as before,
to the bit.

## Context

ADR 0019 priced a moved one-step channel in a plan's own cost. With `W` the plan's decision weight,
`d` the move, `dh` its estimate and `S` the estimate's covariance, keeping the plan loses
`d' W d / 2`, priced without bias as `keep = (dh' W dh - tr(W S)) / 2`, and a plan re-solved on
`dh` loses `replan = tr(W S) / 2` on average. Two things were left out:

- **A re-solve on the estimate.** `causal_plan` plans on a continuous-time field, and the move is
  a one-step one: the plant's one-step map is the model's plus `d u`.
- **A rule.** Re-planning pays in expectation when `d' W d > tr(W S)`. Re-planning wherever the
  keep price beat the re-plan price, both read off one log, lost to the better fixed choice at 30,
  100 and 400 decisions (ADR 0019's table): the logs that read a large move are the ones whose
  estimate errs most along `W`.

## Decision

- **The re-solve is on the plan's one-step map with the move added.** The decision weight perturbs
  the rollout `step(x, u) + E u`, `u` the action the field takes (`_perturbed_task_cost`), and the
  price is read on that map. A private model, `chc.plan._Moved`, steps exactly so: its step is
  the plan's model's, RK4 or Euler, with ruled levers set as the plan's field sets them, plus `dh`
  times the action taken. It is written as the Euler step of the rate `(step + dh u - x) / dt`, so
  `rk4_step`, and every rollout and gradient through it, take the moved map. `causal_plan` solves
  the plan's problem on it from the plan's actions: the same cost, box, rows and support model,
  whose weights `decision_weight` requires to be zero. The re-solved plan has no tube, since the
  tube's inputs are not kept with a plan.
  - The field is not moved. A field plus `dh u / dt` steps to the moved map under Euler's map
    only. Under RK4 a linear field's input matrix is `dt (I + dt A / 2 + (dt A)^2 / 6 +
    (dt A)^3 / 24) B` (`validation/plan_decision_weight.mac` STEP 1), so the move would be
    multiplied by that polynomial. For a linear field the polynomial could be inverted; for a
    nonlinear field no change of the field gives the moved map exactly. The step-level map is
    faithful for both integrators, and it is the map the price reads.
- **The rule is a one-sided test**, with null `d' W d <= tr(W S)`: re-planning does not pay in
  expectation. Its statistic is `keep - replan = (dh' W dh - 2 tr(W S)) / 2`.
  - With `dh ~ N(d, S)`, `dh' W dh = sum_i l_i (y_i + mu_i)^2`, `y` standard normal, `l_i` the
    eigenvalues of `S^1/2 W S^1/2`, and `sum_i l_i mu_i^2 = d' W d`. So the null holds many laws,
    one for each way `d` lies along the eigenvectors.
  - The p-value is read at the law that puts all of `tr(W S)` on the largest eigenvalue, by
    Imhof's inversion with noncentralities (Imhof 1961). The test re-plans where it is at
    most `level`. The margin, the difference at the `1 - level` quantile of that law, is found by
    Brent's method on the same inversion. A Cantelli bound brackets it.
  - That law is the least favourable one in three cases. On one direction the tail grows with
    `|mu|` (`validation/keep_or_replan.mac` STEP 3a). On eigenvalues alike, only the sum counts.
    On two directions with eigenvalues 1 and 0.3, the tail at the critical value falls as the
    noncentrality moves to the smaller one (STEP 4). Beyond these cases it is a measurement: over
    154 spectra of 2 to 11 directions, the noncentrality split at random and whole on each other
    direction, no split read a heavier tail at levels up to 0.25. At 0.27 one read 0.0014 more,
    and at 0.3 one read 0.010 more. So `level` is refused above 0.2.
- **One read, not anytime-valid.** The level holds for one read of the move, with `dh` near
  normal and `S` read as known. Read after every batch, with a re-plan at the first `replan`, the
  test re-plans wrongly more often. The anytime-valid part of the path is `DriftAlarm`: read the
  move once, on the decisions after the alarm sounds.
- **`KeepOrReplan`** is a frozen dataclass. It holds `price`, the `MovePrice` with both
  expectations, and `margin`, `p_value`, `level`, `decision` (`"keep"` or `"replan"`) and
  `replanned`, the re-solved plan, whatever the decision. `difference` is the statistic.
- **A method of `ChannelMove`, like `price`.** A module function would put `ChannelMove` at the top
  level: `tests/test_public_api.py` requires a lifecycle call's argument types there.

## Consequences

Measured by `scripts/bench_channel_move.py`, section `plan`, with `JAX_ENABLE_X64=1`, on ADR 0019's
budgeted plan and logs. Its earlier columns reprint ADR 0019's to the last digit: the second logs
draw from their own stream. Regrets are against the plan that knew the move. Always keeping loses
0.095.

| decisions | always re-plan | prices compared | tested at 0.05 | re-planned on | tested, re-planned on a later log |
|---|---|---|---|---|---|
| 30 | 0.184 | 0.149 | 0.102 ± 0.004 | 0.020 | 0.096 ± 0.001 |
| 100 | 0.080 | 0.100 | 0.099 ± 0.002 | 0.035 | 0.093 ± 0.001 |
| 400 | 0.024 | 0.043 | 0.084 ± 0.002 | 0.215 | 0.081 ± 0.002 |
| 1600 | 0.0060 | 0.0060 | 0.0124 ± 0.0018 | 0.915 | 0.0147 ± 0.0018 |

- **The test does not take the re-plans that lose.** At 30 decisions, re-planning costs twice what
  keeping does. The test re-planned on 2% of the logs and lost 0.102. Comparing the prices lost
  0.149.
- **It is slow to take the re-plans that pay.** At 400 decisions, re-planning pays four times
  over in expectation (keep price 0.093, re-plan price 0.026), but the test re-planned on 21.5% of
  the logs. One log's keep price spreads about as far as its mean. A test of `mu <= 1` with one
  normal draw at `mu = 2` and level 0.05 rejects about a quarter of the time. That is the price of
  a bounded false re-plan rate on one read.
- **The selection ADR 0019 measured stays where the test re-plans.** At 100 decisions the test
  lost 0.099, slightly more than keeping: the few logs it re-planned on erred most along `W`. With
  the choice made on one log and the re-solve on a second log of the same size, the test lost
  0.093, below keeping. `later.keep_or_replan(plan).replanned` re-solves on a later read.
- **The level.** Over 40 000 Gaussian estimates at the least favourable move of a four-entry
  channel, the test re-planned on 0.0523 at 0.05. With the same `d' W d` spread over all four
  directions it re-planned on 0.0263, and with no move on 0.0029. Over 100 logs of 400 decisions
  of the budgeted plan with no move, it re-planned on none.
- **The re-solve.** On the budgeted plan, the re-solved plan is SLSQP's minimiser of
  `_perturbed_task_cost` at the estimate within 1e-5, and within 1e-6 on six probe logs; it
  converges. On the plant it beats the kept plan: the large move of the tests, read over 1600
  decisions, leaves the re-solved plan within a tenth of the kept plan's loss of the oracle's cost.
- **Cost.** Each call computes the decision weight again, one Hessian of the task cost, as
  `price` needs. The test inverts Imhof's integral about a dozen times for the margin and once
  for the p-value. The re-solve is one descent, warm-started at the kept plan. No wall time was
  measured.
- **Checked.**
  - `validation/keep_or_replan.mac`: the re-solve's regret `W (dh - d)^2 / 2` to second order;
    the two expectations; the one-direction tail's monotonicity, its critical values at 0.05 and
    0.01 and a p-value against the library's; the two-direction tails at four splits against the
    library's to 1e-9; and the Wald test's size. It runs in Maxima 5.49 (the CI image) as well.
  - `tests/test_keep_or_replan.py` has 17 tests: the noncentral inversion against scipy's
    `ncx2` and the conditional integral, the closed forms, the least favourable split, the level by
    Monte Carlo, the rate on unmoved logs, the power, the re-solve against SLSQP and on the plant,
    the moved map's step and cost, and the refusals.
  - Three mutants are caught: the noncentrality put on the smallest eigenvalue, the phase term's
    sign flipped, and the move put into the field under RK4.

## Alternatives

- **A Wald test on `keep_error`.** Rejected. The keep price's error grows with the move it reads,
  so at the null's edge the statistic is far from normal. On one direction its false re-plan rate
  is 0.0029 at a nominal 0.05 (STEP 5). By Monte Carlo, at the least favourable move, it was
  0.0028 to 0.037 over spectra of 1 to 128 directions. A test that rejects so rarely at the null's
  edge gives up power.
- **The misspecification test's central mixture, `d = 0`.** Rejected. It tests whether the
  channel moved, which `DriftAlarm` already watches. It does not test whether re-planning pays.
- **An anytime-valid comparison** (Choe and Ramdas, "Comparing sequential forecasters",
  Operations Research, 2024). Not built. It needs a score for each plan on the same windows. The
  re-solved plan is never run, so its loss on a window can be read only through the model, which
  is the quantity under test. CHC holds no confidence sequence for the channel's move. The
  anytime-valid route that exists runs the re-solved plan as a candidate in `DeploymentGate`'s
  shadow, on realised rewards in `[0, 1]` with logged propensities.
- **The supremum over the null, found numerically at each read.** Rejected. It would hold above
  0.2, at the price of an optimisation over splits at every read. The level is capped instead.
- **A Monte Carlo critical value.** Rejected. A decision would depend on a seed, and the run would
  be slow.
- **A field-level re-solve for Euler-stepped models only.** Rejected. Plans would re-solve
  differently by integrator, and the step-level map serves both.
