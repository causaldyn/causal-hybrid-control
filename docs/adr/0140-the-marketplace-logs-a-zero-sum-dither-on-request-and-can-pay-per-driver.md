# ADR 0140 — The marketplace logs a zero-sum dither on request, tests the fit's gradient with it, and can pay per driver

**Status:** accepted, 2026-10-27. Builds on ADR 0134 (the exact oracle) and ADR 0137 (what the
logs identify). Moves no default: the published logs, the lump-sum world and its oracle are bit
for bit those of 0.22.0. Removes `oracle_allocation(steps=)`, as ADR 0134 announced.

## Context

**The logs test the model along one direction only.** `SharedStateMarket.generate_logs`
randomises one number per block, the budget's scale; ADR 0137 recorded that the logs identify only
the response along the logged split, and that `calibrate_budget_scale`'s check of the fit along it
is weak. The planner reads the fit's gradient in every direction that keeps the budget, the
`n - 1` zero-sum ones. `fit_equilibrium`'s F test (`lack_of_fit`) reads the residuals' level, not
that gradient. A market outside the fitted class whose residuals look right need not have the
gradient right: on a nested-logit market (three nests of four zones, nest parameter 0.5, market 2)
the F test reads p = 0.073 on 400 blocks. A local, zero-sum perturbation of the incentives
identifies the gradient at the logging point (Wager and Xu 2021, Management Science 67(11),
experimenting in equilibrium).

**The lump sum pays for a sum of rates.** In the model each driver in zone `i` receives `u_i`, so
the money paid is `sum(u_i * x_i)`. The lump sum `sum(u) = budget` budgets a sum of rates, which no
incentive programme pays (Ong, Freund and Crapis 2021, INFORMS Journal on Applied Analytics
51(5), Lyft's repositioning incentives, paid per driver). Paid per driver, the shares a budget
reaches are convex, and the plug-in plan through a fit is a convex program, so a study's regret
measures the causal error, not the optimiser's luck. The lump sum's reachable shares are not
convex, which is why ADR 0134's oracle enumerates vertices.

**`oracle_allocation(steps=)`** has been ignored with a `DeprecationWarning` since 0.22.0, and was
announced to leave in 0.23.0.

## Decision

- **`generate_logs(n_blocks, key, *, dither=0.0)`.** A dither `h > 0` moves each block's logits
  `confounding * demand` by `+-h`: `n // 2` signs of each kind in a random order, a 0 on the odd
  zone out, drawn from the block's key folded with a tag at least `2**31`, so the block's demand,
  scale and noise draws do not move. Every incentive stays positive and each block's total of the
  rates holds; an additive `+-h` would need every incentive at least `h`. The shift is logged as
  `dither`. With `dither=0.0` the logs carry no `dither` and are the published ones, bit for bit
  (a test compares them with a copy of the 0.22.0 method at both precisions; the digest of four
  markets' logs and oracles is unchanged against the 0.22.0 tree).
- **`dither_check(logs, equilibrium, *, level=0.05, power=0.8)`** returns `DitherCheck`. The
  dither's first-order move of a block's rates is `du = u0 * (shift - pi . shift)`, `u0` the
  undithered rates recovered from the logged ones, zero-sum and of mean zero given the block's
  demand and scale. The block's completions less the fit's prediction at the logged rates are
  regressed on `du` in `n - 1` coordinates and an intercept. The coefficients are the gradient
  error in the zero-sum directions (`error`, summing to zero), with its HC3 covariance; the Wald
  statistic over `n - 1` is referred to F on `n - 1` and `N - n` degrees, `N` the blocks.
  `minimum_detectable` is `sqrt(lambda * l_max)`, `lambda` the chi-square noncentrality at which
  the test has `power` at `level`,
  `l_max` the error covariance's largest eigenvalue: the least norm of an error the test finds in
  every direction.
- **`dither_size(logs, equilibrium, *, error, blocks=None, level=0.05, power=0.8,
  max_cost=0.01, draws=16, seed=0)`** returns `DitherSize`, the rule a study pre-registers. The
  study states the gradient error it must find: the error that costs its tolerance `tau` of the
  plan's lift `L`, `sqrt(2 * k * tau * L)`, `k` the least curvature of the planning objective along
  the budget (a plan read with its gradient off by `e` loses `e'(-H)^-1 e / 2`). The smallest
  dither that finds it at `N` blocks is `h = sqrt(lambda * rho / N) / error`, `rho` read from the
  pilot's residuals and the dither's second moment in closed form, `c_n D(u0)(I - 1 pi')(I - pi
  1')D(u0)`, `c_n = n/(n - 1)` for an even count of zones and 1 for an odd one. Where that dither
  costs more than `max_cost` of the logged completions through the pilot fit, the dither is cut to
  the size that costs `max_cost` and the blocks grow by `(h / size)**2`: a capped perturbation is
  a rate constraint, and the remedy is a longer study, not a larger move.
- **`SharedStateMarket(payment="per_driver")`**, opt-in and experimental. `spend(u)` is the money
  paid, `sum(u)` or `sum(u * x(u))`. `oracle_allocation()` returns
  `chc.games.per_driver_congestion_allocation`'s optimum, and `generate_logs` scales each block's
  rates so that its drivers are paid the block's scale of the budget (a bisection on the total,
  whose money rises with it), logging the money as `spend`. `pessimistic_equilibrium_allocation`
  and `sutva_allocation`, which split a lump sum, refuse it.
- **`chc.games.per_driver_congestion_allocation(value, demand, beta, congestion, mass, budget)`**
  returns `PerDriverAllocation`: the allocation, its rides, shares, money, the dual bound, the
  duality gap and the budget's multiplier. The program in the shares is convex: by Berry's (1994)
  inversion the money that reaches `s` is `m*(sum_i s_i*phi_i(s_i) - min_j phi_j(s_j))`, with
  `s*phi(s)` convex and `phi` concave. It is solved through its KKT conditions: the zones paid
  nothing share one level `t` at their floor shares; the paid zones below their demand share one
  margin of their convex cost, filled by bisection, the margin's inverse the floor's equation at
  twice the congestion; and the least level the budget reaches, by bisection, is the optimum, as
  for the lump sum. The multipliers are read off the optimum, and the Lagrangian dual is evaluated
  there zone by zone, each term a one-dimensional concave supremum: its value bounds every
  allocation within the budget whatever the multipliers, so the gap is a certificate. The lump
  sum stays the default and the published world; the benchmark's marketplace track does not
  move.
- **`oracle_allocation(steps=)` is removed.** Passing it is a `TypeError`; the ascent stays
  `chc.games.stackelberg_allocation`.
- `validation/marketplace_dither.mac` checks the first-order move, the signs' covariance by
  enumeration at 4 and 5 zones, the move's second moment, the plan's loss from a gradient error,
  the noncentrality by the Poisson mixture in 40 digits, and the size's scaling.
  `validation/per_driver_allocation.mac` checks the convexity, the money that reaches the shares,
  the margin's inverse through Lambert's `W`, the dual's concavity and its slope in `t`, and a
  three-zone optimum in 40 digits: its multipliers non-negative, the dual equal to the rides, the
  budget's multiplier equal to the rides' slope by central differences, and the float64 code's
  rides and multiplier.

## Consequences

- **The check over-rejects mildly; HC3 halves the excess.** With the true fit on markets 3 and 7, a
  dither of 0.3 and 400 blocks, it rejected 15 and 16 of 200 logs at 5% and 2 and 4 at 1%; with
  HC0 and the chi-square reference it rejected 21 and 25 at 5%, 8 and 6 at 1%. With the fit read
  from the same logs (markets 0-4, four logs each, dither 0.2) it rejected 2 of 20 at 5%, 1 at 1%,
  and its least detectable error ran 0.49 to 1.38 (median 0.96).
- **It does not reproduce the catch the logit dither was proposed on.** On the nested-logit
  markets 0-5 at 400 blocks the F test on undithered logs reads p = 0.0048, 4.3e-20, 0.073,
  5.1e-13, 1.3e-41 and 1.9e-4. The dither check reads p = 0.55, 0.023, 0.86, 0.047, 0.16 and 0.48
  at a dither of 0.2 and 0.12, 0.075, 0.99, 0.019, 0.0035 and 0.50 at 0.4: on market 2, the one
  the F test misses, it misses too (least detectable error 5.2 and 2.5, against about 1 in the
  fitted class). An additive dither had caught that market in an earlier study; a logit dither
  moves a zone's rate in proportion to it, so the zones the logging policy pays little are barely
  moved, and the fit's error there is not read. The F test on the dithered logs of market 2 reads
  p = 0.0031 and 0.013. The two tests read different things and neither dominates; the dither
  check's value is that it reads the gradient a plan follows, at a stated least detectable error.
  At a nest parameter of 1, the fitted class, the check reads p = 0.71, 0.15 and 0.25 on
  markets 0-2.
- **The size rule reads the error it was sized for, a little above it.** From an undithered pilot
  of 400 blocks with the true fit, markets 3 and 7, the dither that finds an error of 0.5 at 400
  blocks is 0.244 and 0.243 (cost 0.07% and 0.8% of the logged completions), of 1.0 half that;
  five logs dithered at it read a least detectable error of 0.51 to 0.70 and 1.02 to 1.40. The
  rule reads the pilot's residuals, the check the dithered logs' HC3 sandwich. On market 0 an
  error of 0.25 needs a dither of 0.35 on 2,394 blocks at 1% of the completions: the rule says
  "add blocks" there.

- **The per-driver oracle certifies itself.** On markets 0-5 at budgets 0 to 20 the duality gap
  reads below 1.3e-13; over 1,500 random games of 2 to 20 zones (sensitivities 0.3 to 6,
  congestion 0 to 4, budgets 0 to 30) the largest gap was 1.0e-13 of the rides, no allocation was
  negative and none overspent. On a three-zone grid of 19,683 incentives solved by the market's
  fixed point, none within the budget completed more. The budget's multiplier equals central
  differences of the rides to 1e-7. Paid per driver, a budget of 2 buys fewer rides than as a lump
  sum on markets 0-3, since the zones paid hold more than one driver each.
- **The per-driver logs cost more to make**: each block bisects its total, 64 equilibrium solves
  a block.
- **A caller of `oracle_allocation(steps=...)` breaks.** The benchmark's marketplace track passes
  it, and must drop it to run on 0.23.0.

## Alternatives

- **An additive `+-h` dither on the rates.** Rejected in the design: it needs every incentive at
  least `h`, or a clip that breaks the zero sum. It reads the zones the logging policy pays little
  as well as the others; the logit dither reads them in proportion to their rates (see
  Consequences).
- **The design-based reading**, the completions regressed on the dither with no offset. Rejected:
  it reads the whole gradient, not its error, and the completions' variation with the demand and
  the scale stays in its residual.
- **A general convex solver for the per-driver oracle** (SLSQP in the shares, a level and the
  rides). Tried first: it agreed to 1e-12 on markets 0-5 but failed on games whose floor shares
  fall below 1e-15, where its steps divide by them, with gaps up to 34 times the rides. The KKT
  reduction works on log shares and certifies every one.
- **The per-driver world as the default.** Rejected for 0.23.0: it moves the benchmark's
  marketplace track and every published log. It is a modelling choice for the marketplace study.
