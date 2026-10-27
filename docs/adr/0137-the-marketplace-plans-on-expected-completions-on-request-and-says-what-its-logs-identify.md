# ADR 0137 — The marketplace plans on expected completions on request, and says what its logs identify

**Status:** accepted, 2026-10-26. Builds on ADR 0114 (the plan goes through a fitted equilibrium),
ADR 0123 (the fit is tested before it is planned through) and ADR 0134 (the exact oracle, the
polished fit, the held zones). Moves no default. Amended 2026-10-27: both planners refuse a
`radius` that is not a finite non-negative number (nan had sent the equilibrium planner back to
the even split and made the SUTVA plan nan; a negative one turned the shrink into a bonus), and
`pessimistic_equilibrium_allocation`'s `equilibrium` is a keyword with no default, as it was
already required.

## Context

**What one randomised number per block identifies.** `SharedStateMarket.generate_logs` draws each
block's demand season and, independently, a budget scale `S_b`; the logging policy splits
`S_b * budget` by a softmax `pi(d_b)` of every zone's demand. Given the demand, `S_b` is the only
randomised move. So the causal object the logs identify without a model is the response along the
logged split, `psi_bi = sum_j (dY_i/du_j) pi_j(d_b)` per unit of the block's total, and its
unconfounded estimator is the slope on the block's total, with the weights an OLS slope puts on a
uniform regressor (Yitzhaki 1996). The `n x n` Jacobian `dY/du`, and with it a zone's response net
of the drivers it draws from the others, is not identified from one exogenous scalar per block: the
order condition fails, and only a model of how the Jacobian moves with the state identifies it
(Berry and Haile 2014, Econometrica 82:1749). `calibrate_shared_state` controls for that very
total, which takes out the randomised move and keeps the confounded one; its docstring has said so
since ADR 0134's round. In the model's class the interference runs through one price, the level
`tau` all zones' shares answer to (Munro, Kuang and Wager 2025, AER 115:3273, where interference
runs through finitely many prices): an incentive's total effect is
`m*beta*q_i*(1{x_i < d_i} - lambda_d)`, `q = s/(1 + beta*c*s)`, one completion value `lambda_d` of a
displaced driver for every zone. The fit supplies both parts, and the plan needs no per-zone
reduced-form slope.

**The plan sat on a kink with an uncertain fit.** The tilted planner maximises the fit's
completions `sum_i min(d_i, x_i(u))` plus the response's tilt, by 600 Adam steps from the even
split. A plug-in plan through an estimated model onto a `min(d, x)` kink is knife-edge: an error in
either direction costs, and estimate-then-optimise is biased for such objectives (Liyanage and
Shanthikumar 2005, Oper. Res. Letters 33:341; Hirano and Porter 2012, Econometrica 80:1769, on
functionals with no local derivative). The fit's own uncertainty was never read: `fit_equilibrium`
had no covariance, and its docstring said "the parameters' sampling error is not carried into the
plan". And the ascent from one start stops where it stops: on the benchmark's float32 market 18 (logs of
400 blocks from `jax.random.key(19)`) the tilted plan won 0.735 of the exact oracle's lift, below
the naive plan's 0.805.

## Decision

- **`FittedEquilibrium.covariance`**, the block-clustered Gauss-Newton sandwich `H^-1 M H^-1` of
  the parameters `(attract, log beta, log congestion, log mass)` (Cameron and Miller 2015, no
  small-sample factor): `H` from the stacked Jacobian's SVD, `M` from each block's score once, on
  the rows `attract_se` reads. It is read over the zones the planner does not hold and the three
  market parameters, the held zones' attractiveness fixed at the fit (nan in their row and column).
  Where `H` over those is singular at the cut `attract_se` and the polish use, every entry is nan,
  not a pseudo-inverse's zero variance. **`FittedEquilibrium.demand_covariance`** is the planning
  demand's, the blocks' covariance over their count. Both default to None for a fit built by hand.
- **`pessimistic_equilibrium_allocation(..., objective="expected")`**, opt-in. It maximises
  `S(u) = sum_i E[min(D_i, X_i(u))]`, `X_i - D_i` normal about the fit's drivers less the planning
  demand, of variance `g_i' C g_i + h_i' C_d h_i`, `g_i` and `h_i` the drivers' Jacobians in the
  parameters and the demand in closed form from Berry's inversion. Each term is the normal loss
  function `x - (x - d)*Phi(z) - sd*phi(z)`. It ascends from the even split and from the fitted
  game's exact optimum (`chc.games.exact_congestion_allocation`, the held zones' share spread over
  the rest) and keeps the plan better on `S`. It holds the zones the tilted objective holds, reads
  no response (`response` may be None), and refuses a fit with no covariance or a singular one:
  modelled on `incentive_z`'s refusal, a plan whose uncertainty cannot be read is not made. The
  default stays `objective="tilted"`, whose plans do not move.
- **`stackelberg_allocation(..., start=)`**: the ascent may start from a given allocation,
  projected onto the budget simplex. The default start stays the even split.
- **`calibrate_budget_scale(logs, *, equilibrium=None)`** returns `BudgetScaleResponse`: each
  zone's slope on the block's total incentive, its own demand held, the total completions' slope,
  every zone's demand held, HC0 errors, and with `equilibrium` the same total slope read on the
  fit's completions at the logged blocks and their `z`. It is a model check along the one
  randomised direction, and documented as weak.
- **The tilt's comment and ADR 0123 are corrected.** The tilt `-min(max(m, 0), r*se)` is never
  positive, but on `sum(u) = budget` only its differences act, so a zone whose estimate is not
  positive gains `min(max(m_j, 0), r*se_j)` per unit against each zone `j`: the bonus the old
  tilt gave survives in relative form. The tilt stays the default's, as a diagnostic of the
  reduced form; the expected objective does not read it.
- **`CongestionAllocation.shadow_price`**, the budget's Lagrange multiplier at the exact optimum:
  `dV*/dB = m*beta*q_p*Q_O / ((|T| + 1)*q_p + Q_R)`, `T` the zones at demand, `p` the partial zone,
  `Q_R` and `Q_O` sums of `q` over the zones at their floor and those of them past their demand;
  0 where more budget completes nothing, nan at a vertex with no partial zone where drivers are
  wasted.
- `validation/marketplace_expected_completions.mac` derives the Jacobians (symbolically at three
  zones and against central differences of a solved equilibrium in 40 digits) and the normal loss
  function (by integration, its slope, curvature and limit). `validation/marketplace_shadow_price.mac`
  derives the shadow price on two structures and against central differences through Lambert's `W`.
  `validation/marketplace_tilt.mac` gains the shift invariance and the relative bonus.

## Consequences

- **The expected objective wins more of the lift.** Markets 0-29, logs of 400 blocks from
  `jax.random.key(seed + 1)`, each plan's share of the exact oracle's lift over no incentive:

  | precision | tilted (default) | expected | expected less tilted | more / less |
  |---|---|---|---|---|
  | float64 | 0.900 (least 0.754) | 0.971 (least 0.920) | +0.071 ± 0.012 | 27 / 1, by 0.006 |
  | float32 | 0.907 (least 0.694) | 0.984 (least 0.921) | +0.077 ± 0.014 | 28 / 2, by 0.0007 at most |

  Neither precision refused a fit, and no expected plan won less than the naive one; the tilted plan
  did on float32 market 18. There its 600-step ascent stops short: its surrogate reads 9.83 where
  10.67 is reachable (10.64 after 2,400 steps, 10.65 after 9,600), and it wins 0.735 against the
  naive plan's 0.805. The same surrogate ascended from the fitted game's exact optimum reaches 10.67
  and wins 0.978; the expected plan wins 0.978. The default does not move: a caller who wants the
  converged plan passes `objective="expected"`.
- **Nearly all of the hedge is the demand's error.** At the expected plan on markets 0-9 the
  drivers' SD from the parameters is 0.07 to 0.18 of the SD from the planning demand (medians over
  the zones, 0.003 to 0.005 against 0.023 to 0.055).
- **The sandwich covers.** Over 100 logs of 400 blocks, its 95% intervals held the sensitivity, the
  penalty and the total in 99%, 98% and 96% of logs on market 0 and 96%, 96% and 97% on market 1;
  the median zone's attractiveness in 96% and 94%, the worst zone's in 83% and 87%. On 50 blocks of
  market 0, 94%, 95% and 89%, and 96% at the median zone; a zone that draws almost no drivers, held
  on 64 of those logs, 64% on the rest.
- **The budget scale's check is weak, as documented.** On markets 0-4 (400 blocks, float64) the
  fit's total slope sits within 0.14 SE of the logs' (`z` from −0.14 to 0.06); on the same markets
  with the attractiveness fixed while the demand moves, outside the fitted class, `z` reads −1.56
  to −0.02, never past 1.96. The per-zone slopes held the Yitzhaki-weighted true response along the
  split within 1.96 HC0 SE on 53 of 60 zone reads in the class and 48 of 60 outside. `lack_of_fit`
  reads the class's break; this check reads the one direction the design randomises.
- **The shadow price is the value's slope.** On markets 0-7 at budgets 0.5, 1, 2, 6, 20 and 60 it
  equals central differences of the exact optimum's value in the budget to 6e-9 on all 48, 26 of
  them non-zero; at the larger budgets no zone at its floor is past its demand, and it is 0.
- **A fit costs one more SVD**, of the stacked Jacobian it already evaluates. The expected plan
  costs two ascents and one exact allocation of the fitted game, where the tilted plan costs one
  ascent. `FittedEquilibrium`'s two new fields default to None, so a fit built by hand still
  constructs; the expected objective refuses it.
- **The reduced-form tilt's relative bonus stays in the default.** It moves no number here; the
  corrected statement is the contract.

## Alternatives

The three measurements below were read through 0.21.0's fit, which stopped early (ADR 0134), on
the same 30 markets in float64.

- **A lower confidence bound** `S(u) - SD(lift)` instead of the expectation. Rejected: it read
  0.969 of the lift against the expectation's 0.971, with a radius whose meaning (an SD or a
  coverage level) is a further choice.
- **The second start alone**, the tilted objective ascended from the even split and from the
  fitted game's exact optimum. Rejected as the opt-in mode: it gained less, 0.898 to 0.944 of the
  lift, where the expectation reached 0.971.
- **The exact plug-in plan through the fit**, with no ascent. Rejected: it is knife-edge on the
  fit's error, 0.940 of the lift on average and 0.896 on market 0, where a slack budget leaves many
  plans at the optimum and the fit picks a wrong one.
- **The structural tilt** `kappa * sqrt(grad C' Sigma grad C)` subtracted from the fit's
  completions. Not measured; the expectation reads the same uncertainty without a radius to choose.
- **A pseudo-inverse sandwich where `H` is singular.** Rejected: it reads a combination the logs do
  not identify as known exactly, an optimistic plan where `attract_se` already reads `inf`.
- **Refusing on a condition-number threshold.** `cond(H)` ran from 5e3 to 4e5 on the class's
  logs through 0.21.0's fit. Rejected for the rank cut `attract_se` and the polish already use: one
  rule across the fit, where a threshold on `cond(H)` would be a new tuning constant.
- **Scoring plans across the blocks' demand** rather than at the base demand. Not taken: the exact
  oracle certifies one demand, and recorded benchmark results stay comparable.
- **Paying per driver**, a spend of `sum(u_i x_i)`, which makes the plug-in plan a convex program
  (Ong, Freund and Crapis 2021, on Lyft's repositioning incentives). A different model of the
  budget, not a planner for this one; out of this record's scope.
- **A scalar-level equilibrium solver** (Wright's omega, Newton on the level `tau`) as the backend
  of the fit's and the market's drivers. Not taken here: it solves exactly where today's solve stops
  at a relative 1e-6, so the published logs and every fitted plan would move, and with them recorded
  benchmark numbers.
