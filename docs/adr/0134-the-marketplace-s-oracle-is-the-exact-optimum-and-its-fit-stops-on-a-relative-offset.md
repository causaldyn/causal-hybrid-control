# ADR 0134 — The marketplace's oracle is the exact optimum, its fit stops on a relative offset, and the planner holds a zone the fit does not identify

**Status:** accepted, 2026-10-25. Amends ADR 0114 (the best plan's lift, read against an oracle
that was a local ascent) and ADR 0123 (the numbers its Consequences quote, read through a fit that
stopped early).

## Context

**The oracle was a local ascent.** `SharedStateMarket.oracle_allocation(steps=1200)` ran
`stackelberg_allocation`: Adam from the even split, projected onto the budget simplex. The rides a
logit congestion equilibrium completes are not concave in the incentive, so the ascent ends at a
local maximum at best, and at a step count short of one at worst. Every regret and every share of
"the oracle's lift" in the marketplace's docs was read against it. On the 30 markets
`SharedStateMarket(seed=0..29)`, 2500 steps of the ascent ended short of the global optimum on 17
in float64, by up to 0.215 rides, and on 13 in float32, by up to 0.303. At 1200 steps, the default,
market 1 read 10.55 rides where 11.22 are reachable.

**The fit stopped early and said it converged.** `fit_equilibrium` passed only `maxiter` to
L-BFGS-B. scipy's other tolerances applied: it stops where the objective's relative fall is below
2.2e-9 of `max(|f|, 1)`, and `f`, the mean squared residual, is about 4e-4. On the logs of markets
0-3 (400 blocks) it stopped with `converged=True` at Bates and Watts' (1981) relative offsets of
0.26, 0.59, 2.6 and 0.079: the Gauss-Newton step left was up to 2.6 times the radius of the
parameters' confidence region. A polish lowered the mean squared residual by 2.1% on market 2 and
moved an attractiveness by up to 0.59.

## Decision

- **`chc.games.exact_congestion_allocation(value, demand, beta, congestion, mass, budget)`**
  returns the global optimum and the vertex that certifies it, in float64 with numpy, as a
  `CongestionAllocation`. Berry's (1994) inversion reads the equilibrium off the shares: `s` is the
  equilibrium under `u` exactly where `log(s_i)/beta + congestion*s_i - value_i` is `u_i` plus one
  constant. The unpaid zones share one level `mu` at their floor shares, Lambert's `W` of
  `beta*congestion*exp(beta*(mu + value_i))` over `beta*congestion`, solved by Newton's method on
  the log share from a start within 1 of the root. The rides at level `mu` are at most
  `U(mu) = mass - sum_i (mass*g_i(mu) - demand_i)+`, and `U` is reached where the deficit
  `1 - sum g_i(mu)` can be filled with no zone past its demand for at most the budget. That least
  cost is a separable concave minimisation over a box cut by a hyperplane, so it lies at a vertex
  (every funded zone at its demand but one; `k * 2**(k - 1)` vertices over the `k` zones whose floor
  is below their demand), and it falls as `mu` rises, so bisection finds the least level the budget
  reaches. An exchange argument makes that level's `U` the global maximum. The function refuses
  more than 20 zones, where a level would read 22 million vertices; a solve reads at most
  `2*j + 51` levels, 49 on each of the 30 markets at either precision, 0.3 to 1.2 million vertices
  in all. `validation/exact_congestion_allocation.mac` checks the inversion, the floor, the bound,
  the vertex cost at a rational point, the exchange step, and two markets' certificates.
- **`SharedStateMarket.oracle_allocation()` returns it.** `steps=` is ignored with a
  `DeprecationWarning`, and leaves in 0.23.0. The ascent stays reachable as
  `stackelberg_allocation(market.completions, market.n_zones, market.budget, steps=...)`; the
  warning names that call.
- **`fit_equilibrium` polishes the L-BFGS stop and reads its own convergence.** MINPACK's
  Levenberg-Marquardt (Moré 1978) on the blocks' residuals, at most 100 residual evaluations, then
  at most 20 Gauss-Newton steps, each halved until it lowers the sum of squares. On a zone within
  the equilibrium solve's tolerance of its demand, where the residual has a kink, the step takes the
  generalised gradient (Clarke 1983) of least norm in the Gauss-Newton metric. `converged` is True
  where the relative offset is 1e-3 at most, or where no Gauss-Newton step lowers the sum of
  squares at all; False where the steps ran out first. `validation/relative_offset_stop.mac`
  checks the step's drop, the rule, and the kink's step.
- **`FittedEquilibrium.attract_se` carries each zone's standard error** from the Gauss-Newton
  covariance `s**2 (J'J)^+` at the polished fit: `J` the drivers' Jacobian on the rows the
  relative-offset rule reads, `s**2` the residual's mean square over the residuals less the rank,
  the pseudo-inverse cut where `_polish` cuts it. A zone whose unit direction leaves the kept row
  space by a squared share above that cut reads `inf`, not the pseudo-inverse's finite value.
- **`pessimistic_equilibrium_allocation` holds a zone the fit does not identify at zero
  incentive**, its `attract_se` infinite or above `|attract|`, and plans the budget over the rest.
  It refuses a fit that identifies no zone. Its contract, "where the offline estimate is weak or
  unsupported, the incentive is discounted rather than trusted", had been broken by funding such a
  zone. Where every zone is identified the plan is the one it was, step for step.

## Consequences

- **The oracle is at or above the 2500-step ascent on all 60** of the 30 markets at two
  precisions, to the float32 solve's resolution (2.9e-6), and strictly above it on 17 in float64
  (up to +0.215) and 13 in float32 (up to +0.303). The market's own equilibrium at the exact
  allocation reproduces its rides to 1.9e-6, the solve's tolerance; markets 4 and 1 reach 11.5314160
  and 11.2246553 in float64. The value moves with the machine's kernels only through that solve.
  Regrets read against it grow: a published plan's mean regret on the 30 markets moves from 0.196
  to 0.251 rides in float64, its median share of the lift from 0.917 to 0.896.
- **The fit converges where it says so.** On the logs of the fitted class (five markets, 50 logs
  of 400 blocks and 50 of 50), it converged on all 100; on markets 1 and 2 at offsets of 1.5e-10
  and 2.1e-10. The median over 30 markets of the largest error of a zone's attractiveness fell
  from 0.094 to 0.052 on 400 blocks and from 0.20 to 0.11 on 50.
- **A zone the logs do not read runs where the least squares take it, and is held.** The early
  stop held a zone that draws almost no drivers near its start of 1, by accident. On 50 blocks of
  markets 0-29 in float64, four polished fits read such a zone's attractiveness more than 1 off,
  where one did before, and one ran it to -4.8e4. Those four, and one zone on 400 blocks read
  -0.01 with an error of 0.40, are the five of 60 fits whose `attract_se` holds a zone; in float32,
  12 of 60. On market 2's 50 blocks the unheld plan spent 3.1 of the budget of 6 on a zone read
  -4.3 with an error of 9e2.
- **The plan's share of the exact oracle's lift**, markets 0-29, each log's share read against
  the exact oracle, as the median (least) over the 30, and the markets where the plan completed no
  more than the naive plan:

  | logs | early stop | polished, unheld | polished, held (shipped) |
  |---|---|---|---|
  | float64, 400 blocks | 0.896 (0.756) | 0.918 (0.754) | 0.918 (0.754) |
  | float64, 50 blocks | 0.813 (0.517) | 0.799 (0.150) | 0.803 (0.507) |
  | float32, 400 blocks | 0.912 (0.734) | 0.922 (0.297); market 18 | 0.923 (0.694); market 18 |
  | float32, 50 blocks | 0.803 (0.058); market 10 | 0.795 (−0.564); market 10 | 0.807 (0.155); market 10 |

  Market 2 on 50 blocks: 0.734, 0.150 and 0.676 in float64; 0.853, −0.564 and 0.874 in float32.
  Against the early stop, the held plan's paired difference is +0.001 ± 0.010, −0.001 ± 0.009,
  +0.014 ± 0.008 and +0.012 ± 0.017 in the four rows: it recovers the early stop's share, within
  noise, where the unheld plan lost it. The 50-block median does not reach the 0.85 an earlier
  reading quoted; that reading was against the ascent's oracle, and against the exact one the early
  stop's own median is 0.813. The shuffled-incentive logs, 60 at each precision, are refused on
  every one. Market 18 in float32 on 400 blocks stays below the naive plan; no zone is held there.
- **ADR 0123's numbers move.** Measured as there, five markets in float64:

  | logs | `lack_of_fit` < 0.01 | `incentive_z` |
  |---|---|---|
  | the fitted class, 400 blocks | 1 of 50 (0.009) | 11.7 to 16.0 |
  | the fitted class, 50 blocks | 0 of 50 | 3.3 to 8.1 |
  | attractiveness fixed, 400 blocks | 20 of 20 (below 3e-226) | 14.2 to 20.1 |
  | attractiveness fixed, 50 blocks | 20 of 20 (below 1e-15) | 3.8 to 8.2 |
  | incentives shuffled, 400 blocks | 11 of 20 | −5.0 to 1.0 |
  | incentives shuffled, 50 blocks | 4 of 20 | −4.4 to 1.3 |

  The shuffled logs' warnings had come from the early stop: their fits had stopped at up to twice
  the mean squared residual the polish reaches. Their `z` stays below 1.645 on all 40, so the
  planner still refuses every one; 26 of the 40 report `converged=False`, the sensitivity running
  to zero along a ridge.
- **The tilt's hedge is gone.** ADR 0123 kept the response's tilt because on markets outside the
  fitted class it gained share: 0.77 of the lift against 0.41 with a response of zero. Read again
  through the fit that stopped early and against the exact oracle, 0.72 against 0.40; through the
  polished fit with the unidentified zones held, 0.73 against 0.79, a tilt of −0.055 ± 0.039 over
  the 100 logs (−0.11 ± 0.05 over 20 logs of 50 blocks). The response of zero had lost share
  through the early stop, not for want of the tilt. On the fitted class, over five markets, the
  tilt moves the share by −0.013 ± 0.010 on 400 blocks (−0.013 ± 0.012 through the early stop)
  and by −0.104 ± 0.022 on 50 (−0.106 ± 0.026), and over 30 markets by −0.022 ± 0.012 and
  −0.057 ± 0.025. The tilt stays in 0.22.0: removing it changes a default, which ships opt-in
  first. It is re-decided in 0.23.0, together with the per-driver payment model, measured after
  this identification fix.
- **A fit costs more.** Each of its two solves evaluates the Jacobian of every block's drivers,
  by reverse mode through the fixed point, once per Levenberg-Marquardt or Gauss-Newton step: 4 to
  77 times on the fitted class, up to 102 on shuffled logs, where L-BFGS alone evaluated none.

## Alternatives

- **Tighter L-BFGS-B tolerances** (`ftol=1e-16`, `gtol=1e-12`). Rejected: on markets 0-3 it
  stopped on the relative fall again, at offsets of 0.25, 0.012, 0.11 and 0.033. Its test reads
  the objective's change, not the step left against the parameters' error.
- **The ascent on request** through `steps=`. Rejected: a reference that moves with a step count
  is not a reference, and the ascent is one call to `stackelberg_allocation`. The benchmark pins
  its published tracks to the CHC versions that produced them, so their reproduction does not
  read this method.
- **Branch and bound over the zones.** Rejected: twelve zones read 24,576 vertices a level, and
  the enumeration is the certificate; a pruned search would certify less for no measured need.
- **Plan at the attractiveness's lower bound**, `attract - radius * attract_se` per zone, clipped
  at `-30/(beta*demand)` so the logits stay finite. Rejected for the hold, which ties with it on
  the mean and is simpler: the paired difference, held less lower bound, is −0.007 ± 0.012,
  +0.011 ± 0.017, −0.003 ± 0.012 and +0.084 ± 0.053 in the four rows above. The lower bound read
  0.925 (0.738), 0.800 (0.230), 0.936 (0.682) and 0.790 (−0.388), and on 50 blocks in float32 lost
  to the naive plan on market 28 as well as 10. A lower attractiveness does not make a zone less
  worth funding: an unmet demand reads larger, and the plan through market 2's 50 blocks still
  spent 3.0 of 6 on the unidentified zone.
- **`scipy.special.lambertw` for the floor shares.** Rejected: its argument
  `beta*congestion*exp(beta*(mu + value))` overflows where the share is near 1 and underflows
  where it is below the smallest float; Newton's method on the log share does neither.
