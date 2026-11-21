# ADR 0157 — The dither check's instrumented fit holds the adjustment set's push in its drift

**Status:** accepted, 2026-11-21. Experimental: `prescribe(dither=)`, `DitherCheck` and any stop a
caller reads off them stay experimental. Amends ADR 0139 on what the instrumented fit conditions
on.

## Context

ADR 0139 refits the channel with the logged draw as its instrument, conditioned on the states
alone, so that no column the levers move enters it. The plan's fit holds the adjustment set's push
in its drift regression, at the covariates' values at the step's start (ADR 0129). The
instrumented fit held none. `misspecification_cost` prices every parameter, the drift's included
(ADR 0024), so the contrast read any gap between the two drifts as a miss.

The two drifts are two estimands wherever a covariate leans on the state. A drift on the states
alone reads the covariate's push, projected on the state, into its slope. A drift that holds the
push reads the plant's slope. A covariate drawn afresh is apart from the state, so the two agree.
A covariate that persists is not apart from it, since its past pushed the state. Under `"rk4"` the
drift also reaches the channel through RK4's growth factor. Both fits read one step's effect of
the levers, `dt T(A) B`, so to first order in `dt` the instrumented channel moves by
`-dt dA B / 2`, where `dA` is the gap between the drifts.

causaldyn-bench's Track Q version 3 measured the result. It was pre-registered and run on 0.23.0.
On the true graph the contrast rejected 0.155 [0.108, 0.213] of 200 panels at 0.05 in the
persistent-confounder world, and 0.000 in each other world. In that world a promotion AR(0.8) is
read by the logger and pushes supply. On 1.3.1 a diagnostic read 30 panels of that world at
persistence 0.5, on a seed that no study reads. The instrumented drift's slope of supply on supply
read 0.600 above the plan's, against the exact lean 0.587 (the bench's
`scripts/dither_persistence_lean.mac`). The channel's constant read 0.021 (MC SE 0.006) below the
plan's, against `dt dA B / 2 = 0.020`.

## Decision

- **The instrumented fit's drift holds the adjustment set's push**, in its own nuisances' basis,
  as the plan's fit's drift does. Its moment still reads the states alone, and the draw. The two
  fits then differ in what identifies the channel and in nothing else.
- The hold is a private `chc.dynamics_id._fit_causal_residual(held=)`.
  `fit_causal_residual` calls it with nothing held, and its signature and results do not change.
- The plan, its fit, its certificate and the record's keys do not move. The check's p-value,
  cost, noise and share move wherever the adjustment set is not empty.

## Consequences

**The size on the true graph.** causaldyn-bench's `dither_persistence` study used Track Q's
confounder world: 400 units over 12 periods, `rk4`, float64, the oracle arm's graph, and the
promotion's persistence set in each cell. It drew 200 panels a cell, the same panels for both
libraries. The table gives the rate of rejection at 0.05, with the binomial standard error and
Clopper-Pearson's 95 % interval. The lean is the slope of the promotion's push, per unit of time,
that a drift on the states alone reads in its supply row.

| persistence | lean (supply, wait) | 1.3.1 | this change |
|---|---|---|---|
| 0 | (0, 0) | 0.000 ± 0.000 [0.000, 0.018] | 0.035 ± 0.013 [0.014, 0.071] |
| 0.5 | (0.587, 0.159) | 0.000 ± 0.000 [0.000, 0.018] | 0.010 ± 0.007 [0.001, 0.036] |
| 0.8 | (0.956, 0.261) | 0.085 ± 0.020 [0.050, 0.133] | 0.040 ± 0.014 [0.017, 0.077] |
| 0.95 | (1.153, 0.192) | 0.265 ± 0.031 [0.205, 0.332] | 0.030 ± 0.012 [0.011, 0.064] |

In 1.3.1 two errors partly cancelled. The instrumented drift's residual held the covariates'
push, which its influence read as noise. So `tr(W S)` read 4.5 times its value on the fresh
promotion (mean noise 4.86e-5 against 1.09e-5), and the contrast rejected nothing up to a
persistence of 0.5. At 0.8 and 0.95 the lean outgrew that margin. With the push held, the mean
priced cost is within 6e-6 of 0 at every persistence, the median p-value is 0.54 to 0.57, and the
rate is 0.06 to 0.09 at 0.1. At 0.05 every cell reads at most the level. Three cells hold the
level within their interval. The cell at 0.5 reads below it (2 of 200, the interval
[0.001, 0.036]), and at 0.1 and 0.2 that cell reads 0.090 and 0.180. That is the conservative
side, and it is in line with the check's read of `S`, the largest of three reads.

**The power.** The same study read the wrong graph in the three worlds where Track Q version 3's
gate passed, 50 panels a world. The table gives the rate of rejection at 0.05.

| world | the wrong graph | 1.3.1 | this change |
|---|---|---|---|
| mediator | adjusts for a confounded mediator | 1.000 [0.929, 1.000] | 1.000 [0.929, 1.000] |
| collider latent | holds a collider that a latent moves | 1.000 [0.929, 1.000] | 1.000 [0.929, 1.000] |
| confounder AR | omits the persistent promotion | 0.800 [0.663, 0.900] | 0.920 [0.808, 0.978] |

The mean priced cost on the wrong graph rises by 2 to 3 %. In the confounder world the noise
halves (5.02e-5 to 2.55e-5), so the power there rises.

**A scalar plant.** On `x' = 0.9 x + 0.5 u + 0.8 w + e`, `u = w + eta`, with `w` AR(0.9) and
held by the graph, over 100 units and five transitions each, the lean is 0.154 of `w`'s push
(`validation/dither_persistence.mac`). Under `"euler"` the contrast rejected the right graph on 57
of 200 logs at 0.05 in 1.3.1. With the push held it rejects 1 of 200
(`tests/misspecification/test_misspec_dither_persistence.py`). Over three logs of 400 units the
two fits' drift slopes differ by 0.158 in 1.3.1 under either integrator, and by under 0.01 with
the push held (`tests/test_dither_check.py`).

**Limits.** The held push is read in the instrumented fit's own sieve, which is chosen on the
states. Under `nuisance_knots="auto"` the plan's fit may choose a spline in a covariate. A push
outside the polynomial's span can then still leave the two drifts apart by that push's lean. Under
`"rk4"` the held push of a column the levers move, such as a mediator in a wrong graph's adjustment
set, enters the instrumented fit's step, which is second order in `dt`.

## Alternatives

- **Contrast the channel alone.** Rejected. Under `"rk4"` the instrumented channel itself carries
  the drift's lean, by `-dt dA B / 2`, and the plan's regret reads the drift as well as the
  channel.
- **Condition the instrumented moment on the adjustment set.** Rejected. A mediator or a collider
  in a wrong graph's adjustment set would then move the instrumented channel with the plan's. The
  check would lose its power on the graph errors where Track Q version 3's gate passed.
- **State the scope in the docstring and keep the check as it was.** Rejected. The cause is known
  and the fix is small.
- **Make `held=` a public argument of `fit_causal_residual`.** Not done. No caller outside the
  check needs it, and `fit_causal_residual` is in the stable tier.
