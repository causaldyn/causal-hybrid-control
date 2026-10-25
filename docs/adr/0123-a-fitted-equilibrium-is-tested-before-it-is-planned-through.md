# ADR 0123 — A fitted equilibrium is tested before it is planned through

**Status:** accepted, 2026-10-21. Amends ADR 0114. Amended 2026-10-24: on the fitted class the
tilt costs share, most in short logs, where this record said it added nothing. Amended
2026-10-25 by ADR 0134: the fit stopped early, and the table and the shares below were read
through it; ADR 0134 records them read through the polished fit and against the exact oracle.

## Context

ADR 0114 made `pessimistic_equilibrium_allocation` plan through `fit_equilibrium(logs)`. The fit
assumed two things that it did not test: the model class, a logit congestion equilibrium, and an
attractiveness that moves with the demand. It also checked no input.

- On five markets whose attractiveness stays put while the demand moves, the fit converged on
  20 of 20 logs of 400 blocks. Its sensitivity was 0.56 too high on average (1.2 at most), its
  residual was 2.4 to 3 times the noise, and it gave no warning.
- With each zone's incentives shuffled across the blocks, the fit read drivers that do not follow
  an incentive (a sensitivity below 0.06 on 400 blocks). The planner still spent the whole budget
  through the fit, and on some markets the plan lost completions: down to −9 times the best plan's
  lift.
- 3 blocks fitted and reported convergence (a sensitivity of 3.57 for 2.5). A negative demand
  was accepted. Logs of mismatched shapes failed inside JAX.

## Decision

- `fit_equilibrium` refuses columns that are not finite matrices of one shape, fewer blocks than
  its `n_zones + 3` parameters, and a demand that is not positive.
- `FittedEquilibrium.lack_of_fit` is the p-value of an F test of the residuals on each zone's
  own intercept, demand and incentive to the second order, all zones together. Below 0.01 the fit
  warns (`UserWarning`) and still returns the fit.
- `FittedEquilibrium.incentive_z` is Vuong's `z` of the per-block gain in squared residuals from
  the logged incentives, against the same fit with every incentive at zero.
  `pessimistic_equilibrium_allocation` refuses a fit whose `z` is below the one-sided 5% normal
  quantile, 1.645. It also refuses a response or an equilibrium whose zone count is not the
  market's.

## Consequences

Measured in float64, on five markets, logs of 50 and 400 blocks:

| logs | `lack_of_fit` < 0.01 | `incentive_z` |
|---|---|---|
| the fitted class, 400 blocks | 1 of 50 (0.006) | 11.7 to 16.0 |
| the fitted class, 50 blocks | 0 of 50 | 3.3 to 8.1 |
| attractiveness fixed, 400 blocks | 20 of 20 (below 1e-226) | 14.1 to 20.2 |
| attractiveness fixed, 50 blocks | 20 of 20 (below 1e-15) | 3.8 to 8.2 |
| incentives shuffled, 400 blocks | 19 of 20 | −12.5 to −1.1 |
| incentives shuffled, 50 blocks | 16 of 20 | −4.3 to 1.26 |

- A misspecified market is warned of, and its fit is still planned through. On those markets the
  adjusted response's tilt carried value: 0.77 of the best plan's lift against 0.41 with a
  response of zero, over 100 logs of 400 blocks. On the fitted class it cost share: −0.006 ± 0.013
  on 400 blocks and −0.114 ± 0.028 on 50 over five markets, −0.027 ± 0.016 and −0.064 ± 0.024 over
  30 (mean ± standard error). The tilt stays as a hedge: where the fit is right it costs a few
  hundredths of the lift, most in short logs, and where the fit is wrong it gained 0.36 of it.
- A fit with no incentive response is refused, not planned. A caller that planned through one
  now gets a `ValueError`.
- Each fit is solved twice, with and without the incentives. This doubles its cost.
- The F test ignores the projection of the fitted parameters out of the residuals, so it is
  conservative. It assumes one noise variance across zones and blocks: a log with heteroskedastic
  noise can read as lack of fit.
- `FittedEquilibrium` has two new fields with no default. A caller that built one by hand must
  now give them.

## Alternatives

- **The residual's rmse against the noise.** Rejected: the logs do not give the noise's size.
- **A per-zone robust Wald test, Bonferroni across zones.** Rejected: on 50 blocks it rejected
  the fitted class's own logs at p down to 1e-37. The second-order terms of one zone are close to
  collinear, and HC0 on 50 rows cannot carry five of them.
- **A Wald test of the sensitivity against zero.** Rejected: with no incentive response, the fit
  trades a small sensitivity against a large attractiveness along a ridge. The Gauss–Newton
  covariance there is singular, and it read `z` = 11.7 and 35 on two shuffled logs.
- **Warn, not refuse, on a fit with no incentive response.** Rejected: every plan through such a
  fit reads the same, so a warning would still return an arbitrary plan of the whole budget.
