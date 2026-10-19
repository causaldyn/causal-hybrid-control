# ADR 0105 — The tube carries the noise as a VAR(1) and warns where the noise it reads is not the plant's

**Status:** accepted, 2026-10-19. Amends the noise model of ADR 0080.

## Context

`prescribe`'s tube states that the plant stays inside it with probability 0.95 at each step. Its
noise part read the plan's one-step error on the log as a Gaussian AR(1) with one lag-1
correlation, pooled over the states with each state scaled to unit root mean square. Two cases
broke the stated level.

1. **A persistent confounder, logged and adjusted.** On the quickstart's plant with its demand an
   AR(0.8), the supply's one-step error persists at 0.80 and the wait's at 0.02. The pooled
   correlation read 0.41, half as persistent on both. Over 4000 runs of one log the plant stayed
   inside the tube 0.956 of the time at week 1 and 0.784 at week 15. On a second plant with the
   same confounder (4 logs, 2000 runs each) it read 0.952 at step 1 and 0.765 at step 15.
2. **A persistent disturbance that the log does not record.** An AR(0.9) disturbance of standard
   deviation 2 on the supply's rate, beside white demand, correlates with the state the fit
   regresses on. The fit reads part of it as the state's own drift, so the one-step error on the
   log, 0.044 to 0.050 in variance, is smaller than the plant's, 0.063. The tube held 0.918 at
   its one trustworthy step and 0.899 at step 4.

A drift that the fit's polynomial misses has the same signature. Its error is a function of the
state, so it persists along the plant's path. Under a sine drift fitted at degree 1, the tube held
0.95 over its trustworthy steps and 0.876 at step 15.

## Decision

- **The noise is a VAR(1).** `w_{k+1} = Phi w_k + eta_{k+1}`, so `Cov(w_{k+1}, w_k) = Phi S`.
  `Phi` is Yule-Walker's `Gamma_1 S^-1`, `Gamma_1` the mean of `w_t w_{t-1}'` over each unit's
  consecutive periods. The estimate is equivariant: in other units or another basis of the states,
  `w -> T w`, it reads `T Phi T^-1`.
- **The recursion.** The gap's cross covariance with the noise is
  `C_{k+1} = (M_k C_k + S) Phi'`, where the AR(1) read `rho (M_k C_k + S)`.
  `validation/var_noise_tube.mac` checks it against the double sum over all pairs of steps, with
  step matrices that move each step. It also checks the reduction to the AR(1) at `Phi = rho I`.
  Under a norm bound, `||Phi||_2` carries the cross term where `|rho|` did.
- **Stationarity.** A whitened `S^-1/2 Gamma_1 S^-1/2` past a contraction is scaled back to norm 1.
  The innovation's covariance `S - Phi S Phi'` then has no negative eigenvalue. A direction that
  `S` does not span persists at 0. `causal_plan(noise_persistence=)` takes either a correlation or
  an `(n, n)` coefficient. It refuses a coefficient of the wrong shape and one that no stationary
  noise of covariance `noise` has.
- **The certificate reports `Phi`** as `tube_autoregression`. `tube_persistence` keeps the pooled
  correlation for the report.
- **A warning where the channel's residual persists.** The channel's moment residual has the
  adjustment set regressed out. If its persistence check rejects at 0.05 (`noise_persistence_p`),
  the one-step error the tube reads may not be the plant's noise. A `tube` warning then gives the
  p-value, and the report's persistence line says so. `trustworthy_steps` does not change.

## Consequences

- The quickstart's AR(0.8) log holds 0.958 to 0.964 at every week, where it held 0.784 at week
  15. The bar of four Monte Carlo standard errors is 0.936. On the second plant, over 16 logs of
  2000 runs, the AR(0.8) case holds 0.944 to 0.950 at each step, where it held 0.765 to 0.952.
  Its late steps stay about half a point under 0.95, more than the runs' Monte Carlo error of
  0.0012. The source is not found.
- Where the noise does not persist, `Phi` is near 0, and coverage stays within Monte Carlo
  error. On white demand, over 4 logs, the AR(1) tube held 0.943 to 0.952 at each step and the
  VAR(1) tube 0.944 to 0.952. Over 16 logs the VAR(1) tube holds 0.948 to 0.951.
- The warning fires on every log of four cases where the tube falls short: the unrecorded
  AR(0.9) disturbance (p = 1e-27 to 1e-31), the sine drift (p = 0.011 to 1e-9), the pendulum
  demo's wind (20 of 20 logs), and the BOPTEST-like zone behind a floor (12 of 12 logs).
- On logs with white noise it fires at about the check's size: 3 of 40 on the quickstart's plant
  at 0.05, 1 of 40 at 0.01, and 2 of 16 on a second plant.
- The tube does not widen for an unrecorded disturbance. The fit gives no estimate of the noise
  that it reads as drift, so the tube cannot price it. Under the AR(0.9) disturbance the tube
  holds 0.918 at step 1 and 0.899 at step 4.

## Alternatives

- **Conformal calibration of the tube's profile on held-out windows (`HorizonConformal`).**
  Rejected as the default. It needs windows as long as the horizon, and the quickstart logs 12
  periods against a horizon of 15. A one-unit log yields few windows. The scale is calibrated on
  the log's actions, not carried through the plan's step matrices. It stays a reader beside the
  tube (ADR 0099).
- **One correlation per state (a diagonal `Phi`).** Rejected: it drops the cross-lag terms. It
  also depends on the basis of the states, so a rotation of the states moves the tube.
- **ARMA(1,1) autocovariances, `Gamma_h = A^(h-1) Gamma_1`.** The recursion takes it as
  `C_{k+1} = M_k C_k A' + Gamma_1'`. Not taken: on the unrecorded disturbance the error on the
  log has already lost the persistence that the drift absorbed. The fitted autocorrelation
  decays to 0 by lag 4, so a longer memory model reads nothing more. `A = Gamma_2 Gamma_1^-1` is
  also unstable where `Gamma_1` is near 0, which is the white-noise case.
- **No trustworthy step where the check rejects.** Not taken as a default. The trustworthy
  prefix is what an operator acts on, and this rule takes it away where it held. On the
  BOPTEST-like zone behind a floor, 12 logs, the tube held 0.96 and 0.95 at the 2 trusted steps,
  and the rule set the prefix to 0 on every log. It also took the prefix from the sine drift's 3
  steps, which held 0.95, and from 1 of 4 well-specified logs. It is right only for the unrecorded
  disturbance, whose one trusted step held 0.918. A change to the prefix's default is the
  author's decision. It would be opt-in first, with a `FutureWarning`.
