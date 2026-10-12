# ADR 0080 — A plan's tube carries the plant's noise

**Status:** accepted, 2026-10-12. Amends the error tube of `causal_plan` and `prescribe`. The
channel's error that ADR 0053 and ADR 0075 read stays the tube's other part.

## Context

Up to 0.16.0, `prescribe`'s tube was one recursion. From `r_0 = 0`,
`r_{k+1} = r_k + dt (L r_k + eps) phi(L dt)`, with `phi(z) = 1 + z/2 + z^2/6 + z^3/24`. `L` is
`_rate`'s bound on the norm of the fitted field's slope in the state over the levers' box: at
every state where the field is affine in it (`global`), at the plan's start alone otherwise
(`local`). `eps` is the channel's standard error times the largest action the box allows. Under
its premises the recursion bounds the gap between two RK4 rollouts from one start under one
schedule: one of an `L`-Lipschitz field, one of a field within `eps` of it at every step (Rocq
`rk4_rollout_error_bound`, `validation/rk4_rollout.mac`). `certified_horizon` is the last step with
`r_k <= tolerance`, and `trustworthy_steps` cuts it at the barrier's prefix and at the first step
a fit the log cannot tell apart predicts differently. Under `prescribe`, `eps` is a standard error,
so the tube is a one-sigma scale of the estimation error, not a bound on it. What it measures is
how far the noise-free plant could drift from the plan if its channel were off by one standard
error in the worst direction at every step. No plant moves without noise. The part of each
transition the fitted field does not explain is outside the premises: the plant's own noise, and
whatever the plan does not forecast, such as the weather when it is no driver, or a confounder
adjusted for in the fit. Nothing in the tube held that part.

On the BOPTEST case's logs, simulated with its logging policy, size and units
(`scripts/bench_boptest_tube.py`), 0.16.0's tube read about 0.05 K one step ahead. The plant's
own one-step error had a root mean square of 0.11 K. The certificate trusted 6 to 13 of 16 steps
on nine logs in ten, and the plant left the tube at the first step on most draws. `docs/` called
the tube a scale, not coverage. But `trustworthy_steps` is "the number an operator can act on",
and an operator who reads "10 steps within 0.5 K" acts on 10 steps.

## Decision

The tube is predictive: it holds the plant's realised error, at a stated rate, step by step.

- **`causal_plan(noise=, noise_persistence=)`.** `noise` is the covariance `S` of the plant's
  one-step error `w_k = x_{k+1} - step(x_k, u_k)`: the plant's next state less the plan's RK4 step
  of the model from the plant's state. It is a number for each state's variance, or an `(n, n)`
  matrix. `noise_persistence` is `rho`, the error's lag-1 correlation, in `[-1, 1]`. The error is
  taken Gaussian and AR(1): `Cov(w_i, w_j) = rho^|i - j| S`. Both default to zero, and then every
  tube is the same as before, to the bit. A noise that is not finite, not of the state's size, not
  symmetric or with a negative eigenvalue is refused, and so is a `rho` outside `[-1, 1]`.
- **The noise's part of the gap.** `e_{k+1} = M e_k + w_k` from `e_0 = 0`. Its covariance is `P_k`:
  `P_{k+1} = M P_k M' + M C_k + C_k' M' + S` and `C_{k+1} = rho (M C_k + S)`, where `C_k` is the
  covariance of `e_k` with `w_k`. On a state matrix `A`, `M = R(dt A)`, RK4's step of a field affine
  in the state, and `P_k` is exact. On a norm bound `L`, `g = R(L dt)` bounds the norm of the step's
  slope, and the scalar recursion `p_{k+1} = g^2 p_k + 2 g c_k + q`,
  `c_{k+1} = |rho| (g c_k + q)` bounds the largest eigenvalue of `P_k`, with `q` the largest of
  `S`. Both recursions are checked in `validation/predictive_tube.mac`. The matrix recursion
  equals the double sum `sum_{i,j<k} M^(k-1-i) rho^|i-j| S M'^(k-1-j)` on a symbolic 2×2 `M`, `S`
  and `rho` at `k = 1..4`. The scalar recursion equals its own double sum at `k = 1..6`, and its
  closed forms at `rho = 0` and `rho = 1`. The bound follows from Cauchy–Schwarz on each term of
  the double sum.
- **The radius.** The noise's radius at step `k` is `sqrt(chi2_n(0.95) lambda_max(P_k))`. A
  Gaussian of covariance `P` on `n` states lies within that distance of its mean with probability
  at least 0.95, and exactly 0.95 on one state. The tube is the estimation tube plus this radius,
  by the triangle inequality. `certified_horizon` is still the last step inside `tolerance`.
- **`prescribe` reads the noise off the log.** `w` is `dt` times the fit's defect on each
  transition: the next state less the fit's own step from the logged state, levers and drivers.
  `S` is the mean of its outer products. It is not centred: a bias the field leaves on the log is
  an error the plan makes too. `rho` is the defect's lag-1 correlation within units, pooled over
  the states as `persistence_check` pools them, and read across the periods where the fit was
  given them (ADR 0075). Where no unit has two consecutive transitions, `rho` is 0. The fit keeps
  the defect row by row, as `CausalDynamicsFit.defect`; `integrator_defect` is its root mean
  square.
- **No channel error, no noise.** Where the fit reports no channel error, as on an `asserted` fit
  that adjusts for nothing, nothing bounds the field the plan rolls out. There the tube is not
  evaluated, as before. The noise alone would switch on a tube around a field nothing vouches for.
- **The certificate says what it carried.** `DecisionCertificate.tube_noise` is the root of the
  largest eigenvalue of `S`, in the states' units. `tube_persistence` is `rho`. Both are `None`
  where no tube was evaluated, and both go into the report and into `to_json()`. A noise that is
  not finite switches the tube off with the warning that a rate or a budget that is not finite
  gives.

## Consequences

- **On the BOPTEST case's logs.** `scripts/bench_boptest_tube.py cover` makes 100 logs a cell, at
  float64. It makes the case's first call on each, then runs the plant on from the log's last row
  under the plan, 200 times, with the noise and the weather going on from the log's. "Allowed" is
  the number of steps for which the plant's error stays within the tolerance of 0.5 K at 95 %,
  pooled over the cell's logs and draws. "Inside" is the share of draws inside the tube, averaged
  over the logs, with its standard error across them. "Past" is the share of logs whose
  certificate trusts more steps than that log's own draws allow.

  | cell | allowed | 0.16.0: trusted, p5 to p95 | inside at step 1 | past | now: trusted, p5 to p95 (max) | inside at step 1 | inside over the trusted steps | past |
  |---|---|---|---|---|---|---|---|---|
  | one, AR(0) | 3 | 8 to 12 | 0.297 ± 0.008 | 0.93 | 1 to 2 (2) | 0.989 ± 0.002 | 0.991 | 0.02 |
  | one, AR(0.5) | 2 | 6 to 10 | 0.397 ± 0.014 | 1.00 | 1 to 2 (2) | 0.993 ± 0.003 | 0.993 | 0.00 |
  | one, AR(0.9) | 2 | 6 to 9 | 0.443 ± 0.034 | 0.98 | 1 to 2 (2) | 0.989 ± 0.009 | 0.990 | 0.00 |
  | one, AR(0.99) | 2 | 8 to 16 | 0.422 ± 0.042 | 0.93 | 2 to 2 (2) | 0.988 ± 0.010 | 0.988 | 0.01 |
  | slab, AR(0) | 3 | 7 to 13 | 0.330 ± 0.012 | 0.79 | 2 to 3 (3) | 0.972 ± 0.006 | 0.977 | 0.02 |
  | slab, AR(0.9) | 3 | 6 to 11 | 0.508 ± 0.029 | 0.97 | 1 to 2 (2) | 0.994 ± 0.004 | 0.995 | 0.00 |

  The tube one step ahead read 0.051 K on average and now reads 0.21 to 0.29 K, cell by cell. The
  plant's error one step ahead reached 0.16 to 0.22 K at 95 %. The tube now holds the plant at more
  than its stated 0.95, because it adds the channel's error to the noise's radius, and because the
  defect's mean square counts what the field leaves unexplained, the weather included. It trusts 1
  or 2 steps where the plant allows 2 or 3, so it is cautious by about one step, and it trusts more
  than a log's own plant allows on 0 to 2 % of logs.

  Under the case's rerun call, with the outdoor air and the sun as drivers held at their last
  logged values, the plant allows 2 to 5 steps, cell by cell. 0.16.0 trusted 6 to 13 steps on
  nine logs in ten and held the plant at the first step on 35 % to 51 % of draws. The tube now
  holds it at the first step on 97.6 % to 99.3 % of draws and on 98.6 % of the steps it trusts. It
  trusts 2 or 3 steps on nine logs in ten and 4 at most, and more than a log's own plant allows on
  0 to 2 % of logs. The held forecast's error is not in the tube, but over these few steps the
  weather moves little.
- **On linear plants with known noise** (`tests/test_predictive_tube.py`, 20,000 draws, a Monte
  Carlo standard error of 0.0015 at 0.95). Carried by the state matrix on one state, the tube
  held the plant at 0.948 to 0.952 of draws over 12 steps at a persistence of 0, 0.8 and -0.6.
  Carried by a norm bound, it held 0.950 at the first step and up to 1.000 later. On two states it
  held 0.979 to 0.986. Under `prescribe`, on a one-unit log whose fitted field leaves a
  confounder's push in its defect, it held 0.949 at the first step and more after.

- The rate is per step, not joint over the prefix. Each step holds the plant with probability at
  least 0.95. The chance that every step of a 10-step prefix holds is lower, and the certificate
  does not state it.
- The tube is cautious on several states. The radius gives every direction the largest
  variance's share, and a norm bound grows the noise at the field's steepest rate even where the
  field contracts. On the quickstart's two states, the plant stays within the tolerance of 0.5 at
  95 % for 3 weeks, measured on 20,000 runs, and the certificate trusts 1, where it trusted 15.
- The noise is the one-step error given the drivers the log holds. A driver's forecast past the
  log is the caller's (`Driver`), and its error is not in the tube.
- The defect is in-sample. With hundreds of transitions and a few dozen coefficients, it reads
  the out-of-sample error low by their ratio, a few percent.
- `CausalDynamicsFit` gains `defect`, the `(N, n)` array. `DecisionCertificate` gains
  `tube_noise` and `tube_persistence`. `causal_plan` and `RecedingHorizon` gain `noise` and
  `noise_persistence`. No field changes meaning, except that `uncertainty_tube`,
  `certified_horizon` and `trustworthy_steps` under `prescribe` now carry the noise.
- Tests: `tests/test_predictive_tube.py`.

## Alternatives

- **The estimation tube under an honest name, beside a predictive figure, with the trust horizon
  read off the predictive one.** Rejected. The horizon an operator acts on would be the same as
  the one chosen here. But the plan would carry two tubes and two horizons, and every reader of
  `uncertainty_tube` and `certified_horizon` (`certified_actions`, `certificate_status`, the
  report, `to_json()`, the case studies) would have to pick one. The estimation tube on its own
  bounds the gap of a noise-free plant, which no plant is, so reporting it beside the predictive
  one invites the reading this ADR removes. Its information is already in the certificate:
  `identification_radius` is the channel's error. A caller who wants it can still call
  `causal_plan` with no `noise`.
- **Empirical quantiles of the log's k-step errors, conformal or by blocks.** Rejected for now.
  They need rollouts of the fitted model along the log under the logged actions, not the plan's.
  On one unit with a persistent error they also need the blocks that ADR 0075 needed for the
  channel's error, at every horizon. The Gaussian AR(1) recursion is exact on a linear plant, and
  the coverage above is measured on a plant it does not describe exactly.
- **A centred covariance.** Rejected: it drops the defect's mean, a bias the plan carries forward
  at every step. Where the fit's drift carries a constant, as `prescribe`'s does, the in-sample
  mean is zero to rounding (about 1e-17 of the defect's root mean square on the tests' logs), and
  the two agree.
- **A level the caller sets.** Not added. 0.95 is the rate at which the case study read "inside
  the tube", and the radius is `sqrt(chi2_n(level))` times the same root, so a level is a later
  addition that changes no shape.
