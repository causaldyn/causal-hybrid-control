# ADR 0012 — A switchback for a named effect

**Status:** proposed, 2026-09-28.

## Context

`chc.gate` assumes no carryover, and `design_experiment` plans a one-shot decision. Neither says how
to randomise a lever over time in a zone whose state remembers what the lever did. There the
periods are not independent units, and which design is best depends on which effect is wanted.
Switchback designs in the literature target the switchback average effect under a carryover of
known order (Bojinov, Simchi-Levi and Zhao 2023). A controller needs specific effects instead: the
channel `b` to plan with, the effect `tau_H` of holding the lever on for `H` periods, and the steady
state.

The lab's switchback thread worked this out for a first-order state per zone,
`x_{t+1} = a x_t + b u_t + eps_t`, and an independent verifier checked it with several CAS:

- **One number carries the design.** When the analysis regresses `y_{t+1}` on `(1, y_t, u_t)`,
  the design enters only through `A1 = sum_s a^(s-1) rho(s)`, where `rho` is the assignment's
  autocorrelation.
- **Orthogonality by design.** The plug-in `b_hat S_H(a_hat)` loses `q (S_H' - A1 S_H)^2 / D(A1)`
  against the variance with `a` known, and nothing when `A1 = S_H' / S_H`. At `H = 2` that is a
  Markov design flipping with probability `a / (1 + 2a)`.
- **Several effects.** The minimax design sits between the extreme optima. For the channel and the
  steady state it sits at the midpoint `A1 = 1 / (2 (1 - a))`.
- **Measurement noise.** It makes the regression inconsistent for every `H >= 2`. IV with
  `(u_t, u_{t-1})` is consistent at every design.
- **Without a state model.** `DIM(H, H-1)` and the local projection at `p = 1/2` are unbiased for
  `tau_H` on any linear time-invariant plant. No model-free design is unbiased for the steady
  state.
- **The verifier's corrections:**
  - the scope is `0 < a < 1`;
  - the realised-means difference in means is biased at a finite number of blocks, −27% at eight;
  - the prototype's identity "plug-in − DIM(H, H−1)" is false for finite `H`;
  - near `a = 1` the finite-run variance is 1.4–1.9 times the asymptotic one at `T = 200`.

## Decision

- **`design_switchback(estimands, prior, periods, zones)`** returns a `SwitchbackPlan`: the arms,
  each a design and a share of the zone's periods. For each effect it reports the analysis, the
  standard error, the minimum detectable effect, the bias and the loss against the effect's own
  best design. All are taken at the least favourable `a` of the prior's interval.
  - **With a trusted state model:** one Markov design, at the flip probability that minimises the
    worst loss over the effects and the interval, read by the plug-in. Under measurement noise it
    is read by IV at IV's own minimax. `min_switches` floors how rarely the lever may flip, since
    the steady state asks never to switch.
  - **Without one:** the channel regression at `p = 1/2`; for `tau_H`, `DIM(H, H-1)` or the local
    projection, whichever the working model says is tighter; for the steady state, the block
    difference in means with the smallest worst mean squared error, its bias quoted.
  - **Refusals and warnings.** What the model cannot see comes back as a warning: a second state,
    drift, spillover, a short run near the unit root, few switches. It refuses only `a` outside
    `(0, 1)`, and a target MDE smaller than a quoted bias.
- **`read_switchback(lever, outcome, estimand, analysis)`** reads an effect off the data, with a
  standard error from the data:
  - the plug-in and IV by the delta method on a heteroskedasticity-robust covariance, IV's keeping
    lag one of its MA(1) error;
  - the local projection with the `H − 1` lags of its overlapping sums;
  - for the steady state, Fieller's interval, which is unbounded when the data cannot rule out a
    unit root;
  - the block difference in its Horvitz–Thompson form. Each block is centred on the midpoint of
    the on and off blocks before it, which its own coin cannot move, so it is unbiased at any
    number of blocks. The block scores are martingale differences, so their spread is the
    standard error.

## Consequences

The simulations in `tests/test_switchback.py` are the lab's, vectorised and written apart from the
module. The working model runs at `a = 0.8`, `b = 2`, `sigma = 1`, over 2000 periods a zone.

- **The lab's reference planner, reproduced** on nine plans, to its printed digits and its flip
  probabilities to 1e−7.
- **Planned against simulated** (2000 runs, eleven readings of eight plans):
  - the simulated spread is 0.988–1.040 of the planned standard error;
  - a test at the planned MDE rejects 0.782–0.824 of the time, against a planned power of 0.80.
- **The gate.** For `tau_5` the aligned design beats an i.i.d. one, whose `A1 = 0`, by the
  predicted variance factor: 1.687 predicted, 1.768 measured. The ratio is known to about 3% over
  4000 runs each.
- **The data's intervals** covered 0.937–0.959 of 0.95 on the same readings (1000 runs each). The
  data's standard error was 0.98–1.03 of the spread it claims.
- **The block difference at eight one-period blocks:**
  - centred as above, it reads 2.011 ± 0.011 for a channel of 2;
  - the realised means read 27.2% low, as the verifier measured.

  Centring on the plain mean of the earlier blocks is also unbiased. But it carries the past
  coins' imbalance, which costs `(kept tau / 2)^2 sum 1/b`: the spread was 2.3 times the plan's on
  40 blocks of 50. The midpoint brings it to 1.10 times; the rest is the first blocks spent on
  centring, and the centre's own noise.
- **Near the unit root** (`a = 0.95`, 200 periods), the steady state's Fieller interval covered
  0.930 and the delta method's 0.877. The rest is `a_hat`'s O(1/T) bias. The design and the
  reading both warn when one zone's `sd(a_hat)` passes a tenth of `1 − a`.
- **The local projection's lags** matter only when the state is measured with noise, because
  noise leaves past levers in the error. Dropping them puts the standard error 7.8% low at noise 4
  and `H = 5`; without noise they change it by 0.1%.
- **82 tests.** Each of 27 mutations is caught: 15 in the design, 12 in the reading. Dropping the
  local projection's lags survived the coverage tests, since their noisy case moves the standard
  error by only 2.3%, and a test at noise 4 and `H = 5` now catches it.

## Not built

- **Realised power on a marketplace flagship.** It waits for a zones-by-time marketplace model.
- **Shocks common to zones, and spillover between them.** Zones are pooled as independent
  replications, each with an intercept.
- **A second state or a longer carryover.** The model-free readings, run on some zones, are the
  check.
- **A bias-corrected `a_hat`** for short runs near the unit root.
- **The switchback average effect under a carryover of order `m`.** That stays with Bojinov,
  Simchi-Levi and Zhao's estimators, as an oracle rather than a reimplementation.

## Alternatives considered

- **Rerandomising to `A1 ≈ 0`.** It loses `q S_H'^2 / D(0)`: 69% more variance for `tau_5` at
  `a = 0.8`.
- **The minimax block design for the switchback average effect.** It answers a different
  question.
- **The realised-means difference in means.** It is 27% low at eight blocks.
- **Centring on the plain mean of earlier blocks.** It is unbiased, with 2.3 times the spread at
  40 blocks.
- **The delta method's interval for the steady state.** It covered 0.877 at `a = 0.95` over 200
  periods.
