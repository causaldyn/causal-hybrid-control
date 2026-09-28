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
    lag one of its MA(1) error. The plug-in also tests the first-order state it rests on, by
    whether `y_(t-1)` and `u_(t-1)` add to its regression, and warns when the data reject it at 1%.
    The channel at an i.i.d. lever is not tested: it is read right on any linear time-invariant
    plant. A lever that alternates every period leaves only `y_(t-1)` to test, since
    `u_(t-1) = 1 - u_t`, and hides a stock it drives, which then alternates with it;
  - the local projection with the `H − 1` lags of its overlapping sums;
  - for the steady state, Fieller's interval, which is unbounded when the data cannot rule out a
    unit root;
  - the block difference in its Horvitz–Thompson form. Each block is centred on the midpoint of
    the on and off blocks before it, which its own coin cannot move, so it is unbiased at any
    number of blocks. The block scores are martingale differences in time. Each zone's difference
    is its own and the zones are weighted alike, and the zones' scores are summed block by block
    before squaring, so the standard error keeps what a spillover or a shock common to the zones
    puts between them.

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
- **88 tests.** Each of 36 mutations is caught: 15 in the design, 21 in the reading. Dropping the
  local projection's lags survived the coverage tests, since their noisy case moves the standard
  error by only 2.3%, and a test at noise 4 and `H = 5` now catches it.
- **On a plant the working model does not describe** (`scripts/bench_switchback.py`, 8000 runs a
  reading). The plant is the zone market of `chc.zones`, and CHC wrote it, so every number here is
  by construction. Each zone's incentive is switched between 0 and 0.3 by its own coins, and the
  reading is its idle supply. The incentive also acts through a stock, so the state is not first
  order. Half of what a zone recruits comes from its two neighbours, and the zones' channels
  differ. The prior is fitted to an i.i.d. pilot of 200 000 periods a zone. The truth comes from
  branching the runs: from a run's own state, a zone's lever forced on and forced off. The
  numbers are for linear matching, then harmonic.
  - **The plug-in, on the trusted plans, is biased.**
    - `tau_5` at its aligned design reads −2.8% and −2.9%; its intervals cover 0.79 and 0.77.
    - At the design for the channel and the steady state, the channel reads +13.9% and +13.8%
      (covered 0.018 and 0.019) and the steady state +8.6% and +9.5% (covered 0.27 and 0.20).
    - The first-order test warns on 99.9–100% of these runs. On the working model it warned on
      0.8–1.3% of 2000 runs at each of six designs and persistences, against its 1%.
  - **Model-free, the readings are unbiased,** within 0.34%, and their intervals cover 0.944–0.953.
    The plan's variances are the working model's, though, and a test at the planned MDE rejects
    (±0.005):
    - for the channel, whose spread is 0.99 of the plan's, 0.801 and 0.799;
    - for `tau_2`'s local projection, spread 1.04 times as far as planned, 0.774 and 0.771;
    - for `tau_5`'s, 1.06 times as far, 0.742 and 0.737;
    - for the steady state's block difference, 1.73 and 1.64 times as far, 0.426 and 0.459.
      Neighbouring zones' readings correlate at +0.25 and +0.24. Without the spillover it
      spreads 0.94 times as far and rejects 0.90.
  - **So the 0.9.0 gate, realised power within three points of nominal, holds here for the channel,
    and at its edge for `tau_2`,** 2.6 and 2.9 points under. `tau_5`'s local projection misses it
    by 5.8 and 6.3 points and the block difference by 37 and 34.
  - **Under harmonic matching** the truth sits 0.2–1.6% above the linearised one, and the
    model-free readings still read it within 0.3%.
  - **The block difference's standard error, before the change** squared the scores zone by zone.
    It was 0.88 and 0.89 of the spread there, and covered 0.915 and 0.918 (2000 runs). Summed
    block by block it is 0.994 and 0.993, and covers 0.944 and 0.945. On the working model on a
    ring of four, half of each push felt by the neighbours, the old form was 0.77 of the spread.
    With channels 3, 2, 2 and 1 and no spillover it was 2.0, since pooling the blocks weighted each
    zone by how soon its coins showed both settings.

## Not built

- **A plan whose variances see a second state or spillover.** They are the working model's, and on
  the zone market they put the local projections' power 3–6 points under nominal and the block
  difference's 34–37 under. An internal pilot would carry them: in a lab run (4000 runs, linear
  matching), re-reading each standard error from the first quarter of the run and restating the
  MDE brought the local projections within 2.1 points of nominal and put the block difference
  4–5 over, since its early blocks are centred on fewer blocks before them.
- **Shocks common to zones, and spillover between them, in the regressions.** Their standard errors
  pool the zones as independent replications. On the zone market, summing their scores across
  zones moved them by under 1%.
- **A plug-in for a state of higher order.** The test says when the first-order plug-in is biased;
  the effect is then read by the model-free designs.
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
- **Pooling the blocks of every zone.** It weights a zone by how soon its coins showed both
  settings. That weight moves from run to run and carries the zones' differences into the spread.
- **Squaring the block scores zone by zone.** It misses the dependence spillover puts between
  zones: coverage 0.915 on the zone market.
