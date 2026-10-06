# ADR 0019 — Re-reading a moved channel

**Status:** proposed, 2026-09-29.

## Context

The drift monitor (ADR 0018) says when a deployed plan's one-step channel has left its
identification radius, and `CausalPlan.decision_weight()` says what an error in that channel costs
the plan: `vec(E)' W vec(E) / 2`. Nothing read how far the channel had moved, so nothing could put
a number on what the move costs, or on what re-planning would.

The logged dither that powers the monitor is also an instrument for the move itself. A decision
applies `u_j = p_j + dither_j`, with the dither drawn after everything before it. Against the model,
the residual is `r_i = c + sum_k d_ik dither_k`, where `d` is the plant's channel less the model's
and `c` holds everything else: the model's error in the drift, the nominal action through `d`, and
the noise. So `E[r_i dither_j] = d_ij dither_scale_j^2`, whatever `c` holds.

## Decision

- **`chc.gate.channel_move(log, residual, *, dither_scale, forgetting=1.0)`** returns a
  `ChannelMove`: the estimate of `d`, its covariance and its effective size.
  - The estimate is the mean of the products `r_i xi_j / dither_scale_j`, with `xi` the
    standardised dither. Each product is unbiased for `d_ij`
    (`validation/dither_channel_move.mac` STEP 1; `proofs/dither_channel_move.v` (A)).
  - The weights are `forgetting^k`, with `k` the number of decisions since, so the last decision
    weighs 1. Re-read as decisions arrive, the estimate follows a channel that moves, and is worth
    `(1 + forgetting) / (1 − forgetting)` decisions once the log is long (STEP 3).
  - The covariance is the products' own, weighted the same way. It is scaled by the exact factor
    that makes it unbiased when the products share one variance, at any weights (STEP 6). The
    products' errors are a martingale difference sequence, since each dither is drawn after
    everything before it. So the scale holds for noise that moves with the state, and is exact
    under equal weights.
  - Its refusals are the monitor's: no dither, a clipped decision, a dither the chi-square test
    rejects at its stated scale, and a malformed residual. Two more: fewer than two decisions, and
    a `forgetting` outside `(0, 1]`, or one that leaves every weight on the last decision. The
    dither check is one helper, `_standardised_dither`, shared with `channel_drift_evalues`.
    Since ADR 0021 the monitor reads a clipped decision, and `channel_move` still refuses one: a
    clip scales its product by the chance that the draw was not clipped.
- **`ChannelMove.price(weight)`** returns a `MovePrice` from the plan's decision weight. It holds
  two expectations:
  - `keep`: what keeping the plan loses to the plan that knew the move, `d' W d / 2`, estimated
    without bias as `(dh' W dh − tr(W S)) / 2` (STEP 4a; Rocq (B)). It is not clipped at zero,
    because clipping would bias it up. `keep_error` is its standard error, from STEP 4b's variance,
    with `d' W S W d` debiased by `tr((W S)²)` (STEP 4c) and floored at zero.
  - `replan`: what a plan re-solved on the estimate loses on average, `tr(W S) / 2` (STEP 5).
- **No decision rule.** STEP 5 and Rocq (C) state when re-planning pays for an oracle that knows
  `d`. The same comparison made on the log the estimate came from is not shipped: the measurement
  below refutes it.
- **Experimental** until 0.14.0's gate, like the monitor it reads the same dither as.

## Consequences

Measured by `scripts/bench_channel_move.py`, with `JAX_ENABLE_X64=1`. There is no wall time.

**On the drift monitor's plant** (2000 paths). The model's drift is off in slope and intercept,
the noise is Laplace, and the channel sits 0.07 from the model's.

| dither, decisions | bias / SE | spread / error | `± 1.96 SE` covers |
|---|---|---|---|
| 0.3, 500 | −0.09 | 1.025 | 0.943 |
| 0.3, 2000 | +0.02 | 1.009 | 0.952 |
| 1.0, 500 | −0.75 | 1.007 | 0.955 |
| 1.0, 2000 | +0.73 | 0.991 | 0.952 |

With the channel at 1.4 for the last 1000 of 4000 decisions, each estimate follows its own target,
the weights' average of the move. Without forgetting that target is 0.153; at 0.995, which is
worth 399 decisions, it is 0.398. The intervals covered it on 0.960 and 0.951 of the paths. The
bias read +2.1 SE on the bench's seed, and between −0.60 and +1.26 SE on three more.

**On the budgeted plan of `tests/test_decision_weight.py`** (200 logs of each size). This plan
has two states, two levers, a binding box and a budget, `W` with seven free directions, and none
weakly active. Its one-step channel is off by 10–15% of each entry. Always keeping the plan loses
0.095, where the keep price at the true move reads 0.107.

| decisions | keep price | re-plan price | always re-plan | prices compared | shrunk | split |
|---|---|---|---|---|---|---|
| 30 | 0.117 ± 0.052 | 0.367 | 0.184 ± 0.016 | 0.149 | 0.119 | 0.132 |
| 100 | 0.111 ± 0.019 | 0.106 | 0.080 ± 0.006 | 0.100 | 0.077 | 0.110 |
| 400 | 0.093 ± 0.008 | 0.026 | 0.024 ± 0.002 | 0.043 | 0.039 | 0.072 |
| 1600 | 0.104 ± 0.004 | 0.0066 | 0.0060 ± 0.0006 | 0.0060 | 0.0072 | 0.018 |

- **The keep price is unbiased for the quadratic, and the quadratic overstates the regret by the
  cubic term.** It averages 0.093–0.117 against the quadratic's 0.107, while the realised loss is
  0.095.
- **The re-plan price is an expectation that saturates.** From 30 decisions it read 0.37 against
  a realised 0.18: a re-solved plan cannot leave its box, however wrong the estimate. From 400 and
  1600 decisions it read 1.12 and 1.10 of the realised regret. The remainder is the re-solved
  plan's own curvature, which differs from the kept plan's by the move's order.
- **As expectations, the two prices order the choice correctly.** Keep at 30 decisions (0.37
  against 0.12); re-plan at 400 and 1600, where always re-planning is best. At 100 they tie, and
  re-planning is better by 0.015.
- **As a rule on one log, they do not.** Re-planning when a log's keep price beats its re-plan
  price lost to the better of the two fixed choices at 30, 100 and 400 decisions. The logs whose
  estimate reads a large move are the ones whose estimate errs most along `W`, and those are the
  ones re-planned.
- **Two repairs were measured, and neither holds up.**
  - Re-planning on the estimate shrunk by `max(0, 1 − tr(W S) / dh' W dh)`. This is the optimal
    linear shrinkage in `W`'s metric when `d' W d` is known; here it is estimated. It won at 100
    decisions by less than its standard error, and lost at 30, 400 and 1600.
  - Choosing on one half of the log and re-planning on the other, so the choice does not select on
    the error it re-plans with. It lost everywhere, because half a log prices the keep too noisily
    and re-plans on half the data.
- **What stands:** re-read the move on decisions logged after the choice, such as those after a
  `DriftAlarm` sounds. Then the re-plan price is the re-solved plan's.
- **Checked:**
  - `validation/dither_channel_move.mac` runs STEPs 1–6;
  - `proofs/dither_channel_move.v` has 7 lemmas, on Stdlib's classical reals and nothing else;
  - `tests/test_channel_move.py` has 24 tests: a transcription, the bias and covariance by Monte
    Carlo with and without forgetting, the prices, the refusals, and a real plan's moved channel
    read in the entries its `W` prices;
  - 24 mutations of the estimator and the prices, all caught.

## Alternatives considered

- **Regressing the residual on the dither, with an intercept and the dither's own scale.** This
  estimate does not need the stated scale, and the intercept takes the mean of `c` out of its
  variance. Rejected for now: its ratio form is biased at `O(1/n)`, and an exactly unbiased
  estimate with the monitor's chi-square guard on the scale is the trade ADR 0018 made. The
  intercept's variance gain is worth measuring on a plant whose model has a large constant drift
  error.
- **`kish / (kish − 1)` as the covariance's scale.** Rejected: it is exact only for equal weights,
  and read 13% low at eight decisions forgotten at 0.7 (`tests/test_channel_move.py`).
- **`dh` for `d` in the keep price's error.** Rejected: it reads the variance three times too
  large on a channel that has not moved. The debiased form with its floor reads 1.31 of the spread
  there, and 1.04 on a channel that moved.
- **Refitting the residual with the dither as its instrument.** Deferred:
  `fit_causal_residual(instrument=...)` would refit the drift and the channel's state dependence as
  well. That is a new identification, not an update, and it has no forgetting.
- **A decision rule** in any of the three forms above. Refuted by the table.
