# ADR 0018 — A channel-drift monitor

**Status:** proposed, 2026-09-28.

## Context

The deployment gate (ADR 0010) holds a zone, or rolls it back, when drift e-values the caller
supplies raise an alarm. Until now nothing in the library produced them. What a deployed plan most
needs watched is its channel: the plan's actions were chosen for the channel the logs identified,
and a channel that moves makes the plan's regret grow.

Two things make the usual monitors unfit:

- **A monitor of the residual watches everything the model gets wrong, not the channel.** In the
  lab, at `A = 10⁴`, a residual-mean CUSUM alarmed by 3000 decisions on 0.083 of paths on an
  unchanged plant. It alarmed on 0.813 with the drift model off by a further 0.15, and on 0.010
  with the channel at 1.4 instead of 1.07.
- **The model's channel is itself an estimate.** A monitor of "the channel equals the model's"
  alarms on the identification error.

A plan deployed with a logged Gaussian dither carries an instrument: the dither is independent of
everything but the action it perturbs. ADR 0015 made the dither a field of the logged decision,
and EV1 evaluates the smoothed plan that carries one (D27). The lab found an exact e-value built
on it (AG5), and an independent verifier checked it.

## Decision

- **`chc.gate.channel_drift_evalues(log, residual, *, dither_scale, radius, residual_scale)`**
  returns per-decision e-values for "every entry of the one-step channel lies within `radius` of
  the model's". The inputs:
  - `residual`: the next state less the model's one-step prediction at the action as applied;
  - `dither_scale`: the dither's standard deviation per action;
  - `radius`: a bound per entry;
  - `residual_scale`: fixed before the data, the unit the bets are placed in.

  For state `i` and action `j`, `xi` is the dither over its scale and `r` is the residual moved to
  the radius's edge and scaled: `(residual_i ∓ radius_ij u_j) / residual_scale_i`. The e-value is
  `exp(theta r xi − theta² r² / 2)`, with `theta = ±1, ±1/2, …, ±1/128` and the sign given by the
  side watched. On a plant affine in the action over one step, `r = c + k xi`. When `xi` is
  `N(0, 1)` and independent of `c`, the identity `E[· | c] = 1/|1 − theta k|` makes each column an
  e-value on its side of the composite null, whatever the model's error in the drift, the noise
  law or the policy.
- **Every entry, both sides and every bet is a column**, `states × actions × 2 × 8` of them. The
  mixture is taken over the detectors' statistics, never over one decision's e-values. Averaging
  the bets decision by decision would pay `log 8` at every decision rather than once.
- **`chc.gate.DriftAlarm(arl)`** runs e-Shiryaev–Roberts, `R_t = (R_(t−1) + 1) e_t` per detector,
  and alarms when the detectors' average reaches `arl`. That average less `t` is a supermartingale
  under any dependence between the detectors, so the average run length on an unchanged channel
  is at least `arl`. `DeploymentGate` now runs one per zone, so the library holds one
  implementation.
- **Refusals**, each at the boundary:
  - a log with no dither;
  - a log with a clipped decision (`DecisionLog.dither_draws`), because skipping clipped decisions
    after the draw selects on `xi`;
  - a dither whose draws, over the stated scale, a two-sided chi-square test rejects at `1e−9`.
    That catches a slip in units, or a variance passed for a standard deviation. For a hundred
    zones read every fifteen minutes, the threshold makes a false refusal about once in 285 years
    per action;
  - a negative radius, a scale that is not positive, and a residual without one row per decision.
- **`radius` has no default.** A default of 0 is the naive detector, the one that failed.
- **Experimental** until 0.11.0's gate: the monitor's type-I error under continuous monitoring,
  with its detection delay reported.

## Consequences

Measured by `scripts/bench_drift.py`, on 300 paths of the lab's plant. The model's drift is off in
slope and intercept, the noise is Laplace, and the channel sits 0.07 from the model's, inside a
radius of 0.1. Run lengths are censored at 80 000 decisions, so each mean is a lower bound.

| | `A = 10³` | `A = 10⁴` |
|---|---|---|
| unchanged channel, with the radius: run length / `A` | ≥ 3.23 | ≥ 5.60 (41% censored) |
| the same, drift model off by a further 0.15 | ≥ 3.46 | ≥ 5.40 (43% censored) |
| unchanged channel, no radius | 0.96 | 0.36 |
| the same, drift model off by a further 0.15, no radius | 0.92 | 0.34 |
| channel at 1.4 from the start: mean delay (median) | 259 (237) | 456 (411) |
| the oracle, which knows the entry, the side and the best bet | 167 (147) | 332 (278) |

- **The guarantee holds where it was meant to**: under a wrong drift model and non-Gaussian noise,
  and with the drift wrong by more. Without the radius, the alarm comes after 0.34–0.96 of its
  target. These numbers reproduce the lab's within noise: 3.58 and 5.39 with the radius, 0.97 and
  0.36 without it, and delays of 258 and 468.
- **Not knowing which entry moved, which way, or at what rate costs 1.56× the oracle's delay at
  `10³` and 1.37× at `10⁴`.** The price falls as `A` grows, because the `log 16` the mixture over
  16 columns adds to the threshold is a smaller share of `log A`.
- **A channel that moves less than the radius is invisible**, and the radius is subtracted from
  every move. A wider radius buys validity at the price of delay.
- **It cannot check its own conditions:**
  - a response nonlinear in the action: in the lab, a running product passed 20 on 27% of paths
    under `xi² − 1`, where Ville's inequality allows 5%;
  - a dither mis-scaled by less than the chi-square test can see: a fifth more variance than
    stated puts an e-value's mean at 1.105 at `theta r = 1`, where Maxima reads
    `exp(theta² c² (v − 1)/2)`;
  - a radius that does not cover the identification error.

  The docstring says so, with these numbers.
- **A plan whose actions clip cannot be watched this way.** The plan's candidate fix is to skip,
  before the draw, the decisions whose nominal action lies near a bound. It is not built.
- The gate's own drift behaviour is unchanged, and its 28 earlier tests pass as they were.
- The identity is checked three ways:
  - `validation/dither_drift_evalue.mac` derives it in the library's parameterisation (both sides,
    the edge, the scales, the second moment, a mis-scaled dither), with quadrature at the tests'
    columns;
  - `proofs/dither_drift_evalue.v` proves its algebra and the averaged Shiryaev–Roberts step
    (17 lemmas, on Stdlib's classical reals and nothing else);
  - `tests/test_gate.py` checks every column's mean against `1/|1 − theta k|` by Monte Carlo.

## Alternatives considered

- **A monitor of the residual**: a CUSUM on its mean, or a run of misses outside `SplitConformal`'s
  intervals. Rejected: it measures the model's whole error, as the lab's 0.083 → 0.813 shows, and
  it has no instrument for the channel.
- **e-CUSUM.** Rejected: at the same threshold e-Shiryaev–Roberts alarms no later on every path
  (`cusum_below_sr`). For a Gaussian mean shift from the start at `A = 10⁴`, the lab's exact
  computation put it 101.8 and 15.8 decisions earlier at bets of 0.25 and 0.5.
- **A likelihood-ratio detector on the whole model.** Rejected: it needs the noise law and a
  correct drift model, the two things this detector is built not to need.
- **Estimating the dither's scale from the log.** Rejected: the standardised draw is then not
  exactly `N(0, 1)`, which is the one thing the identity needs. The chi-square test only guards the
  stated scale against slips.
- **A full dither covariance, whitened.** Deferred: it would test the entries of the channel times
  the covariance's factor, not the channel's own. A smoothed plan's dither is `tau² I` (EV1), so
  independent components cover the plans the library deploys.
