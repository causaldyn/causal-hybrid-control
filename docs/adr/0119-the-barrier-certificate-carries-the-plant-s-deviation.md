# ADR 0119 — The barrier certificate carries the plant's deviation from the model's path

**Status:** accepted, 2026-10-21. Amends ADR 0101 (what a certified step reads), ADR 0111 (what a
step certified under a sensitivity promises) and ADR 0112 (what the held solve holds at a next
state). Amended by ADR 0132, 2026-10-25: the room is `h - ||grad h|| e - κ e²/2`, `κ` the
barrier's curvature, a bound on a quadratic `h` or under the caller's `curvature`.
Amended 2026-11-06, re-measured on 1.1.0 on the calibrated logs of the Consequences, each plan held
and certified under the fit's own sensitivity at `Gamma = 1.5`: the promise is the one the
Decision narrowed it to, about a plant with the model's drift. On the unstable drift (0.5) the
plants that differ from the model only in the channel, the true channel and both edges of the
radius, stayed in the set over the certified prefix on all 20 logs, least `h` 0.1985, 0.0000 (the
edge the tube meets exactly) and 0.8752, and the tube was never narrower than the gap it bounds;
19 plans are certified on all 30 steps and one on 29, all 20 ending `max_iterations`. The true
plant left the set on 1 of 20 logs, least `h` -0.086, where the Consequences read none of 20 on
the plans of 2026-10-21. Its drift is missed by the fit's sampling error, `drift_error` 0.094 to
0.098 on every log, the largest miss 1.71 of it, which the radius does not carry: that error is
the error tube's question (ADR 0111). On the stable drift (-1) all 20 plans converge and are
certified on all 30 steps, and every one of those plants, the true one too, stays in the set, the
true plant's least `h` 0.34. A misspecification test holds the channel-only promise on the 20
unstable logs, and `certify_safety(sensitivity=)` states it with these numbers.
Amended 2026-11-07: the room may read a bound on the drift's error, opt-in (last section); with
none it reads the channel's error alone, as before.

## Context

`certify_safety` read the barrier condition `a + <w, u> - d ||D^+ u|| >= -alpha h` at the
*model's* predicted states, against every channel within the radius, and `h >= 0` at each state
and the next. Its docs promised that a plan certified there keeps a plant whose channel is in the
radius in the set. It did not. A plant whose channel is not the model's is not at the model's
states: under the same open-loop actions its path drifts away from them, and the condition read
at the model's state says nothing about the plant's.

With no fit involved, a known drift and an exact absolute radius:

- drift 0.3, `dt = 0.1`, channel 0.5021, radius 0.1004: a held plan was certified on all 60
  steps, and the plant at the radius's edge reached `h = -0.132`;
- an integrator at `dt = 1`, channel 1, radius 0.5: certified on all 12 steps, the plant reached
  `h = -0.500`.

On calibrated fits of an unstable drift (`xdot = 0.5 x + u`, a hidden confounder at
`Gamma = 1.5`), the radius held the true channel on 20 logs of 20, every plan was certified on
all 30 steps, and the true plant left the set on 18 of them, least `h` -3.41.

## Decision

- **The deviation tube.** For a plant with the model's drift and a channel within `Delta_k` of the
  model's at step `k`, stepped by the model's one-step map `F` under the plan's actions,

      e_{k+1} = ||dF/dx|| e_k + ||dF/dc|| Delta_k ||D^+ u_k||,   e_0 = 0,

  `c` a constant added to the field at every stage. Both norms are read off the map the plan is
  rolled out by (RK4, or Euler for a model fitted to a discrete-time log) by forward-mode
  differentiation; for a field affine in the state they are `R(dt A)` and `dt phi(dt A)` for RK4,
  `I + dt A` and `dt I` for Euler (`validation/barrier_deviation_tube.mac`).
- **A step is certified** where the condition holds at its state, and `h - ||grad h|| e >= 0` at
  its state and at the next. `SafetyCertificate.deviation` holds `e_0 .. e_H`.
- **The held solve holds the same column**: a next state's shortfall is
  `-(h - ||grad h|| e)` there, so a held plan keeps the model's path inside the set by the width
  of the tube, and is certified by its own audit.
- **The promise is narrowed to what the tube proves.** It is a bound where the model is affine in
  the state, the channel's error does not move with the state (a relative radius, or a
  sensitivity of `channel_degree` 0) and `h` is affine. Elsewhere it reads the map's slope, the
  radius and `grad h` at the model's states, and holds to first order in the tube's width. It is
  about the channel alone: an error in the drift, and a plant stepped by another map, are outside
  it.
- **`||M||_2` is differentiated as `u_1' dM v_1`**, the top singular pair's, with the top right
  singular vector read off `eigh(M'M)` under `stop_gradient`. An SVD's derivative is nan where
  the top singular value repeats, as on the identity, which an Euler step of a zero drift is.

## Consequences

- The cases above, held under the tube: every plant within the radius stays in the set over the
  certified prefix. Least `h` over the plants at the radius's edge: 0.0925 (drift 0.3, radius
  0.1004), 0.3038 (radius 0.3), 0.1138 (channel 1, radius 0.3), 0.1189 (the same at `dt = 1`),
  0.0000 (the integrator, which the tube meets exactly), and all still certified on every step.
- The calibrated unstable logs: the true plant left the set on none of 20 logs, least `h`
  +0.061; 14 of 20 plans were certified on all 30 steps, the rest on 18 to 29.
- On one state, one lever and actions of one sign, the plant at the radius's edge attains the
  tube exactly: no bound of this form is smaller. On several states it is a norm bound and can be
  loose where the map is far from normal.
- `gamma = 1` with a relative radius is a radius of 0: the tube is 0 and nothing changes for the
  ordinary barrier.
- **The held solve is slower and stops short more often.** The tube couples each next state to
  every earlier action, and `||D^+ u||` has a kink at zero where the optimum may rest a lever:
  on drift 0.3 over 60 steps the rounds ended on their budget (`max_iterations`) where they
  converged before, 3 to 4 times slower. The audit, not the status, says what holds.
- `gamma_star` does not read the tube: it asks whether some admissible action clears each step's
  condition, at the model's state. It can be `inf` beside no certified step; the certificate's
  docs and `Prescription.report` say so.

## Alternatives

- **Narrow the promise to closed loop**, a plan re-solved at each step from the plant's own
  state. It would leave every open-loop plan without a guarantee, and the receding-horizon loop
  re-plans from a measured state the audit never sees.
- **A Lipschitz tube** (`chc.uncertainty.time_varying_rollout_bound`) with one rate for the whole
  path. It reads no contraction and grows on a stable drift, where the step's own Jacobian
  shrinks the gap.
- **The worst plant by optimisation** over channels in the radius. Exact for one fixed channel
  error, but a non-convex program per audit, and not differentiable for the held solve.
- **A smooth upper bound on `||u||`**, `sqrt(||u||^2 + tau^2)`, in the held solve. Tried at
  `tau = 1e-3`: the rounds still ended on their budget, so the kink is not what stops them.

## Amended 2026-11-07: the room may read the drift's error

The re-measure of 2026-11-06 left the true plant outside the set on 1 of 20 unstable logs, its
drift missed by the fit's sampling error. The fit gives that error, `drift_error`, a standard
error of the drift's fitted value with its root mean over the log's rows and the states, and the
error tube reads it (ADR 0104), but the room did not.

**Decision.**

- A bound `epsilon` on the drift's error at the centre, in the rate's units and the Euclidean norm
  over the states, enters the field as the channel's error does: a constant added at every stage.
  The tube becomes

      e_{k+1} = ||dF/dx|| e_k + ||dF/dc|| (Delta_k ||D^+ (u_k - c)|| + epsilon),   e_0 = 0.

  It is exact where the model is affine in the state and the error does not move with it: the
  plant whose drift is `+epsilon` from the model's, at the radius's edge, attains it on one state
  (`validation/barrier_deviation_tube.mac` (5), RK4 and Euler). An error `alpha x + epsilon` that
  moves with the state is read at the model's state, to first order in the tube's width; on an
  Euler step the term it misses is `dt alpha g`, `g` the gap (6). On a drift `a` and idle actions
  the tube is `dt phi(dt a) epsilon (R^k - 1) / (R - 1)`, `R = R(dt a)` (7).
- It is opt-in: `certify_safety(drift_error=)`, `BarrierConstraint(drift_error=)`, and
  `prescribe(barrier_drift_error=True)`, which passes `t sqrt(n)` times the fit's `drift_error`,
  `t` Student's at 0.975 on the channel's degrees of freedom (ADR 0104) and `n` the states. With
  none, the tube's push gains no term and every number is the one before, to the bit.
  `SafetyCertificate.drift_error` and `DecisionCertificate.barrier_drift_error` record the bound.
  `prescribe` refuses the flag without a constraint, and on a fit that gives no drift error (an
  observational fit, whose drift is read conditional on an unidentified channel).
- The condition, and `gamma_star` with it, still read the model's drift: the error reaches the
  certificate through the room alone.

**Consequences.** The calibrated logs of the Consequences, at `Gamma = 1.5`, RK4, each plan held
and certified under the fit's own sensitivity and, where it reads it, the bound `t sqrt(n)
drift_error` (0.185 to 0.192 on every log). Prefixes are certified leading steps of 30, without
the bound and with it; "held" plans are held under the bound, "re-audited" ones are the plans held
without it, audited with it; the true plant's least `h` is over the certified prefix.

| log | unstable, held | cost | status with it | true plant's least `h` | unstable, re-audited | stable, re-audited |
|---|---|---|---|---|---|---|
| 0 | 30 → 28 | +4.2 % | `infeasible` | +0.11 → +0.84 | 30 → 15 | 30 → 16 |
| 1 | 30 → 25 | +7.2 % | `max_iterations` | +0.50 → +0.97 | 30 → 16 | 30 → 15 |
| 2 | 30 → 19 | +4.4 % | `max_iterations` | +0.82 → +0.99 | 30 → 17 | 30 → 14 |
| 3 | 30 → 22 | +5.0 % | `max_iterations` | +0.60 → +0.99 | 30 → 17 | 30 → 20 |
| 4 | 30 → 22 | +1.4 % | `infeasible` | +0.48 → +0.92 | 30 → 17 | 30 → 19 |
| 5 | 29 → 20 | +4.1 % | `max_iterations` | +0.72 → +0.98 | 29 → 17 | 30 → 17 |
| 6 | 30 → 28 | +7.8 % | `infeasible` | +0.10 → +0.84 | 30 → 15 | 30 → 19 |
| 7 | 30 → 26 | +8.5 % | `max_iterations` | +0.44 → +0.98 | 30 → 16 | 30 → 16 |
| 8 | 30 → 25 | +3.9 % | `max_iterations` | +0.31 → +0.92 | 30 → 16 | 30 → 25 |
| 9 | 30 → 26 | +5.0 % | `infeasible` | +0.15 → +0.84 | 30 → 15 | 30 → 18 |
| 10 | 30 → 23 | +3.8 % | `max_iterations` | +0.60 → +0.99 | 30 → 17 | 30 → 18 |
| 11 | 30 → 22 | +4.6 % | `max_iterations` | +0.67 → +1.00 | 30 → 17 | 30 → 17 |
| 12 | 30 → 26 | +5.4 % | `infeasible` | +0.12 → +0.82 | 30 → 15 | 30 → 20 |
| 13 | 30 → 24 | +4.9 % | `max_iterations` | +0.50 → +0.97 | 30 → 16 | 30 → 17 |
| 14 | 30 → 29 | +7.5 % | `max_iterations` | +0.10 → +0.88 | 30 → 15 | 30 → 23 |
| 15 | 30 → 28 | +7.8 % | `max_iterations` | +0.18 → +0.91 | 30 → 15 | 30 → 27 |
| 16 | 30 → 25 | +5.9 % | `max_iterations` | +0.48 → +0.98 | 30 → 16 | 30 → 18 |
| 17 | 30 → 21 | +4.0 % | `max_iterations` | +0.69 → +0.99 | 30 → 17 | 30 → 15 |
| 18 | 30 → 30 | +11.5 % | `max_iterations` | -0.09 → +0.72 | 30 → 14 | 30 → 26 |
| 19 | 30 → 28 | +6.9 % | `max_iterations` | +0.18 → +0.88 | 30 → 15 | 30 → 18 |
| mean | 29.95 → 24.85 | +5.7 % | | | 29.95 → 15.90 | 30 → 18.90 |

- On the unstable drift (0.5) the true plant stays in the set over the certified prefix on all
  20 logs, least `h` +0.72, where without the bound it left on 1, least `h` -0.09. Every plan
  ended `max_iterations` without the bound; with it 15 do and 5 end `infeasible`, the rounds'
  measure not met where the room is narrower. Plans that end `max_iterations` are iterates the
  budget stopped: their per-log prefixes and costs are this machine's, and the tests hold only
  bounds on them (the stay-in count, a floor on the prefix).
- On the stable drift (-1) every plan held under the bound converges and is certified on all 30
  steps, as without it; the cost is 0.43 % higher on average (0.01 % to 1.30 %), and the true
  plant's least `h` is 0.43, was 0.34.
- A plan held without the bound and only audited with it loses about half its prefix on both
  drifts (to 14 to 17 on the unstable drift, 14 to 27 on the stable one): the held solve, not the
  audit, is where the bound is spent well. The true plant stays in the set over the re-audited
  prefix on every log of both.
- The bound is one number, read at the log's rows; a plan that heads away from them meets a
  larger drift error (ADR 0104), which the room does not see. `t` is a pointwise level, not a
  simultaneous one over the path.
- A misspecification test asserts the stay-in on the 20 unstable logs at a rate of 0.9, a floor of
  13 by ADR 0115's binomial rule (`validation/misspec_power.mac` step 6: a block at 0.9 fails
  with probability 4.2e-4; the floor has power 0.8 against a stay-in rate of 0.53 or less). 20
  logs cannot tell a rate of 0.9 from one of 0.95.

**Why opt-in.** On the unstable drift the bound costs 5.1 certified steps and 5.7 % of the cost on
average, and turns 5 plans `infeasible`. Reading it by default would move every barrier
certificate a caller holds; that is a change of the default, which needs a release's evidence and a
deprecation cycle first.

**Alternatives.**

- **Read the drift's error per state**, from the fit's drift blocks at the plan's states, as the
  error tube does (ADR 0104). Tighter near the log and larger away from it, but the room would
  then depend on the plan's states through the error, a new derivative path through the held
  solve; left for a later release.
- **Read the drift's error into the condition**, as the channel's radius is. It would move
  `gamma_star`, which is the channel's sensitivity level alone (ADR 0111).
- **A simultaneous multiplier** over the path in place of `t`. Wider; on these logs the pointwise
  level already kept the true plant in on all 20, and the prefix is what a wider bound costs.
- **Calibrate the bound on held-out horizons** (`chc.uncertainty.HorizonConformal`). It needs
  horizons held out of the fit, which `prescribe`'s log does not set aside.
