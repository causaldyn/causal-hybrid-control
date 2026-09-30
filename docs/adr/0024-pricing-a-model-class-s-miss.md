# ADR 0024 — Pricing a model class's miss in the plan's regret

**Status:** proposed, 2026-09-30.

## Context

`fit_causal_residual(weights=...)` fits the channel under a weight on the state. When the class
holds the truth, every weight estimates the same parameters and only the variance moves. When it
misses, each weight estimates its own projection. Two fits of one class on one log therefore land
apart by a symptom of the miss, which is what a Hausman test reads. A plan's question is narrower:
whether the difference costs it anything. A difference along a direction the plan does not move is
harmless, and one along the direction it leans on is not.

Three pieces were missing.

- **The metric.** `CausalPlan.decision_weight` prices an error in the channel, `W = J' M^-1 J` by
  the envelope theorem, with the drift held. But the drift is fitted on what the channel leaves of
  the rate. Two fits whose channels differ by `dB` have drifts that differ by the log's projection
  of `dB u` on the drift's features, so their models differ by `dB (u - uhat(x))`, not by `dB u`.
- **The difference's covariance.** Both fits read the same rows, so their errors are correlated, and
  neither weighting is efficient. Neither the sum of their covariances nor Hausman's difference of
  them is the difference's covariance. That needs each row's influence on each fit, and
  `CausalDynamicsFit` carried a scalar `channel_error`.
- **What the log cannot say.** A direction of the channel the log's actions never moved has no data
  in either fit. The ridge sets it in both, so they agree there by construction, whatever the truth.
  A deployed plan's own log, with no dither, can be such a log: the Berk–Nash trap.

## Decision

- **`misspecification_cost(plan, reference, alternative)`**, experimental, at
  `chc.misspecification`.
  - `d` is the difference in every fitted parameter: the channel raveled, then the drift's transpose
    raveled.
  - `W` is the regret's curvature in those parameters at the plan: the machinery of
    `decision_weight`, factored out of it as `_regret_curvature`. The model is perturbed by a
    residual of the fits' own class inside its field and rolled out by RK4, as the plan is.
  - `S` sums, over rows and states, `(psi_alt - psi_ref)(psi_alt - psi_ref)'`. It holds whether the
    class does or not, and whichever folds each fit drew.
  - The cost is `(dh' W dh - tr(W S)) / 2`, since `E[dh' W dh] = d' W d + tr(W S)`. Its standard
    error is `chc.gate.ChannelMove.price`'s.
  - When the class holds, `d = 0` and `dh' W dh` is a chi-square mixture weighted by the
    eigenvalues of `W S`. `p_value` is its tail by Imhof's inversion. Past `u = 20 / omega` the
    integrand is a slowly moving amplitude times a pure oscillation, and QUADPACK's Fourier rule
    takes that part to the end.
  - `unseen` counts the unmoved directions the plan's regret weighs: the eigenvalues of `V' W V`
    above `sqrt(eps) ||W||`, with `V` the fit's `unmoved`.
- **`fit_causal_residual(influence=True)`** keeps `influence`, `(N, n, p)`:
  `psi = sqrt(n / (n - k)) (J e + D (r - e))`.
  - `J` is the whole fit's response to the row's rate, and `e` the moment's residual.
  - `D` is the part of the response that reaches the drift directly, not through the channel: the
    drift regression's own weight on the row. `r` is that regression's residual.
  - Under `rk4` both terms are carried through the fixed point's `K^-1`, as the covariance carries
    the noise.
- **Every fit carries `unmoved`, `(p, r)`.** These are the channel directions along which the
  actions, less their least-squares projection on the nuisance's features, keep less than
  `sqrt(eps)` of their raw size. Each comes with the drift regression's response to it on the log.
- **Not in `prescribe`'s JSON.** That schema is a type-1 door, and it opens once, with the decision
  record 1.0 fixes.
- **It refuses:**
  - fits made without `influence`;
  - fits that carry drivers, whose gain's price needs the plan's forecast;
  - two fits whose classes, integrators or rows differ;
  - a plan not made on the reference fit's model;
  - whatever `decision_weight` refuses: a barrier-held plan, and one that minimised a pessimism
    penalty.

## Consequences

The bench is `scripts/bench_misspecification.py`: a two-state plant whose channel moves with the
first state at a given slope, fitted by the constant-channel class unweighted and weighted by
`exp(x_0)`.

- **The metric** (section `metric`). On logs whose actions averaged −1, 0 and +1, the channel's
  price alone read 0.14, 0.44 and 2.8 of the regret between the two fits' plans. The quadratic in
  every parameter read 0.49, 0.54 and 0.49.
- **The reach** (section `reach`). The quadratic is second order in the difference. It read about
  half of the regret at the full difference, which was two thirds of the channel's size; 0.74–0.78
  of it at three tenths; and 0.89–0.91 at a tenth. At a large difference the cost is a floor.
- **The influence reads the drift through its own residual.** A first draft read the drift through
  the moment's residual. Over whole logs redrawn under a missed class, the drift's difference then
  spread 2.2 times what the influences said. The projection `L Phi` changes with the log, and only
  the drift regression's residual holds what the drift's features leave. The test now holds the
  ratio within a quarter of 1, and within a fifth for every parameter when the class holds.
- **`influence` under `rk4` is not `channel_error`.** It carries the drift's residual into the
  channel through the RK4 map, which `channel_error` leaves out, since it holds the noise alone
  random. That moved the channel's error by half a percent on the test's log. Under Euler the two
  agree to 1e-9.
- **Calibrated over whole logs** (section `calibration`, 200 logs of 4000 rows).
  - With the class holding the truth, the test rejected 1.0%, 6.0% and 11.5% at 1, 5 and 10%.
    `cost_error` was twice the cost's spread there: the leverage term, estimated from `dh`, is
    positive in about half the logs. It is a scale, not coverage.
  - With the class missing, the cost fell within 1.96 `cost_error` of its value on 400 000 rows in
    0.915 of the logs. It read low by a quarter of `cost_error` on average (−0.26 ± 0.08), and
    spread 1.13 times it.
  - `noise` matched the difference's realised spread in `W` to 1.1 of its standard errors.
- **Power** (section `power`). At 5% it caught 13%, 30%, 78% and 97% of the logs as the slope went
  0.005, 0.01, 0.02 and 0.03. The logs of `metric` have slope 1.
- **The zone plant** (section `zones`, 0.9.0's verification). Two zones of `chc.zones`' market are
  fitted under `rk4`, unweighted and weighted by `exp` of the first zone's standardised supply. The
  plan raises every zone's idle supply by a fifth over 12 periods. The plant was detuned from the
  class two ways, one at a time. Curvature bends the trips towards the harmonic law, which they
  follow at 1. Saturation makes an incentive recruit less the more drivers are idle, so the channel
  moves with the state.
  - Where the class held (400 logs), the test rejected 0.5%, 3.25% and 7.5% at 1, 5 and 10%. The
    p-values' Kolmogorov distance from uniform was at most 0.056, inside the 0.068 that 400 draws
    allow at 5%.
  - Saturation 0.1, 0.2, 0.4 and 0.8 (100 logs each): at 5% it caught 5%, 5%, 9% and 32% of the
    logs. The regret between the two fits' plans went 0.0047, 0.0056, 0.0082 and 0.0145, against
    0.0048 where the class held.
  - Curvature 0.5 and 1: 1% and 2%. The regret between the fits' plans stayed at 0.0046–0.0049.
    The harmonic law, which the class misses too, did not move the cost of choosing between the
    fits, so there was nothing to find.
  - `cost + noise`, the quadratic's mean reading, matched the mean regret between the fits' plans
    within 1% at every setting. At differences this small, second order holds.
  - Planning on the reference fit lost 0.38–0.51 against the truth at every setting, including
    where the class held. That is 32 to 106 times the regret between the fits. It is the error both fits
    share, which their difference cannot see. The gate prices the choice between fits, not either
    fit's distance from the truth.
- **What it cannot see** (section `unseen`). The log's second action was always twice the first,
  and the class held the truth. The gate read `p = 0.33` and two unseen directions, and the plan
  lost 0.55 against the truth. Held to the log's ratio, the plan had none unseen and lost `5e-6`.
- **The trap `unmoved` names.** A log whose actions the covariates' features determine leaves every
  direction unmoved. In float64 the fit was identified and read the channel as `0.0009 ± 0.0013`
  against a true 0.8. In float32 the rounding left in the actions is read as data instead, and the
  error is large. The flag is the same in both.
- **What `influence` costs.** It is `N n p` numbers. Under Euler it takes a reverse pass per
  parameter rather than per channel coefficient. That is why it is opt-in. `unmoved` is one
  least-squares projection and one SVD of the channel's design, so every fit carries it.

## Not built

- **Fits with drivers**, **barrier-held plans** and **pessimism-weighted plans**.
- **What to do on a failure.** The function reports. Refitting, dithering (AD2) or keeping the plan
  is the loop's call (AP3).

## Alternatives considered

- **The channel alone, through `decision_weight`.** Rejected: on three logs of one plant it read
  from 0.14 to 2.8 of the regret, because the drift moves with the channel.
- **Hausman's `Var(alt) - Var(ref)`, or the sum of the two.** Rejected: the first needs one fit to
  be efficient, and the second needs the fits to be independent. Neither holds here.
- **A bootstrap of the difference.** Not needed: the influence gives the covariance in one pass,
  where a bootstrap refits both fits per draw, under `rk4` each through its fixed point.
- **`unseen` from the null space of `S`.** Rejected: `S` is the covariance of a difference.
  - It vanishes wherever the two fits agree by construction, not only where the log is silent. A
    weight close to constant makes it small in every direction.
  - On the pair log above, the unmoved directions did sit at 5e-14 of its largest eigenvalue,
    against 4e-5 for the next one. That gap belongs to this pair of fits, and not to the log.
  - Whether the actions moved a direction is a fact about the log, and a single fit needs it too,
    as the trap above shows.
- **The cross-fit's ridge residuals for `unmoved`.** Rejected: the ridge leaves a bias that grows
  as the log shrinks, and it would hide an unmoved direction in a short log. The full-sample
  least-squares projection leaves rounding, whose threshold holds at 60 rows and at actions of
  size 1e-4.
- **Satterthwaite's approximation, or Monte Carlo, for the p-value.** Rejected. One or two terms
  are the hard case for the tail: a plain rule on `[0, inf)` read 0.0036 against 0.00053 for one
  term at 30 times its weight. Imhof's inversion is exact up to the quadrature. It is checked
  against `chi2.sf`, and against a convolution for two unequal weights.
