# ADR 0111 — Γ* from a sensitivity model of the channel, read at each step

**Status:** accepted, 2026-10-19. Amends ADR 0085: its share is a relative margin, and a
`ChannelSensitivity` gives an absolute radius, read at each step's predicted state, in its place.
Amends ADR 0101: with a sensitivity, each step's `Γ*` reads the radius at that step's state.

## Context

ADR 0085 made the barrier's radius a share of the fitted channel:
`Delta = (Γ-1)/(Γ+1) · cvar_gap · ||B_hat(x_0) D||_2`. At every `Γ`, however large, the radius
stays within `cvar_gap · ||B_hat(x_0) D||` of the estimate. Nothing in it reads the channel's
error. On a plant whose effect is 1, fitted at 0.004, a plan was certified at all 60 steps at
`Γ` 1, 2, 10 and 1e6, and `Γ*` was `inf`. The true plant left the set at the plan's third state,
least `h` -3.99.

The factor `(Γ-1)/(Γ+1)` is the bounded density ratio's factor on a mean. A channel radius spent
as `d ||u||` is a structural model of its own, and no derivation tied the two. The docs called
`Γ` the marginal sensitivity model's level all the same.

The radius was also read once, at `x_0`. ADR 0085 rejected a radius per step: a ball per step is
not one identified set, and the held solve would differentiate a spectral norm along the path.

Over 200 simulated logs, a hidden confounder moved each log's levers within a density ratio of
`Γ_0 = 1.5`. The truth lay inside the radius derived below at every log. The share's certified
prefix still took the true plant out of the set on 27.5 % ± 3.2 % of logs, and on 26.0 % ± 3.1 %
with a channel linear in the state.

## Decision

- **`ChannelSensitivity`** carries an absolute radius on a fitted channel. `from_fit(fit, level)`
  reads it off an Euler fit by adjustment: the channel's design on the action residuals `V`, the
  fit's moment residual `R`, and `channel_region`'s radii at `level`.
- **The model.** Given the state and the adjustment set, a hidden confounder `Z` moves the
  levers' density by a factor within `[1/Γ, Γ]`, the continuous marginal sensitivity model.
  Nothing bounds how much of the rate's noise `Z` carries. `validation/channel_bias_bound.mac`
  derives the bound in four steps:
  1. Bhatia–Davis: `Z` explains at most `ρ² = (Γ-1)²/Γ` of an action residual's variance.
  2. The omitted-variable algebra: the bias `β` of the coefficients has
     `β'Vβ <= κ² E r²`, `κ² = ρ²/(1 - ρ²)`.
  3. `κ(Γ) = (Γ-1)/sqrt(3Γ - Γ² - 1)`, with a closed-form inverse. It is infinite from
     `Γ = (3 + √5)/2 ≈ 2.618`: from there the levers may be a function of the confounder.
  4. An exact discrete example meets the bound, least slack 2.1e-4.
- **The radius at a state `x`** is `Delta(x, Γ) = ν(x) (sampling + κ(Γ) confounding)`.
  `ν(x)² = λ_max(D Φ(x)' V⁻¹ Φ(x) D)` is the width of one coefficient ellipsoid at `x`;
  `confounding = sqrt(λ_max(R'R))`; `sampling` is the radii's norm. `Γ = 1` is the sampling error
  alone, not exact identification.
- **Read at each step's predicted state**, in the audit and in the held solve. A step's `Γ*`
  inverts the radius in closed form: `κ = (c/(n ν) - sampling)/confounding`, then `Γ(κ)`. It is
  nan where the sampling error alone exceeds what the step can absorb.
- **Opt-in.** `certify_safety(sensitivity=)` and `BarrierConstraint(sensitivity=)` take it. A
  level at or past `(3 + √5)/2` raises. Without it the share of ADR 0085 stays, and the docs call
  it a relative margin. `SafetyCertificate.radius_model` says at run time which radius an audit
  read: `"relative"`, the share, or `"absolute"`, the sensitivity model's.
- **`prescribe` keeps the relative margin**, and its summary now says so. Its fit is `rk4`, whose
  fixed point carries the drift's confounding into the channel through the step's inverse. No
  bound on the levers' density limits that, so no absolute radius is derived for it.

## Consequences

- The example above, on a fitted log: a confounder at `Γ_0 = 2` drove the estimate to 0.003 for
  an effect of 1, and the true plant left the set at the third state, least `h` -3.99. The share
  certified 60 of 60 steps at `Γ` 1, 2 and 1e6. The absolute radius certified none at `Γ` from 1
  to 2.5, and refuses 1e6. Held at `Γ = 2`, the plan certified 60 of 60 steps and kept the true
  plant 0.64 or more inside.
- On the 200 logs above, the absolute radius certified no step of the unheld plan, which the
  truth left the set on, and `Γ*` stayed `inf`: an action that keeps the barrier exists, and the
  plan did not take it. Held at `Γ = 1.5`, every plan certified 30 of 30 steps and none left the
  set on the truth. The price is room: the least true `h` along the held path had median 0.505,
  and 0.536 with a channel linear in the state.
- **Read at each state, the radius holds where the plan leaves the log.** A confounder whose push
  grows with the state, `c Z x`, puts its bias into the channel's slope, so the channel's error
  grows away from the logged states. Over 100 logs, read along states 0 to 3, the radius at each
  state held the truth at every state, and the radius frozen at the start at 89.1 % ± 0.9 % of
  them, 98.9 % ± 0.3 % along states 0 to 2. The widest state's radius was 3.36 times the start's.
  On plants whose bias does not move with the state, both held the truth at every state of the
  held path, the widest step 1.16 times the start's.
- **The set holds the drift at its estimate.** It prices the channel's error only. On a plant
  whose drift pushes out of the set (`xdot = 0.5 x + u`, a confounder at `Γ_0 = 1.2`), a held
  plan rides the boundary. Over 100 logs the drift's own error took the truth out of the set on
  25 % ± 4.3 % of them under the absolute radius, least `h` -0.73, and on 33 % ± 4.7 % under the
  share. The plant with the fitted drift and the true channel stayed in on every log, under both.
  The drift's error is the tube's question.
- The default does not move, and no `FutureWarning` is raised: `certify_safety` takes a model, not
  the fit a sensitivity is read from, so there is no default to move to.
- `ν(x)` is the top eigenvalue of an `m × m` matrix. With one lever it is smooth. At a repeated top
  eigenvalue JAX's gradient stays finite (checked at two and three levers). Where two eigenvalues
  cross, the radius has a kink, as a maximum does. The rounds already hold maxima as columns.

## Alternatives

- **Rename only.** It would leave no `Γ` that bounds a channel's error.
- **The share's factor over a floor of `z` standard errors, with an absolute gap.** The factor
  still has no model behind it for a channel, and the caller must calibrate the gap.
- **A two-parameter model, the confounder's partial R² with the lever and with the rate.** At the
  rate's R² of 1 it is this bound. A second parameter tightens it where the rate's share is
  bounded, but `Γ*` inverts one level, not a curve.
- **A share per step, `||B_hat(x_k) D||`.** Still relative: it reads no error.
- **The absolute radius frozen at `x_0`.** It missed the truth at 10.9 % of the states above.
  ADR 0085's first objection does not apply to the radius per step: it is one coefficient
  ellipsoid, read at each state, not a ball per step.
