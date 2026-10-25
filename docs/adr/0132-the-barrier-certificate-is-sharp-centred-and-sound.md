# ADR 0132 — The barrier certificate is sharp, read from the log's mean action, and sound on a curved barrier

**Status:** accepted, 2026-10-25. Amends ADR 0111 (the channel's bias factor and its refusal),
ADR 0091 (the barrier read from each lever's zero), ADR 0002 (`Γ*`'s closed form), ADR 0119 (the
room a certified step reads), and ADR 0125 and ADR 0129 (the refusal of a level under the absolute
radius).

## Context

Three parts of the barrier certificate were looser or less sound than their mathematics allows.
All three set the one closed form `Γ*` reads.

1. **The channel's bias factor was not sharp.** ADR 0111 bounded the share of an action
   residual's variance a hidden confounder `Z` explains by `ρ² = (Γ-1)²/Γ`, the Bhatia–Davis
   bound on the density ratio read at each value of `Z` alone. The factor `κ = ρ/√(1 - ρ²)` was
   then infinite from `Γ = (3 + √5)/2 ≈ 2.618`, and every level from there on was refused, with
   the reason that the levers may be a function of the confounder. At no finite level can they
   be: the ratio's lower side, `p(u | z, x) >= p(u | x)/Γ`, splits the joint law as
   `P = Γ⁻¹ P_U ⊗ P_Z + (1 - Γ⁻¹) Q`, `Q` a law with `P`'s two marginals. So no function of the
   levers correlates with one of `Z` beyond `1 - 1/Γ`, their maximal correlation (Rényi 1959),
   and `ρ² <= (1 - 1/Γ)²`, `Γ` times smaller. A lever of two values that takes the confounder's
   sign with probability `1 - 1/(2Γ)` meets it.
2. **The barrier read each lever from its zero** (ADR 0091, Consequences). A fit regresses the
   drift on the log's actions, so its drift holds the channel at the log's mean action, and the
   field's error at an action `u` is the drift's own plus the channel's times `u - u_bar`. The
   audit read the channel's part as `Δ ||∇h|| ||D⁺u||`, the action set and `u_max` from zero. On
   the `free` log bounded to `[0.2, 1.1]`, `u1`'s zero moved by 100, its log and box with it,
   `prescribe` read `Γ*` 1.04 where it read `∞`.
3. **The room was first order.** A step was certified where `h - ||∇h|| e >= 0` at its state and
   the next, `e` the deviation tube. That is the least `h` within `e` of the state on an affine
   `h`, and above it on an `h` that bends down: on the disk `1 - ||x||²` by exactly `e²`. On an
   integrator with an exact radius of 8/13, one action of 13/20 and `h = 1 - x²`, the step was
   certified, and the plant at the radius's edge reached `h = -41/400`. Held in the solve the
   same way, a plan pushed against the disk was certified on 20 steps of 20 while that plant
   reached `h = -0.056`.

## Decision

1. **The sharp factor.** `ChannelSensitivity.factor(Γ) = (Γ-1)/√(2Γ-1)`, finite at every finite
   level, growing as `√(Γ/2)`; `level_at(κ) = 1 + κ² + κ√(1 + κ²)`. Only `Γ = ∞` is refused.
   `validation/channel_bias_bound.mac`: the splitting (STEP 1), the factor and its inverse (STEP
   3), the attainment (STEP 4). The bound reads `Γ` only through `ρ`, so a Gaussian lever whose
   confounder explains a share `R²` of its variance reads as `Γ = 1/(1 - √R²)`.
2. **The barrier read from the log's mean action.** `certify_safety(centre=)` and
   `BarrierConstraint(centre=)` take the action every action is read from: the spread is
   `Δ ||∇h|| ||D⁺(u - c)||`, `u_max` each lever's largest distance from `c`, the deviation tube's
   push `||D⁺(u_k - c)||`, and the best action of the set `c` plus a step along `D(Dw)`. `Γ*`
   inverts the threshold `d* = ||Dw|| - (-αh - a_c)/U`, `a_c` the drift at `c`
   (`validation/barrier_centred.mac`). `prescribe` passes the log's mean action. `None` reads
   from zero, as before. The centre sets where the spread is measured from, not which actions are
   admissible: those stay the box's. Where the centre is outside the box -- `prescribe`'s lever
   selection pins a lever at 0 that the log moved -- the threshold above, which reads `c` as an
   action, does not count. A step's `Γ*` is then the largest over the box's actions it reads: the
   plan's own, the box's action nearest `c`, and from that one along the set's best direction on
   the levers the box frees, out to the box's edge.
3. **A sound room.** The room is `h - ||∇h|| e - κ e²/2`, `κ` a bound on `λ_max(-∇²h)` within
   the deviation, in the audit and at the next states the held solve holds. By Taylor's theorem
   it is at most the least `h` a plant there can have. `κ` is read from `h`: 0 where its Hessian
   and third derivative are zero along the path, the largest eigenvalue of `-∇²h`, clipped at 0,
   where its Hessian is one matrix there and its third derivative is zero. The caller's
   `curvature=` replaces it. On any other `h` the room stays first order.
   `SafetyCertificate.room_model` reads `affine`, `quadratic`, `bounded` or `first_order`
   (`validation/barrier_room.mac`).

All three are fixes: each moves a number toward the guarantee the docs state. The radius stays a
bound, now a sharp one; the audit reads the field's error where the fit puts it; the room is a
bound where the docs said it was one.

## Consequences

- `κ` narrows by 1.05, 1.12, 1.26, 1.73 and 4.0 at `Γ` 1.1, 1.25, 1.5, 2 and 2.5, so every `Γ*`
  read under `sensitivity=` rises. `κ = 1`, which was `Γ = 2`, is `Γ = 2 + √2 ≈ 3.41`.
- The misspecification test of the default absolute radius read `Γ = 2`, where the radius at the
  start, 0.52 to 0.55, held the plant's channel, 0.39 to 0.41 from the fitted one. At `Γ = 2` the
  radius is now 0.34 to 0.36 and misses it. The log's incentive is a Gaussian lever with half its
  variance from the confounder: it reads at `Γ = 2 + √2`, where the radius is 0.52 to 0.55 again
  and the held plan is certified on every step.
- On the cross-module panel at `gamma = 1.5` (`prescribe`, both changes): `free` bounded to
  `[0.8, 1.1]` read `Γ*` 9.81, now 10.51, held 9.78, now 10.47; `state` bounded to `[0.2, 1.1]`
  read 12.16 and certified no step, now 11.80 and one; held 13.06, now 11.97.
- ADR 0091's example reads `Γ* = ∞` on the moved log, as on the unmoved one. One plan, its actions
  moved with the zero and audited on each log's fit, reads the same certified steps, `Γ*` and
  deviation to `1e-4`. `prescribe`'s plan on the moved log still clears no barrier step: the plan
  itself differs, since its cost prices an action from the lever's zero. That is the cost's
  reading, not the certificate's, and it is not changed here.
- A centre outside the box no longer reads `Γ*` at an action the box excludes. With one lever
  pinned at 0 and its logged mean at -1, a plan that clears every step at `gamma = 1` read `Γ*`
  1.61, and nan where the drift at the centre failed the step; it now reads 1.91,
  where the box's best action just clears. On the cross-module panel the change moves one `Γ*`
  by one ulp.
- A level past `(3 + √5)/2` is read under the absolute radius; `prescribe` no longer raises there.
- Every barrier `prescribe` passes is an affine margin, as are the BOPTEST track's and the docs'
  examples: `room_model` reads `affine` and their numbers do not move, bit for bit on the
  cross-module panel. The quickstart, `00_start_here` and `09_heating_a_room` print the same.
- The deviation tube itself stays first order where the model is not affine in the state or the
  channel's error moves with it; `room_model` says nothing of that.
- `SafetyCertificate` gains `room_model`, last, with the default `first_order`.

## Alternatives

- **Keep the per-value share and lift only the refusal.** The factor would stay `Γ` times too
  wide in `ρ²`, and a certified level would read low by the same 1.05 to 4.0.
- **Centre at the box's midpoint.** It reads no lever's zero either, but the field's error does not
  vanish there; a log at one edge of its box would be priced as if half the box away.
- **Detect the curvature by sampling the Hessian over the deviation ball.** It is a sample, not a
  bound, and a pocket between the samples breaks it. A quadratic `h` needs one Hessian; any other
  needs the caller's bound to be sound.
- **Read a non-quadratic `h` at its Hessian along the path.** Second order, not a bound, and it
  moves the numbers of a curved `h` without making them sound: `first_order` says what they are.
