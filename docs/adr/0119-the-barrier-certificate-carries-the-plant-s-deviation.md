# ADR 0119 — The barrier certificate carries the plant's deviation from the model's path

**Status:** accepted, 2026-10-21. Amends ADR 0101 (what a certified step reads), ADR 0111 (what a
step certified under a sensitivity promises) and ADR 0112 (what the held solve holds at a next
state). Amended by ADR 0132, 2026-10-25: the room is `h - ||grad h|| e - κ e²/2`, `κ` the
barrier's curvature, a bound on a quadratic `h` or under the caller's `curvature`.

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
