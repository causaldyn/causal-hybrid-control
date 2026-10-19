# ADR 0118 — A discrete-time panel is planned by its one-step map

**Status:** accepted, 2026-10-19. Supersedes the refusal of `prescribe(integrator="euler")` made
earlier in the 0.19.0 cycle: the fit and the plan still never read two maps, and `"euler"` now
plans by the map it fits.

## Context

Up to 0.18.0, `prescribe` passed `integrator` to the fit and to nothing else. The plan, its tube,
the barrier's audit and the regret bound stepped the fitted field by RK4. An Euler fit planned by
RK4 is a bias: on a discrete-time plant it missed the plant by 0.27 at the first step of six,
outside a tube of 0.063 that certified all six. The first fix refused `"euler"`, and said that a
discrete-time panel is fitted by `"rk4"` as well, as the field whose RK4 step reproduces the log's
one-step map.

That is false where the log decays fast. RK4's growth factor `1 + z + z^2/2 + z^3/6 + z^4/24` is
never smaller than 0.2704, its value at `z = -1.596` (`validation/euler_tube.mac`). No field
stepped by RK4 keeps less of a linear mode from one step to the next, and the `rk4` fixed point
raises. A weekly panel whose outcome barely persists cannot be logged at a finer `dt`. DCBO's
non-stationary SCM (Aglietti et al., NeurIPS 2021) keeps `-1` of `Y` a step after its change
point, and a prescription on its log stopped running.

## Decision

- `prescribe(integrator="euler")` fits the log's one-step map and plans by it: the field is
  wrapped in `chc.dynamics._EulerStep`, and `chc.integrate.rk4_step` steps a wrapped field by
  `x + dt f(t, x, u)`. Every rollout, linearisation, gradient, held barrier row, regret bound,
  misspecification envelope and evaluation reads that one function, so none can step the field
  by another map. The wrapper sits inside the rule a lever the log set from the state follows,
  which sets its action at the step's start.
- The tube reads the map from the plan's model (`chc.plan._integrator`): its step matrix is
  `I + dt A` and its one stage puts the weight `I` on the field's error, in the norm bound, the
  state-matrix tube, the noise's covariance and the read's offset alike. The noise recursions of
  `var_noise_tube.mac` and `noise_read_tube.mac` hold for any step matrix.
- A driver is read at each step's start, as the Euler fit reads it, in the field's error the tube
  carries.
- `"rk4"` stays the default. Any other value than the two is refused. The rk4 fit's error names
  `prescribe(integrator="euler")` as the remedy for a discrete-time panel.

## Consequences

- On a panel that keeps 0.1 of its state a period, the tube holds the plant at each of six steps
  0.9499 to 0.9505 of the time over 64 logs of 4000 runs. Carried by RK4's step matrix of the
  same slope, it holds 0.965 to 0.969 from the second step on, over 16 logs of 2000.
- The barrier's audit reads the field at each step's start. On an affine margin the Euler step
  moves `h` by exactly `dt grad h . f`, so the condition at a step's start is the discrete-time
  condition on the step: `h` after it is at least `(1 - alpha dt) h`.
- On DCBO's three SCMs a prescription runs again. Its mean total regret over ten logs moves from
  0.573, 5.743 and 7.508 (an Euler fit planned by RK4) to 0.512, 5.966 and 8.715. The model class
  is misspecified there: a plan that reads the fitted map is not a better plan of the true SCM.

## Alternatives

- **Keep refusing `"euler"`.** Rejected: a panel that barely persists then has no fit at all.
- **Pass an `integrator` argument to every planner and audit.** Rejected: about fifty call sites
  step the field, and one that is missed plans by another map. A field that carries its own map
  leaves one place to read it.
- **A field whose RK4 step is the Euler map.** Rejected: none exists for a step that keeps less
  than 0.2704 of a mode, which is the case to cover.
