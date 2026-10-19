# ADR 0085 — Γ*'s gap is a share of the channel

**Status:** accepted, 2026-10-12. Amends ADR 0002: the radius `delta` is now
`(gamma - 1)/(gamma + 1) * cvar_gap * ||B_hat(x_0)||_2`. Amended by ADR 0091, 2026-10-13: under
`prescribe` the radius and the action set read each lever in units of its own largest action.
Amended by ADR 0101, 2026-10-19: `gamma_star` is read on the levers' box, and a step is certified
only where the path is in the safe set at it and the next.

## Context

`certify_safety` prices a plan's barrier against an identification radius on the effect,
`Delta = (Γ-1)/(Γ+1) · cvar_gap`, and `BarrierConstraint` holds the same condition in the solve.
`Delta` is an operator-norm radius on the channel `B`: the guaranteed barrier derivative is
`a + <w, u> - Delta ||grad h|| ||u||`, and `Γ*` inverts it at the radius the best admissible
action can absorb. So `cvar_gap` carries `B`'s units, the state's per the levers'. Its default was
`1.0`, and `prescribe` and `run_spine` passed nothing else: one unit of the state per unit of a
lever, whatever those units were.

`Γ*` and the barrier's certified steps therefore moved with the units of the log. On a log of one
state and two levers, the state bounded to `[0.8, 1.1]` and audited at `Γ = 1.5`, `prescribe`
read `Γ*` 5.28 and one certified step of three. With the levers logged in thousands, or the state
in thousandths, the same problem read `Γ*` 1.0014 and no step; the other way round, every level
and two steps. The channel was 0.8 in one set of units and 8e-4 in the other, against a radius
that did not move.

## Decision

`cvar_gap` is a share of the fitted channel's size: its operator norm at the plan's first state,
`||B_hat(x_0)||_2`, the Jacobian of the model in the actions at `(t = 0, x_0, u = 0)`, where
`certify_safety` reads the channel. The radius is

```text
Delta = (Γ-1)/(Γ+1) · cvar_gap · ||B_hat(x_0)||_2,
```

and a step's ceiling is `Γ* = (g s n + c) / (g s n - c)`, `g` the share, `s` the channel's size,
`n = ||grad h||` and `c` the radius the step can absorb. The default, 1, is the channel's own
size: the calibration `cvar_gap := b_hat` that `chc.benchmark` and `chc.sensitivity`'s recipe
already use, and the convention of `ObservationalCheck.least_gamma`, whose gap is a share of the
predicted lift. `certify_safety`, `BarrierConstraint` (held in the solve at the same radius), and
`run_spine` read it so; `prescribe` passes no gap and gets the default.

One size for the whole audit, read once where the plan starts, keeps the identified set what it
was: one ball around the estimate, the same at every step. The held solve takes it as a value,
not a static field, so a replanning loop whose start moves the channel compiles once.

## Consequences

- `Γ*` and the barrier's certified steps read the same under any common scaling of the state and
  any common scaling of the levers: to 1e-9 in the tests, on a scalar plant whose ceiling is the
  closed form `2 b U / D - 1` and on `prescribe`'s log above, where both now read `Γ*` 10.88 and
  two steps in every set of units. `validation/gamma_star_units.mac` derives the invariance and
  shows the absolute gap's dependence on the units.
- Numbers move wherever the channel's size was not 1. The driver-supply demo's two arms read
  `Γ*` 3.34 and 1.79 where they read 7.48 and 1.17: still apart, by less, since the confounded
  arm's channel, 0.28, now sizes its own radius. The naive arm's certified prefix is 7 steps of 25
  where it was 5. ADR 0007's gate, a band on `temp` under the weather forecast, reads `Γ*` 11.49
  where it read 4.95. ADR 0081's third band, `wait` in units 1, 1e-3 and 1e3 times its own, reads
  `Γ*` 1.220, 1.0002 and 1.609 where it read 1.202, 1.0002 and `inf`: one state's units still
  reshape the ball, as below. Plants whose channel has norm 1, such as `DampedOscillator`, read as
  before.
- A caller who calibrated an absolute operator-norm radius `r` passes `cvar_gap = r / ||B_hat(x_0)||_2`.
- Not covered: levers in units unlike each other, or states unlike each other. The radius is
  isotropic, a ball in the operator norm, and rescaling one lever reshapes that ball; an
  anisotropic set is a different model. A channel that varies with the state is sized at `x_0`
  only. The held solve's own plan still moves with the units, since its rounds descend in the
  caller's units (ADR 0002); its certified steps do not.

## Alternatives

- **Keep `cvar_gap` absolute and have `prescribe` pass the channel's size.** It fixes one caller.
  `certify_safety`, `BarrierConstraint` and `run_spine` keep a default in the channel's units, and
  each caller repeats the computation.
- **Size the channel at each step, `||B_hat(x_k)||`.** The identified set becomes one ball per
  step rather than one around the estimate, and the held solve would differentiate a spectral norm
  along the trajectory, whose derivative is undefined where the top singular value repeats.
- **The largest size along the path.** The same derivative, and a radius that moves with the plan
  it prices.
- **The channel along the barrier, `||B_hat^T grad h||`.** The radius would depend on the barrier,
  and two bounds would price two identified sets for one effect.
- **The Frobenius norm.** Smooth, but not the norm the radius is a ball in: it overstates the
  operator norm by up to the square root of the channel's rank.
- **No default.** It moves the question to every caller without answering it.
