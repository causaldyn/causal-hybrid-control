# ADR 0141 — Under RK4 the tube reads the field's error where each stage reads the field

**Status:** accepted, 2026-10-27. Amends ADR 0133 (the tube's field part, under `rk4`) and the
`rk4` caveat of ADR 0138's plan-action form. Moves no default; the Euler map reads as before.

## Context

The tube's field part states that the fitted field's error moves the plant's next state by no more
than its standard error, read at the planned state and any action of the levers' box (ADR 0133).
`_field_errors` read that error at the state the step starts from, and `_linear_tube` carried it
by the weights `c_1 .. c_4` RK4's four stages put on the field's errors.

That is the step's error only where the field's error does not move with the state. One RK4 step
of a field affine in the state moves with the field's error `H(x, u) theta` by

    dt sum_i c_i(z) H(x_i, u) theta,    z = dt A,

`x_i` the stage's own state: the step's start, its two midpoints and its end
(`validation/rk4_stage_term.mac` (1)). Read at the start alone, the tube drops
`dt sum_i c_i (H(x_i, u) - H(x, u)) theta`. On one state that is `dt^2 S'(z) e1 f(x, u)`, with
`S = sum_i c_i`, `S'(0) = 1/2`: the field's move over the step times the error's slope in the state
(`rk4_stage_term.mac` (2)). It is of the order of `dt^2`. It is not one of the terms the `rk4` fit's
error inherits through its fixed point (ADR 0133): those make the covariance, and every stage reads
it. ADR 0133's decomposition `V_cond + L^2 V_b + 2 L cross` holds at each stage's state
(`rk4_stage_term.mac` (4)). The Euler map has one stage, the step's start, and drops nothing
(`rk4_stage_term.mac` (3)).

The start's read can be too wide or too narrow. At the rational point of
`validation/rk4_inherited.mac`, `h a = -0.7`, it reads 1.48 of the step's error where the stages
move toward the log's states, and 0.86 and 0.89 where they move away. That file's exact oracle
is least squares' one-step prediction. On 400 logs of the exact flow of `x' = -x/2 + u` at
`dt = 0.2`, fitted by `rk4`, the box's form at a plan's first action was 0.88 of the plan-action
form (ADR 0138, median). Its misses had a root mean square of 1.19, and `+-1.96` held 0.89 of the
logs. So ADR 0138 had to say that under RK4 the plan's form "can be the wider".

## Decision

- **Under RK4 each stage reads the field's error at its own state.** Given the plan's model,
  `_field_errors` returns `(H, 4)`. Step `k`'s stage `i` is read at the state RK4 reads the
  field at when the step starts from the planned state with a vertex of the box held. It uses the
  drivers at the stage's time: the step's start, its middle twice, its end. Each stage takes the
  largest over the vertices and over the channel's covariances on its own. The stage states come
  from the fitted field. A lever set by the log's rule takes its rule's action in the stage states,
  as `rk4_step` does, and its box's vertex in the error, as before.
- **`_linear_tube` carries each stage by its own weight.** Step `j`'s stage `i` reaches step `k`
  through `R(dt A_{k-1}) ... R(dt A_{j+1}) c_i(dt A_j)`. The tube sums each stage's gains' norms
  times its own error. That bounds the step's error by Minkowski's inequality, as the gains' norms
  already bound the steps' sum (`rk4_stage_term.mac` (5)). Equal stage errors give the one error's
  tube, to rounding.
- **A norm bound reads the stages' largest.** Its premise is a field off by at most its budget
  wherever the step reads it (`time_varying_rollout_bound`).
- **The Euler map is untouched.** Its tube, certificate and per-log spread are 0.22.0's bit for bit
  on every case compared: the bilinear and misspecification logs, box and plan forms.
- **`_field_error`, the one number the plan is solved with, is unchanged.** It is read over the
  log's rows, where no step is planned. The solved plan's tube is re-read along its steps, as
  before.

## Consequences

- **Exact-flow Monte Carlo**, 400 logs (above), the box's form at the plan's own actions:
  - First step: root mean square of the misses / share within `+-1.96`, start's read → stages'
    read: 1.07 / 0.917 → 1.03 / 0.930 on a held plan, and 1.19 / 0.892 → 1.00 / 0.938 on the
    plans that start at `+1`.
  - Over the held plan's 15 steps: 1.06 to 1.11 / 0.915 to 0.943 → 0.96 to 1.05 / 0.930 to 0.955.
  - The plan-action form reads 1.01 to 1.05 / 0.925 to 0.953.
  - The stages' read was never below the plan-action form, at any step of any of the 1200 plans
    (least ratio 1.0013; medians 1.008 to 1.048 before the plans turn).
  - Where the actions change sign it stays 1.45 to 2.54 times the plan-action form (start's read
    1.37 to 2.67): the cancellation across steps is the plan form's to read (ADR 0138).
- **ADR 0138's `rk4` caveat is retired.** Under either integrator the plan-action form is never
  wider than the box's form at the plan's actions.
- **The tube can narrow.** Where the stages move toward the log's states, the error at the stages
  is smaller than at the start. On the misspecification suite's plant (`dt = 0.1`) the last step's
  radius moved 0.7984 → 0.7978 and 0.8170 → 0.8161 on two logs. On `tests/test_plan_action_tube.py`'s
  bilinear log (`dt = 0.5`, box `+-2`) it widened 0.7015 → 0.7856 at step 12.
- **The misspecification suite**, every case that reads the tube, under `rk4`, same logs and runs.
  Weakest step, before → after: control 0.94937 → 0.94937 (16 logs); dead time 2, 0.93688 →
  0.93691 (16 logs); a saturating response 0.5176 → 0.5271 (4 logs, held by its `lever_range`
  warning); persistent demand 0.94834 → 0.94834 (16 logs); a persistent disturbance 0.90225 →
  0.90250 (warned on every log); a sine drift 0.8875 → 0.8875 at step 15 (warned on every log).
  Any step's share moved by at most 0.0004 except under saturation (up to +0.032) and the
  persistent disturbance (+0.0015). The trustworthy steps and the warnings are the same on every
  log; with the noise read's error removed under persistent demand the weakest step still falls by
  0.0018. These plants step by the Euler map at `dt = 0.1`, where the stage term is small. No
  tolerance moves.
- **The vertices.** Where the channel does not read the state, a stage's state moves affinely with
  the action, and its error's variance is a convex quadratic in it, largest at a vertex
  (`rk4_stage_term.mac` (6); `tests/test_rk4_stage_tube.py` over 401 actions). Under a channel
  that reads the state, the default, the third stage's state is quadratic in the action and the
  variance is of degree 6 in it, so a vertex is not proven the largest. On the bilinear log and
  three exact-flow fits, at every planned and probed state, the largest over 401 actions of
  `[-2, 2]` was the vertices' to six digits. The step matrices that carry the error are read at the
  plan's actions, as before.
- **Cost.** The stage states take one vectorised RK4 step for each vertex and planned step. The
  field's error is read four times as often.

## Alternatives

- **Read the step's exact error at each vertex jointly**, `sqrt(tr(v Sigma v'))`,
  `v = dt sum_i c_i H(x_i, u)`. It is tighter by the Minkowski slack over the stages: 0.1 % to 1.4 %
  at `rk4_stage_term.mac`'s points. But it is carried by `||G||` where the stage weights now carry
  each stage by `||G c_i||`, which is a second carry for one integrator. Not taken.
- **Add only the leading term `dt^2 S'(z) e1 f(x, u)`.** It is exact on one state only, and needs
  the error's slope in the state by hand. The stage read is exact on any field affine in the state,
  at the same cost.
- **Read every stage at the largest stage error**, the norm bound's premise, for the exact tube
  too. It reads every stage at the error of the one farthest from the log's states, wider than
  the step's error wherever the stages' errors differ. Not taken.
- **Keep the start's read.** Its misses read 1.19 of it at a plan's first action, holding 0.89 where
  it states 0.95.
