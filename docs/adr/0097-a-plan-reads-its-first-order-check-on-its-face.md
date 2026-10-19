# ADR 0097 — A plan reads its first-order check on its face

**Status:** accepted, 2026-10-13. Amended 2026-10-13, before any release: the descent reads the
stationarity itself, and does not stop beyond its tolerance while a trial still falls by the
gradients at its ends (the last bullet of the Decision).
Amended by ADR 0102, 2026-10-19: a plan held under a barrier reads the augmented Lagrangian's
stationarity, where it read none.

## Context

`causal_plan` returns a `CausalPlan` with the solver's status and its count of accepted steps.
The solve itself, `projected_gradient_solve` or `pessimistic_solve`, also computes a stationarity
residual `||u - P(u - grad J)||` and the largest excess of a row. `CausalPlan` dropped both, so
`prescribe`'s certificate and its JSON record had neither.

The status does not answer the question either. On `test_plan_units`' problem (one lever, 12
steps, a box of ±5), a descent stopped by `steps = 30` reports `max_iterations`, yet it is
stationary to 1.2e-6 of the descent's own scale. At `steps = 3` it also reports `max_iterations`,
and it is off by 0.065.

The solver's two numbers are in the caller's units. On the same problem the stationarity residual
of the converged plan reads 5.8e-8 with the lever in its own units. It reads 5.8e-5 with the
lever in thousandths and 5.7e-11 in thousands. It reads 5.7e-14 with the cost in millionths and
0.058 with the cost in millions. The plan and its iteration count are the same in each case,
because the descent runs in scaled variables (ADR 0059). A row's excess is in the row's own
units, so a budget in cents reads a hundred times one in euros.

## Decision

- **`CausalPlan` carries four new fields: `stationarity`, `stationarity_tolerance`,
  `constraint_violation` and `violation_tolerance`.** `causal_plan` reads them at the actions it
  returns. They are fields with a default of `None`, so a plan built by hand still constructs.
  `stationarity_status` and `violation_status` are properties that read each number against its
  tolerance: `within`, `beyond`, `not_finite`, or `not_evaluated` (`chc.plan.DiagnosticStatus`).
- **Both are read in the descent's own variables.** `v = sigma * u`, where `sigma_kj^2` is the
  cost's Gauss–Newton curvature along lever `j` at step `k`, over `|J|`. Both are taken at the
  guess, which is zeros clipped to the box (`chc.control._action_units`). The guess does not
  depend on a warm start, so the reading is a function of the problem and the plan alone.
  `v` is a pure number. A lever in other units moves `sigma` inversely, and a cost in other units
  moves the curvature and `|J|` alike. The states do not enter: a problem posed in other units of
  the state is the same function of the actions. Where `|J|` at the guess is zero or not finite,
  `_action_units` takes 1, and the readings are then in the cost's own units.
- **The stationarity is the KKT residual with signs.** In `v`, the gradient is divided by
  `sigma |J|`, and each row by its norm. `_kkt`, the helper that `shadow_prices` and
  `decision_weight` use, finds what binds by its own rule. A bounded least-squares fit (BVLS)
  then gives the least norm of the gradient plus the binding rows' and box sides' pushes. An
  inequality row and a side of the box push only outward. An equality row pushes either way, and
  so does a box of zero width, which fixes its action: `prescribe` holds a lever the log never
  moved that way. Read as one side, such a lever read 0.21 on a converged plan.
  The norm is divided by `max(1, |gradient on the free actions|)`, as `_kkt` divides its
  residual. The reading is zero exactly at a KKT point. It is positive where a multiplier has the
  wrong sign, which `_kkt`'s own residual, on the free actions alone, does not see. Away from the
  constraints, its square is about twice the share of `|J|` that one more diagonal Newton step
  would buy.
- **The violation is a distance.** It is the largest distance in `v` from the plan to a side of
  its box, or to a row's bound that the plan is past. A row's distance is its excess over its
  norm in `v`. The barrier's condition is not among them, because `safety` audits it.
- **Each reading has its own tolerance.**
  - *Stationarity:* `10 sqrt(2 max(1e-14, 4 eps))`, with `eps` the actions' precision. The
    descent stops where no step buys more than `1e-14` of `|J|`, or where rounding hides the fall,
    about `4 eps` of it. A residual `r` predicts a fall of about `r^2 / 2`. Ten times the root
    allows for a curvature along the residual that is not one. The tolerance is 1.4e-6 in float64
    and 9.8e-3 in float32. On `test_plan_units`' problems the converged plans read 4.3e-8 to
    1.5e-7 in float64 and 7.4e-5 to 8.8e-4 in float32.
  - *Violation:* `1024 eps (1 + max|v|)`. The projection holds each side and row to
    `256 eps (1 + max|y|)` in its own `v`. The factor of four covers the plan's own rounding and a
    descent that started from another guess, whose `sigma` differs.
- **A plan held under a barrier has no stationarity reading (`None`, `not_evaluated`).** The
  barrier's condition is a constraint, and its multipliers would enter the residual.
  `shadow_prices` refuses that plan for the same reason. Its violation is read.
- **`prescribe` carries the plan's four numbers on `DecisionCertificate`, in its report and in
  `to_json`.** Each number has its status beside it. A number that is not finite is `null`, with
  the status `not_finite` (ADR 0055). With no plan, both numbers and both statuses are `null`.
- **The descent stops on this reading (amendment).** The stopping rule tested a fall: no step buys
  more than `1e-14` of `|J|`. The reading is a gradient, and its tolerance turns one into the other
  at a curvature of one along the residual, with a factor of ten, so a curvature of up to 100.
  Where the objective's curvature along the residual is `c` times the Gauss–Newton diagonal at
  the guess, a step buys at most `r^2 / (2c)` of `|J|`, and the fall alone stopped at a residual
  of up to `sqrt(2c 1e-14)`. On the four converged plans that read `beyond`, `c` along the
  gradient was 1.0e4 (a support penalty, which the scale does not read), 396 (two supports on the
  oscillator) and 663 (the harvest plan, whose curvature is not Gauss–Newton: its target is far
  out of reach); the best fall over the halvings was 8.6e-12 against 3.2e-11, 2.1e-12 against
  2.7e-12 and 5.0e-11 against 5.0e-11. Now, where the search finds no such step, the descent
  reads the residual in the variables at the guess, as this record reads it
  (`chc.control._first_order_residual`, one unit step of the projected or proximal map: for a
  short step it is the least norm of the gradient plus the binding constraints' pushes, since the
  projection onto a polyhedron then moves in the tangent cone, the polar of the normal cone). If
  it is beyond the tolerance, the search runs again with the fall read off the gradients at both
  ends of each trial, `(g + g') . (v - v') / 2`, Hager and Zhang's approximate Armijo condition
  (SIAM J. Optim. 16, 2005), exact on a quadratic. A trial counts where that fall is positive, at
  least a third of the predicted one, and the value does not rise. The descent stops where the
  residual is within the tolerance, or where no trial falls by its gradients either. Values alone
  could not finish: on the support-penalised plan, at a residual of 3.6e-6 the fall left was
  2.4e-12, below the values' rounding `4 eps |J|` of 2.7e-12; past `c = 1126` the last fall to
  the tolerance is always below it. A descent started at the guess reuses its own variables; one
  started elsewhere reads them at the guess once more. The descent in the caller's units, as the
  barrier rounds run it, has no check.
- **`schema_version` stays 2.** ADR 0055 bumps it when a field changes meaning, not when a field
  is added. These fields are new, and no existing field changed.

## Consequences

- A reader can tell a solve that converged short of the optimum, or stopped near it, from the
  status alone. `stationarity_status == "beyond"` says that the plan, its prices and its regret
  bound are those of a stopped solve.
- Every `causal_plan` pays for two more gradients (at the plan and at the guess), one
  Gauss–Newton pass at the guess, and a BVLS over the binding rows and sides. A receding-horizon
  loop pays it at every step. No timing was taken for this record.
- On `test_plan_units`' problems, in lever units from 1e-6 to 1e6, cost scales from 1e-6 to 1e6,
  and the two levers in opposite units, the readings agree to 1e-9 relative. On the cross-module
  panels they agree to 1e-6 under a budget that binds, with the levers and the state in other
  units.
- As first recorded, a descent that reported `converged` could stop short of the resolution. On
  `test_held_barrier`'s pessimistic oscillator (a support penalty and a second support model as
  the uncertainty penalty), the penalised descent reported `converged` after 37 steps and read
  2.5e-6, `beyond` the tolerance of 1.4e-6. Over 982 plans that 25 test modules solve, three more
  reported `converged` and read `beyond`: a support-penalised plan (8.1e-6 after 16 steps), a
  harvest plan on a singular arc (3.8e-6 after 16,281 steps), and the concave-envelope plan that
  `causal_plan` solves first for a budget split on a Kumaraswamy curve (1.42e-6 after 9 steps).
  That last plan is logged beside the final one, which reads 4.9e-8 after 10 steps in all; the
  two were first read as one plan inside the suite and alone.
- With the amendment, those plans read 1.27e-6 after 18 steps, 1.05e-6 after 42, 1.412e-6 after
  20,369 and 2.2e-7 after 10, each `within`. Over 998 plans that 81 test modules solve, all 801
  that report `converged` read `within`, the highest at 0.999 of the tolerance; the plans that
  read `beyond` were stopped by a step budget or took no step. The check stops a descent as soon
  as the residual is within, so a plan it carries past the fall's stop can read just under the
  tolerance, not at the 1e-7 of a descent the fall stopped. A plan whose descent the fall stopped
  within the tolerance takes the same steps to the same actions: the quickstart, the notebooks
  and the scripts the docs quote print the same numbers. The extra evaluations fall on the plans
  that were beyond: objective values and gradients went from 225 and 56 to 308 and 61 on the
  support-penalised plan, from 166 and 40 to 375 and 54 on the oscillator, and from 166,327 and
  16,282 to 329,137 and 51,435 on the harvest plan, whose variables read 1/663 of its curvature
  along the residual. Each failed search costs up to 40 values, and each trial of the gradients'
  search a gradient. A warm-started descent pays one more Gauss–Newton pass, at the guess. The
  task costs move in the eighth digit: 3.9481322188 to 3.9481322819, 5.3836706262 to
  5.3836709556, and 4745.43490270 to 4745.43490253; the first two minimise their penalties as
  well. `test_support`'s Python recursion, which the compiled penalised descent must match to 500
  units in the last place, takes the same rule.
- *Left:* a barrier-held plan's stationarity. It would need the rounds' multipliers and the
  Jacobian of the barrier's condition in the actions.

## Alternatives considered

- **Loosen the tolerance's factor of ten** to cover `c` up to 1e4. Rejected: it reads every plan
  more loosely to excuse four, and `c` has no bound.
- **Restart the descent from a converged plan that reads `beyond`, in `causal_plan`.** Rejected: a
  restart reads its variables and its stopping level at the plan, not at the guess, so it tests
  another rule (the oscillator's took one step to 9.8e-7), and every caller of the descent but
  `causal_plan` would keep the defect.
- **Put the penalty's curvature in the descent's variables.** A penalty's curvature at the guess
  is not one the guess can be relied on to show (ADR 0073), every penalised plan would take other
  steps, and the harvest plan's curvature is the cost's own.
- **Retry with values, a fall above `4 eps |J|`.** The support-penalised plan then stops at
  3.6e-6, where rounding hides the fall that is left.
- **Read the reading itself in the loop.** Its least squares with signs runs on the host. The
  unit step of the projected map is the same number for a short step and runs inside the
  compiled descent; on the four plans the two agree to every printed digit.

- **Copy the solver's own two numbers.** Rejected: they are in the caller's units (Context).
- **`_kkt`'s residual as it is, on the free actions.** Rejected: it reads zero at an action held
  on a side that its gradient pulls it off. It also takes multipliers by least squares, so
  dependent binding rows can show a multiplier of the wrong sign where a set of the right sign
  exists. BVLS takes the least residual over the multipliers with the right signs.
- **The natural residual `||v - P(v - g_v)||` in `v`.** It needs no binding rule. But it costs a
  Dykstra projection per plan, and a binding row with a large multiplier would read the
  projection's tolerance in place of rounding.
- **The box's width as each lever's scale, in place of `sigma`.** It leaves the stationarity in
  the cost's units, and it has no scale for an infinite box. `sigma` is the scale that the descent
  already uses (ADR 0059).
- **One fixed tolerance, such as `shadow_prices`' `max(1e-6, 1e3 eps)`.** Rejected: a float32
  plan converges to 1e-4 to 1e-3 and would read `beyond`.
