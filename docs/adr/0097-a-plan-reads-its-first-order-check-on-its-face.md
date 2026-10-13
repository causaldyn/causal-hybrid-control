# ADR 0097 — A plan reads its first-order check on its face

**Status:** accepted, 2026-10-13.

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
- A descent that reports `converged` can stop short of the resolution. On `test_held_barrier`'s
  pessimistic oscillator (a support penalty and a second support model as the uncertainty
  penalty), the penalised descent reports `converged` after 37 steps and reads 2.5e-6, `beyond`
  the tolerance of 1.4e-6. Restarted from that plan, it takes one more step and reads 9.8e-7.
  Over 982 plans that 25 test modules solve, three more report `converged` and read `beyond`,
  each within six times the tolerance: a support-penalised plan (8.1e-6), a harvest plan on a
  singular arc after 16,281 steps (3.8e-6), and a budget split on a Kumaraswamy curve (1.42e-6).
  The last one stops after 9 steps inside the suite and after 10 steps alone, where it reads
  4.9e-8. Every other plan that reads `beyond` was stopped by a step budget. The reading reports
  these; the descent is not changed here.
- *Left:* a barrier-held plan's stationarity. It would need the rounds' multipliers and the
  Jacobian of the barrier's condition in the actions.

## Alternatives considered

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
