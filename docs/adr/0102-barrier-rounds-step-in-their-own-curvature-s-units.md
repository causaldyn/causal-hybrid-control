# ADR 0102 — Barrier rounds step in their own curvature's units

**Status:** accepted, 2026-10-19. Amends the barrier rounds of ADR 0059 and ADR 0073, and the
stationarity of a held plan (ADR 0097).

## Context

The augmented-Lagrangian rounds that hold a barrier in the solve read the units of the problem in
four places:

- they stepped in the caller's units, with a first step of 0.2;
- they stopped on an absolute fall of 1e-9;
- they started from a descent in the caller's units;
- their initial penalty was floored at 1.

On an oscillator under a velocity floor, a force in units 1e3 its own ran out of 310 000 steps,
14 % above the optimum. A cost in units 1e-3 its own stopped as `converged` 2.3 % above it. A held
plan's stationarity read `not_evaluated`, so neither failure showed. A penalty that reached its
ceiling with a step still short of the condition reported `max_iterations`.

## Decision

- Each round steps in `v = m·u`, with `m² = c_GN + ρ Σ_on (∂c_i/∂u)²`: the cost's Gauss–Newton
  curvature plus the penalty's on the steps whose term is on.
- Each round stops as the planner does: first step 1, a relative fall, and the first-order check
  in the task cost's units.
- The rounds start from the planner's scaled plan. The initial penalty is Birgin–Martínez's,
  without floors.
- A held plan's `stationarity` is the augmented Lagrangian's residual. The binding steps'
  shortfall gradients join the KKT pushes.
- `SolverStatus` gains `infeasible` for a penalty at its ceiling with the condition failing.
- Constraint rows are coloured for disjoint actions, which holds in any diagonal metric.

## Consequences

- In every unit choice of force and cost, the held plan is the same to 1.4e-5 of the box. It is
  within 4e-6 of the SLSQP optimum and reads `within`.
- Step counts change. On the receding-horizon oscillator the rounds are about 11 times cheaper
  from cold, so warm multipliers save about 3 %, where they saved 20 %. On the pendulum demo they
  take 2.5 times more steps, for the same plan.
- A diagonal metric does not reach the optimum on ill-conditioned problems. On an unstable plant
  at `dt = 1`, the rounds stop at `max_iterations` with stationarity `beyond`, where they used to
  report a false `converged` 54 % above the optimum. A block or Newton subproblem is the open fix.

## Alternatives

- **Per-lever reach as the metric.** This was the previous partial fix. Rejected: it reads the
  levers' units but not the cost's or the penalty's.
- **Grow the penalty only after rounds that finish within their budget.** Tried. It was worse on
  the ill-conditioned case.
