# ADR 0144 — A held barrier's rounds may step by projected Newton

**Status:** accepted, 2026-10-27. Adds an opt-in solve to the barrier rounds of ADR 0102.

## Context

`causal_plan(barrier=...)` holds the barrier condition by augmented-Lagrangian rounds. Each round
minimises the plan's cost plus the condition's penalty over the box, by a descent in a diagonal
metric: the root of the round's Gauss–Newton curvature along each action (ADR 0102). ADR 0102
named the limit: a diagonal metric does not solve an ill-conditioned round, and "a block or
Newton subproblem is the open fix".

The epidemic plant shows it. SIR with `beta = 0.6`, `gamma = 0.1`, 60 steps of 1, actions in
`[0, 0.9]`, at most a tenth infected, `alpha = 0.5`:

- The diagonal rounds spend all 10 000 steps over 30 rounds and end `max_iterations`, with
  stationarity 1.7e-5 against a tolerance of 1.4e-6.
- In the descent's own variables the round's Hessian has condition number 195, 1 715 and 17 102 at
  a hundredth, a tenth and all of the rounds' last penalty (`validation/projected_newton_round.mac`,
  part 5: Jacobi's eigenvalue method in 32 digits agrees with numpy to 2.6e-13). The number grows
  with the penalty, as it must: the penalty adds `rho g g'` along the binding steps' gradients,
  which couple every action before them.

## Decision

- **`causal_plan(inner="newton")`** solves each round by Bertsekas' projected Newton method on the
  box (Bertsekas 1982, SIAM J. Control Optim. 20(2):221–246). An action within the projected
  gradient step's length of a side, with its gradient pushing out, takes that step; the others
  take the Newton step on their block of the matrix; the step is halved along the projection arc
  to an Armijo fall. A round stops where the fall the step predicts is within the descent's
  tolerance of the round's objective.
- **The matrix.** The cost's Gauss–Newton matrix over all actions, `sum_k S_k' L_k S_k`, plus the
  penalty's Hessian on the steps whose term is on. Where that block is not positive definite on
  the free actions, the penalty's Gauss–Newton part `rho sum g g'` replaces its Hessian; the
  diagonal of that second matrix is the diagonal metric's square (`validation/projected_newton_round.mac`,
  parts 1–3). Where neither factors, or no halving buys the Armijo fall above rounding, the
  diagonal descent takes the round over from where Newton stopped, with the steps left.
- **Why the penalty's Hessian, not Gauss–Newton alone.** At `gamma > 1` the condition reads the
  radius' norm `d ||u||`, which is convex and whose curvature grows without bound near `u = 0`.
  Gauss–Newton drops it. On the two-force oscillator under a velocity floor at `gamma = 2` the
  line search then cut each step by thousands, the rounds spent 40 959 steps and ended
  `max_iterations` at stationarity 0.24, 1.3 % above the diagonal rounds' plan. With the
  penalty's Hessian the same plan ends `converged` in 2 175 steps at stationarity 1.7e-10.
- **Counting.** A Newton step counts as one descent step, against the same `steps` budget.
- **Scope.** The box alone: a support model's penalty or linear rows are refused, since projected
  Newton projects onto a box. The matrix is dense, `(H m)^2`, and more than 4 096 actions are
  refused. `RecedingHorizon` carries `inner` as it carries every argument of `causal_plan`. The
  O(H) Riccati form of the Newton step (Dunn & Bertsekas 1989, JOTA 63:23–38) is not built: no
  test of ours reaches the size limit.
- **Opt-in, experimental.** The default stays `"diagonal"`, bit for bit.

## Why opt-in: the default flip, measured

The rule for contracts: a change that moves a number only toward its stated guarantee is a fix;
one that moves results users rely on beyond rounding ships opt-in first. Every held plan in the
plan, barrier, shadow-price, decision-weight, receding-horizon, prescription, lever-selection,
cross-module and misspecification tests was solved both ways (167 plans Newton applies to):

| | plans | what moved |
|---|---|---|
| `converged` both ways | 124 | cost by -5.8e-7 to +1.4e-6 relative, actions by at most 7.2e-4; stationarity at most 1.4e-6 before, median 1.4e-15 after |
| `max_iterations` → `converged` | 15 | the SIR plan; a coarse-step plan 63 % and 81 % cheaper; a held disk 0.55 % cheaper; nine misspecification plans, up to 3.7 % cheaper |
| `max_iterations` both ways | 18 | |
| `infeasible` both ways | 6 | |
| `max_iterations` → `infeasible` | 4 | misspecification plans under a sensitivity model's ceiling: leading certified steps 11 → 4, 10 → 5, 30 → 29 |
| identical | 23 | rounds never entered, or Newton stopped at once |

Descent steps over the 167 plans: 1 016 224 → 335 471. `gamma*` moved with the plans, e.g.
39.74178 → 39.74192 on the oscillator floor.

So the flip moves every converged plan by more than rounding, changes `gamma*` and step counts
that recorded results carry, and on four plans whose rounds never converge it reaches the
penalty's ceiling sooner and keeps a plan certified over fewer leading steps. Three tests that pin
the diagonal descent's mechanics would fail: the rounds' shared budget, one compilation for both
tolerances, and a warm start taking no more steps than a cold one (2 941 against 2 915).

## Consequences

- On the SIR plan at `alpha = 0.5`, `inner="newton"` ends `converged` in 66 steps over 16 rounds,
  stationarity 1.5e-12. Its actions are within 1.7e-7 of SLSQP's on the problem the rounds hold
  (each held column shifted by the back-off), where the diagonal rounds' were 1.1e-4 off. Its cost
  is 3.0e-9 relative above that optimum. Both plans sit 3.2e-6 above SLSQP's optimum of the
  unshifted problem: that is the back-off's price, not the solve's.
- At `alpha = 0.1` both converge; Newton in 100 steps against 4 408, stationarity 2.3e-11
  against 3.5e-7.
- A Newton step costs a dense factorisation and a Hessian of the rollout; the step counts above
  are not wall times.

## Alternatives

- **Newton as the default now.** Rejected for 0.23.0 by the table above. Candidate for a later
  minor once the four non-converging misspecification plans are understood.
- **Gauss–Newton alone.** Rejected: it drops the radius' curvature, measured above.
- **A block-diagonal metric per step.** Rejected: the penalty couples each binding step with every
  action before it, so the ill-conditioning is not within a step's block.
- **Linear rows by a QP subproblem per Newton step.** Not built: prescriptions with rate limits
  keep the diagonal descent.
