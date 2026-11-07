# ADR 0144 — A held barrier's rounds may step by projected Newton

**Status:** accepted, 2026-10-27. Adds an opt-in solve to the barrier rounds of ADR 0102.
Amended 2026-11-06: an opt-in rule for how the same rounds share their budget and grow their
penalty, `outer="safeguarded"`, measured on the same corpus (last section); the default stays.
Amended 2026-11-07: an opt-in solve that takes the radius norm's kink at the barrier's centre by its
sides, `inner="orthant"`, measured on the same corpus (last section); the default stays.

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

## Amended 2026-11-06: the rounds' outer rule

The rounds share one budget of `steps` by halves: each round is given half of the steps left, and
a round whose share runs out grows no penalty (ADR 0102: grown on such rounds, the penalty reached
3e6 on a feasible oscillator and a round stepped to a plan 134 % above the optimum). Where every
round runs out of its share, that rule never grows the penalty at all. On an unstable drift,
`xdot = 0.5 x + u - 3 Z` with a hidden confounder at `Gamma = 1.5`, held under the fit's own
sensitivity model, all 15 rounds of the default 10 000 steps ran out with the penalty at its
start; at 100 000 steps one of those plans was certified over 27 leading steps of 30, where
10 000 steps certified all 30. Halved shares reach one step after about `log2(steps)` rounds, so
the limit of 30 rounds binds only on rounds that stop on their own rule.

`causal_plan(outer="safeguarded")`, and `RecedingHorizon.outer`, give each round a fixed share of
333 steps, the default budget over 30 rounds, run rounds until the budget is spent (at most 30
more than its full shares, for rounds that stop on their own rule), and grow the penalty tenfold
after every round whose measure fails to halve, its share run out or not: Birgin & Martinez's
rule as they state it. A round at the penalty's ceiling whose share ran out stops the rounds
`max_iterations`, not `infeasible`. Two shapes were tried first and rejected. An even split of the
budget over 30 rounds: at 30 steps it ran 30 rounds of one step, the penalty grew after eight of
them, and the oscillator's plan cost 3.5 times the optimum; with the fixed share it ends 2.2 %,
3.6 % and 2.1 % above it at 3, 10 and 30 steps. And a count of rounds equal to the full shares
alone: rounds that stopped on their own rule spent fewer steps, the count ended the rounds 27
steps short of a 100 000-step budget, those steps went to the next states' rounds, and the plan
cost 1.9 % more than at 10 000 steps.

Every held plan of the corpus above was solved both ways, 218 plans (209 by the diagonal rounds,
9 by Newton's; 205 that Newton applies to), the misspecification tests now holding 20 logs of the
unstable drift below:

| | plans | what moved |
|---|---|---|
| identical | 120 | |
| within 1e-9 relative | 5 | |
| cheaper, the same certified steps | 27 | by up to 27.9 % |
| costlier, the same certified steps | 62 | by up to 42.9 %, a coarse-step plan; plans converged both ways by at most 1.1e-7 |
| fewer certified leading steps | 4 | misspecification plans under a sensitivity model: 12 → 7 on the integrator under either radius, both `infeasible` where the halved rounds certified all 12; 30 → 28 and 30 → 25 on the unstable drift |
| more certified leading steps | 0 | |

Statuses: 131 `converged` both ways, 18 `converged` → `max_iterations`, 57 `max_iterations` both
ways (18 cheaper, 34 costlier), 2 `max_iterations` → `infeasible`, 10 `infeasible` both ways.
Descent steps: 1 326 584 → 1 106 353.

On three logs of the unstable drift above, against SLSQP on the same held problem at 10 000
steps: the halved rounds end 0.17 %, 1.54 % and 0.32 % above its cost, each certified on 30 of
30 steps; the safeguarded rounds 0.17 % and 1.15 % above it, and 0.95 % below it on the third,
a plan certified on 28 of 30 and so held to fewer steps. Neither is within 0.2 % on all three. SLSQP's plan rests 27 to 29 of its 30
actions at zero, the kink of the tube's `||D^+ u||` (ADR 0119), where neither the descent nor
Newton's step converges, and SLSQP itself stopped on its iteration limit or a failed line search,
so its cost is a reference, not a certified optimum. At 100 000 steps the safeguarded rounds kept
the second log's plan at 30 of 30 and lowered its cost 65.95203 → 65.95086, and left the third's
unchanged, their rounds having stopped at the penalty's ceiling after 6 448 steps; the penalty
grew 7 times there, where under the halved shares it never grew. On that log the rounds held the
conditions to a shortfall of 1e-4 with the penalty a tenth of its ceiling; the next states then
joined at that penalty, and the first round that failed to halve their shortfall reached the
ceiling, the plan certified on 28 steps.

So the rule moves results users rely on beyond rounding, in both directions, and on four plans
certifies fewer leading steps: by the contract rule above it ships opt-in, and the default stays
`"halved"`. It is not a candidate for the default: what holds those plans short of the optimum is
the kink, not the outer rule.

## Amended 2026-11-07: the kink at the barrier's centre

**Where the kink enters.** Under a radius, a held step reads its action through
`N_k = ||D^+ (u_k - c)||` twice: in its own condition's spread, `a + <w, u_k> - d N_k`, and in the
deviation tube (ADR 0119), `e_{k+1} = ||dF/dx|| e_k + ||dF/dc|| Delta_k N_k`, which every later next
state's room reads. Both rise with `N_k`, so an action off `c` pays in its own step and in every
later one, and the optimum can rest an action exactly on `c`. On the three logs of the unstable
drift above, SLSQP's plan rests 27, 27 and 29 of its 30 actions there.

**Why neither solve converges there.** The rollout's derivative masks the norm at zero, so it
reads no slope on `c` and a slope of `w |D^+|` of either sign beside it. With one lever the norm is
linear on each side, so its second derivative is zero off `c`: Newton's quadratic model has no
term that holds an action on `c`, its step crosses it, and the line search halves it. The diagonal
descent's iterates step across `c` the same way; none of the three default plans had an action
within 1e-9 of it. And the first-order check reads the same masked derivative: a plan that rests
an action on `c` misses by what the rest of the gradient pushes there. A prototype that took
Newton's step on the cut box below but kept that check landed within 1e-7 of SLSQP's cost on all
three logs and still read stationarity 0.65 to 0.78 against a tolerance of 1.4e-6, so its rounds
never stopped on their own rule and the status read `max_iterations`.

**Decision: `causal_plan(inner="orthant")`**, and `RecedingHorizon.inner`, `InnerSolve` gaining
the value: Newton's rounds with the kink taken by its sides, the orthant-wise rule (Andrew & Gao
2007, ICML) on Bertsekas' projected Newton step.

- **Where it applies.** A step with one lever, under a radius or a sensitivity model, and a model
  that takes the action it is given. With several levers the norm of a vector is not linear on any
  orthant, and a model that sets part of its action from the state is not covered: both, and a
  barrier with no radius, step as `"newton"` does, bit for bit.
- **The cut box.** At each Newton step an action off `c` keeps to its side: its box is cut at `c`.
  An action on `c` leaves it only toward a side where the round falls: right where
  `dF/du + w |D^+| < 0`, left where `dF/du - w |D^+| > 0`, with `dF/du` the round's derivative at
  the norm held fixed and `w >= 0` its derivative in the norm. Otherwise it is held on `c` as a
  bound, where zero is in the round's subdifferential along it. Within one cut box the round is
  smooth, and Newton's step converges there.
- **Reading `dF/du` and `w` with no change to the shared terms.** The centre becomes a variable per
  step, `C`. The norm reads `u - C` alone, so the derivatives in `u` and in `C` sum to `dF/du` at the
  norm held fixed, and the derivative in `C` times `u - C` is `-w N`. On `c` the norm has no
  direction, so both are read at a centre moved by 4 eps times the largest of `|c|`, the box's
  finite sides and the actions: the one-sided slopes, within that share of the slopes at the kink.
  The condition, the tube, the audit and the penalty are the library's own, unchanged.
- **The first-order check reads the kink by its subdifferential.** With the rounds' multipliers,
  an action on `c` misses by what is left of its push once `w |D^+|` is taken off, and a side of
  the box takes off a push outward as before. The plan's stationarity is the smaller of this miss
  and the masked one: each is an upper bound on the least residual. The rounds' stopping rule reads
  the same check.

On the three logs, 10 000 steps under the halved rule:

| log | SLSQP | default rounds | `inner="newton"` | `inner="orthant"` |
|---|---:|---:|---:|---:|
| 0 | 76.012813 | +0.17 %, `max_iterations` | +0.69 %, `max_iterations` | +6.3e-8, `converged` in 91 steps |
| 1 | 65.204394 | +1.54 %, `max_iterations` | +0.21 %, `max_iterations` | +8.0e-8, `converged` in 2 814 steps |
| 2 | 50.995133 | +0.32 %, `max_iterations` | +0.52 %, `max_iterations` | +4.2e-8, `converged` in 1 880 steps |

Each plan is certified on all 30 steps. The orthant plans rest 27, 27 and 29 actions exactly on
zero, as SLSQP's do, at stationarity 1.4e-11, 8.8e-15 and 1.9e-13. SLSQP holds the columns with no
back-off, the rounds 1e-6 inside them. The safeguarded rule gives the same three plans, and at
100 000 steps both rules give the same plans on logs 1 and 2, in the same steps.

On the oscillator under a velocity floor with one lever at `gamma = 2`, where no optimal action
rests on the kink, the orthant rounds land on Newton's plan, 4.2278682, in 1 025 steps against
1 089. At `gamma = 1`, with two levers, and on the SIR plan they are Newton's rounds, bit for bit.

## Why opt-in, again

Every held plan of the corpus above, now 240 (the misspecification tests hold more logs), was
solved as its test asked and with `inner="orthant"`; 227 are in its scope (13 hold a support model
or linear rows):

| | plans | what moved |
|---|---|---|
| identical | 31 | |
| within 1e-9 relative | 5 | |
| cheaper, the same certified steps | 90 | 28 `converged` both ways by at most 5.8e-7; 57 `max_iterations` → `converged`, by up to 81 % (a coarse-step plan) and 3.7 % on the misspecification plans; 3 `max_iterations` both ways; 2 `infeasible` both ways |
| more certified leading steps | 6 | all `max_iterations` → `converged`: 11 → 12 and 10 → 12 on the integrator, 29 → 30 on a log of the unstable drift, 28 → 30 on log 2 above (three tests) |
| costlier, the same certified steps | 95 | 89 `converged` both ways by at most 7.5e-7; 4 `max_iterations` → `converged` by at most 5.8e-7; 2 `max_iterations` both ways by 5.4 %, at a budget of 10 steps |
| fewer certified leading steps | 0 | |

Statuses: 147 `converged` both ways, 67 `max_iterations` → `converged`, 7 `max_iterations` both
ways, 6 `infeasible` both ways, none worse. Descent steps: 1 549 757 → 188 638. Of the 98 plans
with one lever under a radius, 52 rest an action on the centre.

The counterexamples to "every plan moves toward the optimum":

- **The 89 converged plans that cost more**, by 1e-9 to 7.5e-7 relative. Both ends meet the
  tolerance; the orthant plans' stationarity is at most 3.4e-10 where the default's reached 1.4e-6.
  They move `gamma*` with them on 59 of the 89. This is the move Newton made in the first table.
- **Two oscillator plans at a budget of 10 steps** (the rounds' shared budget and the projection
  tests) end `max_iterations` either way, 4.21603 → 4.44480, 5.4 % costlier. They hold no radius,
  so there is no kink: they are Newton's steps on a budget too small for its rounds.
- **The longer prefixes** are all moves toward the plan the rounds hold, but five of the six cost
  more (by 0.27 % to 0.96 %), being held on more steps: log 2 above certified 28 of 30 at 50.50846
  under the safeguarded rule and 30 of 30 at 50.99514.

So by the contract rule the orthant rounds ship opt-in, as Newton's did. They are the candidate
for the default on plans with one lever under a radius: of those 98, all end `converged`, 61 of
them where the default rounds ended `max_iterations`, none is certified on fewer steps, and none at
the same steps costs more than 5.5e-7 more.

**Alternatives.**

- **A proximal step on the norm in the diagonal descent**, as ADR 0078 takes the radius' norm, with
  the norm's weight read by the same split. Exact zeros appeared, but the rounds' descents still
  ran out of their shares on the round's conditioning: +0.10 %, +0.58 % and +0.73 % above SLSQP's
  cost on the three logs, the last worse than the default's +0.32 %.
- **A smoothed norm, `sqrt(N^2 + eps^2)`, with `eps -> 0`.** Not exact at any finite `eps`, and
  the smoothing would have to enter the condition and the tube the audit shares. ADR 0119 tried
  it at `tau = 1e-3` under the diagonal descent: the rounds still ended on their budget, which it
  read as the kink not being what stops them. Under the diagonal descent both stop it, as the
  proximal prototype above shows; under Newton's step the kink alone does: the same rounds end
  `max_iterations` without the cut box and `converged` with it.
- **Several levers a step.** A group norm's kink is a point in `m` dimensions, not a sign: a block
  proximal step or a second-order cone step would be needed. Not built; no test plan reaches it.

**Limits.** `CausalPlan.shadow_prices` and the decision weight still read the masked derivative:
an action on the kink is a free action there, and what its push leaves shows in their residual.
`prescribe` passes no `inner`: its held plans keep the default rounds.
