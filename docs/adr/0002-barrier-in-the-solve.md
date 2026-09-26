# ADR 0002 — The barrier condition held inside the solve

**Status:** accepted, 2026-09-26.

## Context

`causal_plan` held a box and linear rows in the solve and nothing about the state. The condition
`certify_safety` audits — at every planned step,

```
grad h(x_k) . f(x_k, 0) + <w_k, u_k> - delta ||grad h(x_k)|| ||u_k||  >=  -alpha h(x_k),
w_k = B_k^T grad h(x_k),   delta = (gamma - 1)/(gamma + 1) * cvar_gap,
```

the control-barrier condition of Ames et al. (2017) in its worst case over the identified set —
was only priced after the solve. A plan could come back and fail its own audit, and the only
enforcement was truncation (`certified_actions`) or the one-step filter. A customer's constraint
is a constraint, not a report, so the façade needs the condition in the solve.

What the solution had to keep, in order:

1. **The audit stays the source of truth.** Whatever the solver believes, the plan's verdict is
   `certify_safety` run on the finished plan.
2. **The same condition, not an approximation of it.** The solve and the audit form it with one
   helper, so the two cannot drift apart.
3. **The optimum, not a neighbourhood of it**, held to an exact QP oracle wherever one exists.
4. **A slack barrier costs nothing**: bit-identical actions, status and iteration count.
5. **The plan's own objective.** The support and uncertainty penalties the plan minimised are
   minimised under the barrier too.
6. **No new dependency**; traceable in JAX, in `float32` and `float64`; the box and the rows as
   ADR 0001 left them.

ADR 0001's first requirement — every iterate feasible — is deliberately **not** on the list. The
condition is nonlinear in the actions through the state, so projecting onto it is itself a
nonlinear program. Only the answer is held, and requirement 1 is what makes that acceptable.

## Decision

- `BarrierConstraint(barrier, alpha, gamma, cvar_gap)` carries the audit's arguments, validated at
  construction; `causal_plan(barrier=...)` holds it.
- The condition enters as a Powell–Hestenes–Rockafellar term, one multiplier per step:
  `sum_k (max(0, lam_k + rho c_k)^2 - lam_k^2) / (2 rho)`, with `c_k` the step's shortfall. The
  rounds run the existing penalised descent (`chc.support._pessimistic_loop`), so the box and the
  rows stay in its projection — the lower-level constraints of Andreani et al. (2008) — and the
  plan's penalties stay in its objective.
- The safeguards are Birgin & Martínez's (2014): `lam <- max(0, lam + rho c)`; `rho` grows tenfold
  whenever the measure `max_k |max(c_k, -lam_k/rho)|` fails to halve; the starting penalty is
  `clip(10 max(1, |J|) / max(1, sum max(c, 0)^2 / 2), 1e-8, 1e8)`. The rounds stop when the measure
  is within half the back-off and the round's descent stopped on its own rule; they give up when
  `rho` would pass `1e8` times its start, or after 30 rounds.
- **A back-off.** `c` is shifted by `1e-6` times the condition's largest term at the starting plan,
  so the rounds settle inside the condition. An active step solved exactly sits on the boundary,
  and rounding alone would then fail the audit about half the time.
- The barrier-free solve runs first. A plan that already clears the shifted condition is returned
  untouched.
- `CausalPlan.safety` is `certify_safety` on the finished plan, with `u_max` the box's largest
  magnitude — the budget `prescribe` prices `Gamma*` at.
- `prescribe(hold_constraints=True)` hands its constraints' barrier to the solve, at its `gamma`.
  Off by default, so no existing schedule moves.
- The solve and `certify_safety` share `_barrier_terms` and `_guaranteed`. The refactored audit is
  bit-identical to the old one on 51 audits in both precisions; its norm masks zero rows so that
  the gradient at a zero action is finite, while the forward value stays `jnp.linalg.norm`'s to the
  bit (a hand-written `sqrt(sum(x*x))` differs by one rounding).

## Evidence

Counts only: no wall time was taken. The reference is the exact QP (a Cholesky change of variables
and one least-distance NNLS, Lawson & Hanson 1974) where the condition is affine in the actions,
and the better of SLSQP and trust-constr otherwise. "Gap" is the plan's task cost above the
reference, relative.

The back-off was chosen by measurement, among three:

| instance | H | steps the free plan meets it at | `1e-6` (shipped): rounds · descent steps (free solve) · certified · gap | `1e-7` | `sqrt(eps) ≈ 1.5e-8` |
|---|---|---|---|---|---|
| two-state plant, mixed `h`, `gamma = 1` (exact) | 12 | 7 | 3 · 2 358 (615) · 12 · `2.5e-7` | 7 rounds, `1.9e-7` | 12 rounds, `1.9e-7` |
| oscillator velocity floor, `gamma = 1` (exact) | 40 | 34 | 11 · 16 965 (2 754) · 40 · `7.0e-6` | 16 rounds, `8.4e-7` | 20 rounds, `2.1e-7` |
| oscillator velocity floor, `gamma = 2` (best of two) | 40 | 32 | 10 · 29 740 (2 754) · 40 · `2.2e-5` | 14 rounds, `2.3e-6` | 15 rounds, `5.4e-7` |
| SIR, capacity `I <= 0.1` (best of two) | 100 | 72 | 16 · 8 551 (0) · 100 · `6.1e-6` | 21 rounds, `7.5e-7`, `rho` at its ceiling | **fails**: 73 of 100 certified, `rho` at its ceiling |

The back-off is most of the price — a smaller one buys a smaller gap on every instance — but it is
also the target the measure has to reach, and a smaller target needs a larger `rho`. On SIR, `1e-6`
converges with `rho = 1e6`, three decades under the ceiling; `1e-7` converges only at the ceiling
itself, and `sqrt(eps)` does not converge.

In `float32` the velocity floor at `H = 20` converges at `gamma = 1` (2 839 descent steps, every
step certified, worst margin `6.7e-6`) and at `gamma = 2` (1 614 steps, every step certified).

A slack barrier returns bit-identical actions, status and iteration count, including when the free
solve was stopped by its budget — the rounds must not resume a descent the caller capped.

The tests were checked by mutation: ten edits to the shipped code, each run against
`tests/test_held_barrier.py` — dropping the uncertainty penalty or the support model from the
rounds' objective, forming the solve's condition at `gamma = 1`, never or always running the rounds,
dropping the back-off, auditing at `gamma = 1` or with no `u_max`, never raising the penalty, and
dropping the multiplier update. Four survived along the way, and each exposed a weak test. Every
step of the pessimism test's instance was active, so the condition alone fixed the plan and no
penalty could move it — the headline test's instance had the same defect, so its oracle could not
see the objective; the slack-barrier test only started from a converged free solve, from which a
resumed descent takes no step; a penalty with no multipliers returns the same plans (below); and
once the headline floor moved to leave steps free, standing still cleared it everywhere, so no step
needed authority and the audit test could no longer see the budget. All ten now fail the tests —
the multipliers only through the log's `active_steps`, which a test holds to the exact optimum's
active set.

## Consequences

- Iterates are not feasible; only the answer is held, and `plan.safety` is the verdict. A solve
  stopped by its budget, or a condition no admissible action can meet — a relative-degree-2
  barrier at its first step, where no action moves the condition — comes back short, reported by
  the audit and not raised.
- Each round is a full descent of at most `steps`; there are at most `1 + 30` descents. On the
  instances above a held plan took 3.8–11× the free solve's descent steps, the free solve's own
  included.
- The back-off prices at `2.2e-5` relative at worst on the instances measured, and it is why the
  plan clears the audit rather than sitting on it.
- `certify_safety` raises where `grad h = 0` at a planned step — a constant barrier, or a ball
  barrier at its centre — because `barrier_gamma_star` cannot invert a radius there. That predates
  this change, and `causal_plan(barrier=...)` inherits it: it raises after the solve.
- **An invariant for later edits:** the solve and the audit must form the condition through the
  same helpers. `test_the_plan_carries_the_audit_itself_rather_than_the_solvers_report` compares
  the two field by field.

## Alternatives

- **Plan, then filter** — the enforcement this module offered before: `robust_safety_filter`
  applied step by step to the free plan, along the trajectory the filtered actions produce.
  Measured on the instances above except SIR, whose one-sided box the filter's symmetric interval
  does not model: the filtered plan costs `3.0e-4`, `1.84` and `1.64` above the reference, relative,
  where the held plan costs `2.5e-7`, `7.0e-6` and `2.2e-5` — the free plan's actions were chosen
  for a trajectory the filter no longer follows. It also fails its own audit on 2 and 16 steps of
  the oscillator's 40, by rounding (at most `1.7e-16`): it clips onto the end of the admissible
  interval, exactly where an unshifted solve lands, which is what the back-off is for.
- **A quadratic penalty with no multipliers** — measured, by running the shipped rounds without
  the multiplier update. The back-off lets it finish inside the condition at a finite penalty, and
  it returns the same plans to within the back-off's price (gaps `2.5e-7`, `2.1e-5`, `2.5e-5`,
  `7.5e-6`). It needs a penalty two to five decades larger (`2.9e3`, `4.8e5`, `3.2e6`, `1e9`
  against `29`, `48`, `32`, `1e6`) and 1.0–4.0× the descent steps, and on SIR it converges only at
  the penalty ceiling — the fragility that ruled out the `1e-7` back-off.
- **A log-barrier (interior point)** — needs a strictly feasible start, and the free plan is
  infeasible exactly when the barrier matters; finding a feasible start is itself the problem.
- **Linearise the condition and project with ADR 0001's Dykstra** — each linearisation is valid
  only near the plan it was taken at, so it needs a trust region and a merit function: an SQP
  method to maintain, for one constraint family. Not measured.
- **A spectral projected gradient inner solver** (Birgin, Martínez & Raydan 2000), as ALGENCAN
  uses — not measured. The existing descent was kept so that the rounds and the free solve are one
  code path; a growing `rho` makes the inner problem ill-conditioned, which a spectral step
  addresses, so it is the first thing to try if the rounds' cost matters.
- **Scaling the inner step by `rho_0 / rho`** — measured on the prototype: it cut cost evaluations
  but raised gradient evaluations two- to four-fold.

## References

- Ames, A. D., Xu, X., Grizzle, J. W. & Tabuada, P. (2017). Control barrier function based
  quadratic programs for safety critical systems. *IEEE Trans. Automat. Control* 62(8).
- Andreani, R., Birgin, E. G., Martínez, J. M. & Schuverdt, M. L. (2008). On augmented Lagrangian
  methods with general lower-level constraints. *SIAM J. Optim.* 18(4).
- Birgin, E. G. & Martínez, J. M. (2014). *Practical Augmented Lagrangian Methods for Constrained
  Optimization.* SIAM.
- Birgin, E. G., Martínez, J. M. & Raydan, M. (2000). Nonmonotone spectral projected gradient
  methods on convex sets. *SIAM J. Optim.* 10(4).
- Hestenes, M. R. (1969). Multiplier and gradient methods. *J. Optim. Theory Appl.* 4(5).
- Lawson, C. L. & Hanson, R. J. (1974). *Solving Least Squares Problems.* Prentice-Hall.
- Powell, M. J. D. (1969). A method for nonlinear constraints in minimization problems. In
  R. Fletcher (ed.), *Optimization*. Academic Press.
- Rockafellar, R. T. (1973). A dual approach to solving nonlinear programming problems by
  unconstrained optimization. *Math. Programming* 5.
