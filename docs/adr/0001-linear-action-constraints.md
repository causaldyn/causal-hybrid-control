# ADR 0001 — Linear constraints on the action sequence

**Status:** accepted, 2026-09-22.

## Context

Until this change every solver in `chc.control` and `chc.support` accepted a box on each action and
nothing else. Plans that get executed carry two more kinds of limit: a **budget** (total spend over
the horizon at most `B`) and a **rate limit** (a lever moves at most `c` per step — a price, a
staffing level or a set-point cannot jump). Both are linear in the flattened action sequence; neither
is a box.

What the solution had to keep, in order:

1. **Every iterate feasible.** A solve stopped by its step budget returns `max_iterations` with a
   plan, and that plan must still be executable. This rules out penalties, and any method whose
   iterates are infeasible until convergence.
2. **The optimum, not a neighbourhood of it.** The tests hold the solver to an exact QP oracle.
3. **No new dependency**, traceable in JAX, in `float32` and `float64`.
4. **Box-only calls unchanged to the bit**, so no existing user's numbers move.

## Decision

- `LinearConstraint(matrix, lower, upper)` over the row-major flattened actions (step `k`, lever `j`
  is column `k·m + j`), two-sided, `±inf` for an absent side. `LinearConstraint.rate_limit(horizon,
  caps)` builds the bands. `constraints=` on `projected_gradient_solve`, `projected_gradient_control`,
  `pessimistic_solve`, `pessimistic_control` and `causal_plan`; `Lever.cap_per_step` in `prescribe`.
- The projected gradient keeps its shape; only the projection changes. The projection onto
  `box ∩ {lower ≤ A u ≤ upper}` is
  1. **Dykstra's algorithm** (Boyle & Dykstra 1986), with the rows coloured into classes of mutually
     orthogonal rows so that each class is projected at once, and the box last in every sweep so that
     it holds exactly. Dykstra is block-coordinate ascent on the dual (Gaffke & Mathar 1989;
     Tibshirani 2017), so the previous projection's increments are a valid warm start for the next;
  2. stopped when the **increments** stop moving, not when the iterate does (Birgin & Raydan 2005);
  3. finished by a **dual active-set polish** started from Dykstra's increments — before the first
     sweep, and every 32 sweeps after — whose steps stop at the first multiplier that would change
     sign (the ratio test of Goldfarb & Idnani 1983), with a ridge of `256 eps |row|²` on the solve.
     Its result is kept only if the primal–dual pair passes the projection's KKT conditions,
     complementary slackness included.
- Emptiness of the feasible set is settled once per solve by a linear program (`scipy.optimize.linprog`,
  HiGHS). Dykstra cannot detect it; it would run every projection to its cap.
- `SolverResult.constraint_violation` reports the returned plan's worst row excess, and
  `stationarity` is measured against the polytope projection rather than the box.

## Evidence

The comparison with the alternative that mattered was **registered before the first run**
(2026-09-22):

> *accuracy gate* — a method is admissible only if, on every instance, its returned plan violates no
> constraint by more than `1e-9` and its task cost is within `1e-6` (relative) of the reference
> optimum. *choice* — the augmented Lagrangian if Dykstra needs more than 3× its wall time (or,
> without wall time, more than 3× its gradient and cost evaluations, the projection priced at one
> evaluation per sweep) on at least 2 of the 3 instances; otherwise Dykstra.

No wall time was taken — the machine did not pass its readiness check — so the rule was applied to
work counts, which are deterministic. `uv run python scripts/bench_linear_constraints.py instances`
reproduces the table; each plan has a rate limit at `0.35×` the box-only optimum's largest step and a
budget at `0.6–0.7×` its total, so both bind.

| instance | rows (binding) | D gap | D violation | D gradients / evaluations / sweeps / solves | A gap | A gradients / evaluations | work D/A | solve price at which A breaks even |
|---|---|---|---|---|---|---|---|---|
| oscillator (H=40, m=1) | 40 (8) | `8.9e-8` | `4.4e-16` | 2 439 / 2 479 / 0 / 2 487 | **`4.1e-6`, fails the gate** | 40 301 / 245 873 | 0.026 | 113 |
| linear (H=30, m=2) | 59 (14) | `1.3e-8` | `4.4e-15` | 1 259 / 1 299 / 0 / 1 347 | `7.9e-7` | 41 704 / 286 090 | 0.012 | 241 |
| marketing mix (H=12, m=3) | 34 (23) | `4.7e-10` | `3.3e-16` | 208 / 248 / 64 / 488 | `6.2e-8` | 11 585 / 83 027 | 0.011 | 193 |

D is the shipped solver; A is a Powell–Hestenes–Rockafellar augmented Lagrangian with the box kept in
its inner projected gradient, returned plans made exactly feasible before scoring. "Work D/A" prices
a sweep and an active-set solve at one evaluation each; the last column is the price of a solve, in
evaluations, at which the two would cost the same. By flop count a dense solve on these `p ≤ 59`
rows costs tens of evaluations, not hundreds — an estimate, not a measurement.

The registration predates the polish. Measured by the same protocol before it existed, Dykstra alone
needed 66, 96 and 163 sweeps per projection, for a work ratio of 0.59, 0.39 and 0.44 — Dykstra under
the rule then too, but only while a sweep cost less than 1.7–2.6 evaluations. The polish was not
added to win the comparison; it was added because the projection was wrong.

**The property test found two defects in the projection**, both against the exact least-distance
projection (one NNLS, Lawson & Hanson 1974):

1. *Stopping on the iterate.* A random three-row instance stopped after 31 sweeps at a point that
   broke a row by `7.8e-2`: a sweep can hand the iterate back exactly where it started while the
   increments are still travelling. Stopping on the increments is what certifies optimality.
2. *Vertices Dykstra takes thousands of sweeps to leave.* Near a vertex where one more constraint is
   almost active, Dykstra holds the iterate still while an increment drains at a constant rate; 17 of
   3000 random polytopes ran out the 4 096-sweep cap, up to `2.7e-3` from the projection. Solving on
   the active set the increments' signs name brought that to 10, and a primal–dual active-set
   correction (Hintermüller, Ito & Kunisch 2002) to 9. All 9 were vertices at which the increments
   named exactly one constraint too many: the set was over-determined, the solve singular, and with
   no solution there are no signs to correct. The ratio test brings it to **0**: over all 3000 the
   worst distance to the exact projection is `4.3e-12`, the median projection ends without a
   sweep, and none needs more than 32 (`scripts/bench_linear_constraints.py polytopes`).

## Consequences

- Every iterate is feasible to the projection's tolerance, `256 eps (1 + max|y|)` along each row's
  normal, whenever the projection settles — which it did on everything measured — so a
  budget-stopped solve returns an executable plan. `constraint_violation` reports the plan's worst
  excess either way.
- A projection costs one to two active-set solves on the instances above and almost no sweeps. Each
  solve is dense and `O(p³)` in the constraint rows, measured only up to `p = 59`; at hundreds of rows
  it will dominate, and a banded factorisation for rate-limit chains is the step to take if that
  regime matters.
- Box-only calls take the old code path and return bit-identical results.
- `plan_regret_bound` in `prescribe` is still priced against the box alone, so under a rate limit it
  is conservative rather than tight: a smaller feasible set only raises the constrained optimum.
- The emptiness check runs on the host, once per call, outside `jit`.
- **An invariant for later edits:** the polish is trusted only through its KKT check, and the check
  must read the primal–dual pair `(x, ν, μ)` itself. A check on feasibility and on the signs the
  method believes it holds accepts the points its ratio test leaves half-way — up to `0.91` from the
  projection, measured with a starved step budget — and
  `test_the_polish_vouches_only_for_the_projection` pins that.

## Alternatives

- **Augmented Lagrangian with the box in the inner solve** — rejected by the registered rule: it
  fails the accuracy gate on the oscillator, and needs 38–94× the work on all three instances. Its
  iterates are also infeasible until the multipliers converge, which requirement 1 excludes anyway.
- **A penalty in the objective** — iterates infeasible, and a budget-stopped plan can break its
  budget.
- **An external QP solver for the projection** (OSQP, a JAX QP) — not measured. A dependency for one
  subproblem, against requirement 3; and OSQP's own solution polishing guesses the active set from
  its duals the same way and falls back to its iterate when the guess does not check (Stellato et al.
  2020), which is the case the ratio test exists for.
- **The active-set method alone, without Dykstra** — it ends nearly every projection above from its
  warm start, but its step budget can run out: two attempts on the marketing-mix instance did not
  settle within 16 steps, and 64 Dykstra sweeps carried those projections to the next attempt.
  Dykstra accepts any duals as a start, so it is the fallback that makes the budget safe to have.

## References

- Birgin, E. G. & Raydan, M. (2005). Robust stopping criteria for Dykstra's algorithm. *SIAM J. Sci.
  Comput.* 26(4).
- Boyle, J. P. & Dykstra, R. L. (1986). A method for finding projections onto the intersection of
  convex sets in Hilbert spaces. *Lecture Notes in Statistics* 37.
- Gaffke, N. & Mathar, R. (1989). A cyclic projection algorithm via duality. *Metrika* 36.
- Goldfarb, D. & Idnani, A. (1983). A numerically stable dual method for solving strictly convex
  quadratic programs. *Math. Programming* 27.
- Hintermüller, M., Ito, K. & Kunisch, K. (2002). The primal-dual active set strategy as a semismooth
  Newton method. *SIAM J. Optim.* 13(3).
- Lawson, C. L. & Hanson, R. J. (1974). *Solving Least Squares Problems.* Prentice-Hall.
- Stellato, B., Banjac, G., Goulart, P., Bemporad, A. & Boyd, S. (2020). OSQP: an operator splitting
  solver for quadratic programs. *Math. Program. Comput.* 12.
- Tibshirani, R. J. (2017). Dykstra's algorithm, ADMM, and coordinate descent: connections,
  insights, and extensions. *NeurIPS*.
