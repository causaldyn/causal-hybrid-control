# ADR 0003 — A receding horizon warm-started from its last plan

**Status:** accepted, 2026-09-26.

## Context

A controller in the loop re-plans at every step from the state it measures, keeps the first
action, and plans again. `causal_plan` started every solve from zeros, and under a held barrier
(ADR 0002) a solve is a barrier-free descent plus up to thirty rounds, each a full descent: the
work that the previous step had already done was thrown away at every step. The plan API also had
no online form. `mpc_control` runs the loop against a simulated plant and returns trajectories,
without the plan's certificate; a caller whose plant is the world had to rebuild the loop, and the
warm start, around `causal_plan` themselves.

Two further costs showed up once the calls were counted rather than assumed. Every call compiled
programs afresh: the cost history was built from a Python list, whose length is the step count, so
each new count compiled a conversion; the last cost was read by indexing on the device, which
compiles per length; the plan's trajectory ran an eager `lax.scan`, which is traced and compiled on
every call; and the pessimistic solve's stationarity gradient was a closure re-traced per call.
And a barrier handed in as a new `lambda` recompiles the descent, because the function is static
to the compiled program.

What the solution had to keep:

1. **Warm starts change the work, not the answer**: a converged step lands where a cold solve from
   the same state does, to the solver's tolerance.
2. **`causal_plan`'s guarantees hold at every step**, the slack barrier's included: a barrier the
   barrier-free plan clears changes no action.
3. **The certificate travels with the action.** A step returns the whole `CausalPlan`.
4. **Nothing new in the public surface that is solver state** rather than a statement about the
   plan.

## Decision

- `causal_plan(warm_start=...)`: the actions a solve starts from, `(horizon, m)`, projected onto
  the box and the rows before the first step, so it need not be admissible. A malformed or
  non-finite start is refused.
- `chc.mpc.RecedingHorizon`: a dataclass of `causal_plan`'s arguments; `step(x)` returns the plan
  and keeps it as the next step's start. A test holds its fields to `causal_plan`'s arguments, name
  and default.
- **What carries over:** the plan shifted one step, its final action repeated, and — under a held
  barrier — the rounds' last multipliers, shifted the same way. **What does not:** the penalty,
  which is chosen afresh each step, and the barrier-free solve, which still runs first, from the
  shifted actions.
- The multipliers pass through a private `_plan`; `CausalPlan` does not expose them. They estimate
  the multipliers of the *shifted* condition, in the units of the plan's objective, at the rounds'
  last penalty — a solver's working state, not a stable statement about the plan.
- Host-side conversions for the cost history and the last cost; the trajectory and the
  stationarity gradient compiled once at module level.

## Evidence

Counts only: no wall time was taken. The instances are ADR 0002's barrier instances run as closed
loops: an oscillator under velocity floors of `0.8` and `0.3` (horizon 20, 40 steps) and SIR under
a capacity limit (horizon 30, 60 steps), each from the same start, each plant its own model. Total
descent steps over the loop, at three per-descent caps:

| instance | cap | cold | shifted actions | + multipliers, fresh penalty (shipped) | + skip the barrier-free solve |
|---|---|---|---|---|---|
| floor `0.8` | 10 000 | 100 084 | 78 739 | **65 317** | 46 255 |
| floor `0.3` | 10 000 | 260 012 | 255 359 | **166 115** | 127 360 |
| SIR | 10 000 | 67 582 | 70 441 | **65 547** | 62 535 |
| floor `0.8` | 300 | 39 112 | 27 554 | **30 545** | 26 316 |
| floor `0.3` | 300 | 81 028 | 73 990 | **58 334** | 64 433 |
| SIR | 300 | 65 741 | 68 600 | **65 214** | 58 266 |
| floor `0.8` | 100 | 17 157 | 12 534 | **12 108** | 15 874 |
| floor `0.3` | 100 | 36 071 | 32 368 | **30 697** | 27 692 |
| SIR | 100 | 54 356 | 57 215 | **57 102** | 50 879 |

The shipped controller reproduces the prototype's cold, actions-only and shipped columns to the
step at caps 10 000 and 100; the 300 row and the last column are the prototype's. Every run keeps
every applied action certified by its plan's audit and the barrier positive along the realised
path.

At the default cap the shipped loop's realised cost is within `2e-6` of the cold loop's, relative.
At a cap of 100 it is within `5e-6` of the *converged* cold loop's, while the cold loop at that cap
costs `0.26 %` more on the `0.8` floor, because each slack step returns a truncated barrier-free
solve.

The multipliers are most of the saving on the oscillator, where the barrier binds: the actions
alone save `21 %` and `2 %` at the default cap, the actions and multipliers `35 %` and `36 %`. On
SIR the warm start saves `3 %` at the default cap and costs `5 %` at a cap of 100: the barrier-free
optimum there is zero, far from the held plan, so a barrier-free solve started from the held plan
walks away from it before the rounds walk back.

On a short loop (horizon 12, 10 steps, floor `0.3`) the shipped controller took 32 083 descent
steps, the actions alone 48 286 and cold solves 48 240; handed its own multipliers at the same
state, a solve needed 6 rounds against 13 from zero. `tests/test_receding_horizon.py` holds both.

Before the conversions moved to the host, a repeated `causal_plan` call compiled two programs, three
under a barrier, four with a support model and five with both; after, none. Across 70 outputs of the solvers and
plans in both precisions every value is bit-identical to before except `pessimistic_solve`'s
`stationarity`, which moved by `3e-13` relative in `float64` and `4e-5` in `float32`, where the
residual's cancellation amplifies the gradient's last bits.

Across processes, JAX's persistent cache (`jax_compilation_cache_dir`) kept nothing at its default
threshold: it writes only programs that took `jax_persistent_cache_min_compile_time_secs` (one
second) to compile, and none of the first step's did on the tests' oscillator. With the threshold at
0 the first process wrote 69 programs and a second loaded all 73 of its compile requests from disk.
JAX logs `Compiling` before it looks in that cache, so the count of those lines does not change;
its monitoring events do.

The tests were checked by mutation: fourteen edits to the shipped code — no shift, no multipliers
in, multipliers unshifted, no warm actions, the warm start ignored in `causal_plan`, the rounds
ignoring or discarding the multipliers, the slack path keeping stale multipliers, no shape check, no
finiteness check, and each of the four compile fixes reverted. Two survived the first run: an
unshifted hand-over of the multipliers lands on the same plans, and the slack path's multipliers
are read by nothing a converged plan shows. Each is now pinned directly — the second step against
`_plan` from the shifted starts, bit for bit, and a slack barrier's multipliers against zeros.

## Consequences

- A converged step's plan is `causal_plan`'s from that state, to tolerance. A step stopped by
  `steps` differs from a cold one: on the oscillator for the better, on SIR by `2e-6`.
- The first step compiles, and so does the first step that needs a path the others did not — the
  rounds compile the first time the barrier binds. After that a step compiles nothing, as long as
  the model, the cost and the barrier's function stay the same objects.
- Each step is one `causal_plan`, so nothing outside its horizon carries over: a budget row caps
  every window rather than the run, and a rate limit does not reach back to the action already
  applied.
- `step` updates the warm start in place, unguarded: one controller per control loop. A new
  controller is a cold start that reuses the compiled programs.

## Alternatives

- **Inherit the grown penalty with the multipliers** — measured on the prototype, and the reason
  the penalty is fresh. The next rounds start so ill-conditioned that their descents stall, and the
  loop's realised cost came out `25 %` (floor `0.3`) and `12 %` (SIR) above the fresh-penalty
  loop's at the default cap, and `112 %` on the `0.8` floor at a cap of 300 — every step still
  reporting `converged`, since the condition is met and the descent stopped by its own rule.
- **Skip the barrier-free solve**, starting the rounds from the warm plan with its multipliers —
  measured, the last column above. It is the cheapest in 7 of 9 runs and 13 % cheaper in total,
  but it costs 31 % more on the mostly slack `0.8` floor at a cap of 100, where every slack step now
  pays for rounds, and it gives up `causal_plan`'s guarantee that a slack barrier changes no
  action. The first thing to revisit if instances like SIR — a barrier-free optimum far from the
  held one — dominate the use.
- **Shifted actions only** — simpler, no private hand-over; the second column. It leaves most of
  the saving on the table where the barrier binds.
- **The multipliers as a public field of `CausalPlan`** — they would read as the price of safety
  per step, but their scale is the rounds' own and they describe the shifted condition; not done
  without a use that needs them.
- **Compile `rollout` itself** — would fix every eager caller in the library, not only the plan's
  trajectory, but it changes a public function everywhere; not done here.

## References

- Diehl, M., Bock, H. G. & Schlöder, J. P. (2005). A real-time iteration scheme for nonlinear
  optimization in optimal feedback control. *SIAM J. Control Optim.* 43(5).
- Rawlings, J. B., Mayne, D. Q. & Diehl, M. (2017). *Model Predictive Control: Theory,
  Computation, and Design*, 2nd ed. Nob Hill Publishing.
- Birgin, E. G. & Martínez, J. M. (2014). *Practical Augmented Lagrangian Methods for Constrained
  Optimization.* SIAM.
