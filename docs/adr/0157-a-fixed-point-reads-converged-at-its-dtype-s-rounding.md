# ADR 0157 — A fixed point reads converged at its dtype's rounding

**Status:** accepted, 2026-11-21.

## Context

`chc.games.fixed_point` iterates `x <- f(params, x)` until the relative residual
`||f(x) - x|| / max(1, ||x||)` is at most `tol`, 1e-6 by default, and reported
`converged = residual <= tol`. In float64 that tolerance sits far above rounding. In float32 it is
8.4 machine epsilons, and the residual of a solve that sits on its equilibrium cannot fall far
below its own rounding. On the logs of `SharedStateMarket(seed=0..29)`, 400 blocks each and both
payments, the logit congestion solve ended above 1e-6 on 88 of 24,000 blocks, at most 2.0e-6 or
16.6 epsilons, and `converged` read False on each. On the nine such blocks of seed 7 the drivers
were within 7.3 epsilons of the float64 equilibrium, no further than on the other 391 blocks.

0.24.1 worked around this inside `SharedStateMarket._equilibrium` alone, which raised where the
residual passed `max(1e-6, 64 eps)` and ignored the flag. Every other reader of the flag, the
research certificates and any caller's own code, still read a float32 solve at its equilibrium as
a failure.

## Decision

- **`converged` reads `residual <= max(tol, 64 eps)`,** `eps` the machine epsilon of the solve's
  dtype. In float64 that is `tol` for any `tol` from 1.4e-14 up. In float32, at the default `tol`,
  it is 7.6e-6.
- **The iteration still stops at `tol`.** A float32 solve at its floor iterates to `max_iter`, as
  before, and the returned `x` keeps every bit in both precisions.
- **The market reads the flag.** `SharedStateMarket._equilibrium` raises where `converged` is
  False, at the same residuals as before, and keeps no floor of its own.

## Consequences

- In float32, `EquilibriumSolution.converged` is True on solves whose residual lies between `tol`
  and 64 epsilons. A solve that 2-cycles still reads False: the 2-cycle of the module's tests ends
  at a relative residual above 0.1.
- In float64 the flag moves only for a caller who asks for less than 1.4e-14, which no caller in
  the library or its tests does: there it reads True at 64 epsilons rather than never.
- No number moves: the iterates, the market's logs, its raises and every plan are bit-identical.
- `tests/test_games.py::test_a_float32_solve_at_its_rounding_reads_converged` solves seed 7's 400
  blocks in float32, finds nine residuals above 1e-6, and needs every flag True and those drivers
  within 16 epsilons of the float64 equilibrium. Before this change the flag read False on the nine.

## Alternatives

- **Stop the iteration at the floor too, or default `tol` to `max(1e-6, 64 eps)`.** Rejected: in
  float32 every solve would stop at 7.6e-6 rather than at 1e-6, up to 7.6 times less accurate, to
  save the iterations of the 0.37 % of solves that reach the floor. Every float32 log would change
  its bits.
- **Stop where the residual stops falling.** Rejected for now: it saves the floored solves'
  iterations but moves their bits, and the cost of those iterations was never measured to matter.
- **Leave the flag and keep the floor in each reader.** Rejected: the 0.24.1 floor lived in one
  reader, and the others read float32 solves at their equilibrium as failures.
