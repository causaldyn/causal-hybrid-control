# ADR 0162 — The CE certificate's empirical radius is bisected

**Status:** accepted, 2026-11-24. Research tier (`chc.research.regret`). Changes the values of
`CeExplicitConstantCurve.empirical_radius` and `conservatism` wherever a draw destabilises. Every
other field is bit-identical.

## Context

`ce_explicit_constant_certificate` computes the radius `rho` inside which a certainty-equivalent
gain provably stabilises the true plant, then sweeps `||dB||` over multiples of `rho`, 0.25, 1, 4,
10, 20, 40 and 80 by default, with `n_dirs` random directions at each. `empirical_radius` was `rho`
times the first multiple at which a draw failed, a gain that does not exist or a true loop of
infinite cost, and `conservatism` was that radius over `rho`, "how much the certificate gives
away".

The first failing multiple brackets the first unstable radius from above only, by up to the step
to the next multiple, a factor of 2.5 to 4. On three scalar plants, `(a, b, q, r)` = `(1, 1, 1, 1)`,
`(1.2, 0.5, 1, 1)` and `(0.5, 1, 1, 0.1)`, the exact first unstable radius is the nearest root of a
polynomial: a crossing of `|a - b khat| = 1` fixes the model's Riccati root linearly, that root
must solve the model's DARE, and at `|a| >= 1` the point `b_hat = 0`, where no gain exists, is a
failure too (`validation/ce_explicit_constants.mac` STEP 9, all real roots by Sturm sequences):

| plant | `rho` | read off the multiples | first unstable radius |
|---|---|---|---|
| `(1, 1, 1, 1)` | 0.22136 | 2.2136 (10 `rho`) | 1 (4.518 `rho`) |
| `(1.2, 0.5, 1, 1)` | 0.12622 | 0.50487 (4 `rho`) | 0.41362 (3.277 `rho`) |
| `(0.5, 1, 1, 0.1)` | 0.46595 | 1.8638 (4 `rho`) | 1.08419 (2.327 `rho`) |

On the third plant the loop fails only for `|dB|` from 1.084 to 1.891 on its failing side, and is
stable again beyond: no multiple from 10 up sees it fail. The guarantee, `rho` and `C`, does not
depend on the sweep; the reported price of it was 1.2 to 2.2 times too high on these plants.

## Decision

- **Each failing draw's direction is bisected** between the true plant, which the optimal gain
  stabilises, and the least radius a draw is known to fail at, wherever the direction fails there
  as well, to a width of `1e-6 rho`. The failing end is kept, so the reading is a radius at which a
  draw was seen to fail, and the least radius known moves down as the directions are taken in turn.
- **The failure rule is the sweep's**: no gain, or an infinite cost of the true loop under it.
- **Where no draw fails, nothing changes**: the reading is the largest radius swept, a lower bound.
- **The width is fixed**, a module constant, not an argument: `conservatism` is then exact to
  `1e-6` wherever the sampled directions hold the worst one and its loop crosses once on the way
  out.

## Consequences

- On the three plants above the radius reads 1, 0.41362 and 1.08419, each within `1e-6 rho` of the
  exact one, and the conservatism 4.518, 3.277 and 2.327.
- On the defaults, a 2-state, 1-input plant, the radius moves from 1.10418 to 0.859676 and the
  conservatism from 20 to 15.571. A scan of 1440 directions on the circle, each bisected from a
  0.005 grid, read no less: 0.859864.
- The reading is an upper bound on the least destabilising `||dB||` over all directions, as any
  failing draw is: the directions are sampled. Where a direction's loop fails, recovers and fails
  again below the least radius known, the bisection finds one of its crossings and can skip it.
- The defaults take 351 more Riccati solves, 2572 against 2221. The sweep's draws, `worst_gap`,
  `worst_lyapunov`, `unstable_fraction` and every constant are bit-identical.
- `tests/test_formal_regret_a.py::test_ce_empirical_radius_is_the_first_unstable_radius` checks
  the three plants against STEP 9's radii to `1e-6 rho`. Before this change it fails on all three.

## Alternatives

- **A finer grid of multiples.** Rejected: still a bracket from above, and the cost grows with the
  levels times the draws.
- **Bisect between the last clean multiple and the first failing one.** Rejected: the last clean
  multiple was clean for other draws, not for this direction. From the true plant the bisection
  takes at most one more halving.
- **Bisect every draw, stable or failing.** Rejected: a stable draw's direction needs a failing
  radius to bracket it, which only a further search finds, at a cost of a bisection per draw.
- **Solve `|a - b khat| = 1` in closed form.** Exact on a scalar plant, which STEP 9 does, but not
  on the matrix plants the certificate takes.
