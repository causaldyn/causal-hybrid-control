# ADR 0164 — The CE certificate's Lipschitz constant is bounded, not sampled

**Status:** accepted, 2026-11-24. Research tier (`chc.research.regret`). Changes the values of
`CeExplicitConstantCurve.l_k_local`, `l_k_ball`, `rho`, `c_const` and, through `rho`, the sweep's
fields wherever `multiples` scale it. Adds the field `bound_radius` after `rho`.

## Context

`ce_explicit_constant_certificate` certifies a ball `||dB|| <= rho` inside which the
certainty-equivalent gain stabilises the true plant, and a constant `C` with regret at most
`C ||dB||^2` there:

    rho = theta/(2 beta_B L_K),   C = 2 kappa_P ||R_K|| L_K^2 ||x0||^2 / theta.

Both hold only if `||Khat - K|| <= L_K ||dB||` at every `dB` of the ball. `L_K` was a maximum over
sampled perturbations: 60 random directions at each of three tiny radii for the derivative
(`l_k_local`), 120 at each of three fractions of a first radius for the ball (`l_k_ball`). A
maximum over samples can only fall short of the supremum, so `rho` read wide and `C` low by the
same factor. The docstring called the hypothesis "certified numerically".

It failed on every plant tried whose `B` has four or six entries. Along the top singular direction
of the gain's derivative, which random directions rarely meet once `dB` has a few entries, the
ratio `||Khat - K||/||dB||` at the certified `rho` exceeds the sampled `L_K` on four plants with
`Q = I`, `R = I`, before this change:

| plant | sampled `L_K` | derivative's norm | ratio at `rho` along it |
|---|---|---|---|
| 4-state chain, 1 input | 3.658 | 3.717 | 3.721 |
| 4-state diagonal, 1 input | 0.2884 | 0.2825 | 0.2906 |
| 4-state, 1 input (`_WIDE_PLANTS[0]` in the tests) | 1.760 | 1.814 | 1.822 |
| 3-state, 2 inputs (`_WIDE_PLANTS[1]`) | 0.727 | 0.847 | 0.855 |

On three of them the sample read below the derivative's own norm. The hypothesis both constants
rest on was false at the radius they certified.

## Decision

- **`L_K` comes from a non-local bound on the gain**, in the manner of Konstantinov, Petkov and
  Christov (Kybernetika 29(1), 1993) and Sun (Linear Algebra Appl. 275–276, 1998), written for the
  gain rather than the Riccati solution and derived in `validation/ce_gain_perturbation.mac`. With
  `E = Khat - K` for the model `B + dB`, `||dB||_F <= delta`:
  - the first two Taylor terms `E1` (linear in `dB`) and `E2` (quadratic) are computed as
    operators from `n m` Stein solves and polarisation; `||E1|| <= j delta` with `j` the exact
    norm of the derivative, `||E2|| <= h2 delta^2` with `h2` the norm of its symmetric bilinear
    form's matrix;
  - the remainder `E3 = E - E1 - E2` obeys an exact identity (STEP 2, checked symbolically and, at
    `n = 3`, `m = 2`, in exact rationals in STEP 6) whose every term is bounded by norms, so
    `c_eta ||E3|| <= sigma(||E3||)`, `sigma` a polynomial with nonnegative coefficients. The true
    gain's cost on the model plant, `U` (STEP 3), and the cost-difference identity
    `L_D(U - X) = E' R_hat E` (STEP 4) give `||X||` and `||R_hat||` without the unknown gain;
  - `||E3||` is 0 at `dB = 0` and continuous along `B + t dB`, and the convex polynomial
    `psi = sigma - c_eta eta` is positive below its least positive root, so `||E3||` stays below
    that root at every `dB` of the ball, not only near 0;
  - `e(delta) = j delta + h2 delta^2 + root` bounds `||E||_F` over the ball. Every term of `sigma`
    is `delta` times a function nondecreasing in `delta` once `eta = delta w`, so `e(delta)/delta`
    is nondecreasing and `e(rho)/rho` is a Lipschitz constant on the whole ball of radius `rho`.
- **`rho` is the largest radius with `e(rho) <= theta/(2 beta_B)`**, `l_k_ball = e(rho)/rho`, `C`
  from `l_k_ball`, and `l_k_local = j`, computed, not differenced. Both norms are Frobenius'; the
  sampled ratio had the spectral norm of `dK` over the Frobenius norm of `dB`, so the new constant
  bounds a larger quantity.
- **Where `psi` has no positive root, no bound exists**, and nothing replaces it. `bound_radius` is
  the largest radius at which the bound exists, and `rho` stops there. Both sets are intervals from
  0, as both conditions are monotone in the radius, so each is bisected. No sampled value enters
  `rho` or `C`.

## Consequences

Before and after (`Q = I` throughout; `rho` and `bound_radius` in `||dB||_F`; `L_K` is
`l_k_ball` now and was the larger of the two sampled values):

| plant | `rho` | `bound_radius` | `L_K` | `C` | `conservatism` |
|---|---|---|---|---|---|
| defaults | 0.055209 → 0.036939 | 0.040790 | 0.5672 → 0.8478 | 136.52 → 304.96 | 15.571 → 23.271 |
| `A = [[0.9, 0.2], [0, 1.05]]`, `B = [0, 1]'`, `R = 2` | 0.110525 → 0.072730 | 0.078723 | 0.4456 → 0.6771 | 47.92 → 110.66 | 8.093 → 12.299 |
| scalar `(1, 1, 1, 1)` | 0.221361 → 0.206890 | 0.206890 | 1.3960 → 0.7425 | 16.51 → 4.671 | 4.518 → 4.833 |
| scalar `(1.2, 0.5, 1, 1)` | 0.126219 → 0.072239 | 0.072239 | 3.0534 → 3.5556 | 94.47 → 128.09 | 3.277 → 5.726 |
| scalar `(0.5, 1, 1, 0.1)` | 0.465951 → 0.237819 | 0.237819 | 1.0253 → 0.9315 | 2.471 → 2.039 | 2.327 → 80 (none fails) |
| `_WIDE_PLANTS[0]`, `x0 = 1` | 0.003608 → 0.003256 | 0.006301 | 1.760 → 1.951 | 108 610 → 133 376 | — |
| `_WIDE_PLANTS[1]`, `x0 = 1` | 0.043996 → 0.027633 | 0.030945 | 0.727 → 1.157 | 588.7 → 1492.4 | — |

- **The bound's price.** On the defaults it reads 0.848 where the derivative is 0.548 and 720
  directions at `rho` read at most 0.560 (a 2001-angle grid of the circle, 0.5602): the ball is
  0.67 of the sampled one, `C` 2.2 times larger. The margin binds on the two 2-state plants and the
  two wide ones; on the three scalar plants the bound stops existing first. On `(1, 1, 1, 1)` and
  `(0.5, 1, 1, 0.1)` `C` falls: the old sample was read out to `theta/(2 beta_B l_k_local)`, 5.1
  and 2.7 times the old `rho`, where the gain moves faster than anywhere inside the ball.
- **The empirical radius moves with `rho`** only through the sweep's multiples. On the defaults it
  reads 0.859629 (0.859676 before): the radius is the same failure, found from other draws. On
  `(0.5, 1, 1, 0.1)` the loop fails only for `|dB|` in [1.084, 1.891], 4.56 to 7.95 times the new
  `rho`, between the default multiples 4 and 10, so no draw fails and `conservatism` reads the
  largest multiple, 80: a lower bound only where the failure set is a ray, as the field comment now
  says. `test_ce_empirical_radius_is_the_first_unstable_radius` adds the multiple 6 so the
  bisection has a failing draw to start from.
- **Floating point.** The bound takes the computed `P` as exact. On a plant near the edge of
  stabilisability the Riccati residual can sit far above rounding (6e-6 on one probe), and the
  bound's arithmetic is then not a guarantee. An interval enclosure of the Riccati solution would
  close this; it is not done here.
- **Cost.** The sampling's 540 Riccati solves are gone. The bound builds the `n^2 x n^2` inverse of
  the Stein operator once and evaluates its operators at every unit and every pair of units of
  `dB`; each radius the bisections try costs one root of a degree-4 polynomial.
- **Tests.** `test_ce_gain_bound_holds_over_the_ball_and_is_lost_past_it` samples the true gain
  error at six radii up to `bound_radius` on seven plants and never exceeds `e(delta)`, checks
  `e(delta)/delta` nondecreasing and that no bound exists 1 % past `bound_radius`.
  `test_ce_gain_bound_holds_where_a_sampled_supremum_falls_short` checks the two wide plants along
  the derivative's top direction at `rho`: gain error, contraction and regret. Before this change
  it fails on both at the gain error, 0.006516 against `L_K rho` = 0.006337 and 0.036805 against
  0.031985. `test_ce_gain_bound_radius_is_the_derivation_s_on_scalar_plants` checks `rho`
  against the bfloat bound of `validation/ce_gain_perturbation.mac` STEP 5 to `1e-9`.

## Alternatives

- **The classical non-local bound on the Riccati solution, then the gain from it.** Rejected: its
  fixed point on `X`, in either the Euclidean or the `P` metric, exists only up to 1e-4 to 0.1 of
  the sampled `rho` on the defaults, whose loop is slow (`theta = 0.073`). The Riccati solution's
  perturbation is first order in `dB` through the gain's own cost, which the policy-evaluation
  route (`U`) carries exactly instead of bounding.
- **A first-order majorant only.** It gave `rho = 0.016` and `C = 1615` on the defaults: the
  second-order operators buy a ball 2.3 times wider.
- **More samples, or an optimiser over directions.** Rejected: still a lower bound on the
  supremum, with no statement about the directions not tried.
- **Fall back to sampling where the bound fails.** Rejected: it would restore an unguaranteed
  `rho` silently. `bound_radius` says where the guarantee ends.
- **A verified (interval) Riccati solution.** Would remove the floating-point caveat, at the cost
  of an interval-arithmetic layer the project does not have.
