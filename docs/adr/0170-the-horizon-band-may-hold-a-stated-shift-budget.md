# ADR 0170 — The horizon's band may hold a stated shift budget

**Status:** accepted, 2026-11-25. Experimental (`HorizonConformal`). Amends ADR 0099 (the tube's
levels are measured beside it). The default, `shift=None`, does not change, and no value it reads
moves.

## Context

`HorizonConformal` (ADR 0099) scales a profile by the `ceil((n + 1)(1 - delta))`-th smallest of
`n` window scores, and a fresh window lies within the band at every step with probability at least
`1 - delta` when it is exchangeable with the calibration windows. A deployed window is not: the
season moves, the policy drifts, the plant ages. The band then states nothing, and a small shift
takes its level. On calibration scores uniform on `[0, 1)`, 39 windows and `delta = 0.1`, a law
that moves the lowest 5 % of the scores beyond every band, a total variation of exactly 0.05 from
the calibration law, is held by the band on `36/40 - 0.05 = 0.85` of draws in expectation, not 0.9.

Cauchois, Gupta, Ali and Duchi (JASA 2024) state conformal coverage over an f-divergence ball
`D_f(P || P0) <= rho`. Their Proposition 1: over the ball, an event of probability `beta` under
`P0` keeps at least

`g(beta) = inf { z in [0, 1] : beta f(z / beta) + (1 - beta) f((1 - z) / (1 - beta)) <= rho }`.

Their Theorem 1 bounds the coverage of the `ceil(n beta)`-th smallest calibration score, for a
fresh score drawn from `P`, by `g(ceil(n beta) / (n + 1))`; the argument holds at any rank `k`,
`g(k / (n + 1))`, since `g` is convex and nondecreasing. So a band at `1 - delta` over the ball
reads the rank `ceil((n + 1) beta*)`, `g(beta*) = 1 - delta`.

## Decision

`HorizonConformal.calibrate(errors, profile, delta, shift=("tv", eps))` or `shift=("kl", rho)`
holds `1 - delta` for every law of the fresh window within that ball of the calibration windows'
law. The band reads only the scores, and a ball on the windows' law holds the scores' law in it.
The fitted band keeps the ball as its field `shift`.

- **Total variation**, `eps = sup_A |P(A) - P0(A)|`, `f(t) = |t - 1| / 2`. The two-point
  divergence is `|z - beta|`, so `g(beta) = max(0, beta - eps)`, `beta* = 1 - delta + eps` and
  the rank is `ceil((n + 1)(1 - delta + eps))` (`validation/conformal_shift_budget.mac` (1)). It
  is computed as written, in exact arithmetic: at `n = 59`, `delta = 0.1`, `eps = 0.05` the
  floats' `60 * 0.95` reads 57.00000000000001 and asks for the 58th score where the 57th holds.
  Their Corollary 2.3 reads the empirical quantile at `1 - delta + eps` and holds
  `1 - delta - 1/n`; the `n + 1` rank holds `1 - delta` itself.
- **Kullback-Leibler**, `KL(P || P0)`, the deployed law's divergence from the calibration law,
  `f(t) = t log t`. The two-point divergence is `KL(Bernoulli(z) || Bernoulli(beta))`. It is
  jointly convex in `(z, beta)`: its Hessian has a positive diagonal and the determinant
  `(z - beta)^2 / (z (1 - z) beta^2 (1 - beta)^2)` (2), so `g` is convex, the step from
  `E g(F0(S_(k)))` to `g(k / (n + 1))`. `beta*` is the root above `1 - delta` of
  `KL(Bernoulli(1 - delta) || Bernoulli(beta)) = rho`, unique since the divergence's derivative
  in `beta` is `(beta - tau) / (beta (1 - beta)) > 0` there (3). The code solves it in
  `log(1 - beta)` and keeps the gap `1 - beta*`, since the root lies within `1.7e-7` of 1 at
  `delta = 0.01`, `rho = 0.1`, and `1 - beta` formed from `beta` keeps `1e-16 / (1 - beta)` of
  itself. The rank is `n + 1 - floor((n + 1) gap)`, the same integer as the unshifted
  `ceil((n + 1)(1 - delta))` where the gap is `delta`. At five points the gap is Maxima's to
  `1e-12` of itself. Near `rho = 0` the root moves by `sqrt(2 delta (1 - delta) rho)` (4).
- **The radius is the caller's statement.** It is the largest shift between the calibration
  windows and the deployed ones the band is to survive. Nothing here estimates it, and the
  library chooses no value for it: `shift=None` is the default, and a radius of 0 reads the
  unshifted band, rank, scale and level, exactly. The band says nothing of a shift larger than
  the radius.
- **Refused.** A ball other than `"tv"` and `"kl"`; a radius that is negative or not finite; a
  total variation of `delta` or more, which can move `delta` of the windows beyond any radius, so
  no number of windows gives a finite band; and a radius whose rank passes `n`, with the number of
  windows that would hold it: 19 for `eps = 0.05` at `delta = 0.1`, against 9 unshifted; 31 for
  `rho = 0.05`, 568,532 for `rho = 1`, and 5,957,353 for `rho = 0.1` at `delta = 0.01`
  (`conformal_shift_budget.mac` (3)). Past `1e15` windows the floats no longer count them, and
  the message says so.
- **On one series** (`mixing=`). The ball bounds the fresh window's law given the calibration
  windows. Given them, the band covers `C` under the series' own law and at least `g(C)` under
  any law in the ball; `g` is convex and nondecreasing and the series keeps `E C >= beta* - eta`
  (ADR 0099), so the band holds `g(beta* - eta)`, which `level` states: `1 - delta - eta` under
  total variation, and under KL the root `z < beta* - eta` of `KL(z || beta* - eta) = rho`, less.
  It refuses where `eta >= delta`, as before, and under KL where `-log(1 - beta* + eta) <= rho`:
  the ball can then take all that is left (5).
- **No calibration-conditional rank.** `HorizonConformal` reads the marginal rank only; the shift
  composes with it and with the series' `eta`, and with nothing else.

## Consequences

- **The band holds its level at the ball's edge** (`tests/test_tube_levels.py`). 20,000 draws of
  39 calibration windows and one fresh window, scores uniform on `[0, 1)`, `delta = 0.1`,
  `eps = 0.05`: the shifted band, the 38th score, holds the law that moves the lowest 5 % beyond
  every band on 0.90265 of the draws, its exact expectation `38/40 - 0.05 = 0.9`, and the
  unshifted band, the 36th, on 0.85135 (0.85). Where a share of 0.05 of the windows lies beyond
  every band instead, they hold 0.90105 (0.9025) and 0.854 (0.855). Under KL, 99 windows and
  `rho = 0.05`, the law that puts `0.9` below `beta* = 0.96872` and the rest above, at a
  divergence of exactly `rho`, is held by the shifted band, the 97th score, on 0.91555 (0.91791),
  and by the unshifted one, the 90th, on 0.8356 (0.83616) (`conformal_shift_budget.mac` (6)).
  Each test fails where the shift is ignored.
- **The price.** A total variation of `eps` costs the rank about `(n + 1) eps` scores, and at
  `delta = 0.1` the windows needed grow from 9 to 19 at `eps = 0.05`. A KL ball is dearer the
  smaller `delta` is, and it moves the level even at a tiny radius: near `rho = 0` the gap falls
  by `sqrt(2 delta (1 - delta) rho)`, and at `delta = 0.01` a radius of 0.1 asks for nearly six
  million windows.
- **What the two balls hold.** A KL ball holds no law with mass where `P0` has none; a total
  variation ball does, at a price that does not grow with how far the mass moves.
- **What it still covers.** Windows whose law is within the radius of the logging policy's. A
  caller who bounds the shift the plan's own actions cause can read the band on them; the library
  does not bound it.
- **No number moves.** The field `shift` defaults to None, `level` reads as before without it, and
  the unshifted refusals keep their words.

## Alternatives

- **Other f-divergences.** The two-point reduction holds for any `f`, and the chi-square ball
  even has a closed form, `g(beta) = max(0, beta - sqrt(rho beta (1 - beta)))`. Not built: the
  two balls cover a shift with new support and one without, and a third adds a radius in other
  units for the caller to state.
- **An estimated radius,** a divergence read between the calibration windows and recent deployed
  ones. Rejected: the radius would then be a random quantity, and the guarantee would have to
  carry its error. The radius stays the caller's statement.
- **Weighted conformal under covariate shift** (Tibshirani, Barber, Candes and Ramdas, NeurIPS
  2019). Exact where the likelihood ratio of the deployed law to the calibration law is known.
  Not built: no such ratio is known here, and a ball needs only its radius.
- **Their Corollary 2.2's corrected miscoverage** `1 - g((1 + 1/n) beta*)`. Not needed: the
  `n + 1` rank holds `g(k / (n + 1)) >= 1 - delta` by Theorem 1's argument directly.
- **Two arguments, a `divergence` name and a `radius`.** One tuple keeps the pair together, and
  None is the one switch that turns it off.
