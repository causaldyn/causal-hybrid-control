# ADR 0014 — The global effect of a marketplace experiment, read off its rents

**Status:** proposed, 2026-09-28.

## Context

An experiment on a matching market treats some of the rows: some riders see a new feature, some
requests get a new price. The treated rows compete with the control rows for the same columns, so
the prices every row faces move with the share treated. The treated-minus-control difference an
A/B test reads is therefore not what treating every row would do (Johari, Li, Liskovich and
Weintraub 2022). Bright, Delarue and Lobel (2025) read the global effect off the LP's shadow
prices instead. `chc.matching` solves the entropic version of the same matching, and its dual
potentials were so far only a surge price.

The lab's shadow-price thread (SH1–SH6) worked this out for the entropic matching, and an
independent verifier checked it with a second solver, 60-digit Newton and FriCAS:

- **SH1.** Each row's rent is the derivative of welfare in its own mass. So the
  treated-minus-control difference of rents is `f'(p)`, the slope of welfare in the treated
  share, and by concavity `f'(1) <= f(1) - f(0) <= f'(0)`. Known (Bright, Delarue and Lobel).
- **SH2.** `f'(p) - (f(1) - f(0)) = (1/2 - p) K delta^2 / eps + O(delta^3)`, and
  `-f''(p) = eps g' C^+ g` can be read off the experiment's own plan. The coefficient of `delta^k`
  is a polynomial in `p` of degree `k - 1` with mean zero on `[0, 1]`. So `p = 1/2` removes the
  second order, and the Gauss–Legendre pair `p = 1/2 -+ 1/(2 sqrt 3)` removes the second, third
  and fourth. Not found in the literature.
- **SH4.** The naive difference misses at first order, by the displacement it prices:
  `(1/eps) sum_i a_i Cov_pi_i(beta, delta_i)`.
- **SH5.** In the LP limit the bias saturates at first order instead of growing, and the prices'
  predicted movement `nu_hat` works as an alarm, not as a measure.
- **The conventions matter.** Each copy's potential is converted with its own mass, and the
  entropy is taken relative to the split masses. Otherwise the arms differ by
  `eps log(p / (1 - p))` at zero effect.

## Decision

- **`shadow_price_effect(cost, supply, demand, treated, *, eps, randomised=None, strata=None,
  iters, tol)`** solves the experiment's market with `sinkhorn` and returns a
  `ShadowPriceEffect`.
  - **The rows are the randomised units.** To randomise the other side, the caller passes the
    transposed market. Rows outside the experiment, such as an idle pool of supply, are priced
    but not counted (`randomised`).
  - **`strata` post-stratify the experiment:** one label per row, fixed before the assignment.
    The effect then weights each stratum's contrast by the stratum's randomised mass, and the
    standard error, the curvature's noise and `nu_hat` are read the same way. A stratum with fewer
    than two rows in an arm is refused, not merged.
  - **The rents come from the column potentials alone,** by the c-transform. That is exact for
    those potentials even where the row potentials lag half an iteration, and it holds for a row
    of any mass, so no mass conversion is needed.
  - **`effect` is a Hajek contrast of the rents,** scaled to the randomised mass. `naive` is the
    same contrast of the surplus each row's matches earn it, and `standard_error` is the effect's
    over the assignment, as for independent coins.
  - **`curvature` is the plug-in `eps g' C^+ g` less its assignment noise,**
    `eps tr(C^+ Var(g))` with the Hajek covariance of `g`. `second_order_bias` is
    `(1/2 - p) curvature`.
  - **`nu_hat` above 1, or a marginal residual over `tol`,** leaves `second_order_bias` at `None`,
    with the reason in `reason` and a warning in the log.
- **`shadow_price_interval(...)`** is the same reading at `eps = 0`. The exact LP's optimal duals
  are not unique, so it returns the range of the effect over the optimal dual face, from two LPs
  (HiGHS). That range is `[f'(p+), f'(p-)]`.

## Consequences

The tests in `tests/test_matching.py` compute welfare with a NumPy log-domain Sinkhorn of their
own, not with `sinkhorn`.

- **The envelope.** On three random markets at shares 0.2, 0.5 and 0.8, one of total mass 2.7,
  the effect equals a five-point difference of welfare in the share to 2e-12 relative. The
  lab's Newton solver gives the same effect, naive difference and plug-in curvature to 1e-14.
- **The bracket.** For a treatment far outside the second-order regime, the effect falls with the
  share at eleven shares, and brackets the global effect.
- **The orders.**
  - The effect at `p = 0.2` misses at order 2 (slope 2.00), and its miss comes to 1.001 of
    `(1/2 - p) K delta^2 / eps` at the smallest treatment.
  - At `p = 1/2` it misses at order 3 (3.04), and the naive difference at order 1 (1.01).
    The naive miss comes to 1.003 of the priced displacement.
  - The Gauss–Legendre pair's miss has local slopes 5.02 and 4.95 in float64 on two markets. On a
    third it is still pre-asymptotic at 5.7, as the verifier saw before a sign change.
- **Over assignments,** 400 at shares 0.2 and 0.5 on six markets (80 units against 80 others at
  three balances of supply and demand, and 80, 320 or 1280 units against three zones):
  - the effect's spread was 0.90–1.06 of its standard error, and 0.39–0.89 of the naive
    difference's;
  - over 200 of them, the plug-in curvature read `-f''(p)` 1.3–4.7 times too large against
    zones, and 2–12 times against 80 units, all of the excess the assignment's noise in `g`;
  - less that noise, it read 0.71–1.09 of it, each within 1.7 standard errors;
  - its spread from one experiment was 2.1–4.2 times its value at 80 and 320 units against zones,
    0.8–1.1 times at 1280, and 1.3–5.8 times on 80 units against 80 others;
  - at `p = 0.2` the second-order bias it prices was 0.4–7% of the effect's standard error on
    those markets, 7% where demand is short. At `p = 1/2` it vanishes.
- **The part of the bias that shrinks with the number of rows,** which `second_order_bias` does
  not price. Read with a control variate, the same contrast of the rents of the market split into
  a treated and a control copy of every row, whose mean over the assignment is exactly that
  market's `f'(p)`, it was under 0.3% of the standard error on five markets. Where demand is short
  on 80 units against 80 others it was -0.0010 and -0.0009: 2–3% of the standard error, 1.1–1.3%
  of the global effect, and at `p = 1/2` the larger part of the bias. The lab's two-sided study
  found the same -1.2% of the global effect there.
- **The alarm.** Over twenty markets swept towards the LP limit, the second-order law gave the bias
  to within 10% in nine cases of ten where `nu_hat < 0.5`, and within 33% where it was below 1.
  Past 1 the median miss was 24–70%. The assignment's noise inflates `nu_hat`, so the alarm errs
  towards withholding the claim:
  - against three zones at `eps = 0.3`, where the prices move 0.20–0.28 `eps` in the
    large-sample limit, it read 0.51–0.64 on 80 units and withheld the claim in 1–7% of the
    experiments; on 320 and 1280 units it read 0.28–0.35 and withheld it in none;
  - on 80 units against 80 others, where they move 0.17–0.55 `eps`, it read 0.77–1.81 and withheld
    the claim in 4–100% of them;
  - towards the limit, on 80 and 320 units against three zones, it withheld the claim in 86% and
    58% of the experiments where the prices move 0.61–0.63 `eps` in the limit, and in 95–100%
    where they move 1.09–4.68.

  Where it withholds the claim for noise, the curvature it would have priced was noisier still, at
  2–6 times its value.
- **Post-stratified** by the rows' four types, with the types too small for two rows an arm at the
  share merged before the assignment, over 400 assignments at `p = 0.2` and `1/2` on the same six
  markets:
  - the effect's spread was 0.52–0.78 of the plain effect's, and 0.85–1.04 of its standard error,
    which errs conservative;
  - the curvature's mean was 0.83–1.11 of `-f''`;
  - the alarm withheld the claim in at most 1% of the experiments on 80 units against three zones,
    where plain it had in 3–7%, and nearly as often as plain on 80 units against 80 others: 75%
    against 85% on the balanced market at `1/2`, within two points everywhere else;
  - the refusal set aside 16–24 assignments of 400 on 80 units at `p = 0.2`, 3 at `1/2`, and none
    on 320 or 1280.
- **Strata fixed after the assignment bias the effect.** Merging the strata an assignment left
  short of two rows into the largest, the natural repair, moved the effect on average by 8–24% of
  the global effect at `p = 0.2`, 3–6 standard errors over 400 assignments on each of the five
  markets where it happened, and by nothing resolved at `p = 1/2`. The shift is in the design,
  not the market: the same weights on the rents of the split market, whose mean is exactly
  `f'(p)` when the strata are fixed, moved as far. With the types merged by size before the
  assignment, that mean was `f'(p)` within the Monte Carlo's error. Falling back to the plain
  difference on the assignments that leave a stratum short is no repair either. The plain
  difference on the other assignments is off by the imbalance the refusal selects, so the
  fallback moved the effect by 5–10% of the global effect. Hence the refusal, and the docstring's
  advice: fix strata large enough that a short arm is unlikely, or assign within them.
- **Float32.** A float32 solve moved every number by about 1e-7 relative, and its residual stayed
  under `tol`. The residual check is the guard, not the dtype.
- **At `eps = 0`.** At a kink of welfare in the share the interval is `[f'(p+), f'(p-)]`: on the
  test's market, 0.035 wide, to 2e-9. At the lab's 40 breakpoints, 4e-4 to 0.51 wide, it matches
  the lab's face range to 3e-16, and the one-sided slopes to 3.5e-9, the finite difference's own
  error. Between kinks it is a point, the slope: on the same 40 markets at three shares it matches
  the lab's to 1e-15.
- **25 tests. 34 mutants, 31 caught.** The first 23 were on the estimator without `strata`; 21
  caught. Then 11 on `strata`: 10 caught, among them the treated share and the curvature's noise
  read from the first stratum alone, which survived until the tests read the share and renamed the
  strata; and one equivalent, the spread about the unweighted mean of the arm's rows, which is the
  weighted mean when the rows' masses are equal. Two survive from the first 23:
  - the within-arm variance's `n/(n - 1)`, 1.016 at the tests' smallest arm: equivalent;
  - the interval's gauge. The contrast sums to zero, so the gauge moves neither end. Without it the
    ends still matched the lab's, but to 1.5e-14 at the breakpoints, 45 times further. It stays.

## Not built

- **The two-sided estimator `f_RS = eps g' C^+ u`.** The verifier refuted "two-sided
  randomisation is dominated" outside the lab's family: a mixed curvature read off a `(1/2, 1/2)`
  design is second-order unbiased too. Over 10 000 paired assignments on the three 80 × 80
  markets, its RMSE was 0.34–0.60 of the one-sided effect's at `p = 1/2`, but 0.85, 0.94 and
  1.04 of the one-sided effect post-stratified by the rows' types, and 0.84–0.98 post-stratified
  itself. It needs both sides randomised, independently, and it has no variance estimator.
  Post-stratification, with one side randomised and a standard error, was built instead. The lab's
  post-stratified effect merged strata left without an arm after the assignment, the repair found
  biased above; at `p = 1/2`, where that study ran, it moved nothing this study resolves.
- **Secondary metrics** (the lab's R6): the global effect on matches or revenue rather than
  welfare. The formula was checked against finite differences to 4e-10 in the lab, but it has no
  bias certificate of its own, and no caller has asked for it.
- **A design helper** that chooses the share and the side. The rule is in the docstring: one side
  at `p = 1/2`, or the Gauss–Legendre pair.
- **A standard error for the curvature,** or a corrected effect `effect - second_order_bias`. At
  the sizes measured, the curvature is noise-dominated, and the bias it removes is a few percent
  of the standard error.
- **A price for the finite-N part of the bias.** It needs the split market, which one experiment
  does not give, and it was at most 3% of the standard error on the markets measured.

## Alternatives considered

- **The plug-in curvature.** It is exact for a market whose rows are the fixed types' treated and
  control copies. On sampled units it reads 1.3–12 times too large.
- **A two-fold cross-fit of the curvature.** It is unbiased within noise too (0.96–1.34), but its
  spread was 1.06–1.30 times the noise-corrected estimate's at every size.
- **An alarm at `nu_hat > 2`.** The lab put the law within 8% up to 2.2 on one market. On twenty,
  the median miss between 1.5 and 2 was 70%.
- **An alarm on the root mean square of the price movement over the rows' matches,**
  `sqrt(curvature / (eps M))`, which the noise-corrected curvature reads without the inflation. On
  the twenty markets it separated the law's misses as well as `nu_hat` did: below 0.25, the 90th
  percentile of the miss was 17%, against 18% below `nu_hat = 1`. But the corrected curvature,
  clipped at zero, is too noisy to raise an alarm on: at `eps = 0.05` on 80 units against three
  zones it fired in 35% of the experiments, where `nu_hat` fired in 97%. The plug-in's root mean
  square fired in 49% of the experiments far from the limit, where `nu_hat` fired in 7%.
- **A `ShadowPriceEstimator` class holding `eps` and the side** (the lab's sketch). A function that
  solves the market itself cannot be handed an `eps` other than the one its potentials were solved
  at, and a transposed market is one argument, not a mode.
- **Taking a finished `SinkhornResult`.** A result does not carry its `eps`, and a mismatched `eps`
  would read wrong rents without an error.
- **A refusal on float32.** The measured error is 1e-7 relative, and the residual check already
  catches the solve it would stop.
