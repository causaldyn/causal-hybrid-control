# ADR 0120 — The nuisances' sieve is chosen out of fold, and the check reads the move in its own error

**Status:** accepted, 2026-10-21. Amends ADR 0107. The default, `nuisance_knots=None`, is
unchanged.

## Context

Three defects of the spline nuisances of ADR 0107 and of the default fit's check on them.

- **The spline cost efficiency on heavy tails.** Beside its spline a covariate's square holds its
  shape where the spline does, between the knots. So the square's coefficient is read off the rows
  past the outer knots alone, where the spline goes on as a line and the square does not. A row of
  the other fold further out reads it at the square of its distance. On a linear plant with two
  `t(3)` confounders, which the polynomial holds, one confounder lay at −18 of its scales; the
  action's cross-fitted residual there was 11 where its noise is 0.5. Over 200 logs of 500 and
  2000 rows the channel scattered 3.0 and 4.4 times as far as the polynomial's. With a nuisance
  quadratic in a `t(5)` confounder, 3.6 to 4.0 times.
- **The check read the move in the fit's own error.** A refit that scatters more than the fit
  moves by more than its error with no bias at all: the check warned on 12.5 % of linear logs
  with `t(3)` covariates. The difference of the two variances (Hausman 1978) is the move's
  variance only where the fit is the efficient one; on heavy tails it was not, and the difference
  can read below nothing. On a log the known physics solves exactly both errors are rounding, and
  their ratio read up to 49.
- **Under `rk4` a covariate's push on the rate entered the drift regression linearly.** The rest
  of the push was left in the drift's residual, and moved the channel through RK4's growth
  factor. With a push `z + 1.5 (z² − 1)` the channel scattered 31 times its error and its interval
  covered on 1 of 40 logs.

## Decision

- **Three sieves, one chosen per log.** The spline with each splined covariate's powers past the
  first, the spline without them, and the polynomial alone. Each is cross-fitted on the log's
  Euler rate less the known physics and on the actions; the sieve whose summed out-of-fold
  residual mean squares, each over its column's variance, are the smallest is read, the first on
  a tie. The choice reads the outcome, as a learner chosen by cross-validation does; the moment is
  orthogonal to the nuisances' error, so the choice enters at second order.
- **The check reads the move in the move's own error.** Both fits are linear in the target, `G y`.
  A row's share of the move is `G_s,i e_s,i − G_i e_i`, each fit's weight on the row times its own
  residual there. Their squares, summed within the fit's clusters or periods as the channel's
  error sums its own, are the move's variance where both fits are right, whichever is the more
  efficient (`validation/paired_score_shift.mac`). The weights come from one `jax.jacrev` of the
  spline fit's channel, and under `rk4` both are carried to the fixed point through the fit's own
  Newton matrix. The error is not read below the square root of the working precision of a
  channel that moves the rate by its own size, where the fixed point's Newton steps stop. Where
  the chosen sieve is the polynomial the move is nothing.
- **The warning's limit is a `t`'s.** The move's error is read off as many independent terms as
  the channel's, `G − 1` clusters or `nu` cosines of the periods. The fit warns where the move
  passes the quantile of that `t` as far into its tail as three is into the normal's: 3.0 at many
  rows, 4.1 at `t(9)`, 236 at `t(1)`.
- **Under `rk4` the pushes are held in the nuisances' own basis**: the sieve's columns in the
  covariates that are not drivers, at degree 1 at the least, about their means, orthonormal.

## Consequences

- Where the polynomial holds the truth, `"auto"` scattered as wide as the polynomial alone or
  less, each over 100 logs: 1.01 and 1.00 times at 500 and 2000 rows with normal covariates, 0.67
  and 1.00 with `t(3)` ones, 0.43 and 0.76 with lognormal ones, and 1.00 with a nuisance quadratic
  in a normal or `t(5)` confounder at 1000 and 4000 rows. On the linear plant of ADR 0107, 1.00
  times over 200 logs of 1000 rows, both covering on 0.965.
- On the plant of ADR 0107, which the polynomial misses, `"auto"` left a bias of 0.13, 0.04 and
  0.08 of its error at 1000, 4000 and 16000 rows and covered on 0.960, 0.965 (200 logs) and 0.967
  (30).
- The check warned on 40 of 40 logs of that plant at 2000 rows, the move reading 31 to 44 of its
  errors. On 200 linear logs it read 0.90 at most, 1.6 where the noise grows as `exp(x)`, and
  warned on none; on 60 logs each with `t(3)`, normal and lognormal covariates at 500 and 2000
  rows, and of nine linear designs under `euler` and `rk4`, on none; on 20 logs each the known
  physics solves exactly, at noise 0 to 1e-3, on none. On fits given 2, 4 and 10
  periods of 50 units, where the error is read off one cosine, the move passed three of its
  errors on 4.5 %, 6.0 % and 2.0 % of 200 linear logs each, and `t(1)`'s limit on none, 0.5 % and
  none.
- Under `rk4` with the quadratic push, the channel scattered 0.98 times its error and covered on
  39 of 40 logs.
- A spline fit cross-fits three sieves more. The check takes one `jax.jacrev` of the spline fit's
  channel through the cross-fit, and no second fit.
- The flip of the default to `"auto"` in 0.20.0 is the author's call. Its efficiency condition,
  no worse than about 1.2 times the polynomial where the polynomial is right, holds on the designs
  above.

## Alternatives

- **The spline without powers always.** Rejected: a quadratic nuisance is then a line past the
  outer knots. With a normal confounder the channel's bias was 2.0 and 2.4 of its errors at 1000
  and 4000 rows, covering on 47 % and 32 %; with a `t(5)` one, 4.4.
- **Compact B-spline bumps in place of the truncated powers.** Rejected: on the plant of ADR 0107
  they read a bias of −0.28.
- **Clamping each covariate to the fold's range before the polynomial.** Rejected: it biased the
  quadratic and the lognormal designs.
- **Two sieves, with and without powers.** Rejected: on a nuisance quadratic in a `t(5)` or
  `t(3)` confounder both miss, one in scatter (3.6 and 2.9 times the polynomial's), the other in
  bias (4.4 and 21 errors), and the choice cannot avoid both.
- **Hausman's difference of variances.** Rejected, as above: it needs the default fit to be the
  efficient one.
