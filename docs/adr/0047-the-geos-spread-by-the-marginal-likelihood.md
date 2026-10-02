# ADR 0047 — The geos' spread by the marginal likelihood

**Status:** proposed, 2026-10-02. Experimental, as `chc.dlm` is.

## Context

ADR 0046's `GeoDLM` takes the spread of the geos about the national coefficients as given: it is
the regional prior's variance. That variance decides how much a geo borrows from the others. At a
vague prior each geo reads its own data alone; near zero every geo has the national coefficient.
The common hierarchical media-mix models put a prior on the spread and sample it with everything
else. The filter's log-likelihood at a spread is the exact marginal likelihood of the data, the
states integrated out, so one filter evaluates a spread's likelihood, and a few hundred filters
integrate over the spread.

## Decision

- **In `chc.dlm`**: `fit_geo_spread(model, y, x, pooled, draws, seed, *, level=0.9)`, returning a
  `GeoSpread(fit, x, pooled, variance, lower, upper, draws, weights)`, with `at(variance)`, the
  model at a spread, and `mixture(quantity)`.
- **A spread is a prior variance.** `pooled` names coordinates of a geo's regional blocks. Each
  one's prior variance is the spread on it: the same in every geo, and independent of every other
  coordinate (both are checked). It is stated at the prior's scale, as `Prior` states a
  covariance, so at a variance `V` the geos spread with variance `spread V / S_0`. The model's own
  is the vaguest spread considered and `1e-12` of it the least; below that the geos are pooled
  completely.
- **The best spread is type-II maximum likelihood** (empirical Bayes). L-BFGS-B searches the log of
  each spread within the range, from the model's own, on central differences of the filter's
  log-likelihood. Each spread's interval is the likelihood ratio's, the others held at their best:
  it ends where the log-likelihood falls `chi2_1(level) / 2` below the best, or at the range's end
  where it does not fall that far.
- **The posterior is under a prior flat on each standard deviation** over the range (Gelman 2006):
  `exp(log_spread / 2)` in the log. A prior flat on the log is improper toward 0, and wherever the
  likelihood is flat there it puts the posterior's mass against the range's floor.
- **Draws by importance sampling in the range's logistic coordinates**, `log_spread = low + (high -
  low) expit(u)`. The proposal is a Student `t` of 4 degrees of freedom about the posterior's mode
  in `u`, its scale 1.5 times the inverse of the posterior's curvature there, by central
  differences; the weights are self-normalised. Every draw is in the range, and the mode in `u` is
  inside it even where the best spread is at an end.
- **`mixture` is the law of total variance** over the draws, `sum_j w_j m_j` and
  `sum_j w_j (v_j + (m_j - mean)^2)`, one filter a draw.

## Consequences

- `tests/test_geo_spread.py` holds:
  - one spread the best of a 241-point grid of the log-likelihood, with the variance known and
    learned, and two spreads a local best;
  - each interval's ends where the likelihood ratio crosses its cut, to `5e-3`, and a spread of 0
    in the interval of geos that do not differ;
  - the spread of a Gaussian hierarchy over 40 geos recovered and covered, and the pooled
    coefficients nearer the truth than each geo's alone;
  - over four geos and twelve weeks, where one spread is weakly identified, the mixture of a geo's
    coefficient over 200 draws against quadrature of the spread's posterior on 300 points, within
    three of the draws' Monte Carlo standard errors, in three cases: the best inside the range,
    near its floor, and at the model's own, below the world's spread. `mixture` matches the draws'
    own moments to `1e-12`, and each case has more than 100 effective draws;
  - `at`, the log event, and the refusals.
- `validation/geo_spread.mac` derives Fisher's identity for the slope of the log-likelihood in one
  geo's spread `lam`, `-1 / (2 lam) + ((m - m0)^2 + C) / (2 lam^2 S)`, from the marginal of two
  observations; summed over the geos it is the slope when nothing is discounted.
- 18 of the 19 mutations of the fit tried fail a test. The 19th maps the draws from the range's
  other end: `u` and `-u` are the same coordinates mirrored, so it is the same sampler. A search
  started at the range's floor, where the likelihood is flat, stalls there, and the grid test
  kills it.
- A first round, on the first proposal and its tests, left six survivors. A doubled gradient is
  absorbed by L-BFGS-B's line search, and a start at the model's own spread rather than at the
  best of 13 is now the start. The tests missed four: a prior flat on the log, the proposal not
  divided out, the mixture without its means' spread, and draws outside the range kept. The
  quadrature test then fitted six geos over 40 weeks, where the posterior is narrow and the priors
  agree. Over four geos and twelve weeks it kills the first three, and the fourth has no range
  left to leave.
- A fit costs `2 k + 1` filters a gradient in each of its two searches, Brent's method's on each
  interval's end, `2 k^2 + 1` for the curvature, and one a draw; `mixture` costs one more a draw.
  On two of the bench's worlds, four spreads over 12 geos and 104 weeks with 64 draws, a fit ran
  323 and 360 filters.
- **What the first proposal got wrong.** A `t` about the likelihood's best in the log of the
  spread, with the likelihood's curvature there, dropped every draw outside the range. Where the
  best is at an end, the posterior sits against it in a sliver narrower than the proposal, and
  the weights collapse onto a few draws. Over four geos and twelve weeks with the best near the
  floor, 3 of 200 draws carried the weight. On a first pilot of the bench, whose prior put the
  model's own spread below the hierarchy's, three of four spreads sat at the ceiling, and 1.2–1.3
  of 64 draws carried it. In the logistic coordinates a sliver against an end is a mode inside:
  over the twelve weakly identified cases tried, at either end and inside, 146–190 of 200 draws.
- **The bench**, `scripts/bench_geo_dlm.py`, pre-registered its gate in its docstring before the
  scored run: pooling recovers the effects, the intervals that carry the spread's uncertainty
  cover them, and the spread recovers the hierarchy's variance.
- No timing is quoted.

## Alternatives

- **Fisher's identity for the gradient**, from one smoother pass rather than `2 k` filters. It
  holds only when nothing is discounted: a discounted block's evolution variance is a share of the
  filtered covariance, which moves with the spread. The central differences hold for any model.
- **A start from the best of 13 spreads moved together** along the range. On eight of the bench's
  worlds it found the best that the start at the model's own found, within `5e-4` of the
  log-likelihood either way, for 13 more filters. A start at the range's floor, where the
  likelihood is flat, stalls; the grid test catches it.
- **Kass and Steffey's (1989) correction** of the plug-in variance: first order in the spread's
  uncertainty, symmetric about the best, and blind to a best at an end.
- **Sampling the spread with the states** by Gibbs or Hamiltonian Monte Carlo. The filter
  integrates the states exactly, leaving `k` dimensions to integrate, and importance sampling
  does that with one filter a draw.
- **An inverse-gamma prior on the spread.** Where the data say little, its `epsilon` decides the
  posterior (Gelman 2006).
