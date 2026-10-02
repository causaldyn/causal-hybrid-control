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
  cover them, and the spread recovers the hierarchy's variance. **The gate is met in every arm.**
- No timing is quoted.

Measured by `scripts/bench_geo_dlm.py --replicates 100 --noise N --hierarchy H`, each arm its own
process: 100 worlds from seed 20261002, in double precision. G1 is the pooled error minus each
other way's, with the paired difference's 95 % half-width. G3's ratios are by channel: search,
social, video, tv.

| arm | G1, minus alone | G1, minus together | G2, integrated coverage | G3, spread coverage | G3, median ratio |
|---|---|---|---|---|---|
| lognormal, noise 30 | −0.025 ± 0.007 | −0.66 ± 0.05 | 0.892 | 0.8325 | 0.85, 0.71, 0.86, 0.88 |
| lognormal, noise 100 | −0.259 ± 0.031 | −0.37 ± 0.04 | 0.882 | 0.835 | 0.91, 0.72, 0.72, 0.93 |
| gaussian, noise 30 | −0.021 ± 0.007 | −0.68 ± 0.02 | 0.889 | 0.91 | 0.94, 1.03, 0.96, 1.04 |
| gaussian, noise 100 | −0.256 ± 0.029 | −0.38 ± 0.03 | 0.890 | 0.9225 | 0.93, 1.04, 0.96, 1.10 |

- **As predicted from the pilot.**
  - The integrated coverage is 0.882–0.892, against a predicted 0.88–0.89.
  - The pooled error is 8.9 % and 10.6 % below each geo's alone at a noise of 30, against 7–11 %.
  - The lognormal arms' spread coverage is 0.8325 and 0.835, against 0.82–0.89.
  - The median effective draws are 39.6–43.0 of 64, near the predicted 40–44.
- **Off the prediction.**
  - At a noise of 100 the pooled error is 33.4 % and 32.9 % below alone's, against 34–36 %. In
    the lognormal arm it is 4.2 % above the error at the world's own spread, against within 4 %.
  - The pooled intervals take the spread as known. At a noise of 100 they cover 0.818 and 0.846,
    against 0.83, and at 30 they cover 0.885, against 0.87–0.88.
  - The gaussian arms' spread coverage, 0.91 and 0.9225, and median ratios, 0.93–1.10, are above
    the predicted 0.82–0.89 and 0.86.
    - That prediction took the spread for the maximum-likelihood variance of 12 observed effects,
      `chi2_11 / 12`, whose median is 0.86.
    - The marginal likelihood integrates the national coefficients out under a vague prior, nearly
      restricted maximum likelihood (Harville 1974). Its reference is `chi2_11 / 11`, median 0.94.
  - In the lognormal arms the most kurtotic hierarchy, social's, has the lowest ratio, 0.71 and
    0.72, as predicted. Video's is as low at a noise of 100, 0.72, and search's is not low, 0.85
    and 0.91.
  - The least effective draws are 11.7 of 64, in the lognormal arm at a noise of 100; the pilot's
    least was 16.
- **The gate averages G3's coverage over the channels, as pre-registered.** In the lognormal arms
  the intervals of the two most kurtotic hierarchies cover less one by one: search's 0.74 and 0.78,
  social's 0.77 and 0.74. A variance read from 12 geos varies more the heavier the hierarchy's
  tails, and a normal hierarchy's likelihood does not see the tails.

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
