# ADR 0046 — A discount DLM over geos

**Status:** proposed, 2026-10-02. Experimental, as `chc.dlm` is.

## Context

`chc.dlm` (ADR 0022) filters one series. A media-mix panel has many geos, and they are not
independent: one country's geos share national media, and a channel moves the KPI in each of them
in much the same way, so a channel's coefficient in a geo is best read as the national coefficient
plus the geo's deviation. A model per geo throws away what the other geos say, and a geo with a
short or noisy history reads its own noise. The common geo models are static and fitted by
sampling. A discount DLM's filter is exact and in closed form, and its coefficients may move.

## Decision

- **In `chc.dlm`**: `GeoDLM(national, regional, geos, prior, form="additive",
  variance_discount=1.0, relative_variance=None)`, `stacked_prior(national, regional, geos)` and
  `forward_filter_geos(model, y, x)`, returning a `GeoDLMFit` in `DLMFit`'s notation, `y` of
  shape `(T, geos)` and `x` of shape `(T, geos, model.regressors)`.
- **One stacked state, in `chc.dlm`'s own terms.** The national blocks, then the regional blocks
  once for each geo, are a `DynamicLinearModel`, so the evolution, the discounts and the prior's
  checks are that module's and there is one implementation of them. A geo's row of `F` reads the
  national blocks and its own regional ones, and is zero on every other geo's.
- **The hierarchy is a sum.** A national and a regional regression block read each geo's columns
  from the first, so a geo's coefficient on a column both read is the national coefficient plus the
  geo's deviation. The regional prior's variance is the spread of the geos about the national
  coefficient; the regional discount says how long it is remembered. At a discount of 1 a deviation
  is a static random effect; below 1 the pull toward the national coefficient fades as the prior is
  forgotten. `stacked_prior` gives every geo the same regional prior, independent of the others'.
- **One variance scale, each geo's share known.** `nu_(g,t) ~ N(0, V w_g)` with `w_g` given, as a
  geo's population sets a count's noise. One scale keeps the update conjugate: the variance is
  learned in closed form, as `chc.dlm` learns it.
- **The update is the vector one** (West and Harrison 1997, section 16.4): one Cholesky factor of
  the observed geos' `Q_t` a step, the gain and `e' Q_t^-1 e` from it, `n_t = beta n_(t-1) + r_t`,
  the covariance in the Joseph form, and the step's score the observed geos' multivariate `t`
  density. A missing geo drops out of its step; a step with none evolves and does not update.
  `validation/geo_dlm.mac` shows the update is the one made a geo at a time and the density the
  product of theirs.
- **A regressor held while idle** (`Regression(hold_when_idle=True)`) is idle on a national
  coordinate when no geo's column on it moves, and on a geo's own when that geo's does not.
- **The filter is dense**, a step `O(p^3)` in the stacked state's `p` coordinates, and the fit
  holds `2 T` covariances of `p^2` entries.

## Consequences

- `tests/test_geo_dlm.py` holds:
  - one geo to `forward_filter`'s model of the national blocks and then the regional ones, every
    output to `1e-12`, in both forms, with a variance discount below 1, a known variance, a
    regressor held while idle and missing steps;
  - four geos with every discount 1, a season and missing observations: the state moves by `G`
    alone, so every observation is a row of one regression on the first state, and the filter's
    log-likelihood is that regression's multivariate `t` marginal, and its last mean, covariance,
    degrees of freedom and variance estimate its Normal-Gamma posterior, to `1e-10`;
  - three geos with no national block and a known variance, each following `forward_filter` of
    itself at its own variance;
  - a step's score against SciPy's multivariate `t` with a geo missing, a step with none observed,
    each geo's row of `F` with the national regression the wider and the narrower, a regressor held
    while idle live where any geo that reads it moves, `stacked_prior`, and the refusals.
- All 22 mutations of the filter tried fail a test. Two survived the first tests and each got one:
  a national regression reading the columns past a wider regional one, and the coordinates held
  while idle read off the first geo alone.
- The precision's pattern: an observation of one geo adds nothing between two others, and the
  multiplicative form keeps that zero through the evolution while component discounting does not
  (`validation/geo_dlm.mac`, step 3). A filter linear in the number of geos exists for the
  multiplicative form, by the arrowhead's block elimination, and is not built: the dense one is
  its oracle, and its speed is to be measured.
- There is a filter over geos and nothing after it: no smoother, sampler, forecast or choice of
  discounts over geos yet. A geo's own variance is not learned.
- No timing is quoted.

## Alternatives

- **A `chc.dlm` filter for each geo.** Nothing pooled.
- **A static hierarchical model fitted by sampling.** Its coefficients cannot move, and its
  posterior is the sampler's; the filter's is exact.
- **The geos one at a time within a step.** The same posterior and the same density, by step 1 and
  step 2, and a Cholesky factor per step reads the score off directly.
- **A variance for each geo.** Not conjugate: the scales would have to be sampled.
- **A module of its own.** A second implementation of the evolution and the discounts, or a second
  module importing the first's internals; the stacked model is already a `DynamicLinearModel`.
