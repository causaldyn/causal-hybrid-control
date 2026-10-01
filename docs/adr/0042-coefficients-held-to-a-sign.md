# ADR 0042 — Coefficients held to a sign

**Status:** proposed, 2026-10-02. Experimental, as `chc.dlm` is.

## Context

A media-mix model holds a channel's coefficient to a sign: spend on a channel is not expected to
lower sales. The usual way to impose it on a fitted model is to clip, either the coefficient's mean
or its draws, at 0. Clipping keeps the rest of the posterior where the unconstrained fit left it.
When the data pull a channel's coefficient below 0, the unconstrained fit has the level carry the
difference, and a clipped coefficient leaves the level as it was. The parts then no longer add up
to the fit, and the channel's covariance with everything else is ignored.

The posterior given the data and the sign is the unconstrained posterior truncated to the sign. In
the discount DLM the evolution variances `W_t`, in units of `V`, are fixed by the design and the
discounts and do not depend on `y` (ADR 0022). Given `V`, the model is therefore a Gaussian prior
over paths, and the truncated posterior is a Gaussian truncated to an orthant.

## Decision

- **`constrained_sample(fit, signs, draws, seed, *, chains=4, warmup=None) -> ConstrainedDraws`**
  in `chc.dlm`. `signs` maps a column of `x` to `"positive"` or `"negative"`. The draws hold every
  state at every step and `V`, by chain. `pooled` gives them as the `PosteriorDraws` that
  `decompose` reads, so a channel's contribution and the base come out of the truncated posterior
  with their intervals and add up draw by draw.
- **Gibbs over three blocks.**
  - *The constrained coefficients, with the other states integrated out.* Their values are a
    Gaussian given `V`, whose mean and covariance the smoother gives in units of `V`. One exact
    Hamiltonian trajectory of length `pi / 2` per sweep moves them (Pakman and Paninski 2014). A
    Gaussian's orbit is a known ellipse, `mean + a cos(t) + b sin(t)`. Where it meets a sign's wall
    the velocity reflects in the covariance's metric, so there is no step size and nothing is
    rejected. With no wall in the way the end is a fresh draw. The velocity is a centred backward
    draw of every state.
  - *The other states, given the constrained paths*, by forward filtering and backward sampling of
    a DLM with known offsets, whose covariances are computed once.
  - *`V`*, from its inverse gamma given the path.
- **Ties.** Steps that a zero evolution variance ties together share one value: a discount of 1, or
  a regressor held while idle. A static coefficient is then a single value.
- **Diagnostics.** `rhat` is the largest rank-normalised split R-hat, bulk or tail, over every
  state at every step and `V`. `ess` is the smallest bulk effective sample size, by Geyer's initial
  monotone sequence (Vehtari et al. 2021). Each run is logged (`chc_event="dlm_constrained_sample"`)
  with both and with the number of walls a trajectory met. The log is a warning when `rhat` is
  above 1.01 or `ess` below 100 a chain; the draws are not to be read then.
- **Scope, and what is refused:**
  - no signs;
  - a column that is not one of `x`'s, or that shares its regression block with another column;
  - a sign that is neither `"positive"` nor `"negative"`;
  - `form="multiplicative"`, whose evolution couples the blocks;
  - a variance discount below 1;
  - a fit with interventions, whose `R_t` is not the discount's;
  - `draws` below 1, `chains` below 2, a negative `warmup`.

## Consequences

`tests/test_dlm.py` checks the draws against oracles that share no code with the sampler. The
oracle path posterior is built from independent innovations (`theta = c + M e`), not by the filter.

- **A sign that does not bind** leaves the smoother's means and variances, and the variance's
  posterior mean `n_T S_T / (n_T - 2)`. No trajectory meets a wall.
- **A static coefficient** held to either sign is the truncated normal, and the level moves by its
  regression on the coefficient.
- **Three steps, a level and a drifting coefficient, with correlated priors:** the truncated mean
  is integrated by adaptive quadrature in two and three dimensions. Once with an idle held step,
  whose value is the last step's in every draw. In both, the truncation moves the means by more
  than 50 Monte Carlo standard errors, and the draws agree within 4.5.
- **Two channels, each held positive, over three steps:** six values, against rejection from two
  million draws of the batch posterior. Over five steps of two coefficients, one idle at a step,
  the values' dense covariance matches the batch posterior's to `1e-9`.
- **A channel whose effect is −0.5, held positive.** The unconstrained mean is below −0.4. The
  truncated posterior holds the coefficient's mean under 0.1, and the level lands within a quarter
  of the unconstrained level's distance from a fit without the channel. A clipped coefficient would
  have kept the unconstrained level.
- **The diagnostics:**
  - over 200 series, the bulk ESS averages AR(1)'s `m n (1 - rho) / (1 + rho)` to 3 %, and a
    moving average's value under the initial monotone sequence;
  - one chain of four shifted by half a deviation takes R-hat above 1.01, and so does one with
    1.6 times the others' spread, through the folded draws;
  - iid chains stay below it.
- **Mutations.** Seventeen mutations of the sampler and its diagnostics each fail a test. Among
  them:
  - a reflection that flips only the hit coordinate;
  - the entry root taken for the exit;
  - a start on a wall, heading out, not reflected;
  - the filtered means taken for the smoothed;
  - the values' covariance transposed across channels;
  - a tie mapped one step off;
  - the other states' first prior not moved by the constrained values.

  Five survived the first tests, each because no test could see it, and a test was added for
  each:
  - no test started on a wall;
  - one channel hides a transposition across channels;
  - diagonal priors hide the prior's shift;
  - an AR(1)'s autocorrelations fall monotonically anyway, which hides the monotone sequence (a
    moving average whose autocorrelation rises at lag 4 does not);
  - no chain differed only in spread, which hides the folded R-hat.
- **Cost.**
  - The constrained values' covariance is held dense, so memory grows as the square of their number:
    a coefficient that drifts over `T` steps has `T` values, and a static one has one.
  - A sweep costs a backward draw of every state and one trajectory. A trajectory meets more walls
    the harder the signs bind.
  - No timing is quoted.
- **It is exact only in the limit.** A finite run is read through `rhat` and `ess`.

## Alternatives

- **Single-site updates**, which were planned first: each value drawn from its truncated full
  conditional given its neighbours, odd and even steps alternating. They were built and did not
  mix on a drifting coefficient with a learned variance. When the evolution variance is small
  against the data's information, neighbouring values are strongly correlated, and so are the
  coefficient and the level it trades off with. Single-site updates then move a path's slow part
  by small steps (Roberts and Sahu 1997).
- **Truncating each backward-sampling step.** This is not the truncated posterior, since the filter
  that feeds it never saw the signs.
- **Rejection from the unconstrained posterior.** It accepts with the posterior probability of the
  signs, which is near 0 wherever a sign binds.
- **Elliptical slice sampling with the signs as the likelihood** (Murray, Adams and MacKay 2010).
  It uses the same ellipse but shrinks its bracket instead of reflecting. Near a wall that binds
  over many steps, the arc it may accept is short.
- **A reparametrised coefficient** (its logarithm, a softplus). This changes the prior, and so the
  model.
- **Clipping**, which leaves the parts not adding up and the base carrying the effect (Context).

## Not built

- **A sign on a coefficient that shares its block.** The trajectory takes any linear walls. The
  restriction keeps each constrained value's ties its own.
- **Constraints other than a sign**, such as bounds or an order between channels.
