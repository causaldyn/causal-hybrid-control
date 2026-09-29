# ADR 0022 — A discount dynamic linear model

**Status:** proposed, 2026-09-29. Experimental.

## Context

A media-mix plan needs coefficients that move, and an interval on what a channel returned over a
period: a period's return is a sum of coefficient times spend over its weeks, so its interval
needs the coefficients' joint posterior over those weeks, not one week's. The adaptation work
asked for the same filter, to follow a channel that drifts.

The standard tool is West and Harrison's (1997) discount dynamic linear model. Its evolution
variance is not a parameter but a discount of the information the last step carried, its
observational variance is learned in closed form, and its one-step forecast is a Student `t` whose
log density, summed, is the criterion for choosing the discounts. Three choices decide what its
numbers mean:

- **How component discounts combine.** West and Harrison inflate each block's own covariance and
  leave the covariance between blocks as it was (additive). Inflating by `D P D` with
  `D = diag(delta_i^(-1/2))` (multiplicative) also inflates the covariance between blocks, and is
  the single discount `P / delta` when the discounts are equal. The additive form with equal
  discounts is not.
- **The units of a smoothed covariance.** With a learned variance the filter states `C_t` at the
  estimate `S_t` it had at `t`. The retrospective distribution carries the whole series' estimate,
  so a smoother that keeps each `C_t` in its own units misstates it by `S_T / S_t`.
- **The evolution variance past the data.** Holding the first step's `W` gives a state variance
  that grows linearly in the horizon; applying the discount again at every step grows it as
  `delta^(-k)`. At `delta = 0.95` the two are 3.854 times apart 52 steps ahead.

## Decision

- **`chc.dlm`**, experimental: `DynamicLinearModel(blocks, prior, form, variance_discount)`,
  `forward_filter`, `smooth`, `backward_sample`, `forecast` and `monitor_evalues`, written from
  West and Harrison (1997), Ameen and Harrison (1984), Kulhavý (1987), West and Harrison (1986) and
  McAlinn and West (2019, appendix A.2), not from another implementation's source.
- **NumPy in float64, not JAX.** The filter needs float64 (the Joseph update of a covariance
  stated at a small `S_t`), and JAX's is a process-wide flag the library does not set for its
  caller. Discounts are chosen on a grid by likelihood, not by a gradient, so differentiating
  through the filter buys nothing; at weekly `T` of 100–300 and a state of 10–40 coordinates a loop
  over steps is the natural shape; and the monitor it feeds, `chc.gate.DriftAlarm`, takes NumPy.
- **Blocks, each with its own discount:** `Polynomial` (a level, a level and a slope, higher
  orders), `Seasonal` (Fourier harmonics of a period that need not be an integer; the Nyquist
  harmonic takes one coordinate) and `Regression` (random-walk coefficients on columns of `x`).
- **`form="additive"` by default**, West and Harrison's; `"multiplicative"` on request, which with
  equal discounts is the single discount. No third form, and no separate type for the single
  discount: a `form` and a discount per block leave no combination illegal.
- **The prior's covariance is positive definite.** Every block's `G` is invertible, so every later
  prior and posterior covariance is too, and the smoother's solves never meet a singular matrix.
- **The observational variance is learned** (Normal–Gamma, `dof = inf` for a known one) and may
  move with a variance discount `beta`, which a known variance refuses. Missing observations are
  `NaN`: the step evolves and does not update, and the variance's degrees of freedom decay by
  `beta`. A step whose one-step squared scale is not positive and finite raises
  `FloatingPointError`.
- **`smooth` returns the exact moments of `backward_sample`'s draws.** Each step's covariance is
  in the units `E[V_t | D_T]`. With `beta = 1` that is `n_T S_T / (n_T − 2)` at every step. Below
  1 the backward precision is `phi_t = beta phi_(t+1) + gamma_t` with independent gammas, so
  `E[1 / phi_t | D_T] = int_0^inf E[exp(−u phi_t)] du` is one quadrature a step. Below 1 the
  marginals are scale mixtures of normals and not Student `t`; `SmoothedStates.exact` says so, and
  its `dof` is that of a gamma matched to the precision's mean and variance.
- **`forecast` holds `W_(T+1)` by default** (`evolution="constant"`); `"compounding"` discounts
  again at every step. Under the multiplicative form with discounts that differ, `D P D − P` is
  indefinite once the blocks are correlated enough — for two blocks exactly when
  `rho² > (d1² − 1)(d2² − 1) / (d1 d2 − 1)²`, `|rho| > 0.406` at discounts 0.99 and 0.8 — and
  holding it would make the state variance indefinite, so `"constant"` refuses such a `W`.
- **`Regression(hold_when_idle=True)`** discounts, at each step, only the coordinates whose
  regressor is non-zero at that step. Without it a long idle run is logged
  (`chc_event="dlm_windup"`); over 52 idle weeks the coefficient's variance grows 2.86 times at
  `delta = 0.98`, 14.4 at 0.95 and 240 at 0.90.
- **Interventions:** `forward_filter(interventions={step: delta})` divides that step's prior
  covariance by `delta`, West and Harrison's feed-back intervention, and logs it
  (`chc_event="dlm_intervention"`).
- **Monitoring by e-values.** `monitor_evalues` returns, per observed step, the density ratio of
  the standardised one-step error under a shifted or an inflated `t` to its density under the
  model, for `chc.gate.DriftAlarm`. West and Harrison's cumulative Bayes-factor monitor is e-CUSUM
  over the same ratios, and its usual threshold `tau = 0.135` guarantees only an average run length
  of at least `1 / tau = 7.4` steps; the alarm takes the one it guarantees as its threshold.
- **The module does not choose discounts.** It returns `log_likelihood`; the grid, the criterion
  and what to report are the caller's (see Consequences).

## Consequences

**Checked against other implementations**, each in a throwaway environment and never locked:

| oracle | what | largest gap |
|---|---|---|
| PyBATS 0.0.5 | the additive filter with a learned, discounted variance: means, covariances, `n`, `S` over 80 steps | 2.7e-16 relative |
| PyBATS 0.0.5, `adapt_discount="positive_regn"` | `hold_when_idle=True`, with PyBATS reading the regressor of the step before | 1.7e-16 relative |
| statsmodels `UnobservedComponents` | a discounted level at steady state as a fixed-`W` level, `W / V = (1 − delta)² / delta`, at `delta` = 0.7, 0.9, 0.97 | 2.8e-7 in the mean after 1000 steps |

PyBATS reads the regressor of the previous step to decide whether to discount: read on the
current step, the two differ by up to 0.4% in the mean on the same series. `chc.dlm` reads the
current step, since that step's prior is the one its observation updates.

**Checked in the repository:**

- `validation/discount_dlm.mac`: that one discount with `G = I` is recursive least squares with
  forgetting, weighted least squares with weights `delta^(T − t)` from a vague prior, with a Kish
  size `(1 + delta) / (1 − delta)`; the level's steady state; the two forecast policies and their
  ratio; the windup; when `D P D − P` is indefinite; the smoother's variance factor.
- `tests/test_dlm.py`: the conjugate regression's closed form when nothing is discounted,
  discount-weighted least squares, the steady state, the log-likelihood against `scipy`'s `t`, the
  degrees of freedom, missing steps, both forms, the windup and its guard, a seasonal cycle, a
  linear-growth forecast, both forecast policies, the constant policy's refusal, the smoother
  against a batch Gaussian posterior built independently (known variance, with lag-one
  covariances) and against the same scaled by `E[V | D_T]` (learned), with a smoother that keeps
  each `C_t` in its own units failing it by more than 5%, the sampler against the smoother at
  `beta = 1` and at 0.9, the monitor's density ratios, an intervention, and every refusal.

**The forecasts are calibrated; contribution intervals depend on how the discounts were chosen.**
`scripts/bench_dlm.py`, 500 series per world, 90% intervals, counts and coverage only (a
coverage's Monte Carlo standard error is 0.013 over 500 series):

| | coverage |
|---|---|
| own world, one step ahead (42 000 steps) | 0.897 |
| random-walk world, one step ahead, discounts picked (68 000 steps) | 0.902 |
| random-walk world, each channel's contribution over the last 13 weeks, discounts picked | 0.782, 0.770 |
| the same, averaged over the grid by likelihood | 0.830, 0.826 |
| the same, averaged, with the first 20 steps left out of the weights | 0.846, 0.828 |
| the same, the coefficients' discount fixed at 0.85 and the level's picked | 0.898, 0.890 |
| fixed at 0.9 | 0.898, 0.884 |
| fixed at 0.95 | 0.874, 0.846 |
| fixed at 0.98 | 0.660, 0.638 |

- **In the own world the forecasts are right.** Each observation is drawn from the model's own
  one-step forecast, so a miss is a bug. At 0.5, 0.8 and 0.95 the forecasts covered 0.494, 0.796
  and 0.947, and a Kolmogorov–Smirnov test of the probability integral transforms gave p = 0.17.
  The 0.947 is 3.1 standard errors below 0.95 on its own; three more seeds (`--seed 7`, `8` and
  `99`, `--own-only`) covered 0.949, 0.950 and 0.951 at 0.95 and 0.899, 0.902 and 0.899 at 0.90,
  with p from 0.14 to 0.66, and the four pooled cover 0.949 at 0.95, 1.3 standard errors low.
- **The picked discounts are too high for the coefficients.** The one-step likelihood picked the
  coefficients' discount at 0.95 or above in 450 of the 500 series and below 0.9 in none, a median
  2.6 above the same level's log-likelihood with the coefficients at 0.9. The 13-week intervals
  then covered 0.78 and 0.77 where 0.90 was asked, and at the discounts the likelihood passed over
  they cover. The discount that forecasts best one step ahead is not the one whose coefficients'
  intervals are calibrated, when the coefficients do not move the way a discount says they do.
- **Averaging over the grid recovers part**, to 0.83, and to 0.83–0.85 with the first 20 steps
  left out of the weights.
- **So a period's return is not reported from the likelihood's pick alone.** Which criterion or
  calibration carries the discounts' uncertainty is the response curves' decision (the ROI set),
  and experiments pin the coefficients directly.

No timing was measured.

## Alternatives considered

- **JAX with `lax.scan`**, as the plan first had it. Rejected for the reasons above; nothing
  downstream differentiates through the filter.
- **A dependency on PyBATS.** Rejected: the module needs the smoother's units, a forecast policy,
  the e-values and its own reading of an idle regressor, and the filter PyBATS would supply is the
  smallest part of it. PyBATS stays the oracle.
- **A sum type `Single(delta) | Component(...)`.** Replaced by a discount per block and a `form`;
  the single discount is equal discounts under the multiplicative form.
- **Other ways to combine component discounts, between the two forms.** Not offered: each is a
  different model, and the two named forms cover West and Harrison's and the single discount.
- **`evolution="compounding"` as the default.** Rejected: 3.854 times the constant policy's state
  variance 52 steps ahead at `delta = 0.95`, from a covariance no data has shrunk since `T`.
- **Discounting an idle coefficient on the step after its regressor was non-zero**, as PyBATS
  does. Rejected: the step whose observation updates the coefficient is the one whose prior should
  carry the forgetting.
- **A gamma for the smoothed precision with degrees of freedom `(1 − beta) n_t + beta n~_(t+1)`**,
  the first draft's. Rejected: it misstated `E[V_t | D_T]` by a median 13% at `beta = 0.9` and 36%
  at 0.8. A gamma matched to the precision's mean and variance was a median 0.5% off, but 38% at
  0.8 on a step where the variance jumped. The quadrature is exact.
- **Choosing discounts inside the module.** Not built: the bench shows that the obvious criterion
  gives intervals that under-cover, and the right one depends on what is reported.
