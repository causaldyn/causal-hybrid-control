# ADR 0121 — The DLM discounts only what the data read

**Status:** accepted, 2026-10-21. Revises ADR 0022's choice of `form="additive"` as the default
and its "no third form"; ADR 0046's filter over geos takes the same default.

## Context

`chc.dlm` discounts the evolved covariance `P_t = G C_(t-1) G'` into the prior `R_t`. Up to
0.19.0 it offered West and Harrison's component discounting (`"additive"`, the default:
`R = P + blockdiag(P_ii (1 - delta_i) / delta_i)`) and `"multiplicative"` (`R = D P D`).

The additive form under-covers or over-covers without a warning wherever the data resolve a
sum of blocks better than the blocks. A constant regressor beside a level, prices held for 26
weeks and a trend leave their difference poorly resolved; the two blocks correlate negatively,
the blockwise inflation ignores that correlation, and the inflated variance of the difference
leaks into the one-step forecast. The learned observational variance is then read low. Over 200
series of 156 steps (a level at 0.95 beside a regressor at 0.9, `V = 0.09`), the 90% one-step
intervals covered 1.000, 0.999 and 0.994 with `S_T` at 0.04-0.11 of `V`. The leak is milder, and
the other way, beside a seasonal: 0.82-0.87. The 100x leak warning caught the first group and
none of the second. A direction the data never resolve (proportional columns, a constant beside
a level) also grows by `1 / delta` a step without bound. The filter then ran 1500 steps with
`Q_t` at 1e25, and its `FloatingPointError` guard fired only where the forecast's variance
happened to round below zero.

The multiplicative form covered 0.88-0.90 on the same grid. It has two defects of its own:

- `D P D - P` is indefinite once blocks with different discounts correlate. `smooth` and
  `backward_sample` then returned negative variances, and `forecast` refused 60-100% of
  level + season + regressor fits.
- It still inflates directions the data never read. A constant, or prices held 26 weeks, beside a
  level and a slope covered 0.002 and 0.58.

## Decision

- **`form="directional"`, the new default, discounts only the directions the data read.** The
  rows of `F` over the last `p` steps (the state's size), carried by `G^-1` into the current
  coordinates, read functionals `A theta`. A direction goes unread where the regression
  columns are combinations of each other and of the other blocks' entries over that window.
  The columns are scaled to a largest entry of 1. A direction is unread when its residual,
  after projecting out the span of the other entries, is below `1e-6`. The prior is then
  `R = P + sum_g r_g E_g`:
  - `P_r = P A' (A P A')^-1 A P` is the part of `P` the window reads;
  - the coordinates are grouped by their block's rate `r = (1 - delta) / delta`, slowest first;
  - `E_g` is the part of `P_r` that group `g` explains beyond the slower groups, a Schur
    complement step.
- **What the core guarantees** (`validation/directional_dlm.mac`; tests in
  `tests/test_formal_estimation.py`):
  - the parts add up to `P_r`, so `W = R - P` is a variance and a row's inflation is an average
    of the rates;
  - a block at `delta = 1` keeps its rows of `P`;
  - `theta` given `A theta` is as `P` had it, so an unread direction keeps its information;
  - with one discount it is Kulhavý and Kárný's directional forgetting,
    `R^-1 = P^-1 - (1 - delta) A' (A P A')^-1 A`, and `P / delta` once every direction is read.
- **The old forms stay**, by name: `form="additive"` and `form="multiplicative"`.
  `constrained_sample` still needs `form="additive"`, because its Gibbs blocks assume an
  evolution that keeps the blocks apart.
- **An unresolved direction is refused under every form.** A posterior whose correlation matrix
  has eigenvalues more than `1e13` apart raises `FloatingPointError` and names the widest
  direction. The check reads the correlations, not the covariance, so a prior tight on one
  coordinate is not refused.
- **A step whose `R - G C G'` is not a variance** is refused by `smooth` and `backward_sample`,
  as `forecast` already refused one.

## Consequences

- On the grid of 23 regressor-and-model worlds, 200 series each, the default's 90% intervals
  covered 0.891-0.902 on every world, with `S_T / V` at 0.88-1.01. The additive form covered
  0.824-1.000.
- At the discounts the one-step likelihood picks, over 40 series, the directional form covered
  0.895-0.905 on static worlds and 0.87-0.90 on moving ones. The additive form covered
  0.835-0.901.
- **A truth that moves in a direction the rows leave unread is not widened for.** A coefficient
  that drifts on a price held 26 steps covered 0.82 at fixed discounts. The additive form's leak
  covered that case at 0.999, by accident.
- **Cost.** A step costs more than an additive one: two singular value decompositions of the
  window, and one eigendecomposition per distinct rate. A model whose discounts are all 1 is
  not discounted and costs no more. Over geos the window spans as many steps as the national
  and one geo's regional coordinates. The directional form is dense across geos, as the
  additive form is.
- Fits that ran under the old default now give other numbers. Code that relied on component
  discounting must pass `form="additive"`. `constrained_sample` refuses a fit of the new default
  and names the form it needs.

## Alternatives

- **Multiplicative by default.** Rejected: it fails the smoother on correlated blocks with
  different discounts, and it still inflates unread directions (coverage 0.002 on a constant
  beside a level and a slope).
- **A calibration check, `mean(e^2 / Q)` near 1, with a warning.** Rejected as the fix: on the
  under-covering rows it read 1.25-1.57, too close to 1 to separate from noise in one series,
  and a warning leaves the intervals wrong.
- **Other cores on the read part.** Each was tried on the same grid:
  - the multiplicative core `D P_r D` is indefinite, as the multiplicative form is;
  - `diag(sqrt(r)) P_r diag(sqrt(r))`, and the multiplicative core clipped to its positive part,
    were variances but leaked beside a seasonal (coverage 0.855-0.878);
  - a rank-one directional forgetting on the newest row alone under-covered on seasonal models.
  Any per-coordinate inflation that is leak-free to first order is indefinite; the nested core
  is a variance and leak-free.
- **Projection instead of refusal for an unresolved direction under the old forms.** Not built:
  the directional form is that projection, and the old forms are kept as they were.
