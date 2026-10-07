# ADR 0057 — A ridge reads the same in any units

**Status:** accepted, 2026-10-07.

## Context

A ridge, a rank cutoff or a floor set in the caller's units weighs a column by the units it was
logged in. A column logged in units `s` times its own enters a Gram matrix at `s^2`, so one ridge
weighed it `s^-2` times as much. In small units the ridge took over, and nothing was raised. Each
of four kinds was somewhere in the library.

- **A ridge added as a constant** to the Gram of the raw columns: `fit_causal_residual`'s
  moment, drift regression and first stage; the double-ML, R-learner and network nuisances; the
  g-formula; the Koopman fit; the zones' slopes; the implied Gamma's logistic fit; the support's
  covariance; augmented synthetic control's outcome model. On the tests' data:
  - with the actions in millionths of their units, the channel read 0.0008 and -0.0004 where the
    fit reads 0.805 and -0.398; with the state there, the drift's slope read -0.0020 where -0.475;
    with a driver there, its gain read 0.0013 where 1.198; `solve_channel_moment`, on residuals
    there, read 0.0016 where 0.80;
  - with the covariates in thousandths, `estimate_effect_dml` read -0.176 where it reads 1.003;
  - with the treatments in millionths, `sequential_g_formula` read 0.0001 where 3.305;
  - with the incentive in millionths, every zone read a response of zero;
  - with a state at 1e-3 of its units, a point 4 standard deviations off the logged support scored
    0.016 where it scores 16.0;
  - with the outcomes at 1e-6 of their units, `augmented_synthetic_control` read plain synthetic
    control's 2.266 where it reads 1.966.
- **A ridge on the bias, or one scaled by a column's mean square**, which grows with the column's
  distance from zero. A unit has an origin as well as a size: a temperature in kelvin is one in
  degrees Celsius moved by 273.15. With the state ten thousand spreads from zero, the drift's
  slopes moved by 0.036 of the largest. A ridge solve at 0.1 on 50 rows kept 0.0006 of the slope of
  a column a thousand spreads from zero.
- **A rank cutoff relative to the largest column.** `jnp.linalg.lstsq` with `rcond=None` cuts at
  `eps * max(N, p)` of the largest singular value, 2.4e-4 of it in float32 at 2000 rows. With the
  action in millionths, `estimate_control_effect` read 0 where it reads 1.0015, and with the state
  at 1e-12 of its units, the split of the directions the moment has no data on (ADR 0054) moved
  the channel by 50.
- **A floor in the caller's units.** The nuisances took a column whose spread was under 1e-12 for
  a constant, so a confounder logged at 1e-13 of its units was not adjusted for and the channel
  moved by 1.27. `fit_behavior_policy` added 1e-8 to the actions' spread, `lalonde_ate` 1e-9 to
  each covariate's, the E-value 1e-12 to the outcome's and `refute_effect` 1e-9 to each tolerance;
  the pessimism penalty smoothed its norm at 1e-6, the plan's curvature test floored at 1 and the
  DLM prior's symmetry test at 1e-12.

**A column the log never moved.** A rule that reads each column in its own units must tell a
column that moves from one that does not. A column of one value does not deviate from its
computed mean by zero: it deviates by that mean's rounding. Centred once, the worst spread of an
exactly constant column was 3.63 eps of its value in float32 and 5.77 eps in float64, over 2 to
1e7 rows and eight values, and 4.13 eps in float32 at 2.5e20; beside a moving column in a
two-column design, 6.03 eps in float64.
NumPy sums a 2-D array's columns row by row, so the rounding of its column means grows with the
rows: centred once, a constant column read 119 eps at 1e3 rows and 8.5e5 eps at 1e7. The sites
tested for constancy in five ways: a spread above zero; a spread within 1e-12 of the column's
size; a corrected two-pass variance within 64 eps; a spread centred once within 64 eps; and
`max == min`.

## Decision

**One rule, in one private module.** `chc._units` holds the rule and its helpers, a NumPy and a
JAX version of each where both are used. It imports only NumPy and JAX, so no module of the library
imports in a cycle through it. NumPy sites stay NumPy: through `jnp`, a float64 array is cast to
float32 when x64 is off.

- **The rule.** A column whose spread about its mean, the population standard deviation, is at
  most `ROUNDING = 64` eps of its dtype times its root mean square is rounding: it moved by rounding
  alone. A column of zeros is rounding. Both sizes are taken relative to the column's largest
  entry, so no square under- or overflows. 64 eps is ten times the worst spread of a constant
  column centred once, 7.6e-6 of the column's size in float32 and 1.4e-14 in float64.
- **Centred twice.** The deviations' own mean is taken off them. In float32 a mean is rounded to
  about 1e-7 of its size, and NumPy's grows with the rows; centred twice, a constant column's
  spread was at most 2.8e-7 eps in JAX's float32, at 2.5e20 in 1000 rows, and 0 in float64, in
  JAX and in NumPy. A column whose entries are all equal reads deviations of exactly 0.
- **Zeroed.** At every centred ridge or least-squares site, a rounding column's centred values are
  0. Its coefficient is exactly 0, the intercept carries its level, and its ridge scale is 1, which
  keeps the solve regular and moves nothing else. In `fit_causal_residual`'s solve its row and
  column of the inverse are 0 too, so its coefficient has no variance, and the bias reads the
  variance of the solve without it.
- **Standardised.** Standardised, a rounding column reads 0 in every row.
- **The ridge's scale.** Beside a free bias, each other column's term is scaled by its variance;
  in a design with no bias, by its mean square, and by 1 for a column of zeros.
- **Least squares in each column's own units.** Each column is multiplied by the power of two
  nearest the reciprocal of its root mean square before `lstsq(rcond=None)`, and the coefficients
  by the same powers after. The cutoff then falls at one share of every column. `ldexp` makes each
  power exactly, so the scaling moves no bit, and a column near unit size reads as before, bit for
  bit.
- **`rk4`'s fixed point in each parameter's own units.** Its Newton matrix holds ratios of the
  parameters' units, so in the caller's units LU compared entries that are not comparable: with the
  state in millionths and the actions in millions, in float32, it pivoted on a drift row's
  rounding, and the channel's error read 30 times its own. The steps, the covariance and the carry
  are solved on the matrix scaled by the power of two nearest each parameter's weight on the log's
  rates.
- **Augmented synthetic control's ridge is a share of the donors' variance, 0.1 by default.** No
  one share suits every panel. On 200 panels of 30 donors and 25 periods before treatment, the root
  mean squared error of the effect at 0.1 was 0.0526 where the factors are white noise, against
  0.0502 at the best share, 1; 0.110 where they are random walks, against 0.105 at the best, 0.03;
  and 0.142 with those indexed to their first period, against 0.126 at the best, 0.01. A tenth
  errs by up to 1.133 times the best fixed share. 0.07 would have stayed within 1.1 times it on all
  three kinds, but the default was set before they were measured, and moving it to suit them would
  tune it on them. The earlier ridge of 1 in the outcomes' squared units erred less on these panels,
  0.0497, 0.107 and 0.113, by a coincidence of units: their noise is 0.1 in those units.
- **The panel estimator gate's test reads 300 draws.** The gate puts a variance ratio from paired
  draws against a prediction, and the new nuisances moved the two-cluster ratio. Under the tests'
  float64 it read 0.544 at 40 draws, 0.537 at 80 and 0.484 at 120, the test's old count, against a
  predicted 0.469, with an interval from 0.35 to 0.66 at 120: the conservatism the gate asserts
  failed. At 300 it reads 0.380, with an interval from 0.31 to 0.48, and the estimator is
  conservative in every cell at `phi = 0.9` on three streams of draws. Under float32, whose panels
  differ at the same seed, the conservatism held at 40, 80 and 120 draws as well. The test takes
  about 100 s longer.

The sites, and what each does now:

| Site | Columns | Now |
|---|---|---|
| `estimate_control_effect`, `estimate_effect_iv`, `sensitivity_analysis`, `refute_effect`, `BackdoorOLS`, `IV2SLS`, `local_projection_irf` and its callers | a design with an intercept | least squares in each column's own units; the E-value's spreads taken relative to their columns' largest entries; `refute_effect`'s tolerances a share of the effect |
| `estimate_effect_dml`, `dml_point_and_se`, `DoubleML`, `estimate_network_effects` | the covariates, then their monomials | standardised; a centred ridge with a free intercept, each term scaled by its column's variance, a rounding column zeroed |
| `RLearner` | the covariates; the R-loss design, which has no intercept | standardised; each term scaled by its column's mean square |
| `estimate_network_effects_gnn` | the final stage | least squares in each column's own units |
| `fit_causal_residual`'s nuisances | the adjustment set | standardised |
| its drift regression, first stage, drift response and their covariance | a design with a bias | centred twice, each term scaled by its column's variance, a rounding column zeroed with its row and column of the inverse |
| its moment, and `solve_channel_moment` | a design with no bias | each term scaled by its column's mean square, weighted as the moment weighs the rows; the fit's on the raw actions |
| its split of the unmoved directions (ADR 0054) | the drift's design | least squares in each column's own units |
| its logged relations, and `prescribe`'s ruled levers | the state | standardised, centred twice |
| its `rk4` fixed point | the parameters | Newton steps, covariance and carry in each parameter's power-of-two unit |
| `sequential_g_formula`, `naive_pooled_effect` | the treatments and confounders, in NumPy | centred twice, each term scaled by its column's variance, a rounding column zeroed |
| `benchmark_gamma` | the covariates, in NumPy | a rounding column zeroed, each term scaled by its column's variance; each Newton step centred on its weighted means |
| `ConfoundingRobustPenalty` | the actions | the norm smoothed at a millionth of the actions' root mean square |
| `KoopmanModel` | the lifted state and the actions, with no bias, in NumPy | each term scaled by its column's mean square |
| `calibrate_predictive`, `calibrate_naive_causal`, `calibrate_shared_state` | each zone's incentive and covariates | centred twice, each term scaled by its column's variance, a rounding column zeroed |
| `fit_behavior_policy` | the states; the actions | the states centred twice, a rounding state zeroed, least squares in each column's own units; the spread floored at 1e-8 of the actions' own |
| `augmented_synthetic_control` | the donors' periods before treatment, in NumPy | each period centred twice, a rounding period zeroed; the ridge a share of the donors' pooled variance |
| `SupportModel.fit` | the states and the actions | each column relative to its largest entry, centred twice; a rounding coordinate refused |
| `partial_corr_test`, `gcm_test` | the conditioning set and the tested columns, in NumPy | each column in its power-of-two units, a rounding column left as it is; the GCM's basis drops a rounding column, and a rounding tested column has nothing to test: its pairs read nan, its partial correlation 0 at p 1 |
| `lalonde_ate` | the covariates, in NumPy | standardised |
| `CausalPlan.decision_weight`'s curvature test, `Prior`'s symmetry test | eigenvalues; a covariance | tolerances relative to the largest eigenvalue and the largest entry |

## Consequences

- The fits read the same from 1e-15 to 1e15 of each column's units, and at any distance from
  zero to the rounding of the column's own values there. Tests hold each site, in float64 and in
  float32. In float32, with the state, the actions and the adjustment set in millionths and in
  millions, 300 of their units from zero, `fit_causal_residual`'s channel reads as at unit scale
  to 2e-4, three times what the rounding of the log's own values moves it by.
- Every fit moves at unit scale, by the ridge's share of each term and by the rounding the scaling
  changes. On the tests' data, measured against 0.14.x, the largest move of each recorded function,
  relative to the largest entry of the same output:

  | Function | Output | Largest move |
  |---|---|---|
  | `fit_causal_residual` | the channel and the drift | 1.7e-5 on a panel of one unit, 1.1e-8 in the fit's own tests; 6.3e-5 with the state's zero moved |
  | | their standard errors | 1.3e-5 on a panel of one unit; 6.3e-5 with the state's zero moved |
  | `solve_channel_moment` | the channel | 5.1e-10 |
  | `causal_plan` | the actions; `uncertainty_tube` | 4.8e-9; 2.7e-5. With the state's zero moved, 1.5e-6 and 6.3e-5 |
  | `causal_plan` under the support's penalty; `pessimistic_control` | the actions | 10 %; 1.3 % |
  | `SupportModel` | the precision | 44 %; 9.3 %; 1.2 % elsewhere |
  | `estimate_effect_dml`, `dml_point_and_se`, `DoubleML` | the effect; its error | 3.5e-4, the LaLonde estimate from 1.03066 to 1.03102 thousand dollars; 1.0e-4 |
  | `RLearner` | the effect; the CATE | 7.7e-6; 3.8e-6 |
  | `estimate_network_effects` | the effects; their errors | 2.5e-3; 13 % on a six-cluster panel, 3.9 % elsewhere |
  | `estimate_control_effect`, `estimate_effect_iv`, `sensitivity_analysis`, `refute_effect`, `BackdoorOLS`, `IV2SLS`, the GNN's final stage, `local_projection_irf` | every result | 1.4e-13; the E-value 1.0e-12 |
  | `sequential_g_formula`; `naive_pooled_effect` | the effect | 1.0e-5; 4.0e-11 |
  | `KoopmanModel` | `A` and `B`; the LQR gain | 2.3e-9; 6.1e-10 |
  | `calibrate_predictive`, `calibrate_naive_causal`, `calibrate_shared_state` | the responses; their errors | 4.9e-5; 7.5e-5 |
  | `sutva_allocation`; `pessimistic_equilibrium_allocation`; `interference_bias` | the allocation; the bias | 2.3e-5; 1.3e-4; 3.5e-6 |
  | `benchmark_gamma` | the implied Gamma | 7.5e-9 |
  | `ConfoundingRobustPenalty`; `ConfoundingRobustTask` | a closed loop's cost; the robust regret | 3.3e-5; 3.0 %, from 3.82 to 3.94 |
  | `fit_behavior_policy`; `off_policy_value` | every output | 5.6e-6; 1.3e-7 |
  | `augmented_synthetic_control` | the effect | 2.1 % |
  | `gcm_test` | the statistic | 2.4e-12; a lever the log never moved is left out, its pairs nan |
  | `partial_corr_test` | every output | 0; a column held at one value reads 0 at p 1 |
  | `lalonde_ate` | the effect, beyond its estimator's move | 6e-14 |
  | `panel_estimator_certificate` | the two-cluster ratio at 300 draws | from 0.324 to 0.357 |

  The largest moves are where the old ridge was a large share of what the log moved, or where the
  model changed. The support's precision moves by 44 % on a log whose spread, 0.05, made the old
  ridge 40 % of its variance, and by 9.3 % on one whose actions' spread, 0.1, made it 10 %. The
  network effects' errors move by 13 % where the old ridge was 2 % to 14 % of the confounders' Gram
  diagonal. On a log whose actions differ from what the covariates determine by a dither of a
  millionth, the moment's ridge outweighs the dither's own Gram about a thousand times and sets the
  fit: its scale there, the actions' mean square, is 0.88, so the channel and its error move by
  14 %, the drift by 25 % and its error by 21 %. Augmented synthetic control gained an intercept
  and a ridge in the donors' units, and `ConfoundingRobustTask`'s penalty smooths at a millionth of
  the actions' root mean square. `moment_norm`, the moment at the fitted channel, is the ridge's own
  footprint and moves with its scale, from 0.09 to 149 times its old value, to at most 3.5e-7.
  `gcm_test`'s p-values move by the sign draws whose signs all agree, which reproduce the statistic
  to its rounding: one draw of 500 with 12 clusters, three of 2000 with 11.

- Zeroing moves a fit only where a column is rounding, and there it reads the fit without the
  column. Kept with a ridge term of its own, as the first versions of these fixes kept it, a column
  constant but for 16 eps of its size read a slope of -2.2e-8 in the drift regression and moved
  the bias by as much, read -2.2e-14 in the double-ML nuisances, and moved augmented synthetic
  control's effect by 7e-12 of itself; beside the drift's bias, such a column, or one at 21.3 in
  every row, gave the bias a variance of 1e6, the inverse of the ridge, where the solve without it
  reads 0.02. Each reads exactly 0 now, and the rest of the fit is the fit without the column, bit
  for bit.
- **What this does not do.** Power-of-two scaling handles a column's units, not its level: in
  float32 a column far from zero against its spread still meets the rank cutoff, and its centred
  solve is what reads it. The GNN's nuisances train with Adam on the raw columns; only its final
  least squares reads in each column's units. With an original effect of exactly 0,
  `refute_effect`'s tolerance is 0. A ridge still shrinks a direction the log moved little next to
  the column's own size, as at unit scale: that is what it is for. `panel_estimator_certificate`'s
  default `draws` stayed 120 here, where under float64 at its default seed it reports `ok=False`;
  0.16.0 raises it to 300, where its table was measured.

## Alternatives considered

- **`rcond=0`, no rank cutoff.** Rejected: a column the design repeats then takes a coefficient
  set by rounding. With the confounder in the design twice, the second copy at twice the first, the
  two copies read -1.7e4 and 8.6e3 in float32 and 5.9e11 and -2.9e11 in float64, where the cutoff
  splits them 0.750 and 0.375, and the effect moved by 4.6e-4 and 1.4e-4.
- **Each column scaled by exactly the reciprocal of its root mean square.** Rejected: every result
  moves at unit scale, by up to 1.3e-6 in float32 and 4.2e-15 in float64 on the OLS family's
  design, where the power of two nearest it keeps every bit. A change of units by a factor that is
  not a power of two moves the power of two's results by their own rounding, 6.8e-7 in float32 and
  3.1e-15 in float64.
- **`exp2` for the power of two.** Rejected: on XLA's CPU backend it returns `2**k` exactly for 21
  of the integers from -120 to 120 in float64 and 33 in float32, and misses the rest by up to 42
  and 34 eps, so the scaling moved bits. `ldexp` is exact.
- **Absolute floors**, a spread or a tolerance in the caller's units. Rejected: no floor in the
  caller's units suits both small and large units. Earnings with a spread of 5,000 dollars, logged
  in units of 1e13 dollars, reached `lalonde_ate`'s estimator at a third of their spread; with the
  outcome in billionths of its units, a failing refutation passed.
- **A ridge scaled by its mean square for a rounding column**, kept beside the others, so that a
  column constant but for its rounding still has a term of its own. Rejected: its coefficient is
  then the rounding's correlation with the target over the ridge, not 0, and the bias takes it up,
  as the consequences measure. Zeroed, the column reads what the fit reads without it.
- **Scale every column to unit size, solve, and scale the coefficients back.** The same solve,
  written in other units; the penalty scaled by each column's second moment writes it in the
  caller's.
- **The raw Gram with the centred penalty.** The same solve in exact arithmetic. Rejected: its
  rounding is about the machine epsilon times the rows times each column's mean square, so a
  column whose spread was 1e-11 of its size read nan.
- **The mean square for every column, the bias penalised with the rest.** Rejected: it reads the
  same in any size of a column's units but not in any origin, and it shrank the slope of a column
  far from zero by the square of its offset over its spread.
- **One ridge relative to the Gram's trace.** Rejected: columns in different units would still
  weigh it differently.
- **The moment's penalty from the residuals' size in the fit.** Rejected: an action the covariates
  nearly determine leaves small residuals, so the ridge would let go of the directions it exists
  to steady. ADR 0054 scales by the raw actions for the same reason.
- **Refusing columns in small units.** Rejected: the units are the caller's, and no threshold
  tells small from wrong. A column the log moved by rounding alone is a different case: the
  support, which measures a distance in each column's spread, refuses it, since it has no spread.
