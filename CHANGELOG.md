# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to adhere to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) once the API stabilises (pre-1.0 it may
still change).

## [Unreleased]

### Added

- **`Saturation.nonconvexity()`: the most a curve's concave envelope rises above it.** A number of
  the shape alone, 0 for a curve concave from zero: Hill's is 0.0674, 0.1547 and 0.2920 at slopes
  2, 3 and 5, roots of `4r³ + 12r² + 14r − 1`, `3r² + 6r − 1` (so `2/√3 − 1`) and
  `125r⁴ + 375r³ + 375r² − 35r − 32`, and rises to 1, a step's, as the slope grows; Kumaraswamy's
  at `b = 2` is Hill's at slope `2a − 1`. Where a curve turns concave it falls as the square of the
  shape's excess over 1: a curve that starts `αu − βu²` in `u = z^(1+ε)` has
  `(α²/β) a(1 − a) ε²`, with `a = −W(−2/e²)/2` the root of `a = e^(2a−2)` below 1, so Hill's is
  `0.1619 ε²`; Gompertz, which starts with a slope, has `ε³/(12(e − 1))` at `b = 1 + ε`. A plan
  on envelopes loses at most a channel's coefficient times it where it leaves the channel inside
  its chord. Derived in `validation/envelope_nonconvexity.mac` and checked at 80 digits by its
  PARI/GP counterpart; the anchors for thirteen families are Maxima's.

### Deprecated

- **`Prescription.evaluate` on windows of one unit; from 1.0 it raises `DecisionError` (ADR
  0049).** The bootstrap over units below has none to resample there, and `chc.decision` is
  stable, so the refusal waits for 1.0. Until then such windows are read as before, as
  independent, under the interval that treats them so, too narrow for them, with
  `PlanEvaluation.bootstrap` and `versus_logger` `None`. Each such call warns once with a
  `DeprecationWarning` and logs the same at `WARNING` (`chc_event="one_unit"`). Evaluate on a
  panel of several units instead. The test holds the reading of one zone's 1200 windows to the
  code before the change, to `1e-9`. Twenty mutations of the warning, its message, its log and its
  path caught.

### Fixed

- **`allocate` and `budget_for` left every equal S-shaped channel inside its chord.** Rates that
  jump at one price on the envelopes were mixed proportionally, so equal curves moved together:
  two Hill curves of slope 3 at a budget of 1.6 scales were split 0.8/0.8 for a worth of 0.677,
  both below their envelopes, a gap of 0.169 past the 0.155 one channel can cost; three at 2.4
  returned 1.016 where 1.267 was there. Such rates are now filled one at a time, which leaves at
  most one channel inside its chord: 1.26/0.34 for 0.705, and 1.264 for the three. With kernels
  of length one, floors at zero and caps past the tangencies, `bound - worth` is then at most the
  largest `periods * coefficient * nonconvexity` (Shapley and Folkman's lemma for one constraint).
  `allocate_geos` and `minimax_allocate` return a linear program's vertex and were not affected
  (1.26/0.34 on the same ties). `cvar_allocate` keeps its reference split wherever no split beats
  it on the envelopes, as it promises, so a reference inside the chords stays there: 0.8/0.8.

- **`Weibull` and `ChapmanRichards` put the tangency of a shape below `1 + 1e-4` at half its
  spend, and of most below `1 + 1e-8` at `nan`.** It is the positive root of `e^u − 1 = k u`, and
  it was read as the gap between Lambert W's two real branches, which both round to −1 there. The
  envelope's chord ended at the wrong point, or with `nan` was left out, and `nonconvexity()`
  raised; either way the envelope was off by no more than the curve's nonconvexity, 3.2e-9 of its
  ceiling at `1 + 1e-4`. The root is now found where `(e^u − 1 − u)/u`, summed term by term below
  1, meets `k − 1`, and from `k = 1 + 1e-15` to `1e4` it is within one ulp of PARI/GP's at 80
  digits.

- **`Prescription.evaluate` reads a panel as the target trial it emulates (ADR 0049).** Three
  parts of that trial were wrong (Hernán and Robins, *Causal Inference: What If*, chapter 22).
  - **Time zero.** Each unit's windows were cut back from its own latest period, so a unit that
    left the panel because of what happened to it had its windows chosen by their outcomes. They
    now start at the panel's periods `H`, `2H`, ... before its last, the same for every unit, and
    a unit gives each window it was observed throughout: `time_zero="calendar"`, the default;
    `"unit"` keeps the old windows. On a market where a zone leaves three periods after its supply
    first strays, over twelve panels of 4000 zones at `model_error=0`, the interval on the old
    windows missed the plan's value on them every time, by 4.6 to 8.1 of its standard errors, and
    the calendar's covered the plan's value on its own windows, and the plan's difference from the
    logger's, every time.
  - **The interval.** A unit's windows share its state, and the delta method over windows took
    them as independent. The intervals now come from `n_resamples=200` draws of the units with
    replacement, from `seed`, the whole evaluation run again on each: the linearisation, the
    logger's fit, the smoothing, the certificate and both values. `PlanEvaluation.bootstrap` counts
    the draws the certificate refused, which are left out and logged as a warning. Over 40
    replicates of 60 units of ten consecutive windows, the windows' own interval covered 28 and the
    bootstrap 37. Windows of one unit alone are read as before, deprecated (above).
  - **The comparator.** `PlanEvaluation.versus_logger` holds the logged policy's value on the same
    windows, the plan's value less it, and that difference's interval, read off the difference
    within each draw. The policy in place can cost less than the plan.
  - The tests hold the bootstrap to the cluster-robust sandwich where the nuisances are held fixed,
    and to `evaluate_plan` on each drawn panel, replayed from the seed. All 35 mutations tried
    fail a test. An evaluation runs the analysis `n_resamples + 1` times; the smoothing's grid is
    read in one recursion batched over its levels, and the linearisation is compiled, once a model.

## [0.11.0] — 2026-10-03

### Added

- **A split for the worst share of the readings, `chc.allocation.cvar_allocate`**
  (*experimental*, ADR 0048). A fit's draws are readings of the channels, many and alike, and over
  them the worst case is one draw. `cvar_allocate(readings, budget, periods, *, level, against,
  lower, upper, history=None)` returns a `CvarAllocation`: the split whose mean gain over the
  reference split `against`, the plan in place, in the worst `level` share of the readings is the
  most, the gain's conditional value at risk (Rockafellar and Uryasev 2000), with each reading's
  gain and a bound above the share's mean. The reference gains nothing under any reading, so the
  split never does worse in that share than keeping it, and at `level = 1` it is the mean return's
  split. Kelley's cutting planes close on it as `minimax_allocate`'s do, one excess a reading in
  the linear program. The tests hold it to closed forms on two linear readings, in any unit; on
  three channels with carryover under thirty readings, to every split of a 241-by-241 grid at three
  levels; to `allocate` where the readings are one; and, over random readings, levels and
  references, to a share's mean never below 0. Fourteen mutations caught. On thirty readings of
  three families each forty percent apart, the mean's split gains 38.4 on average and loses under
  16 of them; at each level tried from 0.1 to 0.8 the split is the reference.

### Changed

- **The licence is Apache-2.0**, with a `NOTICE` file; releases up to 0.10.0 stay under MIT. It
  permits what MIT permits and adds a patent licence from each contributor, which ends for anyone
  who sues claiming the work infringes a patent. A redistributor keeps the `NOTICE` and marks the
  files they changed, and a contribution comes under the same licence unless its author says
  otherwise.

### Fixed

- **`chc.response.Envelope` no longer compiles a program for each curve.**
  - *The defect:* the tangency was a static field, so the envelopes of two draws of one S-shaped
    curve had different tree structures. Every jitted function they entered then compiled once per
    draw and channel, `chc.allocation`'s value-and-slope among them.
  - *Its size:* planned over 400 draws of a posterior of three Hill channels, `cvar_allocate`
    compiled 1 542 programs. Each stayed in memory for the rest of the process, about 0.45 MiB
    apiece.
  - *The fix:* the tangency is now an array leaf. Envelopes share one structure, and a second
    posterior's draws compile nothing; two tests hold both.
  - *Behaviour change:* `Envelope.touch` is a 0-d array where it was a float.

### Notes

- **The marketplace study moves to 0.12.0.** The roadmap had it as 0.11.0. A planner for a
  posterior's draws, and the fix that keeps it from compiling a program per draw, ship first.
  Nothing in the study changes; ADRs 0017 and 0019 carry the new number.

## [0.10.0] — 2026-10-03

### Added

- **Response curves, `chc.response`** (*experimental*, ADR 0029): how a channel's return rises
  with its spend, for the plant, for curves taken as known from another package, and for the
  planner.
  - **One contract.** Each family is a standard shape of spend over its scale `K`, in currency, with
    every other parameter dimensionless, so a change of currency moves `K` alone. A `Saturation` is
    zero at zero spend, increasing, and rises to 1; the channel's coefficient carries the size.
  - **Seventeen bounded families.** Concave from zero: `MichaelisMenten`, `Exponential`, `Tanh`,
    `Arctan`, `Algebraic`, `HalfNormal`. S-shaped over part of their range: `Hill`, `Weibull`,
    `Logistic`, `Gompertz`, `Richards`, `ChapmanRichards`, `GammaCDF`, `LogNormalCDF`, `BurrXII`,
    `BetaCDF`, `Kumaraswamy`. Beside them `Logarithmic` and `Power`, unbounded baselines, and
    `Ricker`, the inverted U of ad fatigue.
  - **Where each bends.** `inflection()` and `tangency()` are in spend, 0 for a curve concave from
    zero. The tangency, where the tangent from the origin touches the curve, is a closed form for
    `Hill`, `Weibull` and `ChapmanRichards` and a root past the inflection otherwise
    (`validation/response_curves.mac`; the tests hold the roots to Maxima's to `1e-13`).
    `Envelope(curve)` is the concave envelope: on an S-curve `h'(0) = 0`, so zero spend is a
    stationary point a planner started there can stop at, and the envelope has no such point.
  - **A floor is not a shape**: Janoschek is `Weibull`, and ADBUDG and Morgan–Mercer–Flodin are
    `Hill`, each with a floor at the plant's base. PyMC-Marketing's `LogisticSaturation` is `Tanh`
    with `K = 2 / lam`.
  - **Every family's slope at zero spend is its own.** JAX's gamma and beta CDFs give `nan` there
    at shape 1; both have a slope of their own in spend. `BetaCDF` has none in its shapes, since
    JAX's `betainc` has none, and asking for one raises. Where a slope at zero is infinite, below
    shape 1 and the curvature below shape 2, it is read a machine epsilon of the scale off zero,
    so a zero weight on it is nothing, not `nan`: a planner reads the curve through one in every
    period a kernel without carryover leaves unreached, and a fit of the scale at every period of
    zero spend. Every slope finite at zero spend, the curvature at shape 1 included, is exact.
  - **`causal_plan` no longer starts alone at zero on an S-curve.** Given no warm start, on a model
    holding a curve that starts convex it plans the same problem on `relax(model)` first and starts
    from that plan, as a `RecedingHorizon`'s cold start does. `CausalPlan.relaxed_cost`
    (*experimental*) is the relaxed problem's task cost: where a larger response never costs more
    and that problem is convex, as a budget over curves of spend is, `task_cost - relaxed_cost`
    bounds the plan's distance from the best. On a one-step budget split between an S-curve and a
    Michaelis-Menten channel, the plan from zero spend now reaches the best split on each of eleven
    S-shaped families; a descent from zeros stopped at the greedy corner on the Hill, and reported
    convergence. A model with no such curve plans as before.
- **Carryover and the return per unit, in `chc.response`** (*experimental*, ADR 0030).
  - **Three kernels**: `GeometricAdstock`, `DelayedAdstock`, whose carryover peaks after the spend,
    and `WeibullAdstock`, the survival of each lag under a Weibull CDF. Each has a length of its
    own, so logging another period moves no earlier adstock, and `normalized` has no default; both
    are keyword-only. `Channel(kernel, curve, coefficient)` is a channel's return, adstock first.
  - **One definition of return, read four ways**, in the outcome's units unless `revenue_per_kpi`
    turns a KPI into revenue: `contribution`, the channel's return in a window's periods; and per
    currency unit, `roi`, the incremental return on a window's spend, carryover included;
    `marginal_roi`, its derivative along the window's spend; and `steady_state_marginal_roi`, the
    slope of a period's return once the adstock of a constant spend has settled.
  - **PyMC-Marketing, mapped.** `scripts/pymc_marketing_reference.py` runs PyMC-Marketing 1.2.0's
    kernels and saturation functions beside these in a throwaway environment: every one agrees to
    `2.2e-16` of its largest value but `hill_function`, to `2.3e-9`, because PyTensor makes the
    Python floats it is called with float32 constants. Its CDF Weibull kernel with `l_max` carries
    `l_max + 1` weights: `WeibullAdstock` of that length.
- **A spend budget for `prescribe`** (ADR 0031). `budgets=` takes `PeriodBudget`s: at most
  `amount` in each `period` steps from the plan's first, a step spending `weights @ u`, held as the
  rows `RecedingHorizon` holds at its first step. A period the horizon cuts short gets its share, so
  a budget for the whole horizon has `period=horizon`. `Prescription.budget_prices()` is what a
  unit more would lower the planned cost by, period by period, read from
  `CausalPlan.shadow_prices`; `report()` states it, or that a plan held under the constraints'
  barrier has none. `to_json` carries the budgets under `"budgets"`, as it carries the drivers,
  since the schedule depends on them, and `schema_version` stays 1; the prices stay out, as ID2's
  gate does, until the schema's one change at 1.0. A budget for another number of levers, one whose
  periods start anywhere but the plan's first step, or one below what the levers' boxes spend at
  the least raises `DecisionError` before anything is fitted: a prescription is made offline,
  where `RecedingHorizon`, which must act, spends the floor and warns. On `chc.mmm`'s plant with
  60% of the unbudgeted plan's spend, the budget binds and its price matches the planned cost's
  central difference in the amount within the test's `1e-3`. `tests/test_plan_oracles.py` holds a
  budget's price to a closed form within `1e-6`: goodwill with linear revenue, Sethi's (1977)
  setting (`validation/planner_oracles.mac`, STEP 5), where the price is what the step the budget
  leaves inside the box is worth, and a budget of whole steps is degenerate between the steps on
  either side. Without a budget nothing moves.
- **The discount DLM's monitor, measured** (`scripts/bench_dlm_monitor.py`). `monitor_evalues`
  read by `DriftAlarm` at an average run length of 1000, on a local level the filter tracks
  exactly, so the null holds by construction: over 300 series it alarmed by step 100 on 0.027
  (0.012-0.052), under its bound of 0.1, where West and Harrison's monitor at `tau = 0.135`
  alarmed on 0.997, within a median of 13 steps. The price is the window a change leaves: at
  `delta = 0.9` the filter absorbs a shift of two one-step deviations within a few steps, and
  `DriftAlarm` caught 74% of them within 1000 steps, where the textbook monitor caught every one
  at once; a shift of three it caught in all but one, with a median delay of 3. A known variance
  set too low reads as a change, 57% by step 100 with every deviation 1.25 times what it allowed
  for; learned, 4.3%. `chc.dlm`'s docstring states both. `tests/test_dlm.py` holds the e-values'
  mean at 1 on the model's own forecasts, where a one-step scale 10% too large moves it eight
  standard errors, and takes West and Harrison's feed-back intervention at the step an alarm on
  a break sounds.
- **An omitted-confounder bound for a fitted channel** (*experimental*, ADR 0032).
  `chc.dynamics_id.omitted_confounder_bound` bounds how far a confounder the adjustment set left
  out could move a linear functional of `fit_causal_residual`'s channel, one lever's effect or the
  value of moving spend between two, by the partial-R² bound of Chernozhukov, Cinelli, Newey,
  Sharma and Syrgkanis (2022), with confidence bounds and the shares at which each reaches a null.
  A fit made with `influence=True` keeps what it reads, the channel's Riesz representer with the
  nuisances held and the moment's residual (`CausalDynamicsFit.representer`, `.moment_residual`),
  under `rk4` through the fixed point. The bound is attained on a linear Gaussian plant with one
  latent, for any functional of any number of levers (`validation/omitted_confounder_bound.mac`),
  and the tests hold its side on the true channel under Euler and under `rk4` at `A dt = -0.7`,
  where the Euler reading of the same log misses by more than 0.2. Over 500 drawn worlds its
  confidence bounds covered the true reallocation in 0.940 (Clopper-Pearson 0.915-0.959) at a
  nominal 0.95, at half the shares in 0.216 (`scripts/bench_omitted_confounder.py`). DoubleML
  0.11.4, fed the same cross-fitted predictions, gives the same bounds to 2e-10
  (`scripts/doubleml_bound_reference.py`).
- **Where a reallocation flips** (`docs/concepts/gamma.md`). In variance shares a move of spend
  flips at `omitted_confounder_bound`'s robustness value on it. In `Γ`, for the analyst's CVaR gap,
  it flips where `barrier_gamma_star` puts a certificate's ceiling, with the value's distance from
  zero as the radius. Past the flip `minimax_action` holds under its cost criterion and still moves,
  hedged, under `criterion="regret"`. `tests/test_reallocation_flip.py` holds the three on a fitted
  two-lever channel, the `Γ` flip against a grid of `Γ`.
- **Lift tests read as structure, `chc.lift`** (*experimental*, ADR 0033). `fit_lift(tests,
  channel)` fits a `chc.response.Channel` to geo tests' gaps period by period, Heusch's (2026a)
  differencing equation, by least squares over every test's readout with one noise variance, and
  gives each parameter a profile-likelihood interval with an `F` cutoff (Bates and Watts 1988). A
  `LiftTest` is a treated and a control `GeoArm`, each its spend over the readout and the periods
  before it, its outcome, and its share of the market, which both are divided by. The coefficient
  is projected out, since the gap is linear in it, and the retention, scales and shapes are fitted
  in logs, so the two ridges a lift fit meets, a line and a step, are lines the solver follows; an
  interval that runs to a bound says so, and `LiftFit.tested_adstock` is the adstock the readouts
  covered, where the curve is identified. The tests hold the retention's interval to an
  independent NumPy profile within `1e-4`. On causaldyn-bench's Track M v2, Heusch's (2026b)
  generator written from his paper, with his four go-dark tests of paid shopping and noise of a
  percent of mean weekly sales in each group, as his, the retention's 95% interval covered the
  truth in 0.954 of 500 histories (Clopper-Pearson 0.932-0.971), the scale's in 0.976 and the
  coefficient's in 0.978, the scale's mostly open upward: his tests bend the curve little. At
  three times the noise the retention still covers, 0.958, but 15% of its intervals close, against
  81% at his; at ten times it covers 0.878, and 4 of the 500 fits raise rather than converge. On
  Meta at his noise it covers 0.918 (0.890-0.941), under the nominal, which the `F` cutoff
  promises exactly only for a model linear in its parameters.
- **An observational channel checked against lift tests, `chc.lift.check_observational`**
  (*experimental*, ADR 0039). Where spend follows the business an observational fit reads the
  business's decisions as the channel's effect, and the geo tests are what can say so.
  `check_observational(fit, observed)` reads a channel fitted to logged spend against the tests
  `fit` read, which `LiftFit.tests` now carries: an `F` test of the channel as a point of the fit's
  model, its least squares over the fit's per parameter over the fit's noise variance, and the
  factor of the lift it predicts that the tests read, `g·d / g·g`, with a `t` interval.
  `ObservationalCheck.least_gamma(cvar_gap)` turns the factor into the least marginal-sensitivity
  `Γ` whose identified set, `1 ± (Γ-1)/(Γ+1)·cvar_gap` in units of the predicted lift, reaches the
  tests: a floor under the confounding, as De Bartolomeis et al. (2024) bound `Γ` from a trial. The
  tests hold the statistic and the factor to NumPy, the fit's own channel to a factor of 1 and `k`
  times it to `1/k` with the closed-form excess, a channel of the right size and the wrong
  carryover to a rejection, and `least_gamma` to a grid of `Γ`; twelve mutations caught. On
  causaldyn-bench's Track M v2, pre-registered, over 500 of Heusch's histories on paid shopping,
  the `F` rejected the truth in 0.054 (Clopper-Pearson 0.036-0.078), the factor's interval
  covered 1 in 0.938 and the observational channel's noise-free factor in 0.939, and the `F`
  rejected the observational channel in 0.998; the tests read a median factor of 0.505 of its
  lift, a median least `Γ` of 2.75 at a gap of 1.
- **A budget allocated over channels, `chc.allocation`** (*experimental*, ADR 0034).
  `allocate(channels, budget, periods, *, lower, upper, history=None)` spends a budget over
  `periods` periods at one rate a period for each `chc.response.Channel`, each within its box, for
  the most return. The history's adstock runs into the plan and the plan's runs on for each
  kernel's length after it. Exact where every curve is concave, by bisection on the budget's price,
  with a linear channel's jump spent to the budget exactly; an S-shaped curve is planned on its
  envelope, and `Allocation.bound - Allocation.worth` bounds the plan's shortfall. The box has no
  default. The tests hold the plan against SciPy's SLSQP, random plans, the worth's own slopes, a
  closed form and a grid; causaldyn-bench's Track M v2 holds it against an oracle written without
  it.
- **A goal in place of a budget, `chc.allocation.budget_for`** (*experimental*, ADR 0037).
  `budget_for(channels, goal, periods, *, lower, upper, history=None)` finds the budget that meets
  `goal` and returns `allocate`'s plan for it: `ReturnTarget(amount)`, the least budget whose plan
  gains `amount`; `MarginalReturnTarget(per_unit)`, the budget at which one more currency unit
  returns `per_unit`; `ReturnOnSpendTarget(per_unit)`, a target return on ad spend, the most budget
  whose plan gains `per_unit` a unit it spends. The gain is `Allocation.gain`, the worth less
  `Allocation.idle`, what the channels return with nothing spent in the plan, so the history's
  carryover is not counted as the plan's return; `Allocation` also carries its `budget`. The plans
  for every budget lie on one path, and each goal is a point on it, found by `allocate`'s bisection
  on the price. A return on spend is sought past the budget where the price is the target, since a
  floor that holds spend on a poor channel can make the average rise before it, and the first
  budget that meets it is then not the most. Exact where every curve is concave; on an S-curve the
  path is the envelope's plans. The tests hold each goal to the closed forms of
  `validation/goal_seek.mac`, one Michaelis–Menten or exponential channel and a floor's two
  crossings at `325 ± 25√13`, and on channels with carryover to the gain the channels return. Nine
  mutations caught.
- **A split for several readings of the channels, `chc.allocation.minimax_allocate`**
  (*experimental*, ADR 0038). Tests that never bent a curve fit several families alike, and the
  families part where the plan goes. `minimax_allocate(readings, budget, periods, *, lower, upper,
  history=None)` takes each reading of every channel and returns the split whose worst regret over
  them is least, the regret under a reading being its best return at the budget over the split's.
  Where the curves are concave the worst regret is convex in the split, and Kelley's cutting planes
  close on it: the linear program over every reading's tangents bounds it from below, and the best
  split tried bounds it from above, so `MinimaxAllocation.worst - MinimaxAllocation.bound` is the
  certificate. An S-shaped curve is hedged on its envelope. The planes are read in units of the
  largest best return: a fit run to the edge of its family reads a slope in currency past `1e15`,
  the largest entry HiGHS accepts, and it refused the program. The tests hold it to a closed form,
  two mirrored linear readings hedged by the even split, in any unit, and on three channels with
  carryover to every split of a 241-by-241 grid; five mutations caught. On causaldyn-bench's
  Track M v2, pre-registered over 100 seeds under six curve families, it did not beat planning on
  the family AIC picks: worst over the families, 0.389 per euro [0.308, 0.469] against 0.370
  [0.292, 0.448], the difference +0.019 [-0.009, +0.055].
- **What a wrong channel costs the allocation, `chc.allocation.decision_weight`** (*experimental*,
  ADR 0040). `decision_weight(channels, budget, periods, *, lower, upper, history=None)` returns an
  `AllocationWeight`: the Hessian `W` of the loss of `allocate`'s plan made on channels whose
  parameters, each channel's kernel, curve and coefficient, are off by `d`, `d' W d / 2` to second
  order. `AllocationWeight.expected_regret(covariance)` is `tr(W S) / 2`, what a test that leaves
  the parameters with covariance `S` leaves the plan. The rates inside their boxes answer an error
  at one price, so `W = D' diag(h) D`, `D` each rate's response less the share that keeps the
  budget spent, `h` the worths' curvatures; a channel at an end of its box carries no weight, and an
  S-shaped curve is refused. `validation/allocation_decision_weight.mac` holds `W` to the realised
  loss's Hessian for two exponential and two Michaelis-Menten channels; on three channels with
  carryover the plan re-made on moved channels meets it to `1e-4`, and with every parameter off by
  a tenth the quadratic is within a tenth of the loss. Fourteen mutations caught. On
  causaldyn-bench's Track M v2, pre-registered over 500 geo panels, choosing where a geo test runs
  by the regret `W` expects it to leave cut the plan's regret to 0.0050 per euro [0.0037, 0.0062],
  against 0.0143 for Abadie and Zhao's representative set and 0.0241 for a random set.
- **One budget over geos and channels, `chc.allocation.allocate_geos`** (*experimental*, ADR 0043).
  `allocate_geos(cells, budget, periods, *, lower, upper, geo_totals=None, channel_totals=None,
  history=None)` plans every geo's channels at once, each cell with its own curve, carryover and
  history, the budget spent exactly and what each geo and each channel spends over the plan held
  within `Totals(least, most)`. It returns a `GeoAllocation`: the spend, the worth, a bound on it,
  and a price for the budget and for each total, what a unit more room in it returns, so a cell
  inside its box returns the budget's price plus its geo's and its channel's. Kelley's cutting
  planes on the cells' envelopes (HiGHS) certify the plan to a share `1e-9` of the bound; Newton's
  method on the binding totals' prices then makes it exact where the free cells are strictly
  concave, a total released where its price comes out on the wrong side and bound where the plan
  breaks it. A Newton step that does not bring the totals closer is replaced by the least of the
  dual, convex in the prices, along it. One geo is `allocate`'s plan to `1e-12`, and fixed geo
  budgets each geo's own; on three geos of three channels with carryover SciPy's SLSQP finds
  nothing better and every interior cell's slope meets its three prices; on two of two no plan of a
  400-step lattice returns more; a cap or floor a millionth from binding is planned exactly, and so
  is a budget `1e-9` to `1e-3` below the most a cap allows, where without the line search the
  cutting planes' plan came back. A plan made in two steps, the budget split over the geos by their
  average return first, leaves more than a percent of the joint plan's gain.
- **A goal over geos and channels, `chc.allocation.budget_for_geos`** (*experimental*, ADR 0044).
  `budget_for_geos(cells, goal, periods, *, lower, upper, geo_totals=None, channel_totals=None,
  history=None)` returns `allocate_geos`'s plan at the budget that meets one of `budget_for`'s
  goals, read on the same gain: a return target, a marginal return target or a target return on
  spend. The budgets run between the least and the most a plan within the boxes and the totals
  spends. On concave curves the best plan's gain is concave in the budget, its slope the budget's
  price, so a return target is met by Newton's method from the least budget and a return on spend
  from the most, in at most ten plans; a marginal target is one plan, its budget left free and each
  unit it spends charged the target. Under totals the gain can fall past some budget: a return
  target is refused only above the most any plan gains, and a negative marginal target is a budget
  past that peak. On S-shaped curves Brent's method runs on the true gain. One geo meets each goal
  at `budget_for`'s budget, spend and gain to `1e-12`; on three geos of three channels with
  carryover under a cap and a floor, a return target is met at the least budget and a return on
  spend at the most, a billionth less or more missing each; where two caps make the gain fall,
  each goal is met to its closed form. Of 48 mutations of the two functions 41 fail a test; the
  other seven change how fast a plan is reached or act at rounding, Newton's method started without
  the duals' binding totals among them (ADR 0043, ADR 0044).
- **A geo world drawn from a stated hierarchy, `chc.geo_world`** (*experimental*, ADR 0045).
  `GeoMediaMix(channels, geos=20, weeks=104, ...).draw(seed)` returns a `GeoMediaWorld`: each geo's
  KPI made by its own media, so a model of geos can be scored on the effects it recovers and a plan
  over geos on what it returns. A `MediaChannel` states a channel nationally: its saturation curve
  per head, the median over the geos of its effect, the KPI per head a week at the curve's ceiling,
  the log-SD of the geos' effects about it, its carryover, the log's mean spend per head and
  whether it is bought nationally. A geo's lift is its population times the curve per head of its
  spend per head through a normalised geometric adstock, times its effect, so geos of any size
  share a curve; populations and bases per head are log-normal about the mix's, national media is
  allotted by population, and the log's spend rises with the season by a policy, the confounding.
  Spillover into a geo's two neighbours on a ring and a national random walk in each channel's
  lift are options. Every variate is drawn whatever the parameters, so two mixes that differ in one
  compare paired. `GeoMediaWorld.cells(week)` returns each geo's channels as `chc.response`
  cells for `allocate_geos`, lifting what the world's media lift to `1e-12`; `lift(spend)` reads
  what any spend lifts, spillover included. The tests hold the geos' spreads to the stated ones
  within five standard errors over 4000 geos, and all 25 mutations of the module tried fail one.
- **A discount DLM over geos, `chc.dlm.forward_filter_geos`** (*experimental*, ADR 0046).
  `GeoDLM(national, regional, geos, prior, ...)` stacks blocks every geo reads and blocks each geo
  has its own copy of; a national and a regional regression block read each geo's columns from the
  first, so a geo's coefficient is the national one plus its deviation, the regional prior the
  spread of the geos and the regional discount how long it is remembered. `stacked_prior` builds
  the prior with every geo's alike. `forward_filter_geos(model, y, x)` filters a KPI a geo with one
  variance scale, each geo's share of it given (`relative_variance`): the evolution and discounts are
  `chc.dlm`'s for the stacked model, and the update West and Harrison's for a vector, the one made a
  geo at a time (`validation/geo_dlm.mac`), its score the observed geos' multivariate `t`. One geo
  is `forward_filter`'s to `1e-12`; with nothing discounted, four geos with a season and missing
  observations are the conjugate regression on every observation at once to `1e-10`. The filter is
  dense; the multiplicative form keeps the posterior precision an arrowhead, so a filter linear in
  the number of geos exists for it, not built. `smooth` and `backward_sample` take its fit and run
  back over the stacked state: over three geos with missing observations the smoother is the joint
  Gaussian posterior of every state to `1e-8`, the variance known or learned, and the draws have
  its moments. No forecast over geos yet. All 22 mutations of the filter and 8 of the smoothing
  and sampling path tried fail a test.
- **The geos' spread by the marginal likelihood, `chc.dlm.fit_geo_spread`** (*experimental*,
  ADR 0047). `fit_geo_spread(model, y, x, pooled, draws, seed, *, level=0.9)` chooses the prior
  variance of named regional coordinates, the same in every geo, by the filter's log-likelihood,
  which is the exact marginal likelihood (type-II maximum likelihood), with each spread's
  likelihood-ratio interval. It draws the spread from its posterior under a prior flat on each
  standard deviation, by importance sampling in the range's logistic coordinates, so that every
  draw is in the range, and `GeoSpread.mixture` mixes a quantity's posterior over the draws. Where
  one spread is weakly identified the mixture agrees with quadrature of its posterior within three
  Monte Carlo standard errors, with the best inside the range, near its floor and at its ceiling.
  `validation/geo_spread.mac` derives Fisher's identity for the slope, the alternative to central
  differences when nothing is discounted. 18 of 19 mutations tried fail a test; the 19th draws
  from the range's other end, the same sampler mirrored. `scripts/bench_geo_dlm.py` pre-registers
  the recovery and coverage of geos' effects and of their spread on `chc.geo_world`'s worlds. On
  its scored run, 100 worlds in each of four arms, the gate is met in every arm:
  - pooling puts each geo's error 9–11 % below the geo's alone at a noise of 30, and 33 % below it
    at 100;
  - the intervals mixed over the draws cover 0.88–0.89 of the effects;
  - the spread's intervals cover 0.83–0.92 of the hierarchies' variances, averaged over channels.
    In the log-normal hierarchy, search's and social's, the most kurtotic, cover only 0.74–0.78.
- **A futility stop for the gate's experiment** (*experimental*, ADR 0035).
  `GateConfig(alpha_futility=...)`, unset by default, runs a fourth e-process per zone, against
  "the candidate beats the baseline by `delta + min_effect`": the harm e-process at that margin,
  negated. When it reaches `1 / alpha_futility`, an experiment returns to shadow and the zone does
  not switch to EXPERIMENT again until its channel drifts. Each stop is wrong with probability at
  most `alpha_futility`; DEPLOY's false-discovery rate is untouched, since the switch is decided on
  the past and the improvement evidence keeps accruing in shadow. On the lab's closed loop at
  `alpha_futility = 0.1`, 400 paired replications (`scripts/bench_gate.py --futility 0.1`), the
  heavy-tailed null's experiment shrinks from 14 276–14 304 decisions to 1 816–1 937, a candidate
  at exactly `delta` saves 11–13 % of its own, and the FDR, the power and every deploy are those
  without the stop. Back in shadow, the heavy-tailed null was held for harm in 5 of 800 runs,
  against 1, within `alpha_harm`. A candidate exactly `delta + min_effect` better, read after each
  of 2000 decisions, was stopped on 1 of 100 paths.
- **Accelerator extras** (ADR 0036). `pip install "causal-hybrid-control[cuda13]"` installs chc
  with JAX's CUDA 13 build; `cpu`, `cuda12`, `cuda13-local`, `cuda12-local`, `rocm7-local`, `tpu`
  and `oneapi` name the others. Each is JAX's own extra of the same name, which pins `jaxlib` and
  the plugin to jax's version, and `oneapi` needs Python 3.12, since the jax the lock resolves on
  3.11 has no oneAPI build. The installation page installs through them; JAX's extra beside chc
  works as before. pip and uv install an extra that does not exist with a warning alone, so
  `tests/test_accelerator_extras.py` fails when one here names an extra the installed jax does not
  provide. Three mutations caught.
- **Python 3.15, and the free-threaded 3.14t and 3.15t** (ADR 0036). `requires-python` loses its
  cap below 3.15. CI's test matrix gains the three, its lint matrix 3.15. On 3.15 the `did` extra
  installs nothing, since diff-diff declares Python below 3.15, and `callaway_santanna_inference`
  says so when it raises; the `trees` extra's CatBoost has no wheel there. The free-threaded legs
  go without polars and diff-diff, which publish no free-threaded wheels, and the tests that need
  either skip.
- **`chc.dlm.decompose`**: the fitted mean split into its parts, draw by draw, from
  `backward_sample`'s draws. A polynomial or seasonal block is one part (the level, the seasonal
  effect) and a regression block one part per column, so each channel has its own; a draw's parts
  add up to its `F_t' θ_t` to rounding. A part's interval, a window's total or a return per unit
  of a column is then a quantile over the draws, taken after the sum, so it carries the
  dependence across steps that per-step intervals lose. `tests/test_dlm.py` holds each part's
  mean and variance to the smoother's exact ones; five mutations caught.
- **`chc.dlm.confidence_set`**: the fits a likelihood-ratio test at a level does not reject, those
  within `chi2_k(level) / 2` of the best log-likelihood, for an interval reported as the union of
  each member's (projection), which carries the discounts' uncertainty as well as the states'
  (ADR 0041). Pre-registered in `scripts/bench_dlm.py` and scored on 500 series: a channel's
  13-week contribution covered at least 0.88 of the time, met in both worlds. On the random walk
  the union covered 0.938 and 0.944 where the likelihood's pick covered 0.782 and 0.770, and the
  likelihood-weighted mixture 0.830 and 0.826. On series of the discount model itself it covered
  0.940 and 0.946 against the pick's 0.852 and 0.848, at 1.21 times the width at the true
  discounts. The rows ADR 0022 reports reproduced exactly.
- **`chc.dlm.constrained_sample`**: draws of the states and the variance with some columns'
  coefficients held to a sign at every step: the posterior truncated to the signs, not a clipped
  one (ADR 0042). A channel's negative pull then moves into the level, and `decompose` over the
  draws still adds up. Gibbs over three blocks:
  - the constrained values, with the other states integrated out, by one exact Hamiltonian
    trajectory a sweep (Pakman and Paninski 2014), reflected at the signs' walls;
  - the other states by forward filtering and backward sampling given them;
  - the variance from its inverse gamma given the constrained values, the other states integrated
    out as well: given every state, each evolution innovation pins it to the path.

  Steps a discount of 1 or an idle held regressor ties together are one value. Rank-normalised
  split R-hat and bulk ESS come with the draws (Vehtari et al. 2021); `ConstrainedDraws.mixed` says
  whether they can be read, and a run whose draws cannot is logged as a warning.
  `tests/test_dlm.py` holds the draws to oracles that share no code with the sampler:
  - the smoother where a sign does not bind, and there the variance's lag-1 autocorrelation,
    `k / (n_T + k - 2)` for `k` values;
  - the truncated normal for a static coefficient;
  - adaptive quadrature over three steps, with and without a tie;
  - rejection from the batch posterior for two channels;
  - and a channel whose effect is −0.5, held positive, whose level lands near a fit without the
    channel.

  Twenty-one mutations each fail a test. Single-site updates, planned first, did not mix on a
  drifting coefficient.

### Changed

- **Every Γ names its model.** CHC's `Γ` is the marginal sensitivity model's (Tan 2006), and
  Rosenbaum's, the one matched-design analyses report, is another model's. `Prescription.report()`
  prints `gamma* ... (marginal sensitivity model)`, `BarrierConstraint`'s refusal of `gamma < 1`
  says whose level it is, and so do the demos that print a `Γ`. The report also states the `gamma`
  the barrier's certified steps were counted at, which it left out, so the steps read as if no
  confounding had been priced; `DecisionCertificate.gamma`, the JSON's `certificate.gamma` and the
  certificate's log record carry it, `None` where no bound was audited. `docs/concepts/gamma.md` gives the
  bracket, Rosenbaum(Γ) ⊆ MSM(Γ) ⊆ Rosenbaum(Γ²) (Zhao, Small and Bhattacharya 2019), so a study
  insensitive to Rosenbaum's `Γ = 5` covers the MSM only up to `√5`; Rosenbaum and Silber's
  amplification; and the benchmark's two limits: it measures the treatment side alone, and nothing
  formal ties an unobserved confounder to an observed covariate. `tests/test_gamma_benchmark.py`
  holds the bracket on a binary confounder: fed the confounder, `benchmark_gamma` reports its MSM
  `Γ`, not Rosenbaum's, which lies between that and its square, and the `Γ` a known-null outcome
  needs rises from 1 with the confounder's pull on the outcome to that MSM `Γ` when the outcome is
  the confounder.
- **`just test` runs on the machine's accelerator.** It installs into `.venv` the extra the NVIDIA
  driver and GPU call for -- `cuda13` from driver 580 on a GPU of compute capability 7.5 or newer,
  `cuda12` from driver 525 -- and the suite runs on the GPU. `just sync` builds `.venv` the same
  way; `just test-cpu`, which `just check` runs, pins the CPU CI tests on. The `cuda` dependency
  group, `.venv-cuda` and `just test-gpu` are gone. `tests/conftest.py` turns JAX's preallocation
  off, so the workers share the card, and computes float32 products in float32 rather than
  TensorFloat-32.
- **`ty` checks each interpreter at its own version.** `[tool.ty.environment] python-version` was
  3.11, which had ty read the development environment's NumPy 2.5 stubs, written for 3.12 and
  later, on a branch never written for them: from ty 0.0.67 it rejected `.min()` on a float64
  array, eight times. It is now 3.14, `.python-version`'s, and CI's lint legs and
  `just types-matrix` pass their own; the 3.11 leg holds the floor.
- **Dependencies relocked at their latest**: jax 0.11.2 (0.10.2 on 3.11), NumPy 2.5.3 (2.4.6),
  SciPy 1.18.1 (1.17.1), ruff 0.16.9, ty 0.0.84.

### Notes

- **What the media-mix plans are worth against the tools, on Track M v2.** causaldyn-bench draws
  worlds from Heusch's (2026b) generator, which CHC did not write, has each arm read the same geo
  lift tests and plan a quarter's budget within a box, and scores each plan's regret per euro of
  the budget against the best plan in the box, carryover included. Every comparison was
  pre-registered before its scored run, paired over the same worlds, with a 95% Student-t
  interval; the arms' fits use each tool's own defaults and optimiser.
  - Against PyMC-Marketing, on 200 drawn worlds: a tie, CHC's regret less PyMC-Marketing's
    +0.0020 [-0.0110, +0.0150]. The gate set before the run, CHC's regret below it, is not met, so
    media mix is this release's example, not its theme. On 100 reference worlds PyMC-Marketing's
    regret is the lower, by 0.0163 [0.0032, 0.0295].
  - Against Meridian 2.1.0 (a national model with the tests as its ROI prior, planned by its
    `BudgetOptimizer`): CHC's regret is lower by 0.0697 [0.0540, 0.0855] on the 200 drawn worlds,
    on 153 of them, and by 0.0591 [0.0452, 0.0730] on the reference worlds. Meridian's plans stay
    near the status quo: 24 of its 600 channel-plans on a bound of the box, the best plan's 347.
  - Against Robyn 3.12.1 (the tests as its calibration, its own model selection, planned by
    `robyn_allocator`), on the first 100 drawn worlds: lower by 0.1815 [0.1343, 0.2287]. Robyn's
    own regret is above the status quo's there, by 0.0458 [0.0115, 0.0801].
  - Against the status quo, on the drawn worlds, lower by 0.1594 [0.1367, 0.1821]. Against the
    same channels planned with nothing carried over, lower by 0.0253 [0.0086, 0.0420]; on the
    reference worlds, whose best plan is a corner, that plan's regret is the lower, by 0.0247
    [0.0118, 0.0376].
  - CHC's reading of the tests left the scale interval open above on 554 of 600 channels, the
    tests not having read where the curve bends, so its plan reads the family's shape past the
    spend the tests moved.
- **Every module this release adds ships experimental, and `dlm` stays so.** The tier table has
  them wait on an external media-mix generator, and there the comparison with PyMC-Marketing was
  a tie.

## [0.9.0] — 2026-09-30

### Added

- **A check on what the logger read** (*experimental*, ADR 0028). `prescribe` asks the panel it
  fits, and `Prescription.evaluate` the panel it evaluates, whether the levers read anything but
  the states and their recorded parents: a column the graph says no lever causes, a driver, or the
  past. `Prescription.logger_check` and `PlanEvaluation.logger_check` hold the answer, a
  `LoggerCheck`, and `Prescription.report()` prints it. A rejection is logged as a warning
  (`chc_event="logger_check"`) and changes nothing else.
  - **`chc.independence.gcm_test`** is the test: Shah and Peters' generalised covariance measure
    over every pair at once, with in-sample regressions on the monomials of the conditioning set,
    sums within each period, and sign changes over the periods. A column the conditioning set
    determines is left out.
  - **Why the periods, not the units.** When the logger read the state and fresh noise, each
    period's sum is a martingale difference, however the state is autocorrelated, and noise the
    units share stays inside it. Over 1000 panels a case, it rejected 4.3-6.9% at 5% on one run
    of 4000 steps, on one or five geos over 104-260 weeks, and on 30-300 units over 10-20 periods,
    shared shocks included. Clusters by unit rejected 26.6% on five geos, and 51.0-88.2% with a
    shared shock; cross-fitting by unit 16.4% there, by time 13.8% on one geo; the GCM over rows
    19.6-85.2% with a shared shock (500 panels a case, `scripts/bench_logger_check.py`).
  - **What a pass means.** `detectable` is the partial correlation each pair would be caught at
    eight times in ten. The statistic normalises itself, so it is read through Student's t on the
    periods' sums: over 10-40 clusters it was caught 77-82% of the time there, where a normal
    reading gave a value caught 63-74% of the time.
  - **What a flag means: a premise failing, not a measured cost.** On the lifecycle test's market,
    400 panels a case, a logger that chased demand was flagged on every panel while the estimate's
    mean error stayed within 0.71 of its standard error and the interval still covered. One that
    kept half its last incentive was flagged too, and the interval at `model_error=0` covered
    91.2%.
  - **What a flag buys against a wrong graph** (causaldyn-bench's Track Q, nine worlds of 200
    panels). Its size there was 0.046 over 1600 panels planned on the true graph. Read as a stop,
    it removed silent failures beyond chance on one error of eight: a persistent input the graph
    omits, which the lever's own past carries. It does not see the same input drawn afresh, since it
    tests only the columns the graph names, a mediator the graph makes a parent of the lever, or a
    collider with a latent parent; and where it flags a harmless error it refuses sound plans.

- **`chc.dlm`: a discount dynamic linear model** (*experimental*, ADR 0022). West and Harrison's
  model, with blocks for a level or trend (`Polynomial`), Fourier seasonality over a period that
  need not be an integer (`Seasonal`) and random-walk coefficients on regressors (`Regression`),
  each with its own discount. `forward_filter` learns the observational variance (Normal–Gamma,
  with a variance discount), returns the one-step Student `t` forecasts and their log-likelihood,
  takes missing steps as `NaN`, and applies and logs a feed-back intervention at a named step.
  `smooth` and `backward_sample` give the retrospective states; `forecast` the `k`-step ones;
  `monitor_evalues` e-values of the one-step errors for `chc.gate.DriftAlarm`. NumPy in float64.
  - **Component discounts combine additively by default**, as West and Harrison's do: each block's
    own covariance is inflated and the covariance between blocks is left alone.
    `form="multiplicative"` inflates by `D P D`, which with equal discounts is the single discount.
  - **The smoother states each covariance in the units `E[V_t | D_T]`**, not the `S_t` the filter
    had at `t`: `n_T S_T / (n_T − 2)` with a fixed variance, one quadrature a step with a moving
    one. Its moments are those of `backward_sample`'s draws exactly, including the lag-one
    covariances a period's total needs.
  - **`forecast` holds the first step's evolution variance by default**; `"compounding"` applies
    the discount at every step, 3.854 times the state variance 52 steps ahead at `delta = 0.95`.
    Under the multiplicative form it refuses to hold a `W = D P D − P` that is not a variance.
  - **An idle regressor winds up.** Its coefficient's variance grows by `1 / delta` a step while
    the regressor is zero, 14.4 times over 52 weeks at 0.95, and a long run is logged
    (`chc_event="dlm_windup"`). `Regression(hold_when_idle=True)` discounts only the steps whose
    regressor is non-zero.
  - **Checked** against PyBATS 0.0.5 (the filter to 2.7e-16; `hold_when_idle` to 1.7e-16 once
    PyBATS's reading of the previous step's regressor is matched) and statsmodels' fixed-`W` level
    at steady state (2.8e-7), each in a throwaway environment; `validation/discount_dlm.mac`; and
    35 tests, among them the smoother against a batch Gaussian posterior built independently.
  - **What the bench found** (`scripts/bench_dlm.py`, 500 series per world, 90% intervals): the
    one-step forecasts cover 0.897 of data drawn from the model and 0.902 on random-walk
    coefficients. A channel's contribution over the last 13 weeks, with the discounts picked by
    one-step likelihood, covered 0.782 and 0.770: the likelihood picked the coefficients' discount
    at 0.95 or above in 450 of the 500 series, and fixed at 0.85 or 0.9 it covers 0.884–0.898.
    Averaging over the grid by likelihood covers 0.83. A period's return reported from the
    likelihood's pick alone is too narrow.

- **`fit_causal_residual(channel_degree=...)`: a channel whose feature degree is not the drift's
  (D29).** `None`, the default, keeps one basis for both, and the default fit is the same to the
  bit. `0` fits the constant channel §18/§19 cover beside a drift of any degree: `n m`
  coefficients where the affine channel has `n m (n + 1)`, and as many fewer tangents in the `rk4`
  fixed point. `ControlAffineResidual` carries it; `control_channel` and `closed_loop_jacobian`
  read it.
  - **Why.** An affine channel is as accurate as a constant one only where the log is centred. On
    eight logs of `ZoneMarketSystem`, against one RK4 period's response
    (`scripts/bench_channel_degree.py`), the constant channel was 3.6% (linear matching) and 4.6%
    (harmonic) off in relative RMS wherever it was read. The affine one was as close at the logs'
    mean state, 16% off at the do-nothing point, and 159% and 123% at `x = 0`, where
    `Prescription.reach` reads the channel's constant term.
  - **The default does not move.** `dynamics_id` is in the stable tier, and what a constant channel
    does to the tube, the regret bound and the marketing-mix case study has not been measured.

- **An installation page.** Every build JAX offers, installed beside chc: CPU, CUDA 13 and 12 from
  wheels or from a local toolkit, TPU, ROCm, Intel, conda-forge and nightlies, with how to check
  the device. And what changes for chc on an accelerator: float32 products in TensorFloat-32 on
  Ampere and later unless asked otherwise, a key's normal draws equal across devices only to
  rounding, the NumPy modules staying on the CPU, and JAX's preallocation of 75% of the card.

- **`AffineSchedule`: a plan evaluated from logs may change with the step (ADR 0023).**
  `AffineSchedule(gains, offsets, covariance)` is `u_t | x_t ~ N(gains[t] x_t + offsets[t],
  covariance)` for the `H` steps of `gains (H, m, n)`, and `AffineSchedule.open_loop(actions,
  states)` builds a prescription's kind, every gain zero. `certify_evaluation` and `evaluate_plan`
  take one for `"pdis"`, over the episodes' horizon, which is the schedule's by default.
  `"mis"`, `"dr"` and `"fqe"` refuse one by name: each evaluates a stationary loop's average cost,
  and a schedule ends.
  - **The weight's moment is carried forward.** The recursion carries the unnormalised law of the
    state along the starred loop, so one pass serves every horizon when the loop changes with the
    step; the backward one needed a loop that does not. A policy is evaluated as the schedule that
    holds it, and its certificate is unchanged: CE1 still escapes at `h = 6`, CE3's moment is
    still 8.134660321708, and the two-state loop still certifies `H* = 7, 15, 23, 31`. On 80
    random schedules the moment matches the whole-trajectory integral to a relative 1e-9.
  - **A target per state.** `"pdis"` scores each step against its own row of an `(H + 1, n)`
    `x_target` (ADR 0006); no step reads the last. The stationary methods still refuse one.
  - **Checked** by a ramped feedback schedule and an open-loop one against a moving target: over 60
    replicates of 3000 episodes each, the interval covers at least 0.85 of the time and the mean
    error is within three standard errors of zero; the smoothing correction is exactly
    `tau^2 sum_t tr(R + B' P_{t+1} B)` over the schedule's own gains.
  - **Not built:** a finite-horizon FQE, which would evaluate a schedule where every weight is
    refused.

- **`Prescription.evaluate(panel)`: a prescription is evaluated from a later panel before it is
  deployed (ADR 0023).** It hands `evaluate_plan`, by `"pdis"`: the windows of `H + 1` consecutive
  periods of each unit, cut back from its latest period; the plan's actions as an open-loop
  `AffineSchedule`; the plan's cost; and the plan's model over one step, linearised at the
  windows' mean state and action, with the covariance of its one-step residuals as the noise.
  - **It refuses** a plan whose levers were logged on a column outside its state, since importance
    weights need the logger's propensity given that column and no policy of the state is that, and
    a plan made against driver forecasts, whose plant changes with the step. What the levers were
    logged on is their parents in the graph, or the covariates an asserted adjustment names, not
    the adjustment set: the set also holds a covariate that only moves the target, and the logger
    never read that one.
  - **Checked** from the top level alone (`tests/test_lifecycle_loop.py`): on a linear-Gaussian
    market with a randomised incentive, a three-step plan made from one panel is certified from a
    later one, and its interval covers the plan's value on the true market in the test's draw; the
    value is `evaluate_plan`'s on the same windows to a relative 1e-9. A linear model's
    linearisation is its own RK4 step to 1e-14, and a nonlinear one's Jacobian matches finite
    differences at the windows' mean. Nine mutations of the bridge, all caught.
  - **Scope.** Windows cut from one unit follow each other, so they depend on one another through
    the state, which the interval does not see. The deployment gate is not connected: it takes
    decisions whose rewards do not depend on earlier ones, and a plan over a horizon on a plant with
    memory breaks that.

- **`chc.misspecification`: whether a plan pays for its model class being wrong** (*experimental*,
  ADR 0024). `misspecification_cost(plan, reference, alternative)` reads two fits of one class on
  one log, such as an unweighted fit and one weighted by the state, and prices their difference `d`
  in the plan's regret: `(d' W d − tr(W S)) / 2`, with a standard error. `W` is the regret's
  curvature in every fitted parameter at the plan, and `S` the difference's covariance, summed
  from the two fits' rows. When the class holds the truth, `d = 0`, and `p_value` is the chance of
  the quadratic reading at least as large, by Imhof's inversion of its chi-square mixture.
  `unseen` counts the directions the plan weighs that the log's actions never moved.
  - **The drift is priced with the channel.** The drift is fitted on what the channel leaves of
    the rate, so two fits whose channels differ have drifts that differ too.
    `CausalPlan.decision_weight` prices the channel alone: on logs whose actions averaged −1, 0
    and +1 it read 0.14, 0.44 and 2.8 of the regret between the two fits' plans. The quadratic in
    every parameter read 0.49, 0.54 and 0.49, and 0.74–0.78 and 0.89–0.91 at three tenths and a
    tenth of the difference (`scripts/bench_misspecification.py`).
  - **Calibrated over whole logs** (200 logs of 4000 rows). With the class holding the truth, the
    test rejected 1.0%, 6.0% and 11.5% at 1, 5 and 10%. With it missing, the cost fell within
    1.96 standard errors of its value on 400 000 rows in 0.915 of the logs. At 5% it caught 13%,
    30%, 78% and 97% of logs whose channel moved with the state at slopes 0.005, 0.01, 0.02 and
    0.03.
  - **On the zone plant** (0.9.0's verification). Where the class held, it rejected 0.5%, 3.25% and
    7.5% of 400 logs at 1, 5 and 10%. An incentive that recruits less the more drivers are idle
    moves the channel with the state. As that saturation went from 0.1 to 0.8, the power at 5% rose
    from 5% to 32%, and the regret between the two fits' plans tripled. Trips bent to the harmonic
    law left that regret, and the test's rate, where they were. Planning on the unweighted fit lost
    0.38–0.51 against the truth at every setting: the error both fits share, which the gate does
    not price.
  - **What it cannot see.** On a log whose second action was always twice the first, it read
    `p = 0.33` while the plan lost 0.55 against the truth, and `unseen` reported two directions
    the plan weighs that the log never moved. Held to the log's ratio, the plan had none and lost
    `5e-6`.
  - **Not in `prescribe`'s JSON**, whose schema changes once, at 1.0. It refuses fits that carry
    drivers, and whatever `decision_weight` refuses.

- **`fit_causal_residual(influence=True)`, and `CausalDynamicsFit.influence` and `.unmoved`.**
  `influence`, `(N, n, p)`, is each transition's share of the error in every fitted parameter when
  the whole log is drawn again: the channel's through the moment's residual, the drift's through
  the channel and through its own regression's residual. It is off by default, and the fit is the
  same either way. Under Euler its channel block gives `channel_error` to 1e-9. Under `rk4` it also
  carries the drift's residual into the channel, which `channel_error` leaves out.
  `unmoved`, `(p, r)`, is on every fit: the directions of the channel that the log's actions,
  less what the covariates' features predict of them, never moved, each with the drift
  regression's response to it.
  - **Why every fit carries it.** A log whose actions the covariates determine, as a deployed
    plan's own may be, has no data on the channel, and the ridge sets it. On such a log, in
    float64, the fit was identified and read the channel as `0.0009 ± 0.0013` against a true 0.8.
    `unmoved` is what says the log cannot tell.

- **`randomisation_test` and `randomisation_interval`: a switchback read off its own
  randomisation** (*experimental*, ADR 0025). `read_switchback`'s intervals are asymptotic in the
  run's length. `randomisation_test` tests the sharp null that the lever moves no reading by
  drawing the design's schedule again: every schedule of a block design up to 2^14 of them,
  otherwise `draws`. Its statistic is the reading's `estimate / se`. It is exact at any number of
  blocks and whatever the plant's memory, given that the data were read: the schedules the reading
  refuses are not part of the reference. `randomisation_interval` inverts it under the working
  model's joint null of persistence and channel, over a range of persistence the caller gives,
  since a null that names the effect alone is not sharp under carryover. Both join the top level,
  and so does `MarkovDesign`, which they take.
  - **Where Wald fails** (`scripts/bench_switchback_randomisation.py`, on the working model). At
    `a = 0.8` the block difference's Wald interval rejected a true zero in 17.2%, 11.9% and 7.2%
    of the runs over 6, 8 and 12 blocks; the test rejected 3.6%, 5.2% and 4.6%. Where Wald covered
    0.886 to 0.919 (8 blocks, a steady state over 60 periods, a memory of 0.95), the interval
    over a range holding the truth covered at least 0.986, and at the true memory 0.938 to 0.956.
    The price is width: at few blocks and a long memory the exact interval is mostly unbounded.
  - **The memory is part of the null.** Without it, the plug-in's interval covered 0.53 of
    `tau_3` and none of the steady states.
  - **Against pyfixest's `ritest`** on an i.i.d. lever: the statistic agreed with its HC1 t to
    4e-15, and the p-values within the permutations' noise.
  - **Scope.** The interval needs a first-order state per zone, and imputes each zone from its own
    lever; the test's null of no effect anywhere needs neither.
- **Inference for a staggered design, through diff-diff** (*experimental*, ADR 0026). The `did`
  extra is `diff-diff>=3.12,<4` (MIT), and the `dev` group holds it, so CI runs its tests.
  `chc.did.callaway_santanna_inference` takes `callaway_santanna`'s arguments and returns an
  `EventStudyInference`: the library's own estimates, with diff-diff's multiplier bootstrap around
  them, which draws whole units. It gives standard errors and a uniform band over the event study,
  and the overall ATT's standard error and interval. `EventStudyInference.robust_interval(m)` is
  Rambachan and Roth's smoothness restriction for the event study's average effect after
  treatment, from diff-diff's HonestDiD.
  - **diff-diff must reproduce the estimates.** Its cells, event study and overall ATT are checked
    against `callaway_santanna`'s to `1e-9` of the largest effect before its bootstrap is read, so
    a release that changed an estimand stops the call instead of moving the intervals.
  - **HonestDiD reads the analytic fit's covariance.** diff-diff's bootstrapped event study carries
    none, and HonestDiD then takes its effects as independent: in an exploration, `Delta^SD(0)`'s
    interval covered 0.68.
  - **What the bench found** (`scripts/bench_did_inference.py`: ten periods, three cohorts and
    never-treated units, AR(1) errors, 1000 panels a case, at 5%). The overall ATT's interval
    rejected a true null 5.7-6.6% of the time from 100 units, and 6.9-7.8% at 30. The band held
    from 300 units only, rejecting 6.4-7.2%; at 100 it rejected 7.9-8.7%, at 30 21-24%.
    `robust_interval(0)` covered 0.95 at 300 units, 0.92 at 100 and 0.78-0.79 at 30.
  - **Against R's did 2.5.1 and HonestDiD 0.2.8**, in a scratch library, on one panel of 300
    units: the cells, the event study and the overall ATT agreed to 6e-15, the analytic standard
    errors to 4e-15, and the smoothness interval to 1e-3. The relative-magnitudes restriction is
    not exposed: diff-diff 3.12 disables the paper's hybrid, and its interval was 0.55-0.72 of R's
    width on that panel.
  - **16 tests**, among them `robust_interval` against R's on a panel drawn from numpy's frozen
    `RandomState` stream.
- **Inference for a synthetic control** (*experimental*, ADR 0027). No extra.
  `chc.scm.synthetic_control_inference` takes `synthetic_control`'s arguments and returns a
  `SyntheticControlInference`: the estimate, Abadie, Diamond and Hainmueller's in-space placebo
  p-value, and the hull of the constant effects Chernozhukov, Wüthrich and Zhu's conformal test
  does not reject.
  - **The placebo is built as ADH built it**: every unit's synthetic control from all the others,
    so the test is exact when the treated unit was drawn at random. diff-diff, which carries both
    tests, leaves the treated unit out of every placebo's donor pool. On the same panels that
    construction rejected 0.5-1.5 points more under no effect, and the panels where only one of
    the two rejected ran 10 to 4 and 18 to 3 its way.
  - **Both tests are owned, over the exact weights.** diff-diff is the tests' oracle: the
    conformal p-value equals its `conformal_test`'s at 25 effects, and the interval ends where its
    test starts to reject. Its Frank-Wolfe needed 200 000 steps to converge where the weights here
    are exact.
  - **What the bench found** (`scripts/bench_scm_inference.py`: 20 periods before treatment and
    5 after, 1000 panels a case). With the treated unit drawn at random, the placebo rejected a
    true null 3.8-4.7% of the time at 5% with 19 donors, and 9.2-9.6% at 10%. The interval covered
    the effect 97.2-98.4% of the time at `alpha = 0.05` and 93.1-94.0% at 0.10, above the test's own
    levels on 25 periods, 4% and 8%. With the treated unit chosen as the one its donors fit best,
    the placebo rejected 27.8% at 5% and the interval covered 81.7% at 0.10: neither test holds for
    a design that picks its test market.
  - **11 tests**, among them the ranks' exactness with each unit treated in turn, which diff-diff's
    construction fails on the same panel. Nine mutations caught.

### Changed

- **The suite runs over worker processes, and on the GPU from an opt-in group.** `pytest-xdist`
  joins the `dev` group. `just test` runs the tests over four worker processes on the CPU, each
  file in one worker; `just test 0` runs them in one process, as CI does. A `cuda` dependency
  group holds JAX's CUDA 13 wheels, Linux only: the package does not depend on it and CI never
  installs it. `just test-gpu` runs the suite on the GPU from its own environment, `.venv-cuda`,
  since in `.venv` the plugin would load its CUDA libraries into every CPU process. How much time
  the workers save has not been measured on a quiet machine.

### Fixed

- **A replanning loop that holds a bound compiles once.** With `hold_constraints=True`, `prescribe`
  built each bound's margin as a closure, and the barrier made of them was a static field of the
  model the held solve compiles. The compile key therefore changed at every call, even with the
  same bounds, and every call compiled the barrier's check and the penalised descent again: two
  programs and about 4.5 MB a call, which the process kept. A week of half-hour calls on the
  BOPTEST building grew a process by about 1.5 GB. A margin is now a module whose bound is an
  array, so a bound that moves between calls is a value, not a program.
  - On a surrogate of that building, from the 12th call to the 144th, the loop compiled nothing
    and its resident memory rose 14 MB. Before, it compiled 264 programs and grew 600 MB.
  - All 144 schedules came out the same, bit for bit, in float64: 2304 actions, 305 of them
    inside the box.
  - A test counts the programs compiled by two calls after a first, and fails on the closures.
- **`synthetic_control` and `augmented_synthetic_control` reach the synthetic control's optimum.**
  Their donor weights came from 5000 steps of projected gradient, which stopped short of it. Over
  50 panels of each design, the pre-period error ended up to 8% above its minimum and the weights
  up to 0.21 from the optimum's. The overall effect moved by up to 0.022 on the tests' 30 donors
  and 25 periods, and by up to 0.13 with 80 donors and 30 periods. The weights now come from one
  non-negative least squares, exact to rounding: the synthetic control is the point of the donors'
  hull nearest the treated unit, and Lawson and Hanson's active set finds it in finitely many
  steps, at most `steps`. They no longer depend on the outcomes' units or origin. Where several
  weightings fit the pre-period equally well, the one returned can differ from before.

### Notes

- **What 0.8.0's robust `channel_error` costs, now timed on a quiet machine.** 0.8.0 shipped it
  untimed.
  - Under Euler, a fit takes 3.5–3.7 times as long once compiled: 0.046 s against 0.170 s on a
    4000-row log with degree-4 nuisances, and 0.056 s against 0.198 s at 16 000 rows. A process's
    first fit, compile included, takes 1.6 times as long (2.4 s against 3.8 s).
  - Under `rk4` the fixed point dominates, and a compiled fit did not move. A first fit got 12–14%
    faster, because the homoskedastic error it dropped compiled a path of its own.
  - The fit's changes since then (the constant channel beside an affine drift, and `unmoved` on
    every fit) add 7–9% under Euler and at most 2% under `rk4`.
  - How it was timed: the three versions alternated over six rounds, one process each, in float64
    on the CPU of one laptop.

## [0.8.0] — 2026-09-29

### Added

- **`chc.evaluation`: what deploying a plan would cost, from logs of another policy, certified
  before any cost is read.** `evaluate_plan(logs, plan, method, plant=..., cost=...)` estimates
  the plan's value with an interval. Four methods:
  - `"pdis"`: per-decision importance sampling over episodes;
  - `"mis"`: marginalised importance sampling on the stationary laws;
  - `"dr"`: MIS with the model's relative value as a control variate;
  - `"fqe"`: fitted Q evaluation with quadratic features.

  First, `certify_evaluation` reads the model and the two policies, never a cost, and gives the
  exact second moment of the weights the method needs, the effective samples it predicts, and the
  margin of each condition that keeps it finite. When the certificate refuses, `evaluate_plan`
  raises `InfeasibleEvaluation` naming the condition. Scope: every number is exact for a
  linear-Gaussian loop, `LinearGaussianPlant`, and an affine plan, `AffinePolicy`, and no more.
  - **The certificate** is the lab's, corrected by an independent verifier, whose loops are now
    tests:
    - The trajectory moment runs an exact recursion from the episodes' own initial law. A loop
      that passes the closed-form small-gain test and the one-step gate is infinite from `h = 6`
      when it starts from the logger's stationary law. The recursion matches an independent
      whole-trajectory integral to 1e-9 on 80 random loops.
    - The stationary condition `2 Sigma_b^z - Sigma_pi^z > 0` is split into a state and an action
      margin. The one-step gate is neither necessary nor sufficient for it, and the tests hold a
      loop for each direction.
  - **The value reported is the deployed plan's.** A deterministic plan has no density, so the
    weights evaluate it smoothed, and smoothing costs exactly `tau^2 tr(R + B' P B)` a step. The
    model's value of that is subtracted, and also carried in the interval, times `model_error`
    (default 1), because its error is first order in the model's. Every estimate reports the share
    of the weighted value the model removed. `tau` minimises the predicted mean squared error of
    the reported value among those the certificate passes, one rule for every weighted method.
  - **The loop `off_policy_value` misjudges.** It reads 2.92 for a deployed cost of 5.76, and its
    overlap flag says nothing is wrong. Here `"mis"` and `"dr"` refuse it by the state margin,
    `2 R_x - P_x = -1.34`. `"fqe"`, with a restricted chi-square of 3.30, returns an interval that
    covers 5.76 and not 2.92.
  - **Coverage**, 500 replicates on a two-sided market loop, with the model fitted to each
    replicate's logs (nominal 0.95, Monte Carlo SE 0.010). With the model's correction trusted:
    `"mis"` 0.940, `"dr"` 0.942, `"fqe"` 0.942 and `"pdis"` 0.968. At the default
    `model_error = 1`: `"mis"` 0.970, `"dr"` 0.972 and `"pdis"` 1.000. The default over-covers
    here because this model is right, and it counts the whole correction as unverified. The loop
    is CHC's own.
  - **The stationary interval's degrees of freedom are read off the batches the weights used.**
    It is Student's `t` over 40 batch means with Satterthwaite's `(sum_k s_k)^2 / sum_k s_k^2`,
    `s_k` each batch's sum of squares, and every stationary estimate reports it as
    `degrees_of_freedom`. On `MountainCarContinuous-v0` linearised at its valley floor, with a
    lightly damped logger and a plan whose states are ten times narrower than the logs', the
    weight sits in a few of the logger's passes through the floor, and the median degrees of
    freedom are 6 to 8. There a fixed `t_39` covered 0.916 (`"dr"`) and 0.890 (`"mis"`) of 500
    replicates, with the model fitted; Satterthwaite's covers 0.954 and 0.930. `"mis"` is still at
    the edge of two points (`scripts/bench_evaluation.py valley`).
  - `off_policy_value` is unchanged: it is the one-step estimator, and its docstring now points
    here. The design record is `docs/adr/0009-evaluating-a-plan.md`, and the coverage campaign is
    `scripts/bench_evaluation.py`.

- **`chc.gate`: a deployment gate that can be read at any time.**
  `DeploymentGate(plans, config).update(batches)` runs a candidate policy in shadow of the
  baseline, per zone, and after every batch returns `"deploy"`, `"shadow"`, `"experiment"`,
  `"hold"` or `"rollback"`. `mode(zone)` says how to log the zone's next batch.
  - **The guarantee holds at every read**, including one chosen by looking at the data. The
    expected share of wrong DEPLOY decisions across zones is at most `alpha`, and each HOLD for
    harm or ROLLBACK is wrong with probability at most `alpha_harm`. Each verdict is an e-process
    over the logged propensities, a mixture over constant bets. Across zones, e-BH runs over
    e-values that freeze when their zone stops gathering. A drift alarm, e-Shiryaev–Roberts over
    e-values the caller supplies, holds a zone and restarts its evidence, or rolls it back.
  - **Its conditions are in the docstring:** propensities logged at decision time, no spillover
    between zones, no carryover, and rewards in `[0, 1]`. Read after every decision, a z-test
    deploys a candidate that sits exactly at the margin on 28 of 100 paths. The gate deploys it on
    none.
  - **It takes propensities, not weights**: the candidate's, the baseline's, and the one logged
    when the action was drawn. It computes the weights itself, so none can arrive clipped the
    wrong way. It refuses a batch whose logged propensities are not those of the policy the zone's
    mode asked for.
  - **`DecisionLog` is what each decision records**, versioned: the action as applied, the
    propensity computed when it was drawn, a flag for a clipped action, and the Gaussian dither as
    drawn, which on a clipped decision is the draw rather than what the clip left of it.
    `DecisionLog.from_records` reads stored rows back, JSON lines or a table's, and refuses a record
    with no version, with another, or without a field, rather than filling one in.
    `ZoneBatch.from_log` takes the logged propensities from it, and `dither_draws` refuses a clipped
    decision. The stored form survives JSON bit for bit. The design record, with what a new version
    would require of stored logs, is `docs/adr/0015-what-a-logged-decision-records.md`.
  - **EXPERIMENT** is chosen when shadow cannot reach a verdict at `min_effect` within the horizon,
    and the mixture `(1 - rho) baseline + rho candidate`, which bounds both weights, would gather
    evidence faster. The speeds come from a two-point growth bound that is exact and minimax.
  - **The lab's eight-zone closed loop is reproduced verdict for verdict** over 400 replications
    (`scripts/bench_gate.py`). At `alpha = 0.10`, the FDR is 0.0075 when every zone is null and
    0.0055 when half are. Power is 1.000, and no deployment was rolled back. The design record is
    `docs/adr/0010-a-deployment-gate.md`.

- **`chc.gate`: a channel-drift monitor read off the logged dither** (*experimental*).
  `channel_drift_evalues(log, residual, dither_scale=, radius=, residual_scale=)` returns, per
  decision, e-values for "every entry of the one-step channel lies within `radius` of the model's".
  `DriftAlarm(arl)` turns them into an alarm, and `ZoneBatch(drift=...)` into the gate's HOLD or
  ROLLBACK.
  - **Each column is an exact e-value, whatever the model gets wrong in the drift, the noise law or
    the policy.** With `xi` the logged dither over its scale and `r` the residual moved to the
    radius's edge, `E[exp(theta r xi - theta^2 r^2 / 2) | c] = 1 / |1 - theta k|`. It is derived in
    `validation/dither_drift_evalue.mac`, its algebra is proved in
    `proofs/dither_drift_evalue.v`, and every column's mean is checked by Monte Carlo.
  - **Measured on the lab's plant** (`scripts/bench_drift.py`, 300 paths), whose model has the
    drift wrong and whose noise is Laplace. On an unchanged channel the alarm's run length is at
    least 3.23 times its target at `10^3` and 5.60 times at `10^4`, and 3.46 and 5.40 times with
    the drift model off by a further 0.15. Without the identification radius it falls to 0.96 and
    0.36. A channel at 1.4 instead of 1.07 is caught in 259 and 456 decisions on average. That is
    1.56 and 1.37 times an oracle that knows which entry moved, which way and at what rate.
  - **A plan whose actions clip is watched as it is.** A clipped decision is read with its draw,
    and the residual at the action as applied. Its e-value's mean is then
    `1 - (Phi(u2) - Phi(u1)) (1 - 1/s)`, `s = 1 - theta k`: at most 1 inside the radius, exactly 1
    on its edge whatever the clip, and at least 1 past it. The bound holds for any action applied as
    a nondecreasing function of its own draw, such as a saturating actuator. On the lab's plant
    boxed so that a quarter of the decisions clip, the alarm ran at least 2.84 and 4.61 times its
    target, and caught the move in 1.69 and 1.64 times the unboxed delay. Setting the clipped
    decisions' e-values to 0, also valid, caught it on none of 300 paths within 3000 decisions.
    `DecisionLog.dither` is now the draw, before any clip; version 1 has not been released. The
    design record is `docs/adr/0021-watching-a-plan-whose-actions-clip.md`.
  - **Refused:** a log with no dither, a dither whose draws a chi-square test at `1e-9` rejects at
    the stated scale, and a negative radius. `radius` has no default: at 0 the alarm came after
    0.36 of its target.
  - `DeploymentGate` now runs its drift alarm through `DriftAlarm`, and its verdicts are unchanged.
    The design record is `docs/adr/0018-a-channel-drift-monitor.md`.

- **`CausalPlan.decision_weight()`: what an error in the one-step channel costs a plan**
  (*experimental*). If the plant's one-step map is the model's plus `E u`, the plan loses
  `vec(E)' W vec(E) / 2` to the plan that knew `E`, up to a cubic term. `W = J_Eu M^-1 J_uE` over
  the directions the plan may move: the actions off the box, along the binding rows.
  `DecisionWeight.regret(E)` reads it. The entries are the ones `channel_drift_evalues` watches, in
  its order, so a radius or a detected move can be priced in the plan's own cost. Until now `W`
  existed only for `chc.experiment`'s one-shot decision.
  - **Checked three ways.** `validation/plan_decision_weight.mac` derives `W` through the RK4 step,
    on one and two steps and with an action on its bound, against the regret's exact second
    derivative. `proofs/plan_decision_weight.v` proves the envelope identity, and that pinning an
    action or binding a row can only lower `W` (9 lemmas). On a two-state, two-lever plan with a
    binding box and budget, the regret re-solved by SLSQP meets `W`'s prediction with a gap that
    shrinks linearly in `E`, the cubic term's.
  - **It says how far it reaches.** On the validation file's scalar plan the quadratic is within
    10% of the regret only for errors up to 8% of the channel. `weakly_active` counts the bounds
    and rows met with a zero multiplier, where the regret is piecewise and `W` its lower branch, and
    `residual` the stationarity miss of an unfinished solve.
  - **Refused:** a plan held under a barrier, whose condition moves with the channel; one solved
    with pessimism penalties, whose regret has a first-order term; one built by hand; and one whose
    Hessian along its free directions is not positive definite.

- **`channel_move`: how far a deployed plan's channel has moved, read off the logged dither, and
  what that costs the plan** (*experimental*). `channel_move(log, residual, dither_scale=,
  forgetting=1.0)` returns a `ChannelMove`: the plant's one-step channel less the model's, its
  covariance and its effective size. `ChannelMove.price(weight)` reads the move through
  `CausalPlan.decision_weight()` as a `MovePrice` with two parts. The first is what keeping the
  plan loses, estimated without bias, with a standard error. The second is what a plan re-solved on
  the estimate loses on average. Until now the drift monitor could say that the channel had moved,
  and nothing could say by how much.
  - **The estimate is unbiased whatever the model gets wrong in the drift, the noise law or the
    policy.** Each decision's residual times its standardised dither reads the move, and everything
    else adds to the spread, not the mean. The weights forget older decisions at a constant rate,
    so an estimate re-read as decisions arrive follows a channel that moves. The covariance is
    scaled to be exactly unbiased at any weights. The usual `kish / (kish - 1)` read 13% low at
    eight decisions forgotten at 0.7.
  - **Measured on the drift monitor's plant** (`scripts/bench_channel_move.py`, 2000 paths). The
    estimate lay within 0.75 standard errors of a move of 0.07, and its spread was 0.99-1.03 of its
    standard error. Its 95% intervals covered 0.943-0.955 of the paths, and 0.951-0.960 of them
    after the channel moved partway through the log.
  - **Two expectations, not a decision rule.** Re-planning when a log's keep price beat its re-plan
    price lost to the better of always keeping and always re-planning at 30, 100 and 400 decisions
    on a budgeted plan. The logs whose estimate reads a large move are the ones it errs most on.
    Shrinking the estimate, or choosing on half the log, did no better. So no rule ships. Re-read
    the move on the decisions logged after the choice, such as those after a `DriftAlarm`, and the
    re-plan price is the re-solved plan's.
  - **Checked:**
    - `validation/dither_channel_move.mac` derives the identities, the forgetting's worth, the
      prices, their variance and the covariance's scale;
    - `proofs/dither_channel_move.v` proves their algebra (7 lemmas);
    - 24 tests;
    - 24 mutations of the estimator and the prices, all caught.

    The dither check is now one helper shared with `channel_drift_evalues`, and its messages and
    verdicts are unchanged. The design record is `docs/adr/0019-re-reading-a-moved-channel.md`.

- **`chc.experiment`: which experiment a decision needs, how large, and whether to run one.**
  `design_experiment(decision, prior, experiment, budget)` returns an `ExperimentDesign` with:
  - whole units per zone and what they spend;
  - the regret of acting now and after the experiment;
  - the value net of spend, with its Monte Carlo error;
  - a regret bound at `level`;
  - a verdict of `"deploy"`, `"experiment"` or `"abstain"`.

  Scope: a one-shot `ZoneDecision` over `K` zones, linear in each zone's channel, with a quadratic
  cost and a box on the levers. The channels enter as a Gaussian `ChannelPrior`, and the
  experiment as a `ZoneExperiment` of unit costs, noise and probe sizes.
  - **Units go where a wrong estimate costs the decision, not where power is lowest.** The
    decision weight `W = J_bu M⁻¹ J_ub` is the Hessian of the regret in the channel; it is
    averaged over the prior's sigma points. The allocation water-fills the diagonal of `W` at the
    price of information, minimises `uses · regret + spend` under the budget, then rounds down
    without overspending.
  - **A lever pinned at its bound is priced by its Gaussian tail**, so a zone the optimiser never
    moves can still be worth testing. Each design says how far each pinned lever is from
    activation.
  - **Monte Carlo reports; the local model only allocates.** Every regret is measured by drawing
    channels from the prior, simulating the experiment, updating, and re-solving. `model_gap` is
    the local model's distance from that, logged over 25% (`chc_event="design_gap"`). It reaches
    0.52 on a prior half as wide as the channel, while the realised regret stays at 0.96–1.01 of
    the reported.
  - **Measured** by `scripts/bench_experiment.py`, which realises each experiment unit by unit by
    least squares, a path to the posterior independent of the design's own:
    - on the lab's market at one budget, decision-weighted units leave `1.18e-3` of regret, against
      `1.84e-3` for classical Neyman, `1.53e-3` for equal units and `1.61e-3` for equal spend;
    - at the design's own spend on a prior 10% wide, `0.0905 ± 0.0028`, against `0.110–0.120` for
      the same three rules;
    - the realised regret, now and after, is 0.96–1.03 of the predicted on three scenarios;
    - a deploy verdict's bound held on 0.938 and 0.954 of worlds drawn from the prior (nominal
      0.95, inside both Clopper–Pearson intervals), and on 0.952 and 0.989 at a fixed truth.
  - **Not built:**
    - the joint design for correlated channels, a matrix geometric mean the lab measured 6% better;
    - a dynamic decision;
    - the comparison on a marketplace flagship, which waits for a zones-by-time marketplace model.

    The design record is `docs/adr/0011-designing-an-experiment.md`.

- **`chc.switchback`: which switchback to run on a plant with memory, and how to read it, for a
  named effect.** A zone's state remembers what the lever did, so its periods are not independent
  units. `design_switchback(estimands, prior, periods, zones)` plans for the effects a controller
  needs: the channel `b` (`CHANNEL`), the effect `tau_H` of holding the lever on for `H` periods
  (`Horizon(H)`), or the steady state (`STEADY_STATE`). Each effect is reported at the least
  favourable persistence in the prior's interval, with its standard error, minimum detectable
  effect and loss against its own best design. `read_switchback(lever, outcome, estimand,
  analysis)` reads the effect off the data with a standard error from the data. Scope: a
  first-order state per zone, `0 < a < 1`. The plug-in's reading warns when the data reject a
  first-order state, and the plan's variances take the zones as independent.
  - **The design is aligned to the effect.** The plug-in `b_hat S_H(a_hat)` loses nothing to the
    unknown persistence when the design's `A1 = S_H'/S_H`. At two periods that is a Markov design
    flipping with probability `a/(1 + 2a)`. Several effects meet at their minimax, and a floor on
    the number of switches stops the steady state asking never to switch. Under measurement noise
    every effect is read by IV. Without a trusted state model, only model-free readings are
    planned.
  - **The reading.**
    - A heteroskedasticity-robust covariance, with the lags that noise and overlapping sums create.
    - Fieller's interval for the steady state.
    - A block difference in means centred on the midpoint of the earlier on and off blocks. It is
      unbiased at any number of blocks; at eight, the realised-means form reads 27% low. Each
      zone's difference is its own and the zones are weighted alike, and the standard error sums
      the zones' scores block by block, so it keeps what spillover puts between them.
    - A test of the plug-in's first-order state: whether `y_(t-1)` and `u_(t-1)` add to its
      regression. It warns (`switchback_second_state`) when the data reject the state at 1%, and
      warned on 0.8–1.3% of runs on the working model. A lever that alternates every period
      hides a stock it drives from the test, since the stock then alternates with it.
  - **Measured** at `a = 0.8`, over 2000 periods a zone:
    - the lab's reference planner is reproduced to its printed digits;
    - the simulated spread is 0.988–1.040 of the planned standard error, and a test at the planned
      MDE rejects 0.782–0.824 of the time against a planned power of 0.80;
    - the data's intervals covered 0.937–0.959 of 0.95;
    - for `tau_5`, the aligned design's variance is 1.77 times smaller than an i.i.d. design's,
      against 1.69 predicted.
  - **Warnings, not refusals,** for what is outside the model: a second state or a longer carryover,
    drift, spillover and shocks common to zones, a short run near the unit root, few switches. Near
    the unit root (`a = 0.95`, 200 periods), Fieller's interval covered 0.930 where the delta method
    covered 0.877.
  - **Measured on a plant the working model does not describe,** the zone market of `chc.zones`,
    by `scripts/bench_switchback.py`. CHC wrote that plant, so the numbers are by construction. The
    incentive acts through a stock as well, so the state is not first order; half of what a zone
    recruits comes from its neighbours.
    - On the trusted plans the plug-in reads 3% to 14% off. Its intervals cover 0.018–0.79, and the
      first-order test warns on 99.9–100% of runs.
    - The model-free readings are within 0.34% and cover 0.944–0.953.
    - Their plans' variances are the working model's. A test at the planned MDE rejects 0.80 for
      the channel, 0.77 for `tau_2`'s local projection and 0.74 for `tau_5`'s, and 0.43–0.46 for
      the steady state's block difference, whose spread is 1.6–1.7 times the plan's under the
      spillover.
    - The 0.9.0 gate, power within three points of nominal, holds there for the channel, and at
      its edge for `tau_2`.
  - **`restate_mde`: an internal pilot restates the MDE the run can detect.** It reads an effect
    off the first periods of its arm with the plan's own analysis, and carries the standard error
    to the whole arm:
    - a regression's by its rows;
    - the block difference's by its variance over the run's blocks against its variance over the
      pilot's, because the blocks before a zone has shown both settings are lost.

    The multiplier is the noncentral t's, so the run's power at the restated MDE, averaged over the
    pilot, is the plan's. The test at the end and its level are unchanged, and the run is not
    resized. `SwitchbackPlan` now keeps the `alpha` and `power` its MDEs are for. `restate_mde`
    refuses a block pilot with fewer than ten blocks a zone, and one in which a zone showed a single
    setting throughout.
    - **Measured on the market above,** 4000 runs under each matching:
      - The steady state's power was 0.42–0.46 at the planned MDE and 0.79–0.84 at the restated
        one, against the truth.
      - About the reading's own mean, which leaves the reading's bias out, it was 0.75, 0.79 and
        0.80 from ten, fifteen and twenty of the forty blocks. The ten-block shortfall comes from
        the spillover between zones, which the block factor treats as if the centring inflated it.
      - From a tenth of the arm, the channel and `tau_2` held 0.80–0.81, and `tau_5` 0.78–0.79.
    - Checked by `validation/switchback_pilot.mac`. The design record is
      `docs/adr/0020-an-internal-pilot-for-a-switchback.md`.
  - **Not built:**
    - a plan whose variances see a second state or spillover (the internal pilot reads them off
      the run instead);
    - resizing the run from the pilot;
    - a bias-corrected `a_hat`.

    The design record is `docs/adr/0012-a-switchback-for-a-named-effect.md`.

- **`fit_causal_residual(..., weights=...)`: a weight on each transition's channel moment, so a
  channel class that cannot fit every state is fitted where the decision needs it.** `weights` is
  a function of the states alone: it is called with `x` and nothing else, and returns one
  non-negative weight per transition. The nuisances and the drift stay unweighted.
  `solve_channel_moment` takes the same weight per row, and the fit reports `weighted` and Kish's
  `effective_sample_size`.
  - **A weight on the state keeps the moment orthogonal; one that reads the action does not.** On
    exact nuisances perturbed along the covariates, the channel's error goes as the square of the
    perturbation under a state weight and linearly under `exp(u/2)` (slopes 2.19 and
    0.87). A callable that never sees the action makes that structural.
  - **Under a class that misses the truth, the weight chooses the estimand**: the linear fit to the
    true channel under `w s^2` times the log's law of states, `s^2(x)` the variance the adjustment
    set leaves in the action. For a one-shot decision at states drawn from `Q`, with regret
    curvature `kappa` in the channel, `w = kappa (dQ/dP) / s^2` makes the fit the best of its class
    for that decision. On a quadratic channel fitted by a line, four weights landed on their four
    closed-form lines. The decision weight's regret was 1.00 of the class's floor, and the
    unweighted fit's 9.7 times it; with nuisances outside the polynomial sieve, 1.00 and
    7.9 times. Over 100 logs of 4000 rows, the excess over the floor came to 1.16 of
    the prediction from the weight's variance, `tr(G V) / 2n`.
  - **A weighted fit's standard error is robust and carries the nuisances.** Each row's squared
    residual goes through the fit's own linear map, the cross-fitted nuisances included, since a
    weight that loads a few rows loads the nuisances' error at them too. On a log whose noise grew
    as `exp(x)`, over redraws of the noise:
    - weighed by `exp(-x)`, a sandwich on the moment alone came to 0.67 of the spread, and this to
      1.04 (1.02 under `rk4`);
    - weighed by `exp(x)`, this came to 0.93 and 0.85, where an error pooling the noise over the
      rows came to 0.16.

    Over fresh logs of a missed class the decision weight's error came to 0.96 of its spread.
    The unweighted fit's homoskedastic error is unchanged; on those logs it came to 0.40 of its
    spread.
  - **When the class contains the truth,** every weight reads the same channel and only the
    variance moves. The inverse noise, `exp(x/2)` here, is efficient in the limit, at 0.873 of the
    unweighted fit's error. It scattered the channel 0.87 times as far at 64 000 rows, and 1.15
    times as far at 4000, because it loads the tail where the nuisances extrapolate.
  - **Not built.** A weight the library estimates, whether `dQ/dP`, `s^2` or the inverse noise:
    under a missed class its error moves the channel at first order. A weight for a dynamic plan,
    where a weight on the state left 36–178 times the best fit's regret. `prescribe` is unchanged.
    The design record is `docs/adr/0013-a-weight-on-the-channel-moment.md`.

- **`shadow_price_effect`: the global effect of an experiment on a matching market, read off the
  rows' rents in the experiment's own matching.** A treatment given to some of the rows moves the
  prices every row faces, so the treated-minus-control difference an A/B test reads is not what
  treating every row would do. `shadow_price_effect(cost, supply, demand, treated, *, eps)` solves
  the experiment's market with `sinkhorn` and returns a `ShadowPriceEffect`:
  - `effect`, the treated-minus-control difference of the rows' rents, which is the slope of
    welfare in the treated share;
  - its `standard_error` over the assignment;
  - `naive`, the difference an A/B test reads;
  - `second_order_bias`, what `effect` misses the global effect by to second order, with the
    `curvature` that prices it.

  The rows are the randomised units. The other side is randomised by passing the transposed
  market, and rows outside the experiment, such as an idle pool of supply, are priced but not
  counted (`randomised`). `strata`, one label per row fixed before the assignment, post-stratify
  it.
  - **Measured.** The effect equals a five-point difference of welfare in the share, from a solver
    written apart, to 2e-12 relative. It misses the global effect at second order, and at third
    at `p = 1/2`; the naive difference misses at first. Averaged over the Gauss–Legendre pair
    `p = 1/2 -+ 1/(2 sqrt 3)`, it misses at fifth order. Over 400 assignments at shares 0.2 and 0.5
    on six markets, its spread was 0.90–1.06 of its standard error, and 0.39–0.89 of the naive
    difference's.
  - **The curvature is read less the assignment's noise.** The plug-in `eps g' C^+ g` read the
    true `-f''(p)` 1.3–12 times too large on those markets. Less that noise it read 0.71–1.09 of
    it. From one experiment it is noisy, 0.8–5.8 times its value, and at `p = 0.2` the bias it
    prices was 0.4–7% of the standard error. Design at `p = 1/2`, where that term vanishes.
  - **What it does not price** is the part of the bias that shrinks with the number of rows. It
    was under 0.3% of the standard error on five of the markets, and 2–3% on 80 units against 80
    others where demand is short, where at `p = 1/2` it is most of the bias.
  - **Near the LP limit the second-order law fails,** and the bias saturates at first order.
    `nu_hat`, how far the treatment moves the prices across the shares in units of `eps`, is the
    alarm. Above 1, or on a solve whose marginal residual is over `tol`, `second_order_bias` is
    `None`, with the reason, and a warning is logged. Over twenty markets swept towards the limit,
    the law held to 10% in nine cases of ten below 0.5 and to 33% below 1. The assignment's noise
    raises `nu_hat`, so with few rows in an arm it withholds the claim far from the limit too: in
    4–100% of the experiments on 80 units against 80 others.
  - **Post-stratified** by `strata`, one label per row fixed before the assignment, such as the
    rows' types. Each stratum's difference is weighted by its mass, so the mix of each stratum the
    coins dealt either arm no longer moves the effect. On the same markets its spread was 0.52–0.78
    of the plain effect's, and 0.85–1.04 of its standard error. A stratum with fewer than two rows
    in an arm is refused, not merged: merging it into another after the assignment moved the
    effect by 8–24% of the global effect at `p = 0.2`, and falling back to the plain difference by
    5–10%.
  - **`shadow_price_interval`** is the same reading at `eps = 0`, where the exact LP's duals are
    not unique: the range of the effect over the optimal dual face, `[f'(p+), f'(p-)]`, from two
    LPs.
  - **Not built:** the two-sided estimator `f_RS`, whose RMSE was 0.85–1.04 of the
    post-stratified effect's, and which needs both sides randomised and has no standard error; the
    global effect on a secondary metric such as matches; and a design helper. The design record
    is `docs/adr/0014-the-global-effect-of-a-marketplace-experiment.md`.

- **`chc.zones`: the marketplace plant the loop runs on, zones over time, control-affine by
  construction.** Per zone: idle supply, open requests and an incentive stock; an incentive and a
  price as levers, moves from the do-nothing point on `[0, 1]`. An incentive recruits at once
  through `(I - P) diag(recruit)` and late through the stock, and a share `spill` of its recruits
  come from the neighbouring zones on a ring, so the city keeps `1 - spill` of them. A price turns
  away `demand * elasticity`. `ZoneMarketSystem` is parameterised by the do-nothing point: each
  zone's idle supply and open requests, from which the supply level and the demand are derived so
  that it is the steady state (`validation/zone_market.mac`). Experimental, as the plant of the
  marketplace study 0.11.0 pre-registers.
  - **Matching** is harmonic, `mu s q / (s + q)`, or its tangent at the do-nothing point, which
    passes through zero; the two share the steady state.
  - **The logged operator chased the shock.** `sample` moves both levers with each zone's own
    demand shock, which also takes supply off the road, and logs it. Over six logs a fit that
    ignores the shock read the price at −0.71 to 0.21 of one period's response (a price rise
    raising demand in 18 of 24 zones) and the incentive at 0.18–0.39 of it over the city; adjusted
    for the shock, at 0.78–1.16 and 0.90–1.08 zone by zone. `graph()` derives that adjustment.
  - **Bridges.** `zone_decision` is the settled trips as a `ZoneDecision` over the incentives, and
    `linear_gaussian` one RK4 period at the do-nothing point as a `LinearGaussianPlant`. Both are
    exact under linear matching. Under harmonic matching `zone_decision` is first order: 3.5% of the
    move it predicts at an incentive of 0.025, 12.8% at 0.1. `stock_dynamics` hands the stock rows
    to a fit as `known=`.
  - **Scope.** Every number from it is by construction. The logged actions share the period's
    noise through the shock, which an evaluation that weights actions by `u | x` alone assumes
    they do not. Interference runs through supply only. The design record is
    `docs/adr/0017-the-marketplace-plant.md`.

### Changed

- **`channel_error` is robust for every fit, as it already was for a weighted one (D28).** The
  unweighted fit reported the homoskedastic 2SLS sandwich, which pools the noise over the rows. It
  now reports each row's squared structural residual carried through the fit's own linear map, the
  cross-fitted nuisances included, and under `rk4` through the fixed point too. The unweighted fit
  and one weighted by ones report the same error, to the bit.
  - **Why.** On logs whose noise moves with the state, the homoskedastic error read 0.80 of the
    channel's spread under a class that contains the truth, and 0.40 under one that misses it (800
    logs of 4000 rows). The robust error's root mean square reads 0.96 and 1.04 of it. On a
    confounded plant it came to 1.08 of the channel's error against the truth on the adjustment
    path and 1.05 on the IV path (200 logs of 2000 rows), where the homoskedastic one ran
    optimistic. Under `rk4` it carries the RK4 map's gain on the estimate: 1.70 against 1.73
    measured over 200 noise draws.
  - **What it costs.** One log's robust error scatters, by 24% and 40% of its value on those 800
    logs, where the homoskedastic one moved by 2–3%: read it as a scale. Its mean falls 4–6% short
    of the spread at 4000 rows for that reason; at 16 000 rows neither shortfall is distinguishable
    from zero. The `jacrev` through the cross-fit it needs has not been timed on a quiet machine
    yet.
  - **What moves.** `prescribe`'s identification radius and its tube read it. On the marketing-mix
    case study over four seeds the channel's error went from 0.029 / 0.034 / 0.035 / 0.031 to
    0.028 / 0.035 / 0.036 / 0.032. The lift, the confounded arm's share, `drift_error` and the
    certificate (12 of 12 steps; the confounded arm none) did not move, and no test's number did.
    The BOPTEST case study's tube, 0.29 K one step ahead, was read under the homoskedastic error.

- **`chc.experiment` and `chc.switchback` are filed experimental, and so are the `weights`
  argument and `shadow_price_effect` / `shadow_price_interval`.** Each is verified on plants CHC
  wrote, and was built ahead of the release that verifies it on one it did not: experiments and
  decision-weighted identification are 0.9.0's, the marketplace 0.11.0's. Until then each may
  change or be withdrawn in any release. The README, the API pages, the navigation, the lifecycle
  page and the three entries' docstrings say so. No release carried them, so no published promise
  moves.

- **CI audits the proofs' assumptions, and runs the SMT, PARI/GP and Octave cross-checks.** The
  proofs job compiled every proof without printing what each rests on; it now fails on an axiom
  outside Stdlib's classical reals or an admitted step, by the audit `just assumptions` runs, from
  one script. A new job runs the SMT files, whose z3 verdicts must match the ones each file states,
  and the PARI/GP and Octave files, which must complete; `just crosschecks` runs the same locally.
  The FriCAS, R and Python files in `validation/` are still run by hand.

### Deprecated

- **421 names leave the top level in 1.0; import each from its module (D17).** `chc` now holds the
  lifecycle: every name `docs/lifecycle.md` files, the classes a call to one of them takes, the
  errors they raise, and `__version__`, 63 entries in `__all__`. Every other name 0.7.0 bound there
  still imports from `chc`, with a `DeprecationWarning` at the importing line that names the module:
  `chc.rollout leaves the top level in 1.0: import it from chc.integrate`. Module paths do not move,
  so the migration is a find-and-replace from the warning's text, and `import chc` still imports
  the modules it did, so `chc.integrate.rollout` keeps working. The record is
  `docs/adr/0016-the-top-level-is-the-lifecycle.md`.
  - **What changes without a warning.** `from chc import *` binds the 63 names, not 459, and
    `dir(chc)` no longer lists the others. Type checkers still see all 484, so a checked caller is warned at
    run time rather than broken at check time.
  - **What was never there.** The result types added since 0.7.0 (`PlanEvaluation`, `Verdict`,
    `ExperimentDesign`, `SwitchbackPlan` and their kin) and `fit_logger` live at their module paths
    only.
  - **What else moved.** The library's tests, scripts and notebooks, the README and the bench
    import from module paths, and pytest turns the warning into an error in both repositories, so
    neither can start depending on the top level again.

### Removed

- **`runtime/`, the Rust harness that measured a compiled control loop.** Its verdict, recorded
  under 0.4.0, was 1.38x over the compiled library: not enough to carry a second runtime. It was
  never in the wheel, and its last state is at the `v0.7.0` tag. With it goes `paste`, which
  RUSTSEC-2024-0436 lists as unmaintained, and which reached the repository only through its
  `nalgebra`.

### Fixed

- **What `degree = 1` fits, as documented.** `ControlAffineResidual` and `fit_causal_residual`
  said `degree = 1` fits a constant channel. The drift and the channel share one basis, so it fits
  a channel affine in the state, `B(x) = C_0 + sum_l C_l x_l`, as it always has; only `degree = 0`
  makes the channel constant, and the drift with it. Nothing computed changes. What it means for
  a caller: a fitted channel is read at a state, and `C_0` alone is the channel at `x = 0`. A plant
  with `n` states and `m` levers fits `n m (n + 1)` channel coefficients rather than `n m`.

## [0.7.0] — 2026-09-28

### Added

- **`sinkhorn` says how far its plan is from its marginals.** `SinkhornResult` gains
  `marginal_residual`, the L1 miss of both marginals over the total mass, and a residual over the
  new `tol` (default `1e-4`) is logged as a warning (`chc_event="sinkhorn"`). The solve runs a fixed
  number of iterations so that it stays a differentiable `lax.scan`, and at small `eps` the residual
  can sit on a plateau for thousands of them: `MarketplaceMatching.synthetic_city(seed=3)` at its
  own `eps = 0.02` is still 2.2e-3 of the mass away after its default 2000 iterations and 1.7e-15
  after 16000, and its dual potentials -- the surge prices -- came back with nothing said. The
  duality gap never showed this: it carries the entropic term and stays positive at convergence.
  Nothing numeric changed; `SinkhornResult` has one more field, so code that constructs one
  positionally needs it.

- **`prescribe` accepts a bound on its target column.** A `Constraint` that names the target
  bounds the steered state's own coordinate and adds no state; it gets the barrier, `gamma*` and
  `hold_constraints` like any other constrained column. Before, it raised `DecisionError`. This is
  how a comfort band around a set point is stated, the first of the three things the BOPTEST case
  study found the façade could not state. A target outside its own bound asks for "as close as the
  bound permits". On the test panel, steering supply to 1 under a cap of 0.5, the priced plan
  crosses the cap at step 4 and its audit certifies no step; the held one peaks at 0.397 and
  certifies all 15, at a task cost of 4.654 against 1.991. It stops short of 0.5 because
  the barrier condition's class-K gain, 1, lets the margin close only geometrically. A column
  constrained twice is still refused, now by a message that names it. The design record is
  `docs/adr/0005-bound-on-the-steered-state.md`.

- **A target that moves inside the horizon: `Target.value` may be a schedule, and
  `QuadraticCost.x_target` one row per state.** `value[k]` is the level for the state after
  `k + 1` actions, so a schedule has `horizon` entries; the cost's `(H + 1, n)` rows reach every
  consumer of its stage costs (`total_cost`, the discrete adjoint, `costate_norms`,
  `perturbation_cost_weights`, `total_cost_diffrax`, `lift_cost`), and `running` and `terminal`
  take the row as an optional `target`. A fixed target computes what it did, to the bit. This is
  the third thing the BOPTEST case study found the façade could not state: its harness steered for
  the highest comfort bound in the next eight hours, which pre-heated up to one horizon early. On
  the test panel, holding supply at 0 for seven steps and at 0.8 after, the schedule's own plan
  costs 0.557 against the schedule, where passing 0 costs 2.561 and passing 0.8 costs 1.532; it is
  already at 0.314 when the level moves. `mpc_control` and `RecedingHorizon` refuse a per-state
  target, since they re-plan with one cost and the window would not move with the loop. The
  design record is `docs/adr/0006-a-target-per-step.md`.

- **Exogenous drivers: a push the plan cannot move but can see coming.** The second of the three
  things the BOPTEST case study found the façade could not state, and with it all three are
  stated. `chc.dynamics.DrivenDynamics` adds `G w(t)` to any dynamics, with `w` moving linearly
  between forecast levels at `k * dt`. `fit_causal_residual(..., drivers=...)` fits the gain
  jointly with the drift. It reads each driver at both ends of a transition (`name`,
  `f"{name}_next"`) and puts both ends into the channel's nuisance covariates.
  `prescribe(..., drivers=[Driver(name, forecast)])` plans against a forecast of `horizon + 1`
  levels. On zones heated against the weather (`tests/test_drivers.py`):
  - without the driver the fitted decay came back between -0.18 and -0.005 against a true -0.5;
    with it the gain, the decay and the channel land within 0.012, 0.014 and 0.010 of the truth
    over eight seeds;
  - the schedule `prescribe` makes with the forecast costs 1.0001-1.0036x the oracle's on the true
    plant, and the one made without it 1.69-1.98x;
  - on the true plant the plan agrees with a hand-written QP to 0.001, and the same forecast read
    one step late misses it by 0.27.

  The forecast moves the drift and never the channel, so `gamma*` is priced with it in place.
  Neither the forecast's error nor the gain's standard error enters the tube. A forecast outside
  the logged range is logged as extrapolation (`chc_event="driver_range"`). The design record is
  `docs/adr/0007-exogenous-drivers.md`.

- **A budget per period on a receding horizon: `chc.mpc.PeriodBudget`.** A budget row in
  `constraints` caps each window, so a loop that re-plans every step spends it again at every
  step: on the ledger lab's plant, 2.0 and 8.3 times a day's budget.
  `RecedingHorizon(budget=PeriodBudget(weights, amount, period))` caps what each period spends,
  and `step(x, t, spent)` reads what the period has spent so far, as measured: the ledger is the
  caller's, since an applied action need not be the planned one. Each period the window touches
  gets one row over its steps in the window: what is left of the current period, or a later
  period's `amount`, pro rata to the share of that period's remaining steps the window holds.
  Every period spends its budget to 3e-15. Against the plan made for the whole run at once:
  - on the lab's plant over three days, a window of one day loses 0.40-0.83% of the budget's
    value, and a window of a sixth of a day 4.6-8.2%;
  - on the marketing-mix plant over six four-week periods, a window of two periods loses
    0.24-0.67% and a window of one period 0.45-1.25%. Its slowest adstock outlasts a period, and
    how far past a period's end to look is the plant's to say;
  - planned on the adjusted arm's fit, which loses 0.6% with no budget, a window of one period
    loses 5.7-8.7% over three periods.

  Where the box forces a spend above a row, the plan spends the least the box allows and a
  warning names the overrun (`chc_event="budget_overrun"`). An unbudgeted loop computes what it
  did. `scripts/bench_period_budget.py` reprints the numbers; the design record is
  `docs/adr/0008-a-budget-per-period.md`.

- **`minimax_action` minimises the worst regret on request.** `criterion="regret"` minimises the
  worst of `D(b) (u - u*(b))^2`, what the action pays over the one that knew `b`; the default,
  `"cost"`, is unchanged. The two answers differ: on the five intervals of its test, the
  minimax-cost action carries 4.1-6.2 times the minimax regret. The worst regret sits at an
  endpoint unless it turns inside the interval, and a root test decides which. With no turn the
  answer is the endpoint equaliser; with one it is found by golden section, and `binding` reads
  the new `"interior"`. The endpoints alone would be wrong there: on `b in [1, 10]` with unit
  weights, the endpoint equaliser's own regret exceeds the worst case it claims by more than 20%.
  A brute-force search agrees on 120 random intervals across both regimes. `MinimaxCriterion` is
  exported.

- **`CausalPlan.shadow_prices()`: what each constraint row is worth to the plan.** Each row of
  `constraints` gets its KKT multiplier at the returned actions: how much the optimum falls per
  unit the row is relaxed, an upper bound raised or a lower bound lowered. It is read off the
  problem the plan was solved for, which `causal_plan` and `RecedingHorizon.step` now keep on the
  plan. The gradient is that of the objective the solve minimised: the task cost, plus the support
  and uncertainty penalties of a pessimistic solve. The box's multipliers absorb every action at a
  bound. `RowPrice.status` says how far a price can be read:
  - `exact`: the binding rows are independent on the actions the box leaves free;
  - `weakly_active`: the row is at its bound with a multiplier of zero;
  - `degenerate`: the binding rows are dependent there, so the multiplier is a set, and two linear
    programs give its range in place of a price;
  - `inactive`.

  `ShadowPrices` carries the stationarity `residual` and `dual_feasible`, to read before the
  prices: three steps of the lab's solve leave a residual 4900 times the converged one. On the
  ledger lab's plant, a day's budget of half what the day spends unbudgeted is priced at
  1.7536641, where Octave's `qp` gives 1.7536659 and the cost's slope is 1.7536660. A budget that
  meets every cap is `degenerate` on `[0, 0.511]`, and the cost's left slope there is 0.511. A
  pessimistic plan is priced on its penalised objective, to 4e-8 of its slope; priced without its
  confounding penalty, the same budget would read 0.2 too high at the same residual. On random
  separable problems under a box, with upper, lower and equality rows, each of the test's 69
  exact prices matches a central difference of the optimum to 4.9e-7. A plan held under a barrier
  is refused, since the barrier's multipliers would enter every row's price. These are the stated
  rows' multipliers, not those of ADR 0003's barrier rounds. `PriceStatus`, `RowPrice` and
  `ShadowPrices` are exported.

- **The docs file the library by the decision it serves.** A new page, "The decision lifecycle",
  takes a decision from logs through six stages: identify, plan, evaluate, experiment, deploy and
  adapt. For each stage it names what is built and where it is documented, and what is not built
  yet:
  - an off-policy evaluation of a whole plan on a plant with memory;
  - an experiment that answers `NotIdentifiedError`;
  - a gate that holds a plan in shadow until the evidence says to deploy it;
  - a monitor that notices the effect has moved.

  Nothing moved, and every page keeps its URL.

### Changed

- **`chc.offpolicy` says what it estimates.** `off_policy_value` weights one step at a time, so
  IPS and SNIPS estimate the candidate's value on the logger's own states,
  `E_{x ~ d_b} E_{u ~ pi}[r]` -- the contextual-bandit value. For a feedback plan on a plant with
  memory that is not the value of deploying it: on a loop where the candidate spreads the state
  past twice the logger's variance, SNIPS converges to -2.92 against a deployed value of -5.76,
  and `overlap_ok` is True, because the stationary state-action ratio a deployed plan needs is not
  the one-step one. The module docstring had promised that it "refuses deployment"; nothing in it
  refuses, and the docs now say so, as does `concepts/pessimism.md`. `max_weight` is documented
  as the share of the largest weight in the total, which is what it always returned, and
  `fit_behavior_policy` says what its single `sigma` costs on heteroscedastic logs. A test pins
  the one-step estimand on that loop. The code is unchanged; the dynamic evaluator is planned
  separately.
- **The positioning table is checked against the tools as they stand, and says what it concedes.**
  EconML was listed as static-only. Its DML estimators return a matrix θ(X), and `DynamicDML`
  estimates sequential effects within the logged horizon: `scripts/econml_reference.py` runs
  EconML 0.17.0 on both, recovering a 2×2 `B(x)` to 0.068 and a linear plant's impulse response to
  0.013. DCBO's licence is ambiguous (MIT in `LICENSE`, GPL-3.0-or-later in its README and
  `setup.py`), not GPL-3.0. Meridian holds each channel's flighting fixed, so it produces no
  schedule. Each row now names the version read on 2026-09-28. The README and `why.md` concede
  four things: `DynamicDML` within the logged horizon, Nerlove–Arrow (1962) for the adstock
  schedule, SCOPE-RL for sequential off-policy evaluation, and CasADi, acados and do-mpc as
  solvers. `chc.estimators` said EconML fails to install beside pandas 3; 0.17 installs, and stays
  out for what it brings with it. Test and proof counts written by hand are gone from the prose:
  `just counts` prints them.
- **Five pages that said more, or less, than the code does.** `chc.mmm` gave "the solver constrains a
  box and not a half-space" as the reason for no total budget; `causal_plan` has taken one since
  0.6.0, and the reason is that `prescribe` has no budget argument. The quickstart listed
  `mpc_control` and `RecedingHorizon` among what `prescribe` wraps; it wraps `fit_causal_residual`
  and `causal_plan`. `theory.md` now says that CI runs only the Maxima derivations, and that the
  PARI/GP, Octave, FriCAS, R, Python and SMT files are run by hand. `why.md` no longer says
  `prescribe` cannot choose its levers: `max_levers` chooses one set for the whole horizon, not one
  per step. The BOPTEST case study names drivers (ADR 0007) as the third of its three statements.
- **`Target.value` and `Driver.forecast` are typed `ArrayLike`.** They were `float |
  Sequence[float]` and `Sequence[float]`, so a numpy schedule or forecast, which both always
  accepted, type-checked only by coincidence. Nothing is converted. A `Target` or a `Driver` that
  holds an array still cannot be compared with `==` or hashed.

### Removed

- **`MatrixRatioAccuracy.ok`, deprecated in 0.6.0 (D13).** Read `cert.status == "not_convicted"`
  where you read `cert.ok`, and `cert.status == "convicted"` where you read `not cert.ok`; the
  values are the ones `ok` returned. The property had warned for one minor, as promised.

### Fixed

- **`mpc_control` stepped its plant at `t = 0`, and every window's plan started its clock at 0.**
  Step `k` now applies its action at `t = k * dt`, and the plan solved there reads the model from
  `t = k * dt` on; `RecedingHorizon.step` takes the loop's clock as `t` (default `0.0`, which is
  what every step used before). An autonomous plant -- every dynamics the library ships -- computes
  what it did, to the bit. A plant with a time-varying term was stepped with its forcing frozen at
  `t = 0` and planned for a window that never moved: on `x' = u + p(t)`, a push arriving at
  `t = 1`, the loop now leans into it from `t = 0.7`, where it did nothing before, and a window
  started at `t = j * dt` is the tail of the plan started at 0 to 1e-4, where the unclocked window
  missed it by 2.7. This is the clock the drivers work needs. ADR 0006 had also said that rollouts
  pass `t = 0` to every step; they step `k * dt`, and the record is corrected.

- **The discrete adjoint read every step at `t = 0`.** `control_gradient_adjoint`,
  `costate_norms` and `perturbation_cost_weights` took each step's Jacobians at time zero, while
  `rollout`, and so `total_cost`, steps the clock to `k * dt`. On a plant that moves in time the
  descent in `causal_plan` and `projected_gradient_control` followed the gradient of a different
  map: on a forced scalar plant the adjoint was off autodiff by 0.14 where the largest entry is
  0.80. Every dynamics the library ships ignores `t`, so none of its numbers move; a plant of the
  caller's own that reads `t` gets the gradient of its own objective now. Each of the four sites
  is pinned by a test that fails when it is put back at `t = 0`.

- **The `rk4` fit stopped short of the fixed point that defines it, and its standard errors were
  not that fixed point's.** Present since `integrator="rk4"` shipped in 0.5.0:
  - **The fixed point.** The fit was iterated -- add what an RK4 step leaves of `x_next` to the
    target, refit -- and a state stopped once a pass cut its defect's RMS by less than 1%. That is
    not where the fixed point is. A bias `b` under a noise floor `f` moves the RMS by only
    `b^2/(2 f^2)`, and a pass contracts the drift only by `1 - (1 + z + z^2/2 + z^3/6)`, 0.51 at
    `z = -0.7` and 0.77 at `-1.2`. On a log at `|A| dt = 0.05` whose action moves the state little,
    the first pass was refused on eight seeds of eight, and `"rk4"` returned the Euler fit to the
    digit: a channel of 0.780 against a true 0.800. Run on until the RMS stopped falling at all, it
    still stopped 0.9 standard errors of the channel and 1.4 of the decay short at
    `theta * dt = 0.7`, `N = 1000` (120 seeds), and 0.09 of the decay short at 1.25 on a log solved
    exactly. The fit is now the solution of `theta = G y(theta)` -- the estimator `G` handed the
    rate `theta` reads off the log, its own fitted rate plus what its RK4 step leaves over -- found
    by Newton's method from the Euler fit in three to six steps from `|A| dt = 0.05` to 1.2. Over
    400 seeds at 0.7 no coefficient is off by more than 0.05 standard errors.
  - **A log with no fixed point.** RK4's growth factor bottoms out at 0.2704, at `z = -1.596`, so
    a log that falls by more than that in one step -- an exact linear mode past `|A| dt = 1.308` --
    has no field that RK4 steps onto it. The fit returned whatever the iteration held: a decay of
    -1.57 at `|A| dt = 1.5`, with the defect at five times the noise. It now raises `ValueError`.
  - **The standard errors.** They were read off the corrected target, which carries the noise once
    more per pass: at `theta * dt = 0.7`, `channel_error` was 0.0032 against a spread of 0.00053
    over ten seeds. They are now the fixed point's own, `K^-1 G S G^T K^-T`: the noise `S` the
    Euler sandwich reads, carried through the fit `G` and through the RK4 map's gain on the
    estimate, `K = I - G dy/dtheta`, the matrix Newton steps with. Over 200 noise draws on one log
    the channel scattered 1.03x what `channel_error` says; on 100 draws on the marketing-mix rows,
    whose sales row leaves out a seasonal push, 0.96x. The channel reads its noise after the
    adjustment set has taken out what it explains, as the Euler sandwich does: the whole defect
    carries that push, and read off it the error would be 12x the spread. `drift_error` stays
    conditional on the channel and keeps the push, as under Euler, so on those rows it is 3.4x the
    drift's scatter over noise draws. On a log the model class fits, the RK4 map couples the drift
    to the channel it is conditional on, and the drift scattered 1.26x it at `theta * dt = 0.7`
    over 400 seeds (1.02x under Euler): a scale, as documented, now with its numbers.

  Every `rk4` fit moves, and `prescribe` fits with `rk4` by default. The marketing-mix case study
  over four seeds:
  - the prescribed schedule's lift over an equal split moved from 4.4 / 4.3 / 7.4 / 8.0% to
    4.5 / 3.2 / 7.7 / 7.9%;
  - the confounded arm keeps 0.87 / 0.78 / 0.81 / 0.78 of the adjusted arm's lift;
  - `channel_error` goes from 0.061 to 0.029 and `drift_error` from 0.109 to 0.097. They feed the
    tube and the identification radius, and the certificate stays at 12 of 12 steps on every seed;
  - the README row now reads +4.5%, channels inflated 2.3-8.1x, 15% under-invested and 87% of the
    lift. It had quoted +4.3%, which was already stale.

  Two claims did not survive. `tests/test_mmm.py` asserted that the adjusted arm recovers the true
  social:search return ratio, 0.4 +- 0.1. The sales row there carries a saturating carryover and a
  seasonal push the model class leaves out, so the ratio moves with how far the fit gets: 0.43 read
  as Euler, 0.33 after the one pass the old rule allowed, 0.20 at the fixed point. Over four seeds
  it spans 0.20-0.58. The test now asserts what holds on every seed: the confounded arm reads it at
  least 0.2 higher. The same file put the 0.025 left on one known adstock row down to the model
  class. It was the noise: with the noise taken out, the fixed point sits at zero on every known
  row to 1e-6, and over noise draws that coefficient scatters by 0.041.

  A third claim was in the benchmark, whose CI the refusal turned red. Track M's
  carryover-dominant plant decayed its fastest channel at `theta * dt = 1.5`, and seed 0's log
  reads that channel's weekly decay at 0.2668 +- 0.0033, below any RK4 step. The plant moves to
  1.0; at 1.2 seed 5 has no fixed point either, stalling in that channel's dependence on its own
  adstock. The whole-horizon plan's win there goes from 6 of 6 seeds to 8 of 8, by 1.20-3.79
  (+9.5%). On the shipped plant the tie stays a tie: 46.43 against 46.11 over eight seeds, sign
  3/5 (was 46.56 against 46.88, 5/3), and the myopic rule now beats the equal split at 7 of 8. The
  numbers in `chc.mmm`'s scope note are re-measured.

## [0.6.0] — 2026-09-27

### Added

- **A BOPTEST case study for `prescribe`, closed loop on a live building emulator
  (`docs/case-studies/boptest.md`).** Pre-registered in `causaldyn-bench` (L8.1): a heat pump
  planned from a weather-compensated log, the call with the graph against the same call with
  nothing adjusted, energy read at the built-in baseline's comfort. The gate reads REFUTED, the
  naive call cheaper in all four replicates both arms could be read in, but both validity checks
  failed, so the verdict does not stand; the naive loop also ran away in eight weeks, where its
  fitted channel changed sign. The page lists what the façade could not state there: a bound on
  the steered state, a weather term in the drift, a target that varies within the horizon.

- **A pendulum case study for `prescribe`: Pendulum-v1 held at an angle, planned from a log whose
  torque was confounded (`scripts/pendulum_demo.py`).** The control audience's counterpart of
  `chc.mmm`. While logging, an operator cancelled half the wind torque it measured, so the logged
  torque moves with a disturbance that enters `omega'` exactly where the torque does: the
  confounding sits on the control channel. Gravity goes in as `known=`, the actuator is what the log
  has to identify, and every schedule is run on the *true* pendulum:

  ```
  | reading | identification | adjusted for | torque channel | trusted steps | final angle error | rms angle error | peak omega |
  |---|---|---|---|---|---|---|---|
  | adjusted | identified | wind | +3.002 | 4 | +0.003 | 0.141 | 0.460 |
  | asserted | asserted | nothing | -1.064 | 0 | +0.610 | 0.667 | 0.869 |
  | latent | not_identified | - | - | 0 | no schedule | - | - |
  | none | - | - | - | - | +0.300 | 0.300 | 0.000 |
  ```

  The task is to raise the pendulum 0.30 rad from hanging and hold it for 40 steps of 0.05 s,
  inside Pendulum-v1's torque box of 2, with `|omega| <= 1.0` held in the solve as the barrier.
  Adjusted for the wind, the channel is `+3.002` against `3.000`. Asserting an empty set returns
  `-1.064`, where the omitted-variable formula for white wind,
  `b (k(k-1) s_w^2 + s_e^2) / (k^2 s_w^2 + s_e^2)`, predicts `-1.052`: an operator who cancels half
  the wind teaches the log that torque pushes backwards, and the plan ends `2.0x` as far from the
  target as never acting. Declared latent, the wind leaves no schedule.

  Four trusted steps of forty is what the certificate can say here, and the script's docstring says
  why: the tube compounds at the hanging pendulum's logarithmic norm, `(15 - 1)/2` per second, and
  `channel_error` averages every channel coefficient's standard error, the constant term included
  -- the channel at upright, where the log never goes. The audit, not the certificate, shows the
  rest of the schedule holding.

  A script with a test that drives the file itself (`tests/test_pendulum_demo.py`) rather than a
  module, because nothing imports the plant. Its numbers are float32, the default, and the header
  says so; in float32 a warning adds that the fits match float64 to the digits printed while the
  barrier-held solves do not, so peak omega and the asserted schedule's errors move.

- **Which levers to use: `prescribe(max_levers=k)` (L1).** `prescribe` planned with every lever it
  was handed, so a pilot allowed to move two things had to guess which two. `max_levers=k` chooses
  at most `k` by greedy forward selection with the plan as its inner loop: from no lever, each step
  plans once per lever not yet chosen, with that lever added -- a cold `causal_plan` with the levers
  outside the set pinned to `[0, 0]`, the plan `prescribe` makes for that set alone -- and keeps the
  cheapest, or under `hold_constraints` the one whose audit clears the longest prefix. An unselected
  lever is held at zero, where the fitted channel credits it with no effect and its `unit_cost`
  charges nothing, so under `max_levers` every lever's box must contain zero; a box that excludes
  it, or `max_levers < 1`, raises `DecisionError`. `Prescription.selection` is a frozen
  `LeverSelection` -- the idle cost and one `SelectionStep(lever, task_cost, regret_bound)` per
  step -- carried by `to_json` under `"selection"` and tabulated by `report()`; `schema_version`
  stays `1`, since a field was added and none changed meaning. A step's regret bound is `plan_regret_bound` priced against every
  lever's box, so it bounds what the levers left out could still buy, and with it greedy's miss
  against the best set of its size. Each step leaves one `selection` log record with every
  candidate's cost. Without `max_levers` nothing moves: four prescriptions hash identically before
  and after, and a test holds the `None` path to its one cold solve.

  Greedy is not exhaustive. Against exhaustive search it found the best set in 5 of 6 cells on three
  named plants, missing on one built for it to miss, and in all 84 on 28 random four-lever plants on
  which some lever lowers the cost. The group-L1 stage first proposed for this was measured and not
  shipped: as a screen before greedy it repaired no miss and broke a hit on each set of plants, and
  as the selector itself it repaired the built miss and broke 2 hits on the named plants and 15 on
  the random ones. Greedy followed by single swaps repaired the miss and broke nothing, and is the
  first thing to add if misses show up in use. The decision, the tables, the rejected alternatives
  and the mutation check of the tests are in `docs/adr/0004-greedy-lever-selection.md`, reproducible
  with `scripts/bench_max_levers.py`.

- **The matrix statements behind the scalar Rocq proofs, proved with MathComp
  (`proofs/mathcomp/`).** Stdlib has no matrices, so the robust barrier margin, the multivariate
  van Trees floor, the certainty-equivalence constant and the dimension limits of the KAN,
  circulant, Riccati and delay results were machine-checked only as scalar or 2×2 shadows, with the
  matrix forms resting on Maxima and numerical sweeps. Six files now prove them over abstract
  fields at any dimension: the operator- and Frobenius-ball reduction of the robust barrier margin
  to the channel ball, and discrete-time forward invariance for an affine barrier under the Euler
  step; the regret identity and trace floor of multivariate van Trees; completing the square, the
  exact regret-to-go and the explicit constant of the certainty-equivalence bound for any `n`, `m`,
  with the positive-semidefinite side of the discrete Riccati recursion; `n`-input KAN floors,
  `N`-mode circulant bounds and the multivariate delay half-line. All 191 lemmas are closed under
  the global context, and `just assumptions-mathcomp`, a separate CI job, fails otherwise.
  Continuous-time forward invariance (Nagumo 1942, Brezis 1970), the Riccati escape time and the
  transcendental delay boundary stay cited or scalar.

- **A documentation site, every result on it printed while it builds.** `just docs` runs
  `docs/build.sh`, the same script CI and the Pages deploy run: it executes the eight notebooks and
  the scripts the pages quote, then `mkdocs build --strict`, so no page carries a number of its own.
  The API reference is one page per public module, filed under the README's stability tiers, and
  `tests/test_docs_api_pages.py` fails when a module has no page, a page renders a module that does
  not exist, or a page's tier disagrees with the README. mkdocs-material and mkdocstrings live in a
  `docs` dependency group, so neither the runtime nor `dev` grows. `release.yml` deploys the site
  to GitHub Pages as its last job; `docs.yml` redeploys by dispatch. The site loads no math
  renderer, for the reason it loads no Google Fonts: either would send a reader's browser to a
  third party. The three TeX spans the tutorials carried are written as inline code with Unicode
  instead, and `tests/test_notebook_prose.py` fails on a TeX span in any notebook's prose.

- **Every GitHub Release carries a CycloneDX SBOM.** `release.yml` exports
  `causal-hybrid-control-<version>.cdx.json` from `uv.lock` with the runtime dependencies only,
  checks that it names the version being released, and attaches it next to the wheel and the sdist.
  Release tags are cut signed (`git tag -s`), as 0.5.0 and 0.5.1 already were.

- **`residual_blindness_sweep`, and `solve_mfg_dgm(optimizer="lbfgs")`: Result 55's blind residual
  as a distribution, and the half of it that is proved (P4).** Result 55 measured one seed, one width
  and one optimiser, and read the raw residual's negative rank correlation with the error as
  structural. The two additions separate what is proved from what was measured.

  `proofs/mean_field_dwr.v` proves the first half: an approximator whose reduced state stays within
  `B` keeps its reduced residual below `(1 + |A| + |qc|)B` at every horizon, while its error exceeds
  any prescribed level close enough to the obstruction (`bounded_residual`,
  `error_exceeds_amplitude_gap`, `bounded_approximator_is_blind`, on Stdlib's classical reals and
  nothing else). `validation/bounded_approximator_blind.smt2` states the negations of both bounds;
  z3 and cvc5 return `unsat`. So the residual cannot rank the error near `T*`. That it ranks it
  backwards is not implied.

  `residual_blindness_sweep(horizons, *, width, optimizer, steps, seed)` returns a `BlindnessSweep`:
  one solve per horizon on the anti-monotone instance in one fixed box, scored by the raw residual,
  the residual conditioned by `1/|den(T)|` and `dual_weighted_error_estimate`, each ranked against the
  true error. `solve_mfg_dgm` takes `optimizer: MeanFieldOptimizer = "adam"`; `"lbfgs"` runs `steps`
  L-BFGS iterations with a zoom line search on one fixed collocation draw. Over seeds 0-4, widths 32,
  64 and 128 and both optimisers (`just paper-4` in `causaldyn-bench`), the raw residual ranks the
  error backwards in 15 of 15 Adam configurations and forwards in 13 of 15 L-BFGS ones, so the sign
  is the optimiser's. Scalar conditioning is positive in 30 of 30 and perfect in none. The
  dual-weighted estimate ranks perfectly in 28 of 30 and at `0.976` in the other two; its worst
  relative discrepancy from the error is `0.12` under Adam and `0.80` under L-BFGS. The default
  optimiser is unchanged, so no existing numeric behaviour changed.

- **The capped-exploration optimum for a whole objective class, and the exploration floor attained
  by an estimator rather than a schedule (P3).** Two gaps paper P3 would have been refereed on.

  `proofs/greedy_fill.v` proves, at every horizon, that the greedy fill -- each round at its cap
  until the total is spent -- costs no more than any feasible schedule of the same total, for
  **every** estimation term that is a sum of non-increasing functions of the information delivered
  before each round, and strictly less when it delivers strictly more before a strictly decreasing
  term. Results 56 and 66 had proved the exchange of two rounds for the one term `K/(I0 + c S)` and
  left the `T`-round step to Maxima and the certificate, on the grounds that Stdlib has no
  schedules. It needs none: every feasible prefix sum is at most `min(M, C_t)`, the greedy fill
  attains that at every `t` at once, and nothing else is used -- no convexity, and not the form of
  the term. A budget constrains only the total, so it changes which total is optimal and never the
  shape. 16 lemmas on Stdlib's classical reals and nothing else; an exhaustive check over 458,899
  schedules on a grid at `T = 3`, with random non-convex step functions, found nothing cheaper.

  `validation/minimax_exploration.mac` STEPs 10-11. `minimax_exploration_certificate` shows that a
  front-loaded *schedule* attains `c_causal sqrt(T)` of the reduced objective, which charges every
  round the van Trees floor. STEP 10 shows a *policy* does: explore-then-commit with a
  constant-magnitude probe `s_t sqrt(M/n)`, fair signs, and least squares has regret
  `n K d^2 + A M + (T - n) K/(c M)` exactly when `u*` is affine, hence `c_causal sqrt(T - n) +
  n K d^2` at the optimal budget -- the committed rounds costing exactly what the probe did. STEP 11:
  the same budget as Gaussian dither is not the same design, because least squares sees the probe's
  realised energy and `E[1/chi2_n] = 1/(n - 2)`: a factor `(n - 1)/(n - 2)`, and for `n <= 2` an
  infinite regret. On the certificate's own non-affine plant (Table 6 of `causaldyn-bench`'s
  `just paper-3`) the constant probe reaches `0.9929 +- 0.0030` of the floor at `T = 1e7`, while one
  round of dither loses the rate: its regret grows with slope `0.752`, not `1/2`, because the bounded
  `u*` clips an estimate whose variance is infinite.

  No numeric behaviour changed.

- **`cross_cluster_mixing_certificate`: the cluster rate does not need independent clusters (A3').**
  `multichannel_control_certificate` and `clustered_lower_bound_certificate` both draw independent
  clusters, so everything downstream of them rested on partial interference with independent groups
  (Hudgens & Halloran 2008) -- the assumption a marketplace violates first, since adjacent cities share
  drivers, weather and campaigns. A3' weakens it to `psi`-dependence in the sense of Kojevnikov, Marmer
  & Song (2021): cluster scores may correlate across clusters provided the coefficient `theta_s` decays
  fast enough for their Condition ND. This certificate measures both halves of the resulting claim,
  because "the rate survived" alone is unfalsifiable -- it is also what an arm with no dependence at
  all reports.

  Each arm draws **two independent** unit-variance AR(1)-across-cluster fields, one entering the
  spillover regressor and one the outcome error. With two, the per-unit cross moment is zero while the
  score picks up `Cov(s_g, s_h) ~ rho^(2|g-h|)`. At `240` seeds x `5` independent replicate blocks over
  `G` in `80..1280`, the fitted slope of RMSE against `G` reads `-0.5013 +- 0.0072`, `-0.5210 +- 0.0099`,
  `-0.5054 +- 0.0072`, `-0.4912 +- 0.0087`, `-0.4702 +- 0.0131` for `rho = 0, 0.3, 0.5, 0.7, 0.85`.

  Two, not one, and the reason is a decoy rather than a nuance. Sharing one field between the regressor
  and the error makes it an omitted confounder, and measured that way the certificate reports slopes
  `-0.001 .. +0.027` -- **indistinguishable from the non-summable arm below**, so the bug would read as
  "the rate broke" rather than as a broken plant. Nothing in the slope separates them; the point
  estimate does. Hence `worst_mean_error` is part of the returned curve and gated in the tests. On the
  run above it is `0.00492`; on a matched pair at reduced settings it is `0.0138` with two fields and
  `0.3377` with one, a `24x` gap.

  The negative control is the half that makes those numbers mean something: one arm replaces the decay
  with a **fixed** count of super-blocks, so `theta_s` never decays and Condition ND(b) fails. Its slope
  is `+0.0000 +- 0.0103` -- the rate is gone, not merely worse, and the separation from the nearest
  summable arm is `28.2` standard errors.

  What A3' actually moves is the CONSTANT, and that is predicted rather than observed: the bread is free
  of `rho` (a unit-variance field has unit variance whatever its correlation length), so `G*MSE` must be
  AFFINE in the HAC sum `sum_{d != 0} psi(d) = 2 rho^2/(1 - rho^2)`. Fitted, `G*MSE = 0.2205 + 0.1182 *
  sum psi` with `R^2 = 0.99487` across arms whose HAC sum spans `0` to `5.207`; the worst point is
  `8.8%` off, against the `4.1%` Monte-Carlo standard error of an MSE from `1200` draws.

  `validation/cross_cluster_hac.mac` carries the symbolic side: the HAC sum `2r/(1-r)` with `r = rho^2`,
  and the finite-`G` Bartlett-weighted sum `s_G = 2 sum_{d=1}^{G-1} (1 - d/G) r^d` whose deficit obeys
  `lim_G G*(s_inf - s_G) = 2r/(1-r)^2` -- both at residual `0` against the stated form, and re-summed
  independently at 40 digits (worst residual `8.2e-16`). That deficit is why the fitted slope comes out
  SHALLOWER than `-1/2` at high decay: `s_G` reaches its limit from below, so `G*MSE` rises with `G`.
  It predicts `+0.005383` on `G in [80,1280]` and `+0.002691` on `[160,2560]` -- halving as the window
  doubles, the shape the measurement shows -- and accounts for `17.2%` and `36.9%` of the measured
  gaps, no more. The sign and the walk are derived; the magnitude at the low window is not.

  Two arguments are rejected at the boundary rather than silently mismeasured. A decay of `1` is **not**
  the limiting case it looks like: an AR(1) at `phi = 1` is a field CONSTANT across clusters, which is
  exactly the common shock KMS's conditional definition conditions on, and which the cross-fit nuisance
  intercept absorbs -- measured, that arm reports `G*MSE = 0.117` against `0.240` for independent
  clusters, i.e. LESS dependence, not more. Fewer than four decay arms is also refused, since below that
  `hac_r_squared` is `1.0` by construction and says nothing.

- **Linear constraints on the action sequence: budgets and rate limits (L3, L3.1).** Every solver
  took a box on each action and nothing else, while the plans that get executed carry a total budget
  and a limit on how far a lever moves per step -- both linear in the flattened actions, neither a
  box. `LinearConstraint(matrix, lower, upper)` states them over the row-major flattened sequence
  (step `k`, lever `j` is column `k*m + j`), two-sided with `+-inf` for an absent side, and
  `LinearConstraint.rate_limit(horizon, caps)` builds the bands. `projected_gradient_solve`,
  `projected_gradient_control`, `pessimistic_solve`, `pessimistic_control` and `causal_plan` take
  `constraints=`; `Lever(cap_per_step=...)` carries a rate limit into `prescribe`, whose plan log
  names the levers it limited. `SolverResult.constraint_violation` reports the returned plan's worst
  row excess, and `stationarity` is measured against the constrained projection. Calls without
  constraints take the old code path and return bit-identical results.

  Every iterate stays feasible, so a solve stopped by its step budget still returns a plan that can
  be executed. Only the projection changed: Dykstra's algorithm onto `box ∩ rows`, with the rows
  coloured into mutually orthogonal classes projected at once and the box last in every sweep,
  warm-started from the previous projection's increments; then a dual active-set polish from those
  increments, kept only if the primal-dual pair passes the projection's KKT conditions. An empty
  feasible set is refused before the solve by one linear program, since Dykstra cannot detect it.

  Against the alternative -- a Powell-Hestenes-Rockafellar augmented Lagrangian with the box in its
  inner solve -- under a decision rule registered before the first run: on three plans where both
  limits bind, the shipped solver lands within a relative `8.9e-8`, `1.3e-8` and `4.7e-10` of the
  exact or best-of-two reference optimum with violations at rounding, at `1/38` to `1/94` of the
  augmented Lagrangian's gradient and cost evaluations; the augmented Lagrangian misses the `1e-6`
  accuracy gate on the oscillator (`4.1e-6`). Against the exact least-distance projection on 3000
  random polytopes, the shipped projection is at most `4.3e-12` away, and none runs out its sweeps.

  The property test found two defects on the way, both now pinned. Stopping when the iterate stops
  moving stopped one instance after 31 sweeps at a point breaking a row by `7.8e-2` -- a sweep can
  hand the iterate back unchanged while the increments still travel -- so the projection stops on the
  increments (Birgin & Raydan 2005). And near a vertex where one more constraint is nearly active,
  Dykstra holds the iterate still for thousands of sweeps while an increment drains: 17 of the 3000
  ran out the cap, up to `2.7e-3` from the projection. A polish on the active set the increments'
  signs name left 10 of them, and a primal-dual active-set correction 9 -- every one a vertex at
  which the increments named exactly one constraint too many, so that the set was over-determined
  and the solve singular. Stepping only as far as the first multiplier that would change sign -- the
  ratio test of Goldfarb & Idnani (1983) -- releases it. That polish is trusted only through its KKT
  check, which reads complementary slackness off the multipliers themselves: a check on feasibility
  and on the signs the method believes it holds accepted points up to `0.91` from the projection
  when its steps were starved, and `test_the_polish_vouches_only_for_the_projection` holds the line.

  Under a rate limit `plan_regret_bound` is still priced against the box alone, so it is
  conservative rather than tight. The decision, the registered rule and every number above are in
  `docs/adr/0001-linear-action-constraints.md`, reproducible by counts with
  `scripts/bench_linear_constraints.py`.

- **The barrier condition held inside the solve: `causal_plan(barrier=...)` (L3).**
  `certify_safety` audits a finished plan against the worst-case control-barrier condition
  `grad h . xdot >= -alpha h` over the identified set, and until now that was all anything did with
  it: a plan could come back and fail its own audit, and the only enforcement was truncation or the
  one-step filter. `BarrierConstraint(barrier, alpha, gamma, cvar_gap)` carries the audit's
  arguments into `causal_plan`, which holds the condition at every planned step by
  Powell-Hestenes-Rockafellar augmented-Lagrangian rounds around the same descent -- one multiplier
  per step, the box and the linear rows kept in the descent's projection, the plan's support and
  uncertainty penalties kept in its objective, and the safeguards of Birgin & Martinez (2014). The
  audit stays the source of truth: `CausalPlan.safety` is `certify_safety` run on the finished
  plan, and the solve and the audit form the condition through one shared helper.
  `prescribe(hold_constraints=True)` hands its constraints to the solve the same way; it is off by
  default, so no existing schedule moves.

  The rounds aim a relative `1e-6` inside the condition, because an active step solved exactly sits
  on the boundary, where rounding alone fails the audit about half the time -- measured rather than
  supposed: the one-step filter clips onto that boundary, and fails its audit by at most `1.7e-16`
  on 16 of 40 steps. Against the exact QP optimum, or the better of SLSQP and trust-constr, on a
  two-state plant, an oscillator's velocity floor at `gamma = 1` and `2`, and a capacity limit on
  SIR, the held plan certifies every step where the barrier-free plan meets the condition at 7 of
  12, 34 and 32 of 40, and 72 of 100 steps, and it costs `2.5e-7`, `7.0e-6`, `2.2e-5` and `6.1e-6`
  above the optimum, relative. A smaller back-off buys a smaller gap and stops converging on SIR;
  filtering the free plan instead costs `1.84` and `1.64` above the optimum on the oscillator; a
  quadratic penalty without the multipliers reaches the same plans at a penalty two to five decades
  larger, converging on SIR only at its ceiling. A barrier the free plan already clears changes no
  action, status or iteration count, even when that solve stopped on its budget.

  Only the answer is held, not every iterate: a budget-stopped solve, or a condition no admissible
  action can meet, comes back short, and the audit says so rather than anything raising. Where
  `grad h = 0` at a planned step -- a constant barrier, a ball at its centre -- `certify_safety`
  cannot price a radius and raises, as it did before; `causal_plan(barrier=...)` inherits that and
  raises after the solve. `certify_safety` is bit-identical to before on 51 audits in both
  precisions, and `chc.support`'s penalised descent now accepts `support=None`. The decision, the
  measured alternatives and the mutation check of the tests are in
  `docs/adr/0002-barrier-in-the-solve.md`.

- **A receding horizon warm-started from its last plan: `RecedingHorizon.step(x)`, and
  `causal_plan(warm_start=...)` (L3).** `chc.mpc.RecedingHorizon` holds `causal_plan`'s arguments
  and plans from each measured state; `step` returns the whole `CausalPlan`, so the audit reaches
  the caller with the action it covers. Each step starts the descent from the last plan shifted one
  step, and a held barrier's rounds from their last multipliers shifted the same way, under a
  penalty chosen afresh; the barrier-free solve still runs first, so a slack barrier still changes
  no action. `causal_plan(warm_start=...)` is the actions half, for a caller's own loop; the start
  is projected onto the box and the rows before the first step, so it need not be admissible.

  Counted in descent steps over closed loops of ADR 0002's barrier instances -- 40 steps of an
  oscillator under velocity floors of `0.8` and `0.3`, 60 of SIR under a capacity limit -- the
  controller takes 35 %, 36 % and 3 % fewer than cold solves at the default cap, with the realised
  cost within `2e-6` of theirs. The multipliers are most of it where the barrier binds: the shifted
  actions alone save 21 %, 2 %, and on SIR cost 4 % more. At a 100-step cap the controller saves
  29 %, 15 %, and costs 5 % more on SIR, and its closed loop costs within `5e-6` of the converged
  one's, where the cold loop's costs 0.26 % more on the looser floor. Carrying the grown penalty
  over as well was measured and rejected: the next rounds stalled 12 % to 112 % above the optimum
  while reporting `converged`.

  **Replanning compiles nothing.** Each `causal_plan` call compiled two to five programs afresh:
  the cost history, built from a Python list whose length is the step count; the last cost, read
  by indexing on the device; the trajectory's eager `lax.scan`; and `pessimistic_solve`'s
  stationarity gradient, a closure re-traced per call. The conversions now run on the host and the
  two programs compile once, and a test holds three steps of a controller to zero compilations,
  with the caches cleared first. Across 70 outputs of the solvers and plans in both precisions every
  value is bit-identical to before except `pessimistic_solve`'s `stationarity`, which moved by
  `3e-13` relative in `float64` and `4e-5` in `float32`, where the residual's cancellation amplifies
  the gradient's last bits. A barrier passed as a new `lambda` each call still compiles the descent
  each call; the controller holds one. Across processes JAX's persistent cache needs its threshold
  lowered as well as its directory set: at the default one second it wrote none of a first step's
  programs, and at `jax_persistent_cache_min_compile_time_secs = 0` a second process loaded all 73
  compile requests from disk.

  Each step is one `causal_plan`, so nothing outside its horizon carries over: a budget row caps
  each window rather than the run, and a rate limit does not reach back to the action already
  applied. The decision, the table and the rejected alternatives are in
  `docs/adr/0003-receding-horizon-warm-starts.md`.

- **`plan_regret_bound`: how far a finished plan is from the best one the same box allows
  (L3.2).** `DecisionCertificate` shipped with no `regret_bound` because nothing took a
  `CausalPlan`, and the obvious candidate does not survive a box. Result 6's self-certifying
  `|grad J|^2/(2 mu)` needs no optimum, which is what makes it a certificate -- but at a lever the
  gradient holds against its own bound `grad J` is nonzero while the true regret is zero, so it
  charges regret at the optimum itself. On a `2x1` LQ plant whose box `[-0.2, 0.2]` clips the
  unconstrained optimum away, all 12 actions pinned and the realised gap exactly `0`, it reports
  **`55.83`**.

  What replaces it is the same certified gain maximised over the *feasible* moves only:
  `bound = sum_i max{ -g_i d - (mu/2)d^2 : lo_i - U_i <= d <= hi_i - U_i }`. Clipping subtracts
  exactly a perfect square (`gain(d0) - gain(d) = (mu/2)(d - d0)^2`), so the box can never loosen
  the bound, and at a pinned lever the maximiser is `d = 0` and the term is **exactly** zero. That
  plant's bound is `0.0`; a `2x2` plant with one tight lever reports `9.8e-8` where Result 6's
  reports `384.5`.

  It is not vacuous. Against the best cost the same box allows, on deliberately unconverged solves,
  bound/realised comes out at **`1.06`, `1.30`, `1.33`, `1.75`** -- at plans where Result 6's bound
  is `292x` and `1105x` looser.

  And the modulus is allowed to collapse. `gain(d) <= -g d`, whose maximum over an interval sits at
  an endpoint, so the bound is capped by the **Frank-Wolfe gap** `sum_i max(-g_i a_i, -g_i b_i)` --
  which does not mention `mu` and is finite on any bounded box. As `mu -> 0` the bound converges to
  it linearly while `|grad J|^2/(2 mu)` diverges. `gain` is also non-increasing in `mu`, so a
  conservative modulus is looser and never invalid; for a plant affine in the action
  `H - R = B'QB >= 0` makes `lambda_min(R)` a valid one with no eigenvalue solve, and the sampled
  curvature reproduces `lambda_min(H)` to 12 digits.

  The box separates, so `per_lever` sums to the bound and names which lever carries the worst-case
  cost. Where the measured curvature is negative the certificate reports `inf` and `ok=False`: no
  convexity argument applies there, and a finite number would be a fabrication. The bound is on the
  **planning objective** -- how far the planning model is from the plant is the Gronwall tube's
  question, and the two must not be added.

  This also closes the one property `tests/test_decision_properties.py` could not write: over
  random boxes and random solver budgets the bound never falls below the realised optimality gap.
  It has to be checked against a deliberately unconverged plan -- against a converged one
  `bound >= 0` passes it by accident.

  Derived in `validation/constrained_plan_regret.mac`, proved in
  `proofs/constrained_plan_regret.v`.

- **A coercive energy for the port-Hamiltonian residual, and the radius it finally prints (A20).**
  `PortHamiltonianResidual` documented `H' = -dH' R dH <= 0` as making `H` a Lyapunov function, "so
  the residual can't blow up off-support like a black box can". The inequality is real -- it is an
  identity of `(J - R) grad H` and the certificate reproduces it to `2.7e-15` -- but it holds for
  **any** energy network whatsoever, so it cannot tell a useful energy from a useless one. Both
  arms of the new certificate report `H' <= -9.9e-9`; the number separates nothing.

  What `H' <= 0` buys is confinement to `{ H <= H(x0) }`, and that bounds the state only if the set
  is bounded. A `tanh` MLP read out linearly is bounded **in `x`**, so every sublevel set above its
  supremum is the whole space: measured as the log-log slope of its range against the box it is
  measured on, the shipped energy scores **`0.074`** -- it saturates. Its gradient decays in every
  direction too, so the unforced flow has near-equilibria everywhere far from the data, which is
  precisely the off-support regime the docstring was claiming to cover: gradient descent on `H`
  from a `7x7` grid of starts reaches **3** distinct rest points.

  `PortHamiltonianResidual(..., energy="icnn", convexity=eps)` swaps in an input-convex network
  (Amos-Xu-Kolter: `softplus` activations, recurrent weights forced nonnegative through `softplus`)
  plus a quadratic floor, giving `H(x) >= (eps/2)|x|^2`. Its range exponent is **`1.24`**, and
  `invariant_radius(level)` returns `sqrt(2 level/eps)` -- the forward-invariant ball, checked
  against the realised excursion of the unforced flow (`6.00` inside a predicted `25.81`). For the
  MLP the same call returns `inf`, which is the honest answer rather than a missing field. And
  because `grad H` of an `eps`-strongly convex function is injective, the convex energy has
  **exactly 1** critical point, with `|grad H| <= 3.8e-4` at every reported rest point so the
  counts are about `H` and not about how far a finite descent happened to get.

  The strong-convexity constant is measured, not assumed, and the measurement has to be taken far
  out: near the origin the network's own curvature dominates the floor and overstates it. Over
  probes spanning the whole `30x` sweep the smallest Hessian eigenvalue of the convex energy is
  **`0.250000000095`** against a declared `eps = 0.25` -- once every `softplus` unit saturates the
  ICNN is affine and what is left is exactly the floor -- while the `tanh` energy reaches
  **`-0.068`**.

  Derived in `validation/convex_port_hamiltonian.mac`, proved in
  `proofs/convex_port_hamiltonian.v`.

- **The van Trees floor when the effect is a MATRIX, and the alignment a scalar plant hides
  (A18).** Result 57 discharged Result 10's `needs LAM` annotation by working in action space,
  where the one-step LQ regret is exactly a squared error, and closed with *"the multivariate case
  has the same structure with `psi'` a Jacobian and the floor a trace."* It does -- and the
  conclusion changes, which is why `multivariate_action_floor` is a new entry point rather than a
  widened signature.

  The regret identity survives at any dimension: `J(u,B) - J(u*(B),B) = (u-u*)' M (u-u*)` with
  `M = B'QB + R`, exactly and for every `u`. So `E[regret] = tr(M Sigma)`, and because trace
  against a PSD weight is monotone in the PSD order, the matrix van Trees inequality
  `Sigma >= Psi' G^-1 Psi'` becomes `E[regret] >= tr(M Psi' G^-1 Psi')`. At `p = q = 1` that is
  Result 10's constant unchanged, and on two decoupled channels it is `h1 C(b1,x1) + h2 C(b2,x2)`
  -- both reproduced to `4.4e-16`.

  The scalar floor is a **product** of a curvature and an information; this one is a **trace** that
  interleaves them, so it decomposes over the eigendirections of the information as
  `sum_i (Psi' v_i)' M (Psi' v_i)/lambda_i`. Confounding therefore stops having a price and starts
  having an **alignment**: cutting the information along one direction by `k` raises the floor by
  `1 + (k-1) a_w/sum(a)`. Measured with the same `k = 4`, the direction the optimal action leans on
  most costs **`3.40x`** and a direction in the kernel of `Psi'` costs **`1.0000000000`** -- exactly
  nothing. Even the worst single direction falls short of `k`, because `Psi'` has rank 2 in a
  4-dimensional parameter family and `80%` of the weight sits in one direction. A scalar plant has
  one direction, always sits at the first corner, and reports `V_exp/V_conf`.

  Result 57(d)'s knife edge becomes one entry of a vector: a channel at `rr = b^2` carries weight
  `1.2e-32` against its neighbour's `0.074`, so confounding *it* is free. And the bound still binds
  where it should -- a Hodges estimator drives the pointwise error to exactly `0` against the
  unbiased Cramer-Rao floor while sitting `31.6x` above the van Trees floor, `19x` worse than the
  efficient plug-in's `1.67x`.

  `Psi'` comes from differentiating `M u* = -B'Qx` rather than from autodiff -- `chc.regret` is
  numpy/scipy only and a `jax` import there would pull the accelerator stack into a module that
  never needs it -- so the certificate checks it against a central finite difference on square and
  both rectangular shapes (`1.0e-10`).

  Derived in `validation/multivariate_van_trees.mac`, proved in `proofs/multivariate_van_trees.v`.

- **A cap schedule and a spending budget, and the invariant that survives both (A17).** Result 56
  answered "how long do you explore under a per-round action cap" with a single number,
  `n* = sqrt(K T/(A c))/cap`, and recorded as honest scope that a cap which *varies* over rounds,
  or a total budget on top of the cap, changes the feasible set so that `n*` is no longer a
  formula. `capped_exploration_policy` now takes `cap` as a per-round sequence and an optional
  `budget`, and the constant-cap path is left numerically untouched.

  What replaces the formula is an invariant. Differentiating the objective along the greedy fill
  gives `dF/dn = cap(n) [A - K c (T - n)/(I0 + c S(n))^2]`, and the cap is a strictly positive
  factor -- it cannot move a root. So the stopping **mass** is cap-free and the stopping **round**
  is whatever prefix sum reaches it: `n* = min{n : sum_{t<=n} cap_t >= min(B, S*)}`, an index read
  off the schedule. Across five schedules at `T = 4000` -- constant, both ramps, a dead first
  third, and uniform noise -- whose blocks differ by **6.9x in length**, every one stops within
  **half a cap** of the predicted mass.

  `predicted_mass` is that root taken against the *remaining* horizon, `A(I0 + c S)^2 = K c (T-n)`,
  solved as a fixed point of the integer map. Result 56's closed form is its `n << T` limit and
  runs **25% high** once the caps open late, so the field would otherwise report a target the
  policy knowingly misses.

  Two consequences worth naming. A dead actuator early on is not an approximation: `m` rounds under
  a zero cap add exactly `m K/I0` and hand the rest of the horizon to the *same* problem -- block
  length, mass and cost all match the shifted solve, to `1e-13` on the cost. And once a budget is
  spent the cost curve is **exactly** flat, because a round that explores nothing adds one
  estimation term and removes one exploitation term at the same information; the policy therefore
  takes the first minimiser rather than letting floating-point noise pick among the ties.

  Cross-checks that can fail: the `O(T)` sweep against a projected-gradient solve of the full
  convex program on a random box (`1.9e-6`), the digamma identity against the explicit harmonic sum
  it replaces (`1e-13`), and the greedy fill against the same mass spread uniformly, reversed, and
  shifted one block later.

  Derived in `validation/capped_exploration_schedule.mac`, proved in
  `proofs/capped_exploration_schedule.v`.

- **A dual-weighted error estimate that does not need its adjoint written out (A16).** Result 55's
  estimator is exact, and the exactness belongs to the *affine* reduced problem: its 2x2 transition
  matrix is the LQ one and cannot be told otherwise. `adjoint_weighted_error(field, times,
  trajectory, rate, terminal_row)` takes any field and builds the adjoint by linearising it, so the
  same construction runs on a game with no closed form. The pairing identity behind it needs no
  structure at all -- only that the adjoint is driven by the **transpose**, and with the
  untransposed Jacobian it fails by exactly `(j21 - j12)(z2 d1 - z1 d2)`, i.e. it is correct
  precisely on a self-adjoint field.

  `CongestedMeanFieldGame` is the game that makes this measurable: pay for distance from a
  congestion-shifted target, `(q/2)(x - c m - gamma m^3)^2`. The value function stays quadratic in
  `x`, so the Riccati root never sees `gamma` and the reduction to `(m, S)` survives, but the
  two-point problem is nonlinear and there is no closed form -- `solve` shoots on `S(0)` from the LQ
  answer. At `gamma = 0.6` it moves `S(0)` by 72%.

  Measured by `nonlinear_dwr_certificate()` on a manufactured error `yhat = y + eta p`, whose true
  value is known exactly so that "the estimate is first-order" cannot be confused with "the solver
  is bad": on an affine field the relative error is **flat at `6.3e-7` across a 16x sweep in `eta`**
  and that floor is quadrature (halving the step divides it by `3.7`); on the congested field the
  absolute error fits an exponent of **`2.03`**, so the remainder is the second variation and the
  relative error is first-order. Two controls make the adjoint earn its place: keeping the defect
  and swapping in the *affine* linearisation costs exactly one order (exponent **`1.02`**, `196x`
  worse at the smallest perturbation), and using the affine field for both -- what the Result 55
  estimator does if pointed at a congested game -- gives a **constant** `0.19` error, exponent
  `7e-8`: it stops tracking entirely.

  Derived in `validation/nonlinear_dwr.mac`, proved in `proofs/nonlinear_dwr.v`.

- **The finite-population gap, priced rather than fitted (A13).** Result 49 returns a
  Fokker-Planck density, which is the `N -> infinity` limit, and nobody deploys to infinity. Under
  the *mean-field* optimal control the feedback is a function of an agent's own state and of
  deterministic coefficients, so the closed-loop agents are **independent** OU processes and the
  empirical mean is a sample mean of `N` i.i.d. draws. That makes the gap exact at every `t` and
  every `N` -- `E[(m_N - m)^2] = v(t)/N` -- rather than an asymptotic rate with an unnamed constant.

  `MeanFieldSolution.finite_population_rms(n)` is that closed form;
  `MeanFieldSolution.simulate_population(...)` runs the primitive plant under the same feedback so
  the two can be compared without one reading the other. `finite_population_gap_certificate()`
  measures four arms: the gap against the closed form (worst relative error 4.8% at
  `R = 400`, exponent `-0.4989` against a Monte-Carlo standard error of `0.0120`); the
  falsification, where one shared Brownian path drives every agent and the exponent collapses to
  `0.0003`; a fixed coupling gain, which moves the constant to the rate `A + kappa` (ratio `0.98`
  against the moved constant, `1.74` against the unmoved one); and `kappa = c/N`, the scaling an
  N-player Nash actually has, whose bias matches its leading-order constant to `0.5%` under common
  random numbers and sits at `0.21` of the fluctuation at `N = 512`.

  So the exponent is the part that transfers and the constant is the part that moves. Derived in
  `validation/finite_population_gap.mac`, proved in `proofs/finite_population_gap.v`. The
  1-Wasserstein *density* gap is a separate claim with a different constant (Fournier-Guillin) and
  the certificate does not assert it.

- **`CausalGraph.unrolled(...)`: lagged edges, without a second kind of graph.** A time series has
  one more thing to say than a cross-section -- *when* a parent acts -- and the obvious way to carry
  it would fork `d_separated`, `adjustment_set`, `_proper_backdoor` and every other method on a
  lag-aware edge type. Instead the time index goes into the node name: an edge `("spend", "sales", 1)`
  unrolls to `spend[t-1] -> sales[t]`, and the Perkovic criterion runs on the result *unchanged*. The
  cost is node count, `(lags + 1)` times the variable count, which is nothing at this scale.

  `("x", "x", 1)` -- the ordinary autoregressive term -- is accepted, because it is not a self-loop
  once time is explicit; `("x", "x", 0)` is rejected, because within a slice it is. A latent variable
  is latent in every slice, latency being a property of the variable rather than of the moment it
  acted. `CausalGraph.lagged_name(name, lag)` is there so a caller naming a treatment does not
  hand-format `"spend[t]"` and drift from what the constructor produced.

  The argument is `lags=`, **not** `horizon=` as the design note proposed. In this library `horizon`
  is a plan length (`prescribe`), and one name meaning two things is how a caller gets a graph one
  slice short.

- **Supply-chain hardening (L6).** `SECURITY.md` with a threat model that says plainly what is *not*
  in scope -- wrong numbers are correctness bugs and belong in the public tracker, where they can be
  argued about -- and what is: any path where a log record, an exception or a `Provenance` carries
  caller data beyond the column names it was given. `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1).
  Dependabot on `uv.lock` and on GitHub Actions, the second being the one that actually gets
  exploited: every workflow here holds `id-token: write` or `contents: write`. CodeQL over `python`
  **and** `actions`, at `security-extended`. OpenSSF Scorecard. Private vulnerability reporting,
  secret scanning and push protection enabled on the repository.

  Releases now also carry a **SLSA build provenance** attestation
  (`actions/attest-build-provenance`), which is a different claim from the PEP 740 one PyPI already
  had: PyPI's says "uploaded by this workflow", this one says "built from this commit", and it
  verifies with `gh attestation verify` without going through the index at all. The PEP 740
  attestations were checked rather than assumed -- `/integrity/.../provenance` returns both
  artifacts, publisher `causaldyn/causal-hybrid-control` via `release.yml`.

  README gains a **stability statement**: what 0.x promises, three tiers by what a break costs you,
  and the road to 1.0.

### Changed

- **`chc.mintime` is filed experimental, not evolving.** It solves the scalar double
  integrator's time-optimal problem in closed form and nothing more general, which is the
  README's definition of a module that carries one result. The tier table is new in this
  release, so no published promise moves.

- **`MatrixRatioAccuracy.ok` is deprecated for `status`: `"convicted"` or `"not_convicted"`, never
  `"certified"` (D13).** `ok = True` read as a certificate, and the refinement residual behind it
  is an estimate, guaranteed to majorise the error only once the per-node convergence rate reaches
  2 (`residual_bounds_iff_rate_reaches_two`). A rate is a ratio of two errors; the certificate holds
  one difference between two grids and no reference value, so nothing it evaluates measures one.
  Nor can the channel count stand in for it: the rate passes 2 on some `q = 3` grids, and on a
  correlated `q = 2` anchor one step from the existence boundary the default grid's rate is below 2
  and the residual understates the error (`test_the_ratio_status_can_convict_and_cannot_certify`
  measures both against the exact answer). `status` is a `chc.regret.MatrixRatioStatus`, derived
  from the bars rather than stored: `"convicted"` when the relative residual, or the isotropy bar
  where the channels are exchangeable, exceeds `tolerance`, and `"not_convicted"` otherwise, which
  is evidence and not a certificate.

  **Deprecated**, removed in 0.7.0: `ok` stays through 0.6.x as a property that raises a
  `DeprecationWarning` and returns `status == "not_convicted"`. Replace `cert.ok` with
  `cert.status == "not_convicted"` and `not cert.ok` with `cert.status == "convicted"`. `ok` is no
  longer a constructor argument, so drop `ok=` where you construct a `MatrixRatioAccuracy`
  yourself. No numeric behaviour changed: `status` is `"convicted"` exactly where `ok` was `False`.

- **`composition_transfer_certificate` now says why its fitted slopes are not the integers, and
  the difference is derived rather than tolerated.** The certificate reports
  `2.048478 / 4.012425 / 6.002273` against a theoretical `2 / 4 / 6`, and that excess had been
  read as agreement-up-to-noise. There is no noise in it -- the certificate is deterministic. With
  `e = delta^p` and `t = log(delta)`, `log R = const + 2 p t + lam e + 2 c2 e^2 + O(e^3)`, so an
  ordinary least-squares fit over a finite window necessarily reports
  `2 p + lam cov(t, e^(p t))/var(t) + 2 c2 cov(t, e^(2 p t))/var(t)`, with
  `lam = (2 b^3 - 6 b rr)/(rr^2 - b^4)` and `c2 = (b^6 - 8 b^4 rr + 5 b^2 rr^2 - 2 rr^3)/(2 (rr^2
  - b^4)^2)`. New derivation `validation/order_transfer_window.mac` (Maxima, cross-checked in
  giac); the two terms account for `4.09e-2` of the `4.85e-2` excess at `p = 1`, and what is left
  is one more power of `delta` across four windows.

  The same derivation bounds the window from the other end, which is the part that changes what a
  reader should do. `u*(b + e) - u*(b)` is a cancellation, so the regret's relative error grows as
  `delta_lo` falls: at `delta in [1e-5, 2e-4]` the `p = 3` fit reads `6.035`, and the two-channel
  and three-channel `delta^4` sweeps in `multivariate_interference_certificate` and
  `exposure_map_certificate` return `nan` outright, because the regret has underflowed to exactly
  zero and the fit takes `log 0`. The same 12-point fit at 60 digits reproduces the two-term
  prediction to `4.3e-25`, which is what identifies the miss as rounding rather than mathematics.
  No behaviour changed; `slopes` returns what it always did.

- **The statistical results the proofs cite are now named predicates in the theorem types, not
  prose in the file headers.** Five files -- `proofs/c2_end_to_end.v`, `van_trees.v`,
  `clustered_van_trees.v`, `action_van_trees.v`, `multivariate_van_trees.v` -- declare each cited
  input as a `Definition ... : Prop` carrying its citation (`CrossFitRemainder` for CCDDHNR 2018
  Lemma 6.1, `ClusterRobustSampling` for Hansen-Lee 2019 Theorem 2, `LocalQuadraticRegret` for
  Mania-Tu-Recht 2019, `ScoreIdentity` and `InformationDecomposition` for Gill-Levit 1995 and
  Gassiat-Stoltz 2024, plus `ClusteredVanTreesFloor`, `LowerLipschitzRegret`, `ActionVanTreesFloor`
  and `PsdDominates`), and the theorems are stated in those names. Three composed statements were
  added that read entirely in that vocabulary: `van_trees_floor_from_cited_inputs`,
  `clustered_regret_floor_from_cited_inputs`, `action_regret_floor_from_cited_inputs`.

  Nothing is assumed that was not assumed before -- every definition unfolds to the inequality the
  statement already carried -- and nothing became an `Axiom`. That choice is the point: an `Axiom`
  would make `Print Assumptions` list the citations by name, which reads like the stronger audit
  and is the weaker formalisation, because it turns theorems that are true outright into theorems
  true only if our transcription of the cited result is, and it would put those files outside
  Stdlib's classical reals. So the audit command is `Check <theorem>` (the cited names appear in
  the type) and `Print <Name>` (what was assumed under it); `just assumptions` still reports every
  lemma resting on Stdlib's four axioms and nothing else.

  A `grep` for one predicate now enumerates every result that rests on that paper, across files,
  which a header comment cannot do.

- **`capped_exploration_policy` documented `n* = sqrt(K T/(A c))/cap` as if it were exact; it
  over-states the stopping mass by a CONSTANT.** Under a constant cap the stopping round is
  `n = S/cap`, so the remaining horizon is itself a function of the mass and the first-order
  condition is a *quadratic* in `w = I0 + c S`: `A w^2 + (K/cap) w = K c T + K I0/cap`. Subtracting
  the balance the closed form solves -- the same one with the horizon held at the full `T` --
  removes `T` entirely and leaves `A(w0^2 - w^2) = K c S/cap`. That is exact at every finite
  horizon, not asymptotic: the cap-free form is an **upper** bound on the mass, and the gap is at
  most `K/(2 A c cap)`.

  So what the closed form drops is a constant, not a vanishing remainder, and the constant is the
  only place the cap level enters the mass above `O(1/sqrt(T))`. At the defaults and `cap = 0.01`
  the ceiling is `2.0165`, approached strictly from below (`1.834, 1.959, 1.998, 2.011, 2.015` over
  `T = 1e4 .. 1e8`) with the residual decaying as
  `(K + 4 A I0 cap) sqrt(K/(A c)) / (8 A c cap^2 sqrt(T))`. It is the tight actuator this hurts: at
  `T = 4000` the over-statement is `0.50%` of the mass at `cap = 0.316` and `236.5%` at
  `cap = 0.001`.

  No numeric behaviour changed -- `predicted_mass` already solved the quadratic as a fixed point,
  which is why the docstring and the code disagreed silently. `capped_exploration_policy` and
  `_self_consistent_mass` now say what is actually computed and where the closed form stands
  relative to it.

  Derived in `validation/capped_exploration_schedule.mac` STEP 7, proved in
  `proofs/capped_exploration_schedule.v` section (H), confirmed to 60 digits over nine horizons by
  `validation/capped_exploration_o1.gp`, and the `T = 3` optimum independently certified by z3 and
  cvc5 in `validation/capped_exploration_t3.smt2`.

### Fixed

- **`certify_safety` raised on a safe plan that crossed a flat point of its barrier.** A smooth
  barrier has states where its gradient vanishes, such as the centre of a ball `1 - |x|^2`, and
  there the identification radius `Delta * |grad h|` is zero at every sensitivity level, so the
  step holds at every `Gamma` or at none. `barrier_gamma_star` treated the zero norm as invalid
  input and raised, so a plan resting at the centre of its safe set could not be audited at all.
  It now returns `inf` where the step's threshold radius is nonnegative and `nan` where it is
  negative, the answers its other cases already give; a negative norm still raises.

- **`prescribe` raised on a plan starting in the middle of a two-sided bound, and checked any tie
  against an average of its margins.** The facade audited `h = min_j` of the bound margins through
  `certify_safety`, which reads `h`'s gradient, and `jnp.min` averages tied gradients. The two
  margins of a two-sided bound tie at its midpoint with opposite ones, so there the gradient was
  zero, the step's check vacuous and `gamma_star` raised `ValueError`: a pendulum at rest inside
  `|omega| <= 1` could not be prescribed at all. Reading the first tied margin instead would still
  certify a state leaving fast through the other, so the audit now runs `certify_safety` on each
  margin as its own barrier and certifies a step only if every margin at the minimum there is, the
  right derivative of a minimum being the smallest of its tied terms'. Off a tie it is the old
  audit field for field, which a test pins, and no number `mmm_demo.py` prints moved. A tied step's
  `gamma_star` is the weakest tied margin's, an upper bound on what a single action reaches. A held
  solve still takes one barrier, whose gradient at a tie is now the first tied margin's rather than
  zero; `plan.safety` carries the per-margin audit, as the certificate does, and `max_levers` ranks
  its candidates under `hold_constraints` on that audit too. Ranked on the solve's own, greedy kept a
  lever whose plan left through the untested bound at the first step and whose certificate then
  cleared none of it.

- **`DecisionCertificate.trustworthy_steps` trusted a plan whose tube was never evaluated.** Since
  0.5.0 it dropped a `None` prefix instead of counting it as zero, which its own docstring promised,
  so a plan with no error bound took the barrier's prefix as its answer: the confounded arm of the
  marketing-mix case study, whose fit carries no channel error, reported 12 trustworthy steps. An
  unevaluated tube now contributes zero. A missing barrier still does not limit the answer, because
  `None` there means no state was bounded rather than that a check was skipped. `mmm_demo.py` also
  called its `+4.3%` "more cumulative sales"; it is lift, cumulative sales over the do-nothing arm,
  and the README's spine numbers now match what `spine_demo.py` prints (13.59, 38.97, 1.17).

- **Three accuracy statements in `exact_matrix_ratio_moment` and `MatrixRatioAccuracy` held only in
  part.** At `q = 2`, a correlated `Omega` (`rho = 0.5`) is not "a flat factor of about 11" over the
  exchangeable error: that holds up to margin 2 only, and at margins 4 and 8 the factor grows from
  `10.9` to `128` and from `15.9` to `202` between 20 and 40 nodes, so correlation also slows the
  convergence. At `q = 3` the per-node rates came from grids that stopped at 7 nodes for `n = 7`;
  taken to 9 (`just paper-2` in `causaldyn-bench`), the geometric-mean rates over 4-9 nodes are
  `1.60` at `n = 5` and `2.16` at `n = 7`, so six digits need 31 and 21 nodes, not `~20-32`. And
  the refinement residual majorises the error iff the rate reaches 2 only for an error that keeps its
  sign and its largest entry; in general residual / true error is at least `|rate - 1|` (reverse
  triangle inequality), so a rate of 2 always suffices, and at `n = 7` with 8 nodes the residual
  majorised at `1.04` with the rate at `1.77`. No numeric behaviour changed.

- **`dual_weighted_error_estimate`'s docstring divided by `den(T)` twice.** Since 0.4.0 it has
  written the identity as `S_hat(0) - S(0) = (eps - int z.g) / den(T)` with the normalised adjoint
  `z(s) = Phi(T-s)^T v / den(T)`, which puts `den(T)` under the interior term twice. The identity is
  `eps/den(T) - int z.g` with that `z` (`validation/mean_field_dwr.mac` STEP 3c); the code weights by
  `v^T Phi(T-s)` and divides once, so it was never wrong, and the certificate's `1.7%` is the code's
  number. The statement had joined STEP 2's un-normalised form to STEP 3's normalised weight. No
  numeric behaviour changed.

- **`panel_estimator_certificate`'s docstring quoted a table this routine does not produce.** The
  measured column came from an exploratory script that drew its panels from `jax.random.key(20000 +
  s)`; the routine draws from `key(seed + draw)` with `seed=20260904`, and nobody ran it at its
  defaults before the table was pasted in. It reproduces the quoted column exactly at `seed=20000`,
  so the code was never wrong -- the provenance was. At its own defaults the `phi = 0.9` verdict is
  unchanged in kind (every cell conservative, now by 17% to 31%, and on two further streams by 15%
  to 32%), but the `phi = 0.3` claims do not survive: at `g >= 6` the predicted reduction is 2% to
  7%, 300 draws cannot resolve it, and `ok` is False there on two streams of three. The docstring now
  carries the routine's own output and says so. No numeric behaviour changed.

- **`numpy>=2.0`, which is what the code already required.** `chc.deep_galerkin` calls
  `np.trapezoid`, added in numpy 2.0 under that name, while the floor said `>=1.26`. Nothing
  noticed because every resolution in practice pulls a jax that pins `numpy>=2.1`; a user who
  pinned an old jax would have got an `AttributeError` from a declared-supported configuration.
- **Two more stale copies of the version, found by looking for siblings of the one 0.5.1 fixed.**
  `CITATION.cff` and the README's BibTeX block both still said `0.3.0` -- so GitHub's "Cite this
  repository" and any paper citing this library were naming a version two minors old. Same cause as
  `__version__`: a hand-maintained duplicate of `pyproject.toml`'s `version`, agreeing with itself
  and therefore invisible.

  `tests/test_release_metadata.py` now pins all of them -- `CITATION.cff`, the README BibTeX, and
  `SECURITY.md`'s supported-versions row -- against `pyproject.toml`, which is the file a release
  commit edits. Confirmed each assertion fails when the stale value is put back.

- `CONTRIBUTING.md` said Python 3.12–3.14; the floor has been 3.11 since the lint matrix was
  widened.

- **A carryover-blind baseline for the marketing-mix case study, and what it measured.**
  `run_marketing_mix` gains a `myopic` arm: the *same identified fit* the `adjusted` arm plans from,
  spent on this week's return alone, at the same total budget. Its purpose is to separate what
  identification buys from what the objective's HORIZON buys, which the previous three arms could
  not distinguish -- `flat` is neither identified nor forward-looking, so `adjusted > flat` credited
  both at once.

  A rule constant in time is not an approximation of myopia on this plant, it is myopia's exact
  answer: the immediate increment `gamma_c * spend_c` is linear in spend, so "re-optimise every week
  for this week's sales" fills the highest-`gamma` channel to its ceiling and spills to the next,
  with the same answer every week.

  What it then measured is not what it was built to show. Over eight seeds (`causaldyn_bench`
  Track M), mean lift over doing nothing: `adjusted 46.56`, `myopic 46.88`, `flat 44.11`,
  `confounded 36.70`. Adjusting for the season is worth `+9.9` and wins at **8 of 8** seeds; looking
  past this week is worth `-0.3` with its sign flipping **5/3**, inside `6.2%` of the mean lift.
  Both identified rules beat the equal split at 8 of 8.

  And the design that flips it is one line: a myopic rule loses when the carryover ordering
  *contradicts* the immediate one, not merely because carryover exists. Here `beta_c/theta_c` ranks
  the channels `(1.29, 1.50, 1.60)` against `gamma_c`'s `(0.50, 0.20, 0.35)` -- the two disagree
  about the top channel but agree about which to drop. Re-parameterise to `beta/theta` of
  `(0.07, 8.00, 1.60)` at unchanged `gamma`, so the myopic ordering is untouched by construction,
  and the whole-horizon plan wins **6 of 6** by `2.11..3.40`. The module's honest scope now says so;
  the case study's headline is the identification, and the horizon is a second, smaller and
  plant-dependent effect.

- **Docstrings and docs pages pointed into a repository no reader can open.** Module and function
  docstrings cited `plans/<n>` and `discoveries/theorems.md`, paths in the author's research
  repository, which is not public, so the API reference sent a reader to nothing. A pointer that
  carried a result now names the public file that holds it -- a proof under `proofs/`, a derivation
  under `validation/`, a page of the site -- and one that only recorded where an idea came from is
  gone. `tests/test_docs_self_contained.py` fails on such a path anywhere under `src/chc/` or
  `docs/`, or in a notebook.

## [0.5.1] — 2026-09-09

### Fixed

- **`chc.__version__` was a literal, and it reported `0.3.0` from a 0.5.0 install.** It had been a
  second copy of `pyproject.toml`'s `version` since 0.3.0 and both releases since forgot it, which
  is what a second copy does. Caught by installing 0.5.0 from the index and asking the wheel what
  it was -- not by any check in the repo, since the literal agreed with itself everywhere.

  It matters past the cosmetic because of what reads it. `Provenance.chc_version` is the field that
  says which version produced the numbers beside it, and a caller reading `chc.__version__` to
  stamp their own artefacts was recording a version that did not. (`Provenance` itself was already
  correct: it has read package metadata since it was written.)

  Fixed at the producer -- `__version__` now comes from the same
  `chc.panel.installed_version()` the provenance uses, so there is one source of truth and it is
  the metadata `uv build` writes. `installed_version` is exported, since a caller stamping their own
  records should not have to reach for a private name. A test pins `chc.__version__` against
  `importlib.metadata.version`, and it fails when the literal is put back.

## [0.5.0] — 2026-09-09

### Added

- **The fit and the planner were using different integrators, and `prescribe` now does not.**
  `fit_causal_residual` reads the state rate off the log as a forward difference
  `(x_next - x)/dt`, while `causal_plan` and `rollout` integrate with RK4. Fitting one map and
  planning with the other is a deterministic bias, and not a small one: on a linear plant at
  `theta*dt = 0.7` the decay comes back `-0.502` against a true `-0.700`, and -- the number that
  actually steers the optimiser -- **the control channel comes back `0.574` against a true
  `0.800`**, a quarter of it gone. The gap is exactly the RK4 amplification
  `1 + z + z^2/2 + z^3/6 + z^4/24`, which is how it was identified rather than guessed at.

  New `integrator=` argument on `fit_causal_residual`. `"rk4"` closes the gap by **defect
  correction** -- fit, step the fitted field with RK4, add the leftover `(x_next - RK4(F))/dt` back
  onto the target rate, refit -- rather than by inverting RK4 in closed form, which does not exist
  for a nonlinear field. It recovers `-0.698` and `0.798` on the same log. It stops **per state**
  rather than on a pooled criterion, which was the one real subtlety: a state the model cannot
  represent has a defect floor orders of magnitude above one it can, and pooling let the
  marketing-mix plant's seasonally-driven sales row halt the adstock rows at half their remaining
  gap. Every solve downstream of the target rate is separable across output columns, so freezing one
  column's target while another keeps moving is well defined.

  It is not free. The corrected target carries the defect's noise, so the channel's standard error
  grows -- `0.016` to `0.061` on the `chc.mmm` plant. The bias removed was deterministic and the
  variance added is not, which is the trade, stated rather than hidden.

  **Defaults differ on purpose.** `fit_causal_residual` keeps `"euler"`, so no shipped fit changes
  meaning and a caller whose panel is genuinely discrete-time (a weekly budget is not a sample of an
  ODE) keeps the one-step map as the model. `prescribe` defaults to `"rk4"`, because it *knows*
  where the field is going. `CausalDynamicsFit` gained `integrator` and `integrator_defect`, the
  latter being the RMS one-step defect under whichever map was asked for -- the honest check that
  the fixed point converged, and it floors at the observation noise rather than at zero.

  The `chc.mmm` headline numbers moved and its conclusion did not: adjusted still beats a
  matched-budget flat split, now by 4.4 / 4.3 / 7.5 / 8.0% across four seeds against
  4.3 / 4.8 / 7.7 / 8.3 before. That is the expected shape, since every arm there is audited on the
  true plant rather than on the planner's own forecast -- which is exactly what the module's honest
  scope note said the gap did and did not touch.

- **`prescribe` now says what it is doing while it does it, and says failure in a type.** Each
  decision point emits one stdlib `logging` record on `chc.decision`, keyed by `chc_event` --
  `precision`, `adjustment`, `fit`, `abort`, `plan`, `certificate` -- with the numbers a downstream
  JSON handler would want as `extra` fields (the fitting method, the identification radius, the
  overlap, the solver status and iteration count, `Gamma*`, the certified horizon, and the wall
  time of the fit and of the solve). No handler is installed and no level is set: a library that
  calls `basicConfig` takes a decision that belongs to the application.

  Two records are `WARNING` rather than `INFO`, being the two worth waking someone for: a panel
  built without `JAX_ENABLE_X64=1`, and a graph under which no observed set identifies the effect.
  The precision one is a **warning and not a refusal** -- JAX is single-precision by default, the
  harm is plant-specific, and `Provenance.x64` already travels with every result -- so a caller who
  needs the guarantee asserts on the provenance rather than having a default chosen for them.

  New `DecisionError(ValueError)` and `NotIdentifiedError(DecisionError)` replace the bare
  `ValueError`s. Both remain `ValueError` subclasses, so nothing that caught the old type stops
  catching them; what they buy is that a caller falling back to `chc.sensitivity`'s partial-
  identification path can trigger on the *one* failure this library exists to produce, instead of
  matching on a message. A column that is simply absent still raises `KeyError`, matching `Panel`
  lookup: a missing name is a lookup failure, not a bad decision.

- **`tests/test_decision_properties.py`: one hypothesis invariant per layer of the facade.**
  Projection idempotence and box membership; the §40 certified prefix monotone in the assumed
  `Gamma`; the schedule inside the lever band over random boxes; `to_json` surviving `json.dumps`;
  the panel fingerprint blind to dictionary order and not to a changed value; d-separation symmetric
  over random DAGs; and the canonical adjustment set never containing the outcome or a descendant of
  the treatment.

  That last one is stated on the *output* rather than by asking `is_valid_adjustment_set`, and the
  difference was measured, not assumed: both methods read the same `forb`, so a consistency check
  between them is a tautology. Dropping `Y` from `forb` -- the exact pre-release defect a
  brute-force cross-check caught before 0.5.0 -- leaves the consistency property green and fails
  the output property. Every property here was confirmed to fail under a mutation of the code it
  claims to constrain.

- **`chc.mmm`: marketing-mix budget scheduling, the case study `prescribe` was built for.** A
  saturating carryover plant where the confounding is not hypothetical -- media spend is planned
  *against demand*, so a model fitted on the log credits the channel with the season.

  ```
  | arm        | total spend | cumulative sales |    lift | lift / extra spend |
  | adjusted   |      41.677 |           84.048 | +42.984 |            +1.0779 |
  | confounded |      35.710 |           78.739 | +37.674 |            +1.1110 |
  | flat       |      41.677 |           82.257 | +41.193 |            +1.0330 |
  | none       |       1.800 |           41.064 |  +0.000 |                nan |
  ```

  Three readings of one log by the same optimiser. **At matched budget the prescribed schedule buys
  4.3% more cumulative sales than an equal split** (4.4 / 4.3 / 7.5 / 8.0% over four seeds), by
  front-loading to build carryover and then tapering. The **confounded** arm -- an empty adjustment
  set asserted, which is what fitting the log directly amounts to -- credits every channel with the
  season and inflates them unevenly (`2.23x`, `4.84x`, `3.04x`), so it believes it needs less
  budget, spends 14% less and buys 88% of the lift (0.78-0.88 over four seeds).

  Two things the module exists to say out loud. **Return-per-unit-spend rewards under-investment**
  under diminishing returns: the confounded arm looks *better* on it (`1.111` against `1.078`) while
  buying less, which is why the arms are compared at matched budget and a test pins that inversion.
  And **cumulative sales, not terminal sales, is the metric**: an optimiser that understands
  carryover front-loads and tapers, which raises the area under the curve and lowers the endpoint.
  The first draft of this module scored on the endpoint and duly ranked the flat split first.

  The plant is control-affine by construction, with each channel acting twice: an immediate
  incremental return `gamma_c` that is the **control channel** and is what gets confounded, and a
  carried-over return `beta_c` through a saturating adstock that is part of the **drift**. The
  adstock rows are mechanical and are handed to `prescribe` as `known=`, so the fit has only the
  sales row to learn. Every arm is audited by rolling its schedule out on the *true* plant, as
  `chc.spine` does, rather than on the planner's own forecast.

  **A library finding fell out of the `known=` check.** `fit_causal_residual` reads the state rate
  as a forward difference `(x_next - x)/dt` while `causal_plan` rolls out with RK4, so at a coarse
  `dt` the residual silently absorbs the gap between the two integrators. It is not small: at
  `theta*dt = 0.7` the fitted decay is `-0.503` against the `-0.7` handed over as known, a 28%
  discrepancy that is pure discretisation. The test asserts the residual on those rows equals the
  closed-form RK4 amplification gap rather than asserting it is zero -- which is both honest and a
  stronger check, since anything else leaking into a "known" row now fails it.

- **`chc.prescribe` runs the whole chain in one call** (`chc.decision`). Panel of logs in,
  certified intervention schedule out: adjustment set from the graph, control channel from
  cross-fit Robinson DML, constrained plan with a Gronwall tube, barrier priced against
  confounding. No new estimator, no new solver, no new guarantee -- what did not exist was a way to
  run the existing layers in that order without re-deriving the wiring, so every demo was five
  imports and forty lines.

  ```python
  panel = Panel.from_frame(logs, unit="region", time="week")
  graph = CausalGraph.from_edges([("demand", "incentive"), ("demand", "supply"), ...])
  out = prescribe(
      panel,
      levers=[Lever("incentive", -2, 2, unit_cost=0.05)],
      target=Target("supply", 1.0),
      constraints=[Constraint("wait", hi=0.5)],
      adjustment=graph,
      horizon=15,
      dt=0.1,
      tolerance=0.5,
  )
  print(out.report())
  ```

  **The causal assumption is a required argument.** `adjustment=` takes either a `CausalGraph`, from
  which the set is *derived* and can come back `not_identified`, or a sequence of names, which
  *asserts* it. There is no default, because the default would be "adjust for nothing" -- a causal
  claim, not the absence of one.

  **Two axes, never merged.** `DecisionCertificate` reports identification (about the data and the
  graph) separately from certification (about model error and the barrier), following
  `CausalPlan.certificate_status` vs `solver_status`. A plan fully certified over a channel nothing
  identifies is a trustworthy tube around a meaningless action, and `trustworthy_steps` is the one
  number that respects both -- zero whenever the effect is not identified, whatever the tube says.
  An unidentified effect produces **no schedule at all**: `Prescription.schedule` raises, as
  `CausalPlan.certified_actions` already does, rather than hand back actions that look like every
  other schedule and mean nothing.

  **Omitting `tolerance=` switches the tube off** instead of setting it to infinity. This library
  cannot know how much trajectory error a caller accepts, and a certificate at infinite tolerance
  passes over the whole horizon while proving nothing.

  `Prescription.report()` is Markdown from string templates -- no plotting dependency, so the
  output is deterministic and a test asserts on it; `to_json()` is schema-versioned; `reach()`
  ranks levers by fitted channel **times box width**, because a large coefficient on a lever that
  may barely move is not a large lever. Ranked from the same fit that produced the plan, so the
  ordering cannot contradict the schedule printed beside it.

  The three-arm test is the load-bearing one: the same logs read three ways. Adjusted for the
  confounder the channel comes back `0.81` against a true `0.80`; with an empty adjustment
  *asserted* it comes back `2.09`; with the confounder declared latent there is no schedule.

  Two deviations from the plan's specification, both deliberate. `Lever.cap_per_step` is **not**
  shipped: the solver constrains a box, not a rate, so the field would have been silently ignored,
  which is worse than absent -- it waits on general constraints (Dykstra). And
  `DecisionCertificate` carries no `regret_bound`, because nothing in `chc.regret` takes a
  `CausalPlan` and a fabricated one would be the only uncertified number on a page of certified
  ones.

  The module is `chc.decision`, not `chc.prescribe`: the headline export is a *function* named
  `prescribe`, and a same-named module inside the same package is shadowed by it -- after
  `import chc.prescribe`, `chc.prescribe.Lever` raises `AttributeError`. It was the library's only
  such collision, and a check for the class is now trivial (`iter_modules` against `__all__`).

- **`chc.graph.CausalGraph` derives the adjustment set instead of asking the caller to type it.**
  Every effect estimator here takes `covariates=("x", "z")` -- a causal claim entered by hand, whose
  two failure modes are silent. Adjusting for a collider *opens* a path that was closed (M-bias);
  adjusting for a mediator removes part of the effect being estimated. Neither raises, neither fits
  worse, and both change the number.

  `CausalGraph.adjustment_set(treatment=..., outcome=...)` returns the canonical set of Perkovic,
  Textor, Kalisch and Maathuis (2018), `Adjust(X, Y, O) = (an(X u Y) n O) \ forb(X, Y)`, which is
  valid **iff any set of observed variables is** -- so one construction answers both questions, and
  a failed check is a proof that nothing else would have worked either. The result is an
  `AdjustmentSet` carrying `status` beside `covariates`, because an empty tuple means two opposite
  things: *identified, nothing to adjust for* (a randomised lever) and *not identified* (confounded
  through a latent). `__bool__` reads the status, not the length, so the first is truthy.
  `not_identified` names the latents whose measurement alone would flip the verdict -- the blame a
  caller can act on, rather than the longer list of latents that merely sit on some open path.

  Also `d_separated` (Koller-Friedman Bayes-ball), `is_valid_adjustment_set`, `parents` /
  `children` / `ancestors` / `descendants`, and `require_columns`, which names every missing column
  at once. Treatment and outcome accept a name or a sequence of names, since the multi-lever case
  is the one this library exists for. Pure Python over names and sets -- no NumPy, no JAX.

  **Verified against implementations that share no code with it.** `d_separated` is checked on
  random DAGs against textbook path enumeration (16 808 cases in the scratch sweep, > 500 kept in
  CI), and the canonical set against exhaustive search over every subset of the observed nodes
  (2 697 cases, both branches exercised). That sweep found one real defect before release: with no
  causal path from `X` to `Y`, `cn(X, Y)` is empty, so the literature's `forb = de(cn) u X` does not
  contain `Y`, and the construction offered to adjust for the *outcome* -- which d-separates `X`
  from `Y` trivially and certifies nothing. `forb` here is `de(cn(X, Y)) u X u Y`.

- **`chc.panel.Panel` checks a long panel's `(unit, time)` index once, and pivots it once.**
  The estimators want panel data in two incompatible shapes -- a dense `(n_units, n_periods)` matrix
  for `chc.did` and `chc.scm`, flat named columns everywhere else -- and callers have been pivoting
  between them by hand. That pivot is where the quiet failures live: a duplicated `(unit, time)` row
  keeps whichever value was written last, a missing one becomes a zero, and neither surfaces until
  an estimate is already in a slide. `Panel.from_frame(data, unit=..., time=...)` refuses all three
  and names the column *and* the entity: which unit appeared twice at which period, which unit has
  no row at which period, which column is `nan` for whom. `wide(name)` is the checked pivot;
  `codes()` ranks labels rather than assuming they are `0..T-1`; `cluster=` is declared once, where
  it belongs, since it is a property of the sampling design and not of the estimator.

  `Provenance` travels with the panel: a sha256 over the column bytes, the library version, the row
  count, the column names, an optional `seed=` for simulated data, and
  **`jax.config.jax_enable_x64`**. The last is not decoration -- this
  library has already been bitten by it, since a threefry key spends a different number of bits per
  element at the two settings and therefore draws a *different* sample from the same seed. A seed
  alone does not name a dataset. Two panels agreeing numerically but differing in dtype hash
  differently, which is the honest answer, because they will not produce the same numbers.

  The private `Panel` type alias in `chc.did` and `chc.scm` -- a wide `NDArray[float64]`, never
  exported -- is renamed `Outcomes`, so one name means one thing.

- **`exact_matrix_ratio_moment` works at any channel count the route allows, not just two**
  (`chc.regret.exact_matrix_ratio_moment`). The channel count is read off `regressor_cov`'s shape
  (`q = regressor_cov.shape[0] // n`), so the signature is unchanged for existing two-channel
  callers and `nodes` now defaults per `q` (40 at `q = 2`, 6 at `q = 3`).

  The enabling identity is that Isserlis' pairings are a sum over the symmetric group,
  `E[prod_i z'K_i z] = sum_{sigma in S_m} 2^(m - c(sigma)) prod_{cycles} tr(...)`, whose weights
  sum to `(2m-1)!!`. That retires the hand-written three-form table rather than generalising it --
  one definition covers every `q`, it reproduces the shipped `q = 2` numbers to `2.3e-15`, and it
  is *faster* than the special case it replaced, because prefix sharing turns 25 920 word
  evaluations at `q = 3` into 3 336 batched gemms.

  Two limits, with different causes, both enforced. The Ingham-Siegel route needs `s = 2` at every
  `q` while `Gamma_q(s)` converges only for `s > (q-1)/2`, so the ROUTE ends at `q <= 4` --
  structural, and no amount of compute buys past it. The plan ends one step earlier, at `q = 3`,
  because `q = 4` would need `5760 x 5040 = 29` million terms.

  Accuracy is reported, not assumed, and at `q = 3` what it reports is a percent, not six digits.
  The cone is six-dimensional, and the grid was taken as far as it will go against the Wishart
  anchor `E[M^-1] = I/(n-q-1)`:

  | nodes | points | abs err, `n = 5` | abs err, `n = 7` |
  |---|---|---|---|
  | 4 | 4 096 | 2.92e-1 | 9.55e-2 |
  | 5 | 15 625 | 4.68e-2 | 5.67e-2 |
  | 6 | 46 656 | 8.66e-2 | 1.87e-2 |
  | 7 | 117 649 | 6.69e-2 | 1.12e-2 |
  | 8 | 262 144 | 4.64e-2 | -- |
  | 9 | 531 441 | 2.76e-2 | -- |

  **1.6 digits for 531 441 points.** The measured rate is `1.6x` per node at `n = q + 2` and `2x`
  at `n = q + 4`, so six digits would need `nodes ~ 20-32`, i.e. `6e7` to `1e9` points: not
  expensive, unreachable. An earlier draft of this entry read the two-point `4.2-6.2x` slope as a
  rate and predicted six digits at `nodes ~ 11-13`; the full curve withdraws that. The `q = 3`
  default is 6 rather than 8 for the same reason -- at `n = 5`, `nodes = 8` evaluates 17x the
  points of `nodes = 5` and is no more accurate (`4.64e-2` against `4.68e-2`). Wall-clock figures
  that appeared in an earlier draft are withdrawn: the same cell measured 3218 s and 1499 s on the
  same machine depending on load, while the accuracy column is bit-identical across both runs.

  Convergence is monotone only with margin from the existence boundary: at `n = q + 2` the
  sequence goes `4.68e-2, 8.66e-2, 6.69e-2, 4.64e-2, 2.76e-2`.

  Seven ways out were measured and all seven lost. Four rules -- a Smolyak sparse grid on nested
  open Fejer-2, per-axis Cholesky scaling, Aitken extrapolation over three grids, and whitening
  the cone by `E[M]^-1/2`. Three diagnosable mechanisms, each of which would have had a fix, were
  ruled out by measurement: ill-conditioning (`cond(tilt) <= 1038` over the whole grid), a peak
  the grid straddles (the top node carries under 1.6% of the mass), and a truncated tail
  (enlarging the domain strictly hurts -- `4.7e-2 -> 7.8e-2 -> 3.1e-1 -> 5.7` at 1x, 2x, 5x, 10x
  the probe scale). What is left is ordinary slow convergence in six dimensions with no lever
  attached, and the docstring says so rather than quoting a convergence the tool does not have.

  An eighth way out, randomised QMC, lost too -- and its failure was mis-diagnosed once before it
  was measured. Scrambled Sobol at `q = 3, n = 5` beats the grid at 4 096 points (`1.5e-1` against
  `2.9e-1`), loses at 16 384, and at 65 536 has a replicate spread larger than its error. An
  earlier draft blamed the half-line map's Jacobian at the cube boundary; measured, the transformed
  integrand is nonnegative, its tail exponent is at least 4 on every ray and the dominant points
  are interior. What they share is a `T` within a few percent of rank one (the three largest carry
  `eig(T) = [350, 0.20, 0.10]`, `[255, 0.86, 0.03]`, `[83, 0.97, 0.11]`), a corner of measure
  `~3e-5` under the Cholesky map, so per-point contributions have kurtosis `1 300-1 900` and one
  point can carry 10% of the sum. The grid sees it from the other side: at `nodes = 5` the three
  outermost `l00` nodes carry 99% of the `(0,0)` entry. In spectral coordinates `T = Q Lambda Q'`
  the kurtosis is `6.6` against `128` at `n = q + 4` but stays near `940` at `n = q + 2`. The two
  cells fall either side of a moment condition: for `B > 0` the sandwich lies between
  `lambda_min(B) M^-1` and `lambda_max(B) M^-1`, so it is square-integrable under the Wishart law
  iff `E[W^-2]` is, and von Rosen's `E[W^-2] = (n-1) I / ((n-q)(n-q-1)(n-q-3))` poles exactly at
  `n = q + 3` -- one step past the `n = q + 1` pole of `E[W^-1]`. Both checked: Maxima identity 28
  verifies the pole for `q = 1..5` and integrates the exact `q = 1` case `1/((n-2)(n-4))` at
  `n = 5..10`, and Monte Carlo at `q = 2, 3, 4` matches the formula to `0.06-0.43%` at margin `+3`.
  That rules out sample-and-average on the
  boundary cell; it does not rule out every map, and the tensor rule converges there regardless, so
  what is bounded is the rescue rather than the integral.

  **A second new anchor, with CORRELATED channels.** For `numerator = denominator = I` and
  `regressor_cov = kron(R, I_n)`, `vec(X) ~ N(0, R (x) I_n)` gives `X'X ~ Wishart_q(n, R)`, so the
  sandwich collapses to `E[(X'X)^-1] = R^-1/(n - q - 1)`. It is the first anchor here whose answer
  is NOT isotropic -- at `q = 2` the off-diagonal is `-r/((1-r^2)(n-3))`, which a rule blind to the
  channel correlation could not produce -- and a 4e6-draw Monte Carlo confirms it to 2.5e-4. It
  generalises: for `regressor_cov = kron(R, S)` and `denominator = S^-1` the answer is
  `(tr(BS)/n) R^-1/(n - q - 1)` for any PSD numerator `B`, which contains every anchor this
  function has. No matrix square root is needed to see it -- factor `S = FF'` and `R = GG'` for any
  invertible factors, and `X = F Z G'` for a standard `Z` makes the denominator `G (Z'Z) G'` (`S`
  cancels entirely) and the numerator `G Z' (F'BF) Z G'`, so the `Omega = I` anchor applies to the
  bracket and `tr(F'BF) = tr(BS)`. Checked symbolically (Maxima identity 30, on symbolic `Z`, `F`,
  `G` and `B`), formally (`proofs/general_q_ratio_moment.v` section (G): the pole is the same
  `E[W^-1]` pole, the Wishart constant is the degeneration, and the three-way scaling law
  `beta*sigma/rho`), and numerically -- at `q = 2, n = 6` the rule matches it to `2.5e-7` relative
  while every plausible variant of the formula (`tr(B)` for `tr(BS)`, `tr(B)tr(S)/n^2`, `R` for
  `R^-1`) is off by more than `0.1`.

  The same factorisation makes a Kronecker `regressor_cov` **reducible** -- run the isotropic
  problem on the congruences and conjugate back -- and that was proposed as a fast path. It is
  **not shipping.** Measured against the direct route the accuracy ratios over three cells are
  `1.82x`, `1.49x` and `0.65x`: the reduction loses on one of them, so the branch would move the
  answer's error by less than a factor of two in an unpredictable direction while adding a second
  implementation to keep in sync. The mechanism proposed for the expected win was wrong too --
  conjugating by `R^-1/2` was supposed to amplify the error by `cond(R)` and amplifies it by
  `0.90-0.94x`. The identity ships as an exact test and a lemma; the branch does not.

  **What the new anchor then corrected was three claims made from it in a first draft, all wrong
  for one reason: they compared numbers computed in different normalisations.** The `n = 7` column
  of the accuracy table is an ABSOLUTE max-entry error while the `n = 5` column is both, and the
  correlated cell had been quoted as a single entry's relative error. Recomputed in one convention
  (max entry, relative):

  - the correlated cell is 1.6x and 1.4x worse than the exchangeable one on the two coarse grids,
    0.63x on the third and 1.3x on the fourth -- not "2x worse at every size";
  - `matrix_ratio_certificate`'s residual / true error there is 2.42, 5.80, 1.44 at
    `nodes = 5, 6, 7` -- CONSERVATIVE, and more so than on the exchangeable cell (2.06, 3.64,
    0.78). The claim that it understated by 41x is withdrawn;
  - and what governs the `q = 2` accuracy is the MARGIN from the existence boundary, not the shape
    of `B` or `Om`. At `nodes = 40` the relative error is 7.34e-6, 4.59e-8, 3.35e-11, 4.23e-13 at
    `n = 5, 6, 8, 12`, and an anisotropic `B` at `n = 6` reproduces the isotropic numbers exactly.
    Both earlier qualifiers -- "machine precision only when the numerator is a multiple of the
    identity", then "only on isotropic cells" -- are withdrawn in favour of the margin.

  The repair named above was built and closed on its budget rather than shipped. A tensor rule in
  spectral coordinates -- ordered eigenvalues through a gap parameterisation, so the Vandermonde is
  a polynomial rather than an absolute value, and an `SO(3)` product rule that is EXACT on 27 nodes
  whenever `regressor_cov = I` -- is 31.8x more accurate than the shipped grid at the matched
  46 656 points on `n = q + 4` (both sides the `(0,0)` entry's relative error). It still misses six digits (3.63e-4 at 74 088 points), does not win at all
  on the existence boundary, and is only 7x better on a correlated cell where both rules are
  erratic. A 7x constant does not pay for a second coordinate system and its Haar quadrature.

  A new exact anchor with an ANISOTROPIC numerator: for `Omega = I`, `C = I` and any PSD `B`,
  `E[M^-1 X'BX M^-1] = (tr B / n) I / (n - q - 1)` (polar decomposition `X = HT`, `H` Haar and
  independent of `T`, `E[H'BH] = (tr B / n) I`; Maxima identities 25-27). It is the first check of
  the general code with a numerator that is neither the identity nor a projection, and it
  the first cell where `B` can be varied against a known answer at all. With a random
  positive-definite `B` the rule lands at `7.7e-6, 1.2e-7, 1.1e-8, 2.0e-9` for
  `nodes = 20, 40, 60, 80`. Those are ABSOLUTE errors at `n = 7`; recomputed as relative errors
  they match what `B = I` gives on the same cell, so the reading of them as "algebraic off
  `B` proportional to the identity" is withdrawn -- see the normalisation note above. What governs
  the `q = 2` accuracy is the margin, and `B` moves only the constant.

  The error bar comes free where it matters: on an exchangeable problem the exact answer is
  isotropic, so half the observed spread of the diagonal lower-bounds the largest entry error with
  no reference value in hand -- measured at 77% and 99% of the true error. It is a necessary
  condition only: a grid can be isotropic and uniformly wrong.

- **`matrix_ratio_certificate` -- the same value, plus what the grid is worth**
  (`chc.regret.matrix_ratio_certificate`, `MatrixRatioAccuracy`). At `q = 3` the returned array
  carries a percent-scale quadrature error and reveals nothing about it, and the isotropy bar above
  only exists on exchangeable problems -- precisely not the anisotropic ones, where convergence is
  worst. This attaches the grid-refinement residual `max|X(nodes) - X(nodes-1)|`, which works on any
  problem. Cost is one coarser grid, about 26% at `nodes = 5`.

  It is an ESTIMATE, not a bound, and exactly when it fails is now known rather than tabulated. A
  refinement residual measures the STEP, not the remainder: for a monotone same-signed error
  sequence it is identically `e(k-1) - e(k)`, confirmed to `5e-5` relative against the exact
  anchor. So `residual / true = r - 1` with `r` the per-node decay, and the residual majorises the
  error **iff `r >= 2`**. Below that it under-states by construction. The measured `q = 3` rates
  are `1.30, 1.44, 1.68` and the measured ratios are `0.30, 0.44, 0.68` -- `r - 1` to two decimals.
  Over 13 cells the ratio ranges `0.30x` to `5.26x` and falls below 1 in five, each one a cell
  whose rate is under 2. An earlier draft of this entry claimed "conservative in 6 of 6 cells";
  that is withdrawn.

  A three-grid rate estimator was designed and rejected on measurement. Under geometric decay
  `residual(k-1)/residual(k)` would equal `r`, making the certificate self-diagnosing for about 4%
  extra cost; measured, the residual ratios are flat (`0.96, 1.09`) while the error ratios are
  `1.44` and `1.68`. The error sequence is not geometric, so no Richardson-type correction exists
  and none was shipped.

  The certificate now carries a SECOND bar where one exists: `exchangeable`, `isotropy_bar` and
  `relative_isotropy_bar`. On an exchangeable problem the exact answer has a constant diagonal, so
  half the observed diagonal spread is a **proved** lower bound on the largest entry error -- and
  it costs nothing, being a function of the returned matrix. The precondition is decided rather
  than assumed: `Omega` must be invariant under permuting the `q` channels of `vec(X)`, which
  `q - 1` transposition checks settle, so a channel-asymmetric problem gets `nan` instead of a
  wrong number. `ok` now requires both bars, which can only tighten it.

  What it is worth, measured at `q = 3, n = 5` against the exact anchor: bar/true error is
  `0.99, 0.55, 0.45, 0.46, 0.48` at `nodes = 5..9`, against the residual's
  `5.26, 0.86, 0.30, 0.44, 0.68`. The bar is far steadier -- 0.45-0.55 past the coarsest grid,
  where the residual swings 17x. It does NOT rescue the cells the residual missed: at a 1%
  tolerance both convict all of them, so this is a safety net, not the fix. Nor is the steadiness
  a calibration -- five cells of one problem family is the evidence base that produced the
  retracted rate claim above.

- **The delayed-network panel leaves the cycle, and the network estimator reports uncertainty**
  (`chc.network_causal.torus_adjacency`, `DelayedNetworkPanel(graph=...)`,
  `estimate_network_effects(exclude_neighbours=...)` and its new `direct_se` / `spillover_se`).
  `DelayedNetworkPanel` was cycles-only, so Result 52's design law could only ever be tested on the
  one topology it has a closed form for. It now takes any **regular** adjacency as nested tuples --
  regular because the `neighbours` column is rectangular and a varying degree would need a sentinel
  every consumer would have to know about. `graph=None` keeps the cycle and a byte-identical draw.

  `estimate_network_effects` returns **cluster-robust** influence-function standard errors,
  clustered on `cid`. Which coefficient the correction reaches is decided by which regressor is
  smooth: measured clustered / i.i.d. is `0.93` on the direct effect and `1.87` on the spillover,
  because the exposure is a shell sum while the treatment's exogenous part is i.i.d. across units.
  One i.i.d. SE for both would understate exactly the coefficient interference is about.

  `exclude_neighbours=True` is the Emmenegger-style baseline -- drop every training row whose unit
  neighbours a test unit. It is off by default and it raises rather than fitting on nothing when
  the test fold's hop-1 neighbourhood covers the training fold, which is what happens at `K = 2` on
  a `3x4` torus or a random cubic graph. Measured against it (`causaldyn-bench` Track N, `C_12`,
  `g = 2`, 120 draws): exclusion costs **+66%** MSE, contiguous graph blocks **+37%**, and the
  Result 52 design ties the graph-blind unit split. The law's own mass ratio -- `0.720` cycle,
  `0.974` torus, `0.966` cubic -- forecasts which topology has a design effect at all, and is right
  on all three.

- **The fold design is a MAXIMUM cut, and the exact optimum is now reachable at any `m`**
  (`chc.regret.fold_exactness_certificate`, `FoldExactnessCurve`; `optimal_fold_partition` gained
  `FoldDesign.route` and `banded_exact=`). Two things.

  **A naming correction with teeth.** Result 52 reduced the cross-fit variance to the same-fold
  2-walk mass `sum_f 1_f' Q 1_f` and called minimising it "the minimum-weight balanced cut". That
  mass is `1'Q1 - 2*cut` with `1'Q1` partition-free, so minimising it is **maximising** the cut of
  `Q(x)`: the problem is a max-bisection. The distinction is not cosmetic -- max-bisection has a
  constant-factor SDP approximation and min-bisection has none. Every formula and Rocq lemma of
  Result 52 was and remains correct; only the sentence drawn from them was inverted. What made the
  wrong name feel right is that `Q`'s edges are 2-WALKS: the optimum keeps *adjacent* units
  together (adjacency cut 6 of 12 on `C_12`, against parity's 12).

  **An exact route past the enumeration limit.** On a circulant the objective collapses to circular
  offset counts and `Q(x)` has bandwidth `2*dmax` -- set by the spillover truncation, never by `m` --
  so a dynamic program over `(window of B labels, running balance)` returns the global optimum in
  `O(4^B m^2)`: **1.0 s at `m = 240`**, where enumeration would walk `~1e70` designs. It reproduces
  enumeration to `2.8e-14` on all fifteen cells of `m in {10..18}`, and it refuses `m <= 2*band`,
  where the offsets `b` and `m-b` name the same pair and the collapse double-counts.

  That separates two numbers the previous certificate conflated. Past the enumeration limit the
  spectral-plus-swap fallback is exactly optimal in **one cell of twelve** (shortfall `0.00-2.06%`
  over `m in {24, 60, 120, 240}`), so the earlier "it finds the optimum on all eighteen instances"
  was a small-`m` fact read as a general one; and the Ky Fan bound's own looseness is `0.58-2.61%`
  and **flat in `m`**. What `FoldDesign.gap` reports is the two stacked, so it always dominates the
  true shortfall: `eps` certified implies `eps`-optimal, while a large gap convicts nothing.

  Cross-checked outside Python: polymake 4.15 maximises the same objective by exact rational LP over
  the 35 balanced cut vectors at `m = 8` and returns `3376/125`; Normaliz 3.11.1 counts the
  hypersimplex's lattice points at `m = 8, 10, 12, 14` as `C(m, m/2)`, all 0/1.

- **Result 51's `Psi` measured against a real cross-fit estimator** (`chc.regret.
  panel_estimator_certificate`, `PanelEstimatorGate`). Result 51 shipped `Psi` saying in its own
  scope note that it is a functional of the process, "not a re-derived estimator". This runs the
  comparison. The partition enters `Psi` once, linearly, with positive weight, and with a
  block-diagonal covariance over `g` independent clusters every partition-free factor cancels from
  the RATIO, leaving a Mobius function of the cluster count with `|ratio - 1| <= C/g`. **The design
  law is a finite-cluster statement**: more data as more clusters erases it, more data as longer
  panels or larger clusters does not.

  Against a ridge-polynomial DML fit over 300 draws (`m=12`, `p=12`, `K=2`): six cells of six agree
  on the sign, both predicted and measured wash out toward 1 with `g`, the functional is
  **conservative in every cell** -- the estimator gains 5-23% more from the good partition than
  `Psi` predicts -- and the bootstrap interval covers the prediction in **two cells of six**. So
  `Psi` is a ranking rule, not a point predictor of an estimator's variance ratio. The certificate
  gates the sign, the washout and the conservatism, and **counts coverage without gating on it**.

- **Folds on both axes of a delayed-network panel** (`chc.network_causal.panel_covariance`,
  `kronecker_spectrum`; `chc.regret.optimal_fold_partition(time_axis=...)`,
  `space_time_fold_certificate`, `SpaceTimeFoldCurve`). Result 51 showed the delayed-network
  covariance is separable only at `delta = 0`; Result 52 designed folds on the space axis and said
  so. The two-axis problem is not intractable. Writing `Sigma = sum_q P_q (x) T_q` with
  `P_{-q} = P_q'` and `T_{-q} = T_q'`, each `+-q` pair splits into a symmetric and an antisymmetric
  channel, so the **Kronecker rank is at most `2*dmax`** -- set by the spillover truncation, never
  by the graph diameter or the panel length -- and never `2*dmax+1`, because the `q = 0` and
  `q = dmax` factors are symmetric on every graph. Exactly `dmax+1` when the shell operators commute
  (cycle: `2, 3, 4, 5`) against `2*dmax` when they do not (path: `2, 4, 6, 8`), and `1` at
  `delta = 0`. Reproduced independently in Octave (`validation/space_time_folds.m`) and verified
  against 2e5 simulated panels.

  Two consequences for the design. A panel too short to resolve a shift cannot see the coupling it
  is designing for: once `delta*q` passes `p-1` the temporal factors become proportional and the
  rank saturates below the law (4 against 5, 7 against 8). And **two different mistakes cross over**
  -- scoring a time-constant fold with the cross-sectional weight costs `3.3, 8.7, 2.8, 0.3%` over
  `delta = 1..4`, while freezing the time axis costs `0.3, 3.0, 13.3, 19.5%`. At short delay the
  mistake to fix is the weight; at long delay it is the axis. `time_axis=None` is the default and
  reproduces the previous behaviour exactly.

- **Gamma is unfalsifiable, but it is not uncalibrated** (`chc.uncertainty.benchmark_gamma`,
  `negative_control_gamma`, `gamma_benchmark_certificate`, `GammaBenchmark`,
  `GammaBenchmarkCertificate`). Result 32 ships `Gamma` as the analyst's input and says so at every
  use. Two calibrations turn it into a number that can be argued about. **Benchmarking** drops an
  observed covariate from the propensity: the two fits differ by exactly the kind of odds ratio the
  MSM bounds, so `Gamma_j = exp(quantile_i |logit e(x_i) - logit e_{-j}(x_i)|)` is the sensitivity a
  confounder as strong as covariate `j` would generate (Cinelli-Hazlett, in MSM units).
  **Negative-control calibration** inverts a known-null outcome for the smallest `Gamma` that
  reconciles it -- a *lower bound* on the confounding actually present, so assuming less is refuted
  by the data rather than merely unappealing, and `inf` when the sample lies wholly on one side of
  zero and the model class is refuted instead.

  `multiples_of_strongest` is `log(Gamma)/log(Gamma_strongest)`, an **exponent**: odds ratios compose
  multiplicatively, so a confounder twice as strong as the benchmark is `Gamma_s^2`, not `2*Gamma_s`.
  The two readings coincide at `Gamma_s = 2` and nowhere else above 1 (Rocq
  `linear_scale_coincides_once`).

  Two measurements changed the design. The MSM's own statistic is the **sup** over units, and under
  an unbounded covariate the sup is an extreme order statistic: measured `309 -> 382 -> 397 -> 734`
  as `n` runs `500 -> 4000 -> 32000 -> 128000`, while the 95th percentile sits at `22, 24, 19, 19`.
  A benchmark that quadruples because more data arrived is not a benchmark, so `quantile` defaults
  to 0.95 and the chosen value is reported; pass `quantile=1.0` for the uniform bound. And the MSM
  interval is **not symmetric about the mean** -- a positive estimate is reconciled by the lower
  endpoint, a negative one by the upper, and the two read opposite tails. Reusing the upper tail for
  both understates the confounding: on a right-skewed null it returns a finite `Gamma` where the
  true answer is that no `Gamma` reconciles the sample at all.

  `validation/gamma_benchmark.mac` also collapses the shipped three-constant bound to one blend,
  `mu + (1 - 1/Gamma)*(CVaR - mu)`, verified against the code to 4.4e-16.

- **What a setpoint-tracked log identifies, and what it manufactures**
  (`chc.dynamics_id.closed_loop_gain_attribution`, `closed_loop_attribution_certificate`,
  `ClosedLoopAttribution`). Result 41 ended with an open item: why the interaction coefficient `b1`
  of `dx/dt = d + a*x + (b0 + b1*x)*u` comes out large and negative on a tracked zone. It is a
  property of the log. A proportional loop puts every sample on an affine manifold `x = c + m*u`
  with `m = -1/gain`, and restricted to it the four-term class collapses to a quadratic in the
  action where only `b1` reaches the `u^2` term. So **the interaction is the identified coefficient
  and the pole is not** -- the inverse of the usual reading -- and `b1 = C/m = -gain*C` with `C` the
  curvature of the response in the action.

  Two arms separate an interaction the plant *has* from one the loop *manufactures*. With a real
  `b1 = -0.30` the fit returns it exactly at every gain while the drift error stays at ~0.06 and the
  design's condition number is `3.7e16`. With **no** interaction but a curvature the class cannot
  represent, the fit answers `-gain*C`, growing linearly: `-0.073, -0.145, -0.378, -1.163` over
  gains `0.5, 1, 2.6, 8`. At gain 2.6 a curvature of 0.1454 is reported as `-0.37804`, against the
  `-0.3779` measured on the emulator. Exploration off the manifold is the remedy and it is cheap:
  `sigma = 0.05` drops the condition number to 893 and recovers the drift exactly.

  This also refutes the guess that `b1 = -1/gain`: that is the manifold *slope*, and it moves the
  opposite way in the gain -- the interaction grows with a tighter loop, the guess shrinks.

- **Global strong monotonicity of the congestion equilibrium**
  (`chc.games.equilibrium_monotonicity_certificate`, `EquilibriumMonotonicityCertificate`).
  Result 39 (b) bounds `||(I - S')^{-1}||` at the equilibrium -- an implicit-function derivative, so
  it prices infinitesimal perturbations, and the entry said so. `F'(x) = I + kappa*J(s(x))` with `J`
  a covariance matrix, so `F' >= I` at *every* `x`: `F` is 1-strongly monotone globally, and a
  finite operator perturbation moves the equilibrium one-for-one rather than only to first order
  (measured 0.829 <= 1 over 24 perturbations). On the fixed-mass tangent space the pairwise form of
  the variance gives a computable bound `1 + kappa*n*s_min^2`, exactly attained at the two-point
  uniform `s` -- the same configuration that makes the Popoviciu bound of Result 39 (a) sharp. z3
  and cvc5 both return unsat on the negation at `n = 3`.

  The measurement that matters for using it: the *tangent* improvement is local and **evaporates**
  over a box. Near a corner the softmax approaches a vertex, `s_min` collapses, and both the
  measured tangent modulus and its bound fall back to 1 (1.000183 over the box against 1.0521 on a
  ball around the equilibrium). Quoting Result 39 (b)'s strictly-better constant globally is the
  mistake this certificate exists to prevent; what survives globally is the ambient 1, which is what
  the finite-perturbation bound needs.

### Changed

- **`equilibrium_transfer_certificate` now measures the regularity its order claim assumes**
  (new fields `constrained_slope`, `vertex_regret`). The quadratic order needs the leader's optimum
  to move smoothly with the operator; an active constraint destroys that. With the budget constraint
  active the log-log slope falls from `1.973` to `1.135`, and at a vertex the argmax is locally
  constant, the plan does not move, and the regret is identically zero across the whole sweep -- not
  a degraded rate but the absence of one. The assumption was in the docstrings; it is now in the
  output, where it can fail.

- **The minimax LQ controller over a partially identified effect** (`chc.regret.minimax_action`,
  `minimax_lq_policy`, `minimax_lq_certificate`, `MinimaxAction`, `MinimaxBranch`,
  `MinimaxLQPolicy`, `MinimaxLQCertificate`). Result 33 proved by counterexample that certainty
  equivalence is *not* minimax for the LQ loss and closed with "a minimax LQ controller over the
  identified interval is not built in this repo". It is now, in closed form: the stage cost is
  convex in the effect so the inner maximum sits at an interval endpoint, the two endpoint branches
  cross at exactly two points (`u = 0` *and* the equalising action `target/b_hat`), and a convex
  piecewise quadratic is minimised at a branch minimiser or a kink -- four candidates, evaluated,
  exact. `minimax_lq_policy` runs the same form as a robust Riccati recursion, so the horizon policy
  is linear feedback rather than a minimax search.

  Three things fell out that are worth knowing before using it. The *optimistic* endpoint never
  supplies the answer -- the choice is between the CE action for the pessimistic endpoint and the
  equalising action, decided by `curvature*b_lo*halfwidth <= effort`. The correction has **no fixed
  sign**: the robust action is smaller than CE only when `curvature*b_lo*b_hat < effort`, and larger
  otherwise, so "be robust, act less" is a statement about expensive effort rather than about
  pessimism. And the dynamic-programming adversary that re-picks the effect every step -- normally a
  strict relaxation of a constant unknown effect -- buys nothing here, because the robust action
  always leaves the lower endpoint worst; a constant `b_lo` attains the value, measured to machine
  precision on 500 random instances. The horizon value is therefore the constant-effect worst case,
  not an upper bound on it.

  Scalar state and scalar effect. The multivariate lift is not built: with a matrix effect the inner
  maximisation is over a matrix ball and the endpoint argument does not survive.

- **`lam_unc = 1` is now a bound, not a knob** (`chc.adjoint.costate_norms`,
  `chc.adjoint.perturbation_cost_weights`, `ConfoundingRobustPenalty.certified`,
  `chc.uncertainty.confounding_cost_bound_certificate`). Result 38 (b) recorded that the §34
  inequality bounds the per-step *transition* error while the objective needs a cost-to-go
  multiplier `L_{V,t+1}`, and that `lam_unc` was absorbing it as an unidentified scale. The
  multiplier is now supplied, so the penalty at `lam_unc = 1` is an upper bound on what a
  mis-identified control channel can cost, in units of cost.

  It took three pieces, and the middle one only appeared because the certificate was written to
  fail. The adjoint norm `||lambda_{t+1}||` is the first-order sensitivity; the exact RK4 input map
  `dt (I + dtJ/2 + (dtJ)^2/6 + (dtJ)^3/24)` converts a *field* error into a *state* error (its
  spectral norm is 0.99990 on the shipped certificate, near `dt` but not `dt`, and it is not always
  on that side); and a second-order deviation tube closes the Taylor expansion. Without the third,
  the bound **fails at every radius** -- at the optimum the Cauchy-Schwarz step is nearly tight, so
  the positive `O(radius^2)` curvature the first-order expression drops is enough to break it. An
  adversary reaches 1.014 of the first-order term at radius 0.005 and 1.69 at 0.2, while the shipped
  weights hold at 0.995 and 0.953. `||lambda||` alone was a calibrated estimate, not a bound.

  `confounding_cost_bound_certificate` is the gate, and it attacks rather than samples: projected
  gradient ascent on the control-channel error over the spectral-norm ball, from several starts and
  both signs. Random sampling was the first version and hid a violation -- in four parameters it
  underestimates the worst case by more than the margin being tested. The bound is exact for a
  linear plant with a quadratic cost (the objective is exactly quadratic along a perturbation
  direction) and carries an `O(radius^3)` remainder otherwise; `radius = 0` returns the first-order
  weights, which is how the certificate reports both.

- **Per-lever action bounds** (`chc.control.Bound`, `broadcast_box`, `check_box`). `u_lo` / `u_hi`
  on `projected_gradient_control`, `lbfgs_box_control`, `box_stationarity`, `pessimistic_control`,
  `causal_plan` and `mpc_control` now take a scalar, a per-lever `(m,)` array, or a full
  `(horizon, m)` schedule. A real actuator set is rarely a cube -- a budget and a discount move in
  different units over different ranges -- and a single scalar pair forced the caller to widen every
  lever to the loosest one, which is a *larger* feasible set than the plant has. Purely widening:
  a vector spelling out the old scalar reproduces its answer exactly (asserted, not approximated).

  Two things are rejected rather than broadcast, because both were silent wrong answers rather than
  errors. A 1-D bound is read as **per-lever**, never per-step: for `m == 1` a `(horizon,)` array
  would broadcast along the lever axis and constrain the wrong thing, so a time-varying bound has to
  be spelled out in two dimensions. And an inverted box now raises -- `jnp.clip` with `lo > hi`
  returns `hi` everywhere without complaint.

  The box also stopped being static to the compilation: it enters the jitted program as an array, so
  a caller sweeping boxes compiles once rather than once per box value.

- **The solver says why it stopped, not only where it landed** (`chc.control.SolverResult`,
  `SolverStatus`, `projected_gradient_solve`, `chc.support.pessimistic_solve`). The tuple-returning
  `projected_gradient_control` / `pessimistic_control` could not distinguish a descent that reached
  its stopping rule from one that ran out of budget, so a planner acting on the second was acting on
  an unfinished solve with nothing to warn it. Three states: `converged` (the backtracking line
  search could not lower the cost by more than `tol` -- the method's own stopping rule),
  `max_iterations` (the budget ran out first), `no_progress` (not one step was accepted, so the
  answer *is* the caller's guess).

  The status deliberately does not claim optimality. A stalled line search is a statement about
  steps, not about gradients, so the stationarity residual `||u - P_box(u - grad J)||` is returned
  beside it and the caller judges. It is measured on the **augmented** objective for
  `pessimistic_solve`, since the penalties are what that descent actually minimised.

  Worth stating because it is the case the feature exists for: on the two-lever instance in the
  tests, a 5 000-step budget leaves the residual at `1e-4` -- an answer that looks finished -- while
  the descent in fact needs 5 574 steps under float64. A small residual is not evidence that the
  solve completed.

  `CausalPlan` gained `solver_status` and `solver_iterations`, which answer a **different** question
  from `certificate_status`: the certificate is about the model (how far the plan can be trusted
  given the error budget), the solver status is about the optimisation. A fully certified plan built
  on a truncated solve is a trustworthy tube around a suboptimal action, and nothing in the tube
  said so. The two existing functions are unchanged thin wrappers, so no caller has to move.

- **Fold design off the cycle** (`chc.network_causal.graph_shells`,
  `chc.regret.fold_heuristic_certificate`, `FoldHeuristicCurve`). `optimal_fold_partition` always
  took a shell decomposition, but the only way to *build* one was `cycle_shells`, so Result 52's
  design law could not be applied to a real topology. `graph_shells` computes distance shells for
  any undirected graph by repeated boolean products -- `dmax` is the spillover truncation and is
  small, so this beats a Python BFS and stays in NumPy. It reproduces `cycle_shells` exactly on
  cycles, preserves the partition property `tr(S_d S_e) = 0` that Result 51 rests on, and leaves
  out-of-range and out-of-component vertices in no shell, which is the right reading of a
  truncated spillover model rather than an invented distance.

  `fold_heuristic_certificate` then measures the thing Result 52's honest-scope note left open. The
  note says the design law is closed-form only on vertex-transitive graphs and that beyond them the
  problem "degrades to combinatorial search". **Measured, and it degrades in the closed form, not
  in the answer:** on nine topologies -- cycle, path, star, barbell, grid and four Erdos-Renyi
  graphs -- at both `m = 12` and `m = 16`, the spectral-plus-swap fallback returns the *exact*
  enumerated optimum on all eighteen instances, ratio `1.0` to machine precision. The random arm is
  there on purpose: the structured graphs all have a cut a human can see, which is the case a
  spectral relaxation is built for, so on their own they would flatter the heuristic.

  The Ky Fan gap is reported beside the ratio because the two are different quantities: at the true
  optimum the bound is still loose by 0.5-3.4%, so a large gap says the *certificate* is weak, not
  the design. And the comparison stops being available exactly where a design would be used -- at
  an `m` too large to enumerate -- which the certificate says rather than hides.

### Fixed

- **`projected_gradient_control` and `pessimistic_control` crashed outright on float32 actions
  under `jax_enable_x64`.** Reproduced on the released 0.4.0 wheel, so this predates the per-lever
  work that surfaced it: `lax.while_loop` rejected the body with *"carry input and carry output
  must have equal types"*. The cause is one line down from the symptom. The descent treats the
  actions' dtype as the working precision, but the gradient does not follow it --
  `control_gradient_adjoint` differentiates a cost whose `Q`/`R`/`Qf`/`x_target` are whatever
  `jnp.array` produced, which with x64 enabled is float64 even when the actions are float32 -- so
  `us - lr * grad` promoted and the candidate re-entered the carry one dtype wider than it left.
  Fixed by casting the candidate back at the point that decides what the carry holds, rather than
  by widening the actions or narrowing the gradient at any of the call sites. `chc.uncertainty`
  already carried a note that a float32 run is a *different* computation, not a cheaper one; this
  is the same lesson one layer down.

- **A green local `ty` was a red CI `ty`, and the gap was structural rather than a slip.**
  `Panel.provenance` read `jax.config.jax_enable_x64`, which jax 0.11 declares on `Config` and jax
  0.10 only injects at import time -- and `uv.lock` resolves 0.11 at Python >= 3.12 and **0.10.2
  below it**, so the check passed on the 3.14 development environment and failed on CI's 3.11
  runner. Fixed by asking the question through a declared public function instead:
  `jax.dtypes.canonicalize_dtype(np.float64) == np.float64`, which is the same fact by
  construction, since `float64` canonicalises down to `float32` with x64 off. Confirmed by
  reverting the fix and watching 3.11 fail again.

  The loop that missed it is now in the `justfile`: `just types-matrix` runs `ty` on 3.11 / 3.12 /
  3.13 / 3.14, each in its own environment so `.venv` is not swapped underneath the caller, and
  `just all` includes it. One interpreter is not enough to type-check a package whose lockfile
  resolves different dependency versions across its own supported range.

- **`CappedExplorationPolicy` was advertised in `chc.__all__` but never imported into it**, so
  `from chc import CappedExplorationPolicy` raised `ImportError` and the return type of the public
  `capped_exploration_policy` had no name at the top level. Fixed at the producer -- the class is
  now imported from `chc.regret` -- rather than by deleting the advertisement, since a function in
  the public namespace returning a type that is not in it is the defect, not the symptom. A new
  `tests/test_public_api.py` pins three namespace invariants that nothing was checking: every
  advertised name resolves, no name is advertised twice, and **no export shadows a submodule**.

## [0.4.0] — 2026-09-03

### Added

- **Result 51 -- the non-separable half of the delayed-exposure gate** (`validation/delayed_network_exposure.mac`,
  `proofs/delayed_network_exposure.v`). Derivation and proof only; no API change. Result 43 priced a
  violated fold structure on an exchangeable cluster, and `delayed_exposure_gate.mac` STEP 7 showed a
  delay is FREE under a separable covariance with unit-level folds (`tr(T)` cancels, `dPsi/dphi = 0`).
  The remaining route is a delay that PROPAGATES THROUGH THE NETWORK -- shell-`d` neighbours arriving at
  lag `delta*d` -- and there `Psi` is a POLYNOMIAL in `phi^delta` of degree the spillover truncation:

  ```
  Psi = (m^2/(tr(Au)^2 v0)) * sum_l u_l phi^(delta l),  Au = r^2 P_fold + P_within,  r = K/(K-1)
  u_l = sum_{|d-e|=l} g_d g_e [ tr(S_d S_e) - r^4 n_d'n_e/m + (r^4-1)(K/m) W_de ]
  ```

  Row-level cross-fitting over-weights the fold-contrast subspace by exactly `r^4` (16 at `K=2`, 5.06
  at `K=3`, 1.52 at `K=10`), so MORE FOLDS is the primary lever -- one Result 43's exchangeable law
  cannot see. Three consequences with numbers: the delay-proof same-fold edge fraction is
  `theta*(K) = K^3/(4K^3-6K^2+4K-1)`, graph-free and always above `1/K`, so random folds undershoot
  and the delay drives `Psi` down monotonically; exact delay-proofness is an INTEGER constraint
  (`8m/15` on `C_m` at `K=2`, so `15 | m`, with a verified `C_30` witness); and `Psi` can dip `14.7%`
  BELOW both endpoints, so checking `phi = 0` and `phi = 1` does not bound the penalty. Reduces to
  STEP 7 at `delta = 0` and to Result 43 on a complete graph, with `c(m,K)` now closed-form and its
  excess exactly `(10m-4)/(m+2)^2` at `K = 2`.

  Certified by `chc.regret.delayed_network_certificate`, which measures `Psi` on trajectories drawn
  from the GENERATIVE definition rather than from the Kronecker formula (all eight cells within 1.5
  Monte-Carlo standard errors at 4e5 draws), and which also measures the design law's LIMIT: the
  `theta*` rule is a `D = 1` statement, and on `C_6` the shell-1<->shell-2 term (`-13.44`) is larger
  than the `+11.20` an aligned partition buys, so alignment SHRINKS the damage (6.57x swing vs 1.34x
  on the certificate's grid) rather than removing it.

- **The panel that makes Result 51's `delta` and `phi` FITTED, and the crossover law that needs them
  both** (`chc.network_causal.DelayedNetworkPanel`, `estimate_propagation`, `within_ar1`;
  `chc.regret.delayed_network_certificate` gains `crossover` and `trace_gap`).

  *The new law.* `tr(Au) = r^2(K-1) + (m-K)` counts eigenvalue multiplicities and `v0` never sees the
  fold operator, so both normalisers are partition-free (`trace_gap` measures exactly 0) and equal
  `Psi` between two partitions is a bare root of `sum_l (u1_l - u2_l) x^l`. Those coefficients carry no
  `delta` -- on `C_6` with `gammas = (1, 7/10, 2/5)` they are `(26, -392/5, 32)` identically at
  `delta = 1,2,3,4` -- so the design crossover is fixed in `x = phi^delta`, at
  `x* = (49 - sqrt(1101))/40 = 0.3954669988481472` (Maxima; the certificate matches to 15 digits).
  Below `x*` ALIGNED folds win, above it ALTERNATING ones do. The `phi` threshold is `x*^(1/delta)`:
  `0.3955` at `delta = 1` but `0.6289` at `delta = 2` -- **the same persistence flips the
  recommendation when the delay changes**, which is why reading `phi` off a panel is not enough.
  Proved in `proofs/delayed_network_exposure.v` (`fold_trace_partition_free`,
  `crossover_is_difference_root`, `x_star_is_the_crossover`, `longer_delay_favours_alignment`).

  *At `D = 1` it closes, and closing it REFUTES the obvious generalisation.* `tr(Au^2) = m - r^4 +
  (r^4-1)K` counts eigenvalue multiplicities just as `tr(Au)` does, so the whole `d = 0` block is
  partition-free and cancels, taking `r`, `K` and `m` with it:

  ```
  x*(D=1) = -(g1 / 4 g0) * Delta W_11 / Delta e_in
  ```

  with `W_11` the same-fold length-2 walk count and `e_in` the same-fold edge count -- verified to
  `1.6e-15` over 72480 partition pairs across `C_6, C_8, C_10, P_6, P_8, K_6` at three values of
  `g1`. So the crossover is exactly LINEAR in the spillover decay ratio, the graph contributing only
  an integer ratio. And unlike the graph-free `theta*`, **`x*` is graph-DEPENDENT**: same-fold edges
  do not determine same-fold 2-walks, and `C_10` and `P_6` carry partition pairs at identical
  `(theta_1, theta_2)` that disagree on `x*`. Tested as a refutation, not left as an open question.

  *The estimator.* `estimate_propagation` runs a shell-resolved panel local projection whose rows are
  cut inside one unit's trajectory, and regresses the peak lag on shell distance through the origin.
  It recovers `delta` exactly (`1.000 / 2.000 / 3.000` at truths `1 / 2 / 3`, degenerate intervals,
  not censored) and reads the spillover truncation off the flat tail (`D_hat = 2`); a shell past the
  truncation has no direct edge, so its peak stops advancing, and including that point halved the
  slope. `within_ar1` inverts the Nickell bias of the within transform, `(1+phi)/(p-1)`: raw `0.5580`
  against `0.6` at `p = 40`, corrected `0.599 / 0.600 / 0.603`. The lag matters -- building the
  exposure at lag 0 instead of the true 2 costs 69% of the spillover estimate (`0.599 -> 0.187`) and
  inflates the direct effect (`1.001 -> 1.413`).

  *`Psi` is an estimator's variance, under conditions.* For `theta_hat = u'A eps / u'A u` with an
  isotropic regressor and a disturbance carrying `Sigma`, the two partitions' variance ratio measured
  `0.7195 +- 0.0032` against the law's `0.7150` -- the first time `Psi` has been tied to an estimator
  rather than to a process. But two conditions decide whether it bites, and both were measured: with
  i.i.d. outcome noise the ratio is `1.0028 +- 0.1003` (the fold assignment cannot matter at all --
  hence the new `disturbance_scale` knob, the analogue of Result 43's cluster random effect), and with
  the shipped degree-2 ridge nuisance it is `0.986 +- 0.099` (rank 15 against 5760 rows is nowhere near
  `Au (x) I_p`). Swapping in the residualiser whose Gram IS `Au (x) I_p` moves the direct coefficient's
  variance ratio to `0.59-0.66` -- a 34-41% reduction from the partition alone.

  *And it moves realised COVERAGE.* On the premise-matched design (60 clusters, 4000 replications,
  `M'M = Au (x) I_p` verified exactly) the cross-fit hat leaves BOTH arms unbiased (`+9e-5`,
  `+3.7e-4`) while a textbook standard error understates the realised spread in both -- and by
  different amounts, so a nominal 95% interval realises `0.862 +- 0.006` under alternating folds
  against `0.820 +- 0.006` under aligned ones. At `phi = 0.6, delta = 1`, `x = 0.6` is above `x*`,
  where the law says ALTERNATING wins, and it does: choosing the partition by the crossover law is
  worth **4.2 points of interval coverage** (5.1 sigma). The realised variance ratio
  `(0.01612/0.01920)^2 = 0.705` independently reproduces the law's `0.7150`.

- **`fold_groups` on `estimate_network_effects`** -- opt-in graph-aware cross-fitting. `None`
  reproduces the historical row permutation byte-for-byte; a supplied labelling permutes and chunks
  the distinct labels instead, keeping each group whole. The default is deliberately untouched: the
  function is public API in 0.2.0 and moving every existing user's numbers silently is not a fix.
  Random rows sit at `theta ~ 1/K`, always on the undershooting side of `theta*` at `D = 1`.

- **`fold_groups` on `estimate_network_effects_gnn` as well -- kept because it was measured, and
  measured to do something the law does not predict.** 400 paired replications on
  `DelayedNetworkPanel` (both designs share each draw; arm correlation 0.88 / 0.97, so the ratio is
  bootstrapped over seeds): parity-vs-block unit folds move the DIRECT effect's variance to
  `0.887` [0.803, 0.976] and the SPILLOVER effect's to `0.999` [0.950, 1.052]. The direct channel
  gains ~11%; the spillover channel is inert; Result 51's scalar `0.715` falls outside BOTH
  intervals. A scalar `Psi` cannot be channel-dependent by construction, so this is the measurement
  that makes the two-column sandwich necessary rather than merely sharper -- Result 51 (j). The
  obvious mechanism (the exposure is a neighbour average and so carries no fold-contrast energy) was
  first tested through a crude proxy and rejected there: the design moves `frac_fold(u)`
  0.0015 -> 0.0030 and `frac_fold(e)` 0.00006 -> 0.00183, differences far too similar to explain
  11.3% against 0.07%. Result 51 (j') below withdraws that rejection -- what failed was the proxy,
  not the idea, which survives in its quadratic-form version.

- **The matrix sandwich behind the channel asymmetry, and the coupling hypothesis REJECTED** --
  Result 51 (j'). Derivation and measurement only; no API change. For the *linear* cross-fit the
  residualisation is an explicit linear map (`A[test,test] = I`,
  `A[test,train] = -phi_te (phi_tr' phi_tr + lam I)^-1 phi_tr'`), so
  `theta_hat = (R0' M R0)^-1 R0' M y` with `M = A'A`, and
  `Cov(theta_hat | draw) = (R0' M R0)^-1 R0' M Sigma M R0 (R0' M R0)^-1` exactly. On 200 paired draws
  it predicts **both** channels -- direct `0.9099` against a measured `0.8904` [0.8033, 0.9812],
  spillover `0.9826` against `0.9814` [0.9182, 1.0531] -- while the scalar `Psi = 0.715` falls outside
  both. The operator was checked against the shipped estimator's own coefficients (`3.8e-7`, the
  float32/float64 gap) and `Sigma` against 40000 draws of the generative disturbance (2%, the
  Monte-Carlo error at that count).

  *Two candidate mechanisms killed by measurement.* Nuisance leakage measures **six orders** below the
  noise term (`1e-8` against `7e-2`), and zeroing the off-diagonal of `R0' M R0` moves the prediction
  only `0.9099 -> 0.9088` and `0.9826 -> 0.9802` -- so the asymmetry is **not** the cross-channel
  coupling. With the bread decoupled each channel is `r'M Sigma M r / (r'M r)^2` for its **own**
  regressor, which is the `Omega`-generalisation applied per channel: the exposure is a neighbour
  average and overlaps `M` and `Sigma` differently from the unit-level treatment. That withdraws the
  earlier `frac_fold` rejection above -- the proxy failed, the idea did not.

  *Scope.* Exact for the linear cross-fit only. The GNN figures come from a learned nuisance for which
  no such `A` exists; the linear estimator on the same panel measures `0.890 / 0.981`, so the
  qualitative asymmetry is shared but the numbers are not interchangeable. A first 150-replication run
  with an independent-arm standard error put everything within `1.0-1.7 sigma` and could decide
  nothing -- the pairing is what makes the comparison identified at all.

- **`chc.regret.exact_ratio_moment`** -- the exact `E[(R'BR)/(R'CR)^2]` for `R ~ N(0, Om)` as one
  resolvent quadrature (an `eigh` plus `scipy.integrate.quad`), Result 51 (l)/(m)'s consumer. Call
  with `numerator = A Sigma A`, `denominator = A` to get the exact variance of the scalar cross-fit
  estimator where the plug-in trace law is 7-11% off and opens a wrong-partition band. Existence is
  a tail exponent and is ENFORCED, not assumed: `k >= 3` nonzero denominator eigenvalues, `k >= 5`
  when the numerator loads on the denominator's null space -- the plug-in number exists in both
  divergent cases, which is exactly how a truncation gets quoted where the moment does not exist.
  Tested against chi-square closed forms (`E[1/chi2_n] = 1/(n-2)`, the split-off null-space case
  `2/3`), both divergence guards, and the (m) immunity: the plug-in crossover bisected in `phi`
  lands the exact ratio on `1` to `1e-7` at `Om = I`.

- **The fold-spectrum law (Result 52)** (`validation/fold_spectrum_law.mac`,
  `proofs/fold_spectrum_law.v`). Derivation only; no API change. The fold partition enters Result
  51's sandwich only through `tr(A^2 S_d S_e)`, and `A^2 = I - r^4 E + (r^4-1) F` -- squaring the
  cross-fit residualiser promotes `r^2` to `r^4` and nothing else -- so `Psi` is AFFINE in the
  same-fold weighted 2-walk count with positive weight: the variance-optimal fold design is the
  MINIMUM-weight balanced graph bisection under delay-dependent weights
  `Q(x) = sum g_d g_e x^|d-e| S_d S_e`. On a cycle this diagonalises over Fourier modes: the
  optimal fold frequency is `cos(theta_star) = -g0 x/(2 g1)`, width-2 stripes beat BOTH partitions
  `delayed_network_certificate` compares (by up to +55% of `Psi` at small `phi`; contiguous blocks
  are never optimal for `m >= 8`), and
  `lambda_parity - lambda_stripes = 4 (g0 x - g1)(2 g2 x - g1)` puts parity exactly between
  `x = g1/g0` and `x = g1/(2 g2)` -- `7/10` and `7/8` at the docstring gammas. At the estimator
  level the plug-in thresholds are panel-length-invariant (the time block enters the trace through
  its diagonal), while `exact_ratio_moment` moves the lower threshold up (`0.70 -> 0.76-0.86`
  across tested panels) and erases the upper, re-entrant one entirely: the plug-in law predicts a
  design regime the exact moment does not have.

- **The time-fold law (Result 53)** (`validation/time_fold_law.mac`, `proofs/time_fold_law.v`).
  Derivation only; no API change. Folds that cut TIME under AR(1): the partition enters the
  sandwich only through the same-fold `phi`-weighted pair count, and the mode score is the AR(1)
  spectral density `(1-phi^2)/(1-2 phi c + phi^2)` -- strictly increasing in `c = cos theta` for
  every `phi`, so the variance-optimal time-fold design ALTERNATES time points between folds at
  every `phi`; the contiguous half-split (the default of most panel pipelines) is the WORST
  balanced design, exhaustively over all partitions at `p <= 14`, with a plug-in price up to 6.9x
  and an exact-moment price of 2.46x at `phi = 0.99` (quote the exact one -- the plug-in headline
  is itself Jensen-inflated). No thresholds, unlike the spatial fold-spectrum law: the time kernel
  is completely monotone in the lag (second difference `phi^r (1-phi)^2`), which places the
  alternating optimum in the Hubbard most-homogeneous family and extends the law to any
  convex-decreasing correlation profile. Scope: nuisance cross-fitting variance, NOT forecasting
  model evaluation -- blocked/hv-CV keeps time contiguous to stop evaluation leakage, and this law
  prices that choice instead of overruling it.

- **The van Trees arm on `information_lower_bound_certificate` -- Result 10's `needs LAM` caveat,
  discharged** (`validation/action_van_trees.mac`, `proofs/action_van_trees.v`). Result 10's own
  scope note recorded that its Cramer-Rao floor is a delta-method statement for UNBIASED estimators
  and that a rigorous version needs local-asymptotic-minimax or van Trees on the estimand `u*(b)`.
  Both caveats dissolve, and the constant does not move.

  The one-step LQ regret is EXACTLY `(b^2+rr)(u - u*(b))^2` -- a squared error in the ACTION, for
  every `u`, with nothing linearised. Result 10 reached a squared error in the EFFECT by linearising
  `u*(b)`, and that step is what forced the "local" caveat. Applying van Trees (already formalised
  for this line in Result 42) to `psi(b) = u*(b)` and multiplying by that exact curvature gives
  `n E[regret] >= n (b^2+rr)(E_lambda psi')^2/(n V_id/sigma^2 + I(lambda))`, whose limit is
  `C sigma^2/V_id` -- the SAME constant, for every estimator, biased or not. The finite-`n`
  shortfall is explicit and `O(1/n)`.

  The new arm is the one that shows this was worth doing. A Hodges estimator drives the regret AT
  `b` to exactly `0` -- ratio `0.000000` against the unbiased Cramer-Rao floor, which is the
  concrete reason that floor was never a minimax statement -- while sitting `144.7x` above the van
  Trees floor, the price superefficiency pays off-centre; the efficient plug-in clears the same
  floor by `2.73x`. The bound separates the two by ~53x instead of being vacuous for both. The
  knife-edge caveat `rr = b^2` survives, `psi'` vanishing there being a property of the problem.
  New fields: `van_trees_action_floor`, `hodges_pointwise_ratio`, `hodges_bayes_ratio`,
  `plugin_bayes_ratio`.

- **`chc.regret.capped_exploration_policy` -- the capped exploration policy, and a REFUTATION of the
  conjecture that a cap makes tapering right** (`validation/capped_exploration.mac`,
  `proofs/capped_exploration.v`). `minimax_exploration_certificate` left one item open ("a matching
  causal policy under a per-round action cap is still open") and one conjecture unchecked ("a
  per-round action cap -- which every real actuator has -- is what makes a taper the right shape").
  The conjecture is wrong, and its docstring is corrected in the same commit.

  The regret `A sum_t v_t + K sum_t 1/(I0 + c S_{t-1})` depends on the schedule only through its
  PREFIX sums and is strictly decreasing in them, so moving exploration earlier at equal budget
  strictly lowers it -- an exchange argument that never mentions the cap. The Hessian is a sum of
  rank-one PSD terms, so the objective is convex and that argument yields a GLOBAL optimum: saturate
  the cap on a prefix, then stop. A clipped burst.

  The optimal block length is `n* = sqrt(K T/(A c))/cap`, and the leading cost there is
  `2 sqrt(A K T/c)` -- with `K = A (du*/db)^2` and `c = eta/sigma^2`, EXACTLY the uncapped constant
  `c_causal sqrt(T)`. So the cap's entire price is the harmonic sum `(K/(2 c cap)) ln T + O(1)`:
  ADDITIVE and logarithmic against a `sqrt(T)` floor, not a constant factor. Measured at
  `cap = 0.03`, the ratio to the uncapped floor falls `1.131 -> 1.077 -> 1.036` over three decades
  of `T` while the clipped taper stays `36%` above the optimum; the shipped block matches a
  projected-gradient solve of the full convex program to seven decimals at every cell. The
  counter-intuitive part, and the reason the conjecture failed: `n*` grows like `sqrt(T)/cap`, so a
  TIGHTER actuator explores for LONGER, not more gently -- a cap is a rate constraint, and the
  response to a rate constraint is duration, not shape.

  Cross-checked: eight Maxima residuals all 0, six Rocq lemmas (including the AM-GM floor and its
  equality case), and four `QF_NRA` negations returning `unsat` from z3 AND cvc5.

- **`chc.deep_galerkin.dual_weighted_error_estimate` -- Result 49's blind residual, fixed**
  (`validation/mean_field_dwr.mac`, `proofs/mean_field_dwr.v`; a fourth arm on
  `lq_mean_field_certificate`). Result 49 measured a Deep Galerkin solve whose own residual FALLS
  as its answer degrades near the mean-field obstruction, and could only advise gating neural
  solvers on closed forms -- useless outside the LQ family, where no closed form exists. Because
  the reduced fixed point is affine, the error is an EXACT quotient rather than a first-order
  estimate::

      S_hat(0) - S(0) = (eps - int_0^T z(s).g(s) ds) / den(T),   z(s) = Phi(T-s)^T v / den(T)

  with `g` the model's own reduced defect and `z` the exact adjoint solution, `z(T) = v/den(T)`.
  Since `Phi` is entire, the determinant is the ONLY factor that can blow up, and its zero is
  SIMPLE -- so Result 49's pole exponent, fitted as `-0.998`, is exactly `-1`.

  The estimator reads `S`, `m` and `P = V_xx` off the trained network by differentiation and
  integrates the transition matrix from the model's OWN closed-loop rate, never calling
  `game.solve()`. Measured over eight horizons on the anti-monotone instance, rank correlation
  with the true error: raw residual `-0.667`, residual conditioned by `1/|den|` `+0.405`, this
  estimator `+1.000` with worst relative discrepancy `6%` -- at a horizon where the error is `72`
  and the raw residual is near its smallest. The middle number is the part worth keeping: scalar
  conditioning is necessary but NOT sufficient; what carries the information is the projection of
  the defect onto the adjoint mode. The anti-correlation itself is forced rather than
  architectural -- the reduced residual is homogeneous of degree 1 in `(m, S)`, so a bounded
  approximator facing a diverging solution keeps a small residual by construction.

  Cross-checked on five independent systems: Maxima (five residuals, all 0), Rocq (six lemmas),
  giac (same identities, independent route), z3 AND cvc5 (four `QF_NRA` negations, `unsat` from
  both), Octave (the identity from `expm`/`trapz` with an arbitrary wrong trajectory, `9.6e-14`
  relative), and PARI/GP at 60 digits (`den'(T*)` against its closed form, agreeing in every
  digit; residue `1.000000...`). Scope: exact for the affine family; for a non-quadratic game the
  same construction is the standard dual-weighted residual and is first-order.

- **`chc.regret.exact_matrix_ratio_moment` -- the exact MATRIX ratio moment, and Result 54 with it**
  (`validation/matrix_ratio_moment.mac`, `proofs/matrix_ratio_moment.v`). `exact_ratio_moment`
  priced the SCALAR cross-fit estimator exactly; the two-channel (direct, spillover) estimator needs
  `V = E[(X'AX)^-1 X'A Sigma A X (X'AX)^-1]` for a Gaussian `n x 2` block `X`, which Result 51 (j')
  recorded as blocked. Route, derived rather than cited: `M^-1 = adj(M)/det(M)` turns every sandwich
  entry into signed sums of THREE quadratic forms, the Ingham--Siegel identity
  `det(M)^-2 = (2/pi) int_{T>0} det(T)^(1/2) etr(-TM) dT` replaces the determinant by a Gaussian
  tilt with covariance `(I + 2 Om (T (x) A))^-1 Om` -- so a SINGULAR `Om` (the normal case: the
  spillover column is a deterministic map of the own column) is handled natively, nothing inverts
  `Om` -- and the Isserlis three-form moment closes each entry. The cone integral runs in Cholesky
  coordinates, which cannot leave the cone.

  Anchored three ways, each able to fail: the Ingham--Siegel constant to 8 digits; the Wishart law
  `E[(X'X)^-1] = I/(n-3)` and its Haar generalisation `(tr Sigma / n) I/(n-3)` for a correlated
  numerator, both to 6; and a 10^6-draw Monte Carlo on a correlated-channel geometry within 1.6
  standard errors. Existence is the inverse-Wishart threshold `n >= q + 2` and is ENFORCED -- below
  it the integral diverges (visible as node-count disagreement, `7.57` vs `8.50` at `n = 3`) while a
  plug-in sandwich still quotes a number.

  What it measures. On the delayed-network panel (`C_6`, `p = 5`, `phi = 0.6`) the matrix Jensen gap
  is `13-23%` -- LARGER than the scalar `7-11%` on the same family, because the determinant couples
  the channels, reaching `+41.5%` under strong channel correlation. The plug-in sandwich OVERSTATES
  the alternating design's advantage by 2-3 points of the per-channel ratio (direct `0.827` exact vs
  `0.808` plug-in; spillover `0.895` vs `0.868`) while keeping the ordering: as in Results 52 and 53,
  plug-in ORDERINGS are sturdier than plug-in MAGNITUDES. Scope: `q = 2` channels (the adjugate route
  is what keeps the degree manageable) and Gaussian `X`; cost is `O(nodes^3 (2n)^3)`, seconds at
  `n = 30`, not a hot-path tool.

- **`chc.regret.optimal_fold_partition` -- Result 52's design law as a solver with a certificate.**
  The fold partition enters the sandwich only through the same-fold weighted 2-walk count with
  positive weight, so the variance-optimal design is the minimum-weight BALANCED `K`-cut under
  `Q(x) = sum g_d g_e x^|d-e| S_d S_e`. Small instances (`K = 2`, at most `exhaustive_limit`
  balanced bisections) are enumerated exactly; larger ones run Fiedler-style spectral rounding plus
  balanced-swap local search from several starts. Every result carries the Ky Fan spectral lower
  bound `(m/K)(1'Q1/m + sum of the K-1 smallest eigenvalues on the mean-free subspace)` and the
  relative gap to it, so a local-search answer that cannot be certified says so instead of passing
  silently. Returns a frozen `FoldDesign`.

- **`chc.reachability.higher_order_barrier_gap` -- the relative-degree-2 hole in the pointwise
  barrier check, closed.** `barrier_reachability_gap` recorded a trap: at PURE relative degree 2
  (`B'grad h == 0`) the first-order condition contains no `B` at all, so its verdict is INVARIANT to
  the disturbance radius while the true backward-reachable tube shrinks -- the filter certifies a
  set the plant cannot hold. The higher-order lift `psi1 = grad h . f + alpha1 h` is control-free by
  exactly that degeneracy, and testing
  `robust_hamiltonian(grad psi1, f, B, u_max, radius) >= -alpha2 psi1` on `{h >= 0} and {psi1 >= 0}`
  puts the radius back in through `B'grad psi1 != 0`. Measured on the double integrator: the
  first-order barrier fraction is IDENTICAL at radius `0` and `0.8` while the reachable fraction
  falls, and the second-order one is strictly smaller at `0.8` -- then saturates at the drift-only
  verdict once the radius swallows the lifted channel, which is the zero-action rule reappearing one
  level up. Returns a frozen `HigherOrderBarrierGap`.

- **The Jensen gap in Result 51's `Omega`-generalisation: the free trace correction, the EXACT
  resolvent-integral moment, and the attribution it settles** (`validation/omega_jensen_gap.mac`,
  `proofs/omega_jensen_gap.v`). Derivation only; no API change. Result 51 (i) blamed
  the `7-11%` looseness of `Var(theta) ~ tr(A Sigma A Om)/tr(A Om)^2` on `E[X/Y^2] != E[X]/E[Y]^2`
  "growing with `Om`'s conditioning". Both halves are now measured. The effect itself is a **trace
  formula in the same two matrices the law already forms**, so it costs nothing:

  ```
  E[X/Y^2] / (tr(B Om)/tr(C Om)^2) - 1 = -4 tr(B Om C Om)/(tr(B Om) tr(C Om)) + 6 tr(C Om C Om)/tr(C Om)^2
  ```

  with `B = A' Sigma A`, `C = A'A`. `Var(X)` never enters -- the `dx^2` coefficient of the delta
  expansion is exactly 0, because `X` appears linearly. Every residual in the file is 0 and the
  Isserlis moments are *verified* at `n = 2` rather than cited. The sign is not free: the `Cov` term
  enters negative and the `Var(Y)` term positive, so the plug-in is **not conservative by
  construction**, and at `Sigma = I` it collapses to `+2 tr(C Om C Om)/tr(C Om)^2 > 0`.

  *Measured.* Sixteen single-arm configurations at `4e5` draws on the panel's exact
  `Om = I (x) ((kappa^2 QQ' + I) (x) T)`: on all thirteen where the gap is resolved (`>= 9 sigma`) the
  formula has the right sign, `measured/predicted` lands in `[0.77, 1.14]`, and it predicts the
  `phi = 0.95, K = 2` row where the gap **reverses sign** (`+2.07%` measured, `+2.18%` predicted).
  Twelve of the thirteen sit below 1, so the second-order form OVER-states the gap by `10-30%` --
  a truncation whose neglected terms carry the opposite sign.

  *At ratio level the effect is operator-dependent.* The law is quoted as a RATIO of two fold
  designs, where a gap common to both arms cancels. On the ridge cross-fit operator it does: across
  `cond(Om)` from `1.1e1` to `7.8e4` the plug-in ratio errs `+0.05% / +0.06% / +0.08%` (all
  `~1 sigma`) for `cond <= 2.8e2` even though the single-arm gaps there reach `2.3%`; it becomes
  resolvable only past `cond ~ 1e3`, is **not monotone** (`-1.75%` at `1.2e3` but `+0.20%` at
  `7.3e3`), and reaches `+4.1%` at `3.8e4`, where the correction cuts it `4x`. So "growing with
  `Om`'s conditioning" is not a safe summary -- Result 51 (k).

  *And the EXACT moment closes the question* -- Result 51 (l). `1/Y^2 = int_0^inf t e^{-tY} dt` plus
  the tilted-Gaussian moment `E[(R'BR)e^{-tR'CR}] = det(I+2t Om C)^{-1/2} tr(B (I+2t Om C)^{-1} Om)`
  turn `E[X/Y^2]` into a one-dimensional resolvent integral (Magnus 1986); both identities are
  verified in STEPs 5-6 (residual 0 at `n = 1, 2`, the symmetric cross term integrating to zero),
  and STEP 7 pins the tail `t^{-n/2}` -- the moment is INFINITE at `n = 2` while the delta-method
  number exists at every `n`. On Result 51 (i)'s actual geometry (`A = A_u (x) I_p`) the integral
  matches a fresh 300k-draw run in every cell (five of six per-arm cells within `0.8 sigma`, worst
  `2.2 sigma`; all three ratios within `1.2 sigma`) and explains the recorded `7-11%` COMPLETELY: exact ratios `0.7187 / 0.3374 / 0.4004` against plug-in
  `0.7150 / 0.3036 / 0.3721` and re-measured `0.7221 / 0.3377 / 0.3986` (`+-0.0031 / 0.0012 /
  0.0015`). Per arm the gaps carry OPPOSITE signs (`+6.9%` parity vs `-3.8%` block at
  `Om = I (x) T_0`), so in the ratio they COMPOUND to `+11.1% / +7.6%` -- while on (k)'s ridge
  operator they were nearly equal and cancelled. The ratio-level effect is the DIFFERENCE of two
  per-arm gaps; whether it cancels is a property of the fold-operator pair. So (i)'s original
  attribution was RIGHT; an earlier draft of (k) withdrew it on a cross-geometry transplant and was
  re-scoped. The `Om`-generalisation now has an exact quantitative form, with the trace correction
  as its free `~1%` approximation (always from above here). Isserlis (1918) and Magnus (1986) are
  the citations; both identities enter the file verified, not cited.

  *The design consequence* -- Result 51 (m). At `Om = I` the plug-in crossover `x*` is EXACT to all
  orders: both fold operators share the spectrum `{0, r^2, 1}` and `P_fold + P_within = I - P_mean`
  is partition-free, so the two spectral loading differences satisfy `d_1 = -d_r` while the plug-in
  crossover imposes `r^4 d_r + d_1 = 0` -- both vanish (STEP 9a), and the resolvent bracket factors
  as `(r-1)(r+1)(2 r^2 t + r^2 + 1) > 0` (STEP 9b), so exact and plug-in NEVER disagree in sign
  there: the (h) rule picks the right partition at every `phi`, proved in
  `proofs/omega_jensen_gap.v` (`isotropic_loading_differences_vanish`,
  `isotropic_bracket_positive`, `plug_in_and_exact_agree_in_sign`; Stdlib Reals only) and measured
  as a `0.0000` crossover shift at both `delta = 1` and `delta = 2`. At `Om != I` the crossover
  SHIFTS toward later switching and there is a wrong-partition band: `phi in [0.1993, 0.2075]`
  (`I (x) T_0`) and `[0.1899, 0.1953]` (panel) at `delta = 1`, widening to `[0.3925, 0.4079]` and
  `[0.3822, 0.3960]` at `delta = 2` -- inside it the plug-in law recommends the partition the exact
  variance disfavours, and the remedy costs one `eigh` plus a scalar quadrature. Non-Gaussianity is
  priced too: elliptical kurtosis moves the second-order gap AFFINELY, `(1+kap)*rel + kap`
  (STEP 8a, Rocq `elliptical_gap_affine`), and for scale mixtures (multivariate t) homogeneity
  gives `E[X/Y^2] = E[W/nu] * E_gauss` EXACTLY -- `t_10` draws agree with the Gaussian integral at
  `0.3 / 1.6 sigma`, while parameterising by the variance instead of the scale errs by exactly
  `-2/nu = -20%`. The trap is the bookkeeping, not the tails.

- **`DelayOscillationTask` -- the leaderboard row where ignoring a delay is a *bifurcation*, not a
  tuning error.** An incentive moves supply `tau` later, so the plant is `x' = channel*u(t - tau)`;
  proportional feedback closes it to `x' = -channel*K*x(t - tau)`, whose exact boundary is
  `channel*K*tau = pi/2` (`chc.delay.delay_margin` at pole 0). Three arms minimise the **same** cost
  by the **same** grid search on the **same** Euler scheme, differing only in the delay they assume:

  ```
  controller            cost      regret    viol     ood
  oracle                0.57        0.00    0.00    0.00
  delay-aware           0.57        0.01    0.00    0.00
  delay-blind       18954.07    18953.50    0.90    0.97
  ```

  *The blind arm fails on its own terms.* It is handed no penalty -- only no delay -- so its optimum
  is the memoryless `sqrt(q/r)`: `3.1013` on the shipped weights against an analytic `3.1623`, the
  2% gap being explicit Euler's `-ln(1 - dt K)/dt > K`. That gain is **1.97x past `pi/2`**, so the
  loop rings up. `delay-aware` closes the whole gap by estimating the delay from the log
  (`chc.irf.delay_estimate`) and turning the interval into a design (`chc.delay.robust_delay_design`)
  -- the first place the two halves of this line run as one chain.

  *The gate is the measured boundary, not the claim.* Sweeping gains on the true plant, the largest
  decaying is `1.5543` and the smallest growing `1.5647` against `pi/2 = 1.5708` -- a relative gap of
  `-0.39%` bracketing the `-1/(2m) = -0.50%` at `m = tau/dt = 100` derived for explicit Euler with an
  exact integer lag. The row therefore *re-measures* a closed form from elsewhere in the package.

  HONEST NOTES. (1) **The safety columns fire here**, which is the mirror image of
  `CausalDynamicsTask`, where a mis-scaled channel concedes regret while `viol = ood = 0`. Both traps
  are real; neither column is a general detector. (2) The blind arm's *cost magnitude* is an artefact
  of `state_cap` clipping a divergent trajectory -- read the ordering and the constraint columns, not
  the number. (3) The delay is estimated from the **rate**: the plant is an integrator, so the level's
  impulse response is a *step*, and `delay_estimate` correctly returns an interval spanning the
  plateau rather than inventing a mode. (4) `tau` sits at 3.33 observation samples, deliberately off
  the grid, which is what `peak_lag(refine=True)` exists for -- it recovers `0.947` of a true `1.0`
  where the integer argmax can only say `0.900`, and the residual 5.3% is not noise but the derived
  shrinkage `f/(4-6f)`, which at `f = 1/3` predicts `0.950`. (5) **That bias has a sign.** Across six
  logs every estimate lands *below* the truth -- the destabilising direction -- so seeds do not
  average it away; the realised 7.7% shortfall is harmless only against the ball's tolerable 76.6%, a
  factor of 10. (6) Scored on a **single seed** on purpose: the closed loop is deterministic given the
  gain, and the ~1% seed spread in the estimate is quantised away by the gain grid, so a multi-seed CI
  would be degenerate rather than informative.

- **`delay_ball`, `delay_design_loss` and `robust_delay_design` -- what Result 44 does *not* survive
  when the uncertain quantity is the delay (Result 50).** Result 44 gives a symmetric ball in the
  dynamics error with a regret quadratic in its radius. Ask the same question about an estimated
  *delay* and **both halves fail, for the same reason**: the decay-optimal design sits at a
  **defective** characteristic root.

  *The ball is a half-line.* Designing `K^ = 1/(e tauhat)` and running it against the true `tau`
  puts the loop gain at `kappa = 1/(e r)`, `r = tauhat/tau`, and the exact boundary caps `kappa` at
  `pi/2`. Since `kappa` is antitone in `r`, the admissible set is `r > 2/(pi e) = 0.23420`:
  under-estimating a delay by more than 76% of it destabilises, **over-estimating never does, at any
  magnitude**. The radius is relative -- a fraction of `tau`, with no length scale -- where Result
  44's `0.0555` is an absolute norm bound.

  *The loss is a square root on one side and linear on the other.* Substituting `s = -1 + u` turns
  the characteristic equation into `(u - 1)e^u + 1 = eps`, whose left side has a vanishing first
  derivative -- the double root. One inverted series `u = w - w^2/3`, `w = sqrt(2 eps)`, covers both
  regimes by which way `w` points: `sqrt(2 eps) - 2eps/3` over-estimating (two real roots) and
  `2|eps|/3` under-estimating (a complex pair leaving the axis). At the same `|eps| = 0.05` those
  are `0.287` and `0.033` -- **8.8x apart from the identical absolute error**. No exponent describes
  both sides, so `J - J* <= C |dtau|^2` has no analogue and `DelayBall` carries a floor with no
  ceiling.

  *So the two directions want opposite things, and the interval decides.* `robust_delay_design`
  takes the ends of a delay interval -- `chc.irf.DelayEstimate.lo` and `.hi` are exactly that -- and
  returns the minimax `tauhat`. Because the loss is asymmetric the answer is **not the centre**: on
  `[0.8, 1.25]` it lands at `0.837` against a geometric mean of `1.0` and halves the worst case,
  `0.528` to `0.270`. The rule has a stated domain -- the shift depends on `hi/lo` alone, deepens to
  `0.754` near `hi/lo = 3.2`, then **crosses back above the mean at `hi/lo = 13.25`**, where the low
  end nears the stabilising floor and its saturating loss takes over. It is a regime, not a law.

  *Verified three ways.* `validation/delay_ball.mac`: the root's multiplicity, the substitution
  residual `0`, the inversion coefficient `-1/3`, the exact complex branch `p = -q cot q` with
  trigonometric residual `0`. `proofs/delay_ball.v` (Stdlib Reals, standard axioms only) leaves the
  design constant and the boundary **abstract**, so what is machine-checked is that *any* rule with
  gain inversely proportional to the assumed delay has this shape -- including `no_upper_radius`, the
  half-line stated as an unbounded-above existence claim. `delay_ball_certificate` brackets the
  floor in `(0.9, 1.1)` of its derived value with the loop genuinely diverging below it, and matches
  the derived loss to the simulated decay rate within `0.93 * dt/tau`, measured flat over an 8x
  range of `dt`.

  *No Lyapunov-Krasovskii functional, deliberately.* An LKF gives a sufficient condition with an
  unquantified gap; the characteristic equation gives the exact boundary. That is also why there is
  no conservatism figure to report against Result 44's 15.5x -- there is no slack to measure.

- **`delay_estimate` -- the shipped IRF turned into a delay with an interval around it.** A lagged
  edge in a causal graph is a claim nobody can check without one. `chc.irf.delay_estimate` locates
  the peak of the identified local-projection IRF and prices it by a **moving-block percentile
  bootstrap** over the aligned projection rows, refit and re-peaked on every resample. Blocks, not
  rows: a local projection's rows overlap by construction, so an i.i.d. row bootstrap would destroy
  the dependence that sets the width. The block length `lags + horizon + 1` is exactly the window,
  which is also the separation at which two rows stop sharing an observation.

  *The sub-step refinement is off by default, and the measurement is why.* Fitting a parabola
  through the peak and its neighbours is the textbook time-delay-estimation move, and on a causal
  impulse response it is **biased**: the response is one-sided -- zero before arrival, decaying
  after -- so it is maximally asymmetric at exactly the point being located, and the vertex lands
  at `lag + phi/(2(2 - phi))` (residual `0` in `validation/delay_estimate_bias.mac`), which is
  `0.409090...` of a step late at `phi = 0.9` -- the measured mean was `5.409` for a true lag of 5.
  That bias is free of the sample size, so it does not shrink with data, and it is three times the
  bootstrap width, so it would have silently decided the answer. Measured coverage of a nominal 95%
  interval with the refinement on: **0.000** over 40 replications; with the integer argmax:
  **1.000**. `refine=True` remains available and earns its place where the response is smooth
  across the grid -- on a delay falling halfway between two samples it recovers `5.503 +- 0.044`
  where the argmax can only quantise to `5.475 +- 0.499`. Even there it is exact only for a
  symmetric peak: Maxima puts a lag split `(1 - f, f)` across two bins at `f/(4 - 6f)`, so `f = 1/2`
  is recovered exactly and `f = 1/3` comes back as `1/6`. It moves the right way, continuously, and
  is not unbiased -- which is the whole reason it is not the default.

  *Lag augmentation is offered, not imposed, because the benefit could not be measured here.*
  `local_projection_irf` gained a `lags` argument (Montiel Olea & Plagborg-Moller 2021), which fixes
  the asymptotic variance of a single projection coefficient under persistence. The peak, though, is
  a location statistic invariant to a common rescaling of the IRF, and the block bootstrap already
  handles the serial dependence non-parametrically: over 40 replications at `phi in {0.98, 0.999,
  1.0}` coverage was 0.95-1.00 with and without, and the widths agreed to 3%. So the default stays
  `0` and the docstring says what was and was not measured.

  *The three degenerate cases are answered by the estimate, not by a guard.* A flat IRF is not
  rejected -- the peak wanders and the interval comes back spanning 78% of the horizon. A
  sign-flipping response is located on `|beta|` and reported with its sign in `peak_response`. A
  peak pressed against the last horizon is flagged `censored`, meaning the response is still rising
  at the edge; an effect that never arrives inside the horizon is *not* censored, because there is
  nothing at the edge to see, and the width is what reports it.

  One branch was deleted rather than tested: the parabolic fit's degenerate-denominator guard is
  unreachable, because `argmax` returns the *first* maximum and so the left neighbour is strictly
  smaller, making the curvature strictly negative.

- **`delay_margin` and `delay_margin_certificate` -- how much measurement delay a loop survives,
  and evidence that the number is where the derivation says.** For `x' = a x - K x(t - tau)` the
  margin is `tau_c = arccos(a/K)/sqrt(K^2 - a^2)`, from the imaginary-axis crossing of
  `lambda - a + K exp(-lambda tau)`; both real and imaginary residuals are exactly `0` in Maxima
  (`validation/delay_margin.mac`), and the Python agrees with the 30-digit table to `5e-11`. At
  `a = 0` this is the textbook `K tau = pi/2`.

  *Two facts that fall out of the formula rather than being cited.* Margin is antitone in gain, so
  every unit of loop gain is bought with delay tolerance; and `tau_c -> 1/a` as `K -> a+` while
  decreasing in `K` from there, which recovers the classical single-real-unstable-pole limitation
  -- an unstable pole `a` admits **no** controller past `tau = 1/a`.

  *Computed from the exact characteristic equation, deliberately not from the delay line.* The two
  available discretisations err in opposite directions, both derived: the `m`-stage chain of
  `DelayedDynamics` sits `+pi^2/(8m)` **above** the true boundary (optimistic), and explicit Euler
  with an exact integer lag sits `-1/(2m)` **below** it (conservative; series `pi/2 - pi e/4` in
  `e = 1/m`). `pi^2/8 = 1.23` against `1/2`, so the optimistic error is also the larger one. The
  certificate therefore simulates with `exact_delayed_rollout`: a conservative simulator that still
  destabilises past `tau_c` is evidence, an optimistic one staying stable just inside it would not
  be. Its `ratios` step over the `~1/(2m)` band where the discretisation rather than the plant
  decides, and it brackets the boundary in `(0.95, 1.05) * tau_c` on all three test plants -- with
  the far side actually unstable, so the certificate can fail.

  *Rocq* (`proofs/delay_margin.v`, Stdlib Reals only, no axioms beyond the standard three): the
  algebraic core -- `a^2 + w^2 = K^2`, `(a/K, w/K)` on the unit circle (which is what makes a
  simultaneous `cos = a/K`, `sin = w/K` possible at all), positivity of the crossing frequency, and
  the `a = 0` branch with its antitonicity. The transcendental half -- that `tau_c` is the
  *smallest* positive crossing, and the `1/a` limit -- stays in Maxima and the certificate, which
  is the same scoping the rest of `proofs/` uses. One lemma was stated strictly (`w < K`) and is
  false at `a = 0`, where the two coincide; it ships non-strict, with the strict version guarded on
  `0 < a`.

- **`chc.delay` -- a delayed plant that every existing solver already knows how to solve.** A
  discrete delay `x(t - tau)` is not a finite-dimensional vector field, so it cannot be a
  `chc.dynamics.Dynamics`. The `m`-stage linear chain is one: `x' = f(t, x, b_m, u)` with
  `b_i' = (b_{i-1} - b_i) m/tau`, which is also the first-order upwind discretisation of transport
  along the delay line. `DelayedDynamics` presents that as an ordinary vector field on
  `z = [x, b_1, ..., b_m]`, with `augment_state` / `state_of` / `delayed_of` / `state_trajectory`
  to move between the two views and `lift_cost` to embed a cost on `x` into one on `z`.

  *Why augmentation rather than a DDE solver.* A method-of-steps solver would have needed the
  discrete adjoint, both projected-gradient solvers, the barrier and the pessimism radius rebuilt
  against it, and `diffrax` -- already a dependency -- does not solve DDEs. Augmenting instead
  makes the claim testable rather than architectural, and it is tested: on a delayed plant the
  discrete adjoint agrees with autodiff to **1.7e-16**, and `projected_gradient_control`,
  `causal_plan` and `mpc_control` run with no delayed variant of anything.

  *What the approximation costs, derived and not asserted* (`validation/delay_chain.mac`; the
  chain's transfer function `(1 + s tau/m)^-m` is its kernel's Laplace transform, so the moments
  come straight off it). The applied delay is Erlang `(m, m/tau)`: mean exactly `tau`, but variance
  `tau^2/m`, so it is *smeared* with relative spread `1/sqrt(m)`. On `x' = -a x(t - tau)` the
  chain's Hopf boundary is `a tau = m tan(pi/2m) sec^m(pi/2m)` -- characteristic residual exactly
  `0` in Maxima -- tending to the true `pi/2` from **above** with relative excess `pi^2/(8m)`
  (measured against the closed form to `1e-6` at `m = 10` and `8e-6` at `m = 50`).

  The direction of that second error is why the delay margin is not read off this object: the chain
  is **optimistic** about stability. It is the right tool for simulating a delayed plant and the
  wrong one for certifying it. The two errors also converge at different speeds -- `O(1/m)` for the
  margin against `O(1/sqrt m)` for the kernel -- so a stability question needs far fewer stages
  than a reproduce-the-waveform question; `stages_for_spread` sizes the second.

  *A cap that an eigenvalue argument gets wrong by exactly a factor of two.* The buffer block is
  defective -- one Jordan block, the eigenvalue `-m/tau` repeated `m` times -- so its spectrum says
  little about what an explicit integrator does to it. Read as advection, the upwind symbol covers a
  disc of radius `m/tau` centred at `-m/tau` and so reaches `-2m/tau`, giving a CFL condition
  `m dt/tau <= 1.3926` rather than RK4's real-axis limit `2.7853`. `max_stages` returns the CFL
  bound. The first implementation here used the eigenvalue and would have permitted twice as many
  stages as are safe; the failure it would have caused is silent then catastrophic -- at
  `tau/dt = 50` the buffer is bounded through `m = 75`, reaches `2.4e1` at `m = 80` and `8.8e30` at
  `m = 100`. `tests/test_delay.py` asserts both sides of that cliff, so a cap that stopped being
  real would fail the suite.

- **Milestone J closed as *measured, not needed* (`runtime/`, outside the wheel)** -- the crate that
  once lived here was deleted (`367a52f`) for being an unmeasured reimplementation, and the
  milestone's own gate was a *measurement* nobody had taken: "identical closed-loop results on
  golden trajectories; single-binary MPC step within the latency budget". Both halves are now taken.
  `runtime/` is a `nalgebra`-only mirror of the control loop (RK4, discrete adjoint,
  projected-gradient OC over a box); `hatchling` packages only `src/chc`, so the published wheel is
  untouched and no dependency is added.

  *Parity, exactly.* `runtime/parity_check.py` compares three arms on the same LQ instance -- the
  Rust binary, `chc.control` as shipped, and the same recursion compiled into one XLA program. All
  three return cost `3.686190095`, first control `-2.965207777`; worst gap **0.00e+00**. The timings
  are therefore of programs doing identical arithmetic, which is the only thing that makes them
  comparable at all.

  *Latency, and the verdict.* Steady state per solve on an idle machine, alternated across two
  rounds: Rust **2.99 / 2.99 ms**, compiled JAX **3.48 / 3.49 ms**, `chc.control` as shipped
  **114.6 / 113.9 ms**. Rust beats a compiled JAX runtime by **1.16x** -- the same order, and not a
  margin that justifies maintaining the control loop twice in two languages. Cold start, `hyperfine`
  in both orderings (476x and 510x, so not startup drift): 4.7 ms against 2.28 s, which is
  interpreter and JAX import cost and is answered by a warm process rather than a rewrite.

  *The finding is about Python, not Rust.* The 38x that looked like a language gap is
  `projected_gradient_control` being a Python loop that spends one dispatch per gradient and one per
  backtracking trial, up to 41 per outer step. Compiling the identical recursion recovers 33x of it
  inside JAX -- see `runtime/mpc_latency.py`'s `steady-jit` arm, whose answer is bit-identical to the
  shipped path.

- **A bound-constrained quasi-Newton beside the projected gradient (`chc.control`)** -- `plans/10`
  §4 asked for a bespoke NLP solver, which is the wrong call for the reason `plans/03` already
  gives (acados and Clarabel occupy that slot). The weak link is the hand-rolled solver every
  `plan`/`mpc`/`benchmark` call site uses, and SciPy is already a core dependency, so the item is
  reframed: `lbfgs_box_control` hands the *same* discrete-adjoint gradient to L-BFGS-B, which
  curves the step with a limited-memory secant approximation and takes the box natively. Same
  signature, same `(controls, cost history)` return, so it is a drop-in.

  A correction to the item's premise while it is being closed: the existing
  `projected_gradient_control` is steepest descent with a *backtracking line search*, not Adam.

  `nlp_solver_certificate` measures the difference instead of asserting it, and `box_stationarity`
  -- the residual `||u - P_box(u - grad J)||`, zero exactly at a KKT point -- makes the comparison
  independent of both a reference solution and a wall clock. Sweeping the control weight, which is
  what sets the conditioning of the reduced Hessian: at `R = 0.001` the first-order solver exhausts
  its 150-step budget at stationarity 1.2e-01 and **19.2%** above the optimum; at `R = 0.01`,
  4.7%; at `R = 0.1` it is fine, 0.003%. L-BFGS-B reaches stationarity below 1.2e-04 everywhere, in
  18-82 iterations. SLSQP and trust-constr were measured too and agree with L-BFGS-B to six digits
  on every instance, so neither is shipped: their extra capability is general nonlinear constraints,
  which this problem does not have, and trust-constr costs 5-10x the wall clock for the same answer.

  The certificate asserts *both* directions -- a gap above 5% on the ill-conditioned instance and
  below 0.5% on the well-conditioned one -- so it fails if the difference stops being about
  conditioning. **Call sites are deliberately unchanged**: swapping the solver under
  `benchmark`/`mpc`/`plan` would move every published leaderboard and regret number at once, which
  is a separate decision with its own blast radius, not a side effect of adding a solver.

- **The deep ensemble trains as one sharded program (`chc.uncertainty.fit_ensemble`)** -- it was a
  Python loop calling `fit_residual` K times, so a K-member ensemble cost K x `steps` device
  dispatches and used exactly one core no matter how many were free. The members are now a single
  stacked parameter pytree: `jax.vmap` over the member axis, `jax.lax.scan` over the Adam steps, and
  the stack committed to `NamedSharding(_member_mesh(K), P("member"))`, which degrades to a
  one-device mesh with no branch at the call site. The signature, the key derivation and the
  returned `EnsembleResidual` are unchanged. Measured at K=8, 2000 steps, warmed up and alternated
  A-B-A-B: serial 8.03 s, stacked on one device 2.25 s (3.6x, from collapsing the dispatches),
  stacked over an 8-device mesh 0.80 s (10.3x total, so sharding contributes 2.8x on top).

  `sharded_ensemble_certificate` checks it against a serial oracle built from the untouched public
  `fit_residual`, and states *every* agreement in ULP of the working dtype: parameters 232 ULP,
  final loss 186, member disagreement 16 under float64, and 82 / 173 / 146 under float32 -- one
  2000-ULP budget covers both, where an absolute threshold passes under float64 and fails under
  float32 for no reason but the dtype. The parity horizon is short on purpose. Adam on this loss is
  chaotic, so a one-ULP difference in reduction order is amplified without bound with the step
  count: measured 1.3 -> 6.6 ULP over 5 -> 200 float32 steps, then **7557 ULP at 400**, against a
  smooth 1.5 -> 12.9 over the same range in float64. What is certifiable is the equivalence of the
  *recursion*, not bit-agreement of a chaotic trajectory. A subprocess test forces
  `--xla_force_host_platform_device_count=8` (the flag is read when the backend initialises, so it
  cannot be set in-process) and asserts the member axis really spans the mesh. **The GPU half is
  untested here**: an NVIDIA GPU is present but no CUDA jaxlib is installed, and adding one is a
  dependency decision rather than a tool call.

- **The coupled mean-field game in `chc.deep_galerkin`** -- the module solved a 1-D Poisson BVP and
  `chc.transport` carried a forward density with no diffusion and no backward value equation; the
  two halves had never been coupled. `solve_mfg_dgm` now trains `V(t,x)` and `log rho(t,x)` jointly
  on the backward HJB and forward Fokker-Planck residuals, joined by `alpha* = -(b/r)V_x` and by the
  population mean. Both boundary conditions are structural rather than penalised: the density
  carries its initial Gaussian as an exact factor and the value carries the terminal cost evaluated
  at the network's *own* terminal mean, which is where the coupling enters the value side.
  `LQMeanFieldGame.solve` is the gate -- an exact closed form (stationary Riccati root plus a 2x2
  trace-free two-point boundary value problem) that annihilates both PDE residuals to 5e-15.
  On the monotone instance the neural solve reproduces it: control 0.057%, mean 0.21%, density 0.46%.

  The gate also prices the failure. The transition matrix turns oscillatory exactly at
  `c = 1 + r a^2/(q b^2)`, and past it the equilibrium degenerates at a horizon available in closed
  form -- `arccot(k/w)/w`, always finite whatever the terminal weight, against the monotone branch
  where it exists only when `k > lam`. Approaching it, `|S(0)|` diverges with measured pole exponent
  -0.998, and the Deep Galerkin solve's error rises 7.9x while its own residual *falls* 6.7x, so a
  residual-based stopping rule reports its cleanest convergence where the answer is worst
  (`lq_mean_field_certificate` asserts that inversion, not merely the error). Derived in
  `validation/lq_mean_field.mac` (eleven residuals, all zero), proved in `proofs/lq_mean_field.v`
  (20 theorems, Stdlib Reals only), cross-checked in `z3` and `cvc5`. Theorems doc: Result 49.

  Fixed while building it: `MeanFieldDGM.quadrature` was an inexact-array field of an `eqx.Module`,
  so `eqx.filter(model, eqx.is_inexact_array)` handed the integration nodes to the optimiser and
  they drifted 0.63 within 300 steps, silently corrupting every mean and mass. The grid is now
  derived from static scalars and cannot be a parameter.

- **`chc.residual.SpectralResidual` + the periodic plant in `chc.transport`** — `plans/18` E was
  skipped under a kill-criterion whose sole reopening condition was tying a learned spectral
  operator into `chc.transport`, so both halves land together and the criterion stays live.
  `advection_diffusion_field` / `advection_diffusion_propagator` give a translation-invariant plant
  with an exact spectral solution operator; `SpectralResidual` IS a circulant on that grid,
  parameterised by its first column (a bijection, unlike a free half-spectrum, whose imaginary parts
  at DC and Nyquist are an unidentified gauge). What it buys over `LipschitzResidual`: its operator
  norm `max_k |lambda_k|` is ATTAINED on a named Fourier mode rather than bounded -- measured ratio
  1.000000, against a generic input's 0.607 and a Schur bound measured 113x slack -- and gains
  multiply exactly under composition, so the Result 28/30 rollout tube is tight rather than merely
  valid (the product-of-norms bound is 32.3x larger on this plant's own two operators). It also
  beats an MLP with 130x more parameters by ten orders of magnitude on held-out one-step error, and
  is translation-equivariant to machine precision where the MLP is off by 0.50 -- a structural gap
  no further training closes. Two findings recorded rather than hidden: fitted by Adam on the MLP's
  own budget the circulant LOSES, because its kernel entries are O(nu n^2 / L^2) = 134.8 away from a
  small initialisation and 400 steps at lr 0.02 travel 8 -- the right estimator is the closed-form
  per-mode least squares in `fit_spectral_residual`, since a circulant is linear in its kernel; and
  the Nyquist bin of a first derivative must be zeroed on an even grid, which is not a patch but the
  correct discrete answer, since the sampled derivative of `(-1)^j` vanishes everywhere. Derived in
  `validation/spectral_circulant.mac`, machine-checked in `proofs/spectral_circulant.v`, with the
  circulant matvec cross-checked against a dense product and the existing `toeplitz_matvec`
  embedding. `chc.toeplitz` gains `circulant_symbol` / `circulant_matvec` /
  `circulant_operator_norm`.

- **Convection-diffusion in `chc.galerkin`** — the module solved only `-u'' = f`, a symmetric
  positive-definite operator where testing with the trial space is optimal by Céa's lemma. The whole
  point of a Petrov-Galerkin method is the case where that fails. `convection_diffusion_1d` adds the
  advection term, whose element integrals are ANTISYMMETRIC (`+-s/2`), so the matrix is no longer
  SPD; `convection_diffusion_exact` is the analytic boundary-layer solution; `optimal_upwind` is the
  nodally-exact SUPG parameter `coth(Pe) - 1/Pe`; and `convection_diffusion_certificate` exhibits
  the whole dichotomy. Above the cell Péclet number `Pe = s*h/(2*eps) = 1` the discrete amplification
  `(1+Pe)/(1-Pe)` turns negative while the exact `exp(2*Pe)` never does, which forces consecutive
  nodal differences to ALTERNATE -- proved as a sign statement, not observed in a plot. Full
  upwinding is monotone but amplifies by exactly `1 + 2*Pe`, the first two terms of `exp(2*Pe)`,
  hence first-order (measured slope 0.959); the optimal parameter is nodally exact. That last
  measurement is reported in ULPs of the working dtype, because the absolute error is 6.05e-9 under
  float32 and 1.07e-16 under float64 and a threshold tuned to either would silently pass at the
  other. Derived in `validation/convection_diffusion.mac`, machine-checked in
  `proofs/convection_diffusion.v`, cross-checked as `unsat` by both z3 and cvc5.

- **`chc.symbolic`** — the extraction `RBFKANLayer` was already promising. Its docstring advertised
  each edge as "an extractable 1D curve (interpretable)" with no API behind it, and an
  interpretability claim with no way to exercise it is not a feature. `kan_edge` returns the exact
  scalar edge map (reconstruction tested to 1e-12), `extract_symbolic_edge` fits it against a
  nine-function library by EXHAUSTIVE best-subset (a greedy path can lock in a wrong first term --
  `sin z` and `z - z^3/6` are close on a short range), and `symbolic_extraction_certificate` plants
  a known formula and recovers it. Two structural facts the API now states rather than assumes: an
  edge's intercept is a GAUGE, since a constant moves freely between an edge and the bias, so only
  the total is identified and centring makes the decomposition unique; and a single layer represents
  only additively separable functions, which the mixed second difference turns into a PROVED error
  floor `sup|F - A| >= |mixed F|/4`, equal to `r^2` for `x*y` on `[-r,r]^2` -- measured 9.72 against
  the proved 9.00. The extracted formula also extrapolates where the layer cannot: outside the grid
  the RBFs have decayed and the layer degenerates to its silu term, 34.65 error against the
  formula's 4.22e-4. Derived in `validation/symbolic_kan.mac`, machine-checked in
  `proofs/symbolic_kan.v`.

- **`conjugate_time_certificate`** (`chc.regret`) — bounds the horizon on which every other constant
  in this module is valid. `confounded_turnpike_certificate` reads as "a long horizon is benign", and
  that is true only for a POSITIVE-DEFINITE stage cost; its hypothesis was never priced. Under an
  indefinite one -- a state that is rewarded rather than penalised, as a growth or market-share
  objective is -- the reverse-time Riccati solution is a uniform rotation of its phase and escapes at
  `t_conj = (pi/2 + phi0)/mu` with `mu = sqrt(-(a^2 + (b^2/r) q))`. Three objects blow up at three
  DIFFERENT orders there: the cost-to-go has a simple pole with residue `-r/b^2` (free of `a`, `q`
  and the terminal weight), the gain sensitivity `L_K` a DOUBLE one, and the regret constant
  `C ~ L_K^2` a fourth-order one -- so inspecting the cost alone understates the obstruction by two
  orders. And `d t_conj/db < 0`: more control authority moves the obstruction EARLIER. The certificate
  carries a positive-definite arm where the same algebra runs with `tanh` and nothing diverges.
  Derived in `validation/conjugate_time.mac`, machine-checked in `proofs/conjugate_time.v`.

- **`ce_explicit_constant_certificate`** (`chc.regret`) — computes the two constants every earlier C1
  statement took as a hypothesis. `proofs/c2_end_to_end.v` universally quantifies over an arbitrary
  `0 <= cc` in `regret <= cc*||dB||^2` and cites Mania-Tu-Recht's LOCAL quadratic bound; nothing
  computed the ball on which a certainty-equivalent gain stabilises the TRUE plant, and nothing
  computed `cc`. The lever is that the Lyapunov increment is an EXACT perfect square in the gain
  error, `Q + K'RK' + (A-BK')'P(A-BK') - P = (K'-K)'R_K(K'-K)` with `R_K = R + B'PB`, for every gain
  and with no smallness assumed -- so the Mania-Tu-Recht citation leaves the regret half entirely and
  stays only as a comparison point. Summing it along the perturbed loop gives
  `rho = theta/(2 beta_B L_K)` and `C = 2 kappa_P ||R_K|| L_K^2 ||x0||^2 / theta` with
  `theta = 1 - sqrt(1 - eta)`, `eta = lmin(Q + K'RK)/lmax(P)`. Inside `rho` the controller provably
  stabilises with the checkable Lyapunov certificate `(A-BKhat)'P(A-BKhat) <= (1-theta/2)^2 P`; the
  sweep runs past `rho` on purpose and shows the controller really does destabilise there. Derived in
  `validation/ce_explicit_constants.mac`, machine-checked in `proofs/ce_explicit_constants.v`.

- **`cluster_fold_leakage_certificate`** (`chc.regret`) — prices what a *violated* cross-fitting
  assumption costs on a clustered design, which none of the cited theorems do: CCDDHNR assume i.i.d.
  rows, Hansen–Lee assume independent cluster scores as a primitive, Chiang–Kato–Ma–Sasaki assume
  folds are already cluster-level. Splitting folds by row rather than by cluster is **not** a bias —
  Frisch–Waugh–Lovell cancels for any fold assignment — it silently substitutes a within-cluster
  estimator whose variance obeys `Psi = c(m,K)*(1 - rho_ICC)` with `c(m,2) = m*(m+14)/(m+2)^2`, so
  every constant derived from the cluster-robust variance belongs to a different estimator and a
  sandwich computed after row folds under-covers. For a cluster-*measurable* exposure — which is what
  a partial-interference spillover is — the channel is annihilated outright and the estimate loses
  exactly that coefficient. Derived in `validation/cluster_fold_leakage.mac`, machine-checked in
  `proofs/cluster_fold_leakage.v`.

- **`minimax_exploration_certificate`** (`chc.regret`) — the sequential exploration lower bound as an
  infimum over **policies**, not over a schedule class, with the constant written out:
  `c_causal = 2·A·|du*/dθ|·σ/√η_exp`. `adaptive_exploration_certificate` bounded an assumed
  `1/√t` family and carried its numerator as an opaque `K`; replacing the schedule template with the
  conditional-variance identity `E[(u_t − u*)²] = Var(u_t | F_{t−1}) + (E[u_t | F_{t−1}] − u*)²`,
  which holds for every policy, removes the template and names `K = A·(du*/dθ)²·σ²`. The bound is
  taken over a `T^{-1/4}` neighbourhood; a `T^{-1/2}` one carries prior information of order `T` and
  is vacuous. Two things follow from making it tight rather than merely valid: a front-loaded design
  **attains** the floor (to 4e-5), and the best `1/√t` taper sits at exactly `√2` above it, so the
  taper is optimal only when a per-round action cap forbids the burst. Derived in
  `validation/minimax_exploration.mac`, machine-checked in `proofs/minimax_exploration.v`.

### Changed

- **A gradient-learned delay was built, measured, and *not shipped* -- it fails its own
  kill-criterion.** `plans/24` F proposed a `DelayResidual` carrying `tau = softplus(alpha)`, to be
  kept only if it (a) recovered `tau` inside `chc.irf.delay_estimate`'s interval **and** (b) beat an
  unconstrained lag-`m` residual at equal parameter count. It passes (b) and fails (a).

  *One design note first, because it removes the usual machinery.* The interpolation trick these
  methods use (arXiv 2304.01329) exists to differentiate through a **discrete** history buffer.
  `DelayedDynamics` has no discrete buffer: `tau` enters as the chain rate `stages/tau`, which is
  already smooth, so the learnable version is one scalar and no interpolation at all.

  *(b) passes, and not narrowly.* On a nonlinear delayed plant, a structured arm (62 parameters, one
  of them `tau`) reaches test rollout MSE `5.9e-4` against `1.04e-3`-`2.91e-3` for an unconstrained
  arm that sees the whole buffer at a matched 65 parameters -- 1.8x to 4.9x better. The structured
  arm is also **flat in the nominal delay** (`5.83`-`5.94e-4` across a 4x range of it) where the
  unconstrained arm degrades 2.8x once the nominal is wrong. The inductive bias is real.

  *(a) fails, for two independent reasons.* On an actuation-delay plant where both routes estimate
  the *same* `tau`, `delay_estimate` returns `0.9995` with interval `[0.9991, 1.0003]` -- 0.12% of
  `tau` wide. The gradient recovers `1.0399`/`1.0406` from initial guesses `0.4`/`1.0`, a **4% bias**
  that no seed averages away, and `3.6594` from an initial `2.2` -- a **local minimum** at 3.7x the
  truth carrying 2.3x the training loss. So it is dominated as an estimator: biased where the
  statistical route is exact, basin-dependent, and offering a point where the other offers an
  interval. (Notably the *state*-delay variant showed no local minima across a 8x range of
  initialisations; the basin problem appears when the delay moves to the actuation path.)

  *The one gap that would reopen it.* `delay_estimate` reads `d x_{t+h} / d u_t`, so it sees
  **actuation** delay only; asked about a plant whose delay sits on the state-feedback path it
  correctly returns `0` with `censored=True`. The chain's `tau` has no such restriction. A learned
  delay earns its place if and only if the target is a state delay, where there is no statistical
  competitor -- not as a second way to estimate an actuation delay.

- **`chc.irf` accepts array-likes, which is what it always did.** `_projection_design`,
  `local_projection_irf`, `delay_estimate` and `structured_irf` annotated `data` as
  `dict[str, Array]` while their bodies only ever call `jnp.asarray` on the values. `dict` is
  invariant in its value type, so a perfectly valid `{"x": np.diff(...), "u": ...}` was rejected.
  Widened to `Mapping[str, ArrayLike]`, which is both the real contract and covariant.

- **The solver budget is now a cap the stopping rule can reach, not a bill paid per step** --
  `steps` defaults to `10_000` in `projected_gradient_control`, `pessimistic_control` and
  `causal_plan`, and `inner_steps` to `10_000` in every `chc.benchmark` control task. This is the
  answer to "should `lbfgs_box_control` replace the projected gradient at the call sites", and the
  answer is no: the measured defect was **under-solving, not the algorithm**.

  *The evidence.* On the instances the control benchmarks actually solve, the old 300-step budget
  left the *model* arms 6.3% and 7.5% above their own optimum while the *oracle* arms were within
  0.001% and 0.007%. That asymmetry sat inside every published regret -- the two things being
  compared were not equally solved. Given its own stopping rule the same projected gradient
  converges at 3973 and 5821 steps to within **0.045%** of L-BFGS-B, which is 60-110x slower in
  wall clock and cannot be compiled at all (SciPy crosses the Python boundary on every iteration,
  which would end `chc.mpc` as a real-time loop). So the reference solver stays the reference and
  the workhorse gets a budget it can finish in.

  *What made a loose cap safe.* The outer loop is a `while_loop` rather than a fixed-length `scan`,
  with the cost history written into a preallocated buffer, so the descent stops exactly where the
  Python `break` did. An instance that converges in 6 steps costs the same at a cap of 300 and of
  12000 (0.40 ms vs 0.40 ms; under the scan it was 0.59 -> 4.01 ms). No benchmark solve reaches the
  new cap -- the largest uses 5821 of 10000 -- so no published number is budget-dependent any more.

  *What moved, and why it is the right direction.* Solved properly, the **greedy baseline gets
  worse**: it exploits the learned model further, so its regret on the true plant rises
  (`causaldyn-bench` D-control `support-shift/greedy` 5.557 -> 6.799). Truncation had been
  regularising the baseline by accident. Every conclusion keeps its sign and the margins widen: the
  12-seed `run_multiseed` gate was re-run and every CHC controller's regret CI is still disjoint
  from its baseline's -- support-shift pessimistic 2.40 [2.34, 2.45] vs greedy 6.80, up from 5.56.
  `chc.mpc` is the deliberate exception and keeps `inner_steps = 40`: there the budget is a
  per-decision latency choice, priced in its docstring at 0.3-0.4% of closed-loop cost for 1.5-2.4x
  less time, because a warm start hands each replan a nearly-optimal iterate.

- **Three more per-call `jit` caches hoisted to module level** (`chc.epidemic`, `chc.meanfield`,
  `chc.transport`) -- the same defect as `pessimistic_control`, found by auditing for the pattern
  rather than waiting for it to resurface. Measured as three identical back-to-back calls, where a
  cache that never survives shows up as a second call no cheaper than the first: `optimal_npi`
  397 -> 143 ms, `MeanFieldControl.plan` 1160 -> 430 ms, `MeanFieldTransport.plan` 372 -> 83 ms.
  The two planners are frozen dataclasses of scalars, hence hashable, hence legal static arguments;
  `epidemic_cost` and its gradient are jitted where they are defined. Bit-identical -- these are
  the same jitted functions with a cache that now outlives the call. `chc.games` was audited too
  and does *not* recompile, so it is left alone.

- **Both projected-gradient solvers now run inside one compiled program** (`chc.control`,
  `chc.support`) -- nested `while_loop`s, the outer over the descent steps and the inner over the
  backtracking, with the cost history written into a preallocated buffer. Signatures, semantics and
  the `1 + accepted steps` history length are unchanged; only the trim to the accepted prefix still
  happens on the host. (This entry first shipped a fixed-length `scan` with a `done` flag, which is
  *equivalent* to the Python `break` -- line-search failure is deterministic in the iterate -- but
  still paid for every skipped step. The entry above replaced it, because only a `while_loop` makes
  an unused step free, and that is what a loose default budget needs.)

  *Verified against the loop it replaced, not asserted.* `tests/test_control.py` and
  `tests/test_support.py` keep a plain Python oracle -- deliberately not imported from `chc`, since
  an oracle sharing the implementation under test cannot detect it changing -- and check agreement
  in **ULP of float64** across all nine `chc.residual` backends plus the penalised objective. Worst
  gap **19 ULP** over 200 outer steps; accepted-step counts identical everywhere, including an
  instance that converges in 6 of 400 steps. On the Milestone-J LQ instance the shipped path is now
  bit-identical to the Rust binary as well (`runtime/parity_check.py`, worst gap `0.00e+00`).

  *Not bit-identical at float32, and the difference is visible downstream.* One fused program does
  not reduce in the same order as separately-dispatched calls, so at float32 the iterate drifts by a
  rounding step and, over a couple of hundred iterations, the answer moves in its last few digits.
  Measured, not inferred: on `causaldyn-bench`'s D-control and D-planner instances the old and new
  solvers agree to all 16 printed digits at float64 and differ in the 4th significant figure of a
  *regret* at float32 (2.4177742 -> 2.4169483, 1.58024e-3 -> 1.57833e-3) -- a regret being a
  difference of costs, which amplifies the underlying ~1e-5 relative shift. Nothing semantic
  changed; the float64 oracle test is what pins that down, which is why it is the gate.

  *Measured, same script before and after, idle machine.* Warm solve, 200 steps, MLP residual:
  `projected_gradient_control` **106 ms -> 30 ms** (3.5x), `pessimistic_control` **943 ms -> 50 ms**
  (18.9x). Cold, compilation included: 828 -> 716 ms and 1259 -> 954 ms, so there is no
  cold-versus-warm trade-off to weigh -- the fused program compiles cheaper than the three separate
  jits it replaced. (Figures re-taken on the shipped `while_loop`; the superseded `scan` was 26 and
  48 ms warm, 651 and 944 cold -- faster on this instance, because a 200-step cap it can never
  exceed is the one case a fixed-length scan is built for.) Milestone J's `chc.control` arm falls
  from **114 ms to 4.5 ms**, which changes its margin (Rust 3.28 ms, so 1.38x) without changing its
  verdict.

- **`pessimistic_control` recompiled its augmented objective on every call** (`chc.support`) --
  `eqx.filter_jit` caches on the wrapped function object, and the jitted objective, its gradient and
  the task cost were built *inside* the function body, so each call got an empty cache. Three
  identical back-to-back solves cost 1259 / 959 / 943 ms: the second call was not cheaper than the
  first, which is the signature of a cache that never survives. The compiled kernel is now a
  module-level function whose captured values are arguments; the same three solves cost
  954 / 53 / 50 ms. Every `chc.benchmark` task that sweeps a penalty weight paid this per solve.

- **`nlp_solver_certificate` made a precision claim in the wrong unit** (`chc.control`) -- it
  asserted `worst_lbfgs_stationarity < 1e-3`, an absolute threshold on a residual whose floor is
  set by the working dtype. It passed at float64 (1.15e-04) and reported `ok=False` at float32
  (5.03e-03) for no reason but the arithmetic, and no test caught it because `conftest.py` forces
  float64. The claim being made is comparative, so it is now stated comparatively:
  `least_stationarity_ratio > 10`, the projected gradient's stationarity over L-BFGS-B's, which
  holds at both precisions (24x at float32, 242x at float64).

- **The `R = 0.01` instance in `nlp_solver_certificate` was labelled "ill-conditioned"** when its
  4.7% gap sits between the certificate's own thresholds (`> 5%` and `< 0.5%`). It is the
  intermediate case and now says so. A previous changelog entry claimed this rename had been made;
  `git log -S` shows the string was never committed, so the claim was wrong and this is the fix.

### Fixed

- **The sdist shipped whatever was lying in the working tree, and could not build a wheel.**
  `[tool.hatch.build.targets.sdist]` was absent, so hatchling fell back to "everything `.gitignore`
  does not exclude" -- and `.gitignore` named `.venv/` while the local environment was `.venv311`,
  outside the pattern. `.hypothesis/` was never listed at all. Built locally, the 0.4.0 sdist came to
  **226 MB** across 6 076 stray files, and `uv build` then failed to produce a wheel from it at all
  (`symlink path ... is absolute, but external symlinks are not allowed` -- the venv's interpreter
  symlink). CI never saw this, because a fresh checkout has neither directory, which is exactly why it
  survived two releases: the published 0.3.0 sdist is 1.07 MB and correct. The fix is an **allow-list**
  rather than another ignore pattern, so the artefact no longer depends on the state of a contributor's
  tree -- the sdist carries the Rocq proofs and Maxima derivations the docstrings cite, and it must be
  reproducible. Contents reproduce the published 0.3.0 sdist exactly, plus the new `justfile`; the
  0.4.0 sdist is 1.33 MB and the wheel builds. `.gitignore` gained `.venv*/` and `.hypothesis/` too,
  since the mismatch was a real hole in it.

- **Three Maxima derivations had never run to completion, and nothing checked.** `maxima -b` exits 0
  after a parse error -- and still echoes the batch filename on the way out -- so neither the exit code
  nor the last output line detects an aborted batch. CI compiles every `proofs/*.v` on each push but ran
  no `validation/*.mac`, so the CAS half of a result was verified once by hand and never again.
  `constrained_ce_regret.mac` (dead since `b503cdf`, 3 of 12 `print`s ran) and `clustered_van_trees.mac`
  (dead since `ce39593`, 0 of 5 ran) both contained `du*/db` inside a `/* */` comment, whose `*/` closed
  the comment early; `confounded_turnpike.mac` hit EOF on `limit`'s "Is `|g|-1` positive, negative or
  zero?" for want of an `assume`. All three fixed and re-run under Maxima 5.50: **every previously
  published formula is confirmed** -- see the provenance notes on Results 13, 14 and 25 in
  `discoveries/theorems.md`. The newly executing part of `constrained_ce_regret.mac` added STEPs 4a-4g:
  the active set is the bounded interval `[b^-, b^+]` with `b^-*b^+ = rr`, not a half-line, and the
  interior sensitivity's zero `b = sqrt(rr)` sits strictly inside it, so both thresholds are genuine
  kinks with opposite-signed inactive-side slopes.
- **`validation/run_all.sh`** runs all 56 derivations in ~4 s and greps the output for
  `incorrect syntax`, Maxima's `-- an error.` banner, a Lisp error or a dropped `MAXIMA>` prompt.
  Mutation-tested: reintroducing the original `clustered_van_trees.mac` defect makes it exit 1.

- **`regret_scaling` and `interference_regret_certificate` silently conditioned on the stabilising
  event** (`chc.regret`); both now return an `infinite_fraction` array, so `RegretCurve` has a fourth
  field. Each `continue`d past draws where the perturbed model is unstabilisable, and dropped draws
  whose gain fails to stabilise the TRUE plant via an `np.isfinite` filter. Regret is `+inf` on that
  event, so `E[R]` does not exist and the reported `exponent` was a quantity conditional on the
  complement -- presented as if it were unconditional. The share is now reported rather than deleted.
  `ce_explicit_constant_certificate` gives the explicit radius inside which it is 0 by construction.

- **The C2 certificates did not implement assumption A8** (`chc.regret`), so the values returned by
  `multichannel_control_certificate`, `end_to_end_c2_certificate` and `clustered_lower_bound_certificate`
  have all moved. A8 asks for `K >= 2` folds of **whole clusters**; all three built folds as
  `np.mod(np.arange(n), 2)` — row parity — while the cluster id sat one line above, unused, leaving
  every cluster in both folds. `validation/clustered_rate_check.R` had the identical construction, so
  the independent-stack cross-check reproduced the same fold rather than catching it. Folds are now
  `np.mod(cid, 2)`. No order and no qualitative conclusion changed (the leak is not a bias); the
  measured constants did: full-orth slope 3.53 -> 3.82, cluster-SE -0.58 -> -0.55, G-sweep
  -0.89 -> -1.08, the `G^{-1}` plateau `c0` 0.18 -> 0.19, and the R cross-check -0.502 -> -0.541.

- **`adaptive_exploration_certificate` ignored its `sigma` argument** (`chc.regret`), so its returned
  `lower_bound`, `schedule` and cumulative-regret arrays are all different now. The van-Trees floor
  numerator is `K = C·σ²` with `C = A·(du*/db)²`; the function computed `C` under the name `coeff`
  and consumed it where `K` belongs, silently pinning `σ² = 1` while its signature advertised
  `sigma=0.5`. The cause was a name rather than a missing multiplication — the two quantities are
  spelled apart in `validation/adaptive_exploration.mac` but the Rocq file calls the numerator `C`,
  and the code followed the Rocq spelling while implementing the Maxima quantity. They are now named
  apart in the code too (`c_curv` vs `k_vt`). Nothing symbolic changed: the derivation and the proofs
  were already correct. Passing `sigma=1.0` reproduces the old output exactly. A test asserts the
  bound is linear in `σ`, since `ruff` does not flag an unused keyword argument.

## [0.3.0] — 2026-08-01

### Added

- **Named data enters as a mapping, a pandas frame or a polars frame** (`chc.frames`). The
  estimators (`chc.estimators`) and the g-methods (`chc.gmethods`) used to spread whatever they were
  handed with `{**data}` / `for name in data`. That reads a pandas frame correctly and a polars one
  **silently wrong**: `dict(frame)` and iteration both yield polars' columns as *values*, so code
  keying on column names received column data and nothing raised. `as_columns` now normalises once
  at the entry point and recognises a frame structurally — `.columns` plus `frame[name]` — so
  neither library becomes a dependency of the wheel. A mapping passes through **unconverted**, which
  keeps the library's own JAX arrays on the device and leaves the entry points usable under
  `jax.jit`; only the frame branch materialises, and it stops at NumPy so the caller keeps deciding
  precision.

- **`ControlAffineResidual.closed_loop_jacobian(x, u)`** (`chc.residual`) — `∂(a_θ + B_θ u)/∂x`, the
  linearisation an MPC horizon actually follows. `drift_jacobian` is the drift at `u = 0` and its
  docstring claimed a non-negative eigenvalue meant "the plant runs away on its own"; that holds only
  at `degree = 0`. With a state-dependent channel the fitted class is closed under an affine change
  of actuator coordinates `u = alpha v + beta` with `a -> a + beta b1`, so the drift spectrum reports
  where the actuator's units put their zero rather than whether the plant decays. Found on BOPTEST: a
  zone whose actuator is a setpoint in `[15, 25] °C` read `+6.42` from the drift and `-1.40` from the
  closed loop at the setpoint it actually held. The claim is corrected; the stability question to ask
  is `sup` over the admissible action set, which for an affine decay is attained at a box endpoint.
  Derivation and cross-checks in `validation/actuator_reparametrisation.{mac,py}`.

- **Causal identification of the residual's control channel** (`chc.dynamics_id`). Until now every
  residual in the library was fitted by prediction error, which under a confounded logging policy
  learns the **observational** control response — measured, the trained channel is `0.02` where the
  truth is `1.0`, and the planner inherits that. `fit_causal_residual` estimates it instead from the
  Robinson moment lifted from a scalar effect to a state-dependent matrix, with cross-fitting:
  channel error `0.002`. Control payoff on the same plant: regret `0.014` against the
  prediction-error fit's `6.41`. When the confounder is never logged, a 2SLS variant identifies the
  channel from an exogenous action shifter instead — consistent, but at a **variance premium** worth
  stating rather than burying: `0.10` channel error and `0.13` regret, because the shifter explains
  only ~18% of the action's variance.
  `solve_channel_moment` exposes the moment alone, so `g` and `m` may come from any learner rather
  than only the built-in ridge-polynomial default. With neither an adjustment set nor an instrument
  the fit reports `identified=False` and points at `chc.sensitivity` instead of returning a confident
  wrong answer.
- **`ControlAffineResidual`** (`chc.residual`) — `r_θ(x, u) = a_θ(x) + B_θ(x) u`, the plant class
  `chc.plan.certify_safety` and `chc.spine` already assume. Restricting identification to it puts the
  identification and safety layers on the same object: `control_channel(x)` is bit-identical to
  `∂r/∂u`, which is what `certify_safety` differentiates out. A general nonlinear residual has no
  partialling-out moment and gets no guarantee from this estimator.
- **`CausalDynamicsTask`** (`chc.benchmark`) — the leaderboard row, with `oracle` / `causal-id` /
  `causal-iv` / `mse-id`. Its honest trap is that the failure is **silent on the safety columns**: the
  attenuated channel makes the biased planner under-actuate (`max|u|` 0.22 against the oracle's 2.49),
  so it never touches the box or leaves the logged action support and reports `viol = ood = 0` while
  conceding most of the achievable improvement. Regret is the only column that sees it.
- **Two ways to read the half of the fit that is *not* identified causally.**
  `ControlAffineResidual.drift_jacobian(x)` returns `∂a_θ/∂x`, the companion to the existing
  `control_channel(x)`, and `CausalDynamicsFit.drift_error` carries the drift stage's own
  homoskedastic OLS scale beside `channel_error`. Both exist because of a failure on a real building
  emulator (`causaldyn-bench` Track D-causal), where three of four arms came back with a **positive**
  thermal pole: a trending outdoor temperature, present in `adjust_for` but absent from the drift's
  own regressors, gets charged to positive feedback in the state, and the fitted plant heats itself.
  An MPC plans on `a_θ(x) + B_θ(x) u` and only the second term is interventional, so a drift with a
  non-negative eigenvalue describes a plant that runs away on its own and no channel accuracy
  rescues the horizon. `drift_error` is deliberately a *different* object from `channel_error` —
  conditional on the fitted channel, whose uncertainty it does not propagate, and homoskedastic: a
  scale, not coverage.

### Changed

- **`CausalPlan` no longer certifies a plan it never checked.** With no error model supplied
  (`model_error` left at its default) the Gronwall tube is identically zero, and the old code reported
  that as `certified_horizon == horizon` — a full-horizon safety pass obtained by proving nothing.
  `uncertainty_tube` and `certified_horizon` are now `None` in that case, a new
  `CausalPlan.certificate_status` property distinguishes `"not_evaluated"` from `"uncertified"` /
  `"partial"` / `"certified"`, and `certified_actions` raises rather than handing back the whole
  sequence via a `None` slice. **Breaking** for any caller that indexed those two fields
  unconditionally. `causal_plan` also rejects a negative `model_error`, which is not an error budget.
- **The three safety modes are named apart** in `chc.plan`'s documentation and in the README, because
  the word "safety" was doing the work of all three: **plan** (`causal_plan` — box constraints in the
  solve), **audit** (`certify_safety` — read-only on a finished plan), **filter**
  (`robust_safety_filter` — the only one that changes an action). Stated plainly: no barrier or tube
  enters the objective, so `causal_plan` can return a plan that fails its own audit. A
  state-constrained (CBF-QP) solve does not exist yet.
- CI gates the formal claims too: a `proofs` job compiles all 37 `proofs/*.v` under Rocq 9.2, pinned
  because the proofs use the post-rename `From Stdlib Require Import`.

### Fixed

- **The nuisance stage is now scale-invariant, and was silently not.** `_cross_fit_residuals` built
  its degree-2 polynomial basis on the caller's raw covariates, so `ridge=1e-6` penalised a column of
  order 1 and its square of order 400 by the same absolute amount, and the Gram inherited whatever
  units the caller happened to log in. Found on a real building emulator (`causaldyn-bench` Track
  D-causal), where the zone enters in Celsius at ~21 beside already-standardised weather: condition
  number **1.4e11**, past what float32 carries, and the fit returned `nan` — while the same rows in
  float64 fitted fine, which is what identified it as conditioning rather than data. Standardising
  the covariates first drops the condition number to order 10 — on a reproducible stand-in with that
  geometry (768 rows, zone at 21 ± 0.9 beside three standardised columns) the degree-2 Gram goes
  from `2.7e10` to `2.4e1` — makes float32 reproduce float64 to 4 decimals, and makes the estimate
  *exactly* invariant to the units and origin of the adjustment set. Two things this exposed are
  worth stating: reverting the fix, an offset of `1e5` returns a **finite** channel wrong by 1.34
  against a baseline error of 0.009 — silently wrong is the worse failure — and `tests/conftest.py`
  enables `jax_enable_x64`, so **no test in this suite can fail from float32 conditioning**, which is
  how the defect survived all of them.
- `fit_causal_residual`'s `folds` documentation no longer claims that own-sample residualisation
  reintroduces bias. Measured across ten regimes, `folds=1` is never worse than cross-fitting and at
  small `N` is 5–10× better: residualising `y` and `u` by the same linear projection whose span
  contains both nuisances is Frisch–Waugh–Lovell, so it is exactly unbiased and out-of-fold prediction
  only adds variance. Cross-fitting is insurance against nuisance learners that are adaptive to the
  sample or saturated (1-NN including the row itself sends the channel to exactly zero), and the
  tests now assert that case rather than the false one.
- `HybridDynamics`'s docstring said causal design "will live" in the residual's feature map. It does
  now, in `chc.dynamics_id`, and the docstring points there.

## [0.2.0] — 2026-07-29

Work landed on `main` since `v0.1.0`. The theme is **guarantees**: most of it is a machine-checked
result line at the causal↔control seam (Maxima derivation → Rocq proof → numeric certificate), with the
matching runtime primitives shipped alongside. The proof scripts themselves are in `proofs/` and the
symbolic derivations in `validation/`.

### Added

- **Sensitivity-aware control under hidden confounding** (`chc.sensitivity`, a facade over
  `chc.regret` + `chc.uncertainty`). Bounded-density-ratio (marginal MSM) worst-case effect as a CVaR
  mixture → pessimism-radius inflation (`confounding_robust_inflation`, `msm_worst_case_mean`,
  `confounding_robust_radius`); the confounding regret floor is *second order* in the effect bias
  (`confounding_robust_lq_regret`, plus a matrix Frobenius lift); a **minimax controller** whose gain
  the radius shifts under asymmetric over/under-shoot loss (`confounding_robust_control`, sign
  dichotomy, piecewise improvement gap); the radius inside the replanning tube
  (`confounding_robust_closed_loop_bound`). Grounded on a synthetic observational confounded
  marketplace, then lifted into a genuine receding-horizon **closed loop** on a confounded plant
  (`confounding_robust_tracking_loop`, `confounding_robust_tracking_benchmark`).
- **`ConfoundingRobustPenalty`** (`chc.uncertainty`) — a `PenaltyModel` carrying the sensitivity radius
  into the general pessimistic-control stack (`radius·Σ‖u_t‖`, from the *transition*-error bound
  `‖Δ_B·u_t‖ ≤ radius·‖u_t‖`; an identification-radius regulariser rather than a certified cost bound,
  since the latter needs a cost-to-go Lipschitz multiplier `lam_unc` currently absorbs), and **`ConfoundingRobustTask`** (`chc.benchmark`), its leaderboard row: under a
  *hidden* confounder no estimator can help, and the radius still cuts regret ~40% vs
  certainty-equivalence with separated multi-seed CIs.
- **Certified planning** (`chc.uncertainty`, `chc.residual`) — certified-Lipschitz rollout-error tubes
  via discrete Grönwall feeding the pessimism radius, with time-varying tubes, constraint tightening,
  a certified-safe horizon and a closed-loop (replanning) variant; **`ContractiveResidual`** with a
  certified negative log-norm, which replaces the `e^{LT}` growth with a bounded radius; a
  **port-Hamiltonian** residual with a machine-checked damping-injection Lyapunov certificate;
  **`WassersteinPenalty`**, a W1-DRO distribution-shift margin.
- **`NestedCVaRPenalty`** (`chc.uncertainty`) — a time-consistent aggregation of ensemble
  disagreement in the same `PenaltyModel` slot. `EnsembleUncertainty` *sums* member variance along the
  trajectory, so one very bad step averages away against many quiet ones; this replaces the sum with
  `rho_t = c_t + CVaR_alpha[rho_{t+1}]`, and `static_penalty_trajectory` keeps the other adversary
  (commit to one member for the whole horizon) so the two are comparable. `nested_risk_certificate`
  checks the ordering that must hold — nested ≥ static ≥ risk-neutral, collapsing at `alpha = 1` — and
  the gap is the price of time consistency, which is what a receding-horizon controller needs if it is
  not to chase its own tail across re-solves. Scoped honestly: with the members re-evaluated
  independently the recursion collapses to `Sum_t CVaR_alpha[c_t]`, a risk-averse *aggregation* rule
  rather than a dynamic-programming solve of a nested-risk MDP.
- **`chc.pathway`** — one `causal_pathway(target)` API over the temporal causal graph, with
  Rocq-certified walk-sum / geometric-truncation / weakest-link structural laws.
- **Marketplace layer** — `chc.matching` (Kantorovich OT dispatch with dual surge prices) and
  `chc.marketplace` (offline causal control under equilibrium interference, where naive and MOPO-style
  baselines go negative); influence-function standard errors and CIs on the cross-fit DML effect.
- **Certified strategic layer** (`chc.games`) — `fixed_point`, a differentiable equilibrium solver that
  iterates to a *relative* residual and returns an `EquilibriumSolution` carrying `residual` and
  `converged`, with the backward pass as the implicit-function VJP (`jax.custom_vjp`, adjoint solved as
  its own fixed point) rather than an unrolled loop; `congestion_contraction_modulus` /
  `congestion_damping` / `congestion_contraction_certificate` to certify or refuse a configuration
  before it runs (`spec(J) ⊆ [0, ½]` sharp ⟹ the uniform Jacobian bound is below 1 iff
  `0 < d < 4/(2+κ)` — sharp over the class of congestion maps, sufficient for any one game — with
  `d* = 4/(4+κ)` certifying every `κ`); `equilibrium_transfer_certificate`, which measures the equilibrium's
  *local* conditioning — exactly 1 in the ambient norm uniformly in `κ` (attained only along the mass
  direction that mass conservation never excites) and strictly below 1 on the fixed-mass tangent space
  where displacements actually live — while the naive contraction constant `1/μ` is loose by up to
  100×. So a `C/μ²` regret bound must not take its constant from *this* solver's contraction margin.
- **`chc.barrier`** — the sensitivity radius spent on a *constraint* instead of on the objective:
  the robust control-barrier margin guaranteed against every effect in the identified set
  (`robust_barrier_margin`), its maximiser — exactly zero once the radius swallows the control
  channel, which is optimal and not conservatism (`robust_safe_action`), the closed-form certified
  action set and the least-restrictive filter that clips a nominal action into it
  (`admissible_action_interval`, `robust_safety_filter`), the sharp radius at which certification
  dies (`identification_radius_threshold`) and the sensitivity level it corresponds to
  (`barrier_gamma_star`) — *the largest sensitivity-model level under which the barrier stays
  certified* (a model parameter, not a measured amount of hidden confounding). Both
  thresholds are **case-split** rather than one formula: a self-satisfying drift gives `inf`, a
  deficit beyond what a perfectly identified channel delivers gives an empty set, and `Δ(Γ)`
  saturating at the CVaR gap gives `Γ* = inf`. The measured consequence: safety margin degrades at
  **first** order in the effect bias — until the radius swallows the channel, past which the loss
  saturates at `U·|g|` — while performance regret degrades at second, because the envelope theorem
  protects an interior optimum and not a binding constraint. `safety_filter_benchmark` pays that argument in a closed loop on an
  unstable plant whose reference sits past the limit: the radius spent on the constraint never
  leaves the safe set, the same-sized budget spent the regret way violates on 93% of steps.
  Re-exported through `chc.sensitivity`.
- **`chc.plan.causal_plan`** — the one-call spine: a `CausalPlan` carrying the actions *together with*
  the certified error tube and the `certified_actions` prefix, so a caller cannot take the plan and
  leave the certificate behind. With no safety arguments it is exactly `projected_gradient_control`;
  each safety argument switches on one existing layer, and an uncertainty penalty without a support
  model is rejected rather than silently ignored.
- **`chc.plan.certify_safety`** — the other half of the spine: §40 evaluated along a finished plan, so
  the safety result has a consumer instead of staying an orphan primitive. Returns the certified
  *prefix* (does the action the planner chose still clear the barrier at this `Γ`?) next to the plan's
  `Γ*` (could **any** admissible action have, and up to which level?) — two questions that come apart
  in both directions and are diagnostic together. The plan-level `Γ*` is the minimum over steps and is
  *attained*: `Δ(Γ)` is increasing, so each step certifies on a down-set and the plan certifies on
  their intersection; a step that certifies at no `Γ` empties it and reports `nan` rather than being
  skipped. A separate call on a finished plan rather than a planner argument — it audits, it does not
  silently change the actions. Re-exported through `chc.sensitivity`.
- **`chc.spine`** (`scripts/spine_demo.py`) — the four layers on one decision, because until now every
  layer had its own demo and none of them ran end to end. Two zones of a mobile driver pool, one
  incentive lever whose `[+b, -b]` column *is* driver conservation, and a supply floor in the zone the
  lever drains: an effect fitted from confounded logs (naively and with the backdoor adjustment), a
  constrained plan with its Grönwall tube, a `certify_safety` audit of that plan, and finally the same
  plan executed on the **true** plant so the offline numbers can be checked against what happened. The
  confounded arm plans a cost of 13.6 and pays 38.5; `Γ*` separates the two arms *before either acts*
  (7.46 vs 1.18) with no access to ground truth. Deliberately control-affine — `certify_safety` reads
  the channel off the Jacobian at `u = 0`, which is exact for an affine plant and only a linearisation
  otherwise, so the softmax-equilibrium market of `chc.marketplace` is the wrong plant to certify.
- **`chc.reachability`** — a Hamilton–Jacobi backward reachable tube whose adversary is the §32
  identification radius: `V(x,T) = max_u min_{ΔB} min_s h(ξ(s))` by Lax–Friedrichs on a 2-D grid, with
  the same robust-margin algebra as `chc.barrier` but `p = ∇V` solved for rather than assumed. It
  exists to price the barrier certificate, and `barrier_reachability_gap` is that price:
  `valid_cbf` makes the CBF theorem executable (condition on all of `{h ≥ 0}` ⟹ the tube **is**
  `{h ≥ 0}`, checked to a cell), and `certified_but_unreachable` measures where the *pointwise* §40
  check certifies a state no controller can hold — on a relative-degree-2 barrier the §40 verdict is
  identical at every radius while the true tube shrinks. Verified against two analytic solutions (a
  rigidly sliding level set to `1e-5`; the double-integrator braking parabola to 100% off-boundary
  agreement), with a CFL guard that refuses rather than reporting an optimistic safe set.
- **Regret / guarantee line** — the orthogonal-to-control transfer theorem (order `p` → `2p`, scalar
  and multivariate-LQ), multi-channel network control (debias *every* channel), the adaptive
  information-exploration duality with its `√T` lower bound, the C2 end-to-end theorem with a clustered
  van-Trees lower bound and an exposure-map generalisation, plus a batch of scoped propositions and
  corollaries (doubly-robust control, H∞-as-pessimism, constrained piecewise-quadratic regret,
  confounded turnpike, transportability, ensemble heterogeneity, partial-identification sign threshold).

### Changed

- Eleven rounds of external review folded in as **scope and honesty corrections**, not new claims: the
  explicit-Euler contraction factor was wrong (`√(1+2μΔt+L²Δt²)`, sufficient step `Δt < 2c/L²`); the
  confounding effect error needed the control magnitude to be dimensionally right; the `§35`
  improvement gap is piecewise (the undershoot-dominant branch was unproved while the benchmark ran in
  it); `§32` is the bounded-density-ratio *marginal* special case of Tan's MSM, not the full model, and
  its monotonicity argument is feasible-set nesting; the confounded-marketplace benchmark is
  *observational*, not a randomised switchback. Several results were relabelled to their honest status
  (order-transfer *lemma*, local-not-global, scalar-not-universal, `≈`-not-`=`). Round eleven: the
  equilibrium conditioning of `1` is an *ambient*-norm statement saturated by the mass-conservation
  direction, and the binding number on the fixed-mass tangent space is strictly smaller; the safety
  margin loss is `U·min(d,|g|)` and so first-order only while control authority lasts; the zero-action
  rule in general reads "the identified interval contains zero", the symmetric ball being one case;
  and the `§38` premium is quoted with both normalisations because they differ by an order of
  magnitude.
- Documentation counts corrected after an audit found a silently drifting entry count in the research
  log; the README test count was stale by two releases.

### Fixed

- `ConfoundingRobustPenalty` used `‖u‖`, whose gradient is NaN at `u = 0` — exactly where the solver
  starts — so `0·NaN` poisoned every step and the control stayed pinned at zero. Now a smoothed
  `√(‖u‖²+ε)`, which zeroes the gradient at the origin and preserves the linear bound.
- The congestion equilibrium ran a fixed trip count and returned whatever point it reached. Outside
  the contraction region that is a 2-cycle, not an equilibrium: at `βc = 200` it returned a point with
  residual 3.21 on a mass of 6.0 and most of the mass on the *least* attractive zone, and no caller
  could tell. The solver now reports convergence; the root cause was the hard-coded damping `d = ½`,
  whose ceiling is `βc < 6`, so `congestion_damping` supplies the modulus-optimal `d*` that contracts
  for every `βc`. Both shipped users (`chc.marketplace`, causaldyn-bench Track H) sit at `βc = 5`,
  inside the certified region, and their docstrings now say so executably.
- The convergence tolerance was absolute while the residual carries the units of the state, so a
  float32 solve at the shipped mass could not reach `1e-8` and reported failure at a residual of
  3.8e-07. The residual is now relative (`‖x−T(x)‖ / max(1, ‖x‖)`).

### Notes

The repository moved to the `causaldyn` organisation and is public as of this release; all project
URLs point there. This is the **first version published to PyPI** — `0.1.0` below was prepared and
tagged locally, then overtaken by 71 commits before it was ever pushed, so it never reached an index.

## [0.1.0] — 2026-07-19

Prepared but never published; superseded by `0.2.0`. `chc` is a small JAX library that fuses
**physics-structured hybrid dynamics**, **causal identification**, and **constrained optimal control**,
made safe on offline/confounded data by an explicit **pessimism/support** layer — evaluating decisions
as interventions, not correlations.

### Added

- **Hybrid dynamics & sensitivity** — additive `f_known + r_θ` (`chc.dynamics`); MLP / RBF-KAN / graph /
  zero residual backends (`chc.residual`); RK4 rollout (`chc.integrate`); a hand-written discrete adjoint
  verified against autodiff and finite differences, plus an adaptive `diffrax` continuous adjoint
  (`chc.adjoint`); Lie–Trotter / Strang–Marchuk operator splitting (`chc.splitting`).
- **Causal frontier** — pluggable effect estimators (backdoor OLS, IV/2SLS, cross-fitted Double ML,
  sensitivity + refutation, VanderWeele–Ding **E-values**) behind one Strategy interface
  (`chc.causal`, `chc.estimators`), with lazy EconML/DoWhy adapters; the **R-learner** CATE meta-learner;
  staggered-adoption **DiD** — Callaway–Sant'Anna group-time ATT and the de Chaisemartin–d'Haultfoeuille
  DID_M (`chc.did`); **augmented synthetic control** (`chc.scm`); Robins **g-methods** for time-varying
  treatment (`chc.gmethods`); autocorrelation-robust CI testing + lagged-parent **discovery**
  (`chc.independence`, `chc.discovery`); network/interference-aware DML with a learned GNN nuisance
  (`chc.network_causal`).
- **Offline safety & guarantees** — density-distance support penalty (`chc.support`) plus **calibrated**
  deep-ensemble + split-conformal uncertainty (`chc.uncertainty`); IPS/SNIPS off-policy value + overlap
  gate (`chc.offpolicy`); an **LQ certainty-equivalence regret bound** and a local nonlinear certificate
  via linearisation (`chc.regret`, `chc.lqr.linearized_regret_certificate`).
- **Control** — projected-gradient optimal control and receding-horizon MPC (`chc.control`, `chc.mpc`);
  LQR / Riccati (`chc.lqr`); Koopman-LQR (`chc.koopman`); mean-field control (`chc.meanfield`); optimal
  transport (`chc.transport`); differentiable Stackelberg games (`chc.games`); PMP time-optimal bang-bang
  (`chc.mintime`); classical step-response metrics (`chc.metrics`).
- **Structured operators & dynamic effects** — Toeplitz FFT matvec, Levinson–Durbin, Gohberg–Semencul
  inverse + few-sample covariance (`chc.toeplitz`); impulse-response / local-projection dynamic effects
  (`chc.irf`); Galerkin FEM and a mesh-free Deep Galerkin Poisson solver (`chc.galerkin`,
  `chc.deep_galerkin`).
- **Benchmark & real-data validation** — oracle-regret tasks (pricing, inventory, support-shift,
  model-uncertainty) with a leaderboard and multi-seed bootstrap CIs (`chc.benchmark`); a causal-methods
  leaderboard vs naive baselines (`chc.causal_bench`); the **LaLonde-Dehejia-Wahba** external causal
  benchmark (`chc.lalonde`, recovers the randomised ATE from CPS-confounded data). The control loop is
  validated live on a real **BOPTEST** `bestest_hydronic_heat_pump` emulator (via the `causaldyn-bench`
  sibling), where the identification + forecast-MPC beat the tuned built-in baseline on every KPI.
- **Tooling** — `src`-layout, `uv`-managed, `py.typed`; `ruff` + astral `ty` gates; CI test matrix on
  Python 3.12 / 3.13 / 3.14.

[0.11.0]: https://github.com/causaldyn/causal-hybrid-control/releases/tag/v0.11.0
[0.10.0]: https://github.com/causaldyn/causal-hybrid-control/releases/tag/v0.10.0
[0.9.0]: https://github.com/causaldyn/causal-hybrid-control/releases/tag/v0.9.0
[0.8.0]: https://github.com/causaldyn/causal-hybrid-control/releases/tag/v0.8.0
[0.7.0]: https://github.com/causaldyn/causal-hybrid-control/releases/tag/v0.7.0
[0.6.0]: https://github.com/causaldyn/causal-hybrid-control/releases/tag/v0.6.0
[0.5.1]: https://github.com/causaldyn/causal-hybrid-control/releases/tag/v0.5.1
[0.5.0]: https://github.com/causaldyn/causal-hybrid-control/releases/tag/v0.5.0
[0.4.0]: https://github.com/causaldyn/causal-hybrid-control/releases/tag/v0.4.0
[0.3.0]: https://github.com/causaldyn/causal-hybrid-control/releases/tag/v0.3.0
[0.2.0]: https://github.com/causaldyn/causal-hybrid-control/releases/tag/v0.2.0
[0.1.0]: https://github.com/causaldyn/causal-hybrid-control/releases/tag/v0.1.0
