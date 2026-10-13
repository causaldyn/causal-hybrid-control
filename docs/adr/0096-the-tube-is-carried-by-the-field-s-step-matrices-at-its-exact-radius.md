# ADR 0096 — The tube is carried by the field's step matrices, at its exact radius

**Status:** accepted, 2026-10-13. Amends the tube of `prescribe` and `causal_plan` (ADR 0080) and
its rate error (ADR 0087, ADR 0091).

## Context

`prescribe`'s tube has two parts (ADR 0080): a one-standard-error scale of the fitted field's
error, and the radius within which the plant's own noise leaves the plan at each step with
probability 0.95. Both are carried through the plan's RK4 steps. Three things in that carriage
were loose, and one was missing.

1. **The rate.** `_rate` returns a bound on the norm of the field's slope in the state over the
   levers' box, even on a field affine in the state, where the tube is `global`. `causal_plan`
   already had an exact path for a state matrix, but `prescribe` never passed one: its fitted
   channel is affine in the state, so the slope moves with the action, and no one matrix is the
   slope at every step. On the quickstart's fitted field the slope's eigenvalues are -0.525 and
   +0.262, and the norm bound is 0.631. RK4 grows the tube by `R(0.0631) = 1.065` a week, where
   the field's own step matrix grows it by 1.027 at most. A bound on the norm is also no
   equivariant reading of the states: with one state in thousandths of its units the same field's
   bound reads 165, and the growth 4027 a week (`validation/exact_tube.mac` (3) shows the step
   matrices move as `D M D^-1`, and the covariance as `D P D`).
2. **The radius.** `sqrt(chi2_n(0.95) lambda_max(P_k))` holds a Gaussian of covariance `P_k` with
   probability at least 0.95: exactly on one state, and up to `sqrt(chi2_n / chi2_1)` too wide where
   one direction's variance dominates, 1.249 times on two states. At the quickstart's first week the
   noise's covariance has eigenvalues 1.0e-4 and 2.1e-2, and the radius held the plant at 0.986.
3. **The drift's error.** The tube's rate error was the channel's alone, priced at each lever's
   distance from the log's mean action (ADR 0091). ADR 0091 named what it left out: the drift's own
   error at that action. On the quickstart's log it is 0.039, where the channel's part is 0.016:
   the demand the fit adjusts for and the plan does not forecast stays in the drift's residual. On
   100 quickstart logs the noise-free plant left the plan after one week by up to 4 times the
   channel's part. The norm bound's slack had hidden it, by growing the tube at the field's
   steepest rate.

So the quickstart trusted 1 week where its plant stays within the tolerance of 0.5 for 3, measured
on 20,000 runs (ADR 0080).

## Decision

### 1. On a field affine in the state, the tube is carried by the slope at each planned action

Where `_rate` names the field `global`, `prescribe` reads the field's slope in the state at each of
the plan's actions, `A_k`, by differentiating the fitted field there, and carries both parts of the
tube by RK4's step matrix of each at its step:

- the field's error: step `j`'s error reaches step `k` through
  `R(dt A_{k-1}) ... R(dt A_{j+1}) c_i(dt A_j)`, and the tube sums those gains' norms times each
  step's error;
- the noise's part: `P_{k+1} = M_k P_k M_k' + M_k C_k + C_k' M_k' + S`,
  `C_{k+1} = rho (M_k C_k + S)`, `M_k = R(dt A_k)`.

`validation/exact_tube.mac` checks the second against the double sum of the gap's covariance on
three distinct symbolic matrices (1), and the first against two RK4 rollouts on a scalar field
whose slope changes each step (2). On a field whose slope moves with the state, `local`, the norm
bound read at the start stays.

The tube moves no action, so `prescribe` solves through `causal_plan` as before and then carries
the finished plan's tube along the slopes at its actions. `causal_plan`'s signature does not
change, which `RecedingHorizon` mirrors argument for argument, and `linear_rollout_bound` still
takes one matrix. A sequence of one matrix repeated gives that matrix's tube to the bit, and so a
model whose slope does not move with the action, a `LinearDynamics`, gets `causal_plan`'s matrix
tube to the bit through `prescribe`'s slope reader.

### 2. The radius is the exact quantile where the covariance is exact

Where the gap's covariance is exact, on every step of a state matrix or of the slopes along the
plan, and at the first step of a norm bound, where it is the noise itself, the radius is the root of
the 0.95 quantile of `sum_i lambda_i chi2_1`, `lambda_i` the covariance's eigenvalues: Imhof's
inversion, already in the library for the misspecification test and now in `chc.uncertainty`,
inverted by Brent's method on the eigenvalues over the largest. The quantile lies between
`chi2_1(0.95) lambda_max` and `chi2_n(0.95) lambda_max`, a bracket that holds it. On one state and
on equal eigenvalues the sum is `lambda chi2_n`, and the radius is the old one to the bit: the
BOPTEST case's one-state cells do not move by it. Past the first step of a norm bound the
recursion bounds the largest eigenvalue alone, and the radius stays `chi2_n`'s.
`validation/exact_tube.mac` (4) checks the closed forms at one, two and three equal terms, and that
the library's radius holds the Gaussian with probability 0.95 to 1.4e-11 at three unequal pairs.

### 3. The field's error carries the drift's own

The rate error is `sqrt(e_channel^2 + drift_error^2)`: the channel's part as ADR 0087 and ADR 0091
price it, and the fit's `drift_error`, the standard error of the drift regression's fitted value at
the log's states. The drift absorbed the channel at the log's mean action, so the field's error at
an action is the drift's error there plus the channel's times the action's distance from it. Read
at that centre the two are uncorrelated to first order, as an intercept read at a regressor's mean
is with its slope, and so add in quadrature. A fit with no channel error carries no tube, as
before, whatever its drift's error.

### 4. The certificate says what each radius is

- `DecisionCertificate.tube_mode`: `exact` where the tube was carried by the slopes along the
  plan, so the noise's radius is its exact 0.95 quantile at each step under the fitted field;
  `bound` where by the norm bound, so it holds at least 0.95, exactly at the first step. In both
  the field's error is a one-standard-error scale beside it. None where no tube was evaluated.
- `DecisionCertificate.tube_state_noise`, under `exact`: each state's own 0.95 radius of the
  noise at each step, `sqrt(chi2_1(0.95) P_k[i, i])`, in that state's units. A report, not the
  tube: the tolerance is still one Euclidean radius over the states, read in the caller's units,
  as ADR 0091 part 3 documents (the author's call, 2026-10-13). The field's error scale is not
  split by state.
- Both go into the report and `to_json()`. No field changes meaning.

## Consequences

- **The quickstart trusts 3 weeks, its plant's own 3.** Its tube reads 0.290, 0.401, 0.482 for
  weeks 1 to 3, where it read 0.359, 0.528, 0.670, and ends at 0.881 where it ended at 2.372. On
  100 fresh quickstart logs, the plant run on from the plan's start under its actions 4,000 times
  each:

  | tube | inside at weeks 1 to 6 | trusts 1, 2, 3 weeks (logs) |
  |---|---|---|
  | the norm bound, the channel's error (before) | 0.986 to 0.9995 | 100, 0, 0 |
  | the step matrices, the channel's error, `chi2_2` | 0.986 to 0.987 | 85, 15, 0 |
  | the step matrices, the channel's error, exact radius | 0.951 to 0.953 | 0, 4, 96 |
  | the step matrices, the channel's and the drift's error, exact radius (now) | 0.953 to 0.959 | 0, 25, 75 |
  | the same, the two errors added | 0.954 to 0.961 | 0, 42, 58 |

  The plant's error stays within 0.5 at 95 % for 3 weeks, pooled over the logs, and no log is
  trusted past that. Without the drift's error the tube missed what the fit leaves in the drift:
  the noise-free plant, run under the plan, left it after one week by more than the channel's
  part on 71 of the 100 logs, and by more than the field's part, the drift's included, on 19.
- **A bilinear one-state plant under the plan's own switching actions** (`x' = -0.4 x +
  (0.8 + 0.3 x) u + 0.2 z`, one unit of 960 half-hours, 100 logs, 4,000 draws each): inside at
  0.957 to 0.986 over 12 steps, where the norm bound held 0.961 to 1.000. It trusts 1 step on 68
  logs and 2 on 32, where the norm bound trusted 1 on every log; one log of 100 is trusted a step
  past what its own plant allows, its fitted channel 2.3 standard errors off, where none was.
- **On a plant the model is,** under actions that switch its slope from growing to contracting,
  the tube holds the plant at 0.95 within four Monte Carlo standard errors at every step, on one
  state and on two, persistent noise included (`tests/test_exact_tube.py`). The two-state test of
  ADR 0080, which held 0.979 to 0.986, now holds 0.95 within four standard errors.
- **Notebook 09's heater,** its eight certified hours unchanged, ends its tube at 0.434 °C where it
  ended at 0.641.
- **The BOPTEST case's one-state logs** (600 one-unit logs, six cells, each run on 200 times from
  its last row, the first call): the tube one step ahead is unchanged, 0.196 to 0.263 K by cell,
  and the plant inside it there on 0.961 to 0.989 of draws, as before. The mean trusted steps move
  from 1.56 to 2.54 by cell to 1.59 to 2.71, the most from 3 to 4. The logs trusted past their own
  plant's horizon rise by one in the cell `one, 0.9`, 2 to 3, and by two in `slab, 0.9`, 0 to 2;
  of the three, two have a tube within 0.03 K of the plant's 95 % error at the step that crosses
  the tolerance, and the third a plant the fit already missed at the first step, 0.555 inside,
  where its noise, persistent at 0.9, goes on from the log's last value. The tube is a marginal
  statement over logs, and a one-standard-error scale of the field's error: the norm bound's
  slack had covered these.
- **The silent-failure track of the graph-error benchmark** reads the same, report for report, on
  50 replicates of each of its nine worlds: no plan there is trusted for a step at its tolerance of
  0.05 since the tube holds the plant's noise (ADR 0080), with this change or without it.
- **The media-mix report's prescription** is still trusted for no week: its tube reads 2.50, 2.96,
  3.13 at weeks 1 to 3 against the tolerance of 1.0, where it read 3.88, 13.06, 41.90; the fitted
  field contracts, which the norm bound of 1.25 a week did not see. **The pendulum demo's** field,
  whose slope moves with the state, keeps the norm bound: its tube one step ahead reads 0.052 where
  it read 0.063, the exact radius at the first step, and grows a little faster past it with the
  drift's error; it is trusted for 1 step, as before.
- **Cost.** Each step of a tube on two or more states under `exact` inverts Imhof's integral by
  Brent's method, a few dozen quadratures. `RecedingHorizon` with a state matrix pays it at every
  re-plan. No timing is claimed here.
- **Not covered.** `drift_error` is the drift regression's homoskedastic standard error at the
  log's rows, conditional on the fitted channel: it does not grow where the plan leaves the log's
  states, and it does not read clusters or persistence as `channel_error` does. Each step holds the
  plant at its rate, not every step at once (ADR 0080). The field's error is still a scale, added to
  the noise's radius by the triangle inequality.
- Tests: `tests/test_exact_tube.py`, `tests/test_predictive_tube.py`;
  `validation/exact_tube.mac`.

## Alternatives

- **Add the two errors** rather than their squares. A bound on any correlation between them, but
  at the log's mean action they are uncorrelated to first order, and on the quickstart's logs the
  sum lost a week on 42 % of logs where the quadrature lost it on 25 %, while both held the plant
  above 0.95 at every week.
- **The drift's standard error at the plan's own states,** from the fit's joint covariance of the
  drift and the channel (`influence`). It would grow where the plan leaves the log. But it needs
  the per-row influence, `N x n x p`, which `prescribe` does not keep, and it is a new reading of
  the fit with its own coverage evidence to gather.
- **A constant channel** (`channel_degree=0`): one slope at every action, one matrix. It changes
  the fit every caller gets, to serve the tube.
- **A Monte Carlo quantile.** Random, and needs many draws to settle the third digit; Imhof's is
  deterministic and accurate to 1e-11 here.
- **The exact radius past the first step of a norm bound.** The recursion there bounds the largest
  eigenvalue alone; no covariance is known to take a quantile of.
- **A `causal_plan` argument for the slopes.** Every caller of `causal_plan` could then carry a
  tube along a moving slope, but `RecedingHorizon` takes every argument `causal_plan` does, and a
  public argument is a door hard to close; the one caller that needs it is `prescribe`.
- **A tolerance per state.** The author kept the one Euclidean tolerance (ADR 0091 part 3); the
  states' own radii are reported beside it.
