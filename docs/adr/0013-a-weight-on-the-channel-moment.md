# ADR 0013 — A weight on the channel moment

**Status:** proposed, 2026-09-28.

## Context

`fit_causal_residual` identifies the control channel by Robinson's partialling-out, and weighs
every transition alike. When the channel class contains the truth, that costs nothing: every weight
on the state estimates the same channel. When it does not, the fit is the projection of the true
channel under the action's leftover variance times the log's law of states. A line fitted to a
curved response is the common case. The projection is taken where the log was, not where the
decision will be taken.

The lab's decision-weighted identification thread (DW1–DW4) worked this out, and an independent
verifier checked it:

- **DW1.** A weighted Robinson score stays Neyman-orthogonal iff `E[w u_res | X] = E[w eps | X] = 0`.
  Weights on the nuisances' conditioning set satisfy it. Weights that read the action or the
  outcome generally pick up a first-order bias.
- **DW2.** Under a missed class the weighted estimand is the projection under `w s^2 P`, with
  `s^2(x)` the action's leftover variance. So `w* = kappa (dQ/dP) / s^2` recovers the in-class
  regret minimiser of a one-shot decision taken at states drawn from `Q`. The weight's own
  nuisance is not orthogonalised: a spline estimate of `s^2` cost 10–17 times the oracle's excess
  regret, a correctly specified parametric one 0.95–1.2 times.
- **DW3.** The crossover sample size between two weights drops a bias term of the same order. The
  verifier's corrected expansion moves it from 1883 to 2106.
- **DW4.** A dynamic plan reads the channel's slope along its path as well as its level. There a
  pointwise weight left 36–178 times the regret of the best fit in the class.

## Decision

- **`fit_causal_residual(..., weights=None)`** takes a function of the states. It is called with
  `x (N, n)` and nothing else, and returns one finite, non-negative weight per transition, not all
  zero. The weights multiply the channel moment's instrument and nothing else: the nuisances and
  the drift regression stay unweighted. They are scaled to mean 1, so the ridge keeps its meaning.
  A callable that never sees the action or the next state makes DW1's condition structural rather
  than documented.
- **A weighted fit's standard error is robust and carries the nuisances.** Each row's squared
  residual goes through the fit's own linear map in its target (`jax.jacrev`), and that map runs
  through the cross-fitted nuisances as well as the moment. A weight that loads a few rows also
  loads the nuisances' error at them, which a sandwich on the moment alone misses. The `rk4` fixed
  point already built its error this way; the Euler fit now does too.
- **The fit says what it did.** `weighted`, and Kish's `effective_sample_size`.
- **`solve_channel_moment(..., weights=...)`** takes one weight per row, for a caller who brings
  their own nuisances.
- **The unweighted fit is unchanged** to the bit, its homoskedastic error included.

## Consequences

The tests in `tests/test_dynamics_id.py` use the lab's toy, written apart from the lab's code. It
has a scalar state, a channel `1 + x/2 + c x^2` fitted by a line, and a confounder. The action
keeps a variance `exp(x/2)` after the confounder, and the rate has a noise of variance `exp(-x/2)`.
Decisions are taken at `Q = N(-0.5, 0.25)`.

- **Orthogonality.** On exact nuisances perturbed along the covariates, the channel's error goes
  as the perturbation squared under the decision weight, slope 2.19. Under `exp(u/2)`, a
  weight that reads the action, it goes linearly, slope 0.87.
- **The estimand is the weight's.** At `c = 0.4` and 64 000 rows, three tilts and the decision
  weight land on their closed-form lines. The lines are at least 0.3 apart, and each lands within
  0.05.
- **Regret.** The decision weight's regret was 1.00 of the class's floor, `c^2 var_Q^2`, and
  the unweighted fit's 9.7 times it. With nuisances outside the polynomial sieve, the
  decision weight's regret was 1.00 of the floor, and the unweighted fit's 7.9 times it.
  - Over 100 logs of 4000 rows, the excess over the floor came to 1.16 of `tr(G V) / 2n`, the
    prediction from the decision weight's variance worked out by quadrature. Over 400 logs it came
    to 1.14 ± 0.06: at 4000 rows the line scatters 5% wider than its limit.
  - The quadrature for `V` reproduces the lab's `[2.979, 10.019]` at `c = 0`, and its `2.91 / n`.
- **Calibration.**
  - Over the same 100 logs the decision weight's reported error came to 0.96 of its spread, and
    the ones weight's to 1.02 over the first 40.
  - On a log whose noise grew as `exp(x)`, over sixteen redraws of the noise:
    - weighed by `exp(-x)`, a sandwich on the moment alone came to 0.67 of the spread, and the
      fit's own map to 1.04, and 1.02 under `rk4`;
    - weighed by `exp(x)`, the fit's own map came to 0.93 and 0.85. An error that pooled the noise
      over the rows came to 0.16 under `rk4`. Under `exp(-x)` it agreed with the robust one on
      average, so only this weight tells the two apart.
- **The inverse noise, `exp(x/2)`,** is the efficient weight when the class contains the truth: in
  the limit its error is 0.873 of the unweighted fit's.
  - Over 60 logs of 64 000 rows it scattered the channel 0.87 times as far as the unweighted fit
    (bootstrap `[0.78, 0.98]`), which is its limit.
  - Over 200 logs of 4000 rows it scattered it 1.15 times as far (`[1.00, 1.30]`). It loads the
    right tail, where the degree-4 nuisances extrapolate.
  - Its reported error came to 0.99 of its spread at both sizes.
- **The unweighted fit's homoskedastic error** came to 0.40 of its spread over 200 logs of the
  missed class. It is a shipped number, so it is left alone here and raised as a decision.
- **17 tests.** Of 16 mutations of the weighted paths, 15 are caught. The survivor drops the
  robust error's degrees-of-freedom factor `n / (n - k)`, which is 1.002 at `k = 2` and 1000 rows:
  equivalent at every size the tests can afford.

## Not built

- **A weight the library estimates**: `dQ/dP`, `s^2`, or the inverse noise. An estimated weight's
  error moves the channel at first order under a missed class, because its own nuisance is not
  orthogonalised. Each is the caller's modelling choice, and the caller's to validate.
- **A weight for a dynamic plan.** A pointwise weight left 36–178 times the best fit's regret
  (DW4). The target there is `argmin ½ g' H^-1 g`, which waits for its own simulation. The
  library's `perturbation_cost_weights` weigh the state equation's residual at a fixed plan, and
  are not an identification weight.
- **The misspecification gate (ID2):** the gap between two fits, and the crossover sample size
  from DW3's corrected expansion. Its verification needs the marketplace flagship.
- **A robust error for the unweighted fit.** Changing a shipped number is the author's decision.
- **`prescribe`** does not take a weight.

## Alternatives considered

- **`weights="efficient"`: the inverse noise, estimated from a first pass.** It was built, and
  then removed. When the class contained the truth, it scattered the channel 0.93 times as far as
  the unweighted fit at 64 000 rows, over 60 logs (95% bootstrap interval `[0.83, 1.03]`), and
  1.07 times as far at 16 000. The limit is 0.87. Under a missed class it read a projection that
  was neither the inverse noise's nor the log's: its first pass's residual carried the class's
  miss. There its spread was 1.61 times the unweighted fit's, and its reported error came to 0.83
  of that spread.
- **`weights="decision"`.** The decision weight needs `Q`, `s^2` and `kappa`, and the fit knows
  none of them. A named option would hide three modelling choices behind one string.
- **An array of weights.** An array can be computed from anything, the action included, so DW1's
  condition would be documented instead of enforced. `solve_channel_moment` takes an array, for
  the caller who brings their own nuisances.
- **A sandwich on the moment alone.** It is cheap, and it misses what the cross-fitted nuisances
  pass on from the rows the weight loads: 0.67 of the spread.
- **Changing the loss:** decision-focused learning (Elmachtoub et al.; Bennouna et al.). It
  gives up the orthogonal moment, and the confounding robustness and root-n inference for the
  channel go with it. The weight on the state keeps both, for the channel's projection. The weight
  itself gets neither.
