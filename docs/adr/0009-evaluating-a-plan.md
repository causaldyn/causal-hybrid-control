# ADR 0009 — Evaluating a plan from logs, certified before any cost is read

**Status:** proposed, 2026-09-28.

## Context

`chc.offpolicy.off_policy_value` weights one step at a time, so it estimates a candidate's value on
the logger's own states. For a feedback plan on a plant with memory, that is not the value of
deploying it. On a loop where the candidate spreads the state past twice the logger's variance,
SNIPS converges to -2.92 against a deployed value of -5.76, and its overlap flag reads True
(`test_the_one_step_estimand_is_not_the_deployed_value`). Nothing in the library could say whether
a log can evaluate a deployed plan at all, let alone evaluate it.

The lab worked out when a log can, on a linear-Gaussian loop. An independent verifier reproduced
every number and corrected two statements:

- **Trajectory weights.** The second moment of the H-step importance weight obeys an exact
  risk-sensitive Riccati recursion along a "starred" loop. So the horizon it certifies can be
  computed before any data is read. The small-gain condition first stated for "finite at every
  horizon" is wrong as stated, for two reasons:
  - it omits the initial-state law: a loop that passes it is infinite from `H = 6` when it starts
    from the logger's stationary law;
  - for matrices it needs an H-infinity norm, not a spectral radius.
- **Stationary weights.** Marginalised importance sampling is feasible iff
  `2 Sigma_b^z - Sigma_pi^z > 0` on the joint state-action laws. That condition splits into a
  state part and an action part. The one-step gate is neither necessary nor sufficient for it, and
  there is a loop for each direction.
- **Smoothing.** A deterministic plan has no density. Weights therefore need a smoothed plan,
  `N(K x + k, tau^2 I)`, which is feasible on an interval `(0, tau_max)`. Smoothing raises the
  average cost by exactly `tau^2 tr(R + B' P B)`. Subtracting the model's value of that term
  reopens a first-order model error, `tau^2 (beta - beta_hat)`.
- **Fitted Q evaluation** is exact on this class with quadratic features. A restricted chi-square
  certifies it where every weight is infinite.
- **Two smoothing rules make two loggers look opposite.** In the lab, MIS kept `tau` near 0.1 and
  PDIS took `tau ≈ 0.83 sigma`. Under those rules MIS and PDIS "preferred opposite loggers", and
  the PDIS estimate was 92 % model at `sigma = 2`. At a common `tau` the opposition disappears.

What the solution had to keep:

1. **The certificate reads no cost.** It is computed from the model and the logger, before the
   estimate. An estimate the certificate refuses is not returned.
2. **The value reported is the deployed plan's.** A smoothed plan's value is never passed off as a
   deterministic one's with only the estimator's error attached.
3. **What the model supplied is named** in every estimate.
4. **`off_policy_value` keeps computing what it did**, the one-step estimand.

## Decision

- **Scope: a linear-Gaussian loop and an affine plan.**
  - `LinearGaussianPlant(a, b, offset, noise)` is one step, `x' = a x + b u + offset + w`, with
    `w ~ N(0, noise)`.
  - `AffinePolicy(gain, offset, covariance)` is `u | x ~ N(gain x + offset, covariance)`. It is
    deterministic when the covariance is zero.
  - A receding-horizon LQ controller whose constraints do not bind is such a plan. The fitted
    plant enters linearised, and everything here is exact on the linearisation and no more.
- **`certify_evaluation(plant, logger, plan, method, samples, ...)`** returns an
  `EvaluationCertificate` from the model and the two policies alone. Per method:
  - `"pdis"`: `log E_b[W_h^2]` for every `h` up to the logged horizon. It is computed by the
    recursion started from the episodes' own initial law (`InitialLaw`). The certificate reports
    the certifiable horizon `H* = max{h : n / E_b[W_h^2] >= min_effective}` and the first `h` at
    which the moment is infinite, if any. The recursion is checked in the tests against an
    independent route that integrates the whole trajectory's Gaussian quadratic form in one step.
  - `"mis"` and `"dr"`: the state and action margins of the stationary condition, `tau_max`, and
    `log(1 + chi^2)` of the stationary weight at the smoothing used.
  - `"fqe"`: the restricted chi-square over the quadratic features.
  - For every method, the one-step margin, for comparison and nothing else.
  - `ci_reliable`: whether the fourth Rényi moment is finite, so that the interval's variance
    estimate is consistent.
- **`evaluate_plan(logs, plan, method, *, plant, cost, logger=None, ...)`** certifies first. When
  the certificate refuses, it raises `InfeasibleEvaluation`, which names the failing condition and
  carries the certificate. Otherwise it returns a `PlanEvaluation` with:
  - the value and its interval;
  - the certificate;
  - the model's correction, and the share of the weighted estimate it removed;
  - the weights' own effective sample size, beside the predicted one.

  The stationary methods estimate the average cost per step. `"pdis"` estimates the expected cost
  over the logged horizon from the episodes' initial states, with each step's weights normalised
  to mean one.
- **Which plan is evaluated.**
  - A randomised plan is evaluated as it is.
  - A deterministic plan is evaluated through a smoothed one. The model's `tau^2 beta_hat` is
    subtracted **and added to the interval**, times a stated `model_error`. The default is 1: the
    correction is counted as unverified.
  - FQE evaluates a deterministic plan directly.
- **One smoothing rule for every weighted method.** `tau` minimises the predicted mean squared
  error of the reported value, `(model_error tau^2 beta)^2 + variance / n`, among the `tau` the
  certificate passes. The variance is exact for the stationary methods. For `"pdis"` it is
  approximated by the weights' own. `certify_evaluation`, which reads no cost, defaults to the
  `tau` with the most effective samples. `evaluate_plan` passes its own.
- **The logger** is fitted by least squares with a full covariance unless the caller passes the one
  that logged, which is better (`fit_behavior_policy`'s docstring says by how much).
- **Named, not built: a claim about every horizon.** An evaluation only needs the logged horizon,
  where the recursion is exact. "Finite at every horizon" needs the bounded-real norm plus a
  decision on its boundary, where the moment is finite but grows like `e^{lambda H} H^{-1/2}`, and
  no estimate asks for it.

## Consequences

- The verifier's loops are tests, each against its published number:
  - CE1 escapes at `h = 6` with a one-step margin of 55/52;
  - CE3's moment is 8.134660321708 at `H = 60`;
  - the two-state loop certifies `H* = 7, 15, 23, 31` at `n = 10^3 ... 10^6`;
  - E1-E3 cover both directions of the one-step gate;
  - M1 escapes at `h = 4`, and M2 stays finite through `h = 3000`.
- TR4's loop is refused by `"mis"` and `"dr"`, with the stationary state margin
  (`2 R_x - P_x = -1.34`) as the reason. `"fqe"` evaluates it: its restricted chi-square is 3.30,
  and its interval covers the deployed value, not the one-step estimand.
- Coverage over 500 replicates of the two-sided market loop (`scripts/bench_evaluation.py`,
  nominal 0.95, Monte Carlo SE 0.010), with the model either true or fitted by least squares to
  each replicate's own logs:

  | method | logs | `model_error = 0`, true / fitted | `model_error = 1`, true / fitted |
  |---|---|---|---|
  | `"mis"` | one run of 4000 transitions | 0.962 / 0.940 | 0.978 / 0.970 |
  | `"dr"` | one run of 4000 transitions | 0.942 / 0.942 | 0.974 / 0.972 |
  | `"fqe"` | one run of 4000 transitions | 0.942 / 0.942 | — |
  | `"pdis"` | 3000 episodes of 5 steps | 0.968 / 0.968 | 1.000 / 1.000 |

  - With the model's correction trusted, every arm is within two points of nominal, and every
    mean error is within two standard errors of zero.
  - The stationary interval is Student's `t` over 40 batch means with Satterthwaite's degrees of
    freedom, `(sum_k s_k)^2 / sum_k s_k^2` over the batches' sums of squares. A fixed `t_39`
    covered as well here, where the weights spread over every batch. It did not on
    `MountainCarContinuous-v0` linearised at its valley floor (`bench_evaluation.py valley`),
    logged by a lightly damped operator and evaluated for an LQR plan whose states are ten times
    narrower than the logs'. There the weight sits in a few of the logger's passes through the
    floor, and the median degrees of freedom are 6 (`"dr"`) and 8 (`"mis"`), from 1.5 to 14.
    Over 500 replicates at `model_error = 0`, true / fitted:

    | method | `t_39` | Satterthwaite |
    |---|---|---|
    | `"dr"` | 0.914 / 0.916 | 0.950 / 0.954 |
    | `"mis"` | 0.898 / 0.890 | 0.932 / 0.930 |

    The fitted model was not the cause: the true one undercovered alike. `"mis"` is still at the
    edge of two points; its mean error is 1.6 standard errors above zero, and a symmetric
    interval does not see a skewed one.
  - The default counts the whole correction as unverified, and over-covers because this model is
    right. The correction is larger than it looks: at `model_error = 0` the rule smooths more, and
    the correction is 23 % of the stationary estimate and 42 % of the episodic one; at the default
    it is 2 % and 18 %, and the episodic interval is 5.4 times as wide.
  - The loop is CHC's own. The 0.8.0 gate asks for two environments CHC did not write.
- Every number is exact for the linearised loop, and the certificate sees nothing outside that
  class, for example:
  - a plant far from linear;
  - a clipped or saturating logger;
  - a logger whose randomisation is confounded with the plant noise;
  - a plan whose constraints bind.
- Stationary effective sample sizes count transitions as if they were independent.
- The intervals are fixed-n, not anytime-valid; the deployment gate is where time enters.

## Alternatives considered

- **An adapter over SCOPE-RL.** SCOPE-RL has the sequential estimators, doubly robust and DICE,
  for gym environments. It computes no certificate, and the certificate is the reason this module
  exists. SCOPE-RL stays a candidate test oracle, and adding it even as a test dependency is a
  dependency decision.
- **The one-step gate, tightened.** Refuted: the one-step margin is neither necessary nor
  sufficient for the stationary condition.
- **Reporting the smoothed plan's value** as the deployed one. Rejected by requirement 2.
- **A smoothing rule per method.** Rejected: it made MIS and PDIS disagree about loggers for a
  reason that was not about the loggers.
