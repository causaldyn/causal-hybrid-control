# ADR 0173 — A plan's value is bounded under confounding that acts one step at a time

**Status:** accepted, 2026-11-25. Research tier (`chc.research.stepwise_confounding`, new). Opt-in:
nothing outside the module reads it, and no existing value moves.

## Context

CHC's sensitivity tools bound one quantity at a time: a mean outcome at one level of a lever
(`msm_worst_case_mean`), a channel's bias (`ChannelSensitivity`), a move's value
(`omitted_confounder_bound`). A plan acts over a horizon, and its value is a sum of steps, each of
which a confounder of the logger's action can bias. Nothing bounded that sum, and nothing said at
which level of confounding a plan stops beating its alternative.

Bruns-Smith and Zhou (arXiv 2302.00662, v3, 2025) bound it where the confounder is *memoryless*:
`U_t` independent of the past given the state `S_t`. The plant seen by a plan that does not read
`U_t` is then a Markov decision process on the observed state (their Proposition 1), and each step's
law under the plan is the logged one reweighted by `w = e(a|s) / e(a|s, U)`, `E[w | s, a] = 1`
(their Theorem 1). Under Tan's marginal sensitivity model at level `Λ` (their Assumption 3), `w`
lies in `[e + (1 - e)/Λ, e + (1 - e) Λ]`, and the least expectation of `Y = R + V(S')` over that box
is a superquantile (their Proposition 5): `α E[Y] + (1 - α) LTM(Y)`, the lowest `1/(Λ + 1)` of the
law's mean, at their `τ = Λ/(1 + Λ)`.

That is CHC's MSM. `msm_worst_case_mean(propensity=e)` reads the same box at the same level, and
`benchmark_gamma` calibrates it, so their `Λ` is CHC's `Γ`, with no conversion.

## Decision

- **A new research module, `chc.research.stepwise_confounding`.** `stepwise_robust_value(logs,
  plan, gamma)` returns the plan's lower, nominal and upper values; `stepwise_breakdown(logs,
  plan, threshold= | baseline=)` returns `Γ*`, the least `Γ` at which the lower value falls below
  a threshold or below a baseline plan's nominal value on the same log. Research, not experimental:
  it carries one result, on a log class no other module reads, and `import chc` loads none of it.
- **Tabular.** States and actions are integer codes; the log is `state`, `action`, `reward`,
  `next_state`, `unit`, `period`, one entry a transition. A plan is a deterministic action
  sequence `(H,)` or a table `(H, S)`. The fit at each step is the saturated one: each cell's mean,
  each cell's empirical tails. Cells pool the log's periods, a time-homogeneous plant.
- **The parameter is `gamma`, on Tan's odds box at the logged propensity by default.** `e(a|s)` is
  the share of the state's transitions that took `a`. `weight_box="density"` takes `[1/Γ, Γ]`, the
  continuous-dose box `msm_worst_case_mean` reads without a propensity. On the odds box a reached
  state whose action the log always took is refused: a logged propensity of 1 reads no confounding,
  which Tan's model does not allow, and the density box is the reading that does not lean on it.
- **The closed form, not a solver.** Each cell's bound is `α mean + (1 - α) tail`, the tail at mass
  `1/(Γ + 1)` read with its fractional last point (`_top_tail_mean`). On one step it is
  `msm_worst_case_mean`'s knapsack at the cell's propensity, to `1e-12`.
- **The premise is checked, and a rejection refuses.** Each transition's innovation, its reward and
  one-hot next state less its cell's means, is read for a lag-1 autocorrelation within units by
  `persistence_check`'s statistic and test (`dynamics_id._persistence`). At `p <= 0.05`
  (`persistence_alpha`) both functions raise `PersistentConfounding`, which carries the check.
  Under memoryless confounding the observed process is Markov given `(S_t, A_t)`, so an innovation
  is uncorrelated with the step's before; a persistent `U` that moves the reward or the next state
  correlates them. A log with no pair to read proceeds and reports a `nan` p-value; one whose
  periods leave no pair one step apart is refused, as `persistence_check` refuses it.
- **`Γ*` by bracketing and bisection.** The lower value is continuous and non-increasing in `Γ`.
  Its limit at `Γ -> inf`, `e E[Y] + (1 - e) min Y` per cell, is computed exactly: at or above the
  comparator, `Γ* = inf`. Below it, doubling from 1 brackets the crossing and a bisection in
  `log Γ` narrows it to a relative `1e-6`. The upper end is kept, a level at which the value was
  read below the comparator (as ADR 0162 keeps the failing end). A nominal value already below the
  comparator reads `Γ* = 1`.

## Consequences

- `validation/stepwise_confounding.mac` derives the cell bound: the share at the upper weight is
  `1/(Γ + 1)` whatever `e`; on a two-point law the greedy fill and the simplex LP equal the closed
  form in both regimes, at six rational points and both sides; on a Gaussian law the weight `β`
  below the `1/(Γ + 1)` quantile and `α` above has mean 1 and attains the closed form
  `μ - (1 - α) σ φ(z)/m`, the weak dual's maximum; `Γ = 1` gives the mean; on the density box the
  upper bound is `confounding_robust_inflation`'s form. It runs under Maxima 5.49 in CI's image.
- On its two-state plant, whose logger's least MSM level is 2, the plan's true value is `65/16`
  and the population bounds at `Γ = 2` are `[31/8, 81/16]`; at `Γ = 6/5` the lower value,
  `23651/5400`, is above the truth; the nominal `41/9` misses it by `71/144`. A log whose empirical
  law is that plant's reads these to `1e-13`, the limit `49/18`, and `Γ*` against the truth,
  1.663311, to the bisection's width. A sampled log of 2000 units over 10 periods holds the truth
  inside its bounds at `Γ = 2`, within 0.03 of the population's, and excludes it at `Γ = 1.2`.
  The bisected `Γ*` against a baseline sequence lies in the bracket a grid of 300 levels reads.
- The same plant with a confounder kept from the period before with probability 0.9 is refused,
  `p < 1e-10`; the memoryless log is not.
- **Scope, as the module states it.** The bounds are plug-in: sharp on the log's empirical law,
  with none of its sampling error, which a cell of few transitions reads in full;
  `least_count` reports the fewest any cell read. They are sharp per cell where the logger has two
  actions. With more, the cells of a state are bounded apart, the paper's rectangular relaxation.
  And a time-homogeneous plant's one law may be read differently by each step's worst case: valid,
  possibly loose. A passing check does not prove the premise: it reads one lag and a correlation.

## Alternatives

- **The paper's orthogonalised estimator with confidence bounds.** Not built: it needs a
  cross-fitted conditional quantile and its bias correction for each step, and on a discrete state
  the plug-in cell bound is already the sharp bound of the empirical law. The intervals would be
  the next piece.
- **The infinite-horizon estimator (Bennett, Kallus et al., NeurIPS 2024).** Not built: CHC's plans
  have a horizon.
- **Read `persistence_check` off a fitted channel.** Rejected: it needs a `CausalDynamicsFit`, and a
  tabular log has none. The same statistic on the log's own innovations reads the same premise.
- **Warn on a rejection rather than refuse.** Rejected: under a persistent confounder the
  per-step bound has no guarantee at all, and a bound that may not hold is worse than none. The
  refusal says how to proceed: add the previous state and action to the state, the paper's
  `k`-order memoryless model, and read the augmented log.
- **A new parameter name, `lambda_`.** Rejected: CHC calls the MSM's level `Γ` everywhere, and this
  is that level on that scale.
- **An experimental module in `chc`.** Rejected for now: CHC's plan evaluation,
  `chc.evaluation`, is linear-Gaussian, and a tabular evaluator beside it would widen the scope its
  docstring states. Research is the tier that may change or be withdrawn without a cycle.
