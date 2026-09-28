# ADR 0011 — The experiment a decision needs

**Status:** proposed, 2026-09-28.

## Context

`prescribe` raises `NotIdentifiedError` when the logs cannot identify an effect, and
`chc.regret` prices the exploration a controller can do in the loop. Nothing in the library said
which experiment to run before a decision, how large, or whether to run one at all.

A power calculation answers a different question. It sizes an experiment to test the effect,
while a decision needs the effect only as far as a wrong estimate changes what it does and what
that costs. Two zones with the same noise can deserve very different sample sizes, and a zone
whose lever the optimiser never moves can still be worth testing.

The lab's value-of-information thread (VI1–VI5) worked out the local theory, and an independent
verifier checked it:

- **The decision weight.** `W = J_bu M⁻¹ J_ub`, where `M` is the Hessian of the cost in the
  action, is the Hessian of the certainty-equivalent regret in the channel estimate. A posterior
  with covariance `Σ` leaves an expected regret of `½ tr(W Σ)` to leading order.
  - On a four-zone market, three routes to `W` agree to 2e−7.
  - Monte Carlo over the prediction runs 0.96–1.02 from 50 to 3000 units a zone.
- **Allocation.** Decision-Neyman sets `n_k ∝ σ_k √(w_kk / (c_k V_k))`. With a prior it becomes
  reverse water-filling at the price of information, `c_k σ_k² / V_k`. At one budget on that
  market it leaves a regret of 1.20e−3, against 1.88e−3 for classical Neyman and 1.51e−3 for equal
  units.
- **A lever pinned at its bound** has no local weight and costs regret only past its activation
  point. What an experiment can buy there is at most `D δ² φ(z) / z³`, in units of the
  multiplier's preposterior sd `δ`, for every `z > 0`. So "a zone the optimiser never picks is not
  worth testing" is false as stated and true as a Gaussian tail.
- **At a knife edge**, where `q b² = r` in one zone, the plug-in weight is zero and the regret is
  fourth order in the prior's width (Raiffa–Schlaifer 1961).
- **Where the local model fails:**
  - on a wide prior, it put the regret after the experiment 32% low at a prior sd of 37–50% of the
    channel;
  - linearising a pinned lever's multiplier at the estimate gave 0.16–0.70 of Monte Carlo on a
    multi-lever plant.

## Decision

- **`design_experiment(decision, prior, experiment, budget, *, uses, level, draws, seed)`** returns
  an `ExperimentDesign`. It contains the whole units per zone, what they spend, the regret now and
  after, the value net of spend with its Monte Carlo error, a regret bound at `level`, the decision
  weight, and each pinned lever's distance from activation. The verdict:
  - **`"experiment"`** when the value exceeds its Monte Carlo error;
  - **`"abstain"`** when it does not and a zone that matters stays within two posterior sd of zero,
    so which way to act there is still unknown;
  - **`"deploy"`** otherwise, with the regret of acting now and its `level` quantile.
- **The decision is a `ZoneDecision`:** one-shot over `K` zones, `x = x0 + coupling · diag(b) u`,
  quadratic cost, a box on the levers. This is the class where the weight, the tail law and the
  allocation have closed forms. It is solved by bounded least squares (BVLS), whose last iterate
  is the exact solve on its free set.
- **The local model only allocates:**
  - the weight is averaged over the prior's sigma points, so a knife edge keeps the weight the
    plug-in loses;
  - a pinned lever is priced by its Gaussian tail;
  - the units start from water-filling on the diagonal of `W` and minimise `uses · regret + spend`
    under the budget, then round down to whole units without overspending.
- **Monte Carlo reports.** Every regret reported is measured by drawing channels from the prior,
  simulating the experiment, updating, and re-solving the decision on every draw. `model_gap` is
  how far the local model was from that measurement, and a gap above 25% is logged as
  `design_gap`, because the allocation was optimised on the model.
- **A channel the logs do not identify** enters as the prior its owner is prepared to state: a
  wide variance about the best guess. Turning a `NotIdentifiedError` into that prior is the
  caller's step, because only the caller knows how wide the ignorance is.

## Consequences

`scripts/bench_experiment.py` realises every experiment unit by unit, each unit's lever at
`±probe`, the channel read by least squares and the prior updated in precision form. That is a
different path to the posterior from the design's own Monte Carlo. The numbers:

- **Calibration** (4000 worlds drawn from each prior). Realised over predicted, with the realised
  mean's 95% half-width:

  | scenario | units | regret now | regret after |
  |---|---|---|---|
  | two wide zones (model gap 0.33) | 65 / 0 / 11 / 0 | 0.98 ± 0.05 | 0.96 ± 0.05 |
  | a pinned zone far from activation | 1021 / 0 / 361 / 0 | 1.03 ± 0.04 | 0.99 ± 0.04 |
  | a pinned zone near activation | 924 / 25 / 332 / 0 | 0.98 ± 0.04 | 1.00 ± 0.04 |

- **The lab's allocation comparison, reproduced** (4000 draws at the true channels, budget 2000):
  decision-Neyman 1.184e−3 ± 0.036e−3 (closed-form optimum 1.1835e−3), classical Neyman
  1.836e−3, equal units 1.527e−3, equal spend 1.606e−3. The weight's diagonal, taken by finite
  differences of the public `regret`, matches the lab's to five digits.
- **The design against the textbook rules at its own spend** (prior sd 10% of the channel, 100
  uses, 4000 common worlds). The regret after is 0.0905 ± 0.0028 for the design, against 0.1202
  for classical Neyman, 0.1164 for equal units and 0.1103 for equal spend, all at a spend of
  0.0200.
- **A wide prior** on the two uncertain zones, swept from 6% to 50% of the channel. The local
  model's gap grows from 0.009 to 0.52 and is logged from 37%. The realised regret after stays
  at 0.96–1.01 of the reported throughout.
- **Deploy verdicts** (500 worlds, no budget, so every verdict is to deploy or to abstain). How
  often acting now stayed within `regret_bound`, nominally 95%, with Clopper–Pearson intervals:

  | prior sd | truth drawn from the prior | truth fixed, estimate drawn around it |
  |---|---|---|
  | 0.05 | 0.938 (0.913–0.957) | 0.952 (0.929–0.969) over 500 deploys |
  | 0.15 | 0.954 (0.932–0.971) | 0.989 (0.974–0.996) over 454 deploys |

  The bound is a posterior quantile, so it is calibrated over the prior by construction and
  holds at a fixed truth only as it happens to; there it was conservative at 0.15. The 46
  abstentions at 0.15 are the draws whose estimate put the weakest channel (0.5) within two
  posterior sd of zero: 9.2% against the Gaussian tail's 9.2%.
- 39 tests. The two mutations that survived the first draft are now caught:
  - a Monte Carlo error from the regret now alone;
  - an error of zero.

  The other survivor was an exact re-solve of BVLS's answer, which moved it by at most 1.3e−13
  relative over 4000 random instances and is gone.

## Not built

- **The joint design for correlated channels.** The allocation water-fills the diagonal of `W`.
  The lab measured the matrix geometric mean `L⁻¹ # W` 6% better than the best independent design
  on its market, and an independent design can lose up to `K` times on a rank-one `W`.
- **The comparison on a marketplace flagship.** It waits for a zones-by-time marketplace model.
- **A dynamic decision.** The weight here is a static plan's; the dynamic analogue needs a
  decision-weighted identification of the dynamics first.

## Alternatives considered

- **A power calculation.** It optimises the wrong objective, and at one budget classical Neyman
  left 1.55 times the regret of the decision-weighted design.
- **Reporting the local model.** It is 32% low on a wide prior. Monte Carlo reports, and the model
  allocates.
- **The weight at the prior mean.** It is zero at a knife edge, where the regret is not.
- **Optimising the allocation on Monte Carlo.** A noisy objective in `K` dimensions. The local model
  is exact to leading order, and Monte Carlo then checks what it chose.
