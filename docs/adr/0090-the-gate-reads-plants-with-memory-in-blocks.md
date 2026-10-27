# ADR 0090 — The gate reads plants with memory in blocks

**Status:** proposed, 2026-10-12.

## Context

`DeploymentGate` (ADR 0010) adds one increment per decision: the decision's reward, weighted by
the candidate's propensity less the baseline's over the logged one. Its guarantee states the
condition "no carryover": a decision's reward does not depend on earlier decisions. On a plant with
memory the per-decision increment estimates the myopic contrast, what one decision of the candidate
changes, not what running the candidate changes.

The gate's own docstring records the failure. On the lab's plant, `x' = 0.9 x + 0.1 u + 0.2 eps`
with a reward of mean `0.3 exp(-(u - 1)^2 / 2) + 0.7 exp(-x^2 / 2)`, a candidate `N(0.4, 1)` beats
the baseline `N(0, 1)` by +0.029 myopically and loses by 0.0096 in the long run. The gate deployed
it on every path within 20 000 decisions (200 of 200, after a median of 1488).

The obvious route, increments over windows of one unit's history, needs the windows to be
independent, and consecutive windows of one unit are not. The design-based literature removes that
need. Bojinov and Shephard (JASA 2019) define causal effects of a time series experiment with the
potential outcomes fixed, so the only randomness is the assignment, and with an `m`-period causal
impact (their assumption in the appendix on it) the effect of a sustained assignment is identified.
Ham, Bojinov, Lindon and Tingley (Management Science 2026) build anytime-valid confidence sequences
on the same footing: inverse-probability-weighted contrasts are martingale differences under the
design, whatever the dependence of the outcomes.

## Decision

- **`GateConfig(block_length=L, burn_in=b)`** (*experimental*), unset by default, runs the gate in
  blocks. Each block of `L` periods draws one policy, the candidate with probability `rho`, runs
  it for the whole block, and logs the probability of that draw. A zone starts in EXPERIMENT,
  since shadow evidence is no evidence on such a plant.
- **`BlockBatch(reward, candidate, logged, drift=None)`** carries one zone's blocks: rewards by
  block and period, burn-in included, whether each block ran the candidate, and the logged
  probability of its draw. The gate drops the first `b` periods of each block and averages the
  rest.
- **A block is one decision whose action is the policy drawn.** The candidate's propensity is 1 on
  its blocks and 0 on the baseline's, and the logged one is `rho` or `1 - rho`, so the block passes
  through the gate's existing check of the logged propensity against the mode's, its existing
  bounded-increment e-processes, and e-BH, unchanged. The block's increment has the bound
  `1 / (1 - rho)` that the experiment's mixture already has.
- **What a DEPLOY claims.** Given everything before a block, its increment's mean is the read
  periods' mean reward under the candidate less that under the baseline, from the block's start.
  Under carryover of order `p <= b` that is the sustained contrast. Under geometric memory, a
  reward within `C a^t` of its policy's long-run value `t` periods after the policy takes over,
  it lies within `2 C a^b (1 - a^r) / (r (1 - a))` of `V_new - V_base`, and `delta` must cover
  that bias. The docstring states both, with the lab's numbers.
  *2026-10-25, ADR 0135:* with `GateConfig(check_burn_in=True)` the gate in blocks tests, in its
  experiment, whether a block's read mean moves with the block before, which carryover of order at
  most `b` rules out, and warns to raise `burn_in`; the docstring's "the gate cannot test `p` or
  `a` on its own data" is withdrawn.
- **Modes in blocks.** A futility stop sends a zone to SHADOW, which runs the baseline's blocks.
  Rule 5 sends a shadow zone back to EXPERIMENT as soon as it may enter one, without reading the
  zone's `ZonePlan`. A deployed zone runs the candidate's blocks, so its rollback e-process has no
  baseline block to read, and only a drift alarm rolls it back.
  *2026-10-27, ADR 0146:* with `GateConfig(rollback_baseline_share=q)` a deployed zone runs the
  baseline's blocks with probability `q`, and its rollback e-process reads their contrast after
  the burn-in, inverse-probability weighted; unset, as here.
- **The kinds do not mix.** A gate in blocks refuses a `ZoneBatch`, a gate not in blocks a
  `BlockBatch`, with a `TypeError`, and blocks of another length with a `ValueError`. `drift`, as
  on a `ZoneBatch`, holds one row per period.
- **Without `block_length`, nothing changes.** Every existing test passes unchanged.

## Consequences

Measured by `tests/test_gate_blocks.py` and `validation/gate_block_contrast.mac`. Verdict counts
only, no wall time.

- **The lab's plant.** With `r = 10` and no burn-in, the block's contrast is +0.014 from a start at
  0.2, the stationary mean under `rho = 0.5`, and +0.062 from one at -1.1. With `b = 40` it lies
  between -0.0115 and -0.0069 for every start in `[-1.1, 1.5]`, against -0.0096 in the long run.
  In blocks of 50 with 40 burned, read every 5 blocks, the gate deployed the candidate on 5 of
  1000 paths of 20 000 periods.
- **Carryover of order 1**, a candidate +0.45 better myopically and exactly as good sustained, read
  after every block of 2: with the first period burned, the gate deployed it on 2 of 300 paths
  within 400 blocks; with none burned, on 242.
- **Across zones**, three such zones and three better by 0.3 sustained, 200 paths of 600 blocks at
  `alpha = 0.1`: the false-discovery rate at the end was 0.0037, a false deploy happened on 1.5 %
  of paths, and 93.3 % of the better zones were deployed. With no burn-in the rate was 0.484.
- **A futility stop adds no wrong DEPLOY.** On the same draws, every zone the gate with
  `alpha_futility = 0.1` deployed, the gate without it deployed too: 1 and 2 false deploys of 600
  zone-runs. The stop ended 77.8 % of the experiments on zones at the margin, and 89.5 % of the
  better zones were deployed, against 90.5 % without it.
- **Verdicts slow down.** A block is one increment however long it is, and its burned periods read
  nothing. For a candidate +0.3 better sustained under carryover of order 1, the median periods to
  a DEPLOY were 562 in blocks of 2 with 1 burned, 639 in blocks of 3, 978 in blocks of 5 and 1404
  in blocks of 9, each with 1 burned; with 2, 4 and 8 burned in blocks one longer, 801, 1288 and
  2156. On the lab's plant, whose effects are a few hundredths, a DEPLOY of a candidate that is
  better is out of reach in 20 000 periods: the block's weighted reward has a variance near
  `E[R^2] / (rho (1 - rho))` whatever `r` is.
  *2026-10-25, ADR 0135:* with `GateConfig(centred=True)` the second moment falls to
  `Var(R) / (rho (1 - rho))` plus the contrast's square, and a candidate +0.030 better in the long
  run was deployed on 36 of 200 paths within 20 000 periods and on 196 within 100 000.
- Thirteen mutations of the change were each caught by a test.

## Alternatives considered

- **Per-decision increments with a washout.** Each increment still reads one decision's
  assignment, so its contrast is still myopic; the washout only thins the data.
- **A model-based long-run test**, LSTD per arm with a sandwich variance (Shi, Wang, Luo, Zhu, Ye
  and Song, JASA 2023). It uses every period and has more power, but it needs a Markov state and is
  asymptotic with looks fixed in advance; the gate's guarantee holds at any read.
- **Centring the block's reward** at a predictable level, which keeps the design's mean and would
  cut the variance on plants whose rewards sit far from 0. Not built: it changes the increments the
  gate's other modes share, and its gain was not measured here.
  *2026-10-25, ADR 0135:* built, opt-in, for the experiment alone, so the other modes' increments
  do not change; its gain is measured there.
- **A separate gate class for blocks.** Rejected: the block reuses every rule and e-process, and
  only how a batch becomes increments differs.
