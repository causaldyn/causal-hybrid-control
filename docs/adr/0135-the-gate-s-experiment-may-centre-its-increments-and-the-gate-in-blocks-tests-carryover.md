# ADR 0135 — The gate's experiment may centre its increments, and the gate in blocks tests carryover

**Status:** proposed, 2026-10-25. Amends ADR 0090 (the gate reads plants with memory in blocks)
and ADR 0035 (a futility stop for the gate's experiment).

## Context

Two statements about `DeploymentGate` in blocks (ADR 0090) did not hold up.

- **"A better candidate is out of reach."** On the lab's plant, whose effects are a few
  hundredths, the block's weighted reward has a second moment near `E[R^2] / (rho (1 - rho))`
  whatever the block's length, and a DEPLOY of a candidate that is better did not come within
  20 000 periods. ADR 0090 listed centring the block's reward as an alternative: not built,
  because it would change the increments the gate's other modes share, and its gain was not
  measured.
- **"The gate cannot test `p` or `a` on its own data"**, said the module's docstring. Liu and
  Zhong (arXiv:2602.23257, 2026, Theorem 4.1) show that a switchback design whose blocks draw
  independent coins can: under no carryover past the burn-in, a block's read periods do not
  depend on the previous block's coin, and the coin is the design's.

The identity both rest on is the design's. Under the experiment's mixture the difference of the two
weights, `D = w_new - w_base`, has mean 0 given the past. Per decision
`rho w_new + (1 - rho) w_base = 1`, so `D = (w_new - 1) / (1 - rho)` and lies in
`[-1 / (1 - rho), 1 / rho]`; in blocks `D` is `1 / rho` or `-1 / (1 - rho)`.

## Decision

- **`GateConfig.centred`** (*experimental*), `False` by default, and opt-in through 1.0: no
  warning announces a later default. Set, the experiment's increments read each reward less a
  level `c` fixed before its decision, `x = D (R - c)`, which keeps the contrast as its mean,
  `E[x] = V_new - V_base`, for every such `c`. This is the doubly robust increment of
  Waudby-Smith et al. (ACM/IMS Journal of Data Science 2024), already the gate's source, with a
  reward model fitted to the earlier decisions.
  - **The level** is `c = clip((1 - rho) m_new + rho m_base, 0, 1)`, each `m` the zone's earlier
    rewards in the experiment averaged under its own policy's weight: in blocks, the two arms'
    mean reads. While one arm is unseen it is the other's mean, and before any decision 1/2. A
    decision's level reads only the decisions before it, in its batch and in earlier ones. A drift
    alarm starts it again with the evidence. At the arms' true means the level is
    `c* = mu + (1 - rho) Delta`, where the second moment is least.
  - **The bounds.** `x` is bilinear on the box of `D` and `R`, so its extremes are corners:
    `x >= -B(c)`, `B(c) = max(c / rho, (1 - c) / (1 - rho))`, and `x <= B'(c)`,
    `B'(c) = max((1 - c) / rho, c / (1 - rho))`, both at least 1. DEPLOY bets on
    `(x - delta) / (B(c) + delta)`, harm on `(-x - delta_harm) / (B'(c) + delta_harm)`, and the
    futility stop on `(w - x) / (B'(c) - w)`, `w = delta + min_effect`. Each is at least -1 and
    has mean at most 0 under its null, so every guarantee holds as it did.
  - **Shadow and deployed increments do not change.** There the candidate's weight has no bound
    from the design, and the truncated doubly robust form would need the candidate's whole action
    law, which a `ZoneBatch` does not carry. This answers ADR 0090's objection: the other modes'
    increments are untouched.
  - **The log says which.** A centred e-value is on another scale than a plain one, so the gate's
    structured record carries `centred`.
- **A carryover check in blocks**, `GateConfig.check_burn_in` (*experimental*), `False` by default.
  Set, a gate in blocks runs an exact test of carryover past the burn-in in its experiment;
  `carryover_lags`, the per-decision check's, is still read only without blocks. The experiment's
  blocks pair up in the order they run, and each pair bets on its first block's score `s` against
  `R - c`: its second block's read mean less the mean read of that block's policy over the earlier
  pairs. Under the null each factor `1 + lam s (R - c) / A`,
  `A = max(1 / rho, 1 / (1 - rho)) max(c, 1 - c)`, has mean 1 whatever the rewards' dependence;
  mixed over the gate's twelve bets of either sign, the product is a test martingale, read after
  every pair at level 0.05 by Ville's inequality. The pairs do not overlap: a block's own read moves
  with its own coin, so it cannot also stand for the score of the pair before. At the rejection the
  gate logs one warning per zone, `gate_carryover`, with the e-value, `p <= 1 / e`, the pairs and
  their mean `s (R - c)`, and the advice to raise `burn_in`. It moves no verdict, as the
  per-decision check does not.
- **The docstring's sentence is corrected.** The gate in blocks tests, on its own data, whether a
  block's read mean moves with the block before. It does not test the order `p` or the memory `a`
  as numbers, and it does not see a carryover that leaves the read mean where it is.

## Consequences

Measured through the shipped gate, verdict counts only, no wall time;
`validation/gate_centred_increment.mac` holds the identities, the bounds, the second moments, the
planning approximation beside each median and the lab plant's contrast, and
`tests/test_gate_blocks.py`, `tests/test_gate.py`, `tests/test_formal_estimation.py` and
`tests/misspecification/test_misspec_gate.py` the tests.

- **Off, the gate is what it was.** With neither `centred` nor `check_burn_in`, every increment,
  verdict and log record but the record's new `centred` field is as before, and the gate's tests
  at the previous release pass against it as they stand.
- **Power, blocks of one period**, read means `clip(mu + Delta u + eta, 0, 1)`, `rho = 1/2`,
  `delta = 0`, read after every block, 200 paths each:

  | `mu` | sd of `eta` | `Delta` | centred: median blocks to DEPLOY | plain: deployed |
  |---|---|---|---|---|
  | 0.5 | 0.05 | 0.03 | 230.5 (planning 249) | 25 of 200 within 3000 |
  | 0.5 | 0.10 | 0.03 | 442.5 (498) | 27 |
  | 0.5 | 0.05, AR(0.9) | 0.03 | 231 | 24 |
  | 0.8 | 0.05 | 0.03 | 373.5 (368) | 8 |
  | 0.5 | 0.05 | 0.01 | 905 (1118) | 9 of 200 within 5000 |
  | 0.5 | 0.5 | 0.03 | 32 of 200 deployed within 5000 (12 190) | 7 |
  | 0.1 | 0.05 | 0.03 | 406.5 (396) | median 728.5 |

  The second moment falls from `((1 - rho)(mu + Delta)^2 + rho mu^2 + s^2) / (rho (1 - rho))` to
  `s^2 / (rho (1 - rho)) + Delta^2`, 1.07 to 0.0109 in the first row. Where the reward sits near
  0, at `mu = 0.1`, the gain is small; where its own noise dominates, at sd 0.5, centring removes
  only the level.
- **Nulls.** Over 5000 blocks and 400 paths, the gate deployed, centred against plain, on 3 and 3
  paths with no effect at `mu = 0.5` and sd 0.05; 2 and 7 with that noise AR(0.9); 6 and 3 with
  the candidate exactly `delta = 0.02` better; 7 and 7 at sd 0.5; and 3 and 2 at `mu = 0.1`. The
  most, 7 of 400, is 0.018, within the level 0.05.
- **The lab's plant**, blocks of 50 periods with 40 burned, read every 5 blocks. The read means
  have sds 0.13 to 0.16 about means near 0.8, so the second moment falls from 2.66 to 0.077.
  - A candidate that acts 0, the baseline's mean without its noise, is +0.030 better in the long
    run (+0.0305 in the reads): centred, it was deployed on 36 of 200 paths within 20 000 periods
    and on 196 within 100 000, after a median of 830 blocks (planning 909); plain, on 2 and 6.
  - ADR 0090's candidate, -0.0096 worse in the long run (-0.0103 in the reads), was deployed on
    none of 1000 paths of 20 000 periods, against 5 plain, and held on none in either mode, since
    it is not `delta_harm = 0.02` worse.
  - With `alpha_futility = 0.1`, the stop ended its experiment on 232 of the 1000 paths, against
    71 plain, and the better candidate's on none of 200 in either mode.
- **ADR 0035's worlds** (`scripts/bench_gate.py {null,mixed} --futility 0.1 --centred`, 400
  replications, the same draws). Plain, they reprint ADR 0035's numbers.

  | world | zone | decisions in the experiment, without → with the stop | stopped |
  |---|---|---|---|
  | null | boundary, chi-square 32.6 (two) | 14 232 and 14 269 → 9 900 and 9 917 | 64 % and 68 % (plain 30 % and 31 %) |
  | null | heavy-tailed | 14 304 → 966 | 99.75 % |
  | mixed | boundary, chi-square 32.6 | 14 189 → 10 027 | 63 % (plain 28 %) |
  | mixed | heavy-tailed | 14 304 → 897 | 99.75 % |
  | mixed | heavy-tailed, better | 958 → 958 (plain 1 966) | 0 |

  The FDR stayed 0.010 ± 0.005 in the null world and was 0.003 ± 0.001 in the mixed one (plain
  0.002), power stayed 1.000, and the mean check of a deploy fell from 10.1 to 7.4. The
  heavy-tailed null was held in 5 of 800 runs with the stop and none without, against 5 and 1
  plain; each such hold is wrong with probability at most `alpha_harm = 0.05`.
- **Read after every decision**, a lone zone in the experiment with its candidate exactly at
  `delta`, at `-delta_harm` or at `delta + min_effect`, 100 paths of 1000 decisions: centred, it
  was deployed on none at `delta`, held on none at `-delta_harm` and stopped on 2 at
  `delta + min_effect`, where the levels allow 10, 5 and 10; plain, on the same draws, 3, 0 and 2.
- **The mutant.** A level that reads the block it is for, the mean of the arm the block did not
  draw, makes the increment's mean `2 Delta`; at `Delta = delta = 0.02` it deployed on 94 of 100
  paths of 1000 blocks, the gate's own level on none of them.
- **The carryover check**, fed every block of 2000 (400 paths, Gaussian noise of sd 0.2, a
  candidate acting 1 that pays 0.4 now and costs 0.6 `p` periods on):
  - with `b >= p`, in blocks of 5 with `p = b = 1` or 3 and of 10 with `p = b = 5`, it warned on
    1 path in each; the per-decision score test fed per block, in a run of the same settings, on
    28, 24 and 14;
  - with `b < p` it warned on 400 of 400 paths: after a median of 72 blocks at `p = 3`, `b = 1`,
    150 at `p = 2`, and 222 in blocks of 10 at `p = 5`, `b = 4`. At a cost of 0.1, whose read
    means move by 0.05, on 400 after a median of 517; at 0.05, on 264;
  - under a geometric stock `m' = 0.9 m + 0.1 u`, in blocks of 50, on 381 with 10 burned and on
    none with 40, where the read mean moves with the block before by 0.051 and 0.0057; in blocks
    of 10 with 5 burned, on 400 after a median of 100 blocks.
- **A DEPLOY can come before the warning.** Through the gate with `check_burn_in` at `p = 3`,
  `b = 1`, where the read periods' contrast is +0.1 and the sustained one -0.2, it deployed on 204
  of 400 paths within 2000 blocks, 199 of them after a warning; centred, on all 400, 385 after a
  warning. The 5 and 15 paths that deployed with no warning before sit within the misspecification
  contract, at most `alpha` of the paths within three Monte Carlo standard deviations. Under the
  stock in blocks of 10 with 5 burned, where the read periods' contrast is +0.09 and the sustained
  one -0.2, the centred gate deployed on all 400 paths, 140 of them with no warning before, against
  372 and 13 plain: there the burn-in, not the check, has to carry the design. Without the check, as
  by default, every such deploy comes with no warning.

## Alternatives considered

- **Centring on by default.** Not through 1.0. It would change the scale of every caller's logged
  e-values; opt-in keeps the scale callers have recorded, and a default that moves would push 1.0
  a minor for a mode that is still experimental.
- **A pooled running mean for the level.** Any predictable level is valid: a level drawn at random
  each block, fed to the gate's e-process directly, still deployed on 133 of 200 paths where plain
  increments deployed on 28. The arm-weighted midpoint is the level of least second moment.
- **Centring shadow and deployed increments.** The candidate's weight has no design bound there.
- **CHC's per-decision score test fed per block.** It reads every consecutive pair, but its level
  is asymptotic: in two runs of 400 paths at `b >= p` it warned on 3.0 % to 7.0 % of them, above
  its 5 % in two settings of one run. Nor did it read more: at a cost of 0.05 it warned on 96 and
  108 of 400 paths, the exact test on 264. The design records the coins, so the exact test is
  preferred.
- **Holding DEPLOY while the check warns.** Not done: the check is a diagnostic, as without
  blocks, and a verdict that waits on it would change the gate's guarantee. Raising `burn_in` is
  the remedy the warning names.
- **The block check on by default**, as the per-decision check is. Not now: a gate in blocks would
  start to log a warning it did not log before, and new behaviour ships opt-in first.
