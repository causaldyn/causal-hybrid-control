# ADR 0035 — A futility stop for the gate's experiment

**Status:** proposed, 2026-09-30.

## Context

A zone of `DeploymentGate` (ADR 0010) enters EXPERIMENT when shadow evidence would come too slowly,
and leaves it only by DEPLOY, by a HOLD for harm, or by a drift alarm. A candidate whose effect
sits near `delta`, neither selectable nor harmful, keeps a `rho` share of the zone's decisions
until the horizon ends. In the lab's closed loop, the heavy-tailed null (contrast −0.012) and the
boundary zones with chi-square 32.6 (contrast exactly `delta`) spend 14 218 to 14 304 of their
14 400 decisions in the experiment.

`GateConfig.min_effect` is the smallest improvement beyond `delta` worth deploying. The experiment
exists to reach a DEPLOY for such a candidate within the horizon. Once the evidence says the
candidate is not worth `min_effect`, the experiment has nothing left to find.

Stopping for futility is standard in group-sequential trials. The gate needs a form that can be
read at any time, as its other rules can.

## Decision

- **`GateConfig.alpha_futility`** (*experimental*), unset by default. Set, each zone runs a fourth
  e-process, against `V_new − V_base ≥ delta + min_effect`. It is the harm e-process at the margin
  `−(delta + min_effect)`, with increments

      (diff (1 − r) + delta + min_effect) / (bound − delta − min_effect).

  `E[diff (1 − r)] = V_base − V_new`, the identity the harm e-process rests on, so again only the
  baseline's weight needs a bound, which the logging design supplies. The scaling needs
  `delta + min_effect < 1`, and the config refuses a larger sum, which no reward in `[0, 1]` can
  beat.
- **At `1 / alpha_futility` the zone is futile for the rest of its epoch.** An experiment returns
  to shadow, with the verdict `"shadow"`, and the EXPERIMENT rule does not fire again. A drift
  alarm opens a new epoch for this e-process with the others, and clears the flag.
- **The rule comes after DEPLOY and after harm.** A candidate better than `delta` but not by
  `delta + min_effect` is deployed, since DEPLOY claims only `V_new − V_base > delta`.
- **The flag stays up.** A crossing rejects, and the e-value falling back afterwards does not undo
  it. An experiment re-entered on the way down would stop again.
- **The stop is logged** on the gate's structured event, with `futile` and `log_futility`, when
  the flag rises.
- **The guarantee.** Each stop is wrong, stopping a candidate that beats the baseline by more than
  `delta + min_effect`, with probability at most `alpha_futility`, by Ville's inequality.
  - DEPLOY's false-discovery rate is untouched by construction. The switch back to shadow is
    decided on the past, like the switch into EXPERIMENT, so the improvement e-process stays a
    test supermartingale.
  - The stop freezes and discards no improvement evidence. Shadow evidence keeps accruing, and a
    later DEPLOY stays possible.

## Consequences

Measured by `scripts/bench_gate.py {null,mixed} --futility 0.1`: 400 replications of each of the
lab's worlds, each run without and with the stop on the same draws in every zone whose modes
agree. Verdicts and decision counts only, no wall time.

| world | zone | contrast | decisions in the experiment, without → with | experiments stopped |
|---|---|---|---|---|
| null | boundary, chi-square 32.6 (two) | 0.020 | 14 218 and 14 304 → 12 360 and 12 440 | 30 % and 31 % |
| null | heavy-tailed | −0.012 | 14 276 → 1 937 | 99.75 % |
| mixed | boundary, chi-square 32.6 | 0.020 | 14 250 → 12 625 | 28 % |
| mixed | heavy-tailed | −0.012 | 14 304 → 1 816 | 99.75 % |
| mixed | heavy-tailed, better | 0.063 | 1 966 → 1 966 | 0 |

- **Nothing else moved.** The FDR is the same in both arms, 0.010 ± 0.005 in the null world and
  0.002 ± 0.001 in the mixed one, and so are the power, 1.000, and the mean check of a deploy,
  10.1. In the tests' twenty paired replications of each world, every zone the stop does not
  reach reads the same verdicts in both runs.
- **A candidate at exactly `delta` stops slowly.** It is `min_effect` from the futility null, the
  same distance the EXPERIMENT rule budgets for a DEPLOY, and the stop saves 11–13 % of its
  experiment.
- **One cost, within its level.** The heavy-tailed null back in shadow meets weights of infinite
  variance again, and its harm e-process spends more of `alpha_harm`: it was held for harm in 5 of
  800 runs with the stop, against 1 without. A HOLD claims `V_base − V_new > delta_harm`, and this
  candidate is 0.012 worse against a `delta_harm` of 0.02. Each such hold is wrong with
  probability at most `alpha_harm = 0.05`.
- **The least favourable worthwhile candidate**, exactly `delta + min_effect` better, with
  chi-square 32.6 and in the experiment from the first decision, read after each of 2000
  decisions: the stop ends 1 of 100 paths, where the level allows 10.
- **Unset, the gate is what it was.** The lab replay reproduces the lab's counts, and the bench's
  default path reprints its previous output key for key.
- Ten mutations of the stop were each caught by a test.

## Alternatives considered

- **A one-step value-of-information look-ahead**: stop when one more batch is worth less than it
  costs. Rejected. It needs a prior on the effect and a cost per batch that the gate does not
  observe, and it states no error. In rewards, an experiment on a candidate between `delta` and
  `delta + min_effect` costs nothing: it gains.
- **Retiring the zone at the stop.** Rejected: it would freeze the improvement e-value of a
  candidate that may still beat `delta`, which free shadow evidence can still deploy.
- **A new verdict.** Rejected: `Verdict` and `GateMode` are literals that callers match on, and
  `"shadow"` already says how to log the next batch.
- **On by default.** Not yet. It would change the experiment counts of every current caller, and
  the numbers above come from the lab's own market. Once the stop is measured on a plant CHC did
  not write, a default can be argued in a changelog entry, as the evolving tier allows.
- **Curtailment on the horizon**: stop when even `min_effect` from now on could not reach DEPLOY
  in the decisions left. Not built. It fires only near the end of the horizon, once most of the
  budget is spent, and it could sit beside this rule.
