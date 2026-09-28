# ADR 0010 — A deployment gate that can be read at any time

**Status:** proposed, 2026-09-28.

## Context

`chc.evaluation` (ADR 0009) says what a plan would cost before it is deployed. Nothing in the
library said when the evidence from a shadow run is enough to deploy it. The obvious rule re-runs
a fixed-sample test at every read, and that rule is wrong: read after every decision, a one-sided
z-test at 0.05 deploys a candidate that is exactly as good as the margin on 28 of 100 paths
(`test_the_type_one_error_stays_below_alpha_when_read_after_every_decision`).

The lab worked out a gate that can be read at any time (AG1–AG7), and an independent verifier
checked its new pieces:

- **AG1.** An improvement e-process over logged propensities keeps its type-I error at any read,
  and only the baseline's weight needs a bound, which the logging design supplies. This is known
  (Waudby-Smith et al. 2022).
- **AG2.** A fixed bet grows at least at a two-point closed form `g2 ≈ μ²/(2s²)` against every law
  on `[−1, ∞)` with mean `μ` and variance `s²`. The verifier proved it is the minimax growth, so
  the time to a verdict can be planned from the chi-square alone.
- **AG3.** A Gaussian candidate at least `√2` times as wide as the baseline has a weight of infinite
  variance. An empirical tail estimate reads far heavier than the true index, so the index must
  come from the two policies.
- **AG4.** Three ways a gate lies:
  - a behaviour law refitted to the logs took its type-I error from 0.013 to 1.000;
  - clipping in the wrong orientation gave false rejections of 1.000;
  - per-decision increments under carryover deployed a candidate that was worse in the long run,
    with probability 1.000.
- **AG6.** e-Shiryaev–Roberts alarms no later than e-CUSUM on every path at the same threshold.
- **AG7.** e-BH over e-values frozen at selection keeps the false-discovery rate at every stopping
  time (Wang, Dandapanthula and Ramdas 2025). It needs each zone's null to hold given every zone's
  past, which spillover breaks. A first EXPERIMENT rule, on the budget alone, was refuted: it sent
  99.8 % of slow nulls to EXPERIMENT.

## Decision

- **`chc.gate.DeploymentGate(plans, config).update(batches)`** returns a verdict per zone after
  every batch: `"deploy"`, `"shadow"`, `"experiment"`, `"hold"` or `"rollback"`. `mode(zone)`
  says how to log the zone's next batch. The rules, in order:
  1. a drift alarm holds the zone, and its evidence opens a new epoch; a deployed zone rolls back;
  2. e-BH over every zone's improvement e-value selects DEPLOY;
  3. a harm e-value at `1 / alpha_harm` holds the zone, which retires;
  4. a shadow zone that cannot reach a verdict at `min_effect` in the horizon left, where the
     mixture `(1 − ρ) baseline + ρ candidate` would gather evidence faster, moves to EXPERIMENT;
  5. otherwise the zone stays where it is.
- **The guarantee is in the docstring with its four conditions:**
  - propensities logged at decision time;
  - no spillover between zones;
  - no carryover;
  - rewards in `[0, 1]`.
- **The gate takes propensities, not weights.** The lab's reference took the two weights. The
  gate takes three propensities of the action taken (the candidate's, the baseline's, and the one
  logged when it was drawn), computes the weights itself, and refuses a batch whose logged
  propensities are not those of the policy the zone's mode asked for, to a relative `1e-6`.
  - This makes AG4's clipping unrepresentable, since no clipped weight can be passed in.
  - It catches the density of the component that drew an action logged in place of the
    mixture's.
  - It catches a baseline that is not what actually logged.
  - A fitted propensity passed as the logged one still gets through. The gate can require the
    field; only the record it comes from can make it honest (ADR 0015).
- **`ZonePlan(chi2, tv)` is the caller's**, computed from the two policies. It steers only the
  EXPERIMENT rule: no guarantee depends on it.
- **Every batch is checked before any is added**, so a refused update changes no zone.
- **The evidence lives in memory.** What each decision records, its logged propensity and its
  dither, is stored as a versioned `DecisionLog` (ADR 0015); persisting the evidence itself is not
  built.
- **Not refused, because the gate cannot see them:**
  - carryover, stated as a condition. On a plant with memory, `chc.evaluation` estimates the
    plan's value, and switchback increments with a washout are the lab's next step;
  - spillover, stated as a condition.

## Consequences

- **The lab's closed loop, verdict for verdict.** Both of the lab's eight-zone worlds were run with
  400 replications of 150 checks. The gate's verdicts match the lab's reference implementation at
  all 120 000 checks, so the lab's error rates are this code's (`scripts/bench_gate.py`):

  | world | FDR at the horizon | FDR at the first null deploy | any null deployed | power |
  |---|---|---|---|---|
  | eight null zones | 0.0075 ± 0.0043 | 0.0075 ± 0.0043 | 3 of 400 | — |
  | four null, four better | 0.0055 ± 0.0016 | 0.0058 ± 0.0018 | 11 of 400 | 1.000 |

  - `alpha` is 0.10, so both rates are far below it.
  - The missed-deploy rate is 0: every better zone is deployed, on average at check 10.0.
  - The EXPERIMENT rule sends the heavy-tailed and high-chi-square zones to the mixture, in 99.8 to
    100 % of replications, and no other zone.
  - False holds: 2 of 2800 zone-runs in the null world and 2 of 3200 in the mixed one.
  - None of the 1611 deployments in the mixed world was rolled back.
- A test reproduces twenty replications of each world count for count. The type-I test reads a
  zone after every decision: the gate deploys it on none of 100 paths, the z-test on 28. Nine
  mutations of the rules were each caught by a test.
- A caller has to log the propensity of each action as it was drawn. That is the price of a
  guarantee that holds at any read, and a refitted one cannot pay it.

## Alternatives considered

- **`confseq` or `expectation`.** `confseq`'s wheels stop at CPython 3.10, and `expectation` is
  GPL-3.0. CHC owns this code, one module.
- **The lab's weight interface.** Rejected for the reasons above. A bound on the weights catches a
  weight that is too large, not one that was clipped.
- **Running e-BH**, a union of per-read selections. In the lab it gave the same rates, but only the
  frozen form has the theorem at every stopping time.
- **e-CUSUM for drift.** Rejected: e-Shiryaev–Roberts alarms no later on every path.
- **An EXPERIMENT rule on the budget alone.** Refuted in the lab.
